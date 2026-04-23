import os
import zipfile
import csv
import io
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QListWidget,
    QWidget,
    QLabel,
    QPushButton,
    QFormLayout,
    QFrame,
    QProgressBar,
    QSizePolicy,
    QAbstractItemView,
    QInputDialog,
    QMessageBox,
    QApplication,
    QListWidgetItem,
    QLineEdit,
    QStackedWidget,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QDateTimeEdit,
    QCheckBox,
)
from PyQt5.QtCore import (
    Qt,
    QThread,
    pyqtSignal,
    QPropertyAnimation,
    QEasingCurve,
    QSize,
    QPoint,
    QDateTime,
)
from PyQt5.QtGui import QFont, QIcon
from datetime import datetime
from typing import List
import send2trash
import pyqtgraph as pg
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants
from QATCH.ui.drawPlateConfig import Architecture
from QATCH.common.fileStorage import secure_open

TAG = "[RunRecovery]"


class UnnamedRun:
    """Data class to hold unnamed run information"""

    def __init__(
        self,
        filepath: str,
        display_name: str,
        timestamp: str,
        duration_seconds: float,
        num_points: int,
        ruling: str,
        file_size_mb: float,
        virtual_csv_path: str = None,
    ):
        self.filepath = filepath
        self.display_name = display_name
        self.timestamp = timestamp
        self.duration_seconds = duration_seconds
        self.num_points = num_points
        self.ruling = ruling
        self.file_size_mb = file_size_mb
        self.virtual_csv_path = virtual_csv_path


class RecoveryWorker(QThread):
    """Worker thread to handle recovery process without blocking UI"""

    progress_updated = pyqtSignal(int)  # Progress percentage
    status_updated = pyqtSignal(str)  # Status message
    recovery_complete = pyqtSignal(bool, str)  # Success flag, message

    def __init__(self, run_data, run_name):
        super().__init__()
        self.run_data = run_data
        self.run_name = run_name
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        """Perform the actual recovery work"""
        try:
            self.status_updated.emit("Parsing raw data...")
            self.progress_updated.emit(10)
            if self._is_cancelled:
                return
            self.msleep(500)

            self.status_updated.emit("Generating viscosity profile...")
            self.progress_updated.emit(35)
            if self._is_cancelled:
                return
            self.msleep(800)

            self.status_updated.emit("Computing statistics...")
            self.progress_updated.emit(60)
            if self._is_cancelled:
                return
            self.msleep(600)

            self.status_updated.emit("Generating missing metadata...")
            self.progress_updated.emit(80)
            if self._is_cancelled:
                return
            self.msleep(400)

            self.status_updated.emit("Writing recovered file...")
            self.progress_updated.emit(95)
            if self._is_cancelled:
                return
            self.msleep(300)

            self.progress_updated.emit(100)
            self.status_updated.emit("Recovery complete!")
            self.recovery_complete.emit(True, f"Successfully recovered run as '{self.run_name}'")

        except Exception as e:
            self.recovery_complete.emit(False, f"Recovery failed: {str(e)}")


class RecoveryProgressDialog(QDialog):
    """Modal dialog showing recovery progress"""

    def __init__(self, parent, run_data, run_name):
        super().__init__(parent)
        self.run_data = run_data
        self.run_name = run_name
        self.worker = None
        self.setup_ui()
        self.start_recovery()

    def setup_ui(self):
        pass

    def start_recovery(self):
        """Start the recovery worker thread"""
        self.worker = RecoveryWorker(self.run_data, self.run_name)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.status_updated.connect(self.update_status)
        self.worker.recovery_complete.connect(self.on_recovery_complete)
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.status_text.append(message)
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum()
        )

    def on_recovery_complete(self, success, message):
        if success:
            self.cancel_button.setText("Close")
            self.cancel_button.clicked.disconnect()
            self.cancel_button.clicked.connect(self.accept)
            QMessageBox.information(self, "Recovery Complete", message)
        else:
            self.cancel_button.setText("Close")
            self.cancel_button.clicked.disconnect()
            self.cancel_button.clicked.connect(self.reject)
            QMessageBox.critical(self, "Recovery Failed", message)

    def on_cancel_clicked(self):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Cancel Recovery",
                "Are you sure you want to cancel the recovery process?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.worker.cancel()
                self.worker.wait()
                self.reject()
        else:
            self.reject()


class ToggleListWidget(QListWidget):
    """QListWidget that deselects an item when it is clicked a second time."""

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.NoModifier:
            item = self.itemAt(event.pos())
            if item is not None and item.isSelected() and len(self.selectedItems()) == 1:
                self.clearSelection()
                self.setCurrentItem(None)
                event.accept()
                return
        super().mousePressEvent(event)


class FilterPopup(QFrame):

    filtersChanged = pyqtSignal(dict)

    def __init__(self, parent=None, current_filters=None):
        super().__init__(parent, Qt.Popup)
        self.current_filters = current_filters or {}
        self.setObjectName("FilterPopup")
        self.setStyleSheet(
            """
            QFrame#FilterPopup {
                background-color: #ffffff;
                border: 1px solid rgba(0, 0, 0, 25);
                border-radius: 8px;
            }
            QLabel {
                color: #555555;
                font-size: 9pt;
                background: transparent;
                border: none;
            }
            QLabel#sectionLabel {
                color: #333333;
                font-size: 9pt;
                font-weight: 600;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: rgba(255, 255, 255, 255);
                border: 1px solid rgba(0, 0, 0, 25);
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 9pt;
                color: #333333;
                min-height: 20px;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid rgba(0, 114, 189, 120);
            }
            QPushButton {
                border-radius: 5px;
                padding: 5px 14px;
                font-size: 9pt;
            }
            QPushButton#applyBtn {
                background-color: rgba(0, 114, 189, 25);
                color: #005b9f;
                border: 1px solid rgba(0, 114, 189, 60);
            }
            QPushButton#applyBtn:hover {
                background-color: rgba(0, 114, 189, 45);
            }
            QPushButton#resetBtn {
                background-color: transparent;
                color: #666666;
                border: 1px solid rgba(0, 0, 0, 25);
            }
            QPushButton#resetBtn:hover {
                background-color: rgba(0, 0, 0, 6);
            }
            """
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(10)

        title = QLabel("Filter Options")
        title.setStyleSheet("color: #222222; font-size: 10pt; font-weight: 600;")
        main_layout.addWidget(title)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        row = 0

        # ---- Status ----
        grid.addWidget(QLabel("Status:"), row, 0, Qt.AlignRight)
        self.status_combo = QComboBox()
        self.status_combo.addItems(["Any", "Good", "Bad"])
        grid.addWidget(self.status_combo, row, 1, 1, 3)
        row += 1

        # ---- Duration ----
        grid.addWidget(QLabel("Duration (s):"), row, 0, Qt.AlignRight)
        self.duration_min = QDoubleSpinBox()
        self.duration_min.setRange(0.0, 1_000_000.0)
        self.duration_min.setDecimals(2)
        self.duration_min.setValue(0.0)
        self.duration_max = QDoubleSpinBox()
        self.duration_max.setRange(0.0, 1_000_000.0)
        self.duration_max.setDecimals(2)
        self.duration_max.setValue(0.0)
        self.duration_max.setToolTip("0 = no upper limit")
        grid.addWidget(QLabel("Min"), row, 1, Qt.AlignRight)
        grid.addWidget(self.duration_min, row, 2)
        grid.addWidget(QLabel("Max"), row, 3, Qt.AlignRight)
        grid.addWidget(self.duration_max, row, 4)
        row += 1

        # ---- Data points ----
        grid.addWidget(QLabel("Data Points:"), row, 0, Qt.AlignRight)
        self.points_min = QSpinBox()
        self.points_min.setRange(0, 100_000_000)
        self.points_min.setValue(0)
        self.points_max = QSpinBox()
        self.points_max.setRange(0, 100_000_000)
        self.points_max.setValue(0)
        self.points_max.setToolTip("0 = no upper limit")
        grid.addWidget(QLabel("Min"), row, 1, Qt.AlignRight)
        grid.addWidget(self.points_min, row, 2)
        grid.addWidget(QLabel("Max"), row, 3, Qt.AlignRight)
        grid.addWidget(self.points_max, row, 4)
        row += 1

        # ---- File size ----
        grid.addWidget(QLabel("File Size (MB):"), row, 0, Qt.AlignRight)
        self.size_min = QDoubleSpinBox()
        self.size_min.setRange(0.0, 1_000_000.0)
        self.size_min.setDecimals(2)
        self.size_min.setValue(0.0)
        self.size_max = QDoubleSpinBox()
        self.size_max.setRange(0.0, 1_000_000.0)
        self.size_max.setDecimals(2)
        self.size_max.setValue(0.0)
        self.size_max.setToolTip("0 = no upper limit")
        grid.addWidget(QLabel("Min"), row, 1, Qt.AlignRight)
        grid.addWidget(self.size_min, row, 2)
        grid.addWidget(QLabel("Max"), row, 3, Qt.AlignRight)
        grid.addWidget(self.size_max, row, 4)
        row += 1

        main_layout.addLayout(grid)

        # ---- Buttons ----
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setObjectName("resetBtn")
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.clicked.connect(self._on_reset)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setObjectName("applyBtn")
        self.apply_btn.setCursor(Qt.PointingHandCursor)
        self.apply_btn.clicked.connect(self._on_apply)

        btn_row.addWidget(self.reset_btn)
        btn_row.addWidget(self.apply_btn)
        main_layout.addLayout(btn_row)

        self._populate_from(current_filters or {})

    def _wire_enable_state(self):
        pass

    def _populate_from(self, f):
        status = f.get("status")
        if status in ("Good", "Bad"):
            self.status_combo.setCurrentText(status)
        else:
            self.status_combo.setCurrentIndex(0)  # "Any"

        self.duration_min.setValue(
            float(f["duration_min"]) if f.get("duration_min") is not None else 0.0
        )
        d_max = f.get("duration_max")
        self.duration_max.setValue(
            0.0 if (d_max is None or d_max == float("inf")) else float(d_max)
        )

        self.points_min.setValue(int(f["points_min"]) if f.get("points_min") is not None else 0)
        p_max = f.get("points_max")
        self.points_max.setValue(0 if (p_max is None or p_max == float("inf")) else int(p_max))

        self.size_min.setValue(float(f["size_min"]) if f.get("size_min") is not None else 0.0)
        s_max = f.get("size_max")
        self.size_max.setValue(0.0 if (s_max is None or s_max == float("inf")) else float(s_max))

    def _collect(self):
        filters = {}

        if self.status_combo.currentText() != "Any":
            filters["status"] = self.status_combo.currentText()

        d_min = self.duration_min.value()
        d_max = self.duration_max.value()
        if d_min > 0.0 or d_max > 0.0:
            filters["duration_min"] = d_min
            filters["duration_max"] = d_max if d_max > 0.0 else float("inf")

        p_min = self.points_min.value()
        p_max = self.points_max.value()
        if p_min > 0 or p_max > 0:
            filters["points_min"] = p_min
            filters["points_max"] = p_max if p_max > 0 else float("inf")

        s_min = self.size_min.value()
        s_max = self.size_max.value()
        if s_min > 0.0 or s_max > 0.0:
            filters["size_min"] = s_min
            filters["size_max"] = s_max if s_max > 0.0 else float("inf")

        return filters

    def _on_apply(self):
        self.filtersChanged.emit(self._collect())
        self.close()

    def _on_reset(self):
        self.status_combo.setCurrentIndex(0)
        self.duration_min.setValue(0.0)
        self.duration_max.setValue(0.0)
        self.points_min.setValue(0)
        self.points_max.setValue(0)
        self.size_min.setValue(0.0)
        self.size_max.setValue(0.0)
        self.filtersChanged.emit({})
        self.close()


class RecoverUnnamedRunsDialog(QDialog):

    _UNNAMED_DIR = os.path.join(Architecture.get_path(), Constants.log_export_path, "_unnamed")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.unnamed_runs: List[UnnamedRun] = []
        self.selected_run = None
        self.active_filters = {}
        self.is_preview_visible = True
        self.setup_ui()
        self.load_unnamed_runs()

    def setup_ui(self):
        self.setStyleSheet(
            """
            QDialog { background-color: transparent; }
            """
        )

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # TOP
        top_master_layout = QVBoxLayout()
        top_master_layout.setSpacing(15)

        # Search, filter, rescan row
        search_layout = QHBoxLayout()
        search_layout.setSpacing(8)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search...")
        self.search_icon_action = self.search_bar.addAction(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "search.svg")),
            QLineEdit.LeadingPosition,
        )
        self.search_bar.setStyleSheet(
            """
            QLineEdit {
                background-color: rgba(255, 255, 255, 180);
                border: 1px solid rgba(0, 0, 0, 15);
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 10pt;
                color: #333333;
            }
            QLineEdit:focus {
                border: 1px solid rgba(0, 114, 189, 100);
                background-color: rgba(255, 255, 255, 255);
            }
            """
        )
        self.search_bar.textChanged.connect(self.refilter_list)

        _icon_btn_ss = """
            QPushButton {
                background-color: rgba(255, 255, 255, 180);
                border: 1px solid rgba(0, 0, 0, 15);
                border-radius: 6px;
            }
            QPushButton:hover   { background-color: rgba(0, 0, 0,  8); }
            QPushButton:pressed { background-color: rgba(0, 0, 0, 15); }
            QPushButton:checked {
                background-color: rgba(0, 114, 189, 35);
                border: 1px solid rgba(0, 114, 189, 80);
            }
        """

        self.filter_btn = QPushButton()
        self.filter_btn.setFixedSize(28, 28)
        self.filter_btn.setCursor(Qt.PointingHandCursor)
        self.filter_btn.setToolTip("Filter Options")
        self.filter_btn.setIcon(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "filter.svg"))
        )
        self.filter_btn.setIconSize(QSize(16, 16))
        self.filter_btn.setStyleSheet(_icon_btn_ss)
        self.filter_btn.setCheckable(True)
        self.filter_btn.clicked.connect(self.show_filter_menu)
        search_layout.addWidget(self.search_bar, stretch=1)
        search_layout.addWidget(self.filter_btn)
        top_master_layout.addLayout(search_layout)

        # 2. Details, Buttons (right)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)

        # LEFT
        self.runs_list = ToggleListWidget()
        self.runs_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.runs_list.setStyleSheet(
            """
            QListWidget {
                border: 1px solid rgba(0, 0, 0, 15);
                border-radius: 6px;
                background-color: rgba(255, 255, 255, 180);
                padding: 4px;
                outline: none;
            }
            QListWidget::item {
                padding: 6px 8px;
                border-radius: 4px;
                margin-bottom: 2px;
                color: #444444;
            }
            QListWidget::item:hover    { background-color: rgba(0, 0, 0,  8); }
            QListWidget::item:selected {
                background-color: #f0f4f8;
                color: #111111;
            }
            """
        )
        self.runs_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.runs_list.itemDoubleClicked.connect(self.on_item_double_clicked)

        self.empty_list_placeholder = QLabel("No recoverable runs")
        self.empty_list_placeholder.setAlignment(Qt.AlignCenter)
        self.empty_list_placeholder.setStyleSheet(
            """
            QLabel {
                border: 1px dashed rgba(0, 0, 0, 25);
                border-radius: 6px;
                background-color: rgba(255, 255, 255, 120);
                color: #999999;
                font-size: 10pt;
                padding: 20px;
            }
            """
        )

        self.list_stack = QStackedWidget()
        self.list_stack.addWidget(self.runs_list)  # index 0
        self.list_stack.addWidget(self.empty_list_placeholder)  # index 1
        self.list_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.list_stack.setMinimumWidth(0)

        content_layout.addWidget(self.list_stack, stretch=1)

        # RIGHT
        right_container = QWidget()
        right_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_container.setMinimumWidth(0)

        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        self.details_frame = QFrame()
        self.details_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.details_frame.setMinimumWidth(0)
        self.details_frame.setStyleSheet(
            """
            QFrame {
                background-color: rgba(255, 255, 255, 180);
                border: 1px solid rgba(0, 0, 0, 15);
                border-radius: 6px;
            }
            QLabel {
                border: none;
                background: transparent;
                color: #444444;
                font-size: 9pt;
            }
            """
        )

        details_form = QFormLayout()
        details_form.setContentsMargins(12, 10, 12, 10)
        details_form.setSpacing(6)
        details_form.setLabelAlignment(Qt.AlignLeft)
        details_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        def make_key_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #888888; font-size: 9pt;")
            return lbl

        def make_value_label():
            lbl = QLabel("—")
            lbl.setStyleSheet("color: #333333; font-size: 9pt;")
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            return lbl

        self.detail_datetime = make_value_label()
        self.detail_duration = make_value_label()
        self.detail_points = make_value_label()
        self.detail_ruling = make_value_label()
        self.detail_filesize = make_value_label()

        details_form.addRow(make_key_label("Date / Time:"), self.detail_datetime)
        details_form.addRow(make_key_label("Duration:"), self.detail_duration)
        details_form.addRow(make_key_label("Data Points:"), self.detail_points)
        details_form.addRow(make_key_label("Ruling:"), self.detail_ruling)
        details_form.addRow(make_key_label("File Size:"), self.detail_filesize)

        self.details_frame.setLayout(details_form)
        right_layout.addWidget(self.details_frame)

        action_button_layout = QHBoxLayout()
        action_button_layout.setSpacing(10)

        btn_common = """
            QPushButton {
                border-radius: 6px;
                padding: 4px;
                min-height: 30px;
                min-width: 30px;
            }
            QPushButton:disabled {
                background-color: transparent;
                border: 1px solid rgba(0, 0, 0, 10);
            }
        """

        # Recover Button
        self.recover_button = QPushButton()
        self.recover_button.setCursor(Qt.PointingHandCursor)
        self.recover_button.setEnabled(False)
        self.recover_button.setToolTip("Recover Run")
        self.recover_button.setIcon(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "restore.svg"))
        )
        self.recover_button.setIconSize(QSize(18, 18))
        self.recover_button.setStyleSheet(
            btn_common
            + """
            QPushButton {
                background-color: rgba(0, 114, 189, 20);
                border: 1px solid rgba(0, 114, 189, 55);
            }
            QPushButton:hover {
                background-color: rgba(0, 114, 189, 40);
                border: 1px solid rgba(0, 114, 189, 80);
            }
            """
        )
        self.recover_button.clicked.connect(self.on_recover_clicked)

        # Rescan
        self.rescan_btn = QPushButton()
        self.rescan_btn.setCursor(Qt.PointingHandCursor)
        self.rescan_btn.setToolTip("Rescan for unnamed runs")
        self.rescan_btn.setIcon(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "refresh-cw.svg"))
        )
        self.rescan_btn.setIconSize(QSize(18, 18))
        self.rescan_btn.setStyleSheet(
            btn_common
            + """
            QPushButton {
                background-color: rgba(255, 255, 255, 180);
                border: 1px solid rgba(0, 0, 0, 15);
            }
            QPushButton:hover { background-color: rgba(0, 0, 0, 8); }
            """
        )
        self.rescan_btn.clicked.connect(self.load_unnamed_runs)

        # Delete
        self.delete_button = QPushButton()
        self.delete_button.setCursor(Qt.PointingHandCursor)
        self.delete_button.setEnabled(False)
        self.delete_button.setToolTip("Delete Selected")
        self.delete_button.setIcon(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "delete.svg"))
        )
        self.delete_button.setIconSize(QSize(18, 18))
        self.delete_button.setStyleSheet(
            btn_common
            + """
            QPushButton {
                background-color: rgba(220, 53, 69, 18);
                border: 1px solid rgba(220, 53, 69, 55);
            }
            QPushButton:hover {
                background-color: rgba(220, 53, 69, 38);
                border: 1px solid rgba(220, 53, 69, 80);
            }
            """
        )
        self.delete_button.clicked.connect(self.on_delete_clicked)

        action_button_layout.addWidget(self.recover_button)
        action_button_layout.addWidget(self.rescan_btn)
        action_button_layout.addWidget(self.delete_button)
        action_button_layout.addStretch(1)
        right_layout.addLayout(action_button_layout)
        right_layout.addStretch(1)
        content_layout.addWidget(right_container, stretch=1)
        top_master_layout.addLayout(content_layout)

        # BOTTOM
        self.plot_container = QFrame()
        self.plot_container.setStyleSheet("QFrame { background: transparent; border: none; }")
        self.plot_container.setMinimumHeight(28)
        self.plot_container.setContentsMargins(0, 0, 0, 0)

        plot_container_layout = QVBoxLayout(self.plot_container)
        plot_container_layout.setContentsMargins(0, 0, 0, 0)
        plot_container_layout.setSpacing(0)

        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumHeight(40)
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_widget.setBackground("#fafafa")
        self.plot_widget.setStyleSheet("border: none;")
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideButtons()
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)

        left_axis = self.plot_widget.getAxis("left")
        left_axis.setPen(pg.mkPen(color="#cccccc", width=1))
        left_axis.setLabel(text="")
        left_axis.setStyle(showValues=False)

        bottom_axis = self.plot_widget.getAxis("bottom")
        bottom_axis.setPen(pg.mkPen(color="#cccccc", width=1))
        bottom_axis.setLabel(text="")
        bottom_axis.setStyle(showValues=False)

        self.plot_widget.hideAxis("right")

        legend = self.plot_widget.addLegend(offset=(8, 8))
        legend.setBrush(pg.mkBrush(255, 255, 255, 200))
        legend.setPen(pg.mkPen(color=(200, 200, 200, 180), width=1))
        try:
            legend.setLabelTextSize("8pt")
        except Exception:
            pass
        try:
            legend.layout.setHorizontalSpacing(4)
            legend.layout.setVerticalSpacing(0)
            legend.layout.setContentsMargins(4, 2, 4, 2)
        except Exception:
            pass
        legend.mouseDragEvent = lambda ev: ev.ignore()
        legend.hoverEvent = lambda ev: None
        self._legend = legend

        pen_freq = pg.mkPen(color=(0, 114, 189, 210), width=2.0)
        pen_diss = pg.mkPen(color=(237, 177, 32, 210), width=2.0)

        self.curve_freq = self.plot_widget.plot(pen=pen_freq)
        legend.addItem(self.curve_freq, "Frequency")

        self.view_box_diss = pg.ViewBox()
        self.plot_widget.scene().addItem(self.view_box_diss)
        self.view_box_diss.setXLink(self.plot_widget)
        self.view_box_diss.setMouseEnabled(x=False, y=False)
        self.view_box_diss.setMenuEnabled(False)

        self.curve_diss = pg.PlotCurveItem(pen=pen_diss)
        self.view_box_diss.addItem(self.curve_diss)
        legend.addItem(self.curve_diss, "Dissipation")

        def updateViews():
            self.view_box_diss.setGeometry(self.plot_widget.plotItem.vb.sceneBoundingRect())
            self.view_box_diss.linkedViewChanged(
                self.plot_widget.plotItem.vb, self.view_box_diss.XAxis
            )

        updateViews()
        self.plot_widget.plotItem.vb.sigResized.connect(updateViews)

        self.empty_plot_placeholder = QLabel("No data to display")
        self.empty_plot_placeholder.setAlignment(Qt.AlignCenter)
        self.empty_plot_placeholder.setStyleSheet(
            """
            QLabel {
                background-color: #fafafa;
                color: #aaaaaa;
                font-size: 10pt;
            }
            """
        )

        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(self.plot_widget)  # index 0
        self.plot_stack.addWidget(self.empty_plot_placeholder)  # index 1
        self.plot_stack.setCurrentIndex(1)

        plot_container_layout.addWidget(self.plot_stack)

        self.icon_show = QIcon(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "down-chevron.svg")
        )
        self.icon_hide = QIcon(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "up-chevron.svg")
        )

        self.toggle_plot_btn = QPushButton(self.plot_container)
        self.toggle_plot_btn.setIcon(self.icon_hide)
        self.toggle_plot_btn.setIconSize(QSize(14, 14))
        self.toggle_plot_btn.setFixedSize(40, 18)
        self.toggle_plot_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_plot_btn.setFlat(True)
        self.toggle_plot_btn.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 0px;
            }
            QPushButton:hover   { background-color: rgba(0, 0, 0, 10); border-radius: 4px; }
            QPushButton:pressed { background-color: rgba(0, 0, 0, 20); }
            """
        )
        self.toggle_plot_btn.clicked.connect(self.toggle_preview)
        self.toggle_plot_btn.raise_()

        self.plot_container.installEventFilter(self)

        main_layout.addLayout(top_master_layout, stretch=1)
        main_layout.addWidget(self.plot_container, stretch=0)

        self.setLayout(main_layout)

    def eventFilter(self, obj, event):
        if obj is self.plot_container and event.type() == event.Resize:
            self._position_toggle_btn()
        return super().eventFilter(obj, event)

    def _position_toggle_btn(self):
        if not hasattr(self, "toggle_plot_btn"):
            return
        cw = self.plot_container.width()
        bw = self.toggle_plot_btn.width()
        self.toggle_plot_btn.move(max(0, (cw - bw) // 2), 2)
        self.toggle_plot_btn.raise_()

    def show_filter_menu(self):
        popup = FilterPopup(self, current_filters=self.active_filters)
        popup.filtersChanged.connect(self._on_filters_changed)

        popup.adjustSize()
        btn_right_global = self.filter_btn.mapToGlobal(
            QPoint(self.filter_btn.width(), self.filter_btn.height() + 4)
        )
        pos = QPoint(btn_right_global.x() - popup.width(), btn_right_global.y())
        popup.move(pos)
        popup.show()

        popup.destroyed.connect(lambda: self.filter_btn.setChecked(bool(self.active_filters)))

    def _on_filters_changed(self, filters):
        self.active_filters = filters
        self.filter_btn.setChecked(bool(filters))
        self.refilter_list()

    def refilter_list(self, *_):
        """Apply search text + active filters to the list."""
        query = self.search_bar.text().lower().strip()
        visible_count = 0

        for index in range(self.runs_list.count()):
            item = self.runs_list.item(index)
            run: UnnamedRun = item.data(Qt.UserRole)

            matches = True

            # Search
            if query:
                matches = query in run.display_name.lower() or query in run.ruling.lower()

            # Status
            if matches and "status" in self.active_filters:
                matches = run.ruling == self.active_filters["status"]

            # Date / time range
            if matches and "date_from" in self.active_filters and "date_to" in self.active_filters:
                run_dt = self._parse_timestamp(run.timestamp)
                if run_dt is None:
                    matches = False
                else:
                    matches = (
                        self.active_filters["date_from"] <= run_dt <= self.active_filters["date_to"]
                    )

            # Duration range
            if matches and "duration_min" in self.active_filters:
                matches = (
                    self.active_filters["duration_min"]
                    <= run.duration_seconds
                    <= self.active_filters["duration_max"]
                )

            # Points range
            if matches and "points_min" in self.active_filters:
                matches = (
                    self.active_filters["points_min"]
                    <= run.num_points
                    <= self.active_filters["points_max"]
                )

            # Size range
            if matches and "size_min" in self.active_filters:
                matches = (
                    self.active_filters["size_min"]
                    <= run.file_size_mb
                    <= self.active_filters["size_max"]
                )

            item.setHidden(not matches)
            if matches:
                visible_count += 1
            if not matches and item.isSelected():
                item.setSelected(False)
        if self.runs_list.count() == 0 or visible_count == 0:
            if self.runs_list.count() == 0:
                self.empty_list_placeholder.setText("No recoverable runs")
            else:
                self.empty_list_placeholder.setText("No runs match the current search / filter")
            self.list_stack.setCurrentIndex(1)
        else:
            self.list_stack.setCurrentIndex(0)

    @staticmethod
    def _parse_timestamp(ts: str):
        """Best-effort parse of a run timestamp string."""
        if not ts or ts == "Unknown":
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M:%S"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
        return None

    def toggle_preview(self):
        if (
            hasattr(self, "plot_animation")
            and self.plot_animation.state() == QPropertyAnimation.Running
        ):
            return

        self.plot_animation = QPropertyAnimation(self.plot_container, b"maximumHeight")
        self.plot_animation.setDuration(250)
        self.plot_animation.setEasingCurve(QEasingCurve.InOutQuad)

        if self.is_preview_visible:
            # Collapsing
            self.toggle_plot_btn.setIcon(self.icon_show)
            self.is_preview_visible = False
            self.plot_animation.setStartValue(self.plot_container.height())
            self.plot_animation.setEndValue(28)
            self.plot_animation.finished.connect(self.plot_stack.hide)
            self.plot_animation.finished.connect(lambda: self.plot_container.setMinimumHeight(28))

        else:
            # Expanding
            self.toggle_plot_btn.setIcon(self.icon_hide)
            self.is_preview_visible = True

            self.plot_stack.show()

            self.plot_animation.setStartValue(self.plot_container.height())
            self.plot_animation.setEndValue(300)
            self.plot_animation.valueChanged.connect(self.plot_container.setMinimumHeight)

            def on_expansion_finished():
                try:
                    self.plot_animation.valueChanged.disconnect(
                        self.plot_container.setMinimumHeight
                    )
                except:
                    pass
                self.plot_container.setMaximumHeight(16777215)
                self.plot_container.setMinimumHeight(150)  # Set a floor for the expanded state
                self.plot_widget.setMinimumHeight(40)
                self.updateGeometry()

            self.plot_animation.finished.connect(on_expansion_finished)

        self.plot_animation.start()

    def _extract_metadata(self, folderpath):
        duration = 0.0
        num_points = 0
        timestamp = "Unknown"
        virtual_csv_path = None

        try:
            zip_filename = next(
                (name for name in os.listdir(folderpath) if name.lower().endswith(".zip")),
                None,
            )
            if not zip_filename:
                return duration, num_points, timestamp, None

            zip_filepath = os.path.join(folderpath, zip_filename)

            csv_filename = None
            with zipfile.ZipFile(zip_filepath, "r") as z:
                csv_filename = next(
                    (
                        name
                        for name in z.namelist()
                        if name.endswith(".csv") and not name.split("/")[-1].startswith("._")
                    ),
                    None,
                )
            if not csv_filename:
                return duration, num_points, timestamp, None

            virtual_csv_path = os.path.join(folderpath, csv_filename)

            with secure_open(virtual_csv_path, "r") as f:
                text_io = io.TextIOWrapper(f, encoding="utf-8-sig")
                reader = csv.reader(text_io)

                header_row = None
                rel_time_idx = -1
                date_idx = -1
                time_idx = -1
                max_time = 0.0

                for row in reader:
                    if not row or not any(row):
                        continue

                    if header_row is None:
                        cleaned_row = [str(col).strip() for col in row]
                        if "Relative_time" in cleaned_row or "Date" in cleaned_row:
                            header_row = cleaned_row
                            rel_time_idx = (
                                header_row.index("Relative_time")
                                if "Relative_time" in header_row
                                else -1
                            )
                            date_idx = header_row.index("Date") if "Date" in header_row else -1
                            time_idx = header_row.index("Time") if "Time" in header_row else -1
                        continue

                    num_points += 1

                    if num_points == 1 and date_idx != -1 and time_idx != -1:
                        if len(row) > max(date_idx, time_idx):
                            date_str = str(row[date_idx]).strip()
                            time_str = str(row[time_idx]).strip()
                            if date_str and time_str:
                                timestamp = f"{date_str} {time_str}"

                    if rel_time_idx != -1 and len(row) > rel_time_idx:
                        try:
                            rel_time = float(str(row[rel_time_idx]).strip())
                            if rel_time > max_time:
                                max_time = rel_time
                        except (ValueError, TypeError):
                            pass

                duration = round(max_time, 2)
        except Exception as e:
            Log.e(TAG, f"Failed to extract metadata from {folderpath}: {str(e)}")

        return duration, num_points, timestamp, virtual_csv_path

    def load_unnamed_runs(self):
        self.unnamed_runs = []
        self.runs_list.clear()

        if not os.path.exists(self._UNNAMED_DIR):
            Log.w(TAG, f"Unnamed directory not found at {self._UNNAMED_DIR}")
            self.list_stack.setCurrentIndex(1)
            return

        try:
            for filename in os.listdir(self._UNNAMED_DIR):
                filepath = os.path.join(self._UNNAMED_DIR, filename)
                if not os.path.isdir(filepath):
                    continue

                if filename.endswith("_BAD"):
                    ruling = "Bad"
                    display_name = filename[:-4]
                else:
                    ruling = "Good"
                    display_name = filename

                (
                    duration,
                    num_points,
                    csv_timestamp,
                    virtual_csv_path,
                ) = self._extract_metadata(filepath)

                total_size = sum(
                    os.path.getsize(os.path.join(filepath, f))
                    for f in os.listdir(filepath)
                    if os.path.isfile(os.path.join(filepath, f))
                )
                file_size_mb = round(total_size / (1024 * 1024), 2)

                if csv_timestamp == "Unknown":
                    mtime = datetime.fromtimestamp(os.stat(filepath).st_mtime)
                    csv_timestamp = mtime.strftime("%Y-%m-%d %H:%M:%S")

                run = UnnamedRun(
                    filepath=filepath,
                    display_name=display_name,
                    timestamp=csv_timestamp,
                    duration_seconds=duration,
                    num_points=num_points,
                    ruling=ruling,
                    file_size_mb=file_size_mb,
                    virtual_csv_path=virtual_csv_path,
                )
                self.unnamed_runs.append(run)
        except Exception as e:
            Log.i(TAG, f"Error loading unnamed runs: {str(e)}")

        for run in self.unnamed_runs:
            item = QListWidgetItem(run.display_name)
            item.setData(Qt.UserRole, run)
            self.runs_list.addItem(item)

        # Toggle placeholder state
        if not self.unnamed_runs:
            self.empty_list_placeholder.setText("No recoverable runs")
            self.list_stack.setCurrentIndex(1)
        else:
            self.list_stack.setCurrentIndex(0)

    def update_plot_preview(self, run: UnnamedRun):
        if not run or not run.virtual_csv_path:
            self.curve_freq.setData([], [])
            self.curve_diss.setData([], [])
            self.plot_stack.setCurrentIndex(1)  # show placeholder
            return

        times, freqs, disss = [], [], []

        try:
            with secure_open(run.virtual_csv_path, "r") as f:
                text_io = io.TextIOWrapper(f, encoding="utf-8-sig")
                reader = csv.reader(text_io)

                header_row = None
                t_idx, f_idx, d_idx = -1, -1, -1

                for row in reader:
                    if not row or not any(row):
                        continue
                    if header_row is None:
                        cleaned_row = [str(col).strip() for col in row]
                        if "Relative_time" in cleaned_row:
                            header_row = cleaned_row
                            t_idx = (
                                header_row.index("Relative_time")
                                if "Relative_time" in header_row
                                else -1
                            )
                            f_idx = (
                                header_row.index("Resonance_Frequency")
                                if "Resonance_Frequency" in header_row
                                else -1
                            )
                            d_idx = (
                                header_row.index("Dissipation")
                                if "Dissipation" in header_row
                                else -1
                            )
                        continue

                    if t_idx != -1 and f_idx != -1 and d_idx != -1:
                        if len(row) > max(t_idx, f_idx, d_idx):
                            try:
                                times.append(float(str(row[t_idx]).strip()))
                                freqs.append(float(str(row[f_idx]).strip()))
                                disss.append(float(str(row[d_idx]).strip()))
                            except (ValueError, TypeError):
                                pass

            max_preview_points = 1500
            total_points = len(times)
            if total_points > max_preview_points:
                step = total_points // max_preview_points
                times = times[::step]
                freqs = freqs[::step]
                disss = disss[::step]

            if not times:
                self.curve_freq.setData([], [])
                self.curve_diss.setData([], [])
                self.plot_stack.setCurrentIndex(1)
                return

            self.curve_freq.setData(times, freqs)
            self.curve_diss.setData(times, disss)
            self.plot_widget.plotItem.vb.autoRange()
            self.view_box_diss.autoRange()
            self.plot_stack.setCurrentIndex(0)

        except Exception as e:
            Log.e(TAG, f"Failed to load preview data: {str(e)}")
            self.curve_freq.setData([], [])
            self.curve_diss.setData([], [])
            self.plot_stack.setCurrentIndex(1)

    def on_selection_changed(self):
        selected_items = self.runs_list.selectedItems()
        count = len(selected_items)

        if count == 1:
            run = selected_items[0].data(Qt.UserRole)
            self.selected_run = run

            self.detail_datetime.setText(run.timestamp)
            self.detail_duration.setText(f"{run.duration_seconds} seconds")
            self.detail_points.setText(f"{run.num_points:,}")

            ruling_color = "#2c8a3d" if run.ruling == "Good" else "#c12c3b"
            self.detail_ruling.setStyleSheet(f"color: {ruling_color}; font-size: 9pt;")
            self.detail_ruling.setText(run.ruling)

            self.detail_filesize.setText(f"{run.file_size_mb} MB")

            self.update_plot_preview(run)

            self.recover_button.setEnabled(True)
            self.delete_button.setEnabled(True)

        elif count > 1:
            self.selected_run = None
            self.clear_details()

            self.recover_button.setEnabled(False)
            self.delete_button.setEnabled(True)

        else:
            self.selected_run = None
            self.clear_details()
            self.recover_button.setEnabled(False)
            self.delete_button.setEnabled(False)

    def on_delete_clicked(self):
        selected_items = self.runs_list.selectedItems()
        if not selected_items:
            return

        count = len(selected_items)
        if count == 1:
            msg = (
                f"Are you sure you want to move "
                f"'{selected_items[0].text()}' to the Recycle Bin?"
            )
        else:
            msg = f"Are you sure you want to move {count} selected runs " f"to the Recycle Bin?"

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        success_count = 0
        runs_to_remove = []

        for item in selected_items:
            run = item.data(Qt.UserRole)
            folder_path = os.path.abspath(run.filepath)
            try:
                if os.path.exists(folder_path):
                    send2trash.send2trash(folder_path)
                    runs_to_remove.append((item, run))
                    success_count += 1
                else:
                    Log.e(TAG, f"Folder not found: {folder_path}")
            except Exception as e:
                Log.e(TAG, f"Failed to trash {folder_path}: {str(e)}")

        for item, run in runs_to_remove:
            row = self.runs_list.row(item)
            self.runs_list.takeItem(row)
            if run in self.unnamed_runs:
                self.unnamed_runs.remove(run)

        self.on_selection_changed()
        self.refilter_list()

        if success_count > 0:
            Log.i(
                TAG,
                f"Successfully moved {success_count} runs to the Recycle Bin.",
            )

    def on_item_double_clicked(self, item):
        run = item.data(Qt.UserRole)
        self.selected_run = run
        self.on_recover_clicked()

    def clear_details(self):
        self.detail_datetime.setText("—")
        self.detail_duration.setText("—")
        self.detail_points.setText("—")
        self.detail_ruling.setText("—")
        self.detail_ruling.setStyleSheet("color: #333333; font-size: 9pt;")
        self.detail_filesize.setText("—")

        self.curve_freq.setData([], [])
        self.curve_diss.setData([], [])
        self.plot_stack.setCurrentIndex(1)  # show empty placeholder

    def on_recover_clicked(self):
        if self.selected_run is None:
            return

        run_name, ok = QInputDialog.getText(
            self,
            "Recover Run",
            f"Enter name for run from {self.selected_run.timestamp}:",
            text="",
        )

        if ok and run_name.strip():
            if self.validate_run_name(run_name):
                recovery_dialog = RecoveryProgressDialog(self, self.selected_run, run_name)
                result = recovery_dialog.exec_()

                if result == QDialog.Accepted:
                    current_item = self.runs_list.currentItem()
                    if current_item is not None:
                        row = self.runs_list.row(current_item)
                        self.runs_list.takeItem(row)
                    if self.selected_run in self.unnamed_runs:
                        self.unnamed_runs.remove(self.selected_run)

                    self.clear_details()
                    self.recover_button.setEnabled(False)
                    self.refilter_list()
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Name",
                    "Run name already exists or contains invalid characters.",
                )

    def validate_run_name(self, name):
        if not name.strip():
            return False
        invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        if any(char in name for char in invalid_chars):
            return False
        return True


# Example usage
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    dialog = RecoverUnnamedRunsDialog()
    dialog.show()

    sys.exit(app.exec_())
