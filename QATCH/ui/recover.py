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
    QTimer,
)
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPainter, QRegion
from datetime import datetime
from typing import List
import numpy as np
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


class ScanWorker(QThread):
    """Worker thread that scans the unnamed-runs directory without blocking the UI."""

    scan_complete = pyqtSignal(list)  # list[UnnamedRun]
    scan_failed = pyqtSignal(str)

    def __init__(self, unnamed_dir):
        super().__init__()
        self.unnamed_dir = unnamed_dir
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        runs = []
        try:
            if not os.path.exists(self.unnamed_dir):
                self.scan_complete.emit(runs)
                return

            for filename in os.listdir(self.unnamed_dir):
                if self._is_cancelled:
                    return

                filepath = os.path.join(self.unnamed_dir, filename)
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

                try:
                    total_size = sum(
                        os.path.getsize(os.path.join(filepath, f))
                        for f in os.listdir(filepath)
                        if os.path.isfile(os.path.join(filepath, f))
                    )
                except OSError:
                    total_size = 0
                file_size_mb = round(total_size / (1024 * 1024), 2)

                if csv_timestamp == "Unknown":
                    try:
                        mtime = datetime.fromtimestamp(os.stat(filepath).st_mtime)
                        csv_timestamp = mtime.strftime("%Y-%m-%d %H:%M:%S")
                    except OSError:
                        pass

                runs.append(
                    UnnamedRun(
                        filepath=filepath,
                        display_name=display_name,
                        timestamp=csv_timestamp,
                        duration_seconds=duration,
                        num_points=num_points,
                        ruling=ruling,
                        file_size_mb=file_size_mb,
                        virtual_csv_path=virtual_csv_path,
                    )
                )

            self.scan_complete.emit(runs)
        except Exception as e:
            self.scan_failed.emit(str(e))

    @staticmethod
    def _extract_metadata(folderpath):
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
                color: #666666;
                font-size: 9pt;
                background: transparent;
                border: none;
            }
            QLabel#titleLabel {
                color: #222222;
                font-size: 10pt;
                font-weight: 600;
            }
            QLabel#sectionLabel {
                color: #333333;
                font-size: 9pt;
                font-weight: 600;
                padding-top: 2px;
            }
            QLabel#rangeSep {
                color: #999999;
                font-size: 9pt;
                padding: 0 2px;
            }
            QComboBox, QSpinBox, QDoubleSpinBox, QDateTimeEdit {
                background-color: #ffffff;
                border: 1px solid rgba(0, 0, 0, 25);
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 9pt;
                color: #333333;
                min-height: 20px;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateTimeEdit:focus {
                border: 1px solid rgba(0, 114, 189, 120);
            }
            QDateTimeEdit::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 18px;
                border: none;
            }
            QCheckBox {
                color: #555555;
                font-size: 9pt;
                spacing: 6px;
                background: transparent;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid rgba(0, 0, 0, 40);
                border-radius: 3px;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                background-color: rgba(0, 114, 189, 200);
                border: 1px solid rgba(0, 114, 189, 220);
                image: none;
            }
            QFrame#filterSeparator {
                background-color: rgba(0, 0, 0, 15);
                border: none;
                max-height: 1px;
                min-height: 1px;
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
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(10)

        title = QLabel("Filter Options")
        title.setObjectName("titleLabel")
        main_layout.addWidget(title)

        # Single vertical layout where each "section" is a label + controls.
        # This keeps alignment predictable regardless of label/field widths.
        sections_layout = QVBoxLayout()
        sections_layout.setContentsMargins(0, 0, 0, 0)
        sections_layout.setSpacing(10)

        # ---- Status ----
        status_section = self._make_section("Status")
        self.status_combo = QComboBox()
        self.status_combo.addItems(["Any", "Good", "Bad"])
        self.status_combo.setMinimumWidth(160)
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(6)
        status_row.addWidget(self.status_combo, 1)
        status_section.addLayout(status_row)
        sections_layout.addLayout(status_section)

        # ---- Date / Time range ----
        date_section = self._make_section("Date / Time")
        self.date_enabled = QCheckBox("Filter by date range")
        self.date_enabled.toggled.connect(self._on_date_toggled)
        date_section.addWidget(self.date_enabled)

        self.date_from = QDateTimeEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.date_from.setDateTime(QDateTime.currentDateTime().addDays(-30))
        self.date_to = QDateTimeEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.date_to.setDateTime(QDateTime.currentDateTime())

        date_row = QHBoxLayout()
        date_row.setContentsMargins(0, 0, 0, 0)
        date_row.setSpacing(6)
        date_row.addWidget(self.date_from, 1)
        sep_lbl = QLabel("–")
        sep_lbl.setObjectName("rangeSep")
        date_row.addWidget(sep_lbl)
        date_row.addWidget(self.date_to, 1)
        date_section.addLayout(date_row)
        sections_layout.addLayout(date_section)

        # ---- Numeric ranges (duration / points / size) ----
        self.duration_min = self._make_double_spin()
        self.duration_max = self._make_double_spin()
        self.duration_max.setToolTip("0 = no upper limit")
        sections_layout.addLayout(
            self._make_range_section("Duration (s)", self.duration_min, self.duration_max)
        )

        self.points_min = self._make_int_spin()
        self.points_max = self._make_int_spin()
        self.points_max.setToolTip("0 = no upper limit")
        sections_layout.addLayout(
            self._make_range_section("Data Points", self.points_min, self.points_max)
        )

        self.size_min = self._make_double_spin()
        self.size_max = self._make_double_spin()
        self.size_max.setToolTip("0 = no upper limit")
        sections_layout.addLayout(
            self._make_range_section("File Size (MB)", self.size_min, self.size_max)
        )

        main_layout.addLayout(sections_layout)

        # Separator above buttons
        sep = QFrame()
        sep.setObjectName("filterSeparator")
        main_layout.addWidget(sep)

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

        self.setMinimumWidth(340)

        self._populate_from(current_filters or {})

    # ---------- builder helpers ----------

    def _make_section(self, title_text):
        """Return a VBox layout with a titled section header already added."""
        section = QVBoxLayout()
        section.setContentsMargins(0, 0, 0, 0)
        section.setSpacing(4)
        lbl = QLabel(title_text)
        lbl.setObjectName("sectionLabel")
        section.addWidget(lbl)
        return section

    def _make_double_spin(self):
        w = QDoubleSpinBox()
        w.setRange(0.0, 1_000_000.0)
        w.setDecimals(2)
        w.setValue(0.0)
        return w

    def _make_int_spin(self):
        w = QSpinBox()
        w.setRange(0, 100_000_000)
        w.setValue(0)
        return w

    def _make_range_section(self, title_text, min_widget, max_widget):
        section = self._make_section(title_text)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        min_lbl = QLabel("Min")
        max_lbl = QLabel("Max")

        row.addWidget(min_lbl)
        row.addWidget(min_widget, 1)
        row.addSpacing(6)
        row.addWidget(max_lbl)
        row.addWidget(max_widget, 1)
        section.addLayout(row)
        return section

    # ---------- state ----------

    def _on_date_toggled(self, checked):
        self.date_from.setEnabled(checked)
        self.date_to.setEnabled(checked)

    def _populate_from(self, f):
        status = f.get("status")
        if status in ("Good", "Bad"):
            self.status_combo.setCurrentText(status)
        else:
            self.status_combo.setCurrentIndex(0)  # "Any"

        # Date range
        has_date = "date_from" in f and "date_to" in f
        self.date_enabled.setChecked(has_date)
        if has_date:
            try:
                self.date_from.setDateTime(
                    QDateTime.fromString(
                        f["date_from"].strftime("%Y-%m-%d %H:%M:%S"),
                        "yyyy-MM-dd HH:mm:ss",
                    )
                )
                self.date_to.setDateTime(
                    QDateTime.fromString(
                        f["date_to"].strftime("%Y-%m-%d %H:%M:%S"),
                        "yyyy-MM-dd HH:mm:ss",
                    )
                )
            except Exception:
                pass
        self._on_date_toggled(self.date_enabled.isChecked())

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

        if self.date_enabled.isChecked():
            filters["date_from"] = self.date_from.dateTime().toPyDateTime()
            filters["date_to"] = self.date_to.dateTime().toPyDateTime()

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
        self.date_enabled.setChecked(False)
        self.date_from.setDateTime(QDateTime.currentDateTime().addDays(-30))
        self.date_to.setDateTime(QDateTime.currentDateTime())
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
            QPushButton:disabled {
                background-color: rgba(255, 255, 255, 120);
                border: 1px solid rgba(0, 0, 0, 8);
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

        # Rescan lives in the top bar; runs on a worker and animates while scanning
        self._rescan_icon_path = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "refresh-cw.svg"
        )
        self._rescan_base_pixmap = QIcon(self._rescan_icon_path).pixmap(QSize(16, 16))
        self._rescan_angle = 0
        self._rescan_timer = QTimer(self)
        self._rescan_timer.setInterval(16)  # ~60 fps
        self._rescan_timer.timeout.connect(self._tick_rescan_spinner)

        self.rescan_btn = QPushButton()
        self.rescan_btn.setFixedSize(28, 28)
        self.rescan_btn.setCursor(Qt.PointingHandCursor)
        self.rescan_btn.setToolTip("Rescan for unnamed runs")
        self.rescan_btn.setIcon(QIcon(self._rescan_base_pixmap))
        self.rescan_btn.setIconSize(QSize(16, 16))
        self.rescan_btn.setStyleSheet(_icon_btn_ss)
        self.rescan_btn.clicked.connect(self.load_unnamed_runs)

        search_layout.addWidget(self.search_bar, stretch=1)
        search_layout.addWidget(self.filter_btn)
        search_layout.addWidget(self.rescan_btn)
        top_master_layout.addLayout(search_layout)

        # 2. Details, Buttons (right)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)

        # LEFT — list column with sort bar above the list
        left_column = QWidget()
        left_column.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_column.setMinimumWidth(0)
        left_column_layout = QVBoxLayout(left_column)
        left_column_layout.setContentsMargins(0, 0, 0, 0)
        left_column_layout.setSpacing(6)

        # Sort bar — minimal chrome, matches the rest of the dialog
        sort_bar = QWidget()
        sort_bar.setStyleSheet(
            """
            QWidget#sortBar { background: transparent; }
            QLabel#sortLabel {
                color: #888888;
                font-size: 9pt;
                background: transparent;
                border: none;
                padding-left: 2px;
            }
            QComboBox#sortCombo {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 2px 6px;
                padding-right: 18px;
                color: #444444;
                font-size: 9pt;
                min-width: 90px;
            }
            QComboBox#sortCombo:hover {
                background-color: rgba(0, 0, 0, 8);
                border: 1px solid rgba(0, 0, 0, 12);
            }
            QComboBox#sortCombo:focus {
                border: 1px solid rgba(0, 114, 189, 100);
            }
            QComboBox#sortCombo::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 16px;
                border: none;
            }
            QPushButton#sortDir {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                color: #555555;
                font-size: 11pt;
                font-weight: 600;
                min-width: 22px;
                min-height: 22px;
                padding: 0px;
            }
            QPushButton#sortDir:hover {
                background-color: rgba(0, 0, 0, 8);
                border: 1px solid rgba(0, 0, 0, 12);
            }
            QPushButton#sortDir:pressed {
                background-color: rgba(0, 0, 0, 15);
            }
            """
        )
        sort_bar.setObjectName("sortBar")

        sort_bar_layout = QHBoxLayout(sort_bar)
        sort_bar_layout.setContentsMargins(4, 0, 2, 0)
        sort_bar_layout.setSpacing(4)

        sort_label = QLabel("Sort by")
        sort_label.setObjectName("sortLabel")

        self.sort_combo = QComboBox()
        self.sort_combo.setObjectName("sortCombo")
        self.sort_combo.setCursor(Qt.PointingHandCursor)
        # (display text, sort key)
        for text, key in (
            ("Name", "name"),
            ("Date", "date"),
            ("Duration", "duration"),
            ("Points", "points"),
            ("File Size", "size"),
        ):
            self.sort_combo.addItem(text, key)
        self.sort_combo.setCurrentIndex(1)  # default: Date
        self.sort_combo.currentIndexChanged.connect(lambda *_: self._sort_runs())

        self._sort_ascending = False  # newest first by default
        self.sort_dir_btn = QPushButton()
        self.sort_dir_btn.setIcon(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "descending.svg"))
        )
        self.sort_dir_btn.setObjectName("sortDir")
        self.sort_dir_btn.setCursor(Qt.PointingHandCursor)
        self.sort_dir_btn.setToolTip("Toggle sort direction")
        self.sort_dir_btn.setFixedSize(24, 24)
        self.sort_dir_btn.clicked.connect(self._toggle_sort_direction)

        sort_bar_layout.addWidget(sort_label)
        sort_bar_layout.addWidget(self.sort_combo)
        sort_bar_layout.addWidget(self.sort_dir_btn)
        sort_bar_layout.addStretch(1)

        left_column_layout.addWidget(sort_bar)

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

        left_column_layout.addWidget(self.list_stack, stretch=1)
        content_layout.addWidget(left_column, stretch=1)

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
            QFrame#detailsFrame {
                background-color: rgba(255, 255, 255, 180);
                border: 1px solid rgba(0, 0, 0, 15);
                border-radius: 6px;
            }
            QFrame#detailsFrame QLabel {
                border: none;
                background: transparent;
                color: #444444;
                font-size: 9pt;
            }
            QFrame#detailsSep {
                background-color: rgba(0, 0, 0, 15);
                border: none;
                max-height: 1px;
                min-height: 1px;
            }
            """
        )
        self.details_frame.setObjectName("detailsFrame")

        details_outer = QVBoxLayout(self.details_frame)
        details_outer.setContentsMargins(12, 10, 12, 8)
        details_outer.setSpacing(8)

        details_form = QFormLayout()
        details_form.setContentsMargins(0, 0, 0, 0)
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

        details_outer.addLayout(details_form)

        # Thin separator between info and actions
        details_sep = QFrame()
        details_sep.setObjectName("detailsSep")
        details_outer.addWidget(details_sep)

        # Borderless text action buttons — hover-only visual affordance
        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(4)

        self.recover_button = QPushButton("  Recover")
        self.recover_button.setCursor(Qt.PointingHandCursor)
        self.recover_button.setEnabled(False)
        self.recover_button.setIcon(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "restore.svg"))
        )
        self.recover_button.setIconSize(QSize(14, 14))
        self.recover_button.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                color: rgba(0, 91, 159, 175);
                font-size: 8pt;
                font-weight: 500;
                padding: 5px 9px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: rgba(0, 114, 189, 22);
                color: rgba(0, 91, 159, 230);
            }
            QPushButton:pressed {
                background-color: rgba(0, 114, 189, 42);
            }
            QPushButton:disabled {
                color: rgba(180, 180, 180, 150);
            }
            """
        )
        self.recover_button.clicked.connect(self.on_recover_clicked)

        self.delete_button = QPushButton("  Delete")
        self.delete_button.setCursor(Qt.PointingHandCursor)
        self.delete_button.setEnabled(False)
        self.delete_button.setIcon(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "delete.svg"))
        )
        self.delete_button.setIconSize(QSize(14, 14))
        self.delete_button.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                color: rgba(176, 42, 56, 175);
                font-size: 8pt;
                font-weight: 500;
                padding: 5px 9px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: rgba(220, 53, 69, 22);
                color: rgba(176, 42, 56, 230);
            }
            QPushButton:pressed {
                background-color: rgba(220, 53, 69, 42);
            }
            QPushButton:disabled {
                color: rgba(180, 180, 180, 150);
            }
            """
        )
        self.delete_button.clicked.connect(self.on_delete_clicked)

        action_row.addWidget(self.recover_button)
        action_row.addStretch(1)
        action_row.addWidget(self.delete_button)

        details_outer.addLayout(action_row)

        right_layout.addWidget(self.details_frame)

        # ---- Small plot preview card, sits right below the details ----
        self.plot_card = QFrame()
        self.plot_card.setObjectName("plotCard")
        self.plot_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_card.setMinimumHeight(150)
        self.plot_card.setStyleSheet(
            """
            QFrame#plotCard {
                background-color: rgba(255, 255, 255, 180);
                border: 1px solid rgba(0, 0, 0, 15);
                border-radius: 6px;
            }
            QLabel#plotLegend {
                background: transparent;
                border: none;
                color: #666666;
                font-size: 8pt;
            }
            """
        )

        plot_card_layout = QVBoxLayout(self.plot_card)
        plot_card_layout.setContentsMargins(6, 6, 6, 4)
        plot_card_layout.setSpacing(2)

        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_widget.setBackground("#fafafa")
        self.plot_widget.setStyleSheet("border: none;")
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideButtons()
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)

        # No visible axes at this preview size — it's just a silhouette
        self.plot_widget.hideAxis("left")
        self.plot_widget.hideAxis("right")
        bottom_axis = self.plot_widget.getAxis("bottom")
        bottom_axis.setPen(pg.mkPen(color="#cccccc", width=1))
        bottom_axis.setLabel(text="")
        bottom_axis.setStyle(showValues=False)

        # Softer / "glassy" primary colors — same hue family as before but
        # lightened a touch and with slightly more transparency.
        self._FREQ_COLOR = (82, 142, 201)  # soft sky blue
        self._DISS_COLOR = (225, 175, 85)  # soft amber
        pen_freq = pg.mkPen(color=(*self._FREQ_COLOR, 170), width=1.4)
        pen_diss = pg.mkPen(color=(*self._DISS_COLOR, 170), width=1.4)

        self.curve_freq = self.plot_widget.plot(pen=pen_freq)

        self.view_box_diss = pg.ViewBox()
        self.plot_widget.scene().addItem(self.view_box_diss)
        self.view_box_diss.setXLink(self.plot_widget)
        self.view_box_diss.setMouseEnabled(x=False, y=False)
        self.view_box_diss.setMenuEnabled(False)

        self.curve_diss = pg.PlotCurveItem(pen=pen_diss)
        self.view_box_diss.addItem(self.curve_diss)

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
                font-size: 9pt;
                border-radius: 4px;
            }
            """
        )

        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(self.plot_widget)  # index 0
        self.plot_stack.addWidget(self.empty_plot_placeholder)  # index 1
        self.plot_stack.setCurrentIndex(1)

        plot_card_layout.addWidget(self.plot_stack, stretch=1)

        # Tiny legend row underneath
        freq_hex = "#%02x%02x%02x" % self._FREQ_COLOR
        diss_hex = "#%02x%02x%02x" % self._DISS_COLOR
        self.plot_legend_label = QLabel(
            f'<span style="color:{freq_hex};">●</span>'
            f'<span style="color:#777777;"> Frequency</span>'
            f"&nbsp;&nbsp;&nbsp;"
            f'<span style="color:{diss_hex};">●</span>'
            f'<span style="color:#777777;"> Dissipation</span>'
        )
        self.plot_legend_label.setObjectName("plotLegend")
        self.plot_legend_label.setAlignment(Qt.AlignCenter)
        self.plot_legend_label.setTextFormat(Qt.RichText)
        plot_card_layout.addWidget(self.plot_legend_label)

        right_layout.addWidget(self.plot_card, stretch=1)

        content_layout.addWidget(right_container, stretch=1)
        top_master_layout.addLayout(content_layout)

        main_layout.addLayout(top_master_layout, stretch=1)

        self.setLayout(main_layout)

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

        start_h = self.plot_container.height()

        if self.is_preview_visible:
            # Collapsing — container drives down to 28; plot_stack fades out
            target_h = self._PLOT_COLLAPSED_HEIGHT
            self.toggle_plot_btn.setIcon(self.icon_show)
            self.is_preview_visible = False
        else:
            # Expanding — show the plot stack BEFORE the animation starts
            target_h = self._PLOT_EXPANDED_HEIGHT
            self.toggle_plot_btn.setIcon(self.icon_hide)
            self.is_preview_visible = True
            self.plot_stack.setVisible(True)

        # Drive maximumHeight; mirror it onto minimumHeight each frame so the
        # layout engine doesn't have any slack to snap against at the ends.
        self.plot_animation = QPropertyAnimation(self.plot_container, b"maximumHeight")
        self.plot_animation.setDuration(self._PLOT_ANIM_DURATION)
        self.plot_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.plot_animation.setStartValue(start_h)
        self.plot_animation.setEndValue(target_h)
        self.plot_animation.valueChanged.connect(self._sync_plot_min_height)

        def on_finished():
            # Lock the final geometry exactly so nothing jumps after the curve ends
            self.plot_container.setMinimumHeight(target_h)
            self.plot_container.setMaximumHeight(target_h)
            # Fully hide the stack after collapse so the plot can't leak through
            if not self.is_preview_visible:
                self.plot_stack.setVisible(False)

        self.plot_animation.finished.connect(on_finished)
        self.plot_animation.start()

    def _sync_plot_min_height(self, h):
        """Keep min height == max height during the toggle animation."""
        self.plot_container.setMinimumHeight(int(h))

    # ----- Rescan spinner animation -----

    def _tick_rescan_spinner(self):
        self._rescan_angle = (self._rescan_angle + 12) % 360
        size = self._rescan_base_pixmap.size()
        if size.isEmpty():
            return
        canvas = QPixmap(size)
        canvas.fill(Qt.transparent)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)
        cx = size.width() / 2
        cy = size.height() / 2
        painter.translate(cx, cy)
        painter.rotate(self._rescan_angle)
        painter.translate(-cx, -cy)
        painter.drawPixmap(0, 0, self._rescan_base_pixmap)
        painter.end()
        self.rescan_btn.setIcon(QIcon(canvas))

    def _start_rescan_animation(self):
        self._rescan_angle = 0
        self.rescan_btn.setEnabled(False)
        self._rescan_timer.start()

    def _stop_rescan_animation(self):
        self._rescan_timer.stop()
        self.rescan_btn.setIcon(QIcon(self._rescan_base_pixmap))
        self.rescan_btn.setEnabled(True)

    # ----- Sorting -----

    def _toggle_sort_direction(self):
        self._sort_ascending = not self._sort_ascending
        icon_asc = QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "ascending.svg"))
        icon_desc = QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "descending.svg"))
        new_icon = icon_asc if self._sort_ascending else icon_desc
        self.sort_dir_btn.setIcon(new_icon)
        self.sort_dir_btn.setText("")  # Ensure text is cleared
        self.sort_dir_btn.setToolTip("Ascending" if self._sort_ascending else "Descending")
        self._sort_runs()

    def _sort_runs(self):
        """Re-order the items in runs_list based on the current sort key/direction.

        Works on the backing self.unnamed_runs list so ordering survives rescans.
        Preserves the current selection and then re-applies the active filter.
        """
        if not hasattr(self, "sort_combo"):
            return

        key = self.sort_combo.currentData()
        ascending = self._sort_ascending

        def sort_key(r: UnnamedRun):
            if key == "name":
                return (r.display_name or "").lower()
            if key == "date":
                dt = self._parse_timestamp(r.timestamp)
                return dt or datetime.min
            if key == "duration":
                return r.duration_seconds or 0.0
            if key == "points":
                return r.num_points or 0
            if key == "size":
                return r.file_size_mb or 0.0
            return (r.display_name or "").lower()

        try:
            self.unnamed_runs.sort(key=sort_key, reverse=not ascending)
        except TypeError:
            # Mixed types (e.g. date parsing produced None vs datetime): fall back to name
            self.unnamed_runs.sort(
                key=lambda r: (r.display_name or "").lower(), reverse=not ascending
            )

        selected_runs = {item.data(Qt.UserRole) for item in self.runs_list.selectedItems()}

        self.runs_list.blockSignals(True)
        self.runs_list.clear()
        for run in self.unnamed_runs:
            item = QListWidgetItem(run.display_name)
            item.setData(Qt.UserRole, run)
            self.runs_list.addItem(item)
            if run in selected_runs:
                item.setSelected(True)
        self.runs_list.blockSignals(False)

        self.refilter_list()
        # Explicitly refresh, since the block above suppressed the signal
        self.on_selection_changed()

    # ----- Scan / reload -----

    def load_unnamed_runs(self):
        """Kick off an async scan. Safe to call from init or from the rescan button."""
        # If a scan is already running, ignore additional requests
        if getattr(self, "_scan_worker", None) and self._scan_worker.isRunning():
            return

        # Reset the list UI immediately so the user sees the scan starting
        self.unnamed_runs = []
        self.runs_list.clear()
        self.clear_details()
        self.recover_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        self.empty_list_placeholder.setText("Scanning for recoverable runs…")
        self.list_stack.setCurrentIndex(1)

        self._start_rescan_animation()

        self._scan_worker = ScanWorker(self._UNNAMED_DIR)
        self._scan_worker.scan_complete.connect(self._on_scan_complete)
        self._scan_worker.scan_failed.connect(self._on_scan_failed)
        self._scan_worker.finished.connect(self._stop_rescan_animation)
        self._scan_worker.start()

    def _on_scan_complete(self, runs):
        self.unnamed_runs = list(runs)

        # Apply current sort choice, then populate the list widget
        self._sort_runs()

        # _sort_runs() already repopulates and calls refilter_list,
        # but when the list is empty sort_runs is a no-op on unnamed_runs,
        # so make sure we show the right placeholder.
        if not self.unnamed_runs:
            self.empty_list_placeholder.setText("No recoverable runs")
            self.list_stack.setCurrentIndex(1)

    def _on_scan_failed(self, message):
        Log.i(TAG, f"Error loading unnamed runs: {message}")
        self.unnamed_runs = []
        self.runs_list.clear()
        self.empty_list_placeholder.setText("No recoverable runs")
        self.list_stack.setCurrentIndex(1)

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

            # Smooth out high-frequency noise for a cleaner preview
            freqs_s = self._smooth(freqs)
            disss_s = self._smooth(disss)

            self.curve_freq.setData(times, freqs_s)
            self.curve_diss.setData(times, disss_s)
            self.plot_widget.plotItem.vb.autoRange()
            self.view_box_diss.autoRange()
            self.plot_stack.setCurrentIndex(0)

        except Exception as e:
            Log.e(TAG, f"Failed to load preview data: {str(e)}")
            self.curve_freq.setData([], [])
            self.curve_diss.setData([], [])
            self.plot_stack.setCurrentIndex(1)

    @staticmethod
    def _smooth(values, window=9):
        """Centered moving-average smoother. Preserves endpoints so the
        first/last samples aren't pulled toward zero.
        """
        n = len(values)
        if n < max(window, 3):
            return values
        arr = np.asarray(values, dtype=float)
        # Pad using edge values so convolution doesn't dip at the boundaries
        pad = window // 2
        padded = np.concatenate([np.full(pad, arr[0]), arr, np.full(pad, arr[-1])])
        kernel = np.ones(window) / window
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed.tolist()

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
