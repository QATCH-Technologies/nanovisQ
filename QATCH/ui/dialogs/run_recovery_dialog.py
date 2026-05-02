"""
run_recovery_ui.py

This module provides a robust framework for recovering and managing unnamed data runs
within the QATCH software environment. It handles the identification of raw data folders,
extraction of metadata, secure generation of XML audit trails with cryptographic signatures,
and the organized relocation of processed runs to the system's logged data storage.

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-04-28

Version:
    1.0.3
"""

import os
import zipfile
import csv
import io
import shutil
from xml.dom import minidom
from contextlib import suppress
from datetime import datetime, timedelta
import datetime as dt
import hashlib
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QWidget,
    QLabel,
    QPushButton,
    QFormLayout,
    QFrame,
    QSizePolicy,
    QAbstractItemView,
    QMessageBox,
    QListWidgetItem,
    QLineEdit,
    QStackedWidget,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QDateEdit,
    QTimeEdit,
    QGraphicsDropShadowEffect,
    QGraphicsBlurEffect,
    QGraphicsOpacityEffect,
    QProgressBar,
    QDialogButtonBox,
    QCheckBox,
    QInputDialog,
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
    QDate,
    QTime,
    QTimer,
    QVariantAnimation,
    QParallelAnimationGroup,
    QAbstractAnimation,
)
from PyQt5.QtGui import (
    QIcon,
    QPixmap,
    QPainter,
    QPaintEvent,
    QMouseEvent,
    QCloseEvent,
    QBrush,
    QColor,
)
from typing import List, Optional, Any, Dict, BinaryIO, Sequence, Set, Tuple, cast
import numpy as np
import send2trash
import pyqtgraph as pg


from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants
from QATCH.common.architecture import Architecture
from QATCH.common.fileStorage import secure_open
from QATCH.common.userProfiles import UserProfiles
from QATCH.ui.workers.recovery_worker import RecoveryWorker
from QATCH.ui.workers.scan_worker import ScanWorker
from QATCH.ui.dialogs.signature_dialog import SignatureDialog
from QATCH.ui.widgets.recovery_filter_widget import RecoveryFilterWidget
from QATCH.ui.widgets.toggle_list_widget import ToggleListWidget

TAG = "[RunRecovery]"


class RunMetadata:
    """A data class representing information for a run that has not been named.

    This class serves as a container for metadata extracted from raw run files
    located in the unnamed recovery directory.

    Attributes:
        filepath (str): The absolute path to the directory containing the run data.
        display_name (str): The name of the run derived from the folder name.
        start (str): The date and time the run started in ISO format.
        stop (str): The date and time the run ended in ISO format.
        duration (float): The total length of the run in seconds.
        samples (int): The total count of data points recorded in the run.
        ruling (str): The classification of the run (e.g., "Good" or "Bad").
        file_size_mb (float): The total size of the run folder in megabytes.
        virtual_csv_path (str | None): The path to the CSV file within a ZIP archive.
    """

    def __init__(
        self,
        filepath: str,
        display_name: str,
        start: str,
        stop: str,
        duration: float,
        samples: int,
        ruling: str,
        file_size_mb: float,
        virtual_csv_path: str | None = None,
    ):
        self.filepath = filepath
        self.display_name = display_name
        self.start = start
        self.stop = stop
        self.duration = duration
        self.samples = samples
        self.ruling = ruling
        self.file_size_mb = file_size_mb
        self.virtual_csv_path = virtual_csv_path


class RecoveryDialog(QDialog):
    """Standard dialog for recovering a run with device creation and progress tracking.

    This dialog allows users to specify a new name for a recovered run, select or
    create a target device directory, and provides visual feedback during the
    asynchronous recovery process.

    Attributes:
        run_metadata (RunMetadata): The metadata object associated with the run.
        name_input (QLineEdit): Widget for entering the new run name.
        device_combo (QComboBox): Dropdown for selecting available device directories.
        add_device_btn (QPushButton): Button to trigger new device folder creation.
        progress_bar (QProgressBar): Visual indicator of the recovery task progress.
        status_label (QLabel): Displays error messages or status updates.
        btn_box (QDialogButtonBox): Standard Ok/Cancel buttons.
        worker (RecoveryWorker): The background thread handling the file operations.
    """

    def __init__(self, run_metadata, available_devices: List[str], parent=None) -> None:
        """Initializes the RecoverDialog with run information and UI components.

        Args:
            run_metadata (RunMetadata): Object containing the original run details.
            available_devices (List[str]): List of device names to populate the dropdown.
            parent (QWidget, optional): Parent widget. Defaults to None to prevent
                transparency inheritance issues.
        """
        super().__init__(None)
        self.setWindowTitle("Recover Run")
        self.setModal(True)
        self.setMinimumWidth(350)
        self.setWindowIcon(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "restore.svg"))
        )

        self.run_metadata = run_metadata
        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("Enter run name...")
        if hasattr(run_metadata, "display_name"):
            self.name_input.setText(run_metadata.display_name)

        # Device Selection Row
        device_layout = QHBoxLayout()
        self.device_combo = QComboBox(self)
        if available_devices:
            self.device_combo.addItems(available_devices)
        else:
            self.device_combo.addItem("No devices found")
            self.device_combo.setEnabled(False)
        self.add_device_btn = QPushButton()
        self.add_device_btn.setIcon(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "add.svg"))
        )
        self.add_device_btn.setFixedWidth(30)
        self.add_device_btn.setToolTip("Add New Device Folder")
        self.add_device_btn.clicked.connect(self._add_new_device)

        device_layout.addWidget(self.device_combo)
        device_layout.addWidget(self.add_device_btn)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()

        self.status_label = QLabel("", self)
        self.status_label.setStyleSheet("color: red;")
        self.status_label.hide()

        # Buttons
        self.btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_button = self.btn_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText("Recover")

        form_layout = QFormLayout()
        form_layout.addRow("Run Name:", self.name_input)
        form_layout.addRow("Target Device:", device_layout)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.btn_box)
        self.btn_box.accepted.connect(self._on_recover_clicked)
        self.btn_box.rejected.connect(self.reject)

    def _add_new_device(self) -> None:
        """Prompts the user for a name and creates a physical folder in the config directory.

        Opens a text input dialog. If a name is provided, it ensures the config
        directory exists, creates the subfolder, and updates the device selection
        dropdown to reflect the new entry.
        """
        name, ok = QInputDialog.getText(self, "Add Device", "Device Serial/Name:")
        if ok and name.strip():
            device_name = name.strip()
            try:
                config_dir = os.path.join(Constants.local_app_data_path, "config")
                # Ensure config directory exists
                os.makedirs(config_dir, exist_ok=True)

                new_device_path = os.path.join(config_dir, device_name)
                # Create the new device folder
                if not os.path.exists(new_device_path):
                    os.makedirs(new_device_path)

                # Update UI
                if not self.device_combo.isEnabled():
                    self.device_combo.clear()
                    self.device_combo.setEnabled(True)

                if self.device_combo.findText(device_name) == -1:
                    self.device_combo.addItem(device_name)

                self.device_combo.setCurrentText(device_name)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not create device folder:\n{str(e)}")

    def _on_recover_clicked(self) -> None:
        """Starts the worker thread and disables UI inputs after capturing signature.

        Validates the user input (run name and device selection) and prompts for an
        electronic signature. Upon successful signature, it initializes and starts
        the RecoveryWorker thread while locking the UI to prevent concurrent edits.
        """
        new_name = self.name_input.text().strip()
        device = self.device_combo.currentText()

        if not new_name:
            self.status_label.setText("Run name cannot be empty.")
            self.status_label.show()
            return

        if not self.device_combo.isEnabled() or device == "No devices found":
            self.status_label.setText("No valid device selected.")
            self.status_label.show()
            return

        # Trigger signature request
        sig_dialog = SignatureDialog(self)
        if sig_dialog.exec_() != QDialog.Accepted:
            return

        initials = sig_dialog.get_initials()

        # Disable UI for processing
        self.name_input.setEnabled(False)
        self.device_combo.setEnabled(False)
        self.add_device_btn.setEnabled(False)
        self.btn_box.button(QDialogButtonBox.Ok).setEnabled(False)

        self.status_label.hide()
        self.progress_bar.show()

        # Start background worker
        self.worker = RecoveryWorker(self.run_metadata, new_name, device, initials)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished_task.connect(self._on_worker_finished)
        self.worker.start()

    def _on_worker_finished(self, success: bool, error_message: str) -> None:
        """Handles the completion of the background recovery task.

        If the task is successful, the dialog is accepted and closed. If it fails,
        the UI is re-enabled to allow the user to correct errors, and the
        error message is displayed.

        Args:
            success (bool): Whether the recovery task finished without errors.
            error_message (str): The error description if success is False.
        """
        if success:
            self.accept()
        else:
            self.name_input.setEnabled(True)
            self.device_combo.setEnabled(True)
            self.add_device_btn.setEnabled(True)
            self.btn_box.button(QDialogButtonBox.Ok).setEnabled(True)
            self.progress_bar.hide()
            self.status_label.setText(f"Error: {error_message}")
            self.status_label.show()


class ToggleListWidget(QListWidget):
    """A QListWidget that allows toggling item selection.

    Extends the standard QListWidget to provide a "toggle" behavior where
    clicking an already-selected item deselects it, provided it is the
    only item currently selected.
    """

    def mousePressEvent(self, event: QMouseEvent):  # noqa: N802
        """Overrides the mouse press event to handle selection toggling.

        If a user left-clicks a single item that is already selected (without
        keyboard modifiers), the selection is cleared. Otherwise, the standard
        QListWidget selection behavior is executed.

        Args:
            event (QMouseEvent): The mouse event containing position,
                button, and modifier information.
        """
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
        ):
            item = self.itemAt(event.pos())
            if item is not None and item.isSelected() and len(self.selectedItems()) == 1:
                self.clearSelection()
                self.setCurrentItem(None)
                event.accept()
                return
        super().mousePressEvent(event)


class RoundedPanel(QFrame):
    """A custom QFrame that renders with rounded corners and a soft border.

    This widget uses QPainter to draw a stylized background. It is designed to
    look like a modern "card" or panel, featuring a semi-transparent white
    fill and a very subtle dark border.

    Attributes:
        None (Inherits from QFrame)
    """

    def __init__(self, parent: Optional[Any] = None) -> None:
        """Initializes the panel and sets transparency attributes.

        Args:
            parent (QWidget, optional): The parent widget.
        """
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """Overridden paint event to draw the rounded rectangle geometry.

        Args:
            event (QPaintEvent): The event triggered by the Qt paint engine.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor(0, 0, 0, 18))
        painter.setBrush(QColor(255, 255, 255, 244))
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.drawRoundedRect(rect, 8.0, 8.0)


class RunRecoveryDialog(QWidget):
    """Dialog for recovering and managing unnamed experimental runs.

    This dialog scans a specific directory for orphaned or unnamed run data,
    displays them in a filterable list, and provides a preview interface for
    inspecting run metadata (duration, points, size, etc.) before recovery.

    Attributes:
        _UNNAMED_DIR (str): The absolute path to the source directory containing
            unnamed run folders.
        unnamed_runs (List[RunMetadata]): The master list of all discovered
            runs found on disk.
        selected_run (Optional[RunMetadata]): The currently highlighted run in
            the list, if any.
        active_filters (Dict[str, Any]): A dictionary of filtering criteria
            currently applied to the list.
        is_preview_visible (bool): Toggle state for the side preview panel.
        _filter_popup (Optional[FilterPopup]): Reference to the persistent
            floating filter menu.
    """

    _UNNAMED_DIR: str = os.path.join(Architecture.get_path(), Constants.log_export_path, "_unnamed")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initializes the RunRecoveryDialog.

        Sets up the internal state, constructs the UI components, and
        triggers the initial asynchronous scan of the unnamed runs directory.

        Args:
            parent: The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.unnamed_runs: List[RunMetadata] = []
        self.selected_run: Optional[RunMetadata] = None
        self.active_filters: dict[str, Any] = {}
        self.is_preview_visible: bool = True
        self._filter_popup: Optional[QWidget] = None
        self.setup_ui()
        # self.load_unnamed_runs()  # defer to show()

    def setup_ui(self) -> None:
        """
        Initializes and configures the user interface components for the dialog.

        This method sets up the styling, layouts, and widget hierarchies,
        including the search bar, sortable run list, detail view, and a
        PyQtGraph-based preview plot. It also connects UI signals to their
        respective logic handlers.
        """
        self.master_container = QWidget(self)
        self.master_container.setObjectName("masterContainer")
        self.master_container.setStyleSheet(
            "QWidget#masterContainer { background-color: transparent; }"
        )

        main_layout = QVBoxLayout(self.master_container)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # Top layout
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
        self.search_bar.setStyleSheet("""
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
            """)
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
        self.filter_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.filter_btn.setToolTip("Filter Options")
        self.filter_btn.setIcon(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "filter.svg"))
        )
        self.filter_btn.setIconSize(QSize(16, 16))
        self.filter_btn.setStyleSheet(_icon_btn_ss)
        self.filter_btn.setCheckable(True)
        self.filter_btn.clicked.connect(self.show_filter_menu)

        self._rescan_icon_path: str = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "refresh-cw.svg"
        )
        self._rescan_base_pixmap: QPixmap = QIcon(self._rescan_icon_path).pixmap(QSize(16, 16))
        # This tracks a float from 0.0 to 360.0
        self._rescan_animation: QVariantAnimation = QVariantAnimation(self)
        self._rescan_animation.setStartValue(0.0)
        self._rescan_animation.setEndValue(360.0)
        self._rescan_animation.setDuration(850)  # Time for one full spin (ms)
        self._rescan_animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._rescan_animation.valueChanged.connect(self._update_rescan_icon)
        self._rescan_animation.finished.connect(self._check_rescan_loop)
        self._is_rescan_loading: bool = False

        self.rescan_btn: QPushButton = QPushButton()
        self.rescan_btn.setFixedSize(28, 28)
        self.rescan_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.rescan_btn.setToolTip("Rescan for unnamed runs")
        self.rescan_btn.setIcon(QIcon(self._rescan_base_pixmap))
        self.rescan_btn.setIconSize(QSize(16, 16))
        self.rescan_btn.setStyleSheet(_icon_btn_ss)
        self.rescan_btn.clicked.connect(self.load_unnamed_runs)

        search_layout.addWidget(self.search_bar, stretch=1)
        search_layout.addWidget(self.filter_btn)
        search_layout.addWidget(self.rescan_btn)
        top_master_layout.addLayout(search_layout)

        # Details & 'Recover' / 'Delete' buttons
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)

        # Left layout
        left_column = QWidget()
        left_column.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_column.setMinimumWidth(0)
        left_column_layout = QVBoxLayout(left_column)
        left_column_layout.setContentsMargins(0, 0, 0, 0)
        left_column_layout.setSpacing(6)

        # Sort bar
        sort_bar = QWidget()
        sort_bar.setStyleSheet("""
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
            """)
        sort_bar.setObjectName("sortBar")

        sort_bar_layout = QHBoxLayout(sort_bar)
        sort_bar_layout.setContentsMargins(4, 0, 2, 0)
        sort_bar_layout.setSpacing(4)

        sort_label = QLabel("Sort by")
        sort_label.setObjectName("sortLabel")

        self.sort_combo = QComboBox()
        self.sort_combo.setObjectName("sortCombo")
        self.sort_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        for text, key in (
            ("Name", "name"),
            ("Date", "date"),
            ("Duration", "duration"),
            ("Points", "points"),
            ("File Size", "size"),
        ):
            self.sort_combo.addItem(text, key)
        self.sort_combo.setCurrentIndex(1)
        self.sort_combo.currentIndexChanged.connect(lambda *_: self._sort_runs())

        self._sort_ascending = False  # newest first by default
        self.sort_dir_btn = QPushButton()
        self.sort_dir_btn.setIcon(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "descending.svg"))
        )
        self.sort_dir_btn.setObjectName("sortDir")
        self.sort_dir_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sort_dir_btn.setToolTip("Toggle sort direction")
        self.sort_dir_btn.setFixedSize(24, 24)
        self.sort_dir_btn.clicked.connect(self._toggle_sort_direction)

        sort_bar_layout.addWidget(sort_label)
        sort_bar_layout.addWidget(self.sort_combo)
        sort_bar_layout.addWidget(self.sort_dir_btn)
        sort_bar_layout.addStretch(1)

        left_column_layout.addWidget(sort_bar)

        # Left layout
        self.runs_list = ToggleListWidget()
        self.runs_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.runs_list.setStyleSheet("""
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
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 10px;          /* 10px width */
                margin: 0px;
                border-radius: 5px;   /* 5px radius makes a 10px wide bar fully rounded */
            }
            QScrollBar::handle:vertical {
                background: rgba(0, 0, 0, 40); /* Base grey */
                min-height: 20px;
                border-radius: 5px;   /* Fully rounded handle corners */
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(0, 0, 0, 80); /* Darker grey on hover */
            }
            QScrollBar::handle:vertical:pressed {
                background: rgba(0, 0, 0, 120); /* Slightly darker grey on click (no more blue) */
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px; 
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            """)
        self.runs_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.runs_list.itemDoubleClicked.connect(self.on_item_double_clicked)

        self.empty_list_placeholder = QLabel("No recoverable runs")
        self.empty_list_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_list_placeholder.setStyleSheet("""
            QLabel {
                border: 1px dashed rgba(0, 0, 0, 25);
                border-radius: 6px;
                background-color: rgba(255, 255, 255, 120);
                color: #999999;
                font-size: 10pt;
                padding: 20px;
            }
            """)

        self.list_stack = QStackedWidget()
        self.list_stack.addWidget(self.runs_list)  # index 0
        self.list_stack.addWidget(self.empty_list_placeholder)  # index 1
        self.list_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.list_stack.setMinimumWidth(0)

        left_column_layout.addWidget(self.list_stack, stretch=1)
        content_layout.addWidget(left_column, stretch=1)

        # Right layout
        right_container = QWidget()
        right_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_container.setMinimumWidth(0)

        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        self.details_frame = QFrame()
        self.details_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.details_frame.setMinimumWidth(0)
        self.details_frame.setStyleSheet("""
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
            """)
        self.details_frame.setObjectName("detailsFrame")

        details_outer = QVBoxLayout(self.details_frame)
        details_outer.setContentsMargins(12, 10, 12, 8)
        details_outer.setSpacing(8)

        details_form = QFormLayout()
        details_form.setContentsMargins(0, 0, 0, 0)
        details_form.setSpacing(6)
        details_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        details_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        def make_key_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #888888; font-size: 9pt;")
            return lbl

        def make_value_label():
            lbl = QLabel("-")
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

        # Action seperator
        details_sep = QFrame()
        details_sep.setObjectName("detailsSep")
        details_outer.addWidget(details_sep)

        # Action buttons
        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(4)

        self.recover_button = QPushButton("  Recover")
        self.recover_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.recover_button.setEnabled(False)
        self.recover_button.setIcon(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "restore.svg"))
        )
        self.recover_button.setIconSize(QSize(14, 14))
        self.recover_button.setStyleSheet("""
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
            """)
        self.recover_button.clicked.connect(self.on_recover_clicked)

        self.delete_button = QPushButton("  Delete")
        self.delete_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_button.setEnabled(False)
        self.delete_button.setIcon(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "delete.svg"))
        )
        self.delete_button.setIconSize(QSize(14, 14))
        # Reserve enough horizontal room so multi-select counts like "Delete (999)"
        # never get clipped. QPushButton's sizeHint does not always grow when the
        # text is changed at runtime, so we baseline on the widest expected label
        # plus icon and horizontal padding (5px+9px each side from the QSS).
        _del_fm = self.delete_button.fontMetrics()
        _del_text_w = _del_fm.horizontalAdvance("  Delete (999)")
        self.delete_button.setMinimumWidth(_del_text_w + 14 + 18 + 6)
        self.delete_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.delete_button.setStyleSheet("""
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
            """)
        self.delete_button.clicked.connect(self.on_delete_clicked)

        action_row.addWidget(self.recover_button)
        action_row.addStretch(1)
        action_row.addWidget(self.delete_button)

        details_outer.addLayout(action_row)

        right_layout.addWidget(self.details_frame)

        # Plot preview
        self.plot_card = QFrame()
        self.plot_card.setObjectName("plotCard")
        self.plot_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_card.setMinimumHeight(150)
        self.plot_card.setStyleSheet("""
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
            """)

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
        self.plot_widget.hideAxis("left")
        self.plot_widget.hideAxis("right")
        bottom_axis = self.plot_widget.getAxis("bottom")
        bottom_axis.setPen(pg.mkPen(color="#cccccc", width=1))
        bottom_axis.setLabel(text="")
        bottom_axis.setStyle(showValues=False)
        self._FREQ_COLOR = (82, 142, 201)
        self._DISS_COLOR = (225, 175, 85)
        pen_freq = pg.mkPen(color=(*self._FREQ_COLOR, 170), width=1.4)
        pen_diss = pg.mkPen(color=(*self._DISS_COLOR, 170), width=1.4)

        self.curve_freq = self.plot_widget.plot(pen=pen_freq)

        self.view_box_diss = pg.ViewBox()
        scene = self.plot_widget.scene()
        if scene is not None:
            scene.addItem(self.view_box_diss)
        else:
            pass
        self.view_box_diss.setXLink(self.plot_widget)
        self.view_box_diss.setMouseEnabled(x=False, y=False)
        self.view_box_diss.setMenuEnabled(False)

        self.curve_diss = pg.PlotCurveItem(pen=pen_diss)
        self.view_box_diss.addItem(self.curve_diss)

        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None and plot_item.vb is not None:
            main_vb = plot_item.vb

            def _update_views() -> None:
                """Synchronizes the dissipation ViewBox geometry with the main plot ViewBox."""
                rect = main_vb.sceneBoundingRect()
                self.view_box_diss.setGeometry(rect)
                self.view_box_diss.linkedViewChanged(main_vb, self.view_box_diss.XAxis)

            main_vb.sigResized.connect(_update_views)
            _update_views()

        self.empty_plot_placeholder = QLabel("No data to display")
        self.empty_plot_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_plot_placeholder.setStyleSheet("""
            QLabel {
                background-color: #fafafa;
                color: #aaaaaa;
                font-size: 9pt;
                border-radius: 4px;
            }
            """)

        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(self.plot_widget)  # index 0
        self.plot_stack.addWidget(self.empty_plot_placeholder)  # index 1
        self.plot_stack.setCurrentIndex(1)

        plot_card_layout.addWidget(self.plot_stack, stretch=1)
        # Legend
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
        self.plot_legend_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_legend_label.setTextFormat(Qt.TextFormat.RichText)
        plot_card_layout.addWidget(self.plot_legend_label)
        right_layout.addWidget(self.plot_card, stretch=1)
        content_layout.addWidget(right_container, stretch=1)
        top_master_layout.addLayout(content_layout)
        main_layout.addLayout(top_master_layout, stretch=1)

        base_layout = QVBoxLayout(self)
        base_layout.setContentsMargins(0, 0, 0, 0)

        base_layout.addWidget(self.master_container)

    def hideEvent(self, event) -> None:
        """Handles the event triggered when the widget is hidden.

        This method is fired when the tab is switched away from, or the parent
        window is closed. It explicitly hides the master container to remove
        the UI tree from the active rendering pipeline, which prevents graphics
        effects (like shadows and blurs) from bleeding through to other tabs.

        Args:
            event (QHideEvent): The native hide event object provided by Qt.
        """
        super().hideEvent(event)
        if hasattr(self, "master_container"):
            self.master_container.setVisible(False)

    def showEvent(self, event) -> None:
        """Handles the event triggered when the widget becomes visible.

        This method is fired when the tab becomes actively viewed. It restores
        the visibility of the master container. On the initial display, it also
        defers the loading of the unnamed runs to the next event loop cycle to
        ensure the UI layout is fully rendered before applying graphic effects.

        Args:
            event (QShowEvent): The native show event object provided by Qt.
        """
        super().showEvent(event)

        if hasattr(self, "master_container"):
            self.master_container.setVisible(True)

        if not getattr(self, "_initial_load_done", False):
            self._initial_load_done = True
            QTimer.singleShot(0, self.load_unnamed_runs)

    def show_filter_menu(self) -> None:
        """
        Handles the creation, positioning, and animation of the recovery filter popup.

        If a popup is already active, it closes it. Otherwise, it instantiates
        a RecoveryFilter, positions it relative to the filter button, and
        executes a slide-down 'open' animation.

        A short debounce window after a popup closes prevents the same click
        from immediately re-opening the menu (e.g., when the user clicks the
        filter button while the popup is already open and ``Qt.WindowType.Popup``
        is auto-dismissing it).
        """
        if getattr(self, "_filter_popup_just_closed", False):
            self.filter_btn.setChecked(bool(self.active_filters))
            return

        if self._filter_popup is not None:
            with suppress(RuntimeError):
                self._filter_popup.close()
            return

        popup = RecoveryFilterWidget(self, current_filters=self.active_filters)
        popup.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        popup.filters_changed.connect(self._on_filters_changed)

        popup.adjustSize()
        full_height = popup.sizeHint().height()

        btn_right_global = self.filter_btn.mapToGlobal(
            QPoint(self.filter_btn.width(), self.filter_btn.height() + 4)
        )
        right_margin = getattr(popup, "_SHADOW_MARGIN_R", 0)
        top_margin = getattr(popup, "_SHADOW_MARGIN_T", 0)
        pos = QPoint(
            btn_right_global.x() - popup.width() + right_margin,
            btn_right_global.y() - top_margin,
        )
        popup.move(pos)
        self.filter_btn.setChecked(True)

        # Open animation: expand height + fade in, anchored to button
        popup.setMaximumHeight(0)
        popup.setWindowOpacity(0.0)
        popup.show()

        height_anim = QPropertyAnimation(popup, b"maximumHeight")
        height_anim.setDuration(180)
        height_anim.setEasingCurve(QEasingCurve.OutCubic)
        height_anim.setStartValue(0)
        height_anim.setEndValue(full_height)

        fade_anim = QPropertyAnimation(popup, b"windowOpacity")
        fade_anim.setDuration(180)
        fade_anim.setEasingCurve(QEasingCurve.OutCubic)
        fade_anim.setStartValue(0.0)
        fade_anim.setEndValue(1.0)

        open_group = QParallelAnimationGroup(popup)
        open_group.addAnimation(height_anim)
        open_group.addAnimation(fade_anim)
        open_group.finished.connect(lambda: popup.setMaximumHeight(16777215))
        open_group.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def _on_filter_popup_closed(self) -> None:
        """
        Cleans up references when the filter popup is closed or destroyed.

        Resets the internal popup reference to None, syncs the filter button's
        visual checked state with the presence of active filters, and arms a
        short debounce flag so the same click event that dismissed the popup
        cannot immediately re-open it.
        """
        self._filter_popup = None
        self.filter_btn.setChecked(bool(self.active_filters))
        self._filter_popup_just_closed = True
        QTimer.singleShot(150, lambda: setattr(self, "_filter_popup_just_closed", False))

    def _on_filters_changed(self, filters: Dict[str, Any]) -> None:
        """
        Updates the active filter state and refreshes the displayed runs.

        Args:
            filters (Dict[str, Any]): A dictionary containing the filter criteria
                emitted by the RecoveryFilter popup.
        """
        self.active_filters = filters
        self.filter_btn.setChecked(bool(filters))
        self.refilter_list()

    def refilter_list(self, *_: object) -> None:
        """
        Applies search text and active filters to the runs list.

        Iterates through all items in the runs list, evaluating them against the
        current search query and filter criteria (status, date, duration, points,
        and size). Updates item visibility and the UI empty state accordingly.
        """
        query = self.search_bar.text().lower().strip()
        visible_count = 0

        for index in range(self.runs_list.count()):
            item = self.runs_list.item(index)
            if item is None:
                continue
            raw_data = item.data(Qt.ItemDataRole.UserRole)
            run = cast("RunMetadata", raw_data)
            matches = True

            # Search
            if query:
                matches = query in run.display_name.lower() or query in run.ruling.lower()

            # Status
            if matches and "status" in self.active_filters:
                matches = run.ruling == self.active_filters["status"]

            # Date / time range
            if matches and "date_from" in self.active_filters and "date_to" in self.active_filters:
                run_dt = self._parse_timestamp(run.start)
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
                    <= run.duration
                    <= self.active_filters["duration_max"]
                )

            # Points range
            if matches and "points_min" in self.active_filters:
                matches = (
                    self.active_filters["points_min"]
                    <= run.samples
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
    def _parse_timestamp(ts: str) -> Optional[datetime]:
        """
        Best-effort parse of a run timestamp string into a datetime object.

        Args:
            ts (str): The timestamp string to parse.

        Returns:
            Optional[datetime]: The parsed datetime object if successful;
            otherwise, None.
        """
        if not ts or ts == "Unknown":
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M:%S"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
        return None

    def _update_rescan_icon(self, angle: float) -> None:
        """
        Renders and applies a rotated version of the rescan icon to the button.

        This method creates a temporary transparent canvas (QPixmap), uses a QPainter
        to rotate the base pixmap around its center point by the specified angle,
        and updates the button icon. It utilizes smooth pixmap transformation and
        antialiasing for a high-quality visual result.

        Args:
            angle (float): The rotation angle in degrees (0.0 to 360.0).
        """
        size = self._rescan_base_pixmap.size()
        if size.isEmpty():
            return

        canvas = QPixmap(size)
        canvas.fill(Qt.GlobalColor.transparent)

        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Translate to center, rotate, translate back
        cx, cy = size.width() / 2.0, size.height() / 2.0
        painter.translate(cx, cy)
        painter.rotate(angle)
        painter.translate(-cx, -cy)

        painter.drawPixmap(0, 0, self._rescan_base_pixmap)
        painter.end()

        self.rescan_btn.setIcon(QIcon(canvas))

    def _check_rescan_loop(self) -> None:
        """
        Evaluates the loading state at the end of an animation cycle to determine continuity.

        This method is connected to the 'finished' signal of the rescan animation.
        It ensures the spinner only stops after completing a full 360-degree
        revolution, maintaining a professional visual transition. If the scan
        is still in progress, it restarts the animation; otherwise, it resets
        the icon to the upright position and re-enables the button.
        """
        if self._is_rescan_loading:
            self._rescan_animation.start()
        else:
            self._update_rescan_icon(0.0)
            self.rescan_btn.setEnabled(True)

            if hasattr(self, "_blur_effect") and self._blur_effect is not None:
                self._blur_out_anim = QPropertyAnimation(self._blur_effect, b"blurRadius", self)
                self._blur_out_anim.setDuration(250)  # Smooth fade out
                self._blur_out_anim.setStartValue(self._blur_effect.blurRadius())
                self._blur_out_anim.setEndValue(0.0)

                def _cleanup_blur():
                    viewport = self.runs_list.viewport()
                    if viewport is not None:
                        viewport.setGraphicsEffect(None)
                    self._blur_effect = None

                self._blur_out_anim.finished.connect(_cleanup_blur)
                self._blur_out_anim.start()

    def _start_rescan_animation(self) -> None:
        """
        Sets the loading state and initiates the rescan spinner animation.

        This method flags the internal loading state as active, disables the
        rescan button to prevent concurrent scan requests, and starts the
        QVariantAnimation loop. If the animation is already running, it
        allows the current cycle to continue without interruption.
        """
        self._is_rescan_loading = True
        if hasattr(self, "rescan_btn"):
            self.rescan_btn.setEnabled(False)
        if self._rescan_animation.state() != QVariantAnimation.State.Running:
            self._rescan_animation.start()

    def _stop_rescan_animation(self) -> None:
        """
        Signals that the loading process is complete.

        The animation will continue until it finishes its current revolution
        to ensure the icon returns to the upright position smoothly.
        """
        self._is_rescan_loading = False

    def _toggle_sort_direction(self) -> None:
        """
        Inverts the current sort order and updates the direction button's icon and tooltip.

        This method toggles 'self._sort_ascending', swaps the button icon between
        ascending and descending SVG assets, and triggers a re-sort of the runs list.
        """
        self._sort_ascending = not self._sort_ascending
        icon_asc = QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "ascending.svg"))
        icon_desc = QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "descending.svg"))
        new_icon = icon_asc if self._sort_ascending else icon_desc
        self.sort_dir_btn.setIcon(new_icon)
        self.sort_dir_btn.setText("")  # Ensure text is cleared
        self.sort_dir_btn.setToolTip("Ascending" if self._sort_ascending else "Descending")
        self._sort_runs()

    def _sort_runs(self) -> None:
        """
        Re-orders the items in runs_list based on the current sort key and direction.

        This method sorts the backing 'self.unnamed_runs' list to ensure ordering
        persists. It preserves the current user selection, clears and repopulates
        the QListWidget, and then re-applies any active filters.
        """
        if not hasattr(self, "sort_combo"):
            return

        key = self.sort_combo.currentData()
        ascending = self._sort_ascending

        def sort_key(r: RunMetadata):
            """Helper to extract the sortable value from metadata."""
            if key == "name":
                return (r.display_name or "").lower()
            if key == "date":
                dt = self._parse_timestamp(r.start)
                return dt or datetime.min
            if key == "duration":
                return r.duration or 0.0
            if key == "points":
                return r.samples or 0
            if key == "size":
                return r.file_size_mb or 0.0
            return (r.display_name or "").lower()

        try:
            self.unnamed_runs.sort(key=sort_key, reverse=not ascending)
        except TypeError:
            self.unnamed_runs.sort(
                key=lambda r: (r.display_name or "").lower(), reverse=not ascending
            )

        selected_runs = {
            item.data(Qt.ItemDataRole.UserRole) for item in self.runs_list.selectedItems()
        }

        self.runs_list.blockSignals(True)
        self.runs_list.clear()
        for run in self.unnamed_runs:
            item = QListWidgetItem(run.display_name)
            item.setData(Qt.ItemDataRole.UserRole, run)
            self.runs_list.addItem(item)
            if run in selected_runs:
                item.setSelected(True)
        self.runs_list.blockSignals(False)

        self.refilter_list()
        self.on_selection_changed()

    def load_unnamed_runs(self) -> None:
        """
        Initiates an asynchronous scan for recoverable runs.

        Resets the UI state, clears current run metadata, and starts the
        rescan animation. A ScanWorker is initialized to process the
        unnamed directory in a background thread.
        """
        if (
            hasattr(self, "_scan_worker")
            and self._scan_worker is not None
            and self._scan_worker.isRunning()
        ):
            return

        self.clear_details()
        self.recover_button.setEnabled(False)
        self.delete_button.setEnabled(False)

        # Conditionally blur existing items or show placeholder
        viewport = self.runs_list.viewport()
        if viewport is not None and self.isVisible():
            self._blur_effect = QGraphicsBlurEffect(viewport)
            self._blur_effect.setBlurRadius(0.0)
            viewport.setGraphicsEffect(self._blur_effect)

            self._blur_anim = QPropertyAnimation(self._blur_effect, b"blurRadius")
            self._blur_anim.setDuration(300)
            self._blur_anim.setStartValue(0.0)
            self._blur_anim.setEndValue(5.0)
            self._blur_anim.start()

        if self.runs_list.count() == 0:
            self.empty_list_placeholder.setText("Scanning for runs…")
            self.list_stack.setCurrentIndex(1)
            if self.isVisible():
                self._placeholder_blur = QGraphicsBlurEffect(self.empty_list_placeholder)
                self.empty_list_placeholder.setGraphicsEffect(self._placeholder_blur)
                self._placeholder_anim = QPropertyAnimation(self._placeholder_blur, b"blurRadius")
                self._placeholder_anim.setDuration(400)
                self._placeholder_anim.setStartValue(10.0)
                self._placeholder_anim.setEndValue(0.0)
                self._placeholder_anim.finished.connect(
                    lambda: self.empty_list_placeholder.setGraphicsEffect(None)
                )
                self._placeholder_anim.start()

        self._start_rescan_animation()
        self._UNNAMED_DIR = os.path.join(
            Constants.log_prefer_path, "_unnamed"
        )  # from user's preferred path
        self._scan_worker = ScanWorker(self._UNNAMED_DIR)

        # Signals
        self._scan_worker.scan_complete.connect(self._on_scan_complete)
        self._scan_worker.scan_failed.connect(self._on_scan_failed)
        self._scan_worker.finished.connect(self._stop_rescan_animation)
        self._scan_worker.start()

    def _on_scan_complete(self, runs: List["RunMetadata"]) -> None:
        """
        Callback triggered when the background ScanWorker successfully finishes.

        Args:
            runs (List[RunMetadata]): The collection of run metadata objects
                found during the scan.
        """
        self.unnamed_runs = list(runs)
        self._sort_runs()

        if not self.unnamed_runs:
            self.empty_list_placeholder.setText("No recoverable runs")
            self.list_stack.setCurrentIndex(1)
        else:
            self.list_stack.setCurrentIndex(0)

    def _on_scan_failed(self, message: str) -> None:
        """
        Callback triggered when the background ScanWorker encounters an error.

        Logs the failure message, resets the local run data, and ensures the UI
        reflects an empty state with an appropriate message.

        Args:
            message (str): The error description emitted by the worker.
        """
        Log.e(TAG, f"Error loading unnamed runs: {message}")
        self.unnamed_runs = []
        self.runs_list.clear()
        self.empty_list_placeholder.setText("No recoverable runs")
        self.list_stack.setCurrentIndex(1)

    def update_plot_preview(self, run: "RunMetadata") -> None:
        """
        Parses a virtual CSV file to update the frequency and dissipation plot preview.

        This method reads data from a virtual path, identifies header indices,
        downsamples the dataset if it exceeds the point threshold, applies a
        smoothing filter, and updates the pyqtgraph curves. It handles encrypted
        files and malformed data gracefully.

        Args:
            run (RunMetadata): The metadata object containing the CSV path.
        """
        if not run or not run.virtual_csv_path:
            self.curve_freq.setData([], [])
            self.curve_diss.setData([], [])
            self.plot_stack.setCurrentIndex(1)  # show placeholder
            return

        times, freqs, disss = [], [], []

        try:
            with secure_open(run.virtual_csv_path, "r") as f:
                binary_f = cast(BinaryIO, f)

                text_io = io.TextIOWrapper(binary_f, encoding="utf-8-sig")
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

                    if (
                        t_idx != -1
                        and f_idx != -1
                        and d_idx != -1
                        and len(row) > max(t_idx, f_idx, d_idx)
                    ):
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

            # Smoothing for a clearner plot
            freqs_s = self._smooth(freqs)
            disss_s = self._smooth(disss)

            self.curve_freq.setData(times, freqs_s)
            self.curve_diss.setData(times, disss_s)
            plot_item = self.plot_widget.plotItem
            if plot_item is not None and plot_item.vb is not None:
                plot_item.vb.autoRange()
            if self.view_box_diss is not None:
                self.view_box_diss.autoRange()
            self.view_box_diss.autoRange()
            self.plot_stack.setCurrentIndex(0)

        except RuntimeError as e:
            # Bad-password errors are expected for encrypted ZIPs show a placeholder.
            if "Bad password" in str(e) or "password" in str(e).lower():
                self.empty_plot_placeholder.setText("Preview unavailable")
            else:
                Log.e(TAG, f"Failed to load preview data: {str(e)}")
            self.curve_freq.setData([], [])
            self.curve_diss.setData([], [])
            self.plot_stack.setCurrentIndex(1)
        except Exception as e:
            Log.e(TAG, f"Failed to load preview data: {str(e)}")
            self.curve_freq.setData([], [])
            self.curve_diss.setData([], [])
            self.plot_stack.setCurrentIndex(1)

    @staticmethod
    def _smooth(values: Sequence[float], window: int = 9) -> List[float]:
        """
        Applies a centered moving-average smoother to a sequence of values.

        Uses edge-padding to preserve endpoints, preventing the smoothed signal
        from tapering toward zero at the boundaries.

        Args:
            values (Sequence[float]): The input numerical data to smooth.
            window (int): The size of the smoothing window (default is 9).
                Must be at least 3.

        Returns:
            List[float]: The smoothed data as a list of floats.
        """
        n = len(values)
        if n < max(window, 3):
            return list(values)

        arr = np.asarray(values, dtype=float)
        # Pad using edge values so convolution doesn't dip at the boundaries
        pad = window // 2
        padded = np.concatenate([np.full(pad, arr[0]), arr, np.full(pad, arr[-1])])
        kernel = np.ones(window) / window
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed.tolist()

    def on_selection_changed(self) -> None:
        """
        Updates the detail panel and button states based on the current selection.

        Handles three states:
        1. Single selection: Display full metadata and plot preview.
        2. Multiple selection: Clear details, enable bulk delete, disable recovery.
        3. No selection: Clear details and disable all actions.
        """
        selected_items = self.runs_list.selectedItems()
        count = len(selected_items)

        if count == 1:
            run = selected_items[0].data(Qt.ItemDataRole.UserRole)
            self.selected_run = run

            self.detail_datetime.setText(run.start)
            self.detail_duration.setText(f"{run.duration} seconds")
            self.detail_points.setText(f"{run.samples:,}")

            ruling_color = "#2c8a3d" if run.ruling == "Good" else "#c12c3b"
            self.detail_ruling.setStyleSheet(f"color: {ruling_color}; font-size: 9pt;")
            self.detail_ruling.setText(run.ruling)

            self.detail_filesize.setText(f"{run.file_size_mb} MB")

            self.update_plot_preview(run)

            self.recover_button.setEnabled(True)
            self.delete_button.setEnabled(True)
            self.delete_button.setText("  Delete")

        elif count > 1:
            self.selected_run = None
            self.clear_details()

            self.recover_button.setEnabled(False)
            self.delete_button.setEnabled(True)
            self.delete_button.setText(f"  Delete ({count})")

        else:
            self.selected_run = None
            self.clear_details()
            self.recover_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            self.delete_button.setText("  Delete")

    def on_delete_clicked(self) -> None:
        """
        Handles moving selected items to the system Recycle Bin after user confirmation.

        If successful, the items are passed to the animation handler for removal
        from the UI. Errors (like missing folders or permission issues) are logged
        without interrupting the bulk process.
        """
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
            msg = f"Are you sure you want to move {count} selected runs to the Recycle Bin?"
        msg_box = QMessageBox(None)
        msg_box.setWindowTitle("Confirm Deletion")
        msg_box.setText(msg)

        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowIcon(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "delete.svg"))
        )

        # Configure buttons
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)

        reply = msg_box.exec_()

        if reply != QMessageBox.Yes:
            return

        success_count = 0
        runs_to_remove = []

        for item in selected_items:
            run = item.data(Qt.ItemDataRole.UserRole)
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

        if success_count > 0:
            Log.i(TAG, f"Successfully moved {success_count} runs to the Recycle Bin.")

        if runs_to_remove:
            self._animate_delete_items(runs_to_remove)
        else:
            self.on_selection_changed()
            self.refilter_list()

    def _animate_delete_items(self, runs_to_remove: List[Tuple[QListWidgetItem, Any]]) -> None:
        """Animates the removal of list items by turning them red, then collapsing vertically.

        The process follows three stages:
        1. Update stylesheet to highlight selected items in red.
        2. Capture pixmaps of items and place them in QLabel overlays.
        3. Animate the height of the items to zero while fading out the overlays.

        Args:
            runs_to_remove (List[Tuple[QListWidgetItem, Any]]): A list of tuples containing
                the QListWidgetItem to animate and its associated data object.
        """
        items = [item for item, _ in runs_to_remove]

        original_stylesheet = self.runs_list.styleSheet()
        red_stylesheet = original_stylesheet + """
            QListWidget::item:selected {
                background-color: rgba(220, 53, 69, 55);
                color: #b02a38;
            }
        """
        self.runs_list.setStyleSheet(red_stylesheet)
        self.runs_list.repaint()

        def do_collapse() -> None:
            """Inner function to handle the vertical collapse and overlay logic."""
            overlays = []
            orig_rects = {}
            for item in items:
                rect = self.runs_list.visualItemRect(item)
                orig_rects[id(item)] = rect

                if rect.height() > 0:
                    viewport = self.runs_list.viewport()
                    if viewport is not None:
                        pixmap = viewport.grab(rect)
                        overlay = QLabel(viewport)
                        overlay.setPixmap(pixmap)
                        overlay.setScaledContents(True)
                        overlay.setGeometry(rect)

                        opacity = QGraphicsOpacityEffect(overlay)
                        opacity.setOpacity(1.0)
                        overlay.setGraphicsEffect(opacity)
                        overlay.show()
                        overlays.append((item, overlay, opacity))

                item.setBackground(QBrush(Qt.GlobalColor.transparent))
                item.setForeground(QBrush(Qt.GlobalColor.transparent))

            anim = QVariantAnimation(self)
            anim.setDuration(220)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.InCubic)

            def on_frame(progress: float) -> None:
                """Updates item heights and overlay positions for each frame.

                Args:
                    progress (float): Animation progress from 0.0 to 1.0.
                """
                remaining = 1.0 - progress
                for item in items:
                    orig_rect = orig_rects.get(id(item))
                    if orig_rect:
                        h = max(0, int(orig_rect.height() * remaining))
                        item.setSizeHint(QSize(-1, h))

                self.runs_list.doItemsLayout()
                for item, overlay, opacity in overlays:
                    orig_rect = orig_rects[id(item)]
                    current_rect = self.runs_list.visualItemRect(item)

                    w = orig_rect.width()
                    h = max(0, int(orig_rect.height() * remaining))
                    x = current_rect.x()
                    y = current_rect.y()

                    overlay.setGeometry(x, y, w, h)
                    opacity.setOpacity(max(0.0, remaining))

            def on_finished() -> None:
                """Clean up overlays and permanently remove items from the list."""
                self.runs_list.setStyleSheet(original_stylesheet)

                for _, overlay, _ in overlays:
                    overlay.deleteLater()
                for item, run in runs_to_remove:
                    row = self.runs_list.row(item)
                    if row >= 0:
                        self.runs_list.takeItem(row)
                    if run in self.unnamed_runs:
                        self.unnamed_runs.remove(run)

                self.on_selection_changed()
                self.refilter_list()

            anim.valueChanged.connect(on_frame)
            anim.finished.connect(on_finished)
            anim.start()
            self._delete_anim = anim

        QTimer.singleShot(180, do_collapse)

    def on_item_double_clicked(self, item: QListWidgetItem) -> None:
        """
        Handles the double-click event on a list item to initiate recovery.

        This convenience method allows users to skip the 'Recover' button by
        directly double-clicking a run in the list. It updates the currently
        selected run and triggers the recovery workflow.

        Args:
            item (QListWidgetItem): The list item that was double-clicked.
        """
        run = item.data(Qt.ItemDataRole.UserRole)
        self.selected_run = run
        self.on_recover_clicked()

    def clear_details(self) -> None:
        """
        Resets the metadata detail panel and plot preview to an empty state.

        This method clears all text labels with a placeholder dash, resets the
        ruling color to a neutral gray, and flushes the plot data. It also
        switches the plot stack to show the 'no data' placeholder.
        """
        placeholder: str = ""
        self.detail_datetime.setText(placeholder)
        self.detail_duration.setText(placeholder)
        self.detail_points.setText(placeholder)
        self.detail_filesize.setText(placeholder)
        self.detail_ruling.setText(placeholder)
        self.detail_ruling.setStyleSheet("color: #333333; font-size: 9pt;")

        if hasattr(self, "curve_freq") and hasattr(self, "curve_diss"):
            self.curve_freq.setData([], [])
            self.curve_diss.setData([], [])

        if hasattr(self, "plot_stack"):
            self.plot_stack.setCurrentIndex(1)

    def on_recover_clicked(self) -> None:
        selected_items = self.runs_list.selectedItems()
        if not selected_items:
            return
        run_to_recover = selected_items[0].data(Qt.ItemDataRole.UserRole)
        devices = []
        try:
            config_dir = os.path.join(Constants.local_app_data_path, "config")

            if os.path.exists(config_dir):
                # List all items in the directory, but only keep them if they are folders
                devices = [
                    folder_name
                    for folder_name in os.listdir(config_dir)
                    if os.path.isdir(os.path.join(config_dir, folder_name))
                ]
            else:
                Log.w(TAG, f"Device config directory not found at: {config_dir}")

        except Exception as e:
            Log.e(TAG, f"Failed to load devices from config path: {str(e)}")

        dialog = RecoveryDialog(run_to_recover, devices, parent=self)

        if dialog.exec_() == QDialog.Accepted:
            Log.i(
                TAG,
                f"Successfully recovered run to {dialog.device_combo.currentText()}",
            )
            row = self.runs_list.row(selected_items[0])
            self.runs_list.takeItem(row)

            if run_to_recover in self.unnamed_runs:
                self.unnamed_runs.remove(run_to_recover)

            self.on_selection_changed()
            self.refilter_list()

    def _validate_run_name(self, name: str) -> bool:
        """
        Validates that a proposed run name is safe for the file system.

        Checks for empty or whitespace-only strings and ensures the name does
        not contain reserved characters that are prohibited by Windows (NTFS)
        or Unix-based file systems.

        Args:
            name (str): The proposed folder/file name.

        Returns:
            bool: True if the name is valid/safe, False otherwise.
        """
        clean_name: str = name.strip()
        if not clean_name:
            return False

        invalid_chars: Set[str] = {"/", "\\", ":", "*", "?", '"', "<", ">", "|"}
        if any(char in name for char in invalid_chars):
            return False

        return True
