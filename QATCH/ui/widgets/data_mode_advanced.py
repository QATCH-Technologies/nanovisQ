"""
QATCH.ui.widgets.data_mode_advanced_widget.py

Advanced data-management controls and supporting widgets.

Provides the :class:`AdvancedMode` data-management view for managing
removable USB storage, monitoring local storage usage, and erasing locally
stored run data.

The module also defines the private :class:`_UsageBar` widget, which displays
local storage utilization as a compact progress indicator and updates its
appearance when the application theme changes.

USB detection, ejection, and local data operations are coordinated through
the shared :class:`DataServices` instance provided by the parent data
management container.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-19
"""

import os
import shutil
import subprocess
import time

from optree.ops import NONE_IS_LEAF
from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants
from QATCH.ui.components import QATCHPanel, QATCHPushButton
from QATCH.ui.components.icon_utils import tinted_icon
from QATCH.ui.dialogs.pop_up_dialog import PopUp
from QATCH.ui.styles.theme_manager import (
    ThemeManager,
    caption_label_qss,
    desc_label_qss,
    tok_css,
)
from QATCH.ui.widgets.data_mode_base import DataModeWidget

TAG = "[DataAdvanced]"

try:
    import send2trash
except Exception as e:  # pragma: no cover - optional dependency
    Log.e(TAG, f"Send to trash is not available: {e}")
    send2trash = None


class _UsageBar(QtWidgets.QWidget):
    """Compact progress bar for displaying resource usage.

    Renders a rounded track with a colored fill whose width represents the
    current usage fraction. The bar updates its appearance when the active
    theme changes.

    Attributes:
        _fraction (float): Current usage fraction, clamped to the range
            `0.0` to `1.0`.
    """

    def __init__(self, parent=None) -> None:
        """Initialize the usage bar.

        Args:
            parent (QtWidgets.QWidget, optional): Parent widget. Defaults to
                None.
        """
        super().__init__(parent)
        self._fraction = 0.0
        self.setFixedHeight(8)
        self.setMinimumWidth(80)
        ThemeManager.instance().themeChanged.connect(lambda _: self.update())

    def set_fraction(self, fraction) -> None:
        """Set the current usage fraction.

        The supplied value is clamped to the inclusive range from `0.0` to
        `1.0` before the bar is redrawn.

        Args:
            fraction (float): Usage fraction, where `0.0` represents no
                usage and `1.0` represents full usage.
        """
        self._fraction = max(0.0, min(1.0, fraction))
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the usage track and current usage fill.

        Draws a rounded neutral track followed by a colored rounded fill whose
        width corresponds to the current usage fraction.

        Args:
            event (QtGui.QPaintEvent): Paint event requesting the widget to
                redraw.
        """
        tok = ThemeManager.instance().tokens()
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)

        rect = self.rect()
        radius = rect.height() / 2.0
        painter.setBrush(QtGui.QColor(*tok["flat_track"]))
        painter.drawRoundedRect(rect, radius, radius)

        fill_w = rect.width() * self._fraction
        if fill_w > 0:
            fill_rect = QtCore.QRectF(rect.x(), rect.y(), fill_w, rect.height())
            painter.setBrush(QtGui.QColor(*tok["flat_accent"]))
            painter.drawRoundedRect(fill_rect, radius, radius)
        painter.end()


class AdvancedMode(DataModeWidget):
    """Data-management mode for removable drives and local storage.

    Provides controls for detecting and safely ejecting removable USB
    storage, displays local storage usage, and provides a destructive action
    for erasing locally logged run data.

    Attributes:
        MODE_KEY (str): Unique key identifying this mode.
        MODE_LABEL (str): Display label used by the mode navigation.
        _exported (bool): Whether an export has occurred during the current
            session.
        usb_status_pill (QtWidgets.QLabel): Displays the current USB drive
            status.
        usb_desc (QtWidgets.QLabel): Displays descriptive USB drive
            information.
        btn_detect (QATCHPushButton): Button used to re-detect connected USB
            storage.
        btn_eject (QATCHPushButton): Button used to safely eject the active
            USB drive.
        storage_summary_label (QtWidgets.QLabel): Displays a summary of local
            storage usage.
        storage_bar (_UsageBar): Displays local storage usage as a fraction.
        storage_legend_label (QtWidgets.QLabel): Describes the storage usage
            shown by the usage bar.
        btn_erase (QATCHPushButton): Button used to initiate local data
            deletion.
        status_label (QtWidgets.QLabel): Displays status or progress messages
            for operations performed by the mode.
    """

    MODE_KEY = "advanced"
    MODE_LABEL = "Advanced"

    def build(self) -> NONE_IS_LEAF:
        """Build the advanced data-management interface.

        Creates the interface for USB drive management, local storage
        monitoring, and local data deletion. Configures the associated
        controls, signal connections, status displays, and theme handling.

        The method also connects USB device events from the shared services
        object so the displayed USB status remains synchronized with device
        changes.
        """
        self._exported = False

        heading = QtWidgets.QLabel("Advanced")
        self._heading = heading
        subtitle = QtWidgets.QLabel("Manage removable drives and local storage.")
        subtitle.setWordWrap(True)
        self._subtitle = subtitle
        self.root.addWidget(heading)
        self.root.addWidget(subtitle)

        # USB Drive card
        usb_card = QATCHPanel()
        ulay = QtWidgets.QVBoxLayout(usb_card)
        ulay.setContentsMargins(14, 12, 14, 12)
        ulay.setSpacing(8)

        usb_head = QtWidgets.QHBoxLayout()
        usb_head.setContentsMargins(0, 0, 0, 0)
        usb_head.setSpacing(10)
        self._usb_swatch_lbl = QtWidgets.QLabel()
        self._usb_swatch_lbl.setFixedSize(32, 32)
        self._usb_swatch_lbl.setAlignment(QtCore.Qt.AlignCenter)
        usb_head.addWidget(self._usb_swatch_lbl)
        usb_title = QtWidgets.QLabel("USB Drive")
        self._usb_title = usb_title
        usb_head.addWidget(usb_title)
        usb_head.addStretch(1)
        self.usb_status_pill = QtWidgets.QLabel()
        usb_head.addWidget(self.usb_status_pill)
        ulay.addLayout(usb_head)

        self.usb_desc = QtWidgets.QLabel()
        self.usb_desc.setWordWrap(True)
        ulay.addWidget(self.usb_desc)

        usb_row = QtWidgets.QHBoxLayout()
        usb_row.setContentsMargins(0, 0, 0, 0)
        usb_row.setSpacing(8)
        self.btn_detect = QATCHPushButton(" Re-detect", variant="ghost")
        self.btn_detect.setFixedHeight(30)
        self.btn_detect.setIcon(self._icon("refresh-cw.svg"))
        self.btn_detect.clicked.connect(self._do_detect)
        self.btn_eject = QATCHPushButton(" Eject safely", variant="ghost")
        self.btn_eject.setFixedHeight(30)
        self.btn_eject.clicked.connect(self._do_eject)
        usb_row.addWidget(self.btn_detect)
        usb_row.addWidget(self.btn_eject)
        usb_row.addStretch(1)
        ulay.addLayout(usb_row)

        # Local storage card
        storage_card = QATCHPanel()
        slay = QtWidgets.QVBoxLayout(storage_card)
        slay.setContentsMargins(14, 12, 14, 12)
        slay.setSpacing(8)

        storage_head = QtWidgets.QHBoxLayout()
        storage_head.setContentsMargins(0, 0, 0, 0)
        storage_title = QtWidgets.QLabel("Local storage")
        self._storage_title = storage_title
        storage_head.addWidget(storage_title)
        storage_head.addStretch(1)
        self.storage_summary_label = QtWidgets.QLabel()
        self.storage_summary_label.setStyleSheet(caption_label_qss())
        storage_head.addWidget(self.storage_summary_label)
        slay.addLayout(storage_head)

        self.storage_bar = _UsageBar()
        slay.addWidget(self.storage_bar)

        legend_row = QtWidgets.QHBoxLayout()
        legend_row.setContentsMargins(0, 2, 0, 0)
        legend_row.setSpacing(6)
        self._legend_swatch_lbl = QtWidgets.QLabel()
        self._legend_swatch_lbl.setFixedSize(9, 9)
        legend_row.addWidget(self._legend_swatch_lbl)
        self.storage_legend_label = QtWidgets.QLabel()
        legend_row.addWidget(self.storage_legend_label)
        legend_row.addStretch(1)
        slay.addLayout(legend_row)
        self._legend_label = self.storage_legend_label

        # Danger zone
        danger_caption = QtWidgets.QLabel("DANGER ZONE")
        self._danger_caption = danger_caption

        danger_card = QATCHPanel(danger=True)
        dlay = QtWidgets.QVBoxLayout(danger_card)
        dlay.setContentsMargins(14, 12, 14, 12)
        dlay.setSpacing(8)
        drow = QtWidgets.QHBoxLayout()
        drow.setContentsMargins(0, 0, 0, 0)
        drow.setSpacing(10)
        self._danger_swatch_lbl = QtWidgets.QLabel()
        self._danger_swatch_lbl.setFixedSize(32, 32)
        self._danger_swatch_lbl.setAlignment(QtCore.Qt.AlignCenter)
        drow.addWidget(self._danger_swatch_lbl, 0, QtCore.Qt.AlignTop)
        dtext = QtWidgets.QVBoxLayout()
        dtext.setContentsMargins(0, 0, 0, 0)
        dtext.setSpacing(2)
        dtitle = QtWidgets.QLabel("Erase local data")
        self._dtitle = dtitle
        ddesc = QtWidgets.QLabel(
            "Removes all locally logged runs from this machine. Erased runs go to the "
            "Recycle Bin first - empty it afterward to erase permanently."
        )
        ddesc.setWordWrap(True)
        self._ddesc = ddesc
        dtext.addWidget(dtitle)
        dtext.addWidget(ddesc)
        drow.addLayout(dtext, 1)
        self.btn_erase = QATCHPushButton(" Erase…", variant="danger")
        self.btn_erase.setFixedHeight(34)
        self.btn_erase.clicked.connect(self._do_erase)
        drow.addWidget(self.btn_erase, 0, QtCore.Qt.AlignVCenter)
        dlay.addLayout(drow)

        # Progress display
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_label.setVisible(False)

        self.root.addWidget(usb_card)
        self.root.addWidget(storage_card)
        self.root.addWidget(danger_caption)
        self.root.addWidget(danger_card)
        self.root.addWidget(self.status_label)
        self.root.addStretch(1)

        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

        # Live USB status
        self.services.usb_add.connect(self._refresh_usb_status)
        self.services.usb_remove.connect(self._refresh_usb_status)

    def _on_theme_changed(self, _mode: str) -> None:
        """Handle a theme change by refreshing themed UI elements and USB status.

        Args:
            _mode (str): Identifier for the newly activated theme mode. The value
                is not used directly because the current theme is retrieved from
                the theme manager.
        """
        self._apply_theme()
        self._refresh_usb_status()

    def _apply_theme(self) -> None:
        """Apply the active theme to the advanced data-management interface.

        Updates text styles, status indicators, icons, backgrounds, borders, and
        other theme-dependent visual properties using the current theme tokens.
        """
        tok = ThemeManager.instance().tokens()
        accent = tok["flat_accent"]
        danger = tok["flat_error"]
        heading_qss = (
            f"QLabel {{ color: {tok_css(tok['flat_text'])}; font-size: 14px; "
            "font-weight: bold; background: transparent; }"
        )
        section_title_qss = (
            f"QLabel {{ color: {tok_css(tok['flat_text'])}; font-size: 13px; "
            "font-weight: 700; background: transparent; }"
        )
        self._heading.setStyleSheet(heading_qss)
        self._subtitle.setStyleSheet(desc_label_qss())
        self._usb_title.setStyleSheet(section_title_qss)
        self.usb_desc.setStyleSheet(desc_label_qss())
        self._storage_title.setStyleSheet(section_title_qss)
        self.storage_summary_label.setStyleSheet(caption_label_qss())
        self._legend_label.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_text_muted'])}; font-size: 11px; "
            "background: transparent; }"
        )
        self._danger_caption.setStyleSheet(
            f"QLabel {{ color: {tok_css(danger)}; font-size: 10px; font-weight: 700; "
            "text-transform: uppercase; letter-spacing: 0.5px; background: transparent; }"
        )
        self._dtitle.setStyleSheet(
            f"QLabel {{ color: {tok_css(danger)}; font-size: 13px; font-weight: 700; "
            "background: transparent; }"
        )
        self._ddesc.setStyleSheet(
            f"QLabel {{ color: {tok_css((danger[0], danger[1], danger[2], 200))}; "
            "font-size: 11px; background: transparent; }"
        )
        self.status_label.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_text'])}; font-size: 12px; "
            f"background: {tok_css(tok['flat_surface2'])}; "
            f"border: 1px solid {tok_css(tok['flat_border'])}; "
            "border-radius: 8px; padding: 8px; }"
        )

        usb_icon = tinted_icon(self._icon_file_path("usb.svg"), QtGui.QColor(*accent), 18)
        self._usb_swatch_lbl.setPixmap(usb_icon.pixmap(18, 18))
        self._usb_swatch_lbl.setStyleSheet(
            f"QLabel {{ background: {tok_css((accent[0], accent[1], accent[2], 45))}; "
            "border-radius: 8px; }"
        )
        danger_icon = tinted_icon(self._icon_file_path("warning.svg"), QtGui.QColor(*danger), 18)
        self._danger_swatch_lbl.setPixmap(danger_icon.pixmap(18, 18))
        self._danger_swatch_lbl.setStyleSheet(
            f"QLabel {{ background: {tok_css((danger[0], danger[1], danger[2], 40))}; "
            "border-radius: 8px; }"
        )
        self._legend_swatch_lbl.setStyleSheet(
            f"QLabel {{ background: {tok_css(accent)}; border-radius: 3px; }}"
        )

    @staticmethod
    def _pill_qss(connected: bool) -> str:
        """Build the style sheet for the USB connection status indicator.

        Args:
            connected (bool): Whether a USB drive is currently connected.

        Returns:
            str: Style sheet configured for the connected or disconnected state.
        """
        tok = ThemeManager.instance().tokens()
        if connected:
            text = tok["flat_success"]
            weak = tok["flat_success_weak"]
            ring = tok["flat_success_ring"]
        else:
            text, weak, ring = tok["flat_text_muted"], tok["flat_surface2"], tok["flat_border"]
        return (
            f"QLabel {{ color: {tok_css(text)}; background: {tok_css(weak)}; "
            f"border: 1px solid {tok_css(ring)}; border-radius: 9px; "
            "font-size: 10px; font-weight: 700; padding: 2px 8px; }"
        )

    def on_enter(self) -> None:
        """Refresh USB and local storage information when entering the mode."""
        self._refresh_usb_status()
        self._refresh_storage()

    def on_freeze(self, frozen: bool) -> None:
        """Enable or disable controls while the mode is frozen.

        The erase control remains enabled when the mode is frozen, allowing
        destructive operations to remain available.

        Args:
            frozen (bool): Whether the mode should be placed in its frozen state.
        """

        self.btn_detect.setDisabled(frozen)
        self.btn_eject.setDisabled(frozen)

    def on_progress(self, label, pct, color) -> None:
        """Update the operation status and refresh data when an operation ends.

        Displays the supplied status message when available. When progress
        reaches 100 percent, both USB and local storage information are refreshed
        because either state may have changed during the operation.

        Args:
            label (str): Status message to display.
            pct (int | float): Operation completion percentage.
            color: Color associated with the current progress state. The value is
                accepted for interface compatibility but is not used directly.
        """

        if label:
            self.status_label.setVisible(True)
            self.status_label.setText(label)
        if pct == 100:
            # Erase/eject finished (success, cancel, or error)
            self._refresh_usb_status()
            self._refresh_storage()

    def note_exported(self, exported: bool = True):
        """Record whether data has been exported during the current session.

        The stored state is used to determine the wording or behavior of the
        local data deletion prompt.

        Args:
            exported (bool, optional): Whether an export has occurred during the
                current session. Defaults to True.
        """
        self._exported = exported

    def _refresh_usb_status(self, *_) -> None:
        """Refresh the displayed USB drive status.

        Updates the connection indicator, tooltip, descriptive text, and eject
        control based on the currently detected USB drive. When a drive is
        connected, its available and total capacity are also displayed when
        readable.

        Args:
            *_: Ignored signal arguments.
        """
        drive = getattr(self.services, "usb_drive", None)
        if drive:
            self.usb_status_pill.setText("●  Connected")
            self.usb_status_pill.setToolTip(drive)
            self.usb_status_pill.setStyleSheet(self._pill_qss(connected=True))
            try:
                usage = shutil.disk_usage(drive)
                self.usb_desc.setText(
                    f"{self._fmt_size(usage.free)} free of {self._fmt_size(usage.total)} "
                    f"on {drive}. Safely eject before unplugging."
                )
            except OSError:
                self.usb_desc.setText(f"{drive} detected, but its capacity can't be read.")
            self.btn_eject.setEnabled(True)
        else:
            self.usb_status_pill.setText("Not connected")
            self.usb_status_pill.setToolTip("")
            self.usb_status_pill.setStyleSheet(self._pill_qss(connected=False))
            self.usb_desc.setText("No USB drive connected. Plug one in, then Re-detect.")
            self.btn_eject.setEnabled(False)

    def _refresh_storage(self) -> None:
        """Refresh the displayed local storage usage.

        Recalculates the number of logged runs and their total disk usage,
        updates the associated summary and legend labels, and adjusts the usage
        bar based on the total capacity of the configured logging volume.
        """

        run_count, total_bytes = self._compute_storage_stats()
        size_txt = self._fmt_size(total_bytes)
        self.storage_summary_label.setText(f"{run_count} runs · {size_txt} logged")
        self.storage_legend_label.setText(f"Active runs {size_txt}")

        fraction = 0.0
        try:
            disk_total = shutil.disk_usage(os.path.abspath(Constants.log_prefer_path)).total
            if disk_total > 0:
                fraction = total_bytes / disk_total
        except OSError:
            pass
        self.storage_bar.set_fraction(fraction)

    @staticmethod
    def _compute_storage_stats() -> tuple[int, int]:
        """Calculate statistics for locally logged run data.

        Traverses the configured logging directory and counts each run directory
        while summing the sizes of files contained directly within those
        directories.

        Returns:
            tuple[int, int]: A tuple containing the number of runs and total file
                size in bytes. Returns `(0, 0)` when the logging directory
                cannot be accessed.
        """
        data_path = Constants.log_prefer_path
        run_count = 0
        total_bytes = 0
        try:
            devices = os.listdir(data_path)
        except OSError:
            return 0, 0
        for device in devices:
            device_path = os.path.join(data_path, device)
            if not os.path.isdir(device_path):
                continue
            try:
                runs = os.listdir(device_path)
            except OSError:
                continue
            for run in runs:
                run_path = os.path.join(device_path, run)
                if not os.path.isdir(run_path):
                    continue
                run_count += 1
                try:
                    files = os.listdir(run_path)
                except OSError:
                    continue
                for f in files:
                    fp = os.path.join(run_path, f)
                    try:
                        if os.path.isfile(fp):
                            total_bytes += os.path.getsize(fp)
                    except OSError:
                        continue
        return run_count, total_bytes

    @staticmethod
    def _fmt_size(num_bytes) -> str:
        """Format a byte count as a human-readable storage size.

        Converts the supplied byte count to the largest appropriate binary unit,
        up to terabytes.

        Args:
            num_bytes (int | float): Number of bytes to format.

        Returns:
            str: Formatted size using `B`, `KB`, `MB`, `GB`, or `TB`.
        """
        size = float(num_bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024.0 or unit == "TB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def _do_detect(self) -> None:
        """Request USB drive detection and refresh the displayed status.

        Uses the shared data services to trigger drive enumeration when the
        detection callback is available. Falls back to logging the request when
        the shared service does not provide a detection trigger.

        The displayed USB status is refreshed immediately after the detection
        request.
        """
        trigger = getattr(self.services, "request_detect", None)
        if callable(trigger):
            trigger()
        else:
            Log.d(f"{TAG} detect requested (shared loop handles enumeration)")
        self._refresh_usb_status()

    def _do_eject(self) -> None:
        """Start the USB drive ejection task.

        Submits the ejection operation to the shared service task runner so the
        potentially blocking device operation does not execute directly on the
        UI thread.
        """
        self.services.run_task(self._eject_task)

    def _eject_task(self, abort) -> None:
        """Safely eject the currently connected USB drive.

        Requests the operating system to eject the detected drive, then verifies
        that the drive is no longer accessible before reporting success. The
        operation supports cancellation and emits progress updates throughout
        the process.

        Args:
            abort (threading.Event): Event used to signal cancellation of the
                ejection task.

        NOTE: Only windows machines are supported!
        """
        self.services.set_freeze(False)
        drive = getattr(self.services, "usb_drive", None)
        if not drive:
            self.services.set_freeze(True)
            return

        try:
            Log.i(TAG, f"[{drive}] USB drive ejecting...")
            self.services.emit_progress(
                self.MODE_KEY, f"[{drive}] USB drive ejecting... please wait...", 33, "b"
            )
            time.sleep(1)

            if abort.is_set():
                self.services.emit_progress(
                    self.MODE_KEY,
                    f"[{drive}] USB drive eject: Operation cancelled.",
                    0,
                    "b",
                )
                Log.w(f"{TAG} Eject task aborted prematurely!")
                return
            subprocess.call(
                "powershell $driveEject = New-Object -comObject Shell.Application; "
                '$driveEject.Namespace(17).ParseName("""{}""").InvokeVerb("""Eject""")'.format(
                    drive
                ),
                shell=True,
            )

            timeout_seconds = 5.0
            elapsed = 0.0
            eject_successful = False

            while elapsed < timeout_seconds:
                if getattr(self.services, "usb_drive", None) != drive:
                    eject_successful = True
                    break
                try:
                    os.stat(drive)
                except OSError:
                    eject_successful = True
                    self.services.usb_drive = None
                    break

                time.sleep(0.5)
                elapsed += 0.5

            if eject_successful:
                Log.i(TAG, "USB drive ejected.")
                self.services.emit_progress(
                    self.MODE_KEY, "USB drive ejected. Safe to remove.", 100, "b"
                )
            else:
                Log.e(TAG, f"[{drive}] USB drive eject failed!")
                self.services.emit_progress(
                    self.MODE_KEY, f"[{drive}] USB drive eject failed! Try again.", 66, "r"
                )

        except Exception as e:
            Log.e(TAG, f"Eject error: {e}")
            self.services.emit_progress(self.MODE_KEY, "Error ejecting USB drive!", 100, "r")

        self.services.set_freeze(True)

    def _do_erase(self) -> None:
        """Confirm and start the local data erasure task.

        Prompts the user for confirmation before deleting locally stored run
        data. The confirmation message varies depending on whether the user has
        exported data during the current session.

        If the operation is confirmed, the erasure task is submitted to the
        shared service runner. If the user declines, the operation is cancelled.
        """
        if not self._exported:
            confirmed = PopUp.question(
                self,
                "Confirm Erase",
                "You have not exported local data yet.\n" "Are you sure you want to erase it?",
            )
            if confirmed:
                Log.w(
                    TAG,
                    "Erasing local data without exporting first. "
                    "Local data can be recovered from the Recycle Bin.",
                )
            else:
                Log.w(TAG, "Erase aborted by user.")
                return
        else:
            confirmed = PopUp.question(
                self, "Confirm Erase", "Are you sure you want to erase all local data?"
            )
            if confirmed:
                Log.i(
                    TAG,
                    "Erasing local data after exporting first. "
                    "Local data can be recovered from the Recycle Bin.",
                )
            else:
                Log.w(TAG, "Erase aborted by user.")
                return
        self.services.run_task(self._erase_task)

    def _erase_task(self, abort) -> None:
        """Erase locally stored run data.

        Traverses the configured logging directory and moves discovered run and
        device directories to the Recycle Bin. Progress updates are emitted as
        runs are processed, and the operation can be cancelled through the
        supplied abort event.

        Args:
            abort (threading.Event): Event used to signal cancellation of the
                erasure task.
        """
        self.services.set_freeze(False)
        try:
            data_path = os.path.join(Constants.log_prefer_path)
            Log.i(TAG, "Erasing local data...")
            self.services.emit_progress(
                self.MODE_KEY, "Erasing local data... please wait...", 0, "r"
            )

            for _folder, devices, _logs in os.walk(data_path):
                y1 = len(devices)
                for x1, device in enumerate(devices):
                    device_path = os.path.join(data_path, device)
                    for _folder2, runs, _files in os.walk(device_path):
                        y2 = len(runs)
                        for x2, run in enumerate(runs):
                            pct = int(100 * ((x1 + (x2 / max(y2, 1))) / max(y1, 1)))
                            if abort.is_set():
                                self.services.emit_progress(
                                    self.MODE_KEY,
                                    "Erase local data: Operation cancelled. "
                                    "See Recycle Bin to restore deleted runs.",
                                    pct,
                                    "b",
                                )
                                Log.w(
                                    f"{TAG} Erase cancelled by user. "
                                    "Recover deleted items from your Recycle Bin."
                                )
                                return
                            self.services.emit_progress(
                                self.MODE_KEY,
                                f"Erasing local data... please wait... Erasing '{run}'",
                                pct,
                                "r",
                            )
                            run_path = os.path.join(data_path, device, run)
                            self._trash(run_path)
                    self._trash(device_path)
            Log.i(TAG, "DONE - All local data erased.")
            self.services.emit_progress(self.MODE_KEY, "All local data erased!", 100, "g")
        except Exception as e:
            Log.e(TAG, f"Erase error: {e}")
            self.services.emit_progress(self.MODE_KEY, "Error erasing local data!", 100, "r")
        self.services.set_freeze(True)

    @staticmethod
    def _trash(path) -> None:
        """Move a file or directory to the Recycle Bin when supported.

        Uses the optional `send2trash` package when available so deleted data
        remains recoverable. If the package is unavailable, permanently removes
        the specified file or directory instead.

        Args:
            path (str): Path to the file or directory to remove.
        """
        if send2trash is not None:
            send2trash.send2trash(path)
        else:
            # Fallback if send2trash is unavailable; not recoverable from Trash.
            if os.path.isdir(path):
                import shutil as _shutil

                _shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)

    def _icon(self, name) -> QtGui.QIcon:
        """Load an icon from the application's icon directory.

        Args:
            name (str): Filename of the icon to load.

        Returns:
            QtGui.QIcon: Loaded icon, or an empty icon if the requested file
                cannot be found.
        """
        path = self._icon_file_path(name)
        return QtGui.QIcon(path) if path else QtGui.QIcon()

    @staticmethod
    def _icon_file_path(name) -> str:
        """Resolve the path to an application icon file.

        Args:
            name (str): Filename of the icon to locate.

        Returns:
            str: Normalized icon path if the file exists, otherwise an empty
                string.
        """
        try:
            path = os.path.join(Architecture.get_path(), "QATCH", "icons", name)
            if os.path.exists(path):
                return path.replace("\\", "/")
        except Exception:
            pass
        return ""
