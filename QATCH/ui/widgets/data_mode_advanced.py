import os
import shutil
import subprocess
import time

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

try:
    import send2trash
except Exception:  # pragma: no cover - optional dependency
    send2trash = None

TAG = "[DataAdvanced]"


class _UsageBar(QtWidgets.QWidget):
    """A thin rounded usage bar: a single coloured fill over a neutral track."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fraction = 0.0
        self.setFixedHeight(8)
        self.setMinimumWidth(80)
        ThemeManager.instance().themeChanged.connect(lambda _: self.update())

    def set_fraction(self, fraction):
        self._fraction = max(0.0, min(1.0, fraction))
        self.update()

    def paintEvent(self, event):
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
    MODE_KEY = "advanced"
    MODE_LABEL = "Advanced"

    # ------------------------------------------------------------------
    #  Build
    # ------------------------------------------------------------------
    def build(self):
        # Whether export happened this session - drives the erase prompt wording.
        self._exported = False

        heading = QtWidgets.QLabel("Advanced")
        self._heading = heading
        subtitle = QtWidgets.QLabel("Manage removable drives and local storage.")
        subtitle.setWordWrap(True)
        self._subtitle = subtitle
        self.root.addWidget(heading)
        self.root.addWidget(subtitle)

        # ---- USB Drive card --------------------------------------------
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

        # ---- Local storage card ----------------------------------------
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

        # ---- Danger zone --------------------------------------------------
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

        # ---- Progress display -----------------------------------------
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

        # Live USB status - connected once; this mode instance lives for the
        # app's lifetime (it's never recreated), so there's no risk of
        # accumulating duplicate connections across visits.
        self.services.usb_add.connect(self._refresh_usb_status)
        self.services.usb_remove.connect(self._refresh_usb_status)

    # ------------------------------------------------------------------
    #  Theming
    # ------------------------------------------------------------------
    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_theme()
        self._refresh_usb_status()

    def _apply_theme(self) -> None:
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

    # ------------------------------------------------------------------
    #  Lifecycle / shared-state hooks
    # ------------------------------------------------------------------
    def on_enter(self):
        self._refresh_usb_status()
        self._refresh_storage()

    def on_freeze(self, frozen: bool):
        """Freeze everything except Erase (the original 'Erase only' mode)."""
        self.btn_detect.setDisabled(frozen)
        self.btn_eject.setDisabled(frozen)
        # btn_erase intentionally stays enabled.

    def on_progress(self, label, pct, color):
        if label:
            self.status_label.setVisible(True)
            self.status_label.setText(label)
        if pct == 100:
            # Erase/eject finished (success, cancel, or error) - local state
            # may have changed either way, so refresh both cards.
            self._refresh_usb_status()
            self._refresh_storage()

    def note_exported(self, exported: bool = True):
        """Called by the container/export mode so the erase prompt knows whether
        data was already exported this session."""
        self._exported = exported

    # ------------------------------------------------------------------
    #  USB status / local storage display
    # ------------------------------------------------------------------
    def _refresh_usb_status(self, *_):
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

    def _refresh_storage(self):
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
    def _compute_storage_stats():
        """(run_count, total_bytes) for everything under the logged-data path."""
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
    def _fmt_size(num_bytes):
        size = float(num_bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024.0 or unit == "TB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    # ------------------------------------------------------------------
    #  Detect
    # ------------------------------------------------------------------
    def _do_detect(self):
        # The shared loop owns detection; ask it to re-scan now. If the service
        # exposes a manual trigger use it, otherwise this is a no-op nudge.
        trigger = getattr(self.services, "request_detect", None)
        if callable(trigger):
            trigger()
        else:
            Log.d(f"{TAG} detect requested (shared loop handles enumeration)")
        self._refresh_usb_status()

    # ------------------------------------------------------------------
    #  Eject
    # ------------------------------------------------------------------
    def _do_eject(self):
        self.services.run_task(self._eject_task)

    def _eject_task(self, abort):
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

            # --- THE FIX: Actively test drive accessibility ---
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
            # --------------------------------------------------

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

    # ------------------------------------------------------------------
    #  Erase
    # ------------------------------------------------------------------
    def _do_erase(self):
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

    def _erase_task(self, abort):
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

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _trash(path):
        if send2trash is not None:
            send2trash.send2trash(path)
        else:
            # Fallback if send2trash is unavailable; not recoverable from Trash.
            if os.path.isdir(path):
                import shutil as _shutil

                _shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)

    def _icon(self, name):
        path = self._icon_file_path(name)
        return QtGui.QIcon(path) if path else QtGui.QIcon()

    @staticmethod
    def _icon_file_path(name):
        try:
            path = os.path.join(Architecture.get_path(), "QATCH", "icons", name)
            if os.path.exists(path):
                return path.replace("\\", "/")
        except Exception:
            pass
        return ""
