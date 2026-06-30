"""Advanced mode - destructive / maintenance operations.

PORT FROM export_widget.Ui_Export:
    build()       <- tabAdv construction (erase warning notice + erase button,
                     plus Detect / Eject USB controls)
    erase         -> _do_erase: confirm, then services.run_task(self._erase_task)
    eraseTask     -> _erase_task(abort): walks Constants.log_prefer_path,
                     send2trash each run/device, emits progress on "advanced"
    eject         -> _do_eject: services.run_task(self._eject_task)
    ejectTask     -> _eject_task(abort)
    detect        -> _do_detect: nudge the shared USB loop to re-scan

Behavioural parity with the original:
  * Erase confirms first (different prompt if data was already exported).
  * Erased runs go to the Recycle Bin (send2trash), recoverable.
  * Tasks freeze the rest of the GUI while running; Erase itself stays enabled
    (the old "Erase only" freeze mode).
  * Detect/eject/erase share the one worker via DataServices.run_task.

LAYOUT: USB Drive / Local storage / Danger zone cards (matches the wireframe).
The USB card shows live connection status + free/total space (from
``self.services.usb_drive`` - the hardware-detected drive letter, refreshed
on usb_add/usb_remove). This is deliberately NOT ``self.services.drive``,
which is Export mode's *chosen target* and may be any local folder; reading
that here was a bug - it made this card claim a plain export folder (e.g.
C:\...\export) was a connected USB stick. The storage card shows a usage bar
for locally logged run data against the host volume's capacity. Recycle Bin
size isn't broken out separately - there's no portable way to query it
without extra OS-specific dependencies, so the bar reflects active local
run data only.
"""

import os
import shutil
import time
import subprocess

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.core.constants import Constants
from QATCH.common.logger import Logger as Log
from QATCH.common.architecture import Architecture
from QATCH.ui.popUp import PopUp

from QATCH.ui.widgets.data_mode_base import DataModeWidget
from QATCH.ui.components import GlassPushButton

try:
    import send2trash
except Exception:  # pragma: no cover - optional dependency
    send2trash = None

TAG = "[DataAdvanced]"


class _UsageBar(QtWidgets.QWidget):
    """A thin rounded usage bar: a single coloured fill over a light track."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fraction = 0.0
        self.setFixedHeight(8)
        self.setMinimumWidth(80)

    def set_fraction(self, fraction):
        self._fraction = max(0.0, min(1.0, fraction))
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)

        rect = self.rect()
        radius = rect.height() / 2.0
        painter.setBrush(QtGui.QColor(225, 230, 238, 235))
        painter.drawRoundedRect(rect, radius, radius)

        fill_w = rect.width() * self._fraction
        if fill_w > 0:
            fill_rect = QtCore.QRectF(rect.x(), rect.y(), fill_w, rect.height())
            painter.setBrush(QtGui.QColor(10, 163, 230, 235))
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
        heading.setStyleSheet(
            "QLabel { color: #333; font-size: 14px; font-weight: bold; background: transparent; }"
        )
        self.root.addWidget(heading)
        subtitle = QtWidgets.QLabel("Manage removable drives and local storage.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(self._desc_qss())
        self.root.addWidget(subtitle)

        # ---- USB Drive card --------------------------------------------
        usb_card = self._panel()
        ulay = usb_card.layout()

        usb_head = QtWidgets.QHBoxLayout()
        usb_head.setContentsMargins(0, 0, 0, 0)
        usb_head.setSpacing(10)
        usb_swatch = self._icon_swatch(
            "usb.svg", QtGui.QColor(0, 118, 174, 235), QtGui.QColor(10, 163, 230, 45)
        )
        usb_head.addWidget(usb_swatch)
        usb_title = QtWidgets.QLabel("USB Drive")
        usb_title.setStyleSheet(
            "QLabel { color: #222; font-size: 13px; font-weight: 700; background: transparent; }"
        )
        usb_head.addWidget(usb_title)
        usb_head.addStretch(1)
        self.usb_status_pill = QtWidgets.QLabel()
        usb_head.addWidget(self.usb_status_pill)
        ulay.addLayout(usb_head)

        self.usb_desc = QtWidgets.QLabel()
        self.usb_desc.setWordWrap(True)
        self.usb_desc.setStyleSheet(self._desc_qss())
        ulay.addWidget(self.usb_desc)

        usb_row = QtWidgets.QHBoxLayout()
        usb_row.setContentsMargins(0, 0, 0, 0)
        usb_row.setSpacing(8)
        self.btn_detect = GlassPushButton(" Re-detect", variant="ghost")
        self.btn_detect.setFixedHeight(30)
        self.btn_detect.setIcon(self._icon("refresh-cw.svg"))
        self.btn_detect.clicked.connect(self._do_detect)
        self.btn_eject = GlassPushButton(" Eject safely", variant="ghost")
        self.btn_eject.setFixedHeight(30)
        self.btn_eject.clicked.connect(self._do_eject)
        usb_row.addWidget(self.btn_detect)
        usb_row.addWidget(self.btn_eject)
        usb_row.addStretch(1)
        ulay.addLayout(usb_row)

        # ---- Local storage card ----------------------------------------
        storage_card = self._panel()
        slay = storage_card.layout()

        storage_head = QtWidgets.QHBoxLayout()
        storage_head.setContentsMargins(0, 0, 0, 0)
        storage_title = QtWidgets.QLabel("Local storage")
        storage_title.setStyleSheet(
            "QLabel { color: #222; font-size: 13px; font-weight: 700; background: transparent; }"
        )
        storage_head.addWidget(storage_title)
        storage_head.addStretch(1)
        self.storage_summary_label = QtWidgets.QLabel()
        self.storage_summary_label.setStyleSheet(self._caption_qss())
        storage_head.addWidget(self.storage_summary_label)
        slay.addLayout(storage_head)

        self.storage_bar = _UsageBar()
        slay.addWidget(self.storage_bar)

        legend_row = QtWidgets.QHBoxLayout()
        legend_row.setContentsMargins(0, 2, 0, 0)
        legend_row.setSpacing(6)
        legend_row.addWidget(self._legend_swatch(QtGui.QColor(10, 163, 230, 235)))
        self.storage_legend_label = QtWidgets.QLabel()
        self.storage_legend_label.setStyleSheet(
            "QLabel { color: rgba(60, 72, 88, 190); font-size: 11px; background: transparent; }"
        )
        legend_row.addWidget(self.storage_legend_label)
        legend_row.addStretch(1)
        slay.addLayout(legend_row)

        # ---- Danger zone --------------------------------------------------
        danger_caption = QtWidgets.QLabel("DANGER ZONE")
        danger_caption.setStyleSheet(
            "QLabel { color: rgba(190, 60, 50, 230); font-size: 10px; font-weight: 700; "
            "text-transform: uppercase; letter-spacing: 0.5px; background: transparent; }"
        )

        danger_card = self._panel(danger=True)
        dlay = danger_card.layout()
        drow = QtWidgets.QHBoxLayout()
        drow.setContentsMargins(0, 0, 0, 0)
        drow.setSpacing(10)
        drow.addWidget(
            self._icon_swatch(
                "warning.svg", QtGui.QColor(190, 50, 40, 235), QtGui.QColor(220, 70, 60, 40)
            ),
            0,
            QtCore.Qt.AlignTop,
        )
        dtext = QtWidgets.QVBoxLayout()
        dtext.setContentsMargins(0, 0, 0, 0)
        dtext.setSpacing(2)
        dtitle = QtWidgets.QLabel("Erase local data")
        dtitle.setStyleSheet(
            "QLabel { color: rgba(150, 40, 35, 240); font-size: 13px; font-weight: 700; "
            "background: transparent; }"
        )
        ddesc = QtWidgets.QLabel(
            "Removes all locally logged runs from this machine. Erased runs go to the "
            "Recycle Bin first - empty it afterward to erase permanently."
        )
        ddesc.setWordWrap(True)
        ddesc.setStyleSheet(
            "QLabel { color: rgba(120, 55, 50, 220); font-size: 11px; background: transparent; }"
        )
        dtext.addWidget(dtitle)
        dtext.addWidget(ddesc)
        drow.addLayout(dtext, 1)
        self.btn_erase = GlassPushButton(" Erase…", variant="danger")
        self.btn_erase.setFixedHeight(34)
        self.btn_erase.clicked.connect(self._do_erase)
        drow.addWidget(self.btn_erase, 0, QtCore.Qt.AlignVCenter)
        dlay.addLayout(drow)

        # ---- Progress display -----------------------------------------
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_label.setVisible(False)
        self.status_label.setStyleSheet(
            "QLabel { color: rgba(30, 42, 56, 210); font-size: 12px; "
            "background: rgba(255,255,255,140); border: 1px solid rgba(218, 224, 232, 170); "
            "border-radius: 8px; padding: 8px; }"
        )

        self.root.addWidget(usb_card)
        self.root.addWidget(storage_card)
        self.root.addWidget(danger_caption)
        self.root.addWidget(danger_card)
        self.root.addWidget(self.status_label)
        self.root.addStretch(1)

        # Live USB status - connected once; this mode instance lives for the
        # app's lifetime (it's never recreated), so there's no risk of
        # accumulating duplicate connections across visits.
        self.services.usb_add.connect(self._refresh_usb_status)
        self.services.usb_remove.connect(self._refresh_usb_status)

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
            # NOTE: subprocess (not os.system) avoids a console blip under pythonw.
            subprocess.call(
                "powershell $driveEject = New-Object -comObject Shell.Application; "
                '$driveEject.Namespace(17).ParseName("""{}""").InvokeVerb("""Eject""")'.format(
                    drive
                ),
                shell=True,
            )
            if getattr(self.services, "usb_drive", None) is None:
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

    @staticmethod
    def _desc_qss():
        return (
            "QLabel { color: rgba(60, 72, 88, 190); font-size: 12px; " "background: transparent; }"
        )

    @staticmethod
    def _caption_qss():
        return (
            "QLabel { color: rgba(60, 72, 88, 160); font-size: 10px; "
            "font-weight: 600; text-transform: uppercase; "
            "letter-spacing: 0.5px; background: transparent; }"
        )

    @staticmethod
    def _pill_qss(connected):
        if connected:
            return (
                "QLabel { color: rgba(20, 110, 70, 240); background: rgba(70, 180, 120, 50); "
                "border: 1px solid rgba(70, 180, 120, 120); border-radius: 9px; "
                "font-size: 10px; font-weight: 700; padding: 2px 8px; }"
            )
        return (
            "QLabel { color: rgba(90, 100, 112, 210); background: rgba(160, 170, 182, 45); "
            "border: 1px solid rgba(170, 180, 192, 120); border-radius: 9px; "
            "font-size: 10px; font-weight: 700; padding: 2px 8px; }"
        )

    def _icon_swatch(self, icon_name, fg_color, bg_color, size=32):
        swatch = QtWidgets.QLabel()
        swatch.setFixedSize(size, size)
        swatch.setAlignment(QtCore.Qt.AlignCenter)
        swatch.setStyleSheet(
            f"QLabel {{ background: rgba({bg_color.red()}, {bg_color.green()}, "
            f"{bg_color.blue()}, {bg_color.alpha()}); border-radius: {size // 4}px; }}"
        )
        icon = self._tinted_icon(icon_name, fg_color, icon_size=int(size * 0.55))
        if icon is not None:
            swatch.setPixmap(icon)
        return swatch

    @staticmethod
    def _legend_swatch(color, size=9):
        sw = QtWidgets.QLabel()
        sw.setFixedSize(size, size)
        sw.setStyleSheet(
            f"QLabel {{ background: rgba({color.red()}, {color.green()}, {color.blue()}, "
            f"{color.alpha()}); border-radius: {size // 3}px; }}"
        )
        return sw

    def _icon(self, name):
        path = self._icon_file_path(name)
        return QtGui.QIcon(path) if path else QtGui.QIcon()

    def _tinted_icon(self, name, color, icon_size=18):
        path = self._icon_file_path(name)
        if not path:
            return None
        src = QtGui.QIcon(path).pixmap(icon_size, icon_size)
        dst = QtGui.QPixmap(src.size())
        dst.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(dst)
        p.drawPixmap(0, 0, src)
        p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
        p.fillRect(dst.rect(), color)
        p.end()
        return dst

    @staticmethod
    def _icon_file_path(name):
        try:
            path = os.path.join(Architecture.get_path(), "QATCH", "icons", name)
            if os.path.exists(path):
                return path.replace("\\", "/")
        except Exception:
            pass
        return ""

    def _panel(self, danger=False):
        """A frosted glass panel with an empty body layout for the caller to
        build a custom header into (unlike ``_card``, no title is pre-added -
        each card here has its own icon-swatch header row)."""
        card = QtWidgets.QFrame()
        card.setObjectName("glassPanel")
        if danger:
            card.setStyleSheet("""
                QFrame#glassPanel {
                    background: rgba(255, 240, 238, 130);
                    border: 1px solid rgba(225, 170, 165, 190);
                    border-radius: 10px;
                }
            """)
        else:
            card.setStyleSheet("""
                QFrame#glassPanel {
                    background: rgba(255, 255, 255, 110);
                    border: 1px solid rgba(218, 224, 232, 170);
                    border-radius: 10px;
                }
            """)
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        return card
