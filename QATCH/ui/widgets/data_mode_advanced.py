"""Advanced mode — destructive / maintenance operations.

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
"""

import os
import time
import subprocess

from PyQt5 import QtCore, QtWidgets

from QATCH.core.constants import Constants
from QATCH.common.logger import Logger as Log
from QATCH.ui.popUp import PopUp

from QATCH.ui.widgets.data_mode_base import DataModeWidget
from QATCH.ui.components.glass_push_button import GlassPushButton

try:
    import send2trash
except Exception:  # pragma: no cover - optional dependency
    send2trash = None

TAG = "[DataAdvanced]"


class AdvancedMode(DataModeWidget):
    MODE_KEY = "advanced"
    MODE_LABEL = "Advanced"

    # ------------------------------------------------------------------
    #  Build
    # ------------------------------------------------------------------
    def build(self):
        # Whether export happened this session — drives the erase prompt wording.
        self._exported = False

        # ---- USB maintenance card -------------------------------------
        usb_card = self._card("USB Drive")
        usb_lay = usb_card.layout()

        usb_desc = QtWidgets.QLabel(
            "Detect a connected USB drive for export, or safely eject the "
            "current drive before removing it."
        )
        usb_desc.setWordWrap(True)
        usb_desc.setStyleSheet(self._desc_qss())
        usb_lay.addWidget(usb_desc)

        usb_caption = QtWidgets.QLabel("Drive actions")
        usb_caption.setStyleSheet(self._caption_qss())
        usb_lay.addWidget(usb_caption)

        usb_row = QtWidgets.QHBoxLayout()
        usb_row.setContentsMargins(0, 0, 0, 0)
        usb_row.setSpacing(8)
        self.btn_detect = GlassPushButton(" Detect USB", variant="default")
        self.btn_detect.setFixedHeight(34)
        self.btn_detect.clicked.connect(self._do_detect)
        self.btn_eject = GlassPushButton(" Eject USB", variant="default")
        self.btn_eject.setFixedHeight(34)
        self.btn_eject.clicked.connect(self._do_eject)
        usb_row.addWidget(self.btn_detect)
        usb_row.addWidget(self.btn_eject)
        usb_row.addStretch(1)
        usb_lay.addLayout(usb_row)

        # ---- Erase local data card (destructive) ----------------------
        erase_card = self._card("Erase Local Data", danger=True)
        erase_lay = erase_card.layout()

        warn = QtWidgets.QLabel(
            "This erases all locally logged data from this machine. Erased runs "
            "can be recovered from the Recycle Bin — to fully erase, empty your "
            "Recycle Bin afterward."
        )
        warn.setWordWrap(True)
        warn.setStyleSheet(
            "QLabel { color: rgba(150, 50, 40, 220); font-size: 12px; background: transparent; }"
        )
        erase_lay.addWidget(warn)

        erase_row = QtWidgets.QHBoxLayout()
        erase_row.setContentsMargins(0, 0, 0, 0)
        erase_row.addStretch(1)
        self.btn_erase = GlassPushButton(" Erase Local Data", variant="danger")
        self.btn_erase.setFixedHeight(34)
        self.btn_erase.clicked.connect(self._do_erase)
        erase_row.addWidget(self.btn_erase)
        erase_lay.addLayout(erase_row)

        # ---- Progress display -----------------------------------------
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_label.setVisible(False)
        self.status_label.setStyleSheet(
            "QLabel { color: rgba(30, 42, 56, 210); font-size: 12px; "
            "background: rgba(255,255,255,40); border: 1px solid rgba(255,255,255,120); "
            "border-radius: 8px; padding: 8px; }"
        )

        self.root.addWidget(usb_card)
        self.root.addWidget(erase_card)
        self.root.addWidget(self.status_label)
        self.root.addStretch(1)

    # ------------------------------------------------------------------
    #  Lifecycle / shared-state hooks
    # ------------------------------------------------------------------
    def on_freeze(self, frozen: bool):
        """Freeze everything except Erase (the original 'Erase only' mode)."""
        self.btn_detect.setDisabled(frozen)
        self.btn_eject.setDisabled(frozen)
        # btn_erase intentionally stays enabled.

    def on_progress(self, label, pct, color):
        if label:
            self.status_label.setVisible(True)
            self.status_label.setText(label)

    def note_exported(self, exported: bool = True):
        """Called by the container/export mode so the erase prompt knows whether
        data was already exported this session."""
        self._exported = exported

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

    # ------------------------------------------------------------------
    #  Eject
    # ------------------------------------------------------------------
    def _do_eject(self):
        self.services.run_task(self._eject_task)

    def _eject_task(self, abort):
        self.services.set_freeze(False)
        drive = getattr(self.services, "drive", None)
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
            if getattr(self.services, "drive", None) is None:
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
                import shutil

                shutil.rmtree(path, ignore_errors=True)
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

    def _card(self, title, danger=False):
        # Frosted glass panel matching the import widget. The danger variant
        # keeps the same frosted base but adds a restrained left accent so the
        # destructive section still reads as distinct.
        card = QtWidgets.QFrame()
        card.setObjectName("glassPanel")
        accent = "border-left: 3px solid rgba(200, 70, 55, 200);" if danger else ""
        card.setStyleSheet(f"""
            QFrame#glassPanel {{
                background: rgba(255, 255, 255, 30);
                border: 1px solid rgba(200, 210, 220, 110);
                {accent}
                border-radius: 10px;
            }}
        """)
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        header = QtWidgets.QLabel(title)
        header.setStyleSheet(
            "QLabel { color: #333; font-size: 12px; font-weight: bold; "
            "background: transparent; }"
        )
        lay.addWidget(header)
        return card
