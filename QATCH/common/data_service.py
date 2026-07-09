"""DataServices - shared, cross-mode machinery for the data-management overlay.

This object is constructed once by `DataManagementWidget` and injected into
every mode submodule. It owns the things the original `Ui_Export` shared
across its tabs:

  * the background worker thread (the old `mainTask` USB-detection loop)
  * USB drive state (`drive`) and add/remove notifications
  * the single `progress` signal, now routed by a named *channel* instead of
    a magic tab index
  * GUI-freeze coordination (import/export enable-state moved in lockstep)
  * a cooperative task-abort flag shared by import/export/erase/eject tasks

Submodules never start threads or flip the global freeze themselves - they call
into the service and subscribe to its signals. That keeps the cross-cutting
concurrency in exactly one place.

NOTE (skeleton): the actual USB polling loop, drive enumeration, and task
plumbing are stubbed. Port the bodies from:
    export_widget.Ui_Export.mainTask / ui_add / ui_remove / diff / cancel
"""

import ctypes
import sys
from threading import Event, Thread
from typing import Callable, Optional

from PyQt5 import QtCore

from QATCH.common.logger import Logger as Log

TAG = "[DataServices]"

# Progress channels - replace the old integer `tab` argument (0=export, 1=import).
CH_EXPORT = "export"
CH_IMPORT = "import"


class DataServices(QtCore.QObject):
    # ---- Signals (mirror the old Ui_Export class-level signals) ----
    usb_add = QtCore.pyqtSignal()
    usb_remove = QtCore.pyqtSignal()
    # (channel, label, pct, color)  - channel replaces the old trailing tab int
    progress = QtCore.pyqtSignal(str, str, int, str)
    # Broadcast freeze state so every mode toggles its controls together.
    freeze_gui = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Shared drive / run state (ported fields from Ui_Export).
        #
        # `drive` and `usb_drive` look redundant but mean different
        # things, and conflating them is what made Advanced's "USB Drive"
        # card show a local export folder (e.g. C:\...\export) as if it
        # were a connected USB stick:
        #   - usb_drive: the actual hardware-detected USB drive letter, if
        #     any. Only the USB-detection loop (`_main_loop`) writes this;
        #     everyone else only reads it. None means "no USB connected."
        #   - drive: the user's current chosen export/erase TARGET, which
        #     may be a USB drive letter or any plain folder path. Export
        #     mode owns writes to this.
        self.drive: Optional[str] = None
        self.usb_drive: Optional[str] = None
        self.source_subfolder: str = ""

        # Concurrency primitives. `_abort` is the cooperative cancel flag the
        # long-running tasks poll; `_worker` is the background USB loop.
        self._abort = Event()
        self._stop = Event()  # tells mainTask to exit (old `do_close`)
        self._worker: Optional[Thread] = None
        self._task: Optional[Thread] = None

    # ------------------------------------------------------------------
    #  Lifecycle - started when the overlay opens, stopped when it closes
    # ------------------------------------------------------------------
    def start(self):
        """Spin up the background USB-detection loop (old `mainTask`)."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = Thread(target=self._main_loop, daemon=True)
        self._worker.start()
        Log.d(f"{TAG} background worker started")

    def stop(self):
        """Signal the loop to exit and join it (old `closeEvent` behavior)."""
        self._stop.set()
        self.request_abort()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        Log.d(f"{TAG} background worker stopped")

    def _main_loop(self):
        """Poll for USB drives, diff against last seen set, and emit signals."""
        Log.d(f"{TAG} USB polling loop started")
        last_drives = self._get_removable_drives()
        if last_drives:
            self.usb_drive = list(last_drives)[0]

        while not self._stop.is_set():
            current_drives = self._get_removable_drives()
            added_drives = current_drives - last_drives
            removed_drives = last_drives - current_drives

            if added_drives:
                new_drive = list(added_drives)[0]
                self.usb_drive = new_drive
                Log.d(f"{TAG} USB drive connected: {self.usb_drive}")
                self.usb_add.emit()

            if removed_drives:
                if self.usb_drive in removed_drives:
                    Log.d(f"{TAG} USB drive disconnected: {self.usb_drive}")
                    self.usb_drive = None

                    # If there's another USB still plugged in, fallback to it
                    if current_drives:
                        self.usb_drive = list(current_drives)[0]

                    self.usb_remove.emit()

            last_drives = current_drives
            self._stop.wait(1.0)

        Log.d(f"{TAG} USB polling loop exited")

    def _get_removable_drives(self) -> set:
        """Query the OS for currently connected removable drives."""
        drives = set()

        # Windows implementation
        if sys.platform == "win32":
            # GetLogicalDrives returns a bitmask of available drives (1=A, 2=B, 4=C...)
            bitmask = ctypes.windll.kernel32.GetLogicalDrives()
            for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                if bitmask & (1 << i):
                    drive_path = f"{letter}:\\"
                    # GetDriveTypeW returns 2 for DRIVE_REMOVABLE (USB sticks)
                    if ctypes.windll.kernel32.GetDriveTypeW(drive_path) == 2:
                        drives.add(drive_path)

        # Linux / macOS implementations could be added here using os.listdir
        # on /media, /run/media, or /Volumes, or by utilizing the `psutil` library.

        return drives

    # ------------------------------------------------------------------
    #  Task management - one foreground-ish worker at a time, cooperative abort
    # ------------------------------------------------------------------
    def run_task(self, fn: Callable[[Event], None]):
        """Run a long task (import/export/erase) on a worker thread.

        `fn` receives the abort Event and should poll it to support Cancel.
        Replaces the ad-hoc `Thread(target=self.importTask, args=(abort, ...))`
        pattern scattered through Ui_Export.
        """
        if self._task is not None and self._task.is_alive():
            Log.w(f"{TAG} task already running; ignoring new request")
            return
        self._abort.clear()
        self._task = Thread(target=fn, args=(self._abort,), daemon=True)
        self._task.start()

    def request_abort(self):
        """Cooperative cancel for the active task (old `cancel`)."""
        self._abort.set()

    @property
    def aborted(self) -> bool:
        return self._abort.is_set()

    # ------------------------------------------------------------------
    #  Convenience emit helpers (so submodules don't construct signal args)
    # ------------------------------------------------------------------
    def emit_progress(self, channel: str, label: str, pct: int, color: str = "b"):
        self.progress.emit(channel, label, pct, color)

    def set_freeze(self, frozen: bool):
        self.freeze_gui.emit(frozen)
