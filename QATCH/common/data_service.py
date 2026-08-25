"""
QATCH.common.data_service.py

Shared services and concurrency infrastructure for data-management modes.

Provides :class:`DataServices`, the central service object shared by all
data-management mode widgets. The service consolidates cross-mode state and
operations that would otherwise be duplicated across individual Import,
Export, Recover, Erase, and related views.

Mode widgets should interact with this service rather than creating their own
threads, managing shared cancellation state, or directly coordinating the
global GUI freeze state. This keeps cross-cutting concurrency and shared state
in one place while allowing individual modes to remain focused on their own
UI and task-specific behavior.

Progress updates use named channels rather than the legacy integer tab index.
The predefined channels are available as :data:`CH_EXPORT` and
:data:`CH_IMPORT`.

Note:
    The USB polling, drive enumeration, and task-management implementation is
    structured as the shared-service replacement for the legacy
    `Ui_Export` machinery. Platform-specific removable-drive detection is
    currently implemented for Windows; Linux and macOS support can be added
    to :meth:`DataServices._get_removable_drives`.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-21
"""

import ctypes
import sys
from threading import Event, Thread
from typing import Callable

from PyQt5 import QtCore

from QATCH.common.logger import Logger as Log

TAG = "[DataServices]"

# Progress channels - replace the old integer `tab` argument (0=export, 1=import).
CH_EXPORT = "export"
CH_IMPORT = "import"


class DataServices(QtCore.QObject):
    """Shared service object for cross-mode data-management operations.

    Centralizes shared state, signals, and concurrency primitives used by the
    data-management mode widgets. A single instance is injected into each
    mode so USB-drive state, task cancellation, background monitoring, and
    GUI-freeze coordination remain synchronized across the entire overlay.

    Signals:
        usb_add: Emitted when a removable USB drive is detected.
        usb_remove: Emitted when the tracked USB drive is disconnected.
        progress: Emitted for mode-specific task progress as
            `(channel, label, pct, color)`.
        freeze_gui: Emitted when shared GUI controls should be frozen or
            restored.

    Attributes:
        drive (str | None): Current user-selected export or erase target. This
            may be a USB drive or any other valid folder path.
        usb_drive (str | None): Currently detected removable USB drive. This
            reflects hardware detection and is independent of `drive`.
        source_subfolder (str): Optional source subfolder associated with the
            current data-management operation.
        _abort (Event): Cooperative cancellation flag shared with the active
            foreground task.
        _stop (Event): Shutdown flag used to terminate the background USB
            monitoring loop.
        _worker (Thread | None): Background thread responsible for USB-drive
            monitoring.
        _task (Thread | None): Worker thread executing the current
            long-running data-management task.

    Args:
        parent (QtCore.QObject, optional): Parent QObject. Defaults to None.
    """

    # Signals
    usb_add = QtCore.pyqtSignal()
    usb_remove = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(str, str, int, str)
    freeze_gui = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        """Initialize the shared data-management service.

        Args:
            parent (QtCore.QObject, optional): Parent QObject. Defaults to None.
        """
        super().__init__(parent)

        # Shared drive / run state
        self.drive: str | None = None
        self.usb_drive: str | None = None
        self.source_subfolder: str = ""

        # Concurrency primitives
        self._abort = Event()
        self._stop = Event()  # tells mainTask to exit (old `do_close`)
        self._worker: Thread | None = None
        self._task: Thread | None = None

    def start(self) -> None:
        """Start the background USB-drive monitoring thread.

        Starts the removable-drive polling loop if it is not already running.
        Repeated calls while the worker is active are ignored. Clears the
        worker stop flag before launching a new daemon thread.
        """
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = Thread(target=self._main_loop, daemon=True)
        self._worker.start()
        Log.d(f"{TAG} background worker started")

    def stop(self) -> None:
        """Stop the background USB-drive monitoring thread.

        Signals the polling loop to exit, requests cancellation of any active
        foreground task, and waits briefly for the USB worker to terminate.
        The worker reference is cleared once shutdown has been requested.
        """
        self._stop.set()
        self.request_abort()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        Log.d(f"{TAG} background worker stopped")

    def _main_loop(self) -> None:
        """Monitor removable USB drives and emit connection state changes.

        Continuously polls the operating system for removable drives, compares
        each result with the previously observed set, and updates
        :attr:`usb_drive` when drives are connected or disconnected. Emits
        `usb_add` and `usb_remove` signals when the corresponding state
        changes occur.

        The loop exits when the internal stop event is set and uses the event
        as its one-second polling delay so shutdown can interrupt the wait
        immediately.
        """
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
                    if current_drives:
                        self.usb_drive = list(current_drives)[0]

                    self.usb_remove.emit()

            last_drives = current_drives
            self._stop.wait(1.0)

        Log.d(f"{TAG} USB polling loop exited")

    def _get_removable_drives(self) -> set:
        """Return the currently connected removable drives.

        Queries the operating system for removable storage devices and returns
        their paths as a set. On Windows, uses the native drive enumeration
        APIs to identify drives reported as `DRIVE_REMOVABLE`. Other
        platforms currently return an empty set.

        Returns:
            set: Drive paths for currently detected removable drives.
        """
        drives = set()

        # Windows implementation
        if sys.platform == "win32":
            bitmask = ctypes.windll.kernel32.GetLogicalDrives()
            for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                if bitmask & (1 << i):
                    drive_path = f"{letter}:\\"
                    if ctypes.windll.kernel32.GetDriveTypeW(drive_path) == 2:
                        drives.add(drive_path)

        # TODO: Linux / macOS implementations could be added here

        return drives

    def run_task(self, fn: Callable[[Event], None]) -> None:
        """Run a long-running data-management task in a worker thread.

        Starts the supplied callable on a daemon thread and provides it with
        the shared cooperative-abort event. If another task is already
        running, the request is ignored.

        Args:
            fn (Callable[[Event], None]): Task function to execute. The
                function receives the shared abort event and should
                periodically check it to support cooperative cancellation.
        """
        if self._task is not None and self._task.is_alive():
            Log.w(f"{TAG} task already running; ignoring new request")
            return
        self._abort.clear()
        self._task = Thread(target=fn, args=(self._abort,), daemon=True)
        self._task.start()

    def request_abort(self):
        """Request cooperative cancellation of the active task.

        Sets the shared abort event used by the currently running task. The
        task is responsible for periodically checking the event and exiting
        cleanly when cancellation is requested.
        """
        self._abort.set()

    @property
    def aborted(self) -> bool:
        """Return whether cancellation has been requested for the active task.

        Returns:
            bool: `True` if the shared abort event is set, otherwise `False`.
        """
        return self._abort.is_set()

    def emit_progress(self, channel: str, label: str, pct: int, color: str = "b") -> None:
        """Emit a progress update for a named data-management channel.

        Args:
            channel (str): Named channel identifying the mode receiving the
                progress update.
            label (str): Human-readable description of the current operation.
            pct (int): Progress percentage.
            color (str, optional): Progress indicator color identifier.
                Defaults to `"b"`.
        """
        self.progress.emit(channel, label, pct, color)

    def set_freeze(self, frozen: bool) -> None:
        """Broadcast a GUI freeze-state change to all subscribed modes.

        Args:
            frozen (bool): `True` to disable interactive controls while a
                task is running, or `False` to restore interaction.
        """
        self.freeze_gui.emit(frozen)
