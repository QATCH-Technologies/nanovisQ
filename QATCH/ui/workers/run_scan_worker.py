# Standard Library
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from PyQt5 import QtCore

from QATCH.common.fileStorage import FileStorage


class RunScanWorker(QtCore.QThread):
    """Background thread to handle run scanning without freezing the UI.

    This worker offloads heavy filesystem I/O and XML parsing from the main
    event loop. It iterates through a list of devices, identifies data folders,
    and uses a ThreadPoolExecutor to parse individual run metadata in parallel
    within this background thread.

    Attributes:
        scan_finished (QtCore.pyqtSignal): Signal emitted after each device is
            processed. Emits:
            - List[Dict[str, Any]]: The parsed scan results.
            - str: The name of the device just scanned.
            - List[str]: Keys of runs that were not found on disk.
            - bool: True if this was the last device in the queue.
    """

    scan_finished = QtCore.pyqtSignal(list, str, list, bool)

    def __init__(
        self,
        devices_to_scan: List[str],
        known_timestamps: List[str],
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        """Initializes the worker with device and cache context.

        Args:
            devices_to_scan (List[str]): List of device names to be audited.
            known_timestamps (List[str]): Current keys in the timestamp cache
                to determine if a run needs a fresh XML parse.
            parent (Optional[QtCore.QObject]): The Qt parent object.
        """
        super().__init__(parent)
        self.devices_to_scan = devices_to_scan
        self.known_timestamps = known_timestamps

    def run(self) -> None:
        """Executes the background scanning logic.

        For each device, it identifies existing folders and calculates which
        cached runs are missing. It then uses a thread pool to perform parallel
        scans of individual run folders.
        """
        from QATCH.ui.interfaces import UIAnalyze

        num_devices = len(self.devices_to_scan)

        for i, data_device in enumerate(self.devices_to_scan):
            # Fetch directories, excluding internal naming conventions
            runs = FileStorage.DEV_get_logged_data_folders(data_device)
            runs = [x for x in runs if x != "_unnamed"]
            unchecked_runs = [x for x in self.known_timestamps if x.endswith(data_device)]

            scan_args = []
            for run in runs:
                dict_key = f"{run}:{data_device}"
                # Only parse XML if the run is new (not in known_timestamps)
                needs_parse = dict_key not in self.known_timestamps
                scan_args.append((data_device, run, needs_parse))

            scan_results: List[Dict[str, Any]] = []
            if scan_args:
                with ThreadPoolExecutor(
                    max_workers=min(8, len(scan_args)), thread_name_prefix=f"scan-{data_device}"
                ) as ex:
                    scan_results = list(ex.map(lambda a: UIAnalyze._scan_run(*a), scan_args))

            is_last = i == num_devices - 1
            self.scan_finished.emit(scan_results, data_device, unchecked_runs, is_last)
