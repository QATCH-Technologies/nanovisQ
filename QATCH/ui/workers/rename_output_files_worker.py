"""
rename_output_files_worker.py

This module provides the RenameOutputFilesWorker, which handles the post-processing
of data runs. It manages temperature interpolation via multiprocessing, checks run
quality, prompts for user naming, reallocates files to their permanent directories,
and securely zips the contents.

Author(s)
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-06-17
"""

import hashlib
import logging
import multiprocessing
import os
from datetime import date
from time import localtime, strftime
from typing import Any, List

import pyzipper
from PyQt5 import QtCore, QtWidgets

from QATCH.common.fileStorage import FileStorage
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles
from QATCH.core.constants import Constants
from QATCH.processors.Analyze import AnalyzeProcess
from QATCH.processors.InterpTemps import (
    ActionType,
    InterpTempsProcess,
    QueueCommandFormat,
)
from QATCH.ui.dialogs.pop_up_dialog import PopUp
from QATCH.ui.widgets.query_run_info_widget import QueryRunInfoWidget
from QATCH.ui.widgets.run_info_widget import RunInfoWindow


class RenameOutputFilesWorker(QtCore.QObject):
    """Worker object designed to run in a QThread to process and rename capture files.

    Handles file I/O, data quality analysis, and encryption off the main UI thread.
    Uses multiprocessing to offload temperature interpolation.

    Attributes:
        finished (QtCore.pyqtSignal): Emitted when the worker has completed all tasks.
        update_infobar (QtCore.pyqtSignal): Emitted to safely update the UI infobar (color_hex, message).
        error_occurred (QtCore.pyqtSignal): Emitted when a critical failure requires UI notification (title, message).
    """

    finished = QtCore.pyqtSignal()

    # Signals for thread-safe UI updates
    update_infobar = QtCore.pyqtSignal(str, str)
    error_occurred = QtCore.pyqtSignal(str, str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        """Initializes the RenameOutputFilesWorker.

        Args:
            parent (QtCore.QObject, optional): The parent object, typically the main window.
                Defaults to None.
        """
        super().__init__(parent)

        # Rename the attribute to avoid shadowing the built-in QObject.parent() method
        self.main_window: Any = parent

        self.finished.connect(self.indicate_done)
        self.DockingWidgets = []
        self.bThread = []
        self.bWorker = []

        # Data queues for InterpTemps.
        self._queueLog = multiprocessing.Queue()
        self._queueCmd = multiprocessing.Queue()
        self._queueOut = multiprocessing.Queue()
        self.pInterpTemps = InterpTempsProcess(self._queueLog, self._queueCmd, self._queueOut)
        self._logHandler = QtCore.QTimer()
        self._logHandler.timeout.connect(self.interp_logger)

        # Connect internal signal to the handler
        self.update_infobar.connect(self._safe_infobar_write)

    def indicate_analyzing(self) -> None:
        """Emits a signal to update the UI indicating analysis has started."""
        self.update_infobar.emit("#333333", "Run Stopped. Analyzing run...")

    def indicate_saving(self) -> None:
        """Emits a signal to update the UI indicating saving has started."""
        self.update_infobar.emit("#333333", "Run Stopped. Saving run...")

    def indicate_finalizing(self) -> None:
        """Emits a signal to update the UI indicating run info is being requested."""
        self.update_infobar.emit("#333333", "Run Stopped. Requesting run info...")

    def indicate_error(self) -> None:
        """Emits a signal to update the UI indicating a file saving error occurred."""
        self.update_infobar.emit("#ff0000", "Run Stopped. Error saving data!")

    def indicate_done(self) -> None:
        """Checks if all sub-threads are closed before finalizing the UI state."""
        remaining_threads = sum(1 for worker in self.bWorker if worker.isVisible())

        if remaining_threads > 0:
            Log.d(f"Waiting on {remaining_threads} Run Info threads to close...")
            return

        self.update_infobar.emit("#333333", "Run Stopped. Saved to local data.")

    @QtCore.pyqtSlot(str, str)
    def _safe_infobar_write(self, color_err: str, labelbar: str) -> None:
        """Thread-safe slot to execute the UI update on the main thread.

        Args:
            color_err (str): The hex color code for the text formatting.
            labelbar (str): The message to display in the infobar.
        """
        if self.main_window and hasattr(self.main_window, "ControlsWin"):
            text = f"<font color=#0000ff> Infobar </font><font color={color_err}>{labelbar}</font>"
            self.main_window.ControlsWin.ui.infobar.setText(text)

    def interp_temps(self, new_files: List[str]) -> None:
        """Starts the temperature interpolation process for the given files.

        Spawns the `pInterpTemps` multiprocessing thread and feeds file paths
        into its command queue.

        Args:
            new_files (List[str]): A list of absolute file paths to be processed.
        """
        self._logHandler.start(100)
        self.pInterpTemps.start()
        cache_loaded = False

        for file in new_files:
            if "output_tec.csv" in file:
                continue

            self.pInterpTemps._queueCmd.put(
                QueueCommandFormat(
                    file.rstrip(),
                    ActionType.load if not cache_loaded else ActionType.interp,
                ).asdict()
            )
            cache_loaded = True

        self.pInterpTemps._queueCmd.put(None)

    def interp_logger(self) -> None:
        """Pulls log messages from the multiprocessing queue and handles them."""
        if not self._queueLog.empty():
            logger = logging.getLogger("QATCH")
            while not self._queueLog.empty():
                logger.handle(self._queueLog.get(False))

    def interp_report(self) -> None:
        """Processes the output queue of the interpolation thread and reports failures."""
        while not self._queueOut.empty():
            result = self._queueOut.get(False)
            if not result.get("result"):
                self.error_occurred.emit(
                    "Temp Propagation Failure",
                    f"Failed to write temps to secondary.\nFilename: '{result.get('filename')}'\nDetails: {result.get('details')}",
                )

    def run(self) -> None:
        """Executes the main workflow for renaming, analyzing, and saving capture files.

        Reads temporary files, delegates temperature interpolation, analyzes run
        quality using AI models, prompts the user for naming, and securely archives
        the data into ZIP formats.

        Optimized for batch I/O and cached lookups to prevent pipeline stalling.
        """
        TAG = "RenameWorker"
        try:
            if not os.path.exists(Constants.new_files_path):
                Log.w(TAG, "WARNING: No new files were generated by this run!")
                self.indicate_error()
                return

            with open(Constants.new_files_path, "r") as file:
                self.indicate_analyzing()
                new_file_time = strftime(Constants.csv_default_prefix, localtime())
                content = sorted(line.rstrip() for line in file.readlines())

            # Start Interpolation
            self.interp_temps(content)

            # Wait for InterpTempsProcess to finish before looping over files.
            # This prevents the worker from freezing on the first file iteration.
            self._logHandler.stop()
            while self.pInterpTemps.is_running():
                self.interp_logger()
            self.interp_report()

            current_directory = ""
            run_directory = ""
            run_parent_directory = ""
            input_text = ""
            ask_for_info = False

            # Setup Caches and Batch Structures
            prefs_cache = (
                UserProfiles.user_preferences.get_preferences()
                if UserProfiles.user_preferences
                else {}
            )

            write_data_path = prefs_cache.get("write_data_path", None)
            dev_info_cache = {}

            zip_batches = {}  # Dict structure: { "zip_path": [("source_file", "archive_file")] }
            runs_to_query = []  # List structure: [(run_dir, new_run_path, is_good)]

            # Main File Processing Loop
            for old_path in content:
                this_dir, this_file = os.path.split(old_path)
                this_name = this_file[: this_file.rindex("_")]

                if this_dir != current_directory:
                    current_directory = this_dir
                    path_root, dev_name = os.path.split(this_dir)

                    if write_data_path:
                        path_root = os.path.join(write_data_path)

                    _dev_pid = 0
                    _dev_name = dev_name

                    try:
                        if "_" in _dev_name:
                            _dev_parts = _dev_name.split("_")
                            _dev_pid = int(_dev_parts[0], base=16) % 9
                            _dev_name = _dev_parts[1]

                        # Cache FileStorage lookups to prevent redundant DB hits
                        cache_key = f"{_dev_pid}_{_dev_name}"
                        if cache_key not in dev_info_cache:
                            dev_info_cache[cache_key] = FileStorage.DEV_info_get(
                                _dev_pid, _dev_name
                            )

                        dev_info = dev_info_cache[cache_key]
                        if dev_info.get("NAME") != _dev_name:
                            dev_name = dev_info.get("NAME", dev_name)

                        if (
                            _dev_pid != 0
                            and self.main_window
                            and hasattr(self.main_window, "has_active_multi_port")
                            and self.main_window.has_active_multi_port()
                        ):
                            _dev_pid = (
                                (_dev_pid + 9) << 4
                            ) | self.main_window.get_active_multi_port()

                    except Exception as e:
                        Log.e(TAG, f"Unable to lookup device info for: {dev_name}. Error: {e}")

                    force_save = True
                    is_good = AnalyzeProcess.Model_Data(old_path)

                    # WARNING: Blocking UI call
                    # WARNING: Blocking UI call
                    if input_text == "" and not is_good:
                        force_save = PopUp.critical(
                            self.main_window,
                            "Capture Run",
                            "Are you sure you want to save this run?",
                            "This software contains AI technology that is trained to detect run quality and it has determined this run does not match the standard characteristics of a complete run. Analysis of this run data may encounter issues. If so, trying again with a different crystal may yield better results.",
                            True,
                        )

                    self.indicate_saving()
                    status_ok = False

                    while True:
                        if input_text == "":
                            if force_save:
                                # WARNING: Blocking UI call
                                input_text, status_ok = QtWidgets.QInputDialog.getText(
                                    self.main_window,
                                    "Name this run...",
                                    "Enter a name for this run:",
                                    text=input_text,
                                )
                            else:
                                status_ok = False

                        for character in Constants.invalidChars:
                            input_text = input_text.replace(character, "")

                        prefs = UserProfiles.user_preferences
                        if prefs:
                            run_directory = prefs.get_file_save_path(
                                runname=input_text, device_id=_dev_name, port_id=_dev_pid  # type: ignore
                            )
                            run_parent_directory = prefs.get_folder_save_path(
                                runname=input_text, device_id=_dev_name, port_id=_dev_pid  # type: ignore
                            )
                        else:
                            run_directory = f"{input_text}_{_dev_name}"
                            run_parent_directory = "_unnamed"

                        if status_ok:
                            ask_for_info = True
                            if not input_text:
                                PopUp.warning(
                                    self.main_window,
                                    "Enter a Run Name",
                                    "Please enter a run name to save this run.",
                                )
                                continue

                            try:
                                target_dir = os.path.join(
                                    path_root, run_parent_directory, run_directory
                                )
                                os.makedirs(target_dir, exist_ok=False)
                            except FileExistsError:
                                PopUp.warning(
                                    self.main_window,
                                    "Duplicate Run Name",
                                    "A run with this name already exists.",
                                )
                                input_text = ""
                                continue

                            input_text = input_text.strip().replace(" ", "_")
                        else:
                            ask_for_info = False
                            input_text = new_file_time
                            if not is_good:
                                input_text += "_BAD"

                            run_parent_directory = "_unnamed"
                            prefs = UserProfiles.user_preferences
                            if prefs:
                                run_directory = prefs.get_file_save_path(
                                    runname=input_text, device_id=_dev_name, port_id=_dev_pid  # type: ignore
                                )
                            else:
                                run_directory = f"{input_text}_{_dev_name}"  # Adding as a fallback directory path.

                            run_directory = run_directory[: run_directory.rfind("_")]
                            run_directory = run_directory[: run_directory.rfind("_")]
                            target_dir = os.path.join(
                                path_root, run_parent_directory, run_directory
                            )
                            os.makedirs(target_dir, exist_ok=True)
                        break

                new_run_path = os.path.join(
                    path_root,
                    run_parent_directory,
                    run_directory,
                    this_file.replace(this_name, input_text),
                )

                try:
                    os.rename(old_path, new_run_path)
                    Log.i(f' Renamed "{old_path}" -> "{new_run_path}"')

                    # Queue file for batch zipping
                    file_dir, archive_file = os.path.split(new_run_path)
                    if run_parent_directory == "_unnamed":
                        zn = new_run_path[: new_run_path.rfind("_")] + ".zip"
                        if _dev_pid > 0:
                            zn = zn.replace(".zip", f"_{_dev_pid}.zip")
                    else:
                        zn = os.path.join(file_dir, "capture.zip")

                    if zn not in zip_batches:
                        zip_batches[zn] = []
                    zip_batches[zn].append((new_run_path, archive_file))

                except OSError as e:
                    Log.e(f' ERROR: Failed to rename "{old_path}" to "{new_run_path}": {e}')
                    self.indicate_error()

                # Queue the UI Info Query to be fired at the end
                if ask_for_info:
                    runs_to_query.append((run_directory, new_run_path, is_good))
                    ask_for_info = False

                old_path_dir = os.path.split(old_path)[0]
                try:
                    if os.path.exists(old_path_dir) and not os.listdir(old_path_dir):
                        os.rmdir(old_path_dir)
                except OSError:
                    pass

            # Execute Batch Zipping
            enabled, error, expires = UserProfiles.checkDevMode()
            should_encrypt = UserProfiles.count() > 0 and not enabled

            if not enabled and (error or expires != ""):
                PopUp.warning(
                    self.main_window,
                    "Developer Mode Expired",
                    "Developer Mode has expired. Encrypting data.",
                )

            for zip_name, files_to_zip in zip_batches.items():
                with pyzipper.AESZipFile(
                    zip_name,
                    "a",
                    compression=pyzipper.ZIP_DEFLATED,
                    allowZip64=True,
                    encryption=pyzipper.WZ_AES,
                ) as zf:
                    friendly_name = f"Capture Archive ({date.today()})"
                    zf.comment = friendly_name.encode()

                    if should_encrypt:
                        zf.setpassword(hashlib.sha256(zf.comment).hexdigest().encode())
                    else:
                        zf.setencryption(None)

                    for copy_file, archive_file in files_to_zip:
                        zf.write(copy_file, arcname=archive_file)
                        if archive_file.endswith(".csv"):
                            crc_file = archive_file[:-4] + ".crc"
                            zf.writestr(crc_file, str(hex(zf.getinfo(archive_file).CRC)))

                        # Clean up file after zipping
                        os.remove(copy_file)

            # Finalize UI and Threads
            self.indicate_finalizing()
            user_name = (
                "" if not self.main_window else self.main_window.ControlsWin.username.text()[6:]
            )

            for r_dir, r_path, r_good in runs_to_query:
                self.bThread.append(QtCore.QThread())

                worker = QueryRunInfoWidget(
                    r_dir, r_path, r_good, user_name, parent=self.main_window
                )
                self.bWorker.append(worker)
                self.bThread[-1].started.connect(worker.show)
                worker.finished.connect(self.bThread[-1].quit)
                worker.finished.connect(self.indicate_done)

            num_runs_saved = len(self.bThread)
            for i in range(num_runs_saved):
                self.bWorker[i].setRuns(num_runs_saved, i)

            if num_runs_saved == 1:
                self.bThread[-1].start()
            elif num_runs_saved > 1:
                self.RunInfoWindow = RunInfoWindow(self.bWorker, self.bThread)

        except Exception as e:
            import traceback

            Log.e("Critical failure in RenameOutputFilesWorker:")
            for line in traceback.format_exc().splitlines():
                Log.e(line)
        finally:
            if os.path.exists(Constants.new_files_path):
                try:
                    os.remove(Constants.new_files_path)
                except OSError:
                    pass
            self.finished.emit()

    def _portIDfromIndex(self, pid: int) -> str:
        """Converts an ASCII byte to a port character string.

        Args:
            pid (int): Port ID index (e.g., 1-4 for 4x1, or 0xA1-0xD6 for 4x6 systems).

        Returns:
            str: The hexadecimal string representation of the PID in uppercase.
        """
        return hex(pid)[2:].upper()
