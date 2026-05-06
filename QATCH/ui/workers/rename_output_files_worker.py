from datetime import date
import hashlib
import logging
import multiprocessing
import os
import sys
from time import localtime, strftime
from typing import List
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
from QATCH.ui.widgets.query_run_info_widget import QueryRunInfoWidget



class RenameOutputFilesWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
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

    def indicate_analyzing(self):
        color_err = "#333333"
        labelbar = "Run Stopped. Analyzing run..."
        self.infobar_write(color_err, labelbar)

    def indicate_saving(self):
        color_err = "#333333"
        labelbar = "Run Stopped. Saving run..."
        self.infobar_write(color_err, labelbar)

    def indicate_finalizing(self):
        color_err = "#333333"
        labelbar = "Run Stopped. Requesting run info..."
        self.infobar_write(color_err, labelbar)

    def indicate_error(self):
        color_err = "#ff0000"
        labelbar = "Run Stopped. Error saving data!"
        self.infobar_write(color_err, labelbar)

    def indicate_done(self):
        # are we really done?
        remaining_threads = 0
        for i in range(len(self.bThread)):
            if self.bWorker[i].isVisible():
                remaining_threads += 1
        if remaining_threads > 0:
            Log.d(f"Waiting on {remaining_threads} Run Info threads to close...")
            return  # not done yet
        color_err = "#333333"
        labelbar = "Run Stopped. Saved to local data."
        self.infobar_write(color_err, labelbar)

    def infobar_write(self, color_err, labelbar):
        self.parent.ControlsWin.ui1.infobar.setText(
            "<font color=#0000ff> Infobar </font><font color={}>{}</font>".format(
                color_err, labelbar
            )
        )

    def interp_temps(self, new_files: List[str]) -> None:
        """
        The handler method for performing temperature interpolation on a list of files.

        This method handles starting the `pInterpTemps` thread which interpolates temperature
        through a list of file paths.

        Args:
            new_files (List[str]): A list of file paths as strings.

        Returns:
            None
        """
        # NOTE: Uncomment these 3 lines to skip entire process:
        # self.pInterpTemps._started.set()
        # self.pInterpTemps._done.set()
        # return
        self._logHandler.start(100)
        self.pInterpTemps.start()
        cache_loaded = False
        for file in new_files:
            # Skip TEC files.
            if "output_tec.csv" in file:
                continue

            # NOTE: Might need better filtering of files to load/interp correctly
            # Assumption here is that alphabetical order yields primary dev 1st.
            # TODO: This will likely need modification to work with 4x6 devices.
            self.pInterpTemps._queueCmd.put(
                QueueCommandFormat(
                    file.rstrip(),
                    ActionType.load if not cache_loaded else ActionType.interp,
                ).asdict()
            )
            cache_loaded = True
        # Signal finished and end process
        self.pInterpTemps._queueCmd.put(None)

    def interp_logger(self) -> None:
        """
        Initializes the logging utility for the pInterpTemps thread.

        Returns:
            None
        """
        if not self._queueLog.empty():
            logger = logging.getLogger("QATCH")
            while not self._queueLog.empty():
                logger.handle(self._queueLog.get(False))

    def interp_report(self) -> None:
        """
        Generates the report after pInterpTemps thread executes.
        Critical failures are propogated to the user via UI popups.

        Returns:
            None
        """
        while not self._queueOut.empty():
            result = self._queueOut.get(False)
            if not result["result"]:
                # inform user of any failures
                PopUp.critical(
                    parent=self.parent,
                    title="Temp Propagation Failure",
                    message="ERROR: Failed to write temps to secondary.\n"
                    + f'Filename: "{result["filename"]}"',
                    details=result["details"],
                    ok_only=True,
                )

    #######################################################################
    # Prompt user for run name(s) and rename new output file(s) accordingly
    #######################################################################
    def run(self) -> None:
        """
        Executes the run process to handle new files, analyze data quality, prompt user input,
        and manage file storage and encryption.

        This method checks for newly generated files, processes them by analyzing quality,
        renaming, and saving them into user-defined directories. It also handles user input
        for naming runs, encrypts output files if required, and manages error conditions.

        Raises:
            Exception: If the run name entered by the user is empty.

        Returns:
            None
        """
        try:
            # Check if new files were generated.  If no new files generated, log as a warnings
            # and emit the finished signal to parent along with the error.
            if not os.path.exists(Constants.new_files_path):
                Log.w(TAG, "WARNING: No new files were generated by this run!")
                self.finished.connect(self.indicate_error)
                self.finished.emit()
                return

            # Copy and rename log files from the temporary directory in alphabetical order
            # to include a user-defined run name string
            content = []
            with open(Constants.new_files_path, "r") as file:
                self.indicate_analyzing()
                new_file_time = strftime(Constants.csv_default_prefix, localtime())
                content = file.readlines()
                content.sort()

            # Start InterpTempsProcess, if a multiplex run
            self.interp_temps(content)

            # Current device directory.
            current_directory = ""

            # Sub-device directory or the name of the run.
            run_directory = ""

            # Pop-up input text from user.
            input_text = ""

            # Pop-up status.
            status_ok = False

            # Additional_info_control request.
            ask_for_info = False

            # Loop through the paths of each data file re-writing them to
            # the user defined directory.
            for line in content:
                old_path = line.rstrip()
                path_parts = os.path.split(old_path)
                this_dir = path_parts[0]
                this_file = path_parts[1]
                this_name = this_file[: this_file.rindex("_")]
                copy_file = None

                # Skip if the new directory is the same as the temporary directory.
                if this_dir != current_directory:
                    current_directory = this_dir
                    path_split = os.path.split(this_dir)
                    preferences_write_path = UserProfiles.user_preferences.get_preferences().get(
                        "write_data_path", None
                    )
                    if not preferences_write_path:
                        path_root = path_split[0]
                    else:
                        # If user preferences are set, load from the prefered write path.
                        path_root = os.path.join(preferences_write_path)
                    dev_name = path_split[1]
                    try:
                        _dev_pid = 0
                        _dev_name = dev_name
                        if _dev_name.count("_") > 0:
                            _dev_parts = _dev_name.split("_")
                            _dev_pid = int(_dev_parts[0], base=16) % 9
                            # do not override 'dev_name'
                            _dev_name = _dev_parts[1]

                        # Retrieve device info based on the device name and port-id
                        dev_info = FileStorage.DEV_info_get(_dev_pid, _dev_name)
                        if "NAME" in dev_info:
                            if dev_info["NAME"] != _dev_name:
                                dev_name = dev_info["NAME"]

                        if _dev_pid != 0:  # append Port ID 1-4 for 4x1, ID A1-D6 for 4x6
                            # Convert PID to multiplex designation (i.e. int(1) -> int(162) for "A2")
                            if self.parent.has_active_multi_port():  # 4x6 system
                                # mask in port, e.g. "A" -> "A1"
                                _dev_pid = (
                                    (_dev_pid + 9) << 4
                                ) | self.parent.get_active_multi_port()
                            # else: 4x1 system, nothing to do
                    except:
                        Log.e(TAG, f"Unable to lookup device info for: {dev_name}")

                    # Controls saving with errors.
                    force_save = True
                    is_good = AnalyzeProcess.Model_Data(old_path)

                    # If run is not deemed "good" report this to the user asking if they should save the
                    # force save with bad data.
                    if input_text == "" and not is_good:
                        force_save = PopUp.critical(
                            self.parent,
                            "Capture Run",
                            "Are you sure you want to save this run?",
                            "This software contains AI technology that is trained to detect run quality and it has determined this run does not match the standard characteristics of a complete run. Analysis of this run data may enocunter issues. If so, trying again with a different crystal may yield better results.",
                            True,
                        )
                    self.indicate_saving()

                    # Prompt user for a runname.
                    while True:
                        if input_text == "":
                            if force_save:
                                input_text, status_ok = QtWidgets.QInputDialog.getText(
                                    self.parent,
                                    "Name this run...",
                                    # for device "{}":'.format(dev_name),
                                    "Enter a name for this run:",
                                    text=input_text,
                                )
                            else:
                                status_ok = False  # bad run, don't save with custom name

                        # Remove any invalid characters from user input
                        # invalid_characters = "\\/:*?\"'<>|"
                        for character in Constants.invalidChars:
                            input_text = input_text.replace(character, "")

                        # Fetch run and run_parent directories based on user preferences.
                        run_directory = UserProfiles.user_preferences.get_file_save_path(
                            runname=input_text,
                            device_id=_dev_name,
                            port_id=_dev_pid,
                        )
                        run_parent_directory = UserProfiles.user_preferences.get_folder_save_path(
                            runname=input_text,
                            device_id=_dev_name,
                            port_id=_dev_pid,
                        )

                        if status_ok:
                            ask_for_info = True

                            # Raise exception if runname retrieved from user is empty.
                            try:
                                if len(input_text) == 0:
                                    raise Exception("No text entered. Please try again.")

                                # Using Device folder path from UserPreferences class.
                                # os.makedirs(os.path.join(
                                #     path_root, dev_name, run_directory), exist_ok=False)

                                os.makedirs(
                                    os.path.join(path_root, run_parent_directory, run_directory),
                                    exist_ok=False,
                                )
                                # break (done below)
                            except:
                                if len(input_text) > 0:
                                    PopUp.warning(
                                        self.parent,
                                        "Duplicate Run Name",
                                        "A run with this name already exists...\nPlease try again with a different name.",
                                    )
                                else:
                                    PopUp.warning(
                                        self.parent,
                                        "Enter a Run Name",
                                        "Please enter a run name to save this run...\nPlease try again with a valid name.",
                                    )
                                input_text = ""
                                continue  # no break (try again)
                            input_text = input_text.strip().replace(
                                " ", "_"
                            )  # word spaces -> underscores
                        else:
                            ask_for_info = False
                            input_text = new_file_time  # uniquify
                            if not is_good:
                                input_text += "_BAD"
                            run_parent_directory = "_unnamed"
                            run_directory = UserProfiles.user_preferences.get_file_save_path(
                                runname=input_text,
                                device_id=_dev_name,
                                port_id=_dev_pid,
                            )
                            # trim off dev_id
                            run_directory = run_directory[: run_directory.rfind("_")]
                            os.makedirs(
                                os.path.join(path_root, run_parent_directory, run_directory),
                                exist_ok=True,
                            )
                        break

                # We *must* wait here until InterpTempsProcess has finished
                self._logHandler.stop()  # stop log handler timer
                if self.pInterpTemps.is_running():
                    Log.w("Waiting for InterpTempsProcess to finish...")
                while self.pInterpTemps.is_running():
                    self.interp_logger()  # wait for exit, handle logger
                self.interp_report()  # report any failures to user

                # Construct new run path to the new run path under run parent directory.
                new_run_path = os.path.join(
                    path_root,
                    run_parent_directory,
                    run_directory,
                    this_file.replace(this_name, input_text),
                )

                try:
                    # Attempt to rename temporary files to the new run path.
                    os.rename(old_path, new_run_path)
                    Log.i(' Renamed "{}" ->\n         "{}"'.format(old_path, new_run_path))
                    copy_file = new_run_path
                except Exception:
                    # Log and raise any errors terminating early if rename fails.
                    Log.e(' ERROR: Failed to rename "{}" to "{}"!!!'.format(old_path, new_run_path))
                    self.finished.connect(self.indicate_error)
                    if os.path.isfile(old_path):
                        copy_file = old_path
                    if os.path.isfile(new_run_path):
                        copy_file = new_run_path

                old_path_parts = os.path.split(old_path)
                try:
                    # Only try to delete path if the path is empty.
                    if len(os.listdir(old_path_parts[0])) == 0:
                        # Delete the old path folder (throws error if not empty)
                        os.rmdir(old_path_parts[0])
                except:
                    Log.e(
                        ' ERROR: Failed to clean-up after renaming "{}"!!!'.format(
                            old_path_parts[1]
                        )
                    )
                    self.finished.connect(self.indicate_error)

                if copy_file != None:  # require access controls
                    file_parts = os.path.split(copy_file)
                    if run_parent_directory == "_unnamed":
                        zn = copy_file[: copy_file.rfind("_")] + ".zip"
                        if _dev_pid > 0:  # multiplex systems, add dev_id
                            zn = zn.replace(".zip", f"_{_dev_pid}.zip")
                    else:
                        zn = os.path.join(file_parts[0], f"capture.zip")
                    archive_file = file_parts[1]
                    crc_file = archive_file[:-4] + ".crc"

                    # Create a new zip file with a password
                    with pyzipper.AESZipFile(
                        zn,
                        "a",
                        compression=pyzipper.ZIP_DEFLATED,
                        allowZip64=True,
                        encryption=pyzipper.WZ_AES,
                    ) as zf:
                        # Add a protected file to the zip archive
                        friendly_name = f"{run_directory} ({date.today()})"
                        zf.comment = friendly_name.encode()  # run name

                        enabled, error, expires = UserProfiles.checkDevMode()
                        if enabled == False and (error == True or expires != ""):
                            PopUp.warning(
                                self,
                                "Developer Mode Expired",
                                "Developer Mode has expired and this data capture will now be encrypted.\n"
                                + 'An admin must renew or disable "Developer Mode" to suppress this warning.',
                            )

                        if UserProfiles.count() > 0 and enabled == False:
                            # create a protected archive
                            zf.setpassword(hashlib.sha256(zf.comment).hexdigest().encode())
                        else:
                            zf.setencryption(None)
                            if enabled:
                                Log.w("Developer Mode is ENABLED - NOT encrypting ZIP file")

                        zf.write(copy_file, arcname=archive_file)
                        if archive_file.endswith(".csv"):
                            zf.writestr(crc_file, str(hex(zf.getinfo(archive_file).CRC)))

                    os.remove(copy_file)

                if ask_for_info:
                    ask_for_info = False
                    self.indicate_finalizing()
                    self.bThread.append(QtCore.QThread())
                    user_name = (
                        None if self.parent is None else self.parent.ControlsWin.username.text()[6:]
                    )
                    # TODO: more secure to pass user_hash (filename)
                    self.bWorker.append(
                        QueryRunInfoWidget(
                            run_directory,
                            new_run_path,
                            is_good,
                            user_name,
                            parent=self.parent,
                        )
                    )
                    self.bThread[-1].started.connect(self.bWorker[-1].show)
                    self.bWorker[-1].finished.connect(self.bThread[-1].quit)
                    # add here
                    self.bWorker[-1].finished.connect(self.indicate_done)
                    # self.finished.disconnect(self.indicate_done) # remove here

            num_runs_saved = len(self.bThread)
            for i in range(num_runs_saved):
                # if '1' more fields shown in QueryRunInfo
                self.bWorker[i].setRuns(num_runs_saved, i)
            if num_runs_saved == 0:
                pass  # do nothing
            elif num_runs_saved == 1:
                self.bThread[-1].start()  # only 1 run to save
            else:
                self.RunInfoWindow = RunInfoWindow(
                    self.bWorker, self.bThread
                )  # more than 1 run to save

        except:
            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        finally:
            if os.path.exists(Constants.new_files_path):
                os.remove(Constants.new_files_path)
            self.finished.emit()

    def _portIDfromIndex(self, pid):  # convert ASCII byte to character
        # For 4x1 system: expect pid 1-4, return "1" thru "4"
        # For 4x6 system: expect pid 0xA1-0xD6, return "A1" thru "D6"
        return hex(pid)[2:].upper()
