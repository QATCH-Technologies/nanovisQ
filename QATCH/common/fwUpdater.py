from QATCH.ui.popUp import PopUp
from QATCH.core.constants import Constants
from QATCH.common.userProfiles import UserProfiles, UserRoles
from QATCH.common.tyUpdater import QATCH_TyUpdater
from QATCH.common.logger import Logger as Log
from QATCH.common.fileStorage import FileStorage
from QATCH.common.architecture import Architecture, OSType
from serial import serialutil
from PyQt5 import QtCore, QtWidgets, QtGui
from enum import IntEnum, unique
from time import time, sleep
from subprocess import Popen, PIPE
from datetime import datetime
import mmap
import os
import shlex
import shutil
import stat
import requests

USE_PROGRESS_BAR_MODULE = False
if USE_PROGRESS_BAR_MODULE:
    from progressbar import Bar, Percentage, ProgressBar, RotatingMarker, Timer


if Constants.serial_simulate_device:
    from QATCH.processors.Simulator import serial  # simulator
else:
    from QATCH.processors.Device import serial  # real device hardware


TAG = "[FW_Updater]"


@unique
class FW_UPDATE(IntEnum):
    RESULT_REQUIRED = -3
    RESULT_FAILED = -2
    RESULT_UNKNOWN = -1
    RESULT_NONE = 0
    RESULT_UPTODATE = 1
    RESULT_OUTDATED = 2


@unique
class HW_TYPE(IntEnum):
    UNKNOWN = -1
    TEENSY36 = 1
    TEENSY41 = 2

    def parse(value):
        try:
            return HW_TYPE[value.translate({ord(c): None for c in "_ -"})]
        except:
            return HW_TYPE.UNKNOWN


###############################################################################
# Handles checking and updating an QATCH device firmware on Teensy 3.6 boards
###############################################################################
class FW_Updater:

    _check = True
    _serial = serial.Serial()
    _hw = HW_TYPE.UNKNOWN

    _active_uid = None
    _port_changed = False
    _port = None

    # Flag indicating the presence of a [TRANSIENT] error in the error message.
    # Errors are suppressed in error popup but collected for Device Info.
    transient_err_cnt = 0

    ###########################################################################
    # Check for firmware update and (if user agrees) push update to the device
    ###########################################################################
    def run(self, parent, askPermission=True):
        """
        :param port: Serial port name :type port: str.
        """
        result = FW_UPDATE.RESULT_NONE
        userResponse = None
        abort = False

        if isinstance(parent.worker._port, list):
            num_ports = len(parent.worker._port)
        else:
            num_ports = 1

        i = 0
        while i < num_ports:  # for i in range(num_ports):
            if num_ports == 1:
                self._serial.port = parent.worker._port
            else:
                self._serial.port = parent.worker._port[i]
            i += 1  # increment for next device

            port = self._serial.port

            if self._check:
                success = self.open(port)
                if not success:
                    return FW_UPDATE.RESULT_FAILED

                result, version, target, written, abort = self.checkUpdate(
                    parent, port)

                if not result == FW_UPDATE.RESULT_FAILED:
                    if not result == FW_UPDATE.RESULT_UPTODATE:
                        # re-import each time to update settings from file
                        from QATCH.common.userProfiles import UserConstants

                        check_result = True
                        if UserConstants.REQ_ADMIN_UPDATES:
                            action_role = UserRoles.ADMIN
                            check_result = UserProfiles().check(
                                parent.ControlsWin.userrole, action_role
                            )
                        if askPermission and check_result == True:
                            question = "Device is running "
                            if result == FW_UPDATE.RESULT_REQUIRED:
                                question += "INCOMPATIBLE"
                            if result == FW_UPDATE.RESULT_UNKNOWN:
                                question += "UNKNOWN"
                            if result == FW_UPDATE.RESULT_OUTDATED:
                                question += "OUTDATED"
                            if False:  # self._serial.com_port == None:
                                question += " firmware.\nPlease connect PC to the USB service port to update firmware."
                                PopUp.critical(
                                    parent,
                                    "Firmware Update Available",
                                    question,
                                    "Running FW: {} ({})\nRecommended: {}\n\n".format(
                                        version, self._hw.name, target
                                    )
                                    + "Updating only takes a few seconds and guarantees operational compatibility.",
                                )
                                doUpdate = False
                            else:
                                question += (
                                    " firmware.\nWould you like to update it now?"
                                )
                                if userResponse == None:
                                    userResponse = PopUp.question_FW(
                                        parent,
                                        "Firmware Update Available",
                                        question,
                                        "Running FW: {} ({})\nRecommended: {}\n\n".format(
                                            version, self._hw.name, target
                                        )
                                        + "Updating only takes a few seconds and guarantees operational compatibility.",
                                    )
                                doUpdate = userResponse  # remember their answer for multiplex devices
                        elif check_result != True:
                            Log.w(
                                "A firmware update is available! Please ask your administrator to install update."
                            )
                            return True  # skip update checks until logged in as ADMIN role, allow user to continue
                        else:
                            doUpdate = parent.ReadyToShow

                        if doUpdate:
                            parent.ControlsWin.ui1.infobar.setText(
                                "<font color=#0000ff> Infobar </font><font color={}>{}</font>".format(
                                    "#333333",
                                    "Programming device firmware... please wait...",
                                )
                            )
                            parent.ControlsWin.ui1.infobar.repaint()

                            # # They said "YES" or we did not ask, attempt the update
                            multistep = (
                                f" ({i} of {num_ports})" if num_ports > 1 else ""
                            )
                            self.progressBar = QtWidgets.QProgressDialog(
                                f"Programming FW {target}...", "Cancel", 0, 100, parent
                            )
                            icon_path = os.path.join(
                                Architecture.get_path(), "QATCH/icons/download_icon.ico"
                            )
                            self.progressBar.setWindowIcon(
                                QtGui.QIcon(icon_path))
                            self.progressBar.setWindowTitle(
                                f"Programming FW {target}{multistep}"
                            )
                            self.progressBar.setWindowFlag(
                                QtCore.Qt.WindowContextHelpButtonHint, False
                            )
                            self.progressBar.setWindowFlag(
                                QtCore.Qt.WindowStaysOnTopHint, True
                            )
                            self.progressBar.canceled.disconnect()
                            self.progressBar.canceled.connect(
                                self.update_cancel)
                            self.progressBar.setFixedSize(
                                int(self.progressBar.width() * 1.5),
                                int(self.progressBar.height() * 1.1),
                            )

                            self.upd_thread = QtCore.QThread()
                            PRIMARY_DEV = (
                                port if num_ports == 1 else parent.worker._port[0]
                            )
                            self.updater = UpdaterTask(
                                port, self._serial, PRIMARY_DEV, self._active_uid
                            )
                            self.updater.moveToThread(self.upd_thread)
                            self.upd_finished = False

                            self.upd_thread.started.connect(self.updater.run)
                            self.updater.finished.connect(self.upd_thread.quit)
                            self.updater.finished.connect(
                                self.updater.deleteLater)
                            self.upd_thread.finished.connect(
                                self.upd_thread.deleteLater
                            )
                            self.updater.progress.connect(self.update_progress)
                            self.updater.finished.connect(self.update_finished)

                            self.upd_thread.start()
                            self.progressBar.exec_()  # wait here until finished installing

                            if not self.upd_finished:
                                # Log.w("Progress bar window was closed prematurely.")
                                # Log.e("Waiting for update process to finish...")
                                while not self.upd_finished:
                                    Log.e(
                                        TAG,
                                        "Cannot abort FW update once programming starts... please wait!",
                                    )
                                    self.progressBar.exec_()  # show it again
                                # Log.i(TAG, "Update process finished. Resuming...")

                            # self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
                            if self.updater._cancel and self._serial.hid_port == None:
                                Log.w(TAG, "Download canceled.")
                                box = QtWidgets.QMessageBox(None)
                                box.setIcon(QtWidgets.QMessageBox.Information)
                                # box.setWindowIcon(QtGui.QIcon(tempicon))
                                box.setWindowTitle("Program FW Update")
                                box.setText("Download canceled.")
                                box.setDetailedText("")
                                box.setStandardButtons(
                                    QtWidgets.QMessageBox.Ok)
                                box.exec_()

                            result, output, error = self.updater.get_results()

                            # port may have changed if COM -> HID
                            if port != self.updater._port:
                                self._port_changed = True
                                self._port = self.updater._port
                            else:
                                self._port_changed = False
                                self._port = port

                            parent.ControlsWin.ui1.infobar.setText(
                                "<font color=#0000ff> Infobar </font><font color={}>{}</font>".format(
                                    "#333333", ""
                                )
                            )

                            tryAgain = False
                            if result == FW_UPDATE.RESULT_UPTODATE:
                                check, now, _, _, _ = self.checkUpdate(
                                    parent, self._port
                                )
                                if check == FW_UPDATE.RESULT_UPTODATE:
                                    # Only indicate "success" after all active ports are finished
                                    if i == num_ports:
                                        # Update successful
                                        PopUp.question_FW(
                                            parent,
                                            "Firmware Update Successful",
                                            "Device is now running the latest firmware.",
                                            "Updated from {} to {}.\n\nDETAILS:\n{}".format(
                                                version, now, output
                                            ),
                                            True,
                                        )
                                else:
                                    # Verification Failed (ask to try again)
                                    message = (
                                        "UPDATE FAILED\n\nDevice is still running "
                                    )
                                    if check == FW_UPDATE.RESULT_REQUIRED:
                                        message += "INCOMPATIBLE"
                                    if check == FW_UPDATE.RESULT_FAILED:
                                        message += "UNKNOWN"
                                    if check == FW_UPDATE.RESULT_UNKNOWN:
                                        message += "UNKNOWN"
                                    if check == FW_UPDATE.RESULT_OUTDATED:
                                        message += "OUTDATED"
                                    message += " firmware!"
                                    if PopUp.critical(
                                        parent,
                                        "Firmware Update Failed",
                                        message,
                                        "Post-update version verification failed. (Error {})\n\n".format(
                                            check
                                        )
                                        + 'To fix this error, try modifying "best_fw_version" in Constants.py '
                                        + 'to match the currently running device firmware version: "{}".'.format(
                                            now
                                        ),
                                    ):
                                        tryAgain = True
                            elif not result == FW_UPDATE.RESULT_OUTDATED:
                                # Update failed (ask to try again)
                                if PopUp.critical(
                                    parent,
                                    "Firmware Update Failed",
                                    "UPDATE FAILED\n\nA problem was encountered while attempting to update the device firmware.",
                                    "If the problem persists, please update manually.\n\nDETAILS:\n"
                                    + error.replace(
                                        "Please try again",
                                        "Please close the other app(s) and try again.",
                                    ),
                                ):
                                    tryAgain = True

                            if tryAgain == True:
                                i -= 1  # decrement for next device
                                continue  # restart while loop

                        else:
                            # They said "NO", so mark it as "outdated" and return
                            # NOTE: If we allowed "unknown" to return we would ask again (undesireable)
                            if result == FW_UPDATE.RESULT_UNKNOWN:
                                result = FW_UPDATE.RESULT_OUTDATED

                        # Do not ask again if we were successful or told "NO"...
                        if result >= 0 and i == num_ports:
                            self._check = False
                    else:
                        if i == num_ports:
                            self._check = False
                else:
                    # check version failed
                    Log.w(
                        TAG,
                        "WARNING: Attempt to read device firmware and hardware versions failed. Skipping update check.",
                    )

                self.close()

                if written:  # Device Info file was written (new or update)
                    parent._refresh_ports()

            if result == FW_UPDATE.RESULT_REQUIRED:
                if PopUp.critical(
                    parent,
                    "Firmware Update Required",
                    "Device is running INCOMPATIBLE firmware.\n\n"
                    + "You MUST update the device to use this software version. Press 'retry' to update.",
                ):
                    i -= 1  # decrement for next device
                    continue  # restart while loop
                return False
            if abort:
                return False
        return False if abort else True

    ###########################################################################
    # Configure the serial port
    ###########################################################################

    def open(self, port):
        """
        :param port: Serial port name :type port: str.
        """
        # Configure serial port (assume baud to check before update)
        self._serial.port = port
        self._serial.baudrate = Constants.serial_default_speed  # 115200
        self._serial.stopbits = serial.STOPBITS_ONE
        self._serial.bytesize = serial.EIGHTBITS
        self._serial.timeout = Constants.serial_timeout_ms
        self._serial.write_timeout = Constants.serial_writetimeout_ms

        # Attempt to open port and print errors (if any)
        try:
            self._serial.open()
            return self._serial.is_open
        except:
            Log.e(
                TAG,
                "ERROR: Failure opening port to check and/or update device firmware.",
            )
            return False

    ###########################################################################
    # Close the serial port
    ###########################################################################

    def close(self):
        if self._serial.is_open:
            self._serial.close()

    ###########################################################################
    # Parent informs self to check again on next run (port changed)
    ###########################################################################

    def checkAgain(self):
        self._check = True

    ###########################################################################
    # Checks the current version and indicates whether an update is Recommended
    ###########################################################################

    def checkUpdate(self, parent, port):
        """
        :param port: Serial port name :type port: str.
        """
        if not self._serial.is_open:
            self.open(port)

        version = "COULD NOT CHECK"  # default
        result = FW_UPDATE.RESULT_UNKNOWN
        written = False
        abort_action = False

        if self._serial.is_open:
            try:
                # Make sure device is not streaming before starting!!!
                self._serial.write("stop\n".encode())
                # wait for "STOP" reply
                stopped = 0
                stop = time()
                waitFor = 3  # timeout delay (seconds)
                while time() - stop < waitFor:
                    while time() - stop < waitFor and self._serial.in_waiting == 0:
                        pass
                    if time() - stop < waitFor:
                        while not self._serial.read(1).hex() == "53":  # "S"
                            pass
                        if self._serial.read(1).hex() == "54":  # "T"
                            if self._serial.read(1).hex() == "4f":  # "O"
                                if self._serial.read(1).hex() == "50":  # "P"
                                    if self._serial.read(1).hex() == "0d":  # "\r"
                                        if self._serial.read(1).hex() == "0a":  # "\n"
                                            stopped = 1
                                            break
                if stopped == 0:
                    Log.w(
                        TAG,
                        "WARNING: Device did not respond to STOP command. May still be streaming or running old FW...",
                    )

                # Read and show the FW version from the device
                self._serial.write("version\n".encode())
                timeoutAt = time() + 3
                while (
                    self._serial.in_waiting == 0 and time() < timeoutAt
                ):  # timeout needed if old FW
                    pass
                if time() < timeoutAt:
                    version_reply = (
                        self._serial.read(
                            self._serial.in_waiting).decode().split("\n")
                    )
                    length = len(version_reply)
                    if length >= 3:
                        build = version_reply[0].strip()
                        version = version_reply[1].strip()
                        date = version_reply[2].strip()
                        Log.d(
                            "SERIAL BUILD INFORMATION\n FW Device: {}\n FW Version: {}\n FW Date: {}\n".format(
                                build, version, date
                            )
                        )
                        branch = Constants.best_fw_version[0:4]
                        if not branch in version:
                            Log.w(
                                "WARNING: Device is running outdated firmware ({})!\n You MUST upgrade to continue... (Recommended: {})\n".format(
                                    version, Constants.best_fw_version
                                )
                            )
                            result = FW_UPDATE.RESULT_REQUIRED
                        elif not version == Constants.best_fw_version:
                            Log.w(
                                "WARNING: Device is running outdated firmware ({})!\n Please consider upgrading... (Recommended: {})\n".format(
                                    version, Constants.best_fw_version
                                )
                            )
                            result = FW_UPDATE.RESULT_OUTDATED
                        else:
                            Log.i(
                                "STATUS: Device is running up-to-date firmware ({})!\n".format(
                                    version, Constants.best_fw_version
                                )
                            )
                            result = FW_UPDATE.RESULT_UPTODATE
                    else:
                        # issue reading port
                        Log.w(
                            TAG,
                            "WARNING: Cannot read device firmware version. Incorrect response.",
                        )
                        version = "UNKNOWN"
                        result = FW_UPDATE.RESULT_UNKNOWN
                else:
                    # timeout reading port
                    Log.w(
                        TAG,
                        "WARNING: Cannot read device firmware version. No response from device.",
                    )
                    version = "UNKNOWN"
                    result = FW_UPDATE.RESULT_UNKNOWN

                # Read and show the HW type from the device
                self._serial.write("info\n".encode())
                timeoutAt = time() + 3
                info_reply = b""
                while (
                    self._serial.in_waiting == 0 and time() < timeoutAt
                ):  # timeout needed if old FW
                    pass
                while time() < timeoutAt:
                    if self._serial.in_waiting != 0:
                        next_line = self._serial.read_until()
                    else:
                        break
                    info_reply += next_line

                # Generate the path to the device info file from the given parameters
                if time() < timeoutAt:
                    # info_reply = self._serial.read(self._serial.in_waiting).decode().split('\n')
                    info_reply = info_reply.decode().split("\n")
                    length = len(info_reply)

                    hw = ip = uid = mac = usb = pid = rev = err = None
                    for line in info_reply:
                        line = line.split(":", 1)
                        if line[0] == "HW":
                            hw = line[1].strip()
                        if line[0] == "IP":
                            ip = line[1].strip()
                        if line[0] == "UID":
                            uid = line[1].strip()
                        if line[0] == "MAC":
                            mac = line[1].strip()
                        if line[0] == "USB":
                            usb = line[1].strip()
                        if line[0] == "PID":
                            pid = line[1].strip()
                        if line[0] == "REV":
                            rev = line[1].strip()
                        if line[0] == "ERR":
                            # Presuming line[1] contains the remaining error message and does not need to be
                            # parsed further, simply indicate a transient error is present and suppress the message
                            # later.
                            # If/when the transient error is cleared or another one appears instead, reset the flag
                            if "[TRANSIENT]" in line[1].strip():
                                self.transient_err_cnt += 1
                            else:
                                self.transient_err_cnt = 0
                            err = line[1].strip()
                    Log.d(TAG, "Detected HW TYPE is {}.".format(hw))
                    self._hw = HW_TYPE.parse(hw)
                    # store USB for port change detection
                    self._active_uid = usb
                    # Write device info file (if needed)
                    path = Constants.csv_calibration_export_path
                    self._port = port
                    if self._port == None:
                        self._com = None
                        self._net = None
                    elif ":" in self._port:
                        self._com = self._port
                        self._net = None
                    elif ";" in self._port:
                        self._com = self._port.split(";")[0]
                        self._net = self._port.split(";")[1]
                    elif self._port.count(".") == 3:
                        self._com = None
                        self._net = self._port
                    else:
                        self._com = self._port
                        self._net = None
                    name = self.__getDeviceName__(usb, self._com)
                    idx = 0 if pid in [None, "FF"] else int(pid, base=16) % 9
                    # write dev info file after conflict resolution

                    # check for and resolve any conflicting device infos
                    try:
                        if FileStorage.DEV_get_active(idx) == "" and idx > 0:
                            FileStorage.DEV_set_active(
                                idx, name)  # for multiplex
                        dev_folder = "{}_{}".format(
                            idx, name) if idx > 0 else name
                        # wrong dev selected, must be a conflict
                        if FileStorage.DEV_get_active(idx) != dev_folder:
                            device_list = FileStorage.DEV_get_device_list()
                            for i, dev_name in device_list:
                                dev_info = FileStorage.DEV_info_get(
                                    i, dev_name)
                                if "USB" in dev_info and "PORT" in dev_info:
                                    if (
                                        dev_info["PORT"] == self._com
                                        and dev_info["USB"] != usb
                                    ):
                                        _name = _fw = _hw = _port = _ip = _uid = (
                                            _mac
                                        ) = _usb = _pid = _rev = _err = None
                                        for entry in dev_info.keys():
                                            if entry == "NAME":
                                                _name = dev_info[entry]
                                            if entry == "FW":
                                                _fw = dev_info[entry]
                                            if entry == "HW":
                                                _hw = dev_info[entry]
                                            if entry == "PORT":
                                                # mark as conflicted
                                                _port = f"{dev_info[entry]}!"
                                            if entry == "IP":
                                                _ip = dev_info[entry]
                                            if entry == "UID":
                                                _uid = dev_info[entry]
                                            if entry == "MAC":
                                                _mac = dev_info[entry]
                                            if entry == "USB":
                                                _usb = dev_info[entry]
                                            if entry == "PID":
                                                _pid = dev_info[entry]
                                            if entry == "REV":
                                                _rev = dev_info[entry]
                                            if entry == "ERR":
                                                _err = dev_info[entry]

                                        FileStorage.DEV_info_set(
                                            i,
                                            Constants.txt_device_info_filename,
                                            path,
                                            _name,
                                            _fw,
                                            _hw,
                                            port=_port,
                                            ip=_ip,
                                            uid=_uid,
                                            mac=_mac,
                                            usb=_usb,
                                            pid=_pid,
                                            rev=_rev,
                                            err=_err,
                                        )
                    except:
                        Log.e(
                            "Conflicting device info could not be automatically resolved. Please correct 'config' manually."
                        )

                    # write dev info file now
                    written = FileStorage.DEV_info_set(
                        idx,
                        Constants.txt_device_info_filename,
                        path,
                        name,
                        version,
                        self._hw.name,
                        port=self._com,
                        ip=ip,
                        uid=uid,
                        mac=mac,
                        usb=usb,
                        pid=pid,
                        rev=rev,
                        err=err,
                    )

                    # Modified conditional to not display pop-up if transient error appeared in message.
                    if (
                        err not in {None, "NONE"}
                        and self.transient_err_cnt <= 1
                        and parent.ReadyToShow
                    ):
                        if PopUp.critical(
                            parent,
                            "Hardware Error Detected",
                            "<b>SERVICE REQUIRED</b>: HARDWARE ERROR DETECTED!<br/>"
                            + "It is not recommended to use this device until serviced.",
                            details=f"Error Detected:\n{err}",
                            btn1_text="Ok",
                        ):
                            abort_action = True
                        # keep showing this error for each action taken
                        QtCore.QTimer.singleShot(500, self.checkAgain)
                else:
                    # timeout reading port
                    Log.w(
                        TAG,
                        "WARNING: Cannot read device hardware info. Assuming legacy FW running on TEENSY36.",
                    )
                    self._hw = HW_TYPE.TEENSY36
                    # Set active device pointer ONLY (DO NOT SAVE INFO)
                    name = self.__getDeviceName__(None, port)
                    FileStorage.DEV_set_active(0, name)
                if self._hw == HW_TYPE.UNKNOWN:
                    # detected unknown hardware response, cannot proceed so skip it this time
                    Log.w(
                        TAG,
                        "WARNING: Received unexpected hardware info reply. Please enter HW type (if prompted)...",
                    )
                    # result = FW_UPDATE.RESULT_FAILED

            except:
                version = "FAILED TO READ FROM DEVICE"
                # if reading version causes error, just skip it this time
                result = FW_UPDATE.RESULT_FAILED
            finally:
                self.close()

        return result, version, Constants.best_fw_version, written, abort_action

    ###########################################################################
    # Translate 'INFO' reply into a consistent device name across platforms
    ###########################################################################

    def __getDeviceName__(self, preferred, backup):
        """
        :param preferred: USB serial number from device INFO
        :param backup: Port name from filesystem (OS-specific)
        """
        if not preferred == None:
            if not preferred.isnumeric():
                Log.w(TAG, "WARN: Preferred device name is not numeric!")
            return preferred  # INFO CMD supported, normal operation
        if not backup == None:
            if backup.startswith("COM"):
                return backup  # INFO CMD failed, running WINDOWS OS
            else:
                # INFO CMD failed, running non-WINDOWS OS
                numeric_filter = filter(str.isdigit, backup)
                backup_digits = "".join(numeric_filter)
                return backup_digits[0:-1]  # all digits in backup (but last)

    def update_progress(self, label_str=None, progress_pct=None):
        try:
            need_repaint = False
            if progress_pct == self.progressBar.maximum():
                return  # ignore this signal, move on to "programming" mode
            if label_str != None and self.updater._cancel == False:
                # need_repaint = True
                self.progressBar.setLabelText(label_str)
            if progress_pct != None:
                # need_repaint = True
                self.progressBar.setValue(progress_pct)
            if self.progressBar.labelText().find("Cancel") >= 0:
                return
            if isinstance(label_str, str):
                if label_str.find("Wait") >= 0:
                    return
            if self._serial.hid_port != None:
                return

            curr_val = self.progressBar.value()
            if (
                curr_val == 99 and self.progressBar.labelText().find("Transfer") >= 0
            ) or self.progressBar.labelText().find("Transfer") == -1:
                need_repaint = True
                delay = 500
                if self.progressBar.labelText().find("Transfer") >= 0:
                    cancelButton = self.progressBar.findChild(
                        QtWidgets.QPushButton)
                    cancelButton.setEnabled(False)
                    status_str = "Programming device firmware...<br/><b>DO NOT POWER CYCLE DEVICE!</b>"
                    self.progressBar.setLabelText(status_str)
                    curr_val = 0
                if self.upd_finished:  # or int(curr_val / 10) % 2 == 0:
                    status_str = "Checking device firmware version..."
                    self.progressBar.setLabelText(status_str)
                    delay = 50
                elif curr_val == 99:
                    curr_val -= 1  # prevent ending until updater is finished
                self.progressBar.setValue(curr_val + 1)
                if curr_val + 1 != self.progressBar.maximum():
                    QtCore.QTimer.singleShot(delay, self.update_progress)
            if need_repaint:
                self.progressBar.repaint()
        except Exception as e:
            limit = None
            import sys

            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                print(line)

    def update_finished(self):
        self.upd_finished = True
        if self.updater._cancel or self._serial.hid_port != None:
            # Log.w("Closing bar after cancel.")
            self.progressBar.close()

    def update_cancel(self):
        cancelButton = self.progressBar.findChild(QtWidgets.QPushButton)
        cancelButton.setEnabled(False)
        if self._serial.hid_port == None:
            # Log.d("GUI: Toggle progress mode")
            self.update_progress("Canceling installation...", 99)
            self.updater.cancel()
        # self.updater.wait()
        # self.progressBar.close()


class UpdaterTask(QtCore.QThread):
    finished = QtCore.pyqtSignal()
    # exception = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str, int)
    _cancel = False

    def __init__(self, port, serial, primary, active_uid):
        super().__init__()
        self._port = port
        self._serial = serial
        self._primary = primary
        self._active_uid = active_uid
        self._hw = HW_TYPE.TEENSY41  # Teensy 3.6 not supported
        self._result = FW_UPDATE.RESULT_UNKNOWN
        self._output = ""
        self._error = "ERROR: No result to give."

    def cancel(self):
        # Log.d("GUI: Toggle progress mode")
        # Log.w("Process kill request")
        self._cancel = True
        # self.progressTaskHandle.terminate()
        # self.progressTaskHandle.kill()
        # self._dbx_connection.close() # force abort of active file download

    def run(self):
        self._result, self._output, self._error = self.doUpdate(
            None, self._port)
        self.finished.emit()

    def get_results(self):
        return self._result, self._output, self._error

    ###########################################################################
    # Configure the serial port
    ###########################################################################

    def open(self, port):
        """
        :param port: Serial port name :type port: str.
        """
        # Configure serial port (assume baud to check before update)
        self._serial.port = port
        self._serial.baudrate = Constants.serial_default_speed  # 115200
        self._serial.stopbits = serial.STOPBITS_ONE
        self._serial.bytesize = serial.EIGHTBITS
        self._serial.timeout = Constants.serial_timeout_ms
        self._serial.write_timeout = Constants.serial_writetimeout_ms

        # Attempt to open port and print errors (if any)
        try:
            self._serial.open()
            return self._serial.is_open
        except:
            Log.e(
                TAG,
                "ERROR: Failure opening port to check and/or update device firmware.",
            )
            return False

    ###########################################################################
    # Close the serial port
    ###########################################################################

    def close(self):
        if self._serial.is_open:
            self._serial.close()

    ###########################################################################
    # Updates the running Teensy 3.6 device firmware to the Recommended version
    ###########################################################################

    def doUpdate(self, parent, port):
        """
        :param port: Serial port name :type port: str.
        """

        if self._port != self._primary:
            self.open(self._primary)
            # for LCD on primary multiplex device
            self._serial.write("program\n".encode())
            self.close()

        if self._serial.is_open == False and port != None:
            self.open(port)

        output = ""
        error = "ERROR: The port is closed."  # default error
        result = FW_UPDATE.RESULT_UNKNOWN

        if self._hw == HW_TYPE.UNKNOWN or port == None:
            self._hw = HW_TYPE.TEENSY41  # default (don't ask)
            # force legacy CLI (but these files don't exist anymore)
            self.do_legacy_updates = True
            # if PopUp.question(parent, "Enter Teensy HW Type", "Is this hardware a TEENSY36 device?"):
            #     self._hw = HW_TYPE.TEENSY36
            # elif PopUp.question(parent, "Enter Teensy HW Type", "Is this hardware a TEENSY41 device?"):
            #     self._hw = HW_TYPE.TEENSY41
            # else:
            #     PopUp.warning(parent, "Unknown Teensy HW Type", "Unknown hardware device. Cannot proceed.\nPlease check HW type and try again.")
            #     return result, output, error

        # parent.ControlsWin.ui1.infobar.setText("<font color=#0000ff> Infobar </font><font color={}>{}</font>".format("#333333","Programming device firmware... please wait..."))
        # parent.ControlsWin.ui1.infobar.repaint()

        if self._serial.is_open == True or port == None:
            # NOTE: LEGACY UPDATES NO LONGER SUPPORTED (FLASHERX BOOTLOADER REQUIRED)
            # if Constants.do_legacy_updates:
            #     return doUpdate_legacy(parent, port)

            try:
                # Log.d("GUI: Force repaint events")

                # Find path where FW image should be
                basepath = Architecture.get_path()
                folder_name = "QATCH_Q-1_FW_py_" + Constants.best_fw_version.strip()
                for fname in os.listdir(basepath):
                    path = os.path.join(basepath, fname)
                    if os.path.isdir(path):
                        if folder_name in fname:
                            folder_name = fname
                            break
                # Construct path to HEX image (from found folder name)
                path_to_hex = os.path.join(
                    basepath, folder_name, f"{folder_name}.ino.{self._hw.name}.hex"
                )
                expected = sum(1 for _ in open(path_to_hex))

                f = open(path_to_hex, "rb")
                Log.i(TAG, "Programming device firmware...")
                Log.w(TAG, "DO NOT DISCONNECT YOUR DEVICE!")
                # placeholder for progress bar, drawn later
                Log.i(TAG, "Status:")
                # Log.d("GUI: Toggle progress mode")

                if self._serial.net_port != None:
                    Log.d(
                        TAG, f"QUERY-PUT: http://{self._serial.net_port}:8080/program"
                    )

                    start = time()
                    d = f.read()
                    d += str(expected).encode()
                    d += b"\r\n"
                    r = requests.put(
                        f"http://{self._serial.net_port}:8080/program", data=d
                    )
                    output += r.text
                    if "SUCCESS" in output:
                        error = ""  # success
                    if "FAILURE" in output:
                        error = "The upgrade operation failed to complete."

                    Log.i(
                        TAG,
                        "Waiting for device to flash and reboot... (this may take a minute)",
                    )
                    stop = time()
                    waitFor = 90  # 90 secs timeout for network reboot
                    successes = 0
                    while time() - stop < waitFor:
                        sleep(1)  # try once a second, don't go crazy!
                        try:
                            r = requests.get(
                                f"http://{self._serial.net_port}:8080/version"
                            )
                            if r.ok:
                                successes += 1
                            if successes > 5:
                                break
                        except:
                            successes = 0
                            continue
                    if time() - stop >= waitFor:
                        Log.e(
                            TAG,
                            f"Failed to establish network connection within {waitFor} secs.",
                        )

                else:
                    if USE_PROGRESS_BAR_MODULE:
                        bar = ProgressBar(
                            widgets=[
                                TAG,
                                " ",
                                Bar(marker=">"),
                                " ",
                                Percentage(),
                                " ",
                                Timer(),
                            ]
                        ).start()
                        bar.maxval = expected
                    else:
                        step = 0
                        last_step = -1
                        num_steps = 20
                        scale_wide = 1
                        last_pct = -1
                    lines = 0

                    self._serial.write("program\n".encode())

                    start = time()
                    waitFor = 3  # timeout delay (seconds)
                    while time() - start < waitFor:
                        while time() - start < waitFor and self._serial.in_waiting == 0:
                            pass
                        if time() - start < waitFor:
                            reply = self._serial.read(self._serial.in_waiting)
                            output += reply.decode().lstrip()
                            if "waiting for hex lines...\n" in output:
                                break
                    if time() - start >= waitFor:
                        error = "Failed to enter bootloader."
                        raise TimeoutError(error)

                    if self._serial.com_port != None:

                        # COM port connected, use FlasherX bootloader
                        Log.d("GUI: Toggle progress mode")
                        for x in f:
                            lines += 1
                            # draw progress bar, of desired style
                            if USE_PROGRESS_BAR_MODULE:
                                bar.update(lines)
                            else:
                                step = int(lines / expected * num_steps)
                                percent = int(100 * lines / expected)
                                if step != last_step:
                                    Log.i(
                                        TAG,
                                        "Transfer: ["
                                        + ("#" * scale_wide * step)
                                        + ("_" * scale_wide * (num_steps - step))
                                        + f"] | {percent}%",
                                    )
                                    last_step = step
                                if percent != last_pct:
                                    # 100% is ignored by handler
                                    self.progress.emit(
                                        "Transferring firmware to device...", percent
                                    )
                                    last_pct = percent
                            # Log.w(f"dbg: {lines} : {x}")
                            self._serial.write(x)
                            if False:
                                reply = self._serial.read(
                                    self._serial.in_waiting)
                                # Log.d(">> {}".format(x.encode()))
                                # Log.d("<< {}".format(reply))
                                if b"abort -" in reply:
                                    # self._serial.reset_output_buffer()
                                    error = reply.decode()
                                    break
                            if self._cancel:
                                # self._serial.reset_output_buffer()
                                result = FW_UPDATE.RESULT_OUTDATED
                                error = "User canceled firmware update.\n"
                                output += error.lstrip()
                                Log.w(error.strip())
                                break
                        Log.d("GUI: Toggle progress mode")
                        f.close()

                        if not lines == expected:
                            Log.d(
                                TAG, f"incorrect # of lines sent: {lines} / {expected}"
                            )
                            lines = 0  # abort
                            if not "abort" in error:
                                error = "Incorrect number of lines sent to device."

                        self._serial.write(f"{lines}\n".encode())
                        # Log.d(">> {}".format(f"{lines}\n".encode()))
                        if USE_PROGRESS_BAR_MODULE:
                            bar.finish()

                        Log.i(
                            TAG,
                            "Waiting for device to flash and reboot... (this may take a minute)",
                        )
                        stop = time()
                        waitFor = 45  # copy to RAM can take some time
                        while True:
                            try:
                                reply = self._serial.read(
                                    self._serial.in_waiting)
                                output += reply.decode().lstrip()
                            except Exception as e:
                                Log.d(
                                    "Port read error. Device appears to have rebooted.",
                                    type(e),
                                )
                                error = ""  # assume success so far
                                break
                            if output.strip().endswith("SUCCESS"):
                                error = ""  # success
                                continue  # wait for port to close and reboot
                            if "abort -" in output:
                                error = output.strip().split("\n")[-1]
                                break
                            if time() - stop > waitFor:
                                error = "Device failed to reboot at end of update."
                                break

                    else:

                        # HID port connected, use TyCmd and Teensy_Loader_CLI EXEs

                        f.close()
                        if USE_PROGRESS_BAR_MODULE:
                            bar.finish()

                        # Pass the update responsibilities to TyUpdater!
                        dev_sernum = self._serial.hid_port.split(":")[0]
                        tyUpdater = QATCH_TyUpdater(self.progress)
                        error = tyUpdater.update(dev_sernum, path_to_hex)

                    stop = time()
                    if error == "" or self._cancel:
                        try:
                            stop = time()
                            waitFor = 15  # timeout delay (seconds)
                            # Log.w(TAG, "Waiting for device reset...")
                            while (
                                time() - stop < waitFor and self._serial.in_waiting >= 0
                            ):  # on reboot, port will disappear, triggering a SerialException here
                                Log.w(TAG, "Waiting for device reset...")
                                # exit bootloader, if still stuck in it
                                self._serial.write("\n".encode())
                        except serialutil.SerialException:
                            self.progress.emit(
                                "Waiting for device to reboot...", None)
                            Log.d(TAG, "Device reset!")

                        if self._port != self._primary:
                            self.open(self._primary)
                            # for LCD on primary multiplex device
                            self._serial.write("\n".encode())
                            self.close()

                        # give time for port to re-establish and initial boot to occur
                        sleep(6)

                        # Check if port changed (COM -> HID) and update port, if need be
                        all_ports = serial.enumerate()
                        for p in all_ports:
                            if p == port:
                                Log.d("Port did not change after updating.")
                                break  # port did not change
                            if self._active_uid == None:
                                Log.w(
                                    "No active USB device. Cannot re-detect port, if changed. Assuming it did not change."
                                )
                                break  # cannot detect change
                            if p.startswith(self._active_uid):
                                Log.d(
                                    "Port changed after updating. Using new port for post-checks."
                                )
                                Log.d(f"port = {p}")
                                port = p
                                break  # port change detected!
                        self._port = port  # save to return with get_results()

                        # Perform post update version check
                        self.open(port)
                        self._serial.write("version\n".encode())

                        stop = time()
                        waitFor = 60  # timeout delay (seconds)
                        msg = ""
                        while time() - stop < waitFor:
                            while (
                                time() - stop < waitFor and self._serial.in_waiting == 0
                            ):
                                pass
                            if time() - stop < waitFor:
                                reply = self._serial.read(
                                    self._serial.in_waiting)
                                msg += reply.decode()
                                if "QATCH" in msg:
                                    sleep(1)
                                    self._serial.reset_input_buffer()
                                    break
                        if time() - stop >= waitFor:
                            error = "Failed to reboot within one minute."
                            raise TimeoutError(
                                "Failed to reboot within one minute.")

                # common code for both COM and NET devices
                status = f"Target build: {os.path.split(path_to_hex)[1]}\n"
                status += f"Image size:   {expected} lines\n"
                status += f"Connection:   {port}\n"
                status += f"Time to TX:   {stop - start} secs\n"
                status += f"Time to BOOT: {time() - stop} secs\n"
                output = status + output.lstrip()

                if error == "":
                    Log.i(TAG, "Device firmware programmed successfully!")
                    result = FW_UPDATE.RESULT_UPTODATE  # SUCCESS!
                elif not self._cancel:
                    if error == "":
                        error = output
                    Log.e(TAG, "ERROR: {}".format(error))
                    result = FW_UPDATE.RESULT_FAILED  # ERROR!
            except Exception as e:
                result = FW_UPDATE.RESULT_FAILED
                Log.e(
                    TAG,
                    "ERROR: Failure programming and/or rebooting device to update device firmware.",
                )

                limit = None
                import sys

                t, v, tb = sys.exc_info()
                from traceback import format_tb

                a_list = ["Traceback (most recent call last):"]
                a_list = a_list + format_tb(tb, limit)
                a_list.append(f"{t.__name__}: {str(v)}")
                for line in a_list:
                    print(line)

            finally:
                self.close()
                # Log.d("GUI: Normal repaint events")
                # Log.d("GUI: Toggle progress mode")

        # parent.ControlsWin.ui1.infobar.setText("<font color=#0000ff> Infobar </font><font color={}>{}</font>".format("#333333",""))

        # Append results to the log file
        path_to_log = os.path.join(os.getcwd(), Constants.log_export_path)
        f = open(os.path.join(path_to_log, "FlasherX.log"), "a")
        f.write(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + "\n")
        f.write(output + error + "\n\n")
        f.close()

        return result, output, error

    def doUpdate_legacy(self, parent, port):
        if True:
            try:
                # Check to make sure the teensy loader CLI (or other conflicts) are not Running

                # Command is based on OS running
                if Architecture.get_os() is OSType.windows:
                    prereq_task = Popen(
                        [
                            'tasklist /FI "IMAGENAME eq Teensy*" 2>NUL | find /I /C "Teensy">NUL'
                        ],
                        shell=True,
                        stdout=PIPE,
                        stderr=PIPE,
                    )
                else:
                    prereq_task = Popen(
                        ["ps aux | grep teensy -c | awk '$1<=\"2\" {exit 1}'"],
                        shell=True,
                        stdout=PIPE,
                        stderr=PIPE,
                    )

                # Both forms of the prereq command return exitcode 1 when successful
                # so if exitcode is NOT 1 then prereq conditions were NOT satisfied!
                if not prereq_task.wait() == 1:
                    error = 'ABORT: "Teensy" app(s) already running!\nPlease try again'

                # Prereqs satisfied, proceed to flashing
                else:
                    # Find path where FW image should be
                    basepath = Architecture.get_path()
                    folder_name = "QATCH_Q-1_FW_py_" + Constants.best_fw_version.strip()
                    for fname in os.listdir(basepath):
                        path = os.path.join(basepath, fname)
                        if os.path.isdir(path):
                            if folder_name in fname:
                                folder_name = fname
                                break
                    # Construct path to HEX image (from found folder name)
                    path_to_hex = '"{1}{0}{2}{0}{2}.ino.{3}.hex"'.format(
                        Constants.slash, basepath, folder_name, self._hw.name
                    )
                    teensy_loader_cwd = "{1}{0}teensy_loader_cli{0}bin{0}".format(
                        Constants.slash, basepath
                    )

                    # Allow protected EXEs to be executed
                    self.__unpackFiles__(teensy_loader_cwd)

                    # Invoke the teensy loader CLI (with shell involvement) and pass its output streams through.
                    # run()'s return value is an object with information about the completed process.
                    command_line = '".{0}teensy_loader_cli" --mcu={1} -w -v {2}'.format(
                        Constants.slash, self._hw.name, path_to_hex
                    )

                    # Windows needs parameters to be split
                    # Mac only works if they are NOT split
                    if not Architecture.get_os() is OSType.macosx:
                        command_line = shlex.split(command_line)

                    update_task = Popen(
                        command_line,
                        cwd=teensy_loader_cwd,
                        shell=True,
                        stdout=PIPE,
                        stderr=PIPE,
                    )

                    Log.d(TAG, "Waiting for program...")
                    sleep(3)  # give script time to load in the background

                    # If the program bailed early there will be a returncode, so stop here
                    # If it is still None then the program is waiting for PROGRAM MODE...
                    if update_task.poll() is None:
                        # Start with trying "PROGRAM" cmd, then try "secret sauce" as backup

                        Log.d(TAG, "Programming device firmware...")
                        # Log.d(TAG, "Reply: {}".format(reply))

                        if self._serial.is_open and port != None:
                            deviceReset = False
                            if self._hw == HW_TYPE.TEENSY36:
                                # Preferred method for TEENSY36
                                # Not supported for TEENSY41
                                # Attempt to reboot device and print errors (if any)
                                # if device responds to cmd it will close the port (resulting in a "good" exception)
                                try:
                                    self._serial.write("program\n".encode())
                                    timeoutAt = time() + 1
                                    while (
                                        self._serial.in_waiting == 0
                                        and time() < timeoutAt
                                    ):  # timeout needed if old FW
                                        pass
                                    deviceReset = False
                                except:
                                    deviceReset = True
                            if (
                                not deviceReset
                            ):  # preferred method not supported by device firmware, try backup method
                                # Preferred method for TEENSY41
                                # Backup method for TEENSY36
                                # Attempt to reboot device and throw errors (if any)
                                self._serial.close()
                                self._serial.baudrate = 134  # secret sauce
                                self._serial.open()  # kick to bootloader
                        else:
                            # Port not open and/or provided (corrupt or virgin board, stuck in bootloader)
                            # Do the program by requiring the user to have previously pushed the PROGRAM button on Teensy HW directly
                            PopUp.information(
                                parent,
                                "FW Recovery Tool",
                                'Press the "PROGRAM" button to initiate the flash operation!',
                            )
                            pass

                    # Gather task output/error/retcode
                    output, error = update_task.communicate()
                    retcode = update_task.wait()  # wait for exit (up to 10 seconds)
                    output = output.decode().strip()
                    error = error.decode().strip()
                    # Log.d(TAG, "RETCODE = {}".format(retcode))

                    # Remove executables so the project can be sent as an attachment
                    self.__removeFiles__(teensy_loader_cwd)

                # Append results to the log file
                f = open("{}output_log.txt".format(teensy_loader_cwd), "a")
                f.write(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + "\n")
                f.write(output + error + "\n\n")
                f.close()

                if error == "" and retcode == 0:
                    Log.d(TAG, "Device firmware programmed successfully!")
                    result = FW_UPDATE.RESULT_UPTODATE  # SUCCESS!
                    sleep(3)  # give device time to reboot before proceeding
                else:
                    if error == "":
                        error = output
                    Log.d(TAG, "ERROR: {}".format(error))
                    result = FW_UPDATE.RESULT_FAILED  # ERROR!

                if port == None:
                    if error == "":
                        PopUp.information(
                            parent,
                            "FW Recovery Tool",
                            "No errors to report. Recovery complete.\n\nRefresh port list to see new device.",
                        )
                    else:
                        PopUp.warning(
                            parent,
                            "FW Recovery Tool",
                            "The following error was reported:\n\n{}".format(
                                error),
                        )
            except:
                result = FW_UPDATE.RESULT_FAILED
                Log.d(
                    TAG,
                    "ERROR: Failure programming and/or rebooting device to update device firmware.",
                )
            finally:
                self.close()

        parent.ControlsWin.ui1.infobar.setText(
            "<font color=#0000ff> Infobar </font><font color={}>{}</font>".format(
                "#333333", ""
            )
        )

        return result, output, error

    ###########################################################################
    # Private method to unpack files on filesystem that are ZIP'd and not EXE's
    ###########################################################################

    def __unpackFiles__(
        self,
        basepath,
        exts=".windows" if Architecture.get_os() == OSType.windows else ".macosx",
    ):
        skipped = True  # default
        for fname in os.listdir(basepath):
            file = os.path.join(basepath, fname)
            # Log.d(TAG, "Scanning file {}".format(os.path.basename(file)))
            for ext in exts.split("|"):
                if os.path.basename(file).startswith("._"):
                    continue
                if file[-len(ext):] == ext:
                    try:
                        file = shutil.copy(
                            file,
                            file.replace(
                                ext,
                                (
                                    ".exe"
                                    if Architecture.get_os() == OSType.windows
                                    else ""
                                ),
                            ),
                        )
                        Log.d(
                            TAG, 'Unpacking file "{}"...'.format(
                                os.path.basename(file))
                        )
                        os.chmod(
                            file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO
                        )  # set permissions
                        with open(file, "r+b") as f:
                            # memory-map the file, size 0 means whole file
                            # just the first 4 bytes
                            mm = mmap.mmap(f.fileno(), 4)
                            # note that new content must have same size
                            if Architecture.get_os() == OSType.windows:
                                # file signature for '.exe' files
                                mm[0:4] = b"\x4d\x5a\x90\x00"
                            else:
                                # file signature for mach-o binary, 64-bit
                                mm[0:4] = b"\xcf\xfa\xed\xfe"
                            # close the map
                            mm.close()
                        Log.d("DONE!")
                    except:
                        Log.d(TAG, "Unpacking failed.")
                    finally:
                        skipped = False
        if skipped:
            Log.d(TAG, "Unpacking skipped.")

    ###########################################################################
    # Private method to remove files on filesystem that are "virus" executables
    ###########################################################################

    def __removeFiles__(self, basepath, exts="_cli|.exe"):
        for fname in os.listdir(basepath):
            file = os.path.join(basepath, fname)
            # Log.d("scanning file {}".format(os.path.basename(file)))
            for ext in exts.split("|"):
                if os.path.basename(file).startswith("._"):
                    continue
                if file[-len(ext):] == ext:
                    Log.d(TAG, 'Cleaning file: "{}"...'.format(
                        os.path.basename(file)))
                    os.remove(file)
                    Log.d("DONE!")

    ###########################################################################
    # Private method to rename files on filesystem to be executable or ZIP safe
    ###########################################################################

    def __renameFiles__(self, basepath, exts, exec):
        for fname in os.listdir(basepath):
            file = os.path.join(basepath, fname)
            # Log.d("scanning file {}".format(os.path.basename(file)))
            for ext in exts.split("|"):
                if exec:
                    # remove ".safe" extension
                    if file[-(len(ext) + 5):] == "{}.safe".format(ext):
                        Log.d(
                            TAG, 'Renaming file "{}" to "{}"'.format(
                                file, file[0:-5])
                        )
                        os.rename(file, file[0:-5])
                else:
                    # add ".safe" extension
                    if file[-len(ext):] == ext:
                        Log.d(
                            TAG,
                            'Renaming file "{}" to "{}"'.format(
                                file, "{}.safe".format(file)
                            ),
                        )
                        os.rename(file, "{}.safe".format(file))
