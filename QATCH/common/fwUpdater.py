"""
fwUpdater.py

Firmware update utilities for QATCH devices.

Provides :class:`FWUpdater` for checking and applying firmware updates to
QATCH devices over serial (COM/HID) or network (HTTP) transports, and
:class:`UpdaterTask` which runs the actual programming sequence on a
background :class:`~PyQt5.QtCore.QThread`.  Supporting enumerations
:class:`FWUpdate` and :class:`HWType` encode the possible update-check
results and hardware board variants respectively.

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol  (paul.macnichol@qatchtech.com)
    Other QATCH Technologies contributors

Date:
    2026-03-20
"""

from datetime import datetime
from enum import IntEnum, unique
import mmap
import os
import shlex
import shutil
import stat
from subprocess import PIPE, Popen
import sys
from time import sleep, time
from traceback import format_tb

from PyQt5 import QtCore, QtGui, QtWidgets
import requests
from serial import serialutil

from QATCH.common.architecture import Architecture, OSType
from QATCH.common.fileStorage import FileStorage
from QATCH.common.logger import Logger as Log
from QATCH.common.tyUpdater import QATCH_TyUpdater
from QATCH.common.userProfiles import UserConstants, UserProfiles, UserRoles
from QATCH.core.constants import Constants
from QATCH.ui.popUp import PopUp

USE_PROGRESS_BAR_MODULE = False
if USE_PROGRESS_BAR_MODULE:
    from progressbar import Bar, Percentage, ProgressBar, Timer


if Constants.serial_simulate_device:
    from QATCH.processors.Simulator import serial  # type: ignore[no-redef, assignment]
else:
    from QATCH.processors.Device import serial  # type: ignore[no-redef, assignment]


TAG = "[FWUpdater]"


@unique
class FWUpdate(IntEnum):
    RESULT_REQUIRED = -3
    RESULT_FAILED = -2
    RESULT_UNKNOWN = -1
    RESULT_NONE = 0
    RESULT_UPTODATE = 1
    RESULT_OUTDATED = 2


@unique
class HWType(IntEnum):
    UNKNOWN = -1
    TEENSY36 = 1
    TEENSY41 = 2

    @staticmethod
    def parse(value: str) -> "HWType":
        try:
            return HWType[value.translate({ord(c): None for c in "_ -"})]
        except Exception:
            return HWType.UNKNOWN


class FWUpdater:
    """Checks and applies firmware updates for QATCH devices.

    Manages the full update lifecycle: opening the serial port, querying the
    running firmware version, prompting the user for permission, launching the
    programming task on a background thread, and verifying the result.  Supports
    COM, HID, and network (HTTP) transports via the injected ``serial`` backend.

    Class Attributes:
        _check (bool): When ``False``, skips the version check on the next
            :meth:`run` call.
        _serial (serial.Serial): Shared serial-port handle used across methods.
        _hw (HWType): Board variant detected during the last :meth:`check_update`.
        _active_uid (str | None): USB serial number of the active device.
        _port_changed (bool): ``True`` when the port identifier changed during
            the last programming task (e.g., after a reboot).
        _port (str | None): Port identifier set after the last update attempt.
        transient_err_cnt (int): Number of ``[TRANSIENT]`` errors seen so far;
            used to suppress repeated error dialogs.
    """

    _check = True
    _serial = serial.Serial()
    _hw = HWType.UNKNOWN

    _active_uid = None
    _port_changed = False
    _port: str | None = None

    # Flag indicating the presence of a [TRANSIENT] error in the error message.
    # Errors are suppressed in error popup but collected for Device Info.
    transient_err_cnt = 0

    def _get_update_permission(
        self,
        parent,
        ask_permission: bool,
        result: int,
        version: str,
        target: str,
        user_resp: bool | None,
    ) -> tuple[bool | None, bool | None]:
        """Checks user roles and prompts for firmware update permission.

        Args:
            parent: Parent UI instance exposing ``ControlsWin`` and ``ReadyToShow``.
            ask_permission (bool): Whether to show the update confirmation dialog.
            result (int): ``FWUpdate.RESULT_*`` constant from ``check_update``.
            version (str): Currently running firmware version string.
            target (str): Recommended firmware version string.
            user_resp (bool | None): Cached user response from a prior prompt, or
                ``None`` if no prompt has been shown yet.

        Returns:
            tuple[bool | None, bool | None]
        """
        check_result = True
        if UserConstants.REQ_ADMIN_UPDATES:
            action_role = UserRoles.ADMIN
            check_result = UserProfiles().check(parent.ControlsWin.userrole, action_role)

        if ask_permission and check_result:
            question = "Device is running "
            if result == FWUpdate.RESULT_REQUIRED:
                question += "INCOMPATIBLE"
            elif result == FWUpdate.RESULT_UNKNOWN:
                question += "UNKNOWN"
            elif result == FWUpdate.RESULT_OUTDATED:
                question += "OUTDATED"
            question += " firmware.\nWould you like to update it now?"

            if user_resp is None:
                user_resp = PopUp.question_FW(
                    parent,
                    "Firmware Update Available",
                    question,
                    f"Running FW: {version} ({self._hw.name})\nRecommended: {target}\n\n"
                    "Updating only takes a few seconds and guarantees operational compatibility.",
                )
            return user_resp, user_resp

        elif not check_result:
            Log.w(
                TAG,
                "A firmware update is available! "
                "Please ask your administrator to install update.",
            )
            return None, user_resp

        return parent.ReadyToShow, user_resp

    def _execute_update_task(
        self, parent, port: str, target: str, i: int, num_ports: int
    ) -> tuple[int, str, str]:
        """Sets up the UI and runs the QThread for the UpdaterTask.

        Args:
            parent: Parent UI instance exposing ``ControlsWin`` and ``worker``.
            port (str): Serial port identifier for the device being programmed.
            target (str): Target firmware version string shown in the progress
                dialog title and label.
            i (int): One-based index of the current port within the multiplex
                sequence, used to build the ``"(N of M)"`` subtitle.
            num_ports (int): Total number of ports being updated in this run,
                used to conditionally append the multiplex subtitle.

        Returns:
            tuple[int, str, str]
        """
        parent.ControlsWin.ui1.infobar.setText(
            "<font color=#0000ff> Infobar </font><font color=#333333>"
            "Programming device firmware... please wait...</font>"
        )
        parent.ControlsWin.ui1.infobar.repaint()

        multistep = f" ({i} of {num_ports})" if num_ports > 1 else ""
        self.progressBar = QtWidgets.QProgressDialog(
            f"Programming FW {target}...", "Cancel", 0, 100, parent
        )
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/download_icon.ico")
        self.progressBar.setWindowIcon(QtGui.QIcon(icon_path))
        self.progressBar.setWindowTitle(f"Programming FW {target}{multistep}")
        self.progressBar.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)  # type: ignore[attr-defined]
        self.progressBar.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)  # type: ignore[attr-defined]
        self.progressBar.canceled.disconnect()
        self.progressBar.canceled.connect(self.update_cancel)
        self.progressBar.setFixedSize(
            int(self.progressBar.width() * 1.5), int(self.progressBar.height() * 1.1)
        )

        self.upd_thread = QtCore.QThread()
        ports = (
            parent.worker._port if isinstance(parent.worker._port, list) else [parent.worker._port]
        )
        primary_dev = ports[0]

        self.updater = UpdaterTask(port, self._serial, primary_dev, self._active_uid)
        self.updater.moveToThread(self.upd_thread)
        self.upd_finished = False

        self.upd_thread.started.connect(self.updater.run)
        self.updater.finished.connect(self.upd_thread.quit)
        self.updater.finished.connect(self.updater.deleteLater)
        self.upd_thread.finished.connect(self.upd_thread.deleteLater)
        self.updater.progress.connect(self.update_progress)
        self.updater.finished.connect(self.update_finished)

        self.upd_thread.start()
        self.progressBar.exec_()  # wait here until finished installing

        while not self.upd_finished:
            Log.e(TAG, "Cannot abort FW update once programming starts... please wait!")
            self.progressBar.exec_()

        if self.updater._cancel and getattr(self._serial, "hid_port", None) is None:
            Log.w(TAG, "Download canceled.")
            box = QtWidgets.QMessageBox(None)
            box.setIcon(QtWidgets.QMessageBox.Information)
            box.setWindowTitle("Program FW Update")
            box.setText("Download canceled.")
            box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            box.exec_()

        result, output, error = self.updater.get_results()

        if port != self.updater._port:
            self._port_changed = True
            self._port = self.updater._port
        else:
            self._port_changed = False
            self._port = port

        parent.ControlsWin.ui1.infobar.setText(
            "<font color=#0000ff> Infobar </font><font color=#333333></font>"
        )
        return result, output, error

    def _verify_update_results(
        self, parent, upd_result: int, output: str, error: str, version: str, i: int, num_ports: int
    ) -> bool:
        """Verifies the device after update and handles Success/Failure PopUps.

        Args:
            parent: Parent UI instance passed through to ``check_update`` and
                ``PopUp`` dialogs.
            upd_result (int): ``FWUpdate.RESULT_*`` constant returned by the
                programming task.
            output (str): Standard output string captured from the programmer,
                shown in the success dialog details.
            error (str): Standard error string captured from the programmer,
                shown in the failure dialog details after string substitution.
            version (str): Pre-update firmware version string, shown in the
                success dialog for comparison.
            i (int): One-based current port index; success dialog is only shown
                after the final port (``i == num_ports``) is processed.
            num_ports (int): Total number of ports being updated in this run.

        Returns:
            bool: ``True`` if the caller should decrement ``i`` and retry the
            current port; ``False`` to advance normally.
        """
        try_again = False

        if upd_result == FWUpdate.RESULT_UPTODATE:
            assert self._port is not None
            check, now, _, _, _ = self.check_update(parent, self._port)

            if check == FWUpdate.RESULT_UPTODATE and i == num_ports:
                PopUp.question_FW(
                    parent,
                    "Firmware Update Successful",
                    "Device is now running the latest firmware.",
                    f"Updated from {version} to {now}.\n\nDETAILS:\n{output}",
                    True,
                )
            elif check != FWUpdate.RESULT_UPTODATE:
                message = "UPDATE FAILED\n\nDevice is still running "
                if check == FWUpdate.RESULT_REQUIRED:
                    message += "INCOMPATIBLE"
                elif check in (FWUpdate.RESULT_FAILED, FWUpdate.RESULT_UNKNOWN):
                    message += "UNKNOWN"
                elif check == FWUpdate.RESULT_OUTDATED:
                    message += "OUTDATED"
                message += " firmware!"

                if PopUp.critical(
                    parent,
                    "Firmware Update Failed",
                    message,
                    f"Post-update version verification failed. (Error {check})\n\n"
                    "To fix this error, try modifying `best_fw_version` in Constants.py "
                    f"to match the currently running device firmware version: `{now}`.",
                ):
                    try_again = True

        elif upd_result != FWUpdate.RESULT_OUTDATED:
            err_msg = error.replace(
                "Please try again", "Please close the other app(s) and try again."
            )
            if PopUp.critical(
                parent,
                "Firmware Update Failed",
                "UPDATE FAILED\n\nA problem was encountered while attempting to update the "
                "device firmware.",
                f"If the problem persists, please update manually.\n\nDETAILS:\n{err_msg}",
            ):
                try_again = True

        return try_again

    def _required_or_abort_signal(self, parent, result: int, abort: bool) -> str | None:
        """Returns a loop-control signal for the RESULT_REQUIRED and abort guards.

        Encapsulates the repeated end-of-iteration check that appears both inside
        the ``if not self._check`` early-skip block and at the bottom of the main
        loop body, eliminating the duplication and reducing cyclomatic complexity
        in ``run()``.

        Args:
            parent: Parent UI instance passed through to ``PopUp.critical``.
            result (int): Current ``FWUpdate.RESULT_*`` constant.
            abort (bool): Abort flag set by ``check_update`` when the user
                acknowledged a critical hardware error dialog.

        Returns:
            str | None: One of the following sentinel strings, or ``None`` if no
            special action is required.
        """
        if result == FWUpdate.RESULT_REQUIRED:
            if PopUp.critical(
                parent,
                "Firmware Update Required",
                "Device is running INCOMPATIBLE firmware.\n\n"
                "You MUST update the device to use this software version. "
                "Press 'retry' to update.",
            ):
                return "retry"
            return "return_false"
        if abort:
            return "return_false"
        return None

    def _handle_update_flow(
        self,
        parent,
        port: str,
        result: int,
        version: str,
        target: str,
        user_response: bool | None,
        askPermission: bool,
        i: int,
        num_ports: int,
    ) -> tuple[int, bool | None, str]:
        """Runs the permission check, optional programming task, and result verification.

        Args:
            parent: Parent UI instance.
            port (str): Serial port identifier for the current device.
            result (int): Current ``FWUpdate.RESULT_*`` constant.
            version (str): Firmware version string reported by the device.
            target (str): Recommended firmware version string.
            user_response (bool | None): Cached user response from a prior prompt.
            askPermission (bool): Whether to show the update confirmation dialog.
            i (int): One-based index of the current port in the multiplex loop.
            num_ports (int): Total number of ports being iterated.

        Returns:
            tuple[int, bool | None, str]
        """
        do_update, user_response = self._get_update_permission(
            parent, askPermission, result, version, target, user_response
        )

        if do_update is None:
            return result, user_response, "return_true"

        if do_update:
            upd_result, output, error = self._execute_update_task(
                parent, port, target, i, num_ports
            )
            try_again = self._verify_update_results(
                parent, upd_result, output, error, version, i, num_ports
            )
            if try_again:
                return result, user_response, "retry"
        else:
            if result == FWUpdate.RESULT_UNKNOWN:
                result = FWUpdate.RESULT_OUTDATED

        if result >= 0 and i == num_ports:
            self._check = False

        return result, user_response, "continue"

    def _process_port_iteration(
        self,
        parent,
        port: str,
        result: int,
        user_response: bool | None,
        abort: bool,
        askPermission: bool,
        i: int,
        num_ports: int,
    ) -> tuple[bool, int, bool | None, bool, str | None]:
        """Executes one full port iteration of the firmware update loop.

        Handles the ``if not self._check`` early-skip path, the ``open`` call,
        ``check_update``, and the ``_handle_update_flow`` dispatch. Returns a
        signal string that ``run()`` uses for loop control, keeping all branching
        for a single port self-contained and out of the main loop body.

        Args:
            parent: Parent UI instance.
            port (str): Serial port identifier for the current device.
            result (int): ``FWUpdate.RESULT_*`` constant carried in from the
                previous iteration.
            user_response (bool | None): Cached user response from a prior prompt.
            abort (bool): Abort flag carried in from the previous iteration.
            askPermission (bool): Whether to prompt the user before flashing.
            i (int): One-based current port index (already incremented by ``run()``
                before this call).
            num_ports (int): Total number of ports being iterated.

        Returns:
            tuple[bool, int, bool | None, bool, str | None]
        """
        written = False

        if not self._check:
            signal = self._required_or_abort_signal(parent, result, abort)
            return written, result, user_response, abort, signal or "skip"

        if not self.open(port):
            return written, result, user_response, abort, "return_failed"

        result, version, target, written, abort = self.check_update(parent, port)

        if result not in (FWUpdate.RESULT_FAILED, FWUpdate.RESULT_UPTODATE):
            result, user_response, action = self._handle_update_flow(
                parent, port, result, version, target, user_response, askPermission, i, num_ports
            )
            if action in ("return_true", "retry"):
                self.close()
                return written, result, user_response, abort, action
        elif i == num_ports:
            self._check = False

        self.close()
        return written, result, user_response, abort, None

    def run(self, parent, askPermission: bool = True) -> None:
        """Checks firmware versions across all active ports and performs updates.

        Iterates over all ports associated with ``parent.worker``, opens each
        one, checks firmware and hardware info, prompts the user if an update is
        available (subject to role checks), runs the programming task, and
        verifies the result. Supports multiplex (multi-device) configurations via
        the retry loop.

        Args:
            parent: Parent UI instance exposing ``worker``, ``ControlsWin``,
                ``ReadyToShow``, and ``_refresh_ports``.
            askPermission (bool): When ``True``, shows a confirmation dialog
                before flashing. When ``False``, proceeds without prompting.

        Returns:
            bool | int: ``True`` if all ports completed without a hard abort,
            ``False`` if the user declined a required update or a hardware abort
            was signalled, or ``FWUpdate.RESULT_FAILED`` if the port could not
            be opened.
        """
        result = FWUpdate.RESULT_NONE
        user_response = None
        abort = False

        ports = (
            parent.worker._port if isinstance(parent.worker._port, list) else [parent.worker._port]
        )
        num_ports = len(ports)

        # Maps signal strings returned by _process_port_iteration to their
        # corresponding run() return values, avoiding multiple if/elif branches.
        _return_map = {
            "return_true": True,
            "return_false": False,
            "return_failed": FWUpdate.RESULT_FAILED,
        }

        i = 0
        while i < num_ports:
            port = ports[i]
            self._serial.port = port
            i += 1

            written, result, user_response, abort, signal = self._process_port_iteration(
                parent, port, result, user_response, abort, askPermission, i, num_ports
            )

            if signal in _return_map:
                return _return_map[signal]
            if signal in ("retry", "skip"):
                if signal == "retry":
                    i -= 1
                continue

            if written:
                parent._refresh_ports()

            end_signal = self._required_or_abort_signal(parent, result, abort)
            if end_signal:
                if end_signal == "retry":
                    i -= 1
                    continue
                return False

        return not abort

    def open(self, port: str) -> bool:
        """Configures and opens the serial port for firmware operations.

        Args:
            port (str): Serial port name (e.g. ``"COM3"`` or ``"/dev/ttyUSB0"``).

        Returns:
            bool: ``True`` if the port opened successfully, ``False`` otherwise.
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
        except Exception:
            Log.e(
                TAG,
                "ERROR: Failure opening port to check and/or update device firmware.",
            )
            return False

    def close(self) -> None:
        """Closes the serial port if it is currently open."""
        if self._serial.is_open:
            self._serial.close()

    def check_again(self) -> None:
        """Resets the check flag so the next :meth:`run` call re-queries the device."""
        self._check = True

    def _send_stop_command(self) -> None:
        """Sends the stop command to the device and waits for a STOP reply.

        Writes the ``stop\\n`` command over serial and listens for the 5-byte
        ASCII sequence ``STOP\\r\\n`` (hex: 53 54 4f 50 0d 0a) within a 3-second
        timeout window.

        NOTE Behavioral change:
            The original code used an inner ``while`` loop that spun indefinitely
            waiting for the ``S`` byte with no timeout guard::

                while self._serial.read(1).hex() != "53":
                    pass

            If the device never sent ``S``, this loop would block forever. The
            refactored version replaces this with a single ``if`` check inside the
            outer timeout-bounded ``while`` loop. A non-matching first byte is
            simply discarded and the outer loop retries until the 3-second window
            expires.
        """
        self._serial.write("stop\n".encode())
        stopped = False
        stop = time()
        wait_for = 3  # timeout delay (seconds)
        while time() - stop < wait_for:
            while time() - stop < wait_for and self._serial.in_waiting == 0:
                pass
            # NOTE All nested ifs collapsed into one compound condition.
            # The timeout guard, the leading "S" byte, and the remaining
            # bytes are all chained with `and`. Python short-circuit evaluation
            # should ensure each byte is only read if the previous condition
            # was met, preserving the original byte-by-byte validation semantics.
            if (
                time() - stop < wait_for
                and self._serial.read(1).hex() == "53"  # "S"
                and self._serial.read(1).hex() == "54"  # "T"
                and self._serial.read(1).hex() == "4f"  # "O"
                and self._serial.read(1).hex() == "50"  # "P"
                and self._serial.read(1).hex() == "0d"  # "\r"
                and self._serial.read(1).hex() == "0a"  # "\n"
            ):
                stopped = True
                break
        if not stopped:
            Log.w(
                TAG,
                "WARNING: Device did not respond to STOP command. "
                "May still be streaming or running old FW...",
            )

    def _read_firmware_version(self) -> tuple[str, FWUpdate]:
        """Reads and evaluates the firmware version string from the device.

        Writes the ``version\\n`` command over serial and parses the multi-line
        response into build, version, and date fields. Compares the reported
        version against ``Constants.best_fw_version`` to determine the upgrade
        status.

        Returns:
            tuple[str, int]: A two-element tuple containing the
            version and update result status.
        """
        self._serial.write("version\n".encode())
        timeout_at = time() + 3
        while self._serial.in_waiting == 0 and time() < timeout_at:
            pass

        if time() >= timeout_at:
            Log.w(
                TAG,
                "Cannot read device firmware version. No response from device.",
            )
            return "UNKNOWN", FWUpdate.RESULT_UNKNOWN

        reply = self._serial.read(self._serial.in_waiting).decode().split("\n")
        if len(reply) < 3:
            Log.w(
                TAG,
                "Cannot read device firmware version. Incorrect response.",
            )
            return "UNKNOWN", FWUpdate.RESULT_UNKNOWN

        build, version, date = reply[0].strip(), reply[1].strip(), reply[2].strip()
        Log.d(
            TAG,
            f"SERIAL BUILD INFORMATION\n FW Device: {build}\n "
            f"FW Version: {version}\n FW Date: {date}\n",
        )

        branch = Constants.best_fw_version[0:4]
        if branch not in version:
            Log.w(
                TAG,
                f"Device is running outdated firmware ({version})!\n"
                " You MUST upgrade to continue..."
                f"(Recommended: {Constants.best_fw_version})\n",
            )
            return version, FWUpdate.RESULT_REQUIRED
        elif version != Constants.best_fw_version:
            Log.w(
                TAG,
                f"Device is running outdated firmware ({version})!\n"
                "Please consider upgrading..."
                f" (Recommended: {Constants.best_fw_version})\n",
            )
            return version, FWUpdate.RESULT_OUTDATED

        Log.i(TAG, f"Device is running up-to-date firmware ({version})!\n")
        return version, FWUpdate.RESULT_UPTODATE

    def _read_hardware_info(self) -> dict[str, str] | None:
        """Reads the hardware info block from the device and parses it into a dict.

        Writes the ``info\\n`` command over serial and reads lines until no more
        data arrives within a 3-second timeout. Each line is expected to be in
        ``KEY: VALUE`` format.

        Returns:
            dict[str, str] | None: A dictionary mapping field names (e.g.
            ``"HW"``, ``"USB"``, ``"PID"``, ``"ERR"``) to their string values, or
            ``None`` if no data was received before the timeout.
        """

        self._serial.write("info\n".encode())
        timeout_at = time() + 3
        info_reply = b""

        while self._serial.in_waiting == 0 and time() < timeout_at:
            pass

        while time() < timeout_at:
            if self._serial.in_waiting != 0:
                info_reply += self._serial.read_until()
            else:
                break

        if not info_reply:
            return None

        info_dict = {}
        for line in info_reply.decode().split("\n"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                key, val = parts[0].strip(), parts[1].strip()
                info_dict[key] = val
                if key == "ERR":
                    if "[TRANSIENT]" in val:
                        self.transient_err_cnt += 1
                    else:
                        self.transient_err_cnt = 0
        return info_dict

    def _resolve_info_conflicts(self, idx: int, name: str | None, usb: str, path: str) -> None:
        """Resolves file storage conflicts for device info entries.

        Iterates the known device list and marks any existing entry that shares
        the current COM port (``self._com``) but has a different USB identifier
        as conflicted by appending ``!`` to its stored PORT value. This prevents
        a stale entry from a previously connected device from silently overriding
        the current device's info.

        Args:
            idx (int): Device slot index derived from the PID. ``0`` is used for
                solo devices; ``1-8`` for indexed multiplex devices.
            name (str): Resolved device name returned by ``_get_device_name``.
            usb (str): USB identifier string from the current device's info
                block, used to detect identity mismatches on the same port.
            path (str): File system path used when writing the conflict-marked
                device info via ``FileStorage.dev_info_set``.

        Raises:
            Exception: Catches all exceptions internally and logs an error
                message instructing the user to correct the ``config`` file
                manually. No exception propagates to the caller.
        """
        try:
            if FileStorage.dev_get_active(idx) == "" and idx > 0:
                FileStorage.dev_set_active(idx, name)
            dev_folder = f"{idx}_{name}" if idx > 0 else name

            if FileStorage.dev_get_active(idx) != dev_folder:
                for i, dev_name in FileStorage.dev_get_device_list():
                    dev_info = FileStorage.dev_info_get(i, dev_name)
                    if (
                        "USB" in dev_info
                        and "PORT" in dev_info
                        and dev_info["PORT"] == self._com
                        and dev_info["USB"] != usb
                    ):
                        FileStorage.dev_info_set(
                            i,
                            Constants.txt_device_info_filename,
                            path,
                            dev_info.get("NAME"),
                            dev_info.get("FW"),
                            dev_info.get("HW"),
                            port=f"{dev_info.get('PORT')}!",
                            ip=dev_info.get("IP"),
                            uid=dev_info.get("UID"),
                            mac=dev_info.get("MAC"),
                            usb=dev_info.get("USB"),
                            pid=dev_info.get("PID"),
                            rev=dev_info.get("REV"),
                            err=dev_info.get("ERR"),
                        )
        except Exception:
            Log.e(
                TAG,
                "Conflicting device info could not be automatically resolved. "
                "Please correct `config` manually.",
            )

    def _parse_port(self, port: str | None) -> None:
        """Parses a port string and sets ``_com`` and ``_net`` accordingly.

        Determines whether the supplied port string represents a COM/serial port,
        a network (IP) address, or a combined ``COM;NET`` pair, and assigns the
        instance attributes ``_port``, ``_com``, and ``_net`` based on the result.

        Args:
            port (str | None): Raw port identifier from the caller.
        """

        self._port = port
        if port is None:
            self._com = self._net = None
        elif ":" in port:
            self._com, self._net = port, None
        elif ";" in port:
            self._com, self._net = port.split(";")[:2]
        elif port.count(".") == 3:
            self._com, self._net = None, port
        else:
            self._com, self._net = port, None

    def check_update(
        self, parent: QtWidgets.QWidget, port: str
    ) -> tuple[int, str, str, bool, bool]:
        """Checks the device firmware version and writes hardware info to storage.

        Opens the serial port if not already open, issues stop/version/info
        commands in sequence, resolves any device info conflicts, and writes the
        device info file. Optionally presents a critical error dialog if the
        device reports a non-transient hardware error.

        NOTE: Behavioral changes:
            The original ``check_update`` contained all serial communication
            inline within a single large ``if self._serial.is_open`` block,
            returning default values via a fall-through rather than an early
            return. This version returns early if the port fails to open.
            Hardware attribute access is also guarded with ``getattr`` to prevent
            ``AttributeError`` if ``_hw`` was never assigned due to an earlier
            exception.

        Args:
            parent (QWidget): Parent widget used to anchor any modal dialogs
                (e.g. the hardware error ``PopUp``). Must expose a
                ``ReadyToShow`` boolean attribute.
            port (str): Serial port identifier passed to ``open()`` and
                ``_parse_port()``.

        Returns:
            tuple: A five-element tuple:

                - result (int): FWUpdate.RESULT_* constant describing
                the firmware update status.
                - version (str): Version string read from the device, or a
                descriptive error string on failure.
                - best_version (str): Constants.best_fw_version - the
                recommended firmware version for comparison.
                - written (bool): True if the device info file was
                successfully written to storage.
                - abort_action (bool): True if the user acknowledged a
                critical hardware error dialog, signalling the caller to abort
                the current action.
        """
        if not self._serial.is_open:
            self.open(port)

        version = "COULD NOT CHECK"
        result = FWUpdate.RESULT_UNKNOWN
        written = False
        abort_action = False

        if not self._serial.is_open:
            return result, version, Constants.best_fw_version, written, abort_action

        try:
            self._send_stop_command()
            version, result = self._read_firmware_version()
            info_dict = self._read_hardware_info()

            if info_dict:
                hw = info_dict.get("HW")
                err = info_dict.get("ERR")
                usb = info_dict.get("USB")
                pid = info_dict.get("PID")

                Log.d(TAG, f"Detected HW TYPE is {hw}.")
                self._hw = HWType.parse(hw or "UNKNOWN")
                self._active_uid = usb

                path = Constants.csv_calibration_export_path
                self._parse_port(port)
                name = self._get_device_name(usb, self._com)
                idx = 0 if pid in [None, "FF"] else int(pid, base=16) % 9  # type: ignore[arg-type]

                self._resolve_info_conflicts(idx, name, usb, path)
                written = FileStorage.dev_info_set(
                    idx,
                    Constants.txt_device_info_filename,
                    path,
                    name,
                    version,
                    self._hw.name,
                    port=self._com,
                    ip=info_dict.get("IP"),
                    uid=info_dict.get("UID"),
                    mac=info_dict.get("MAC"),
                    usb=usb,
                    pid=pid,
                    rev=info_dict.get("REV"),
                    err=err,
                )

                if err not in {None, "NONE"} and self.transient_err_cnt <= 1 and parent.ReadyToShow:
                    if PopUp.critical(
                        parent,
                        "Hardware Error Detected",
                        "<b>SERVICE REQUIRED</b>: HARDWARE ERROR DETECTED!<br/>"
                        "It is not recommended to use this device until serviced.",
                        details=f"Error Detected:\n{err}",
                        btn1_text="Ok",
                    ):
                        abort_action = True
                    QtCore.QTimer.singleShot(500, self.check_again)
            else:
                Log.w(
                    TAG,
                    "Cannot read device hardware info. " "Assuming legacy FW running on TEENSY36.",
                )
                self._hw = HWType.TEENSY36
                name = self._get_device_name(None, port)
                FileStorage.dev_set_active(0, name)

            if getattr(self, "_hw", HWType.UNKNOWN) == HWType.UNKNOWN:
                Log.w(
                    TAG,
                    "Received unexpected hardware info reply. "
                    "Please enter HW type (if prompted)...",
                )
        except Exception:
            version = "FAILED TO READ FROM DEVICE"
            result = FWUpdate.RESULT_FAILED
        finally:
            self.close()

        return result, version, Constants.best_fw_version, written, abort_action

    def _get_device_name(self, preferred: str | None, backup: str | None) -> str | None:
        """Returns the best available device name from two candidate sources.

        Args:
            preferred: USB serial number obtained from the device ``INFO``
                command; ``None`` if the command is not supported.
            backup: OS-level port name (e.g. ``"COM3"`` or ``"/dev/ttyUSB0"``);
                used when ``preferred`` is unavailable.

        Returns:
            str | None: The preferred USB serial number when available, the
            port name on Windows, or a stripped numeric string on other
            platforms.  ``None`` if both inputs are ``None``.
        """
        if preferred is not None:
            if not preferred.isnumeric():
                Log.w(TAG, "Preferred device name is not numeric!")
            return preferred  # INFO CMD supported, normal operation
        if backup is not None:
            if backup.startswith("COM"):
                return backup  # INFO CMD failed, running WINDOWS OS
            else:
                # INFO CMD failed, running non-WINDOWS OS
                numeric_filter = filter(str.isdigit, backup)
                backup_digits = "".join(numeric_filter)
                return backup_digits[0:-1]  # all digits in backup (but last)

    def _progress_should_skip(self, label_str: str | None) -> bool:
        """Returns True if the current progress signal should be silently ignored.

        Consolidates all early-exit guard conditions from ``update_progress``
        into a single predicate, eliminating the inline chain of ``return``
        statements and reducing cyclomatic complexity in the caller.

        Args:
            label_str (str | None): The label string received by
                ``update_progress``, or ``None`` if no label was supplied.

        Returns:
            bool: ``True`` if ``update_progress`` should return immediately
            without further processing; ``False`` to continue normally.
        """
        if self.progressBar.labelText().find("Cancel") >= 0:
            return True
        # NOTE: Collapsed nested isinstance + find check into one condition.
        if isinstance(label_str, str) and label_str.find("Wait") >= 0:
            return True
        if self._serial.hid_port is not None:
            return True
        return False

    def _progress_advance_programming(self, curr_val: int) -> None:
        """Drives the progress bar through the programming and verification phases.

        Called when the progress bar has entered programming mode (transfer
        complete or not yet started). Manages label text, cancel-button state,
        value advancement, and the recurring ``QTimer`` callback that animates
        the bar while the programmer is running.

        Args:
            curr_val (int): The current progress bar value at the time this
                method is called.
        """
        delay = 500
        if self.progressBar.labelText().find("Transfer") >= 0:
            cancelButton = self.progressBar.findChild(QtWidgets.QPushButton)
            cancelButton.setEnabled(False)
            self.progressBar.setLabelText(
                "Programming device firmware...<br/><b>DO NOT POWER CYCLE DEVICE!</b>"
            )
            curr_val = 0
        if self.upd_finished:
            self.progressBar.setLabelText("Checking device firmware version...")
            delay = 50
        elif curr_val == 99:
            curr_val -= 1  # prevent ending until updater is finished
        self.progressBar.setValue(curr_val + 1)
        if curr_val + 1 != self.progressBar.maximum():
            QtCore.QTimer.singleShot(delay, self.update_progress)

    def update_progress(
        self, label_str: str | None = None, progress_pct: int | None = None
    ) -> None:
        """Slot that receives progress signals from ``UpdaterTask`` and drives the UI.

        Args:
            label_str (str | None): Label text to display on the progress
                dialog, or ``None`` to leave the current label unchanged.
            progress_pct (int | None): Progress percentage value (0-100) to
                set on the bar, or ``None`` to leave the current value
                unchanged.
        """
        try:
            need_repaint = False
            if progress_pct == self.progressBar.maximum():
                return  # ignore this signal, move on to "programming" mode
            if label_str is not None and not self.updater._cancel:
                self.progressBar.setLabelText(label_str)
            if progress_pct is not None:
                self.progressBar.setValue(progress_pct)

            if self._progress_should_skip(label_str):
                return

            curr_val = self.progressBar.value()
            in_programming_mode = (
                curr_val == 99 and self.progressBar.labelText().find("Transfer") >= 0
            ) or self.progressBar.labelText().find("Transfer") == -1

            if in_programming_mode:
                need_repaint = True
                self._progress_advance_programming(curr_val)

            if need_repaint:
                self.progressBar.repaint()
        except Exception:
            limit = None
            t, v, tb = sys.exc_info()
            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.d(TAG, line)

    def update_finished(self) -> None:
        """Slot called when the updater thread emits ``finished``.

        Marks the update as complete and closes the progress dialog when the
        operation was cancelled or used the HID transport.
        """
        self.upd_finished = True
        if self.updater._cancel or self._serial.hid_port is not None:
            # Log.w("Closing bar after cancel.")
            self.progressBar.close()

    def update_cancel(self) -> None:
        """Slot called when the user clicks the progress-dialog Cancel button.

        Disables the cancel button to prevent re-entry, advances the progress
        label to ``"Canceling installation..."`` on COM ports, and signals the
        :class:`UpdaterTask` to abort.
        """
        cancelButton = self.progressBar.findChild(QtWidgets.QPushButton)
        cancelButton.setEnabled(False)
        if self._serial.hid_port is None:
            # Log.d("GUI: Toggle progress mode")
            self.update_progress("Canceling installation...", 99)
            self.updater.cancel()
        # self.updater.wait()
        # self.progressBar.close()


class UpdaterTask(QtCore.QThread):
    """Background thread that programs firmware onto a QATCH device.

    Emits ``progress(label, pct)`` signals during the update so the owning
    :class:`FWUpdater` can drive the progress dialog, and emits ``finished``
    when the programming sequence completes (successfully or otherwise).

    Signals:
        finished: Emitted when :meth:`run` returns.
        progress (str, int): Label text and integer percentage (0-100) for
            the progress dialog.

    Attributes:
        _cancel (bool): Set to ``True`` by :meth:`cancel` to request an abort.
    """

    finished = QtCore.pyqtSignal()
    # exception = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str, int)
    _cancel = False

    def __init__(self, port: str, serial, primary: str, active_uid: str | None) -> None:
        """Initialises the updater task with connection and device parameters.

        Args:
            port (str): Serial port identifier for the target device.
            serial: Open serial-port handle shared with the owning
                :class:`FWUpdater`.
            primary (str): Port identifier of the primary (first) device when
                multiple devices are being updated in sequence.
            active_uid (str | None): USB serial number of the active device,
                used for port-change detection after reboot.
        """
        super().__init__()
        self._port = port
        self._serial = serial
        self._primary = primary
        self._active_uid = active_uid
        self._hw = HWType.TEENSY41  # Teensy 3.6 not supported
        self._result = FWUpdate.RESULT_UNKNOWN
        self._output = ""
        self._error = "ERROR: No result to give."

    def cancel(self) -> None:
        """Requests cancellation of the in-progress update.

        Sets the ``_cancel`` flag; the running :meth:`do_update` implementation
        polls this flag at safe checkpoints and aborts when it is ``True``.
        """
        # Log.d("GUI: Toggle progress mode")
        # Log.w("Process kill request")
        self._cancel = True
        # self.progressTaskHandle.terminate()
        # self.progressTaskHandle.kill()
        # self._dbx_connection.close() # force abort of active file download

    def run(self) -> None:
        """Thread entry point; delegates to :meth:`do_update` and emits ``finished``."""
        self._result, self._output, self._error = self.do_update(None, self._port)
        self.finished.emit()

    def get_results(self) -> tuple[int, str, str]:
        """Returns the programming result captured by :meth:`run`.

        Returns:
            tuple[FWUpdate, str, str]: A three-element tuple containing the
            result code, standard output, and error string from the programmer.
        """
        return self._result, self._output, self._error

    def open(self, port: str) -> bool:
        """Configures and opens the serial port for the programming sequence.

        Args:
            port (str): Serial port name (e.g. ``"COM3"`` or ``"/dev/ttyUSB0"``).

        Returns:
            bool: ``True`` if the port opened successfully, ``False`` otherwise.
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
        except Exception:
            Log.e(
                TAG,
                "Failure opening port to check and/or update device firmware.",
            )
            return False

    def close(self) -> None:
        """Closes the serial port if it is currently open."""
        if self._serial.is_open:
            self._serial.close()

    def _find_hex_path(self) -> tuple[str, int]:
        """Locates the firmware folder and constructs the path to the hex image.

        Searches ``Architecture.get_path()`` for a directory whose name contains
        the expected firmware folder prefix and returns the full path to the
        ``*.ino.{hw}.hex`` image file along with its line count.

        Returns:
            tuple[str, int]: A two-element tuple containing the path
            to the hex file and the expected total line count for progress tracking.
        """
        basepath = Architecture.get_path()
        folder_name = "QATCH_Q-1_FW_py_" + Constants.best_fw_version.strip()
        for fname in os.listdir(basepath):
            path = os.path.join(basepath, fname)
            if os.path.isdir(path) and folder_name in fname:
                folder_name = fname
                break
        path_to_hex = os.path.join(
            basepath,
            folder_name,
            f"{folder_name}.ino.{self._hw.name}.hex",
        )
        # Replaced bare open() with a context manager so the file
        # handle is guaranteed to close even if the generator raises.
        with open(path_to_hex) as fh:
            expected = sum(1 for _ in fh)
        return path_to_hex, expected

    def _do_network_update(self, f, expected: int, port: str) -> tuple[str, str, float, float]:
        """Uploads the firmware image to the device over HTTP (NET port).

        Sends a PUT request containing the full hex image and the expected line
        count, then polls the device's ``/version`` endpoint until it responds
        successfully (up to 90 seconds), indicating that flashing and reboot are
        complete.

        Args:
            f (BinaryIO): Open file handle to the hex image.
            expected (int): Total line count of the hex image.
            port (str): Port identifier used only for log context.

        Returns:
            tuple[str, str, float, float]: A four-element tuple containing
            the output, error, and timing information for the network update process.
        """
        Log.d(TAG, f"QUERY-PUT: http://{self._serial.net_port}:8080/program")
        start = time()
        d = f.read()
        d += str(expected).encode()
        d += b"\r\n"
        r = requests.put(f"http://{self._serial.net_port}:8080/program", data=d)
        output = r.text
        error = ""
        if "FAILURE" in output:
            error = "The upgrade operation failed to complete."

        Log.i(TAG, "Waiting for device to flash and reboot... (this may take a minute)")
        stop = time()
        wait_for = 90
        successes = 0
        while time() - stop < wait_for:
            sleep(1)
            try:
                r = requests.get(f"http://{self._serial.net_port}:8080/version")
                if r.ok:
                    successes += 1
                if successes > 5:
                    break
            except Exception:
                successes = 0
                continue
        if time() - stop >= wait_for:
            Log.e(TAG, f"Failed to establish network connection within {wait_for} secs.")
        return output, error, start, stop

    def _enter_bootloader(self) -> tuple[str, float]:
        """Sends the ``program`` command and waits for the bootloader prompt.

        Writes ``program\\n`` over serial and reads incoming data until the
        ``"waiting for hex lines..."`` marker is received or the 3-second
        timeout expires.

        Returns:
            tuple[str, float]: A two-element tuple containing the output received
            from the device and the timestamp when the command was sent, used as
            the reference point for transfer timing.

        Raises:
            TimeoutError: If the bootloader prompt is not received within 3
                seconds.
        """
        self._serial.write("program\n".encode())
        output = ""
        start = time()
        wait_for = 3
        while time() - start < wait_for:
            while time() - start < wait_for and self._serial.in_waiting == 0:
                pass
            if time() - start < wait_for:
                reply = self._serial.read(self._serial.in_waiting)
                output += reply.decode().lstrip()
                if "waiting for hex lines...\n" in output:
                    break
        if time() - start >= wait_for:
            raise TimeoutError("Failed to enter bootloader.")
        return output, start

    def _transfer_firmware_com(self, f, expected: int) -> tuple[int, str]:
        """Streams hex lines to the device over the COM port.

        Iterates over every line in the open hex file, writes it to the serial
        port, emits ``progress`` signals for the UI, and honours cancellation.
        Validates the total line count after the loop and resets it to ``0``
        (signalling abort to the device) if the count does not match
        ``expected``.

        Args:
            f (BinaryIO): Open file handle to the hex image, iterated
                line-by-line.
            expected (int): Total number of lines in the hex image.

        Returns:
            tuple[int, str]: A two-element tuple containing number of lines
            successfully sent and an error message if applicable.
        """
        if USE_PROGRESS_BAR_MODULE:
            bar = ProgressBar(
                widgets=[TAG, " ", Bar(marker=">"), " ", Percentage(), " ", Timer()]
            ).start()
            bar.maxval = expected
        else:
            last_step = -1
            num_steps = 20
            scale_wide = 1
            last_pct = -1

        # Replaced manual `lines = 0` / `lines += 1` counter with
        # enumerate(f, start=1). Initialising lines=0 before the loop handles
        # the edge case where f is empty and the loop body never executes.
        lines = 0
        error = ""
        Log.d(TAG, "GUI: Toggle progress mode")

        for lines, x in enumerate(f, start=1):
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
                    self.progress.emit("Transferring firmware to device...", percent)
                    last_pct = percent
            self._serial.write(x)
            if self._cancel:
                error = "User canceled firmware update.\n"
                break

        Log.d("GUI: Toggle progress mode")
        if USE_PROGRESS_BAR_MODULE:
            bar.finish()

        if lines != expected:
            Log.d(TAG, f"incorrect # of lines sent: {lines} / {expected}")
            lines = 0
            if "abort" not in error:
                error = "Incorrect number of lines sent to device."
        return lines, error

    def _await_com_flash(self) -> tuple[str, str, float]:
        """Reads the serial port until the device confirms flash success or failure.

        Loops indefinitely reading available bytes until one of three exit
        conditions is met: the output ends with ``"SUCCESS"``, an
        ``"abort -"`` marker is received, or the 45-second timeout expires.
        A ``SerialException`` is treated as a successful reboot — the device
        dropped the port, which is the expected behavior at the end of a flash.

        Returns:
            tuple[str, str, float]: A three-element tuple containing all data
            read from the device during the flash phase, an error message if
            applicable, and the timestamp when the loop exited, used as the
            flash-complete reference for status reporting.
        """
        output = ""
        error = ""
        stop = time()
        wait_for = 45
        while True:
            try:
                reply = self._serial.read(self._serial.in_waiting)
                output += reply.decode().lstrip()
            except Exception as e:
                Log.d("Port read error. Device appears to have rebooted.", type(e))
                error = ""
                break
            if output.strip().endswith("SUCCESS"):
                error = ""
                continue
            if "abort -" in output:
                error = output.strip().split("\n")[-1]
                break
            if time() - stop > wait_for:
                error = "Device failed to reboot at end of update."
                break
        return output, error, stop

    def _detect_port_change(self, port: str) -> str:
        """Checks enumerated serial ports for a UID-matched port change after reboot.

        Iterates all currently visible serial ports and returns the updated port
        identifier if a COM→HID transition is detected via the active USB UID.
        Falls through to the original port if no change is found or if the UID
        is unavailable.

        Args:
            port (str): Port identifier in use before the reboot.

        Returns:
            str: The updated port identifier, which may be unchanged if no
            transition was detected.
        """
        for p in serial.enumerate():
            if p == port:
                Log.d(TAG, "Port did not change after updating.")
                break
            if self._active_uid is None:
                Log.w(
                    TAG,
                    "No active USB device. Cannot re-detect port, if changed. "
                    "Assuming it did not change.",
                )
                break
            if p.startswith(self._active_uid):
                Log.d(TAG, "Port changed after updating. Using new port for post-checks.")
                Log.d(TAG, f"port = {p}")
                port = p
                break
        return port

    def _verify_firmware_banner(self, port: str) -> float:
        """Opens the port and waits for the ``QATCH`` firmware banner after reboot.

        Sends the ``version\\n`` command and reads incoming data until the
        ``"QATCH"`` string appears in the response, confirming the device has
        rebooted successfully into the new firmware.

        Args:
            port (str): Port identifier to open for the post-update version check.

        Returns:
            float: ``time()`` timestamp when the banner was confirmed, used as
            the boot-complete reference for status reporting.

        Raises:
            TimeoutError: If the banner is not received within 60 seconds.
        """
        self.open(port)
        self._serial.write("version\n".encode())
        stop = time()
        wait_for = 60
        msg = ""
        while time() - stop < wait_for:
            while time() - stop < wait_for and self._serial.in_waiting == 0:
                pass
            if time() - stop < wait_for:
                reply = self._serial.read(self._serial.in_waiting)
                msg += reply.decode()
                if "QATCH" in msg:
                    sleep(1)
                    self._serial.reset_input_buffer()
                    break
        if time() - stop >= wait_for:
            raise TimeoutError("Failed to reboot within one minute.")
        return stop

    def _await_reboot_and_verify(self, port: str) -> tuple[str, float]:
        """Waits for device reboot, detects port changes, and verifies firmware.

        Args:
            port (str): Current port identifier. May be replaced if a port
                change is detected after reboot.

        Returns:
            tuple[str, float]: A two-element tuple containing the port identifier in
            use after reboot and verification, and the timestamp when the firmware
            banner was confirmed.

        Raises:
            TimeoutError: If the device does not respond with a version banner
                within 60 seconds.
        """
        try:
            stop = time()
            wait_for = 15
            while time() - stop < wait_for and self._serial.in_waiting >= 0:
                Log.w(TAG, "Waiting for device reset...")
                self._serial.write("\n".encode())
        except serialutil.SerialException:
            self.progress.emit("Waiting for device to reboot...", None)
            Log.d(TAG, "Device reset!")

        if self._port != self._primary:
            self.open(self._primary)
            self._serial.write("\n".encode())
            self.close()

        sleep(6)

        port = self._detect_port_change(port)
        self._port = port
        stop = self._verify_firmware_banner(port)
        return port, stop

    def _write_update_log(self, output, error):
        """Appends the update result to the ``FlasherX.log`` file.

        Args:
            output (str): Status and device output from the completed update.
            error (str): Error string (empty on success) appended after output.
        """
        path_to_log = os.path.join(os.getcwd(), Constants.log_export_path)
        with open(os.path.join(path_to_log, "FlasherX.log"), "a") as f:
            f.write(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + "\n")
            f.write(output + error + "\n\n")

    def _do_serial_update(
        self, f, path_to_hex, expected: int, port: str
    ) -> tuple[str, str, float, float, str, int]:
        """Orchestrates the COM or HID serial update path.

        Enters the bootloader, delegates to the appropriate transfer method
        (COM line-streaming or HID via ``QATCH_TyUpdater``), and triggers the
        post-flash reboot wait when the transfer succeeds or the user cancels.

        Args:
            f (BinaryIO): Open file handle to the hex image.
            path_to_hex (str): Absolute path to the hex file (required for HID
                path which passes it directly to ``QATCH_TyUpdater``).
            expected (int): Total line count of the hex image.
            port (str): Serial port identifier for the current device.

        Returns:
            tuple[str, str, float, float, str, int]: A six-element tuple
            containing accumulated device output, error message if applicable,
            timing information, errors, port identifier, and result hint for cancellation.
        """
        output, start = self._enter_bootloader()
        result_hint = FWUpdate.RESULT_UNKNOWN

        if self._serial.com_port is not None:
            lines, error = self._transfer_firmware_com(f, expected)
            if self._cancel:
                result_hint = FWUpdate.RESULT_OUTDATED
                Log.w(error.strip())
                output += error.lstrip()
            self._serial.write(f"{lines}\n".encode())
            Log.i(TAG, "Waiting for device to flash and reboot... (this may take a minute)")
            flash_output, error, stop = self._await_com_flash()
            output += flash_output
        else:
            dev_sernum = self._serial.hid_port.split(":")[0]
            tyUpdater = QATCH_TyUpdater(self.progress)
            error = tyUpdater.update(dev_sernum, path_to_hex)
            stop = time()

        if error == "" or self._cancel:
            port, stop = self._await_reboot_and_verify(port)
        return output, error, start, stop, port, result_hint

    def _build_update_status(
        self, path_to_hex: str, expected: int, port: str, start: float, stop: float, output: str
    ) -> str:
        """Prepends a human-readable status block to the raw device output.

        Args:
            path_to_hex (str): Absolute path to the hex file (filename used in
                the ``Target build`` line).
            expected (int): Line count of the hex image.
            port (str): Port used during flashing.
            start (float): Transfer start timestamp (from ``_enter_bootloader``
                or ``_do_network_update``).
            stop (float): Flash/reboot completion timestamp.
            output (str): Raw accumulated device output to append after the
                status block.

        Returns:
            str: Status block prepended to the stripped device output.
        """
        status = f"Target build: {os.path.split(path_to_hex)[1]}\n"
        status += f"Image size:   {expected} lines\n"
        status += f"Connection:   {port}\n"
        status += f"Time to TX:   {stop - start} secs\n"
        status += f"Time to BOOT: {time() - stop} secs\n"
        return status + output.lstrip()

    def do_update(self, parent, port: str | None) -> tuple[int, str, str]:
        """Programs the device firmware and verifies the result.

        Opens the appropriate serial port, locates the firmware hex image,
        and delegates to the network or serial update path. After flashing,
        verifies the new firmware version and presents success or failure
        dialogs. Appends a timestamped entry to ``FlasherX.log`` regardless
        of outcome.

        Args:
            parent: Parent UI instance (unused directly; passed through to
                sub-methods that present dialogs).
            port (str | None): Serial port identifier for the target device,
                or ``None`` for legacy/unknown hardware.

        Returns:
            tuple[int, str, str]: A three-element tuple containing result
            of the update outcome, status block of raw device output,
            and any error messages.
        """
        if self._port != self._primary:
            self.open(self._primary)
            self._serial.write("program\n".encode())
            self.close()

        if not self._serial.is_open and port is not None:
            self.open(port)

        output = ""
        error = "ERROR: The port is closed."
        result = FWUpdate.RESULT_UNKNOWN

        if self._hw == HWType.UNKNOWN or port is None:
            self._hw = HWType.TEENSY41  # default (don't ask)
            # force legacy CLI (but these files don't exist anymore)
            self.do_legacy_updates = True
            # if PopUp.question(
            #     parent, "Enter Teensy HW Type",
            #     "Is this hardware a TEENSY36 device?"
            # ):
            #     self._hw = HW_TYPE.TEENSY36
            # elif PopUp.question(
            #     parent, "Enter Teensy HW Type",
            #     "Is this hardware a TEENSY41 device?"
            # ):
            #     self._hw = HW_TYPE.TEENSY41
            # else:
            #     PopUp.warning(
            #         parent, "Unknown Teensy HW Type",
            #         "Unknown hardware device. Cannot proceed.\n"
            #         "Please check HW type and try again."
            #     )
            #     return result, output, error

        if self._serial.is_open or port is None:
            try:
                path_to_hex, expected = self._find_hex_path()

                with open(path_to_hex, "rb") as f:
                    Log.i(TAG, "Programming device firmware...")
                    Log.w(TAG, "DO NOT DISCONNECT YOUR DEVICE!")
                    Log.i(TAG, "Status:")

                    if self._serial.net_port is not None:
                        output, error, start, stop = self._do_network_update(f, expected, port)
                    else:
                        output, error, start, stop, port, result_hint = self._do_serial_update(
                            f, path_to_hex, expected, port
                        )
                output = self._build_update_status(path_to_hex, expected, port, start, stop, output)

                if error == "":
                    Log.i(TAG, "Device firmware programmed successfully!")
                    result = FWUpdate.RESULT_UPTODATE
                elif not self._cancel:
                    Log.e(TAG, f"ERROR: {error}")
                    result = FWUpdate.RESULT_FAILED
                else:
                    result = result_hint

            except Exception as e:
                result = FWUpdate.RESULT_FAILED
                Log.e(
                    TAG,
                    "ERROR: Failure programming and/or rebooting "
                    f"device to update device firmware. {e}",
                )
                limit = None
                t, v, tb = sys.exc_info()
                a_list = ["Traceback (most recent call last):"]
                a_list = a_list + format_tb(tb, limit)
                a_list.append(f"{t.__name__}: {str(v)}")
                for line in a_list:
                    Log.d(TAG, line)
            finally:
                self.close()

        self._write_update_log(output, error)
        return result, output, error

    def _check_teensy_prereqs(self) -> bool:
        """Checks that no conflicting Teensy processes are already running.

        Issues an OS-appropriate shell command to detect running Teensy
        processes. Both the Windows and Unix variants exit with code ``1``
        when the prerequisite conditions are satisfied (i.e. no conflicts).

        Returns:
            bool: ``True`` if prerequisites are satisfied and flashing may
            proceed; ``False`` if a conflicting Teensy process was detected.
        """
        if Architecture.get_os() is OSType.windows:
            prereq_task = Popen(
                ['tasklist /FI "IMAGENAME eq Teensy*" 2>NUL | find /I /C "Teensy">NUL'],
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
        # Both forms exit with code 1 on success; any other code = conflict.
        return prereq_task.wait() == 1

    def _find_legacy_hex_path(self) -> tuple[str, str, str]:
        """Locates the legacy firmware folder and constructs the hex image path.

        Searches ``Architecture.get_path()`` for a directory whose name
        contains the expected firmware folder prefix.

        Returns:
            tuple[str, str, str]: A three-element tuple containing the
            shell safe path to the hex image, cwd for the executable,
            and the resolved firmware folder name for log path construction.
        """
        basepath = Architecture.get_path()
        folder_name = "QATCH_Q-1_FW_py_" + Constants.best_fw_version.strip()
        for fname in os.listdir(basepath):
            path = os.path.join(basepath, fname)
            if os.path.isdir(path) and folder_name in fname:
                folder_name = fname
                break
        path_to_hex = '"{1}{0}{2}{0}{2}.ino.{3}.hex"'.format(
            Constants.slash, basepath, folder_name, self._hw.name
        )
        teensy_loader_cwd = "{1}{0}teensy_loader_cli{0}bin{0}".format(Constants.slash, basepath)
        return path_to_hex, teensy_loader_cwd, folder_name

    def _launch_teensy_loader(self, path_to_hex: str, teensy_loader_cwd: str) -> Popen:
        """Unpacks protected executables and launches the ``teensy_loader_cli``.

        Constructs the command line, handles the Windows/macOS split
        difference, and starts the subprocess. Does not wait for it to
        complete — the caller is responsible for polling and collecting
        output.

        Args:
            path_to_hex (str): Quoted shell-safe path to the firmware hex
                image, passed directly to the loader CLI.
            teensy_loader_cwd (str): Working directory for the loader
                executable.

        Returns:
            Popen: The running ``teensy_loader_cli`` subprocess.
        """
        self._unpack_files(teensy_loader_cwd)
        command_line = '".{0}teensy_loader_cli" --mcu={1} -w -v {2}'.format(
            Constants.slash, self._hw.name, path_to_hex
        )
        if not Architecture.get_os() is OSType.macosx:
            command_line = shlex.split(command_line)
        return Popen(
            command_line,
            cwd=teensy_loader_cwd,
            shell=True,
            stdout=PIPE,
            stderr=PIPE,
        )

    def _enter_legacy_bootloader(self, port: str | None, parent) -> None:
        """Puts the device into bootloader mode for the legacy update path.

        Tries the ``program\\n`` serial command first (preferred for TEENSY36),
        falling back to the 134-baud secret-sauce reboot (preferred for
        TEENSY41). If the port is not open or not provided, prompts the user
        to press the physical PROGRAM button on the device.

        Args:
            port (str | None): Serial port identifier, or ``None`` for a
                virgin/corrupt board that must be manually put into bootloader
                mode.
            parent: Parent UI instance used to present the recovery
                ``PopUp.information`` dialog when ``port`` is ``None``.
        """
        if self._serial.is_open and port is not None:
            device_reset = False
            if self._hw == HWType.TEENSY36:
                try:
                    self._serial.write("program\n".encode())
                    timeout_at = time() + 1
                    while self._serial.in_waiting == 0 and time() < timeout_at:
                        pass
                    device_reset = False
                except Exception:
                    device_reset = True
            if not device_reset:
                self._serial.close()
                self._serial.baudrate = 134  # secret sauce
                self._serial.open()
        else:
            PopUp.information(
                parent,
                "FW Recovery Tool",
                'Press the "PROGRAM" button to initiate the flash operation!',
            )

    def _write_legacy_log(self, teensy_loader_cwd: str, output: str, error: str) -> None:
        """Appends the legacy update result to the ``output_log.txt`` file.

        Args:
            teensy_loader_cwd (str): Working directory of the loader CLI,
                used to resolve the log file path.
            output (str): Standard output captured from ``teensy_loader_cli``.
            error (str): Standard error captured from ``teensy_loader_cli``.
        """
        log_path = "{}output_log.txt".format(teensy_loader_cwd)
        with open(log_path, "a") as f:
            f.write(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + "\n")
            f.write(output + error + "\n\n")

    def _show_recovery_result(self, parent, port: str | None, error: str) -> None:
        """Presents a recovery-mode result dialog when no port was provided.

        Only shown when ``port is None`` (virgin or corrupt board recovery
        flow). Displays an information dialog on success or a warning dialog
        containing the error message on failure.

        Args:
            parent: Parent UI instance used to anchor the dialog.
            port (str | None): Port identifier; dialog is suppressed when
                this is not ``None``.
            error (str): Error string; empty on success.
        """
        if port is not None:
            return
        if error == "":
            PopUp.information(
                parent,
                "FW Recovery Tool",
                "No errors to report. Recovery complete" "\n\nRefresh port list to see new device.",
            )
        else:
            PopUp.warning(
                parent,
                "FW Recovery Tool",
                "The following error was reported:\n\n{}".format(error),
            )

    def do_update_legacy(self, parent, port: str | None) -> tuple[int, str, str]:
        """Programs device firmware using the legacy ``teensy_loader_cli`` path.

        Checks that no conflicting Teensy processes are running, locates the
        firmware hex image, launches the loader CLI, puts the device into
        bootloader mode, collects results, and writes the output log.
        Presents a recovery-result dialog when running in port-less recovery
        mode.

        Args:
            parent: Parent UI instance exposing ``ControlsWin`` and used to
                anchor dialogs.
            port (str | None): Serial port identifier for the target device,
                or ``None`` for board recovery mode.

        Returns:
            tuple[int, str, str]: A three-element tuple containing the update
            statue, output from the loader, and the error message if applicable.
        """
        output = ""
        error = ""
        result = FWUpdate.RESULT_FAILED

        try:
            if not self._check_teensy_prereqs():
                error = 'ABORT: "Teensy" app(s) already running!\nPlease try again'
            else:
                path_to_hex, teensy_loader_cwd, _ = self._find_legacy_hex_path()
                update_task = self._launch_teensy_loader(path_to_hex, teensy_loader_cwd)

                Log.d(TAG, "Waiting for program...")
                sleep(3)

                if update_task.poll() is None:
                    Log.d(TAG, "Programming device firmware...")
                    self._enter_legacy_bootloader(port, parent)

                output, error = update_task.communicate()
                retcode = update_task.wait()
                output = output.decode().strip()
                error = error.decode().strip()

                self._remove_files(teensy_loader_cwd)
                self._write_legacy_log(teensy_loader_cwd, output, error)

                if error == "" and retcode == 0:
                    Log.d(TAG, "Device firmware programmed successfully!")
                    result = FWUpdate.RESULT_UPTODATE
                    sleep(3)
                else:
                    if error == "":
                        error = output
                    Log.d(TAG, "ERROR: {}".format(error))
                    result = FWUpdate.RESULT_FAILED

                self._show_recovery_result(parent, port, error)

        except Exception:
            result = FWUpdate.RESULT_FAILED
            Log.d(
                TAG,
                "Failure programming and/or rebooting device to update device firmware.",
            )
        finally:
            self.close()

        parent.ControlsWin.ui1.infobar.setText(
            "<font color=#0000ff> Infobar </font><font color={}>{}</font>".format("#333333", "")
        )
        return result, output, error

    def _unpack_files(
        self,
        basepath: str,
        exts: str = ".windows" if Architecture.get_os() == OSType.windows else ".macosx",
    ) -> None:
        """Unpacks platform-specific executables from their renamed container files.

        Iterates ``basepath`` looking for files whose extension matches one of
        the ``|``-separated patterns in ``exts``.  Each match is copied to a
        platform-appropriate destination (``*.exe`` on Windows, no extension
        otherwise), the file permissions are set to fully executable, and the
        first four bytes are patched with the correct binary magic number.

        Args:
            basepath (str): Directory to scan for packaged executable files.
            exts (str): ``|``-separated file-extension patterns that identify
                packed files.  Defaults to ``".windows"`` on Windows and
                ``".macosx"`` on macOS/Linux.
        """
        skipped = True  # default
        for fname in os.listdir(basepath):
            file = os.path.join(basepath, fname)
            # Log.d(TAG, "Scanning file {}".format(os.path.basename(file)))
            for ext in exts.split("|"):
                if os.path.basename(file).startswith("._"):
                    continue
                if file[-len(ext) :] == ext:
                    try:
                        file = shutil.copy(
                            file,
                            file.replace(
                                ext,
                                (".exe" if Architecture.get_os() == OSType.windows else ""),
                            ),
                        )
                        Log.d(TAG, 'Unpacking file "{}"...'.format(os.path.basename(file)))
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
                    except Exception:
                        Log.d(TAG, "Unpacking failed.")
                    finally:
                        skipped = False
        if skipped:
            Log.d(TAG, "Unpacking skipped.")

    def _remove_files(self, basepath: str, exts: str = "_cli|.exe") -> None:
        """Deletes files whose names end with one of the given extension patterns.

        Used to clean up native executables (e.g. ``_cli`` binaries and
        ``.exe`` files) after a legacy update so they are not left on disk.

        Args:
            basepath (str): Directory to scan for files to remove.
            exts (str): ``|``-separated suffix patterns identifying files to
                delete.  Defaults to ``"_cli|.exe"``.
        """
        for fname in os.listdir(basepath):
            file = os.path.join(basepath, fname)
            # Log.d("scanning file {}".format(os.path.basename(file)))
            for ext in exts.split("|"):
                if os.path.basename(file).startswith("._"):
                    continue
                if file[-len(ext) :] == ext:
                    Log.d(TAG, 'Cleaning file: "{}"...'.format(os.path.basename(file)))
                    os.remove(file)
                    Log.d("DONE!")

    def _rename_files(self, basepath: str, exts: str, exec: bool) -> None:
        """Renames files to make them executable or ZIP-safe depending on mode.

        When ``exec`` is ``True``, strips the ``.safe`` suffix from matching
        files to restore them to their runnable names.  When ``exec`` is
        ``False``, appends ``.safe`` so that archive tools and antivirus
        software do not flag the embedded executables.

        Args:
            basepath (str): Directory to scan for files to rename.
            exts (str): ``|``-separated file-extension patterns to match
                against (without the ``.safe`` suffix).
            exec (bool): ``True`` to restore runnable names; ``False`` to
                add the ``.safe`` suffix.
        """
        for fname in os.listdir(basepath):
            file = os.path.join(basepath, fname)
            # Log.d("scanning file {}".format(os.path.basename(file)))
            for ext in exts.split("|"):
                if exec:
                    # remove ".safe" extension
                    if file[-(len(ext) + 5) :] == "{}.safe".format(ext):
                        Log.d(TAG, 'Renaming file "{}" to "{}"'.format(file, file[0:-5]))
                        os.rename(file, file[0:-5])
                else:
                    # add ".safe" extension
                    if file[-len(ext) :] == ext:
                        Log.d(
                            TAG,
                            'Renaming file "{}" to "{}"'.format(file, "{}.safe".format(file)),
                        )
                        os.rename(file, "{}.safe".format(file))
