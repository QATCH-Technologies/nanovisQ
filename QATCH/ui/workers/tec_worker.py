import os
import sys
from time import localtime, strftime, time, monotonic
import numpy as np
from PyQt5 import QtCore
from serial import serialutil
from QATCH.common.architecture import Architecture, OSType
from QATCH.common.fileStorage import FileStorage
from QATCH.common.logger import Logger as Log
from QATCH.processors.Device import serial  # real device hardware


class TECWorker(QtCore.QThread):
    update_now = QtCore.pyqtSignal()
    auto_off = QtCore.pyqtSignal()
    volt_err = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()
    lTemp_setText = QtCore.pyqtSignal(str)
    lTemp_setStyleSheet = QtCore.pyqtSignal(str)
    infobar_setText = QtCore.pyqtSignal(str)

    port = None

    slider_value = 0
    slider_down = False
    slider_enable = False

    _tec_initialized = False
    _tec_state = "OFF"
    _tec_cycling = False
    _tec_status = "CYCLE"
    _tec_setpoint = -1
    _tec_temp = 0
    _tec_power = -1
    _tec_voltage = "0V (0)"
    _tec_voltage_error_seen = False
    _tec_offset1 = "0"
    _tec_offset2 = "0"
    _tec_locked = False
    _tec_update_now = False
    _tec_stop_thread = False
    _tec_debug = False
    _tec_out_of_sync = 0  # counter, task aborts if it gets to 3

    _task_timer = None
    _task_rate = 5000
    _task_timeout = 15 * 60 * 1000 / _task_rate
    _task_counter = 0
    _task_active = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self._task_timer = QtCore.QTimer(self)
        self._task_timer.timeout.connect(self.task)
        self.update_now.connect(self.task_update)

    def set_port(self, value):
        self.port = value

    def set_slider_value(self, value):
        self.slider_value = value

    def set_slider_down(self, value):
        self.slider_down = value

    def set_slider_enable(self, value):
        self.slider_enable = value

    def run(self):
        Log.i(
            TAG,
            f"Temp Control started ({strftime('%Y-%m-%d %H:%M:%S', localtime())})",
        )
        self.infobar_setText.emit(
            "<font color=#0000ff> Infobar </font><font color={}>{}</font>".format(
                "#333333", "Temp Control started."
            )
        )

        self._task_active = True
        self._task_timer.start(self._task_rate)
        self.task()  # fire immediately

    def task(self):
        try:
            if not self._task_active:
                return
            if True:  # was while()
                try:
                    sp = ""  # only update TEC if changed
                    # Log.d("TEC debug: {}, {}, {}".format(
                    #     self.slider_value, self._tec_setpoint, self.slider_down))
                    if self.slider_value != self._tec_setpoint and not self.slider_down:
                        # Try to update now to re-sync; if that fails, then auto-off.
                        if self._tec_out_of_sync < 3:
                            Log.d("Scheduling TEC for immediate update (out-of-sync)!")
                            self._tec_out_of_sync += 1
                            self._tec_update_now = True
                        else:
                            Log.w("Shutting down TEC to re-sync states (out-of-sync)!")
                            self._tec_out_of_sync = 0
                            new_l1 = "[AUTO-OFF ERROR]"
                            self._tec_update("OFF")
                            self._task_stop()
                            self.auto_off.emit()
                            self.lTemp_setText.emit(new_l1)
                            self.lTemp_setStyleSheet.emit("background-color: {}".format("red"))
                            return
                    else:
                        # Log.d("TEC is in-sync!")
                        self._tec_out_of_sync = 0
                    if self._tec_update_now and not self._tec_locked:
                        sp = self.slider_value
                    if self.slider_enable:
                        Log.d(
                            f"{self._task_counter:.0f}/{self._task_timeout:.0f}: Querying TEC status..."
                        )
                        self._tec_update(sp)
                        self._tec_locked = False
                        if self.slider_value == self._tec_setpoint and not self.slider_down:
                            # Log.d("TEC sync success!")
                            self._tec_out_of_sync = 0
                    elif not self._tec_stop_thread:
                        if not self._tec_locked:
                            Log.d("Temp Control is locked while main thead is busy!")
                        self._tec_locked = True
                        return  # stop task silently
                    self._tec_update_now = False
                    if self._tec_update_now == False:
                        sp = self.slider_value
                    pv = self._tec_temp
                    op = self._tec_power
                    self._task_counter += 1
                    if sp == 0.00:
                        sp = 0.25
                    if op == 0 or self._task_counter > self._task_timeout:
                        if self._tec_voltage_error_seen:
                            new_l1 = "[VOLTAGE ERROR]"
                            self._tec_voltage_error_seen = False
                            self.volt_err.emit()
                        else:
                            new_l1 = "[AUTO-OFF ERROR]" if np.isnan(pv) else "[AUTO-OFF TIMEOUT]"
                            self._tec_update("OFF")
                            self._task_stop()
                            self.auto_off.emit()
                    else:
                        new_l1 = "PV:{0:2.2f}C SP:{1:2.2f}C OP:{2:+04.0f}".format(pv, sp, op)
                    self.lTemp_setText.emit(new_l1)
                    bgcolor = "yellow"
                    if op == 0 or self._task_counter > self._task_timeout:
                        self._tec_stop_thread = True
                        self._tec_update_now = False  # invalidate flag to update TEC again
                        bgcolor = "red" if np.isnan(pv) else "yellow"
                    else:
                        if self._tec_status == "CYCLE":
                            # Log.i(TAG, "{0}: TEC setpoint is rapid cycle. Wait for READY!".format(strftime('%Y-%m-%d %H:%M:%S', localtime())))
                            self.infobar_setText.emit(
                                "<font color=#0000ff> Infobar </font><font color={}><b>{}</b></font>".format(
                                    "#ff0000",
                                    "Temp Control is cycling to target. Wait for READY! (this may take a few minutes)",
                                )
                            )

                        if self._tec_status == "WAIT":
                            self.infobar_setText.emit(
                                "<font color=#0000ff> Infobar </font><font color={}><b>{}</b></font>".format(
                                    "#ff9900",
                                    "Temp Control is stabilizing. Wait for READY! (about one minute remaining)",
                                )
                            )

                        if self._tec_status == "CLOSE":
                            self.infobar_setText.emit(
                                "<font color=#0000ff> Infobar </font><font color={}><b>{}</b></font>".format(
                                    "#ff9900",
                                    "Temp Control is about ready. Wait for READY! (only a few seconds left)",
                                )
                            )

                        if self._tec_status == "STABLE":
                            bgcolor = "lightgreen"
                            # Log.i(TAG, "{0}: TEC setpoint has stabilized. Ready for START!".format(strftime('%Y-%m-%d %H:%M:%S', localtime())))
                            self.infobar_setText.emit(
                                "<font color=#0000ff> Infobar </font><font color={}><b>{}</b></font>".format(
                                    "#009900",
                                    "Temp Control has stabilized. Ready for START!",
                                )
                            )

                        if self._tec_status == "ERROR":
                            bgcolor = "red"
                            Log.e(
                                TAG,
                                "TEC status is in an unkown state. Please restart Temp Control.".format(
                                    strftime("%Y-%m-%d %H:%M:%S", localtime())
                                ),
                            )

                    self.lTemp_setStyleSheet.emit("background-color: {}".format(bgcolor))
                except Exception as e:
                    Log.e(
                        TAG,
                        "ERROR: Port read error during TEC task".format(
                            strftime("%Y-%m-%d %H:%M:%S", localtime())
                        ),
                    )
                    if self._tec_debug:
                        raise e  # debug only
                finally:
                    pass

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
            pass

    def task_update(self):
        self._task_counter = 0
        if self._tec_stop_thread:
            self._task_stop()  # stop immediately
        if self._tec_update_now:
            self.task()  # fire immediately

    def _task_stop(self):
        Log.i(
            TAG,
            f"Temp Control stopped ({strftime('%Y-%m-%d %H:%M:%S', localtime())})",
        )
        self.infobar_setText.emit(
            "<font color=#0000ff> Infobar </font><font color={}>{}</font>".format(
                "#333333", "Temp Control stopped."
            )
        )
        self._task_active = False
        self.finished.emit()  # stop task

    def _tec_update(self, dac=""):
        # Open, write, read and close the port accordingly
        selected_port = self.port
        if selected_port is None:
            selected_port = ""  # Dissallow None
        if selected_port == "CMD_DEV_INFO":
            selected_port = ""  # Dissallow Action

        if self._tec_initialized == False:
            self._tec_initialized = True

            if len(selected_port) == 0:
                Log.e("No active device is currently available for TEC status updates.")
                Log.e('Please connect a device, hit "Reset", and try "Temp Control" again.')
                self._tec_stop_thread = True  # queue thread for 'quit' on next update
                self._tec_update_now = False  # invalidate flag to update TEC again
                self.auto_off.emit()  # toggle off button state now
                # fire next instance soon-ish
                QtCore.QTimer.singleShot(500, self.task_update)
                return

            if not self._is_port_available(selected_port):
                Log.e(f'ERROR: The selected device "{selected_port}" is no longer available.')
                Log.e('Please "Reset" to detect devices and then try "Temp Control" again.')
                self._tec_stop_thread = True  # queue thread for 'quit' on next update
                self._tec_update_now = False  # invalidate flag to update TEC again
                self.auto_off.emit()  # toggle off button state now
                # fire next instance soon-ish
                QtCore.QTimer.singleShot(500, self.task_update)
                return

            # set offsetB, offsetH and offsetC prior to initialization
            if hasattr(Constants, "temp_offset_both"):
                self._tec_update("=OFFSET {0:2.2f}".format(Constants.temp_offset_both))
                Log.d(
                    TAG,
                    "{1}: Set offsetB={0:+02.2f}".format(
                        Constants.temp_offset_both,
                        strftime("%Y-%m-%d %H:%M:%S", localtime()),
                    ),
                )
            if self._tec_stop_thread:
                return
            if hasattr(Constants, "temp_offset_heat"):
                self._tec_update("+OFFSET {0:2.2f}".format(Constants.temp_offset_heat))
                Log.d(
                    TAG,
                    "{1}: Set offsetH={0:+02.2f}".format(
                        Constants.temp_offset_heat,
                        strftime("%Y-%m-%d %H:%M:%S", localtime()),
                    ),
                )
            if self._tec_stop_thread:
                return
            if hasattr(Constants, "temp_offset_cool"):
                self._tec_update("-OFFSET {0:2.2f}".format(Constants.temp_offset_cool))
                Log.d(
                    TAG,
                    "{1}: Set offsetC={0:+02.2f}".format(
                        Constants.temp_offset_cool,
                        strftime("%Y-%m-%d %H:%M:%S", localtime()),
                    ),
                )
            if self._tec_stop_thread:
                return

            if (
                hasattr(Constants, "tune_pid_cp")
                and hasattr(Constants, "tune_pid_ci")
                and hasattr(Constants, "tune_pid_cd")
                and hasattr(Constants, "tune_pid_hp")
                and hasattr(Constants, "tune_pid_hi")
                and hasattr(Constants, "tune_pid_hd")
            ):
                self._tec_update(
                    "TUNE {0:.3g},{1:.3g},{2:.3g},{3:.3g},{4:.3g},{5:.3g}".format(
                        Constants.tune_pid_cp,
                        Constants.tune_pid_ci,
                        Constants.tune_pid_cd,  # cool PID
                        Constants.tune_pid_hp,
                        Constants.tune_pid_hi,
                        Constants.tune_pid_hd,
                    )
                )  # heat PID
                Log.w(
                    TAG,
                    "{6}: Set PID tuning per Constants.py parameters: {0:.3g},{1:.3g},{2:.3g},{3:.3g},{4:.3g},{5:.3g}".format(
                        Constants.tune_pid_cp,
                        Constants.tune_pid_ci,
                        Constants.tune_pid_cd,  # cool PID
                        Constants.tune_pid_hp,
                        Constants.tune_pid_hi,
                        Constants.tune_pid_hd,  # heat PID
                        strftime("%Y-%m-%d %H:%M:%S", localtime()),
                    ),
                )
            if self._tec_stop_thread:
                return

        # Attempt to open port and print errors (if any)
        TEC_serial = serial.Serial()
        try:
            # Configure serial port (assume baud to check before update)
            TEC_serial.port = selected_port
            TEC_serial.baudrate = Constants.serial_default_speed  # 115200
            TEC_serial.stopbits = serial.STOPBITS_ONE
            TEC_serial.bytesize = serial.EIGHTBITS
            TEC_serial.timeout = Constants.serial_timeout_ms
            TEC_serial.write_timeout = Constants.serial_writetimeout_ms
            TEC_serial.open()

            # Handle special values, only send to TEC FW if not the current setpoint
            if str(dac).isnumeric():
                if int(dac) < 0 or int(dac) > 60:
                    dac = "OFF"  # turn off if temp is outside valid range
                if int(dac) == 0:
                    dac = 0.25  # temp sensor clips at 0, so target 0.25C
                Log.i(
                    TAG,
                    "Temp Control setpoint: {1}C".format(
                        strftime("%Y-%m-%d %H:%M:%S", localtime()), dac
                    ),
                )
                self.infobar_setText.emit(
                    "<font color=#0000ff> Infobar </font><font color={}><b>{}</b></font>".format(
                        "#ff0000",
                        "Cycling temperature to Temp Control setpoint... please wait...",
                    )
                )
                self._tec_cycling = True

            # Read and show the TEC temp status from the device
            TEC_serial.write("temp {}\n".format(dac).encode())
            timeoutAt = time() + 3
            temp_reply = ""
            lines_in_reply = 11
            # timeout needed if old FW
            while temp_reply.count("\n") < lines_in_reply and time() < timeoutAt:
                while (
                    TEC_serial.in_waiting == 0 and time() < timeoutAt
                ):  # timeout needed if old FW:
                    pass
                temp_reply += TEC_serial.read(TEC_serial.in_waiting).decode()

            if time() < timeoutAt:
                temp_reply = temp_reply.split("\n")
                actual_lines = len(temp_reply)
                # ends with blank line (+1)
                sl = actual_lines - (lines_in_reply + 1)
                status_line = temp_reply[sl + 0]  # line 1
                setpoint_line = temp_reply[sl + 1]  # line 2
                power_line = temp_reply[sl + 2]  # line 3
                voltage_line = temp_reply[sl + 3]  # line 4
                stable_total_line = temp_reply[sl + 4]  # line 5
                min_max_line = temp_reply[sl + 5]  # line 6
                temp_status_line = temp_reply[sl + 6]  # line 7
                ambient_line = temp_reply[sl + 7]  # line 8
                temp_line = temp_reply[sl + 8]  # line 9
                offsets_line = temp_reply[sl + 9]  # line 10 (unused)
                tune_pid_line = temp_reply[sl + 10]  # line 11 (unused)
                status_text = status_line.split(":")[1].strip()
                # cycle, wait, close, stable, error
                cycle_stable = status_text.split(",")[0]
                heat_cool_off = status_text.split(",")[1]
                setpoint_val = float(setpoint_line.split(":")[1].strip())
                power_val = int(power_line.split(":")[1].strip())
                voltage_val = voltage_line.split(":")[1].strip()
                voltage_volts = float(voltage_val.split()[0][0:-1])
                voltage_raw = int(voltage_val.split()[1][1:-1])
                stable_total = stable_total_line.split(":")[1].strip()
                stable = int(stable_total.split(",")[0])
                total = int(stable_total.split(",")[1])
                min_max = min_max_line.split(":")[1].strip()
                min = float(min_max.split(",")[0])
                max = float(min_max.split(",")[1])
                if min == 50:
                    min = -1
                if max == 0:
                    max = -1
                temp_status = temp_status_line.split(":")[1].strip()
                ambient = float(ambient_line.split(":")[1].strip())
                temp = float(temp_line.split(":")[1].strip())
                offsets_text = offsets_line.split(":")[1].strip().split(",")
                tune_pid_text = tune_pid_line.split(":")[1].strip().split(",")
                # throw variables into global module scope
                self._tec_status = cycle_stable
                self._tec_state = heat_cool_off
                if "COOL" == self._tec_state:
                    power_val = -power_val
                self._tec_setpoint = setpoint_val
                self._tec_temp = temp
                self._tec_power = power_val
                self._tec_voltage = voltage_val
                self._tec_offset1 = offsets_text[0]  # first: A (always)
                self._tec_offset2 = offsets_text[-1]  # last: M (measure)
                self._tec_pid_tune = []
                for kp in tune_pid_text:
                    self._tec_pid_tune.append(float(kp))

                if "VOLTAGE" == self._tec_state:
                    self._tec_voltage_error_seen = True
                    Log.e(f"External voltage is out of bounds: {voltage_val}")

                # Append to log file for temperature controller
                # checks the path for the header insertion
                tec_log_path = FileStorage.DEV_populate_path(Constants.tec_log_path, 0)
                os.makedirs(os.path.split(tec_log_path)[0], exist_ok=True)
                header_exists = os.path.exists(tec_log_path)
                with open(tec_log_path, "a") as tempFile:
                    if not header_exists:
                        tempFile.write(
                            "Date/Time,Command,Status/Mode,Power(raw),Stable/Total(sec),Min/Max(C),Temp(C),Ambient(C)\n"
                        )
                    log_line = "{},{},{},{},{},{},{},{}\n".format(
                        strftime("%Y-%m-%d %H:%M:%S", localtime()),  # Date/Time
                        "SET '{}'".format(dac) if not dac == "" else "GET",  # Command
                        # Status/Mode
                        "{}/{}".format(cycle_stable, heat_cool_off),
                        # Power(raw)
                        power_val,
                        # Stable/Total(sec)
                        "{}/{}".format(stable, total),
                        # Min/Max(C)
                        "{}/{}".format(min, max),
                        # Temp(C)
                        temp,
                        ambient,
                    )  # Ambient(C)
                    tempFile.write(log_line)
            else:
                Log.e(
                    TAG,
                    "ERROR: Timeout during check and/or update to TEC controller.".format(
                        strftime("%Y-%m-%d %H:%M:%S", localtime())
                    ),
                )
        except serialutil.SerialException as e:
            self._tec_state = "OFF" if dac == "OFF" else "ERROR"
            # Ignore the following exception types:
            # errno exception           cause
            ##################################################
            #   2   FileNotFoundError   scanning the source
            #  13   PermissionError     active run in progress
            if not any([s in str(e) for s in ["FileNotFoundError", "PermissionError"]]):
                Log.e(
                    TAG,
                    "ERROR: Serial exception reading port to check and/or update TEC controller.".format(
                        strftime("%Y-%m-%d %H:%M:%S", localtime())
                    ),
                )
                if self._tec_debug:
                    raise e  # debug only
        except PermissionError as e:
            self._tec_state = "OFF" if dac == "OFF" else "ERROR"
            if (
                "OFFSET" in dac
            ):  # unable to open port during initialize phase is a hard-stop condition
                # if dac == "OFF" else "ERROR" # always off here, since we are OFF in 500ms
                self._tec_state = "OFF"
                Log.e(
                    f'ERROR: The selected device "{selected_port}" cannot be opened. Is the port already open in another program?'
                )
                Log.e('Please close the device port, hit "Reset", and try "Temp Control" again.')
                self._tec_stop_thread = True  # queue thread for 'quit' on next update
                self._tec_update_now = False  # invalidate flag to update TEC again
                self.auto_off.emit()  # toggle off button state now
                # fire next instance soon-ish
                QtCore.QTimer.singleShot(500, self.task_update)
        except Exception as e:
            self._tec_state = "OFF" if dac == "OFF" else "ERROR"
            if not any([s in str(e) for s in ["Permission"]]):
                Log.e(
                    TAG,
                    "ERROR: Failure reading port to check and/or update TEC controller.".format(
                        strftime("%Y-%m-%d %H:%M:%S", localtime())
                    ),
                )
            else:
                Log.e(
                    TAG,
                    "ERROR: File permission error occurred while logging TEC controller data.".format(
                        strftime("%Y-%m-%d %H:%M:%S", localtime())
                    ),
                )
            if self._tec_debug:
                raise e  # debug only
        finally:
            if TEC_serial.is_open:
                TEC_serial.close()

    ###########################################################################
    # Automatically selects the serial ports for Teensy (macox/windows)
    ###########################################################################
    @staticmethod
    def get_ports():
        return serial.enumerate()
        from serial.tools import list_ports

        from QATCH.common.architecture import Architecture, OSType

        if Architecture.get_os() is OSType.macosx:
            import glob

            return glob.glob("/dev/tty.usbmodem*")
        elif Architecture.get_os() is OSType.linux:
            import glob

            return glob.glob("/dev/ttyACM*")
        else:
            found_ports = []
            port_connected = []
            found = False
            ports_avaiable = list(list_ports.comports())
            for port in ports_avaiable:
                if port[2].startswith("USB VID:PID=16C0:0483"):
                    found = True
                    port_connected.append(port[0])
            if found:
                found_ports = port_connected
            return found_ports

    ###########################################################################
    # Checks if the serial port is currently connected
    ###########################################################################
    def _is_port_available(self, port):
        """
        :param port: Port name to be verified.
        :return: True if the port is connected to the host :rtype: bool.
        """
        # dm = Discovery()
        # if self._serial.net_port is None:
        #     net_exists = False
        # else:
        #     net_exists = dm.ping(self._serial.net_port)
        for p in self.get_ports():
            if p == port:
                return True
        # if port is None:
        #     if len(dm.doDiscover()) > 0:
        #         return net_exists
        return False
