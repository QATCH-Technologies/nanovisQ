from time import time

from PyQt5 import QtCore

from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants
from QATCH.processors.Device import serial  # real device hardware


class FLUXControlWorker(QtCore.QThread):
    result = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

    def set_ports(self, controller, next_port):
        self._controller = controller
        self._next_port = next_port

    def run(self):
        success = False
        FLUX_serial = None
        try:
            controller_port = self._controller
            next_port_num = self._next_port

            # Attempt to open port and print errors (if any)
            FLUX_serial = serial.Serial()

            # Configure serial port (assume baud to check before update)
            FLUX_serial.port = controller_port
            FLUX_serial.baudrate = Constants.serial_default_speed  # 115200
            FLUX_serial.stopbits = serial.STOPBITS_ONE
            FLUX_serial.bytesize = serial.EIGHTBITS
            FLUX_serial.timeout = Constants.serial_timeout_ms
            FLUX_serial.write_timeout = Constants.serial_writetimeout_ms
            FLUX_serial.open()

            step = next_port_num if next_port_num > 1 else 0  # re-home on "step 1"

            tecs = []
            if step in [0, 1, 2]:
                tecs.append("L")
            if step in [2, 3, 4]:
                tecs.append("C")
            if step in [4, 5, 6]:
                tecs.append("R")
            tec = ", ".join(tecs)

            probe = str(next_port_num)

            # NOTE: The stepper is interrupted in FW by pending serial
            #       so the STEP command must be last in the order sent
            flux_cmds = f"TEC {tec}\nPROBE {probe}\nSTEP {step}\n"

            Log.d(f"Port {next_port_num} control cmds: {flux_cmds}")

            # Read and show the TEC temp status from the device
            FLUX_serial.write(flux_cmds.encode())
            timeoutAt = time() + Constants.stepper_timeout_sec
            flux_reply = ""
            # timeout needed if old FW
            while time() < timeoutAt:
                if "Stepper: DONE!" in flux_reply:
                    break
                while (
                    FLUX_serial.in_waiting == 0 and time() < timeoutAt
                ):  # timeout needed if old FW:
                    QtCore.QThread.msleep(5)
                waiting = FLUX_serial.in_waiting
                if waiting > 0:
                    flux_reply += FLUX_serial.read(waiting).decode(errors="replace")

            if time() < timeoutAt:
                if (
                    "Stepper: DONE!" in flux_reply  # indicates stepper finished moving
                    and "Unknown input." not in flux_reply  # indicates TEC or PROBE cmd issue
                    and "Stopped" not in flux_reply
                ):  # indicates serial interrupted home action
                    Log.i(f"SUCCESS - Port {next_port_num} selected.")
                    success = True
                else:
                    Log.e(f"FAILURE - Port {next_port_num} NOT selected. Unexpected reply...")
            else:
                Log.e(f"TIMEOUT - Port {next_port_num} NOT selected. Controller timeout...")

            if not success:
                Log.d("Error Details: (serial response from port selection request)")
                for line in flux_reply.splitlines():
                    Log.d(f"ERROR >> {line}")
                Log.d('Expected last line from controller to be "Stepper: DONE!"')

        except Exception as e:
            Log.e(f"FLUXControl ERROR: {e}")

        finally:
            if FLUX_serial is not None and FLUX_serial.is_open:
                FLUX_serial.close()

            # always notify caller even on early failures
            self.result.emit(success)
            self.finished.emit()
