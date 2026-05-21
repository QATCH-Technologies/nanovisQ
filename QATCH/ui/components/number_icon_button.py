from time import time
from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.logger import Logger as Log
from QATCH.processors.Device import serial  # real device hardware


class NumberIconButton(QtWidgets.QToolButton):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._error = False
        self._value = 1
        self._iconSize = QtCore.QSize(32, 32)

        self.setIconSize(self._iconSize)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.updateIcon()
        self.clicked.connect(self.advance)

    def advance(self):
        self._value += 1
        if self._value > 6 or self._error:
            self._error = False
            self._value = 1
        self.updateIcon()

    def value(self):
        # Used to get the current port step by the caller
        return self._value

    def setIconError(self):
        self._error = True
        self.updateIcon()  # redraw with error colors, clears on next "advance"

    def updateIcon(self, running=False):
        self.setIcon(self.makeIcon(self._value, running))

    def makeIcon(self, number, running=False, size=None):
        if size is None:
            size = self.iconSize()

        # Define hourglass shape vertices (16x16)
        points = [
            QtCore.QPoint(8 + 2, 8 + 2),  # Top Left
            QtCore.QPoint(8 + 14, 8 + 2),  # Top Right
            QtCore.QPoint(8 + 9, 8 + 8),  # Middle Right
            QtCore.QPoint(8 + 14, 8 + 14),  # Bottom Right
            QtCore.QPoint(8 + 2, 8 + 14),  # Bottom Left
            QtCore.QPoint(8 + 7, 8 + 8),  # Middle Left
        ]

        if not running:
            pm_hourglass = QtGui.QPixmap(size)
            self._beginPainter(pm_hourglass, False)

            # Circle (disabled)
            self.painter.drawEllipse(pm_hourglass.rect().adjusted(2, 2, -2, -2))

            # Hourglass (disabled)
            self.painter.drawPolygon(QtGui.QPolygon(points))

            self.painter.end()

        pm_number = QtGui.QPixmap(size)
        self._beginPainter(pm_number)

        # Circle (enabled)
        self.painter.drawEllipse(pm_number.rect().adjusted(2, 2, -2, -2))

        # Number (enabled)
        if not self._error:
            self.painter.drawText(pm_number.rect(), QtCore.Qt.AlignCenter, str(number))
        else:
            # Change pen to red, mark an X instead of the port number
            pen = QtGui.QPen(QtGui.QColor("#FF0000"), 2)
            self.painter.setPen(pen)

            self.painter.drawText(pm_number.rect(), QtCore.Qt.AlignCenter, "X")

        self.painter.end()

        icon = QtGui.QIcon()
        if not running:
            icon.addPixmap(pm_hourglass, QtGui.QIcon.Mode.Disabled, QtGui.QIcon.State.On)
            icon.addPixmap(pm_number, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On)
        else:
            icon.addPixmap(pm_number)  # same for enabled and disabled

        return icon

    def _beginPainter(self, device, enabled=True):
        device.fill(QtCore.Qt.transparent)

        self.painter = QtGui.QPainter()
        self.painter.begin(device)
        self.painter.setRenderHint(QtGui.QPainter.Antialiasing)

        pen = QtGui.QPen(QtGui.QColor("#444444" if enabled else "#888888"), 2)
        self.painter.setPen(pen)

        font = QtGui.QFont(self.font())
        font.setBold(True)
        font.setPointSize(int(self.iconSize().height() * 0.35))
        self.painter.setFont(font)


class FLUXControl(QtCore.QThread):
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
