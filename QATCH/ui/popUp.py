from QATCH.common.logger import Logger as Log
from QATCH.ui.components.glass_dialog import GlassDialog
from PyQt5 import QtCore, QtWidgets

TAG = "[PopUp]"


class PopUp:

    @staticmethod
    def question_QCM(parent, title, message):  # noqa: N802
        d = GlassDialog(
            parent, title, message,
            buttons=[("@5MHz", "neutral", 0), ("@10MHz", "primary", 1)],
            icon_type="question",
        )
        d.exec_()
        result = d.result_value()
        if result == 1:
            Log.i(TAG, "Quartz Crystal Sensor installed on the QATCH Q-1 Device: @10MHz")
        else:
            Log.i(TAG, "Quartz Crystal Sensor installed on the QATCH Q-1 Device: @5MHz")
        return result

    @staticmethod
    def question_FW(parent, title, message, details="", onlyOK=False):  # noqa: N802, N803
        if onlyOK:
            buttons = [("Awesome!", "primary", 1)]
        else:
            buttons = [("No", "neutral", 0), ("Yes", "primary", 1)]
        d = GlassDialog(
            parent, title, message,
            details=details,
            buttons=buttons,
            icon_type="question",
        )
        d.exec_()
        return d.result_value() == 1

    @staticmethod
    def warning(parent, title, message):
        d = GlassDialog(
            parent, title, message,
            buttons=[("OK", "primary", 1)],
            icon_type="warning",
        )
        d.exec_()

    @staticmethod
    def question(parent, title, message, default=False):
        buttons = [("No", "neutral", 0), ("Yes", "primary", 1)]
        d = GlassDialog(
            parent, title, message,
            buttons=buttons,
            icon_type="question",
        )
        d.exec_()
        return d.result_value() == 1

    @staticmethod
    def critical(  # noqa: N802 (name kept for public API compatibility)
        parent, title, message, details="", question=False, ok_only=False, btn1_text="Retry"
    ):
        if question:
            buttons = [("No", "neutral", 0), ("Yes", "primary", 1)]
            icon_type = "question"
        elif ok_only:
            buttons = [("OK", "primary", 1)]
            icon_type = "critical"
        else:
            buttons = [("Ignore", "neutral", 0), (btn1_text, "danger", 1)]
            icon_type = "critical"
        d = GlassDialog(
            parent, title, message,
            details=details,
            buttons=buttons,
            icon_type=icon_type,
        )
        d.exec_()
        return d.result_value() == 1

    @staticmethod
    def information(parent, title, message, details=""):
        d = GlassDialog(
            parent, title, message,
            details=details,
            buttons=[("OK", "primary", 1)],
            icon_type="information",
        )
        d.exec_()


class QueryComboBox(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal()
    confirmed = False

    def __init__(self, items, type="item", parent=None):
        super(QueryComboBox, self).__init__(parent)

        layout_h = QtWidgets.QHBoxLayout()
        self.cb = QtWidgets.QComboBox()
        self.cb.addItems(items)
        self.cb.setCurrentIndex(-1)
        self.btn = QtWidgets.QPushButton("OK")
        self.btn.pressed.connect(self.confirm)
        layout_h.addWidget(self.cb)
        layout_h.addWidget(self.btn)

        layout_v = QtWidgets.QVBoxLayout()
        self.tb = QtWidgets.QLabel()
        vowel = 'aeiou'
        a_n = "an" if type[0].lower() in vowel else "a"
        self.tb.setText("Select {} {}:".format(a_n, type))
        layout_v.addWidget(self.tb)
        layout_v.addLayout(layout_h)

        self.setLayout(layout_v)
        self.setWindowTitle("Select")

    def confirm(self):
        self.confirmed = True
        self.close()

    def closeEvent(self, event):  # noqa: N802
        if not self.confirmed:
            self.cb.setCurrentIndex(-1)
        self.finished.emit()
        event.ignore()

    def clickedButton(self):  # noqa: N802
        ret = self.cb.currentIndex()
        self.hide()
        return ret
