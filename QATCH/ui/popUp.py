from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from PyQt5 import QtCore, QtGui, QtWidgets
import os

TAG = "[PopUp]"

###############################################################################
# Warning dialog module
###############################################################################


class PopUp:

    ###########################################################################
    # Shows a pop-up question dialog with yes and no buttons (unused)
    ###########################################################################
    @staticmethod
    def question_QCM(parent, title, message):
        """
        :param parent: Parent window for the dialog.
        :param title: Title of the dialog :type title: str.
        :param message: Message to be shown in the dialog :type message: str.
        :return: 1 if button1 was pressed, 0 if button2   :rtype: int.
        """
        if not isinstance(parent, QtWidgets.QWidget):
            parent = None  # make dialog have no parent if invalid type

        # ans = QtWidgets.QMessageBox.question(parent, title, message, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        # if ans == QtWidgets.QMessageBox.Yes:
        #    Log.d('Si')
        #    return True
        # elif ans == QtWidgets.QMessageBox.No:
        #    Log.d('No')
        #    return False
        width = 340
        height = 220
        area = QtWidgets.QDesktopWidget().availableGeometry()
        left = int((area.width() - width) / 2)
        top = int((area.height() - height) / 2)
        box = QtWidgets.QMessageBox(parent)
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setWindowTitle(title)
        box.setGeometry(left, top, width, height)
        box.setText(message)
        box.setStandardButtons(QtWidgets.QMessageBox.Yes |
                               QtWidgets.QMessageBox.No)
        button1 = box.button(QtWidgets.QMessageBox.Yes)
        button1.setText('@10MHz')
        button2 = box.button(QtWidgets.QMessageBox.No)
        button2.setText(' @5MHz')
        box.exec_()

        if box.clickedButton() == button1:
            Log.i(TAG, 'Quartz Crystal Sensor installed on the QATCH Q-1 Device: @10MHz')
            return 1
        elif box.clickedButton() == button2:
            Log.i(TAG, 'Quartz Crystal Sensor installed on the QATCH Q-1 Device: @5MHz')
            return 0

    ###########################################################################
    # Shows a pop-up question dialog with Yes/No (or Ok) buttons
    ###########################################################################
    @staticmethod
    def question_FW(parent, title, message, details="", onlyOK=False):
        """
        :param parent: Parent window for the dialog.
        :param title: Title of the dialog :type title: str.
        :param message: Message to be shown in the dialog :type message: str.
        :return: 1 if button1 was pressed, 0 if button2   :rtype: int.
        """
        if not isinstance(parent, QtWidgets.QWidget):
            parent = None  # make dialog have no parent if invalid type

        # ans = QtWidgets.QMessageBox.question(parent, title, message, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        # if ans == QtWidgets.QMessageBox.Yes:
        #    Log.d('Si')
        #    return True
        # elif ans == QtWidgets.QMessageBox.No:
        #    Log.d('No')
        #    return False
        width = 340
        height = 220
        area = QtWidgets.QDesktopWidget().availableGeometry()
        left = int((area.width() - width) / 2)
        top = int((area.height() - height) / 2)
        box = QtWidgets.QMessageBox(parent)
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/download_icon.ico')
        box.setIconPixmap(QtGui.QPixmap(icon_path))
        box.setWindowTitle(title)
        box.setGeometry(left, top, width, height)
        box.setText(message)
        box.setDetailedText(details)

        if not onlyOK:
            box.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            box.setDefaultButton(QtWidgets.QMessageBox.Yes)
            button1 = box.button(QtWidgets.QMessageBox.Yes)
            button1.setText('Yes')
            button2 = box.button(QtWidgets.QMessageBox.No)
            button2.setText('No')
        else:
            box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            box.setDefaultButton(QtWidgets.QMessageBox.Ok)
            button1 = box.button(QtWidgets.QMessageBox.Ok)
            button1.setText('Awesome!')

        box.exec_()

        return box.clickedButton() == button1

    ###########################################################################
    # Shows a Pop up warning dialog with Ok button
    ###########################################################################
    @staticmethod
    def warning(parent, title, message):
        """
        :param parent: Parent window for the dialog.
        :param title: Title of the dialog :type title: str.
        :param message: Message to be shown in the dialog :type message: str.
        """
        if not isinstance(parent, QtWidgets.QWidget):
            parent = None  # make dialog have no parent if invalid type

        QtWidgets.QMessageBox.warning(
            parent, title, message, QtWidgets.QMessageBox.Ok)
        # msgBox=QtWidgets.QMessageBox.warning(parent, title, message, QtWidgets.QMessageBox.Ok)
        # msgBox = QtWidgets.QMessageBox()
        # msgBox.setIconPixmap( QtGui.QPixmap("favicon.png"))
        # msgBox.exec_()

    ###########################################################################
    # Shows a pop-up question dialog with yes and no buttons
    ###########################################################################
    @staticmethod
    def question(parent, title, message, default=False):
        """
        :param parent: Parent window for the dialog.
        :param title: Title of the dialog :type title: str.
        :param message: Message to be shown in the dialog :type message: str.
        :return: True if Yes button was pressed :rtype: bool.
        """
        if not isinstance(parent, QtWidgets.QWidget):
            parent = None  # make dialog have no parent if invalid type

        defaultButton = QtWidgets.QMessageBox.Yes if default else QtWidgets.QMessageBox.No
        ans = QtWidgets.QMessageBox.question(  # center dialog over parent (by default)
            parent, title, message, QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, defaultButton)
        if ans == QtWidgets.QMessageBox.Yes:
            return True
        else:
            return False

    ###########################################################################
    # Shows a Pop up critical dialog with Retry/Ignore buttons
    ###########################################################################
    @staticmethod
    def critical(parent, title, message, details="", question=False, ok_only=False, btn1_text="Retry"):
        """
        :param parent: Parent window for the dialog (must inherit QWidget).
        :param title: Title of the dialog :type title: str.
        :param message: Message to be shown in the dialog :type message: str.
        """
        if not isinstance(parent, QtWidgets.QWidget):
            parent = None  # make dialog have no parent if invalid type

        # return (QtWidgets.QMessageBox.critical(parent, title, message, QtWidgets.QMessageBox.Retry, QtWidgets.QMessageBox.Ignore)
        #     == QtWidgets.QMessageBox.Retry)
        width = 340
        height = 220
        # area = QtWidgets.QDesktopWidget().availableGeometry()
        # left = int((area.width() - width) / 2)
        # top = int((area.height() - height) / 2)
        box = QtWidgets.QMessageBox(parent)
        if question:
            box.setIcon(QtWidgets.QMessageBox.Question)
        else:
            box.setIcon(QtWidgets.QMessageBox.Critical)
        box.setWindowTitle(title)
        # box.setGeometry(left, top, width, height)
        box.setFixedSize(width, height)  # center dialog over parent (by default)
        box.setText(message)
        box.setDetailedText(details)
        if question:
            box.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            box.setDefaultButton(QtWidgets.QMessageBox.No)
            button1 = box.button(QtWidgets.QMessageBox.Yes)
            button1.setText('Yes')
            button2 = box.button(QtWidgets.QMessageBox.No)
            button2.setText('No')
        elif ok_only:
            box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            box.setDefaultButton(QtWidgets.QMessageBox.Ok)
            button1 = box.button(QtWidgets.QMessageBox.Ok)
            button1.setText('Ok')
        else:
            box.setStandardButtons(
                QtWidgets.QMessageBox.Retry | QtWidgets.QMessageBox.Ignore)
            box.setDefaultButton(QtWidgets.QMessageBox.Retry)
            button1 = box.button(QtWidgets.QMessageBox.Retry)
            button1.setText(btn1_text)
            button2 = box.button(QtWidgets.QMessageBox.Ignore)
            button2.setText('Ignore')
        box.exec_()

        return box.clickedButton() == button1

    ###########################################################################
    # Shows a Pop up information dialog with Ok button
    ###########################################################################
    @staticmethod
    def information(parent, title, message, details=""):
        """
        :param parent: Parent window for the dialog.
        :param title: Title of the dialog :type title: str.
        :param message: Message to be shown in the dialog :type message: str.
        """
        if not isinstance(parent, QtWidgets.QWidget):
            parent = None  # make dialog have no parent if invalid type

        QtWidgets.QMessageBox.information(
            parent, title, message, QtWidgets.QMessageBox.Ok)
        # msgBox=QtWidgets.QMessageBox.warning(parent, title, message, QtWidgets.QMessageBox.Ok)
        # msgBox = QtWidgets.QMessageBox()
        # msgBox.setIconPixmap( QtGui.QPixmap("favicon.png"))
        # msgBox.exec_()


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

    def closeEvent(self, event):
        if not self.confirmed:
            self.cb.setCurrentIndex(-1)
        self.finished.emit()  # queue call for clickedButton
        event.ignore()  # keep open until clickedButton call

    def clickedButton(self):
        ret = self.cb.currentIndex()
        self.hide()
        return ret
