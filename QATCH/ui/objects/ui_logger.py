import logging
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from QATCH.common.architecture import Architecture, OSType
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants


class UILogger:
    def setupUi(self, MainWindow4):
        MainWindow4.setMinimumSize(QtCore.QSize(1000, 100))
        MainWindow4.move(0, 0)
        self.centralwidget = QtWidgets.QWidget(MainWindow4)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)

        logTextBox = QTextEditLogger(self.centralwidget)

        # log to text box
        logTextBox.setFormatter(
            logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt=None)
        )
        logging.getLogger("QATCH").addHandler(logTextBox)
        Log._show_user_info()

        MainWindow4.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow4)
        QtCore.QMetaObject.connectSlotsByName(MainWindow4)

    def retranslateUi(self, MainWindow4):
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/qatch-icon.png")
        MainWindow4.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        MainWindow4.setWindowTitle(
            _translate(
                "MainWindow4",
                "{} {} - Console".format(Constants.app_title, Constants.app_version),
            )
        )


class QTextEditLogger(logging.Handler, QtCore.QObject):
    appendInfoText = QtCore.pyqtSignal(str)
    appendDebugText = QtCore.pyqtSignal(str)
    forceRepaintEvents = False
    progressMode = False

    def __init__(self, parent):
        super().__init__()
        QtCore.QObject.__init__(self)

        # Initialize tab screen
        self.tabs = QtWidgets.QTabWidget()
        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()

        # Add tabs
        self.logInfo = QtWidgets.QTextEdit(parent)
        self.logInfo.setReadOnly(True)
        layout_v1 = QtWidgets.QVBoxLayout()
        layout_v1.addWidget(self.logInfo)
        self.tab1.setLayout(layout_v1)
        self.tabs.addTab(self.tab1, "Info")

        self.logDebug = QtWidgets.QTextEdit(parent)
        self.logDebug.setReadOnly(True)
        layout_v2 = QtWidgets.QVBoxLayout()
        layout_v2.addWidget(self.logDebug)
        self.tab2.setLayout(layout_v2)
        self.tabs.addTab(self.tab2, "Debug")

        self.tabs.setTabPosition(QtWidgets.QTabWidget.East)

        layout_v = QtWidgets.QVBoxLayout()
        layout_v.addWidget(self.tabs)
        parent.setLayout(layout_v)

        self.appendInfoText.connect(self.appendToInfo)  # logInfo.insertHtml)
        # logDebug.insertHtml)
        self.appendDebugText.connect(self.appendToDebug)
        self.last_record_msg = None

    def appendToInfo(self, html):
        if self.forceRepaintEvents and "[Device] ERROR:" in html:
            return  # do not show serial errors during firmware update on info console
        self.logInfo.moveCursor(QtGui.QTextCursor.End, QtGui.QTextCursor.MoveAnchor)
        if self.progressMode:
            # replace the most recent line with this new html line
            self.logInfo.textCursor().deletePreviousChar()
            self.logInfo.moveCursor(QtGui.QTextCursor.StartOfLine, QtGui.QTextCursor.MoveAnchor)
            self.logInfo.moveCursor(QtGui.QTextCursor.End, QtGui.QTextCursor.KeepAnchor)
            self.logInfo.textCursor().removeSelectedText()
        self.logInfo.insertHtml(html)
        self.logInfo.ensureCursorVisible()
        if self.forceRepaintEvents:
            self.logInfo.repaint()

    def appendToDebug(self, html):
        self.logDebug.moveCursor(QtGui.QTextCursor.End, QtGui.QTextCursor.MoveAnchor)
        self.logDebug.insertHtml(html)
        self.logDebug.ensureCursorVisible()
        if self.forceRepaintEvents:
            self.logDebug.repaint()
        if "GUI: Clear console window" in html:
            self.logInfo.clear()
        if "GUI: Force repaint events" in html:
            self.forceRepaintEvents = True
        if "GUI: Normal repaint events" in html:
            self.forceRepaintEvents = False
        if "GUI: Toggle progress mode" in html:
            self.progressMode = not self.progressMode

    def emit(self, record):
        msg = self.format(record)
        if msg == self.last_record_msg:
            # This must be print(), not Log.d(), or else an endless loop could occur!
            print(msg, "(duplicate record ignored)")
            return  # ignore duplicate records when they are handled back-to-back
        self.last_record_msg = msg
        msg = msg[msg.index(" ") + 1 :]  # trim date from console
        html_fmt = '<font style=\'font-family:"Lucida Console","Courier New",monospace;color:{};font-weight:{};\'>{}</font><br/><br/>'
        color = "black" if record.levelno <= logging.INFO else "red"
        weight = "normal" if record.levelno <= logging.WARNING else "bold"
        time_only = msg[0 : msg.index(",")]
        padding = "&nbsp;&nbsp;&nbsp;" if weight == "normal" else "&nbsp;&nbsp;"
        msg_info = time_only + padding + record.msg
        msg_debug = msg
        html_info = html_fmt.format(color, weight, msg_info)
        html_debug = html_fmt.format(color, weight, msg_debug)
        if record.levelno >= logging.INFO:
            self.appendInfoText.emit(html_info)
        self.appendDebugText.emit(html_debug)
