"""
QATCH.ui.mainWindow_ui_logger

Log-console UI for the nanovisQ application with glass-morphism styling
that matches the modeMenuScrollArea palette from ui_main_theme.qss.

Author(s)
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-05-05
"""

import logging
import os

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants

# ---------------------------------------------------------------------------
# Glass-morphism QSS — applied to the QTabWidget and cascades to children
# ---------------------------------------------------------------------------

_LOGGER_GLASS_QSS = """
    /* ---- Tab widget container ---- */
    QTabWidget {
        background: transparent;
    }
    QTabWidget::pane {
        background: rgba(255, 255, 255, 120);
        border: 1px solid rgba(255, 255, 255, 200);
        border-radius: 8px;
    }

    /* ---- Tab bar (East / vertical orientation) ---- */
    QTabBar {
        background: transparent;
    }
    QTabBar::tab {
        background: transparent;
        color: rgba(30, 40, 55, 175);
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 10px 6px;
        margin: 2px 2px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     Helvetica, Arial, sans-serif;
        font-size: 12px;
        min-height: 24px;
    }
    QTabBar::tab:selected {
        background: rgba(10, 163, 230, 38);
        color: #0AA3E6;
        border: 1px solid rgba(10, 163, 230, 80);
    }
    QTabBar::tab:!selected:hover {
        background: rgba(229, 229, 229, 150);
    }

    /* ---- Log text areas ---- */
    QTextEdit {
        background: transparent;
        border: none;
        color: rgba(30, 40, 55, 200);
        selection-background-color: rgba(10, 163, 230, 60);
        selection-color: rgba(0, 0, 0, 220);
    }

    /* ---- Scrollbars — verbatim from ui_main_theme.qss ---- */
    QScrollBar:vertical {
        border: none;
        background: transparent;
        width: 10px;
        margin: 10px 0px 10px 0px;
    }
    QScrollBar:horizontal {
        border: none;
        background: transparent;
        height: 10px;
        margin: 0px 10px 0px 10px;
    }
    QScrollBar::handle:vertical,
    QScrollBar::handle:horizontal {
        background: rgba(130, 130, 130, 100);
        border-radius: 5px;
    }
    QScrollBar::handle:vertical:hover,
    QScrollBar::handle:horizontal:hover {
        background: rgba(130, 130, 130, 180);
    }
    QScrollBar::handle:vertical   { min-height: 20px; }
    QScrollBar::handle:horizontal { min-width:  20px; }
    QScrollBar::add-line:vertical,  QScrollBar::sub-line:vertical,
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px; height: 0px; background: none;
    }
    QScrollBar::add-page:vertical,  QScrollBar::sub-page:vertical,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
        background: none;
    }
"""


# ---------------------------------------------------------------------------
# UILogger
# ---------------------------------------------------------------------------


class LoggerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui4 = UILogger()
        self.ui4.setupUi(self)

    def closeEvent(self, event):
        # Log.d(" Exit Real-Time Plot GUI")
        res = PopUp.question(
            self,
            Constants.app_title,
            "Are you sure you want to quit QATCH Q-1 application now?",
            True,
        )
        if res:
            # self.close()
            QtWidgets.QApplication.quit()
        else:
            event.ignore()


class UILogger:

    def setupUi(self, MainWindow4):
        MainWindow4.setMinimumSize(QtCore.QSize(1000, 100))
        MainWindow4.move(0, 0)

        self.centralwidget = QtWidgets.QWidget(MainWindow4)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)

        logTextBox = QTextEditLogger(self.centralwidget)

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
        MainWindow4.setWindowIcon(QtGui.QIcon(icon_path))
        MainWindow4.setWindowTitle(
            _translate(
                "MainWindow4",
                "{} {} - Console".format(Constants.app_title, Constants.app_version),
            )
        )


# ---------------------------------------------------------------------------
# QTextEditLogger
# ---------------------------------------------------------------------------


class QTextEditLogger(logging.Handler, QtCore.QObject):

    appendInfoText = QtCore.pyqtSignal(str)
    appendDebugText = QtCore.pyqtSignal(str)
    forceRepaintEvents = False
    progressMode = False

    def __init__(self, parent):
        super().__init__()
        QtCore.QObject.__init__(self)

        # ---- Tab widget --------------------------------------------------
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet(_LOGGER_GLASS_QSS)
        self.tabs.setTabPosition(QtWidgets.QTabWidget.East)
        self.tabs.setDocumentMode(True)  # removes the outer frame around the pane

        # ---- Info tab ----------------------------------------------------
        self.tab1 = QtWidgets.QWidget()
        self.tab1.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.logInfo = QtWidgets.QTextEdit(parent)
        self.logInfo.setReadOnly(True)
        self.logInfo.setFrameShape(QtWidgets.QFrame.NoFrame)

        layout_v1 = QtWidgets.QVBoxLayout(self.tab1)
        layout_v1.setContentsMargins(4, 4, 4, 4)
        layout_v1.setSpacing(0)
        layout_v1.addWidget(self.logInfo)
        self.tabs.addTab(self.tab1, "Info")

        # ---- Debug tab ---------------------------------------------------
        self.tab2 = QtWidgets.QWidget()
        self.tab2.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.logDebug = QtWidgets.QTextEdit(parent)
        self.logDebug.setReadOnly(True)
        self.logDebug.setFrameShape(QtWidgets.QFrame.NoFrame)

        layout_v2 = QtWidgets.QVBoxLayout(self.tab2)
        layout_v2.setContentsMargins(4, 4, 4, 4)
        layout_v2.setSpacing(0)
        layout_v2.addWidget(self.logDebug)
        self.tabs.addTab(self.tab2, "Debug")

        # ---- Root layout — small margin so glass pane border breathes ----
        layout_v = QtWidgets.QVBoxLayout()
        layout_v.setContentsMargins(4, 4, 4, 4)
        layout_v.setSpacing(0)
        layout_v.addWidget(self.tabs)
        parent.setLayout(layout_v)

        # ---- Signals -----------------------------------------------------
        self.appendInfoText.connect(self.appendToInfo)
        self.appendDebugText.connect(self.appendToDebug)
        self.last_record_msg = None

    # ------------------------------------------------------------------
    # Append helpers
    # ------------------------------------------------------------------

    def appendToInfo(self, html):
        if self.forceRepaintEvents and "[Device] ERROR:" in html:
            return  # suppress serial errors during firmware update
        self.logInfo.moveCursor(QtGui.QTextCursor.End, QtGui.QTextCursor.MoveAnchor)
        if self.progressMode:
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

    # ------------------------------------------------------------------
    # logging.Handler interface
    # ------------------------------------------------------------------

    def emit(self, record):
        msg = self.format(record)
        if msg == self.last_record_msg:
            print(msg, "(duplicate record ignored)")
            return
        self.last_record_msg = msg

        msg = msg[msg.index(" ") + 1 :]  # trim date from console

        html_fmt = (
            '<font style=\'font-family:"Lucida Console","Courier New",monospace;'
            "color:{};font-weight:{};'>{}</font><br/><br/>"
        )
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
