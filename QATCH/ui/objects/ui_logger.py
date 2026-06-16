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

import os

from PyQt5 import QtCore, QtGui, QtWidgets
from loguru import logger as _loguru

from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants
from QATCH.ui.popUp import PopUp
from QATCH.ui.components.animated_combo_box import AnimatedComboBox

_LOGGER_GLASS_QSS = """
    /* ---- Main Container ---- */
    QWidget#ConsoleContainer {
        background: rgba(255, 255, 255, 120);
        border: 1px solid rgba(255, 255, 255, 200);
        border-radius: 8px;
    }

    /* ---- Animated ComboBox & Search Bar ---- */
    QComboBox, QLineEdit#SearchBar {
        background: rgba(255, 255, 255, 100);
        border: 1px solid rgba(255, 255, 255, 150);
        border-radius: 14px; /* Perfect pill shape for 28px height */
        padding: 0px 14px;
        color: rgba(51, 51, 51, 255);
        font-weight: bold;
    }
    QComboBox:hover, QLineEdit#SearchBar:hover {
        background: rgba(255, 255, 255, 150);
    }
    QComboBox:pressed, QLineEdit#SearchBar:focus {
        background: rgba(255, 255, 255, 180);
        border: 1px solid rgba(10, 163, 230, 100); /* Blue tint on focus */
    }
    QComboBox::drop-down {
        border: none;
        width: 32px; 
    }
    
    /* Drop-down menu list styling */
    QComboBox QAbstractItemView {
        background-color: rgb(245, 247, 250); 
        border: 1px solid rgba(200, 200, 200, 180);
        border-radius: 8px;
        color: rgba(51, 51, 51, 255);
        selection-background-color: rgba(10, 163, 230, 40);
        selection-color: #0AA3E6;
        outline: none; 
    }

    /* ---- Icon-Only Buttons (Clear & Search Nav) ---- */
    QPushButton#ClearBtn, QPushButton#SearchPrevBtn, QPushButton#SearchNextBtn {
        background: transparent;
        border: none;
        border-radius: 4px;
        padding: 4px;
        color: rgba(51, 51, 51, 200);
        font-weight: bold;
    }
    QPushButton#ClearBtn:hover, QPushButton#SearchPrevBtn:hover, QPushButton#SearchNextBtn:hover {
        background: rgba(255, 255, 255, 120);
    }
    QPushButton#ClearBtn:pressed, QPushButton#SearchPrevBtn:pressed, QPushButton#SearchNextBtn:pressed {
        background: rgba(255, 255, 255, 180);
    }

    /* ---- Log text area ---- */
    QTextEdit {
        background: transparent;
        border: none;
        color: rgba(30, 40, 55, 200);
        selection-background-color: rgba(10, 163, 230, 60);
        selection-color: rgba(0, 0, 0, 220);
        padding: 2px 4px;
    }

    /* ---- Scrollbars ---- */
    QScrollBar:vertical {
        border: none;
        background: transparent;
        width: 8px;
        margin: 4px 0px 4px 0px;
    }
    QScrollBar:horizontal {
        border: none;
        background: transparent;
        height: 8px;
        margin: 0px 4px 0px 4px;
    }
    QScrollBar::handle:vertical,
    QScrollBar::handle:horizontal {
        background: rgba(130, 130, 130, 100);
        border-radius: 4px;
    }
    QScrollBar::handle:vertical:hover,
    QScrollBar::handle:horizontal:hover {
        background: rgba(130, 130, 130, 180);
    }
    QScrollBar::add-line:vertical,  QScrollBar::sub-line:vertical,
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:vertical,  QScrollBar::sub-page:vertical,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
        width: 0px; height: 0px; background: none;
    }
"""


class LoggerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui4 = UILogger()
        self.ui4.setupUi(self)

    def closeEvent(self, event):
        res = PopUp.question(
            self,
            Constants.app_title,
            "Are you sure you want to quit QATCH Q-1 application now?",
            True,
        )
        if res:
            QtWidgets.QApplication.quit()
        else:
            event.ignore()


class UILogger:
    def setupUi(self, MainWindow4):
        MainWindow4.setMinimumSize(QtCore.QSize(1000, 250))
        MainWindow4.move(0, 0)

        self.centralwidget = QtWidgets.QWidget(MainWindow4)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(4, 4, 4, 4)

        logTextBox = QTextEditLogger(self.centralwidget)

        _loguru.add(
            logTextBox,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
            enqueue=True,
            colorize=False,
        )

        Log._show_user_info()

        MainWindow4.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow4)
        QtCore.QMetaObject.connectSlotsByName(MainWindow4)

    def retranslateUi(self, MainWindow4):
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "qatch-icon.png")
        MainWindow4.setWindowIcon(QtGui.QIcon(icon_path))
        MainWindow4.setWindowTitle(
            _translate(
                "MainWindow4",
                "{} {} - Console".format(Constants.app_title, Constants.app_version),
            )
        )


class QTextEditLogger(QtCore.QObject):
    appendLogText = QtCore.pyqtSignal(str, int)

    def __init__(self, parent):
        super().__init__()

        # ---- Main Container & Layout ----
        self.container = QtWidgets.QWidget(parent)
        self.container.setObjectName("ConsoleContainer")
        self.container.setStyleSheet(_LOGGER_GLASS_QSS)

        main_layout = QtWidgets.QVBoxLayout(self.container)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ---- Control Bar ----
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setContentsMargins(4, 2, 4, 0)
        control_layout.setSpacing(8)

        # 1. Transparent Icon-Only Clear Button (Left)
        self.btn_clear = QtWidgets.QPushButton(parent=self.container)
        self.btn_clear.setObjectName("ClearBtn")
        self.btn_clear.setFixedSize(24, 24)
        self.btn_clear.setToolTip("Clear Console")
        self.btn_clear.setCursor(QtCore.Qt.PointingHandCursor)

        clear_icon_path = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "clear-console.svg"
        )
        self.btn_clear.setIcon(QtGui.QIcon(clear_icon_path))
        self.btn_clear.setIconSize(QtCore.QSize(14, 14))
        self.btn_clear.clicked.connect(self.clear_console)
        control_layout.addWidget(self.btn_clear)

        # 2. Filter Dropdown
        arrow_icon_path = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "down-chevron.png"
        )
        self.level_filter = AnimatedComboBox(icon_path=arrow_icon_path, parent=self.container)
        self.level_filter.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.level_filter.setCurrentText("DEBUG")
        self.level_filter.setFixedSize(120, 28)
        self.level_filter.currentTextChanged.connect(self.apply_filter)
        self.current_filter_level = 10
        control_layout.addWidget(self.level_filter)
        control_layout.addStretch()

        # Search Field
        self.search_input = QtWidgets.QLineEdit(parent=self.container)
        self.search_input.setObjectName("SearchBar")
        self.search_input.setPlaceholderText("Find in logs...")
        self.search_input.setFixedSize(180, 28)
        self.search_input.returnPressed.connect(self.find_next)  # Enter key triggers next
        control_layout.addWidget(self.search_input)

        # Search Navigation
        self.btn_find_prev = QtWidgets.QPushButton("↑", parent=self.container)
        self.btn_find_prev.setObjectName("SearchPrevBtn")
        self.btn_find_prev.setFixedSize(24, 24)
        self.btn_find_prev.setToolTip("Find Previous (Shift+Enter)")
        self.btn_find_prev.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_find_prev.clicked.connect(self.find_prev)
        control_layout.addWidget(self.btn_find_prev)

        # Search Navigation
        self.btn_find_next = QtWidgets.QPushButton("↓", parent=self.container)
        self.btn_find_next.setObjectName("SearchNextBtn")
        self.btn_find_next.setFixedSize(24, 24)
        self.btn_find_next.setToolTip("Find Next (Enter)")
        self.btn_find_next.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_find_next.clicked.connect(self.find_next)
        control_layout.addWidget(self.btn_find_next)

        main_layout.addLayout(control_layout)

        # ---- Unified Log Text Area ----
        self.logText = QtWidgets.QTextEdit()
        self.logText.setReadOnly(True)
        self.logText.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.logText.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        main_layout.addWidget(self.logText)

        parent_layout = QtWidgets.QVBoxLayout(parent)
        parent_layout.setContentsMargins(0, 0, 0, 0)
        parent_layout.addWidget(self.container)

        # ---- Global Shortcuts ----
        # Ctrl+F to focus search
        self.shortcut_find = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+F"), self.container)
        self.shortcut_find.activated.connect(self.focus_search)

        # Shift+Enter while in search box to find previous
        self.shortcut_find_prev = QtWidgets.QShortcut(
            QtGui.QKeySequence("Shift+Return"), self.search_input
        )
        self.shortcut_find_prev.activated.connect(self.find_prev)

        self.appendLogText.connect(self.appendToConsole)
        self.last_record_msg = None
        self.log_cache = []

    def focus_search(self):
        """Focus the search bar and select existing text for quick retyping."""
        self.search_input.setFocus()
        self.search_input.selectAll()

    def find_next(self):
        """Search forward. Wraps around to start if it hits the bottom."""
        search_term = self.search_input.text()
        if not search_term:
            return

        found = self.logText.find(search_term)
        if not found:
            self.logText.moveCursor(QtGui.QTextCursor.Start)
            self.logText.find(search_term)

    def find_prev(self):
        """Search backward. Wraps around to end if it hits the top."""
        search_term = self.search_input.text()
        if not search_term:
            return

        options = QtGui.QTextDocument.FindBackward
        found = self.logText.find(search_term, options)
        if not found:
            self.logText.moveCursor(QtGui.QTextCursor.End)
            self.logText.find(search_term, options)

    def clear_console(self):
        self.logText.clear()
        self.log_cache.clear()

    def apply_filter(self, level_text):
        levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        self.current_filter_level = levels.get(level_text, 10)

        self.logText.clear()
        for html, lvl in self.log_cache:
            if lvl >= self.current_filter_level:
                self.logText.insertHtml(html)

        self.logText.moveCursor(QtGui.QTextCursor.End)

    def appendToConsole(self, html, level_no):
        self.log_cache.append((html, level_no))

        if level_no >= self.current_filter_level:
            vsb = self.logText.verticalScrollBar()
            is_at_bottom = vsb.value() >= (vsb.maximum() - 5)

            self.logText.moveCursor(QtGui.QTextCursor.End)
            self.logText.insertHtml(html)

            if is_at_bottom:
                vsb.setValue(vsb.maximum())

    def write(self, message):
        if message == self.last_record_msg:
            return
        self.last_record_msg = message

        record = message.record
        level_no = record["level"].no
        level_name = record["level"].name

        time_str = record["time"].strftime("%Y-%m-%d %H:%M:%S")
        name_line = f"{record['name']}:{record['line']}"

        raw_msg = record["message"].replace("<", "&lt;").replace(">", "&gt;")

        if level_name == "DEBUG":
            lvl_color = "#78909C"
            weight = "normal"
        elif level_name == "INFO":
            lvl_color = "#2E7D32"
            weight = "normal"
        elif level_name == "WARNING":
            lvl_color = "#E65100"
            weight = "bold"
        elif level_name in ["ERROR", "CRITICAL"]:
            lvl_color = "#C62828"
            weight = "bold"
        else:
            lvl_color = "#333333"
            weight = "normal"

        time_html = f"<span style='color:#00838F;'>{time_str}</span>"
        padded_level = f"{level_name:<8}".replace(" ", "&nbsp;")
        lvl_html = f"<span style='color:{lvl_color}; font-weight:{weight};'>{padded_level}</span>"
        loc_html = f"<span style='color:#8E24AA;'>{name_line}</span>"
        msg_html = f"<span style='color:{lvl_color}; font-weight:{weight};'>{raw_msg}</span>"

        html_line = f"<span style='font-family: Consolas, \"Courier New\", monospace;'>{time_html} | {lvl_html} | {loc_html} | {msg_html}</span><br>"

        self.appendLogText.emit(html_line, level_no)
