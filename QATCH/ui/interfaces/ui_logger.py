"""
QATCH.ui.mainWindow_ui_logger

Log-console UI for the nanovisQ application with glass-morphism styling
that matches the modeMenuScrollArea palette from ui_main_theme.qss.

Author(s)
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-06-17
"""

import os

from PyQt5 import QtCore, QtGui, QtWidgets
from loguru import logger as _loguru

from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants
from QATCH.ui.popUp import PopUp
from QATCH.ui.components import AnimatedComboBox

_LOGGER_GLASS_QSS = """
    /* ---- Main Container ---- */
    QWidget#ConsoleContainer {
        background: rgba(255, 255, 255, 120);
        border: 1px solid rgba(255, 255, 255, 200);
        border-radius: 8px;
    }

    /* ---- Animated ComboBox ----
       Scoped to AnimatedComboBox via #LevelFilter so the native arrow
       region and the custom QLabel arrow don't fight over the same zone. */
    QComboBox#LevelFilter {
        background: rgba(255, 255, 255, 160);
        border: 1px solid rgba(120, 130, 145, 160);   /* clear, visible outline */
        border-radius: 14px;          /* pill for 28px height */
        padding-left: 14px;
        padding-right: 30px;          /* reserve room for the spinning arrow */
        color: rgb(51, 51, 51);
        font-weight: bold;
    }
    QComboBox#LevelFilter:hover {
        background: rgba(255, 255, 255, 200);
        border: 1px solid rgba(90, 100, 115, 200);
    }
    QComboBox#LevelFilter:on,
    QComboBox#LevelFilter:focus {
        background: rgba(255, 255, 255, 220);
        border: 1px solid rgba(10, 163, 230, 200);
    }
    /* The custom QLabel arrow lives here, so kill the native drop-down box
       entirely instead of letting it draw a second arrow underneath. */
    QComboBox#LevelFilter::drop-down {
        border: none;
        background: transparent;
        width: 0px;
    }
    QComboBox#LevelFilter::down-arrow {
        image: none;
        width: 0px;
        height: 0px;
    }

    /* Drop-down menu list styling */
    QComboBox#LevelFilter QAbstractItemView {
        background-color: rgb(245, 247, 250);
        border: 1px solid rgba(200, 200, 200, 180);
        border-radius: 8px;
        color: rgb(51, 51, 51);
        padding: 4px;
        selection-background-color: rgba(10, 163, 230, 40);
        selection-color: #0AA3E6;
        outline: none;
    }
    QComboBox#LevelFilter QAbstractItemView::item {
        min-height: 24px;
        border-radius: 6px;
        padding-left: 8px;
    }

    /* ---- Search Bar ---- */
    QLineEdit#SearchBar {
        background: rgba(255, 255, 255, 160);
        border: 1px solid rgba(120, 130, 145, 160);   /* clear, visible outline */
        border-radius: 14px;
        padding: 0px 8px;
        color: rgb(51, 51, 51);
        font-weight: bold;
    }
    QLineEdit#SearchBar:hover {
        background: rgba(255, 255, 255, 200);
        border: 1px solid rgba(90, 100, 115, 200);
    }
    QLineEdit#SearchBar:focus {
        background: rgba(255, 255, 255, 220);
        border: 1px solid rgba(10, 163, 230, 200);
    }

    /* ---- Toolbar / log separator ---- */
    QFrame#ToolbarSeparator {
        border: none;
        background: rgba(120, 130, 145, 110);
        max-height: 1px;
        min-height: 1px;
    }

    /* ---- Match counter label ---- */
    QLabel#MatchCounter {
        color: rgba(80, 90, 105, 220);
        font-family: Consolas, "Courier New", monospace;
        font-size: 11px;
        padding: 0px 2px;
    }
    QLabel#MatchCounter[state="nomatch"] {
        color: #C62828;
    }

    /* ---- Clear (text-only, no outline) ---- */
    QPushButton#ClearBtn {
        background: transparent;
        border: none;
        padding: 0px 12px;
        color: rgba(51, 51, 51, 230);
        font-weight: bold;
    }
    QPushButton#ClearBtn:hover {
        color: rgba(51, 51, 51, 150);   /* slightly lighter on hover */
    }
    QPushButton#ClearBtn:pressed {
        color: rgba(51, 51, 51, 110);
    }

    /* ---- Icon-Only Buttons (Search Nav) ---- */
    QPushButton#SearchPrevBtn, QPushButton#SearchNextBtn {
        background: transparent;
        border: none;
        border-radius: 14px;
    }
    QPushButton#SearchPrevBtn:hover, QPushButton#SearchNextBtn:hover {
        background: rgba(255, 255, 255, 180);
    }
    QPushButton#SearchPrevBtn:pressed, QPushButton#SearchNextBtn:pressed {
        background: rgba(255, 255, 255, 220);
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
            enqueue=False,  # Qt queued signal handles thread-safety; no loguru queue
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

    # Highlight color for the currently-selected search match.
    _MATCH_FMT_COLOR = QtGui.QColor(10, 163, 230, 90)

    # UI batching / memory bounds.
    _FLUSH_INTERVAL_MS = 150  # how often pending records are rendered
    _MAX_BLOCKS = 5000  # max QTextEdit blocks retained on screen
    _MAX_CACHE = 20000  # max records retained for filtering/search

    def __init__(self, parent):
        super().__init__()

        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")

        # Main Container & Layout
        self.container = QtWidgets.QWidget(parent)
        self.container.setObjectName("ConsoleContainer")
        self.container.setStyleSheet(_LOGGER_GLASS_QSS)

        main_layout = QtWidgets.QVBoxLayout(self.container)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Control Bar
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setContentsMargins(2, 0, 2, 0)
        control_layout.setSpacing(8)

        # Filter Dropdown
        arrow_icon_path = os.path.join(icons_dir, "down-chevron.svg")
        self.level_filter = AnimatedComboBox(icon_path=arrow_icon_path, parent=self.container)
        self.level_filter.setObjectName("LevelFilter")
        self.level_filter.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.level_filter.setFixedSize(120, 28)
        self.level_filter.currentTextChanged.connect(self.apply_filter)
        control_layout.addWidget(self.level_filter)

        control_layout.addStretch()

        # Search field with leading search icon + trailing clear icon
        self.search_input = QtWidgets.QLineEdit(parent=self.container)
        self.search_input.setObjectName("SearchBar")
        self.search_input.setPlaceholderText("Find in logs...")
        self.search_input.setFixedSize(220, 28)
        self.search_input.returnPressed.connect(self.find_next)
        self.search_input.textChanged.connect(self.on_search_changed)

        # Leading search glyph (decorative, left side of the field)
        search_icon = QtGui.QIcon(os.path.join(icons_dir, "search.svg"))
        self.search_input.addAction(search_icon, QtWidgets.QLineEdit.LeadingPosition)

        # Trailing clear-text action
        self._clear_icon_normal = QtGui.QIcon(os.path.join(icons_dir, "clear.svg"))
        self._clear_icon_hover = self._make_lighter_icon(
            os.path.join(icons_dir, "clear.svg"), opacity=0.55
        )
        self.clear_text_action = self.search_input.addAction(
            self._clear_icon_normal,
            QtWidgets.QLineEdit.TrailingPosition,
        )
        self.clear_text_action.setToolTip("Clear search text")
        self.clear_text_action.triggered.connect(self.clear_search_text)
        self.clear_text_action.setVisible(False)  # only show when there's text

        # Hover handling
        self._clear_hovering = False
        self.search_input.setMouseTracking(True)
        self.search_input.installEventFilter(self)
        control_layout.addWidget(self.search_input)

        # Match counter ("3 / 12")
        self.match_counter = QtWidgets.QLabel("No results", parent=self.container)
        self.match_counter.setObjectName("MatchCounter")
        self.match_counter.setMinimumWidth(64)
        self.match_counter.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(self.match_counter)

        # Search navigation
        up_chevron = QtGui.QIcon(os.path.join(icons_dir, "up-chevron.svg"))
        down_chevron = QtGui.QIcon(os.path.join(icons_dir, "down-chevron.svg"))

        self.btn_find_prev = QtWidgets.QPushButton(parent=self.container)
        self.btn_find_prev.setObjectName("SearchPrevBtn")
        self.btn_find_prev.setFixedSize(28, 28)
        self.btn_find_prev.setIcon(up_chevron)
        self.btn_find_prev.setIconSize(QtCore.QSize(12, 12))
        self.btn_find_prev.setToolTip("Find Previous (Shift+Enter)")
        self.btn_find_prev.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_find_prev.clicked.connect(self.find_prev)
        control_layout.addWidget(self.btn_find_prev)

        self.btn_find_next = QtWidgets.QPushButton(parent=self.container)
        self.btn_find_next.setObjectName("SearchNextBtn")
        self.btn_find_next.setFixedSize(28, 28)
        self.btn_find_next.setIcon(down_chevron)
        self.btn_find_next.setIconSize(QtCore.QSize(12, 12))
        self.btn_find_next.setToolTip("Find Next (Enter)")
        self.btn_find_next.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_find_next.clicked.connect(self.find_next)
        control_layout.addWidget(self.btn_find_next)

        sep = QtWidgets.QFrame(self.container)
        sep.setFrameShape(QtWidgets.QFrame.VLine)
        sep.setStyleSheet("color: rgba(150, 150, 150, 90);")
        sep.setFixedHeight(20)
        control_layout.addWidget(sep)

        self.btn_clear = QtWidgets.QPushButton("Clear", parent=self.container)
        self.btn_clear.setObjectName("ClearBtn")
        self.btn_clear.setFixedHeight(28)
        self.btn_clear.setToolTip("Clear all console output")
        self.btn_clear.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_clear.clicked.connect(self.clear_console)
        control_layout.addWidget(self.btn_clear)

        main_layout.addLayout(control_layout)

        # Toolbar
        self.toolbar_separator = QtWidgets.QFrame(self.container)
        self.toolbar_separator.setObjectName("ToolbarSeparator")
        self.toolbar_separator.setFrameShape(QtWidgets.QFrame.HLine)
        self.toolbar_separator.setFixedHeight(1)
        main_layout.addWidget(self.toolbar_separator)

        #  Unified Log Text Area
        self.logText = QtWidgets.QTextEdit()
        self.logText.setReadOnly(True)
        self.logText.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.logText.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        # Cap the document so insertHtml/relayout cost stays bounded during
        # long, log-heavy runs (e.g. Analyze). Oldest blocks drop off the top.
        self.logText.document().setMaximumBlockCount(self._MAX_BLOCKS)
        main_layout.addWidget(self.logText)

        parent_layout = QtWidgets.QVBoxLayout(parent)
        parent_layout.setContentsMargins(0, 0, 0, 0)
        parent_layout.addWidget(self.container)

        self.shortcut_find = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+F"), self.container)
        self.shortcut_find.activated.connect(self.focus_search)

        self.shortcut_find_prev = QtWidgets.QShortcut(
            QtGui.QKeySequence("Shift+Return"), self.search_input
        )
        self.shortcut_find_prev.activated.connect(self.find_prev)

        self.appendLogText.connect(self.appendToConsole)
        self.last_record_msg = None
        self.log_cache = []

        # Batched UI updates: log records accumulate here and are flushed to
        # the QTextEdit on a timer rather than one insertHtml per record. This
        # keeps the GUI thread responsive when logging floods (e.g. Analyze).
        self._pending = []
        self._flush_timer = QtCore.QTimer(self)
        self._flush_timer.setInterval(self._FLUSH_INTERVAL_MS)
        self._flush_timer.timeout.connect(self._flush_pending)
        self._flush_timer.start()

        self._update_match_ui(0, 0)

        # Set the default filter now that every attribute apply_filter touches
        # (self._pending, self.logText, self.log_cache, self.search_input)
        # actually exists - setting this earlier fired currentTextChanged
        # synchronously during construction, before those existed, and
        # crashed with "QTextEditLogger object has no attribute '_pending'".
        self.level_filter.setCurrentText("INFO")  # calls apply_filter

    def focus_search(self):
        self.search_input.setFocus()
        self.search_input.selectAll()

    def clear_search_text(self):
        self.search_input.clear()
        self.search_input.setFocus()

    @staticmethod
    def _make_lighter_icon(svg_path, opacity=0.55):
        src = QtGui.QPixmap(svg_path)
        if src.isNull():
            return QtGui.QIcon(svg_path)
        faded = QtGui.QPixmap(src.size())
        faded.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(faded)
        painter.setOpacity(opacity)
        painter.drawPixmap(0, 0, src)
        painter.end()
        return QtGui.QIcon(faded)

    def _clear_icon_rect(self):
        if not self.clear_text_action.isVisible():
            return QtCore.QRect()
        w = self.search_input.width()
        h = self.search_input.height()
        side = h  # trailing action occupies a roughly square zone on the right
        return QtCore.QRect(w - side - 2, 0, side, h)

    def eventFilter(self, obj, event):
        if obj is self.search_input:
            et = event.type()
            if et == QtCore.QEvent.MouseMove:
                over = self._clear_icon_rect().contains(event.pos())
                if over != self._clear_hovering:
                    self._clear_hovering = over
                    self.clear_text_action.setIcon(
                        self._clear_icon_hover if over else self._clear_icon_normal
                    )
                    self.search_input.setCursor(
                        QtCore.Qt.CursorShape.PointingHandCursor if over else QtCore.Qt.IBeamCursor
                    )
            elif et == QtCore.QEvent.Leave:
                if self._clear_hovering:
                    self._clear_hovering = False
                    self.clear_text_action.setIcon(self._clear_icon_normal)
                    self.search_input.setCursor(QtCore.Qt.IBeamCursor)
        return super().eventFilter(obj, event)

    def _count_matches(self, term):
        if not term:
            return 0
        doc = self.logText.document()
        count = 0
        cursor = QtGui.QTextCursor(doc)
        while True:
            cursor = doc.find(term, cursor)
            if cursor.isNull():
                break
            count += 1
        return count

    def _current_match_index(self, term):
        if not term:
            return 0
        doc = self.logText.document()
        sel = self.logText.textCursor()
        if not sel.hasSelection():
            return 0
        sel_end = sel.selectionEnd()
        idx = 0
        cursor = QtGui.QTextCursor(doc)
        while True:
            cursor = doc.find(term, cursor)
            if cursor.isNull():
                break
            idx += 1
            if cursor.selectionEnd() == sel_end:
                return idx
        return 0

    def _update_match_ui(self, current, total):
        term = self.search_input.text()
        if not term:
            self.match_counter.setText("No results")
            self.match_counter.setProperty("state", "")
        elif total == 0:
            self.match_counter.setText("No results")
            self.match_counter.setProperty("state", "nomatch")
        else:
            shown = current if current else 1
            self.match_counter.setText(f"{shown} / {total}")
            self.match_counter.setProperty("state", "")

        self.match_counter.style().unpolish(self.match_counter)
        self.match_counter.style().polish(self.match_counter)

        has_matches = bool(term) and total > 0
        self.btn_find_next.setEnabled(has_matches)
        self.btn_find_prev.setEnabled(has_matches)
        self.clear_text_action.setVisible(bool(term))

    def on_search_changed(self, text):
        total = self._count_matches(text)
        self._update_match_ui(0, total)

    def find_next(self):
        term = self.search_input.text()
        if not term:
            self._update_match_ui(0, 0)
            return

        found = self.logText.find(term)
        if not found:
            self.logText.moveCursor(QtGui.QTextCursor.Start)
            found = self.logText.find(term)

        total = self._count_matches(term)
        current = self._current_match_index(term) if found else 0
        self._update_match_ui(current, total)

    def find_prev(self):
        term = self.search_input.text()
        if not term:
            self._update_match_ui(0, 0)
            return

        options = QtGui.QTextDocument.FindBackward
        found = self.logText.find(term, options)
        if not found:
            self.logText.moveCursor(QtGui.QTextCursor.End)
            found = self.logText.find(term, options)

        total = self._count_matches(term)
        current = self._current_match_index(term) if found else 0
        self._update_match_ui(current, total)

    def clear_console(self):
        self._pending.clear()
        self.logText.clear()
        self.log_cache.clear()
        self._update_match_ui(0, 0)

    def apply_filter(self, level_text):
        levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        self.current_filter_level = levels.get(level_text, 10)
        self._flush_pending()

        self.logText.clear()
        for html, lvl in self.log_cache:
            if lvl >= self.current_filter_level:
                self.logText.insertHtml(html)

        self.logText.moveCursor(QtGui.QTextCursor.End)
        self.on_search_changed(self.search_input.text())

    def appendToConsole(self, html, level_no):
        self._pending.append((html, level_no))

    def _flush_pending(self):
        """Render all buffered records in a single insertHtml pass."""
        if not self._pending:
            return

        batch = self._pending
        self._pending = []

        # Append to the searchable/filterable cache, bounded to _MAX_CACHE.
        self.log_cache.extend(batch)
        if len(self.log_cache) > self._MAX_CACHE:
            del self.log_cache[: len(self.log_cache) - self._MAX_CACHE]

        visible_html = "".join(html for html, lvl in batch if lvl >= self.current_filter_level)
        if not visible_html:
            return

        vsb = self.logText.verticalScrollBar()
        is_at_bottom = vsb.value() >= (vsb.maximum() - 5)

        self.logText.moveCursor(QtGui.QTextCursor.End)
        self.logText.insertHtml(visible_html)

        if is_at_bottom:
            vsb.setValue(vsb.maximum())

        # Keep the live match total honest as new lines stream in.
        term = self.search_input.text()
        if term:
            self._update_match_ui(self._current_match_index(term), self._count_matches(term))

    def write(self, message):
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
