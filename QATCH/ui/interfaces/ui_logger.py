"""QATCH.ui.interfaces.ui_logger.py

Log-console UI module for the nanovisQ application.

Author(s):
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-07-01
"""

import os
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from loguru import logger as _loguru
from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants
from QATCH.ui.components import AnimatedComboBox
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css

if TYPE_CHECKING:
    from QATCH.ui.windows import LoggerWindow


class UILogger:
    """Sets up the user interface for the logging console window."""

    def setup_ui(self, logger_window: "LoggerWindow") -> None:
        """Initializes and structures the UI components of the logger window.

        Args:
            logger_window (LoggerWindow): The main window instance where
                the logger components will be rendered.
        """
        logger_window.setMinimumSize(QtCore.QSize(1000, 250))
        logger_window.move(0, 0)

        self.centralwidget = QtWidgets.QWidget(logger_window)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(4, 4, 4, 4)

        log_text_box = QTextEditLogger(self.centralwidget)

        # Qt queued signal handles thread-safety; no loguru queue needed here
        _loguru.add(
            log_text_box,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
            enqueue=False,
            colorize=False,
        )

        Log._show_user_info()

        logger_window.setCentralWidget(self.centralwidget)
        self.retranslateUi(logger_window)
        QtCore.QMetaObject.connectSlotsByName(logger_window)

    def retranslateUi(self, logger_window: QtWidgets.QMainWindow) -> None:
        """Translates the UI components and sets window metadata.

        Args:
            logger_window (QtWidgets.QMainWindow): The target window.
        """
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "qatch-icon.png")
        logger_window.setWindowIcon(QtGui.QIcon(icon_path))
        logger_window.setWindowTitle(
            _translate(
                "loggerWindow",
                f"{Constants.app_title} {Constants.app_version} - Console",
            )
        )


class QTextEditLogger(QtCore.QObject):
    """A custom Qt object that acts as a Loguru sink and a UI text display.

    Intercepts log records, caches them, and handles batched rendering into a
    QTextEdit widget to ensure thread safety and GUI responsiveness.

    Attributes:
        appendLogText (QtCore.pyqtSignal): Signal emitted when a new log arrives.
    """

    # Raw record fields: (time_str, level_name, level_no, name_line, raw_msg)
    appendLogText = QtCore.pyqtSignal(str, str, int, str, str)

    # UI batching / memory bounds
    _FLUSH_INTERVAL_MS = 150
    _MAX_BLOCKS = 5000
    _MAX_CACHE = 20000

    # Log level constants
    _LOG_LEVELS: Tuple[str, ...] = (
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
    )

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        """Initialize the log viewer widget.

        Creates the user interface, configures search and filtering support,
        initializes log caching structures, and connects theme and timer
        handlers required for log display updates.

        Args:
            parent (QtWidgets.QWidget): Parent Qt widget that owns this log viewer.
        """
        super().__init__()

        self._icons_dir: str = os.path.join(
            Architecture.get_path(),
            "QATCH",
            "icons",
        )

        self._clear_hovering: bool = False

        # Default log level filter (DEBUG = 10).
        self.current_filter_level: int = 10

        # Pending log entries waiting to be processed.
        self._pending: List[Tuple[str, str, int, str, str]] = []

        # Cached log entries used for filtering and search operations.
        self.log_cache: List[Tuple[str, str, int, str, str]] = []

        self._setup_ui(parent)
        self._setup_signals_and_timers()

        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

        # Initialize search result indicators.
        self._update_match_ui(0, 0)

        # Apply the default log level filter.
        self.level_filter.setCurrentText("INFO")

    def _setup_ui(self, parent: QtWidgets.QWidget) -> None:
        """Create and arrange the primary user interface widgets.

        Constructs the main container, toolbar area, separator, and log
        display region, then attaches the completed layout hierarchy to the
        specified parent widget.

        Args:
            parent (QtWidgets.QWidget): Parent widget that will host the log viewer interface.
        """
        self.container: QtWidgets.QWidget = QtWidgets.QWidget(parent)
        self.container.setObjectName("ConsoleContainer")

        main_layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout(self.container)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Control bar containing search, filtering, and action widgets.
        main_layout.addLayout(self._setup_control_bar())

        # Visual separator between the toolbar and log display.
        self.toolbar_separator: QtWidgets.QFrame = QtWidgets.QFrame(self.container)
        self.toolbar_separator.setObjectName("ToolbarSeparator")
        self.toolbar_separator.setFrameShape(QtWidgets.QFrame.HLine)
        self.toolbar_separator.setFixedHeight(1)

        main_layout.addWidget(self.toolbar_separator)

        # Main log output area.
        main_layout.addWidget(self._setup_text_area())

        parent_layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout(parent)
        parent_layout.setContentsMargins(0, 0, 0, 0)
        parent_layout.addWidget(self.container)

    def _setup_control_bar(self) -> QtWidgets.QHBoxLayout:
        """Create the control bar used to manage log viewing and search.

        Builds the toolbar containing log level filtering, search controls,
        match navigation buttons, and console management actions.

        Returns:
            QtWidgets.QHBoxLayout: The fully configured horizontal layout containing all control
            bar widgets.
        """
        control_layout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        control_layout.setContentsMargins(2, 0, 2, 0)
        control_layout.setSpacing(8)

        # Log level filter.
        arrow_icon_path: str = os.path.join(
            self._icons_dir,
            "down-chevron.svg",
        )

        self.level_filter: AnimatedComboBox = AnimatedComboBox(
            icon_path=arrow_icon_path,
            parent=self.container,
        )
        self.level_filter.setObjectName("LevelFilter")
        self.level_filter.addItems(self._LOG_LEVELS)
        self.level_filter.setFixedSize(120, 28)
        self.level_filter.currentTextChanged.connect(self.apply_filter)

        control_layout.addWidget(self.level_filter)
        control_layout.addStretch()

        # Search controls.
        self._setup_search_bar(control_layout)

        # Search match counter.
        self.match_counter: QtWidgets.QLabel = QtWidgets.QLabel(
            "No results",
            parent=self.container,
        )
        self.match_counter.setObjectName("MatchCounter")
        self.match_counter.setMinimumWidth(64)
        self.match_counter.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        control_layout.addWidget(self.match_counter)

        # Search navigation buttons.
        up_chevron: QtGui.QIcon = QtGui.QIcon(os.path.join(self._icons_dir, "up-chevron.svg"))
        down_chevron: QtGui.QIcon = QtGui.QIcon(os.path.join(self._icons_dir, "down-chevron.svg"))

        self.btn_find_prev: QtWidgets.QPushButton = self._create_nav_button(
            "SearchPrevBtn",
            up_chevron,
            "Find Previous (Shift+Enter)",
            self.find_prev,
        )

        self.btn_find_next: QtWidgets.QPushButton = self._create_nav_button(
            "SearchNextBtn",
            down_chevron,
            "Find Next (Enter)",
            self.find_next,
        )

        control_layout.addWidget(self.btn_find_prev)
        control_layout.addWidget(self.btn_find_next)

        # Separator between search controls and console actions.
        sep: QtWidgets.QFrame = QtWidgets.QFrame(self.container)
        sep.setObjectName("ToolbarVLineSep")
        sep.setFrameShape(QtWidgets.QFrame.VLine)
        sep.setFixedHeight(20)

        control_layout.addWidget(sep)

        # Console clear button.
        self.btn_clear: QtWidgets.QPushButton = QtWidgets.QPushButton(
            "Clear",
            parent=self.container,
        )
        self.btn_clear.setObjectName("ClearBtn")
        self.btn_clear.setFixedHeight(28)
        self.btn_clear.setToolTip("Clear all console output")
        self.btn_clear.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_clear.clicked.connect(self.clear_console)

        control_layout.addWidget(self.btn_clear)

        return control_layout

    def _setup_search_bar(self, layout: QtWidgets.QHBoxLayout) -> None:
        """Create and configure the log search input field.

        Builds the search box, attaches search and clear actions, connects
        search-related signals, and installs mouse tracking to support
        hover effects for the custom clear icon.

        Args:
            layout (QtWidgets.QHBoxLayout): Layout that will receive the configured search widget.
        """
        self.search_input: QtWidgets.QLineEdit = QtWidgets.QLineEdit(parent=self.container)
        self.search_input.setObjectName("SearchBar")
        self.search_input.setPlaceholderText("Find in logs...")
        self.search_input.setFixedSize(220, 28)

        # Pressing Enter advances to the next search match.
        self.search_input.returnPressed.connect(self.find_next)

        # Update search results as the query changes.
        self.search_input.textChanged.connect(self.on_search_changed)
        search_icon: QtGui.QIcon = QtGui.QIcon(os.path.join(self._icons_dir, "search.svg"))
        self.search_input.addAction(
            search_icon,
            QtWidgets.QLineEdit.LeadingPosition,
        )

        clear_svg: str = os.path.join(
            self._icons_dir,
            "clear.svg",
        )

        self._clear_icon_normal: QtGui.QIcon = QtGui.QIcon(clear_svg)
        self._clear_icon_hover: QtGui.QIcon = self._make_lighter_icon(
            clear_svg,
            opacity=0.55,
        )

        action = self.search_input.addAction(
            self._clear_icon_normal,
            QtWidgets.QLineEdit.TrailingPosition,
        )

        assert action is not None
        self.clear_text_action: QtWidgets.QAction = action
        self.clear_text_action.setToolTip("Clear search text")
        self.clear_text_action.triggered.connect(self.clear_search_text)
        self.clear_text_action.setVisible(False)
        self.search_input.setMouseTracking(True)
        self.search_input.installEventFilter(self)

        layout.addWidget(self.search_input)

    def _create_nav_button(
        self,
        name: str,
        icon: QtGui.QIcon,
        tooltip: str,
        callback: Any,
    ) -> QtWidgets.QPushButton:
        """Create a standardized navigation button for search controls.

        Builds a small icon-based QPushButton configured for use in search
        navigation (e.g., next/previous match controls). The button is
        pre-styled with consistent sizing, tooltip, and cursor behavior.

        Args:
            name: Qt object name used for styling and identification.
            icon: Icon displayed inside the button.
            tooltip: Tooltip text shown on hover.
            callback: Function or slot connected to the button's clicked
                signal.

        Returns:
            A fully configured QPushButton instance ready for use in the UI.
        """
        btn: QtWidgets.QPushButton = QtWidgets.QPushButton(parent=self.container)
        btn.setObjectName(name)
        btn.setFixedSize(28, 28)
        btn.setIcon(icon)
        btn.setIconSize(QtCore.QSize(12, 12))
        btn.setToolTip(tooltip)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(callback)
        return btn

    def _setup_text_area(self) -> QtWidgets.QTextEdit:
        """Create and configure the main log display text area.

        Builds the read-only QTextEdit used to render log output. The
        widget is configured for performance-constrained log viewing,
        including bounded document size and word-wrapped layout.

        Returns:
            A configured QTextEdit instance used as the primary log output
            display.
        """
        self.logText: QtWidgets.QTextEdit = QtWidgets.QTextEdit()
        self.logText.setReadOnly(True)
        self.logText.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.logText.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)

        doc = self.logText.document()
        assert doc is not None
        doc.setMaximumBlockCount(self._MAX_BLOCKS)

        return self.logText

    def _setup_signals_and_timers(self) -> None:
        """Initialize keyboard shortcuts, signal connections, and timers.

        Configures global and context-specific keyboard shortcuts for log
        navigation and search, connects internal Qt signals for log
        updating, and starts a periodic timer used to batch-flush pending
        log entries into the UI.
        """
        self.shortcut_find: QtWidgets.QShortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence("Ctrl+F"),
            self.container,
        )
        self.shortcut_find.activated.connect(self.focus_search)

        self.shortcut_find_prev: QtWidgets.QShortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence("Shift+Return"),
            self.search_input,
        )
        self.shortcut_find_prev.activated.connect(self.find_prev)

        self.appendLogText.connect(self.appendToConsole)

        self._flush_timer: QtCore.QTimer = QtCore.QTimer(self)
        self._flush_timer.setInterval(self._FLUSH_INTERVAL_MS)
        self._flush_timer.timeout.connect(self._flush_pending)
        self._flush_timer.start()

    def focus_search(self) -> None:
        """Focus the search bar and select all text.

        Moves keyboard focus to the search input field and highlights all
        existing text to allow immediate replacement.
        """
        self.search_input.setFocus()
        self.search_input.selectAll()

    def clear_search_text(self) -> None:
        """Clear the search input field and restore focus.

        Removes all text from the search bar and returns keyboard focus to
        the input field for continued typing.
        """
        self.search_input.clear()
        self.search_input.setFocus()

    @staticmethod
    def _make_lighter_icon(svg_path: str, opacity: float = 0.55) -> QtGui.QIcon:
        """Create a faded (lower-opacity) version of an icon.

        Loads an SVG (or image) from disk and renders a new pixmap with
        reduced opacity, producing a "disabled/hover/faded" visual state.

        Args:
            svg_path: File path to the source SVG or image asset.
            opacity: Opacity level applied to the rendered icon. Must be
                between 0.0 (fully transparent) and 1.0 (fully opaque).

        Returns:
            A QIcon containing the faded rendering. If the source cannot be
            loaded, returns a fallback QIcon created directly from the path.
        """
        src = QtGui.QPixmap(svg_path)
        if src.isNull():
            return QtGui.QIcon(svg_path)

        faded = QtGui.QPixmap(src.size())
        faded.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(faded)
        painter.setOpacity(opacity)
        painter.drawPixmap(0, 0, src)
        painter.end()

        return QtGui.QIcon(faded)

    def _clear_icon_rect(self) -> QtCore.QRect:
        """Calculate the clickable rectangle for the search clear icon.

        Determines the bounding box of the trailing clear-action icon inside
        the search input field. Used for hit-testing mouse events when
        manually handling interaction with the embedded QAction icon.

        Returns:
            QRect representing the clickable area of the clear icon.
            Returns an empty QRect if the clear action is not visible.
        """
        if not self.clear_text_action.isVisible():
            return QtCore.QRect()

        w: int = self.search_input.width()
        h: int = self.search_input.height()

        side: int = h
        return QtCore.QRect(w - side - 2, 0, side, h)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Handle hover interaction for the embedded clear-action icon.

        Tracks mouse movement and leave events on the search input field to
        provide custom hover behavior for the trailing clear icon. This
        includes swapping icons and changing the cursor when the mouse is
        over the clickable region.

        Args:
            obj: QObject receiving the event (expected to be search_input).
            event: Qt event being processed.

        Returns:
            True if the event is handled; otherwise False to continue normal
            processing.
        """
        if obj is self.search_input:
            et = event.type()

            if et == QtCore.QEvent.Type.MouseMove:
                over = self._clear_icon_rect().contains(event.pos())  # type: ignore

                if over != self._clear_hovering:
                    self._clear_hovering = over

                    assert self.clear_text_action is not None

                    self.clear_text_action.setIcon(
                        self._clear_icon_hover if over else self._clear_icon_normal
                    )

                    self.search_input.setCursor(
                        (
                            QtCore.Qt.CursorShape.PointingHandCursor
                            if over
                            else QtCore.Qt.CursorShape.IBeamCursor
                        ),
                    )

            elif et == QtCore.QEvent.Type.Leave:
                if self._clear_hovering:
                    self._clear_hovering = False

                    assert self.clear_text_action is not None

                    self.clear_text_action.setIcon(self._clear_icon_normal)
                    self.search_input.setCursor(QtCore.Qt.CursorShape.IBeamCursor)

        return super().eventFilter(obj, event)

    def _count_matches(self, term: str) -> int:
        """Counts occurrences of a search term in the log text."""
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

    def _count_matches(self, term: str) -> int:
        """Count occurrences of a search term in the log text.

        Performs a sequential document search using QTextDocument.find()
        to determine how many matches of the given term exist in the
        current log output.

        Args:
            term: Search string to locate within the log text.

        Returns:
            Number of matches found. Returns 0 if the search term is empty
            or no matches exist.
        """
        if not term:
            return 0

        doc = self.logText.document()
        assert doc is not None

        count = 0
        cursor = QtGui.QTextCursor(doc)

        while True:
            cursor = doc.find(term, cursor)
            if cursor.isNull():
                break
            count += 1

        return count

    def _update_match_ui(self, current: int, total: int) -> None:
        """Update the search match counter and related UI state.

        Updates the match counter label (e.g., "3 / 12"), applies visual
        state properties for styling, and enables/disables navigation
        controls based on whether search results exist.

        Args:
            current: Index of the currently selected match (1-based or 0 if
                not applicable).
            total: Total number of matches found for the current search term.
        """
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
        style = self.match_counter.style()

        if style is not None:
            style.unpolish(self.match_counter)
            style.polish(self.match_counter)

        has_matches = bool(term) and total > 0

        self.btn_find_next.setEnabled(has_matches)
        self.btn_find_prev.setEnabled(has_matches)

        assert self.clear_text_action is not None
        self.clear_text_action.setVisible(bool(term))

    def on_search_changed(self, text: str) -> None:
        """Handle updates to the search input text.

        Recomputes the number of matches whenever the search query changes
        and resets the match UI state.
        """
        total = self._count_matches(text)
        self._update_match_ui(0, total)

    def _current_match_index(self, term: str) -> int:
        """Find the index of the currently highlighted search term.

        Iterates through the document to find all matches for the search
        term and compares their positions to the current text selection
        to determine the active match's 1-based index.

        Args:
            term: The active search string.

        Returns:
            The 1-based index of the current match, or 0 if no match
            is actively selected or the term is empty.
        """
        if not term:
            return 0

        doc = self.logText.document()
        assert doc is not None

        # Get the current visible selection in the text editor
        active_cursor = self.logText.textCursor()
        if not active_cursor.hasSelection():
            return 0

        # The position where the currently highlighted word ends
        active_selection_end = active_cursor.selectionEnd()

        idx = 0
        search_cursor = QtGui.QTextCursor(doc)

        while True:
            # Find the next occurrence in the document
            search_cursor = doc.find(term, search_cursor)
            if search_cursor.isNull():
                break

            idx += 1
            if search_cursor.selectionEnd() == active_selection_end:
                return idx

        return 0

    def find_next(self) -> None:
        """Select the next occurrence of the search term in the log view.

        Performs a forward search in the QTextEdit document. If the end of
        the document is reached, wraps around to the beginning. Updates the
        match counter and current selection state after navigation.
        """
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

    def find_prev(self) -> None:
        """Select the previous occurrence of the search term in the log view.

        Performs a backward search in the QTextEdit document. If the
        beginning of the document is reached without a match, wraps around
        to the end. Updates the match counter and current selection state
        after navigation.

        """
        term = self.search_input.text()

        if not term:
            self._update_match_ui(0, 0)
            return

        options = QtGui.QTextDocument.FindFlag.FindBackward
        found = self.logText.find(term, options)

        if not found:
            self.logText.moveCursor(QtGui.QTextCursor.End)
            found = self.logText.find(term, options)

        total = self._count_matches(term)
        current = self._current_match_index(term) if found else 0

        self._update_match_ui(current, total)

    def clear_console(self) -> None:
        """Clear all console output and reset internal log state.

        Empties pending log entries, clears the visible QTextEdit, clears
        cached log history, and resets the search match UI.
        """
        self._pending.clear()
        self.logText.clear()
        self.log_cache.clear()
        self._update_match_ui(0, 0)

    def apply_filter(self, level_text: str) -> None:
        """Filter displayed logs by severity level.

        Updates the active log level filter, flushes any pending log
        entries, and rebuilds the visible log view using cached entries
        that meet the selected severity threshold.

        Args:
            level_text: Log level string (e.g., 'INFO', 'WARNING').
        """
        levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        self.current_filter_level = levels.get(level_text, 10)

        self._flush_pending()

        tokens = ThemeManager.instance().tokens()

        self.logText.clear()

        for record in self.log_cache:
            if record[2] >= self.current_filter_level:
                self.logText.insertHtml(self._build_html_line(*record, tokens=tokens))  # type: ignore

        self.logText.moveCursor(QtGui.QTextCursor.End)

        # Refresh search state after filter rebuild
        self.on_search_changed(self.search_input.text())

    def appendToConsole(
        self,
        time_str: str,
        level_name: str,
        level_no: int,
        name_line: str,
        raw_msg: str,
    ) -> None:
        """Buffer a log record for later rendering in the UI.

        Stores incoming log entries into a pending queue that is processed
        during the next GUI flush cycle. This avoids blocking UI updates
        by batching log rendering.

        Args:
            time_str: Formatted timestamp string for the log entry.
            level_name: Human-readable log level (e.g., INFO, WARNING).
            level_no: Numeric log severity level used for filtering.
            name_line: Source/module identifier associated with the log.
            raw_msg: Raw log message content.
        """
        self._pending.append((time_str, level_name, level_no, name_line, raw_msg))

    def _build_html_line(
        self,
        time_str: str,
        level_name: str,
        level_no: int,
        name_line: str,
        raw_msg: str,
        tokens: Optional[dict] = None,
    ) -> str:
        """Render a single log record as a themed HTML string.

        Converts a structured log entry into an HTML-formatted line using
        the active theme color tokens. The output is designed for display
        inside a Qt rich-text log viewer.

        Args:
            time_str: Timestamp string for the log entry.
            level_name: Log level name (e.g., DEBUG, INFO, WARNING).
            level_no: Numeric log severity level (used for filtering logic;
                not directly used in rendering).
            name_line: Source or logger identifier for the message.
            raw_msg: Raw log message text.
            tokens: Optional theme token dictionary. If not provided, the
                current ThemeManager instance tokens are used.

        Returns:
            HTML string representing the formatted log line.
        """
        if tokens is None:
            tokens = ThemeManager.instance().tokens()  # type: ignore
        assert tokens is not None
        if level_name == "DEBUG":
            lvl_color = tok_css(tokens["log_debug"])
            weight = "normal"
        elif level_name == "INFO":
            lvl_color = tok_css(tokens["log_info"])
            weight = "normal"
        elif level_name in ("WARNING", "ERROR", "CRITICAL"):
            lvl_color = tok_css(
                tokens["log_warning"] if level_name == "WARNING" else tokens["log_error"]
            )
            weight = "bold"
        else:
            lvl_color = tok_css(tokens["log_default"])
            weight = "normal"

        time_color = tok_css(tokens["log_time"])
        loc_color = tok_css(tokens["log_location"])

        time_html = f"<span style='color:{time_color};'>{time_str}</span>"
        padded_level = f"{level_name:<8}".replace(" ", "&nbsp;")
        lvl_html = (
            f"<span style='color:{lvl_color}; font-weight:{weight};'>" f"{padded_level}</span>"
        )
        loc_html = f"<span style='color:{loc_color};'>{name_line}</span>"
        msg_html = f"<span style='color:{lvl_color}; font-weight:{weight};'>{raw_msg}</span>"

        return (
            f"<span style='font-family: Consolas, \"Courier New\", monospace;'>"
            f"{time_html} | {lvl_html} | {loc_html} | {msg_html}</span><br>"
        )

    def _on_theme_changed(self, _mode: str) -> None:
        """Handle application theme changes by refreshing log rendering.

        Re-applies the current log level filter, forcing the log view to
        re-render with updated theme tokens and colors.

        Args:
            _mode: Theme mode identifier (unused but provided by signal).
        """
        self.apply_filter(self.level_filter.currentText())

    def _flush_pending(self) -> None:
        """Flush buffered log records into the QTextEdit in a single batch.

        Transfers all pending log entries into the cache, enforces cache
        size limits, renders only currently visible entries as HTML, and
        appends them to the log view in one insertion pass for performance.

        Also preserves scroll position behavior and updates search state
        if a query is active.
        """
        if not self._pending:
            return

        batch = self._pending
        self._pending = []

        # Maintain cache constraints
        self.log_cache.extend(batch)
        if len(self.log_cache) > self._MAX_CACHE:
            del self.log_cache[: len(self.log_cache) - self._MAX_CACHE]

        tokens = ThemeManager.instance().tokens()

        visible_html = "".join(
            self._build_html_line(*rec, tokens=tokens)  # type: ignore
            for rec in batch
            if rec[2] >= self.current_filter_level
        )

        if not visible_html:
            return

        vsb = self.logText.verticalScrollBar()
        assert vsb is not None
        is_at_bottom = vsb.value() >= (vsb.maximum() - 5)

        self.logText.moveCursor(QtGui.QTextCursor.End)
        self.logText.insertHtml(visible_html)

        if is_at_bottom:
            vsb.setValue(vsb.maximum())

        # Update live search matches if a query is active
        term = self.search_input.text()
        if term:
            self._update_match_ui(
                self._current_match_index(term),
                self._count_matches(term),
            )

    def write(self, message: Any) -> None:
        """Receive log messages and forward them to the UI.

        Acts as the log sink entry point. Extracts structured log
        metadata from the Loguru record and emits it through a Qt signal
        for buffered, thread-safe UI rendering.

        Args:
            message: Log message object containing a structured log
                record.
        """
        record = message.record

        level_no = record["level"].no
        level_name = record["level"].name

        time_str = record["time"].strftime("%Y-%m-%d %H:%M:%S")

        name_line = f"{record['name']}:{record['line']}"

        raw_msg = record["message"].replace("<", "&lt;").replace(">", "&gt;")

        self.appendLogText.emit(
            time_str,
            level_name,
            level_no,
            name_line,
            raw_msg,
        )
