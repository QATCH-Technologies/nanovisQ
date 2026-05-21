import os

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QFrame,
    QSizePolicy,
    QStackedWidget,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QDateEdit,
    QTimeEdit,
    QGraphicsDropShadowEffect,
)

from PyQt5.QtCore import (
    Qt,
    pyqtSignal,
    QPropertyAnimation,
    QEasingCurve,
    QDateTime,
    QDate,
    QTime,
    QParallelAnimationGroup,
)
from PyQt5.QtGui import (
    QIcon,
    QCloseEvent,
    QColor,
)
from QATCH.common.architecture import Architecture


class RecoveryFilterWidget(QWidget):
    """A specialized popup frame for configuring run recovery filters.

    This widget provides a flyout menu containing various filter criteria,
    including status (Good/Bad), date/time ranges, and numeric ranges for
    duration, data points, and file size. It emits a signal when filters
    are applied or reset.

    The popup is implemented as a transparent outer ``QWidget`` containing
    an inner ``QFrame`` (``self._panel``). The outer container reserves
    margin space around the inner frame so that a ``QGraphicsDropShadowEffect``
    applied to the panel can render a soft, rounded shadow that follows the
    panel's ``border-radius`` instead of the rectangular OS popup outline.

    Attributes:
        filtersChanged (pyqtSignal): Signal emitted when the user clicks 'Apply'
            or 'Reset', carrying a dict of the active filter criteria.
        current_filters (dict): The initial filter state used to populate the UI.
        status_combo (QComboBox): Dropdown for selecting run status.
        date_from (QDateEdit): Input for the start date of the search range.
        time_from (QTimeEdit): Input for the start time of the search range.
        date_to (QDateEdit): Input for the end date of the search range.
        time_to (QTimeEdit): Input for the end time of the search range.
        duration_min (QDoubleSpinBox): Minimum duration filter input.
        duration_max (QDoubleSpinBox): Maximum duration filter input.
        points_min (QSpinBox): Minimum data points filter input.
        points_max (QSpinBox): Maximum data points filter input.
        size_min (QDoubleSpinBox): Minimum file size filter input.
        size_max (QDoubleSpinBox): Maximum file size filter input.

    TODO: The drop shadow has sharp corners on the bottom of the filter for some reason and I cannot
        figure out what is causing it.
    """

    filters_changed = pyqtSignal(dict)

    # Was: all 0
    _SHADOW_MARGIN_L = 28
    _SHADOW_MARGIN_T = 24
    _SHADOW_MARGIN_R = 28
    _SHADOW_MARGIN_B = 32  # extra to accommodate offset(0, 4)

    def __init__(self, parent: QWidget | None = None, current_filters: dict | None = None) -> None:
        """Initializes the RecoveryFilter with customizable UI elements and initial state.

        The constructor builds a multi-section layout containing status selection,
        dynamic date/time rows that can be toggled on/off, and range-based
        numeric inputs. It also applies a custom stylesheet with icons for
        calendars and chevrons.

        Args:
            parent (QWidget, optional): The parent widget for the popup.
                Defaults to None.
            current_filters (dict, optional): A dictionary containing existing
                filter values to pre-populate the fields. Defaults to None.
        """
        super().__init__(
            parent,
            Qt.WindowFlags(
                Qt.WindowType.Popup
                | Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.NoDropShadowWindowHint
            ),
        )
        self.current_filters = current_filters or {}
        self._from_set = False
        self._to_set = False

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self._panel = RoundedPanel(self)
        self._panel.setObjectName("RecoveryFilterInner")

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(
            self._SHADOW_MARGIN_L,
            self._SHADOW_MARGIN_T,
            self._SHADOW_MARGIN_R,
            self._SHADOW_MARGIN_B,
        )
        outer_layout.setSpacing(0)
        outer_layout.addWidget(self._panel)

        shadow = QGraphicsDropShadowEffect(self._panel)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 70))
        self._panel.setGraphicsEffect(shadow)

        self.setStyleSheet(
            RecoveryFilterWidget._build_stylesheet(
                icon_cal=os.path.join(Architecture.get_path(), "QATCH", "icons", "date-range.svg"),
                icon_up=os.path.join(Architecture.get_path(), "QATCH", "icons", "up-chevron.svg"),
                icon_down=os.path.join(
                    Architecture.get_path(), "QATCH", "icons", "down-chevron.svg"
                ),
            )
        )
        main_layout = QVBoxLayout(self._panel)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(4)

        title = QLabel("Filter Options")
        title.setObjectName("titleLabel")

        self.close_btn = QPushButton()
        self.close_btn.setIcon(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "clear.svg"))
        )
        self.close_btn.setObjectName("closeBtn")
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.setToolTip("Close")
        self.close_btn.clicked.connect(self.close)

        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self.close_btn)
        main_layout.addLayout(title_row)

        sections_layout = QVBoxLayout()
        sections_layout.setContentsMargins(0, 0, 0, 0)
        sections_layout.setSpacing(10)

        # Ruling Filters
        ruling_section = self._make_section("RULING")
        self.ruling_combo = QComboBox()
        self.ruling_combo.addItems(["Any", "Good", "Bad"])
        self.ruling_combo.setMinimumWidth(160)
        ruling_row = QHBoxLayout()
        ruling_row.setContentsMargins(0, 0, 0, 0)
        ruling_row.setSpacing(6)
        ruling_row.addWidget(self.ruling_combo, 1)
        ruling_section.addLayout(ruling_row)
        sections_layout.addLayout(ruling_section)

        # Date/Time Ranges
        date_section = self._make_section("DATE / TIME")

        (
            from_row_layout,
            self._date_from_placeholder,
            self._date_from_active,
            self.date_from,
            self.time_from,
            self._date_from_clear,
            self._date_from_stack,
        ) = self._build_date_row(True)

        (
            to_row_layout,
            self._date_to_placeholder,
            self._date_to_active,
            self.date_to,
            self.time_to,
            self._date_to_clear,
            self._date_to_stack,
        ) = self._build_date_row(False)

        self._date_from_placeholder.clicked.connect(lambda: self._activate_date_row(True))
        self._date_from_clear.clicked.connect(lambda: self._clear_date_row(True))
        self._date_to_placeholder.clicked.connect(lambda: self._activate_date_row(False))
        self._date_to_clear.clicked.connect(lambda: self._clear_date_row(False))

        date_section.addLayout(from_row_layout)
        date_section.addLayout(to_row_layout)
        sections_layout.addLayout(date_section)

        # Numeric Ranges
        self.duration_min = self._make_double_spin()
        self.duration_max = self._make_double_spin()
        self.duration_max.setToolTip("Any = no upper limit")
        self.duration_max.setSpecialValueText("Any")
        sections_layout.addLayout(
            self._make_range_section("DURATION (s)", self.duration_min, self.duration_max)
        )

        self.points_min = self._make_int_spin()
        self.points_max = self._make_int_spin()
        self.points_max.setToolTip("Any = no upper limit")
        self.points_max.setSpecialValueText("Any")
        sections_layout.addLayout(
            self._make_range_section("DATA POINTS", self.points_min, self.points_max)
        )

        self.size_min = self._make_double_spin()
        self.size_max = self._make_double_spin()
        self.size_max.setToolTip("Any = no upper limit")
        self.size_max.setSpecialValueText("Any")
        sections_layout.addLayout(
            self._make_range_section("FILE SIZE (MB)", self.size_min, self.size_max)
        )

        main_layout.addLayout(sections_layout)

        # Separator above buttons
        sep = QFrame()
        sep.setObjectName("filterSeparator")
        main_layout.addWidget(sep)

        # Reset and Apply Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setObjectName("resetBtn")
        self.reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_btn.clicked.connect(self._on_reset)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setObjectName("applyBtn")
        self.apply_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.apply_btn.clicked.connect(self._on_apply)

        btn_row.addWidget(self.reset_btn)
        btn_row.addWidget(self.apply_btn)
        main_layout.addLayout(btn_row)

        self.setMinimumWidth(340)
        self._populate_from(current_filters or {})

    @staticmethod
    def _build_stylesheet(icon_cal: str = "", icon_up: str = "", icon_down: str = "") -> str:
        """Generates the Qt Style Sheet (QSS) for the Recovery Filter popup.

        This method constructs a comprehensive CSS-like string used to style the
        FilterPopup. It handles theme colors, border radii, and custom SVG icon
        injection for interactive elements like dropdowns and spin boxes.

        Args:
            icon_cal: The file path to the calendar SVG icon used in QDateEdit.
                Defaults to an empty string (standard rendering).
            icon_up: The file path to the upward chevron SVG icon for spin boxes.
                Defaults to an empty string.
            icon_down: The file path to the downward chevron SVG icon for spin boxes.
                Defaults to an empty string.

        Returns:
            str: A formatted QSS string containing all style rules for the popup
                and its child widgets.
        """

        def _url(p: str) -> str:
            """Formats a file path into a QSS image URL property."""
            return ("image: url(" + p.replace(os.sep, "/") + ");") if p else ""

        _cal_img = _url(icon_cal)
        _up_img = _url(icon_up)
        _down_img = _url(icon_down)

        _btn_bg = "rgba(245, 245, 245, 210)"
        _btn_hover = "rgba(0, 114, 189, 30)"
        _btn_pressed = "rgba(0, 114, 189, 58)"
        _btn_border = "rgba(0, 0, 0, 14)"

        return f"""
            /* Labels */
            QLabel {{
                color: #555555;
                font-size: 9pt;
                background: transparent;
                border: none;
            }}
            QLabel#titleLabel {{
                color: #888888;
                font-size: 9pt;
                font-weight: 400;
            }}
            QLabel#sectionLabel {{
                color: #444444;
                font-size: 9pt;
                font-weight: 600;
                padding-top: 2px;
            }}
            QLabel#rangeSep {{
                color: #999999;
                font-size: 9pt;
                padding: 0 2px;
            }}
            QLabel#dateRangeLbl {{
                color: #888888;
                font-size: 9pt;
                min-width: 28px;
                max-width: 28px;
            }}

            /* Input fields */
            QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit {{
                background-color: rgba(255, 255, 255, 180);
                border: 1px solid rgba(0, 0, 0, 20);
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 9pt;
                color: #333333;
                min-height: 20px;
                selection-background-color: rgba(0, 114, 189, 80);
            }}
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover,
            QDateEdit:hover, QTimeEdit:hover {{
                background-color: rgba(255, 255, 255, 220);
                border-color: rgba(0, 0, 0, 32);
            }}
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
            QDateEdit:focus, QTimeEdit:focus {{
                border: 1px solid rgba(0, 114, 189, 120);
                background-color: rgba(255, 255, 255, 255);
            }}

            /* ComboBox drop-down */
            QComboBox::drop-down {{
                subcontrol-origin: border;
                subcontrol-position: right center;
                width: 22px;
                border-left: 1px solid {_btn_border};
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
                background: {_btn_bg};
            }}
            QComboBox::drop-down:hover  {{ background: {_btn_hover}; }}
            QComboBox::drop-down:pressed {{ background: {_btn_pressed}; }}
            QComboBox::down-arrow {{
                width: 8px;
                height: 8px;
                {_down_img}
            }}

            /* QDateEdit calendar button */
            QDateEdit::drop-down {{
                subcontrol-origin: border;
                subcontrol-position: right center;
                width: 22px;
                border-left: 1px solid {_btn_border};
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
                background: {_btn_bg};
            }}
            QDateEdit::drop-down:hover  {{ background: {_btn_hover}; }}
            QDateEdit::drop-down:pressed {{ background: {_btn_pressed}; }}
            QDateEdit::down-arrow {{
                width: 12px;
                height: 12px;
                {_cal_img}
            }}

            /* Spin-box & TimeEdit up/down buttons */
            QSpinBox::up-button, QDoubleSpinBox::up-button, QTimeEdit::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 16px;
                border-left: 1px solid {_btn_border};
                border-bottom: 1px solid {_btn_border};
                border-top-right-radius: 3px;
                background: {_btn_bg};
            }}
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QTimeEdit::up-button:hover {{
                background: {_btn_hover};
            }}
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
            QTimeEdit::up-button:pressed {{
                background: {_btn_pressed};
            }}

            QSpinBox::down-button, QDoubleSpinBox::down-button, QTimeEdit::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 16px;
                border-left: 1px solid {_btn_border};
                border-top: 1px solid {_btn_border};
                border-bottom-right-radius: 3px;
                background: {_btn_bg};
            }}
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover,
            QTimeEdit::down-button:hover {{
                background: {_btn_hover};
            }}
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed,
            QTimeEdit::down-button:pressed {{
                background: {_btn_pressed};
            }}

            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow, QTimeEdit::up-arrow {{
                width: 6px;
                height: 6px;
                {_up_img}
            }}
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow, QTimeEdit::down-arrow {{
                width: 6px;
                height: 6px;
                {_down_img}
            }}

            /* Separator */
            QFrame#filterSeparator {{
                background-color: rgba(0, 0, 0, 14);
                border: none;
                max-height: 1px;
                min-height: 1px;
            }}

            /* Footer buttons */
            QPushButton {{
                border-radius: 5px;
                padding: 5px 14px;
                font-size: 9pt;
            }}
            QPushButton#applyBtn {{
                background-color: rgba(0, 114, 189, 25);
                color: #005b9f;
                border: 1px solid rgba(0, 114, 189, 60);
            }}
            QPushButton#applyBtn:hover  {{ background-color: rgba(0, 114, 189, 45); }}
            QPushButton#resetBtn {{
                background-color: transparent;
                color: #666666;
                border: 1px solid rgba(0, 0, 0, 22);
            }}
            QPushButton#resetBtn:hover  {{ background-color: rgba(0, 0, 0, 6); }}

            /* Date row widgets (from new date section) */
            QPushButton#addDateBtn {{
                background-color: transparent;
                color: #aaaaaa;
                border: 1px dashed rgba(0, 0, 0, 22);
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 9pt;
                text-align: left;
            }}
            QPushButton#addDateBtn:hover {{
                background-color: rgba(0, 114, 189, 8);
                border-color: rgba(0, 114, 189, 55);
                color: rgba(0, 100, 170, 200);
            }}
            QPushButton#clearDateBtn {{
                background-color: transparent;
                border: none;
                color: #aaaaaa;
                font-size: 11pt;
                padding: 0px;
                min-width: 20px;  max-width: 20px;
                min-height: 20px; max-height: 20px;
                border-radius: 10px;
            }}
            QPushButton#clearDateBtn:hover {{
                background-color: rgba(0, 0, 0, 10);
                color: #555555;
            }}
            QPushButton#closeBtn {{
                background-color: transparent;
                border: none;
                color: #aaaaaa;
                padding: 0px;
                min-width: 22px;  max-width: 22px;
                min-height: 22px; max-height: 22px;
                border-radius: 11px;
                qproperty-iconSize: 10px 10px;
            }}
            QPushButton#closeBtn:hover {{
                background-color: rgba(0, 0, 0, 12);
                color: #555555;
            }}
            QPushButton#closeBtn:pressed {{
                background-color: rgba(0, 0, 0, 22);
            }}
        """

    def _make_section(self, title_text: str) -> QVBoxLayout:
        """Creates a layout container with a styled section header.

        This helper method generates a vertical layout with zero margins and
        standardized spacing, pre-populated with a QLabel acting as a header.
        The label is assigned the 'sectionLabel' object name for QSS styling.

        Args:
            title_text: The string to display as the section heading.

        Returns:
            QVBoxLayout: A layout object ready to have additional widgets
                or layouts added to it.
        """
        section = QVBoxLayout()
        section.setContentsMargins(0, 0, 0, 0)
        section.setSpacing(4)
        lbl = QLabel(title_text)
        lbl.setObjectName("sectionLabel")
        section.addWidget(lbl)
        return section

    def _build_date_row(self, is_from: bool) -> tuple[
        QVBoxLayout,
        QPushButton,
        QWidget,
        QDateEdit,
        QTimeEdit,
        QPushButton,
        QStackedWidget,
    ]:
        """Constructs a toggleable date/time input row.

        Creates a dual-state UI component: a dashed 'placeholder' button for
        inactive states and an 'active' container with date/time editors and
        a clear button.

        Args:
            is_from: If True, prefixes the row with 'From:'; otherwise uses 'To:'.

        Returns:
            A tuple containing:
                - layout (QHBoxLayout): The main row container.
                - placeholder (QPushButton): The dashed button shown when unset.
                - active_container (QWidget): The widget holding the editors.
                - date_edit (QDateEdit): The date input widget.
                - time_edit (QTimeEdit): The time input widget.
                - clear_btn (QPushButton): The 'x' button to reset the row.
        """
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Inactive placeholder
        label_text = "+  From date" if is_from else "+  To date"
        placeholder = QPushButton(label_text)
        placeholder.setObjectName("addDateBtn")
        placeholder.setCursor(Qt.CursorShape.PointingHandCursor)
        placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Active row widget
        active_widget = QWidget()
        active_widget.setObjectName("dateActiveRow")
        row_layout = QHBoxLayout(active_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        bound_lbl = QLabel("From" if is_from else "To")
        bound_lbl.setObjectName("dateRangeLbl")

        default_dt = (
            QDateTime.currentDateTime().addDays(-30) if is_from else QDateTime.currentDateTime()
        )

        date_edit = QDateEdit()
        date_edit.setCalendarPopup(True)
        date_edit.setDisplayFormat("yyyy-MM-dd")
        date_edit.setDate(default_dt.date())
        date_edit.setMinimumWidth(75)

        time_edit = QTimeEdit()
        time_edit.setDisplayFormat("HH:mm")
        time_edit.setTime(QTime(0, 0) if is_from else default_dt.time())
        time_edit.setFixedWidth(78)

        clear_btn = QPushButton()
        clear_btn.setIcon(
            QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "clear.svg"))
        )
        clear_btn.setObjectName("clearDateBtn")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.setToolTip("Clear this date bound")

        row_layout.addWidget(bound_lbl)
        row_layout.addWidget(date_edit, 1)
        row_layout.addWidget(time_edit)
        row_layout.addWidget(clear_btn)

        # Stack: index 0 = placeholder, index 1 = active editors
        stack = QStackedWidget()
        stack.addWidget(placeholder)  # index 0
        stack.addWidget(active_widget)  # index 1
        stack.setCurrentIndex(0)
        # Match height to the taller (active) child so the popup doesn't jump
        stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        outer.addWidget(stack)

        return outer, placeholder, active_widget, date_edit, time_edit, clear_btn, stack

    def _make_double_spin(self) -> QSpinBox:
        """Factory method to create a standardized QDoubleSpinBox.

        Configures a spin box with a large range (0 to 1,000,000), two decimal
        places of precision, and a default starting value of 0.0. This is
        primarily used for duration and file size filters.

        Returns:
            QDoubleSpinBox: A configured double precision spin box widget.
        """
        w = QDoubleSpinBox()
        w.setRange(0.0, 1_000_000.0)
        w.setDecimals(2)
        w.setValue(0.0)
        return w

    def _make_int_spin(self) -> QSpinBox:
        """Factory method to create a standardized QSpinBox for integer values.

        Configures a spin box with a broad range (0 to 100,000,000) and a default
        starting value of 0. This is typically used for data point count filters.

        Returns:
            QSpinBox: A configured integer spin box widget.
        """
        w = QSpinBox()
        w.setRange(0, 100_000_000)
        w.setValue(0)
        return w

    def _make_range_section(
        self, title_text: str, min_widget: QWidget, max_widget: QWidget
    ) -> QVBoxLayout:
        """Creates a labeled section containing a Min/Max input range row.

        This helper utilizes `_make_section` to create the header and then
        assembles a horizontal row containing the 'Min' label, the minimum
        input widget, the 'Max' label, and the maximum input widget.

        Args:
            title_text: The category name to display as the section header
                (e.g., "DURATION (s)").
            min_widget: The widget used for the lower bound input
                (typically a QSpinBox or QDoubleSpinBox).
            max_widget: The widget used for the upper bound input
                (typically a QSpinBox or QDoubleSpinBox).

        Returns:
            QVBoxLayout: A vertical layout containing the header and the
                horizontal range row.
        """
        section = self._make_section(title_text)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        min_lbl = QLabel("Min")
        max_lbl = QLabel("Max")

        row.addWidget(min_lbl)
        row.addWidget(min_widget, 1)
        row.addSpacing(6)
        row.addWidget(max_lbl)
        row.addWidget(max_widget, 1)
        section.addLayout(row)
        return section

    def _activate_date_row(self, is_from: bool) -> None:
        """Transitions a date/time row from placeholder to active state.

        Sets the internal tracking boolean for the specified row and triggers
        a UI update to swap the placeholder button for the actual
        date/time editors.

        Args:
            is_from: If True, activates the 'From' (start) date row.
                If False, activates the 'To' (end) date row.
        """
        if is_from:
            self._from_set = True
        else:
            self._to_set = True
        self._update_date_visuals()

    def _clear_date_row(self, is_from: bool) -> None:
        """Resets a date/time row to its default state and hides it.

        This method sets the internal tracking flag to False, restores the
        default date and time values for the specified row, and updates
        the UI to show the placeholder button again.

        Args:
            is_from: If True, resets the 'From' (start) date row to 30 days ago
                at 00:00. If False, resets the 'To' (end) date row to the
                current date and time.
        """
        if is_from:
            self._from_set = False
            self.date_from.setDate(QDateTime.currentDateTime().addDays(-30).date())
            self.time_from.setTime(QTime(0, 0))
        else:
            self._to_set = False
            self.date_to.setDate(QDateTime.currentDateTime().date())
            self.time_to.setTime(QDateTime.currentDateTime().time())
        self._update_date_visuals()

    def _update_date_visuals(self) -> None:
        """Synchronizes the visibility of date/time widgets with their active state.

        This method checks the internal `_from_set` and `_to_set` boolean flags
        to determine whether to show the '+ Set date/time' placeholder buttons
        or the actual date/time input editors for both the 'From' and 'To' rows.
        """
        self._date_from_stack.setCurrentIndex(1 if self._from_set else 0)
        self._date_to_stack.setCurrentIndex(1 if self._to_set else 0)

    def _populate_from(self, f: dict) -> None:
        """Populates the popup widgets with values from a filter dictionary.

        This method synchronizes the UI state with a provided dictionary. It handles
        conditional logic for 'Any' (infinite) values in numeric ranges and
        independently restores the 'From' and 'To' date bounds, ensuring the
        visual state (placeholder vs. active editors) is updated accordingly.

        Args:
            f: A dictionary containing filter criteria. Expected keys include:
                - 'status': str ("Good", "Bad", or others for "Any")
                - 'date_from'/'date_to': datetime objects
                - 'duration_min'/'duration_max': float
                - 'points_min'/'points_max': int
                - 'size_min'/'size_max': float
        """
        status = f.get("status")
        if status in ("Good", "Bad"):
            self.ruling_combo.setCurrentText(status)
        else:
            self.ruling_combo.setCurrentIndex(0)  # "Any"

        self._from_set = "date_from" in f
        self._to_set = "date_to" in f

        if self._from_set:
            try:
                dt = f["date_from"]
                self.date_from.setDate(QDate(dt.year, dt.month, dt.day))
                self.time_from.setTime(QTime(dt.hour, dt.minute))
            except Exception:
                self._from_set = False

        if self._to_set:
            try:
                dt = f["date_to"]
                self.date_to.setDate(QDate(dt.year, dt.month, dt.day))
                self.time_to.setTime(QTime(dt.hour, dt.minute))
            except Exception:
                self._to_set = False

        self._update_date_visuals()

        self.duration_min.setValue(
            float(f["duration_min"]) if f.get("duration_min") is not None else 0.0
        )
        d_max = f.get("duration_max")
        self.duration_max.setValue(
            0.0 if (d_max is None or d_max == float("inf")) else float(d_max)
        )

        self.points_min.setValue(int(f["points_min"]) if f.get("points_min") is not None else 0)
        p_max = f.get("points_max")
        self.points_max.setValue(0 if (p_max is None or p_max == float("inf")) else int(p_max))

        self.size_min.setValue(float(f["size_min"]) if f.get("size_min") is not None else 0.0)
        s_max = f.get("size_max")
        self.size_max.setValue(0.0 if (s_max is None or s_max == float("inf")) else float(s_max))

    def _collect(self) -> dict:
        """Collects all current UI values into a filter criteria dictionary.

        This method reads the state of every widget in the popup. It converts
        Qt-specific objects (like QDate and QTime) into standard Python
        datetimes and handles the 'Any' logic for numeric ranges by
        substituting a value of 0 with positive infinity where appropriate.

        Returns:
            dict: A dictionary of active filters. Keys are only included if
                they deviate from default 'Any' states. Possible keys:
                - 'status' (str): "Good" or "Bad"
                - 'date_from' (datetime): Start boundary
                - 'date_to' (datetime): End boundary
                - 'duration_min'/'duration_max' (float)
                - 'points_min'/'points_max' (int/float)
                - 'size_min'/'size_max' (float)
        """
        filters = {}

        if self.ruling_combo.currentText() != "Any":
            filters["status"] = self.ruling_combo.currentText()

        if self._from_set:
            filters["date_from"] = QDateTime(
                self.date_from.date(), self.time_from.time()
            ).toPyDateTime()
        if self._to_set:
            filters["date_to"] = QDateTime(self.date_to.date(), self.time_to.time()).toPyDateTime()

        d_min = self.duration_min.value()
        d_max = self.duration_max.value()
        if d_min > 0.0 or d_max > 0.0:
            filters["duration_min"] = d_min
            filters["duration_max"] = d_max if d_max > 0.0 else float("inf")

        p_min = self.points_min.value()
        p_max = self.points_max.value()
        if p_min > 0 or p_max > 0:
            filters["points_min"] = p_min
            filters["points_max"] = p_max if p_max > 0 else float("inf")

        s_min = self.size_min.value()
        s_max = self.size_max.value()
        if s_min > 0.0 or s_max > 0.0:
            filters["size_min"] = s_min
            filters["size_max"] = s_max if s_max > 0.0 else float("inf")

        return filters

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Intercepts the close event to perform a collapse animation.

        Instead of closing immediately, the event is ignored while a
        QPropertyAnimation shrinks the widget's height to zero. Once
        the animation finishes, the close method is called again with
        an internal flag set to allow the window to actually close.

        Args:
            event (QCloseEvent): The window close event received from the OS or
                parent widget.
        """
        """Collapse the popup before actually closing."""
        if getattr(self, "_closing_animated", False):
            event.accept()
            return
        event.ignore()
        self._closing_animated = True

        height_anim = QPropertyAnimation(self, b"maximumHeight")
        height_anim.setDuration(180)
        height_anim.setEasingCurve(QEasingCurve.InCubic)
        height_anim.setStartValue(self.height())
        height_anim.setEndValue(0)

        fade_anim = QPropertyAnimation(self, b"windowOpacity")
        fade_anim.setDuration(180)
        fade_anim.setEasingCurve(QEasingCurve.InCubic)
        fade_anim.setStartValue(self.windowOpacity())
        fade_anim.setEndValue(0.0)

        group = QParallelAnimationGroup(self)
        group.addAnimation(height_anim)
        group.addAnimation(fade_anim)
        group.finished.connect(self.close)  # type: ignore
        group.start()
        self._close_anim = group  # keep alive

    def _on_apply(self) -> None:
        """Gathers current filter settings and signals the parent to apply them.

        This method acts as the finalization step for the filter popup. It
        executes the data collection logic, emits the `filters_changed` signal
        with the resulting dictionary, and initiates the animated close sequence.
        """
        self.filters_changed.emit(self._collect())
        self.close()

    def _on_reset(self) -> None:
        """Resets all filter widgets to their default values and closes the popup.

        This method clears the status selection, resets the date range to the
        default 30-day window, zeros out all numeric ranges (triggering the 'Any'
        state), and emits an empty dictionary via `filters_changed` before
        initiating the close animation.
        """
        self.ruling_combo.setCurrentIndex(0)
        self._from_set = False
        self._to_set = False
        self.date_from.setDate(QDateTime.currentDateTime().addDays(-30).date())
        self.time_from.setTime(QTime(0, 0))
        self.date_to.setDate(QDateTime.currentDateTime().date())
        self.time_to.setTime(QDateTime.currentDateTime().time())
        self._update_date_visuals()
        self.duration_min.setValue(0)
        self.duration_max.setValue(0)
        self.points_min.setValue(0)
        self.points_max.setValue(0)
        self.size_min.setValue(0)
        self.size_max.setValue(0)
        self.filters_changed.emit({})
        self.close()
