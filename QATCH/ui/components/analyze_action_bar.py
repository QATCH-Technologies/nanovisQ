"""
QATCH.ui.components.analyze_action_bar.py

This module defines the task-bar widget used by the Analyze UI.  The bar
replaces the legacy flat toolbar with a themed card that follows the same
visual language as the application's PlotsUI and ControlsUI components.

Tool buttons are created through :class:`TaskBarBase` so they share the
application-wide `CtrlToolBar` styling.  Icons are tinted from the active
theme and re-tinted whenever the theme changes.

Author(s):
    Paul MacNichol  (paul.macnichol@qatchtech.com)

Date:
    2026-08-18
"""

from __future__ import annotations

import os

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.ui.components import AnimatedComboBox
from QATCH.ui.components.icon_utils import tinted_icon
from QATCH.ui.components.task_bar_base import TaskBarBase, _icon_path
from QATCH.ui.styles.theme_manager import ThemeManager
from QATCH.ui.widgets.saved_state_dot import SavedStateDot


class AnalyzeActionBar(TaskBarBase):
    """Searchable run field + Run Info/Restore/Close + Auto-Fit/position-
    stepper/Modify/Analyze + Advanced/User, grouped into three uncaptioned
    zones, as one themed card.

    Attributes:
        cBox_Runs: the searchable run selector. Typing filters the list via
            an attached QCompleter; `filter_action` is the trailing
            "filters" affordance embedded in the field.
        saved_state_dot, saved_state_widget: the saved-state indicator dot,
            shown inline in `run_zone` beside cBox_Runs/run_actions_bar -
            dot only, no text label (see `UIAnalyze._set_saved_state`,
            which sets a tooltip instead).
        text_Created: hidden internal-state label, not shown in the bar -
            see `_build_run_selector`.
        tBtn_Predict: the Auto-Fit button.
        tBtn_Info, tool_Restore, tool_Cancel: Run Info/Restore/Close - grouped
            in their own `run_actions_bar` toolbar (matching this bar's other
            CtrlToolBar rows) since all three act on the currently loaded
            run specifically.
        tool_Back, tool_Next: the position-step buttons, built via the same
            `_tool_button()` helper as Auto-Fit/Modify/Analyze so they share
            identical chrome.
        tool_Modify, tool_Analyze, tool_Advanced, tool_User: the remaining
            action buttons.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the Analyze action bar.

        The constructor builds each control group, assembles the groups into
        the task-bar layout, applies the current theme to icons, and registers
        a listener so icon styling follows subsequent theme changes.

        Args:
            parent (QtWidgets.QWidget | None): Parent widget that owns this
                action bar. Defaults to `None`.
        """
        super().__init__(parent)

        self._build_run_selector()
        self._build_fit_group()
        self._build_app_group()
        self._assemble()

        self.retint_icon_buttons()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        """Refresh icon styling after the application theme changes.

        Re-tints toolbar icons and the embedded line-edit icons using the
        current theme tokens, then requests a repaint of the action bar.

        Args:
            _mode (str): Theme mode reported by `ThemeManager`. The value is
                intentionally unused because the current token set is queried
                directly from the theme manager.
        """
        self.retint_icon_buttons()
        self._restyle_filter_icon()
        self._restyle_static_line_edit_icons()
        self.update()

    def _build_run_selector(self) -> None:
        """Build the run-selection zone and its associated actions.

        Creates the searchable run combo box, saved-state indicator, inline
        filter controls, and Run Info/Restore/Close toolbar.  The controls are
        constructed here but their signals remain unconnected so `UIAnalyze`
        can retain ownership of application behavior.

        The hidden `text_Created` label is also preserved for compatibility
        with existing code that uses it to track the current run identity.
        """
        self.saved_state_dot = SavedStateDot()
        self.saved_state_widget = QtWidgets.QWidget()
        self.saved_state_widget.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        saved_state_layout = QtWidgets.QHBoxLayout(self.saved_state_widget)
        saved_state_layout.setContentsMargins(0, 0, 0, 0)
        saved_state_layout.setSpacing(0)
        saved_state_layout.addWidget(self.saved_state_dot)

        # Searchable run field
        self.cBox_Runs = AnimatedComboBox(
            icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "down-chevron.svg")
        )
        self.cBox_Runs.setFixedHeight(30)
        self.cBox_Runs.setFixedWidth(260)
        self.cBox_Runs.setEditable(True)
        self.cBox_Runs.setInsertPolicy(QtWidgets.QComboBox.NoInsert)

        completer = QtWidgets.QCompleter(self.cBox_Runs.model(), self.cBox_Runs)
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
        self.cBox_Runs.setCompleter(completer)
        self.cBox_Runs.style_completer_popup(completer)

        line_edit = self.cBox_Runs.lineEdit()
        # Shows whenever nothing is selected
        assert line_edit is not None, "Run combo box has no line edit"
        line_edit.setPlaceholderText("Select a run to load…")
        line_edit.setFrame(False)
        self.search_action = line_edit.addAction(QtGui.QIcon(), QtWidgets.QLineEdit.LeadingPosition)
        self.clear_filter_action = line_edit.addAction(
            QtGui.QIcon(), QtWidgets.QLineEdit.TrailingPosition
        )
        assert self.clear_filter_action is not None, "Run combo box has no clear filter action"
        self.clear_filter_action.setToolTip("Clear filter")
        self.clear_filter_action.setVisible(False)
        self._restyle_static_line_edit_icons()
        self.filter_action = line_edit.addAction(
            QtGui.QIcon(), QtWidgets.QLineEdit.TrailingPosition
        )
        assert self.filter_action is not None, "Run combo box has no filter action"
        self.filter_action.setToolTip("Filter runs…")
        self._filter_active = False
        self._restyle_filter_icon()
        self.tBtn_Info = self._tool_button("Run Info", "info-circle.svg")
        self.tool_Restore = self._tool_button("Restore", "restore.svg")
        self.tool_Cancel = self._tool_button("Close", "cancel.svg")
        self.run_actions_bar = self._make_toolbar()
        self.run_actions_bar.addWidget(self.tBtn_Info)
        self.run_actions_bar.addWidget(self.tool_Restore)
        self.run_actions_bar.addWidget(self.tool_Cancel)
        self.run_zone = QtWidgets.QHBoxLayout()
        self.run_zone.setContentsMargins(0, 0, 0, 0)
        self.run_zone.setSpacing(8)
        self.run_zone.addWidget(self.cBox_Runs, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.run_zone.addWidget(self.saved_state_widget, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.run_zone.addWidget(self.run_actions_bar, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        # UIAnalyze/MainWindow track the current run's identity via this
        # label's text
        self.text_Created = QtWidgets.QLabel("[NONE]", self)
        self.text_Created.hide()

    def set_filter_active(self, active: bool) -> None:
        """Update the visual state of the run-filter controls.

        An active filter changes the filter icon to the theme accent color and
        reveals the inline clear action.  An inactive filter restores the
        muted icon and hides the clear action.

        Args:
            active (bool): Whether one or more run filters are currently
                applied.
        """
        self._filter_active = active
        assert self.clear_filter_action is not None, "Run combo box has no clear filter action"
        self.clear_filter_action.setVisible(active)
        self._restyle_filter_icon()

    def _restyle_filter_icon(self) -> None:
        """Apply the current theme color to the filter icon.

        The icon uses the accent token while filtering is active and the
        muted text token otherwise.
        """
        tok = ThemeManager.instance().tokens()
        color = QtGui.QColor(
            *(tok["flat_accent"] if self._filter_active else tok["flat_text_muted"])
        )
        assert self.filter_action is not None, "Run combo box has no filter action"
        self.filter_action.setIcon(tinted_icon(_icon_path("filter.svg"), color, 18))

    def _restyle_static_line_edit_icons(self) -> None:
        """Refresh the search and clear icons embedded in the run field.

        Both icons remain muted regardless of filter state.  This keeps their
        visual role distinct from the filter icon, which uses the accent color
        when a filter is active.
        """
        tok = ThemeManager.instance().tokens()
        color = QtGui.QColor(*tok["flat_text_muted"])
        assert self.search_action is not None, "Run combo box has no search action"
        self.search_action.setIcon(tinted_icon(_icon_path("search.svg"), color, 16))
        assert self.clear_filter_action is not None, "Run combo box has no clear filter action"
        self.clear_filter_action.setIcon(tinted_icon(_icon_path("clear.svg"), color, 14))

    def _build_fit_group(self) -> None:
        """Build the fit and analysis action group.

        Creates the Auto-Fit, previous/next position, Modify, and Analyze
        controls and places them in a shared themed toolbar.  Loading is not
        handled here; run selection is performed by the searchable run field.
        """
        self.tBtn_Predict = self._tool_button("Auto-Fit", "stars.svg")
        self.tool_Back = self._tool_button("Back", "previous.svg")
        self.tool_Next = self._tool_button("Next", "next.svg")
        self.tool_Modify = self._tool_button("Modify", "modify.svg", checkable=True)
        self.tool_Analyze = self._tool_button("Analyze", "play-circle.svg")

        self.fit_bar = self._make_toolbar()
        self.fit_bar.addWidget(self.tBtn_Predict)
        self.fit_bar.addWidget(self.tool_Back)
        self.fit_bar.addWidget(self.tool_Next)
        self.fit_bar.addWidget(self.tool_Modify)
        self.fit_bar.addWidget(self.tool_Analyze)

    def _build_app_group(self) -> None:
        """Build the application-level action group.

        Creates the Advanced and User controls, places them in the application
        toolbar, and initially disables the User control until the owning UI
        has refreshed the actual signed-in state.
        """
        self.tool_Advanced = self._tool_button("Advanced", "gear.svg", checkable=True)
        self.tool_User = self._tool_button("Anonymous", "user-circle.svg", checkable=True)
        # Starts disabled
        self.tool_User.setEnabled(False)

        self.app_bar = self._make_toolbar()
        self.app_bar.addWidget(self.tool_Advanced)
        self.app_bar.addWidget(self.tool_User)

    def _assemble(self) -> None:
        """Assemble the three action-bar zones into the final layout.

        The run, fit/analyze, and application toolbars are separated with
        dividers whose height is derived from the fit toolbar's size hint.
        This keeps divider geometry aligned with the themed toolbar controls
        without relying on a hard-coded height.
        """
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(*self.OUTER_MARGINS)
        layout.setSpacing(self.ZONE_SPACING)
        toolbar_h = self.fit_bar.sizeHint().height()
        layout.addLayout(self.run_zone)
        layout.addWidget(self._make_divider(toolbar_h), 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.fit_bar, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)
        layout.addWidget(self._make_divider(toolbar_h), 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.app_bar, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
