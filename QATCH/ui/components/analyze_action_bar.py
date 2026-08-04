"""Themed top action bar for AnalyzeUI.

Replaces AnalyzeUI's old flat #DDDDDD toolbar with a themed card matching
the PlotsUI/ControlsUI visual language. The internal toolbars use
objectName "CtrlToolBar" so they pick up the exact same app-wide QSS that
already themes ControlsUI's toolbar (see app_theme.qss) - no new QSS rules
needed.

Laid out as three captioned zones - RUN / FIT & ANALYZE / APP - separated
by hairline dividers, per the "2a" task bar redesign: the run selector is
a searchable, filterable field with the saved-state indicator folded into
the RUN caption itself, and Back/Next now flank a plain "pos N/6" label
instead of standing alone - both still built via the same `_tool_button()`
helper as every other button in the bar, so they read as one family rather
than a visually distinct control.

Every toolbutton icon (and the search/filter/clear icons embedded in the
run field) is tinted from the active theme's `flat_*` tokens at build time
and re-tinted on every theme change - the source SVGs are plain dark line
art with no light/dark variants of their own, so painting them untinted
would read fine in light mode and go nearly invisible in dark mode.

This widget only constructs and lays out buttons/labels; it does not wire
their signals. All the callbacks (load_run, action_back, action_next, ...)
live on UIAnalyze, not here, so UIAnalyze.setup_ui connects them after
construction - the same "wrap, don't own" pattern PlotContainer uses for
its wrapped plot_widget.
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.ui.components import AnimatedComboBox
from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.components.icon_utils import tinted_icon
from QATCH.ui.labels.section_label import SectionHeader
from QATCH.ui.styles.fonts import FONT_SANS_SEMIBOLD
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css
from QATCH.ui.widgets.saved_state_dot import SavedStateDot


def _icon_path(icon_name: str) -> str:
    return os.path.join(Architecture.get_path(), "QATCH", "icons", icon_name)


class _VDivider(QtWidgets.QFrame):
    """A hairline vertical divider between action-bar zones, themed from
    `flat_border` so it stays correct across light/dark switches.

    Sized to `height` (the zones' own CtrlToolBar row height, passed in by
    _assemble) and added with AlignVCenter there - not left to stretch to
    the full zone height (which reads as a full edge-to-edge rule spanning
    the caption line too), and not an arbitrary short fixed height either
    (which reads as stubbier than ControlsUI's own toolbar separators,
    whose `margin: 5px 4px` QSS makes them nearly as tall as their row).
    Centered on the bar as a whole, not bottom-anchored to the toolbar row
    specifically - anchoring to the row alone left the divider sitting
    visibly low relative to the bar's overall vertical center once the
    caption line above it is taken into account.
    """

    def __init__(self, height: int, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setFixedWidth(1)
        self.setFixedHeight(height)
        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_theme()

    def _apply_theme(self) -> None:
        tok = ThemeManager.instance().tokens()
        self.setStyleSheet(f"background-color: {tok_css(tok['flat_border'])}; border: none;")


class AnalyzeActionBar(QtWidgets.QWidget):
    """Searchable run field + Auto-Fit/position-stepper/Modify/Analyze +
    Advanced/Close/User, grouped into three captioned zones (RUN / FIT &
    ANALYZE / APP), as one themed card.

    Public attributes (all plain Qt widgets - the caller wires their
    signals and owns their behavior):
        active_run_header, cBox_Runs: the searchable run selector (header
            above the combo box). Typing filters the list via an attached
            QCompleter; `filter_action` is the trailing "filters" affordance
            embedded in the field.
        saved_state_dot, saved_state_label, saved_state_widget: the
            "Loaded & saved" status pill, now shown inline beside the RUN
            caption (folded up per the 2a redesign) rather than beneath
            cBox_Runs.
        text_Created: hidden internal-state label, not shown in the bar -
            see `_build_run_selector`.
        tBtn_Predict, tBtn_Info: Auto-Fit/Run Info buttons - tBtn_Info sits
            in its own `run_info_bar` toolbar so it picks up the same
            CtrlToolBar theming as every other button in the bar.
        tool_Back, tool_Next: the position-step buttons, built via the same
            `_tool_button()` helper as Auto-Fit/Modify/Analyze so they share
            identical chrome; `position_label` is the plain "pos N/6" text
            sitting between them.
        tool_Modify, tool_Analyze, tool_Advanced, tool_Cancel, tool_User:
            the remaining action buttons (Close now lives in the APP zone
            alongside Advanced/User, not beside Back/Next).
    """

    _R = 12.0

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

        # (button, icon_name) pairs built via _tool_button(), retinted as a
        # batch on every theme change (see _retint_icon_buttons).
        self._icon_buttons: list = []

        self._build_run_selector()
        self._build_fit_group()
        self._build_app_group()
        self._assemble()

        self._style_position_label()
        self._retint_icon_buttons()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        self._style_position_label()
        self._retint_icon_buttons()
        self._restyle_filter_icon()
        self._restyle_static_line_edit_icons()
        self.update()

    def _tool_button(
        self, text: str, icon_name: Optional[str] = None, checkable: bool = False
    ) -> QtWidgets.QToolButton:
        """Builds a themed toolbutton, matching every other button in the
        bar (see module docstring re: icon tinting)."""
        btn = QtWidgets.QToolButton()
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        btn.setText(text)
        btn.setCheckable(checkable)
        if icon_name:
            self._icon_buttons.append((btn, icon_name))
        return btn

    def _retint_icon_buttons(self) -> None:
        tok = ThemeManager.instance().tokens()
        color = QtGui.QColor(*tok["flat_text"])
        for btn, icon_name in self._icon_buttons:
            # QIcon never upscales past its registered pixmap's own
            # resolution (it deliberately avoids the blur that would cause)
            # - rendering smaller than the toolbar's own iconSize (30px
            # tall, matching ControlsUI's CtrlToolBar) left these looking
            # shrunk inside their slot rather than filling it the way the
            # original untinted (natively-sized) icons did.
            btn.setIcon(tinted_icon(_icon_path(icon_name), color, 30))

    def _build_run_selector(self) -> None:
        self.run_caption = SectionHeader("Run")

        # Saved-state pill - folded up beside the RUN caption per the 2a
        # redesign (was a separate row beneath cBox_Runs). UIAnalyze drives
        # its actual color/text (blank/unsaved/saved/error) and wires its
        # click behavior (jump to step 1); this class only builds/positions
        # it, and keeps the same `saved_state_widget` container so that
        # wiring (mousePressEvent) keeps working unchanged.
        self.saved_state_dot = SavedStateDot()
        self.saved_state_label = QtWidgets.QLabel("Loaded & saved")
        self.saved_state_widget = QtWidgets.QWidget()
        self.saved_state_widget.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        saved_state_layout = QtWidgets.QHBoxLayout(self.saved_state_widget)
        saved_state_layout.setContentsMargins(0, 0, 0, 0)
        saved_state_layout.setSpacing(6)
        saved_state_layout.addWidget(self.saved_state_dot)
        saved_state_layout.addWidget(self.saved_state_label)

        caption_row = QtWidgets.QHBoxLayout()
        caption_row.setContentsMargins(0, 0, 0, 0)
        caption_row.setSpacing(10)
        caption_row.addWidget(self.run_caption)
        caption_row.addWidget(self.saved_state_widget)
        caption_row.addStretch(1)

        # Searchable run field: an editable AnimatedComboBox with a
        # substring-filtering QCompleter sharing its own item model, so
        # every existing cBox_Runs call site (addItems/clear/currentText/
        # findText/currentIndexChanged/activated/...) keeps working
        # unchanged - only typed input gains live filtering.
        self.cBox_Runs = AnimatedComboBox(
            icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "down-chevron.svg")
        )
        self.cBox_Runs.setFixedHeight(30)
        # Fixed, not content-driven: _refresh_cbox_runs used to call
        # setFixedWidth(sizeHint()) on every rebuild, which made the whole
        # bar visibly jump/reflow depending on which run happened to be
        # widest/first/selected at that moment.
        self.cBox_Runs.setFixedWidth(260)
        self.cBox_Runs.setEditable(True)
        self.cBox_Runs.setInsertPolicy(QtWidgets.QComboBox.NoInsert)

        completer = QtWidgets.QCompleter(self.cBox_Runs.model(), self.cBox_Runs)
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchContains)
        completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.cBox_Runs.setCompleter(completer)
        # Otherwise the type-to-filter suggestion list falls back to Qt's
        # bare default popup chrome, which reads as a different dropdown
        # style from cBox_Runs's own rounded/flat-token-driven popup.
        self.cBox_Runs.style_completer_popup(completer)

        line_edit = self.cBox_Runs.lineEdit()
        line_edit.setPlaceholderText("Find run…")
        line_edit.setFrame(False)
        # Background/border/color/selection colors are all handled by
        # AnimatedComboBox itself (_apply_text_qss, re-run on every theme
        # change/setEditable) - setting a stylesheet here too would win
        # (last-applied wins) and silently drop back to unstyled default
        # (near-black, invisible in dark mode) text the next time this
        # widget's own theming runs without this line also re-running.

        self.search_action = line_edit.addAction(QtGui.QIcon(), QtWidgets.QLineEdit.LeadingPosition)
        # Quick-clear "x" - only shown once a filter is actually active (see
        # set_filter_active), sitting just before the filter icon so a
        # filter can be cleared right from the field without opening the
        # popover. UIAnalyze.setup_ui connects its `triggered` straight to
        # the same clear handler the popover's own "Clear" link uses.
        self.clear_filter_action = line_edit.addAction(
            QtGui.QIcon(), QtWidgets.QLineEdit.TrailingPosition
        )
        self.clear_filter_action.setToolTip("Clear filter")
        self.clear_filter_action.setVisible(False)
        self._restyle_static_line_edit_icons()  # sets the two icons above from theme tokens

        # "▾ filters" affordance embedded in the field - UIAnalyze.setup_ui
        # connects `filter_action.triggered` to open the filter popover
        # (this class only builds it, per the module's wrap-don't-own rule).
        # Its icon switches to an accent tint (see set_filter_active) so an
        # active filter is visible without opening the popover.
        self.filter_action = line_edit.addAction(
            QtGui.QIcon(), QtWidgets.QLineEdit.TrailingPosition
        )
        self.filter_action.setToolTip("Filter runs…")
        self._filter_active = False
        self._restyle_filter_icon()

        # Run Info gets its own CtrlToolBar-named toolbar (matching
        # tBtn_Predict/tool_Modify/tool_Analyze's own wrapping toolbars)
        # rather than sitting bare in this QHBoxLayout - objectName
        # "CtrlToolBar" is what actually picks up the app-wide QSS that
        # gives every other button in this bar its chrome, so without it
        # Run Info would render unstyled next to a themed search field.
        self.tBtn_Info = self._tool_button("Run Info", "info-circle.svg")
        self.run_info_bar = QtWidgets.QToolBar()
        self.run_info_bar.setObjectName("CtrlToolBar")
        self.run_info_bar.setIconSize(QtCore.QSize(50, 30))
        self.run_info_bar.addWidget(self.tBtn_Info)

        # AlignVCenter on both: cBox_Runs (a fixed 30px field) and
        # run_info_bar (taller - icon-over-label) would otherwise be
        # stretched to the row's full height by the layout and read as
        # top-aligned relative to each other.
        field_row = QtWidgets.QHBoxLayout()
        field_row.setContentsMargins(0, 0, 0, 0)
        field_row.setSpacing(8)
        field_row.addWidget(self.cBox_Runs, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        field_row.addWidget(self.run_info_bar, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        self.run_zone = QtWidgets.QVBoxLayout()
        self.run_zone.setContentsMargins(0, 0, 0, 0)
        # Tight caption-to-content gap - keeps the caption line from adding
        # much more than its own text height to the bar's overall height.
        self.run_zone.setSpacing(1)
        self.run_zone.addLayout(caption_row)
        self.run_zone.addLayout(field_row)

        # UIAnalyze/MainWindow track the current run's identity via this
        # label's text (see e.g. MainWindow.set_captured_data and
        # UIAnalyze._current_run) instead of a plain attribute, so it has to
        # keep existing even though the task bar no longer displays it.
        self.text_Created = QtWidgets.QLabel("[NONE]", self)
        self.text_Created.hide()

    def set_filter_active(self, active: bool) -> None:
        """Reflects whether any run filter is currently applied.

        Tints the filter icon accent-color instead of muted, and reveals
        the inline quick-clear "x" action beside it, so an active filter is
        visible - and clearable - right from the search field without
        opening the popover. UIAnalyze calls this after every filter
        change (device/date/new-only/sort/clear) and once at startup.
        """
        self._filter_active = active
        self.clear_filter_action.setVisible(active)
        self._restyle_filter_icon()

    def _restyle_filter_icon(self) -> None:
        tok = ThemeManager.instance().tokens()
        color = QtGui.QColor(
            *(tok["flat_accent"] if self._filter_active else tok["flat_text_muted"])
        )
        self.filter_action.setIcon(tinted_icon(_icon_path("filter.svg"), color, 18))

    def _restyle_static_line_edit_icons(self) -> None:
        """Retints the leading search icon and the quick-clear "x" - both
        always-muted, unlike the filter icon's active/inactive states."""
        tok = ThemeManager.instance().tokens()
        color = QtGui.QColor(*tok["flat_text_muted"])
        self.search_action.setIcon(tinted_icon(_icon_path("search.svg"), color, 16))
        self.clear_filter_action.setIcon(tinted_icon(_icon_path("clear.svg"), color, 14))

    def _build_fit_group(self) -> None:
        # The Load button was removed - loading now happens by picking a run
        # from cBox_Runs above (auto-loads on selection) or, when no run is
        # loaded yet, via the Signal Overview plot's own placeholder card
        # (see UIAnalyze._show_no_run_overlay), which owns "Load from
        # folder..." instead.
        self.tBtn_Predict = self._tool_button("Auto-Fit", "stars.svg")

        # Back/Next - same _tool_button() helper (and so the same chrome)
        # as Auto-Fit/Modify/Analyze, now with the dedicated previous/next
        # icon files instead of a native arrow glyph. position_label is a
        # plain, unstyled-as-a-button label sitting between them - not a
        # separate composite control.
        self.tool_Back = self._tool_button("Back", "previous.svg")
        self.tool_Next = self._tool_button("Next", "next.svg")

        self.position_label = QtWidgets.QLabel("1 / 6")
        self.position_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.tool_Modify = self._tool_button("Modify", "modify.svg", checkable=True)
        self.tool_Analyze = self._tool_button("Analyze", "play-circle.svg")

        self.fit_bar = QtWidgets.QToolBar()
        self.fit_bar.setObjectName("CtrlToolBar")
        self.fit_bar.setIconSize(QtCore.QSize(50, 30))
        self.fit_bar.addWidget(self.tBtn_Predict)
        self.fit_bar.addWidget(self.tool_Back)
        self.fit_bar.addWidget(self.position_label)
        self.fit_bar.addWidget(self.tool_Next)
        self.fit_bar.addWidget(self.tool_Modify)
        self.fit_bar.addWidget(self.tool_Analyze)

        self.fit_zone = QtWidgets.QVBoxLayout()
        self.fit_zone.setContentsMargins(0, 0, 0, 0)
        self.fit_zone.setSpacing(1)
        self.fit_zone.addWidget(SectionHeader("Fit & Analyze"))
        self.fit_zone.addWidget(self.fit_bar)

    def _style_position_label(self) -> None:
        tok = ThemeManager.instance().tokens()
        self.position_label.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_text'])}; "
            f"font-family: '{FONT_SANS_SEMIBOLD}'; font-size: 11.5px; "
            "background: transparent; border: none; }"
        )

    def _build_app_group(self) -> None:
        self.tool_Advanced = self._tool_button("Advanced", "gear.svg", checkable=True)
        self.tool_Cancel = self._tool_button("Close", "cancel.svg")
        self.tool_User = self._tool_button("Anonymous", "user-circle.svg", checkable=True)
        # Starts disabled; UIAnalyze.setup_ui/check_user_info refresh this to
        # the real signed-in state (mirrors UIControls.refresh_user_button_state).
        self.tool_User.setEnabled(False)

        self.app_bar = QtWidgets.QToolBar()
        self.app_bar.setObjectName("CtrlToolBar")
        self.app_bar.setIconSize(QtCore.QSize(50, 30))
        self.app_bar.addWidget(self.tool_Advanced)
        self.app_bar.addWidget(self.tool_Cancel)
        self.app_bar.addWidget(self.tool_User)

        self.app_zone = QtWidgets.QVBoxLayout()
        self.app_zone.setContentsMargins(0, 0, 0, 0)
        self.app_zone.setSpacing(1)
        self.app_zone.addWidget(SectionHeader("App"))
        self.app_zone.addWidget(self.app_bar)

    def _assemble(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        # Tighter than ControlsUI's own toolbar row (`toolLayout.
        # setContentsMargins(8, 4, 8, 4)`) because this bar also carries a
        # caption line ControlsUI's doesn't - matching its margins on top of
        # that would stack the extra height twice. Keeping vertical margins
        # minimal here is what actually keeps the two bars close in height.
        layout.setContentsMargins(10, 1, 10, 1)
        layout.setSpacing(14)
        # All three CtrlToolBar rows (run_info_bar/fit_bar/app_bar) share
        # the same icon size/QSS, so they hint to the same height - use
        # that as the divider height rather than a guessed constant.
        toolbar_h = self.fit_bar.sizeHint().height()
        layout.addLayout(self.run_zone)
        layout.addWidget(_VDivider(toolbar_h), 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(self.fit_zone)
        layout.addStretch(1)
        layout.addWidget(_VDivider(toolbar_h), 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(self.app_zone)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        tok = ThemeManager.instance().tokens()
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        paint_flat_surface(
            self,
            radius=self._R,
            fill=QtGui.QColor(*tok["surface"]),
            border=QtGui.QColor(*tok["surface_border"]),
            painter=painter,
        )
        painter.end()
