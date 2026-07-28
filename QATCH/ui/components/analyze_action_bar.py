"""Themed top action bar for AnalyzeUI.

Replaces AnalyzeUI's old flat #DDDDDD toolbar with a themed card matching
the PlotsUI/ControlsUI visual language. The internal toolbars use
objectName "CtrlToolBar" so they pick up the exact same app-wide QSS that
already themes ControlsUI's toolbar (see app_theme.qss) - no new QSS rules
needed.

Laid out as three captioned zones - RUN / FIT & ANALYZE / APP - separated
by hairline dividers, per the "2a" task bar redesign: the run selector is
a searchable, filterable field with the saved-state indicator folded into
the RUN caption itself, and Back/Next collapse into a single position
stepper chip rather than two separate buttons.

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
from QATCH.ui.styles.fonts import FONT_SANS, FONT_SANS_SEMIBOLD
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css
from QATCH.ui.widgets.saved_state_dot import SavedStateDot


def _icon_path(icon_name: str) -> str:
    return os.path.join(Architecture.get_path(), "QATCH", "icons", icon_name)


def _icon(icon_name: str) -> QtGui.QIcon:
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap(_icon_path(icon_name)), QtGui.QIcon.Normal)
    return icon


def _tool_button(
    text: str, icon_name: Optional[str] = None, checkable: bool = False
) -> QtWidgets.QToolButton:
    btn = QtWidgets.QToolButton()
    btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
    if icon_name:
        btn.setIcon(_icon(icon_name))
    btn.setText(text)
    btn.setCheckable(checkable)
    return btn


class _VDivider(QtWidgets.QFrame):
    """A hairline vertical divider between action-bar zones, themed from
    `flat_border` so it stays correct across light/dark switches."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setFixedWidth(1)
        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_theme()

    def _apply_theme(self) -> None:
        tok = ThemeManager.instance().tokens()
        self.setStyleSheet(f"background-color: {tok_css(tok['flat_border'])}; border: none;")


class _PositionStepper(QtWidgets.QWidget):
    """"◀ pos N / total ▶" chip - the 2a redesign's replacement for separate
    Back/Next buttons, folding both into one compact stepper.

    The arrow buttons are plain `QToolButton`s (`setArrowType`, no icon
    file) so their click/enabled behavior is indistinguishable from the
    Back/Next buttons they replace - UIAnalyze wires and enables them
    exactly as before via the public `back_btn`/`next_btn` attributes.
    """

    _RADIUS = 15.0

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setFixedHeight(30)

        self.back_btn = QtWidgets.QToolButton()
        self.back_btn.setArrowType(QtCore.Qt.LeftArrow)
        self.back_btn.setAutoRaise(True)
        self.back_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.back_btn.setFixedSize(20, 20)

        self.next_btn = QtWidgets.QToolButton()
        self.next_btn.setArrowType(QtCore.Qt.RightArrow)
        self.next_btn.setAutoRaise(True)
        self.next_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.next_btn.setFixedSize(20, 20)

        self.label = QtWidgets.QLabel("pos 1 / 6")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumWidth(58)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(2)
        layout.addWidget(self.back_btn)
        layout.addWidget(self.label)
        layout.addWidget(self.next_btn)

        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def set_position(self, current: int, total: int) -> None:
        self.label.setText(f"pos {current} / {total}")

    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_theme()
        self.update()

    def _apply_theme(self) -> None:
        tok = ThemeManager.instance().tokens()
        self.label.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_accent'])}; "
            f"font-family: '{FONT_SANS_SEMIBOLD}'; font-size: 11.5px; "
            "background: transparent; border: none; }"
        )
        arrow_qss = f"""
            QToolButton {{ background: transparent; border: none; border-radius: 10px; color: {tok_css(tok["flat_accent"])}; }}
            QToolButton:hover {{ background: {tok_css(tok["flat_accent_ring"])}; }}
            QToolButton:disabled {{ background: transparent; color: {tok_css(tok["flat_text_muted"])}; }}
        """
        self.back_btn.setStyleSheet(arrow_qss)
        self.next_btn.setStyleSheet(arrow_qss)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        tok = ThemeManager.instance().tokens()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        paint_flat_surface(
            self,
            radius=self._RADIUS,
            fill=QtGui.QColor(*tok["flat_accent_weak"]),
            border=QtGui.QColor(*tok["flat_accent_ring"]),
            painter=p,
        )
        p.end()


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
        tBtn_Predict, tBtn_Info: Auto-Fit/Run Info buttons.
        position_stepper: the "◀ pos N/6 ▶" chip that replaces separate
            Back/Next buttons; `tool_Back`/`tool_Next` are its arrow
            buttons, kept under those names so existing wiring/enable
            logic elsewhere is unaffected.
        tool_Modify, tool_Analyze, tool_Advanced, tool_Cancel, tool_User:
            the remaining action buttons (Close now lives in the APP zone
            alongside Advanced/User, not beside Back/Next).
    """

    _R = 12.0

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

        self._build_run_selector()
        self._build_fit_group()
        self._build_app_group()
        self._assemble()

        ThemeManager.instance().themeChanged.connect(self.update)

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
        self.cBox_Runs.setEditable(True)
        self.cBox_Runs.setInsertPolicy(QtWidgets.QComboBox.NoInsert)

        completer = QtWidgets.QCompleter(self.cBox_Runs.model(), self.cBox_Runs)
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchContains)
        completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.cBox_Runs.setCompleter(completer)

        line_edit = self.cBox_Runs.lineEdit()
        line_edit.setPlaceholderText("Find run…")
        line_edit.setFrame(False)
        line_edit.setStyleSheet("QLineEdit { background: transparent; border: none; }")

        self.search_action = line_edit.addAction(
            _icon("search.svg"), QtWidgets.QLineEdit.LeadingPosition
        )
        # "▾ filters" affordance embedded in the field - UIAnalyze.setup_ui
        # connects `filter_action.triggered` to open the filter popover
        # (this class only builds it, per the module's wrap-don't-own rule).
        self.filter_action = line_edit.addAction(
            _icon("filter.svg"), QtWidgets.QLineEdit.TrailingPosition
        )
        self.filter_action.setToolTip("Filter runs…")

        field_row = QtWidgets.QHBoxLayout()
        field_row.setContentsMargins(0, 0, 0, 0)
        field_row.setSpacing(8)
        field_row.addWidget(self.cBox_Runs)

        self.tBtn_Info = _tool_button("Run Info", "info-circle.svg")
        field_row.addWidget(self.tBtn_Info)

        self.run_zone = QtWidgets.QVBoxLayout()
        self.run_zone.setContentsMargins(0, 0, 0, 0)
        self.run_zone.setSpacing(6)
        self.run_zone.addLayout(caption_row)
        self.run_zone.addLayout(field_row)

        # UIAnalyze/MainWindow track the current run's identity via this
        # label's text (see e.g. MainWindow.set_captured_data and
        # UIAnalyze._current_run) instead of a plain attribute, so it has to
        # keep existing even though the task bar no longer displays it.
        self.text_Created = QtWidgets.QLabel("[NONE]", self)
        self.text_Created.hide()

    def _build_fit_group(self) -> None:
        # The Load button was removed - loading now happens by picking a run
        # from cBox_Runs above (auto-loads on selection) or, when no run is
        # loaded yet, via the Signal Overview plot's own placeholder card
        # (see UIAnalyze._show_no_run_overlay), which owns "Load from
        # folder..." instead.
        self.tBtn_Predict = _tool_button("Auto-Fit", "stars.svg")

        self.position_stepper = _PositionStepper()
        self.tool_Back = self.position_stepper.back_btn
        self.tool_Next = self.position_stepper.next_btn
        self.position_label = self.position_stepper.label

        self.tool_Modify = _tool_button("Modify", "modify.svg", checkable=True)
        self.tool_Analyze = _tool_button("Analyze", "play-circle.svg")

        self.fit_bar = QtWidgets.QToolBar()
        self.fit_bar.setObjectName("CtrlToolBar")
        self.fit_bar.setIconSize(QtCore.QSize(50, 30))
        self.fit_bar.addWidget(self.tBtn_Predict)
        self.fit_bar.addWidget(self.position_stepper)
        self.fit_bar.addWidget(self.tool_Modify)
        self.fit_bar.addWidget(self.tool_Analyze)

        self.fit_zone = QtWidgets.QVBoxLayout()
        self.fit_zone.setContentsMargins(0, 0, 0, 0)
        self.fit_zone.setSpacing(6)
        self.fit_zone.addWidget(SectionHeader("Fit & Analyze"))
        self.fit_zone.addWidget(self.fit_bar)

    def _build_app_group(self) -> None:
        self.tool_Advanced = _tool_button("Advanced", "gear.svg")
        self.tool_Cancel = _tool_button("Close", "cancel.svg")
        self.tool_User = _tool_button("Anonymous", "user-circle.svg", checkable=True)
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
        self.app_zone.setSpacing(6)
        self.app_zone.addWidget(SectionHeader("App"))
        self.app_zone.addWidget(self.app_bar)

    def _assemble(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        # Same margins as ControlsUI's own toolbar row (see
        # UIControls.setup_ui's `self.toolLayout.setContentsMargins(8, 4, 8, 4)`)
        # so the two task bars render at the same height/placement.
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(14)
        layout.addLayout(self.run_zone)
        layout.addWidget(_VDivider())
        layout.addLayout(self.fit_zone)
        layout.addStretch(1)
        layout.addWidget(_VDivider())
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
