"""Themed top action bar for AnalyzeUI.

Replaces AnalyzeUI's old flat #DDDDDD toolbar with a themed card matching
the PlotsUI/ControlsUI visual language. The internal toolbars use
objectName "CtrlToolBar" so they pick up the exact same app-wide QSS that
already themes ControlsUI's toolbar (see app_theme.qss) - no new QSS rules
needed.

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
from QATCH.ui.labels.section_label import SectionHeader
from QATCH.ui.styles.theme_manager import ThemeManager


def _icon(icon_name: str) -> QtGui.QIcon:
    path = os.path.join(Architecture.get_path(), "QATCH", "icons", icon_name)
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap(path), QtGui.QIcon.Normal)
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


class AnalyzeActionBar(QtWidgets.QWidget):
    """Run selector + Auto-Fit/Run Info + Back/Next/Modify/Analyze +
    Advanced/User, as one themed card.

    Public attributes (all plain Qt widgets - the caller wires their
    signals and owns their behavior):
        active_run_header, cBox_Runs: the run selector.
        text_Created: hidden internal-state label, not shown in the bar -
            see `_build_run_selector`.
        tBtn_Predict, tBtn_Info: Auto-Fit/Run Info buttons.
        tool_Cancel, tool_Back, tool_Next, tool_Modify, tool_Analyze,
        tool_Advanced, tool_User: the navigation/settings buttons.
    """

    _R = 12.0

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

        self._build_run_selector()
        self._build_load_group()
        self._build_nav_group()
        self._assemble()

        ThemeManager.instance().themeChanged.connect(self.update)

    def _build_run_selector(self) -> None:
        self.active_run_header = SectionHeader("Active Run")
        self.cBox_Runs = AnimatedComboBox(
            icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "down-chevron.svg")
        )
        self.cBox_Runs.setFixedHeight(20)

        self.run_row = QtWidgets.QHBoxLayout()
        self.run_row.setContentsMargins(0, 0, 0, 0)
        self.run_row.setSpacing(8)
        self.run_row.addWidget(self.active_run_header)
        self.run_row.addWidget(self.cBox_Runs)

        # UIAnalyze/MainWindow track the current run's identity via this
        # label's text (see e.g. MainWindow.set_captured_data and
        # UIAnalyze._current_run) instead of a plain attribute, so it has to
        # keep existing even though the task bar no longer displays it.
        self.text_Created = QtWidgets.QLabel("[NONE]", self)
        self.text_Created.hide()

    def _build_load_group(self) -> None:
        # The Load button was removed - loading now happens by picking a run
        # from cBox_Runs above (auto-loads on selection) or, when no run is
        # loaded yet, via the Signal Overview plot's own placeholder card
        # (see UIAnalyze._show_no_run_overlay), which owns "Load from
        # folder..." instead.
        self.tBtn_Predict = _tool_button("Auto-Fit", "stars.svg")
        self.tBtn_Info = _tool_button("Run Info", "info-circle.svg")

        self.load_bar = QtWidgets.QToolBar()
        self.load_bar.setObjectName("CtrlToolBar")
        self.load_bar.setIconSize(QtCore.QSize(50, 30))
        self.load_bar.addWidget(self.tBtn_Predict)
        self.load_bar.addWidget(self.tBtn_Info)

    def _build_nav_group(self) -> None:
        self.tool_Cancel = _tool_button("Close", "cancel.svg")
        self.tool_Back = _tool_button("Back")
        self.tool_Back.setArrowType(QtCore.Qt.LeftArrow)
        self.tool_Next = _tool_button("Next")
        self.tool_Next.setArrowType(QtCore.Qt.RightArrow)
        self.tool_Modify = _tool_button("Modify", "modify.svg", checkable=True)
        self.tool_Analyze = _tool_button("Analyze", "play-circle.svg")
        self.tool_Advanced = _tool_button("Advanced", "gear.svg")
        self.tool_User = _tool_button("Anonymous", "user-circle.svg", checkable=True)
        # Starts disabled; UIAnalyze.setup_ui/check_user_info refresh this to
        # the real signed-in state (mirrors UIControls.refresh_user_button_state).
        self.tool_User.setEnabled(False)

        self.nav_bar = QtWidgets.QToolBar()
        self.nav_bar.setObjectName("CtrlToolBar")
        self.nav_bar.setIconSize(QtCore.QSize(50, 30))
        self.nav_bar.addWidget(self.tool_Cancel)
        self.nav_bar.addSeparator()
        self.nav_bar.addWidget(self.tool_Back)
        self.nav_bar.addWidget(self.tool_Next)
        self.nav_bar.addSeparator()
        self.nav_bar.addWidget(self.tool_Modify)
        self.nav_bar.addWidget(self.tool_Analyze)
        self.nav_bar.addSeparator()
        self.nav_bar.addWidget(self.tool_Advanced)
        self.nav_bar.addSeparator()
        self.nav_bar.addWidget(self.tool_User)

    def _assemble(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        # Same margins as ControlsUI's own toolbar row (see
        # UIControls.setup_ui's `self.toolBar.setContentsMargins(8, 4, 8, 4)`)
        # so the two task bars render at the same height/placement.
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        layout.addLayout(self.run_row)
        layout.addWidget(self.load_bar)
        layout.addStretch(1)
        layout.addWidget(self.nav_bar)

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
