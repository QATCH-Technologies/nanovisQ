"""Themed top task bar for ControlsUI.

Extracted from ~200 lines previously built inline in `UIControls.setup_ui`
so its chrome (icon size, toolbutton construction, icon retinting, checked-
state repaint) shares `QATCH.ui.components.task_bar_base.TaskBarBase` with
`AnalyzeActionBar` instead of drifting from it - see that module's history
for the two concrete inconsistencies this fixed (buttons here always had a
hand cursor and a checked-state repolish; `AnalyzeActionBar`'s never did
until it also moved onto `TaskBarBase`).

Laid out as two captioned zones - Run (Next Port / Initialize / Start-Stop
/ Reset / Temp Control, plus the temp-control side panel) and App
(Advanced / Account) - separated by a `TaskBarDivider`, the same
`SectionHeader`-captioned-zone language `AnalyzeActionBar` uses (including
reusing its exact "App" caption for the analogous Advanced/Account group),
for actual cross-UI visual consistency rather than just shared helpers.

This widget only constructs and lays out buttons/labels; it does not wire
their signals. All the callbacks (action_initialize, action_start, ...)
live on `UIControls`, not here, so `UIControls.setup_ui` connects them
after construction - the same "wrap, don't own" pattern
`QATCH.ui.components.analyze_action_bar.AnalyzeActionBar` uses.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components import NumberIconButton, RunControls
from QATCH.ui.components.task_bar_base import TaskBarBase, TaskBarDivider
from QATCH.ui.labels.section_label import SectionHeader
from QATCH.ui.styles.typography import make_qfont


class ControlsActionBar(TaskBarBase):
    """Next Port / Initialize / Start-Stop / Reset / Temp Control + the
    temp-control side panel, and Advanced / Account, as one themed bar.

    Public attributes (all plain Qt widgets - the caller wires their
    signals and owns their behavior):
        tool_NextPortRow: `NumberIconButton` - manages its own icon/theming,
            unlike the plain toolbuttons built via `_tool_button()`.
        action_NextPortRow, action_NextPortSep: the `QAction`s
            `QToolBar.addWidget()`/`addSeparator()` return for
            `tool_NextPortRow` - kept since callers already reference them.
        tool_Initialize, tool_Reset, tool_TempControl, tool_Advanced,
            tool_User: plain toolbuttons built via `_tool_button()`.
        run_controls: the `RunControls` composite (Start/Stop) - manages
            its own icon/theming, like `tool_NextPortRow`.
        tempController: the collapsible temp-control side panel (status
            banner + PID readout), embedded between `run_bar` and `app_bar`.
        tempCollapsedHeight: `run_bar`'s own toolbar row height - the cap
            `tempController.maximumHeight()` is pinned to while collapsed
            (see `_build_run_group`). `UIControls._animate_temp_controller`
            reuses this to restore the cap once a collapse finishes.
        tempStatusBar, lPV, lSP, lOP, tempPidInfo: widgets inside
            `tempController` - `UIControls` reads/writes these directly
            (e.g. `_update_temp_display`).
        run_bar, app_bar: the two `CtrlToolBar` rows themselves. `UIControls`
            aliases these as `self.tool_bar`/`self.tool_bar_2` (its
            pre-extraction names - `self.tool_bar_2.iconSize()` is read
            elsewhere) rather than renaming every existing call site.
        run_zone, app_zone: the captioned (`SectionHeader` + row) `QVBoxLayout`
            groups `_assemble()` lays out either side of the `TaskBarDivider`.

    Args:
        temp_slider: The (externally owned) temperature-setpoint slider to
            embed in `tempController` - built and behaviorally owned by
            `UIControls`, this class only lays it out. `UIControls`'s
            temperature-readout label isn't needed here - its only tie to
            this bar (`text_updated.connect(self._update_temp_display)`) is
            behavior wiring, connected by `UIControls.setup_ui` itself.
    """

    def __init__(
        self,
        temp_slider: QtWidgets.QWidget,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._build_run_group(temp_slider)
        self._build_app_group()
        self._assemble()

    def _build_run_group(self, temp_slider: QtWidgets.QWidget) -> None:
        self.tool_NextPortRow = NumberIconButton()
        self.tool_NextPortRow.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_NextPortRow.setText("Next Port")
        self.tool_NextPortRow.setFixedHeight(self.BUTTON_HEIGHT)
        self.tool_NextPortRow.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

        self.tool_Initialize = self._tool_button("Initialize", "speedometer.svg")

        # RunControls composite widget - manages its own icon/theming.
        self.run_controls = RunControls()
        self.run_controls.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.run_controls.setEnabled(False)

        self.tool_Reset = self._tool_button("Reset", "reset.svg")

        self.tool_TempControl = self._tool_button("Temp Control", "temperature-control.svg")
        self.tool_TempControl.setCheckable(True)

        self.run_bar = self._make_toolbar()
        self.action_NextPortRow = self.run_bar.addWidget(self.tool_NextPortRow)
        self.action_NextPortSep = self.run_bar.addSeparator()
        self.run_bar.addWidget(self.tool_Initialize)
        self.run_bar.addSeparator()
        self.run_bar.addWidget(self.run_controls)
        self.run_bar.addSeparator()
        self.run_bar.addWidget(self.tool_Reset)
        self.run_bar.addSeparator()
        self.run_bar.addWidget(self.tool_TempControl)

        # TEC temperature side-panel
        self.tempController = QtWidgets.QWidget()
        self.tempController.setObjectName("tempController")
        self.tempController.setMinimumWidth(0)
        self.tempController.setMaximumWidth(0)  # collapsed until activated

        self.tempStatusBar = QtWidgets.QLabel("Offline")
        self.tempStatusBar.setObjectName("tempStatusBanner")
        self.tempStatusBar.setFixedHeight(18)
        self.tempStatusBar.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        _status_font = QtGui.QFont()
        _status_font.setPointSize(7)
        _status_font.setBold(True)
        self.tempStatusBar.setFont(_status_font)

        # Status (top) above slider (bottom)
        left_col = QtWidgets.QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(4)
        left_col.addWidget(self.tempStatusBar)
        left_col.addWidget(temp_slider)

        # PID Info panel
        value_font = make_qfont(
            families=["Consolas", "Courier New"], style_hint=QtGui.QFont.Monospace, point_size=7
        )
        self.lPV = QtWidgets.QLabel("PV  --.--°C")
        self.lSP = QtWidgets.QLabel("SP  --.--°C")
        self.lOP = QtWidgets.QLabel("OP  ----")
        for lbl in (self.lPV, self.lSP, self.lOP):
            lbl.setObjectName("TempPidValue")
            lbl.setFont(value_font)
            lbl.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
            )

        self.tempPidInfo = QtWidgets.QFrame()
        self.tempPidInfo.setObjectName("tempPidInfo")
        pid_layout = QtWidgets.QVBoxLayout(self.tempPidInfo)
        pid_layout.setContentsMargins(8, 4, 8, 4)
        pid_layout.setSpacing(1)

        pid_header = QtWidgets.QLabel("PID INFO")
        pid_header.setObjectName("tempPidHeader")
        pid_header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        pid_layout.addWidget(pid_header)
        pid_layout.addWidget(self.lPV)
        pid_layout.addWidget(self.lSP)
        pid_layout.addWidget(self.lOP)

        # Assemble panel
        temp_layout = QtWidgets.QHBoxLayout()
        temp_layout.setContentsMargins(8, 6, 8, 6)
        temp_layout.setSpacing(8)
        temp_layout.addLayout(left_col, 1)
        temp_layout.addWidget(self.tempPidInfo, 0)
        self.tempController.setLayout(temp_layout)

        # Collapsed by default (setMaximumWidth(0) above) - but a
        # QWidgetItem's sizeHint().height() is governed by maximumHeight,
        # not maximumWidth, so without this cap tempController's own
        # natural content height (status banner + slider + tempPidInfo,
        # taller than BUTTON_HEIGHT) still leaks into run_row's
        # QHBoxLayout height computation via AlignVCenter even while the
        # panel is visibly 0px wide - inflating this whole bar above
        # AnalyzeActionBar's and knocking run_bar's buttons out of
        # vertical alignment with app_bar's. UIControls
        # ._animate_temp_controller lifts this cap in lockstep with the
        # width animation whenever the panel is actually opened, and
        # restores it once a collapse animation finishes.
        self.tempCollapsedHeight = self.run_bar.sizeHint().height()
        self.tempController.setMaximumHeight(self.tempCollapsedHeight)

        # Run zone: caption above a row pairing run_bar with the temp panel
        # (AlignVCenter on both, matching how AnalyzeActionBar's RUN zone
        # nests run_info_bar beside cBox_Runs) - temp control is a run-time
        # control, not its own category, so it doesn't get its own caption.
        run_row = QtWidgets.QHBoxLayout()
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.setSpacing(8)
        run_row.addWidget(self.run_bar, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        run_row.addWidget(self.tempController, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        self.run_zone = QtWidgets.QVBoxLayout()
        self.run_zone.setContentsMargins(0, 0, 0, 0)
        self.run_zone.setSpacing(1)
        self.run_zone.addWidget(SectionHeader("Run"))
        self.run_zone.addLayout(run_row)

    def _build_app_group(self) -> None:
        self.tool_Advanced = self._tool_button("Advanced", "gear.svg", checkable=True)
        self.tool_User = self._tool_button("Account", "user-circle.svg", checkable=True)
        self.tool_User.setEnabled(False)

        self.app_bar = self._make_toolbar()
        self.app_bar.addWidget(self.tool_Advanced)
        self.app_bar.addSeparator()
        self.app_bar.addWidget(self.tool_User)

        # Same "App" caption AnalyzeActionBar uses for its analogous
        # Advanced/Account zone - actual cross-UI vocabulary consistency.
        self.app_zone = QtWidgets.QVBoxLayout()
        self.app_zone.setContentsMargins(0, 0, 0, 0)
        self.app_zone.setSpacing(1)
        self.app_zone.addWidget(SectionHeader("App"))
        self.app_zone.addWidget(self.app_bar)

    def _assemble(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(*self.OUTER_MARGINS)
        layout.setSpacing(self.ZONE_SPACING)
        toolbar_h = self.run_bar.sizeHint().height()
        layout.addLayout(self.run_zone)
        layout.addStretch(1)
        layout.addWidget(TaskBarDivider(toolbar_h), 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(self.app_zone)
