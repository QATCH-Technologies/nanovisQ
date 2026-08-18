"""
QATCH.ui.components.controls_action_bar.py

Shared :class:`QATCH.ui.components.task_bar_base.TaskBarBase` with
:class:`AnalyzeActionBar` instead of drifting from it.

Laid out as two uncaptioned zones: Run (Next Port / Initialize / Start-Stop
/ Reset / Temp Control, plus the temp-control side panel) and App
(Advanced / Account). These zones are separated by a :class:`TaskBarDivider`, matching
:class:`AnalyzeActionBar`'s own uncaptioned zones exactly[cite: 1]. Buttons within a zone
are packed directly via :meth:`addWidget` with no :meth:`addSeparator` between
them (except the one before `tool_Initialize`), relying purely on :class:`CtrlToolBar`'s
own QSS `spacing: 2px`.

This widget only constructs and lays out buttons/labels; it does not wire
their signals. All the callbacks live on :class:`UIControls`, not here, so
:meth:`UIControls.setup_ui` connects them after construction.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-18
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components import NumberIconButton, RunControls
from QATCH.ui.components.task_bar_base import TaskBarBase
from QATCH.ui.styles.typography import make_qfont


class ControlsActionBar(TaskBarBase):
    """Provides the application's primary run and application controls.

    Combines run-management controls, temperature-control UI, and
    application-level controls into a single themed action bar. The run
    section contains port navigation, initialization, start/stop controls,
    reset, and temperature control. A collapsible temperature-control panel
    is positioned between the run and application toolbars.

    The class is responsible only for constructing and arranging the widgets.
    Signal connections, hardware interaction, and other behavioral logic are
    owned by the caller, primarily `UIControls`.

    Attributes:
        tool_NextPortRow (NumberIconButton): Button for advancing to the next
            port row. Unlike standard toolbuttons, this widget manages its
            own icon and theming.
        action_NextPortRow (QtWidgets.QAction): QAction returned by
            `run_bar.addWidget()` for `tool_NextPortRow`.
        action_NextPortSep (QtWidgets.QAction): Separator QAction immediately
            following `tool_NextPortRow`. These two actions are retained
            because `MainWindow` enables or disables them together based on
            the availability of flux-controller hardware.
        tool_Initialize (QtWidgets.QToolButton): Button used to initialize
            the device or run.
        tool_Reset (QtWidgets.QToolButton): Button used to reset the current
            run state.
        tool_TempControl (QtWidgets.QToolButton): Checkable button used to
            toggle the temperature-control panel.
        tool_Advanced (QtWidgets.QToolButton): Checkable button used to access
            advanced application controls.
        tool_User (QtWidgets.QToolButton): Checkable account button. Its
            initial enabled state is determined during construction.
        run_controls (RunControls): Composite start/stop control that manages
            its own icons and theming.
        tempController (QtWidgets.QWidget): Collapsible temperature-control
            side panel containing the temperature status banner, setpoint
            slider, and PID information.
        tempCollapsedHeight (int): Height limit applied to
            `tempController` while it is collapsed. This corresponds to
            the run toolbar's row height and is reused by
            `UIControls._animate_temp_controller` when restoring the
            collapsed state.
        tempStatusBar (QtWidgets.QLabel): Status banner displayed at the top
            of the temperature-control panel.
        lPV (QtWidgets.QLabel): Process-variable temperature readout.
        lSP (QtWidgets.QLabel): Temperature setpoint readout.
        lOP (QtWidgets.QLabel): PID controller output readout.
        tempPidInfo (QtWidgets.QFrame): Frame containing the PID header and
            PV, SP, and OP readouts.
        run_bar (CtrlToolBar): Toolbar containing the primary run controls.
            `UIControls` aliases this widget as `self.tool_bar` for
            compatibility with existing call sites.
        app_bar (CtrlToolBar): Toolbar containing application-level controls.
            `UIControls` aliases this widget as `self.tool_bar_2` for
            compatibility with existing call sites.
        run_zone (QtWidgets.QHBoxLayout): Layout pairing `run_bar` with
            `tempController` on the left side of the application's divider.

    Args:
        temp_slider (QtWidgets.QWidget): Externally owned temperature-setpoint
            slider to embed in `tempController`. The widget's behavior is
            managed by `UIControls`; this class only incorporates it into
            the temperature-control layout.
        parent (QtWidgets.QWidget | None): Optional parent widget.

    Notes:
        Temperature readout behavior is intentionally not connected here.
        The `UIControls` class owns the relevant behavior wiring, including
        the connection used to update the temperature display.
    """

    def __init__(
        self,
        temp_slider: QtWidgets.QWidget,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initializes and assembles the controls action bar.

        Args:
            temp_slider (QtWidgets.QWidget): Temperature-setpoint slider to
                embed in the temperature-control panel.
            parent (QtWidgets.QWidget | None): Optional parent widget.
        """
        super().__init__(parent)

        self._build_run_group(temp_slider)
        self._build_app_group()
        self._assemble()

    def _build_run_group(self, temp_slider: QtWidgets.QWidget) -> None:
        """Builds the run-control toolbar and temperature-control side panel.

        Creates and configures the controls used to operate a run, including
        port navigation, initialization, run controls, reset, and temperature
        control. The method also constructs the TEC temperature side panel,
        including its status indicator, temperature slider, and PID information
        display.

        The temperature-control panel is initially collapsed to zero width and
        constrained to `BUTTON_HEIGHT` so that its hidden contents do not
        increase the height of the surrounding run-control bar. The corresponding
        animation logic is responsible for temporarily lifting this height
        constraint when the panel is expanded.

        Args:
            temp_slider (QtWidgets.QWidget): Widget used to adjust the TEC
                temperature setpoint. The widget is placed below the temperature
                status banner in the temperature-control side panel.
        """
        self.tool_NextPortRow = NumberIconButton()
        self.tool_NextPortRow.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.tool_NextPortRow.setText("Next Port")
        self.tool_NextPortRow.setFixedHeight(self.BUTTON_HEIGHT)
        self.tool_NextPortRow.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

        self.tool_Initialize = self._tool_button("Initialize", "speedometer.svg")
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
        self.run_bar.addWidget(self.run_controls)
        self.run_bar.addWidget(self.tool_Reset)
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
        self.tempCollapsedHeight = self.BUTTON_HEIGHT
        self.tempController.setMaximumHeight(self.tempCollapsedHeight)

        # Run zone
        self.run_zone = QtWidgets.QHBoxLayout()
        self.run_zone.setContentsMargins(0, 0, 0, 0)
        self.run_zone.setSpacing(8)
        self.run_zone.addWidget(self.run_bar, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.run_zone.addWidget(self.tempController, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

    def _build_app_group(self) -> None:
        """Builds the application-level toolbar controls.

        Creates the toolbar buttons used to access application-wide settings and
        account functionality. The Advanced button is configured as a checkable
        control, while the Account button is initially disabled until account
        functionality becomes available.
        """
        self.tool_Advanced = self._tool_button("Advanced", "gear.svg", checkable=True)
        self.tool_User = self._tool_button("Account", "user-circle.svg", checkable=True)
        self.tool_User.setEnabled(False)

        self.app_bar = self._make_toolbar()
        self.app_bar.addWidget(self.tool_Advanced)
        self.app_bar.addWidget(self.tool_User)

    def _assemble(self) -> None:
        """Assembles the main action-bar layout.

        Creates the top-level horizontal layout for the action bar and arranges
        the run-control zone, flexible spacing, visual divider, and application
        controls. The divider height is derived from the run toolbar's size hint
        so that it remains vertically aligned with the primary run controls.
        """
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(*self.OUTER_MARGINS)
        layout.setSpacing(self.ZONE_SPACING)
        toolbar_h = self.run_bar.sizeHint().height()
        layout.addLayout(self.run_zone)
        layout.addStretch(1)
        layout.addWidget(self._make_divider(toolbar_h), 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.app_bar, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
