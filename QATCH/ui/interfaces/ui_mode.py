"""QATCH.ui.interfaces.ui_mode.py

This module provides the structural foundation for the application's
mode-based interface. It manages the layout, navigation, and state
transitions between various functional views such as 'Run', 'Analyze',
and tool modules.

Author(s)
    Alexander Ross  (alexander.ross@qatchtech.com)
    Paul MacNichol  (paul.macnichol@qatchtech.com)

Date:
    2026-07-01
"""

import datetime
import os
from typing import TYPE_CHECKING, Any, Callable, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.ui.widgets.update_status_badge import UpdateStatusIcon
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles
from QATCH.core.constants import Constants, UserRoles
from QATCH.tools.donnan_gibbs_calculator import DonnanCalculatorModule
from QATCH.tools.injection_force_calculator import InjectionForceCalculatorModule
from QATCH.ui.dialogs.pop_up_dialog import PopUp
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css
from QATCH.ui.widgets.floating_menu_widget import FloatingMenuWidget

if TYPE_CHECKING:
    from QATCH.ui.main_window import MainWindow
    from QATCH.ui.windows import ModeWindow


class UIMode:
    """
    Manages the UI layout, mode switching logic, and view transitions for the
    application's main interface.

    This class handles the initialization of the left-hand navigation menu,
    the central stacked view (controlled by mode-specific widgets), and the
    integrated logging panel. It maintains references to various child modules
    and handles state transitions between 'Run', 'Analyze', 'VisQ.AI', and
    calculator modes.
    """

    @staticmethod
    def _click(handler: Callable[..., Any]):
        """Adapts a mode-switch method to the `QLabel.mousePressEvent` signature.

        The mode methods return `bool | None` for programmatic callers but the
        event-handler slot must return `None`.  This wrapper discards the return
        value and uses a position-only `ev` parameter.
        """

        def _h(ev: Optional[QtGui.QMouseEvent] = None) -> None:
            handler(ev)

        return _h

    def setup_ui(self, mode_window: "ModeWindow", parent: "MainWindow") -> None:
        """
        Constructs and configures the UI elements, layouts, and signal connections
        for the mode-based navigation system.

        This method builds the sidebar menu (including logos and mode labels),
        initializes the various sub-modules (calculators, loggers, and view frames),
        and sets up the splitter-based layout for the log panel.

        Args:
            mode_window (ModeWIndow): The main container window being configured.
            parent (MainWindow): The main application window instance containing global
                    data and sub-windows.
        """
        self.parent = parent
        mode_window.setObjectName("modeWindow")
        mode_window.setMinimumSize(QtCore.QSize(1331, 711))
        self.centralwidget = QtWidgets.QWidget(mode_window)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        layout_h = QtWidgets.QHBoxLayout()
        layout_h.setSpacing(10)

        # Run / Analyze / VisQ.AI
        modewidget = QtWidgets.QWidget()
        modewidget.setObjectName("modeWidgetContainer")
        modelayout = QtWidgets.QVBoxLayout()
        modelayout.setContentsMargins(0, 0, 0, 0)
        modelayout.setSpacing(0)

        # Sliding Highlight Widget
        self.active_highlight = QtWidgets.QWidget(modewidget)
        self.active_highlight.setObjectName("activeHighlight")
        self.active_highlight.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.active_highlight.hide()

        # Logo
        icon_path = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "high-res-nanovisq-logo-no-bg.png"
        )
        original_pixmap = QtGui.QPixmap(icon_path).scaledToWidth(
            100, QtCore.Qt.TransformationMode.SmoothTransformation
        )
        rounded_pixmap = QtGui.QPixmap(original_pixmap.size())
        rounded_pixmap.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(rounded_pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        path = QtGui.QPainterPath()
        path.addRoundedRect(0, 0, original_pixmap.width(), original_pixmap.height(), 12, 12)

        painter.setClipPath(path)
        painter.drawPixmap(0, 0, original_pixmap)
        painter.end()
        self.logolabel = QtWidgets.QLabel()
        self.logolabel.setObjectName("nanovisqLogo")
        self.logolabel.setPixmap(rounded_pixmap)
        self.logolabel.resize(rounded_pixmap.width(), rounded_pixmap.height())

        # Mode Header
        self.mode_mode = QtWidgets.QLabel("<b>MODE</b>")
        self.mode_mode.setObjectName("menuSectionHeader")

        # Run Mode
        self.mode_run = QtWidgets.QLabel("Run")
        self.mode_run.setObjectName("menuItem")
        self.mode_run.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.mode_run.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.mode_run.mousePressEvent = self._click(self._set_run_mode)

        # Analyze Mode
        self.mode_analyze = QtWidgets.QLabel("Analyze")
        self.mode_analyze.setObjectName("menuItem")
        self.mode_analyze.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.mode_analyze.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.mode_analyze.mousePressEvent = self._click(self._set_analyze_mode)

        # Tools Header
        self.mode_tools = QtWidgets.QLabel("<b>TOOLS</b>")
        self.mode_tools.setObjectName("menuSectionHeader")

        # VisQ.AI Tool
        self.mode_learn = QtWidgets.QLabel("VisQ.AI™")
        self.mode_learn.setObjectName("menuItem")
        self.mode_learn.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.mode_learn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.mode_learn.mousePressEvent = self._click(self._set_learn_mode)

        # Donnan Calculator Tool
        self.mode_donnan = QtWidgets.QLabel("Donnan-Gibbs Calculator")
        self.mode_donnan.setObjectName("menuItem")
        self.mode_donnan.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.mode_donnan.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.mode_donnan.setWordWrap(True)
        self.mode_donnan.mousePressEvent = self._click(self._set_donnan_mode)

        # Injection Force Calculator Tool
        self.mode_injection = QtWidgets.QLabel("Injection Force Calculator")
        self.mode_injection.setObjectName("menuItem")
        self.mode_injection.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.mode_injection.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.mode_injection.setWordWrap(True)
        self.mode_injection.mousePressEvent = self._click(self._set_injection_mode)

        # Text antialiasing
        smooth_font = QtGui.QFont()
        smooth_font.setStyleStrategy(QtGui.QFont.PreferAntialias | QtGui.QFont.PreferQuality)
        self.mode_mode.setFont(smooth_font)
        self.mode_run.setFont(smooth_font)
        self.mode_analyze.setFont(smooth_font)
        self.mode_tools.setFont(smooth_font)
        self.mode_learn.setFont(smooth_font)
        self.mode_donnan.setFont(smooth_font)
        self.mode_injection.setFont(smooth_font)

        # Initialize dynamic properties
        self.mode_run.setProperty("active", "false")
        self.mode_analyze.setProperty("active", "false")
        self.mode_learn.setProperty("active", "false")
        self.mode_donnan.setProperty("active", "false")
        self.mode_injection.setProperty("active", "false")

        # Add to Layout
        modelayout.addWidget(self.logolabel)
        modelayout.addWidget(self.mode_mode)
        modelayout.addWidget(self.mode_run)
        modelayout.addWidget(self.mode_analyze)
        modelayout.addWidget(self.mode_tools)
        if Constants.show_visQ_in_R_builds:
            modelayout.addWidget(self.mode_learn)
        modelayout.addWidget(self.mode_donnan)
        modelayout.addWidget(self.mode_injection)
        modelayout.addStretch()

        modewidget.setLayout(modelayout)
        self.modemenu = QtWidgets.QScrollArea()
        self.modemenu.setObjectName("modeMenuScrollArea")
        self.modemenu.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.modemenu.setLineWidth(0)
        self.modemenu.setMidLineWidth(0)
        self.modemenu.setWidgetResizable(True)
        self.modemenu.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.modemenu.setMinimumSize(QtCore.QSize(100, 700))
        self.modemenu.setWidget(modewidget)

        # Create floating menu widget for VisQ.AI Toolkit
        self.floating_widget = FloatingMenuWidget(self)
        self.floating_widget.addItems(self.parent.visq_window.getToolNames())

        # run mode view frame: Controls and Plots
        runwidget = QtWidgets.QWidget()
        runlayout = QtWidgets.QVBoxLayout()
        runlayout.setContentsMargins(0, 0, 0, 0)
        runlayout.setSpacing(0)
        runlayout.addWidget(parent.controls_window.ui.centralwidget, 1)
        runlayout.addWidget(parent.plots_window.ui.centralwidget, 255)
        runwidget.setLayout(runlayout)
        self.runview = QtWidgets.QScrollArea()
        self.runview.setObjectName("runview")
        self.runview.setFrameShape(QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Plain)
        self.runview.setLineWidth(0)
        self.runview.setMidLineWidth(0)
        self.runview.setWidgetResizable(True)
        self.runview.setWidget(runwidget)
        self.runview.setMinimumSize(QtCore.QSize(1000, 122))

        # analyze mode view frame: Analyze
        self.analyze = QtWidgets.QScrollArea()
        self.analyze.setObjectName("analyze")
        self.analyze.setFrameShape(QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Plain)
        self.analyze.setLineWidth(0)
        self.analyze.setMidLineWidth(0)
        self.analyze.setWidgetResizable(True)
        self.analyze.setWidget(parent.analyze_process)
        self.analyze.setMinimumSize(QtCore.QSize(1000, 122))

        # learn mode view frame: VisQ.AI
        self.learn_ui = QtWidgets.QScrollArea()
        self.learn_ui.setObjectName("learn_ui")
        self.learn_ui.setFrameShape(QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Plain)
        self.learn_ui.setLineWidth(0)
        self.learn_ui.setMidLineWidth(0)
        self.learn_ui.setWidgetResizable(True)
        self.learn_ui.setWidget(parent.visq_window)
        self.learn_ui.setMinimumSize(QtCore.QSize(1000, 122))

        self.donnan_calc_module = DonnanCalculatorModule()  # Instantiate the module
        self.donnan_ui = QtWidgets.QScrollArea()
        self.donnan_ui.setObjectName("donnan_ui")
        self.donnan_ui.setFrameShape(QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Plain)
        self.donnan_ui.setLineWidth(0)
        self.donnan_ui.setMidLineWidth(0)
        self.donnan_ui.setWidgetResizable(True)
        self.donnan_ui.setWidget(self.donnan_calc_module)
        self.donnan_ui.setMinimumSize(QtCore.QSize(1000, 122))

        # injection mode view frame: placeholder
        self.injection_calc_module = InjectionForceCalculatorModule()  # Instantiate the module
        self.injection_ui = QtWidgets.QScrollArea()
        self.injection_ui.setObjectName("injection_ui")
        self.injection_ui.setFrameShape(QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Plain)
        self.injection_ui.setLineWidth(0)
        self.injection_ui.setMidLineWidth(0)
        self.injection_ui.setWidgetResizable(True)
        self.injection_ui.setWidget(self.injection_calc_module)
        self.injection_ui.setMinimumSize(QtCore.QSize(1000, 122))
        # log view frame: Logger
        self.logview = QtWidgets.QScrollArea()
        self.logview.setObjectName("logview")
        self.logview.setFrameShape(QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Plain)
        self.logview.setLineWidth(0)
        self.logview.setMidLineWidth(0)
        self.logview.setWidgetResizable(True)
        self.logview.setWidget(parent.logger_window.ui.centralwidget)
        # Min height relaxed to 0 so the pane can slide/drag closed.
        self.logview.setMinimumSize(QtCore.QSize(1000, 0))
        self.logview.viewport().setAutoFillBackground(False)  # type: ignore

        layout_h.addWidget(self.modemenu, 1)
        layout_v = QtWidgets.QVBoxLayout()
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        # NOTE: this widget must not be changed at load time (or else it disappears)
        self.splitter.addWidget(self.runview)
        layout_v.addWidget(self.splitter)
        layout_h.addLayout(layout_v, 255)

        # Main area wrapped so it can sit in the outer splitter.
        self.main_area = QtWidgets.QWidget()
        self.main_area.setObjectName("mainArea")
        self.main_area.setLayout(layout_h)
        parent.login_window.ui.centralwidget.attach_to(self.main_area)

        # Controls/footer bar
        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")
        self._log_chevron_down = QtGui.QIcon(os.path.join(icons_dir, "down-chevron.svg"))
        self._log_chevron_up = QtGui.QIcon(os.path.join(icons_dir, "up-chevron.svg"))
        dimmed_tint = QtGui.QColor(40, 48, 56, 235)
        self._log_chevron_down_light = self._tinted_icon(
            os.path.join(icons_dir, "down-chevron.svg"), dimmed_tint
        )
        self._log_chevron_up_light = self._tinted_icon(
            os.path.join(icons_dir, "up-chevron.svg"), dimmed_tint
        )

        self.log_toggle_bar = QtWidgets.QWidget()
        self.log_toggle_bar.setObjectName("logToggleBar")
        self.log_toggle_bar.setFixedHeight(20)
        toggle_layout = QtWidgets.QHBoxLayout(self.log_toggle_bar)
        toggle_layout.setContentsMargins(8, 0, 6, 0)
        toggle_layout.setSpacing(6)

        # Footer copyright text
        current_year = datetime.date.today().year
        footer_text = f"\u00a9 {current_year} QATCH Technologies. All rights reserved."
        self.copy_foot = QtWidgets.QLabel(footer_text)
        self.copy_foot.setObjectName("footerLabel")
        self.copy_foot.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.copy_foot.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.copy_foot.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        footer_font = QtGui.QFont()
        footer_font.setStyleStrategy(QtGui.QFont.PreferAntialias | QtGui.QFont.PreferQuality)
        self.copy_foot.setFont(footer_font)
        toggle_layout.addWidget(self.copy_foot)

        toggle_layout.addStretch()

        # Software update status icon — bottom right, left of the log toggle
        _sw_icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "sw-update.svg")
        self.sw_update_icon = UpdateStatusIcon(_sw_icon_path, size=16)
        toggle_layout.addWidget(self.sw_update_icon)

        # Log toggle button
        self.btnLogToggle = QtWidgets.QToolButton(self.log_toggle_bar)
        self.btnLogToggle.setObjectName("logToggleBtn")
        self.btnLogToggle.setFixedSize(22, 18)
        self.btnLogToggle.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btnLogToggle.setIcon(self._log_chevron_down)
        self.btnLogToggle.setIconSize(QtCore.QSize(11, 11))
        self.btnLogToggle.setToolTip("Hide console")
        self.btnLogToggle.clicked.connect(self._toggle_logger)
        toggle_layout.addWidget(self.btnLogToggle)

        # Logger container
        self.log_container = QtWidgets.QWidget()
        self.log_container.setObjectName("logContainer")
        self.log_container.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        log_container_layout = QtWidgets.QVBoxLayout(self.log_container)
        log_container_layout.setContentsMargins(0, 0, 0, 0)
        log_container_layout.setSpacing(0)
        log_container_layout.addWidget(self.logview)

        self._log_expanded_height = 200
        self._log_collapsed = False

        # Outer vertical splitter
        self.log_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.log_splitter.setObjectName("logSplitter")
        self.log_splitter.setHandleWidth(6)
        self.log_splitter.addWidget(self.main_area)
        self.log_splitter.addWidget(self.log_container)
        self.log_splitter.setStretchFactor(0, 1)
        self.log_splitter.setStretchFactor(1, 0)
        self.log_splitter.setCollapsible(0, False)
        self.log_splitter.setCollapsible(1, True)
        self.log_splitter.setSizes([1000, self._log_expanded_height])
        self.log_splitter.splitterMoved.connect(self._on_log_splitter_moved)

        self._signed_in = True
        self._update_log_toggle_bar_theme()  # also sets the splitter handle style
        # Re-tint the dim strip/divider if the user flips light/dark mode
        ThemeManager.instance().themeChanged.connect(lambda _: self._update_log_toggle_bar_theme())

        # Animate the splitter sizes for the smooth open/close slide.
        self._log_anim = QtCore.QVariantAnimation(self.centralwidget)
        self._log_anim.setDuration(260)
        self._log_anim.setEasingCurve(QtCore.QEasingCurve.InOutCubic)
        self._log_anim.valueChanged.connect(self._on_log_anim_frame)

        self._force_splitter_mode_set = True
        if UserProfiles.count() > 0:
            self._set_no_user_mode(self.mode_mode, instant=True)
        else:
            # TODO: implement user sign in widget (show accordingly)
            self._set_run_mode(self.mode_run)
        self._force_splitter_mode_set = False
        # NOTE: splitter[0] widget must not change at load or else it disappears
        # (ignore the warning: "Trying to replace a widget with itself")
        elems = [parent.logger_window.ui.centralwidget, parent.plots_window.ui.centralwidget]
        for e in elems:
            not_resize = e.sizePolicy()
            not_resize.setHorizontalStretch(1)
            e.setSizePolicy(not_resize)

        elems = [parent.plots_window.ui.plt, parent.plots_window.ui.pltB]
        for i, e in enumerate(elems):
            not_resize = e.sizePolicy()
            not_resize.setVerticalStretch(i + 2)
            e.setSizePolicy(not_resize)

        outer_v = QtWidgets.QVBoxLayout()
        outer_v.setContentsMargins(0, 0, 0, 0)
        outer_v.setSpacing(0)
        outer_v.addWidget(self.log_splitter, 1)
        outer_v.addWidget(self.log_toggle_bar)
        self.centralwidget.setLayout(outer_v)
        mode_window.setCentralWidget(self.centralwidget)

        self._retranslate_ui(mode_window)
        QtCore.QMetaObject.connectSlotsByName(mode_window)

    def animate_mode_highlight(self, target_widget: QtWidgets.QWidget) -> None:
        """
        Animates the highlight to the selected menu item.

        Handles the smooth transition of the background highlight between modes
        using a QPropertyAnimation. It also updates dynamic properties to toggle
        hover effects and manages initial geometry delays if the layout is not
        yet rendered.

        Args:
            target_widget (QtWidgets.QWidget): The menu item widget (e.g., mode_run)
                that is becoming active.
        """
        # If layout hasn't calculated geometry yet, retry after a short delay
        if target_widget.geometry().width() == 0:
            QtCore.QTimer.singleShot(50, lambda: self.animate_mode_highlight(target_widget))
            return
        self.active_highlight.lower()
        if not self.active_highlight.isVisible():
            self.active_highlight.setGeometry(target_widget.geometry())
            self.active_highlight.show()
        else:
            self.mode_anim = QtCore.QPropertyAnimation(self.active_highlight, b"geometry")
            self.mode_anim.setDuration(250)
            self.mode_anim.setStartValue(self.active_highlight.geometry())
            self.mode_anim.setEndValue(target_widget.geometry())
            self.mode_anim.setEasingCurve(QtCore.QEasingCurve.InOutQuad)
            self.mode_anim.start()
        menu_items = [self.mode_run, self.mode_analyze, self.mode_learn]
        for widget in menu_items:
            is_active = widget == target_widget
            widget.setProperty("active", "true" if is_active else "false")
            widget.style().unpolish(widget)  # type: ignore
            widget.style().polish(widget)  # type: ignore

    def _toggle_logger(self) -> None:
        """
        Smoothly slide the logger open or closed by animating the splitter sizes.

        Collapsing animates the log pane to 0 height; expanding animates it back
        to the last user-chosen (or default) height. The chevron icon, tooltip,
        and strip color swap to reflect the next action.
        """
        self._log_anim.stop()
        sizes = self.log_splitter.sizes()
        total = sum(sizes)
        start = sizes[1] if len(sizes) > 1 else 0

        if self._log_collapsed:
            # Expand to the remembered height
            end = min(self._log_expanded_height, max(total - 60, 60))
        else:
            # Remember the current height, then collapse.
            if start > 0:
                self._log_expanded_height = start
            end = 0

        self._log_collapsed = not self._log_collapsed
        self._update_log_toggle_bar_theme()
        self._log_anim.setStartValue(start)
        self._log_anim.setEndValue(end)
        self._log_anim.start()

    def _on_log_anim_frame(self, value) -> None:
        """Apply an animation frame to the splitter sizes."""
        total = sum(self.log_splitter.sizes())
        log_h = int(value)
        self.log_splitter.setSizes([max(total - log_h, 0), log_h])

    def _on_log_splitter_moved(self, pos: int, index: int) -> None:
        """
        Track manual drags of the splitter handle.

        Keeps the toggle state and remembered height in sync so a manual resize
        is preserved across collapse/expand and the chevron points correctly.
        """
        if self._log_anim.state() == QtCore.QAbstractAnimation.State.Running:
            return  # ignore programmatic moves during animation
        log_h = self.log_splitter.sizes()[1]
        if log_h <= 0:
            if not self._log_collapsed:
                self._log_collapsed = True
                self._update_log_toggle_bar_theme()
        else:
            self._log_expanded_height = log_h
            if self._log_collapsed:
                self._log_collapsed = False
                self._update_log_toggle_bar_theme()

    @staticmethod
    def _tinted_icon(path: str, color: QtGui.QColor, size: int = 14) -> QtGui.QIcon:
        """Creates a color-tinted version of an icon while preserving transparency.

        Args:
            path (str): Filesystem path to the source icon (e.g., SVG or image file).
            color (QtGui.QColor): The color used to tint the icon.
            size (int, optional): Desired pixel size of the icon. Defaults to 14.

        Returns:
            QtGui.QIcon: A new QIcon instance containing the tinted pixmap.
        """
        src = QtGui.QIcon(path).pixmap(size, size)
        dst = QtGui.QPixmap(src.size())
        dst.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(dst)
        p.drawPixmap(0, 0, src)
        p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
        p.fillRect(dst.rect(), color)
        p.end()
        return QtGui.QIcon(dst)

    def mark_signed_in(self) -> None:
        """Restores the normal (light) console-strip look once a user signs in."""
        self._signed_in = True
        self._update_log_toggle_bar_theme()

    def _update_log_toggle_bar_theme(self) -> None:
        """Updates the console panel, divider, and footer/toggle bar color.

        The logger panel's own backdrop (`log_container`, behind its glass
        console), the drag handle dividing it from the main area, and the
        footer/toggle bar beneath it are all tinted with the same ~30%
        darkening the login overlay applies to the blurred dashboard
        whenever signed out - expanded or collapsed - so the whole strip
        reads as part of the same dimmed scene instead of bright elements
        poking out from under it. Everything stays its normal light self
        while signed in.
        """
        dark = not self._signed_in
        tokens = ThemeManager.instance().tokens()
        # "overlay_dim" keeps the strip/divider tint in sync with the login
        # overlay's blurred-dashboard dim level for the active theme.
        dim_css = tok_css(tokens["overlay_dim"])
        self.log_container.setStyleSheet(
            "QWidget#logContainer { background: %s; }" % (dim_css if dark else "transparent")
        )

        self.btnLogToggle.setToolTip("Show console" if self._log_collapsed else "Hide console")

        if self._log_collapsed:
            self.btnLogToggle.setIcon(self._log_chevron_up_light if dark else self._log_chevron_up)
        else:
            self.btnLogToggle.setIcon(
                self._log_chevron_down_light if dark else self._log_chevron_down
            )

        if dark:
            self.log_toggle_bar.setStyleSheet("QWidget#logToggleBar { background: %s; }" % dim_css)
            # Override QSS-driven color with the dim-state token so text
            # stays legible on whatever overlay_dim resolves to in this theme.
            self.copy_foot.setStyleSheet(
                "QLabel#footerLabel { color: %s; }" % tok_css(tokens["mode_footer_dim_text"])
            )
            self.log_splitter.setStyleSheet(
                "QSplitter#logSplitter::handle { background: %s; }"
                " QSplitter#logSplitter::handle:hover { background: %s; }"
                % (dim_css, tok_css(tokens["mode_splitter_dim_handle_hover"]))
            )
        else:
            self.log_toggle_bar.setStyleSheet("QWidget#logToggleBar { background: transparent; }")
            # Clear the widget-level override so the QSS-driven {{MODE_FOOTER_TEXT}}
            # token takes effect for the normal signed-in color.
            self.copy_foot.setStyleSheet("")
            self.log_splitter.setStyleSheet(
                "QSplitter#logSplitter::handle { background: transparent; }"
                " QSplitter#logSplitter::handle:hover { background: %s; }"
                % tok_css(tokens["mode_splitter_handle_hover"])
            )

    def _check_mode_change_allowed(self) -> bool:
        """
        Checks if the mode change is allowable in the current application state.

        Validates the current application state, prompts the user for unsaved changes
        in active modes, and warns if there is an active task running preventing it.

        Returns:
            bool: True if the mode change is allowed to continue; otherwise, False
        """
        # Active run in progress
        if (
            self.splitter.widget(0) == self.runview
            and not self.parent.controls_window.ui.pButton_Start.isEnabled()
        ):
            Log.e("Please stop the current run before switching modes.")
            return False

        # Check for busy and/or unsaved changes in Analyze or VisQ.AI
        for processor, name in [
            (self.parent.analyze_process, "Analyze"),
            (self.parent.visq_window, "VisQ.AI™"),
        ]:
            if processor.isBusy():
                PopUp.warning(
                    self.parent,
                    f"{name} Task In-Progress...",
                    "Mode change is not allowed while busy.",
                )
                return False
            if processor.hasUnsavedChanges():
                if PopUp.question(
                    self.parent,
                    Constants.app_title,
                    f"You have unsaved changes in {name}!\n\nAre you sure you want to close this window?",
                    False,
                ):
                    processor.clear()
                else:
                    Log.e(f"Please save or close your changes in {name} before switching modes.")
                    return False
        return True

    def _set_no_user_mode(
        self,
        obj: Optional[Any] = None,
        instant: bool = False,
    ) -> Optional[bool]:
        """
        Switches the application to 'No User' (Sign-In) mode.

        Validates the current application state, prompts the user for unsaved changes
        in active modes, resets the UI state (including hiding active menu highlights),
        and navigates to the login view.

        Args:
            obj (Optional[Any]): The event object triggering the mode change (e.g., QMouseEvent).
                If None, it indicates the function was called programmatically (e.g., session timeout)
                rather than via a UI interaction.
            instant (bool, optional): If True, the overlay snaps straight to its
                fully blurred/dimmed end state with no animation and no live
                dashboard capture. Used only for the app's cold start (called
                from `setup_ui` before the window has ever been shown), so the
                dashboard is never visible bright/sharp even for one frame.

        Returns:
            Optional[bool]:
                - True if the mode was successfully changed or already active (programmatic call).
                - False: the mode change was aborted due to not being allowed (programmatic call).
                - None if triggered via a UI event.
        """
        login = self.parent.login_window.ui.centralwidget

        # Already in No User / Sign-In mode
        if login.isVisible() and not self._force_splitter_mode_set:
            Log.d("User sign-in mode already active. Skipping mode change request.")
            if obj is None:
                return True
            # Continue execution if triggered by UI to ensure styles are forcefully reset

        # Check if the mode change is allowed
        if not self._check_mode_change_allowed():
            return False if obj is None else None

        # Apply UI Changes for Sign-In Mode
        self.parent.controls_window.ui_preferences.hide()
        self.active_highlight.hide()
        for widget in [self.mode_run, self.mode_analyze, self.mode_learn]:
            widget.setProperty("active", "false")
            widget.style().unpolish(widget)  # type: ignore
            widget.style().polish(widget)  # type: ignore
        self.parent.controls_window.set_signed_in_menu_state(False)
        self._signed_in = False
        self._update_log_toggle_bar_theme()

        if instant:
            login.reveal_signed_out(instant=True)

            def _refresh_backdrop() -> None:
                login.refresh_backdrop_instant(self.main_area)

            def _wait_then_refresh() -> None:
                self._wait_for_stable_size(self.main_area, _refresh_backdrop)

            QtCore.QTimer.singleShot(0, _wait_then_refresh)
        else:
            session_expired = obj is None and not UserProfiles.session_info()[0]
            session_loggedout = obj is None and not session_expired

            def _do_reveal() -> None:
                login.reveal_signed_out(self.main_area)
                if session_expired:
                    self.parent.login_window.ui.error_expired()
                elif session_loggedout:
                    self.parent.login_window.ui.error_loggedout()

            def _reveal() -> None:
                self._wait_for_stable_size(self.main_area, _do_reveal)

            QtCore.QTimer.singleShot(0, _reveal)

        self.parent.viewTutorialPage([1, 2, 0])

        # Focus the user initials input field after a short UI rendering delay
        QtCore.QTimer.singleShot(500, self.parent.login_window.ui.user_initials.setFocus)

        if obj is None:
            return True

        return None

    def _wait_for_stable_size(
        self,
        widget: QtWidgets.QWidget,
        callback: Callable[[], None],
        _last_size: Optional[QtCore.QSize] = None,
        _attempts: int = 0,
    ) -> None:
        """Calls `callback` once `widget`'s size stops changing between polls.

        On cold start, `showMaximized()` is called just before the event
        loop starts, but the window manager's maximize and Qt's own layout
        cascade for deeply nested splitters/scroll areas can take a couple
        of extra event-loop iterations to fully settle. A single deferred
        capture can grab `main_area` at its pre-settle (smaller) size,
        producing a visibly stretched backdrop that then "snaps" to the
        correct size once revealed. Polling until the size is unchanged
        between two checks (or a small attempt cap, as a safety net) avoids
        guessing a fixed delay while staying imperceptible once the window
        is already stable (the common case for sign-out/session expiry).

        Args:
            widget (QtWidgets.QWidget): The widget whose size must settle.
            callback (Callable[[], None]): Called once settled.
            _last_size (QtCore.QSize, optional): Internal - previous poll's size.
            _attempts (int, optional): Internal - poll count so far.
        """
        current_size = widget.size()
        settled = (
            current_size == _last_size and current_size.width() > 0 and current_size.height() > 0
        )
        if settled or _attempts >= 12:
            callback()
            return
        QtCore.QTimer.singleShot(
            30,
            lambda: self._wait_for_stable_size(widget, callback, current_size, _attempts + 1),
        )

    def _set_run_mode(
        self,
        obj: Optional[Any] = None,
    ) -> Optional[bool]:
        """
        Switches the application to 'Run' mode.

        Validates the current application state, prompts the user for unsaved changes
        in other modes, verifies user permissions, and updates the UI layout and animations.

        Args:
            obj (Optional[Any]): The event object triggering the mode change (e.g., QMouseEvent).
                If None, it indicates the function was called programmatically rather than
                via a UI interaction.

        Returns:
            Optional[bool]:
                - True if the mode was successfully changed or already active (programmatic call).
                - False: the mode change was aborted due to not being allowed (programmatic call).
                - None if triggered via a UI event.
        """
        current_widget = self.splitter.widget(0)
        target_widget = self.runview

        # Already in Run mode
        if current_widget == target_widget and not self._force_splitter_mode_set:
            Log.d("Run mode already active. Skipping mode change request.")
            self.animate_mode_highlight(self.mode_run)
            return True if obj is None else None

        # Check if the mode change is allowed
        if not self._check_mode_change_allowed():
            return False if obj is None else None

        # Check User Permissions
        action_role = UserRoles.CAPTURE
        check_result = UserProfiles().check(self.parent.controls_window.userrole, action_role)

        if check_result is None:  # User check required, but no user signed in
            Log.w(
                f"Not signed in: User with role {action_role.name} is required to perform this action."
            )
            Log.i("Please sign in to continue.")
            self.parent.controls_window.set_user_profile()  # Prompt for sign-in
            check_result = UserProfiles().check(
                self.parent.controls_window.userrole, action_role
            )  # Check again

        if not check_result:  # Explicitly denied
            Log.w(
                f"ACTION DENIED: User with role {self.parent.controls_window.userrole.name} does not have permission to {action_role.name}."
            )
            Log.e(
                "Please sign in to access Run mode."
                if check_result is None
                else "You are not authorized to access Run mode."
            )
            return False if obj is None else None

        self.parent._enable_ui(True)
        self.parent.visq_window.enable(False)
        self.animate_mode_highlight(self.mode_run)
        self.splitter.replaceWidget(0, target_widget)

        if UserProfiles.count() == 0:
            # Measure, Next Steps, Create Accounts
            self.parent.viewTutorialPage([3, 4, 0])
        else:
            # Measure, Next Steps
            self.parent.viewTutorialPage([3, 4])

        return True if obj is None else None

    def _set_analyze_mode(
        self,
        obj: Optional[Any] = None,
    ) -> Optional[bool]:
        """
        Switches the application to 'Analyze' mode.

        Performs state validation to check for busy processes or unsaved changes,
        verifies 'ANALYZE' role permissions, triggers the mode-switch animation,
        and updates the central view to the analysis processor.

        Args:
            obj (Optional[Any]): The event object triggering the mode change (e.g., QMouseEvent).
                If None, it indicates the function was called programmatically rather than
                via a UI interaction.

        Returns:
            Optional[bool]:
                - True if the mode was successfully changed or already active (programmatic call).
                - False: the mode change was aborted due to not being allowed (programmatic call).
                - None if triggered via a UI event.
        """
        current_widget = self.splitter.widget(0)
        target_widget = self.analyze

        # Already in Analyze mode
        if current_widget == target_widget and not self._force_splitter_mode_set:
            Log.d("Analyze mode already active. Skipping mode change request.")
            self.animate_mode_highlight(self.mode_analyze)
            return True if obj is None else None

        # Check if the mode change is allowed
        if not self._check_mode_change_allowed():
            return False if obj is None else None

        # Check User Permissions
        action_role = UserRoles.ANALYZE
        check_result = UserProfiles().check(self.parent.controls_window.userrole, action_role)

        if not check_result:
            if check_result is None:
                Log.e("Please sign in to access Analyze mode.")
            else:
                Log.e("You are not authorized to access Analyze mode.")
            return False if obj is None else None

        self.parent.analyze_data()
        self.parent._enable_ui(False)
        self.parent.visq_window.enable(False)
        self.animate_mode_highlight(self.mode_analyze)
        self.splitter.replaceWidget(0, target_widget)
        self.parent.viewTutorialPage([5, 6])  # analyze / prior results

        return True if obj is None else None

    def _set_learn_mode(
        self,
        obj: Optional[Any] = None,
        tab_index: int = 0,
    ) -> Optional[bool]:
        """
        Switches the application to 'VisQ.AI' mode.

        Validates state to prevent data loss or interruption of active runs. Verifies
        'OPERATE' role permissions, performs license validation, updates the
        internal toolkit tab index, and triggers the mode-switch animation.

        Args:
            obj (Optional[Any]): The event object triggering the mode change (e.g., QMouseEvent).
                If None, it indicates the function was called programmatically rather than
                via a UI interaction.
            tab_index (int): The index of the specific AI toolkit tab to display.
                Defaults to 0.

        Returns:
            Optional[bool]:
                - True if the mode was successfully changed or already active (programmatic call).
                - False: the mode change was aborted due to not being allowed (programmatic call).
                - None if triggered via a UI event.
        """
        current_widget = self.splitter.widget(0)
        target_widget = self.learn_ui

        # Already in Learn mode
        if current_widget == target_widget and not self._force_splitter_mode_set:
            if self.parent.visq_window.tab_widget.currentIndex() != tab_index:
                Log.d(f"VisQ.AI showing toolkit at index {tab_index}.")
                self.parent.visq_window.tab_widget.setCurrentIndex(tab_index)
            else:
                Log.d("VisQ.AI mode already active. Skipping mode change request.")

            self.animate_mode_highlight(self.mode_learn)
            return True if obj is None else None

        # Check if the mode change is allowed
        if not self._check_mode_change_allowed():
            return False if obj is None else None

        # Check User Permissions
        action_role = UserRoles.OPERATE
        check_result = UserProfiles().check(self.parent.controls_window.userrole, action_role)

        if check_result is None:  # Prompt sign-in if no user is active
            Log.w(f"Not signed in: {action_role.name} role required.")
            self.parent.controls_window.set_user_profile()
            check_result = UserProfiles().check(self.parent.controls_window.userrole, action_role)

        if not check_result:
            Log.e(
                "Please sign in to access VisQ.AI™ mode."
                if check_result is None
                else "You are not authorized to access VisQ.AI™ mode."
            )
            return False if obj is None else None

        self.parent.visq_window.reset()
        self.parent._enable_ui(False)
        self.parent.visq_window.enable(True)
        self.parent.visq_window.check_license(getattr(self.parent, "_license_manager", None))
        self.parent.visq_window.tab_widget.setCurrentIndex(tab_index)
        self.animate_mode_highlight(self.mode_learn)
        self.splitter.replaceWidget(0, target_widget)
        self.parent.viewTutorialPage(8)

        return True if obj is None else None

    def _set_donnan_mode(
        self,
        obj: Optional[Any] = None,
    ) -> Optional[bool]:
        """
        Switches the application to 'Donnan-Gibbs Calculator' mode.

        Validates the current application state, prompts the user for unsaved changes
        in other modes, verifies user permissions, and updates the UI layout and animations.

        Args:
            obj (Optional[Any]): The event object triggering the mode change (e.g., QMouseEvent).
                If None, it indicates the function was called programmatically rather than
                via a UI interaction.

        Returns:
            Optional[bool]:
                - True if the mode was successfully changed or already active (programmatic call).
                - False: the mode change was aborted due to not being allowed (programmatic call).
                - None if triggered via a UI event.
        """
        current_widget = self.splitter.widget(0)
        target_widget = self.donnan_ui

        # Already in Donnan-Gibbs Calculator mode
        if current_widget == target_widget and not self._force_splitter_mode_set:
            Log.d("Donnan-Gibbs Calculator already active. Skipping mode change request.")
            self.animate_mode_highlight(self.mode_donnan)
            return True if obj is None else None

        # Check if the mode change is allowed
        if not self._check_mode_change_allowed():
            return False if obj is None else None

        # Check User Permissions
        action_role = UserRoles.ANALYZE
        check_result = UserProfiles().check(self.parent.controls_window.userrole, action_role)

        if check_result is None:  # User check required, but no user signed in
            Log.w(
                f"Not signed in: User with role {action_role.name} is required to perform this action."
            )
            Log.i("Please sign in to continue.")
            self.parent.controls_window.set_user_profile()  # Prompt for sign-in
            check_result = UserProfiles().check(
                self.parent.controls_window.userrole, action_role
            )  # Check again

        if not check_result:  # Explicitly denied
            Log.w(
                f"ACTION DENIED: User with role {self.parent.controls_window.userrole.name} does not have permission to {action_role.name}."
            )
            Log.e(
                "Please sign in to access the Donnan-Gibbs Calculator."
                if check_result is None
                else "You are not authorized to access the Donnan-Gibbs Calculator."
            )
            return False if obj is None else None

        self.parent._enable_ui(False)
        self.parent.visq_window.enable(False)
        self.animate_mode_highlight(self.mode_donnan)
        self.splitter.replaceWidget(0, target_widget)

        # No tutorial pages specified for this mode

        return True if obj is None else None

    def _set_injection_mode(
        self,
        obj: Optional[Any] = None,
    ) -> Optional[bool]:
        """
        Switches the application to 'Injection Force Calculator' mode.

        Validates the current application state, prompts the user for unsaved changes
        in other modes, verifies user permissions, and updates the UI layout and animations.

        Args:
            obj (Optional[Any]): The event object triggering the mode change (e.g., QMouseEvent).
                If None, it indicates the function was called programmatically rather than
                via a UI interaction.

        Returns:
            Optional[bool]:
                - True if the mode was successfully changed or already active (programmatic call).
                - False: the mode change was aborted due to not being allowed (programmatic call).
                - None if triggered via a UI event.
        """
        current_widget = self.splitter.widget(0)
        target_widget = self.injection_ui

        # Already in Injection Force Calculator mode
        if current_widget == target_widget and not self._force_splitter_mode_set:
            Log.d("Injection Force Calculator already active. Skipping mode change request.")
            self.animate_mode_highlight(self.mode_injection)
            return True if obj is None else None

        # Check if the mode change is allowed
        if not self._check_mode_change_allowed():
            return False if obj is None else None

        # Check User Permissions
        action_role = UserRoles.ANALYZE
        check_result = UserProfiles().check(self.parent.controls_window.userrole, action_role)

        if check_result is None:  # User check required, but no user signed in
            Log.w(
                f"Not signed in: User with role {action_role.name} is required to perform this action."
            )
            Log.i("Please sign in to continue.")
            self.parent.controls_window.set_user_profile()  # Prompt for sign-in
            check_result = UserProfiles().check(
                self.parent.controls_window.userrole, action_role
            )  # Check again

        if not check_result:  # Explicitly denied
            Log.w(
                f"ACTION DENIED: User with role {self.parent.controls_window.userrole.name} does not have permission to {action_role.name}."
            )
            Log.e(
                "Please sign in to access the Injection Force Calculator."
                if check_result is None
                else "You are not authorized to access the Injection Force Calculator."
            )
            return False if obj is None else None

        self.parent._enable_ui(False)
        self.parent.visq_window.enable(False)
        self.animate_mode_highlight(self.mode_injection)
        self.splitter.replaceWidget(0, target_widget)

        # No tutorial pages specified for this mode

        return True if obj is None else None

    def _retranslate_ui(self, mode_window: Any) -> None:
        """
        Updates the localized text and window properties for the main application.

        This method handles the translation of UI strings, sets the application
        window icon from the local assets, and formats the window title to
        include the current application name and version.

        Args:
            mode_window (Any): The top-level QMainWindow instance to be updated.
                Expected to be a QtWidgets.QMainWindow or similar.
        """
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "high-res-qatch-logo-no-bg.png"
        )
        mode_window.setWindowIcon(QtGui.QIcon(icon_path))
        app_title_full = f"{Constants.app_title} {Constants.app_version}"
        mode_window.setWindowTitle(_translate("modeWindow", app_title_full))
