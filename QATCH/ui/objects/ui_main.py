import datetime
import os
from typing import Optional, Any
from PyQt5 import QtCore, QtGui, QtWidgets
from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles
from QATCH.core.constants import Constants, UserRoles
from QATCH.tools.donnan_gibbs_calculator import DonnanCalculatorModule
from QATCH.tools.injection_force_calculator import InjectionForceCalculatorModule
from QATCH.ui.popUp import PopUp
from QATCH.ui.widgets.floating_menu_widget import FloatingMenuWidget


class _MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent):
        super().__init__()
        parent.ControlsWin._createMenu(self)
        self.ui0 = UIMain()
        self.ui0.setup_ui(self, parent)

        # Get the application instance safely and connect the signals
        app_instance = QtWidgets.QApplication.instance()
        if app_instance:
            app_instance.focusWindowChanged.connect(self.focusWindowChanged)
            # capture clicks anywhere on gui
            app_instance.installEventFilter(self)

    def eventFilter(self, obj, event):
        # Handle mouse click events (e.g. hide on click)
        if event.type() == QtCore.QEvent.MouseButtonPress:
            widget_clicked = QtWidgets.QApplication.widgetAt(event.globalPos())
            allow_hide = True
            if widget_clicked is self.ui0.mode_learn:
                allow_hide = False
            # if widget_clicked is self.ui0.mode_learn_text:
            #     allow_hide = False
            # if widget_clicked is self.ui0.mode_learn_arrow:
            #     allow_hide = False
            if self.ui0.floating_widget.isVisible() and allow_hide:
                self.ui0.floating_widget.hide()
        return super().eventFilter(obj, event)

    def focusWindowChanged(self, focus_window):
        # Hide the floating widget only when the focus leaves this window
        # NOTE: This is a signal slot event firing, there is no `super()`
        if focus_window is None or focus_window != self.windowHandle():
            self.ui0.floating_widget.hide()

    def moveEvent(self, event):
        # Hide the floating widget whenever the main window moves
        # NOTE: Its position will be recalculated on next `show()`
        self.ui0.floating_widget.hide()
        super().moveEvent(event)

    def closeEvent(self, event):
        # Log.d(" Exit Real-Time Plot GUI")
        res = PopUp.question(
            self,
            Constants.app_title,
            "Are you sure you want to quit QATCH Q-1 application now?",
            True,
        )
        if res:
            # self.close()
            QtWidgets.QApplication.quit()
        else:
            event.ignore()


class UIMain:
    """The primary user interface controller for the nanovisQ application.

    This class manages the lifecycle and state of the main application window.
    It coordinates  navigation between core operational modes (Run, Analyze, and VisQ.AI),
    handles dynamic UI animations, and manages global styling via QSS.

    The architecture utilizes a centralized splitter system to toggle between
    primary workspace views and a collapsible diagnostic log area.

    Attributes:
        parent (Any): Reference to the main controller/logic parent containing
            sub-window instances (e.g., AnalyzeProc, VisQAIWin).
        centralwidget (QtWidgets.QWidget): The root container for the window
            utilizing a gradient-based glass background.
        active_highlight (QtWidgets.QWidget): An animated translucent widget
        used to indicate the currently selected menu mode.
        modemenu (QtWidgets.QScrollArea): The sidebar container for mode
            navigation.
        splitter (QtWidgets.QSplitter): The vertical divider separating the
            active workspace from the log view.
        mode_run (QtWidgets.QLabel): Interactive label for 'Run' mode.
        mode_analyze (QtWidgets.QLabel): Interactive label for 'Analyze' mode.
        mode_learn (QtWidgets.QLabel): Interactive label for 'VisQ.AI' mode.
        logview (QtWidgets.QScrollArea): The bottom diagnostic logging pane.
    """

    def setup_ui(self, main_window: QtWidgets.QMainWindow, parent: Any) -> None:
        """
        Initializes the main user interface for the nanovisQ application.

        Constructs the glassmorphic layout, including the sidebar navigation,
        central view splitter, and log area. Loads global styling from QSS and
        handles initial state configuration for user sessions.

        Args:
            main_window (QtWidgets.QMainWindow): The primary window instance.
            parent (Any): The controller or parent object containing sub-window logic
                (e.g., AnalyzeProc, VisQAIWin).
        """
        self.parent = parent
        main_window.setObjectName("main_window_0")
        main_window.setMinimumSize(QtCore.QSize(1331, 711))
        self.centralwidget = QtWidgets.QWidget(main_window)
        self.centralwidget.setObjectName("centralwidget")
        qss_file_path = os.path.join(
            Architecture.get_path(), "QATCH", "ui", "styles", "ui_main_theme.qss"
        )
        try:
            with open(qss_file_path, "r") as style_file:
                self.centralwidget.setStyleSheet(style_file.read())
        except FileNotFoundError:
            Log.e(f"Could not find the stylesheet at: {qss_file_path}")
        self.centralwidget.setContentsMargins(10, 10, 0, 10)
        layout_h = QtWidgets.QHBoxLayout()
        layout_h.setSpacing(10)

        # Add to mode menu here:
        # Run / Analyze / VisQ.AI
        modewidget = QtWidgets.QWidget()
        modewidget.setObjectName("modeWidgetContainer")

        # Apply the style to the PARENT widget so the CSS cascade works properly
        modelayout = QtWidgets.QVBoxLayout()
        modelayout.setContentsMargins(0, 0, 0, 0)
        modelayout.setSpacing(0)

        # Sliding Highlight Widget
        self.active_highlight = QtWidgets.QWidget(modewidget)
        self.active_highlight.setObjectName("activeHighlight")
        self.active_highlight.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.active_highlight.hide()

        # Logo
        icon_path = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "high-res-nanovisq-logo-no-bg.png"
        )
        original_pixmap = QtGui.QPixmap(icon_path).scaledToWidth(
            100, QtCore.Qt.SmoothTransformation
        )
        rounded_pixmap = QtGui.QPixmap(original_pixmap.size())
        rounded_pixmap.fill(QtCore.Qt.transparent)

        painter = QtGui.QPainter(rounded_pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        path = QtGui.QPainterPath()
        path.addRoundedRect(0, 0, original_pixmap.width(), original_pixmap.height(), 12, 12)

        painter.setClipPath(path)
        painter.drawPixmap(0, 0, original_pixmap)
        painter.end()
        self.logolabel = QtWidgets.QLabel()
        self.logolabel.setStyleSheet("padding-bottom:10px; background: transparent;")
        self.logolabel.setPixmap(rounded_pixmap)
        self.logolabel.resize(rounded_pixmap.width(), rounded_pixmap.height())

        # Mode Header
        self.mode_mode = QtWidgets.QLabel("<b>MODE</b>")
        self.mode_mode.setStyleSheet("padding: 10px; padding-top: 15px;")

        # Run Mode
        self.mode_run = QtWidgets.QLabel("Run")
        self.mode_run.setObjectName("menuItem")
        self.mode_run.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.mode_run.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.mode_run.mousePressEvent = self._set_run_mode

        # Analyze Mode
        self.mode_analyze = QtWidgets.QLabel("Analyze")
        self.mode_analyze.setObjectName("menuItem")
        self.mode_analyze.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.mode_analyze.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.mode_analyze.mousePressEvent = self._set_analyze_mode

        # Tools Header
        self.mode_tools = QtWidgets.QLabel("<b>TOOLS</b>")
        self.mode_tools.setStyleSheet("padding: 10px; padding-top: 15px;")

        # VisQ.AI Tool
        self.mode_learn = QtWidgets.QLabel("VisQ.AI™")
        # Using this instead of <sup> for beter antialiasing.
        self.mode_learn.setObjectName("menuItem")
        self.mode_learn.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.mode_learn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.mode_learn.mousePressEvent = self._set_learn_mode

        # Donnan Calculator Tool
        self.mode_donnan = QtWidgets.QLabel("Donnan-Gibbs Calculator")
        self.mode_donnan.setObjectName("menuItem")
        self.mode_donnan.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.mode_donnan.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.mode_donnan.setWordWrap(True)
        self.mode_donnan.mousePressEvent = self._set_donnan_mode

        # Injection Force Calculator Tool
        self.mode_injection = QtWidgets.QLabel("Injection Force Calculator")
        self.mode_injection.setObjectName("menuItem")
        self.mode_injection.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.mode_injection.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.mode_injection.setWordWrap(True)
        self.mode_injection.mousePressEvent = self._set_injection_mode

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
        self.mode_mode.setStyleSheet("padding: 10px; padding-top: 15px;")
        self.modemenu.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.modemenu.setLineWidth(0)
        self.modemenu.setMidLineWidth(0)
        self.modemenu.setWidgetResizable(True)
        self.modemenu.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.modemenu.setMinimumSize(QtCore.QSize(100, 700))
        self.modemenu.setWidget(modewidget)

        # Create floating menu widget for VisQ.AI Toolkit
        self.floating_widget = FloatingMenuWidget(self)
        self.floating_widget.addItems(self.parent.VisQAIWin.getToolNames())

        # user sign-in view frame: TODO
        self.userview = QtWidgets.QScrollArea()
        self.userview.setObjectName("userview")
        self.userview.setFrameShape(QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Plain)
        self.userview.setLineWidth(0)
        self.userview.setMidLineWidth(0)
        self.userview.setWidgetResizable(True)
        # TODO: Implement user widget
        self.userview.setWidget(parent.LoginWin.ui5.centralwidget)
        self.userview.setMinimumSize(QtCore.QSize(1000, 122))

        # run mode view frame: Controls and Plots
        runwidget = QtWidgets.QWidget()
        runlayout = QtWidgets.QVBoxLayout()
        runlayout.setContentsMargins(0, 0, 0, 0)
        runlayout.setSpacing(0)
        runlayout.addWidget(parent.ControlsWin.ui1.centralwidget, 1)
        runlayout.addWidget(parent.PlotsWin.ui2.centralwidget, 255)
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
        self.analyze.setWidget(parent.AnalyzeProc)
        self.analyze.setMinimumSize(QtCore.QSize(1000, 122))

        # learn mode view frame: VisQ.AI
        self.learn_ui = QtWidgets.QScrollArea()
        self.learn_ui.setObjectName("learn_ui")
        self.learn_ui.setFrameShape(QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Plain)
        self.learn_ui.setLineWidth(0)
        self.learn_ui.setMidLineWidth(0)
        self.learn_ui.setWidgetResizable(True)
        self.learn_ui.setWidget(parent.VisQAIWin)
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
        self.logview.setWidget(parent.LogWin.ui4.centralwidget)
        # Min height relaxed to 0 so the pane can slide/drag closed.
        self.logview.setMinimumSize(QtCore.QSize(1000, 0))
        # Transparent background to match the glass effect of other windows.
        self.logview.setStyleSheet(
            "QScrollArea#logview, QScrollArea#logview > QWidget > QWidget"
            " { background: transparent; }"
        )
        self.logview.viewport().setAutoFillBackground(False)

        layout_h.addWidget(self.modemenu, 1)
        layout_v = QtWidgets.QVBoxLayout()
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background: transparent;
            }
        """)
        if UserProfiles.count() > 0:
            # NOTE: this widget must not be changed at load time (or else it disappears)
            self.splitter.addWidget(self.userview)
        else:
            self.splitter.addWidget(self.runview)
        # Logger removed from the splitter; it is now a full-width bottom row.
        # The splitter holds only the active view, so the main row's height is
        # governed by the view/plots content.
        layout_v.addWidget(self.splitter)
        layout_h.addLayout(layout_v, 255)

        # ----- Controls/footer bar: footer text (left) + subtle toggle (right) -----
        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")
        self._log_chevron_down = QtGui.QIcon(os.path.join(icons_dir, "down-chevron.svg"))
        self._log_chevron_up = QtGui.QIcon(os.path.join(icons_dir, "up-chevron.svg"))

        self.log_toggle_bar = QtWidgets.QWidget()
        self.log_toggle_bar.setObjectName("logToggleBar")
        self.log_toggle_bar.setFixedHeight(20)
        toggle_layout = QtWidgets.QHBoxLayout(self.log_toggle_bar)
        toggle_layout.setContentsMargins(8, 0, 6, 0)
        toggle_layout.setSpacing(6)

        # Footer copyright text — small, non-italic, clearly static.
        current_year = datetime.date.today().year
        footer_text = f"\u00a9 {current_year} QATCH Technologies. All rights reserved."
        self.copy_foot = QtWidgets.QLabel(footer_text)
        self.copy_foot.setObjectName("footerLabel")
        self.copy_foot.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.copy_foot.setFocusPolicy(QtCore.Qt.NoFocus)
        self.copy_foot.setCursor(QtCore.Qt.ArrowCursor)
        self.copy_foot.setStyleSheet("""
            QLabel#footerLabel {
                background: transparent;
                border: none;
                color: rgba(70, 80, 95, 140);
                font-size: 9px;
                font-weight: normal;
            }
        """)
        footer_font = QtGui.QFont()
        footer_font.setStyleStrategy(QtGui.QFont.PreferAntialias | QtGui.QFont.PreferQuality)
        self.copy_foot.setFont(footer_font)
        toggle_layout.addWidget(self.copy_foot)

        toggle_layout.addStretch()

        # Subtle, flat toggle tucked into the right edge of the controls bar.
        self.btnLogToggle = QtWidgets.QToolButton(self.log_toggle_bar)
        self.btnLogToggle.setObjectName("logToggleBtn")
        self.btnLogToggle.setFixedSize(22, 18)
        self.btnLogToggle.setCursor(QtCore.Qt.PointingHandCursor)
        self.btnLogToggle.setIcon(self._log_chevron_down)  # expanded -> shows collapse action
        self.btnLogToggle.setIconSize(QtCore.QSize(11, 11))
        self.btnLogToggle.setToolTip("Hide console")
        self.btnLogToggle.setStyleSheet("""
            QToolButton#logToggleBtn {
                background: transparent;
                border: none;
                border-radius: 4px;
            }
            QToolButton#logToggleBtn:hover {
                background: rgba(120, 130, 145, 45);
            }
            QToolButton#logToggleBtn:pressed {
                background: rgba(120, 130, 145, 80);
            }
        """)
        self.btnLogToggle.clicked.connect(self._toggle_logger)
        toggle_layout.addWidget(self.btnLogToggle)

        # ----- Full-width logger container (transparent, drag-resizable) -----
        self.log_container = QtWidgets.QWidget()
        self.log_container.setObjectName("logContainer")
        # Transparent so the glass effect of inner widgets shows through.
        self.log_container.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.log_container.setStyleSheet("QWidget#logContainer { background: transparent; }")
        log_container_layout = QtWidgets.QVBoxLayout(self.log_container)
        log_container_layout.setContentsMargins(0, 0, 0, 0)
        log_container_layout.setSpacing(0)
        log_container_layout.addWidget(self.logview)

        self._log_expanded_height = 200
        self._log_collapsed = False

        # self.splitter.replaceWidget(0, self.userview)
        self._force_splitter_mode_set = True
        if UserProfiles.count() > 0:
            self._set_no_user_mode(self.mode_mode)
        else:
            # TODO: implement user sign in widget (show accordingly)
            self._set_run_mode(self.mode_run)
        self._force_splitter_mode_set = False
        # NOTE: splitter[0] widget must not change at load or else it disappears
        # (ignore the warning: "Trying to replace a widget with itself")

        # retain sizing of view menu toggle elements
        elems = [parent.LogWin.ui4.centralwidget, parent.PlotsWin.ui2.centralwidget]
        for e in elems:
            not_resize = e.sizePolicy()
            not_resize.setHorizontalStretch(1)
            e.setSizePolicy(not_resize)

        elems = [parent.PlotsWin.ui2.plt, parent.PlotsWin.ui2.pltB]
        for i, e in enumerate(elems):
            not_resize = e.sizePolicy()
            not_resize.setVerticalStretch(i + 2)
            e.setSizePolicy(not_resize)

        # Main area (modemenu + views) wrapped so it can sit in the outer splitter.
        main_area = QtWidgets.QWidget()
        main_area.setObjectName("mainArea")
        main_area.setLayout(layout_h)

        # Outer vertical splitter: draggable handle lets the user resize the log.
        self.log_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.log_splitter.setObjectName("logSplitter")
        self.log_splitter.setHandleWidth(6)
        self.log_splitter.setStyleSheet("""
            QSplitter#logSplitter::handle {
                background: transparent;
            }
            QSplitter#logSplitter::handle:hover {
                background: rgba(120, 130, 145, 60);
            }
        """)
        self.log_splitter.addWidget(main_area)
        self.log_splitter.addWidget(self.log_container)
        self.log_splitter.setStretchFactor(0, 1)  # main area takes extra space
        self.log_splitter.setStretchFactor(1, 0)  # logger keeps its set size
        # Allow the logger pane to collapse fully (to 0) when toggled/dragged,
        # but keep the main area from vanishing.
        self.log_splitter.setCollapsible(0, False)
        self.log_splitter.setCollapsible(1, True)
        self.log_splitter.setSizes([1000, self._log_expanded_height])
        # Track manual drags so the toggle restores to the user's chosen height.
        self.log_splitter.splitterMoved.connect(self._on_log_splitter_moved)

        # Animate the splitter sizes for the smooth open/close slide.
        self._log_anim = QtCore.QVariantAnimation(self.centralwidget)
        self._log_anim.setDuration(260)
        self._log_anim.setEasingCurve(QtCore.QEasingCurve.InOutCubic)
        self._log_anim.valueChanged.connect(self._on_log_anim_frame)

        outer_v = QtWidgets.QVBoxLayout()
        outer_v.setContentsMargins(0, 0, 0, 0)
        outer_v.setSpacing(4)
        outer_v.addWidget(self.log_splitter, 1)  # main area over draggable logger
        outer_v.addWidget(self.log_toggle_bar)  # full-width toggle + footer bar
        self.centralwidget.setLayout(outer_v)
        main_window.setCentralWidget(self.centralwidget)

        self._retranslate_ui(main_window)
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def animate_mode_highlight(self, target_widget: QtWidgets.QWidget) -> None:
        """
        Animates the glassmorphic highlight to the selected menu item.

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

        # Snap to position if hidden (initial load), otherwise animate
        if not self.active_highlight.isVisible():
            self.active_highlight.setGeometry(target_widget.geometry())
            self.active_highlight.show()
        else:
            # Create smooth geometry interpolation
            self.mode_anim = QtCore.QPropertyAnimation(self.active_highlight, b"geometry")
            self.mode_anim.setDuration(250)
            self.mode_anim.setStartValue(self.active_highlight.geometry())
            self.mode_anim.setEndValue(target_widget.geometry())
            self.mode_anim.setEasingCurve(QtCore.QEasingCurve.InOutQuad)
            self.mode_anim.start()

        # Update 'active' property to toggle QSS hover effects
        menu_items = [self.mode_run, self.mode_analyze, self.mode_learn]
        for widget in menu_items:
            is_active = widget == target_widget
            widget.setProperty("active", "true" if is_active else "false")
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def _toggle_logger(self) -> None:
        """
        Smoothly slide the logger open or closed by animating the splitter sizes.

        Collapsing animates the log pane to 0 height; expanding animates it back
        to the last user-chosen (or default) height. The chevron icon and tooltip
        swap to reflect the next action.
        """
        self._log_anim.stop()
        sizes = self.log_splitter.sizes()
        total = sum(sizes)
        start = sizes[1] if len(sizes) > 1 else 0

        if self._log_collapsed:
            # Expand to the remembered height (clamped to available space).
            end = min(self._log_expanded_height, max(total - 60, 60))
            self.btnLogToggle.setIcon(self._log_chevron_down)
            self.btnLogToggle.setToolTip("Hide console")
        else:
            # Remember the current height, then collapse.
            if start > 0:
                self._log_expanded_height = start
            end = 0
            self.btnLogToggle.setIcon(self._log_chevron_up)
            self.btnLogToggle.setToolTip("Show console")

        self._log_collapsed = not self._log_collapsed
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
        if self._log_anim.state() == QtCore.QAbstractAnimation.Running:
            return  # ignore programmatic moves during animation
        log_h = self.log_splitter.sizes()[1]
        if log_h <= 0:
            if not self._log_collapsed:
                self._log_collapsed = True
                self.btnLogToggle.setIcon(self._log_chevron_up)
                self.btnLogToggle.setToolTip("Show console")
        else:
            self._log_expanded_height = log_h
            if self._log_collapsed:
                self._log_collapsed = False
                self.btnLogToggle.setIcon(self._log_chevron_down)
                self.btnLogToggle.setToolTip("Hide console")

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
            and not self.parent.ControlsWin.ui1.pButton_Start.isEnabled()
        ):
            Log.e("Please stop the current run before switching modes.")
            return False

        # Check for busy and/or unsaved changes in Analyze or VisQ.AI
        for processor, name in [
            (self.parent.AnalyzeProc, "Analyze"),
            (self.parent.VisQAIWin, "VisQ.AI™"),
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

    def _set_no_user_mode(self, obj: Optional[Any] = None) -> Optional[bool]:
        """
        Switches the application to 'No User' (Sign-In) mode.

        Validates the current application state, prompts the user for unsaved changes
        in active modes, resets the UI state (including hiding active menu highlights),
        and navigates to the login view.

        Args:
            obj (Optional[Any]): The event object triggering the mode change (e.g., QMouseEvent).
                If None, it indicates the function was called programmatically (e.g., session timeout)
                rather than via a UI interaction.

        Returns:
            Optional[bool]:
                - True if the mode was successfully changed or already active (programmatic call).
                - False: the mode change was aborted due to not being allowed (programmatic call).
                - None if triggered via a UI event.
        """
        current_widget = self.splitter.widget(0)
        target_widget = self.userview

        # Already in No User / Sign-In mode
        if current_widget == target_widget and not self._force_splitter_mode_set:
            Log.d("User sign-in mode already active. Skipping mode change request.")
            if obj is None:
                return True
            # Continue execution if triggered by UI to ensure styles are forcefully reset

        # Check if the mode change is allowed
        if not self._check_mode_change_allowed():
            return False if obj is None else None

        # Apply UI Changes for Sign-In Mode
        self.parent.ControlsWin.ui_preferences.hide()
        self.active_highlight.hide()
        for widget in [self.mode_run, self.mode_analyze, self.mode_learn]:
            widget.setProperty("active", "false")
            widget.style().unpolish(widget)
            widget.style().polish(widget)
        self.splitter.replaceWidget(0, target_widget)

        self.parent.viewTutorialPage([1, 2, 0])

        # Focus the user initials input field after a short UI rendering delay
        QtCore.QTimer.singleShot(500, self.parent.LoginWin.ui5.user_initials.setFocus)

        if obj is None:
            if not UserProfiles.session_info()[0]:  # User session expired
                self.parent.LoginWin.ui5.error_expired()
            else:  # User manually logged out
                self.parent.LoginWin.ui5.error_loggedout()
            return True

        return None

    def _set_run_mode(self, obj: Optional[Any] = None) -> Optional[bool]:
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
            return True if obj is None else None

        # Check if the mode change is allowed
        if not self._check_mode_change_allowed():
            return False if obj is None else None

        # Check User Permissions
        action_role = UserRoles.CAPTURE
        check_result = UserProfiles().check(self.parent.ControlsWin.userrole, action_role)

        if check_result is None:  # User check required, but no user signed in
            Log.w(
                f"Not signed in: User with role {action_role.name} is required to perform this action."
            )
            Log.i("Please sign in to continue.")
            self.parent.ControlsWin.set_user_profile()  # Prompt for sign-in
            check_result = UserProfiles().check(
                self.parent.ControlsWin.userrole, action_role
            )  # Check again

        if not check_result:  # Explicitly denied
            Log.w(
                f"ACTION DENIED: User with role {self.parent.ControlsWin.userrole.name} does not have permission to {action_role.name}."
            )
            Log.e(
                "Please sign in to access Run mode."
                if check_result is None
                else "You are not authorized to access Run mode."
            )
            return False if obj is None else None

        self.parent._enable_ui(True)
        self.parent.VisQAIWin.enable(False)
        self.animate_mode_highlight(self.mode_run)
        self.splitter.replaceWidget(0, target_widget)
        self.parent.PlotsWin.ui2.handleSplitterButton(collapse=False)

        if UserProfiles.count() == 0:
            # Measure, Next Steps, Create Accounts
            self.parent.viewTutorialPage([3, 4, 0])
        else:
            # Measure, Next Steps
            self.parent.viewTutorialPage([3, 4])

        return True if obj is None else None

    def _set_analyze_mode(self, obj: Optional[Any] = None) -> Optional[bool]:
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
            return True if obj is None else None

        # Check if the mode change is allowed
        if not self._check_mode_change_allowed():
            return False if obj is None else None

        # Check User Permissions
        action_role = UserRoles.ANALYZE
        check_result = UserProfiles().check(self.parent.ControlsWin.userrole, action_role)

        if not check_result:
            if check_result is None:
                Log.e("Please sign in to access Analyze mode.")
            else:
                Log.e("You are not authorized to access Analyze mode.")
            return False if obj is None else None

        self.parent.analyze_data()
        self.parent._enable_ui(False)
        self.parent.VisQAIWin.enable(False)
        self.animate_mode_highlight(self.mode_analyze)
        self.splitter.replaceWidget(0, target_widget)
        self.parent.viewTutorialPage([5, 6])  # analyze / prior results

        return True if obj is None else None

    def _set_learn_mode(self, obj: Optional[Any] = None, tab_index: int = 0) -> Optional[bool]:
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
            if self.parent.VisQAIWin.tab_widget.currentIndex() != tab_index:
                Log.d(f"VisQ.AI showing toolkit at index {tab_index}.")
                self.parent.VisQAIWin.tab_widget.setCurrentIndex(tab_index)
            else:
                Log.d("VisQ.AI mode already active. Skipping mode change request.")

            return True if obj is None else None

        # Check if the mode change is allowed
        if not self._check_mode_change_allowed():
            return False if obj is None else None

        # Check User Permissions
        action_role = UserRoles.OPERATE
        check_result = UserProfiles().check(self.parent.ControlsWin.userrole, action_role)

        if check_result is None:  # Prompt sign-in if no user is active
            Log.w(f"Not signed in: {action_role.name} role required.")
            self.parent.ControlsWin.set_user_profile()
            check_result = UserProfiles().check(self.parent.ControlsWin.userrole, action_role)

        if not check_result:
            Log.e(
                "Please sign in to access VisQ.AI™ mode."
                if check_result is None
                else "You are not authorized to access VisQ.AI™ mode."
            )
            return False if obj is None else None

        self.parent.VisQAIWin.reset()
        self.parent._enable_ui(False)
        self.parent.VisQAIWin.enable(True)
        self.parent.VisQAIWin.check_license(getattr(self.parent, "_license_manager", None))
        self.parent.VisQAIWin.tab_widget.setCurrentIndex(tab_index)
        self.animate_mode_highlight(self.mode_learn)
        self.splitter.replaceWidget(0, target_widget)
        self.parent.viewTutorialPage(8)

        return True if obj is None else None

    def _set_donnan_mode(self, obj: Optional[Any] = None) -> Optional[bool]:
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

        # Already in Run mode
        if current_widget == target_widget and not self._force_splitter_mode_set:
            Log.d("Donnan-Gibbs Calculator already active. Skipping mode change request.")
            return True if obj is None else None

        # Check if the mode change is allowed
        if not self._check_mode_change_allowed():
            return False if obj is None else None

        # Check User Permissions
        action_role = UserRoles.ANALYZE
        check_result = UserProfiles().check(self.parent.ControlsWin.userrole, action_role)

        if check_result is None:  # User check required, but no user signed in
            Log.w(
                f"Not signed in: User with role {action_role.name} is required to perform this action."
            )
            Log.i("Please sign in to continue.")
            self.parent.ControlsWin.set_user_profile()  # Prompt for sign-in
            check_result = UserProfiles().check(
                self.parent.ControlsWin.userrole, action_role
            )  # Check again

        if not check_result:  # Explicitly denied
            Log.w(
                f"ACTION DENIED: User with role {self.parent.ControlsWin.userrole.name} does not have permission to {action_role.name}."
            )
            Log.e(
                "Please sign in to access the Donnan-Gibbs Calculator."
                if check_result is None
                else "You are not authorized to access the Donnan-Gibbs Calculator."
            )
            return False if obj is None else None

        self.parent._enable_ui(False)
        self.parent.VisQAIWin.enable(False)
        self.animate_mode_highlight(self.mode_donnan)
        self.splitter.replaceWidget(0, target_widget)

        # No tutorial pages specified for this mode

        return True if obj is None else None

    def _set_injection_mode(self, obj: Optional[Any] = None) -> Optional[bool]:
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

        # Already in Run mode
        if current_widget == target_widget and not self._force_splitter_mode_set:
            Log.d("Injection Force Calculator already active. Skipping mode change request.")
            return True if obj is None else None

        # Check if the mode change is allowed
        if not self._check_mode_change_allowed():
            return False if obj is None else None

        # Check User Permissions
        action_role = UserRoles.ANALYZE
        check_result = UserProfiles().check(self.parent.ControlsWin.userrole, action_role)

        if check_result is None:  # User check required, but no user signed in
            Log.w(
                f"Not signed in: User with role {action_role.name} is required to perform this action."
            )
            Log.i("Please sign in to continue.")
            self.parent.ControlsWin.set_user_profile()  # Prompt for sign-in
            check_result = UserProfiles().check(
                self.parent.ControlsWin.userrole, action_role
            )  # Check again

        if not check_result:  # Explicitly denied
            Log.w(
                f"ACTION DENIED: User with role {self.parent.ControlsWin.userrole.name} does not have permission to {action_role.name}."
            )
            Log.e(
                "Please sign in to access the Injection Force Calculator."
                if check_result is None
                else "You are not authorized to access the Injection Force Calculator."
            )
            return False if obj is None else None

        self.parent._enable_ui(False)
        self.parent.VisQAIWin.enable(False)
        self.animate_mode_highlight(self.mode_injection)
        self.splitter.replaceWidget(0, target_widget)

        # No tutorial pages specified for this mode

        return True if obj is None else None

    def _retranslate_ui(self, main_window: Any) -> None:
        """
        Updates the localized text and window properties for the main application.

        This method handles the translation of UI strings, sets the application
        window icon from the local assets, and formats the window title to
        include the current application name and version.

        Args:
            main_window (Any): The top-level QMainWindow instance to be updated.
                Expected to be a QtWidgets.QMainWindow or similar.
        """
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "high-res-qatch-logo-no-bg.png"
        )
        main_window.setWindowIcon(QtGui.QIcon(icon_path))
        app_title_full = f"{Constants.app_title} {Constants.app_version}"
        main_window.setWindowTitle(_translate("main_window_0", app_title_full))
