"""
QATCH.ui.mainWindow_ui_plots

Fixed but resizable Splitter layout with Glass Morphic styling, Kebab Menus
(using custom icons), and Integrated Glass Tabs.

Author(s)
    Alexander Ross  (alexander.ross@qatchtech.com)
    Paul MacNichol  (paul.macnichol@qatchtech.com)

Date:
    2026-05-11
"""

from __future__ import annotations

import os
import math
import time
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDesktopWidget
from pyqtgraph import GraphicsLayoutWidget

from QATCH.common.architecture import Architecture
from QATCH.core.constants import Constants
from QATCH.ui.popUp import PopUp

# ============================================================
#  PlotsWindow  (outer QMainWindow shell — API unchanged)
# ============================================================


class PlotsWindow(QtWidgets.QMainWindow):
    def __init__(self, samples=Constants.argument_default_samples):
        super().__init__()
        self.ui2 = UIPlots()
        self.ui2.setupUi(self)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        res = PopUp.question(
            self,
            Constants.app_title,
            "Are you sure you want to quit QATCH Q-1 application now?",
            True,
        )
        if res:
            QtWidgets.QApplication.quit()
        else:
            event.ignore()


# ============================================================
#  Glass Containers
# ============================================================


class GlassContainer(QtWidgets.QWidget):
    """Standard Frosted-glass card wrapping a single pyqtgraph GraphicsLayoutWidget."""

    colorRequested = QtCore.pyqtSignal()
    fullscreenRequested = QtCore.pyqtSignal()

    _R = 10.0
    _M = 3
    _HEADER_H = 28

    def __init__(
        self,
        plot_widget: GraphicsLayoutWidget,
        title: str = None,
        accent_color: QtGui.QColor = None,
        parent: QtWidgets.QWidget = None,
        show_menu: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)

        self.has_header = title is not None or show_menu

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(self._M, self._M, self._M, self._M)
        layout.setSpacing(0)

        # Custom Title Bar
        if self.has_header:
            self.header = QtWidgets.QWidget()
            self.header.setFixedHeight(self._HEADER_H)
            h_layout = QtWidgets.QHBoxLayout(self.header)
            h_layout.setContentsMargins(8, 0, 4, 0)
            h_layout.setSpacing(5)

            if title:
                lbl = QtWidgets.QLabel(title)
                lbl.setStyleSheet("color:rgba(30,40,55,200);font-size:10px;font-weight:600;")
                h_layout.addWidget(lbl, 1)
            else:
                h_layout.addStretch(1)

            if show_menu:
                self.btn_fs = QtWidgets.QToolButton()
                fs_icon_path = os.path.join(
                    Architecture.get_path(), "QATCH", "icons", "fullscreen.svg"
                )
                self.btn_fs.setIcon(QtGui.QIcon(fs_icon_path))
                self.btn_fs.setIconSize(QtCore.QSize(14, 14))
                self.btn_fs.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
                self.btn_fs.setToolTip("Toggle Fullscreen")
                self.btn_fs.setStyleSheet("""
                    QToolButton { 
                        background: transparent; border: none; 
                        padding: 4px; border-radius: 4px;
                    }
                    QToolButton:hover { 
                        background: rgba(255, 255, 255, 120); 
                    }
                """)
                self.btn_fs.clicked.connect(self.fullscreenRequested.emit)

                h_layout.addWidget(self.btn_fs)
                h_layout.addWidget(self._build_menu())

            layout.addWidget(self.header)

        layout.addWidget(plot_widget, 1)

    def _build_menu(self) -> QtWidgets.QToolButton:
        menu_btn = QtWidgets.QToolButton()

        # --- CUSTOM ICON INTEGRATION ---
        # Provide your custom hamburger menu icon at this path
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "gear.svg")

        menu_btn.setIcon(QtGui.QIcon(icon_path))
        menu_btn.setIconSize(QtCore.QSize(14, 14))
        menu_btn.setCursor(QCursor(Qt.PointingHandCursor))
        menu_btn.setToolTip("Plot Options")
        menu_btn.setStyleSheet("""
            QToolButton { 
                background: transparent; border: none; 
                padding: 4px; border-radius: 4px;
            }
            QToolButton:hover { 
                background: rgba(255, 255, 255, 120); 
            }
            QToolButton::menu-indicator { image: none; } 
        """)
        menu_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        self.plot_menu = QtWidgets.QMenu(menu_btn)
        self.plot_menu.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.plot_menu.setStyleSheet("""
            QMenu { 
                background-color: rgba(240, 246, 252, 240); 
                border: 1px solid rgba(200, 210, 220, 180); 
                border-radius: 6px; padding: 4px; 
            }
            QMenu::item { 
                padding: 6px 24px; border-radius: 4px; 
                color: rgba(30,40,55,200); font-size: 11px; font-weight: 500;
            }
            QMenu::item:selected { 
                background-color: rgba(46, 155, 218, 180); color: white; 
            }
        """)

        action_color = self.plot_menu.addAction("Change Line Colors...")
        action_fullscreen = self.plot_menu.addAction("Toggle Fullscreen")

        action_color = self.plot_menu.addAction("Change Line Colors...")
        action_color.triggered.connect(self.colorRequested.emit)

        menu_btn.setMenu(self.plot_menu)
        return menu_btn

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        rf = QtCore.QRectF(self.rect())
        path = QtGui.QPainterPath()
        path.addRoundedRect(rf, self._R, self._R)
        p.setClipPath(path)

        # ── Frosted base: near-opaque white with a breath of cool blue-white ──
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 160))  # was (255,255,255,160)
        p.fillRect(self.rect(), QtGui.QColor(228, 235, 241, 18))  # was (228,235,241,18)

        # ── Top-edge highlight — stronger shimmer for the "glass rim" look ──
        sh = QtGui.QLinearGradient(0, 0, 0, 40)
        sh.setColorAt(0, QtGui.QColor(255, 255, 255, 100))  # was 50
        sh.setColorAt(0.5, QtGui.QColor(255, 255, 255, 20))
        sh.setColorAt(1, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(sh))

        # ── Bottom-edge soft vignette — grounds the card visually ──
        vg = QtGui.QLinearGradient(0, self.height() - 30, 0, self.height())
        vg.setColorAt(0, QtGui.QColor(200, 218, 240, 0))
        vg.setColorAt(1, QtGui.QColor(200, 218, 240, 18))
        p.fillRect(self.rect(), QtGui.QBrush(vg))

        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)

        # ── Outer border: bright white rim ──
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 230), 1.0))
        p.drawRoundedRect(rf.adjusted(0.5, 0.5, -0.5, -0.5), self._R, self._R)
        # ── Inner border: subtle cool-grey inset ──
        p.setPen(QtGui.QPen(QtGui.QColor(190, 210, 235, 70), 1.0))
        p.drawRoundedRect(rf.adjusted(1.5, 1.5, -1.5, -1.5), self._R - 1.5, self._R - 1.5)

        if self.has_header:
            p.setPen(QtGui.QPen(QtGui.QColor(195, 215, 238, 70), 1.0))
            y_line = self._HEADER_H + self._M
            p.drawLine(0, y_line, self.width(), y_line)

        p.end()


class GlassTabContainer(GlassContainer):
    """Specialized Glass Container that acts as a Tab manager with Activity Indicators."""

    deviceColorRequested = QtCore.pyqtSignal(int)  # Emits the currently active device index

    def __init__(self, parent: QtWidgets.QWidget = None):
        # Initialize as a header-only glass container with no initial plot
        dummy = QtWidgets.QWidget()
        super().__init__(plot_widget=dummy, title=None, parent=parent, show_menu=False)

        self.has_header = True

        layout = self.layout()
        layout.removeWidget(dummy)
        dummy.deleteLater()

        # Build Integrated Header
        self.header = QtWidgets.QWidget()
        self.header.setFixedHeight(self._HEADER_H)
        h_layout = QtWidgets.QHBoxLayout(self.header)
        h_layout.setContentsMargins(4, 0, 4, 0)
        h_layout.setSpacing(2)

        self.btn_group = QtWidgets.QButtonGroup(self)
        self.btn_group.setExclusive(True)
        self.btn_group.idClicked.connect(self._on_tab_clicked)

        self.tabs_layout = QtWidgets.QHBoxLayout()
        self.tabs_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs_layout.setSpacing(4)

        h_layout.addLayout(self.tabs_layout)
        h_layout.addStretch(1)

        # --- FIX: Dedicated Fullscreen Button for the Tab Container ---
        self.btn_fs = QtWidgets.QToolButton()
        fs_icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "fullscreen.svg")
        self.btn_fs.setIcon(QtGui.QIcon(fs_icon_path))
        self.btn_fs.setIconSize(QtCore.QSize(14, 14))
        self.btn_fs.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_fs.setToolTip("Toggle Fullscreen")
        self.btn_fs.setStyleSheet("""
            QToolButton { 
                background: transparent; border: none; 
                padding: 4px; border-radius: 4px;
            }
            QToolButton:hover { 
                background: rgba(255, 255, 255, 120); 
            }
        """)
        self.btn_fs.clicked.connect(self.fullscreenRequested.emit)
        h_layout.addWidget(self.btn_fs)

        menu_btn = self._build_menu()
        self.plot_menu.actions()[0].disconnect()
        self.plot_menu.actions()[0].triggered.connect(
            lambda: self.deviceColorRequested.emit(self.stack.currentIndex())
        )
        h_layout.addWidget(menu_btn)

        layout.insertWidget(0, self.header)

        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 1)

        # --- Activity Indicator State Management ---
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.timeout.connect(self._animate_icons)
        self._device_states = {}  # Tracks state per device index

        # Define exact colors for states
        self._state_colors = {
            "idle": QtGui.QColor(220, 53, 69),  # Static Red
            "error": QtGui.QColor(220, 53, 69),  # Static Red (Failed Calibration)
            "init": QtGui.QColor(255, 193, 7),  # Blinking Yellow
            "success": QtGui.QColor(40, 167, 69),  # Static Green (Calibration Success)
            "recording": QtGui.QColor(40, 167, 69),  # Breathing Green
        }

    def _create_dot_icon(self, color: QtGui.QColor) -> QtGui.QIcon:
        """Draws a crisp dot icon for the tab buttons."""
        pixmap = QtGui.QPixmap(16, 16)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setBrush(color)
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(4, 4, 8, 8)
        painter.end()
        return QtGui.QIcon(pixmap)

    def add_device(
        self, name: str, plot_widget: GraphicsLayoutWidget, accent_color: QtGui.QColor = None
    ) -> int:
        index = self.stack.addWidget(plot_widget)

        btn = QtWidgets.QPushButton(name)
        btn.setCheckable(True)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        btn.setStyleSheet("""
            QPushButton {
                background: transparent; border: none; border-radius: 4px;
                color: rgba(30, 40, 55, 160); font-size: 11px; font-weight: 600;
                padding: 4px 8px; text-align: left;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 120);
            }
            QPushButton:checked {
                background: rgba(255, 255, 255, 200);
                color: rgba(30, 40, 55, 220);
                border: 1px solid rgba(255, 255, 255, 255);
            }
        """)

        self.btn_group.addButton(btn, index)
        self.tabs_layout.addWidget(btn)

        if index == 0:
            btn.setChecked(True)

        # Initialize new device strictly to the "idle" state
        self._device_states[index] = "idle"
        self._update_button_icon(index)

        return index

    def set_device_state(self, index: int, state: str) -> None:
        """
        Sets the activity state for a device tab.
        Valid states: 'idle', 'error', 'init', 'success', 'recording'.
        """
        if index not in self._device_states or state not in self._state_colors:
            return

        self._device_states[index] = state

        # Manage global animation timer if ANY device needs to blink or breathe
        needs_anim = any(s in ("init", "recording") for s in self._device_states.values())
        if needs_anim and not self._anim_timer.isActive():
            self._anim_timer.start(50)  # 50ms tick rate for smooth 60fps breathing
        elif not needs_anim and self._anim_timer.isActive():
            self._anim_timer.stop()

        self._update_button_icon(index)

    def _animate_icons(self) -> None:
        """Called every 50ms to push the next animation frame to active icons."""
        for idx, state in self._device_states.items():
            if state in ("init", "recording"):
                self._update_button_icon(idx)

    def _toggle_blink(self) -> None:
        self._blink_state = not self._blink_state
        for idx, state in self._device_states.items():
            if state == "init":
                self._update_button_icon(idx)

    def _update_button_icon(self, index: int) -> None:
        state = self._device_states.get(index, "idle")
        btn = self.btn_group.button(index)
        if not btn:
            return

        color = QtGui.QColor(self._state_colors[state])

        if state == "init":
            # Hard blink: toggles alpha every ~500ms using modulo
            alpha = 255 if (int(time.time() * 2) % 2 == 0) else 60
            color.setAlpha(alpha)
        elif state == "recording":
            # Smooth breathing: sine wave over time (multiplier adjusts breath speed)
            alpha = int(140 + 115 * math.sin(time.time() * 3.5))
            color.setAlpha(alpha)

        btn.setIcon(self._create_dot_icon(color))

    def _on_tab_clicked(self, index: int) -> None:
        self.stack.setCurrentIndex(index)


# ============================================================
#  Shared helpers
# ============================================================


def _make_plot_widget(parent: QtWidgets.QWidget) -> GraphicsLayoutWidget:
    w = GraphicsLayoutWidget(parent)
    w.setAutoFillBackground(False)
    w.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
    w.setBackground(None)
    w.setStyleSheet("border:0px; background: transparent;")
    w.setFrameShape(QtWidgets.QFrame.NoFrame)
    w.setFrameShadow(QtWidgets.QFrame.Plain)
    w.setLineWidth(0)
    w.setMinimumSize(80, 60)
    return w


# ============================================================
#  UIPlots  — consumed by PlotsWindow
# ============================================================


class UIPlots:

    def setupUi(self, MainWindow2: QtWidgets.QMainWindow) -> None:
        USE_FULLSCREEN = QDesktopWidget().availableGeometry().width() == 2880

        MainWindow2.setObjectName("MainWindow2")
        MainWindow2.setMinimumSize(QtCore.QSize(1000, 250))
        if USE_FULLSCREEN:
            MainWindow2.resize(1701, 1435)
            MainWindow2.move(0, 0)
        else:
            MainWindow2.move(692, 0)
        MainWindow2.setStyleSheet("")
        MainWindow2.setTabShape(QtWidgets.QTabWidget.Rounded)

        self._fullscreen_active_widget = None

        # Central widget ─────────────────────────────────────────────
        self.centralwidget = QtWidgets.QWidget(MainWindow2)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)

        root = QtWidgets.QVBoxLayout(self.centralwidget)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Main Horizontal Splitter ───────────────────────────────────
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self.centralwidget)
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.setStyleSheet("QSplitter::handle { background: transparent; }")

        # --- LEFT SIDE: Integrated Glass Tabs ---
        self.left_pane = GlassTabContainer(parent=self.main_splitter)

        self.pltB = _make_plot_widget(self.left_pane)
        self.pltB.setObjectName("pltB")

        # Add the first device. Future devices can be appended identically.
        self.left_pane.add_device("Device 1", self.pltB, QtGui.QColor(72, 190, 120))

        # --- RIGHT SIDE: Stacked Splitter for Amp & Temp ---
        self.right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self.main_splitter)
        self.right_splitter.setHandleWidth(6)
        self.right_splitter.setStyleSheet("QSplitter::handle { background: transparent; }")

        # Amplitude Group
        self.plt = _make_plot_widget(self.right_splitter)
        self.plt.setObjectName("plt")
        self.amp_glass = GlassContainer(
            plot_widget=self.plt,
            title="Amplitude (dB)",
            accent_color=QtGui.QColor(46, 155, 218),
            parent=self.right_splitter,
        )
        self.right_splitter.addWidget(self.amp_glass)

        # Temperature Group
        self.plt_temp = _make_plot_widget(self.right_splitter)
        self.plt_temp.setObjectName("plt_temp")
        self.temp_glass = GlassContainer(
            plot_widget=self.plt_temp,
            title="Temperature °C",
            accent_color=QtGui.QColor(240, 156, 53),  # was QColor(240, 156, 53)
            parent=self.right_splitter,
        )
        self.right_splitter.addWidget(self.temp_glass)

        # Connect Fullscreen Actions
        self.left_pane.fullscreenRequested.connect(lambda: self._toggle_fullscreen(self.left_pane))
        self.amp_glass.fullscreenRequested.connect(lambda: self._toggle_fullscreen(self.amp_glass))
        self.temp_glass.fullscreenRequested.connect(
            lambda: self._toggle_fullscreen(self.temp_glass)
        )

        # Build final view hierarchy ─────────────────────────────────
        root.addWidget(self.main_splitter)
        MainWindow2.setCentralWidget(self.centralwidget)

        # Set Splitter Ratios
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 1)
        self.right_splitter.setStretchFactor(0, 1)
        self.right_splitter.setStretchFactor(1, 1)

        # Legacy stubs
        self.Layout_graphs = QtWidgets.QWidget()
        self.btnCollapse = QtWidgets.QToolButton()
        self.btnExpand = QtWidgets.QToolButton()

        self.retranslateUi(MainWindow2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow2)

    def _toggle_fullscreen(self, target_widget: QtWidgets.QWidget) -> None:
        """Animates expansion of the target_widget to take up the full layout space and toggles the icon."""

        # Define icon paths
        expand_icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "fullscreen.svg")
        contract_icon_path = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "fullscreen-exit.svg"
        )

        if getattr(self, "_fullscreen_active_widget", None) == target_widget:
            # Revert to normal sizes
            target_main = self._normal_main_sizes
            target_right = self._normal_right_sizes

            # Change icon back to expand
            target_widget.btn_fs.setIcon(QtGui.QIcon(expand_icon_path))
            target_widget.btn_fs.setToolTip("Toggle Fullscreen")

            self._fullscreen_active_widget = None
        else:
            # Maximize
            if getattr(self, "_fullscreen_active_widget", None) is None:
                # Save normal sizes only if we aren't currently in fullscreen
                self._normal_main_sizes = self.main_splitter.sizes()
                self._normal_right_sizes = self.right_splitter.sizes()
            else:
                # Reset the previously expanded widget's icon just in case
                self._fullscreen_active_widget.btn_fs.setIcon(QtGui.QIcon(expand_icon_path))

            self._fullscreen_active_widget = target_widget

            # Change icon to contract
            target_widget.btn_fs.setIcon(QtGui.QIcon(contract_icon_path))
            target_widget.btn_fs.setToolTip("Restore Size")

            total_main = sum(self.main_splitter.sizes())
            total_right = sum(self.right_splitter.sizes())

            if target_widget == self.left_pane:
                target_main = [total_main, 0]
                target_right = self._normal_right_sizes  # Main splitter covers it
            elif target_widget == self.amp_glass:
                target_main = [0, total_main]
                target_right = [total_right, 0]
            elif target_widget == self.temp_glass:
                target_main = [0, total_main]
                target_right = [0, total_right]

        # Stop existing animation if one is currently mid-run
        if hasattr(self, "_fs_anim") and self._fs_anim.state() == QtCore.QAbstractAnimation.Running:
            self._fs_anim.stop()

        # Build QVariantAnimation to transition the QSplitter sizes
        self._fs_anim = QtCore.QVariantAnimation()
        self._fs_anim.setDuration(400)  # Length in milliseconds
        self._fs_anim.setEasingCurve(QtCore.QEasingCurve.InOutCubic)  # Smooth start and stop
        self._fs_anim.setStartValue(0.0)
        self._fs_anim.setEndValue(1.0)

        start_main = self.main_splitter.sizes()
        start_right = self.right_splitter.sizes()

        def update_sizes(val):
            cur_main = [int(s + (e - s) * val) for s, e in zip(start_main, target_main)]
            cur_right = [int(s + (e - s) * val) for s, e in zip(start_right, target_right)]
            self.main_splitter.setSizes(cur_main)
            self.right_splitter.setSizes(cur_right)

        self._fs_anim.valueChanged.connect(update_sizes)
        self._fs_anim.start()

    # ────────────────────────────────────────────────────────────────

    def retranslateUi(self, MainWindow2: QtWidgets.QMainWindow) -> None:
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/qatch-icon.png")
        MainWindow2.setWindowIcon(QtGui.QIcon(icon_path))
        MainWindow2.setWindowTitle(
            _translate(
                "MainWindow2", "{} {} - Plots".format(Constants.app_title, Constants.app_version)
            )
        )

    def Ui(self, MainWindow2: QtWidgets.QMainWindow) -> None:
        self.retranslateUi(MainWindow2)

    # Legacy no-op shims
    def handleSplitterMoved(self, pos: int = 0, index: int = 0) -> None:
        pass

    def handleSplitterButton(self, collapse: bool = True) -> None:
        pass
