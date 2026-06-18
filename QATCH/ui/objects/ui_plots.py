"""
QATCH.ui.mainWindow_ui_plots

Frosted-glass plot layout with:
  - Smooth fullscreen expand/contract animation (viewport-freeze prevents chop)
  - Glassmorphic dropdown menus with per-section color + visibility controls
  - Circular hover effects on all header buttons
  - Glassmorphic device-selection tabs

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
from typing import List, Tuple

from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDesktopWidget, QColorDialog
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
#  _SectionMenuRow  — one row inside the glass dropdown
# ============================================================


class _SectionMenuRow(QtWidgets.QWidget):
    """
    A compact interactive row: [color-swatch] [label ···] [show/hide toggle]
    Used as a QWidgetAction payload inside each plot's option menu.
    """

    colorChanged = QtCore.pyqtSignal(str, QtGui.QColor)  # (key, new_color)
    visibilityChanged = QtCore.pyqtSignal(str, bool)  # (key, is_visible)

    def __init__(
        self,
        key: str,
        label: str,
        color: QtGui.QColor,
        parent: QtWidgets.QWidget = None,
    ) -> None:
        super().__init__(parent)
        self._key = key
        self._color = QtGui.QColor(color)
        self._visible = True

        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setMinimumWidth(190)
        self.setFixedHeight(34)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(8)

        # ── Color swatch ──────────────────────────────────────────
        self._swatch = QtWidgets.QToolButton()
        self._swatch.setFixedSize(16, 16)
        self._swatch.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self._swatch.setToolTip("Change color")
        self._swatch.clicked.connect(self._pick_color)
        self._apply_swatch_style()

        # ── Label ─────────────────────────────────────────────────
        self._lbl = QtWidgets.QLabel(label)
        self._lbl.setStyleSheet(
            "color: rgba(30,40,55,210); font-size: 11px;"
            " font-weight: 500; background: transparent;"
        )

        # ── Show / Hide toggle ────────────────────────────────────
        self._eye = QtWidgets.QToolButton()
        self._eye.setFixedSize(22, 22)
        self._eye.setCheckable(True)
        self._eye.setChecked(True)
        self._eye.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self._eye.setToolTip("Show / Hide")
        self._eye.clicked.connect(self._toggle_visibility)
        self._apply_eye_style()

        lay.addWidget(self._swatch)
        lay.addWidget(self._lbl, 1)
        lay.addWidget(self._eye)

    # ── Row hover ─────────────────────────────────────────────────
    def enterEvent(self, _event):
        self.setStyleSheet("background: rgba(46,155,218,28); border-radius: 6px;")

    def leaveEvent(self, _event):
        self.setStyleSheet("background: transparent;")

    # ── Style helpers ─────────────────────────────────────────────
    def _apply_swatch_style(self) -> None:
        c = self._color
        self._swatch.setStyleSheet(f"""
            QToolButton {{
                background-color: rgb({c.red()},{c.green()},{c.blue()});
                border: 1.5px solid rgba(255,255,255,210);
                border-radius: 8px;
            }}
            QToolButton:hover {{
                border: 2px solid rgba(255,255,255,255);
            }}
        """)

    def _apply_eye_style(self) -> None:
        symbol = "◉" if self._visible else "○"
        alpha = 200 if self._visible else 80
        self._eye.setText(symbol)
        self._eye.setStyleSheet(f"""
            QToolButton {{
                background: transparent; border: none;
                color: rgba(30,40,55,{alpha}); font-size: 13px;
                border-radius: 4px;
            }}
            QToolButton:hover {{ background: rgba(255,255,255,130); }}
        """)

    # ── Actions ───────────────────────────────────────────────────
    def _pick_color(self) -> None:
        # Close the parent menu so the dialog appears front-and-centre
        menu = self._find_parent_menu()
        if menu:
            menu.close()
        color = QColorDialog.getColor(self._color, None, "Choose Color")
        if color.isValid():
            self._color = color
            self._apply_swatch_style()
            self.colorChanged.emit(self._key, self._color)

    def _toggle_visibility(self, checked: bool) -> None:
        self._visible = checked
        self._apply_eye_style()
        self.visibilityChanged.emit(self._key, checked)

    def _find_parent_menu(self) -> QtWidgets.QMenu | None:
        p = self.parentWidget()
        while p:
            if isinstance(p, QtWidgets.QMenu):
                return p
            p = p.parentWidget()
        return None


# ============================================================
#  _SectionHeaderRow  — non-interactive section label
# ============================================================


class _SectionHeaderRow(QtWidgets.QWidget):
    def __init__(self, text: str, parent: QtWidgets.QWidget = None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setFixedHeight(26)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 10, 0)
        lbl = QtWidgets.QLabel(text.upper())
        lbl.setStyleSheet(
            "color: rgba(30,40,55,110); font-size: 8px; font-weight: 700;"
            " letter-spacing: 1.2px; background: transparent;"
        )
        lay.addWidget(lbl)


# ============================================================
#  GlassContainer
# ============================================================


class GlassContainer(QtWidgets.QWidget):
    """Standard Frosted-glass card wrapping a single pyqtgraph GraphicsLayoutWidget."""

    # Legacy signal (kept for compatibility)
    colorRequested = QtCore.pyqtSignal()
    fullscreenRequested = QtCore.pyqtSignal()

    # Per-section signals
    sectionColorChanged = QtCore.pyqtSignal(str, QtGui.QColor)  # (key, color)
    sectionVisibilityChanged = QtCore.pyqtSignal(str, bool)  # (key, visible)

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
        sections: List[Tuple[str, str, QtGui.QColor]] = None,
    ) -> None:
        super().__init__(parent)
        # sections: list of (key, label, default_color)
        self._sections = sections or []

        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)

        self.has_header = title is not None or show_menu

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(self._M, self._M, self._M, self._M)
        layout.setSpacing(0)

        if self.has_header:
            self.header = QtWidgets.QWidget()
            self.header.setFixedHeight(self._HEADER_H)
            h_layout = QtWidgets.QHBoxLayout(self.header)
            h_layout.setContentsMargins(8, 0, 4, 0)
            h_layout.setSpacing(4)

            if title:
                lbl = QtWidgets.QLabel(title)
                lbl.setStyleSheet("color:rgba(30,40,55,200);font-size:10px;font-weight:600;")
                h_layout.addWidget(lbl, 1)
            else:
                h_layout.addStretch(1)

            if show_menu:
                # Fullscreen button (circular hover)
                self.btn_fs = self._make_icon_button("fullscreen.svg", "Toggle Fullscreen")
                self.btn_fs.clicked.connect(self.fullscreenRequested.emit)
                h_layout.addWidget(self.btn_fs)

                # Gear / options button
                h_layout.addWidget(self._build_menu())

            layout.addWidget(self.header)

        layout.addWidget(plot_widget, 1)

    # ── Round icon button factory (circular hover) ────────────────
    def _make_icon_button(self, icon_name: str, tooltip: str) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton()
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", icon_name)
        btn.setIcon(QtGui.QIcon(icon_path))
        btn.setIconSize(QtCore.QSize(13, 13))
        btn.setFixedSize(24, 24)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        btn.setToolTip(tooltip)
        btn.setStyleSheet("""
            QToolButton {
                background: transparent;
                border: none;
                border-radius: 12px;
            }
            QToolButton:hover {
                background: rgba(255, 255, 255, 160);
                border: 1px solid rgba(255, 255, 255, 200);
            }
            QToolButton:pressed {
                background: rgba(180, 215, 255, 190);
            }
        """)
        return btn

    # ── Menu builder ──────────────────────────────────────────────
    def _build_menu(self) -> QtWidgets.QToolButton:
        menu_btn = self._make_icon_button("gear.svg", "Plot Options")
        menu_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        # Remove the default dropdown arrow
        menu_btn.setStyleSheet(
            menu_btn.styleSheet() + "\nQToolButton::menu-indicator { image: none; }"
        )

        self.plot_menu = self._build_glass_menu(menu_btn)
        menu_btn.setMenu(self.plot_menu)
        return menu_btn

    def _build_glass_menu(self, parent_widget: QtWidgets.QWidget) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(parent_widget)
        menu.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        menu.setWindowFlags(
            menu.windowFlags() | QtCore.Qt.FramelessWindowHint | QtCore.Qt.NoDropShadowWindowHint
        )
        menu.setStyleSheet("""
            QMenu {
                background-color: rgba(232, 242, 252, 248);
                border: 1px solid rgba(255, 255, 255, 230);
                border-radius: 10px;
                padding: 5px 0px;
            }
            QMenu::item {
                padding: 0px;
                margin: 0px;
            }
            QMenu::separator {
                height: 1px;
                background: rgba(175, 200, 228, 90);
                margin: 4px 12px;
            }
        """)

        if self._sections:
            for i, (key, label, color) in enumerate(self._sections):
                row = _SectionMenuRow(key, label, color)
                row.colorChanged.connect(self.sectionColorChanged)
                row.visibilityChanged.connect(self.sectionVisibilityChanged)

                wa = QtWidgets.QWidgetAction(menu)
                wa.setDefaultWidget(row)
                menu.addAction(wa)

                if i < len(self._sections) - 1:
                    menu.addSeparator()
        else:
            # Fallback: legacy colour action
            action_color = menu.addAction("Change Line Colors…")
            action_color.triggered.connect(self.colorRequested.emit)

        return menu

    # ── Paint — frosted glass card ────────────────────────────────
    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        rf = QtCore.QRectF(self.rect())
        path = QtGui.QPainterPath()
        path.addRoundedRect(rf, self._R, self._R)
        p.setClipPath(path)

        # Frosted base
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 160))
        p.fillRect(self.rect(), QtGui.QColor(228, 235, 241, 18))

        # Top shimmer
        sh = QtGui.QLinearGradient(0, 0, 0, 40)
        sh.setColorAt(0, QtGui.QColor(255, 255, 255, 100))
        sh.setColorAt(0.5, QtGui.QColor(255, 255, 255, 20))
        sh.setColorAt(1, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(sh))

        # Bottom vignette
        vg = QtGui.QLinearGradient(0, self.height() - 30, 0, self.height())
        vg.setColorAt(0, QtGui.QColor(200, 218, 240, 0))
        vg.setColorAt(1, QtGui.QColor(200, 218, 240, 18))
        p.fillRect(self.rect(), QtGui.QBrush(vg))

        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)

        # Outer bright rim
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 230), 1.0))
        p.drawRoundedRect(rf.adjusted(0.5, 0.5, -0.5, -0.5), self._R, self._R)
        # Inner cool-grey inset
        p.setPen(QtGui.QPen(QtGui.QColor(190, 210, 235, 70), 1.0))
        p.drawRoundedRect(rf.adjusted(1.5, 1.5, -1.5, -1.5), self._R - 1.5, self._R - 1.5)

        if self.has_header:
            p.setPen(QtGui.QPen(QtGui.QColor(195, 215, 238, 70), 1.0))
            y_line = self._HEADER_H + self._M
            p.drawLine(0, y_line, self.width(), y_line)

        p.end()


# ============================================================
#  GlassTabContainer
# ============================================================


class GlassTabContainer(GlassContainer):
    """Specialized Glass Container that manages device tabs with activity indicators."""

    deviceColorRequested = QtCore.pyqtSignal(int)  # Legacy: emits active device index

    # Sections shown in the main plot's dropdown
    _MAIN_SECTIONS: List[Tuple[str, str, QtGui.QColor]] = [
        ("dissipation", "Dissipation", QtGui.QColor(46, 155, 218)),
        ("resonance_freq", "Resonance Frequency", QtGui.QColor(240, 100, 53)),
    ]

    def __init__(self, parent: QtWidgets.QWidget = None) -> None:
        dummy = QtWidgets.QWidget()
        # Pass sections so _build_glass_menu picks them up
        super().__init__(
            plot_widget=dummy,
            title=None,
            parent=parent,
            show_menu=False,
            sections=self._MAIN_SECTIONS,
        )

        self.has_header = True

        layout = self.layout()
        layout.removeWidget(dummy)
        dummy.deleteLater()

        # ── Integrated header ──────────────────────────────────────
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

        # Fullscreen button (circular)
        self.btn_fs = self._make_icon_button("fullscreen.svg", "Toggle Fullscreen")
        self.btn_fs.clicked.connect(self.fullscreenRequested.emit)
        h_layout.addWidget(self.btn_fs)

        # Gear menu (uses self._sections = _MAIN_SECTIONS)
        h_layout.addWidget(self._build_menu())

        layout.insertWidget(0, self.header)

        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 1)

        # ── Activity indicator state management ────────────────────
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.timeout.connect(self._animate_icons)
        self._device_states: dict[int, str] = {}

        self._state_colors = {
            "idle": QtGui.QColor(220, 53, 69),
            "error": QtGui.QColor(220, 53, 69),
            "init": QtGui.QColor(255, 193, 7),
            "success": QtGui.QColor(40, 167, 69),
            "recording": QtGui.QColor(40, 167, 69),
        }

    # ── Tab creation ──────────────────────────────────────────────
    def add_device(
        self,
        name: str,
        plot_widget: GraphicsLayoutWidget,
        accent_color: QtGui.QColor = None,
    ) -> int:
        index = self.stack.addWidget(plot_widget)

        btn = QtWidgets.QPushButton(name)
        btn.setCheckable(True)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        btn.setFixedHeight(20)
        # Glassmorphic pill tabs
        btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 55);
                border: 1px solid rgba(255, 255, 255, 110);
                border-radius: 10px;
                color: rgba(30, 40, 55, 155);
                font-size: 10px;
                font-weight: 600;
                padding: 2px 10px;
                text-align: center;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 130);
                border: 1px solid rgba(255, 255, 255, 200);
                color: rgba(30, 40, 55, 200);
            }
            QPushButton:checked {
                background: rgba(255, 255, 255, 215);
                border: 1.5px solid rgba(255, 255, 255, 255);
                color: rgba(30, 40, 55, 235);
            }
            QPushButton:checked:hover {
                background: rgba(255, 255, 255, 240);
            }
        """)

        self.btn_group.addButton(btn, index)
        self.tabs_layout.addWidget(btn)

        if index == 0:
            btn.setChecked(True)

        self._device_states[index] = "idle"
        self._update_button_icon(index)
        return index

    # ── Activity indicator API ────────────────────────────────────
    def set_device_state(self, index: int, state: str) -> None:
        """
        Sets the activity state for a device tab.
        Valid states: 'idle', 'error', 'init', 'success', 'recording'.
        """
        if index not in self._device_states or state not in self._state_colors:
            return

        self._device_states[index] = state

        needs_anim = any(s in ("init", "recording") for s in self._device_states.values())
        if needs_anim and not self._anim_timer.isActive():
            self._anim_timer.start(50)
        elif not needs_anim and self._anim_timer.isActive():
            self._anim_timer.stop()

        self._update_button_icon(index)

    def _animate_icons(self) -> None:
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
            alpha = 255 if (int(time.time() * 2) % 2 == 0) else 60
            color.setAlpha(alpha)
        elif state == "recording":
            alpha = int(140 + 115 * math.sin(time.time() * 3.5))
            color.setAlpha(alpha)

        btn.setIcon(self._create_dot_icon(color))

    def _create_dot_icon(self, color: QtGui.QColor) -> QtGui.QIcon:
        pixmap = QtGui.QPixmap(16, 16)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setBrush(color)
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(4, 4, 8, 8)
        painter.end()
        return QtGui.QIcon(pixmap)

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

        # Central widget ──────────────────────────────────────────────
        self.centralwidget = QtWidgets.QWidget(MainWindow2)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)

        root = QtWidgets.QVBoxLayout(self.centralwidget)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(6)

        # Main horizontal splitter ────────────────────────────────────
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self.centralwidget)
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.setStyleSheet("QSplitter::handle { background: transparent; }")

        # LEFT — Integrated glass tabs (Dissipation / Resonance Freq)
        self.left_pane = GlassTabContainer(parent=self.main_splitter)
        self.pltB = _make_plot_widget(self.left_pane)
        self.pltB.setObjectName("pltB")
        self.left_pane.add_device("Device 1", self.pltB, QtGui.QColor(72, 190, 120))

        # RIGHT — stacked splitter for Amp & Temp
        self.right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self.main_splitter)
        self.right_splitter.setHandleWidth(6)
        self.right_splitter.setStyleSheet("QSplitter::handle { background: transparent; }")

        # Amplitude pane
        self.plt = _make_plot_widget(self.right_splitter)
        self.plt.setObjectName("plt")
        self.amp_glass = GlassContainer(
            plot_widget=self.plt,
            title="Amplitude (dB)",
            accent_color=QtGui.QColor(46, 155, 218),
            parent=self.right_splitter,
            sections=[
                ("amplitude", "Amplitude", QtGui.QColor(46, 155, 218)),
            ],
        )
        self.right_splitter.addWidget(self.amp_glass)

        # Temperature pane
        self.plt_temp = _make_plot_widget(self.right_splitter)
        self.plt_temp.setObjectName("plt_temp")
        self.temp_glass = GlassContainer(
            plot_widget=self.plt_temp,
            title="Temperature °C",
            accent_color=QtGui.QColor(240, 156, 53),
            parent=self.right_splitter,
            sections=[
                ("temperature", "Temperature", QtGui.QColor(240, 156, 53)),
            ],
        )
        self.right_splitter.addWidget(self.temp_glass)

        # Connect fullscreen signals
        self.left_pane.fullscreenRequested.connect(lambda: self._toggle_fullscreen(self.left_pane))
        self.amp_glass.fullscreenRequested.connect(lambda: self._toggle_fullscreen(self.amp_glass))
        self.temp_glass.fullscreenRequested.connect(
            lambda: self._toggle_fullscreen(self.temp_glass)
        )

        # Build final hierarchy ────────────────────────────────────────
        root.addWidget(self.main_splitter)
        MainWindow2.setCentralWidget(self.centralwidget)

        # Splitter ratios
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

    # ── Smooth fullscreen toggle ──────────────────────────────────
    def _toggle_fullscreen(self, target_widget: QtWidgets.QWidget) -> None:
        """
        Animates the splitter sizes to expand/contract the target pane.

        Smoothness strategy
        ───────────────────
        1. A QTimer fires at ~60 fps (every 16 ms) — finer-grained than
           QVariantAnimation's default repaint scheduling.
        2. Before each animation, viewport updates are frozen on all three
           pyqtgraph widgets so OpenGL/QPainter doesn't redraw mid-resize.
        3. A quartic ease-in-out gives a natural, snappy feel.
        4. Viewport updates are restored (and a single repaint triggered)
           the moment the animation completes.
        """
        expand_icon = os.path.join(Architecture.get_path(), "QATCH", "icons", "fullscreen.svg")
        contract_icon = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "fullscreen-exit.svg"
        )

        if getattr(self, "_fullscreen_active_widget", None) == target_widget:
            # ── Contracting back to normal ──
            target_main = self._normal_main_sizes
            target_right = self._normal_right_sizes
            target_widget.btn_fs.setIcon(QtGui.QIcon(expand_icon))
            target_widget.btn_fs.setToolTip("Toggle Fullscreen")
            self._fullscreen_active_widget = None
        else:
            # ── Expanding ──
            if getattr(self, "_fullscreen_active_widget", None) is None:
                self._normal_main_sizes = self.main_splitter.sizes()
                self._normal_right_sizes = self.right_splitter.sizes()
            else:
                # Reset icon on the previously-expanded pane
                self._fullscreen_active_widget.btn_fs.setIcon(QtGui.QIcon(expand_icon))

            self._fullscreen_active_widget = target_widget
            target_widget.btn_fs.setIcon(QtGui.QIcon(contract_icon))
            target_widget.btn_fs.setToolTip("Restore Size")

            total_main = sum(self.main_splitter.sizes())
            total_right = sum(self.right_splitter.sizes())

            if target_widget is self.left_pane:
                target_main = [total_main, 0]
                target_right = self._normal_right_sizes
            elif target_widget is self.amp_glass:
                target_main = [0, total_main]
                target_right = [total_right, 0]
            elif target_widget is self.temp_glass:
                target_main = [0, total_main]
                target_right = [0, total_right]
            else:
                return

        # Stop any running animation
        if hasattr(self, "_fs_timer") and self._fs_timer.isActive():
            self._fs_timer.stop()
            self._freeze_plots(False)  # Ensure plots are unfrozen

        # Freeze plots so they don't stutter during resize
        self._freeze_plots(True)

        # Animation parameters
        DURATION_MS = 380
        INTERVAL_MS = 16  # ~60 fps
        total_steps = max(1, DURATION_MS // INTERVAL_MS)

        start_main = list(self.main_splitter.sizes())
        start_right = list(self.right_splitter.sizes())
        step_counter = [0]

        def _ease_in_out_quart(t: float) -> float:
            """Quartic ease — snappy acceleration, smooth landing."""
            if t < 0.5:
                return 8.0 * t * t * t * t
            t -= 1.0
            return 1.0 - 8.0 * t * t * t * t

        def _tick() -> None:
            step_counter[0] += 1
            raw_t = step_counter[0] / total_steps
            clamped = min(raw_t, 1.0)
            val = _ease_in_out_quart(clamped)

            new_main = [int(s + (e - s) * val) for s, e in zip(start_main, target_main)]
            new_right = [int(s + (e - s) * val) for s, e in zip(start_right, target_right)]
            self.main_splitter.setSizes(new_main)
            self.right_splitter.setSizes(new_right)

            if clamped >= 1.0:
                self._fs_timer.stop()
                # Unfreeze and repaint now that animation is done
                self._freeze_plots(False)

        self._fs_timer = QtCore.QTimer(self.centralwidget)
        self._fs_timer.setInterval(INTERVAL_MS)
        self._fs_timer.timeout.connect(_tick)
        self._fs_timer.start()

    def _freeze_plots(self, freeze: bool) -> None:
        """
        Suspend / resume pyqtgraph viewport redraws.
        GraphicsLayoutWidget extends QGraphicsView, so we can swap its
        viewport update mode to prevent costly OpenGL repaints on every
        setSizes() call during the animation.
        """
        from PyQt5.QtWidgets import QGraphicsView

        mode = QGraphicsView.NoViewportUpdate if freeze else QGraphicsView.MinimalViewportUpdate
        for attr in ("plt", "plt_temp", "pltB"):
            w = getattr(self, attr, None)
            if w is None:
                continue
            try:
                w.setViewportUpdateMode(mode)
                if not freeze:
                    w.viewport().update()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────

    def retranslateUi(self, MainWindow2: QtWidgets.QMainWindow) -> None:
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/qatch-icon.png")
        MainWindow2.setWindowIcon(QtGui.QIcon(icon_path))
        MainWindow2.setWindowTitle(
            _translate(
                "MainWindow2",
                "{} {} - Plots".format(Constants.app_title, Constants.app_version),
            )
        )

    def Ui(self, MainWindow2: QtWidgets.QMainWindow) -> None:
        self.retranslateUi(MainWindow2)

    # Legacy no-op shims
    def handleSplitterMoved(self, pos: int = 0, index: int = 0) -> None:
        pass

    def handleSplitterButton(self, collapse: bool = True) -> None:
        pass
