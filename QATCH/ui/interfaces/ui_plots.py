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

import math
import os
import time
from typing import List, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QColorDialog, QDesktopWidget
from pyqtgraph import GraphicsLayoutWidget

from QATCH.common.architecture import Architecture
from QATCH.core.constants import Constants
from QATCH.ui.components.update_status_icon import UpdateStatusIcon
from QATCH.ui.styles.theme_manager import ThemeManager, ThemeMode, tok_css
from QATCH.ui.styles.tokens import PALETTES


class PlotMenuRow(QtWidgets.QWidget):
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
        self._icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("PlotMenuRow")
        self.setMinimumWidth(190)
        self.setFixedHeight(34)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(8)

        # ── Color swatch ──────────────────────────────────────────
        self._swatch = QtWidgets.QToolButton()
        self._swatch.setFixedSize(20, 20)
        self._swatch.setIconSize(QtCore.QSize(16, 16))
        self._swatch.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._swatch.setToolTip("Change color")
        self._swatch.clicked.connect(self._pick_color)
        self._apply_swatch_style()

        # ── Label ─────────────────────────────────────────────────
        self._lbl = QtWidgets.QLabel(label)
        self._lbl.setObjectName("PlotMenuItemLabel")

        # ── Show / Hide toggle ────────────────────────────────────
        self._eye = QtWidgets.QToolButton()
        self._eye.setFixedSize(22, 22)
        self._eye.setIconSize(QtCore.QSize(16, 16))
        self._eye.setCheckable(True)
        self._eye.setChecked(True)
        self._eye.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._eye.setToolTip("Show / Hide")
        self._eye.clicked.connect(self._toggle_visibility)
        self._apply_eye_style()

        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

        lay.addWidget(self._swatch)
        lay.addWidget(self._lbl, 1)
        lay.addWidget(self._eye)

    # ── Style helpers ─────────────────────────────────────────────
    def _apply_swatch_style(self) -> None:
        path = os.path.join(self._icons_dir, "color-mode.svg")
        self._swatch.setIcon(PlotContainer._tinted_icon(path, self._color, size=16))
        tok = ThemeManager.instance().tokens()
        br = tok["plot_swatch_border"]
        border = tok_css(br)
        border_full = tok_css((*br[:3], 255))
        self._swatch.setStyleSheet(f"""
            QToolButton {{
                background: transparent;
                border: 1.5px solid {border};
                border-radius: 8px;
            }}
            QToolButton:hover {{
                border: 2px solid {border_full};
            }}
        """)

    def _apply_eye_style(self) -> None:
        icon_name = "eye-on.svg" if self._visible else "eye-off.svg"
        path = os.path.join(self._icons_dir, icon_name)
        tok = ThemeManager.instance().tokens()
        r, g, b, _ = tok["plot_text_normal"]
        tint = QtGui.QColor(r, g, b, 200 if self._visible else 80)
        self._eye.setIcon(PlotContainer._tinted_icon(path, tint, size=16))
        hover_bg = tok_css(tok["plot_tab_bg_hover"])
        self._eye.setStyleSheet(f"""
            QToolButton {{
                background: transparent; border: none;
                border-radius: 4px;
            }}
            QToolButton:hover {{ background: {hover_bg}; }}
        """)

    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_swatch_style()
        self._apply_eye_style()

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


class GridMenuRow(QtWidgets.QWidget):
    """A labeled checkbox for toggling a grid-line axis in the plot gear menu."""

    toggled = QtCore.pyqtSignal(str, bool)  # (key, visible)

    def __init__(
        self,
        key: str,
        label: str,
        parent: QtWidgets.QWidget = None,
    ) -> None:
        super().__init__(parent)
        self._key = key

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("PlotMenuRow")
        self.setMinimumWidth(190)
        self.setFixedHeight(34)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(8)

        self._checkbox = QtWidgets.QCheckBox(label)
        self._checkbox.setObjectName("PlotMenuItemLabel")
        self._checkbox.setChecked(False)  # off by default
        self._checkbox.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._checkbox.stateChanged.connect(
            lambda _: self.toggled.emit(self._key, self._checkbox.isChecked())
        )

        ThemeManager.instance().themeChanged.connect(self._apply_style)

        lay.addWidget(self._checkbox, 1)

        self._apply_style()

    def _apply_style(self, _=None) -> None:
        tok = ThemeManager.instance().tokens()
        text_css = tok_css(tok["plot_text_normal"])
        border_css = tok_css(tok["plot_swatch_border"])
        checked_css = tok_css(tok["plot_data_primary"])
        self._checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {text_css};
                spacing: 6px;
            }}
            QCheckBox::indicator {{
                width: 15px;
                height: 15px;
                border-radius: 3px;
                border: 1.5px solid {border_css};
                background: transparent;
            }}
            QCheckBox::indicator:checked {{
                background: {checked_css};
                border-color: {checked_css};
            }}
        """)


class PlotContainer(QtWidgets.QWidget):
    """Standard Frosted-glass card wrapping a single pyqtgraph GraphicsLayoutWidget."""

    # Legacy signal (kept for compatibility)
    colorRequested = QtCore.pyqtSignal()
    fullscreenRequested = QtCore.pyqtSignal()

    # Per-section signals
    sectionColorChanged = QtCore.pyqtSignal(str, QtGui.QColor)  # (key, color)
    sectionVisibilityChanged = QtCore.pyqtSignal(str, bool)  # (key, visible)
    gridChanged = QtCore.pyqtSignal(str, bool)  # (key, visible)

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
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._is_fullscreen = False
        self._preload_icons()

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
                lbl.setObjectName("PlotGlassTitle")
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

        ThemeManager.instance().themeChanged.connect(self.update)
        ThemeManager.instance().themeChanged.connect(lambda _: self._apply_icon_theme())
        self._apply_icon_theme()

    # ── Icon theming ──────────────────────────────────────────────
    @staticmethod
    def _tinted_icon(path: str, color: QtGui.QColor, size: int = 13) -> QtGui.QIcon:
        src = QtGui.QIcon(path).pixmap(size, size)
        dst = QtGui.QPixmap(src.size())
        dst.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(dst)
        p.drawPixmap(0, 0, src)
        p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
        p.fillRect(dst.rect(), color)
        p.end()
        return QtGui.QIcon(dst)

    def _preload_icons(self) -> None:
        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")
        tint = QtGui.QColor(*PALETTES["dark"]["plot_text_normal"][:3])
        for stem in ("fullscreen", "fullscreen-exit", "gear"):
            path = os.path.join(icons_dir, f"{stem}.svg")
            attr = "_icon_" + stem.replace("-", "_")
            setattr(self, attr, QtGui.QIcon(path))
            setattr(self, attr + "_lit", self._tinted_icon(path, tint))

    def _apply_icon_theme(self) -> None:
        dark = ThemeManager.instance().mode() == ThemeMode.DARK
        if hasattr(self, "_menu_btn"):
            self._menu_btn.setIcon(self._icon_gear_lit if dark else self._icon_gear)
        if hasattr(self, "btn_fs"):
            if self._is_fullscreen:
                icon = self._icon_fullscreen_exit_lit if dark else self._icon_fullscreen_exit
            else:
                icon = self._icon_fullscreen_lit if dark else self._icon_fullscreen
            self.btn_fs.setIcon(icon)

    # ── Round icon button factory (circular hover) ────────────────
    def _make_icon_button(self, icon_name: str, tooltip: str) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton()
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", icon_name)
        btn.setIcon(QtGui.QIcon(icon_path))
        btn.setIconSize(QtCore.QSize(13, 13))
        btn.setFixedSize(24, 24)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        btn.setToolTip(tooltip)
        btn.setObjectName("PlotIconBtn")
        return btn

    # ── Menu builder ──────────────────────────────────────────────
    def _build_menu(self) -> QtWidgets.QToolButton:
        self._menu_btn = self._make_icon_button("gear.svg", "Plot Options")
        self._menu_btn.setObjectName("PlotMenuBtn")
        self._menu_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        self.plot_menu = self._build_glass_menu(self._menu_btn)
        self._menu_btn.setMenu(self.plot_menu)
        return self._menu_btn

    def _build_glass_menu(self, parent_widget: QtWidgets.QWidget) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(parent_widget)
        menu.setObjectName("PlotGlassMenu")
        menu.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        menu.setWindowFlags(
            menu.windowFlags()
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.NoDropShadowWindowHint
        )

        if self._sections:
            for i, (key, label, color) in enumerate(self._sections):
                row = PlotMenuRow(key, label, color)
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

        # Grid line toggles (always present)
        menu.addSeparator()
        for grid_key, grid_label in (
            ("grid_major", "Major Gridlines"),
            ("grid_minor", "Minor Gridlines"),
        ):
            grid_row = GridMenuRow(grid_key, grid_label)
            grid_row.toggled.connect(self.gridChanged)
            gwa = QtWidgets.QWidgetAction(menu)
            gwa.setDefaultWidget(grid_row)
            menu.addAction(gwa)

        return menu

    # ── Paint - frosted glass card ────────────────────────────────
    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        tok = ThemeManager.instance().tokens()
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        rf = QtCore.QRectF(self.rect())
        path = QtGui.QPainterPath()
        path.addRoundedRect(rf, self._R, self._R)
        p.setClipPath(path)

        # Frosted base
        p.fillRect(self.rect(), QtGui.QColor(*tok["plot_glass_base"]))
        p.fillRect(self.rect(), QtGui.QColor(*tok["plot_glass_overlay"]))

        # Top shimmer
        sh = QtGui.QLinearGradient(0, 0, 0, 40)
        sh.setColorAt(0, QtGui.QColor(*tok["plot_glass_shimmer_top"]))
        sh.setColorAt(0.5, QtGui.QColor(*tok["plot_glass_shimmer_mid"]))
        sh.setColorAt(1, QtGui.QColor(0, 0, 0, 0))
        p.fillRect(self.rect(), QtGui.QBrush(sh))

        # Bottom vignette
        vg = QtGui.QLinearGradient(0, self.height() - 30, 0, self.height())
        vg.setColorAt(0, QtGui.QColor(0, 0, 0, 0))
        vg.setColorAt(1, QtGui.QColor(*tok["plot_glass_vignette_end"]))
        p.fillRect(self.rect(), QtGui.QBrush(vg))

        p.setClipping(False)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        # Outer bright rim
        p.setPen(QtGui.QPen(QtGui.QColor(*tok["plot_glass_rim"]), 1.0))
        p.drawRoundedRect(rf.adjusted(0.5, 0.5, -0.5, -0.5), self._R, self._R)
        # Inner inset
        p.setPen(QtGui.QPen(QtGui.QColor(*tok["plot_glass_inset"]), 1.0))
        p.drawRoundedRect(rf.adjusted(1.5, 1.5, -1.5, -1.5), self._R - 1.5, self._R - 1.5)

        if self.has_header:
            p.setPen(QtGui.QPen(QtGui.QColor(*tok["plot_glass_header_line"]), 1.0))
            y_line = self._HEADER_H + self._M
            p.drawLine(0, y_line, self.width(), y_line)

        p.end()


# ============================================================
#  GlassTabContainer
# ============================================================


class PlotTabContainer(PlotContainer):
    """Specialized Glass Container that manages device tabs with activity indicators."""

    deviceColorRequested = QtCore.pyqtSignal(int)  # Legacy: emits active device index

    def __init__(self, parent: QtWidgets.QWidget = None) -> None:
        dummy = QtWidgets.QWidget()
        tok = ThemeManager.instance().tokens()
        main_sections = [
            ("dissipation", "Dissipation", QtGui.QColor(*tok["plot_data_temperature"][:3])),
            (
                "resonance_freq",
                "Resonance Frequency",
                QtGui.QColor(*tok["plot_data_primary"][:3]),
            ),
        ]
        # Pass sections so _build_glass_menu picks them up
        super().__init__(
            plot_widget=dummy,
            title=None,
            parent=parent,
            show_menu=False,
            sections=main_sections,
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

        # Firmware update status icon
        _fw_icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "fw-update.svg")
        self._fw_status_icon = UpdateStatusIcon(_fw_icon_path, size=18)
        h_layout.addWidget(self._fw_status_icon)

        # Fullscreen button (circular)
        self.btn_fs = self._make_icon_button("fullscreen.svg", "Toggle Fullscreen")
        self.btn_fs.clicked.connect(self.fullscreenRequested.emit)
        h_layout.addWidget(self.btn_fs)

        # Gear menu (uses self._sections = _MAIN_SECTIONS)
        h_layout.addWidget(self._build_menu())

        layout.insertWidget(0, self.header)

        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 1)

        self._apply_icon_theme()

        # ── Activity indicator state management ────────────────────
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.timeout.connect(self._animate_icons)
        self._device_states: dict[int, str] = {}
        self._valid_states = {"idle", "error", "init", "success", "recording"}
        # Per-port firmware states for the status icon (port string → FW_UPDATE int)
        self._fw_port_states: dict[str, int] = {}

    # ── Menu override: per-section grid controls ──────────────────
    def _build_glass_menu(self, parent_widget: QtWidgets.QWidget) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(parent_widget)
        menu.setObjectName("PlotGlassMenu")
        menu.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        menu.setWindowFlags(
            menu.windowFlags()
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.NoDropShadowWindowHint
        )

        # Section-specific grid key mapping
        _grid_keys = {
            "dissipation": ("grid_diss_major", "grid_diss_minor"),
            "resonance_freq": ("grid_rf_major", "grid_rf_minor"),
        }

        for i, (key, label, color) in enumerate(self._sections):
            row = PlotMenuRow(key, label, color)
            row.colorChanged.connect(self.sectionColorChanged)
            row.visibilityChanged.connect(self.sectionVisibilityChanged)
            wa = QtWidgets.QWidgetAction(menu)
            wa.setDefaultWidget(row)
            menu.addAction(wa)

            major_key, minor_key = _grid_keys.get(key, (f"grid_{key}_major", f"grid_{key}_minor"))
            for grid_key, grid_label in (
                (major_key, "Major Gridlines"),
                (minor_key, "Minor Gridlines"),
            ):
                grid_row = GridMenuRow(grid_key, grid_label)
                grid_row.toggled.connect(self.gridChanged)
                gwa = QtWidgets.QWidgetAction(menu)
                gwa.setDefaultWidget(grid_row)
                menu.addAction(gwa)

            if i < len(self._sections) - 1:
                menu.addSeparator()

        return menu

    # ── Menu override: per-section grid controls ──────────────────
    def _build_glass_menu(self, parent_widget: QtWidgets.QWidget) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(parent_widget)
        menu.setObjectName("PlotGlassMenu")
        menu.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        menu.setWindowFlags(
            menu.windowFlags()
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.NoDropShadowWindowHint
        )

        # Section-specific grid key mapping
        _grid_keys = {
            "dissipation": ("grid_diss_major", "grid_diss_minor"),
            "resonance_freq": ("grid_rf_major", "grid_rf_minor"),
        }

        for i, (key, label, color) in enumerate(self._sections):
            row = PlotMenuRow(key, label, color)
            row.colorChanged.connect(self.sectionColorChanged)
            row.visibilityChanged.connect(self.sectionVisibilityChanged)
            wa = QtWidgets.QWidgetAction(menu)
            wa.setDefaultWidget(row)
            menu.addAction(wa)

            major_key, minor_key = _grid_keys.get(key, (f"grid_{key}_major", f"grid_{key}_minor"))
            for grid_key, grid_label in (
                (major_key, "Major Gridlines"),
                (minor_key, "Minor Gridlines"),
            ):
                grid_row = GridMenuRow(grid_key, grid_label)
                grid_row.toggled.connect(self.gridChanged)
                gwa = QtWidgets.QWidgetAction(menu)
                gwa.setDefaultWidget(grid_row)
                menu.addAction(gwa)

            if i < len(self._sections) - 1:
                menu.addSeparator()

        return menu

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
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        btn.setFixedHeight(20)
        btn.setObjectName("PlotDeviceTab")

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
        if index not in self._device_states or state not in self._valid_states:
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

        tok = ThemeManager.instance().tokens()
        state_tok = {
            "idle": tok["danger"],
            "error": tok["danger"],
            "init": tok["warning"],
            "success": tok["success"],
            "recording": tok["success"],
        }
        color = QtGui.QColor(*state_tok.get(state, tok["danger"])[:3])

        if state == "init":
            alpha = 255 if (int(time.time() * 2) % 2 == 0) else 60
            color.setAlpha(alpha)
        elif state == "recording":
            alpha = int(140 + 115 * math.sin(time.time() * 3.5))
            color.setAlpha(alpha)

        btn.setIcon(self._create_dot_icon(color))

    def _create_dot_icon(self, color: QtGui.QColor) -> QtGui.QIcon:
        pixmap = QtGui.QPixmap(16, 16)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setBrush(color)
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(4, 4, 8, 8)
        painter.end()
        return QtGui.QIcon(pixmap)

    def _on_tab_clicked(self, index: int) -> None:
        self.stack.setCurrentIndex(index)

    # ── Firmware update status icon ───────────────────────────────
    def on_fw_status_changed(self, port: str, fw_value: int) -> None:
        """Update the firmware status icon based on a per-port check result.

        Tracks each port's FW_UPDATE value and displays the worst-case state
        across all connected devices as a single icon (green/yellow/red).

        Args:
            port: Serial port string (e.g. "COM3").
            fw_value: Integer value of FW_UPDATE enum from fwUpdater.
        """
        from QATCH.common.fwUpdater import FW_UPDATE

        self._fw_port_states[port] = fw_value

        # Compute worst-case state across all known ports.
        worst = UpdateStatusIcon.State.UP_TO_DATE
        detail_lines = []
        for p, v in self._fw_port_states.items():
            if v == int(FW_UPDATE.RESULT_REQUIRED):
                worst = UpdateStatusIcon.State.MANDATORY
                detail_lines.append(f"{p}: incompatible firmware")
            elif v in (int(FW_UPDATE.RESULT_OUTDATED), int(FW_UPDATE.RESULT_UNKNOWN)):
                if worst != UpdateStatusIcon.State.MANDATORY:
                    worst = UpdateStatusIcon.State.OPTIONAL
                label = (
                    "update available" if v == int(FW_UPDATE.RESULT_OUTDATED) else "unknown version"
                )
                detail_lines.append(f"{p}: {label}")
            elif v == int(FW_UPDATE.RESULT_FAILED):
                if worst == UpdateStatusIcon.State.UP_TO_DATE:
                    worst = UpdateStatusIcon.State.UNKNOWN
                detail_lines.append(f"{p}: check failed")

        self._fw_status_icon.setState(worst, "\n".join(detail_lines))


def _make_plot_widget(parent: QtWidgets.QWidget) -> GraphicsLayoutWidget:
    w = GraphicsLayoutWidget(parent)
    w.setAutoFillBackground(False)
    w.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
    w.setBackground(None)
    w.setFrameShape(QtWidgets.QFrame.NoFrame)
    w.setFrameShadow(QtWidgets.QFrame.Plain)
    w.setLineWidth(0)
    w.setMinimumSize(80, 60)
    return w


# ============================================================
#  UIPlots  - consumed by PlotsWindow
# ============================================================


class UIPlots:

    def setup_ui(self, plots_window: QtWidgets.QMainWindow) -> None:
        USE_FULLSCREEN = QDesktopWidget().availableGeometry().width() == 2880

        plots_window.setObjectName("plotsWindow")
        plots_window.setMinimumSize(QtCore.QSize(1000, 250))
        if USE_FULLSCREEN:
            plots_window.resize(1701, 1435)
            plots_window.move(0, 0)
        else:
            plots_window.move(692, 0)
        plots_window.setTabShape(QtWidgets.QTabWidget.Rounded)

        self._fullscreen_active_widget = None

        # Central widget ──────────────────────────────────────────────
        self.centralwidget = QtWidgets.QWidget(plots_window)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)

        root = QtWidgets.QVBoxLayout(self.centralwidget)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(6)

        tok = ThemeManager.instance().tokens()

        # Main horizontal splitter ────────────────────────────────────
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self.centralwidget)
        self.main_splitter.setHandleWidth(6)

        # LEFT - Integrated glass tabs (Dissipation / Resonance Freq)
        self.left_pane = PlotTabContainer(parent=self.main_splitter)
        self.pltB = _make_plot_widget(self.left_pane)
        self.pltB.setObjectName("pltB")
        self.left_pane.add_device(
            "Device 1", self.pltB, QtGui.QColor(*tok["plot_data_device_accent"][:3])
        )

        # RIGHT - stacked splitter for Amp & Temp
        self.right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self.main_splitter)
        self.right_splitter.setHandleWidth(6)

        # Amplitude pane
        self.plt = _make_plot_widget(self.right_splitter)
        self.plt.setObjectName("plt")
        _AMP_INIT = QtGui.QColor(220, 68, 80)  # rose-red, matches _init_plot_curves
        self.amp_glass = PlotContainer(
            plot_widget=self.plt,
            title="Amplitude (dB)",
            accent_color=_AMP_INIT,
            parent=self.right_splitter,
            sections=[
                ("amplitude", "Amplitude", _AMP_INIT),
            ],
        )
        self.right_splitter.addWidget(self.amp_glass)

        # Temperature pane
        self.plt_temp = _make_plot_widget(self.right_splitter)
        self.plt_temp.setObjectName("plt_temp")
        _TEMP_INIT = QtGui.QColor(148, 99, 210)  # soft violet, matches _init_plot_curves
        self.temp_glass = PlotContainer(
            plot_widget=self.plt_temp,
            title="Temperature °C",
            accent_color=_TEMP_INIT,
            parent=self.right_splitter,
            sections=[
                ("temperature", "Temperature", _TEMP_INIT),
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
        plots_window.setCentralWidget(self.centralwidget)

        # Splitter ratios
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 1)
        self.right_splitter.setStretchFactor(0, 1)
        self.right_splitter.setStretchFactor(1, 1)

        # Legacy stubs
        self.Layout_graphs = QtWidgets.QWidget()
        self.btnCollapse = QtWidgets.QToolButton()
        self.btnExpand = QtWidgets.QToolButton()

        self.retranslateUi(plots_window)
        QtCore.QMetaObject.connectSlotsByName(plots_window)

    # ── Smooth fullscreen toggle ──────────────────────────────────
    def _toggle_fullscreen(self, target_widget: QtWidgets.QWidget) -> None:
        """
        Animates the splitter sizes to expand/contract the target pane.

        Smoothness strategy
        ───────────────────
        1. A QTimer fires at ~60 fps (every 16 ms) - finer-grained than
           QVariantAnimation's default repaint scheduling.
        2. Before each animation, viewport updates are frozen on all three
           pyqtgraph widgets so OpenGL/QPainter doesn't redraw mid-resize.
        3. A quartic ease-in-out gives a natural, snappy feel.
        4. Viewport updates are restored (and a single repaint triggered)
           the moment the animation completes.
        """
        if getattr(self, "_fullscreen_active_widget", None) == target_widget:
            # ── Contracting back to normal ──
            target_main = self._normal_main_sizes
            target_right = self._normal_right_sizes
            target_widget._is_fullscreen = False
            target_widget._apply_icon_theme()
            target_widget.btn_fs.setToolTip("Toggle Fullscreen")
            self._fullscreen_active_widget = None
        else:
            # ── Expanding ──
            prev = getattr(self, "_fullscreen_active_widget", None)
            if prev is None:
                self._normal_main_sizes = self.main_splitter.sizes()
                self._normal_right_sizes = self.right_splitter.sizes()
            else:
                # Reset icon on the previously-expanded pane
                prev._is_fullscreen = False
                prev._apply_icon_theme()

            self._fullscreen_active_widget = target_widget
            target_widget._is_fullscreen = True
            target_widget._apply_icon_theme()
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
            """Quartic ease - snappy acceleration, smooth landing."""
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

    def retranslateUi(self, plots_window: QtWidgets.QMainWindow) -> None:
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "qatch-icon.png")
        plots_window.setWindowIcon(QtGui.QIcon(icon_path))
        plots_window.setWindowTitle(
            _translate(
                "plotsWindow",
                "{} {} - Plots".format(Constants.app_title, Constants.app_version),
            )
        )

    def Ui(self, plots_window: QtWidgets.QMainWindow) -> None:
        self.retranslateUi(plots_window)

    # Legacy no-op shims
    def handleSplitterMoved(self, pos: int = 0, index: int = 0) -> None:
        pass

    def handleSplitterButton(self, collapse: bool = True) -> None:
        pass
