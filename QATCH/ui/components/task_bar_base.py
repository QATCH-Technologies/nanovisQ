"""
QATCH.ui.components.task_bar_base

Shared chrome for the app's "CtrlToolBar"-styled task bars (AnalyzeActionBar,
ControlsActionBar): icon size, themed toolbutton construction (with a fixed
button height), icon retinting, a QSS checked-state repolish fix, and the
rounded-card shape both bars paint. Both bars already share
`app_theme.qss`'s `QToolBar#CtrlToolBar QToolButton` rules (hover/pressed/
checked/disabled, separators) - this centralizes the *Python* that builds
and behaves around those buttons, which had drifted between the two bars
(see module docstring history in analyze_action_bar.py / controls_action_bar.py).

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.components.icon_utils import tinted_icon
from QATCH.ui.styles.theme_manager import ThemeManager, ThemeMode, tok_css


def _icon_path(icon_name: str) -> str:
    return os.path.join(Architecture.get_path(), "QATCH", "icons", icon_name)


class TaskBarBase(QtWidgets.QWidget):
    """Base class for the app's themed top task bars.

    Subclasses build their own zones/layout in `__init__`, but should
    construct every `QToolBar` via `_make_toolbar()` and every
    `QToolButton` via `_tool_button()` so icon size, button height, cursor,
    and checked-state repaint behavior stay identical across bars; read
    `OUTER_MARGINS`/`ZONE_SPACING` in their own `_assemble()` rather than
    repeating those literals; and call `retint_icon_buttons()` from the
    subclass's own `themeChanged` handler if it has one (subclasses
    typically have additional theme-driven state of their own, so this
    base does not wire icon retinting to `themeChanged` itself - only the
    card background, which every subclass needs regardless).
    """

    ICON_SIZE = QtCore.QSize(50, 30)
    # Empirically today's natural CtrlToolBar QToolButton height (icon +
    # text-under-icon + QSS padding) - see RunControls.btn's own
    # QSize(60, 56), sized to match tool_Initialize/tool_Reset's sizeHint
    # before either bar set an explicit height. Fixing it here is a
    # lock-in of that existing reality, not a redesign.
    BUTTON_HEIGHT = 56
    _CARD_RADIUS = 12.0
    OUTER_MARGINS = (10, 1, 10, 1)
    ZONE_SPACING = 14

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        # (button, icon_name) pairs built via _tool_button(), retinted as a
        # batch by retint_icon_buttons().
        self._icon_buttons: List[Tuple[QtWidgets.QAbstractButton, str]] = []
        self._bg_cache: Optional[QtGui.QPixmap] = None
        self._bg_cache_mode: Optional[ThemeMode] = None
        ThemeManager.instance().themeChanged.connect(lambda _: self.update())

    def _make_toolbar(self) -> QtWidgets.QToolBar:
        """A `QToolBar` pre-wired for `app_theme.qss`'s `CtrlToolBar` rules
        and this bar's icon size."""
        bar = QtWidgets.QToolBar()
        bar.setObjectName("CtrlToolBar")
        bar.setIconSize(self.ICON_SIZE)
        return bar

    def _tool_button(
        self, text: str, icon_name: Optional[str] = None, checkable: bool = False
    ) -> QtWidgets.QToolButton:
        """Builds a themed toolbutton - text-under-icon, fixed height, hand
        cursor, auto-repolished on toggle (see `_repolish`) so a checked
        highlight set programmatically (not just by a user click) repaints
        immediately."""
        btn = QtWidgets.QToolButton()
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        btn.setText(text)
        btn.setFixedHeight(self.BUTTON_HEIGHT)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        btn.setCheckable(checkable)
        if checkable:
            btn.toggled.connect(lambda _checked, b=btn: self._repolish(b))
        if icon_name:
            self._icon_buttons.append((btn, icon_name))
        return btn

    @staticmethod
    def _repolish(widget: QtWidgets.QWidget) -> None:
        """Forces an immediate QSS re-evaluation of `widget`.

        Qt doesn't always repaint a `:checked`/`:disabled` pseudo-state
        change when it's triggered by `setChecked()`/`setEnabled()` rather
        than an actual user click - only unpolish+polish forces the style
        to re-evaluate right away.
        """
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def retint_icon_buttons(self) -> None:
        """Retints every `_tool_button()`-registered icon to the active
        theme's `flat_text` token.

        Public (unlike `_tool_button`/`_make_toolbar`, only ever called from
        a subclass's own construction code): `ControlsActionBar` is retinted
        from `UIControls`'s existing single `themeChanged` fan-out
        (`_refresh_toolbar_icons`) rather than subscribing itself, so this
        needs to be callable across that wrap boundary. Call from the
        subclass's own `themeChanged` handler if it has one (like
        `AnalyzeActionBar` does).
        """
        tok = ThemeManager.instance().tokens()
        color = QtGui.QColor(*tok["flat_text"])
        for btn, icon_name in self._icon_buttons:
            btn.setIcon(tinted_icon(_icon_path(icon_name), color, self.ICON_SIZE.height()))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Paints the shared rounded-card background behind the bar.

        Ported from the now-retired `QATCH.ui.widgets.controls_widget
        .ControlsWidget` (same radius/tokens `AnalyzeActionBar` already
        painted its own copy of) rather than a naive per-paint redraw:
        `ControlsActionBar`'s temp/PID readouts repaint on every ~100ms
        plot tick during a run, forcing this to repaint just as often, so
        the fill/border - which only actually changes on resize or theme
        switch - is rendered once into an offscreen pixmap and blitted on
        every other call.
        """
        mode = ThemeManager.instance().mode()
        size = self.size()
        if (
            self._bg_cache is None
            or self._bg_cache.size() != size
            or self._bg_cache_mode != mode
        ):
            self._bg_cache = self._render_background(size)
            self._bg_cache_mode = mode

        p = QtGui.QPainter(self)
        p.drawPixmap(0, 0, self._bg_cache)
        p.end()

    def _render_background(self, size: QtCore.QSize) -> QtGui.QPixmap:
        tok = ThemeManager.instance().tokens()
        pm = QtGui.QPixmap(size)
        pm.fill(QtCore.Qt.GlobalColor.transparent)

        p = QtGui.QPainter(pm)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        paint_flat_surface(
            self,
            radius=self._CARD_RADIUS,
            fill=QtGui.QColor(*tok["surface"]),
            border=QtGui.QColor(*tok["surface_border"]),
            painter=p,
        )
        p.end()
        return pm


class TaskBarDivider(QtWidgets.QFrame):
    """A hairline vertical divider between task-bar zones, themed from
    `flat_border` so it stays correct across light/dark switches.

    Sized to `height` (the zones' own CtrlToolBar row height) and meant to
    be added with `AlignVCenter` - not left to stretch to the full zone
    height (reads as a full edge-to-edge rule spanning a caption line too,
    if the bar has one), and not an arbitrary short fixed height either
    (reads stubbier than `CtrlToolBar`'s own QSS separators, whose
    `margin: 5px 4px` makes them nearly as tall as their row).
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
