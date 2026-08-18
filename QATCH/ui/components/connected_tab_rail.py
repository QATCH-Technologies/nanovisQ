"""
QATCH.ui.components.connected_tab_rail.py

This module provides :class:`ConnectedTabRail`, a vertical navigation rail
whose active-row highlight visually joins the adjacent content pane into one
continuous surface.

The connected appearance is produced by constructing the content and active
highlight as separate :class:`QPainterPath` objects and combining them with
`QPainterPath.united()` before painting.  This prevents a visible seam
between two merely adjacent rounded rectangles and allows the active region
to slide smoothly between navigation rows.

The widget owns the navigation controls and paints the shared backdrop, while
the caller owns the actual content placed inside `content_area`.

Usage:
    Create the rail with a sequence of `(key, label)` or
    `(key, label, icon_path)` tuples, connect `modeChanged` to the
    application's navigation handler, and populate `content_area` with the
    desired content widget or layout.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-18
"""

from __future__ import annotations

import os

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.icon_utils import tinted_icon
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css


class ConnectedTabRail(QtWidgets.QWidget):
    """Vertical navigation rail connected visually to its content pane.

    The widget creates a vertical set of checkable navigation buttons beside
    a transparent content area.  The active button is represented visually
    by a rounded highlight that extends into the content pane.  During
    painting, the highlight and content surface are united into a single
    painter path so their shared boundary has no visible seam.

    Navigation changes are animated with a cubic easing curve.  Icons are
    tinted from the active theme and updated automatically when the theme
    changes.

    Attributes:
        content_area (QtWidgets.QWidget): Transparent placeholder that callers
            populate with their own layout and content.  The rail paints the
            backdrop behind this area but does not manage its contents.
        modeChanged (pyqtSignal): Emitted with the selected mode key whenever
            the active key changes.
    """

    modeChanged = QtCore.pyqtSignal(str)

    _ANIM_DURATION = 220

    def __init__(
        self,
        modes,
        parent: QtWidgets.QWidget | None = None,
        *,
        rail_width: int = 132,
        row_height: int = 38,
        row_spacing: int = 4,
        content_radius: float = 12.0,
        icon_size: int = 18,
    ) -> None:
        """Initialize the navigation rail.

        Args:
            modes (Sequence[tuple]): Navigation definitions. Each entry must
                contain either `(key, label)` or
                `(key, label, icon_path)`.
            parent (Optional[QtWidgets.QWidget]): Parent widget. Defaults to
                `None`.
            rail_width (int): Width of the navigation rail in pixels.
                Defaults to `132`.
            row_height (int): Height of each navigation row in pixels.
                Defaults to `38`.
            row_spacing (int): Vertical spacing between navigation rows.
                Defaults to `4`.
            content_radius (float): Corner radius of the content surface.
                Defaults to `12.0`.
            icon_size (int): Width and height of navigation icons in pixels.
                Defaults to `18`.

        The constructor creates the navigation controls, content placeholder,
        active-row animation, themed icons, and button styling.  The initial
        active mode remains unset until :meth:`set_active` is called.
        """
        # modes: list of (key, label) or (key, label, icon_path)
        super().__init__(parent)
        self.setObjectName("connectedTabRail")
        self._rail_width = rail_width
        self._row_height = row_height
        self._row_spacing = row_spacing
        self._row_margin = 6  # matches SegmentedControl's vertical rail margins
        self._radius = content_radius
        self._pill_radius = min(10.0, content_radius)
        self._icon_size = icon_size

        self._keys: list = [m[0] for m in modes]
        self._buttons: dict = {}
        self._icons: dict = {}
        self._icon_paths: dict = {}
        self._active_key: str | None = None
        self._pill_y = float(self._row_margin)  # animated top-y of the connector, local coords

        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(self._ANIM_DURATION)
        self._anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_anim_value)

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        rail_col = QtWidgets.QWidget(self)
        rail_col.setFixedWidth(rail_width)
        rail_col.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        rail_layout = QtWidgets.QVBoxLayout(rail_col)
        rail_layout.setContentsMargins(
            self._row_margin, self._row_margin, self._row_margin, self._row_margin
        )
        rail_layout.setSpacing(row_spacing)

        for mode in modes:
            if len(mode) == 3:
                key, label, icon_path = mode
            else:
                key, label = mode
                icon_path = None
            btn = QtWidgets.QToolButton(rail_col)
            btn.setText(f" {label}" if icon_path else label)
            btn.setCheckable(True)
            btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            btn.setFixedHeight(row_height)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            if icon_path and os.path.exists(icon_path):
                self._icon_paths[key] = icon_path
                btn.setIconSize(QtCore.QSize(icon_size, icon_size))
                btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            else:
                btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            btn.clicked.connect(lambda _=False, k=key: self.set_active(k))
            rail_layout.addWidget(btn)
            self._buttons[key] = btn

        rail_layout.addStretch()

        self.content_area = QtWidgets.QWidget(self)
        self.content_area.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        outer.addWidget(rail_col, 0)
        outer.addWidget(self.content_area, 1)

        self._refresh_icons()
        self._apply_button_qss()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        """Refresh the rail after the application theme changes.

        Re-tints navigation icons, reapplies button colors from the current
        theme tokens, and requests a repaint of the connected surface.

        Args:
            _mode (str): Theme mode reported by `ThemeManager`. The value is
                unused because current tokens are queried directly.
        """
        self._refresh_icons()
        self._apply_button_qss()
        self.update()

    def _refresh_icons(self) -> None:
        """Rebuild navigation icons using the current theme colors.

        Each configured icon receives both an inactive muted tint and an
        active accent tint.  The appropriate version is assigned according to
        the current active key.
        """
        tok = ThemeManager.instance().tokens()
        inactive_color = QtGui.QColor(*tok["flat_text_muted"])
        active_color = QtGui.QColor(*tok["flat_accent"])
        for key, path in self._icon_paths.items():
            icon_inactive = tinted_icon(path, inactive_color, self._icon_size)
            icon_active = tinted_icon(path, active_color, self._icon_size)
            self._icons[key] = (icon_inactive, icon_active)
            btn = self._buttons[key]
            btn.setIcon(icon_active if key == self._active_key else icon_inactive)

    def _apply_button_qss(self) -> None:
        """Apply themed styling to all navigation buttons.

        Buttons are transparent and left-aligned in the inactive state.  The
        active button uses the theme accent color and increased font weight.
        """
        tok = ThemeManager.instance().tokens()
        qss = f"""
            QToolButton {{
                background: transparent;
                border: none;
                color: {tok_css(tok["flat_text_muted"])};
                font-size: 12px; font-weight: 600;
                padding: 0px 9px;
                text-align: left;
            }}
            QToolButton:checked {{
                color: {tok_css(tok["flat_accent"])};
                font-weight: 700;
            }}
        """
        for btn in self._buttons.values():
            btn.setStyleSheet(qss)

    def set_active(self, key: str) -> None:
        """Select a navigation mode and move the connected highlight.

        Invalid keys are ignored.  For the first selection, or while the
        widget is hidden, the highlight snaps directly to the target row.
        Otherwise the highlight animates from its current vertical position
        to the selected row.

        The `modeChanged` signal is emitted only when the selected key
        differs from the previously active key.

        Args:
            key (str): Navigation key to activate.
        """
        if key not in self._buttons:
            return
        for k, btn in self._buttons.items():
            is_active = k == key
            btn.setChecked(is_active)
            icons = self._icons.get(k)
            if icons is not None:
                btn.setIcon(icons[1] if is_active else icons[0])

        changed = key != self._active_key
        was_first_selection = self._active_key is None
        self._active_key = key
        target_y = float(self._row_top(key))

        self._anim.stop()
        if was_first_selection or not self.isVisible():
            self._pill_y = target_y
            self.update()
        else:
            self._anim.setStartValue(self._pill_y)
            self._anim.setEndValue(target_y)
            self._anim.start()

        if changed:
            self.modeChanged.emit(key)

    def active_key(self) -> str | None:
        """Return the key of the currently active navigation mode.

        Returns:
            str | None: Active mode key, or `None` when no mode has yet
                been selected.
        """
        return self._active_key

    def _row_top(self, key: str) -> int:
        idx = self._keys.index(key)
        return self._row_margin + idx * (self._row_height + self._row_spacing)

    def _on_anim_value(self, v) -> None:
        """Calculate the vertical position of a navigation row.

        Args:
            key (str): Navigation key whose row position should be returned.

        Returns:
            int: Top coordinate of the row in the rail's local coordinates.

        Raises:
            ValueError: If `key` is not present in the configured modes.
        """
        self._pill_y = float(v)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the connected content surface and active-row highlight.

        The content pane is represented by a rounded rectangle.  When a mode
        is active, its highlight is extended into the content pane and united
        with the content path before painting.  Both the fill and border are
        then rendered from that single path, eliminating the seam that would
        otherwise appear between adjacent rounded surfaces.

        Args:
            event (QtGui.QPaintEvent): Qt paint event supplied by Qt.
        """
        tok = ThemeManager.instance().tokens()
        fill = QtGui.QColor(*tok["flat_surface"])
        border = QtGui.QColor(*tok["flat_border"])

        content_rect = QtCore.QRectF(
            float(self._rail_width),
            0.0,
            float(self.width() - self._rail_width),
            float(self.height()),
        )
        if content_rect.width() <= 0 or content_rect.height() <= 0:
            return

        content_path = QtGui.QPainterPath()
        content_path.addRoundedRect(content_rect, self._radius, self._radius)

        union_path = content_path
        if self._active_key is not None:
            pill_rect = QtCore.QRectF(
                float(self._row_margin),
                self._pill_y,
                float(self._rail_width - self._row_margin) + self._radius + 6.0,
                float(self._row_height),
            )
            pill_path = QtGui.QPainterPath()
            pill_path.addRoundedRect(pill_rect, self._pill_radius, self._pill_radius)
            union_path = content_path.united(pill_path)

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawPath(union_path)
        p.setPen(QtGui.QPen(border, 1))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawPath(union_path)
        p.end()
