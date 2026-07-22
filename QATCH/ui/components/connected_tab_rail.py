"""
QATCH.ui.components.connected_tab_rail

`ConnectedTabRail`: a vertical nav rail whose active-item highlight reads as
part of the same surface as the content pane beside it - not two separate
panels that merely touch, but one continuous shape with the highlighted row
flowing directly into the content area (the look used by e.g. macOS System
Settings' sidebar). Selecting a different row slides the connected region to
meet it.

The seam is drawn correctly (no visible border where the highlight overlaps
the content pane) by building both shapes as `QPainterPath`s and taking their
boolean union (`QPainterPath.united`) before filling/stroking once - so it is
always exactly one silhouette, never two adjacent rounded rects that happen
to touch.

Usage
-----
    rail = ConnectedTabRail([
        ("import", "Import", icon_path),
        ("export", "Export", icon_path2),
    ])
    rail.modeChanged.connect(handler)
    content_layout = QtWidgets.QVBoxLayout(rail.content_area)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.addWidget(my_content_stack)
    rail.set_active("import")
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.icon_utils import tinted_icon
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css


class ConnectedTabRail(QtWidgets.QWidget):
    """Vertical nav rail + content pane painted as one continuous surface.

    Attributes:
        content_area (QtWidgets.QWidget): Transparent placeholder the caller
            populates with its own layout/content (e.g. a QStackedWidget) -
            this widget only paints the backdrop behind it.
        modeChanged (pyqtSignal(str)): Emitted when the active row changes.
    """

    modeChanged = QtCore.pyqtSignal(str)

    _ANIM_DURATION = 220

    def __init__(
        self,
        modes,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        rail_width: int = 132,
        row_height: int = 38,
        row_spacing: int = 4,
        content_radius: float = 12.0,
        icon_size: int = 18,
    ) -> None:
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
        self._active_key: Optional[str] = None
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
                btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            else:
                btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
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

    # ------------------------------------------------------------------
    def _on_theme_changed(self, _mode: str) -> None:
        self._refresh_icons()
        self._apply_button_qss()
        self.update()

    def _refresh_icons(self) -> None:
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

    # ------------------------------------------------------------------
    def set_active(self, key: str) -> None:
        """Selects `key`, sliding the connected highlight to meet its row."""
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
            # No prior row to slide from (or not on screen yet to animate
            # meaningfully) - snap straight to position.
            self._pill_y = target_y
            self.update()
        else:
            self._anim.setStartValue(self._pill_y)
            self._anim.setEndValue(target_y)
            self._anim.start()

        if changed:
            self.modeChanged.emit(key)

    def active_key(self) -> Optional[str]:
        return self._active_key

    def _row_top(self, key: str) -> int:
        idx = self._keys.index(key)
        return self._row_margin + idx * (self._row_height + self._row_spacing)

    def _on_anim_value(self, v) -> None:
        self._pill_y = float(v)
        self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        tok = ThemeManager.instance().tokens()
        fill = QtGui.QColor(*tok["flat_surface"])
        border = QtGui.QColor(*tok["flat_border"])

        content_rect = QtCore.QRectF(
            float(self._rail_width), 0.0, float(self.width() - self._rail_width), float(self.height())
        )
        if content_rect.width() <= 0 or content_rect.height() <= 0:
            return

        content_path = QtGui.QPainterPath()
        content_path.addRoundedRect(content_rect, self._radius, self._radius)

        union_path = content_path
        if self._active_key is not None:
            # Extends from inside the rail across into the content pane by
            # more than the corner radius, so the overlap fully swallows the
            # seam - united() below then yields one continuous silhouette
            # rather than two rounded rects that merely touch.
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
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(fill)
        p.drawPath(union_path)
        p.setPen(QtGui.QPen(border, 1))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawPath(union_path)
        p.end()
