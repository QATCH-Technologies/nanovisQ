"""
QATCH.ui.components.flat_paint.py

Shared flat-surface paint recipe

Typical use inside a widget's `paintEvent`::

    from QATCH.ui.components.flat_paint import paint_flat_surface
    from QATCH.ui.styles.theme_manager import ThemeManager

    def paintEvent(self, event):
        tok = ThemeManager.instance().tokens()
        paint_flat_surface(
            self,
            radius=7.0,
            fill=QtGui.QColor(*tok["flat_surface"]),
            border=QtGui.QColor(*tok["flat_border"]),
            ring=QtGui.QColor(*tok["flat_accent_ring"]) if self._focused else None,
        )

Author (s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-18
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets


def paint_flat_surface(
    widget: QtWidgets.QWidget,
    *,
    radius: float,
    fill: QtGui.QColor,
    border: QtGui.QColor,
    border_width: float = 1.0,
    ring: QtGui.QColor | None = None,
    ring_width: float = 3.0,
    painter: QtGui.QPainter | None = None,
) -> None:
    """Paint a flat rounded-rectangle surface onto a widget.

    The surface consists of a filled rounded rectangle with an optional
    border. A focus ring may also be drawn inside the widget's bounds,
    allowing the ring to remain fully rounded without extending beyond or
    being clipped by the widget.

    Args:
        widget: Widget whose rectangle defines the area to be painted.
        radius: Corner radius of the rounded fill and border, in pixels.
        fill: Background color used to fill the rounded rectangle.
        border: Color used for the rounded-rectangle border.
        border_width: Width of the border stroke, in pixels. Set to `0` or
            a negative value to omit the border.
        ring: Optional color for the focus ring. When provided, a ring is
            drawn inside the widget's bounds on top of the fill and border.
            Pass `None` to omit the focus ring.
        ring_width: Width of the focus ring stroke, in pixels.
        painter: Optional active `QPainter` to use for rendering. If
            `None`, a new painter is created for `widget` and ended
            before the function returns.

    Returns:
        None.

    Note:
        The focus-ring parameters are currently part of the public painting
        interface, but the ring is only rendered when its drawing logic is
        enabled. The function configures antialiasing to ensure smooth
        rounded corners.
    """
    owns_painter = painter is None
    p = painter or QtGui.QPainter(widget)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setPen(QtCore.Qt.PenStyle.NoPen)

    rect = QtCore.QRectF(widget.rect())

    half_bw = border_width / 2.0
    fill_rect = rect.adjusted(half_bw, half_bw, -half_bw, -half_bw)
    p.setBrush(QtGui.QBrush(fill))
    p.drawRoundedRect(fill_rect, radius, radius)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)

    if border_width > 0:
        p.setPen(QtGui.QPen(border, border_width))
        p.drawRoundedRect(fill_rect, radius, radius)
        p.setPen(QtCore.Qt.PenStyle.NoPen)

    if ring is not None:
        half_rw = ring_width / 2.0
        ring_rect = rect.adjusted(half_rw, half_rw, -half_rw, -half_rw)
        ring_radius = max(radius - half_rw, 0.0)
        p.setPen(QtGui.QPen(ring, ring_width))
        p.drawRoundedRect(ring_rect, ring_radius, ring_radius)
        p.setPen(QtCore.Qt.PenStyle.NoPen)

    if owns_painter:
        p.end()
