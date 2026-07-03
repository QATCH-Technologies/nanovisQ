"""
QATCH.ui.components.flat_paint

Shared flat-surface paint recipe: fill + border + optional focus ring.

This is the flat-design sibling of QATCH.ui.components.glass_paint - it
backs the six "flat control system" components (Pushbutton, Line Edit,
Combo Box, Spin Box, Toggle, Option Card) rather than the frosted-glass
family. Unlike glass_paint's multi-layer shimmer/vignette/rim recipe, this
is a pure geometry+stroke primitive with no state logic: fill a rounded
rect, stroke its border, and optionally stroke a second, larger concentric
rounded rect as a focus ring.

The ring is the flat-language stand-in for a CSS `box-shadow: 0 0 0 3px
<ring-color>` - Qt Style Sheets have no box-shadow support, so a hard-edged
translucent outer stroke at the ring color's own alpha is the cheapest
close approximation (Qt has no cheap blur to spend on a soft glow here).

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

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets


def paint_flat_surface(
    widget: QtWidgets.QWidget,
    *,
    radius: float,
    fill: QtGui.QColor,
    border: QtGui.QColor,
    border_width: float = 1.0,
    ring: Optional[QtGui.QColor] = None,
    ring_width: float = 3.0,
    painter: Optional[QtGui.QPainter] = None,
) -> None:
    """Paints a flat rounded-rect surface into `widget`: fill, border, and an
    optional outer focus ring.

    Args:
        widget: The widget being painted (its `rect()` is used).
        radius: Corner radius of the fill/border rounded rect, in px.
        fill: Background fill color.
        border: Border stroke color.
        border_width: Border stroke width, in px.
        ring: If not None, a translucent outer stroke is drawn `ring_width`
            px outside the border - the flat equivalent of a CSS focus-ring
            box-shadow. Pass None to omit it entirely.
        ring_width: Width in px of the outer ring stroke, and how far
            outside the border rect it sits.
        painter: An active `QPainter` to draw with. If `None`, one is
            created on `widget` and ended before returning.
    """
    owns_painter = painter is None
    p = painter or QtGui.QPainter(widget)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setPen(QtCore.Qt.NoPen)

    rect = QtCore.QRectF(widget.rect())

    if ring is not None:
        ring_rect = rect.adjusted(
            -ring_width + 0.5, -ring_width + 0.5, ring_width - 0.5, ring_width - 0.5
        )
        ring_radius = radius + ring_width
        p.setPen(QtGui.QPen(ring, ring_width))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(ring_rect, ring_radius, ring_radius)
        p.setPen(QtCore.Qt.NoPen)

    half_bw = border_width / 2.0
    fill_rect = rect.adjusted(half_bw, half_bw, -half_bw, -half_bw)
    p.setBrush(QtGui.QBrush(fill))
    p.drawRoundedRect(fill_rect, radius, radius)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)

    if border_width > 0:
        p.setPen(QtGui.QPen(border, border_width))
        p.drawRoundedRect(fill_rect, radius, radius)

    if owns_painter:
        p.end()
