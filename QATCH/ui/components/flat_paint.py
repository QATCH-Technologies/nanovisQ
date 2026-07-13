"""
QATCH.ui.components.flat_paint

Shared flat-surface paint recipe: fill + border + optional focus ring.

This is the flat-design sibling of QATCH.ui.components.glass_paint - it
backs the six "flat control system" components (Pushbutton, Line Edit,
Combo Box, Spin Box, Toggle, Option Card) rather than the frosted-glass
family. Unlike glass_paint's multi-layer shimmer/vignette/rim recipe, this
is a pure geometry+stroke primitive with no state logic: fill a rounded
rect, stroke its border, and optionally stroke a third, concentric rounded
rect inset from the widget's own edge as a focus ring.

The ring is the flat-language stand-in for a CSS `box-shadow: 0 0 0 3px
<ring-color>` - Qt Style Sheets have no box-shadow support, so a hard-edged
translucent stroke at the ring color's own alpha is the cheapest close
approximation (Qt has no cheap blur to spend on a soft glow here). It is
inset rather than protruding past `widget.rect()`: a widget's paintEvent
cannot paint outside its own backing-store rect, so an outward-facing ring
gets silently clipped flat at every corner instead of curving away,
regardless of how round the body underneath is.

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
    optional focus ring inset from the widget's own edge.

    Args:
        widget: The widget being painted (its `rect()` is used).
        radius: Corner radius of the fill/border rounded rect, in px.
        fill: Background fill color.
        border: Border stroke color.
        border_width: Border stroke width, in px.
        ring: If not None, a translucent stroke is drawn `ring_width` px
            inset from the widget's own edge, on top of the fill/border -
            the flat equivalent of a CSS focus-ring box-shadow, kept
            on-widget so it stays fully rounded at every corner instead of
            clipping flat. Pass None to omit it entirely.
        ring_width: Width in px of the ring stroke.
        painter: An active `QPainter` to draw with. If `None`, one is
            created on `widget` and ended before returning.
    """
    owns_painter = painter is None
    p = painter or QtGui.QPainter(widget)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setPen(QtCore.Qt.NoPen)

    rect = QtCore.QRectF(widget.rect())

    half_bw = border_width / 2.0
    fill_rect = rect.adjusted(half_bw, half_bw, -half_bw, -half_bw)
    p.setBrush(QtGui.QBrush(fill))
    p.drawRoundedRect(fill_rect, radius, radius)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)

    if border_width > 0:
        p.setPen(QtGui.QPen(border, border_width))
        p.drawRoundedRect(fill_rect, radius, radius)
        p.setPen(QtCore.Qt.NoPen)

    if ring is not None:
        # Stroked *inset* from the widget's own edge, on top of the fill/
        # border, rather than protruding past it. A widget's paintEvent
        # can't paint outside its own backing-store rect, so a ring
        # stroked outward from `rect` (the previous approach) got clipped
        # flat right at every corner instead of curving away - it read as
        # a squared-off highlight no matter how round the body underneath
        # was. Insetting keeps the whole stroke, corners included, on the
        # widget where it can actually be painted.
        half_rw = ring_width / 2.0
        ring_rect = rect.adjusted(half_rw, half_rw, -half_rw, -half_rw)
        ring_radius = max(radius - half_rw, 0.0)
        p.setPen(QtGui.QPen(ring, ring_width))
        p.drawRoundedRect(ring_rect, ring_radius, ring_radius)
        p.setPen(QtCore.Qt.NoPen)

    if owns_painter:
        p.end()
