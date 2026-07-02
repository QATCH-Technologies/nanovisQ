"""
QATCH.ui.components.glass_paint

Shared frosted-glass paint recipe.

Before this module, three components each hand-rolled the same
"fill base -> overlay tint -> top shimmer -> optional vignette -> rounded rim
+ inset highlight" sequence in their own `paintEvent` - and two of them
hardcoded light-mode-only colors, so they broke in dark mode. This centralizes
that recipe and drives every color from the active `ColorTokens` palette, so
the whole glass family stays visually identical and theme-correct from one
place.

The canonical color source is the `plot_glass_*` token group (the same one
the plot cards use), so a glass surface here matches a glass plot card exactly.

Typical use inside a widget's `paintEvent`::

    from QATCH.ui.components.glass_paint import paint_glass_surface
    from QATCH.ui.styles.theme_manager import ThemeManager

    def paintEvent(self, event):
        paint_glass_surface(
            self,
            radius=18.0,
            tokens=ThemeManager.instance().tokens(),
            header_line_y=52,     # optional divider under a header
        )

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.styles.tokens import ColorTokens


def _c(rgba) -> QtGui.QColor:
    """(r, g, b, a) token tuple -> QColor."""
    return QtGui.QColor(*rgba)


def paint_glass_surface(
    widget: QtWidgets.QWidget,
    *,
    radius: float,
    tokens: ColorTokens,
    painter: Optional[QtGui.QPainter] = None,
    shimmer_height: float = 50.0,
    draw_vignette: bool = True,
    header_line_y: Optional[float] = None,
    rim: bool = True,
) -> None:
    """Paints the standard frosted-glass surface into `widget`.

    Draws, in order: the base glass fill, a faint overlay tint, a top-down
    white shimmer, an optional bottom vignette, an optional header divider,
    and (optionally) the two-stroke rounded border (outer rim + inner inset
    highlight). All colors come from `tokens` so light/dark both work.

    Args:
        widget: The widget being painted (its `rect()` is used).
        radius: Corner radius of the rounded-rectangle clip/border.
        tokens: Active `ColorTokens` palette (from `ThemeManager`).
        painter: An active `QPainter` to draw with. If `None`, one is
            created on `widget` and ended before returning.
        shimmer_height: Height in px of the top shimmer gradient.
        draw_vignette: Whether to draw the bottom vignette darkening.
        header_line_y: If set, draws a 1px divider across the widget at this
            y (used by dialogs with a titled header).
        rim: Whether to draw the outer/inner border strokes.
    """
    owns_painter = painter is None
    p = painter or QtGui.QPainter(widget)
    p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

    rect = widget.rect()
    rf = QtCore.QRectF(rect)

    clip = QtGui.QPainterPath()
    clip.addRoundedRect(rf, radius, radius)
    p.setClipPath(clip)

    # Base fill + faint tint overlay
    p.fillRect(rect, _c(tokens["plot_glass_base"]))
    p.fillRect(rect, _c(tokens["plot_glass_overlay"]))

    # Top-down shimmer
    shimmer = QtGui.QLinearGradient(0, 0, 0, shimmer_height)
    shimmer.setColorAt(0.0, _c(tokens["plot_glass_shimmer_top"]))
    shimmer.setColorAt(0.5, _c(tokens["plot_glass_shimmer_mid"]))
    shimmer.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
    p.fillRect(rect, QtGui.QBrush(shimmer))

    # Bottom vignette
    if draw_vignette:
        h = widget.height()
        vg = QtGui.QLinearGradient(0, h - 30, 0, h)
        vg.setColorAt(0.0, QtGui.QColor(0, 0, 0, 0))
        vg.setColorAt(1.0, _c(tokens["plot_glass_vignette_end"]))
        p.fillRect(rect, QtGui.QBrush(vg))

    p.setClipping(False)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)

    # Two-stroke border: muted outer rim + inner highlight inset
    if rim:
        p.setPen(QtGui.QPen(_c(tokens["plot_glass_rim"]), 1.0))
        p.drawRoundedRect(rf.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        p.setPen(QtGui.QPen(_c(tokens["plot_glass_inset"]), 1.0))
        p.drawRoundedRect(rf.adjusted(1.5, 1.5, -1.5, -1.5), radius - 1.5, radius - 1.5)

    # Optional header divider
    if header_line_y is not None:
        p.setPen(QtGui.QPen(_c(tokens["plot_glass_header_line"]), 1.0))
        p.drawLine(0, int(header_line_y), widget.width(), int(header_line_y))

    if owns_painter:
        p.end()
