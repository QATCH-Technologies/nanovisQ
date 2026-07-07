"""
QATCH.ui.components.icon_utils

Shared icon-tinting helper.

A handful of components each hand-rolled the same "load an SVG, recolor it
solid via SourceAtop compositing" routine independently (glass_dialog,
GlassSegmentedControl, and each of the data-management mode widgets). This
centralizes that recipe so every caller gets the same result from one place.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui


def tinted_icon(path: str, color: QtGui.QColor, size: int = 18) -> QtGui.QIcon:
    """Returns a copy of the icon/SVG at `path` fully recolored to `color`.

    Uses SourceAtop compositing, so transparent areas of the source stay
    transparent - only opaque pixels are recolored.

    Args:
        path: Filesystem path to an icon (SVG, PNG, etc).
        color: Solid color to paint the icon.
        size: Square pixmap side length, in px.

    Returns:
        A QIcon wrapping the tinted pixmap.
    """
    src = QtGui.QIcon(path).pixmap(size, size)
    dst = QtGui.QPixmap(src.size())
    dst.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(dst)
    p.drawPixmap(0, 0, src)
    p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
    p.fillRect(dst.rect(), color)
    p.end()
    return QtGui.QIcon(dst)


def tinted_pixmap(path: str, color: QtGui.QColor, size: int = 18) -> QtGui.QPixmap:
    """Same as `tinted_icon`, returning the raw QPixmap instead of a QIcon."""
    src = QtGui.QIcon(path).pixmap(size, size)
    dst = QtGui.QPixmap(src.size())
    dst.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(dst)
    p.drawPixmap(0, 0, src)
    p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
    p.fillRect(dst.rect(), color)
    p.end()
    return dst
