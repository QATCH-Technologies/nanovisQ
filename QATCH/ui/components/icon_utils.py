"""
QATCH.ui.components.icon_utils.py

Utility functions for creating tinted Qt icons and pixmaps.

This module provides a centralized implementation for recoloring icons using
Qt's `CompositionMode_SourceAtop` composition mode. Several UI components
require monochrome versions of the same SVG or raster assets that match the
current application theme. By consolidating the tinting logic into a single
module, all components produce consistent visual results while avoiding
duplicate rendering code.

The tinting process preserves the original alpha channel, allowing transparent
regions of the source image to remain transparent while replacing all visible
pixels with the requested color.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-05
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui


def tinted_icon(path: str, color: QtGui.QColor, size: int = 18) -> QtGui.QIcon:
    """Creates a solid-color version of an icon.

    Loads an icon from disk, renders it into a square pixmap, and recolors all
    visible pixels using Qt's `CompositionMode_SourceAtop` composition mode.
    The source image's transparency is preserved, making the function suitable
    for tinting SVGs and other icons used throughout the application's user
    interface.

    Args:
        path: Filesystem path to the source icon. Any image format supported by
            Qt (such as SVG or PNG) may be used.
        color: Color used to tint the rendered icon.
        size: Width and height, in pixels, of the rendered square pixmap.
            Defaults to `18`.

    Returns:
        A `QIcon` containing the tinted image.

    Notes:
        The original image is not modified. A new pixmap is rendered and
        returned each time this function is called.
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
    """Creates a solid-color version of an image as a `QPixmap`.

    Performs the same rendering and tinting operation as :func:`tinted_icon`,
    but returns the resulting `QPixmap` directly instead of wrapping it in a
    `QIcon`. This is useful when a pixmap is required for custom painting,
    animations, or other image processing operations.

    Args:
        path: Filesystem path to the source icon or image.
        color: Color used to tint the rendered image.
        size: Width and height, in pixels, of the rendered square pixmap.
            Defaults to `18`.

    Returns:
        A tinted `QPixmap` with the source image's transparency preserved.

    Notes:
        The tinting operation preserves the original alpha channel so only
        opaque portions of the image are recolored.
    """
    src = QtGui.QIcon(path).pixmap(size, size)
    dst = QtGui.QPixmap(src.size())
    dst.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(dst)
    p.drawPixmap(0, 0, src)
    p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
    p.fillRect(dst.rect(), color)
    p.end()
    return dst
