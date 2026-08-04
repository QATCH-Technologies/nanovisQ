"""
QATCH.ui.labels.header_label.py

Custom header label widget.

This module provides :class:`HeaderLabel`, a lightweight replacement for
:class:`QtWidgets.QLabel` that renders a rounded, blue gradient background
with translucent glass effects. The widget is intended for use as a section
header throughout the application, providing a consistent visual style while
preserving all standard QLabel behavior such as text alignment, word wrapping,
and stylesheet-based text rendering.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-04
"""

from typing import Any

from PyQt5 import QtCore, QtGui, QtWidgets


class HeaderLabel(QtWidgets.QLabel):
    """A QLabel styled as a branded panel section header.

    The widget paints a rounded blue background with layered translucent
    gradients while delegating text rendering to :class:`QtWidgets.QLabel`.
    This ensures compatibility with standard QLabel features such as alignment,
    eliding, word wrapping, and stylesheet-defined text properties.

    Attributes:
        _RADIUS (float): Corner radius, in pixels, used when drawing the
            rounded background and border.
    """

    _RADIUS: float = 4.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the custom header label.

        Configures the widget to disable automatic system background painting
        so that all background rendering is performed by the custom
        :meth:`paintEvent`. A transparent stylesheet is also applied so only
        the text is handled by Qt's standard QLabel implementation.

        Args:
            *args: Positional arguments forwarded to
                :class:`QtWidgets.QLabel`.
            **kwargs: Keyword arguments forwarded to
                :class:`QtWidgets.QLabel`.
        """
        super().__init__(*args, **kwargs)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setStyleSheet(
            "QLabel { color: rgba(255, 255, 255, 230); "
            "padding: 2px 6px; font-weight: bold; background: transparent; }"
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paints the custom header background.

        Rendering is performed in multiple layers to create depth and improve
        readability:

        1. Clip drawing to a rounded rectangle.
        2. Paint the branded blue gradient background.
        3. Apply translucent tint overlays.
        4. Draw a subtle top shimmer highlight.
        5. Render inner and outer rounded borders.
        6. Delegate text painting to the base QLabel implementation.

        The text itself is intentionally rendered by the base class so that
        alignment, eliding, word wrapping, and stylesheet properties continue
        to behave exactly as they do for a standard QLabel.

        Args:
            event (QtGui.QPaintEvent): The Qt paint event triggering the
                repaint.
        """
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rect_f = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)

        # Blue gradient base
        grad = QtGui.QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0.0, QtGui.QColor(0, 118, 174))
        grad.setColorAt(1.0, QtGui.QColor(0, 158, 210))
        p.fillRect(self.rect(), QtGui.QBrush(grad))

        # Tints
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 45))
        p.fillRect(self.rect(), QtGui.QColor(180, 220, 245, 30))

        # Top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, self.height() * 0.65)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 55))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Borders
        p.setClipping(False)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(80, 160, 215, 130), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 130), 1.0))
        p.drawRoundedRect(
            rect_f.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        p.end()
        # Render text via base class (respects alignment, QSS color)
        super().paintEvent(event)
