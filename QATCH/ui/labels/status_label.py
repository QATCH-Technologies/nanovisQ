"""
QATCH.ui.labels.status_label.py

Status label widget.

This module provides :class:`StatusLabel`, a custom
:class:`QtWidgets.QLabel` that renders a translucent panel for
displaying status messages, informational text, and other secondary UI
content.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-04
"""

from typing import Any

from PyQt5 import QtCore, QtGui, QtWidgets


class StatusLabel(QtWidgets.QLabel):
    """
    The widget paints a rounded translucent panel beneath the label text,
    making it suitable for status indicators, informational messages, or
    other UI elements that benefit from subtle visual emphasis without the
    prominence of a full banner.

    Text rendering is delegated to :class:`QtWidgets.QLabel`, allowing the
    widget to retain standard QLabel functionality such as alignment, word
    wrapping, eliding, and stylesheet-based text formatting.

    Attributes:
        _RADIUS (float): Corner radius, in pixels, used when drawing the
            rounded background and borders.
    """

    _RADIUS: float = 5.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the frosted glass status label.

        Configures the widget to disable automatic system background painting
        so the custom :meth:`paintEvent` has full control over rendering the
        glass panel. The label itself remains transparent so only the text is
        painted by the base QLabel implementation.

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
            "QLabel { color: rgba(28, 40, 52, 210); " "padding: 2px 6px; background: transparent; }"
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paints the background.

        Rendering effect(s):

        1. Clip drawing to a rounded rectangle.
        2. Paint a translucent white base.
        3. Apply a soft blue tint overlay.
        4. Draw a top shimmer highlight.
        5. Render inner and outer rounded borders.
        6. Delegate text rendering to the base QLabel implementation.

        Args:
            event (QtGui.QPaintEvent): The Qt paint event requesting the
                widget redraw.
        """
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing)

        rect_f = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 155))
        p.fillRect(self.rect(), QtGui.QColor(210, 225, 240, 40))
        shimmer = QtGui.QLinearGradient(0, 0, 0, 36)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 80))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))
        p.setClipping(False)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(120, 160, 200, 110), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 160), 1.0))
        p.drawRoundedRect(
            rect_f.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        p.end()
        super().paintEvent(event)
