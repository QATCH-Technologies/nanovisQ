from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Any


class HeaderLabel(QtWidgets.QLabel):
    """Section-header label rendered as a blue panel.

    Attributes:
        _RADIUS (float): The corner radius applied to the panel
            background.
    """

    _RADIUS: float = 4.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the HeaderLabel with custom glass-style styling.

        Configures widget attributes to disable system background rendering,
        ensuring the custom paintEvent handles the glass aesthetic.
        """
        super().__init__(*args, **kwargs)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setStyleSheet(
            "QLabel { color: rgba(255, 255, 255, 230); "
            "padding: 2px 6px; font-weight: bold; background: transparent; }"
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Performs custom painting to render the glass-panel header.

        The painting sequence follows a specific layering order:
        1. Base Brand-Blue gradient.
        2. Semi-transparent glass tints.
        3. Top-down shimmer effect.
        4. Inner and outer anti-aliased borders.
        5. Default label text rendering via base class.

        Args:
            event (QtGui.QPaintEvent): The paint event provided by the Qt system.
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
