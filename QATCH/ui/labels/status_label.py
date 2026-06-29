from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Any


class StatusLabel(QtWidgets.QLabel):
    """Frosted glass panel for status and info displays."""

    _RADIUS: float = 5.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Frosted glass panel for status and info displays.

        Attributes:
            _RADIUS (float): The corner radius applied to the glass panel.
        """
        super().__init__(*args, **kwargs)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setStyleSheet(
            "QLabel { color: rgba(28, 40, 52, 210); " "padding: 2px 6px; background: transparent; }"
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Performs custom painting to render the frosted glass status panel.

        The painting sequence applies a frosted-white base, a subtle blue
        tint, a top-down shimmer gradient, and a double-stroke border
        for depth.

        Args:
            event (QtGui.QPaintEvent): The paint event provided by the system.
        """
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing)

        rect_f = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)

        # Frosted white glass base
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 155))
        p.fillRect(self.rect(), QtGui.QColor(210, 225, 240, 40))

        # Top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, 36)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 80))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Borders
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
