from __future__ import annotations
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

# ---------------------------------------------------------------------------
# Constants matching ui_login.py
# ---------------------------------------------------------------------------
_INPUT_H: int = 34
_BTN_H: int = 34


class GlassLineEdit(QtWidgets.QLineEdit):
    """A custom QLineEdit with a translucent glass aesthetic and shimmer animations.

    This widget overrides the standard QPaintEvent to manually render a frosted
    glass background and a dynamic border. On focus, a 'shimmer' sweep effect
    animates across the border. Visual states for 'error' and 'focused' are
    handled through manual painting rather than QSS.

    Attributes:
        _shimmer_t (float): Animation progress normalized from 0.0 to 1.0.
        _focused (bool): Internal state tracking focus for rendering logic.
        _in_error (bool): Internal state tracking validation failure for rendering logic.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initializes the line edit with custom styles and animation timers."""
        super().__init__(parent)
        self._shimmer_t: float = 0.0
        self._focused: bool = False
        self._in_error: bool = False
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(12)
        self._timer.timeout.connect(self._tick)

        self.setFrame(False)
        self.setAutoFillBackground(False)
        self.setStyleSheet("""
            QLineEdit {
                background: transparent;
                border: none;
                padding: 0px 15px;
                color: rgba(38, 48, 58, 230);
                font-size: 10pt;
                selection-background-color: rgba(10, 163, 230, 60);
                selection-color: rgba(0, 0, 0, 255);
            }
            QLineEdit QToolButton { 
                background: transparent; 
                border: none; 
            }
            QLineEdit QToolButton:hover {
                background: rgba(255, 255, 255, 55);
                border-radius: 12px;
            }
        """)

    def set_error(self, on: bool) -> None:
        """Toggles the visual error state of the widget.

        Args:
            on (bool): If True, the widget paints with a red 'error' theme.
        """
        if on != self._in_error:
            self._in_error = on
            self.update()

    def _tick(self) -> None:
        """Increments the shimmer progress and triggers a repaint."""
        self._shimmer_t = min(1.0, self._shimmer_t + 0.022)
        self.update()
        if self._shimmer_t >= 1.0:
            self._timer.stop()

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        """Handles focus entry: resets error and triggers the shimmer animation."""
        super().focusInEvent(event)
        self._focused = True
        self._in_error = False
        self._shimmer_t = 0.0
        self._timer.start()
        self.update()

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        """Handles focus exit: stops the shimmer animation."""
        super().focusOutEvent(event)
        self._focused = False
        self._timer.stop()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Manually paints the glass background and animated border.

        This method executes before the standard QLineEdit paint event to
        draw the underlying 'glass' container. It then delegates text and
        cursor rendering back to the base class.
        """
        # Calculate geometry
        radius = self.height() / 2.0
        r = radius - 1.0
        rect = QtCore.QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        if self._in_error:
            fill = QtGui.QColor(255, 220, 220, 68)
        elif self._focused:
            fill = QtGui.QColor(255, 255, 255, 100)
        else:
            fill = QtGui.QColor(255, 255, 255, 58)

        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QBrush(fill))
        p.drawRoundedRect(rect, r, r)
        p.setBrush(QtCore.Qt.NoBrush)

        if self._in_error:
            p.setPen(QtGui.QPen(QtGui.QColor(210, 55, 55, 150), 1.0))

        elif self._focused:
            t = self._shimmer_t
            width = float(self.width())
            grad = QtGui.QLinearGradient(0.0, 0.0, width, 0.0)

            if t < 1.0:
                spread = 0.30
                accent_color = QtGui.QColor(185, 218, 248, 115)  # Soft blue
                peak_color = QtGui.QColor(255, 255, 255, 240)  # Bright white

                grad.setColorAt(0.0, accent_color)

                # Pre-peak
                pre = max(0.0, t - spread)
                if pre > 0.0:
                    grad.setColorAt(pre, accent_color)

                # Peak
                grad.setColorAt(max(0.0, t - spread * 0.12), peak_color)
                grad.setColorAt(min(1.0, t + spread * 0.12), peak_color)

                # Post-peak
                post = min(1.0, t + spread)
                if post < 1.0:
                    grad.setColorAt(post, accent_color)

                grad.setColorAt(1.0, accent_color)
            else:
                settled_color = QtGui.QColor(185, 218, 248, 130)
                grad.setColorAt(0.0, settled_color)
                grad.setColorAt(1.0, settled_color)

            p.setPen(QtGui.QPen(QtGui.QBrush(grad), 1.5))

        else:
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 105), 1.0))

        p.drawRoundedRect(rect, r, r)
        p.end()
        super().paintEvent(event)
