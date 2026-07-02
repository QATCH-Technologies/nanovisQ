from __future__ import annotations
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.styles.theme_manager import ThemeManager

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
        self._apply_text_qss()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    @staticmethod
    def _rgba(rgba) -> str:
        """Format a token (r, g, b, a) tuple as a CSS rgba() string."""
        return f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]})"

    def _apply_text_qss(self) -> None:
        """Style text color and selection from the active palette. Fills and
        borders are painted in paintEvent (also from tokens)."""
        tok = ThemeManager.instance().tokens()
        self.setStyleSheet(
            "QLineEdit {"
            "  background: transparent;"
            "  border: none;"
            "  padding: 0px 15px;"
            f"  color: {self._rgba(tok['input_glass_text'])};"
            "  font-size: 10pt;"
            f"  selection-background-color: {self._rgba(tok['input_glass_selection_bg'])};"
            f"  selection-color: {self._rgba(tok['text_primary'])};"
            "}"
            "QLineEdit QToolButton {"
            "  background: transparent;"
            "  border: none;"
            "}"
            "QLineEdit QToolButton:hover {"
            f"  background: {self._rgba(tok['plot_icon_btn_hover_bg'])};"
            "  border-radius: 12px;"
            "}"
        )

    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_text_qss()
        self.update()

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
        tok = ThemeManager.instance().tokens()
        if self._in_error:
            fill = QtGui.QColor(*tok["input_glass_fill_error"])
        elif self._focused:
            fill = QtGui.QColor(*tok["input_glass_fill_focus"])
        else:
            fill = QtGui.QColor(*tok["input_glass_fill"])

        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QBrush(fill))
        p.drawRoundedRect(rect, r, r)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        if self._in_error:
            p.setPen(QtGui.QPen(QtGui.QColor(*tok["input_glass_border_error"]), 1.0))

        elif self._focused:
            t = self._shimmer_t
            width = float(self.width())
            grad = QtGui.QLinearGradient(0.0, 0.0, width, 0.0)

            if t < 1.0:
                spread = 0.30
                accent_color = QtGui.QColor(*tok["input_glass_shimmer_accent"])  # Soft accent
                peak_color = QtGui.QColor(*tok["input_glass_shimmer_peak"])  # Bright peak

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
                settled_color = QtGui.QColor(*tok["input_glass_shimmer_accent"])
                grad.setColorAt(0.0, settled_color)
                grad.setColorAt(1.0, settled_color)

            p.setPen(QtGui.QPen(QtGui.QBrush(grad), 1.5))

        else:
            p.setPen(QtGui.QPen(QtGui.QColor(*tok["input_glass_border"]), 1.0))

        p.drawRoundedRect(rect, r, r)
        p.end()
        super().paintEvent(event)
