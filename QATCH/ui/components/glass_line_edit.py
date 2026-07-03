from __future__ import annotations
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.fonts import FONT_SANS
from QATCH.ui.styles.theme_manager import ThemeManager

_RADIUS = 7.0


class GlassLineEdit(QtWidgets.QLineEdit):
    """A QLineEdit styled to match the app's flat control system.

    Chrome (fill/border/focus-ring) is painted manually in `paintEvent` via
    `QATCH.ui.components.flat_paint.paint_flat_surface`; text/cursor
    rendering is left to the base `QLineEdit` implementation. Visual states
    - default, hover, focused, disabled, error - are resolved from the
    "flat_*" tokens in `QATCH.ui.styles.tokens` so the widget follows
    light/dark theme changes automatically.

    Leading icons and a trailing password-reveal button are supported via
    Qt's native `QLineEdit.addAction()` (no custom widget code needed -
    already used elsewhere in the app, e.g. `create_user_widget.py`).

    Attributes:
        _hovered (bool): True while the cursor is inside the widget.
        _in_error (bool): True while `set_error(True)` is active.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initializes the line edit with the flat control chrome."""
        super().__init__(parent)
        self._hovered: bool = False
        self._in_error: bool = False

        self.setFrame(False)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_Hover, True)
        self._apply_text_qss()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    @staticmethod
    def _rgba(rgba) -> str:
        """Format a token (r, g, b, a) tuple as a CSS rgba() string."""
        return f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]})"

    def _apply_text_qss(self) -> None:
        """Style text color, font, and selection from the active palette.
        Fill/border/ring are painted in paintEvent (also from tokens)."""
        tok = ThemeManager.instance().tokens()
        text_color = tok["flat_text_muted"] if not self.isEnabled() else tok["flat_text"]
        self.setStyleSheet(
            "QLineEdit {"
            "  background: transparent;"
            "  border: none;"
            "  padding: 9px 12px;"
            f"  color: {self._rgba(text_color)};"
            f"  font-family: '{FONT_SANS}';"
            "  font-size: 13px;"
            f"  selection-background-color: {self._rgba(tok['flat_accent_weak'])};"
            f"  selection-color: {self._rgba(tok['flat_accent'])};"
            "}"
            "QLineEdit QToolButton {"
            "  background: transparent;"
            "  border: none;"
            "}"
            "QLineEdit QToolButton:hover {"
            f"  background: {self._rgba(tok['flat_surface2'])};"
            "  border-radius: 12px;"
            "}"
        )
        self.setCursor(
            QtCore.Qt.CursorShape.ForbiddenCursor
            if not self.isEnabled()
            else QtCore.Qt.CursorShape.IBeamCursor
        )

    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_text_qss()
        self.update()

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        super().setEnabled(enabled)
        self._apply_text_qss()
        self.update()

    def set_error(self, on: bool) -> None:
        """Toggles the visual error state of the widget.

        Args:
            on (bool): If True, the widget paints with the error border/ring.
        """
        if on != self._in_error:
            self._in_error = on
            self.update()

    def enterEvent(self, event: QtCore.QEvent) -> None:
        super().enterEvent(event)
        self._hovered = True
        self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        super().leaveEvent(event)
        self._hovered = False
        self.update()

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Paints the flat fill/border/focus-ring, then delegates text and
        cursor rendering to the base QLineEdit implementation."""
        tok = ThemeManager.instance().tokens()

        if not self.isEnabled():
            fill = QtGui.QColor(*tok["flat_surface2"])
            border = QtGui.QColor(*tok["flat_border"])
            ring = None
        elif self._in_error:
            fill = QtGui.QColor(*tok["flat_surface"])
            border = QtGui.QColor(*tok["flat_error"])
            ring = QtGui.QColor(*tok["flat_error_ring"])
        elif self.hasFocus():
            fill = QtGui.QColor(*tok["flat_surface"])
            border = QtGui.QColor(*tok["flat_accent"])
            ring = QtGui.QColor(*tok["flat_accent_ring"])
        elif self._hovered:
            fill = QtGui.QColor(*tok["flat_surface"])
            border = QtGui.QColor(*tok["flat_border_strong"])
            ring = None
        else:
            fill = QtGui.QColor(*tok["flat_surface"])
            border = QtGui.QColor(*tok["flat_border"])
            ring = None

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        paint_flat_surface(self, radius=_RADIUS, fill=fill, border=border, ring=ring, painter=p)
        p.end()

        super().paintEvent(event)
