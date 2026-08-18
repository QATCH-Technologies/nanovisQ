from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.theme_manager import ThemeManager
from QATCH.ui.styles.typography import FONT_SANS_STACK

_RADIUS = 7.0


class QATCHSpinBox(QtWidgets.QSpinBox):
    """A QSpinBox styled to match the app's flat control system.

    Pixel-identical chrome to `QATCHLineEdit` (fill/border/focus-ring
    painted manually via `flat_paint.paint_flat_surface`, same 7px radius,
    the same token-driven default/hover/focus/error/disabled states) with
    Qt's ordinary native up/down step buttons - unlike `AnimatedSpinBox`,
    there is no odometer digit-roll animation and no custom recoloring
    chevron icons. Use this wherever a spin box needs to sit alongside
    QATCHLineEdit fields and read as the same kind of control rather than
    calling out its own distinct animated style.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._hovered = False
        self._in_error = False
        self.setObjectName("QATCHSpinBox")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._apply_text_qss()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    @staticmethod
    def _rgba(rgba) -> str:
        """Format a token (r, g, b, a) tuple as a CSS rgba() string."""
        return f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]})"

    def _apply_text_qss(self) -> None:
        """Style text color/font from the active palette. Fill/border/ring
        are painted in paintEvent (also from tokens)."""
        tok = ThemeManager.instance().tokens()
        text_color = tok["flat_text_muted"] if not self.isEnabled() else tok["flat_text"]
        self.setStyleSheet(
            "QSpinBox#QATCHSpinBox {"
            "  background: transparent;"
            "  border: none;"
            "  padding: 9px 6px 9px 12px;"
            f"  color: {self._rgba(text_color)};"
            f"  font-family: {FONT_SANS_STACK};"
            "  font-size: 13px;"
            f"  selection-background-color: {self._rgba(tok['flat_accent_weak'])};"
            f"  selection-color: {self._rgba(tok['flat_accent'])};"
            "}"
        )
        self.setCursor(
            QtCore.Qt.CursorShape.ForbiddenCursor
            if not self.isEnabled()
            else QtCore.Qt.CursorShape.ArrowCursor
        )

    def _on_theme_changed(self, _mode: Optional[str] = None) -> None:
        self._apply_text_qss()
        self.update()

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        super().setEnabled(enabled)
        self._apply_text_qss()
        self.update()

    def set_error(self, on: bool) -> None:
        """Toggles the visual error state (border/ring)."""
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
        """Paints the same flat fill/border/focus-ring as `QATCHLineEdit`,
        then draws ONLY the native up/down step buttons on top - not the
        full `CC_SpinBox` complex control, whose frame draw would erase the
        chrome just painted (the same `QStyleSheetStyle` quirk documented
        in `AnimatedSpinBox.paintEvent`: every widget in this app runs
        under a global QSS stylesheet, which forces that erase-on-panel-
        draw behavior even with a "transparent" background). The internal
        line-edit sub-control paints its own text/cursor afterward as an
        ordinary child widget - no extra call needed for that part.
        """
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

        opt = QtWidgets.QStyleOptionSpinBox()
        self.initStyleOption(opt)
        opt.subControls = QtWidgets.QStyle.SC_SpinBoxUp | QtWidgets.QStyle.SC_SpinBoxDown
        sp = QtWidgets.QStylePainter(self)
        sp.drawComplexControl(QtWidgets.QStyle.CC_SpinBox, opt)
