"""
qatch_stepper_field.py

A tiny ▲/▼ stepper affordance embedded inside an existing `QATCHLineEdit`,
for compact numeric fields (e.g. the Advanced Options "Difference Factor"
value) that need increment/decrement buttons without growing into a full
`QAbstractSpinBox`.

Follows the standard Qt "widget inside a QLineEdit" technique: the stepper
widget is parented directly onto the line edit itself and positioned via
`move()`/`resize()` at its right edge (kept in sync via an installed event
filter rather than subclassing `QATCHLineEdit`), while
`QLineEdit.setTextMargins()` reserves room so typed text never runs
underneath it.

Painted flat and themed via `ThemeManager` - same connect-to-themeChanged /
re-derive-from-tokens() pattern as `QATCH.ui.components.qatch_toggle`.

Usage
-----
    from QATCH.ui.components.qatch_stepper_field import attach_stepper

    stepper = attach_stepper(
        self.tbox_diff_factor, step=0.05, minimum=0.5, maximum=2.0, decimals=3,
    )
    # Keep the reference around to toggle it in lockstep with the field:
    self.tbox_diff_factor.setEnabled(False)
    stepper.setEnabled(False)

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.qatch_line_edit import QATCHLineEdit
from QATCH.ui.styles.theme_manager import ThemeManager

_RIGHT_GAP = 4  # px between the stepper's right edge and the line edit's own edge


class _StepperButtons(QtWidgets.QWidget):
    """Stacked ▲/▼ affordance meant to be parented inside a `QLineEdit`.

    Emits `stepUp`/`stepDown` on click of the respective glyph. Painted by
    hand (no child QToolButtons) so the two hit regions fit a very small
    footprint. Repositions itself against its parent's right edge whenever
    the parent resizes (see `eventFilter`, installed by `attach_stepper`).
    """

    stepUp = QtCore.pyqtSignal()
    stepDown = QtCore.pyqtSignal()

    _WIDTH = 15
    _HEIGHT = 20

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setFixedSize(self._WIDTH, self._HEIGHT)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self._hover_zone: Optional[str] = None  # "up" | "down" | None
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    # ------------------------------------------------------------------
    # Positioning (driven by the host line edit's resize, via eventFilter
    # installed by attach_stepper - see module docstring).
    # ------------------------------------------------------------------
    def reposition(self) -> None:
        host = self.parentWidget()
        if host is None:
            return
        self.move(
            host.width() - self.width() - _RIGHT_GAP,
            int((host.height() - self.height()) / 2),
        )

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if obj is self.parentWidget() and event.type() == QtCore.QEvent.Type.Resize:
            self.reposition()
        return False

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------
    def _on_theme_changed(self, _mode: str) -> None:
        self.update()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def _zone_at(self, pos: QtCore.QPoint) -> str:
        return "up" if pos.y() < self.height() / 2 else "down"

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if not self.isEnabled():
            return
        if self._zone_at(event.pos()) == "up":
            self.stepUp.emit()
        else:
            self.stepDown.emit()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._hover_zone = self._zone_at(event.pos()) if self.isEnabled() else None
        self.update()

    def enterEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        self._hover_zone = None
        self.update()

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        super().setEnabled(enabled)
        self.setCursor(
            QtCore.Qt.CursorShape.PointingHandCursor
            if enabled
            else QtCore.Qt.CursorShape.ForbiddenCursor
        )
        if not enabled:
            self._hover_zone = None
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        tok = ThemeManager.instance().tokens()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        if not self.isEnabled():
            p.setOpacity(0.4)

        base = QtGui.QColor(*tok["flat_text_muted"])
        accent = QtGui.QColor(*tok["flat_accent"])

        w = self.width()
        h = self.height()
        tri_w, tri_h = 7.0, 4.0

        def draw_tri(cy: float, pointing_up: bool, zone: str) -> None:
            color = accent if self._hover_zone == zone and self.isEnabled() else base
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QBrush(color))
            cx = w / 2.0
            path = QtGui.QPainterPath()
            if pointing_up:
                path.moveTo(cx - tri_w / 2, cy + tri_h / 2)
                path.lineTo(cx + tri_w / 2, cy + tri_h / 2)
                path.lineTo(cx, cy - tri_h / 2)
            else:
                path.moveTo(cx - tri_w / 2, cy - tri_h / 2)
                path.lineTo(cx + tri_w / 2, cy - tri_h / 2)
                path.lineTo(cx, cy + tri_h / 2)
            path.closeSubpath()
            p.drawPath(path)

        draw_tri(h * 0.28, True, "up")
        draw_tri(h * 0.72, False, "down")
        p.end()


def attach_stepper(
    line_edit: QATCHLineEdit,
    *,
    step: float,
    minimum: float,
    maximum: float,
    decimals: int = 3,
) -> _StepperButtons:
    """Embeds a ▲/▼ stepper pair inside `line_edit`'s right edge.

    Increment/decrement clicks parse the current text as a float, clamp it
    to `[minimum, maximum]`, write the new value back with `setText`, and
    then emit `line_edit.editingFinished` so the field's normal commit path
    (whatever is already connected to `editingFinished`) runs exactly as if
    the user had pressed Enter or moved focus away.

    Args:
        line_edit: The `QATCHLineEdit` to embed the stepper into.
        step: Amount added/subtracted per click.
        minimum: Lower clamp bound (inclusive).
        maximum: Upper clamp bound (inclusive).
        decimals: Digits after the decimal point when reformatting the value.

    Returns:
        The `_StepperButtons` widget, parented to `line_edit`. Keep the
        reference and call `.setEnabled(...)` on it wherever `line_edit`
        itself is enabled/disabled - the two are not linked automatically.
    """
    buttons = _StepperButtons(line_edit)
    line_edit.installEventFilter(buttons)
    line_edit.setTextMargins(0, 0, buttons.width() + _RIGHT_GAP, 0)
    buttons.reposition()

    def _step(delta: float) -> None:
        if not line_edit.isEnabled():
            return
        try:
            value = float(line_edit.text())
        except ValueError:
            value = minimum
        value = max(minimum, min(maximum, round(value + delta, decimals)))
        line_edit.setText(f"{value:.{decimals}f}")
        line_edit.editingFinished.emit()

    buttons.stepUp.connect(lambda: _step(step))
    buttons.stepDown.connect(lambda: _step(-step))

    return buttons
