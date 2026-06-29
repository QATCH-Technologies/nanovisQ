from typing import Optional, Dict
from PyQt5 import QtCore, QtGui, QtWidgets


class SavedStateDot(QtWidgets.QWidget):
    """A small glowing status dot that reflects a field's save state.

    This widget uses property animations to provide visual feedback for four
    distinct states:
        * ``querying``: Red, pulsing (awaiting device response).
        * ``blank``: Quiet gray (no pending changes/initial state).
        * ``unsaved``: Amber, gently pulsing (changes waiting to be saved).
        * ``saved``: Green, steady (confirmed state).

    Attributes:
        _COLORS (Dict[str, QtGui.QColor]): Color mapping for each state.
        _PULSING (tuple): Collection of states that require active animation.
        _SIZE (int): The fixed diameter of the widget.
    """

    _COLORS: Dict[str, QtGui.QColor] = {
        "blank": QtGui.QColor(150, 165, 180),
        "unsaved": QtGui.QColor(240, 170, 50),
        "saved": QtGui.QColor(60, 190, 120),
        "querying": QtGui.QColor(228, 70, 70),
    }
    _PULSING: tuple = ("unsaved", "querying")
    _SIZE: int = 14

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initializes the animation controllers and visual state."""
        super().__init__(parent)
        self.setFixedSize(self._SIZE, self._SIZE)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._state: str = "blank"
        self._glow: float = 0.0
        self._flash: float = 0.0

        # Pulse animation
        self._pulse = QtCore.QVariantAnimation(self)
        self._pulse.setStartValue(0.25)
        self._pulse.setEndValue(1.0)
        self._pulse.setDuration(900)
        self._pulse.setEasingCurve(QtCore.QEasingCurve.InOutSine)
        self._pulse.setLoopCount(-1)
        self._pulse.valueChanged.connect(self._on_pulse)

        # Attention flash animation
        self._flash_anim = QtCore.QVariantAnimation(self)
        self._flash_anim.setStartValue(1.0)
        self._flash_anim.setEndValue(0.0)
        self._flash_anim.setDuration(520)
        self._flash_anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._flash_anim.valueChanged.connect(self._on_flash)

    def state(self) -> str:
        """Returns the current status state identifier."""
        return self._state

    def set_state(self, state: str) -> None:
        """Updates the widget state and manages animation lifecycle.

        If the state transitions to a pulsing mode, the pulse animation is
        started. Otherwise, the animation is stopped. This method is
        idempotent to prevent unnecessary animation restarts.
        """
        if state not in self._COLORS:
            state = "blank"
        if state == self._state:
            return

        self._state = state
        if state in self._PULSING:
            self._pulse.stop()
            self._pulse.start()
        else:
            self._pulse.stop()
            self._glow = 1.0 if state == "saved" else 0.0
        self.update()

    def flash(self, times: int = 3) -> None:
        """Plays a brief attention-grabbing pulse (e.g., for navigation warnings)."""
        self._flash_anim.stop()
        self._flash_anim.setLoopCount(max(1, times))
        self._flash_anim.start()

    def _on_pulse(self, v: float) -> None:
        self._glow = v
        self.update()

    def _on_flash(self, v: float) -> None:
        self._flash = v
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Renders the status dot with dynamic glow and flash overlays."""
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        c = QtGui.QColor(self._COLORS[self._state])
        cx, cy = self.width() / 2.0, self.height() / 2.0

        # Outer halo
        glow = max(self._glow, self._flash)
        if glow > 0.01:
            halo = QtGui.QColor(c)
            halo.setAlphaF(0.35 * glow)
            radius = 4.0 + 3.0 * glow
            grad = QtGui.QRadialGradient(cx, cy, radius)
            grad.setColorAt(0.0, halo)
            transparent = QtGui.QColor(c)
            transparent.setAlpha(0)
            grad.setColorAt(1.0, transparent)
            p.setBrush(QtGui.QBrush(grad))
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

        # Core dot
        core_r = 3.4
        p.setBrush(QtGui.QBrush(c))
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 170), 1.0))
        p.drawEllipse(QtCore.QPointF(cx, cy), core_r, core_r)
        p.end()
