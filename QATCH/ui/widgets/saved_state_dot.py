"""
QATCH.ui.widgets.saved_state_dot.py

Themed animated status indicator for saved-state feedback.

This module provides the :class:`SavedStateDot` widget, a compact status
indicator designed to communicate the current save or load state of a field
or configuration item. The indicator uses theme-aware colors, continuous
pulse animations for active states, steady glows for settled states, and a
transient flash animation for drawing attention to important events.

Supported states include blank, unsaved, loading, saved, querying, and
error. The widget integrates with :class:`ThemeManager` so its appearance
automatically follows the application's active light or dark theme.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-21
"""

from typing import ClassVar

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.styles.theme_manager import ThemeManager


class SavedStateDot(QtWidgets.QWidget):
    """A small glowing status indicator that reflects a field's save state.

    The widget displays one of several semantic states using theme-aware
    colors and optional animations. Active states pulse continuously, while
    settled states display a steady glow. A separate flash animation can be
    triggered to draw attention to an event such as a navigation warning or
    failed save operation.

    Attributes:
        _COLORS (Dict[str, Dict[str, QtGui.QColor]]): Theme-specific color
            mappings keyed first by theme mode and then by state name.
        _PULSING (tuple): State identifiers that use continuous pulsing
            animation.
        _STEADY_GLOW (tuple): State identifiers that display a steady glow
            without continuous animation.
        _SIZE (int): Fixed width and height of the status indicator in
            pixels.
        _state (str): Current status state identifier.
        _glow (float): Current intensity of the continuous glow animation.
        _flash (float): Current intensity of the transient attention flash.
    """

    _COLORS: ClassVar[dict[str, dict[str, QtGui.QColor]]] = {
        "light": {
            "blank": QtGui.QColor(150, 165, 180),
            "unsaved": QtGui.QColor(240, 170, 50),
            "loading": QtGui.QColor(60, 190, 120),
            "saved": QtGui.QColor(60, 190, 120),
            "querying": QtGui.QColor(228, 70, 70),
            "error": QtGui.QColor(228, 70, 70),
        },
        "dark": {
            "blank": QtGui.QColor(170, 180, 195),
            "unsaved": QtGui.QColor(255, 185, 60),
            "loading": QtGui.QColor(80, 210, 140),
            "saved": QtGui.QColor(80, 210, 140),
            "querying": QtGui.QColor(240, 90, 90),
            "error": QtGui.QColor(240, 90, 90),
        },
    }
    _PULSING: tuple = ("unsaved", "querying", "loading")
    _STEADY_GLOW: tuple = ("saved", "error")
    _SIZE: int = 14

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the status indicator and its animation controllers.

        Configures the widget's fixed dimensions and translucent background,
        initializes the indicator in the `"blank"` state, and creates the
        pulse and attention-flash animations. The widget also subscribes to theme
        changes so its appearance is repainted using the active theme colors.

        Args:
            parent: Optional parent widget that owns this status indicator.
        """
        super().__init__(parent)
        self.setFixedSize(self._SIZE, self._SIZE)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._state: str = "blank"
        self._glow: float = 0.0
        self._flash: float = 0.0

        ThemeManager.instance().themeChanged.connect(lambda _mode: self.update())

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
        """Return the current status state identifier.

        Returns:
            The identifier of the currently active status state, such as
            `"blank"`, `"unsaved"`, `"loading"`, `"saved"`,
            `"querying"`, or `"error"`.
        """
        return self._state

    def set_state(self, state: str) -> None:
        """Set the current status state and update its animation lifecycle.

        Invalid state identifiers are treated as `"blank"`. Reapplying the
        current state is ignored to avoid unnecessarily restarting animations.
        Pulsing states start the continuous pulse animation, while steady states
        stop it and establish the appropriate fixed glow intensity.

        Args:
            state: Status state identifier to display. Supported values are
                `"blank"`, `"unsaved"`, `"loading"`, `"saved"`,
                `"querying"`, and `"error"`.
        """
        if state not in self._COLORS["light"]:
            state = "blank"
        if state == self._state:
            return

        self._state = state
        if state in self._PULSING:
            self._pulse.stop()
            self._pulse.start()
        else:
            self._pulse.stop()
            self._glow = 1.0 if state in self._STEADY_GLOW else 0.0
        self.update()

    def flash(self, times: int = 3) -> None:
        """Play a brief attention-grabbing flash animation.

        The transient flash is independent of the widget's current state and
        can be used to draw immediate attention to an event without changing
        the underlying status. The animation repeats for the requested number
        of cycles, with at least one cycle always performed.

        Args:
            times: Number of flash animation cycles to play. Values less than
                one are clamped to one.
        """
        self._flash_anim.stop()
        self._flash_anim.setLoopCount(max(1, times))
        self._flash_anim.start()

    def _on_pulse(self, v: float) -> None:
        """Update the continuous glow intensity from the pulse animation.

        Args:
            v: Current normalized glow intensity emitted by the pulse animation.
        """
        self._glow = v
        self.update()

    def _on_flash(self, v: float) -> None:
        """Update the transient flash intensity from the flash animation.

        Args:
            v: Current normalized flash intensity emitted by the attention-flash
                animation.
        """
        self._flash = v
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the themed status dot and its dynamic glow effects.

        Renders the indicator using the active theme's color for the current
        state. When either the continuous glow or transient flash is active, a
        radial halo is drawn around the core dot. The core is then rendered as
        an antialiased colored circle with a subtle light outline.

        Args:
            event: Qt paint event describing the region that needs to be
                repainted.
        """
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        mode = ThemeManager.instance().mode().value
        c = QtGui.QColor(self._COLORS[mode][self._state])
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
