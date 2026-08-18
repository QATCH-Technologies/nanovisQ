"""
QATCH.ui.components.qatch_toggle.py

Animated toggle switch component.

Provides the :class:`QATCHToggle` widget, an animated pill-shaped toggle
switch designed to match the application's flat control system. The toggle
uses theme-defined `flat_*` tokens from
`QATCH.ui.styles.tokens` so its appearance remains synchronized with both
light and dark themes.

The track color smoothly interpolates between the neutral `flat_track` color
when the toggle is off and the `flat_accent` color when it is on. The thumb
slides between the left and right positions using an `OutCubic` easing curve
over 150 milliseconds.

The widget inherits from `QAbstractButton` and exposes the standard checked
state and `toggled(bool)` signal, making it suitable as a drop-in
replacement for checkbox-style controls where only the checked state is
required.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-18
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.theme_manager import ThemeManager


class QATCHToggle(QtWidgets.QAbstractButton):
    """Display an animated, pill-shaped toggle switch.

    The toggle provides a compact alternative to `QCheckBox` while
    retaining the standard `QAbstractButton` checked-state API, including
    the `toggled(bool)` signal. The thumb smoothly animates between the
    off and on positions when the checked state changes.

    The widget supports a compact geometry variant for secondary or
    space-constrained controls. Its appearance is resolved from the active
    application theme and automatically refreshed when the theme changes.

    Attributes:
        _anim_t: Normalized animated thumb position, where `0.0` represents
            the off state and `1.0` represents the on state.
        _anim: QVariantAnimation responsible for animating the thumb between
            its off and on positions.
    """

    # Geometry
    _TRACK_W: int = 42
    _TRACK_H: int = 23
    _THUMB_D: int = 18
    # Smaller footprint used when `compact=True` is passed to __init__
    _COMPACT_TRACK_W: int = 34
    _COMPACT_TRACK_H: int = 19
    _COMPACT_THUMB_D: int = 15

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        compact: bool = False,
    ) -> None:
        """Initialize the toggle switch.

        Args:
            parent: Optional parent widget.
            compact: If `True`, use the smaller compact track and thumb
                geometry. This variant is intended for inline or secondary
                controls where a reduced visual footprint is desirable.
                Defaults to `False`, preserving the standard toggle size.
        """
        super().__init__(parent)
        if compact:
            self._TRACK_W = self._COMPACT_TRACK_W
            self._TRACK_H = self._COMPACT_TRACK_H
            self._THUMB_D = self._COMPACT_THUMB_D
        self.setCheckable(True)
        self.setFixedSize(self._TRACK_W, self._TRACK_H)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self._anim_t: float = 0.0

        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_anim_step)

        self.toggled.connect(self._start_anim)

        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        """Refresh the toggle after the application theme changes.

        Schedules the widget for repainting so its track and thumb are rendered
        using the newly active theme tokens.

        Args:
            _mode: Theme mode identifier emitted by the `themeChanged` signal.
                The value is not otherwise used by this handler.

        Returns:
            None.
        """
        self.update()

    def _start_anim(self, checked: bool) -> None:
        """Start the thumb animation toward the new checked state.

        Stops any currently running animation and starts a new animation from
        the thumb's current interpolated position. This allows rapid state
        changes to transition smoothly without jumping back to the previous
        endpoint.

        Args:
            checked: Whether the toggle is transitioning to the checked (on)
                state. A checked state targets `1.0`; an unchecked state targets
                `0.0`.

        Returns:
            None.
        """
        self._anim.stop()
        self._anim.setStartValue(float(self._anim_t))
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def _on_anim_step(self, v: float) -> None:
        """Update the animated thumb position and repaint the toggle.

        Args:
            v: Current interpolated animation value, where `0.0` represents
                the off position and `1.0` represents the on position.

        Returns:
            None.
        """
        self._anim_t = v
        self.update()

    def setChecked(self, checked: bool) -> None:
        """Set the toggle's checked state and synchronize its thumb position.

        The animated thumb position is snapped immediately to the requested
        checked state before delegating to `QAbstractButton.setChecked`. This
        prevents an intermediate animation frame from being rendered when the
        checked state is changed programmatically.

        Args:
            checked: `True` to set the toggle to the checked (on) state, or
                `False` to set it to the unchecked (off) state.

        Returns:
            None.
        """
        self._anim_t = 1.0 if checked else 0.0
        super().setChecked(checked)

    @staticmethod
    def _lerp(a: int, b: int, t: float) -> int:
        """Linearly interpolate between two integer values.

        Args:
            a: Starting integer value.
            b: Ending integer value.
            t: Interpolation factor. A value of `0.0` returns `a` and a
                value of `1.0` returns `b`. Intermediate values produce
                proportional values between the two endpoints.

        Returns:
            The interpolated value converted to an integer.
        """
        return int(a + (b - a) * t)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Render the toggle track, focus ring, shadow, and animated thumb.

        Draws the toggle using the active theme's flat-control tokens. The track
        color is linearly interpolated between the inactive track and active
        accent colors based on the current animation position. When focused, a
        theme-defined focus ring is rendered around the track.

        The thumb is positioned between the left and right track endpoints using
        the same animation value. A subtle drop shadow is drawn beneath the thumb,
        followed by the thumb face, which uses the theme's knob color when off and
        white when on.

        Disabled toggles are rendered with reduced opacity.

        Args:
            event: Qt paint event provided by Qt when the widget needs to be
                repainted. The event is not otherwise used by this implementation.

        Returns:
            None.
        """
        t = self._anim_t
        w, h = self.width(), self.height()
        r = h / 2.0  # track corner radius - full pill

        tok = ThemeManager.instance().tokens()

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        if not self.isEnabled():
            p.setOpacity(0.45)

        # Track fill
        lo, hi = tok["flat_track"], tok["flat_accent"]
        track_color = QtGui.QColor(
            self._lerp(lo[0], hi[0], t),
            self._lerp(lo[1], hi[1], t),
            self._lerp(lo[2], hi[2], t),
            self._lerp(lo[3], hi[3], t),
        )
        ring = QtGui.QColor(*tok["flat_accent_ring"]) if self.hasFocus() else None
        paint_flat_surface(
            self,
            radius=r,
            fill=track_color,
            border=track_color,
            border_width=0.0,
            ring=ring,
            painter=p,
        )
        # Thumb
        margin = (h - self._THUMB_D) / 2.0
        x_left = margin
        x_right = w - margin - self._THUMB_D
        thumb_x = x_left + (x_right - x_left) * t
        thumb = QtCore.QRectF(thumb_x, margin, self._THUMB_D, self._THUMB_D)

        # Drop shadow
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QBrush(QtGui.QColor(*tok["flat_shadow"])))
        p.drawEllipse(thumb.adjusted(0.0, 1.0, 0.0, 1.0))

        knob_color = (
            QtGui.QColor(255, 255, 255) if self.isChecked() else QtGui.QColor(*tok["flat_knob"])
        )
        p.setBrush(QtGui.QBrush(knob_color))
        p.drawEllipse(thumb)

        p.end()

    def sizeHint(self) -> QtCore.QSize:
        """Return the preferred size of the toggle.

        The returned dimensions reflect the toggle's active track geometry,
        including the compact dimensions when the widget was initialized with
        `compact=True`.

        Returns:
            A `QSize` containing the toggle's preferred width and height.
        """
        return QtCore.QSize(self._TRACK_W, self._TRACK_H)


class LabeledToggle(QtWidgets.QWidget):
    """Display a QATCHToggle with an adjacent text label.

    Combines a :class:`QATCHToggle` and `QLabel` in a horizontal layout
    while exposing the subset of the `QCheckBox` API used by the
    application. This allows the composite widget to serve as a drop-in
    replacement for checkbox-style controls without requiring changes to
    existing call sites.

    The label can be positioned on either side of the toggle, and the entire
    control supports a compact presentation for secondary or space-constrained
    UI elements.

    Attributes:
        toggle: The internal :class:`QATCHToggle` that provides the toggle
            behavior and checked state.
        label: The `QLabel` displaying the control's descriptive text.
        toggled: Signal forwarded from `toggle` that is emitted whenever
            the toggle's checked state changes.
        clicked: Signal forwarded from `toggle` that is emitted whenever
            the toggle is clicked by the user.
    """

    def __init__(
        self,
        text: str = "",
        parent=None,
        *,
        label_left: bool = False,
        compact: bool = False,
    ) -> None:
        """Initialize the labeled toggle.

        Args:
            text: Text to display next to the toggle.
            parent: Optional parent widget.
            label_left: If `True`, place the label to the left of the
                toggle. If `False`, place the label to the right.
            compact: If `True`, use the compact geometry of
                :class:`QATCHToggle` and a smaller label font. This is
                intended for secondary or inline controls. Defaults to
                `False`.
        """
        super().__init__(parent)
        self.toggle = QATCHToggle(self, compact=compact)
        self.label = QtWidgets.QLabel(text, self)
        self.label.setObjectName("CtrlToggleLabel")
        self.label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        if compact:
            self.label.setStyleSheet("QLabel#CtrlToggleLabel { font-size: 11px; }")

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6 if compact else 8)
        if label_left:
            lay.addWidget(self.label)
            lay.addWidget(self.toggle)
            lay.addStretch()
        else:
            lay.addWidget(self.toggle)
            lay.addWidget(self.label)
            lay.addStretch()
        self.toggled = self.toggle.toggled
        self.clicked = self.toggle.clicked

    def isChecked(self) -> bool:
        """Return the current checked state of the toggle.

        Provides `QCheckBox`-compatible API behavior by forwarding the query
        to the internal :class:`QATCHToggle`.

        Returns:
            `True` if the toggle is checked; otherwise, `False`.
        """
        return self.toggle.isChecked()

    def setChecked(self, checked: bool) -> None:
        """Set the checked state of the toggle.

        Provides `QCheckBox`-compatible API behavior by forwarding the requested
        state to the internal :class:`QATCHToggle`.

        Args:
            checked: `True` to check the toggle, or `False` to uncheck it.

        Returns:
            None.
        """
        self.toggle.setChecked(checked)

    def setText(self, text: str) -> None:
        """Set the text displayed beside the toggle.

        Provides `QCheckBox`-compatible API behavior by forwarding the supplied
        text to the internal label widget.

        Args:
            text: New text to display next to the toggle.

        Returns:
            None.
        """
        self.label.setText(text)

    def text(self) -> str:
        """Return the current text displayed beside the toggle.

        Returns:
            The text currently displayed by the internal label.
        """
        return self.label.text()

    def setEnabled(self, enabled: bool) -> None:
        """Set the enabled state of the toggle and its label.

        Applies the requested enabled state to the composite widget and explicitly
        propagates it to both child widgets.

        Args:
            enabled: `True` to enable the widget and its children, or `False`
                to disable them.

        Returns:
            None.
        """
        super().setEnabled(enabled)
        self.toggle.setEnabled(enabled)
        self.label.setEnabled(enabled)
