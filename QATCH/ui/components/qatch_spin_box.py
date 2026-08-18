"""QATCH flat-styled spin box controls.

Provides spin box widgets that integrate with the application's flat control
system and share the same visual language as `QATCHLineEdit`. The controls
use theme-driven tokens and the shared `paint_flat_surface` rendering
recipe to provide consistent fills, borders, focus rings, hover states,
error states, and disabled states across light and dark themes.

The primary widget, :class:`QATCHSpinBox`, extends `QSpinBox` while
preserving Qt's standard numeric input behavior and native up/down step
buttons. Unlike `AnimatedSpinBox`, it does not provide animated digit
transitions or custom step-button icons. It is intended for numeric fields
that should visually match the application's standard flat line-edit
controls.

`QATCHSpinBox` separates its rendering responsibilities between the custom
flat-control surface and Qt's native spin-box controls. The widget paints its
background, border, and focus/error ring manually through
`paint_flat_surface`. The native up/down step buttons are then rendered
using Qt's `QStylePainter`. This avoids a `QStyleSheetStyle` interaction
where drawing the complete `CC_SpinBox` control can overwrite the custom
surface under the application's global stylesheet.

The widget responds dynamically to theme and state changes. Text and
selection styling is applied through a small widget-specific stylesheet,
while the control surface is resolved from the active theme tokens during
painting.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-18
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.theme_manager import ThemeManager
from QATCH.ui.styles.typography import FONT_SANS_STACK

_RADIUS = 7.0


class QATCHSpinBox(QtWidgets.QSpinBox):
    """Provide a themed spin box matching the application's line edits.

    Extends `QSpinBox` with manually painted control chrome so that the
    widget visually matches :class:`QATCHLineEdit`. The control uses the
    shared :func:`flat_paint.paint_flat_surface` recipe with the same radius
    and theme-token-driven states for default, hover, focus, error, and
    disabled conditions.

    Unlike `AnimatedSpinBox`, this widget retains Qt's native up/down
    controls and does not provide odometer-style digit animations or custom
    chevron icons. It is intended for interfaces where numeric input should
    visually integrate with adjacent `QATCHLineEdit` controls.

    Attributes:
        _hovered: Whether the mouse cursor is currently hovering over the
            spin box.
        _in_error: Whether the spin box is currently displaying its error
            state.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the flat-styled spin box.

        Args:
            parent: Optional parent widget.

        Returns:
            None.
        """
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
        """Format an RGBA color token as a CSS `rgba()` string.

        Args:
            rgba: A four-element sequence containing red, green, blue, and alpha
                channel values in that order.

        Returns:
            A CSS-compatible `rgba()` color string.
        """
        return f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]})"

    def _apply_text_qss(self) -> None:
        """Apply theme-aware text and selection styling to the spin box.

        Updates the widget's stylesheet using the current theme tokens while
        leaving the fill, border, and focus ring to be rendered by
        `paintEvent`. The text color is selected based on the widget's enabled
        state, and the text, font, padding, and selection colors are configured to
        match the application's flat control system.

        The cursor is also updated to indicate whether the control is enabled.

        Returns:
            None.
        """
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

    def _on_theme_changed(self, _mode: str | None = None) -> None:
        """Refresh the spin box after the application theme changes.

        Reapplies the theme-dependent text and selection styling and schedules the
        widget for repainting so its manually painted surface reflects the new
        theme tokens.

        Args:
            _mode: Optional theme mode identifier emitted by the theme manager.
                The value is not otherwise used by this handler.

        Returns:
            None.
        """
        self._apply_text_qss()
        self.update()

    def setEnabled(self, enabled: bool) -> None:
        """Set the enabled state and refresh the control's appearance.

        Updates the widget's enabled state, reapplies the corresponding text and
        cursor styling, and schedules the control for repainting.

        Args:
            enabled: `True` to enable the spin box, or `False` to disable it.

        Returns:
            None.
        """
        super().setEnabled(enabled)
        self._apply_text_qss()
        self.update()

    def set_error(self, on: bool) -> None:
        """Set the visual error state of the spin box.

        When enabled, the error state changes the border and focus-ring styling
        rendered by the widget. If the requested state differs from the current
        state, the widget is scheduled for repainting.

        Args:
            on: `True` to enable the visual error state, or `False` to clear
                it.

        Returns:
            None.
        """
        if on != self._in_error:
            self._in_error = on
            self.update()

    def enterEvent(self, event: QtCore.QEvent) -> None:
        """Handle the mouse cursor entering the spin box.

        Updates the internal hover state and schedules the widget for repainting
        so the hover-specific surface styling can be rendered.

        Args:
            event: Qt event generated when the mouse cursor enters the widget.

        Returns:
            None.
        """
        super().enterEvent(event)
        self._hovered = True
        self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        """Handle the mouse cursor leaving the spin box.

        Clears the internal hover state and schedules the widget for repainting so
        the normal surface styling can be restored.

        Args:
            event: Qt event generated when the mouse cursor leaves the widget.

        Returns:
            None.
        """
        super().leaveEvent(event)
        self._hovered = False
        self.update()

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        """Handle the spin box receiving keyboard focus.

        Delegates focus handling to the base class and schedules the widget for
        repainting so its focus-ring styling can be displayed.

        Args:
            event: Qt focus event generated when the widget receives focus.

        Returns:
            None.
        """
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        """Handle the spin box losing keyboard focus.

        Delegates focus handling to the base class and schedules the widget for
        repainting so that any active focus-ring styling is removed.

        Args:
            event: Qt focus event generated when the widget loses focus.

        Returns:
            None.
        """
        super().focusOutEvent(event)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the spin box's surface and native step buttons.

        Renders the fill, border, and optional focus or error ring using the
        shared `paint_flat_surface` recipe and the active theme tokens. The
        surface styling follows the same state hierarchy as `QATCHLineEdit`:
        disabled, error, focused, hovered, and default.

        After the custom surface is painted, only the native up/down spin box
        sub-controls are rendered using Qt's style system. The full
        `CC_SpinBox` control is intentionally not painted as a single unit,
        because the global application stylesheet can cause the style to erase
        the custom surface when drawing the complete control.

        The internal line-edit portion of the spin box remains responsible for
        rendering its own text and cursor as a child widget.

        Args:
            event: Qt paint event provided when the widget needs to be repainted.
                The event is not otherwise used by this implementation.

        Returns:
            None.
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
