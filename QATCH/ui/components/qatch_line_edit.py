"""
QATCH.common.ui.qatch_line_edit.py

Flat-styled single-line text input for the nanovisQ application.

Provides :class:`QATCHLineEdit`, a `QLineEdit` subclass that matches the
application's flat control system. The widget separates its visual chrome
from Qt's native text-editing behavior: the fill, border, and focus ring are
painted manually through
`QATCH.ui.components.flat_paint.paint_flat_surface`, while the base
`QLineEdit` implementation remains responsible for rendering text, the
cursor, selection, and native line-edit actions.

The widget resolves its visual states from the active theme's `flat_*`
tokens, allowing it to automatically follow light/dark theme changes without
maintaining separate palettes. Supported states include default, hover,
focused, disabled, and error.

The line edit also preserves Qt's standard action system, allowing callers to
add leading icons, password-reveal buttons, or other `QAction` instances
through `QLineEdit.addAction()` without requiring custom widget handling.

Example:
    Create a standard flat line edit::

        field = QATCHLineEdit(parent)
        field.setPlaceholderText("Enter a value")

    Add a leading icon using Qt's native action support::

        action = field.addAction(
            QtGui.QIcon(icon_path),
            QtWidgets.QLineEdit.LeadingPosition,
        )

    Display an error state::

        field.set_error(True)

    Clear the error state::

        field.set_error(False)

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


class QATCHLineEdit(QtWidgets.QLineEdit):
    """Single-line text input styled for the QATCH flat control system.

    Extends :class:`QtWidgets.QLineEdit` with theme-aware flat-design chrome.
    The widget's fill, border, and focus ring are painted manually in
    :meth:`paintEvent` using
    `QATCH.ui.components.flat_paint.paint_flat_surface`. Text, cursor,
    selection, and native line-edit actions remain under Qt's standard
    `QLineEdit` rendering pipeline.

    Visual states are resolved from the active `flat_*` theme tokens, so the
    widget automatically follows light/dark theme changes.

    Leading icons and trailing password-reveal buttons can be added using
    Qt's native `QLineEdit.addAction()` API.

    Visual States:
        The widget resolves its appearance according to the following
        priority:

        * Disabled: Uses muted text, a secondary surface, and the standard
          border.
        * Error: Uses the error border and error focus ring.
        * Focused: Uses the accent border and accent focus ring.
        * Hovered: Uses the stronger border while the cursor is over the
          widget.
        * Default: Uses the standard surface and border.

    Attributes:
        _hovered (bool): Whether the cursor is currently inside the widget.
        _in_error (bool): Whether the widget is currently displaying its
            visual error state.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the line edit with flat control styling.

        Configures the widget to use custom-painted chrome, enables hover
        tracking for state-aware border styling, applies the initial
        theme-aware text stylesheet, and subscribes to theme changes.

        Args:
            parent: Optional parent widget.

        Returns:
            None.
        """
        super().__init__(parent)
        self._hovered: bool = False
        self._in_error: bool = False

        self.setFrame(False)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self._apply_text_qss()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    @staticmethod
    def _rgba(rgba) -> str:
        """Convert an RGBA token tuple to a CSS `rgba()` string.

        Args:
            rgba: A four-element iterable containing red, green, blue, and
                alpha channel values.

        Returns:
            A CSS-compatible `rgba(r, g, b, a)` color string.
        """
        return f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]})"

    def _apply_text_qss(self) -> None:
        """Apply theme-aware text and selection styling.

        Updates the line edit's stylesheet using colors from the active
        theme. Text color is selected based on the widget's enabled state,
        while selection colors use the theme's accent tokens. The stylesheet
        also provides transparent backgrounds and borders so the manually
        painted control chrome remains visible.

        Native `QLineEdit` tool buttons are kept transparent, with a subtle
        surface highlight applied while hovered. The cursor shape is also
        updated to indicate whether the widget is currently editable.

        Returns:
            None.
        """
        tok = ThemeManager.instance().tokens()
        text_color = tok["flat_text_muted"] if not self.isEnabled() else tok["flat_text"]
        self.setStyleSheet(
            "QLineEdit {"
            "  background: transparent;"
            "  border: none;"
            "  padding: 9px 12px;"
            f"  color: {self._rgba(text_color)};"
            f"  font-family: {FONT_SANS_STACK};"
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
        """Refresh the widget styling after a theme change.

        Reapplies the theme-dependent text stylesheet and schedules a repaint
        so the manually painted fill, border, and focus ring also reflect the
        newly active theme.

        Args:
            _mode: Theme mode identifier supplied by the
                `ThemeManager.themeChanged` signal.

        Returns:
            None.
        """
        self._apply_text_qss()
        self.update()

    def setEnabled(self, enabled: bool) -> None:
        """Enable or disable the line edit and refresh its visual state.

        Updates the text stylesheet and schedules a repaint so the widget's
        text color, cursor, fill, and border immediately reflect the new
        enabled state.

        Args:
            enabled: Whether the line edit should be enabled for user input.

        Returns:
            None.
        """
        super().setEnabled(enabled)
        self._apply_text_qss()
        self.update()

    def set_error(self, on: bool) -> None:
        """Set or clear the widget's visual error state.

        When enabled, the widget paints using the theme's error border and
        error focus-ring tokens. This method only changes the visual state;
        it does not affect the line edit's enabled state or text contents.

        Args:
            on: Whether the error appearance should be enabled.

        Returns:
            None.
        """
        if on != self._in_error:
            self._in_error = on
            self.update()

    def enterEvent(self, event: QtCore.QEvent) -> None:
        """Handle the cursor entering the line edit.

        Delegates the event to the base `QLineEdit` implementation, marks
        the widget as hovered, and schedules a repaint so the hover-specific
        border styling is applied.

        Args:
            event: Qt event generated when the cursor enters the widget.

        Returns:
            None
        """
        super().enterEvent(event)
        self._hovered = True
        self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        """Handle the cursor leaving the line edit.

        Delegates the event to the base `QLineEdit` implementation, clears
        the internal hover state, and schedules a repaint to restore the
        normal border styling.

        Args:
            event: Qt event generated when the cursor leaves the widget.

        Returns:
            None.
        """
        super().leaveEvent(event)
        self._hovered = False
        self.update()

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        """Handle the line edit receiving keyboard focus.

        Delegates focus handling to the base implementation and schedules a
        repaint so the theme's focused border and focus ring are rendered.

        Args:
            event: Qt focus event generated when the widget receives focus.

        Returns:
            None.
        """
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        """Handle the line edit losing keyboard focus.

        Delegates focus handling to the base implementation and schedules a
        repaint so the focused border and focus ring are removed.

        Args:
            event: Qt focus event generated when the widget loses focus.

        Returns:
            None.
        """
        super().focusOutEvent(event)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the flat control chrome and delegate text rendering to Qt.

        Resolves the widget's fill, border, and focus-ring colors from the
        active theme based on the current interaction state, then paints the
        rounded flat surface using :func:`paint_flat_surface`. Text, cursor,
        selection, and native line-edit actions are subsequently rendered by
        the base :class:`QtWidgets.QLineEdit` implementation.

        Visual state precedence is:

            1. Disabled state.
            2. Error state.
            3. Focused state.
            4. Hovered state.
            5. Default state.

        Args:
            event: Qt paint event describing the region that needs to be
                repainted.

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

        super().paintEvent(event)
