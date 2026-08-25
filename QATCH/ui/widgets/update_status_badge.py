"""
QATCH.ui.widgets.update_status_banner.py

Themed update-status controls and notification badge widgets.

This module provides :class:`UpdateNotificationBadge` and
:class:`UpdateStatusIcon`, which together present software or firmware update
availability within the application UI.

The notification badge is a small floating, theme-aware pill positioned
relative to an update-status icon. It supports explicit dismissal, action
requests, and automatic repositioning when the application's top-level
window moves, resizes, or changes window state.

The status icon displays the current update state using a solid,
theme-dependent tint and exposes an action signal for states that require
user attention. It manages the lifecycle of its notification badge,
including suppression while overlays or other blocking activity are active,
restoration when that activity ends, and dismissal for the duration of an
update cycle.

Both widgets are designed to remain visually integrated with the
application's theme while avoiding system-wide always-on-top behavior.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-21
"""

from __future__ import annotations

from enum import IntEnum
from typing import ClassVar

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.overlay_shell import OverlayActivity
from QATCH.ui.components.window_utils import app_window_bounds_global, find_app_window
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css


class UpdateNotificationBadge(QtWidgets.QWidget):
    """A floating notification pill associated with an update-status icon.

    The badge displays a short update-availability message and provides a
    dismiss button. Clicking the badge body emits :attr:`action_requested`,
    while clicking the dismiss button hides the badge and emits
    :attr:`dismissed`.

    The badge is owned by the application's top-level window and uses a
    frameless tool-window configuration so it remains above the application's
    own UI without behaving as a system-wide always-on-top window. Its
    position follows the associated application window when that window is
    moved, resized, or changes window state.

    Args:
        anchor: Widget to which the badge is visually anchored.
        text: Notification message displayed inside the badge.
        prefer_below: Whether the badge should attempt to position itself
            below the anchor before falling back to the opposite side.

    Attributes:
        action_requested (pyqtSignal): Emitted when the badge body is clicked.
        dismissed (pyqtSignal): Emitted when the user dismisses the badge.
        _RADIUS (float): Corner radius used when painting the badge.
    """

    action_requested = QtCore.pyqtSignal()
    dismissed = QtCore.pyqtSignal()

    _RADIUS = 10.0

    def __init__(
        self,
        anchor: QtWidgets.QWidget,
        text: str = "Software update available",
        prefer_below: bool = False,
    ) -> None:
        """Initialize the notification badge.

        Configures the badge as a frameless, translucent tool window owned by the
        application's top-level window, creates its notification label and
        dismiss button, applies the current application theme, and installs an
        event filter so the badge follows its owning application window.

        Args:
            anchor: Widget to which the badge should remain visually attached.
            text: Notification message displayed by the badge.
            prefer_below: Whether to prefer positioning the badge below the
                anchor when both sides are available.
        """
        app_window = find_app_window() or anchor.window()
        super().__init__(app_window)
        self._anchor = anchor
        self._app_window = app_window
        self._prefer_below = prefer_below
        self._bg = QtGui.QColor(30, 38, 48, 235)
        self._border = QtGui.QColor(255, 255, 255, 45)

        self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint, True)
        self.setWindowFlag(QtCore.Qt.WindowType.Tool, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        if self._app_window is not None:
            self._app_window.installEventFilter(self)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 5, 5)
        layout.setSpacing(6)

        self._label = QtWidgets.QLabel(text)
        layout.addWidget(self._label)

        self._dismiss_btn = QtWidgets.QToolButton()
        self._dismiss_btn.setText("✕")
        self._dismiss_btn.setFixedSize(16, 16)
        self._dismiss_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._dismiss_btn.clicked.connect(self._on_dismiss)
        layout.addWidget(self._dismiss_btn)

        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(lambda _: self._apply_theme())

        self.adjustSize()

    def _apply_theme(self) -> None:
        """Apply the current application theme to the badge.

        Updates the badge background and border colors from the active theme and
        refreshes the label and dismiss-button text styling.
        """
        tok = ThemeManager.instance().tokens()
        self._bg = QtGui.QColor(*tok["plot_menu_bg"])
        self._border = QtGui.QColor(*tok["plot_menu_border"])
        text_css = tok_css(tok["plot_text_normal"])
        muted_css = tok_css(tok["plot_text_muted"])
        self._label.setStyleSheet(
            f"QLabel {{ color: {text_css}; font-size: 11px; font-weight: 600;"
            " background: transparent; }"
        )
        self._dismiss_btn.setStyleSheet(
            f"QToolButton {{ color: {muted_css}; font-size: 10px;"
            " background: transparent; border: none; }"
            f"QToolButton:hover {{ color: {text_css}; }}"
        )
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the rounded notification badge.

        Draws the badge background and one-pixel border using the colors derived
        from the active application theme.

        Args:
            event: Paint event describing the region that needs to be redrawn.
        """
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        rf = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setBrush(QtGui.QBrush(self._bg))
        p.setPen(QtGui.QPen(self._border, 1.0))
        p.drawRoundedRect(rf, self._RADIUS, self._RADIUS)
        p.end()

    def reposition(self) -> None:
        """Reposition the badge relative to its anchor widget.

        Attempts to place the badge above or below the anchor according to
        `prefer_below`. If the preferred side does not fit within the
        application's window bounds, the opposite side is used when possible.
        The resulting position is clamped to the application's bounds, with
        screen geometry used as a fallback when the application window cannot
        be resolved.
        """
        global_pos = self._anchor.mapToGlobal(QtCore.QPoint(0, 0))
        anchor_right = global_pos.x() + self._anchor.width()
        x = anchor_right - self.width()
        y_above = global_pos.y() - self.height() - 4
        y_below = global_pos.y() + self._anchor.height() + 4
        app_window = find_app_window()
        if app_window is not None:
            bounds = app_window_bounds_global(app_window)
        else:
            screen = QtWidgets.QApplication.screenAt(global_pos)
            bounds = screen.geometry() if screen else None

        if bounds is not None:
            fits_above = y_above >= bounds.top()
            fits_below = y_below + self.height() <= bounds.bottom()
            if self._prefer_below:
                y = y_below if (fits_below or not fits_above) else y_above
            else:
                y = y_above if (fits_above or not fits_below) else y_below
            x = max(bounds.left(), min(x, bounds.right() - self.width()))
            y = max(bounds.top(), min(y, bounds.bottom() - self.height()))
        else:
            y = y_below if self._prefer_below else y_above

        self.move(x, y)

    def show_near_anchor(self) -> None:
        """Resize, position, and display the badge beside its anchor.

        Recalculates the badge's preferred position using its current size,
        shows the badge without activating it, and raises it above the
        application's other owned widgets.
        """
        self.adjustSize()
        self.reposition()
        self.show()
        self.raise_()

    def eventFilter(
        self,
        watched: QtCore.QObject,
        event: QtCore.QEvent,
    ) -> bool:
        """Reposition the badge when its application window changes geometry.

        The badge follows application-window move, resize, and window-state
        changes while it is visible.

        Args:
            watched: Object that generated the event.
            event: Qt event being filtered.

        Returns:
            The result of the base class event-filter implementation.
        """
        if (
            watched is self._app_window
            and self.isVisible()
            and event.type()
            in (
                QtCore.QEvent.Type.Move,
                QtCore.QEvent.Type.Resize,
                QtCore.QEvent.Type.WindowStateChange,
            )
        ):
            self.reposition()
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle clicks on the notification badge body.

        A left-click on the badge emits :attr:`action_requested`. The dismiss
        button handles its own click events separately.

        Args:
            event: Mouse event describing the button and position of the click.
        """
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.action_requested.emit()
        super().mousePressEvent(event)

    def _on_dismiss(self) -> None:
        """Hide the badge and notify listeners that it was dismissed."""
        self.hide()
        self.dismissed.emit()


class UpdateStatusIcon(QtWidgets.QToolButton):
    """Themed update-status button with an optional notification badge.

    The icon communicates the current update state through a solid,
    theme-dependent color without pulsing or flashing. Actionable states
    emit :attr:`update_requested` when the icon is clicked.

    When the state becomes :attr:`State.OPTIONAL` or
    :attr:`State.MANDATORY`, the icon can display an
    :class:`UpdateNotificationBadge`. The badge is automatically hidden when
    the state is no longer actionable, suppressed while overlays or other
    blocking activity are active, and restored when appropriate.

    Args:
        icon_path: Filesystem path to the SVG icon displayed by the button.
        size: Square width and height of the button in pixels.
        parent: Optional parent widget.
        badge_text: Text displayed in the floating notification badge.
        prefer_below: Whether the badge should prefer positioning below the
            icon rather than above it.

    Attributes:
        update_requested (pyqtSignal): Emitted when the user requests an
            update action from an actionable state.
        State (IntEnum): Enumeration of supported update-status states.
    """

    update_requested = QtCore.pyqtSignal()

    class State(IntEnum):
        """Enumeration of update-status states.

        Attributes:
            UNKNOWN: Update status has not yet been determined.
            CHECKING: An update check is currently in progress.
            UP_TO_DATE: No update is currently required or available.
            OPTIONAL: An update is available but is not mandatory.
            MANDATORY: An update is required and must be installed.
        """

        UNKNOWN = 0
        CHECKING = 1
        UP_TO_DATE = 2
        OPTIONAL = 3
        MANDATORY = 4

    _DISABLED_ALPHA_SCALE = 0.4
    _COLORS: ClassVar[dict[str, dict[UpdateStatusIcon.State, QtGui.QColor]]] = {
        "light": {
            State.UNKNOWN: QtGui.QColor(150, 165, 180),
            State.CHECKING: QtGui.QColor(150, 165, 180),
            State.UP_TO_DATE: QtGui.QColor(60, 190, 120),
            State.OPTIONAL: QtGui.QColor(240, 170, 50),
            State.MANDATORY: QtGui.QColor(228, 70, 70),
        },
        "dark": {
            State.UNKNOWN: QtGui.QColor(170, 180, 195),
            State.CHECKING: QtGui.QColor(170, 180, 195),
            State.UP_TO_DATE: QtGui.QColor(80, 210, 140),
            State.OPTIONAL: QtGui.QColor(255, 185, 60),
            State.MANDATORY: QtGui.QColor(240, 90, 90),
        },
    }

    _TOOLTIPS: ClassVar[dict[UpdateStatusIcon.State, str]] = {
        State.UNKNOWN: "Update status unknown - click to check",
        State.CHECKING: "Checking for updates...",
        State.UP_TO_DATE: "Up to date",
        State.OPTIONAL: "Update available - click to update",
        State.MANDATORY: "Required update - click to install",
    }

    def __init__(
        self,
        icon_path: str,
        size: int = 20,
        parent: QtWidgets.QWidget | None = None,
        badge_text: str = "Software update available",
        prefer_below: bool = False,
    ) -> None:
        """Initialize the update-status icon.

        Loads the base icon, initializes the status and notification state,
        configures the button geometry and interaction behavior, connects theme
        and overlay activity notifications, and prepares deferred icon rendering.

        Args:
            icon_path: Filesystem path to the SVG icon to display.
            size: Square width and height of the button in pixels.
            parent: Optional parent widget.
            badge_text: Text displayed by the floating notification badge.
            prefer_below: Whether the badge should prefer positioning below the
                icon when displayed.
        """
        super().__init__(parent)
        self._icon_path = icon_path
        self._size = size
        self._state = self.State.UNKNOWN
        self._detail = ""
        self._badge_text = badge_text
        self._prefer_below = prefer_below
        self._badge_dismissed = False
        self._badge: UpdateNotificationBadge | None = None
        self._icon_dirty = True

        self.setFixedSize(size, size)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setIconSize(QtCore.QSize(size, size))
        self.setStyleSheet("QToolButton { background: transparent; border: none; }")
        self.setAutoRaise(True)

        self._base_pixmap = QtGui.QPixmap(icon_path)

        ThemeManager.instance().themeChanged.connect(lambda _: self._mark_icon_dirty())
        self.clicked.connect(self._on_clicked)
        OverlayActivity.instance().any_open_changed.connect(self._on_overlay_activity_changed)
        self._update_tooltip()

    def _on_overlay_activity_changed(self, any_open: bool) -> None:
        """Suppress or restore the notification badge around overlay activity.

        Hides the badge while any application overlay is open and resynchronizes
        its visibility when all overlays have closed. This composes with other
        badge-suppression mechanisms without changing the underlying update
        state.

        Args:
            any_open: Whether at least one application overlay is currently open.
        """
        if any_open:
            self.dismiss_badge()
        else:
            self.resync_badge()

    def state(self) -> UpdateStatusIcon.State:
        """Return the current update-status state.

        Returns:
            The currently active :class:`State` value.
        """
        return self._state

    def setState(
        self,
        state: UpdateStatusIcon.State,
        detail: str = "",
    ) -> None:
        """Set the update state and associated tooltip detail.

        Updates the icon's state, tooltip, deferred rendering state, and
        notification-badge visibility. Entering an actionable state displays the
        badge unless it has already been dismissed for the current update cycle.
        Leaving the actionable states resets the dismissal flag.

        Args:
            state: New update-status state.
            detail: Optional additional text appended to the status tooltip,
                such as a version number or release description.
        """
        self._state = state
        self._detail = detail
        self._update_tooltip()
        self._mark_icon_dirty()
        self._sync_badge()

    def _sync_badge(self) -> None:
        """Synchronize notification-badge visibility with the current state.

        Actionable states display the badge when it has not been explicitly
        dismissed. Non-actionable states hide the badge and reset the dismissal
        state for the next update cycle.
        """
        if self._state in (self.State.OPTIONAL, self.State.MANDATORY):
            if not self._badge_dismissed:
                self._show_badge()
        else:
            # Reset dismissed flag when state leaves actionable range
            self._badge_dismissed = False
            if self._badge and self._badge.isVisible():
                self._badge.hide()

    def _show_badge(self) -> None:
        """Create and display the notification badge when permitted.

        Lazily creates the badge, connects its action and dismissal signals, and
        displays it near the icon when the icon is visible and enabled and no
        application overlay is currently open.
        """
        if self._badge is None:
            self._badge = UpdateNotificationBadge(
                self, text=self._badge_text, prefer_below=self._prefer_below
            )
            self._badge.action_requested.connect(self._on_clicked)
            self._badge.dismissed.connect(self._on_badge_dismissed)
        if not self.isVisible() or not self.isEnabled():
            return
        if OverlayActivity.instance().is_any_open():
            return
        self._badge.show_near_anchor()

    def _on_badge_dismissed(self) -> None:
        """Record that the user explicitly dismissed the current badge."""
        self._badge_dismissed = True

    def dismiss_badge(self) -> None:
        """Temporarily hide the notification badge.

        This method suppresses the visible badge without marking it as
        permanently dismissed. It is intended for temporary blocking activity,
        such as an active run, after which :meth:`resync_badge` can restore the
        badge if the update remains relevant.
        """
        if self._badge and self._badge.isVisible():
            self._badge.hide()

    def resync_badge(self) -> None:
        """Restore badge visibility according to the current update state.

        Re-evaluates the current state and suppression conditions so that a
        temporarily hidden badge reappears when appropriate. A badge explicitly
        dismissed by the user remains suppressed for the current update cycle.
        """
        self._sync_badge()

    def _mark_icon_dirty(self) -> None:
        """Mark the displayed icon for deferred rebuilding.

        Sets the icon-dirty flag and schedules a repaint. Actual pixmap and icon
        construction is deferred until :meth:`paintEvent`, allowing multiple
        state or theme changes occurring before a repaint to be coalesced into a
        single icon rebuild.
        """
        self._icon_dirty = True
        self.update()

    def _update_tooltip(self) -> None:
        """Update the button tooltip for the current state and detail text."""
        tip = self._TOOLTIPS.get(self._state, "")
        if self._detail:
            tip = f"{tip}\n{self._detail}"
        self.setToolTip(tip)

    def _tinted_pixmap(
        self,
        color: QtGui.QColor,
        alpha_f: float = 1.0,
    ) -> QtGui.QPixmap:
        """Create a scaled pixmap tinted with the specified color.

        The source icon is scaled to the configured button size and recolored
        using a `SourceAtop` composition pass. When `alpha_f` is less than
        one, the completed tinted pixmap is subsequently dimmed so that the
        source icon's original color cannot bleed through the tint.

        Args:
            color: Color used to tint the source icon.
            alpha_f: Overall opacity multiplier applied after tinting.

        Returns:
            A scaled and tinted pixmap, or a null pixmap if the source icon could
            not be loaded.
        """
        if self._base_pixmap.isNull():
            return QtGui.QPixmap()
        base = self._base_pixmap.scaled(
            self._size,
            self._size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        dst = QtGui.QPixmap(base.size())
        dst.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(dst)
        p.drawPixmap(0, 0, base)
        p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
        p.fillRect(dst.rect(), color)
        p.end()

        if alpha_f >= 1.0:
            return dst
        dimmed = QtGui.QPixmap(dst.size())
        dimmed.fill(QtCore.Qt.GlobalColor.transparent)
        dp = QtGui.QPainter(dimmed)
        dp.setOpacity(alpha_f)
        dp.drawPixmap(0, 0, dst)
        dp.end()
        return dimmed

    def _tinted_icon(self, color: QtGui.QColor, alpha_f: float = 1.0) -> QtGui.QIcon:
        """Create a QIcon containing a tinted version of the base icon.

        Args:
            color: Color used to tint the source icon.
            alpha_f: Overall opacity multiplier applied to the resulting icon.

        Returns:
            A tinted :class:`QtGui.QIcon`, or an empty icon if the source pixmap
            is unavailable.
        """
        pix = self._tinted_pixmap(color, alpha_f)
        return QtGui.QIcon(pix) if not pix.isNull() else QtGui.QIcon()

    def _refresh_icon(self) -> None:
        """Rebuild the status icon using the current state and theme.

        Applies the current state's theme-specific color to the base icon and
        creates explicit Normal and Disabled icon variants. The UNKNOWN state is
        rendered at reduced opacity, while disabled variants preserve the
        semantic status color rather than using Qt's default gray/desaturated
        disabled rendering.
        """
        mode = ThemeManager.instance().mode().value
        colors = self._COLORS.get(mode, self._COLORS["light"])
        color = colors.get(self._state, colors[self.State.UNKNOWN])
        alpha = 0.5 if self._state == self.State.UNKNOWN else 1.0

        icon = QtGui.QIcon()
        normal_pix = self._tinted_pixmap(color, alpha)
        if not normal_pix.isNull():
            icon.addPixmap(normal_pix, QtGui.QIcon.Mode.Normal)
            disabled_pix = self._tinted_pixmap(color, alpha * self._DISABLED_ALPHA_SCALE)
            icon.addPixmap(disabled_pix, QtGui.QIcon.Mode.Disabled)
        self.setIcon(icon)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Refresh the icon when necessary and paint the tool button.

        Rebuilds the tinted icon when a state or theme change has marked it dirty,
        then delegates the actual button painting to :class:`QToolButton`.

        Args:
            event: Paint event describing the region that needs to be redrawn.
        """
        if self._icon_dirty:
            self._refresh_icon()
            self._icon_dirty = False
        super().paintEvent(event)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Handle the icon becoming visible.

        Delegates normal show-event processing to Qt and schedules restoration of
        an undisclosed actionable-state badge when one already exists and has not
        been explicitly dismissed.

        Args:
            event: Show event generated by the Qt framework.
        """
        super().showEvent(event)
        if (
            self._state in (self.State.OPTIONAL, self.State.MANDATORY)
            and not self._badge_dismissed
            and self._badge
        ):
            QtCore.QTimer.singleShot(50, self._show_badge)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        """Handle the icon becoming hidden.

        Delegates normal hide-event processing to Qt and hides any currently
        visible notification badge so it cannot remain detached from its hidden
        anchor.

        Args:
            event: Hide event generated by the Qt framework.
        """
        super().hideEvent(event)
        if self._badge and self._badge.isVisible():
            self._badge.hide()

    def _on_clicked(self) -> None:
        """Handle activation of the update-status icon or notification badge.

        Emits :attr:`update_requested` when the current state represents an
        actionable update condition. Any visible notification badge is hidden
        before the request is emitted.
        """
        if self._state not in (self.State.UP_TO_DATE, self.State.CHECKING):
            if self._badge and self._badge.isVisible():
                self._badge.hide()
            self.update_requested.emit()
