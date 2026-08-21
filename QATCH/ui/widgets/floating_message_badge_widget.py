"""
QATCH.ui.widgets.floating_message_badge_widget.py

Non-blocking overlay notification system for nanovisQ.

This module provides the FloatingMessageBadge class, a frameless, translucent
overlay designed to display status updates, errors, and system notifications
without interrupting the user's workflow.

Key features include:
    - Anchored positioning: The badge automatically tracks and snaps to a
      parent widget or window, maintaining its relative position during
      resizing or multi-monitor movement.
    - Event Filtering: Monitors parent events to handle automatic dismissal
      and repositioning synchronously, eliminating visual lag.
    - Aesthetic Consistency: Implements a frosted-glass visual style with
      support for distinct 'info' and 'error' color palettes.
    - Animated Lifecycle: Utilizes QPropertyAnimation for smooth fade-in and
      fade-out transitions, paired with a display timer for auto-dismissal.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-05-05
"""

from PyQt5 import QtCore, QtGui, QtWidgets


class FloatingMessageBadgeWidget(QtWidgets.QWidget):
    """Display a frameless, glass-style floating message badge.

    Provides a transient notification widget for informational and error
    messages. The badge supports themed glass styling, optional custom
    close icons, automatic dismissal, fade and slide animations, and
    positioning relative to an anchor widget.

    The badge tracks both its anchor widget and the anchor's top-level
    window so that it remains visually aligned as the application moves
    or resizes. It automatically closes when the anchor is hidden, closed,
    or destroyed.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        close_icon_path: str | None = None,
    ) -> None:
        """Initialize the floating message badge.

        Configures the badge as a frameless translucent tool window, creates
        the glass-style message panel and close button, initializes the fade,
        slide, and dismissal timer animations, and applies the default visual
        styling.

        Args:
            parent (QtWidgets.QWidget): Parent widget used to establish the
                badge's ownership within the application.
            close_icon_path (str, optional): Filesystem path to a custom close
                button icon. If omitted or invalid, the close button falls back
                to an `x` character.
        """
        super().__init__(parent)

        self._display_duration_ms = 15_000
        self._fade_duration_ms = 260
        self._hide_when_animation_finishes = False
        self._close_icon_path = close_icon_path
        self._anchor_widget: QtWidgets.QWidget | None = None
        self._tracked_anchor_widgets: list[QtWidgets.QWidget] = []
        self._position_gap_px = 15
        self.setWindowFlags(QtCore.Qt.Tool | QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.setWindowOpacity(0.0)

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(20, 16, 20, 22)  # room for the softer glass shadow
        root_layout.setSpacing(0)

        self.panel = QtWidgets.QFrame(self)
        self.panel.setObjectName("floatingMessagePanel")
        self.panel.setProperty("messageType", "info")
        root_layout.addWidget(self.panel)

        panel_layout = QtWidgets.QHBoxLayout(self.panel)
        panel_layout.setContentsMargins(17, 11, 10, 11)
        panel_layout.setSpacing(10)

        self.label = QtWidgets.QLabel("")
        self.label.setObjectName("floatingMessageText")
        self.label.setProperty("messageType", "info")
        self.label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self.label.setWordWrap(True)
        self.label.setMaximumWidth(420)
        panel_layout.addWidget(self.label, 1, QtCore.Qt.AlignmentFlag.AlignVCenter)

        self.close_button = QtWidgets.QPushButton("x")
        self.close_button.setObjectName("floatingMessageClose")
        self.close_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.close_button.setToolTip("Close")
        self.close_button.setAccessibleName("Close message")
        self.close_button.setFixedSize(22, 22)
        self.close_button.clicked.connect(self.clear)
        panel_layout.addWidget(
            self.close_button, 0, QtCore.Qt.AlignTop | QtCore.Qt.AlignmentFlag.AlignRight
        )

        shadow = QtWidgets.QGraphicsDropShadowEffect(self.panel)
        shadow.setBlurRadius(34)
        shadow.setOffset(0, 10)
        shadow.setColor(QtGui.QColor(35, 55, 70, 42))
        self.panel.setGraphicsEffect(shadow)

        self._fade_animation = QtCore.QPropertyAnimation(self, b"windowOpacity", self)
        self._fade_animation.setDuration(self._fade_duration_ms)
        self._fade_animation.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._fade_animation.finished.connect(self._on_fade_animation_finished)
        self._slide_animation = QtCore.QPropertyAnimation(self, b"pos", self)

        self._dismiss_timer = QtCore.QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self.fade_out)

        self._apply_styles()
        self.set_close_icon_path(close_icon_path)
        self.hide()

    def set_close_icon_path(self, icon_path: str | None) -> None:
        """Set or clear the custom close icon for the badge.

        Loads the supplied icon path when available and uses it for the close
        button. If no path is supplied or the icon cannot be loaded, restores
        the default `x` text fallback.

        Args:
            icon_path (str, optional): Filesystem path to the close icon, or
                `None` to use the default text-based close control.
        """
        self._close_icon_path = icon_path

        if icon_path:
            icon = QtGui.QIcon(icon_path)
            if not icon.isNull():
                self.close_button.setIcon(icon)
                self.close_button.setIconSize(QtCore.QSize(11, 11))
                self.close_button.setText("")
                return

        # Fallback when no custom icon is supplied or the path cannot be loaded.
        self.close_button.setIcon(QtGui.QIcon())
        self.close_button.setText("x")

    def show_message(
        self,
        text: str,
        is_error: bool = False,
        parent_widget: QtWidgets.QWidget | None = None,
        drop_in: bool = False,
        drop_in_duration: int = 500,
    ) -> None:
        """Display a message badge with optional drop-in animation.

        Updates the message text and visual state, anchors and positions the
        badge, then fades it into view. The badge automatically dismisses after
        the configured display duration.

        Args:
            text (str): Message text to display.
            is_error (bool, optional): Whether the message should use the error
                visual style instead of the informational style. Defaults to
                `False`.
            parent_widget (QtWidgets.QWidget, optional): Widget to use as the
                badge's positioning anchor. If omitted, the badge's parent
                widget is used. Defaults to `None`.
            drop_in (bool, optional): Whether to animate the badge downward
                from 50 pixels above its final position. Defaults to `False`.
            drop_in_duration (int, optional): Duration of the drop-in animation
                in milliseconds. Defaults to `500`.
        """
        message_type = "error" if is_error else "info"
        self.panel.setProperty("messageType", message_type)
        self.label.setProperty("messageType", message_type)
        self.label.setText(text)
        self._refresh_polish(self.panel)
        self._refresh_polish(self.label)

        self._set_anchor_widget(parent_widget or self.parentWidget())
        self.adjustSize()
        self._reposition_to_anchor()

        self._dismiss_timer.stop()
        self._fade_animation.stop()
        self._slide_animation.stop()
        self._hide_when_animation_finishes = False

        self.setWindowOpacity(0.0)

        if drop_in:
            final_pos = self.pos()
            self.move(final_pos.x(), final_pos.y() - 50)
            self._slide_animation.setDuration(drop_in_duration)
            self._slide_animation.setEasingCurve(QtCore.QEasingCurve.OutBack)
            self._slide_animation.setStartValue(self.pos())
            self._slide_animation.setEndValue(final_pos)
            self._slide_animation.start()

        self.show()
        self.raise_()

        self._fade_animation.setDuration(self._fade_duration_ms)
        self._fade_animation.setStartValue(0.0)
        self._fade_animation.setEndValue(1.0)
        self._fade_animation.start()
        self._dismiss_timer.start(self._display_duration_ms)

    def fade_out(self) -> None:
        """Fade the visible badge out and hide it when the animation completes.

        Stops the dismissal timer and any existing fade animation before
        starting a new fade from the badge's current opacity to fully
        transparent.
        """
        if not self.isVisible():
            return

        self._dismiss_timer.stop()
        self._fade_animation.stop()
        self._fade_animation.setDuration(self._fade_duration_ms)
        self._hide_when_animation_finishes = True
        self._fade_animation.setStartValue(self.windowOpacity())
        self._fade_animation.setEndValue(0.0)
        self._fade_animation.start()

    def slide_out(self, duration: int = 250) -> None:
        """Slide the badge upward while fading it out.

        Moves the badge 50 pixels upward while simultaneously reducing its
        opacity. The badge is hidden after the animations complete. This is
        intended for coordinated dismissals where a visible toast should
        leave with motion rather than simply fading in place.

        Args:
            duration (int, optional): Duration of the slide and fade animations
                in milliseconds. Defaults to `250`.
        """
        if not self.isVisible():
            return

        self._dismiss_timer.stop()
        self._fade_animation.stop()
        self._slide_animation.stop()
        self._hide_when_animation_finishes = True

        self._fade_animation.setDuration(duration)
        self._fade_animation.setStartValue(self.windowOpacity())
        self._fade_animation.setEndValue(0.0)
        self._fade_animation.start()

        start_pos = self.pos()
        self._slide_animation.setDuration(duration)
        self._slide_animation.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._slide_animation.setStartValue(start_pos)
        self._slide_animation.setEndValue(QtCore.QPoint(start_pos.x(), start_pos.y() - 50))
        self._slide_animation.start()

    def clear(self) -> None:
        """Immediately hide and reset the floating message badge.

        Stops any active dismissal or fade animation, clears the pending
        hide-on-animation state, resets the window opacity to zero, and hides
        the badge without waiting for an animation to finish.
        """
        self._dismiss_timer.stop()
        self._fade_animation.stop()
        self._hide_when_animation_finishes = False
        self.setWindowOpacity(0.0)
        self.hide()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Keep the badge synchronized with its tracked anchor widgets.

        Closes the badge when a tracked anchor is hidden or closed. While the
        badge is visible, repositions it synchronously when a tracked anchor
        or its top-level window moves, resizes, is shown, or changes window
        state.

        Args:
            watched (QtCore.QObject): Object that generated the event.
            event (QtCore.QEvent): Event being processed.

        Returns:
            bool: The result of the base class event-filter implementation.
        """
        if watched in self._tracked_anchor_widgets:
            event_type = event.type()

            if event_type in (QtCore.QEvent.Hide, QtCore.QEvent.Close):
                self.clear()
            elif self.isVisible() and event_type in (
                QtCore.QEvent.Type.Move,
                QtCore.QEvent.Type.Resize,
                QtCore.QEvent.Type.Show,
                QtCore.QEvent.Type.WindowStateChange,
            ):
                # Reposition synchronously to eliminate drag lag
                self._reposition_to_anchor()

        return super().eventFilter(watched, event)

    def _on_fade_animation_finished(self) -> None:
        """Hide the badge after a fade-out animation completes.

        Hides the badge and restores zero window opacity when the current fade
        operation was initiated as part of a dismissal.
        """
        if self._hide_when_animation_finishes:
            self._hide_when_animation_finishes = False
            self.hide()
            self.setWindowOpacity(0.0)

    def _set_anchor_widget(self, target: QtWidgets.QWidget | None) -> None:
        """Set the widget used to anchor and track badge positioning.

        Replaces the current anchor, removes obsolete event filters, and
        installs event filters on the new anchor and its top-level window.
        This allows the badge to respond to geometry and visibility changes
        affecting either object.

        Args:
            target (QtWidgets.QWidget, optional): Widget to use as the badge's
                anchor, or `None` to remove the current anchor.
        """
        if target is self._anchor_widget and self._tracked_anchor_widgets:
            return

        self._remove_anchor_event_filters()
        self._anchor_widget = target

        if target is None:
            return

        self._install_anchor_event_filter(target)

        window = target.window()
        if window is not None and window is not target:
            self._install_anchor_event_filter(window)

    def _install_anchor_event_filter(self, widget: QtWidgets.QWidget) -> None:
        """Install event tracking on an anchor-related widget.

        Registers this badge as an event filter and connects the widget's
        destruction signal so the badge can cleanly dismiss itself if the
        tracked widget is deleted.

        Args:
            widget (QtWidgets.QWidget): Widget whose lifecycle and geometry
                should be tracked.
        """
        if widget in self._tracked_anchor_widgets:
            return

        widget.installEventFilter(self)
        widget.destroyed.connect(self._on_anchor_destroyed)
        self._tracked_anchor_widgets.append(widget)

    def _remove_anchor_event_filters(self) -> None:
        """Remove event filters from all currently tracked anchor widgets.

        Safely unregisters the badge from each tracked widget, including
        widgets that may already have been deleted by Qt.
        """
        for widget in list(self._tracked_anchor_widgets):
            try:
                widget.removeEventFilter(self)
            except RuntimeError:
                # The Qt object may already be deleted.
                pass

        self._tracked_anchor_widgets.clear()

    def _on_anchor_destroyed(self, *_args: object) -> None:
        """Handle destruction of the badge's anchor widget.

        Clears the stored anchor and tracked-widget state and immediately
        dismisses the badge so it cannot remain visible after its positioning
        target has been destroyed.

        Args:
            *_args (object): Arguments emitted with the Qt `destroyed`
                signal.
        """
        self._anchor_widget = None
        self._tracked_anchor_widgets.clear()
        self.clear()

    def _reposition_to_anchor(self) -> None:
        """Recalculate and apply the badge position relative to its anchor.

        Adjusts the badge size to its current contents before positioning it
        above the tracked anchor widget.
        """
        if self._anchor_widget is None:
            return

        self.adjustSize()
        self._position_above(self._anchor_widget)

    def _position_above(self, target: QtWidgets.QWidget | None) -> None:
        """Position the badge above an anchor widget.

        Centers the badge horizontally over the target and places it above the
        target with the configured vertical gap. The resulting position is
        clamped to the available geometry of the screen containing the target.

        Args:
            target (QtWidgets.QWidget, optional): Widget above which the badge
                should be positioned.
        """
        if target is None:
            return

        global_pos = target.mapToGlobal(QtCore.QPoint(0, 0))
        x = global_pos.x() + (target.width() - self.width()) // 2
        y = global_pos.y() - self.height() - self._position_gap_px

        screen = QtWidgets.QApplication.screenAt(QtCore.QPoint(x, y))
        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()

        if screen is not None:
            bounds = screen.availableGeometry()
            x = max(bounds.left() + 8, min(x, bounds.right() - self.width() - 8))
            y = max(bounds.top() + 8, min(y, bounds.bottom() - self.height() - 8))

        self.move(x, y)

    @staticmethod
    def _refresh_polish(widget: QtWidgets.QWidget) -> None:
        """Refresh a widget's Qt style after a dynamic property change.

        Unpolishes and reapplies the widget's current style, then requests a
        repaint so stylesheet rules dependent on dynamic properties take
        effect immediately.

        Args:
            widget (QtWidgets.QWidget): Widget whose styling should be
                refreshed.
        """
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _apply_styles(self) -> None:
        """Apply the default glass and message-state styles to the badge.

        Configures the translucent panel, informational and error backgrounds,
        message text, and close button appearance, including hover and pressed
        states.
        """
        self.setStyleSheet("""
            QFrame#floatingMessagePanel {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(255, 255, 255, 178),
                    stop: 0.38 rgba(255, 255, 255, 125),
                    stop: 0.72 rgba(236, 246, 252, 92),
                    stop: 1 rgba(218, 234, 244, 72)
                );
                border: 1px solid rgba(255, 255, 255, 105);
                border-radius: 17px;
            }

            QFrame#floatingMessagePanel[messageType="error"] {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(255, 250, 250, 185),
                    stop: 0.42 rgba(255, 238, 238, 122),
                    stop: 0.76 rgba(255, 220, 220, 88),
                    stop: 1 rgba(245, 205, 205, 68)
                );
                border: 1px solid rgba(255, 255, 255, 92);
            }

            QLabel#floatingMessageText {
                background: transparent;
                border: none;
                color: rgba(63, 77, 89, 228);
                font-size: 8.5pt;
                font-weight: 600;
                padding: 0px;
            }

            QLabel#floatingMessageText[messageType="error"] {
                color: rgba(180, 45, 45, 238);
            }

            QPushButton#floatingMessageClose {
                background: transparent;
                border: none;
                border-radius: 11px;
                color: rgba(58, 72, 84, 165);
                font-size: 11pt;
                font-weight: 700;
                padding: 0px;
            }

            QPushButton#floatingMessageClose:hover {
                background-color: rgba(255, 255, 255, 82);
                border: none;
                color: rgba(36, 48, 58, 220);
            }

            QPushButton#floatingMessageClose:pressed {
                background-color: rgba(210, 226, 238, 95);
                border: none;
            }
            """)
