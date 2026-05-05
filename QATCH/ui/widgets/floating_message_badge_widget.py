"""
floating_message_badge_widget.py

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

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets


class FloatingMessageBadgeWidget(QtWidgets.QWidget):
    """A frameless, glass-style floating badge for alerts and info."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        close_icon_path: Optional[str] = None,
    ) -> None:
        super().__init__(parent)

        self._display_duration_ms = 15_000
        self._fade_duration_ms = 260
        self._hide_when_animation_finishes = False
        self._close_icon_path = close_icon_path
        self._anchor_widget: Optional[QtWidgets.QWidget] = None
        self._tracked_anchor_widgets: list[QtWidgets.QWidget] = []
        self._position_gap_px = 15

        # Frameless tool window that stays above the app without stealing focus.
        self.setWindowFlags(
            QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
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
        self.label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.label.setWordWrap(True)
        self.label.setMaximumWidth(420)
        panel_layout.addWidget(self.label, 1, QtCore.Qt.AlignVCenter)

        self.close_button = QtWidgets.QPushButton("×")
        self.close_button.setObjectName("floatingMessageClose")
        self.close_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.close_button.setToolTip("Close")
        self.close_button.setAccessibleName("Close message")
        self.close_button.setFixedSize(22, 22)
        self.close_button.clicked.connect(self.clear)
        panel_layout.addWidget(self.close_button, 0, QtCore.Qt.AlignTop | QtCore.Qt.AlignRight)

        shadow = QtWidgets.QGraphicsDropShadowEffect(self.panel)
        shadow.setBlurRadius(34)
        shadow.setOffset(0, 10)
        shadow.setColor(QtGui.QColor(35, 55, 70, 42))
        self.panel.setGraphicsEffect(shadow)

        self._fade_animation = QtCore.QPropertyAnimation(self, b"windowOpacity", self)
        self._fade_animation.setDuration(self._fade_duration_ms)
        self._fade_animation.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._fade_animation.finished.connect(self._on_fade_animation_finished)

        self._dismiss_timer = QtCore.QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self.fade_out)

        self._apply_styles()
        self.set_close_icon_path(close_icon_path)
        self.hide()

    def set_close_icon_path(self, icon_path: Optional[str]) -> None:
        """Set or clear the custom icon used by the close button."""
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
        self.close_button.setText("×")

    def show_message(
        self,
        text: str,
        is_error: bool = False,
        parent_widget: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Update the message, position the badge, fade it in, then auto-dismiss it."""
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
        self._hide_when_animation_finishes = False

        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()

        self._fade_animation.setStartValue(0.0)
        self._fade_animation.setEndValue(1.0)
        self._fade_animation.start()
        self._dismiss_timer.start(self._display_duration_ms)

    def fade_out(self) -> None:
        """Fade the badge out, then hide it."""
        if not self.isVisible():
            return

        self._dismiss_timer.stop()
        self._fade_animation.stop()
        self._hide_when_animation_finishes = True
        self._fade_animation.setStartValue(self.windowOpacity())
        self._fade_animation.setEndValue(0.0)
        self._fade_animation.start()

    def clear(self) -> None:
        """Immediately close the badge without waiting for the fade-out animation."""
        self._dismiss_timer.stop()
        self._fade_animation.stop()
        self._hide_when_animation_finishes = False
        self.setWindowOpacity(0.0)
        self.hide()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Keep the floating badge visually locked to its anchor widget/window."""
        if watched in self._tracked_anchor_widgets:
            event_type = event.type()

            if event_type in (QtCore.QEvent.Hide, QtCore.QEvent.Close):
                self.clear()
            elif self.isVisible() and event_type in (
                QtCore.QEvent.Move,
                QtCore.QEvent.Resize,
                QtCore.QEvent.Show,
                QtCore.QEvent.WindowStateChange,
            ):
                # Reposition synchronously to eliminate drag lag
                self._reposition_to_anchor()

        return super().eventFilter(watched, event)

    def _on_fade_animation_finished(self) -> None:
        if self._hide_when_animation_finishes:
            self._hide_when_animation_finishes = False
            self.hide()
            self.setWindowOpacity(0.0)

    def _set_anchor_widget(self, target: Optional[QtWidgets.QWidget]) -> None:
        """Track the widget used for badge placement and its top-level window."""
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
        if widget in self._tracked_anchor_widgets:
            return

        widget.installEventFilter(self)
        widget.destroyed.connect(self._on_anchor_destroyed)
        self._tracked_anchor_widgets.append(widget)

    def _remove_anchor_event_filters(self) -> None:
        for widget in list(self._tracked_anchor_widgets):
            try:
                widget.removeEventFilter(self)
            except RuntimeError:
                # The Qt object may already be deleted.
                pass

        self._tracked_anchor_widgets.clear()

    def _on_anchor_destroyed(self, *_args: object) -> None:
        self._anchor_widget = None
        self._tracked_anchor_widgets.clear()
        self.clear()

    def _reposition_to_anchor(self) -> None:
        if self._anchor_widget is None:
            return

        self.adjustSize()
        self._position_above(self._anchor_widget)

    def _position_above(self, target: Optional[QtWidgets.QWidget]) -> None:
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
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _apply_styles(self) -> None:
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
