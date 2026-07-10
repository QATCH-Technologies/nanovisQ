from __future__ import annotations

from enum import IntEnum
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.styles.theme_manager import ThemeManager


class UpdateNotificationBadge(QtWidgets.QWidget):
    """Small floating pill that appears above an UpdateStatusIcon.

    Dismissing hides the badge but leaves the icon in its current state.
    Clicking anywhere on the badge body (other than the dismiss button)
    emits `action_requested`, identical to clicking the icon itself.
    """

    action_requested = QtCore.pyqtSignal()
    dismissed = QtCore.pyqtSignal()

    _BG = QtGui.QColor(30, 38, 48, 235)
    _BORDER = QtGui.QColor(255, 255, 255, 45)
    _RADIUS = 10.0

    def __init__(self, anchor: QtWidgets.QWidget) -> None:
        super().__init__(None)
        self._anchor = anchor

        self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint, True)
        self.setWindowFlag(QtCore.Qt.WindowType.Tool, True)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 5, 5)
        layout.setSpacing(6)

        self._label = QtWidgets.QLabel("Software update available")
        self._label.setStyleSheet(
            "QLabel { color: #e8edf2; font-size: 11px; font-weight: 600;"
            " background: transparent; }"
        )
        layout.addWidget(self._label)

        self._dismiss_btn = QtWidgets.QToolButton()
        self._dismiss_btn.setText("✕")
        self._dismiss_btn.setFixedSize(16, 16)
        self._dismiss_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._dismiss_btn.setStyleSheet(
            "QToolButton { color: rgba(200,210,220,180); font-size: 10px;"
            " background: transparent; border: none; }"
            "QToolButton:hover { color: white; }"
        )
        self._dismiss_btn.clicked.connect(self._on_dismiss)
        layout.addWidget(self._dismiss_btn)

        self.adjustSize()

    # ── Painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        rf = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setBrush(QtGui.QBrush(self._BG))
        p.setPen(QtGui.QPen(self._BORDER, 1.0))
        p.drawRoundedRect(rf, self._RADIUS, self._RADIUS)
        p.end()

    # ── Positioning ───────────────────────────────────────────────────────────

    def reposition(self) -> None:
        """Place the badge below the anchor icon, right-aligned with it."""
        global_pos = self._anchor.mapToGlobal(QtCore.QPoint(0, 0))
        anchor_right = global_pos.x() + self._anchor.width()
        x = anchor_right - self.width()
        y = global_pos.y() + self._anchor.height() + 4

        # Constrain to the screen that contains the anchor widget so the
        # badge never drifts to a secondary monitor.
        screen = QtWidgets.QApplication.screenAt(global_pos)
        if screen:
            sg = screen.geometry()
            x = max(sg.left(), min(x, sg.right() - self.width()))
            y = max(sg.top(), min(y, sg.bottom() - self.height()))

        self.move(x, y)

    def show_below(self) -> None:
        self.adjustSize()
        self.reposition()
        self.show()
        self.raise_()

    # ── Interaction ───────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        # Clicks on the body (not the dismiss button) trigger the update action
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.action_requested.emit()
        super().mousePressEvent(event)

    def _on_dismiss(self) -> None:
        self.hide()
        self.dismissed.emit()


class UpdateStatusIcon(QtWidgets.QToolButton):
    """A themed, animated icon button that reflects update status.

    Renders an SVG icon tinted to green / yellow / red / gray based on the
    current :class:`State`. Non-static states pulse via a QVariantAnimation.
    Clicking the button emits :pyqtSignal:`update_requested` when the state
    is actionable (not UP_TO_DATE or CHECKING).

    A small :class:`UpdateNotificationBadge` floats above this icon whenever
    the state transitions to OPTIONAL or MANDATORY. It is auto-dismissed when
    the state returns to UP_TO_DATE or CHECKING.

    Args:
        icon_path: Filesystem path to the SVG icon to display.
        size: Square pixel dimension of the button (default 20).
        parent: Optional parent widget.
    """

    update_requested = QtCore.pyqtSignal()

    class State(IntEnum):
        UNKNOWN = 0
        CHECKING = 1
        UP_TO_DATE = 2
        OPTIONAL = 3
        MANDATORY = 4

    _COLORS = {
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

    _TOOLTIPS = {
        State.UNKNOWN: "Update status unknown - click to check",
        State.CHECKING: "Checking for updates...",
        State.UP_TO_DATE: "Up to date",
        State.OPTIONAL: "Update available - click to update",
        State.MANDATORY: "Required update - click to install",
    }

    _PULSING = (State.CHECKING, State.OPTIONAL, State.MANDATORY)

    def __init__(
        self,
        icon_path: str,
        size: int = 20,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._icon_path = icon_path
        self._size = size
        self._state = self.State.UNKNOWN
        self._detail = ""
        self._glow: float = 1.0
        self._badge_dismissed = False
        self._badge: Optional[UpdateNotificationBadge] = None
        # `_refresh_icon` rebuilds a scaled+tinted QPixmap/QIcon from scratch;
        # while pulsing that would otherwise happen on every ~16ms animation
        # frame, indefinitely, for as long as an update stays pending. Defer
        # it to paintEvent (guarded by this flag) so it only actually runs
        # once per real screen refresh - and not at all while hidden, since
        # paintEvent doesn't fire for an invisible widget.
        self._icon_dirty = True

        self.setFixedSize(size, size)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setIconSize(QtCore.QSize(size, size))
        self.setStyleSheet("QToolButton { background: transparent; border: none; }")
        self.setAutoRaise(True)

        self._base_pixmap = QtGui.QPixmap(icon_path)

        self._pulse = QtCore.QVariantAnimation(self)
        self._pulse.setStartValue(0.25)
        self._pulse.setEndValue(1.0)
        self._pulse.setDuration(900)
        self._pulse.setEasingCurve(QtCore.QEasingCurve.InOutSine)
        self._pulse.setLoopCount(-1)
        self._pulse.valueChanged.connect(self._on_pulse)

        ThemeManager.instance().themeChanged.connect(lambda _: self._mark_icon_dirty())
        self.clicked.connect(self._on_clicked)
        self._update_tooltip()

    def state(self) -> "UpdateStatusIcon.State":
        return self._state

    def setState(self, state: "UpdateStatusIcon.State", detail: str = "") -> None:
        """Set the update state and optional detail text shown in the tooltip.

        Transitions to OPTIONAL or MANDATORY automatically show the floating
        notification badge (unless the user has already dismissed it for this
        update cycle). Transitioning back to UP_TO_DATE or CHECKING hides the
        badge and resets the dismissed flag for the next update cycle.

        Args:
            state: New :class:`State` value.
            detail: Additional tooltip text, e.g. version strings.
        """
        old_pulsing = self._state in self._PULSING
        self._state = state
        self._detail = detail
        self._update_tooltip()

        new_pulsing = state in self._PULSING
        if new_pulsing:
            if not old_pulsing:
                self._glow = 0.25
                self._pulse.stop()
                self._pulse.start()
        else:
            self._pulse.stop()
            self._glow = 1.0

        self._mark_icon_dirty()
        self._sync_badge()

    def _sync_badge(self) -> None:
        """Show or hide the notification badge based on current state."""
        if self._state in (self.State.OPTIONAL, self.State.MANDATORY):
            if not self._badge_dismissed:
                self._show_badge()
        else:
            # Reset dismissed flag when state leaves actionable range
            self._badge_dismissed = False
            if self._badge and self._badge.isVisible():
                self._badge.hide()

    def _show_badge(self) -> None:
        if self._badge is None:
            self._badge = UpdateNotificationBadge(self)
            self._badge.action_requested.connect(self._on_clicked)
            self._badge.dismissed.connect(self._on_badge_dismissed)
        if not self.isVisible():
            return
        self._badge.show_below()

    def _on_badge_dismissed(self) -> None:
        self._badge_dismissed = True

    # ── Pulse animation ───────────────────────────────────────────────────────

    def _on_pulse(self, v: float) -> None:
        self._glow = v
        self._mark_icon_dirty()

    def _mark_icon_dirty(self) -> None:
        """Marks the icon for rebuild on the next paint and requests one.

        Called on every pulse frame (up to 60fps, indefinitely while an
        update is pending) as well as on state/theme changes. Actually
        rebuilding the icon here every time would mean paying for a
        QPixmap+QPainter+QIcon pass on every frame regardless of whether a
        screen refresh ever happens for it; deferring to `paintEvent`
        collapses that to one rebuild per real repaint.
        """
        self._icon_dirty = True
        self.update()

    # ── Tooltip ───────────────────────────────────────────────────────────────

    def _update_tooltip(self) -> None:
        tip = self._TOOLTIPS.get(self._state, "")
        if self._detail:
            tip = f"{tip}\n{self._detail}"
        self.setToolTip(tip)

    # ── Icon rendering ────────────────────────────────────────────────────────

    def _tinted_icon(self, color: QtGui.QColor, alpha_f: float = 1.0) -> QtGui.QIcon:
        if self._base_pixmap.isNull():
            return QtGui.QIcon()
        base = self._base_pixmap.scaled(
            self._size,
            self._size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        dst = QtGui.QPixmap(base.size())
        dst.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(dst)
        p.setOpacity(alpha_f)
        p.drawPixmap(0, 0, base)
        p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
        p.fillRect(dst.rect(), color)
        p.end()
        return QtGui.QIcon(dst)

    def _refresh_icon(self) -> None:
        mode = ThemeManager.instance().mode().value
        colors = self._COLORS.get(mode, self._COLORS["light"])
        color = colors.get(self._state, colors[self.State.UNKNOWN])
        if self._state in self._PULSING:
            alpha = max(0.35, self._glow)
        elif self._state == self.State.UNKNOWN:
            alpha = 0.5
        else:
            alpha = 1.0
        self.setIcon(self._tinted_icon(color, alpha))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Rebuilds the icon (if dirty) immediately before painting, then
        defers to the normal QToolButton paint.

        Args:
            event (QtGui.QPaintEvent): The paint event parameters provided
                by the Qt framework.
        """
        if self._icon_dirty:
            self._refresh_icon()
            self._icon_dirty = False
        super().paintEvent(event)

    # ── Interaction ───────────────────────────────────────────────────────────

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        # Re-show badge after icon becomes visible (e.g. window restore)
        if self._state in (self.State.OPTIONAL, self.State.MANDATORY):
            if not self._badge_dismissed and self._badge:
                QtCore.QTimer.singleShot(50, self._badge.show_below)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        super().hideEvent(event)
        if self._badge and self._badge.isVisible():
            self._badge.hide()

    def _on_clicked(self) -> None:
        if self._state not in (self.State.UP_TO_DATE, self.State.CHECKING):
            if self._badge and self._badge.isVisible():
                self._badge.hide()
            self.update_requested.emit()
