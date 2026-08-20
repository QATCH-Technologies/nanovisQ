from __future__ import annotations

from enum import IntEnum
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.overlay_shell import OverlayActivity
from QATCH.ui.components.window_utils import app_window_bounds_global, find_app_window
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css


class UpdateNotificationBadge(QtWidgets.QWidget):
    """Small floating pill that appears above an UpdateStatusIcon.

    Dismissing hides the badge but leaves the icon in its current state.
    Clicking anywhere on the badge body (other than the dismiss button)
    emits `action_requested`, identical to clicking the icon itself.
    """

    action_requested = QtCore.pyqtSignal()
    dismissed = QtCore.pyqtSignal()

    _RADIUS = 10.0

    def __init__(
        self,
        anchor: QtWidgets.QWidget,
        text: str = "Software update available",
    ) -> None:
        # Own this badge by the app's top-level window instead of leaving it
        # parentless. Combined with dropping WindowStaysOnTopHint below, this
        # keeps the badge above the app's own windows (owned-window Z-order)
        # without making it a system-wide always-on-top window that floats
        # over other applications too.
        app_window = find_app_window() or anchor.window()
        super().__init__(app_window)
        self._anchor = anchor
        self._app_window = app_window
        self._bg = QtGui.QColor(30, 38, 48, 235)
        self._border = QtGui.QColor(255, 255, 255, 45)

        self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint, True)
        self.setWindowFlag(QtCore.Qt.WindowType.Tool, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        # Track the app window so the badge follows it (e.g. when dragged to
        # a different display) instead of staying anchored to wherever it
        # was first shown.
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
        """Re-derives the pill's background/border/text colors from the
        active theme so the badge reads correctly in both light and dark
        mode instead of a single fixed dark-glass look.
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

    # ── Painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        rf = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setBrush(QtGui.QBrush(self._bg))
        p.setPen(QtGui.QPen(self._border, 1.0))
        p.drawRoundedRect(rf, self._RADIUS, self._RADIUS)
        p.end()

    # ── Positioning ───────────────────────────────────────────────────────────

    def reposition(self) -> None:
        """Place the badge above the anchor icon, right-aligned with it.

        Above rather than below: several anchors (e.g. the software-update
        icon in the bottom status bar) sit close to the app window's own
        bottom edge, where a below-placed badge got clamped by the
        window-bounds constraint further down and ended up rendering
        directly on top of the icon instead of offset from it.
        """
        global_pos = self._anchor.mapToGlobal(QtCore.QPoint(0, 0))
        anchor_right = global_pos.x() + self._anchor.width()
        x = anchor_right - self.width()
        y = global_pos.y() - self.height() - 4

        # Constrain to the app's own window, not just the screen: the badge
        # is a separate top-level (frameless) widget, so clamping only to
        # screen geometry lets it drift past the app window's own edge onto
        # the desktop whenever the anchor icon sits near a window edge (e.g.
        # a maximized-but-not-fullscreen app, or a smaller window). Falls
        # back to screen geometry if the app window can't be resolved.
        app_window = find_app_window()
        if app_window is not None:
            bounds = app_window_bounds_global(app_window)
            x = max(bounds.left(), min(x, bounds.right() - self.width()))
            y = max(bounds.top(), min(y, bounds.bottom() - self.height()))
        else:
            screen = QtWidgets.QApplication.screenAt(global_pos)
            if screen:
                sg = screen.geometry()
                x = max(sg.left(), min(x, sg.right() - self.width()))
                y = max(sg.top(), min(y, sg.bottom() - self.height()))

        self.move(x, y)

    def show_above(self) -> None:
        self.adjustSize()
        self.reposition()
        self.show()
        self.raise_()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        """Follows the app window when it moves/resizes (e.g. dragged to a
        different display) instead of staying anchored to its old position.
        """
        if watched is self._app_window and self.isVisible() and event.type() in (
            QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.WindowStateChange,
        ):
            self.reposition()
        return super().eventFilter(watched, event)

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
    """A themed icon button that reflects update status with a solid color.

    Renders an SVG icon tinted solid green / yellow / red / gray based on
    the current :class:`State` - no pulsing/flashing. Clicking the button
    emits :pyqtSignal:`update_requested` when the state is actionable (not
    UP_TO_DATE or CHECKING).

    A small :class:`UpdateNotificationBadge` floats above this icon whenever
    the state transitions to OPTIONAL or MANDATORY. It is auto-dismissed when
    the state returns to UP_TO_DATE or CHECKING.

    Args:
        icon_path: Filesystem path to the SVG icon to display.
        size: Square pixel dimension of the button (default 20).
        parent: Optional parent widget.
        badge_text: Text shown in the floating notification badge, e.g.
            "Software update available" or "Firmware update available".
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

    def __init__(
        self,
        icon_path: str,
        size: int = 20,
        parent: Optional[QtWidgets.QWidget] = None,
        badge_text: str = "Software update available",
    ) -> None:
        super().__init__(parent)
        self._icon_path = icon_path
        self._size = size
        self._state = self.State.UNKNOWN
        self._detail = ""
        self._badge_text = badge_text
        self._badge_dismissed = False
        self._badge: Optional[UpdateNotificationBadge] = None
        # `_refresh_icon` rebuilds a scaled+tinted QPixmap/QIcon from scratch;
        # defer it to paintEvent (guarded by this flag) so it only actually
        # runs once per real screen refresh, and not at all while hidden,
        # since paintEvent doesn't fire for an invisible widget.
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
        """Suppresses the badge for as long as any overlay is open (see
        `_show_badge`), then brings it back once every overlay has closed.
        Reuses `dismiss_badge`/`resync_badge` - the same suppress/restore
        pair a caller already uses to hold the badge off screen for the
        duration of a run in progress - so the two suppression reasons
        compose correctly instead of racing (`resync_badge` re-checks
        `isEnabled()`, which a still-active run keeps `False` regardless
        of overlay state)."""
        if any_open:
            self.dismiss_badge()
        else:
            self.resync_badge()

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
        self._state = state
        self._detail = detail
        self._update_tooltip()
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
            self._badge = UpdateNotificationBadge(self, text=self._badge_text)
            self._badge.action_requested.connect(self._on_clicked)
            self._badge.dismissed.connect(self._on_badge_dismissed)
        # Also gated on isEnabled(): a caller suppressing the badge for the
        # duration of some blocking activity (see dismiss_badge/resync_badge)
        # disables the icon for that same span, so any _sync_badge() call
        # that sneaks in during it (e.g. a firmware status update arriving
        # mid-run) can't pop the badge back up early.
        if not self.isVisible() or not self.isEnabled():
            return
        # The badge is a separate owned top-level (Tool) window, so it would
        # otherwise always paint above an overlay's child-widget content
        # regardless of which is logically "on top" - suppress it for as
        # long as any overlay is open instead (see _on_overlay_activity_
        # changed, which resyncs it once every overlay has closed).
        if OverlayActivity.instance().is_any_open():
            return
        self._badge.show_above()

    def _on_badge_dismissed(self) -> None:
        self._badge_dismissed = True

    def dismiss_badge(self) -> None:
        """Hides the notification badge for now, WITHOUT marking it
        permanently dismissed - unlike a real user-initiated dismiss (click
        the badge's own ✕), which stays hidden for the rest of this update
        cycle regardless of anything else. Meant for a caller that needs to
        suppress the popup for the duration of some other blocking activity
        (a run in progress) and bring it back afterward via
        `resync_badge()` - the icon's underlying state is left untouched
        either way."""
        if self._badge and self._badge.isVisible():
            self._badge.hide()

    def resync_badge(self) -> None:
        """Re-applies the current state's badge visibility.

        Call after `dismiss_badge()`'s suppression ends (e.g. once the icon
        is re-enabled) so the popup reappears if it's still relevant and the
        user never actually dismissed it themselves."""
        self._sync_badge()

    def _mark_icon_dirty(self) -> None:
        """Marks the icon for rebuild on the next paint and requests one.

        Called on state/theme changes. Deferring the actual rebuild (a
        QPixmap+QPainter+QIcon pass) to `paintEvent` means a burst of
        changes right before a repaint collapses into a single rebuild.
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

    # Applied on top of the state's own alpha when the button is disabled -
    # keeps the same status color (still reads as "green"/"yellow"/"red")
    # instead of Qt's default auto-generated Disabled-mode icon, which
    # desaturates to gray and loses that meaning entirely.
    _DISABLED_ALPHA_SCALE = 0.4

    def _tinted_pixmap(self, color: QtGui.QColor, alpha_f: float = 1.0) -> QtGui.QPixmap:
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

        # Applying `alpha_f` via painter opacity in the pass above (the
        # original approach) left it in effect for the SourceAtop fillRect
        # too - at low alpha_f that let the source SVG's own baked-in
        # color bleed back through the recolor instead of cleanly tinting
        # it, producing a muddy, hue-shifted result rather than a dimmer
        # version of `color`. Locking in the pure tint first, then scaling
        # the whole already-tinted pixmap's alpha as a separate pass,
        # keeps the hue exact at any alpha_f.
        dimmed = QtGui.QPixmap(dst.size())
        dimmed.fill(QtCore.Qt.GlobalColor.transparent)
        dp = QtGui.QPainter(dimmed)
        dp.setOpacity(alpha_f)
        dp.drawPixmap(0, 0, dst)
        dp.end()
        return dimmed

    def _tinted_icon(self, color: QtGui.QColor, alpha_f: float = 1.0) -> QtGui.QIcon:
        pix = self._tinted_pixmap(color, alpha_f)
        return QtGui.QIcon(pix) if not pix.isNull() else QtGui.QIcon()

    def _refresh_icon(self) -> None:
        """Renders the icon as a solid tint for the current state - no
        pulsing/flashing. UNKNOWN (no info yet) renders dimmed since it
        isn't one of the three status colors; every other state is full
        opacity.

        Explicitly supplies a Disabled-mode pixmap (same hue, dimmer alpha)
        rather than leaving it to QIcon's own auto-generated Disabled
        variant - a plain single-pixmap QIcon would otherwise render
        desaturated/gray the instant `setEnabled(False)` is called, which
        reads as "status unknown" rather than "still yellow, just inert".
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
        # Re-show badge after icon becomes visible (e.g. window restore) -
        # routed through _show_badge() rather than self._badge.show_above
        # directly so this path also gets its isEnabled()/overlay-open
        # guards, not just the state check above.
        if self._state in (self.State.OPTIONAL, self.State.MANDATORY):
            if not self._badge_dismissed and self._badge:
                QtCore.QTimer.singleShot(50, self._show_badge)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        super().hideEvent(event)
        if self._badge and self._badge.isVisible():
            self._badge.hide()

    def _on_clicked(self) -> None:
        if self._state not in (self.State.UP_TO_DATE, self.State.CHECKING):
            if self._badge and self._badge.isVisible():
                self._badge.hide()
            self.update_requested.emit()
