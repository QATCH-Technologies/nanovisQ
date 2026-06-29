from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Optional


class AvatarLabel(QtWidgets.QWidget):
    """Circular avatar rendered with QATCH brand-blue gradient + user initials."""

    def __init__(self, initials: str, parent=None) -> None:
        super().__init__(parent)
        self._initials = initials[:2].upper() if initials else "?"
        self.setAutoFillBackground(False)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing)
        r = min(self.width(), self.height()) - 2
        x = (self.width() - r) / 2
        y = (self.height() - r) / 2
        rect = QtCore.QRectF(x, y, r, r)

        grad = QtGui.QRadialGradient(rect.center(), r / 2)
        grad.setColorAt(0.0, QtGui.QColor(0, 158, 210))
        grad.setColorAt(1.0, QtGui.QColor(0, 100, 160))
        p.setBrush(QtGui.QBrush(grad))
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 90), 1.5))
        p.drawEllipse(rect)

        # Shimmer half-circle
        shimmer = QtGui.QLinearGradient(0, float(rect.top()), 0, float(rect.center().y()))
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 55))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.setBrush(QtGui.QBrush(shimmer))
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(rect)

        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QtGui.QColor(255, 255, 255, 235))
        p.drawText(rect.toRect(), QtCore.Qt.AlignmentFlag.AlignCenter, self._initials)
        p.end()


class AccountInnerPanel(QtWidgets.QWidget):
    """Inner glass-morphism panel for the account popup.

    Paints the frosted-glass background with rounded corners.  The outer
    :class:`GlassAccountPopup` applies a :class:`QGraphicsDropShadowEffect`
    to this widget so the shadow follows the painted alpha mask, producing
    a soft, rounded drop shadow.  This mirrors the pattern used by
    ``RecoveryFilterWidget`` to avoid the rectangular OS popup outline.
    """

    _RADIUS: float = 10.0

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rect_f = QtCore.QRectF(self.rect())
        _R = self._RADIUS

        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, _R, _R)
        p.setClipPath(clip)

        # Frosted white base - slightly higher alpha than before because the
        # outer widget is fully transparent (no manual shadow underlay)
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 235))
        p.fillRect(self.rect(), QtGui.QColor(228, 235, 241, 28))

        # Top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, 44)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 80))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Dual borders (outer warm white, inner cool grey)
        p.setClipping(False)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), _R, _R)
        p.setPen(QtGui.QPen(QtGui.QColor(200, 210, 220, 90), 1.0))
        p.drawRoundedRect(rect_f.adjusted(1.5, 1.5, -1.5, -1.5), _R - 1.5, _R - 1.5)

        p.end()


class AccountPopup(QtWidgets.QWidget):
    """Frosted-glass dropdown panel for the Account toolbar button.

    Displays the active user's avatar, full name, and role badge.  Admin users
    additionally see a "Manage Users…" shortcut.  The popup uses ``Qt.Popup``
    so it closes automatically on any outside click.

    Implementation notes
    --------------------
    The popup is built as a transparent outer ``QWidget`` (this class) wrapping
    an inner :class:`_GlassAccountInnerPanel`.  The outer widget reserves margin
    space around the inner panel so a :class:`QGraphicsDropShadowEffect` applied
    to the inner panel renders a soft, rounded shadow that follows the panel's
    border-radius - exactly the trick used by ``RecoveryFilterWidget`` to fix
    the sharp shadow corners produced by manual painted shadows.

    The popup also tracks its main window: when the main window is resized or
    moved, the popup closes itself so it never floats outside the application.
    """

    # Margins reserved around the inner panel for the drop shadow.  Bottom is
    # larger to accommodate the shadow's positive Y offset.
    _SHADOW_MARGIN_L = 22
    _SHADOW_MARGIN_T = 18
    _SHADOW_MARGIN_R = 22
    _SHADOW_MARGIN_B = 26

    def __init__(
        self,
        open_manager_cb=None,
        sign_out_cb=None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(
            parent,
            QtCore.Qt.WindowType.Popup
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.NoDropShadowWindowHint,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

        self._open_manager_cb = open_manager_cb
        self._sign_out_cb = sign_out_cb
        self._main_window: Optional[QtWidgets.QWidget] = None  # set by show_anchored_to

        # -- outer container with shadow margins --
        self._panel = AccountInnerPanel(self)
        self._panel.setObjectName("AccountPopupInner")

        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(
            self._SHADOW_MARGIN_L,
            self._SHADOW_MARGIN_T,
            self._SHADOW_MARGIN_R,
            self._SHADOW_MARGIN_B,
        )
        outer_layout.setSpacing(0)
        outer_layout.addWidget(self._panel)

        # Soft drop shadow that follows the inner panel's painted alpha mask
        shadow = QtWidgets.QGraphicsDropShadowEffect(self._panel)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 4)
        shadow.setColor(QtGui.QColor(0, 20, 40, 110))
        self._panel.setGraphicsEffect(shadow)

        # -- entrance animation (matches the advanced menu) --
        # Fade the whole popup window in (setWindowOpacity, NOT a graphics
        # effect - an opacity effect on this panel would clash with the drop
        # shadow above and cause the same ghosting seen in the advanced panel),
        # paired with a brief downward slide so it eases out from the anchor.
        self._enter_fade = QtCore.QVariantAnimation(self)
        self._enter_fade.setDuration(200)
        self._enter_fade.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._enter_fade.setStartValue(0.0)
        self._enter_fade.setEndValue(1.0)
        self._enter_fade.valueChanged.connect(lambda v: self.setWindowOpacity(float(v)))
        self._enter_fade.finished.connect(lambda: self.setWindowOpacity(1.0))

        self._enter_slide = QtCore.QPropertyAnimation(self, b"pos", self)
        self._enter_slide.setDuration(220)
        self._enter_slide.setEasingCurve(QtCore.QEasingCurve.OutCubic)

        # -- resolve current session info (lazy import avoids circular deps) --
        # session_info() returns: [name, initials, role.name, created, modified, accessed]
        accessed: Optional[str] = None
        try:
            from QATCH.common.userProfiles import UserProfiles, UserRoles  # noqa: PLC0415

            is_valid, user_info = UserProfiles.session_info()
            if is_valid and user_info:
                name = user_info[0] or "Unknown"
                initials = user_info[1] or "?"
                role_name = user_info[2] or "NONE"
                # Index 5 = "accessed" timestamp ("Today, HH:MM:SS" or "YYYY-MM-DD HH:MM:SS")
                accessed = user_info[5] if len(user_info) > 5 else None
            else:
                name, initials, role_name = "Anonymous", "?", "NONE"
            is_admin = role_name == UserRoles.ADMIN.name
            is_signed_in = is_valid
        except Exception:
            name, initials, role_name = "Anonymous", "?", "NONE"
            is_admin = False
            is_signed_in = False

        # -- inner panel layout (all visible content lives here) --
        layout = QtWidgets.QVBoxLayout(self._panel)
        layout.setContentsMargins(14, 14, 14, 12)
        layout.setSpacing(8)

        # Avatar + name/role column
        header_row = QtWidgets.QHBoxLayout()
        header_row.setSpacing(12)

        avatar = AvatarLabel(initials)
        avatar.setFixedSize(44, 44)
        header_row.addWidget(avatar, 0, QtCore.Qt.AlignTop)

        info_col = QtWidgets.QVBoxLayout()
        info_col.setSpacing(3)
        info_col.setContentsMargins(0, 1, 0, 0)

        name_lbl = QtWidgets.QLabel(name)
        name_lbl.setStyleSheet(
            "color: rgba(28,40,52,235); font-weight: bold; font-size: 13px; "
            "background: transparent; border: none;"
        )
        info_col.addWidget(name_lbl)

        # Subtle initials line under the name
        initials_lbl = QtWidgets.QLabel(f"Initials: {initials}")
        initials_lbl.setStyleSheet(
            "color: rgba(70, 90, 110, 180); font-size: 10px; "
            "background: transparent; border: none;"
        )
        info_col.addWidget(initials_lbl)

        _role_palette = {
            "ADMIN": ("rgba(0,118,174,215)", "white"),
            "OPERATE": ("rgba(40,155,75,200)", "white"),
            "ANALYZE": ("rgba(130,80,200,200)", "white"),
            "CAPTURE": ("rgba(200,125,0,200)", "white"),
        }
        bg, fg = _role_palette.get(role_name, ("rgba(140,150,160,160)", "rgba(28,40,52,180)"))
        role_badge = QtWidgets.QLabel(role_name)
        role_badge.setFixedHeight(17)
        role_badge.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 3px; "
            "padding: 1px 6px; font-size: 10px; font-weight: bold; border: none;"
        )
        # Wrap the badge so it doesn't stretch to full column width
        role_row = QtWidgets.QHBoxLayout()
        role_row.setContentsMargins(0, 2, 0, 0)
        role_row.setSpacing(0)
        role_row.addWidget(role_badge)
        role_row.addStretch()
        info_col.addLayout(role_row)

        header_row.addLayout(info_col, 1)
        layout.addLayout(header_row)

        # Last sign-in / status line
        if is_signed_in and accessed:
            last_lbl = QtWidgets.QLabel(f"Last access: {accessed}")
            last_lbl.setStyleSheet(
                "color: rgba(70, 90, 110, 175); font-size: 10px; "
                "background: transparent; border: none; padding-left: 1px;"
            )
            layout.addWidget(last_lbl)
        elif not is_signed_in:
            status_lbl = QtWidgets.QLabel("No active session")
            status_lbl.setStyleSheet(
                "color: rgba(140, 90, 30, 200); font-size: 10px; font-style: italic; "
                "background: transparent; border: none; padding-left: 1px;"
            )
            layout.addWidget(status_lbl)

        show_manage = is_admin
        show_sign_out = is_signed_in
        if show_manage or show_sign_out:
            # Hairline divider
            divider = QtWidgets.QFrame()
            divider.setFrameShape(QtWidgets.QFrame.HLine)
            divider.setStyleSheet(
                "QFrame { background: rgba(200,210,220,130); border: none; max-height: 1px; }"
            )
            layout.addWidget(divider)

        if show_manage:
            icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "user-circle.svg")
            manage_btn = GlassPushButton(" Manage Users…")
            manage_btn.setIcon(QtGui.QIcon(icon_path))
            manage_btn.setIconSize(QtCore.QSize(16, 16))
            manage_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(0, 118, 174, 18);
                    color: rgba(0, 118, 174, 230);
                    border: 1px solid rgba(0, 118, 174, 55);
                    border-radius: 5px;
                    padding: 8px 14px 8px 12px;
                    text-align: left;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover  {
                    background: rgba(0, 142, 192, 35);
                    border: 1px solid rgba(0, 118, 174, 110);
                }
                QPushButton:pressed {
                    background: rgba(0, 118, 174, 70);
                    border: 1px solid rgba(0, 118, 174, 160);
                }
            """)
            manage_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            manage_btn.clicked.connect(self._on_manage_users)
            layout.addWidget(manage_btn)

        if show_sign_out:
            icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "sign-out.svg")
            sign_out_btn = GlassPushButton(" Sign Out")
            if os.path.exists(icon_path):
                sign_out_btn.setIcon(QtGui.QIcon(icon_path))
                sign_out_btn.setIconSize(QtCore.QSize(16, 16))
            sign_out_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(200, 70, 40, 16);
                    color: rgba(170, 55, 30, 235);
                    border: 1px solid rgba(200, 70, 40, 60);
                    border-radius: 5px;
                    padding: 8px 14px 8px 12px;
                    text-align: left;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover  {
                    background: rgba(210, 80, 45, 38);
                    border: 1px solid rgba(200, 70, 40, 120);
                }
                QPushButton:pressed {
                    background: rgba(200, 70, 40, 75);
                    border: 1px solid rgba(200, 70, 40, 170);
                }
            """)
            sign_out_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            sign_out_btn.clicked.connect(self._on_sign_out)
            layout.addWidget(sign_out_btn)

        self._panel.setMinimumWidth(230)

    # -- public API -----------------------------------------------------------

    def show_anchored_to(
        self,
        anchor: QtWidgets.QWidget,
        main_window: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Show the popup pinned to ``anchor`` and constrained to ``main_window``.

        The popup's *visible* right edge aligns with the anchor button's right
        edge; the visible top edge sits 2 px below the anchor's bottom.  If the
        popup would extend past the main window's frame, its position is clamped
        so the visible panel stays inside the main window.  When the main window
        is later resized or moved while the popup is open, the popup closes
        itself to avoid floating outside the application.
        """
        self._main_window = main_window
        self.adjustSize()

        size = self.sizeHint()
        popup_w, popup_h = size.width(), size.height()

        # Anchor at the bottom-right corner of the button (in screen coords)
        anchor_br = anchor.mapToGlobal(QtCore.QPoint(anchor.width(), anchor.height()))

        # Position so the *visible* panel right edge aligns with the button's
        # right edge, 2 px below the button.  Account for the transparent
        # shadow margins on the outer widget.
        x = anchor_br.x() + self._SHADOW_MARGIN_R - popup_w
        y = anchor_br.y() + 2 - self._SHADOW_MARGIN_T

        # Clamp so the visible panel stays inside the main window
        x, y = self._clamp_to_main_window(x, y, popup_w, popup_h, anchor)

        # Track resize/move events on the main window so the popup never
        # ends up floating outside the application after a resize.
        if self._main_window is not None:
            self._main_window.installEventFilter(self)

        # Entrance animation: start slightly above the final position and fade
        # in as it slides down to (x, y) - same feel as the advanced menu.
        final_pos = QtCore.QPoint(x, y)
        start_pos = QtCore.QPoint(x, y - 12)
        self.move(start_pos)
        self.setWindowOpacity(0.0)
        self.show()

        self._enter_slide.stop()
        self._enter_slide.setStartValue(start_pos)
        self._enter_slide.setEndValue(final_pos)
        self._enter_slide.start()

        self._enter_fade.stop()
        self._enter_fade.start()

    # -- positioning helpers --------------------------------------------------

    def _visible_rect_for(self, x: int, y: int, w: int, h: int) -> QtCore.QRect:
        """Return the *visible* panel rect for an outer-widget position.

        The outer widget reserves transparent shadow margins, so the visible
        rect is the outer rect minus those margins.
        """
        return QtCore.QRect(
            x + self._SHADOW_MARGIN_L,
            y + self._SHADOW_MARGIN_T,
            w - self._SHADOW_MARGIN_L - self._SHADOW_MARGIN_R,
            h - self._SHADOW_MARGIN_T - self._SHADOW_MARGIN_B,
        )

    def _clamp_to_main_window(
        self,
        x: int,
        y: int,
        popup_w: int,
        popup_h: int,
        anchor: QtWidgets.QWidget,
    ) -> tuple:
        """Adjust ``(x, y)`` so the visible panel stays inside the main window.

        Falls back to the anchor's screen geometry if no main window is set.
        """
        # Prefer the anchor widget's own top-level window (content geometry, screen
        # coords) so the popup is always clamped against the window that actually
        # contains the button - regardless of which QWidget was passed as
        # main_window.  Fall back to main_window, then the screen.
        top_level = anchor.window() if anchor is not None else None
        if top_level is not None:
            bounds = top_level.geometry()
        elif self._main_window is not None:
            bounds = self._main_window.geometry()
        else:
            screen = QtWidgets.QApplication.screenAt(anchor.mapToGlobal(QtCore.QPoint(0, 0)))
            bounds = screen.availableGeometry() if screen is not None else QtCore.QRect()

        if bounds.isNull():
            return x, y

        visible = self._visible_rect_for(x, y, popup_w, popup_h)

        # Horizontal clamp
        if visible.right() > bounds.right():
            x -= visible.right() - bounds.right()
            visible = self._visible_rect_for(x, y, popup_w, popup_h)
        if visible.left() < bounds.left():
            x += bounds.left() - visible.left()
            visible = self._visible_rect_for(x, y, popup_w, popup_h)

        # Vertical clamp - if the popup spills off the bottom, flip it above
        # the anchor button.
        if visible.bottom() > bounds.bottom():
            anchor_top = anchor.mapToGlobal(QtCore.QPoint(0, 0)).y()
            y_above = anchor_top - 2 - popup_h + self._SHADOW_MARGIN_B
            visible_above = self._visible_rect_for(x, y_above, popup_w, popup_h)
            if visible_above.top() >= bounds.top():
                y = y_above
            else:
                # Neither orientation fits - just clamp to the bottom edge
                y -= visible.bottom() - bounds.bottom()

        return x, y

    # -- event handling -------------------------------------------------------

    def eventFilter(  # noqa: N802 - Qt naming
        self, watched: QtCore.QObject, event: QtCore.QEvent
    ) -> bool:
        """Close the popup if the main window is resized or moved.

        The popup is positioned in screen coordinates against the anchor at the
        time of show.  Re-anchoring on every resize would race the layout
        engine, so the safer behaviour is to dismiss the popup and let the user
        re-open it once the new window geometry has settled.
        """
        if watched is self._main_window and event.type() in (
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.WindowStateChange,
        ):
            self.close()
        return super().eventFilter(watched, event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt naming
        if self._main_window is not None:
            try:
                self._main_window.removeEventFilter(self)
            except Exception:
                pass
            self._main_window = None
        super().closeEvent(event)

    # -- slots ----------------------------------------------------------------

    def _on_manage_users(self) -> None:
        self.close()
        if self._open_manager_cb:
            self._open_manager_cb()

    def _on_sign_out(self) -> None:
        self.close()
        if self._sign_out_cb:
            self._sign_out_cb()
