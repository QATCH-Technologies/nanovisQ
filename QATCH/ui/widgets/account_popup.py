from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Optional
import os
from QATCH.ui.components.glass_push_button import GlassPushButton
from QATCH.common.architecture import Architecture
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css


def _tinted_icon(path: str, color: QtGui.QColor, size: int = 16) -> QtGui.QIcon:
    """Returns a copy of the icon at *path* fully painted in *color*.

    Uses SourceAtop composition so the tint respects the original alpha
    channel - transparent SVG areas stay transparent. Mirrors the
    established pattern in glass_dialog._tinted_icon /
    user_profiles_manager_widget._tinted_icon.
    """
    src = QtGui.QIcon(path).pixmap(size, size)
    dst = QtGui.QPixmap(src.size())
    dst.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(dst)
    p.drawPixmap(0, 0, src)
    p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
    p.fillRect(dst.rect(), color)
    p.end()
    return QtGui.QIcon(dst)


class AvatarLabel(QtWidgets.QWidget):
    """Circular avatar rendered with QATCH brand-blue gradient + user initials."""

    def __init__(self, initials: str, parent=None) -> None:
        super().__init__(parent)
        self._initials = initials[:2].upper() if initials else "?"
        self.setAutoFillBackground(False)
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        tok = ThemeManager.instance().tokens()
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing)
        r = min(self.width(), self.height()) - 2
        x = (self.width() - r) / 2
        y = (self.height() - r) / 2
        rect = QtCore.QRectF(x, y, r, r)

        grad = QtGui.QRadialGradient(rect.center(), r / 2)
        grad.setColorAt(0.0, QtGui.QColor(*tok["account_avatar_grad_start"]))
        grad.setColorAt(1.0, QtGui.QColor(*tok["account_avatar_grad_end"]))
        p.setBrush(QtGui.QBrush(grad))
        p.setPen(QtGui.QPen(QtGui.QColor(*tok["account_avatar_ring"]), 1.5))
        p.drawEllipse(rect)

        # Shimmer half-circle
        shimmer_rgb = tok["account_avatar_shimmer"][:3]
        shimmer = QtGui.QLinearGradient(0, float(rect.top()), 0, float(rect.center().y()))
        shimmer.setColorAt(0.0, QtGui.QColor(*tok["account_avatar_shimmer"]))
        shimmer.setColorAt(1.0, QtGui.QColor(*shimmer_rgb, 0))
        p.setBrush(QtGui.QBrush(shimmer))
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(rect)

        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QtGui.QColor(*tok["account_avatar_text"]))
        p.drawText(rect.toRect(), QtCore.Qt.AlignmentFlag.AlignCenter, self._initials)
        p.end()


class AccountInnerPanel(QtWidgets.QWidget):
    """Inner glass-morphism panel for the account popup.

    Paints the frosted-glass background with rounded corners.  The outer
    :class:`GlassAccountPopup` applies a :class:`QGraphicsDropShadowEffect`
    to this widget so the shadow follows the painted alpha mask, producing
    a soft, rounded drop shadow.  This mirrors the pattern used by
    `RecoveryFilterWidget` to avoid the rectangular OS popup outline.
    """

    _RADIUS: float = 10.0

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        tok = ThemeManager.instance().tokens()
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rect_f = QtCore.QRectF(self.rect())
        _R = self._RADIUS

        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, _R, _R)
        p.setClipPath(clip)

        # Frosted glass base - same "glass card" tokens as GlassDialog/plot
        # cards, so the popup matches the app's other frosted surfaces in
        # both light and dark mode.
        p.fillRect(self.rect(), QtGui.QColor(*tok["plot_glass_base"]))
        p.fillRect(self.rect(), QtGui.QColor(*tok["plot_glass_overlay"]))

        # Top shimmer
        shimmer_rgb = tok["plot_glass_shimmer_top"][:3]
        shimmer = QtGui.QLinearGradient(0, 0, 0, 44)
        shimmer.setColorAt(0.0, QtGui.QColor(*tok["plot_glass_shimmer_top"]))
        shimmer.setColorAt(1.0, QtGui.QColor(*shimmer_rgb, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Dual borders (outer rim, inner inset)
        p.setClipping(False)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(*tok["plot_glass_rim"]), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), _R, _R)
        p.setPen(QtGui.QPen(QtGui.QColor(*tok["plot_glass_inset"]), 1.0))
        p.drawRoundedRect(rect_f.adjusted(1.5, 1.5, -1.5, -1.5), _R - 1.5, _R - 1.5)

        p.end()


class AccountPopup(QtWidgets.QWidget):
    """Frosted-glass dropdown panel for the Account toolbar button.

    Displays the active user's avatar, full name, and role badge.  Admin users
    additionally see a "Manage Users…" shortcut.  The popup uses `Qt.Popup`
    so it closes automatically on any outside click.

    Implementation notes
    --------------------
    The popup is built as a transparent outer `QWidget` (this class) wrapping
    an inner :class:`_GlassAccountInnerPanel`.  The outer widget reserves margin
    space around the inner panel so a :class:`QGraphicsDropShadowEffect` applied
    to the inner panel renders a soft, rounded shadow that follows the panel's
    border-radius - exactly the trick used by `RecoveryFilterWidget` to fix
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

        # -- entrance animation --
        self._enter_fade = QtCore.QVariantAnimation(self)
        self._enter_fade.setDuration(200)
        self._enter_fade.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._enter_fade.setStartValue(0.0)
        self._enter_fade.setEndValue(1.0)
        self._enter_fade.valueChanged.connect(lambda v: self.setWindowOpacity(float(v)))

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

        self._role_name = role_name

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

        self._name_lbl = QtWidgets.QLabel(name)
        info_col.addWidget(self._name_lbl)

        # Subtle initials line under the name
        self._initials_lbl = QtWidgets.QLabel(f"Initials: {initials}")
        info_col.addWidget(self._initials_lbl)

        self._role_badge = QtWidgets.QLabel(role_name)
        self._role_badge.setFixedHeight(17)
        # Wrap the badge so it doesn't stretch to full column width
        role_row = QtWidgets.QHBoxLayout()
        role_row.setContentsMargins(0, 2, 0, 0)
        role_row.setSpacing(0)
        role_row.addWidget(self._role_badge)
        role_row.addStretch()
        info_col.addLayout(role_row)

        header_row.addLayout(info_col, 1)
        layout.addLayout(header_row)

        # Last sign-in / status line
        self._last_lbl: Optional[QtWidgets.QLabel] = None
        self._status_lbl: Optional[QtWidgets.QLabel] = None
        if is_signed_in and accessed:
            self._last_lbl = QtWidgets.QLabel(f"Last access: {accessed}")
            layout.addWidget(self._last_lbl)
        elif not is_signed_in:
            self._status_lbl = QtWidgets.QLabel("No active session")
            self._status_lbl.setStyleSheet("font-style: italic;")
            layout.addWidget(self._status_lbl)

        show_manage = is_admin
        show_sign_out = is_signed_in
        self._divider: Optional[QtWidgets.QFrame] = None
        if show_manage or show_sign_out:
            self._divider = QtWidgets.QFrame()
            self._divider.setFrameShape(QtWidgets.QFrame.HLine)
            layout.addWidget(self._divider)

        self._manage_btn: Optional[GlassPushButton] = None
        if show_manage:
            self._manage_btn = GlassPushButton(" Manage Users…")
            self._manage_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            self._manage_btn.clicked.connect(self._on_manage_users)
            layout.addWidget(self._manage_btn)

        self._sign_out_btn: Optional[GlassPushButton] = None
        if show_sign_out:
            self._sign_out_btn = GlassPushButton(" Sign Out")
            self._sign_out_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            self._sign_out_btn.clicked.connect(self._on_sign_out)
            layout.addWidget(self._sign_out_btn)

        self._panel.setMinimumWidth(230)

        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    # -- theming ----------------------------------------------------------------

    _ROLE_TOKEN_KEYS = {
        "ADMIN": ("account_role_admin_bg", "account_role_admin_text"),
        "OPERATE": ("account_role_operate_bg", "account_role_operate_text"),
        "ANALYZE": ("account_role_analyze_bg", "account_role_analyze_text"),
        "CAPTURE": ("account_role_capture_bg", "account_role_capture_text"),
    }
    _ROLE_DEFAULT_TOKEN_KEYS = ("account_role_default_bg", "account_role_default_text")

    @staticmethod
    def _apply_action_button_style(
        btn: GlassPushButton,
        base_rgba: tuple,
        bg_alphas: tuple,
        border_alphas: tuple,
        text_alpha: int,
    ) -> None:
        """Styles a GlassPushButton as a labelled glass action row - a
        translucent wash of *base_rgba*'s hue at the given (normal, hover,
        pressed) alpha levels, with left-aligned icon + text."""
        r, g, b, _ = base_rgba
        bg_n, bg_h, bg_p = bg_alphas
        bd_n, bd_h, bd_p = border_alphas
        btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba({r}, {g}, {b}, {bg_n});
                color: rgba({r}, {g}, {b}, {text_alpha});
                border: 1px solid rgba({r}, {g}, {b}, {bd_n});
                border-radius: 5px;
                padding: 8px 14px 8px 12px;
                text-align: left;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover  {{
                background: rgba({r}, {g}, {b}, {bg_h});
                border: 1px solid rgba({r}, {g}, {b}, {bd_h});
            }}
            QPushButton:pressed {{
                background: rgba({r}, {g}, {b}, {bg_p});
                border: 1px solid rgba({r}, {g}, {b}, {bd_p});
            }}
        """)

    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_theme()

    def _apply_theme(self) -> None:
        tok = ThemeManager.instance().tokens()

        self._name_lbl.setStyleSheet(
            f"color: {tok_css(tok['text_primary'])}; font-weight: bold; font-size: 13px; "
            "background: transparent; border: none;"
        )
        self._initials_lbl.setStyleSheet(
            f"color: {tok_css(tok['text_secondary'])}; font-size: 10px; "
            "background: transparent; border: none;"
        )

        bg_key, fg_key = self._ROLE_TOKEN_KEYS.get(self._role_name, self._ROLE_DEFAULT_TOKEN_KEYS)
        self._role_badge.setStyleSheet(
            f"background: {tok_css(tok[bg_key])}; color: {tok_css(tok[fg_key])}; "
            "border-radius: 3px; padding: 1px 6px; font-size: 10px; "
            "font-weight: bold; border: none;"
        )

        if self._last_lbl is not None:
            self._last_lbl.setStyleSheet(
                f"color: {tok_css(tok['text_secondary'])}; font-size: 10px; "
                "background: transparent; border: none; padding-left: 1px;"
            )
        if self._status_lbl is not None:
            r, g, b, _ = tok["warning"]
            self._status_lbl.setStyleSheet(
                f"color: rgba({r}, {g}, {b}, 200); font-size: 10px; font-style: italic; "
                "background: transparent; border: none; padding-left: 1px;"
            )
        if self._divider is not None:
            self._divider.setStyleSheet(
                f"QFrame {{ background: {tok_css(tok['ctrl_hairline'])}; "
                "border: none; max-height: 1px; }}"
            )

        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")

        if self._manage_btn is not None:
            self._apply_action_button_style(
                self._manage_btn, tok["accent"], (18, 35, 70), (55, 110, 160), 230
            )
            icon_path = os.path.join(icons_dir, "user-circle.svg")
            if os.path.exists(icon_path):
                self._manage_btn.setIcon(_tinted_icon(icon_path, QtGui.QColor(*tok["accent"][:3])))
                self._manage_btn.setIconSize(QtCore.QSize(16, 16))

        if self._sign_out_btn is not None:
            self._apply_action_button_style(
                self._sign_out_btn, tok["danger"], (16, 38, 75), (60, 120, 170), 235
            )
            icon_path = os.path.join(icons_dir, "sign-out.svg")
            if os.path.exists(icon_path):
                self._sign_out_btn.setIcon(
                    _tinted_icon(icon_path, QtGui.QColor(*tok["danger"][:3]))
                )
                self._sign_out_btn.setIconSize(QtCore.QSize(16, 16))

    # -- public API -----------------------------------------------------------

    def show_anchored_to(
        self,
        anchor: QtWidgets.QWidget,
        main_window: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Show the popup pinned to `anchor` and constrained to `main_window`.

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

        # Show off-screen at opacity=0 so the unavoidable one-frame DWM flash
        # (ShowWindow fires before SetLayeredWindowAttributes can commit alpha=0)
        # occurs at an invisible position.  By the time singleShot(0) fires the
        # event loop has processed SetLayeredWindowAttributes, so opacity=0 is
        # committed before we move the window into the visible anchor area.
        self.setWindowOpacity(0.0)
        self.move(QtCore.QPoint(-9999, -9999))
        self.show()

        self._enter_slide.stop()
        self._enter_slide.setStartValue(start_pos)
        self._enter_slide.setEndValue(final_pos)
        self._enter_fade.stop()

        def _start():
            self.move(start_pos)
            self._enter_slide.start()
            self._enter_fade.start()

        QtCore.QTimer.singleShot(0, _start)

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
        """Adjust `(x, y)` so the visible panel stays inside the main window.

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
