"""
QATCH.ui.widgets.account_popup.py

Account toolbar popup widgets and role-based avatar styling.

Provides the custom Qt widgets and supporting helpers used to display the
application's account dropdown. The account interface includes a role-aware
avatar, session information, role badge, and context-sensitive account
actions such as user preferences, user management, and sign-out.

The popup uses the application's flat surface styling and theme token system
to remain visually consistent across light and dark themes. Its inner panel
is custom painted with rounded geometry and a themed border, while a
drop-shadow effect provides separation from the underlying application
without relying on the operating system's rectangular popup shadow.

The module also provides role-specific color palettes for account avatars and
badges, entrance fade/slide animations, popup anchoring and boundary
clamping, and automatic dismissal when the associated main window moves,
resizes, or changes window state.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-21
"""

import os

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.components.icon_utils import tinted_icon
from QATCH.ui.components.qatch_push_button import QATCHPushButton
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css
from QATCH.ui.styles.typography import FONT_SANS_STACK


class AvatarLabel(QtWidgets.QWidget):
    """Display a circular user avatar using role-based colors and initials.

    The avatar renders the user's initials centered within a circular,
    role-colored background. The background includes a radial gradient,
    border, and subtle shimmer effect. Colors are resolved from the active
    theme and the currently assigned user role.

    Args:
        initials: Text used to identify the user. At most the first two
            characters are displayed in uppercase. If empty, `"?"` is
            displayed.
        parent: Optional parent widget.

    Attributes:
        _initials: The uppercase initials displayed in the avatar, limited to
            two characters.
        _role_name: Name of the user's current role, used to select the
            avatar's color palette.
    """

    def __init__(self, initials: str, parent=None) -> None:
        """Initialize the avatar label.

        Args:
            initials: Text used to derive the displayed user initials.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._initials = initials[:2].upper() if initials else "?"
        self._role_name = "NONE"
        self.setAutoFillBackground(False)
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def set_role_name(self, role_name: str) -> None:
        """Set the user's role and refresh the avatar appearance.

        Args:
            role_name: Name of the role whose color palette should be used
                when rendering the avatar.
        """
        self._role_name = role_name
        self.update()

    def _on_theme_changed(self, _mode: str) -> None:
        """Refresh the avatar when the application theme changes.

        Args:
            _mode: Theme mode identifier emitted by `ThemeManager`. The
                value is not used directly because the current theme tokens
                are queried during painting.
        """
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the circular avatar and its initials.

        The avatar is rendered using the current theme's text color and the
        color palette associated with the user's role. A radial gradient forms
        the avatar background, while a translucent linear gradient provides a
        subtle shimmer effect across the upper portion.

        Args:
            event: Qt paint event describing the region that requires
                repainting.
        """
        tok = ThemeManager.instance().tokens()
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing)
        r = min(self.width(), self.height()) - 2
        x = (self.width() - r) / 2
        y = (self.height() - r) / 2
        rect = QtCore.QRectF(x, y, r, r)

        grad = QtGui.QRadialGradient(rect.center(), r / 2)
        role_colors = _role_colors(self._role_name)
        grad.setColorAt(0.0, QtGui.QColor(*role_colors["avatar_start"]))
        grad.setColorAt(1.0, QtGui.QColor(*role_colors["avatar_end"]))
        p.setBrush(QtGui.QBrush(grad))
        p.setPen(QtGui.QPen(QtGui.QColor(*role_colors["border"]), 1.5))
        p.drawEllipse(rect)

        # Shimmer half-circle
        shimmer_rgb = role_colors["shimmer"][:3]
        shimmer = QtGui.QLinearGradient(0, float(rect.top()), 0, float(rect.center().y()))
        shimmer.setColorAt(0.0, QtGui.QColor(*role_colors["shimmer"]))
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


def _role_colors(role_name: str) -> dict:
    """Return the color palette associated with a user role.

    Role matching is case-insensitive and is based on whether the supplied
    role name contains a recognized role identifier. Each palette provides
    colors for role indicators, borders, text, avatar gradients, and the
    avatar shimmer effect.

    Args:
        role_name: User role name used to select the appropriate color
            palette. Recognized roles include `ADMIN`, `OPERATE`,
            `CAPTURE`, and `ANALYZE`. Unrecognized or empty role names
            use the default neutral palette.

    Returns:
        A dictionary containing RGBA color tuples for the selected role.
        The dictionary contains the following keys:

        * `bg`: Semi-transparent background color.
        * `border`: Semi-transparent border color.
        * `text`: Role text color.
        * `avatar_start`: Opaque starting color for the avatar gradient.
        * `avatar_end`: Opaque ending color for the avatar gradient.
        * `shimmer`: Semi-transparent white color used for the avatar
          shimmer effect.
    """
    role_upper = str(role_name).upper()
    if "ADMIN" in role_upper:
        base = (220, 53, 69)
        text = (200, 35, 51, 255)
    elif "OPERATE" in role_upper:
        base = (40, 167, 69)
        text = (30, 126, 52, 255)
    elif "CAPTURE" in role_upper:
        base = (255, 193, 7)
        text = (179, 134, 0, 255)
    elif "ANALYZE" in role_upper:
        base = (111, 66, 193)
        text = (111, 66, 193, 255)
    else:
        base = (108, 117, 125)
        text = (73, 80, 87, 255)

    r, g, b = base
    return {
        "bg": (r, g, b, 31),
        "border": (r, g, b, 115),
        "text": text,
        "avatar_start": (min(255, r + 32), min(255, g + 32), min(255, b + 32), 255),
        "avatar_end": (r, g, b, 255),
        "shimmer": (255, 255, 255, 80),
    }


class AccountInnerPanel(QtWidgets.QWidget):
    """Inner panel used by the account popup.

    Provides the painted surface for the account popup using the application's
    flat control-system styling. The panel renders as a rounded card with a
    one-pixel themed border rather than using the recipe
    used by other application panels.

    The containing :class:`AccountPopup` applies a
    :class:`QGraphicsDropShadowEffect` to this widget. Because the panel is
    custom-painted with a transparent background and rounded geometry, the
    resulting shadow follows the painted alpha mask instead of producing a
    rectangular operating-system popup outline.

    The panel only repaints when the application theme changes, making the
    custom-painted surface safe for use with the shadow effect without the
    repaint overhead associated with hover-animated widgets.

    Attributes:
        _RADIUS: Corner radius, in pixels, used when painting the panel
            surface.
    """

    _RADIUS: float = 12.0

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the account popup's inner panel.

        Args:
            parent: Optional parent widget that owns this panel.
        """
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        """Refresh the panel when the application theme changes.

        Args:
            _mode: Theme mode identifier emitted by `ThemeManager`. The
                value is not used directly because the current theme tokens
                are retrieved during painting.
        """
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the themed flat panel surface.

        The surface fill and border colors are retrieved from the active
        theme and rendered using the shared `paint_flat_surface` recipe.
        Antialiasing and smooth pixmap transformation are enabled to preserve
        smooth rounded corners and surface edges.

        Args:
            event: Qt paint event describing the region that requires
                repainting.
        """
        tok = ThemeManager.instance().tokens()
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        paint_flat_surface(
            self,
            radius=self._RADIUS,
            fill=QtGui.QColor(*tok["flat_surface"]),
            border=QtGui.QColor(*tok["flat_border"]),
            painter=p,
        )
        p.end()


class AccountPopup(QtWidgets.QWidget):
    """Display the account dropdown for the currently active user.

    The popup presents the active user's avatar, name, initials, role, and
    session status, along with context-appropriate account actions. Signed-in
    users can access user preferences and sign out, while administrators also
    receive a shortcut for managing users.

    The widget uses `Qt.Popup` with a frameless, translucent outer container
    so that it closes automatically when the user clicks outside the popup.
    The outer widget reserves space around an :class:`AccountInnerPanel` for
    its drop shadow. The shadow is applied directly to the inner panel so its
    shape follows the panel's rounded painted surface rather than producing a
    rectangular operating-system window shadow.

    The popup also supports animated entrance behavior using a fade and
    positional slide animation. When anchored to the application's main
    window, the popup tracks that window and can close itself if the main
    window is moved or resized.

    Callback arguments are invoked when the corresponding account actions are
    selected. Session information is resolved during initialization using a
    lazy import of the user-profile subsystem to avoid circular dependencies.

    Signals:
        closed: Emitted when the popup closes.

    Attributes:
        _SHADOW_MARGIN_L: Horizontal space reserved on the left for the panel
            drop shadow.
        _SHADOW_MARGIN_T: Vertical space reserved above the panel for the drop
            shadow.
        _SHADOW_MARGIN_R: Horizontal space reserved on the right for the panel
            drop shadow.
        _SHADOW_MARGIN_B: Vertical space reserved below the panel for the drop
            shadow and its positive Y offset.
        _open_manager_cb: Optional callback invoked when the Manage Users
            action is selected.
        _open_preferences_cb: Optional callback invoked when the User
            Preferences action is selected.
        _sign_out_cb: Optional callback invoked when the Sign Out action is
            selected.
        _main_window: Main application window associated with the popup after
            it is anchored.
        _panel: Inner painted account panel containing all visible content.
        _role_name: Role name associated with the current session.
        _name_lbl: Label displaying the current user's full name.
        _initials_lbl: Label displaying the user's initials.
        _role_badge: Label displaying the user's role.
        _last_lbl: Optional label displaying the user's last access time.
        _status_lbl: Optional label displayed when there is no active session.
        _divider: Optional horizontal divider separating account information
            from available actions.
        _preferences_btn: Optional button for opening user preferences.
        _manage_btn: Optional button for opening user management.
        _sign_out_btn: Optional button for signing out.
        _enter_fade: Animation controlling the popup's entrance opacity.
        _enter_slide: Animation controlling the popup's entrance position.

    Args:
        open_manager_cb: Optional callback invoked when the user selects
            `Manage Users`.
        open_preferences_cb: Optional callback invoked when the user selects
            `User Preferences`.
        sign_out_cb: Optional callback invoked when the user selects
            `Sign Out`.
        parent: Optional parent widget.
    """

    closed = QtCore.pyqtSignal()

    # Margins reserved around the inner panel for the drop shadow.  Bottom is
    # larger to accommodate the shadow's positive Y offset.
    _SHADOW_MARGIN_L = 22
    _SHADOW_MARGIN_T = 18
    _SHADOW_MARGIN_R = 22
    _SHADOW_MARGIN_B = 26

    def __init__(
        self,
        open_manager_cb=None,
        open_preferences_cb=None,
        sign_out_cb=None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the account popup and populate its session content.

        The constructor creates the transparent popup container, configures
        the inner painted panel and drop shadow, initializes the entrance
        animations, resolves the current user session, and creates the
        account information and action controls appropriate for that session.

        Args:
            open_manager_cb: Optional callback invoked when `Manage Users`
                is selected.
            open_preferences_cb: Optional callback invoked when
                `User Preferences` is selected.
            sign_out_cb: Optional callback invoked when `Sign Out` is
                selected.
            parent: Optional parent widget.
        """
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
        self._open_preferences_cb = open_preferences_cb
        self._sign_out_cb = sign_out_cb
        self._main_window: QtWidgets.QWidget | None = None  # set by show_anchored_to

        # Outer container with shadow margins
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

        # Drop shadow that follows the inner panel's painted alpha mask
        shadow = QtWidgets.QGraphicsDropShadowEffect(self._panel)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 4)
        shadow.setColor(QtGui.QColor(0, 20, 40, 110))
        self._panel.setGraphicsEffect(shadow)

        # Entrance animation
        self._enter_fade = QtCore.QVariantAnimation(self)
        self._enter_fade.setDuration(200)
        self._enter_fade.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._enter_fade.setStartValue(0.0)
        self._enter_fade.setEndValue(1.0)
        self._enter_fade.valueChanged.connect(lambda v: self.setWindowOpacity(float(v)))

        self._enter_slide = QtCore.QPropertyAnimation(self, b"pos", self)
        self._enter_slide.setDuration(220)
        self._enter_slide.setEasingCurve(QtCore.QEasingCurve.OutCubic)

        # Resolve current session info
        # session_info() returns: [name, initials, role.name, created, modified, accessed]
        accessed: str | None = None
        try:
            from QATCH.common.userProfiles import (  # noqa: PLC0415
                UserProfiles,
                UserRoles,
            )

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

        # Inner panel layout (all visible content lives here)
        layout = QtWidgets.QVBoxLayout(self._panel)
        layout.setContentsMargins(14, 14, 14, 12)
        layout.setSpacing(8)

        # Avatar, name/role column
        header_row = QtWidgets.QHBoxLayout()
        header_row.setSpacing(12)

        avatar = AvatarLabel(initials)
        avatar.set_role_name(role_name)
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
        role_row = QtWidgets.QHBoxLayout()
        role_row.setContentsMargins(0, 2, 0, 0)
        role_row.setSpacing(0)
        role_row.addWidget(self._role_badge)
        role_row.addStretch()
        info_col.addLayout(role_row)

        header_row.addLayout(info_col, 1)
        layout.addLayout(header_row)

        # Last sign-in / status line
        self._last_lbl: QtWidgets.QLabel | None = None
        self._status_lbl: QtWidgets.QLabel | None = None
        if is_signed_in and accessed:
            self._last_lbl = QtWidgets.QLabel(f"Last access: {accessed}")
            layout.addWidget(self._last_lbl)
        elif not is_signed_in:
            self._status_lbl = QtWidgets.QLabel("No active session")
            self._status_lbl.setStyleSheet("font-style: italic;")
            layout.addWidget(self._status_lbl)

        show_preferences = is_signed_in
        show_manage = is_admin
        show_sign_out = is_signed_in
        self._divider: QtWidgets.QFrame | None = None
        if show_preferences or show_manage or show_sign_out:
            self._divider = QtWidgets.QFrame()
            self._divider.setFrameShape(QtWidgets.QFrame.HLine)
            layout.addWidget(self._divider)

        self._preferences_btn: QATCHPushButton | None = None
        if show_preferences:
            self._preferences_btn = QATCHPushButton("User Preferences", variant="ghost")
            self._preferences_btn.set_menu_row(True)
            self._preferences_btn.setFixedHeight(34)
            self._preferences_btn.setIconSize(QtCore.QSize(16, 16))
            self._preferences_btn.clicked.connect(self._on_preferences)
            layout.addWidget(self._preferences_btn)

        self._manage_btn: QATCHPushButton | None = None
        if show_manage:
            self._manage_btn = QATCHPushButton("Manage Users", variant="ghost")
            self._manage_btn.set_menu_row(True)
            self._manage_btn.setFixedHeight(34)
            self._manage_btn.setIconSize(QtCore.QSize(16, 16))
            self._manage_btn.clicked.connect(self._on_manage_users)
            layout.addWidget(self._manage_btn)

        self._sign_out_btn: QATCHPushButton | None = None
        if show_sign_out:
            self._sign_out_btn = QATCHPushButton("Sign Out", variant="ghost_danger")
            self._sign_out_btn.set_menu_row(True)
            self._sign_out_btn.setFixedHeight(34)
            self._sign_out_btn.setIconSize(QtCore.QSize(16, 16))
            self._sign_out_btn.clicked.connect(self._on_sign_out)
            layout.addWidget(self._sign_out_btn)

        self._panel.setMinimumWidth(230)

        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        """Refresh the popup styling when the application theme changes.

        Args:
            _mode: Theme mode identifier emitted by `ThemeManager`. The value
                is not used directly because the current theme tokens are
                retrieved by :meth:`_apply_theme`.
        """
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Apply the current theme styling to the account popup controls.

        Updates labels, the role badge, session status indicators, divider, and
        action-button icons using the active theme tokens. Role-specific colors
        are applied to the role badge, while action icons use the theme's accent
        or error color as appropriate.

        Missing icon files are ignored, allowing the corresponding buttons to
        remain functional without an icon.
        """
        tok = ThemeManager.instance().tokens()

        self._name_lbl.setStyleSheet(
            f"color: {tok_css(tok['flat_text'])}; font-family: {FONT_SANS_STACK}; "
            "font-size: 13px; font-weight: 600; background: transparent; border: none;"
        )
        self._initials_lbl.setStyleSheet(
            f"color: {tok_css(tok['flat_text_muted'])}; font-family: {FONT_SANS_STACK}; "
            "font-size: 10px; background: transparent; border: none;"
        )

        role_colors = _role_colors(self._role_name)
        self._role_badge.setStyleSheet(
            f"background: {tok_css(role_colors['bg'])}; color: {tok_css(role_colors['text'])}; "
            f"font-family: {FONT_SANS_STACK}; font-weight: 600; border-radius: 4px; padding: 1px 7px; "
            f"font-size: 10px; border: 1px solid {tok_css(role_colors['border'])};"
        )

        if self._last_lbl is not None:
            self._last_lbl.setStyleSheet(
                f"color: {tok_css(tok['flat_text_muted'])}; font-family: {FONT_SANS_STACK}; "
                "font-size: 10px; background: transparent; border: none; padding-left: 1px;"
            )
        if self._status_lbl is not None:
            self._status_lbl.setStyleSheet(
                f"color: {tok_css(tok['flat_error'])}; font-family: {FONT_SANS_STACK}; "
                "font-size: 10px; font-style: italic; background: transparent; "
                "border: none; padding-left: 1px;"
            )
        if self._divider is not None:
            self._divider.setStyleSheet(
                f"QFrame {{ background: {tok_css(tok['flat_border'])}; "
                "border: none; max-height: 1px; }"
            )

        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")

        if self._preferences_btn is not None:
            icon_path = os.path.join(icons_dir, "preferences.svg")
            if os.path.exists(icon_path):
                self._preferences_btn.setIcon(
                    tinted_icon(icon_path, QtGui.QColor(*tok["flat_accent"][:3]))
                )

        if self._manage_btn is not None:
            icon_path = os.path.join(icons_dir, "manage_users.svg")
            if os.path.exists(icon_path):
                self._manage_btn.setIcon(
                    tinted_icon(icon_path, QtGui.QColor(*tok["flat_accent"][:3]))
                )

        if self._sign_out_btn is not None:
            icon_path = os.path.join(icons_dir, "sign-out.svg")
            if os.path.exists(icon_path):
                self._sign_out_btn.setIcon(
                    tinted_icon(icon_path, QtGui.QColor(*tok["flat_error"][:3]))
                )

    def show_anchored_to(
        self,
        anchor: QtWidgets.QWidget,
        main_window: QtWidgets.QWidget | None = None,
    ) -> None:
        """Show the popup anchored to a widget and constrained to a main window.

        The visible right edge of the popup aligns with the right edge of
        `anchor`, and the visible top edge is positioned 2 pixels below the
        anchor's bottom edge. The popup position is clamped when necessary so the
        visible panel remains within the bounds of `main_window`.

        The main window is monitored while the popup is open. If it is moved or
        resized, the popup can close itself rather than remain detached from the
        application window.

        Args:
            anchor: Widget to which the popup should be anchored.
            main_window: Optional main application window used to constrain the
                popup position and monitor subsequent window movement or resizing.
        """
        self._main_window = main_window
        self.adjustSize()

        size = self.sizeHint()
        popup_w, popup_h = size.width(), size.height()

        # Anchor at the bottom-right corner of the button (in screen coords)
        anchor_br = anchor.mapToGlobal(QtCore.QPoint(anchor.width(), anchor.height()))
        x = anchor_br.x() + self._SHADOW_MARGIN_R - popup_w
        y = anchor_br.y() + 2 - self._SHADOW_MARGIN_T

        # Clamp so the visible panel stays inside the main window
        x, y = self._clamp_to_main_window(x, y, popup_w, popup_h, anchor)

        # Track resize/move events on the main window.
        if self._main_window is not None:
            self._main_window.installEventFilter(self)

        # Entrance animation
        final_pos = QtCore.QPoint(x, y)
        start_pos = QtCore.QPoint(x, y - 12)
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

    def _visible_rect_for(self, x: int, y: int, w: int, h: int) -> QtCore.QRect:
        """Return the visible panel rectangle for an outer-widget position.

        The popup's outer widget includes transparent margins reserved for the
        drop shadow. This method converts the outer widget geometry into the
        rectangle occupied by the visible inner panel by removing those margins.

        Args:
            x: X-coordinate of the outer popup widget in screen coordinates.
            y: Y-coordinate of the outer popup widget in screen coordinates.
            w: Width of the outer popup widget.
            h: Height of the outer popup widget.

        Returns:
            A `QRect` describing the visible inner panel's geometry in screen
            coordinates.
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
        """Adjust the popup position so its visible panel remains in bounds.

        The anchor widget's top-level window is preferred as the bounding
        rectangle, followed by the configured main window and finally the
        available geometry of the screen containing the anchor. Horizontal
        overflow is corrected by shifting the popup into the bounds. If the
        popup would extend below the window, it is flipped above the anchor when
        there is sufficient space; otherwise, its position is clamped to the
        bottom edge.

        Args:
            x: Proposed X-coordinate of the outer popup widget.
            y: Proposed Y-coordinate of the outer popup widget.
            popup_w: Width of the outer popup widget.
            popup_h: Height of the outer popup widget.
            anchor: Widget to which the popup is anchored and whose top-level
                window is preferred for determining the available bounds.

        Returns:
            A `(x, y)` tuple containing the adjusted popup position.
        """
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

        # Vertical clamp
        if visible.bottom() > bounds.bottom():
            anchor_top = anchor.mapToGlobal(QtCore.QPoint(0, 0)).y()
            y_above = anchor_top - 2 - popup_h + self._SHADOW_MARGIN_B
            visible_above = self._visible_rect_for(x, y_above, popup_w, popup_h)
            if visible_above.top() >= bounds.top():
                y = y_above
            else:
                # Neither orientation fits, just clamp to the bottom edge
                y -= visible.bottom() - bounds.bottom()

        return x, y

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Close the popup when its associated main window changes geometry.

        The popup is positioned in screen coordinates when shown and is not
        continuously re-anchored while the main window is being laid out. Closing
        the popup on resize, move, or window-state changes avoids race conditions
        with the layout engine and ensures the popup does not become detached from
        its anchor.

        Args:
            watched: Object whose event is being filtered.
            event: Event being processed by the filter.

        Returns:
            The result of the base class event-filter implementation.
        """
        if watched is self._main_window and event.type() in (
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.WindowStateChange,
        ):
            self.close()
        return super().eventFilter(watched, event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Clean up the main-window event filter and emit the close signal.

        Removes the popup from the associated main window's event-filter chain
        before completing the normal Qt close-event handling. The
        :attr:`closed` signal is emitted after the base implementation completes.

        Args:
            event: Qt close event generated when the popup is being closed.
        """
        if self._main_window is not None:
            try:
                self._main_window.removeEventFilter(self)
            except Exception:
                pass
            self._main_window = None
        super().closeEvent(event)
        self.closed.emit()

    def _on_manage_users(self) -> None:
        """Close the popup and invoke the user-management callback.

        The popup is closed before invoking the callback so that the account menu
        does not remain visible while the user-management interface is opened.

        If no management callback was provided, the method only closes the popup.
        """
        self.close()
        if self._open_manager_cb:
            self._open_manager_cb()

    def _on_preferences(self) -> None:
        """Close the popup and invoke the user-preferences callback.

        The popup is closed before invoking the callback so that the account menu
        does not remain visible while the preferences interface is opened.

        If no preferences callback was provided, the method only closes the popup.
        """
        self.close()
        if self._open_preferences_cb:
            self._open_preferences_cb()

    def _on_sign_out(self) -> None:
        """Close the popup and invoke the sign-out callback.

        The popup is closed before invoking the callback so that the account menu
        is dismissed before the active session is changed.

        If no sign-out callback was provided, the method only closes the popup.
        """
        self.close()
        if self._sign_out_cb:
            self._sign_out_cb()
