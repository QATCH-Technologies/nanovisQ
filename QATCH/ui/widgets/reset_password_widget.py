"""
QATCH.ui.widgets.reset_password_widget.py

Overlay widget for resetting a user password.

Provides the `ResetPasswordWidget` overlay used by administrators to
reset an existing user's password. The interface follows the visual and
interaction conventions established by the user-creation workflow,
including the centered glass card, translucent scrim, animated
open/close transitions, inline validation feedback, close control, and
arrow-based submit control.

The overlay emits the validated plaintext password through
`password_confirmed` immediately before beginning its close animation.

Typical usage::

    overlay = ResetPasswordWidget(
        name="Jane Smith",
        initials="JS",
        role="ADMIN",
        parent=self,
    )
    overlay.resize(self.size())
    overlay.show()
    overlay.raise_()
    overlay.password_confirmed.connect(
        lambda pwd: self._update_user_xml(filename, new_pwd_plain=pwd)
    )

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-21
"""

from __future__ import annotations

import os
import re

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.ui.components.icon_utils import tinted_icon, tinted_pixmap
from QATCH.ui.components.qatch_line_edit import QATCHLineEdit
from QATCH.ui.styles.theme_manager import (
    ThemeManager,
    accent_avatar_qss,
    auth_card_qss,
    auth_separator_qss,
    auth_shadow_color,
    card_title_qss,
    close_button_qss,
    dim_badge_qss,
    dim_field_qss,
    dim_text_qss,
    error_label_qss,
    gradient_button_qss,
    info_wash_card_qss,
    role_badge_qss,
    tok_css,
)

_INPUT_H: int = 34
_CARD_W: int = 420


class ResetPasswordWidget(QtWidgets.QWidget):
    """Display a full-screen overlay for resetting an existing user's password.

    Presents an administrative password-reset interface over the parent
    user-profile panel. The overlay displays the target user's profile
    information, disabled placeholder fields for backend-dependent account
    details, and live controls for entering and confirming a new password.

    The widget manages its own translucent scrim, opening animation, theme
    updates, and validation state. A validated password is emitted through
    `password_confirmed` before the overlay closes.

    Attributes:
        is_accepted (bool): Whether a valid new password has been submitted.
    """

    # Emitted with the validated plaintext password before the close animation.
    password_confirmed: QtCore.pyqtSignal = QtCore.pyqtSignal(str)

    def __init__(
        self,
        name: str,
        initials: str,
        role: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the password-reset overlay.

        Stores the target user's display information, initializes the overlay
        state and animation tracking, configures the widget for custom
        translucent background painting, and builds the reset-password UI.
        The widget also subscribes to theme changes and begins its opening
        animation.

        Args:
            name (str): Display name of the user whose password is being reset.
            initials (str): User initials displayed in the profile card.
            role (str): User role displayed in the profile card.
            parent (QtWidgets.QWidget, optional): Parent widget over which the
                reset-password overlay is displayed. When provided, the overlay
                tracks the parent's size and event lifecycle. Defaults to
                `None`.
        """
        super().__init__(parent)

        self._name = name
        self._initials = initials
        self._role = role

        self.is_accepted: bool = False
        self._shake_anims: list[QtCore.QPropertyAnimation] = []
        self._bg_alpha: int = 0
        self._future_fields: list = []
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)

        if parent is not None:
            parent.installEventFilter(self)
            self.resize(parent.size())

        self._setup_ui()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)
        self.raise_()
        self._animate_open()

    def _on_theme_changed(self, _mode: str) -> None:
        """Refresh the widget styling after a theme change.

        Reapplies all theme-dependent styles and icons so the password-reset
        overlay immediately reflects the newly selected application theme.

        Args:
            _mode (str): Theme mode reported by `ThemeManager`. The value is
                not used directly because the active theme is retrieved from
                `ThemeManager` when styles are reapplied.
        """
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Apply the active theme to all password-reset overlay elements.

        Refreshes the glass card, shadow, buttons, title, profile information,
        disabled placeholder fields, password visibility icons, validation
        message, and submit icon using the current theme tokens and shared
        application styling helpers.

        This method is intended to be called both during initial widget setup
        and when `ThemeManager.themeChanged` signals a live theme change.
        """
        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")
        tok = ThemeManager.instance().tokens()

        self.glass_frame.setStyleSheet(auth_card_qss("resetPwdView"))
        self._shadow.setColor(auth_shadow_color())
        self.btn_close.setStyleSheet(close_button_qss())

        icon_color = QtGui.QColor(*tok["flat_text"])
        icon_pm = QtGui.QPixmap(os.path.join(icons_dir, "reset-password.svg"))
        if not icon_pm.isNull():
            self._icon_lbl.setPixmap(
                tinted_pixmap(os.path.join(icons_dir, "reset-password.svg"), icon_color, size=48)
            )
        self._lbl_title.setStyleSheet(card_title_qss())

        self._info_card.setStyleSheet(info_wash_card_qss())
        self._avatar.setStyleSheet(accent_avatar_qss())
        self._name_lbl.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_text'])}; font-size: 11pt; "
            "font-weight: 600; background: transparent; }"
        )
        self._role_badge.setStyleSheet(role_badge_qss(self._role))

        for frame, field_lbl, placeholder_lbl, soon_badge in self._future_fields:
            frame.setStyleSheet(dim_field_qss())
            field_lbl.setStyleSheet(dim_text_qss())
            placeholder_lbl.setStyleSheet(dim_text_qss(italic=True))
            soon_badge.setStyleSheet(dim_badge_qss())

        eye_color = QtGui.QColor(*tok["flat_text_muted"])
        self._eye_on = tinted_icon(os.path.join(icons_dir, "eye-on.svg"), eye_color)
        self._eye_off = tinted_icon(os.path.join(icons_dir, "eye-off.svg"), eye_color)
        self._act_eye1.setIcon(self._eye_off if self._pwd1_visible else self._eye_on)
        self._act_eye2.setIcon(self._eye_off if self._pwd2_visible else self._eye_on)

        self.err_password.setStyleSheet(error_label_qss())

        submit_icon_color = QtGui.QColor(*tok["flat_on_accent"])
        self.btn_submit.setIcon(
            tinted_icon(os.path.join(icons_dir, "right-arrow.svg"), submit_icon_color, size=20)
        )
        self.btn_submit.setStyleSheet(gradient_button_qss())

    def _setup_ui(self) -> None:
        """Build and configure the password-reset overlay interface.

        Constructs the centered card and all of its child controls,
        including the close button, reset-password header, target-user
        information card, disabled future account fields, password inputs,
        validation message, and submit button.

        Initializes password visibility controls and connects the relevant
        input and button signals to their handlers. The resulting glass card
        is added to the overlay's centered base layout.
        """
        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")

        # Outer centred layout
        self.base_layout = QtWidgets.QVBoxLayout(self)
        self.base_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.glass_frame = QtWidgets.QFrame(self)
        self.glass_frame.setObjectName("resetPwdView")
        self.glass_frame.setFixedWidth(_CARD_W)
        self.glass_frame.setStyleSheet(auth_card_qss("resetPwdView"))

        self._shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(44)
        self._shadow.setColor(auth_shadow_color())
        self._shadow.setOffset(0, 10)
        self.glass_frame.setGraphicsEffect(self._shadow)

        self.main_layout = QtWidgets.QVBoxLayout(self.glass_frame)
        self.main_layout.setContentsMargins(28, 14, 28, 28)
        self.main_layout.setSpacing(6)

        # close button
        close_row = QtWidgets.QHBoxLayout()
        close_row.setContentsMargins(0, 0, 0, 0)
        close_row.addStretch()

        self.btn_close = QtWidgets.QPushButton("x")
        self.btn_close.setFixedSize(28, 28)
        self.btn_close.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_close.setStyleSheet(close_button_qss())
        self.btn_close.clicked.connect(self._reject)
        close_row.addWidget(self.btn_close)
        self.main_layout.addLayout(close_row)

        # Header icon
        icon_lbl = QtWidgets.QLabel()
        icon_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        icon_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        icon_pm = QtGui.QPixmap(os.path.join(icons_dir, "reset-password.svg"))
        if not icon_pm.isNull():
            icon_color = QtGui.QColor(*ThemeManager.instance().tokens()["flat_text"])
            icon_lbl.setPixmap(
                tinted_pixmap(os.path.join(icons_dir, "reset-password.svg"), icon_color, size=48)
            )
        else:
            icon_lbl.setStyleSheet("font-size: 30px; background: transparent;")
        icon_lbl.setFixedHeight(52)
        self._icon_lbl = icon_lbl
        self.main_layout.addWidget(icon_lbl, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Title
        lbl_title = QtWidgets.QLabel("Reset Password")
        lbl_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_title.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        lbl_title.setStyleSheet(card_title_qss())
        self._lbl_title = lbl_title
        self.main_layout.addWidget(lbl_title)
        self.main_layout.addSpacing(12)

        # User info card
        self.main_layout.addWidget(self._build_user_info_card())
        self.main_layout.addSpacing(10)

        # Separator
        self.main_layout.addWidget(self._make_separator())
        self.main_layout.addSpacing(10)

        # Placeholder fields
        # TODO: Implement backend support
        self.main_layout.addWidget(self._make_future_field("Email", "example@domain.com"))
        self.main_layout.addSpacing(5)
        self.main_layout.addWidget(self._make_future_field("Username", "optional username"))
        self.main_layout.addSpacing(10)

        # Separator
        self.main_layout.addWidget(self._make_separator())
        self.main_layout.addSpacing(10)

        # Password inputs
        eye_color = QtGui.QColor(*ThemeManager.instance().tokens()["flat_text_muted"])
        self._eye_on = tinted_icon(os.path.join(icons_dir, "eye-on.svg"), eye_color)
        self._eye_off = tinted_icon(os.path.join(icons_dir, "eye-off.svg"), eye_color)
        self._pwd1_visible = False
        self._pwd2_visible = False

        self.inp_pwd1 = QATCHLineEdit()
        self.inp_pwd1.setFixedHeight(_INPUT_H)
        self.inp_pwd1.setPlaceholderText("New Password")
        self.inp_pwd1.setEchoMode(QtWidgets.QLineEdit.Password)
        self.inp_pwd1.textChanged.connect(
            lambda _: self._clear_field_errors([self.inp_pwd1, self.inp_pwd2], self.err_password)
        )
        self._act_eye1 = self.inp_pwd1.addAction(self._eye_on, QtWidgets.QLineEdit.TrailingPosition)
        self._act_eye1.triggered.connect(self._toggle_pwd1)
        self.main_layout.addWidget(self.inp_pwd1)

        self.inp_pwd2 = QATCHLineEdit()
        self.inp_pwd2.setFixedHeight(_INPUT_H)
        self.inp_pwd2.setPlaceholderText("Confirm New Password")
        self.inp_pwd2.setEchoMode(QtWidgets.QLineEdit.Password)
        self.inp_pwd2.textChanged.connect(
            lambda _: self._clear_field_errors([self.inp_pwd1, self.inp_pwd2], self.err_password)
        )
        self._act_eye2 = self.inp_pwd2.addAction(self._eye_on, QtWidgets.QLineEdit.TrailingPosition)
        self._act_eye2.triggered.connect(self._toggle_pwd2)
        self.main_layout.addWidget(self.inp_pwd2)

        self.err_password = self._make_error_label()
        self.main_layout.addWidget(self.err_password)

        self.main_layout.addSpacing(10)

        # Submit button
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.btn_submit = QtWidgets.QPushButton("")
        submit_icon_color = QtGui.QColor(*ThemeManager.instance().tokens()["flat_on_accent"])
        self.btn_submit.setIcon(
            tinted_icon(os.path.join(icons_dir, "right-arrow.svg"), submit_icon_color, size=20)
        )
        self.btn_submit.setIconSize(QtCore.QSize(20, 20))
        self.btn_submit.setFixedSize(40, 40)
        self.btn_submit.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_submit.setToolTip("Confirm password reset")
        self.btn_submit.setStyleSheet(gradient_button_qss())
        self.btn_submit.clicked.connect(self._validate_and_accept)

        btn_row.addWidget(self.btn_submit)
        self.main_layout.addLayout(btn_row)

        self.base_layout.addWidget(self.glass_frame)

    def _build_user_info_card(self) -> QtWidgets.QFrame:
        """Build a compact profile card for the target user.

        Creates a themed information card containing the user's initials in an
        avatar, followed by their display name and role badge. The card styling
        and its child widgets are stored on the instance so they can be
        refreshed when the application theme changes.

        Returns:
            QtWidgets.QFrame: The constructed user information card.
        """
        card = QtWidgets.QFrame()
        card.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        card.setStyleSheet(info_wash_card_qss())
        self._info_card = card

        lay = QtWidgets.QHBoxLayout(card)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(14)

        # Initials avatar
        avatar = QtWidgets.QLabel(self._initials)
        avatar.setFixedSize(46, 46)
        avatar.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        avatar.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        avatar.setStyleSheet(accent_avatar_qss())
        self._avatar = avatar

        # Name, role column
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(5)
        col.setContentsMargins(0, 0, 0, 0)

        name_lbl = QtWidgets.QLabel(self._name)
        name_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        name_lbl.setStyleSheet(
            f"QLabel {{ color: {tok_css(ThemeManager.instance().tokens()['flat_text'])}; "
            "font-size: 11pt; font-weight: 600; background: transparent; }"
        )
        self._name_lbl = name_lbl

        self._role_badge = self._make_role_badge(self._role)
        col.addWidget(name_lbl)
        col.addWidget(self._role_badge)

        lay.addWidget(avatar)
        lay.addLayout(col)
        lay.addStretch()

        return card

    def _make_role_badge(self, role_name: str) -> QtWidgets.QLabel:
        """Create a themed role badge for the target user.

        Creates a compact label displaying the supplied role name and applies
        the role-specific badge styling used by the user profile interface.

        Args:
            role_name (str): Name of the user's role to display.

        Returns:
            QtWidgets.QLabel: The styled role badge label.
        """
        badge = QtWidgets.QLabel(role_name)
        badge.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        badge.setStyleSheet(role_badge_qss(role_name))
        return badge

    def _make_future_field(self, label_text: str, placeholder: str) -> QtWidgets.QFrame:
        """Create a disabled-looking placeholder field for future account data.

        Builds a non-interactive pill-shaped field that visually matches the
        active glass line-edit controls while using dimmed styling to indicate
        that the associated backend feature is not yet available.

        Args:
            label_text (str): Descriptive label identifying the future field.
            placeholder (str): Placeholder text displayed alongside the field
                label.

        Returns:
            QtWidgets.QFrame: The constructed placeholder field.
        """
        frame = QtWidgets.QFrame()
        frame.setFixedHeight(_INPUT_H)
        frame.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        frame.setStyleSheet(dim_field_qss())

        lay = QtWidgets.QHBoxLayout(frame)
        lay.setContentsMargins(15, 0, 10, 0)
        lay.setSpacing(8)

        field_lbl = QtWidgets.QLabel(label_text)
        field_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        field_lbl.setStyleSheet(dim_text_qss())

        placeholder_lbl = QtWidgets.QLabel(placeholder)
        placeholder_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        placeholder_lbl.setStyleSheet(dim_text_qss(italic=True))

        soon_badge = QtWidgets.QLabel("coming soon")
        soon_badge.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        soon_badge.setStyleSheet(dim_badge_qss())

        lay.addWidget(field_lbl)
        lay.addWidget(placeholder_lbl)
        lay.addStretch()
        lay.addWidget(soon_badge)

        self._future_fields.append((frame, field_lbl, placeholder_lbl, soon_badge))
        return frame

    @staticmethod
    def _make_separator() -> QtWidgets.QWidget:
        """Create a thin horizontal separator for the reset-password form.

        Creates a 1-pixel-high translucent widget using the shared glass
        separator styling.

        Returns:
            QtWidgets.QWidget: A themed horizontal separator.
        """
        sep = QtWidgets.QWidget()
        sep.setFixedHeight(1)
        sep.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        sep.setStyleSheet(auth_separator_qss())
        return sep

    @staticmethod
    def _make_error_label() -> QtWidgets.QLabel:
        """Create a hidden inline error message label.

        Creates a word-wrapped label using the shared error styling. The label
        is initially hidden and can be shown when password validation fails.

        Returns:
            QtWidgets.QLabel: A hidden, styled error message label.
        """
        lbl = QtWidgets.QLabel("")
        lbl.setWordWrap(True)
        lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        lbl.setStyleSheet(error_label_qss())
        lbl.setVisible(False)
        return lbl

    def _clear_field_errors(
        self,
        fields: list,
        error_label: QtWidgets.QLabel,
    ) -> None:
        """Clear validation errors from the specified password fields.

        Resets the error state of any `QATCHLineEdit` instances in the
        supplied field collection and hides the associated error message.

        Args:
            fields (list): Fields whose validation error state should be
                cleared.
            error_label (QtWidgets.QLabel): Error message label to hide.
        """
        for f in fields:
            if isinstance(f, QATCHLineEdit):
                f.set_error(False)
        error_label.setVisible(False)

    def _show_field_error(
        self,
        fields: list,
        error_label: QtWidgets.QLabel,
        message: str,
        shake_target: QtWidgets.QWidget | None = None,
    ) -> None:
        """Display a validation error for the specified fields.

        Marks applicable `QATCHLineEdit` widgets as invalid, displays the
        supplied error message, and provides visual feedback by shaking the
        specified widget or, when no target is provided, the first field.

        Args:
            fields (list): Fields to mark as having validation errors.
            error_label (QtWidgets.QLabel): Label used to display the error
                message.
            message (str): Validation message to display.
            shake_target (QtWidgets.QWidget, optional): Widget to animate as
                error feedback. If omitted, the first field is used when
                available. Defaults to `None`.
        """
        for f in fields:
            if isinstance(f, QATCHLineEdit):
                f.set_error(True)
        error_label.setText(message)
        error_label.setVisible(True)
        self._shake_widget(shake_target or (fields[0] if fields else None))

    def _shake_widget(self, widget: QtWidgets.QWidget | None) -> None:
        """Animate a horizontal jiggle to provide error feedback.

        Applies a short left-to-right positional animation to the supplied
        visible widget. The animation is retained in `_shake_anims` until it
        finishes to ensure its lifetime extends through the complete
        animation.

        Args:
            widget (QtWidgets.QWidget, optional): Widget to shake. No action is
                taken when the widget is `None` or not visible.
        """
        if not widget or not widget.isVisible():
            return

        anim = QtCore.QPropertyAnimation(widget, b"pos")
        anim.setDuration(380)
        base = widget.pos()
        anim.setKeyValueAt(0.0, base)
        anim.setKeyValueAt(0.1, base + QtCore.QPoint(-6, 0))
        anim.setKeyValueAt(0.3, base + QtCore.QPoint(6, 0))
        anim.setKeyValueAt(0.5, base + QtCore.QPoint(-4, 0))
        anim.setKeyValueAt(0.7, base + QtCore.QPoint(4, 0))
        anim.setKeyValueAt(0.9, base + QtCore.QPoint(-2, 0))
        anim.setKeyValueAt(1.0, base)
        anim.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

        self._shake_anims.append(anim)
        anim.finished.connect(
            lambda: self._shake_anims.remove(anim) if anim in self._shake_anims else None
        )

    def _toggle_pwd1(self) -> None:
        """Toggle visibility of the new-password field.

        Switches the first password input between masked and plain-text echo
        modes and updates its trailing visibility icon to reflect the current
        state.
        """
        self._pwd1_visible = not self._pwd1_visible
        self.inp_pwd1.setEchoMode(
            QtWidgets.QLineEdit.Normal if self._pwd1_visible else QtWidgets.QLineEdit.Password
        )
        self._act_eye1.setIcon(self._eye_off if self._pwd1_visible else self._eye_on)

    def _toggle_pwd2(self) -> None:
        """Toggle visibility of the password-confirmation field.

        Switches the second password input between masked and plain-text echo
        modes and updates its trailing visibility icon to reflect the current
        state.
        """
        self._pwd2_visible = not self._pwd2_visible
        self.inp_pwd2.setEchoMode(
            QtWidgets.QLineEdit.Normal if self._pwd2_visible else QtWidgets.QLineEdit.Password
        )
        self._act_eye2.setIcon(self._eye_off if self._pwd2_visible else self._eye_on)

    def _validate_and_accept(self) -> None:
        """Validate the new password and accept the reset when valid.

        Clears any existing validation errors, validates the new password
        against the required minimum length and character composition, and
        verifies that the confirmation password matches. Validation failures
        are displayed through the shared field-error mechanism.

        When validation succeeds, marks the reset as accepted, emits the
        validated password through `password_confirmed`, and begins the
        overlay's close animation.
        """
        self._clear_field_errors([self.inp_pwd1, self.inp_pwd2], self.err_password)

        pwd1 = self.inp_pwd1.text()
        pwd2 = self.inp_pwd2.text()

        pwd_regex = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$"
        if not re.match(pwd_regex, pwd1):
            self._show_field_error(
                [self.inp_pwd1],
                self.err_password,
                "Password must be ≥ 8 characters with an uppercase, lowercase, and digit.",
            )
            return

        if pwd1 != pwd2:
            self._show_field_error(
                [self.inp_pwd2],
                self.err_password,
                "Passwords do not match.",
            )
            return

        self.is_accepted = True
        self.password_confirmed.emit(pwd1)
        self._close_with_animation()

    def _reject(self) -> None:
        """Reject the password reset and close the overlay.

        Marks the reset as not accepted and begins the overlay's close
        animation without emitting a confirmed password.
        """
        self.is_accepted = False
        self._close_with_animation()

    def _animate_open(self) -> None:
        """Animate the password-reset overlay into view.

        Fades in the overlay scrim while transitioning the glass card upward
        into its centered resting position using an eased animation.
        """
        self.anim_in = QtCore.QVariantAnimation(self)
        self.anim_in.setDuration(300)
        self.anim_in.setStartValue(0.0)
        self.anim_in.setEndValue(1.0)
        self.anim_in.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self.anim_in.valueChanged.connect(self._on_anim_frame)
        self.anim_in.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def _close_with_animation(self) -> None:
        """Animate the password-reset overlay out of view before closing.

        Disables interaction with the glass card, reverses the opening
        animation to fade out the scrim and move the card downward, and closes
        the widget when the animation completes.
        """
        self.glass_frame.setEnabled(False)

        self.anim_out = QtCore.QVariantAnimation(self)
        self.anim_out.setDuration(200)
        self.anim_out.setStartValue(1.0)
        self.anim_out.setEndValue(0.0)
        self.anim_out.setEasingCurve(QtCore.QEasingCurve.InQuad)
        self.anim_out.valueChanged.connect(self._on_anim_frame)
        self.anim_out.finished.connect(self.close)
        self.anim_out.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def _on_anim_frame(self, progress: float) -> None:
        """Update the overlay appearance for an animation frame.

        Uses the supplied animation progress to adjust the scrim opacity and
        vertically offset the centered card, creating the combined fade and
        slide transition used when opening and closing the overlay.

        Args:
            progress (float): Normalized animation progress, typically ranging
                from `0.0` to `1.0`.
        """
        self._bg_alpha = int(100 * progress)
        offset = int(50 * (1.0 - progress))
        self.base_layout.setContentsMargins(0, offset, 0, 0)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the translucent scrim behind the password-reset card.

        Fills the entire overlay with a black color whose alpha channel is
        controlled by `_bg_alpha`. This provides the dimmed backdrop used
        during the overlay's opening and closing animations.

        Args:
            event (QtGui.QPaintEvent): The Qt paint event requesting the overlay
                to be repainted.
        """
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, self._bg_alpha))
        p.end()

    def eventFilter(self, obj, event) -> bool:
        """Keep the overlay sized to its parent during resize events.

        Detects resize events from the overlay's parent widget and updates the
        overlay size to match the parent's new dimensions.

        Args:
            obj (QtCore.QObject): Object that generated the event.
            event (QtCore.QEvent): Event being processed.

        Returns:
            bool: The result of the base class event-filter implementation.
        """
        if obj is self.parent() and event.type() == QtCore.QEvent.Type.Resize:
            self.resize(event.size())
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Dismiss the overlay when the scrim outside the card is clicked.

        Treats clicks outside the glass card as a request to reject the
        password reset. Clicks occurring inside the card are passed to the
        base class implementation for normal child-widget handling.

        Args:
            event (QtGui.QMouseEvent): Mouse press event received by the
                overlay.
        """
        if not self.glass_frame.geometry().contains(event.pos()):
            self._reject()
        else:
            super().mousePressEvent(event)
