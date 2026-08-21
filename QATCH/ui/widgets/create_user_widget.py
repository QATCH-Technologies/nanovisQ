"""
QATCH.ui.widgets.create_user_widget.py

User account creation overlay for the QATCH application.

Provides the :class:`CreateUserWidget`, a modal-style overlay used to collect
and validate information for creating a new user account.

The widget presents a centered form over a semi-transparent application scrim
and provides fields for the user's name, role, optional username, email
address, and password. Validation errors are displayed inline with visual
error styling and animated shake feedback.

The interface is theme-aware and uses the application's shared authentication
styles, icons, and theme tokens. The overlay also provides animated entrance
and dismissal transitions and exposes the validated account information
through the widget's result state when creation is accepted.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-06-19
"""

from __future__ import annotations

import os
import re

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.core.constants import UserRoles
from QATCH.ui.components import AnimatedComboBox, QATCHLineEdit
from QATCH.ui.components.icon_utils import tinted_icon, tinted_pixmap
from QATCH.ui.styles.theme_manager import (
    ThemeManager,
    auth_card_qss,
    auth_shadow_color,
    card_title_qss,
    close_button_qss,
    error_label_qss,
    glass_combo_qss,
    gradient_button_qss,
)

_INPUT_H: int = 34


class CreateUserWidget(QtWidgets.QWidget):
    """Full-screen overlay for creating a new user account.

    Presents a modal-style user creation form over its parent widget. The form
    collects account details, validates the entered values, and exposes the
    validated account data when creation is accepted.

    The overlay adapts its styling to the active application theme and
    maintains its size with respect to the parent widget. It also provides
    animated visual feedback for opening the form and invalid input.

    Attributes:
        existing_initials: Initials already assigned to existing users, used
            to prevent duplicate initials.
        is_accepted: Whether the form has been successfully validated and
            submitted.
        result_data: Validated data for the newly created user, including
            name, username, email, initials, role, and password.
        base_layout: Layout used to center the main form container within the
            overlay.
        glass_frame: Main visual container for the user creation form.
        main_layout: Layout containing the form controls and action buttons.
        btn_close: Button used to close the user creation overlay.
        inp_first_name: Input field for the user's first name.
        inp_last_name: Input field for the user's last name.
        err_name: Label used to display name validation errors.
        cmb_role: Combo box used to select the user's role.
        inp_username: Optional input field for a custom username.
        inp_email: Input field for the user's email address.
        err_email: Label used to display email validation errors.
        inp_pwd1: Input field for the user's password.
        inp_pwd2: Input field used to confirm the user's password.
        err_password: Label used to display password validation errors.
        btn_create: Button used to validate and submit the form.
        _shake_anims: Active input-error shake animations.
        _bg_alpha: Current alpha value used for the overlay background
            animation.

    Args:
        existing_initials: Initials already assigned to existing users.
        parent: Optional parent widget over which the overlay is displayed.
    """

    def __init__(
        self,
        existing_initials: list,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the user creation overlay.

        Sets up the overlay state, sizes it to its parent when available,
        initializes the form controls, subscribes to theme changes, and
        starts the entrance animation.

        Args:
            existing_initials: Initials already assigned to existing users.
            parent: Optional parent widget over which the overlay is displayed.
        """
        super().__init__(parent)
        self.existing_initials = existing_initials
        self.is_accepted: bool = False
        self.result_data: dict = {}
        self._shake_anims: list[QtCore.QPropertyAnimation] = []
        self._bg_alpha: int = 0
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

        The supplied theme mode is not used directly; the current theme tokens
        are retrieved by the widget's theme application logic.

        Args:
            _mode: Identifier for the newly activated theme mode.
        """
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Apply the active theme to the user creation interface.

        Refreshes theme-dependent styles, colors, shadows, and icons for the
        user creation form. Password visibility state is preserved while the
        password visibility icons are regenerated using the current theme colors.

        This method is called during initialization and whenever the application
        theme changes.
        """
        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")
        tok = ThemeManager.instance().tokens()

        self.glass_frame.setStyleSheet(auth_card_qss("createUserView"))
        self._shadow.setColor(auth_shadow_color())
        self.btn_close.setStyleSheet(close_button_qss())

        icon_color = QtGui.QColor(*tok["flat_text"])
        self._icon_lbl.setPixmap(
            tinted_pixmap(os.path.join(icons_dir, "user-circle.svg"), icon_color, size=48)
        )
        self._lbl_title.setStyleSheet(card_title_qss())

        eye_color = QtGui.QColor(*tok["flat_text_muted"])
        self._eye_on = tinted_icon(os.path.join(icons_dir, "eye-on.svg"), eye_color)
        self._eye_off = tinted_icon(os.path.join(icons_dir, "eye-off.svg"), eye_color)
        self._act_eye1.setIcon(self._eye_off if self._pwd1_visible else self._eye_on)
        self._act_eye2.setIcon(self._eye_off if self._pwd2_visible else self._eye_on)

        self.cmb_role.setStyleSheet(glass_combo_qss(error=False))

        for err_lbl in (self.err_name, self.err_email, self.err_password):
            err_lbl.setStyleSheet(error_label_qss())

        submit_icon_color = QtGui.QColor(*tok["flat_on_accent"])
        self.btn_create.setIcon(
            tinted_icon(os.path.join(icons_dir, "right-arrow.svg"), submit_icon_color, size=20)
        )
        self.btn_create.setStyleSheet(gradient_button_qss())

    def _setup_ui(self) -> None:
        """Build and configure the user creation form.

        Creates the centered form card, header and close controls, user identity
        fields, role selector, optional username field, email field, password
        fields, validation labels, and user creation action button. Configures
        the associated layouts, widget properties, icons, styles, and signal
        connections required for interactive account creation.

        The resulting interface is added to the widget's centered base layout.
        """
        # Centred outer layout
        self.base_layout = QtWidgets.QVBoxLayout(self)
        self.base_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Card Frame
        self.glass_frame = QtWidgets.QFrame(self)
        self.glass_frame.setObjectName("createUserView")
        self.glass_frame.setFixedWidth(440)
        self.glass_frame.setStyleSheet(auth_card_qss("createUserView"))

        self._shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(44)
        self._shadow.setColor(auth_shadow_color())
        self._shadow.setOffset(0, 10)
        self.glass_frame.setGraphicsEffect(self._shadow)

        self.main_layout = QtWidgets.QVBoxLayout(self.glass_frame)
        self.main_layout.setContentsMargins(28, 14, 28, 28)
        self.main_layout.setSpacing(6)

        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")

        # Close button
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

        # User icon
        icon_lbl = QtWidgets.QLabel()
        icon_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        icon_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        icon_path = os.path.join(icons_dir, "user-circle.svg")
        icon_color = QtGui.QColor(*ThemeManager.instance().tokens()["flat_text"])
        icon_lbl.setPixmap(tinted_pixmap(icon_path, icon_color, size=48))
        icon_lbl.setFixedHeight(52)
        self._icon_lbl = icon_lbl
        self.main_layout.addWidget(icon_lbl, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Title
        lbl_title = QtWidgets.QLabel("Create User")
        lbl_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_title.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        lbl_title.setStyleSheet(card_title_qss())
        self._lbl_title = lbl_title
        self.main_layout.addWidget(lbl_title)
        self.main_layout.addSpacing(6)

        # Hide / show icons
        eye_color = QtGui.QColor(*ThemeManager.instance().tokens()["flat_text_muted"])
        self._eye_on = tinted_icon(os.path.join(icons_dir, "eye-on.svg"), eye_color)
        self._eye_off = tinted_icon(os.path.join(icons_dir, "eye-off.svg"), eye_color)
        self._pwd1_visible = False
        self._pwd2_visible = False

        # First/last name rows
        self.name_container = QtWidgets.QWidget()
        self.name_container.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        name_row = QtWidgets.QHBoxLayout(self.name_container)
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.setSpacing(10)

        self.inp_first_name = QATCHLineEdit()
        self.inp_first_name.setFixedHeight(_INPUT_H)
        self.inp_first_name.setPlaceholderText("First Name")

        self.inp_last_name = QATCHLineEdit()
        self.inp_last_name.setFixedHeight(_INPUT_H)
        self.inp_last_name.setPlaceholderText("Last Name")

        name_row.addWidget(self.inp_first_name)
        name_row.addWidget(self.inp_last_name)
        self.main_layout.addWidget(self.name_container)

        self.err_name = self._make_error_label()
        self.main_layout.addWidget(self.err_name)

        # Clear name error on any change
        for f in (self.inp_first_name, self.inp_last_name):
            f.textChanged.connect(
                lambda _, _f=f: self._clear_field_errors(
                    [self.inp_first_name, self.inp_last_name], self.err_name
                )
            )

        # Role combobox
        self.cmb_role = AnimatedComboBox(os.path.join(icons_dir, "down-arrow.svg"))
        self.cmb_role.setFixedHeight(_INPUT_H)
        roles = [e.name for e in UserRoles][1:]
        for r in roles:
            label = r + " (Capture & Analyze)" if r == UserRoles.OPERATE.name else r
            self.cmb_role.addItem(label, r)

        self.cmb_role.setStyleSheet(glass_combo_qss(error=False))
        self.main_layout.addWidget(self.cmb_role)

        # Username
        self.inp_username = QATCHLineEdit()
        self.inp_username.setFixedHeight(_INPUT_H)
        self.inp_username.setPlaceholderText("Username (Optional)")
        self.main_layout.addWidget(self.inp_username)

        # Email
        self.inp_email = QATCHLineEdit()
        self.inp_email.setFixedHeight(_INPUT_H)
        self.inp_email.setPlaceholderText("Email")
        self.inp_email.textChanged.connect(
            lambda _: self._clear_field_errors([self.inp_email], self.err_email)
        )
        self.main_layout.addWidget(self.inp_email)

        self.err_email = self._make_error_label()
        self.main_layout.addWidget(self.err_email)

        # Password
        self.inp_pwd1 = QATCHLineEdit()
        self.inp_pwd1.setFixedHeight(_INPUT_H)
        self.inp_pwd1.setPlaceholderText("Password")
        self.inp_pwd1.setEchoMode(QtWidgets.QLineEdit.Password)
        self.inp_pwd1.textChanged.connect(
            lambda _: self._clear_field_errors([self.inp_pwd1, self.inp_pwd2], self.err_password)
        )
        self._act_eye1 = self.inp_pwd1.addAction(self._eye_on, QtWidgets.QLineEdit.TrailingPosition)
        assert self._act_eye1 is not None
        self._act_eye1.triggered.connect(self._toggle_pwd1)
        self.main_layout.addWidget(self.inp_pwd1)

        # Confirm password
        self.inp_pwd2 = QATCHLineEdit()
        self.inp_pwd2.setFixedHeight(_INPUT_H)
        self.inp_pwd2.setPlaceholderText("Confirm Password")
        self.inp_pwd2.setEchoMode(QtWidgets.QLineEdit.Password)
        self.inp_pwd2.textChanged.connect(
            lambda _: self._clear_field_errors([self.inp_pwd1, self.inp_pwd2], self.err_password)
        )
        self._act_eye2 = self.inp_pwd2.addAction(self._eye_on, QtWidgets.QLineEdit.TrailingPosition)
        self._act_eye2.triggered.connect(self._toggle_pwd2)  # type: ignore
        self.main_layout.addWidget(self.inp_pwd2)

        self.err_password = self._make_error_label()
        self.main_layout.addWidget(self.err_password)

        self.main_layout.addSpacing(10)

        # Button row
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.btn_create = QtWidgets.QPushButton("")
        submit_icon_color = QtGui.QColor(*ThemeManager.instance().tokens()["flat_on_accent"])
        self.btn_create.setIcon(
            tinted_icon(os.path.join(icons_dir, "right-arrow.svg"), submit_icon_color, size=20)
        )
        self.btn_create.setIconSize(QtCore.QSize(20, 20))
        self.btn_create.setFixedSize(40, 40)
        self.btn_create.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_create.setToolTip("Create User")
        self.btn_create.setStyleSheet(gradient_button_qss())
        self.btn_create.clicked.connect(self._validate_and_accept)
        btn_layout.addWidget(self.btn_create)
        self.main_layout.addLayout(btn_layout)

        self.base_layout.addWidget(self.glass_frame)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the semi-transparent dark overlay behind the form.

        Fills the widget's entire rectangular area using the current background
        alpha value to visually dim the parent content beneath the user creation
        form.

        Args:
            event: Qt paint event that triggered the repaint.
        """
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, self._bg_alpha))
        p.end()

    @staticmethod
    def _make_error_label() -> QtWidgets.QLabel:
        """Create a hidden inline label for displaying validation errors.

        Configures a compact, word-wrapped label using the application's standard
        error styling. The returned label is hidden until a validation error needs
        to be displayed.

        Returns:
            A configured, initially hidden error label.
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
        """Clear validation errors for the specified input fields.

        Removes error styling from supported line-edit widgets and hides the
        associated inline error message.

        Args:
            fields: Input widgets whose error state should be cleared.
            error_label: Error label associated with the specified fields.
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
        """Display a validation error and provide visual feedback.

        Applies error styling to the specified input fields, displays the supplied
        error message, and triggers a horizontal shake animation on the selected
        widget. If no explicit shake target is provided, the first field in
        `fields` is used.

        Args:
            fields: Input widgets to mark as invalid.
            error_label: Label used to display the validation error message.
            message: Error message to display.
            shake_target: Optional widget to animate. Defaults to the first widget
                in `fields` when available.
        """
        for f in fields:
            if isinstance(f, QATCHLineEdit):
                f.set_error(True)
        error_label.setText(message)
        error_label.setVisible(True)
        self._shake_widget(shake_target or (fields[0] if fields else None))

    def _shake_widget(self, widget: QtWidgets.QWidget | None) -> None:
        """Animate a widget horizontally to indicate a validation error.

        Applies a short back-and-forth jiggle to the widget's position and retains
        the animation until completion so it is not garbage-collected prematurely.

        Args:
            widget: Widget to animate. No animation is performed if the widget is
                `None` or not currently visible.
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
        """Toggle visibility of the primary password field.

        Switches the password field between masked and plain-text modes and updates
        its eye action icon to reflect the current visibility state.
        """
        if self._act_eye1 is None:
            return

        self._pwd1_visible = not self._pwd1_visible
        self.inp_pwd1.setEchoMode(
            QtWidgets.QLineEdit.EchoMode.Normal
            if self._pwd1_visible
            else QtWidgets.QLineEdit.EchoMode.Password
        )
        self._act_eye1.setIcon(self._eye_off if self._pwd1_visible else self._eye_on)

    def _toggle_pwd2(self) -> None:
        """Toggle visibility of the password confirmation field.

        Switches the confirmation field between masked and plain-text modes and
        updates its eye action icon to reflect the current visibility state.
        """
        if self._act_eye2 is None:
            return

        self._pwd2_visible = not self._pwd2_visible
        self.inp_pwd2.setEchoMode(
            QtWidgets.QLineEdit.EchoMode.Normal
            if self._pwd2_visible
            else QtWidgets.QLineEdit.EchoMode.Password
        )
        self._act_eye2.setIcon(self._eye_off if self._pwd2_visible else self._eye_on)

    def _generate_initials(self, first: str, last: str) -> str:
        """Generate unique initials for a new user.

        Creates initials from the first characters of the supplied first and last
        names. If those initials are already assigned to an existing user, appends
        an incrementing numeric suffix until a unique value is produced.

        Args:
            first: User's first name.
            last: User's last name.

        Returns:
            A unique uppercase initials string.
        """
        base = f"{first[0]}{last[0]}".upper()
        initials, counter = base, 1
        while initials in self.existing_initials:
            initials = f"{base}{counter}"
            counter += 1
        return initials

    def _validate_and_accept(self) -> None:
        """Validate the form and accept the new user when all inputs are valid.

        Clears existing validation errors, validates the user's name, email
        address, password strength, and password confirmation, and provides
        visual error feedback for invalid fields. When validation succeeds,
        populates :attr:`result_data`, marks the form as accepted, and starts the
        closing animation.

        The generated user data includes the normalized full name, optional
        username, email address, unique initials, selected role, and password.
        """
        # Reset all error states
        self._clear_field_errors([self.inp_first_name, self.inp_last_name], self.err_name)
        self._clear_field_errors([self.inp_email], self.err_email)
        self._clear_field_errors([self.inp_pwd1, self.inp_pwd2], self.err_password)

        first = self.inp_first_name.text().strip().title()
        last = self.inp_last_name.text().strip().title()
        email = self.inp_email.text().strip()
        pwd1 = self.inp_pwd1.text()
        pwd2 = self.inp_pwd2.text()

        has_error = False

        # Name
        bad_first = len(first) < 2
        bad_last = len(last) < 2
        if bad_first or bad_last:
            if bad_first:
                self.inp_first_name.set_error(True)
            if bad_last:
                self.inp_last_name.set_error(True)
            self.err_name.setText("Please enter a valid First and Last name.")
            self.err_name.setVisible(True)
            self._shake_widget(self.name_container)
            has_error = True

        # Email
        if not email or not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
            self._show_field_error(
                [self.inp_email],
                self.err_email,
                "Please enter a valid email address.",
            )
            has_error = True

        # Password
        pwd_regex = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$"
        if not re.match(pwd_regex, pwd1):
            self._show_field_error(
                [self.inp_pwd1],
                self.err_password,
                "Password must be ≥ 8 characters with an uppercase, lowercase, and digit.",
            )
            has_error = True
        elif pwd1 != pwd2:
            self._show_field_error(
                [self.inp_pwd2],
                self.err_password,
                "Passwords do not match.",
            )
            has_error = True

        if has_error:
            return

        # All good
        self.result_data = {
            "name": f"{first} {last}",
            "username": self.inp_username.text().strip(),
            "email": email,
            "initials": self._generate_initials(first, last),
            "role": UserRoles[self.cmb_role.currentData()],
            "password": pwd1,
        }
        self.is_accepted = True
        self._close_with_animation()

    def _reject(self) -> None:
        """Reject the user creation request and close the overlay.

        Marks the form as not accepted and starts the closing animation without
        populating or modifying the validated result data.
        """
        self.is_accepted = False
        self._close_with_animation()

    def _animate_open(self) -> None:
        """Animate the user creation overlay into view.

        Fades in the dark background overlay while transitioning the form card
        into its visible position using an easing curve for a smooth entrance
        effect.
        """
        self.anim_in = QtCore.QVariantAnimation(self)
        self.anim_in.setDuration(300)
        self.anim_in.setStartValue(0.0)
        self.anim_in.setEndValue(1.0)
        self.anim_in.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self.anim_in.valueChanged.connect(self._on_anim_frame)
        self.anim_in.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def _close_with_animation(self) -> None:
        """Animate the overlay closed before destroying the widget.

        Disables the form to prevent further interaction during dismissal, then
        reverses the entrance effect by fading out the overlay and moving the form
        card downward. The widget is closed automatically when the animation
        completes.
        """
        # Disable interactions so the user can't double-click 'close' or 'create'
        self.glass_frame.setEnabled(False)

        self.anim_out = QtCore.QVariantAnimation(self)
        self.anim_out.setDuration(200)
        self.anim_out.setStartValue(1.0)
        self.anim_out.setEndValue(0.0)
        self.anim_out.setEasingCurve(QtCore.QEasingCurve.InQuad)
        self.anim_out.valueChanged.connect(self._on_anim_frame)
        self.anim_out.finished.connect(self.close)  # type: ignore
        self.anim_out.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def _on_anim_frame(self, progress: float) -> None:
        """Update the overlay and form position for an animation frame.

        Adjusts the background opacity and vertical offset of the form card based
        on the current animation progress.

        Args:
            progress: Normalized animation progress, typically ranging from
                `0.0` to `1.0`.
        """
        self._bg_alpha = int(130 * progress)
        offset = int(50 * (1.0 - progress))
        self.base_layout.setContentsMargins(0, offset, 0, 0)
        self.update()
