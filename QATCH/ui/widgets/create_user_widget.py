"""
create_user_widget.py

Provides an overlay widget for creating a new user account.

This module contains the `CreateUserWidget` class, which covers the parent application
with a semi-opaque scrim and centers a glass-morphism card containing a user creation form.
It features per-field inline error reporting, animated jiggle feedback, and styling
consistent with the application's modern UI conventions.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-06-19
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

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
    """Full-screen overlay widget for user account creation.

    Covers the parent widget with a semi-opaque scrim and centres a
    glass-morphism card containing the creation form.

    Attributes:
        existing_initials (list): A list of initials already in use by other users.
        is_accepted (bool): Indicates whether the form was successfully validated
            and submitted.
        result_data (dict): A dictionary containing the newly created user's
            validated data (name, username, email, initials, role, password).
        base_layout (QtWidgets.QVBoxLayout): The centered outer layout.
        glass_frame (QtWidgets.QFrame): The main glass card container.
        main_layout (QtWidgets.QVBoxLayout): The inner layout of the glass card.
        btn_close (QtWidgets.QPushButton): The close window button.
        inp_first_name (GlassLineEdit): Input field for the user's first name.
        inp_last_name (GlassLineEdit): Input field for the user's last name.
        err_name (QtWidgets.QLabel): Inline error label for the name inputs.
        cmb_role (AnimatedComboBox): Dropdown selection for the user's role.
        inp_username (GlassLineEdit): Optional input field for a custom username.
        inp_email (GlassLineEdit): Input field for the user's email address.
        err_email (QtWidgets.QLabel): Inline error label for the email input.
        inp_pwd1 (GlassLineEdit): Input field for the user's password.
        inp_pwd2 (GlassLineEdit): Input field to confirm the password.
        err_password (QtWidgets.QLabel): Inline error label for password inputs.
        btn_create (QtWidgets.QPushButton): The submit button to create the user.
    """

    def __init__(
        self,
        existing_initials: list,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initializes the CreateUserWidget.

        Args:
            existing_initials (list): A list of strings representing already taken initials.
            parent (QtWidgets.QWidget, optional): The parent widget to overlay.
                Defaults to None.
        """
        super().__init__(parent)
        self.existing_initials = existing_initials
        self.is_accepted: bool = False
        self.result_data: dict = {}
        self._shake_anims: List[QtCore.QPropertyAnimation] = []
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

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------
    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Re-applies every themed style on this card to the active palette -
        wired to ThemeManager.themeChanged so switching light/dark live
        re-colors it instead of only picking up the new theme on next
        construction."""
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
        """Builds and arranges the UI components of the widget."""
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

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Paints a semi-transparent dark overlay background.

        Args:
            event (QtGui.QPaintEvent): The Qt paint event.
        """
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, self._bg_alpha))
        p.end()

    @staticmethod
    def _make_error_label() -> QtWidgets.QLabel:
        """Constructs a compact, red inline error label.

        Returns:
            QtWidgets.QLabel: The newly created error label, hidden by default.
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
        """Clears the error styling on inputs and hides the associated error label.

        Args:
            fields (list): A list of input widgets (e.g., GlassLineEdit) to reset.
            error_label (QtWidgets.QLabel): The inline error label to hide.
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
        shake_target: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Applies error styling, displays the error message, and triggers a jiggle.

        Args:
            fields (list): A list of input widgets to mark with error styling.
            error_label (QtWidgets.QLabel): The label to display the error text.
            message (str): The error message text.
            shake_target (QtWidgets.QWidget, optional): The specific widget to apply
                the shake animation to. If None, uses the first widget in `fields`.
        """
        for f in fields:
            if isinstance(f, QATCHLineEdit):
                f.set_error(True)
        error_label.setText(message)
        error_label.setVisible(True)
        self._shake_widget(shake_target or (fields[0] if fields else None))

    def _shake_widget(self, widget: Optional[QtWidgets.QWidget]) -> None:
        """Triggers a horizontal jiggle animation for visual error feedback.

        Args:
            widget (QtWidgets.QWidget | None): The widget to animate.
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
        """Toggles visibility and the eye icon for the primary password field."""
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
        """Toggles visibility and the eye icon for the confirm password field."""
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
        """Generates a unique set of initials based on the user's name.

        If the direct initials are already in `existing_initials`, appends an
        incrementing numeric counter until a unique string is found.

        Args:
            first (str): The user's first name.
            last (str): The user's last name.

        Returns:
            str: The validated, unique initials string.
        """
        base = f"{first[0]}{last[0]}".upper()
        initials, counter = base, 1
        while initials in self.existing_initials:
            initials = f"{base}{counter}"
            counter += 1
        return initials

    def _validate_and_accept(self) -> None:
        """Validates all form inputs, saves data on success, and initiates closure.

        Validates name length, email format, password strength, and password match.
        If validation fails, the relevant fields are highlighted and shaken. If successful,
        populates `self.result_data` and begins the closing animation.
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
        """Marks the action as rejected and closes the widget via animation."""
        self.is_accepted = False
        self._close_with_animation()

    def _animate_open(self) -> None:
        """Fades in the dark overlay and slides the form card up into place."""
        self.anim_in = QtCore.QVariantAnimation(self)
        self.anim_in.setDuration(300)
        self.anim_in.setStartValue(0.0)
        self.anim_in.setEndValue(1.0)
        self.anim_in.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self.anim_in.valueChanged.connect(self._on_anim_frame)
        self.anim_in.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def _close_with_animation(self) -> None:
        """Reverses the intro animation, fading out before destroying the widget."""
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
        """Updates the background opacity and card position based on animation progress.

        Args:
            progress (float): The current interpolation value (0.0 to 1.0).
        """
        self._bg_alpha = int(130 * progress)
        offset = int(50 * (1.0 - progress))
        self.base_layout.setContentsMargins(0, offset, 0, 0)
        self.update()
