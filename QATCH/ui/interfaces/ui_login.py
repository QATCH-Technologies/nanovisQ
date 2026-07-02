"""QATCH.ui.interfaces.ui_login

Defines UILogin, the setup class for the nanovisQ login authentication overlay,
and UserInfo, a compatibility shim that routes legacy info calls to the floating
message badge widget.

Author(s):
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-07-01
"""

import os
from typing import TYPE_CHECKING, Callable

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles
from QATCH.core.constants import Constants, UserRoles
from QATCH.ui.components import GlassCard, GlassLineEdit
from QATCH.ui.widgets import (
    FloatingMessageBadgeWidget,
    LoginCentralWidget,
    SlidingPanel,
)

if TYPE_CHECKING:
    from QATCH.ui.main_window import MainWindow
    from QATCH.ui.windows import LoginWindow


class UserInfo:
    """Compatibility shim that redirects legacy user info calls to a floating badge.

    Attributes:
        _badge: FloatingMessageBadgeWidget used for displaying messages.
        _card: Parent card widget used as an anchor for positioning the badge.
        suppress: If True, ignores all incoming text updates (used during
            authentication transitions).
    """

    def __init__(self, badge: FloatingMessageBadgeWidget, card) -> None:
        """Initializes the UserInfo adapter.

        Args:
            badge: The floating badge widget responsible for rendering messages.
            card: The parent card used as a positional anchor for the badge.
        """
        self._badge = badge
        self._card = card
        self.suppress = False  # Suppresses legacy updates during transitions.

    def clear(self) -> None:
        """Clears any currently displayed user info message."""
        self._badge.clear()

    def setText(self, text: str) -> None:  # noqa: N802
        """Displays a message via the floating badge or clears it.

        If suppression is enabled, updates are ignored entirely. Otherwise,
        non-empty text is shown as an informational message; empty or blank
        strings clear the badge.

        Args:
            text: Message string to display.
        """
        if self.suppress:
            return  # Ignore legacy text updates during auth success

        if text and text.strip():
            self._badge.show_message(
                text.strip(),
                is_error=False,
                parent_widget=self._card,
            )
        else:
            self._badge.clear()

    def text(self) -> str:
        """Returns an empty string for compatibility with legacy interfaces.

        Returns:
            Always returns an empty string since state is managed externally.
        """
        return ""

    def __getattr__(self, name):
        """Fallback handler for unknown legacy attribute/method calls.

        Any undefined attribute access returns a no-op callable to prevent
        runtime errors from legacy UI code paths that expect a richer API.

        Args:
            name: Attribute name being accessed.

        Returns:
            A no-op function accepting arbitrary arguments.
        """
        return lambda *a, **kw: None


class UILogin:
    """
    Builds and manages the login user interface.

    This class constructs a complete authentication UI for the application,
    including sign-in, password recovery, session persistence, and animated
    transition effects. It is responsible for both layout creation and runtime
    behavior of the login experience.

    The login interface is built around a card-based design with a sliding
    internal panel that switches between authentication pages. The visual
    system emphasizes a "Deep Focus" metaphor, where the background application
    is dynamically blurred and dimmed while the login card is brought into
    focus.

    NOTE:
        - UI elements are created dynamically in `setup_ui()`.
        - The class assumes a parent MainWindow that manages post-authentication
          application state.
    """

    # Login card width
    _CARD_W: int = 320

    # Login card height
    _INPUT_H: int = 34

    # Button height(s)
    _BTN_H: int = 32

    # Page height
    _PAGE_H: int = 400

    # Sign-in page index
    _P_SIGNIN = 0

    # Password recovery page index
    _P_RECOVER = 1

    def setup_ui(
        self,
        login_window: "LoginWindow",
        parent: "MainWindow",
    ) -> None:
        self.parent = parent
        self.caps_lock_on = False

        # Window
        login_window.setObjectName("loginWindow")
        login_window.setMinimumSize(QtCore.QSize(800, 500))
        login_window.resize(800, 500)
        login_window.setTabShape(QtWidgets.QTabWidget.Rounded)

        # Central widget
        self.centralwidget = LoginCentralWidget(login_window)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        login_window.setCentralWidget(self.centralwidget)

        # Sign In
        signInPage = QtWidgets.QWidget()
        signInPage.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        si = QtWidgets.QVBoxLayout(signInPage)
        si.setContentsMargins(28, 26, 28, 22)
        si.setSpacing(10)
        si.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # Logo
        logoLabel = QtWidgets.QLabel()
        logoLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        logoLabel.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        logo_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "qatch-icon.png")
        logo_pm = QtGui.QPixmap(logo_path)

        if not logo_pm.isNull():
            logo_pm = logo_pm.scaled(
                54,
                54,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            logoLabel.setPixmap(logo_pm)
            logoLabel.setFixedSize(54, 54)

        si.addWidget(logoLabel, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        si.addSpacing(2)

        # Title
        siTitle = QtWidgets.QLabel("Sign In")
        siTitle.setObjectName("cardTitle")
        siTitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        si.addWidget(siTitle)
        si.addSpacing(4)

        self.user_username = GlassLineEdit()
        self.user_username.setObjectName("user_username")
        self.user_username.setFixedHeight(self._INPUT_H)
        self.user_username.setPlaceholderText("Username")
        self.user_username.textChanged.connect(lambda _: self.user_username.set_error(False))
        # NOTE: Until "username" is used for something other than "initials",
        #       normalize all user input to uppercase as they type characters
        self.user_username.textEdited.connect(
            lambda _: self.user_username.setText(self.user_username.text().upper())
        )
        si.addWidget(self.user_username)
        self.user_password = GlassLineEdit()
        self.user_password.setObjectName("user_password")
        self.user_password.setFixedHeight(self._INPUT_H)
        self.user_password.setPlaceholderText("Password")
        self.user_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.user_password.textChanged.connect(lambda _: self.user_password.set_error(False))
        self.user_password.returnPressed.connect(self.action_sign_in)
        self.user_username.returnPressed.connect(self.user_password.setFocus)
        self.user_initials = self.user_username

        # Password field
        self.user_password = GlassLineEdit()
        self.user_password.setObjectName("user_password")
        self.user_password.setFixedHeight(self._INPUT_H)
        self.user_password.setPlaceholderText("Password")
        self.user_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.user_password.textChanged.connect(lambda _: self.user_password.set_error(False))
        self.user_password.returnPressed.connect(self.action_sign_in)
        self.user_password.installEventFilter(login_window)
        self.user_username.installEventFilter(login_window)
        login_window.installEventFilter(login_window)

        # Eye action
        self.visibleIcon = QtGui.QIcon(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "eye-on.svg")
        )
        self.hiddenIcon = QtGui.QIcon(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "eye-off.svg")
        )
        self.password_shown = False
        self.togglepasswordAction = self.user_password.addAction(
            self.visibleIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.togglepasswordAction.triggered.connect(self.on_toggle_password_Action)
        si.addWidget(self.user_password)

        # Caps Lock indicator - always occupies its row; text is blank when off
        self.caps_indicator = QtWidgets.QLabel("")
        self.caps_indicator.setObjectName("capsIndicator")
        self.caps_indicator.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.caps_indicator.setFixedHeight(16)
        si.addWidget(self.caps_indicator, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Remember Me Toggle
        self.remember_me_cb = QtWidgets.QCheckBox("Remember me")
        self.remember_me_cb.setObjectName("rememberMe")
        self.remember_me_cb.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

        # Wrap in a horizontal layout to keep it left-aligned cleanly
        rm_layout = QtWidgets.QHBoxLayout()
        rm_layout.setContentsMargins(4, 0, 0, 0)
        rm_layout.addWidget(self.remember_me_cb)
        rm_layout.addStretch()
        si.addLayout(rm_layout)
        si.addSpacing(6)

        # Sign In button
        self.sign_in_btn = QtWidgets.QPushButton("Sign In")
        self.sign_in_btn.setObjectName("signInBtn")
        self.sign_in_btn.setFixedHeight(self._BTN_H)
        self.sign_in_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.sign_in_btn.clicked.connect(self.action_sign_in)
        si.addWidget(self.sign_in_btn)

        si.addStretch()

        # Forgot Password link
        forgotPasswordLbl = QtWidgets.QLabel("Forgot Password?")
        forgotPasswordLbl.setObjectName("forgotPassword")
        forgotPasswordLbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        forgotPasswordLbl.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        forgotPasswordLbl.mousePressEvent = lambda _e: self._slide_to(self._P_RECOVER)
        si.addWidget(forgotPasswordLbl)

        # Forgot Password
        recoverPage = QtWidgets.QWidget()
        recoverPage.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        rec = QtWidgets.QVBoxLayout(recoverPage)
        rec.setContentsMargins(28, 18, 28, 18)
        rec.setSpacing(10)
        rec.setAlignment(QtCore.Qt.AlignTop)

        back_icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "left-arrow.svg")

        rec.addWidget(
            self._make_back_btn(
                "Back to Sign In",
                lambda: self._slide_to(self._P_SIGNIN),
                icon_path=back_icon_path,
            ),
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft,
        )

        recTitle = QtWidgets.QLabel("Reset Password")
        recTitle.setObjectName("recoverTitle")
        recTitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        rec.addWidget(recTitle)

        recInfo = QtWidgets.QLabel("Contact your administrator to reset your password.")
        recInfo.setObjectName("recoverInfo")
        recInfo.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        recInfo.setWordWrap(True)
        rec.addWidget(recInfo)
        # TODO: Currently placeholder workflow to contact admin to reset
        self.recoverEmail = GlassLineEdit()
        self.recoverEmail.setObjectName("recoverEmail")
        self.recoverEmail.setPlaceholderText("Email Address")
        self.recoverEmail.setFixedHeight(self._INPUT_H)
        self.recoverEmail.setVisible(False)
        rec.addWidget(self.recoverEmail)

        self.recoverStatus = QtWidgets.QLabel("")
        self.recoverStatus.setObjectName("recoverStatus")
        self.recoverStatus.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.recoverStatus.setWordWrap(True)
        self.recoverStatus.setFixedHeight(34)
        rec.addWidget(self.recoverStatus)
        rec.addStretch()

        # Sliding panel
        self._slider = SlidingPanel(self._CARD_W)
        self._slider.setObjectName("slidingPanel")

        # Page layout: [0=SignIn, 1=Recover]
        self._slider.add_page(signInPage)
        self._slider.add_page(recoverPage)
        self._slider.setFixedHeight(self._PAGE_H)

        def _init_slider() -> None:
            self._slider.finalize(self._PAGE_H)
            # Position instantly at Sign In (page 0) with no animation
            self._slider._inner.move(0, 0)

        QtCore.QTimer.singleShot(0, _init_slider)

        self.loginCard = GlassCard(self.centralwidget)
        self.loginCard.setObjectName("loginCard")
        self.loginCard.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, False)
        self.loginCard.setContentsMargins(0, 0, 0, 0)
        # NOTE: intentionally no QGraphicsDropShadowEffect here - the card
        # gets its own CardPopEffect below (via register_dismissable_card,
        # for the sign-out/sign-in pop), and Qt doesn't compose two graphics
        # effects on the same widget. The card's own manual border painting
        # (see GlassCard.paintEvent) provides sufficient visual separation
        # in place of the shadow.
        self.centralwidget.register_dismissable_card(self.loginCard)

        card_vbox = QtWidgets.QVBoxLayout(self.loginCard)
        card_vbox.setContentsMargins(0, 0, 0, 0)
        card_vbox.setSpacing(0)
        card_vbox.addWidget(self._slider)

        v_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        v_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        v_layout.addStretch(2)
        v_layout.addSpacing(20)
        v_layout.addWidget(self.loginCard, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        v_layout.addStretch(3)

        # Floating badge for error / info messages
        self.floating_badge = FloatingMessageBadgeWidget(
            login_window,
            os.path.join(Architecture.get_path(), "QATCH", "icons", "clear.svg"),
        )

        # user_info proxy - legacy code calls .user_info.clear() / .setText(); route to badge
        self.user_info = UserInfo(self.floating_badge, self.loginCard)

        self._sessionTimer = QtCore.QTimer()
        self._sessionTimer.setSingleShot(True)
        self._sessionTimer.timeout.connect(self.check_user_session)
        self._sessionTimer.setInterval(1000 * 60 * 60)

        # Initialize settings and load remembered user
        self._settings = QtCore.QSettings("QATCH", "nanovisQ")
        self._load_remembered_user()

    @staticmethod
    def _make_back_btn(
        text: str,
        callback: Callable,
        icon_path: str = "",
    ) -> QtWidgets.QPushButton:
        """Creates a back-navigation button.

        Args:
            text (str): Text label displayed on the button.
            callback (Callable): Function invoked when the button is clicked.
            icon_path (str): Optional filesystem path to an SVG/bitmap icon displayed
                before the text. If empty or invalid, no icon is applied.

        Returns:
            A fully configured QPushButton ready for use in navigation layouts.
        """
        btn = QtWidgets.QPushButton(text)
        btn.setObjectName("backBtn")
        btn.setFixedHeight(24)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

        if icon_path and os.path.exists(icon_path):
            btn.setIcon(QtGui.QIcon(icon_path))
            btn.setIconSize(QtCore.QSize(14, 14))

        btn.clicked.connect(callback)
        return btn

    def _slide_to(self, page_idx: int) -> None:
        """Navigates the sliding panel to a target page and resets transient UI state.

        This method drives page navigation within the authentication flow. It not only
        triggers the animated transition in the SlidingPanel but also ensures that
        any temporary or context-specific UI state from secondary pages is cleared
        when returning to the primary Sign In screen.

        This prevents stale error messages, disabled controls, or partially completed
        recovery flows from persisting across navigation.

        Args:
            page_idx (int): Index of the target page to display (e.g., _P_SIGNIN or
                _P_RECOVER).
        """
        self._slider.slide_to(page_idx)

        if page_idx == self._P_SIGNIN:
            # Reset the password recovery page to its default state
            self.recoverEmail.clear()
            self.recoverStatus.clear()
            self.recoverStatus.setStyleSheet("")

    def on_toggle_password_Action(self) -> None:
        """Toggles password visibility in the password input field.

        This method switches the password field between masked and plain-text
        display modes. It also updates the associated action icon so the UI
        reflects the current visibility state.

        The toggle is stateful, driven by `self.password_shown`.

        Notes:
            - Uses QLineEdit.Password for masked input.
            - Uses QLineEdit.Normal for plain-text display.
            - Updates the trailing action icon to match the new state.
        """
        # Toggle state
        self.password_shown = not self.password_shown

        # Select echo mode and icon based on updated state
        mode = QtWidgets.QLineEdit.Normal if self.password_shown else QtWidgets.QLineEdit.Password

        icon = self.hiddenIcon if self.password_shown else self.visibleIcon

        # Apply UI updates
        self.user_password.setEchoMode(mode)
        assert self.togglepasswordAction is not None
        self.togglepasswordAction.setIcon(icon)

    def update_caps_lock_state(self, caps_lock_on: bool) -> None:
        """Updates the UI to reflect the current Caps Lock state.

        This method synchronizes the internal Caps Lock state with a visual
        indicator in the UI. It is typically called from an event filter that
        detects keyboard state changes.

        When Caps Lock is enabled, a warning message is displayed to help prevent
        unintended uppercase password input; otherwise, the indicator is cleared.

        Args:
            caps_lock_on (bool): True if Caps Lock is currently active, otherwise False.
        """
        self.caps_lock_on = caps_lock_on
        self.caps_indicator.setText("Caps Lock is On" if caps_lock_on else "")

    def action_sign_in(self) -> None:
        """Handles user authentication and transitions into the main application.

        This method orchestrates the complete sign-in workflow, including input
        validation, authentication, UI updates, permission handling, and the
        transition from the login overlay into the main application.
        """

        # Input validation
        username = self.user_username.text().strip()
        password = self.user_password.text()

        if not username:
            self._shake_widget(self.user_username)
            self.user_username.set_error(True)
            self.floating_badge.show_message(
                "Please enter your username",
                is_error=True,
                parent_widget=self.loginCard,
            )
            return

        if not password:
            self._shake_widget(self.user_password)
            self.user_password.set_error(True)
            self.floating_badge.show_message(
                "Password required",
                is_error=True,
                parent_widget=self.loginCard,
            )
            return

        # Authentication
        authenticated, _, params = UserProfiles.auth(username, password, UserRoles.ANY)

        if params is None:
            Log.e("Authorization could not be completed.")
            self.floating_badge.show_message(
                "Authorization could not be completed",
                is_error=True,
                parent_widget=self.loginCard,
            )
            self._clear_credentials()
            self.user_info.suppress = False
            return

        # Failure path
        if not authenticated:
            self.user_info.suppress = False
            self._clear_credentials()
            self.error_invalid()
            return

        # Success path
        name, _, role = params[0], params[1], params[2].value

        self.user_info.suppress = True
        self.floating_badge.slide_out()

        self._save_remembered_user(username)
        self._sessionTimer.start()

        Log.i(f"Welcome, {params[0]}! Role: {params[2].name}.")

        # Post-login UI setup
        controls = self.parent.controls_window

        controls.username.setText(f"User: {name}")
        controls.userrole = UserRoles(role)
        controls.signinout.setText("&Sign Out")
        controls.ui.tool_User.setText(name)
        self.parent.analyze_process.tool_User.setText(name)

        controls.set_signed_in_menu_state(True)
        self.parent.mode_window.ui.mark_signed_in()

        if controls.userrole != UserRoles.ADMIN:
            controls.manage.setText("&Change Password...")

        has_capture_perm = UserProfiles().check(
            controls.userrole,
            UserRoles.CAPTURE,
        )

        if has_capture_perm:
            self.parent.mode_window.ui._set_run_mode(None)
        else:
            self.parent.mode_window.ui._set_analyze_mode(None)

        # Transition + cleanup
        self.centralwidget.dismiss_for_signin()

        if hasattr(self.parent, "url_download"):
            delattr(self.parent, "url_download")

        self._clear_not_remembered_user()
        self._clear_credentials()
        self.user_info.suppress = False

    def _clear_credentials(self) -> None:
        """Clears sensitive authentication inputs and restores secure UI state.

        This method ensures that no residual credential data remains in the UI
        after authentication attempts or transitions. It also enforces a secure
        default visual state by restoring password masking if the password field
        was previously set to visible.

        """
        self.user_password.clear()
        if self.password_shown:
            self.on_toggle_password_Action()

    def clear_form(self) -> None:
        """Resets the login interface to its initial default state.

        This method performs a full UI reset of the authentication screen. It is
        used during application startup, sign-out, or when the user explicitly
        cancels/reset the login flow (e.g., Escape key).
        """
        self.loginCard.show()

        # Clear inputs and error styling
        self.user_username.clear()
        self.user_username.set_error(False)
        self.user_password.set_error(False)
        self._clear_credentials()

        # Reset indicators and navigation state
        self.update_caps_lock_state(False)
        self._slide_to(self._P_SIGNIN)

    def error_invalid(self, message: str = "Invalid Credentials") -> None:
        """Triggers the UI sequence for a failed authentication attempt.

        Applies red error styling to the password field, executes a shake
        animation, clears the invalid input, and displays a floating
        error badge with the provided message.

        Args:
            message (str, optional): The error text to display in the floating badge.
                Defaults to "Invalid Credentials".
        """
        self.user_password.set_error(True)
        self._shake_widget(self.user_password)
        self.user_password.clear()
        self.user_password.setFocus()
        self.floating_badge.show_message(message, is_error=True, parent_widget=self.loginCard)

    def error_loggedout(self) -> None:
        """Displays a logout notification and resets the password field state."""
        self.user_password.clear()
        self.user_password.set_error(False)
        self.show_signout_message()

    def error_expired(self) -> None:
        """Displays a session expiration warning and resets the password field state."""
        self.user_password.clear()
        self.user_password.set_error(False)
        self.floating_badge.show_message(
            "Your session has expired", is_error=True, parent_widget=self.loginCard
        )

    def show_signout_message(self) -> None:
        """Triggers a floating badge notification confirming the user has signed out.

        Drops in with a spring overshoot timed to land as the login card
        settles, per the "Deep Focus" sign-out reveal.
        """
        self.floating_badge.show_message(
            "You have been signed out.",
            is_error=True,
            parent_widget=self.loginCard,
            drop_in=True,
        )

    def check_user_session(self) -> None:
        """Validates the current user session and handles expiration recovery.

        This method is periodically invoked by a session timer to ensure the
        user's authentication state remains valid. It checks the session status
        via `UserProfiles.session_info()` and responds accordingly.
        """
        valid, _ = UserProfiles().session_info()

        if not valid:
            if self.parent.controls_window.userrole != UserRoles.NONE:
                Log.w("User session has expired.")
                self.parent.controls_window.set_user_profile()
        else:
            Log.d("User session valid at hourly check.")
            self._sessionTimer.start()

    def _clear_not_remembered_user(self) -> None:
        """Clears the not remembered username from login form on sign in action.

        Retrieves the 'Remember me' toggle state from checkbox widget. If not enabled,
        it clears the entered username value in the field to be ready for the next sign
        in and to make sure their username is not shown again when this session ends.
        """
        not_remembered = not self.remember_me_cb.isChecked()

        if not_remembered:
            self.user_username.clear()

    def _load_remembered_user(self) -> None:
        """Loads the remembered username from system settings on startup.

        Retrieves the 'Remember me' toggle state from QSettings. If previously enabled,
        it restores the saved username and automatically shifts keyboard focus to the
        password input field for a quicker sign-in experience.
        """
        remembered = self._settings.value("login/remember_me", False, type=bool)
        self.remember_me_cb.setChecked(remembered)

        if remembered:
            saved_user = self._settings.value("login/username", "", type=str)
            if saved_user:
                self.user_username.setText(saved_user)
                self.user_password.setFocus()

    def _save_remembered_user(self, username: str) -> None:
        """Saves or clears the remembered username based on the toggle state.

        Synchronizes the current state of the 'Remember me' checkbox with QSettings.
        If checked, the username is stored for future sessions. If unchecked,
        any previously stored username is wiped from the system's settings registry.

        Args:
            username (str): The authenticated username to store if the toggle is active.
        """
        is_remembered = self.remember_me_cb.isChecked()
        self._settings.setValue("login/remember_me", is_remembered)

        if is_remembered:
            self._settings.setValue("login/username", username)
        else:
            self._settings.remove("login/username")

    def retranslateUi(self, login_window: QtWidgets.QMainWindow) -> None:  # noqa: N802
        """Sets the window titles and localizable strings for the interface.

        Configures the primary window title using application constants and
        loads the software icon from the localized asset path.

        Args:
            login_window (QtWidgets.QMainWindow): The window instance to update.
        """
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "qatch-icon.png")
        login_window.setWindowIcon(QtGui.QIcon(icon_path))
        title_text = f"{Constants.app_title} {Constants.app_version} - Login"
        login_window.setWindowTitle(_translate("loginWindow", title_text))

    def _shake_widget(self, widget: QtWidgets.QWidget) -> None:
        """Executes a rapid horizontal jiggle animation to provide error feedback.

        Uses a QPropertyAnimation to manipulate the 'pos' property of the widget.
        The animation uses a series of decaying keyframes to simulate a physical
        shake that settles back at the original position.

        Args:
            widget (QtWidgets.QWidget): The target widget to animate.
        """
        if not widget or not widget.isVisible():
            return

        self._shake_anim = QtCore.QPropertyAnimation(widget, b"pos")
        self._shake_anim.setDuration(380)

        base = widget.pos()
        self._shake_anim.setKeyValueAt(0.0, base)
        self._shake_anim.setKeyValueAt(0.1, base + QtCore.QPoint(-6, 0))
        self._shake_anim.setKeyValueAt(0.3, base + QtCore.QPoint(6, 0))
        self._shake_anim.setKeyValueAt(0.5, base + QtCore.QPoint(-4, 0))
        self._shake_anim.setKeyValueAt(0.7, base + QtCore.QPoint(4, 0))
        self._shake_anim.setKeyValueAt(0.9, base + QtCore.QPoint(-2, 0))
        self._shake_anim.setKeyValueAt(1.0, base)

        self._shake_anim.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
