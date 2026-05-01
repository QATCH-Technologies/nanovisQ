import os
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles, UserRoles
from QATCH.core.constants import Constants
from QATCH.ui.popUp import PopUp


# ──────────────────────────────────────────────────────────────────────────────
class _LoginCentralWidget(QtWidgets.QWidget):
    """Central widget that paints a blurred, lightly frosted backdrop.

    The backdrop is sourced by calling :meth:`capture_backdrop` with the *run*
    window (``parent`` in ``UILogin``).  We call ``.grab()`` on that window
    directly — **no hide/show of the login window is ever needed**, which
    eliminates the spurious second-window event that toggling visibility caused.

    ``paintEvent`` draws the blurred pixmap wall-to-wall and then lays a very
    light neutral tint on top (~24 % opacity) so the effect reads as "frosted"
    without overpowering the glass card in front.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._blurred: Optional[QtGui.QPixmap] = None
        # We own all painting — prevent Qt pre-clearing with palette colour.
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    # ------------------------------------------------------------------ public
    def capture_backdrop(self, run_window: QtWidgets.QMainWindow) -> None:
        """Grab *run_window*, blur it, and schedule a repaint.

        Grabbing the run window directly (via ``.grab()``) means we never have
        to hide or show the login window, so no spurious window events fire.
        """
        # Render the run window into a pixmap at its native resolution.
        raw: QtGui.QPixmap = run_window.grab()

        # Scale to fill our widget so the backdrop covers every pixel.
        if not self.size().isEmpty():
            raw = raw.scaled(
                self.size(),
                QtCore.Qt.KeepAspectRatioByExpanding,
                QtCore.Qt.SmoothTransformation,
            )

        # ── Blur via QGraphicsScene (no Pillow / OpenCV dependency) ──────────
        scene = QtWidgets.QGraphicsScene()
        item = QtWidgets.QGraphicsPixmapItem(raw)
        blur = QtWidgets.QGraphicsBlurEffect()
        blur.setBlurRadius(22)
        blur.setBlurHints(QtWidgets.QGraphicsBlurEffect.QualityHint)
        item.setGraphicsEffect(blur)
        scene.addItem(item)

        out = QtGui.QPixmap(raw.size())
        out.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(out)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        scene.render(p, source=QtCore.QRectF(item.boundingRect()))
        p.end()

        self._blurred = out
        self.update()

    # --------------------------------------------------------------- Qt events
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        if self._blurred:
            p.drawPixmap(self.rect(), self._blurred, self._blurred.rect())
        else:
            # Gradient fallback shown for the instant before capture completes.
            grad = QtGui.QLinearGradient(0, 0, self.width(), self.height())
            grad.setColorAt(0.0, QtGui.QColor(0xE4, 0xEB, 0xF1))
            grad.setColorAt(1.0, QtGui.QColor(0xF4, 0xF7, 0xF9))
            p.fillRect(self.rect(), QtGui.QBrush(grad))

        # Very light, near-neutral frost tint (~24 % opacity).
        # Low alpha keeps the run-window content readable through the blur
        # without washing out the glass card that sits in front.
        p.fillRect(self.rect(), QtGui.QColor(238, 243, 247, 62))
        p.end()


# ──────────────────────────────────────────────────────────────────────────────
class UILogin:
    """
    User Interface for the Login Window.

    This class provides and manages the login interface for the QATCH application.
    It sets up and configures the UI elements for user authentication, including labels,
    text fields, buttons, and password visibility toggles. The class also handles events
    such as text transformation, session validation, and error reporting, ensuring a smooth
    user experience during the sign-in process.

    The UI is dynamically updated to reflect changes such as password visibility toggling,
    session expiration, and invalid credential notifications. Timers are used to clear error
    messages after a brief interval and to periodically check the validity of the user session.
    An action-sign in method is also provided to process and validate user credentials upon
    clicking the sign-in button.

    Attributes:
        parent (QtWidgets.QMainWindow): The parent window that the login UI interacts with.
        caps_lock_on (bool): Indicates if the Caps Lock is active.
        centralwidget (_LoginCentralWidget): The main widget containing the login interface.
        layout (QtWidgets.QGridLayout): Grid layout manager for arranging UI components.
        user_welcome (QtWidgets.QLabel): Label displaying a welcome message.
        user_label (QtWidgets.QLabel): Label prompting the user to sign in.
        user_initials (QtWidgets.QLineEdit): Text field for entering user initials.
        user_password (QtWidgets.QLineEdit): Text field for entering the password.
        sign_in (QtWidgets.QPushButton): Button that initiates the sign-in process.
        user_info (QtWidgets.QLabel): Label for displaying informational messages.
        user_error (QtWidgets.QLabel): Label for displaying error messages.
        _errorTimer (QtCore.QTimer): Timer to clear error messages after a set interval.
        _sessionTimer (QtCore.QTimer): Timer to periodically validate the user session.
        visibleIcon (QtGui.QIcon): Icon shown when the password is visible.
        hiddenIcon (QtGui.QIcon): Icon shown when the password is hidden.
        password_shown (bool): Flag indicating whether the password is currently shown.
        togglepasswordAction (QAction): Action to toggle password visibility.
    """

    # ─────────────────────────────────────────────────────────────── setup
    def setup_ui(self, MainWindow5: QtWidgets.QMainWindow, parent: QtWidgets.QMainWindow) -> None:
        """Initialise and arrange all UI elements for the login window."""
        self.parent = parent
        self.caps_lock_on = False  # set on focus of `user_password` field

        # ── Window basics ──────────────────────────────────────────────────────
        MainWindow5.setObjectName("MainWindow5")
        MainWindow5.setMinimumSize(QtCore.QSize(1000, 500))
        MainWindow5.resize(500, 500)
        MainWindow5.setTabShape(QtWidgets.QTabWidget.Rounded)

        # ── Antialiased system font ────────────────────────────────────────────
        font = QtGui.QFont("Segoe UI", 10)
        font.setStyleHint(QtGui.QFont.SansSerif)
        font.setHintingPreference(QtGui.QFont.PreferFullHinting)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias | QtGui.QFont.PreferQuality)
        MainWindow5.setFont(font)

        # ── Custom central widget (paints blurred backdrop) ────────────────────
        self.centralwidget = _LoginCentralWidget(MainWindow5)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        MainWindow5.setCentralWidget(self.centralwidget)

        # ── Load stylesheet ────────────────────────────────────────────────────
        qss_path = os.path.join(Architecture.get_path(), "QATCH", "ui", "ui_login_theme.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r") as fh:
                MainWindow5.setStyleSheet(fh.read())
        else:
            MainWindow5.setStyleSheet("")

        # ── Welcome label (sits above the card, on the frosted backdrop) ───────
        self.user_welcome = QtWidgets.QLabel(
            "<span>Welcome to QATCH nanovisQ<sup>TM</sup> Real-Time GUI</span>"
        )
        self.user_welcome.setObjectName("user_welcome")
        self.user_welcome.setAlignment(QtCore.Qt.AlignCenter)
        self.user_welcome.setFixedHeight(50)

        # ── Glass card frame ───────────────────────────────────────────────────
        # All form widgets live inside this QFrame so they appear as a single
        # elevated, semi-opaque panel distinct from the frosted backdrop.
        self.loginCard = QtWidgets.QFrame()
        self.loginCard.setObjectName("loginCard")
        self.loginCard.setContentsMargins(36, 28, 36, 28)

        # Soft diffuse shadow lifts the card off the backdrop.
        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(36)
        shadow.setOffset(0, 6)
        shadow.setColor(QtGui.QColor(15, 40, 70, 80))
        self.loginCard.setGraphicsEffect(shadow)

        # ── Card inner grid ────────────────────────────────────────────────────
        self.layout = QtWidgets.QGridLayout(self.loginCard)
        self.layout.setSpacing(12)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.user_label = QtWidgets.QLabel("<span><b>User Sign-In Required</b></span>")
        self.user_label.setObjectName("user_label")
        self.user_label.setFixedHeight(38)
        self.layout.addWidget(self.user_label, 0, 0, QtCore.Qt.AlignCenter)

        self.user_initials = QtWidgets.QLineEdit()
        self.user_initials.setObjectName("user_initials")
        self.user_initials.textEdited.connect(self.text_transform)
        self.user_initials.setMinimumWidth(230)
        self.user_initials.setFixedHeight(36)
        self.user_initials.setPlaceholderText("Initials")
        self.user_initials.setMaxLength(4)
        self.user_initials.installEventFilter(MainWindow5)
        self.layout.addWidget(self.user_initials, 1, 0, QtCore.Qt.AlignCenter)

        self.user_password = QtWidgets.QLineEdit()
        self.user_password.setObjectName("user_password")
        self.user_password.setMinimumWidth(230)
        self.user_password.setFixedHeight(36)
        self.user_password.setPlaceholderText("Password")
        self.user_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.user_password.installEventFilter(MainWindow5)
        self.layout.addWidget(self.user_password, 2, 0, QtCore.Qt.AlignCenter)

        self.sign_in = QtWidgets.QPushButton("&Sign In")
        self.sign_in.setObjectName("sign_in")
        self.sign_in.setMinimumWidth(230)
        self.sign_in.setFixedHeight(36)
        self.sign_in.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.sign_in.clicked.connect(self.action_sign_in)
        self.sign_in.installEventFilter(MainWindow5)
        # Force Fusion style so QSS background-color overrides the native
        # Windows button painter (which ignores stylesheet gradients).
        self.sign_in.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
        self.layout.addWidget(self.sign_in, 3, 0, QtCore.Qt.AlignCenter)

        self.user_info = QtWidgets.QLabel("")
        self.user_info.setObjectName("user_info")
        self.layout.addWidget(self.user_info, 4, 0, QtCore.Qt.AlignCenter)

        self.user_error = QtWidgets.QLabel("")
        self.user_error.setObjectName("user_error")
        self.layout.addWidget(self.user_error, 5, 0, QtCore.Qt.AlignCenter)

        # ── Outer layout ───────────────────────────────────────────────────────
        v_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        v_layout.setAlignment(QtCore.Qt.AlignCenter)
        v_layout.addStretch(2)
        v_layout.addWidget(self.user_welcome, alignment=QtCore.Qt.AlignCenter)
        v_layout.addSpacing(20)
        v_layout.addWidget(self.loginCard, alignment=QtCore.Qt.AlignCenter)
        v_layout.addStretch(3)

        # ── Timers ─────────────────────────────────────────────────────────────
        self._errorTimer = QtCore.QTimer()
        self._errorTimer.setSingleShot(True)
        self._errorTimer.timeout.connect(self.user_error.clear)

        self._sessionTimer = QtCore.QTimer()
        self._sessionTimer.setSingleShot(True)
        self._sessionTimer.timeout.connect(self.check_user_session)
        self._sessionTimer.setInterval(1000 * 60 * 60)  # once an hour

        # ── Password toggle icons ──────────────────────────────────────────────
        self.visibleIcon = QtGui.QIcon(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "eye.svg")
        )
        self.hiddenIcon = QtGui.QIcon(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "hide.svg")
        )
        self.password_shown = False
        self.togglepasswordAction = self.user_password.addAction(
            self.visibleIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.togglepasswordAction.triggered.connect(self.on_toggle_password_Action)

        # ── Backdrop capture ───────────────────────────────────────────────────
        # Grab the run window (parent) directly — no hide/show of MainWindow5
        # needed, so no spurious window creation events fire.
        QtCore.QTimer.singleShot(0, lambda: self.centralwidget.capture_backdrop(parent))

    # ─────────────────────────────────────────────── password toggle
    def on_toggle_password_Action(self) -> None:
        """Toggle password field between masked and plain text."""
        if not self.password_shown:
            self.user_password.setEchoMode(QtWidgets.QLineEdit.Normal)
            self.password_shown = True
            self.togglepasswordAction.setIcon(self.hiddenIcon)
        else:
            self.user_password.setEchoMode(QtWidgets.QLineEdit.Password)
            self.password_shown = False
            self.togglepasswordAction.setIcon(self.visibleIcon)

    # ─────────────────────────────────────────────── session
    def check_user_session(self) -> None:
        """Check the current user's session and prompt for sign-in if expired."""
        valid, infos = UserProfiles().session_info()
        if not valid:
            if self.parent.ControlsWin.userrole == UserRoles.NONE:
                Log.d("Hourly session check: user already signed out, skipping prompt.")
            else:
                Log.w("User session has expired.")
                Log.i("Please sign in to continue.")
                self.parent.ControlsWin.set_user_profile()
        else:
            Log.d("User session is still valid at the hourly check.")
            self._sessionTimer.start()

    # ─────────────────────────────────────────────── retranslate
    def retranslateUi(self, MainWindow5: QtWidgets.QMainWindow) -> None:
        """Update the window icon and title."""
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/qatch-icon.png")
        MainWindow5.setWindowIcon(QtGui.QIcon(icon_path))
        MainWindow5.setWindowTitle(
            _translate(
                "MainWindow5",
                "{} {} - Login".format(Constants.app_title, Constants.app_version),
            )
        )

    # ─────────────────────────────────────────────── error helpers
    def error_loggedout(self) -> None:
        """Show 'signed out' error (auto-clears after 10 s)."""
        self.kickErrorTimer()
        self.user_error.setText("You have been signed out")

    def error_invalid(self) -> None:
        """Show 'invalid credentials' error (auto-clears after 10 s)."""
        self.kickErrorTimer()
        self.user_error.setText("Invalid credentials")

    def error_expired(self) -> None:
        """Show 'session expired' error (persistent)."""
        self.user_error.setText("Your session has expired")

    def kickErrorTimer(self) -> None:
        """(Re-)start the 10-second error-clear timer."""
        if self._errorTimer.isActive():
            Log.d("Error Timer was restarted while running")
            self._errorTimer.stop()
        self._errorTimer.start(10000)

    # ─────────────────────────────────────────────── input helpers
    def text_transform(self) -> None:
        """Force the initials field to uppercase without re-firing textEdited."""
        text = self.user_initials.text()
        if text:
            self.user_initials.setText(text.upper())

    # ─────────────────────────────────────────────── sign-in
    def action_sign_in(self) -> None:
        """Validate credentials and open a user session on success."""
        if not self.user_initials.text() and not self.user_password.text():
            Log.d("No initials or password entered. Ignoring Sign-In request.")
            return

        initials = self.user_initials.text().upper()
        pwd = self.user_password.text()
        authenticated, filename, params = UserProfiles.auth(initials, pwd, UserRoles.ANY)
        if authenticated:
            Log.i(f"Welcome, {params[0]}! Your assigned role is {params[2].name}.")
            name, init, role = params[0], params[1], params[2].value
            self._sessionTimer.start()
        else:
            name, init, role = None, None, 0
        self.clear_form()

        if name is not None:
            self.parent.ControlsWin.username.setText(f"User: {name}")
            self.parent.ControlsWin.userrole = UserRoles(role)
            self.parent.ControlsWin.signinout.setText("&Sign Out")
            self.parent.ControlsWin.ui1.tool_User.setText(name)
            self.parent.AnalyzeProc.tool_User.setText(name)
            if self.parent.ControlsWin.userrole != UserRoles.ADMIN:
                self.parent.ControlsWin.manage.setText("&Change Password...")

            check_result = UserProfiles().check(self.parent.ControlsWin.userrole, UserRoles.CAPTURE)
            if check_result:
                self.parent.MainWin.ui0._set_run_mode(self.user_label)
            else:
                self.parent.MainWin.ui0._set_analyze_mode(self.user_label)

            if UserProfiles().check(self.parent.ControlsWin.userrole, UserRoles.ADMIN):
                enabled, error, expires = UserProfiles.checkDevMode()
                if enabled is not True and error is not False:
                    is_expired = expires != ""
                    from QATCH.common.userProfiles import (
                        UserConstants,
                        UserProfilesManager,
                    )

                    if PopUp.question(
                        self.parent,
                        "Developer Mode " + ("Expired" if is_expired else "Error"),
                        (
                            "<b>Developer Mode "
                            + ("has expired" if is_expired else "is invalid")
                            + " and is no longer active!</b><br/>"
                            + f"Renewal Period: Every {UserConstants.DEV_EXPIRE_LEN} days<br/><br/>"
                            + "Would you like to renew Developer Mode now?<br/><br/>"
                            + "<small>NOTE: This setting can be changed in the"
                            + ' "Manage Users" window.</small>'
                        ),
                    ):
                        temp_upm = UserProfilesManager(self.parent, name)
                        temp_upm.developerModeChk.setChecked(True)
                        Log.i("Developer Mode renewed!")
                    else:
                        Log.w("Developer Mode NOT renewed!")

            if hasattr(self.parent, "url_download"):
                delattr(self.parent, "url_download")
            QtCore.QTimer.singleShot(1, self.parent.start_download)
        else:
            self.error_invalid()

    # ─────────────────────────────────────────────── clear / caps-lock
    def clear_form(self) -> None:
        """Clear initials, password, and error fields; revert password visibility."""
        self.user_initials.clear()
        self.user_password.clear()
        self.user_error.clear()

        if self.password_shown:
            self.on_toggle_password_Action()

        if self.user_password.hasFocus():
            self.user_initials.setFocus()

    def update_caps_lock_state(self, caps_lock_state: bool) -> None:
        """Show or clear the Caps Lock warning in the info label."""
        if caps_lock_state:
            self.user_info.setText("Caps Lock is On")
        else:
            self.user_info.clear()
