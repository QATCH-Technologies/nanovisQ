import os
from PyQt5 import QtCore, QtGui, QtWidgets
from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles, UserRoles
from QATCH.core.constants import Constants
from QATCH.ui.components.floating_message_badge import FloatingMessageBadge
from QATCH.ui.components.glass_card import GlassCard
from QATCH.ui.components.sliding_panel import SlidingPanel
from QATCH.ui.widgets.login_centeral_widget import LoginCentralWidget

_CARD_W: int = 320
_INPUT_H: int = 34
_BTN_H: int = 38
_PAGE_H: int = 400

# Page indices — Create is LEFT of Sign In so it slides in from the left
_P_CREATE = 0
_P_SIGNIN = 1
_P_RECOVER = 2


class GlassLineEdit(QtWidgets.QLineEdit):
    """QLineEdit with a fully custom glass background and an animated shimmer border on focus.

    All background/border painting is owned by this widget so Qt's stylesheet
    cannot interfere.  Only typography (text, placeholder, cursor, trailing
    actions) is delegated back to the base class.
    """

    _R: float = _INPUT_H / 2.0  # pill radius — stays in sync with _INPUT_H

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._shimmer_t: float = 0.0
        self._focused: bool = False
        self._in_error: bool = False

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(12)  # ~83 fps — smooth sweep
        self._timer.timeout.connect(self._tick)

        self.setFrame(False)
        self.setAutoFillBackground(False)
        # Only define typography; background and border are painted manually.
        self.setStyleSheet("""
            QLineEdit {
                background: transparent;
                border: none;
                padding: 0px 15px;
                color: rgba(38, 48, 58, 230);
                font-size: 10pt;
                selection-background-color: transparent;
                selection-color: rgba(38, 48, 58, 230);
            }
            QLineEdit QToolButton { background: transparent; border: none; }
            QLineEdit QToolButton:hover {
                background: rgba(255, 255, 255, 55);
                border-radius: 12px;
            }
        """)

    # ── public ────────────────────────────────────────────────────────────────
    def set_error(self, on: bool) -> None:
        if on != self._in_error:
            self._in_error = on
            self.update()

    # ── animation ─────────────────────────────────────────────────────────────
    def _tick(self) -> None:
        self._shimmer_t = min(1.0, self._shimmer_t + 0.022)
        self.update()
        if self._shimmer_t >= 1.0:
            self._timer.stop()

    # ── Qt events ─────────────────────────────────────────────────────────────
    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusInEvent(event)
        self._focused = True
        self._in_error = False
        self._shimmer_t = 0.0
        self._timer.start()
        self.update()

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusOutEvent(event)
        self._focused = False
        self._timer.stop()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        r = self._R - 1.0
        rect = QtCore.QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        # ── 1. Glass background ───────────────────────────────────────────────
        if self._in_error:
            fill = QtGui.QColor(255, 220, 220, 68)
        elif self._focused:
            fill = QtGui.QColor(255, 255, 255, 100)
        else:
            fill = QtGui.QColor(255, 255, 255, 58)

        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QBrush(fill))
        p.drawRoundedRect(rect, r, r)

        # ── 2. Border ─────────────────────────────────────────────────────────
        p.setBrush(QtCore.Qt.NoBrush)

        if self._in_error:
            p.setPen(QtGui.QPen(QtGui.QColor(210, 55, 55, 150), 1.0))

        elif self._focused:
            t = self._shimmer_t
            W = float(self.width())
            grad = QtGui.QLinearGradient(0.0, 0.0, W, 0.0)

            if t < 1.0:
                # Bright peak sweeps left → right; trailing edge fades to accent
                spread = 0.30
                lo = QtGui.QColor(185, 218, 248, 115)  # settled accent
                hi = QtGui.QColor(255, 255, 255, 240)  # shimmer peak

                grad.setColorAt(0.0, lo)
                pre = max(0.0, t - spread)
                if pre > 0.0:
                    grad.setColorAt(pre, lo)
                grad.setColorAt(max(0.0, t - spread * 0.12), hi)
                grad.setColorAt(min(1.0, t + spread * 0.12), hi)
                post = min(1.0, t + spread)
                if post < 1.0:
                    grad.setColorAt(post, lo)
                grad.setColorAt(1.0, lo)
            else:
                # Settled: uniform soft accent
                c = QtGui.QColor(185, 218, 248, 130)
                grad.setColorAt(0.0, c)
                grad.setColorAt(1.0, c)

            p.setPen(QtGui.QPen(QtGui.QBrush(grad), 1.5))

        else:
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 105), 1.0))

        p.drawRoundedRect(rect, r, r)
        p.end()

        # Delegate text / placeholder / cursor / trailing-action rendering to Qt
        super().paintEvent(event)


class _SignInTransition(QtWidgets.QWidget):
    """Full-screen overlay that plays the sign-in success animation.

    Timeline (total ~700 ms, InOutCubic easing):
      t = 0.00 → 0.35  — dark veil rises while the login snapshot zooms in + fades
      t = 0.35 → 0.65  — veil holds dark; main UI starts loading underneath
      t = 0.65 → 1.00  — veil dissolves, revealing the newly loaded main UI

    The caller fires ``start_download`` independently via a short timer so the
    main window is ready by the time the overlay is fully transparent.
    """

    def __init__(
        self,
        login_window: QtWidgets.QMainWindow,
        duration_ms: int = 700,
    ) -> None:
        super().__init__(
            None,
            QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool,
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)

        # Grab the login window before anything changes
        self._snapshot: QtGui.QPixmap = login_window.grab()

        screen = QtWidgets.QApplication.screenAt(login_window.pos())
        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()
        self.setGeometry(screen.geometry())
        self.show()
        self.raise_()

        self._t: float = 0.0
        anim = QtCore.QPropertyAnimation(self, b"animProgress", self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(duration_ms)
        anim.setEasingCurve(QtCore.QEasingCurve.InOutCubic)
        anim.finished.connect(self.close)
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)

    # ── Animated property ─────────────────────────────────────────────────────
    @QtCore.pyqtProperty(float)
    def animProgress(self) -> float:  # type: ignore[override]
        return self._t

    @animProgress.setter  # type: ignore[override]
    def animProgress(self, val: float) -> None:
        self._t = val
        self.update()

    # ── Painting ──────────────────────────────────────────────────────────────
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        t = self._t
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        # Dark veil: ramps up to t=0.38, holds, then fades out by t=1.0
        if t < 0.38:
            veil_alpha = t / 0.38
        else:
            veil_alpha = max(0.0, 1.0 - (t - 0.38) / 0.62)
        p.fillRect(self.rect(), QtGui.QColor(10, 18, 34, int(veil_alpha * 215)))

        # Login snapshot: zooms gently in (1.0 → 1.08) and fades out over first 60%
        snap_alpha = max(0.0, 1.0 - t / 0.60)
        if snap_alpha > 0.0:
            scale = 1.0 + t * 0.10
            pm = self._snapshot
            nw = int(pm.width() * scale)
            nh = int(pm.height() * scale)
            cx = self.width() // 2
            cy = self.height() // 2
            dest = QtCore.QRect(cx - nw // 2, cy - nh // 2, nw, nh)
            p.setOpacity(snap_alpha)
            p.drawPixmap(dest, pm)

        p.end()


class _UserInfoProxy:
    """Compatibility shim: routes legacy user_info label calls to the floating badge."""

    def __init__(self, badge: FloatingMessageBadge, card) -> None:
        self._badge = badge
        self._card = card

    def clear(self) -> None:
        self._badge.clear()

    def setText(self, text: str) -> None:
        if text and text.strip():
            self._badge.show_message(text.strip(), is_error=False, parent_widget=self._card)
        else:
            self._badge.clear()

    def text(self) -> str:
        return ""

    # Absorb any other QLabel-like calls silently so legacy code never crashes.
    def __getattr__(self, name):
        return lambda *a, **kw: None


class _CapsWatcher(QtCore.QObject):
    """Event filter attached to the password field.

    Checks the real OS caps-lock state both on FocusIn (so the indicator
    appears as soon as the user clicks into the field) and on every KeyRelease
    (so toggling caps lock while typing is caught immediately).
    """

    def __init__(self, login_ui: "UILogin", parent: QtCore.QObject = None) -> None:
        super().__init__(parent)
        self._ui = login_ui

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() in (QtCore.QEvent.FocusIn, QtCore.QEvent.KeyRelease):
            self._ui.update_caps_lock_state(self._ui._check_caps_lock())
        return False  # never consume the event


class UILogin:
    # ── Setup ─────────────────────────────────────────────────────────────────
    def setup_ui(
        self,
        MainWindow5: QtWidgets.QMainWindow,
        parent: QtWidgets.QMainWindow,
    ) -> None:
        """Initialise all UI elements for the simplified login window."""
        self.parent = parent
        self.caps_lock_on = False
        self._login_window: QtWidgets.QMainWindow = MainWindow5  # used by transition

        global _CARD_W, _INPUT_H, _PAGE_H, _BTN_H
        _CARD_W = 320
        _INPUT_H = 34
        _BTN_H = 38
        _PAGE_H = 400

        # ── Window ────────────────────────────────────────────────────────────
        MainWindow5.setObjectName("MainWindow5")
        MainWindow5.setMinimumSize(QtCore.QSize(800, 500))
        MainWindow5.resize(800, 500)
        MainWindow5.setTabShape(QtWidgets.QTabWidget.Rounded)

        # ── Central widget ────────────────────────────────────────────────────
        self.centralwidget = LoginCentralWidget(MainWindow5)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        MainWindow5.setCentralWidget(self.centralwidget)

        MainWindow5.setStyleSheet("""
            QWidget {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            }
            #centralwidget { background: transparent; border: none; }
            #slidingPanel  { background: transparent; border: none; }
            #loginCard     { background: transparent; border: none; }
        """)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 0 — Sign In
        # ══════════════════════════════════════════════════════════════════════
        signInPage = QtWidgets.QWidget()
        signInPage.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        si = QtWidgets.QVBoxLayout(signInPage)
        si.setContentsMargins(28, 26, 28, 22)
        si.setSpacing(10)
        si.setAlignment(QtCore.Qt.AlignTop)

        # ── Logo ──────────────────────────────────────────────────────────────
        logo_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "qatch-icon.png")
        logo_pm = QtGui.QPixmap(logo_path)
        if not logo_pm.isNull():
            logo_pm = logo_pm.scaled(
                54,
                54,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        logoLabel = QtWidgets.QLabel()
        logoLabel.setAlignment(QtCore.Qt.AlignCenter)
        logoLabel.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        if not logo_pm.isNull():
            logoLabel.setPixmap(logo_pm)
            logoLabel.setFixedSize(54, 54)
        si.addWidget(logoLabel, alignment=QtCore.Qt.AlignCenter)
        si.addSpacing(2)

        # Title
        siTitle = QtWidgets.QLabel("Sign In")
        siTitle.setObjectName("cardTitle")
        siTitle.setAlignment(QtCore.Qt.AlignCenter)
        siTitle.setStyleSheet("color: rgba(50, 55, 65, 220); font-size: 12pt; font-weight: 700;")
        si.addWidget(siTitle)
        si.addSpacing(4)

        # Username field
        self.user_username = GlassLineEdit()
        self.user_username.setObjectName("user_username")
        self.user_username.setFixedHeight(_INPUT_H)
        self.user_username.setPlaceholderText("Username")
        self.user_username.textChanged.connect(lambda: self.user_username.set_error(False))
        self.user_username.returnPressed.connect(lambda: self.user_password.setFocus())
        si.addWidget(self.user_username)

        # Alias: legacy code references user_initials — username serves the same role
        self.user_initials = self.user_username

        # Password field
        self.user_password = GlassLineEdit()
        self.user_password.setObjectName("user_password")
        self.user_password.setFixedHeight(_INPUT_H)
        self.user_password.setPlaceholderText("Password")
        self.user_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.user_password.textChanged.connect(lambda: self.user_password.set_error(False))
        self.user_password.returnPressed.connect(self.action_sign_in)
        self.user_password.installEventFilter(MainWindow5)

        # Eye (show/hide password) action
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

        # Caps Lock indicator — always occupies its row; text is blank when off
        self.caps_indicator = QtWidgets.QLabel("")
        self.caps_indicator.setObjectName("capsIndicator")
        self.caps_indicator.setAlignment(QtCore.Qt.AlignCenter)
        self.caps_indicator.setFixedHeight(16)
        self.caps_indicator.setStyleSheet(
            "color: rgba(200, 130, 30, 235); font-size: 7.5pt; font-weight: 600;"
        )
        si.addWidget(self.caps_indicator, alignment=QtCore.Qt.AlignCenter)

        # Sign In button (primary / accent)
        self.sign_in_btn = QtWidgets.QPushButton("Sign In")
        self.sign_in_btn.setObjectName("signInBtn")
        self.sign_in_btn.setFixedHeight(_BTN_H)
        self.sign_in_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.sign_in_btn.setStyleSheet(self._make_primary_btn_style())
        self.sign_in_btn.clicked.connect(self.action_sign_in)
        si.addWidget(self.sign_in_btn)

        si.addStretch()

        # Create Account link
        createAccountLbl = QtWidgets.QLabel("Create Account")
        createAccountLbl.setObjectName("createAccountLink")
        createAccountLbl.setAlignment(QtCore.Qt.AlignCenter)
        createAccountLbl.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        createAccountLbl.setStyleSheet("""
            QLabel {
                color: rgba(10, 163, 230, 210);
                font-size: 9pt; font-weight: 500;
            }
            QLabel:hover { color: rgba(10, 130, 200, 255); text-decoration: underline; }
        """)
        createAccountLbl.mousePressEvent = lambda _e: self._slide_to(_P_CREATE)
        si.addWidget(createAccountLbl)

        # Forgot Password link
        forgotPasswordLbl = QtWidgets.QLabel("Forgot Password?")
        forgotPasswordLbl.setObjectName("forgotPassword")
        forgotPasswordLbl.setAlignment(QtCore.Qt.AlignCenter)
        forgotPasswordLbl.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        forgotPasswordLbl.setStyleSheet("""
            QLabel {
                color: rgba(100, 110, 120, 180);
                font-size: 9pt; font-weight: 500;
            }
            QLabel:hover { color: rgba(60, 60, 60, 220); text-decoration: underline; }
        """)
        forgotPasswordLbl.mousePressEvent = lambda _e: self._slide_to(_P_RECOVER)
        si.addWidget(forgotPasswordLbl)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 1 — Forgot Password
        # ══════════════════════════════════════════════════════════════════════
        recoverPage = QtWidgets.QWidget()
        recoverPage.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        rec = QtWidgets.QVBoxLayout(recoverPage)
        rec.setContentsMargins(28, 18, 28, 18)
        rec.setSpacing(10)
        rec.setAlignment(QtCore.Qt.AlignTop)

        rec.addWidget(
            self._make_back_btn("← Back to Sign In", lambda: self._slide_to(_P_SIGNIN)),
            alignment=QtCore.Qt.AlignLeft,
        )

        recTitle = QtWidgets.QLabel("Reset Password")
        recTitle.setObjectName("recoverTitle")
        recTitle.setAlignment(QtCore.Qt.AlignCenter)
        recTitle.setStyleSheet("color: rgba(50, 55, 65, 220); font-size: 12pt; font-weight: 700;")
        rec.addWidget(recTitle)

        recInfo = QtWidgets.QLabel(
            "Enter your email address and we'll\nsend you a password reset link."
        )
        recInfo.setAlignment(QtCore.Qt.AlignCenter)
        recInfo.setWordWrap(True)
        recInfo.setStyleSheet("color: rgba(100, 110, 120, 220); font-size: 8.5pt;")
        rec.addWidget(recInfo)

        self.recoverEmail = GlassLineEdit()
        self.recoverEmail.setObjectName("recoverEmail")
        self.recoverEmail.setPlaceholderText("Email Address")
        self.recoverEmail.setFixedHeight(_INPUT_H)
        rec.addWidget(self.recoverEmail)

        self.sendResetBtn = QtWidgets.QPushButton("Send Reset Link")
        self.sendResetBtn.setObjectName("sendResetBtn")
        self.sendResetBtn.setFixedHeight(_BTN_H)
        self.sendResetBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.sendResetBtn.setStyleSheet(self._make_primary_btn_style())
        self.sendResetBtn.clicked.connect(self._on_send_reset)
        rec.addWidget(self.sendResetBtn)

        self.recoverStatus = QtWidgets.QLabel("")
        self.recoverStatus.setAlignment(QtCore.Qt.AlignCenter)
        self.recoverStatus.setWordWrap(True)
        self.recoverStatus.setFixedHeight(34)
        self.recoverStatus.setStyleSheet(
            "color: rgba(46, 139, 87, 220); font-size: 8.5pt; font-weight: 500;"
        )
        rec.addWidget(self.recoverStatus)
        rec.addStretch()

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 2 — Create Account
        # ══════════════════════════════════════════════════════════════════════
        createPage = QtWidgets.QWidget()
        createPage.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        cr = QtWidgets.QVBoxLayout(createPage)
        cr.setContentsMargins(28, 18, 28, 18)
        cr.setSpacing(8)
        cr.setAlignment(QtCore.Qt.AlignTop)

        cr.addWidget(
            self._make_back_btn("← Back to Sign In", lambda: self._slide_to(_P_SIGNIN)),
            alignment=QtCore.Qt.AlignLeft,
        )

        crTitle = QtWidgets.QLabel("Create Account")
        crTitle.setAlignment(QtCore.Qt.AlignCenter)
        crTitle.setStyleSheet("color: rgba(50, 55, 65, 220); font-size: 12pt; font-weight: 700;")
        cr.addWidget(crTitle)

        self.newUsername = GlassLineEdit()
        self.newUsername.setPlaceholderText("Username")
        self.newUsername.setFixedHeight(_INPUT_H)
        cr.addWidget(self.newUsername)

        self.newPassword = GlassLineEdit()
        self.newPassword.setPlaceholderText("Password")
        self.newPassword.setEchoMode(QtWidgets.QLineEdit.Password)
        self.newPassword.setFixedHeight(_INPUT_H)
        cr.addWidget(self.newPassword)

        self.confirmPassword = GlassLineEdit()
        self.confirmPassword.setPlaceholderText("Confirm Password")
        self.confirmPassword.setEchoMode(QtWidgets.QLineEdit.Password)
        self.confirmPassword.setFixedHeight(_INPUT_H)
        self.confirmPassword.returnPressed.connect(self._on_create_account)
        cr.addWidget(self.confirmPassword)

        createBtn = QtWidgets.QPushButton("Create Account")
        createBtn.setObjectName("createBtn")
        createBtn.setFixedHeight(_BTN_H)
        createBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        createBtn.setStyleSheet(self._make_primary_btn_style())
        createBtn.clicked.connect(self._on_create_account)
        cr.addWidget(createBtn)

        self.createStatus = QtWidgets.QLabel("")
        self.createStatus.setAlignment(QtCore.Qt.AlignCenter)
        self.createStatus.setWordWrap(True)
        self.createStatus.setFixedHeight(30)
        self.createStatus.setStyleSheet("font-size: 8.5pt;")
        cr.addWidget(self.createStatus)
        cr.addStretch()

        # ══════════════════════════════════════════════════════════════════════
        # Sliding panel + Glass Card
        # ══════════════════════════════════════════════════════════════════════
        self._slider = SlidingPanel(_CARD_W)
        self._slider.setObjectName("slidingPanel")
        self._slider.setStyleSheet(
            "QWidget#slidingPanel { background: transparent; border: none; }"
        )
        # Page layout: [0=Create, 1=SignIn(default), 2=Recover]
        # Create slides in from the LEFT; Recover slides in from the RIGHT.
        self._slider.add_page(createPage)
        self._slider.add_page(signInPage)
        self._slider.add_page(recoverPage)
        self._slider.setFixedHeight(_PAGE_H)

        def _init_slider():
            self._slider.finalize(_PAGE_H)
            # Position instantly at Sign In (page 1) with no animation
            self._slider._inner.move(-_CARD_W, 0)

        QtCore.QTimer.singleShot(0, _init_slider)

        self.loginCard = GlassCard(self.centralwidget)
        self.loginCard.setObjectName("loginCard")
        self.loginCard.setAttribute(QtCore.Qt.WA_StyledBackground, False)
        self.loginCard.setContentsMargins(0, 0, 0, 0)

        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(44)
        shadow.setOffset(0, 10)
        shadow.setColor(QtGui.QColor(15, 40, 70, 90))
        self.loginCard.setGraphicsEffect(shadow)

        card_vbox = QtWidgets.QVBoxLayout(self.loginCard)
        card_vbox.setContentsMargins(0, 0, 0, 0)
        card_vbox.setSpacing(0)
        card_vbox.addWidget(self._slider)

        v_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        v_layout.setAlignment(QtCore.Qt.AlignCenter)
        v_layout.addStretch(2)
        v_layout.addSpacing(20)
        v_layout.addWidget(self.loginCard, alignment=QtCore.Qt.AlignCenter)
        v_layout.addStretch(3)

        # Floating badge for error / info messages
        self.floating_badge = FloatingMessageBadge(
            MainWindow5,
            os.path.join(Architecture.get_path(), "QATCH", "icons", "clear.svg"),
        )

        # user_info proxy — legacy code calls .user_info.clear() / .setText(); route to badge
        self.user_info = _UserInfoProxy(self.floating_badge, self.loginCard)

        # Caps-lock watcher — checks real OS state on focus and every key release
        self._caps_watcher = _CapsWatcher(self, MainWindow5)
        self.user_password.installEventFilter(self._caps_watcher)

        self._sessionTimer = QtCore.QTimer()
        self._sessionTimer.setSingleShot(True)
        self._sessionTimer.timeout.connect(self.check_user_session)
        self._sessionTimer.setInterval(1000 * 60 * 60)

        QtCore.QTimer.singleShot(0, lambda: self.centralwidget.set_background_pixmap())

    # ── Style helpers ─────────────────────────────────────────────────────────
    def _make_input_style(self, error: bool = False) -> str:
        r = _INPUT_H // 2
        if error:
            return f"""
                QLineEdit {{
                    background-color: rgba(255, 230, 230, 160);
                    border: 1.5px solid rgba(230, 50, 50, 200);
                    border-style: solid;
                    border-radius: {r}px;
                    padding: 0px 15px;
                    color: rgba(200, 30, 30, 255);
                    font-size: 10pt;
                    selection-background-color: rgba(230, 50, 50, 80);
                }}
                QLineEdit:focus {{
                    background-color: rgba(255, 245, 245, 255);
                    border: 2px solid rgba(255, 50, 50, 255);
                }}
                QLineEdit QToolButton {{ background: transparent; border: none; }}
                QLineEdit QToolButton:hover {{ background: rgba(230, 50, 50, 20); border-radius: 12px; }}
            """
        return f"""
            QLineEdit {{
                background-color: rgba(255, 255, 255, 72);
                border: 1px solid rgba(255, 255, 255, 130);
                border-style: solid;
                border-radius: {r}px;
                padding: 0px 15px;
                color: rgba(40, 50, 60, 230);
                font-size: 10pt;
                selection-background-color: rgba(10, 163, 230, 80);
            }}
            QLineEdit:hover {{
                background-color: rgba(255, 255, 255, 110);
                border-color: rgba(255, 255, 255, 200);
            }}
            QLineEdit:focus {{
                background-color: rgba(255, 255, 255, 145);
                border: 1.5px solid rgba(10, 163, 230, 140);
            }}
            QLineEdit QToolButton {{ background: transparent; border: none; }}
            QLineEdit QToolButton:hover {{ background: rgba(255, 255, 255, 60); border-radius: 12px; }}
        """

    def _make_primary_btn_style(self) -> str:
        r = _BTN_H // 2
        return f"""
            QPushButton {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0   rgba(60, 185, 250, 145),
                    stop:0.5 rgba(10, 155, 225, 115),
                    stop:1   rgba(8,  130, 200, 100)
                );
                border-top:    1.5px solid rgba(255, 255, 255, 190);
                border-left:   1px   solid rgba(255, 255, 255, 130);
                border-right:  1px   solid rgba(255, 255, 255, 100);
                border-bottom: 1px   solid rgba(180, 215, 240, 90);
                border-radius: {r}px;
                color: rgba(255, 255, 255, 235);
                font-size: 10pt;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0   rgba(80, 200, 255, 175),
                    stop:0.5 rgba(20, 168, 238, 145),
                    stop:1   rgba(10, 142, 215, 128)
                );
                border-top: 1.5px solid rgba(255, 255, 255, 220);
            }}
            QPushButton:pressed {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0   rgba(8,  130, 200, 140),
                    stop:1   rgba(5,  105, 170, 120)
                );
                border-top: 1px solid rgba(255, 255, 255, 130);
            }}
            QPushButton:disabled {{
                background: rgba(180, 200, 215, 80);
                color: rgba(255, 255, 255, 120);
                border: 1px solid rgba(255, 255, 255, 80);
            }}
        """

    @staticmethod
    def _make_back_btn(text: str, callback) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        btn.setObjectName("backBtn")
        btn.setFixedHeight(24)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: rgba(100, 110, 120, 200);
                border: none;
                font-size: 8.5pt;
                font-weight: 600;
                text-align: left;
            }
            QPushButton:hover { color: rgba(60, 60, 60, 220); text-decoration: underline; }
        """)
        btn.clicked.connect(callback)
        return btn

    # ── Navigation ────────────────────────────────────────────────────────────
    def _slide_to(self, page_idx: int) -> None:
        """Animate to the given page index and reset any stale state."""
        self._slider.slide_to(page_idx)
        if page_idx == _P_SIGNIN:
            # Reset recover page
            self.recoverEmail.clear()
            self.recoverStatus.clear()
            self.sendResetBtn.setEnabled(True)
            # Reset create page
            self.newUsername.clear()
            self.newPassword.clear()
            self.confirmPassword.clear()
            self.createStatus.clear()

    # ── Password show/hide ────────────────────────────────────────────────────
    def on_toggle_password_Action(self) -> None:
        if not self.password_shown:
            self.user_password.setEchoMode(QtWidgets.QLineEdit.Normal)
            self.password_shown = True
            self.togglepasswordAction.setIcon(self.hiddenIcon)
        else:
            self.user_password.setEchoMode(QtWidgets.QLineEdit.Password)
            self.password_shown = False
            self.togglepasswordAction.setIcon(self.visibleIcon)

    # ── Caps Lock ─────────────────────────────────────────────────────────────
    @staticmethod
    def _check_caps_lock() -> bool:
        """Return the real OS caps-lock state, cross-platform best-effort."""
        # Windows — fastest, most reliable
        try:
            import ctypes

            return bool(ctypes.WinDLL("User32.dll").GetKeyState(0x14) & 1)
        except Exception:
            pass
        # Linux (X11) — via xset
        try:
            import subprocess

            out = subprocess.run(["xset", "q"], capture_output=True, text=True, timeout=0.3).stdout
            return "Caps Lock:   on" in out
        except Exception:
            pass
        # macOS — via ioreg
        try:
            import subprocess

            out = subprocess.run(
                ["ioreg", "-n", "IOHIDSystem", "-l"],
                capture_output=True,
                text=True,
                timeout=0.3,
            ).stdout
            for line in out.splitlines():
                if "HIDCapsLockState" in line:
                    return "= 1" in line
        except Exception:
            pass
        return False

    def update_caps_lock_state(self, caps_lock_on: bool) -> None:
        """Called externally (via eventFilter) whenever Caps Lock state changes."""
        self.caps_lock_on = caps_lock_on
        self.caps_indicator.setText("⇪  Caps Lock is On" if caps_lock_on else "")

    # ── Forgot Password flow ──────────────────────────────────────────────────
    def _on_send_reset(self) -> None:
        email = self.recoverEmail.text().strip()
        if not email:
            self.recoverStatus.setStyleSheet(
                "color: rgba(200, 30, 30, 230); font-size: 8.5pt; font-weight: 500;"
            )
            self.recoverStatus.setText("Please enter your email address.")
            return
        Log.i(f"Password reset requested for: {email}")
        self.recoverStatus.setStyleSheet(
            "color: rgba(46, 139, 87, 220); font-size: 8.5pt; font-weight: 500;"
        )
        self.recoverStatus.setText(
            "If an account exists for that address,\na reset link has been sent."
        )
        self.sendResetBtn.setEnabled(False)

    # ── Create Account flow ───────────────────────────────────────────────────
    def _on_create_account(self) -> None:
        username = self.newUsername.text().strip()
        password = self.newPassword.text()
        confirm = self.confirmPassword.text()

        error_style = "color: rgba(200, 30, 30, 230); font-size: 8.5pt; font-weight: 500;"
        ok_style = "color: rgba(46, 139, 87, 220); font-size: 8.5pt; font-weight: 500;"

        if not username or not password:
            self.createStatus.setStyleSheet(error_style)
            self.createStatus.setText("All fields are required.")
            return
        if password != confirm:
            self.createStatus.setStyleSheet(error_style)
            self.createStatus.setText("Passwords do not match.")
            return

        # Delegate to backend (UserProfiles.create_new_user or similar)
        Log.i(f"Create account requested for: {username}")
        self.createStatus.setStyleSheet(ok_style)
        self.createStatus.setText("Account created! Please sign in.")

    # ── Sign In ───────────────────────────────────────────────────────────────
    def action_sign_in(self) -> None:
        username = self.user_username.text().strip()
        if not username:
            self._shake_widget(self.user_username)
            self.user_username.set_error(True)
            self.floating_badge.show_message(
                "Please enter your username", is_error=True, parent_widget=self.loginCard
            )
            return

        if not self.user_password.text():
            self._shake_widget(self.user_password)
            self.user_password.set_error(True)
            self.floating_badge.show_message(
                "Password required", is_error=True, parent_widget=self.loginCard
            )
            return

        pwd = self.user_password.text()
        authenticated, filename, params = UserProfiles.auth(username, pwd, UserRoles.ANY)

        if authenticated:
            Log.i(f"Welcome, {params[0]}! Role: {params[2].name}.")
            name, init, role = params[0], params[1], params[2].value
            self._sessionTimer.start()
        else:
            name, init, role = None, None, 0

        self._clear_credentials()

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
                self.parent.MainWin.ui0._set_run_mode(None)
            else:
                self.parent.MainWin.ui0._set_analyze_mode(None)

            if hasattr(self.parent, "url_download"):
                delattr(self.parent, "url_download")

            # Play the zoom-fade transition; kick off start_download at ~40%
            # through so the main UI is ready when the veil lifts (~280 ms in).
            _SignInTransition(self._login_window, duration_ms=700)
            QtCore.QTimer.singleShot(280, self.parent.start_download)
        else:
            self.error_invalid()

    # ── Clear helpers ─────────────────────────────────────────────────────────
    def _clear_credentials(self) -> None:
        """Clear the username and password fields."""
        self.user_password.clear()
        if self.password_shown:
            self.on_toggle_password_Action()

    def clear_form(self) -> None:
        """Full form reset (called on Escape or sign-out)."""
        self.user_username.clear()
        self.user_username.set_error(False)
        self._clear_credentials()
        self.user_password.set_error(False)
        self.caps_indicator.setText("")
        self._slide_to(_P_SIGNIN)

    # ── Error display ─────────────────────────────────────────────────────────
    def error_invalid(self, message: str = "Invalid Credentials") -> None:
        self.user_password.set_error(True)
        self._shake_widget(self.user_password)
        self.user_password.clear()
        self.user_password.setFocus()
        self.floating_badge.show_message(message, is_error=True, parent_widget=self.loginCard)

    def error_loggedout(self) -> None:
        self.user_password.clear()
        self.user_password.set_error(False)
        self.floating_badge.show_message(
            "You have been signed out", is_error=True, parent_widget=self.loginCard
        )

    def error_expired(self) -> None:
        self.user_password.clear()
        self.user_password.set_error(False)
        self.floating_badge.show_message(
            "Your session has expired", is_error=True, parent_widget=self.loginCard
        )

    def show_signout_message(self) -> None:
        self.floating_badge.show_message(
            "You have been signed out.", is_error=True, parent_widget=self.loginCard
        )

    # ── Session ───────────────────────────────────────────────────────────────
    def check_user_session(self) -> None:
        valid, _ = UserProfiles().session_info()
        if not valid:
            if self.parent.ControlsWin.userrole != UserRoles.NONE:
                Log.w("User session has expired.")
                self.parent.ControlsWin.set_user_profile()
        else:
            Log.d("User session valid at hourly check.")
            self._sessionTimer.start()

    # ── Retranslate ───────────────────────────────────────────────────────────
    def retranslateUi(self, MainWindow5: QtWidgets.QMainWindow) -> None:
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/qatch-icon.png")
        MainWindow5.setWindowIcon(QtGui.QIcon(icon_path))
        MainWindow5.setWindowTitle(
            _translate(
                "MainWindow5",
                "{} {} - Login".format(Constants.app_title, Constants.app_version),
            )
        )

    # ── Shake animation ───────────────────────────────────────────────────────
    def _shake_widget(self, widget: QtWidgets.QWidget) -> None:
        """Rapid left-right jiggle to signal an error."""
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
        self._shake_anim.start(QtCore.QPropertyAnimation.DeleteWhenStopped)
