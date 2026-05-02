import os
from typing import List, Optional, Tuple
from PyQt5 import QtCore, QtGui, QtWidgets
from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles, UserRoles
from QATCH.core.constants import Constants
from QATCH.ui.popUp import PopUp
from QATCH.ui.components.floating_message_badge import FloatingMessageBadge
from QATCH.ui.components.glass_card import GlassCard
from QATCH.ui.components.sliding_panel import SlidingPanel
from QATCH.ui.widgets.login_centeral_widget import LoginCentralWidget
from QATCH.ui.dialogs.switch_user_dialog import SwitchUserDialog

_CARD_W: int = 290
_AVATAR_D: int = 70
_INPUT_H: int = 26
_BTN_H: int = 28
_PAGE_H: int = 354


class UILogin:
    # ── setup ─────────────────────────────────────────────────────────────────
    def setup_ui(
        self,
        MainWindow5: QtWidgets.QMainWindow,
        parent: QtWidgets.QMainWindow,
    ) -> None:
        """Initialise and arrange all UI elements for the login window."""
        self.parent = parent
        self.caps_lock_on = False

        global _AVATAR_D, _CARD_W, _INPUT_H, _PAGE_H, _BTN_H
        _AVATAR_D = 70
        _CARD_W = 320
        _INPUT_H = 34
        _PAGE_H = 300
        _BTN_H = 38

        # ── Window basics ──────────────────────────────────────────────────────
        MainWindow5.setObjectName("MainWindow5")
        MainWindow5.setMinimumSize(QtCore.QSize(1000, 500))
        MainWindow5.resize(500, 500)
        MainWindow5.setTabShape(QtWidgets.QTabWidget.Rounded)

        # ── Custom central widget ──────────────────────────────────────────────
        self.centralwidget = LoginCentralWidget(MainWindow5)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        self.centralwidget.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        MainWindow5.setCentralWidget(self.centralwidget)

        # ── INJECT MASTER STYLESHEET ───────────────────────────────────────────
        MASTER_QSS = f"""
            QWidget {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }}
            
            #centralwidget {{ background: transparent; border: none; }}
            #slidingPanel {{ background: transparent; }}

            /* GlassCard renders its own background, border, and rounding in paintEvent. */
            #loginCard {{ background: transparent; border: none; }}

            #user_welcome {{ color: rgba(60, 60, 60, 230); font-size: 15pt; font-weight: 700; }}
        """
        MainWindow5.setStyleSheet(MASTER_QSS)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 0 — Sign-In
        # ══════════════════════════════════════════════════════════════════════
        signInPage = QtWidgets.QWidget()
        signInPage.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        si = QtWidgets.QVBoxLayout(signInPage)
        si.setContentsMargins(28, 22, 28, 18)
        si.setSpacing(9)
        si.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)

        cardTitle = QtWidgets.QLabel("Sign In")
        cardTitle.setObjectName("cardTitle")
        cardTitle.setAlignment(QtCore.Qt.AlignCenter)
        cardTitle.setStyleSheet("color: rgba(60, 60, 60, 220); font-size: 11pt; font-weight: 700;")
        si.addWidget(cardTitle)

        # ── Avatar ─────────────────────────────────────────────────────────────
        avatarOuter = QtWidgets.QWidget()
        avatarOuter.setFixedSize(_AVATAR_D, _AVATAR_D)
        avatarOuter.setContentsMargins(0, 0, 0, 0)

        self.userAvatarBtn = QtWidgets.QPushButton("", avatarOuter)
        self.userAvatarBtn.setObjectName("userAvatarBtn")
        self.userAvatarBtn.setFixedSize(_AVATAR_D, _AVATAR_D)
        self.userAvatarBtn.move(0, 0)
        self.userAvatarBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.userAvatarBtn.setToolTip("Click to switch user account")

        self.userAvatarBtn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(229, 229, 229, 100);
                border-radius: {_AVATAR_D // 2}px; 
                border: 2px solid transparent;
            }}
            QPushButton:hover {{
                background-color: rgba(210, 215, 220, 150);
                border: 2px solid rgba(255, 255, 255, 150);
            }}
            QPushButton[hasIcon="true"] {{
                background-color: transparent;
                border: 2px solid rgba(255, 255, 255, 180);
            }}
            QPushButton[hasIcon="true"]:hover {{
                border-color: rgba(255, 255, 255, 240);
                background-color: rgba(255, 255, 255, 40);
            }}
        """)
        self.userAvatarBtn.clicked.connect(self._show_user_switch_dialog)
        si.addWidget(avatarOuter, alignment=QtCore.Qt.AlignCenter)

        # ── Selected user label ────────────────────────────────────────────────
        self.user_label = QtWidgets.QLabel("Select a User")
        self.user_label.setObjectName("user_label")
        self.user_label.setAlignment(QtCore.Qt.AlignCenter)
        self.user_label.setFixedHeight(18)
        self.user_label.setProperty("placeholder", True)
        self.user_label.setStyleSheet(
            "color: rgba(60, 60, 60, 200); font-size: 8.5pt; font-weight: 600;"
        )
        si.addWidget(self.user_label, alignment=QtCore.Qt.AlignCenter)

        si.addSpacing(4)

        self.user_initials = QtWidgets.QLineEdit()
        self.user_initials.setObjectName("user_initials")
        self.user_initials.setMaxLength(4)
        self.user_initials.setVisible(False)

        # ── Password + compact sign-in row ──────────────────────────────────────
        credentialsRow = QtWidgets.QWidget()
        credentialsRow.setObjectName("credentialsRow")
        credentialsRow.setFixedWidth(_CARD_W - 56)
        credentialsRow.setFixedHeight(_INPUT_H)

        credentials = QtWidgets.QHBoxLayout(credentialsRow)
        credentials.setContentsMargins(0, 0, 0, 0)
        credentials.setSpacing(8)

        # 1. DIRECTLY STYLED PASSWORD FIELD
        self.user_password = QtWidgets.QLineEdit()
        self.user_password.setObjectName("user_password")
        self.user_password.setFixedHeight(_INPUT_H)
        self.user_password.setPlaceholderText("Password")
        self.user_password.setEchoMode(QtWidgets.QLineEdit.Password)

        # ── NEW: Persistent Glass Border & Neutral Selection ──
        # ── STORE STYLES FOR STATE SWAPPING ──

        self._pw_style_normal = f"""
            QLineEdit {{
                background-color: rgba(250, 252, 255, 160);
                border: 1.5px solid rgba(180, 195, 210, 180);
                border-style: solid;
                border-radius: {_INPUT_H // 2}px; 
                padding: 0px 15px;
                color: rgba(40, 50, 60, 240);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 10pt;
                selection-background-color: rgba(10, 163, 230, 100);
            }}
            QLineEdit:hover {{ background-color: rgba(255, 255, 255, 200); border-color: rgba(10, 163, 230, 150); }}
            QLineEdit:focus {{ background-color: rgba(255, 255, 255, 255); border: 2px solid #0AA3E6; outline: none; }}
            QLineEdit QToolButton {{ background: transparent; border: none; margin: 0px; padding: 0px; }}
            QLineEdit QToolButton:hover {{ background: rgba(10, 163, 230, 20); border-radius: 12px; }}
        """

        self._pw_style_error = f"""
            QLineEdit {{
                background-color: rgba(255, 230, 230, 160); /* Light red frosted tint */
                border: 1.5px solid rgba(230, 50, 50, 200); /* Glassy red border */
                border-style: solid;
                border-radius: {_INPUT_H // 2}px; 
                padding: 0px 15px;
                color: rgba(200, 30, 30, 255);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 10pt;
                selection-background-color: rgba(230, 50, 50, 80);
            }}
            QLineEdit:focus {{ background-color: rgba(255, 245, 245, 255); border: 2px solid rgba(255, 50, 50, 255); outline: none; }}
            QLineEdit QToolButton {{ background: transparent; border: none; margin: 0px; padding: 0px; }}
            QLineEdit QToolButton:hover {{ background: rgba(230, 50, 50, 20); border-radius: 12px; }}
        """

        # Apply normal style to start
        self.user_password.setStyleSheet(self._pw_style_normal)

        # Revert back to normal automatically when the user starts typing a correction
        self.user_password.textChanged.connect(
            lambda: self.user_password.setStyleSheet(self._pw_style_normal)
        )

        self.user_password.installEventFilter(MainWindow5)
        self.user_password.returnPressed.connect(self.action_sign_in)
        credentials.addWidget(self.user_password, stretch=1)

        # 2. DIRECTLY STYLED SIGN-IN BUTTON
        self.sign_in = QtWidgets.QPushButton()
        self.sign_in.setObjectName("sign_in")
        self.sign_in.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(229, 229, 229, 150);
                border: 1.5px solid rgba(255, 255, 255, 200);
                border-style: solid;
                border-radius: {_INPUT_H // 2}px; 
            }}
            QPushButton:hover {{ background-color: rgba(210, 215, 220, 180); }}
            QPushButton:pressed {{ background-color: rgba(190, 200, 210, 200); }}
        """)

        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "right-arrow.svg")
        self.sign_in.setIcon(QtGui.QIcon(icon_path))
        self.sign_in.setIconSize(QtCore.QSize(24, 24))
        self.sign_in.setFixedSize(_INPUT_H, _INPUT_H)
        self.sign_in.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.sign_in.setToolTip("Sign in")
        self.sign_in.setAccessibleName("Sign In")
        self.sign_in.clicked.connect(self.action_sign_in)
        self.sign_in.installEventFilter(MainWindow5)

        credentials.addWidget(self.sign_in, alignment=QtCore.Qt.AlignVCenter)
        si.addWidget(credentialsRow, alignment=QtCore.Qt.AlignCenter)

        # ── CAPS LOCK INDICATOR ────────────────────────────────────────────────
        self.floating_badge = FloatingMessageBadge(MainWindow5)

        # ── REMEMBER ME CHECKBOX ───────────────────────────────────────────────
        self.remember_me = QtWidgets.QCheckBox("Remember me")
        self.remember_me.setObjectName("rememberMe")
        self.remember_me.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # Inline glassy style for the checkbox
        self.remember_me.setStyleSheet("""
            QCheckBox {
                color: rgba(100, 110, 120, 200);
                font-size: 8.5pt;
                font-weight: 500;
                spacing: 8px; /* Space between box and text */
            }
            QCheckBox:hover { color: rgba(60, 60, 60, 220); }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border-radius: 4px;
                border: 1px solid rgba(180, 195, 210, 180);
                background-color: rgba(255, 255, 255, 140);
            }
            QCheckBox::indicator:hover {
                border-color: #0AA3E6;
                background-color: rgba(255, 255, 255, 220);
            }
            QCheckBox::indicator:checked {
                background-color: #0AA3E6;
                border: 1px solid #0AA3E6;
            }
        """)
        si.addWidget(self.remember_me, alignment=QtCore.Qt.AlignCenter)

        # ── ERROR LABEL ────────────────────────────────────────────────────────
        self.user_error = QtWidgets.QLabel("")
        self.user_error = QtWidgets.QLabel("")
        self.user_error.setObjectName("user_error")
        self.user_error.setFixedHeight(16)
        si.addWidget(self.user_error, alignment=QtCore.Qt.AlignCenter)

        si.addSpacing(10)
        self.forgotPassword = QtWidgets.QLabel("Forgot Password?")
        self.forgotPassword.setObjectName("forgotPassword")
        self.forgotPassword.setAlignment(QtCore.Qt.AlignCenter)
        self.forgotPassword.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.forgotPassword.setFixedHeight(20)
        self.forgotPassword.setStyleSheet("""
            QLabel { color: rgba(100, 110, 120, 180); font-size: 9pt; font-weight: 500; }
            QLabel:hover { color: rgba(60, 60, 60, 220); text-decoration: underline; }
        """)
        self.forgotPassword.mousePressEvent = lambda _e: self._slide_to_recover()
        si.addWidget(self.forgotPassword)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 1 — Recover Password
        # ══════════════════════════════════════════════════════════════════════
        recoverPage = QtWidgets.QWidget()
        recoverPage.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        rec = QtWidgets.QVBoxLayout(recoverPage)
        rec.setContentsMargins(28, 18, 28, 18)
        rec.setSpacing(9)
        rec.setAlignment(QtCore.Qt.AlignTop)

        backBtn = QtWidgets.QPushButton("← Back to Sign In")
        backBtn.setObjectName("backBtn")
        backBtn.setFixedHeight(24)
        backBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        backBtn.setStyleSheet("""
            QPushButton { background: transparent; color: rgba(100, 110, 120, 200); border: none; font-size: 8.5pt; font-weight: 600; text-align: left; }
            QPushButton:hover { color: rgba(60, 60, 60, 220); text-decoration: underline; }
        """)
        backBtn.clicked.connect(self._slide_to_signin)
        rec.addWidget(backBtn, alignment=QtCore.Qt.AlignLeft)

        recoverTitle = QtWidgets.QLabel("Reset Password")
        recoverTitle.setObjectName("recoverTitle")
        recoverTitle.setAlignment(QtCore.Qt.AlignCenter)
        recoverTitle.setStyleSheet(
            "color: rgba(60, 60, 60, 220); font-size: 11pt; font-weight: 700;"
        )
        rec.addWidget(recoverTitle)

        recoverInfo = QtWidgets.QLabel(
            "Enter the email address linked to your account\n"
            "and we'll send you a password reset link."
        )
        recoverInfo.setObjectName("recoverInfo")
        recoverInfo.setAlignment(QtCore.Qt.AlignCenter)
        recoverInfo.setWordWrap(True)
        recoverInfo.setStyleSheet("color: rgba(100, 110, 120, 220); font-size: 8.5pt;")
        rec.addWidget(recoverInfo)

        self.recoverEmail = QtWidgets.QLineEdit()
        self.recoverEmail.setObjectName("recoverEmail")
        self.recoverEmail.setPlaceholderText("Email Address")
        self.recoverEmail.setFixedHeight(_INPUT_H)

        self.recoverEmail.setStyleSheet(f"""
            QLineEdit {{
                background-color: rgba(255, 255, 255, 140);
                border: 1.5px solid rgba(255, 255, 255, 160);
                border-style: solid;
                border-radius: {_INPUT_H // 2}px; 
                padding: 0px 15px;
                color: rgba(60, 60, 60, 220);
                font-size: 10pt;
                selection-background-color: rgba(200, 210, 220, 150);
                selection-color: rgba(40, 40, 40, 255);
            }}
            QLineEdit:hover {{ background-color: rgba(255, 255, 255, 180); border-color: rgba(255, 255, 255, 220); }}
            QLineEdit:focus {{ background-color: rgba(255, 255, 255, 240); border: 1.5px solid rgba(255, 255, 255, 255); outline: none; }}
        """)
        rec.addWidget(self.recoverEmail)

        self.sendResetBtn = QtWidgets.QPushButton("Send Reset Link")
        self.sendResetBtn.setObjectName("sendResetBtn")
        self.sendResetBtn.setFixedHeight(_BTN_H)
        self.sendResetBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.sendResetBtn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(229, 229, 229, 150);
                border: 1.5px solid rgba(255, 255, 255, 200);
                border-style: solid;
                border-radius: {_BTN_H // 2}px; 
                color: rgba(60, 60, 60, 220);
                font-size: 10pt; font-weight: 600;
            }}
            QPushButton:hover {{ background-color: rgba(210, 215, 220, 180); }}
            QPushButton:pressed {{ background-color: rgba(190, 200, 210, 200); }}
        """)
        self.sendResetBtn.clicked.connect(self._on_send_reset)
        rec.addWidget(self.sendResetBtn)

        self.recoverStatus = QtWidgets.QLabel("")
        self.recoverStatus.setObjectName("recoverStatus")
        self.recoverStatus.setAlignment(QtCore.Qt.AlignCenter)
        self.recoverStatus.setWordWrap(True)
        self.recoverStatus.setFixedHeight(36)
        self.recoverStatus.setStyleSheet(
            "color: rgba(46, 139, 87, 220); font-size: 8.5pt; font-weight: 500;"
        )
        rec.addWidget(self.recoverStatus)
        rec.addStretch()

        # ══════════════════════════════════════════════════════════════════════
        # ══════════════════════════════════════════════════════════════════════
        # Sliding panel and Glass Card
        # ══════════════════════════════════════════════════════════════════════
        self._slider = SlidingPanel(_CARD_W)
        self._slider.setObjectName("slidingPanel")

        # Keep slider completely transparent so the card shows through
        self._slider.setStyleSheet(
            "QWidget#slidingPanel { background: transparent; border: none; }"
        )

        self._slider.add_page(signInPage)
        self._slider.add_page(recoverPage)
        self._slider.setFixedHeight(_PAGE_H)
        QtCore.QTimer.singleShot(0, lambda: self._slider.finalize(_PAGE_H))

        # --- REVERTED GLASSCARD SETUP ---
        self.loginCard = GlassCard(self.centralwidget)
        self.loginCard.setObjectName("loginCard")
        self.loginCard.setAttribute(QtCore.Qt.WA_StyledBackground, False)
        self.loginCard.setContentsMargins(0, 0, 0, 0)
        # REMOVED: setAttribute(WA_StyledBackground)
        # REMOVED: loginCard.setStyleSheet(...)

        # Keep your existing drop shadow
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

        self._errorTimer = QtCore.QTimer()
        self._errorTimer.setSingleShot(True)
        self._errorTimer.timeout.connect(self.user_error.clear)

        self._sessionTimer = QtCore.QTimer()
        self._sessionTimer.setSingleShot(True)
        self._sessionTimer.timeout.connect(self.check_user_session)
        self._sessionTimer.setInterval(1000 * 60 * 60)

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

        QtCore.QTimer.singleShot(0, lambda: self.centralwidget.set_background_pixmap())

    # ── Avatar / user-switch ──────────────────────────────────────────────────

    @staticmethod
    def _make_circular_pixmap(initials: str, size: int) -> QtGui.QPixmap:
        """2. Shared method to generate muted pastel/slate avatars"""
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.transparent)

        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        hash_val = sum(ord(c) for c in initials)
        hues = [210, 200, 220, 190, 215]
        hue = hues[hash_val % len(hues)]
        base_color = QtGui.QColor.fromHsl(hue, 90, 190)

        rect = QtCore.QRectF(2.0, 2.0, size - 4.0, size - 4.0)
        p.setBrush(QtGui.QBrush(base_color))
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1.5))
        p.drawEllipse(rect)

        p.setPen(QtGui.QColor(60, 60, 60, 200))
        font = p.font()
        font.setPixelSize(int(size * 0.42))
        font.setBold(True)
        p.setFont(font)
        p.drawText(rect, QtCore.Qt.AlignCenter, initials)

        p.end()
        return pm

    def _show_user_switch_dialog(self) -> None:
        try:
            _, raw = UserProfiles.get_all_user_info()
            users: List[Tuple[str, str]] = [
                (info[0], info[1]) for info in raw if info[0] is not None and info[1] is not None
            ]
        except Exception as exc:
            Log.w(f"Could not load user profiles for switcher: {exc}")
            users = []

        current_name = self._current_user_display_name_for_switcher()

        dlg = SwitchUserDialog(
            users,
            parent=self.centralwidget,
            current_name=current_name,
        )
        dlg.user_selected.connect(self._on_user_selected)
        dlg.add_user_requested.connect(self._on_add_user_requested)

        dlg.adjustSize()
        self._position_user_switch_dialog(dlg)

        dlg.exec_()

    def _current_user_display_name_for_switcher(self) -> Optional[str]:
        for attr_name in (
            "current_user_display_name",
            "_current_user_display_name",
            "current_user_name",
            "_current_user_name",
        ):
            value = getattr(self, attr_name, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        prop_value = self.userAvatarBtn.property("display_name")
        if isinstance(prop_value, str) and prop_value.strip():
            return prop_value.strip()

        return None

    def _position_user_switch_dialog(self, dlg: QtWidgets.QDialog) -> None:
        avatar = self.userAvatarBtn
        gap = 10
        screen_padding = 12

        avatar_top_left = avatar.mapToGlobal(QtCore.QPoint(0, 0))
        avatar_center_x = avatar_top_left.x() + avatar.width() // 2

        # 5. Dialog Centering Fix: Use sizeHint() because the geometry is rarely final here
        preferred_x = avatar_center_x - dlg.sizeHint().width() // 2
        preferred_y = avatar_top_left.y() + avatar.height() + gap

        screen = QtWidgets.QApplication.screenAt(QtCore.QPoint(avatar_center_x, preferred_y))
        if screen is None:
            window_handle = self.window().windowHandle()
            screen = window_handle.screen() if window_handle is not None else None
        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()

        available = screen.availableGeometry()

        if preferred_y + dlg.height() > available.bottom() - screen_padding:
            preferred_y = avatar_top_left.y() - dlg.sizeHint().height() - gap

        x = max(
            available.left() + screen_padding,
            min(
                preferred_x,
                available.right() - dlg.sizeHint().width() - screen_padding,
            ),
        )

        y = max(
            available.top() + screen_padding,
            min(
                preferred_y,
                available.bottom() - dlg.sizeHint().height() - screen_padding,
            ),
        )

        dlg.move(x, y)

    def _on_user_selected(self, display_name: str, initials: str) -> None:
        self.user_initials.setText(initials)

        self.user_label.setText(display_name)
        self.user_label.setProperty("placeholder", False)
        self.user_label.style().unpolish(self.user_label)
        self.user_label.style().polish(self.user_label)

        self.userAvatarBtn.setProperty("display_name", display_name)
        self.userAvatarBtn.setProperty("hasIcon", True)

        # 2. Re-generates utilizing the shared slate/pastel aesthetic
        avatar_px = self._make_circular_pixmap(initials, _AVATAR_D)
        self.userAvatarBtn.setText("")
        self.userAvatarBtn.setIcon(QtGui.QIcon(avatar_px))
        self.userAvatarBtn.setIconSize(QtCore.QSize(_AVATAR_D, _AVATAR_D))

        self.userAvatarBtn.style().unpolish(self.userAvatarBtn)
        self.userAvatarBtn.style().polish(self.userAvatarBtn)

        self.user_password.setFocus()

    def _on_add_user_requested(self) -> None:
        Log.i("Add user requested from the login screen.")
        UserProfiles.create_new_user(UserRoles.OPERATE)
        QtCore.QTimer.singleShot(150, self._show_user_switch_dialog)

    # ── Slide transitions ──────────────────────────────────────────────────────
    def _slide_to_recover(self) -> None:
        self._slider.slide_to(1)

    def _slide_to_signin(self) -> None:
        self._slider.slide_to(0)
        self.recoverEmail.clear()
        self.recoverStatus.clear()
        self.sendResetBtn.setEnabled(True)

    # ── Reset-link handler ────────────────────────────────────────────────────
    def _on_send_reset(self) -> None:
        email = self.recoverEmail.text().strip()
        if not email:
            self.recoverStatus.setText("Please enter your email address.")
            return
        Log.i(f"Password reset requested for: {email}")
        self.recoverStatus.setText(
            f"If an account exists for that address,\na reset link has been sent."
        )
        self.sendResetBtn.setEnabled(False)

    # ── Password toggle ────────────────────────────────────────────────────────
    def on_toggle_password_Action(self) -> None:
        if not self.password_shown:
            self.user_password.setEchoMode(QtWidgets.QLineEdit.Normal)
            self.password_shown = True
            self.togglepasswordAction.setIcon(self.hiddenIcon)
        else:
            self.user_password.setEchoMode(QtWidgets.QLineEdit.Password)
            self.password_shown = False
            self.togglepasswordAction.setIcon(self.visibleIcon)

    # ── Session ───────────────────────────────────────────────────────────────
    def check_user_session(self) -> None:
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

    # ── Error helpers ─────────────────────────────────────────────────────────
    def error_loggedout(self) -> None:
        """Handles the forced logout state by notifying the user via the floating badge."""
        self.kickErrorTimer()

        # 1. Clear any leftover password text
        self.user_password.clear()

        # 2. Trigger the message on the floating overlay
        if hasattr(self, "floating_badge"):
            self.floating_badge.show_message(
                "You have been signed out", is_error=True, parent_widget=self.loginCard
            )

            # Auto-hide the badge after 5 seconds
            QtCore.QTimer.singleShot(5000, self.floating_badge.clear)

        # 3. Optional: Reset the password field style to normal in case it was red
        if hasattr(self, "_pw_style_normal"):
            self.user_password.setStyleSheet(self._pw_style_normal)

    def error_invalid(self, message: str = "Invalid Credentials") -> None:
        """
        Triggers the visual error state.
        The current user remains selected, but the password field alerts the user.
        """
        # 1. Apply the glassy red stylesheet to the password box
        if hasattr(self, "_pw_style_error"):
            self.user_password.setStyleSheet(self._pw_style_error)

        # 2. Trigger the physical jiggle animation
        if hasattr(self, "_shake_widget"):
            self._shake_widget(self.user_password)

        # 3. Clear only the password, keep the user selected
        self.user_password.clear()
        self.user_password.setFocus()

        # 4. Fire the message to the floating secondary window
        if hasattr(self, "floating_badge"):
            self.floating_badge.show_message(message, is_error=True, parent_widget=self.loginCard)

            # Auto-hide error after 4 seconds
            QtCore.QTimer.singleShot(4000, self.floating_badge.clear)

    def show_signout_message(self) -> None:
        # Show red message above the login card
        self.floating_badge.show_message(
            "You have been signed out.", is_error=True, parent_widget=self.loginCard
        )

        # Optional: Auto-hide after a few seconds
        QtCore.QTimer.singleShot(3000, self.floating_badge.clear)

    def error_expired(self) -> None:
        """Handles the session expiration state by notifying the user via the floating badge."""
        # 1. Clear the password field for security
        self.user_password.clear()

        # 2. Fire the message to the floating secondary window
        if hasattr(self, "floating_badge"):
            self.floating_badge.show_message(
                "Your session has expired", is_error=True, parent_widget=self.loginCard
            )

            # Auto-hide the message after 5 seconds
            QtCore.QTimer.singleShot(5000, self.floating_badge.clear)

        # 3. Optional: Reset the password box style if it was left in an error state
        if hasattr(self, "_pw_style_normal"):
            self.user_password.setStyleSheet(self._pw_style_normal)

    def kickErrorTimer(self) -> None:
        if self._errorTimer.isActive():
            Log.d("Error Timer was restarted while running")
            self._errorTimer.stop()
        self._errorTimer.start(10000)

    # ── Input helpers ─────────────────────────────────────────────────────────
    def text_transform(self) -> None:
        text = self.user_initials.text()
        if text:
            self.user_initials.setText(text.upper())

    # ── Sign-In ───────────────────────────────────────────────────────────────
    def action_sign_in(self) -> None:
        # Check if a user is actually selected (initials are not empty)
        if not self.user_initials.text():
            self.floating_badge.show_message(
                "Please select a user account first", is_error=True, parent_widget=self.loginCard
            )
            return

        # Check if password was entered
        if not self.user_password.text():
            # Jiggle the empty box and show the floating error
            self._shake_widget(self.user_password)
            self.floating_badge.show_message(
                "Password required", is_error=True, parent_widget=self.loginCard
            )
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

    # ── Clear / caps-lock ─────────────────────────────────────────────────────
    def clear_form(self) -> None:
        """Clears the form gracefully. Pressing once clears password, pressing again clears user."""
        if len(self.user_password.text()) > 0:
            # If there's a password typed, Escape just clears the password
            self.user_password.clear()
        else:
            # If the password box is already empty, Escape clears the selected user
            self.user_initials.clear()
            self.user_label.setText("Select a User")
            self.user_label.setProperty("placeholder", True)
            self.userAvatarBtn.setIcon(QtGui.QIcon())
            self.userAvatarBtn.setProperty("hasIcon", False)

            # Re-apply styling to remove the avatar border
            self.user_label.style().unpolish(self.user_label)
            self.user_label.style().polish(self.user_label)
            self.userAvatarBtn.style().unpolish(self.userAvatarBtn)
            self.userAvatarBtn.style().polish(self.userAvatarBtn)

        self.user_error.clear()

        if self.password_shown:
            self.on_toggle_password_Action()

    def update_caps_lock_state(self, caps_lock_on: bool) -> None:
        if caps_lock_on:
            # Show amber warning above the login card
            self.floating_badge.show_message(
                "Caps Lock is ON", is_error=False, parent_widget=self.loginCard
            )
        else:
            # Hide it
            self.floating_badge.clear()

    def load_saved_credentials(self) -> None:
        """Loads the saved password and user state on launch."""
        settings = QtCore.QSettings("QATCH", "nanovisQ")

        # Retrieve the boolean (defaults to False)
        remembered = settings.value("login/remember_me", False, type=bool)
        self.remember_me.setChecked(remembered)

        if remembered:
            # Load the saved data
            saved_password = settings.value("login/password", "")
            saved_user_name = settings.value("login/user_name", "")

            if saved_password:
                self.user_password.setText(saved_password)

            if saved_user_name:
                # Assuming you have a method to programmatically select a user
                # e.g., self.set_active_user(saved_user_name)
                pass

    def save_credentials_on_success(self, user_name: str, password: str) -> None:
        """Called upon successful login to save or clear stored credentials."""
        settings = QtCore.QSettings("QATCH", "nanovisQ")

        if self.remember_me.isChecked():
            settings.setValue("login/remember_me", True)
            settings.setValue("login/user_name", user_name)
            # IMPORTANT: In a production app, you should hash or encrypt this before saving.
            # QSettings saves in plain text (registry on Windows, .plist on Mac)
            settings.setValue("login/password", password)
        else:
            # Wipe out the saved data if they unchecked the box
            settings.setValue("login/remember_me", False)
            settings.remove("login/user_name")
            settings.remove("login/password")

    def _shake_widget(self, widget: QtWidgets.QWidget) -> None:
        """Applies a rapid left-right jiggle animation to indicate an error."""
        # Store animation as class attribute so it doesn't get garbage collected
        self._shake_anim = QtCore.QPropertyAnimation(widget, b"pos")
        self._shake_anim.setDuration(400)

        base_pos = widget.pos()

        # Keyframes for a smooth, decaying shake
        self._shake_anim.setKeyValueAt(0.0, base_pos)
        self._shake_anim.setKeyValueAt(0.1, base_pos + QtCore.QPoint(-6, 0))
        self._shake_anim.setKeyValueAt(0.3, base_pos + QtCore.QPoint(6, 0))
        self._shake_anim.setKeyValueAt(0.5, base_pos + QtCore.QPoint(-4, 0))
        self._shake_anim.setKeyValueAt(0.7, base_pos + QtCore.QPoint(4, 0))
        self._shake_anim.setKeyValueAt(0.9, base_pos + QtCore.QPoint(-2, 0))
        self._shake_anim.setKeyValueAt(1.0, base_pos)

        self._shake_anim.start(QtCore.QPropertyAnimation.DeleteWhenStopped)
