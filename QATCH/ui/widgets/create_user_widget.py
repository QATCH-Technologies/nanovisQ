"""
create_user_widget.py

Glassmorphism overlay widget for creating a new NanovisQ user account.

Improvements over v1:
  - Per-field inline error labels positioned directly below the offending input(s).
  - Error fields animate with a horizontal jiggle (ported from UILogin._shake_widget).
  - Role QComboBox styled to match GlassLineEdit (glass bg, pill border, custom popup).
  - User icon (icons/user.svg) centred above the title.
  - Circular × close button in the top-right corner of the card.
  - Show / hide password toggle using eye-on.svg / eye-off.svg (via QAction).
  - Submit replaced by a compact circular → button; Cancel kept as a full-width pill.

Styling follows ui_login.py conventions (GlassLineEdit, primary button gradient,
shake animation, transparent backgrounds).
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.core.constants import UserRoles
from QATCH.ui.components.glass_line_edit import GlassLineEdit
from QATCH.ui.components.animated_combo_box import AnimatedComboBox

# ---------------------------------------------------------------------------
# Constants matching ui_login.py
# ---------------------------------------------------------------------------
_INPUT_H: int = 34
_BTN_H: int = 34


# ---------------------------------------------------------------------------
# CreateUserWidget
# ---------------------------------------------------------------------------
class CreateUserWidget(QtWidgets.QWidget):
    """Full-screen overlay widget for user account creation.

    Covers the parent widget with a semi-opaque scrim and centres a
    glass-morphism card containing the creation form.
    """

    def __init__(
        self,
        existing_initials: list,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.existing_initials = existing_initials
        self.is_accepted: bool = False
        self.result_data: dict = {}

        # Active shake animations kept alive until finished
        self._shake_anims: List[QtCore.QPropertyAnimation] = []
        self._bg_alpha: int = 0  # Track background dimming alpha

        # Overlay covers the parent completely
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        if parent is not None:
            parent.installEventFilter(self)
            self.resize(parent.size())

        self._setup_ui()
        self.raise_()

        # Trigger the new, safe animation
        self._animate_open()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        # Centred outer layout
        self.base_layout = QtWidgets.QVBoxLayout(self)
        self.base_layout.setAlignment(QtCore.Qt.AlignCenter)

        # Glass card frame
        self.glass_frame = QtWidgets.QFrame(self)
        self.glass_frame.setObjectName("createUserView")
        self.glass_frame.setFixedWidth(440)
        self.glass_frame.setStyleSheet("""
            QFrame#createUserView {
                background: rgba(248, 251, 255, 215);
                border: 1.5px solid rgba(255, 255, 255, 235);
                border-radius: 18px;
            }
        """)

        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(44)
        shadow.setColor(QtGui.QColor(15, 40, 70, 90))
        shadow.setOffset(0, 10)
        self.glass_frame.setGraphicsEffect(shadow)

        self.main_layout = QtWidgets.QVBoxLayout(self.glass_frame)
        self.main_layout.setContentsMargins(28, 14, 28, 28)
        self.main_layout.setSpacing(6)

        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")

        # ── Row 1: close × button ──────────────────────────────────────
        close_row = QtWidgets.QHBoxLayout()
        close_row.setContentsMargins(0, 0, 0, 0)
        close_row.addStretch()

        self.btn_close = QtWidgets.QPushButton("×")
        self.btn_close.setFixedSize(28, 28)
        self.btn_close.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_close.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: rgba(110, 120, 130, 190);
                border: none;
                font-size: 18px;
                font-weight: bold;
                padding-bottom: 1px;
            }
            QPushButton:hover { color: rgba(210, 55, 55, 230); }
            QPushButton:pressed { color: rgba(160, 30, 30, 255); }
        """)
        self.btn_close.clicked.connect(self._reject)
        close_row.addWidget(self.btn_close)
        self.main_layout.addLayout(close_row)

        # ── Row 2: user.svg icon ───────────────────────────────────────
        icon_lbl = QtWidgets.QLabel()
        icon_lbl.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        icon_lbl.setAlignment(QtCore.Qt.AlignCenter)
        icon_path = os.path.join(icons_dir, "user-circle.svg")
        icon_pm = QtGui.QPixmap(icon_path)
        if not icon_pm.isNull():
            icon_pm = icon_pm.scaled(
                48, 48, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            icon_lbl.setPixmap(icon_pm)
        else:
            icon_lbl.setText("👤")
            icon_lbl.setStyleSheet("font-size: 30px; background: transparent;")
        icon_lbl.setFixedHeight(52)
        self.main_layout.addWidget(icon_lbl, alignment=QtCore.Qt.AlignCenter)

        # ── Row 3: title ───────────────────────────────────────────────
        lbl_title = QtWidgets.QLabel("Create User")
        lbl_title.setAlignment(QtCore.Qt.AlignCenter)
        lbl_title.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        lbl_title.setStyleSheet("""
            QLabel {
                color: rgba(50, 55, 65, 220);
                font-size: 14pt;
                font-weight: 700;
                background: transparent;
            }
        """)
        self.main_layout.addWidget(lbl_title)
        self.main_layout.addSpacing(6)

        # Eye icons shared by both password fields
        self._eye_on = QtGui.QIcon(os.path.join(icons_dir, "eye-on.svg"))
        self._eye_off = QtGui.QIcon(os.path.join(icons_dir, "eye-off.svg"))
        self._pwd1_visible = False
        self._pwd2_visible = False

        # ── First Name / Last Name row ─────────────────────────────────
        #    Wrapped in a QWidget so the shake animation has a single target.
        self.name_container = QtWidgets.QWidget()
        self.name_container.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        name_row = QtWidgets.QHBoxLayout(self.name_container)
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.setSpacing(10)

        self.inp_first_name = GlassLineEdit()
        self.inp_first_name.setFixedHeight(_INPUT_H)
        self.inp_first_name.setPlaceholderText("First Name")

        self.inp_last_name = GlassLineEdit()
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

        # ── Role combobox ──────────────────────────────────────────────
        self.cmb_role = AnimatedComboBox(os.path.join(icons_dir, "down-arrow.svg"))
        self.cmb_role.setFixedHeight(_INPUT_H)
        roles = [e.name for e in UserRoles][1:]
        for r in roles:
            label = r + " (Capture & Analyze)" if r == UserRoles.OPERATE.name else r
            self.cmb_role.addItem(label, r)

        # We no longer need to pass icons_dir to the style sheet
        self.cmb_role.setStyleSheet(self._combo_style(error=False))
        self.main_layout.addWidget(self.cmb_role)
        # ── Username (optional) ────────────────────────────────────────
        self.inp_username = GlassLineEdit()
        self.inp_username.setFixedHeight(_INPUT_H)
        self.inp_username.setPlaceholderText("Username (Optional)")
        self.main_layout.addWidget(self.inp_username)

        # ── Email ──────────────────────────────────────────────────────
        self.inp_email = GlassLineEdit()
        self.inp_email.setFixedHeight(_INPUT_H)
        self.inp_email.setPlaceholderText("Email")
        self.inp_email.textChanged.connect(
            lambda _: self._clear_field_errors([self.inp_email], self.err_email)
        )
        self.main_layout.addWidget(self.inp_email)

        self.err_email = self._make_error_label()
        self.main_layout.addWidget(self.err_email)

        # ── Password ───────────────────────────────────────────────────
        self.inp_pwd1 = GlassLineEdit()
        self.inp_pwd1.setFixedHeight(_INPUT_H)
        self.inp_pwd1.setPlaceholderText("Password")
        self.inp_pwd1.setEchoMode(QtWidgets.QLineEdit.Password)
        self.inp_pwd1.textChanged.connect(
            lambda _: self._clear_field_errors([self.inp_pwd1, self.inp_pwd2], self.err_password)
        )
        self._act_eye1 = self.inp_pwd1.addAction(self._eye_on, QtWidgets.QLineEdit.TrailingPosition)
        self._act_eye1.triggered.connect(self._toggle_pwd1)
        self.main_layout.addWidget(self.inp_pwd1)

        # ── Confirm Password ───────────────────────────────────────────
        self.inp_pwd2 = GlassLineEdit()
        self.inp_pwd2.setFixedHeight(_INPUT_H)
        self.inp_pwd2.setPlaceholderText("Confirm Password")
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

        # ── Button row ─────────────────────────────────────────────────
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setAlignment(QtCore.Qt.AlignCenter)  # This centers the widget

        self.btn_create = QtWidgets.QPushButton("")
        self.btn_create.setIcon(QtGui.QIcon(os.path.join(icons_dir, "right-arrow.svg")))
        self.btn_create.setIconSize(QtCore.QSize(20, 20))
        self.btn_create.setFixedSize(40, 40)
        self.btn_create.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_create.setToolTip("Create User")
        self.btn_create.setStyleSheet("""
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(45, 165, 250, 210),
                    stop:1 rgba(15, 125, 210, 190)
                );
                border-top:    1px solid rgba(255, 255, 255, 100);
                border-left:   1px solid rgba(255, 255, 255, 50);
                border-right:  1px solid rgba(0, 80, 150, 50);
                border-bottom: 1px solid rgba(0, 80, 150, 80);
                border-radius: 20px;
                color: rgba(255, 255, 255, 255);
                font-size: 17px;
                font-weight: bold;
                padding-left: 2px;
            }
            QPushButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(65, 185, 255, 240),
                    stop:1 rgba(25, 145, 230, 220)
                );
                border-top: 1px solid rgba(255, 255, 255, 140);
            }
            QPushButton:pressed {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(15, 115, 200, 220),
                    stop:1 rgba(5, 95, 160, 200)
                );
            }
        """)
        self.btn_create.clicked.connect(self._validate_and_accept)

        # Removed the addStretch() that was pushing it to the right
        btn_layout.addWidget(self.btn_create)
        self.main_layout.addLayout(btn_layout)

        self.base_layout.addWidget(self.glass_frame)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, self._bg_alpha))
        p.end()

    # ------------------------------------------------------------------
    # Style helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_error_label() -> QtWidgets.QLabel:
        """Returns a compact red inline error label, hidden by default."""
        lbl = QtWidgets.QLabel("")
        lbl.setWordWrap(True)
        lbl.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        lbl.setStyleSheet("""
            QLabel {
                color: rgba(210, 55, 55, 220);
                font-size: 8.5pt;
                font-weight: 600;
                background: transparent;
                padding-left: 6px;
            }
        """)
        lbl.setVisible(False)
        return lbl

    @staticmethod
    def _combo_style(*, error: bool = False) -> str:
        """Returns QSS for the role combobox, matching GlassLineEdit visuals."""
        if error:
            bg = "rgba(255, 220, 220, 80)"
            border = "1.5px solid rgba(210, 55, 55, 180)"
            color = "rgba(210, 55, 55, 230)"
        else:
            bg = "rgba(255, 255, 255, 72)"
            border = "1px solid rgba(255, 255, 255, 130)"
            color = "rgba(38, 48, 58, 230)"

        return f"""
            QComboBox {{
                background: {bg};
                border: {border};
                border-radius: {_INPUT_H // 2}px;
                padding: 0px 14px;
                color: {color};
                font-size: 10pt;
                min-height: {_INPUT_H}px;
            }}
            QComboBox:hover {{
                background: rgba(255, 255, 255, 110);
                border: 1px solid rgba(255, 255, 255, 200);
            }}
            QComboBox:on {{
                background: rgba(255, 255, 255, 130);
                border: 1.5px solid rgba(185, 218, 248, 150);
            }}
            QComboBox::drop-down {{
                subcontrol-origin:   padding;
                subcontrol-position: right center;
                width: 32px;
                border: none;
            }}
            QComboBox::down-arrow {{
                image: none; /* Handled dynamically by AnimatedComboBox */
            }}
            
            /* --- Dropdown Window --- */
            QComboBox QAbstractItemView {{
                background:               rgba(245, 249, 254, 250);
                border:                   1px solid rgba(185, 218, 248, 160);
                border-radius:            10px;
                selection-background-color: transparent;
                selection-color:          rgba(38, 48, 58, 230);
                color:                    rgba(38, 48, 58, 230);
                font-size:                10pt;
                padding:                  4px; /* Uniform padding prevents items from touching corners */
                outline:                  none;
            }}
            
            QComboBox QAbstractItemView::viewport {{
                background: transparent;
                border-radius: 10px;
            }}
            
            /* --- Shrunken Menu Items --- */
            QComboBox QAbstractItemView::item {{
                padding:       4px 10px; /* Reduced from 7px 14px */
                min-height:    20px;     /* Reduced from 28px */
                border-radius: 5px;      /* Rounds the highlight so it doesn't bleed */
            }}
            
            QComboBox QAbstractItemView::item:hover {{
                background: rgba(10, 163, 230, 45);
            }}
            
            QComboBox QAbstractItemView::item:selected {{
                background: rgba(10, 163, 230, 80);
            }}
        """

    # ------------------------------------------------------------------
    # Error / animation helpers
    # ------------------------------------------------------------------
    def _clear_field_errors(
        self,
        fields: list,
        error_label: QtWidgets.QLabel,
    ) -> None:
        """Clears error styling on the given fields and hides the error label."""
        for f in fields:
            if isinstance(f, GlassLineEdit):
                f.set_error(False)
        error_label.setVisible(False)

    def _show_field_error(
        self,
        fields: list,
        error_label: QtWidgets.QLabel,
        message: str,
        shake_target: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Marks fields red, shows the inline error, and jiggle-shakes the target."""
        for f in fields:
            if isinstance(f, GlassLineEdit):
                f.set_error(True)
        error_label.setText(message)
        error_label.setVisible(True)
        self._shake_widget(shake_target or (fields[0] if fields else None))

    def _shake_widget(self, widget: Optional[QtWidgets.QWidget]) -> None:
        """Horizontal jiggle animation for error feedback (from UILogin._shake_widget)."""
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
        anim.start(QtCore.QPropertyAnimation.DeleteWhenStopped)

        self._shake_anims.append(anim)
        anim.finished.connect(
            lambda: self._shake_anims.remove(anim) if anim in self._shake_anims else None
        )

    # ------------------------------------------------------------------
    # Password visibility toggles
    # ------------------------------------------------------------------
    def _toggle_pwd1(self) -> None:
        self._pwd1_visible = not self._pwd1_visible
        self.inp_pwd1.setEchoMode(
            QtWidgets.QLineEdit.Normal if self._pwd1_visible else QtWidgets.QLineEdit.Password
        )
        self._act_eye1.setIcon(self._eye_off if self._pwd1_visible else self._eye_on)

    def _toggle_pwd2(self) -> None:
        self._pwd2_visible = not self._pwd2_visible
        self.inp_pwd2.setEchoMode(
            QtWidgets.QLineEdit.Normal if self._pwd2_visible else QtWidgets.QLineEdit.Password
        )
        self._act_eye2.setIcon(self._eye_off if self._pwd2_visible else self._eye_on)

    # ------------------------------------------------------------------
    # Initials helper
    # ------------------------------------------------------------------
    def _generate_initials(self, first: str, last: str) -> str:
        base = f"{first[0]}{last[0]}".upper()
        initials, counter = base, 1
        while initials in self.existing_initials:
            initials = f"{base}{counter}"
            counter += 1
        return initials

    # ------------------------------------------------------------------
    # Validation & acceptance
    # ------------------------------------------------------------------
    def _validate_and_accept(self) -> None:
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

        # ── Name ──────────────────────────────────────────────────────
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

        # ── Email ──────────────────────────────────────────────────────
        if not email or not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
            self._show_field_error(
                [self.inp_email],
                self.err_email,
                "Please enter a valid email address.",
            )
            has_error = True

        # ── Password ───────────────────────────────────────────────────
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

        # ── All good ───────────────────────────────────────────────────
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
        self.is_accepted = False
        self._close_with_animation()

    # ------------------------------------------------------------------
    # Intro / Outro Animations
    # ------------------------------------------------------------------
    def _animate_open(self) -> None:
        """Fades in the dark overlay and slides the card up into place."""
        self.anim_in = QtCore.QVariantAnimation(self)
        self.anim_in.setDuration(300)
        self.anim_in.setStartValue(0.0)
        self.anim_in.setEndValue(1.0)
        self.anim_in.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self.anim_in.valueChanged.connect(self._on_anim_frame)
        self.anim_in.start(QtCore.QPropertyAnimation.DeleteWhenStopped)

    def _close_with_animation(self) -> None:
        """Reverses the animation before safely destroying the widget."""
        # Disable interactions so the user can't double-click 'close' or 'create'
        self.glass_frame.setEnabled(False)

        self.anim_out = QtCore.QVariantAnimation(self)
        self.anim_out.setDuration(200)
        self.anim_out.setStartValue(1.0)
        self.anim_out.setEndValue(0.0)
        self.anim_out.setEasingCurve(QtCore.QEasingCurve.InQuad)
        self.anim_out.valueChanged.connect(self._on_anim_frame)
        self.anim_out.finished.connect(self.close)
        self.anim_out.start(QtCore.QPropertyAnimation.DeleteWhenStopped)

    def _on_anim_frame(self, progress: float) -> None:
        """Drives both the background fade and the layout margin slide."""
        # Max alpha is 130; scale it by our 0.0 -> 1.0 progress
        self._bg_alpha = int(130 * progress)

        # Start the card 50px lower and slide it up to 0px
        offset = int(50 * (1.0 - progress))
        self.base_layout.setContentsMargins(0, offset, 0, 0)

        # Force the paintEvent to redraw the darker background
        self.update()
