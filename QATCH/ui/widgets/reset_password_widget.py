"""
reset_password_widget.py

Glassmorphism overlay widget for resetting a NanovisQ user's password.

Structure mirrors create_user_widget.py exactly: same card geometry, same
open/close animation (scrim fade + slide-up), same inline error system, and
the same x close / → submit button conventions.

Signal
------
  password_confirmed(str)  — emitted with the validated plaintext password
                             before the close animation starts.

Usage
-----
    overlay = ResetPasswordWidget(
        name="Jane Smith", initials="JS", role="ADMIN",
        parent=self,
    )
    overlay.resize(self.size())
    overlay.show()
    overlay.raise_()
    overlay.password_confirmed.connect(
        lambda pwd: self._update_user_xml(filename, new_pwd_plain=pwd)
    )
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.ui.components.glass_line_edit import GlassLineEdit

# ---------------------------------------------------------------------------
# Constants — kept in sync with create_user_widget.py
# ---------------------------------------------------------------------------
_INPUT_H: int = 34
_CARD_W: int = 420  # slightly narrower than CreateUserWidget (440) — fewer fields


class ResetPasswordWidget(QtWidgets.QWidget):
    """Full-screen overlay for resetting an existing user's password.

    Sits on top of UserProfilesManagerWidget's glass panel.  The admin sees
    the target user's profile card, two disabled placeholder fields (email /
    username — backend not yet implemented), and the two live password inputs.

    Attributes:
        is_accepted (bool):  True if the admin submitted a valid new password.
    """

    # Emitted with the validated plaintext password before the close animation.
    password_confirmed: QtCore.pyqtSignal = QtCore.pyqtSignal(str)

    def __init__(
        self,
        name: str,
        initials: str,
        role: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._name = name
        self._initials = initials
        self._role = role

        self.is_accepted: bool = False

        # Keeps shake animations alive for their full duration
        self._shake_anims: List[QtCore.QPropertyAnimation] = []
        # Controls the scrim alpha painted in paintEvent
        self._bg_alpha: int = 0

        # Overlay must be transparent (scrim is drawn manually in paintEvent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)

        if parent is not None:
            parent.installEventFilter(self)
            self.resize(parent.size())

        self._setup_ui()
        self.raise_()
        self._animate_open()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")

        # ── Outer centred layout ───────────────────────────────────────
        self.base_layout = QtWidgets.QVBoxLayout(self)
        self.base_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # ── Glass card ─────────────────────────────────────────────────
        self.glass_frame = QtWidgets.QFrame(self)
        self.glass_frame.setObjectName("resetPwdView")
        self.glass_frame.setFixedWidth(_CARD_W)
        self.glass_frame.setStyleSheet("""
            QFrame#resetPwdView {
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

        # ── Row 1: x close button ──────────────────────────────────────
        close_row = QtWidgets.QHBoxLayout()
        close_row.setContentsMargins(0, 0, 0, 0)
        close_row.addStretch()

        self.btn_close = QtWidgets.QPushButton("x")
        self.btn_close.setFixedSize(28, 28)
        self.btn_close.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_close.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: rgba(110, 120, 130, 190);
                border: none;
                font-size: 18px;
                font-weight: bold;
                padding-bottom: 1px;
            }
            QPushButton:hover   { color: rgba(210, 55, 55, 230); }
            QPushButton:pressed { color: rgba(160, 30, 30, 255); }
        """)
        self.btn_close.clicked.connect(self._reject)
        close_row.addWidget(self.btn_close)
        self.main_layout.addLayout(close_row)

        # ── Row 2: header icon ─────────────────────────────────────────
        icon_lbl = QtWidgets.QLabel()
        icon_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        icon_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        icon_pm = QtGui.QPixmap(os.path.join(icons_dir, "reset-password.svg"))
        if not icon_pm.isNull():
            icon_pm = icon_pm.scaled(
                48,
                48,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            icon_lbl.setPixmap(icon_pm)
        else:
            icon_lbl.setText("🔑")
            icon_lbl.setStyleSheet("font-size: 30px; background: transparent;")
        icon_lbl.setFixedHeight(52)
        self.main_layout.addWidget(icon_lbl, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # ── Row 3: title ───────────────────────────────────────────────
        lbl_title = QtWidgets.QLabel("Reset Password")
        lbl_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_title.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        lbl_title.setStyleSheet("""
            QLabel {
                color: rgba(50, 55, 65, 220);
                font-size: 14pt;
                font-weight: 700;
                background: transparent;
            }
        """)
        self.main_layout.addWidget(lbl_title)
        self.main_layout.addSpacing(12)

        # ── Row 4: user info card ──────────────────────────────────────
        self.main_layout.addWidget(self._build_user_info_card())
        self.main_layout.addSpacing(10)

        # ── Separator ─────────────────────────────────────────────────
        self.main_layout.addWidget(self._make_separator())
        self.main_layout.addSpacing(10)

        # ── Row 5 & 6: placeholder fields (future backend) ────────────
        self.main_layout.addWidget(self._make_future_field("Email", "example@domain.com"))
        self.main_layout.addSpacing(5)
        self.main_layout.addWidget(self._make_future_field("Username", "optional username"))
        self.main_layout.addSpacing(10)

        # ── Separator ─────────────────────────────────────────────────
        self.main_layout.addWidget(self._make_separator())
        self.main_layout.addSpacing(10)

        # ── Row 7 & 8: password inputs ─────────────────────────────────
        self._eye_on = QtGui.QIcon(os.path.join(icons_dir, "eye-on.svg"))
        self._eye_off = QtGui.QIcon(os.path.join(icons_dir, "eye-off.svg"))
        self._pwd1_visible = False
        self._pwd2_visible = False

        self.inp_pwd1 = GlassLineEdit()
        self.inp_pwd1.setFixedHeight(_INPUT_H)
        self.inp_pwd1.setPlaceholderText("New Password")
        self.inp_pwd1.setEchoMode(QtWidgets.QLineEdit.Password)
        self.inp_pwd1.textChanged.connect(
            lambda _: self._clear_field_errors([self.inp_pwd1, self.inp_pwd2], self.err_password)
        )
        self._act_eye1 = self.inp_pwd1.addAction(self._eye_on, QtWidgets.QLineEdit.TrailingPosition)
        self._act_eye1.triggered.connect(self._toggle_pwd1)
        self.main_layout.addWidget(self.inp_pwd1)

        self.inp_pwd2 = GlassLineEdit()
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

        # ── Row 9: submit button ───────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.btn_submit = QtWidgets.QPushButton("")
        self.btn_submit.setIcon(QtGui.QIcon(os.path.join(icons_dir, "right-arrow.svg")))
        self.btn_submit.setIconSize(QtCore.QSize(20, 20))
        self.btn_submit.setFixedSize(40, 40)
        self.btn_submit.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_submit.setToolTip("Confirm password reset")
        self.btn_submit.setStyleSheet("""
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
        self.btn_submit.clicked.connect(self._validate_and_accept)

        btn_row.addWidget(self.btn_submit)
        self.main_layout.addLayout(btn_row)

        self.base_layout.addWidget(self.glass_frame)

    # ------------------------------------------------------------------
    # Sub-widget builders
    # ------------------------------------------------------------------
    def _build_user_info_card(self) -> QtWidgets.QFrame:
        """Returns a mini profile card showing the target user's identity."""
        card = QtWidgets.QFrame()
        card.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        card.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 55);
                border: 1px solid rgba(185, 218, 248, 100);
                border-radius: 12px;
            }
        """)

        lay = QtWidgets.QHBoxLayout(card)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(14)

        # ── Initials avatar ───────────────────────────────────────────
        avatar = QtWidgets.QLabel(self._initials)
        avatar.setFixedSize(46, 46)
        avatar.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        avatar.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        avatar.setStyleSheet("""
            QLabel {
                background: rgba(45, 165, 250, 100);
                color: rgba(10, 70, 150, 240);
                border: 1.5px solid rgba(45, 165, 250, 180);
                border-radius: 23px;
                font-weight: 700;
                font-size: 13pt;
            }
        """)

        # ── Name + role column ────────────────────────────────────────
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(5)
        col.setContentsMargins(0, 0, 0, 0)

        name_lbl = QtWidgets.QLabel(self._name)
        name_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        name_lbl.setStyleSheet("""
            QLabel {
                color: rgba(35, 40, 55, 225);
                font-size: 11pt;
                font-weight: 600;
                background: transparent;
            }
        """)

        col.addWidget(name_lbl)
        col.addWidget(self._make_role_badge(self._role))

        lay.addWidget(avatar)
        lay.addLayout(col)
        lay.addStretch()

        return card

    def _make_role_badge(self, role_name: str) -> QtWidgets.QLabel:
        """Returns a colour-coded role pill label matching the table's role badges."""
        role_upper = str(role_name).upper()
        if "ADMIN" in role_upper:
            bg, border, color = "rgba(220,53,69,0.15)", "rgba(220,53,69,0.40)", "#C82333"
        elif "AUDIT" in role_upper or "MANAGER" in role_upper:
            bg, border, color = "rgba(255,193,7,0.18)", "rgba(255,193,7,0.45)", "#B38600"
        elif "OPERATE" in role_upper:
            bg, border, color = "rgba(40,167,69,0.15)", "rgba(40,167,69,0.40)", "#1E7E34"
        elif "ANALYZE" in role_upper:
            bg, border, color = "rgba(111,66,193,0.15)", "rgba(111,66,193,0.40)", "#6F42C1"
        elif "CAPTURE" in role_upper:
            bg, border, color = "rgba(255,193,7,0.15)", "rgba(255,193,7,0.40)", "#B38600"
        else:
            bg, border, color = "rgba(10,163,230,0.15)", "rgba(10,163,230,0.40)", "#0AA3E6"

        badge = QtWidgets.QLabel(role_name)
        badge.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        badge.setStyleSheet(f"""
            QLabel {{
                background: {bg};
                border: 1px solid {border};
                color: {color};
                border-radius: 9px;
                padding: 2px 10px;
                font-weight: bold;
                font-size: 9pt;
            }}
        """)
        return badge

    def _make_future_field(self, label_text: str, placeholder: str) -> QtWidgets.QFrame:
        """Returns a non-interactive pill matching the GlassLineEdit shape.

        Visually dimmed relative to the active password fields to clearly
        signal that these fields are placeholders pending a backend feature.
        """
        frame = QtWidgets.QFrame()
        frame.setFixedHeight(_INPUT_H)
        frame.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        frame.setStyleSheet("""
            QFrame {
                background: rgba(245, 247, 252, 30);
                border: 1px solid rgba(200, 210, 222, 60);
                border-radius: 17px;
            }
        """)

        lay = QtWidgets.QHBoxLayout(frame)
        lay.setContentsMargins(15, 0, 10, 0)
        lay.setSpacing(8)

        field_lbl = QtWidgets.QLabel(label_text)
        field_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        field_lbl.setStyleSheet("""
            QLabel {
                color: rgba(140, 152, 168, 140);
                font-size: 10pt;
                background: transparent;
            }
        """)

        placeholder_lbl = QtWidgets.QLabel(placeholder)
        placeholder_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        placeholder_lbl.setStyleSheet("""
            QLabel {
                color: rgba(160, 170, 185, 100);
                font-size: 10pt;
                font-style: italic;
                background: transparent;
            }
        """)

        soon_badge = QtWidgets.QLabel("coming soon")
        soon_badge.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        soon_badge.setStyleSheet("""
            QLabel {
                color: rgba(150, 160, 175, 160);
                background: rgba(185, 195, 210, 40);
                border: 1px solid rgba(185, 195, 210, 75);
                border-radius: 9px;
                padding: 1px 8px;
                font-size: 8pt;
            }
        """)

        lay.addWidget(field_lbl)
        lay.addWidget(placeholder_lbl)
        lay.addStretch()
        lay.addWidget(soon_badge)

        return frame

    @staticmethod
    def _make_separator() -> QtWidgets.QWidget:
        """Returns a 1 px horizontal glass rule."""
        sep = QtWidgets.QWidget()
        sep.setFixedHeight(1)
        sep.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        sep.setStyleSheet("QWidget { background: rgba(185, 218, 248, 75); }")
        return sep

    @staticmethod
    def _make_error_label() -> QtWidgets.QLabel:
        """Returns a compact inline error label, hidden by default."""
        lbl = QtWidgets.QLabel("")
        lbl.setWordWrap(True)
        lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
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

    # ------------------------------------------------------------------
    # Error helpers  (ported verbatim from CreateUserWidget)
    # ------------------------------------------------------------------
    def _clear_field_errors(
        self,
        fields: list,
        error_label: QtWidgets.QLabel,
    ) -> None:
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
        for f in fields:
            if isinstance(f, GlassLineEdit):
                f.set_error(True)
        error_label.setText(message)
        error_label.setVisible(True)
        self._shake_widget(shake_target or (fields[0] if fields else None))

    def _shake_widget(self, widget: Optional[QtWidgets.QWidget]) -> None:
        """Horizontal jiggle animation for error feedback."""
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
    # Validation & acceptance
    # ------------------------------------------------------------------
    def _validate_and_accept(self) -> None:
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
        self.is_accepted = False
        self._close_with_animation()

    # ------------------------------------------------------------------
    # Open / close animations  (identical to CreateUserWidget)
    # ------------------------------------------------------------------
    def _animate_open(self) -> None:
        """Fades in the scrim and slides the card up into place."""
        self.anim_in = QtCore.QVariantAnimation(self)
        self.anim_in.setDuration(300)
        self.anim_in.setStartValue(0.0)
        self.anim_in.setEndValue(1.0)
        self.anim_in.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self.anim_in.valueChanged.connect(self._on_anim_frame)
        self.anim_in.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def _close_with_animation(self) -> None:
        """Disables interactions then reverses the animation before destruction."""
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
        """Drives both the scrim fade and the card slide-up."""
        # Slightly lower max alpha (100 vs 130) — this overlay sits on top of
        # the manager's own scrim so the combined darkness stays comfortable.
        self._bg_alpha = int(100 * progress)
        offset = int(50 * (1.0 - progress))
        self.base_layout.setContentsMargins(0, offset, 0, 0)
        self.update()

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, self._bg_alpha))
        p.end()

    def eventFilter(self, obj, event) -> bool:
        """Keeps the overlay filling its parent if the parent is resized."""
        if obj is self.parent() and event.type() == QtCore.QEvent.Type.Resize:
            self.resize(event.size())
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Clicking the scrim outside the card dismisses the overlay."""
        if not self.glass_frame.geometry().contains(event.pos()):
            self._reject()
        else:
            super().mousePressEvent(event)
