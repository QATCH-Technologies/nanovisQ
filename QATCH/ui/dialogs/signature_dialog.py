import os
from typing import Any, Callable, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.common.userProfiles import UserProfiles
from QATCH.core.constants import Constants
from QATCH.ui.components.qatch_dialog import DialogBase, DialogCard, tinted_icon
from QATCH.ui.components.qatch_line_edit import QATCHLineEdit
from QATCH.ui.components.qatch_push_button import QATCHPushButton
from QATCH.ui.components.qatch_warning_label import QATCHWarningLabel
from QATCH.ui.dialogs.pop_up_dialog import PopUp
from QATCH.ui.styles.theme_manager import (
    ThemeManager,
    dialog_message_qss,
    dialog_title_qss,
    hairline_qss,
)

_ICON_PATH = os.path.join(Architecture.get_path(), "QATCH", "icons", "signature.svg")
_CARD_W = 360
_HEADER_H = 52

_SESSION_KEY_PATH = os.path.join(Constants.user_profiles_path, "session.key")


def _read_key(path: str) -> Optional[str]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.readline()
    return None


def auto_sign_matches_session() -> bool:
    """Dev-mode only: True if a persisted "do not ask again" auto-sign key
    matches the current session's key - i.e. every other signature call
    site should silently bypass showing `SignatureDialog` entirely."""
    session_key = _read_key(_SESSION_KEY_PATH)
    return session_key is not None and _read_key(Constants.auto_sign_key_path) == session_key


def clear_stale_auto_sign_key() -> None:
    """Removes a persisted auto-sign key that no longer matches the
    current session (called once that mismatch is detected)."""
    if os.path.exists(Constants.auto_sign_key_path):
        os.remove(Constants.auto_sign_key_path)


def persist_auto_sign_key() -> None:
    """Persists the current session key as the auto-sign key, so
    `auto_sign_matches_session()` returns True for the rest of this
    session (i.e. every later signature request is silently bypassed)."""
    session_key = _read_key(_SESSION_KEY_PATH)
    if session_key is not None and not os.path.exists(Constants.auto_sign_key_path):
        with open(Constants.auto_sign_key_path, "w") as f:
            f.write(session_key)


class SignatureDialog(DialogBase):
    """A self-contained modal dialog to capture a user's signature/initials.

    Built on the same frosted-glass chrome as `QATCHDialog`/`PopUp` (see
    `QATCH.ui.components.qatch_dialog.GlassDialogBase`), so it renders as a
    centred glass card over a dimmed backdrop with the same header, body,
    and button styling as every other modal in the app.

    This dialog verifies the identity of the current user by requiring their
    initials. it supports 'Dev Mode' auto-signing and allows users to switch
    profiles if the current session info is incorrect.

    Attributes:
        username (str): The display name of the currently logged-in user.
        expected_initials (Optional[str]): The initials required to pass validation.
        sign (QATCHLineEdit): Input field for user initials.
        sign_do_not_ask (QCheckBox): Toggle for persistent session signing.
    """

    def __init__(
        self,
        parent: Optional[Any] = None,
        on_switch_user: Optional[Callable[[], Optional[Tuple[str, str]]]] = None,
    ) -> None:
        """Initializes the SignatureDialog and loads session metadata.

        Args:
            parent (QWidget, optional): The parent widget for the modal dialog.
            on_switch_user (callable, optional): Invoked when "Switch User" is
                clicked, in place of the default placeholder message. Should
                perform the actual profile switch and return the new
                `(username, initials)` on a real change, or `None` if the
                switch failed or the user didn't change - callers that also
                need to update their own app-global state (toolbars, session
                flags, etc.) should do so inside this callback.
        """
        super().__init__(parent)

        self._on_switch_user = on_switch_user
        self.username = "[NONE]"
        self.expected_initials = "N/A"

        try:
            valid, infos = UserProfiles.session_info()
            if valid:
                if infos and len(infos) >= 2:
                    self.username = infos[0]
                    self.expected_initials = infos[1]
                else:
                    self.username = "Unknown User"
                    self.expected_initials = ""
            else:
                try:
                    self.expected_initials = None
                except Exception:
                    self.expected_initials = "N/A"
        except ImportError:
            pass

        self._build_ui()

        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addStretch()

        card_row = QtWidgets.QHBoxLayout()
        card_row.setContentsMargins(0, 0, 0, 0)
        card_row.addStretch()

        self._card = DialogCard(self, header_line_y=_HEADER_H)
        self._card.setFixedWidth(_CARD_W)

        card_v = QtWidgets.QVBoxLayout(self._card)
        card_v.setContentsMargins(0, 0, 0, 0)
        card_v.setSpacing(0)
        header_w = QtWidgets.QWidget()
        header_w.setFixedHeight(_HEADER_H)
        header_w.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        header_layout = QtWidgets.QHBoxLayout(header_w)
        header_layout.setContentsMargins(16, 0, 16, 0)
        header_layout.setSpacing(10)

        self._icon_label = QtWidgets.QLabel()
        self._icon_label.setFixedSize(24, 24)
        self._icon_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._refresh_icon()
        header_layout.addWidget(self._icon_label)

        self._title_label = QtWidgets.QLabel("Signature")
        self._title_label.setWordWrap(True)
        self._title_label.setStyleSheet(dialog_title_qss())
        header_layout.addWidget(self._title_label, 1)

        card_v.addWidget(header_w)
        body_w = QtWidgets.QWidget()
        body_w.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        body_layout = QtWidgets.QVBoxLayout(body_w)
        body_layout.setContentsMargins(20, 16, 20, 0)
        body_layout.setSpacing(10)

        signed_row = QtWidgets.QHBoxLayout()
        self._signed_caption = QtWidgets.QLabel("Signed in as: ")
        self._signed_caption.setStyleSheet(dialog_message_qss())
        self._signed_caption.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        signed_row.addWidget(self._signed_caption)

        self.signedInAs = QtWidgets.QLabel(self.username)
        self.signedInAs.setStyleSheet(dialog_message_qss())
        self.signedInAs.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        signed_row.addWidget(self.signedInAs, 1)
        body_layout.addLayout(signed_row)

        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.HLine)
        divider.setStyleSheet(hairline_qss())
        body_layout.addWidget(divider)

        initials_row = QtWidgets.QHBoxLayout()
        self.signerInit = QtWidgets.QLabel(f"Initials: <b>{self.expected_initials or 'N/A'}</b>")
        self.signerInit.setStyleSheet(dialog_message_qss())
        initials_row.addWidget(self.signerInit)
        initials_row.addStretch()

        switch_user = QATCHPushButton("Switch User", variant="ghost")
        switch_user.clicked.connect(self.switch_user_at_sign_time)
        initials_row.addWidget(switch_user)
        body_layout.addLayout(initials_row)

        self.sign = QATCHLineEdit()
        self.sign.setMaxLength(4)
        self.sign.setPlaceholderText("Initials")
        self.sign.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.sign.textChanged.connect(self._clear_error)
        body_layout.addWidget(self.sign)

        self._error_label = QATCHWarningLabel(severity="danger")
        self._error_label.hide()
        body_layout.addWidget(self._error_label)

        self.sign_do_not_ask = QtWidgets.QCheckBox("Do not ask again this session")
        self.sign_do_not_ask.setObjectName("themedCheckBox")
        self.sign_do_not_ask.setEnabled(False)

        # Dev Mode Logic
        if UserProfiles.checkDevMode()[0]:
            if auto_sign_matches_session():
                self.sign_do_not_ask.setChecked(True)
            else:
                self.sign_do_not_ask.setChecked(False)
                clear_stale_auto_sign_key()

        if self.sign_do_not_ask.isEnabled() or self.sign_do_not_ask.isChecked():
            body_layout.addWidget(self.sign_do_not_ask)
        else:
            self.sign_do_not_ask.hide()

        card_v.addWidget(body_w)

        btn_row = QtWidgets.QWidget()
        btn_row.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        btn_layout = QtWidgets.QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(20, 16, 20, 20)
        btn_layout.setSpacing(8)
        btn_layout.addStretch()

        self.sign_cancel = QATCHPushButton("Cancel", variant="secondary")
        self.sign_cancel.setFixedHeight(34)
        self.sign_cancel.setMinimumWidth(90)
        self.sign_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.sign_cancel)

        self.sign_ok = QATCHPushButton("Sign", variant="primary")
        self.sign_ok.setFixedHeight(34)
        self.sign_ok.setMinimumWidth(90)
        self.sign_ok.clicked.connect(self.validate_and_accept)
        btn_layout.addWidget(self.sign_ok)
        self.sign_ok.setDefault(True)
        self.sign_ok.setFocus()

        card_v.addWidget(btn_row)

        card_row.addWidget(self._card)
        card_row.addStretch()
        outer.addLayout(card_row)
        outer.addStretch()

    def _refresh_icon(self) -> None:
        tok = ThemeManager.instance().tokens()
        color = QtGui.QColor(*tok["accent"])
        if os.path.isfile(_ICON_PATH):
            pm = tinted_icon(_ICON_PATH, color, size=22)
        else:
            pm = QtGui.QPixmap(22, 22)
            pm.fill(QtCore.Qt.GlobalColor.transparent)
        self._icon_label.setPixmap(pm)

    def _on_theme_changed(self, _mode: str) -> None:
        self._title_label.setStyleSheet(dialog_title_qss())
        self._signed_caption.setStyleSheet(dialog_message_qss())
        self.signedInAs.setStyleSheet(dialog_message_qss())
        self.signerInit.setStyleSheet(dialog_message_qss())
        self._refresh_icon()
        self._card.update()

    def _show_error(self, message: str) -> None:
        self._error_label.setText(message)
        self._error_label.show()
        self.sign.set_error(True)
        self._card.adjustSize()

    def _clear_error(self) -> None:
        if not self._error_label.isHidden():
            self._error_label.hide()
            self.sign.set_error(False)
            self._card.adjustSize()

    def switch_user_at_sign_time(self) -> None:
        """Invokes the user profile switching logic.

        Delegates to the `on_switch_user` callback passed to the
        constructor, if any. On a real user change (callback returns
        `(username, initials)`), refreshes the displayed "Signed in as" /
        expected-initials and clears the initials field so the new user can
        sign. Falls back to a placeholder message when no callback was
        provided.
        """
        if self._on_switch_user is None:
            PopUp.information(self, "Switch User", "Switch User functionality invoked.")
            return

        result = self._on_switch_user()
        if result is None:
            return

        self.username, self.expected_initials = result
        self.signedInAs.setText(self.username)
        self.signerInit.setText(f"Initials: <b>{self.expected_initials or 'N/A'}</b>")
        self.sign.clear()
        self._clear_error()

    def validate_and_accept(self) -> None:
        """Validates the input initials against expected session data.

        Checks for empty input and ensures the entered initials match the
        expected initials of the logged-in user (case-insensitive).
        If successful, it calls accept().
        """
        entered_initials = self.sign.text().strip().upper()
        if not entered_initials:
            self._show_error("Please enter your initials to sign.")
            return
        if (
            isinstance(self.expected_initials, str)
            and self.expected_initials != "N/A"
            and entered_initials != self.expected_initials.upper()
        ):
            self._show_error(
                f"Initials do not match the signed in user ({self.expected_initials})."
            )
            return

        self.accept()

    def get_initials(self) -> str:
        """Retrieves the initials entered by the user.

        Returns:
            str: The sanitized, uppercase initials from the input field.
        """
        return self.sign.text().strip().upper()
