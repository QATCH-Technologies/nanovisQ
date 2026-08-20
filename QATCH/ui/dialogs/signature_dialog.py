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
from QATCH.ui.widgets.sliding_panel import SlidingPanel

_ICON_PATH = os.path.join(Architecture.get_path(), "QATCH", "icons", "signature.svg")
_BACK_ICON_PATH = os.path.join(Architecture.get_path(), "QATCH", "icons", "left-arrow.svg")
_CARD_W = 360
_HEADER_H = 52
# Fixed height every page of the sliding panel is sized to (see _build_ui) -
# SlidingPanel.finalize() requires one, since a page-carousel widget can't
# grow to fit whichever page happens to be showing the way a plain layout
# would. Sized to comfortably fit the taller of the two pages (the main
# signing page, worst case with the dev-mode "do not ask again" checkbox
# visible) - trimmed down from an initial too-generous 340 after visual
# review left a large empty gap (each page's body layout ends in
# addStretch(), which absorbs any leftover space above the button row).
_PAGE_H = 300

_SESSION_KEY_PATH = os.path.join(Constants.user_profiles_path, "session.key")


def _read_key(path: str) -> str | None:
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
        parent: Any | None = None,
        on_switch_user: Optional[Callable[[str, str], Optional[Tuple[str, str]]]] = None,
    ) -> None:
        """Initializes the SignatureDialog and loads session metadata.

        Args:
            parent (QWidget, optional): The parent widget for the modal dialog.
            on_switch_user (callable, optional): Invoked with the initials and
                password entered on the dialog's own inline "Switch User" page
                (see `_build_switch_user_page`/`_submit_switch_user`), in
                place of the default placeholder message when not provided.
                Should authenticate those credentials (e.g. via
                `UserProfiles.auth(initials, password, ...)`) and return the
                new `(username, initials)` on a successful switch, or `None`
                if authentication failed or the user didn't actually change -
                callers that also need to update their own app-global state
                (toolbars, session flags, etc.) should do so inside this
                callback. Unlike the previous `UserProfiles.change(...)`-based
                flow, this callback must NOT prompt for credentials itself -
                the dialog now collects them inline.
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

    # Sliding-panel page indices (see _build_ui).
    _P_SIGN = 0
    _P_SWITCH = 1

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

        # "Switch User" used to hand off to UserProfiles.change(), which
        # prompted for initials/password through a pair of raw
        # QInputDialog popups. It now slides this same card sideways to an
        # inline credentials page instead (see _build_switch_user_page) -
        # SlidingPanel (QATCH/ui/widgets/sliding_panel.py) is the same
        # page-carousel widget UILogin's own Sign In <-> Reset Password
        # transition already uses (QATCH/ui/interfaces/ui_login.py), so
        # both places that slide between credential pages behave
        # identically. Every page must share one fixed height (_PAGE_H) -
        # a page-carousel widget can't grow to fit whichever page is
        # currently showing the way a plain layout would.
        self._slider = SlidingPanel(_CARD_W)
        self._slider.add_page(self._build_main_page())
        self._slider.add_page(self._build_switch_user_page())
        self._slider.setFixedHeight(_PAGE_H)
        card_v.addWidget(self._slider)

        card_row.addWidget(self._card)
        card_row.addStretch()
        outer.addLayout(card_row)
        outer.addStretch()

        def _init_slider() -> None:
            self._slider.finalize(_PAGE_H)
            self._slider._inner.move(0, 0)  # land on the sign page, no animation

        QtCore.QTimer.singleShot(0, _init_slider)

    def _build_main_page(self) -> QtWidgets.QWidget:
        """Builds the primary "enter your initials to sign" page.

        Returns:
            The page widget, added to `self._slider` by `_build_ui`.
        """
        page = QtWidgets.QWidget()
        page.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        page_v = QtWidgets.QVBoxLayout(page)
        page_v.setContentsMargins(0, 0, 0, 0)
        page_v.setSpacing(0)

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

        page_v.addWidget(header_w)
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
        # QDialog used to auto-route Enter/Return from a focused, non-button
        # child (like this field) to whichever button was marked default -
        # DialogBase is a plain overlay widget now (see its class docstring
        # in qatch_dialog.py), so that convenience needs wiring explicitly
        # here instead of coming for free.
        self.sign.returnPressed.connect(self.validate_and_accept)
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

        body_layout.addStretch()

        page_v.addWidget(body_w, 1)

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

        page_v.addWidget(btn_row)
        return page

    def _build_switch_user_page(self) -> QtWidgets.QWidget:
        """Builds the inline "sign in as someone else" page.

        Slid into view by `switch_user_at_sign_time` (the main page's
        "Switch User" button) instead of the old
        `UserProfiles.change()`-driven popup pair. Submitting authenticates
        via the `on_switch_user` callback (see `_submit_switch_user`); the
        back button/Cancel return to the main page unchanged (see
        `_cancel_switch_user`).

        Returns:
            The page widget, added to `self._slider` by `_build_ui`.
        """
        page = QtWidgets.QWidget()
        page.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        page_v = QtWidgets.QVBoxLayout(page)
        page_v.setContentsMargins(0, 0, 0, 0)
        page_v.setSpacing(0)

        header_w = QtWidgets.QWidget()
        header_w.setFixedHeight(_HEADER_H)
        header_w.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        header_layout = QtWidgets.QHBoxLayout(header_w)
        header_layout.setContentsMargins(8, 0, 16, 0)
        header_layout.setSpacing(6)

        back_btn = QtWidgets.QToolButton()
        back_btn.setAutoRaise(True)
        back_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        back_btn.setToolTip("Back to Sign")
        if os.path.exists(_BACK_ICON_PATH):
            back_btn.setIcon(QtGui.QIcon(_BACK_ICON_PATH))
            back_btn.setIconSize(QtCore.QSize(16, 16))
        back_btn.clicked.connect(self._cancel_switch_user)
        header_layout.addWidget(back_btn)

        self._switch_title_label = QtWidgets.QLabel("Switch User")
        self._switch_title_label.setWordWrap(True)
        self._switch_title_label.setStyleSheet(dialog_title_qss())
        header_layout.addWidget(self._switch_title_label, 1)

        page_v.addWidget(header_w)

        body_w = QtWidgets.QWidget()
        body_w.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        body_layout = QtWidgets.QVBoxLayout(body_w)
        body_layout.setContentsMargins(20, 16, 20, 0)
        body_layout.setSpacing(10)

        self.switch_initials = QATCHLineEdit()
        self.switch_initials.setMaxLength(4)
        self.switch_initials.setPlaceholderText("Initials")
        self.switch_initials.textChanged.connect(
            lambda _: self.switch_initials.set_error(False)
        )
        body_layout.addWidget(self.switch_initials)

        self.switch_password = QATCHLineEdit()
        self.switch_password.setPlaceholderText("Password")
        self.switch_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.switch_password.textChanged.connect(
            lambda _: self.switch_password.set_error(False)
        )
        body_layout.addWidget(self.switch_password)

        self.switch_initials.returnPressed.connect(self.switch_password.setFocus)
        self.switch_password.returnPressed.connect(self._submit_switch_user)

        self._switch_error_label = QATCHWarningLabel(severity="danger")
        self._switch_error_label.hide()
        body_layout.addWidget(self._switch_error_label)

        body_layout.addStretch()

        page_v.addWidget(body_w, 1)

        btn_row = QtWidgets.QWidget()
        btn_row.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        btn_layout = QtWidgets.QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(20, 16, 20, 20)
        btn_layout.setSpacing(8)
        btn_layout.addStretch()

        switch_cancel = QATCHPushButton("Cancel", variant="secondary")
        switch_cancel.setFixedHeight(34)
        switch_cancel.setMinimumWidth(90)
        switch_cancel.clicked.connect(self._cancel_switch_user)
        btn_layout.addWidget(switch_cancel)

        self.switch_submit = QATCHPushButton("Sign In", variant="primary")
        self.switch_submit.setFixedHeight(34)
        self.switch_submit.setMinimumWidth(90)
        self.switch_submit.clicked.connect(self._submit_switch_user)
        btn_layout.addWidget(self.switch_submit)

        page_v.addWidget(btn_row)
        return page

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
        self._switch_title_label.setStyleSheet(dialog_title_qss())
        self._refresh_icon()
        self._card.update()

    def _show_error(self, message: str) -> None:
        self._error_label.setText(message)
        self._error_label.show()
        self.sign.set_error(True)

    def _clear_error(self) -> None:
        if not self._error_label.isHidden():
            self._error_label.hide()
            self.sign.set_error(False)

    def switch_user_at_sign_time(self) -> None:
        """Slides the card to the inline "Switch User" page.

        Falls back to a placeholder message when no `on_switch_user`
        callback was provided to the constructor - there's nothing for the
        switch-user page's submit action to authenticate against otherwise.
        Resets that page to a blank, error-free state on every reveal, so a
        previous attempt's leftover credentials/errors never resurface.
        """
        if self._on_switch_user is None:
            PopUp.information(self, "Switch User", "Switch User functionality invoked.")
            return

        self.switch_initials.clear()
        self.switch_password.clear()
        self.switch_initials.set_error(False)
        self.switch_password.set_error(False)
        self._switch_error_label.hide()

        self._slider.slide_to(self._P_SWITCH)
        self.switch_initials.setFocus()

    def _submit_switch_user(self) -> None:
        """Authenticates the switch-user page's credentials.

        Mirrors `UILogin.action_sign_in`'s validation/feedback shape
        (QATCH/ui/interfaces/ui_login.py) - empty fields shake and get a red
        error ring in place, rather than the old popup-based flow's
        `QInputDialog` retry loop.

        On success (the `on_switch_user` callback returns a real
        `(username, initials)`), refreshes the main page's displayed
        "Signed in as" / expected-initials, clears its initials field so the
        new user can sign, and slides back. On failure (callback returns
        `None` - wrong credentials or insufficient role), shows an inline
        error on the switch-user page instead and stays there so the user
        can retry immediately.
        """
        initials = self.switch_initials.text().strip()
        password = self.switch_password.text()

        if not initials:
            self.switch_initials.set_error(True)
            self._shake_widget(self.switch_initials)
            return
        if not password:
            self.switch_password.set_error(True)
            self._shake_widget(self.switch_password)
            return

        result = self._on_switch_user(initials, password)
        if result is None:
            self.switch_password.set_error(True)
            self._shake_widget(self.switch_password)
            self.switch_password.clear()
            self.switch_password.setFocus()
            self._switch_error_label.setText("Invalid credentials or insufficient role.")
            self._switch_error_label.show()
            return

        self.username, self.expected_initials = result
        self.signedInAs.setText(self.username)
        self.signerInit.setText(f"Initials: <b>{self.expected_initials or 'N/A'}</b>")
        self.sign.clear()
        self._clear_error()

        self._cancel_switch_user()

    def _cancel_switch_user(self) -> None:
        """Slides back to the main signing page without switching users."""
        self.switch_password.clear()
        self._slider.slide_to(self._P_SIGN)
        self.sign.setFocus()

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
