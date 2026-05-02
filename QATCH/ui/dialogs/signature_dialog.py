import os

from contextlib import suppress
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QMessageBox,
    QLineEdit,
    QCheckBox,
)

from PyQt5.QtCore import (
    Qt,
)
from PyQt5.QtGui import (
    QIcon,
)
from typing import Optional, Any
import pyqtgraph as pg


from QATCH.core.constants import Constants
from QATCH.common.architecture import Architecture
from QATCH.common.userProfiles import UserProfiles


class SignatureDialog(QDialog):
    """A self-contained modal dialog to capture a user's signature/initials.

    This dialog verifies the identity of the current user by requiring their
    initials. it supports 'Dev Mode' auto-signing and allows users to switch
    profiles if the current session info is incorrect.

    Attributes:
        username (str): The display name of the currently logged-in user.
        expected_initials (Optional[str]): The initials required to pass validation.
        sign (QLineEdit): Input field for user initials.
        sign_do_not_ask (QCheckBox): Toggle for persistent session signing.
    """

    def __init__(self, parent: Optional[Any] = None) -> None:
        """Initializes the SignatureDialog and loads session metadata.

        Args:
            parent (QWidget, optional): The parent widget for the modal dialog.
        """
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Dialog)
        self.setWindowTitle("Signature")
        self.setModal(True)

        # NOTE: Sizing may need to be updated!
        self.setFixedWidth(240)
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

        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "sign.png")
        self.setWindowIcon(QIcon(icon_path))

        layout_sign = QVBoxLayout(self)
        layout_sign.setContentsMargins(12, 12, 12, 12)
        layout_sign.setSpacing(8)

        layout_curr = QHBoxLayout()
        signed_in_as_label = QLabel("Signed in as: ")
        signed_in_as_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout_curr.addWidget(signed_in_as_label)

        self.signedInAs = QLabel(self.username)
        self.signedInAs.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout_curr.addWidget(self.signedInAs)
        layout_sign.addLayout(layout_curr)

        line_sep = QFrame()
        line_sep.setFrameShape(QFrame.HLine)
        line_sep.setFrameShadow(QFrame.Sunken)
        layout_sign.addWidget(line_sep)

        layout_switch = QHBoxLayout()
        self.signerInit = QLabel(f"Initials: <b>{self.expected_initials or 'N/A'}</b>")
        layout_switch.addWidget(self.signerInit)

        switch_user = QPushButton("Switch User")
        switch_user.clicked.connect(self.switch_user_at_sign_time)
        layout_switch.addWidget(switch_user)
        layout_sign.addLayout(layout_switch)

        self.sign = QLineEdit()
        self.sign.setMaxLength(4)
        self.sign.setPlaceholderText("Initials")
        self.sign.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout_sign.addWidget(self.sign)

        self.sign_do_not_ask = QCheckBox("Do not ask again this session")
        self.sign_do_not_ask.setEnabled(False)

        # Dev Mode Logic
        try:
            if UserProfiles.checkDevMode()[0]:
                auto_sign_key = None
                session_key = None
                if os.path.exists(Constants.auto_sign_key_path):
                    with open(Constants.auto_sign_key_path, "r") as f:
                        auto_sign_key = f.readline()

                session_key_path = os.path.join(Constants.user_profiles_path, "session.key")
                if os.path.exists(session_key_path):
                    with open(session_key_path, "r") as f:
                        session_key = f.readline()

                if auto_sign_key == session_key and session_key is not None:
                    self.sign_do_not_ask.setChecked(True)
                else:
                    self.sign_do_not_ask.setChecked(False)
                    if os.path.exists(Constants.auto_sign_key_path):
                        os.remove(Constants.auto_sign_key_path)
        except NameError:
            suppress(NameError)
            pass

        if self.sign_do_not_ask.isEnabled() or self.sign_do_not_ask.isChecked():
            layout_sign.addWidget(self.sign_do_not_ask)
        else:
            self.sign_do_not_ask.hide()

        # Buttons
        self.sign_ok = QPushButton("OK")
        self.sign_ok.setDefault(True)
        self.sign_ok.setAutoDefault(True)

        self.sign_cancel = QPushButton("Cancel")

        layout_ok_cancel = QHBoxLayout()
        layout_ok_cancel.addWidget(self.sign_ok)
        layout_ok_cancel.addWidget(self.sign_cancel)
        layout_sign.addLayout(layout_ok_cancel)

        self.sign_ok.clicked.connect(self.validate_and_accept)
        self.sign_cancel.clicked.connect(self.reject)

    def switch_user_at_sign_time(self) -> None:
        """Invokes the user profile switching logic.

        Displays an information dialog to the user indicating functionality invocation.

        TODO: Needs to be implemented fully!
        """
        QMessageBox.information(self, "Switch User", "Switch User functionality invoked.")

    def validate_and_accept(self) -> None:
        """Validates the input initials against expected session data.

        Checks for empty input and ensures the entered initials match the
        expected initials of the logged-in user (case-insensitive).
        If successful, it calls accept().
        """
        entered_initials = self.sign.text().strip().upper()
        if not entered_initials:
            QMessageBox.warning(self, "Required", "Please enter your initials to sign.")
            return
        if (
            isinstance(self.expected_initials, str)
            and self.expected_initials != "N/A"
            and entered_initials != self.expected_initials.upper()
        ):
            QMessageBox.warning(
                self,
                "Mismatch",
                f"Initials do not match the signed in user ({self.expected_initials}).",
            )
            return

        self.accept()

    def get_initials(self) -> str:
        """Retrieves the initials entered by the user.

        Returns:
            str: The sanitized, uppercase initials from the input field.
        """
        return self.sign.text().strip().upper()
