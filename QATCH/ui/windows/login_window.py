"""login_window.py

This module provides the LoginWindow class, which manages the main user
authentication screen, handling keyboard navigation, CapsLock state tracking,
and application exit confirmations.

Author(s):
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-07-01
"""

from typing import TYPE_CHECKING

from PyQt5 import QtCore, QtGui

from QATCH.core.constants import Constants
from QATCH.ui.interfaces import UILogin
from QATCH.ui.windows.base_window import BaseWindow

if TYPE_CHECKING:
    from QATCH.ui.main_window import MainWindow


class LoginWindow(BaseWindow):
    """A main window subclass representing the user login screen.

    This window manages the login interface, intercepts keyboard events to
    provide quality-of-life shortcuts (like hitting Enter to submit), and
    monitors the OS for CapsLock changes to warn the user.

    Attributes:
        ui (UILogin): The user interface setup instance for the login screen.
    """

    def __init__(self, parent: "MainWindow") -> None:
        """Initializes the LoginWindow and sets up its UI components.

        Args:
            parent (MainWindow): The main application window that spawned or
                owns this login window.
        """
        super().__init__()
        self.ui = UILogin()
        self.ui.setup_ui(self, parent)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        """Filter system events for login UI behavior and keyboard shortcuts.

        Intercepts keyboard input to support form submission, form clearing,
        and Caps Lock state synchronization. Also updates UI state when the
        window regains focus to ensure OS-level Caps Lock status is accurate.

        Args:
            obj: QObject receiving the event.
            event: Qt event triggered by user input or system state changes.

        Returns:
            True if the event is handled and should not propagate further;
            False otherwise.
        """
        if isinstance(event, QtGui.QKeyEvent):
            if event.type() == QtCore.QEvent.Type.KeyPress:
                # Submit or focus password field on Enter/Return
                if event.key() in (QtCore.Qt.Key.Key_Enter, QtCore.Qt.Key.Key_Return):
                    if len(self.ui.user_password.text()) == 0:
                        self.ui.user_password.setFocus()
                    else:
                        self.ui.action_sign_in()
                    return True

                # Clear login form on Escape
                if event.key() == QtCore.Qt.Key.Key_Escape:
                    self.ui.clear_form()
                    return True

            # Update Caps Lock state on key release
            if event.type() == QtCore.QEvent.Type.KeyRelease:
                if event.key() == QtCore.Qt.Key.Key_CapsLock:
                    self.ui.caps_lock_on = Constants.windll_is_caps_lock_on()
                    self.ui.update_caps_lock_state(self.ui.caps_lock_on)

        # Sync Caps Lock state when window regains focus (standard QEvent)
        if event.type() == QtCore.QEvent.Type.FocusIn:
            self.ui.caps_lock_on = Constants.windll_is_caps_lock_on()
            self.ui.update_caps_lock_state(self.ui.caps_lock_on)

        return super().eventFilter(obj, event)
