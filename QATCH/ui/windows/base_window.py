"""QATCH.ui.windows.base_window

Provides BaseWindow, the shared base class for every top-level window in the
QATCH application.

All windows share one behaviour: closing any of them terminates the
application process after asking the user for confirmation.  Centralising that
logic here means each subclass simply inherits from BaseWindow instead of
duplicating the confirm-and-quit pattern.

Subclasses that need pre-quit cleanup (removing event filters, stopping timers,
etc.) should override `_before_quit` — the base `closeEvent` calls it after
the user confirms and before `QApplication.quit()`.

Setting the instance attribute `close_no_confirm = True` on a subclass
instance bypasses the confirmation dialog entirely, enabling programmatic or
test-driven closure without user interaction.

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol   (paul.macnichol@qatchtech.com)

Date:
    2026-07-01
"""

from PyQt5 import QtGui, QtWidgets

from QATCH.core.constants import Constants
from QATCH.ui.dialogs.pop_up_dialog import PopUp

_QUIT_MESSAGE = "Are you sure you want to quit QATCH Q-1 application now?"


class BaseWindow(QtWidgets.QMainWindow):
    """Base class for all top-level QATCH application windows.

    Provides the standard close-with-confirmation behaviour shared by every
    window in the application.  Subclasses call `super().__init__()` and then
    perform their own initialisation.

    Subclassing:
        - Override :meth:`_before_quit` to run cleanup before the process exits.
        - Set `self.close_no_confirm = True` to skip the confirmation dialog
          for programmatic or automated closes.
    """

    # Set to `True` on an instance to bypass the confirmation dialog on close.
    close_no_confirm: bool

    def _before_quit(self) -> None:
        """Called once after the user confirms quitting, before `QApplication.quit()`.

        The default implementation is a no-op.  Override in subclasses to
        perform cleanup — for example, removing global event filters or stopping
        background timers.
        """

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        """Intercepts close events with an optional confirmation dialog.

        If `self.close_no_confirm` is truthy the dialog is skipped and the
        application exits immediately.  Otherwise the user is prompted; declining
        leaves the window open.

        Args:
            event (QCloseEvent): The Qt close event to accept or ignore.
        """
        if getattr(self, "close_no_confirm", False):
            confirmed = True
        else:
            confirmed = PopUp.question(
                self,
                Constants.app_title,
                _QUIT_MESSAGE,
                True,
            )

        if confirmed:
            self._before_quit()
            event.accept()
            QtWidgets.QApplication.quit()
        else:
            event.ignore()
