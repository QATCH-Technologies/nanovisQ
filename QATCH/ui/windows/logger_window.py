"""
Logger Window Module for QATCH.

This module provides the LoggerWindow class, which represents a main application
window that includes a confirmation prompt before closing to prevent accidental exits.

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-07-01
"""

from QATCH.ui.interfaces import UILogger
from QATCH.ui.windows.base_window import BaseWindow


class LoggerWindow(BaseWindow):
    """A main window subclass tailored for logging or application monitoring.

    This window initializes the logger UI and overrides the default close event
    to prompt the user for confirmation before quitting the application.

    Attributes:
        ui (UILogger): The user interface setup instance for the logger.
    """

    def __init__(self) -> None:
        """Initializes the LoggerWindow and sets up its UI components."""
        super().__init__()
        self.ui = UILogger()
        self.ui.setup_ui(self)
