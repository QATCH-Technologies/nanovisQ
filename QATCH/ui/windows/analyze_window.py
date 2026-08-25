"""
QATCH.ui.windows.analyze_window.py

Top-level window wrapper for the analysis interface.

This module provides :class:`AnalyzeWindow`, a lightweight
:class:`BaseWindow` subclass that hosts the :class:`UIAnalyze` interface and
connects it to the application's main window.

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol   (paul.macnichol@qatchtech.com)

Date:
    2026-07-01
"""

from typing import TYPE_CHECKING

from QATCH.ui.interfaces import UIAnalyze
from QATCH.ui.windows.base_window import BaseWindow

if TYPE_CHECKING:
    from QATCH.ui.main_window import MainWindow


class AnalyzeWindow(BaseWindow):
    """Main application window for the analysis interface.

    Wraps :class:`UIAnalyze` in a :class:`BaseWindow` and initializes the
    analysis interface with a reference to the application's main window.

    Args:
        parent: The application's main window used by the analysis UI for
            parent-window coordination and access to shared application
            state.
    """

    def __init__(self, parent: "MainWindow") -> None:
        """Initialize the analysis window and its user interface.

        Creates the :class:`UIAnalyze` interface, initializes it against this
        window and the application's main window, and installs the interface as
        the window's central widget.

        Args:
            parent: The application's main window passed to the analysis UI
                during setup.
        """
        super().__init__()
        self.ui = UIAnalyze()
        self.ui.setup_ui(self, parent)
        self.setCentralWidget(self.ui)
