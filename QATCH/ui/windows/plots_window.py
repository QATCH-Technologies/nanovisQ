"""QATCH.ui.windows.plots_window.py

This module provides the primary window container for displaying analytical
plots within the QATCH application. It encapsulates the plot interface
components and provides window lifecycle management, specifically
handling user exit confirmation to prevent accidental data loss or
unexpected application termination.

Author(s)
    Alexander Ross  (alexander.ross@qatchtech.com)
    Paul MacNichol  (paul.macnichol@qatchtech.com)

Date:
    2026-07-01
"""

from QATCH.core.constants import Constants
from QATCH.ui.interfaces import UIPlots
from QATCH.ui.windows.base_window import BaseWindow


class PlotsWindow(BaseWindow):
    """A main window for displaying analytical plots.

    This class initializes the plot interface and manages the application
    exit workflow by prompting the user for confirmation before closing.

    Attributes:
        ui (UIPlots): The interface component containing the plot
            layout and configuration.
    """

    def __init__(self, samples: int = Constants.argument_default_samples):
        """Initializes the PlotsWindow with a specific number of samples.

        Args:
            samples (int, optional): The default number of samples to use
                for plot initialization. Defaults to
                Constants.argument_default_samples.
        """
        super().__init__()
        self.ui = UIPlots()
        self.ui.setup_ui(self)
