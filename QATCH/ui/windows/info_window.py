"""info_window.py

This module provides the :class:`InfoWindow`, which hosts the
application information interface and manages user confirmation when
attempting to close the application.

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-07-01
"""

from QATCH.core.constants import Constants
from QATCH.ui.interfaces import UIInfo
from QATCH.ui.windows.base_window import BaseWindow


class InfoWindow(BaseWindow):
    """Main information window for the QATCH Q-1 application.

    This window displays application information through the
    :class:`UIInfo` interface and prompts the user for confirmation
    before exiting the application.

    Attributes:
        ui: User interface instance responsible for creating and
            managing the window widgets.
    """

    def __init__(
        self,
        samples: int = Constants.argument_default_samples,
    ) -> None:
        """Initialize the information window.

        Args:
            samples: Number of samples to configure. This argument is
                currently retained for API compatibility but is not used
                by this class.
        """
        super().__init__()

        self.ui: UIInfo = UIInfo()
        self.ui.setup_ui(self)
