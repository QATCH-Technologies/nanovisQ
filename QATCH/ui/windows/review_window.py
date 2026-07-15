"""review_window.py

This module provides the :class:`ReviewWindow`, which hosts the placeholder
"Review" mode interface pending future support for multi-dataset cross
analysis.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-07-14
"""

from QATCH.ui.interfaces import UIReview
from QATCH.ui.windows.base_window import BaseWindow


class ReviewWindow(BaseWindow):
    """Placeholder window for the QATCH Q-1 Review mode.

    Multi-dataset cross analysis is not yet implemented; this window
    displays a "coming soon" placeholder via the :class:`UIReview`
    interface until that feature is built out.

    Attributes:
        ui: User interface instance responsible for creating and
            managing the window widgets.
    """

    def __init__(self) -> None:
        """Initialize the review window."""
        super().__init__()

        self.ui: UIReview = UIReview()
        self.ui.setup_ui(self)
