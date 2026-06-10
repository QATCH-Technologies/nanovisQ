"""donnan_gibbs_calculator.py

Provides a thin subclass of :class:`~QATCH.tools.web_viewer.WebViewer`
that opens the hosted Donnan Calculator single-page application in a sandboxed,
offline-aware Chromium web view inside the nanovisQ application.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-06-03
"""

from QATCH.tools.web_viewer import WebViewer


class DonnanCalculatorModule(WebViewer):
    """Tool window for the Donnan-Gibbs Calculator web application.

    Inherits all navigation sandboxing, offline retry logic, and the 60 FPS
    render loop from :class:`~QATCH.tools.web_viewer.WebViewer`.
    The only responsibility of this subclass is to supply the window title and
    the target URL for the hosted calculator page.
    """

    def __init__(self):
        """Initialise the Donnan Calculator window.

        Delegates immediately to the base-class constructor with the
        pre-configured title and URL for the hosted calculator page.
        """
        super().__init__(
            title="Donnan Calculator",
            target_url="https://qatch-technologies.github.io/Donnan-Calculator/donnan_calculator.html",
        )
