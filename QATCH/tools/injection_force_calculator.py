"""injection_force_calculator.py

Provides a thin subclass of :class:`~QATCH.tools.web_viewer.WebViewer`
that opens the hosted Injection Force Calculator single-page application in a
sandboxed, offline-aware Chromium web view inside the nanovisQ application.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-06-03
"""

from QATCH.tools.web_viewer import WebViewer


class InjectionForceCalculatorModule(WebViewer):
    """Tool window for the Injection Force Calculator web application.

    Inherits all navigation sandboxing, offline retry logic, and the 60 FPS
    render loop from :class:`~QATCH.tools.web_viewer.WebViewer`.
    The only responsibility of this subclass is to supply the window title and
    the target URL for the hosted calculator page.
    """

    def __init__(self):
        """Initialise the Injection Force Calculator window.

        Delegates immediately to the base-class constructor with the
        pre-configured title and URL for the hosted calculator page.
        """
        super().__init__(
            title="Injection Force Calculator",
            target_url="https://qatch-technologies.github.io/Injection-Force-Calculator/injection_force_calculator.html",
        )
