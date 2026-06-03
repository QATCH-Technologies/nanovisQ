"""web_viewer.py

This module provides a set of PyQt5 widgets that embed a Chromium-based web
view and wrap it with connectivity awareness, navigation sandboxing, and a
smooth render loop.  The intended use case is displaying single-page web
applications (e.g. calculator tools served by a local HTTP server) inside a
native QMainWindow without giving those pages the ability to navigate away or
open new browser windows.

Classes:
    RestrictedPage: A QWebEnginePage that limits navigation to one allowed URL.
    RefreshButton: A QPushButton whose icon rotates with a parabolic speed
        curve to signal an in-progress load.
    OfflineViewer: A QWebEngineView with automatic retry logic and a
        styled offline/error page.
    WebViewer: A QMainWindow that composes the above widgets
        with a status bar and a 60 FPS render timer.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-06-03
"""

import os
from PyQt5 import QtCore
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebEngineWidgets import QWebEnginePage
from PyQt5.QtCore import QUrl, QTimer, Qt
from PyQt5.QtGui import QIcon, QPixmap, QTransform, QDesktopServices
from PyQt5.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
)
import base64
from QATCH.common.architecture import Architecture


class RestrictedPage(QWebEnginePage):
    """A QWebEnginePage that sandboxes navigation to a single allowed URL.

    All main-frame navigations are checked against ``allowed_url``.
    Navigations that would leave that page are blocked and the
    ``navigation_blocked`` flag is set so that callers can distinguish a
    deliberate block from a genuine network failure.  ``mailto:`` links are
    forwarded to the OS mail client rather than opening inside the view.
    New-window creation requests are suppressed entirely.

    Attributes:
        allowed_url (str): The URL string that main-frame navigation is
            restricted to.
        navigation_blocked (bool): Set to ``True`` whenever a main-frame
            navigation is refused, reset to ``False`` on allowed navigations.
    """

    def __init__(self, allowed_url, *args, **kwargs):
        """Initialise the page and store the URL whitelist entry.

        Args:
            allowed_url (str): The sole URL that main-frame navigation is
                permitted to reach.
            *args: Positional arguments forwarded to ``QWebEnginePage``.
            **kwargs: Keyword arguments forwarded to ``QWebEnginePage``.
        """
        super().__init__(*args, **kwargs)
        self.allowed_url = allowed_url
        self._allowed_qurl = QUrl(allowed_url)
        # Set True whenever we refuse a main-frame navigation, so the view can
        # distinguish a deliberate block from a real load failure.
        self.navigation_blocked = False

    def _same_page(self, url):
        """Check whether *url* matches the allowed URL on scheme, host and path.

        Args:
            url (QUrl): The URL to test.

        Returns:
            bool: ``True`` if *url* has the same scheme, host, and path as the
            allowed URL; ``False`` otherwise.
        """
        return (
            url.scheme() == self._allowed_qurl.scheme()
            and url.host() == self._allowed_qurl.host()
            and url.path() == self._allowed_qurl.path()
        )

    def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
        """Decide whether to allow a navigation request.

        Navigation policy:

        * ``mailto:`` URLs are forwarded to the OS mail client and blocked
          inside the web view; ``navigation_blocked`` is set so that the
          resulting ``loadFinished(False)`` signal is not misinterpreted as a
          network error.
        * Internal engine schemes (``data``, ``about``, ``blob``, ``qrc``) are
          always permitted.
        * Main-frame navigations are only allowed when the destination matches
          the allowed URL exactly (scheme + host + path).
        * Sub-frame navigations triggered by a link click are restricted to the
          same page; all other sub-frame requests (e.g. asset loads) are
          allowed.

        Args:
            url (QUrl): The target URL of the navigation request.
            navigation_type (QWebEnginePage.NavigationType): The kind of
                navigation event that triggered the request.
            is_main_frame (bool): ``True`` if the request targets the main
                frame of the page.

        Returns:
            bool: ``True`` to allow the navigation, ``False`` to block it.
        """
        # Hand mailto: links to the OS mail client, don't navigate the view.
        # Flag as blocked so a resulting loadFinished(False) is not treated
        # as a connection failure.
        if url.scheme() == "mailto":
            QDesktopServices.openUrl(url)
            self.navigation_blocked = True
            return False

        # Internal schemes used by the offline page / engine.
        if url.scheme() in ("data", "about", "blob", "qrc"):
            return True

        # Any main-frame navigation must stay on the allowed page.
        if is_main_frame:
            allowed = self._same_page(url)
            self.navigation_blocked = not allowed
            return allowed

        # Sub-frames: allow assets/other, block cross-page link clicks.
        if navigation_type == QWebEnginePage.NavigationTypeLinkClicked:
            return self._same_page(url)
        return True

    def createWindow(self, window_type):
        """Suppress all requests to open a new browser window or tab.

        Args:
            window_type (QWebEnginePage.WebWindowType): The type of window
                requested by the page (e.g. tab, dialog).

        Returns:
            None: Always returns ``None`` to prevent window creation.
        """
        return None


class RefreshButton(QPushButton):
    """A QPushButton whose icon spins continuously with a parabolic speed curve.

    The rotation speed varies over each full revolution: it is slowest at the
    bottom of the cycle (0°/360°) and fastest at the top (180°), following a
    ``speed = k·x² + min_speed`` curve.  This gives a more organic feel than a
    constant-rate spin.  The animation is driven by an internal ``QTimer`` and
    is started/stopped explicitly via :meth:`start_spin` and
    :meth:`stop_spin`.

    Attributes:
        icon_path (str): Filesystem path to the source icon image.
        base_pixmap (QPixmap): The unrotated source pixmap used as the basis
            for each rotated frame.
        angle (float): Current rotation angle in degrees (0-360).
        timer (QTimer): Internal timer that drives the per-frame rotation.
    """

    def __init__(self, icon_path, parent=None):
        """Create a RefreshButton with the given icon.

        Args:
            icon_path (str): Path to the icon image file to display and spin.
            parent (QWidget, optional): Parent widget. Defaults to ``None``.
        """
        super().__init__(parent)
        self.icon_path = icon_path
        self.base_pixmap = QPixmap(self.icon_path)
        self.setIcon(QIcon(self.base_pixmap))
        self.setIconSize(QtCore.QSize(16, 16))
        self.setFixedSize(24, 24)

        self.setStyleSheet("border: none; background: transparent;")
        self.setCursor(Qt.PointingHandCursor)

        self.angle = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._rotate)

    def start_spin(self):
        """Reset the angle and start the spinning animation at ~33 FPS."""
        self.angle = 0.0
        self.timer.start(30)  # ~33fps update rate

    def stop_spin(self):
        """Stop the spinning animation and restore the icon to its upright position."""
        self.timer.stop()
        self.angle = 0.0
        if not self.base_pixmap.isNull():
            self.setIcon(QIcon(self.base_pixmap))

    def _rotate(self):
        """Advance the rotation by one parabolic speed step and repaint the icon.

        Called on each timer tick.  Maps the current angle to a normalised
        position ``x ∈ [-1, 1]`` and computes the angular step as
        ``(max_speed - min_speed) * x² + min_speed``, producing the slowest
        speed at 0°/360° and the fastest at 180°.  If the base pixmap is
        invalid the method returns immediately without doing anything.
        """
        if self.base_pixmap.isNull():
            return

        # Map angle 0->360 to a range of -1 to 1 (center of parabola at 180 degrees)
        x = (self.angle - 180.0) / 180.0

        # Parabolic speed mapping: speed = k * x^2 + min_speed
        min_speed = 2.0  # Degrees to turn when at the very bottom (slowest)
        max_speed = 25.0  # Degrees to turn when at the very top (fastest)
        speed = (max_speed - min_speed) * (x**2) + min_speed

        self.angle = (self.angle + speed) % 360
        transform = QTransform().rotate(self.angle)

        rotated_pixmap = self.base_pixmap.transformed(transform, Qt.SmoothTransformation)
        self.setIcon(QIcon(rotated_pixmap))


class OfflineViewer(QWebEngineView):
    """A QWebEngineView that retries failed loads and shows a styled offline page.

    On a load failure the view attempts up to ``_max_retries`` automatic
    retries with a 1.5-second delay between each attempt.  If all retries are
    exhausted it renders a self-contained HTML offline/error page that
    auto-redirects back to the target URL every 5 seconds.  Failures caused by
    intentionally blocked navigations (as reported by :class:`RestrictedPage`)
    are silently ignored and do not trigger retries.

    Attributes:
        target_url (str): The URL string that the view is currently targeting.
        _retry_count (int): Number of retry attempts made for the current load.
        _max_retries (int): Maximum number of automatic retries before the
            offline page is shown.
        _retry_timer (QTimer): Single-shot timer used to space out retry
            attempts.
    """

    def __init__(self, url=None):
        """Initialise the view, attach a RestrictedPage, and begin loading.

        The HTTP cache is cleared on startup so that stale cached content from
        a previous session cannot mask connectivity problems.  If *url* is
        provided, a :class:`RestrictedPage` is installed before the first load
        so that the whitelist is enforced from the very first navigation.

        Args:
            url (str, optional): The URL to load immediately. If ``None`` the
                view is created in an empty state. Defaults to ``None``.
        """
        super().__init__()
        self.target_url = url if url else ""
        self._retry_count = 0
        self._max_retries = 3
        self._retry_timer = QTimer(self)
        self._retry_timer.setSingleShot(True)
        self._retry_timer.timeout.connect(self._retry_load)

        if url:
            self.setPage(RestrictedPage(url, self))

        self.page().profile().clearHttpCache()
        self.loadStarted.connect(self._load_started)
        self.loadFinished.connect(self._load_finished)

        if url:
            self.setUrl(QUrl(url))

    def setUrl(self, url):
        """Override setUrl to keep ``target_url`` in sync before delegating.

        Args:
            url (QUrl): The new URL to navigate to.
        """
        self.target_url = url.toString()
        super().setUrl(url)

    def _load_started(self):
        """Slot called when a page load begins. Reserved for future use."""
        pass

    def _retry_load(self):
        """Retry loading ``target_url`` after a short delay.

        Triggered by ``_retry_timer``.  Bypasses the :meth:`setUrl` override
        so the ``target_url`` attribute is not reset during a retry cycle.
        """
        if self.target_url:
            super().setUrl(QUrl(self.target_url))

    def _load_finished(self, success):
        """Handle the outcome of a page load attempt.

        On success the retry counter is reset.  On failure the method
        distinguishes three cases:

        1. **Blocked navigation** - ``RestrictedPage.navigation_blocked`` is
           ``True``; treated as a non-error and ignored.
        2. **Transient failure within retry budget** - the retry counter is
           incremented and ``_retry_timer`` is armed for another attempt after
           1.5 seconds.
        3. **Retry budget exhausted** - :meth:`show_offline_page` is called.

        Args:
            success (bool): ``True`` if the page loaded successfully.
        """
        if success:
            self._retry_count = 0
            return
        # Deliberately blocked navigation — not a failure.
        if getattr(self.page(), "navigation_blocked", False):
            return
        # Transient failure (e.g. slow first load at startup): retry a few
        # times with a short delay before falling back to the offline page.
        if self._retry_count < self._max_retries:
            self._retry_count += 1
            self._retry_timer.start(1500)  # 1.5s grace before retry
            return
        self.show_offline_page()

    def show_offline_page(self):
        """Render a styled offline error page directly into the web view.

        Reads the offline SVG icon from the application's icons directory,
        encodes it as a base64 data URI, and injects it into a self-contained
        HTML page.  The page includes a JavaScript snippet that redirects back
        to ``target_url`` after 5 seconds so that the view recovers
        automatically once connectivity is restored.

        If the offline icon file cannot be found a warning is printed and the
        icon ``src`` attribute is left empty; the rest of the page still
        renders correctly.
        """
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "offline.svg")

        # Safely read and encode the SVG
        try:
            with open(icon_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                # Format as a Data URI
                img_src = f"data:image/svg+xml;base64,{encoded_string}"
        except FileNotFoundError:
            print(f"Warning: Offline icon not found at {icon_path}")
            img_src = ""

        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    background-color: #f8f9fa;
                    color: #343a40;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                }}
                .error-card {{
                    text-align: center;
                    padding: 40px;
                    background: #ffffff;
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                    max-width: 400px;
                }}
                .icon {{
                    width: 64px;
                    height: 64px;
                    /* These two lines force the icon to perfectly center */
                    display: block;
                    margin: 0 auto 16px auto;
                }}
                h1 {{
                    font-size: 24px;
                    margin: 0 0 12px 0;
                    color: #212529;
                }}
                p {{
                    font-size: 15px;
                    color: #6c757d;
                    line-height: 1.5;
                    margin: 0;
                }}
                .loading-dots:after {{
                    content: ' .';
                    animation: dots 1.5s steps(5, end) infinite;
                }}
                @keyframes dots {{
                    0%, 20% {{ color: rgba(0,0,0,0); text-shadow: .25em 0 0 rgba(0,0,0,0), .5em 0 0 rgba(0,0,0,0); }}
                    40% {{ color: #6c757d; text-shadow: .25em 0 0 rgba(0,0,0,0), .5em 0 0 rgba(0,0,0,0); }}
                    60% {{ text-shadow: .25em 0 0 #6c757d, .5em 0 0 rgba(0,0,0,0); }}
                    80%, 100% {{ text-shadow: .25em 0 0 #6c757d, .5em 0 0 #6c757d; }}
                }}
            </style>
        </head>
        <body>
            <div class="error-card">
                <img class="icon" src="{img_src}" alt="Connection Lost">
                <h1>Connection Lost</h1>
                <p>Unable to reach the calculator. Retrying automatically<span class="loading-dots"></span></p>
            </div>

            <script>
                // Attempt to reach the target URL every 5 seconds
                setTimeout(function() {{
                    window.location.href = '{self.target_url}';
                }}, 5000);
            </script>
        </body>
        </html>
        """
        self.setHtml(html_template)


class WebViewer(QMainWindow):
    """QMainWindow hosting a sandboxed web view with a 60 FPS render loop.

    Composes an :class:`OfflineViewer` (with an embedded
    :class:`RestrictedPage`) with a thin status bar containing a
    :class:`RefreshButton` refresh control and a status label.  A
    persistent ``QTimer`` drives ``QWebEngineView.update()`` at approximately
    60 FPS while a page is loaded so that animated content renders smoothly;
    the timer is suspended during page loads to avoid unnecessary GPU work.

    Attributes:
        loaded (bool): ``True`` when the current page has finished loading
            successfully.
        closing (bool): Set to ``True`` in :meth:`closeEvent` so that
            in-flight callbacks can detect that the window is being torn down.
        browser (OfflineViewer): The embedded Chromium web view.
        refresh_btn (RefreshButton): The animated refresh / reload button.
        status_label (QLabel): Text label showing the current load status.
        render_timer (QTimer): Persistent timer that triggers repaints at
            ~60 FPS.
    """

    def __init__(self, title, target_url):
        """Create and lay out the calculator window.

        Constructs the central widget hierarchy (browser + status bar),
        wires up all signals, and starts the initial page load.  The
        :class:`OfflineViewer` constructor performs the first
        ``setUrl`` call so no additional navigation is needed here.

        Args:
            title (str): Window title text shown in the title bar.
            target_url (str): The URL to load and restrict navigation to.
        """
        super().__init__()
        self.loaded = False
        self.closing = False

        self.setWindowTitle(title)
        self.setGeometry(100, 100, 1024, 768)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Set up the Chromium browser view WITH the restricted page installed.
        # Passing target_url ensures RestrictedPage is attached before any load.
        self.browser = OfflineViewer(target_url)

        # Set up the bottom status bar layout
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(5, 2, 5, 2)

        self.refresh_btn = RefreshButton(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "refresh-cw.svg")
        )
        self.refresh_btn.clicked.connect(self._trigger_refresh)

        self.status_label = QLabel("Ready")
        self.status_label.setFixedHeight(15)

        status_layout.addWidget(self.refresh_btn)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()

        # Add widgets to main layout
        layout.addWidget(self.browser)
        layout.addLayout(status_layout)

        # Connect browser signals
        self.browser.loadStarted.connect(self._load_started)
        self.browser.loadFinished.connect(self._load_finished)

        # --- OPTIMIZED RENDER LOOP ---
        # Create a persistent timer instead of recursive singleShots
        self.render_timer = QTimer(self)
        self.render_timer.setTimerType(Qt.PreciseTimer)  # Prevents OS timer drifting
        self.render_timer.timeout.connect(self.browser.update)

        # 16ms = ~60 FPS (Standard smooth rendering)
        # 8ms = ~125 FPS (If you want ultra-high refresh rate support)
        self.render_timer.setInterval(16)

        # OfflineViewer(target_url) already loads the URL in its
        # constructor, so no extra setUrl call is needed here.

    def _trigger_refresh(self):
        """Forces the web engine to reload the current or target URL."""
        if not self.loaded:
            self.browser.setUrl(QUrl(self.browser.target_url))
        else:
            self.browser.reload()

    def _load_started(self):
        """Slot called when the browser begins loading a page.

        Updates the status label, starts the refresh button spin animation,
        marks the window as not-yet-loaded, and pauses the render timer to
        avoid unnecessary GPU work during the load.
        """
        self.status_label.setText("Reloading...")
        self.refresh_btn.start_spin()
        self.loaded = False
        self.render_timer.stop()  # Pause heavy rendering while loading

    def _load_finished(self, success):
        """Slot called when the browser finishes (or fails) a page load.

        Stops the refresh button spin and updates the status label and render
        timer according to the outcome:

        * **Success** or **blocked navigation** - label set to "Reload",
          ``loaded`` flag set, render timer started.
        * **Retry in progress** - label set to "Connecting..." with no alarm.
        * **All retries exhausted** - label set to "No connection...".

        Args:
            success (bool): ``True`` if the page loaded without error.
        """
        self.refresh_btn.stop_spin()
        if success:
            self.status_label.setText("Reload")
            self.loaded = True
            self.render_timer.start()
        elif getattr(self.browser.page(), "navigation_blocked", False):
            self.status_label.setText("Reload")
            self.loaded = True
            self.render_timer.start()
        elif self.browser._retry_count < self.browser._max_retries:
            # Retry in progress — keep the spinner-style status, don't alarm.
            self.status_label.setText("Connecting...")
        else:
            self.status_label.setText("No connection...")

    def closeEvent(self, event):
        """Handle the window close event by stopping the render timer.

        Sets ``closing`` to ``True`` so that any in-flight asynchronous
        callbacks can detect that the window is being destroyed, then stops
        the render timer before accepting the event.

        Args:
            event (QCloseEvent): The close event to accept.
        """
        self.closing = True
        self.render_timer.stop()  # Safely kill the timer
        event.accept()
