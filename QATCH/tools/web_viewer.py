import sys
import os
from PyQt5 import QtCore
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebEngineWidgets import QWebEnginePage
from PyQt5.QtCore import QUrl, QTimer, Qt
from PyQt5.QtGui import QIcon, QPixmap, QTransform
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
    def __init__(self, allowed_url, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_url = allowed_url

    def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
        print(f"DEBUG: Type={navigation_type}, URL={url.toString()}")
        req_url = url.toString()
        if req_url.startswith(self.allowed_url):
            return True
        if navigation_type == QWebEnginePage.NavigationTypeOther:
            return True
        return False

    def createWindow(self, window_type):
        return None


class SpinningIconButton(QPushButton):
    """A custom button that rotates its icon continuously with a parabolic speed curve."""

    def __init__(self, icon_path, parent=None):
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
        self.angle = 0.0
        self.timer.start(30)  # ~33fps update rate

    def stop_spin(self):
        self.timer.stop()
        self.angle = 0.0
        if not self.base_pixmap.isNull():
            self.setIcon(QIcon(self.base_pixmap))

    def _rotate(self):
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


class OfflineAwareWebView(QWebEngineView):
    def __init__(self, url=None):
        super().__init__()
        self.target_url = url if url else ""

        if url:
            self.setPage(RestrictedPage(url, self))
            self.setUrl(QUrl(url))
        self.page().profile().clearHttpCache()
        self.loadStarted.connect(self._load_started)
        self.loadFinished.connect(self._load_finished)

        if url:
            self.setUrl(QUrl(url))

    def setUrl(self, url):
        self.target_url = url.toString()
        super().setUrl(url)

    def _load_started(self):
        pass

    def _load_finished(self, success):
        if not success:
            self.show_offline_page()

    def show_offline_page(self):
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


class BaseAcceleratedCalculator(QMainWindow):
    """Base module containing the optimized layout and render logic."""

    def __init__(self, title, target_url):
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

        # Set up the Chromium browser view
        self.browser = OfflineAwareWebView()

        # Set up the bottom status bar layout
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(5, 2, 5, 2)

        self.refresh_btn = SpinningIconButton(
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

        # Load the targeted URL
        self.browser.setUrl(QUrl(target_url))

    def _trigger_refresh(self):
        """Forces the web engine to reload the current or target URL."""
        if not self.loaded:
            self.browser.setUrl(QUrl(self.browser.target_url))
        else:
            self.browser.reload()

    def _load_started(self):
        self.status_label.setText("Reloading...")
        self.refresh_btn.start_spin()
        self.loaded = False
        self.render_timer.stop()  # Pause heavy rendering while loading

    def _load_finished(self, success):
        self.refresh_btn.stop_spin()
        if success:
            self.status_label.setText("Reload")
            self.loaded = True

            # Start the optimized rendering loop
            self.render_timer.start()
        else:
            self.status_label.setText("Connection lost...")

    def closeEvent(self, event):
        self.closing = True
        self.render_timer.stop()  # Safely kill the timer
        event.accept()
