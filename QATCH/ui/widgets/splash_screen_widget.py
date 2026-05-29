import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtWidgets import QApplication, QSplashScreen, QProgressBar

try:

    from QATCH.core.constants import Constants
    from QATCH.common.architecture import Architecture

except (ModuleNotFoundError, ImportError):

    class Constants:
        app_title = "QATCH nanovisQ Real-Time GUI"
        app_version = "v2.7b5_headless"
        app_date = "2026-05-28"

    class Architecture:
        @staticmethod
        def get_path():
            return os.getcwd()


###############################################################################
# QATCH Splash Screen
###############################################################################
class QatchSplashScreen(QSplashScreen):

    ###########################################################################
    # Initializing values for application
    ###########################################################################
    def __init__(self, argv=sys.argv):
        """Create a custom splash screen widget to show while the app loads."""

        # Initialize Splash Screen
        build_info = f" {Constants.app_title}\n Version: {Constants.app_version}\n Build Date: {Constants.app_date}\n"
        build_info = "\n\n\n\n" + "\n".join([f"            {s}" for s in build_info.split("\n")])
        icon_path = os.path.join(Architecture.get_path(), "QATCH\\icons\\qatch-splash.png")
        super().__init__(
            pixmap=QPixmap(icon_path),
            # flags=...,
        )

        # Add Progress Bar to Splash Screen
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(2, self.height() - 13, self.width() - 5, 10)
        self.progress_bar.setRange(0, 0)  # indeterminate mode
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                background-color: #333333;
                text-align: center;
            }

            QProgressBar::chunk {
                background-color: #3CB3E3;
                width: 20px;
            }
        """)

        # Style Splash Screen and Show
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.showMessage(
            build_info,
            Qt.AlignTop | Qt.AlignLeft,
            QColor(255, 255, 255, 255),  # QColor(60, 179, 227, 255),
        )
        self.setStyleSheet(
            "font-size: 12pt;"
            "font-family: system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;"
            "font-weight: bold;"
        )
        self.show()

    def mousePressEvent(self, event):
        """Triggered when the user clicks `QSplashScreen` widget. Default handler calls `self.hide`"""
        # We do not want super() handler to hide the widget, so ignore clicks:
        event.ignore()

    def closeEvent(self, event):
        """Triggered gracefully when the main app calls splash_process.terminate()"""
        # We want an instant clean close, just let it pass:
        event.accept()


def main():
    app = QApplication(sys.argv)
    splash = QatchSplashScreen()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
