"""
PlaceholderWidget - Empty state indicator for the prediction UI

This widget displays when there's no data to show, providing
visual feedback and instructions to the user.
"""

import os

from architecture import Architecture
from PyQt5 import QtCore, QtGui, QtWidgets


class PlaceholderWidget(QtWidgets.QWidget):
    """
    A centered placeholder/empty state widget with an icon and message.

    Styling is handled by theme.qss via #placeholderWidget, #placeholderIcon,
    and #placeholderText selectors.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set object name for QSS styling
        self.setObjectName("placeholderWidget")

        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # Icon
        self.lbl_icon = QtWidgets.QLabel()
        self.lbl_icon.setObjectName("placeholderIcon")
        self.lbl_icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Load and display icon
        pixmap = QtGui.QPixmap(
            os.path.join(Architecture.get_path(), "icons/info-circle-svgrepo-com.svg")
        )

        if not pixmap.isNull():
            # Apply cyan tint to match theme
            self._apply_icon_tint(pixmap)
            self.lbl_icon.setPixmap(
                pixmap.scaled(
                    64,  # Slightly larger for better visibility
                    64,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            )

        layout.addWidget(self.lbl_icon)

        # Message text
        self.lbl_text = QtWidgets.QLabel(
            "No predictions yet.\nClick the + button to add a new prediction."
        )
        self.lbl_text.setObjectName("placeholderText")
        self.lbl_text.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_text.setWordWrap(True)

        layout.addWidget(self.lbl_text)

    def _apply_icon_tint(self, pixmap):
        """
        Apply a subtle cyan tint to the icon to match the theme.
        This creates a colored overlay on the SVG.
        """
        painter = QtGui.QPainter(pixmap)
        painter.setCompositionMode(
            QtGui.QPainter.CompositionMode.CompositionMode_SourceIn
        )
        painter.fillRect(pixmap.rect(), QtGui.QColor("#4EC4EB"))  # Cyan medium
        painter.end()

    def set_message(self, message):
        """
        Update the placeholder message.

        Args:
            message: New text to display
        """
        self.lbl_text.setText(message)

    def set_icon(self, icon_path, size=64):
        """
        Update the placeholder icon.

        Args:
            icon_path: Path to the icon file
            size: Icon size in pixels (default: 64)
        """
        pixmap = QtGui.QPixmap(icon_path)
        if not pixmap.isNull():
            self._apply_icon_tint(pixmap)
            self.lbl_icon.setPixmap(
                pixmap.scaled(
                    size,
                    size,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            )
