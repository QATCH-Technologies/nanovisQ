"""QATCH.ui.interfaces.ui_review

This module provides the UIReview class, which builds the "Review" mode
placeholder shown in place of the eventual multi-dataset cross-analysis view.

Author(s):
    Paul MacNichol  (paul.macnichol@qatchtech.com)

Date:
    2026-07-14
"""

import os
from typing import TYPE_CHECKING

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css

if TYPE_CHECKING:
    from QATCH.ui.windows import ReviewWindow


class UIReview:
    """Sets up the user interface for the Review window.

    Review (multi-dataset cross analysis) is not yet implemented; this class
    builds a themed "coming soon" placeholder that stands in for that future
    view within the mode-switching UI.
    """

    def setup_ui(self, review_window: "ReviewWindow") -> None:
        """Initializes and structures the UI components of the window.

        Args:
            review_window (ReviewWindow): The window instance where the UI
                components will be rendered.
        """
        review_window.setObjectName("reviewWindow")
        review_window.setStyleSheet("")
        review_window.setTabShape(QtWidgets.QTabWidget.Rounded)
        review_window.setMinimumSize(QtCore.QSize(1000, 122))

        self.centralwidget = QtWidgets.QWidget(review_window)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)

        layout = QtWidgets.QVBoxLayout(self.centralwidget)

        self.review_placeholder_label = QtWidgets.QLabel("Coming soon...")
        self.review_placeholder_label.setObjectName("reviewPlaceholderLabel")
        self.review_placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.review_placeholder_label)

        self._update_placeholder_theme()
        ThemeManager.instance().themeChanged.connect(lambda _: self._update_placeholder_theme())

        review_window.setCentralWidget(self.centralwidget)

        self.retranslateUi(review_window)
        QtCore.QMetaObject.connectSlotsByName(review_window)

    def retranslateUi(self, review_window: QtWidgets.QMainWindow) -> None:  # noqa: N802
        """Translates the UI components for internationalization and sets window icons.

        Args:
            review_window (QtWidgets.QMainWindow): The window containing the elements
                to be translated.
        """
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "qatch-icon.png")
        review_window.setWindowIcon(QtGui.QIcon(icon_path))
        review_window.setWindowTitle(_translate("review_window", "Review"))

    def _update_placeholder_theme(self) -> None:
        """Re-tints the 'Coming soon...' label from the active theme."""
        tok = ThemeManager.instance().tokens()
        self.review_placeholder_label.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_text_muted'])}; font-size: 16px; "
            "font-style: italic; background: transparent; }"
        )
