"""QATCH.ui.interfaces.info.py

This module provides the UIInfo class which is responsible for setting up
the information panel layout, styling, and basic widgets using PyQt5.

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-07-01
"""

import os
from typing import TYPE_CHECKING

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture

if TYPE_CHECKING:
    from QATCH.ui.windows import InfoWindow


class UIInfo:
    """Sets up the user interface for the Information window.

    This class configures a grid layout containing various informative labels
    grouped by categories such as Setup, Data, Reference Settings, Current Data,
    and Update Status.
    """

    def setup_ui(self, info_window: "InfoWindow") -> None:
        """Initializes and structures the UI components of the window.

        Args:
            info_window (InfoWindow): The main window instance
                where the UI components will be rendered.
        """
        info_window.setStyleSheet("")
        info_window.setTabShape(QtWidgets.QTabWidget.Rounded)
        info_window.setMinimumSize(QtCore.QSize(268, 518))
        info_window.move(820, 0)

        self.centralwidget = QtWidgets.QWidget(info_window)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)

        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")

        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")

        # Setup Information
        self.info1 = self._create_header_label(" Setup Information")
        self.gridLayout_2.addWidget(self.info1, 0, 0, 1, 1)

        self.info1a = self._create_info_label(" Device Setup")
        self.gridLayout_2.addWidget(self.info1a, 1, 0, 1, 1)

        self.info11 = self._create_info_label(" Operation Mode ")
        self.gridLayout_2.addWidget(self.info11, 2, 0, 1, 1)

        # Data Information
        self.info = self._create_header_label(" Data Information ")
        self.gridLayout_2.addWidget(self.info, 3, 0, 1, 1)

        self.info2 = self._create_info_label(" Selected Frequency ")
        self.gridLayout_2.addWidget(self.info2, 4, 0, 1, 1)

        self.info6 = self._create_info_label(" Frequency Value ")
        self.gridLayout_2.addWidget(self.info6, 5, 0, 1, 1)

        self.info3 = self._create_info_label(" Start Frequency ")
        self.gridLayout_2.addWidget(self.info3, 6, 0, 1, 1)

        self.info4 = self._create_info_label(" Stop Frequency ")
        self.gridLayout_2.addWidget(self.info4, 7, 0, 1, 1)

        self.info4a = self._create_info_label(" Frequency Range ")
        self.gridLayout_2.addWidget(self.info4a, 8, 0, 1, 1)

        self.info5 = self._create_info_label(" Sample Rate ")
        self.gridLayout_2.addWidget(self.info5, 9, 0, 1, 1)

        self.info7 = self._create_info_label(" Sample Number ")
        self.gridLayout_2.addWidget(self.info7, 10, 0, 1, 1)

        # Reference Settings
        self.inforef = self._create_header_label(" Reference Settings ")
        self.gridLayout_2.addWidget(self.inforef, 11, 0, 1, 1)

        self.inforef1 = self._create_info_label(" Ref. Frequency ")
        self.gridLayout_2.addWidget(self.inforef1, 12, 0, 1, 1)

        self.inforef2 = self._create_info_label(" Ref. Dissipation ")
        self.gridLayout_2.addWidget(self.inforef2, 13, 0, 1, 1)

        # Current Data
        self.l8 = self._create_header_label(" Current Data ")
        self.gridLayout_2.addWidget(self.l8, 14, 0, 1, 1)

        self.l7 = self._create_info_label(" Resonance Frequency ")
        self.gridLayout_2.addWidget(self.l7, 15, 0, 1, 1)

        self.l6 = self._create_info_label(" Dissipation ")
        self.gridLayout_2.addWidget(self.l6, 16, 0, 1, 1)

        self.l6a = self._create_info_label(" Temperature ")
        self.gridLayout_2.addWidget(self.l6a, 17, 0, 1, 1)

        # Update status section — widgets are kept as hidden stubs so that
        # existing code in main_window.py can still call setText() on them
        # without errors, but the controls bar icon now owns the visible display.
        self.lweb = self._create_header_label(" Check for Updates ")
        self.lweb.hide()

        self.lweb3 = self._create_info_label(" Update Status ")
        self.lweb3.hide()

        self.pButton_Download = QtWidgets.QPushButton(self.centralwidget)
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "refresh-circle.svg")
        self.pButton_Download.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Download.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Download.setObjectName("pButton_Download")
        self.pButton_Download.setFixedWidth(145)
        self.pButton_Download.hide()

        # Finalize Layout
        self.gridLayout.addLayout(self.gridLayout_2, 3, 1, 1, 1)
        info_window.setCentralWidget(self.centralwidget)

        self.retranslateUi(info_window)
        QtCore.QMetaObject.connectSlotsByName(info_window)

    def retranslateUi(self, info_window: QtWidgets.QMainWindow) -> None:  # noqa: N802
        """Translates the UI components for internationalization and sets window icons.

        Args:
            info_window (QtWidgets.QMainWindow): The main window containing the elements
                to be translated.
        """
        _translate = QtCore.QCoreApplication.translate
        self.pButton_Download.setText(_translate("info_window", " Check Again"))

        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "qatch-icon.png")
        info_window.setWindowIcon(QtGui.QIcon(icon_path))
        info_window.setWindowTitle(_translate("info_window", "Information"))

    def _create_header_label(self, text: str) -> QtWidgets.QLabel:
        """Creates a styled header label.

        Args:
            text (str): The text to display in the header label.

        Returns:
            QtWidgets.QLabel: A styled QLabel instance for headers.
        """
        label = QtWidgets.QLabel()
        label.setStyleSheet("background: #008EC0; padding: 1px; color: #ffffff;")
        label.setText(text)
        return label

    def _create_info_label(self, text: str) -> QtWidgets.QLabel:
        """Creates a styled standard information label.

        Args:
            text (str): The text to display in the info label.

        Returns:
            QtWidgets.QLabel: A styled QLabel instance for general information rows.
        """
        label = QtWidgets.QLabel()
        label.setStyleSheet(
            "background: white; padding: 1px; border: 1px solid #cccccc; color: #0000ff;"
        )
        label.setText(text)
        return label
