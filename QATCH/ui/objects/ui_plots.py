import os

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QDesktopWidget,
)
from pyqtgraph import GraphicsLayoutWidget
from QATCH.common.architecture import Architecture
from QATCH.core.constants import Constants

#######################################################################################################################


class UIPlots:

    def setupUi(self, MainWindow2):
        USE_FULLSCREEN = QDesktopWidget().availableGeometry().width() == 2880

        MainWindow2.setObjectName("MainWindow2")
        # MainWindow2.setGeometry(100, 100, 890, 750)
        # MainWindow2.setFixedSize(1091, 770)
        # MainWindow2.resize(1091, 770)
        MainWindow2.setMinimumSize(QtCore.QSize(1000, 250))
        if USE_FULLSCREEN:
            MainWindow2.resize(1701, 1435)
            MainWindow2.move(0, 0)
        else:
            MainWindow2.move(692, 0)
        MainWindow2.setStyleSheet("")
        MainWindow2.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow2)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        # Remove top margin, leaving the rest as "default"
        self.gridLayout.setContentsMargins(11, 0, 11, 11)
        self.Layout_graphs = QtWidgets.QSplitter(QtCore.Qt.Horizontal)  # QGridLayout()
        self.Layout_graphs.setObjectName("Layout_graphs")

        self.plt = GraphicsLayoutWidget(self.centralwidget)
        self.pltB = GraphicsLayoutWidget(self.centralwidget)

        self.plt.setAutoFillBackground(False)
        self.plt.setStyleSheet("border: 0px;")
        self.plt.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.plt.setFrameShadow(QtWidgets.QFrame.Plain)
        self.plt.setLineWidth(0)
        self.plt.setObjectName("plt")
        self.plt.setMinimumWidth(333)

        self.pltB.setAutoFillBackground(False)
        self.pltB.setStyleSheet("border: 0px;")
        self.pltB.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.pltB.setFrameShadow(QtWidgets.QFrame.Plain)
        self.pltB.setLineWidth(0)
        self.pltB.setObjectName("pltB")
        self.pltB.setMinimumWidth(666)

        """
        self.label = QtWidgets.QLabel()
        self.Layout_graphs.addWidget(self.label, 0, 0, 1, 1)
        def link1(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))
        self.label.linkActivated.connect(link1)
        self.label.setText(
            '<a href="https://openqcm.com/"> <font color=#333333 >Open-source Python application for displaying, processing and storing real-time data from openQCM Q-1 Device</font> </a>')
        """

        self.Layout_graphs.addWidget(self.pltB)
        self.Layout_graphs.addWidget(self.plt)

        # add collapse/expand icon arrows
        self.Layout_graphs.setHandleWidth(10)
        handle = self.Layout_graphs.handle(1)
        layout_s = QtWidgets.QVBoxLayout()
        layout_s.setContentsMargins(0, 0, 0, 0)
        layout_s.addStretch()
        self.btnCollapse = QtWidgets.QToolButton(handle)
        self.btnCollapse.setArrowType(QtCore.Qt.RightArrow)
        self.btnCollapse.clicked.connect(lambda: self.handleSplitterButton(True))
        layout_s.addWidget(self.btnCollapse)
        self.btnExpand = QtWidgets.QToolButton(handle)
        self.btnExpand.setArrowType(QtCore.Qt.LeftArrow)
        self.btnExpand.clicked.connect(lambda: self.handleSplitterButton(False))
        layout_s.addWidget(self.btnExpand)
        layout_s.addStretch()
        handle.setLayout(layout_s)
        self.btnExpand.setVisible(False)
        # self.handleSplitterButton(False)
        self.Layout_graphs.splitterMoved.connect(self.handleSplitterMoved)

        self.gridLayout.addWidget(self.Layout_graphs, 2, 1, 1, 1)
        MainWindow2.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow2)

    def handleSplitterMoved(self, pos, index):
        collapsed = self.Layout_graphs.sizes()[-1] == 0
        self.btnCollapse.setVisible(not collapsed)
        self.btnExpand.setVisible(collapsed)

    def handleSplitterButton(self, collapse=True):
        if collapse:
            self.btnCollapse.setVisible(False)
            self.btnExpand.setVisible(True)
            self.Layout_graphs.setSizes([1, 0])
        else:
            self.btnCollapse.setVisible(True)
            self.btnExpand.setVisible(False)
            width = self.Layout_graphs.width()
            self.Layout_graphs.setSizes([int(width * 0.65), int(width * 0.35)])

    def retranslateUi(self, MainWindow2):
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/qatch-icon.png")
        MainWindow2.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        MainWindow2.setWindowTitle(
            _translate(
                "MainWindow2",
                "{} {} - Plots".format(Constants.app_title, Constants.app_version),
            )
        )

    def Ui(self, MainWindow2):
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/qatch-icon.png")
        MainWindow2.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        MainWindow2.setWindowTitle(
            _translate(
                "MainWindow2",
                "{} {} - Plots".format(Constants.app_title, Constants.app_version),
            )
        )
