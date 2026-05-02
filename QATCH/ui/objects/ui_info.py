import os
from PyQt5 import QtCore, QtGui, QtWidgets
from QATCH.common.architecture import Architecture


class UIInfo:

    def setupUi(self, MainWindow3):
        # MainWindow3.setObjectName("MainWindow3")
        # MainWindow3.setGeometry(500, 50, 100, 500)
        # MainWindow3.setFixedSize(100, 500)
        # MainWindow3.resize(100, 500)
        # MainWindow3.setMinimumSize(QtCore.QSize(0, 0))
        MainWindow3.setStyleSheet("")
        MainWindow3.setTabShape(QtWidgets.QTabWidget.Rounded)
        MainWindow3.setMinimumSize(QtCore.QSize(268, 518))
        MainWindow3.move(820, 0)
        self.centralwidget = QtWidgets.QWidget(MainWindow3)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")

        # Setup Information -------------------------------------------------------------------
        self.info1 = QtWidgets.QLabel()
        self.info1.setStyleSheet("background: #008EC0; padding: 1px;")
        self.info1.setText(
            "<font color=#ffffff > Setup Information&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</font>"
        )
        # self.info1.setFixedWidth(250)
        # self.info1.setFixedHeight(15)
        self.gridLayout_2.addWidget(self.info1, 0, 0, 1, 1)

        # Device Setup -------------------------------------------------------------------------
        self.info1a = QtWidgets.QLabel()
        self.info1a.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        self.info1a.setText("<font color=#0000ff > Device Setup</font>")
        # self.info1a.setFixedWidth(250)
        # self.info1a.setFixedHeight(22)
        self.gridLayout_2.addWidget(self.info1a, 1, 0, 1, 1)

        # Operation Mode -----------------------------------------------------------------------
        self.info11 = QtWidgets.QLabel()
        self.info11.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        self.info11.setText("<font color=#0000ff > Operation Mode </font>")
        # self.info11.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.info11, 2, 0, 1, 1)

        # Data Information ---------------------------------------------------------------------
        self.info = QtWidgets.QLabel()
        self.info.setStyleSheet("background: #008EC0; padding: 1px;")
        self.info.setText("<font color=#ffffff > Data Information&nbsp;</font>")
        # self.info.setFixedWidth(250)
        # self.info.setFixedHeight(15)
        self.gridLayout_2.addWidget(self.info, 3, 0, 1, 1)

        # Selected Frequency -------------------------------------------------------------------
        self.info2 = QtWidgets.QLabel()
        self.info2.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        self.info2.setText("<font color=#0000ff > Selected Frequency </font>")
        # self.info2.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.info2, 4, 0, 1, 1)

        # Frequency Value ----------------------------------------------------------------------
        self.info6 = QtWidgets.QLabel()
        self.info6.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        self.info6.setText("<font color=#0000ff > Frequency Value </font>")
        # self.info6.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.info6, 5, 0, 1, 1)

        # Start Frequency ----------------------------------------------------------------------
        self.info3 = QtWidgets.QLabel()
        self.info3.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        self.info3.setText("<font color=#0000ff > Start Frequency </font>")
        # self.info3.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.info3, 6, 0, 1, 1)

        # Stop Frequency -----------------------------------------------------------------------
        self.info4 = QtWidgets.QLabel()
        self.info4.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        self.info4.setText("<font color=#0000ff > Stop Frequency </font>")
        # self.info4.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.info4, 7, 0, 1, 1)

        # Frequency Range----------------------------------------------------------------------
        self.info4a = QtWidgets.QLabel()
        self.info4a.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        self.info4a.setText("<font color=#0000ff > Frequency Range </font>")
        # self.info4a.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.info4a, 8, 0, 1, 1)

        # Sample Rate----------------------------------------------------------------------
        self.info5 = QtWidgets.QLabel()
        self.info5.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        self.info5.setText("<font color=#0000ff > Sample Rate </font>")
        # self.info5.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.info5, 9, 0, 1, 1)

        # Sample Number----------------------------------------------------------------------
        self.info7 = QtWidgets.QLabel()
        self.info7.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        self.info7.setText("<font color=#0000ff > Sample Number </font>")
        # self.info7.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.info7, 10, 0, 1, 1)

        # Reference Settings -------------------------------------------------------------------
        self.inforef = QtWidgets.QLabel()
        self.inforef.setStyleSheet("background: #008EC0; padding: 1px;")
        # self.inforef1.setAlignment(QtCore.Qt.AlignCenter)
        self.inforef.setText("<font color=#ffffff > Reference Settings </font>")
        # self.inforef.setFixedHeight(15)
        # self.inforef.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.inforef, 11, 0, 1, 1)

        # Ref. Frequency -----------------------------------------------------------------------
        self.inforef1 = QtWidgets.QLabel()
        self.inforef1.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        # self.inforef1.setAlignment(QtCore.Qt.AlignCenter)
        self.inforef1.setText("<font color=#0000ff > Ref. Frequency </font>")
        # self.inforef1.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.inforef1, 12, 0, 1, 1)
        # Ref. Dissipation -----------------------------------------------------------------------

        self.inforef2 = QtWidgets.QLabel()
        self.inforef2.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        # self.inforef2.setAlignment(QtCore.Qt.AlignCenter)
        self.inforef2.setText("<font color=#0000ff > Ref. Dissipation </font>")
        # self.inforef2.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.inforef2, 13, 0, 1, 1)

        # Current Data ---------------------------------------------------------------------------
        self.l8 = QtWidgets.QLabel()
        self.l8.setStyleSheet("background: #008EC0; padding: 1px;")
        self.l8.setText("<font color=#ffffff > Current Data </font>")
        # self.l8.setFixedHeight(15)
        # self.l8.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.l8, 14, 0, 1, 1)

        # Resonance Frequency -------------------------------------------------------------------
        self.l7 = QtWidgets.QLabel()
        self.l7.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        self.l7.setText("<font color=#0000ff >  Resonance Frequency </font>")
        # self.l7.setAlignment(QtCore.Qt.AlignCenter)
        # self.l7.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.l7, 15, 0, 1, 1)

        # Dissipation ---------------------------------------------------------------------------
        self.l6 = QtWidgets.QLabel()
        self.l6.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        self.l6.setText("<font color=#0000ff > Dissipation  </font>")
        # self.l6.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.l6, 16, 0, 1, 1)

        # Temperature ---------------------------------------------------------------------------
        self.l6a = QtWidgets.QLabel()
        self.l6a.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        self.l6a.setText("<font color=#0000ff >  Temperature </font>")
        # self.l6a.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.l6a, 17, 0, 1, 1)

        # Info from QATCH website -------------------------------------------------------------
        self.lweb = QtWidgets.QLabel()
        self.lweb.setStyleSheet("background: #008EC0; padding: 1px;")
        self.lweb.setText("<font color=#ffffff > Check for Updates </font>")
        # self.lweb.setFixedHeight(15)
        # self.lweb.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.lweb, 18, 0, 1, 1)

        # Check internet connection -------------------------------------------------------------
        """self.lweb2 = QtWidgets.QLabel()
        self.lweb2.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.lweb2.setText(
            "<font color=#0000ff > Checking your internet connection </font>")
        # self.lweb2.setFixedHeight(20)
        #self.lweb2.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.lweb2, 19, 0, 1, 1)"""

        # Software update status ----------------------------------------------------------------
        self.lweb3 = QtWidgets.QLabel()
        self.lweb3.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        self.lweb3.setText("<font color=#0000ff > Update Status </font>")
        # self.lweb3.setFixedHeight(16)
        # self.lweb3.setFixedWidth(300)
        self.gridLayout_2.addWidget(self.lweb3, 20, 0, 1, 1)

        # Download button -----------------------------------------------------------------------
        self.pButton_Download = QtWidgets.QPushButton(self.centralwidget)
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/refresh-icon.png")
        self.pButton_Download.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Download.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Download.setObjectName("pButton_Download")
        self.pButton_Download.setFixedWidth(145)
        self.gridLayout_2.addWidget(self.pButton_Download, 21, 0, 1, 1, QtCore.Qt.AlignRight)
        ##########################################################################################

        self.gridLayout.addLayout(self.gridLayout_2, 3, 1, 1, 1)
        MainWindow3.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow3)
        QtCore.QMetaObject.connectSlotsByName(MainWindow3)

    def retranslateUi(self, MainWindow3):
        _translate = QtCore.QCoreApplication.translate
        self.pButton_Download.setText(_translate("MainWindow3", " Check Again"))
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/qatch-icon.png")
        MainWindow3.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        MainWindow3.setWindowTitle(_translate("MainWindow3", "Information"))
