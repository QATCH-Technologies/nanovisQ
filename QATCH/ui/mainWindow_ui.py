"""
QATCH.ui.mainWindow_ui

The mainWindow_ui module handles the drawing, UI control elements,
and UI actions for the main window of the main application.

Author(s)
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)
    Others...

Date:
    2026-01-26
"""

import logging
import os
from time import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
    QVariantAnimation,
    pyqtSignal,
)
from PyQt5.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QDesktopWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph import GraphicsLayoutWidget

from QATCH.common.architecture import Architecture, OSType
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants, OperationType
from QATCH.processors.Device import serial  # real device hardware
from QATCH.ui.drawPlateConfig import WellPlate
from QATCH.ui.popUp import PopUp


class Ui_Controls(object):  # QtWidgets.QMainWindow
    def setupUi(self, MainWindow1):
        USE_FULLSCREEN = QDesktopWidget().availableGeometry().width() == 2880
        SHOW_SIMPLE_CONTROLS = True
        self.cal_initialized = False
        self.parent = MainWindow1

        MainWindow1.setObjectName("MainWindow1")
        # MainWindow1.setGeometry(50, 50, 975, 70)
        # MainWindow1.setFixedSize(980, 150)
        # MainWindow1.resize(550, 50)
        MainWindow1.setMinimumSize(QtCore.QSize(1000, 50))
        if Architecture.get_os() is OSType.macosx:
            MainWindow1.resize(1080, 188)
        elif USE_FULLSCREEN:
            MainWindow1.resize(2880, 390)
            MainWindow1.move(0, 1485)
        else:
            MainWindow1.resize(1503, 175)
            MainWindow1.move(7, 567)
        MainWindow1.setStyleSheet("")
        MainWindow1.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow1)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.Layout_controls = QtWidgets.QGridLayout()
        self.Layout_controls.setObjectName("Layout_controls")

        # frequency/quartz combobox -------------------------------------------
        self.cBox_Speed = QtWidgets.QComboBox()
        self.cBox_Speed.setEditable(False)
        self.cBox_Speed.setObjectName("cBox_Speed")
        if USE_FULLSCREEN:
            self.cBox_Speed.setFixedHeight(50)
        self.Layout_controls.addWidget(self.cBox_Speed, 4, 1, 1, 1)

        # stop button ---------------------------------------------------------
        self.pButton_Stop = QtWidgets.QPushButton()
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/stop_icon.ico")
        self.pButton_Stop.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))  # .png
        self.pButton_Stop.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Stop.setObjectName("pButton_Stop")
        if USE_FULLSCREEN:
            self.pButton_Stop.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pButton_Stop, 3, 6, 1, 1)

        # COM port combobox ---------------------------------------------------
        self.cBox_Port = QtWidgets.QComboBox()
        self.cBox_Port.setEditable(False)
        self.cBox_Port.setObjectName("cBox_Port")
        if USE_FULLSCREEN:
            self.cBox_Port.setFixedHeight(50)
        self.Layout_controls.addWidget(self.cBox_Port, 2, 1, 1, 1)

        # Identify button ---------------------------------------------------------
        self.pButton_ID = QtWidgets.QPushButton()
        self.pButton_ID.setToolTip("Identify selected Serial COM Port")
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/identify-icon.png")
        self.pButton_ID.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))  # .png
        self.pButton_ID.setStyleSheet("background:white;padding:3px;")
        if USE_FULLSCREEN:
            self.pButton_ID.setMinimumSize(QtCore.QSize(60, 50))
        else:
            self.pButton_ID.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_ID.setObjectName("pButton_ID")
        self.Layout_controls.addWidget(self.pButton_ID, 2, 2, 1, 1)

        # Refresh button ---------------------------------------------------------
        self.pButton_Refresh = QtWidgets.QPushButton()
        self.pButton_Refresh.setToolTip("Refresh Serial COM Port list")
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/refresh-icon.png")
        self.pButton_Refresh.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))  # .png
        self.pButton_Refresh.setStyleSheet("background:white;padding:3px;margin-right:9px;")
        if USE_FULLSCREEN:
            self.pButton_Refresh.setMinimumSize(QtCore.QSize(70, 50))
        else:
            self.pButton_Refresh.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Refresh.setObjectName("pButton_Refresh")
        self.Layout_controls.addWidget(self.pButton_Refresh, 2, 3, 1, 1)

        # Operation mode - source ---------------------------------------------
        self.cBox_Source = QtWidgets.QComboBox()
        self.cBox_Source.setObjectName("cBox_Source")
        if USE_FULLSCREEN:
            self.cBox_Source.setFixedHeight(50)
        self.Layout_controls.addWidget(self.cBox_Source, 2, 0, 1, 1)

        # Frequency hopping checkbox ------------------------------------------
        self.chBox_freqHop = QtWidgets.QCheckBox()
        self.chBox_freqHop.setEnabled(True)
        self.chBox_freqHop.setChecked(False)
        self.chBox_freqHop.setObjectName("chBox_freqHop")
        self.Layout_controls.addWidget(self.chBox_freqHop, 4, 2, 1, 2)

        # Noise correction checkbox -------------------------------------------
        self.chBox_correctNoise = QtWidgets.QCheckBox()
        self.chBox_correctNoise.setEnabled(True)
        self.chBox_correctNoise.setChecked(True)
        self.chBox_correctNoise.setObjectName("chBox_correctNoise")
        # self.chBox_correctNoise.setVisible(False)
        self.Layout_controls.addWidget(self.chBox_correctNoise, 5, 1, 1, 3)

        # Cartridge Auto-Lock -------------------------------------------------
        self.l9 = QtWidgets.QLabel()
        self.l9.setStyleSheet("background: #008EC0; padding: 1px;")
        self.l9.setText("<font color=#ffffff > Cartridge Auto-Lock </font>")
        if USE_FULLSCREEN:
            self.l9.setFixedHeight(50)
        # else:
        #    self.l9.setFixedHeight(15)
        self.Layout_controls.addWidget(self.l9, 1, 4, 1, 1)

        # Cartridge Controls --------------------------------------------------
        self.rButton_Automatic = QtWidgets.QRadioButton("Automatic")
        self.rButton_Automatic.setToolTip("""
            <b><u>Automatic:</u></b><br/>
            - Locks before init/run<br/>
            - Useful if/when user forgets
            """)
        self.rButton_Automatic.setChecked(True)  # default
        self.rButton_Manual = QtWidgets.QRadioButton("Manual")
        self.rButton_Manual.setToolTip("""
            <b><u>Manual:</u></b><br/>
            - You control lock position<br/>
            - Must lock before init/run
            """)
        self.rCartridgeMode = QtWidgets.QButtonGroup()
        self.rCartridgeMode.addButton(self.rButton_Automatic, 1)
        self.rCartridgeMode.addButton(self.rButton_Manual, 0)
        self.layMode = QtWidgets.QVBoxLayout()
        self.layMode.addWidget(self.rButton_Automatic)
        self.layMode.addWidget(self.rButton_Manual)
        self.grpMode = QtWidgets.QGroupBox("Auto-Lock Mode:")
        self.grpMode.setLayout(self.layMode)
        self.Layout_controls.addWidget(self.grpMode, 2, 4, 3, 1)

        # start button --------------------------------------------------------
        self.pButton_Start = QtWidgets.QPushButton()
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/start_icon.ico")
        self.pButton_Start.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Start.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Start.setObjectName("pButton_Start")
        if USE_FULLSCREEN:
            self.pButton_Start.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pButton_Start, 2, 6, 1, 1)

        # Add signal for Run Controls UI to handle START from Advanced menu ---
        self.pButton_Start.clicked.connect(
            lambda: (
                self.run_controls.set_running(True)
                if (
                    (OperationType(self.cBox_Source.currentIndex()) == OperationType.measurement)
                    and hasattr(self, "run_controls")
                )
                else None
            )
        )

        # clear plots button --------------------------------------------------
        self.pButton_Clear = QtWidgets.QPushButton()
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/clear_icon.ico")
        self.pButton_Clear.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Clear.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Clear.setObjectName("pButton_Clear")
        if USE_FULLSCREEN:
            self.pButton_Clear.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pButton_Clear, 2, 5, 1, 1)

        # reference button ----------------------------------------------------
        self.pButton_Reference = QtWidgets.QPushButton()
        # self.pButton_Reference.setIcon(QtGui.QIcon(QtGui.QPixmap("ref_icon.ico")))
        self.pButton_Reference.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Reference.setObjectName("pButton_Reference")
        self.pButton_Reference.setCheckable(True)
        if USE_FULLSCREEN:
            self.pButton_Reference.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pButton_Reference, 3, 5, 1, 1)

        # restore factory defaults --------------------------------------------
        self.pButton_ResetApp = QtWidgets.QPushButton()
        self.pButton_ResetApp.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_ResetApp.setObjectName("pButton_ResetApp")
        if USE_FULLSCREEN:
            self.pButton_ResetApp.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pButton_ResetApp, 4, 5, 1, 1)

        # samples SpinBox -----------------------------------------------------
        self.sBox_Samples = QtWidgets.QSpinBox()
        self.sBox_Samples.setMinimum(1)
        self.sBox_Samples.setMaximum(100000)
        self.sBox_Samples.setProperty("value", 500)
        self.sBox_Samples.setObjectName("sBox_Samples")
        # self.sBox_Samples.setEnabled(False)
        self.sBox_Samples.setVisible(False)
        self.Layout_controls.addWidget(self.sBox_Samples, 2, 4, 1, 1)

        # export file CheckBox ------------------------------------------------
        self.chBox_export = QtWidgets.QCheckBox()
        self.chBox_export.setEnabled(True)
        self.chBox_export.setObjectName("chBox_export")
        self.chBox_export.setVisible(False)
        self.Layout_controls.addWidget(self.chBox_export, 4, 4, 1, 1)

        # temperature Control slider ------------------------------------------
        self.slTemp = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slTemp.setMinimum(8)
        self.slTemp.setMaximum(40)
        self.slTemp.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slTemp.setTickInterval(1)
        self.slTemp.setSingleStep(1)
        self.slTemp.setPageStep(5)
        self.Layout_controls.addWidget(self.slTemp, 3, 4, 1, 1)

        # temperature Control label -------------------------------------------
        self.lTemp = QtWidgets.QLabel()
        # self.lTemp.setStyleSheet('background: #008EC0; padding: 1px;')
        self.lTemp.setText("PV:--.--C SP:--.--C OP:----")
        self.lTemp.setAlignment(QtCore.Qt.AlignCenter)
        self.lTemp.setFont(QtGui.QFont("Consolas", -1))
        # self.lTemp.setFixedHeight(15)
        self.Layout_controls.addWidget(self.lTemp, 2, 4, 1, 1)

        # temperature Control label -------------------------------------------
        self.pTemp = QtWidgets.QPushButton()
        # self.pTemp.setStyleSheet('background: #008EC0; padding: 1px;')
        self.pTemp.setText("Start Temp Control")
        # self.pTemp.setAlignment(QtCore.Qt.AlignCenter)
        if USE_FULLSCREEN:
            self.pTemp.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pTemp, 4, 4, 1, 1)

        # Samples Number / History Buffer Size---------------------------------
        # self.l5 = QtWidgets.QLabel()
        # self.l5.setStyleSheet('background: #008EC0; padding: 1px;')
        # self.l5.setText("<font color=#ffffff > Samples Number / History Buffer Size </font>")
        # self.l5.setFixedHeight(15)
        # self.Layout_controls.addWidget(self.l5, 1, 4, 1, 1)

        # Control Buttons------------------------------------------------------
        self.l = QtWidgets.QLabel()
        self.l.setStyleSheet("background: #008EC0; padding: 1px;")
        self.l.setText("<font color=#ffffff > Control Buttons </font>")
        if USE_FULLSCREEN:
            self.l.setFixedHeight(50)
        # else:
        #    self.l.setFixedHeight(15)
        self.Layout_controls.addWidget(self.l, 1, 5, 1, 2)

        # Operation Mode ------------------------------------------------------
        self.l0 = QtWidgets.QLabel()
        self.l0.setStyleSheet("background: #008EC0; padding: 1px;")
        self.l0.setText("<font color=#ffffff >Operation Mode</font> </a>")
        if USE_FULLSCREEN:
            self.l0.setFixedHeight(50)
        # else:
        #    self.l0.setFixedHeight(15)
        self.Layout_controls.addWidget(self.l0, 1, 0, 1, 1)

        # Resonance Frequency / Quartz Sensor ---------------------------------
        self.l2 = QtWidgets.QLabel()
        self.l2.setStyleSheet("background: #008EC0; padding: 1px;")
        self.l2.setText("<font color=#ffffff > Resonance Frequency / Quartz Sensor </font>")
        if USE_FULLSCREEN:
            self.l2.setFixedHeight(50)
        # else:
        #    self.l2.setFixedHeight(15)
        self.Layout_controls.addWidget(self.l2, 3, 1, 1, 3)

        # Serial COM Port -----------------------------------------------------
        self.l1 = QtWidgets.QLabel()
        self.l1.setStyleSheet("background: #008EC0; padding: 1px;")
        self.l1.setText("<font color=#ffffff > Serial COM Port </font>")
        if USE_FULLSCREEN:
            self.l1.setFixedHeight(50)
        # else:
        #    self.l1.setFixedHeight(15)
        self.Layout_controls.addWidget(self.l1, 1, 1, 1, 3)

        # logo---------------------------------------------------------
        self.l3 = QtWidgets.QLabel()
        self.l3.setAlignment(QtCore.Qt.AlignRight)
        self.Layout_controls.addWidget(self.l3, 4, 7, 1, 1)
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/qatch-logo_full.jpg")
        if USE_FULLSCREEN:
            pixmap = QtGui.QPixmap(icon_path)
            pixmap = pixmap.scaled(250, 50, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            self.l3.setPixmap(pixmap)
        else:
            self.l3.setPixmap(QtGui.QPixmap(icon_path))

        # qatch link --------------------------------------------------------
        self.l4 = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.l4, 3, 7, 1, 1)

        def link(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))

        self.l4.linkActivated.connect(link)
        self.l4.setAlignment(QtCore.Qt.AlignRight)
        self.l4.setText(
            '<a href="https://qatchtech.com/"> <font size=4 color=#008EC0 >qatchtech.com</font>'
        )  # &nbsp;

        # info@qatchtech.com Mail -----------------------------------------------
        self.lmail = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.lmail, 2, 7, 1, 1)  # 25 40

        def linkmail(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))

        self.lmail.linkActivated.connect(linkmail)
        self.lmail.setAlignment(QtCore.Qt.AlignRight)
        # self.lmail.setAlignment(QtCore.Qt.AlignLeft)
        self.lmail.setText(
            '<a href="mailto:info@qatchtech.com"> <font color=#008EC0 >info@qatchtech.com</font>'
        )

        # software user guide --------------------------------------------------------
        self.lg = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.lg, 1, 7, 1, 1)  # 30

        def link(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))

        self.lg.linkActivated.connect(link)
        self.lg.setAlignment(QtCore.Qt.AlignRight)
        self.lg.setText(
            '<a href="file://{}/docs/userguide.pdf"> <font color=#008EC0 >User Guide</font>'.format(
                Architecture.get_path()
            )
        )  # &nbsp;
        #####################################
        """
        self.ico = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.ico, 2, 5, 1, 1)
        self.title = QtWidgets.QLabel()
        def link(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))
        self.title.linkActivated.connect(link)
        self.title.setText(
            '<a href="https://openqcm.com/openqcm-q-1-software"> <font color=#008EC0 >user guide</font>')
        self.Layout_controls.addWidget(self.title, 2, 5, 1, 1)
        self.pixmap = QtGui.QPixmap("guide.ico")
        self.ico.setPixmap(self.pixmap)
        self.ico.setAlignment(QtCore.Qt.AlignRight)
        self.title.setMinimumHeight(self.pixmap.height())
        self.title.setAlignment(QtCore.Qt.AlignRight)
        """
        #####################################
        # Save file -----------------------------------------------------------
        self.infosave = QtWidgets.QLabel()
        self.infosave.setStyleSheet("background: #008EC0; padding: 1px;")
        # self.infosave.setAlignment(QtCore.Qt.AlignCenter)
        if USE_FULLSCREEN:
            self.infosave.setFixedHeight(50)
        # else:
        #    self.infosave.setFixedHeight(15)
        self.infosave.setText("<font color=#ffffff > TEC Temperature Control </font>")
        self.Layout_controls.addWidget(self.infosave, 1, 4, 1, 1)

        # Program Status standby ----------------------------------------------
        self.infostatus = QtWidgets.QLabel()
        self.infostatus.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        self.infostatus.setAlignment(QtCore.Qt.AlignCenter)
        self.infostatus.setText("<font color=#333333 > Program Status Standby </font>")
        if USE_FULLSCREEN:
            self.infostatus.setFixedHeight(50)
        self.Layout_controls.addWidget(self.infostatus, 5, 5, 1, 2)

        # Infobar -------------------------------------------------------------
        self.infobar = QtWidgets.QLineEdit()
        self.infobar.setReadOnly(True)
        self.infobar_label = QtWidgets.QLabel()
        self.infobar_label.setStyleSheet(
            "background: white; padding: 1px; border: 1px solid #cccccc"
        )
        # self.infobar_label.setAlignment(QtCore.Qt.AlignCenter)
        self.infobar.textChanged.connect(self.infobar_label.setText)
        if SHOW_SIMPLE_CONTROLS:
            self.infobar.textChanged.connect(self._update_progress_text)
        if USE_FULLSCREEN:
            self.infobar_label.setFixedHeight(50)
        # self.infobar.setText("<font color=#0000ff > Infobar </font>") # WAIT until progressBar exists to trigger signals
        self.Layout_controls.addWidget(self.infobar_label, 0, 0, 1, 7)

        # Multiplex -----------------------------------------------------------
        self.lmp = QtWidgets.QLabel()
        self.lmp.setStyleSheet("background: #008EC0; padding: 1px;")
        self.lmp.setText("<font color=#ffffff > Multiplex Mode </font>")
        if USE_FULLSCREEN:
            self.lmp.setFixedHeight(50)
        # else:
        #    self.lmp.setFixedHeight(15)
        self.Layout_controls.addWidget(self.lmp, 3, 0, 1, 1)

        self.cBox_MultiMode = QtWidgets.QComboBox()
        self.cBox_MultiMode.setObjectName("cBox_MultiMode")
        self.cBox_MultiMode.addItems(["1 Channel", "2 Channels", "3 Channels", "4 Channels"])
        self.cBox_MultiMode.setCurrentIndex(0)  # default 1
        if USE_FULLSCREEN:
            self.cBox_MultiMode.setFixedHeight(50)

        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/")
        self.pButton_PlateConfig = QtWidgets.QPushButton(
            QtGui.QIcon(os.path.join(icon_path, "advanced.png")), ""
        )
        self.pButton_PlateConfig.setToolTip("Plate Configuration...")
        self.pButton_PlateConfig.clicked.connect(self.doPlateConfig)
        self.hBox_MultiConfig = QtWidgets.QHBoxLayout()
        self.hBox_MultiConfig.addWidget(self.cBox_MultiMode, 3)
        self.hBox_MultiConfig.addWidget(self.pButton_PlateConfig, 1)
        self.Layout_controls.addLayout(self.hBox_MultiConfig, 4, 0, 1, 1)

        self.chBox_MultiAuto = QtWidgets.QCheckBox()
        self.chBox_MultiAuto.setEnabled(True)
        self.chBox_MultiAuto.setChecked(True)
        self.chBox_MultiAuto.setObjectName("chBox_MultiAuto")
        self.Layout_controls.addWidget(self.chBox_MultiAuto, 5, 0, 1, 1)

        # Progressbar -------------------------------------------------------------
        styleBar = """
                    QProgressBar
                    {
                     border: 0.5px solid #B8B8B8;
                     border-radius: 1px;
                     text-align: center;
                     color: #333333;
                    }
                     QProgressBar::chunk
                    {
                     background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(184, 184, 184, 200), stop:1 rgba(221, 221, 221, 200));
                    }
                 """  # background:url("openQCM/icons/openqcm-logo.png")
        self.run_progress_bar = QtWidgets.QProgressBar()
        self.run_progress_bar.setGeometry(QtCore.QRect(0, 0, 50, 10))
        self.run_progress_bar.setObjectName("progressBar")
        self.run_progress_bar.setStyleSheet(styleBar)

        # self.fill_status_progress_bar = QtWidgets.QProgressBar()
        # self.fill_status_progress_bar.setMinimum(0)
        # self.fill_status_progress_bar.setMaximum(4)
        # self.fill_status_progress_bar.setGeometry(QtCore.QRect(0, 0, 50, 10))
        # self.fill_status_progress_bar.setObjectName("fillProgressBar")
        # self.fill_status_progress_bar.setStyleSheet(styleBar)

        if USE_FULLSCREEN:
            self.run_progress_bar.setFixedHeight(50)
            # self.fill_status_progress_bar.setFixedHeight(50)
        if SHOW_SIMPLE_CONTROLS:
            self.run_progress_bar.valueChanged.connect(self._update_progress_value)

        self.run_progress_bar.setValue(0)
        self.run_progress_bar.setHidden(True)
        # self.fill_status_progress_bar.setValue(0)

        # self.fill_status_progress_bar.setFormat("Run: %v/%m (No Fill)")

        self.Layout_controls.setColumnStretch(0, 0)
        self.Layout_controls.setColumnStretch(1, 1)
        self.Layout_controls.setColumnStretch(2, 0)
        self.Layout_controls.setColumnStretch(3, 0)
        self.Layout_controls.setColumnStretch(4, 2)
        self.Layout_controls.setColumnStretch(5, 2)
        self.Layout_controls.setColumnStretch(6, 2)
        self.Layout_controls.addWidget(self.run_progress_bar, 0, 7, 1, 1)
        self.gridLayout.addLayout(self.Layout_controls, 7, 1, 1, 1)
        # ---------------------------------------------------------------------

        # define simple layout - only add to central widget if requested
        self.toolLayout = QtWidgets.QVBoxLayout()
        self.toolBar = QtWidgets.QHBoxLayout()

        self.tool_bar = QtWidgets.QToolBar()
        self.tool_bar.setIconSize(QtCore.QSize(50, 30))
        self.tool_bar.setStyleSheet("color: #333333;")

        self.tool_NextPortRow = NumberIconButton()
        self.tool_NextPortRow.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_NextPortRow.setText("Next Port")
        self.tool_NextPortRow.clicked.connect(self.action_next_port)
        self.action_NextPortRow = self.tool_bar.addWidget(self.tool_NextPortRow)

        self.action_NextPortSep = self.tool_bar.addSeparator()

        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/")

        icon_init = QtGui.QIcon()
        icon_init.addPixmap(
            QtGui.QPixmap(os.path.join(icon_path, "initialize.png")), QtGui.QIcon.Normal
        )
        # icon_init.addPixmap(QtGui.QPixmap(os.path.join(icon_path, 'initialize-disabled.png')), QtGui.QIcon.Disabled)
        self.tool_Initialize = QtWidgets.QToolButton()
        self.tool_Initialize.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Initialize.setIcon(icon_init)  # normal and disabled pixmaps
        self.tool_Initialize.setText("Initialize")
        self.tool_Initialize.clicked.connect(self.action_initialize)
        self.tool_bar.addWidget(self.tool_Initialize)

        self.tool_bar.addSeparator()

        # ----------------------------------------------------------------------
        # Init RunControls object.  Replaces the old Start/Stop buttons and fill
        # status bar.
        self.run_controls = RunControls()
        self.run_controls.startRequested.connect(self.action_start)
        self.run_controls.stopRequested.connect(self.action_stop)
        self.run_controls.setEnabled(False)  # Set disabled initially
        self.tool_Start = self.run_controls  # backward-compat alias for legacy callers
        self.tool_Stop = self.run_controls
        self.tool_bar.addWidget(self.run_controls)
        self.tool_bar.addSeparator()
        # ----------------------------------------------------------------------

        icon_reset = QtGui.QIcon()
        icon_reset.addPixmap(
            QtGui.QPixmap(os.path.join(icon_path, "reset.png")), QtGui.QIcon.Normal
        )
        # icon_reset.addPixmap(QtGui.QPixmap(os.path.join(icon_path, 'reset-disabled.png')), QtGui.QIcon.Disabled) # not provided
        self.tool_Reset = QtWidgets.QToolButton()
        self.tool_Reset.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Reset.setIcon(icon_reset)
        self.tool_Reset.setText("Reset")
        self.tool_Reset.clicked.connect(self.action_reset)
        self.tool_bar.addWidget(self.tool_Reset)

        self.tool_bar.addSeparator()

        self._warningTimer = QtCore.QTimer()
        self._warningTimer.setSingleShot(True)
        self._warningTimer.timeout.connect(self.action_tempcontrol_warning)
        self._warningTimer.setInterval(2000)  # 2 second delay

        icon_temp = QtGui.QIcon()
        icon_temp.addPixmap(QtGui.QPixmap(os.path.join(icon_path, "temp.png")), QtGui.QIcon.Normal)
        # icon_temp.addPixmap(QtGui.QPixmap(os.path.join(icon_path, 'temp-disabled.png')), QtGui.QIcon.Disabled)
        self.tool_TempControl = QtWidgets.QToolButton()
        self.tool_TempControl.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_TempControl.setIcon(icon_temp)
        self.tool_TempControl.setText("Temp Control")
        self.tool_TempControl.setCheckable(True)
        self.tool_TempControl.clicked.connect(self.action_tempcontrol)
        # warn to "stop" before changing when disabled:
        self.tool_TempControl.enterEvent = self.action_tempcontrol_warn_start
        self.tool_TempControl.leaveEvent = self.action_tempcontrol_warn_stop
        # self.tool_TempControl.enterEvent = self.action_tempcontrol_warn_now
        self.tool_bar.addWidget(self.tool_TempControl)

        self.toolBar.addWidget(self.tool_bar)

        self.tempController = QtWidgets.QWidget()
        # warn to "stop" before changing when disabled:
        self.tempController.enterEvent = self.action_tempcontrol_warn_start
        self.tempController.leaveEvent = self.action_tempcontrol_warn_stop
        # self.tempController.enterEvent = self.action_tempcontrol_warn_now
        self.tempController.setMinimumSize(QtCore.QSize(200, 40))
        self.tempController.setFixedWidth(200)
        self.tempLayout = QtWidgets.QVBoxLayout()
        self.tempLayout.setContentsMargins(0, 5, 0, 5)
        self.tempLayout.addWidget(self.lTemp)
        self.tempLayout.addWidget(self.slTemp)
        self.tempController.setLayout(self.tempLayout)
        self.toolBar.addWidget(self.tempController)
        self.tempController.setEnabled(False)

        self.toolBar.addStretch()

        self.tool_bar_2 = QtWidgets.QToolBar()
        self.tool_bar_2.setIconSize(QtCore.QSize(50, 30))
        self.tool_bar_2.setStyleSheet("color: #333333;")

        # self.tool_Advanced = QtWidgets.QLabel("Advanced Settings")
        # self.tool_Advanced.setStyleSheet("color: #0D4AAF; text-decoration: none;")
        # self.tool_Advanced.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        # self.tool_Advanced.mousePressEvent = self.action_advanced
        # self.toolBar.addWidget(self.tool_Advanced)

        icon_advanced = QtGui.QIcon()
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/advanced.png")
        icon_advanced.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal)
        # icon_advanced.addPixmap(QtGui.QPixmap('QATCH/icons/advanced-disabled.png'), QtGui.QIcon.Disabled)
        self.tool_Advanced = QtWidgets.QToolButton()
        self.tool_Advanced.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        # normal and disabled pixmaps
        self.tool_Advanced.setIcon(icon_advanced)
        self.tool_Advanced.setText("Advanced")
        self.tool_Advanced.clicked.connect(self.action_advanced)
        self.tool_bar_2.addWidget(self.tool_Advanced)

        self.tool_bar_2.addSeparator()

        icon_user = QtGui.QIcon()
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/user.png")
        icon_user.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal)
        icon_user.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Disabled)
        self.tool_User = QtWidgets.QToolButton()
        self.tool_User.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_User.setIcon(icon_user)  # normal and disabled pixmaps
        self.tool_User.setText("Anonymous")
        self.tool_User.setEnabled(False)
        # self.tool_User.clicked.connect(self.action_user)
        self.tool_bar_2.addWidget(self.tool_User)

        self.toolBar.addWidget(self.tool_bar_2)

        self.toolBar.setContentsMargins(10, 10, 5, 5)
        self.toolBarWidget = QtWidgets.QWidget()
        self.toolBarWidget.setLayout(self.toolBar)
        self.toolBarWidget.setStyleSheet("background: #DDDDDD;")

        self.toolLayout.addWidget(self.toolBarWidget)
        self.toolLayout.addWidget(self.run_progress_bar)
        # self.toolLayout.addWidget(self.fill_status_progress_bar)

        if SHOW_SIMPLE_CONTROLS:
            # Remove bottom margin, leaving the rest as "default"
            self.toolLayout.setContentsMargins(11, 11, 11, 0)
            self.centralwidget.setLayout(self.toolLayout)

            self.Layout_controls.removeWidget(self.infosave)  # tec controller
            self.Layout_controls.removeWidget(self.lTemp)  # label
            self.Layout_controls.removeWidget(self.slTemp)  # slider
            self.Layout_controls.removeWidget(self.pTemp)  # start/stop button
            self.Layout_controls.removeWidget(self.run_progress_bar)
            self.Layout_controls.removeWidget(self.lg)  # user guide
            self.Layout_controls.removeWidget(self.lmail)  # email
            self.Layout_controls.removeWidget(self.l4)  # website
            self.Layout_controls.removeWidget(self.l3)  # logo

            self.advancedwidget = QtWidgets.QWidget()
            self.advancedwidget.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.WindowStaysOnTopHint)
            self.advancedwidget.setWhatsThis("These settings are for Advanced Users ONLY!")
            warningWidget = QtWidgets.QLabel(f"WARNING: {self.advancedwidget.whatsThis()}")
            warningWidget.setStyleSheet("background: #FF6600; padding: 1px; font-weight: bold;")
            warningLayout = QtWidgets.QVBoxLayout()
            warningLayout.addWidget(warningWidget)
            warningLayout.addLayout(self.gridLayout)
            self.advancedwidget.setLayout(warningLayout)
        else:
            self.centralwidget.setLayout(self.gridLayout)

        MainWindow1.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow1)

    def _update_progress_text(self):
        # get innerText from HTML in infobar
        plain_text = self.infobar.text()
        color = plain_text[plain_text.rindex("color=") + 6 : plain_text.rindex("color=") + 6 + 7]
        plain_text = plain_text[plain_text.index(">") + 1 :]
        plain_text = plain_text[plain_text.index(">") + 1 :]
        plain_text = plain_text[plain_text.index(">") + 1 :]
        plain_text = plain_text[0 : plain_text.rindex("<")]
        # remove any formatting tags: <b>, <i>, <u>
        while plain_text.rfind("<") != plain_text.find("<"):
            plain_text = plain_text[0 : plain_text.rindex("<")]
            plain_text = plain_text[plain_text.index(">") + 1 :]
        if len(plain_text) == 0:
            plain_text = "Progress: Not Started"
        else:
            plain_text = "Status: {}".format(plain_text)
        self.run_progress_bar.setFormat(plain_text)
        styleBar = """
                    QProgressBar
                    {
                     border: 0.5px solid #B8B8B8;
                     border-radius: 1px;
                     text-align: center;
                     color: {COLOR};
                     font-weight: bold;
                    }
                     QProgressBar::chunk
                    {
                     background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(184, 184, 184, 200), stop:1 rgba(221, 221, 221, 200));
                    }
                 """.replace("{COLOR}", color)
        self.run_progress_bar.setStyleSheet(styleBar)

    def _update_progress_value(self):
        if self.cBox_Source.currentIndex() == OperationType.measurement.value:
            pass  # self._update_progress_text() # defer to infobar text, not percentage
        else:
            self.run_progress_bar.setFormat("Progress: %p%")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(
            _translate(
                "MainWindow",
                "{} {} - Setup/Control".format(Constants.app_title, Constants.app_version),
            )
        )
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/")
        MainWindow.setWindowIcon(QtGui.QIcon(os.path.join(icon_path, "qatch-icon.png")))  # .png
        self.advancedwidget.setWindowIcon(
            QtGui.QIcon(os.path.join(icon_path, "advanced.png"))
        )  # .png
        self.advancedwidget.setWindowTitle(_translate("MainWindow2", "Advanced Settings"))
        self.pButton_Stop.setText(_translate("MainWindow", " STOP"))
        self.pButton_Start.setText(_translate("MainWindow", "START"))
        self.pButton_Clear.setText(_translate("MainWindow", "Clear Plots"))
        self.pButton_Reference.setText(_translate("MainWindow", "Set/Reset Reference"))
        self.pButton_ResetApp.setText(_translate("MainWindow", "Factory Defaults"))
        self.sBox_Samples.setSuffix(_translate("MainWindow", " / 5 min"))
        self.sBox_Samples.setPrefix(_translate("MainWindow", ""))
        self.chBox_export.setText(_translate("MainWindow", "Txt Export Sweep File"))
        self.chBox_freqHop.setText(_translate("MainWindow", "Mode Hop"))
        self.chBox_correctNoise.setText(_translate("MainWindow", "Show amplitude curve"))
        self.chBox_MultiAuto.setText(_translate("MainWindow", "Auto-detect channel count"))

    def action_next_port(self):
        """Method to handle advancing to the next port."""
        try:
            # Disable port button to prevent multi-clicks
            self.action_NextPortRow.setEnabled(False)

            controller_port = None
            for i in range(self.cBox_Port.count()):
                if self.cBox_Port.itemText(i).startswith("80:"):
                    controller_port = self.cBox_Port.itemData(i)
                    break

            if controller_port is None:
                Log.e("FLUX controller not found. Is it connected and powered on?")
                self.tool_NextPortRow.setIconError()
                self.action_NextPortRow.setEnabled(True)
                return

            next_port_num = self.tool_NextPortRow.value()

            if hasattr(self, "fluxThread"):
                if self.fluxThread.isRunning():
                    Log.d("Waiting for FLUX controller to stop.")
                    # wait for thread to quit, gracefully
                    if not self.fluxThread.wait(msecs=3000):
                        Log.w(
                            "Prior Flux controller thread still busy; skipping new Next Port request."
                        )
                        self.tool_NextPortRow.setEnabled(True)
                        return
            Log.d("Starting FLUX controller thread.")
            self.fluxThread = QtCore.QThread()
            self.fluxWorker = FLUXControl()
            self.fluxWorker.set_ports(controller=controller_port, next_port=next_port_num)
            self.fluxWorker.moveToThread(self.fluxThread)
            self.fluxThread.worker = self.fluxWorker
            self.fluxThread.started.connect(self.fluxWorker.run)
            self.fluxWorker.finished.connect(self.fluxThread.quit)
            self.fluxWorker.result.connect(self.next_port_result)
            self.fluxThread.start()

        except Exception as e:
            Log.e(f"action_next_port ERROR: {e}")

            self.tool_NextPortRow.setIconError()
            self.action_NextPortRow.setEnabled(True)  # show X error indicator

    def next_port_result(self, success):
        try:
            # Enable button to clear hourglass, show result (either number or red X)
            self.action_NextPortRow.setEnabled(True)

            if success:
                # Write to global port variable
                self.parent.parent.active_multi_ch = self.tool_NextPortRow.value()
                self.parent.parent.set_multi_mode()

            else:
                self.tool_NextPortRow.setIconError()  # trasient red text, resets on next update

                if PopUp.critical(
                    self,
                    "Next Port Failed",
                    "ERROR: Flux controller failed to move to the next port.",
                    btn1_text="Reset",
                ):
                    self.tool_NextPortRow.click()  # re-home stepper

        except Exception as e:
            Log.e(f"next_port_result ERROR: {e}")

    def action_initialize(self):
        """Method to handle initialization UI actions."""
        if self.pButton_Start.isEnabled():
            self.cBox_Source.setCurrentIndex(OperationType.calibration.value)
            if hasattr(self, "run_controls"):
                self.run_controls.set_running(False)
                self.run_controls.update_progress(0, 5, "Ready")
                self.run_controls.setEnabled(False)
            self.pButton_Start.clicked.emit()
            self.cal_initialized = True

    def action_start(self):
        """Method to handle start UI actions."""
        if self.pButton_Start.isEnabled():
            self.cBox_Source.setCurrentIndex(OperationType.measurement.value)
            # NOTE: This is done in separate signal on pButton_Start widget
            #       Doing it here breaks UI sync when started from Advanced
            # if hasattr(self, "run_controls"):
            #     self.run_controls.set_running(True)
            self.pButton_Start.clicked.emit()

    def action_stop(self):
        """Method to handle stop UI actions."""
        if self.pButton_Stop.isEnabled():
            self.cal_initialized = False
            self.pButton_Stop.clicked.emit()

    def action_reset(self):
        """Method to handle reset UI actions."""
        if self.tool_TempControl.isChecked():
            self.tool_TempControl.setChecked(False)
            self.tool_TempControl.clicked.emit()  # if running, stop
        self.slTemp.setValue(25)
        if self.pButton_Start.isEnabled():
            self.pButton_Clear.clicked.emit()
            self.pButton_Refresh.clicked.emit()
        self.infostatus.setStyleSheet("background: white; padding: 1px; border: 1px solid #cccccc")
        self.infostatus.setText("<font color=#333333 > Program Status Standby </font>")

        self.cal_initialized = False
        if hasattr(self, "run_controls"):
            self.run_controls.set_running(False)
            self.run_controls.update_progress(0, 5, "Idle")
            self.run_controls.setEnabled(False)

        # at least one device connected
        self.tool_TempControl.setEnabled(self.cBox_Port.count() > 1)

    def action_tempcontrol(self):
        self.tempController.setEnabled(self.tool_TempControl.isChecked())
        if self.tool_TempControl.isChecked():
            if self.pTemp.text().find("Stop") < 0:  # not found (i.e. "start" or "resume")
                self.pTemp.clicked.emit()
                self.slTemp.setFocus()
        else:
            if self.pTemp.text().find("Stop") >= 0:  # found (i.e. currently running, not locked)
                self.pTemp.clicked.emit()

    def action_tempcontrol_warn_start(self, event):
        self.event_windowPos = event.windowPos()
        self._warningTimer.start()

    def action_tempcontrol_warn_stop(self, event):
        self._warningTimer.stop()

    def action_tempcontrol_warn_now(self, event):
        self.event_windowPos = event.windowPos()
        self.action_tempcontrol_warning()

    def action_tempcontrol_warning(self):
        if self.tool_TempControl.isChecked() and not self.tool_TempControl.isEnabled():
            # Temp Control is checked (running) and not enabled (during measurement run)
            # Log.e("window pos:", self.event_windowPos)
            # Log.e("widget pos:", self.tempController.mapToGlobal(QtCore.QPoint(0, 0)))
            Log.w("WARNING: Temp Control mode cannot be changed during an active run.")
            if self.event_windowPos.x() >= self.tempController.mapToGlobal(QtCore.QPoint(0, 0)).x():
                Log.w(
                    'To adjust Temp Control: Press "Stop" first, then adjust setpoint accordingly.'
                )
            else:
                Log.w('To stop Temp Control: Press "Stop" first, then click "Temp Control" button.')

        # TODO: Not implemented; only show once per measurement run (maybe not the best idea)
        # else:
        #     if hasattr(self, "cached_warning_adjust"):
        #         # pass

    def action_advanced(self, obj):
        if self.advancedwidget.isVisible():
            self.advancedwidget.hide()
        self.advancedwidget.move(0, 0)
        self.advancedwidget.show()
        # make plate config button square
        self.pButton_PlateConfig.setFixedWidth(self.pButton_PlateConfig.height())
        # QtWidgets.QWhatsThis.enterWhatsThisMode()
        # QtWidgets.QWhatsThis.showText(
        #     QtCore.QPoint(int(self.advancedwidget.width() / 2), int(self.advancedwidget.height() * (2/3))),
        #     self.advancedwidget.whatsThis(),
        #     self.advancedwidget)

    def doPlateConfig(self):
        if hasattr(self, "wellPlateUI"):
            if self.wellPlateUI.isVisible():
                # close if already open, don't bother to ask to save unsaved changes (TODO)
                self.wellPlateUI.close()

        # Dynamically specify plate dimensions and number of devices connected to constructor
        # num port currently detected / connected
        num_ports = self.cBox_Port.count() - 1
        # allow 5 for flux controller (pid 0x80)
        if num_ports == 5:
            num_ports = 4
        i = self.cBox_Port.currentText()
        i = 0 if i.find(":") == -1 else int(i.split(":")[0], base=16)
        if i % 9 == i:  # 4x1 system
            well_width = 4  # number of well on a single device sensor for a multiplex device
            well_height = 1  # num of multiplex devices, ceil
        else:  # 4x6 system
            well_width = 6
            well_height = 4
        num_channels = self.cBox_MultiMode.currentIndex() + 1  # user define device count
        if num_ports not in [well_width, well_height] or num_ports == 1:
            PopUp.warning(
                self.parent,
                "Plate Configuration",
                f"<b>Multiplex device(s) are required for plate configuration.</b><br/>"
                + f"You must have exactly 4 device ports connected for this mode.<br/>"
                + f"Currently connected device port count is: {num_ports} (not 4)",
            )
        else:
            # creation of widget also shows UI to user
            self.wellPlateUI = WellPlate(well_width, well_height, num_channels)


#######################################################################################################################


class Ui_Plots(object):
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


############################################################################################################


class Ui_Info(object):
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


class Ui_Logger(object):
    def setupUi(self, MainWindow4):
        MainWindow4.setMinimumSize(QtCore.QSize(1000, 100))
        MainWindow4.move(0, 0)
        self.centralwidget = QtWidgets.QWidget(MainWindow4)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)

        logTextBox = QTextEditLogger(self.centralwidget)

        # log to text box
        logTextBox.setFormatter(
            logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt=None)
        )
        logging.getLogger("QATCH").addHandler(logTextBox)
        Log._show_user_info()

        MainWindow4.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow4)
        QtCore.QMetaObject.connectSlotsByName(MainWindow4)

    def retranslateUi(self, MainWindow4):
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/qatch-icon.png")
        MainWindow4.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        MainWindow4.setWindowTitle(
            _translate(
                "MainWindow4",
                "{} {} - Console".format(Constants.app_title, Constants.app_version),
            )
        )


class QTextEditLogger(logging.Handler, QtCore.QObject):
    appendInfoText = QtCore.pyqtSignal(str)
    appendDebugText = QtCore.pyqtSignal(str)
    forceRepaintEvents = False
    progressMode = False

    def __init__(self, parent):
        super().__init__()
        QtCore.QObject.__init__(self)

        # Initialize tab screen
        self.tabs = QtWidgets.QTabWidget()
        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()

        # Add tabs
        self.logInfo = QtWidgets.QTextEdit(parent)
        self.logInfo.setReadOnly(True)
        layout_v1 = QtWidgets.QVBoxLayout()
        layout_v1.addWidget(self.logInfo)
        self.tab1.setLayout(layout_v1)
        self.tabs.addTab(self.tab1, "Info")

        self.logDebug = QtWidgets.QTextEdit(parent)
        self.logDebug.setReadOnly(True)
        layout_v2 = QtWidgets.QVBoxLayout()
        layout_v2.addWidget(self.logDebug)
        self.tab2.setLayout(layout_v2)
        self.tabs.addTab(self.tab2, "Debug")

        self.tabs.setTabPosition(QtWidgets.QTabWidget.East)

        layout_v = QtWidgets.QVBoxLayout()
        layout_v.addWidget(self.tabs)
        parent.setLayout(layout_v)

        self.appendInfoText.connect(self.appendToInfo)  # logInfo.insertHtml)
        # logDebug.insertHtml)
        self.appendDebugText.connect(self.appendToDebug)
        self.last_record_msg = None

    def appendToInfo(self, html):
        if self.forceRepaintEvents and "[Device] ERROR:" in html:
            return  # do not show serial errors during firmware update on info console
        self.logInfo.moveCursor(QtGui.QTextCursor.End, QtGui.QTextCursor.MoveAnchor)
        if self.progressMode:
            # replace the most recent line with this new html line
            self.logInfo.textCursor().deletePreviousChar()
            self.logInfo.moveCursor(QtGui.QTextCursor.StartOfLine, QtGui.QTextCursor.MoveAnchor)
            self.logInfo.moveCursor(QtGui.QTextCursor.End, QtGui.QTextCursor.KeepAnchor)
            self.logInfo.textCursor().removeSelectedText()
        self.logInfo.insertHtml(html)
        self.logInfo.ensureCursorVisible()
        if self.forceRepaintEvents:
            self.logInfo.repaint()

    def appendToDebug(self, html):
        self.logDebug.moveCursor(QtGui.QTextCursor.End, QtGui.QTextCursor.MoveAnchor)
        self.logDebug.insertHtml(html)
        self.logDebug.ensureCursorVisible()
        if self.forceRepaintEvents:
            self.logDebug.repaint()
        if "GUI: Clear console window" in html:
            self.logInfo.clear()
        if "GUI: Force repaint events" in html:
            self.forceRepaintEvents = True
        if "GUI: Normal repaint events" in html:
            self.forceRepaintEvents = False
        if "GUI: Toggle progress mode" in html:
            self.progressMode = not self.progressMode

    def emit(self, record):
        msg = self.format(record)
        if msg == self.last_record_msg:
            # This must be print(), not Log.d(), or else an endless loop could occur!
            print(msg, "(duplicate record ignored)")
            return  # ignore duplicate records when they are handled back-to-back
        self.last_record_msg = msg
        msg = msg[msg.index(" ") + 1 :]  # trim date from console
        html_fmt = '<font style=\'font-family:"Lucida Console","Courier New",monospace;color:{};font-weight:{};\'>{}</font><br/><br/>'
        color = "black" if record.levelno <= logging.INFO else "red"
        weight = "normal" if record.levelno <= logging.WARNING else "bold"
        time_only = msg[0 : msg.index(",")]
        padding = "&nbsp;&nbsp;&nbsp;" if weight == "normal" else "&nbsp;&nbsp;"
        msg_info = time_only + padding + record.msg
        msg_debug = msg
        html_info = html_fmt.format(color, weight, msg_info)
        html_debug = html_fmt.format(color, weight, msg_debug)
        if record.levelno >= logging.INFO:
            self.appendInfoText.emit(html_info)
        self.appendDebugText.emit(html_debug)


class StartStopButton(QToolButton):
    """A custom Start/Stop/Progress button that displays progress and success animations.

    This widget draws a circular progress ring, handles state transitions between
    running, stopped, and completed, and renders specific icons (Play, Stop, Checkmark)
    based on the current state.

    Attributes:
        progress (float): Current progress value ranging from 0.0 to 1.0.
        is_running (bool): Flag indicating if the button is in the 'running' (stop icon) state.
        is_complete (bool): Flag indicating if the task has finished successfully.
        success_angle (float): Current angle for the success animation arc.
        animating_success (bool): Flag indicating if the success animation is currently active.
        color_blue (QColor): Color used for the active progress ring.
        color_green (QColor): Color used for the success state.
        color_track (QColor): Color of the background ring track.
        color_icon (QColor): Default color for the internal icons.
        color_disabled (QColor): Color used when the widget is disabled.
        success_anim (QVariantAnimation): Animation object for the success spin effect.
        progress_anim (QVariantAnimation): Animation object for smooth progress updates.
    """

    def __init__(self, parent=None):
        """Initializes the button with default states, colors, and animations.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.setAutoRaise(True)
        self.progress = 0.0
        self.is_running = False
        self.is_complete = False
        self.success_angle = 0
        self.animating_success = False
        self.color_blue = QColor("#00A3DA")
        self.color_green = QColor("#4CAF50")
        self.color_track = QColor("#9E9E9E")
        self.color_icon = QColor("#696969")
        self.color_disabled = QColor("#696969")
        self.color_darkgreen = QColor("#25B101")
        self.color_darkred = QColor("#FA4A3E")

        # Success Animation
        self.success_anim = QVariantAnimation()
        self.success_anim.setStartValue(0)
        self.success_anim.setEndValue(360)
        self.success_anim.setDuration(800)
        self.success_anim.valueChanged.connect(self._update_success_anim)
        self.success_anim.finished.connect(self._finish_success_anim)

        # Progress Animation
        self.progress_anim = QVariantAnimation()
        self.progress_anim.setDuration(500)
        self.progress_anim.setEasingCurve(QEasingCurve.OutQuad)
        self.progress_anim.valueChanged.connect(self._update_progress_anim)

    def animate_progress(self, target_value):
        """Smoothly interpolates the blue ring to a new progress value.

        Args:
            target_value (float): The progress value to animate towards (0.0 to 1.0).
        """
        self.progress_anim.stop()
        self.progress_anim.setStartValue(self.progress)
        self.progress_anim.setEndValue(target_value)
        self.progress_anim.start()

    def _update_progress_anim(self, value):
        """Slot called by the progress animation to update the ring value.

        Args:
            value (float): The current interpolated progress value.
        """
        self.progress = value
        self.update()

    def trigger_success(self):
        """Transitions the button to the success state.

        Stops any active progress animation, sets the state to complete,
        and initiates the success animation.
        """
        # Stop progress animation if we reach a successful fill
        self.progress_anim.stop()

        self.is_running = False
        self.is_complete = True
        self.animating_success = True
        self.success_anim.start()
        self.update()

    def reset(self):
        """Resets the button to its initial idle state.

        Clears progress, stops animations, and resets internal flags.
        """
        self.progress_anim.stop()
        self.is_running = False
        self.is_complete = False
        self.progress = 0.0
        self.animating_success = False
        self.update()

    def _update_success_anim(self, value):
        """Slot called by the success animation to update the angle.

        Args:
            value (int): The current angle of the success arc.
        """
        self.success_angle = value
        self.update()

    def _finish_success_anim(self):
        """Slot called when the success animation finishes to clean up state."""
        self.animating_success = False
        self.update()

    def update(self):
        """Overrides the default update method to trigger a repaint."""
        super().update()

        icon_size = QtCore.QSize(30, 30)
        self.setIcon(self._make_icon(icon_size))
        self.setIconSize(icon_size)

    def _make_icon(self, size: QtCore.QSize) -> QtGui.QIcon:
        """Handles the custom painting of the widget.

        Draws the three main components, the background track ring, the active progress ring/success ring,
        and the central icon.

        Args:
            size (QSize): The size of the QIcon to create.

        Returns:
            QIcon: The custom drawn icon representing the button state and progress.
        """
        pm = QtGui.QPixmap(size)
        pm.fill(Qt.transparent)

        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)

        # Custom drawing
        rect = pm.rect()
        center = rect.center()
        radius = min(rect.width(), rect.height()) / 2 - 2

        # Coloring
        if not self.isEnabled():
            track_color = self.color_disabled
            icon_color = self.color_disabled
            ring_color = self.color_disabled
        elif self.is_complete:
            track_color = self.color_track
            icon_color = self.color_green
            ring_color = self.color_green
        elif self.is_running:
            track_color = self.color_track
            icon_color = self.color_icon
            ring_color = self.color_blue
        else:  # enabled, not running or finished
            track_color = self.color_icon
            icon_color = self.color_icon
            ring_color = self.color_disabled

        #  Progress track
        painter.setPen(QPen(track_color, 2.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, radius, radius)

        # Active ring
        if self.isEnabled():
            if self.is_complete:
                # On success, trigger success animation.
                pen = QPen(ring_color, 2.5, Qt.SolidLine, Qt.RoundCap)
                painter.setPen(pen)

                if self.animating_success:
                    start = 90 * 16
                    span = -self.success_angle * 16
                    painter.drawArc(
                        QRectF(
                            center.x() - radius,
                            center.y() - radius,
                            radius * 2,
                            radius * 2,
                        ),
                        int(start),
                        int(span),
                    )
                else:
                    painter.drawEllipse(center, radius, radius)

            elif self.is_running and self.progress > 0:
                # While running, trigger progress animation.
                pen = QPen(ring_color, 2.5, Qt.SolidLine, Qt.RoundCap)
                painter.setPen(pen)
                angle_span = -self.progress * 360 * 16
                painter.drawArc(
                    QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2),
                    90 * 16,
                    int(angle_span),
                )

        # Render icons
        painter.setPen(Qt.NoPen)
        icon_size = 2 * radius * 0.75

        # NOTE: These have to be drawn dynamically to animate them.  The icons in the icon directory cannot be
        # animated properly so the checkmark is manually drawn, the Stop icon is drawn as a square, and the
        # Start icon is drawn as a triangle.
        if self.is_complete:
            # Checkmark
            painter.setBrush(QBrush(self.color_darkgreen))
            path = QPainterPath()
            path.moveTo(center.x() - icon_size * 0.4, center.y())
            path.lineTo(center.x() - icon_size * 0.1, center.y() + icon_size * 0.3)
            path.lineTo(center.x() + icon_size * 0.4, center.y() - icon_size * 0.4)

            check_pen = QPen(icon_color, 2.5)
            check_pen.setCapStyle(Qt.RoundCap)
            painter.strokePath(path, check_pen)

        elif self.is_running:
            # Stop Square
            painter.setBrush(QBrush(self.color_darkred))
            s = icon_size * 0.5
            painter.drawRect(QRectF(center.x() - s / 2, center.y() - s / 2, s, s))

        else:
            # Start Triangle
            painter.setBrush(QBrush(self.color_darkgreen))
            path = QPainterPath()
            h = icon_size * 0.6
            w = icon_size * 0.5
            x = center.x() - (w / 2) + 1.5
            y = center.y() - (h / 2)
            path.moveTo(x, y)
            path.lineTo(x + w, center.y())
            path.lineTo(x, y + h)
            path.closeSubpath()
            painter.drawPath(path)

        painter.end()

        return QtGui.QIcon(pm)


class RunControls(QWidget):
    """A composite RunControls widget combining a StartStopButton with a sliding status label.

    This control acts as a run controller. When the button is clicked, it emits
    signals to start or stop a process. When running, a status label slides out
    to the right to display textual progress details.

    Attributes:
        startRequested (pyqtSignal): Signal emitted when the user requests to start.
        stopRequested (pyqtSignal): Signal emitted when the user requests to stop.
        layout (QHBoxLayout): Main horizontal layout.
        btn (StartStopButton): The custom circular button instance with text label.
        status_container (QFrame): The collapsible container for the status text.
        status_label (QLabel): The label displaying current step information.
        anim (QPropertyAnimation): Animation for sliding the status container.
    """

    startRequested = pyqtSignal()
    stopRequested = pyqtSignal()

    def __init__(self, parent=None):
        """Initializes the UnifiedProgressControl.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Left container is button and label.
        self.btn = StartStopButton()
        # QSize taken from sizeHint() of tool_Initialize and/or tool_Reset
        self.btn.setFixedSize(QtCore.QSize(60, 56))
        self.btn.setText("Start")
        self.btn.clicked.connect(self.toggle_state)
        self.layout.addWidget(self.btn)

        # Right sliding status container.
        self.status_container = QFrame()
        self.status_container.setFixedWidth(0)
        self.status_container.setStyleSheet("background-color: transparent;")

        self.status_layout = QVBoxLayout(self.status_container)
        self.status_layout.setContentsMargins(5, 5, 5, 5)
        self.status_layout.setAlignment(Qt.AlignBottom)

        self.status_label = QLabel("Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #555; font-size: 12px; font-weight: bold;")
        self.status_label.setWordWrap(True)
        self.status_layout.addWidget(self.status_label)

        self.lbl_status = QLabel("Run Status")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #333; font-size: 11px; margin-top: 2px;")
        self.status_layout.addWidget(self.lbl_status)

        self.layout.addWidget(self.status_container)

        self.anim = QPropertyAnimation(self.status_container, b"minimumWidth")
        self.anim.setEasingCurve(QEasingCurve.InOutQuad)
        self.anim.setDuration(1000)

    def setEnabled(self, enabled):
        """Sets the enabled state of the control and its sub-widgets.

        Args:
            enabled (bool): True to enable, False to disable.
        """
        super().setEnabled(enabled)
        self.btn.setEnabled(enabled)
        self.btn.update()

    def toggle_state(self):
        """Toggles the state based on the current button status.

        Emits `stopRequested` if the button is currently running or complete.
        Emits `startRequested` if the button is idle.
        """
        if self.btn.is_complete:
            self.stopRequested.emit()
        elif self.btn.is_running:
            self.stopRequested.emit()
        else:
            self.startRequested.emit()

    def set_running(self, running=True):
        """Updates the UI to reflect whether a process is running.

        If running is True, the status container slides open. If False,
        it slides closed.

        Args:
            running (bool, optional): The target running state. Defaults to True.
        """
        if self.btn.is_running == running and not self.btn.is_complete:
            return

        if running:
            self.btn.is_complete = False
            self.btn.setText("Stop")
            self.anim.setStartValue(0)
            self.anim.setEndValue(160)
            self.anim.start()
        else:
            self.btn.is_complete = False
            self.btn.progress = 0.0
            self.btn.setText("Start")
            self.anim.setStartValue(160)
            self.anim.setEndValue(0)
            self.anim.start()

        self.btn.is_running = running
        self.btn.update()

    def update_progress(self, current_step, max_steps, fill_type_text):
        """Updates the progress button and status label text.

        If `current_step` meets or exceeds `max_steps`, the control triggers
        the success state on the button.

        Args:
            current_step (int): The current step number in the process.
            max_steps (int): The total number of steps.
            fill_type_text (str): Descriptive text to display in the status label.
        """
        if current_step >= max_steps and max_steps > 0:
            if not self.btn.is_complete:
                self.btn.trigger_success()
                self.status_label.setText(f"Done ({fill_type_text})")
                self.btn.setText("Done")
            return

        if max_steps > 0:
            percentage = current_step / max_steps
        else:
            percentage = 0
        self.btn.animate_progress(percentage)
        self.status_label.setText(f"{fill_type_text}")


class NumberIconButton(QtWidgets.QToolButton):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._error = False
        self._value = 1
        self._iconSize = QtCore.QSize(32, 32)

        self.setIconSize(self._iconSize)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.updateIcon()
        self.clicked.connect(self.advance)

    def advance(self):
        self._value += 1
        if self._value > 6 or self._error:
            self._error = False
            self._value = 1
        self.updateIcon()

    def value(self):
        # Used to get the current port step by the caller
        return self._value

    def setIconError(self):
        self._error = True
        self.updateIcon()  # redraw with error colors, clears on next "advance"

    def updateIcon(self, running=False):
        self.setIcon(self.makeIcon(self._value, running))

    def makeIcon(self, number, running=False, size=None):
        if size is None:
            size = self.iconSize()

        # Define hourglass shape vertices (16x16)
        points = [
            QtCore.QPoint(8 + 2, 8 + 2),  # Top Left
            QtCore.QPoint(8 + 14, 8 + 2),  # Top Right
            QtCore.QPoint(8 + 9, 8 + 8),  # Middle Right
            QtCore.QPoint(8 + 14, 8 + 14),  # Bottom Right
            QtCore.QPoint(8 + 2, 8 + 14),  # Bottom Left
            QtCore.QPoint(8 + 7, 8 + 8),  # Middle Left
        ]

        if not running:
            pm_hourglass = QtGui.QPixmap(size)
            self._beginPainter(pm_hourglass, False)

            # Circle (disabled)
            self.painter.drawEllipse(pm_hourglass.rect().adjusted(2, 2, -2, -2))

            # Hourglass (disabled)
            self.painter.drawPolygon(QtGui.QPolygon(points))

            self.painter.end()

        pm_number = QtGui.QPixmap(size)
        self._beginPainter(pm_number)

        # Circle (enabled)
        self.painter.drawEllipse(pm_number.rect().adjusted(2, 2, -2, -2))

        # Number (enabled)
        if not self._error:
            self.painter.drawText(pm_number.rect(), QtCore.Qt.AlignCenter, str(number))
        else:
            # Change pen to red, mark an X instead of the port number
            pen = QtGui.QPen(QtGui.QColor("#FF0000"), 2)
            self.painter.setPen(pen)

            self.painter.drawText(pm_number.rect(), QtCore.Qt.AlignCenter, "X")

        self.painter.end()

        icon = QtGui.QIcon()
        if not running:
            icon.addPixmap(pm_hourglass, QtGui.QIcon.Mode.Disabled, QtGui.QIcon.State.On)
            icon.addPixmap(pm_number, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On)
        else:
            icon.addPixmap(pm_number)  # same for enabled and disabled

        return icon

    def _beginPainter(self, device, enabled=True):
        device.fill(QtCore.Qt.transparent)

        self.painter = QtGui.QPainter()
        self.painter.begin(device)
        self.painter.setRenderHint(QtGui.QPainter.Antialiasing)

        pen = QtGui.QPen(QtGui.QColor("#444444" if enabled else "#888888"), 2)
        self.painter.setPen(pen)

        font = QtGui.QFont(self.font())
        font.setBold(True)
        font.setPointSize(int(self.iconSize().height() * 0.35))
        self.painter.setFont(font)


class FLUXControl(QtCore.QThread):
    result = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

    def set_ports(self, controller, next_port):
        self._controller = controller
        self._next_port = next_port

    def run(self):
        success = False
        FLUX_serial = None
        try:
            controller_port = self._controller
            next_port_num = self._next_port

            # Attempt to open port and print errors (if any)
            FLUX_serial = serial.Serial()

            # Configure serial port (assume baud to check before update)
            FLUX_serial.port = controller_port
            FLUX_serial.baudrate = Constants.serial_default_speed  # 115200
            FLUX_serial.stopbits = serial.STOPBITS_ONE
            FLUX_serial.bytesize = serial.EIGHTBITS
            FLUX_serial.timeout = Constants.serial_timeout_ms
            FLUX_serial.write_timeout = Constants.serial_writetimeout_ms
            FLUX_serial.open()

            step = next_port_num if next_port_num > 1 else 0  # re-home on "step 1"

            tecs = []
            if step in [0, 1, 2]:
                tecs.append("L")
            if step in [2, 3, 4]:
                tecs.append("C")
            if step in [4, 5, 6]:
                tecs.append("R")
            tec = ", ".join(tecs)

            probe = str(next_port_num)

            # NOTE: The stepper is interrupted in FW by pending serial
            #       so the STEP command must be last in the order sent
            flux_cmds = f"TEC {tec}\nPROBE {probe}\nSTEP {step}\n"

            Log.d(f"Port {next_port_num} control cmds: {flux_cmds}")

            # Read and show the TEC temp status from the device
            FLUX_serial.write(flux_cmds.encode())
            timeoutAt = time() + Constants.stepper_timeout_sec
            flux_reply = ""
            # timeout needed if old FW
            while time() < timeoutAt:
                if "Stepper: DONE!" in flux_reply:
                    break
                while (
                    FLUX_serial.in_waiting == 0 and time() < timeoutAt
                ):  # timeout needed if old FW:
                    QtCore.QThread.msleep(5)
                waiting = FLUX_serial.in_waiting
                if waiting > 0:
                    flux_reply += FLUX_serial.read(waiting).decode(errors="replace")

            if time() < timeoutAt:
                if (
                    "Stepper: DONE!" in flux_reply  # indicates stepper finished moving
                    and "Unknown input." not in flux_reply  # indicates TEC or PROBE cmd issue
                    and "Stopped" not in flux_reply
                ):  # indicates serial interrupted home action
                    Log.i(f"SUCCESS - Port {next_port_num} selected.")
                    success = True
                else:
                    Log.e(f"FAILURE - Port {next_port_num} NOT selected. Unexpected reply...")
            else:
                Log.e(f"TIMEOUT - Port {next_port_num} NOT selected. Controller timeout...")

            if not success:
                Log.d("Error Details: (serial response from port selection request)")
                for line in flux_reply.splitlines():
                    Log.d(f"ERROR >> {line}")
                Log.d('Expected last line from controller to be "Stepper: DONE!"')

        except Exception as e:
            Log.e(f"FLUXControl ERROR: {e}")

        finally:
            if FLUX_serial is not None and FLUX_serial.is_open:
                FLUX_serial.close()

            # always notify caller even on early failures
            self.result.emit(success)
            self.finished.emit()
