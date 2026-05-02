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

import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QDesktopWidget,
)

from QATCH.common.architecture import Architecture, OSType
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants, OperationType
from QATCH.ui.widgets.well_plate_widget import WellPlate
from QATCH.ui.popUp import PopUp
from QATCH.ui.components.number_icon_button import NumberIconButton
from QATCH.ui.components.run_controls_button import RunControls


class UIControls:  # QtWidgets.QMainWindow

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
