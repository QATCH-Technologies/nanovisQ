import pyzipper
import hashlib
import threading
import sys
import pyqtgraph as pg
import datetime as dt
from time import strftime, localtime, sleep
from xml.dom import minidom
from numpy import loadtxt
import numpy as np
from io import BytesIO
from PyQt5 import QtCore, QtGui, QtWidgets
from QATCH.QModel.q_image_clusterer import QClusterer
from QATCH.QModel.q_multi_model import QPredictor
from QATCH.common.architecture import Architecture
from QATCH.common.fileStorage import FileStorage, secure_open
from QATCH.common.fileManager import FileManager
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles, UserRoles
from QATCH.core.constants import Constants
from QATCH.models.ModelData import ModelData
from QATCH.ui.popUp import PopUp
from QATCH.ui.runInfo import QueryRunInfo
from QATCH.processors.CurveOptimizer import DifferenceFactorOptimizer, DropEffectCorrection

# from QATCH.QModel.QModel import QModelPredict
# import joblib
# from QATCH.QModel.q_data_pipeline import QDataPipeline
import os

# hide info/warning logs from tf # lazy load
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# from scipy.interpolate import UnivariateSpline # unused
# from scipy.optimize import curve_fit # lazy load
# from scipy.signal import argrelextrema # lazy load
# from scipy.signal import savgol_filter # lazy load

# import matplotlib.backends.backend_pdf # lazy load
# import matplotlib.pyplot as plt # lazy load

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hide info/warning logs from tf # lazy load
# import tensorflow as tf # lazy load

TAG = "[Analyze]"


###############################################################################
# Elaborate on the raw data gathered from the SerialProcess in parallel timing
###############################################################################
class AnalyzeProcess(QtWidgets.QWidget):
    progressValue = QtCore.pyqtSignal(int)
    progressFormat = QtCore.pyqtSignal(str)
    progressUpdate = QtCore.pyqtSignal()

    @staticmethod
    def Lookup_ST(surfactant, concentration):
        ST1 = 72
        if concentration > 2:  # mg/ml
            ST1 = 57.5
        return ST1  # always

        if concentration < 0.01:
            ST1 = 71
        else:
            X1 = np.log10(concentration)
            ST1 = np.polyval([-0.9092, -3.5982, 67], X1)
        X2 = np.log10(surfactant / 125)  # NOTE: np.log10(0) = -INF
        if X2 < -6:
            ST2 = ST1
        elif X2 < -5:
            ST2 = ST1 - 1
        elif -5 <= X2 <= -2.8:
            ST2 = ST1 - np.polyval([11.5, 59.5], X2)
        else:  # X2 > -2.8:
            ST2 = ST1 - 27
        # AnalyzeProcess.Lookup_Table("QATCH/resources/lookup_ST.csv", surfactant, concentration)
        return ST2

    @staticmethod
    def Lookup_CA(surfactant, concentration):
        CA = 55
        if concentration > 10:
            CA = CA - 0
        elif concentration >= 1:
            CA = CA - 0
        # AnalyzeProcess.Lookup_Table("QATCH/resources/lookup_CA.csv", surfactant, concentration)
        return CA

    @staticmethod
    def Lookup_DN(surfactant, concentration):
        return 1 + 2.62e-4 * concentration

    @staticmethod
    def Lookup_Table(table_path, surfactant, concentration):
        debug = False
        table = loadtxt(table_path, delimiter="\t")
        log_surfactant = np.log10(surfactant)
        # first row values, without A1 cell (empty)
        pcts = np.log10(table[0][1:])
        cons = table[:, 0][1:]  # first col values, without A1 cell (empty)
        data = table[1:, 1:]  # data, without header row/col (for indexing)
        if debug:
            Log.d(data)

        # find surfactant position in table
        s_idx = [None]
        if surfactant in pcts:
            s_idx = np.where(pcts == log_surfactant)
        elif log_surfactant < pcts[0]:
            s_idx = [0, 1]
        elif log_surfactant > pcts[-1]:
            tmp = len(pcts) - 1
            s_idx = [tmp - 1, tmp]
        else:
            tmp = next(i for i, t in enumerate(pcts) if t > log_surfactant)
            s_idx = [tmp - 1, tmp]
        if debug:
            Log.d(s_idx)

        # find concentration position in table
        c_idx = [None]
        if concentration in cons:
            c_idx = np.where(cons == concentration)
        elif concentration < cons[0]:
            c_idx = [0, 1]
        elif concentration > cons[-1]:
            tmp = len(cons) - 1
            c_idx = [tmp - 1, tmp]
        else:
            tmp = next(i for i, t in enumerate(cons) if t > concentration)
            c_idx = [tmp - 1, tmp]
        if debug:
            Log.d(c_idx)

        ret = 0
        if len(c_idx) == 2 and len(s_idx) == 2:
            # most complex case, extrapolate both ways, then average together
            row1 = data[c_idx[0]]
            if debug:
                Log.d(f"row1 = {row1}")
            s_ratio = (log_surfactant - pcts[s_idx[0]]) / (
                pcts[s_idx[1]] - pcts[s_idx[0]]
            )
            if debug:
                Log.d(f"s_ratio = {s_ratio}")
            val1 = row1[s_idx[0]] + (row1[s_idx[1]] - row1[s_idx[0]]) * s_ratio
            if debug:
                Log.d(f"val1 = {val1}")
            row2 = data[c_idx[1]]
            if debug:
                Log.d(f"row2 = {row2}")
            val2 = row2[s_idx[0]] + (row2[s_idx[1]] - row2[s_idx[0]]) * s_ratio
            if debug:
                Log.d(f"val2 = {val2}")
            c_ratio = (concentration - cons[c_idx[0]]) / (
                cons[c_idx[1]] - cons[c_idx[0]]
            )
            if debug:
                Log.d(f"ratio = {c_ratio}")
            ret1 = val1 + (val2 - val1) * c_ratio
            if debug:
                Log.d(f"ret1 = {ret1}")
            col1 = data[:, s_idx[0]]
            if debug:
                Log.d(f"col1 = {col1}")
            val1 = col1[c_idx[0]] + (col1[c_idx[1]] - col1[c_idx[0]]) * c_ratio
            if debug:
                Log.d(f"val1 = {val1}")
            col2 = data[:, s_idx[1]]
            if debug:
                Log.d(f"col2 = {col2}")
            val2 = col2[c_idx[0]] + (col2[c_idx[1]] - col2[c_idx[0]]) * c_ratio
            if debug:
                Log.d(f"val2 = {val2}")
            ret2 = val1 + (val2 - val1) * s_ratio
            if debug:
                Log.d(f"ret2 = {ret2}")
            ret = (ret1 + ret2) / 2
        if len(c_idx) == 2 and len(s_idx) == 1:
            col = data[:, s_idx]
            if debug:
                Log.d(col)
            ratio = (concentration - cons[c_idx[0]]) / \
                (cons[c_idx[1]] - cons[c_idx[0]])
            if debug:
                Log.d(ratio)
            ret = (col[c_idx[0]] + (col[c_idx[1]] -
                   col[c_idx[0]]) * ratio)[0][0]
        if len(c_idx) == 1 and len(s_idx) == 2:
            row = data[c_idx][0]
            if debug:
                Log.d(row)
            ratio = (log_surfactant - pcts[s_idx[0]]) / (
                pcts[s_idx[1]] - pcts[s_idx[0]]
            )
            if debug:
                Log.d(ratio)
            ret = row[s_idx[0]] + (row[s_idx[1]] - row[s_idx[0]]) * ratio
        if len(c_idx) == 1 and len(s_idx) == 1:
            ret = data[c_idx, s_idx][0][0]

        lookup_type = table_path[table_path.rindex("/") + 1: -4]
        Log.d(f"{lookup_type}({surfactant:1.3f}, {concentration:3.0f}) = {ret:2.2f}")
        return ret

    @staticmethod
    def Model_Data(data_path):
        val = False
        try:
            with secure_open(data_path, "r", "capture") as f:
                csv_headers = next(f)

                if "Ambient" in csv_headers:
                    csv_cols = (2, 4, 6, 7)
                else:
                    csv_cols = (2, 3, 5, 6)

                data = np.loadtxt(
                    f.readlines(), delimiter=",", skiprows=0, usecols=csv_cols
                )
                relative_time = data[:, 0]
                temperature = data[:, 1]
                resonance_frequency = data[:, 2]
                dissipation = data[:, 3]

                if Constants.ModelData_predict:
                    try:
                        dataModel = ModelData()
                        model_result = dataModel.IdentifyPoints(
                            data_path=data_path,
                            times=relative_time,
                            freq=resonance_frequency,
                            diss=dissipation,
                        )
                        if model_result != -1:
                            val = True

                        return val  # if we got here, skip the rest of this method, return now
                    except:
                        Log.e(
                            "Error modeling data... Using 'tensorflow' as a backup (slow)."
                        )

                if Constants.Tensorflow_predict:
                    # raw data
                    xs = relative_time
                    ys = dissipation

                    t_0p5 = (
                        0
                        if (xs[-1] < 0.5)
                        else next(x for x, t in enumerate(xs) if t > 0.5)
                    )
                    t_1p0 = (
                        100
                        if (len(xs) < 500)
                        else next(x for x, t in enumerate(xs) if t > 1.0)
                    )

                    # t_1p0, done = QtWidgets.QInputDialog.getDouble(None, 'Input Dialog', 'Confirm rough start index:', value=t_1p0)

                    # new maths for resonance and dissipation (scaled)
                    avg = np.average(resonance_frequency[t_0p5:t_1p0])
                    ys = ys * avg / 2
                    # ys_fit = ys_fit * avg / 2
                    ys = ys - np.amin(ys)
                    # ys_fit = ys_fit - np.amin(ys_fit)
                    ys_freq = avg - resonance_frequency
                    # ys_freq_fit = savgol_filter(ys_freq, smooth_factor, 1)

                    baseline = np.average(dissipation[t_0p5:t_1p0])
                    diff_factor = (
                        Constants.default_diff_factor
                    )  # 1.0 if baseline < 50e-6 else 1.5
                    # if hasattr(self, "diff_factor"):
                    #     diff_factor = self.diff_factor
                    ys_diff = (
                        ys_freq - diff_factor * ys
                    )  # NOTE: For temporary testing as of Pi Day 2023! (3 places in this file)

                    # Invert difference curve if drop applied to outlet
                    if np.average(ys_diff) < 0:
                        Log.w(
                            "Inverting DIFFERENCE curve due to negative initial fill deltas"
                        )
                        ys_diff *= -1

                    # ys_diff_fit = savgol_filter(ys_diff, smooth_factor, 1)
                    Log.d(f"Difference factor: {diff_factor:1.3f}x")

                    lin_xs = np.linspace(
                        xs[0], xs[-1], 1000
                    )  # model trained with 1000 points
                    lin_ys = np.interp(lin_xs, xs, ys)
                    lin_ys_freq = np.interp(lin_xs, xs, ys_freq)
                    lin_ys_diff = np.interp(lin_xs, xs, ys_diff)

                    # lazy load tensorflow module
                    os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
                        "3"  # hide info/warning logs from tf
                    )
                    import tensorflow as tf

                    # import tensorflow, load model, and predict good or bad
                    model_path = os.path.join(
                        Architecture.get_path(), "QATCH/models/")
                    time_model = tf.keras.models.load_model(
                        os.path.join(model_path, "time_model")
                    )
                    diss_model = tf.keras.models.load_model(
                        os.path.join(model_path, "diss_model")
                    )
                    freq_model = tf.keras.models.load_model(
                        os.path.join(model_path, "freq_model")
                    )
                    diff_model = tf.keras.models.load_model(
                        os.path.join(model_path, "diff_model")
                    )

                    data_time = lin_xs
                    data_diss = lin_ys
                    data_freq = lin_ys_freq
                    data_diff = lin_ys_diff

                    predict_time = (
                        # max(0, min(1, time_model([data_time]).numpy()[0][0]))
                        0
                    )
                    predict_diss = max(
                        0, min(1, diss_model([data_diss]).numpy()[0][0]))
                    predict_freq = max(
                        0, min(1, freq_model([data_freq]).numpy()[0][0]))
                    predict_diff = max(
                        0, min(1, diff_model([data_diff]).numpy()[0][0]))

                    predictors_count = 3  # ignore time
                    predict_data = (
                        predict_time + predict_diss + predict_freq + predict_diff
                    ) / predictors_count
                    val = max(0, min(1, np.round(predict_data).astype(int)))
        except Exception as e:
            # raise e
            Log.e("ERROR: Model encountered an exception while analyzing run data.")
        return val  # true if good

    def __init__(self, parent=None):
        super(AnalyzeProcess, self).__init__(None)
        self.parent = parent
        self.stateStep = -1
        self.zoomLevel = 1
        self.xml_path = None
        self.poi_markers = []
        self.sort_order = 1  # by date, default
        self.scan_for_most_recent_run = True
        self.run_timestamps = {}
        self.run_devices = {}
        self.run_names = {}
        self.step_direction = "forwards"
        self.allow_modify = False
        self.moved_markers = [False, False, False, False, False, False]
        self.signed_at = "[NEVER]"
        self.model_result = -1
        self.model_candidates = None
        self.model_engine = "None"
        self.analyzer_task = QtCore.QThread()
        self.dataModel = ModelData()

        # lazy load these modules on 'loadRun()' call
        self.QModel_modules_loaded = False
        # QClusterer(model_path=cluster_model_path)
        self.QModel_clusterer = None
        self.QModel_predict_0 = (
            None  # QPredictor(model_path=predict_model_path.format(0))
        )
        self.QModel_predict_1 = (
            None  # QPredictor(model_path=predict_model_path.format(1))
        )
        self.QModel_predict_2 = (
            None  # QPredictor(model_path=predict_model_path.format(2))
        )

        screen = QtWidgets.QDesktopWidget().availableGeometry()
        USE_FULLSCREEN = screen.width() == 2880
        pct_width = 75
        pct_height = 75
        self.resize(
            int(screen.width() * pct_width / 100),
            int(screen.height() * pct_height / 100),
        )
        self.move(
            int(screen.width() * (100 - pct_width) / 200),
            int(screen.height() * (100 - pct_width) / 200),
        )

        self.layout = QtWidgets.QVBoxLayout(self)

        # Fixes #30
        self.text_Devices = QtWidgets.QLabel("Show Only:")
        self.cBox_Devices = QtWidgets.QComboBox()
        self.text_Runs = QtWidgets.QLabel("Run:")
        self.cBox_Runs = QtWidgets.QComboBox()
        self.btn_Load = QtWidgets.QPushButton("Load")
        self.btn_Back = QtWidgets.QPushButton("Back")
        self.btn_Next = QtWidgets.QPushButton("Next")
        self.text_Loaded = QtWidgets.QLabel("Loaded:")
        self.text_Created = QtWidgets.QLabel("[NONE]")
        self.btn_Info = QtWidgets.QPushButton("Run Info")

        # START ANALYZE SIGNATURE CODE:
        # This code also exists in popUp.py in class QueryRunInfo for "CAPTURE SIGNATURE CODE"
        # The following method also is duplicated in both files: 'self.switch_user_at_sign_time'
        # There is duplicated logic code within the submit button handler: 'self.action_analyze'
        # The method for handling keystroke shortcuts is also duplicated too: 'self.eventFilter'
        self.signForm = QtWidgets.QWidget()
        self.signForm.setWindowFlags(
            QtCore.Qt.Dialog
        )  # | QtCore.Qt.WindowStaysOnTopHint)
        icon_path = os.path.join(
            Architecture.get_path(), "QATCH/icons/sign.png")
        self.signForm.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        self.signForm.setWindowTitle("Signature")
        layout_sign = QtWidgets.QVBoxLayout()
        layout_curr = QtWidgets.QHBoxLayout()
        signedInAs = QtWidgets.QLabel("Signed in as: ")
        signedInAs.setAlignment(QtCore.Qt.AlignLeft)
        layout_curr.addWidget(signedInAs)
        self.signedInAs = QtWidgets.QLabel("[NONE]")
        self.signedInAs.setAlignment(QtCore.Qt.AlignRight)
        layout_curr.addWidget(self.signedInAs)
        layout_sign.addLayout(layout_curr)
        line_sep = QtWidgets.QFrame()
        line_sep.setFrameShape(QtWidgets.QFrame.HLine)
        line_sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout_sign.addWidget(line_sep)
        layout_switch = QtWidgets.QHBoxLayout()
        self.signerInit = QtWidgets.QLabel(f"Initials: <b>N/A</b>")
        layout_switch.addWidget(self.signerInit)
        switch_user = QtWidgets.QPushButton("Switch User")
        switch_user.clicked.connect(self.switch_user_at_sign_time)
        layout_switch.addWidget(switch_user)
        layout_sign.addLayout(layout_switch)
        self.sign = QtWidgets.QLineEdit()
        self.sign.installEventFilter(self)
        layout_sign.addWidget(self.sign)
        self.sign_do_not_ask = QtWidgets.QCheckBox(
            "Do not ask again this session")
        self.sign_do_not_ask.setEnabled(False)
        if UserProfiles.checkDevMode()[0]:  # DevMode enabled
            auto_sign_key = None
            session_key = None
            if os.path.exists(Constants.auto_sign_key_path):
                with open(Constants.auto_sign_key_path, "r") as f:
                    auto_sign_key = f.readline()
            session_key_path = os.path.join(
                Constants.user_profiles_path, "session.key")
            if os.path.exists(session_key_path):
                with open(session_key_path, "r") as f:
                    session_key = f.readline()
            if auto_sign_key == session_key and session_key != None:
                self.sign_do_not_ask.setChecked(True)
            else:
                self.sign_do_not_ask.setChecked(False)
                if os.path.exists(Constants.auto_sign_key_path):
                    os.remove(Constants.auto_sign_key_path)
            layout_sign.addWidget(self.sign_do_not_ask)
        self.sign_ok = QtWidgets.QPushButton("OK")
        self.sign_ok.clicked.connect(self.signForm.hide)
        self.sign_ok.clicked.connect(self.action_analyze)
        self.sign_ok.setDefault(True)
        self.sign_ok.setAutoDefault(True)
        self.sign_cancel = QtWidgets.QPushButton("Cancel")
        self.sign_cancel.clicked.connect(self.signForm.hide)
        layout_ok_cancel = QtWidgets.QHBoxLayout()
        layout_ok_cancel.addWidget(self.sign_ok)
        layout_ok_cancel.addWidget(self.sign_cancel)
        layout_sign.addLayout(layout_ok_cancel)
        self.signForm.setLayout(layout_sign)
        # END ANALYZE SIGNATURE CODE

        self.graphStack = (
            QtWidgets.QStackedWidget()
        )  # must define this here, before connecting "self._update_progress_value" in the next section

        # Progressbar -------------------------------------------------------------
        styleBar = """
                    QProgressBar
                    {
                     border: 0.5px solid #B8B8B8;
                     border-radius: 1px;
                     text-align: center;
                     color: #333333;
                     font-weight: bold;
                    }
                     QProgressBar::chunk
                    {
                     background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(184, 184, 184, 200), stop:1 rgba(221, 221, 221, 200));
                    }
                 """  # background:url("openQCM/icons/openqcm-logo.png")
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setStyleSheet(styleBar)
        # self.progressBar.setProperty("value", 0)
        self.progressBar.setGeometry(QtCore.QRect(0, 0, 50, 10))
        self.progressBar.setObjectName("progressBar")
        # self.progressBar.setFixedHeight(50)
        self.progressBar.valueChanged.connect(self._update_progress_value)
        self.progressBar.setValue(0)

        self.tool_Load = QtWidgets.QToolBar()
        self.tool_Load.setIconSize(QtCore.QSize(50, 30))
        self.tool_Load.setStyleSheet("color: #333333;")

        icon_load = QtGui.QIcon()
        icon_path = os.path.join(
            Architecture.get_path(), "QATCH/icons/load.png")
        icon_load.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal)
        # icon_load.addPixmap(QtGui.QPixmap('QATCH/icons/load-disabled.png'), QtGui.QIcon.Disabled)
        self.tBtn_Load = QtWidgets.QToolButton()
        self.tBtn_Load.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tBtn_Load.setIcon(icon_load)  # normal and disabled pixmaps
        self.tBtn_Load.setText("Load")
        self.tBtn_Load.clicked.connect(self.loadRun)
        self.tool_Load.addWidget(self.tBtn_Load)

        icon_reset = QtGui.QIcon()
        icon_path = os.path.join(
            Architecture.get_path(), "QATCH/icons/reset.png")
        icon_reset.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal)
        # icon_reset.addPixmap(QtGui.QPixmap('QATCH/icons/load-disabled.png'), QtGui.QIcon.Disabled)
        self.tBtn_Rescan = QtWidgets.QToolButton()
        self.tBtn_Rescan.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tBtn_Rescan.setIcon(icon_reset)  # normal and disabled pixmaps
        self.tBtn_Rescan.setText("Rescan")
        self.tBtn_Rescan.clicked.connect(self.rescanRuns)
        self.tool_Load.addWidget(self.tBtn_Rescan)

        self.tool_Load.addSeparator()

        icon_info = QtGui.QIcon()
        icon_path = os.path.join(
            Architecture.get_path(), "QATCH/icons/info.png")
        icon_info.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal)
        # icon_info.addPixmap(QtGui.QPixmap('QATCH/icons/load-disabled.png'), QtGui.QIcon.Disabled)
        self.tBtn_Info = QtWidgets.QToolButton()
        self.tBtn_Info.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tBtn_Info.setIcon(icon_info)  # normal and disabled pixmaps
        self.tBtn_Info.setText("Run Info")
        self.tBtn_Info.clicked.connect(self.getRunInfo)
        self.tool_Load.addWidget(self.tBtn_Info)

        self.tool_Load.addSeparator()

        # define simple layout - only add to central widget if requested
        self.toolLayout = QtWidgets.QVBoxLayout()
        self.toolBar = QtWidgets.QHBoxLayout()

        self.sort_by = QtWidgets.QLabel("Sort by:")
        self.sort_by_name = QtWidgets.QLabel("Name")
        # self.sort_by_name.setStyleSheet("color: #0D4AAF; text-decoration: none; padding-left: 15px;")
        self.sort_by_name.setCursor(
            QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.sort_by_name.mousePressEvent = self.action_sort_by_name
        self.sort_by_date = QtWidgets.QLabel("Date")
        # self.sort_by_date.setStyleSheet("color: #0D4AAF; text-decoration: none; padding-left: 15px; font-weight: bold;")
        self.sort_by_date.setCursor(
            QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.sort_by_date.mousePressEvent = self.action_sort_by_date

        sort_by_layout = QtWidgets.QHBoxLayout()
        sort_by_layout.setContentsMargins(0, 0, 0, 0)
        sort_by_layout.addWidget(self.sort_by)
        sort_by_layout.addWidget(self.sort_by_name)
        sort_by_layout.addWidget(self.sort_by_date)
        sort_by_layout.addStretch()

        self.sort_by_widget = QtWidgets.QWidget()
        self.sort_by_widget.setLayout(sort_by_layout)

        self.runGrid = QtWidgets.QGridLayout()
        self.runGrid.setContentsMargins(0, 0, 0, 0)
        # row, col, rowspan, colspan
        self.runGrid.addWidget(self.sort_by_widget, 1, 2)
        self.runGrid.addWidget(self.text_Runs, 2, 1)
        self.runGrid.addWidget(self.cBox_Runs, 2, 2)
        self.runGrid.addWidget(self.text_Created, 3, 2)
        # self.toolBar.addWidget(self.text_Runs)
        # self.toolBar.addWidget(self.cBox_Runs)
        self.toolBar.addLayout(self.runGrid)
        self.toolBar.addWidget(self.tool_Load)
        # self.toolBar.addWidget(self.text_Devices)
        # self.toolBar.addWidget(self.cBox_Devices)

        self.sort_by_widget.setFixedHeight(14)
        self.cBox_Runs.setFixedHeight(20)
        self.text_Created.setFixedHeight(14)
        self.sort_by.setStyleSheet("padding-left: 1px;")
        self.text_Created.setStyleSheet("padding-left: 1px;")
        self.text_Runs.setFixedWidth(50)
        self.text_Runs.setStyleSheet("padding-left: 10px;")
        # self.text_Devices.setStyleSheet("padding-left: 0px;")
        # self.cBox_Devices.setFixedHeight(25)
        # self.cBox_Devices.setStyleSheet("padding-right: 5px;")

        self.toolBar.addStretch()

        self.tool_bar = QtWidgets.QToolBar()
        self.tool_bar.setIconSize(QtCore.QSize(50, 30))
        self.tool_bar.setStyleSheet("color: #333333;")

        self.tool_bar.addSeparator()

        icon_cancel = QtGui.QIcon()
        icon_path = os.path.join(
            Architecture.get_path(), "QATCH/icons/cancel.png")
        icon_cancel.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal)
        # icon_cancel.addPixmap(QtGui.QPixmap('QATCH/icons/cancel-disabled.png'), QtGui.QIcon.Disabled)
        self.tool_Cancel = QtWidgets.QToolButton()
        self.tool_Cancel.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Cancel.setIcon(icon_cancel)
        self.tool_Cancel.setText("Close")
        self.tool_Cancel.clicked.connect(self.action_cancel)
        self.tool_bar.addWidget(self.tool_Cancel)

        self.tool_bar.addSeparator()

        # icon_back = QtGui.QIcon()
        # icon_back.addPixmap(QtGui.QPixmap("QATCH/icons/back.png"), QtGui.QIcon.Normal)
        # icon_back.addPixmap(QtGui.QPixmap('QATCH/icons/back-disabled.png'), QtGui.QIcon.Disabled)
        self.tool_Back = QtWidgets.QToolButton()
        self.tool_Back.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Back.setArrowType(QtCore.Qt.LeftArrow)
        # self.tool_Back.setIcon(icon_back)
        self.tool_Back.setText("Back")
        self.tool_Back.clicked.connect(self.action_back)
        self.tool_bar.addWidget(self.tool_Back)

        # icon_next = QtGui.QIcon()
        # icon_next.addPixmap(QtGui.QPixmap("QATCH/icons/next.png"), QtGui.QIcon.Normal)
        # icon_next.addPixmap(QtGui.QPixmap('QATCH/icons/next-disabled.png'), QtGui.QIcon.Disabled) # not provided
        self.tool_Next = QtWidgets.QToolButton()
        self.tool_Next.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Next.setArrowType(QtCore.Qt.RightArrow)
        # self.tool_Next.setIcon(icon_next)
        self.tool_Next.setText("Next")
        self.tool_Next.clicked.connect(self.action_next)
        self.tool_bar.addWidget(self.tool_Next)

        self.tool_bar.addSeparator()

        icon_modify = QtGui.QIcon()
        icon_path = os.path.join(
            Architecture.get_path(), "QATCH/icons/modify.png")
        icon_modify.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal)
        # icon_modify.addPixmap(QtGui.QPixmap('QATCH/icons/modify-disabled.png'), QtGui.QIcon.Disabled)
        self.tool_Modify = QtWidgets.QToolButton()
        self.tool_Modify.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Modify.setIcon(icon_modify)
        self.tool_Modify.setText("Modify")
        self.tool_Modify.setCheckable(True)
        self.tool_Modify.clicked.connect(self.action_modify)
        self.tool_bar.addWidget(self.tool_Modify)

        icon_analyze = QtGui.QIcon()
        icon_path = os.path.join(
            Architecture.get_path(), "QATCH/icons/start.png")
        icon_analyze.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal)
        # icon_analyze.addPixmap(QtGui.QPixmap('QATCH/icons/analyze-disabled.png'), QtGui.QIcon.Disabled)
        self.tool_Analyze = QtWidgets.QToolButton()
        self.tool_Analyze.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Analyze.setIcon(icon_analyze)  # normal and disabled pixmaps
        self.tool_Analyze.setText("Analyze")
        self.tool_Analyze.clicked.connect(
            self.action_analyze
        )  # TODO: skip ahead to analyze (if pois are all set)
        self.tool_bar.addWidget(self.tool_Analyze)

        self.tool_bar.addSeparator()

        self.toolBar.addWidget(self.tool_bar)

        # self.toolBar.addStretch()

        # self.tool_Advanced = QtWidgets.QLabel("Advanced Settings")
        # self.tool_Advanced.setStyleSheet("color: #0D4AAF; text-decoration: none; padding-left: 50px;")
        # self.tool_Advanced.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        # self.tool_Advanced.mousePressEvent = self.action_advanced
        # self.toolBar.addWidget(self.tool_Advanced)

        icon_advanced = QtGui.QIcon()
        icon_path = os.path.join(
            Architecture.get_path(), "QATCH/icons/advanced.png")
        icon_advanced.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal)
        # icon_advanced.addPixmap(QtGui.QPixmap('QATCH/icons/advanced-disabled.png'), QtGui.QIcon.Disabled)
        self.tool_Advanced = QtWidgets.QToolButton()
        self.tool_Advanced.setToolButtonStyle(
            QtCore.Qt.ToolButtonTextUnderIcon)
        # normal and disabled pixmaps
        self.tool_Advanced.setIcon(icon_advanced)
        self.tool_Advanced.setText("Advanced")
        self.tool_Advanced.clicked.connect(self.action_advanced)
        self.tool_bar.addWidget(self.tool_Advanced)

        self.tool_bar.addSeparator()

        icon_user = QtGui.QIcon()
        icon_path = os.path.join(
            Architecture.get_path(), "QATCH/icons/user.png")
        icon_user.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal)
        icon_user.addPixmap(QtGui.QPixmap(
            "QATCH/icons/user.png"), QtGui.QIcon.Disabled)
        self.tool_User = QtWidgets.QToolButton()
        self.tool_User.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_User.setIcon(icon_user)  # normal and disabled pixmaps
        self.tool_User.setText("Anonymous")
        self.tool_User.setEnabled(False)
        # self.tool_User.clicked.connect(self.action_user)
        self.tool_bar.addWidget(self.tool_User)

        # self.toolBar.addWidget(self.tool_bar_2)

        self.toolBar.setContentsMargins(10, 10, 5, 5)
        self.toolBarWidget = QtWidgets.QWidget()
        self.toolBarWidget.setObjectName("toolBarWidget")
        self.toolBarWidget.setLayout(self.toolBar)
        self.toolBarWidget.setStyleSheet(
            "#toolBarWidget, QToolButton { background: #DDDDDD; }"
        )

        self.toolLayout.addWidget(self.toolBarWidget)
        self.toolLayout.addWidget(self.progressBar)

        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")

        # Devices ------------------------------------------------------
        self.l0 = QtWidgets.QLabel()
        self.l0.setStyleSheet("background: #008EC0; padding: 1px;")

        # Fixing issue #30
        self.l0.setText("<font color=#ffffff >Run Selection</font> </a>")
        if USE_FULLSCREEN:
            self.l0.setFixedHeight(50)
        # else:
        #    self.l0.setFixedHeight(15)
        self.gridLayout.addWidget(self.l0, 1, 1, 1, 4)

        self.gridLayout.addWidget(self.text_Devices, 2, 1)
        self.gridLayout.addWidget(
            self.cBox_Devices, 2, 2, 1, 2
        )  # row, col, rowspan, colspan

        # Fixes #30
        self.showRunsFromAllDevices = QtWidgets.QCheckBox(
            "Show all available runs")
        self.showRunsFromAllDevices.setChecked(True)
        self.showRunsFromAllDevices.clicked.connect(
            self.showRunsFromAllDevices_clicked)
        self.cBox_Devices.setEnabled(False)

        self.gridLayout.addWidget(self.showRunsFromAllDevices, 3, 2, 1, 2)

        # Parameters ------------------------------------------------------
        self.l1 = QtWidgets.QLabel()
        self.l1.setStyleSheet("background: #008EC0; padding: 1px;")
        self.l1.setText("<font color=#ffffff >Parameters</font> </a>")
        if USE_FULLSCREEN:
            self.l1.setFixedHeight(50)
        # else:
        #    self.l1.setFixedHeight(15)
        self.gridLayout.addWidget(self.l1, 4, 1, 1, 4)

        self.gridLayout.addWidget(QtWidgets.QLabel("Difference Factor:"), 5, 1)
        self.validFactor = QtGui.QDoubleValidator(
            0.5, 2, 3
        )  # allow exponential notation
        self.tbox_diff_factor = QtWidgets.QLineEdit()
        self.tbox_diff_factor.setValidator(self.validFactor)
        self.tbox_diff_factor.setFixedWidth(75)
        self.gridLayout.addWidget(self.tbox_diff_factor, 5, 2)
        self.btn_diff_factor = QtWidgets.QPushButton("Set/Reload")
        self.btn_diff_factor.pressed.connect(self.set_new_diff_factor)
        self.gridLayout.addWidget(self.btn_diff_factor, 5, 3)

        self.gridLayout.addWidget(QtWidgets.QLabel("Channel Thickness:"), 6, 1)
        self.validThickness = QtGui.QDoubleValidator(
            0, 1, 3
        )  # allow exponential notation
        self.tbox_ch_thick = QtWidgets.QLineEdit()
        self.tbox_ch_thick.setValidator(self.validThickness)
        self.tbox_ch_thick.setFixedWidth(75)
        self.tbox_ch_thick.setText(str(Constants.channel_thickness))
        self.tbox_ch_thick.textEdited.connect(self.set_new_ch_thick)
        self.gridLayout.addWidget(self.tbox_ch_thick, 6, 2)
        self.h0 = QtWidgets.QLabel()
        self.h0.setAlignment(QtCore.Qt.AlignCenter)
        self.h0.setText("<u>?</u>")
        self.h0.setToolTip(
            "<b>Changes here apply to this session ONLY</b> Modify 'constants.py' to make a constant change value forever."
        )
        self.gridLayout.addWidget(self.h0, 6, 3)

        self.gridLayout.addWidget(QtWidgets.QLabel("Custom POIs:"), 7, 1)
        self.custom_poi_text = QtWidgets.QLineEdit()
        self.custom_poi_text.setFixedWidth(250)
        self.custom_poi_text.editingFinished.connect(self.update_custom_pois)
        self.gridLayout.addWidget(self.custom_poi_text, 7, 2, 1, 3)

        # Options ------------------------------------------------------
        self.l2 = QtWidgets.QLabel()
        self.l2.setStyleSheet("background: #008EC0; padding: 1px;")
        self.l2.setText("<font color=#ffffff >Options</font> </a>")
        if USE_FULLSCREEN:
            self.l2.setFixedHeight(50)
        # else:
        #    self.l2.setFixedHeight(15)
        self.gridLayout.addWidget(self.l2, 1, 5, 1, 3)

        self.option_remove_dups = QtWidgets.QCheckBox(
            "Remove duplicate analysis output files"
        )
        self.option_remove_dups.setChecked(True)
        self.gridLayout.addWidget(self.option_remove_dups, 2, 5, 1, 3)
        # self.correct_drop_effect = QtWidgets.QCheckBox(
        #     "Apply drop effect vectors")
        # # per issue #26, disable by default
        # self.correct_drop_effect.setChecked(False)
        # self.correct_drop_effect.clicked.connect(self.change_drop_effect)
        # self.gridLayout.addWidget(self.correct_drop_effect, 3, 5, 1, 3)

        # Add the checkbox and call-backs for using the curve-optimizer utility.
        self.difference_factor_optimizer_checkbox = QtWidgets.QCheckBox(
            "Auto-Calculate \"Difference Factor\"")

        self.difference_factor_optimizer_checkbox.setChecked(False)
        self.difference_factor_optimizer_checkbox.clicked.connect(
            self.use_difference_factor_optimizer)
        self.gridLayout.addWidget(
            self.difference_factor_optimizer_checkbox, 3, 5, 1, 3)

        self.drop_effect_cancelation_checkbox = QtWidgets.QCheckBox(
            "Drop effect correction")

        self.drop_effect_cancelation_checkbox.setChecked(False)
        self.drop_effect_cancelation_checkbox.clicked.connect(
            self.use_drop_effect_cancelation)
        self.gridLayout.addWidget(
            self.drop_effect_cancelation_checkbox, 4, 5, 1, 3)

        # Per Zehra, for v2.6r53: Hide these corrections (not working)
        self.difference_factor_optimizer_checkbox.setVisible(False)
        self.drop_effect_cancelation_checkbox.setVisible(False)

        self.advancedwidget = QtWidgets.QWidget()
        self.advancedwidget.setWindowFlags(
            QtCore.Qt.Dialog | QtCore.Qt.WindowStaysOnTopHint
        )
        self.advancedwidget.setWhatsThis(
            "These settings are for Advanced Users ONLY!")
        warningWidget = QtWidgets.QLabel(
            f"WARNING: {self.advancedwidget.whatsThis()}")
        warningWidget.setStyleSheet(
            "background: #FF6600; padding: 1px; font-weight: bold;"
        )
        warningLayout = QtWidgets.QVBoxLayout()
        warningLayout.addWidget(warningWidget)
        warningLayout.addLayout(self.gridLayout)
        self.advancedwidget.setLayout(warningLayout)
        icon_path = os.path.join(
            Architecture.get_path(), "QATCH/icons/advanced.png")
        self.advancedwidget.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        self.advancedwidget.setWindowTitle("Advanced Settings")

        # Create dot buttons to skip directly to a particular step
        widget_h4 = QtWidgets.QWidget()
        layout_v4 = QtWidgets.QVBoxLayout()
        layout_h4 = QtWidgets.QHBoxLayout()
        layout_v4.setContentsMargins(0, 0, 0, 0)
        layout_h4.setContentsMargins(0, 0, 0, 0)

        self.dot1 = QtWidgets.QLabel("\u2b24")  # load pixmap
        self.dot2 = QtWidgets.QLabel("\u25ef")  # load pixmap
        self.dot3 = QtWidgets.QLabel("\u25ef")  # load pixmap
        self.dot4 = QtWidgets.QLabel("\u25ef")  # load pixmap
        self.dot5 = QtWidgets.QLabel("\u25ef")  # load pixmap
        self.dot6 = QtWidgets.QLabel("\u25ef")  # load pixmap
        self.dot7 = QtWidgets.QLabel("\u25ef")  # load pixmap
        self.dot8 = QtWidgets.QLabel("\u25ef")  # load pixmap
        self.dot9 = QtWidgets.QLabel("\u25ef")  # load pixmap
        self.dot10 = QtWidgets.QLabel("\u2b24")  # ('\u25fb') # load pixmap
        self.addDotStepHandlers()

        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setBackground("w")

        data, rows, cols = [
            {
                "A": ["", "", "", ""],
                "B": ["", "", "", ""],
                "C": ["", "", "", ""],
                "D": ["", "", "", ""],
            },
            4,
            4,
        ]
        results_table = TableView(data, rows, cols)
        results_figure = pg.PlotWidget()
        results_figure.setBackground("w")
        plot_text = pg.TextItem("", (51, 51, 51), anchor=(0.5, 0.5))
        plot_text.setHtml(
            "<span style='font-size: 10pt'><b>No Results To View</b><br/> \
                            Load a run, follow the prompts to select points,<br/> \
                            and press \"Analyze\" action to view results.</span>"
        )
        it = plot_text.textItem
        option = it.document().defaultTextOption()
        option.setAlignment(QtCore.Qt.AlignCenter)
        it.document().setDefaultTextOption(option)
        it.setTextWidth(it.boundingRect().width())
        plot_text.setPos(0.5, 0.5)
        results_figure.addItem(plot_text, ignoreBounds=True)

        self.results_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.results_split.addWidget(results_table)
        self.results_split.addWidget(results_figure)
        # self.results_split.setEnabled(False)
        # self.results_split.setSizes([1, 1])

        # self.graphStack = QtWidgets.QStackedWidget()
        self.graphStack.addWidget(self.graphWidget)
        self.graphStack.addWidget(self.results_split)
        self.graphStack.setCurrentIndex(0)

        layout_h4.addStretch()
        layout_h4.addWidget(self.dot1)
        layout_h4.addWidget(self.dot2)
        layout_h4.addWidget(self.dot3)
        layout_h4.addWidget(self.dot4)
        layout_h4.addWidget(self.dot5)
        layout_h4.addWidget(self.dot6)
        layout_h4.addWidget(self.dot7)
        layout_h4.addWidget(self.dot8)
        layout_h4.addWidget(self.dot9)
        layout_h4.addWidget(self.dot10)
        layout_h4.addStretch()
        widget_dots = QtWidgets.QWidget()
        widget_dots.setLayout(layout_h4)
        widget_dots.setStyleSheet("color: #515151;")
        layout_v4.addWidget(widget_dots)
        layout_v4.addWidget(self.graphStack)
        widget_h4.setLayout(layout_v4)
        widget_h4.setObjectName("AnalyzerFrame")
        widget_h4.setStyleSheet(
            "QWidget#AnalyzerFrame { background-color: #ffffff; }")

        self.graphWidget1 = pg.PlotWidget()
        self.graphWidget1.setBackground("w")
        self.graphWidget2 = pg.PlotWidget()
        self.graphWidget2.setBackground("w")
        self.graphWidget3 = pg.PlotWidget()
        self.graphWidget3.setBackground("w")

        layout_h2 = QtWidgets.QHBoxLayout()
        layout_h2.addWidget(self.graphWidget1)
        layout_h2.addWidget(self.graphWidget2)
        layout_h2.addWidget(self.graphWidget3)
        layout_h2.setContentsMargins(0, 0, 0, 0)
        self.lowerGraphs = QtWidgets.QWidget()
        self.lowerGraphs.setLayout(layout_h2)

        self.graph_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.graph_split.addWidget(widget_h4)
        self.graph_split.addWidget(self.lowerGraphs)
        self.graph_split.setSizes([1, 1])

        # add collapse/expand icon arrows
        self.graph_split.setHandleWidth(10)
        handle1 = self.graph_split.handle(1)
        layout_s1 = QtWidgets.QHBoxLayout()
        layout_s1.setContentsMargins(0, 0, 0, 0)
        layout_s1.addStretch()
        self.btnMovable1 = QtWidgets.QToolButton(handle1)
        self.btnMovable1.setText("=")
        self.btnMovable1.setFont(QtGui.QFont("Arial", 8))
        self.btnMovable1.clicked.connect(
            lambda: self.graph_split.setSizes([1, 1]))
        layout_s1.addWidget(self.btnMovable1)
        layout_s1.addStretch()
        handle1.setLayout(layout_s1)

        # add collapse/expand icon arrows
        self.results_split.setHandleWidth(12)
        handle2 = self.results_split.handle(1)
        layout_s2 = QtWidgets.QVBoxLayout()
        layout_s2.setContentsMargins(0, 0, 0, 0)
        layout_s2.addStretch()
        self.btnMovable2 = QtWidgets.QToolButton(handle2)
        self.btnMovable2.setText("||")
        self.btnMovable2.setFont(QtGui.QFont("Arial", 8))
        self.btnMovable2.clicked.connect(
            lambda: self.results_split.setSizes(
                self.get_results_split_auto_sizes())
        )
        layout_s2.addWidget(self.btnMovable2)
        layout_s2.addStretch()
        handle2.setLayout(layout_s2)

        self.footerText_left = QtWidgets.QLabel(
            "<b><u>Keyboard Shortcuts:</u></b> &nbsp; <b>Up/Down:</b> Zoom In/Out &nbsp; <b>Left/Right:</b> Move Point &nbsp; <b>Escape:</b> Back &nbsp; <b>Enter:</b> Next"
        )
        self.footerText_right = QtWidgets.QLabel(
            "<i>Drag markers for rough placement. &nbsp; Click bottom plots for precise placement.</i>"
        )
        self.footerText_left.setAlignment(QtCore.Qt.AlignLeft)
        self.footerText_right.setAlignment(QtCore.Qt.AlignRight)

        layout_h3 = QtWidgets.QHBoxLayout()
        layout_h3.addWidget(self.footerText_left)
        layout_h3.addWidget(self.footerText_right)

        # Add widgets to layout
        self.layout.addLayout(self.toolLayout)
        self.layout.addLayout(layout_h3)
        # self.layout.addWidget(widget_h4)
        self.layout.addWidget(self.graph_split)

        self.setLayout(self.layout)
        self.setWindowTitle("Analyze Data")

        # self.cBox_Devices.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        # self.cBox_Devices.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cBox_Runs.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        self.cBox_Runs.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToContents)
        self.cBox_Runs.setEditable(True)
        self.cBox_Runs.setEnabled(False)
        self.cBox_Runs.setEditText("No Runs Found")
        self.cBox_Runs.currentIndexChanged.connect(self.updateDev)
        self.cBox_Devices.currentIndexChanged.connect(self.updateRunOnChange)
        self.btn_Load.pressed.connect(self.loadRun)
        self.btn_Back.pressed.connect(self.goBack)
        self.btn_Next.pressed.connect(self.getPoints)
        self.sign.textEdited.connect(self.sign_edit)
        self.sign.textEdited.connect(self.text_transform)
        self.btn_Info.pressed.connect(self.getRunInfo)
        # self.graphWidget.scene().sigMouseClicked.connect(self.summaryClick)
        self.graphWidget1.scene().sigMouseClicked.connect(self.onClick)
        self.graphWidget2.scene().sigMouseClicked.connect(self.onClick)
        self.graphWidget3.scene().sigMouseClicked.connect(self.onClick)

        self.sign.setPlaceholderText("Initials")

        self.askForPOIs = True

        '''
        # create main graph summary point selection tool (initially hidden)
        self.AI_SelectTool_At = 0
        self.AI_Guess_Idxs = [0, 0, 0, 0, 0, 0]
        self.AI_Guess_Maxs = [5, 5, 5, 5, 5, 5]
        self.AI_Start_Vals = []
        self.AI_has_starting_values = False
        self.AI_moving_marker = False

        self.AI_SelectTool_Frame = QtWidgets.QWidget(self)
        self.AI_SelectTool_Layout = QtWidgets.QVBoxLayout()
        self.AI_SelectTool_Layout.setSpacing(0)
        self.AI_SelectTool_TitleBar = QtWidgets.QWidget()
        self.ai_layout_t = QtWidgets.QHBoxLayout()
        self.ai_layout_t.setSpacing(5)
        self.ai_layout_t.setContentsMargins(5, 0, 5, 0)
        self.AI_SelectTool_TitleBar.setLayout(self.ai_layout_t)
        self.ai_title = QtWidgets.QLabel("AI Point Selection Tool")
        self.ai_layout_t.addWidget(self.ai_title)
        self.ai_layout_t.addStretch()
        self.ai_close = QtWidgets.QLabel("X")
        self.ai_close.mouseReleaseEvent = self.hideSelectTool
        self.ai_layout_t.addWidget(self.ai_close)
        self.AI_SelectTool_TitleBar.setObjectName("AI_TitleBar")
        self.AI_SelectTool_TitleBar.setStyleSheet(
            "#AI_TitleBar { background: #DDDDDD; border: 1px solid black; border-bottom: 0; }"
        )
        self.AI_SelectTool_Layout.addWidget(self.AI_SelectTool_TitleBar)
        self.AI_SelectTool_Body = QtWidgets.QWidget()
        self.AI_SelectTool_Layout.addWidget(self.AI_SelectTool_Body)
        self.AI_SelectTool_Frame.setLayout(self.AI_SelectTool_Layout)
        self.AI_SelectTool_Frame.setVisible(False)
        self.layout.addChildWidget(self.AI_SelectTool_Frame)

        self.ai_layout = QtWidgets.QGridLayout()
        self.ai_layout.setSpacing(5)
        self.ai_layout.setContentsMargins(0, 0, 0, 0)
        self.ai_layout.setRowMinimumHeight(0, self.ai_layout.spacing())
        self.ai_layout.setRowMinimumHeight(5, self.ai_layout.spacing())
        self.AI_SelectTool_Body.setLayout(self.ai_layout)
        self.ai_backBtn = QtWidgets.QToolButton()
        self.ai_backBtn.setArrowType(QtCore.Qt.LeftArrow)
        self.ai_backBtn.adjustSize()
        self.ai_backBtn.clicked.connect(
            self.AI_Prev_Guess)  # (self.summaryBack)
        self.ai_layout.addWidget(self.ai_backBtn, 0, 0, 6, 1)
        self.ai_nextBtn = QtWidgets.QToolButton()
        self.ai_nextBtn.setArrowType(QtCore.Qt.RightArrow)
        self.ai_nextBtn.adjustSize()
        self.ai_nextBtn.clicked.connect(
            self.AI_Next_Guess)  # (self.summaryNext)
        self.ai_layout.addWidget(self.ai_nextBtn, 0, 3, 6, 1)
        self.ai_label = QtWidgets.QLabel("Point [Unknown]")
        self.ai_layout.addWidget(
            self.ai_label, 1, 1, 1, 2, QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.ai_score = QtWidgets.QLabel("Confidence Score: 95%")
        self.ai_layout.addWidget(
            self.ai_score, 2, 1, 1, 2, QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.ai_guess = QtWidgets.QLabel("Guess #: 1 of 5")
        self.ai_layout.addWidget(
            self.ai_guess, 3, 1, 1, 2, QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.ai_prev = QtWidgets.QPushButton("&Prev")
        self.ai_prev.setFixedWidth(50)
        self.ai_prev.clicked.connect(self.AI_Prev_Guess)
        self.ai_layout.addWidget(
            self.ai_prev, 4, 1, 1, 1, QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.ai_next = QtWidgets.QPushButton("&Next")
        self.ai_next.setFixedWidth(50)
        self.ai_next.clicked.connect(self.AI_Next_Guess)
        self.ai_layout.addWidget(
            self.ai_next, 4, 2, 1, 1, QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.AI_SelectTool_Body.setObjectName("AI_Tool")
        self.AI_SelectTool_Body.setStyleSheet(
            "#AI_Tool { background: #A9E1FA; border: 1px solid black; }"
        )
        self.ai_prev.setVisible(False)
        self.ai_next.setVisible(False)
        '''

        self.progressValue.connect(
            lambda value: self.progressBar.setValue(value))
        self.progressFormat.connect(
            lambda value: self.progressBar.setFormat(value))
        self.progressUpdate.connect(self.progressBar.repaint)
        self.progressUpdate.connect(QtCore.QCoreApplication.processEvents)

    # def hideSelectTool(self, event):
    #     self.AI_SelectTool_Frame.hide()

    def get_results_split_auto_sizes(self, setMinimumWidth=True):
        tableWidget = self.results_split.widget(0)
        full_width = self.results_split.width()
        min_width = tableWidget.verticalHeader().width() + 6  # +6 seems to be needed
        min_width += tableWidget.verticalScrollBar().width()
        for i in range(tableWidget.columnCount()):
            # seems to include gridline
            min_width += tableWidget.columnWidth(i)
            if i == 0 and setMinimumWidth:
                tableWidget.setMinimumWidth(min_width)
        setSizes = [min_width, full_width - min_width]
        return setSizes

    def update_custom_pois(self):
        new_pois = self.custom_poi_text.text()
        new_pois = (
            new_pois.replace("[", "").replace("]", "").replace(",", "")
        )  # remove array chars: '[],'
        new_pois = np.fromstring(
            new_pois, sep=" "
        ).tolist()  # convert string to numpy array and then to a list
        Log.w(f"Set Custom POIs: {new_pois}")
        for px, pm in enumerate(self.poi_markers):
            try:
                index = self.xs[int(new_pois[px])]
                if pm.value() != index:
                    Log.d(f"Moving marker {px} to position {index}")
                    self.detect_change()
                    self.poi_markers[px].setValue(index)
                    self.poi_markers[px].sigPositionChangeFinished.emit(
                        self.poi_markers[px]
                    )
                else:
                    Log.d(f"Moving marker {px} not required. Already there.")
            except Exception as e:
                Log.e(f"Moving marker {px} failed: {str(e)}")

    def showRunsFromAllDevices_clicked(self):
        self.cBox_Devices.setEnabled(
            not self.showRunsFromAllDevices.isChecked())
        self.updateRun(self.cBox_Devices.currentIndex())

    def switch_user_at_sign_time(self):
        new_username, new_initials, new_userrole = UserProfiles.change(
            UserRoles.ANALYZE
        )
        if UserProfiles.check(UserRoles(new_userrole), UserRoles.ANALYZE):
            if self.username != new_username:
                self.username = new_username
                self.initials = new_initials
                self.signedInAs.setText(self.username)
                self.signerInit.setText(f"Initials: <b>{self.initials}</b>")
                self.signature_received = False
                self.signature_required = True
                self.sign.setReadOnly(False)
                self.sign.setMaxLength(4)
                self.sign.clear()

                Log.d("User name changed. Changing sign-in user info.")
                self.parent.ControlsWin.username.setText(
                    f"User: {new_username}")
                self.parent.ControlsWin.userrole = UserRoles(new_userrole)
                self.parent.ControlsWin.signinout.setText("&Sign Out")
                self.parent.ControlsWin.ui1.tool_User.setText(new_username)
                self.parent.AnalyzeProc.tool_User.setText(new_username)
                if self.parent.ControlsWin.userrole != UserRoles.ADMIN:
                    self.parent.ControlsWin.manage.setText(
                        "&Change Password...")
            else:
                Log.d(
                    "User switched users to the same user profile. Nothing to change."
                )
            # PopUp.warning(self, Constants.app_title, "User has been switched.\n\nPlease sign now.")
        # elif new_username == None and new_initials == None and new_userrole == 0:
        else:
            if new_username == None and not UserProfiles.session_info()[0]:
                Log.d("User session invalidated. Switch users credentials incorrect.")
                self.parent.ControlsWin.username.setText("User: [NONE]")
                self.parent.ControlsWin.userrole = UserRoles.NONE
                self.parent.ControlsWin.signinout.setText("&Sign In")
                self.parent.ControlsWin.manage.setText("&Manage Users...")
                self.parent.ControlsWin.ui1.tool_User.setText("Anonymous")
                self.parent.AnalyzeProc.tool_User.setText("Anonymous")
                PopUp.warning(
                    self,
                    Constants.app_title,
                    "User has not been switched.\n\nReason: Not authenticated.",
                )
            if new_username != None and UserProfiles.session_info()[0]:
                Log.d("User name changed. Changing sign-in user info.")
                self.parent.ControlsWin.username.setText(
                    f"User: {new_username}")
                self.parent.ControlsWin.userrole = UserRoles(new_userrole)
                self.parent.ControlsWin.signinout.setText("&Sign Out")
                self.parent.ControlsWin.ui1.tool_User.setText(new_username)
                self.parent.AnalyzeProc.tool_User.setText(new_username)
                if self.parent.ControlsWin.userrole != UserRoles.ADMIN:
                    self.parent.ControlsWin.manage.setText(
                        "&Change Password...")
                PopUp.warning(
                    self,
                    Constants.app_title,
                    "User has not been switched.\n\nReason: Not authorized.",
                )

            Log.d("User did not authenticate for role to switch users.")

    def action_sort_by_name(self, obj):
        self.sort_order = 0
        self.updateRun(self.cBox_Devices.currentIndex())
        self.sort_by_name.setStyleSheet(
            "color: #0D4AAF; text-decoration: none; padding-left: 15px; font-weight: bold;"
        )
        self.sort_by_date.setStyleSheet(
            "color: #0D4AAF; text-decoration: none; padding-left: 15px;"
        )

    def action_sort_by_date(self, obj):
        self.sort_order = 1
        self.updateRun(self.cBox_Devices.currentIndex())
        self.sort_by_name.setStyleSheet(
            "color: #0D4AAF; text-decoration: none; padding-left: 15px;"
        )
        self.sort_by_date.setStyleSheet(
            "color: #0D4AAF; text-decoration: none; padding-left: 15px; font-weight: bold;"
        )

    def action_cancel(self):
        if self.hasUnsavedChanges():
            if not PopUp.question(
                self,
                Constants.app_title,
                "You have unsaved changes!\n\nAre you sure you want to cancel without saving?",
            ):
                return

        data, rows, cols = [
            {
                "A": ["", "", "", ""],
                "B": ["", "", "", ""],
                "C": ["", "", "", ""],
                "D": ["", "", "", ""],
            },
            4,
            4,
        ]
        results_table = TableView(data, rows, cols)
        results_figure = pg.PlotWidget()
        results_figure.setBackground("w")
        plot_text = pg.TextItem("", (51, 51, 51), anchor=(0.5, 0.5))
        plot_text.setHtml(
            "<span style='font-size: 10pt'><b>No Results To View</b><br/> \
                            Load a run, follow the prompts to select points,<br/> \
                            and press \"Analyze\" action to view results.</span>"
        )
        it = plot_text.textItem
        option = it.document().defaultTextOption()
        option.setAlignment(QtCore.Qt.AlignCenter)
        it.document().setDefaultTextOption(option)
        it.setTextWidth(it.boundingRect().width())
        plot_text.setPos(0.5, 0.5)
        results_figure.addItem(plot_text, ignoreBounds=True)

        self.graphStack.setCurrentIndex(1)
        # self.results_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.results_split.replaceWidget(0, results_table)
        self.results_split.replaceWidget(1, results_figure)
        # self.graphStack.setCurrentIndex(0)

        self.clear()  # calls self.enable_buttons()

    def action_back(self):
        try:
            self.step_direction = "backwards"
            self.goBack()
        except Exception as e:
            Log.e(
                f"An error occurred while moving to the prior step: {str(e)}")

            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        self.enable_buttons()

    def action_next(self):
        try:
            self.step_direction = "forwards"
            self.getPoints()
        except Exception as e:
            Log.e(f"An error occurred while moving to the next step: {str(e)}")

            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        self.enable_buttons()

    def action_modify(self):
        self.allow_modify = self.tool_Modify.isChecked()
        self.enable_buttons()

        if self.tool_Analyze.isEnabled():
            if self.allow_modify:
                self.gotoStepNum(None, 2)  # step 2: select rough points
            else:
                self.gotoStepNum(None, 9)  # summary

    def action_analyze(self):
        if self.signature_required and (
            self.unsaved_changes or self.model_run_this_load
        ):
            if self.signature_received == False and self.sign_do_not_ask.isChecked():
                Log.w(
                    f"Signing ANALYZE with initials {self.initials} (not asking again)"
                )
                self.signed_at = dt.datetime.now().isoformat()
                self.signature_received = True  # Do not ask again this session
            if not self.signature_received:
                if self.signForm.isVisible():
                    self.signForm.hide()
                self.signedInAs.setText(self.username)
                self.signerInit.setText(f"Initials: <b>{self.initials}</b>")
                screen = QtWidgets.QDesktopWidget().availableGeometry()
                left = int(
                    (screen.width() - self.signForm.sizeHint().width()) / 2) + 50
                top = (
                    int((screen.height() - self.signForm.sizeHint().height()) / 2) - 50
                )
                self.signForm.move(left, top)
                self.signForm.setVisible(True)
                self.sign.setFocus()
                return

        if self.sign_do_not_ask.isChecked():
            session_key_path = os.path.join(
                Constants.user_profiles_path, "session.key")
            if os.path.exists(session_key_path):
                with open(session_key_path, "r") as f:
                    session_key = f.readline()
                if not os.path.exists(Constants.auto_sign_key_path):
                    with open(Constants.auto_sign_key_path, "w") as f:
                        f.write(session_key)

        try:
            self.moved_markers = [False, False, False, False, False, False]
            self.enable_buttons(False, False)
            results_figure = pg.PlotWidget()
            results_figure.setBackground("w")
            self.graphStack.setCurrentIndex(
                1
            )  # 'results_split' must be shown prior to replacing
            self.results_split.replaceWidget(1, results_figure)
            self.stateStep = 6  # skip to show
            self.getPoints()  # show summary
            self.getPoints()  # show analysis
        except Exception as e:
            Log.e(
                f"An error occurred while analyzing the selected run: {str(e)}")

            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

    def action_advanced(self, obj):
        if self.advancedwidget.isVisible():
            self.advancedwidget.hide()

        try:
            poi_vals = []
            for pm in self.poi_markers:
                cur_val = pm.value()
                cur_idx = next(x for x, y in enumerate(
                    self.xs) if y >= cur_val)
                poi_vals.append(cur_idx)
            poi_vals.sort()
            self.custom_poi_text.setText(f"{poi_vals}")
        except Exception as e:
            Log.e(
                "Error: An exception occurred while pre-filling current POIs. Is a run even loaded?"
            )
            Log.e(f"Error Details: {str(e)}")

        self.advancedwidget.move(0, 0)
        self.advancedwidget.show()
        # QtWidgets.QWhatsThis.enterWhatsThisMode()
        # QtWidgets.QWhatsThis.showText(
        #     QtCore.QPoint(int(self.advancedwidget.width() / 2), int(self.advancedwidget.height() * (2/3))),
        #     self.advancedwidget.whatsThis(),
        #     self.advancedwidget)

    def enable_buttons(self, refocus=True, enable=True):
        enable_load = (
            len(self.cBox_Runs.currentText().strip()) > 0
            and self.cBox_Runs.currentText() != "No Runs Found"
        )
        enable_info = self.xml_path != None
        enable_cancel = self.xml_path != None
        enable_back = self.stateStep >= 0
        enable_next = enable_cancel and self.stateStep < 7
        enable_modify = self.xml_path != None
        enable_analyze = len(self.poi_markers) > 2
        if not enable:  # False when busy
            enable_load = enable_info = enable_cancel = enable_back = enable_next = (
                enable_modify
            ) = enable_analyze = False
        if enable_cancel and not enable_analyze:
            if not self.tool_Modify.isChecked():
                self.tool_Modify.setChecked(True)
                self.tool_Modify.clicked.emit()
                self.allow_modify = True
        if not enable_cancel:
            if self.tool_Modify.isChecked():
                self.tool_Modify.setChecked(False)
                self.tool_Modify.clicked.emit()
                self.allow_modify = False
        if not self.allow_modify:
            enable_back = enable_next = False
        self.tool_Load.setEnabled(enable_load)
        self.tBtn_Info.setEnabled(enable_info)
        self.tool_Cancel.setEnabled(enable_cancel)
        self.tool_Back.setEnabled(enable_back)
        self.tool_Next.setEnabled(enable_next)
        self.tool_Modify.setEnabled(enable_modify)
        self.tool_Analyze.setEnabled(enable_analyze)
        if refocus:
            self.graphWidget2.setFocus()

    # def change_drop_effect(self, object):
    #     if not self.correct_drop_effect.isChecked():
    #         self.tbox_diff_factor.setText(
    #             f"{Constants.default_diff_factor:1.3f}")
    #     self.set_new_diff_factor()

    def use_difference_factor_optimizer(self, object):
        """
        Adjusts the difference factor based on the state of the curve optimizer checkbox.

        If the curve optimizer checkbox is not checked, this method resets the 
        diff factor to the default value specified in `Constants.default_diff_factor`. 
        It then updates the new diff factor value by calling `self.set_new_diff_factor()`.

        Args:
            object (QWidget): The widget or object interacting with this method. Typically, 
                this could represent the checkbox or related UI component triggering the event.
        """
        if not self.difference_factor_optimizer_checkbox.isChecked():
            self.tbox_diff_factor.setText(
                f"{Constants.default_diff_factor:1.3f}")
        self.set_new_diff_factor()

    def use_drop_effect_interpolation(self, object):
        try:
            self.action_cancel()  # ask if they mean it if there are unsaved changes
            if not self.hasUnsavedChanges():  # only proceed if they say yes
                try:
                    self.diff_factor = round(
                        float(self.tbox_diff_factor.text()), 3)
                    Log.d(f"Difference Factor = {self.diff_factor}")
                except:
                    if hasattr(self, "diff_factor"):
                        del (
                            self.diff_factor
                        )  # unset to revert to default auto-calc value
                        Log.d("Difference Factor deleted")
                self.loadRun()  # refresh plots to show new diff factor
        except:
            Log.e("Failed to set new difference factor!")

    def use_drop_effect_cancelation(self, object):
        try:
            self.action_cancel()  # ask if they mean it if there are unsaved changes
            if not self.hasUnsavedChanges():  # only proceed if they say yes
                try:
                    self.diff_factor = round(
                        float(self.tbox_diff_factor.text()), 3)
                    Log.d(f"Difference Factor = {self.diff_factor}")
                except:
                    if hasattr(self, "diff_factor"):
                        del (
                            self.diff_factor
                        )  # unset to revert to default auto-calc value
                        Log.d("Difference Factor deleted")
                self.loadRun()  # refresh plots to show new diff factor
        except:
            Log.e("Failed to set new difference factor!")

    def set_new_diff_factor(self):
        """
        Validates and sets a new difference factor based on user input.

        This method checks if the input in `tbox_diff_factor` is valid and within 
        the acceptable range defined by `self.validFactor`. If valid, it confirms 
        any unsaved changes before proceeding to update the `diff_factor` with the 
        provided input. If the input is invalid, it logs an error message. After 
        updating the difference factor, it refreshes the plots by calling `self.loadRun()`.

        Raises an error if the process fails.

        Behavior:
            - If `tbox_diff_factor` input is invalid:
                - Logs an error message.
                - Exits without making changes.
            - If there are unsaved changes:
                - Calls `self.action_cancel()` to confirm changes.
            - If confirmed:
                - Updates `diff_factor` with the validated input value.
                - If input parsing fails, reverts to the default auto-calculated value.
                - Logs the updated or reverted state.
            - Refreshes the plots to reflect the new difference factor.

        Exceptions:
            Logs an error message if setting the new difference factor fails.

        """
        try:
            if not self.tbox_diff_factor.hasAcceptableInput():
                Log.e(
                    "Input Error: Difference Factor must be between {} and {}.".format(
                        self.validFactor.bottom(), self.validFactor.top()
                    )
                )
                return

            self.action_cancel()  # ask if they mean it if there are unsaved changes
            if not self.hasUnsavedChanges():  # only proceed if they say yes
                try:
                    self.diff_factor = round(
                        float(self.tbox_diff_factor.text()), 3)
                    Log.d(f"Difference Factor = {self.diff_factor}")
                except:
                    if hasattr(self, "diff_factor"):
                        del (
                            self.diff_factor
                        )  # unset to revert to default auto-calc value
                        Log.d("Difference Factor deleted")
                self.loadRun()  # refresh plots to show new diff factor
        except:
            Log.e("Failed to set new difference factor!")

    def set_new_ch_thick(self):
        try:
            if not self.tbox_ch_thick.hasAcceptableInput():
                Log.e(
                    "Input Error: Channel Thickness must be between {} and {}.".format(
                        self.validThickness.bottom(), self.validThickness.top()
                    )
                )
                return

            Constants.channel_thickness = float(self.tbox_ch_thick.text())
            Log.d(f"Channel thickness = {Constants.channel_thickness}")
        except:
            Log.e("Failed to set new channel thickness!")

    def _update_progress_value(self, value=0, status=None):
        pct = self.progressBar.value()
        if status != None:
            self.progressBar.setValue(value)
            self.progressBar.setFormat(status)
        elif pct == 0 and self.graphStack.currentIndex() == 0:
            self.progressBar.setFormat("Progress: Not Started")
        elif pct == 100:
            self.progressBar.setFormat("Progress: Finished")
        else:
            if self.analyzer_task.isRunning():
                pass  # see _update_analyze_progress() # self.progressBar.setFormat("Progress: %p%")
            elif (
                self.graphStack.currentIndex() == 1
                and self.analyze_work.exitCode() == False
            ):
                self.progressBar.setFormat(
                    "Status: Exception during Analyze Task! (See Console for details)"
                )
                self.progress_value_steps = []
            else:
                self.progressBar.setFormat("Loading...")
        self.progressBar.repaint()

    def _update_analyze_progress(self, value, status):
        if not hasattr(self, "progress_value_scanning"):
            self.progress_value_scanning = False
        if not hasattr(self, "progress_status_step"):
            self.progress_status_step = {}
        if not value > 0:
            self.progress_status_step.clear()
        start = self.progressBar.value() + 1 if value else 0
        stops = (
            min(100, value + 25) if value < 99 else value + 1
        )  # 0-98:+25(100); 99-100:+1
        self.progress_value_steps = list(
            range(start, stops, 1 if start < stops else -1)
        )
        self.raw_val = value
        if not status in self.progress_status_step:
            self.progress_status_step[value] = status
        if not self.progress_value_scanning:
            self.progress_value_scanning = True
            self._step_to_next_value()
        elif self.analyzer_task.isRunning():
            for key in list(self.progress_status_step.keys())[
                ::-1
            ]:  # iterate keys in reverse order
                # print(f"Check key {key} against value {value}")
                if value >= key:
                    status = self.progress_status_step.get(key)
                    # print(f"Setting status {status} @ value {key}...")
                    self.progressBar.setFormat(f"{status} %p%")
                    break  # stop after finding valid current status label
        self.progressBar.repaint()

    def _step_to_next_value(self):
        if True:
            # NOTE: 'speed' starts slow and ends fast (1, not 0)
            # <-- enter 'found' count when you search for "progress.emit" (including this one)
            total_steps = 9
            speed = max(
                1, int((total_steps - len(self.progress_status_step)) / 2)
            )  # larger numbers mean slower progressBar speed
            go_slow = len(self.progress_value_steps) < 25
            if len(self.progress_value_steps) == 0:
                self.progress_value_steps.append(100)
            if not self.analyzer_task.isRunning():
                # force fast
                speed = 0 if self.progress_value_steps[-1] > 99 else 1
            if self.progress_value_steps[-1] > 99:
                go_slow = False  # force fast
            value = self.progress_value_steps.pop(0)
            if (
                not self.analyzer_task.isRunning()
            ):  # value in self.progress_status_step.keys():
                keys = list(self.progress_status_step.keys())[
                    ::-1]  # in reverse order
                status = self.progress_status_step.get(
                    keys[0]
                )  # most recent label # .get(value)
                self.progressFormat.emit(str(f"{status} %p%"))
                # self.progressBar.setFormat(f"{status} %p%")
            if not self.analyzer_task.isRunning():
                if value != 100:
                    value *= 2
                    value = min(99, value)
            else:
                # progress_value_steps[-1]
                value = max(value, self.raw_val - 5)
            self.progressValue.emit(int(value))
            # self.progressBar.setValue(value)
            # Log.w(f"value={self.progressBar.value()}")
            self.progressUpdate.emit()
        if len(self.progress_value_steps):
            wait_time = 0.1 * speed if go_slow else 0.01 * speed
            # if wait_time > 0:
            #     sleep(wait_time)
            QtCore.QTimer.singleShot(
                int(1000 * wait_time), self._step_to_next_value)
        else:
            self.progress_value_scanning = False

    def hasUnsavedChanges(self):
        if hasattr(self, "unsaved_changes"):
            return self.unsaved_changes
        else:
            return False

    def rescanRuns(self):
        if self.hasUnsavedChanges():
            if not PopUp.question(
                self,
                Constants.app_title,
                "You have unsaved changes!\n\nAre you sure you want to refresh without saving?",
            ):
                return

        self.scan_for_most_recent_run = True
        self.reset()

    def reset(self):
        self.cBox_Devices.clear()
        self.cBox_Devices.addItems(self.parent.data_devices)
        # self.cBox_Devices.setFixedWidth(self.cBox_Devices.sizeHint().width())

        self.analyzer_task = QtCore.QThread()

        # find most recent device run
        if self.scan_for_most_recent_run:
            self.scan_for_most_recent_run = False
            # call as timer to allow repaint of Analyze view mode
            QtCore.QTimer.singleShot(500, self.find_most_recent_run)

        self.action_sort_by_date(None)  # dummy 'obj' passed

        self.username = None
        self.initials = None

        self.clear()

    def find_most_recent_run(self):
        device_idx = 0
        most_recent = "0 / No Date"
        self.enable_buttons(False, False)
        self._update_progress_value(1, f"Loading...")
        for idx in range(self.cBox_Devices.count()):
            try:
                self.updateRun(idx)
                most_recent_run_this_dev = self.sorted_runs[0][1]
                compare = [most_recent, most_recent_run_this_dev]
                if compare[1] == sorted(compare)[1]:
                    device_idx = idx
                    most_recent = most_recent_run_this_dev
            except:
                Log.w(
                    f"Device {self.cBox_Devices.itemText(idx)} has no available runs!"
                )
        Log.d(
            f"Most recent run detected on device {self.cBox_Devices.itemText(device_idx)} from {most_recent}."
        )

        self.progressBar.setValue(0)  # Not started
        self.cBox_Devices.setCurrentIndex(device_idx)
        self.action_sort_by_date(None)  # dummy 'obj' passed
        self.enable_buttons()

    def clear(self):
        self.text_Created.clear()
        self.graphWidget.clear()
        self.graphWidget.setTitle(None)
        self.graphWidget.showGrid(x=False, y=False)
        self.graphWidget1.clear()
        self.graphWidget1.setTitle(None)
        self.graphWidget1.showGrid(x=False, y=False)
        self.graphWidget2.clear()
        self.graphWidget2.setTitle(None)
        self.graphWidget2.showGrid(x=False, y=False)
        self.graphWidget3.clear()
        self.graphWidget3.setTitle(None)
        self.graphWidget3.showGrid(x=False, y=False)
        self.graphStack.setCurrentIndex(0)
        self.lowerGraphs.setVisible(False)
        self.btn_Back.setEnabled(False)
        self.btn_Next.setEnabled(False)
        self.sign.setReadOnly(False)
        self.signForm.setVisible(False)
        self.sign.setMaxLength(4)
        self.sign.clear()

        self.progressBar.setValue(0)  # Not started
        self.setDotStepMarkers(0)

        self.stateStep = -1
        self.poi_markers = []
        self.xml_path = None  # used to indicate whether a run is loaded
        self.unsaved_changes = False
        self.signed_at = "[NEVER]"
        self.signature_required = True  # secure assumption, set on load
        self.signature_received = False
        self.model_result = -1
        self.model_candidates = None
        self.model_engine = "None"

        self._annotate_welcome_text()
        self.check_user_info()
        self.enable_buttons()

        self.parent.viewTutorialPage([5, 6])  # analyze / prior results

    def _annotate_welcome_text(self):
        self._text5 = pg.TextItem("", (51, 51, 51), anchor=(0.5, 0.5))
        self._text5.setHtml(
            "<span style='font-size: 10pt'>Please \"Load\" a run from the menu above to perform data analysis. </span>"
        )
        self._text6 = pg.TextItem("", (51, 51, 51), anchor=(0.5, 0.5))
        self._text6.setHtml(
            "<span style='font-size: 10pt'>Prior point selections (if any) will be loaded from saved data. </span>"
        )
        self._text5.setPos(0.5, 0.55)
        self._text6.setPos(0.5, 0.45)
        self.graphWidget.addItem(self._text5, ignoreBounds=True)
        self.graphWidget.addItem(self._text6, ignoreBounds=True)
        self.graphWidget.setXRange(min=0, max=1)
        self.graphWidget.setYRange(min=0, max=1)

    def closeEvent(self, event):
        if self.unsaved_changes:
            if not PopUp.question(
                self,
                Constants.app_title,
                "You have unsaved changes!\n\nAre you sure you want to close this window?",
                False,
            ):
                event.ignore()

    def check_user_info(self):
        # get active session info, if available
        active, info = UserProfiles.session_info()
        if active:
            self.signature_required = True
            self.signature_received = False
            self.username, self.initials = info[0], info[1]
        else:
            self.signature_required = False

    def detect_change(self):
        if not self.unsaved_changes:
            Log.d("There are unsaved changes detected.")
        if self.signature_received:
            self.signature_received = False
            self.sign.setReadOnly(False)
            self.sign.setMaxLength(4)
            self.sign.clear()
        self.unsaved_changes = True

    def sign_edit(self):
        if self.sign.text().upper() == self.initials:
            sign_text = f"{self.username} ({self.sign.text().upper()})"
            self.sign.setMaxLength(len(sign_text))
            self.sign.setText(sign_text)
            self.sign.setReadOnly(True)
            self.signed_at = dt.datetime.now().isoformat()
            self.signature_received = True
            self.sign_do_not_ask.setEnabled(True)

    def text_transform(self):
        text = self.sign.text()
        if len(text) in [1, 2, 3, 4]:  # are these initials?
            # will not fire 'textEdited' signal again
            self.sign.setText(text.upper())

    '''
    def AI_Prev_Guess(self):
        min_val = 0 if not self.AI_has_starting_values else -1
        cur_val = self.AI_Guess_Idxs[self.AI_SelectTool_At]
        new_val = max(min_val, cur_val - 1)
        self.AI_Guess_Idxs[self.AI_SelectTool_At] = new_val
        self.ai_guess_write_summary_from_cache()

    def AI_Next_Guess(self):
        min_val = 0 if not self.AI_has_starting_values else -1
        max_val = self.AI_Guess_Maxs[self.AI_SelectTool_At]
        cur_val = self.AI_Guess_Idxs[self.AI_SelectTool_At]
        if cur_val < min_val:
            # manually selected point will be lost
            Log.w("Manually selected point was replaced with an AI guess!")
            # Log.w("You will need to re-select the manual point if you want it back.")
            # TODO: Offer warning dialog, and allow to abort if not wanted
        new_val = min(max_val - 1, cur_val + 1)
        self.AI_Guess_Idxs[self.AI_SelectTool_At] = new_val
        self.ai_guess_write_summary_from_cache()

    def ai_guess_write_summary_from_cache(self):
        error = False
        px = self.AI_SelectTool_At
        try:
            min_val = 0 if not self.AI_has_starting_values else -1
            cur_val = self.AI_Guess_Idxs[px]
            max_val = self.AI_Guess_Maxs[px]
            if cur_val == -1:
                if self.AI_has_starting_values:
                    marker_xs = self.AI_Start_Vals[px]
                else:
                    marker_xs = self.poi_markers[px].value()
            elif (
                self.moved_markers[px] and "manual" not in self.ai_score.text(
                ).lower()
            ):  # custom point
                self.moved_markers[px] = False
                cur_val = -1
                self.AI_Guess_Idxs[px] = -1
                marker_xs = self.poi_markers[px].value()
            else:
                try:
                    marker_xs = self.xs[self.model_candidates[px][0][cur_val]]
                except Exception as e:
                    Log.e("ERROR:", e)
                    Log.e(
                        f"Failed to update prediction for POI{px} to guess #{cur_val}"
                    )
            try:
                t_idx = next(x for x, y in enumerate(
                    self.xs) if y >= marker_xs)
                index = self.xs[int(t_idx)]
                if self.poi_markers[px].value() != index:
                    Log.d(f"Moving marker {px} to position {index}")
                    # self.detect_change() # not needed if calling 'sigPositionChangeFinished' on marker move
                    self.AI_moving_marker = True
                    self.poi_markers[px].setValue(index)
                    self.poi_markers[px].sigPositionChangeFinished.emit(
                        self.poi_markers[px]
                    )
                    self.summaryAt(
                        px
                    )  # will recall this function after moving tool to new marker location
                    self.AI_moving_marker = False
                    return
                else:
                    Log.d(f"Moving marker {px} not required. Already there.")
            except Exception as e:
                Log.e(f"Moving marker {px} failed: {str(e)}")
        except Exception as e:
            error = True
            Log.e("AI ERROR:", e)
            cur_val == -1
            min_val = -1
            max_val = 0
            marker_xs = self.poi_markers[px].value()
        if cur_val == -1:
            if self.AI_has_starting_values:
                # marker_xs = self.AI_Start_Vals[px]
                self.ai_score.setText(
                    "ERROR!" if error else "Loaded from prior run")
            else:
                self.ai_score.setText("Manually selected point")
            self.ai_score.adjustSize()
            self.ai_guess.setText(f"AI has {max_val} guesses")
            self.ai_guess.adjustSize()
        else:
            # marker_xs = self.poi_markers[px].value()
            confidence = int(100 * (self.model_candidates[px][1][cur_val]))
            self.ai_score.setText(f"Confidence Score: {confidence}%")
            self.ai_score.adjustSize()
            self.ai_guess.setText(f"Guess #: {cur_val + 1} of {max_val}")
            self.ai_guess.adjustSize()
        self.ai_label.setText(f"Point {px+1} @ {marker_xs:.1f}s")
        self.ai_label.adjustSize()
        enable_prev = cur_val > min_val
        enable_next = cur_val < max_val - 1
        self.ai_backBtn.setEnabled(enable_prev)  # was 'ai_prev'
        self.ai_nextBtn.setEnabled(enable_next)  # was 'ai_next'
        self.graphWidget2.setFocus()  # allow arrow keys to work immediately

    def summaryBack(self):
        if self.stateStep in range(1, 7):
            self.action_back()
        else:
            self.summaryAt(max(0, self.AI_SelectTool_At - 1))

    def summaryNext(self):
        if self.stateStep in range(1, 7):
            self.action_next()
        else:
            self.summaryAt(min(5, self.AI_SelectTool_At + 1))

    def summaryClick(self, event):
        if self.stateStep in range(1, 7):
            if not self.AI_SelectTool_Frame.isVisible():
                self.summaryAt(self.AI_SelectTool_At)
            else:
                Log.d("Ignoring main graph click while not in 'summary' step.")
            return
        mousePoint = None
        if self.graphWidget.sceneBoundingRect().contains(event._scenePos):
            mousePoint = self.graphWidget.getPlotItem().vb.mapSceneToView(
                event._scenePos
            )
        closest_marker = None
        if mousePoint != None and len(self.poi_markers) > 0:
            index = mousePoint.x()
            Log.d(f"Mouse click @ xs = {index}")
            # find nearest POI by X value, show popup there
            closest_delta_xs = self.poi_markers[-1].value()
            for idx, marker in enumerate(self.poi_markers):
                this_delta_xs = abs(marker.value() - index)
                if this_delta_xs < closest_delta_xs:
                    closest_marker = idx
                    closest_delta_xs = this_delta_xs
        if closest_marker != None:
            Log.d(f"Closest marker @ poi = {closest_marker}")
            if self.AI_has_starting_values and not any(self.AI_Guess_Idxs):
                self.AI_Guess_Idxs = [-1, -1, -1, -1, -1, -1]
                self.AI_Start_Vals = []
                for marker in self.poi_markers:
                    self.AI_Start_Vals.append(marker.value())
                self.AI_Start_Vals.sort()
            self.summaryAt(closest_marker)

    def summaryAt(self, idx):
        if not self.AI_SelectTool_Frame.isVisible():
            self.AI_Guess_Maxs = []
            for candidates, confidences in self.model_candidates:
                self.AI_Guess_Maxs.append(len(candidates))
        self.AI_SelectTool_At = idx
        marker_xs = self.poi_markers[idx].value()
        self.ai_guess_write_summary_from_cache()
        self.AI_SelectTool_Body.adjustSize()
        self.ai_backBtn.setFixedHeight(self.AI_SelectTool_Body.height())
        self.ai_nextBtn.setFixedHeight(self.AI_SelectTool_Body.height())
        # self.AI_SelectTool_Frame.setVisible(True) # per issue #25, keep hidden
        self.AI_SelectTool_Frame.adjustSize()
        scene_pos_x = self.graphWidget.getPlotItem().vb.mapViewToScene(
            QtCore.QPointF(marker_xs, 0)
        ).x() - (self.AI_SelectTool_Frame.width() / 2.5)
        scene_pos_y = 153 + (
            (self.graphWidget.height() - self.AI_SelectTool_Frame.height()) / 2
        )
        self.AI_SelectTool_Frame.move(int(scene_pos_x), int(scene_pos_y))
        # enable_back = (self.AI_SelectTool_At > 0) # repurposed these buttons for prev/next guess
        # self.ai_backBtn.setEnabled(enable_back)
        # enable_next = (self.AI_SelectTool_At < 5)
        # self.ai_nextBtn.setEnabled(enable_next)
    '''

    def onClick(self, event):
        ax1 = self.graphWidget1
        ax2 = self.graphWidget2
        ax3 = self.graphWidget3
        mousePoint = None
        if ax1.sceneBoundingRect().contains(event._scenePos):
            mousePoint = ax1.getPlotItem().vb.mapSceneToView(event._scenePos)
        if ax2.sceneBoundingRect().contains(event._scenePos):
            mousePoint = ax2.getPlotItem().vb.mapSceneToView(event._scenePos)
        if ax3.sceneBoundingRect().contains(event._scenePos):
            mousePoint = ax3.getPlotItem().vb.mapSceneToView(event._scenePos)
        if mousePoint != None:
            px = self.stateStep - 1
            index = mousePoint.x()
            Log.d(f"Mouse click @ xs = {index}")
            self.poi_markers[px].setValue(index)
            self.poi_markers[px].sigPositionChangeFinished.emit(
                self.poi_markers[px])

    def eventFilter(self, obj, event):
        if (
            event.type() == QtCore.QEvent.KeyPress
            and obj is self.sign
            and self.sign.hasFocus()
        ):
            if event.key() in [
                QtCore.Qt.Key_Enter,
                QtCore.Qt.Key_Return,
                QtCore.Qt.Key_Space,
            ]:
                if self.signature_received:
                    self.sign_ok.clicked.emit()
            if event.key() == QtCore.Qt.Key_Escape:
                self.sign_cancel.clicked.emit()
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        key = event.key()
        if key in [QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return, QtCore.Qt.Key_Space]:
            if self.tool_Next.isEnabled():
                self.tool_Next.clicked.emit()
            elif self.tool_Analyze.isEnabled():
                self.tool_Analyze.clicked.emit()
        if key == QtCore.Qt.Key_Escape:
            if self.tool_Back.isEnabled():
                self.tool_Back.clicked.emit()
        elif key == QtCore.Qt.Key_Left:
            self.moveCurrentMarker(-1)
        elif key == QtCore.Qt.Key_Right:
            self.moveCurrentMarker(+1)
        elif key == QtCore.Qt.Key_Up:
            self.zoomFinderPlots(0.5)
        elif key == QtCore.Qt.Key_Down:
            self.zoomFinderPlots(2.0)

    def moveCurrentMarker(self, offset):
        px = self.stateStep - 1
        # 100 steps per window
        offset *= max(1, int(self.getContextWidth()[0] / 50))
        if px in range(0, len(self.poi_markers)):
            cur_val = self.poi_markers[px].value()
            new_idx = next(x for x, y in enumerate(
                self.xs) if y >= cur_val) + offset
            if new_idx < 0:
                new_idx = 0
            if new_idx >= len(self.xs):
                new_idx = len(self.xs) - 1
            new_val = self.xs[new_idx]
            self.poi_markers[px].setValue(new_val)
            self.poi_markers[px].sigPositionChangeFinished.emit(
                self.poi_markers[px])
        else:
            pass

    def zoomFinderPlots(self, offset):
        px = self.stateStep - 1
        if px in range(0, len(self.poi_markers)):
            was_clipped = self.getContextWidth()[1]
            self.zoomLevel = float(self.zoomLevel * offset)
            is_clipped = self.getContextWidth()[1]
            if was_clipped == True and is_clipped == True:
                if self.stateStep <= 3:  # start, end of fill, post point
                    self.zoomLevel = 5 * \
                        self.getContextWidth()[0] / self.smooth_factor
                else:  # blips
                    self.zoomLevel = (
                        self.stateStep *
                        self.getContextWidth()[0] / self.smooth_factor
                    )
                Log.d(f"Adjusted initial zoom level to x{self.zoomLevel:2.2f}")
            if was_clipped == False and is_clipped == True:
                # revert to original
                self.zoomLevel = float(self.zoomLevel / offset)
                Log.w("Zoom level at edge limit. Up/down key event ignored.")
            elif self.zoomLevel < 1 / (2**5):
                self.zoomLevel = 1 / (2**5)
                Log.w("Zoom level at lower limit. Up/down key event ignored.")
            elif self.zoomLevel > 1 * (2**5):
                self.zoomLevel = 1 * (2**5)
                Log.w("Zoom level at upper limit. Up/down key event ignored.")
            self.poi_markers[px].sigPositionChangeFinished.emit(
                self.poi_markers[px])

    def setXmlPath(self, xml_path):
        Log.d(TAG, f'Setting xml filepath to: {xml_path}')
        self.xml_path = xml_path

    def updateDev(self, idx):
        self.enable_buttons(False)
        run = self.cBox_Runs.currentText()
        if len(run.strip()) == 0:
            self.cBox_Runs.setEditable(True)
            self.cBox_Runs.setEnabled(False)
            self.cBox_Runs.setEditText("No Runs Found")
            return
        self.cBox_Runs.setEditable(False)
        self.cBox_Runs.setEnabled(True)
        run = run[0: run.rfind("(") - 1]
        dev = self.run_devices.get(run)
        if dev != None:
            self.cBox_Devices.setCurrentText(dev)
        else:
            Log.w(f"Device not found for run {run}")

    def updateRunOnChange(self, idx):
        if not self.showRunsFromAllDevices.isChecked():
            self.updateRun(idx)

    def updateRun(self, idx):
        runs = FileStorage.DEV_get_logged_data_folders(
            self.cBox_Devices.itemText(idx))
        runs = [x for x in runs if not x == "_unnamed"]  # remove _unnamed
        unchecked_runs = [
            x
            for x in list(self.run_timestamps.keys())
            if x.endswith(self.cBox_Devices.itemText(idx))
        ]
        for run in runs:
            try:
                data_device = self.cBox_Devices.itemText(idx)
                data_folder = run
                data_files = FileStorage.DEV_get_logged_data_files(
                    data_device, data_folder
                )
                dict_key = f"{data_folder}:{data_device}"
                self.run_names[dict_key] = data_folder
                if self.run_timestamps.get(dict_key) == None:
                    doc = None
                    zn = os.path.join(
                        Constants.log_export_path, data_device, data_folder, "audit.zip"
                    )
                    if FileManager.file_exists(zn):
                        with pyzipper.AESZipFile(
                            zn,
                            "r",
                            compression=pyzipper.ZIP_DEFLATED,
                            allowZip64=True,
                            encryption=pyzipper.WZ_AES,
                        ) as zf:
                            # Add a protected file to the zip archive
                            try:
                                zf.testzip()
                            except:
                                zf.setpassword(
                                    hashlib.sha256(
                                        zf.comment).hexdigest().encode()
                                )
                            files = zf.namelist()
                            xml_filename = [
                                x for x in files if x.endswith(".xml")][0]
                            with zf.open(xml_filename, "r") as fh:
                                xml_str = fh.read().decode()
                            doc = minidom.parseString(xml_str)
                    else:
                        try:
                            xml_filename = [
                                x for x in data_files if x.endswith(".xml")
                            ][0]
                            xml_path = os.path.join(
                                Constants.log_export_path,
                                data_device,
                                data_folder,
                                xml_filename,
                            )
                            if os.path.exists(xml_path):
                                doc = minidom.parse(xml_path)
                        except IndexError as e:
                            Log.w(
                                f'WARNING: XML file not found in data files for run "{data_folder}"'
                            )
                            Log.w(
                                'Unable to parse "Date" without XML file. Treating as "Undated".'
                            )
                            Log.d(
                                f"Warning debug: {data_device}/{data_folder}/{data_files}"
                            )
                        except Exception as e:
                            raise e  # throw it, not an expected error condition

                    if doc != None:
                        metrics = doc.getElementsByTagName("metric")
                        for m in metrics:
                            name = m.getAttribute("name")
                            value = m.getAttribute("value")
                            if name == "start":
                                # value[0:value.find("T")]
                                captured_datetime = value
                                self.run_timestamps[dict_key] = captured_datetime
                        params = doc.getElementsByTagName("param")
                        for p in params:
                            name = p.getAttribute("name")
                            value = p.getAttribute("value")
                            if name == "run_name":
                                self.run_names[dict_key] = value
                                self.run_devices[value] = data_device
                # folder exists, but does a file still exist?
                if len(data_files) > 0:  # files exist
                    if dict_key in unchecked_runs:
                        unchecked_runs.remove(
                            dict_key
                        )  # it's been checked, and is not blank
                    if self.run_timestamps.get(dict_key) == None:
                        self.run_timestamps[dict_key] = "0 / No Date"
                else:
                    Log.w(f"Removing empty run info ({dict_key})")
                self.run_devices[run] = data_device
            except Exception as e:
                # import traceback
                # traceback.print_exc()
                Log.e(
                    f'Error getting timestamp from XML for run "{data_folder}"!')
                Log.d(f"Error message: {str(e)}")

        for dict_key in unchecked_runs:
            Log.w(f"Removing missing run info ({dict_key})")
            self.run_timestamps.pop(dict_key, None)

        sort_options = [[0, False], [1, True]]
        sort_select = self.sort_order  # 0 = by name, 1 = by date
        self.sorted_runs = list(
            sorted(
                self.run_timestamps.items(),
                key=lambda item: item[sort_options[sort_select][0]].lower(),
                reverse=sort_options[sort_select][1],
            )
        )
        runs = []
        for run in self.sorted_runs:
            run_name = run[0]
            device_name = run_name[run_name.find(":") + 1:]
            # run_name[0:run_name.find(":")]
            run_name = self.run_names[run_name]
            captured_datetime = run[1]
            if captured_datetime == "0 / No Date":
                captured_date = "Undated"
            else:
                captured_date = captured_datetime[0: captured_datetime.find(
                    "T")]
            if (
                self.showRunsFromAllDevices.isChecked()
                or self.cBox_Devices.currentText() == device_name
            ):
                # append date to run name
                runs.append(f"{run_name} ({captured_date})")

        self.cBox_Runs.clear()
        self.cBox_Runs.addItems(runs)
        w = self.cBox_Runs.sizeHint().width()
        if w < 200:
            w = 200
        self.cBox_Runs.setFixedWidth(w)
        self.sort_by_widget.setFixedWidth(self.cBox_Runs.width())

    def loadRun(self):
        self.action_cancel()  # ask them if they want to lose unsaved changes
        if self.hasUnsavedChanges():
            Log.d("User declined load action. There are unsaved changes.")
            return

        # if self.AI_SelectTool_Frame.isVisible():
        #     self.AI_SelectTool_Frame.setVisible(
        #         False
        #     )  # require re-click to show popup tool incorrect position

        try:
            if not self.QModel_modules_loaded:
                cluster_model_path = os.path.join(
                    Architecture.get_path(), "QATCH/QModel/SavedModels/cluster.joblib"
                )
                self.QModel_clusterer = QClusterer(
                    model_path=cluster_model_path)

                predict_model_path = os.path.join(
                    Architecture.get_path(),
                    "QATCH/QModel/SavedModels/QMultiType_{}.json",
                )
                self.QModel_predict_0 = QPredictor(
                    model_path=predict_model_path.format(0)
                )
                self.QModel_predict_1 = QPredictor(
                    model_path=predict_model_path.format(1)
                )
                self.QModel_predict_2 = QPredictor(
                    model_path=predict_model_path.format(2)
                )
            self.QModel_modules_loaded = True
        except Exception as e:
            Log.e("ERROR:", e)
            Log.e("Failed to load 'QModel' modules at load of run.")

        enabled, error, expires = UserProfiles.checkDevMode()
        if enabled == False and (error == True or expires != ""):
            PopUp.warning(
                self,
                "Developer Mode Expired",
                "Developer Mode has expired and these analysis results will be encrypted.\n"
                + 'An admin must renew or disable "Developer Mode" to suppress this warning.',
            )

        if (
            enabled
        ):  # DevMode enabled, check if auto-sign (do not ask again) key is active
            auto_sign_key = None
            session_key = None
            if os.path.exists(Constants.auto_sign_key_path):
                with open(Constants.auto_sign_key_path, "r") as f:
                    auto_sign_key = f.readline()
            session_key_path = os.path.join(
                Constants.user_profiles_path, "session.key")
            if os.path.exists(session_key_path):
                with open(session_key_path, "r") as f:
                    session_key = f.readline()
            if auto_sign_key == session_key and session_key != None:
                self.sign_do_not_ask.setChecked(True)
            else:
                self.sign_do_not_ask.setChecked(False)
                if os.path.exists(Constants.auto_sign_key_path):
                    os.remove(Constants.auto_sign_key_path)

            # NOTE: This check is needed for Analyze since it persists across run loads
            #       and is only init'd once per application instance (across sessions).
            # just in case, make it visible
            self.sign_do_not_ask.setVisible(True)
        else:  # DevMode may have been manually disabled since last run load
            # Force compliance by removing auto-sign feature support
            self.sign_do_not_ask.setChecked(False)  # uncheck
            self.sign_do_not_ask.setEnabled(False)  # disable
            self.sign_do_not_ask.setVisible(False)  # hide

        self.askForPOIs = True
        self.btn_Next.setText("Next")

        self._text1 = pg.TextItem("", (51, 51, 51), anchor=(0.5, 0.5))
        self._text1.setHtml(
            "<span style='font-size: 14pt'>Loading data for analysis... </span>"
        )
        self._text2 = pg.TextItem("", (51, 51, 51), anchor=(0.5, 0.5))
        self._text2.setHtml(
            "<span style='font-size: 10pt'>This operation may take a few seconds. </span>"
        )
        self._text3 = pg.TextItem("", (51, 51, 51), anchor=(0.5, 0.5))
        self._text3.setHtml(
            "<span style='font-size: 10pt'>Please be more patient with longer runs. </span>"
        )
        self._text1.setPos(0.5, 0.50)
        self._text2.setPos(0.5, 0.35)
        self._text3.setPos(0.5, 0.25)

        ax = self.graphWidget  # .plot(hour, temperature)
        ax1 = self.graphWidget1
        ax2 = self.graphWidget2
        ax3 = self.graphWidget3

        elems = [ax, ax1, ax2, ax3]
        for e in elems:
            if e != None:
                e.clear()
                e.setLimits(yMin=None, yMax=None,
                            minYRange=None, maxYRange=None)
                e.setXRange(min=0, max=1)
                e.setYRange(min=0, max=1)
            if e is ax:
                e.addItem(self._text1, ignoreBounds=True)

        try:
            self.progressBar.valueChanged.disconnect(
                self._update_progress_value)
        except:
            Log.w("Cannot disconnect non-existent method from ProgressBar.")

        self.enable_buttons(False, False)
        self._update_analyze_progress(
            0, "Reading Run Data..."
        )  # reset internal buffer with 0
        self._update_analyze_progress(
            75, "Reading Run Data..."
        )  # run up to 75% then slow, stop @ 99% until loaded

        # call as timer to allow repaint of pyqtgraph TextItems (loading)
        QtCore.QTimer.singleShot(500, self.action_load_run)

    def getFolderFromRun(self, run_string):
        idx = run_string.rfind("(")
        if idx > 0:
            folder = run_string[0: idx - 1]
        else:
            folder = run_string
        dict_key = list(self.run_names.keys())[
            list(self.run_names.values()).index(folder)
        ]
        folder = dict_key[0: dict_key.find(":")]
        # Log.w(f"folder from run = '{folder}'")
        return folder

    def action_load_run(self):
        try:
            self.moved_markers = [False, False, False, False, False, False]
            self.parent.analyze_data(
                self.cBox_Devices.currentText(),
                self.getFolderFromRun(self.cBox_Runs.currentText()),
                None,
            )
            self.enable_buttons()
        except Exception as e:
            Log.e(
                f"An error occurred while loading the selected run: {str(e)}")
            self.action_cancel()

    def goBack(self):
        self.btn_Next.setEnabled(True)
        if self.stateStep == 7:
            for marker in self.poi_markers:
                marker.setMovable(True)
        self.stateStep -= 2
        if self.stateStep < -1:
            if PopUp.question(
                self,
                "Are you sure?",
                "Any manual points will be lost if you go back to Step 1.\n\nProceed?",
            ):
                self.parent.analyze_data(
                    self.cBox_Devices.currentText(),
                    self.getFolderFromRun(self.cBox_Runs.currentText()),
                    None,
                )  # force back to step 1 of 8
            else:
                self.stateStep = 0
        else:
            self.moved_markers = [False, False, False, False, False, False]
            self.getPoints()

    def getContextWidth(self):
        clipped = False
        if self.stateStep <= 3:  # start, end of fill, post point
            ws = int(self.zoomLevel * self.smooth_factor / 2)  # context width
        else:  # blips
            ws = int(
                self.zoomLevel * self.smooth_factor * self.stateStep
            )  # context width
        if ws > len(self.xs) / 2:
            ws = int(len(self.xs) / 20)
        px = self.stateStep - 1
        if px in range(len(self.poi_markers)):
            tt1 = self.poi_markers[px].value()
        else:
            # self.poi_markers[self.AI_SelectTool_At].value()
            tt1 = self.xs[-1]
        tx1 = next(x for x, y in enumerate(self.xs) if y >= tt1)
        if tx1 - ws < 0:
            clipped = True
            ws = tx1
        elif tx1 + ws >= len(self.xs):
            clipped = True
            ws = len(self.xs) - 1 - tx1
        elif ws < 10:
            clipped = True
            ws = 10
        return [ws, clipped]

    def getPoints(self):
        self.graphStack.setCurrentIndex(0)
        self.btn_Back.setEnabled(True)
        if not self.stateStep == 7:
            self.btn_Next.setText("Next")
        self.stateStep += 1
        step_num = self.stateStep + 2
        if step_num < 3 and self.tool_Modify.isChecked():
            self.parent.viewTutorialPage(7)  # analyze (summary)
        elif step_num in range(3, 8 + 1) and self.tool_Modify.isChecked():
            tutorial_ids = [round(7 + (step_num - 2) / 10, 2)]
            if step_num > 5:
                tutorial_ids.append(7.7)
            self.parent.viewTutorialPage(
                tutorial_ids)  # analyze (precise point)
        else:  # "Modify" not checked or step_num > 8
            self.parent.viewTutorialPage([5, 6])  # analyze / prior results
        ax = self.graphWidget  # .plot(hour, temperature)
        ax1 = self.graphWidget1
        ax2 = self.graphWidget2
        ax3 = self.graphWidget3
        w123 = True if self.stateStep in range(1, 7) else False
        was_vis = ax1.isVisible()
        self.lowerGraphs.setVisible(w123)
        # if w123 and not was_vis:
        #     ax2.setFocus() # allow keyboard shortcuts left/right/up/down to work immediately
        # ax1.setVisible(w123)
        # ax2.setVisible(w123)
        # ax3.setVisible(w123)
        if self.stateStep == 0:
            self._update_progress_value(
                11 * (step_num -
                      1), f"Step {step_num} of 8: Select Rough Fill Points"
            )
            ax.setTitle(None)
            ax.setXRange(0, self.xs[-1], padding=0.05)
            ax.setYRange(
                0,
                max(
                    np.amax(self.ys_freq_fit),
                    np.amax(self.ys_fit),
                    np.amax(self.ys_diff_fit),
                ),
                padding=0.05,
            )
            self.textItem = pg.TextItem(
                "Drag any unused markers all the way to the right side of the plot.",
                color=(0, 0, 0),
                anchor=(1, 1),
                angle=270,
            )
            self.textItem.setPos(QtCore.QPointF(self.xs[-1], 0))
            ax.addItem(self.textItem)
            self.fit1.setAlpha(1, False)
            self.fit2.setAlpha(1, False)
            self.fit3.setAlpha(1, False)
            self.scat1.setAlpha(0.01, False)
            self.scat2.setAlpha(0.01, False)
            self.scat3.setAlpha(0.01, False)

            # ModelData
            poi_vals = []
            if len(self.poi_markers) != 6:
                self.model_result = -1
                self.model_candidates = None
                self.model_engine = "None"

                if Constants.QModel_predict:
                    try:
                        with secure_open(self.loaded_datapath, "r", "capture") as f:
                            fh = BytesIO(f.read())
                            label = self.QModel_clusterer.predict_label(fh)
                            fh.seek(0)
                            act_poi = [None] * 6  # no initial guesses
                            candidates = getattr(
                                self, f"QModel_predict_{label}"
                            ).predict(fh, run_type=label, act=act_poi)
                            predictions = []
                            for p, c in candidates:
                                predictions.append(
                                    p[0]
                                )  # assumes 1st point is best point
                            self.model_result = predictions
                            self.model_candidates = candidates
                            self.model_engine = "QModel"
                        if (
                            isinstance(self.model_result, list)
                            and len(self.model_result) == 6
                        ):
                            poi_vals = self.model_result
                        else:
                            self.model_result = -1  # try fallback model
                    except Exception as e:
                        Log.e(e)
                        Log.e(
                            "Error using 'QModel'... Using 'ModelData' as fallback (less accurate)."
                        )
                        # raise e # debug only
                        self.model_result = -1  # try fallback model

                start_time = min(
                    self.poi_markers[0].value(), self.poi_markers[-1].value()
                )
                start_time = next(x for x, y in enumerate(
                    self.xs) if y >= start_time)
                stop_time = max(
                    self.poi_markers[0].value(), self.poi_markers[-1].value()
                )
                stop_time = next(x for x, y in enumerate(
                    self.xs) if y >= stop_time)

                if self.model_result == -1 and Constants.ModelData_predict:
                    try:
                        model_starting_points = [
                            start_time,
                            None,
                            None,
                            None,
                            None,
                            stop_time,
                        ]
                        self.model_result = self.dataModel.IdentifyPoints(
                            data_path=self.loaded_datapath,
                            times=self.data_time,
                            freq=self.data_freq,
                            diss=self.data_diss,
                            start_at=model_starting_points,
                        )
                        self.model_engine = "ModelData"
                        if isinstance(self.model_result, list):
                            poi_vals.clear()
                            # show point with highest confidence for each:
                            self.model_select = []
                            self.model_candidates = []
                            for point in self.model_result:
                                self.model_select.append(0)
                                if isinstance(point, list):
                                    self.model_candidates.append(point)
                                    select_point = point[self.model_select[-1]]
                                    select_index = select_point[0]
                                    select_confidence = select_point[1]
                                    poi_vals.append(select_index)
                                else:
                                    self.model_candidates.append([point])
                                    poi_vals.append(point)
                        elif self.model_result == -1:
                            Log.w(
                                "Model failed to auto-calculate points of interest for this run!"
                            )
                            pass
                        else:
                            Log.e(
                                "Model encountered an unexpected response. Please manually select points."
                            )
                            pass
                    except:
                        limit = None
                        t, v, tb = sys.exc_info()
                        from traceback import format_tb

                        a_list = ["Traceback (most recent call last):"]
                        a_list = a_list + format_tb(tb, limit)
                        a_list.append(f"{t.__name__}: {str(v)}")
                        for line in a_list:
                            Log.e(line)

                try:  # if isinstance(self.model_result, list):
                    poi2_time = self.xs[poi_vals[1]]  # end of fill
                    poi3_time = self.xs[poi_vals[2]]  # post
                    poi4_time = self.xs[poi_vals[3]]  # blip1
                    poi5_time = self.xs[poi_vals[4]]  # blip2
                except:  # else:
                    Log.e(
                        "Model returned insufficient points. Please manually select points."
                    )
                    fill_time = self.xs[stop_time] - self.xs[start_time]
                    poi2_time = self.xs[start_time] + \
                        (fill_time * 0.05)  # end of fill
                    poi3_time = self.xs[start_time] + \
                        (fill_time * 0.10)  # post
                    poi4_time = self.xs[start_time] + \
                        (fill_time * 0.25)  # blip1
                    poi5_time = self.xs[start_time] + \
                        (fill_time * 0.50)  # blip2

                self.moved_markers = [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ]  # model can adjust all points on next step

            else:  # all points already set
                for pm in self.poi_markers:
                    cur_val = pm.value()
                    cur_idx = next(x for x, y in enumerate(
                        self.xs) if y >= cur_val)
                    poi_vals.append(cur_idx)

            if len(self.poi_markers) != 6:
                self.detect_change()

                for pt in [poi2_time, poi3_time, poi4_time, poi5_time]:
                    poi_marker = pg.InfiniteLine(
                        pos=pt,
                        angle=90,
                        pen="b",
                        bounds=[self.xs[0], self.xs[-1]],
                        movable=True,
                    )
                    ax.addItem(poi_marker)
                    poi_marker.sigPositionChangeFinished.connect(
                        self.markerMoveFinished
                    )
                    self.poi_markers.insert(-1, poi_marker)
            for marker in self.poi_markers:
                marker.setMovable(True)
                marker.setPen(color="blue")
                marker.addMarker("<|>")
            # self.AI_SelectTool_Frame.setVisible(False)  # Hide AI Tool
        elif self.stateStep in range(1, 7):
            if self.stateStep + 2 == 3:  # stateStep 1 = Step 3 of 8
                # sort poi_markers by time, in case the user messed up the order moving things around manually in Step 2
                out_of_order = False
                for i in range(1, len(self.poi_markers)):
                    if self.poi_markers[i - 1].value() > self.poi_markers[i].value():
                        Log.d("Detected POI markers are out-of-order... sorting...")
                        out_of_order = True
                        break  # no need to keep searching, the order is wrong, so fix it
                if out_of_order:
                    try:
                        poi_vals = []
                        for pm in self.poi_markers:
                            cur_val = pm.value()
                            cur_idx = next(
                                x for x, y in enumerate(self.xs) if y >= cur_val
                            )
                            poi_vals.append(cur_idx)
                        poi_vals.sort()
                        self.custom_poi_text.setText(f"{poi_vals}")
                        self.update_custom_pois()  # write POI markers in correct order
                    except Exception as e:
                        Log.e(
                            "Error: An exception occurred while sorting POI markers.")
                        Log.e(f"Error Details: {str(e)}")

                poi_vals = []
                for pm in self.poi_markers:  # already sorted
                    cur_val = pm.value()
                    cur_idx = next(x for x, y in enumerate(
                        self.xs) if y >= cur_val)
                    poi_vals.append(cur_idx)

                if self.model_engine == "ModelData" and Constants.ModelData_predict:
                    try:
                        # Run Model again, to get an initial automatic fine tuning of points prior to user input
                        model_starting_points = (
                            poi_vals.copy()
                        )  # NOTE: len(poi_vals) must equal 6
                        self.model_result = self.dataModel.IdentifyPoints(
                            data_path=self.loaded_datapath,
                            times=self.data_time,
                            freq=self.data_freq,
                            diss=self.data_diss,
                            start_at=model_starting_points,
                        )
                        self.model_engine = "ModelData"
                        if isinstance(self.model_result, list):
                            poi_vals.clear()
                            # show point with highest confidence for each:
                            self.model_select = []
                            self.model_candidates = []
                            for idx, point in enumerate(self.model_result):
                                if isinstance(point, list):
                                    self.model_candidates.append(point)
                                else:
                                    self.model_candidates.append([point])
                                self.model_select.append(0)
                                if self.moved_markers[idx] == False:
                                    poi_vals.append(model_starting_points[idx])
                                else:
                                    self.moved_markers[idx] = False
                                    if isinstance(point, list):
                                        select_point = point[self.model_select[-1]]
                                        select_index = select_point[0]
                                        select_confidence = select_point[1]
                                        poi_vals.append(select_index)
                                    else:
                                        poi_vals.append(point)
                            # Track which markers have been moved and only update model for those points, otherwise take starting point
                            if poi_vals != model_starting_points:
                                # Updating custom POIs also re-writes the POI markers
                                self.custom_poi_text.setText(f"{poi_vals}")
                                self.update_custom_pois()  # write POI markers in correct order
                                self.moved_markers = [
                                    False,
                                    False,
                                    False,
                                    False,
                                    False,
                                    False,
                                ]
                        elif self.model_result == -1:
                            Log.w(
                                "Model failed to auto-calculate points of interest for this run!"
                            )
                            pass
                        else:
                            Log.e(
                                "Model encountered an unexpected response. Please manually select points."
                            )
                            pass
                    except:
                        Log.e(
                            "An error occurred while running the model and organizing markers."
                        )

                    # sort poi_markers one more time, just in case model returned out-of-order points (which should never happen)
                    out_of_order = False
                    for i in range(1, len(self.poi_markers)):
                        if (
                            self.poi_markers[i - 1].value()
                            > self.poi_markers[i].value()
                        ):
                            Log.d(
                                "Detected POI markers are out-of-order... sorting...")
                            out_of_order = True
                            break  # no need to keep searching, the order is wrong, so fix it
                    if out_of_order:
                        try:
                            poi_vals = []
                            for pm in self.poi_markers:
                                cur_val = pm.value()
                                cur_idx = next(
                                    x for x, y in enumerate(self.xs) if y >= cur_val
                                )
                                poi_vals.append(cur_idx)
                            poi_vals.sort()
                            self.custom_poi_text.setText(f"{poi_vals}")
                            self.update_custom_pois()  # write POI markers in correct order
                        except Exception as e:
                            Log.e(
                                "Error: An exception occurred while sorting POI markers."
                            )
                            Log.e(f"Error Details: {str(e)}")

                else:  # self.model_engine != "ModelData":
                    # do nothing here if "QModel" or "None"
                    pass

            # in stateStep 2 thru 6 (Steps 4 thru 8 of 8)
            elif self.stateStep != 7:
                if (
                    self.poi_markers[self.stateStep - 1].value()
                    < self.poi_markers[self.stateStep - 2].value()
                ):
                    cur_val = self.poi_markers[self.stateStep - 2].value()
                    cur_idx = next(x for x, y in enumerate(
                        self.xs) if y >= cur_val)
                    self.poi_markers[self.stateStep - 1].setValue(
                        self.xs[int(cur_idx + 2)]
                    )
            self.zoomLevel = 1  # reset default zoom level for each point
            show_fits = 1.0 if self.stateStep >= 4 else 0.0
            show_scat = 0.1 if self.stateStep >= 4 else 1.0
            pad = 0.05 if self.stateStep >= 4 else 0.05
            self.fit_1.setAlpha(show_fits, False)
            self.fit_2.setAlpha(show_fits, False)
            self.fit_3.setAlpha(show_fits, False)
            self.scat_1.setAlpha(show_scat, False)
            self.scat_2.setAlpha(show_scat, False)
            self.scat_3.setAlpha(show_scat, False)
            self._update_progress_value(
                11 * (step_num - 1),
                f"Step {step_num} of 8: Select Precise Fill Point {self.stateStep}",
            )
            ax.setTitle(None)
            px = self.stateStep - 1
            tt0 = self.poi_markers[0].value()
            tx0 = next(x for x, y in enumerate(self.xs) if y >= tt0)
            tt1 = self.poi_markers[px].value()
            tx1 = next(x for x, y in enumerate(self.xs) if y >= tt1)
            tt2 = self.poi_markers[-1].value()
            tx2 = next(x for x, y in enumerate(self.xs) if y >= tt2)
            ws = self.getContextWidth()[0]
            ax.setXRange(self.xs[tx0], self.xs[tx2], padding=0.12)
            ax1.setXRange(self.xs[tx1 - ws], self.xs[tx1 + ws], padding=0)
            ax2.setXRange(self.xs[tx1 - ws], self.xs[tx1 + ws], padding=0)
            ax3.setXRange(self.xs[tx1 - ws], self.xs[tx1 + ws], padding=0)
            if False:  # diff_only
                mn = np.amin(self.ys_diff[tx0:tx2])
                mx = np.amax(self.ys_diff[tx0:tx2])
            else:
                mn = min(
                    np.amin(self.ys_freq_fit[tx0:tx2]),
                    np.amin(self.ys_fit[tx0:tx2]),
                    np.amin(self.ys_diff_fit[tx0:tx2]),
                )
                mx = max(
                    np.amax(self.ys_freq_fit[tx0:tx2]),
                    np.amax(self.ys_fit[tx0:tx2]),
                    np.amax(self.ys_diff_fit[tx0:tx2]),
                )
            ax.setYRange(mn, mx, padding=pad)
            if self.stateStep >= 4:
                try:
                    ax1.setYRange(
                        np.min(self.ys_freq_fit[tx1 - ws: tx1 + ws]),
                        np.max(self.ys_freq_fit[tx1 - ws: tx1 + ws]),
                        padding=pad,
                    )
                    ax2.setYRange(
                        np.min(self.ys_diff_fit[tx1 - ws: tx1 + ws]),
                        np.max(self.ys_diff_fit[tx1 - ws: tx1 + ws]),
                        padding=pad,
                    )
                    ax3.setYRange(
                        np.min(self.ys_fit[tx1 - ws: tx1 + ws]),
                        np.max(self.ys_fit[tx1 - ws: tx1 + ws]),
                        padding=pad,
                    )
                except ValueError as e:
                    Log.d(
                        "Skipping to next step, due to missing channel in data selection (represented by ValueError exception below):"
                    )
                    limit = None
                    t, v, tb = sys.exc_info()
                    from traceback import format_tb

                    a_list = ["Traceback (most recent call last):"]
                    a_list = a_list + format_tb(tb, limit)
                    a_list.append(f"{t.__name__}: {str(v)}")
                    for line in a_list:
                        Log.d(line)
                    # skip to next view
                    Log.w(
                        f"Skipping Step {self.stateStep+2}... User indicated this point is missing from the dataset in Step 2."
                    )
                    if self.step_direction == "backwards":
                        self.action_back()  # repeat last action
                    else:
                        self.action_next()  # repeat last action
                    return  # do not execute remainder of this function, let the above nested 'action_next' call supercede
                except Exception as e:
                    Log.e(
                        f"An error occurred while moving to the next step: {str(e)}")
                    limit = None
                    t, v, tb = sys.exc_info()
                    from traceback import format_tb

                    a_list = ["Traceback (most recent call last):"]
                    a_list = a_list + format_tb(tb, limit)
                    a_list.append(f"{t.__name__}: {str(v)}")
                    for line in a_list:
                        Log.e(line)

                pos1 = np.column_stack((self.xs[tx1], self.ys_freq_fit[tx1]))
                pos2 = np.column_stack((self.xs[tx1], self.ys_diff_fit[tx1]))
                pos3 = np.column_stack((self.xs[tx1], self.ys_fit[tx1]))
            else:
                ax1.setYRange(
                    np.min(self.ys_freq[tx1 - ws: tx1 + ws]),
                    np.max(self.ys_freq[tx1 - ws: tx1 + ws]),
                    padding=pad,
                )
                ax2.setYRange(
                    np.min(self.ys_diff[tx1 - ws: tx1 + ws]),
                    np.max(self.ys_diff[tx1 - ws: tx1 + ws]),
                    padding=pad,
                )
                ax3.setYRange(
                    np.min(self.ys[tx1 - ws: tx1 + ws]),
                    np.max(self.ys[tx1 - ws: tx1 + ws]),
                    padding=pad,
                )
                pos1 = np.column_stack((self.xs[tx1], self.ys_freq[tx1]))
                pos2 = np.column_stack((self.xs[tx1], self.ys_diff[tx1]))
                pos3 = np.column_stack((self.xs[tx1], self.ys[tx1]))
            self.star1.setData(pos=pos1)
            self.star2.setData(pos=pos2)
            self.star3.setData(pos=pos3)
            gstar_idxs = []
            for idx, marker in enumerate(self.poi_markers):
                if (
                    idx == px - 1
                ):  # check last point, move this marker if it's out of time sequence from last one
                    if (
                        marker.value() >= self.poi_markers[px].value()
                    ):  # last marker time greater than this marker
                        t_idx = next(
                            x for x, y in enumerate(self.xs) if y >= marker.value()
                        )
                        marker.setValue(self.xs[t_idx + 3])
                if idx != px:
                    t_idx = next(
                        x for x, y in enumerate(self.xs) if y >= marker.value()
                    )
                    gstar_idxs.append(t_idx)
                marker.setMovable(idx == px)  # only current marker is movable
                marker.setPen(color=("blue" if idx == px else "blue"))
                marker.addMarker("<|>") if idx == px else marker.clearMarkers()
            if self.stateStep >= 4:
                pos1 = np.column_stack(
                    (self.xs[gstar_idxs], self.ys_freq_fit[gstar_idxs])
                )
                pos2 = np.column_stack(
                    (self.xs[gstar_idxs], self.ys_diff_fit[gstar_idxs])
                )
                pos3 = np.column_stack(
                    (self.xs[gstar_idxs], self.ys_fit[gstar_idxs]))
            else:
                pos1 = np.column_stack(
                    (self.xs[gstar_idxs], self.ys_freq[gstar_idxs]))
                pos2 = np.column_stack(
                    (self.xs[gstar_idxs], self.ys_diff[gstar_idxs]))
                pos3 = np.column_stack(
                    (self.xs[gstar_idxs], self.ys[gstar_idxs]))
            self.gstars1.setData(pos=pos1)
            self.gstars2.setData(pos=pos2)
            self.gstars3.setData(pos=pos3)
            # # Show AI Tool on current point marker after everything settles:
            # QtCore.QTimer.singleShot(
            #     1, lambda: self.summaryAt(max(0, min(5, self.stateStep - 1)))
            # )
        elif self.stateStep == 7:
            self._update_progress_value(
                100,
                f'Summary: Press "Analyze" to compute results for these selected points',
            )
            # ax.setTitle(f"Summary: All Selected Points of Interest")
            self.fit1.setAlpha(1, False)
            self.fit2.setAlpha(1, False)
            self.fit3.setAlpha(1, False)
            self.scat1.setAlpha(0.01, False)
            self.scat2.setAlpha(0.01, False)
            self.scat3.setAlpha(0.01, False)
            for marker in self.poi_markers:
                marker.setMovable(False)
                marker.setPen(color="blue")
                marker.clearMarkers()
            tt0 = self.poi_markers[0].value()
            tx0 = next(x for x, y in enumerate(self.xs) if y >= tt0)
            tt2 = self.poi_markers[-1].value()
            tx2 = next(x for x, y in enumerate(self.xs) if y >= tt2)
            ax.setXRange(self.xs[tx0], self.xs[tx2], padding=0.12)
            mn = min(
                np.amin(self.ys_freq_fit[tx0:tx2]),
                np.amin(self.ys_fit[tx0:tx2]),
                np.amin(self.ys_diff_fit[tx0:tx2]),
            )
            mx = max(
                np.amax(self.ys_freq_fit[tx0:tx2]),
                np.amax(self.ys_fit[tx0:tx2]),
                np.amax(self.ys_diff_fit[tx0:tx2]),
            )
            ax.setYRange(mn, mx, padding=0.05)
            for i, marker in enumerate(self.poi_markers):
                Log.d(f"Marker {i} = ", marker.value())
            self.btn_Next.setText("Analyze")
            # self.AI_SelectTool_Frame.setVisible(False)  # Hide AI Tool
        else:
            self.stateStep = 8
            if self.unsaved_changes:
                if self.signature_required and not self.signature_received:
                    Log.e(
                        f"Input Error: Initials do not match current user info ({self.initials})"
                    )
                    self.sign.setFocus()
                    return
            self.btn_Back.setEnabled(True)
            self.btn_Next.setEnabled(False)
            poi_vals = []
            for marker in self.poi_markers:
                t_idx = next(x for x, y in enumerate(
                    self.xs) if y >= marker.value())
                poi_vals.append(t_idx)
            poi_vals.sort()
            if self.unsaved_changes:
                Log.d("Storing new <points> in XML file")
                self.unsaved_changes = False
                if self.signature_required:
                    self.appendAuditToXml()
                self.appendPointsToXml(poi_vals)
            # self.showAnalysis(poi_vals)
            # self.analyzer_task = threading.Thread(target=self.showAnalysis, args=(poi_vals,))
            # self.analyzer_task.start()
            allow_start = True
            if hasattr(self, "analyze_work"):
                if self.analyze_work.is_running():
                    Log.w(
                        "Double-click detected on Analyze action. Skipping duplicate action."
                    )
                    allow_start = False
            if allow_start:
                self._update_progress_value(1, "Status: Starting...")
                self.graphStack.setCurrentIndex(1)
                self.analyzer_task = QtCore.QThread()
                self.analyze_work = AnalyzerWorker(
                    self,  # pass in parent
                    self.loaded_datapath,
                    self.xml_path,
                    poi_vals,
                    self.diff_factor if hasattr(self, "diff_factor") else None,
                )
                self.analyzer_task.started.connect(self.analyze_work.run)
                self.analyze_work.finished.connect(self.analyzer_task.quit)
                self.analyze_work.progress.connect(
                    self._update_analyze_progress)
                self.analyze_work.finished.connect(self._update_progress_value)
                self.analyze_work.finished.connect(self.enable_buttons)
                self.analyzer_task.start()
        self.setDotStepMarkers(step_num)

    def addDotStepHandlers(self):
        dots = [
            self.dot1,
            self.dot2,
            self.dot3,
            self.dot4,
            self.dot5,
            self.dot6,
            self.dot7,
            self.dot8,
            self.dot9,
            self.dot10,
        ]
        for i, d in enumerate(dots):
            # d.setObjectName(str(i))
            d.mousePressEvent = self.gotoStepNum
            # d.mousePressEvent = lambda: event, step_num=i: self.gotoStepNum(step_num)

    def setDotStepMarkers(self, step_num):
        dots = [
            self.dot1,
            self.dot2,
            self.dot3,
            self.dot4,
            self.dot5,
            self.dot6,
            self.dot7,
            self.dot8,
            self.dot9,
            self.dot10,
        ]
        for i, d in enumerate(dots):
            if i in [0, 9]:  # first and last dot
                # uc = '\u00ae' if step_num - 1 != i else '\u2b24' # circled 'R' vs filled circle, respectively
                # pad = '0' if step_num - 1 != i else '1'
                # d.setStyleSheet(f"padding-top: {pad}px;")
                color = "#cccccc" if step_num - 1 != i else "#515151"
                d.setStyleSheet(f"color: {color};")
            else:
                uc = (
                    "\u25ef" if step_num - 1 != i else "\u2b24"
                )  # open circle vs filled circle, respectively
                # d.setStyleSheet(f"padding-top: 1px;")
                d.setText(uc)

    def gotoStepNum(self, obj, step_num=1):

        dots = [
            self.dot1,
            self.dot2,
            self.dot3,
            self.dot4,
            self.dot5,
            self.dot6,
            self.dot7,
            self.dot8,
            self.dot9,
            self.dot10,
        ]
        for i, d in enumerate(dots):
            if d.underMouse():
                step_num = i + 1
                break

        # if self.AI_SelectTool_Frame.isVisible():
        #     self.AI_SelectTool_Frame.setVisible(False)

        if self.allow_modify == False and step_num < 9:
            self.tool_Modify.setChecked(True)
            self.action_modify()  # self.tool_Modify.clicked.emit()

        # determine step direction
        if step_num < self.stateStep + 2:
            self.step_direction = "backwards"
        else:
            self.step_direction = "forwards"

        if step_num == 10:
            self.progressBar.setValue(100)  # Finished
            self.progressBar.setFormat(
                'Finished: View most recent "Analyze" results')
            self.stateStep = 8
            self.tool_Cancel.setEnabled(True)
            self.lowerGraphs.setVisible(False)
            self.graphStack.setCurrentIndex(1)
            self.setDotStepMarkers(step_num)
            return

        enable_cancel = self.xml_path != None
        enable_analyze = len(self.poi_markers) > 2
        if not enable_cancel:
            Log.w("Please load a run prior to using the step jumper dots.")
        elif self.stateStep + 2 == step_num:
            Log.d("User clicked step jumper dot of current step. No action.")
        elif step_num == 1:
            if PopUp.question(
                self,
                "Are you sure?",
                "Any manual points will be lost if you go back to Step 1.\n\nProceed?",
            ):
                self.parent.analyze_data(
                    self.cBox_Devices.currentText(),
                    self.getFolderFromRun(self.cBox_Runs.currentText()),
                    None,
                )  # force back to step 1 of 8
                self.enable_buttons()
        elif enable_analyze:
            self.stateStep = step_num - 3
            self.getPoints()  # increment to next step
            self.enable_buttons()
        elif enable_analyze == False and step_num == 2:
            # special case: allow next action if dot is clicked instead of button
            self.tool_Next.clicked.emit()  # calls enable_buttons()
        else:
            Log.w(
                "Please select begin and end points prior to using the step jumper dots."
            )

    def appendAuditToXml(self):
        data_path = self.loaded_datapath
        xml_path = data_path[0:-4] + \
            ".xml" if self.xml_path == None else self.xml_path
        xml_params = {}
        if secure_open.file_exists(xml_path, "audit"):
            xml_text = ""
            with open(xml_path, "r") as f:
                xml_text = f.read()
            if isinstance(xml_text, bytes):
                xml_text = xml_text.decode()
            run = minidom.parseString(xml_text)
            xml = run.documentElement

            # create or append new audits element
            try:
                audits = xml.getElementsByTagName("audits")[-1]
            except:
                audits = run.createElement("audits")
                xml.appendChild(audits)

            valid, infos = UserProfiles.session_info()
            if valid:
                Log.d(f"Found valid session: {infos}")
                username = infos[0]
                initials = infos[1]
                salt = UserProfiles.find(username, initials)[1][:-4]
                userrole = infos[2]
            else:
                Log.w(
                    f"Found invalid session: searching for user ({self.initials})")
                username = None  # not known in this context (yet)
                initials = self.initials
                salt = UserProfiles.find(username, initials)[1][:-4]
                userinfo = UserProfiles.get_user_info(f"{salt}.xml")
                username = userinfo[0]
                initials = userinfo[1]
                userrole = userinfo[2]

            audit_action = "ANALYZE"
            timestamp = self.signed_at
            machine = Architecture.get_os_name()
            hash = hashlib.sha256()
            hash.update(salt.encode())  # aka 'profile'
            hash.update(audit_action.encode())
            hash.update(timestamp.encode())
            hash.update(machine.encode())
            hash.update(username.encode())
            hash.update(initials.encode())
            hash.update(userrole.encode())
            signature = hash.hexdigest()

            audit1 = run.createElement("audit")
            audit1.setAttribute("profile", salt)
            audit1.setAttribute("action", audit_action)
            audit1.setAttribute("recorded", timestamp)
            audit1.setAttribute("machine", machine)
            audit1.setAttribute("username", username)
            audit1.setAttribute("initials", initials)
            audit1.setAttribute("role", userrole)
            audit1.setAttribute("signature", signature)
            audits.appendChild(audit1)

            hash = hashlib.sha256()
            a_tags = xml.getElementsByTagName("audit")
            for i, a in enumerate(a_tags):
                if i == len(a_tags) - 1:
                    ref_signature = hash.hexdigest()
                if a.hasAttribute("signature"):
                    hash.update(a.getAttribute("signature").encode())
            audits_signature = hash.hexdigest()
            if audits.hasAttribute("signature"):
                if audits.getAttribute("signature") == ref_signature:
                    audits.setAttribute("signature", audits_signature)
                else:
                    if audits.attributes["signature"].value.find("X") < 0:
                        audits.attributes["signature"].value += "X"
                    Log.e(
                        "Audits signature does not match for this run! Unable to apply new signature."
                    )
            else:
                audits.setAttribute("signature", audits_signature)

            with open(self.xml_path, "w") as f:
                xml_str = run.toxml()
                f.write(xml_str)
                Log.d(f"Added <audit> to XML file: {self.xml_path}")

    def appendPointsToXml(self, poi_vals):
        data_path = self.loaded_datapath
        xml_path = data_path[0:-4] + \
            ".xml" if self.xml_path == None else self.xml_path
        xml_params = {}
        if secure_open.file_exists(xml_path, "audit"):
            xml_text = ""
            with open(xml_path, "r") as f:
                xml_text = f.read()
            if isinstance(xml_text, bytes):
                xml_text = xml_text.decode()
            run = minidom.parseString(xml_text)
            xml = run.documentElement

            # create new points element
            recorded_at = (
                self.signed_at
                if self.signature_required
                else dt.datetime.now().isoformat()
            )
            points = run.createElement("points")
            points.setAttribute("recorded", recorded_at)
            xml.appendChild(points)

            for x, y in enumerate(poi_vals):
                point = run.createElement("point")
                point.setAttribute("name", str(x))
                point.setAttribute("value", str(y))
                points.appendChild(point)

            hash = hashlib.sha256()
            for p in points.childNodes:
                for name, value in p.attributes.items():
                    hash.update(name.encode())
                    hash.update(value.encode())
            signature = hash.hexdigest()
            points.setAttribute("signature", signature)

            with open(self.xml_path, "w") as f:
                xml_str = run.toxml()
                f.write(xml_str)
                Log.d(f"Added <points> to XML file: {self.xml_path}")

    def markerMoveFinished(self, marker):
        ax = self.graphWidget
        ax1 = self.graphWidget1
        ax2 = self.graphWidget2
        ax3 = self.graphWidget3
        tt1 = marker.value()
        tx1 = next(x for x, y in enumerate(self.xs) if y >= tt1)
        if abs(self.xs[tx1] - tt1) > abs(self.xs[tx1 - 1] - tt1):
            tx1 -= 1
        marker.setValue(self.xs[tx1])  # snap to nearest point
        marker_idx = -1
        for idx, pm in enumerate(self.poi_markers):
            if pm.value() == marker.value():
                marker_idx = idx
                break
        if self.moved_markers[marker_idx] == False:
            Log.d(
                f"Marker {marker_idx} has been moved by the user! Flagged for model tuning."
            )
        # clear flag if it moved from AI directive; only set on manual movement
        # if not self.AI_moving_marker else False
        self.moved_markers[marker_idx] = True
        self.detect_change()
        # setXRange for 'ax' all the time on marker move to keep markers in view (except for Step 2)
        if self.stateStep > 0:
            tt0 = self.poi_markers[0].value()
            tx0 = next(x for x, y in enumerate(self.xs) if y >= tt0)
            tt2 = self.poi_markers[-1].value()
            tx2 = next(x for x, y in enumerate(self.xs) if y >= tt2)
            ax.setXRange(tt0, tt2, padding=0.12)
            mn = min(
                np.amin(self.ys_freq_fit[tx0:tx2]),
                np.amin(self.ys_fit[tx0:tx2]),
                np.amin(self.ys_diff_fit[tx0:tx2]),
            )
            mx = max(
                np.amax(self.ys_freq_fit[tx0:tx2]),
                np.amax(self.ys_fit[tx0:tx2]),
                np.amax(self.ys_diff_fit[tx0:tx2]),
            )
            ax.setYRange(mn, mx, padding=0.05)
        if self.stateStep in range(1, 7):
            cur_val = marker.value()
            cur_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val)
            if cur_idx == len(self.xs) - 1:
                return  # do not process skipped points on marker move
            ws = self.getContextWidth()[0]
            pad = 0.05 if self.stateStep >= 4 else 0.05
            if tx1 + ws >= len(self.xs) - 1:
                ws = len(self.xs) - tx1 - 1
            ax1.setXRange(self.xs[tx1 - ws], self.xs[tx1 + ws], padding=0)
            ax2.setXRange(self.xs[tx1 - ws], self.xs[tx1 + ws], padding=0)
            ax3.setXRange(self.xs[tx1 - ws], self.xs[tx1 + ws], padding=0)
            if self.stateStep >= 4:
                ax1.setYRange(
                    np.min(self.ys_freq_fit[tx1 - ws: tx1 + ws]),
                    np.max(self.ys_freq_fit[tx1 - ws: tx1 + ws]),
                    padding=pad,
                )
                ax2.setYRange(
                    np.min(self.ys_diff_fit[tx1 - ws: tx1 + ws]),
                    np.max(self.ys_diff_fit[tx1 - ws: tx1 + ws]),
                    padding=pad,
                )
                ax3.setYRange(
                    np.min(self.ys_fit[tx1 - ws: tx1 + ws]),
                    np.max(self.ys_fit[tx1 - ws: tx1 + ws]),
                    padding=pad,
                )
                pos1 = np.column_stack((self.xs[tx1], self.ys_freq_fit[tx1]))
                pos2 = np.column_stack((self.xs[tx1], self.ys_diff_fit[tx1]))
                pos3 = np.column_stack((self.xs[tx1], self.ys_fit[tx1]))
            else:
                ax1.setYRange(
                    np.min(self.ys_freq[tx1 - ws: tx1 + ws]),
                    np.max(self.ys_freq[tx1 - ws: tx1 + ws]),
                    padding=pad,
                )
                ax2.setYRange(
                    np.min(self.ys_diff[tx1 - ws: tx1 + ws]),
                    np.max(self.ys_diff[tx1 - ws: tx1 + ws]),
                    padding=pad,
                )
                ax3.setYRange(
                    np.min(self.ys[tx1 - ws: tx1 + ws]),
                    np.max(self.ys[tx1 - ws: tx1 + ws]),
                    padding=pad,
                )
                pos1 = np.column_stack((self.xs[tx1], self.ys_freq[tx1]))
                pos2 = np.column_stack((self.xs[tx1], self.ys_diff[tx1]))
                pos3 = np.column_stack((self.xs[tx1], self.ys[tx1]))
            self.star1.setData(pos=pos1)
            self.star2.setData(pos=pos2)
            self.star3.setData(pos=pos3)
        # if (
        #     self.moved_markers[self.AI_SelectTool_At]
        #     and self.AI_SelectTool_Frame.isVisible()
        # ):
        #     # move AI Tool to new marker location
        #     self.summaryAt(self.AI_SelectTool_At)

    def getRunInfo(self):
        """
        Load and display information about a run from an XML file, initializing
        a GUI to view or edit the run's details.

        This method reads an XML file specified by `self.xml_path` to extract
        attributes such as the run's name, associated CSV file path, ruling
        (e.g., good or bad), and optionally, the username of the parent control.
        It ensures that only one instance of the Run Info GUI is active, and
        manages communication between the main thread and a worker thread for
        GUI display and user interaction.

        If the XML path is invalid or not provided, the method does nothing.

        Attributes:
            self.xml_path (str): Path to the XML file containing the run information.
            self.parent: Reference to the parent object (if any), used to extract the
                username for run metadata.
            self.bThread (QtCore.QThread): Thread handling the Run Info GUI worker.
            self.bWorker (QueryRunInfo): Worker object for the Run Info GUI.

        Raises:
            Exception: If there are issues reading or parsing the XML file, or if
                GUI initialization fails.

        Example:
            self.xml_path = "path/to/run_info.xml"
            self.getRunInfo()
        """
        # Check if the XML path is provided
        if self.xml_path != None:
            Log.d(tag=TAG, msg=f"Loaded xml_path={self.xml_path}")

            # Read the XML file's content.
            xml_text = ""
            with open(self.xml_path, "r") as f:
                xml_text = f.read()

            # Decode if the content is in bytes format.
            if isinstance(xml_text, bytes):
                xml_text = xml_text.decode()

            # Parse the XML content and extract attributes from
            # the XML.
            xml = minidom.parseString(xml_text)
            run = xml.documentElement
            run_name = run.getAttribute("name")
            run_path = self.xml_path[0:-4] + ".csv"
            is_good = run.getAttribute("ruling")

            # Get the username from the parent control, if available.
            user_name = (
                None
                if self.parent == None
                else self.parent.ControlsWin.username.text()[6:]
            )
            # check signatures of XML, render a new QueryRunInfo() and allow saving changes
            # (when editing runinfo, append to existing audit, not overwrite as new CAPTURE).
            if hasattr(self, "bThread"):
                if self.bThread.isRunning():
                    Log.w("Run Info GUI already open. Re-showing instead.")
                    self.bWorker.hide()
                    self.bWorker.show()
                    return

            # Initialize the thread and worker for the Run Info GUI.
            self.bThread = QtCore.QThread()
            self.bWorker = QueryRunInfo(
                run_name=run_name,
                run_path=run_path,
                run_ruling=is_good,
                user_name=user_name,
                recall_from=self.xml_path,
                parent=self.parent,
            )  # TODO: more secure to pass user_hash (filename)

            # Configure the Run Info GUI worker.
            self.bWorker.setRuns(1, 0)
            self.bThread.started.connect(self.bWorker.show)
            self.bWorker.finished.connect(self.bThread.quit)
            self.bWorker.finished.connect(self.update_run_names)

            # IPC signal to get the updated path name from the Run Info window on
            # change.
            self.bWorker.updated_run.connect(self.update_current_run_info)
            self.bWorker.updated_xml_path.connect(self.setXmlPath)

            # Start the thread to display the Run Info GUI
            self.bThread.start()

    def update_current_run_info(self, xml_path, new_name, old_name, date):
        """
        Updates the current run information in the combo box and the `run_names` dictionary.

        Args:
            new_name (str): The new name to update in the combo box and dictionary.
            old_name (str): The old name to search for in the combo box and dictionary.
            date (str): The date associated with the run, used to form the complete name.

        Raises:
            None: Logs an error message if the old name with the specified date is not found in the combo box.

        Updates:
            - If the item with the old name exists in the combo box, updates it with the new name.
            - Searches for a key in the `run_names` dictionary that contains the old name followed by a colon (:).
            If found, extracts the part of the key after the colon, removes the old key, and adds a new key with 
            the new name and the extracted value.
            - Updates the `text_Created` field to display the new name and date.

        Example:
            If the combo box contains "OldName (2024-11-20)" and the `run_names` dictionary contains:
                {
                    "OldName:Details": "value1"
                }
            Calling `update_current_run_info("NewName", "OldName", "2024-11-20")` will:
            - Update the combo box to "NewName (2024-11-20)"
            - Update the dictionary to:
                {
                    "NewName:Details": "NewName"
                }
            - Set `text_Created` to "NewName (2024-11-20)".
        """
        index = self.cBox_Runs.findText(f"{old_name} ({date})")

        # Check if the old name exists in the combo box
        if index != -1:
            # Update the item with the new name
            self.cBox_Runs.setItemText(index, f"{new_name} ({date})")
        else:
            Log.e(
                TAG, f"Item with name '{old_name} ({date})' not found in the combo box.")
        for key in list(self.run_names.keys()):  # Use list to avoid runtime changes
            if f"{old_name}:" in key:
                # Extract the part of the key after the ':'
                _, after_colon = key.split(":", 1)  # Split at the first ':'
                # Store the value and remove the entry
                value = self.run_names.pop(key)
                after_colon = after_colon.strip()
                break
        value = self.run_timestamps.pop(key)
        self.run_timestamps[f'{new_name}:{after_colon}'] = value
        self.run_names[f'{new_name}:{after_colon}'] = new_name
        self.text_Created.setText(f'Loaded: {new_name} ({date})')
        self.loaded_datapath = xml_path[:-4] + ".csv"

    def update_run_names(self):
        """
        Used as a reciever from QueryRunInfo to update the xml_path name
        to the modified xml_path name.
        """
        if self.bWorker.run_name_changed:
            loaded_idx = self.cBox_Runs.currentIndex()
            devs = FileStorage.DEV_get_all_device_dirs()
            for i, _ in enumerate(devs):
                self.updateRun(i)
            self.cBox_Runs.setCurrentIndex(loaded_idx)

    def Analyze_Data(self, data_path):

        # lazy load scipy modules
        from scipy.signal import argrelextrema
        from scipy.signal import savgol_filter

        self.stateStep = -1
        self.loaded_datapath = data_path

        self.btn_Back.setEnabled(False)
        self.btn_Next.setEnabled(True)

        try:
            Log.i("Analysis file = {}".format(data_path))
            data_title = os.path.splitext(os.path.basename(data_path))[0]

            if True:
                with secure_open(data_path, "r", "capture") as f:
                    csv_headers = next(f)

                    if isinstance(csv_headers, bytes):
                        csv_headers = csv_headers.decode()

                    if "Ambient" in csv_headers:
                        csv_cols = (2, 4, 6, 7)
                    else:
                        csv_cols = (2, 3, 5, 6)

                    data = loadtxt(
                        f.readlines(), delimiter=",", skiprows=0, usecols=csv_cols
                    )

            relative_time = data[:, 0]
            temperature = data[:, 1]
            resonance_frequency = data[:, 2]
            dissipation = data[:, 3]

            # check for and remove time jumps that would break analysis
            t_last = 0
            rows_to_toss = []
            for x, t in enumerate(relative_time):
                if t < t_last:
                    rows_to_toss.append(x - 1)
                t_last = t
            if len(rows_to_toss) > 0:
                Log.w(
                    f"Warning: time jump(s) observed at the following indices: {rows_to_toss}"
                )
                relative_time = np.delete(relative_time, rows_to_toss)
                temperature = np.delete(temperature, rows_to_toss)
                resonance_frequency = np.delete(
                    resonance_frequency, rows_to_toss)
                dissipation = np.delete(dissipation, rows_to_toss)
                Log.w(
                    "Time jumps removed from dataset for analysis purposes (original file unchanged)"
                )

            poi_vals = []
            if self.askForPOIs:
                xml_path = (
                    data_path[0:-4] +
                    ".xml" if self.xml_path == None else self.xml_path
                )
                if os.path.exists(xml_path):
                    doc = minidom.parse(xml_path)
                    points = doc.getElementsByTagName("points")
                    if len(points) > 0:
                        points = points[-1]  # most recent element
                        for p in points.childNodes:
                            if p.nodeType == p.TEXT_NODE:
                                continue  # only process elements
                            value = p.getAttribute("value")
                            try:
                                poi_vals.append(int(value))
                            except:
                                Log.e(
                                    f'Point value "{value}" in XML is not an integer.'
                                )
                        poi_vals.sort()
                    else:
                        Log.d("No points found in XML file for this run.")
                else:
                    Log.w(TAG,
                          f'Missing XML file: Expected at "{xml_path}" for this run.')
            self.show_analysis_immediately = False
            self.model_run_this_load = False
            if self.askForPOIs and len(poi_vals) == 6:
                self.askForPOIs = False
                Log.d(f"Found prior POIs from XML file: {poi_vals}")

                if (
                    False
                    and QtWidgets.QMessageBox.No
                    == QtWidgets.QMessageBox.question(
                        None,
                        "Run Already Analyzed",
                        'Would you like to re-analyze this run?\n\nSelect "No" to view the saved results.',
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.No,
                    )
                ):
                    Log.i("Showing prior saved analysis results...")
                    self.stateStep = 6  # show summary and then analyze
                    self.show_analysis_immediately = True
                else:
                    self.stateStep = 6  # show summary
            if self.askForPOIs and len(self.poi_markers) != 0:
                self.askForPOIs = (
                    False  # re-analyze Step 1, don't auto advance to Summary
                )
            if self.model_result == -1:  # self.stateStep != 6:
                self.model_result = -1
                self.model_candidates = None
                self.model_engine = "None"
                if Constants.QModel_predict:
                    try:
                        with secure_open(data_path, "r", "capture") as f:
                            fh = BytesIO(f.read())
                            label = self.QModel_clusterer.predict_label(fh)
                            fh.seek(0)
                            act_poi = [None] * 6  # no initial guesses
                            candidates = getattr(
                                self, f"QModel_predict_{label}"
                            ).predict(fh, run_type=label, act=act_poi)
                            predictions = []
                            for p, c in candidates:
                                predictions.append(
                                    p[0]
                                )  # assumes 1st point is best point
                            self.model_run_this_load = True
                            self.model_result = predictions
                            self.model_candidates = candidates
                            self.model_engine = "QModel"
                        if isinstance(self.model_result, list) and len(self.model_result) == 6:
                            if len(poi_vals) != 6:
                                Log.d(
                                    "Model ran, updating 'poi_vals' since we DO NOT have prior points")
                                poi_vals = self.model_result.copy()
                                out_of_order = False
                                last_p = 0
                                for i, p in enumerate(poi_vals):
                                    if p < last_p:
                                        if not out_of_order:
                                            # print this on 1st indication only
                                            Log.e(
                                                tag=f"[{self.model_engine}]",
                                                msg=f"Predictions are out of order! They have been corrected to prevent errors."
                                            )
                                        out_of_order = True
                                        if i == 0:  # first POI
                                            poi_vals[i] = int(poi_vals[1] / 2)
                                        elif i == len(poi_vals) - 1:  # last POI
                                            poi_vals[i] = int(
                                                (poi_vals[i-1] + len(dissipation)) / 2)
                                        else:  # any other POI, not first nor last
                                            poi_vals[i] = int(
                                                (poi_vals[i-1] + poi_vals[i+1]) / 2)
                                        Log.e(
                                            tag=f"[{self.model_engine}]",
                                            msg=f"Corrected point {i+1}: idx {p} -> {poi_vals[i]}"
                                        )
                                    last_p = p
                            else:
                                Log.d(
                                    "Model ran, but not updating 'poi_vals' since we DO have prior points")
                        else:
                            self.model_result = -1  # try fallback model
                    except Exception as e:
                        Log.e(e)
                        Log.e(
                            "Error using 'QModel'... Using 'ModelData' as fallback (less accurate)."
                        )
                        # raise e # debug only
                        self.model_result = -1  # try fallback model
                if self.model_result == -1 and Constants.ModelData_predict:
                    try:
                        self.model_run_this_load = True
                        self.model_result = self.dataModel.IdentifyPoints(
                            data_path, relative_time, resonance_frequency, dissipation
                        )
                        self.model_engine = "ModelData"
                        if isinstance(self.model_result, list):
                            poi_vals.clear()
                            # show point with highest confidence for each:
                            self.model_select = []
                            self.model_candidates = []
                            for point in self.model_result:
                                self.model_select.append(0)
                                if isinstance(point, list):
                                    self.model_candidates.append(point)
                                    select_point = point[self.model_select[-1]]
                                    select_index = select_point[0]
                                    select_confidence = select_point[1]
                                    poi_vals.append(select_index)
                                else:
                                    self.model_candidates.append([point])
                                    poi_vals.append(point)
                        elif self.model_result == -1:
                            Log.w(
                                "Model failed to auto-calculate points of interest for this run!"
                            )
                            pass
                        else:
                            Log.e(
                                "Model encountered an unexpected response. Please manually select points."
                            )
                            pass
                    except:
                        limit = None
                        t, v, tb = sys.exc_info()
                        from traceback import format_tb

                        a_list = ["Traceback (most recent call last):"]
                        a_list = a_list + format_tb(tb, limit)
                        a_list.append(f"{t.__name__}: {str(v)}")
                        for line in a_list:
                            Log.e(line)

                if len(self.poi_markers) != 0:
                    poi_vals = [
                        poi_vals[0],
                        poi_vals[-1],
                    ]  # take first and last only, allow user input

            # raw data
            xs = relative_time
            ys = dissipation

            # Computes initial difference cancelations for difference, resonance frequency
            # and dissipation and applies them to the UI curves.
            canceled_diss, canceled_diff, canceled_rf = None, None, None
            if self.drop_effect_cancelation_checkbox.isChecked():
                canceled_diss, canceled_diff, canceled_rf = self._correct_drop_effect(
                    self.loaded_datapath)
                if canceled_diss is not None:
                    ys = canceled_diss

            # use rough smoothing based on total runtime to figure start/stop
            total_runtime = xs[-1]
            smooth_factor = total_runtime * Constants.smooth_factor_ratio
            smooth_factor = int(smooth_factor) + (int(smooth_factor + 1) % 2)
            if smooth_factor < 3:
                smooth_factor = 3
            Log.i(TAG, f"Total run time: {total_runtime} secs")
            Log.d(
                TAG, f"Smoothing: {smooth_factor}"
            )  # the nearest odd number of seconds (runtime)
            Log.d(TAG, f"Applying smooth factor for first 90s ONLY.")

            t_first_90_split = (
                len(xs)
                if total_runtime <= 90
                else next(x for x, t in enumerate(xs) if t > 90)
            )
            extend_data = True if total_runtime > 90 else False
            extend_smf = int(
                smooth_factor / 20
            )  # downsample factor for extended data > 90s
            extend_smf += int(extend_smf + 1) % 2  # force to odd number

            if extend_data and len(xs) < t_first_90_split + 2 * extend_smf:
                Log.w(
                    "Not enough points after 90s to downsample effectively when plotting. Not downsampling this dataset!"
                )
                t_first_90_split = len(xs)
                extend_data = False

            ys_fit = savgol_filter(ys[:t_first_90_split], smooth_factor, 1)
            if extend_data:
                ys_fit_ext = savgol_filter(
                    ys[t_first_90_split:],
                    min(len(ys[t_first_90_split:]), extend_smf),
                    1,
                )
                ys_fit = np.concatenate((ys_fit, ys_fit_ext))

            ys_diss_diff = savgol_filter(
                ys_fit[:t_first_90_split], smooth_factor, 1, 1)
            if extend_data:
                ys_diss_diff_ext = savgol_filter(
                    ys_fit[t_first_90_split:],
                    min(len(ys_fit[t_first_90_split:]), extend_smf),
                    1,
                    1,
                )
                ys_diss_diff = np.concatenate((ys_diss_diff, ys_diss_diff_ext))

            ys_diss_2ndd = savgol_filter(
                ys_diss_diff[:t_first_90_split], smooth_factor, 1, 1
            )
            if extend_data:
                ys_diss_2ndd_ext = savgol_filter(
                    ys_diss_diff[t_first_90_split:],
                    min(len(ys_diss_diff[t_first_90_split:]), extend_smf),
                    1,
                    1,
                )
                ys_diss_2ndd = np.concatenate((ys_diss_2ndd, ys_diss_2ndd_ext))

            ys_diss_diff_avg = np.average(
                ys_diss_diff
            )  # AJR TODO 4/14: pick up here, this line is too high for the 109cp run
            ys_diss_diff_offset = ys_diss_diff - ys_diss_diff_avg
            zeros3 = np.where(np.diff(np.sign(ys_diss_diff_offset)))[0]
            while len(zeros3) < 2:
                zeros3 = np.append(zeros3, 100)
            ys_diss_diff_avg = np.average(ys_diss_diff[zeros3[1]:])

            minima_idx = argrelextrema(ys_diss_2ndd, np.less)[0]
            minima_val = ys_diss_2ndd[minima_idx]
            minima_dict = {minima_idx[i]: minima_val[i]
                           for i in range(len(minima_idx))}
            minima_sort = sorted(minima_dict.items(), key=lambda kv: (kv[0]))

            maxima_idx = argrelextrema(ys_diss_2ndd, np.greater)[0]
            maxima_val = ys_diss_2ndd[maxima_idx]
            maxima_dict = {maxima_idx[i]: maxima_val[i]
                           for i in range(len(maxima_idx))}
            maxima_sort = sorted(maxima_dict.items(),
                                 key=lambda kv: (kv[1], kv[0]))

            start_stop = sorted(maxima_sort[-2:])
            start_stop = [
                start_stop[0][0],
                start_stop[1][0],
            ]  # , start_stop[2][0], start_stop[3][0]]
            t_start = np.amin(start_stop)
            t_stop = np.amax(start_stop) + (3 * smooth_factor)
            if t_stop < len(xs) / 2 or t_stop >= len(xs):
                if self.model_run_this_load == False and len(poi_vals) == 0:
                    Log.w(
                        f"Warning: t_stop was {t_stop} out of {len(xs)} but that seems unlikely!"
                    )
                    Log.w('Please confirm "End Point" during Step 1 point selection.')
                t_stop = len(xs) - 1
            if t_stop - t_start < len(xs) / 3 or t_start > len(xs) / 2:
                if self.model_run_this_load == False and len(poi_vals) == 0:
                    Log.w(
                        f"Warning: t_start was {t_start} out of {len(xs)} but that seems unlikely!"
                    )
                    Log.w('Please confirm "Begin Point" during Step 1 point selection.')
                t_start = 100

            if total_runtime < 3:
                Log.e(
                    "ERROR: Data run must be at least 3 seconds in total runtime to analyze."
                )
                return

            # get indices for 0.5 seconds to start of run
            t_0p5 = (
                0
                if xs[t_start] < 0.5
                else next(x + 0 for x, t in enumerate(xs) if t > 0.5)
            )
            t_1p0 = (
                t_start
                if xs[t_start] < 2.0
                else next(x + 1 for x, t in enumerate(xs) if t > 2.0)
            )

            # new maths for resonance and dissipation (scaled)
            avg = np.average(resonance_frequency[t_0p5:t_1p0])
            ys = ys * avg / 2

            ys_fit = ys_fit * avg / 2
            ys = ys - np.amin(ys_fit)
            ys_fit = ys_fit - np.amin(ys_fit)
            ys_freq = avg - resonance_frequency
            # 'RF' Drop Effect Correction
            if self.drop_effect_cancelation_checkbox.isChecked():
                if canceled_rf is not None:
                    ys_freq = avg - canceled_rf

            ys_freq_fit = savgol_filter(
                ys_freq[:t_first_90_split], smooth_factor, 1)
            if extend_data:
                ys_freq_fit_ext = savgol_filter(
                    ys_freq[t_first_90_split:],
                    min(len(ys_freq[t_first_90_split:]), extend_smf),
                    1,
                )
                ys_freq_fit = np.concatenate((ys_freq_fit, ys_freq_fit_ext))

            # # APPLY DROP EFFECT VECTORS
            # drop_offsets = np.zeros(ys.shape)
            # try:
            #     if self.correct_drop_effect.isChecked():
            #         baseline = np.average(ys[t_0p5:t_1p0])
            #         base_std = np.std(ys[t_0p5:t_1p0])
            #         drop_start = next(
            #             x - 1 for x, y in enumerate(ys) if y > baseline + 4*base_std and x > t_1p0)
            #         drop_start = next(x for x, t in enumerate(
            #             xs) if t > xs[drop_start] + 0.1)
            #         # next(ys[x + 2] for x,y in enumerate(ys) if y > Constants.drop_effect_cutoff_freq / 2 and x > t_1p0)
            #         drop_diss = ys[drop_start]
            #         if drop_diss > Constants.drop_effect_cutoff_freq:
            #             self.diff_factor = Constants.drop_effect_multiplier_high
            #         else:
            #             self.diff_factor = Constants.drop_effect_multiplier_low
            #         with open("QATCH/resources/lookup_drop_effect.csv", "r") as f:
            #             data = np.loadtxt(
            #                 f.readlines(), delimiter=",", skiprows=1)
            #             col = (
            #                 1
            #                 if self.diff_factor == Constants.drop_effect_multiplier_low
            #                 else 2
            #             )
            #             RR_offset = data[:, col]
            #             if drop_start + len(RR_offset) > len(drop_offsets):
            #                 # RR vector is longer than the actual run data, truncate it
            #                 drop_offsets[drop_start:] = RR_offset[
            #                     : len(drop_offsets) - drop_start
            #                 ]
            #             else:
            #                 # RR vector is shorter and needs to be padded with the final value
            #                 drop_offsets[drop_start: drop_start + len(RR_offset)] = (
            #                     RR_offset
            #                 )
            #                 drop_offsets[drop_start +
            #                              len(RR_offset):] = RR_offset[-1]
            #         Log.d(
            #             f"Applying vectors starting at time 't = {xs[drop_start]:1.3f}s'"
            #         )
            #         Log.d(
            #             f"Drop effect 'cutoff' dissipation frequency is {drop_diss:1.1f}Hz"
            #         )
            #         Log.d(
            #             f"Using {'low' if col == 1 else 'high'} viscosity drop effect 'diff_factor' and vector"
            #         )
            # except Exception as e:
            #     Log.e("ERROR:", e)

            baseline = np.average(dissipation[t_0p5:t_1p0])
            diff_factor = (
                Constants.default_diff_factor
            )  # 1.0 if baseline < 50e-6 else 1.5

            # Automatically compute optimal difference factor
            if self.difference_factor_optimizer_checkbox.isChecked():
                self.diff_factor = self._optimize_curve(self.loaded_datapath)

            if hasattr(self, "diff_factor"):
                diff_factor = self.diff_factor
            ys_diff = ys_freq - (diff_factor * ys)

            # 'Difference' Drop Effect Correction
            if self.drop_effect_cancelation_checkbox.isChecked():
                canceled_diss, canceled_diff, canceled_rf = self._correct_drop_effect(
                    self.loaded_datapath)
                if canceled_diff is not None:
                    ys_diff = canceled_diff

            # Invert difference curve if drop applied to outlet
            if np.average(ys_diff) < 0:
                Log.w("Inverting DIFFERENCE curve due to negative initial fill deltas")
                ys_diff *= -1

            ys_diff_fit = savgol_filter(
                ys_diff[:t_first_90_split], smooth_factor, 1)
            if extend_data:
                ys_diff_fit_ext = savgol_filter(
                    ys_diff[t_first_90_split:],
                    min(len(ys_diff[t_first_90_split:]), extend_smf),
                    1,
                )
                ys_diff_fit = np.concatenate((ys_diff_fit, ys_diff_fit_ext))
            Log.d(f"Difference factor: {diff_factor:1.3f}x")

            Log.d(f"Setting diff_factor on Advanced Settings menu")
            self.tbox_diff_factor.setText(f"{diff_factor:1.3f}")

            smf = max(3, int(smooth_factor / 10))
            if smf % 2 == 0:
                smf += 1  # force odd number
            ys_diff_fine = savgol_filter(ys_diff, smf, 1)
            ys_diff_diff = savgol_filter(
                ys_diff, smf, 1, 1
            )  # difference derivatives, not dissipation
            ys_diff_2ndd = savgol_filter(ys_diff_diff, smf, 1, 1)

            start_stop.clear()
            em1 = 0  # np.amax(ys_diff_fit[t_0p5:t_1p0])
            eh1 = abs(np.amax(ys_diff[t_0p5:t_1p0]) - em1)
            em2 = np.amax(ys_diff_fit)
            am2 = np.argmax(ys_diff_fit)
            eh2 = eh1 * 2  # np.amax(ys_diff[am2-100:]) - em2

            t0 = t_start
            try:
                t0 = next(
                    x for x, y in enumerate(ys_diff_fit) if y > 5 * eh2 and x > t_1p0
                )
            except:
                if self.model_run_this_load and len(poi_vals) == 0:
                    Log.w(
                        "Failed to locate rough start point using noise floor approximation."
                    )
                    Log.w('Please confirm "Begin Point" during Step 1 point selection.')
            dir = ys[t0] < 5
            while (
                True
            ):  # work back in time to find actual minimum difference (true start)
                if t0 < 0 or t0 > len(ys) - 1:
                    if self.model_run_this_load and len(poi_vals) == 0:
                        Log.w("Hit a limit (start)")
                    t0 = 0
                    break
                if ys[t0] < 5:
                    t0 += 1
                    if not dir:
                        break
                else:
                    t0 -= 1
                    if dir:
                        break
            start_stop.append(t0)
            t1 = am2
            try:
                t1 += next(x for x,
                           y in enumerate(ys_diff_fit[am2:]) if y < em2 - eh2)
            except:
                if self.model_run_this_load and len(poi_vals) == 0:
                    Log.w(
                        "Failed to locate rough end point using noise floor approximation."
                    )
                    Log.w('Please confirm "End Point" during Step 1 point selection.')
            while (
                True
            ):  # work back in time to find actual minimum difference (true start)
                if t1 - 50 < 0:
                    if self.model_run_this_load and len(poi_vals) == 0:
                        Log.w("Hit a limit (end)")
                    t1 = len(ys) - 1
                    break
                if ys_diff_fit[t1 - 50] > ys_diff_fit[t1]:
                    t1 -= 1
                else:
                    break
            start_stop.append(t1)

            self._update_analyze_progress(
                100, "Reading Run Data..."
            )  # 100% forces 'go fast'

        except Exception as e:
            self.progress_value_steps.clear()  # abort progressbar updates

            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

            Log.w(
                "An error occurred loading this run! Please manually select points for Analysis."
            )

        finally:
            # Create any missing required vars using available resources
            correction_needed = False
            if not "ys_freq" in locals():
                correction_needed = True
            if not "ys_diff" in locals():
                correction_needed = True
            if not "ys" in locals():
                correction_needed = True
            if not "ys_freq_fit" in locals():
                correction_needed = True
            if not "ys_diff_fit" in locals():
                correction_needed = True
            if not "ys_fit" in locals():
                correction_needed = True
            if not "ys_diss_2ndd" in locals():
                correction_needed = True
            if correction_needed:
                Log.w(
                    "Correcting missing parameters for manual point selection (no smoothing)..."
                )
                avg = resonance_frequency[0]
                restore_ys = False
                if not "ys" in locals():
                    restore_ys = True
                if not "ys_fit" in locals():
                    restore_ys = True
                if restore_ys:
                    ys = dissipation
                    ys_fit = ys
                    ys = ys * avg / 2
                    ys_fit = ys_fit * avg / 2
                    ys = ys - np.amin(ys_fit)
                    ys_fit = ys_fit - np.amin(ys_fit)
                if not "ys_freq" in locals():
                    ys_freq = avg - resonance_frequency
                if not "ys_freq_fit" in locals():
                    ys_freq_fit = ys_freq
                if not "ys_diff" in locals():
                    diff_factor = Constants.default_diff_factor
                    ys_diff = ys_freq - diff_factor * ys
                if not "ys_diff_fit" in locals():
                    ys_diff_fit = ys_diff
                if not "ys_diss_2ndd" in locals():
                    try:
                        ys_diss_diff = savgol_filter(ys_fit, 2, 1, 1)
                        ys_diss_2ndd = savgol_filter(ys_diss_diff, 2, 1, 1)
                    except:
                        Log.e("Unable to calculate 2nd derivative of 'ys' data!")
                        ys_diss_2ndd = ys_fit

            # fill self.progress_status_step dictionary to go full speed
            i = 0
            Log.d("Waiting on progress bar to finish")
            while self.progress_value_scanning and i < 300:
                i += 1
                QtCore.QCoreApplication.processEvents()
            self.progressBar.valueChanged.connect(self._update_progress_value)
            Log.d("Finished progress bar... proceed!")

        ax = self.graphWidget  # .plot(hour, temperature)
        ax1 = self.graphWidget1
        ax2 = self.graphWidget2
        ax3 = self.graphWidget3

        ax.clear()
        ax1.clear()
        ax2.clear()
        ax3.clear()

        self._update_progress_value(
            1, f"Step 1 of 8: Select Begin and End Points")
        self.setDotStepMarkers(1)
        ax.setTitle(None)
        ax.addLegend()

        ax1.setTitle("Resonance", color="green")
        ax2.setTitle("Difference", color="blue")
        ax3.setTitle("Dissipation", color="red")

        style = {"color": "b", "font-size": "12px"}
        ax.showAxis("left")
        ax.setLabel("left", "Frequency (Hz)", **style)
        ax.showAxis("bottom")
        ax.setLabel("bottom", "Time (secs)", **style)

        ax.showButtons()
        ax1.hideButtons()
        ax2.hideButtons()
        ax3.hideButtons()

        # Add grid
        ax.showGrid(x=True, y=True)
        # Set Range
        ax.setXRange(0, xs[-1], padding=0.05)
        ax.setYRange(
            0, max(np.amax(ys_freq), np.amax(ys), np.amax(ys_diff)), padding=0.05
        )

        self.lowerGraphs.setVisible(False)
        # ax1.setVisible(False)
        # ax2.setVisible(False)
        # ax3.setVisible(False)

        ax1.showGrid(x=True, y=True)
        ax2.showGrid(x=True, y=True)
        ax3.showGrid(x=True, y=True)

        mask = np.arange(0, len(xs), 1)
        self.fit1 = ax.plot(xs[mask], ys_freq_fit[mask],
                            pen="green", name="Resonance")
        self.fit2 = ax.plot(xs[mask], ys_diff_fit[mask],
                            pen="blue", name="Difference")
        self.fit3 = ax.plot(xs[mask], ys_fit[mask],
                            pen="red", name="Dissipation")

        noPen = pg.mkPen(color=(255, 255, 255), width=0,
                         style=QtCore.Qt.DotLine)
        self.scat1 = ax.plot(
            xs[mask],
            ys_freq[mask],
            pen=noPen,
            symbol="o",
            symbolSize=5,
            symbolBrush=("green"),
        )
        self.scat2 = ax.plot(
            xs[mask],
            ys_diff[mask],
            pen=noPen,
            symbol="o",
            symbolSize=5,
            symbolBrush=("blue"),
        )
        self.scat3 = ax.plot(
            xs[mask], ys[mask], pen=noPen, symbol="o", symbolSize=5, symbolBrush=("red")
        )
        # setting alpha value of scatter plots
        self.scat1.setAlpha(0.01, False)
        self.scat2.setAlpha(0.01, False)
        self.scat3.setAlpha(0.01, False)

        self.fit_1 = ax1.plot(
            xs[mask], ys_freq_fit[mask], pen="green", name="Resonance"
        )
        self.fit_2 = ax2.plot(
            xs[mask], ys_diff_fit[mask], pen="blue", name="Difference"
        )
        self.fit_3 = ax3.plot(xs[mask], ys_fit[mask],
                              pen="red", name="Dissipation")

        self.scat_1 = ax1.plot(
            xs[mask],
            ys_freq[mask],
            pen=noPen,
            symbol="o",
            symbolSize=5,
            symbolBrush=("green"),
        )
        self.scat_2 = ax2.plot(
            xs[mask],
            ys_diff[mask],
            pen=noPen,
            symbol="o",
            symbolSize=5,
            symbolBrush=("blue"),
        )
        self.scat_3 = ax3.plot(
            xs[mask], ys[mask], pen=noPen, symbol="o", symbolSize=5, symbolBrush=("red")
        )

        pos1 = np.column_stack((xs[0], ys_freq[0]))
        pos2 = np.column_stack((xs[0], ys_diff[0]))
        pos3 = np.column_stack((xs[0], ys[0]))
        self.star1 = pg.ScatterPlotItem(
            pos=pos1, symbol="star", size=25, brush=("black")
        )
        self.star2 = pg.ScatterPlotItem(
            pos=pos2, symbol="star", size=25, brush=("black")
        )
        self.star3 = pg.ScatterPlotItem(
            pos=pos3, symbol="star", size=25, brush=("black")
        )
        ax1.addItem(self.star1)
        ax2.addItem(self.star2)
        ax3.addItem(self.star3)
        self.gstars1 = pg.ScatterPlotItem(
            pos=pos1, symbol="star", size=10, brush=("gray")
        )
        self.gstars2 = pg.ScatterPlotItem(
            pos=pos2, symbol="star", size=10, brush=("gray")
        )
        self.gstars3 = pg.ScatterPlotItem(
            pos=pos3, symbol="star", size=10, brush=("gray")
        )
        ax1.addItem(self.gstars1)
        ax2.addItem(self.gstars2)
        ax3.addItem(self.gstars3)

        if len(poi_vals) > 0:
            start_stop = poi_vals

        self.poi_markers = []
        for pt in start_stop:
            poi_marker = pg.InfiniteLine(
                pos=xs[pt], angle=90, pen="b", bounds=[xs[0], xs[-1]], movable=True
            )
            poi_marker.setPen(color="blue")
            poi_marker.addMarker("<|>")
            ax.addItem(poi_marker)
            poi_marker.sigPositionChangeFinished.connect(
                self.markerMoveFinished)
            self.poi_markers.append(poi_marker)

        self.xs = xs
        self.ys = ys
        self.ys_freq = ys_freq
        self.ys_diff = ys_diff
        self.ys_fit = ys_fit
        self.ys_freq_fit = ys_freq_fit
        self.ys_diff_fit = ys_diff_fit
        self.ys_diss_2ndd = ys_diss_2ndd
        self.smooth_factor = smooth_factor
        self.data_time = relative_time
        self.data_freq = resonance_frequency
        self.data_diss = dissipation

        # self.AI_Guess_Idxs = [0, 0, 0, 0, 0, 0]
        # self.AI_has_starting_values = False
        if (
            self.model_run_this_load and self.stateStep != 6
        ):  # model has guess(es) and there is no prior run
            if len(poi_vals) == 6:
                Log.i(
                    "Model successfully calculated points of interest for this dataset."
                )
                Log.d(
                    f"Model Result = {self.model_engine}: {self.model_result}")
                self.stateStep = 6  # show summary

                def get_logger_for_confidence(confidence):
                    logger = Log.e  # less than 33%
                    if confidence > 66:
                        logger = Log.i  # greater than 66%
                    elif confidence > 33:
                        logger = Log.w  # from 33% to 66%
                    return logger

                point_names = ["start", "end_fill",
                               "post", "ch1", "ch2", "ch3"]
                for i, (candidates, confidences) in enumerate(self.model_candidates):
                    if i == 2:
                        # do not print confidence of "post" point, it doesn't matter
                        continue
                    point_name = point_names[i]
                    confidence = 100 * \
                        confidences[0] if len(confidences) > 0 else 0
                    num_spaces = len(point_names[1]) - len(point_name) + 1
                    get_logger_for_confidence(confidence)(
                        tag=f"[{self.model_engine}]",
                        msg=f"Confidence @ {point_name}:{' '*num_spaces}{confidence:2.0f}%"
                    )

            else:
                Log.e(
                    "Please manually select points of interest to Analyze this dataset."
                )
        else:
            # model not run this load
            if self.stateStep == 6:
                # self.AI_has_starting_values = True
                Log.i("Loaded points of interest from a prior run of Analyze tool.")
            else:
                Log.e(
                    "Please manually select points of interest to Analyze this dataset."
                )
        # if len(poi_vals) > 0:
        #     self.getPoints() # show summary page if they want to view previous results or rough step 2 if they said re-analyze
        if self.stateStep == 6:
            Log.d("Skipping to summary step")
            self.getPoints()  # show summary page if points already exist in XML
        if self.show_analysis_immediately:
            Log.d("Showing analysis immediately")
            self.getPoints()  # confirm and analyze only if they want to view previous results

    def resizeEvent(self, event):
        # if self.AI_SelectTool_Frame.isVisible():
        #     self.AI_SelectTool_Frame.setVisible(
        #         False
        #     )  # require re-click to show popup tool incorrect position
        pass

    def _optimize_curve(self, data_path):
        """
        Optimizes the difference factor for a given data file.

        This method reads a data file securely, processes its header, and runs a 
        curve optimization algorithm to determine the optimal difference factor 
        and its associated score. If an optimal factor is found, it is returned. 
        Otherwise, the default difference factor is used.

        Args:
            data_path (str): Path to the data file to be optimized.

        Returns:
            float: The optimal difference factor if found; otherwise, the default 
            difference factor (`Constants.default_diff_factor`).

        Raises:
            Any exception during the secure file operation or optimization process 
            will propagate and should be handled by the caller.

        Example:
            optimal_factor = self._optimize_curve("path/to/data/file")
        """
        try:
            optimal_factor = None
            with secure_open(data_path, "r", "capture") as f:
                file_header = BytesIO(f.read())
                optimizer = DifferenceFactorOptimizer(file_header)
                optimal_factor, lb, rb = optimizer.optimize()
                Log.i(
                    TAG, f'Using difference factor {optimal_factor} optimized between {lb}s and {rb}s.')

            if optimal_factor is not None:
                Log.d(
                    TAG, f"Reporting difference factor of {optimal_factor}.")
                return optimal_factor
            else:
                Log.d(
                    TAG, f"No optimal difference factor found, reporting default of {Constants.default_diff_factor}.")
                return Constants.default_diff_factor
        except Exception as e:
            Log.e(
                TAG, f"Difference factor optimizer failed due to error. Using default factor.")
            Log.e(TAG, f"Error Details: {str(e)}")
            return Constants.default_diff_factor

    def _correct_drop_effect(self, file_header: str) -> tuple:
        """
        Corrects the dissipation drop effect in the provided file.

        This method reads the contents of the file specified by `file_header`, 
        applies a drop effect correction algorithm using the specified 
        difference factor, and returns the corrected data if successful.

        Args:
            file_header (str): Path to the file containing the data to be corrected.

        Returns:
            tuple or None: The corrected data if the correction is successful; 
            otherwise, returns None and logs that the original data will be used.

        Logs:
            - Info: Indicates the start of the drop effect cancellation process with the difference factor.
            - Debug: Indicates whether the drop effect cancellation was successful or not.

        Raises:
            IOError: If there is an issue opening or reading the file.
            Exception: For any unexpected errors during the correction process.
        """
        try:
            with secure_open(file_header, "r", "capture") as f:
                file_header = BytesIO(f.read())
                dec = DropEffectCorrection(
                    file_buffer=file_header, initial_diff_factor=self.diff_factor)
                corrected_data = dec.correct_drop_effect()

                Log.i(
                    TAG, f'Performing drop effect cancelation with difference factor {self.diff_factor}.')

            if corrected_data is not None:
                Log.d(
                    TAG, f"Dissipation drop effect cancelation successful.")
                return corrected_data
            else:
                Log.d(
                    TAG, f"Dissipation drop effect cancelation failed. Using original data.")
                return [None, None, None]
        except Exception as e:
            Log.e(
                TAG, f"Dissipation drop effect cancelation failed due to error. Using original data.")
            Log.e(TAG, f"Error Details: {str(e)}")
            return [None, None, None]


class AnalyzerWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int, str)

    def __init__(self, parent, data_path, xml_path, poi_vals, diff_factor):
        super().__init__()
        self.parent = parent
        self._exitSuccess = False

        # set global expectations
        self.loaded_datapath = data_path
        self.xml_path = xml_path
        self.poi_vals = poi_vals
        if diff_factor != None:
            self.diff_factor = diff_factor
        # else: self.diff_factor not set

        self._running = False
        self.progress.connect(self._started)
        self.progress.connect(QtCore.QCoreApplication.processEvents)
        self.finished.connect(self._stopped)

    def _started(self, val, status):
        self._running = True

    def _stopped(self):
        self._running = False

    def is_running(self):
        return self._running

    def exitCode(self):
        return self._exitSuccess

    def run(self):
        try:
            # self.progress.emit(0, "Analyzing...")
            status_label = "Analyzing..."
            self.update(status_label)

            # lazy load required modules
            from scipy.optimize import curve_fit
            from scipy.signal import argrelextrema
            from scipy.signal import savgol_filter
            import matplotlib.backends.backend_pdf
            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg,
                NavigationToolbar2QT as NavigationToolbar,
            )
            import matplotlib.pyplot as plt

            matplotlib.use("Qt5Agg")

            data, rows, cols = [
                {
                    "A": ["", "", "", ""],
                    "B": ["", "", "", ""],
                    "C": ["", "", "", ""],
                    "D": ["", "", "", ""],
                },
                4,
                4,
            ]
            results_table = TableView(data, rows, cols)
            results_figure = pg.PlotWidget()
            results_figure.setBackground("w")
            plot_text = pg.TextItem("", (51, 51, 51), anchor=(0.5, 0.5))
            plot_text.setHtml(
                "<span style='font-size: 10pt'>Analyze in-progress...</span>"
            )
            plot_text.setPos(0.5, 0.5)
            results_figure.addItem(plot_text, ignoreBounds=True)
            self.parent.results_split.replaceWidget(0, results_table)
            self.parent.results_split.replaceWidget(1, results_figure)
            self.parent.results_split.setEnabled(False)

            # self.progress.emit(50, "Analyzing...")

            confirm_envelopeSize = False
            confirm_startIndex = False
            confirm_stopIndex = False
            confirm_postIndex = False
            confirm_blipIndices = False

            poi_vals = self.poi_vals
            # poi_vals = np.insert(poi_vals, 2, poi_vals[1]+2)

            data_path = self.loaded_datapath
            data_title = os.path.splitext(os.path.basename(data_path))[0]
            Log.i("Starting Analysis process of file: {}".format(data_path))

            self.update(status_label)

            batch_input_type = "none"
            batch = "N/A"
            xml_path = (
                data_path[0:-4] +
                ".xml" if self.xml_path == None else self.xml_path
            )
            xml_params = {}
            if os.path.exists(xml_path):
                doc = minidom.parse(xml_path)
                params = doc.getElementsByTagName(
                    "params")[-1]  # most recent element

                for p in params.childNodes:
                    if p.nodeType == p.TEXT_NODE:
                        continue  # only process elements

                    name = p.getAttribute("name")
                    value = p.getAttribute("value")
                    xml_params[name] = value

                    if name == "batch_number" and p.hasAttribute("input"):
                        batch_input_type = p.getAttribute("input")

                    # if name == "bioformulation":
                    # if name == "protein":
                    # if name == "surfactant":
                    # if name == "concentration":
                    # if name == "surface_tension":
                    # if name == "contact_angle":
                    # if name == "density":

                batch = str(
                    xml_params.get("batch_number", "N/A")
                )  # used later on to pull batch params during analysis

                # START BATCH PARAMS INSERT #
                batch_params_old = {}
                batch_params_xml = doc.getElementsByTagName("batch_params")
                if len(batch_params_xml) > 0:
                    # most recent element
                    batch_params_xml = batch_params_xml[-1]
                    for p in batch_params_xml.childNodes:
                        if p.nodeType == p.TEXT_NODE:
                            continue  # only process elements

                        name = p.getAttribute("name")
                        value = p.getAttribute("value")
                        batch_params_old[name] = value
                else:
                    batch_params_xml = None

                batch_found = Constants.get_batch_param(batch)
                batch_params_all = Constants.get_batch_param(
                    batch, "ALL"
                )  # dictionary of {param_names:param_vals}
                batch_params_now = {}
                batch_params_now["BATCH"] = batch
                batch_params_now.update(
                    batch_params_all
                )  # update dict so that "BATCH" comes first, followed by other params

                # Look for changes
                changes = True
                if batch_params_xml != None:
                    changes = False
                    for key, val in batch_params_old.items():
                        if key in batch_params_now.keys():
                            if batch_params_now[key] != val:
                                changes = True
                                break
                        else:
                            changes = True
                            break
                    if not changes:
                        for key, val in batch_params_now.items():
                            if key in batch_params_old.keys():
                                if batch_params_old[key] != val:
                                    changes = True
                                    break
                            else:
                                changes = True
                                break

                # Add changed <batch_params> to XML
                if changes:
                    xml = doc.documentElement

                    # create new batch_params element
                    recorded_at = dt.datetime.now().isoformat()
                    batch_params = doc.createElement("batch_params")
                    batch_params.setAttribute("recorded", recorded_at)
                    xml.appendChild(batch_params)

                    # param = doc.createElement('batch_param')
                    # param.setAttribute('name', str("BATCH"))
                    # param.setAttribute('value', str(batch))
                    # batch_params.appendChild(param)

                    for k, v in batch_params_now.items():
                        param = doc.createElement("batch_param")
                        param.setAttribute("name", str(k))
                        param.setAttribute("value", str(v))
                        if k.upper() == "BATCH":
                            param.setAttribute("found", str(batch_found))
                        batch_params.appendChild(param)

                    hash = hashlib.sha256()
                    for p in batch_params.childNodes:
                        for name, value in p.attributes.items():
                            hash.update(name.encode())
                            hash.update(value.encode())
                    signature = hash.hexdigest()
                    batch_params.setAttribute("signature", signature)

                    with open(self.xml_path, "w") as f:
                        xml_str = doc.toxml()
                        f.write(xml_str)
                        Log.d(
                            f"Added <batch_params> to XML file: {self.xml_path}")
                # END BATCH PARAMS INSERT #

            self.update(status_label)

            Log.d(f"xml_path = {xml_path}")
            Log.d(f"xml_found = {os.path.exists(xml_path)}")
            Log.d(xml_params)

            BIOFORMULATION = xml_params.get(
                "bioformulation", "False") == "True"
            ST = float(xml_params.get("surface_tension", 69.0))
            CA = float(xml_params.get("contact_angle", 55.0))
            DENSITY = float(xml_params.get("density", 1.2))

            # only do this if "contact_angle" is auto-calculated (NOT if 'manual')
            if (
                batch_input_type == "auto" or True
            ):  # Per Zehra 2023-10-09, do this ALWAYS
                CA += float(Constants.get_batch_param(batch, "CA_offset"))

            self.update(status_label)

            if True:
                with secure_open(data_path, "r", "capture") as f:
                    csv_headers = next(f)

                    if isinstance(csv_headers, bytes):
                        csv_headers = csv_headers.decode()

                    if "Ambient" in csv_headers:
                        csv_cols = (2, 4, 6, 7)
                    else:
                        csv_cols = (2, 3, 5, 6)

                    data = loadtxt(
                        f.readlines(), delimiter=",", skiprows=0, usecols=csv_cols
                    )

            self.update(status_label)

            relative_time = data[:, 0]
            temperature = data[:, 1]
            resonance_frequency = data[:, 2]
            dissipation = data[:, 3]

            self.update(status_label)

            # check for and remove time jumps that would break analysis
            t_last = 0
            rows_to_toss = []
            for x, t in enumerate(relative_time):
                if t < t_last:
                    rows_to_toss.append(x - 1)
                t_last = t
            if len(rows_to_toss) > 0:
                Log.w(
                    f"Warning: time jump(s) observed at the following indices: {rows_to_toss}"
                )
                relative_time = np.delete(relative_time, rows_to_toss)
                temperature = np.delete(temperature, rows_to_toss)
                resonance_frequency = np.delete(
                    resonance_frequency, rows_to_toss)
                dissipation = np.delete(dissipation, rows_to_toss)
                Log.w(
                    "Time jumps removed from dataset for analysis purposes (original file unchanged)"
                )

            self.update(status_label)

            poi_path = os.path.join(
                os.path.split(data_path)[0], f"{data_title}_poi.csv"
            )
            cal_path = os.path.join(
                os.path.split(data_path)[0], f"{data_title}_cal.csv"
            )

            # calculate and apply temperature adjusted contact angle offset
            avg_run_temp = round(np.average(temperature), 1)
            CA_temp_factor = round(
                (avg_run_temp - 25.0) * Constants.temp_adjusted_CA_factor, 1
            )
            Log.d(f"Applying temperature adjusted CA offset:")
            Log.d(
                f"Temp CA offset = ({avg_run_temp}-25.0)*{Constants.temp_adjusted_CA_factor} = {CA_temp_factor}"
            )
            Log.d(
                f"Changing CA from {CA} to {CA + CA_temp_factor} with temperature offset {CA_temp_factor}"
            )
            CA += CA_temp_factor

            START_IDX = 0  # start-of-fill
            FILL_IDX = 1  # end-of-fill
            # NORMAL_PTS: 2-5 # 20%, 40%, 60%, 80%
            BLIP1_IDX = 6  # ch 1 fill
            # MIDP2_IDX = 6   # not used
            BLIP2_IDX = 7  # ch 2 fill
            # MIDP3_IDX = 8   # not used
            BLIP3_IDX = 8  # ch 3 fill

            # NOTE: start, eof, mid1, blip1, mid2, blip2, mid3, blip3
            # Support flexible array formatting in batch params lookup file:
            # [1.15, 1.61, 2.17, 2.67, 3.23, 5.00, 10.90, 16.2]  -or-
            # [1.15,1.61,2.17,2.67,3.23,5.00,10.90,16.2] -or-
            # [1.15 1.61 2.17 2.67 3.23 5.00 10.90 16.2]
            distances = str(Constants.get_batch_param(batch, "distances"))
            distances = (
                distances.replace("[", "")
                .replace("]", "")
                .replace(",", " ")
                .replace("  ", " ")
            )  # remove array chars: '[],'
            distances = np.fromstring(
                distances, sep=" "
            ).tolist()  # convert string to numpy array and then to a list
            normal_pts = [0.2, 0.4, 0.6, 0.8]

            # raw data
            xs = relative_time
            ys = dissipation

            self.update(status_label)

            # Computes initial difference cancelations for difference, resonance frequency
            # and dissipation and applies them to the UI curves.
            canceled_diss, canceled_diff, canceled_rf = None, None, None
            if self.parent.drop_effect_cancelation_checkbox.isChecked():
                canceled_diss, canceled_diff, canceled_rf = self.parent._correct_drop_effect(
                    self.loaded_datapath)
                if canceled_diss is not None:
                    ys = canceled_diss

            self.update(status_label)

            # use rough smoothing based on total runtime to figure start/stop
            total_runtime = xs[-1]
            smooth_factor = total_runtime * Constants.smooth_factor_ratio
            smooth_factor = int(smooth_factor) + (int(smooth_factor + 1) % 2)
            if smooth_factor < 3:
                smooth_factor = 3
            Log.i(TAG, f"Total run time: {total_runtime} secs")
            Log.d(
                TAG, f"Smoothing: {smooth_factor}"
            )  # the nearest odd number of seconds (runtime)
            Log.d(TAG, f"Applying smooth factor for first 90s ONLY.")

            t_first_90_split = (
                len(xs)
                if total_runtime <= 90
                else next(x for x, t in enumerate(xs) if t > 90)
            )
            extend_data = True if total_runtime > 90 else False
            extend_smf = int(
                smooth_factor / 20
            )  # downsample factor for extended data > 90s
            extend_smf += int(extend_smf + 1) % 2  # force to odd number

            if extend_data and len(xs) < t_first_90_split + 2 * extend_smf:
                Log.w(
                    "Not enough points after 90s to downsample effectively when analyzing. Not downsampling this dataset!"
                )
                t_first_90_split = len(xs)
                extend_data = False

            ys_fit = savgol_filter(ys[:t_first_90_split], smooth_factor, 1)
            if extend_data:
                ys_fit_ext = savgol_filter(
                    ys[t_first_90_split:],
                    min(len(ys[t_first_90_split:]), extend_smf),
                    1,
                )
                ys_fit = np.concatenate((ys_fit, ys_fit_ext))

            ys_diss_diff = savgol_filter(
                ys_fit[:t_first_90_split], smooth_factor, 1, 1)
            if extend_data:
                ys_diss_diff_ext = savgol_filter(
                    ys_fit[t_first_90_split:],
                    min(len(ys_fit[t_first_90_split:]), extend_smf),
                    1,
                    1,
                )
                ys_diss_diff = np.concatenate((ys_diss_diff, ys_diss_diff_ext))

            ys_diss_2ndd = savgol_filter(
                ys_diss_diff[:t_first_90_split], smooth_factor, 1, 1
            )
            if extend_data:
                ys_diss_2ndd_ext = savgol_filter(
                    ys_diss_diff[t_first_90_split:],
                    min(len(ys_diss_diff[t_first_90_split:]), extend_smf),
                    1,
                    1,
                )
                ys_diss_2ndd = np.concatenate((ys_diss_2ndd, ys_diss_2ndd_ext))

            ys_diss_diff_avg = np.average(
                ys_diss_diff
            )  # AJR TODO 4/14: pick up here, this line is too high for the 109cp run
            ys_diss_diff_offset = ys_diss_diff - ys_diss_diff_avg
            zeros3 = np.where(np.diff(np.sign(ys_diss_diff_offset)))[0]
            while len(zeros3) < 2:
                zeros3 = np.append(zeros3, 100)
            ys_diss_diff_avg = np.average(ys_diss_diff[zeros3[1]:])

            self.update(status_label)

            minima_idx = argrelextrema(ys_diss_2ndd, np.less)[0]
            minima_val = ys_diss_2ndd[minima_idx]
            minima_dict = {minima_idx[i]: minima_val[i]
                           for i in range(len(minima_idx))}
            minima_sort = sorted(minima_dict.items(), key=lambda kv: (kv[0]))

            maxima_idx = argrelextrema(ys_diss_2ndd, np.greater)[0]
            maxima_val = ys_diss_2ndd[maxima_idx]
            maxima_dict = {maxima_idx[i]: maxima_val[i]
                           for i in range(len(maxima_idx))}
            maxima_sort = sorted(maxima_dict.items(),
                                 key=lambda kv: (kv[1], kv[0]))

            self.update(status_label)

            start_stop = sorted(maxima_sort[-2:])
            start_stop = [start_stop[0][0], start_stop[1][0]]
            t_start = np.amin(start_stop)
            t_stop = np.amax(start_stop) + (3 * smooth_factor)
            if t_stop < len(xs) / 2 or t_stop >= len(xs):
                Log.d(
                    f"Warning: t_stop was {t_stop} out of {len(xs)} but that was deemed too big/small! (This can usually be ignored.)"
                )
                t_stop = len(xs) - 1
            if t_stop - t_start < len(xs) / 3 or t_start > len(xs) / 2:
                Log.d(
                    f"Warning: t_start was {t_start} out of {len(xs)} but that was deemed too big/small! (This can usually be ignored.)"
                )
                t_start = 100

            if total_runtime < 3:
                Log.e(
                    "ERROR: Data run must be at least 3 seconds in total runtime to analyze."
                )
                return

            self.update(status_label)

            # get indices for 0.5 seconds to start of run
            t_0p5 = (
                0
                if xs[t_start] < 0.5
                else next(x + 0 for x, t in enumerate(xs) if t > 0.5)
            )
            t_1p0 = (
                t_start
                if xs[t_start] < 2.0
                else next(x + 1 for x, t in enumerate(xs) if t > 2.0)
            )

            # new maths for resonance and dissipation (scaled)
            avg = np.average(resonance_frequency[t_0p5:t_1p0])
            ys = ys * avg / 2

            ys_fit = ys_fit * avg / 2
            ys = ys - np.amin(ys_fit)
            ys_fit = ys_fit - np.amin(ys_fit)
            ys_freq = avg - resonance_frequency
            # 'RF' Drop Effect Correction
            if self.parent.drop_effect_cancelation_checkbox.isChecked():
                if canceled_rf is not None:
                    ys_freq = avg - canceled_rf

            ys_freq_fit = savgol_filter(
                ys_freq[:t_first_90_split], smooth_factor, 1)
            if extend_data:
                ys_freq_fit_ext = savgol_filter(
                    ys_freq[t_first_90_split:],
                    min(len(ys_freq[t_first_90_split:]), extend_smf),
                    1,
                )
                ys_freq_fit = np.concatenate((ys_freq_fit, ys_freq_fit_ext))

            self.update(status_label)

            # # APPLY DROP EFFECT VECTORS
            # drop_offsets = np.zeros(ys.shape)
            # try:
            #     if self.parent.correct_drop_effect.isChecked():
            #         # baseline = np.average(ys[t_0p5:t_1p0])
            #         # base_std = np.std(ys[t_0p5:t_1p0])
            #         # next(x - 1 for x,y in enumerate(ys) if y > baseline + 4*base_std and x > t_1p0)
            #         drop_start = poi_vals[0]
            #         # next(ys[x + 2] for x,y in enumerate(ys) if y > Constants.drop_effect_cutoff_freq / 2 and x > t_1p0)
            #         drop_diss = ys[drop_start]
            #         if drop_diss > Constants.drop_effect_cutoff_freq:
            #             self.diff_factor = Constants.drop_effect_multiplier_high
            #         else:
            #             self.diff_factor = Constants.drop_effect_multiplier_low
            #         with open("QATCH/resources/lookup_drop_effect.csv", "r") as f:
            #             data = np.loadtxt(
            #                 f.readlines(), delimiter=",", skiprows=1)
            #             col = (
            #                 1
            #                 if self.diff_factor == Constants.drop_effect_multiplier_low
            #                 else 2
            #             )
            #             RR_offset = data[:, col]
            #             if drop_start + len(RR_offset) > len(drop_offsets):
            #                 # RR vector is longer than the actual run data, truncate it
            #                 drop_offsets[drop_start:] = RR_offset[
            #                     : len(drop_offsets) - drop_start
            #                 ]
            #             else:
            #                 # RR vector is shorter and needs to be padded with the final value
            #                 drop_offsets[drop_start: drop_start + len(RR_offset)] = (
            #                     RR_offset
            #                 )
            #                 drop_offsets[drop_start +
            #                              len(RR_offset):] = RR_offset[-1]
            #         Log.d(
            #             f"Applying vectors starting at time 't = {xs[drop_start]:1.3f}s'"
            #         )
            #         Log.d(
            #             f"Drop effect 'cutoff' dissipation frequency is {drop_diss:1.1f}Hz"
            #         )
            #         Log.d(
            #             f"Using {'low' if col == 1 else 'high'} viscosity drop effect 'diff_factor' and vector"
            #         )
            # except Exception as e:
            #     Log.e("ERROR:", e)

            # Automatically compute optimal difference factor
            if self.parent.difference_factor_optimizer_checkbox.isChecked():
                self.diff_factor = self.parent._optimize_curve(
                    self.loaded_datapath)

            baseline = np.average(dissipation[t_0p5:t_1p0])
            diff_factor = (
                Constants.default_diff_factor
            )  # 1.0 if baseline < 50e-6 else 1.5
            if hasattr(self, "diff_factor"):
                diff_factor = self.diff_factor
            ys_diff = ys_freq - (diff_factor * ys)

            # 'Difference' Drop Effect Correction
            if self.parent.drop_effect_cancelation_checkbox.isChecked():
                canceled_diss, canceled_diff, canceled_rf = self.parent._correct_drop_effect(
                    self.loaded_datapath)
                if canceled_diff is not None:
                    ys_diff = canceled_diff

            # Invert difference curve if drop applied to outlet
            if np.average(ys_diff) < 0:
                Log.w("Inverting DIFFERENCE curve due to negative initial fill deltas")
                ys_diff *= -1

            ys_diff_fit = savgol_filter(
                ys_diff[:t_first_90_split], smooth_factor, 1)
            if extend_data:
                ys_diff_fit_ext = savgol_filter(
                    ys_diff[t_first_90_split:],
                    min(len(ys_diff[t_first_90_split:]), extend_smf),
                    1,
                )
                ys_diff_fit = np.concatenate((ys_diff_fit, ys_diff_fit_ext))
            Log.d(f"Difference factor: {diff_factor:1.3f}x")

            self.update(status_label)

            smf = max(3, int(smooth_factor / 10))
            if smf % 2 == 0:
                smf += 1  # force odd number
            ys_diff_fine = savgol_filter(ys_diff, smf, 1)
            ys_diff_diff = savgol_filter(
                ys_diff, smf, 1, 1
            )  # difference derivatives, not dissipation
            ys_diff_2ndd = savgol_filter(ys_diff_diff, smf, 1, 1)

            self.update(status_label)

            # plt.ion()
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(2, 3, (1, 3))
            ax2 = fig.add_subplot(234)
            ax3 = fig.add_subplot(235)
            ax4 = fig.add_subplot(236)
            ax.set_title(f"Confirm: {data_title}")

            self.update(status_label)

            mask = np.arange(0, len(xs), 1)

            ax.legend(["Resonance", "Difference", "Dissipation"])

            d_avg = np.average(ys_diff[t_0p5:t_1p0])
            d_max = np.amax(ys_diff[t_0p5:t_1p0])
            d_min = np.amin(ys_diff[t_0p5:t_1p0])
            envelope_size = int(d_max - d_min)

            start_stop.clear()

            self.update(status_label)

            if len(poi_vals) > 0:
                t0 = poi_vals[0]
            t0_was = t0
            cw = max(10, int(smooth_factor / 10))  # context width
            while confirm_startIndex:
                ax2.cla()  # clear axis state without closing it
                ax3.cla()
                ax4.cla()
                ax2.plot(
                    xs[t0 - cw: t0 + cw],
                    ys_freq[t0 - cw: t0 + cw],
                    "g.",
                    label="freq",
                )
                ax2.scatter(xs[t0], ys_freq[t0], marker="*",
                            s=75, c="black", zorder=10)
                ax3.plot(
                    xs[t0 - cw: t0 + cw],
                    ys_diff[t0 - cw: t0 + cw],
                    "b.",
                    label="diff",
                )
                ax3.scatter(xs[t0], ys_diff[t0], marker="*",
                            s=75, c="black", zorder=10)
                ax4.plot(
                    xs[t0 - cw: t0 + cw], ys[t0 - cw: t0 + cw], "r.", label="diss"
                )
                ax4.scatter(xs[t0], ys[t0], marker="*",
                            s=75, c="black", zorder=10)
                t0, done = QtWidgets.QInputDialog.getDouble(
                    None, "Input Dialog", "Confirm precise start index:", value=t0
                )
                if t0.is_integer() and int(t0) in [-1] + list(
                    range(t0_was - cw, t0_was + cw)
                ):
                    t0 = int(t0)
                else:
                    try:
                        t0 = next(x for x, t in enumerate(xs) if t > t0)
                    except:
                        Log.d(
                            "Re-interpreting user input as an index, not a timestamp")
                        t0 = int(t0)
                if not done:
                    return
                ax2.cla()  # clear axis state without closing it
                ax3.cla()
                ax4.cla()
                if t0_was == t0:
                    break
                t0_was = t0
            start_stop.append(t0)

            self.update(status_label)

            if len(poi_vals) > 1:
                t1 = poi_vals[1]
            t1_was = t1
            cw = max(10, int(smooth_factor / 2))  # context width
            while confirm_stopIndex:
                ax2.cla()  # clear axis state without closing it
                ax3.cla()
                ax4.cla()
                ax2.plot(
                    xs[t1 - cw: t1 + cw],
                    ys_freq[t1 - cw: t1 + cw],
                    "g.",
                    label="freq",
                )
                ax2.scatter(xs[t1], ys_freq[t1], marker="*",
                            s=75, c="black", zorder=10)
                ax3.plot(
                    xs[t1 - cw: t1 + cw],
                    ys_diff[t1 - cw: t1 + cw],
                    "b.",
                    label="diff",
                )
                ax3.scatter(xs[t1], ys_diff[t1], marker="*",
                            s=75, c="black", zorder=10)
                ax4.plot(
                    xs[t1 - cw: t1 + cw], ys[t1 - cw: t1 + cw], "r.", label="diss"
                )
                ax4.scatter(xs[t1], ys[t1], marker="*",
                            s=75, c="black", zorder=10)
                t1, done = QtWidgets.QInputDialog.getDouble(
                    None, "Input Dialog", "Confirm precise stop index:", value=t1
                )
                if t1.is_integer() and int(t1) in [-1] + list(
                    range(t1_was - cw, t1_was + cw)
                ):
                    t1 = int(t1)
                else:
                    try:
                        t1 = next(x for x, t in enumerate(xs) if t > t1)
                    except:
                        Log.d(
                            "Re-interpreting user input as an index, not a timestamp")
                        t1 = int(t1)
                if not done:
                    return
                ax2.cla()  # clear axis state without closing it
                ax3.cla()
                ax4.cla()
                if t1_was == t1:
                    break
                t1_was = t1
            start_stop.append(t1)

            self.update(status_label)

            tp = t1 + 2
            if len(poi_vals) > 2:
                tp = poi_vals[2]
            tp_was = tp
            while confirm_postIndex:
                ax2.cla()  # clear axis state without closing it
                ax3.cla()
                ax4.cla()
                ax2.plot(
                    xs[tp - cw: tp + cw],
                    ys_freq[tp - cw: tp + cw],
                    "g.",
                    label="freq",
                )
                ax2.scatter(xs[tp], ys_freq[tp], marker="*",
                            s=75, c="black", zorder=10)
                ax3.plot(
                    xs[tp - cw: tp + cw],
                    ys_diff[tp - cw: tp + cw],
                    "b.",
                    label="diff",
                )
                ax3.scatter(xs[tp], ys_diff[tp], marker="*",
                            s=75, c="black", zorder=10)
                ax4.plot(
                    xs[tp - cw: tp + cw], ys[tp - cw: tp + cw], "r.", label="diss"
                )
                ax4.scatter(xs[tp], ys[tp], marker="*",
                            s=75, c="black", zorder=10)
                tp, done = QtWidgets.QInputDialog.getDouble(
                    None, "Input Dialog", "Confirm precise post index:", value=tp
                )
                if tp.is_integer() and int(tp) in [-1] + list(
                    range(tp_was - cw, tp_was + cw)
                ):
                    tp = int(tp)
                else:
                    try:
                        tp = next(x for x, t in enumerate(xs) if t > tp)
                    except:
                        Log.d(
                            "Re-interpreting user input as an index, not a timestamp")
                        tp = int(tp)
                if not done:
                    return
                ax2.cla()  # clear axis state without closing it
                ax3.cla()
                ax4.cla()
                if tp_was == tp:
                    break
                tp_was = tp

            self.update(status_label)

            # offset time by start point
            xs -= xs[t0]

            # zero all three datasets (and their fits) at start point
            ys_fit -= ys[t0]
            ys_freq_fit -= ys_freq[t0]
            ys_diff_fit -= ys_diff[t0]
            ys -= ys[t0]
            ys_freq -= ys_freq[t0]
            ys_diff -= ys_diff[t0]

            self.update(status_label)

            def monoLine(x, m, b):
                return m * x + b

            def monoCube(x, a, b, c):
                return a * (x**3) + b * (x**2) + c * x

            def monoCurve(x, a, b, c, d):
                return a * np.exp(b * x + c) + d

            # calculate normalized curve
            normal_x = xs[t0: t1 + 1]
            normal_y = ys_diff[t0: t1 + 1]
            n_max = np.amax(normal_y)
            n_min = np.amin(normal_y)

            self.update(status_label)

            if len(normal_y) <= 5:
                Log.w("Initial fill region contains too few points to apply smoothing.")

            sm1 = min(len(normal_y), max(5, int(len(normal_y) / 2)))
            if sm1 % 2 == 0:
                sm1 -= 1  # force odd number
            initial_fill = normal_y  # save for later plot
            initial_smooth = (
                savgol_filter(initial_fill, sm1,
                              1) if sm1 > 1 else initial_fill
            )

            # approximate linear fit
            n_slope = 1 / (normal_x[-1] - normal_x[0])
            n_offset = -n_slope * normal_x[0]

            self.update(status_label)

            p0 = (0, 0, n_slope)  # start with values near those we expect
            a, b, n_slope = p0  # default, not yet optimized
            best_fit_pts = normal_y  # default, not yet optimized
            try:
                fit_ignore = 0  # int((t1 - t0) / 4)
                params, cv = curve_fit(
                    monoCube, normal_x[fit_ignore:], normal_y[fit_ignore:], p0
                )
                a, b, n_slope = params
                best_fit_pts = monoCube(normal_x, a, b, n_slope)
                Log.d(f"Normalized fit coeffs: {params}")
            except:
                Log.w(
                    'Curve fit 1 failed to find optimal parameters for Figure 1 "Normalized" curve.'
                )
                Log.w("Using raw points in place of fit line.")

            self.update(status_label)

            Df = n_max - n_min

            # normalize both raw data and best fit points
            y_max = np.amax(normal_y)
            y_offset = np.amin(normal_y)
            normal_y = (normal_y - y_offset) / (y_max - y_offset)
            y_max = np.amax(best_fit_pts)
            y_offset = np.amin(best_fit_pts)
            best_fit_pts = (best_fit_pts - y_offset) / (y_max - y_offset)

            #############################################################
            ### TODO: THIS IS A "BAND-AID" IMPLEMENTATION - REMOVE IT ###
            ### PURPOSE: APPLY POLYNOMIAL CORRECTION TO INITIAL FILL  ###
            ###          WHEN FILLING TIME IS GREATER THAN 1 SECOND.  ###
            ### DATE ADDED: 2024-01-14                                ###
            #############################################################
            enable_bandaid_code = False  # Use to disable modified behavior
            line1_x = normal_x
            t_filling = line1_x[-1]
            Log.i(f"t_filling = {t_filling} secs")
            if enable_bandaid_code and t_filling > 1.0:  # t_filling > 1 sec
                Log.w(
                    "Applying polynomial correction to initial fill region (for long runs)"
                )
                line1_y = (
                    np.sqrt(np.polyval([0.183, 0.8234, 0],
                            normal_y)) * distances[0]
                )
                line1_y_fit = (
                    np.sqrt(np.polyval([0.183, 0.8234, 0],
                            best_fit_pts)) * distances[0]
                )
            else:
                line1_y = np.sqrt(normal_y) * distances[0]
                line1_y_fit = np.sqrt(best_fit_pts) * distances[0]
            line1_y[np.isnan(line1_y)] = 0
            line1_y_fit[np.isnan(line1_y_fit)] = 0
            ### END OF CODE BLOCK: "BAND-AID" IMPLEMENTATION ############

            self.update(status_label)

            line1_smooth = savgol_filter(
                line1_y, sm1, 3) if sm1 > 3 else line1_y
            line1_smooth[0] = 0  # force first value to zero
            mask = np.where(line1_smooth < 0)
            line1_smooth[mask] = 0

            self.update(status_label)

            # start with values near those we expect
            p0 = (-1, -1, 1, distances[0])
            a, b, c, d = p0  # default, not yet optimized
            line1_curve = line1_y  # default, not yet optimized
            try:
                fit_ignore = 0  # int((t1 - t0) / 4)
                params, cv = curve_fit(
                    monoCurve, line1_x[fit_ignore:], line1_y[fit_ignore:], p0
                )
                a, b, c, d = params
                line1_curve = monoCurve(line1_x, a, b, c, d)
            except:
                Log.w(
                    'Curve fit 2 failed to find optimal parameters for Figure 1 "Position" curve.'
                )
                Log.w("Using raw points in place of fit line.")

            self.update(status_label)

            x_fit_pts = 0
            for x in range(len(line1_x)):
                if line1_curve[x] < line1_y[x]:
                    x_fit_pts = x + 1
                    break
            x_fit_val = np.linspace(0, line1_curve[x_fit_pts], x_fit_pts)
            for x in range(x_fit_pts - 1):
                line1_curve[x] = x_fit_val[x]

            self.update(status_label)

            # normalize endpoint to max of 1.15mm
            y_ratio = np.amax(line1_smooth) / distances[0]
            line1_y /= y_ratio
            line1_smooth /= y_ratio
            a_max = np.amax(line1_curve)
            line1_curve -= np.amin(line1_curve)
            line1_curve *= a_max / np.amax(line1_curve)
            y_ratio = np.amax(line1_curve) / distances[0]
            line1_curve /= y_ratio

            self.update(status_label)

            Log.d(TAG, f"Df = {Df}")
            Log.d(TAG, f"# pts = {t1+1-t0}")

            # search for locations of blips @ 5.6mm, 11.3mm, 15.7mm
            times = []

            # find zero2 crossings
            zeros2 = np.where(np.diff(np.sign(ys_diss_2ndd)))[0]

            # find zero3 crossings
            ys_diss_diff_offset = ys_diss_diff - ys_diss_diff_avg
            zeros3 = np.where(np.diff(np.sign(ys_diss_diff_offset)))[0]
            Log.d(zeros3)

            self.update(status_label)

            # define rough blip zones
            # t0 = next(t for t in zeros if t > t0) # first zero crossing to right of max value
            # t3r = t0 + np.argmin(ys_diss_diff[t0:])
            t0 = (
                t1  # t1 is from different context, refers to end of initial fill period
            )
            t3 = t_stop
            td = int((t3 - t0) / 3)
            t1 = t0 + td
            t2 = t1 + td
            t0r = float(t0)
            t1r = float(t1)
            t2r = float(t2)
            t3r = float(t3)

            self.update(status_label)

            # search for precise blips
            blips = [1, 2, 3]
            range_list = []
            # Log.d(zeros2)
            for key, val in maxima_sort[::-1]:  # iterate from big to small
                this_max = key
                try:
                    this_min = next(t for t, y in minima_sort if t > key)
                    this_zero = next(t for t in zeros2 if t > key)
                except:
                    this_min = len(ys_diss_2ndd) - 1
                    this_zero = len(ys_diss_2ndd) - 2
                if this_max > this_zero or this_min < this_zero:
                    if ys_diss_2ndd[this_max] > 0 and ys_diss_2ndd[this_min] < 0:
                        # Log.w("Warning: ")
                        Log.d(
                            TAG,
                            "Something is off! The maxima MUST come before the minima, with the zero crossing in between. (This can usually be ignored.)",
                        )
                    continue
                x_range = this_min - this_max
                y_range = ys_diss_2ndd[this_max] - ys_diss_2ndd[this_min]
                # Log.d((this_max, this_zero, this_min), xs[key], ys_diss_2ndd[this_max], ys_diss_2ndd[this_min], y_range)
                zone = -1
                if key > t0r and key < t1r:
                    zone = 1
                if key > t1r and key < t2r:
                    zone = 2
                if key > t2r and key <= t3r:
                    zone = 3
                range_list.append((this_zero, zone, x_range, y_range))
            range_sort = sorted(range_list, key=lambda kv: (kv[3], kv[1]))

            self.update(status_label)

            # Log.d(range_sort)
            # iterate from big to small
            for key, zone, x, y in range_sort[::-1]:
                # Log.d(xs[key], zone, y)
                if zone in blips:
                    blips.remove(zone)
                    # Log.d("Using:", zone, key, xs[key])
                    if zone == 1:
                        t1 = key
                    if zone == 2:
                        t2 = key
                    if zone == 3:
                        t3 = key
                        t3r = min(len(xs) - 1, int(key + x))
                if len(blips) == 0:
                    break

            self.update(status_label)

            # overload blips for new method (test)
            idx = 0
            t_num = 0
            t_len = 0
            t_minima = []
            t_size = []
            while True:
                if len(zeros3) > idx + 1:
                    mid_val = ys_diss_diff_offset[
                        int((zeros3[idx] + zeros3[idx + 1]) / 2)
                    ]
                    min_pt = zeros3[idx] + np.argmin(
                        ys_diss_diff_offset[zeros3[idx]: zeros3[idx + 1]]
                    )
                    this_size = zeros3[idx + 1] - zeros3[idx]
                    idx += 1
                    if mid_val > ys_diss_diff_offset[zeros3[idx]]:
                        continue  # skip if this interval is a maximum, not a minimum
                    t_minima.append(min_pt)
                    t_size.append(this_size)
                else:
                    break

            self.update(status_label)

            v_minima = []
            v_size = []
            for i in range(len(t_minima)):
                if t_size[i] > t_len:
                    v_minima.append(t_minima[i])
                    v_size.append(t_size[i])
                    t_len = t_size[i]

            self.update(status_label)

            if len(v_minima) == 3:
                t1 = int((v_minima[0] + v_minima[1]) / 2)
                t2 = v_minima[2]
            if len(v_minima) == 2:
                t1 = v_minima[0]
                t2 = v_minima[1]
            if len(v_minima) == 1:
                t1 = v_minima[0]
                t2 = t_minima[-1] + \
                    np.argmin(ys_diss_diff_offset[t_minima[-1]:])

            self.update(status_label)

            np.asarray(t_minima)

            ax2.plot(xs[zeros3[0]:], ys_diss_diff_offset[zeros3[0]:], "b:")
            ax2.plot(xs[t_minima], ys_diss_diff_offset[t_minima], "rx")
            ax2.plot(xs[t1], ys_diss_diff_offset[t1], "gx")
            ax2.plot(xs[t2], ys_diss_diff_offset[t2], "gx")
            ax2.axhline(y=0)  # ys_diss_diff_avg)

            self.update(status_label)

            if len(poi_vals) > 5:
                # write prior blips
                times.append(poi_vals[3])
                times.append(poi_vals[4])
                times.append(poi_vals[5])
            else:
                # write found blips
                times.append(t1)
                times.append(t2)
                times.append(t3r)
            Log.d(times)

            self.update(status_label)

            bounds = [int(t0r), int(t1r), int(t2r), int(t3r)]
            for b in range(len(times)):
                # confirm blips (if desired)
                time = times[b]
                time_was = time
                cw = max(10, int(smooth_factor * 3))  # context width
                while confirm_blipIndices:
                    ax.cla()  # clear axis state without closing it
                    num_points = int((bounds[b + 1] - bounds[b]) / 2)
                    mask = np.arange(
                        max(0, time - num_points),
                        min(len(xs) - 1, time + num_points),
                        1,
                    ).astype(
                        int
                    )  # keep centered in wide-context window
                    time = int(time)
                    Log.d(mask)
                    ax.plot(
                        xs[mask], ys_freq_fit[mask], ":", color="green", label="fit"
                    )
                    ax.plot(xs[mask], ys_diff_fit[mask],
                            ":", color="blue", label="fit")
                    ax.plot(xs[mask], ys_fit[mask], ":",
                            color="red", label="fit")
                    ax.plot(xs[mask], ys_freq[mask], "g,", label="freq")
                    ax.scatter(
                        xs[time],
                        ys_freq_fit[time],
                        marker="*",
                        s=75,
                        c="black",
                        zorder=10,
                    )
                    ax.plot(xs[mask], ys_diff[mask], "b,", label="diff")
                    ax.scatter(
                        xs[time],
                        ys_diff_fit[time],
                        marker="*",
                        s=75,
                        c="black",
                        zorder=10,
                    )
                    ax.plot(xs[mask], ys[mask], "r,", label="diss")
                    ax.scatter(
                        xs[time], ys_fit[time], marker="*", s=75, c="black", zorder=10
                    )
                    ax.legend(["Resonance", "Difference", "Dissipation"])
                    ax2.cla()  # clear axis state without closing it
                    ax3.cla()
                    ax4.cla()
                    ax2.plot(
                        xs[time - cw: time + cw],
                        ys_freq[time - cw: time + cw],
                        "g,",
                        label="freq",
                    )
                    ax2.plot(
                        xs[time - cw: time + cw],
                        ys_freq_fit[time - cw: time + cw],
                        "g.",
                        label="freq",
                    )
                    ax2.scatter(
                        xs[time],
                        ys_freq_fit[time],
                        marker="*",
                        s=75,
                        c="black",
                        zorder=10,
                    )
                    ax3.plot(
                        xs[time - cw: time + cw],
                        ys_diff[time - cw: time + cw],
                        "b,",
                        label="diff",
                    )
                    ax3.plot(
                        xs[time - cw: time + cw],
                        ys_diff_fit[time - cw: time + cw],
                        "b.",
                        label="diff",
                    )
                    ax3.scatter(
                        xs[time],
                        ys_diff_fit[time],
                        marker="*",
                        s=75,
                        c="black",
                        zorder=10,
                    )
                    ax4.plot(
                        xs[time - cw: time + cw],
                        ys[time - cw: time + cw],
                        "r,",
                        label="diss",
                    )
                    ax4.plot(
                        xs[time - cw: time + cw],
                        ys_fit[time - cw: time + cw],
                        "r.",
                        label="diss",
                    )
                    ax4.scatter(
                        xs[time], ys_fit[time], marker="*", s=75, c="black", zorder=10
                    )
                    time, done = QtWidgets.QInputDialog.getDouble(
                        None,
                        "Input Dialog",
                        f"Confirm precise blip_{b+1} index:",
                        value=time,
                    )
                    if time.is_integer():
                        time = int(time)
                    else:
                        try:
                            time = next(
                                x for x, t in enumerate(xs) if t > time)
                        except:
                            Log.d(
                                "Re-interpreting user input as an index, not a timestamp"
                            )
                            time = int(time)
                    if not done:
                        return
                    ax2.cla()  # clear axis state without closing it
                    ax3.cla()
                    ax4.cla()
                    if time_was == time:
                        break
                    time_was = time
                times[b] = time

                self.update(status_label)

            # display and export selected points
            points_of_interest = np.concatenate((start_stop, [tp], times))
            Log.d(f"Points of interest (index only): {points_of_interest}")
            Log.i(
                f"Points of interest: {points_of_interest} {xs[points_of_interest]}")
            np.savetxt(poi_path, points_of_interest, fmt="%i")

            self.update(status_label)

            # pop blips if user input -1
            while True:
                if times[-1] == -1:
                    times.pop(-1)
                else:
                    break

            # Log.d(times)
            # Log.d("times:", xs[times]) #-xs[times[0]])

            self.update(status_label)

            t_start = max(0, np.amin(start_stop) - 50)
            t_stop = min(len(xs), np.amax(times) + 50)

            self.update(status_label)

            ax.cla()  # clear axis state without closing it
            ax2.cla()
            ax3.cla()
            ax4.cla()
            ax.set_title(f"Raw Data: {data_title}")
            ax.grid(axis="y", which="major")
            mask = np.arange(t_start, t_stop, 1)
            ax.plot(xs[mask], ys_freq_fit[mask], "--",
                    color="green", label="freq fit")
            ax.plot(xs[mask], ys_diff_fit[mask], "--",
                    color="blue", label="diff fit")
            ax.plot(xs[mask], ys_fit[mask], "--",
                    color="red", label="diss fit")

            self.update(status_label)

            ax.plot(xs[mask], ys_freq[mask], "g,", label="freq")
            ax.plot(xs[mask], ys[mask], "r,", label="diss")
            ax.plot(xs[mask], ys_diff[mask], "b,", label="diff")

            # ax.plot(xs[mask], ys_diff_fine[mask], ":", color="blue", label="diff fine")
            # ax.plot(xs[mask], ys_diff_diff[mask], ':', color="orange", label="diss diff")
            # ax.plot(xs[mask], ys_diff_2ndd[mask], ':', color="red", label="freq fit")
            # ax.plot([0, xs[times[-1]]], [0, 0], "-", color="black", markersize=0)

            ax.plot(xs[start_stop], ys_diff[start_stop], "d", color="black")
            ax.plot(xs[times], ys_fit[times], "d", color="black")
            ax.legend(["Resonance", "Difference", "Dissipation"])

            ax2.plot(normal_x, initial_fill, "r.", label="init")
            ax2.plot(normal_x, initial_smooth, "-", label="fit")
            leg = ax2.legend(
                ["Initial Fill"], handlelength=0, handletextpad=0, fancybox=True
            )
            for item in leg.legend_handles:
                item.set_visible(False)

            self.update(status_label)

            mask = np.where(normal_y >= 0)
            ax3.plot(normal_x[mask], normal_y[mask], "r.", label="normal")
            ax3.plot(normal_x, best_fit_pts, "-", label="fit")
            leg = ax3.legend(
                ["Normalized"], handlelength=0, handletextpad=0, fancybox=True
            )
            for item in leg.legend_handles:
                item.set_visible(False)

            mask = np.where(line1_y >= 0)
            ax4.plot(line1_x[mask], line1_y[mask], "r.", label="line1")
            ax4.plot(line1_x, line1_y_fit, "-", label="curve")
            # ax4.plot(line1_x, line1_smooth, ':', label="fit")
            leg = ax4.legend(
                ["Position"], handlelength=0, handletextpad=0, fancybox=True
            )
            for item in leg.legend_handles:
                item.set_visible(False)

            self.update(status_label)

            # show final constructed distance vs time curve
            times.append(t0)  # overloaded: end of inital fill
            times.sort()

            # insert midpoints into "times" array (to match length of "distances" array)
            Log.i("(The following midpoints are shown as blue Xs on Figure 1):")

            if len(times) >= 2:
                midpoint_ch1_y = (ys_fit[times[1]] + ys_fit[times[0]]) / 2
                midpoint_ch1_i = next(
                    x for x, y in enumerate(ys_fit) if y > midpoint_ch1_y
                )
                midpoint_ch1_x = xs[midpoint_ch1_i]
                Log.i(
                    f"1st channel dissipation midpoint = {midpoint_ch1_y:2.2f} Hz @ {midpoint_ch1_x:2.2f} secs"
                )
                # ax.plot(midpoint_ch1_x, midpoint_ch1_y, "bd")
                # times.append(midpoint_ch1_i)
            else:
                Log.w(
                    "1st channel midpoint not available from dataset. Confirm Precise Fill Points 3 and 4 for accuracy."
                )
                Log.w(
                    "See Figure 2 to check if one of these points is being dropped due to time delta not being 2x last."
                )

            self.update(status_label)

            if len(times) >= 4:
                midpoint_ch2_y = (
                    ys_freq_fit[times[2]] + ys_freq_fit[times[1]]) / 2
                midpoint_ch2_i = next(
                    x for x, y in enumerate(ys_freq_fit) if y > midpoint_ch2_y
                )
                midpoint_ch2_x = xs[midpoint_ch2_i]
                Log.i(
                    f"2nd channel frequency midpoint = {midpoint_ch2_y:2.2f} Hz @ {midpoint_ch2_x:2.2f} secs"
                )
                # ax.plot(midpoint_ch2_x, midpoint_ch2_y, "bd")  # MIDP2
                # times.append(midpoint_ch2_i)                   # mid2
            else:
                Log.w(
                    "2nd channel midpoint not available from dataset. Confirm Precise Fill Points 4 and 5 for accuracy."
                )
                Log.w(
                    "See Figure 2 to check if one of these points is being dropped due to time delta not being 2x last."
                )

            self.update(status_label)

            if len(times) >= 6:
                midpoint_ch3_y = (
                    ys_freq_fit[times[3]] + ys_freq_fit[times[2]]) / 2
                midpoint_ch3_i = next(
                    x for x, y in enumerate(ys_freq_fit) if y > midpoint_ch3_y
                )
                midpoint_ch3_x = xs[midpoint_ch3_i]
                Log.i(
                    f"3rd channel frequency midpoint = {midpoint_ch3_y:2.2f} Hz @ {midpoint_ch3_x:2.2f} secs"
                )
                # ax.plot(midpoint_ch3_x, midpoint_ch3_y, "bd")  # MIDP3
                # times.append(midpoint_ch3_i)                   # mid3
            else:
                Log.w(
                    "3rd channel midpoint not available from dataset. Confirm Precise Fill Points 5 and 6 for accuracy."
                )
                Log.w(
                    "See Figure 2 to check if one of these points is being dropped due to time delta not being 2x last."
                )

            self.update(status_label)

            fine_smf = int(
                smooth_factor / 2.5
            )  # downsample factor for extended data > 90s
            fine_smf += int(fine_smf + 1) % 2  # force to odd number
            ys_fit_fine = savgol_filter(ys, fine_smf, 1)

            # fine fit points: smoothed, but only a little for accurate fast fill
            ys_normal = ys_fit_fine - ys_fit_fine[tp]
            ys_normal /= ys_fit_fine[times[1]] - ys_fit_fine[tp]
            ys_normal = ys_normal[tp: times[1]]

            debug = False
            if debug:
                import matplotlib.pyplot as plt

                fig_dbg = plt.figure(figsize=(12, 9))
                ax_dbg = fig_dbg.add_subplot(111)

                # raw points
                xs_dbg = xs[tp: times[1]]
                ys_normal2 = ys - ys_fit_fine[tp]
                ys_normal2 /= ys_fit_fine[times[1]] - ys_fit_fine[tp]
                ys_normal2 = ys_normal2[tp: times[1]]

                Log.i("ys_normal: ")
                Log.i(ys_normal)
                ax_dbg.plot(xs_dbg, ys_normal, color="red", marker=",")
                ax_dbg.plot(xs_dbg, ys_normal2, color="green", marker=",")
                # ax_dbg.plot(xs, ys_fit_fine, color="blue", marker=",")

                Log.i("times:")
                Log.i(times)
                Log.i(xs[times])
                ax_dbg.plot(xs[times[0]], 0, color="black", marker="X")
                ax_dbg.plot(xs[times[1]], 1, color="black", marker="X")

                fig_dbg.show()

            idx_of_normal_pts_to_remove = []
            for p in normal_pts:
                midpoint_p_i = next(
                    x for x, y in enumerate(ys_normal) if y >= p) + tp
                midpoint_p_x = xs[midpoint_p_i]
                midpoint_p_y = ys_fit[midpoint_p_i]
                Log.i(
                    f"1st channel dissipation @ {p:0.1f} = {midpoint_p_y:2.2f} Hz @ {midpoint_p_x:2.2f} secs"
                )
                ax.plot(
                    midpoint_p_x, midpoint_p_y, color="blue", marker="d", markersize=4
                )
                if debug:
                    ax_dbg.plot(midpoint_p_x, p, color="blue", marker="X")
                times.append(midpoint_p_i)
                if p == 0.2 or p == 0.4:
                    idx_of_normal_pts_to_remove.append(midpoint_p_i)
            times.sort()  # sort again, so midpoints are in proper order

            self.update(status_label)

            last_window_size = 0
            last_x = 0
            bad_idx = []
            bad_times = []
            bad_distances = []
            for x in range(1, len(times)):
                this_window_size = xs[times[x]] - xs[times[last_x]]
                # Log.e(f"Compare {times[x]} to {len(xs)-1}...")
                if (
                    this_window_size < 0.75 * last_window_size
                    or times[x] == len(xs) - 1
                ):
                    bad_x = x
                    if bad_x == 5:  # trust channel 1 pt more than estimated 80% point
                        bad_x = 4
                    Log.w(
                        f"Point {bad_x} @ {xs[times[bad_x]]}s is 'bad' and will be ignored!"
                    )
                    bad_idx.append(bad_x)
                    bad_times.append(times[bad_x])
                    bad_distances.append(distances[bad_x])
                if (
                    x == 5
                ):  # set ch1 window size compared to start, not estimated points
                    this_window_size = xs[times[x]] - xs[times[0]]
                last_window_size = this_window_size
                last_x = x

            self.update(status_label)

            # hide bad points from Figure 3 and Figure 4
            for x in bad_idx[::-1]:
                times.pop(x)
                distances.pop(x)

            np.asarray(times)
            np.asarray(bad_times)
            distances = np.asarray(distances)[0: len(times)]

            self.update(status_label)

            ext_line1_x = np.linspace(0, xs[times[-1]], 1000)
            ext_index = np.concatenate(([start_stop[0]], times))
            ext_times = np.concatenate(([0], xs[times]))
            ext_dists = np.concatenate(([0], distances))
            all_times = np.sort(
                np.concatenate([[points_of_interest[0]], times, bad_times])
            ).astype(int)
            Log.i("times and distances:")
            Log.d("indexes: {}".format(ext_index))
            Log.i(ext_times)
            Log.i(ext_dists)
            ext_line1_curve = np.interp(ext_line1_x, ext_times, ext_dists)

            self.update(status_label)

            fig2 = plt.figure(figsize=(12, 6))
            ax5 = fig2.add_subplot(111)
            ax5.plot(line1_x[0], 0, "d", color="black")
            ax5.plot(ext_line1_x, ext_line1_curve, ":", color="orange")
            for i in range(len(distances)):
                if not distances[i] in bad_distances:
                    ax5.plot(xs[times[i]], distances[i], "d", color="black")
            ax5.plot(xs[bad_times], bad_distances, "x", color="red")
            ax5.set_title(f"Position: {data_title}")
            ax5.set_xlabel("Time (s)")
            ax5.set_ylabel("Position (mm)")

            self.update(status_label)

            norm_fit_xs = xs[start_stop[1]: times[-1]]
            norm_fit_dists = ys_fit[start_stop[1]: times[-1]]
            norm_fit_dists -= norm_fit_dists[0]
            norm_fit_dists /= norm_fit_dists[-1]
            norm_fit_dists *= distances[-1] - distances[0]
            norm_fit_dists += distances[0]

            # Generate log() plot curves for velocity and position^-1
            log_velocity = np.concatenate(
                (line1_y / line1_x, distances / xs[times]))
            log_position = np.concatenate((1 / line1_y, 1 / distances))

            raw_velocity = norm_fit_dists / norm_fit_xs
            raw_position = 1 / norm_fit_dists

            self.update(status_label)

            log_velocity = np.log10(log_velocity)
            log_position = np.log10(log_position)

            raw_velocity = np.log10(raw_velocity)
            raw_position = np.log10(raw_position)

            self.update(status_label)

            log_velocity[np.isnan(log_velocity)] = 0
            log_position[np.isnan(log_position)] = 0

            raw_velocity[np.isnan(raw_velocity)] = 0
            raw_position[np.isnan(raw_position)] = 0

            log_velocity[np.isinf(log_velocity)] = 0
            log_position[np.isinf(log_position)] = 0

            raw_velocity[np.isinf(raw_velocity)] = 0
            raw_position[np.isinf(raw_position)] = 0

            # fit_ignore = next(x for x,y in enumerate(log_velocity) if y > 0)
            # log_velocity = log_velocity[fit_ignore:]
            # log_position = log_position[fit_ignore:]

            self.update(status_label)

            dropUnder5 = 0
            ####################################
            # NEW CODE for 2023-11-07 TESTING:
            # Drop 5 Hz or 3.33% of initial fill (whichever is larger)
            # default, if/when Band-Aid #2 disabled
            dropBelowPct = float(1 / 30)
            #######################################
            # Band-Aid #2: Drop more initial fill
            # To disable: Comment out line below:
            dropBelowPct = 0.10  # 0.15 if BIOFORMULATION else 0.40
            Log.w(
                f"Dropping {int(dropBelowPct * 100)}% of initial fill region...")
            ### END Band-Aid #2 ###################
            dropFreqBelow = max(5, initial_fill[-1] * dropBelowPct)
            ### END NEW CODE ###################
            for i in range(len(initial_fill)):
                if initial_fill[i] > dropFreqBelow:
                    Log.d(f"Dropped {i} initial samples under 5 Hz threshold")
                    log_velocity = log_velocity[i:]
                    log_position = log_position[i:]
                    dropUnder5 = i
                    ####################################
                    # NEW CODE for 02/03/2023 TESTING:
                    # REMOVED 2023-11-07:
                    # if initial_fill[-1] > 1.1*DENSITY*300:
                    #     dropUnder5 = int(np.floor(len(initial_fill)/5))
                    # else:
                    #     dropUnder5 = i
                    ### END NEW CODE ###################
                    Log.d(f"dropUnder5 = {dropUnder5}")
                    break

            self.update(status_label)

            log_velocity_46 = log_velocity
            log_position_46 = log_position
            Log.d(f"log_velocity = {log_velocity}")
            Log.d(f"initial_fill = {initial_fill}")
            if initial_fill[-1] < 90:
                log_velocity_46 = log_velocity_46[-len(distances):]
                log_position_46 = log_position_46[-len(distances):]
                cp = xs[all_times[FILL_IDX]] / 2
                ax2.annotate("Not Analyzed", (cp, 0), ha="center")
                ax3.annotate("Not Analyzed", (cp, 0), ha="center")
                ax4.annotate("Not Analyzed", (cp, 0), ha="center")
            else:
                for i in range(len(initial_fill)):
                    if initial_fill[i] > 46:
                        Log.d(
                            f"Dropped {i} initial samples under 46 Hz threshold")
                        log_velocity_46 = log_velocity_46[i:]
                        log_position_46 = log_position_46[i:]
                        break

            self.update(status_label)

            def reject_outliers(data, m):
                d = np.abs(data - np.median(data))
                mdev = np.median(d)
                s = d / mdev if mdev else 0.0
                # Log.d(f"reject_s = {s}")
                return s < m

            self.update(status_label)

            keep_ids = reject_outliers(log_velocity[0: -len(distances)], 11.0)
            if isinstance(keep_ids, bool):
                keep_ids = [keep_ids]
            trues = [True for x in distances]
            keep_ids = np.concatenate((keep_ids, trues))
            log_velocity_skip = log_velocity[~keep_ids]
            log_position_skip = log_position[~keep_ids]
            log_velocity = log_velocity[keep_ids]
            log_position = log_position[keep_ids]
            Log.d(
                f"Rejected {len(log_position_skip)} samples as initial outliers: {log_position_skip}"
            )

            self.update(status_label)

            p0 = (1, 0)  # start with values near those we expect
            n_slope, n_offset = p0  # default, not yet optimized
            best_fit_pts = log_position_46  # default, not yet optimized
            try:
                params, cv = curve_fit(
                    monoLine, log_velocity_46, log_position_46, p0)
                n_slope, n_offset = params
                best_fit_pts = monoLine(log_velocity_46, n_slope, n_offset)
            except:
                Log.w(
                    'Curve fit 3 failed to find optimal parameters for Figure 3 "slope" fit.'
                )
                Log.w('Using raw points in place of fit line (assuming "slope = 1").')

            n_rounded = max(
                0.05, min(1, round((n_slope + 0.05) * 20) / 20)
            )  # round up to nearest 0.05, max of 1
            n = n_rounded

            self.update(status_label)

            ####################################
            # NEW CODE for 2022-12-06 TESTING:
            m = len(initial_fill) - (len(log_velocity) - len(log_velocity_46))
            mlen = int(np.floor(m / 5))
            log_velocity_20p = []
            log_position_20p = []
            for hh in range(0, mlen):
                log_velocity_20p.append(log_velocity_46[hh])
                log_position_20p.append(log_position_46[hh])
            for hh in range(1, 5):
                log_velocity_20p.append(
                    np.average(log_velocity_46[hh * mlen: (hh + 1) * mlen - 1])
                )
                log_position_20p.append(
                    np.average(log_position_46[hh * mlen: (hh + 1) * mlen - 1])
                )
                # Log.d(f"idx = {(hh+1)*mlen-1}, max = {m-1}")
            ### END NEW CODE ###################

            self.update(status_label)

            fig3 = plt.figure(figsize=(12, 6))
            ax6 = fig3.add_subplot(111)
            ax6.plot(log_velocity_20p, log_position_20p, ".", color="red")
            ax6.plot(log_velocity_46, log_position_46, ":", color="orange")
            ax6.plot(log_velocity_46, best_fit_pts, "-", color="blue")
            try:
                for i in range(-len(distances), 0):
                    ax6.plot(
                        log_velocity_46[i], log_position_46[i], "d", color="black")
                ax6.set_title(
                    f"Power log coefficient: {data_title}\nn = {n:.2f}"
                    + r"$ \pm $"
                    + "0.05"
                )
            except:
                Log.e(TAG, "An error occurred while annotating Figure 3")
            ax6.set_xlabel("Log(velocity) (mm/s)")
            ax6.set_ylabel("Log(1/position) (1/mm)")

            self.update(status_label)

            Log.d("the distances were:", distances)
            Log.d("the times were:", times)
            Log.d("the times to remove are:", idx_of_normal_pts_to_remove)
            for i in idx_of_normal_pts_to_remove:
                try:
                    idx = times.index(i)
                    Log.d(
                        f"Removing index {idx} from distances with value {distances[idx]}."
                    )
                    distances = np.delete(distances, idx)
                    Log.d(f"Removing index {idx} from times with value {i}.")
                    times.remove(i)
                except Exception as e:
                    Log.e("Error removing midpoint from dataset:", str(e))
            Log.d("the distances are now:", distances)
            Log.d("the times are now:", times)

            all_pos = np.concatenate((line1_y[dropUnder5:], distances))
            all_time = np.concatenate((line1_x[dropUnder5:], xs[times]))
            all_temp = np.concatenate(
                (temperature[t0: t0 + len(line1_x[dropUnder5:])],
                 temperature[times])
            )
            avg_temp = np.average(temperature[t0: times[-1]])
            all_velocity = all_pos / all_time
            # all_velocity[-7] /= 2 # 1.61 (#2 in distances)
            # all_velocity[-6] /= 1.5 # 2.17 (#3 in distances)
            # all_velocity[-5] /= 1 # 2.67 (#4 in distances)

            fill_pos = line1_y_fit[dropUnder5:]
            fill_time = line1_x[dropUnder5:]
            fill_velocity = fill_pos / fill_time

            self.update(status_label)

            Log.d(f"Channel thickness = {Constants.channel_thickness}")
            viscosity = (
                ST
                * np.cos(np.radians(CA))
                * all_time
                * Constants.channel_thickness
                / 6
                / (all_pos**2)
                * 1e6
                * (3 * (n + 1) / (2 * n + 1))
            )
            shear_rate = (
                6
                * all_velocity
                / Constants.channel_thickness
                * (2 / 3 + 1 / 3 / n)
                * 1e-3
                / (n + 1)
                * n
            )

            fill_visc = (
                ST
                * np.cos(np.radians(CA))
                * fill_time
                * Constants.channel_thickness
                / 6
                / (fill_pos**2)
                * 1e6
                * (3 * (n + 1) / (2 * n + 1))
            )
            fill_shear = (
                6
                * fill_velocity
                / Constants.channel_thickness
                * (2 / 3 + 1 / 3 / n)
                * 1e-3
                / (n + 1)
                * n
            )

            self.update(status_label)

            fig4 = plt.figure(figsize=(12, 6))
            ax7 = fig4.add_subplot(111)

            high_shear_5x = 0
            high_shear_15x = 0

            self.update(status_label)

            if len(all_times) >= 6:
                high_shear_15x = 15e6
                f0 = ys_freq[all_times[FILL_IDX]]
                d0 = dissipation[all_times[FILL_IDX]]
                f2 = ys_freq[all_times[BLIP1_IDX]]
                d2 = dissipation[all_times[BLIP1_IDX]]
                Log.i(f"f0 = {f0:2.2f} Hz")
                Log.i(f"f2 = {f2:2.2f} Hz")
                Log.i(f"f2-f0 = {f2-f0} Hz")

                self.update(status_label)

                if f2 - f0 > float(
                    Constants.get_batch_param(batch, "freq_delta_15MHz")
                ):
                    freq_factor_15MHz = float(
                        Constants.get_batch_param(batch, "freq_factor_15MHz")
                    )
                    high_shear_15y = (
                        ((f2 - f0) * freq_factor_15MHz) ** 2) / DENSITY
                    Log.i(
                        f"15MHz High shear = ((f2-f0) * {freq_factor_15MHz})^2 / {DENSITY} = {high_shear_15y:2.2f} cP"
                    )
                else:
                    diss_factor1_15MHz = float(
                        Constants.get_batch_param(batch, "diss_factor1_15MHz")
                    )
                    diss_factor2_15MHz = float(
                        Constants.get_batch_param(batch, "diss_factor2_15MHz")
                    )
                    bandaid_compensate_high_shear_viscosity = False
                    if bandaid_compensate_high_shear_viscosity:
                        E3 = (
                            ys_freq[all_times[FILL_IDX]] -
                            ys_freq[all_times[START_IDX]]
                        )  # from CAL file (Freq_fill)
                        D = (d2 - d0) - \
                            ((0.023112 * (E3) / DENSITY - 4.6868) * 1e-6)
                        high_shear_15y = (
                            (D * diss_factor1_15MHz - diss_factor2_15MHz) ** 2
                        ) / DENSITY
                    else:
                        high_shear_15y = (
                            ((d2 - d0) * diss_factor1_15MHz -
                             diss_factor2_15MHz) ** 2
                        ) / DENSITY
                    Log.i(f"d0 = {d0:1.4E}")
                    Log.i(f"d2 = {d2:1.4E}")
                    Log.i(f"d2-d0 = {d2-d0:1.4E}")
                    if bandaid_compensate_high_shear_viscosity:
                        Log.i(f"E3 = {E3}")
                        Log.i(f"D = {D}")
                        Log.i(
                            f"15MHz High shear = ({D} * {diss_factor1_15MHz}-{diss_factor2_15MHz})^2 / {DENSITY} = {high_shear_15y:2.2f} cP"
                        )
                    else:
                        Log.i(
                            f"15MHz High shear = ((d2-d0) * {diss_factor1_15MHz}-{diss_factor2_15MHz})^2 / {DENSITY} = {high_shear_15y:2.2f} cP"
                        )
                high_shear_15x = self.correctHighShear(
                    high_shear_15x, high_shear_15y)
                ax7.plot(high_shear_15x, high_shear_15y, "bd")
                ax7.errorbar(
                    high_shear_15x,
                    high_shear_15y,
                    0.30 * high_shear_15y,
                    fmt="b.",
                    ecolor="blue",
                    capsize=3,
                )

                self.update(status_label)

                if True:
                    data_path_fun = data_path.replace("_3rd.csv", "_lower.csv")
                    fun_file_exists = secure_open.file_exists(
                        data_path_fun, "capture")

                if (
                    f2 - f0 < 900 and fun_file_exists
                ):  # frequency check added 2023-02-01
                    if True:
                        with secure_open(data_path_fun, "r", "capture") as f:
                            csv_headers_fun = next(f)

                            if isinstance(csv_headers_fun, bytes):
                                csv_headers_fun = csv_headers_fun.decode()

                            if "Ambient" in csv_headers_fun:
                                csv_cols_fun = (2, 4, 6, 7)
                            else:
                                csv_cols_fun = (2, 3, 5, 6)

                            data_fun = loadtxt(
                                f.readlines(),
                                delimiter=",",
                                skiprows=0,
                                usecols=csv_cols_fun,
                            )

                    self.update(status_label)

                    relative_time_fun = data_fun[:, 0]
                    temperature_fun = data_fun[:, 1]
                    resonance_frequency_fun = data_fun[:, 2]
                    dissipation_fun = data_fun[:, 3]

                    self.update(status_label)

                    Log.i("Analyzing fundamental frequency dataset...")
                    times_fun = []
                    for i in range(len(all_times)):
                        t_fun = 0
                        try:
                            t_fun = (
                                next(
                                    x
                                    for x, t in enumerate(relative_time_fun)
                                    if t >= xs[all_times[i]] - xs[0]
                                )
                                - 1
                            )
                        except:
                            Log.e(
                                f"Failed to locate POI_{i} @ timestamp {xs[all_times[i]] - xs[0]} from fundamental dataset. Attempting to proceed with index 0..."
                            )
                        Log.d(
                            f"time[{i}] must be >= {xs[all_times[i]] - xs[0]}")
                        Log.d(
                            f"time[{i}] = {relative_time_fun[t_fun]}, index {t_fun}")
                        times_fun.append(t_fun)
                    ys_freq_fun = (
                        np.average(
                            resonance_frequency_fun[0: times_fun[FILL_IDX]])
                        - resonance_frequency_fun
                    )
                    high_shear_5x = 5e6
                    xp = relative_time_fun
                    fp = ys_freq_fun
                    t0 = (
                        xs[all_times[FILL_IDX]] - xs[0]
                    )  # absolute time of 15MHz start idx
                    t2 = (
                        xs[all_times[BLIP1_IDX]] - xs[0]
                    )  # absolute time of 15MHz blip1 idx
                    f0 = np.interp(t0, xp, fp)
                    d0 = dissipation_fun[10]
                    f2 = np.interp(t2, xp, fp)
                    d2 = dissipation_fun[times_fun[BLIP1_IDX]]
                    Log.d(f"fun values to interpolate: [{t0}, {t2}]")
                    Log.d(f"{0}: ({xp[0]}, {fp[0]})")
                    Log.d("...")
                    for i in range(len(xp)):
                        if xp[i - 1] <= t0 and xp[i] >= t0:
                            Log.d(f"{i-1}: ({xp[i-1]}, {fp[i-1]})")
                            Log.d(f"## INTERP t0 HERE: ({t0}, {f0})")
                            Log.d(f"{i+1}: ({xp[i+1]}, {fp[i+1]})")
                            Log.d("...")
                        if xp[i - 1] <= t2 and xp[i] >= t2:
                            Log.d(f"{i-1}: ({xp[i-1]}, {fp[i-1]})")
                            Log.d(f"## INTERP t2 HERE: ({t2}, {f2})")
                            Log.d(f"{i+1}: ({xp[i+1]}, {fp[i+1]})")
                            Log.d("...")
                        # Log.d(f"{i}: ({xp[i]}, {fp[i]})")
                    # ending 'i' from last 'for' loop
                    Log.d(f"{i}: ({xp[i]}, {fp[i]})")
                    Log.i(f"f0 = {f0:2.2f} Hz")
                    Log.i(f"f2 = {f2:2.2f} Hz")
                    Log.i(f"f2-f0 = {f2-f0} Hz")
                    if f2 - f0 > float(
                        Constants.get_batch_param(batch, "freq_delta_5MHz")
                    ):
                        freq_factor_5MHz = float(
                            Constants.get_batch_param(
                                batch, "freq_factor_5MHz")
                        )
                        high_shear_5y = (
                            ((f2 - f0) * freq_factor_5MHz) ** 2) / DENSITY
                        Log.i(
                            f"5MHz High shear = ((f2-f0) * {freq_factor_5MHz})^2 / {DENSITY} = {high_shear_5y:2.2f} cP"
                        )
                    else:
                        diss_factor1_5MHz = float(
                            Constants.get_batch_param(
                                batch, "diss_factor1_5MHz")
                        )
                        diss_factor2_5MHz = float(
                            Constants.get_batch_param(
                                batch, "diss_factor2_5MHz")
                        )
                        high_shear_5y = (
                            ((d2 - d0) * diss_factor1_5MHz - diss_factor2_5MHz) ** 2
                        ) / DENSITY
                        Log.i(f"d0 = {d0:1.4E}")
                        Log.i(f"d2 = {d2:1.4E}")
                        Log.i(f"d2-d0 = {d2-d0:1.4E}")
                        Log.i(
                            f"5MHz High shear = ((d2-d0) * {diss_factor1_5MHz}-{diss_factor2_5MHz})^2 / {DENSITY} = {high_shear_5y:2.2f} cP"
                        )
                    high_shear_5x = self.correctHighShear(
                        high_shear_5x, high_shear_5y)
                    ax7.plot(high_shear_5x, high_shear_5y, "bd")
                    ax7.errorbar(
                        high_shear_5x,
                        high_shear_5y,
                        0.30 * high_shear_5y,
                        fmt="b.",
                        ecolor="blue",
                        capsize=3,
                    )
                else:
                    Log.w("5 MHz high-shear calculation not available from dataset.")
                    if not fun_file_exists:
                        Log.w(
                            "The 5 MHz mode does not exist in the dataset for this captured run."
                        )
                    else:
                        Log.w(
                            "The frequency shift of the initial fill region is too small (<900 Hz) for high-shear calculation accuracy."
                        )
            else:
                Log.w("5 MHz high-shear calculation not available from dataset.")
                Log.w("15 MHz high-shear calculation not available from dataset.")

                Log.w(
                    "Too few valid time points are available in Figure 2 for any high-shear calculation accuracy."
                )
                Log.w(
                    "See Figure 2 to check if any of these points is being dropped due to time delta not being 2x last."
                )
                Log.w(
                    "If so, please adjust the Precise Fill Points for this run accordingly and try this analysis again."
                )

            self.update(status_label)

            # viscosity = ST*np.cos(np.radians(CA))*all_time*Constants.channel_thickness/6/(all_pos**2)*1e3*(3*(n+1)/(2*n+1))
            # viscosity = viscosity * 1000

            # viscosity_2 = ST*np.cos(np.radians(CA))*line1_x*Constants.channel_thickness/6/(line1_curve**2)*1e3*(3*(n+1)/(2*n+1))
            # viscosity_2 = viscosity_2 * 1000

            # keep_ids = reject_outliers(viscosity, 11.)
            # out_shear_rate = shear_rate[~keep_ids]
            # out_viscosity = viscosity[~keep_ids]
            in_shear_rate = shear_rate  # [keep_ids]
            in_viscosity = viscosity  # [keep_ids]
            in_temp = all_temp
            # outliers = np.where(keep_ids == False)
            # Log.d(f"in_visc = {viscosity}")
            # Log.d(f"outliers = {outliers}")

            self.update(status_label)

            if len(in_shear_rate) == 0 or len(in_viscosity) == 0:
                in_shear_rate = shear_rate
                in_viscosity = viscosity
                in_temp = all_temp
                Log.w(
                    "WARN: Initial fill region contains nothing but outlier. Attempting to continue with outliers."
                )
                Log.w("Please check the first 2 points of interest for accuracy.")

            # remove NANs from datasets
            to_remove = np.isnan(in_shear_rate)
            to_remove |= np.isnan(in_viscosity)
            to_remove |= np.isinf(in_shear_rate)
            to_remove |= np.isinf(in_viscosity)
            in_shear_rate = in_shear_rate[~to_remove]
            in_viscosity = in_viscosity[~to_remove]
            in_temp = in_temp[~to_remove]

            self.update(status_label)

            if len(in_shear_rate) == 0 or len(in_viscosity) == 0:
                in_shear_rate = shear_rate
                in_viscosity = viscosity
                in_temp = all_temp
                Log.w(
                    "WARN: Initial fill region contains nothing but inf/nan. Attempting to continue with outliers."
                )
                Log.w("Please check the first 2 points of interest for accuracy.")

            viscosity_at_1p15 = viscosity[-len(distances)]

            try:
                idx0 = -6
                idx1 = -5
                idx2 = -4
                idx3 = -3
                avg_viscosity = np.average(
                    [in_viscosity[idx0], in_viscosity[idx3]])
                std_viscosity = np.std(
                    np.delete(in_viscosity, [idx1, idx2])
                )  # all of in_viscosity, just not 2 points
                min_visc = min(np.min(avg_viscosity),
                               np.min(fill_visc)) - std_viscosity
                max_visc = max(np.max(avg_viscosity),
                               np.max(fill_visc)) + std_viscosity
                Log.d(
                    f"Expected viscosity = {avg_viscosity} +/- {std_viscosity}, min = {min_visc}, max = {max_visc}"
                )
                Log.d("Indices 0-3 are:", [idx0, idx1, idx2, idx3])
                for i in range(idx1, idx3):
                    if min_visc <= viscosity[i] <= max_visc:
                        continue
                    Log.d(
                        f"Removed point '{viscosity[i]}' for being outside the standard deviation of expected viscosity."
                    )
                    in_shear_rate = np.delete(in_shear_rate, i)
                    in_viscosity = np.delete(in_viscosity, i)
                    in_temp = np.delete(in_temp, i)
                    viscosity = np.delete(viscosity, i)
                    shear_rate = np.delete(shear_rate, i)
                    fill_visc = np.delete(fill_visc, i)
                    fill_shear = np.delete(fill_shear, i)
            except Exception as e:
                Log.e("ERROR:", e)
                Log.e("Unable to remove outliers from the dataset prior to plotting.")

            self.update(status_label)

            # np.linspace(lin_shear_rate[0], lin_shear_rate[-1]) # default: 50 points
            fit_shear = fill_shear
            fit_visc = (
                # cube_fit(fit_shear) # plot this one, evenly spaced points
                fill_visc
            )
            lin_viscosity = fill_visc

            debug = False
            if debug:
                raw_shear = in_shear_rate[0: -len(distances)]
                raw_visc = in_viscosity[0: -len(distances)]
                import matplotlib.pyplot as plt

                fig_dbg = plt.figure(figsize=(12, 9))
                ax_dbg = fig_dbg.add_subplot(111)
                # ax_dbg.scatter(raw_shear_out, raw_visc_out, color="red", marker="x")
                ax_dbg.scatter(raw_shear, raw_visc, color="blue", marker=".")
                ax_dbg.plot(fit_shear, fit_visc, color="black", marker=",")
                fig_dbg.show()

            self.update(status_label)

            # ax7.annotate("START", (shear_rate[0],viscosity[0]),
            #    textcoords="offset points", xytext=(0,-15), ha='center')
            # ax7.plot(out_shear_rate, out_viscosity, 'rx')
            # ax7.plot(shear_rate, viscosity, 'r:')

            ### BANDAID #3 ###
            # PURPOSE: Hide initial fill points when trending in the wrong direction of high-shear
            enable_bandaid_3 = True
            hide_initial_fill = False  # if disabled, never force hide initial fill
            remove_initial_fill = False
            point_factor_limit = 0.25
            if point_factor_limit < 0 or point_factor_limit > 1:
                Log.e(
                    f"Invalid 'point_factor_limit' set: {point_factor_limit:2.2f} (Must be between zero and one)"
                )
                Log.e(
                    "Disabling initial fill limit check due to invalid parameter specified: 'point_factor_limit'"
                )
                enable_bandaid_3 = False
            if enable_bandaid_3:
                P1_value = fit_visc[-1]
                P2_value = high_shear_15y
                lower_factor = 1 - point_factor_limit
                upper_factor = 1 + point_factor_limit
                min_fit_end = min(P1_value, P2_value) * lower_factor
                max_fit_end = max(P1_value, P2_value) * upper_factor
                Log.d(
                    f"Point Factor Limit for Initial Fill is: {point_factor_limit:2.2f}x"
                )
                Log.d(
                    f"Trendline must be within range from {min_fit_end:2.2f} to {max_fit_end:2.2f}"
                )
                Log.d(f"Initial Fill Trendline ends at: {fit_visc[0]:2.2f}")
                if (
                    min_fit_end > fit_visc[0] or max_fit_end < fit_visc[0]
                ):  # Point 2 (right) is less than Point 1 (left)
                    Log.w(
                        f"Dropping initial fill region due to being outside of the accepted limits (see Debug for more info)"
                    )
                    hide_initial_fill = True
            ##################

            if initial_fill[-1] >= 90 and not hide_initial_fill:
                mlen = int(np.floor((len(in_shear_rate) - len(distances)) / 5))
                ax7.scatter(
                    in_shear_rate[:mlen],
                    in_viscosity[:mlen],
                    marker=".",
                    s=15,
                    c="blue",
                )
                ax7.scatter(
                    in_shear_rate[mlen:], in_viscosity[mlen:], marker="d", s=1, c="blue"
                )
                ax7.plot(fit_shear, fit_visc, color="black", marker=",")
                for hh in range(1, 5):
                    xp = np.average(
                        in_shear_rate[hh * mlen: (hh + 1) * mlen - 1])
                    yp = np.average(
                        in_viscosity[hh * mlen: (hh + 1) * mlen - 1])
                    stdev = np.std(
                        in_viscosity[hh * mlen: (hh + 1) * mlen - 1])
                    # ax7.plot(xp, yp, 'b.')
                    ax7.errorbar(xp, yp, stdev, fmt="b.",
                                 ecolor="blue", capsize=3)
            else:
                # Remove initial fill points from output table later
                remove_initial_fill = True

            self.update(status_label)

            avg_viscosity = np.average(in_viscosity)
            std_viscosity = np.std(in_viscosity)
            # lin_viscosity = np.flip(lin_viscosity)
            for i in range(-len(distances), 0):
                percent_error = (
                    abs(
                        (viscosity[i] - viscosity[-len(distances)])
                        / viscosity[-len(distances)]
                    )
                    * 100
                )
                Log.d(
                    f"Percent error for calculated viscosity is: {percent_error}")
                if percent_error < 20.0:
                    ax7.plot(
                        shear_rate[i], viscosity[i], "bd"
                    )  # show bigly if it's not an outlier
                else:
                    ax7.plot(
                        shear_rate[i], viscosity[i], "bd"
                    )  # show bigly even if it is (for now)

                if lin_viscosity[-1] == viscosity[i] and i == -len(distances):
                    continue  # skip adding the first viscosity point if it is a duplicate
                lin_viscosity = np.append(lin_viscosity, viscosity[i])
            if len(lin_viscosity) == len(in_shear_rate) - 1:
                # re-add the skipped viscosity point if the lengths do not match
                lin_viscosity = np.insert(
                    lin_viscosity, -len(distances), viscosity[-len(distances)]
                )

            if remove_initial_fill:
                # Remove initial fill points from output table
                in_shear_rate = in_shear_rate[-len(distances):]
                in_viscosity = in_viscosity[-len(distances):]
                lin_viscosity = lin_viscosity[-len(distances):]
                in_temp = in_temp[-len(distances):]

            in_shear_rate = np.flip(in_shear_rate)
            in_viscosity = np.flip(in_viscosity)
            lin_viscosity = np.flip(lin_viscosity)
            in_temp = np.flip(in_temp)
            if high_shear_5x != 0:
                in_shear_rate = np.append(in_shear_rate, high_shear_5x)
                in_viscosity = np.append(in_viscosity, high_shear_5y)
                lin_viscosity = np.append(lin_viscosity, high_shear_5y)
                in_temp = np.append(in_temp, avg_temp)
            if high_shear_15x != 0:
                in_shear_rate = np.append(in_shear_rate, high_shear_15x)
                in_viscosity = np.append(in_viscosity, high_shear_15y)
                lin_viscosity = np.append(lin_viscosity, high_shear_15y)
                in_temp = np.append(in_temp, avg_temp)

            self.update(status_label)

            ax7.set_title(f"Shear-rate vs. Viscosity: {data_title}")
            ax7.set_xlabel("Shear-rate (1/s)")
            ax7.set_ylabel("Viscosity (cP)")
            lower_limit = np.amin(in_viscosity) / 1.5
            power = 1
            while power > -5:
                if lower_limit > 10**power:
                    lower_limit = 10**power
                    break
                power -= 1
            upper_limit = np.amax(in_viscosity) * 1.5
            power = 0
            while power < 5:
                if upper_limit < 10**power:
                    upper_limit = 10**power
                    break
                power += 1
            if lower_limit >= upper_limit:
                Log.w(
                    "Limits were auto-calculated but are in an invalid range! Using ylim [0, 1000]."
                )
                ax7.set_ylim([0, 1000])
            elif np.isfinite(lower_limit) and np.isfinite(upper_limit):
                Log.d(
                    f"Auto-calculated y-range limits for Figure 4 are: [{lower_limit}, {upper_limit}]"
                )
                ax7.set_ylim([lower_limit, upper_limit])
            else:
                Log.w(
                    "Limits were auto-calculated but were not finite values! Using ylim [0, 1000]."
                )
                ax7.set_ylim([0, 1000])

            ax7.set_xscale("log")
            ax7.set_yscale("log")

            self.update(status_label)

            err_viscosity = []
            str_viscosity = []
            for i in range(len(in_shear_rate)):
                err_viscosity.append(in_viscosity[i] * 0.10)
                str_viscosity.append(
                    f"{lin_viscosity[i]:2.2f} \u00b1 {err_viscosity[i]:2.2f}"
                )  # plus-or-minus = \u00b1

            try:
                # export data to csv
                export_path = data_path
                export_path = export_path.replace(
                    ".csv", Constants.export_file_format)
                export_path = export_path.replace("_fundamental", "")
                export_path = export_path.replace("_3rd", "")
                Log.i(f"Exporting analyze output to:\n\t{export_path}")
                np.savetxt(
                    export_path,
                    np.column_stack(
                        [
                            in_shear_rate,
                            in_viscosity,
                            lin_viscosity,
                            err_viscosity,
                            in_temp,
                        ]
                    ),
                    fmt="%.2f",
                    delimiter=",",
                    header="shear_rate,viscosity_raw,viscosity_avg,percent_error,temperature",
                )
            except Exception as e:
                Log.e("Error generating output file: " + str(e))
                if not os.path.exists(export_path):
                    with open(export_path, "w") as f:
                        f.write(str(e))
                # raise e # Debug only!

            self.update(status_label)

            # Add a footnote below and to the right side of the chart
            footnote = "Generated by {} {} ({}) at {}."
            for axis in [ax4, ax5, ax6, ax7]:
                axis.annotate(
                    footnote.format(
                        Constants.app_title,
                        Constants.app_version,
                        Constants.app_date,
                        strftime("%Y-%m-%d %I:%M:%S %p", localtime()),
                    ),
                    xy=(0.5, 1),
                    xycoords=("figure fraction", "figure pixels"),
                    ha="center",
                    va="bottom",
                    color="dimgray",
                    fontsize=8,
                )

            status_label = "Saving Results..."
            self.update(status_label)

            # def export_figures(fig1, fig2, fig3, fig4):
            # export figures to file
            Log.i("Saving figures to file...")
            # self.progress.emit(75, "Saving Results...")
            Log.i(
                f'Exporting Figure 1 to:\n\t{export_path.replace(".csv", "_1.pdf")}')
            fig.savefig(export_path.replace(".csv", "_1.pdf"))
            self.update(status_label)
            # self.progress.emit(80, "Saving Results...")
            Log.i(
                f'Exporting Figure 2 to:\n\t{export_path.replace(".csv", "_2.pdf")}')
            fig2.savefig(export_path.replace(".csv", "_2.pdf"))
            self.update(status_label)
            # self.progress.emit(85, "Saving Results...")
            Log.i(
                f'Exporting Figure 3 to:\n\t{export_path.replace(".csv", "_3.pdf")}')
            fig3.savefig(export_path.replace(".csv", "_3.pdf"))
            self.update(status_label)
            # self.progress.emit(90, "Saving Results...")
            Log.i(
                f'Exporting Figure 4 to:\n\t{export_path.replace(".csv", "_4.pdf")}')
            fig4.savefig(export_path.replace(".csv", "_4.pdf"))
            self.update(status_label)

            enabled, error, expires = UserProfiles.checkDevMode()
            if enabled == False and (error == True or expires != ""):
                PopUp.warning(
                    self,
                    "Developer Mode Expired",
                    "Developer Mode has expired and these analysis results will now be encrypted.\n"
                    + 'An admin must renew or disable "Developer Mode" to suppress this warning.',
                )

            self.update(status_label)

            # Generate CAL file if dev mode is enabled, and not expired
            try:
                if enabled:

                    Log.i("Generating CAL file in output folder for manual analysis...")
                    cal_pts = np.array(
                        [
                            "start",
                            "fill",
                            "20%",
                            "40%",
                            "60%",
                            "80%",
                            "ch1",
                            "ch2",
                            "ch3",
                        ],
                        dtype=str,
                    )
                    cal_idxs = np.unique(
                        np.sort(
                            np.concatenate(
                                [
                                    [points_of_interest[0]],
                                    times,
                                    bad_times,
                                    idx_of_normal_pts_to_remove,
                                ]
                            )
                        ).astype(int)
                    )
                    na_val = len(xs) - 1
                    na_pts = bad_times.count(na_val)
                    if na_pts > 1:
                        for i in range(na_pts - 1):
                            cal_idxs = np.append(cal_idxs, na_val)
                    cal_times = np.array(
                        np.round(xs[cal_idxs], 4), dtype=float)
                    cal_disss = np.array(
                        np.round(ys[cal_idxs] - ys[points_of_interest[0]], 4),
                        dtype=float,
                    )
                    cal_freqs = np.array(
                        np.round(ys_freq[cal_idxs] -
                                 ys_freq[points_of_interest[0]], 4),
                        dtype=float,
                    )
                    cal_notes = [
                        "Not Analyzed" if x in bad_times else "" for x in cal_idxs
                    ]
                    cal_data = np.column_stack(
                        [cal_pts, cal_idxs, cal_times,
                            cal_disss, cal_freqs, cal_notes]
                    )

                    np.savetxt(
                        cal_path,
                        cal_data,
                        delimiter=",",
                        fmt="%s",
                        header="Point,Index,Time,Diss,Freq,Notes",
                    )

                    Log.i("Successfully generated CAL file: " + cal_path)

            except Exception as e:
                Log.e("Error generating CAL file: " + str(e))
                if not os.path.exists(cal_path):
                    with open(cal_path, "w") as f:
                        f.write(str(e))
                # raise e # Debug only!

            self.update(status_label)

            # Move generated PDFs (and CSVs) to secure ZIP folder
            Log.i(f"Compressing exported files to ZIP...")
            QtCore.QCoreApplication.processEvents()

            num = 1
            fullpath = os.path.split(export_path)[0]
            folder = os.path.split(fullpath)[1]
            last_zn = None
            while True:
                zn = os.path.join(fullpath, f"analyze-{num}.zip")
                if os.path.exists(zn):
                    last_zn = zn
                    num += 1
                else:
                    break
            with pyzipper.AESZipFile(
                zn,
                "w",
                compression=pyzipper.ZIP_DEFLATED,
                allowZip64=True,
                encryption=pyzipper.WZ_AES,
            ) as zf:
                # Add a protected file to the zip archive
                friendly_name = f"{folder} ({dt.date.today()})"
                zf.comment = friendly_name.encode()  # run name
                if (
                    False and UserProfiles.count() > 0 and enabled == False
                ):  # NEVER do this (is this still CFR compliant?)
                    # create a protected archive
                    zf.setpassword(hashlib.sha256(
                        zf.comment).hexdigest().encode())
                else:
                    zf.setencryption(None)
                    if enabled:
                        Log.w("Developer Mode is ENABLED - NOT encrypting ZIP file")

                copy_file = poi_path
                zf.write(copy_file, arcname=os.path.split(copy_file)[1])
                # os.remove(copy_file) # do not remove POIs file

                copy_file = export_path
                zf.write(copy_file, arcname=os.path.split(copy_file)[1])
                os.remove(copy_file)

                copy_file = export_path.replace(".csv", "_1.pdf")
                zf.write(copy_file, arcname=os.path.split(copy_file)[1])
                os.remove(copy_file)

                copy_file = export_path.replace(".csv", "_2.pdf")
                zf.write(copy_file, arcname=os.path.split(copy_file)[1])
                os.remove(copy_file)

                copy_file = export_path.replace(".csv", "_3.pdf")
                zf.write(copy_file, arcname=os.path.split(copy_file)[1])
                os.remove(copy_file)

                copy_file = export_path.replace(".csv", "_4.pdf")
                zf.write(copy_file, arcname=os.path.split(copy_file)[1])
                os.remove(copy_file)

                this_poi_csv_crc = str(
                    hex(zf.getinfo(os.path.split(poi_path)[1]).CRC))
                this_out_csv_crc = str(
                    hex(zf.getinfo(os.path.split(export_path)[1]).CRC)
                )
                this_files_count = len(zf.namelist())

                Log.i(f"Compressed exported files to ZIP:\n\t{zn}")

            self.update(status_label)

            if last_zn != None and self.parent.option_remove_dups.isChecked():
                Log.d("Checking for duplicate analysis output file...")
                with pyzipper.AESZipFile(
                    last_zn,
                    "r",
                    compression=pyzipper.ZIP_DEFLATED,
                    allowZip64=True,
                    encryption=pyzipper.WZ_AES,
                ) as zf:
                    try:
                        last_poi_csv_crc = str(
                            hex(zf.getinfo(os.path.split(poi_path)[1]).CRC)
                        )
                        last_out_csv_crc = str(
                            hex(zf.getinfo(os.path.split(export_path)[1]).CRC)
                        )
                        last_files_count = len(zf.namelist())
                    except Exception as e:
                        Log.w(f"Error checking prior archive: {str(e)}")
                        last_poi_csv_crc = 0
                        last_out_csv_crc = 0
                        last_files_count = 0

                    Log.d(f"Last poi.csv CRC: {last_poi_csv_crc}")
                    Log.d(f"This poi.csv CRC: {this_poi_csv_crc}")
                    Log.d(f"Last out.csv CRC: {last_out_csv_crc}")
                    Log.d(f"This out.csv CRC: {this_out_csv_crc}")
                    Log.d(f"Last files count: {last_files_count}")
                    Log.d(f"This files count: {this_files_count}")

                    if (
                        last_poi_csv_crc == this_poi_csv_crc
                        and last_out_csv_crc == this_out_csv_crc
                        and last_files_count == this_files_count
                    ):
                        Log.w("Removing duplicate analysis output file.")
                        Log.w(f"See prior analyze output file:\n\t{last_zn}")
                        os.remove(zn)  # duplicate

            # self.progress.emit(99, "Showing Results...")
            status_label = "Showing Results..."
            self.update(status_label)
            Log.i("\tDONE!")
            # sub = threading.Thread(target=export_figures, args=(fig,fig2,fig3,fig4,))
            # sub.start()
            # sub.join()

            i = 0
            Log.d("Waiting for progressbar...")
            while (
                self.parent.progress_value_scanning and i < 300
            ):  # wait for progressBar to reach 99%
                i += 1
                QtCore.QCoreApplication.processEvents()

            Log.i("Showing results...")
            # sleep(1)

            self.parent.results_split.setEnabled(True)
            # self.parent.widget_h4.setStyleSheet("background-color: #ffffff; color: #515151")

            if len(log_velocity_46) == len(distances):  # not checked
                Log.w(
                    "WARNING: Initial fill values are not considered to be reliably accurate for this run."
                )
                Log.w(
                    "Initial fill values are marked as 'light red' in the tabular data for reference only."
                )

                in_shear_rate = np.array(in_shear_rate, dtype=str)
                in_viscosity = np.array(in_viscosity, dtype=str)
                # str_viscosity = np.array(str_viscosity, dtype=str) # NOTE: Numpy doesn't handle unicode chars in array strings
                in_temp = np.array(in_temp, dtype=str)

                pts_to_modify = range(len(distances), len(in_shear_rate))
                for i in range(len(in_shear_rate)):
                    is_error_cell = i in pts_to_modify
                    if in_shear_rate[i] in [str(5e6), str(15e6)]:
                        is_error_cell = False

                    if is_error_cell:
                        # Log.i(f"Converting {in_shear_rate[i]}")
                        # Log.i(f"into *{in_shear_rate[i]:2.2f}*")
                        in_shear_rate[i] = f"*{float(in_shear_rate[i]):2.2f}*"
                        in_viscosity[i] = f"*{float(in_viscosity[i]):2.2f}*"
                        str_viscosity[i] = f"*{str_viscosity[i]}*"
                        in_temp[i] = f"*{float(in_temp[i]):2.2f}*"
                    else:
                        in_shear_rate[i] = f"{float(in_shear_rate[i]):2.2f}"
                        in_viscosity[i] = f"{float(in_viscosity[i]):2.2f}"
                        # str_viscosity[i] = f"{str_viscosity[i]}"
                        in_temp[i] = f"{float(in_temp[i]):2.2f}"

                in_shear_rate = in_shear_rate.tolist()
                in_viscosity = in_viscosity.tolist()
                # str_viscosity = str_viscosity.tolist()
                in_temp = in_temp.tolist()

            # add data to table view of results
            data = {
                "Shear Rate (1/s)": in_shear_rate,
                "Raw Viscosity (cP)": in_viscosity,
                "Avg Viscosity (cP)": str_viscosity,
                "Temperature (C)": in_temp,
            }
            rows = len(in_shear_rate)
            cols = 4
            # data, rows, cols = [{"col1": ["Hello", "This"], "col2": ["World", "Is"], "col3": ["Foo", "A"], "col4": ["Bar", "Test"]}, 2, 4]
            tableWidget = TableView(data, rows, cols)
            # tableWidget.setStyleSheet("QScrollBar:vertical { width: 15px; }")
            # tableWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            self.parent.results_split.replaceWidget(0, tableWidget)
            self.parent.results_split.setSizes(
                self.parent.get_results_split_auto_sizes()
            )

            # add figure to plot view of results
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)
            sc = FigureCanvasQTAgg(fig4)
            mp_toolbar = NavigationToolbar(sc, self.parent)
            mp_layout = QtWidgets.QVBoxLayout()
            mp_layout.addWidget(mp_toolbar)
            mp_layout.addWidget(sc)
            plotWidget = QtWidgets.QWidget()
            plotWidget.setLayout(mp_layout)
            self.parent.results_split.replaceWidget(1, plotWidget)

            # all_figs = [fig4] # [fig,fig2,fig3,fig4]
            # for f in all_figs:
            #     f.show()
            #     mngr = f.canvas.manager     # plt.get_current_fig_manager()
            #     mngr.window.showMaximized() # setGeometry(7, 30, 1503, 742)

            self.progress.emit(
                100, "Finishing..."
            )  # will change to "Progress: Finished" once finished() handlers fire
            Log.i("Analyze process finished.")
            self._exitSuccess = True

        except:
            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        finally:
            self.finished.emit()  # queue callback

    def update(self, status):
        try:
            from inspect import currentframe, getframeinfo

            frameinfo = getframeinfo(currentframe().f_back)
            # print(frameinfo.filename, frameinfo.lineno)
            start = self.run.__code__.co_firstlineno
            stop = self.update.__code__.co_firstlineno
            pct = 100 * (frameinfo.lineno - start) / (stop - start)
            # Log.i(f"line #: {start}, {frameinfo.lineno}, {stop}, {pct}%")
            self.progress.emit(int(pct), status)

        except:
            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

    def correctHighShear(self, initial, visc):
        output = initial
        try:
            with open("QATCH/resources/lookup_shear_correction.csv", "r") as f:
                data = np.loadtxt(f.readlines(), delimiter=",", skiprows=1)
                col = 1 if initial == 5e6 else 2
                lookup_visc = data[:, 0]
                lookup_freq = data[:, col]
                nearest_idx = (np.abs(lookup_visc - visc)).argmin()
                correction_factor = lookup_freq[nearest_idx]
                output *= correction_factor
        except Exception as e:
            Log.e("ERROR:", e)
        return np.round(output, 2)


class TableView(QtWidgets.QTableWidget):

    def __init__(self, data, *args):
        QtWidgets.QTableWidget.__init__(self, *args)
        self.setData(data)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def setData(self, data):
        self.data = data
        self.clear()
        horHeaders = []
        for n, key in enumerate(self.data.keys()):
            horHeaders.append(key)
            for m, item in enumerate(self.data[key]):
                error_cell = False
                if str(item).startswith("*") and str(item).endswith("*"):
                    item = str(item)[1:-1]
                    error_cell = True
                if item == str(item):
                    newitem = QtWidgets.QTableWidgetItem(item)
                else:
                    newitem = QtWidgets.QTableWidgetItem(f"{item:2.2f}")
                newitem.setFlags(QtCore.Qt.ItemIsEnabled)
                if error_cell:
                    newitem.setForeground(QtGui.QBrush(
                        QtGui.QColor(255, 127, 127)))
                self.setItem(m, n, newitem)
        self.setHorizontalHeaderLabels(horHeaders)
        header = self.horizontalHeader()
        header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )  # refactored for Python 3.11: was setResizeMode()
        header.setStretchLastSection(False)
        # for i in range(len(horHeaders)):
        #     header.setResizeMode(i, QtWidgets.QHeaderView.ResizeToContents
        #               if i < 3 else QtWidgets.QHeaderView.Stretch)
