# Standard Library
import atexit
import datetime as dt
import hashlib
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from time import monotonic
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)
from xml.dom import minidom

import numpy as np
import pyqtgraph as pg
import pyzipper
from numpy import loadtxt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from scipy.signal import argrelextrema, savgol_filter

from QATCH.common.architecture import Architecture
from QATCH.common.fileManager import FileManager
from QATCH.common.fileStorage import FileStorage, secure_open
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles
from QATCH.core.constants import Constants, UserRoles
from QATCH.processors.CurveOptimizer import (
    DifferenceFactorOptimizer,
    DropEffectCorrection,
)
from QATCH.QModel import QModelIndus, QModelOnyx, QModelTweed, QModelVolta
from QATCH.ui.components import AnimatedComboBox
from QATCH.ui.components.analyze_action_bar import AnalyzeActionBar
from QATCH.ui.components.analyze_plot_cards import (
    SIGNAL_COLORS,
    DetailPlotCard,
    SignalOverviewCard,
)
from QATCH.ui.components.stepper import Stepper
from QATCH.ui.dialogs.pop_up_dialog import PopUp
from QATCH.ui.dialogs.signature_dialog import (
    SignatureDialog,
    auto_sign_matches_session,
    persist_auto_sign_key,
)
from QATCH.ui.styles.theme_manager import ThemeManager, desc_label_qss, tok_css
from QATCH.ui.widgets.query_run_info_widget import QueryRunInfoWidget
from QATCH.ui.widgets.saved_state_dot import SavedStateDot
from QATCH.ui.widgets.table_view_widget import TableView
from QATCH.ui.workers.analyze_worker import AnalyzeWorker
from QATCH.ui.workers.run_scan_worker import RunScanWorker

if TYPE_CHECKING:
    from QATCH.ui.main_window import MainWindow
    from QATCH.ui.windows.analyze_window import AnalyzeWindow
TAG = "[UIAnalyze]"
USE_NEW_FILL_METHOD = True

_DEV_MODE_TTL_SECONDS = 30.0

_LOAD_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="qmodel-preload")


def _shutdown_executor() -> None:
    """
    Ensures the background executor tears down cleanly when the application exits.

    - wait=False: Prevents the main application thread from hanging indefinitely
                  if a background task is currently stuck or executing.
    - cancel_futures=True: Ensures any pending tasks in the queue that haven't
                           started yet are immediately discarded (Requires Python 3.9+).
    """
    _LOAD_EXECUTOR.shutdown(wait=False, cancel_futures=True)


# Register the cleanup handler to fire automatically upon Python interpreter exit
atexit.register(_shutdown_executor)


###############################################################################
# Elaborate on the raw data gathered from the SerialProcess in parallel timing
###############################################################################
class UIAnalyze(QtWidgets.QWidget):
    progressValue = QtCore.pyqtSignal(int)
    progressFormat = QtCore.pyqtSignal(str)
    progressUpdate = QtCore.pyqtSignal()
    indus_predict_progress = QtCore.pyqtSignal(int, str)
    volta_predict_progress = QtCore.pyqtSignal(int, str)
    onyx_predict_progress = QtCore.pyqtSignal(int, str)

    def setup_ui(self, analyze_window: "AnalyzeWindow", parent: "MainWindow"):
        super(UIAnalyze, self).__init__(None)
        self.parent: "MainWindow" = parent
        assert (
            self.parent is not None
        ), "AnalyzeProcess requires a valid MainWindow parent for proper operation."
        self.stateStep = -1
        self.zoomLevel = 1
        self.xml_path = None
        self.poi_markers = []
        self.sort_order = 1  # by date, default
        self.scan_for_most_recent_run = True
        self.run_timestamps = {}
        self.run_devices = {}
        self.run_names = {}
        self.run_is_new = {}  # dict_key -> bool, see _scan_run's is_new / the "New" sort filter

        # Filesystem-watcher state that keeps the run list auto-maintained
        # instead of requiring a manual Rescan button or a full rescan on
        # every mode switch (see _ensure_watcher_armed/_rearm_watcher).
        self._run_watcher = QtCore.QFileSystemWatcher(self)
        self._run_watcher.directoryChanged.connect(self._on_watched_dir_changed)
        self._watched_load_path: Optional[str] = None
        self._pending_dirty_paths: set = set()
        self._watch_debounce_timer = QtCore.QTimer(self)
        self._watch_debounce_timer.setSingleShot(True)
        self._watch_debounce_timer.setInterval(750)
        self._watch_debounce_timer.timeout.connect(self._process_pending_watch_events)
        self._incremental_workers: list = []

        self.step_direction = "forwards"
        self.allow_modify = False
        self.moved_markers = [False, False, False, False, False, False]
        self.parent.signed_at = "[NEVER]"
        self.model_result = -1
        self.model_candidates = None
        self.model_engine = "None"
        self.analyzer_task = QtCore.QThread()
        self.qmodel_tweed_predictor = QModelTweed()

        self.qmodel_indus_modules_loaded = False
        self.qmodel_indus_predictor = None

        # QModel Volta  Constants
        self.QModel_volta_modules_loaded = False
        self.QModel_volta_predictor = None

        # QModel Onyx Constants
        self.QModel_onyx_modules_loaded = False
        self.QModel_onyx_predictor = None
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

        self.btn_Load = QtWidgets.QPushButton("Load")
        self.btn_Back = QtWidgets.QPushButton("Back")
        self.btn_Next = QtWidgets.QPushButton("Next")
        self.text_Loaded = QtWidgets.QLabel("Loaded:")
        self.text_Created = QtWidgets.QLabel("[NONE]")
        self.btn_Info = QtWidgets.QPushButton("Run Info")
        self.cBox_Runs = AnimatedComboBox(
            icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "down-chevron.svg")
        )

        self.graphStack = (
            QtWidgets.QStackedWidget()
        )  # must define this here, before connecting "self._update_progress_value" in the next section

        # Progressbar -------------------------------------------------------------
        # No inline stylesheet here - QProgressBar#progressBar is themed
        # app-wide (see app_theme.qss), driven by the ctrl_progress_* tokens
        # so it repaints correctly on light/dark switch.
        self.progressBar = QtWidgets.QProgressBar()
        # self.progressBar.setProperty("value", 0)
        self.progressBar.setGeometry(QtCore.QRect(0, 0, 50, 10))
        self.progressBar.setObjectName("progressBar")
        # self.progressBar.setFixedHeight(50)
        self.progressBar.valueChanged.connect(self._update_progress_value)
        self.progressBar.setValue(0)
        self.progressBar.setHidden(True)
        # Themed action bar: run selector + Load/Auto-Fit/Run Info +
        # Back/Next/Modify/Analyze + Advanced/User, as one card matching
        # PlotsUI/ControlsUI's visual language (see AnalyzeActionBar). It
        # only builds/lays out the widgets - every callback below is wired
        # here since those methods live on UIAnalyze, not the bar itself.
        self.actionbar = AnalyzeActionBar()
        self.text_Runs = self.actionbar.text_Runs
        self.text_Created = self.actionbar.text_Created
        self.cBox_Runs = self.actionbar.cBox_Runs
        self.sort_by = self.actionbar.sort_by
        self.sort_by_name = self.actionbar.sort_by_name
        self.sort_by_date = self.actionbar.sort_by_date
        self.sort_by_new = self.actionbar.sort_by_new
        self.sort_by_widget = self.actionbar.sort_by_widget
        self.runGrid = self.actionbar.runGrid
        self.tBtn_Load = self.actionbar.tBtn_Load
        self.tBtn_Predict = self.actionbar.tBtn_Predict
        self.tBtn_Info = self.actionbar.tBtn_Info
        self.tool_Cancel = self.actionbar.tool_Cancel
        self.tool_Back = self.actionbar.tool_Back
        self.tool_Next = self.actionbar.tool_Next
        self.tool_Modify = self.actionbar.tool_Modify
        self.tool_Analyze = self.actionbar.tool_Analyze
        self.tool_Advanced = self.actionbar.tool_Advanced
        self.tool_User = self.actionbar.tool_User

        self.sort_by_name.mousePressEvent = self.action_sort_by_name
        self.sort_by_date.mousePressEvent = self.action_sort_by_date
        self.sort_by_new.mousePressEvent = self.action_sort_by_new

        self.tBtn_Load.clicked.connect(self.load_run)  # main action
        load_menu = QtWidgets.QMenu()
        load_menu.addAction("Load runs from...", self.load_all_from_folder)
        self.tBtn_Load.setMenu(load_menu)
        self.tBtn_Predict.clicked.connect(self._restore_qmodel_predictions)
        self.tBtn_Info.clicked.connect(self.getRunInfo)

        self.tool_Cancel.clicked.connect(
            lambda: self.action_cancel(exit_batched_processing_mode=True)
        )
        self.tool_Back.clicked.connect(self.action_back)
        self.tool_Next.clicked.connect(self.action_next)
        self.tool_Modify.clicked.connect(self.action_modify)
        self.tool_Analyze.clicked.connect(
            self.action_analyze
        )  # TODO: skip ahead to analyze (if pois are all set)
        self.tool_Advanced.clicked.connect(self.action_advanced)
        # self.tool_User.clicked.connect(self.action_user)

        self.toolLayout = QtWidgets.QVBoxLayout()
        self.toolLayout.addWidget(self.actionbar)
        self.toolLayout.addWidget(self.progressBar)

        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")

        # Devices ------------------------------------------------------
        self.l0 = QtWidgets.QLabel()

        # Fixing issue #30
        self.l0.setText("<font color=#ffffff >Run Selection</font> </a>")
        if USE_FULLSCREEN:
            self.l0.setFixedHeight(50)
        # else:
        #    self.l0.setFixedHeight(15)
        self.gridLayout.addWidget(self.l0, 1, 1, 1, 4)

        self.gridLayout.addWidget(self.text_Devices, 2, 1)
        # row, col, rowspan, colspan
        self.gridLayout.addWidget(self.cBox_Devices, 2, 2, 1, 2)

        # Fixes #30
        self.showRunsFromAllDevices = QtWidgets.QCheckBox("Show all available runs")
        self.showRunsFromAllDevices.setChecked(True)
        self.showRunsFromAllDevices.clicked.connect(self.showRunsFromAllDevices_clicked)
        self.cBox_Devices.setEnabled(False)

        self.gridLayout.addWidget(self.showRunsFromAllDevices, 3, 2, 1, 2)

        # Parameters ------------------------------------------------------
        self.l1 = QtWidgets.QLabel()
        self.l1.setText("<font color=#ffffff >Parameters</font> </a>")
        if USE_FULLSCREEN:
            self.l1.setFixedHeight(50)
        # else:
        #    self.l1.setFixedHeight(15)
        self.gridLayout.addWidget(self.l1, 4, 1, 1, 4)

        self.gridLayout.addWidget(QtWidgets.QLabel("Difference Factor:"), 5, 1)
        self.validFactor = QtGui.QDoubleValidator(0.5, 2, 3)  # allow exponential notation
        self.tbox_diff_factor = QtWidgets.QLineEdit()
        self.tbox_diff_factor.setValidator(self.validFactor)
        self.tbox_diff_factor.setFixedWidth(75)
        self.gridLayout.addWidget(self.tbox_diff_factor, 5, 2)
        self.btn_diff_factor = QtWidgets.QPushButton("Set/Reload")
        self.btn_diff_factor.pressed.connect(self.set_new_diff_factor)
        self.gridLayout.addWidget(self.btn_diff_factor, 5, 3)

        self.gridLayout.addWidget(QtWidgets.QLabel("Channel Thickness:"), 6, 1)
        self.validThickness = QtGui.QDoubleValidator(0, 1, 3)  # allow exponential notation
        self.tbox_ch_thick = QtWidgets.QLineEdit()
        self.tbox_ch_thick.setValidator(self.validThickness)
        self.tbox_ch_thick.setFixedWidth(75)
        self.tbox_ch_thick.setText(str(Constants.channel_thickness))
        self.tbox_ch_thick.textEdited.connect(self.set_new_ch_thick)
        self.gridLayout.addWidget(self.tbox_ch_thick, 6, 2)
        self.h0 = QtWidgets.QLabel()
        self.h0.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
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
        self.l2.setText("<font color=#ffffff >Options</font> </a>")
        if USE_FULLSCREEN:
            self.l2.setFixedHeight(50)
        # else:
        #    self.l2.setFixedHeight(15)
        self.gridLayout.addWidget(self.l2, 1, 5, 1, 3)

        self.option_remove_dups = QtWidgets.QCheckBox("Remove duplicate analysis output files")
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
            'Auto-Calculate "Difference Factor"'
        )
        self.difference_factor_optimizer_checkbox.setChecked(False)
        self.difference_factor_optimizer_checkbox.clicked.connect(
            self.use_difference_factor_optimizer
        )
        self.gridLayout.addWidget(self.difference_factor_optimizer_checkbox, 3, 5, 1, 3)

        self.drop_effect_cancelation_checkbox = QtWidgets.QCheckBox("Drop effect correction")
        self.drop_effect_cancelation_checkbox.setChecked(True)
        self.drop_effect_cancelation_checkbox.clicked.connect(self.use_drop_effect_cancelation)
        self.gridLayout.addWidget(self.drop_effect_cancelation_checkbox, 4, 5, 1, 3)

        self.partial_fills_checkbox = QtWidgets.QCheckBox("Enable Partial-Fills")
        self.partial_fills_checkbox.setChecked(False)
        self.gridLayout.addWidget(self.partial_fills_checkbox, 5, 5, 1, 3)

        # Predict Model ------------------------------------------------------
        self.l3 = QtWidgets.QLabel()
        self.l3.setText("<font color=#ffffff >Auto-Fit Model</font> </a>")
        if USE_FULLSCREEN:
            self.l3.setFixedHeight(50)
        # else:
        #    self.l3.setFixedHeight(15)
        self.gridLayout.addWidget(self.l3, 6, 5, 1, 3)

        self.cBox_Models = QtWidgets.QComboBox()
        self.cBox_Models.addItems(Constants.list_predict_models)
        if Constants.qmodel_onyx_predict:
            self.cBox_Models.setCurrentIndex(3)
        elif Constants.qmodel_volta_predict:
            self.cBox_Models.setCurrentIndex(2)
        elif Constants.qmodel_indus_predict:
            self.cBox_Models.setCurrentIndex(1)
        elif Constants.qmodel_tweed_predict:
            self.cBox_Models.setCurrentIndex(0)
        self.cBox_Models.currentTextChanged.connect(self.set_new_prediction_model)
        self.gridLayout.addWidget(self.cBox_Models, 7, 5, 1, 3)

        self.advancedwidget = QtWidgets.QWidget()
        self.advancedwidget.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.WindowStaysOnTopHint)
        self.advancedwidget.setWhatsThis("These settings are for Advanced Users ONLY!")
        self._advanced_warning_label = QtWidgets.QLabel(f"WARNING: {self.advancedwidget.whatsThis()}")
        warningLayout = QtWidgets.QVBoxLayout()
        warningLayout.addWidget(self._advanced_warning_label)
        warningLayout.addLayout(self.gridLayout)
        self.advancedwidget.setLayout(warningLayout)
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "gear.svg")
        self.advancedwidget.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        self.advancedwidget.setWindowTitle("Advanced Settings")

        # Create dot buttons to skip directly to a particular step
        widget_h4 = QtWidgets.QWidget()
        layout_v4 = QtWidgets.QVBoxLayout()
        layout_h4 = QtWidgets.QHBoxLayout()
        layout_v4.setContentsMargins(0, 0, 0, 0)
        layout_h4.setContentsMargins(0, 0, 0, 0)

        # Leading "Loaded & saved" status indicator (was dot1) - a persistent
        # status, not a step cursor, so it lives outside the numbered Stepper.
        self.saved_state_dot = SavedStateDot()
        self.saved_state_label = QtWidgets.QLabel("Loaded & saved")
        self.saved_state_widget = QtWidgets.QWidget()
        self.saved_state_widget.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.saved_state_widget.mousePressEvent = lambda _evt: self.gotoStepNum(None, 1)
        saved_state_layout = QtWidgets.QHBoxLayout(self.saved_state_widget)
        saved_state_layout.setContentsMargins(0, 0, 0, 0)
        saved_state_layout.setSpacing(6)
        saved_state_layout.addWidget(self.saved_state_dot)
        saved_state_layout.addWidget(self.saved_state_label)

        # Numbered step indicator (was dot2..dot7, dot9, dot10 - dot8 was
        # already permanently hidden, "for POI3 removal"). See _STEP_NUMS for
        # the mapping between Stepper index and the legacy 1-based step_num
        # values gotoStepNum/setDotStepMarkers use everywhere else.
        self.stepper = Stepper(
            ["Load", "Fill Start", "Fill End", "Post", "Blip 1", "Blip 2", "Blip 3", "Analyze"]
        )
        self.stepper.stepClicked.connect(self._on_stepper_clicked)

        self.graphWidget = pg.PlotWidget()
        # Background/axis colors are applied by _apply_pg_theme() (called at
        # the end of setup_ui and on every themeChanged) rather than a
        # hardcoded literal, since pyqtgraph doesn't consume QSS.
        self.overview_card = SignalOverviewCard(self.graphWidget)
        self.overview_card.btn_zoom_in.clicked.connect(lambda: self.zoomFinderPlots(0.5))
        self.overview_card.btn_zoom_out.clicked.connect(lambda: self.zoomFinderPlots(2.0))
        self.overview_card.btn_move_left.clicked.connect(lambda: self.moveCurrentMarker(-1))
        self.overview_card.btn_move_right.clicked.connect(lambda: self.moveCurrentMarker(+1))

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
        plot_text.setHtml("<span style='font-size: 10pt'><b>No Results To View</b><br/> \
                            Load a run, follow the prompts to select points,<br/> \
                            and press \"Analyze\" action to view results.</span>")
        it = plot_text.textItem
        option = it.document().defaultTextOption()
        option.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
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
        self.graphStack.addWidget(self.overview_card)
        self.graphStack.addWidget(self.results_split)
        self.graphStack.setCurrentIndex(0)

        layout_h4.addWidget(self.saved_state_widget)
        layout_h4.addSpacing(12)
        layout_h4.addWidget(self.stepper, 1)
        widget_dots = QtWidgets.QWidget()
        widget_dots.setLayout(layout_h4)
        layout_v4.addWidget(widget_dots)

        # self.QModel_widget = QtWidgets.QWidget(self)
        # self.QModel_widget.setWindowFlags(
        #     QtCore.Qt.Tool | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.WindowType.FramelessWindowHint)
        # self.QModel_widget.setWindowTitle("QModel Widget")
        # self.QModel_runBtn = QtWidgets.QPushButton("Run QModel Again")
        # self.QModel_runBtn.clicked.connect(self._restore_qmodel_predictions)
        # # self.layout.addWidget(self.QModel_runBtn)
        # # self.QModel_runBtn.setParent(None)
        # self.QModel_widget.setFixedSize(self.QModel_runBtn.sizeHint())
        # self.QModel_widget.hide()
        # floating_layout = QtWidgets.QHBoxLayout()
        # floating_layout.setContentsMargins(0, 0, 0, 0)
        # floating_layout.addWidget(self.QModel_runBtn)
        # self.QModel_widget.setLayout(floating_layout)
        # # floating_widget.move(100, 100)  # Position relative to main window
        # # floating_widget.show()
        # # layout_v4.addWidget(floating_widget)

        layout_v4.addWidget(self.graphStack)
        # No background/objectName styling here - the overview plot's own
        # themed card (SignalOverviewCard, added to graphStack below)
        # supplies the visible card surface now; widget_h4 is just a plain
        # grouping container for the stepper row + graphStack.
        widget_h4.setLayout(layout_v4)

        self.graphWidget1 = pg.PlotWidget()
        self.graphWidget2 = pg.PlotWidget()
        self.graphWidget3 = pg.PlotWidget()
        # Background/axis colors for graphWidget1/2/3 are applied by
        # _apply_pg_theme() alongside graphWidget, above.

        self.resonance_card = DetailPlotCard(self.graphWidget1, "Resonance", "resonance")
        self.difference_card = DetailPlotCard(self.graphWidget2, "Difference", "difference")
        self.dissipation_card = DetailPlotCard(self.graphWidget3, "Dissipation", "dissipation")

        layout_h2 = QtWidgets.QHBoxLayout()
        layout_h2.addWidget(self.resonance_card)
        layout_h2.addWidget(self.difference_card)
        layout_h2.addWidget(self.dissipation_card)
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
        self.btnMovable1.clicked.connect(lambda: self.graph_split.setSizes([1, 1]))
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
            lambda: self.results_split.setSizes(self.get_results_split_auto_sizes())
        )
        layout_s2.addWidget(self.btnMovable2)
        layout_s2.addStretch()
        handle2.setLayout(layout_s2)

        # Drag-marker hint (left) + terse keyboard-shortcut hints (right),
        # matching the target layout - previously reversed (keyboard hints
        # were on the left, drag hint on the right).
        self.footerText_hint = QtWidgets.QLabel(
            "<i>Drag markers for rough placement &nbsp;·&nbsp; "
            "click the detail plots for precise placement.</i>"
        )
        self.footerText_keys = QtWidgets.QLabel(
            "<b>Esc</b> | Back &nbsp;&nbsp; <b>Enter</b> | Next"
        )
        self.footerText_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.footerText_keys.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.footerText_hint.setStyleSheet(desc_label_qss())
        self.footerText_keys.setStyleSheet(desc_label_qss())

        layout_h3 = QtWidgets.QHBoxLayout()
        layout_h3.addWidget(self.footerText_hint)
        layout_h3.addWidget(self.footerText_keys)

        # Add widgets to layout - hint bar sits at the very bottom, under
        # the detail-plot row, per the target layout.
        self.layout.addLayout(self.toolLayout)
        # self.layout.addWidget(widget_h4)
        self.layout.addWidget(self.graph_split)
        self.layout.addLayout(layout_h3)

        self.setLayout(self.layout)
        self.setWindowTitle("Analyze Data")

        # self.cBox_Devices.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        # self.cBox_Devices.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cBox_Runs.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.cBox_Runs.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cBox_Runs.addItem("No Runs Found")
        self.cBox_Runs.setEnabled(False)
        self.cBox_Runs.currentIndexChanged.connect(self.updateDev)
        # Loads implicitly on genuine user selection (click or keyboard
        # confirm) - `activated` only fires for real interaction, unlike
        # `currentIndexChanged` above, which also fires for the programmatic
        # clear()/addItems()/setCurrentText() calls background refreshes
        # (the run-list filesystem watcher, force_full_resync, etc.) make.
        self.cBox_Runs.activated.connect(self._on_run_activated)
        self.cBox_Devices.currentIndexChanged.connect(self.updateRunOnChange)
        self.btn_Load.pressed.connect(self.load_run)
        self.btn_Back.pressed.connect(self.goBack)
        self.btn_Next.pressed.connect(self.getPoints)
        self.btn_Info.pressed.connect(self.getRunInfo)
        # self.graphWidget.scene().sigMouseClicked.connect(self.summaryClick)
        self.graphWidget1.scene().sigMouseClicked.connect(self.onClick)
        self.graphWidget2.scene().sigMouseClicked.connect(self.onClick)
        self.graphWidget3.scene().sigMouseClicked.connect(self.onClick)

        self.askForPOIs = True

        """
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
        """

        self.progressValue.connect(lambda value: self.progressBar.setValue(value))
        self.progressFormat.connect(lambda value: self.progressBar.setFormat(value))
        self.progressUpdate.connect(self.progressBar.repaint)
        self.progressUpdate.connect(QtCore.QCoreApplication.processEvents)
        self.indus_predict_progress.connect(self._qmodel_indus_progress_update)
        self.volta_predict_progress.connect(self._QModel_volta_progress_update)
        self.onyx_predict_progress.connect(self._QModel_onyx_progress_update)

        # Arm the run-list watcher and do the one full scan now, so the run
        # list is already warm by the time the user first opens Analyze mode
        # instead of rescanning on every mode switch (see
        # _ensure_watcher_armed).
        self._rearm_watcher()

        # Apply the current theme once at startup, then keep it live -
        # unlike PlotsUI/ControlsUI, this panel previously had no
        # themeChanged subscription at all.
        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._apply_theme)

    def _apply_theme(self, _mode: Optional[str] = None) -> None:
        """Re-applies token-driven colors to the chrome this panel still
        styles with inline QSS (the Advanced Settings banners) and to the
        pyqtgraph plot widgets, which don't consume QSS at all.

        Args:
            _mode: Optional theme mode string, provided when connected
                directly to ThemeManager.themeChanged. Unused - the tokens
                are always re-read fresh from ThemeManager.instance().
        """
        tok = ThemeManager.instance().tokens()
        section_qss = f"background: {tok_css(tok['flat_accent'])}; padding: 1px;"
        for label in (self.l0, self.l1, self.l2, self.l3):
            label.setStyleSheet(section_qss)
        self._advanced_warning_label.setStyleSheet(
            f"background: {tok_css(tok['flat_warning'])}; padding: 1px; font-weight: bold;"
        )
        for label in (self.footerText_hint, self.footerText_keys):
            label.setStyleSheet(desc_label_qss())
        self._apply_pg_theme()

    def _apply_pg_theme(self) -> None:
        """Applies token-driven background/axis colors to the pyqtgraph plot
        widgets. pyqtgraph ignores QSS entirely, so this must be called
        explicitly on init, on every themeChanged, and after ax.clear()
        (which resets axis pens on some pyqtgraph versions).
        """
        tok = ThemeManager.instance().tokens()
        bg = QtGui.QColor(*tok["surface"][:3])
        text_pen = pg.mkPen(QtGui.QColor(*tok["plot_text_muted"][:3]))

        for plot_widget in (self.graphWidget, self.graphWidget1, self.graphWidget2, self.graphWidget3):
            plot_widget.setBackground(bg)
            plot_item = plot_widget.getPlotItem()
            for axis_name in ("left", "bottom"):
                axis = plot_item.getAxis(axis_name)
                axis.setPen(text_pen)
                axis.setTextPen(text_pen)

    def _show_analyze_plot_overlay(self) -> None:
        """Creates and displays a progress overlay on the analysis plot.

        This method initializes a placeholder `pg.PlotWidget`, embeds a
        semi-transparent status overlay (containing a label and a progress bar),
        and inserts it into the results splitter. The overlay is wrapped in a
        `QGraphicsProxyWidget` to float above the `PlotItem`.

        The overlay is automatically centered within the ViewBox using a
        single-shot timer to ensure layout calculations are complete before
        positioning.
        """
        self._hide_analyze_plot_overlay()
        results_figure = pg.PlotWidget()
        results_figure.setBackground("w")

        plot_text = pg.TextItem("", (51, 51, 51), anchor=(0.5, 0.5))
        plot_text.setHtml("<span style='font-size: 10pt'>Analyze in-progress...</span>")
        plot_text.setPos(0.5, 0.5)

        plot_text.setFlag(plot_text.GraphicsItemFlag.ItemHasNoContents, False)
        results_figure.addItem(plot_text)
        self.results_split.replaceWidget(1, results_figure)
        self.results_split.setEnabled(False)
        self._analyze_results_figure = results_figure

        # Progress overlay
        plot_item = results_figure.getPlotItem()

        container = QtWidgets.QWidget()
        container.setFixedSize(380, 62)
        container.setStyleSheet(
            "QWidget {"
            "  background: rgba(255, 255, 255, 230);"
            "}"
            "QLabel {"
            "  background: transparent;"
            "  border: none;"
            "  font-size: 10pt;"
            "  color: #333333;"
            "}"
            "QProgressBar {"
            "  border: none;"
            "  border-radius: 3px;"
            "  background: #e8f4fb;"
            "}"
            "QProgressBar::chunk {"
            "  background: #2E9BDA;"
            "  border-radius: 3px;"
            "}"
        )

        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(6)

        status_label = QtWidgets.QLabel("Starting\u2026")
        status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(False)
        progress_bar.setFixedHeight(6)

        layout.addWidget(status_label)
        layout.addWidget(progress_bar)

        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(container)
        plot_item = cast(pg.PlotItem, results_figure.getPlotItem())
        graphics_container = plot_item.graphicsItem()

        if graphics_container:
            proxy.setParentItem(graphics_container)
        proxy.setZValue(1000)

        def _center() -> None:
            try:
                plot_item = results_figure.getPlotItem()
                if plot_item is None:
                    return
                vb = plot_item.getViewBox()
                vb_rect = vb.mapRectToItem(plot_item.graphicsItem(), vb.boundingRect())
                pw = proxy.boundingRect().width()
                ph = proxy.boundingRect().height()
                proxy.setPos(
                    vb_rect.x() + (vb_rect.width() - pw) / 2.0,
                    vb_rect.y() + (vb_rect.height() - ph) / 2.0,
                )
            except RuntimeError:
                pass

        QtCore.QTimer.singleShot(0, _center)

        self._analyze_overlay = (proxy, progress_bar, status_label)

    def _update_analyze_plot_overlay(self, value: int, status: str) -> None:
        """Updates the progress percentage and status message on the plot overlay.

        This method updates the visual state of the `QProgressBar` and
        `QLabel` stored in the overlay tuple. It caps the progress bar value
        at 99 to prevent it from showing a "Complete" state before the
        finalization logic triggers. It also forces a UI event loop process
        to ensure the display refreshes during long-running operations.

        Args:
            value: The current progress percentage, typically between 0 and 100.
                The visual bar is capped at 99 internally.
            status: A string description of the current analysis step (e.g.,
                "Calculating FFT...", "Filtering data...").

        Note:
            This method calls `QCoreApplication.processEvents()`, which
            temporarily allows the UI to stay responsive but should be used
            cautiously to avoid re-entrancy issues.
        """
        overlay = getattr(self, "_analyze_overlay", None)
        if overlay is None:
            return
        _proxy, progress_bar, status_label = overlay
        progress_bar.setValue(min(value, 99))
        if status and len(status):
            progress_bar.setFormat(status)
            status_label.setText(status)
        QtCore.QCoreApplication.processEvents()

    def _hide_analyze_plot_overlay(self) -> None:
        """Removes the analysis progress overlay and handles task termination state.

        This method cleans up the graphics overlay by detaching the proxy widget
        from the scene. If the underlying analysis task finished successfully,
        the progress bar is briefly set to 100%. If the task failed (based on
        the worker's exit code), a warning dialog is displayed to the user.

        The method is designed to be idempotent and is safe to call even if the
        overlay has already been removed or was never initialized.

        Raises:
            RuntimeError: Silently handles cases where the underlying Qt objects
                have already been deleted by the C++ runtime.
        """
        overlay = getattr(self, "_analyze_overlay", None)
        if overlay is None:
            return

        proxy, progress_bar, _status_label = overlay

        failed = hasattr(self, "analyze_work") and not self.analyze_work.exitCode()
        if failed:
            PopUp.warning(self, Constants.app_title, "Analyze task failed.")
        else:
            progress_bar.setValue(100)
            progress_bar.setFormat("Progress: Finished")
            QtCore.QCoreApplication.processEvents()

        try:
            proxy.setParentItem(None)
            scene = proxy.scene()
            if scene is not None:
                scene.removeItem(proxy)
        except RuntimeError:
            pass

        self._analyze_overlay = None

    def _show_qmodel_plot_overlay(self) -> None:
        """Embeds a dimming layer and progress overlay into the main graph widget.

        This method initializes a visual overlay for QModel inference. Unlike the
        analysis plot, this does not replace the widget; instead, it layers a
        semi-transparent `QGraphicsRectItem` over the existing plot to "dim" it,
        then places a progress bar and label on top.

        The dimming rectangle and the progress widget are both parented to the
        `PlotItem`'s graphics item and managed via a tracking tuple for
        dynamic resizing and eventual removal.

        Note:
            Uses a single-shot timer to execute centering logic (`_center`)
            to ensure that the `ViewBox` geometry is fully calculated before
            the dimming rectangle is drawn.
        """
        self._hide_qmodel_plot_overlay()

        plot_item = self.graphWidget.getPlotItem()
        vb = plot_item.getViewBox()

        #  Dimming rect
        dim_rect = QtWidgets.QGraphicsRectItem()
        dim_rect.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 160)))
        dim_rect.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        dim_rect.setParentItem(plot_item.graphicsItem())
        dim_rect.setZValue(999)

        # Progress container
        container = QtWidgets.QWidget()
        container.setFixedSize(320, 62)
        container.setStyleSheet(
            "QWidget {"
            "  background: rgba(255, 255, 255, 0);"
            "}"
            "QLabel {"
            "  background: transparent;"
            "  border: none;"
            "  font-size: 10pt;"
            "  color: #333333;"
            "}"
            "QProgressBar {"
            "  border: none;"
            "  border-radius: 3px;"
            "  background: #e8f4fb;"
            "}"
            "QProgressBar::chunk {"
            "  background: #2E9BDA;"
            "  border-radius: 3px;"
            "}"
        )

        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(6)

        status_label = QtWidgets.QLabel("Auto-fitting points\u2026")
        status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setRange(0, 0)
        progress_bar.setTextVisible(False)
        progress_bar.setFixedHeight(6)

        layout.addWidget(status_label)
        layout.addWidget(progress_bar)

        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(container)
        proxy.setParentItem(plot_item.graphicsItem())
        proxy.setZValue(1000)

        def _center() -> None:
            try:
                vb_rect = vb.mapRectToItem(plot_item.graphicsItem(), vb.boundingRect())
                dim_rect.setRect(vb_rect)
                pw = proxy.boundingRect().width()
                ph = proxy.boundingRect().height()
                proxy.setPos(
                    vb_rect.x() + (vb_rect.width() - pw) / 2.0,
                    vb_rect.y() + (vb_rect.height() - ph) / 2.0,
                )
            except RuntimeError:
                pass

        QtCore.QTimer.singleShot(0, _center)

        self._qmodel_overlay = (proxy, dim_rect, progress_bar, status_label)

    def _update_qmodel_plot_overlay(self, pct: int, status: str, is_error: bool = False) -> None:
        """Updates the QModel overlay with smoothed progress and status text.

        This method implements a animation pattern to decouple
        the UI refresh rate from the worker signal frequency. It uses a
        high-precision range (0-10,000) and a 60 FPS timer to animate the
        progress bar toward the target value using a liquid ease-out effect
        (interpolating 10% of the remaining distance per frame).

        The animation protects the UI from 'jitter' caused by rapid, successive
        progress updates from the inference worker.

        Args:
            pct: The current progress percentage (0-100). The visual target
                is capped at 99 internally until the hide method is called.
            status: A human-readable string describing the current inference
                step.
            is_error: If True, forces the overlay to treat the state as a failure,
        """
        overlay = getattr(self, "_qmodel_overlay", None)
        if overlay is None:
            return

        proxy, dim_rect, progress_bar, status_label = overlay
        error_detected = is_error or "error" in status.lower() or "failed" in status.lower()
        is_finished = pct >= 100 or error_detected

        if pct > 0 and progress_bar.maximum() == 0:
            progress_bar.setRange(0, 10000)
        target_value = 10000 if is_finished else int(min(pct, 99) * 100)

        if not hasattr(progress_bar, "_chase_timer"):
            progress_bar._target_value = 0
            progress_bar._current_float = float(progress_bar.value())
            progress_bar._chase_timer = QtCore.QTimer()
            progress_bar._chase_timer.setInterval(16)

            def chase_target():
                diff = progress_bar._target_value - progress_bar._current_float

                if abs(diff) < 5.0:
                    progress_bar._current_float = float(progress_bar._target_value)
                    progress_bar.setValue(int(progress_bar._current_float))
                    progress_bar._chase_timer.stop()
                else:
                    progress_bar._current_float += diff * 0.10
                    progress_bar.setValue(int(progress_bar._current_float))

            progress_bar._chase_timer.timeout.connect(chase_target)

        progress_bar._target_value = target_value
        if not progress_bar._chase_timer.isActive():
            progress_bar._chase_timer.start()

        if status and len(status):
            status_label.setText(status)

        QtCore.QCoreApplication.processEvents()
        if is_finished:
            if getattr(self, "_qmodel_is_fading", False):
                return
            self._qmodel_is_fading = True

            if error_detected:
                progress_bar.setStyleSheet(
                    "QProgressBar { border: none; border-radius: 3px; background: #ffe6e6; }"
                    "QProgressBar::chunk { background: #DA2E2E; border-radius: 3px; }"
                )

            anim = QtCore.QVariantAnimation()
            anim.setDuration(400)
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)

            def update_opacity(val: float):
                proxy.setOpacity(val)
                dim_rect.setOpacity(val)

            anim.valueChanged.connect(update_opacity)
            anim.finished.connect(lambda: self._hide_qmodel_plot_overlay(failed=error_detected))
            anim.finished.connect(lambda: setattr(self, "_qmodel_is_fading", False))

            self._qmodel_fade_anim = anim

            QtCore.QTimer.singleShot(800, anim.start)

    def _hide_qmodel_plot_overlay(self, failed: bool = False) -> None:
        """Removes the QModel dimming layer and progress overlay.

        This method performs a comprehensive cleanup of the QModel UI state. It
        stops the internal animation timer (Target Chaser), optionally snaps
        the progress bar to 100% on success, and removes both the
        `QGraphicsProxyWidget` and the `QGraphicsRectItem` from the scene.

        Args:
            failed: If True, skips the 100% progress snap and displays a
                warning popup. Defaults to False.
        """
        overlay = getattr(self, "_qmodel_overlay", None)
        if overlay is None:
            return

        proxy, dim_rect, progress_bar, _status_label = overlay
        if hasattr(progress_bar, "_chase_timer"):
            progress_bar._chase_timer.stop()
        if not failed and progress_bar.maximum() > 0:
            progress_bar.setValue(10000)
            QtCore.QCoreApplication.processEvents()

        for item in (proxy, dim_rect):
            try:
                item.setParentItem(None)
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
            except RuntimeError:
                pass

        self._qmodel_overlay = None

        if failed:
            PopUp.warning(self, Constants.app_title, "QModel inference failed.")

    # def hideSelectTool(self, event):
    #     self.AI_SelectTool_Frame.hide()

    def get_results_split_auto_sizes(self, setMinimumWidth=True):
        tableWidget = self.results_split.widget(0).findChild(QtWidgets.QTableWidget)
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
                    self.poi_markers[px].sigPositionChangeFinished.emit(self.poi_markers[px])
                else:
                    Log.d(f"Moving marker {px} not required. Already there.")
            except Exception as e:
                Log.e(f"Moving marker {px} failed: {str(e)}")

    def showRunsFromAllDevices_clicked(self):
        self.cBox_Devices.setEnabled(not self.showRunsFromAllDevices.isChecked())
        self.update_run(self.cBox_Devices.currentIndex())

    def _switch_user_for_signature(self) -> Optional[Tuple[str, str]]:
        """Callback passed to `SignatureDialog(on_switch_user=...)`. Performs
        the actual profile switch and pushes the result into the toolbar/
        controls window; returns the new `(username, initials)` on a real
        change so the dialog can refresh its own displayed labels, or `None`
        if the switch failed or the user didn't change."""
        new_username, new_initials, new_userrole = UserProfiles.change(UserRoles.ANALYZE)
        if UserProfiles.check(UserRoles(new_userrole), UserRoles.ANALYZE):
            if self.username != new_username:
                self.username = new_username
                self.initials = new_initials
                self.parent.signature_received = False
                self.parent.signature_required = True

                Log.d("User name changed. Changing sign-in user info.")
                self.parent.controls_window.username.setText(f"User: {new_username}")
                self.parent.controls_window.userrole = UserRoles(new_userrole)
                self.parent.controls_window.signinout.setText("&Sign Out")
                self.parent.controls_window.ui.tool_User.setText(new_username)
                self.parent.analyze_window.ui.tool_User.setText(new_username)
                if self.parent.controls_window.userrole != UserRoles.ADMIN:
                    self.parent.controls_window.manage.setText("&Change Password...")
                return new_username, new_initials
            else:
                Log.d("User switched users to the same user profile. Nothing to change.")
                return None
            # PopUp.warning(self, Constants.app_title, "User has been switched.\n\nPlease sign now.")
        # elif new_username == None and new_initials == None and new_userrole == 0:
        else:
            if new_username == None and not UserProfiles.session_info()[0]:
                Log.d("User session invalidated. Switch users credentials incorrect.")
                self.parent.controls_window.username.setText("User: [NONE]")
                self.parent.controls_window.userrole = UserRoles.NONE
                self.parent.controls_window.signinout.setText("&Sign In")
                self.parent.controls_window.manage.setText("&Manage Users...")
                self.parent.controls_window.ui.tool_User.setText("Anonymous")
                self.parent.analyze_window.ui.tool_User.setText("Anonymous")
                PopUp.warning(
                    self,
                    Constants.app_title,
                    "User has not been switched.\n\nReason: Not authenticated.",
                )
            if new_username != None and UserProfiles.session_info()[0]:
                Log.d("User name changed. Changing sign-in user info.")
                self.parent.controls_window.username.setText(f"User: {new_username}")
                self.parent.controls_window.userrole = UserRoles(new_userrole)
                self.parent.controls_window.signinout.setText("&Sign Out")
                self.parent.controls_window.ui.tool_User.setText(new_username)
                self.parent.analyze_window.ui.tool_User.setText(new_username)
                if self.parent.controls_window.userrole != UserRoles.ADMIN:
                    self.parent.controls_window.manage.setText("&Change Password...")
                PopUp.warning(
                    self,
                    Constants.app_title,
                    "User has not been switched.\n\nReason: Not authorized.",
                )

            Log.d("User did not authenticate for role to switch users.")
            return None

    def _set_sort_order(self, order: int) -> None:
        """Shared implementation for the three mutually-exclusive Sort by:
        labels (Name/Date/New) - sets `self.sort_order`, refreshes the run
        list, and bolds only the now-active label."""
        self.sort_order = order
        self._refresh_cbox_runs()
        active_qss = "color: #0D4AAF; text-decoration: none; padding-left: 15px; font-weight: bold;"
        inactive_qss = "color: #0D4AAF; text-decoration: none; padding-left: 15px;"
        self.sort_by_name.setStyleSheet(active_qss if order == 0 else inactive_qss)
        self.sort_by_date.setStyleSheet(active_qss if order == 1 else inactive_qss)
        self.sort_by_new.setStyleSheet(active_qss if order == 2 else inactive_qss)

    def action_sort_by_name(self, obj: Any) -> None:
        """Sets the run sorting order to alphabetical by name and updates the UI.

        Args:
            obj (Any): The event object passed by the Qt signal (e.g., a
                QMouseEvent), or None if called programmatically.
        """
        self._set_sort_order(0)

    def action_sort_by_date(self, obj: Any) -> None:
        """Sets the run sorting order to chronological by date and updates the UI.

        Args:
            obj (Any): The event object passed by the Qt signal (e.g., a
                QMouseEvent), or None if called programmatically.
        """
        self._set_sort_order(1)

    def action_sort_by_new(self, obj: Any) -> None:
        """Filters the run list down to runs with no saved analysis yet
        (see _scan_run's is_new / self.run_is_new cache), sorted newest
        first, and updates the UI.

        Args:
            obj (Any): The event object passed by the Qt signal (e.g., a
                QMouseEvent), or None if called programmatically.
        """
        self._set_sort_order(2)

    def action_cancel(self, exit_batched_processing_mode=False):
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
        plot_text.setHtml("<span style='font-size: 10pt'><b>No Results To View</b><br/> \
                            Load a run, follow the prompts to select points,<br/> \
                            and press \"Analyze\" action to view results.</span>")
        it = plot_text.textItem
        option = it.document().defaultTextOption()
        option.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        it.document().setDefaultTextOption(option)
        it.setTextWidth(it.boundingRect().width())
        plot_text.setPos(0.5, 0.5)
        results_figure.addItem(plot_text, ignoreBounds=True)

        self.graphStack.setCurrentIndex(1)
        # self.results_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.results_split.replaceWidget(0, results_table)
        self.results_split.replaceWidget(1, results_figure)
        # self.graphStack.setCurrentIndex(0)

        # Clear subset for batched processing
        if hasattr(self, "_batched_runs") and self._batched_runs and exit_batched_processing_mode:
            last_run_in_batch_loaded = False
            if self.cBox_Runs.itemText(self.cBox_Runs.count() - 1) in self._current_run:
                last_run_in_batch_loaded = True
            self._batched_runs = None
            self.showRunsFromAllDevices_clicked()

            if last_run_in_batch_loaded:
                end_reason = "All runs in the batch have been processed."
            else:
                end_reason = "User aborted the batch before it finished."

            PopUp.information(
                self,
                "Batch Processing Mode Ended",
                "You have exited batch processing mode.<br/><br/>" + f"<b>REASON: {end_reason}</b>",
            )
            # details="This is either because you have finished processing all runs in the batch " +
            # "or because you clicked \"Close\" while in the middle of processing the batch.")

        self.clear()  # calls self.enable_buttons()

    def action_back(self):
        try:
            self.step_direction = "backwards"
            self.goBack()
        except Exception as e:
            Log.e(f"An error occurred while moving to the prior step: {str(e)}")

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
                # self.QModel_widget.hide()
                self.gotoStepNum(None, 9)  # summary

    def action_analyze(self):
        if self.parent.signature_required and (self.unsaved_changes or self.model_run_this_load):
            if self.parent.signature_received == False and auto_sign_matches_session():
                Log.w(f"Signing ANALYZE with initials {self.initials} (not asking again)")
                self.parent.signed_at = dt.datetime.now().isoformat()
                self.parent.signature_received = True  # Do not ask again this session
            if not self.parent.signature_received:
                dlg = SignatureDialog(self, on_switch_user=self._switch_user_for_signature)
                if dlg.exec_() != QtWidgets.QDialog.Accepted:
                    return
                self.parent.signed_at = dt.datetime.now().isoformat()
                self.parent.signature_received = True
                if dlg.sign_do_not_ask.isChecked():
                    persist_auto_sign_key()

        try:
            self.moved_markers = [False, False, False, False, False, False]
            self.enable_buttons(False, False)
            results_figure = pg.PlotWidget()
            results_figure.setBackground("w")
            # 'results_split' must be shown prior to replacing
            self.graphStack.setCurrentIndex(1)
            self.results_split.replaceWidget(1, results_figure)
            self.stateStep = 6  # skip to show
            self.getPoints()  # show summary
            self.getPoints()  # show analysis
        except Exception as e:
            Log.e(f"An error occurred while analyzing the selected run: {str(e)}")

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
                cur_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val)
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

    def enable_buttons(self, refocus: bool = True, enable: bool = True) -> None:
        """Enables or disables UI buttons based on the current state.

        This function adjusts the availability of various UI buttons based on, the presence of an XML path,
        the selected run, the current step in the state machine, whether modifications are allowed, or Whether the
        system is busy.

        Args:
            refocus (bool, optional): If True, refocuses the UI on `graphWidget2`. Defaults to True.
            enable (bool, optional): If False, disables all buttons (e.g., during processing). Defaults to True.

        Behavior:
        - If `enable` is False, all buttons are disabled.
        - The "Modify" button state is toggled based on whether `enable_cancel` is True and `enable_analyze` is False.
        - Navigation buttons ("Back" and "Next") are disabled if modifications are not allowed.
        - "Advanced" options are only enabled when `enable_cancel` is True.
        """
        # Determine initial button states
        enable_load = (
            bool(self.cBox_Runs.currentText().strip())
            and self.cBox_Runs.currentText() != "No Runs Found"
        )
        enable_cancel = enable_info = enable_modify = self.xml_path is not None
        enable_back = self.stateStep >= 0
        enable_next = enable_cancel and self.stateStep < 7
        enable_analyze = len(self.poi_markers) > 2

        # If disabled globally (e.g., busy state), disable everything
        if not enable:
            enable_load = enable_info = enable_cancel = enable_back = (
                enable_next
            ) = enable_modify = enable_analyze = False

        # Handle tool_Modify state
        if enable_cancel and not enable_analyze:
            if not self.tool_Modify.isChecked():
                self.tool_Modify.setChecked(True)
                self.tool_Modify.clicked.emit()
                self.allow_modify = True
        elif not enable_cancel:
            if self.tool_Modify.isChecked():
                self.tool_Modify.setChecked(False)
                self.tool_Modify.clicked.emit()
                self.allow_modify = False

        # If modification is not allowed, disable navigation buttons
        if not self.allow_modify:
            enable_back = enable_next = False

        # Apply button states
        self.tBtn_Load.setEnabled(enable_load)
        self.tBtn_Info.setEnabled(enable_info)
        self.tool_Cancel.setEnabled(enable_cancel)
        self.tool_Back.setEnabled(enable_back)
        self.tool_Next.setEnabled(enable_next)
        self.tool_Modify.setEnabled(enable_modify)
        self.tool_Analyze.setEnabled(enable_analyze)

        # Handle advanced tool enabling
        self.tool_Advanced.setEnabled(enable_cancel)

        # Handle predict tool enabling
        self.tBtn_Predict.setEnabled(enable_info)

        # Refocus if required
        if refocus:
            self.graphWidget2.setFocus()

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
            self.tbox_diff_factor.setText(f"{Constants.default_diff_factor:1.3f}")
        self.set_new_diff_factor()

    def use_drop_effect_interpolation(self, object):
        try:
            self.action_cancel()  # ask if they mean it if there are unsaved changes
            if not self.hasUnsavedChanges():  # only proceed if they say yes
                try:
                    self.diff_factor = round(float(self.tbox_diff_factor.text()), 3)
                    Log.d(f"Difference Factor = {self.diff_factor}")
                except:
                    if hasattr(self, "diff_factor"):
                        del self.diff_factor  # unset to revert to default auto-calc value
                        Log.d("Difference Factor deleted")
                self.load_run()  # refresh plots to show new diff factor
        except:
            Log.e("Failed to set new difference factor!")

    def use_drop_effect_cancelation(self, object):
        try:
            self.action_cancel()  # ask if they mean it if there are unsaved changes
            if not self.hasUnsavedChanges():  # only proceed if they say yes
                try:
                    self.diff_factor = round(float(self.tbox_diff_factor.text()), 3)
                    Log.d(f"Difference Factor = {self.diff_factor}")
                except:
                    if hasattr(self, "diff_factor"):
                        del self.diff_factor  # unset to revert to default auto-calc value
                        Log.d("Difference Factor deleted")
                self.load_run()  # refresh plots to show new diff factor
        except:
            Log.e("Failed to set new difference factor!")

    def set_new_diff_factor(self):
        """
        Validates and sets a new difference factor based on user input.

        This method checks if the input in `tbox_diff_factor` is valid and within
        the acceptable range defined by `self.validFactor`. If valid, it confirms
        any unsaved changes before proceeding to update the `diff_factor` with the
        provided input. If the input is invalid, it logs an error message. After
        updating the difference factor, it refreshes the plots by calling `self.load_run()`.

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
                    self.diff_factor = round(float(self.tbox_diff_factor.text()), 3)
                    Log.d(f"Difference Factor = {self.diff_factor}")
                except:
                    if hasattr(self, "diff_factor"):
                        del self.diff_factor  # unset to revert to default auto-calc value
                        Log.d("Difference Factor deleted")
                self.load_run()  # refresh plots to show new diff factor
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

    def set_new_prediction_model(self, text):
        index = len(Constants.list_predict_models) - 1
        default = Constants.list_predict_models[index]
        if text in Constants.list_predict_models:
            index = Constants.list_predict_models.index(text)
        else:
            Log.e(TAG, f"Unknown predict model '{text}', using default '{default}'")
        try:
            # these flags are set above `index` as a fallback option
            Constants.qmodel_tweed_predict = True if index >= 0 else False
            Constants.qmodel_indus_predict = True if index >= 1 else False
            Constants.qmodel_volta_predict = True if index >= 2 else False
            Constants.qmodel_onyx_predict = True if index >= 3 else False
        except:
            Log.e(TAG, "Failed to set new prediction model flags in Constants.py")
        try:
            self.cBox_Models.setCurrentIndex(index)
        except:
            Log.e(TAG, "Failed to set model dropdown menu in Advanced Settings")
        try:
            self.parent.controls_window.qmodel_tweed_version.setChecked(
                True if index == 0 else False
            )
            self.parent.controls_window.qmodel_indus_version.setChecked(
                True if index == 1 else False
            )
            self.parent.controls_window.qmodel_volta_version.setChecked(
                True if index == 2 else False
            )
            self.parent.controls_window.qmodel_onyx_version.setChecked(
                True if index == 3 else False
            )
        except:
            Log.e(TAG, "Failed to check the selected prediction model in the Help menu")

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
            elif self.graphStack.currentIndex() == 1 and self.analyze_work.exitCode() == False:
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
        stops = min(100, value + 25) if value < 99 else value + 1  # 0-98:+25(100); 99-100:+1
        self.progress_value_steps = list(range(start, stops, 1 if start < stops else -1))
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
            # value in self.progress_status_step.keys():
            if not self.analyzer_task.isRunning():
                keys = list(self.progress_status_step.keys())[::-1]  # in reverse order
                status = self.progress_status_step.get(keys[0])  # most recent label # .get(value)
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
            QtCore.QTimer.singleShot(int(1000 * wait_time), self._step_to_next_value)
        else:
            self.progress_value_scanning = False

    def hasUnsavedChanges(self):
        if hasattr(self, "unsaved_changes"):
            return self.unsaved_changes
        else:
            return False

    def isBusy(self) -> bool:
        if hasattr(self, "analyze_work"):
            if self.analyze_work.is_running():
                return True
        return False

    def force_full_resync(self):
        """Nuclear-option full rescan: wipes all cached run info and re-walks
        every device from disk. No longer wired to any button (the run list
        is normally kept current automatically by the filesystem watcher -
        see _ensure_watcher_armed/_rearm_watcher) - kept as an internal
        fallback for the watcher's own error-recovery path (e.g. a watched
        path becomes unreachable, or the incremental diff logic repeatedly
        fails) and as a hook for a future manual "force refresh" affordance
        if one is ever needed."""
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

        # Rescan device folders from user preference path
        for _, dirs, _ in os.walk(os.path.join(Constants.log_prefer_path)):
            if "_unnamed" in dirs:
                dirs.remove("_unnamed")
            self.parent.data_devices = dirs  # show all available devices in logged data
            break

        self.cBox_Devices.addItems(self.parent.data_devices)
        # self.cBox_Devices.setFixedWidth(self.cBox_Devices.sizeHint().width())

        self.analyzer_task = QtCore.QThread()

        # Clear out any cached run info
        self.run_timestamps = {}
        self.run_devices = {}
        self.run_names = {}
        self.run_is_new = {}

        # find most recent device run
        if self.scan_for_most_recent_run:
            self.scan_for_most_recent_run = False
            # call as timer to allow repaint of Analyze view mode
            QtCore.QTimer.singleShot(500, self.find_most_recent_run)

        self.action_sort_by_date(None)  # dummy 'obj' passed

        self.username = None
        self.initials = None

        self.clear()

    def clear(self):
        # Cheap no-op unless the load-directory preference changed since the
        # watcher was last armed (see _ensure_watcher_armed) - a second
        # safety net alongside the call in MainWindow.analyze_data(), in
        # case something else ever calls clear() directly.
        self._ensure_watcher_armed()

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

        self.progressBar.setValue(0)  # Not started
        # self.QModel_widget.hide()
        self.setDotStepMarkers(0)

        self.stateStep = -1
        self.poi_markers = []
        self.xml_path = None  # used to indicate whether a run is loaded
        self.unsaved_changes = False
        self.parent.signed_at = "[NEVER]"
        self.parent.signature_required = True  # secure assumption, set on load
        self.parent.signature_received = False
        self.model_result = -1
        self.model_candidates = None
        self.model_engine = "None"

        self._annotate_welcome_text()
        self.check_user_info()
        self.enable_buttons()

        self.parent.viewTutorialPage([5, 6])  # analyze / prior results

    # ------------------------------------------------------------------
    # Run-list filesystem watcher - auto-maintains cBox_Devices/cBox_Runs
    # from the on-disk contents of Constants.log_prefer_path instead of
    # requiring a manual Rescan or a full rescan on every mode switch. See
    # force_full_resync() for the old/manual full-reset path, kept as an
    # internal fallback.
    # ------------------------------------------------------------------
    def _ensure_watcher_armed(self) -> None:
        """Cheap, idempotent: re-arms the watcher only if the user's load
        directory preference changed `Constants.log_prefer_path` since it
        was last armed. Safe to call on every Analyze-mode entry."""
        if self._watched_load_path == Constants.log_prefer_path:
            return
        self._rearm_watcher()

    def _rearm_watcher(self) -> None:
        """(Re)points the watcher at the current `Constants.log_prefer_path`
        and its device subdirectories, then does one full resync - the one
        case a full rescan is still correct, since the watched root's
        identity just changed and nothing cached can be trusted."""
        watched = self._run_watcher.directories()
        if watched:
            self._run_watcher.removePaths(watched)

        self._watched_load_path = Constants.log_prefer_path

        device_dirs: List[str] = []
        for _, dirs, _ in os.walk(Constants.log_prefer_path):
            device_dirs = [d for d in dirs if d != "_unnamed"]
            break

        paths_to_watch = [Constants.log_prefer_path] + [
            os.path.join(Constants.log_prefer_path, d) for d in device_dirs
        ]
        existing = [p for p in paths_to_watch if os.path.isdir(p)]
        if existing:
            self._run_watcher.addPaths(existing)

        self._full_resync(device_dirs)

    def _full_resync(self, device_dirs: Optional[List[str]] = None) -> None:
        """Rebuilds the device list and launches a full multi-device scan.
        Used only when the watched root itself changes (see
        _rearm_watcher) - everyday updates go through the cheap
        _diff_devices/_diff_runs_for_device path instead. Mirrors reset()'s
        device-list rebuild without the state/graph teardown, which is
        clear()'s separate, per-mode-entry concern."""
        if device_dirs is None:
            device_dirs = []
            for _, dirs, _ in os.walk(Constants.log_prefer_path):
                device_dirs = [d for d in dirs if d != "_unnamed"]
                break

        self.cBox_Devices.clear()
        self.parent.data_devices = device_dirs
        self.cBox_Devices.addItems(device_dirs)

        self.run_timestamps = {}
        self.run_devices = {}
        self.run_names = {}
        self.run_is_new = {}

        self.find_most_recent_run()

    def _on_watched_dir_changed(self, path: str) -> None:
        """QFileSystemWatcher only reports that a directory changed, never
        what changed inside it - every fire re-lists the directory and
        diffs it (see _process_pending_watch_events), it never trusts the
        signal payload beyond "this path may be stale now"."""
        self._pending_dirty_paths.add(path)
        # Restarts if already running - coalesces bursts (e.g. many rapid
        # writes while a capture is in progress) into one pass.
        self._watch_debounce_timer.start()

    def _process_pending_watch_events(self) -> None:
        dirty = self._pending_dirty_paths
        self._pending_dirty_paths = set()
        if not dirty:
            return

        try:
            root = os.path.normcase(os.path.normpath(Constants.log_prefer_path))

            if any(os.path.normcase(os.path.normpath(p)) == root for p in dirty):
                self._diff_devices()

            for device_dir in dirty:
                if os.path.normcase(os.path.normpath(device_dir)) == root:
                    continue
                device_name = os.path.basename(device_dir)
                self._diff_runs_for_device(device_name)
        except Exception as e:
            Log.e(TAG, f"Error processing filesystem watch events: {e}")

    def _diff_devices(self) -> None:
        """Cheap directory-listing diff for the watched root: adds/removes
        device entries without touching anything that hasn't changed."""
        on_disk: set = set()
        for _, dirs, _ in os.walk(Constants.log_prefer_path):
            on_disk = {d for d in dirs if d != "_unnamed"}
            break

        known = {self.cBox_Devices.itemText(i) for i in range(self.cBox_Devices.count())}
        new_devices = on_disk - known
        removed_devices = known - on_disk

        for dev in sorted(new_devices):
            self.cBox_Devices.addItem(dev)
            device_path = os.path.join(Constants.log_prefer_path, dev)
            if os.path.isdir(device_path) and device_path not in self._run_watcher.directories():
                self._run_watcher.addPath(device_path)
            self._diff_runs_for_device(dev)

        for dev in removed_devices:
            self._remove_device(dev)

    def _remove_device(self, data_device: str) -> None:
        """Prunes a device that no longer exists on disk, unless it backs
        the currently-loaded run - a background watcher event must never
        silently yank an active session out from under the user."""
        if self.xml_path is not None:
            # self.xml_path is built as <log_prefer_path>/<device>/<folder>/<file>.xml
            # (see MainWindow.analyze_data), so its device is two dirnames up.
            loaded_device = os.path.basename(os.path.dirname(os.path.dirname(self.xml_path)))
            if loaded_device == data_device:
                Log.w(
                    TAG,
                    f'Device "{data_device}" was removed from disk, but its run is '
                    "currently loaded - leaving it in the list.",
                )
                return

        device_path = os.path.join(Constants.log_prefer_path, data_device)
        if device_path in self._run_watcher.directories():
            self._run_watcher.removePath(device_path)

        for dict_key in [k for k in self.run_timestamps if k.endswith(f":{data_device}")]:
            self.run_timestamps.pop(dict_key, None)
            self.run_names.pop(dict_key, None)
            self.run_is_new.pop(dict_key, None)

        idx = self.cBox_Devices.findText(data_device)
        if idx != -1:
            self.cBox_Devices.removeItem(idx)

        self._refresh_cbox_runs()

    def _diff_runs_for_device(self, data_device: str) -> None:
        """Cheap directory-listing diff for one device: only launches a
        `RunScanWorker` (which does the real XML/zip parsing) when the set
        of run folders on disk actually differs from what's cached - the
        vast majority of watcher fires (e.g. a live capture writing into an
        already-known run folder) exit here without touching a thread."""
        on_disk = set(FileStorage.DEV_get_logged_data_folders(data_device))
        on_disk.discard("_unnamed")
        known_keys_for_dev = {k for k in self.run_timestamps if k.endswith(f":{data_device}")}
        known_folders_for_dev = {k.rsplit(":", 1)[0] for k in known_keys_for_dev}

        if on_disk == known_folders_for_dev:
            return

        known_keys: List[str] = list(self.run_timestamps.keys())
        worker = RunScanWorker([data_device], known_keys, parent=self)
        worker.scan_finished.connect(self._on_single_device_scanned)
        self._incremental_workers.append(worker)
        worker.finished.connect(lambda w=worker: self._forget_incremental_worker(w))
        worker.start()

    def _forget_incremental_worker(self, worker: RunScanWorker) -> None:
        """Keeps self._incremental_workers from growing unbounded once a
        watcher-triggered scan completes. Tracked as a list rather than a
        single attribute (like the foreground single_scan_worker) because
        two devices can legitimately have watcher-triggered scans in
        flight close together."""
        if worker in self._incremental_workers:
            self._incremental_workers.remove(worker)

    def on_run_saved(self, save_root: str, data_device: str, run_directory: str) -> None:
        """Slot for `RenameOutputFilesWorker.run_saved` - fires once a run's
        files are fully renamed AND zipped into capture.zip, i.e. genuinely
        complete on disk. Forces an immediate, targeted re-scan of just this
        run instead of waiting on the filesystem watcher's own next
        incidental re-touch of that device's directory (see
        _diff_runs_for_device, which would otherwise see this run's folder
        as already-known - from the watcher's earlier "folder created"
        event - and skip re-scanning it, leaving it "Undated" indefinitely).
        """
        if os.path.normcase(os.path.normpath(save_root)) != os.path.normcase(
            os.path.normpath(Constants.log_prefer_path)
        ):
            return  # saved under a tree Analyze mode isn't watching (e.g. write path != load path)

        if data_device not in {
            self.cBox_Devices.itemText(i) for i in range(self.cBox_Devices.count())
        }:
            self.cBox_Devices.addItem(data_device)
            device_path = os.path.join(Constants.log_prefer_path, data_device)
            if os.path.isdir(device_path) and device_path not in self._run_watcher.directories():
                self._run_watcher.addPath(device_path)

        # Exclude just this run's key so RunScanWorker treats it as needing
        # a fresh parse, while every other already-known run for this
        # device is still skipped as usual.
        dict_key = f"{run_directory}:{data_device}"
        known_keys: List[str] = [k for k in self.run_timestamps if k != dict_key]
        worker = RunScanWorker([data_device], known_keys, parent=self)
        worker.scan_finished.connect(self._on_single_device_scanned)
        self._incremental_workers.append(worker)
        worker.finished.connect(lambda w=worker: self._forget_incremental_worker(w))
        worker.start()

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
            self.parent.signature_required = True
            self.parent.signature_received = False
            self.username, self.initials = info[0], info[1]
        else:
            self.parent.signature_required = False

    def detect_change(self):
        if not self.unsaved_changes:
            Log.d("There are unsaved changes detected.")
        if self.parent.signature_received:
            self.parent.signature_received = False
        self.unsaved_changes = True

    """
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
    """

    def onClick(self, event):
        """
        Handle a mouse click on any of the three plot widgets and move the current POI marker to the clicked x-position.

        If the click occurs inside one of the three graph widgets, the x-coordinate of the click is used to set the corresponding POI marker's value and emit its finished-move signal. The function accounts for the file's removed/hidden third POI by skipping index 2 when mapping the current stateStep to a marker index.

        Parameters:
            event (QEvent): Mouse click event from the plot scene containing the scene position.

        Side effects:
            - Updates self.poi_markers[...] by calling setValue(...) for the selected marker.
            - Emits sigPositionChangeFinished on the moved marker.
        """
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
            px = self._current_visible_poi_index()
            if px < 0 or px >= len(self.poi_markers):
                return
            index = mousePoint.x()
            Log.d(f"Mouse click @ xs = {index}")
            self.poi_markers[px].setValue(index)
            self.poi_markers[px].sigPositionChangeFinished.emit(self.poi_markers[px])

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
        """
        Move the currently selected POI marker by a number of index steps within the data x-axis.

        Parameters:
            offset (int): Number of discrete steps to move the marker; positive moves right, negative moves left. The step size is scaled by the current context width.

        Description:
            Computes which POI marker corresponds to the current step (skipping the hidden POI3 index), finds the nearest data index for that marker on self.xs, applies the offset (bounded to the valid index range), updates the marker position to the new x value, and emits the marker's sigPositionChangeFinished signal to trigger any follow-up updates. If no valid marker exists for the current step, the call does nothing.
        """
        px = self._current_visible_poi_index()
        if px < 0 or px >= len(self.poi_markers):
            return
        # 100 steps per window
        offset *= max(1, int(self.getContextWidth()[0] / 50))
        if px in range(0, len(self.poi_markers)):
            cur_val = self.poi_markers[px].value()
            new_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val) + offset
            if new_idx < 0:
                new_idx = 0
            if new_idx >= len(self.xs):
                new_idx = len(self.xs) - 1
            new_val = self.xs[new_idx]
            self.poi_markers[px].setValue(new_val)
            self.poi_markers[px].sigPositionChangeFinished.emit(self.poi_markers[px])
        else:
            pass

    def zoomFinderPlots(self, offset):
        """
        Adjust the finder-plot zoom level by a multiplicative offset and refresh the current POI context.

        This updates self.zoomLevel within safe bounds, applies special initial-clipping adjustments based
        on the current step, and emits a position-change-finished signal for the active POI marker so the
        UI refreshes. Note: the method uses the current step (self.stateStep) to select the marker but
        intentionally skips the hidden POI3 (index 2) when mapping step -> marker.

        Parameters:
            offset (float): Multiplicative zoom factor (e.g., >1 to zoom out, <1 to zoom in).

        Side effects:
            - Mutates self.zoomLevel.
            - Emits self.poi_markers[marker_index].sigPositionChangeFinished to trigger UI updates.
            - Logs warnings when zoom attempts hit configured limits or edge conditions.
        """
        if not hasattr(self, "smooth_factor"):
            Log.d("Ignoring arrow key input when no run is loaded.")
            return
        px = self._current_visible_poi_index()
        if px < 0 or px >= len(self.poi_markers):
            return
        if px in range(0, len(self.poi_markers)):
            was_clipped = self.getContextWidth()[1]
            self.zoomLevel = float(self.zoomLevel * offset)
            is_clipped = self.getContextWidth()[1]
            if was_clipped == True and is_clipped == True:
                # Compute visible ordinal 1..5; POI1/POI2 are the only "early" cases now
                visible_ord = (px if px <= 1 else px - 1) + 1
                if visible_ord <= 2:  # start, end of fill
                    self.zoomLevel = 5 * self.getContextWidth()[0] / self.smooth_factor
                else:  # blips
                    self.zoomLevel = self.stateStep * self.getContextWidth()[0] / self.smooth_factor
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
            self.poi_markers[px].sigPositionChangeFinished.emit(self.poi_markers[px])

    def setXmlPath(self, xml_path):
        Log.d(TAG, f"Setting xml filepath to: {xml_path}")
        self.xml_path = xml_path

    def updateDev(self, idx):
        # Disable all toolbar buttons, then selectively re-enable based on state
        self.enable_buttons(False, False)
        if self.xml_path is not None:
            self.tool_Cancel.setEnabled(True)  # enable Cancel
        run = self.cBox_Runs.currentText()
        if len(run.strip()) == 0 or run == "No Runs Found":
            # Don't mutate cBox_Runs here (clear()/addItem() would recurse
            # back into this currentIndexChanged handler) - populating the
            # "No Runs Found" placeholder is _refresh_cbox_runs's job; this
            # only needs to detect the transient/settled empty state and
            # bail without touching the combo itself.
            self.tBtn_Load.setEnabled(False)
            return
        if self.text_Created.text().endswith(run):
            self.enable_buttons()  # enable ALL buttons
        else:
            self.tBtn_Load.setEnabled(True)  # enable Load
        self.cBox_Runs.setEnabled(True)
        run = run[0 : run.rfind("(") - 1]
        dev = self.run_devices.get(run)
        if dev != None:
            self.cBox_Devices.setCurrentText(dev)
        else:
            Log.w(f"Device not found for run {run}")

    def updateRunOnChange(self, idx):
        if not self.showRunsFromAllDevices.isChecked():
            self.update_run(idx)

    def _on_run_activated(self, idx: int) -> None:
        """Loads a run the moment the user actually picks it from cBox_Runs
        (click or keyboard confirm) - `activated` only fires for genuine
        user interaction, never for the programmatic clear()/addItems()/
        setCurrentText() calls _refresh_cbox_runs and friends make."""
        run = self.cBox_Runs.currentText()
        if not run or run == "No Runs Found":
            return
        if self.text_Created.text().endswith(run):
            return  # already loaded - reselecting the same run is a no-op
        self.load_run()

    @staticmethod
    def _scan_run(data_device: str, data_folder: str, parse_xml: bool = True) -> Dict[str, Any]:
        """Scans a single run to fetch its file list and extract metadata.

        Worker (thread-safe): for one run, fetches its file list and, if
        parse_xml is True, extracts the 'start' metric and 'run_name'
        parameter from its XML metadata.

        This method uses ElementTree (much faster than minidom on small
        documents) and reports warnings or errors via the returned dictionary
        instead of modifying shared state directly.

        Args:
            data_device (str): The name of the device associated with the run.
            data_folder (str): The specific folder name containing the run data.
            parse_xml (bool, optional): Whether to parse the XML file for
                metadata. Defaults to True.

        Returns:
            Dict[str, Any]: A dictionary containing the scan results. Keys include:
                - "device" (str): The device name.
                - "folder" (str): The folder name.
                - "dict_key" (str): A unique key formatted as "{folder}:{device}".
                - "files" (List[str]): A list of data files (may be empty on error).
                - "timestamp" (Optional[str]): The value of the <metric name="start">.
                - "run_name" (Optional[str]): The value of the <param name="run_name">.
                - "is_new" (bool): True if no analyze-N.zip exists yet for this run
                    (i.e. it has never been analyzed/saved).
                - "warnings" (List[str]): Warnings to be logged on the UI thread.
                - "error" (Optional[str]): Error messages to be logged via Log.e.
        """
        result = {
            "device": data_device,
            "folder": data_folder,
            "dict_key": f"{data_folder}:{data_device}",
            "files": [],
            "timestamp": None,
            "run_name": None,
            "is_new": False,
            "warnings": [],
            "error": None,
        }

        try:
            data_files = FileStorage.DEV_get_logged_data_files(data_device, data_folder)
            result["files"] = data_files or []
            # Cheap - reuses the file listing just fetched above, no extra
            # I/O - and computed unconditionally (even when parse_xml=False)
            # since it doesn't depend on XML parsing at all.
            result["is_new"] = not any(
                f.startswith("analyze-") and f.endswith(".zip") for f in result["files"]
            )

            if not parse_xml:
                return result

            root = None

            # "audit.zip" is a legacy/back-compat name checked first for any
            # pre-existing data that might still use it; "capture.zip" is
            # the actual archive name every current save/export/import path
            # (RenameOutputFilesWorker, main_window.load_run, data_mode_export,
            # etc.) uses - without checking it here too, every fully-saved,
            # zipped run falls through to the loose-XML fallback below, which
            # also always misses (the loose XML is deleted once it's zipped),
            # permanently showing the run as "Undated".
            run_dir = os.path.join(Constants.log_prefer_path, data_device, data_folder)
            zn = next(
                (
                    candidate
                    for candidate in (
                        os.path.join(run_dir, "audit.zip"),
                        os.path.join(run_dir, "capture.zip"),
                    )
                    if FileManager.file_exists(candidate)
                ),
                None,
            )
            if zn is not None:
                with pyzipper.AESZipFile(
                    zn,
                    "r",
                    compression=pyzipper.ZIP_DEFLATED,
                    allowZip64=True,
                    encryption=pyzipper.WZ_AES,
                ) as zf:
                    try:
                        zf.testzip()
                    except Exception:
                        zf.setpassword(hashlib.sha256(zf.comment).hexdigest().encode())
                    files = zf.namelist()
                    xml_filename = next((x for x in files if x.endswith(".xml")), None)
                    if xml_filename is not None:
                        with zf.open(xml_filename, "r") as fh:
                            xml_bytes = fh.read()
                        root = ET.fromstring(xml_bytes)
            else:
                xml_filename = next((x for x in result["files"] if x.endswith(".xml")), None)
                if xml_filename is None:
                    result["warnings"].append(
                        f'WARNING: XML file not found in data files for run "{data_folder}"'
                    )
                    result["warnings"].append(
                        'Unable to parse "Date" without XML file. Treating as "Undated".'
                    )
                else:
                    xml_path = os.path.join(
                        Constants.log_prefer_path,
                        data_device,
                        data_folder,
                        xml_filename,
                    )
                    if os.path.exists(xml_path):
                        root = ET.parse(xml_path).getroot()

            if root is not None:
                for m in root.iter("metric"):
                    if m.get("name") == "start":
                        result["timestamp"] = m.get("value")
                        break
                for p in root.iter("param"):
                    if p.get("name") == "run_name":
                        result["run_name"] = p.get("value")
                        break

        except Exception as e:
            result["error"] = str(e)

        return result

    def _apply_scan_results(
        self, scan_results: List[Dict[str, Any]], data_device: str, unchecked_runs: List[str]
    ) -> None:
        """Merges the background scan results into the shared state dictionaries.

        Iterates through the parsed results from the background thread, logs any
        warnings or errors, and updates the internal state trackers (`run_names`,
        `run_timestamps`, and `run_devices`). Also handles cleaning up runs that
        are empty or no longer exist on the filesystem.

        Args:
            scan_results (List[Dict[str, Any]]): A list of dictionaries containing
                the parsed metadata for each run (typically the output from `_scan_run`).
            data_device (str): The name of the device associated with these runs.
            unchecked_runs (List[str]): A list of dict keys (formatted as
                "{folder}:{device}") representing runs that were previously known
                but need verification of their continued existence.
        """
        for r in scan_results:
            data_folder = r["folder"]
            dict_key = r["dict_key"]
            data_files = r["files"]

            for msg in r["warnings"]:
                Log.w(msg)
            if r["error"] is not None:
                Log.e(f'Error getting timestamp from XML for run "{data_folder}"!')
                Log.d(f"Error message: {r['error']}")

            self.run_names[dict_key] = data_folder

            if self.run_timestamps.get(dict_key) is None:
                if r["timestamp"] is not None:
                    self.run_timestamps[dict_key] = r["timestamp"]
                if r["run_name"] is not None:
                    self.run_names[dict_key] = r["run_name"]
                    self.run_devices[r["run_name"]] = data_device

            if len(data_files) > 0:
                if dict_key in unchecked_runs:
                    unchecked_runs.remove(dict_key)
                if self.run_timestamps.get(dict_key) is None:
                    self.run_timestamps[dict_key] = "0 / No Date"
                # Unconditional (unlike timestamp/run_name above): the file
                # listing is always freshly fetched by _scan_run regardless
                # of whether XML re-parsing was skipped, so this cache is
                # always as current as the run's last scan.
                self.run_is_new[dict_key] = r.get("is_new", False)
            else:
                Log.w(f"Removing empty run info ({dict_key})")

            self.run_devices[data_folder] = data_device

        for dict_key in unchecked_runs:
            Log.w(f"Removing missing run info ({dict_key})")
            self.run_timestamps.pop(dict_key, None)
            self.run_is_new.pop(dict_key, None)

    def _refresh_cbox_runs(self) -> None:
        """Sorts the cached run data and repopulates the UI run selection combobox.

        This method takes the current state of `run_timestamps`, sorts them based on
        the user's selected preference (alphabetical by name or chronological by date),
        and filters them according to the selected device or active batch subsets.
        It then updates the `cBox_Runs` widget and adjusts the UI layout width
        to fit the new content.

        The sorting logic follows:
            - `self.sort_order = 0`: Sort by Name (Ascending)
            - `self.sort_order = 1`: Sort by Date (Descending)
            - `self.sort_order = 2`: New (filters to unanalyzed runs, Date Descending)
        """
        # Define sorting configuration for readability: {order_index: (sort_key_index, reverse_bool)}
        # item[0] is the dict_key (name-based), item[1] is the timestamp
        sort_config = {0: (0, False), 1: (1, True)}  # Name: Ascending  # Date: Descending

        key_idx, is_reverse = sort_config.get(self.sort_order, (1, True))

        # Sort the items based on the current UI selection
        self.sorted_runs: List[Tuple[str, str]] = sorted(
            self.run_timestamps.items(), key=lambda item: item[key_idx].lower(), reverse=is_reverse
        )

        display_runs: List[str] = []
        selected_device = self.cBox_Devices.currentText()
        show_all = self.showRunsFromAllDevices.isChecked()

        for dict_key, captured_datetime in self.sorted_runs:
            # Extract device name from f"{folder}:{device}"
            device_name = dict_key.split(":")[-1] if ":" in dict_key else ""
            run_name = self.run_names.get(dict_key, dict_key)

            # Format the date string
            if captured_datetime == "0 / No Date":
                captured_date = "Undated"
            else:
                captured_date = captured_datetime.split("T")[0]

            formatted_display_name = f"{run_name} ({captured_date})"

            # Filter: Check if we are restricted to a specific batch subset
            if (
                hasattr(self, "_batched_runs")
                and self._batched_runs
                and formatted_display_name not in self._batched_runs
            ):
                continue

            # Filter: "New" sort mode shows only runs with no saved analysis
            # yet (see _scan_run's is_new / run_is_new cache) - default to
            # excluded (False) if a run has never actually been scanned, so
            # nothing shows up here before a scan has confirmed it's new.
            if self.sort_order == 2 and not self.run_is_new.get(dict_key, False):
                continue

            # Filter: Check device ownership
            if show_all or selected_device == device_name:
                display_runs.append(formatted_display_name)

        # Update UI Components. Preserve the current selection across the
        # rebuild if it's still present - without this, a background
        # (watcher-triggered) refresh would visibly deselect the user's
        # current pick every time, even when its own entry didn't change.
        previous_selection = self.cBox_Runs.currentText()
        self.cBox_Runs.clear()
        if display_runs:
            self.cBox_Runs.addItems(display_runs)
            self.cBox_Runs.setEnabled(True)
            if previous_selection and previous_selection in display_runs:
                self.cBox_Runs.setCurrentText(previous_selection)
        else:
            # Always leave at least one (placeholder) item in place, so
            # updateDev never sees a transiently/truly empty combo outside
            # of this method's own clear()/addItems() sequence.
            self.cBox_Runs.addItem("No Runs Found")
            self.cBox_Runs.setEnabled(False)

        # Reset internal lookup caches if they exist
        if hasattr(self, "_run_to_folder_cache"):
            self._run_to_folder_cache = {}

        # Dynamically resize the combobox and associated widgets
        new_width = max(self.cBox_Runs.sizeHint().width(), 200)
        self.cBox_Runs.setFixedWidth(new_width)
        self.sort_by_widget.setFixedWidth(new_width)

    def find_most_recent_run(self) -> None:
        """Initiates an asynchronous background scan to find the most recent run across all devices.

        This method prepares the UI by disabling buttons and initializing the progress
        bar. It then identifies all available devices from the `cBox_Devices` combobox
        and launches a `RunScanWorker` thread to scan each device's file system
        without freezing the main UI thread.

        The results are processed incrementally via the `_on_find_most_recent_scanned`
        slot as each device scan completes.
        """
        # Prevent user interaction during initial scan
        self.enable_buttons(False, False)
        self._update_progress_value(1, "Loading runs...")
        self.progressBar.setValue(0)

        # Collect device names and existing cache keys for the worker
        devices: List[str] = [
            self.cBox_Devices.itemText(i) for i in range(self.cBox_Devices.count())
        ]
        known_keys: List[str] = list(self.run_timestamps.keys())
        self._async_best_date: str = "0 / No Date"
        self._async_best_dev_idx: int = 0
        self.multi_scan_worker = RunScanWorker(devices, known_keys, parent=self)

        # Connect the signal to the handler that merges results and updates the UI
        self.multi_scan_worker.scan_finished.connect(self._on_find_most_recent_scanned)

        self.multi_scan_worker.start()

    @QtCore.pyqtSlot(list, str, list, bool)
    def _on_find_most_recent_scanned(
        self,
        scan_results: List[Dict[str, Any]],
        data_device: str,
        unchecked_runs: List[str],
        is_last: bool,
    ) -> None:
        """Handles the incremental results from the background most-recent-run scan.

        This slot is triggered whenever the `RunScanWorker` finishes scanning a
        specific device. It updates the internal state with the new results and
        tracks the globally "most recent" run across all processed devices. Once
        the final device is scanned, it finalizes the UI state.

        Args:
            scan_results (List[Dict[str, Any]]): The metadata results for the
                runs found on the scanned device.
            data_device (str): The name of the device that was just scanned.
            unchecked_runs (List[str]): List of run keys to be verified/removed
                if they no longer exist on disk.
            is_last (bool): Flag indicating if this was the last device in the
                scanning queue.
        """
        # Merge results from this device into the main state dictionaries
        self._apply_scan_results(scan_results, data_device, unchecked_runs)
        dev_idx: int = self.cBox_Devices.findText(data_device)

        if dev_idx != -1:
            for r in scan_results:
                timestamp: str = r.get("timestamp") or "0 / No Date"
                if timestamp > self._async_best_date:
                    self._async_best_date = timestamp
                    self._async_best_dev_idx = dev_idx

        if is_last:
            Log.d(
                f"Most recent run detected on device "
                f"{self.cBox_Devices.itemText(self._async_best_dev_idx)} "
                f"from {self._async_best_date}."
            )

            self.cBox_Devices.setCurrentIndex(self._async_best_dev_idx)
            self.action_sort_by_date(None)
            self.enable_buttons()

    def update_run(self, idx: int) -> None:
        """Initiates an asynchronous background scan for runs associated with a specific device.

        This method serves as the entry point for refreshing the run list when a user
        selects a new device or a manual refresh is triggered. It disables UI
        interaction, identifies the target device based on the provided index, and
        offloads the file system scanning and XML parsing to a `RunScanWorker` thread.

        The results are processed by the `_on_single_device_scanned` slot once the
        background task completes.

        Args:
            idx (int): The index of the device in the `cBox_Devices` combobox
                to be scanned.
        """
        self.enable_buttons(False, False)
        data_device: str = self.cBox_Devices.itemText(idx)

        # Provide the worker with current cache keys to avoid redundant XML parsing
        known_keys: List[str] = list(self.run_timestamps.keys())
        self.single_scan_worker = RunScanWorker([data_device], known_keys, parent=self)
        self.single_scan_worker.scan_finished.connect(self._on_single_device_scanned)
        self.single_scan_worker.start()

    @QtCore.pyqtSlot(list, str, list, bool)
    def _on_single_device_scanned(
        self,
        scan_results: List[Dict[str, Any]],
        data_device: str,
        unchecked_runs: List[str],
        is_last: bool,
    ) -> None:
        """Handles the results of a background scan for a single device selection.

        This slot is triggered when the `RunScanWorker` completes the scanning
        process for the currently selected device. It updates the internal state
        dictionaries with the new results, refreshes the run selection combobox
        to reflect changes, and re-enables the UI buttons for user interaction.

        Args:
            scan_results (List[Dict[str, Any]]): A list of dictionaries containing
                parsed run metadata (timestamp, name, files, etc.) for the device.
            data_device (str): The name of the device that was scanned.
            unchecked_runs (List[str]): Keys of runs that were previously cached
                but were not found during this scan and should be purged.
            is_last (bool): Flag indicating if this was the final device in the
                worker's queue (always True for single-device updates).
        """
        self._apply_scan_results(scan_results, data_device, unchecked_runs)
        self._refresh_cbox_runs()
        self.enable_buttons()

    def _load_qmodels(self) -> None:
        """
        Asynchronously schedules the loading of QModel predictive models.

        This method submits the heavy model-loading routines to a background thread pool
        executor (`_LOAD_EXECUTOR`). It is fully idempotent and safe to call repeatedly
        (e.g., in a polling loop or UI refresh cycle).

        Configuration constraints:
            - Models are only loaded if their respective prediction flags
            (`qmodel_onyx_predict` / `qmodel_volta_predict`) are active in `Constants`.
            - Models are skipped if they are already successfully loaded.
            - Models are skipped if a loading task is already in-flight (tracked via futures).
        """
        # Ensure future tracking attributes exist (fallback if not set in __init__)
        if not hasattr(self, "_qmodel_indus_future"):
            self._qmodel_indus_future = None
        if not hasattr(self, "_qmodel_volta_future"):
            self._qmodel_volta_future = None
        if not hasattr(self, "_qmodel_onyx_future"):
            self._qmodel_onyx_future = None

        # QModel Indus Preload
        try:
            requries_indus = getattr(Constants, "qmodel_indus_predict", False)
            is_indus_pending = self._qmodel_indus_future is not None

            if requries_indus and not self.qmodel_indus_modules_loaded and not is_indus_pending:
                self._qmodel_indus_future = _LOAD_EXECUTOR.submit(self._load_qmodel_indus)

        except Exception as e:
            Log.e("ERROR", f"Failed to schedule 'QModel Indus' preload. Details: {e}")

        # QModel Volta  Preload
        try:
            requires_volta = getattr(Constants, "qmodel_volta_predict", False)
            is_volta_pending = self._qmodel_volta_future is not None

            if requires_volta and not self.QModel_volta_modules_loaded and not is_volta_pending:
                self._qmodel_volta_future = _LOAD_EXECUTOR.submit(self._load_qmodel_volta)

        except Exception as e:
            Log.e("ERROR", f"Failed to schedule 'QModel Volta ' preload. Details: {e}")

        # QModel Onyx (onyx) Preload
        try:
            requires_onyx = getattr(Constants, "qmodel_onyx_predict", False)
            is_onyx_pending = self._qmodel_onyx_future is not None

            if requires_onyx and not self.QModel_onyx_modules_loaded and not is_onyx_pending:
                self._qmodel_onyx_future = _LOAD_EXECUTOR.submit(self._load_qmodel_onyx)

        except Exception as e:
            Log.e("ERROR", f"Failed to schedule 'QModel Onyx' preload. Details: {e}")

    @staticmethod
    def _load_qmodel_indus() -> "QModelIndus":
        """
        Instantiates and loads the QModelIndus prediction model into memory.

        This worker method performs heavy disk I/O to load PyTorch model weights
        (.pth files) for both regressors and classifiers. It is designed to be
        executed safely off the main UI thread via a concurrent futures executor.

        Returns:
            QModelIndus: A fully initialized QModelIndus prediction model ready for inference.
        """
        base_path = os.path.join(
            Architecture.get_path(),
            "QATCH",
            "QModel",
            "assets",
            "qmodel_indus",
        )

        return QModelIndus(
            reg_path_1=os.path.join(base_path, "poi_model_mini_window_0_1600.pth"),
            reg_path_2=os.path.join(base_path, "poi_model_mini_window_1_1600.pth"),
            clf_path=os.path.join(base_path, "v4_model_pytorch_2100.pth"),
            reg_batch_size=2048,
            clf_batch_size=1024,
        )

    @staticmethod
    def _load_qmodel_volta() -> "QModelVolta":
        """
        Instantiates and loads the Volta prediction model into memory.

        This worker method constructs the asset map and performs heavy disk I/O
        to load the YOLO-based PyTorch weights (.pt files) for various detectors
        and classifiers. It is designed to be executed off the main UI thread.

        Returns:
            QModelVolta: A fully initialized Volta prediction model ready for inference.
        """
        base_path = os.path.join(
            Architecture.get_path(),
            "QATCH",
            "QModel",
            "assets",
            "qmodel_volta",
        )

        # Map out the required weights files for the model suite
        model_assets = {
            "spacing_prior": os.path.join(base_path, "spacing_prior.json"),
            "fill_classifier": os.path.join(
                base_path, "classifiers", "fill_classifier", "type_cls.pt"
            ),
            "detectors": {
                "init": os.path.join(base_path, "detectors", "init_detector", "init.pt"),
                "ch1": os.path.join(base_path, "detectors", "ch1_detector", "ch1.pt"),
                "ch2": os.path.join(base_path, "detectors", "ch2_detector", "ch2.pt"),
                "ch3": os.path.join(base_path, "detectors", "ch3_detector", "ch3.pt"),
                "poi5_fine": os.path.join(base_path, "detectors", "eof_detector", "eof.pt"),
            },
        }

        return QModelVolta(model_assets=model_assets)

    @staticmethod
    def _load_qmodel_onyx() -> "QModelOnyx":
        """
        Instantiates and loads the QModel Onyx (onyx) prediction model into memory.

        This worker method constructs the asset map and performs heavy disk I/O
        to load the YOLO-based PyTorch weights (.pt files) for various detectors
        and classifiers, plus the onyx-specific spacing-prior decode and zoom
        refinement assets. It is designed to be executed off the main UI thread.

        Returns:
            QModelOnyx: A fully initialized onyx prediction model ready for inference.
        """
        base_path = os.path.join(
            Architecture.get_path(),
            "QATCH",
            "QModel",
            "assets",
            "qmodel_onyx",
        )

        # Map out the required weights files for the model suite. The
        # zoom-refiner detectors are optional (see QModelOnyx.ZOOM_REFINE_MAP in
        # onyx_yolo.py); absent files simply keep refine_pois a no-op.
        model_assets = {
            "spacing_prior": os.path.join(base_path, "spacing_prior.json"),
            "fill_classifier": os.path.join(
                base_path, "classifiers", "fill_classifier", "type_cls.pt"
            ),
            "detectors": {
                "init": os.path.join(base_path, "detectors", "init_detector", "init.pt"),
                "ch1": os.path.join(base_path, "detectors", "ch1_detector", "ch1.pt"),
                "ch2": os.path.join(base_path, "detectors", "ch2_detector", "ch2.pt"),
                "ch3": os.path.join(base_path, "detectors", "ch3_detector", "ch3.pt"),
                "ch1_zoom": os.path.join(
                    base_path, "detectors", "ch1_zoom_detector", "ch1_zoom.pt"
                ),
                "ch2_zoom": os.path.join(
                    base_path, "detectors", "ch2_zoom_detector", "ch2_zoom.pt"
                ),
                "ch3_zoom": os.path.join(
                    base_path, "detectors", "ch3_zoom_detector", "ch3_zoom.pt"
                ),
            },
        }

        return QModelOnyx(model_assets=model_assets)

    def _await_qmodels(self, timeout: float = 120.0) -> None:
        """
        Blocks the calling thread until in-flight prediction models finish loading.

        This method resolves the asynchronous futures generated by `_load_qmodels()`.
        Once a future completes, it installs the resulting model instance onto the
        class and updates the corresponding load-state flags.

        Args:
            timeout: Maximum seconds to wait for a model load to complete before
                    a TimeoutError is raised. Defaults to 120.0 seconds.

        Notes:
            - Errors encountered during model instantiation are caught and logged
            (not raised), mimicking the original synchronous `try/except` behavior.
            - Future attributes are explicitly nulled out after resolution to
            prevent memory leaks and allow for clean reloading later.
        """
        # Resolve QModel Indus
        v4_fut = getattr(self, "_qmodel_indus_future", None)

        if v4_fut is not None and not getattr(self, "qmodel_indus_modules_loaded", False):
            try:
                # Block and wait for the thread pool to return the loaded V4 model
                self.qmodel_indus_predictor = v4_fut.result(timeout=timeout)
                self.qmodel_indus_modules_loaded = True
                Log.i(TAG, "'QModel Indus' modules loaded successfully.")
            except Exception as e:
                Log.e(TAG, f"Failed to load 'QModel Indus' modules. Details: {e}")
            finally:
                self._qmodel_indus_future = None

        # Resolve QModel Volta
        volta_fut = getattr(self, "_qmodel_volta_future", None)

        if volta_fut is not None and not getattr(self, "QModel_volta_modules_loaded", False):
            try:
                # Block and wait for the thread pool to return the loaded Volta model
                self.QModel_volta_predictor = volta_fut.result(timeout=timeout)
                self.QModel_volta_modules_loaded = True
                Log.i(TAG, "'QModel Volta ' modules loaded successfully.")
            except Exception as e:
                Log.e(TAG, f"Failed to load 'QModel Volta ' modules. Details: {e}")
            finally:
                self._qmodel_volta_future = None

        # Resolve QModel Onyx (onyx)
        onyx_fut = getattr(self, "_qmodel_onyx_future", None)

        if onyx_fut is not None and not getattr(self, "QModel_onyx_modules_loaded", False):
            try:
                # Block and wait for the thread pool to return the loaded onyx model
                self.QModel_onyx_predictor = onyx_fut.result(timeout=timeout)
                self.QModel_onyx_modules_loaded = True
                Log.i(TAG, "'QModel Onyx' modules loaded successfully.")
            except Exception as e:
                Log.e(TAG, f"Failed to load 'QModel Onyx' modules. Details: {e}")
            finally:
                self._qmodel_onyx_future = None

    def _check_dev_mode_cached(self, force_refresh: bool = False) -> bool:
        """
        Retrieves the developer mode status, utilizing a time-to-live (TTL) cache.

        This acts as a high-performance wrapper around `UserProfiles.checkDevMode()`.
        Because checking dev mode requires file system I/O, the result is cached for
        `_DEV_MODE_TTL_SECONDS` to prevent UI bottlenecks during rapid reloads or
        frequent polling.

        Args:
            force_refresh: If True, bypasses the cache, forces a fresh file system
                        read, and resets the TTL timer. Defaults to False.

        Returns:
            bool: True if developer mode is active, False otherwise.
        """
        now = monotonic()

        # Evaluate cache validity
        is_missing = not hasattr(self, "_dev_mode_cache_value")
        is_expired = (now - getattr(self, "_dev_mode_cache_time", 0.0)) > _DEV_MODE_TTL_SECONDS

        if force_refresh or is_missing or is_expired:
            # Cache miss or invalidation: fetch fresh data and reset the timer
            self._dev_mode_cache_value = UserProfiles.checkDevMode()
            self._dev_mode_cache_time = now

        return self._dev_mode_cache_value

    def _compute_folder_from_run(self, run_string: str) -> str:
        """
        Extracts the parent folder identifier from a formatted run string.

        This acts as a cache-miss fallback for resolving a folder name. It parses
        the base folder name by stripping the run suffix (e.g., "(Run 1)"), then
        performs an O(N) reverse lookup in `self.run_names` to find the corresponding
        key prefix.

        Args:
            run_string: The formatted run string (e.g., "Experiment_A (1)").

        Returns:
            str: The extracted key prefix (e.g., "KeyPrefix" from "KeyPrefix:Details"),
                or the raw parsed folder name if the reverse lookup fails.
        """
        # Parse the base folder name by stripping the trailing " (" suffix
        idx = run_string.rfind("(")
        folder_name = run_string[: idx - 1] if idx > 0 else run_string

        try:
            matching_key = next(
                key for key, value in self.run_names.items() if value == folder_name
            )
            return matching_key.split(":", 1)[0]

        except StopIteration:
            return folder_name

    def get_folder_from_run(self, run_string: str) -> str:
        """
        Retrieves the folder identifier for a given run string using a memoized cache.

        This acts as a high-performance wrapper around `_compute_folder_from_run`.
        It prevents redundant O(N) reverse dictionary lookups by caching previously
        resolved run strings in memory, which is especially useful during rapid UI
        refreshes or batch processing.

        Args:
            run_string: The formatted run string (e.g., "Experiment_A (1)").

        Returns:
            str: The extracted folder identifier or key prefix.
        """
        # Lazy-initialize the cache dictionary if it doesn't exist
        if not hasattr(self, "_run_to_folder_cache"):
            self._run_to_folder_cache = {}

        # Return the cached result immediately if it's already been computed
        if run_string in self._run_to_folder_cache:
            return self._run_to_folder_cache[run_string]

        # Cache miss
        folder = self._compute_folder_from_run(run_string)
        self._run_to_folder_cache[run_string] = folder

        return folder

    def load_all_from_folder(self, from_folder: Optional[str] = None) -> None:
        """
        Initiates a batch-loading process for all unanalyzed runs within a selected directory.

        This method scans a target directory (either the root data folder or a specific
        device folder) for runs. It uses a ThreadPoolExecutor to rapidly scan the file
        system for runs that lack an 'analyze-1.zip' file. Unanalyzed runs are queued
        up in the UI, and the first run is automatically loaded.

        Args:
            from_folder: An optional absolute path to bypass the file dialog.
                        Must be a subdirectory of `Constants.log_prefer_path`.
        """
        # Guard: Check for unsaved changes
        self.action_cancel()
        if self.hasUnsavedChanges():
            Log.d("User declined load action. There are unsaved changes.")
            return

        # Determine target directory
        selected_directory = from_folder or QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory", Constants.log_prefer_path
        )

        # Guard: Ensure directory is valid and within the allowed working path
        if not selected_directory:
            return
        if not selected_directory.startswith(Constants.log_prefer_path):
            Log.w("User selected an inaccessible directory for batch loading")
            Log.w("NOTE: The load directory must be within the working directory")
            return

        Log.i(f'Batch loading from "{selected_directory}"')

        # Preload ML models early since batch loads will require them
        self._load_qmodels()

        # Parse the directory depth to determine intent (Root vs Device vs Single Run)
        rel_path = os.path.relpath(selected_directory, Constants.log_prefer_path)
        path_parts = [p for p in rel_path.replace("\\", "/").split("/") if p and p != "."]

        all_runs = []

        if len(path_parts) == 0:
            # Root folder selected: gather all runs from all devices
            for dev_name in getattr(self.parent, "data_devices", []):
                for run in FileStorage.DEV_get_logged_data_folders(dev_name):
                    all_runs.append((dev_name, run))

        elif len(path_parts) == 1:
            # Device folder selected: gather all runs for this specific device
            dev_name = path_parts[0]
            for run in FileStorage.DEV_get_logged_data_folders(dev_name):
                all_runs.append((dev_name, run))

        else:
            # Single run folder selected (depth >= 2)
            Log.w("User selected a single run folder for batch loading")
            Log.w("NOTE: Use the normal Load button for single run operation")
            return

        if not all_runs:
            Log.w("No runs found in the selected folder")
            return

        # Filter out unnamed runs before dispatching to threads
        scan_items = [
            (dev, run) for dev, run in all_runs if "_unnamed" not in dev and "_unnamed" not in run
        ]

        # Parallel I/O Scan: Check for missing 'analyze-1.zip' files
        def _scan(dev_run: Tuple[str, str]) -> Tuple[str, str, Optional[List[str]]]:
            dev, run = dev_run
            try:
                files = FileStorage.DEV_get_logged_data_files(dev, run)
                return dev, run, files
            except Exception as e:
                Log.w(f"Error scanning {dev}/{run}: {e}")
                return dev, run, None

        new_runs = []
        if scan_items:
            with ThreadPoolExecutor(
                max_workers=min(8, len(scan_items)), thread_name_prefix="batch-scan"
            ) as ex:
                for dev, run, files in ex.map(_scan, scan_items):
                    if files and "analyze-1.zip" not in files:
                        new_runs.append(run)
                    elif not files:
                        Log.w(f"No files found for run: {dev}/{run}")

        Log.i(f"New runs to be analyzed: {new_runs}")
        if not new_runs:
            Log.w("No new runs require analysis.")
            return

        # Synchronize found runs with the UI Combo Box
        # Build a fast O(1) lookup dictionary: {"RunBaseName": "RunBaseName (idx)"}
        combo_texts = [self.cBox_Runs.itemText(i) for i in range(self.cBox_Runs.count())]
        run_name_to_combo_text = {text.rsplit(" ", 1)[0]: text for text in combo_texts}

        sorted_new_runs = []
        for new_run in new_runs:
            if new_run in run_name_to_combo_text:
                sorted_new_runs.append(run_name_to_combo_text[new_run])
            else:
                Log.e(f"Cannot load missing run (not found in UI): {new_run}")

        # Apply batch state and trigger the first load
        self._batched_runs = sorted_new_runs
        self.showRunsFromAllDevices_clicked()

        # Clear the folder cache since the cBox_Runs items have been entirely replaced
        if hasattr(self, "_run_to_folder_cache"):
            self._run_to_folder_cache.clear()

        Log.i(f"Loading first batch run: {self.cBox_Runs.itemText(0)} (at idx=0)")
        self.cBox_Runs.setCurrentIndex(0)
        self.btn_Load.click()

        # Notify the user
        PopUp.information(
            self,
            "Batch Processing Mode Started",
            f"<b>SUCCESS: {len(sorted_new_runs)} runs found for batch processing.</b><br/><br/>"
            "When finished analyzing a run (or to skip a run), <br/>"
            'click "Load" again to move to the next queued run.<br/>'
            "You'll get another popup when the batch is finished.",
        )

    def load_run(self) -> None:
        """
        Prepares the UI and application state for loading a new analysis run.

        This method acts as a pre-flight coordinator. It ensures unsaved changes
        are handled, verifies Developer Mode compliance (for encrypted results),
        evaluates auto-signing session keys, resets the plotting UI into a "Loading"
        state, and finally queues the actual heavy data-load operation on the event loop.
        """
        # Guard: Check for unsaved changes before proceeding
        self.action_cancel()
        if self.hasUnsavedChanges():
            Log.d("User declined load action. There are unsaved changes.")
            return

        # Kick off background preloading of ML models early
        self._load_qmodels()

        # Developer Mode & Encryption Compliance
        enabled, error, expires = self._check_dev_mode_cached()

        if not enabled and (error or expires):
            PopUp.warning(
                self,
                "Developer Mode Expired",
                "Developer Mode has expired and these analysis results will be encrypted.\n"
                "An admin must renew or disable 'Developer Mode' to suppress this warning.",
            )

        # Reset UI Control States
        self.askForPOIs = True
        self.btn_Next.setText("Next")

        # Build and position Loading Annotations
        self._text1 = pg.TextItem("", color=(51, 51, 51), anchor=(0.5, 0.5))
        self._text1.setHtml("<span style='font-size: 14pt'>Loading data for analysis... </span>")
        self._text1.setPos(0.5, 0.50)

        self._text2 = pg.TextItem("", color=(51, 51, 51), anchor=(0.5, 0.5))
        self._text2.setHtml("<span style='font-size: 10pt'>(may take a few seconds) </span>")
        self._text2.setPos(0.5, 0.40)

        self._text3 = pg.TextItem("", color=(51, 51, 51), anchor=(0.5, 0.5))
        # self._text3.setHtml("<span style='font-size: 10pt'>Please be more patient with longer runs. </span>")
        self._text3.setHtml("")
        self._text3.setPos(0.5, 0.25)

        # Clear plotting canvases and apply Loading text
        primary_ax = self.graphWidget
        plot_elements = [primary_ax, self.graphWidget1, self.graphWidget2, self.graphWidget3]

        for plot_item in plot_elements:
            if plot_item is not None:
                plot_item.clear()
                plot_item.setLimits(yMin=None, yMax=None, minYRange=None, maxYRange=None)
                plot_item.setXRange(min=0, max=1)
                plot_item.setYRange(min=0, max=1)

                # Inject text onto the main plotting axis
                if plot_item is primary_ax:
                    plot_item.addItem(self._text1, ignoreBounds=True)
                    plot_item.addItem(self._text2, ignoreBounds=True)
                    plot_item.addItem(self._text3, ignoreBounds=True)

        # Reset Progress Bar and decouple previous signals
        try:
            self.progressBar.valueChanged.disconnect(self._update_progress_value)
        except Exception:
            # Fails silently if the signal was never connected in the first place
            Log.w("Cannot disconnect non-existent method from ProgressBar.")

        self.enable_buttons(False, False)
        self._update_analyze_progress(0, "Reading Run Data...")
        self._update_analyze_progress(75, "Reading Run Data...")
        QtWidgets.QApplication.processEvents()
        QtCore.QTimer.singleShot(0, self.action_load_run)

    def action_load_run(self) -> None:
        """
        Executes the final stages of loading a run and triggers data analysis.

        This method acts as the execution phase following `loadRun()`. It handles:
        1. Batch Processing Navigation: If in batch mode and the current run is
        already loaded, it auto-increments the UI to the next run in the queue,
        or exits batch mode if the queue is finished.
        2. Model Synchronization: Blocks until background ML models finish loading.
        3. Data Analysis: Dispatches the selected run to the parent coordinator
        for processing and resets UI interaction states.
        """
        try:
            # Handle Batch Processing Queue Navigation
            if getattr(self, "_batched_runs", None):
                current_text = self.cBox_Runs.currentText()
                current_idx = self.cBox_Runs.currentIndex()
                last_idx = self.cBox_Runs.count() - 1
                current_run = getattr(self, "_current_run", "")

                # If the currently selected run is already loaded, "Load" acts as a "Next" button
                if current_run and current_text in current_run:
                    if current_idx < last_idx:
                        Log.i(TAG, "Incrementing batch processing to next file in subset of list.")
                        self.cBox_Runs.setCurrentIndex(current_idx + 1)
                    else:
                        # We reached the end of the batch list
                        Log.w(TAG, "No more runs to batch process. Finished batch processing!")
                        self.action_cancel(exit_batched_processing_mode=True)
                        return

            # Synchronize ML Models
            self._await_qmodels()

            # Reset state and trigger analysis
            self.moved_markers = [False, False, False, False, False, False]

            # Use the memoized folder lookup to prevent redundant file system reads
            folder_name = self.get_folder_from_run(self.cBox_Runs.currentText())
            device_name = self.cBox_Devices.currentText()

            self.parent.analyze_data(device_name, folder_name, None)

            # Re-enable the UI controls post-analysis
            self.enable_buttons()

        except Exception as e:
            Log.e(f"An error occurred while loading the selected run: {e}")
            self.action_cancel()

    def goBack(self):
        """
        Step backwards in the analysis workflow by updating internal step state and UI, skipping the hidden POI3 step.

        If called from the final visible step, re-enables marker movement for all POI markers and decrements the step counter an extra time to bypass the hidden POI3 step. Then advances the internal state two steps backward. If the resulting state falls before the first step, resets the workflow to the initial step and restores any QModel predictions; otherwise, clears the moved-markers flags and re-enters the workflow by calling getPoints().

        Side effects:
        - Modifies self.stateStep.
        - Enables the Next button.
        - May call self._restore_qmodel_predictions() when resetting to the start.
        - Resets self.moved_markers and invokes self.getPoints().
        - Toggles movability on POI markers when stepping back from the final step.
        """
        self.btn_Next.setEnabled(True)
        if self.stateStep == 7:
            for marker in self.poi_markers:
                marker.setMovable(True)
            self.stateStep -= 1  # Skip over step 6 since POI3 is hidden
            # Log.w("State step 7 triggered")
        self.stateStep -= 2
        if self.stateStep < -1:
            self.stateStep = 0
            self._restore_qmodel_predictions()
            # if PopUp.question(
            #     self,
            #     "Are you sure?",
            #     "Any manual points will be lost if you run QModel again.\n\nProceed?",
            # ):
            # self.parent.analyze_data(
            #     self.cBox_Devices.currentText(),
            #     self.get_folder_from_run(self.cBox_Runs.currentText()),
            #     None,
            # )  # force back to step 1 of 6
            # else:
            # self.stateStep = 0
        else:
            self.moved_markers = [False, False, False, False, False, False]
            self.getPoints()

    def getContextWidth(self):
        if not hasattr(self, "smooth_factor"):
            Log.d("Ignoring arrow key input when no run is loaded.")
            return
        clipped = False
        if self.stateStep <= 2:  # start, end of fill, (no longer post point)
            ws = int(self.zoomLevel * self.smooth_factor / 2)  # context width
        else:  # blips
            ws = int(self.zoomLevel * self.smooth_factor * self.stateStep)  # context width
        if ws > len(self.xs) / 2:
            ws = int(len(self.xs) / 20)

        # Use visible-aware px
        px = self._current_visible_poi_index()
        if 0 <= px < len(self.poi_markers):
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

    def _qmodel_indus_progress_update(self, pct: int, status: Optional[str]):
        if getattr(self, "_qmodel_overlay", None) is None:
            self._show_qmodel_plot_overlay()
        self._update_qmodel_plot_overlay(pct, status or "")

    def _QModel_volta_progress_update(self, pct: int, status: Optional[str]):
        if getattr(self, "_qmodel_overlay", None) is None:
            self._show_qmodel_plot_overlay()
        self._update_qmodel_plot_overlay(pct, status or "")

    def _QModel_onyx_progress_update(self, pct: int, status: Optional[str]):
        if getattr(self, "_qmodel_overlay", None) is None:
            self._show_qmodel_plot_overlay()
        self._update_qmodel_plot_overlay(pct, status or "")

    def _restore_qmodel_predictions(self):
        try:
            if self.model_engine == "None":
                # no run is loaded
                PopUp.information(
                    self,
                    "Auto-Fit Not Available",
                    "Auto-fit cannot be run at this time.\nPlease load a run first.",
                )
                # special exception case: indicates software declined action
                raise ConnectionRefusedError()
            if not PopUp.question(
                self,
                "Are you sure?",
                'Any manual points will be lost if you run "Auto-Fit" again.\n\nProceed?',
            ):
                # special exception case: indicates user declined action
                raise ConnectionAbortedError()

            # Flag used to check when finished (hides progress bar)
            self.prediction_restored = False

            self.timer = QtCore.QTimer()
            self.timer.setInterval(100)
            self.timer.setSingleShot(False)
            self.timer.timeout.connect(self.check_finished)
            self.timer.start()

            # restore QModel predictions
            poi_vals = []
            self.model_result = -1
            self.model_candidates = None
            self.model_engine = "None"
            if Constants.qmodel_onyx_predict:
                Log.w("Auto-fitting points with QModel Onyx... (may take a few seconds)")
                QtCore.QCoreApplication.processEvents()
                try:
                    with secure_open(self.loaded_datapath, "r", "capture") as f:
                        fh = BytesIO(f.read())
                        predictor = self.QModel_onyx_predictor
                        predict_result, detected_channels = predictor.predict(
                            file_buffer=fh, progress_signal=self.onyx_predict_progress
                        )
                        # Restoring predictions restores the channel count.
                        self.parent.num_channels = detected_channels
                        Log.i(
                            TAG,
                            f"QModel Onyx Inference Complete. Detected Config: {detected_channels} Channel(s)",
                        )
                        predictions = []
                        candidates = []
                        for i in range(6):
                            poi_key = f"POI{i+1}"
                            data = predict_result.get(poi_key, {})
                            indices = data.get("indices", [-1])
                            confidences = data.get("confidences", [-1])
                            if not indices:
                                indices = [-1]
                            if not confidences:
                                confidences = [-1]
                            predictions.append(indices[0])
                            candidates.append((indices, confidences))
                        self.model_run_this_load = True
                        self.model_result = predictions
                        self.model_candidates = candidates
                        self.model_engine = f"Onyx - {detected_channels}ch"
                        if isinstance(self.model_result, list) and len(self.model_result) == 6:
                            poi_vals = self.model_result.copy()
                            if poi_vals[2] == -1 and poi_vals[1] != -1:
                                # Correct POST point to End-of-fill + 2
                                poi_vals[2] = poi_vals[1] + 2
                        else:
                            self.model_result = -1  # Invalid result format

                except Exception as e:
                    import traceback

                    Log.e(TAG, f"Error using 'QModel Onyx': {e}")
                    for line in traceback.format_tb(sys.exc_info()[2]):
                        Log.d(line.strip())
                    self.model_result = -1  # Trigger fallback handling
                    # raise e
            if self.model_result == -1 and Constants.qmodel_volta_predict:
                Log.w("Auto-fitting points with QModel Volta ... (may take a few seconds)")
                QtCore.QCoreApplication.processEvents()
                try:
                    with secure_open(self.loaded_datapath, "r", "capture") as f:
                        fh = BytesIO(f.read())
                        predictor = self.QModel_volta_predictor
                        # self._QModel_create_new_progress_dialog()
                        # self.progressBarDiag.setRange(0, 100)
                        predict_result, detected_channels = predictor.predict(
                            file_buffer=fh, progress_signal=self.volta_predict_progress
                        )
                        # Restoring predictions restores the channel count.
                        self.parent.num_channels = detected_channels
                        Log.i(
                            TAG,
                            f"QModel Volta Inference Complete. Detected Config: {detected_channels} Channel(s)",
                        )
                        predictions = []
                        candidates = []
                        for i in range(6):
                            poi_key = f"POI{i+1}"
                            data = predict_result.get(poi_key, {})
                            indices = data.get("indices", [-1])
                            confidences = data.get("confidences", [-1])
                            if not indices:
                                indices = [-1]
                            if not confidences:
                                confidences = [-1]
                            predictions.append(indices[0])
                            candidates.append((indices, confidences))
                        self.model_run_this_load = True
                        self.model_result = predictions
                        self.model_candidates = candidates
                        self.model_engine = f"Volta  - {detected_channels}ch"
                        if isinstance(self.model_result, list) and len(self.model_result) == 6:
                            poi_vals = self.model_result.copy()
                            if poi_vals[2] == -1 and poi_vals[1] != -1:
                                # Correct POST point to End-of-fill + 2
                                poi_vals[2] = poi_vals[1] + 2
                        else:
                            self.model_result = -1  # Invalid result format

                except Exception as e:
                    import traceback

                    Log.e(TAG, f"Error using 'QModel Volta ': {e}")
                    for line in traceback.format_tb(sys.exc_info()[2]):
                        Log.d(line.strip())
                    self.model_result = -1  # Trigger fallback handling
                    # raise e
            if self.model_result == -1 and Constants.qmodel_indus_predict:
                Log.w("Auto-fitting points with QModel Indus... (may take a few seconds)")
                QtCore.QCoreApplication.processEvents()
                try:
                    with secure_open(self.loaded_datapath, "r", "capture") as f:
                        fh = BytesIO(f.read())
                        predictor = self.qmodel_indus_predictor
                        # self._QModel_create_new_progress_dialog()
                        # self.progressBarDiag.setRange(0, 100)  # percentage
                        predict_result = predictor.predict(
                            file_buffer=fh,
                            visualize=False,
                            progress_signal=self.indus_predict_progress,
                            use_partial_fills=self.partial_fills_checkbox.isChecked(),
                        )
                        predictions = []
                        candidates = []
                        for i in range(6):
                            poi_key = f"POI{i+1}"
                            poi_indices = predict_result.get(poi_key, {}).get("indices", [])
                            poi_confidences = predict_result.get(poi_key, {}).get("confidences", [])
                            best_pair = (poi_indices[0], poi_confidences[0])
                            predictions.append(best_pair[0])
                            candidates.append((poi_indices, poi_confidences))
                        self.model_run_this_load = True
                        self.model_result = predictions
                        self.model_candidates = candidates
                        self.model_engine = "Indus"
                        if isinstance(self.model_result, list) and len(self.model_result) == 6:
                            poi_vals = self.model_result.copy()
                            if poi_vals[2] == -1 and poi_vals[1] != -1:
                                # Correct POST point to End-of-fill + 2
                                poi_vals[2] = poi_vals[1] + 2
                        else:
                            self.model_result = -1  # try fallback model
                except Exception as e:
                    limit = None
                    t, v, tb = sys.exc_info()
                    from traceback import format_tb

                    a_list = ["Traceback (most recent call last):"]
                    a_list = a_list + format_tb(tb, limit)
                    a_list.append(f"{t.__name__}: {str(v)}")
                    for line in a_list:
                        Log.d(line)
                    Log.e(e)
                    Log.e(
                        TAG,
                        f"Error using 'QModel Indus'... Using a fallback model for auto-fitting.",
                    )
                    raise e  # debug only
                    self.model_result = -1  # try fallback model

            if self.model_result == -1 and Constants.qmodel_tweed_predict:
                try:
                    with secure_open(self.loaded_datapath, "r", "capture") as f:
                        csv_headers = next(f)

                        if isinstance(csv_headers, bytes):
                            csv_headers = csv_headers.decode()

                        if "Ambient" in csv_headers:
                            csv_cols = (2, 4, 6, 7)
                        else:
                            csv_cols = (2, 3, 5, 6)

                        data = loadtxt(f.readlines(), delimiter=",", skiprows=0, usecols=csv_cols)
                    relative_time = data[:, 0]
                    # temperature = data[:, 1]
                    resonance_frequency = data[:, 2]
                    dissipation = data[:, 3]

                    self.model_run_this_load = True
                    self.model_result = self.qmodel_tweed_predictor.IdentifyPoints(
                        self.loaded_datapath,
                        relative_time,
                        resonance_frequency,
                        dissipation,
                    )
                    self.model_engine = "Tweed"
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
                        Log.w("Model failed to auto-calculate POIs for this run!")
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

            if self.model_result != -1 and len(self.poi_markers) == 6:
                Log.i(f"[Auto-Fit] Auto-fit points with '{self.model_engine}' for this run.")
                for i, pm in enumerate(self.poi_markers):
                    idx = int(poi_vals[i])

                    if idx == -1:
                        # Mark "missing" points at end of data
                        idx = len(self.xs) - 1
                    elif idx <= 0:
                        Log.w(f"[Auto-Fit] Clamped POI{i+1} index {idx} to {1}")
                        idx = 1
                    elif idx >= len(self.xs):
                        Log.w(f"[Auto-Fit] Clamped POI{i+1} index {idx} to {len(self.xs)-1}")
                        idx = len(self.xs) - 1

                    # Update marker position to new index
                    pm.setValue(self.xs[idx])

                self._log_model_confidences()
                self.detect_change()
            else:
                Log.w(
                    "[Auto-Fit] No auto-fit points available for this run. Leaving points unchanged."
                )
        except ConnectionRefusedError:
            Log.d("Attempt to auto-fit with no run loaded. No action taken.")

        except ConnectionAbortedError:
            Log.d("User declined auto-fit restore prompt. No action taken.")

        except Exception as e:
            Log.e(f"Auto-fit restore failed: {str(e)}")

            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        finally:
            self.prediction_restored = True

    def _current_visible_poi_index(self):
        px = self.stateStep - 1
        return px if px < 2 else px + 1  # skip POI3 (index 2)

    def check_finished(self):
        if self.prediction_restored:
            # finished, but keep the dialog open to retain `wasCanceled()` state
            QtCore.QTimer.singleShot(1000, self._hide_qmodel_plot_overlay)
            self.timer.stop()

    def getPoints(self):
        """
        Advance the analysis workflow by one step, updating UI, markers, plots, and progress state.

        This method drives the point-selection workflow (with POI3 intentionally hidden). It:
        - Advances or clamps the internal step counter and maps it to the visible tutorial/step index (skipping the removed POI3).
        - Updates progress text, enables/disables navigation buttons, and switches the main graph view.
        - On initial step, attempts to auto-populate POI candidates using available prediction engines (QModel v3, QModel v2, QModel Tweed) and creates/mutates vertical POI markers when needed.
        - For intermediate steps it restricts which POI marker is movable, recenters/zooms the context plots around the current POI (skipping POI3), and updates star/current-point indicators and small-context plots.
        - On the summary/analysis step it finalizes marker positions, validates signatures if required, persists POIs and audit information to the run XML when there are unsaved changes, and launches the AnalyzeWorker in a background thread to run the heavy analysis pipeline.
        - Ensures any hidden POI (index 2 / POI3) is not shown or edited and adjusts all marker index calculations accordingly.

        Side effects:
        - Mutates many UI widgets and internal state (stateStep, poi_markers, moved_markers, model_result, model_candidates, model_engine, zoomLevel, gstars/star plots, progress bar, etc.).
        - May write <audit> and <points> entries to the run XML when changes are saved.
        - May start a background AnalyzeWorker thread that performs the full analysis and emits progress signals.

        No return value.
        """
        self.graphStack.setCurrentIndex(0)
        self.btn_Back.setEnabled(True)
        if self.stateStep != 7:
            self.btn_Next.setText("Next")
        self.stateStep += 1  # Increment to next step
        if self.stateStep == 6:  # Skip over hidden dot marker 8
            # Log.w("State step 6 trigger")
            self.stateStep = 7  # straight to Summary
        # Hide POI3 from UI steps: skip step 4 (POI3)
        # There are originally 6 points (POI1-POI6), POI3 is at index 2
        # When stepping, skip index 2
        step_num = self.stateStep + 2
        # Calculate the visible step index, skipping POI3
        visible_step = self._current_visible_poi_index() + 1
        # Only show 5 points to the user
        if step_num < 3 and self.tool_Modify.isChecked():
            self.parent.viewTutorialPage(7)  # analyze (summary)
        elif step_num in range(3, 8 + 1) and self.tool_Modify.isChecked():
            # Show 7.1, 7.2, 7.4, 7.5, 7.6 (skip 7.3)
            tutorial_ids = [round(7 + (visible_step) / 10, 2)]
            if visible_step in range(1, 7):
                tutorial_ids.append(7.7)
            self.parent.viewTutorialPage(tutorial_ids)  # analyze (precise point)
        else:  # "Modify" not checked or step_num > 8
            self.parent.viewTutorialPage([5, 6])  # analyze / prior results
        ax = self.graphWidget  # .plot(hour, temperature)
        ax1 = self.graphWidget1
        ax2 = self.graphWidget2
        ax3 = self.graphWidget3
        # Only show 5 points (skip POI3)
        w123 = self.stateStep in range(1, 6)
        self.lowerGraphs.setVisible(w123)
        # was_vis = ax1.isVisible()
        # if w123 and not was_vis:
        #     ax2.setFocus() # allow keyboard shortcuts left/right/up/down to work immediately
        # ax1.setVisible(w123)
        # ax2.setVisible(w123)
        # ax3.setVisible(w123)
        # When stateStep == 0, normal behavior
        if self.stateStep == 0:
            self._update_progress_value(
                12 * (step_num - 1),
                f"Step {step_num - 1} of 6: Select Rough Fill Points",
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

            # QModel Tweed
            poi_vals = []
            if len(self.poi_markers) != 6:
                self.model_result = -1
                self.model_candidates = None
                self.model_engine = "None"
                if Constants.qmodel_tweed_predict:
                    Log.w("Auto-fitting points with QModel Onyx... (may take a few seconds)")
                    QtCore.QCoreApplication.processEvents()
                    try:
                        with secure_open(self.loaded_datapath, "r", "capture") as f:
                            fh = BytesIO(f.read())
                            predictor = self.QModel_onyx_predictor
                            predict_result, detected_channels = predictor.predict(
                                file_buffer=fh, progress_signal=self.onyx_predict_progress
                            )
                            if not self.parent.num_channels:
                                self.parent.num_channels = detected_channels
                            Log.i(
                                TAG,
                                f"QModel Onyx Inference Complete. Detected Config: {detected_channels} Channel(s)",
                            )

                            predictions = []
                            candidates = []
                            for i in range(6):
                                poi_key = f"POI{i+1}"
                                data = predict_result.get(poi_key, {})
                                indices = data.get("indices", [-1])
                                confidences = data.get("confidences", [-1])
                                if not indices:
                                    indices = [-1]
                                if not confidences:
                                    confidences = [-1]
                                predictions.append(indices[0])
                                candidates.append((indices, confidences))
                            self.model_run_this_load = True
                            self.model_result = predictions
                            self.model_candidates = candidates
                            self.model_engine = f"Onyx - {detected_channels}ch"
                            if isinstance(self.model_result, list) and len(self.model_result) == 6:
                                poi_vals = self.model_result.copy()
                                if poi_vals[2] == -1 and poi_vals[1] != -1:
                                    # Correct POST point to End-of-fill + 2
                                    poi_vals[2] = poi_vals[1] + 2
                            else:
                                self.model_result = -1  # Invalid result format

                    except Exception as e:
                        # --- ERROR HANDLING ---
                        import traceback

                        Log.e(TAG, f"Error using 'QModel Onyx': {e}")
                        # Print full stack trace to debug log
                        for line in traceback.format_tb(sys.exc_info()[2]):
                            Log.d(line.strip())
                        self.model_result = -1  # Trigger fallback handling
                        # raise e # Uncomment for strict debugging
                if self.model_result == -1 and Constants.qmodel_volta_predict:
                    Log.w("Auto-fitting points with QModel Volta ... (may take a few seconds)")
                    QtCore.QCoreApplication.processEvents()
                    try:
                        with secure_open(self.loaded_datapath, "r", "capture") as f:
                            fh = BytesIO(f.read())
                            predictor = self.QModel_volta_predictor
                            # self._QModel_create_new_progress_dialog()
                            # self.progressBarDiag.setRange(0, 100)
                            predict_result, detected_channels = predictor.predict(
                                file_buffer=fh, progress_signal=self.volta_predict_progress
                            )
                            if not self.parent.num_channels:
                                self.parent.num_channels = detected_channels
                            Log.i(
                                TAG,
                                f"QModel Volta Inference Complete. Detected Config: {detected_channels} Channel(s)",
                            )

                            predictions = []
                            candidates = []
                            for i in range(6):
                                poi_key = f"POI{i+1}"
                                data = predict_result.get(poi_key, {})
                                indices = data.get("indices", [-1])
                                confidences = data.get("confidences", [-1])
                                if not indices:
                                    indices = [-1]
                                if not confidences:
                                    confidences = [-1]
                                predictions.append(indices[0])
                                candidates.append((indices, confidences))
                            self.model_run_this_load = True
                            self.model_result = predictions
                            self.model_candidates = candidates
                            self.model_engine = f"Volta  - {detected_channels}ch"
                            if isinstance(self.model_result, list) and len(self.model_result) == 6:
                                poi_vals = self.model_result.copy()
                                if poi_vals[2] == -1 and poi_vals[1] != -1:
                                    # Correct POST point to End-of-fill + 2
                                    poi_vals[2] = poi_vals[1] + 2
                            else:
                                self.model_result = -1  # Invalid result format

                    except Exception as e:
                        # --- ERROR HANDLING ---
                        import traceback

                        Log.e(TAG, f"Error using 'QModel Volta ': {e}")
                        # Print full stack trace to debug log
                        for line in traceback.format_tb(sys.exc_info()[2]):
                            Log.d(line.strip())
                        self.model_result = -1  # Trigger fallback handling
                        # raise e # Uncomment for strict debugging
                if self.model_result == -1 and Constants.qmodel_indus_predict:
                    Log.w("Auto-fitting points with QModel Indus... (may take a few seconds)")
                    QtCore.QCoreApplication.processEvents()
                    try:
                        with secure_open(self.loaded_datapath, "r", "capture") as f:
                            fh = BytesIO(f.read())
                            predictor = self.qmodel_indus_predictor
                            predict_result = predictor.predict(
                                file_buffer=fh,
                                visualize=False,
                                progress_signal=self.indus_predict_progress,
                                use_partial_fills=self.partial_fills_checkbox.isChecked(),
                            )

                            predictions = []
                            candidates = []
                            for i in range(6):
                                poi_key = f"POI{i+1}"
                                poi_indices = predict_result.get(poi_key, {}).get("indices", [])
                                poi_confidences = predict_result.get(poi_key, {}).get(
                                    "confidences", []
                                )
                                best_pair = (poi_indices[0], poi_confidences[0])
                                predictions.append(best_pair[0])
                                candidates.append(best_pair)
                            self.model_result = predictions
                            self.model_candidates = candidates
                            self.model_engine = "Indus"
                            if isinstance(self.model_result, list) and len(self.model_result) == 6:
                                poi_vals = self.model_result.copy()
                                if poi_vals[2] == -1 and poi_vals[1] != -1:
                                    # Correct POST point to End-of-fill + 2
                                    poi_vals[2] = poi_vals[1] + 2
                            else:
                                self.model_result = -1  # try fallback model
                    except Exception as e:
                        limit = None
                        t, v, tb = sys.exc_info()
                        from traceback import format_tb

                        a_list = ["Traceback (most recent call last):"]
                        a_list = a_list + format_tb(tb, limit)
                        a_list.append(f"{t.__name__}: {str(v)}")
                        for line in a_list:
                            Log.d(line)
                        Log.e(e)
                        Log.e(
                            "Error using 'QModel Indus'... Using a fallback model for auto-fitting."
                        )
                        raise e  # debug only
                        self.model_result = -1  # try fallback model

                if self.model_result == -1 and Constants.qmodel_tweed_predict:
                    try:
                        start_time = poi_vals[0] if len(poi_vals) > 0 else 0
                        stop_time = poi_vals[5] if len(poi_vals) > 5 else len(self.xs) - 1
                        model_starting_points = [
                            start_time,
                            None,
                            None,
                            None,
                            None,
                            stop_time,
                        ]
                        self.model_result = self.qmodel_tweed_predictor.IdentifyPoints(
                            data_path=self.loaded_datapath,
                            times=self.data_time,
                            freq=self.data_freq,
                            diss=self.data_diss,
                            start_at=model_starting_points,
                        )
                        self.model_engine = "Tweed"
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
                            Log.w("Model failed to auto-calculate POIs for this run!")
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
                    Log.e("Model returned insufficient points. Please manually select points.")
                    start_time = poi_vals[0] if len(poi_vals) > 0 else 0
                    stop_time = poi_vals[5] if len(poi_vals) > 5 else len(self.xs) - 1
                    fill_time = self.xs[stop_time] - self.xs[start_time]
                    poi2_time = self.xs[start_time] + (fill_time * 0.05)  # end of fill
                    poi3_time = self.xs[start_time] + (fill_time * 0.10)  # post
                    poi4_time = self.xs[start_time] + (fill_time * 0.25)  # blip1
                    poi5_time = self.xs[start_time] + (fill_time * 0.50)  # blip2

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
                    cur_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val)
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
                    poi_marker.sigPositionChangeFinished.connect(self.markerMoveFinished)
                    self.poi_markers.insert(-1, poi_marker)
            for idx, marker in enumerate(self.poi_markers):
                marker.setMovable(True)
                marker.setPen(color="blue")
                marker.addMarker("<|>")
                if idx == 2:
                    marker.setVisible(False)
            # self.AI_SelectTool_Frame.setVisible(False)  # Hide AI Tool
        # Only allow steps for POI1, POI2, POI4, POI5, POI6 (skip POI3)
        elif self.stateStep in range(1, 7):
            if self.stateStep + 2 == 3:  # stateStep 1 = Step 3 of 6
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
                            cur_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val)
                            poi_vals.append(cur_idx)
                        poi_vals.sort()
                        self.custom_poi_text.setText(f"{poi_vals}")
                        self.update_custom_pois()  # write POI markers in correct order
                    except Exception as e:
                        Log.e("Error: An exception occurred while sorting POI markers.")
                        Log.e(f"Error Details: {str(e)}")

                poi_vals = []
                for pm in self.poi_markers:  # already sorted
                    cur_val = pm.value()
                    cur_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val)
                    poi_vals.append(cur_idx)

                if self.model_engine == "Tweed" and Constants.qmodel_tweed_predict:
                    try:
                        # Run Model again, to get an initial automatic fine tuning of points prior to user input
                        model_starting_points = poi_vals.copy()  # NOTE: len(poi_vals) must equal 6
                        self.model_result = self.qmodel_tweed_predictor.IdentifyPoints(
                            data_path=self.loaded_datapath,
                            times=self.data_time,
                            freq=self.data_freq,
                            diss=self.data_diss,
                            start_at=model_starting_points,
                        )
                        self.model_engine = "Tweed"
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
                            Log.w("Model failed to auto-calculate POIs for this run!")
                            pass
                        else:
                            Log.e(
                                "Model encountered an unexpected response. Please manually select points."
                            )
                            pass
                    except:
                        Log.e("An error occurred while running the model and organizing markers.")

                    # sort poi_markers one more time, just in case model returned out-of-order points (which should never happen)
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
                                cur_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val)
                                poi_vals.append(cur_idx)
                            poi_vals.sort()
                            self.custom_poi_text.setText(f"{poi_vals}")
                            self.update_custom_pois()  # write POI markers in correct order
                        except Exception as e:
                            Log.e("Error: An exception occurred while sorting POI markers.")
                            Log.e(f"Error Details: {str(e)}")

                else:  # self.model_engine != "Tweed":
                    # do nothing here if "QModel v2" or "None"
                    pass

            # in stateStep 2 thru 6 (Steps 4 thru 8 of 6, skipping POI3)
            elif self.stateStep != 7:
                if self.stateStep == 3:
                    cur_val = self.poi_markers[self.stateStep - 2].value()
                    cur_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val)
                    new_idx = min(cur_idx + 2, len(self.xs) - 1)
                    if new_idx > cur_idx:
                        self.poi_markers[self.stateStep - 1].setValue(self.xs[int(new_idx)])
                    else:
                        Log.d(
                            "Current marker cannot be bumped forward without exceeding data bounds; leaving as-is."
                        )
                px = self._current_visible_poi_index()
                if px < 0 or px >= len(self.poi_markers):
                    return
                if self.poi_markers[px].value() < self.poi_markers[px - 1].value():
                    cur_val = self.poi_markers[px - 1].value()
                    cur_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val)
                    new_idx = min(cur_idx + 2, len(self.xs) - 1)
                    if new_idx > cur_idx:
                        self.poi_markers[self.stateStep - 1].setValue(self.xs[int(new_idx)])
                    else:
                        Log.d(
                            "Current marker cannot be bumped forward without exceeding data bounds; leaving as-is."
                        )
            self.zoomLevel = 1  # reset default zoom level for each point
            show_fits = 1.0 if self.stateStep >= 3 else 0.0
            show_scat = 0.1 if self.stateStep >= 3 else 1.0
            pad = 0.05 if self.stateStep >= 3 else 0.05
            self.fit_1.setAlpha(show_fits, False)
            self.fit_2.setAlpha(show_fits, False)
            self.fit_3.setAlpha(show_fits, False)
            self.scat_1.setAlpha(show_scat, False)
            self.scat_2.setAlpha(show_scat, False)
            self.scat_3.setAlpha(show_scat, False)
            # px is the index in poi_markers, skip POI3 (index 2)
            px = self._current_visible_poi_index()
            visible_ord = (px if px <= 1 else px - 1) + 1  # 1..5
            self._update_progress_value(
                12 * (step_num - 1),
                f"Step {step_num - 1} of 6: Select Precise Fill Point {visible_ord}",
            )
            ax.setTitle(None)
            if px < 0 or px >= len(self.poi_markers):
                return
            tt0 = self.poi_markers[0].value()
            tx0 = next(x for x, y in enumerate(self.xs) if y >= tt0)
            tt1 = self.poi_markers[px].value()
            tx1 = next(x for x, y in enumerate(self.xs) if y >= tt1)
            tt2 = self.poi_markers[-1].value()
            tx2 = next(x for x, y in enumerate(self.xs) if y >= tt2)
            ws = self.getContextWidth()[0]
            # Calculate safe index boundaries prior to setting ranges
            slice_start, slice_end = [tx1 - ws, tx1 + ws]
            clipped = False
            if slice_start < 0:
                slice_start = 0
                clipped = True
            if slice_end > len(self.xs) - 1:
                slice_end = len(self.xs) - 1
                clipped = True
            if slice_start >= slice_end:
                slice_start = slice_end - 1  # 0
                slice_end = slice_start + 1  # len(self.xs) - 1
                clipped = True
            ax.setXRange(self.xs[tx0], self.xs[tx2], padding=0.12)
            ax1.setXRange(self.xs[slice_start], self.xs[slice_end], padding=0)
            ax2.setXRange(self.xs[slice_start], self.xs[slice_end], padding=0)
            ax3.setXRange(self.xs[slice_start], self.xs[slice_end], padding=0)
            # Prevent empty slices
            if tx0 >= tx2:
                tx0 = 0
                tx2 = len(self.xs) - 1
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
            if self.stateStep >= 3:
                if not clipped:
                    ax1.setYRange(
                        np.min(self.ys_freq_fit[slice_start:slice_end]),
                        np.max(self.ys_freq_fit[slice_start:slice_end]),
                        padding=pad,
                    )
                    ax2.setYRange(
                        np.min(self.ys_diff_fit[slice_start:slice_end]),
                        np.max(self.ys_diff_fit[slice_start:slice_end]),
                        padding=pad,
                    )
                    ax3.setYRange(
                        np.min(self.ys_fit[slice_start:slice_end]),
                        np.max(self.ys_fit[slice_start:slice_end]),
                        padding=pad,
                    )
                else:  # clipped
                    Log.d(
                        "Skipping to next step, due to missing channel in data selection (represented by ValueError exception below):"
                    )
                    # skip to next view
                    Log.w(
                        f"Skipping Step {self.stateStep+2}... User indicated this point is missing from the dataset in Step 2."
                    )
                    if self.step_direction == "backwards":
                        self.action_back()  # repeat last action
                    else:
                        self.action_next()  # repeat last action
                    return  # do not execute remainder of this function, let the above nested 'action_next' call supercede
                pos1 = np.column_stack((self.xs[tx1], self.ys_freq_fit[tx1]))
                pos2 = np.column_stack((self.xs[tx1], self.ys_diff_fit[tx1]))
                pos3 = np.column_stack((self.xs[tx1], self.ys_fit[tx1]))
            else:
                ax1.setYRange(
                    np.min(self.ys_freq[slice_start:slice_end]),
                    np.max(self.ys_freq[slice_start:slice_end]),
                    padding=pad,
                )
                ax2.setYRange(
                    np.min(self.ys_diff[slice_start:slice_end]),
                    np.max(self.ys_diff[slice_start:slice_end]),
                    padding=pad,
                )
                ax3.setYRange(
                    np.min(self.ys[slice_start:slice_end]),
                    np.max(self.ys[slice_start:slice_end]),
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
                # Skip POI3 (index 2) for UI
                if idx == 2:
                    continue
                if (
                    idx == px - 1
                ):  # check last point, move this marker if it's out of time sequence from last one
                    if (
                        marker.value() >= self.poi_markers[px].value()
                    ):  # last marker time greater than this marker
                        t_idx = next(x for x, y in enumerate(self.xs) if y >= marker.value())
                        marker.setValue(self.xs[t_idx + 3])
                if idx != px:
                    t_idx = next(x for x, y in enumerate(self.xs) if y >= marker.value())
                    gstar_idxs.append(t_idx)
                marker.setMovable(idx == px)  # only current marker is movable
                marker.setPen(color=("blue" if idx == px else "blue"))
                marker.addMarker("<|>") if idx == px else marker.clearMarkers()
            if self.stateStep >= 3:
                pos1 = np.column_stack((self.xs[gstar_idxs], self.ys_freq_fit[gstar_idxs]))
                pos2 = np.column_stack((self.xs[gstar_idxs], self.ys_diff_fit[gstar_idxs]))
                pos3 = np.column_stack((self.xs[gstar_idxs], self.ys_fit[gstar_idxs]))
            else:
                pos1 = np.column_stack((self.xs[gstar_idxs], self.ys_freq[gstar_idxs]))
                pos2 = np.column_stack((self.xs[gstar_idxs], self.ys_diff[gstar_idxs]))
                pos3 = np.column_stack((self.xs[gstar_idxs], self.ys[gstar_idxs]))
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
            # ax.setTitle(f"Summary: All Selected POIs")
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
            # Prevent empty slices
            if tx0 >= tx2:
                tx0 = 0
                tx2 = len(self.xs) - 1
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
                if self.parent.signature_required and not self.parent.signature_received:
                    Log.e(f"Input Error: Initials do not match current user info ({self.initials})")
                    return
            self.btn_Back.setEnabled(True)
            self.btn_Next.setEnabled(False)
            poi_vals = []
            for marker in self.poi_markers:
                t_idx = next(x for x, y in enumerate(self.xs) if y >= marker.value())
                poi_vals.append(t_idx)
            poi_vals.sort()
            if self.unsaved_changes:
                Log.d("Storing new <points> in XML file")
                self.unsaved_changes = False
                if self.parent.signature_required:
                    self.appendAuditToXml()
                self.appendPointsToXml(poi_vals)
            # self.showAnalysis(poi_vals)
            # self.analyzer_task = threading.Thread(target=self.showAnalysis, args=(poi_vals,))
            # self.analyzer_task.start()
            allow_start = True
            if hasattr(self, "analyze_work"):
                if self.analyze_work.is_running():
                    Log.w("Double-click detected on Analyze action. Skipping duplicate action.")
                    allow_start = False
            if allow_start:
                self._update_progress_value(1, "Status: Starting...")
                self.graphStack.setCurrentIndex(1)
                self.analyzer_task = QtCore.QThread()
                self.analyze_work = AnalyzeWorker(
                    self,  # pass in parent
                    self.loaded_datapath,
                    self.xml_path,
                    poi_vals,
                    self.diff_factor if hasattr(self, "diff_factor") else None,
                )
                self.analyzer_task.started.connect(self.analyze_work.run)
                self.analyze_work.finished.connect(self.analyzer_task.quit)
                self.analyze_work.progress.connect(self._update_analyze_progress)
                self.analyze_work.finished.connect(self._update_progress_value)
                self.analyze_work.finished.connect(self.enable_buttons)

                # New progress dialog popup instead of run progress bar...
                self._show_analyze_plot_overlay()  # creates figure + overlay on main thread
                self.analyze_work.progress.connect(self._update_analyze_plot_overlay)
                self.analyze_work.finished.connect(self._hide_analyze_plot_overlay)
                # self._create_analyze_progress_dialog()
                # self.analyze_work.progress.connect(self._update_analyze_popup_progress)
                # self.analyze_work.finished.connect(self._close_analyze_progress_dialog)

                self.analyzer_task.start()
        self.setDotStepMarkers(step_num)

        # # Show/Hide QModel re-run button if on Step 2 and run has prior points
        # if step_num == 2 and len(self.poi_markers) == 6:
        #     self._position_floating_widget()
        #     self.QModel_widget.show()
        # elif self.QModel_widget.isVisible():
        #     self.QModel_widget.hide()

    # Maps Stepper index (0..7) -> legacy 1-based step_num used everywhere
    # else in this class (gotoStepNum, stateStep arithmetic, etc). The old
    # dots array was [dot1(status), dot2..dot7, dot8(dead), dot9, dot10], so
    # step_num skips 1 (the status dot) and 8 (permanently hidden, "for POI3
    # removal") - this table preserves that exact numbering without needing
    # to touch any of the arithmetic downstream that still speaks step_num.
    _STEP_NUMS = [2, 3, 4, 5, 6, 7, 9, 10]

    def _on_stepper_clicked(self, index: int) -> None:
        """Adapter from Stepper.stepClicked(index) to the legacy
        gotoStepNum(obj, step_num) call every other navigation path uses."""
        self.gotoStepNum(None, self._STEP_NUMS[index])

    def setDotStepMarkers(self, step_num):
        if step_num == 0:
            # No run loaded / reset - clear both the status dot and all
            # step progress.
            self.saved_state_dot.set_state("blank")
            self.stepper.reset()
            return
        if step_num == 1:
            # Run loaded/saved; wizard hasn't stepped into the numbered
            # steps yet for this load, so clear any prior step progress.
            self.saved_state_dot.set_state("saved")
            self.stepper.reset()
            return
        try:
            index = self._STEP_NUMS.index(step_num)
        except ValueError:
            Log.w(f"{TAG} setDotStepMarkers: unexpected step_num {step_num}")
            return
        self.stepper.set_current(index)

    def gotoStepNum(self, obj, step_num=1):
        """
        Navigate to a given analysis step using the step-dot controls and update UI state.

        This method interprets a clicked step dot (or the provided step_num) and advances or rewinds the AnalyzeProcess workflow accordingly. It enforces modify-mode rules, sets the step direction, handles the special "finished" step (10), validates prerequisites (loaded run and POIs), and triggers the appropriate actions: restoring QModel predictions for step 1, invoking getPoints() to move into the selected step, toggling modify mode via action_modify(), or emitting the next-button when appropriate. Side effects include updating self.stateStep, self.step_direction, progress bar text/value, enabling/disabling controls, showing/hiding graph panes, and calling other UI handlers (setDotStepMarkers, enable_buttons, _restore_qmodel_predictions, action_modify, getPoints).

        Parameters:
            obj: The UI object that triggered the call (unused - kept for
                signature compatibility with the direct saved-state-dot
                click wiring; step navigation from the numbered stepper
                comes through the _on_stepper_clicked adapter instead).
            step_num (int): Target step dot index (1-based, legacy numbering
                - see _STEP_NUMS).

        Notes:
        - If modify mode is disabled and the target step is less than 9, the method forces modify mode and returns (action_modify will re-enter this method).
        - Step 10 is treated as "Finished" and moves the UI to the results view.
        - Requires a loaded run (self.xml_path) and at least three POI markers to jump to analysis steps; otherwise it logs a warning and no step change occurs.
        """
        # if self.AI_SelectTool_Frame.isVisible():
        #     self.AI_SelectTool_Frame.setVisible(False)

        if self.allow_modify == False and step_num < 9:
            self.tool_Modify.setChecked(True)
            self.action_modify()  # self.tool_Modify.clicked.emit()
            return  # action_modify() always calls this function again

        # determine step direction
        if step_num < self.stateStep + 2:
            self.step_direction = "backwards"
        else:
            self.step_direction = "forwards"

        if step_num == 10:
            self.progressBar.setValue(100)  # Finished
            self.progressBar.setFormat('Finished: View most recent "Analyze" results')
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
            self._restore_qmodel_predictions()
            self.enable_buttons()
            # if PopUp.question(
            #     self,
            #     "Are you sure?",
            #     "Any manual points will be lost if you run QModel again.\n\nProceed?",
            # ):
            # self.parent.analyze_data(
            #     self.cBox_Devices.currentText(),
            #     self.get_folder_from_run(self.cBox_Runs.currentText()),
            #     None,
            # )  # force back to step 1 of 6
            # self.enable_buttons()
        elif enable_analyze:
            self.stateStep = step_num - 3
            self.getPoints()  # increment to next step
            self.enable_buttons()
        elif enable_analyze == False and step_num == 2:
            # special case: allow next action if dot is clicked instead of button
            self.tool_Next.clicked.emit()  # calls enable_buttons()
        else:
            Log.w("Please select begin and end points prior to using the step jumper dots.")

    def appendAuditToXml(self):
        data_path = self.loaded_datapath
        xml_path = data_path[0:-4] + ".xml" if self.xml_path == None else self.xml_path
        xml_params = {}
        if secure_open.file_exists(xml_path, "audit"):
            xml_text = ""
            with open(xml_path, "r", encoding="utf-8") as f:
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
                Log.w(f"Found invalid session: searching for user ({self.initials})")
                username = None  # not known in this context (yet)
                initials = self.initials
                salt = UserProfiles.find(username, initials)[1][:-4]
                userinfo = UserProfiles.get_user_info(f"{salt}.xml")
                username = userinfo[0]
                initials = userinfo[1]
                userrole = userinfo[2]

            audit_action = "ANALYZE"
            timestamp = self.parent.signed_at
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

            try:
                with open(xml_path, "w", encoding="utf-8") as f:
                    xml_str = run.toxml(encoding="ascii").decode(encoding="utf-8", errors="ignore")
                    f.write(xml_str)
                    Log.d(f"Added <audit> to XML file: {xml_path}")
            except OSError as ose:  # FileNotFoundError
                Log.e(f"Filesystem error writing XML: {xml_path}")
                Log.e("Error Details:", ose.strerror)
                self.detect_change()
            except UnicodeError as ue:  # UnicodeEncodeError, UnicodeDecodeError
                Log.e(f"Unicode error writing XML: {xml_path}")
                Log.e("Error Details:", ue.reason)
                self.detect_change()

    def appendPointsToXml(self, poi_vals):
        data_path = self.loaded_datapath
        xml_path = data_path[0:-4] + ".xml" if self.xml_path == None else self.xml_path
        xml_params = {}
        if secure_open.file_exists(xml_path, "audit"):
            xml_text = ""
            with open(xml_path, "r", encoding="utf-8") as f:
                xml_text = f.read()
            if isinstance(xml_text, bytes):
                xml_text = xml_text.decode()
            run = minidom.parseString(xml_text)
            xml = run.documentElement

            # create new points element
            recorded_at = (
                self.parent.signed_at
                if self.parent.signature_required
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

            try:
                with open(xml_path, "w", encoding="utf-8") as f:
                    xml_str = run.toxml(encoding="ascii").decode(encoding="utf-8", errors="ignore")
                    f.write(xml_str)
                    Log.d(f"Added <points> to XML file: {xml_path}")
            except OSError as ose:  # FileNotFoundError
                Log.e(f"Filesystem error writing XML: {xml_path}")
                Log.e("Error Details:", ose.strerror)
                self.detect_change()
            except UnicodeError as ue:  # UnicodeEncodeError, UnicodeDecodeError
                Log.e(f"Unicode error writing XML: {xml_path}")
                Log.e("Error Details:", ue.reason)
                self.detect_change()

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
            Log.d(f"Marker {marker_idx} has been moved by the user! Flagged for model tuning.")
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
            # Prevent empty slices
            if tx0 >= tx2:
                tx0 = 0
                tx2 = len(self.xs) - 1
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
            pad = 0.05 if self.stateStep >= 3 else 0.05
            # Calculate safe index boundaries prior to setting ranges
            slice_start, slice_end = [tx1 - ws, tx1 + ws]
            if slice_start < 0:
                slice_start = 0
            if slice_end > len(self.xs) - 1:
                slice_end = len(self.xs) - 1
            if slice_start >= slice_end:
                slice_start = 0
                slice_end = len(self.xs) - 1
            ax1.setXRange(self.xs[slice_start], self.xs[slice_end], padding=0)
            ax2.setXRange(self.xs[slice_start], self.xs[slice_end], padding=0)
            ax3.setXRange(self.xs[slice_start], self.xs[slice_end], padding=0)
            if self.stateStep >= 3:
                ax1.setYRange(
                    np.min(self.ys_freq_fit[slice_start:slice_end]),
                    np.max(self.ys_freq_fit[slice_start:slice_end]),
                    padding=pad,
                )
                ax2.setYRange(
                    np.min(self.ys_diff_fit[slice_start:slice_end]),
                    np.max(self.ys_diff_fit[slice_start:slice_end]),
                    padding=pad,
                )
                ax3.setYRange(
                    np.min(self.ys_fit[slice_start:slice_end]),
                    np.max(self.ys_fit[slice_start:slice_end]),
                    padding=pad,
                )
                pos1 = np.column_stack((self.xs[tx1], self.ys_freq_fit[tx1]))
                pos2 = np.column_stack((self.xs[tx1], self.ys_diff_fit[tx1]))
                pos3 = np.column_stack((self.xs[tx1], self.ys_fit[tx1]))
            else:
                ax1.setYRange(
                    np.min(self.ys_freq[slice_start:slice_end]),
                    np.max(self.ys_freq[slice_start:slice_end]),
                    padding=pad,
                )
                ax2.setYRange(
                    np.min(self.ys_diff[slice_start:slice_end]),
                    np.max(self.ys_diff[slice_start:slice_end]),
                    padding=pad,
                )
                ax3.setYRange(
                    np.min(self.ys[slice_start:slice_end]),
                    np.max(self.ys[slice_start:slice_end]),
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
            with open(self.xml_path, "r", encoding="utf-8") as f:
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
                None if self.parent == None else self.parent.controls_window.username.text()[6:]
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
            self.bWorker = QueryRunInfoWidget(
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
            Log.e(TAG, f"Item with name '{old_name} ({date})' not found in the combo box.")
        for key in list(self.run_names.keys()):  # Use list to avoid runtime changes
            if f"{old_name}:" in key:
                # Extract the part of the key after the ':'
                _, after_colon = key.split(":", 1)  # Split at the first ':'
                # Store the value and remove the entry
                value = self.run_names.pop(key)
                after_colon = after_colon.strip()
                break
        value = self.run_timestamps.pop(key)
        self.run_timestamps[f"{new_name}:{after_colon}"] = value
        self.run_names[f"{new_name}:{after_colon}"] = new_name
        self.text_Created.setText(f"Loaded: {new_name} ({date})")
        if hasattr(self, "_batched_runs") and self._batched_runs:
            self._current_run = self.text_Created.text()
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
                self.update_run(i)
            self.cBox_Runs.setCurrentIndex(loaded_idx)

    def analyze_data(self, data_path: str) -> None:
        """Load and prepare run data for analysis.

        Reads the CSV at `data_path`, sanitizes backward-time jumps, computes
        smoothed resonance / dissipation / difference curves, restores or predicts
        POIs from a companion XML file or predictive models, applies optional drop-effect
        corrections and difference-factor optimization, then populates all plotting
        widgets and internal state for the analysis workflow.

        Args:
            data_path: Absolute or relative path to the run CSV file.  A companion
                XML file with the same base name may be read to restore prior POIs
                and run parameters.  The CSV must span at least 3 seconds; shorter
                runs are rejected with a logged error and return early.

        Side Effects:
            Mutates many instance attributes including `xs`, `ys`, `ys_freq`,
            `ys_diff`, `ys_fit`, `poi_markers`, `smooth_factor`,
            `loaded_datapath`, `stateStep`, `model_result`, and
            `model_candidates`.  Performs several UI updates (clears/repopulates
            plot widgets, updates progress text, enables/disables navigation
            buttons).

        Note:
            When a QModel finds all six valid POIs, the routine
            auto-advances to the summary step (`stateStep = 6`) and marks the
            dataset as changed.  On any non-fatal failure the method falls back to
            manual POI selection rather than re-raising.
        """
        self._init_analysis_state(data_path)
        relative_time = resonance_frequency = dissipation = None
        poi_vals, start_stop, curves = [], [0, -1], {}

        try:
            Log.i(f"Analysis file = {data_path}")
            relative_time, _, resonance_frequency, dissipation = self._load_run_data(data_path)

            if relative_time[-1] < 3:
                Log.e("ERROR: Data run must be at least 3 seconds in total runtime to analyze.")
                return  # Finally still executes; the plotting section below is skipped

            poi_vals, fill_type = self._load_xml_pois(data_path)
            self._apply_poi_state(poi_vals, fill_type)

            poi_vals = self._run_model_prediction(
                poi_vals, relative_time, resonance_frequency, dissipation
            )

            self.graphWidget.removeItem(self._text2)
            self._text1.setHtml(
                "<span style='font-size: 14pt'>Showing data for analysis... </span>"
            )

            dissipation, resonance_frequency = self._apply_signal_corrections(
                dissipation, resonance_frequency, poi_vals
            )

            curves, start_stop = self._compute_signal_curves(
                relative_time,
                resonance_frequency,
                dissipation,
                poi_vals,
                savgol_filter,
                argrelextrema,
            )

            self._update_analyze_progress(100, "Reading Run Data...")

        except Exception:
            self.progress_value_steps.clear()
            self._log_traceback()
            Log.w("An error occurred loading this run! Please manually select points for Analysis.")

        finally:
            curves = self._recover_missing_curves(
                curves, relative_time, resonance_frequency, dissipation, savgol_filter
            )
            self._wait_for_progress_bar()

        self._render_analysis_plots(curves, poi_vals, start_stop)
        self._save_analysis_state(curves, relative_time, resonance_frequency, dissipation)
        self._advance_analysis_step(poi_vals)

    def _init_analysis_state(self, data_path):
        """Reset per-run analysis state and configure navigation buttons."""
        self.stateStep = -1
        self.loaded_datapath = data_path
        self.btn_Back.setEnabled(False)
        self.btn_Next.setEnabled(True)

    def _load_run_data(
        self, data_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read a run CSV and sanitize backward-time rows.

        Detects the presence of an "Ambient" temperature column in the header to
        select the correct column indices, then removes any rows where the relative
        timestamp goes backward using a single vectorized pass.

        Args:
            data_path: Path to the run CSV file.  The file must be openable via
                `secure_open` and must contain at least the standard QATCH columns
                for time, temperature, resonance frequency, and dissipation.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A four-element
            tuple of `(relative_time, temperature, resonance_frequency,
            dissipation)` as 1-D float arrays, sanitized of time-jump artefacts.
            The original file is never modified.
        """
        with secure_open(data_path, "r", "capture") as f:
            header = next(f)
            if isinstance(header, bytes):
                header = header.decode()
            csv_cols = (2, 4, 6, 7) if "Ambient" in header else (2, 3, 5, 6)
            data = loadtxt(f.readlines(), delimiter=",", usecols=csv_cols)

        relative_time = data[:, 0]
        temperature = data[:, 1]
        resonance_frequency = data[:, 2]
        dissipation = data[:, 3]

        # Vectorized time-jump detection(s).
        backward = np.where(np.diff(relative_time) < 0)[0]
        if backward.size:
            Log.w(f"Warning: time jump(s) observed at the following indices: {backward.tolist()}")

            keep = np.ones(len(relative_time), dtype=bool)
            keep[backward] = False
            relative_time = relative_time[keep]
            temperature = temperature[keep]
            resonance_frequency = resonance_frequency[keep]
            dissipation = dissipation[keep]

            Log.w("Time jumps removed from dataset for analysis purposes (original file unchanged)")

        return relative_time, temperature, resonance_frequency, dissipation

    def _load_xml_pois(self, data_path: str):
        """Parse the companion XML file to restore prior POI indices and run params.

        Uses `xml.etree.ElementTree` to extract the most-recent
        `<points>` and `<params>` blocks.  Child elements are iterated directly.

        Args:
            data_path: Path to the run CSV.  The XML file is expected at the same
                location with a `.xml` extension (or at `self.xml_path` when
                that override is set).  If the file does not exist or
                `self.askForPOIs` is `False`, defaults are returned immediately.

        Returns:
            tuple[list[int], int]: A two-element tuple of
            `(poi_vals, fill_type)` where `poi_vals` is a sorted list of
            integer POI indices (empty when none are found) and `fill_type` is
            an integer channel count (-1` when not present in the XML).
        """
        poi_vals = []
        fill_type = -1
        if not self.askForPOIs:
            return poi_vals, fill_type

        xml_path = (data_path[:-4] + ".xml") if self.xml_path is None else self.xml_path
        if not os.path.exists(xml_path):
            Log.w(TAG, f'Missing XML file: Expected at "{xml_path}" for this run.')
            return poi_vals, fill_type

        root = ET.parse(xml_path).getroot()

        # Get POI point values
        all_points = root.findall(".//points")
        if all_points:
            for child in all_points[-1]:  # most-recent <points> block
                raw = child.get("value", "")
                try:
                    poi_vals.append(int(raw))
                except ValueError:
                    Log.e(f'Point value "{raw}" in XML is not an integer.')
            poi_vals.sort()
        else:
            Log.d("No points found in XML file for this run.")

        # Run parameters
        all_params = root.findall(".//params")
        if all_params:
            for child in all_params[-1]:  # most-recent <params> block
                name = child.get("name", "")
                raw = child.get("value", "")
                try:
                    if name == "fill_type":
                        fill_type = int(raw)
                except ValueError:
                    Log.e(f'Param "{name}" in XML is not an integer.')
        else:
            Log.d("No params found in XML file for this run.")

        return poi_vals, fill_type

    def _apply_poi_state(self, poi_vals: List[Any], fill_type: int) -> None:
        """Updates instance state flags based on restored POIs and data fill type.

        This method synchronizes the UI state with data recovered from an XML file.
        If a full set of 6 POIs is found, the workflow skips directly to the summary
        view (Step 6). If partial markers exist, it disables the initial POI prompt
        to allow for manual re-analysis.

        Args:
            poi_vals: A list of POI values retrieved from the XML.
            fill_type: The number of channels detected in the current data set.
        """
        # Reset transient state flags
        self.show_analysis_immediately = False
        self.model_run_this_load = False
        self.prior_points_in_xml = False

        if self.askForPOIs:
            if len(poi_vals) == 6:
                self.askForPOIs = False
                self.prior_points_in_xml = True
                self.stateStep = 6  # Skip straight to summary
                Log.d(f"Found prior POIs from XML file: {poi_vals}")

            elif len(self.poi_markers) > 0:
                # Re-analyse Step 1; prevent auto-advancing to Summary
                self.askForPOIs = False

        # Update channel configuration
        Log.d(f"Number of channels (fill_type): {fill_type}")
        self.parent.num_channels = fill_type

    def _run_model_prediction(
        self,
        poi_vals: List[Any],
        relative_time: np.ndarray,
        resonance_frequency: np.ndarray,
        dissipation: np.ndarray,
    ) -> List[Any]:
        """Orchestrates the POI auto-fitting fallback chain.

        Attempts to predict POIs the current fallback strategy:
            1. QModel Onyx (onyx)
            2. QModel Volta
            3. QModel Indus
            4. QModel Tweed.

        Args:
            poi_vals: Current list of POI values.
            relative_time: Array of time offsets for the dataset.
            resonance_frequency: Array of frequency shifts.
            dissipation: Array of dissipation values.

        Returns:
            The best available list of POIs. If pre-existing markers exist,
            returns a truncated list containing only [start, end].
        """
        # Exit early if a result already exists from a prior load
        if self.model_result != -1:
            return poi_vals

        # Initialize/Reset model state
        self.model_result = -1
        self.model_candidates = None
        self.model_engine = "None"

        poi_vals = self._try_qmodel_onyx(poi_vals)

        if self.model_result == -1:
            poi_vals = self._try_qmodel_volta(poi_vals)

        if self.model_result == -1:
            poi_vals = self._try_qmodel_indus(poi_vals)

        if self.model_result == -1:
            poi_vals = self._try_model_data(
                poi_vals, relative_time, resonance_frequency, dissipation
            )

        # If partial markers exist, force manual entry for the middle points
        if self.poi_markers:
            return [poi_vals[0], poi_vals[-1]]

        return poi_vals

    def _try_qmodel_onyx(self, poi_vals: List[int]) -> List[int]:
        """Attempts POI prediction using the QModel Onyx (onyx) engine.

        NOTE: The POST point is not predicted with this model and is simply filled!

        Args:
            poi_vals: The current list of POI indices.

        Returns:
            The updated list of POI indices if successful; otherwise, returns
            the original `poi_vals` and sets `self.model_result` to -1 for fallback.
        """
        if not Constants.qmodel_onyx_predict:
            return poi_vals

        # Skip inference if XML already provided a full set of valid points
        if self.prior_points_in_xml:
            self.model_result = poi_vals
            self.model_engine = "Onyx skipped (using prior points)"
            return poi_vals

        msg = "Auto-fitting points with QModel Onyx..."
        Log.w(f"{msg} (may take a few seconds)")
        self._text1.setHtml(f"<span style='font-size: 14pt'>{msg}</span>")
        self.graphWidget.addItem(self._text2, ignoreBounds=True)
        QtCore.QCoreApplication.processEvents()

        try:
            # File access and inference
            with secure_open(self.loaded_datapath, "r", "capture") as f:
                fh = BytesIO(f.read())

            predict_result, detected_channels = self.QModel_onyx_predictor.predict(
                file_buffer=fh, progress_signal=self.onyx_predict_progress
            )

            if not self.parent.num_channels:
                self.parent.num_channels = detected_channels

            # Extract predictions
            predictions = []
            candidates = []

            for i in range(6):
                data = predict_result.get(f"POI{i + 1}", {})
                indices = data.get("indices", [-1]) or [-1]
                confidences = data.get("confidences", [-1]) or [-1]

                predictions.append(indices[0])
                candidates.append((indices, confidences))

            # Update model state
            self.model_run_this_load = True
            self.model_result = predictions
            self.model_candidates = candidates
            self.model_engine = f"Onyx - {detected_channels}ch"

            # Validate
            # NOTE: Legacy POI3 (POST point) is ignored here!
            if isinstance(self.model_result, list) and len(self.model_result) == 6:
                poi_vals = list(self.model_result)
                if poi_vals[2] == -1 and poi_vals[1] != -1:
                    poi_vals[2] = poi_vals[1] + 2
            else:
                self.model_result = -1

        except Exception as e:
            self._log_traceback(debug=True)
            Log.e(f"Error using 'QModel Onyx': {e}")
            Log.e(TAG, "Falling back to next available model.")
            self.model_result = -1

        return poi_vals

    def _try_qmodel_volta(self, poi_vals: List[int]) -> List[int]:
        """Attempts POI prediction using the QModel Volta  engine.

        NOTE: The POST point is not predicted with this model and is simply filled!

        Args:
            poi_vals: The current list of POI indices.

        Returns:
            The updated list of POI indices if successful; otherwise, returns
            the original `poi_vals` and sets `self.model_result` to -1 for fallback.
        """
        if not Constants.qmodel_volta_predict:
            return poi_vals

        # Skip inference if XML already provided a full set of valid points
        if self.prior_points_in_xml:
            self.model_result = poi_vals
            self.model_engine = "Volta  skipped (using prior points)"
            return poi_vals

        msg = "Auto-fitting points with QModel Volta ..."
        Log.w(f"{msg} (may take a few seconds)")
        self._text1.setHtml(f"<span style='font-size: 14pt'>{msg}</span>")
        self.graphWidget.addItem(self._text2, ignoreBounds=True)
        QtCore.QCoreApplication.processEvents()

        try:
            # File access and inference
            with secure_open(self.loaded_datapath, "r", "capture") as f:
                fh = BytesIO(f.read())

            # self._QModel_create_new_progress_dialog()
            # self.progressBarDiag.setRange(0, 100)

            predict_result, detected_channels = self.QModel_volta_predictor.predict(
                file_buffer=fh, progress_signal=self.volta_predict_progress
            )
            # QtCore.QTimer.singleShot(1000, self.progressBarDiag.hide)

            if not self.parent.num_channels:
                self.parent.num_channels = detected_channels

            # Extract predictions
            predictions = []
            candidates = []

            for i in range(6):
                data = predict_result.get(f"POI{i + 1}", {})
                indices = data.get("indices", [-1]) or [-1]
                confidences = data.get("confidences", [-1]) or [-1]

                predictions.append(indices[0])
                candidates.append((indices, confidences))

            # Update model state
            self.model_run_this_load = True
            self.model_result = predictions
            self.model_candidates = candidates
            self.model_engine = f"Volta  - {detected_channels}ch"

            # Validate
            # NOTE: Legacy POI3 (POST point) is ignored here!
            if isinstance(self.model_result, list) and len(self.model_result) == 6:
                poi_vals = list(self.model_result)
                if poi_vals[2] == -1 and poi_vals[1] != -1:
                    poi_vals[2] = poi_vals[1] + 2
            else:
                self.model_result = -1

        except Exception as e:
            self._log_traceback(debug=True)
            Log.e(f"Error using 'QModel Volta ': {e}")
            Log.e(TAG, "Falling back to next available model.")
            self.model_result = -1

        return poi_vals

    def _try_qmodel_indus(self, poi_vals: List[int]) -> List[int]:
        """Attempts POI prediction using the QModel Indus engine.

        NOTE: The POST point is not predicted with this model and is simply filled!

        Args:
            poi_vals: The current list of POI indices.

        Returns:
            The updated list of POI indices if successful; otherwise, returns
            the original `poi_vals` and sets `self.model_result` to -1.
        """
        if not Constants.qmodel_indus_predict:
            return poi_vals

        msg = "Auto-fitting points with QModel Indus..."
        Log.w(f"{msg} (may take a few seconds)")
        self._text1.setHtml(f"<span style='font-size: 14pt'>{msg}</span>")
        self.graphWidget.addItem(self._text2, ignoreBounds=True)
        QtCore.QCoreApplication.processEvents()

        try:
            with secure_open(self.loaded_datapath, "r", "capture") as f:
                fh = BytesIO(f.read())

            predict_result = self.qmodel_indus_predictor.predict(
                file_buffer=fh,
                visualize=False,
                progress_signal=self.indus_predict_progress,
                use_partial_fills=self.partial_fills_checkbox.isChecked(),
            )

            predictions = []
            candidates = []

            for i in range(6):
                data = predict_result.get(f"POI{i + 1}", {})
                indices = data.get("indices", [-1]) or [-1]
                confidences = data.get("confidences", [-1]) or [-1]
                predictions.append(indices[0])
                candidates.append((indices, confidences))

            # Update model state
            self.model_run_this_load = True
            self.model_result = predictions
            self.model_candidates = candidates
            self.model_engine = "Indus"

            # Validate
            # NOTE: Legacy POI3 (POST point) is ignored here!
            if isinstance(self.model_result, list) and len(self.model_result) == 6:
                poi_vals = list(self.model_result)
                if poi_vals[2] == -1 and poi_vals[1] != -1:
                    poi_vals[2] = poi_vals[1] + 2
            else:
                self.model_result = -1

        except Exception as e:
            self._log_traceback(debug=True)
            Log.e(f"Error using 'QModel Indus': {e}")
            self.model_result = -1

        return poi_vals

    def _try_model_data(
        self,
        poi_vals: List[int],
        relative_time: np.ndarray,
        resonance_frequency: np.ndarray,
        dissipation: np.ndarray,
    ) -> List[int]:
        """Attempts POI prediction using the legacy QModel Tweed predictor.

        This serves as the final algorithmic fallback tier if everything went wrong :)

        NOTE: Unlike the YOLO models, this method directly clears and repopulates
        the provided `poi_vals`!

        Args:
            poi_vals: The current list of POI indices to be updated.
            relative_time: Array of time offsets.
            resonance_frequency: Array of frequency shift data.
            dissipation: Array of dissipation data.

        Returns:
            The updated list of POI indices. Returns the original list if the
            feature is disabled or a critical error occurs.
        """
        if not Constants.qmodel_tweed_predict:
            return poi_vals

        try:
            self.model_run_this_load = True
            # IdentifyPoints returns a list of indices or -1 on failure
            result = self.qmodel_tweed_predictor.IdentifyPoints(
                self.loaded_datapath, relative_time, resonance_frequency, dissipation
            )
            self.model_result = result
            self.model_engine = "Tweed"

            if isinstance(result, list):
                poi_vals.clear()
                self.model_select = []
                self.model_candidates = []

                for point in result:
                    self.model_select.append(0)
                    if isinstance(point, list):
                        self.model_candidates.append(point)
                        # Extract the index (first element of the first candidate)
                        poi_vals.append(point[0][0])
                    else:
                        # Data format: simple index integer
                        self.model_candidates.append([point])
                        poi_vals.append(point)

            elif result == -1:
                Log.w("QModel Tweed failed to auto-calculate points for this run.")
            else:
                Log.e("QModel Tweed returned an unexpected response format.")

        except Exception as e:
            self._log_traceback(debug=False)
            Log.e(f"Legacy QModel Tweed execution failed: {e}")

        return poi_vals

    def _apply_signal_corrections(
        self, dissipation: np.ndarray, resonance_frequency: np.ndarray, poi_vals: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Applies drop-effect correction vectors to signal data.

        If the drop-effect correction UI option is enabled, this method invokes
        a correction algorithm to adjust the dissipation and resonance frequency
        arrays based on the provided POIs.

        Args:
            dissipation: Array of raw dissipation values.
            resonance_frequency: Array of raw resonance frequency values.
            poi_vals: List of indices identifying key POIs.

        Returns:
            A tuple of (dissipation, resonance_frequency). Arrays will be the
            corrected versions if the feature is active and valid; otherwise,
            the original input arrays are returned.
        """
        if not self.drop_effect_cancelation_checkbox.isChecked():
            return dissipation, resonance_frequency

        corrected_diss, corrected_rf = self._correct_drop_effect(
            self.loaded_datapath, poi_vals, "process"
        )

        if corrected_diss is not None:
            dissipation = corrected_diss

        if corrected_rf is not None:
            resonance_frequency = corrected_rf

        return dissipation, resonance_frequency

    def _compute_smooth_params(self, xs: np.ndarray) -> Tuple[int, int, bool, int]:
        """Derives Savitzky-Golay window sizes and the 90-second split index.

        Calculates smoothing factors based on total runtime. It defines a 'split index'
        at the 90-second mark to allow for different smoothing intensities between
        the initial phase and extended data.

        Args:
            xs: Array of time samples (assumed to be sorted).

        Returns:
            A tuple containing:
                - smooth_factor (int): Primary window size (forced odd, 3-69).
                - t_split (int): Index of the first sample past 90s.
                - extend_data (bool): True if there is sufficient data beyond 90s.
                - extend_smf (int): Coarser window size for extended data (forced odd).
        """
        total_runtime = xs[-1]

        # Compute primary smooth factor
        smooth_factor = int(total_runtime * Constants.smooth_factor_ratio)
        if smooth_factor % 2 == 0:
            smooth_factor += 1
        smooth_factor = max(3, min(69, smooth_factor))

        Log.i(TAG, f"Total run time: {total_runtime} secs")
        Log.d(TAG, f"Smoothing: {smooth_factor} | Applying for first 90s.")

        # np.searchsorted uses binary search O(log n) vs the O(n) generator next()
        t_split = int(np.searchsorted(xs, 90, side="right"))
        extend_data = total_runtime > 90

        # Compute extended smoothing factor (forced to odd)
        extend_smf = int(smooth_factor / 20)
        if extend_smf % 2 == 0:
            extend_smf += 1

        # Check if there is enough "padding" to apply extended smoothing
        if extend_data and len(xs) < (t_split + 2 * extend_smf):
            Log.w(
                "Insufficient points after 90s for effective downsampling. "
                "Disabling extended smoothing."
            )
            t_split = len(xs)
            extend_data = False

        return smooth_factor, t_split, extend_data, extend_smf

    def _apply_savgol(
        self,
        signal: np.ndarray,
        smooth_factor: int,
        t_split: int,
        extend_data: bool,
        extend_smf: int,
        savgol_filter: Callable[..., np.ndarray],
        deriv: int = 0,
    ) -> np.ndarray:
        """Applies a Savitzky-Golay filter with an optional dual-window strategy.

        Smoothes the input signal using a primary window size up to the split point.
        If extended data processing is enabled, it applies a secondary (usually
        coarser) window to the remainder of the signal and concatenates the results.

        Args:
            signal: The input numerical array to be smoothed.
            smooth_factor: Window length for the primary segment (must be odd).
            t_split: The index at which to split the signal for dual-window processing.
            extend_data: Whether to apply a different smoothing factor beyond t_split.
            extend_smf: Target window length for the extended segment.
            savgol_filter: The filter function (e.g., scipy.signal.savgol_filter).
            deriv: The order of the derivative to compute. Defaults to 0 (smoothing).

        Returns:
            A smoothed NumPy array of the same length as the input signal.
        """
        # Process the primary segment (up to t_split)
        primary_window = max(smooth_factor, 3)
        result = savgol_filter(signal[:t_split], primary_window, 1, deriv)

        if not extend_data:
            return result

        # Process the extended segment
        remainder = signal[t_split:]
        rem_len = len(remainder)

        if rem_len > 0:
            # Window size must be odd and: 1 < window_size <= segment_length
            ext_window = min(rem_len, extend_smf)
            if ext_window % 2 == 0:
                ext_window = max(3, ext_window - 1)
            ext_window = max(3, ext_window)

            ext = savgol_filter(remainder, ext_window, 1, deriv)
            return np.concatenate((result, ext))

        return result

    def _compute_signal_curves(
        self,
        relative_time: np.ndarray,
        resonance_frequency: np.ndarray,
        dissipation: np.ndarray,
        poi_vals: list,
        savgol_filter: Callable,
        argrelextrema: Callable,
    ) -> Tuple[dict, list[int, int]]:
        """Compute all smoothed signal curves and derive initial run-boundary candidates.

        Applies Savitzky-Golay smoothing to the raw dissipation, resonance-
        frequency shift, and computed difference signals.  Uses the two highest
        2nd-derivative maxima of the smoothed dissipation to locate the rough
        analysis window, then refines the start/stop indices using a noise-floor
        threshold on the difference curve.

        Args:
            relative_time: 1-D monotonically increasing array of sample timestamps
                in seconds.
            resonance_frequency: 1-D array of raw resonance-frequency measurements
                aligned with `relative_time`.
            dissipation: 1-D array of raw dissipation measurements aligned with
                `relative_time`.
            poi_vals: Current candidate POI indices.  Used only to suppress
                spurious boundary-detection warnings when a model or XML has
                already provided valid points.
            savgol_filter: `scipy.signal.savgol_filter`.
            argrelextrema: `scipy.signal.argrelextrema`.

        Returns:
            tuple[dict, list[int, int]]: A two-element tuple `(curves,
            start_stop)` where `curves` is a dict with keys `xs`, `ys`,
            `ys_freq`, `ys_diff`, `ys_fit`, `ys_freq_fit`,
            `ys_diff_fit`, `ys_diss_2ndd`, and `smooth_factor`; and
            `start_stop` is `[begin_idx, end_idx]` representing the estimated
            analysis window in sample-index space.

        NOTE: Several arrays that existed in the original implementation
            (`ys_diss_diff_avg`/ `zeros3`, `minima_*`, `ys_diff_fine`,
            `ys_diff_diff`, `ys_diff_2ndd`) were computed but never used. These have been
            removed!
        """
        xs = relative_time
        n = len(xs)
        smooth_factor, t_split, extend_data, extend_smf = self._compute_smooth_params(xs)

        def sg(sig: np.ndarray, deriv: int = 0) -> np.ndarray:
            """Convenience closure over the smoothing parameters"""
            return self._apply_savgol(
                sig, smooth_factor, t_split, extend_data, extend_smf, savgol_filter, deriv
            )

        # Dissipation smoothing and 2nd derivative
        ys_fit_raw = sg(dissipation)
        ys_diss_diff = sg(ys_fit_raw, deriv=1)
        ys_diss_2ndd = sg(ys_diss_diff, deriv=1)

        # Rough boundary detection via the two highest 2nd-derivative maxima
        maxima_idx = argrelextrema(ys_diss_2ndd, np.greater)[0]
        maxima_val = ys_diss_2ndd[maxima_idx]

        if len(maxima_idx) < 2:
            # Degenerate signal: fall back to a full-width window
            t_start, t_stop = 0, n - 1
        else:
            top2_pos = np.argpartition(maxima_val, -2)[-2:]
            top2_idx = np.sort(maxima_idx[top2_pos])  # ascending by sample position
            t_start = int(top2_idx[0])
            t_stop = int(top2_idx[1]) + 3 * smooth_factor

        no_model_no_poi = not self.model_run_this_load and len(poi_vals) == 0
        if t_stop < n / 2 or t_stop >= n:
            if no_model_no_poi:
                Log.w(f"Stop time index was {t_stop} out of {n} but that seems unlikely!")
                Log.w('Please confirm "End Point" during Step 1 point selection.')
            t_stop = n - 1
        if t_stop - t_start < n / 3 or t_start > n / 2:
            if no_model_no_poi:
                Log.w(f"Start time index was {t_start} out of {n} but that seems unlikely!")
                Log.w('Please confirm "Begin Point" during Step 1 point selection.')
            t_start = 100

        # Baseline window (first ~0.5-2 s of the run)
        if xs[t_start] < 0.5:
            t_0p5 = 0
        else:
            t_0p5 = min(int(np.searchsorted(xs, 0.5, side="right")), n - 1)

        if xs[t_start] < 2.0:
            t_1p0 = t_start
        else:
            t_1p0 = min(int(np.searchsorted(xs, 2.0, side="right")), n - 1)

        if t_0p5 == t_1p0:
            t_1p0 = min(int(np.searchsorted(xs, xs[t_1p0] + 1.5, side="right")), n - 1)
            if t_1p0 == t_0p5:
                # still equal at the array boundary
                t_1p0 = min(t_0p5 + 1, n - 1)

        # Scale dissipation relative to the baseline resonance frequency
        avg = np.mean(resonance_frequency[t_0p5:t_1p0])
        scale = avg / 2
        ys = dissipation * scale
        ys_fit = ys_fit_raw * scale
        offset = np.amin(ys_fit)
        ys -= offset
        ys_fit -= offset

        # Resonance frequency shift and smoothed fit
        ys_freq = avg - resonance_frequency
        ys_freq_fit = sg(ys_freq)

        # Difference factor and difference curve
        diff_factor = Constants.default_diff_factor
        if self.difference_factor_optimizer_checkbox.isChecked():
            self.diff_factor = self._optimize_curve(self.loaded_datapath)
        if hasattr(self, "diff_factor"):
            diff_factor = self.diff_factor

        ys_diff = ys_freq - diff_factor * ys

        # Invert when the drop was applied at the outlet (negative initial deltas)
        if np.mean(np.abs(ys_freq_fit)) < np.mean(np.abs(diff_factor * ys_fit)) and abs(
            ys_diff[t_1p0:].min()
        ) > 5 * abs(ys_diff[t_1p0:].max()):
            Log.w("Inverting DIFFERENCE curve due to negative initial fill deltas")
            ys_diff *= -1

        ys_diff_fit = sg(ys_diff)

        Log.d(f"Difference factor: {diff_factor:.3f}x")
        Log.d("Setting diff_factor on Advanced Settings menu")
        self.tbox_diff_factor.setText(f"{diff_factor:.3f}")

        # Noise-floor parameters for start/stop thresholding
        eh1 = float(abs(np.amax(ys_diff[t_0p5:t_1p0])))
        em2 = float(np.amax(ys_diff_fit))
        am2 = int(np.argmax(ys_diff_fit))
        eh2 = eh1 * 2  # start threshold: 2x baseline noise

        # Start candidate
        t0 = t_start
        above = np.where(ys_diff_fit[t_1p0 + 1 :] > 5 * eh2)[0]
        if above.size:
            t0 = t_1p0 + 1 + int(above[0])
        elif no_model_no_poi:
            Log.w("Failed to locate rough start point using noise floor approximation.")
            Log.w('Please confirm "Begin Point" during Step 1 point selection.')

        # Walk backward to the true fill start
        going_up = ys[t0] < 5
        while True:
            if not (0 <= t0 < n):
                if no_model_no_poi:
                    Log.w("Hit a limit (start)")
                t0 = 0
                break
            if ys[t0] < 5:
                t0 += 1
                if not going_up:
                    break
            else:
                t0 -= 1
                if going_up:
                    break

        # End candidate.
        t1 = am2
        below = np.where(ys_diff_fit[am2:] < em2 - eh2)[0]
        if below.size:
            t1 = am2 + int(below[0])
        elif no_model_no_poi:
            Log.w("Failed to locate rough end point using noise floor approximation.")
            Log.w('Please confirm "End Point" during Step 1 point selection.')

        # Walk backward to the true end
        while True:
            if t1 - 50 < 0:
                if no_model_no_poi:
                    Log.w("Hit a limit (end)")
                t1 = n - 1
                break
            if ys_diff_fit[t1 - 50] > ys_diff_fit[t1]:
                t1 -= 1
            else:
                break

        curves = {
            "xs": xs,
            "ys": ys,
            "ys_freq": ys_freq,
            "ys_diff": ys_diff,
            "ys_fit": ys_fit,
            "ys_freq_fit": ys_freq_fit,
            "ys_diff_fit": ys_diff_fit,
            "ys_diss_2ndd": ys_diss_2ndd,
            "smooth_factor": smooth_factor,
        }
        return curves, [t0, t1]

    def _recover_missing_curves(
        self,
        curves: dict,
        relative_time: np.ndarray | None,
        resonance_frequency: np.ndarray | None,
        dissipation: np.ndarray | None,
        savgol_filter,
    ) -> dict:
        """Reconstruct any signal arrays missing from curves after a partial failure.

        Called unconditionally from the `finally` block of `analyze_data` so
        that the UI can always fall back to manual POI selection even when signal
        processing threw partway through.  Each key is only written when absent,
        so pre-existing values are always preserved.

        NOTE: Reconstruction order matters: later entries depend on keys built earlier
        in this function (e.g. `ys_diff` needs `ys_freq` and `ys`).

        Args:
            curves: Partially or fully populated signal dict.  Modified in-place
                and returned.  Keys that are already present are never overwritten.
            relative_time: Raw timestamp array, or `None` when data loading
                failed before it could be read.
            resonance_frequency: Raw resonance-frequency array, or `None`.
                Together with `dissipation`, required for any reconstruction;
                if either is `None` the dict is returned unchanged.
            dissipation: Raw dissipation array, or `None`.
            savgol_filter: `scipy.signal.savgol_filter`

        Returns:
            dict: The same curves dict, with any previously absent required keys
            now populated with zero-smoothing fallback values.
        """

        required_keys: frozenset = frozenset(
            {
                "xs",
                "ys",
                "ys_fit",
                "ys_freq",
                "ys_freq_fit",
                "ys_diff",
                "ys_diff_fit",
                "ys_diss_2ndd",
                "smooth_factor",
            }
        )

        if not (required_keys - curves.keys()):
            return curves

        # Without raw sensor data no reconstruction is possible.
        if resonance_frequency is None or dissipation is None:
            return curves

        Log.w("Correcting missing parameters for manual point selection (no smoothing)...")

        # Use the first sample as the baseline resonance value.
        avg = float(resonance_frequency[0])

        if "xs" not in curves and relative_time is not None:
            curves["xs"] = relative_time

        # ys / ys_fit
        if "ys" not in curves or "ys_fit" not in curves:
            scaled = dissipation * (avg / 2)
            offset = float(np.amin(scaled))
            curves["ys"] = scaled - offset
            curves["ys_fit"] = scaled - offset  # independent array; same values

        if "ys_freq" not in curves:
            curves["ys_freq"] = avg - resonance_frequency

        if "ys_freq_fit" not in curves:
            curves["ys_freq_fit"] = curves["ys_freq"]

        if "ys_diff" not in curves:
            curves["ys_diff"] = curves["ys_freq"] - Constants.default_diff_factor * curves["ys"]
        if "ys_diff_fit" not in curves:
            curves["ys_diff_fit"] = curves["ys_diff"]
        if "ys_diss_2ndd" not in curves:
            try:
                ys_diss_diff = savgol_filter(curves["ys_fit"], 2, 1, 1)
                curves["ys_diss_2ndd"] = savgol_filter(ys_diss_diff, 2, 1, 1)
            except Exception:
                Log.e("Unable to calculate 2nd derivative of 'ys' data!")
                curves["ys_diss_2ndd"] = curves["ys_fit"]

        if "smooth_factor" not in curves:
            curves["smooth_factor"] = 3

        return curves

    def _log_traceback(self, debug: bool = False) -> None:
        """Logs the full traceback of the currently handled exception.

        Args:
            debug: If True, logs each line at the DEBUG level (Log.d).
                If False, logs at the ERROR level (Log.e).
        """
        log_func: Callable[[str], None] = Log.d if debug else Log.e
        exc_traceback: str = traceback.format_exc()
        for line in exc_traceback.strip().split("\n"):
            log_func(line)

    def _wait_for_progress_bar(self) -> None:
        """Spins the Qt event loop until the background scan completes or times out.

        This method prevents the UI from freezing during a background scan by
        manually processing pending events. It includes a safety timeout to
        prevent infinite loops. Once finished, it connects the progress bar's
        value change signal to the internal updater.
        """
        timeout_limit = 300
        iterations = 0

        Log.d("Waiting on progress bar to finish background scan...")

        while self.progress_value_scanning and iterations < timeout_limit:
            iterations += 1
            QtCore.QCoreApplication.processEvents()
            time.sleep(0.01)
        try:
            self.progressBar.valueChanged.disconnect(self._update_progress_value)
        except (TypeError, RuntimeError):
            pass
        self.progressBar.valueChanged.connect(self._update_progress_value)
        Log.d(f"Progress bar wait completed after {iterations} iterations. Proceeding...")

    def _render_analysis_plots(
        self, curves: Dict[str, Any], poi_vals: List[int], start_stop: Tuple[int, int]
    ) -> None:
        """Coordinates the rendering of signal data and analysis markers.

        This top-level coordinator manages the visual pipeline by setting up
        graph axes, drawing the primary signal curves, and overlaying channel/POI markers.
        It delegates specific rendering tasks to specialized sub-helper methods.

        Args:
            curves: A dictionary containing signal data. Expected keys include:
                - 'xs': The shared x-axis (time) array.
                - Individual signal keys (e.g., 'df', 'dg') containing y-axis arrays.
            poi_vals: A list of indices representing the detected POI.
            start_stop: A tuple of (start_index, stop_index) defining the active
                analysis window.
        """
        self._setup_graph_axes(curves)
        self._plot_signal_curves(curves)
        self._add_poi_markers(curves["xs"], poi_vals, start_stop)

    def _setup_graph_axes(self, curves: Dict[str, Any]) -> None:
        """Configures titles, labels, ranges, grids, and legends for all four axes.

        This method clears existing plots, initializes the progress UI, sets specific
        titles and colors for four distinct graph widgets, and establishes the
        coordinate ranges based on the provided curve data.

        Args:
            curves: A dictionary containing the data to be plotted.
                Expected keys include:
                - 'xs': The x-axis data (typically time).
                - 'ys': Primary y-axis data.
                - 'ys_freq': Frequency-related y-axis data.
                - 'ys_diff': Difference-related y-axis data.
        """
        ax, ax1, ax2, ax3 = (
            self.graphWidget,
            self.graphWidget1,
            self.graphWidget2,
            self.graphWidget3,
        )
        xs = curves["xs"]
        ys = curves["ys"]
        ys_freq = curves["ys_freq"]
        ys_diff = curves["ys_diff"]

        ax.clear()
        ax1.clear()
        ax2.clear()
        ax3.clear()
        # ax.clear() can reset axis pens on some pyqtgraph versions, so
        # theme colors must be reapplied every time these axes are cleared.
        self._apply_pg_theme()

        self._update_progress_value(1, "Step 1 of 6: Select Begin and End Points")
        self.setDotStepMarkers(1)

        ax.setTitle(None)
        ax1.setTitle("Resonance", color=SIGNAL_COLORS["resonance"])
        ax2.setTitle("Difference", color=SIGNAL_COLORS["difference"])
        ax3.setTitle("Dissipation", color=SIGNAL_COLORS["dissipation"])

        axis_label_color = tok_css(ThemeManager.instance().tokens()["plot_text_normal"])
        style = {"color": axis_label_color, "font-size": "12px"}
        ax.showAxis("left")
        ax.setLabel("left", "Frequency (Hz)", **style)
        ax.showAxis("bottom")
        ax.setLabel("bottom", "Time (secs)", **style)

        # pg's native corner autoscale button is redundant now that the
        # overview card header has explicit Zoom/Move-point controls.
        ax.hideButtons()
        ax1.hideButtons()
        ax2.hideButtons()
        ax3.hideButtons()

        ax.showGrid(x=True, y=True)
        ax1.showGrid(x=True, y=True)
        ax2.showGrid(x=True, y=True)
        ax3.showGrid(x=True, y=True)

        ax.setXRange(0, xs[-1], padding=0.05)
        ax.setYRange(0, max(np.amax(ys_freq), np.amax(ys), np.amax(ys_diff)), padding=0.05)

        self.lowerGraphs.setVisible(False)

    def _plot_signal_curves(self, curves: Dict[str, np.ndarray]) -> None:
        """Adds fit lines, scatter dots, and star highlights to all graph widgets.

        This method populates the main graph and the three specialized sub-graphs
        (Resonance, Difference, Dissipation) with raw data points, fitted curves,
        and interactive star markers representing POI.

        Args:
            curves: A dictionary containing NumPy arrays for plotting.
                Required keys:
                - 'xs': Shared x-axis time values.
                - 'ys', 'ys_freq', 'ys_diff': Raw signal data.
                - 'ys_fit', 'ys_freq_fit', 'ys_diff_fit': Calculated fit line data.
        """
        ax, ax1, ax2, ax3 = (
            self.graphWidget,
            self.graphWidget1,
            self.graphWidget2,
            self.graphWidget3,
        )
        xs = curves["xs"]
        ys = curves["ys"]
        ys_freq = curves["ys_freq"]
        ys_diff = curves["ys_diff"]
        ys_fit = curves["ys_fit"]
        ys_freq_fit = curves["ys_freq_fit"]
        ys_diff_fit = curves["ys_diff_fit"]

        mask = np.arange(0, len(xs), 1)
        noPen = pg.mkPen(color=(255, 255, 255), width=0, style=QtCore.Qt.DotLine)

        # Main graph - fit lines
        self.fit1 = ax.plot(
            xs[mask], ys_freq_fit[mask], pen=SIGNAL_COLORS["resonance"], name="Resonance"
        )
        self.fit2 = ax.plot(
            xs[mask], ys_diff_fit[mask], pen=SIGNAL_COLORS["difference"], name="Difference"
        )
        self.fit3 = ax.plot(xs[mask], ys_fit[mask], pen=SIGNAL_COLORS["dissipation"], name="Dissipation")

        # Main graph - scatter dots (nearly transparent)
        self.scat1 = ax.plot(
            xs[mask],
            ys_freq[mask],
            pen=noPen,
            symbol="o",
            symbolSize=5,
            symbolBrush=SIGNAL_COLORS["resonance"],
        )
        self.scat2 = ax.plot(
            xs[mask],
            ys_diff[mask],
            pen=noPen,
            symbol="o",
            symbolSize=5,
            symbolBrush=SIGNAL_COLORS["difference"],
        )
        self.scat3 = ax.plot(
            xs[mask],
            ys[mask],
            pen=noPen,
            symbol="o",
            symbolSize=5,
            symbolBrush=SIGNAL_COLORS["dissipation"],
        )
        self.scat1.setAlpha(0.01, False)
        self.scat2.setAlpha(0.01, False)
        self.scat3.setAlpha(0.01, False)

        # Sub-graphs - fit lines
        self.fit_1 = ax1.plot(
            xs[mask], ys_freq_fit[mask], pen=SIGNAL_COLORS["resonance"], name="Resonance"
        )
        self.fit_2 = ax2.plot(
            xs[mask], ys_diff_fit[mask], pen=SIGNAL_COLORS["difference"], name="Difference"
        )
        self.fit_3 = ax3.plot(
            xs[mask], ys_fit[mask], pen=SIGNAL_COLORS["dissipation"], name="Dissipation"
        )

        # Sub-graphs - scatter dots
        self.scat_1 = ax1.plot(
            xs[mask],
            ys_freq[mask],
            pen=noPen,
            symbol="o",
            symbolSize=5,
            symbolBrush=SIGNAL_COLORS["resonance"],
        )
        self.scat_2 = ax2.plot(
            xs[mask],
            ys_diff[mask],
            pen=noPen,
            symbol="o",
            symbolSize=5,
            symbolBrush=SIGNAL_COLORS["difference"],
        )
        self.scat_3 = ax3.plot(
            xs[mask],
            ys[mask],
            pen=noPen,
            symbol="o",
            symbolSize=5,
            symbolBrush=SIGNAL_COLORS["dissipation"],
        )

        # Star markers (current-POI highlights)
        pos1 = np.column_stack((xs[0], ys_freq[0]))
        pos2 = np.column_stack((xs[0], ys_diff[0]))
        pos3 = np.column_stack((xs[0], ys[0]))

        self.star1 = pg.ScatterPlotItem(pos=pos1, symbol="star", size=25, brush="black")
        self.star2 = pg.ScatterPlotItem(pos=pos2, symbol="star", size=25, brush="black")
        self.star3 = pg.ScatterPlotItem(pos=pos3, symbol="star", size=25, brush="black")
        ax1.addItem(self.star1)
        ax2.addItem(self.star2)
        ax3.addItem(self.star3)

        self.gstars1 = pg.ScatterPlotItem(pos=pos1, symbol="star", size=10, brush="gray")
        self.gstars2 = pg.ScatterPlotItem(pos=pos2, symbol="star", size=10, brush="gray")
        self.gstars3 = pg.ScatterPlotItem(pos=pos3, symbol="star", size=10, brush="gray")
        ax1.addItem(self.gstars1)
        ax2.addItem(self.gstars2)
        ax3.addItem(self.gstars3)

    def _add_poi_markers(self, xs: np.ndarray, poi_vals: List[int], start_stop: List[int]) -> None:
        """Places movable InfiniteLine POI markers on the main graph.

        This method initializes vertical markers (POI) on the
        graphWidget. If `poi_vals` is provided, it undergoes validation to ensure
        indices are within the bounds of the `xs` array. Validated `poi_vals`
        will take precedence over the `start_stop` values.

        Args:
            xs: The x-axis data array used to determine marker coordinate bounds.
            poi_vals: A list of integer indices representing predefined POIs.
                Indices that are out of bounds (except -1) are reset to -1 and logged.
            start_stop: A fallback list of integer indices used if `poi_vals`
                is not provided or to define the initial marker set.
        """
        ax = self.graphWidget

        if poi_vals:
            for i, pt in enumerate(poi_vals):
                if pt != -1 and not 0 <= pt < len(xs):
                    Log.w(f"Model point {pt} cannot be used. Skipping point {i + 1}.")
                    poi_vals[i] = -1
            start_stop = poi_vals

        self.poi_markers = []
        for idx, pt in enumerate(start_stop):
            marker = pg.InfiniteLine(
                pos=xs[pt],
                angle=90,
                pen="b",
                bounds=[xs[0], xs[-1]],
                movable=True,
            )
            marker.setPen(color="blue")
            marker.addMarker("<|>")
            if idx == 2:
                marker.setVisible(False)
            ax.addItem(marker)
            marker.sigPositionChangeFinished.connect(self.markerMoveFinished)
            self.poi_markers.append(marker)

    def _save_analysis_state(
        self,
        curves: Optional[Dict[str, Any]],
        relative_time: Union[np.ndarray, List[float]],
        resonance_frequency: Union[np.ndarray, List[float]],
        dissipation: Union[np.ndarray, List[float]],
    ) -> None:
        """Persists computed signal arrays and raw sensor data to instance attributes.

        This method acts as the primary data synchronization point, moving processed
        results from local function scope into the class instance attributes. This
        allows downstream analysis steps, exports, or UI updates to access the
        most recent calculation results.

        Args:
            curves: A dictionary containing processed signal arrays and metadata.
                If None or empty, the save operation is aborted.
                Expected keys include:
                - 'xs', 'ys', 'ys_freq', 'ys_diff': Raw/Processed signal data.
                - 'ys_fit', 'ys_freq_fit', 'ys_diff_fit': Fit results.
                - 'ys_diss_2ndd': Second derivative or dissipation data.
                - 'smooth_factor': The value used for signal smoothing.
            relative_time: Array of time values relative to the experiment start.
            resonance_frequency: Array of calculated resonance frequency values.
            dissipation: Array of calculated dissipation values.
        """
        if not curves:
            return  # Nothing to save i.e., a very early failure occurred

        self.xs = curves["xs"]
        self.ys = curves["ys"]
        self.ys_freq = curves["ys_freq"]
        self.ys_diff = curves["ys_diff"]
        self.ys_fit = curves["ys_fit"]
        self.ys_freq_fit = curves["ys_freq_fit"]
        self.ys_diff_fit = curves["ys_diff_fit"]
        self.ys_diss_2ndd = curves["ys_diss_2ndd"]
        self.smooth_factor = curves["smooth_factor"]
        self.data_time = relative_time
        self.data_freq = resonance_frequency
        self.data_diss = dissipation

    def _advance_analysis_step(self, poi_vals: List[int]) -> None:
        """Determines the next UI state based on model results and POI completeness.

        This method acts as a workflow controller. It evaluates whether the
        automated model successfully identified all six POIs.
        If successful, it advances the application to the summary step
        (Step 6); otherwise, it prompts the user for manual intervention.

        Args:
            poi_vals: A list of indices representing the POI
                identified by the model or loaded from a previous session.
                A complete set must contain exactly six points to trigger
                automated advancement.
        """
        if self.model_run_this_load and self.stateStep != 6:
            # Model produced a guess and there is no prior analysis to show
            if len(poi_vals) == 6:
                Log.i("Model successfully calculated POIs for this dataset.")
                Log.d(f"Model Result = {self.model_engine}: {self.model_result}")
                self.stateStep = 6
                self._log_model_confidences()
                self.detect_change()
            else:
                Log.e("Please manually select POIs to Analyze this dataset.")
        else:
            # Model was not run this load
            if self.stateStep == 6:
                Log.i("Loaded POIs from a prior run of Analyze tool.")
            else:
                Log.e("Please manually select POIs to Analyze this dataset.")

        if self.stateStep == 6:
            Log.d("Skipping to summary step")
            self.getPoints()  # show summary when all six points are already known

        if self.show_analysis_immediately:
            Log.d("Showing analysis immediately")
            self.getPoints()  # confirm and run full analysis for prior-result view

    # def _position_floating_widget(self):
    #     pos_X = 20 + self.parent.mode_window.pos().x() + self.parent.mode_window.ui0.modemenu.width() + \
    #         (self.width() - self.QModel_widget.width()) // 2
    #     pos_Y = self.parent.mode_window.pos().y() + 250
    #     self.QModel_widget.move(pos_X, pos_Y)

    def _log_model_confidences(self):
        """Logs the confidence scores for the model's candidate points.

        Iterates through `self.model_candidates` to log the confidence percentage
        of the primary prediction for specific named points ("start", "end_fill",
        "ch1", "ch2", "ch3"). The log severity (Info, Warning, Error) scales
        dynamically based on the confidence level. The "post" point is explicitly ignored.

        Note:
            The "Tweed" model engine does not support confidence logging. If the
            engine is set to "Tweed", this method will log an informational message
            and exit early.

        Raises:
            Exception: Captures and logs any errors encountered while parsing or
                logging the confidences from the model response.
        """
        if self.model_engine == "Tweed":
            Log.i(
                tag=f"[{self.model_engine}]", msg="Confidence logging is not available for Tweed."
            )
            return

        if self.model_candidates is None:
            Log.w(
                tag=f"[{self.model_engine}]",
                msg="No model candidates available to log confidences!",
            )
            return

        def get_logger_for_confidence(confidence):
            logger = Log.e  # less than 33%
            if confidence > 66:
                logger = Log.i  # greater than 66%
            elif confidence > 33:
                logger = Log.w  # from 33% to 66%
            return logger

        try:
            point_names = ["start", "end_fill", "post", "ch1", "ch2", "ch3"]
            for i, (_, confidences) in enumerate(self.model_candidates):
                if i == 2:
                    # do not print confidence of "post" point, it doesn't matter
                    continue

                point_name = point_names[i]

                # issue: QModel is returning a single `float` instead of a `list`
                if type(confidences) is float:
                    confidences = [confidences]

                confidence = 100 * confidences[0] if len(confidences) > 0 else 0
                num_spaces = len(point_names[1]) - len(point_name) + 1

                get_logger_for_confidence(confidence)(
                    tag=f"[{self.model_engine}]",
                    msg=f"Confidence @ {point_name}:{' '*num_spaces}{confidence:2.0f}%",
                )
        except Exception:
            Log.e(
                tag=f"[{self.model_engine}]", msg="Error logging confidences from QModel response."
            )

    def resizeEvent(self, event):
        # # Position relative to main window
        # if self.QModel_widget.isVisible():
        #     QtCore.QTimer.singleShot(100, self._position_floating_widget)
        # if self.AI_SelectTool_Frame.isVisible():
        #     self.AI_SelectTool_Frame.setVisible(
        #         False
        #     )  # require re-click to show popup tool incorrect position
        pass

    def _optimize_curve(self, data_path: str) -> float:
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
                optimizer = DifferenceFactorOptimizer(data_path, file_header)
                optimal_factor, lb, rb = optimizer.optimize()
                Log.i(
                    TAG,
                    f"Using difference factor {optimal_factor} optimized between {lb}s and {rb}s.",
                )

            if optimal_factor is not None:
                Log.d(TAG, f"Reporting difference factor of {optimal_factor}.")
                return optimal_factor
            else:
                Log.d(
                    TAG,
                    f"No optimal difference factor found, reporting default of {Constants.default_diff_factor}.",
                )
                return Constants.default_diff_factor
        except Exception as e:
            Log.e(
                TAG,
                f"Difference factor optimizer failed due to error. Using default factor.",
            )
            Log.e(TAG, f"Error Details: {str(e)}")
            return Constants.default_diff_factor

    def _correct_drop_effect(
        self, file_path: str, poi_vals: list, context: str = "process"
    ) -> tuple:
        """
        Corrects the dissipation and resonance drop effect in the provided file.

        This method reads the contents of the file specified by `file_path`,
        applies a drop effect correction algorithm using the specified
        difference factor, and returns the corrected data if successful.

        Args:
            file_path (str): Path to the file containing the data to be corrected.
            poi_vals (list): List of points-of-interest passed from QModel or user-input.
            context (str): Indicate the context of the call. Values: 'process', 'worker'.

        Returns:
            tuple or None: The corrected data if the correction is successful;
            otherwise, returns None and logs that the original data will be used.

        Logs:
            - Debug: Indicates the start of the drop effect cancellation process with the difference factor.
            - Info: Indicates the drop effect result when successful.
            - Warning: Indicates the drop effect result when not result was returned.
            - Error: Indicates the drop effect result when an unhandled error occurred.

        Raises:
            IOError: If there is an issue opening or reading the file.
            Exception: For any unexpected errors during the correction process.
        """
        try:
            with secure_open(file_path, "r", "capture") as f:
                file_buffer = BytesIO(f.read())
                if hasattr(self, "diff_factor"):
                    diff_factor = self.diff_factor
                else:
                    diff_factor = 2.0

                Log.d(
                    TAG,
                    f"Performing drop effect cancelation with difference factor {diff_factor}.",
                )

                dec = DropEffectCorrection(
                    file_path=file_path,
                    file_buffer=file_buffer,
                    initial_diff_factor=diff_factor,
                    bounds=poi_vals,
                )
                corrected_data = dec.correct_drop_effects(
                    save_corrections=True if context == "worker" else False
                )

            if corrected_data is not None:
                Log.i(TAG, f"Drop effect cancelation successful.")
                return corrected_data
            else:
                Log.w(TAG, f"Drop effect cancelation failed. Using original data.")
                return [None, None]
        except Exception as e:
            Log.e(
                TAG,
                f"Drop effect cancelation failed due to error. Using original data.",
            )
            Log.e(TAG, f"Error Details: {str(e)}")
            return [None, None]

class ResizeFilter(QtCore.QObject):
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        self._draw_pending = False
        self._draw_delay = 250  # ms

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Resize:
            self._resize_time = monotonic()
            if not self._draw_pending:
                self._draw_pending = True
                QtCore.QTimer.singleShot(self._draw_delay, self._draw_idle)
        return super().eventFilter(obj, event)

    def _draw_idle(self):
        # convert secs -> ms: compare ms to ms
        if (monotonic() - self._resize_time) * 1000 < self._draw_delay:
            # resize event still occurring, try again later
            QtCore.QTimer.singleShot(self._draw_delay, self._draw_idle)
        else:
            # resize event finished, hysteresis elapsed: redraw!
            self.worker.place_text_avoiding_data()
            self._draw_pending = False
