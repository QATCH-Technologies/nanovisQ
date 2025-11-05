try:
    from QATCH.core.constants import Constants
    from QATCH.common.logger import Logger as Log
    from QATCH.common.architecture import Architecture
except:
    print("Running VisQAI as standalone app")

    class Log:
        def d(tag, msg=""): print("DEBUG:", tag, msg)
        def i(tag, msg=""): print("INFO:", tag, msg)
        def w(tag, msg=""): print("WARNING:", tag, msg)
        def e(tag, msg=""): print("ERROR:", tag, msg)

from xml.dom import minidom
from PyQt5 import QtCore, QtGui, QtWidgets
# from random import randint
import copy
import os
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit
import webbrowser
from PyQt5.QtPrintSupport import QPrinter
from scipy.interpolate import interp1d
from typing import Optional
from typing import TYPE_CHECKING

try:
    from src.io.file_storage import SecureOpen
    from src.models.formulation import Formulation, ViscosityProfile
    from src.models.ingredient import Ingredient, Protein, Surfactant, Stabilizer, Salt, Buffer, ProteinClass, Excipient
    from src.models.predictor import Predictor
    from src.db.db import Database
    from src.processors.sampler import Sampler
    from src.threads.executor import Executor, ExecutionRecord
    from src.utils.constraints import Constraints
    from src.utils.icon_utils import IconUtils
    from src.utils.list_utils import ListUtils
    from src.view.checkable_combo_box import CheckableComboBox
    from src.view.table_view import TableView, Color
    from src.view.constraints_ui import ConstraintsUI
    from src.io.parser import Parser
    if TYPE_CHECKING:
        from src.view.frame_step2 import FrameStep2
        from src.view.main_window import VisQAIWindow

except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.io.file_storage import SecureOpen
    from QATCH.VisQAI.src.models.formulation import Formulation, ViscosityProfile
    from QATCH.VisQAI.src.models.ingredient import Ingredient, Protein, Surfactant, Stabilizer, Salt, Buffer, ProteinClass, Excipient
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.processors.sampler import Sampler
    from QATCH.VisQAI.src.threads.executor import Executor, ExecutionRecord
    from QATCH.VisQAI.src.utils.constraints import Constraints
    from QATCH.VisQAI.src.utils.icon_utils import IconUtils
    from QATCH.VisQAI.src.utils.list_utils import ListUtils
    from QATCH.VisQAI.src.view.checkable_combo_box import CheckableComboBox
    from QATCH.VisQAI.src.view.table_view import TableView, Color
    from QATCH.VisQAI.src.view.constraints_ui import ConstraintsUI
    from QATCH.VisQAI.src.io.parser import Parser
    if TYPE_CHECKING:
        from QATCH.VisQAI.src.view.frame_step2 import FrameStep2
        from QATCH.VisQAI.src.view.main_window import VisQAIWindow


class FrameStep1(QtWidgets.QDialog):
    def __init__(self, parent=None, step=1):
        super().__init__(parent)
        self.parent: VisQAIWindow = parent
        self.step = step

        self.all_files = {}
        self.model_path = None
        self.run_file_run = None
        self.run_file_xml = None
        self.run_file_analyze = None

        self.profile_shears = [1e2, 1e3, 1e4, 1e5, 15000000]
        self.profile_viscos = []

        if step == 1:
            self.setWindowTitle("Select Run")
        elif step == 2:
            self.setWindowTitle("Suggest Experiments")
        elif step == 3:
            self.setWindowTitle("Select Experiments")
        else:
            self.setWindowTitle(f"FrameStep{step}")

        if step == 2:  # Suggest
            self.constraints_ui = ConstraintsUI(self, self.step)

        # Main layout
        main_layout = QtWidgets.QHBoxLayout(self)
        self.h_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.h_splitter)

        # Left panel: Run selection
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        form_layout = QtWidgets.QFormLayout()
        if step == 1:
            left_group = QtWidgets.QGroupBox("Select Run")
        elif step == 2:
            left_group = QtWidgets.QGroupBox("Suggested Runs")
        elif step == 3:
            left_group = QtWidgets.QGroupBox("Experiment Runs")
        elif step == 5:
            left_group = QtWidgets.QGroupBox("Predictions")
        left_group_layout = QtWidgets.QVBoxLayout(left_group)
        left_group_layout.addLayout(form_layout)

        # Select model (for step 5: Predict)
        if True:  # step == 5:
            # Browse model layout
            self.model_dialog = QtWidgets.QFileDialog()
            self.model_dialog.setOption(
                QtWidgets.QFileDialog.DontUseNativeDialog, True)
            model_path = os.path.join(Architecture.get_path(),
                                      "QATCH/VisQAI/assets")
            if os.path.exists(model_path):
                # working or bundled directory, if exists
                self.model_dialog.setDirectory(model_path)
            else:
                # fallback to their local logged data folder
                self.model_dialog.setDirectory(Constants.log_prefer_path)
            self.model_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            self.model_dialog.setNameFilter("VisQ.AI Models (VisQAI-*.zip)")
            self.model_dialog.selectNameFilter("VisQ.AI Models (VisQAI-*.zip)")

            self.select_model_group = QtWidgets.QGroupBox("Select Model")
            select_model_layout = QtWidgets.QHBoxLayout(
                self.select_model_group)
            self.select_model_btn = QtWidgets.QPushButton("Browse...")
            self.select_model_label = QtWidgets.QLineEdit()
            self.select_model_label.setPlaceholderText("No model selected")
            self.select_model_label.setReadOnly(True)
            if step == 1:
                predictor_path = os.path.join(model_path,
                                              "VisQAI-base.zip")
                if os.path.exists(predictor_path):
                    # working or bundled predictor, if exists
                    self.model_selected(path=predictor_path)
            select_model_layout.addWidget(self.select_model_btn)
            select_model_layout.addWidget(self.select_model_label)
            select_model_layout.addStretch()

            left_layout.addWidget(self.select_model_group)

        left_layout.addWidget(left_group)

        self.select_run = QtWidgets.QPushButton(
            "Add Run(s)..." if step == 3 else "Browse...")
        self.select_label = QtWidgets.QLineEdit()
        self.select_label.setPlaceholderText("No run selected")
        self.select_label.setReadOnly(True)
        # run_select = QtWidgets.QHBoxLayout()
        # run_select.addWidget(self.select_run)
        # run_select.addWidget(self.select_label)

        if step == 1:
            form_layout.addRow(self.select_run, self.select_label)
        elif step == 2 or step == 3 or step == 5:
            self.list_view = QtWidgets.QListView()
            self.list_view.setEditTriggers(
                QtWidgets.QAbstractItemView.NoEditTriggers)
            self.model = QtGui.QStandardItemModel()
            string_list = []
            if False:  # step == 2 or step == 5:
                for i in range(4):
                    base_name = "Suggestion" if step == 2 else "Prediction"
                    string_list.append(f"{base_name} {i+1}")
            for string in string_list:
                self.model.appendRow(QtGui.QStandardItem(string))
            self.list_view_addPlaceholderText()
            self.list_view.setModel(self.model)
            if step == 1 or step == 3:
                form_layout.addRow(self.select_run, self.list_view)
            elif step == 2:
                form_layout.addRow("Experiment:", self.list_view)
            elif step == 5:
                form_layout.addRow("Prediction:", self.list_view)
            if step == 3:
                self.list_view.clicked.connect(self.user_run_clicked)
            else:
                # For steps 2 and 5, pull suggestions/predictions from model
                self.list_view.clicked.connect(
                    lambda: self.feature_table.setData(self.loaded_features[self.list_view.selectedIndexes()[0].row()]) if len(self.loaded_features) else None)
                self.list_view.clicked.connect(self.hide_extended_features)
            self.list_view.clicked.connect(
                lambda: self.btn_update.setEnabled(True))

            add_remove_export_widget = QtWidgets.QWidget()
            add_remove_export_layout = QtWidgets.QHBoxLayout(
                add_remove_export_widget)
            add_remove_export_layout.setContentsMargins(0, 0, 0, 0)

            if step in [2, 5]:
                btn_text = "Suggestion" if step == 2 else "Prediction"
                self.btn_add = QtWidgets.QPushButton(f"Add {btn_text}")
                self.btn_add.clicked.connect(self.add_another_item)
                add_remove_export_layout.addWidget(self.btn_add)

            # Remove Selected Run
            self.btn_remove = QtWidgets.QPushButton("Remove Selected")
            self.btn_remove.clicked.connect(self.user_run_removed)
            add_remove_export_layout.addWidget(self.btn_remove)

            # Remove All Runs
            self.btn_remove_all = QtWidgets.QPushButton("Remove All")
            self.btn_remove_all.clicked.connect(
                self.user_all_runs_removed)
            add_remove_export_layout.addWidget(self.btn_remove_all)

            if step in [2, 5]:  # Suggest, Predict
                self.btn_export = QtWidgets.QPushButton("Export as PDF")
                self.btn_export.clicked.connect(self.export_table_data)
                add_remove_export_layout.addWidget(self.btn_export)

            form_layout.addRow("", add_remove_export_widget)
        self.run_notes = QtWidgets.QTextEdit()
        self.run_notes.setPlaceholderText("None")
        self.run_notes.setReadOnly(True)

        # Run information
        self.run_name = QtWidgets.QLabel()
        self.run_date_time = QtWidgets.QLabel()
        self.run_duration = QtWidgets.QLabel()
        self.run_temperature = QtWidgets.QLabel()
        self.run_batch = QtWidgets.QLabel()
        self.run_fill_type = QtWidgets.QLabel()

        # Audits
        self.run_captured = QtWidgets.QLabel()
        self.run_updated = QtWidgets.QLabel()
        self.run_analyzed = QtWidgets.QLabel()

        if step == 2 or step == 5:
            self.run_captured.setText("N/A")
            self.run_updated.setText("N/A")
            self.run_analyzed.setText("N/A")
        else:
            form_layout.addRow("Notes:", self.run_notes)
            form_layout.addRow("<b>Run Information</b>", None)
            form_layout.addRow("Name:", self.run_name)
            form_layout.addRow("Date / Time:", self.run_date_time)
            form_layout.addRow("Duration:", self.run_duration)
            form_layout.addRow("Temperature (avg):", self.run_temperature)
            form_layout.addRow("Batch Number:", self.run_batch)
            form_layout.addRow("Fill Type:", self.run_fill_type)
            form_layout.addRow("Captured:", self.run_captured)
            form_layout.addRow("Updated:", self.run_updated)
            form_layout.addRow("Analyzed:", self.run_analyzed)

        # Action buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        if step == 1:
            self.btn_next = QtWidgets.QPushButton(
                "Next Step: Suggest Experiments")
        elif step == 2:
            self.btn_next = QtWidgets.QPushButton(
                "Next Step: Import Experiments")
        elif step == 3:
            self.btn_next = QtWidgets.QPushButton(
                "Next Step: Learn")
        elif step == 5:
            self.btn_next = QtWidgets.QPushButton(
                "Next Step: Optimize")
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_next)
        left_layout.addLayout(btn_layout)

        # Right panel: Initialize features
        step_verb = "Initialize"
        if step == 2:
            step_verb = "Suggested"
        if step == 5:
            step_verb = "Predicted"
        right_header = QtWidgets.QGroupBox(f"{step_verb} Features")
        right_group = QtWidgets.QVBoxLayout(right_header)
        if step == 5:
            ci_widget = QtWidgets.QWidget()
            ci_layout = QtWidgets.QVBoxLayout(ci_widget)
            ci_layout.setContentsMargins(0, 0, 0, 0)

            # Label and value display
            ci_header_layout = QtWidgets.QHBoxLayout()
            ci_label = QtWidgets.QLabel("Confidence Interval:")
            ci_header_layout.addWidget(ci_label)

            self.ci_value_label = QtWidgets.QLabel("95%")
            self.ci_value_label.setStyleSheet("font-weight: bold;")
            ci_header_layout.addWidget(self.ci_value_label)
            ci_header_layout.addStretch()
            ci_layout.addLayout(ci_header_layout)

            # Slider
            self.ci_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.ci_slider.setMinimum(50)
            self.ci_slider.setMaximum(99)
            self.ci_slider.setValue(95)
            self.ci_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
            self.ci_slider.setTickInterval(10)
            self.ci_slider.valueChanged.connect(self.update_ci_label)
            ci_layout.addWidget(self.ci_slider)

            right_group.addWidget(ci_widget)
            # Features table
        self.load_all_ingredients()
        self.default_features = {"Feature": ["Protein Type", "Protein Concentration",
                                             "Protein Class", "Protein Molecular Weight",  # not in Run Info
                                             "Protein pI Mean", "Protein pI Range",  # not in Run Info
                                             "Buffer Type", "Buffer Concentration",
                                             "Buffer pH",  # not in Run Info
                                             "Surfactant Type", "Surfactant Concentration",
                                             "Stabilizer Type", "Stabilizer Concentration",
                                             "Salt Type", "Salt Concentration",
                                             "Excipient Type", "Excipient Concentration",
                                             "Temperature"],  # only displayed on Predict tab (for now)
                                 "Value": [{"choices": self.proteins, "selected": ""}, "",
                                           # class, molecular weight, pI mean, pI range
                                           {"choices": self.class_types,
                                               "selected": ""}, "", "", "",
                                           {"choices": self.buffers,
                                               "selected": ""}, "",
                                           "",  # buffer pH
                                           {"choices": self.surfactants,
                                               "selected": ""}, "",
                                           {"choices": self.stabilizers,
                                               "selected": ""}, "",
                                           {"choices": self.salts,
                                               "selected": ""}, "",
                                           {"choices": self.excipients,
                                               "selected": ""}, "",
                                           ""],
                                 "Units": ["", "mg/mL",
                                           "", "kDa", "", "",  # pI
                                           "", "mM",
                                           "",  # pH
                                           "", "%w",
                                           "", "M",
                                           "", "mM",
                                           "", "mM",
                                           "\u00b0C"]}  # degrees Celsius
        self.default_rows, self.default_cols = (len(list(self.default_features.values())[0]),
                                                len(list(self.default_features.keys())))

        self.feature_table = TableView(self.default_features,
                                       self.default_rows, self.default_cols)
        self.feature_table.clear()
        right_group.addWidget(self.feature_table)

        # Update proteins by class for table view auto-selection
        self.feature_table.setProteinsByClass(self.proteins_by_class)

        self.btn_update = QtWidgets.QPushButton()
        self.btn_update.setEnabled(False)
        if step == 1:  # Select
            self.btn_update.setText("Save Formulation")
            self.btn_update.clicked.connect(self.save_formulation)
        if step == 2:  # Suggest
            self.btn_update.setText("Decline Suggestion")
            self.btn_update.clicked.connect(self.user_run_removed)
        if step == 3:  # Import
            self.btn_update.setText("Save Formulation")
            self.btn_update.clicked.connect(self.save_formulation)
        # step 4 is not in this class: Learn
        if step == 5:  # Predict
            self.btn_update.setText("Update Prediction")
            self.btn_update.clicked.connect(self.make_predictions)
        right_group.addWidget(self.btn_update)

        self.loaded_features = []

        # # Testing only, create dummy features
        # self.dummy_features = []
        # for i in range(4):
        #     dummy_feature = copy.deepcopy(self.default_features)
        #     value_tags = [0, range(5, 95),
        #                   0, 0, 0,
        #                   range(3), range(5, 95),
        #                   0,
        #                   range(2), range(5, 95),
        #                   range(2), range(5, 95),
        #                   0, range(5, 95)]
        #     for x in range(len(dummy_feature["Value"])):
        #         try:
        #             current_value = dummy_feature["Value"][x]
        #             current_tag = value_tags[x]
        #             if isinstance(current_value, list):
        #                 if isinstance(current_tag, int):
        #                     dummy_feature["Value"][x] = \
        #                         current_value[current_tag]
        #                 else:
        #                     dummy_feature["Value"][x] = current_value[randint(
        #                         current_tag[0], current_tag[-1])]
        #             else:
        #                 if isinstance(current_tag, range):
        #                     dummy_feature["Value"][x] = randint(
        #                         current_tag[0], current_tag[-1])
        #         except Exception as e:
        #             print(e)
        #     # Hide protein and buffer characteristics
        #     # for values in dummy_feature.values():
        #     #     del values[8]  # buffer PH
        #     #     del values[5]  # protein pI range
        #     #     del values[4]  # protein pI mean
        #     #     del values[3]  # protein weight
        #     #     del values[2]  # protein class
        #     self.dummy_features.append(dummy_feature)

        self.run_figure = Figure()
        self.run_figure_valid = False
        self.run_canvas = FigureCanvas(self.run_figure)

        # Build main layout
        self.h_splitter.addWidget(left_widget)
        self.h_splitter.addWidget(right_header)
        self.h_splitter.addWidget(self.run_canvas)

        # Set fixed width for left widget
        left_widget.setMinimumWidth(450)
        right_header.setMinimumWidth(350)

        # add collapse/expand icon arrows
        self.h_splitter.setHandleWidth(10)
        handle = self.h_splitter.handle(2)
        layout_s = QtWidgets.QVBoxLayout()
        layout_s.setContentsMargins(0, 0, 0, 0)
        layout_s.addStretch()
        self.btnCollapse = QtWidgets.QToolButton(handle)
        self.btnCollapse.setArrowType(QtCore.Qt.LeftArrow)
        self.btnCollapse.clicked.connect(
            lambda: self.handleSplitterButton(True))
        layout_s.addWidget(self.btnCollapse)
        self.btnExpand = QtWidgets.QToolButton(handle)
        self.btnExpand.setArrowType(QtCore.Qt.RightArrow)
        self.btnExpand.clicked.connect(
            lambda: self.handleSplitterButton(False))
        layout_s.addWidget(self.btnExpand)
        layout_s.addStretch()
        handle.setLayout(layout_s)
        self.btnExpand.setVisible(False)
        self.handleSplitterButton(False)
        self.h_splitter.splitterMoved.connect(self.handleSplitterMoved)

        # Signals
        self.btn_cancel.clicked.connect(
            lambda: self.file_selected(None, cancel=True))
        self.btn_next.clicked.connect(
            getattr(self, f"proceed_to_step_{self.step+1}"))
        self.select_run.clicked.connect(self.user_run_browse)
        if True:  # step == 5:
            self.select_model_btn.clicked.connect(self.model_dialog.show)
            global_handler = getattr(
                self.parent, 'set_global_model_path', None)
            self.model_dialog.fileSelected.connect(
                global_handler if callable(global_handler) else self.model_selected)

    def reload_all_ingredient_choices(self):

        # Reload all ingredients from DB
        self.load_all_ingredients()

        # Update choices lists in default features for dropdown items
        self.default_features["Value"][0]["choices"] = self.proteins
        self.default_features["Value"][2]["choices"] = self.class_types
        self.default_features["Value"][6]["choices"] = self.buffers
        self.default_features["Value"][9]["choices"] = self.surfactants
        self.default_features["Value"][11]["choices"] = self.stabilizers
        self.default_features["Value"][13]["choices"] = self.salts
        self.default_features["Value"][15]["choices"] = self.excipients

        # Update proteins by class for table view auto-selection
        self.feature_table.setProteinsByClass(self.proteins_by_class)

    def on_tab_selected(self):

        # Reload ingredients from DB and update default choices
        self.reload_all_ingredient_choices()

        if self.step == 2:  # Suggest
            # self.load_suggestion()
            pass
        if self.step == 5:  # Predict
            if len(self.loaded_features) == 0:
                self.model.removeRow(0)  # no_item placeholder
                self.add_formulation(self.parent.select_formulation)
        if True:  # self.step == 5:  # Predict
            # Select a pre-selected model, if none selected here
            if not self.model_path:
                select_tab: FrameStep1 = self.parent.tab_widget.widget(0)
                suggest_tab: FrameStep1 = self.parent.tab_widget.widget(1)
                import_tab: FrameStep1 = self.parent.tab_widget.widget(2)
                learn_tab: FrameStep2 = self.parent.tab_widget.widget(3)
                predict_tab: FrameStep1 = self.parent.tab_widget.widget(5)
                optimize_tab: FrameStep2 = self.parent.tab_widget.widget(6)
                all_model_paths = [select_tab.model_path,
                                   suggest_tab.model_path,
                                   import_tab.model_path,
                                   learn_tab.model_path,
                                   predict_tab.model_path,
                                   optimize_tab.model_path]
                found_model_path = next(
                    (x for x in all_model_paths if x is not None), None)
                if found_model_path:
                    self.model_selected(found_model_path)

    def handleSplitterMoved(self, pos=0, index=0):
        collapsed = self.h_splitter.sizes()[1] == 0
        self.btnCollapse.setVisible(not collapsed)
        self.btnExpand.setVisible(collapsed)

    def handleSplitterButton(self, collapse=True):
        if collapse:
            self.h_splitter.setSizes([10, 0, 10000])
        else:
            if self.step == 2:  # Suggest
                self.h_splitter.setSizes([10, 10000, 0])
            else:
                self.h_splitter.setSizes([10, 10, 10000])
        self.handleSplitterMoved()

    def list_view_addPlaceholderText(self):
        if self.model.rowCount() == 0:
            no_item_text = "No items in list"
            if self.step == 2:
                no_item_text = "No suggestions available"
            if self.step == 3:
                no_item_text = "No experiments selected"
            if self.step == 5:
                no_item_text = "No predictions available"
            no_item = QtGui.QStandardItem(no_item_text)
            no_item.setEnabled(False)
            no_item.setSelectable(False)
            self.model.appendRow(no_item)

    def load_all_ingredients(self):
        self.proteins: list[str] = []
        self.buffers: list[str] = []
        self.surfactants: list[str] = []
        self.stabilizers: list[str] = []
        self.salts: list[str] = []
        self.excipients: list[str] = []
        self.class_types: list[str] = []
        self.proteins_by_class: dict[str, list[str]] = {}

        self.proteins, self.buffers, self.surfactants, \
            self.stabilizers, self.salts, self.excipients, \
            self.class_types, self.proteins_by_class = ListUtils.load_all_ingredient_types(
                self.parent.ing_ctrl)

        Log.d("Proteins:", self.proteins)
        Log.d("Buffers:", self.buffers)
        Log.d("Surfactants:", self.surfactants)
        Log.d("Stabilizers:", self.stabilizers)
        Log.d("Salts:", self.salts)
        Log.d("Excipients:", self.excipients)
        Log.d("Class Types:", self.class_types)
        Log.d("Proteins By Class:", self.proteins_by_class)

    def hide_extended_features(self):
        hide_rows = []
        if self.step in [2, 5]:
            hide_rows.extend([2, 3, 4, 5, 8])
        if self.step != 5:
            # Hide Temperature everywhere other than Predict
            hide_rows.append(17)
        for row in hide_rows:
            self.feature_table.hideRow(row)

    def update_ci_label(self, value):
        self.ci_value_label.setText(f"{value}%")

    def get_ci_range(self):
        ci_percent = self.ci_slider.value()
        lower = (100.0 - ci_percent) / 2
        upper = 100.0 - lower
        return (lower, upper)

    def save_formulation(self, cancel: bool = False) -> bool:
        if not self.feature_table.allSet():
            Log.e("Not all features have been set. " +
                  "Cannot save formulation info. " +
                  "Enter missing values and try again.")
            return False

        protein_type = self.feature_table.cellWidget(0, 1).currentText()
        protein_conc = self.feature_table.item(1, 1).text()
        protein_class = self.feature_table.cellWidget(2, 1).currentText()
        protein_weight = self.feature_table.item(3, 1).text()
        protein_pI_mean = self.feature_table.item(4, 1).text()
        protein_pI_range = self.feature_table.item(5, 1).text()
        buffer_type = self.feature_table.cellWidget(6, 1).currentText()
        buffer_conc = self.feature_table.item(7, 1).text()
        buffer_pH = self.feature_table.item(8, 1).text()
        surfactant_type = self.feature_table.cellWidget(9, 1).currentText()
        surfactant_conc = self.feature_table.item(10, 1).text()
        stabilizer_type = self.feature_table.cellWidget(
            11, 1).currentText()
        stabilizer_conc = self.feature_table.item(12, 1).text()
        salt_type = self.feature_table.cellWidget(13, 1).currentText()
        salt_conc = self.feature_table.item(14, 1).text()
        excipient_type = self.feature_table.cellWidget(15, 1).currentText()
        excipient_conc = self.feature_table.item(16, 1).text()
        temp = self.feature_table.item(17, 1).text()
        # save run info to XML (if changed, request audit sign)
        if self.step in [1, 3]:  # Select, Import
            self.parent.save_run_info(self.run_file_xml, [
                protein_type, protein_conc,
                buffer_type, buffer_conc,
                surfactant_type, surfactant_conc,
                stabilizer_type, stabilizer_conc,
                salt_type, salt_conc,
                excipient_type, excipient_conc], cancel)
            if self.parent.hasUnsavedChanges():
                if cancel:
                    Log.w("Unsaved changes lost, per user discretion.")
                    return True
                Log.w("There are still unsaved changes. Cannot continue.")
                QtWidgets.QMessageBox.information(
                    None,
                    Constants.app_title,
                    "There are still unsaved changes!\n\n" +
                    "To save: Try again and sign when prompted.\n" +
                    "Click \"Cancel\" to discard these changes.")
                return False
            elif cancel:
                Log.d("User canceled with nothing to save.")
                return True

        if self.step == 5:  # Predict
            # save table to loaded_features
            # NOTE: combobox items are `dict[choices: list[str], selected: str]`
            feature = copy.deepcopy(self.default_features)
            feature["Value"][0]["selected"] = protein_type
            feature["Value"][1] = protein_conc
            feature["Value"][2]["selected"] = protein_class
            feature["Value"][3] = protein_weight
            feature["Value"][4] = protein_pI_mean
            feature["Value"][5] = protein_pI_range
            feature["Value"][6]["selected"] = buffer_type
            feature["Value"][7] = buffer_conc
            feature["Value"][8] = buffer_pH
            feature["Value"][9]["selected"] = surfactant_type
            feature["Value"][10] = surfactant_conc
            feature["Value"][11]["selected"] = stabilizer_type
            feature["Value"][12] = stabilizer_conc
            feature["Value"][13]["selected"] = salt_type
            feature["Value"][14] = salt_conc
            feature["Value"][15]["selected"] = excipient_type
            feature["Value"][16] = excipient_conc
            feature["Value"][17] = temp

            self.loaded_features[self.list_view.selectedIndexes()[
                0].row()] = feature

        protein = self.parent.ing_ctrl.get_protein_by_name(
            name=protein_type)
        if protein is None:
            protein = self.parent.ing_ctrl.add_protein(
                Protein(enc_id=-1, name=protein_type))

        buffer = self.parent.ing_ctrl.get_buffer_by_name(
            name=buffer_type)
        if buffer is None:
            buffer = self.parent.ing_ctrl.add_buffer(
                Buffer(enc_id=-1, name=buffer_type))

        surfactant = self.parent.ing_ctrl.get_surfactant_by_name(
            name=surfactant_type)
        if surfactant is None:
            surfactant = self.parent.ing_ctrl.add_surfactant(
                Surfactant(enc_id=-1, name=surfactant_type))

        stabilizer = self.parent.ing_ctrl.get_stabilizer_by_name(
            name=stabilizer_type)
        if stabilizer is None:
            stabilizer = self.parent.ing_ctrl.add_stabilizer(
                Stabilizer(enc_id=-1, name=stabilizer_type))

        salt = self.parent.ing_ctrl.get_salt_by_name(
            name=salt_type)
        if salt is None:
            salt = self.parent.ing_ctrl.add_salt(
                Salt(enc_id=-1, name=salt_type))

        excipient = self.parent.ing_ctrl.get_excipient_by_name(
            name=excipient_type)
        if excipient is None:
            excipient = self.parent.ing_ctrl.add_excipient(
                Excipient(enc_id=-1, name=excipient_type))

        def is_number(s: str):
            try:
                float(s)
                return True
            except ValueError:
                return False

        # update protein and buffer characteristics
        # bail if any extended features are missing
        if protein_class in self.class_types:
            protein.class_type = ProteinClass.from_value(protein_class)
        elif not protein.class_type:
            Log.e("Missing protein class!")
            return False
        if is_number(protein_weight):
            protein.molecular_weight = float(protein_weight)
        elif not protein.molecular_weight:
            Log.e("Missing protein molecular weight!")
            return False
        if is_number(protein_pI_mean):
            protein.pI_mean = float(protein_pI_mean)
        elif not protein.pI_mean:
            Log.e("Missing protein pI mean!")
            return False
        if is_number(protein_pI_range):
            protein.pI_range = float(protein_pI_range)
        elif not protein.pI_range:
            Log.e("Missing protein pI range!")
            return False
        if is_number(buffer_pH):
            buffer.pH = float(buffer_pH)
        elif not buffer.pH:
            Log.e("Missing buffer pH!")
            return False

        # if no changes, nothing is done on 'update' call
        self.parent.ing_ctrl.update_protein(protein.id, protein)
        self.parent.ing_ctrl.update_buffer(buffer.id, buffer)

        while len(self.profile_viscos) < len(self.profile_shears):
            self.profile_viscos.append(-1)

        # pull in viscosity profile from run load
        vp = ViscosityProfile(shear_rates=self.profile_shears,
                              viscosities=self.profile_viscos,
                              units='cP')
        vp.is_measured = self.run_figure_valid

        # pull temperaure (already pulled from feature table)
        # temp = self.run_temperature.text()
        # if temp.endswith('C'):
        #     temp = temp[:-1]  # strip Celsius unit character
        if not is_number(temp):
            temp = "nan"  # not a number, casts to float as nan
        elif self.step == 5:
            # Predict tab can specify custom Temperature target
            if float(temp) < 0 or float(temp) > 100:
                Log.e(
                    f"Temperature input {temp}\u00b0C is out-of-range! (Allowed: 0 - 100)")
                return False

        form = Formulation()
        form.set_protein(
            protein=protein, concentration=float(protein_conc), units='mg/mL')
        form.set_buffer(buffer, concentration=float(
            buffer_conc), units='mM')
        form.set_surfactant(surfactant=surfactant,
                            concentration=float(surfactant_conc), units='%w')
        form.set_stabilizer(stabilizer=stabilizer,
                            concentration=float(stabilizer_conc), units='M')
        form.set_salt(salt, concentration=float(salt_conc), units='mM')
        form.set_excipient(excipient=excipient, concentration=float(
            excipient_conc), units="mM")
        form.set_viscosity_profile(profile=vp)
        form.set_temperature(float(temp))

        form_saved = self.parent.form_ctrl.add_formulation(
            formulation=form)

        if self.step == 1:
            Log.d("Saving selected formulation to parent for later")
            self.parent.select_formulation = form_saved
            # print(self.parent.form_ctrl.get_all_as_dataframe())
        if self.step == 3 and form_saved not in self.parent.import_formulations:
            num_forms = len(self.parent.import_formulations)
            Log.d(
                f"Saving imported formulation #{num_forms+1} to parent for later")
            self.parent.import_formulations.append(form_saved)
            # Store the same label used in the left list (select_label),
            # so FrameStep2 can resolve indices reliably.
            self.parent.import_run_names.append(self.select_label.text())
        if self.step == 5:
            Log.d("Saving prediction formulation to parent for later")
            self.parent.predict_formulation = form_saved

        # Collapse feature table on save (all set)
        self.handleSplitterButton(collapse=True)

        return True

    def load_suggestion(self, constraints):
        model_name = self.select_model_label.text()
        if hasattr(self, "timer") and self.timer.isActive():
            Log.w("Busy canceling... Please wait...")
            return
        if len(self.select_model_label.text()) == 0 or self.model_path is None:
            Log.e("No model selected. Cannot load suggestions.")
            return
        if not self.parent.database.is_open:
            Log.e("No database connection. Cannot load suggestions.")
            return

        self.progressBar = QtWidgets.QProgressDialog(
            "Suggesting...", "Cancel", 0, 0, self)
        # Disable auto-reset and auto-close to retain `wasCanceled()` state
        self.progressBar.setAutoReset(False)
        self.progressBar.setAutoClose(False)
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/reset.png')
        self.progressBar.setWindowIcon(QtGui.QIcon(icon_path))
        self.progressBar.setWindowTitle("Busy")
        self.progressBar.setWindowFlag(
            QtCore.Qt.WindowContextHelpButtonHint, False)
        self.progressBar.setWindowFlag(
            QtCore.Qt.WindowStaysOnTopHint, True)
        self.progressBar.setFixedSize(
            int(self.progressBar.width()*1.5), int(self.progressBar.height()*1.1))
        self.progressBar.setModal(True)
        self.progressBar.show()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.check_finished)
        self.timer.start()

        def add_new_suggestion(record: Optional[ExecutionRecord] = None):
            if self.progressBar.wasCanceled():
                Log.d("User canceled suggestion. Ignoring results.")
                return

            Log.d("Processing suggestion results!")

            if not record:
                Log.e("ERROR: No `record` provided to `add_new_suggestion(record)`")
                return

            exception = record.exception
            traceback = record.traceback
            if exception:
                Log.e(
                    f"ERROR: Failed to suggest: {str(exception)}, {str(traceback)}")
                return

            form = record.result
            self.add_formulation(form)

        Log.d("Waiting for suggestion results...")
        self.parent.enable(False)
        self.executor = Executor()
        self.executor.run(
            self,
            method_name="get_new_suggestion",
            asset_name=model_name,
            constraints=copy.deepcopy(constraints),
            callback=add_new_suggestion)

    def get_new_suggestion(self, asset_name, constraints: Constraints | None = None):
        database = Database(parse_file_key=True)
        constraints.set_db(database)  # Needed for cross-threading
        sampler = Sampler(asset_name=asset_name,
                          database=database,
                          constraints=constraints)
        form = sampler.get_next_sample()
        database.close()
        return form

    def add_formulation(self, form: Formulation):
        feature = copy.deepcopy(self.default_features)
        if form.protein:  # NOT an empty Formulation() object
            # NOTE: combobox items are `dict[choices: list[str], selected: str]`
            feature["Value"][0]["selected"] = form.protein.ingredient.name
            feature["Value"][1] = form.protein.concentration
            feature["Value"][2]["selected"] = form.protein.ingredient.class_type.value \
                if form.protein.ingredient.class_type else "None"  # class_type could be None
            feature["Value"][3] = form.protein.ingredient.molecular_weight
            feature["Value"][4] = form.protein.ingredient.pI_mean
            feature["Value"][5] = form.protein.ingredient.pI_range
            feature["Value"][6]["selected"] = form.buffer.ingredient.name
            feature["Value"][7] = form.buffer.concentration
            feature["Value"][8] = form.buffer.ingredient.pH
            feature["Value"][9]["selected"] = form.surfactant.ingredient.name
            feature["Value"][10] = form.surfactant.concentration
            feature["Value"][11]["selected"] = form.stabilizer.ingredient.name
            feature["Value"][12] = form.stabilizer.concentration
            feature["Value"][13]["selected"] = form.salt.ingredient.name
            feature["Value"][14] = form.salt.concentration
            feature["Value"][15]["selected"] = form.excipient.ingredient.name
            feature["Value"][16] = form.excipient.concentration
            feature["Value"][17] = form.temperature

        if len(self.loaded_features) == 0:
            self.model.removeRow(0)  # no_item placeholder
        self.loaded_features.append(feature)
        num = len(self.loaded_features)
        form_type = "Suggestion" if self.step == 2 else "Prediction"
        self.model.appendRow(QtGui.QStandardItem(f"{form_type} {num}"))

    def _get_viscosity_list(self, formulation: Formulation) -> list:
        rate_list = []
        vp = formulation.viscosity_profile
        for rate in [100, 1000, 10000, 100000, 15000000]:
            rate_list.append(vp.get_viscosity(rate))
        return rate_list

    def make_predictions(self):
        if hasattr(self, "timer") and self.timer.isActive():
            Log.w("Busy canceling... Please wait...")
            return
        if len(self.select_model_label.text()) == 0 or self.model_path is None:
            Log.e("No model selected. Cannot make predictions.")
            return
        if not self.parent.database.is_open:
            Log.e("No database connection. Cannot make predictions.")
            return

        if not self.feature_table.allSet():
            message = "Please correct the highlighted fields first."
            QtWidgets.QMessageBox.information(
                None, Constants.app_title, message, QtWidgets.QMessageBox.Ok)
            return

        if not self.save_formulation():
            Log.w("Aborting prediction. Failed to save formulation.")
            return

        self.predictor = Predictor(zip_path=self.model_path)
        predict_df = self.parent.predict_formulation.to_dataframe(
            encoded=False, training=False)

        self.progressBar = QtWidgets.QProgressDialog(
            "Updating...", "Cancel", 0, 0, self)
        # Disable auto-reset and auto-close to retain `wasCanceled()` state
        self.progressBar.setAutoReset(False)
        self.progressBar.setAutoClose(False)
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/reset.png')
        self.progressBar.setWindowIcon(QtGui.QIcon(icon_path))
        self.progressBar.setWindowTitle("Busy")
        self.progressBar.setWindowFlag(
            QtCore.Qt.WindowContextHelpButtonHint, False)
        self.progressBar.setWindowFlag(
            QtCore.Qt.WindowStaysOnTopHint, True)
        self.progressBar.setFixedSize(
            int(self.progressBar.width()*1.5), int(self.progressBar.height()*1.1))
        self.progressBar.setModal(True)
        self.progressBar.show()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.check_finished)
        self.timer.start()

        def run_prediction_result(record: Optional[ExecutionRecord] = None):

            if record and record.exception:
                # NOTE: Progress bar and timer will end on next call to `check_finished()`
                Log.e(
                    f"Error occurred while updating the model: {record.exception}")
                return

            Log.d("Waiting for prediction results...")
            self.progressBar.setLabelText("Predicting...")

            self.executor.run(
                self.predictor,
                method_name="predict_uncertainty",
                df=predict_df,
                ci_range=self.get_ci_range(),
                callback=get_prediction_result)

        def get_prediction_result(record: Optional[ExecutionRecord] = None):

            if self.progressBar.wasCanceled():
                Log.d("User canceled prediction. Ignoring results.")
                return

            Log.d("Processing prediction results!")

            # The returns from this are a predicted viscosity profile [val1,val2,...val5]
            # predicted_vp = self.parent.predictor.predict(df=form_df)

            # The returns from this are a predicted viscosity profile [val1,val2,...val5] and
            # a series of standard deviations for each predicted value.

            if not record:
                Log.e("ERROR: No `record` provided to `get_prediction_result(record)`")
                return

            exception = record.exception
            if exception:
                Log.e(f"ERROR: Prediction exception: {str(exception)}")
                return

            predicted_mean_vp, uncertainty_dict = record.result

            # Helper functions for plotting
            def smooth_log_interpolate(x, y, num=200, expand_factor=0.05):
                xlog = np.log10(x)
                ylog = np.log10(y)
                f_interp = interp1d(xlog, ylog, kind='linear',
                                    fill_value='extrapolate')
                xlog_min, xlog_max = xlog.min(), xlog.max()
                margin = (xlog_max - xlog_min) * expand_factor
                xs_log = np.linspace(xlog_min - margin, xlog_max + margin, num)
                xs = 10**xs_log
                ys = 10**f_interp(xs_log)
                return xs, ys

            def make_plot(name, shear, mean_arr, uncertainty_dict, title, color):
                self.run_figure_valid = False
                self.run_figure.clear()
                self.run_canvas.draw()
                shear = np.asarray(shear)
                mean_arr = np.asarray(mean_arr)
                lower_ci = uncertainty_dict['lower_ci']
                upper_ci = uncertainty_dict['upper_ci']
                if lower_ci.ndim > 1:
                    lower_ci = lower_ci.flatten()
                if upper_ci.ndim > 1:
                    upper_ci = upper_ci.flatten()
                ax = self.run_figure.add_subplot(111)
                ax.set_facecolor('#ffffff')
                self.run_figure.patch.set_facecolor('#ffffff')
                xs, ys = smooth_log_interpolate(shear, mean_arr)
                xs_up, ys_up = smooth_log_interpolate(shear, upper_ci)
                xs_dn, ys_dn = smooth_log_interpolate(shear, lower_ci)

                # Updated color scheme - teal/cyan palette
                main_color = '#00A3DA'  # Deep cyan
                ci_color = '#69EAC5'    # Light teal

                ax.fill_between(xs_up, ys_dn, ys_up, alpha=0.15, color=ci_color,
                                linewidth=0, label=f'{self.ci_value_label.text()} CI')
                ax.plot(xs, ys, '-', lw=2.5, color=main_color,
                        label='Estimated Viscosity', zorder=3, alpha=0.95)
                ax.plot(xs_up, ys_up, '-', lw=1,
                        color=ci_color, alpha=0.7, zorder=2)
                ax.plot(xs_dn, ys_dn, '-', lw=1,
                        color=ci_color, alpha=0.7, zorder=2)
                ax.scatter(shear, mean_arr, s=80, color=main_color, zorder=5,
                           edgecolors='white', linewidths=2.5, alpha=1)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(xs.min() * 0.8, xs.max() * 1.2)
                ax.set_ylim(self.calc_limits(
                    yall=np.concatenate((ys_dn, ys_up))))
                ax.set_xlabel("Shear Rate (s⁻¹)", fontsize=11, fontweight='600',
                              color='#2d3436')
                ax.set_ylabel("Viscosity (cP)", fontsize=11, fontweight='600',
                              color='#2d3436')
                ax.set_title(title, fontsize=13, fontweight='600', pad=15,
                             color='#2d3436')
                ax.grid(True, which="major", ls='-',
                        alpha=0.15, color='#636e72', lw=0.8)
                ax.grid(True, which="minor", ls='-',
                        alpha=0.07, color='#b2bec3', lw=0.5)
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
                ax.tick_params(axis='both', which='major', labelsize=9,
                               colors='#2d3436', width=1)
                ax.tick_params(axis='both', which='minor', labelsize=8,
                               colors='#636e72', width=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#dfe6e9')
                ax.spines['bottom'].set_color('#dfe6e9')
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                legend = ax.legend(loc='best', frameon=True, fancybox=False,
                                   shadow=False, fontsize=9, framealpha=1,
                                   edgecolor='#dfe6e9', borderpad=1)
                legend.get_frame().set_facecolor('#ffffff')
                legend.get_frame().set_linewidth(1)
                ylim = ax.get_ylim()
                y_range = np.log10(ylim[1]) - np.log10(ylim[0])
                for i in range(len(mean_arr)):
                    annotation = f'{mean_arr[i]:.1f}\n[{lower_ci[i]:.1f}-{upper_ci[i]:.1f}]'
                    y_offset = mean_arr[i] * (10 ** (y_range * 0.06))
                    ax.annotate(annotation,
                                xy=(shear[i], mean_arr[i]),
                                xytext=(shear[i], y_offset),
                                ha='center', va='bottom',
                                fontsize=8,
                                color='#2d3436',
                                weight='500',
                                bbox=dict(boxstyle='round,pad=0.4',
                                          facecolor='white',
                                          edgecolor=main_color,
                                          alpha=0.95,
                                          linewidth=1.2),
                                arrowprops=dict(arrowstyle='-',
                                                connectionstyle='arc3,rad=0',
                                                color=main_color,
                                                alpha=0.4,
                                                lw=1.2))
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                ax.text(0.98, 0.02, timestamp,
                        transform=ax.transAxes, fontsize=7,
                        verticalalignment='bottom', horizontalalignment='right',
                        alpha=0.4, style='italic', color='#636e72')
                self.run_figure.tight_layout()

                self.run_figure_valid = True
                self.run_canvas.draw()

            Log.d(f"VP={predicted_mean_vp}, {uncertainty_dict}")
            make_plot("Viscosity Profile", self.profile_shears,
                      predicted_mean_vp[0], uncertainty_dict,
                      "Estimated Viscosity Profile", "blue")
            self.predictor.cleanup()

        self.executor = Executor()

        run_prediction_result()

    def check_finished(self):
        # at least 1 record expected, but may be more based on task count
        expect_record_count = max(1, self.executor.task_count())
        if self.executor.active_count() == 0 and len(self.executor.get_task_records()) == expect_record_count:
            # finished, but keep the dialog open to retain `wasCanceled()` state
            self.progressBar.hide()
            self.timer.stop()
            if self.step == 2:
                self.parent.enable(True)

    def proceed_to_step_2(self):
        # Are we ready to proceed?
        # Yes, if and only if:
        #   1. All audits contain valid values
        #   2. All initial features are set
        #   3. Analyze results are valid
        #   4. All formulations saved, and XMLs up-to-date
        if (len(self.run_captured.text()) and
            len(self.run_updated.text()) and
            len(self.run_analyzed.text()) and
                self.feature_table.allSet() and
                self.run_figure_valid):
            # ready to proceed
            if not self.save_formulation():
                return
            if self.parent is not None:
                i = self.parent.tab_widget.currentIndex()
                self.parent.tab_widget.setCurrentIndex(i+1)
                # next_widget: FrameStep1 = self.parent.tab_widget.currentWidget()
                # next_widget.load_suggestions()
            else:
                self.run_notes.setText(
                    "ERROR: self.parent is None.\n" +
                    "Cannot proceed to next step!")
        else:  # not ready
            message = "Please select a run."
            if self.select_label.text():
                message = "Please correct the highlighted fields first."
            QtWidgets.QMessageBox.information(
                None, Constants.app_title, message, QtWidgets.QMessageBox.Ok)

    def proceed_to_step_3(self):
        # ready to proceed
        if self.parent is not None:
            i = self.parent.tab_widget.currentIndex()
            self.parent.tab_widget.setCurrentIndex(i+1)

    def proceed_to_step_4(self):
        # First of all, there must be at least 1 imported experiment (no longer a requirement)
        # For each run in list, must pass the same criteria from Step 1
        #   1. All audits contain valid values
        #   2. All initial features are set
        #   3. Analyze results are valid
        #   4. All formulations saved, and XMLs up-to-date

        # NOTE: No longer a requirement, but can be added back if needed
        # if len(self.all_files) == 0:
        #     QtWidgets.QMessageBox.information(
        #         None, Constants.app_title,
        #         "Please import at least 1 experiment before proceeding.",
        #         QtWidgets.QMessageBox.Ok)
        #     return

        # Show progress bar while loading imported experiments for learning
        self.progressBarDiag = QtWidgets.QProgressDialog(
            "Preparing...", "Cancel", 0, len(self.all_files), self)
        # Disable auto-reset and auto-close to retain `wasCanceled()` state
        self.progressBarDiag.setAutoReset(False)
        self.progressBarDiag.setAutoClose(False)
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/reset.png')
        self.progressBarDiag.setWindowIcon(QtGui.QIcon(icon_path))
        self.progressBarDiag.setWindowTitle("Busy")
        self.progressBarDiag.setWindowFlag(
            QtCore.Qt.WindowContextHelpButtonHint, False)
        self.progressBarDiag.setWindowFlag(
            QtCore.Qt.WindowStaysOnTopHint, True)
        self.progressBarDiag.setFixedSize(
            int(self.progressBarDiag.width()*1.5), int(self.progressBarDiag.height()*1.1))
        self.progressBarDiag.setModal(True)
        if len(self.all_files):
            self.progressBarDiag.show()
        else:
            self.progressBarDiag.reset()

        all_is_good = True
        self.parent.import_formulations.clear()
        self.parent.import_run_names.clear()

        try:
            for i, (_file_name, file_path) in enumerate(self.all_files.items()):

                # draw progress bar immediately (it will freeze)
                # and handle any other pending events (i.e. cancel click)
                self.progressBarDiag.setValue(i)
                for _ in range(2):
                    QtCore.QCoreApplication.processEvents()

                # Check for cancel and stop if true
                if self.progressBarDiag.wasCanceled():
                    Log.w("User canceled import preparing!")
                    self.progressBarDiag.close()  # close it
                    return

                self.file_selected(file_path, loading=True)  # load each run
                QtCore.QCoreApplication.processEvents()  # redraw plot now
                if (len(self.run_captured.text()) and
                    len(self.run_updated.text()) and
                    len(self.run_analyzed.text()) and
                        self.feature_table.allSet() and
                        self.run_figure_valid):
                    if not self.save_formulation():
                        Log.w("Unable to save formulation while preparing!")
                        self.progressBarDiag.close()  # close it
                        return
                else:
                    all_is_good = False
                    break  # highlight *first* run with errors on "Next"
        finally:
            # Close the active progress bar dialog window
            self.progressBarDiag.close()

        if all_is_good:
            # ready to proceed
            if self.parent is not None:
                i = self.parent.tab_widget.currentIndex()
                self.parent.tab_widget.setCurrentIndex(i+1)
                # next_widget: FrameStep2 = self.parent.tab_widget.currentWidget()
                # next_widget.learn()
            else:
                self.run_notes.setText(
                    "ERROR: self.parent is None.\n" +
                    "Cannot proceed to next step!")
        else:  # not ready
            QtWidgets.QMessageBox.information(
                None, Constants.app_title, "Please correct the highlighted fields first.", QtWidgets.QMessageBox.Ok)

    # NOTE: step_5 would be handled in FrameStep2

    def proceed_to_step_6(self):
        # ready to proceed
        if self.parent is not None:
            i = self.parent.tab_widget.currentIndex()
            self.parent.tab_widget.setCurrentIndex(i+1)
        else:
            self.run_notes.setText(
                "ERROR: self.parent is None.\n" +
                "Cannot proceed to next step!")

    def user_run_clicked(self):
        try:
            self.file_selected(self.all_files[self.model.itemFromIndex(
                self.list_view.selectedIndexes()[0]).text()])
        except IndexError as e:
            if len(self.all_files):
                raise e
            else:  # no files in list, this error can occur when user cliks on the placeholder text
                pass  # ignore the click
        # raise any other exception type

    def add_another_item(self):
        if self.step == 2:
            self.constraints_ui.add_suggestion_dialog()
        if self.step == 5:
            self.add_formulation(Formulation())

    def user_all_runs_removed(self) -> None:
        """Prompt the user to confirm and remove all runs.

        Displays a confirmation dialog asking the user if they want to
        remove all runs. If confirmed, clears all internal data
        structures (`all_files` and `loaded_features`), removes all
        items from the list view model, resets the placeholder text,
        and clears any active file selections, suppressing prompts.

        Any errors encountered will be logged to `Log.e`.

        """
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Remove All Runs",
            "Are you sure you want to remove all runs?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            try:
                self.all_files.clear()
                if self.step in [2, 5]:
                    self.loaded_features.clear()
                self.model.clear()
                self.list_view_addPlaceholderText()
                # Discard any pending changes and suppress save prompts when bulk-clearing.
                self.file_selected(None, cancel=True)
            except Exception as e:
                Log.e("Failed to remove all runs.", str(e))

    def user_run_removed(self):
        try:
            selected = self.list_view.selectedIndexes()
            if len(selected) == 0:
                return  # nothing selected, nothing to do
            file_name = self.model.itemFromIndex(selected[0]).text()
            self.all_files.pop(file_name, None)  # remove key from dict
            if self.step in [2, 5]:
                self.loaded_features.pop(selected[0].row())
            self.model.removeRow(selected[0].row())
            self.list_view_addPlaceholderText()
            self.file_selected(None)  # clear selection
        except IndexError as e:
            if len(self.all_files):
                raise e
            else:  # no files in list, this error can occur when user cliks on the placeholder text
                pass  # ignore the click
        # raise any other exception type

    def export_table_data(self):
        info_on_success = False
        open_on_success = True

        default_export_folder = os.path.expanduser(os.path.join(
            "~", "Documents", f"{Constants.app_publisher} {Constants.app_name}", "exported_pdfs"))
        if os.path.exists(os.path.dirname(default_export_folder)):
            os.makedirs(default_export_folder, exist_ok=True)
        else:
            default_export_folder = os.getcwd()

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, "Export as PDF", default_export_folder, "PDF files (*.pdf)"
        )

        if file_path:
            try:
                # Create printer to make PDFs
                printer = QPrinter(QPrinter.PrinterResolution)
                printer.setOutputFormat(QPrinter.PdfFormat)
                printer.setOutputFileName(file_path)
                printer.setPageSize(QPrinter.Letter)

                # Set 1-inch margins on all sides
                # left, top, right, bottom
                margins = QtCore.QMarginsF(1.0, 1.0, 1.0, 1.0)
                printer.setPageMargins(
                    margins.left(), margins.top(), margins.right(), margins.bottom(),
                    QPrinter.Inch)
                painter = QtGui.QPainter(printer)

                # Set font
                font = QtGui.QFont("Times New Roman", 12)
                # NOTE: Bold will be set/unset for header/data rows

                for i in range(len(self.loaded_features)):

                    # Select the item
                    index = self.model.index(i, 0)
                    self.list_view.setCurrentIndex(index)
                    self.list_view.selectionModel().select(
                        index, QtCore.QItemSelectionModel.ClearAndSelect)
                    self.list_view.clicked.emit(index)

                    if i > 0:
                        # Add new page to PDF for each table in list
                        printer.newPage()

                    # Start at top of page
                    y = 0

                    # Page spacing parameters
                    cell_pad_top = 7
                    cell_pad_left = 10
                    row_height = 30

                    # Calculate cell dimensions
                    table_widget = self.feature_table
                    page_rect = printer.pageRect()
                    col_width = page_rect.width() // table_widget.columnCount()

                    # Draw page header/title
                    font.setBold(True)
                    painter.setFont(font)
                    title = "Suggested Experiment" if self.step == 2 else (
                        "Prediction" if self.step == 5 else "Formulation")
                    painter.drawText(cell_pad_left, y, f"{title} {i+1}")
                    y += row_height

                    # Draw headers
                    for col in range(table_widget.columnCount()):
                        header = table_widget.horizontalHeaderItem(col)
                        text = header.text() if header else f"Column {col}"
                        border_rect = QtCore.QRect(
                            col * col_width, y, col_width, row_height)
                        text_rect = QtCore.QRect(border_rect)
                        text_rect.adjust(cell_pad_left, cell_pad_top, 0, 0)
                        painter.drawText(text_rect, 0, text)
                        painter.drawRect(border_rect)
                    y += row_height

                    # Draw data rows
                    font.setBold(False)
                    painter.setFont(font)
                    for row in range(table_widget.rowCount()):
                        skip = True
                        for col in range(table_widget.columnCount()):
                            item = table_widget.item(row, col)
                            if item:
                                text = item.text()
                            else:
                                widget = table_widget.cellWidget(row, col)
                                if widget:
                                    text = widget.currentText()
                                else:
                                    continue  # skip blank rows
                            if table_widget.isRowHidden(row):
                                continue  # skip hidden rows
                            skip = False
                            border_rect = QtCore.QRect(
                                col * col_width, y, col_width, row_height)
                            text_rect = QtCore.QRect(border_rect)
                            text_rect.adjust(
                                cell_pad_left, cell_pad_top, 0, 0)
                            painter.drawText(text_rect, 0, text)
                            painter.drawRect(border_rect)
                        if not skip:
                            y += row_height

                painter.end()

                if info_on_success:
                    QtWidgets.QMessageBox.information(
                        None, "Success", "PDF exported successfully!")

                if open_on_success:
                    webbrowser.open(file_path)

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    None, "Error", f"Failed to export PDF: {str(e)}")

    def user_run_browse(self) -> None:
        """Open a dialog to browse and add run capture files or directories.

        This method allows the user to select one or more directories containing
        run files. It will search each selected directory for `capture.zip` files
        and load them all in batch.
        """
        # Use a custom directory dialog that supports multi-selection
        dir_dialog = QtWidgets.QFileDialog(self)
        dir_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dir_dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        dir_dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        file_view = dir_dialog.findChild(QtWidgets.QListView, 'listView')
        if file_view:
            file_view.setSelectionMode(
                QtWidgets.QAbstractItemView.ExtendedSelection)

        tree_view = dir_dialog.findChild(QtWidgets.QTreeView)
        if tree_view:
            tree_view.setSelectionMode(
                QtWidgets.QAbstractItemView.ExtendedSelection)
        prefer_abs = os.path.abspath(Constants.log_prefer_path)
        dir_dialog.setDirectory(prefer_abs)

        if dir_dialog.exec_():
            selected_dirs = dir_dialog.selectedFiles()
            if selected_dirs:
                self.load_runs_from_directories(selected_dirs)

    def load_runs_from_directories(self, directories: list) -> None:
        """Search multiple directories for capture files and load them.

        Args:
            directories (list): List of directory paths to search.
        """
        all_capture_files = []
        for directory in directories:
            capture_files = self.find_capture_files(directory)
            all_capture_files.extend(capture_files)

        directory_ies = f"director{'y' if len(directories) == 1 else 'ies'}"
        if not all_capture_files:
            QtWidgets.QMessageBox.information(
                self,
                "No Capture Files Found",
                f"No capture.zip files found in {len(directories)} selected {directory_ies}."
            )
            return
        dir_summary = f"{len(directories)} {directory_ies}"

        # NOTE: This block is unique to `frame_step1.py` (not in evaluation_ui.py)
        if len(all_capture_files) > 1 and not self.step == 3:  # 3: Imported Experiments
            # User selected multiple runs on a step that only support a single run
            reply = QtWidgets.QMessageBox.warning(
                self,
                "Multiple Runs Selected",
                f"Found {len(all_capture_files)} run file(s) in {dir_summary}.\n"
                f"However, this step only supports loading a single run.\n"
                f"Please select a directory containing only one run.",
                QtWidgets.QMessageBox.Retry | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Retry
            )
            if reply == QtWidgets.QMessageBox.Retry:
                # Try again, using a timer, to prevent stack overflow.
                QtCore.QTimer.singleShot(0, self.user_run_browse)
            return

        # NOTE: This block is modified from `evaluation_ui.py` to return on "No" reply
        # Also, this block is only executed when on the Imported Experiments tab step.
        if self.step == 3:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Batch Load Runs",
                f"Found {len(all_capture_files)} run file(s) in {dir_summary}.\n"
                f"Do you want to load {'all of them' if len(all_capture_files) > 1 else 'it'}?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes
            )
            if reply == QtWidgets.QMessageBox.No:
                return
        # If we haven't bailed at a `return` yet, proceed to loading run files.
        self.batch_add_run_files(all_capture_files)

    def find_capture_files(self, directory: str, filename: str = "capture.zip") -> list:
        """Recursively search a directory for capture files.

        Args:
            directory (str): The root directory to search.
            filename (str): The filename to search for (default: "capture.zip").

        Returns:
            list: A list of absolute paths to found capture files.
        """
        capture_files = []

        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower() == filename.lower():
                        capture_files.append(os.path.join(root, file))
        except Exception as e:
            Log.e(f"Error searching directory {directory}: {e}")

        return capture_files

    def batch_add_run_files(self, file_paths: list) -> None:
        """Load multiple run files in batch and update the UI.

        This method processes a list of run files, attempting to load each one.
        It provides feedback on the batch loading process, including success
        and failure counts.

        Args:
            file_paths (list): List of file paths to be loaded.
        """
        if not file_paths:
            return

        # Show progress dialog for batch loading
        file_s = f"file{'s' if len(file_paths) > 1 else ''}"
        progress = QtWidgets.QProgressDialog(
            f"Loading run {file_s}...",
            "Cancel",
            0,
            len(file_paths),
            self
        )
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)

        loaded_count = 0
        failed_files = []

        for i, file_path in enumerate(file_paths):
            if progress.wasCanceled():
                break

            progress.setValue(i)
            progress.setLabelText(
                f"Loading: {os.path.basename(os.path.dirname(file_path))}")
            QtWidgets.QApplication.processEvents()

            self.file_selected(file_path, loading=True)
            success = self.run_figure_valid  # True on successful load
            if success:
                loaded_count += 1
            else:
                failed_files.append(file_path)

        progress.setValue(len(file_paths))

        # Show summary
        summary_msg = f"Successfully loaded {loaded_count} of {len(file_paths)} run file(s)."
        if failed_files:
            summary_msg += f"\n\nFailed to load {len(failed_files)} file(s):"
            for failed in failed_files[:5]:  # Show first 5 failures
                # bullet point (U+2022)
                summary_msg += f"\n  \u2022 {os.path.basename(os.path.dirname(failed))}"
            if len(failed_files) > 5:
                summary_msg += f"\n  ... and {len(failed_files) - 5} more"

        if failed_files:
            QtWidgets.QMessageBox.warning(
                self,
                "Batch Load Complete",
                summary_msg
            )
        elif loaded_count > 0:
            QtWidgets.QMessageBox.information(
                self,
                "Batch Load Complete",
                summary_msg
            )

    def file_selected(self, path: str | None, cancel: bool = False, loading: bool = False):
        # If run already loaded, try saving formulation to write any changed Run Info to XML
        if self.run_file_xml and self.step in [1, 3] and not loading:
            if not self.feature_table.allSet():
                result = QtWidgets.QMessageBox.question(
                    None,
                    Constants.app_title,
                    "You have missing feature values.\n\nAre you sure you want to reload the features table?")
                if result != QtWidgets.QMessageBox.Yes:
                    return
            elif not self.feature_table.isEmpty():
                Log.i("Saving formulation for fully populated feature table.")
                if not self.save_formulation(cancel):
                    return

        self.run_file_run = path
        self.run_file_xml = None
        self.run_file_analyze = None

        # clear all fields, before repopulating them
        self.select_label.clear()
        self.run_notes.clear()
        self.run_name.clear()
        self.run_date_time.clear()
        self.run_duration.clear()
        self.run_temperature.clear()
        self.run_batch.clear()
        self.run_fill_type.clear()
        self.run_captured.clear()
        self.run_updated.clear()
        self.run_analyzed.clear()
        self.run_figure.clear()
        self.run_figure_valid = False
        self.run_canvas.draw()
        self.feature_table.clear()

        self.btn_update.setEnabled(False)

        if path is None:
            if self.step == 1:  # Select
                self.parent.select_formulation = Formulation()
            if self.step == 3:  # Import Experiments
                self.list_view.clearSelection()
                self.parent.import_formulations.clear()
                self.parent.import_run_names.clear()
            if self.step == 5:  # Predict
                self.parent.predict_formulation = Formulation()
            return

        prefer_abs = os.path.abspath(Constants.log_prefer_path)
        path_abs = os.path.abspath(path)
        try:
            inside = os.path.commonpath([prefer_abs, path_abs]) == prefer_abs
        except ValueError:
            inside = False
        if not inside:
            Log.e("The selected run is not in your working directory and cannot be used.")
            Log.e("If desired, please change your working directory to use this run.")
            # deselect file run, will show log_prefer_path next time
            self.run_file_run = None
            return

        self.btn_update.setEnabled(True)

        namelist = SecureOpen.get_namelist(self.run_file_run)
        for file in namelist:
            if file.endswith(".csv"):
                self.run_file_run = os.path.join(
                    os.path.dirname(self.run_file_run), file)
                break

        self.select_label.setText(
            os.path.basename(os.path.dirname(self.run_file_run)))

        if self.step == 3:
            item = QtGui.QStandardItem(self.select_label.text())

            # Disallow user from selecting same run for Step 1 and Step 3.
            if item.text() == self.parent.tab_widget.widget(0).select_label.text():
                QtCore.QTimer.singleShot(100, lambda: QtWidgets.QMessageBox.information(
                    None,
                    Constants.app_title,
                    "The selected run from Step 1 cannot also be an imported experiment run.",
                    QtWidgets.QMessageBox.Ok))
                return

            found = self.model.findItems(item.text())
            if len(found) == 0:
                if len(self.all_files) == 0:
                    self.model.removeRow(0)  # no_item placeholder
                self.model.appendRow(item)
                new_index = self.model.indexFromItem(item)
                self.list_view.setCurrentIndex(new_index)
                self.all_files[item.text()] = path

        folder = os.path.dirname(self.run_file_run)
        files: list[str] = os.listdir(folder)

        # Find XML and analyze files
        for f in files:
            if f.endswith(".xml"):
                self.run_file_xml = os.path.join(folder, f)
            if f.startswith("analyze") and f.endswith(".zip"):
                # Store the latest analyze file (already handled by Parser)
                self.run_file_analyze = os.path.join(folder, f)

        if self.run_file_xml is None:
            self.run_notes.setTextBackgroundColor(Color.light_red)
            self.run_notes.setText("ERROR: Cannot find XML file for this run!")
            return
        if self.run_file_analyze is None:
            self.run_notes.setTextBackgroundColor(Color.light_yellow)
            self.run_notes.setText("This run has not been analyzed yet.\n" +
                                   "Please Analyze and try again!")
            return

        # Initialize Parser object
        try:
            xml_parser = Parser(self.run_file_xml)
        except Exception as e:
            self.run_notes.setTextBackgroundColor(Color.light_red)
            self.run_notes.setText(
                f"ERROR: Failed to parse XML file!\n{str(e)}")
            Log.e(f"Parser initialization failed: {e}")
            return

        # Check if bioformulation
        if not xml_parser.is_bioformulation():
            self.run_notes.setTextBackgroundColor(Color.light_red)
            self.run_notes.setText("ERROR: This run is not a bioformulation!")
            return

        # Get metrics, audits, and parameters from Parser
        try:
            xml_metrics = xml_parser.get_metrics()
            xml_audits = xml_parser.get_audits()

            # Populate run info fields
            self.run_notes.setTextBackgroundColor(Color.white)
            notes = xml_parser.get_run_notes()
            self.run_notes.setPlainText(notes if notes else "")

            run_name = xml_parser.get_run_name()
            self.run_name.setText(
                run_name if run_name else self.select_label.text())

            # Set date/time
            start_time = xml_metrics.get("start", "(Unknown)")
            self.run_date_time.setText(start_time.replace(
                "T", " ") if start_time != "(Unknown)" else start_time)

            # Set duration
            duration = xml_metrics.get("duration", "(Unknown)")
            self.run_duration.setText(duration)

            # Set batch and fill type
            batch = xml_parser.get_batch_number()
            self.run_batch.setText(batch if batch else "(Not Provided)")

            fill_type = xml_parser.get_fill_type()
            self.run_fill_type.setText(fill_type)

            # Set audit information
            def format_audit(audit_tuple):
                """Helper to format audit tuple (username, timestamp)"""
                if audit_tuple and len(audit_tuple) == 2:
                    username, timestamp = audit_tuple
                    # Remove milliseconds from timestamp
                    if "." in timestamp:
                        timestamp = timestamp[:timestamp.index(".")]
                    timestamp = timestamp.replace("T", " ")
                    return f"{username} at {timestamp}"
                return "(Not Performed)"

            capture_audit = xml_audits.get('CAPTURE', None)
            self.run_captured.setText(format_audit(capture_audit))

            params_audit = xml_audits.get('PARAMS', None)
            # If no PARAMS audit, use CAPTURE timestamp
            self.run_updated.setText(format_audit(
                params_audit) if params_audit else self.run_captured.text())

            analyze_audit = xml_audits.get('ANALYZE', None)
            self.run_analyzed.setText(format_audit(analyze_audit))

        except Exception as e:
            self.run_notes.setTextBackgroundColor(Color.light_red)
            self.run_notes.setText(
                f"ERROR: Failed to extract run information!\n{str(e)}")
            Log.e(f"Failed to extract run info: {e}")
            return

        # Get formulation data from Parser
        try:
            formulation_obj = xml_parser.get_formulation()
            vp = formulation_obj.viscosity_profile
            temp = formulation_obj.temperature
            found_protein = xml_parser.get_protein().get("found")
            found_buffer = xml_parser.get_buffer().get("found")
            found_stabilizer = xml_parser.get_stabilizer().get("found")
            found_surfactant = xml_parser.get_surfactant().get("found")
            found_excipient = xml_parser.get_excipient().get("found")
            found_salt = xml_parser.get_salt().get("found")
        except Exception as e:
            self.run_notes.setTextBackgroundColor(Color.light_red)
            self.run_notes.setText(
                f"ERROR: Failed to parse formulation data!\n{str(e)}")
            Log.e(f"Failed to parse formulation: {e}")
            return

        # Build run_features dictionary from formulation
        run_features = copy.deepcopy(self.default_features)

        # Helper function to normalize ingredient names
        def normalize_name(name):
            """Normalize ingredient names for consistent display"""
            if name == "TWEEN80":
                return "Tween-80"
            elif name == "TWEEN20":
                return "Tween-20"
            return name

        # Populate feature values from formulation
        protein = formulation_obj.protein
        buffer = formulation_obj.buffer
        surfactant = formulation_obj.surfactant
        stabilizer = formulation_obj.stabilizer
        salt = formulation_obj.salt
        excipient = formulation_obj.excipient

        # Protein (indices 0-5)
        if protein and protein.ingredient.name and found_protein.get("name", True):
            protein_name = normalize_name(protein.ingredient.name)
            if isinstance(run_features["Value"][0], dict):
                run_features["Value"][0]["selected"] = protein_name
            else:
                run_features["Value"][0] = protein_name

            # Protein concentration
            if formulation_obj.protein.concentration is not None and found_protein.get("conc", True):
                run_features["Value"][1] = str(
                    formulation_obj.protein.concentration)

            # Protein characteristics from database
            db_protein = self.parent.ing_ctrl.get_protein_by_name(
                name=protein.ingredient.name)
            if db_protein:
                if db_protein.class_type:
                    run_features["Value"][2]["selected"] = db_protein.class_type.value
                if db_protein.molecular_weight:
                    run_features["Value"][3] = db_protein.molecular_weight
                if db_protein.pI_mean:
                    run_features["Value"][4] = db_protein.pI_mean
                if db_protein.pI_range:
                    run_features["Value"][5] = db_protein.pI_range

        # Buffer (indices 6-8)
        if buffer and buffer.ingredient.name and found_buffer.get("name", True):
            buffer_name = normalize_name(buffer.ingredient.name)
            if isinstance(run_features["Value"][6], dict):
                run_features["Value"][6]["selected"] = buffer_name
            else:
                run_features["Value"][6] = buffer_name

            # Buffer concentration
            if formulation_obj.buffer.concentration and found_buffer.get("conc", True):
                run_features["Value"][7] = str(
                    formulation_obj.buffer.concentration)

            # Buffer pH from database
            db_buffer = self.parent.ing_ctrl.get_buffer_by_name(
                name=buffer.ingredient.name)
            if db_buffer and db_buffer.pH:
                run_features["Value"][8] = db_buffer.pH

        # Surfactant (indices 9-10)
        if surfactant and surfactant.ingredient.name and found_surfactant.get("name", True):
            surfactant_name = normalize_name(surfactant.ingredient.name)
            if isinstance(run_features["Value"][9], dict):
                run_features["Value"][9]["selected"] = surfactant_name
            else:
                run_features["Value"][9] = surfactant_name

            if formulation_obj.surfactant.concentration is not None and found_surfactant.get("conc", True):
                run_features["Value"][10] = str(
                    formulation_obj.surfactant.concentration)

        # Stabilizer (indices 11-12)
        if stabilizer and stabilizer.ingredient.name and found_stabilizer.get("name", True):
            stabilizer_name = normalize_name(stabilizer.ingredient.name)
            if isinstance(run_features["Value"][11], dict):
                run_features["Value"][11]["selected"] = stabilizer_name
            else:
                run_features["Value"][11] = stabilizer_name

            if formulation_obj.stabilizer.concentration is not None and found_stabilizer.get("conc", True):
                run_features["Value"][12] = str(
                    formulation_obj.stabilizer.concentration)

        # Salt (indices 13-14)
        if salt and salt.ingredient.name and found_salt.get("name", True):
            salt_name = normalize_name(salt.ingredient.name)
            if isinstance(run_features["Value"][13], dict):
                run_features["Value"][13]["selected"] = salt_name
            else:
                run_features["Value"][13] = salt_name

            if formulation_obj.salt.concentration is not None and found_salt.get("conc", True):
                run_features["Value"][14] = str(
                    formulation_obj.salt.concentration)

        # Excipient (indices 15-16)
        if excipient and excipient.ingredient.name and found_excipient.get("name", True):
            excipient_name = normalize_name(excipient.ingredient.name)
            if isinstance(run_features["Value"][15], dict):
                run_features["Value"][15]["selected"] = excipient_name
            else:
                run_features["Value"][15] = excipient_name

            if formulation_obj.excipient.concentration is not None and found_excipient.get("conc", True):
                run_features["Value"][16] = str(
                    formulation_obj.excipient.concentration)

        # Temperature (index 17)
        if temp and not np.isnan(temp):
            self.run_temperature.setText(f"{temp:2.2f}C")
            run_features["Value"][17] = f"{temp:0.2f}"
        else:
            self.run_temperature.setText("(Unknown)")
            run_features["Value"][17] = ""

        # Check for new and/or missing ingredients in database
        reload_ingredients = False
        ingredient_checks = [
            (0, protein, "Protein Type", Protein),
            (6, buffer, "Buffer Type", Buffer),
            (9, surfactant, "Surfactant Type", Surfactant),
            (11, stabilizer, "Stabilizer Type", Stabilizer),
            (13, salt, "Salt Type", Salt),
            (15, excipient, "Excipient Type", Excipient)
        ]

        for idx, ing, label, ingredient_class in ingredient_checks:
            if ing and ing.ingredient.name:
                item = run_features["Value"][idx]
                if isinstance(item, dict):
                    choices = item["choices"]
                    selected = item["selected"]
                else:
                    # Handle simple string values
                    selected = item
                    choices = []

                value = str(selected).strip()
                if value.casefold() not in [str(c).casefold() for c in choices] and \
                   value.casefold() != "none" and len(value) != 0:
                    reload_ingredients = True

                    # Skip protein class check (idx 2) as it's handled separately
                    if idx == 2:
                        Log.w(f"Unknown Protein Class Type: \"{value}\"")
                    else:
                        Log.w(f"Adding new {label}: \"{value}\"")
                        self.parent.ing_ctrl.add(
                            ingredient_class(enc_id=-1, name=value))

        # Reload ingredient choices if needed
        if reload_ingredients:
            self.reload_all_ingredient_choices()

            # Update run_features with new choices
            run_features["Value"][0]["choices"] = self.proteins
            run_features["Value"][2]["choices"] = self.class_types
            run_features["Value"][6]["choices"] = self.buffers
            run_features["Value"][9]["choices"] = self.surfactants
            run_features["Value"][11]["choices"] = self.stabilizers
            run_features["Value"][13]["choices"] = self.salts
            run_features["Value"][15]["choices"] = self.excipients

        # Plot viscosity profile
        self.run_figure.clear()
        self.run_figure_valid = False

        if vp and len(vp.shear_rates) > 0:
            # Store profile data
            self.profile_shears = vp.shear_rates
            self.profile_viscos = vp.viscosities

            minidx = np.argmin(self.profile_viscos)
            maxidx = np.argmax(self.profile_viscos)
            Log.i(
                f"Viscosity profile ranges from {self.profile_viscos[minidx]:.2f} to {self.profile_viscos[maxidx]:.2f} cP.")

            # Helper function for smooth plotting
            def smooth_log_interpolate(x, y, num=200, expand_factor=0.05):
                xlog = np.log10(x)
                ylog = np.log10(y)
                f_interp = interp1d(xlog, ylog, kind='linear',
                                    fill_value='extrapolate')
                xlog_min, xlog_max = xlog.min(), xlog.max()
                margin = (xlog_max - xlog_min) * expand_factor
                xs_log = np.linspace(xlog_min - margin, xlog_max + margin, num)
                xs = 10**xs_log
                ys = 10**f_interp(xs_log)
                return xs, ys

            # Create plot with modern styling
            ax = self.run_figure.add_subplot(111)
            ax.set_facecolor('#ffffff')
            self.run_figure.patch.set_facecolor('#ffffff')

            # Smooth interpolation for plotting
            shear_arr = np.asarray(self.profile_shears)
            viscos_arr = np.asarray(self.profile_viscos)
            xs, ys = smooth_log_interpolate(shear_arr, viscos_arr)

            # Color scheme - teal/cyan palette
            main_color = '#00A3DA'  # Deep cyan

            # Plot the viscosity profile
            ax.plot(xs, ys, '-', lw=2.5, color=main_color,
                    label='Measured Viscosity', zorder=3, alpha=0.95)
            ax.scatter(shear_arr, viscos_arr, s=80, color=main_color, zorder=5,
                       edgecolors='white', linewidths=2.5, alpha=1)

            # Set scales and limits
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(xs.min() * 0.8, xs.max() * 1.2)
            ax.set_ylim(self.calc_limits(yall=viscos_arr))

            # Labels and title with modern styling
            ax.set_xlabel("Shear Rate (s⁻¹)", fontsize=11, fontweight='600',
                          color='#2d3436')
            ax.set_ylabel("Viscosity (cP)", fontsize=11, fontweight='600',
                          color='#2d3436')
            ax.set_title("Viscosity Profile", fontsize=13, fontweight='600', pad=15,
                         color='#2d3436')

            # Grid styling
            ax.grid(True, which="major", ls='-',
                    alpha=0.15, color='#636e72', lw=0.8)
            ax.grid(True, which="minor", ls='-',
                    alpha=0.07, color='#b2bec3', lw=0.5)

            # Axis formatting
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
            ax.tick_params(axis='both', which='major', labelsize=9,
                           colors='#2d3436', width=1)
            ax.tick_params(axis='both', which='minor', labelsize=8,
                           colors='#636e72', width=0.5)

            # Spine styling - hide top and right, style left and bottom
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#dfe6e9')
            ax.spines['bottom'].set_color('#dfe6e9')
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

            # Legend with modern styling
            legend = ax.legend(loc='best', frameon=True, fancybox=False,
                               shadow=False, fontsize=9, framealpha=1,
                               edgecolor='#dfe6e9', borderpad=1)
            legend.get_frame().set_facecolor('#ffffff')
            legend.get_frame().set_linewidth(1)

            # Add value annotations with modern styling
            ylim = ax.get_ylim()
            y_range = np.log10(ylim[1]) - np.log10(ylim[0])

            for i in range(len(viscos_arr)):
                annotation = f'{viscos_arr[i]:.1f}'
                y_offset = viscos_arr[i] * (10 ** (y_range * 0.06))
                ax.annotate(annotation,
                            xy=(shear_arr[i], viscos_arr[i]),
                            xytext=(shear_arr[i], y_offset),
                            ha='center', va='bottom',
                            fontsize=8,
                            color='#2d3436',
                            weight='500',
                            bbox=dict(boxstyle='round,pad=0.4',
                                      facecolor='white',
                                      edgecolor=main_color,
                                      alpha=0.95,
                                      linewidth=1.2),
                            arrowprops=dict(arrowstyle='-',
                                            connectionstyle='arc3,rad=0',
                                            color=main_color,
                                            alpha=0.4,
                                            lw=1.2))

            # Add timestamp in corner
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            ax.text(0.98, 0.02, timestamp,
                    transform=ax.transAxes, fontsize=7,
                    verticalalignment='bottom', horizontalalignment='right',
                    alpha=0.4, style='italic', color='#636e72')

            self.run_figure.tight_layout()
            self.run_figure_valid = True

        else:
            # No valid data - show error message
            ax = self.run_figure.add_subplot(111)
            ax.set_facecolor('#ffffff')
            self.run_figure.patch.set_facecolor('#ffffff')
            ax.text(0.5, 0.5, "Invalid Results",
                    transform=ax.transAxes,
                    ha='center', va='center',
                    fontsize=12, fontweight='600',
                    bbox=dict(facecolor='#fff3cd', edgecolor='#856404',
                              boxstyle='round,pad=0.8', linewidth=2))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

        self.run_canvas.draw()

        # Set feature table data
        self.feature_table.setData(run_features)
        self.hide_extended_features()

        if self.feature_table.allSet():
            self.handleSplitterButton(collapse=True)
        else:
            self.handleSplitterButton(collapse=False)

    def calc_limits(self, yall):
        ymin, ymax = 0, 1000
        lower_limit = np.amin(yall) / 1.5
        power = 1
        while power > -5:
            if lower_limit > 10**power:
                lower_limit = 10**power
                break
            power -= 1
        upper_limit = np.amax(yall) * 1.5
        power = 0
        while power < 5:
            if upper_limit < 10**power:
                upper_limit = 10**power
                break
            power += 1
        if lower_limit >= upper_limit:
            Log.d(
                "Limits were auto-calculated but are in an invalid range! Using ylim [0, 1000]."
            )
        elif np.isfinite(lower_limit) and np.isfinite(upper_limit):
            Log.d(
                f"Auto-calculated y-range limits for figure are: [{lower_limit}, {upper_limit}]"
            )
            ymin = lower_limit
            ymax = upper_limit
        else:
            Log.d(
                "Limits were auto-calculated but were not finite values! Using ylim [0, 1000]."
            )
        return ymin, ymax

    def model_selected(self, path: Optional[str] = None):
        self.model_path = path

        if path is None:
            self.select_model_label.clear()
            return

        self.select_model_label.setText(
            path.split('\\')[-1].split('/')[-1].split('.')[0])
