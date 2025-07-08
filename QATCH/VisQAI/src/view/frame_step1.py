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
from typing import Dict, Any, List, Tuple, Type
from typing import TYPE_CHECKING

try:
    from src.io.file_storage import SecureOpen
    from src.models.formulation import Formulation, ViscosityProfile
    from src.models.ingredient import Ingredient, Protein, Surfactant, Stabilizer, Salt, Buffer
    from src.models.predictor import Predictor
    from src.db.db import Database
    from src.processors.sampler import Sampler
    from src.threads.executor import Executor, ExecutionRecord
    from src.utils.constraints import Constraints
    from src.view.checkable_combo_box import CheckableComboBox
    from src.view.table_view import TableView, Color
    if TYPE_CHECKING:
        from src.view.frame_step2 import FrameStep2
        from src.view.main_window import VisQAIWindow

except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.io.file_storage import SecureOpen
    from QATCH.VisQAI.src.models.formulation import Formulation, ViscosityProfile
    from QATCH.VisQAI.src.models.ingredient import Ingredient, Protein, Surfactant, Stabilizer, Salt, Buffer
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.processors.sampler import Sampler
    from QATCH.VisQAI.src.threads.executor import Executor, ExecutionRecord
    from QATCH.VisQAI.src.utils.constraints import Constraints
    from QATCH.VisQAI.src.view.checkable_combo_box import CheckableComboBox
    from QATCH.VisQAI.src.view.table_view import TableView, Color
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

        # Main layout
        main_layout = QtWidgets.QHBoxLayout(self)
        h_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(h_splitter)

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
            model_path = os.path.join(
                os.getcwd(), "QATCH/VisQAI/assets")
            self.model_dialog.setDirectory(model_path)
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
                predictor_path = "QATCH/VisQAI/assets/VisQAI-base.zip"
                if os.path.exists(predictor_path):
                    self.model_selected(path=predictor_path)
            select_model_layout.addWidget(self.select_model_btn)
            select_model_layout.addWidget(self.select_model_label)
            select_model_layout.addStretch()

            left_layout.addWidget(self.select_model_group)

        left_layout.addWidget(left_group)

        # Browse run
        self.file_dialog = QtWidgets.QFileDialog()
        self.file_dialog.setOption(
            QtWidgets.QFileDialog.DontUseNativeDialog, True)
        # NOTE: `setDirectory()` called when VisQAI mode is enabled.
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self.file_dialog.setNameFilter("Captured Runs (capture.zip)")
        self.file_dialog.selectNameFilter("Captured Runs (capture.zip)")

        self.select_run = QtWidgets.QPushButton(
            "Add Run..." if step == 3 else "Browse...")
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
            self.btn_remove = QtWidgets.QPushButton("Remove Selected Run")
            self.btn_remove.clicked.connect(self.user_run_removed)
            add_remove_export_layout.addWidget(self.btn_remove)
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
            self.btn_next = QtWidgets.QPushButton(""
                                                  "Next Step: Optimize")
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_next)
        left_layout.addLayout(btn_layout)

        # Right panel: Initialize features
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        v_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_layout.addWidget(v_splitter)
        right_header = QtWidgets.QGroupBox("Initialize Features")
        right_group = QtWidgets.QVBoxLayout(right_header)
        v_splitter.addWidget(right_header)

        # Features table
        self.load_all_excipient_types()
        self.default_features = {"Feature": ["Protein Type", "Protein Concentration",
                                             "Protein Molecular Weight", "Protein pI Mean", "Protein pI Range",  # not in Run Info
                                             "Buffer Type", "Buffer Concentration",
                                             "Buffer pH",  # not in Run Info
                                             "Surfactant Type", "Surfactant Concentration",
                                             "Stabilizer Type", "Stabilizer Concentration",
                                             "Salt Type", "Salt Concentration"],
                                 "Value": [self.proteins, "",
                                           "", "", "",  # molecular weight, pI mean, pI range
                                           self.buffers, "",
                                           "",  # buffer pH
                                           self.surfactants, "",
                                           self.stabilizers, "",
                                           self.salts, ""],
                                 "Units": ["", "mg/mL",
                                           "kDa", "", "",  # pI
                                           "", "mM",
                                           "",  # pH
                                           "", "%w",
                                           "", "M",
                                           "", "mM"]}
        self.default_rows, self.default_cols = (len(list(self.default_features.values())[0]),
                                                len(list(self.default_features.keys())))

        self.feature_table = TableView(self.default_features,
                                       self.default_rows, self.default_cols)
        self.feature_table.clear()
        right_group.addWidget(self.feature_table)

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
        #     #     del values[7]  # buffer PH
        #     #     del values[4]  # protein pI range
        #     #     del values[3]  # protein pI mean
        #     #     del values[2]  # protein weight
        #     self.dummy_features.append(dummy_feature)

        self.run_figure = Figure()
        self.run_figure_valid = False
        self.run_canvas = FigureCanvas(self.run_figure)
        v_splitter.addWidget(self.run_canvas)

        # Build main layout
        h_splitter.addWidget(left_widget)
        h_splitter.addWidget(right_widget)
        h_splitter.setSizes([100, 300])
        v_splitter.setSizes([180, 100])

        # Signals
        self.btn_cancel.clicked.connect(
            lambda: self.file_selected(None, cancel=True))
        self.btn_next.clicked.connect(
            getattr(self, f"proceed_to_step_{self.step+1}"))
        self.select_run.clicked.connect(self.file_dialog.show)
        self.file_dialog.fileSelected.connect(self.file_selected)
        if True:  # step == 5:
            self.select_model_btn.clicked.connect(self.model_dialog.show)
            self.model_dialog.fileSelected.connect(self.model_selected)

    def on_tab_selected(self):

        # Set run directory from User Preferences.
        self.file_dialog.setDirectory(Constants.log_prefer_path)

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
                predict_tab: FrameStep1 = self.parent.tab_widget.widget(4)
                optimize_tab: FrameStep2 = self.parent.tab_widget.widget(5)
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

    def unique_case_insensitive_sort(self, list):
        seen = set()
        result = []
        for item in list:
            lower_item = item.lower()
            if lower_item not in seen:
                seen.add(lower_item)
                result.append(item)

        # Sort case-insensitive
        result.sort(key=str.lower)
        return result

    def load_all_excipient_types(self):
        self.proteins: list[str] = []
        self.buffers: list[str] = []
        self.surfactants: list[str] = []
        self.stabilizers: list[str] = []
        self.salts: list[str] = []

        ingredients = self.parent.ing_ctrl.get_all_ingredients()
        for i in ingredients:
            if i.name.casefold() == "none":
                continue  # skip "none"
            if i.type == "Protein":
                self.proteins.append(i.name)
            elif i.type == "Buffer":
                self.buffers.append(i.name)
            elif i.type == "Surfactant":
                self.surfactants.append(i.name)
            elif i.type == "Stabilizer":
                self.stabilizers.append(i.name)
            elif i.type == "Salt":
                self.salts.append(i.name)

        # this is case-sensitive, which is not what we want:
        # self.excipient_proteins.sort()
        # self.excipient_surfactants.sort()
        # self.excipient_stabilizers.sort()
        # this is using a case-insensitive sorting method:
        # self.proteins = sorted(
        #     self.proteins, key=str.casefold)
        # self.buffers = sorted(
        #     self.buffers, key=str.casefold)
        # self.surfactants = sorted(
        #     self.surfactants, key=str.casefold)
        # self.stabilizers = sorted(
        #     self.stabilizers, key=str.casefold)
        # self.salts = sorted(
        #     self.salts, key=str.casefold)
        # this is unique, case-insensitive sorting method:
        self.proteins = self.unique_case_insensitive_sort(self.proteins)
        self.buffers = self.unique_case_insensitive_sort(self.buffers)
        self.surfactants = self.unique_case_insensitive_sort(self.surfactants)
        self.stabilizers = self.unique_case_insensitive_sort(self.stabilizers)
        self.salts = self.unique_case_insensitive_sort(self.salts)

        Log.d("Proteins:", self.proteins)
        Log.d("Buffers:", self.buffers)
        Log.d("Surfactants:", self.surfactants)
        Log.d("Stabilizers:", self.stabilizers)
        Log.d("Salts", self.salts)

    def hide_extended_features(self):
        hide_rows = [2, 3, 4, 7]
        for row in hide_rows:
            self.feature_table.hideRow(row)

    def save_formulation(self, cancel: bool = False) -> bool:
        protein_type = self.feature_table.cellWidget(0, 1).currentText()
        protein_conc = self.feature_table.item(1, 1).text()
        protein_weight = self.feature_table.item(2, 1).text()
        protein_pI_mean = self.feature_table.item(3, 1).text()
        protein_pI_range = self.feature_table.item(4, 1).text()
        buffer_type = self.feature_table.cellWidget(5, 1).currentText()
        buffer_conc = self.feature_table.item(6, 1).text()
        buffer_pH = self.feature_table.item(7, 1).text()
        surfactant_type = self.feature_table.cellWidget(8, 1).currentText()
        surfactant_conc = self.feature_table.item(9, 1).text()
        stabilizer_type = self.feature_table.cellWidget(
            10, 1).currentText()
        stabilizer_conc = self.feature_table.item(11, 1).text()
        salt_type = self.feature_table.cellWidget(12, 1).currentText()
        salt_conc = self.feature_table.item(13, 1).text()

        # save run info to XML (if changed, request audit sign)
        if self.step in [1, 3]:  # Select, Import
            self.parent.save_run_info(self.run_file_xml, [
                protein_type, protein_conc,
                buffer_type, buffer_conc,
                surfactant_type, surfactant_conc,
                stabilizer_type, stabilizer_conc,
                salt_type, salt_conc], cancel)
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
            feature = copy.deepcopy(self.default_features)
            feature["Value"][0] = protein_type
            feature["Value"][1] = protein_conc
            feature["Value"][2] = protein_weight
            feature["Value"][3] = protein_pI_mean
            feature["Value"][4] = protein_pI_range
            feature["Value"][5] = buffer_type
            feature["Value"][6] = buffer_conc
            feature["Value"][7] = buffer_pH
            feature["Value"][8] = surfactant_type
            feature["Value"][9] = surfactant_conc
            feature["Value"][10] = stabilizer_type
            feature["Value"][11] = stabilizer_conc
            feature["Value"][12] = salt_type
            feature["Value"][13] = salt_conc
            self.loaded_features[self.list_view.selectedIndexes()[
                0].row()] = feature

        protein = self.parent.ing_ctrl.get_protein_by_name(name=protein_type)
        if protein == None:
            protein = self.parent.ing_ctrl.add_protein(
                Protein(enc_id=-1, name=protein_type))

        buffer = self.parent.ing_ctrl.get_buffer_by_name(name=buffer_type)
        if buffer == None:
            buffer = self.parent.ing_ctrl.add_buffer(
                Buffer(enc_id=-1, name=buffer_type))

        surfactant = self.parent.ing_ctrl.get_surfactant_by_name(
            name=surfactant_type)
        if surfactant == None:
            surfactant = self.parent.ing_ctrl.add_surfactant(
                Surfactant(enc_id=-1, name=surfactant_type))

        stabilizer = self.parent.ing_ctrl.get_stabilizer_by_name(
            name=stabilizer_type)
        if stabilizer == None:
            stabilizer = self.parent.ing_ctrl.add_stabilizer(
                Stabilizer(enc_id=-1, name=stabilizer_type))

        salt = self.parent.ing_ctrl.get_salt_by_name(name=salt_type)
        if salt == None:
            salt = self.parent.ing_ctrl.add_salt(
                Salt(enc_id=-1, name=salt_type))

        def is_number(s: str):
            try:
                float(s)
                return True
            except ValueError:
                return False

        # update protein and buffer characteristics
        # bail if any extended features are missing
        if is_number(protein_weight):
            protein.molecular_weight = float(protein_weight)
        elif not protein.molecular_weight:
            Log.e("Missing protein molecular weight!")
            return
        if is_number(protein_pI_mean):
            protein.pI_mean = float(protein_pI_mean)
        elif not protein.pI_mean:
            Log.e("Missing protein pI mean!")
            return
        if is_number(protein_pI_range):
            protein.pI_range = float(protein_pI_range)
        elif not protein.pI_range:
            Log.e("Missing protein pI range!")
            return
        if is_number(buffer_pH):
            buffer.pH = float(buffer_pH)
        elif not buffer.pH:
            Log.e("Missing buffer pH!")
            return

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

        # pull temperaure
        temp = self.run_temperature.text()
        if temp.endswith('C'):
            temp = temp[:-1]  # strip Celsius unit character
        if not is_number(temp):
            temp = "nan"  # not a number, casts to float as nan

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
        form.set_viscosity_profile(profile=vp)
        form.set_temperature(float(temp))

        form_saved = self.parent.form_ctrl.add_formulation(
            formulation=form)

        if self.step == 1:
            Log.d("Saving selected formulation to parent for later")
            self.parent.select_formulation = form_saved
            # print(self.parent.form_ctrl.get_all_as_dataframe())
        if self.step == 5:
            Log.d("Saving prediction formulation to parent for later")
            self.parent.predict_formulation = form_saved

        return True

    def load_suggestion(self, constraints):
        model_name = self.select_model_label.text()
        if hasattr(self, "timer") and self.timer.isActive():
            Log.w("Busy canceling... Please wait...")
            return
        if len(self.select_model_label.text()) == 0 or self.model_path == None:
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

        def add_new_suggestion(record: ExecutionRecord):
            if self.progressBar.wasCanceled():
                Log.d("User canceled suggestion. Ignoring results.")
                return

            Log.d("Processing suggestion results!")

            form = record.result
            exception = record.exception
            if exception:
                Log.e(f"ERROR: Failed to suggest: {str(exception)}")
                return

            self.add_formulation(form)

        Log.d("Waiting for suggestion results...")
        self.parent.enable(False)
        self.executor = Executor()
        self.executor.run(
            self,
            method_name="get_new_suggestion",
            asset_name=model_name,
            # TODO: constraints=constraints,
            callback=add_new_suggestion)

    def get_new_suggestion(self, asset_name):
        database = Database(parse_file_key=True)
        sampler = Sampler(asset_name=asset_name,
                          database=database)
        form = sampler.get_next_sample()
        database.close()
        return form

    def add_formulation(self, form: Formulation):
        feature = copy.deepcopy(self.default_features)
        if form.protein:  # NOT an empty Formulation() object
            feature["Value"][0] = form.protein.ingredient.name
            feature["Value"][1] = form.protein.concentration
            feature["Value"][2] = form.protein.ingredient.molecular_weight
            feature["Value"][3] = form.protein.ingredient.pI_mean
            feature["Value"][4] = form.protein.ingredient.pI_range
            feature["Value"][5] = form.buffer.ingredient.name
            feature["Value"][6] = form.buffer.concentration
            feature["Value"][7] = form.buffer.ingredient.pH
            feature["Value"][8] = form.surfactant.ingredient.name
            feature["Value"][9] = form.surfactant.concentration
            feature["Value"][10] = form.stabilizer.ingredient.name
            feature["Value"][11] = form.stabilizer.concentration
            feature["Value"][12] = form.salt.ingredient.name
            feature["Value"][13] = form.salt.concentration

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
        if len(self.select_model_label.text()) == 0 or self.model_path == None:
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

        self.save_formulation()

        self.predictor = Predictor(zip_path=self.model_path)
        select_df = self.parent.select_formulation.to_dataframe(
            encoded=False, training=False)
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

        def run_prediction_result(record: ExecutionRecord):

            Log.d("Waiting for prediction results...")
            self.progressBar.setLabelText("Predicting...")

            self.executor.run(
                self.predictor,
                method_name="predict_uncertainty",
                data=predict_df,
                callback=get_prediction_result)

        def get_prediction_result(record: ExecutionRecord):

            if self.progressBar.wasCanceled():
                Log.d("User canceled prediction. Ignoring results.")
                return

            Log.d("Processing prediction results!")

            # The returns from this are a predicted viscosity profile [val1,val2,...val5]
            # predicted_vp = self.parent.predictor.predict(data=form_df)

            # The returns from this are a predicted viscosity profile [val1,val2,...val5] and
            # a series of standard deviations for each predicted value.
            predicted_mean_vp, mean_std = record.result
            exception = record.exception
            if exception:
                Log.e(f"ERROR: Prediction exception: {str(exception)}")

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

            def make_plot(name, shear, mean_arr, std_arr, title, color):
                # clear existing plot before making a new one
                self.run_figure_valid = False
                self.run_figure.clear()
                self.run_canvas.draw()

                ax = self.run_figure.add_subplot(111)
                xs, ys = smooth_log_interpolate(shear, mean_arr)
                xs_up, ys_up = smooth_log_interpolate(
                    shear, mean_arr + std_arr)
                xs_dn, ys_dn = smooth_log_interpolate(
                    shear, mean_arr - std_arr)
                ax.plot(xs, ys, '-', lw=2.5, color=color)
                ax.fill_between(xs_dn, ys_dn, ys_up, alpha=0.25, color=color)
                ax.scatter(shear, mean_arr, s=40, color=color, zorder=5)
                ax.set_xlim(xs.min(), xs.max())
                ann = "\n".join(f"{x:.0e}: {m:.1f}±{s:.1f}" for x, m,
                                s in zip(shear, mean_arr, std_arr))
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_ylim(self.calc_limits(yall=np.concat((ys_dn, ys_up))))
                ax.set_xlabel("Shear rate (s⁻¹)", fontsize=10)
                ax.set_ylabel("Viscosity (cP)", fontsize=10)
                ax.grid(True, which="both", ls=":")
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))

                self.run_figure_valid = True
                self.run_canvas.draw()

            # Plot
            make_plot("name", self.profile_shears,
                      predicted_mean_vp[0], mean_std[0], "title", "blue")

        self.executor = Executor()

        if self.parent.select_formulation.viscosity_profile.is_measured:

            # Get the viscosity profile or y target to update with.
            vp = self._get_viscosity_list(self.parent.select_formulation)

            # Target needs to be form np.array([[Viscosity_100, ..., Viscosity_15000000]])
            # Also I have this set so updating does not overwrite the existing model until
            # we figure out how model storage works
            self.executor.run(
                self.predictor,
                method_name="update",
                new_data=select_df,
                new_targets=np.array([vp]),
                epochs=10,
                batch_size=32,
                save=False,
                callback=run_prediction_result)

        else:
            run_prediction_result()

    def check_finished(self):
        record_count = 1
        if self.step == 5 and self.parent.select_formulation.viscosity_profile.is_measured:
            record_count += 1
        if self.executor.active_count() == 0 and len(self.executor.get_task_records()) == record_count:
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
        # First of all, there must be at least 1 imported experiment
        # For each run in list, must pass the same criteria from Step 1
        #   1. All audits contain valid values
        #   2. All initial features are set
        #   3. Analyze results are valid
        #   4. All formulations saved, and XMLs up-to-date
        if len(self.all_files) == 0:
            QtWidgets.QMessageBox.information(
                None, Constants.app_title,
                "Please import at least 1 experiment before proceeding.",
                QtWidgets.QMessageBox.Ok)
            return
        all_is_good = True
        for file_name, file_path in self.all_files.items():
            self.file_selected(file_path)  # load each run
            if (len(self.run_captured.text()) and
                len(self.run_updated.text()) and
                len(self.run_analyzed.text()) and
                    self.feature_table.allSet() and
                    self.run_figure_valid):
                if not self.save_formulation():
                    return
            else:
                all_is_good = False
                # break # maybe not, if we want to highlight *all* errors on "Next"
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
            self.add_suggestion_dialog()
        if self.step == 5:
            self.add_formulation(Formulation())

    def add_suggestion_dialog(self):
        self.suggest_dialog = QtWidgets.QDialog(self)
        self.suggest_dialog.setWindowTitle("Add Suggestion(s)")
        self.suggest_dialog.setModal(True)
        # hide question mark from title bar of window
        self.suggest_dialog.setWindowFlag(
            QtCore.Qt.WindowContextHelpButtonHint, False)
        self.suggest_dialog.setMinimumSize(500, 300)
        self.suggest_dialog.setWindowIcon(QtGui.QIcon(
            os.path.join(Architecture.get_path(), 'QATCH/icons/qmodel.png')))

        layout = QtWidgets.QVBoxLayout(self.suggest_dialog)

        label = QtWidgets.QLabel("How many suggestions do you want to add?")
        layout.addWidget(label)

        self.suggestion_text = QtWidgets.QComboBox(self.suggest_dialog)
        self.suggestion_text.addItems(
            list(map(str, range(1, 11))))  # 1 to 10 suggestions
        self.suggestion_text.setEditable(True)
        layout.addWidget(self.suggestion_text)

        self.constraints_group = QtWidgets.QGroupBox(
            "Constraints", self.suggest_dialog)
        self.constraints_layout = QtWidgets.QVBoxLayout(self.constraints_group)

        self.constraints_none = QtWidgets.QLabel(
            "None", self.constraints_group)
        self.constraints_none.setToolTip("No constraints on the suggestions")
        self.constraints_layout.addWidget(self.constraints_none)

        self.constraints_rows = []
        self.constraints_ingredients = []
        self.constraints_features = []
        self.constraints_verbs = []
        self.constraints_values = []
        self.constraints_delete_buttons = []

        layout.addWidget(self.constraints_group)

        self.add_constraints_btn = QtWidgets.QPushButton(
            icon=self.rotate_and_crop_icon(QtGui.QIcon(
                os.path.join(Architecture.get_path(), 'QATCH/icons/cancel.png')), 45, 50),
            text="   Add Constraint",
            parent=self.suggest_dialog)
        self.add_constraints_btn.setToolTip(
            "Add a new constraint for the suggestions")
        self.add_constraints_btn.clicked.connect(self.add_new_constraint)
        layout.addWidget(self.add_constraints_btn)

        layout.addStretch(1)  # add stretch to push buttons to the bottom

        button_box = QtWidgets.QDialogButtonBox(self.suggest_dialog)
        button_box.setStandardButtons(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        button_box.accepted.connect(self.accept_suggestions)
        button_box.rejected.connect(self.suggest_dialog.reject)

        self.suggest_dialog.open()  # non-blocking

    def accept_suggestions(self):
        Log.d("Accepting suggestions...")
        num_suggestions = self.suggestion_text.currentText()
        if not num_suggestions.isdigit() or int(num_suggestions) < 1:
            Log.w("Invalid number of suggestions:", num_suggestions)
            return
        constraints = self.build_constraints()
        if not constraints:
            Log.w(
                "Missing/invalid fields in Constraints. Please fill out or remove them.")
            return

        self.suggest_dialog.accept()  # Hide the dialog

        Log.d("Constraints:", constraints.build())
        for _ in range(int(num_suggestions)):
            self.load_suggestion(constraints)  # Add a new suggestion
            while self.timer.isActive():
                QtWidgets.QApplication.processEvents()
            if self.progressBar.wasCanceled():
                Log.d("User canceled adding suggestions. Stopping.")
                break

    def add_new_constraint(self):
        self.constraints_none.setVisible(False)  # hide "None" label

        # Create a new row for the constraint
        self.constraints_rows.append(QtWidgets.QHBoxLayout())
        # The constraint ingredient can be one of the following:
        #   Protein, Buffer, Surfactant, Stabilizer, Salt
        self.constraints_ingredients.append(
            QtWidgets.QComboBox(self.constraints_group))
        self.constraints_ingredients[-1].addItems([
            "Protein", "Buffer", "Surfactant", "Stabilizer", "Salt"])
        # No selection by default
        self.constraints_ingredients[-1].setCurrentIndex(-1)
        self.constraints_ingredients[-1].currentIndexChanged.connect(
            lambda index, idx=len(self.constraints_ingredients)-1: self.autofill_constraint_values(idx))
        # When ingredient changes, autofill possible values
        self.constraints_rows[-1].addWidget(self.constraints_ingredients[-1])
        # The constraint feature can be one of the following:
        #   Type, Concentration
        self.constraints_features.append(
            QtWidgets.QComboBox(self.constraints_group))
        self.constraints_features[-1].addItems([
            "Type", "Concentration"])
        # No selection by default
        self.constraints_features[-1].setCurrentIndex(-1)
        self.constraints_features[-1].currentIndexChanged.connect(
            lambda index, idx=len(self.constraints_features)-1: self.autofill_constraint_values(idx))
        # When feature changes, autofill possible values
        self.constraints_rows[-1].addWidget(self.constraints_features[-1])
        # The constraint verb can be one of the following:
        #   is, is not
        self.constraints_verbs.append(
            QtWidgets.QComboBox(self.constraints_group))
        self.constraints_verbs[-1].addItems(["is", "is not"])
        # No selection by default
        self.constraints_verbs[-1].setCurrentIndex(-1)
        self.constraints_verbs[-1].currentIndexChanged.connect(
            lambda index, idx=len(self.constraints_verbs)-1: self.autofill_constraint_values(idx))
        self.constraints_rows[-1].addWidget(self.constraints_verbs[-1])
        # The constraint value can be a single, multiple or range of values
        # (i.e. "PBS", "tween-20,tween-80" or "0.01-0.2")
        self.constraints_values.append(
            CheckableComboBox(self.constraints_group))
        # self.constraints_values[-1].setEditable(True)
        # TODO: Add items for now just for debugging the combination selection
        # self.constraints_values[-1].addItems([
        #     "PBS", "tween-20", "tween-80", "0.01", "0.02", "0.03",
        #     "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.1",
        #     "0.2", "0.3", "0.4", "0.5"])
        # No selection by default
        self.constraints_values[-1].setCurrentIndex(-1)
        # with stretch
        self.constraints_rows[-1].addWidget(self.constraints_values[-1], 1)
        # Delete button to clear this constraint from the list
        self.constraints_delete_buttons.append(QtWidgets.QToolButton(
            icon=QtGui.QIcon(
                os.path.join(Architecture.get_path(), 'QATCH/icons/cancel.png')),
            text=None,
            parent=self.constraints_group))
        self.constraints_delete_buttons[-1].clicked.connect(
            lambda: self.remove_constraint(len(self.constraints_delete_buttons) - 1))
        self.constraints_delete_buttons[-1].setFixedWidth(
            self.constraints_delete_buttons[-1].sizeHint().height())  # make it square
        self.constraints_delete_buttons[-1].setToolTip(
            "Delete this constraint")
        self.constraints_delete_buttons[-1].setCursor(
            QtCore.Qt.PointingHandCursor)
        # Add the delete button to the row
        self.constraints_rows[-1].addWidget(
            self.constraints_delete_buttons[-1])
        self.constraints_layout.addLayout(self.constraints_rows[-1])

    def autofill_constraint_values(self, index: int):
        if index < 0 or index >= len(self.constraints_ingredients) or \
                index >= len(self.constraints_features) or \
                index >= len(self.constraints_verbs) or \
                index >= len(self.constraints_values):
            Log.e("Invalid constraint index:", index)
            return

        ingredient = self.constraints_ingredients[index].currentText()
        feature = self.constraints_features[index].currentText()
        verb = self.constraints_verbs[index].currentText()

        model = self.constraints_values[index].model()
        current_items = [model.data(model.index(i, 0))
                         for i in range(model.rowCount())]

        if not ingredient or not feature or not verb:
            return  # nothing selected yet

        autofill_items = []
        editable = False
        if feature == "Type":
            if ingredient == "Protein":
                autofill_items = self.proteins.copy()
            elif ingredient == "Buffer":
                autofill_items = self.buffers.copy()
            elif ingredient == "Surfactant":
                autofill_items = self.surfactants.copy()
            elif ingredient == "Stabilizer":
                autofill_items = self.stabilizers.copy()
            elif ingredient == "Salt":
                autofill_items = self.salts.copy()
            if ingredient != "Protein" and "None" not in autofill_items:
                autofill_items.insert(0, "None")  # allow "none" selection
        elif feature == "Concentration":
            editable = True

        # Clear and set new items or placeholder (if different)
        if autofill_items != current_items or editable != self.constraints_values[index].isEditable():
            self.constraints_values[index].clear()
            self.constraints_values[index].setEditable(editable)
            if editable:
                self.constraints_values[index].update_label(autofill_items)
            else:
                self.constraints_values[index].addItems(autofill_items)
                self.constraints_values[index].setCurrentIndex(-1)

    def remove_constraint(self, index: int):
        if index < 0 or index >= len(self.constraints_delete_buttons):
            Log.e("Invalid constraint index:", index)
            return

        # Remove the constraint from the layout and delete the widgets
        self.constraints_rows[index].setParent(None)
        self.constraints_rows[index].deleteLater()
        del self.constraints_rows[index]
        self.constraints_ingredients[index].setParent(None)
        self.constraints_ingredients[index].deleteLater()
        del self.constraints_ingredients[index]
        self.constraints_features[index].setParent(None)
        self.constraints_features[index].deleteLater()
        del self.constraints_features[index]
        self.constraints_verbs[index].setParent(None)
        self.constraints_verbs[index].deleteLater()
        del self.constraints_verbs[index]
        self.constraints_values[index].setParent(None)
        self.constraints_values[index].deleteLater()
        del self.constraints_values[index]
        self.constraints_delete_buttons[index].setParent(None)
        self.constraints_delete_buttons[index].deleteLater()
        del self.constraints_delete_buttons[index]

        if len(self.constraints_rows) == 0:
            # If no constraints left, show the "None" label again
            self.constraints_none.setVisible(True)

    def rotate_and_crop_icon(self, icon: QtGui.QIcon, angle: float, size: int = 64) -> QtGui.QIcon:
        # Get original pixmap
        original_pixmap = icon.pixmap(size, size)

        # Rotate the pixmap
        transform = QtGui.QTransform()
        transform.rotate(angle)
        rotated_pixmap = original_pixmap.transformed(
            transform, QtCore.Qt.SmoothTransformation)

        # Calculate crop rectangle to center crop back to original size
        rotated_size = rotated_pixmap.size()
        x = (rotated_size.width() - size) // 2
        y = (rotated_size.height() - size) // 2

        # Crop from center
        crop_rect = QtCore.QRect(x, y, size, size)
        cropped_pixmap = rotated_pixmap.copy(crop_rect)

        return QtGui.QIcon(cropped_pixmap)

    def build_constraints(self) -> Constraints | None:
        added_constraints = []
        for i in range(len(self.constraints_ingredients)):
            ingredient = self.constraints_ingredients[i].currentText()
            feature = self.constraints_features[i].currentText()
            verb = self.constraints_verbs[i].currentText()
            values = self.constraints_values[i].currentText()
            if not ingredient or not feature or not verb or not values:
                return None  # missing fields

            # Split values by semicolon and strip whitespace
            values = [v.strip() for v in values.split(";") if v.strip()]
            if not values:
                return None  # no valid values

            if feature == "Concentration":
                # If feature is "Concentration", values can be a single value or a range
                # Check if value contains a range (e.g. "0.01-0.1") and convert to tuple (0.01, 0.1)
                # Otherwise, convert the single value to float
                # If conversion fails, return None
                for j in range(len(values)):
                    if "-" in values[j]:
                        try:
                            # NOTE: This will not handle negative values correctly
                            start, end = values[j].split("-")
                            start = float(start.strip())
                            end = float(end.strip())
                            values[j] = (start, end)
                        except ValueError:
                            return None  # invalid range format
                    else:
                        try:
                            values[j] = float(values[j].strip())
                        except ValueError:
                            return None  # invalid float format

            added_constraints.append((ingredient, feature, verb, values))

        constraints = Constraints(self.parent.database)

        # Populate constraints object from user added constraints
        for ingredient, feature, verb, values in added_constraints:
            constraint_name = f"{ingredient}_{feature[:4].lower()}"
            if constraint_name in Constraints._CATEGORICAL:
                # Categorical constraint
                choices: List[Ingredient] = []
                for value in values:
                    choice = self.parent.ing_ctrl.get_by_name(
                        name=value,
                        ingredient=Constraints._FEATURE_CLASS[constraint_name]
                        (enc_id=-1, name="Dummy Ingredient subclass instance"))
                    choices.append(choice)
                constraints.add_choices(
                    feature=constraint_name,
                    choices=choices)
            elif constraint_name in Constraints._NUMERIC:
                # Numerical constraint
                for value in values:
                    constraints.add_range(
                        feature=constraint_name,
                        low=min(value) if isinstance(value, tuple) else value,
                        high=max(value) if isinstance(value, tuple) else value)
            else:
                Log.e(f"Unknown constraint: {constraint_name}")
                return None

        return constraints

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
                    painter.drawText(cell_pad_left, y,
                                     f"Suggested Experiment {i+1}")
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

    def file_selected(self, path: str | None, cancel: bool = False):
        # If run already loaded, try saving formulation to write any changed Run Info to XML
        if self.run_file_xml and self.step in [1, 3]:
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
            if self.step == 5:  # Predict
                self.parent.predict_formulation = Formulation()
            if True:  # Always, all tabs
                self.model_selected(None)
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
        max_index = 0
        for f in files:
            if f.endswith(".xml"):
                self.run_file_xml = os.path.join(folder, f)
            if f.startswith("analyze") and f.endswith(".zip"):
                this_index = int(f[f.index("-")+1:f.index(".")])
                if this_index > max_index:
                    max_index = this_index
                self.run_file_analyze = os.path.join(folder,
                                                     f.replace(str(this_index), str(max_index)))
        if self.run_file_xml == None:
            self.run_notes.setTextBackgroundColor(Color.light_red)
            self.run_notes.setText("ERROR: Cannot find XML file for this run!")
            return
        if self.run_file_analyze == None:
            self.run_notes.setTextBackgroundColor(Color.light_yellow)
            self.run_notes.setText("This run has not been analyzed yet.\n" +
                                   "Please Analyze and try again!")
            return

        doc = minidom.parse(self.run_file_xml)

        xml_metrics = {}
        metrics = doc.getElementsByTagName(
            "metrics")[-1]  # most recent element
        for m in metrics.childNodes:
            if m.nodeType == m.TEXT_NODE:
                continue  # only process elements
            name = m.getAttribute("name")
            value = m.getAttribute("value")
            if m.hasAttribute("units"):
                value = f"{value} {m.getAttribute('units')}"
            xml_metrics[name] = value

        xml_audits = {}
        audits = doc.getElementsByTagName(
            "audits")[-1]  # most recent element
        for a in audits.childNodes:
            if a.nodeType == a.TEXT_NODE:
                continue  # only process elements
            key = a.getAttribute("action")
            captured_by = a.getAttribute("username")
            captured_at = a.getAttribute("recorded")
            value = (captured_by, captured_at)
            xml_audits[key] = value

        xml_params = {}
        params = doc.getElementsByTagName(
            "params")[-1]  # most recent element
        for p in params.childNodes:
            if p.nodeType == p.TEXT_NODE:
                continue  # only process elements
            name = p.getAttribute("name")
            value = p.getAttribute("value")
            if p.hasAttribute("found"):
                value = f"{value} ({'Valid' if eval(p.getAttribute('found')) else 'Unknown'})"
            xml_params[name] = value

        if xml_params.get("bioformulation", False) != 'True':
            self.run_notes.setTextBackgroundColor(Color.light_red)
            self.run_notes.setText("ERROR: This run is not a bioformulation!")
            return

        try:
            self.run_notes.setTextBackgroundColor(Color.white)
            self.run_notes.setPlainText(
                xml_params["notes"].replace("\\n", "\n"))
        except:
            self.run_notes.setPlainText(None)
        try:
            self.run_name.setText(xml_params["run_name"])
        except:
            self.run_name.setText(self.select_label.text())
        try:
            self.run_date_time.setText(xml_metrics["start"].replace("T", " "))
        except:
            self.run_date_time.setText("(Unknown)")
        try:
            self.run_duration.setText(xml_metrics["duration"])
        except:
            self.run_duration.setText("(Unknown)")
        try:
            self.run_batch.setText(xml_params["batch_number"])
        except:
            self.run_batch.setText("(Not Provided)")
        try:
            self.run_fill_type.setText(xml_params["fill_type"])
        except:
            self.run_fill_type.setText("3")
        try:
            audit: tuple[str, str] = xml_audits['CAPTURE']
            captured_by, captured_at = audit
            captured_at = captured_at.replace(
                "T", " ")[:captured_at.index(".")]
            self.run_captured.setText(f"{captured_by} at {captured_at}")
        except:
            self.run_captured.setText("(Not Performed)")
        try:
            audit: tuple[str, str] = xml_audits['PARAMS']
            captured_by, captured_at = audit
            captured_at = captured_at.replace(
                "T", " ")[:captured_at.index(".")]
            self.run_updated.setText(f"{captured_by} at {captured_at}")
        except:
            # if no PARAMS in records, then last updated is time of CAPTURE:
            self.run_updated.setText(self.run_captured.text())
        try:
            audit: tuple[str, str] = xml_audits['ANALYZE']
            captured_by, captured_at = audit
            captured_at = captured_at.replace(
                "T", " ")[:captured_at.index(".")]
            self.run_analyzed.setText(f"{captured_by} at {captured_at}")
        except:
            self.run_analyzed.setText("(Not Performed)")

        run_features = copy.deepcopy(self.default_features)
        value_tags = ["protein_type", "protein_concentration",
                      "", "", "",  # molecular weight, pI mean, pI range
                      "buffer_type", "buffer_concentration",
                      "",  # pH
                      "surfactant_type", "surfactant_concentration",
                      "stabilizer_type", "stabilizer_concentration",
                      "salt_type", "salt_concentration"]
        for x, y in enumerate(value_tags):
            try:
                if y == "":
                    continue
                if y in xml_params.keys():
                    # TODO: quick fix for demo
                    value = xml_params[y]
                    if value == "TWEEN80":
                        value = "Tween-80"
                    if value == "TWEEN20":
                        value = "Tween-20"
                    run_features["Value"][x] = value
            except Exception as e:
                print(e)

        if False:  # self.step == 3:
            # Hide protein and buffer characteristics
            for values in run_features.values():
                del values[7]  # buffer PH
                del values[4]  # protein pI range
                del values[3]  # protein pI mean
                del values[2]  # protein weight
        else:
            # Pull protein and buffer characteristics from database (if available)
            protein = self.parent.ing_ctrl.get_protein_by_name(
                name=xml_params.get("protein_type", None))
            if protein != None:
                if protein.molecular_weight != None:
                    run_features["Value"][2] = protein.molecular_weight
                if protein.pI_mean != None:
                    run_features["Value"][3] = protein.pI_mean
                if protein.pI_range != None:
                    run_features["Value"][4] = protein.pI_range
            buffer = self.parent.ing_ctrl.get_buffer_by_name(
                name=xml_params.get("buffer_type", None))
            if buffer != None:
                if buffer.pH != None:
                    run_features["Value"][7] = buffer.pH

        self.feature_table.setData(run_features)

        # Import most recent analysis
        in_shear_rate = []
        in_viscosity = []
        in_temperature = []
        try:
            base_run_name: str = os.path.basename(self.run_file_run)
            base_run_name = base_run_name[:base_run_name.rfind("_")]
            csv_file = os.path.join(os.path.dirname(
                self.run_file_analyze), f"{base_run_name}_analyze_out.csv")
            zip_filename = os.path.splitext(
                os.path.basename(self.run_file_analyze))[0]
            with SecureOpen(csv_file, "r", zip_filename, insecure=True) as f:
                csv_headers = next(f)
                csv_cols = (0, 2, 4)
                data = np.loadtxt(
                    f.readlines(), delimiter=",", skiprows=0, usecols=csv_cols
                )
            in_shear_rate = data[:, 0]
            in_viscosity = data[:, 1]
            in_temperature = data[:, 2]
        except Exception as e:
            print(e)
        pass_to_models = {"shear_rate": in_shear_rate,
                          "viscosity": in_viscosity}

        # self.profile_shears = [1e2, 1e3, 1e4, 1e5, 15000000] # already set
        self.profile_viscos = []
        has_high_shear_pt = in_shear_rate[-1] > 1e6
        has_curve_fit_est = False

        for shear_rate in self.profile_shears:
            viscosity = np.interp(shear_rate, in_shear_rate, in_viscosity,
                                  left=np.nan)
            self.profile_viscos.append(viscosity)

        if np.any(np.isnan(self.profile_viscos)):
            try:
                # Define the logarithmic function to fit
                def logarithmic_func(x, a, b, c):
                    return a * np.log10(x - c) + b

                # Define bounds for parameters [a, b, c]
                lower_bounds = [-np.inf, -np.inf, -np.inf]
                upper_bounds = [np.inf, np.inf, 99]
                fit_bounds = (lower_bounds, upper_bounds)

                # Perform the curve fit (not including high-shear point)
                initial_guess = (2, 1, 0.5)
                popt, pcov = curve_fit(
                    logarithmic_func,
                    in_shear_rate[:-1] if has_high_shear_pt else in_shear_rate,
                    in_viscosity[:-1] if has_high_shear_pt else in_viscosity,
                    p0=initial_guess,
                    bounds=fit_bounds)
                a_fit, b_fit, c_fit = popt
                has_curve_fit_est = True

                # Generate missing points for the profile using curve fitting
                for i in range(len(self.profile_viscos)):
                    if np.isnan(self.profile_viscos[i]):
                        self.profile_viscos[i] = logarithmic_func(
                            self.profile_shears[i], a_fit, b_fit, c_fit)
                    else:
                        break

            except Exception as e:
                Log.w(
                    "Failed to fit logarithmic curve to viscosity profile. Using entirely interpolated data instead.")

                # Generate missing points for the profile using interpolation
                for i in range(len(self.profile_viscos)):
                    if np.isnan(self.profile_viscos[i]):
                        new_value = np.interp(
                            self.profile_shears[i], in_shear_rate, in_viscosity)
                        self.profile_viscos[i] = new_value
                    else:
                        break

        if has_high_shear_pt:
            self.profile_viscos[-1] = in_viscosity[-1]

        expected_point_count = 13
        if len(in_viscosity) > expected_point_count:
            indices_to_drop = list(range(4, len(in_shear_rate)-2))
            in_shear_rate = [item for i, item in enumerate(
                in_shear_rate) if i not in indices_to_drop]
            in_viscosity = [item for i, item in enumerate(
                in_viscosity) if i not in indices_to_drop]

        minidx = np.argmin(self.profile_viscos)
        maxidx = np.argmax(self.profile_viscos)
        Log.i(
            f"Viscosity profile ranges from {self.profile_viscos[minidx]:.2f} to {self.profile_viscos[maxidx]:.2f} cP.")

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

        self.run_figure.clear()
        self.run_figure_valid = False
        ax = self.run_figure.add_subplot(111)
        ax.set_xlabel("Shear rate (s⁻¹)", fontsize=10)
        ax.set_ylabel("Viscosity (cP)", fontsize=10)
        ax.grid(True, which="both", ls=":")
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
        if len(in_viscosity) > 0:
            xs, ys = smooth_log_interpolate(
                self.profile_shears, self.profile_viscos)
            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(self.calc_limits(yall=in_viscosity))
            ax.plot(self.profile_shears, self.profile_viscos,
                    lw=2.5, color="blue")
            ax.scatter(self.profile_shears, self.profile_viscos,
                       s=40, color="blue", zorder=5)
            ax.plot(in_shear_rate, in_viscosity, "b,")
            self.run_figure_valid = True

            DEBUG = False
            if has_curve_fit_est and DEBUG:
                x_trend = np.logspace(2, 5)
                y_trend = logarithmic_func(
                    x_trend, a_fit, b_fit, c_fit)
                y_trend = [1 if np.isnan(x) else x for x in y_trend]
                ax.plot(x_trend, y_trend, color='blue', linewidth=0.5)
        else:
            ax.text(0.5, 0.5, "Invalid Results",
                    transform=ax.transAxes,
                    ha='center', va='center',
                    bbox=dict(facecolor='yellow', edgecolor='black'))
        ax.set_xscale("log")
        ax.set_yscale("log")
        self.run_canvas.draw()

        avg_temp = np.average(data[:, 2])
        if np.isnan(avg_temp):
            self.run_temperature.setText("(Unknown)")
        else:
            self.run_temperature.setText(f"{avg_temp:2.2f}C")

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

    def model_selected(self, path: str | None):
        self.model_path = path

        if path is None:
            self.select_model_label.clear()
            return

        self.select_model_label.setText(
            path.split('\\')[-1].split('/')[-1].split('.')[0])
