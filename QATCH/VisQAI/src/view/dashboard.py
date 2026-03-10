import os
import sys

import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

try:
    from architecture import Architecture
    from dialogs.database_table_dialog import DatabaseTableDialog
    from src.controller.formulation_controller import FormulationController
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.models.formulation import ViscosityProfile
    from src.utils.metrics import Metrics
    from styles.style_loader import load_stylesheet
    from widgets.evaluation_widget import EvaluationWidget
    from widgets.formulation_config_card_widget import (
        FormulationConfigCard,
    )
    from widgets.generate_sample_widget import GenerateSampleWidget
    from widgets.optimize_widget import OptimizeWidget
    from widgets.placeholder_widget import PlaceholderWidget
    from widgets.prediction_filter_widget import PredictionFilterWidget
    from widgets.reordable_container_widget import ReorderableCardContainer
    from widgets.visualization_panel import VisualizationPanel
    from workers.import_worker import ImportWorker
    from workers.optimization_worker import OptimizationWorker
    from workers.prediction_worker import PredictionThread
    from workers.sample_generation_worker import SampleGenerationWorker
except (ImportError, ModuleNotFoundError):
    from QATCH.common.architecture import Architecture
    from QATCH.common.logger import Logger as Log
    from QATCH.common.userProfiles import UserPreferences, UserProfiles
    from QATCH.core.constants import Constants
    from QATCH.VisQAI.src.controller.formulation_controller import FormulationController
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.models.formulation import ViscosityProfile
    from QATCH.VisQAI.src.utils.metrics import Metrics
    from QATCH.VisQAI.src.view.dialogs.database_table_dialog import DatabaseTableDialog
    from QATCH.VisQAI.src.view.styles.style_loader import load_stylesheet
    from QATCH.VisQAI.src.view.widgets.evaluation_widget import EvaluationWidget
    from QATCH.VisQAI.src.view.widgets.formulation_config_card_widget import (
        FormulationConfigCard,
    )
    from QATCH.VisQAI.src.view.widgets.generate_sample_widget import (
        GenerateSampleWidget,
    )
    from QATCH.VisQAI.src.view.widgets.optimize_widget import OptimizeWidget
    from QATCH.VisQAI.src.view.widgets.placeholder_widget import PlaceholderWidget
    from QATCH.VisQAI.src.view.widgets.prediction_filter_widget import (
        PredictionFilterWidget,
    )
    from QATCH.VisQAI.src.view.widgets.reordable_container_widget import (
        ReorderableCardContainer,
    )
    from QATCH.VisQAI.src.view.widgets.visualization_panel import VisualizationPanel
    from QATCH.VisQAI.src.view.workers.import_worker import ImportWorker
    from QATCH.VisQAI.src.view.workers.optimization_worker import OptimizationWorker
    from QATCH.VisQAI.src.view.workers.prediction_worker import PredictionThread
    from QATCH.VisQAI.src.view.workers.sample_generation_worker import (
        SampleGenerationWorker,
    )

TAG = "[VisQ.AI]"


class DashboardUI(QtWidgets.QWidget):
    INGREDIENT_TYPES = [
        "Protein",
        "Buffer",
        "Surfactant",
        "Stabilizer",
        "Excipient",
        "Salt",
    ]
    INGREDIENT_UNITS = {
        "Protein": "mg/mL",
        "Buffer": "mM",
        "Surfactant": "%w",
        "Stabilizer": "M",
        "Excipient": "mM",
        "Salt": "mM",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ingredients_by_type = {}
        self.selection_mode_active = False
        self._pending_color = None

        self.db = Database(parse_file_key=True)
        self.ing_ctrl = IngredientController(self.db)
        self.form_ctrl = FormulationController(self.db)
        self._load_database_data()

        self.init_ui()
        self.setStyleSheet(load_stylesheet())
        self._batch_queue = []
        self._batch_results = []
        self._is_batch_collecting = False
        self._is_batch_running = False
        self._is_silencing_runs = False
        self._zombie_tasks = []
        self._is_evaluation_mode = False
        self._pending_eval_config = None

    def _load_database_data(self):
        """Initializes DB connection and fetches ingredients for dropdowns."""

        # Fetch lists for each ingredient type
        self.ingredients_by_type["Protein"] = self.ing_ctrl.get_all_proteins()
        self.ingredients_by_type["Buffer"] = self.ing_ctrl.get_all_buffers()
        self.ingredients_by_type["Salt"] = self.ing_ctrl.get_all_salts()
        self.ingredients_by_type["Surfactant"] = self.ing_ctrl.get_all_surfactants()
        self.ingredients_by_type["Stabilizer"] = self.ing_ctrl.get_all_stabilizers()
        self.ingredients_by_type["Excipient"] = self.ing_ctrl.get_all_excipients()

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Create Top Bar (Buttons Only)
        # Note: We do NOT create eval_widget here anymore
        top_bar = self._create_unified_top_bar()
        main_layout.addWidget(top_bar)

        # 2. Splitter Configuration
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet("QSplitter::handle { background-color: #d1d5db; }")

        # --- Left Panel ---
        self.left_widget = QtWidgets.QWidget()
        self.left_widget.setObjectName("leftPanel")
        left_layout = QtWidgets.QVBoxLayout(self.left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # --- OVERLAY WIDGETS (Created NOW, parenting to left_widget) ---

        # 1. Filter Widget
        self.filter_widget = PredictionFilterWidget(
            self.ingredients_by_type, parent=self.left_widget
        )
        self.filter_widget.filter_changed.connect(self.apply_filters)
        self.filter_widget.hide()

        # Evaluation Widget
        self.eval_widget = EvaluationWidget(parent=self)
        self.eval_widget.run_requested.connect(self.run_evaluation_analysis)
        self.eval_widget.clear_requested.connect(self.exit_evaluation_mode)
        self.eval_widget.hide()

        # Generate Widget
        self.generate_widget = GenerateSampleWidget(
            self.ingredients_by_type, parent=self
        )
        self.generate_widget.generate_requested.connect(self.run_sample_generation)
        self.generate_widget.closed.connect(lambda: self.btn_generate.setChecked(False))
        self.generate_widget.resized.connect(self._update_overlay_geometry)
        self.generate_widget.hide()
        # Optimize Widget
        self.optimize_widget = OptimizeWidget(self.ingredients_by_type, parent=self)
        self.optimize_widget.optimize_requested.connect(self.run_optimization)
        self.optimize_widget.closed.connect(lambda: self.btn_optimize.setChecked(False))
        self.optimize_widget.resized.connect(self._update_overlay_geometry)
        self.optimize_widget.hide()
        # --- Content ---
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self.cards_container = ReorderableCardContainer()
        self.cards_container.setObjectName("scrollContent")
        self.cards_layout = self.cards_container.main_layout

        self.scroll_area.setWidget(self.cards_container)
        self.cards_layout.setContentsMargins(15, 15, 15, 15)
        self.cards_layout.setSpacing(10)

        self.placeholder = PlaceholderWidget()
        self.cards_layout.addWidget(self.placeholder)
        self.placeholder.hide()

        left_layout.addWidget(self.scroll_area)
        self._create_fab()

        # Install Event Filter for Resizing Overlays
        self.left_widget.installEventFilter(self)
        splitter.addWidget(self.left_widget)

        # --- Right Panel ---
        self.right_widget = QtWidgets.QWidget()
        self.right_widget.setObjectName("rightPanel")
        self.right_widget.setStyleSheet("background-color: #ffffff;")
        right_layout = QtWidgets.QVBoxLayout(self.right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.viz_panel = VisualizationPanel()
        self.viz_panel.eval_point_clicked.connect(self.on_eval_point_clicked)
        right_layout.addWidget(self.viz_panel)

        splitter.addWidget(self.right_widget)
        splitter.setSizes([450, 700])
        main_layout.addWidget(splitter)

        self.current_task = None
        self.update_placeholder_visibility()

    def _create_unified_top_bar(self):
        """Creates the single unified top toolbar."""
        container = QtWidgets.QWidget()
        container.setObjectName("topBar")
        container.setFixedHeight(50)
        container.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        shadow = QtWidgets.QGraphicsDropShadowEffect(container)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QtGui.QColor(0, 0, 0, 25))
        container.setGraphicsEffect(shadow)

        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(15, 5, 15, 5)
        layout.setSpacing(10)

        # --- Left Side Controls ---
        self.search_bar = QtWidgets.QLineEdit()
        self.search_bar.setObjectName("searchBar")
        self.search_bar.setPlaceholderText("Search...")
        self.search_bar.setClearButtonEnabled(True)
        self.search_bar.addAction(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "search-svgrepo-com.svg",
                )
            ),
            QtWidgets.QLineEdit.ActionPosition.LeadingPosition,
        )
        self.search_bar.textChanged.connect(self.filter_cards)
        layout.addWidget(self.search_bar, stretch=1)

        # Filter Button
        self.btn_filter = QtWidgets.QToolButton()
        self.btn_filter.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "filter-svgrepo-com.svg",
                )
            )
        )
        self.btn_filter.setToolTip("Filter Options")
        self.btn_filter.setCheckable(True)
        self.btn_filter.setFixedSize(32, 32)  # <--- Added to match others
        self.btn_filter.clicked.connect(self.toggle_filter_menu_manual)
        layout.addWidget(self.btn_filter)

        # Select Mode Button
        self.btn_select_mode = QtWidgets.QToolButton()
        self.btn_select_mode.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "select-svgrepo-com.svg",
                )
            )
        )
        self.btn_select_mode.setToolTip("Enter Selection Mode")
        self.btn_select_mode.setCheckable(True)
        self.btn_select_mode.setFixedSize(32, 32)  # <--- Added to match others
        self.btn_select_mode.toggled.connect(self.toggle_selection_mode)
        layout.addWidget(self.btn_select_mode)

        self.btn_select_all = QtWidgets.QToolButton()
        self.btn_select_all.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "select-multiple-svgrepo-com.svg",
                )
            )
        )
        self.btn_select_all.setToolTip("Select All")
        self.btn_select_all.setFixedSize(32, 32)
        self.btn_select_all.clicked.connect(self.select_all_cards)
        self.btn_select_all.setEnabled(False)
        layout.addWidget(self.btn_select_all)

        self.btn_import = QtWidgets.QToolButton()
        self.btn_import.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "import-content-svgrepo-com.svg",
                )
            )
        )
        self.btn_import.setToolTip("Import Data")
        self.btn_import.setFixedSize(32, 32)
        self.btn_import.clicked.connect(self.import_run)
        layout.addWidget(self.btn_import)

        self.btn_export_top = QtWidgets.QToolButton()
        self.btn_export_top.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "export-content-svgrepo-com.svg",
                )
            )
        )
        self.btn_export_top.setToolTip("Export Selected")
        self.btn_export_top.setFixedSize(32, 32)
        self.btn_export_top.clicked.connect(self.export_analysis)
        layout.addWidget(self.btn_export_top)

        # --- NEW BUTTONS (Inline Styles Removed to use theme.qss) ---

        # Generate Sample
        self.btn_generate = QtWidgets.QToolButton()
        self.btn_generate.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "flask-sample-test-svgrepo-com.svg",
                )
            )
        )
        self.btn_generate.setToolTip("Generate Sample")
        self.btn_generate.setFixedSize(32, 32)
        self.btn_generate.setCheckable(True)
        self.btn_generate.clicked.connect(self.handle_generate_sample)
        layout.addWidget(self.btn_generate)

        # Evaluate
        self.btn_evaluate = QtWidgets.QToolButton()
        self.btn_evaluate.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "scorecard-svgrepo-com.svg",
                )
            )
        )
        self.btn_evaluate.setToolTip("Evaluate")
        self.btn_evaluate.setFixedSize(32, 32)
        self.btn_evaluate.setCheckable(True)
        # No setStyleSheet here; theme.qss handles Checked state styling
        self.btn_evaluate.clicked.connect(self.handle_evaluate)
        layout.addWidget(self.btn_evaluate)

        # Hypothesis
        self.btn_hypothesis = QtWidgets.QToolButton()
        self.btn_hypothesis.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "dice-svgrepo-com.svg",
                )
            )
        )
        self.btn_hypothesis.setToolTip("Hypothesis")
        self.btn_hypothesis.setFixedSize(32, 32)
        # No setStyleSheet here
        self.btn_hypothesis.clicked.connect(self.handle_hypothesis)
        layout.addWidget(self.btn_hypothesis)

        # Optimize
        self.btn_optimize = QtWidgets.QToolButton()
        self.btn_optimize.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "optimize-svgrepo-com.svg",
                )
            )
        )
        self.btn_optimize.setToolTip("Optimize")
        self.btn_optimize.setFixedSize(32, 32)
        self.btn_optimize.setCheckable(True)
        self.btn_optimize.clicked.connect(self.handle_optimize)
        layout.addWidget(self.btn_optimize)

        # --- Separator ---
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        line.setFixedHeight(20)
        layout.addWidget(line)

        # Run & Delete
        self.btn_run_top = QtWidgets.QToolButton()
        self.btn_run_top.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "play-svgrepo-com.svg",
                )
            )
        )
        self.btn_run_top.setToolTip("Run Inference (Selected or Open Card)")
        self.btn_run_top.setFixedSize(32, 32)
        self.btn_run_top.clicked.connect(self.run_analysis)
        layout.addWidget(self.btn_run_top)

        self.btn_delete_top = QtWidgets.QToolButton()
        self.btn_delete_top.setObjectName("btnDelete")
        self.btn_delete_top.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "delete-2-svgrepo-com.svg",
                )
            )
        )
        self.btn_delete_top.setToolTip("Delete (Selected or Open Card)")
        self.btn_delete_top.setFixedSize(32, 32)
        self.btn_delete_top.clicked.connect(self.delete_analysis)
        layout.addWidget(self.btn_delete_top)

        # --- Spacer ---
        layout.addStretch(1)
        self.btn_right_options = QtWidgets.QToolButton()
        self.btn_right_options.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "three-dots-svgrepo-com.svg",
                )
            )
        )
        self.btn_right_options.setToolTip("More Options")
        self.btn_right_options.setFixedSize(32, 32)
        # Options
        self.btn_right_options.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.btn_right_options.setStyleSheet(
            "QToolButton::menu-indicator { image: none; }"
        )  # Hide small triangle

        self.top_options_menu = QtWidgets.QMenu(self.btn_right_options)

        # Action: View Ingredients
        act_view_ing = self.top_options_menu.addAction("View Ingredients")
        act_view_ing.triggered.connect(self.show_ingredients_view)

        # Action: View Formulations
        act_view_form = self.top_options_menu.addAction("View Formulations")
        act_view_form.triggered.connect(self.show_formulations_view)

        self.btn_right_options.setMenu(self.top_options_menu)

        layout.addWidget(self.btn_right_options)

        return container

    def show_ingredients_view(self):
        """Opens a dialog listing all unique ingredients with detailed columns."""
        # 1. Define Headers
        headers = [
            "Type",
            "Name",
            "ID",
            "Class Type",
            "C-Class",
            "MW (Da)",
            "pH",
            "pI Mean",
            "pI Range",
            "kP",
            "HCI",
        ]
        rows = []

        # 2. Iterate and Populate
        for ing_type, ingredients in self.ingredients_by_type.items():
            for ing in ingredients:
                row = []

                # Basic Info
                row.append(ing_type)
                row.append(getattr(ing, "name", "Unknown"))
                row.append(str(getattr(ing, "id", "")))

                # --- Detailed Attributes with Nested Class_Type Check ---

                # Defaults
                c_type_val = "-"
                c_class = "-"
                kp_val = "-"
                hci_val = "-"

                # Check for nested class_type object (common in Proteins)
                if hasattr(ing, "class_type") and ing.class_type:
                    ct = ing.class_type
                    # Access nested attributes safely
                    c_type_val = str(getattr(ct, "value", getattr(ct, "name", str(ct))))
                    c_class = str(getattr(ct, "c_class", "-"))
                    kp_val = str(getattr(ct, "kP", "-"))
                    hci_val = str(getattr(ct, "hci", "-"))
                elif hasattr(ing, "group"):
                    # Fallback for Surfactants/etc that might use 'group'
                    c_type_val = str(ing.group)

                row.append(c_type_val)  # Class Type
                row.append(c_class)  # C-Class

                # Standard Attributes
                row.append(str(getattr(ing, "molecular_weight", "-")))
                row.append(str(getattr(ing, "pH", "-")))
                row.append(str(getattr(ing, "pI_mean", "-")))
                row.append(str(getattr(ing, "pI_range", "-")))

                # Nested Attributes
                row.append(kp_val)  # kP
                row.append(hci_val)  # HCI

                rows.append(row)

        # 3. Show Dialog
        dlg = DatabaseTableDialog("Ingredient Database", headers, rows, self)
        dlg.resize(1200, 600)
        dlg.exec_()

    def show_formulations_view(self):
        """Opens a dialog listing all formulations with delete support, ICL toggles, and CSV export."""
        try:
            # 1. Fetch Formulations
            formulations = self.form_ctrl.get_all_formulations()

            # --- Handlers ---
            def delete_handler(f_id):
                target_f = next(
                    (f for f in formulations if str(f.id) == str(f_id)), None
                )
                if not target_f:
                    return False

                if not (target_f.name and target_f.signature):
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Deletion Denied",
                        "Only imported formulations can be deleted.",
                    )
                    return False

                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Confirm Delete",
                    f"Are you sure you want to delete formulation '{target_f.name}'?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                )
                if reply == QtWidgets.QMessageBox.Yes:
                    try:
                        self.form_ctrl.delete_formulation_by_id(target_f.id)
                        return True
                    except Exception as e:
                        QtWidgets.QMessageBox.critical(
                            self, "Error", f"Could not delete formulation:\n{e}"
                        )
                return False

            def icl_toggled(f_id, state):
                """Updates the ICL flag safely using the new Controller method."""
                try:
                    # 1. Update DB safely (Preserves ID)
                    success = self.form_ctrl.update_formulation_metadata(
                        int(f_id), icl=state
                    )

                    if success:
                        # 2. Update local list object
                        local_f = next(
                            (f for f in formulations if str(f.id) == str(f_id)), None
                        )
                        if local_f:
                            local_f.icl = state

                        # 3. Synchronize Active UI Cards
                        for i in range(self.cards_layout.count()):
                            item = self.cards_layout.itemAt(i)
                            widget = item.widget()
                            if isinstance(widget, FormulationConfigCard):
                                card_id = getattr(widget.formulation, "id", None)
                                if card_id is not None and str(card_id) == str(f_id):
                                    widget.set_icl_usage(state, save_db=False)
                                    break
                    else:
                        print(f"Warning: Could not update ICL for ID {f_id}")

                except Exception as e:
                    print(f"Error updating ICL state: {e}")

            def export_handler():
                """Exports the entire database to a CSV file."""
                try:
                    df = self.form_ctrl.get_all_as_dataframe(encoded=False)

                    if df is None or df.empty:
                        QtWidgets.QMessageBox.information(
                            self,
                            "Export Info",
                            "The database is empty. Nothing to export.",
                        )
                        return
                    path, _ = QtWidgets.QFileDialog.getSaveFileName(
                        self,
                        "Export Database to CSV",
                        "formulations_database_export.csv",
                        "CSV Files (*.csv)",
                    )

                    if path:
                        df.to_csv(path, index=False)
                        QtWidgets.QMessageBox.information(
                            self,
                            "Success",
                            f"Database successfully exported to:\n{path}",
                        )

                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        self, "Export Error", f"Failed to export data:\n{e}"
                    )

            headers = [
                "ID",
                "Name",
                "Temp (°C)",
                "ICL",
                "Last Model",
                "Protein Type",
                "Class",
                "MW (Da)",
                "pI Mean",
                "pI Range",
                "Conc (mg/mL)",
                "Buffer",
                "pH",
                "Conc (mM)",
                "Stabilizer",
                "Conc (mM)",
                "Surfactant",
                "Conc (%)",
                "Salt",
                "Conc (mM)",
                "Excipient",
                "Conc (mM)",
                "η @ 100",
                "η @ 1k",
                "η @ 10k",
                "η @ 100k",
                "η @ 15m",
            ]

            rows = []
            shear_rates = [100, 1000, 10000, 100000, 15000000]

            for f in formulations:
                row = []
                # Basic Info
                row.append(str(getattr(f, "id", "") or ""))
                row.append(str(f.name or "Unnamed"))
                row.append(str(f.temperature if f.temperature is not None else "25.0"))
                row.append(str(getattr(f, "icl", True)))
                row.append(str(getattr(f, "last_model", "-") or "-"))

                # Protein
                if hasattr(f, "protein") and f.protein and f.protein.ingredient:
                    p = f.protein
                    ing = p.ingredient
                    class_name = "-"
                    if hasattr(ing, "class_type") and ing.class_type:
                        class_name = str(
                            getattr(
                                ing.class_type,
                                "value",
                                getattr(ing.class_type, "name", "-"),
                            )
                        )
                    row.extend(
                        [
                            str(ing.name or "-"),
                            class_name,
                            str(getattr(ing, "molecular_weight", "")),
                            str(getattr(ing, "pI_mean", "")),
                            str(getattr(ing, "pI_range", "")),
                            str(p.concentration),
                        ]
                    )
                else:
                    row.extend(["-", "-", "-", "-", "-", "-"])

                # Buffer
                if hasattr(f, "buffer") and f.buffer and f.buffer.ingredient:
                    b = f.buffer
                    ing = b.ingredient
                    row.extend(
                        [
                            str(ing.name or "-"),
                            str(getattr(ing, "pH", "")),
                            str(b.concentration),
                        ]
                    )
                else:
                    row.extend(["-", "-", "-"])

                # Others (Stabilizer, Surfactant, Salt, Excipient)
                def add_simple_comp(comp_attr):
                    if hasattr(f, comp_attr):
                        c = getattr(f, comp_attr)
                        if c and c.ingredient:
                            row.extend(
                                [str(c.ingredient.name or "-"), str(c.concentration)]
                            )
                            return
                    row.extend(["-", "-"])

                add_simple_comp("stabilizer")
                add_simple_comp("surfactant")
                add_simple_comp("salt")
                add_simple_comp("excipient")

                # Viscosity
                if hasattr(f, "viscosity_profile") and f.viscosity_profile:
                    vp = f.viscosity_profile
                    for sr in shear_rates:
                        try:
                            val = vp.get_viscosity(sr)
                            row.append(f"{val:.2f}")
                        except Exception:
                            row.append("-")
                else:
                    row.extend(["-"] * 5)

                rows.append(row)

            dlg = DatabaseTableDialog(
                "Formulation Database",
                headers,
                rows,
                self,
                delete_callback=delete_handler,
                check_col_idx=3,
                check_callback=icl_toggled,
                export_callback=export_handler,
            )
            dlg.resize(1500, 600)
            dlg.exec_()

        except Exception as e:
            import traceback

            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to load formulations:\n{e}"
            )

    def _on_card_selection_changed(self):
        """
        Called whenever a card is selected/deselected.
        Auto-exits selection mode if the last item is deselected.
        """
        if not self.selection_mode_active:
            return

        selected_count = 0
        for i in range(self.cards_layout.count()):
            widget = self.cards_layout.itemAt(i).widget()
            if isinstance(widget, FormulationConfigCard) and widget.is_selected:
                selected_count += 1

        if selected_count == 0:
            self.btn_select_mode.setChecked(False)

    def handle_generate_sample(self):
        """Toggles the Sample Generation overlay widget."""
        # Ensure eval widget is closed to prevent overlapping
        if self._is_evaluation_mode:
            self.exit_evaluation_mode()

        if self.generate_widget.isVisible():
            self.generate_widget.hide()
            self.btn_generate.setChecked(False)
        else:
            self.generate_widget.show()
            self.generate_widget.raise_()
            self.btn_generate.setChecked(True)
            self._update_overlay_geometry()

    def handle_evaluate(self):
        """
        First click: enters evaluation mode, shows the eval widget menu, and
        keeps the button visually active (checked/colored) for the duration.

        Subsequent clicks while eval mode is active: toggle the menu open/closed
        WITHOUT exiting eval mode — the button stays active-colored either way.

        Eval mode is only exited by the Clear button on the eval widget, which
        fires clear_requested -> exit_evaluation_mode().
        """
        if not self._is_evaluation_mode:
            # ── Enter eval mode ──────────────────────────────────────────────
            self._is_evaluation_mode = True
            # Keep the button forced-checked so its QSS active/checked color
            # stays on regardless of future clicks.
            self.btn_evaluate.setChecked(True)

            # Hide the filter widget if it's open
            if self.filter_widget.isVisible():
                self.filter_widget.hide()
                self.btn_filter.setChecked(False)

            # Show only cards that have measured data
            visible_count = 0
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, FormulationConfigCard):
                    if widget.is_measured:
                        widget.show()
                        visible_count += 1
                    else:
                        widget.hide()

            self.viz_panel.set_plot_title(
                f"Evaluation Mode: {visible_count} Datasets Ready"
            )

            # Show the eval menu
            self.eval_widget.show()
            self.eval_widget.raise_()
            self._update_overlay_geometry()

        else:
            # ── Eval mode already active — just toggle the menu ──────────────
            # Qt toggled the checked state on click; force it back on so the
            # button stays colored as long as eval mode is active.
            self.btn_evaluate.setChecked(True)

            if self.eval_widget.isVisible():
                self.eval_widget.hide()
            else:
                self.eval_widget.show()
                self.eval_widget.raise_()
                self._update_overlay_geometry()

        self.update_placeholder_visibility()

    def handle_hypothesis(self):
        """Handler for the Hypothesis button."""
        # TODO: Replace with: dialog = HypothesisInputDialog(self); dialog.exec_()
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Hypothesis Testing", "Enter hypothesis name or ID:"
        )
        if ok and text:
            print(f"Hypothesis '{text}' initialized.")

    def handle_optimize(self):
        """Toggles the Optimize overlay widget."""
        if self.optimize_widget.isVisible():
            self.optimize_widget.hide()
            self.btn_optimize.setChecked(False)
        else:
            self.optimize_widget.show()
            self.optimize_widget.raise_()
            self.btn_optimize.setChecked(True)
            self._update_overlay_geometry()

    def _create_fab(self):
        """Creates the Floating Action Button for Adding Cards."""
        self.btn_add_fab = QtWidgets.QPushButton("+", self.left_widget)
        self.btn_add_fab.setToolTip("New Prediction")
        self.btn_add_fab.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_fab.resize(50, 50)

        # Shadow Effect
        shadow = QtWidgets.QGraphicsDropShadowEffect(self.btn_add_fab)
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QtGui.QColor(0, 0, 0, 80))
        self.btn_add_fab.setGraphicsEffect(shadow)
        self.btn_add_fab.setObjectName("fabAdd")
        self.btn_add_fab.clicked.connect(lambda: self.add_prediction_card(None))
        self.btn_add_fab.show()

    def eventFilter(self, source, event):
        """
        Handle resize events for the left panel to resize the filter overlay
        and reposition the Floating Action Button (FAB).
        """
        if source == self.left_widget and event.type() == QtCore.QEvent.Type.Resize:
            self._update_filter_geometry()
            fab_size = self.btn_add_fab.size()
            margin = 20
            x = self.left_widget.width() - fab_size.width() - margin
            y = self.left_widget.height() - fab_size.height() - margin
            self.btn_add_fab.move(x, y)
            self.btn_add_fab.raise_()

        return super().eventFilter(source, event)

    def update_placeholder_visibility(self):
        """
        Updates the placeholder state based on visible cards.
        """
        visible_count = 0
        total_cards = 0

        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()

            if isinstance(widget, FormulationConfigCard):
                total_cards += 1
                if not widget.isHidden():
                    visible_count += 1

        if visible_count > 0:
            self.placeholder.hide()
        else:
            self.placeholder.show()

            if total_cards == 0:
                self.placeholder.lbl_text.setText(
                    "No data yet.\nClick the + button to add new data."
                )
            else:
                self.placeholder.lbl_text.setText(
                    "No results found.\nTry adjusting your filters or search."
                )

    def check_evaluation_eligibility(self):
        """
        Checks if any card has measured data enabled.
        Enables/Disables the Evaluate button accordingly.
        """
        has_measured_data = False

        # Iterate over all cards in the layout
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()

            if isinstance(widget, FormulationConfigCard):
                # FIX: Check the internal boolean property instead of a checkbox
                if widget.is_measured:
                    has_measured_data = True
                    break

        # Update Button State
        if hasattr(self, "btn_evaluate"):
            self.btn_evaluate.setEnabled(has_measured_data)

            # If we just disabled the button and it was currently active (checked),
            # we must force exit the evaluation mode to restore the UI.
            if not has_measured_data and self.btn_evaluate.isChecked():
                self.btn_evaluate.setChecked(False)
                self.exit_evaluation_mode()

    def _is_filter_default(self, filters):
        """Checks if the provided filter dict matches the default state."""
        if not filters["show_measured"] or not filters["show_predicted"]:
            return False
        if filters["model_text"] != "":
            return False
        if filters["temp_min"] != 0 or filters["temp_max"] != 100:
            return False
        if filters["ingredients"]:
            for selected_list in filters["ingredients"].values():
                if selected_list:
                    return False
        return True

    def exit_evaluation_mode(self):
        """Restores UI to normal state and re-runs prediction for the open card."""
        self._is_evaluation_mode = False
        self.eval_widget.hide()
        self.btn_evaluate.setChecked(False)

        # Repopulate regular plot and track which card is currently expanded
        results = []
        expanded_card = None
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, FormulationConfigCard):
                widget.show()
                if widget.is_expanded:
                    expanded_card = widget
                if hasattr(widget, "last_results") and widget.last_results:
                    data = widget.last_results.copy()
                    data["config_name"] = widget.name_input.text()
                    results.append(data)

        self.viz_panel.set_data(results)
        self.viz_panel.set_plot_title("")
        self.update_placeholder_visibility()

        # Re-run prediction for the currently expanded card so the plot reflects
        # its inference curve (not the evaluation overlay that was just cleared).
        if expanded_card is not None:
            self.running_card = expanded_card
            expanded_card.emit_run_request()

    def on_eval_point_clicked(self, point_data):
        """Handles clicks on the parity plot scatter points."""
        card = point_data.get("card")
        if not card:
            return

        card.show()
        if not card.is_expanded:
            card.expand_silent()  # <-- was: card.toggle_content()

        self._scroll_to_card(card)

    def run_evaluation_analysis(self, config):
        """
        Step 1: Batch-predict all visible measured cards.
        Step 2: After batch completes, compute and display the selected metric.
        """
        # Collect all visible measured cards
        target_cards = []
        for i in range(self.cards_layout.count()):
            card = self.cards_layout.itemAt(i).widget()
            if not isinstance(card, FormulationConfigCard) or card.isHidden():
                continue
            if not card.is_measured:
                continue
            target_cards.append(card)

        if not target_cards:
            QtWidgets.QMessageBox.warning(
                self,
                "Evaluation Failed",
                "No cards with measured data are currently visible.",
            )
            return

        # Store config for use after batch completes
        self._pending_eval_config = config

        # Collect prediction configs via the existing batch-collection mechanism
        self._batch_queue = []
        self._batch_results = []
        self._is_batch_collecting = True
        for card in target_cards:
            card.emit_run_request()
        self._is_batch_collecting = False

        if self._batch_queue:
            self._is_batch_running = True
            metric_name = config.get("metric_name", config.get("metric", ""))
            self.viz_panel.set_plot_title(
                f"Running predictions for evaluation ({len(self._batch_queue)} cards)..."
            )
            self.viz_panel.show_loading()
            self._process_next_in_batch()
        else:
            # Cards may have emitted nothing (e.g., incomplete configs); try computing directly
            self._compute_evaluation(config)

    def _compute_evaluation(self, config):
        """Performs metric computation and updates the visualization panel."""

        metric_key = config.get("metric", "rmse")
        metric_name = config.get("metric_name", metric_key)
        shear_min = config.get("shear_min", 100)
        shear_max = config.get("shear_max", 15000000)
        log_visc = config.get("log_viscosity", False)

        if metric_key == "true_vs_pred":
            parity_data = []
            for i in range(self.cards_layout.count()):
                card = self.cards_layout.itemAt(i).widget()
                if not isinstance(card, FormulationConfigCard) or card.isHidden():
                    continue
                if not hasattr(card, "last_results") or not card.last_results:
                    continue
                data = card.last_results
                if (
                    "measured_y" not in data
                    or "y" not in data
                    or data["measured_y"] is None
                ):
                    continue

                shear = np.array(data["x"])
                y_pred = np.array(data["y"])
                y_true = np.array(data["measured_y"])

                mask = (shear >= shear_min) & (shear <= shear_max)
                if not np.any(mask):
                    continue

                points = [
                    {
                        "shear": float(s),
                        "true": float(yt),
                        "pred": float(yp),
                        "card": card,
                    }
                    for s, yt, yp in zip(shear[mask], y_true[mask], y_pred[mask])
                ]
                parity_data.append(
                    {
                        "config_name": card.name_input.text(),
                        "color": card.plot_color,
                        "points": points,
                    }
                )

            if not parity_data:
                QtWidgets.QMessageBox.warning(
                    self, "Evaluation Failed", "No valid data found in range."
                )
                return

            self.viz_panel.set_parity_data(parity_data, log_visc)
            self.viz_panel.set_plot_title("Evaluation: True vs. Predicted Viscosity")

        else:

            metrics_engine = Metrics()
            results_to_plot = []
            scores = []

            for i in range(self.cards_layout.count()):
                card = self.cards_layout.itemAt(i).widget()
                if not isinstance(card, FormulationConfigCard) or card.isHidden():
                    continue
                if not hasattr(card, "last_results") or not card.last_results:
                    continue
                data = card.last_results
                if (
                    "measured_y" not in data
                    or "y" not in data
                    or data["measured_y"] is None
                ):
                    continue

                shear = np.array(data["x"])
                y_pred = np.array(data["y"])
                y_true = np.array(data["measured_y"])

                mask = (shear >= shear_min) & (shear <= shear_max)
                if not np.any(mask):
                    continue

                df = pd.DataFrame(
                    {
                        "actual": y_true[mask],
                        "predicted": y_pred[mask],
                        "residual": y_true[mask] - y_pred[mask],
                        "abs_error": np.abs(y_true[mask] - y_pred[mask]),
                        "percentage_error": np.abs(
                            (y_true[mask] - y_pred[mask]) / y_true[mask]
                        )
                        * 100,
                    }
                )

                try:
                    score = metrics_engine.metrics[metric_key](df)
                    scores.append(score)
                    data_copy = data.copy()
                    data_copy["config_name"] = (
                        f"{card.name_input.text()} ({metric_name}={score:.2f})"
                    )
                    results_to_plot.append(data_copy)
                except Exception as e:
                    print(
                        f"Failed to calculate {metric_key} for {card.name_input.text()}: {e}"
                    )

            if not results_to_plot:
                QtWidgets.QMessageBox.warning(
                    self, "Evaluation Failed", "No valid data found in range."
                )
                return

            self.viz_panel.set_data(results_to_plot)
            if scores:
                avg = sum(scores) / len(scores)
                self.viz_panel.set_plot_title(
                    f"Evaluation Results: Avg {metric_name} = {avg:.4f}"
                )
            else:
                self.viz_panel.set_plot_title(f"Evaluation Results: {metric_name}")

    def run_sample_generation(self, num_samples, model_file, constraints_data):
        self.generate_widget.hide()
        self.btn_generate.setChecked(False)

        self.progress_dialog = QtWidgets.QProgressDialog(
            "Starting generation...", "Cancel", 0, 100, self
        )
        self.progress_dialog.setWindowTitle("Generating Samples")
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)

        worker = SampleGenerationWorker(
            num_samples=num_samples,
            model_file=model_file,
            constraints_data=constraints_data,
            # parent=self,
        )

        # --- THE FIX: Anchor the thread so it avoids garbage collection ---
        if not hasattr(self, "_active_workers"):
            self._active_workers = []
        self._active_workers.append(worker)

        worker.progress_update.connect(self._on_generation_progress)
        worker.generation_complete.connect(self._on_generation_complete)
        worker.generation_error.connect(self._on_generation_error)

        # Safely detach and delete the worker when it naturally finishes
        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))
        self.progress_dialog.canceled.connect(worker.stop)

        worker.start()

    def run_optimization(self, model_file, targets, constraints_data):
        """Launches the OptimizationWorker, routing progress into the viz panel
        loading overlay — identical to how predictions display progress."""
        self.optimize_widget.hide()
        self.btn_optimize.setChecked(False)

        maxiter = self.optimize_widget.spin_maxiter.value()

        # Snapshot the current plot state so Cancel can restore it exactly.
        self._pre_opt_data_series = list(getattr(self.viz_panel, "data_series", []))
        self._pre_opt_plot_title = getattr(self, "_last_plot_title", "")

        self.viz_panel.set_plot_title("Optimizing formulation…")

        worker = OptimizationWorker(
            model_file=model_file,
            targets=targets,
            constraints_data=constraints_data,
            maxiter=maxiter,
        )

        if not hasattr(self, "_active_workers"):
            self._active_workers = []
        self._active_workers.append(worker)
        self._current_opt_worker = worker

        # Pass a cancel callback that stops the worker AND restores the plot.
        self.viz_panel.show_loading(
            cancel_callback=lambda: self._cancel_optimization(worker)
        )

        worker.progress_update.connect(self._on_optimization_progress)
        worker.optimization_complete.connect(self._on_optimization_complete)
        worker.optimization_error.connect(self._on_optimization_error)
        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))

        worker.start()

    def _on_optimization_progress(self, value, text):
        """Drive the viz-panel loading overlay with real optimizer progress."""
        # Stop the free-running animation so we can show real percentage
        if (
            hasattr(self.viz_panel, "anim_timer")
            and self.viz_panel.anim_timer.isActive()
        ):
            self.viz_panel.anim_timer.stop()
        if hasattr(self.viz_panel, "progress_bar"):
            self.viz_panel.progress_bar.setValue(value)
        if hasattr(self.viz_panel, "loading_label"):
            self.viz_panel.loading_label.setText(text)
        self.viz_panel.set_plot_title(f"Optimizing… {value}%")

    def _on_optimization_complete(self, card_data):
        """Add the optimized formulation card and update the plot."""
        # Hide the overlay and reset the label for the next prediction run
        self.viz_panel.hide_loading()
        if hasattr(self.viz_panel, "loading_label"):
            self.viz_panel.loading_label.setText("Calculating…")

        # Add the card; lock its fields to prevent accidental edits
        card = self.add_prediction_card(card_data)
        if hasattr(card, "set_optimized_state"):
            card.set_optimized_state(True)

        # Show the estimated viscosity profile immediately
        est = card_data.get("estimated_profile", {})
        shear = est.get("shear_rates", [])
        viscs = est.get("viscosities", [])
        if shear and viscs:
            data_package = {
                "x": shear,
                "y": viscs,
                "config_name": card_data.get("name", "Optimized Formulation"),
                "measured": False,
            }
            if hasattr(card, "set_results"):
                card.set_results(data_package)
            name = card_data.get("name", "Optimized Formulation")
            self._last_plot_title = name
            self.viz_panel.set_plot_title(name)
            self.viz_panel.set_data(data_package)

    def _on_optimization_error(self, error_msg):
        self.viz_panel.hide_loading()
        if hasattr(self.viz_panel, "loading_label"):
            self.viz_panel.loading_label.setText("Calculating…")
        self.viz_panel.set_plot_title("Optimization failed")
        QtWidgets.QMessageBox.critical(
            self,
            "Optimization Failed",
            f"The optimizer encountered an error:\n\n{error_msg}",
        )

    def _cancel_optimization(self, worker):
        """Stop the running optimizer and restore the plot to its pre-run state."""
        worker.stop()
        self.viz_panel.hide_loading()
        if hasattr(self.viz_panel, "loading_label"):
            self.viz_panel.loading_label.setText("Calculating…")

        # Restore whatever was on the plot before optimization started
        prev_series = getattr(self, "_pre_opt_data_series", [])
        prev_title = getattr(self, "_pre_opt_plot_title", "")
        if prev_series:
            self.viz_panel.set_plot_title(prev_title)
            self.viz_panel.set_data(list(prev_series))
        else:
            # No previous data — clear to placeholder state
            self.viz_panel.set_plot_title("")
            self.viz_panel.set_data([])

    def _cleanup_worker(self, worker):
        if not worker.wait(5000):  # 5 second timeout
            import warnings

            warnings.warn("Worker thread did not finish in time during cleanup.")
        if hasattr(self, "_active_workers") and worker in self._active_workers:
            self._active_workers.remove(worker)

    def _on_generation_progress(self, value, text):
        if hasattr(self, "progress_dialog"):
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(text)

    def _on_generation_complete(self, generated_cards_data):
        if hasattr(self, "progress_dialog"):
            self.progress_dialog.close()

        if not generated_cards_data:
            return

        self._is_batch_collecting = True
        self._batch_queue = []
        self._batch_results = []

        for card_data in generated_cards_data:
            self.add_prediction_card(card_data)

        self._is_batch_collecting = False

        if self._batch_queue:
            self._is_batch_running = True
            self._process_next_in_batch()

        QtWidgets.QMessageBox.information(
            self,
            "Success",
            f"Successfully generated {len(generated_cards_data)} samples.",
        )

    def _on_generation_error(self, error_msg):
        if hasattr(self, "progress_dialog"):
            self.progress_dialog.close()

        QtWidgets.QMessageBox.critical(
            self, "Generation Error", f"Failed to generate samples:\n{error_msg}"
        )

    def apply_filters(self, filter_data):
        """Iterates over cards and toggles visibility based on match."""
        search_text = self.search_bar.text().lower()

        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()

            if widget and isinstance(widget, FormulationConfigCard):
                matches_complex = widget.matches_filter(filter_data)
                matches_search = True
                if search_text:
                    matches_search = search_text in widget.get_searchable_text()

                if matches_complex and matches_search:
                    widget.show()
                else:
                    widget.hide()

        self.update_placeholder_visibility()

        # Check if filters are currently default
        self.update_placeholder_visibility()
        is_default = (
            filter_data["show_measured"]
            and filter_data["show_predicted"]
            and filter_data["model_text"] == ""
            and filter_data["temp_min"] == 0
            and filter_data["temp_max"] == 100
            and not any(filter_data["ingredients"].values())
        )

        self.btn_filter.setProperty("active", not is_default)
        self.btn_filter.style().unpolish(self.btn_filter)
        self.btn_filter.style().polish(self.btn_filter)
        self.filter_widget.hide()

    def toggle_filter_menu_manual(self):
        """Toggles filter menu visibility on click."""
        if self.filter_widget.isVisible():
            self.filter_widget.hide()
        else:
            self.filter_widget.show()
            self.filter_widget.raise_()
            self._update_filter_geometry()

    def _update_filter_geometry(self):
        """Positions the filter widget directly below the top bar (which is outside the panel now)."""
        if self.filter_widget.isVisible():
            # Coordinate 0,0 is now the top of the left_widget (content area)
            self.filter_widget.setGeometry(
                0, 0, self.left_widget.width(), self.filter_widget.sizeHint().height()
            )

    def run_analysis(self):
        """
        Modified to support Batch Analysis of selected cards.
        """
        target_cards = self.get_target_cards()
        if not target_cards:
            return

        # 1. Initialize Batch State
        self._batch_queue = []
        self._batch_results = []
        self._is_batch_collecting = True  # Start listening for run requests

        # 2. Trigger requests from cards (which call run_prediction via signals)
        # Because _is_batch_collecting is True, run_prediction will Queue them instead of running.
        for card in target_cards:
            card.emit_run_request()

        self._is_batch_collecting = False  # Stop listening

        # 3. Start processing the queue
        if self._batch_queue:
            self._is_batch_running = True
            self.viz_panel.set_plot_title("Initializing Batch Analysis...")
            self.viz_panel.show_loading()
            self._process_next_in_batch()
        else:
            QtWidgets.QMessageBox.warning(
                self, "Error", "Failed to collect configuration data."
            )

    def _process_next_in_batch(self):
        if not self._batch_queue:
            self._is_batch_running = False
            self.viz_panel.hide_loading()

            if self._is_evaluation_mode and self._pending_eval_config is not None:
                # Batch was for evaluation — now compute the metric
                config = self._pending_eval_config
                self._pending_eval_config = None
                self._compute_evaluation(config)  # <-- NEW BRANCH
            else:
                count = len(self._batch_results)
                self.viz_panel.set_plot_title(f"Analysis Results ({count} Profiles)")
                self.viz_panel.set_data(self._batch_results)
            return

        card, config = self._batch_queue.pop(0)
        self.running_card = card
        self.viz_panel.set_plot_title(f"Calculating: {config.get('name')}...")
        QtCore.QTimer.singleShot(0, lambda: self.run_prediction(config))

    def delete_analysis(self):
        """Deletes Selected cards (if any) or the currently Open card and updates the plot."""
        target_cards = []

        # 1. Identify Target Cards
        if self.selection_mode_active:
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, FormulationConfigCard) and widget.is_selected:
                    target_cards.append(widget)
        else:
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, FormulationConfigCard) and widget.is_expanded:
                    target_cards.append(widget)
                    break

        if not target_cards:
            return

        # 2. Confirm Deletion
        count = len(target_cards)
        msg = f"Are you sure you want to delete {count} prediction(s)?"
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Delete",
            msg,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            # --- UPDATE PLOT: Remove data associated with deleted cards ---

            # Gather references to the data objects and names from the cards being deleted
            results_to_remove = []
            names_to_remove = []

            for card in target_cards:
                # If the card has a stored result object, we target that specific object
                if hasattr(card, "last_results") and card.last_results:
                    results_to_remove.append(card.last_results)
                # We also track the name as a fallback
                names_to_remove.append(card.name_input.text())

            # Filter the current data series in the Visualization Panel
            current_series = self.viz_panel.data_series
            new_series = []

            for data_pkg in current_series:
                # FIX: Use 'is' identity check to avoid numpy array comparison errors
                # Replaces: if data_pkg in results_to_remove:
                if any(data_pkg is res for res in results_to_remove):
                    continue

                # Exclude if the name matches (fallback if object identity fails)
                if data_pkg.get("config_name") in names_to_remove:
                    continue

                new_series.append(data_pkg)

            # Update the visualization with the filtered list
            self.viz_panel.set_data(new_series)

            # --- REMOVE WIDGETS ---
            for card in target_cards:
                self.remove_card(card)

            # Cleanup selection state
            if self.selection_mode_active:
                self.btn_select_mode.setChecked(False)
            elif self.cards_layout.count() > 0:
                pass

    def toggle_selection_mode(self, active):
        """Toggles selection mode and handles FAB state."""
        self.selection_mode_active = active

        # Disable Add FAB and Import when selecting
        self.btn_add_fab.setEnabled(not active)
        self.btn_import.setEnabled(not active)
        self.btn_select_all.setEnabled(active)

        # Dim the FAB visually if disabled
        opacity = 0.5 if active else 1.0
        effect = QtWidgets.QGraphicsOpacityEffect(self.btn_add_fab)
        effect.setOpacity(opacity)
        self.btn_add_fab.setGraphicsEffect(effect)

        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()
            if widget and isinstance(widget, FormulationConfigCard):
                if hasattr(widget, "set_selectable"):
                    widget.set_selectable(active)

    def select_all_cards(self):
        """Selects all visible cards. If all are already selected, deselects all."""
        visible_cards = []
        for i in range(self.cards_layout.count()):
            w = self.cards_layout.itemAt(i).widget()
            if isinstance(w, FormulationConfigCard) and not w.isHidden():
                visible_cards.append(w)

        if not visible_cards:
            return

        selected_visible = [w for w in visible_cards if w.is_selected]

        # If everything visible is already selected, deselect them all
        should_select = len(selected_visible) < len(visible_cards)

        for widget in visible_cards:
            if widget.is_selected != should_select:
                widget.toggle_selection()

    def filter_cards(self, text):
        """Filters cards based on search text and updates placeholder."""
        search_text = text.lower().strip()
        search_tokens = search_text.split()

        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()

            if widget and isinstance(widget, FormulationConfigCard):
                # Get content
                card_content = widget.get_searchable_text()

                # Check Search Match
                matches_search = True
                if search_tokens:
                    for token in search_tokens:
                        if token not in card_content:
                            matches_search = False
                            break

                # Check Filter Widget Match (if active)
                matches_filter = True
                if hasattr(self, "filter_widget") and self.filter_widget.isVisible():
                    pass

                # Apply Visibility
                if matches_search:
                    widget.show()
                else:
                    widget.hide()
        self.update_placeholder_visibility()

    def add_prediction_card(self, data=None):
        if data and "name" in data:
            name = data["name"]
        else:
            count = 0
            for i in range(self.cards_layout.count()):
                if isinstance(
                    self.cards_layout.itemAt(i).widget(), FormulationConfigCard
                ):
                    count += 1
            name = f"Prediction {count + 1}"

        card = FormulationConfigCard(
            default_name=name,
            ingredients_data=self.ingredients_by_type,
            ingredient_types=self.INGREDIENT_TYPES,
            ingredient_units=self.INGREDIENT_UNITS,
            parent=self,
        )
        card.removed.connect(self.remove_card)
        card.run_requested.connect(self.run_prediction)
        card.expanded.connect(self.on_card_expanded)
        card.selection_changed.connect(self._on_card_selection_changed)
        card.visibility_toggled.connect(self._on_card_visibility_toggled)
        if card.is_measured:
            self.check_evaluation_eligibility()
        insert_idx = self.cards_layout.count()
        self.cards_layout.insertWidget(insert_idx, card)
        card.show()
        self.update_placeholder_visibility()

        if data:
            if hasattr(card, "load_data"):
                # load_data handles ingredient population if data['ingredients'] is a dict
                card.load_data(data)

            if data.get("measured", False):
                card.set_measured_state(True)

            # Only trigger a run if it's a prediction (not measured data)
            if not data.get("measured", False):
                card.emit_run_request()

        self.on_card_expanded(card)
        QtCore.QTimer.singleShot(100, lambda: self._scroll_to_card(card))

        self.update_placeholder_visibility()
        self.check_evaluation_eligibility()

        return card

    def _scroll_to_card(self, card_widget):
        """Helper to ensure the new card is visible in the scroll area."""
        self.scroll_area.ensureWidgetVisible(card_widget, 0, 0)

    def _on_card_visibility_toggled(self, card):
        """
        Called when a card's 'Hide from Plot' menu action is toggled.
        Forwards the request to the viz panel and keeps the action's
        check-state in sync with the actual hidden state.
        """
        card_name = card.name_input.text()
        new_hidden = self.viz_panel.toggle_card_series(card_name)
        # Sync the checkmark without re-firing the signal
        card.act_hide_series.blockSignals(True)
        card.act_hide_series.setChecked(new_hidden)
        card.act_hide_series.blockSignals(False)

    def on_card_expanded(self, active_card):
        """
        Handles card expansion: collapses others and updates the visualization
        with the active card's data (including measured profiles).
        """
        # 1. Collapse other cards
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, FormulationConfigCard) and widget is not active_card:
                widget.collapse()

        # Block the individual card from hijacking the plot if we are in Evaluation Mode
        if hasattr(self, "btn_evaluate") and self.btn_evaluate.isChecked():
            return

        # 2. Automatically plot data if available (Fixes missing plot on open)
        if hasattr(active_card, "last_results") and active_card.last_results:
            # Sync name in case it changed
            data = active_card.last_results
            data["config_name"] = active_card.name_input.text()

            # Send to Visualization Panel immediately
            self.viz_panel.set_plot_title(data["config_name"])
            self.viz_panel.set_data(data)

    def remove_card(self, card_widget):
        # --- Remove associated plot data from the visualization panel ---
        if hasattr(card_widget, "last_results"):
            target_result = card_widget.last_results
            target_name = card_widget.name_input.text()
            current_series = getattr(self.viz_panel, "data_series", [])
            new_series = []
            for data_pkg in current_series:
                if target_result and data_pkg is target_result:
                    continue
                if data_pkg.get("config_name") == target_name:
                    continue
                new_series.append(data_pkg)
            self.viz_panel.set_data(new_series)

        card_widget.setDisabled(True)
        anim = QtCore.QPropertyAnimation(card_widget, b"maximumHeight", card_widget)
        anim.setDuration(200)
        anim.setStartValue(card_widget.height())
        anim.setEndValue(0)
        anim.setEasingCurve(QtCore.QEasingCurve.InBack)

        def cleanup():
            self.cards_layout.removeWidget(card_widget)
            card_widget.deleteLater()
            QtCore.QTimer.singleShot(10, self.update_placeholder_visibility)
            self.check_evaluation_eligibility()

        anim.finished.connect(cleanup)
        anim.start()

    def import_run(self):
        """
        Initiates the background import process for one or multiple directories.
        """
        # 1. Retrieve Load Path from Preferences
        try:
            if (
                not hasattr(UserProfiles, "user_preferences")
                or UserProfiles.user_preferences is None
            ):
                UserProfiles.user_preferences = UserPreferences(
                    UserProfiles.get_session_file()
                )
            prefs = UserProfiles.user_preferences.get_preferences()
            path_from_prefs = prefs.get("load_data_path")
            if (
                path_from_prefs
                and isinstance(path_from_prefs, str)
                and path_from_prefs.strip()
            ):
                self.load_data_path = path_from_prefs
            else:
                self.load_data_path = Constants.working_logged_data_path
        except Exception as e:
            Log.e(TAG, f"Error reading load path from preferences: {e}")
            self.load_data_path = Constants.working_logged_data_path

        dialog = QtWidgets.QFileDialog(
            self, "Select Run Directory(s)", self.load_data_path
        )
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)

        if dialog.exec_():
            fnames = dialog.selectedFiles()
        else:
            return

        if not fnames:
            return

        self.progress_dialog = QtWidgets.QProgressDialog(
            "Scanning directories...", "Cancel", 0, 100, self
        )
        self.progress_dialog.setWindowTitle("Importing Data")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)

        self.worker = ImportWorker(fnames)

        # Connect Worker Signals
        self.worker.progress_changed.connect(self.progress_dialog.setValue)
        self.worker.status_changed.connect(self.progress_dialog.setLabelText)

        # [UPDATED] Store results temporarily, process only when thread is completely done
        self.worker.import_finished.connect(self._on_import_data_received)
        self.worker.finished.connect(self._on_import_thread_finished)
        self.worker.import_error.connect(self._on_import_error)

        self.progress_dialog.canceled.connect(self.worker.stop)
        self.worker.start()

        # Reset pending results
        self._pending_results = []

    def _on_import_data_received(self, results):
        """Buffer results until thread finishes cleanup."""
        self._pending_results = results

    def _on_import_thread_finished(self):
        """
        Called when the import thread has fully exited (database closed).
        Now safe to reload DB and process UI.
        """
        # 1. Reload Database (Critical Step)
        try:
            self.db.close()
            self.db = Database(parse_file_key=True)
            self.ing_ctrl = IngredientController(self.db)
            self.form_ctrl = FormulationController(self.db)
            self._load_database_data()  # Refresh ingredient dropdowns
        except Exception as e:
            print(f"Error reloading database after import: {e}")

        # 2. Process Results
        if hasattr(self, "_pending_results") and self._pending_results:
            self._process_imported_results(self._pending_results)
            self._pending_results = []  # Clear buffer

    def _process_imported_results(self, results):
        """
        Creates cards for each imported formulation.
        (Formerly _on_import_finished, DB reload logic removed)
        """
        if not results:
            return
        self._is_silencing_runs = True

        if not hasattr(self, "imported_runs"):
            self.imported_runs = []
        self.imported_runs.extend(results)

        count = 0
        for formulation in results:
            try:
                # [Ingredient Mapping Logic]
                ingredients_map = {}
                attr_map = {
                    "protein": "Protein",
                    "buffer": "Buffer",
                    "surfactant": "Surfactant",
                    "stabilizer": "Stabilizer",
                    "excipient": "Excipient",
                    "salt": "Salt",
                }
                for attr_name, type_name in attr_map.items():
                    if hasattr(formulation, attr_name):
                        component = getattr(formulation, attr_name)
                        if (
                            component
                            and hasattr(component, "ingredient")
                            and component.ingredient
                        ):
                            ing_obj = component.ingredient
                            name = getattr(ing_obj, "name", "None")
                            conc = getattr(component, "concentration", 0.0)
                            units = getattr(component, "units", "")
                            if name and name != "None":
                                ingredients_map[type_name] = {
                                    "name": name,
                                    "component": name,
                                    "concentration": float(conc) if conc else 0.0,
                                    "units": units,
                                }

                # Viscosity Data
                shear_rates = []
                viscosities = []
                temp = 25.0

                if (
                    hasattr(formulation, "viscosity_profile")
                    and formulation.viscosity_profile
                ):
                    if hasattr(formulation.viscosity_profile, "shear_rates"):
                        shear_rates = formulation.viscosity_profile.shear_rates
                    if hasattr(formulation.viscosity_profile, "viscosities"):
                        viscosities = formulation.viscosity_profile.viscosities

                if hasattr(formulation, "temperature"):
                    temp = float(formulation.temperature)

                notes = getattr(formulation, "notes", "")
                missing_fields = getattr(formulation, "missing_fields", [])

                card_data = {
                    "id": formulation.id,
                    "name": (
                        formulation.name
                        if formulation.name
                        else f"Imported Run {len(self.imported_runs)}"
                    ),
                    "ingredients": ingredients_map,
                    "measured": True,
                    "temperature": temp,
                    "notes": notes,
                    "missing_fields": missing_fields,
                    "icl": formulation.icl,
                    "last_model": formulation.last_model,
                }

                card = self.add_prediction_card(card_data)

                if card:
                    data_package = {
                        "config_name": card_data["name"],
                        "shear_rate": shear_rates,
                        "viscosity": viscosities,
                        "measured_viscosity": viscosities,
                        "measured_y": viscosities,
                        "y": viscosities,
                        "x": shear_rates,
                        "temperature": temp,
                        "color": card.plot_color,
                        "measured": True,
                    }

                    card.set_results(data_package)
                    self.viz_panel.set_plot_title(f"Imported: {card_data['name']}")
                    self.viz_panel.set_data(data_package)
                    count += 1

            except Exception as e:
                # Log.e(TAG, f"Error creating card for imported run: {e}")
                print(f"Error creating card: {e}")

        if count > 0:
            QtWidgets.QMessageBox.information(
                self,
                "Import Complete",
                f"Successfully created {count} formulation cards.",
            )
        else:
            QtWidgets.QMessageBox.warning(
                self, "Import Warning", "No valid formulations could be processed."
            )

        # --- UNMUTE RUNS ---
        self._is_silencing_runs = False

    def _on_import_error(self, error_msg):
        """
        Callback when the import worker encounters an error.
        """
        # Ensure progress dialog is closed
        if hasattr(self, "progress_dialog") and self.progress_dialog.isVisible():
            self.progress_dialog.cancel()

        QtWidgets.QMessageBox.critical(
            self, "Import Failed", f"An error occurred during import:\n{error_msg}"
        )

    def get_target_cards(self):
        """
        Returns a list of cards to act upon.
        Priority:
        """
        targets = []

        if self.selection_mode_active:
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, FormulationConfigCard) and widget.is_selected:
                    targets.append(widget)
        else:
            # Fallback to the currently open/expanded card
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, FormulationConfigCard) and widget.is_expanded:
                    targets.append(widget)
                    break  # Only one can be expanded at a time

        if not targets:
            QtWidgets.QMessageBox.information(
                self,
                "No Selection",
                "Please select cards or expand one to perform this action.",
            )

        return targets

    def export_analysis(self):
        """Batch export logic for the top bar."""
        target_cards = self.get_target_cards()

        if not target_cards:
            return

        if len(target_cards) == 1:
            target_cards[0].export_formulation()
        else:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Export Directory"
            )
            if folder:
                success = 0
                for card in target_cards:
                    if not card.last_results:
                        continue
                    base_name = card.name_input.text().replace(" ", "_")
                    file_path = os.path.join(folder, f"{base_name}.csv")

                    try:
                        card.save_to_csv(file_path)
                        success += 1
                    except Exception as e:
                        print(f"Error exporting {base_name}: {e}")

                QtWidgets.QMessageBox.information(
                    self, "Batch Export", f"Exported {success} files to {folder}"
                )

    # In prediction_ui.py

    def run_prediction(self, config=None):
        """
        Handles requests. Now supports Queuing and Zombie Task protection.
        """
        # --- BLOCK SPAM DURING IMPORT ---
        if self._is_silencing_runs:
            return

        sender_card = self.sender()

        # INTERCEPTION: If in batch collection mode, just queue and return
        if self._is_batch_collecting:
            if isinstance(sender_card, FormulationConfigCard) and config:
                self._batch_queue.append((sender_card, config))
            return

        if isinstance(sender_card, FormulationConfigCard):
            self.running_card = sender_card

        if self.running_card and hasattr(self.running_card, "last_results"):
            last_res = self.running_card.last_results
            if last_res and "x" in last_res and len(last_res["x"]) > 0:
                if config:
                    config["shear_rates"] = last_res["x"]

        self._pending_color = None
        if config and "color" in config and config["color"]:
            self._pending_color = config["color"]
        elif self.running_card:
            self._pending_color = self.running_card.plot_color

        # --- ZOMBIE TASK MANAGEMENT ---
        if self.current_task is not None and self.current_task.isRunning():
            self.current_task.stop()
            # Append to zombie list to prevent Python GC from destroying the C++ thread
            self._zombie_tasks.append(self.current_task)

        name = config.get("name", "Unknown Sample") if config else "Unknown Sample"
        if not self._is_batch_running:
            self.viz_panel.set_plot_title(f"Calculating: {name}...")
            self.viz_panel.show_loading()

        self.current_task = PredictionThread(config)
        self.current_task.data_ready.connect(self._on_prediction_finished)
        self.current_task.finished.connect(self._on_task_complete)
        self.current_task.start()

    # In prediction_ui.py

    def _on_prediction_finished(self, data_package):
        # Restore color
        if self._pending_color and "color" not in data_package:
            data_package["color"] = self._pending_color

        # [FIXED] Robust Restoration of Measured Data
        if hasattr(self, "running_card") and self.running_card:
            existing_results = self.running_card.last_results

            # 1. Check if we have valid existing measured data (Not None)
            if existing_results and existing_results.get("measured_y") is not None:

                # 2. Only restore if the NEW result doesn't have measured data
                if data_package.get("measured_y") is None:

                    # 3. Get the data arrays safely
                    old_measured = existing_results["measured_y"]
                    new_x = data_package.get("x")

                    # 4. Compare lengths (ensure new_x is valid list/array)
                    if new_x is not None and len(old_measured) == len(new_x):
                        data_package["measured_y"] = old_measured
                        data_package["measured"] = True  # Enable VizPanel toggle

        # Update the specific card that requested this
        if hasattr(self, "running_card") and self.running_card:
            self.running_card.set_results(data_package)

        if self._is_batch_running:
            self._batch_results.append(data_package)
            self._process_next_in_batch()
        elif self._is_evaluation_mode:
            # In eval mode but not a batch run (e.g., a single card re-run from expand).
            # Card results are updated above; don't clobber the eval/parity plot.
            pass
        else:
            final_name = data_package.get("config_name", "Unknown")
            self.viz_panel.set_plot_title(final_name)
            self.viz_panel.set_data(data_package)
            self.viz_panel.hide_loading()

    def _on_task_complete(self):
        """Called when thread naturally finishes. Cleans up zombie references."""
        sender = self.sender()
        if sender in self._zombie_tasks:
            self._zombie_tasks.remove(sender)

    def closeEvent(self, event):
        """
        Guaranteed cleanup on close.
        """
        if self.current_task is not None and self.current_task.isRunning():
            print("Closing application: Stopping background thread...")
            self.current_task.stop()
        self.db.close()
        super().closeEvent(event)

    def _update_overlay_geometry(self):
        """Updates positions of overlay widgets."""

        # 1. Filter Widget (Relative to Left Panel)
        if hasattr(self, "filter_widget") and self.filter_widget.isVisible():
            self.filter_widget.setGeometry(
                0, 0, self.left_widget.width(), self.filter_widget.sizeHint().height()
            )

        # 2. Evaluation Widget (Relative to Toolbar Button)
        if hasattr(self, "eval_widget") and self.eval_widget.isVisible():
            # Calculate Global Position of the Button's bottom-left corner
            btn_geo = self.btn_evaluate.geometry()
            global_pos = self.btn_evaluate.mapToGlobal(
                QtCore.QPoint(0, btn_geo.height())
            )

            # Map back to PredictionUI (self) coordinates
            local_pos = self.mapFromGlobal(global_pos)

            # Set Width (e.g., 300px standard dropdown width)
            menu_width = 300

            # Ensure it doesn't go off the right side of the screen
            x = local_pos.x()
            if x + menu_width > self.width():
                x = self.width() - menu_width - 10

            self.eval_widget.setGeometry(
                x, local_pos.y(), menu_width, self.eval_widget.sizeHint().height()
            )

            # Ensure it sits on top of everything (splitter, right panel, etc.)
            self.eval_widget.raise_()
        # 3. Generate Widget (Relative to Toolbar Button)
        if hasattr(self, "generate_widget") and self.generate_widget.isVisible():
            btn_geo = self.btn_generate.geometry()
            global_pos = self.btn_generate.mapToGlobal(
                QtCore.QPoint(0, btn_geo.height())
            )
            local_pos = self.mapFromGlobal(global_pos)

            menu_width = 550
            x = local_pos.x()
            if x + menu_width > self.width():
                x = self.width() - menu_width - 10

            self.generate_widget.setGeometry(
                x, local_pos.y(), menu_width, self.generate_widget.sizeHint().height()
            )
            self.generate_widget.raise_()
        # 4. Optimize Widget (Relative to Toolbar Button)
        if hasattr(self, "optimize_widget") and self.optimize_widget.isVisible():
            btn_geo = self.btn_optimize.geometry()
            global_pos = self.btn_optimize.mapToGlobal(
                QtCore.QPoint(0, btn_geo.height())
            )
            local_pos = self.mapFromGlobal(global_pos)

            menu_width = 600
            x = local_pos.x()
            if x + menu_width > self.width():
                x = self.width() - menu_width - 10

            self.optimize_widget.setGeometry(
                x, local_pos.y(), menu_width, self.optimize_widget.sizeHint().height()
            )
            self.optimize_widget.raise_()

    def resizeEvent(self, event):
        self._update_overlay_geometry()
        super().resizeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet())

    win = DashboardUI()
    win.setWindowTitle("Viscosity AI - Hyperparameter Tuning")
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())
