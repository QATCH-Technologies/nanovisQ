"""Dashboard UI module for the VisQAI viscosity prediction application.

Provides the main ``DashboardUI`` widget that manages formulation configuration
cards, viscosity predictions, evaluation, sample generation, optimization, and
data import/export.

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.3
"""

import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import numpy as np
import pandas as pd

try:
    TAG = "[Dashboard (HEADLESS)]"

    class Log:
        """Minimal stdout logger used when running in headless/standalone mode.

        Used when production imports fail. Mirrors the interface of the
        production Logger.
        """

        @staticmethod
        def d(TAG, msg=""):
            """Log a debug message to stdout.

            Args:
                TAG: Identifier string for the log source.
                msg: Message to log.
            """
            print("DEBUG:", TAG, msg)

        @staticmethod
        def i(TAG, msg=""):
            """Log an info message to stdout.

            Args:
                TAG: Identifier string for the log source.
                msg: Message to log.
            """
            print("INFO:", TAG, msg)

        @staticmethod
        def w(TAG, msg=""):
            """Log a warning message to stdout.

            Args:
                TAG: Identifier string for the log source.
                msg: Message to log.
            """
            print("WARNING:", TAG, msg)

        @staticmethod
        def e(TAG, msg=""):
            """Log an error message to stdout.

            Args:
                TAG: Identifier string for the log source.
                msg: Message to log.
            """
            print("ERROR:", TAG, msg)

    from architecture import Architecture
    from dialogs.database_table_dialog import DatabaseTableDialog
    from src.controller.formulation_controller import FormulationController
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.utils.metrics import Metrics
    from styles.style_loader import load_stylesheet
    from widgets.evaluation_widget import EvaluationWidget
    from widgets.formulation_config_card_widget import FormulationConfigCard
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
    TAG = "[Dashboard]"
    from QATCH.VisQAI.src.controller.formulation_controller import FormulationController
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.utils.metrics import Metrics
    from QATCH.VisQAI.src.view.dialogs.database_table_dialog import DatabaseTableDialog
    from QATCH.VisQAI.src.view.styles.style_loader import load_stylesheet
    from QATCH.VisQAI.src.view.widgets.evaluation_widget import EvaluationWidget
    from QATCH.VisQAI.src.view.widgets.formulation_config_card_widget import FormulationConfigCard
    from QATCH.VisQAI.src.view.widgets.generate_sample_widget import GenerateSampleWidget
    from QATCH.VisQAI.src.view.widgets.optimize_widget import OptimizeWidget
    from QATCH.VisQAI.src.view.widgets.placeholder_widget import PlaceholderWidget
    from QATCH.VisQAI.src.view.widgets.prediction_filter_widget import PredictionFilterWidget
    from QATCH.VisQAI.src.view.widgets.reordable_container_widget import ReorderableCardContainer
    from QATCH.VisQAI.src.view.widgets.visualization_panel import VisualizationPanel
    from QATCH.VisQAI.src.view.workers.import_worker import ImportWorker
    from QATCH.VisQAI.src.view.workers.optimization_worker import OptimizationWorker
    from QATCH.VisQAI.src.view.workers.prediction_worker import PredictionThread
    from QATCH.VisQAI.src.view.workers.sample_generation_worker import SampleGenerationWorker
    from QATCH.common.architecture import Architecture
    from QATCH.common.logger import Logger as Log
    from QATCH.common.userProfiles import UserPreferences, UserProfiles
    from QATCH.core.constants import Constants


class DashboardUI(QtWidgets.QWidget):
    """Main dashboard widget for VisQAI.

    Manages a scrollable list of FormulationConfigCard widgets on the left and
    a VisualizationPanel on the right. Coordinates inference predictions, batch
    analysis, evaluation mode, sample generation, and formulation optimization.

    Attributes:
        INGREDIENT_TYPES (list[str]): Ordered list of supported ingredient
            categories.
        INGREDIENT_UNITS (dict[str, str]): Mapping of ingredient type to its
            concentration unit.
    """

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
        """Initialize the dashboard, load the database, build the UI, and set up batch/state flags.

        Args:
            parent (QtWidgets.QWidget, optional): Parent widget. Defaults to None.
        """
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
        """Populate ``ingredients_by_type`` by fetching each ingredient category from the database via the ingredient controller."""
        # Fetch lists for each ingredient type
        self.ingredients_by_type["Protein"] = self.ing_ctrl.get_all_proteins()
        self.ingredients_by_type["Buffer"] = self.ing_ctrl.get_all_buffers()
        self.ingredients_by_type["Salt"] = self.ing_ctrl.get_all_salts()
        self.ingredients_by_type["Surfactant"] = self.ing_ctrl.get_all_surfactants()
        self.ingredients_by_type["Stabilizer"] = self.ing_ctrl.get_all_stabilizers()
        self.ingredients_by_type["Excipient"] = self.ing_ctrl.get_all_excipients()

    def init_ui(self):
        """Build the full dashboard layout.

        Constructs the top toolbar, left scrollable card panel (with filter,
        eval, generate, and optimize overlays), and right visualization panel
        in a horizontal splitter.
        """
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create Top Bar
        top_bar = self._create_unified_top_bar()
        main_layout.addWidget(top_bar)

        # Splitter Configuration
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet("QSplitter::handle { background-color: #d1d5db; }")

        # Left Panel
        self.left_widget = QtWidgets.QWidget()
        self.left_widget.setObjectName("leftPanel")
        left_layout = QtWidgets.QVBoxLayout(self.left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # Filter Widget
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
        self.generate_widget = GenerateSampleWidget(self.ingredients_by_type, parent=self)
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

        # Content
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

        # Right Panel
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
        """Build and return the fixed-height top toolbar.

        Contains the search bar and all action buttons: filter, select, import,
        export, generate, evaluate, hypothesis, optimize, run, delete, and the
        options menu.

        Returns:
            QtWidgets.QWidget: The fully constructed top toolbar widget.
        """
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

        # Left Side Controls
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
        self.btn_filter.setFixedSize(32, 32)
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
        self.btn_select_mode.setFixedSize(32, 32)
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

        # Separator
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

        # Spacer
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
        self.btn_right_options.setStyleSheet("QToolButton::menu-indicator { image: none; }")

        self.top_options_menu = QtWidgets.QMenu(self.btn_right_options)
        act_view_ing = self.top_options_menu.addAction("View Ingredients")
        act_view_ing.triggered.connect(self.show_ingredients_view)
        act_view_form = self.top_options_menu.addAction("View Formulations")
        act_view_form.triggered.connect(self.show_formulations_view)

        self.btn_right_options.setMenu(self.top_options_menu)

        layout.addWidget(self.btn_right_options)

        return container

    def show_ingredients_view(self):
        """Open a read-only dialog displaying all ingredients from the database.

        Groups ingredients by type with columns for name, ID, class, molecular
        weight, pH, pI, kP, and HCI.
        """
        # Define Headers
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

        # Iterate and Populate
        for ing_type, ingredients in self.ingredients_by_type.items():
            for ing in ingredients:
                row = []

                # Basic Info
                row.append(ing_type)
                row.append(getattr(ing, "name", "Unknown"))
                row.append(str(getattr(ing, "id", "")))

                # Defaults
                c_type_val = "-"
                c_class = "-"
                kp_val = "-"
                hci_val = "-"

                # Check for nested class_type object (common in Proteins)
                if hasattr(ing, "class_type") and ing.class_type:
                    ct = ing.class_type
                    c_type_val = str(getattr(ct, "value", getattr(ct, "name", str(ct))))
                    c_class = str(getattr(ct, "c_class", "-"))
                    kp_val = str(getattr(ct, "kP", "-"))
                    hci_val = str(getattr(ct, "hci", "-"))
                elif hasattr(ing, "group"):
                    c_type_val = str(ing.group)

                row.append(c_type_val)
                row.append(c_class)

                # Standard Attributes
                row.append(str(getattr(ing, "molecular_weight", "-")))
                row.append(str(getattr(ing, "pH", "-")))
                row.append(str(getattr(ing, "pI_mean", "-")))
                row.append(str(getattr(ing, "pI_range", "-")))

                # Nested Attributes
                row.append(kp_val)
                row.append(hci_val)

                rows.append(row)

        # how Dialog
        dlg = DatabaseTableDialog("Ingredient Database", headers, rows, self)
        dlg.resize(1200, 600)
        dlg.exec_()

    def show_formulations_view(self):
        """Open an interactive dialog showing all formulations in the database.

        Supports row deletion (imported-only), ICL toggle, and CSV export of
        the full database.
        """
        try:
            # Fetch Formulations
            formulations = self.form_ctrl.get_all_formulations()

            def delete_handler(f_id):
                target_f = next((f for f in formulations if str(f.id) == str(f_id)), None)
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
                try:
                    # Update DB safely
                    success = self.form_ctrl.update_formulation_metadata(int(f_id), icl=state)

                    if success:
                        # Update local list object
                        local_f = next((f for f in formulations if str(f.id) == str(f_id)), None)
                        if local_f:
                            local_f.icl = state

                        # Synchronize Active UI Cards
                        for i in range(self.cards_layout.count()):
                            item = self.cards_layout.itemAt(i)
                            widget = item.widget()
                            if isinstance(widget, FormulationConfigCard):
                                card_id = getattr(widget.formulation, "id", None)
                                if card_id is not None and str(card_id) == str(f_id):
                                    widget.set_icl_usage(state, save_db=False)
                                    break
                    else:
                        Log.w(TAG, f"Could not update ICL for ID {f_id}")

                except Exception as e:
                    Log.e(TAG, f"Error updating ICL state: {e}")

            def export_handler():
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

                # Others
                def add_simple_comp(comp_attr):
                    if hasattr(f, comp_attr):
                        c = getattr(f, comp_attr)
                        if c and c.ingredient:
                            row.extend([str(c.ingredient.name or "-"), str(c.concentration)])
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
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load formulations:\n{e}")

    def _on_card_selection_changed(self):
        """Deactivate selection mode when the last selected card is deselected."""
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
        """Toggle the GenerateSampleWidget overlay.

        Exits evaluation mode first if active.
        """
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
        """Toggle evaluation mode.

        On first activation, hides cards without measured data and shows the
        evaluation configuration widget. Subsequent clicks toggle the eval
        widget's visibility.
        """
        if not self._is_evaluation_mode:
            self._is_evaluation_mode = True
            self.btn_evaluate.setChecked(True)
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

            self.viz_panel.set_plot_title(f"Evaluation Mode: {visible_count} Datasets Ready")

            # Show the eval menu
            self.eval_widget.show()
            self.eval_widget.raise_()
            self._update_overlay_geometry()

        else:
            self.btn_evaluate.setChecked(True)

            if self.eval_widget.isVisible():
                self.eval_widget.hide()
            else:
                self.eval_widget.show()
                self.eval_widget.raise_()
                self._update_overlay_geometry()

        self.update_placeholder_visibility()

    def handle_hypothesis(self):
        """Prompt the user for a hypothesis name/ID and print it.

        This is a placeholder for future implementation.
        """
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Hypothesis Testing", "Enter hypothesis name or ID:"
        )
        if ok and text:
            Log.i(TAG, f"Hypothesis '{text}' initialized.")

    def handle_optimize(self):
        """Toggle the OptimizeWidget overlay."""
        if self.optimize_widget.isVisible():
            self.optimize_widget.hide()
            self.btn_optimize.setChecked(False)
        else:
            self.optimize_widget.show()
            self.optimize_widget.raise_()
            self.btn_optimize.setChecked(True)
            self._update_overlay_geometry()

    def _create_fab(self):
        """Create and position the floating action button ("+") that adds a new blank prediction card."""
        self.btn_add_fab = QtWidgets.QPushButton("+", self.left_widget)
        self.btn_add_fab.setToolTip("New Prediction")
        self.btn_add_fab.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_fab.resize(50, 50)
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
        """Reposition the FAB and filter overlay when the left panel is resized.

        Args:
            source (QtCore.QObject): The object that generated the event.
            event (QtCore.QEvent): The event to filter.

        Returns:
            bool: Result of the parent ``eventFilter`` call.
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
        """Show the placeholder widget when no cards are visible.

        Updates the placeholder message depending on whether cards exist but
        are filtered out, or none have been added yet.
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
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()

            if isinstance(widget, FormulationConfigCard):
                if widget.is_measured:
                    has_measured_data = True
                    break

        # Update Button State
        if hasattr(self, "btn_evaluate"):
            self.btn_evaluate.setEnabled(has_measured_data)
            if not has_measured_data and self.btn_evaluate.isChecked():
                self.btn_evaluate.setChecked(False)
                self.exit_evaluation_mode()

    def _is_filter_default(self, filters):
        """Return True if the given filter dict represents the default (no-op) filter state.

        Args:
            filters (dict): Filter configuration dictionary to evaluate.

        Returns:
            bool: True if the filter is at its default (no-op) state.
        """
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
        """Exit evaluation mode.

        Restores all card visibility, re-sends the last prediction to the plot
        for the currently expanded card, and resets the plot title.
        """
        self._is_evaluation_mode = False
        self.eval_widget.hide()
        self.btn_evaluate.setChecked(False)
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
        if expanded_card is not None:
            self.running_card = expanded_card
            expanded_card.emit_run_request()

    def on_eval_point_clicked(self, point_data):
        """Reveal and scroll to the card associated with a clicked point in the parity/eval plot.

        Args:
            point_data (dict): Dictionary containing at least a ``card`` key
                mapping to the ``FormulationConfigCard`` that was clicked.
        """
        card = point_data.get("card")
        if not card:
            return

        card.show()
        if not card.is_expanded:
            card.expand_silent()

        self._scroll_to_card(card)

    def run_evaluation_analysis(self, config):
        """Collect all visible measured cards and trigger their predictions.

        Triggers prediction requests via the batch mechanism, then calls
        ``_compute_evaluation`` once all predictions are ready.

        Args:
            config (dict): Evaluation configuration including metric key,
                shear range, and display options.
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
            self.viz_panel.set_plot_title(
                f"Running predictions for evaluation ({len(self._batch_queue)} cards)..."
            )
            self.viz_panel.show_loading()
            self._process_next_in_batch()
        else:
            self._compute_evaluation(config)

    def _compute_evaluation(self, config):
        """Compute and display evaluation results.

        For the ``true_vs_pred`` metric, builds parity data and calls
        ``viz_panel.set_parity_data``. For other metrics, uses the ``Metrics``
        engine to score each card and annotates plot titles with average scores.

        Args:
            config (dict): Evaluation configuration including metric key,
                shear range, and display options.
        """
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
                if "measured_y" not in data or "y" not in data or data["measured_y"] is None:
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
                if "measured_y" not in data or "y" not in data or data["measured_y"] is None:
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
                        "percentage_error": np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
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
                    Log.e(
                        TAG,
                        f"Failed to calculate {metric_key} for {card.name_input.text()}: {e}",
                    )

            if not results_to_plot:
                QtWidgets.QMessageBox.warning(
                    self, "Evaluation Failed", "No valid data found in range."
                )
                return

            self.viz_panel.set_data(results_to_plot)
            if scores:
                avg = sum(scores) / len(scores)
                self.viz_panel.set_plot_title(f"Evaluation Results: Avg {metric_name} = {avg:.4f}")
            else:
                self.viz_panel.set_plot_title(f"Evaluation Results: {metric_name}")

    def run_sample_generation(self, num_samples, model_file, constraints_data):
        """Launch a ``SampleGenerationWorker`` with the given constraints and display a cancellable progress dialog.

        Args:
            num_samples (int): Number of samples to generate.
            model_file (str): Path to the model file used for generation.
            constraints_data (dict): Constraint configuration for the generator.
        """
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
        )

        if not hasattr(self, "_active_workers"):
            self._active_workers = []
        self._active_workers.append(worker)

        worker.progress_update.connect(self._on_generation_progress)
        worker.generation_complete.connect(self._on_generation_complete)
        worker.generation_error.connect(self._on_generation_error)
        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))
        self.progress_dialog.canceled.connect(worker.stop)

        worker.start()

    def run_optimization(self, model_file, targets, constraints_data):
        """Launch an ``OptimizationWorker`` with the given model, targets, and constraints.

        Shows a loading overlay with cancel support and stores pre-optimization
        plot state for rollback on cancel.

        Args:
            model_file (str): Path to the model file used for optimization.
            targets (dict): Target viscosity profile constraints.
            constraints_data (dict): Formulation constraint configuration.
        """
        self.optimize_widget.hide()
        self.btn_optimize.setChecked(False)
        self._opt_cancelled = False

        maxiter = self.optimize_widget.spin_maxiter.value()
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
        self.viz_panel.show_loading(cancel_callback=lambda: self._cancel_optimization(worker))

        worker.progress_update.connect(self._on_optimization_progress)
        worker.optimization_complete.connect(self._on_optimization_complete)
        worker.optimization_error.connect(self._on_optimization_error)
        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))

        worker.start()

    def _on_optimization_progress(self, value, text):
        """Update the loading overlay progress bar and label text during optimization.

        Args:
            value (int): Progress percentage (0–100).
            text (str): Status message to display in the overlay label.
        """
        if hasattr(self.viz_panel, "anim_timer") and self.viz_panel.anim_timer.isActive():
            self.viz_panel.anim_timer.stop()
        if hasattr(self.viz_panel, "progress_bar"):
            self.viz_panel.progress_bar.setValue(value)
        if hasattr(self.viz_panel, "loading_label"):
            self.viz_panel.loading_label.setText(text)
        self.viz_panel.set_plot_title(f"Optimizing… {value}%")

    def _on_optimization_complete(self, card_data):
        """Hide loading, add an optimized prediction card, and display its estimated viscosity profile.

        Args:
            card_data (dict): Data dict for the optimized formulation card,
                including an optional ``estimated_profile`` key.
        """
        if getattr(self, "_opt_cancelled", False):
            return
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
        """Hide loading and show a critical error dialog with the optimizer's error message.

        Args:
            error_msg (str): Error message emitted by the optimization worker.
        """
        if getattr(self, "_opt_cancelled", False):
            return
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
        """Stop the optimization worker and restore the pre-optimization plot state.

        Args:
            worker (OptimizationWorker): The worker instance to stop.
        """
        self._opt_cancelled = True
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
            self.viz_panel.set_plot_title("")
            self.viz_panel.set_data([])

    def _cleanup_worker(self, worker):
        """Wait for a worker thread to finish (up to 5 s) then remove it from the active workers list.

        Args:
            worker (QtCore.QThread): The worker thread to clean up.
        """
        if not worker.wait(5000):
            Log.w(TAG, "Worker thread did not finish in time during cleanup.")
        if hasattr(self, "_active_workers") and worker in self._active_workers:
            self._active_workers.remove(worker)

    def _on_generation_progress(self, value, text):
        """Update the progress dialog value and label during sample generation.

        Args:
            value (int): Progress percentage (0–100).
            text (str): Status message to display in the progress dialog.
        """
        if hasattr(self, "progress_dialog"):
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(text)

    def _on_generation_complete(self, generated_cards_data):
        """Close the progress dialog, add a prediction card for each generated sample, and notify the user.

        Triggers batch predictions for the new cards and notifies the user of
        the total generated count.

        Args:
            generated_cards_data (list[dict]): List of card data dicts for
                each generated sample.
        """
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
        """Close the progress dialog and show a critical error dialog.

        Args:
            error_msg (str): Error message emitted by the generation worker.
        """
        if hasattr(self, "progress_dialog"):
            self.progress_dialog.close()

        QtWidgets.QMessageBox.critical(
            self, "Generation Error", f"Failed to generate samples:\n{error_msg}"
        )

    def apply_filters(self, filter_data):
        """Apply the complex filter dict together with the current search-bar text.

        Shows/hides cards accordingly, updates the filter button's active
        state, and hides the filter widget.

        Args:
            filter_data (dict): Filter configuration emitted by
                ``PredictionFilterWidget``.
        """
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
        """Show the filter widget if hidden, or hide it if visible, and update its geometry."""
        if self.filter_widget.isVisible():
            self.filter_widget.hide()
        else:
            self.filter_widget.show()
            self.filter_widget.raise_()
            self._update_filter_geometry()

    def _update_filter_geometry(self):
        """Resize the filter widget to span the full width of the left panel at the top."""
        if self.filter_widget.isVisible():
            self.filter_widget.setGeometry(
                0, 0, self.left_widget.width(), self.filter_widget.sizeHint().height()
            )

    def run_analysis(self):
        """Collect target cards, gather their prediction configs via the batch-collection mechanism, then start sequential batch processing."""
        target_cards = self.get_target_cards()
        if not target_cards:
            return

        # Initialize Batch State
        self._batch_queue = []
        self._batch_results = []
        self._is_batch_collecting = True
        # Trigger requests from cards
        for card in target_cards:
            card.emit_run_request()
        self._is_batch_collecting = False

        # Start processing the queue
        if self._batch_queue:
            self._is_batch_running = True
            self.viz_panel.set_plot_title("Initializing Batch Analysis...")
            self.viz_panel.show_loading()
            self._process_next_in_batch()
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "Failed to collect configuration data.")

    def _process_next_in_batch(self):
        """Pop the next (card, config) pair from the batch queue and start its prediction.

        When the queue is empty, finalizes by either triggering evaluation
        computation or displaying all batch results on the plot.
        """
        if not self._batch_queue:
            self._is_batch_running = False
            self.viz_panel.hide_loading()

            if self._is_evaluation_mode and self._pending_eval_config is not None:
                config = self._pending_eval_config
                self._pending_eval_config = None
                self._compute_evaluation(config)
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
        """Prompt for confirmation, then remove the selected (or expanded) card(s) and their corresponding data series from the visualization panel."""
        target_cards = []

        # Identify Target Cards
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

        # Confirm Deletion
        count = len(target_cards)
        msg = f"Are you sure you want to delete {count} prediction(s)?"
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Delete",
            msg,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            # Gather references to the data objects and names from the cards being deleted
            results_to_remove = []
            names_to_remove = []

            for card in target_cards:
                if hasattr(card, "last_results") and card.last_results:
                    results_to_remove.append(card.last_results)
                names_to_remove.append(card.name_input.text())
            current_series = getattr(self.viz_panel, "data_series", [])
            new_series = []

            for data_pkg in current_series:
                if any(data_pkg is res for res in results_to_remove):
                    continue
                if data_pkg.get("config_name") in names_to_remove:
                    continue

                new_series.append(data_pkg)
            self.viz_panel.set_data(new_series)
            for card in target_cards:
                self.remove_card(card)

            # Cleanup selection state
            if self.selection_mode_active:
                self.btn_select_mode.setChecked(False)
            elif self.cards_layout.count() > 0:
                pass

    def toggle_selection_mode(self, active):
        """Enable or disable multi-select mode.

        Disables FAB/Import when active, enables Select-All, and propagates
        the selectable state to all cards.

        Args:
            active (bool): True to enable selection mode, False to disable.
        """
        self.selection_mode_active = active

        # Disable Add FAB and Import when selecting
        self.btn_add_fab.setEnabled(not active)
        self.btn_import.setEnabled(not active)
        self.btn_select_all.setEnabled(active)

        # Dim the FAB visually if disabled
        if active:
            # Use opacity effect when dimmed (loses shadow temporarily)
            effect = QtWidgets.QGraphicsOpacityEffect(self.btn_add_fab)
            effect.setOpacity(0.5)
            self.btn_add_fab.setGraphicsEffect(effect)
        else:
            # Restore shadow effect when enabled
            shadow = QtWidgets.QGraphicsDropShadowEffect(self.btn_add_fab)
            shadow.setBlurRadius(15)
            shadow.setXOffset(0)
            shadow.setYOffset(4)
            shadow.setColor(QtGui.QColor(0, 0, 0, 80))
            self.btn_add_fab.setGraphicsEffect(shadow)

        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()
            if widget and isinstance(widget, FormulationConfigCard):
                if hasattr(widget, "set_selectable"):
                    widget.set_selectable(active)

    def select_all_cards(self):
        """Select all visible cards if any are unselected; deselect all if all are already selected."""
        visible_cards = []
        for i in range(self.cards_layout.count()):
            w = self.cards_layout.itemAt(i).widget()
            if isinstance(w, FormulationConfigCard) and not w.isHidden():
                visible_cards.append(w)

        if not visible_cards:
            return

        selected_visible = [w for w in visible_cards if w.is_selected]
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
                card_content = widget.get_searchable_text()

                # Check Search Match
                matches_search = True
                if search_tokens:
                    for token in search_tokens:
                        if token not in card_content:
                            matches_search = False
                            break
                if hasattr(self, "filter_widget") and self.filter_widget.isVisible():
                    pass

                # Apply Visibility
                if matches_search:
                    widget.show()
                else:
                    widget.hide()
        self.update_placeholder_visibility()

    def add_prediction_card(self, data=None):
        """Create and insert a new ``FormulationConfigCard`` at the end of the card list.

        If ``data`` is provided, loads its contents and triggers a prediction
        for non-measured cards.

        Args:
            data (dict, optional): Card configuration data to pre-populate.
                If None, an empty card is created. Defaults to None.

        Returns:
            FormulationConfigCard: The newly created card widget.
        """
        if data and "name" in data:
            name = data["name"]
        else:
            count = 0
            for i in range(self.cards_layout.count()):
                if isinstance(self.cards_layout.itemAt(i).widget(), FormulationConfigCard):
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
        card.color_changed.connect(self._on_card_color_changed)
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
                card.load_data(data)
            if data.get("measured", False):
                card.set_measured_state(True)
            if not data.get("measured", False):
                card.emit_run_request()

        self.on_card_expanded(card)
        QtCore.QTimer.singleShot(100, lambda: self._scroll_to_card(card))

        self.update_placeholder_visibility()
        self.check_evaluation_eligibility()

        return card

    def _scroll_to_card(self, card_widget):
        """Scroll the scroll area to ensure the given card widget is visible.

        Args:
            card_widget (FormulationConfigCard): The card to scroll into view.
        """
        self.scroll_area.ensureWidgetVisible(card_widget, 0, 0)

    def _on_card_color_changed(self, new_color: str):
        """Update the plot series colour when a card's colour picker changes.

        Finds the data-series entry in the visualization panel whose
        ``"config_name"`` matches the emitting card's name, updates its
        ``"color"`` key, and triggers a full plot redraw.

        Args:
            new_color (str): Hex colour string selected by the user (e.g.
                ``"#2596be"``).
        """
        card = self.sender()
        if not isinstance(card, FormulationConfigCard):
            return
        card_name = card.name_input.text()
        for series in self.viz_panel.data_series:
            if series.get("config_name") == card_name:
                series["color"] = new_color
        self.viz_panel.update_plot()

    def _on_card_visibility_toggled(self, card):
        """Toggle the corresponding data series in the visualization panel and sync the card's hide-series action state.

        Args:
            card (FormulationConfigCard): The card whose series visibility
                should be toggled.
        """
        card_name = card.name_input.text()
        new_hidden = self.viz_panel.toggle_card_series(card_name)
        card.act_hide_series.blockSignals(True)
        card.act_hide_series.setChecked(new_hidden)
        card.act_hide_series.blockSignals(False)

    def on_card_expanded(self, active_card):
        """Collapse all other cards when one expands.

        If not in evaluation mode, updates the plot to show the expanded
        card's last results.

        Args:
            active_card (FormulationConfigCard): The card that was just
                expanded.
        """
        # Collapse other cards
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, FormulationConfigCard) and widget is not active_card:
                widget.collapse()
        if hasattr(self, "btn_evaluate") and self.btn_evaluate.isChecked():
            return

        if hasattr(active_card, "last_results") and active_card.last_results:
            # Sync name in case it changed
            data = active_card.last_results
            data["config_name"] = active_card.name_input.text()
            self.viz_panel.set_plot_title(data["config_name"])
            self.viz_panel.set_data(data)

    def remove_card(self, card_widget):
        """Remove a card's data series from the visualization panel, then animate the card's collapse and delete it from the layout.

        Args:
            card_widget (FormulationConfigCard): The card to remove.
        """
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
        """Open a directory picker, launch an ``ImportWorker`` for the selected path(s), and display a cancellable progress dialog."""
        # Retrieve Load Path from Preferences
        try:
            if (
                not hasattr(UserProfiles, "user_preferences")
                or UserProfiles.user_preferences is None
            ):
                UserProfiles.user_preferences = UserPreferences(UserProfiles.get_session_file())
            prefs = UserProfiles.user_preferences.get_preferences()
            path_from_prefs = prefs.get("load_data_path")
            if path_from_prefs and isinstance(path_from_prefs, str) and path_from_prefs.strip():
                self.load_data_path = path_from_prefs
            else:
                self.load_data_path = Constants.working_logged_data_path
        except Exception as e:
            Log.e(TAG, f"Error reading load path from preferences: {e}")
            self.load_data_path = Constants.working_logged_data_path

        dialog = QtWidgets.QFileDialog(self, "Select Run Directory(s)", self.load_data_path)
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
        self.worker.import_finished.connect(self._on_import_data_received)
        self.worker.finished.connect(self._on_import_thread_finished)
        self.worker.import_error.connect(self._on_import_error)

        self.progress_dialog.canceled.connect(self.worker.stop)
        self.worker.start()

        # Reset pending results
        self._pending_results = []

    def _on_import_data_received(self, results):
        """Buffer the raw import results emitted by the worker.

        Args:
            results (list): List of ``Formulation`` objects returned by the
                import worker.
        """
        self._pending_results = results

    def _on_import_thread_finished(self):
        """Reload the database and ingredient controllers after the import thread finishes, then process any buffered results."""
        # Reload Database
        try:
            self.db.close()
            self.db = Database(parse_file_key=True)
            self.ing_ctrl = IngredientController(self.db)
            self.form_ctrl = FormulationController(self.db)
            self._load_database_data()  # Refresh ingredient dropdowns
        except Exception as e:
            Log.e(TAG, f"Error reloading database after import: {e}")

        # Process Results
        if hasattr(self, "_pending_results") and self._pending_results:
            self._process_imported_results(self._pending_results)
            self._pending_results = []  # Clear buffer

    def _process_imported_results(self, results):
        """Convert imported ``Formulation`` objects into card data dicts and create cards for each.

        Sets their measured viscosity data and notifies the user of the total
        imported count.

        Args:
            results (list): List of ``Formulation`` objects to process.
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
                        if component and hasattr(component, "ingredient") and component.ingredient:
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

                if hasattr(formulation, "viscosity_profile") and formulation.viscosity_profile:
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
                Log.e(TAG, f"Error creating card for imported run: {e}")

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
        self._is_silencing_runs = False

    def _on_import_error(self, error_msg):
        """Cancel the progress dialog and show a critical import-error dialog.

        Args:
            error_msg (str): Error message emitted by the import worker.
        """
        if hasattr(self, "progress_dialog") and self.progress_dialog.isVisible():
            self.progress_dialog.cancel()

        QtWidgets.QMessageBox.critical(
            self, "Import Failed", f"An error occurred during import:\n{error_msg}"
        )

    def get_target_cards(self):
        """Return the list of cards to act on.

        Returns selected cards in selection mode, or the single expanded card
        otherwise. Shows an info dialog if no target is found.

        Returns:
            list[FormulationConfigCard]: Cards that are currently targeted for
                an action.
        """
        targets = []

        if self.selection_mode_active:
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, FormulationConfigCard) and widget.is_selected:
                    targets.append(widget)
        else:
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
        """Export the target card(s).

        Single-card export delegates to the card's own export method;
        multi-card batch export saves each to a CSV in a user-chosen folder.
        """
        target_cards = self.get_target_cards()

        if not target_cards:
            return

        if len(target_cards) == 1:
            target_cards[0].export_formulation()
        else:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Export Directory")
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
                        Log.e(TAG, f"Error exporting {base_name}: {e}")

                QtWidgets.QMessageBox.information(
                    self, "Batch Export", f"Exported {success} files to {folder}"
                )

    def run_prediction(self, config=None):
        """Start a ``PredictionThread`` for the given config.

        If batch-collecting, queues the request instead. Stops any currently
        running task (adds it to zombie list) before launching the new one.

        Args:
            config (dict, optional): Prediction configuration dict. If None,
                no prediction is started. Defaults to None.
        """
        if self._is_silencing_runs:
            return

        sender_card = self.sender()

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
        if self.current_task is not None and self.current_task.isRunning():
            self.current_task.stop()
            self._zombie_tasks.append(self.current_task)

        name = config.get("name", "Unknown Sample") if config else "Unknown Sample"
        if not self._is_batch_running:
            self.viz_panel.set_plot_title(f"Calculating: {name}...")
            self.viz_panel.show_loading()

        self.current_task = PredictionThread(config)
        self.current_task.data_ready.connect(self._on_prediction_finished)
        self.current_task.finished.connect(self._on_task_complete)
        self.current_task.start()

    def _on_prediction_finished(self, data_package):
        """Receive prediction results and update the card and visualization panel.

        Preserves any existing measured data if the new result lacks it.
        Updates the card's display, and advances the batch queue if a batch is
        running.

        Args:
            data_package (dict): Prediction results containing at minimum ``x``
                (shear rates) and ``y`` (viscosities) arrays.
        """
        if self._pending_color and "color" not in data_package:
            data_package["color"] = self._pending_color

        if hasattr(self, "running_card") and self.running_card:
            existing_results = self.running_card.last_results
            if existing_results and existing_results.get("measured_y") is not None:
                if data_package.get("measured_y") is None:
                    old_measured = existing_results["measured_y"]
                    new_x = data_package.get("x")
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
            pass
        else:
            final_name = data_package.get("config_name", "Unknown")
            self.viz_panel.set_plot_title(final_name)
            self.viz_panel.set_data(data_package)
            self.viz_panel.hide_loading()

    def _on_task_complete(self):
        """Clean up a finished task from the zombie list."""
        sender = self.sender()
        if sender in self._zombie_tasks:
            self._zombie_tasks.remove(sender)

    def closeEvent(self, event):
        """Stop the running prediction thread and close the database connection before the widget closes.

        Args:
            event (QtGui.QCloseEvent): The close event to handle.
        """
        if self.current_task is not None and self.current_task.isRunning():
            Log.i(TAG, "Closing application: Stopping background thread...")
            self.current_task.stop()

        # Stop all active workers (optimization, generation, etc.)
        if hasattr(self, "_active_workers"):
            for worker in list(self._active_workers):
                if hasattr(worker, "stop"):
                    worker.stop()
                if not worker.wait(2000):
                    Log.w(TAG, f"Worker {worker} did not finish in time on close.")
            self._active_workers.clear()

        self.db.close()
        super().closeEvent(event)

    def _update_overlay_geometry(self):
        """Reposition all visible overlay widgets (filter, eval, generate, optimize) relative to their trigger buttons and the dashboard boundaries."""
        # Filter Widget
        if hasattr(self, "filter_widget") and self.filter_widget.isVisible():
            self.filter_widget.setGeometry(
                0, 0, self.left_widget.width(), self.filter_widget.sizeHint().height()
            )

        # Evaluation Widget
        if hasattr(self, "eval_widget") and self.eval_widget.isVisible():
            btn_geo = self.btn_evaluate.geometry()
            global_pos = self.btn_evaluate.mapToGlobal(QtCore.QPoint(0, btn_geo.height()))
            local_pos = self.mapFromGlobal(global_pos)
            menu_width = 300
            x = local_pos.x()
            if x + menu_width > self.width():
                x = self.width() - menu_width - 10

            self.eval_widget.setGeometry(
                x, local_pos.y(), menu_width, self.eval_widget.sizeHint().height()
            )
            self.eval_widget.raise_()

        # Generate Widget
        if hasattr(self, "generate_widget") and self.generate_widget.isVisible():
            btn_geo = self.btn_generate.geometry()
            global_pos = self.btn_generate.mapToGlobal(QtCore.QPoint(0, btn_geo.height()))
            local_pos = self.mapFromGlobal(global_pos)

            menu_width = 550
            x = local_pos.x()
            if x + menu_width > self.width():
                x = self.width() - menu_width - 10

            self.generate_widget.setGeometry(
                x, local_pos.y(), menu_width, self.generate_widget.sizeHint().height()
            )
            self.generate_widget.raise_()

        # Optimize Widget
        if hasattr(self, "optimize_widget") and self.optimize_widget.isVisible():
            btn_geo = self.btn_optimize.geometry()
            global_pos = self.btn_optimize.mapToGlobal(QtCore.QPoint(0, btn_geo.height()))
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
        """Reposition overlays when the dashboard is resized.

        Args:
            event (QtGui.QResizeEvent): The resize event to handle.
        """
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
