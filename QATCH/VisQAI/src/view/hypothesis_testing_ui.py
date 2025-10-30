"""
hypothesis_testing_ui.py

This module provides the HypothesisTestingUI class for the Hypothesis Testing tab in VisQAI.
It supports creating custom formulations, defining hypotheses about viscosity behavior,
and visualizing prediction results with hypothesis outcomes using area-based CI polygon containment.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-10-30

Version:
   2.0 - Adapted for area-based CI polygon containment
"""

import sys
import os
import traceback
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING
import json
import numpy as np

try:
    from QATCH.common.logger import Logger as Log
    from QATCH.common.architecture import Architecture
    from QATCH.core.constants import Constants

except (ModuleNotFoundError, ImportError):

    class Log:
        @staticmethod
        def d(tag, msg=""): print("DEBUG:", tag, msg)
        @staticmethod
        def i(tag, msg=""): print("INFO:", tag, msg)
        @staticmethod
        def w(tag, msg=""): print("WARNING:", tag, msg)
        @staticmethod
        def e(tag, msg=""): print("ERROR:", tag, msg)

    class Constants:
        app_title = "VisQAI"
        log_prefer_path = os.path.expanduser("~/Documents")
    Log.i("Running VisQAI as standalone app")

    class Architecture:
        @staticmethod
        def get_path():
            return os.path.dirname(os.path.abspath(__file__))

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon

try:
    from src.models.formulation import Formulation
    from src.models.predictor import Predictor
    from src.processors.hypothesis_testing import HypothesisTesting
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.models.ingredient import Protein, Buffer, Salt, Stabilizer, Surfactant, Excipient
    if TYPE_CHECKING:
        from src.view.main_window import VisQAIWindow
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.formulation import Formulation
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.processors.hypothesis_testing import HypothesisTesting
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.models.ingredient import Protein, Buffer, Salt, Stabilizer, Surfactant, Excipient
    if TYPE_CHECKING:
        from QATCH.VisQAI.src.view.main_window import VisQAIWindow

TAG = "[HypothesisTestingUI]"


class HypothesisTestingUI(QtWidgets.QDialog):
    """
    A UI for hypothesis testing of viscosity predictions in VisQAI.

    This interface allows users to:
    - Build custom formulations from available ingredients
    - Define hypotheses about viscosity behavior
    - Run model predictions with area-based CI polygon containment
    - Visualize results with hypothesis outcomes
    """

    HYPOTHESIS_TYPES = {
        'upper': 'Upper Bound (Less Than)',
        'lower': 'Lower Bound (Greater Than)',
        'between': 'Between Bounds'
    }

    SHEAR_RATES = [100, 1000, 10000, 100000, 15000000]

    SHEAR_RATE_LABELS = {
        100: 'Viscosity_100',
        1000: 'Viscosity_1000',
        10000: 'Viscosity_10000',
        100000: 'Viscosity_100000',
        15000000: 'Viscosity_15000000'
    }

    INGREDIENT_UNITS = {
        'Protein': 'mg/ml',
        'Buffer': 'mM',
        'Surfactant': '%w',
        'Stabilizer': 'M',
        'Excipient': 'mM',
        'Salt': 'mM'
    }

    INGREDIENT_TYPES = ['Protein', 'Buffer',
                        'Surfactant', 'Stabilizer', 'Excipient', 'Salt']

    def __init__(self, parent=None):
        """Initialize the HypothesisTestingUI window.

        Args:
            parent (VisQAIWindow, optional): The parent window instance.
        """
        super().__init__(parent)
        self.parent: 'VisQAIWindow' = parent
        self.setWindowTitle("Hypothesis Testing (Area-Based)")

        # Database and controllers
        self.db: Optional[Database] = None
        self.ing_ctrl: Optional[IngredientController] = None

        # Core components
        self.predictor: Optional[Predictor] = None
        self.hypothesis_tester: Optional[HypothesisTesting] = None
        self.model_path: Optional[str] = None

        # Current formulation and results
        self.current_formulation: Optional[Dict[str, float]] = {}
        self.formulation_obj: Optional[Formulation] = None
        self.hypothesis_result: Optional[Dict] = None

        # Temperature
        self.temperature: Optional[float] = None

        # Available ingredients by type
        self.ingredients_by_type: Dict[str, List] = {
            'Protein': [],
            'Buffer': [],
            'Surfactant': [],
            'Stabilizer': [],
            'Excipient': [],
            'Salt': []
        }

        # Selected ingredients by type
        self.selected_ingredients: Dict[str, tuple] = {
            'Protein': None,
            'Buffer': None,
            'Surfactant': None,
            'Stabilizer': None,
            'Excipient': None,
            'Salt': None
        }

        # Initialize file dialogs
        self._init_file_dialogs()

        # Initialize database and load ingredients
        self._init_database()
        self._load_ingredients_by_type()

        self.init_ui()

    def _init_file_dialogs(self) -> None:
        """Initialize file dialog for model selection."""
        self.model_dialog = QtWidgets.QFileDialog()
        self.model_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self.model_dialog.setNameFilter(
            "VisQAI Model Files (*.zip);;All Files (*)")
        self.model_dialog.setViewMode(QtWidgets.QFileDialog.Detail)

        # Set default directory
        model_path = os.path.join(
            Architecture.get_path(), "QATCH/VisQAI/assets")
        if os.path.exists(model_path):
            self.model_dialog.setDirectory(model_path)
        else:
            self.model_dialog.setDirectory(Constants.log_prefer_path)

    def _init_database(self) -> None:
        """Initialize database and ingredient controller."""
        try:
            self.db = Database(parse_file_key=True)
            self.ing_ctrl = IngredientController(self.db)
            Log.i(TAG, "Database initialized successfully")
        except Exception as e:
            Log.e(TAG, f"Failed to initialize database: {e}")
            self.db = None
            self.ing_ctrl = None

    def _load_ingredients_by_type(self) -> None:
        """Load ingredients from database and organize by type."""
        if self.ing_ctrl is None:
            Log.w(TAG, "Cannot load ingredients - controller not available")
            return

        try:
            # Get all ingredients from database
            all_ingredients = self.ing_ctrl.get_all_ingredients()

            # Organize by type (subclass name)
            for ingredient in all_ingredients:
                ingredient_type = ingredient.type
                if ingredient_type in self.ingredients_by_type:
                    self.ingredients_by_type[ingredient_type].append(
                        ingredient)

            # Sort each type by name and remove duplicates
            for ing_type in self.ingredients_by_type:
                ingredients = self.ingredients_by_type[ing_type]

                # Remove duplicates by name (case-insensitive)
                unique_ingredients = {}
                for ing in ingredients:
                    key = ing.name.lower()
                    if key not in unique_ingredients:
                        unique_ingredients[key] = ing

                # Replace with unique, sorted list
                self.ingredients_by_type[ing_type] = sorted(
                    unique_ingredients.values(), key=lambda x: x.name.lower())

            Log.i(TAG, f"Loaded ingredients by type: " +
                  ", ".join([f"{k}:{len(v)}" for k, v in self.ingredients_by_type.items()]))

        except Exception as e:
            Log.e(TAG, f"Failed to load ingredients by type: {e}")
            Log.e(TAG, traceback.format_exc())

    def init_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        left_panel = self.create_configuration_panel()
        splitter.addWidget(left_panel)
        right_panel = self.create_visualization_panel()
        splitter.addWidget(right_panel)
        splitter.setSizes([560, 840])
        main_layout.addWidget(splitter)
        self.resize(1400, 900)

    def create_configuration_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # Model Selection Section
        model_group = self.create_model_selection_group()
        layout.addWidget(model_group)

        # Formulation Builder Section
        formulation_group = self.create_formulation_builder_group()
        layout.addWidget(formulation_group)

        # Hypothesis Configuration Section
        hypothesis_group = self.create_hypothesis_configuration_group()
        layout.addWidget(hypothesis_group)

        # Action Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.run_test_btn = QtWidgets.QPushButton("Run Hypothesis Test")
        self.run_test_btn.setEnabled(False)
        self.run_test_btn.clicked.connect(self.run_hypothesis_test)
        self.clear_btn = QtWidgets.QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all)
        self.save_btn = QtWidgets.QPushButton("Save Results")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_results)
        button_layout.addWidget(self.run_test_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.save_btn)
        layout.addLayout(button_layout)
        layout.addStretch()

        return panel

    def create_model_selection_group(self) -> QtWidgets.QGroupBox:
        """Create model selection controls."""
        group = QtWidgets.QGroupBox("Model Selection")
        layout = QtWidgets.QVBoxLayout()

        # Model file selection
        model_layout = QtWidgets.QHBoxLayout()
        self.model_label = QtWidgets.QLabel("No model loaded")
        self.model_label.setWordWrap(True)

        self.select_model_btn = QtWidgets.QPushButton("Select Model")
        self.select_model_btn.clicked.connect(self.select_model)

        model_layout.addWidget(self.model_label, stretch=1)
        model_layout.addWidget(self.select_model_btn)

        layout.addLayout(model_layout)

        group.setLayout(layout)
        return group

    def create_formulation_builder_group(self) -> QtWidgets.QGroupBox:
        """Create formulation builder controls with type-specific ingredient selection, units, and temperature."""
        group = QtWidgets.QGroupBox("Formulation Builder")
        layout = QtWidgets.QVBoxLayout()

        # Create form layout for ingredient types and temperature
        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)

        # Store references to combo boxes and spin boxes
        self.ingredient_combos = {}
        self.concentration_spins = {}

        for ing_type in self.INGREDIENT_TYPES:
            row_layout = QtWidgets.QHBoxLayout()

            # Combo box for ingredient selection
            combo = QtWidgets.QComboBox()
            combo.setMinimumWidth(200)

            # Populate with ingredients of this type
            ingredients = self.ingredients_by_type.get(ing_type, [])
            if ingredients:
                combo.addItem(f"-- Select {ing_type} --", None)

                # Sort ingredients: 'none' first, then alphabetically
                sorted_ingredients = sorted(
                    ingredients,
                    key=lambda ing: (ing.name.lower() !=
                                     'none', ing.name.lower())
                )

                for ingredient in sorted_ingredients:
                    combo.addItem(ingredient.name, ingredient)
            else:
                combo.addItem(f"No {ing_type}s available", None)
                combo.setEnabled(False)

            # Concentration spin box with appropriate units
            spin = QtWidgets.QDoubleSpinBox()
            unit = self.INGREDIENT_UNITS[ing_type]

            if unit == 'M':
                spin.setRange(0.0, 10.0)
                spin.setDecimals(3)
                spin.setSingleStep(0.001)
            elif unit == 'mM':
                spin.setRange(0.0, 1000.0)
                spin.setDecimals(2)
                spin.setSingleStep(0.1)
            elif unit == 'mg/ml':
                spin.setRange(0.0, 500.0)
                spin.setDecimals(2)
                spin.setSingleStep(0.5)
            elif unit == '%w':
                spin.setRange(0.0, 100.0)
                spin.setDecimals(3)
                spin.setSingleStep(0.01)

            spin.setValue(0.0)
            spin.setSuffix(f" {unit}")
            spin.setMinimumWidth(120)

            combo.currentIndexChanged.connect(
                lambda idx, t=ing_type: self.on_ingredient_changed(t, idx))
            spin.valueChanged.connect(self.update_formulation)

            self.ingredient_combos[ing_type] = combo
            self.concentration_spins[ing_type] = spin

            row_layout.addWidget(combo, stretch=1)
            row_layout.addWidget(spin)

            label = QtWidgets.QLabel(f"{ing_type} ({unit}):")
            label.setStyleSheet("font-weight: bold;")
            form_layout.addRow(label, row_layout)

        temp_layout = QtWidgets.QHBoxLayout()
        self.temperature_spin = QtWidgets.QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 100.0)
        self.temperature_spin.setValue(25.0)
        self.temperature_spin.setSuffix(" °C")
        self.temperature_spin.setDecimals(1)
        self.temperature_spin.setSingleStep(0.5)
        self.temperature_spin.setMinimumWidth(120)
        self.temperature_spin.valueChanged.connect(self.update_formulation)

        temp_layout.addWidget(self.temperature_spin)
        temp_layout.addStretch()

        temp_label = QtWidgets.QLabel("Temperature (°C):")
        temp_label.setStyleSheet("font-weight: bold;")
        form_layout.addRow(temp_label, temp_layout)

        layout.addLayout(form_layout)
        group.setLayout(layout)
        return group

    def create_hypothesis_configuration_group(self) -> QtWidgets.QGroupBox:
        """Create hypothesis configuration controls."""
        group = QtWidgets.QGroupBox("Hypothesis Configuration")
        layout = QtWidgets.QVBoxLayout()

        # Info label
        info_label = QtWidgets.QLabel(
            "Note: Area-based testing evaluates the entire CI polygon across all shear rates"
        )
        info_label.setStyleSheet(
            "color: #0066cc; font-style: italic; padding: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Hypothesis type selection
        type_layout = QtWidgets.QHBoxLayout()
        type_layout.addWidget(QtWidgets.QLabel("Hypothesis Type:"))

        self.hypothesis_type_combo = QtWidgets.QComboBox()
        for key, label in self.HYPOTHESIS_TYPES.items():
            self.hypothesis_type_combo.addItem(label, key)
        self.hypothesis_type_combo.currentIndexChanged.connect(
            self.on_hypothesis_type_changed)

        type_layout.addWidget(self.hypothesis_type_combo, stretch=1)
        layout.addLayout(type_layout)

        # Value input (changes based on hypothesis type)
        self.value_widget = QtWidgets.QWidget()
        self.value_layout = QtWidgets.QFormLayout(self.value_widget)

        # Create all possible input widgets
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1000000.0)
        self.threshold_spin.setValue(280.0)
        self.threshold_spin.setSuffix(" cP")
        self.threshold_spin.setDecimals(2)

        self.min_spin = QtWidgets.QDoubleSpinBox()
        self.min_spin.setRange(0.0, 1000000.0)
        self.min_spin.setValue(150.0)
        self.min_spin.setSuffix(" cP")
        self.min_spin.setDecimals(2)

        self.max_spin = QtWidgets.QDoubleSpinBox()
        self.max_spin.setRange(0.0, 1000000.0)
        self.max_spin.setValue(280.0)
        self.max_spin.setSuffix(" cP")
        self.max_spin.setDecimals(2)

        layout.addWidget(self.value_widget)

        # Initialize with first hypothesis type
        self.on_hypothesis_type_changed(0)

        # Confidence level
        confidence_layout = QtWidgets.QHBoxLayout()
        confidence_layout.addWidget(QtWidgets.QLabel("Confidence Level:"))

        self.confidence_spin = QtWidgets.QDoubleSpinBox()
        self.confidence_spin.setRange(80.0, 99.0)
        self.confidence_spin.setValue(95.0)
        self.confidence_spin.setSuffix(" %")
        self.confidence_spin.setDecimals(1)
        self.confidence_spin.setSingleStep(1.0)

        confidence_layout.addWidget(self.confidence_spin)
        confidence_layout.addStretch()
        layout.addLayout(confidence_layout)

        group.setLayout(layout)
        return group

    def create_visualization_panel(self) -> QtWidgets.QWidget:
        """Create the right visualization panel."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # Tab widget for different visualizations
        self.viz_tabs = QtWidgets.QTabWidget()

        # Results Summary Tab
        summary_widget = self.create_results_summary_widget()
        self.viz_tabs.addTab(summary_widget, "Results Summary")

        # Viscosity Profile Tab
        profile_widget = self.create_viscosity_profile_widget()
        self.viz_tabs.addTab(profile_widget, "Viscosity Profile")

        layout.addWidget(self.viz_tabs)

        return panel

    def create_results_summary_widget(self) -> QtWidgets.QWidget:
        """Create the results summary widget."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # Hypothesis outcome display
        self.outcome_frame = QtWidgets.QFrame()
        self.outcome_frame.setFrameStyle(
            QtWidgets.QFrame.Box | QtWidgets.QFrame.Raised)
        self.outcome_frame.setLineWidth(2)
        outcome_layout = QtWidgets.QVBoxLayout(self.outcome_frame)

        self.outcome_label = QtWidgets.QLabel(
            "Run a hypothesis test to see results")
        self.outcome_label.setAlignment(Qt.AlignCenter)
        self.outcome_label.setStyleSheet(
            "font-size: 18pt; font-weight: bold; padding: 20px;")
        outcome_layout.addWidget(self.outcome_label)

        self.probability_label = QtWidgets.QLabel("")
        self.probability_label.setAlignment(Qt.AlignCenter)
        self.probability_label.setStyleSheet("font-size: 14pt; padding: 10px;")
        outcome_layout.addWidget(self.probability_label)

        layout.addWidget(self.outcome_frame)

        # Detailed results table
        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.verticalHeader().setVisible(False)
        layout.addWidget(self.results_table)

        return widget

    def create_viscosity_profile_widget(self) -> QtWidgets.QWidget:
        """Create the viscosity profile visualization widget."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # Matplotlib figure for viscosity profile
        self.profile_figure = Figure(figsize=(8, 6))
        self.profile_canvas = FigureCanvas(self.profile_figure)
        layout.addWidget(self.profile_canvas)

        # Controls for profile plot
        controls_layout = QtWidgets.QHBoxLayout()

        self.show_confidence_check = QtWidgets.QCheckBox(
            "Show CI Polygon")
        self.show_confidence_check.setChecked(True)
        self.show_confidence_check.stateChanged.connect(
            self.update_profile_plot)

        self.show_threshold_check = QtWidgets.QCheckBox(
            "Show Hypothesis Threshold")
        self.show_threshold_check.setChecked(True)
        self.show_threshold_check.stateChanged.connect(
            self.update_profile_plot)

        self.log_scale_check = QtWidgets.QCheckBox("Log Scale (X-axis)")
        self.log_scale_check.setChecked(True)
        self.log_scale_check.stateChanged.connect(self.update_profile_plot)

        controls_layout.addWidget(self.show_confidence_check)
        controls_layout.addWidget(self.show_threshold_check)
        controls_layout.addWidget(self.log_scale_check)
        controls_layout.addStretch()

        save_btn = QtWidgets.QPushButton("Save Plot")
        save_btn.clicked.connect(self.save_profile_plot)
        controls_layout.addWidget(save_btn)

        layout.addLayout(controls_layout)

        return widget

    def select_model(self) -> None:
        """Open a file dialog to select a VisQAI model and initialize the predictor."""
        if self.model_dialog.exec_():
            model_path = os.path.join(
                Architecture.get_path(), "QATCH/VisQAI/assets")
            if os.path.exists(model_path):
                self.model_dialog.setDirectory(model_path)
            else:
                self.model_dialog.setDirectory(Constants.log_prefer_path)

            selected_files = self.model_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                try:
                    self.model_path = file_path
                    self.predictor = Predictor(file_path)
                    self.hypothesis_tester = HypothesisTesting(file_path)
                    display_name = file_path.split(
                        '\\')[-1].split('/')[-1].split('.')[0]
                    self.model_label.setText(f"Model: {display_name}")
                    Log.i(TAG, f"Model loaded: {file_path}")
                    self.check_run_button_state()

                except Exception as e:
                    Log.e(TAG, f"Failed to load model: {e}")
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Model Loading Error",
                        f"Failed to load model: {str(e)}"
                    )
                    self.model_path = None
                    self.predictor = None
                    self.hypothesis_tester = None
                    self.model_label.setText("Failed to load model")

    def update_formulation(self) -> None:
        """Update the current formulation based on selected ingredients and concentrations."""
        self.current_formulation.clear()

        # Build formulation from selected ingredients with their units
        for ing_type in self.INGREDIENT_TYPES:
            combo = self.ingredient_combos[ing_type]
            spin = self.concentration_spins[ing_type]

            # Get selected ingredient
            ingredient = combo.currentData()
            concentration = spin.value()

            if ingredient is not None and concentration > 0:
                # Store with ingredient name as key and value with unit info
                self.current_formulation[ingredient.name] = {
                    'value': concentration,
                    'unit': self.INGREDIENT_UNITS[ing_type],
                    'type': ing_type
                }

        # Update temperature
        self.temperature = self.temperature_spin.value()

        # Check if run button should be enabled
        self.check_run_button_state()

    def on_ingredient_changed(self, ing_type: str, index: int) -> None:
        """Handle ingredient combo box changes and auto-set concentration to 0 if 'none' is selected."""
        combo = self.ingredient_combos[ing_type]
        spin = self.concentration_spins[ing_type]

        # Get the current ingredient
        current_ingredient = combo.currentData()

        # Check if 'none' or placeholder is selected
        is_none_or_placeholder = (
            current_ingredient is None or
            (hasattr(current_ingredient, 'name')
             and current_ingredient.name.lower() == 'none')
        )

        # If the selected item has None as data (the "Select" option)
        # or if ingredient name is 'none', set concentration to 0 and disable spin box
        if is_none_or_placeholder:
            spin.setValue(0.0)
            spin.setEnabled(False)
        else:
            spin.setEnabled(True)

        # Update formulation
        self.update_formulation()

    def on_hypothesis_type_changed(self, index: int) -> None:
        """Handle hypothesis type selection change."""
        # Clear existing layout
        while self.value_layout.count():
            child = self.value_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)

        hypothesis_type = self.hypothesis_type_combo.currentData()

        if hypothesis_type == 'upper':
            self.value_layout.addRow("Upper Threshold:", self.threshold_spin)

        elif hypothesis_type == 'lower':
            self.value_layout.addRow("Lower Threshold:", self.threshold_spin)

        elif hypothesis_type == 'between':
            self.value_layout.addRow("Lower Bound:", self.min_spin)
            self.value_layout.addRow("Upper Bound:", self.max_spin)

    def check_run_button_state(self) -> None:
        """Check if run button should be enabled."""
        has_model = self.hypothesis_tester is not None

        # Check that all required ingredient types are selected with non-zero concentration
        all_types_selected = True
        for ing_type in self.INGREDIENT_TYPES:
            ingredient = self.ingredient_combos[ing_type].currentData()
            concentration = self.concentration_spins[ing_type].value()

            # No ingredient selected (placeholder)
            if ingredient is None:
                all_types_selected = False
                break

            # Check if ingredient is 'none'
            is_none_ingredient = (hasattr(ingredient, 'name') and
                                  ingredient.name.lower() == 'none')

            # If not 'none' ingredient, concentration must be > 0
            if not is_none_ingredient and concentration <= 0:
                all_types_selected = False
                break

        # Temperature must be set (default is 25.0, so always valid)
        has_temperature = self.temperature is not None

        self.run_test_btn.setEnabled(
            has_model and all_types_selected and has_temperature)

    def run_hypothesis_test(self) -> None:
        """Run the hypothesis test using the area-based CI polygon containment approach."""
        try:
            # Build formulation object
            self.formulation_obj = self._build_formulation_object()

            if self.formulation_obj is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Formulation",
                    "Please ensure all required ingredients are selected."
                )
                return

            # Get hypothesis parameters
            hypothesis_type = self.hypothesis_type_combo.currentData()
            confidence_level = self.confidence_spin.value()

            # Calculate CI range from confidence level
            alpha = (100 - confidence_level) / 2.0
            ci_range = (alpha, 100 - alpha)

            # Determine bounds based on hypothesis type
            if hypothesis_type == 'upper':
                threshold = self.threshold_spin.value()
                bounds = (-np.inf, threshold)
            elif hypothesis_type == 'lower':
                threshold = self.threshold_spin.value()
                bounds = (threshold, np.inf)
            elif hypothesis_type == 'between':
                min_val = self.min_spin.value()
                max_val = self.max_spin.value()

                if min_val >= max_val:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Invalid Bounds",
                        "Lower bound must be less than upper bound."
                    )
                    return

                bounds = (min_val, max_val)
            else:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    "Unknown hypothesis type."
                )
                return

            # Run hypothesis test using area-based approach
            Log.i(
                TAG, f"Running hypothesis test: type={hypothesis_type}, bounds={bounds}, CI={ci_range}")

            result = self.hypothesis_tester.evaluate_hypothesis(
                formulation=self.formulation_obj,
                hypothesis_type=hypothesis_type,
                shear_rates=self.SHEAR_RATES,
                bounds=bounds,
                ci_range=ci_range
            )

            # Store results
            self.hypothesis_result = {
                'hypothesis_type': hypothesis_type,
                'bounds': bounds,
                'confidence_level': confidence_level,
                'ci_range': ci_range,
                'result': result
            }

            # Update visualizations
            self.update_results_summary()
            self.update_profile_plot()

            # Enable save button
            self.save_btn.setEnabled(True)

            Log.i(
                TAG, f"Hypothesis test completed: {result['pct_contained']:.2f}% contained")

        except Exception as e:
            Log.e(TAG, f"Error running hypothesis test: {e}")
            Log.e(TAG, traceback.format_exc())
            QtWidgets.QMessageBox.critical(
                self,
                "Hypothesis Test Error",
                f"Failed to run hypothesis test:\n{str(e)}"
            )

    def _build_formulation_object(self) -> Optional[Formulation]:
        """Build a Formulation object from the current formulation."""
        try:
            # Get ingredients
            protein = self.ingredient_combos['Protein'].currentData()
            buffer = self.ingredient_combos['Buffer'].currentData()
            surfactant = self.ingredient_combos['Surfactant'].currentData()
            stabilizer = self.ingredient_combos['Stabilizer'].currentData()
            excipient = self.ingredient_combos['Excipient'].currentData()
            salt = self.ingredient_combos['Salt'].currentData()

            # Get concentrations
            protein_conc = self.concentration_spins['Protein'].value()
            buffer_conc = self.concentration_spins['Buffer'].value()
            surfactant_conc = self.concentration_spins['Surfactant'].value()
            stabilizer_conc = self.concentration_spins['Stabilizer'].value()
            excipient_conc = self.concentration_spins['Excipient'].value()
            salt_conc = self.concentration_spins['Salt'].value()

            # Create formulation
            formulation = Formulation()
            formulation.set_protein(
                protein=protein, concentration=protein_conc, units=self.INGREDIENT_UNITS["Protein"])
            formulation.set_buffer(
                buffer=buffer, concentration=buffer_conc, units=self.INGREDIENT_UNITS["Buffer"])
            formulation.set_excipient(
                excipient=excipient, concentration=excipient_conc, units=self.INGREDIENT_UNITS["Excipient"])
            formulation.set_salt(
                salt=salt, concentration=salt_conc, units=self.INGREDIENT_UNITS["Salt"])
            formulation.set_stabilizer(
                stabilizer=stabilizer, concentration=stabilizer_conc, units=self.INGREDIENT_UNITS["Stabilizer"])
            formulation.set_surfactant(
                surfactant=surfactant, concentration=surfactant_conc, units=self.INGREDIENT_UNITS["Surfactant"])
            formulation.set_temperature(self.temperature)
            return formulation

        except Exception as e:
            Log.e(TAG, f"Failed to build formulation object: {e}")
            return None

    def update_results_summary(self) -> None:
        """Update the results summary display with area-based metrics."""
        if self.hypothesis_result is None:
            return

        result = self.hypothesis_result['result']
        hypothesis_type = self.hypothesis_result['hypothesis_type']
        bounds = self.hypothesis_result['bounds']
        pct_contained = result['pct_contained']

        # Update outcome label
        if pct_contained >= 90:
            outcome_text = "HIGH CONFIDENCE"
            color = "#28a745"  # Green
            interpretation = "The formulation is very likely to meet specifications"
        elif pct_contained >= 70:
            outcome_text = "MODERATE CONFIDENCE"
            color = "#ffc107"  # Yellow/Orange
            interpretation = "The formulation likely meets specifications"
        elif pct_contained >= 50:
            outcome_text = "LOW CONFIDENCE"
            color = "#ff8c00"  # Orange
            interpretation = "The formulation may or may not meet specifications"
        else:
            outcome_text = "VERY LOW CONFIDENCE"
            color = "#dc3545"  # Red
            interpretation = "The formulation is unlikely to meet specifications"

        self.outcome_label.setText(outcome_text)
        self.outcome_frame.setStyleSheet(
            f"background-color: {color}; border-radius: 10px;")
        self.probability_label.setText(
            f"{pct_contained:.1f}% of CI polygon within bounds\n{interpretation}")

        # Build hypothesis statement
        if hypothesis_type == 'upper':
            hyp_statement = f"Viscosity < {bounds[1]:.2f} cP"
        elif hypothesis_type == 'lower':
            hyp_statement = f"Viscosity > {bounds[0]:.2f} cP"
        elif hypothesis_type == 'between':
            hyp_statement = f"{bounds[0]:.2f} ≤ Viscosity ≤ {bounds[1]:.2f} cP"

        # Populate results table
        self.results_table.setRowCount(0)

        rows = [
            ("Hypothesis", hyp_statement),
            ("Test Type", hypothesis_type.title()),
            ("Confidence Level",
             f"{self.hypothesis_result['confidence_level']:.1f}%"),
            ("CI Range",
             f"{self.hypothesis_result['ci_range'][0]:.3f} - {self.hypothesis_result['ci_range'][1]:.3f}"),
            ("", ""),  # Separator
            ("% CI Contained", f"{pct_contained:.2f}%"),
            ("Area Contained",
             f"{result['area_contained']:.4f} (log₁₀[1/s] × cP)"),
            ("Total CI Area", f"{result['total_area']:.4f} (log₁₀[1/s] × cP)"),
            ("", ""),  # Separator
            ("Mean Predictions:", "")
        ]

        # Add mean predictions for each shear rate
        for sr, pred in result['mean_predictions'].items():
            rows.append((f"  @ {sr:,} 1/s", f"{pred:.2f} cP"))

        self.results_table.setRowCount(len(rows))
        for i, (metric, value) in enumerate(rows):
            metric_item = QtWidgets.QTableWidgetItem(metric)
            value_item = QtWidgets.QTableWidgetItem(value)

            # Make separators and headers bold
            if metric == "" or metric == "Mean Predictions:":
                font = metric_item.font()
                font.setBold(True)
                metric_item.setFont(font)
                value_item.setFont(font)

            self.results_table.setItem(i, 0, metric_item)
            self.results_table.setItem(i, 1, value_item)

        self.results_table.resizeColumnsToContents()

    def update_profile_plot(self) -> None:
        """Update the viscosity profile plot showing the CI polygon and bounds."""
        if self.hypothesis_result is None:
            return

        self.profile_figure.clear()
        ax = self.profile_figure.add_subplot(111)

        result = self.hypothesis_result['result']
        mean_preds = result['mean_predictions']
        bounds = self.hypothesis_result['bounds']

        # Get CI bounds from predictor
        try:
            ci_range = self.hypothesis_result['ci_range']
            _, pred_stats = self.predictor.predict_uncertainty(
                df=self.formulation_obj.to_dataframe(
                    encoded=False, training=False),
                ci_range=ci_range
            )
            lower_ci = pred_stats['lower_ci']
            upper_ci = pred_stats['upper_ci']
            upper_ci = upper_ci.flatten()
            lower_ci = lower_ci.flatten()
        except Exception as e:
            Log.e(TAG, f"Failed to get CI bounds: {e}")
            return

        shear_rates = self.SHEAR_RATES
        mean_values = [mean_preds[sr] for sr in shear_rates]

        # Plot mean prediction
        if self.log_scale_check.isChecked():
            ax.semilogx(shear_rates, mean_values, 'bo-', linewidth=2,
                        markersize=8, label='Mean Prediction', zorder=5)
        else:
            ax.plot(shear_rates, mean_values, 'bo-', linewidth=2,
                    markersize=8, label='Mean Prediction', zorder=5)

        # Show CI polygon
        if self.show_confidence_check.isChecked():
            if self.log_scale_check.isChecked():
                polygon_points = []

                # Upper boundary (left to right)
                for i, sr in enumerate(shear_rates):
                    # [x, y] not (x, y)
                    polygon_points.append([sr, upper_ci[i]])

                # Lower boundary (right to left)
                for i in range(len(shear_rates) - 1, -1, -1):
                    polygon_points.append([shear_rates[i], lower_ci[i]])

                # Explicitly convert to numpy array with float type
                polygon_points = np.array(polygon_points, dtype=np.float64)
        # Show hypothesis threshold
        if self.show_threshold_check.isChecked():
            if bounds[0] > -np.inf:
                ax.axhline(y=bounds[0], color='red', linestyle='--',
                           linewidth=2, label=f'Lower Bound ({bounds[0]:.1f} cP)', zorder=3)
            if bounds[1] < np.inf:
                ax.axhline(y=bounds[1], color='red', linestyle='--',
                           linewidth=2, label=f'Upper Bound ({bounds[1]:.1f} cP)', zorder=3)

        ax.set_xlabel('Shear Rate (1/s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Viscosity (cP)', fontsize=12, fontweight='bold')
        ax.set_title('Viscosity Profile with CI Polygon',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        self.profile_canvas.draw()

    def clear_all(self) -> None:
        """Clear all inputs and results."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear All",
            "Are you sure you want to clear all inputs and results?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            # Reset formulation inputs
            for ing_type in self.INGREDIENT_TYPES:
                self.ingredient_combos[ing_type].setCurrentIndex(0)
                self.concentration_spins[ing_type].setValue(0.0)

            # Reset temperature to default
            self.temperature_spin.setValue(25.0)

            # Clear formulation
            self.current_formulation.clear()
            self.formulation_obj = None
            self.temperature = None

            # Clear results
            self.hypothesis_result = None

            # Reset visualizations
            self.outcome_label.setText("Run a hypothesis test to see results")
            self.outcome_label.setStyleSheet(
                "font-size: 18pt; font-weight: bold; padding: 20px;"
            )
            self.outcome_frame.setStyleSheet("")
            self.probability_label.setText("")
            self.results_table.setRowCount(0)

            self.profile_figure.clear()
            self.profile_canvas.draw()

            self.save_btn.setEnabled(False)
            self.check_run_button_state()

    def save_results(self) -> None:
        """Save hypothesis test results."""
        if not self.hypothesis_result:
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "hypothesis_test_results.json",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            try:
                result = self.hypothesis_result['result']

                # Build hypothesis statement
                hypothesis_type = self.hypothesis_result['hypothesis_type']
                bounds = self.hypothesis_result['bounds']

                if hypothesis_type == 'upper':
                    statement = f"Viscosity < {bounds[1]:.2f} cP across all shear rates"
                elif hypothesis_type == 'lower':
                    statement = f"Viscosity > {bounds[0]:.2f} cP across all shear rates"
                elif hypothesis_type == 'between':
                    statement = f"{bounds[0]:.2f} ≤ Viscosity ≤ {bounds[1]:.2f} cP across all shear rates"

                # Prepare data
                data = {
                    'formulation': self.current_formulation,
                    'temperature': self.temperature,
                    'hypothesis': {
                        'type': hypothesis_type,
                        'bounds': bounds,
                        'confidence_level': self.hypothesis_result['confidence_level'],
                        'ci_range': self.hypothesis_result['ci_range'],
                        'statement': statement
                    },
                    'results': {
                        'pct_contained': result['pct_contained'],
                        'area_contained': result['area_contained'],
                        'total_area': result['total_area'],
                        'mean_predictions': result['mean_predictions'],
                        'test_type': result['test_type']
                    }
                }

                # Save to file
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)

                QtWidgets.QMessageBox.information(
                    self,
                    "Success",
                    f"Results saved successfully to:\n{file_path}"
                )

            except Exception as e:
                Log.e(TAG, f"Failed to save results: {e}")
                QtWidgets.QMessageBox.critical(
                    self,
                    "Save Error",
                    f"Failed to save results:\n{str(e)}"
                )

    def save_profile_plot(self) -> None:
        """Save the viscosity profile plot."""
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Profile Plot",
            "viscosity_profile.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
        )

        if file_path:
            try:
                self.profile_figure.savefig(
                    file_path, dpi=300, bbox_inches='tight')
                QtWidgets.QMessageBox.information(
                    self,
                    "Success",
                    f"Plot saved successfully to:\n{file_path}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save plot:\n{str(e)}"
                )


# Example usage
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = HypothesisTestingUI()
    window.show()
    sys.exit(app.exec_())
