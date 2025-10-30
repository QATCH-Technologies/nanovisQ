"""
hypothesis_testing_ui.py

This module provides the HypothesisTestingUI class for the Hypothesis Testing tab in VisQAI.
It supports creating custom formulations, defining hypotheses about viscosity behavior,
and visualizing prediction results with hypothesis outcomes.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-10-28

Version:
   1.0
"""

import sys
import os
import traceback
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING
import json
from scipy import stats

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
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from src.models.formulation import Formulation
    from src.models.predictor import Predictor
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.models.ingredient import Protein, Buffer, Salt, Stabilizer, Surfactant, Excipient
    if TYPE_CHECKING:
        from src.view.main_window import VisQAIWindow
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.formulation import Formulation
    from QATCH.VisQAI.src.models.predictor import Predictor
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
    - Run model predictions
    - Visualize results with hypothesis outcomes
    """

    HYPOTHESIS_TYPES = {
        'less_than': 'Less Than',
        'greater_than': 'Greater Than',
        'between': 'Between',
        'within_range': 'Within ±X%'
    }

    SHEAR_RATES = {
        'Viscosity_100': 100,
        'Viscosity_1000': 1000,
        'Viscosity_10000': 10000,
        'Viscosity_100000': 100000,
        'Viscosity_15000000': 15000000
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
        self.setWindowTitle("Hypothesis Testing")

        # Database and controllers
        self.db: Optional[Database] = None
        self.ing_ctrl: Optional[IngredientController] = None

        # Core components
        self.predictor: Optional[Predictor] = None
        self.model_path: Optional[str] = None

        # Current formulation and results
        self.current_formulation: Optional[Dict[str, float]] = {}
        self.prediction_results: Optional[Dict] = None
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

            # Sort each type by name
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
        self.run_test_btn.setEnabled(True)
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

        # Shear rate selection
        shear_layout = QtWidgets.QHBoxLayout()
        shear_layout.addWidget(QtWidgets.QLabel("Shear Rate:"))

        self.shear_rate_combo = QtWidgets.QComboBox()
        self.shear_rate_combo.addItem("All Shear Rates", "all")
        for visc_label, rate in self.SHEAR_RATES.items():
            self.shear_rate_combo.addItem(f"{rate:,} 1/s", visc_label)

        shear_layout.addWidget(self.shear_rate_combo, stretch=1)
        layout.addLayout(shear_layout)

        # Value input (changes based on hypothesis type)
        self.value_widget = QtWidgets.QWidget()
        self.value_layout = QtWidgets.QFormLayout(self.value_widget)

        # Create all possible input widgets
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1000000.0)
        self.threshold_spin.setValue(0.0)
        self.threshold_spin.setSuffix(" cP")
        self.threshold_spin.setDecimals(2)

        self.min_spin = QtWidgets.QDoubleSpinBox()
        self.min_spin.setRange(0.0, 1000000.0)
        self.min_spin.setValue(0.0)
        self.min_spin.setSuffix(" cP")
        self.min_spin.setDecimals(2)

        self.max_spin = QtWidgets.QDoubleSpinBox()
        self.max_spin.setRange(0.0, 1000000.0)
        self.max_spin.setValue(0.0)
        self.max_spin.setSuffix(" cP")
        self.max_spin.setDecimals(2)

        self.range_spin = QtWidgets.QDoubleSpinBox()
        self.range_spin.setRange(0.0, 100.0)
        self.range_spin.setValue(0.0)
        self.range_spin.setSuffix(" %")
        self.range_spin.setDecimals(1)

        self.target_spin = QtWidgets.QDoubleSpinBox()
        self.target_spin.setRange(0.0, 1000000.0)
        self.target_spin.setValue(0.0)
        self.target_spin.setSuffix(" cP")
        self.target_spin.setDecimals(2)

        layout.addWidget(self.value_widget)

        # Initialize with first hypothesis type
        self.on_hypothesis_type_changed(0)

        # Confidence level
        confidence_layout = QtWidgets.QHBoxLayout()
        confidence_layout.addWidget(QtWidgets.QLabel("Confidence Level:"))

        self.confidence_spin = QtWidgets.QDoubleSpinBox()
        self.confidence_spin.setRange(50.0, 99.0)
        self.confidence_spin.setValue(95.0)
        self.confidence_spin.setSuffix(" %")
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

        # Probability Distribution Tab
        probability_widget = self.create_probability_widget()
        self.viz_tabs.addTab(probability_widget, "Probability Distribution")

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
            "Show Confidence Intervals")
        self.show_confidence_check.setChecked(True)
        self.show_confidence_check.stateChanged.connect(
            self.update_profile_plot)

        self.show_threshold_check = QtWidgets.QCheckBox(
            "Show Hypothesis Threshold")
        self.show_threshold_check.setChecked(True)
        self.show_threshold_check.stateChanged.connect(
            self.update_profile_plot)

        self.log_scale_check = QtWidgets.QCheckBox("Log Scale")
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

    def create_probability_widget(self) -> QtWidgets.QWidget:
        """Create the probability distribution widget."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # Matplotlib figure for probability distribution
        self.probability_figure = Figure(figsize=(8, 6))
        self.probability_canvas = FigureCanvas(self.probability_figure)
        layout.addWidget(self.probability_canvas)

        # Controls
        controls_layout = QtWidgets.QHBoxLayout()

        controls_layout.addWidget(QtWidgets.QLabel("Select Shear Rate:"))

        self.prob_shear_combo = QtWidgets.QComboBox()
        for visc_label, rate in self.SHEAR_RATES.items():
            self.prob_shear_combo.addItem(f"{rate:,} 1/s", visc_label)
        self.prob_shear_combo.currentIndexChanged.connect(
            self.update_probability_plot)

        controls_layout.addWidget(self.prob_shear_combo, stretch=1)
        controls_layout.addStretch()

        save_btn = QtWidgets.QPushButton("Save Plot")
        save_btn.clicked.connect(self.save_probability_plot)
        controls_layout.addWidget(save_btn)

        layout.addLayout(controls_layout)

        return widget

    # Event Handlers

    def select_model(self) -> None:
        """Open a file dialog to select a VisQAI model and initialize the predictor.

        This method allows the user to browse and select a trained VisQAI model
        archive (ZIP file). Upon selection, it attempts to initialize a `Predictor`
        instance with the chosen model and updates the UI to reflect the loaded model.
        """
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

        if hypothesis_type == 'less_than':
            self.value_layout.addRow("Threshold:", self.threshold_spin)

        elif hypothesis_type == 'greater_than':
            self.value_layout.addRow("Threshold:", self.threshold_spin)

        elif hypothesis_type == 'between':
            self.value_layout.addRow("Minimum:", self.min_spin)
            self.value_layout.addRow("Maximum:", self.max_spin)

        elif hypothesis_type == 'within_range':
            self.value_layout.addRow("Target:", self.target_spin)
            self.value_layout.addRow("Range (±):", self.range_spin)

    def check_run_button_state(self) -> None:
        """Check if run button should be enabled."""
        has_model = self.predictor is not None

        # Check that all required ingredient types are selected with non-zero concentration
        # (unless 'none' is selected, which can have zero concentration)
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
        """Run the hypothesis test with current configuration."""
        try:
            QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

            # Create formulation object
            formulation = Formulation()

            # Helper function to set ingredient only if not 'none'
            def set_ingredient_if_valid(set_method, ingredient, ing_type):
                if ingredient and hasattr(ingredient, 'name') and ingredient.name.lower() != 'none':
                    concentration = self.current_formulation[ingredient.name]['value']
                    units = self.current_formulation[ingredient.name]['unit']
                    set_method(ingredient,
                               concentration=concentration, units=units)
                else:
                    # Set with None for 'none' ingredients
                    type_map = {
                        "Buffer": Buffer(enc_id=-1, name="None"),
                        "Protein": Protein(enc_id=-1, name="None"),
                        "Surfactant": Surfactant(enc_id=-1, name="None"),
                        "Stabilizer": Stabilizer(enc_id=-1, name="None"),
                        "Salt": Salt(enc_id=-1, name="None"),
                        "Excipient": Excipient(enc_id=-1, name="None"),
                    }
                    obj = self.ing_ctrl.get_by_name("none", type_map[ing_type])
                    set_method(obj, concentration=0.0,
                               units=self.INGREDIENT_UNITS[ing_type])

            protein = self.ingredient_combos["Protein"].currentData()
            set_ingredient_if_valid(
                formulation.set_protein, protein, "Protein")

            buffer = self.ingredient_combos["Buffer"].currentData()
            set_ingredient_if_valid(formulation.set_buffer, buffer, "Buffer")

            surfactant = self.ingredient_combos["Surfactant"].currentData()
            set_ingredient_if_valid(
                formulation.set_surfactant, surfactant, "Surfactant")

            stabilizer = self.ingredient_combos["Stabilizer"].currentData()
            set_ingredient_if_valid(
                formulation.set_stabilizer, stabilizer, "Stabilizer")

            excipient = self.ingredient_combos["Excipient"].currentData()
            set_ingredient_if_valid(
                formulation.set_excipient, excipient, "Excipient")

            salt = self.ingredient_combos["Salt"].currentData()
            set_ingredient_if_valid(formulation.set_salt, salt, "Salt")

            formulation.set_temperature(self.temperature)

            # Run predictions for CI levels from 50% to 100% in 1% steps
            self.prediction_results = {}
            formulation_df = formulation.to_dataframe(
                encoded=False, training=False)

            for confidence_level in range(50, 101):
                ci_lower = (100 - confidence_level) / 2
                ci_upper = 100 - ci_lower
                ci_range = (ci_lower, ci_upper)

                pred_means, pred_stats = self.predictor.predict_uncertainty(
                    formulation_df,
                    ci_range=ci_range)

                # Extract the bounds - the keys might be 'lower_ci', 'upper_ci' or similar
                # We need to find the actual keys being returned
                # Typically they will be named based on the CI range or have generic names

                # Store with standardized keys for this confidence level
                standardized_stats = {
                    'std': pred_stats['std'],
                    'coefficient_of_variation': pred_stats['coefficient_of_variation'],
                }

                # Find the lower and upper bound keys (they might vary)
                # Common patterns: 'lower_ci', 'upper_ci', 'lower', 'upper', 'lower_ci', 'upper_ci'
                lower_key = None
                upper_key = None

                for key in pred_stats.keys():
                    if 'lower' in key.lower():
                        lower_key = key
                    if 'upper' in key.lower():
                        upper_key = key

                if lower_key and upper_key:
                    standardized_stats[f'lower_{confidence_level}'] = pred_stats[lower_key]
                    standardized_stats[f'upper_{confidence_level}'] = pred_stats[upper_key]
                else:
                    # Fallback: calculate from mean and std using normal distribution
                    # This should match what predict_uncertainty does internally
                    z_score = stats.norm.ppf(ci_upper / 100)
                    standardized_stats[f'lower_{confidence_level}'] = pred_means - \
                        z_score * pred_stats['std']
                    standardized_stats[f'upper_{confidence_level}'] = pred_means + \
                        z_score * pred_stats['std']

                self.prediction_results[confidence_level] = {
                    'means': pred_means,
                    'stats': standardized_stats
                }

            # Evaluate hypothesis across all confidence levels
            self.hypothesis_result = self.evaluate_hypothesis()

            # Update visualizations
            self.update_results_summary()
            self.update_profile_plot()
            self.update_probability_plot()

            # Enable save button
            self.save_btn.setEnabled(True)

            # Switch to results tab
            self.viz_tabs.setCurrentIndex(0)

        except Exception as e:
            Log.e(TAG, f"Hypothesis test failed: {e}")
            Log.e(TAG, traceback.format_exc())
            QtWidgets.QMessageBox.critical(
                self,
                "Test Error",
                f"Failed to run hypothesis test:\n{str(e)}"
            )

        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def evaluate_hypothesis(self) -> Dict:
        """Evaluate the hypothesis based on predictions using CI bounds across all confidence levels."""
        hypothesis_type = self.hypothesis_type_combo.currentData()
        shear_rate_data = self.shear_rate_combo.currentData()

        result = {
            'hypothesis_type': hypothesis_type,
            'shear_rate': shear_rate_data,
            'max_confidence_level': 0,  # Maximum confidence level at which hypothesis passes
            'predictions_by_ci': {},
            'details': {}
        }

        # Get predictions for relevant shear rate(s)
        if shear_rate_data == "all":
            shear_rates = list(self.SHEAR_RATES.keys())
        else:
            shear_rates = [shear_rate_data]

        # Evaluate based on hypothesis type
        if hypothesis_type == 'less_than':
            threshold = self.threshold_spin.value()
            result['details']['threshold'] = threshold
            result = self._evaluate_less_than_ci_profile(
                shear_rates, threshold, result)

        elif hypothesis_type == 'greater_than':
            threshold = self.threshold_spin.value()
            result['details']['threshold'] = threshold
            result = self._evaluate_greater_than_ci_profile(
                shear_rates, threshold, result)

        elif hypothesis_type == 'between':
            min_val = self.min_spin.value()
            max_val = self.max_spin.value()
            result['details']['min'] = min_val
            result['details']['max'] = max_val
            result = self._evaluate_between_ci_profile(
                shear_rates, min_val, max_val, result)

        elif hypothesis_type == 'within_range':
            target = self.target_spin.value()
            range_pct = self.range_spin.value()
            lower = target * (1 - range_pct / 100)
            upper = target * (1 + range_pct / 100)
            result['details']['target'] = target
            result['details']['range_pct'] = range_pct
            result['details']['lower'] = lower
            result['details']['upper'] = upper
            result = self._evaluate_between_ci_profile(
                shear_rates, lower, upper, result)

        return result

    def _evaluate_less_than_ci_profile(self, shear_rates: List[str], threshold: float,
                                       result: Dict) -> Dict:
        """Evaluate 'less than' hypothesis at the selected CI level.

        Returns percentage of CI interval that falls below the threshold for each shear rate.
        """
        # Get the selected confidence level from the spin box
        confidence_level = self.confidence_spin.value()

        pred_data = self.prediction_results[confidence_level]
        pred_means = pred_data['means']
        pred_stats = pred_data['stats']

        mean_values = pred_means.flatten()
        std_values = pred_stats["std"].flatten()
        upper_values = pred_stats[f"upper_{int(confidence_level)}"].flatten()
        lower_values = pred_stats[f"lower_{int(confidence_level)}"].flatten()

        predictions_at_ci = {}
        percentages = []

        for i, sr in enumerate(shear_rates):
            mean = float(mean_values[i])
            std = float(std_values[i])
            upper_ci = float(upper_values[i])
            lower_ci = float(lower_values[i])

            # Calculate percentage of CI interval below threshold
            if upper_ci < threshold:
                # Entire CI is below threshold -> 100%
                percentage = 100.0
                passes = True
            elif lower_ci >= threshold:
                # Entire CI is above threshold -> 0%
                percentage = 0.0
                passes = False
            else:
                # Threshold bisects the CI interval
                # Calculate what fraction of CI is below threshold
                ci_width = upper_ci - lower_ci
                if ci_width > 0:
                    below_threshold = threshold - lower_ci
                    percentage = (below_threshold / ci_width) * 100.0
                else:
                    percentage = 50.0  # Edge case: CI has zero width
                passes = percentage >= 50.0

            percentages.append(percentage)

            predictions_at_ci[sr] = {
                'mean': mean,
                'std': std,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'passes': passes,
                'percentage': percentage
            }

        # Store results
        result['confidence_level'] = confidence_level
        result['predictions_by_ci'][confidence_level] = predictions_at_ci
        result['percentages_by_shear_rate'] = {
            sr: pct for sr, pct in zip(shear_rates, percentages)}

        # Overall percentage is the average across all shear rates
        result['average_percentage'] = sum(
            percentages) / len(percentages) if percentages else 0.0
        result['max_confidence_level'] = result['average_percentage']
        result['passed'] = result['average_percentage'] >= 50.0

        return result

    def _evaluate_greater_than_ci_profile(self, shear_rates: List[str], threshold: float,
                                          result: Dict) -> Dict:
        """Evaluate 'greater than' hypothesis at the selected CI level.

        Returns percentage of CI interval that falls above the threshold for each shear rate.
        """
        # Get the selected confidence level from the spin box
        confidence_level = self.confidence_spin.value()

        pred_data = self.prediction_results[confidence_level]
        pred_means = pred_data['means']
        pred_stats = pred_data['stats']

        mean_values = pred_means.flatten()
        std_values = pred_stats["std"].flatten()
        upper_values = pred_stats[f"upper_{int(confidence_level)}"].flatten()
        lower_values = pred_stats[f"lower_{int(confidence_level)}"].flatten()

        predictions_at_ci = {}
        percentages = []

        for i, sr in enumerate(shear_rates):
            mean = float(mean_values[i])
            std = float(std_values[i])
            upper_ci = float(upper_values[i])
            lower_ci = float(lower_values[i])

            # Calculate percentage of CI interval above threshold
            if lower_ci > threshold:
                # Entire CI is above threshold -> 100%
                percentage = 100.0
                passes = True
            elif upper_ci <= threshold:
                # Entire CI is below threshold -> 0%
                percentage = 0.0
                passes = False
            else:
                # Threshold bisects the CI interval
                # Calculate what fraction of CI is above threshold
                ci_width = upper_ci - lower_ci
                if ci_width > 0:
                    above_threshold = upper_ci - threshold
                    percentage = (above_threshold / ci_width) * 100.0
                else:
                    percentage = 50.0  # Edge case: CI has zero width
                passes = percentage >= 50.0

            percentages.append(percentage)

            predictions_at_ci[sr] = {
                'mean': mean,
                'std': std,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'passes': passes,
                'percentage': percentage
            }

        # Store results
        result['confidence_level'] = confidence_level
        result['predictions_by_ci'][confidence_level] = predictions_at_ci
        result['percentages_by_shear_rate'] = {
            sr: pct for sr, pct in zip(shear_rates, percentages)}

        # Overall percentage is the average across all shear rates
        result['average_percentage'] = sum(
            percentages) / len(percentages) if percentages else 0.0
        result['max_confidence_level'] = result['average_percentage']
        result['passed'] = result['average_percentage'] >= 50.0

        return result

    def _evaluate_between_ci_profile(self, shear_rates: List[str], min_val: float,
                                     max_val: float, result: Dict) -> Dict:
        """Evaluate 'between' or 'within range' hypothesis at the selected CI level.

        Returns percentage of CI interval that falls within [min_val, max_val] for each shear rate.
        """
        # Get the selected confidence level from the spin box
        confidence_level = self.confidence_spin.value()

        pred_data = self.prediction_results[confidence_level]
        pred_means = pred_data['means']
        pred_stats = pred_data['stats']

        mean_values = pred_means.flatten()
        std_values = pred_stats["std"].flatten()
        upper_values = pred_stats[f"upper_{int(confidence_level)}"].flatten()
        lower_values = pred_stats[f"lower_{int(confidence_level)}"].flatten()

        predictions_at_ci = {}
        percentages = []

        for i, sr in enumerate(shear_rates):
            mean = float(mean_values[i])
            std = float(std_values[i])
            upper_ci = float(upper_values[i])
            lower_ci = float(lower_values[i])

            # Calculate percentage of CI interval within [min_val, max_val]
            if lower_ci >= min_val and upper_ci <= max_val:
                # Entire CI is within range -> 100%
                percentage = 100.0
                passes = True
            elif upper_ci < min_val or lower_ci > max_val:
                # Entire CI is outside range -> 0%
                percentage = 0.0
                passes = False
            else:
                # Range bisects the CI interval
                # Calculate what fraction of CI is within [min_val, max_val]
                ci_width = upper_ci - lower_ci
                if ci_width > 0:
                    # Find the overlap between CI interval and target range
                    overlap_lower = max(lower_ci, min_val)
                    overlap_upper = min(upper_ci, max_val)
                    overlap_width = max(0, overlap_upper - overlap_lower)
                    percentage = (overlap_width / ci_width) * 100.0
                else:
                    # Edge case: CI has zero width, check if it's in range
                    percentage = 100.0 if (min_val <= mean <= max_val) else 0.0
                passes = percentage >= 50.0

            percentages.append(percentage)

            predictions_at_ci[sr] = {
                'mean': mean,
                'std': std,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'passes': passes,
                'percentage': percentage
            }

        # Store results
        result['confidence_level'] = confidence_level
        result['predictions_by_ci'][confidence_level] = predictions_at_ci
        result['percentages_by_shear_rate'] = {
            sr: pct for sr, pct in zip(shear_rates, percentages)}

        # Overall percentage is the average across all shear rates
        result['average_percentage'] = sum(
            percentages) / len(percentages) if percentages else 0.0
        result['max_confidence_level'] = result['average_percentage']
        result['passed'] = result['average_percentage'] >= 50.0

        return result

    def _add_result_row(self, metric: str, value: str) -> None:
        """Add a row to the results table."""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)

        metric_item = QtWidgets.QTableWidgetItem(metric)
        metric_item.setFlags(metric_item.flags() & ~Qt.ItemIsEditable)
        metric_item.setFont(QtGui.QFont("", -1, QtGui.QFont.Bold))

        value_item = QtWidgets.QTableWidgetItem(value)
        value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)

        self.results_table.setItem(row, 0, metric_item)
        self.results_table.setItem(row, 1, value_item)

    def update_results_summary(self) -> None:
        """Update the results summary display with hypothesis test outcomes."""
        if not self.hypothesis_result:
            return

        # Clear existing results
        self.results_table.setRowCount(0)

        # Get hypothesis information
        hypothesis_type = self.hypothesis_result['hypothesis_type']
        shear_rate_data = self.hypothesis_result['shear_rate']
        max_confidence_level = self.hypothesis_result['max_confidence_level']
        passed = self.hypothesis_result['passed']
        details = self.hypothesis_result['details']

        # Update outcome label with confidence level
        # Update outcome label with average percentage
        if passed:
            self.outcome_label.setText(
                f"✓ HYPOTHESIS PASSED\n({max_confidence_level:.1f}% average confidence)")
            self.outcome_frame.setStyleSheet(
                "QFrame { background-color: #d4edda; border: 2px solid #28a745; border-radius: 5px; }"
            )
            self.outcome_label.setStyleSheet(
                "font-size: 16pt; font-weight: bold; padding: 20px; color: #155724;"
            )
        else:
            self.outcome_label.setText(
                f"✗ HYPOTHESIS FAILED\n({max_confidence_level:.1f}% average confidence)")
            self.outcome_frame.setStyleSheet(
                "QFrame { background-color: #f8d7da; border: 2px solid #dc3545; border-radius: 5px; }"
            )
            self.outcome_label.setStyleSheet(
                "font-size: 16pt; font-weight: bold; padding: 20px; color: #721c24;"
            )

        # Update probability label
        confidence_level = self.hypothesis_result.get('confidence_level', 95)
        self.probability_label.setText(
            f"Evaluated at {confidence_level}% CI | Average: {max_confidence_level:.1f}%")
        # Build hypothesis statement
        if shear_rate_data == "all":
            shear_text = "all shear rates"
        else:
            shear_rate_value = self.SHEAR_RATES[shear_rate_data]
            shear_text = f"shear rate of {shear_rate_value:,} 1/s"

        # Add hypothesis statement to table
        if hypothesis_type == 'less_than':
            threshold = details['threshold']
            statement = f"Viscosity at {shear_text} < {threshold:.2f} cP"
        elif hypothesis_type == 'greater_than':
            threshold = details['threshold']
            statement = f"Viscosity at {shear_text} > {threshold:.2f} cP"
        elif hypothesis_type == 'between':
            min_val = details['min']
            max_val = details['max']
            statement = f"Viscosity at {shear_text} between {min_val:.2f} and {max_val:.2f} cP"
        elif hypothesis_type == 'within_range':
            target = details['target']
            range_pct = details['range_pct']
            lower = details['lower']
            upper = details['upper']
            statement = f"Viscosity at {shear_text} within ±{range_pct:.1f}% of {target:.2f} cP ({lower:.2f}-{upper:.2f} cP)"
        else:
            statement = "Unknown hypothesis type"

        self._add_result_row("Hypothesis", statement)
        self._add_result_row("Maximum Confidence Level",
                             f"{max_confidence_level}%")
        self._add_result_row(
            "Overall Result", "PASSED" if passed else "FAILED")

        # Add separator
        separator_row = self.results_table.rowCount()
        self.results_table.insertRow(separator_row)
        separator_item = QtWidgets.QTableWidgetItem("─" * 50)
        separator_item.setFlags(separator_item.flags() & ~Qt.ItemIsEditable)
        self.results_table.setItem(separator_row, 0, separator_item)
        self.results_table.setSpan(separator_row, 0, 1, 2)

        # Add per-shear-rate results at the confidence level used for evaluation
        if shear_rate_data == "all":
            shear_rates = list(self.SHEAR_RATES.keys())
        else:
            shear_rates = [shear_rate_data]

        # Use the actual confidence level that was used for evaluation
        display_ci = self.hypothesis_result['confidence_level']
        predictions_at_ci = self.hypothesis_result['predictions_by_ci'].get(
            display_ci, {})

        for sr in shear_rates:
            shear_rate_value = self.SHEAR_RATES[sr]

            # Add shear rate header
            header_row = self.results_table.rowCount()
            self.results_table.insertRow(header_row)
            header_item = QtWidgets.QTableWidgetItem(
                f"Shear Rate: {shear_rate_value:,} 1/s")
            header_item.setFlags(header_item.flags() & ~Qt.ItemIsEditable)
            header_item.setFont(QtGui.QFont("", -1, QtGui.QFont.Bold))
            header_item.setBackground(QtGui.QColor(240, 240, 240))
            self.results_table.setItem(header_row, 0, header_item)
            self.results_table.setSpan(header_row, 0, 1, 2)

            # Get metrics at display CI level
            pred = predictions_at_ci.get(sr, {})
            mean = pred.get('mean', 0)
            std = pred.get('std', 0)
            lower_ci = pred.get('lower_ci', 0)
            upper_ci = pred.get('upper_ci', 0)
            passes = pred.get('passes', False)
            percentage = pred.get('percentage', 0.0)

            self._add_result_row("  Predicted Mean", f"{mean:.2f} cP")
            self._add_result_row("  Standard Deviation", f"{std:.2f} cP")
            self._add_result_row(f"  {display_ci}% Confidence Interval",
                                 f"[{lower_ci:.2f}, {upper_ci:.2f}] cP")
            self._add_result_row("  CI Satisfaction", f"{percentage:.1f}%")
            self._add_result_row("  Result",
                                 "PASS" if passes else "FAIL")
        # Resize columns to content
        self.results_table.resizeColumnsToContents()

    def update_profile_plot(self) -> None:
        """Update the viscosity profile plot showing predictions across shear rates."""
        if not self.prediction_results or not self.hypothesis_result:
            return

        # Clear the figure
        self.profile_figure.clear()
        ax = self.profile_figure.add_subplot(111)

        # Get the confidence level from hypothesis result
        confidence_level = self.hypothesis_result['confidence_level']
        pred_data = self.prediction_results[confidence_level]

        # Get predictions at the selected confidence level
        pred_means = pred_data['means'].flatten()
        pred_stats = pred_data['stats']

        # Get shear rate values and corresponding predictions
        shear_rate_keys = list(self.SHEAR_RATES.keys())
        shear_rate_values = [self.SHEAR_RATES[key] for key in shear_rate_keys]

        # Extract mean and CI bounds
        means = pred_means
        lower_bounds = pred_stats[f'lower_{int(confidence_level)}'].flatten()
        upper_bounds = pred_stats[f'upper_{int(confidence_level)}'].flatten()

        # Plot the mean predictions
        ax.plot(shear_rate_values, means, 'o-', linewidth=2, markersize=8,
                label='Predicted Mean', color='#2E86AB', zorder=3)

        # Plot confidence intervals if checkbox is checked
        if self.show_confidence_check.isChecked():
            ax.fill_between(shear_rate_values, lower_bounds, upper_bounds,
                            alpha=0.3, color='#2E86AB',
                            label=f'{int(confidence_level)}% Confidence Interval')
            ax.plot(shear_rate_values, lower_bounds, '--', linewidth=1,
                    color='#2E86AB', alpha=0.6)
            ax.plot(shear_rate_values, upper_bounds, '--', linewidth=1,
                    color='#2E86AB', alpha=0.6)

        # Plot hypothesis threshold(s) if checkbox is checked
        if self.show_threshold_check.isChecked():
            hypothesis_type = self.hypothesis_result['hypothesis_type']
            details = self.hypothesis_result['details']

            if hypothesis_type == 'less_than':
                threshold = details['threshold']
                ax.axhline(y=threshold, color='#A23B72', linestyle='--',
                           linewidth=2, label=f'Threshold: {threshold:.2f} cP')
                ax.fill_between(shear_rate_values, 0, threshold,
                                alpha=0.1, color='#A23B72')

            elif hypothesis_type == 'greater_than':
                threshold = details['threshold']
                ax.axhline(y=threshold, color='#A23B72', linestyle='--',
                           linewidth=2, label=f'Threshold: {threshold:.2f} cP')
                y_max = ax.get_ylim()[1]
                ax.fill_between(shear_rate_values, threshold, y_max * 1.2,
                                alpha=0.1, color='#A23B72')

            elif hypothesis_type == 'between':
                min_val = details['min']
                max_val = details['max']
                ax.axhline(y=min_val, color='#A23B72', linestyle='--',
                           linewidth=2, alpha=0.7)
                ax.axhline(y=max_val, color='#A23B72', linestyle='--',
                           linewidth=2, alpha=0.7)
                ax.fill_between(shear_rate_values, min_val, max_val,
                                alpha=0.1, color='#A23B72',
                                label=f'Target Range: {min_val:.2f}-{max_val:.2f} cP')

            elif hypothesis_type == 'within_range':
                target = details['target']
                lower = details['lower']
                upper = details['upper']
                range_pct = details['range_pct']
                ax.axhline(y=target, color='#F18F01', linestyle='-',
                           linewidth=2, label=f'Target: {target:.2f} cP')
                ax.axhline(y=lower, color='#A23B72', linestyle='--',
                           linewidth=1.5, alpha=0.7)
                ax.axhline(y=upper, color='#A23B72', linestyle='--',
                           linewidth=1.5, alpha=0.7)
                ax.fill_between(shear_rate_values, lower, upper,
                                alpha=0.1, color='#A23B72',
                                label=f'±{range_pct:.1f}% Range')

        # Formatting
        ax.set_xlabel('Shear Rate (1/s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Viscosity (cP)', fontsize=12, fontweight='bold')
        ax.set_title('Viscosity Profile with Hypothesis Test',
                     fontsize=14, fontweight='bold', pad=15)

        # Set log scale if checkbox is checked
        if self.log_scale_check.isChecked():
            ax.set_xscale('log')
            ax.set_yscale('log')

        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        # Add legend
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=9)

        # Adjust layout to prevent label cutoff
        self.profile_figure.tight_layout()

        # Redraw the canvas
        self.profile_canvas.draw()

    def update_probability_plot(self) -> None:
        """Update the probability distribution plot for the selected shear rate."""
        if not self.prediction_results or not self.hypothesis_result:
            return

        # Clear the figure
        self.probability_figure.clear()
        ax = self.probability_figure.add_subplot(111)

        # Get selected shear rate from combo box
        selected_shear_key = self.prob_shear_combo.currentData()
        if not selected_shear_key:
            return

        shear_rate_value = self.SHEAR_RATES[selected_shear_key]
        shear_rate_keys = list(self.SHEAR_RATES.keys())
        shear_rate_index = shear_rate_keys.index(selected_shear_key)

        # Get the confidence level from hypothesis result
        confidence_level = self.hypothesis_result['confidence_level']
        pred_data = self.prediction_results[confidence_level]

        # Get mean and std for the selected shear rate
        mean = float(pred_data['means'].flatten()[shear_rate_index])
        std = float(pred_data['stats']['std'].flatten()[shear_rate_index])
        lower_ci = float(pred_data['stats'][f'lower_{int(confidence_level)}'].flatten()[
                         shear_rate_index])
        upper_ci = float(pred_data['stats'][f'upper_{int(confidence_level)}'].flatten()[
                         shear_rate_index])

        # Create x values for the normal distribution curve
        x_min = max(0, mean - 4 * std)  # Don't go below 0 for viscosity
        x_max = mean + 4 * std
        x = np.linspace(x_min, x_max, 500)

        # Calculate the normal distribution PDF
        pdf = stats.norm.pdf(x, mean, std)

        # Plot the probability distribution
        ax.plot(x, pdf, linewidth=2, color='#2E86AB',
                label='Probability Distribution')
        ax.fill_between(x, 0, pdf, alpha=0.2, color='#2E86AB')

        # Highlight the confidence interval
        ci_mask = (x >= lower_ci) & (x <= upper_ci)
        ax.fill_between(x[ci_mask], 0, pdf[ci_mask], alpha=0.4, color='#2E86AB',
                        label=f'{int(confidence_level)}% Confidence Interval')

        # Add vertical lines for mean and CI bounds
        ax.axvline(mean, color='#F18F01', linestyle='-', linewidth=2,
                   label=f'Mean: {mean:.2f} cP')
        ax.axvline(lower_ci, color='#2E86AB',
                   linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(upper_ci, color='#2E86AB',
                   linestyle='--', linewidth=1.5, alpha=0.7)

        # Highlight hypothesis regions
        hypothesis_type = self.hypothesis_result['hypothesis_type']
        details = self.hypothesis_result['details']

        if hypothesis_type == 'less_than':
            threshold = details['threshold']
            ax.axvline(threshold, color='#A23B72', linestyle='--', linewidth=2,
                       label=f'Threshold: {threshold:.2f} cP')

            # Shade region below threshold
            threshold_mask = x <= threshold
            ax.fill_between(x[threshold_mask], 0, pdf[threshold_mask],
                            alpha=0.15, color='#A23B72',
                            label='Hypothesis Region')

            # Calculate probability of being below threshold
            prob = stats.norm.cdf(threshold, mean, std)
            ax.text(0.05, 0.95, f'P(viscosity < {threshold:.2f}) = {prob:.3f}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        elif hypothesis_type == 'greater_than':
            threshold = details['threshold']
            ax.axvline(threshold, color='#A23B72', linestyle='--', linewidth=2,
                       label=f'Threshold: {threshold:.2f} cP')

            # Shade region above threshold
            threshold_mask = x >= threshold
            ax.fill_between(x[threshold_mask], 0, pdf[threshold_mask],
                            alpha=0.15, color='#A23B72',
                            label='Hypothesis Region')

            # Calculate probability of being above threshold
            prob = 1 - stats.norm.cdf(threshold, mean, std)
            ax.text(0.05, 0.95, f'P(viscosity > {threshold:.2f}) = {prob:.3f}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        elif hypothesis_type == 'between':
            min_val = details['min']
            max_val = details['max']
            ax.axvline(min_val, color='#A23B72',
                       linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axvline(max_val, color='#A23B72',
                       linestyle='--', linewidth=1.5, alpha=0.7)

            # Shade region between min and max
            between_mask = (x >= min_val) & (x <= max_val)
            ax.fill_between(x[between_mask], 0, pdf[between_mask],
                            alpha=0.15, color='#A23B72',
                            label=f'Target Range: {min_val:.2f}-{max_val:.2f} cP')

            # Calculate probability of being in range
            prob = stats.norm.cdf(max_val, mean, std) - \
                stats.norm.cdf(min_val, mean, std)
            ax.text(0.05, 0.95, f'P({min_val:.2f} < viscosity < {max_val:.2f}) = {prob:.3f}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        elif hypothesis_type == 'within_range':
            target = details['target']
            lower = details['lower']
            upper = details['upper']
            range_pct = details['range_pct']

            ax.axvline(target, color='#F18F01', linestyle='-', linewidth=2,
                       label=f'Target: {target:.2f} cP', alpha=0.7)
            ax.axvline(lower, color='#A23B72', linestyle='--',
                       linewidth=1.5, alpha=0.7)
            ax.axvline(upper, color='#A23B72', linestyle='--',
                       linewidth=1.5, alpha=0.7)

            # Shade region within range
            range_mask = (x >= lower) & (x <= upper)
            ax.fill_between(x[range_mask], 0, pdf[range_mask],
                            alpha=0.15, color='#A23B72',
                            label=f'±{range_pct:.1f}% Range')

            # Calculate probability of being in range
            prob = stats.norm.cdf(upper, mean, std) - \
                stats.norm.cdf(lower, mean, std)
            ax.text(0.05, 0.95, f'P(within ±{range_pct:.1f}% of {target:.2f}) = {prob:.3f}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Formatting
        ax.set_xlabel('Viscosity (cP)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        ax.set_title(f'Probability Distribution at Shear Rate {shear_rate_value:,} 1/s',
                     fontsize=14, fontweight='bold', pad=15)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        # Add legend
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=9)

        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)

        # Adjust layout to prevent label cutoff
        self.probability_figure.tight_layout()

        # Redraw the canvas
        self.probability_canvas.draw()

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
            self.temperature = None

            # Clear results
            self.prediction_results = None
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

            self.probability_figure.clear()
            self.probability_canvas.draw()

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
                # Build hypothesis statement
                hypothesis_type = self.hypothesis_result['hypothesis_type']
                shear_rate_data = self.hypothesis_result['shear_rate']

                if shear_rate_data == "all":
                    shear_text = "all shear rates"
                else:
                    shear_rate_value = self.SHEAR_RATES[shear_rate_data]
                    shear_text = f"shear rate of {shear_rate_value:,} 1/s"

                if hypothesis_type == 'less_than':
                    threshold = self.hypothesis_result['details']['threshold']
                    statement = f"Viscosity at {shear_text} < {threshold:.2f} cP"
                elif hypothesis_type == 'greater_than':
                    threshold = self.hypothesis_result['details']['threshold']
                    statement = f"Viscosity at {shear_text} > {threshold:.2f} cP"
                elif hypothesis_type == 'between':
                    min_val = self.hypothesis_result['details']['min']
                    max_val = self.hypothesis_result['details']['max']
                    statement = f"Viscosity at {shear_text} between {min_val:.2f} and {max_val:.2f} cP"
                elif hypothesis_type == 'within_range':
                    target = self.hypothesis_result['details']['target']
                    range_pct = self.hypothesis_result['details']['range_pct']
                    statement = f"Viscosity at {shear_text} within ±{range_pct:.1f}% of {target:.2f} cP"
                else:
                    statement = "Unknown hypothesis type"

                # Prepare data
                data = {
                    'formulation': self.current_formulation,
                    'temperature': self.temperature,
                    'hypothesis': {
                        'type': self.hypothesis_result['hypothesis_type'],
                        'shear_rate': self.hypothesis_result['shear_rate'],
                        'confidence_level': self.hypothesis_result['confidence_level'],
                        'statement': statement
                    },
                    'results': {
                        'passed': self.hypothesis_result['passed'],
                        'probability': self.hypothesis_result['probability'],
                        'details': self.hypothesis_result['details']
                    },
                    'predictions': self.prediction_results
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
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save plot:\n{str(e)}"
                )

    def save_probability_plot(self) -> None:
        """Save the probability distribution plot."""
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Probability Plot",
            "probability_distribution.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
        )

        if file_path:
            try:
                self.probability_figure.savefig(
                    file_path, dpi=300, bbox_inches='tight')
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
