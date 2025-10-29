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

            # Run prediction
            self.prediction_results = self.predictor.predict_uncertainty(
                formulation.to_dataframe(encoded=False, training=False))

            # Evaluate hypothesis
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
        """Evaluate the hypothesis based on predictions."""
        hypothesis_type = self.hypothesis_type_combo.currentData()
        shear_rate_data = self.shear_rate_combo.currentData()
        confidence_level = self.confidence_spin.value()

        result = {
            'hypothesis_type': hypothesis_type,
            'shear_rate': shear_rate_data,
            'confidence_level': confidence_level,
            'passed': False,
            'probability': 0.0,
            'details': {}
        }

        # Get predictions for relevant shear rate(s)
        if shear_rate_data == "all":
            shear_rates = list(self.SHEAR_RATES.keys())
        else:
            shear_rates = [shear_rate_data]

        # Extract predictions and uncertainties
        predictions = {}
        pred_means, pred_stats = self.prediction_results
        mean_values = pred_means.flatten()
        std_values = pred_stats["std"].flatten()
        lower_values = pred_stats["lower_95"].flatten()
        upper_values = pred_stats["upper_95"].flatten()
        cv_values = pred_stats["coefficient_of_variation"].flatten()

        for i, sr in enumerate(shear_rates):
            predictions[sr] = {
                "mean": float(mean_values[i]),
                "std": float(std_values[i]),
                "lower_95": float(lower_values[i]),
                "upper_95": float(upper_values[i]),
                "coefficient_of_variation": float(cv_values[i]),
            }

        # Evaluate based on hypothesis type
        if hypothesis_type == 'less_than':
            threshold = self.threshold_spin.value()
            result['details']['threshold'] = threshold
            result = self._evaluate_less_than(
                predictions, threshold, confidence_level, result)

        elif hypothesis_type == 'greater_than':
            threshold = self.threshold_spin.value()
            result['details']['threshold'] = threshold
            result = self._evaluate_greater_than(
                predictions, threshold, confidence_level, result)

        elif hypothesis_type == 'between':
            min_val = self.min_spin.value()
            max_val = self.max_spin.value()
            result['details']['min'] = min_val
            result['details']['max'] = max_val
            result = self._evaluate_between(
                predictions, min_val, max_val, confidence_level, result)

        elif hypothesis_type == 'within_range':
            target = self.target_spin.value()
            range_pct = self.range_spin.value()
            lower = target * (1 - range_pct / 100)
            upper = target * (1 + range_pct / 100)
            result['details']['target'] = target
            result['details']['range_pct'] = range_pct
            result['details']['lower'] = lower
            result['details']['upper'] = upper
            result = self._evaluate_between(
                predictions, lower, upper, confidence_level, result)

        return result

    def _evaluate_less_than(self, predictions: Dict, threshold: float,
                            confidence: float, result: Dict) -> Dict:
        """Evaluate 'less than' hypothesis."""
        probabilities = []

        for sr, pred in predictions.items():
            mean = pred['mean']
            std = pred['std']

            # Calculate probability using normal CDF
            if std > 0:
                from scipy import stats
                z_score = (threshold - mean) / std
                prob = stats.norm.cdf(z_score)
            else:
                prob = 1.0 if mean < threshold else 0.0

            probabilities.append(prob)
            result['details'][f'{sr}_probability'] = prob
            result['details'][f'{sr}_mean'] = mean
            result['details'][f'{sr}_std'] = std

        # Overall probability (all must pass)
        avg_prob = np.mean(probabilities) * 100
        result['probability'] = avg_prob
        result['passed'] = avg_prob >= confidence

        return result

    def _evaluate_greater_than(self, predictions: Dict, threshold: float,
                               confidence: float, result: Dict) -> Dict:
        """Evaluate 'greater than' hypothesis."""
        probabilities = []

        for sr, pred in predictions.items():
            mean = pred['mean']
            std = pred['std']

            # Calculate probability using normal CDF
            if std > 0:
                from scipy import stats
                z_score = (threshold - mean) / std
                prob = 1 - stats.norm.cdf(z_score)
            else:
                prob = 1.0 if mean > threshold else 0.0

            probabilities.append(prob)
            result['details'][f'{sr}_probability'] = prob
            result['details'][f'{sr}_mean'] = mean
            result['details'][f'{sr}_std'] = std

        # Overall probability
        avg_prob = np.mean(probabilities) * 100
        result['probability'] = avg_prob
        result['passed'] = avg_prob >= confidence

        return result

    def _evaluate_between(self, predictions: Dict, min_val: float, max_val: float,
                          confidence: float, result: Dict) -> Dict:
        """Evaluate 'between' or 'within range' hypothesis."""
        probabilities = []

        for sr, pred in predictions.items():
            mean = pred['mean']
            std = pred['std']

            # Calculate probability using normal CDF
            if std > 0:
                from scipy import stats
                z_lower = (min_val - mean) / std
                z_upper = (max_val - mean) / std
                prob = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
            else:
                prob = 1.0 if min_val <= mean <= max_val else 0.0

            probabilities.append(prob)
            result['details'][f'{sr}_probability'] = prob
            result['details'][f'{sr}_mean'] = mean
            result['details'][f'{sr}_std'] = std

        # Overall probability
        avg_prob = np.mean(probabilities) * 100
        result['probability'] = avg_prob
        result['passed'] = avg_prob >= confidence

        return result

    def update_results_summary(self) -> None:
        """Update the results summary display."""
        if not self.hypothesis_result:
            return

        # Update outcome label
        passed = self.hypothesis_result['passed']
        probability = self.hypothesis_result['probability']
        confidence = self.hypothesis_result['confidence_level']

        if passed:
            self.outcome_frame.setStyleSheet(
                "QFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #E8F8F7, stop:1 #D4F1EF); border: 2px solid #69EAC5; border-radius: 8px; }"
            )
            self.outcome_label.setText("Hypothesis Supported")
            self.outcome_label.setStyleSheet(
                "font-size: 16pt; font-weight: 500; color: #00695C; padding: 20px; background: transparent;"
            )
        else:
            self.outcome_frame.setStyleSheet(
                "QFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FFF3E0, stop:1 #FFE0B2); border: 2px solid #FF9800; border-radius: 8px; }"
            )
            self.outcome_label.setText("Hypothesis Not Supported")
            self.outcome_label.setStyleSheet(
                "font-size: 16pt; font-weight: 500; color: #E65100; padding: 20px; background: transparent;"
            )

        self.probability_label.setText(
            f"Probability: {probability:.1f}% (Required: {confidence:.0f}%)"
        )

        # Update results table
        self.results_table.setRowCount(0)

        # Add hypothesis details
        hypothesis_type = self.HYPOTHESIS_TYPES[self.hypothesis_result['hypothesis_type']]
        self._add_result_row("Hypothesis Type", hypothesis_type)

        shear_data = self.hypothesis_result['shear_rate']
        if shear_data == "all":
            shear_text = "All Shear Rates"
        else:
            shear_text = f"{self.SHEAR_RATES[shear_data]:,} 1/s"
        self._add_result_row("Shear Rate", shear_text)

        self._add_result_row("Confidence Level", f"{confidence:.0f}%")
        self._add_result_row("Calculated Probability", f"{probability:.1f}%")
        self._add_result_row("Result", "PASS" if passed else "FAIL")

        # Add prediction details
        details = self.hypothesis_result['details']
        for key, value in details.items():
            if '_mean' in key:
                sr = key.replace('_mean', '')
                self._add_result_row(f"Predicted ({sr})", f"{value:.2f} cP")
            elif '_std' in key:
                sr = key.replace('_std', '')
                self._add_result_row(f"Std Dev ({sr})", f"{value:.2f} cP")

        # Update details text
        details_html = "<b>Formulation:</b><br>"
        for ingredient, data in self.current_formulation.items():
            value = data['value']
            unit = data['unit']
            details_html += f"• {ingredient}: {value:.2f} {unit}<br>"

        details_html += f"• Temperature: {self.temperature:.1f} °C<br>"

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
            hypothesis_text = f"Viscosity at {shear_text} &lt; {threshold:.2f} cP"
        elif hypothesis_type == 'greater_than':
            threshold = self.hypothesis_result['details']['threshold']
            hypothesis_text = f"Viscosity at {shear_text} &gt; {threshold:.2f} cP"
        elif hypothesis_type == 'between':
            min_val = self.hypothesis_result['details']['min']
            max_val = self.hypothesis_result['details']['max']
            hypothesis_text = f"Viscosity at {shear_text} between {min_val:.2f} and {max_val:.2f} cP"
        elif hypothesis_type == 'within_range':
            target = self.hypothesis_result['details']['target']
            range_pct = self.hypothesis_result['details']['range_pct']
            hypothesis_text = f"Viscosity at {shear_text} within ±{range_pct:.1f}% of {target:.2f} cP"
        else:
            hypothesis_text = "Unknown hypothesis type"

        details_html += f"<br><b>Hypothesis:</b><br>{hypothesis_text}"

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

    def update_profile_plot(self) -> None:
        """Update the viscosity profile plot."""
        if not self.prediction_results:
            return

        self.profile_figure.clear()
        ax = self.profile_figure.add_subplot(111)

        # Extract data
        shear_rates = [100, 1000, 10000, 100000, 15000000]
        predictions = []
        uncertainties = []
        prediction_dict = {}

        pred_means, pred_stats = self.prediction_results
        mean_values = pred_means.flatten()
        std_values = pred_stats.get("std", [0] * len(mean_values)).flatten()

        # Optional keys
        lower_values = pred_stats.get("lower_95", [None] * len(mean_values))
        upper_values = pred_stats.get("upper_95", [None] * len(mean_values))
        cv_values = pred_stats.get("coefficient_of_variation", [
                                   None] * len(mean_values))

        # Build prediction lists and dict
        for i, sr in enumerate(shear_rates):
            mean = float(mean_values[i])
            std = float(std_values[i])
            predictions.append(mean)
            uncertainties.append(std)

            prediction_dict[sr] = {
                "mean": mean,
                "std": std,
                # "lower_95": float(lower_values[i]) if lower_values is not None else None,
                # "upper_95": float(upper_values[i]) if upper_values is not None else None,
                # "coefficient_of_variation": float(cv_values[i]) if cv_values is not None else None,
            }

        # Plot prediction line
        ax.plot(shear_rates, predictions, 'o-', linewidth=2, markersize=8,
                label='Predicted Viscosity', color='#00A3DA')

        # Show confidence intervals if enabled
        if self.show_confidence_check.isChecked():
            confidence = self.confidence_spin.value()
            z_score = 1.96 if confidence >= 95 else 1.645  # 95% or 90%

            lower = [p - z_score * u for p,
                     u in zip(predictions, uncertainties)]
            upper = [p + z_score * u for p,
                     u in zip(predictions, uncertainties)]

            ax.fill_between(shear_rates, lower, upper, alpha=0.3, color='#00A3DA',
                            label=f'{confidence:.0f}% Confidence Interval')

        # Show hypothesis threshold if enabled
        if self.show_threshold_check.isChecked() and self.hypothesis_result:
            hypothesis_type = self.hypothesis_result['hypothesis_type']
            details = self.hypothesis_result['details']

            if hypothesis_type in ['less_than', 'greater_than']:
                threshold = details.get('threshold')
                if threshold:
                    ax.axhline(y=threshold, color='red', linestyle='--',
                               linewidth=2, label=f'Threshold ({threshold:.0f} cP)')

            elif hypothesis_type in ['between', 'within_range']:
                if 'min' in details:
                    min_val = details['min']
                    max_val = details['max']
                else:
                    min_val = details['lower']
                    max_val = details['upper']

                ax.axhline(y=min_val, color='red', linestyle='--', linewidth=2)
                ax.axhline(y=max_val, color='red', linestyle='--', linewidth=2)
                ax.axhspan(min_val, max_val, alpha=0.2, color='red',
                           label='Acceptable Range')

        # Formatting
        ax.set_xlabel('Shear Rate (1/s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Viscosity (cP)', fontsize=12, fontweight='bold')
        ax.set_title('Predicted Viscosity Profile',
                     fontsize=14, fontweight='bold')

        if self.log_scale_check.isChecked():
            ax.set_xscale('log')
            ax.set_yscale('log')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        self.profile_figure.tight_layout()
        self.profile_canvas.draw()

    def update_probability_plot(self) -> None:
        """Update the probability distribution plot."""
        if not self.prediction_results or not self.hypothesis_result:
            return

        self.probability_figure.clear()
        ax = self.probability_figure.add_subplot(111)

        # Get selected shear rate
        visc_label = self.prob_shear_combo.currentData()
        shear_rate = self.SHEAR_RATES[visc_label]

        # --- Extract prediction data from new tuple format ---
        pred_means, pred_stats = self.prediction_results
        mean_values = pred_means.flatten()
        std_values = pred_stats["std"].flatten()

        # Match shear rate index to label position
        shear_labels = list(self.SHEAR_RATES.keys())
        if visc_label not in shear_labels:
            return
        i = shear_labels.index(visc_label)

        mean = float(mean_values[i])
        std = float(std_values[i])

        # --- Generate normal distribution ---
        from scipy import stats
        x = np.linspace(max(0, mean - 4 * std), mean + 4 * std, 1000)
        y = stats.norm.pdf(x, mean, std)

        # --- Plot distribution ---
        ax.plot(x, y, linewidth=2, color='#00A3DA',
                label='Probability Distribution')
        ax.fill_between(x, y, alpha=0.3, color='#00A3DA')

        # --- Mark mean ---
        ax.axvline(
            x=mean,
            color='blue',
            linestyle='--',
            linewidth=2,
            label=f'Predicted Mean ({mean:.1f} cP)'
        )

        # --- Show hypothesis region ---
        hypothesis_type = self.hypothesis_result['hypothesis_type']
        details = self.hypothesis_result['details']

        if hypothesis_type == 'less_than':
            threshold = details['threshold']
            ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold ({threshold:.1f} cP)')

            # Shade region
            mask = x <= threshold
            ax.fill_between(x[mask], y[mask], alpha=0.5, color='#69EAC5',
                            label='Hypothesis Region')

        elif hypothesis_type == 'greater_than':
            threshold = details['threshold']
            ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold ({threshold:.1f} cP)')

            mask = x >= threshold
            ax.fill_between(x[mask], y[mask], alpha=0.5, color='#69EAC5',
                            label='Hypothesis Region')

        elif hypothesis_type in ['between', 'within_range']:
            if 'min' in details:
                min_val = details['min']
                max_val = details['max']
            else:
                min_val = details['lower']
                max_val = details['upper']

            ax.axvline(x=min_val, color='red', linestyle='--', linewidth=2)
            ax.axvline(x=max_val, color='red', linestyle='--', linewidth=2)

            mask = (x >= min_val) & (x <= max_val)
            ax.fill_between(x[mask], y[mask], alpha=0.5, color='#69EAC5',
                            label='Hypothesis Region')

        # Get probability for this shear rate
        prob_key = f'{visc_label}_probability'
        prob = details.get(prob_key, 0) * 100

        # Formatting
        ax.set_xlabel('Viscosity (cP)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Probability Distribution at {shear_rate:,} 1/s\n'
            f'P(Hypothesis) = {prob:.1f}%',
            fontsize=14, fontweight='bold'
        )
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        self.probability_figure.tight_layout()
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
