"""
optimization_ui.py

This module provides the OptimizationUI class for the Optimization tab in VisQAI.
It supports defining target viscosity profiles, setting formulation constraints,
and finding optimal formulation parameters using differential evolution.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-11-04

Version:
    1.0 - Initial implementation
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
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from src.models.formulation import Formulation, ViscosityProfile
    from src.models.predictor import Predictor
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.models.ingredient import Protein, Buffer, Salt, Stabilizer, Surfactant, Excipient
    from src.utils.constraints import Constraints
    from src.processors.optimizer import Optimizer
    from src.view.constraints_ui import ConstraintsUI
    if TYPE_CHECKING:
        from src.view.main_window import VisQAIWindow
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.formulation import Formulation, ViscosityProfile
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.models.ingredient import Protein, Buffer, Salt, Stabilizer, Surfactant, Excipient
    from QATCH.VisQAI.src.utils.constraints import Constraints
    from QATCH.VisQAI.src.processors.optimizer import Optimizer
    from QATCH.VisQAI.src.view.constraints_ui import ConstraintsUI
    if TYPE_CHECKING:
        from QATCH.VisQAI.src.view.main_window import VisQAIWindow

TAG = "[OptimizationUI]"


class OptimizationWorker(QThread):
    """Worker thread for running optimization to avoid UI blocking."""

    finished = pyqtSignal(object)  # Emits the optimized formulation
    error = pyqtSignal(str)  # Emits error message
    progress = pyqtSignal(str)  # Emits progress updates

    def __init__(self, optimizer: Optimizer):
        super().__init__()
        self.optimizer = optimizer

    def run(self):
        """Run the optimization in a separate thread."""
        try:
            self.progress.emit("Starting optimization...")
            result = self.optimizer.optimize()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class OptimizationUI(QtWidgets.QDialog):
    """
    A UI for optimization of formulation parameters to match target viscosity profiles.

    This interface allows users to:
    - Define target viscosity profiles across different shear rates
    - Set constraints for formulation parameters
    - Configure optimization settings
    - Run differential evolution optimization
    - Visualize optimal formulation and predicted profile
    """

    SHEAR_RATES = [100, 1000, 10000, 100000, 15000000]

    SHEAR_RATE_LABELS = {
        100: 'Viscosity @ 100 s⁻¹',
        1000: 'Viscosity @ 1000 s⁻¹',
        10000: 'Viscosity @ 10000 s⁻¹',
        100000: 'Viscosity @ 100000 s⁻¹',
        15000000: 'Viscosity @ 15M s⁻¹'
    }

    INGREDIENT_UNITS = {
        'Protein': 'mg/mL',
        'Buffer': 'mM',
        'Surfactant': '%w',
        'Stabilizer': 'M',
        'Excipient': 'mM',
        'Salt': 'mM'
    }

    INGREDIENT_TYPES = ['Protein', 'Buffer',
                        'Surfactant', 'Stabilizer', 'Excipient', 'Salt']

    def __init__(self, parent=None):
        """Initialize the OptimizationUI window.

        Args:
            parent (VisQAIWindow, optional): The parent window instance.
        """
        super().__init__(parent)
        self.parent: 'VisQAIWindow' = parent
        self.setWindowTitle("Formulation Optimization")

        # Database and controllers
        self.db: Optional[Database] = None
        self.ing_ctrl: Optional[IngredientController] = None

        # Core components
        self.predictor: Optional[Predictor] = None
        self.optimizer: Optional[Optimizer] = None
        self.constraints: Optional[Constraints] = None
        self.model_path: Optional[str] = None

        # Optimization results
        self.optimal_formulation: Optional[Formulation] = None
        self.target_profile: Optional[ViscosityProfile] = None

        # Worker thread
        self.worker: Optional[OptimizationWorker] = None

        # Available ingredients by type
        self.ingredients_by_type: Dict[str, List] = {
            'Protein': [],
            'Buffer': [],
            'Surfactant': [],
            'Stabilizer': [],
            'Excipient': [],
            'Salt': []
        }

        # Properties needed for constraints UI integration
        self.proteins = []
        self.buffers = []
        self.surfactants = []
        self.stabilizers = []
        self.salts = []
        self.excipients = []
        self.class_types = []
        self.proteins_by_class = {}
        self.database = None

        # Constraints UI
        self.constraints_ui: Optional[ConstraintsUI] = None

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
            self.database = self.db  # Store reference for constraints UI
            self.ing_ctrl = IngredientController(self.db)
            Log.i(TAG, "Database initialized successfully")
        except Exception as e:
            Log.e(TAG, f"Failed to initialize database: {e}")
            self.db = None
            self.database = None
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

                self.ingredients_by_type[ing_type] = sorted(
                    unique_ingredients.values(),
                    key=lambda x: x.name
                )

            # Populate lists needed by constraints UI
            self.proteins = [
                ing.name for ing in self.ingredients_by_type['Protein']]
            self.buffers = [
                ing.name for ing in self.ingredients_by_type['Buffer']]
            self.surfactants = [
                ing.name for ing in self.ingredients_by_type['Surfactant']]
            self.stabilizers = [
                ing.name for ing in self.ingredients_by_type['Stabilizer']]
            self.excipients = [
                ing.name for ing in self.ingredients_by_type['Excipient']]
            self.salts = [ing.name for ing in self.ingredients_by_type['Salt']]

            # Get class types for proteins (if available)
            self.class_types = []
            self.proteins_by_class = {}
            for protein in self.ingredients_by_type['Protein']:
                if hasattr(protein, 'protein_class') and protein.protein_class:
                    if protein.protein_class not in self.class_types:
                        self.class_types.append(protein.protein_class)
                    if protein.protein_class not in self.proteins_by_class:
                        self.proteins_by_class[protein.protein_class] = []
                    self.proteins_by_class[protein.protein_class].append(
                        protein.name)

            Log.i(
                TAG, f"Loaded ingredients: {sum(len(ings) for ings in self.ingredients_by_type.values())}")

        except Exception as e:
            Log.e(TAG, f"Failed to load ingredients: {e}")

    def init_ui(self) -> None:
        """Initialize the user interface."""
        self.setFixedSize(1200, 900)
        self.setModal(True)

        # Create main layout
        main_layout = QtWidgets.QHBoxLayout(self)

        # Left panel for inputs
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 1)

        # Right panel for results
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 2)

    def _create_left_panel(self) -> QtWidgets.QWidget:
        """Create the left panel with input controls."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # Model selection
        model_group = self._create_model_selection_group()
        layout.addWidget(model_group)

        # Target profile definition
        target_group = self._create_target_profile_group()
        layout.addWidget(target_group)

        # Constraints definition
        constraints_group = self._create_constraints_group()
        layout.addWidget(constraints_group)

        # Optimization settings
        settings_group = self._create_optimization_settings_group()
        layout.addWidget(settings_group)

        # Control buttons
        buttons_layout = self._create_control_buttons()
        layout.addLayout(buttons_layout)

        layout.addStretch()
        return panel

    def _create_model_selection_group(self) -> QtWidgets.QGroupBox:
        """Create model selection group."""
        group = QtWidgets.QGroupBox("Model Selection")

        layout = QtWidgets.QVBoxLayout(group)

        # Model path display and browse button
        model_layout = QtWidgets.QHBoxLayout()

        self.model_path_label = QtWidgets.QLabel("No model selected")
        model_layout.addWidget(self.model_path_label, 1)

        self.browse_model_btn = QtWidgets.QPushButton("Browse...")
        self.browse_model_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.browse_model_btn)

        layout.addLayout(model_layout)

        return group

    def _create_target_profile_group(self) -> QtWidgets.QGroupBox:
        """Create target viscosity profile definition group."""
        group = QtWidgets.QGroupBox("Target Viscosity Profile")

        layout = QtWidgets.QVBoxLayout(group)

        # Create spinboxes for each shear rate
        self.target_spins = {}

        for shear_rate in self.SHEAR_RATES:
            row_layout = QtWidgets.QHBoxLayout()

            label = QtWidgets.QLabel(self.SHEAR_RATE_LABELS[shear_rate])
            label.setMinimumWidth(150)
            row_layout.addWidget(label)

            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.1, 10000.0)
            spin.setValue(10.0)
            spin.setDecimals(2)
            spin.setSuffix(" cP")
            spin.setMinimumWidth(100)
            self.target_spins[shear_rate] = spin
            row_layout.addWidget(spin)

            layout.addLayout(row_layout)
        return group

    def _create_constraints_group(self) -> QtWidgets.QGroupBox:
        """Create formulation constraints group."""
        group = QtWidgets.QGroupBox("Formulation Constraints")

        layout = QtWidgets.QVBoxLayout(group)

        # Status label
        self.constraints_status_label = QtWidgets.QLabel(
            "No constraints defined")
        layout.addWidget(self.constraints_status_label)

        # Configure constraints button
        self.configure_constraints_btn = QtWidgets.QPushButton(
            "Configure Constraints...")
        self.configure_constraints_btn.clicked.connect(
            self.open_constraints_dialog)
        layout.addWidget(self.configure_constraints_btn)

        return group

    def _create_optimization_settings_group(self) -> QtWidgets.QGroupBox:
        """Create optimization settings group."""
        group = QtWidgets.QGroupBox("Optimization Settings")
        layout = QtWidgets.QVBoxLayout(group)

        # Max iterations
        iter_layout = QtWidgets.QHBoxLayout()
        iter_layout.addWidget(QtWidgets.QLabel("Max Iterations:"))
        self.maxiter_spin = QtWidgets.QSpinBox()
        self.maxiter_spin.setRange(10, 1000)
        self.maxiter_spin.setValue(100)
        iter_layout.addWidget(self.maxiter_spin)
        layout.addLayout(iter_layout)

        # Population size
        pop_layout = QtWidgets.QHBoxLayout()
        pop_layout.addWidget(QtWidgets.QLabel("Population Size:"))
        self.popsize_spin = QtWidgets.QSpinBox()
        self.popsize_spin.setRange(5, 50)
        self.popsize_spin.setValue(15)
        pop_layout.addWidget(self.popsize_spin)
        layout.addLayout(pop_layout)

        # Tolerance
        tol_layout = QtWidgets.QHBoxLayout()
        tol_layout.addWidget(QtWidgets.QLabel("Tolerance:"))
        self.tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.tolerance_spin.setRange(1e-9, 1e-3)
        self.tolerance_spin.setValue(1e-6)
        self.tolerance_spin.setDecimals(9)
        self.tolerance_spin.setStepType(
            QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        tol_layout.addWidget(self.tolerance_spin)
        layout.addLayout(tol_layout)

        # Absolute tolerance
        atol_layout = QtWidgets.QHBoxLayout()
        atol_layout.addWidget(QtWidgets.QLabel("Absolute Tolerance:"))
        self.atol_spin = QtWidgets.QSpinBox()
        self.atol_spin.setRange(0, 10)
        self.atol_spin.setValue(0)
        atol_layout.addWidget(self.atol_spin)
        layout.addLayout(atol_layout)

        # Random seed
        seed_layout = QtWidgets.QHBoxLayout()
        self.use_seed_check = QtWidgets.QCheckBox("Use Random Seed:")
        seed_layout.addWidget(self.use_seed_check)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        self.seed_spin.setEnabled(False)
        self.use_seed_check.toggled.connect(self.seed_spin.setEnabled)
        seed_layout.addWidget(self.seed_spin)
        layout.addLayout(seed_layout)

        # Strategy
        strategy_layout = QtWidgets.QHBoxLayout()
        strategy_layout.addWidget(QtWidgets.QLabel("Strategy:"))
        self.strategy_combo = QtWidgets.QComboBox()
        self.strategy_combo.addItems([
            "best1bin", "best1exp", "best2bin", "best2exp",
            "rand1bin", "rand1exp", "rand2bin", "rand2exp",
            "randtobest1bin", "randtobest1exp"
        ])
        self.strategy_combo.setCurrentText("best1bin")
        strategy_layout.addWidget(self.strategy_combo)
        layout.addLayout(strategy_layout)

        # Mutation
        mutation_layout = QtWidgets.QHBoxLayout()
        mutation_layout.addWidget(QtWidgets.QLabel("Mutation:"))
        self.mutation_min_spin = QtWidgets.QDoubleSpinBox()
        self.mutation_min_spin.setRange(0.0, 2.0)
        self.mutation_min_spin.setValue(0.5)
        self.mutation_min_spin.setSingleStep(0.1)
        mutation_layout.addWidget(QtWidgets.QLabel("Min:"))
        mutation_layout.addWidget(self.mutation_min_spin)
        self.mutation_max_spin = QtWidgets.QDoubleSpinBox()
        self.mutation_max_spin.setRange(0.0, 2.0)
        self.mutation_max_spin.setValue(1.0)
        self.mutation_max_spin.setSingleStep(0.1)
        mutation_layout.addWidget(QtWidgets.QLabel("Max:"))
        mutation_layout.addWidget(self.mutation_max_spin)
        layout.addLayout(mutation_layout)

        # Recombination
        recomb_layout = QtWidgets.QHBoxLayout()
        recomb_layout.addWidget(QtWidgets.QLabel("Recombination:"))
        self.recombination_spin = QtWidgets.QDoubleSpinBox()
        self.recombination_spin.setRange(0.0, 1.0)
        self.recombination_spin.setValue(0.7)
        self.recombination_spin.setSingleStep(0.1)
        recomb_layout.addWidget(self.recombination_spin)
        layout.addLayout(recomb_layout)

        # Initialization
        init_layout = QtWidgets.QHBoxLayout()
        init_layout.addWidget(QtWidgets.QLabel("Initialization:"))
        self.init_combo = QtWidgets.QComboBox()
        self.init_combo.addItems(
            ['latinhypercube', 'sobol', 'halton', 'random'])
        self.init_combo.setCurrentText("latinhypercube")
        init_layout.addWidget(self.init_combo)
        layout.addLayout(init_layout)
        return group

    def _create_control_buttons(self) -> QtWidgets.QHBoxLayout:
        """Create control buttons layout."""
        layout = QtWidgets.QHBoxLayout()

        self.optimize_btn = QtWidgets.QPushButton("Start Optimization")
        self.optimize_btn.clicked.connect(self.start_optimization)
        self.optimize_btn.setEnabled(False)
        layout.addWidget(self.optimize_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_optimization)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)

        return layout

    def _create_right_panel(self) -> QtWidgets.QWidget:
        """Create the right panel with results and visualization."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # Progress and status
        self.progress_label = QtWidgets.QLabel("Ready to optimize")
        layout.addWidget(self.progress_label)

        # Results display
        results_group = self._create_results_group()
        layout.addWidget(results_group)

        # Visualization
        viz_group = self._create_visualization_group()
        layout.addWidget(viz_group)

        # Action buttons
        action_buttons = self._create_action_buttons()
        layout.addLayout(action_buttons)

        return panel

    def _create_results_group(self) -> QtWidgets.QGroupBox:
        """Create optimal formulation results group."""
        group = QtWidgets.QGroupBox("Optimal Formulation")
        layout = QtWidgets.QVBoxLayout(group)

        # Create table for displaying optimal formulation
        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(
            ["Component", "Value", "Units"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setMaximumHeight(200)
        layout.addWidget(self.results_table)

        return group

    def _create_visualization_group(self) -> QtWidgets.QGroupBox:
        """Create visualization group."""
        group = QtWidgets.QGroupBox("Viscosity Profile Comparison")

        layout = QtWidgets.QVBoxLayout(group)

        # Create matplotlib figure
        self.profile_figure = Figure(figsize=(8, 5))
        self.profile_canvas = FigureCanvas(self.profile_figure)
        layout.addWidget(self.profile_canvas)

        # Plot options
        options_layout = QtWidgets.QHBoxLayout()

        self.log_scale_check = QtWidgets.QCheckBox("Log Scale X-axis")
        self.log_scale_check.setChecked(True)
        self.log_scale_check.toggled.connect(self.update_plot)
        options_layout.addWidget(self.log_scale_check)

        options_layout.addStretch()
        layout.addLayout(options_layout)

        return group

    def _create_action_buttons(self) -> QtWidgets.QHBoxLayout:
        """Create action buttons layout."""
        layout = QtWidgets.QHBoxLayout()

        self.clear_btn = QtWidgets.QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all)
        layout.addWidget(self.clear_btn)

        self.save_results_btn = QtWidgets.QPushButton("Save Results")
        self.save_results_btn.clicked.connect(self.save_results)
        self.save_results_btn.setEnabled(False)
        layout.addWidget(self.save_results_btn)

        self.save_plot_btn = QtWidgets.QPushButton("Save Plot")
        self.save_plot_btn.clicked.connect(self.save_profile_plot)
        self.save_plot_btn.setEnabled(False)
        layout.addWidget(self.save_plot_btn)

        return layout

    def browse_model(self) -> None:
        """Browse for model file."""
        if self.model_dialog.exec_() == QtWidgets.QDialog.Accepted:
            files = self.model_dialog.selectedFiles()
            if files:
                self.model_path = files[0]
                self.model_path_label.setText(
                    os.path.basename(self.model_path))
                self._load_model()

    def _load_model(self) -> None:
        """Load the selected model."""
        if not self.model_path:
            return

        try:
            self.predictor = Predictor(self.model_path)
            Log.i(TAG, f"Model loaded successfully: {self.model_path}")
            self._check_ready_state()
        except Exception as e:
            Log.e(TAG, f"Failed to load model: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Model Error",
                f"Failed to load model:\n{str(e)}"
            )
            self.predictor = None

    def _check_ready_state(self) -> None:
        """Check if all requirements are met for optimization."""
        ready = (
            self.predictor is not None and
            self.ing_ctrl is not None and
            self.database is not None
        )
        self.optimize_btn.setEnabled(ready)

        if ready:
            self.progress_label.setText("Ready to optimize")
        else:
            missing = []
            if self.predictor is None:
                missing.append("model")
            if self.ing_ctrl is None or self.database is None:
                missing.append("database")

            self.progress_label.setText(
                f"Load {', '.join(missing)} to continue")

    def save_target_profile(self) -> None:
        """Save target profile to file."""
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Target Profile",
            "target_profile.json",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            try:
                profile_data = {}
                for shear_rate, spin in self.target_spins.items():
                    profile_data[str(shear_rate)] = spin.value()

                data = {
                    'target_profile': profile_data,
                    'shear_rates': self.SHEAR_RATES
                }

                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)

                QtWidgets.QMessageBox.information(
                    self,
                    "Success",
                    f"Target profile saved to:\n{file_path}"
                )

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Save Error",
                    f"Failed to save target profile:\n{str(e)}"
                )

    def open_constraints_dialog(self) -> None:
        """Open the constraints configuration dialog."""
        if not self.database or not self.ing_ctrl:
            QtWidgets.QMessageBox.warning(
                self,
                "Database Error",
                "Database not initialized. Cannot configure constraints."
            )
            return

        # Create constraints UI (step 6 for optimization)
        self.constraints_ui = ConstraintsUI(self, step=6)
        self.constraints_ui.add_suggestion_dialog()

    def set_constraints(self, constraints: Constraints) -> None:
        """Set the constraints object from the constraints dialog."""
        self.constraints = constraints

        # Update status label
        if constraints:
            # Count the number of constraints
            bounds, encoding = constraints.build()
            num_constraints = len(
                [enc for enc in encoding if enc.get('choices') or enc.get('range')])

            if num_constraints > 0:
                self.constraints_status_label.setText(
                    f"{num_constraints} constraint(s) defined")
            else:
                self.constraints_status_label.setText(
                    "No specific constraints (using defaults)")
        else:
            self.constraints_status_label.setText("No constraints defined")

        self._check_ready_state()

    def _build_constraints(self) -> Constraints:
        """Build constraints object from UI inputs."""
        if self.constraints:
            return self.constraints

        # If no constraints are defined, create a default constraints object
        if self.database:
            default_constraints = Constraints(self.database)
            return default_constraints

        return None

    def _build_target_profile(self) -> ViscosityProfile:
        """Build target viscosity profile from UI inputs."""
        viscosities = [self.target_spins[sr].value()
                       for sr in self.SHEAR_RATES]
        return ViscosityProfile(
            shear_rates=self.SHEAR_RATES,
            viscosities=viscosities
        )

    def start_optimization(self) -> None:
        """Start the optimization process."""
        try:
            # Build constraints and target profile
            constraints = self._build_constraints()
            self.target_profile = self._build_target_profile()

            if constraints is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Constraints Error",
                    "Unable to build constraints. Please check your database connection."
                )
                return

            # Create optimizer
            seed = self.seed_spin.value() if self.use_seed_check.isChecked() else None

            self.optimizer = Optimizer(
                constraints=constraints,
                predictor=self.predictor,
                target=self.target_profile,
                maxiter=self.maxiter_spin.value(),
                popsize=self.popsize_spin.value(),
                tol=self.tolerance_spin.value(),
                seed=seed
            )

            # Update UI state - disable optimization button
            self.optimize_btn.setEnabled(False)
            # No stop functionality in main thread
            self.stop_btn.setEnabled(False)
            self.progress_label.setText("Optimization in progress...")

            # Force UI update
            QtWidgets.QApplication.processEvents()
            mutation = (self.mutation_min_spin.value(),
                        self.mutation_max_spin.value())
            atol = self.atol_spin.value()
            recombination = self.recombination_spin.value()
            try:
                result = self.optimizer.optimize(
                    strategy=self.strategy_combo.currentText(),
                    mutation=mutation,
                    recombination=recombination,
                    init=self.init_combo.currentText(),
                    atol=atol,
                    workers=1,
                )
                self.on_optimization_finished(result)
            except Exception as opt_error:
                self.on_optimization_error(str(opt_error))

        except Exception as e:
            Log.e(TAG, f"Failed to start optimization: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Optimization Error",
                f"Failed to start optimization:\n{str(e)}"
            )

    def stop_optimization(self) -> None:
        """Stop the optimization process."""
        # Not applicable for main thread execution
        QtWidgets.QMessageBox.information(
            self,
            "Stop Optimization",
            "Optimization cannot be stopped when running on main thread."
        )

    def on_optimization_progress(self, message: str) -> None:
        """Handle optimization progress updates."""
        self.progress_label.setText(message)
        QtWidgets.QApplication.processEvents()  # Force UI update

    def on_optimization_finished(self, result: Formulation) -> None:
        """Handle optimization completion."""
        self.optimal_formulation = result

        # Update UI state
        self.optimize_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_results_btn.setEnabled(True)
        self.save_plot_btn.setEnabled(True)

        self.progress_label.setText("Optimization completed successfully!")

        # Display results
        self._display_results()
        self.update_plot()

    def on_optimization_error(self, error_message: str) -> None:
        """Handle optimization errors."""
        self.optimize_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        self.progress_label.setText("Optimization failed")

        QtWidgets.QMessageBox.critical(
            self,
            "Optimization Error",
            f"Optimization failed:\n{error_message}"
        )

    def _display_results(self) -> None:
        """Display optimization results in the table."""
        if not self.optimal_formulation:
            return

        # Convert formulation to dictionary for display
        form_dict = self.optimal_formulation.to_dict()

        self.results_table.setRowCount(len(form_dict))

        row = 0
        for component, value in form_dict.items():
            self.results_table.setItem(
                row, 0, QtWidgets.QTableWidgetItem(component))

            if isinstance(value, (int, float)):
                self.results_table.setItem(
                    row, 1, QtWidgets.QTableWidgetItem(f"{value:.3f}"))
            else:
                self.results_table.setItem(
                    row, 1, QtWidgets.QTableWidgetItem(str(value)))

            # Add units based on component type
            units = ""
            for ing_type, unit in self.INGREDIENT_UNITS.items():
                if ing_type.lower() in component.lower():
                    units = unit
                    break
            if "temperature" in component.lower():
                units = "°C"

            self.results_table.setItem(
                row, 2, QtWidgets.QTableWidgetItem(units))
            row += 1

        self.results_table.resizeColumnsToContents()

    def update_plot(self) -> None:
        """Update the viscosity profile comparison plot."""
        if not self.optimal_formulation or not self.target_profile:
            return

        self.profile_figure.clear()
        ax = self.profile_figure.add_subplot(111)

        # Get predicted profile for optimal formulation
        try:
            df = self.optimal_formulation.to_dataframe(training=False)
            pred = self.predictor.predict(df)
            pred_viscosities = pred.flatten().tolist()

            # Plot target profile
            ax.plot(
                self.target_profile.shear_rates,
                self.target_profile.viscosities,
                'o-',
                color='#e74c3c',
                linewidth=2.5,
                markersize=8,
                label='Target Profile',
                zorder=5
            )

            # Plot predicted profile
            ax.plot(
                self.SHEAR_RATES,
                pred_viscosities,
                's-',
                color='#3498db',
                linewidth=2.5,
                markersize=8,
                label='Predicted Profile',
                zorder=5
            )

            if self.log_scale_check.isChecked():
                ax.set_xscale('log')

            ax.set_xlabel('Shear Rate (s⁻¹)', fontsize=11, color="#4C566A")
            ax.set_ylabel('Viscosity (cP)', fontsize=11, color="#4C566A")
            ax.set_title(
                'Target vs Predicted Viscosity Profile',
                fontsize=13,
                fontweight=600,
                color="#2E3440",
                pad=15
            )

            ax.legend(
                loc='best',
                fontsize=10,
                frameon=True,
                framealpha=0.95,
                edgecolor="#E5E9F0",
                fancybox=False
            )

            ax.grid(True, alpha=0.2, which="major",
                    linestyle="-", linewidth=0.5, color="#D8DEE9")
            ax.grid(True, alpha=0.1, which="minor",
                    linestyle="-", linewidth=0.3, color="#D8DEE9")

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color("#E5E9F0")
            ax.spines['bottom'].set_color("#E5E9F0")
            ax.spines['left'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)
            ax.tick_params(colors="#4C566A", which="both", labelsize=9)

        except Exception as e:
            Log.e(TAG, f"Failed to update plot: {e}")

        self.profile_figure.tight_layout()
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
            # Reset target profile
            for spin in self.target_spins.values():
                spin.setValue(10.0)

            # Reset constraints
            self.constraints = None
            self.constraints_status_label.setText("No constraints defined")

            # Reset optimization settings
            self.maxiter_spin.setValue(100)
            self.popsize_spin.setValue(15)
            self.tolerance_spin.setValue(1e-6)
            self.use_seed_check.setChecked(False)
            self.seed_spin.setValue(42)

            # Clear results
            self.optimal_formulation = None
            self.target_profile = None
            self.results_table.setRowCount(0)

            # Clear plot
            self.profile_figure.clear()
            self.profile_canvas.draw()

            # Reset buttons
            self.save_results_btn.setEnabled(False)
            self.save_plot_btn.setEnabled(False)

            self._check_ready_state()

    def save_results(self) -> None:
        """Save optimization results."""
        if not self.optimal_formulation:
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Optimization Results",
            "optimization_results.json",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            try:
                # Get constraints information
                constraints_info = {}
                if self.constraints:
                    try:
                        bounds, encoding = self.constraints.build()
                        constraints_info = {
                            'bounds': bounds,
                            'encoding': encoding,
                            'num_constraints': len([enc for enc in encoding if enc.get('choices') or enc.get('range')])
                        }
                    except Exception as e:
                        Log.w(TAG, f"Could not serialize constraints: {e}")
                        constraints_info = {
                            'error': 'Could not serialize constraints'}

                # Prepare results data
                data = {
                    'target_profile': {
                        'shear_rates': self.target_profile.shear_rates,
                        'viscosities': self.target_profile.viscosities
                    },
                    'optimal_formulation': self.optimal_formulation.to_dict(),
                    'constraints': constraints_info,
                    'optimization_settings': {
                        'max_iterations': self.maxiter_spin.value(),
                        'population_size': self.popsize_spin.value(),
                        'tolerance': self.tolerance_spin.value(),
                        'seed': self.seed_spin.value() if self.use_seed_check.isChecked() else None
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
            "optimization_profile.png",
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
    window = OptimizationUI()
    window.show()
    sys.exit(app.exec_())
