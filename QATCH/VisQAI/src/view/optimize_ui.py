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
    1.0
"""

import sys
import os
import traceback
from typing import Optional, List, Dict, Any, TYPE_CHECKING
import json
import numpy as np
import pandas as pd
from datetime import datetime
import csv
from datetime import datetime
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

from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from src.models.formulation import Formulation, ViscosityProfile
    from src.models.predictor import Predictor
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.utils.constraints import Constraints
    from src.processors.optimizer import Optimizer, OptimizationStatus
    from src.view.constraints_ui import ConstraintsUI
    from src.threads.executor import Executor, ExecutionRecord
    if TYPE_CHECKING:
        from src.view.main_window import VisQAIWindow
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.formulation import Formulation, ViscosityProfile
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.utils.constraints import Constraints
    from QATCH.VisQAI.src.processors.optimizer import Optimizer, OptimizationStatus
    from QATCH.VisQAI.src.view.constraints_ui import ConstraintsUI
    from QATCH.VisQAI.src.threads.executor import Executor, ExecutionRecord
    if TYPE_CHECKING:
        from QATCH.VisQAI.src.view.main_window import VisQAIWindow

TAG = "[OptimizationUI]"


class OptimizationHelpDialog(QtWidgets.QDialog):
    """Dialog showing detailed help for optimization parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimization Settings Help")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the help dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)

        # Create tabs for different categories
        tabs = QtWidgets.QTabWidget()

        # Basic Settings Tab
        basic_widget = self._create_basic_help()
        tabs.addTab(basic_widget, "Basic Settings")

        # Advanced Settings Tab
        advanced_widget = self._create_advanced_help()
        tabs.addTab(advanced_widget, "Advanced Settings")

        # Strategy Guide Tab
        strategy_widget = self._create_strategy_help()
        tabs.addTab(strategy_widget, "Strategy Guide")

        layout.addWidget(tabs)

        # Close button
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Close)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)

    def _create_basic_help(self) -> QtWidgets.QWidget:
        """Create basic settings help page."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)
        text.setHtml("""
        <h2>Basic Optimization Settings</h2>
        
        <h3>Max Iterations</h3>
        <p>Maximum number of optimization iterations to perform.</p>
        <ul>
            <li><b>Range:</b> 10-1000</li>
            <li><b>Default:</b> 100</li>
            <li><b>Recommendation:</b> Start with 100. Increase if optimization hasn't converged.</li>
        </ul>
        
        <h3>Population Size</h3>
        <p>Number of candidate solutions maintained in each iteration. Larger populations explore 
        the search space more thoroughly but increase computation time.</p>
        <ul>
            <li><b>Range:</b> 5-50</li>
            <li><b>Default:</b> 15</li>
            <li><b>Recommendation:</b> 15 × (number of parameters) is a good starting point.</li>
        </ul>
        
        <h3>Tolerance</h3>
        <p>Relative convergence threshold. Optimization stops when the fractional improvement 
        between iterations falls below this value.</p>
        <ul>
            <li><b>Range:</b> 1e-9 to 1e-3</li>
            <li><b>Default:</b> 1e-6</li>
            <li><b>Recommendation:</b> Use 1e-6 for most cases. Lower values for high precision.</li>
        </ul>
        
        <h3>Absolute Tolerance</h3>
        <p>Absolute convergence threshold. Optimization stops when the absolute improvement 
        falls below this value.</p>
        <ul>
            <li><b>Range:</b> 0-10</li>
            <li><b>Default:</b> 0 (disabled)</li>
            <li><b>Recommendation:</b> Usually left at 0; use relative tolerance instead.</li>
        </ul>
        """)

        layout.addWidget(text)
        return widget

    def _create_advanced_help(self) -> QtWidgets.QWidget:
        """Create advanced settings help page."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)
        text.setHtml("""
        <h2>Advanced Optimization Settings</h2>
        
        <h3>Random Seed</h3>
        <p>Fixed seed for reproducible optimization results. Useful for debugging and 
        comparing different settings.</p>
        <ul>
            <li><b>Range:</b> 0-999999</li>
            <li><b>Default:</b> 42 (when enabled)</li>
            <li><b>Recommendation:</b> Enable for reproducibility, disable for production runs.</li>
        </ul>
        
        <h3>Mutation (Min/Max)</h3>
        <p>Controls the mutation constant F, which scales the difference vectors used to 
        create new candidates. Can be a single value or a range for dithering.</p>
        <ul>
            <li><b>Range:</b> 0.0-2.0</li>
            <li><b>Default:</b> [0.5, 1.0]</li>
            <li><b>Effects:</b>
                <ul>
                    <li>Low values (&lt;0.5): More conservative, slower convergence</li>
                    <li>High values (&gt;1.0): More exploratory, may overshoot</li>
                    <li>Dithering (min≠max): Adapts during optimization</li>
                </ul>
            </li>
        </ul>
        
        <h3>Recombination</h3>
        <p>Crossover probability that determines how much of the mutant vector is mixed 
        with the target vector.</p>
        <ul>
            <li><b>Range:</b> 0.0-1.0</li>
            <li><b>Default:</b> 0.7</li>
            <li><b>Effects:</b>
                <ul>
                    <li>Low values (&lt;0.5): Less mixing, slower convergence</li>
                    <li>High values (&gt;0.7): More mixing, faster but less stable</li>
                </ul>
            </li>
        </ul>
        
        <h3>Initialization</h3>
        <p>Method for generating the initial population.</p>
        <ul>
            <li><b>latinhypercube:</b> Ensures good coverage (recommended)</li>
            <li><b>sobol:</b> Low-discrepancy sequence, excellent coverage</li>
            <li><b>halton:</b> Alternative low-discrepancy sequence</li>
            <li><b>random:</b> Standard random initialization</li>
        </ul>
        """)

        layout.addWidget(text)
        return widget

    def _create_strategy_help(self) -> QtWidgets.QWidget:
        """Create strategy guide help page."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)
        text.setHtml("""
        <h2>Differential Evolution Strategies</h2>
        
        <p>Strategy format: <b>[base]/[# of diff vectors]/[crossover type]</b></p>
        
        <h3>Base Vector Selection</h3>
        <ul>
            <li><b>best:</b> Uses the current best solution (exploitative)</li>
            <li><b>rand:</b> Uses a random solution (explorative)</li>
            <li><b>randtobest:</b> Combines random and best (balanced)</li>
        </ul>
        
        <h3>Number of Difference Vectors</h3>
        <ul>
            <li><b>1:</b> Single difference vector (faster, less diverse)</li>
            <li><b>2:</b> Two difference vectors (slower, more diverse)</li>
        </ul>
        
        <h3>Crossover Type</h3>
        <ul>
            <li><b>bin:</b> Binomial crossover (standard)</li>
            <li><b>exp:</b> Exponential crossover (alternative)</li>
        </ul>
        
        <h3>Recommended Strategies</h3>
        <table border="1" cellpadding="5" style="border-collapse: collapse;">
            <tr style="background-color: #f0f0f0;">
                <th>Strategy</th>
                <th>Use Case</th>
                <th>Characteristics</th>
            </tr>
            <tr>
                <td><b>best1bin</b></td>
                <td>Default choice</td>
                <td>Fast convergence, good for unimodal problems</td>
            </tr>
            <tr>
                <td><b>rand1bin</b></td>
                <td>Multimodal problems</td>
                <td>Better exploration, slower convergence</td>
            </tr>
            <tr>
                <td><b>best2bin</b></td>
                <td>Need more diversity</td>
                <td>More exploration than best1bin</td>
            </tr>
            <tr>
                <td><b>randtobest1bin</b></td>
                <td>Balanced approach</td>
                <td>Good mix of exploration/exploitation</td>
            </tr>
        </table>
        
        <h3>Tips</h3>
        <ul>
            <li>Start with <b>best1bin</b> for most problems</li>
            <li>Switch to <b>rand1bin</b> if stuck in local minima</li>
            <li>Use <b>best2bin</b> for increased diversity</li>
            <li>Exponential crossover (exp) rarely outperforms binomial (bin)</li>
        </ul>
        """)

        layout.addWidget(text)
        return widget


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
    optimization_finished = QtCore.pyqtSignal(object)  # result
    optimization_error = QtCore.pyqtSignal(str)  # error message
    optimization_progress = QtCore.pyqtSignal(
        object)  # OptimizationStatus object

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

        self.executor = Executor()
        self.optimization_finished.connect(self.on_optimization_finished)
        self.optimization_error.connect(self.on_optimization_error)
        self.optimization_progress.connect(self._handle_progress_update)
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
        self.current_constraints = None
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
                if hasattr(protein, 'class_type') and protein.class_type:
                    if protein.class_type not in self.class_types:
                        self.class_types.append(protein.class_type)
                    if protein.class_type not in self.proteins_by_class:
                        self.proteins_by_class[protein.class_type] = []
                    self.proteins_by_class[protein.class_type].append(
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

    def open_constraints_dialog(self) -> None:
        """Open the constraints configuration dialog."""
        if not self.database or not self.ing_ctrl:
            QtWidgets.QMessageBox.warning(
                self,
                "Database Error",
                "Database not initialized. Cannot configure constraints."
            )
            return

        # Create constraints UI only once
        if self.constraints_ui is None:
            self.constraints_ui = ConstraintsUI(self, step=6)

        # Show the existing dialog (preserves all UI state)
        self.constraints_ui.add_suggestion_dialog()

    def set_constraints(self, constraints: Constraints) -> None:
        """Store the constraints (called from ConstraintsUI.accept_suggestions)."""
        self.current_constraints = constraints
        # Update your status label
        num_constraints = len(constraints._ranges) + len(constraints._choices)
        self.constraints_status_label.setText(
            f"{num_constraints} constraint(s) defined")

    def _create_optimization_settings_group(self) -> QtWidgets.QGroupBox:
        """Create optimization settings group with improved UI/UX."""
        group = QtWidgets.QGroupBox("Optimization Settings")
        main_layout = QtWidgets.QVBoxLayout(group)

        # Add info button at the top
        header_layout = QtWidgets.QHBoxLayout()
        info_label = QtWidgets.QLabel(
            "Configure optimization parameters")
        info_label.setStyleSheet(
            "color: #666; font-style: italic; font-size: 10px;")
        header_layout.addWidget(info_label)
        header_layout.addStretch()

        info_button = QtWidgets.QPushButton("ℹ Help")
        info_button.setMaximumWidth(80)
        info_button.setToolTip("Show detailed help for optimization settings")
        info_button.clicked.connect(self._show_optimization_help)
        header_layout.addWidget(info_button)
        main_layout.addLayout(header_layout)

        main_layout.addSpacing(5)

        # Create form layout for settings
        form_layout = QtWidgets.QFormLayout()
        form_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.ExpandingFieldsGrow)
        form_layout.setLabelAlignment(QtCore.Qt.AlignRight)
        form_layout.setVerticalSpacing(10)

        # Consistent widget width
        widget_width = 200

        # Basic Settings Label
        basic_label = QtWidgets.QLabel("Basic Settings")
        basic_label.setStyleSheet(
            "font-weight: bold; color: #333; font-size: 11px;")
        form_layout.addRow(basic_label)

        # Max iterations
        self.maxiter_spin = QtWidgets.QSpinBox()
        self.maxiter_spin.setRange(10, 1000)
        self.maxiter_spin.setValue(100)
        self.maxiter_spin.setMaximumWidth(widget_width)
        self.maxiter_spin.setToolTip(
            "Maximum number of optimization iterations.\n"
            "Increase if optimization hasn't converged.\n"
            "Typical range: 50-200")
        form_layout.addRow("Max Iterations:", self.maxiter_spin)

        # Population size
        self.popsize_spin = QtWidgets.QSpinBox()
        self.popsize_spin.setRange(5, 50)
        self.popsize_spin.setValue(15)
        self.popsize_spin.setMaximumWidth(widget_width)
        self.popsize_spin.setToolTip(
            "Number of candidate solutions per iteration.\n"
            "Rule of thumb: 15 × (number of parameters)\n"
            "Larger = better exploration, but slower")
        form_layout.addRow("Population Size:", self.popsize_spin)

        # Tolerance
        self.tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.tolerance_spin.setRange(1e-9, 1e-3)
        self.tolerance_spin.setValue(1e-6)
        self.tolerance_spin.setDecimals(9)
        self.tolerance_spin.setMaximumWidth(widget_width)
        self.tolerance_spin.setStepType(
            QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.tolerance_spin.setToolTip(
            "Relative convergence threshold.\n"
            "Stops when improvement < tolerance.\n"
            "1e-6 is good for most cases")
        form_layout.addRow("Tolerance:", self.tolerance_spin)

        # Strategy
        self.strategy_combo = QtWidgets.QComboBox()
        self.strategy_combo.addItems([
            "best1bin", "best1exp", "best2bin", "best2exp",
            "rand1bin", "rand1exp", "rand2bin", "rand2exp",
            "randtobest1bin", "randtobest1exp"
        ])
        self.strategy_combo.setCurrentText("best1bin")
        self.strategy_combo.setMaximumWidth(widget_width)
        self.strategy_combo.setToolTip(
            "DE strategy: [base]/[#vectors]/[crossover]\n"
            "best1bin (default) - Fast, good for most problems\n"
            "rand1bin - Better for multimodal problems\n"
            "Click Help for detailed strategy guide")
        form_layout.addRow("Strategy:", self.strategy_combo)

        # Section separator
        separator1 = QtWidgets.QFrame()
        separator1.setFrameShape(QtWidgets.QFrame.HLine)
        separator1.setFrameShadow(QtWidgets.QFrame.Sunken)
        separator1.setStyleSheet("QFrame { color: #CCC; }")
        form_layout.addRow(separator1)

        # Advanced Settings Label
        advanced_label = QtWidgets.QLabel("Advanced Settings")
        advanced_label.setStyleSheet(
            "font-weight: bold; color: #333; font-size: 11px;")
        form_layout.addRow(advanced_label)

        # Mutation Min
        self.mutation_min_spin = QtWidgets.QDoubleSpinBox()
        self.mutation_min_spin.setRange(0.0, 2.0)
        self.mutation_min_spin.setValue(0.5)
        self.mutation_min_spin.setSingleStep(0.1)
        self.mutation_min_spin.setMaximumWidth(widget_width)
        self.mutation_min_spin.setToolTip(
            "Minimum mutation factor F.\n"
            "0.5 = conservative, 1.0 = aggressive\n"
            "Use [0.5, 1.0] for dithering")
        form_layout.addRow("Mutation Min (F):", self.mutation_min_spin)

        # Mutation Max
        self.mutation_max_spin = QtWidgets.QDoubleSpinBox()
        self.mutation_max_spin.setRange(0.0, 2.0)
        self.mutation_max_spin.setValue(1.0)
        self.mutation_max_spin.setSingleStep(0.1)
        self.mutation_max_spin.setMaximumWidth(widget_width)
        self.mutation_max_spin.setToolTip(
            "Maximum mutation factor F.\n"
            "Set equal to min for constant F")
        form_layout.addRow("Mutation Max (F):", self.mutation_max_spin)

        # Recombination
        self.recombination_spin = QtWidgets.QDoubleSpinBox()
        self.recombination_spin.setRange(0.0, 1.0)
        self.recombination_spin.setValue(0.7)
        self.recombination_spin.setSingleStep(0.1)
        self.recombination_spin.setMaximumWidth(widget_width)
        self.recombination_spin.setToolTip(
            "Crossover probability (CR).\n"
            "0.7 is standard for most problems.\n"
            "Higher = more mixing, faster convergence")
        form_layout.addRow("Recombination (CR):", self.recombination_spin)

        # Initialization
        self.init_combo = QtWidgets.QComboBox()
        self.init_combo.addItems(
            ['latinhypercube', 'sobol', 'halton', 'random'])
        self.init_combo.setCurrentText("latinhypercube")
        self.init_combo.setMaximumWidth(widget_width)
        self.init_combo.setToolTip(
            "Initial population generation method.\n"
            "latinhypercube (default) - Good coverage\n"
            "sobol/halton - Low-discrepancy sequences\n"
            "random - Standard random")
        form_layout.addRow("Initialization:", self.init_combo)

        # Absolute tolerance
        self.atol_spin = QtWidgets.QSpinBox()
        self.atol_spin.setRange(0, 10)
        self.atol_spin.setValue(0)
        self.atol_spin.setMaximumWidth(widget_width)
        self.atol_spin.setToolTip(
            "Absolute convergence threshold.\n"
            "Usually left at 0 (disabled).\n"
            "Use relative tolerance instead")
        form_layout.addRow("Absolute Tolerance:", self.atol_spin)

        # Random seed
        seed_widget = QtWidgets.QWidget()
        seed_layout = QtWidgets.QHBoxLayout(seed_widget)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.setSpacing(8)

        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        self.seed_spin.setEnabled(False)
        self.seed_spin.setMaximumWidth(120)
        self.seed_spin.setToolTip("Random seed value (0-999999)")
        seed_layout.addWidget(self.seed_spin)

        self.use_seed_check = QtWidgets.QCheckBox("Use seed")
        self.use_seed_check.setToolTip(
            "Enable for reproducible results.\n"
            "Useful for debugging and comparisons")
        self.use_seed_check.toggled.connect(self.seed_spin.setEnabled)
        seed_layout.addWidget(self.use_seed_check)

        seed_layout.addStretch()

        form_layout.addRow("Random Seed:", seed_widget)

        main_layout.addLayout(form_layout)

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
        """Create a modern, resizable results group with progress tracking."""
        group = QtWidgets.QGroupBox("Optimal Formulation")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # Progress section
        progress_layout = QtWidgets.QVBoxLayout()
        progress_layout.setSpacing(4)

        # Status label
        self.progress_label = QtWidgets.QLabel("Ready")
        progress_layout.addWidget(self.progress_label)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready")
        self.progress_bar.setMinimumHeight(25)
        progress_layout.addWidget(self.progress_bar)

        layout.addLayout(progress_layout)

        # Separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        separator.setStyleSheet("background-color: #cccccc;")
        layout.addWidget(separator)

        # Results table with 4 columns
        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(
            ["Component", "Ingredient", "Concentration", "Units"])
        self.results_table.horizontalHeader().setStretchLastSection(False)
        self.results_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents)  # Component
        self.results_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch)  # Ingredient (stretches)
        self.results_table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeToContents)  # Concentration
        self.results_table.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.ResizeToContents)  # Units
        self.results_table.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # Enable alternating row colors
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #f5f5f5;
                background-color: white;
            }
        """)
        layout.addWidget(self.results_table)

        # Export button
        export_layout = QtWidgets.QHBoxLayout()
        export_layout.addStretch()

        self.export_btn = QtWidgets.QPushButton("Export Results")
        self.export_btn.setMinimumWidth(150)
        # Disabled until results are available
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_results_to_csv)
        export_layout.addWidget(self.export_btn)
        layout.addLayout(export_layout)
        return group
    # Add this method to your class as well:

    def _show_optimization_help(self):
        """Show the optimization help dialog."""
        dialog = OptimizationHelpDialog(self)
        dialog.exec_()

    def _create_visualization_group(self) -> QtWidgets.QGroupBox:
        """Create a modern, resizable visualization group."""
        group = QtWidgets.QGroupBox("Viscosity Profile")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # Matplotlib figure
        self.profile_figure = Figure(figsize=(8, 5))
        self.profile_canvas = FigureCanvas(self.profile_figure)
        self.profile_canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self.profile_canvas, stretch=1)

        # Plot options bar
        options_layout = QtWidgets.QHBoxLayout()
        options_layout.setSpacing(10)

        self.log_scale_xaxis_check = QtWidgets.QCheckBox("Log Scale X-axis")
        self.log_scale_xaxis_check.setChecked(True)
        self.log_scale_xaxis_check.toggled.connect(self.update_plot)
        options_layout.addWidget(self.log_scale_xaxis_check)

        self.log_scale_yaxis_check = QtWidgets.QCheckBox("Log Scale Y-axis")
        self.log_scale_yaxis_check.setChecked(True)
        self.log_scale_yaxis_check.toggled.connect(self.update_plot)
        options_layout.addWidget(self.log_scale_yaxis_check)

        self.show_ci_check = QtWidgets.QCheckBox("Show CI")
        self.show_ci_check.setChecked(True)
        self.show_ci_check.toggled.connect(self.update_plot)
        options_layout.addWidget(self.show_ci_check)
        self.show_target_profile_check = QtWidgets.QCheckBox(
            "Show Target Viscosity Profile")
        self.show_target_profile_check.setChecked(True)
        self.show_target_profile_check.toggled.connect(self.update_plot)
        options_layout.addWidget(self.show_target_profile_check)

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

    def _update_optimization_progress(self, status) -> None:
        """Update UI with optimization progress."""
        # Update progress bar
        self.progress_bar.setValue(int(status.progress_percent))

        # Calculate improvement if we have history
        if hasattr(self.optimizer, '_initial_best'):
            improvement = self.optimizer._initial_best - status.best_value
            improvement_pct = (
                improvement / self.optimizer._initial_best) * 100

            self.progress_bar.setFormat(
                f"{status.iteration}/{status.num_iterations} - "
                f"Best: {status.best_value:.6f} "
                f"(↓{improvement_pct:.1f}%)"
            )
        else:
            self.progress_bar.setFormat(
                f"{status.iteration}/{status.num_iterations} - "
                f"Best: {status.best_value:.6f}"
            )

        # Detailed label
        self.progress_label.setText(
            f"Iteration {status.iteration}/{status.num_iterations} "
            f"({status.progress_percent:.1f}%) - "
            f"Best MSE: {status.best_value:.6f}"
        )

        # Keep UI responsive
        QtWidgets.QApplication.processEvents()

    def start_optimization(self) -> None:
        """Start the optimization process using executor."""
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

            # Update UI state
            self.optimize_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

            # Reset and show progress bar
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setValue(0)
                self.progress_bar.setFormat("Starting optimization...")

            if hasattr(self, 'progress_label'):
                self.progress_label.setText("Optimization in progress...")

            # Show starting status
            if hasattr(self, 'status_label'):
                self.status_label.setText("Starting optimization...")

            # Force UI update
            QtWidgets.QApplication.processEvents()

            # Get optimization parameters
            opt_params = {
                'strategy': self.strategy_combo.currentText(),
                'mutation': (self.mutation_min_spin.value(),
                             self.mutation_max_spin.value()),
                'recombination': self.recombination_spin.value(),
                'init': self.init_combo.currentText(),
                'atol': self.atol_spin.value(),
                'workers': 1,
            }

            # Start optimization in background thread using executor
            self.current_optimization_record = self.executor.run(
                obj=self,
                method_name='_run_optimization_worker',
                opt_params=opt_params,
                thread_name='VisQAI-Optimization',
                callback=self._on_optimization_complete
            )

            Log.i(TAG, "Optimization started in background thread")

        except Exception as e:
            Log.e(TAG, f"Failed to start optimization: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Optimization Error",
                f"Failed to start optimization:\n{str(e)}"
            )
            # Re-enable buttons on error
            self.optimize_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

        except Exception as e:
            Log.e(TAG, f"Failed to start optimization: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Optimization Error",
                f"Failed to start optimization:\n{str(e)}"
            )

    def _run_optimization_worker(self, opt_params: Dict[str, Any]) -> Any:
        """
        Worker method that runs in background thread.
        This method should NOT directly update any UI elements.
        """
        try:
            result = self.optimizer.optimize(
                strategy=opt_params['strategy'],
                mutation=opt_params['mutation'],
                recombination=opt_params['recombination'],
                init=opt_params['init'],
                atol=opt_params['atol'],
                workers=opt_params['workers'],
                progress_callback=self._thread_safe_progress_callback  # Pass callback
            )
            return result

        except Exception as e:
            # Re-raise to be caught by executor
            raise

    def _on_optimization_complete(self, record: ExecutionRecord) -> None:
        """
        Callback executed after optimization thread completes.
        This is called from the background thread, so use signals for UI updates.
        """
        if record.exception:
            # Emit error signal
            error_msg = str(record.exception)
            if record.traceback:
                Log.e(TAG, f"Traceback:\n{record.traceback}")
                error_msg = f"{error_msg}\n\nTraceback:\n{record.traceback}"
            self.optimization_error.emit(error_msg)
        else:
            # Emit success signal with result
            self.optimization_finished.emit(record.result)

    def _thread_safe_progress_callback(self, status: OptimizationStatus) -> None:
        """
        Progress callback that's called from the background thread.
        Uses signals to safely update UI from main thread.

        Args:
            status: OptimizationStatus object containing all progress information
        """
        # Emit the entire status object to the main thread
        # The signal will handle thread-safe delivery to the UI
        self.optimization_progress.emit(status)

    @QtCore.pyqtSlot(object)
    def _handle_progress_update(self, status: OptimizationStatus) -> None:
        """
        Slot that handles progress updates on the main thread.
        Safe to update UI here.

        Args:
            status: OptimizationStatus object with progress information
        """
        # Update progress bar if you have one
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(int(status.progress_percent))
            self.progress_bar.setFormat(f"{status.progress_percent:.1f}%")

        # Update progress label with detailed status
        if hasattr(self, 'progress_label'):
            self.progress_label.setText(
                f"Iteration {status.iteration}/{status.num_iterations}"
            )

        # Update status label with best value
        if hasattr(self, 'status_label'):
            self.status_label.setText(
                f"Iteration {status.iteration}/{status.num_iterations} - "
                f"Best MSE: {status.best_value:.6e} - "
                f"Convergence: {status.convergence:.6e}"
            )

        # Log progress
        if status.iteration % 10 == 0:  # Log every 10 iterations
            Log.d(TAG,
                  f"Progress: {status.progress_percent:.1f}% - "
                  f"Best MSE: {status.best_value:.6e}")

    def stop_optimization(self) -> None:
        """Stop the running optimization."""
        if self.current_optimization_record and self.current_optimization_record.is_alive():
            Log.i(TAG, "Stopping optimization...")

            # Request stop in optimizer
            if hasattr(self, 'optimizer') and self.optimizer:
                self.optimizer.stop()

            # Update UI
            if hasattr(self, 'status_label'):
                self.status_label.setText("Stopping optimization...")
            self.stop_btn.setEnabled(False)
        else:
            Log.w(TAG, "No active optimization to stop")

    def on_optimization_progress(self, message: str) -> None:
        """Handle optimization progress updates."""
        self.progress_label.setText(message)
        QtWidgets.QApplication.processEvents()  # Force UI update

    @QtCore.pyqtSlot(object)
    def on_optimization_finished(self, result: any) -> None:
        """Handle optimization completion."""
        self.optimal_formulation = result
        self._display_results()

        # Update progress display
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Optimization complete!")
        self.progress_label.setText(
            "Optimization complete - Results displayed below")
        # Re-enable buttons
        self.optimize_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

        df = self.optimal_formulation.to_dataframe(
            encoded=False, training=False)
        self.pred, self.pred_stats = self.predictor.predict_uncertainty(df)
        self.update_plot()

    @QtCore.pyqtSlot(str)
    def on_optimization_error(self, error_message: str) -> None:
        """
        Handle optimization errors.
        This runs on the main thread via signal connection.
        """
        # Update UI state
        self.optimize_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # Update progress
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Failed")

        if hasattr(self, 'progress_label'):
            self.progress_label.setText("Optimization failed")

        if hasattr(self, 'status_label'):
            self.status_label.setText("Optimization failed")

        # Show error dialog
        QtWidgets.QMessageBox.critical(
            self,
            "Optimization Error",
            f"Optimization failed:\n\n{error_message}"
        )

        Log.e(TAG, f"Optimization failed: {error_message}")

    def _display_results(self) -> None:
        """Display optimization results in the table with separate columns."""
        if not self.optimal_formulation:
            return

        # Build list of components to display
        components = []

        # Protein
        if self.optimal_formulation.protein:
            components.append({
                'component': 'Protein',
                'ingredient': self.optimal_formulation.protein.ingredient.name,
                'concentration': self.optimal_formulation.protein.concentration,
                'units': self.optimal_formulation.protein.units,
            })

        # Buffer
        if self.optimal_formulation.buffer:
            components.append({
                'component': 'Buffer',
                'ingredient': self.optimal_formulation.buffer.ingredient.name,
                'concentration': self.optimal_formulation.buffer.concentration,
                'units': self.optimal_formulation.buffer.units,
            })

        # Stabilizer
        if self.optimal_formulation.stabilizer:
            components.append({
                'component': 'Stabilizer',
                'ingredient': self.optimal_formulation.stabilizer.ingredient.name,
                'concentration': self.optimal_formulation.stabilizer.concentration,
                'units': self.optimal_formulation.stabilizer.units,
            })

        # Salt
        if self.optimal_formulation.salt:
            components.append({
                'component': 'Salt',
                'ingredient': self.optimal_formulation.salt.ingredient.name,
                'concentration': self.optimal_formulation.salt.concentration,
                'units': self.optimal_formulation.salt.units,
            })

        # Surfactant
        if self.optimal_formulation.surfactant:
            components.append({
                'component': 'Surfactant',
                'ingredient': self.optimal_formulation.surfactant.ingredient.name,
                'concentration': self.optimal_formulation.surfactant.concentration,
                'units': self.optimal_formulation.surfactant.units,
            })

        # Excipient
        if self.optimal_formulation.excipient:
            components.append({
                'component': 'Excipient',
                'ingredient': self.optimal_formulation.excipient.ingredient.name,
                'concentration': self.optimal_formulation.excipient.concentration,
                'units': self.optimal_formulation.excipient.units,
            })

        # Temperature
        if self.optimal_formulation.temperature is not None:
            components.append({
                'component': 'Temperature',
                'ingredient': '—',  # Em dash for N/A
                'concentration': self.optimal_formulation.temperature,
                'units': '\u00B0C'
            })

        # Set table row count
        self.results_table.setRowCount(len(components))

        # Define colors for styling
        color_row_even = "#F8F9FA"  # Light gray
        color_row_odd = "#FFFFFF"   # White

        # Populate table
        for row, comp in enumerate(components):
            # Component type (bold)
            type_item = QtWidgets.QTableWidgetItem(comp['component'])
            type_font = type_item.font()
            type_font.setBold(True)
            type_font.setPointSize(10)
            type_item.setFont(type_font)
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)

            # Ingredient name
            ingredient_item = QtWidgets.QTableWidgetItem(comp['ingredient'])
            ingredient_item.setFlags(
                ingredient_item.flags() & ~Qt.ItemIsEditable)

            # Concentration (right-aligned for better readability)
            conc_item = QtWidgets.QTableWidgetItem(
                f"{comp['concentration']:.3f}")
            conc_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            conc_item.setFlags(conc_item.flags() & ~Qt.ItemIsEditable)

            # Units
            units_item = QtWidgets.QTableWidgetItem(comp['units'])
            units_item.setFlags(units_item.flags() & ~Qt.ItemIsEditable)

            # Apply alternating row colors
            bg_color = QtGui.QColor(color_row_even if row %
                                    2 == 0 else color_row_odd)
            type_item.setBackground(bg_color)
            ingredient_item.setBackground(bg_color)
            conc_item.setBackground(bg_color)
            units_item.setBackground(bg_color)

            # Set items in table
            self.results_table.setItem(row, 0, type_item)
            self.results_table.setItem(row, 1, ingredient_item)
            self.results_table.setItem(row, 2, conc_item)
            self.results_table.setItem(row, 3, units_item)

        # Adjust column widths
        self.results_table.resizeColumnsToContents()

        # Enable export button
        self.export_btn.setEnabled(True)

    def update_plot(self) -> None:
        """Update the viscosity profile comparison plot with professional styling."""
        if not self.optimal_formulation or not self.target_profile:
            return
        title_name = "Optimized Viscosity Profile"
        self.profile_figure.clear()
        ax = self.profile_figure.add_subplot(111)

        # Get predicted profile for optimal formulation
        try:

            pred, pred_stats = self.pred, self.pred_stats
            pred_viscosities = pred.flatten().tolist()
            lower_ci = pred_stats['lower_ci']
            upper_ci = pred_stats['upper_ci']
            upper_ci = upper_ci.flatten()
            lower_ci = lower_ci.flatten()
            if self.show_target_profile_check.isChecked():
                title_name = "Target vs. Optimized Viscosity Profile"
                ax.plot(
                    self.target_profile.shear_rates,
                    self.target_profile.viscosities,
                    color="#BF616A",
                    linewidth=2.5,
                    markersize=8,
                    label='Target Profile',
                    zorder=5
                )

            # Plot predicted profile
            ax.plot(
                self.SHEAR_RATES,
                pred_viscosities,
                color="#32E2DF",
                linewidth=2.5,
                markersize=8,
                label='Optimized Profile',
                zorder=5
            )
            if self.show_ci_check.isChecked():
                ax.fill_between(
                    self.target_profile.shear_rates,
                    lower_ci,
                    upper_ci,
                    alpha=0.2,
                    color="#69EAC5",
                    label='95% CI'
                )

            # Set scale
            if self.log_scale_xaxis_check.isChecked():
                ax.set_xscale('log')
            if self.log_scale_yaxis_check.isChecked():
                ax.set_yscale('log')

            # Axis labels with larger font
            ax.set_xlabel('Shear Rate (s⁻¹)', fontsize=12,
                          fontweight='bold', color="#4C566A")
            ax.set_ylabel('Viscosity (cP)', fontsize=12,
                          fontweight='bold', color="#4C566A")

            ax.set_title(
                title_name,
                fontsize=12,
                fontweight='bold',
                color="#2E3440",
                pad=15
            )
            ax.legend(
                loc='best',
                fontsize=10,
                frameon=True,
                framealpha=0.95,
                edgecolor="#D8DEE9",
                fancybox=False,
                shadow=False
            )

            # Grid styling
            ax.grid(True, alpha=0.3, which="major",
                    linestyle="-", linewidth=0.8, color="#D8DEE9")
            if self.log_scale_xaxis_check.isChecked():
                ax.grid(True, alpha=0.15, which="minor",
                        linestyle="-", linewidth=0.4, color="#D8DEE9")

            # Spine styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color("#D8DEE9")
            ax.spines['bottom'].set_color("#D8DEE9")
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

            # Tick styling
            ax.tick_params(
                colors="#4C566A",
                which="both",
                labelsize=10,
                width=1.5,
                length=6
            )

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

    def closeEvent(self, event) -> None:
        """Clean up threads on close."""
        # Check if optimization is running
        if self.current_optimization_record and self.current_optimization_record.is_alive():
            reply = QtWidgets.QMessageBox.question(
                self,
                "Optimization Running",
                "An optimization is currently running. Do you want to stop it and close?",
                QtWidgets.QMessageBox.StandardButton.Yes |
                QtWidgets.QMessageBox.StandardButton.No
            )

            if reply == QtWidgets.QMessageBox.StandardButton.No:
                event.ignore()
                return

            # Stop the optimization
            if hasattr(self, 'optimizer') and self.optimizer:
                self.optimizer.stop()

        # Wait for all threads to complete (with timeout)
        Log.i(TAG, "Waiting for threads to complete...")
        self.executor.join_all(timeout=5.0)

        # Clean up
        self.executor.cleanup_finished()

        Log.i(TAG, "Widget closed, threads cleaned up")
        super().closeEvent(event)

    def _export_results_to_csv(self) -> None:
        """Export optimization results to CSV file."""
        if not self.optimal_formulation:
            QtWidgets.QMessageBox.warning(
                self,
                "No Results",
                "No optimization results to export."
            )
            return

        # Get save file path
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Results to CSV",
            "optimization_results.csv",
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return  # User cancelled

        try:

            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                writer.writerow(['Component', 'Ingredient',
                                'Concentration', 'Units'])

                # Write formulation data
                if self.optimal_formulation.protein:
                    writer.writerow([
                        'Protein',
                        self.optimal_formulation.protein.ingredient.name,
                        f"{self.optimal_formulation.protein.concentration:.3f}",
                        self.optimal_formulation.protein.units,
                    ])

                if self.optimal_formulation.buffer:
                    writer.writerow([
                        'Buffer',
                        self.optimal_formulation.buffer.ingredient.name,
                        f"{self.optimal_formulation.buffer.concentration:.3f}",
                        self.optimal_formulation.buffer.units,
                    ])

                if self.optimal_formulation.stabilizer:
                    writer.writerow([
                        'Stabilizer',
                        self.optimal_formulation.stabilizer.ingredient.name,
                        f"{self.optimal_formulation.stabilizer.concentration:.3f}",
                        self.optimal_formulation.stabilizer.units,
                    ])

                if self.optimal_formulation.salt:
                    writer.writerow([
                        'Salt',
                        self.optimal_formulation.salt.ingredient.name,
                        f"{self.optimal_formulation.salt.concentration:.3f}",
                        self.optimal_formulation.salt.units,
                    ])

                if self.optimal_formulation.surfactant:
                    writer.writerow([
                        'Surfactant',
                        self.optimal_formulation.surfactant.ingredient.name,
                        f"{self.optimal_formulation.surfactant.concentration:.3f}",
                        self.optimal_formulation.surfactant.units,
                    ])

                if self.optimal_formulation.excipient:
                    writer.writerow([
                        'Excipient',
                        self.optimal_formulation.excipient.ingredient.name,
                        f"{self.optimal_formulation.excipient.concentration:.3f}",
                        self.optimal_formulation.excipient.units,
                    ])

                if self.optimal_formulation.temperature is not None:
                    writer.writerow([
                        'Temperature',
                        '—',
                        f"{self.optimal_formulation.temperature:.3f}",
                        '\u00B0C'
                    ])

                # Add blank rows for separation
                writer.writerow([])
                writer.writerow([])

                # Write optimization parameters metadata
                writer.writerow(['Optimization Parameters'])
                writer.writerow(['Parameter', 'Value'])
                writer.writerow(
                    ['Export Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow(['Max Iterations', self.maxiter_spin.value()])
                writer.writerow(['Population Size', self.popsize_spin.value()])
                writer.writerow(['Tolerance', self.tolerance_spin.value()])
                writer.writerow(
                    ['Strategy', self.strategy_combo.currentText()])
                writer.writerow(
                    ['Mutation Range', f"{self.mutation_min_spin.value()} - {self.mutation_max_spin.value()}"])
                writer.writerow(
                    ['Recombination', self.recombination_spin.value()])
                writer.writerow(
                    ['Initialization', self.init_combo.currentText()])
                writer.writerow(['Seed', self.seed_spin.value(
                ) if self.use_seed_check.isChecked() else 'Random'])

                # Add target profile if available
                if self.target_profile:
                    writer.writerow([])
                    writer.writerow([])
                    writer.writerow(['Target Viscosity Profile'])
                    writer.writerow(['Shear Rate (1/s)', 'Viscosity (cP)'])
                    for shear, visc in zip(self.target_profile.shear_rates, self.target_profile.viscosities):
                        writer.writerow([shear, visc])

            QtWidgets.QMessageBox.information(
                self,
                "Export Successful",
                f"Results exported successfully to:\n{file_path}"
            )

        except Exception as e:
            Log.e(TAG, f"Failed to export results: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export results:\n{str(e)}"
            )

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
