"""
evaluation_ui.py

This module provides the EvaluationUI class for the Evaluate tab in VisQAI.
It supports model performance evaluation against existing data on both overall
and per-shear-rate basis with customizable metrics and visualizations.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-10-22

Version:
    2.0 - Improved with zip export, per-run investigation, better formulation loading
"""

import os
import copy
import zipfile
import traceback
from typing import Optional, List, Dict, TYPE_CHECKING
from pathlib import Path
import tempfile
import shutil

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

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

try:
    from src.models.formulation import Formulation
    from src.models.predictor import Predictor
    from src.utils.metrics import Metrics
    from src.io.file_storage import SecureOpen
    from src.io.parser import Parser
    if TYPE_CHECKING:
        from src.view.main_window import VisQAIWindow
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.formulation import Formulation
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.utils.metrics import Metrics
    from QATCH.VisQAI.src.io.file_storage import SecureOpen
    from QATCH.VisQAI.src.io.parser import Parser
    if TYPE_CHECKING:
        from QATCH.VisQAI.src.view.main_window import VisQAIWindow


class EvaluationUI(QtWidgets.QDialog):
    """
    UI for evaluating model performance against experimental data.

    This class provides an interface for:
    - Selecting runs and formulations to evaluate
    - Configuring evaluation metrics and shear rates
    - Displaying results in tables and plots
    - Comparing predictions vs actual values
    - Investigating individual run performance
    """

    # Metric descriptions
    METRIC_DESCRIPTIONS = {
        'mae': 'Mean Absolute Error - Average absolute difference between predicted and actual values',
        'rmse': 'Root Mean Square Error - Square root of average squared differences',
        'mape': 'Mean Absolute Percentage Error - Average percentage error',
        'r2': 'R² Score - Coefficient of determination (1.0 = perfect fit)',
        'coverage': 'Prediction Interval Coverage - % of actual values within confidence intervals',
        'max_error': 'Maximum Error - Largest absolute error in predictions',
        'median_ae': 'Median Absolute Error - Middle value of absolute errors',
        'explained_variance': 'Explained Variance - Proportion of variance explained by model'
    }

    def __init__(self, parent=None):
        """Initialize the EvaluationUI.

        Args:
            parent: Parent VisQAIWindow instance.
        """
        super().__init__(parent)
        self.parent: 'VisQAIWindow' = parent
        self.setWindowTitle("Model Evaluation")

        # Initialize data structures
        self.model_path: Optional[str] = None
        self.predictor: Optional[Predictor] = None
        self.metrics_calculator = Metrics()
        self.current_results_df: Optional[pd.DataFrame] = None
        self.selected_formulations: List[Formulation] = []
        self.available_metrics = self.metrics_calculator.get_available_metrics()

        # Store formulations by run for per-run investigation
        self.formulations_by_run: Dict[str, List[Formulation]] = {}
        self.run_results: Dict[str, pd.DataFrame] = {}

        # File tracking for runs (like in frame_step1.py)
        self.all_files = {}
        self.run_file_run = None
        self.run_file_xml = None
        self.run_file_analyze = None

        # Default selected metrics
        self.selected_metrics = ['mae', 'rmse', 'mape', 'r2', 'coverage']

        # Shear rate options
        self.shear_rates = {
            'Viscosity_100': 100,
            'Viscosity_1000': 1000,
            'Viscosity_10000': 10000,
            'Viscosity_100000': 100000,
            'Viscosity_15000000': 15000000
        }

        # Selected shear rates for evaluation
        self.selected_shear_rates = list(self.shear_rates.keys())

        # Initialize file dialogs (same pattern as frame_step1.py)
        self._init_file_dialogs()

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface components."""
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)

        # Create tab widget for main sections
        self.tab_widget = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Tab 1: Overall Evaluation
        overall_widget = self.create_overall_evaluation_tab()
        self.tab_widget.addTab(overall_widget, "Overall Evaluation")

        # Tab 2: Per-Run Investigation
        per_run_widget = self.create_per_run_investigation_tab()
        self.tab_widget.addTab(per_run_widget, "Per-Run Investigation")

        # Status bar at the bottom
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; padding: 5px; }")
        main_layout.addWidget(self.status_label)

        # Set window properties
        self.resize(1400, 800)

    def _init_file_dialogs(self):
        """Initialize file dialogs with settings matching frame_step1.py."""
        # Model dialog setup (matching frame_step1.py lines 117-130)
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

        # Run file dialog setup (matching frame_step1.py lines 154-160)
        self.file_dialog = QtWidgets.QFileDialog()
        self.file_dialog.setOption(
            QtWidgets.QFileDialog.DontUseNativeDialog, True)
        # NOTE: `setDirectory()` called when VisQAI mode is enabled.
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self.file_dialog.setNameFilter("Captured Runs (capture.zip)")
        self.file_dialog.selectNameFilter("Captured Runs (capture.zip)")

    def create_overall_evaluation_tab(self):
        """Create the overall evaluation tab."""
        widget = QtWidgets.QWidget()

        # Create splitter for horizontal layout
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        layout = QtWidgets.QVBoxLayout(widget)
        layout.addWidget(splitter)

        # Left panel - Configuration
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        # Model selection group
        self.select_model_group = QtWidgets.QGroupBox("Select Model")
        select_model_layout = QtWidgets.QHBoxLayout(self.select_model_group)
        self.select_model_btn = QtWidgets.QPushButton("Browse...")
        self.select_model_label = QtWidgets.QLineEdit()
        self.select_model_label.setPlaceholderText("No model selected")
        self.select_model_label.setReadOnly(True)

        select_model_layout.addWidget(self.select_model_btn)
        select_model_layout.addWidget(self.select_model_label)
        select_model_layout.addStretch()

        left_layout.addWidget(self.select_model_group)

        # Data selection group
        data_group = QtWidgets.QGroupBox("Import Experiments")
        data_layout = QtWidgets.QVBoxLayout(data_group)

        # Browse run button and list
        form_layout = QtWidgets.QFormLayout()

        self.select_run = QtWidgets.QPushButton("Add Run...")
        self.select_label = QtWidgets.QLineEdit()
        self.select_label.setPlaceholderText("No run selected")
        self.select_label.setReadOnly(True)

        # List view for runs
        self.list_view = QtWidgets.QListView()
        self.list_view.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.model = QtGui.QStandardItemModel()
        self.list_view_addPlaceholderText()
        self.list_view.setModel(self.model)
        self.list_view.clicked.connect(self.user_run_clicked)

        form_layout.addRow(self.select_run, self.list_view)
        data_layout.addLayout(form_layout)

        # Control buttons for run management
        run_controls = QtWidgets.QHBoxLayout()

        self.remove_run_btn = QtWidgets.QPushButton("Remove Selected")
        self.remove_run_btn.clicked.connect(self.user_run_removed)

        self.remove_all_btn = QtWidgets.QPushButton("Remove All")
        self.remove_all_btn.clicked.connect(self.user_all_runs_removed)

        run_controls.addWidget(self.remove_run_btn)
        run_controls.addWidget(self.remove_all_btn)
        run_controls.addStretch()

        data_layout.addLayout(run_controls)

        # Formulation info
        self.formulation_info = QtWidgets.QTextEdit()
        self.formulation_info.setReadOnly(True)
        self.formulation_info.setMaximumHeight(80)
        self.formulation_info.setPlaceholderText(
            "Formulation information will appear here...")
        data_layout.addWidget(self.formulation_info)

        left_layout.addWidget(data_group)

        # Evaluation settings group
        settings_group = QtWidgets.QGroupBox("Evaluation Settings")
        settings_layout = QtWidgets.QVBoxLayout(settings_group)

        # Metric selection
        metrics_label = QtWidgets.QLabel("Select Metrics:")
        settings_layout.addWidget(metrics_label)

        self.metrics_list = QtWidgets.QListWidget()
        self.metrics_list.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection)

        for metric in self.available_metrics:
            item = QtWidgets.QListWidgetItem(self.format_metric_name(metric))
            item.setData(Qt.UserRole, metric)
            self.metrics_list.addItem(item)
            if metric in self.selected_metrics:
                item.setSelected(True)

        self.metrics_list.itemSelectionChanged.connect(
            self.on_metric_selection_changed)
        self.metrics_list.setMaximumHeight(150)
        settings_layout.addWidget(self.metrics_list)

        # Shear rate selection
        shear_label = QtWidgets.QLabel("Select Shear Rates:")
        settings_layout.addWidget(shear_label)

        self.shear_rate_list = QtWidgets.QListWidget()
        self.shear_rate_list.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection)

        for shear_name, shear_value in self.shear_rates.items():
            item = QtWidgets.QListWidgetItem(
                f"{shear_name} ({shear_value} s⁻¹)")
            item.setData(Qt.UserRole, shear_name)
            self.shear_rate_list.addItem(item)
            if shear_name in self.selected_shear_rates:
                item.setSelected(True)

        self.shear_rate_list.itemSelectionChanged.connect(
            self.on_shear_rate_selection_changed)
        self.shear_rate_list.setMaximumHeight(120)
        settings_layout.addWidget(self.shear_rate_list)

        # Evaluation type radio buttons
        eval_type_label = QtWidgets.QLabel("Evaluation Type:")
        settings_layout.addWidget(eval_type_label)

        self.overall_radio = QtWidgets.QRadioButton("Overall Metrics")
        self.per_shear_radio = QtWidgets.QRadioButton(
            "Per-Shear Rate Metrics")
        self.both_radio = QtWidgets.QRadioButton("Both")
        self.both_radio.setChecked(True)

        eval_type_layout = QtWidgets.QHBoxLayout()
        eval_type_layout.addWidget(self.overall_radio)
        eval_type_layout.addWidget(self.per_shear_radio)
        eval_type_layout.addWidget(self.both_radio)
        settings_layout.addLayout(eval_type_layout)

        left_layout.addWidget(settings_group)

        # Action buttons
        action_layout = QtWidgets.QHBoxLayout()

        self.evaluate_btn = QtWidgets.QPushButton("Evaluate Model")
        self.evaluate_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        self.evaluate_btn.clicked.connect(self.evaluate_model)

        self.export_btn = QtWidgets.QPushButton("Export Results (ZIP)")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results_as_zip)

        action_layout.addWidget(self.evaluate_btn)
        action_layout.addWidget(self.export_btn)

        left_layout.addLayout(action_layout)
        left_layout.addStretch()

        # Right panel - Results display
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        # Results tabs
        self.results_tabs = QtWidgets.QTabWidget()

        # Metrics table tab
        self.metrics_table = QtWidgets.QTableWidget()
        self.results_tabs.addTab(self.metrics_table, "Metrics")

        # Plot tab
        plot_widget = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_widget)

        # Plot controls
        plot_controls = QtWidgets.QHBoxLayout()
        plot_type_label = QtWidgets.QLabel("Plot Type:")
        self.plot_type_combo = QtWidgets.QComboBox()
        self.plot_type_combo.addItems([
            "Predicted vs Actual",
            "Residuals Distribution",
            "Q-Q Plot",
            "Overall Metrics",
            "Metrics vs Shear Rate",
            "Error vs Shear Rate",
            "Confidence Intervals"
        ])
        self.plot_type_combo.currentTextChanged.connect(self.update_plot)

        plot_controls.addWidget(plot_type_label)
        plot_controls.addWidget(self.plot_type_combo)
        plot_controls.addStretch()
        plot_layout.addLayout(plot_controls)

        # Plot canvas
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        self.results_tabs.addTab(plot_widget, "Plots")

        # Detailed results tab
        self.details_table = QtWidgets.QTableWidget()
        self.results_tabs.addTab(self.details_table, "Detailed Results")

        right_layout.addWidget(self.results_tabs)

        # Add panels to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([500, 900])

        # Connect signals
        self.select_model_btn.clicked.connect(self.select_model)
        self.select_run.clicked.connect(self.user_run_browse)

        return widget

    def create_per_run_investigation_tab(self):
        """Create the per-run investigation tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)

        # Left panel - Run selection
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)

        # Run selection group
        run_group = QtWidgets.QGroupBox("Select Run to Investigate")
        run_layout = QtWidgets.QVBoxLayout(run_group)

        self.run_combo = QtWidgets.QComboBox()
        self.run_combo.currentTextChanged.connect(
            self.on_run_selected_for_investigation)
        run_layout.addWidget(self.run_combo)

        # Run info
        self.run_info_text = QtWidgets.QTextEdit()
        self.run_info_text.setReadOnly(True)
        self.run_info_text.setMaximumHeight(150)
        run_layout.addWidget(QtWidgets.QLabel("Run Information:"))
        run_layout.addWidget(self.run_info_text)

        # Metrics for selected run
        self.run_metrics_table = QtWidgets.QTableWidget()
        run_layout.addWidget(QtWidgets.QLabel("Run Metrics:"))
        run_layout.addWidget(self.run_metrics_table)

        left_layout.addWidget(run_group)
        left_layout.addStretch()

        # Right panel - Visualization
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        # Plot controls
        plot_controls = QtWidgets.QHBoxLayout()
        plot_controls.addWidget(QtWidgets.QLabel("Visualization:"))

        self.run_plot_type = QtWidgets.QComboBox()
        self.run_plot_type.addItems([
            "Viscosity Profile Comparison",
            "Relative Error by Shear Rate",
            "Residuals Analysis",
            "Component Contributions"
        ])
        self.run_plot_type.currentTextChanged.connect(self.update_run_plot)
        plot_controls.addWidget(self.run_plot_type)
        plot_controls.addStretch()

        right_layout.addLayout(plot_controls)

        # Plot canvas for per-run analysis
        self.run_figure = Figure(figsize=(10, 7))
        self.run_canvas = FigureCanvas(self.run_figure)
        right_layout.addWidget(self.run_canvas)

        # Details table for selected run
        self.run_details_table = QtWidgets.QTableWidget()
        self.run_details_table.setMaximumHeight(200)
        right_layout.addWidget(QtWidgets.QLabel("Detailed Predictions:"))
        right_layout.addWidget(self.run_details_table)

        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 2)

        return widget

    def list_view_addPlaceholderText(self):
        """Add placeholder text to the list view."""
        placeholder_item = QtGui.QStandardItem("No runs imported")
        placeholder_item.setEnabled(False)
        placeholder_item.setSelectable(False)
        font = placeholder_item.font()
        font.setItalic(True)
        placeholder_item.setFont(font)
        placeholder_item.setForeground(QtGui.QBrush(Qt.gray))
        self.model.appendRow(placeholder_item)

    def select_model(self):
        """Handle model selection using persistent dialog (matching frame_step1.py pattern)."""
        if self.model_dialog.exec_():
            selected_files = self.model_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                try:
                    self.model_path = file_path
                    self.predictor = Predictor(file_path)
                    # Extract filename without path and extension (matching frame_step1.py line 2018-2019)
                    display_name = file_path.split(
                        '\\')[-1].split('/')[-1].split('.')[0]
                    self.select_model_label.setText(display_name)
                    self.status_label.setText(
                        f"Model loaded: {display_name}")
                    Log.i("EvaluationUI", f"Model loaded: {file_path}")
                    self.check_ready_to_evaluate()
                except Exception as e:
                    Log.e("EvaluationUI", f"Failed to load model: {e}")
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Model Loading Error",
                        f"Failed to load model: {str(e)}"
                    )
                    self.model_path = None
                    self.predictor = None
                    self.select_model_label.setText("Failed to load model")

    def user_run_browse(self):
        """Handle run file selection using persistent dialog (matching frame_step1.py pattern)."""
        if self.file_dialog.exec_():
            selected_files = self.file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                self.add_run_file(file_path)

    def add_run_file(self, file_path: str):
        """Add a run file and load its formulation."""
        try:
            # Parse the run file - create Parser with file_path and get single formulation
            parser = Parser(file_path)
            formulation = parser.get_formulation()

            if not formulation:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Formulation",
                    "No valid formulation found in the selected file."
                )
                return

            # Get run name
            run_name = os.path.basename(file_path)

            # Store formulation by run (as a list with single item for consistency)
            self.formulations_by_run[run_name] = [formulation]

            # Add formulation to the selected list
            self.selected_formulations.append(formulation)

            # Update the list view
            if self.model.rowCount() == 1 and not self.model.item(0).isEnabled():
                self.model.clear()

            run_item = QtGui.QStandardItem(run_name)
            run_item.setData(file_path, Qt.UserRole)
            self.model.appendRow(run_item)

            # Update run combo for per-run investigation
            self.run_combo.addItem(run_name)

            # Update formulation info
            self.update_formulation_info()

            self.status_label.setText(
                f"Added formulation from {run_name}")
            self.check_ready_to_evaluate()

        except Exception as e:
            Log.e("EvaluationUI", f"Failed to load run: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Run Loading Error",
                f"Failed to load run file: {str(e)}"
            )

    def update_formulation_info(self):
        """Update the formulation information display."""
        if not self.selected_formulations:
            self.formulation_info.setText("No formulations loaded")
            return

        info_text = f"Total Formulations: {len(self.selected_formulations)}\n"
        info_text += f"Runs Loaded: {len(self.formulations_by_run)}\n"

        # Count formulations per run
        for run_name, forms in self.formulations_by_run.items():
            info_text += f"  • {run_name}: {len(forms)} formulations\n"

        self.formulation_info.setText(info_text)

    def check_ready_to_evaluate(self):
        """Check if ready to evaluate and update UI accordingly."""
        ready = (self.predictor is not None and
                 len(self.selected_formulations) > 0 and
                 len(self.selected_metrics) > 0)

        self.evaluate_btn.setEnabled(ready)

        if ready:
            self.status_label.setText("Ready to evaluate")
        elif not self.predictor:
            self.status_label.setText("Please select a model")
        elif not self.selected_formulations:
            self.status_label.setText("Please add run files with formulations")
        else:
            self.status_label.setText("Please select metrics")

    def user_run_clicked(self, index):
        """Handle run item click in list view."""
        item = self.model.itemFromIndex(index)
        if item and item.isEnabled():
            file_path = item.data(Qt.UserRole)
            if file_path:
                self.select_label.setText(os.path.basename(file_path))

    def user_run_removed(self):
        """Remove selected run from the list."""
        selected_indexes = self.list_view.selectedIndexes()
        if not selected_indexes:
            return

        for index in reversed(selected_indexes):
            item = self.model.itemFromIndex(index)
            if item and item.isEnabled():
                run_name = item.text()

                # Remove formulations for this run
                if run_name in self.formulations_by_run:
                    forms_to_remove = self.formulations_by_run[run_name]
                    for form in forms_to_remove:
                        if form in self.selected_formulations:
                            self.selected_formulations.remove(form)
                    del self.formulations_by_run[run_name]

                    # Remove from per-run combo
                    idx = self.run_combo.findText(run_name)
                    if idx >= 0:
                        self.run_combo.removeItem(idx)

                self.model.removeRow(index.row())

        if self.model.rowCount() == 0:
            self.list_view_addPlaceholderText()
            self.select_label.clear()

        self.update_formulation_info()
        self.check_ready_to_evaluate()

    def user_all_runs_removed(self):
        """Remove all runs from the list."""
        self.model.clear()
        self.selected_formulations.clear()
        self.formulations_by_run.clear()
        self.run_combo.clear()
        self.list_view_addPlaceholderText()
        self.select_label.clear()
        self.update_formulation_info()
        self.check_ready_to_evaluate()

    def on_metric_selection_changed(self):
        """Handle metric selection change."""
        self.selected_metrics = []
        for i in range(self.metrics_list.count()):
            item = self.metrics_list.item(i)
            if item.isSelected():
                self.selected_metrics.append(item.data(Qt.UserRole))

        self.check_ready_to_evaluate()

    def on_shear_rate_selection_changed(self):
        """Handle shear rate selection change."""
        self.selected_shear_rates = []
        for i in range(self.shear_rate_list.count()):
            item = self.shear_rate_list.item(i)
            if item.isSelected():
                self.selected_shear_rates.append(item.data(Qt.UserRole))

    def format_metric_name(self, metric: str) -> str:
        """Format metric name for display."""
        names = {
            'mae': 'MAE (Mean Absolute Error)',
            'rmse': 'RMSE (Root Mean Square Error)',
            'mape': 'MAPE (Mean Absolute Percentage Error)',
            'r2': 'R² Score',
            'coverage': 'Coverage (95% CI)',
            'max_error': 'Maximum Error',
            'median_ae': 'Median Absolute Error',
            'explained_variance': 'Explained Variance'
        }
        return names.get(metric, metric.upper())

    def evaluate_model(self):
        """Perform model evaluation using Predictor's evaluate() method."""
        if not self.predictor or not self.selected_formulations:
            return

        try:
            QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
            self.status_label.setText("Evaluating model...")

            # Store results by run for per-run investigation
            self.run_results.clear()
            all_results_list = []

            # Process each run separately to maintain run association
            for run_name, formulations in self.formulations_by_run.items():
                Log.i(
                    f"Evaluating run: {run_name} with {len(formulations)} formulations")

                # Convert formulations to DataFrame for predictor.evaluate()
                # Note: Each formulation is already a single formulation object
                eval_dfs = []
                for form in formulations:
                    try:
                        # Convert formulation to DataFrame (encoded=False, training=True)
                        form_df = form.to_dataframe(
                            encoded=False, training=True)
                        eval_dfs.append(form_df)
                    except Exception as e:
                        Log.w(
                            f"Failed to convert formulation to DataFrame: {e}")
                        continue

                if not eval_dfs:
                    Log.w(
                        f"No valid formulations to evaluate in run: {run_name}")
                    continue

                # Combine all formulations for this run
                eval_data = pd.concat(eval_dfs, ignore_index=True)

                # Get target shear rate columns (matching selected shear rates)
                target_cols = [f"Viscosity_{rate}" for rate in
                               [self.shear_rates[name] for name in self.selected_shear_rates]]

                Log.i(
                    f"Evaluating {len(eval_data)} samples with targets: {target_cols}")

                # Use predictor's evaluate method for efficient batch evaluation
                results_df = self.predictor.evaluate(
                    eval_data=eval_data,
                    targets=target_cols,
                    n_samples=None  # Use default uncertainty sampling
                )

                # Add run name and formulation info to results
                results_df['run'] = run_name

                # Map sample indices back to formulation IDs
                formulation_ids = []
                for idx in results_df['sample_idx']:
                    if idx < len(formulations):
                        formulation_ids.append(formulations[idx].id)
                    else:
                        formulation_ids.append(f"sample_{idx}")
                results_df['formulation_id'] = formulation_ids

                # Rename columns to match expected format
                results_df = results_df.rename(columns={
                    'pct_error': 'percentage_error'
                })

                # Extract numeric shear rate from column name (e.g., "Viscosity_100" -> 100)
                # The 'shear_rate' column contains the column name like "Viscosity_100"
                def extract_numeric_shear_rate(shear_col_name):
                    try:
                        # Extract number from "Viscosity_XXX" format
                        if isinstance(shear_col_name, str) and shear_col_name.startswith("Viscosity_"):
                            return int(shear_col_name.split("_")[1])
                        # If already numeric, return as is
                        return float(shear_col_name)
                    except (ValueError, IndexError):
                        return shear_col_name

                # Keep the original column name for reference, but also add numeric version
                results_df['shear_rate_name'] = results_df['shear_rate']
                results_df['shear_rate'] = results_df['shear_rate_name'].apply(
                    extract_numeric_shear_rate)

                # Store per-run results
                self.run_results[run_name] = results_df.copy()

                # Add to overall results
                all_results_list.append(results_df)

            # Combine all results
            if all_results_list:
                self.current_results_df = pd.concat(
                    all_results_list, ignore_index=True)

                Log.i(
                    f"Evaluation complete: {len(self.current_results_df)} predictions")

                # Display results
                self.display_results()

                # Enable export
                self.export_btn.setEnabled(True)

                self.status_label.setText(
                    f"Evaluation complete: {len(self.current_results_df)} predictions")
            else:
                raise ValueError("No valid evaluation results generated")

        except Exception as e:
            Log.e("EvaluationUI", f"Evaluation failed: {e}")
            Log.e("EvaluationUI", traceback.format_exc())
            QtWidgets.QMessageBox.critical(
                self,
                "Evaluation Error",
                f"Failed to evaluate model: {str(e)}"
            )
            self.status_label.setText("Evaluation failed")

        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def display_results(self):
        """Display evaluation results in tables and plots."""
        if self.current_results_df is None or self.current_results_df.empty:
            return

        # Display metrics
        self.display_metrics_table()

        # Display detailed results
        self.display_details_table()

        # Update plot
        self.update_plot()

    def display_metrics_table(self):
        """Display metrics in the metrics table."""
        metrics_data = []

        # Overall metrics
        if self.overall_radio.isChecked() or self.both_radio.isChecked():
            overall_metrics = self.metrics_calculator.compute_overall(
                self.current_results_df,
                self.selected_metrics
            )

            for metric, value in overall_metrics.items():
                metrics_data.append({
                    'Type': 'Overall',
                    'Metric': self.format_metric_name(metric),
                    'Value': f"{value:.4f}",
                    'Description': self.METRIC_DESCRIPTIONS.get(metric, '')
                })

        # Per-shear rate metrics
        if self.per_shear_radio.isChecked() or self.both_radio.isChecked():
            per_shear_metrics = self.metrics_calculator.compute_per_shear_rate(
                self.current_results_df,
                self.selected_metrics
            )

            for _, row in per_shear_metrics.iterrows():
                for metric in self.selected_metrics:
                    if metric in row:
                        metrics_data.append({
                            'Type': f"Shear Rate {row['shear_rate']:.0f}",
                            'Metric': self.format_metric_name(metric),
                            'Value': f"{row[metric]:.4f}",
                            'Description': self.METRIC_DESCRIPTIONS.get(metric, '')
                        })

        # Populate table
        df = pd.DataFrame(metrics_data)
        self.metrics_table.setRowCount(len(df))
        self.metrics_table.setColumnCount(len(df.columns))
        self.metrics_table.setHorizontalHeaderLabels(df.columns.tolist())

        for i, row in df.iterrows():
            for j, value in enumerate(row):
                item = QtWidgets.QTableWidgetItem(str(value))
                self.metrics_table.setItem(i, j, item)

        self.metrics_table.resizeColumnsToContents()
        self.metrics_table.horizontalHeader().setStretchLastSection(True)

    def display_details_table(self):
        """Display detailed results in the details table."""
        df = self.current_results_df.copy()

        # Format columns for display
        display_columns = ['run', 'formulation_id', 'shear_rate', 'actual',
                           'predicted', 'abs_error', 'percentage_error',
                           'lower_95', 'upper_95']

        df = df[display_columns]

        # Set up table
        self.details_table.setRowCount(len(df))
        self.details_table.setColumnCount(len(df.columns))
        self.details_table.setHorizontalHeaderLabels(df.columns.tolist())

        # Populate table
        for i, row in df.iterrows():
            for j, value in enumerate(row):
                if isinstance(value, float):
                    item = QtWidgets.QTableWidgetItem(f"{value:.4f}")
                else:
                    item = QtWidgets.QTableWidgetItem(str(value))
                self.details_table.setItem(i, j, item)

        self.details_table.resizeColumnsToContents()

    def update_plot(self):
        """Update the plot based on selected plot type."""
        if self.current_results_df is None or self.current_results_df.empty:
            return

        self.figure.clear()

        plot_type = self.plot_type_combo.currentText()

        if plot_type == "Predicted vs Actual":
            self.plot_predicted_vs_actual()
        elif plot_type == "Residuals Distribution":
            self.plot_residuals_distribution()
        elif plot_type == "Q-Q Plot":
            self.plot_qq()
        elif plot_type == "Overall Metrics":
            self.plot_metrics_comparison_with_descriptions()
        elif plot_type == "Metrics vs Shear Rate":
            self.plot_shear_rate_performance()
        elif plot_type == "Error vs Shear Rate":
            self.plot_error_vs_shear_rate()
        elif plot_type == "Confidence Intervals":
            self.plot_confidence_intervals()

        self.canvas.draw()

    def plot_predicted_vs_actual(self):
        """Create predicted vs actual scatter plot."""
        ax = self.figure.add_subplot(111)
        df = self.current_results_df

        # Create scatter plot
        scatter = ax.scatter(df['actual'], df['predicted'],
                             c=df['percentage_error'], cmap='coolwarm',
                             s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

        # Add perfect prediction line
        min_val = min(df['actual'].min(), df['predicted'].min())
        max_val = max(df['actual'].max(), df['predicted'].max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'k--', alpha=0.5, label='Perfect Prediction')

        # Add ±10% error bands
        ax.fill_between([min_val, max_val],
                        [min_val*0.9, max_val*0.9],
                        [min_val*1.1, max_val*1.1],
                        alpha=0.2, color='gray', label='±10% Error')

        ax.set_xlabel('Actual Viscosity (cP)', fontsize=12)
        ax.set_ylabel('Predicted Viscosity (cP)', fontsize=12)
        ax.set_title('Predicted vs Actual Viscosity',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = self.figure.colorbar(scatter, ax=ax)
        cbar.set_label('Percentage Error (%)', rotation=270, labelpad=20)

        # Add R² annotation
        from sklearn.metrics import r2_score
        r2 = r2_score(df['actual'], df['predicted'])
        ax.text(0.05, 0.95, f'R² = {r2:.4f}',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def plot_residuals_distribution(self):
        """Create residuals distribution histogram."""
        ax = self.figure.add_subplot(111)
        df = self.current_results_df

        # Create histogram
        n, bins, patches = ax.hist(df['residual'], bins=30,
                                   edgecolor='black', alpha=0.7)

        # Color bars by distance from zero
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor('#ff7f0e')
            else:
                patch.set_facecolor('#1f77b4')

        # Add zero line
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Zero')

        # Add normal distribution overlay
        from scipy import stats
        mu, std = df['residual'].mean(), df['residual'].std()
        x = np.linspace(df['residual'].min(), df['residual'].max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, std) * len(df) * (df['residual'].max() - df['residual'].min()) / 30,
                'r-', linewidth=2, label='Normal Distribution')

        ax.set_xlabel('Residuals (cP)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Residuals',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add statistics annotation
        ax.text(0.70, 0.95, f'Mean: {mu:.4f}\nStd: {std:.4f}',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def plot_qq(self):
        """Create Q-Q plot."""
        from scipy import stats

        ax = self.figure.add_subplot(111)
        df = self.current_results_df

        stats.probplot(df['residual'], dist="norm", plot=ax)
        ax.set_title('Q-Q Plot of Residuals', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def plot_metrics_comparison_with_descriptions(self):
        """Create bar chart comparing metrics with descriptions."""
        # Create subplot layout for metrics and descriptions
        gs = self.figure.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        ax_bars = self.figure.add_subplot(gs[0])
        ax_text = self.figure.add_subplot(gs[1])
        ax_text.axis('off')

        # Get overall metrics
        overall_metrics = self.metrics_calculator.compute_overall(
            self.current_results_df,
            self.selected_metrics[:5]  # Limit to 5 metrics for clarity
        )

        metrics_names = list(overall_metrics.keys())
        metrics_values = list(overall_metrics.values())

        x = np.arange(len(metrics_names))
        bars = ax_bars.bar(x, metrics_values, color=plt.cm.Set3(
            np.linspace(0, 1, len(metrics_names))))

        ax_bars.set_xlabel('Metrics', fontsize=12)
        ax_bars.set_ylabel('Value', fontsize=12)
        ax_bars.set_title('Overall Performance Metrics',
                          fontsize=14, fontweight='bold')
        ax_bars.set_xticks(x)
        ax_bars.set_xticklabels([self.format_metric_name(m).split('(')[0].strip()
                                 for m in metrics_names])

        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax_bars.text(bar.get_x() + bar.get_width()/2., height,
                         f'{value:.3f}', ha='center', va='bottom')

        ax_bars.grid(True, alpha=0.3, axis='y')

        # Add metric descriptions as text
        descriptions_text = "Metric Descriptions:\n"
        for metric in metrics_names:
            desc = self.METRIC_DESCRIPTIONS.get(metric, '')
            short_name = self.format_metric_name(metric).split('(')[0].strip()
            descriptions_text += f"• {short_name}: {desc}\n"

        ax_text.text(0.05, 0.95, descriptions_text, transform=ax_text.transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    def plot_shear_rate_performance(self):
        """Create line plot of metrics vs shear rate."""
        ax = self.figure.add_subplot(111)

        # Get per-shear metrics
        per_shear_metrics = self.metrics_calculator.compute_per_shear_rate(
            self.current_results_df,
            ['mae', 'rmse', 'r2']
        )

        shear_rates = per_shear_metrics['shear_rate'].values

        for metric in ['mae', 'rmse', 'r2']:
            if metric in per_shear_metrics.columns:
                ax.plot(shear_rates, per_shear_metrics[metric],
                        'o-', label=self.format_metric_name(metric),
                        linewidth=2, markersize=8)

        ax.set_xscale('log')
        ax.set_xlabel('Shear Rate (s⁻¹)', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Performance Metrics vs Shear Rate',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, which='both')

    def plot_error_vs_shear_rate(self):
        """Create box plot of errors at each shear rate."""
        ax = self.figure.add_subplot(111)
        df = self.current_results_df

        # Group errors by shear rate
        shear_rates = sorted(df['shear_rate'].unique())
        errors_by_shear = [df[df['shear_rate'] == s]
                           ['abs_error'].values for s in shear_rates]

        bp = ax.boxplot(errors_by_shear, labels=[f'{s}' for s in shear_rates],
                        patch_artist=True)

        # Color the boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(shear_rates)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xlabel('Shear Rate (s⁻¹)', fontsize=12)
        ax.set_ylabel('Absolute Error (cP)', fontsize=12)
        ax.set_title('Error Distribution by Shear Rate',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    def plot_confidence_intervals(self):
        """Create plot showing predictions with confidence intervals."""
        ax = self.figure.add_subplot(111)
        df = self.current_results_df.sort_values('actual')

        # Create index for x-axis
        x = np.arange(len(df))

        ax.fill_between(x, df['lower_95'].values, df['upper_95'].values,
                        alpha=0.3, color='blue', label='95% CI')
        ax.plot(x, df['predicted'].values, 'b-',
                label='Predicted', linewidth=2)
        ax.scatter(x, df['actual'].values, c='red', s=20, alpha=0.6,
                   label='Actual', zorder=5)

        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Viscosity (cP)', fontsize=12)
        ax.set_title('Predictions with 95% Confidence Intervals',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add coverage annotation
        coverage = self.metrics_calculator._coverage(df)
        ax.text(0.05, 0.95, f'Coverage: {coverage:.1f}%',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def on_run_selected_for_investigation(self, run_name: str):
        """Handle run selection for per-run investigation."""
        if not run_name or run_name not in self.run_results:
            return

        # Get run data
        run_df = self.run_results[run_name]
        run_formulations = self.formulations_by_run.get(run_name, [])

        # Update run info
        info_text = f"Run: {run_name}\n"
        info_text += f"Formulations: {len(run_formulations)}\n"
        info_text += f"Predictions: {len(run_df)}\n"
        info_text += f"Shear Rates Evaluated: {run_df['shear_rate'].nunique()}\n"
        self.run_info_text.setText(info_text)

        # Update metrics table for this run
        self.display_run_metrics(run_df)

        # Update details table for this run
        self.display_run_details(run_df)

        # Update plot
        self.update_run_plot()

    def display_run_metrics(self, run_df: pd.DataFrame):
        """Display metrics for a specific run."""
        metrics_data = []

        # Compute overall metrics for this run
        overall_metrics = self.metrics_calculator.compute_overall(
            run_df, self.selected_metrics
        )

        for metric, value in overall_metrics.items():
            metrics_data.append({
                'Metric': self.format_metric_name(metric),
                'Value': f"{value:.4f}"
            })

        # Populate table
        df = pd.DataFrame(metrics_data)
        self.run_metrics_table.setRowCount(len(df))
        self.run_metrics_table.setColumnCount(len(df.columns))
        self.run_metrics_table.setHorizontalHeaderLabels(df.columns.tolist())

        for i, row in df.iterrows():
            for j, value in enumerate(row):
                item = QtWidgets.QTableWidgetItem(str(value))
                self.run_metrics_table.setItem(i, j, item)

        self.run_metrics_table.resizeColumnsToContents()

    def display_run_details(self, run_df: pd.DataFrame):
        """Display detailed predictions for a specific run."""
        # Select relevant columns
        display_columns = ['formulation_id', 'shear_rate', 'actual',
                           'predicted', 'abs_error', 'percentage_error']

        df = run_df[display_columns].copy()

        # Set up table
        self.run_details_table.setRowCount(len(df))
        self.run_details_table.setColumnCount(len(df.columns))
        self.run_details_table.setHorizontalHeaderLabels(df.columns.tolist())

        # Populate table
        for i, row in df.iterrows():
            for j, value in enumerate(row):
                if isinstance(value, float):
                    item = QtWidgets.QTableWidgetItem(f"{value:.4f}")
                else:
                    item = QtWidgets.QTableWidgetItem(str(value))
                self.run_details_table.setItem(i, j, item)

        self.run_details_table.resizeColumnsToContents()

    def update_run_plot(self):
        """Update the per-run investigation plot."""
        run_name = self.run_combo.currentText()
        if not run_name or run_name not in self.run_results:
            return

        self.run_figure.clear()
        run_df = self.run_results[run_name]

        plot_type = self.run_plot_type.currentText()

        if plot_type == "Viscosity Profile Comparison":
            self.plot_viscosity_profile_comparison(run_df)
        elif plot_type == "Relative Error by Shear Rate":
            self.plot_relative_error_by_shear(run_df)
        elif plot_type == "Residuals Analysis":
            self.plot_run_residuals_analysis(run_df)
        elif plot_type == "Component Contributions":
            self.plot_component_contributions(run_df)

        self.run_canvas.draw()

    def plot_viscosity_profile_comparison(self, run_df: pd.DataFrame):
        """Plot predicted vs actual viscosity profiles for the selected run."""
        ax = self.run_figure.add_subplot(111)

        # Group by formulation
        for form_id in run_df['formulation_id'].unique():
            form_data = run_df[run_df['formulation_id']
                               == form_id].sort_values('shear_rate')

            # Plot actual and predicted
            ax.plot(form_data['shear_rate'], form_data['actual'],
                    'o-', alpha=0.6, label=f'Actual - {form_id}')
            ax.plot(form_data['shear_rate'], form_data['predicted'],
                    's--', alpha=0.6, label=f'Predicted - {form_id}')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Shear Rate (s⁻¹)', fontsize=12)
        ax.set_ylabel('Viscosity (cP)', fontsize=12)
        ax.set_title('Viscosity Profile: Predicted vs Actual',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

    def plot_relative_error_by_shear(self, run_df: pd.DataFrame):
        """Plot relative error by shear rate for the selected run."""
        ax = self.run_figure.add_subplot(111)

        # Calculate relative error
        run_df['rel_error'] = (run_df['predicted'] -
                               run_df['actual']) / run_df['actual'] * 100

        # Group by shear rate
        shear_rates = sorted(run_df['shear_rate'].unique())

        for shear in shear_rates:
            shear_data = run_df[run_df['shear_rate'] == shear]
            ax.scatter([shear] * len(shear_data), shear_data['rel_error'],
                       alpha=0.6, s=50)

        # Add zero line
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        # Add ±10% bands
        ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=-10, color='gray', linestyle=':', alpha=0.5)

        ax.set_xscale('log')
        ax.set_xlabel('Shear Rate (s⁻¹)', fontsize=12)
        ax.set_ylabel('Relative Error (%)', fontsize=12)
        ax.set_title('Relative Error Distribution by Shear Rate',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def plot_run_residuals_analysis(self, run_df: pd.DataFrame):
        """Plot residuals analysis for the selected run."""
        # Create 2x2 subplot
        gs = self.run_figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Residuals vs Predicted
        ax1 = self.run_figure.add_subplot(gs[0, 0])
        ax1.scatter(run_df['predicted'], run_df['residual'], alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Predicted', fontsize=10)
        ax1.set_ylabel('Residuals', fontsize=10)
        ax1.set_title('Residuals vs Predicted', fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Residuals histogram
        ax2 = self.run_figure.add_subplot(gs[0, 1])
        ax2.hist(run_df['residual'], bins=20, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Residuals', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Residuals Distribution', fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        ax3 = self.run_figure.add_subplot(gs[1, 0])
        stats.probplot(run_df['residual'], dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot', fontsize=11)
        ax3.grid(True, alpha=0.3)

        # Residuals vs Shear Rate
        ax4 = self.run_figure.add_subplot(gs[1, 1])
        for shear in sorted(run_df['shear_rate'].unique()):
            shear_data = run_df[run_df['shear_rate'] == shear]
            ax4.scatter([shear] * len(shear_data), shear_data['residual'],
                        alpha=0.6, label=f'{shear:.0e}')
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax4.set_xscale('log')
        ax4.set_xlabel('Shear Rate', fontsize=10)
        ax4.set_ylabel('Residuals', fontsize=10)
        ax4.set_title('Residuals vs Shear Rate', fontsize=11)
        ax4.grid(True, alpha=0.3)

        self.run_figure.suptitle('Residuals Analysis',
                                 fontsize=14, fontweight='bold')

    def plot_component_contributions(self, run_df: pd.DataFrame):
        """Plot component contributions (placeholder for feature importance)."""
        ax = self.run_figure.add_subplot(111)

        # This is a placeholder - would need actual feature importance from model
        # For now, create a simple bar chart of error by formulation
        form_errors = run_df.groupby('formulation_id')[
            'abs_error'].mean().sort_values()

        y_pos = np.arange(len(form_errors))
        ax.barh(y_pos, form_errors.values, color=plt.cm.RdYlGn_r(
            form_errors.values / form_errors.max()))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(form_errors.index)
        ax.set_xlabel('Mean Absolute Error (cP)', fontsize=12)
        ax.set_ylabel('Formulation ID', fontsize=12)
        ax.set_title('Average Error by Formulation',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    def export_results_as_zip(self):
        """Export all evaluation results and plots as a ZIP file."""
        if self.current_results_df is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Results",
                "No evaluation results to export."
            )
            return

        # Get save location for ZIP file
        zip_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Evaluation Results as ZIP",
            "evaluation_results.zip",
            "ZIP Files (*.zip)"
        )

        if not zip_path:
            return

        try:
            QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
            self.status_label.setText("Exporting results to ZIP...")

            # Create temporary directory for files
            with tempfile.TemporaryDirectory() as temp_dir:

                # Export CSV files
                # 1. Detailed results
                details_file = os.path.join(temp_dir, "detailed_results.csv")
                self.current_results_df.to_csv(details_file, index=False)

                # 2. Overall metrics
                if self.overall_radio.isChecked() or self.both_radio.isChecked():
                    overall_metrics = self.metrics_calculator.compute_overall(
                        self.current_results_df,
                        self.selected_metrics
                    )
                    overall_df = pd.DataFrame(list(overall_metrics.items()),
                                              columns=['Metric', 'Value'])
                    overall_file = os.path.join(
                        temp_dir, "overall_metrics.csv")
                    overall_df.to_csv(overall_file, index=False)

                # 3. Per-shear metrics
                if self.per_shear_radio.isChecked() or self.both_radio.isChecked():
                    per_shear_metrics = self.metrics_calculator.compute_per_shear_rate(
                        self.current_results_df,
                        self.selected_metrics
                    )
                    per_shear_file = os.path.join(
                        temp_dir, "per_shear_metrics.csv")
                    per_shear_metrics.to_csv(per_shear_file, index=False)

                # 4. Per-run results
                for run_name, run_df in self.run_results.items():
                    run_file = os.path.join(
                        temp_dir, f"run_{run_name.replace('.', '_')}_results.csv")
                    run_df.to_csv(run_file, index=False)

                # Export all plot types
                plot_types = [
                    "Predicted vs Actual",
                    "Residuals Distribution",
                    "Q-Q Plot",
                    "Overall Metrics",
                    "Metrics vs Shear Rate",
                    "Error vs Shear Rate",
                    "Confidence Intervals"
                ]

                # Save current plot type to restore later
                current_plot_type = self.plot_type_combo.currentText()

                # Generate and save each plot type
                plots_dir = os.path.join(temp_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)

                for plot_type in plot_types:
                    self.plot_type_combo.setCurrentText(plot_type)
                    QtWidgets.QApplication.processEvents()  # Update the plot

                    plot_file = os.path.join(
                        plots_dir,
                        f"{plot_type.replace(' ', '_').lower()}.png"
                    )
                    self.figure.savefig(plot_file, dpi=300,
                                        bbox_inches='tight')

                # Export per-run plots if we have run results
                if self.run_results:
                    run_plots_dir = os.path.join(temp_dir, "per_run_plots")
                    os.makedirs(run_plots_dir, exist_ok=True)

                    current_run = self.run_combo.currentText()
                    current_run_plot_type = self.run_plot_type.currentText()

                    run_plot_types = [
                        "Viscosity Profile Comparison",
                        "Relative Error by Shear Rate",
                        "Residuals Analysis",
                        "Component Contributions"
                    ]

                    for run_name in self.run_results.keys():
                        self.run_combo.setCurrentText(run_name)
                        QtWidgets.QApplication.processEvents()

                        for plot_type in run_plot_types:
                            self.run_plot_type.setCurrentText(plot_type)
                            QtWidgets.QApplication.processEvents()

                            plot_file = os.path.join(
                                run_plots_dir,
                                f"{run_name.replace('.', '_')}_{plot_type.replace(' ', '_').lower()}.png"
                            )
                            self.run_figure.savefig(
                                plot_file, dpi=300, bbox_inches='tight')

                    # Restore selections
                    self.run_combo.setCurrentText(current_run)
                    self.run_plot_type.setCurrentText(current_run_plot_type)

                # Restore original plot type
                self.plot_type_combo.setCurrentText(current_plot_type)

                # Create summary report
                summary_file = os.path.join(temp_dir, "evaluation_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write("=" * 60 + "\n")
                    f.write("VISQAI MODEL EVALUATION REPORT\n")
                    f.write("=" * 60 + "\n\n")

                    f.write(f"Model: {self.select_model_label.text()}\n")
                    f.write(
                        f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(
                        f"Total Formulations: {len(self.selected_formulations)}\n")
                    f.write(
                        f"Total Predictions: {len(self.current_results_df)}\n")
                    f.write(
                        f"Runs Evaluated: {len(self.formulations_by_run)}\n\n")

                    f.write("OVERALL METRICS:\n")
                    f.write("-" * 40 + "\n")
                    overall_metrics = self.metrics_calculator.compute_overall(
                        self.current_results_df,
                        self.selected_metrics
                    )
                    for metric, value in overall_metrics.items():
                        f.write(
                            f"{self.format_metric_name(metric)}: {value:.4f}\n")
                        f.write(
                            f"  {self.METRIC_DESCRIPTIONS.get(metric, '')}\n\n")

                    f.write("\nPER-RUN SUMMARY:\n")
                    f.write("-" * 40 + "\n")
                    for run_name, run_df in self.run_results.items():
                        f.write(f"\n{run_name}:\n")
                        run_metrics = self.metrics_calculator.compute_overall(
                            run_df, ['mae', 'rmse', 'r2']
                        )
                        for metric, value in run_metrics.items():
                            f.write(
                                f"  {self.format_metric_name(metric)}: {value:.4f}\n")

                # Create ZIP file
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, temp_dir)
                            zipf.write(file_path, arcname)

            QtWidgets.QMessageBox.information(
                self,
                "Export Complete",
                f"Results exported successfully to:\n{zip_path}"
            )

            self.status_label.setText(
                f"Results exported to {os.path.basename(zip_path)}")

        except Exception as e:
            Log.e("EvaluationUI", f"ZIP export failed: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export results: {str(e)}"
            )

        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
