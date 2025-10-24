"""
evaluation_ui.py

This module provides the EvaluationUI class for the Evaluate tab in VisQAI.
It supports model performance evaluation against existing data on both overall
and per-shear-rate basis with customizable metrics and visualizations.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-10-24

Version:
   1.3
"""

import os
import copy
import zipfile
import traceback
from typing import Optional, List, Dict, TYPE_CHECKING
import json
import tempfile
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
    from src.io.parser import Parser
    if TYPE_CHECKING:
        from src.view.main_window import VisQAIWindow
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.formulation import Formulation
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.utils.metrics import Metrics
    from QATCH.VisQAI.src.io.parser import Parser
    if TYPE_CHECKING:
        from QATCH.VisQAI.src.view.main_window import VisQAIWindow

TAG = "[EvaluationUI]"


class EvaluationUI(QtWidgets.QDialog):
    METRIC_DESCRIPTIONS = {
        'mae': 'Average absolute difference between predicted and actual values',
        'rmse': 'Square root of average squared differences',
        'mape': 'Average percentage error',
        'r2': 'Coefficient of determination (1.0 = perfect fit)',
        'coverage': '% of actual values within confidence intervals',
        'max_error': 'Largest absolute error in predictions',
        'median_ae': 'Middle value of absolute errors',
        'explained_variance': 'Proportion of variance explained by model'
    }

    def __init__(self, parent=None):
        """Initialize the EvaluationUI.

        Args:
            parent: Parent VisQAIWindow instance.
        """
        super().__init__(parent)
        self.parent: 'VisQAIWindow' = parent
        self.setWindowTitle("Model Evaluation")
        self.model_path: Optional[str] = None
        self.predictor: Optional[Predictor] = None
        self.metrics = Metrics()
        self.current_results_df: Optional[pd.DataFrame] = None
        self.selected_formulations: List[Formulation] = []
        self.available_metrics = self.metrics.get_available_metrics()

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
        self.shear_rates = {
            'Viscosity_100': 100,
            'Viscosity_1000': 1000,
            'Viscosity_10000': 10000,
            'Viscosity_100000': 100000,
            'Viscosity_15000000': 15000000
        }
        self.selected_shear_rates = list(self.shear_rates.keys())
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
        per_run_widget = self.create_per_run_evaluation_tab()
        self.tab_widget.addTab(per_run_widget, "Per-Run Evaluation")
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

        # Top plot controls
        plot_controls = QtWidgets.QHBoxLayout()
        plot_type_label = QtWidgets.QLabel("Plot Type:")
        self.plot_type_combo = QtWidgets.QComboBox()
        self.plot_type_combo.addItems([
            "Predicted vs Actual",
        ])
        self.plot_type_combo.currentTextChanged.connect(self.update_plot)

        plot_controls.addWidget(plot_type_label)
        plot_controls.addWidget(self.plot_type_combo)
        plot_controls.addStretch()

        # Save figure button (top right)
        self.save_figure_btn = QtWidgets.QPushButton("Save Figure")
        self.save_figure_btn.clicked.connect(self.save_current_figure_overall)
        self.save_figure_btn.setEnabled(False)
        plot_controls.addWidget(self.save_figure_btn)

        plot_layout.addLayout(plot_controls)

        # Plot canvas
        self.current_overall_figure = Figure(figsize=(8, 6))
        self.overall_canvas = FigureCanvas(self.current_overall_figure)
        plot_layout.addWidget(self.overall_canvas)

        # Shear rate navigation controls (below plot)
        self.current_shear_index = 0  # 0 = all shear rates, 1+ = individual shear rates

        shear_nav_layout = QtWidgets.QHBoxLayout()
        shear_nav_layout.addStretch()

        self.prev_shear_btn = QtWidgets.QPushButton("◀ Previous")
        self.prev_shear_btn.clicked.connect(self.navigate_shear_rate_prev)
        self.prev_shear_btn.setEnabled(False)

        self.shear_rate_label = QtWidgets.QLabel("All Shear Rates")
        self.shear_rate_label.setAlignment(Qt.AlignCenter)
        self.shear_rate_label.setMinimumWidth(200)

        self.next_shear_btn = QtWidgets.QPushButton("Next ▶")
        self.next_shear_btn.clicked.connect(self.navigate_shear_rate_next)
        self.next_shear_btn.setEnabled(False)

        shear_nav_layout.addWidget(self.prev_shear_btn)
        shear_nav_layout.addWidget(self.shear_rate_label)
        shear_nav_layout.addWidget(self.next_shear_btn)
        shear_nav_layout.addStretch()

        plot_layout.addLayout(shear_nav_layout)

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

    def navigate_shear_rate_prev(self):
        """Navigate to previous shear rate view."""
        if self.current_shear_index > 0:
            self.current_shear_index -= 1
            self.update_shear_navigation()
            self.update_plot()

    def navigate_shear_rate_next(self):
        """Navigate to next shear rate view."""
        max_index = len(self.selected_shear_rates)
        if self.current_shear_index < max_index:
            self.current_shear_index += 1
            self.update_shear_navigation()
            self.update_plot()

    def update_shear_navigation(self):
        """Update navigation button states and label based on current index."""
        if not self.selected_shear_rates:
            self.prev_shear_btn.setEnabled(False)
            self.next_shear_btn.setEnabled(False)
            self.shear_rate_label.setText("No Shear Rates Selected")
            return

        # Update button states
        self.prev_shear_btn.setEnabled(self.current_shear_index > 0)
        self.next_shear_btn.setEnabled(
            self.current_shear_index < len(self.selected_shear_rates)
        )

        # Update label
        if self.current_shear_index == 0:
            self.shear_rate_label.setText("All Shear Rates")
        else:
            shear_name = list(self.selected_shear_rates)[
                self.current_shear_index - 1]
            shear_value = self.shear_rates[shear_name]
            self.shear_rate_label.setText(f"{shear_name} ({shear_value} s⁻¹)")

    def get_current_shear_filter(self):
        """Get the current shear rate(s) to plot based on navigation index.

        Returns:
            None if all shear rates should be plotted (index 0)
            str of specific shear rate name if individual view (index > 0)
        """
        if self.current_shear_index == 0:
            return None  # Plot all shear rates
        else:
            # Return specific shear rate name
            return list(self.selected_shear_rates)[self.current_shear_index - 1]

    def create_per_run_evaluation_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # === LEFT PANEL - Run Selection and Info ===
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Run selection group
        run_group = QtWidgets.QGroupBox("Run Selection")
        run_layout = QtWidgets.QVBoxLayout(run_group)
        run_layout.setSpacing(8)

        # Run combo with label
        run_label = QtWidgets.QLabel("Select Run:")
        self.run_combo = QtWidgets.QComboBox()
        self.run_combo.setMinimumWidth(200)
        self.run_combo.setToolTip("Select a run to investigate its details")
        self.run_combo.currentTextChanged.connect(
            self.on_run_selected_for_investigation
        )
        run_layout.addWidget(run_label)
        run_layout.addWidget(self.run_combo)

        left_layout.addWidget(run_group)

        # Formulation information group
        formulation_group = QtWidgets.QGroupBox("Formulation Information")
        formulation_layout = QtWidgets.QVBoxLayout(formulation_group)

        self.run_info_table = QtWidgets.QTableWidget()
        self.run_info_table.setAlternatingRowColors(True)
        self.run_info_table.horizontalHeader().setStretchLastSection(True)
        self.run_info_table.verticalHeader().setVisible(False)
        self.run_info_table.setSelectionMode(
            QtWidgets.QAbstractItemView.NoSelection
        )
        formulation_layout.addWidget(self.run_info_table)

        left_layout.addWidget(formulation_group)

        # Prediction information group
        prediction_group = QtWidgets.QGroupBox("Prediction Information")
        prediction_layout = QtWidgets.QVBoxLayout(prediction_group)

        self.run_metrics_table = QtWidgets.QTableWidget()
        self.run_metrics_table.setAlternatingRowColors(True)
        self.run_metrics_table.horizontalHeader().setStretchLastSection(True)
        self.run_metrics_table.verticalHeader().setVisible(False)
        self.run_metrics_table.setSelectionMode(
            QtWidgets.QAbstractItemView.NoSelection
        )
        prediction_layout.addWidget(self.run_metrics_table)

        left_layout.addWidget(prediction_group)
        left_layout.addStretch()

        # === RIGHT PANEL - Visualization ===
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Visualization controls group
        viz_group = QtWidgets.QGroupBox("Visualization Controls")
        viz_layout = QtWidgets.QHBoxLayout(viz_group)
        viz_layout.setSpacing(10)

        viz_label = QtWidgets.QLabel("Plot Type:")
        viz_layout.addWidget(viz_label)

        self.run_plot_type = QtWidgets.QComboBox()
        self.run_plot_type.addItems([
            "Viscosity Profile Comparison",
        ])
        self.run_plot_type.setMinimumWidth(200)
        self.run_plot_type.setToolTip("Select visualization type")
        self.run_plot_type.currentTextChanged.connect(self.update_run_plot)
        viz_layout.addWidget(self.run_plot_type)

        viz_layout.addStretch()

        self.save_plot_button = QtWidgets.QPushButton("Save Figure")
        self.save_plot_button.setToolTip("Save the current plot to file")
        self.save_plot_button.clicked.connect(self.save_current_figure_per_run)
        viz_layout.addWidget(self.save_plot_button)

        right_layout.addWidget(viz_group)

        # Plot canvas
        self.run_figure = Figure(figsize=(10, 7))
        self.run_canvas = FigureCanvas(self.run_figure)
        self.run_canvas.setMinimumHeight(400)
        right_layout.addWidget(self.run_canvas)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)  # Left panel
        splitter.setStretchFactor(1, 2)  # Right panel gets more space
        splitter.setSizes([300, 600])  # Initial sizes

        layout.addWidget(splitter)

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
        if self.model_dialog.exec_():

            model_path = os.path.join(Architecture.get_path(),
                                      "QATCH/VisQAI/assets")
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
                    # Extract filename without path and extension (matching frame_step1.py line 2018-2019)
                    display_name = file_path.split(
                        '\\')[-1].split('/')[-1].split('.')[0]
                    self.select_model_label.setText(display_name)
                    Log.i(TAG, f"Model loaded: {file_path}")
                    self.check_ready_to_evaluate()
                except Exception as e:
                    Log.e(TAG, f"Failed to load model: {e}")
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Model Loading Error",
                        f"Failed to load model: {str(e)}"
                    )
                    self.model_path = None
                    self.predictor = None
                    self.select_model_label.setText("Failed to load model")

    def user_run_browse(self):
        selected_files = self.file_dialog.selectedFiles()
        inside = True
        if selected_files:
            prefer_abs = os.path.abspath(Constants.log_prefer_path)
            path_abs = os.path.abspath(selected_files[0])
            try:
                inside = os.path.commonpath(
                    [prefer_abs, path_abs]) == prefer_abs
            except ValueError:
                inside = False

        if not selected_files or not inside:
            self.file_dialog.setDirectory(Constants.log_prefer_path)
        else:
            set_directory, select_file = os.path.split(
                os.path.dirname(path_abs))
            self.file_dialog.setDirectory(set_directory)
            self.file_dialog.selectFile(select_file)

        if self.file_dialog.exec_():
            selected_files = self.file_dialog.selectedFiles()
            if selected_files:
                for file_path in selected_files:
                    self.add_run_file(file_path)

    def add_run_file(self, file_path: str):
        try:
            parser = Parser(file_path)
            formulation = parser.get_formulation()
            run_name = parser.get_run_name()
            if not formulation:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Formulation",
                    "No valid formulation found in the selected file."
                )
                return
            self.formulations_by_run[run_name] = [formulation]
            self.selected_formulations.append(formulation)
            if self.model.rowCount() == 1 and not self.model.item(0).isEnabled():
                self.model.clear()
            run_item = QtGui.QStandardItem(run_name)
            run_item.setData(file_path, Qt.UserRole)
            self.model.appendRow(run_item)
            self.run_combo.addItem(run_name)
            self.update_formulation_info()
            self.check_ready_to_evaluate()

        except Exception as e:
            Log.e(TAG, f"Failed to load run: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Run Loading Error",
                f"Failed to load run file: {str(e)}"
            )

    def update_formulation_info(self):
        if not self.selected_formulations:
            Log.i(TAG, "No formulations loaded")
            return
        info_text = f"Total Formulations: {len(self.selected_formulations)}\n"
        info_text += f"Runs Loaded: {len(self.formulations_by_run)}\n"
        Log.i(TAG, info_text)

    def check_ready_to_evaluate(self):
        """Check if ready to evaluate and update UI accordingly."""
        ready = (self.predictor is not None and
                 len(self.selected_formulations) > 0 and
                 len(self.selected_metrics) > 0)
        self.evaluate_btn.setEnabled(ready)

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
        """Handle shear rate selection changes."""
        selected_items = self.shear_rate_list.selectedItems()
        self.selected_shear_rates = [
            item.data(Qt.UserRole) for item in selected_items
        ]

        # Reset navigation to show all shear rates
        self.current_shear_index = 0
        self.update_shear_navigation()

        # Update plot if results exist
        if hasattr(self, 'evaluation_results') and self.evaluation_results is not None:
            self.update_plot()

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
            self.run_results.clear()
            all_results_list = []
            for run_name, formulations in self.formulations_by_run.items():
                Log.i(
                    f"Evaluating run: {run_name} with {len(formulations)} formulations")

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
                eval_data = pd.concat(eval_dfs, ignore_index=True)
                target_cols = [f"Viscosity_{rate}" for rate in
                               [self.shear_rates[name] for name in self.selected_shear_rates]]
                Log.i(
                    f"Evaluating {len(eval_data)} samples with targets: {target_cols}")
                results_df = self.predictor.evaluate(
                    eval_data=eval_data,
                    targets=target_cols,
                    n_samples=None
                )
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
            else:
                raise ValueError("No valid evaluation results generated")

        except Exception as e:
            Log.e(TAG, f"Evaluation failed: {e}")
            Log.e(TAG, traceback.format_exc())
            QtWidgets.QMessageBox.critical(
                self,
                "Evaluation Error",
                f"Failed to evaluate model: {str(e)}"
            )

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
            overall_metrics = self.metrics.compute_overall(
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
            per_shear_metrics = self.metrics.compute_per_shear_rate(
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
        """Update the plot based on current settings with enhanced visualization."""
        if not hasattr(self, 'evaluation_results') or self.evaluation_results is None:
            return

        # Enable save button when plot is available
        self.save_figure_btn.setEnabled(True)

        self.current_overall_figure.clear()
        ax = self.current_overall_figure.add_subplot(111)

        # Get current shear filter
        shear_filter = self.get_current_shear_filter()

        plot_type = self.plot_type_combo.currentText()

        if plot_type == "Predicted vs Actual":
            # Filter data based on current shear rate selection
            if shear_filter is None:
                # Plot all selected shear rates
                plot_data = self.evaluation_results
                title_suffix = "(All Shear Rates)"
            else:
                # Plot only specific shear rate
                plot_data = self.evaluation_results[
                    self.evaluation_results['shear_rate'] == shear_filter
                ]
                shear_value = self.shear_rates[shear_filter]
                title_suffix = f"({shear_filter}: {shear_value} s⁻¹)"

            # Create enhanced scatter plot with color-coded errors
            scatter = ax.scatter(
                plot_data['actual'],
                plot_data['predicted'],
                c=plot_data['percentage_error'],
                cmap='coolwarm',
                s=50,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )

            # Add perfect prediction line
            min_val = min(plot_data['actual'].min(),
                          plot_data['predicted'].min())
            max_val = max(plot_data['actual'].max(),
                          plot_data['predicted'].max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                'k--',
                alpha=0.5,
                label='Perfect Prediction'
            )

            # Add ±10% error bands
            ax.fill_between(
                [min_val, max_val],
                [min_val*0.9, max_val*0.9],
                [min_val*1.1, max_val*1.1],
                alpha=0.2,
                color='gray',
                label='±10% Error'
            )

            # Labels and title
            ax.set_xlabel('Actual Viscosity (cP)', fontsize=12)
            ax.set_ylabel('Predicted Viscosity (cP)', fontsize=12)
            ax.set_title(
                f'Predicted vs Actual Viscosity {title_suffix}',
                fontsize=14,
                fontweight='bold'
            )
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            # Add colorbar for percentage error
            cbar = self.current_overall_figure.colorbar(scatter, ax=ax)
            cbar.set_label('Percentage Error (%)', rotation=270, labelpad=20)

            # Calculate and display R² score
            from sklearn.metrics import r2_score
            r2 = r2_score(plot_data['actual'], plot_data['predicted'])
            ax.text(
                0.05, 0.95,
                f'R² = {r2:.4f}',
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            )

        self.overall_canvas.draw()

    def plot_predicted_vs_actual(self):
        """Create predicted vs actual scatter plot."""
        ax = self.current_overall_figure.add_subplot(111)
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
        cbar = self.current_overall_figure.colorbar(scatter, ax=ax)
        cbar.set_label('Percentage Error (%)', rotation=270, labelpad=20)

        # Add R² annotation
        from sklearn.metrics import r2_score
        r2 = r2_score(df['actual'], df['predicted'])
        ax.text(0.05, 0.95, f'R² = {r2:.4f}',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def plot_confidence_intervals(self):
        """Create plot showing predictions with confidence intervals."""
        ax = self.current_overall_figure.add_subplot(111)
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
        coverage = self.metrics._coverage(df)
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

        self.display_formulation_info(run_formulations[0])
        self.display_prediction_info(run_df)
        self.update_run_plot()

    def display_formulation_info(self, formulation: Formulation) -> None:
        formulation_df = formulation.to_dataframe(
            encoded=False, training=False)

        # Rename columns for display
        name_mapping = {
            "Protein_type": "Protein Type",
            "MW": "Molecular Weight",
            "PI_mean": "Protein pI Mean",
            "PI_range": "Protein pI Range",
            "Protein_class_type": "Protein Class",
            "Protein_conc": "Protein Concentration",
            "Buffer_type": "Buffer Type",
            "Buffer_pH": "Buffer pH",
            "Salt_type": "Salt Type",
            "Salt_conc": "Salt Concentration",
            "Stabilizer_type": "Stabilizer Type",
            "Stabilizer_conc": "Stabilizer Concentration",
            "Surfactant_type": "Surfactant Type",
            "Surfactant_conc": "Surfactant Concentration",
            "Excipient_type": "Excipient Type",
            "Excipient_conc": "Excipient Concentration",
        }
        units_map = {
            "Protein_conc": formulation.protein.units,
            "MW": "kDa",
            "Buffer_pH": "",
            "Salt_conc": formulation.salt.units,
            "Stabilzer_conc": formulation.stabilizer.units,
            "Surfactant_conc": formulation.surfactant.units,
            "Excipient_conc": formulation.excipient.units,
            "Temperature": "\u00b0C"
        }

        formulation_df = formulation_df.drop(columns=["ID"], errors="ignore")
        formulation_df = formulation_df.reset_index(drop=True)

        if formulation_df.empty:
            self.run_info_table.clear()
            self.run_info_table.setRowCount(0)
            self.run_info_table.setColumnCount(0)
            return

        row_data = formulation_df.iloc[0]

        self.run_info_table.setRowCount(len(row_data))
        self.run_info_table.setColumnCount(3)
        self.run_info_table.setHorizontalHeaderLabels(
            ["Feature", "Value", "Units"])

        for i, (key, value) in enumerate(row_data.items()):
            display_name = name_mapping.get(key, key)
            unit = units_map.get(key, "")
            if not unit and "conc" in key.lower():
                if "protein" in key.lower():
                    unit = formulation.protein.units
                elif "salt" in key.lower():
                    unit = formulation.salt.units
                elif "stabil" in key.lower():
                    unit = formulation.stabilizer.units
                elif "surf" in key.lower():
                    unit = formulation.surfactant.units
                elif "excipient" in key.lower():
                    unit = formulation.excipient.units
            if key.lower() == "Temperature":
                unit = units_map["Temperature"]
            # Create table items (read-only)
            name_item = QtWidgets.QTableWidgetItem(str(display_name))
            value_item = QtWidgets.QTableWidgetItem(str(value))
            unit_item = QtWidgets.QTableWidgetItem(str(unit))
            for item in (name_item, value_item, unit_item):
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)

            self.run_info_table.setItem(i, 0, name_item)
            self.run_info_table.setItem(i, 1, value_item)
            self.run_info_table.setItem(i, 2, unit_item)

        # Table aesthetics
        self.run_info_table.resizeColumnsToContents()
        self.run_info_table.verticalHeader().setVisible(False)   # hide index column
        self.run_info_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.run_info_table.setSelectionMode(
            QtWidgets.QAbstractItemView.NoSelection)
        self.run_info_table.setFocusPolicy(QtCore.Qt.NoFocus)

    def display_prediction_info(self, run_df: pd.DataFrame) -> None:
        display_columns = ["shear_rate", "actual",
                           "predicted", "abs_error", "percentage_error"]
        df = run_df[display_columns].copy()
        name_mapping = {
            "shear_rate": "Shear Rate (1/s)",
            "actual": "Actual Viscosity (cP)",
            "predicted": "Estimated Viscosity (cP)",
            "abs_error": "Absolute Error",
            "percentage_error": "Percentage Error (%)",
        }
        df.rename(columns=name_mapping, inplace=True)
        self.run_metrics_table.setRowCount(len(df))
        self.run_metrics_table.setColumnCount(len(df.columns))
        self.run_metrics_table.setHorizontalHeaderLabels(df.columns.tolist())
        for i, row in df.iterrows():
            for j, value in enumerate(row):
                if isinstance(value, (float, int)):
                    display_value = f"{value:.2f}"
                else:
                    display_value = str(value)

                item = QtWidgets.QTableWidgetItem(display_value)

                # Make item read-only and right-align numeric columns
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                if isinstance(value, (float, int)):
                    item.setTextAlignment(
                        QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

                self.run_metrics_table.setItem(i, j, item)
        self.run_metrics_table.resizeColumnsToContents()
        self.run_metrics_table.verticalHeader().setVisible(False)
        self.run_metrics_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.run_metrics_table.setSelectionMode(
            QtWidgets.QAbstractItemView.NoSelection)
        self.run_metrics_table.setFocusPolicy(QtCore.Qt.NoFocus)

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

        self.run_canvas.draw()

    def plot_viscosity_profile_comparison(self, run_df: pd.DataFrame):
        ax = self.run_figure.add_subplot(111)
        ax.clear()

        color_actual = "#00A3DA"
        color_predicted = "#32E2DF"
        color_ci = "#69EAC5"
        if {"lower_95", "upper_95"}.issubset(run_df.columns):
            ax.fill_between(
                run_df["shear_rate"],
                run_df["lower_95"],
                run_df["upper_95"],
                color=color_ci,
                alpha=0.15,
                label="95% CI",
                linewidth=0
            )
        ax.plot(
            run_df["shear_rate"],
            run_df["actual"],
            "-",
            color=color_actual,
            alpha=0.85,
            label="Actual",
            linewidth=2.5
        )

        ax.plot(
            run_df["shear_rate"],
            run_df["predicted"],
            "--",
            color=color_predicted,
            alpha=0.85,
            label="Estimated",
            linewidth=2.5,
            dashes=(5, 3)
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Shear Rate (s⁻¹)", fontsize=11, color="#4C566A")
        ax.set_ylabel("Viscosity (cP)", fontsize=11, color="#4C566A")
        ax.set_title(
            "Viscosity Profile: Estimated vs Actual",
            fontsize=13,
            fontweight=600,
            color="#2E3440",
            pad=15
        )
        ax.legend(
            loc="best",
            fontsize=9,
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

        self.run_figure.tight_layout()
        self.run_canvas.draw()

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
            # Create temporary directory for files
            with tempfile.TemporaryDirectory() as temp_dir:

                # Export CSV files
                # 1. Detailed results
                details_file = os.path.join(temp_dir, "detailed_results.csv")
                self.current_results_df.to_csv(details_file, index=False)

                # 2. Overall metrics
                if self.overall_radio.isChecked() or self.both_radio.isChecked():
                    overall_metrics = self.metrics.compute_overall(
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
                    per_shear_metrics = self.metrics.compute_per_shear_rate(
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
                    self.current_overall_figure.savefig(plot_file, dpi=300,
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
                    overall_metrics = self.metrics.compute_overall(
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
                        run_metrics = self.metrics.compute_overall(
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

        except Exception as e:
            Log.e(TAG, f"ZIP export failed: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export results: {str(e)}"
            )

        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def save_current_figure_per_run(self):
        """Save the current plot to a file."""
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
        )

        if file_path:
            try:
                self.run_figure.savefig(
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

    def save_current_figure_overall(self):
        """Save the current figure to a file."""
        if not hasattr(self, 'evaluation_results') or self.evaluation_results is None:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "No plot data available to save."
            )
            return

        # Determine default filename based on current view
        if self.current_shear_index == 0:
            default_name = "plot_all_shear_rates.png"
        else:
            shear_name = list(self.selected_shear_rates)[
                self.current_shear_index - 1]
            default_name = f"plot_{shear_name.replace(' ', '_').lower()}.png"

        # Open save dialog
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Figure",
            default_name,
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Image (*.svg);;All Files (*)"
        )

        if file_path:
            try:
                self.current_overall_figure.savefig(
                    file_path, dpi=300, bbox_inches='tight')
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Figure saved to:\n{file_path}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to save figure:\n{str(e)}"
                )
