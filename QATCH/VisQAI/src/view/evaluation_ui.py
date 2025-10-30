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
import re
import os
import zipfile
import traceback
from typing import Optional, List, Dict, TYPE_CHECKING
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
from matplotlib.colors import LinearSegmentedColormap

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
        """Initialize the EvaluationUI window for model evaluation and visualization.

        This class manages the evaluation of trained models, including
        loading formulations, computing metrics, and displaying results.
        It integrates with the main VisQAIWindow interface and provides
        tools for inspecting per-run results, metrics, and viscosity profiles.

        Args:
            parent (VisQAIWindow, optional): The parent window instance of the EvaluationUI.
                Defaults to None.

        Attributes:
            parent (VisQAIWindow): Reference to the parent application window.
            model_path (Optional[str]): Path to the currently loaded model file.
            predictor (Optional[Predictor]): Active predictor instance used for generating predictions.
            metrics (Metrics): Metrics utility for computing evaluation statistics.
            current_results_df (Optional[pd.DataFrame]): DataFrame containing the latest evaluation results.
            selected_formulations (List[Formulation]): List of currently selected formulations for evaluation.
            available_metrics (List[str]): List of metrics available for computation.
            formulations_by_run (Dict[str, List[Formulation]]): Mapping of run names to their associated formulations.
            run_results (Dict[str, pd.DataFrame]): Mapping of run names to their corresponding results DataFrames.
            all_files (dict): Dictionary tracking relevant files for each run.
            run_file_run (Optional[Path]): Path to the selected run file.
            run_file_xml (Optional[Path]): Path to the associated XML metadata file.
            run_file_analyze (Optional[Path]): Path to the `analyze_out.csv` file within the run data.
            selected_metrics (List[str]): Default metrics selected for evaluation (e.g., MAE, RMSE).
            shear_rates (Dict[str, int]): Mapping of viscosity labels to corresponding shear rate values.
            selected_shear_rates (List[str]): Currently selected viscosity/shear rate labels for display.
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
        self.formulations_by_run: Dict[str, List[Formulation]] = {}
        self.run_results: Dict[str, pd.DataFrame] = {}
        self.all_files = {}
        self.run_file_run = None
        self.run_file_xml = None
        self.run_file_analyze = None
        self.selected_metrics = []
        self.shear_rates = {
            'Viscosity_100': 100,
            'Viscosity_1000': 1000,
            'Viscosity_10000': 10000,
            'Viscosity_100000': 100000,
            'Viscosity_15000000': 15000000
        }
        self.selected_shear_rates = []
        self._init_file_dialogs()
        self.init_ui()

    def init_ui(self) -> None:
        """Initialize and configure the main user interface for the EvaluationUI window.

        This method constructs the overall layout and tabbed interface for model evaluation.
        It sets up two primary tabs:
        1. Overall Evaluation - for viewing aggregate metrics and model performance summaries.
        2. Per-Run Evaluation - for inspecting individual run-level results and details.

        The method also defines layout structure, window size, and event connections for tab changes.
        """
        main_layout = QtWidgets.QVBoxLayout(self)
        self.tab_widget = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tab_widget)
        overall_widget = self.create_overall_evaluation_tab()
        self.tab_widget.addTab(overall_widget, "Overall Evaluation")
        per_run_widget = self.create_per_run_evaluation_tab()
        self.tab_widget.addTab(per_run_widget, "Per-Run Evaluation")
        self.resize(1400, 800)
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index: int):
        """Handle tab switching events in the EvaluationUI.

        When the user switches tabs, this method performs appropriate actions depending
        on the selected tab. If the Per-Run Evaluation tab (index 1) is selected, it
        automatically loads data for the currently selected run if available. If no runs
        have been evaluated yet, it displays an informational popup guiding the user
        through the necessary steps to perform an evaluation.

        Args:
            index (int): The index of the newly selected tab in the tab widget.
        """
        if index == 1:
            if self.run_combo.count() > 0:
                current_run = self.run_combo.currentText()
                self.on_run_selected_for_investigation(current_run)

            else:
                # No runs available - show popup message
                QtWidgets.QMessageBox.information(
                    self,
                    "No Runs Available",
                    "Please perform an evaluation from the 'Overall Evaluation' tab first.\n\n"
                    "Steps:\n"
                    "1. Switch to 'Overall Evaluation' tab\n"
                    "2. Select a model\n"
                    "3. Import run files\n"
                    "4. Click 'Run Evaluation'"
                )
                Log.w(TAG, "Per-Run tab accessed with no runs available")

    def _init_file_dialogs(self) -> None:
        """Initialize file dialogs for model and run file selection.

        This method configures two `QFileDialog` instances used within the EvaluationUI:
        one for selecting trained VisQ.AI model archives and another for selecting
        captured run data files. It sets dialog options, default directories, and file
        filters to streamline user file selection.

        Attributes:
            model_dialog (QtWidgets.QFileDialog): Dialog for selecting VisQ.AI model ZIP files.
            file_dialog (QtWidgets.QFileDialog): Dialog for selecting captured run ZIP files.
        """
        self.model_dialog = QtWidgets.QFileDialog()
        self.model_dialog.setOption(
            QtWidgets.QFileDialog.DontUseNativeDialog, True)
        model_path = os.path.join(Architecture.get_path(),
                                  "QATCH/VisQAI/assets")
        if os.path.exists(model_path):
            self.model_dialog.setDirectory(model_path)
        else:
            self.model_dialog.setDirectory(Constants.log_prefer_path)
        self.model_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self.model_dialog.setNameFilter("VisQ.AI Models (VisQAI-*.zip)")
        self.model_dialog.selectNameFilter("VisQ.AI Models (VisQAI-*.zip)")

        self.file_dialog = QtWidgets.QFileDialog()
        self.file_dialog.setOption(
            QtWidgets.QFileDialog.DontUseNativeDialog, True)
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self.file_dialog.setNameFilter("Captured Runs (capture.zip)")
        self.file_dialog.selectNameFilter("Captured Runs (capture.zip)")

    def create_overall_evaluation_tab(self) -> None:
        """Create and configure the 'Overall Evaluation' tab in the EvaluationUI.

        This method constructs the primary user interface for evaluating trained
        models across multiple runs. It provides controls for model selection,
        experiment import, metric and shear-rate configuration, and visualization
        of results through tables and plots.

        The layout consists of a split view:
        - Left panel: Configuration controls for model loading, run selection,
            metric/shear rate selection, evaluation settings, and export options.
        - Right panel: Displays evaluation results including metrics tables,
            plots, and detailed result breakdowns.

        Returns:
            QtWidgets.QWidget: The fully configured QWidget representing the
            "Overall Evaluation" tab.

        Layout Overview:
            - Select Model Group
            - Button: Browse and select a trained VisQ.AI model ZIP file.
            - LineEdit: Displays the selected model path.
            - Import Experiments Group
            - Button: Add run capture files (`capture.zip`).
            - ListView: Displays added runs with placeholder text when empty.
            - Buttons: Remove selected runs or clear all.
            - Evaluation Settings Group
            - Multi-select lists for:
                - Metrics (MAE, RMSE, MAPE, R², etc.)
                - Shear rates (e.g., 100, 1000, 10000 s⁻¹)
            - Radio buttons for evaluation type (Overall, Per-Shear, Both).
            - Action Controls
            - Buttons for running model evaluation and exporting results as ZIP.
            - Results Display (Right Panel)
            - Tabs for:
                - Metrics table
                - Plot visualization (Predicted vs Actual, metrics)
                - Detailed result table
            - Plot controls for choosing visualization type (Bar, Line, Box, etc.)
            - Buttons for figure navigation and saving.

        Signals:
            - `self.select_model_btn.clicked` -> Opens model file dialog.
            - `self.select_run.clicked` -> Opens run file selection dialog.
            - `self.evaluate_btn.clicked` -> Triggers model evaluation.
            - `self.export_btn.clicked` -> Exports evaluation results.
            - `self.metrics_list.itemSelectionChanged` -> Updates selected metrics.
            - `self.shear_rate_list.itemSelectionChanged` -> Updates selected shear rates.
            - `self.plot_type_combo.currentTextChanged` -> Updates available plot types.
            - `self.viz_type_combo.currentTextChanged` -> Updates visualization display.
            - `self.save_figure_btn.clicked` -> Saves the current figure.
            - `self.prev_shear_btn` / `self.next_shear_btn` -> Navigate through shear rates.

        """
        widget = QtWidgets.QWidget()
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        layout = QtWidgets.QVBoxLayout(widget)
        layout.addWidget(splitter)
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
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
        data_group = QtWidgets.QGroupBox("Import Experiments")
        data_layout = QtWidgets.QVBoxLayout(data_group)
        form_layout = QtWidgets.QFormLayout()
        self.select_run = QtWidgets.QPushButton("Add Run(s)...")
        self.select_label = QtWidgets.QLineEdit()
        self.select_label.setPlaceholderText("No run selected")
        self.select_label.setReadOnly(True)
        self.list_view = QtWidgets.QListView()
        self.list_view.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.model = QtGui.QStandardItemModel()
        self.list_view_addPlaceholderText()
        self.list_view.setModel(self.model)
        self.list_view.clicked.connect(self.user_run_clicked)
        form_layout.addRow(self.select_run, self.list_view)
        data_layout.addLayout(form_layout)
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
        settings_group = QtWidgets.QGroupBox("Evaluation Settings")
        settings_layout = QtWidgets.QVBoxLayout(settings_group)
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
        shear_label = QtWidgets.QLabel("Select Shear Rates:")
        settings_layout.addWidget(shear_label)
        self.shear_rate_list = QtWidgets.QListWidget()
        self.shear_rate_list.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection)
        for shear_name, shear_value in self.shear_rates.items():
            item = QtWidgets.QListWidgetItem(
                f"{shear_value} s⁻¹")
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
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        self.results_tabs = QtWidgets.QTabWidget()
        self.metrics_table = QtWidgets.QTableWidget()
        self.results_tabs.addTab(self.metrics_table, "Metrics")
        plot_widget = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_widget)
        plot_controls = QtWidgets.QHBoxLayout()
        plot_type_label = QtWidgets.QLabel("Plot Type:")
        self.plot_type_combo = QtWidgets.QComboBox()
        self.plot_type_combo.addItem("Predicted vs Actual")
        self.plot_type_combo.currentTextChanged.connect(
            self.on_plot_type_changed)
        plot_controls.addWidget(plot_type_label)
        plot_controls.addWidget(self.plot_type_combo)
        viz_type_label = QtWidgets.QLabel("Visualization:")
        self.viz_type_combo = QtWidgets.QComboBox()
        self.viz_type_combo.addItems([
            "Bar Chart",
            "Line Plot",
            "Box Plot",
            "Violin Plot",
            "Heatmap",
            "Radar Chart"
        ])
        self.viz_type_combo.currentTextChanged.connect(self.update_plot)
        self.viz_type_combo.setEnabled(False)
        plot_controls.addWidget(viz_type_label)
        plot_controls.addWidget(self.viz_type_combo)
        plot_controls.addStretch()
        self.save_figure_btn = QtWidgets.QPushButton("Save Figure")
        self.save_figure_btn.clicked.connect(self.save_current_figure_overall)
        self.save_figure_btn.setEnabled(False)
        plot_controls.addWidget(self.save_figure_btn)
        plot_layout.addLayout(plot_controls)
        self.current_overall_figure = Figure(figsize=(8, 6))
        self.overall_canvas = FigureCanvas(self.current_overall_figure)
        plot_layout.addWidget(self.overall_canvas)
        self.current_shear_index = 0
        shear_nav_layout = QtWidgets.QHBoxLayout()
        shear_nav_layout.addStretch()
        self.prev_shear_btn = QtWidgets.QPushButton("\u25C0 Previous")
        self.prev_shear_btn.clicked.connect(self.navigate_shear_rate_prev)
        self.prev_shear_btn.setEnabled(False)
        self.shear_rate_label = QtWidgets.QLabel("All Shear Rates")
        self.shear_rate_label.setAlignment(Qt.AlignCenter)
        self.shear_rate_label.setMinimumWidth(200)
        self.next_shear_btn = QtWidgets.QPushButton("Next \u25B6")
        self.next_shear_btn.clicked.connect(self.navigate_shear_rate_next)
        self.next_shear_btn.setEnabled(False)
        shear_nav_layout.addWidget(self.prev_shear_btn)
        shear_nav_layout.addWidget(self.shear_rate_label)
        shear_nav_layout.addWidget(self.next_shear_btn)
        shear_nav_layout.addStretch()
        plot_layout.addLayout(shear_nav_layout)
        self.results_tabs.addTab(plot_widget, "Plots")
        self.details_table = QtWidgets.QTableWidget()
        self.results_tabs.addTab(self.details_table, "Detailed Results")
        right_layout.addWidget(self.results_tabs)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([500, 900])
        self.select_model_btn.clicked.connect(self.select_model)
        self.select_run.clicked.connect(self.user_run_browse)
        return widget

    def navigate_shear_rate_prev(self) -> None:
        """Navigate to the previous shear rate in the results visualization.

        This method updates the current shear rate index to the previous available
        shear rate (if one exists), and refreshes both the navigation label and
        the displayed plot accordingly.

        Note:
            The navigation buttons are typically enabled or disabled based on the
            current index through `update_shear_navigation()`.
        """
        if self.current_shear_index > 0:
            self.current_shear_index -= 1
            self.update_shear_navigation()
            self.update_plot()

    def navigate_shear_rate_next(self) -> None:
        """Navigate to the next shear rate in the results visualization.

        This method increments the current shear rate index to move forward
        through the list of selected shear rates, updating both the displayed
        shear rate label and the active plot.

        Note:
            The navigation buttons are managed through `update_shear_navigation()`
            to ensure the "Next" button is disabled when the user reaches the last shear rate.
        """
        max_index = len(self.selected_shear_rates)
        if self.current_shear_index < max_index:
            self.current_shear_index += 1
            self.update_shear_navigation()
            self.update_plot()

    def update_shear_navigation(self) -> None:
        """Update the shear rate navigation controls and display label.

        This method refreshes the navigation buttons and the shear rate label
        based on the current shear rate index (`self.current_shear_index`).
        It ensures that users can only navigate to valid shear rate views
        and that the displayed label accurately reflects the current selection.

        Notes:
            - `self.selected_shear_rates` contains the names of available shear rates.
            - `self.shear_rates` maps shear rate names to numeric shear values.
            - The navigation buttons and label are UI elements defined elsewhere in the class.
        """
        if not self.selected_shear_rates:
            self.prev_shear_btn.setEnabled(False)
            self.next_shear_btn.setEnabled(False)
            self.shear_rate_label.setText("No Shear Rates Selected")
            return
        self.prev_shear_btn.setEnabled(self.current_shear_index > 0)
        self.next_shear_btn.setEnabled(
            self.current_shear_index < len(self.selected_shear_rates)
        )
        if self.current_shear_index == 0:
            self.shear_rate_label.setText("All Shear Rates")
        else:
            shear_name = list(self.selected_shear_rates)[
                self.current_shear_index - 1]
            shear_value = self.shear_rates[shear_name]
            self.shear_rate_label.setText(f"{shear_value} s⁻¹")

    def get_current_shear_filter(self) -> None:
        """Retrieve the currently selected shear rate filter for plotting.

        This method determines which shear rate(s) should be visualized based on
        the user's current navigation index. It is used by plotting functions to
        decide whether to display all available shear rates or a specific one.

        Returns:
            Optional[str]:
                - ``None`` if all shear rates should be plotted (i.e., index is 0).
                - The name of a specific shear rate (e.g., "Viscosity_1000") if an
                individual view is selected (i.e., index > 0).

        Notes:
            - The mapping between shear rate names and their numeric values is
            stored in ``self.shear_rates``.
            - The current navigation position is tracked by ``self.current_shear_index``.
            - ``self.selected_shear_rates`` holds the available shear rate names.
        """
        if self.current_shear_index == 0:
            return None  # Plot all shear rates
        else:
            # Return specific shear rate name
            return list(self.selected_shear_rates)[self.current_shear_index - 1]

    def create_per_run_evaluation_tab(self) -> None:
        """Create and configure the 'Per-Run Evaluation' tab in the EvaluationUI.

        This method constructs the user interface for investigating individual runs
        in detail. Users can select a specific run, view formulation and prediction
        information, and visualize run-specific results.

        The layout consists of a split view:
        - Left panel: Controls and tables for selecting a run and displaying
            run-related data.
        - Right panel: Visualization controls and plot canvas for inspecting
            run-level results.

        Returns:
            QtWidgets.QWidget: The fully configured QWidget representing the
            "Per-Run Evaluation" tab.

        Layout Overview:
            - Run Selection Group
            - ComboBox: Select a run from the list of available runs.
            - Connects `currentTextChanged` signal to `on_run_selected_for_investigation`.
            - Formulation Information Group
            - Table: Displays properties of formulations within the selected run.
            - Read-only, no row selection, alternating row colors.
            - Prediction Information Group
            - Table: Displays run-level predicted metrics for the selected run.
            - Read-only, no row selection, alternating row colors.
            - Visualization Controls (Right Panel)
            - ComboBox: Choose the type of plot (currently only "Viscosity Profile Comparison").
            - Button: Save the current figure to file.
            - Connects signals to `update_run_plot` and `save_current_figure_per_run`.
            - Plot Canvas
            - Displays the generated plot for the selected run.
            - Uses Matplotlib FigureCanvas for embedding.

        Signals:
            - `self.run_combo.currentTextChanged` → Loads selected run for investigation.
            - `self.run_plot_type.currentTextChanged` → Updates the plot type.
            - `self.save_plot_button.clicked` → Saves the currently displayed figure.

        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        run_group = QtWidgets.QGroupBox("Run Selection")
        run_layout = QtWidgets.QVBoxLayout(run_group)
        run_layout.setSpacing(8)
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
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
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
        self.run_figure = Figure(figsize=(10, 7))
        self.run_canvas = FigureCanvas(self.run_figure)
        self.run_canvas.setMinimumHeight(400)
        right_layout.addWidget(self.run_canvas)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([300, 600])
        layout.addWidget(splitter)
        return widget

    def list_view_addPlaceholderText(self) -> None:
        """Add a placeholder item to the runs list view when no runs are imported.

        This method inserts a disabled, non-selectable, and italicized entry
        ("No runs imported") into the `QListView` to indicate that no run files
        have been added. The placeholder helps guide users before any runs are loaded.

        """
        placeholder_item = QtGui.QStandardItem("No runs imported")
        placeholder_item.setEnabled(False)
        placeholder_item.setSelectable(False)
        font = placeholder_item.font()
        font.setItalic(True)
        placeholder_item.setFont(font)
        placeholder_item.setForeground(QtGui.QBrush(Qt.gray))
        self.model.appendRow(placeholder_item)

    def select_model(self) -> None:
        """Open a file dialog to select a VisQ.AI model and initialize the predictor.

        This method allows the user to browse and select a trained VisQ.AI model
        archive (ZIP file). Upon selection, it attempts to initialize a `Predictor`
        instance with the chosen model and updates the UI to reflect the loaded model.
        """
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
                    self.predictor = Predictor(file_path, mc_samples=300)
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

    def user_run_browse(self) -> None:
        """Open a dialog to browse and add run capture files or directories.

        This method allows the user to select one or more directories containing
        run files. It will search each selected directory for `capture.zip` files
        and load them all in batch.
        """
        # Use a custom directory dialog that supports multi-selection
        dir_dialog = QtWidgets.QFileDialog(self)
        dir_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dir_dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        dir_dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        file_view = dir_dialog.findChild(QtWidgets.QListView, 'listView')
        if file_view:
            file_view.setSelectionMode(
                QtWidgets.QAbstractItemView.ExtendedSelection)

        tree_view = dir_dialog.findChild(QtWidgets.QTreeView)
        if tree_view:
            tree_view.setSelectionMode(
                QtWidgets.QAbstractItemView.ExtendedSelection)
        prefer_abs = os.path.abspath(Constants.log_prefer_path)
        dir_dialog.setDirectory(prefer_abs)

        if dir_dialog.exec_():
            selected_dirs = dir_dialog.selectedFiles()
            if selected_dirs:
                self.load_runs_from_directories(selected_dirs)

    def load_runs_from_directories(self, directories: list) -> None:
        """Search multiple directories for capture files and load them.

        Args:
            directories (list): List of directory paths to search.
        """
        all_capture_files = []
        for directory in directories:
            capture_files = self.find_capture_files(directory)
            all_capture_files.extend(capture_files)

        if not all_capture_files:
            QtWidgets.QMessageBox.information(
                self,
                "No Capture Files Found",
                f"No capture.zip files found in {len(directories)} selected director{'y' if len(directories) == 1 else 'ies'}."
            )
            return
        dir_summary = f"{len(directories)} director{'y' if len(directories) == 1 else 'ies'}"
        reply = QtWidgets.QMessageBox.question(
            self,
            "Batch Load Runs",
            f"Found {len(all_capture_files)} run file(s) in {dir_summary}.\n"
            f"Do you want to load all of them?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes
        )

        if reply == QtWidgets.QMessageBox.Yes:
            self.batch_add_run_files(all_capture_files)

    def find_capture_files(self, directory: str, filename: str = "capture.zip") -> list:
        """Recursively search a directory for capture files.

        Args:
            directory (str): The root directory to search.
            filename (str): The filename to search for (default: "capture.zip").

        Returns:
            list: A list of absolute paths to found capture files.
        """
        capture_files = []

        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower() == filename.lower():
                        capture_files.append(os.path.join(root, file))
        except Exception as e:
            Log.e(TAG, f"Error searching directory {directory}: {e}")

        return capture_files

    def batch_add_run_files(self, file_paths: list) -> None:
        """Load multiple run files in batch and update the UI.

        This method processes a list of run files, attempting to load each one.
        It provides feedback on the batch loading process, including success
        and failure counts.

        Args:
            file_paths (list): List of file paths to be loaded.
        """
        if not file_paths:
            return

        # Show progress dialog for batch loading
        progress = QtWidgets.QProgressDialog(
            "Loading run files...",
            "Cancel",
            0,
            len(file_paths),
            self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        loaded_count = 0
        failed_files = []

        for i, file_path in enumerate(file_paths):
            if progress.wasCanceled():
                break

            progress.setValue(i)
            progress.setLabelText(f"Loading: {os.path.basename(file_path)}")
            QtWidgets.QApplication.processEvents()

            success = self.add_run_file(file_path, show_errors=False)
            if success:
                loaded_count += 1
            else:
                failed_files.append(file_path)

        progress.setValue(len(file_paths))

        # Show summary
        summary_msg = f"Successfully loaded {loaded_count} of {len(file_paths)} run file(s)."
        if failed_files:
            summary_msg += f"\n\nFailed to load {len(failed_files)} file(s):"
            for failed in failed_files[:5]:  # Show first 5 failures
                summary_msg += f"\n  • {os.path.basename(failed)}"
            if len(failed_files) > 5:
                summary_msg += f"\n  ... and {len(failed_files) - 5} more"

        if failed_files:
            QtWidgets.QMessageBox.warning(
                self,
                "Batch Load Complete",
                summary_msg
            )
        elif loaded_count > 0:
            QtWidgets.QMessageBox.information(
                self,
                "Batch Load Complete",
                summary_msg
            )

    def add_run_file(self, file_path: str, show_errors: bool = True) -> bool:
        """Load a run file, extract formulation data, and update the UI.

        This method parses a selected run file (e.g., `capture.zip`) using the
        `Parser` class, retrieves the associated formulation, and updates the
        internal data structures and user interface.

        Args:
            file_path (str): Path to the run file to be loaded.
            show_errors (bool): Whether to show error dialogs (default: True).

        Returns:
            bool: True if the file was loaded successfully, False otherwise.
        """
        try:
            parser = Parser(file_path)
            formulation = parser.get_formulation()
            run_name = parser.get_run_name()

            if not formulation:
                if show_errors:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "No Formulation",
                        "No valid formulation found in the selected file."
                    )
                Log.w(TAG, f"No formulation found in: {file_path}")
                return False

            # Check for duplicate run names
            if run_name in self.formulations_by_run:
                Log.w(
                    TAG, f"Duplicate run name '{run_name}', skipping: {file_path}")
                if show_errors:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Duplicate Run",
                        f"Run '{run_name}' is already loaded."
                    )
                return False

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

            return True

        except Exception as e:
            Log.e(TAG, f"Failed to load run: {e}")
            if show_errors:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Run Loading Error",
                    f"Failed to load run file: {str(e)}"
                )
            return False

    def update_formulation_info(self) -> None:
        """Update and log information about currently loaded formulations."""
        if not self.selected_formulations:
            Log.i(TAG, "No formulations loaded")
            return
        info_text = f"Total Formulations: {len(self.selected_formulations)}\n"
        info_text += f"Runs Loaded: {len(self.formulations_by_run)}\n"
        Log.i(TAG, info_text)

    def check_ready_to_evaluate(self) -> None:
        """Check if ready to evaluate and update UI accordingly."""
        ready = (self.predictor is not None and
                 len(self.selected_formulations) > 0 and
                 len(self.selected_metrics) > 0)
        self.evaluate_btn.setEnabled(ready)

    def user_run_clicked(self, index: int):
        """Handle the user clicking a run item in the runs list view.

        This method responds to a click event on an item in the run `QListView`.
        When a valid, enabled run item is clicked, it updates the corresponding
        label in the UI to display the selected run file name.

        Args:
            index (int): The index of the clicked item in the list view.
        """
        item = self.model.itemFromIndex(index)
        if item and item.isEnabled():
            file_path = item.data(Qt.UserRole)
            if file_path:
                self.select_label.setText(os.path.basename(file_path))

    def user_run_removed(self) -> None:
        """Remove the currently selected run(s) from the list and update UI.

        This method handles removal of selected run items from the runs `QListView`.
        It also updates internal data structures and UI elements to reflect the
        removed runs.
        """
        selected_indexes = self.list_view.selectedIndexes()
        if not selected_indexes:
            return

        for index in reversed(selected_indexes):
            item = self.model.itemFromIndex(index)
            if item and item.isEnabled():
                run_name = item.text()
                if run_name in self.formulations_by_run:
                    forms_to_remove = self.formulations_by_run[run_name]
                    for form in forms_to_remove:
                        if form in self.selected_formulations:
                            self.selected_formulations.remove(form)
                    del self.formulations_by_run[run_name]
                    idx = self.run_combo.findText(run_name)
                    if idx >= 0:
                        self.run_combo.removeItem(idx)
                self.model.removeRow(index.row())

        if self.model.rowCount() == 0:
            self.list_view_addPlaceholderText()
            self.select_label.clear()

        self.update_formulation_info()
        self.check_ready_to_evaluate()

    def user_all_runs_removed(self) -> None:
        """Remove all runs from the list and reset related UI and data.

        This method clears all imported runs from the EvaluationUI, including
        the list view, per-run combo box, and internal data structures that
        track selected formulations and run-to-formulation mappings. It also
        resets associated UI elements.
        """
        self.model.clear()
        self.selected_formulations.clear()
        self.formulations_by_run.clear()
        self.run_combo.clear()
        self.list_view_addPlaceholderText()
        self.select_label.clear()
        self.update_formulation_info()
        self.check_ready_to_evaluate()

    def on_metric_selection_changed(self) -> None:
        """Update selected metrics when the user changes selection in the metrics list.

        This method responds to selection changes in `self.metrics_list` (a
        multi-selection `QListWidget`) and updates the internal list of metrics
        to evaluate. It also checks whether the UI is ready for model evaluation.
        """
        self.selected_metrics = []
        for i in range(self.metrics_list.count()):
            item = self.metrics_list.item(i)
            if item.isSelected():
                self.selected_metrics.append(item.data(Qt.UserRole))

        self.check_ready_to_evaluate()

    def on_shear_rate_selection_changed(self) -> None:
        """Update selected shear rates when the user changes selection in the shear rate list.

        This method responds to selection changes in `self.shear_rate_list` (a
        multi-selection `QListWidget`) and updates the internal list of shear rates
        to display in plots. It also updates navigation and refreshes plots if
        evaluation results are available.
        """
        selected_items = self.shear_rate_list.selectedItems()

        # Helper to extract the numeric part from strings like "Viscosity_1000"
        def extract_shear_rate(value: str) -> float:
            match = re.search(r'(\d+(?:\.\d+)?)$', str(value))
            return float(match.group(1)) if match else float('inf')

        # Sort by numeric suffix
        self.selected_shear_rates = sorted(
            (item.data(Qt.UserRole) for item in selected_items),
            key=extract_shear_rate
        )

        self.current_shear_index = 0
        self.update_shear_navigation()

        if hasattr(self, 'evaluation_results') and self.evaluation_results is not None:
            self.update_plot()
            self.display_details_table()
            self.display_metrics_table()

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

    def evaluate_model(self) -> None:
        """Evaluate the selected model on imported formulations.

        This method performs a full evaluation of the currently selected
        `Predictor` model using the imported formulations from `self.formulations_by_run`.
        It generates per-run and overall results, maps formulation IDs, extracts
        numeric shear rates, and updates the UI with evaluation metrics and plots.
        """
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
                               [self.shear_rates[name] for name in self.shear_rates.keys()]]
                Log.i(
                    f"Evaluating {len(eval_data)} samples with targets: {target_cols}")
                results_df = self.predictor.evaluate(
                    eval_data=eval_data,
                    targets=target_cols,
                    n_samples=None
                )
                results_df['run'] = run_name
                formulation_ids = []
                for idx in results_df['sample_idx']:
                    if idx < len(formulations):
                        formulation_ids.append(formulations[idx].id)
                    else:
                        formulation_ids.append(f"sample_{idx}")
                results_df['formulation_id'] = formulation_ids
                results_df = results_df.rename(columns={
                    'pct_error': 'percentage_error'
                })

                def extract_numeric_shear_rate(shear_col_name):
                    try:
                        if isinstance(shear_col_name, str) and shear_col_name.startswith("Viscosity_"):
                            return int(shear_col_name.split("_")[1])
                        return float(shear_col_name)
                    except (ValueError, IndexError):
                        return shear_col_name
                results_df['shear_rate_name'] = results_df['shear_rate']
                results_df['shear_rate'] = results_df['shear_rate_name'].apply(
                    extract_numeric_shear_rate)
                self.run_results[run_name] = results_df.copy()

                all_results_list.append(results_df)
            if all_results_list:
                self.current_results_df = pd.concat(
                    all_results_list, ignore_index=True)
                self.evaluation_results = self.current_results_df
                Log.i(
                    f"Evaluation complete: {len(self.current_results_df)} predictions")
                self.display_results()
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

    def display_results(self) -> None:
        """Display evaluation results in the UI tables and plots.

        This method updates the EvaluationUI to show the results of the most
        recent model evaluation. It handles metrics, detailed results, and
        visualization setup.
        """
        if self.current_results_df is None or self.current_results_df.empty:
            return
        self.display_metrics_table()
        self.display_details_table()
        self.populate_plot_type_combo()
        self.update_plot()

    def populate_plot_type_combo(self) -> None:
        """Update the plot type combo box based on currently selected metrics.

        This method refreshes the options available in `self.plot_type_combo`
        to reflect the metrics currently selected by the user. It ensures that
        the previous selection is preserved if possible.
        """
        current_selection = self.plot_type_combo.currentText()
        self.plot_type_combo.blockSignals(True)
        self.plot_type_combo.clear()
        self.plot_type_combo.addItem("Predicted vs Actual")
        for metric in self.selected_metrics:
            display_name = self.format_metric_name(metric)
            self.plot_type_combo.addItem(
                display_name, metric)
        index = self.plot_type_combo.findText(current_selection)
        if index >= 0:
            self.plot_type_combo.setCurrentIndex(index)
        else:
            self.plot_type_combo.setCurrentIndex(0)

        self.plot_type_combo.blockSignals(False)

    def on_plot_type_changed(self, text: str):
        """Handle changes in the plot type combo box selection.

        This method is triggered when the user selects a different plot type
        from `self.plot_type_combo`. It enables or disables the visualization
        type combo box depending on whether the selection corresponds to a metric
        or a default plot, and refreshes the plot.
        """
        is_metric = (text != "Predicted vs Actual" and text != "")
        self.viz_type_combo.setEnabled(is_metric)
        self.update_plot()

    def display_metrics_table(self) -> None:
        """Populate the metrics table with evaluation results.

        This method computes and displays evaluation metrics in `self.metrics_table`
        based on the currently selected metrics and shear rates. It handles both
        overall metrics and per-shear rate metrics depending on the user's evaluation
        type selection.
        """
        metrics_data = []
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
        if self.per_shear_radio.isChecked() or self.both_radio.isChecked():
            per_shear_metrics = self.metrics.compute_per_shear_rate(
                self.current_results_df,
                self.selected_metrics
            )
            selected_shear_rate_values = [
                self.shear_rates[name] for name in self.selected_shear_rates]

            for _, row in per_shear_metrics.iterrows():
                if row['shear_rate'] not in selected_shear_rate_values:
                    continue

                for metric in self.selected_metrics:
                    if metric in row:
                        metrics_data.append({
                            'Type': f"Shear Rate {row['shear_rate']:.0f}",
                            'Metric': self.format_metric_name(metric),
                            'Value': f"{row[metric]:.4f}",
                            'Description': self.METRIC_DESCRIPTIONS.get(metric, '')
                        })
        df = pd.DataFrame(metrics_data)
        self.metrics_table.setRowCount(len(df))
        self.metrics_table.setColumnCount(len(df.columns))
        self.metrics_table.setHorizontalHeaderLabels(df.columns.tolist())
        self.metrics_table.verticalHeader().setVisible(False)
        header_font = self.metrics_table.horizontalHeader().font()
        header_font.setBold(True)
        header_font.setPointSize(10)
        self.metrics_table.horizontalHeader().setFont(header_font)
        self.metrics_table.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
                background-color: #E8F4F8;
                color: #1a1a1a;
                padding: 6px;
                border: 1px solid #B8D4E0;
                font-weight: bold;
            }
        """)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #D0D0D0;
                background-color: white;
                alternate-background-color: #F5F5F5;
                selection-background-color: #B8D4E0;
            }
            QTableWidget::item {
                padding: 4px;
            }
        """)
        for i, row in df.iterrows():
            for j, value in enumerate(row):
                item = QtWidgets.QTableWidgetItem(str(value))
                if df.columns[j] == 'Value':
                    item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                if df.columns[j] in ['Type']:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                if df.columns[j] == 'Description':
                    item.setToolTip(str(value))

                self.metrics_table.setItem(i, j, item)
        self.metrics_table.resizeColumnsToContents()
        header = self.metrics_table.horizontalHeader()
        for i in range(len(df.columns) - 1):
            header.setSectionResizeMode(
                i, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(
            len(df.columns) - 1, QtWidgets.QHeaderView.Stretch)
        self.metrics_table.setColumnWidth(
            0, max(100, self.metrics_table.columnWidth(0)))  # Type
        self.metrics_table.setColumnWidth(
            1, max(120, self.metrics_table.columnWidth(1)))  # Metric
        self.metrics_table.setColumnWidth(
            2, max(80, self.metrics_table.columnWidth(2)))   # Value
        for i in range(self.metrics_table.rowCount()):
            self.metrics_table.setRowHeight(i, 28)

    def display_details_table(self) -> None:
        """Populate the detailed results table with individual predictions.

        This method displays per-sample evaluation results in `self.details_table`,
        filtered by the currently selected shear rates. It includes actual and
        predicted values, errors, confidence intervals, and run information.
        """
        # Clear the table first
        self.details_table.clearContents()
        self.details_table.setRowCount(0)
        self.details_table.setSortingEnabled(False)

        # Disable editing
        self.details_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)

        df = self.current_results_df.copy()
        selected_shear_rate_values = [self.shear_rates[name]
                                      for name in self.selected_shear_rates]
        df = df[df['shear_rate'].isin(selected_shear_rate_values)]

        display_columns = ['run', 'shear_rate', 'actual',
                           'predicted', 'abs_error', 'percentage_error',
                           'lower_ci', 'upper_ci']
        df = df[display_columns]

        # Reset index to ensure sequential row numbering
        df = df.reset_index(drop=True)

        column_display_names = {
            'run': 'Run',
            'shear_rate': 'Shear Rate (s⁻¹)',
            'actual': 'Actual',
            'predicted': 'Predicted',
            'abs_error': 'Abs Error',
            'percentage_error': 'Error %',
            'lower_ci': 'CI Lower',
            'upper_ci': 'CI Upper'
        }
        display_labels = [column_display_names.get(
            col, col) for col in df.columns]
        self.details_table.setRowCount(len(df))
        self.details_table.setColumnCount(len(df.columns))
        self.details_table.setHorizontalHeaderLabels(display_labels)
        self.details_table.verticalHeader().setVisible(False)

        # Configure header for sorting
        header = self.details_table.horizontalHeader()
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header_font = header.font()
        header_font.setBold(True)
        header_font.setPointSize(10)
        header.setFont(header_font)
        header.setStyleSheet("""
            QHeaderView::section {
                background-color: #E8F4F8;
                color: #1a1a1a;
                padding: 6px;
                border: 1px solid #B8D4E0;
                font-weight: bold;
            }
            QHeaderView::section:hover {
                background-color: #D0E8F0;
            }
            QHeaderView::up-arrow {
                width: 12px;
                height: 12px;
                image: url(none);
                subcontrol-origin: padding;
                subcontrol-position: center right;
            }
            QHeaderView::down-arrow {
                width: 12px;
                height: 12px;
                image: url(none);
                subcontrol-origin: padding;
                subcontrol-position: center right;
            }
        """)

        self.details_table.setAlternatingRowColors(True)
        self.details_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #D0D0D0;
                background-color: white;
                alternate-background-color: #F5F5F5;
                selection-background-color: #B8D4E0;
            }
            QTableWidget::item {
                padding: 4px;
            }
        """)

        for i, row in df.iterrows():
            for j, value in enumerate(row):
                col_name = df.columns[j]
                if isinstance(value, float):
                    if col_name == 'percentage_error':
                        item = QtWidgets.QTableWidgetItem()
                        item.setData(Qt.DisplayRole, f"{value:.2f}%")
                        # Store numeric value for sorting
                        item.setData(Qt.UserRole, value)
                    elif col_name == 'shear_rate':
                        item = QtWidgets.QTableWidgetItem()
                        item.setData(Qt.DisplayRole, f"{value:.0f}")
                        item.setData(Qt.UserRole, value)
                    else:
                        item = QtWidgets.QTableWidgetItem()
                        item.setData(Qt.DisplayRole, f"{value:.4f}")
                        item.setData(Qt.UserRole, value)
                else:
                    item = QtWidgets.QTableWidgetItem(str(value))

                if col_name == 'run':
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)

                if col_name in ['abs_error', 'percentage_error']:
                    if isinstance(value, float):
                        if col_name == 'percentage_error':
                            if value < 5.0:
                                item.setBackground(QtGui.QColor(220, 255, 220))
                            elif value < 15.0:
                                item.setBackground(QtGui.QColor(255, 255, 200))
                            else:
                                item.setBackground(QtGui.QColor(255, 220, 220))
                        elif col_name == 'abs_error':
                            if value < 0.1:
                                item.setBackground(QtGui.QColor(220, 255, 220))
                            elif value < 0.3:
                                item.setBackground(QtGui.QColor(255, 255, 200))
                            else:
                                item.setBackground(QtGui.QColor(255, 220, 220))

                self.details_table.setItem(i, j, item)

        self.details_table.resizeColumnsToContents()

        min_widths = {
            0: 100,  # Run
            1: 90,   # Shear Rate
            2: 90,   # Actual
            3: 90,   # Predicted
            4: 90,   # Abs Error
            5: 90,   # Error %
            6: 90,   # CI Lower
            7: 90    # CI Upper
        }

        for col, min_width in min_widths.items():
            if col < self.details_table.columnCount():
                current_width = self.details_table.columnWidth(col)
                self.details_table.setColumnWidth(
                    col, max(min_width, current_width))

        # Stretch Run column to fill horizontal space, keep others interactive
        header.setStretchLastSection(False)
        header.setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch)  # Run column stretches
        for col in range(1, self.details_table.columnCount()):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.Interactive)

        for i in range(self.details_table.rowCount()):
            self.details_table.setRowHeight(i, 26)

        self.details_table.setSortingEnabled(True)

    def update_plot(self) -> None:
        """Render the current plot based on the selected plot type and visualization.

        This method updates `self.current_overall_figure` to display either
        a predicted vs actual plot or a metric-based visualization depending
        on the user's selection in `self.plot_type_combo` and `self.viz_type_combo`.
        """
        if not hasattr(self, 'evaluation_results') or self.evaluation_results is None:
            return
        self.save_figure_btn.setEnabled(True)
        self.current_overall_figure.clear()
        plot_type = self.plot_type_combo.currentText()
        if plot_type == "Predicted vs Actual":
            self._plot_predicted_vs_actual()
        elif plot_type != "":
            current_index = self.plot_type_combo.currentIndex()
            metric_key = self.plot_type_combo.itemData(current_index)
            if metric_key:
                viz_type = self.viz_type_combo.currentText()
                self._plot_metric_visualization(metric_key, viz_type)
        self.overall_canvas.draw()

    def _plot_predicted_vs_actual(self) -> None:
        """Generate a Predicted vs Actual viscosity scatter plot.

        This method creates a scatter plot comparing predicted vs actual
        viscosity values. Points are colored by percentage error, and
        visual guides indicate perfect predictions and ±10% error bounds.
        """
        ax = self.current_overall_figure.add_subplot(111)
        shear_filter = self.get_current_shear_filter()
        if shear_filter is None:
            plot_data = self.evaluation_results
            title_suffix = "(All Shear Rates)"
        else:
            plot_data = self.evaluation_results[
                self.evaluation_results['shear_rate'] == self.shear_rates[shear_filter]
            ]
            shear_value = self.shear_rates[shear_filter]
            title_suffix = f"({shear_filter}: {shear_value} s⁻¹)"

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
        min_val = min(plot_data['actual'].min(), plot_data['predicted'].min())
        max_val = max(plot_data['actual'].max(), plot_data['predicted'].max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            'k--',
            alpha=0.5,
            label='Perfect Prediction'
        )
        ax.fill_between(
            [min_val, max_val],
            [min_val*0.9, max_val*0.9],
            [min_val*1.1, max_val*1.1],
            alpha=0.2,
            color='gray',
            label='±10% Error'
        )
        ax.set_xlabel('Actual Viscosity (cP)', fontsize=12)
        ax.set_ylabel('Predicted Viscosity (cP)', fontsize=12)
        ax.set_title(
            f'Predicted vs Actual Viscosity {title_suffix}',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        cbar = self.current_overall_figure.colorbar(scatter, ax=ax)
        cbar.set_label('Percentage Error (%)', rotation=270, labelpad=20)
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

    def _plot_metric_visualization(self, metric: str, viz_type: str):
        """Visualize a selected metric across all chosen shear rates.

        This method aggregates metric values for each selected shear rate and 
        renders them using the specified visualization type.

        Args:
            metric (str): The metric key to plot (e.g., 'mae', 'rmse').
            viz_type (str): The type of visualization to use 
                            (e.g., 'Bar Chart', 'Line Plot', 'Heatmap', 'Radar Chart').
        """
        shear_filter = self.get_current_shear_filter()
        if shear_filter is None:
            self._plot_metric_across_shear_rates(metric, viz_type)
        else:
            self._plot_metric_for_shear_rate(metric, shear_filter, viz_type)

    def _plot_metric_across_shear_rates(self, metric: str, viz_type: str):
        """Plot a selected metric across all chosen shear rates.

        This method aggregates the metric values for each selected shear rate
        from `self.evaluation_results` and renders a visualization using the
        specified type (bar chart, line plot, heatmap, or radar chart).

        Args:
            metric (str): The metric key to visualize (e.g., 'mae', 'rmse').
            viz_type (str): The type of visualization (e.g., 'Bar Chart', 
                            'Line Plot', 'Heatmap', 'Radar Chart').
        """
        ax = self.current_overall_figure.add_subplot(111)
        metric_values = []
        shear_labels = []

        for shear_name in self.selected_shear_rates:
            shear_value = self.shear_rates[shear_name]
            shear_data = self.evaluation_results[
                self.evaluation_results['shear_rate'] == shear_value
            ]

            if not shear_data.empty:
                metric_result = self.metrics.compute_overall(
                    shear_data, [metric])
                metric_values.append(metric_result[metric])
                shear_labels.append(f"{shear_value} s⁻¹")

        if not metric_values:
            ax.text(0.5, 0.5, 'No data available',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, color='#4C566A')
            return
        if viz_type == "Bar Chart":
            self._plot_bar_chart(ax, metric_values, shear_labels, metric)
        elif viz_type == "Line Plot":
            self._plot_line_plot(ax, metric_values, shear_labels, metric)
        elif viz_type == "Heatmap":
            self._plot_heatmap(ax, metric_values, shear_labels, metric)
        elif viz_type == "Radar Chart":
            self._plot_radar_chart(metric_values, shear_labels, metric)
        else:
            self._plot_bar_chart(ax, metric_values, shear_labels, metric)

        self.current_overall_figure.tight_layout()

    def _plot_metric_across_shear_rates(self, metric: str, viz_type: str):
        """Visualize a metric across all selected shear rates with the chosen plot type.

        This method computes the overall metric value for each selected shear rate
        and renders a visualization using the specified type (bar chart, line plot,
        heatmap, or radar chart). Data points are sorted by shear rate for consistent display.

        Args:
            metric (str): The metric key to visualize (e.g., 'mae', 'rmse').
            viz_type (str): The type of visualization to use 
                            (e.g., 'Bar Chart', 'Line Plot', 'Heatmap', 'Radar Chart').
        """
        ax = self.current_overall_figure.add_subplot(111)
        data_points = []
        for shear_name in self.selected_shear_rates:
            shear_value = self.shear_rates[shear_name]
            shear_data = self.evaluation_results[
                self.evaluation_results['shear_rate'] == shear_value
            ]

            if not shear_data.empty:
                metric_result = self.metrics.compute_overall(
                    shear_data, [metric])
                data_points.append({
                    'shear_value': shear_value,
                    'metric_value': metric_result[metric],
                    'label': f"{shear_value} s⁻¹"
                })

        data_points.sort(key=lambda x: x['shear_value'])
        if not data_points:
            self._show_no_data_message(ax)
            return
        metric_values = [dp['metric_value'] for dp in data_points]
        shear_labels = [dp['label'] for dp in data_points]
        if viz_type == "Bar Chart":
            self._plot_bar_chart(ax, metric_values, shear_labels, metric,
                                 color_scheme='primary')
        elif viz_type == "Line Plot":
            self._plot_line_plot(ax, metric_values, shear_labels, metric,
                                 color_scheme='primary')
        elif viz_type == "Heatmap":
            self._plot_heatmap(ax, metric_values, shear_labels, metric)
        elif viz_type == "Radar Chart":
            self._plot_radar_chart(metric_values, shear_labels, metric)
        else:
            self._plot_bar_chart(ax, metric_values, shear_labels, metric,
                                 color_scheme='primary')

        self.current_overall_figure.tight_layout()

    def _plot_metric_for_shear_rate(self, metric: str, shear_name: str, viz_type: str):
        """Visualize a metric for a specific shear rate using the selected plot type.

        This method extracts the metric values for a given shear rate from
        `self.evaluation_results` and generates a visualization. If multiple
        runs are present, it displays metric values per run; otherwise, it shows
        a single overall value.

        Args:
            metric (str): The metric key to visualize (e.g., 'mae', 'rmse').
            shear_name (str): The name of the shear rate (must exist in `self.shear_rates`).
            viz_type (str): The type of visualization to use 
                            (e.g., 'Bar Chart', 'Line Plot', 'Box Plot', 'Violin Plot').
        """
        ax = self.current_overall_figure.add_subplot(111)
        shear_value = self.shear_rates[shear_name]
        shear_data = self.evaluation_results[
            self.evaluation_results['shear_rate'] == shear_value
        ]

        if shear_data.empty:
            self._show_no_data_message(ax)
            ax.set_title(
                f'{self.format_metric_name(metric)} at {shear_value} s⁻¹',
                fontsize=13, fontweight=600, color='#2E3440', pad=15
            )
            return
        if 'run' in shear_data.columns and len(shear_data['run'].unique()) > 1:
            run_names = shear_data['run'].unique()
            metric_values = []
            labels = []

            for run_name in run_names:
                run_data = shear_data[shear_data['run'] == run_name]
                metric_result = self.metrics.compute_overall(
                    run_data, [metric])
                metric_values.append(metric_result[metric])
                label = run_name if len(
                    run_name) < 20 else run_name[:17] + '...'
                labels.append(label)
        else:
            metric_result = self.metrics.compute_overall(shear_data, [metric])
            metric_values = [metric_result[metric]]
            labels = ['Overall']
        if viz_type == "Bar Chart":
            self._plot_bar_chart(ax, metric_values, labels, metric,
                                 color_scheme='secondary', shear_value=shear_value)
        elif viz_type == "Line Plot":
            self._plot_line_plot(ax, metric_values, labels, metric,
                                 color_scheme='secondary', shear_value=shear_value)
        elif viz_type == "Box Plot":
            self._plot_box_plot(ax, shear_data, metric, labels, metric_values,
                                run_names if 'run' in shear_data.columns else None,
                                shear_value)
        elif viz_type == "Violin Plot":
            self._plot_violin_plot(ax, shear_data, metric, labels, metric_values,
                                   run_names if 'run' in shear_data.columns else None,
                                   shear_value)
        else:
            self._plot_bar_chart(ax, metric_values, labels, metric,
                                 color_scheme='secondary', shear_value=shear_value)

        self.current_overall_figure.tight_layout()

    def _show_no_data_message(self, ax):
        """Display a 'no data available' message on the axes.

        Args:
            ax: Matplotlib axes object
        """
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, color='#4C566A')

    def _get_color_scheme(self, scheme: str):
        """Get color scheme for plotting.

        Args:
            scheme: 'primary' for across-shear plots, 'secondary' for shear-specific plots

        Returns:
            dict: Dictionary with color keys
        """
        if scheme == 'primary':
            return {
                'main': '#00A3DA',
                'accent': '#32E2DF',
                'light': '#69EAC5',
                'edge': '#2E3440'
            }
        else:  # secondary
            return {
                'main': '#32E2DF',
                'accent': '#00A3DA',
                'light': '#69EAC5',
                'edge': '#2E3440'
            }

    def _setup_axes_labels(self, ax, labels: list, ylabel: str):
        """Setup x-axis ticks and y-axis label.

        Args:
            ax: Matplotlib axes object
            labels: List of x-axis labels
            ylabel: Y-axis label text
        """
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(ylabel, fontsize=11, color="#4C566A")

    def _add_value_labels(self, ax, positions, values: list, offset='bar'):
        """Add value labels to plot elements.

        Args:
            ax: Matplotlib axes object
            positions: X positions or bar objects
            values: List of values to display
            offset: 'bar' for bar charts, 'point' for line plots
        """
        if offset == 'bar':
            for bar, val in zip(positions, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{val:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight=600,
                    color="#2E3440"
                )
        else:
            for i, val in enumerate(values):
                ax.text(
                    i,
                    val,
                    f'{val:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight=600,
                    color="#2E3440"
                )

    def _extract_error_data(self, shear_data: pd.DataFrame, metric: str, run_names):
        """Extract error data for box/violin plots.

        Args:
            shear_data: DataFrame with shear rate data
            metric: The metric key
            run_names: List of run names or None

        Returns:
            list: List of arrays with error data per run, or None if not applicable
        """
        if metric not in ['mae', 'rmse', 'max_error', 'median_ae'] or 'abs_error' not in shear_data.columns:
            return None

        if run_names is None or len(run_names) <= 1:
            return None

        data_per_run = []
        for run_name in run_names:
            run_data = shear_data[shear_data['run'] == run_name]
            if metric == 'mae':
                data_per_run.append(run_data['abs_error'].values)
            elif metric == 'rmse':
                data_per_run.append(np.sqrt(run_data['abs_error'].values,  2))
            else:
                data_per_run.append(run_data['abs_error'].values)

        return data_per_run if data_per_run else None

    def _apply_standard_styling(self, ax, metric: str, title_suffix: str = "Across All Shear Rates",
                                include_grid: bool = True):
        """Apply consistent styling to axes.

        Args:
            ax: Matplotlib axes object
            metric: The metric key
            title_suffix: Suffix for the title
            include_grid: Whether to include grid lines
        """
        ax.set_title(
            f'{self.format_metric_name(metric)} {title_suffix}',
            fontsize=13,
            fontweight=600,
            color="#2E3440",
            pad=15
        )

        if include_grid:
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

    def _plot_bar_chart(self, ax, metric_values: list, labels: list, metric: str,
                        color_scheme: str = 'primary', shear_value: float = None):
        """Create a bar chart visualization.

        Args:
            ax: Matplotlib axes object
            metric_values: List of metric values
            labels: List of labels
            metric: The metric key
            color_scheme: 'primary' or 'secondary' color scheme
            shear_value: Optional shear rate value for title
        """
        colors = self._get_color_scheme(color_scheme)

        bars = ax.bar(
            range(len(metric_values)),
            metric_values,
            color=colors['main'],
            alpha=0.85,
            edgecolor=colors['edge'],
            linewidth=1.2
        )

        self._setup_axes_labels(ax, labels, self.format_metric_name(metric))
        self._add_value_labels(ax, bars, metric_values, offset='bar')

        if shear_value is not None:
            title_suffix = f"at {shear_value} s⁻¹"
        else:
            title_suffix = "Across All Shear Rates"

        self._apply_standard_styling(ax, metric, title_suffix)

    def _plot_line_plot(self, ax, metric_values: list, labels: list, metric: str,
                        color_scheme: str = 'primary', shear_value: float = None):
        """Create a line plot visualization.

        Args:
            ax: Matplotlib axes object
            metric_values: List of metric values
            labels: List of labels
            metric: The metric key
            color_scheme: 'primary' or 'secondary' color scheme
            shear_value: Optional shear rate value for title
        """
        colors = self._get_color_scheme(color_scheme)

        ax.plot(
            range(len(metric_values)),
            metric_values,
            marker='o',
            linewidth=2.5,
            markersize=10,
            color=colors['main'],
            markerfacecolor=colors['accent'],
            markeredgewidth=2,
            markeredgecolor=colors['main'],
            alpha=0.85
        )

        self._setup_axes_labels(ax, labels, self.format_metric_name(metric))
        self._add_value_labels(ax, range(len(metric_values)),
                               metric_values, offset='point')
        if shear_value is not None:
            title_suffix = f"at {shear_value} s⁻¹"
        else:
            title_suffix = "Across All Shear Rates"

        self._apply_standard_styling(ax, metric, title_suffix)

    def _plot_heatmap(self, ax, metric_values: list, labels: list, metric: str):
        """Create a heatmap visualization.

        Args:
            ax: Matplotlib axes object
            metric_values: List of metric values
            labels: List of labels
            metric: The metric key
        """
        data = np.array(metric_values).reshape(1, -1)
        colors_map = ['#69EAC5', '#32E2DF', '#00A3DA']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list(
            'custom', colors_map, N=n_bins)

        im = ax.imshow(data, cmap=cmap, aspect='auto', alpha=0.85)
        self._setup_axes_labels(ax, labels, '')
        ax.set_yticks([])
        cbar = self.current_overall_figure.colorbar(
            im, ax=ax, orientation='horizontal', pad=0.15
        )
        cbar.set_label(self.format_metric_name(
            metric), fontsize=11, color="#4C566A")
        cbar.ax.tick_params(colors="#4C566A", labelsize=9)
        cbar.outline.set_edgecolor("#E5E9F0")
        cbar.outline.set_linewidth(1)
        for i, val in enumerate(metric_values):
            ax.text(
                i,
                0,
                f'{val:.3f}',
                ha='center',
                va='center',
                color="#2E3440",
                fontweight=600,
                fontsize=10
            )
        self._apply_standard_styling(
            ax, metric, "Across All Shear Rates", include_grid=False)

    def _plot_radar_chart(self, metric_values: list, labels: list, metric: str):
        """Create a radar chart visualization.

        Args:
            metric_values: List of metric values
            labels: List of labels
            metric: The metric key
        """
        colors = self._get_color_scheme('primary')

        angles = np.linspace(
            0, 2 * np.pi, len(metric_values), endpoint=False
        ).tolist()
        values = metric_values + [metric_values[0]]
        angles += angles[:1]

        ax = self.current_overall_figure.add_subplot(111, projection='polar')

        ax.plot(
            angles,
            values,
            'o-',
            linewidth=2.5,
            color=colors['main'],
            markersize=8,
            alpha=0.85
        )
        ax.fill(angles, values, alpha=0.15, color=colors['light'])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=9, color="#4C566A")
        ax.set_ylabel(
            self.format_metric_name(metric),
            labelpad=30,
            fontsize=11,
            color="#4C566A"
        )
        ax.set_title(
            f'{self.format_metric_name(metric)} Across All Shear Rates',
            fontsize=13,
            fontweight=600,
            color="#2E3440",
            pad=20
        )
        ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5, color="#D8DEE9")
        ax.tick_params(colors="#4C566A", which="both", labelsize=9)
        ax.spines['polar'].set_color("#E5E9F0")
        ax.spines['polar'].set_linewidth(1)
        for angle, val in zip(angles[:-1], metric_values):
            ax.text(
                angle,
                val,
                f'{val:.2f}',
                ha='center',
                va='center',
                fontsize=9,
                fontweight=600,
                color="#2E3440",
                bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=0.95,
                    edgecolor='#E5E9F0',
                    pad=0.3
                )
            )

    def _plot_box_plot(self, ax, shear_data: pd.DataFrame, metric: str,
                       labels: list, metric_values: list, run_names,
                       shear_value: float):
        """Create a box plot for shear rate specific metrics.

        Args:
            ax: Matplotlib axes object
            shear_data: DataFrame with shear rate data
            metric: The metric key
            labels: List of run labels
            metric_values: List of metric values(fallback)
            run_names: List of run names or None
            shear_value: The shear rate value
        """
        colors = self._get_color_scheme('secondary')
        data_per_run = self._extract_error_data(shear_data, metric, run_names)
        if data_per_run:
            bp = ax.boxplot(
                data_per_run,
                labels=labels,
                patch_artist=True,
                medianprops=dict(color=colors['accent'], linewidth=2),
                boxprops=dict(facecolor=colors['main'], alpha=0.85,
                              edgecolor=colors['edge'], linewidth=1.2),
                whiskerprops=dict(color=colors['edge'], linewidth=1.2),
                capprops=dict(color=colors['edge'], linewidth=1.2),
                flierprops=dict(marker='o', markerfacecolor=colors['main'],
                                markersize=6, markeredgecolor=colors['edge'],
                                alpha=0.6)
            )
            self._setup_axes_labels(
                ax, labels, self.format_metric_name(metric))
            self._apply_standard_styling(ax, metric, f"at {shear_value} s⁻¹")
        else:
            self._plot_bar_chart(ax, metric_values, labels, metric,
                                 color_scheme='secondary', shear_value=shear_value)

    def _plot_violin_plot(self, ax, shear_data: pd.DataFrame, metric: str,
                          labels: list, metric_values: list, run_names,
                          shear_value: float):
        """Create a violin plot for shear rate specific metrics.

        Args:
            ax: Matplotlib axes object
            shear_data: DataFrame with shear rate data
            metric: The metric key
            labels: List of run labels
            metric_values: List of metric values(fallback)
            run_names: List of run names or None
            shear_value: The shear rate value
        """
        colors = self._get_color_scheme('secondary')
        data_per_run = self._extract_error_data(shear_data, metric, run_names)

        if data_per_run and all(len(d) > 0 for d in data_per_run):
            parts = ax.violinplot(
                data_per_run,
                positions=range(len(labels)),
                showmeans=True,
                showmedians=True
            )

            for pc in parts['bodies']:
                pc.set_facecolor(colors['main'])
                pc.set_alpha(0.85)
                pc.set_edgecolor(colors['edge'])
                pc.set_linewidth(1.2)
            for partname in ('cbars', 'cmins', 'cmaxes'):
                if partname in parts:
                    parts[partname].set_edgecolor(colors['edge'])
                    parts[partname].set_linewidth(1.2)

            if 'cmedians' in parts:
                parts['cmedians'].set_edgecolor(colors['accent'])
                parts['cmedians'].set_linewidth(2)

            if 'cmeans' in parts:
                parts['cmeans'].set_edgecolor(colors['light'])
                parts['cmeans'].set_linewidth(2)

            self._setup_axes_labels(
                ax, labels, self.format_metric_name(metric))
            self._apply_standard_styling(ax, metric, f"at {shear_value} s⁻¹")
        else:
            self._plot_bar_chart(ax, metric_values, labels, metric,
                                 color_scheme='secondary', shear_value=shear_value)

    def plot_confidence_intervals(self) -> None:
        """Plot predictions with 95% confidence intervals.

        This method plots the predicted values with shaded 95% confidence intervals
        and overlays the actual observed values as scatter points. It also annotates
        the coverage percentage of the confidence intervals.

        The x-axis represents sample indices, and the y-axis represents viscosity
        in centipoise (cP).

        Raises:
            AttributeError: If required attributes (current_results_df, current_overall_figure)
                are not initialized.
        """
        ax = self.current_overall_figure.add_subplot(111)
        df = self.current_results_df.sort_values('actual')
        x = np.arange(len(df))
        ax.fill_between(x, df['lower_ci'].values, df['upper_ci'].values,
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
        coverage = self.metrics._coverage(df)
        ax.text(0.05, 0.95, f'Coverage: {coverage:.1f}%',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def on_run_selected_for_investigation(self, run_name: str) -> None:
        """Handle the selection of a run for per-run investigation.

        This method is triggered when a user selects a run from the per-run
        evaluation tab. It updates the UI to show formulation information,
        prediction metrics, and the associated plot for the selected run.

        Args:
            run_name (str): The name of the selected run. Must exist in
                `self.run_results`.

        Returns:
            None
        """
        if not run_name or run_name not in self.run_results:
            return

        run_df = self.run_results[run_name]
        run_formulations = self.formulations_by_run.get(run_name, [])

        self.display_formulation_info(run_formulations[0])
        self.display_prediction_info(run_df)
        self.update_run_plot()

    def display_formulation_info(self, formulation: Formulation) -> None:
        """Display detailed information of a single formulation in the UI table.

        Converts a `Formulation` object into a DataFrame and populates the 
        `run_info_table` with feature names, values, and units. Applies formatting 
        for readability, including alternating row colors, font styling, and column widths.

        Args:
            formulation (Formulation): The formulation object whose details 
                will be displayed.

        Returns:
            None
        """
        formulation_df = formulation.to_dataframe(
            encoded=False, training=False)
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
        self.run_info_table.verticalHeader().setVisible(False)
        header_font = self.run_info_table.horizontalHeader().font()
        header_font.setBold(True)
        header_font.setPointSize(10)
        self.run_info_table.horizontalHeader().setFont(header_font)
        self.run_info_table.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
                background-color: #E8F4F8;
                color: #1a1a1a;
                padding: 6px;
                border: 1px solid #B8D4E0;
                font-weight: bold;
            }
        """)
        self.run_info_table.setAlternatingRowColors(True)
        self.run_info_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #D0D0D0;
                background-color: white;
                alternate-background-color: #F5F5F5;
                selection-background-color: #B8D4E0;
            }
            QTableWidget::item {
                padding: 4px;
            }
        """)
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
            if key.lower() == "temperature":
                unit = units_map["Temperature"]
            name_item = QtWidgets.QTableWidgetItem(str(display_name))
            value_item = QtWidgets.QTableWidgetItem(str(value))
            unit_item = QtWidgets.QTableWidgetItem(str(unit))
            name_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            value_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            unit_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            font = name_item.font()
            font.setBold(True)
            name_item.setFont(font)
            for item in (name_item, value_item, unit_item):
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)

            self.run_info_table.setItem(i, 0, name_item)
            self.run_info_table.setItem(i, 1, value_item)
            self.run_info_table.setItem(i, 2, unit_item)
        self.run_info_table.resizeColumnsToContents()
        self.run_info_table.setColumnWidth(
            0, max(180, self.run_info_table.columnWidth(0)))  # Feature
        self.run_info_table.setColumnWidth(
            1, max(120, self.run_info_table.columnWidth(1)))  # Value
        self.run_info_table.setColumnWidth(
            2, max(80, self.run_info_table.columnWidth(2)))   # Units
        for i in range(self.run_info_table.rowCount()):
            self.run_info_table.setRowHeight(i, 28)
        self.run_info_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.run_info_table.setSelectionMode(
            QtWidgets.QAbstractItemView.NoSelection)
        self.run_info_table.setFocusPolicy(QtCore.Qt.NoFocus)

    def display_prediction_info(self, run_df: pd.DataFrame) -> None:
        """Display prediction results for a single run in a table.

        Converts the evaluation results DataFrame for a given run into a nicely
        formatted table (`run_metrics_table`) showing actual vs. predicted viscosities,
        absolute and percentage errors, and shear rates. Applies conditional 
        formatting to highlight errors and sets table styles for readability.

        Args:
            run_df (pd.DataFrame): DataFrame containing prediction results for a 
                specific run. Expected columns: ["shear_rate", "actual", "predicted",
                "abs_error", "percentage_error"].

        Returns:
            None

        Notes:
            - Columns are renamed to user-friendly labels.
            - Cells are non-editable.
            - Conditional coloring is applied based on error magnitude:
                - Green for small errors
                - Yellow for moderate errors
                - Red for large errors
            - Column widths and row heights are adjusted for readability.
        """
        display_columns = ["shear_rate", "actual",
                           "predicted", "abs_error", "percentage_error"]
        df = run_df[display_columns].copy()
        name_mapping = {
            "shear_rate": "Shear Rate (s⁻¹)",
            "actual": "Actual Viscosity (cP)",
            "predicted": "Estimated Viscosity (cP)",
            "abs_error": "Absolute Error",
            "percentage_error": "Percentage Error (%)",
        }
        df.rename(columns=name_mapping, inplace=True)
        self.run_metrics_table.setRowCount(len(df))
        self.run_metrics_table.setColumnCount(len(df.columns))
        self.run_metrics_table.setHorizontalHeaderLabels(df.columns.tolist())
        self.run_metrics_table.verticalHeader().setVisible(False)
        header_font = self.run_metrics_table.horizontalHeader().font()
        header_font.setBold(True)
        header_font.setPointSize(10)
        self.run_metrics_table.horizontalHeader().setFont(header_font)

        self.run_metrics_table.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
                background-color: #E8F4F8;
                color: #1a1a1a;
                padding: 6px;
                border: 1px solid #B8D4E0;
                font-weight: bold;
            }
        """)
        self.run_metrics_table.setAlternatingRowColors(True)
        self.run_metrics_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #D0D0D0;
                background-color: white;
                alternate-background-color: #F5F5F5;
                selection-background-color: #B8D4E0;
            }
            QTableWidget::item {
                padding: 4px;
            }
        """)
        for i, row in df.iterrows():
            for j, value in enumerate(row):
                col_name = df.columns[j]
                if isinstance(value, (float, int)):
                    display_value = f"{value:.2f}"
                else:
                    display_value = str(value)

                item = QtWidgets.QTableWidgetItem(display_value)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                if isinstance(value, (float, int)):
                    item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                if "Error" in col_name and isinstance(value, (float, int)):
                    if "Percentage" in col_name:
                        if value < 5.0:
                            item.setBackground(
                                QtGui.QColor(220, 255, 220))
                        elif value < 15.0:
                            item.setBackground(
                                QtGui.QColor(255, 255, 200))
                        else:
                            item.setBackground(
                                QtGui.QColor(255, 220, 220))
                    elif "Absolute" in col_name:

                        if value < 0.1:
                            item.setBackground(
                                QtGui.QColor(220, 255, 220))
                        elif value < 0.3:
                            item.setBackground(
                                QtGui.QColor(255, 255, 200))
                        else:
                            item.setBackground(
                                QtGui.QColor(255, 220, 220))
                self.run_metrics_table.setItem(i, j, item)
        self.run_metrics_table.resizeColumnsToContents()
        min_widths = {
            0: 120,  # Shear Rate
            1: 140,  # Actual Viscosity
            2: 150,  # Estimated Viscosity
            3: 120,  # Absolute Error
            4: 140,  # Percentage Error
        }

        for col, min_width in min_widths.items():
            if col < self.run_metrics_table.columnCount():
                current_width = self.run_metrics_table.columnWidth(col)
                self.run_metrics_table.setColumnWidth(
                    col, max(min_width, current_width))
        for i in range(self.run_metrics_table.rowCount()):
            self.run_metrics_table.setRowHeight(i, 28)
        self.run_metrics_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.run_metrics_table.setSelectionMode(
            QtWidgets.QAbstractItemView.NoSelection)
        self.run_metrics_table.setFocusPolicy(QtCore.Qt.NoFocus)

    def update_run_plot(self) -> None:
        """Update the visualization for the selected run in the per-run tab.

        Fetches the currently selected run from the run combo box, clears the
        existing figure, and redraws the plot based on the selected visualization type.
        Currently supports "Viscosity Profile Comparison".

        Returns:
            None
        """
        run_name = self.run_combo.currentText()
        if not run_name or run_name not in self.run_results:
            return

        self.run_figure.clear()
        run_df = self.run_results[run_name]

        plot_type = self.run_plot_type.currentText()

        if plot_type == "Viscosity Profile Comparison":
            self.plot_viscosity_profile_comparison(run_df)

        self.run_canvas.draw()

    def plot_viscosity_profile_comparison(self, run_df: pd.DataFrame) -> None:
        """Plot the actual vs predicted viscosity profile for a single run.

        The plot uses a log-log scale for both shear rate and viscosity, and optionally
        includes the 95% confidence interval if `lower_ci` and `upper_ci` columns are present.

        Args:
            run_df (pd.DataFrame): DataFrame containing the run data with the following columns:
                - 'shear_rate': Shear rate values (x-axis)
                - 'actual': Measured viscosity values
                - 'predicted': Predicted viscosity values
                - 'lower_ci' (optional): Lower bound of 95% confidence interval
                - 'upper_ci' (optional): Upper bound of 95% confidence interval

        Behavior:
            - Clears the existing per-run figure before plotting.
            - Plots the actual viscosity as a solid line and predicted as a dashed line.
            - Fills the confidence interval if available.
            - Applies custom colors, line styles, labels, and grid styling.
            - Sets axes labels, title, and legend with consistent styling.
            - Draws the updated figure on the `run_canvas`.

        Returns:
            None
        """
        ax = self.run_figure.add_subplot(111)
        ax.clear()

        color_actual = "#00A3DA"
        color_predicted = "#32E2DF"
        color_ci = "#69EAC5"
        if {"lower_ci", "upper_ci"}.issubset(run_df.columns):
            ax.fill_between(
                run_df["shear_rate"],
                run_df["lower_ci"],
                run_df["upper_ci"],
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

    def export_results_as_zip(self) -> None:
        """Export all evaluation results and plots as a ZIP file.

        This method exports:
        - All dataframes: full results, overall metrics, per-shear metrics, per-run metrics, 
        and individual run dataframes
        - All plots: "Predicted vs Actual" plot and each selected metric with all visualization types
        - Per-run plots: All available per-run plot types for each run
        - Summary report with model information and metrics
        """
        if self.current_results_df is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Results",
                "No evaluation results to export."
            )
            return
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
            with tempfile.TemporaryDirectory() as temp_dir:
                details_file = os.path.join(temp_dir, "full_results.csv")
                self.current_results_df.to_csv(details_file, index=False)
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
                if self.per_shear_radio.isChecked() or self.both_radio.isChecked():
                    per_shear_metrics = self.metrics.compute_per_shear_rate(
                        self.current_results_df,
                        self.selected_metrics
                    )
                    per_shear_file = os.path.join(
                        temp_dir, "per_shear_metrics.csv")
                    per_shear_metrics.to_csv(per_shear_file, index=False)
                if self.run_results:
                    per_run_data = []
                    for run_name, run_df in self.run_results.items():
                        run_metrics = self.metrics.compute_overall(
                            run_df, self.selected_metrics
                        )
                        run_metrics['run'] = run_name
                        per_run_data.append(run_metrics)

                    if per_run_data:
                        pd.DataFrame(per_run_data).to_csv(
                            os.path.join(temp_dir, "per_run_metrics.csv"),
                            index=False
                        )
                for run_name, run_df in self.run_results.items():
                    safe_run_name = run_name.replace(
                        '.', '_').replace('/', '_').replace('\\', '_')
                    run_file = os.path.join(
                        temp_dir, f"run_{safe_run_name}_results.csv")
                    run_df.to_csv(run_file, index=False)

                current_plot_type = self.plot_type_combo.currentText()
                current_viz_type = self.viz_type_combo.currentText()
                current_shear_index = self.current_shear_index
                plots_dir = os.path.join(temp_dir, "overall_plots")
                os.makedirs(plots_dir, exist_ok=True)
                all_plot_types = []
                for i in range(self.plot_type_combo.count()):
                    all_plot_types.append(self.plot_type_combo.itemText(i))

                viz_types = [
                    "Bar Chart",
                    "Line Plot",
                    "Box Plot",
                    "Violin Plot",
                    "Heatmap",
                    "Radar Chart"
                ]

                if "Predicted vs Actual" in all_plot_types:
                    self.plot_type_combo.setCurrentText(
                        "Predicted vs Actual")
                    QtWidgets.QApplication.processEvents()

                    # Export for all shear rates
                    self.current_shear_index = 0
                    self.update_shear_navigation()
                    self.update_plot()
                    QtWidgets.QApplication.processEvents()

                    plot_file = os.path.join(
                        plots_dir, "predicted_vs_actual_all_shear_rates.png")
                    self.current_overall_figure.savefig(
                        plot_file, dpi=300, bbox_inches='tight')
                    for idx, shear_name in enumerate(self.selected_shear_rates, start=1):
                        self.current_shear_index = idx
                        self.update_shear_navigation()
                        self.update_plot()
                        QtWidgets.QApplication.processEvents()

                        safe_shear_name = shear_name.replace(
                            ' ', '_').lower()
                        plot_file = os.path.join(
                            plots_dir, f"predicted_vs_actual_{safe_shear_name}.png")
                        self.current_overall_figure.savefig(
                            plot_file, dpi=300, bbox_inches='tight')
                for plot_type in all_plot_types:
                    if plot_type == "Predicted vs Actual" or plot_type == "":
                        continue
                    self.plot_type_combo.setCurrentText(plot_type)
                    QtWidgets.QApplication.processEvents()
                    metric_name = plot_type.replace(' ', '_').lower()

                    for viz_type in viz_types:
                        self.viz_type_combo.setCurrentText(viz_type)
                        QtWidgets.QApplication.processEvents()
                        self.current_shear_index = 0
                        self.update_shear_navigation()
                        self.update_plot()
                        QtWidgets.QApplication.processEvents()

                        viz_name = viz_type.replace(' ', '_').lower()
                        plot_file = os.path.join(
                            plots_dir,
                            f"{metric_name}_{viz_name}_all_shear_rates.png"
                        )
                        self.current_overall_figure.savefig(
                            plot_file, dpi=300, bbox_inches='tight')
                        for idx, shear_name in enumerate(self.selected_shear_rates, start=1):
                            self.current_shear_index = idx
                            self.update_shear_navigation()
                            self.update_plot()
                            QtWidgets.QApplication.processEvents()

                            safe_shear_name = shear_name.replace(
                                ' ', '_').lower()
                            plot_file = os.path.join(
                                plots_dir,
                                f"{metric_name}_{viz_name}_{safe_shear_name}.png"
                            )
                            self.current_overall_figure.savefig(
                                plot_file, dpi=300, bbox_inches='tight')
                self.plot_type_combo.setCurrentText(current_plot_type)
                self.viz_type_combo.setCurrentText(current_viz_type)
                self.current_shear_index = current_shear_index
                self.update_shear_navigation()
                self.update_plot()
                if self.run_results:
                    run_plots_dir = os.path.join(temp_dir, "per_run_plots")
                    os.makedirs(run_plots_dir, exist_ok=True)
                    current_run = self.run_combo.currentText()
                    current_run_plot_type = self.run_plot_type.currentText()
                    run_plot_types = []
                    for i in range(self.run_plot_type.count()):
                        run_plot_types.append(
                            self.run_plot_type.itemText(i))
                    for run_name in self.run_results.keys():
                        self.run_combo.setCurrentText(run_name)
                        QtWidgets.QApplication.processEvents()

                        safe_run_name = run_name.replace(
                            '.', '_').replace('/', '_').replace('\\', '_')

                        for plot_type in run_plot_types:
                            self.run_plot_type.setCurrentText(plot_type)
                            QtWidgets.QApplication.processEvents()

                            safe_plot_type = plot_type.replace(
                                ' ', '_').lower()
                            plot_file = os.path.join(
                                run_plots_dir,
                                f"{safe_run_name}_{safe_plot_type}.png"
                            )
                            self.run_figure.savefig(
                                plot_file, dpi=300, bbox_inches='tight')
                    self.run_combo.setCurrentText(current_run)
                    self.run_plot_type.setCurrentText(
                        current_run_plot_type)
                    QtWidgets.QApplication.processEvents()
                summary_file = os.path.join(
                    temp_dir, "evaluation_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write("=" * 70 + "\n")
                    f.write("VISQAI MODEL EVALUATION REPORT\n")
                    f.write("=" * 70 + "\n\n")

                    f.write(f"Model: {self.select_model_label.text()}\n")
                    f.write(
                        f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(
                        f"Total Formulations: {len(self.selected_formulations)}\n")
                    f.write(
                        f"Total Predictions: {len(self.current_results_df)}\n")
                    f.write(
                        f"Runs Evaluated: {len(self.formulations_by_run)}\n")
                    f.write(
                        f"Metrics Evaluated: {', '.join([self.format_metric_name(m) for m in self.selected_metrics])}\n")
                    f.write(
                        f"Shear Rates: {', '.join([f'{self.shear_rates[sr]} 1/s' for sr in self.selected_shear_rates])}\n\n")

                    f.write("OVERALL METRICS:\n")
                    f.write("-" * 70 + "\n")
                    overall_metrics = self.metrics.compute_overall(
                        self.current_results_df,
                        self.selected_metrics
                    )
                    for metric, value in overall_metrics.items():
                        f.write(
                            f"{self.format_metric_name(metric)}: {value:.4f}\n")
                        f.write(
                            f"  {self.METRIC_DESCRIPTIONS.get(metric, '')}\n\n")

                    if self.per_shear_radio.isChecked() or self.both_radio.isChecked():
                        f.write("\nPER-SHEAR RATE METRICS:\n")
                        f.write("-" * 70 + "\n")
                        per_shear_metrics = self.metrics.compute_per_shear_rate(
                            self.current_results_df,
                            self.selected_metrics
                        )
                        for _, row in per_shear_metrics.iterrows():
                            f.write(
                                f"\nShear Rate: {row['shear_rate']:.0f} 1/s\n")
                            for metric in self.selected_metrics:
                                if metric in row:
                                    f.write(
                                        f"  {self.format_metric_name(metric)}: {row[metric]:.4f}\n")

                    f.write("\n\nPER-RUN SUMMARY:\n")
                    f.write("-" * 70 + "\n")
                    for run_name, run_df in self.run_results.items():
                        f.write(f"\n{run_name}:\n")
                        f.write(
                            f"  Number of predictions: {len(run_df)}\n")
                        run_metrics = self.metrics.compute_overall(
                            run_df, self.selected_metrics
                        )
                        for metric, value in run_metrics.items():
                            f.write(
                                f"  {self.format_metric_name(metric)}: {value:.4f}\n")

                    f.write("\n" + "=" * 70 + "\n")
                    f.write("END OF REPORT\n")
                    f.write("=" * 70 + "\n")
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, temp_dir)
                            zipf.write(file_path, arcname)

            num_csv_files = 1
            if self.overall_radio.isChecked() or self.both_radio.isChecked():
                num_csv_files += 1
            if self.per_shear_radio.isChecked() or self.both_radio.isChecked():
                num_csv_files += 1
            if self.run_results:
                num_csv_files += 1
                num_csv_files += len(self.run_results)
            num_overall_plots = 0
            if "Predicted vs Actual" in all_plot_types:
                num_overall_plots += 1 + len(self.selected_shear_rates)
            num_metric_plots = len(all_plot_types) - 1
            if num_metric_plots > 0:
                num_overall_plots += num_metric_plots * \
                    len(viz_types) * (1 + len(self.selected_shear_rates))
            num_run_plots = len(self.run_results) * \
                len(run_plot_types) if self.run_results else 0

            QtWidgets.QMessageBox.information(
                self,
                "Export Complete",
                f"Results exported successfully to:\n{zip_path}\n\n"
                f"Exported:\n"
                f"• {num_csv_files} CSV files\n"
                f"• {num_overall_plots} overall plots\n"
                f"• {num_run_plots} per-run plots\n"
                f"• 1 summary report"
            )

        except Exception as e:
            Log.e(TAG, f"ZIP export failed: {e}")
            Log.e(TAG, traceback.format_exc())
            QtWidgets.QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export results: {str(e)}"
            )

        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def save_current_figure_per_run(self) -> None:
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

    def save_current_figure_overall(self) -> None:
        if not hasattr(self, 'evaluation_results') or self.evaluation_results is None:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "No plot data available to save."
            )
            return
        if self.current_shear_index == 0:
            default_name = "plot_all_shear_rates.png"
        else:
            shear_name = list(self.selected_shear_rates)[
                self.current_shear_index - 1]
            default_name = f"plot_{shear_name.replace(' ', '_').lower()}.png"
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
