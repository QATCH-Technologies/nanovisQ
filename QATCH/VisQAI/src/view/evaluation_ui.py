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
    1.0
"""

import os
import copy
from typing import Optional, List, Dict, TYPE_CHECKING
from pathlib import Path

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
    """

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

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface components."""
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)

        # Create splitter for horizontal layout
        splitter = QtWidgets.QSplitter(Qt.Horizontal)

        # Left panel - Configuration
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        # Model selection group (similar to frame_step1.py and frame_step2.py)
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

        # Data selection group (similar to frame_step1.py step 3)
        data_group = QtWidgets.QGroupBox("Import Experiments")
        data_layout = QtWidgets.QVBoxLayout(data_group)

        # Browse run button and list (same as frame_step1.py)
        form_layout = QtWidgets.QFormLayout()

        self.select_run = QtWidgets.QPushButton("Add Run...")
        self.select_label = QtWidgets.QLineEdit()
        self.select_label.setPlaceholderText("No run selected")
        self.select_label.setReadOnly(True)

        # List view for runs (same as frame_step1.py)
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
        self.formulation_info.setMaximumHeight(60)
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
        self.metrics_list.setMaximumHeight(150)

        # Add all available metrics
        for metric in self.available_metrics:
            item = QtWidgets.QListWidgetItem(self.format_metric_name(metric))
            item.setData(Qt.UserRole, metric)
            self.metrics_list.addItem(item)
            # Select default metrics
            if metric in self.selected_metrics:
                item.setSelected(True)

        self.metrics_list.itemSelectionChanged.connect(
            self.on_metrics_selection_changed)
        settings_layout.addWidget(self.metrics_list)

        # Shear rate selection
        shear_label = QtWidgets.QLabel("Select Shear Rates:")
        settings_layout.addWidget(shear_label)

        self.shear_rate_list = QtWidgets.QListWidget()
        self.shear_rate_list.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection)
        self.shear_rate_list.setMaximumHeight(120)

        for shear_name, shear_value in self.shear_rates.items():
            item = QtWidgets.QListWidgetItem(f"{shear_value:.0e} s⁻¹")
            item.setData(Qt.UserRole, shear_name)
            self.shear_rate_list.addItem(item)
            item.setSelected(True)  # Select all by default

        self.shear_rate_list.itemSelectionChanged.connect(
            self.on_shear_selection_changed)
        settings_layout.addWidget(self.shear_rate_list)

        # Evaluation mode
        mode_label = QtWidgets.QLabel("Evaluation Mode:")
        settings_layout.addWidget(mode_label)

        mode_layout = QtWidgets.QHBoxLayout()
        self.overall_radio = QtWidgets.QRadioButton("Overall")
        self.per_shear_radio = QtWidgets.QRadioButton("Per Shear Rate")
        self.both_radio = QtWidgets.QRadioButton("Both")
        self.both_radio.setChecked(True)

        mode_layout.addWidget(self.overall_radio)
        mode_layout.addWidget(self.per_shear_radio)
        mode_layout.addWidget(self.both_radio)
        settings_layout.addLayout(mode_layout)

        left_layout.addWidget(settings_group)

        # Control buttons
        control_layout = QtWidgets.QHBoxLayout()

        self.evaluate_btn = QtWidgets.QPushButton("Evaluate")
        self.evaluate_btn.clicked.connect(self.run_evaluation)
        self.evaluate_btn.setEnabled(False)
        self.evaluate_btn.setStyleSheet("""
            QPushButton:enabled {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
        """)

        self.export_btn = QtWidgets.QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)

        control_layout.addWidget(self.evaluate_btn)
        control_layout.addWidget(self.export_btn)

        left_layout.addLayout(control_layout)
        left_layout.addStretch()

        # Right panel - Results
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        # Results tabs
        self.results_tabs = QtWidgets.QTabWidget()

        # Overall metrics tab
        overall_widget = QtWidgets.QWidget()
        overall_layout = QtWidgets.QVBoxLayout(overall_widget)

        self.overall_table = QtWidgets.QTableWidget()
        self.overall_table.setSortingEnabled(True)
        overall_layout.addWidget(self.overall_table)

        self.results_tabs.addTab(overall_widget, "Overall Metrics")

        # Per shear rate metrics tab
        per_shear_widget = QtWidgets.QWidget()
        per_shear_layout = QtWidgets.QVBoxLayout(per_shear_widget)

        self.per_shear_table = QtWidgets.QTableWidget()
        self.per_shear_table.setSortingEnabled(True)
        per_shear_layout.addWidget(self.per_shear_table)

        self.results_tabs.addTab(per_shear_widget, "Per Shear Rate Metrics")

        # Plots tab
        plots_widget = QtWidgets.QWidget()
        plots_layout = QtWidgets.QVBoxLayout(plots_widget)

        # Plot controls
        plot_controls = QtWidgets.QHBoxLayout()

        plot_type_label = QtWidgets.QLabel("Plot Type:")
        plot_controls.addWidget(plot_type_label)

        self.plot_type_combo = QtWidgets.QComboBox()
        self.plot_type_combo.addItems([
            "Predicted vs Actual",
            "Residuals",
            "Residuals Distribution",
            "Q-Q Plot",
            "Metrics Comparison",
            "Shear Rate Performance",
            "Error vs Shear Rate",
            "Confidence Intervals"
        ])
        self.plot_type_combo.currentTextChanged.connect(self.update_plot)
        plot_controls.addWidget(self.plot_type_combo)

        plot_controls.addStretch()
        plots_layout.addLayout(plot_controls)

        # Plot canvas
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        plots_layout.addWidget(self.canvas)

        self.results_tabs.addTab(plots_widget, "Plots")

        # Detailed results tab
        details_widget = QtWidgets.QWidget()
        details_layout = QtWidgets.QVBoxLayout(details_widget)

        self.details_table = QtWidgets.QTableWidget()
        self.details_table.setSortingEnabled(True)
        details_layout.addWidget(self.details_table)

        self.results_tabs.addTab(details_widget, "Detailed Results")

        right_layout.addWidget(self.results_tabs)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        right_layout.addWidget(self.status_label)

        # Add panels to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 600])

        main_layout.addWidget(splitter)
        # Initialize file dialogs (same as frame_step1.py)
        self.init_file_dialogs()
        # Connect signals
        self.select_model_btn.clicked.connect(self.model_dialog.show)
        global_handler = getattr(self.parent, 'set_global_model_path', None)
        self.model_dialog.fileSelected.connect(
            global_handler if callable(global_handler) else self.model_selected)

        self.select_run.clicked.connect(self.file_dialog_show)
        self.file_dialog.fileSelected.connect(self.file_selected)

    def init_file_dialogs(self):
        """Initialize file dialogs for model and run selection (same as frame_step1.py)."""
        # Browse model dialog
        self.model_dialog = QtWidgets.QFileDialog()
        self.model_dialog.setOption(
            QtWidgets.QFileDialog.DontUseNativeDialog, True)
        model_path = os.path.join(
            Architecture.get_path(), "QATCH/VisQAI/assets")
        if os.path.exists(model_path):
            self.model_dialog.setDirectory(model_path)
        else:
            self.model_dialog.setDirectory(Constants.log_prefer_path)
        self.model_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self.model_dialog.setNameFilter("VisQ.AI Models (VisQAI-*.zip)")
        self.model_dialog.selectNameFilter("VisQ.AI Models (VisQAI-*.zip)")

        # Check for default model
        predictor_path = os.path.join(model_path, "VisQAI-base.zip")
        if os.path.exists(predictor_path):
            self.model_selected(path=predictor_path)

        # Browse run dialog
        self.file_dialog = QtWidgets.QFileDialog()
        self.file_dialog.setOption(
            QtWidgets.QFileDialog.DontUseNativeDialog, True)
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self.file_dialog.setNameFilter("Captured Runs (capture.zip)")
        self.file_dialog.selectNameFilter("Captured Runs (capture.zip)")

    def list_view_addPlaceholderText(self):
        """Add placeholder text to list view when empty."""
        if self.model.rowCount() == 0:
            no_item = QtGui.QStandardItem("No experiments selected")
            no_item.setEnabled(False)
            no_item.setSelectable(False)
            self.model.appendRow(no_item)

    def file_dialog_show(self):
        """Show file dialog for run selection (same logic as frame_step1.py)."""
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

        self.file_dialog.show()

    def file_selected(self, path: str | None):
        """Handle file selection (similar to frame_step1.py)."""
        if path is None:
            self.select_label.clear()
            self.list_view.clearSelection()
            return

        # Check if file is in working directory
        prefer_abs = os.path.abspath(Constants.log_prefer_path)
        path_abs = os.path.abspath(path)
        try:
            inside = os.path.commonpath([prefer_abs, path_abs]) == prefer_abs
        except ValueError:
            inside = False

        if not inside:
            Log.e("EvaluationUI", "Selected run is not in working directory")
            QtWidgets.QMessageBox.warning(
                self,
                Constants.app_title,
                "The selected run is not in your working directory and cannot be used."
            )
            return

        # Get CSV file from zip if needed
        self.run_file_run = path
        namelist = SecureOpen.get_namelist(self.run_file_run)
        for file in namelist:
            if file.endswith(".csv"):
                self.run_file_run = os.path.join(
                    os.path.dirname(self.run_file_run), file)
                break

        self.select_label.setText(os.path.basename(
            os.path.dirname(self.run_file_run)))

        # Add to list view
        item = QtGui.QStandardItem(self.select_label.text())
        found = self.model.findItems(item.text())

        if len(found) == 0:
            if len(self.all_files) == 0:
                self.model.removeRow(0)  # Remove placeholder
            self.model.appendRow(item)
            new_index = self.model.indexFromItem(item)
            self.list_view.setCurrentIndex(new_index)
            self.all_files[item.text()] = path

            # Update formulation info
            self.update_formulation_info()

    def user_run_clicked(self):
        """Handle click on run in list view."""
        try:
            selected = self.list_view.selectedIndexes()
            if selected:
                run_name = self.model.itemFromIndex(selected[0]).text()
                if run_name in self.all_files:
                    # Could load run details here if needed
                    pass
        except IndexError:
            pass

    def user_run_removed(self):
        """Remove selected run from list."""
        try:
            selected = self.list_view.selectedIndexes()
            if len(selected) == 0:
                return
            file_name = self.model.itemFromIndex(selected[0]).text()
            self.all_files.pop(file_name, None)
            self.model.removeRow(selected[0].row())
            self.list_view_addPlaceholderText()
            self.select_label.clear()
            self.update_formulation_info()
        except IndexError:
            pass

    def user_all_runs_removed(self):
        """Remove all runs from list."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Remove All",
            "Are you sure you want to remove all runs?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            try:
                self.all_files.clear()
                self.model.clear()
                self.list_view_addPlaceholderText()
                self.select_label.clear()
                self.update_formulation_info()
            except Exception as e:
                Log.e("EvaluationUI", f"Failed to remove all runs: {e}")

    def update_formulation_info(self):
        """Update formulation info display."""
        num_runs = len(self.all_files)
        if num_runs == 0:
            self.formulation_info.setText("No runs selected")
        else:
            self.formulation_info.setText(f"Runs selected: {num_runs}")
            # Load formulations from parent if available
            if hasattr(self.parent, 'import_formulations'):
                num_forms = len(self.parent.import_formulations)
                self.formulation_info.append(
                    f"Formulations available: {num_forms}")

    def model_selected(self, path: Optional[str] = None):
        """Handle model selection (same as frame_step1.py)."""
        self.model_path = path

        if path is None:
            self.select_model_label.clear()
            self.predictor = None
            self.check_ready_to_evaluate()
            return

        self.select_model_label.setText(
            path.split('\\')[-1].split('/')[-1].split('.')[0])

        # Load predictor
        try:
            self.predictor = Predictor(self.model_path)
            self.status_label.setText("Model loaded successfully")
            self.check_ready_to_evaluate()
        except Exception as e:
            Log.e("EvaluationUI", f"Failed to load model: {e}")
            self.status_label.setText("Failed to load model")
            QtWidgets.QMessageBox.warning(
                self,
                Constants.app_title,
                f"Failed to load model: {str(e)}"
            )

    def format_metric_name(self, metric: str) -> str:
        """Format metric name for display.

        Args:
            metric: Raw metric name.

        Returns:
            Formatted metric name.
        """
        names = {
            'mae': 'Mean Absolute Error (MAE)',
            'rmse': 'Root Mean Squared Error (RMSE)',
            'mse': 'Mean Squared Error (MSE)',
            'mape': 'Mean Absolute Percentage Error (MAPE)',
            'median_ae': 'Median Absolute Error',
            'r2': 'R-squared (R²)',
            'coverage': 'Confidence Interval Coverage (%)',
            'mean_cv': 'Mean Coefficient of Variation',
            'median_cv': 'Median Coefficient of Variation',
            'max_error': 'Maximum Error',
            'std_error': 'Standard Deviation of Error',
            'mean_std': 'Mean Standard Deviation',
            'count': 'Number of Observations'
        }
        return names.get(metric, metric.replace('_', ' ').title())

    def on_tab_selected(self):
        """Called when this tab is selected."""
        Log.i("EvaluationUI", "Evaluate tab selected")
        self.refresh_data_from_parent()

        # Check if model selected elsewhere
        if not self.model_path:
            # Try to get model from other tabs (same as frame_step1.py)
            if hasattr(self.parent, 'tab_widget'):
                all_model_paths = []
                for i in range(self.parent.tab_widget.count()):
                    widget = self.parent.tab_widget.widget(i)
                    if hasattr(widget, 'model_path'):
                        all_model_paths.append(widget.model_path)

                found_model_path = next(
                    (x for x in all_model_paths if x is not None), None)
                if found_model_path:
                    self.model_selected(found_model_path)

    def refresh_data_from_parent(self):
        """Refresh data from parent window."""
        # Check for imported formulations
        if hasattr(self.parent, 'import_formulations') and self.parent.import_formulations:
            self.selected_formulations = copy.copy(
                self.parent.import_formulations)

            # Update run list if we have run names
            if hasattr(self.parent, 'import_run_names') and self.parent.import_run_names:
                # Clear and repopulate list
                self.model.clear()
                self.all_files.clear()

                for run_name in self.parent.import_run_names:
                    item = QtGui.QStandardItem(run_name)
                    self.model.appendRow(item)
                    # Store dummy path for now (actual path would come from parent)
                    self.all_files[run_name] = f"run_{run_name}"

                if self.model.rowCount() == 0:
                    self.list_view_addPlaceholderText()

            self.update_formulation_info()
            self.check_ready_to_evaluate()

    def check_ready_to_evaluate(self):
        """Check if we have everything needed to run evaluation."""
        ready = (
            self.predictor is not None and
            len(self.all_files) > 0 and
            len(self.selected_metrics) > 0 and
            len(self.selected_shear_rates) > 0
        )
        self.evaluate_btn.setEnabled(ready)

        if not self.predictor:
            self.status_label.setText("No model loaded")
        elif len(self.all_files) == 0:
            self.status_label.setText("No runs selected")
        elif ready:
            self.status_label.setText("Ready to evaluate")

    def on_metrics_selection_changed(self):
        """Handle metric selection changes."""
        selected_items = self.metrics_list.selectedItems()
        self.selected_metrics = [
            item.data(Qt.UserRole) for item in selected_items]
        self.check_ready_to_evaluate()

    def on_shear_selection_changed(self):
        """Handle shear rate selection changes."""
        selected_items = self.shear_rate_list.selectedItems()
        self.selected_shear_rates = [
            item.data(Qt.UserRole) for item in selected_items]
        self.check_ready_to_evaluate()

    def make_formulations(self):
        forms: List[Formulation] = []
        for k, v in self.all_files.items():
            path = Path(v)
            base_path = path.parent
            xml_file = next(base_path.glob(f"*{k}*.xml"), None)
            form_parser = Parser(xml_path=xml_file)
            f = form_parser.get_formulation()
            forms.append(f)
        for f in forms:
            Log.i(f.to_dataframe())

    def run_evaluation(self):
        """Execute the evaluation process."""
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("Running evaluation...")
            self.evaluate_btn.setEnabled(False)

            # Make sure we have predictor
            if self.predictor is None:
                raise ValueError("No model loaded")
            self.make_formulations()
            # Get formulations from parent or prepare from runs
            if len(self.selected_formulations) == 0:
                # Try to get from parent again
                if hasattr(self.parent, 'import_formulations'):
                    self.selected_formulations = copy.copy(
                        self.parent.import_formulations)

                if len(self.selected_formulations) == 0:
                    raise ValueError(
                        "No formulations available for evaluation")

            # Prepare data for evaluation
            formulations_df = self.prepare_formulations_dataframe()

            self.progress_bar.setValue(30)

            # Run evaluation
            self.current_results_df = self.predictor.evaluate(formulations_df)

            self.progress_bar.setValue(60)

            # Compute metrics
            if self.overall_radio.isChecked() or self.both_radio.isChecked():
                overall_metrics = self.metrics_calculator.compute_overall(
                    self.current_results_df,
                    self.selected_metrics
                )
                self.display_overall_metrics(overall_metrics)

            self.progress_bar.setValue(80)

            if self.per_shear_radio.isChecked() or self.both_radio.isChecked():
                per_shear_metrics = self.metrics_calculator.compute_per_shear_rate(
                    self.current_results_df,
                    self.selected_metrics
                )
                self.display_per_shear_metrics(per_shear_metrics)

            # Display detailed results
            self.display_detailed_results()

            # Update plot
            self.update_plot()

            self.progress_bar.setValue(100)
            self.status_label.setText("Evaluation completed successfully")
            self.export_btn.setEnabled(True)

        except Exception as e:
            Log.e("EvaluationUI", f"Evaluation failed: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                Constants.app_title,
                f"Failed to run evaluation: {str(e)}"
            )
            self.status_label.setText("Evaluation failed")

        finally:
            self.progress_bar.setVisible(False)
            self.evaluate_btn.setEnabled(True)

    def prepare_formulations_dataframe(self) -> pd.DataFrame:
        """Prepare formulations data for evaluation.

        Returns:
            DataFrame with formulation data and actual viscosity values.
        """
        data = []

        for formulation in self.selected_formulations:
            row = {
                'Run_Name': formulation.run_name if hasattr(formulation, 'run_name') else 'Unknown',
                'Protein_type': formulation.protein.name if formulation.protein else 'none',
                'MW': formulation.protein.mw if formulation.protein else 0,
                'Protein_conc': formulation.protein.concentration if formulation.protein else 0,
                'Temperature': formulation.temperature,
                'Buffer_type': formulation.buffer.name if formulation.buffer else 'none',
                'Buffer_pH': formulation.buffer.ph if formulation.buffer else 7.0,
                'Buffer_conc': formulation.buffer.concentration if formulation.buffer else 0,
                'Salt_type': formulation.salt.name if formulation.salt else 'none',
                'Salt_conc': formulation.salt.concentration if formulation.salt else 0,
                'Stabilizer_type': formulation.stabilizer.name if formulation.stabilizer else 'none',
                'Stabilizer_conc': formulation.stabilizer.concentration if formulation.stabilizer else 0,
                'Surfactant_type': formulation.surfactant.name if formulation.surfactant else 'none',
                'Surfactant_conc': formulation.surfactant.concentration if formulation.surfactant else 0,
                'Protein_class_type': formulation.protein.protein_class if formulation.protein and hasattr(formulation.protein, 'protein_class') else 'mAb'
            }

            # Add actual viscosity values for selected shear rates
            for shear_name in self.selected_shear_rates:
                if hasattr(formulation, 'viscosity_profile'):
                    # Get actual viscosity value at this shear rate
                    shear_value = self.shear_rates[shear_name]
                    actual_visc = formulation.viscosity_profile.get_viscosity_at_shear(
                        shear_value)
                    row[shear_name] = actual_visc if actual_visc else np.nan
                else:
                    row[shear_name] = np.nan

            data.append(row)

        return pd.DataFrame(data)

    def display_overall_metrics(self, metrics: Dict[str, float]):
        """Display overall metrics in the table.

        Args:
            metrics: Dictionary of metric names and values.
        """
        self.overall_table.setRowCount(len(metrics))
        self.overall_table.setColumnCount(2)
        self.overall_table.setHorizontalHeaderLabels(['Metric', 'Value'])

        for i, (metric_name, value) in enumerate(metrics.items()):
            # Metric name
            name_item = QtWidgets.QTableWidgetItem(
                self.format_metric_name(metric_name))
            self.overall_table.setItem(i, 0, name_item)

            # Metric value
            if metric_name in ['coverage', 'mape']:
                value_str = f"{value:.2f}%"
            elif metric_name == 'count':
                value_str = f"{int(value)}"
            else:
                value_str = f"{value:.4f}"

            value_item = QtWidgets.QTableWidgetItem(value_str)
            value_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.overall_table.setItem(i, 1, value_item)

        self.overall_table.resizeColumnsToContents()

    def display_per_shear_metrics(self, metrics_df: pd.DataFrame):
        """Display per-shear-rate metrics in the table.

        Args:
            metrics_df: DataFrame with per-shear-rate metrics.
        """
        rows = len(metrics_df)
        cols = len(metrics_df.columns)

        self.per_shear_table.setRowCount(rows)
        self.per_shear_table.setColumnCount(cols)

        # Set headers
        headers = []
        for col in metrics_df.columns:
            if col == 'shear_rate':
                headers.append('Shear Rate (s⁻¹)')
            else:
                headers.append(self.format_metric_name(col))
        self.per_shear_table.setHorizontalHeaderLabels(headers)

        # Populate table
        for i in range(rows):
            for j, col in enumerate(metrics_df.columns):
                value = metrics_df.iloc[i, j]

                if col == 'shear_rate':
                    value_str = f"{value:.0e}"
                elif col in ['coverage', 'mape']:
                    value_str = f"{value:.2f}%"
                elif col == 'count':
                    value_str = f"{int(value)}"
                else:
                    value_str = f"{value:.4f}"

                item = QtWidgets.QTableWidgetItem(value_str)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.per_shear_table.setItem(i, j, item)

        self.per_shear_table.resizeColumnsToContents()

    def display_detailed_results(self):
        """Display detailed prediction results."""
        if self.current_results_df is None:
            return

        df = self.current_results_df
        rows = len(df)
        cols = len(df.columns)

        self.details_table.setRowCount(rows)
        self.details_table.setColumnCount(cols)

        # Set headers
        headers = [col.replace('_', ' ').title() for col in df.columns]
        self.details_table.setHorizontalHeaderLabels(headers)

        # Populate table
        for i in range(rows):
            for j, col in enumerate(df.columns):
                value = df.iloc[i, j]

                if isinstance(value, bool):
                    value_str = "Yes" if value else "No"
                elif isinstance(value, (int, float)):
                    if col in ['shear_rate']:
                        value_str = f"{value:.0e}"
                    elif col in ['pct_error']:
                        value_str = f"{value:.2f}%"
                    else:
                        value_str = f"{value:.4f}"
                else:
                    value_str = str(value)

                item = QtWidgets.QTableWidgetItem(value_str)

                # Color coding for within_ci column
                if col == 'within_ci':
                    if value:
                        item.setBackground(QtGui.QColor(
                            200, 255, 200))  # Light green
                    else:
                        item.setBackground(QtGui.QColor(
                            255, 200, 200))  # Light red

                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.details_table.setItem(i, j, item)

        self.details_table.resizeColumnsToContents()

    def update_plot(self):
        """Update the plot based on selected plot type."""
        if self.current_results_df is None:
            return

        self.figure.clear()
        plot_type = self.plot_type_combo.currentText()

        if plot_type == "Predicted vs Actual":
            self.plot_predicted_vs_actual()
        elif plot_type == "Residuals":
            self.plot_residuals()
        elif plot_type == "Residuals Distribution":
            self.plot_residuals_distribution()
        elif plot_type == "Q-Q Plot":
            self.plot_qq()
        elif plot_type == "Metrics Comparison":
            self.plot_metrics_comparison()
        elif plot_type == "Shear Rate Performance":
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

        # Color by shear rate
        shear_rates = df['shear_rate'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(shear_rates)))

        for shear, color in zip(shear_rates, colors):
            mask = df['shear_rate'] == shear
            ax.scatter(df[mask]['actual'], df[mask]['predicted'],
                       alpha=0.6, c=[color], label=f'{shear:.0e} s⁻¹',
                       edgecolors='black', linewidth=0.5)

        # Add perfect prediction line
        min_val = min(df['actual'].min(), df['predicted'].min())
        max_val = max(df['actual'].max(), df['predicted'].max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', alpha=0.5, label='Perfect Prediction')

        ax.set_xlabel('Actual Viscosity (cP)', fontsize=12)
        ax.set_ylabel('Predicted Viscosity (cP)', fontsize=12)
        ax.set_title('Predicted vs Actual Viscosity',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add R² annotation
        r2 = self.metrics_calculator._r2(df)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def plot_residuals(self):
        """Create residuals plot."""
        ax = self.figure.add_subplot(111)
        df = self.current_results_df

        # Color by shear rate
        shear_rates = df['shear_rate'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(shear_rates)))

        for shear, color in zip(shear_rates, colors):
            mask = df['shear_rate'] == shear
            ax.scatter(df[mask]['predicted'], df[mask]['residual'],
                       alpha=0.6, c=[color], label=f'{shear:.0e} s⁻¹',
                       edgecolors='black', linewidth=0.5)

        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Predicted Viscosity (cP)', fontsize=12)
        ax.set_ylabel('Residuals (cP)', fontsize=12)
        ax.set_title('Residuals vs Predicted Values',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    def plot_residuals_distribution(self):
        """Create histogram of residuals."""
        ax = self.figure.add_subplot(111)
        df = self.current_results_df

        ax.hist(df['residual'], bins=30, edgecolor='black', alpha=0.7)
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

    def plot_metrics_comparison(self):
        """Create bar chart comparing metrics."""
        ax = self.figure.add_subplot(111)

        # Get overall metrics
        overall_metrics = self.metrics_calculator.compute_overall(
            self.current_results_df,
            ['mae', 'rmse', 'mape']
        )

        metrics_names = list(overall_metrics.keys())
        metrics_values = list(overall_metrics.values())

        x = np.arange(len(metrics_names))
        bars = ax.bar(x, metrics_values, color=[
                      '#1f77b4', '#ff7f0e', '#2ca02c'])

        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Overall Performance Metrics',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([self.format_metric_name(m) for m in metrics_names])

        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')

        ax.grid(True, alpha=0.3, axis='y')

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

        bp = ax.boxplot(errors_by_shear, labels=[f'{s:.0e}' for s in shear_rates],
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

    def export_results(self):
        """Export evaluation results to files."""
        if self.current_results_df is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Results",
                "No evaluation results to export."
            )
            return

        # Get save directory
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory"
        )

        if not dir_path:
            return

        try:
            # Export detailed results
            details_file = os.path.join(
                dir_path, "evaluation_detailed_results.csv")
            self.current_results_df.to_csv(details_file, index=False)

            # Export overall metrics
            if self.overall_radio.isChecked() or self.both_radio.isChecked():
                overall_metrics = self.metrics_calculator.compute_overall(
                    self.current_results_df,
                    self.selected_metrics
                )
                overall_df = pd.DataFrame(list(overall_metrics.items()),
                                          columns=['Metric', 'Value'])
                overall_file = os.path.join(
                    dir_path, "evaluation_overall_metrics.csv")
                overall_df.to_csv(overall_file, index=False)

            # Export per-shear metrics
            if self.per_shear_radio.isChecked() or self.both_radio.isChecked():
                per_shear_metrics = self.metrics_calculator.compute_per_shear_rate(
                    self.current_results_df,
                    self.selected_metrics
                )
                per_shear_file = os.path.join(
                    dir_path, "evaluation_per_shear_metrics.csv")
                per_shear_metrics.to_csv(per_shear_file, index=False)

            # Export current plot
            plot_file = os.path.join(
                dir_path, f"evaluation_{self.plot_type_combo.currentText().replace(' ', '_').lower()}.png")
            self.figure.savefig(plot_file, dpi=300, bbox_inches='tight')

            QtWidgets.QMessageBox.information(
                self,
                "Export Complete",
                f"Results exported successfully to:\n{dir_path}"
            )

            self.status_label.setText(f"Results exported to {dir_path}")

        except Exception as e:
            Log.e("EvaluationUI", f"Export failed: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export results: {str(e)}"
            )
