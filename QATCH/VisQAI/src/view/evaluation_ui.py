"""
evaluation_ui.py

Evaluation UI module for VisQ.AI system that provides comprehensive model evaluation
capabilities including metrics computation, visualization, and results export.

Author:
    [Your Name]

Date:
    2025-01-27

Version:
    1.0
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

try:
    from src.utils.metrics import Metrics
    from src.models.formulation import Formulation
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    class Log:
        """Logging utility for standardized log messages."""
        _logger = logging.getLogger("Predictor")

        @classmethod
        def i(cls, msg: str) -> None:
            """Log an informational message."""
            cls._logger.info(msg)

        @classmethod
        def w(cls, msg: str) -> None:
            """Log a warning message."""
            cls._logger.warning(msg)

        @classmethod
        def e(cls, msg: str) -> None:
            """Log an error message."""
            cls._logger.error(msg)

        @classmethod
        def d(cls, msg: str) -> None:
            """Log a debug message."""
            cls._logger.debug(msg)

except (ImportError, ModuleNotFoundError):
    from QATCH.VisQAI.src.utils.metrics import Metrics
    from QATCH.VisQAI.src.models.formulation import Formulation
    from QATCH.common.logger import Logger as Log
TAG = "[EvaluationUI]"


class EvaluationUI(QtWidgets.QWidget):
    """
    Main evaluation UI widget for the VisQ.AI system.

    This widget provides functionality to:
    - Select formulations for evaluation
    - Compute various metrics using the Metrics module
    - Visualize results with plots
    - Export evaluation results
    """

    def __init__(self, parent=None):
        """Initialize the Evaluation UI.

        Args:
            parent: Parent VisQAIWindow instance.
        """
        super().__init__(parent)
        self.parent = parent
        self.metrics_calculator = Metrics()
        self.evaluation_results = None
        self.current_results_df = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface components."""
        # Main layout
        main_layout = QtWidgets.QVBoxLayout()

        # Create toolbar
        toolbar = self.create_toolbar()
        main_layout.addWidget(toolbar)

        # Create splitter for left panel and main content
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Left panel for configuration
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel for results
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # Set initial splitter sizes (30% left, 70% right)
        splitter.setSizes([300, 700])

        main_layout.addWidget(splitter)

        # Status bar
        self.status_label = QtWidgets.QLabel("Ready to evaluate")
        self.status_label.setStyleSheet("QLabel { padding: 5px; }")
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)

    def create_toolbar(self):
        """Create the toolbar with action buttons.

        Returns:
            QtWidgets.QToolBar: Configured toolbar widget.
        """
        toolbar = QtWidgets.QToolBar()
        toolbar.setStyleSheet("QToolBar { spacing: 10px; padding: 5px; }")

        # Evaluate button
        self.evaluate_btn = QtWidgets.QPushButton("Evaluate")
        self.evaluate_btn.clicked.connect(self.run_evaluation)
        toolbar.addWidget(self.evaluate_btn)

        toolbar.addSeparator()

        # Export button
        self.export_btn = QtWidgets.QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        toolbar.addWidget(self.export_btn)

        # Clear button
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_results)
        toolbar.addWidget(self.clear_btn)

        toolbar.addSeparator()

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        toolbar.addWidget(self.progress_bar)

        return toolbar

    def create_left_panel(self):
        """Create the left configuration panel.

        Returns:
            QtWidgets.QWidget: Configured left panel widget.
        """
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Title
        title = QtWidgets.QLabel("Evaluation Configuration")
        title.setStyleSheet(
            "QLabel { font-weight: bold; font-size: 14px; padding: 5px; }")
        layout.addWidget(title)

        # Formulation selection section
        form_group = QtWidgets.QGroupBox("Select Formulations")
        form_layout = QtWidgets.QVBoxLayout()

        # Formulation list
        self.formulation_list = QtWidgets.QListWidget()
        self.formulation_list.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection)
        self.formulation_list.setMinimumHeight(150)

        # Add sample formulations (replace with actual data in production)
        self.load_available_formulations()

        form_layout.addWidget(self.formulation_list)

        # Select all/none buttons
        btn_layout = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_formulations)
        select_none_btn = QtWidgets.QPushButton("Select None")
        select_none_btn.clicked.connect(self.select_no_formulations)
        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(select_none_btn)
        form_layout.addLayout(btn_layout)

        form_group.setLayout(form_layout)
        layout.addWidget(form_group)

        # Metrics selection section
        metrics_group = QtWidgets.QGroupBox("Select Metrics")
        metrics_layout = QtWidgets.QVBoxLayout()

        self.metric_checkboxes = {}

        # Create checkboxes for each available metric
        metric_scroll = QtWidgets.QScrollArea()
        metric_widget = QtWidgets.QWidget()
        metric_widget_layout = QtWidgets.QVBoxLayout()

        available_metrics = self.metrics_calculator.get_available_metrics()

        # Group metrics by category
        error_metrics = ['mae', 'rmse', 'mse', 'mape',
                         'median_ae', 'max_error', 'std_error']
        performance_metrics = ['r2', 'coverage']
        uncertainty_metrics = ['mean_cv', 'median_cv', 'mean_std']
        other_metrics = ['count']

        # Add error metrics
        error_label = QtWidgets.QLabel("Error Metrics:")
        error_label.setStyleSheet("font-weight: bold;")
        metric_widget_layout.addWidget(error_label)
        for metric in error_metrics:
            if metric in available_metrics:
                checkbox = QtWidgets.QCheckBox(self.format_metric_name(metric))
                # Default selections
                checkbox.setChecked(metric in ['mae', 'rmse', 'r2'])
                self.metric_checkboxes[metric] = checkbox
                metric_widget_layout.addWidget(checkbox)

        # Add performance metrics
        perf_label = QtWidgets.QLabel("Performance Metrics:")
        perf_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        metric_widget_layout.addWidget(perf_label)
        for metric in performance_metrics:
            if metric in available_metrics:
                checkbox = QtWidgets.QCheckBox(self.format_metric_name(metric))
                checkbox.setChecked(metric in ['r2'])
                self.metric_checkboxes[metric] = checkbox
                metric_widget_layout.addWidget(checkbox)

        # Add uncertainty metrics
        uncert_label = QtWidgets.QLabel("Uncertainty Metrics:")
        uncert_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        metric_widget_layout.addWidget(uncert_label)
        for metric in uncertainty_metrics:
            if metric in available_metrics:
                checkbox = QtWidgets.QCheckBox(self.format_metric_name(metric))
                self.metric_checkboxes[metric] = checkbox
                metric_widget_layout.addWidget(checkbox)

        metric_widget.setLayout(metric_widget_layout)
        metric_scroll.setWidget(metric_widget)
        metric_scroll.setWidgetResizable(True)
        metrics_layout.addWidget(metric_scroll)

        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        # Options section
        options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QVBoxLayout()

        self.per_shear_checkbox = QtWidgets.QCheckBox(
            "Compute metrics per shear rate")
        self.per_shear_checkbox.setChecked(True)
        options_layout.addWidget(self.per_shear_checkbox)

        self.show_plots_checkbox = QtWidgets.QCheckBox("Generate plots")
        self.show_plots_checkbox.setChecked(True)
        options_layout.addWidget(self.show_plots_checkbox)

        self.export_plots_checkbox = QtWidgets.QCheckBox(
            "Export plots with results")
        options_layout.addWidget(self.export_plots_checkbox)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        layout.addStretch()
        panel.setLayout(layout)

        return panel

    def create_right_panel(self):
        """Create the right results panel.

        Returns:
            QtWidgets.QWidget: Configured right panel widget.
        """
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Create tab widget for different result views
        self.results_tabs = QtWidgets.QTabWidget()

        # Summary tab
        self.summary_widget = self.create_summary_tab()
        self.results_tabs.addTab(self.summary_widget, "Summary")

        # Detailed metrics tab
        self.metrics_table_widget = self.create_metrics_table_tab()
        self.results_tabs.addTab(self.metrics_table_widget, "Detailed Metrics")

        # Plots tab
        self.plots_widget = self.create_plots_tab()
        self.results_tabs.addTab(self.plots_widget, "Visualizations")

        # Raw data tab
        self.raw_data_widget = self.create_raw_data_tab()
        self.results_tabs.addTab(self.raw_data_widget, "Raw Data")

        layout.addWidget(self.results_tabs)
        panel.setLayout(layout)

        return panel

    def create_summary_tab(self):
        """Create the summary tab widget.

        Returns:
            QtWidgets.QWidget: Configured summary tab widget.
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Summary text area
        self.summary_text = QtWidgets.QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlainText(
            "No evaluation results yet. Click 'Evaluate' to begin.")

        layout.addWidget(self.summary_text)
        widget.setLayout(layout)

        return widget

    def create_metrics_table_tab(self):
        """Create the metrics table tab widget.

        Returns:
            QtWidgets.QWidget: Configured metrics table tab widget.
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Metrics table
        self.metrics_table = QtWidgets.QTableWidget()
        self.metrics_table.setSortingEnabled(True)
        self.metrics_table.setAlternatingRowColors(True)

        layout.addWidget(self.metrics_table)
        widget.setLayout(layout)

        return widget

    def create_plots_tab(self):
        """Create the plots tab widget.

        Returns:
            QtWidgets.QWidget: Configured plots tab widget.
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Plot selection dropdown
        plot_control_layout = QtWidgets.QHBoxLayout()
        plot_control_layout.addWidget(QtWidgets.QLabel("Select Plot:"))

        self.plot_selector = QtWidgets.QComboBox()
        self.plot_selector.currentTextChanged.connect(self.update_plot)
        plot_control_layout.addWidget(self.plot_selector)

        plot_control_layout.addStretch()
        layout.addLayout(plot_control_layout)

        # Plot canvas
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        widget.setLayout(layout)

        return widget

    def create_raw_data_tab(self):
        """Create the raw data tab widget.

        Returns:
            QtWidgets.QWidget: Configured raw data tab widget.
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Raw data table
        self.raw_data_table = QtWidgets.QTableWidget()
        self.raw_data_table.setSortingEnabled(True)
        self.raw_data_table.setAlternatingRowColors(True)

        layout.addWidget(self.raw_data_table)
        widget.setLayout(layout)

        return widget

    def format_metric_name(self, metric_name: str) -> str:
        """Format metric name for display.

        Args:
            metric_name: Raw metric name.

        Returns:
            str: Formatted metric name.
        """
        name_mapping = {
            'mae': 'Mean Absolute Error (MAE)',
            'rmse': 'Root Mean Square Error (RMSE)',
            'mse': 'Mean Square Error (MSE)',
            'mape': 'Mean Absolute Percentage Error (MAPE)',
            'median_ae': 'Median Absolute Error',
            'r2': 'R-squared (R²)',
            'coverage': '95% CI Coverage (%)',
            'mean_cv': 'Mean Coefficient of Variation',
            'median_cv': 'Median Coefficient of Variation',
            'max_error': 'Maximum Error',
            'std_error': 'Std Dev of Errors',
            'mean_std': 'Mean Prediction Std Dev',
            'count': 'Sample Count'
        }
        return name_mapping.get(metric_name, metric_name.replace('_', ' ').title())

    def load_available_formulations(self):
        """Load available formulations from the parent window."""
        formulation_names = self.parent.import_run_names
        Log.i(f"Formualtions available{formulation_names}")
        for formulation in formulation_names:
            self.formulation_list.addItem(formulation)

    def select_all_formulations(self):
        """Select all formulations in the list."""
        for i in range(self.formulation_list.count()):
            self.formulation_list.item(i).setSelected(True)

    def select_no_formulations(self):
        """Deselect all formulations in the list."""
        for i in range(self.formulation_list.count()):
            self.formulation_list.item(i).setSelected(False)

    def get_selected_metrics(self) -> List[str]:
        """Get list of selected metrics.

        Returns:
            List[str]: Names of selected metrics.
        """
        selected = []
        for metric_name, checkbox in self.metric_checkboxes.items():
            if checkbox.isChecked():
                selected.append(metric_name)
        return selected

    def get_selected_formulations(self) -> List[str]:
        """Get list of selected formulations.

        Returns:
            List[str]: Names of selected formulations.
        """
        selected = []
        for item in self.formulation_list.selectedItems():
            selected.append(item.text())
        return selected

    def run_evaluation(self):
        """Run the evaluation process."""
        # Get selected formulations
        selected_formulations = self.get_selected_formulations()
        if not selected_formulations:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Please select at least one formulation to evaluate."
            )
            return

        # Get selected metrics
        selected_metrics = self.get_selected_metrics()
        if not selected_metrics:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Please select at least one metric to compute."
            )
            return

        try:
            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, len(selected_formulations))
            self.evaluate_btn.setEnabled(False)
            self.status_label.setText("Running evaluation...")

            # Get test data for selected formulations
            # This would come from the parent window's data
            test_data = self.get_test_data(selected_formulations)

            if test_data is None or test_data.empty:
                QtWidgets.QMessageBox.warning(
                    self, "Warning", "No test data available for selected formulations."
                )
                return

            # Run predictor's evaluate method
            if hasattr(self.parent, 'predictor') and self.parent.predictor is not None:
                self.current_results_df = self.parent.predictor.evaluate(
                    test_data)
            else:
                # Generate sample results for demonstration
                self.current_results_df = self.generate_sample_results(
                    test_data)

            # Compute metrics
            self.compute_and_display_metrics(selected_metrics)

            # Update plots if requested
            if self.show_plots_checkbox.isChecked():
                self.generate_plots()

            # Enable export
            self.export_btn.setEnabled(True)

            self.status_label.setText("Evaluation completed successfully")

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Evaluation failed: {str(e)}"
            )
            self.status_label.setText("Evaluation failed")

        finally:
            self.progress_bar.setVisible(False)
            self.evaluate_btn.setEnabled(True)

    def get_test_data(self, formulation_names: List[str]) -> pd.DataFrame:
        formulations = self.parent.import_formulations
        data_list = [form.to_dataframe(trianing=True) for form in formulations]
        return pd.DataFrame(data_list)

    def compute_and_display_metrics(self, selected_metrics: List[str]):
        """Compute and display the selected metrics.

        Args:
            selected_metrics: List of metric names to compute.
        """
        if self.current_results_df is None:
            return

        # Compute overall metrics
        overall_metrics = self.metrics_calculator.compute_overall(
            self.current_results_df, selected_metrics
        )

        # Compute per-shear metrics if requested
        per_shear_metrics = None
        if self.per_shear_checkbox.isChecked():
            per_shear_metrics = self.metrics_calculator.compute_per_shear_rate(
                self.current_results_df, selected_metrics
            )

        # Update summary
        self.update_summary(overall_metrics, per_shear_metrics)

        # Update metrics table
        self.update_metrics_table(overall_metrics, per_shear_metrics)

        # Update raw data table
        self.update_raw_data_table()

    def update_summary(self, overall_metrics: Dict, per_shear_metrics: Optional[pd.DataFrame]):
        """Update the summary text with evaluation results.

        Args:
            overall_metrics: Dictionary of overall metrics.
            per_shear_metrics: DataFrame of per-shear-rate metrics.
        """
        summary = "=" * 60 + "\n"
        summary += "EVALUATION SUMMARY\n"
        summary += "=" * 60 + "\n\n"

        # Formulation info
        formulations = self.current_results_df['formulation_id'].unique()
        summary += f"Formulations evaluated: {len(formulations)}\n"
        summary += f"Total data points: {len(self.current_results_df)}\n"
        summary += "\n"

        # Overall metrics
        summary += "Overall Metrics:\n"
        summary += "-" * 40 + "\n"
        for metric_name, value in overall_metrics.items():
            display_name = self.format_metric_name(metric_name)
            summary += f"{display_name}: {value:.4f}\n"

        # Key insights
        summary += "\n"
        summary += "Key Insights:\n"
        summary += "-" * 40 + "\n"

        if 'mae' in overall_metrics:
            summary += f"• Average prediction error: {overall_metrics['mae']:.2f} units\n"

        if 'r2' in overall_metrics:
            r2_pct = overall_metrics['r2'] * 100
            summary += f"• Model explains {r2_pct:.1f}% of variance\n"

        if 'coverage' in overall_metrics:
            summary += f"• {overall_metrics['coverage']:.1f}% of actual values within 95% CI\n"

        if 'mape' in overall_metrics:
            summary += f"• Mean percentage error: {overall_metrics['mape']:.1f}%\n"

        # Per-shear insights if available
        if per_shear_metrics is not None and not per_shear_metrics.empty:
            summary += "\n"
            summary += "Performance by Shear Rate:\n"
            summary += "-" * 40 + "\n"

            if 'mae' in per_shear_metrics.columns:
                best_shear = per_shear_metrics.loc[per_shear_metrics['mae'].idxmin(
                )]
                worst_shear = per_shear_metrics.loc[per_shear_metrics['mae'].idxmax(
                )]
                summary += f"• Best performance: Shear rate {best_shear['shear_rate']:.0f} s⁻¹ "
                summary += f"(MAE: {best_shear['mae']:.2f})\n"
                summary += f"• Worst performance: Shear rate {worst_shear['shear_rate']:.0f} s⁻¹ "
                summary += f"(MAE: {worst_shear['mae']:.2f})\n"

        self.summary_text.setPlainText(summary)

    def update_metrics_table(self, overall_metrics: Dict, per_shear_metrics: Optional[pd.DataFrame]):
        """Update the metrics table with computed values.

        Args:
            overall_metrics: Dictionary of overall metrics.
            per_shear_metrics: DataFrame of per-shear-rate metrics.
        """
        if per_shear_metrics is not None and not per_shear_metrics.empty:
            # Show per-shear metrics
            self.populate_table_from_dataframe(
                self.metrics_table, per_shear_metrics)
        else:
            # Show overall metrics in table format
            self.metrics_table.setRowCount(len(overall_metrics))
            self.metrics_table.setColumnCount(2)
            self.metrics_table.setHorizontalHeaderLabels(['Metric', 'Value'])

            for i, (metric_name, value) in enumerate(overall_metrics.items()):
                # Metric name
                name_item = QtWidgets.QTableWidgetItem(
                    self.format_metric_name(metric_name))
                self.metrics_table.setItem(i, 0, name_item)

                # Metric value
                value_item = QtWidgets.QTableWidgetItem(f"{value:.4f}")
                value_item.setTextAlignment(
                    QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.metrics_table.setItem(i, 1, value_item)

        self.metrics_table.resizeColumnsToContents()

    def update_raw_data_table(self):
        """Update the raw data table with evaluation results."""
        if self.current_results_df is not None:
            self.populate_table_from_dataframe(
                self.raw_data_table, self.current_results_df)

    def populate_table_from_dataframe(self, table_widget: QtWidgets.QTableWidget, df: pd.DataFrame):
        """Populate a QTableWidget with data from a DataFrame.

        Args:
            table_widget: The table widget to populate.
            df: The DataFrame containing the data.
        """
        table_widget.setRowCount(len(df))
        table_widget.setColumnCount(len(df.columns))
        table_widget.setHorizontalHeaderLabels(df.columns.tolist())

        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                value = df.iloc[i, j]

                # Format the value based on type
                if isinstance(value, (int, np.integer)):
                    item_text = str(value)
                elif isinstance(value, (float, np.floating)):
                    if col in ['shear_rate']:
                        item_text = f"{value:.0f}"
                    elif col in ['pct_error', 'cv', 'coverage']:
                        item_text = f"{value:.2f}%"
                    else:
                        item_text = f"{value:.4f}"
                else:
                    item_text = str(value)

                item = QtWidgets.QTableWidgetItem(item_text)

                # Align numbers to the right
                if isinstance(value, (int, float, np.integer, np.floating)):
                    item.setTextAlignment(
                        QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

                table_widget.setItem(i, j, item)

        table_widget.resizeColumnsToContents()

    def generate_plots(self):
        """Generate visualization plots for the evaluation results."""
        if self.current_results_df is None:
            return

        # Clear existing plot options
        self.plot_selector.clear()

        # Add available plot options
        plot_options = [
            "Actual vs Predicted",
            "Residuals Distribution",
            "Error by Shear Rate",
            "Q-Q Plot",
            "Confidence Intervals",
            "Percentage Error Distribution"
        ]

        self.plot_selector.addItems(plot_options)

        # Generate first plot
        self.update_plot("Actual vs Predicted")

    def update_plot(self, plot_type: str):
        """Update the displayed plot based on selection.

        Args:
            plot_type: Type of plot to display.
        """
        if self.current_results_df is None:
            return

        # Clear the figure
        self.figure.clear()

        if plot_type == "Actual vs Predicted":
            self.plot_actual_vs_predicted()
        elif plot_type == "Residuals Distribution":
            self.plot_residuals_distribution()
        elif plot_type == "Error by Shear Rate":
            self.plot_error_by_shear()
        elif plot_type == "Q-Q Plot":
            self.plot_qq()
        elif plot_type == "Confidence Intervals":
            self.plot_confidence_intervals()
        elif plot_type == "Percentage Error Distribution":
            self.plot_percentage_error()

        self.canvas.draw()

    def plot_actual_vs_predicted(self):
        """Create actual vs predicted scatter plot."""
        ax = self.figure.add_subplot(111)

        df = self.current_results_df

        # Create scatter plot
        scatter = ax.scatter(df['actual'], df['predicted'],
                             c=df['shear_rate'], cmap='viridis',
                             alpha=0.6, edgecolors='black', linewidth=0.5)

        # Add perfect prediction line
        min_val = min(df['actual'].min(), df['predicted'].min())
        max_val = max(df['actual'].max(), df['predicted'].max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', alpha=0.7, label='Perfect Prediction')

        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(df['actual'], df['predicted'])

        ax.set_xlabel('Actual Viscosity', fontsize=12)
        ax.set_ylabel('Predicted Viscosity', fontsize=12)
        ax.set_title(
            f'Actual vs Predicted Values (R² = {r2:.3f})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add colorbar for shear rate
        cbar = self.figure.colorbar(scatter, ax=ax)
        cbar.set_label('Shear Rate (s⁻¹)', fontsize=10)

        self.figure.tight_layout()

    def plot_residuals_distribution(self):
        """Create residuals distribution plot."""
        ax = self.figure.add_subplot(111)

        df = self.current_results_df

        # Create histogram
        ax.hist(df['residual'], bins=30, edgecolor='black',
                alpha=0.7, color='steelblue')

        # Add normal distribution overlay
        from scipy import stats
        mu = df['residual'].mean()
        sigma = df['residual'].std()
        x = np.linspace(df['residual'].min(), df['residual'].max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma) * len(df) * (df['residual'].max() - df['residual'].min()) / 30,
                'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')

        ax.axvline(x=0, color='green', linestyle='--',
                   alpha=0.7, label='Zero Error')
        ax.set_xlabel('Residuals', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Residuals', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.figure.tight_layout()

    def plot_error_by_shear(self):
        """Create box plot of errors by shear rate."""
        ax = self.figure.add_subplot(111)

        df = self.current_results_df

        # Prepare data for box plot
        shear_rates = sorted(df['shear_rate'].unique())
        data_by_shear = [df[df['shear_rate'] == sr]
                         ['abs_error'].values for sr in shear_rates]

        # Create box plot
        bp = ax.boxplot(data_by_shear, labels=[f"{sr:.0f}" for sr in shear_rates],
                        patch_artist=True)

        # Color the boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(shear_rates)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xlabel('Shear Rate (s⁻¹)', fontsize=12)
        ax.set_ylabel('Absolute Error', fontsize=12)
        ax.set_title('Error Distribution by Shear Rate', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        self.figure.tight_layout()

    def plot_qq(self):
        """Create Q-Q plot for residuals."""
        ax = self.figure.add_subplot(111)

        from scipy import stats
        df = self.current_results_df

        # Create Q-Q plot
        stats.probplot(df['residual'], dist="norm", plot=ax)

        ax.set_title('Q-Q Plot of Residuals', fontsize=14)
        ax.grid(True, alpha=0.3)

        self.figure.tight_layout()

    def plot_confidence_intervals(self):
        """Create confidence interval coverage plot."""
        ax = self.figure.add_subplot(111)

        df = self.current_results_df.sort_values('predicted')

        # Plot predicted values with confidence intervals
        x_range = range(len(df))
        ax.plot(x_range, df['actual'].values, 'o', color='blue',
                alpha=0.5, label='Actual', markersize=4)
        ax.plot(x_range, df['predicted'].values, '-',
                color='red', alpha=0.7, label='Predicted', linewidth=1)
        ax.fill_between(x_range, df['lower_95'].values, df['upper_95'].values,
                        color='red', alpha=0.2, label='95% CI')

        ax.set_xlabel('Sample Index (sorted by predicted value)', fontsize=12)
        ax.set_ylabel('Viscosity', fontsize=12)
        ax.set_title('Predictions with 95% Confidence Intervals', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.figure.tight_layout()

    def plot_percentage_error(self):
        """Create percentage error distribution plot."""
        ax = self.figure.add_subplot(111)

        df = self.current_results_df

        # Create violin plot for percentage errors by formulation
        formulations = df['formulation_id'].unique()
        data_by_form = [df[df['formulation_id'] == f]
                        ['pct_error'].values for f in formulations]

        parts = ax.violinplot(data_by_form, positions=range(len(formulations)),
                              showmeans=True, showmedians=True)

        ax.set_xticks(range(len(formulations)))
        ax.set_xticklabels(formulations, rotation=45, ha='right')
        ax.set_xlabel('Formulation', fontsize=12)
        ax.set_ylabel('Percentage Error (%)', fontsize=12)
        ax.set_title(
            'Percentage Error Distribution by Formulation', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        self.figure.tight_layout()

    def export_results(self):
        """Export evaluation results to files."""
        if self.current_results_df is None:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "No results to export. Please run evaluation first."
            )
            return

        # Get save directory from user
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Results"
        )

        if not directory:
            return

        try:
            from datetime import datetime
            import os

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"evaluation_results_{timestamp}"

            # Export raw results
            results_path = os.path.join(directory, f"{base_name}_raw.csv")
            self.current_results_df.to_csv(results_path, index=False)

            # Export metrics
            selected_metrics = self.get_selected_metrics()

            # Overall metrics
            overall_metrics = self.metrics_calculator.compute_overall(
                self.current_results_df, selected_metrics
            )
            overall_df = pd.DataFrame([overall_metrics])
            overall_path = os.path.join(
                directory, f"{base_name}_overall_metrics.csv")
            overall_df.to_csv(overall_path, index=False)

            # Per-shear metrics if requested
            if self.per_shear_checkbox.isChecked():
                per_shear_metrics = self.metrics_calculator.compute_per_shear_rate(
                    self.current_results_df, selected_metrics
                )
                per_shear_path = os.path.join(
                    directory, f"{base_name}_per_shear_metrics.csv")
                per_shear_metrics.to_csv(per_shear_path, index=False)

            # Export plots if requested
            if self.export_plots_checkbox.isChecked() and self.show_plots_checkbox.isChecked():
                self.export_all_plots(directory, base_name)

            QtWidgets.QMessageBox.information(
                self, "Success", f"Results exported successfully to:\n{directory}"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to export results: {str(e)}"
            )

    def export_all_plots(self, directory: str, base_name: str):
        """Export all plot types to files.

        Args:
            directory: Directory to save plots to.
            base_name: Base filename for plots.
        """
        import os

        plot_types = [
            "Actual vs Predicted",
            "Residuals Distribution",
            "Error by Shear Rate",
            "Q-Q Plot",
            "Confidence Intervals",
            "Percentage Error Distribution"
        ]

        for plot_type in plot_types:
            # Update plot
            self.update_plot(plot_type)

            # Save plot
            filename = plot_type.lower().replace(' ', '_')
            plot_path = os.path.join(directory, f"{base_name}_{filename}.png")
            self.figure.savefig(plot_path, dpi=300, bbox_inches='tight')

    def clear_results(self):
        """Clear all evaluation results."""
        # Clear data
        self.current_results_df = None
        self.evaluation_results = None

        # Clear displays
        self.summary_text.setPlainText(
            "No evaluation results yet. Click 'Evaluate' to begin.")
        self.metrics_table.clear()
        self.raw_data_table.clear()
        self.figure.clear()
        self.canvas.draw()
        self.plot_selector.clear()

        # Update status
        self.export_btn.setEnabled(False)
        self.status_label.setText("Results cleared")


# Additional helper class for potential future enhancements
class EvaluationReport:
    """Class for generating detailed evaluation reports."""

    @staticmethod
    def generate_html_report(results_df: pd.DataFrame,
                             metrics: Dict,
                             plots: Dict[str, bytes]) -> str:
        """Generate an HTML report of evaluation results.

        Args:
            results_df: DataFrame of evaluation results.
            metrics: Dictionary of computed metrics.
            plots: Dictionary mapping plot names to image bytes.

        Returns:
            str: HTML content of the report.
        """
        # This would generate a comprehensive HTML report
        # Implementation would go here
        pass
