"""
evaluation_widget.py

Provides a configuration interface for statistical data evaluation.

This module contains the EvaluationWidget, a specialized UI panel that allows
users to configure parameters for comparing predicted vs. measured viscosity
data. It supports dynamic metric selection, log-scale transformations, and
discrete shear rate range filtering.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

import os

try:
    from components.range_slider import RangeSlider
    from src.utils.metrics import Metrics
    from src.view.architecture import Architecture

except (ImportError, ModuleNotFoundError):
    from QATCH.common.architecture import Architecture
    from QATCH.VisQAI.src.utils.metrics import Metrics
    from QATCH.VisQAI.src.view.components.range_slider import RangeSlider
from PyQt5 import QtCore, QtGui, QtWidgets


class EvaluationWidget(QtWidgets.QFrame):
    """A configuration panel for statistical evaluation of model performance.

    This widget provides a form-like interface where users can select specific
    metrics (e.g., RMSE, R²), toggle log transformations, and define the shear
    rate boundaries for the evaluation. It emits a configuration dictionary
    to a controller to trigger calculation and plotting.

    Attributes:
        run_requested (QtCore.pyqtSignal): Emits a dict containing configuration:
            {'metric': str, 'shear_min': float, 'shear_max': float,
             'log_shear': bool, 'log_viscosity': bool}.
        clear_requested (QtCore.pyqtSignal): Emitted to signal that evaluation
            plots or results should be cleared.
        closed (QtCore.pyqtSignal): Emitted when the user closes the panel.
        SHEAR_STEPS (list[int]): The discrete shear rate points (1/s) used for
            slider snapping.
        metric_mapping (dict): Maps user-friendly display names (e.g., 'RMSE')
            to internal metric keys (e.g., 'rmse').
    """

    run_requested = QtCore.pyqtSignal(dict)
    clear_requested = QtCore.pyqtSignal()
    closed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        """Initializes the widget with styling, metrics, and interactive controls.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("class", "card")
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setYOffset(10)
        shadow.setColor(QtGui.QColor(0, 0, 0, 40))
        self.setGraphicsEffect(shadow)

        self.setVisible(False)
        self.setMinimumWidth(520)
        self.SHEAR_STEPS = [100, 1000, 10000, 100000, 15000000]
        self._load_supported_metrics()
        self._init_ui()

    def _load_supported_metrics(self):
        """Dynamically builds the metric mapping from the Metrics utility class.

        Iterates through metric categories (Basic, Advanced, etc.) and formats
        internal snake_case keys into Title Case display names with proper
        acronym capitalization (e.g., 'rmse' -> 'RMSE').
        """
        self.metric_mapping = {"True vs. Predicted Plot": "true_vs_pred"}
        metric_dicts = [
            Metrics.BASIC_METRICS,
            Metrics.ADVANCED_METRICS,
            Metrics.DISTRIBUTION_METRICS,
            Metrics.CORRELATION_METRICS,
        ]

        for m_dict in metric_dicts:
            for key in m_dict.keys():
                display_name = key.replace("_", " ").title()

                # Pretty print common acronyms
                replacements = {
                    "Rmse": "RMSE",
                    "Mae": "MAE",
                    "Mape": "MAPE",
                    "Mse": "MSE",
                    "Smape": "SMAPE",
                    "Msle": "MSLE",
                    "Cv": "CV",
                    "Iqr": "IQR",
                    "R2": "R²",
                }
                for old, new in replacements.items():
                    display_name = display_name.replace(old, new)

                self.metric_mapping[display_name] = key

    def _init_ui(self):
        """Builds and layouts the user interface components."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 20)
        layout.setSpacing(15)

        header = QtWidgets.QHBoxLayout()
        lbl_title = QtWidgets.QLabel("Evaluation Configuration")
        lbl_title.setObjectName("evalTitle")
        header.addWidget(lbl_title)

        btn_close = QtWidgets.QToolButton()
        btn_close.setObjectName("btnEvalClose")
        btn_close.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "close-circle-svgrepo-com.svg",
                )
            )
        )
        btn_close.setIconSize(QtCore.QSize(18, 18))
        btn_close.setFixedSize(24, 24)
        btn_close.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn_close.clicked.connect(self.close_widget)

        header.addStretch()
        header.addWidget(btn_close)
        layout.addLayout(header)

        # Configuration Group
        grp_config = QtWidgets.QGroupBox("Metric & Transformation")
        config_layout = QtWidgets.QFormLayout(grp_config)
        config_layout.setSpacing(12)

        ## Metric Dropdown
        self.combo_metric = QtWidgets.QComboBox()
        self.combo_metric.setFixedHeight(28)
        self.combo_metric.setStyleSheet("background-color: #ffffff;")
        dynamic_metrics = [
            k for k in self.metric_mapping.keys() if k != "True vs. Predicted Plot"
        ]
        sorted_display_names = ["True vs. Predicted Plot"] + sorted(dynamic_metrics)
        self.combo_metric.addItems(sorted_display_names)
        self.combo_metric.setCurrentIndex(0)
        config_layout.addRow("Evaluation Metric:", self.combo_metric)

        # Data Transformation Checkboxes
        transform_layout = QtWidgets.QHBoxLayout()
        self.chk_log_shear = QtWidgets.QCheckBox("Log Shear Rate")
        self.chk_log_visc = QtWidgets.QCheckBox("Log Viscosity")
        self.chk_log_shear.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.chk_log_visc.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        transform_layout.addWidget(self.chk_log_shear)
        transform_layout.addWidget(self.chk_log_visc)
        transform_layout.addStretch()

        config_layout.addRow("Log Transforms:", transform_layout)
        layout.addWidget(grp_config)

        # Shear Rate Range Group
        grp_shear = QtWidgets.QGroupBox("Shear Rate Range (1/s)")
        shear_layout = QtWidgets.QHBoxLayout(grp_shear)
        shear_layout.setSpacing(12)
        shear_layout.setContentsMargins(15, 15, 15, 15)

        spin_style = """
            QDoubleSpinBox {
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 4px;
                background-color: #ffffff;
            }
            QDoubleSpinBox:focus { border: 1px solid #2596be; }
        """

        self.spin_shear_min = QtWidgets.QDoubleSpinBox()
        self.spin_shear_min.setRange(100, 15000000)
        self.spin_shear_min.setValue(100)
        self.spin_shear_min.setFixedWidth(100)
        self.spin_shear_min.setDecimals(0)
        self.spin_shear_min.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spin_shear_min.setStyleSheet(spin_style)

        # Instantiate with the actual values and pass the snap points
        self.range_slider = RangeSlider(100, 15000000)
        self.range_slider.setStepValues(self.SHEAR_STEPS)
        self.range_slider.setValues(100, 15000000)
        self.range_slider.setFixedHeight(24)

        self.spin_shear_max = QtWidgets.QDoubleSpinBox()
        self.spin_shear_max.setRange(100, 15000000)
        self.spin_shear_max.setValue(15000000)
        self.spin_shear_max.setFixedWidth(100)
        self.spin_shear_max.setDecimals(0)
        self.spin_shear_max.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spin_shear_max.setStyleSheet(spin_style)

        # Connect Sliders <-> Spins
        self.range_slider.rangeChanged.connect(self._on_slider_changed)
        self.spin_shear_min.valueChanged.connect(self._on_spin_changed)
        self.spin_shear_max.valueChanged.connect(self._on_spin_changed)

        shear_layout.addWidget(self.spin_shear_min)
        shear_layout.addWidget(self.range_slider, stretch=1)
        shear_layout.addWidget(self.spin_shear_max)

        layout.addWidget(grp_shear)

        # Footer Actions
        layout.addSpacing(5)
        btn_layout = QtWidgets.QHBoxLayout()

        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_clear.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_clear.setFixedHeight(34)
        self.btn_clear.setFixedWidth(80)
        self.btn_clear.clicked.connect(self.emit_clear)

        self.btn_run = QtWidgets.QPushButton("Run Evaluation")
        self.btn_run.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_run.setFixedHeight(34)
        self.btn_run.setFixedWidth(140)
        self.btn_run.setObjectName("btnApplyFilters")
        self.btn_run.clicked.connect(self.emit_run)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addWidget(self.btn_run)
        layout.addLayout(btn_layout)

    def _on_slider_changed(self, low: float, high: float):
        """Synchronizes spinboxes when the range slider values change.

        Args:
            low (float): The current lower value of the slider.
            high (float): The current upper value of the slider.
        """
        self.spin_shear_min.blockSignals(True)
        self.spin_shear_max.blockSignals(True)
        self.spin_shear_min.setValue(low)
        self.spin_shear_max.setValue(high)
        self.spin_shear_min.blockSignals(False)
        self.spin_shear_max.blockSignals(False)

    def _on_spin_changed(self):
        """Synchronizes the range slider when the spinbox values change.

        Includes a validation check to ensure the lower bound does not exceed
        the upper bound.
        """
        low = self.spin_shear_min.value()
        high = self.spin_shear_max.value()

        if low > high:
            low = high
            self.spin_shear_min.setValue(low)

        self.range_slider.setValues(low, high)

    def emit_run(self):
        """Collects form data and emits the run_requested signal.

        The signal contains a dictionary with metric keys and transformation
        settings formatted for the evaluation engine.
        """
        metric_display = self.combo_metric.currentText()
        metric_key = self.metric_mapping.get(metric_display, "rmse")

        self.run_requested.emit(
            {
                "metric": metric_key,
                "metric_name": metric_display,
                "shear_min": self.spin_shear_min.value(),
                "shear_max": self.spin_shear_max.value(),
                "log_shear": self.chk_log_shear.isChecked(),
                "log_viscosity": self.chk_log_visc.isChecked(),
            }
        )

    def emit_clear(self):
        """Emits the clear_requested signal."""
        self.clear_requested.emit()

    def close_widget(self):
        """Hides the widget and emits the closed signal."""
        self.hide()
        self.closed.emit()
