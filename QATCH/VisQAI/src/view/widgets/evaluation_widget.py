try:
    from components.range_slider import RangeSlider
except (ImportError, ModuleNotFoundError):
    from QATCH.VisQAI.src.view.components.range_slider import RangeSlider
from PyQt5 import QtCore, QtGui, QtWidgets


class EvaluationWidget(QtWidgets.QFrame):
    """
    A specific configuration panel for running statistical evaluations
    on prediction vs. measured data.
    """

    # Signal emitted when "Run Evaluation" is clicked
    # Carries a dictionary with: {'metric': str, 'shear_min': float, 'shear_max': float}
    run_requested = QtCore.pyqtSignal(dict)

    # Signal to close/cancel evaluation mode
    closed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # 1. Enable QSS Styling
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        # 2. Inherit "Card" styles for inputs, but override container geometry
        # By setting class="card", we get the nice input/combo/spinbox styles from theme.qss
        self.setProperty("class", "card")

        # 3. Specific overrides for the dropdown panel look (overriding generic card borders)
        self.setStyleSheet(
            """
            EvaluationWidget {
                background-color: #ffffff;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
                border: 1px solid #d1d5db;
                border-top: none; 
            }
            QLabel {
                color: #1f2937;
                font-weight: 600;
            }
            /* Remove standard card hover effect for this panel */
            EvaluationWidget:hover {
                border: 1px solid #d1d5db;
                border-top: none;
                background-color: #ffffff;
            }
        """
        )

        # Shadow effect (Same as FilterWidget)
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setYOffset(10)
        shadow.setColor(QtGui.QColor(0, 0, 0, 40))
        self.setGraphicsEffect(shadow)

        self.setVisible(False)
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 20)
        layout.setSpacing(15)

        # --- Header ---
        header = QtWidgets.QHBoxLayout()
        lbl_title = QtWidgets.QLabel("Evaluation Configuration")
        lbl_title.setStyleSheet("color: #2596be; font-size: 11px; font-weight: 700;")
        header.addWidget(lbl_title)

        btn_close = QtWidgets.QToolButton()
        btn_close.setText("✕")
        btn_close.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn_close.setStyleSheet(
            "background: transparent; border: none; font-weight: bold; color: #6b7280;"
        )
        btn_close.clicked.connect(self.close_widget)

        header.addStretch()
        header.addWidget(btn_close)
        layout.addLayout(header)

        # --- Metric Selection ---
        # Wrapped in a layout for better spacing
        metric_layout = QtWidgets.QHBoxLayout()
        lbl_metric = QtWidgets.QLabel("Metric:")
        lbl_metric.setFixedWidth(50)

        self.combo_metric = QtWidgets.QComboBox()
        self.combo_metric.addItems(
            [
                "Root Mean Squared Error (RMSE)",
                "R-Squared (R²)",
                "Mean Absolute Percentage Error (MAPE)",
                "Mean Absolute Error (MAE)",
            ]
        )
        metric_layout.addWidget(lbl_metric)
        metric_layout.addWidget(self.combo_metric, stretch=1)
        layout.addLayout(metric_layout)

        # --- Shear Rate Range ---
        grp_shear = QtWidgets.QGroupBox("Shear Rate Range (1/s)")
        # Using QGroupBox style from theme.qss (inherits via parent or generic QGroupbox style if added)
        grp_shear.setStyleSheet(
            """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 15px;
                padding-bottom: 5px;
                padding-left: 5px;
                padding-right: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: #2596be;
            }
        """
        )
        shear_layout = QtWidgets.QHBoxLayout(grp_shear)
        shear_layout.setSpacing(10)

        self.spin_shear_min = QtWidgets.QDoubleSpinBox()
        self.spin_shear_min.setRange(0, 100000)
        self.spin_shear_min.setValue(100)
        self.spin_shear_min.setFixedWidth(70)
        self.spin_shear_min.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)

        self.range_slider = RangeSlider(0, 10000)
        self.range_slider.setFixedHeight(20)

        self.spin_shear_max = QtWidgets.QDoubleSpinBox()
        self.spin_shear_max.setRange(0, 100000)
        self.spin_shear_max.setValue(5000)
        self.spin_shear_max.setFixedWidth(70)
        self.spin_shear_max.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)

        # Connect Sliders <-> Spins
        self.range_slider.rangeChanged.connect(self._on_slider_changed)
        self.spin_shear_min.valueChanged.connect(self._on_spin_changed)
        self.spin_shear_max.valueChanged.connect(self._on_spin_changed)

        shear_layout.addWidget(self.spin_shear_min)
        shear_layout.addWidget(self.range_slider)
        shear_layout.addWidget(self.spin_shear_max)
        layout.addWidget(grp_shear)

        # --- Footer Actions ---
        btn_layout = QtWidgets.QHBoxLayout()

        self.btn_run = QtWidgets.QPushButton("Run Evaluation")
        self.btn_run.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        # ID set to match the blue button style in theme.qss
        self.btn_run.setObjectName("btnApplyFilters")
        self.btn_run.clicked.connect(self.emit_run)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_run)
        layout.addLayout(btn_layout)

    def _on_slider_changed(self, low, high):
        """Update spinboxes when slider changes"""
        self.spin_shear_min.blockSignals(True)
        self.spin_shear_max.blockSignals(True)
        self.spin_shear_min.setValue(low)
        self.spin_shear_max.setValue(high)
        self.spin_shear_min.blockSignals(False)
        self.spin_shear_max.blockSignals(False)

    def _on_spin_changed(self):
        """Update slider when spinboxes change"""
        low = self.spin_shear_min.value()
        high = self.spin_shear_max.value()

        if low > high:
            low = high
            self.spin_shear_min.setValue(low)

        self.range_slider.setValues(low, high)

    def emit_run(self):
        self.run_requested.emit(
            {
                "metric": self.combo_metric.currentText(),
                "shear_min": self.spin_shear_min.value(),
                "shear_max": self.spin_shear_max.value(),
            }
        )

    def close_widget(self):
        self.hide()
        self.closed.emit()
