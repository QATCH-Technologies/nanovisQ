from PyQt5 import QtCore, QtWidgets


class ModelOptionsDialog(QtWidgets.QDialog):
    def __init__(self, current_params, is_measured=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Inference Options")
        self.resize(350, 250)
        self.setModal(True)

        self.setStyleSheet(
            """
            QDialog { background-color: #ffffff; }
            QLabel { color: #333; }
            QDoubleSpinBox, QSpinBox { 
                border: 1px solid #ccc; border-radius: 4px; padding: 4px; min-height: 20px; 
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb; background: white; height: 6px; border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #00adee; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: white; border: 1px solid #888; width: 14px; margin: -5px 0; border-radius: 7px;
            }
            /* Style for the Reset Button */
            QPushButton#btnReset {
                color: #d32f2f; background: transparent; border: none; text-align: left;
            }
            QPushButton#btnReset:hover {
                text-decoration: underline; background: #ffebee; border-radius: 4px;
            }
        """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # -- Header --
        lbl_header = QtWidgets.QLabel("Hyperparameters")
        lbl_header.setStyleSheet("font-weight: bold; font-size: 11pt; color: #00adee;")
        layout.addWidget(lbl_header)

        # -- Form Layout --
        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)

        # Learning Rate
        self.spin_lr = QtWidgets.QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 1.0)
        self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setValue(current_params.get("lr", 0.01))
        self.spin_lr.setReadOnly(is_measured)
        form_layout.addRow("Learning Rate:", self.spin_lr)

        # Steps
        self.spin_steps = QtWidgets.QSpinBox()
        self.spin_steps.setRange(1, 10000)
        self.spin_steps.setSingleStep(10)
        self.spin_steps.setValue(current_params.get("steps", 50))
        self.spin_steps.setReadOnly(is_measured)
        form_layout.addRow("Inference Steps:", self.spin_steps)

        layout.addLayout(form_layout)

        # -- CI Slider Section --
        ci_group = QtWidgets.QVBoxLayout()
        ci_group.setSpacing(5)

        ci_header_layout = QtWidgets.QHBoxLayout()
        ci_header_layout.addWidget(QtWidgets.QLabel("Confidence Interval:"))
        self.lbl_ci_val = QtWidgets.QLabel(f"{current_params.get('ci', 95)}%")
        self.lbl_ci_val.setStyleSheet("font-weight: bold; color: #555;")
        self.lbl_ci_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        ci_header_layout.addWidget(self.lbl_ci_val)

        ci_group.addLayout(ci_header_layout)

        self.slider_ci = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_ci.setRange(0, 100)
        self.slider_ci.setValue(current_params.get("ci", 95))
        self.slider_ci.setEnabled(not is_measured)
        self.slider_ci.valueChanged.connect(lambda v: self.lbl_ci_val.setText(f"{v}%"))
        ci_group.addWidget(self.slider_ci)

        layout.addLayout(ci_group)
        layout.addStretch()

        # -- Footer Buttons (Reset + OK/Cancel) --
        btn_layout = QtWidgets.QHBoxLayout()

        # Reset Button (Left aligned)
        self.btn_reset = QtWidgets.QPushButton("Reset to Defaults")
        self.btn_reset.setObjectName("btnReset")
        self.btn_reset.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_reset.clicked.connect(self.reset_defaults)
        # Disable reset if card is measured/read-only
        self.btn_reset.setEnabled(not is_measured)

        btn_layout.addWidget(self.btn_reset)
        btn_layout.addStretch()

        # Standard Dialog Buttons (Right aligned)
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        btn_layout.addWidget(btn_box)

        layout.addLayout(btn_layout)

    def reset_defaults(self):
        """Resets the UI elements to default values."""
        self.spin_lr.setValue(0.01)
        self.spin_steps.setValue(50)
        self.slider_ci.setValue(95)

    def get_settings(self):
        return {
            "lr": self.spin_lr.value(),
            "steps": self.spin_steps.value(),
            "ci": self.slider_ci.value(),
        }
