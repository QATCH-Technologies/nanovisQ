from PyQt5 import QtCore, QtWidgets

# ---------------------------------------------------------------------------
# ICL field definitions
# Maps human-readable label -> dataframe column name used for filtering.
# ---------------------------------------------------------------------------
ICL_FILTER_FIELDS = [
    ("Protein Type", "Protein_type"),
    ("Protein Class", "Protein_class_type"),
    ("Buffer Type", "Buffer_type"),
    ("Salt Type", "Salt_type"),
    ("Stabilizer Type", "Stabilizer_type"),
    ("Surfactant Type", "Surfactant_type"),
    ("Excipient Type", "Excipient_type"),
]

# Default ICL filter applied when no stored config is present
_DEFAULT_ICL_FILTER = {
    "logic": "AND",
    "fields": ["Protein_type"],
}


class ModelOptionsDialog(QtWidgets.QDialog):
    def __init__(self, current_params, is_measured=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Inference Options")
        self.resize(420, 520)
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
                background: white; border: 1px solid #888; width: 14px;
                margin: -5px 0; border-radius: 7px;
            }
            QPushButton#btnReset {
                color: #d32f2f; background: transparent; border: none; text-align: left;
            }
            QPushButton#btnReset:hover {
                text-decoration: underline; background: #ffebee; border-radius: 4px;
            }
            /* AND/OR logic toggle buttons */
            QPushButton#btnLogicAnd, QPushButton#btnLogicOr {
                border: 1px solid #d1d5db;
                background-color: #f9fafb;
                color: #555;
                padding: 3px 14px;
                font-size: 9pt;
                font-weight: 500;
            }
            QPushButton#btnLogicAnd { border-radius: 4px 0 0 4px; border-right: none; }
            QPushButton#btnLogicOr  { border-radius: 0 4px 4px 0; }
            QPushButton#btnLogicAnd[active=true],
            QPushButton#btnLogicOr[active=true] {
                background-color: #00adee;
                color: #ffffff;
                border-color: #00adee;
            }
            QCheckBox { color: #333; spacing: 6px; }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 1px solid #d1d5db; border-radius: 3px;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                background-color: #00adee;
                border-color: #00adee;
                image: none;
            }
            QCheckBox::indicator:checked:hover { background-color: #0096d1; }
            QFrame#divider {
                color: #e5e7eb;
            }
        """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # ── Header ────────────────────────────────────────────────────────
        lbl_header = QtWidgets.QLabel("Hyperparameters")
        lbl_header.setStyleSheet("font-weight: bold; font-size: 11pt; color: #00adee;")
        layout.addWidget(lbl_header)

        # ── Hyperparameter Form ───────────────────────────────────────────
        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)

        self.spin_lr = QtWidgets.QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 1.0)
        self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setValue(current_params.get("lr", 0.01))
        self.spin_lr.setReadOnly(is_measured)
        form_layout.addRow("Learning Rate:", self.spin_lr)

        self.spin_steps = QtWidgets.QSpinBox()
        self.spin_steps.setRange(1, 10000)
        self.spin_steps.setSingleStep(10)
        self.spin_steps.setValue(current_params.get("steps", 50))
        self.spin_steps.setReadOnly(is_measured)
        form_layout.addRow("Inference Steps:", self.spin_steps)

        layout.addLayout(form_layout)

        # ── CI Slider ─────────────────────────────────────────────────────
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

        # ── Divider ───────────────────────────────────────────────────────
        layout.addWidget(self._make_divider())

        # ── ICL Query Configuration ───────────────────────────────────────
        icl_header = QtWidgets.QLabel("ICL Match Criteria")
        icl_header.setStyleSheet("font-weight: bold; font-size: 11pt; color: #00adee;")
        layout.addWidget(icl_header)

        icl_desc = QtWidgets.QLabel(
            "Choose which formulation attributes must match when selecting "
            "In-Context Learning examples. Fields are combined using the "
            "selected logic operator."
        )
        icl_desc.setWordWrap(True)
        icl_desc.setStyleSheet("color: #6b7280; font-size: 8.5pt;")
        layout.addWidget(icl_desc)

        # AND / OR toggle
        logic_row = QtWidgets.QHBoxLayout()
        logic_lbl = QtWidgets.QLabel("Combine with:")
        logic_lbl.setStyleSheet("color: #555; font-size: 9pt;")
        logic_row.addWidget(logic_lbl)

        self.btn_logic_and = QtWidgets.QPushButton("AND")
        self.btn_logic_and.setObjectName("btnLogicAnd")
        self.btn_logic_and.setFixedHeight(26)
        self.btn_logic_and.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_logic_and.clicked.connect(lambda: self._set_logic("AND"))

        self.btn_logic_or = QtWidgets.QPushButton("OR")
        self.btn_logic_or.setObjectName("btnLogicOr")
        self.btn_logic_or.setFixedHeight(26)
        self.btn_logic_or.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_logic_or.clicked.connect(lambda: self._set_logic("OR"))

        logic_row.addWidget(self.btn_logic_and)
        logic_row.addWidget(self.btn_logic_or)
        logic_row.addStretch()
        layout.addLayout(logic_row)

        # Field checkboxes — 2-column grid
        self._field_checkboxes = {}  # column_name -> QCheckBox
        stored_filter = current_params.get("icl_filter", _DEFAULT_ICL_FILTER)
        active_fields = stored_filter.get("fields", _DEFAULT_ICL_FILTER["fields"])

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)

        for i, (label, col) in enumerate(ICL_FILTER_FIELDS):
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(col in active_fields)
            self._field_checkboxes[col] = cb
            grid.addWidget(cb, i // 2, i % 2)

        layout.addLayout(grid)

        # Initialise the AND/OR toggle to match stored state (no signal yet)
        self._current_logic = stored_filter.get("logic", _DEFAULT_ICL_FILTER["logic"])
        self._refresh_logic_buttons()

        # ── Footer ────────────────────────────────────────────────────────
        layout.addStretch()
        btn_layout = QtWidgets.QHBoxLayout()

        self.btn_reset = QtWidgets.QPushButton("Reset to Defaults")
        self.btn_reset.setObjectName("btnReset")
        self.btn_reset.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_reset.clicked.connect(self.reset_defaults)
        self.btn_reset.setEnabled(not is_measured)
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addStretch()

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        btn_layout.addWidget(btn_box)

        layout.addLayout(btn_layout)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_divider():
        line = QtWidgets.QFrame()
        line.setObjectName("divider")
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        return line

    def _set_logic(self, logic: str):
        self._current_logic = logic
        self._refresh_logic_buttons()

    def _refresh_logic_buttons(self):
        """Update the AND/OR button highlight to reflect _current_logic."""
        for btn, value in ((self.btn_logic_and, "AND"), (self.btn_logic_or, "OR")):
            active = self._current_logic == value
            btn.setProperty("active", "true" if active else "false")
            # Force stylesheet re-evaluation
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # ── Public API ────────────────────────────────────────────────────────

    def reset_defaults(self):
        """Reset all controls to factory defaults."""
        self.spin_lr.setValue(0.01)
        self.spin_steps.setValue(50)
        self.slider_ci.setValue(95)

        # Reset ICL filter
        self._set_logic(_DEFAULT_ICL_FILTER["logic"])
        default_fields = _DEFAULT_ICL_FILTER["fields"]
        for col, cb in self._field_checkboxes.items():
            cb.setChecked(col in default_fields)

    def get_settings(self):
        """Return all current settings as a flat dict suitable for card.ml_params."""
        checked_fields = [col for col, cb in self._field_checkboxes.items() if cb.isChecked()]

        # Ensure at least one field is always selected so the ICL query is meaningful.
        # If the user unchecked everything fall back to the default field.
        if not checked_fields:
            checked_fields = list(_DEFAULT_ICL_FILTER["fields"])

        return {
            "lr": self.spin_lr.value(),
            "steps": self.spin_steps.value(),
            "ci": self.slider_ci.value(),
            "icl_filter": {
                "logic": self._current_logic,
                "fields": checked_fields,
            },
        }
