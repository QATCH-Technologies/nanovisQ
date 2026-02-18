from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from src.models.ingredient import Protein, ProteinClass
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.ingredient import Protein, ProteinClass

# Per-widget stylesheets for unset vs. normal state.
# These are applied directly to individual widgets so they override the
# dialog-level stylesheet without touching other controls.
_STYLE_UNSET = (
    "border: 1.5px solid #e53935; "
    "border-radius: 4px; "
    "padding: 6px; "
    "min-height: 24px; "
    "background-color: #fff5f5;"
)
_STYLE_NORMAL = (
    "border: 1px solid #d1d5db; "
    "border-radius: 4px; "
    "padding: 6px; "
    "min-height: 24px; "
    "background-color: #ffffff;"
)


class ProteinConfigDialog(QtWidgets.QDialog):
    """Dialog for configuring protein ingredients."""

    # A stored value of exactly 0 is treated as "not set" because 0 kDa / 0 pI
    # are physically meaningless for real proteins.  Red border styling signals
    # the unset state; no setSpecialValueText is used so typing is never blocked.
    _UNSET_SENTINEL = 0.0

    def __init__(self, ing_ctrl, existing_protein=None, parent=None):
        super().__init__(parent)
        self.controller = ing_ctrl
        self.existing_protein = existing_protein
        self.result_ingredient = None
        is_edit = existing_protein is not None

        self.setWindowTitle("Edit Protein" if is_edit else "Add New Protein")
        self.resize(400, 420)
        self.setModal(True)

        self.setStyleSheet(
            """
            QDialog { background-color: #ffffff; }
            QLabel { color: #333; }
            QLineEdit, QDoubleSpinBox, QComboBox { 
                border: 1px solid #d1d5db; 
                border-radius: 4px; 
                padding: 6px; 
                min-height: 24px;
                background-color: #ffffff;
            }
            QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #00adee;
            }
        """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        lbl_header = QtWidgets.QLabel("Protein Properties")
        lbl_header.setStyleSheet("font-weight: bold; font-size: 11pt; color: #00adee;")
        layout.addWidget(lbl_header)

        # Incomplete-fields banner — hidden until there are flagged fields
        self.lbl_incomplete = QtWidgets.QLabel(
            "\u26a0  Some fields are required. Please fill in the highlighted fields below."
        )
        self.lbl_incomplete.setStyleSheet(
            "color: #b71c1c; background-color: #ffebee; "
            "border: 1px solid #ef9a9a; border-radius: 4px; padding: 6px;"
        )
        self.lbl_incomplete.setWordWrap(True)
        self.lbl_incomplete.setVisible(False)
        layout.addWidget(self.lbl_incomplete)

        # Form
        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)

        self.edit_name = QtWidgets.QLineEdit()
        self.edit_name.setPlaceholderText("e.g., mAb-1")
        form_layout.addRow("Name*:", self.edit_name)

        self.combo_class = QtWidgets.QComboBox()
        # Index 0 is a placeholder — treated as “unset” for validation.
        self.combo_class.addItem("—  Required", None)
        self.combo_class.addItems(
            [s for s in ProteinClass.all_strings() if s.lower() not in ("none", "")]
        )
        # Style the placeholder item grey so it reads as a hint, not a real choice
        self.combo_class.setItemData(
            0, QtGui.QColor("#9e9e9e"), QtCore.Qt.ItemDataRole.ForegroundRole
        )
        self.combo_class.currentIndexChanged.connect(self._on_class_changed)
        form_layout.addRow("Class*:", self.combo_class)

        self.spin_mw = QtWidgets.QDoubleSpinBox()
        self.spin_mw.setRange(0, 1000000)
        self.spin_mw.setSuffix(" kDa")
        self.spin_mw.setDecimals(1)
        form_layout.addRow("Molecular Weight*:", self.spin_mw)

        self.spin_pi_mean = QtWidgets.QDoubleSpinBox()
        self.spin_pi_mean.setRange(0, 14)
        self.spin_pi_mean.setDecimals(2)
        form_layout.addRow("pI Mean*:", self.spin_pi_mean)

        self.spin_pi_range = QtWidgets.QDoubleSpinBox()
        self.spin_pi_range.setRange(0, 14)
        self.spin_pi_range.setDecimals(2)
        form_layout.addRow("pI Range*:", self.spin_pi_range)

        layout.addLayout(form_layout)

        # Wire up value-change handlers so red clears as the user types
        self.spin_mw.valueChanged.connect(
            lambda v: self._on_numeric_changed(self.spin_mw, v)
        )
        self.spin_pi_mean.valueChanged.connect(
            lambda v: self._on_numeric_changed(self.spin_pi_mean, v)
        )
        self.spin_pi_range.valueChanged.connect(
            lambda v: self._on_numeric_changed(self.spin_pi_range, v)
        )

        # Load existing data — must happen after widgets and signal connections exist
        if existing_protein:
            self._populate_fields(existing_protein)

        # Buttons
        layout.addStretch()
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        btn_save = QtWidgets.QPushButton("Save")
        btn_save.setDefault(True)
        btn_save.setStyleSheet(
            """
            QPushButton {
                background-color: #00adee;
                color: white;
                border: none;
                padding: 6px 20px;
                border-radius: 4px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0096d1;
            }
        """
        )
        btn_save.clicked.connect(self.save_and_accept)
        btn_layout.addWidget(btn_save)

        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def protein_needs_completion(protein) -> bool:
        """Return True if *protein* has any unset required fields.

        This static method can be called by the card widget (or any other
        caller) without opening the dialog, so the edit-pen button can be
        highlighted in red when the card is first displayed.
        """
        if not isinstance(protein, Protein):
            return False
        class_missing = protein.class_type is None or (
            hasattr(protein.class_type, "value")
            and protein.class_type.value.lower() in ("none", "")
        )
        return (
            class_missing
            or protein.molecular_weight is None
            or protein.molecular_weight == 0.0
            or protein.pI_mean is None
            or protein.pI_mean == 0.0
            or protein.pI_range is None
            or protein.pI_range == 0.0
        )

    def has_incomplete_fields(self) -> bool:
        """Return True if any required field is still at its sentinel / placeholder."""
        return (
            self.combo_class.currentIndex() == 0  # placeholder selected
            or self.spin_mw.value() == self._UNSET_SENTINEL
            or self.spin_pi_mean.value() == self._UNSET_SENTINEL
            or self.spin_pi_range.value() == self._UNSET_SENTINEL
        )

    def _mark_unset(self, widget: QtWidgets.QDoubleSpinBox) -> None:
        """Apply the red 'required' style to *widget* and show the banner."""
        widget.setStyleSheet(_STYLE_UNSET)
        self.lbl_incomplete.setVisible(True)

    def _mark_set(self, widget: QtWidgets.QDoubleSpinBox) -> None:
        """Remove the red style from *widget* and hide the banner if no others remain."""
        widget.setStyleSheet(_STYLE_NORMAL)
        if not self.has_incomplete_fields():
            self.lbl_incomplete.setVisible(False)

    def _on_class_changed(self, index: int) -> None:
        """Apply or clear the red border on the class combo."""
        if index == 0:  # placeholder still selected
            self.combo_class.setStyleSheet(_STYLE_UNSET)
            self.lbl_incomplete.setVisible(True)
        else:
            self.combo_class.setStyleSheet(_STYLE_NORMAL)
            if not self.has_incomplete_fields():
                self.lbl_incomplete.setVisible(False)

    def _on_numeric_changed(
        self, widget: QtWidgets.QDoubleSpinBox, value: float
    ) -> None:
        """Clear the unset indicator as soon as the user moves away from zero."""
        if value > self._UNSET_SENTINEL:
            self._mark_set(widget)
        else:
            self._mark_unset(widget)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def _populate_fields(self, data):
        """Populate UI fields from a Protein object or a dictionary.

        Fields that are absent or explicitly None/zero are left at the
        sentinel minimum value and highlighted red so the user knows they
        need attention.  Fields with real data are filled normally.
        """
        name = ""
        p_class = "None"
        mw = None
        pi_mean = None
        pi_range = None

        if isinstance(data, Protein):
            name = data.name or ""
            if data.class_type:
                p_class = (
                    data.class_type.value
                    if hasattr(data.class_type, "value")
                    else str(data.class_type)
                )
            mw = data.molecular_weight  # may be None
            pi_mean = data.pI_mean  # may be None
            pi_range = data.pI_range  # may be None

        elif isinstance(data, dict):
            name = data.get("name", "")
            p_class = data.get("class", "None")
            mw = data.get("mw")  # intentionally no default — None means unset
            pi_mean = data.get("pi_mean")
            pi_range = data.get("pi_range")

        self.edit_name.setText(name)

        # Map old "None" / missing class to the placeholder at index 0.
        # findText returns -1 for unknown strings, which also maps to the placeholder.
        index = self.combo_class.findText(p_class, QtCore.Qt.MatchFixedString)
        self.combo_class.setCurrentIndex(index if index > 0 else 0)
        # Apply styling immediately based on whether class is set
        if self.combo_class.currentIndex() == 0:
            self.combo_class.setStyleSheet(_STYLE_UNSET)
            self.lbl_incomplete.setVisible(True)
        else:
            self.combo_class.setStyleSheet(_STYLE_NORMAL)

        # For each numeric field: treat None or 0 as "unset".
        # _apply_numeric sets the value and applies the appropriate style.
        self._apply_numeric(self.spin_mw, mw)
        self._apply_numeric(self.spin_pi_mean, pi_mean)
        self._apply_numeric(self.spin_pi_range, pi_range)

    def _apply_numeric(self, widget: QtWidgets.QDoubleSpinBox, value) -> None:
        """Set *widget* to *value*, marking it unset if the value is absent or zero."""
        is_unset = value is None or float(value) == self._UNSET_SENTINEL
        widget.setValue(self._UNSET_SENTINEL if is_unset else float(value))
        if is_unset:
            self._mark_unset(widget)
        else:
            self._mark_set(widget)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_and_accept(self):
        name = self.edit_name.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Name is required.")
            return

        if self.has_incomplete_fields():
            # Re-flag any still-unset widgets before showing the warning
            if self.combo_class.currentIndex() == 0:
                self.combo_class.setStyleSheet(_STYLE_UNSET)
                self.lbl_incomplete.setVisible(True)
            for widget in (self.spin_mw, self.spin_pi_mean, self.spin_pi_range):
                if widget.value() == self._UNSET_SENTINEL:
                    self._mark_unset(widget)
            QtWidgets.QMessageBox.warning(
                self,
                "Incomplete Fields",
                "Please provide values for all required fields "
                "(Molecular Weight, pI Mean, pI Range).",
            )
            return

        try:
            class_str = self.combo_class.currentText()
            class_type = ProteinClass.from_value(class_str)

            if self.existing_protein and isinstance(self.existing_protein, Protein):
                # UPDATE
                self.existing_protein.name = name
                self.existing_protein.class_type = class_type
                self.existing_protein.molecular_weight = self.spin_mw.value()
                self.existing_protein.pI_mean = self.spin_pi_mean.value()
                self.existing_protein.pI_range = self.spin_pi_range.value()
                self.controller.update(self.existing_protein.id, self.existing_protein)
                self.result_ingredient = self.existing_protein
            else:
                # ADD NEW
                new_protein = Protein(
                    enc_id=-1,
                    name=name,
                    class_type=class_type,
                    molecular_weight=self.spin_mw.value(),
                    pI_mean=self.spin_pi_mean.value(),
                    pI_range=self.spin_pi_range.value(),
                )
                self.result_ingredient = self.controller.add(new_protein)

            self.accept()

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Database Error", f"Failed to save Protein:\n{str(e)}"
            )

    def get_data(self):
        """Returns the protein configuration as a dictionary matching Protein attributes."""
        return {
            "name": self.edit_name.text().strip(),
            "class": self.combo_class.currentText(),
            "mw": self.spin_mw.value(),
            "pi_mean": self.spin_pi_mean.value(),
            "pi_range": self.spin_pi_range.value(),
        }
