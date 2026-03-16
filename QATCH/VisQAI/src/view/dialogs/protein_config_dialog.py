"""Provides a configuration dialog for Protein ingredients.

This module defines the ProteinConfigDialog, which allows users to specify
detailed biochemical properties for proteins, including molecular weight,
pI mean, and pI range. It features real-time validation and visual cues
for mandatory fields.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from src.models.ingredient import Protein, ProteinClass
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.ingredient import Protein, ProteinClass

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
    """A modal dialog for adding or editing Protein ingredient properties.

    This dialog manages biochemical data for proteins. It treats 0.0 as a
    sentinel value for "unset" data, as molecular weight and pI values cannot
    be zero in a physical context. Mandatory fields are highlighted with a red
    border until valid data is entered.

    Attributes:
        _UNSET_SENTINEL (float): The numeric value (0.0) used to represent
            missing or uninitialized data.
        controller: The ingredient controller responsible for persistence.
        existing_protein (Protein, optional): The protein instance being
            edited, if any.
        result_ingredient (Protein, optional): The resulting Protein object
            after a successful save.
    """

    _UNSET_SENTINEL = 0.0

    def __init__(self, ing_ctrl, existing_protein=None, parent=None):
        """Initializes the dialog with protein data and validation state.

        Args:
            ing_ctrl: The controller logic for managing ingredients.
            existing_protein (Protein, optional): An existing Protein object
                to edit. Defaults to None.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
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
            QLineEdit[readOnly="true"] {
                background-color: #f3f4f6;
                color: #6b7280;
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

        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)

        self.edit_name = QtWidgets.QLineEdit()
        self.edit_name.setPlaceholderText("Protein name...")
        self.edit_name.setReadOnly(True)
        form_layout.addRow("Name*:", self.edit_name)

        self.combo_class = QtWidgets.QComboBox()
        self.combo_class.addItem("—  Required", None)
        self.combo_class.addItems(
            [s for s in ProteinClass.all_strings() if s.lower() not in ("none", "")]
        )
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

        self.spin_mw.valueChanged.connect(
            lambda v: self._on_numeric_changed(self.spin_mw, v)
        )
        self.spin_pi_mean.valueChanged.connect(
            lambda v: self._on_numeric_changed(self.spin_pi_mean, v)
        )
        self.spin_pi_range.valueChanged.connect(
            lambda v: self._on_numeric_changed(self.spin_pi_range, v)
        )

        # Load existing data
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

    @staticmethod
    def protein_needs_completion(protein) -> bool:
        """Determines if a Protein object is missing mandatory biochemical data.

        Args:
            protein (Protein): The protein instance to check.

        Returns:
            bool: True if any required field is None or 0.0.
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
        """ "Checks if the form contains any unset or placeholder values.

        Returns:
            bool: True if the user hasn't selected a class or provided numbers.
        """
        return (
            self.combo_class.currentIndex() == 0
            or self.spin_mw.value() == self._UNSET_SENTINEL
            or self.spin_pi_mean.value() == self._UNSET_SENTINEL
            or self.spin_pi_range.value() == self._UNSET_SENTINEL
        )

    def _mark_unset(self, widget: QtWidgets.QDoubleSpinBox) -> None:
        """Applies red styling and reveals the warning banner.

        Args:
            widget (QWidget): The input widget to highlight.
        """
        widget.setStyleSheet(_STYLE_UNSET)
        self.lbl_incomplete.setVisible(True)

    def _mark_set(self, widget: QtWidgets.QDoubleSpinBox) -> None:
        """Clears red styling and hides the banner if the form is complete.

        Args:
            widget (QWidget): The input widget to reset.
        """
        widget.setStyleSheet(_STYLE_NORMAL)
        if not self.has_incomplete_fields():
            self.lbl_incomplete.setVisible(False)

    def _on_class_changed(self, index: int) -> None:
        """Validates the class selection and updates UI styling.

        Args:
            index (int): The current index of the combo box.
        """
        if index == 0:
            self.combo_class.setStyleSheet(_STYLE_UNSET)
            self.lbl_incomplete.setVisible(True)
        else:
            self.combo_class.setStyleSheet(_STYLE_NORMAL)
            if not self.has_incomplete_fields():
                self.lbl_incomplete.setVisible(False)

    def _on_numeric_changed(
        self, widget: QtWidgets.QDoubleSpinBox, value: float
    ) -> None:
        """Validates numeric input as the user types.

        Args:
            widget (QDoubleSpinBox): The spinbox being changed.
            value (float): The current value in the spinbox.
        """
        if value > self._UNSET_SENTINEL:
            self._mark_set(widget)
        else:
            self._mark_unset(widget)

    def _populate_fields(self, data):
        """Fills the UI fields with data from a Protein object or dictionary.

        Args:
            data (Protein or dict): Data source for protein attributes.
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
            mw = data.get("mw")
            pi_mean = data.get("pi_mean")
            pi_range = data.get("pi_range")

        self.edit_name.setText(name)
        index = self.combo_class.findText(p_class, QtCore.Qt.MatchFixedString)
        self.combo_class.setCurrentIndex(index if index > 0 else 0)
        if self.combo_class.currentIndex() == 0:
            self.combo_class.setStyleSheet(_STYLE_UNSET)
            self.lbl_incomplete.setVisible(True)
        else:
            self.combo_class.setStyleSheet(_STYLE_NORMAL)
        self._apply_numeric(self.spin_mw, mw)
        self._apply_numeric(self.spin_pi_mean, pi_mean)
        self._apply_numeric(self.spin_pi_range, pi_range)

    def _apply_numeric(self, widget: QtWidgets.QDoubleSpinBox, value) -> None:
        """Safely sets a numeric value and applies validation styling.

        Args:
            widget (QDoubleSpinBox): Target widget.
            value (float or None): Data value to apply.
        """
        is_unset = value is None or float(value) == self._UNSET_SENTINEL
        widget.setValue(self._UNSET_SENTINEL if is_unset else float(value))
        if is_unset:
            self._mark_unset(widget)
        else:
            self._mark_set(widget)

    def save_and_accept(self):
        """Validates the entire form and persists the Protein via the controller.

        Displays a warning if any required fields are still at their sentinel
        values. If successful, accepts the dialog.
        """
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
                self.existing_protein.name = name
                self.existing_protein.class_type = class_type
                self.existing_protein.molecular_weight = self.spin_mw.value()
                self.existing_protein.pI_mean = self.spin_pi_mean.value()
                self.existing_protein.pI_range = self.spin_pi_range.value()
                self.controller.update(self.existing_protein.id, self.existing_protein)
                self.result_ingredient = self.existing_protein
            else:
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
        """Returns the current form data as a dictionary.

        Returns:
            dict: Dictionary with keys 'name', 'class', 'mw', 'pi_mean', 'pi_range'.
        """
        return {
            "name": self.edit_name.text().strip(),
            "class": self.combo_class.currentText(),
            "mw": self.spin_mw.value(),
            "pi_mean": self.spin_pi_mean.value(),
            "pi_range": self.spin_pi_range.value(),
        }
