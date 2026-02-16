from PyQt5 import QtCore, QtWidgets

try:
    from src.models.ingredient import Protein, ProteinClass
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.ingredient import Protein, ProteinClass


class ProteinConfigDialog(QtWidgets.QDialog):
    """Dialog for configuring protein ingredients."""

    def __init__(self, existing_protein=None, parent=None):
        super().__init__(parent)
        self.existing_protein = existing_protein
        is_edit = existing_protein is not None
        self.setWindowTitle("Edit Protein" if is_edit else "Add New Protein")
        self.resize(400, 380)
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

        # Form
        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)

        self.edit_name = QtWidgets.QLineEdit()
        self.edit_name.setPlaceholderText("e.g., mAb-1")
        form_layout.addRow("Name*:", self.edit_name)

        # Changed to ComboBox to match ProteinClass Enum
        self.combo_class = QtWidgets.QComboBox()
        # Populate from ProteinClass Enum
        self.combo_class.addItems(ProteinClass.all_strings())
        form_layout.addRow("Class:", self.combo_class)

        self.spin_mw = QtWidgets.QDoubleSpinBox()
        self.spin_mw.setRange(0, 1000000)
        self.spin_mw.setSuffix(" kDa")
        self.spin_mw.setDecimals(1)
        form_layout.addRow("Molecular Weight:", self.spin_mw)

        self.spin_pi_mean = QtWidgets.QDoubleSpinBox()
        self.spin_pi_mean.setRange(0, 14)
        self.spin_pi_mean.setDecimals(2)
        self.spin_pi_mean.setValue(7.0)
        form_layout.addRow("pI Mean:", self.spin_pi_mean)

        self.spin_pi_range = QtWidgets.QDoubleSpinBox()
        self.spin_pi_range.setRange(0, 14)
        self.spin_pi_range.setDecimals(2)
        self.spin_pi_range.setValue(0.5)
        form_layout.addRow("pI Range:", self.spin_pi_range)

        layout.addLayout(form_layout)

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
        btn_save.clicked.connect(self.accept)
        btn_layout.addWidget(btn_save)

        layout.addLayout(btn_layout)

    def _populate_fields(self, data):
        """Populates UI fields from a Protein object or a dictionary."""
        name = ""
        p_class = "None"
        mw = 0.0
        pi_mean = 7.0
        pi_range = 0.5

        # Case 1: Input is a Protein Object
        if isinstance(data, Protein):
            name = data.name
            # Handle class_type being an Enum or None
            if data.class_type:
                p_class = (
                    data.class_type.value
                    if hasattr(data.class_type, "value")
                    else str(data.class_type)
                )
            mw = data.molecular_weight if data.molecular_weight is not None else 0.0
            pi_mean = data.pI_mean if data.pI_mean is not None else 7.0
            pi_range = data.pI_range if data.pI_range is not None else 0.5

        # Case 2: Input is a Dictionary (Legacy/Fallback)
        elif isinstance(data, dict):
            name = data.get("name", "")
            p_class = data.get("class", "None")
            mw = data.get("mw", 0.0)
            pi_mean = data.get("pi_mean", 7.0)
            pi_range = data.get("pi_range", 0.5)

        self.edit_name.setText(name)

        # Set ComboBox Index safely
        index = self.combo_class.findText(p_class, QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.combo_class.setCurrentIndex(index)
        else:
            self.combo_class.setCurrentIndex(0)  # Default to None/First item

        self.spin_mw.setValue(mw)
        self.spin_pi_mean.setValue(pi_mean)
        self.spin_pi_range.setValue(pi_range)

    def get_data(self):
        """Returns the protein configuration as a dictionary matching Protein attributes."""
        return {
            "name": self.edit_name.text().strip(),
            "class": self.combo_class.currentText(),  # Returns string matching Enum value
            "mw": self.spin_mw.value(),
            "pi_mean": self.spin_pi_mean.value(),
            "pi_range": self.spin_pi_range.value(),
        }
