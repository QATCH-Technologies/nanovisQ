from PyQt5 import QtWidgets


class ProteinConfigDialog(QtWidgets.QDialog):
    """Dialog for configuring protein ingredients."""

    def __init__(self, existing_protein=None, parent=None):
        super().__init__(parent)
        is_edit = existing_protein is not None
        self.setWindowTitle("Edit Protein" if is_edit else "Add New Protein")
        self.resize(400, 350)
        self.setModal(True)

        self.setStyleSheet(
            """
            QDialog { background-color: #ffffff; }
            QLabel { color: #333; }
            QLineEdit, QDoubleSpinBox { 
                border: 1px solid #d1d5db; 
                border-radius: 4px; 
                padding: 6px; 
                min-height: 24px;
                background-color: #ffffff;
            }
            QLineEdit:focus, QDoubleSpinBox:focus {
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

        self.edit_class = QtWidgets.QLineEdit()
        self.edit_class.setPlaceholderText("e.g., IgG1")
        form_layout.addRow("Class:", self.edit_class)

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

        # Load existing data if provided
        if existing_protein:
            self.edit_name.setText(existing_protein.get("name", ""))
            self.edit_class.setText(existing_protein.get("class", ""))
            self.spin_mw.setValue(existing_protein.get("mw", 0))
            self.spin_pi_mean.setValue(existing_protein.get("pi_mean", 7.0))
            self.spin_pi_range.setValue(existing_protein.get("pi_range", 0.5))

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

    def get_data(self):
        """Returns the protein configuration as a dictionary."""
        return {
            "name": self.edit_name.text().strip(),
            "class": self.edit_class.text().strip(),
            "mw": self.spin_mw.value(),
            "pi_mean": self.spin_pi_mean.value(),
            "pi_range": self.spin_pi_range.value(),
        }
