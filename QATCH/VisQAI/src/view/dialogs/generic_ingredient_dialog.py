from PyQt5 import QtWidgets


class GenericIngredientDialog(QtWidgets.QDialog):
    """Dialog for configuring generic ingredients (Surfactant, Stabilizer, Excipient, Salt)."""

    def __init__(self, ingredient_type, existing_ingredient=None, parent=None):
        super().__init__(parent)
        self.ingredient_type = ingredient_type
        is_edit = existing_ingredient is not None
        self.setWindowTitle(
            f"Edit {ingredient_type}" if is_edit else f"Add New {ingredient_type}"
        )
        self.resize(350, 150)
        self.setModal(True)

        self.setStyleSheet(
            """
            QDialog { background-color: #ffffff; }
            QLabel { color: #333; }
            QLineEdit { 
                border: 1px solid #d1d5db; 
                border-radius: 4px; 
                padding: 6px; 
                min-height: 24px;
                background-color: #ffffff;
            }
            QLineEdit:focus {
                border: 1px solid #00adee;
            }
        """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        lbl_header = QtWidgets.QLabel(f"{ingredient_type} Properties")
        lbl_header.setStyleSheet("font-weight: bold; font-size: 11pt; color: #00adee;")
        layout.addWidget(lbl_header)

        # Form
        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)

        self.edit_name = QtWidgets.QLineEdit()
        self.edit_name.setPlaceholderText(f"Enter {ingredient_type.lower()} name")
        form_layout.addRow("Name*:", self.edit_name)

        layout.addLayout(form_layout)

        # Load existing data if provided
        if existing_ingredient:
            self.edit_name.setText(existing_ingredient.get("name", ""))

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
        """Returns the ingredient configuration as a dictionary."""
        return {"name": self.edit_name.text().strip()}
