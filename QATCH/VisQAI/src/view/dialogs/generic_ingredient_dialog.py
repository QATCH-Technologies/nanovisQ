from PyQt5 import QtCore, QtWidgets

try:
    from src.models.ingredient import (
        Excipient,
        Ingredient,
        Salt,
        Stabilizer,
        Surfactant,
    )
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.ingredient import (
        Excipient,
        Ingredient,
        Salt,
        Stabilizer,
        Surfactant,
    )


class GenericIngredientDialog(QtWidgets.QDialog):
    """Dialog for configuring generic ingredients (Surfactant, Stabilizer, Excipient, Salt)."""

    def __init__(
        self, ingredient_type, ing_ctrl, existing_ingredient=None, parent=None
    ):
        super().__init__(parent)
        self._parent = parent
        self.ingredient_type = ingredient_type
        self.controller = ing_ctrl
        self.existing_ingredient = existing_ingredient
        is_edit = existing_ingredient is not None

        # Attribute to store result for the caller if needed
        self.result_ingredient = None

        self.setWindowTitle(
            f"Edit {ingredient_type}" if is_edit else f"Add New {ingredient_type}"
        )
        self.resize(350, 150)
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
        lbl_header = QtWidgets.QLabel(f"{ingredient_type} Properties")
        lbl_header.setStyleSheet("font-weight: bold; font-size: 11pt; color: #00adee;")
        layout.addWidget(lbl_header)

        # Form
        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)

        self.edit_name = QtWidgets.QLineEdit()
        self.edit_name.setPlaceholderText("Component name...")
        self.edit_name.setReadOnly(True)
        form_layout.addRow("Name*:", self.edit_name)

        layout.addLayout(form_layout)

        # Load existing data if provided
        if existing_ingredient:
            self._populate_fields(existing_ingredient)

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
        # Connect to custom save handler instead of default accept
        btn_save.clicked.connect(self.save_and_accept)
        btn_layout.addWidget(btn_save)

        layout.addLayout(btn_layout)

    def _get_placeholder(self, ing_type):
        """Returns a helpful placeholder based on type."""
        placeholders = {
            "Salt": "NaCl",
            "Surfactant": "PS80",
            "Stabilizer": "Sucrose",
            "Excipient": "Mannitol",
        }
        return placeholders.get(ing_type, "Component Name")

    def _populate_fields(self, data):
        """Populate fields from either an Ingredient object or a dictionary."""
        name = ""
        if isinstance(data, Ingredient):
            name = data.name
        elif isinstance(data, dict):
            name = data.get("name", "")
        self.edit_name.setText(name)

    def _create_instance(self, name):
        """Factory to create the specific Ingredient subclass instance."""
        # Use -1 for enc_id; controller will assign the correct one on add()
        if self.ingredient_type == "Salt":
            return Salt(enc_id=-1, name=name)
        elif self.ingredient_type == "Surfactant":
            return Surfactant(enc_id=-1, name=name)
        elif self.ingredient_type == "Stabilizer":
            return Stabilizer(enc_id=-1, name=name)
        elif self.ingredient_type == "Excipient":
            return Excipient(enc_id=-1, name=name)
        else:
            raise ValueError(f"Unknown ingredient type: {self.ingredient_type}")

    def save_and_accept(self):
        """Validates input, updates database via controller, and closes dialog."""
        name = self.edit_name.text().strip()

        # 1. Validation
        if not name:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Name is required.")
            return

        try:
            if self.existing_ingredient and isinstance(
                self.existing_ingredient, Ingredient
            ):
                # --- UPDATE EXISTING ---
                # Update the object's local state
                self.existing_ingredient.name = name

                # Persist to DB using the controller
                # Controller.update() takes (id, ingredient_object)
                self.controller.update(
                    self.existing_ingredient.id, self.existing_ingredient
                )
                self.result_ingredient = self.existing_ingredient
                print(self.controller.get_all_ingredients())
            else:
                # --- ADD NEW ---
                new_ing = self._create_instance(name)
                # Controller.add() handles enc_id assignment and DB insertion
                self.result_ingredient = self.controller.add(new_ing)

            # Close dialog with Accepted status
            self.accept()

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Database Error",
                f"Failed to save {self.ingredient_type}:\n{str(e)}",
            )

    def get_data(self):
        """Returns the updated/created ingredient object."""
        # Return the actual object if available, otherwise fallback to dict
        if self.result_ingredient:
            return self.result_ingredient
        return {"name": self.edit_name.text().strip()}
