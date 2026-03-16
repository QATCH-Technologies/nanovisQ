"""
generic_ingredient_dialog.py

Provides a polymorphic configuration dialog for various ingredient types.

This module defines the GenericIngredientDialog, which serves as a unified
interface for creating or editing Surfactants, Stabilizers, Excipients, and Salts.
It utilizes a factory pattern to instantiate the correct model subclass based
on the provided type string.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

from PyQt5 import QtWidgets

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
    """A modal dialog for configuring properties of generic ingredient types.

    This dialog dynamically adapts its title and labels based on the
    `ingredient_type`. It handles both the creation of new ingredient
    instances and the modification of existing ones via an ingredient controller.

    Attributes:
        ingredient_type (str): The category of ingredient (e.g., 'Salt', 'Surfactant').
        controller: The ingredient controller responsible for persistence logic.
        existing_ingredient (Ingredient, optional): An existing instance if in edit mode.
        result_ingredient (Ingredient, optional): The ingredient object after
            it has been saved or updated.
    """

    def __init__(
        self,
        ingredient_type,
        ing_ctrl,
        existing_ingredient=None,
        parent=None,
    ):
        """Initializes the dialog with the specific type and data context.

        Args:
            ingredient_type (str): The specific type of ingredient to configure.
            ing_ctrl: The controller used to interface with the database.
            existing_ingredient (Ingredient, optional): An ingredient instance
                to edit. Defaults to None.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self._parent = parent
        self.ingredient_type = ingredient_type
        self.controller = ing_ctrl
        self.existing_ingredient = existing_ingredient
        is_edit = existing_ingredient is not None
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
        btn_save.clicked.connect(self.save_and_accept)
        btn_layout.addWidget(btn_save)
        layout.addLayout(btn_layout)

    def _get_placeholder(self, ing_type):
        """Returns a helpful placeholder based on type.

        Args:
            ing_type (str): The ingredient category.

        Returns:
            str: A common example name for the given category.
        """
        placeholders = {
            "Salt": "NaCl",
            "Surfactant": "PS80",
            "Stabilizer": "Sucrose",
            "Excipient": "Mannitol",
        }
        return placeholders.get(ing_type, "Component Name")

    def _populate_fields(self, data):
        """Populate form fields from either an Ingredient object or a dictionary.

        Args:
            data (Ingredient or dict): The data source to load into the UI.
        """
        name = ""
        if isinstance(data, Ingredient):
            name = data.name
        elif isinstance(data, dict):
            name = data.get("name", "")
        self.edit_name.setText(name)

    def _create_instance(self, name):
        """Factory to create the specific Ingredient subclass instance.

        Args:
            name (str): The name of the ingredient.

        Returns:
            Ingredient: A new instance of Salt, Surfactant, Stabilizer, or Excipient.

        Raises:
            ValueError: If the `ingredient_type` does not match a known subclass.
        """
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
        """Validates input, updates database via controller, and closes dialog.

        In edit mode, this updates the existing object's state and persists it.
        In add mode, it creates a new instance via the factory and adds it
        to the controller. Displays an error message if the database operation fails.
        """
        name = self.edit_name.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Name is required.")
            return

        try:
            if self.existing_ingredient and isinstance(
                self.existing_ingredient, Ingredient
            ):
                self.existing_ingredient.name = name
                self.controller.update(
                    self.existing_ingredient.id, self.existing_ingredient
                )
                self.result_ingredient = self.existing_ingredient
                print(self.controller.get_all_ingredients())
            else:
                new_ing = self._create_instance(name)
                self.result_ingredient = self.controller.add(new_ing)

            self.accept()

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Database Error",
                f"Failed to save {self.ingredient_type}:\n{str(e)}",
            )

    def get_data(self):
        """Returns the updated or created ingredient object.

        Returns:
            Ingredient or dict: The resulting ingredient object if saved,
                otherwise a dictionary containing the current form name.
        """
        if self.result_ingredient:
            return self.result_ingredient
        return {"name": self.edit_name.text().strip()}
