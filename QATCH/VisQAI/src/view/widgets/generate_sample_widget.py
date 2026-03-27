"""
generate_sample_widget.py

Provides a user interface for defining formulation constraints and generating samples.

This module contains the GenerateSampleWidget, which allows users to set up
design-of-experiment (DoE) parameters. It features a dynamic constraint builder
where users can mix categorical and numeric filters to guide the sample
generation engine.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

import glob
import os
import shutil

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

try:
    from src.view.architecture import Architecture
    from src.view.components.checkable_combo_box import CheckableComboBox
    from src.view.dialogs.model_selection_dialog import (
        ModelSelectionDialog,
    )
except (ImportError, ModuleNotFoundError):
    from QATCH.common.architecture import Architecture
    from QATCH.VisQAI.src.view.components.checkable_combo_box import CheckableComboBox
    from QATCH.VisQAI.src.view.dialogs.model_selection_dialog import (
        ModelSelectionDialog,
    )


class CompactCheckableComboBox(CheckableComboBox):
    """A CheckableComboBox subclass that summarizes selections to prevent crowding.

    Overrides the paint event to display "Select...", the single item name,
    or "X selected" based on the number of checked items in the dropdown.
    """

    def paintEvent(self, event):
        """Custom paint event to render summary text instead of a long comma-separated list.

        Args:
            event (QPaintEvent): The paint event triggered by Qt.
        """
        painter = QtGui.QPainter(self)
        opt = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(opt)

        try:
            from PyQt5.QtCore import Qt

            model = self.model()
            checked = []
            for i in range(model.rowCount()):
                item = model.item(i)
                if item is not None and item.checkState() == Qt.Checked:
                    checked.append(item.text())
            if not checked:
                opt.currentText = "Select..."
            elif len(checked) == 1:
                opt.currentText = checked[0]
            else:
                opt.currentText = f"{len(checked)} selected"
        except Exception:
            pass

        self.style().drawComplexControl(
            QtWidgets.QStyle.CC_ComboBox, opt, painter, self
        )
        self.style().drawControl(QtWidgets.QStyle.CE_ComboBoxLabel, opt, painter, self)


class GenerateSampleWidget(QtWidgets.QFrame):
    """An overlay panel for configuring and initiating sample generation.

    This widget manages a list of dynamic constraint rows. Each row allows
    the user to filter by ingredient type, specific attributes, and conditions.
    The UI automatically toggles between a multi-select dropdown and a
    numeric spinbox based on the chosen attribute.

    Attributes:
        generate_requested (QtCore.pyqtSignal): Emits (int, str, list) containing
            the sample count, model filename, and a list of constraint dicts.
        closed (QtCore.pyqtSignal): Emitted when the widget is hidden.
        resized (QtCore.pyqtSignal): Emitted when the internal layout size changes.
        ingredients_by_type (dict): A mapping of ingredient categories to
            their corresponding model objects.
        constraint_rows (list[dict]): Internal storage for the widgets and
            logic maps associated with each added constraint row.
    """

    generate_requested = QtCore.pyqtSignal(int, str, list)
    closed = QtCore.pyqtSignal()
    resized = QtCore.pyqtSignal()

    def __init__(self, ingredients_by_type, parent=None):
        """Initializes the generation widget with styling and ingredient data.

        Args:
            ingredients_by_type (dict): Data source for populating ingredient
                and class filters.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.ingredients_by_type = ingredients_by_type

        self.assets_path = os.path.join(
            Architecture.get_path(), "QATCH", "VisQAI", "assets"
        )
        if not os.path.exists(self.assets_path):
            os.makedirs(self.assets_path, exist_ok=True)

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("class", "card")

        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setYOffset(10)
        shadow.setColor(QtGui.QColor(0, 0, 0, 40))
        self.setGraphicsEffect(shadow)

        self.setVisible(False)
        self.setMinimumWidth(600)

        self.constraint_rows = []
        self._init_ui()

    def _init_ui(self):
        """Builds the primary UI structure, including the header and scroll area."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 20)
        layout.setSpacing(15)

        # Header
        header = QtWidgets.QHBoxLayout()
        lbl_title = QtWidgets.QLabel("Generate Samples")
        lbl_title.setObjectName("evalTitle")
        header.addWidget(lbl_title)

        btn_close = QtWidgets.QToolButton()
        btn_close.setObjectName("btnEvalClose")
        btn_close.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "close-circle-svgrepo-com.svg",
                )
            )
        )
        btn_close.setIconSize(QtCore.QSize(18, 18))
        btn_close.setFixedSize(24, 24)
        btn_close.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn_close.clicked.connect(self.close_widget)

        header.addStretch()
        header.addWidget(btn_close)
        layout.addLayout(header)

        # Configuration Group
        grp_settings = QtWidgets.QGroupBox("Configuration")
        settings_layout = QtWidgets.QFormLayout(grp_settings)
        settings_layout.setSpacing(12)

        # Model Selector
        model_layout = QtWidgets.QHBoxLayout()
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setStyleSheet("background-color: #ffffff; height: 26px;")
        self.model_combo.setToolTip("Select a prediction model from assets")
        self._populate_model_list()

        self.btn_select_model = QtWidgets.QPushButton()
        self.btn_select_model.setFixedWidth(40)
        self.btn_select_model.setFixedHeight(26)
        self.btn_select_model.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_select_model.setToolTip("Import New Model (.visq)")
        self.btn_select_model.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "file-plus-2-svgrepo-com.svg",
                )
            )
        )
        self.btn_select_model.clicked.connect(self.browse_model_file)

        model_layout.addWidget(self.model_combo, stretch=1)
        model_layout.addWidget(self.btn_select_model)

        settings_layout.addRow("Model:", model_layout)

        # Number of Samples
        self.spin_samples = QtWidgets.QSpinBox()
        self.spin_samples.setRange(1, 100)
        self.spin_samples.setValue(5)
        self.spin_samples.setFixedWidth(100)
        self.spin_samples.setFixedHeight(26)

        settings_layout.addRow("Number of Samples:", self.spin_samples)
        layout.addWidget(grp_settings)

        #  Group with Scroll Area
        self.grp_constraints = QtWidgets.QGroupBox("Constraints")
        grp_constraints_layout = QtWidgets.QVBoxLayout(self.grp_constraints)
        grp_constraints_layout.setContentsMargins(15, 15, 15, 15)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.constraints_container = QtWidgets.QWidget()
        self.constraints_layout = QtWidgets.QVBoxLayout(self.constraints_container)
        self.constraints_layout.setContentsMargins(0, 0, 5, 0)
        self.constraints_layout.setSpacing(8)

        self.lbl_none = QtWidgets.QLabel("No constraints added.")
        self.lbl_none.setStyleSheet("color: #6b7280; font-style: italic;")
        self.constraints_layout.addWidget(self.lbl_none)
        self.constraints_layout.addStretch()

        self.scroll_area.setWidget(self.constraints_container)
        grp_constraints_layout.addWidget(self.scroll_area)

        layout.addWidget(self.grp_constraints)

        # Footer Actions
        layout.addSpacing(5)
        btn_layout = QtWidgets.QHBoxLayout()

        self.btn_add_constraint = QtWidgets.QPushButton("+ Add Constraint")
        self.btn_add_constraint.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_add_constraint.setFixedHeight(34)
        self.btn_add_constraint.clicked.connect(self.add_constraint_row)

        self.btn_generate = QtWidgets.QPushButton("Generate")
        self.btn_generate.setObjectName("btnApplyFilters")
        self.btn_generate.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_generate.setFixedHeight(34)
        self.btn_generate.setFixedWidth(140)
        self.btn_generate.clicked.connect(self.emit_generate)

        btn_layout.addWidget(self.btn_add_constraint)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_generate)
        layout.addLayout(btn_layout)

        self._update_scroll_height()
        self._validate_rows()

    def _populate_model_list(self):
        """Scans the assets directory for .visq files and populates the model selector."""
        self.model_combo.clear()
        pattern = os.path.join(self.assets_path, "*.visq")
        files = glob.glob(pattern)
        files.sort()

        if not files:
            self.model_combo.addItem("No models found")
            self.model_combo.setEnabled(False)
        else:
            self.model_combo.setEnabled(True)
            for f in files:
                filename = os.path.basename(f)
                self.model_combo.addItem(filename, f)
            index = self.model_combo.findText("VisQAI(base).visq")
            if index >= 0:
                self.model_combo.setCurrentIndex(index)

    def browse_model_file(self):
        """Opens a file dialog to import a .visq model into the local assets folder."""
        fname = None
        try:
            model_dialog = ModelSelectionDialog()
            model_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            model_dialog.setNameFilter("VisQAI Models (*.visq)")
            model_dialog.setViewMode(QtWidgets.QFileDialog.Detail)

            model_path = os.path.join(
                Architecture.get_path(), "QATCH", "VisQAI", "assets"
            )
            if os.path.exists(model_path):
                model_dialog.setDirectory(model_path)

            if model_dialog.exec_():
                selected_files = model_dialog.selectedFiles()
                if selected_files:
                    fname = selected_files[0]
        except Exception:
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Import Model File", "", "VisQAI Models (*.visq)"
            )
        if fname:
            try:
                filename = os.path.basename(fname)
                dest_path = os.path.join(self.assets_path, filename)
                if os.path.abspath(fname) != os.path.abspath(dest_path):
                    shutil.copy2(fname, dest_path)

                self._populate_model_list()
                index = self.model_combo.findText(filename)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Import Failed", f"Could not import model:\n{str(e)}"
                )

    def _update_scroll_height(self):
        """Dynamically resizes the scroll area based on the number of constraint rows."""
        row_height = 36
        num_rows = len(self.constraint_rows)

        if num_rows == 0:
            self.scroll_area.setFixedHeight(30)
        else:
            visible_rows = min(num_rows, 3)
            target_height = (visible_rows * row_height) + 10
            self.scroll_area.setFixedHeight(target_height)

        self.adjustSize()
        self.resized.emit()

    @staticmethod
    def _checked_items(combo_box) -> list:
        """Retrieves a list of checked item texts from a CheckableComboBox.

        Args:
            combo_box (CheckableComboBox): The widget to inspect.

        Returns:
            list[str]: The texts of all items currently checked.
        """

        checked = []
        model = combo_box.model()
        for i in range(model.rowCount()):
            item = model.item(i)
            if item is not None and item.checkState() == Qt.Checked:
                checked.append(item.text())
        return checked

    def _validate_rows(self):
        """Checks if all constraint rows are complete to enable the Generate button."""
        if not self.constraint_rows:
            self.btn_add_constraint.setEnabled(True)
            self.btn_generate.setEnabled(self.model_combo.isEnabled())
            return

        all_complete = True

        for row in self.constraint_rows:
            ing_valid = row["ingredient"].currentIndex() > 0
            attr_valid = row["attribute"].currentIndex() > 0
            cond_valid = row["condition"].currentIndex() > 0

            val_stack = row["value_stack"]

            if val_stack.currentIndex() == 0:
                val_box = row["value_cb"]
                val_valid = len(self._checked_items(val_box)) > 0
            else:
                val_valid = True

            if not (ing_valid and attr_valid and cond_valid and val_valid):
                all_complete = False
                break

        self.btn_add_constraint.setEnabled(all_complete)
        self.btn_generate.setEnabled(all_complete and self.model_combo.isEnabled())

    def add_constraint_row(self):
        """Adds a new row of configuration widgets to the constraint layout."""
        self.lbl_none.hide()

        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        combo_style = "background-color: #ffffff; height: 26px; border: 1px solid #d1d5db; border-radius: 4px;"

        # Ingredient
        cb_ingredient = QtWidgets.QComboBox()
        cb_ingredient.addItem("Ingredient...")
        cb_ingredient.model().item(0).setEnabled(False)
        cb_ingredient.addItems(
            ["Protein", "Buffer", "Surfactant", "Stabilizer", "Salt", "Excipient"]
        )
        cb_ingredient.setStyleSheet(combo_style)

        # Attribute (Type, Concentration, [Class])
        cb_attribute = QtWidgets.QComboBox()
        cb_attribute.addItem("Attribute...")
        cb_attribute.model().item(0).setEnabled(False)
        cb_attribute.setStyleSheet(combo_style)

        # Condition (is, is not, >, >=, =, <=, <, !=)
        cb_condition = QtWidgets.QComboBox()
        cb_condition.addItem("Condition...")
        cb_condition.model().item(0).setEnabled(False)
        cb_condition.setStyleSheet(combo_style)

        # Value Stack
        val_stack = QtWidgets.QStackedWidget()

        cb_value = CompactCheckableComboBox()
        cb_value.setStyleSheet(combo_style)

        spin_value = QtWidgets.QDoubleSpinBox()
        spin_value.setStyleSheet(combo_style)
        spin_value.setRange(0.0, 10000.0)
        spin_value.setDecimals(3)
        spin_value.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)

        val_stack.addWidget(cb_value)
        val_stack.addWidget(spin_value)

        btn_delete = QtWidgets.QToolButton()
        btn_delete.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "delete-2-svgrepo-com.svg",
                )
            )
        )
        btn_delete.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn_delete.setStyleSheet("border: none;")

        row_layout.addWidget(cb_ingredient)
        row_layout.addWidget(cb_attribute)
        row_layout.addWidget(cb_condition)
        row_layout.addWidget(val_stack, stretch=1)
        row_layout.addWidget(btn_delete)

        self.constraints_layout.insertWidget(len(self.constraint_rows), row_widget)

        row_data = {
            "widget": row_widget,
            "ingredient": cb_ingredient,
            "attribute": cb_attribute,
            "condition": cb_condition,
            "value_stack": val_stack,
            "value_cb": cb_value,
            "value_spin": spin_value,
        }
        self.constraint_rows.append(row_data)

        # Cascading Logic
        btn_delete.clicked.connect(lambda: self.remove_constraint_row(row_data))
        cb_ingredient.currentIndexChanged.connect(
            lambda: self._on_ingredient_changed(row_data)
        )
        cb_attribute.currentIndexChanged.connect(
            lambda: self._on_attribute_changed(row_data)
        )
        cb_condition.currentIndexChanged.connect(self._validate_rows)

        # Ensure validation triggers when values change
        cb_value.model().dataChanged.connect(self._validate_rows)
        spin_value.valueChanged.connect(lambda: self._validate_rows())

        self._update_scroll_height()
        self._validate_rows()

    def _on_ingredient_changed(self, row_data):
        ing_type = row_data["ingredient"].currentText()
        cb_attribute = row_data["attribute"]

        cb_attribute.blockSignals(True)
        cb_attribute.clear()
        cb_attribute.addItem("Attribute...")
        cb_attribute.model().item(0).setEnabled(False)

        if row_data["ingredient"].currentIndex() > 0:
            attrs = ["Type", "Concentration"]
            if ing_type == "Protein":
                attrs.append("Class")
            cb_attribute.addItems(attrs)

        cb_attribute.setCurrentIndex(0)
        cb_attribute.blockSignals(False)

        self._on_attribute_changed(row_data)

    def _on_attribute_changed(self, row_data):
        """Toggles the condition and value widgets based on numeric vs categorical choice.

        Args:
            row_data (dict): The dictionary of widgets for the specific row.
        """
        attr_type = row_data["attribute"].currentText()
        cb_condition = row_data["condition"]
        val_stack = row_data["value_stack"]

        cb_condition.blockSignals(True)
        cb_condition.clear()
        cb_condition.addItem("Condition...")
        cb_condition.model().item(0).setEnabled(False)

        if row_data["attribute"].currentIndex() > 0:
            if attr_type == "Concentration":
                cb_condition.addItems([">", ">=", "=", "!=", "<=", "<"])
                val_stack.setCurrentIndex(1)
            elif attr_type in ["Type", "Class"]:
                cb_condition.addItems(["is", "is not"])
                val_stack.setCurrentIndex(0)

        cb_condition.setCurrentIndex(0)
        cb_condition.blockSignals(False)

        self._populate_values(row_data)

    def _populate_values(self, row_data):
        """Populates the multi-select dropdown with specific item names or classes.

        Args:
            row_data (dict): The dictionary of widgets for the specific row.
        """
        ing_idx = row_data["ingredient"].currentIndex()
        attr_idx = row_data["attribute"].currentIndex()
        val_stack = row_data["value_stack"]
        val_cb = row_data["value_cb"]

        if ing_idx <= 0 or attr_idx <= 0:
            if val_stack.currentIndex() == 0:
                val_cb.clear()
            self._validate_rows()
            return

        ing_type = row_data["ingredient"].currentText()
        attr_type = row_data["attribute"].currentText()
        if attr_type in ["Type", "Class"]:
            val_cb.clear()

            items = []
            if attr_type == "Class" and ing_type == "Protein":
                classes = set()
                for p in self.ingredients_by_type.get("Protein", []):
                    if hasattr(p, "class_type") and p.class_type:
                        c_val = str(
                            getattr(
                                p.class_type,
                                "value",
                                getattr(p.class_type, "name", str(p.class_type)),
                            )
                        )
                        if c_val != "-":
                            classes.add(c_val)
                items = sorted(list(classes))
            else:
                items = [obj.name for obj in self.ingredients_by_type.get(ing_type, [])]
                if ing_type not in ["Protein", "Buffer"]:
                    if "None" not in items:
                        items.insert(0, "None")

            val_cb.addItems(items)

        self._validate_rows()

    def remove_constraint_row(self, row_data):
        """Removes a constraint row from the layout and storage.

        Args:
            row_data (dict): The dictionary of widgets for the specific row.
        """
        row_data["widget"].deleteLater()
        self.constraint_rows.remove(row_data)
        if not self.constraint_rows:
            self.lbl_none.show()
        self._update_scroll_height()
        self._validate_rows()

    def emit_generate(self):
        """Aggregates all row data into a list of dicts and emits the generate signal."""
        constraints_data = []
        for row in self.constraint_rows:
            val_stack = row["value_stack"]
            if val_stack.currentIndex() == 0:
                val = self._checked_items(row["value_cb"])
            else:
                val = row["value_spin"].value()

            constraints_data.append(
                {
                    "ingredient": row["ingredient"].currentText(),
                    "attribute": row["attribute"].currentText(),
                    "condition": row["condition"].currentText(),
                    "values": val,
                }
            )

        model_file = self.model_combo.currentText()
        self.generate_requested.emit(
            self.spin_samples.value(), model_file, constraints_data
        )

    def close_widget(self):
        """Hides the panel and emits the closed signal."""
        self.hide()
        self.closed.emit()
