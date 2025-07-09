try:
    from QATCH.core.constants import Constants
    from QATCH.common.logger import Logger as Log
    from QATCH.common.architecture import Architecture
except:
    print("Running VisQAI as standalone app")

    class Log:
        def d(tag, msg=""): print("DEBUG:", tag, msg)
        def i(tag, msg=""): print("INFO:", tag, msg)
        def w(tag, msg=""): print("WARNING:", tag, msg)
        def e(tag, msg=""): print("ERROR:", tag, msg)

from PyQt5 import QtCore, QtGui, QtWidgets
import os
from typing import Dict, Any, List, Tuple, Type
from typing import TYPE_CHECKING

try:
    from src.utils.constraints import Constraints
    from src.utils.icon_utils import IconUtils
    from src.view.checkable_combo_box import CheckableComboBox
    if TYPE_CHECKING:
        from src.models.ingredient import Ingredient
        from src.view.main_window import VisQAIWindow

except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.utils.constraints import Constraints
    from QATCH.VisQAI.src.utils.icon_utils import IconUtils
    from QATCH.VisQAI.src.view.checkable_combo_box import CheckableComboBox
    if TYPE_CHECKING:
        from QATCH.VisQAI.src.models.ingredient import Ingredient
        from QATCH.VisQAI.src.view.main_window import VisQAIWindow


class ConstraintsUI(QtWidgets.QWidget):
    def __init__(self, parent, step):
        super().__init__(parent)
        self.parent = parent
        self.step = step
        self.main_window: VisQAIWindow = parent.parent

        self.suggest_dialog = QtWidgets.QDialog(self)
        if self.step == 2:
            self.suggest_dialog.setWindowTitle("Add Suggestion(s)")
        elif self.step == 6:
            self.suggest_dialog.setWindowTitle("Add Optimize Feature(s)")
        self.suggest_dialog.setModal(True)
        # hide question mark from title bar of window
        self.suggest_dialog.setWindowFlag(
            QtCore.Qt.WindowContextHelpButtonHint, False)
        self.suggest_dialog.setMinimumSize(500, 300)
        self.suggest_dialog.setWindowIcon(QtGui.QIcon(
            os.path.join(Architecture.get_path(), 'QATCH/icons/qmodel.png')))

        layout = QtWidgets.QVBoxLayout(self.suggest_dialog)

        if self.step == 2:

            label = QtWidgets.QLabel(
                "How many suggestions do you want to add?")
            layout.addWidget(label)

            self.suggestion_text = QtWidgets.QComboBox(self.suggest_dialog)
            self.suggestion_text.addItems(
                list(map(str, range(1, 11))))  # 1 to 10 suggestions
            self.suggestion_text.setEditable(True)
            layout.addWidget(self.suggestion_text)

        self.constraints_group = QtWidgets.QGroupBox(
            "Constraints", self.suggest_dialog)
        self.constraints_layout = QtWidgets.QVBoxLayout(self.constraints_group)

        self.constraints_none = QtWidgets.QLabel(
            "None", self.constraints_group)
        self.constraints_none.setToolTip("No constraints on the suggestions")
        self.constraints_layout.addWidget(self.constraints_none)

        self.constraints_rows: List[QtWidgets.QHBoxLayout] = []
        self.constraints_ingredients: List[QtWidgets.QComboBox] = []
        self.constraints_features: List[QtWidgets.QComboBox] = []
        self.constraints_verbs: List[QtWidgets.QComboBox] = []
        self.constraints_values: List[CheckableComboBox] = []
        self.constraints_delete_buttons: List[QtWidgets.QPushButton] = []

        layout.addWidget(self.constraints_group)

        self.add_constraints_btn = QtWidgets.QPushButton(
            icon=IconUtils.rotate_and_crop_icon(QtGui.QIcon(
                os.path.join(Architecture.get_path(), 'QATCH/icons/cancel.png')), 45, 50),
            text="   Add Constraint",
            parent=self.suggest_dialog)
        self.add_constraints_btn.setToolTip(
            "Add a new constraint for the suggestions")
        self.add_constraints_btn.clicked.connect(self.add_new_constraint)
        layout.addWidget(self.add_constraints_btn)

        layout.addStretch(1)  # add stretch to push buttons to the bottom

        button_box = QtWidgets.QDialogButtonBox(self.suggest_dialog)
        button_box.setStandardButtons(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        button_box.accepted.connect(self.accept_suggestions)
        button_box.rejected.connect(self.suggest_dialog.reject)

    def add_suggestion_dialog(self):
        self.suggest_dialog.open()  # non-blocking

    def accept_suggestions(self):
        Log.d("Accepting suggestions...")
        if self.step == 2:
            num_suggestions = self.suggestion_text.currentText()
            if not num_suggestions.isdigit() or int(num_suggestions) < 1:
                Log.w("Invalid number of suggestions:", num_suggestions)
                return
        constraints = self.build_constraints()
        if not constraints:
            Log.w(
                "Missing/invalid fields in Constraints. Please fill out or remove them.")
            return

        self.suggest_dialog.accept()  # Hide the dialog

        Log.d("Constraints:", constraints.build())
        if self.step == 2:
            for _ in range(int(num_suggestions)):
                self.parent.load_suggestion(
                    constraints)  # Add a new suggestion
                while self.parent.timer.isActive():
                    QtWidgets.QApplication.processEvents()
                if self.parent.progressBar.wasCanceled():
                    Log.d("User canceled adding suggestions. Stopping.")
                    break
        if self.step == 6:
            # For optimization, we just need to build the constraints
            # and pass them to the optimizer
            self.parent.set_constraints(constraints)

    def add_new_constraint(self):
        self.constraints_none.setVisible(False)  # hide "None" label

        # Create a new row for the constraint
        self.constraints_rows.append(QtWidgets.QHBoxLayout())
        # The constraint ingredient can be one of the following:
        #   Protein, Buffer, Surfactant, Stabilizer, Salt
        self.constraints_ingredients.append(
            QtWidgets.QComboBox(self.constraints_group))
        self.constraints_ingredients[-1].addItems([
            "Protein", "Buffer", "Surfactant", "Stabilizer", "Salt"])
        # No selection by default
        self.constraints_ingredients[-1].setCurrentIndex(-1)
        self.constraints_ingredients[-1].currentIndexChanged.connect(
            lambda index, idx=len(self.constraints_ingredients)-1: self.autofill_constraint_values(idx))
        # When ingredient changes, autofill possible values
        self.constraints_rows[-1].addWidget(self.constraints_ingredients[-1])
        # The constraint feature can be one of the following:
        #   Type, Concentration
        self.constraints_features.append(
            QtWidgets.QComboBox(self.constraints_group))
        self.constraints_features[-1].addItems([
            "Type", "Concentration"])
        # No selection by default
        self.constraints_features[-1].setCurrentIndex(-1)
        self.constraints_features[-1].currentIndexChanged.connect(
            lambda index, idx=len(self.constraints_features)-1: self.autofill_constraint_values(idx))
        # When feature changes, autofill possible values
        self.constraints_rows[-1].addWidget(self.constraints_features[-1])
        # The constraint verb can be one of the following:
        #   is, is not
        self.constraints_verbs.append(
            QtWidgets.QComboBox(self.constraints_group))
        self.constraints_verbs[-1].addItems(["is", "is not"])
        # No selection by default
        self.constraints_verbs[-1].setCurrentIndex(-1)
        self.constraints_verbs[-1].currentIndexChanged.connect(
            lambda index, idx=len(self.constraints_verbs)-1: self.autofill_constraint_values(idx))
        self.constraints_rows[-1].addWidget(self.constraints_verbs[-1])
        # The constraint value can be a single, multiple or range of values
        # (i.e. "PBS", "tween-20,tween-80" or "0.01-0.2")
        self.constraints_values.append(
            CheckableComboBox(self.constraints_group))
        # self.constraints_values[-1].setEditable(True)
        # TODO: Add items for now just for debugging the combination selection
        # self.constraints_values[-1].addItems([
        #     "PBS", "tween-20", "tween-80", "0.01", "0.02", "0.03",
        #     "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.1",
        #     "0.2", "0.3", "0.4", "0.5"])
        # No selection by default
        self.constraints_values[-1].setCurrentIndex(-1)
        # with stretch
        self.constraints_rows[-1].addWidget(self.constraints_values[-1], 1)
        # Delete button to clear this constraint from the list
        self.constraints_delete_buttons.append(QtWidgets.QToolButton(
            icon=QtGui.QIcon(
                os.path.join(Architecture.get_path(), 'QATCH/icons/cancel.png')),
            text=None,
            parent=self.constraints_group))
        self.constraints_delete_buttons[-1].clicked.connect(
            lambda: self.remove_constraint(len(self.constraints_delete_buttons) - 1))
        self.constraints_delete_buttons[-1].setFixedWidth(
            self.constraints_delete_buttons[-1].sizeHint().height())  # make it square
        self.constraints_delete_buttons[-1].setToolTip(
            "Delete this constraint")
        self.constraints_delete_buttons[-1].setCursor(
            QtCore.Qt.PointingHandCursor)
        # Add the delete button to the row
        self.constraints_rows[-1].addWidget(
            self.constraints_delete_buttons[-1])
        self.constraints_layout.addLayout(self.constraints_rows[-1])

    def autofill_constraint_values(self, index: int):
        if index < 0 or index >= len(self.constraints_ingredients) or \
                index >= len(self.constraints_features) or \
                index >= len(self.constraints_verbs) or \
                index >= len(self.constraints_values):
            Log.e("Invalid constraint index:", index)
            return

        ingredient = self.constraints_ingredients[index].currentText()
        feature = self.constraints_features[index].currentText()
        verb = self.constraints_verbs[index].currentText()

        model = self.constraints_values[index].model()
        current_items = [model.data(model.index(i, 0))
                         for i in range(model.rowCount())]

        if not ingredient or not feature or not verb:
            return  # nothing selected yet

        autofill_items = []
        editable = False
        if feature == "Type":
            if ingredient == "Protein":
                autofill_items = self.parent.proteins.copy()
            elif ingredient == "Buffer":
                autofill_items = self.parent.buffers.copy()
            elif ingredient == "Surfactant":
                autofill_items = self.parent.surfactants.copy()
            elif ingredient == "Stabilizer":
                autofill_items = self.parent.stabilizers.copy()
            elif ingredient == "Salt":
                autofill_items = self.parent.salts.copy()
            if ingredient not in ["Protein", "Buffer"] and "None" not in autofill_items:
                autofill_items.insert(0, "None")  # allow "none" selection
        elif feature == "Concentration":
            editable = True

        # Clear and set new items or placeholder (if different)
        if autofill_items != current_items or editable != self.constraints_values[index].isEditable():
            self.constraints_values[index].clear()
            self.constraints_values[index].setEditable(editable)
            if editable:
                self.constraints_values[index].update_label(autofill_items)
            else:
                self.constraints_values[index].addItems(autofill_items)
                self.constraints_values[index].setCurrentIndex(-1)

    def remove_constraint(self, index: int):
        if index < 0 or index >= len(self.constraints_delete_buttons):
            Log.e("Invalid constraint index:", index)
            return

        # Remove the constraint from the layout and delete the widgets
        self.constraints_rows[index].setParent(None)
        self.constraints_rows[index].deleteLater()
        del self.constraints_rows[index]
        self.constraints_ingredients[index].setParent(None)
        self.constraints_ingredients[index].deleteLater()
        del self.constraints_ingredients[index]
        self.constraints_features[index].setParent(None)
        self.constraints_features[index].deleteLater()
        del self.constraints_features[index]
        self.constraints_verbs[index].setParent(None)
        self.constraints_verbs[index].deleteLater()
        del self.constraints_verbs[index]
        self.constraints_values[index].setParent(None)
        self.constraints_values[index].deleteLater()
        del self.constraints_values[index]
        self.constraints_delete_buttons[index].setParent(None)
        self.constraints_delete_buttons[index].deleteLater()
        del self.constraints_delete_buttons[index]

        if len(self.constraints_rows) == 0:
            # If no constraints left, show the "None" label again
            self.constraints_none.setVisible(True)

    def build_constraints(self) -> Constraints | None:
        added_constraints = []
        for i in range(len(self.constraints_ingredients)):
            ingredient = self.constraints_ingredients[i].currentText()
            feature = self.constraints_features[i].currentText()
            verb = self.constraints_verbs[i].currentText()
            values = self.constraints_values[i].currentText()
            if not ingredient or not feature or not verb or not values:
                return None  # missing fields

            # Split values by semicolon and strip whitespace
            values = [v.strip() for v in values.split(";") if v.strip()]
            if not values:
                return None  # no valid values

            if feature == "Concentration":
                # If feature is "Concentration", values can be a single value or a range
                # Check if value contains a range (e.g. "0.01-0.1") and convert to tuple (0.01, 0.1)
                # Otherwise, convert the single value to float
                # If conversion fails, return None
                for j in range(len(values)):
                    if "-" in values[j]:
                        try:
                            # NOTE: This will not handle negative values correctly
                            start, end = values[j].split("-")
                            start = float(start.strip())
                            end = float(end.strip())
                            values[j] = (start, end)
                        except ValueError:
                            return None  # invalid range format
                    else:
                        try:
                            values[j] = float(values[j].strip())
                        except ValueError:
                            return None  # invalid float format

            added_constraints.append((ingredient, feature, verb, values))

        constraints = Constraints(self.main_window.database)

        # Populate constraints object from user added constraints
        for ingredient, feature, verb, values in added_constraints:
            feature = str(feature)  # type set
            constraint_name = f"{ingredient}_{feature[:4].lower()}"
            if constraint_name in Constraints._CATEGORICAL:
                # Categorical constraint
                choices: List[Ingredient] = []
                for value in values:
                    choice = self.main_window.ing_ctrl.get_by_name(
                        name=value,
                        ingredient=Constraints._FEATURE_CLASS[constraint_name]
                        (enc_id=-1, name="Dummy Ingredient subclass instance"))
                    choices.append(choice)
                constraints.add_choices(
                    feature=constraint_name,
                    choices=choices)
            elif constraint_name in Constraints._NUMERIC:
                # Numerical constraint
                for value in values:
                    constraints.add_range(
                        feature=constraint_name,
                        low=min(value) if isinstance(value, tuple) else value,
                        high=max(value) if isinstance(value, tuple) else value)
            else:
                Log.e(f"Unknown constraint: {constraint_name}")
                return None

        return constraints
