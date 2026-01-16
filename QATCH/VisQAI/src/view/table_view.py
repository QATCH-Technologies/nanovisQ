# table_view.py

"""Custom Table Widget for Formulation Data Entry used during FrameStep1.

This module provides a specialized QTableWidget for displaying and editing
protein configuration data. It handles validation, dynamic dropdown dependencies,
and visual feedback (color coding and tooltips) for user inputs.

Author(s):
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-01-16

Version:
    1.2
"""


from typing import Any, Dict, List, Optional, Tuple, Union, cast

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.logger import Logger as Log


class Color:
    """Defines static QColor constants used for UI feedback and styling.

    Attributes:
        black (QtGui.QColor): Standard black (0, 0, 0).
        light_red (QtGui.QColor): Pastel red (255, 127, 127) used for error states.
        light_yellow (QtGui.QColor): Pastel yellow (255, 255, 127) used for empty/warning states.
        white (QtGui.QColor): Standard white (255, 255, 255).
    """

    black = QtGui.QColor(0, 0, 0)
    light_red = QtGui.QColor(255, 127, 127)
    light_yellow = QtGui.QColor(255, 255, 127)
    white = QtGui.QColor(255, 255, 255)


class TableView(QtWidgets.QTableWidget):
    """A custom table widget for editing scientific property data.

    This table manages the display and validation of configuration data,
    specifically designed for protein and buffer inputs. It supports numeric
    range validation, dependent dropdown logic (e.g., protein type determining class),
    and visual status indicators.

    Attributes:
        validationFailed (QtCore.pyqtSignal): Signal emitted when a validation error
            occurs. The argument is the error message string.
    """

    TAG = "[TableView]"
    # Special row indices
    PROTEIN_TYPE_ROW = 0
    PROTEIN_CLASS_ROW = 2
    BUFFER_TYPE_ROW = 6

    # Signal to broadcast validation errors
    validationFailed = QtCore.pyqtSignal(str)

    def __init__(self, data: Dict[str, Any], *args) -> None:
        """Initializes the TableView with data and sets up the UI.

        Args:
            data (Dict[str, Any]): The initial dataset to populate the table.
                Keys represent column headers, and values are lists of row data.
            *args: Variable length argument list passed to the QTableWidget constructor.
        """
        super().__init__(*args)

        self._protein_type_to_class: Dict[str, str] = {}
        self._limits: Dict[str, Tuple[float, float]] = {}
        self._limit_mapping: Dict[str, str] = {}
        self._is_empty = True

        self.itemChanged.connect(self._on_item_changed)

        # Initialize UI
        header = self.verticalHeader()
        if header:
            header.setVisible(False)

        self.setData(data)

    def clear(self) -> None:
        """Clears all items from the table and resets the empty state flag."""
        super().clear()
        self._is_empty = True

    def setLimits(
        self,
        limits: Dict[str, Tuple[float, float]],
        label_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """Sets input limits for numeric rows.

        Args:
            limits (Dict[str, Tuple[float, float]]): A dictionary mapping a key
                (either the row label or a mapped key) to a tuple of (min, max).
            label_mapping (Optional[Dict[str, str]]): An optional dictionary mapping
                the visible row label text to the specific key used in `limits`.
        """
        self._limits = limits
        if label_mapping:
            self._limit_mapping = label_mapping

    def setData(self, data: Dict[str, Any]) -> None:
        """Populates the table with the provided data dictionary.

        This method clears the existing table, suspends signal updates for performance,
        generates the necessary widgets (text items or comboboxes), and resizes
        columns to fit contents.

        Args:
            data (Dict[str, Any]): The dictionary containing column headers as keys
                and lists of cell values as values.
        """
        self.data = data
        self.clear()

        # Disable updates during bulk operation for performance
        self.setUpdatesEnabled(False)
        self.blockSignals(True)

        try:
            keys = list(self.data.keys())
            self.setColumnCount(len(keys))
            # Determine row count from the first column's data length
            row_count = len(self.data[keys[0]]) if keys else 0
            self.setRowCount(row_count)
            self.setHorizontalHeaderLabels(keys)

            for col, key in enumerate(keys):
                column_data = self.data[key]
                for row, raw_item in enumerate(column_data):
                    self._create_and_set_cell(row, col, raw_item)

            # Adjust headers
            header = self.horizontalHeader()
            if header:
                header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
                header.setStretchLastSection(False)

            # Show hidden rows if any were hidden previously
            for r in range(row_count):
                if self.isRowHidden(r):
                    self.showRow(r)

            self._is_empty = False

        finally:
            self.blockSignals(False)
            self.setUpdatesEnabled(True)

    def _create_and_set_cell(self, row: int, col: int, raw_item: Any) -> None:
        """Creates the appropriate QTableWidgetItem or QWidget for a specific cell.

        If the item is a dictionary or list, a QComboBox is created.
        Otherwise, a QTableWidgetItem is created.

        Args:
            row (int): The row index.
            col (int): The column index.
            raw_item (Any): The data object for the cell (str, int, float, list, or dict).
        """
        # Determine if this is a structural column
        is_label_col = col == 0
        is_unit_col = col == 2

        # Handle Structural Columns or Simple Strings
        if (
            is_label_col
            or is_unit_col
            or isinstance(raw_item, (str, int, float))
            and not isinstance(raw_item, (dict, list))
        ):
            display_text = str(raw_item)
            if self._is_number(raw_item) and len(display_text) > 0:
                display_text = f"{float(raw_item):.2f}".rstrip("0").rstrip(".")

            item = QtWidgets.QTableWidgetItem(display_text)
            if is_label_col or is_unit_col:
                # Calculate flags (returns an int)
                raw_flags = item.flags() & ~(
                    QtCore.Qt.ItemFlag.ItemIsSelectable
                    | QtCore.Qt.ItemFlag.ItemIsEditable
                )

                flags = cast(QtCore.Qt.ItemFlags, raw_flags)
                item.setFlags(flags)
                if is_label_col:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)

            self.setItem(row, col, item)
        elif isinstance(raw_item, (dict, list)):
            combo = self._create_combobox(row, raw_item)
            self.setCellWidget(row, col, combo)

    def _create_combobox(
        self, row: int, item_data: Union[Dict, List]
    ) -> QtWidgets.QComboBox:
        """Creates and configures a QComboBox for the given data.

        Handles logic for adding "None" options to non-required fields and setting
        the initial selection index.

        Args:
            row (int): The row index for this combobox.
            item_data (Union[Dict, List]): The source data containing options.
                If a dict, values are (choices, selected). If a list, it is just choices.

        Returns:
            QtWidgets.QComboBox: The configured combobox widget.
        """
        combo = QtWidgets.QComboBox()

        # Determine choices and selected item
        if isinstance(item_data, dict):
            choices, selected = list(item_data.values())
            local_choices = list(choices)
            if len(selected):
                self.data["Units"][row] = ""
            else:
                self.data["Units"][row] = "\u2190"
        else:
            local_choices = list(item_data)
            selected = None
            self.data["Units"][row] = "\u2190"

        # Add "None" option if applicable
        is_special_row = row in [
            self.PROTEIN_TYPE_ROW,
            self.PROTEIN_CLASS_ROW,
            self.BUFFER_TYPE_ROW,
        ]
        if not is_special_row:
            if not any(str(c).casefold() == "none" for c in local_choices):
                if isinstance(item_data, list):
                    local_choices.append("None")
                else:
                    local_choices.insert(0, "None")

        combo.addItems(local_choices)

        # Set Selection
        if selected:
            try:
                idx = [
                    combo.itemText(i).casefold() for i in range(combo.count())
                ].index(str(selected).casefold())
                combo.setCurrentIndex(idx)
            except ValueError:
                Log.w(self.TAG, f'Entry "{selected}" is not a known type.')
                combo.setCurrentText(str(selected))
        else:
            combo.setCurrentIndex(-1)
        if isinstance(item_data, dict) and not selected:
            combo.currentIndexChanged.connect(lambda idx, r=row: self._row_combo_set(r))
        elif isinstance(item_data, list):
            combo.currentIndexChanged.connect(lambda idx, r=row: self._row_combo_set(r))

        combo.currentIndexChanged.connect(
            lambda idx, r=row: self._on_combo_change(idx, r)
        )

        return combo

    def allSet(self) -> bool:
        """Checks if all visible, required fields in the table are filled and valid.

        This method iterates through the table and checks the background color of cells.
        Cells colored yellow (empty) or red (error) cause validation to fail.

        Returns:
            bool: True if all fields are valid, False otherwise.
        """
        for col_idx, _ in enumerate(self.data.keys()):
            for row_idx in range(self.rowCount()):
                if self.isRowHidden(row_idx):
                    continue

                item = self.item(row_idx, col_idx)
                if item is None:
                    continue
                bg_color = item.background().color().name()
                if bg_color in [Color.light_yellow.name(), Color.light_red.name()]:
                    return False
        return True

    def setProteinsByClass(self, proteins_by_class: dict) -> None:
        """Configures the mapping between protein classes and their types.

        This mapping is used to automatically update the 'Protein Class' dropdown
        when a 'Protein Type' is selected.

        Args:
            proteins_by_class (dict): A dictionary where keys are protein classes
                and values are lists of protein types belonging to that class.
        """
        result = {}
        for p_class, types in proteins_by_class.items():
            for p_type in types:
                p_type_cf = str(p_type).casefold()
                if p_type_cf in result:
                    Log.w(self.TAG, f"'{p_type}' appears multiple times")
                result[p_type_cf] = str(p_class)
        self._protein_type_to_class = result

    def isEmpty(self) -> bool:
        """Returns the empty state of the table.

        Returns:
            bool: True if the table has been cleared or not yet populated.
        """
        return self._is_empty

    def _get_limit_for_row(self, row_idx: int) -> Optional[Tuple[float, float]]:
        """Resolves the limit tuple for a specific row index.

        Args:
            row_idx (int): The index of the row to check.

        Returns:
            Optional[Tuple[float, float]]: A tuple of (min, max) if limits exist
            for the row, otherwise None.
        """
        item = self.item(row_idx, 0)
        if not item:
            return None

        label = item.text()
        limit_key = self._limit_mapping.get(label, label)
        return self._limits.get(limit_key)

    def _row_combo_set(self, row_idx: int) -> None:
        """Clears the 'arrow' indicator in the units column when a selection is made.

        Args:
            row_idx (int): The row index of the combo box that changed.
        """
        item = self.item(row_idx, 2)
        if item:
            self.blockSignals(True)
            try:
                item.setBackground(QtGui.QBrush(Color.white))
                item.setText("")
            finally:
                self.blockSignals(False)

    def _is_number(self, s: Any) -> bool:
        """Checks if a given input can be converted to a float.

        Args:
            s (Any): The input to check.

        Returns:
            bool: True if the input is numeric, False otherwise.
        """
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        """Slot called when a table item changes to handle validation.

        This method validates numeric inputs against defined limits, updates
        cell background colors (Yellow=Empty, Red=Error, White=OK), updates tooltips,
        and emits the `validationFailed` signal if errors are found.

        Args:
            item (QtWidgets.QTableWidgetItem): The item that changed.
        """
        row, col = item.row(), item.column()
        text = item.text()

        if col == 2 and text == "\u2190":
            item.setBackground(QtGui.QBrush(Color.light_yellow))
            return

        # Only validate editable items
        if (item.flags() & QtCore.Qt.ItemFlag.ItemIsEditable) == 0:
            return
        is_empty = len(text) == 0
        is_number = self._is_number(text)
        is_in_range = True
        limit_msg = ""
        row_label = ""

        if not is_empty and is_number:
            limits = self._get_limit_for_row(row)
            if limits:
                val = float(text)
                min_val, max_val = limits
                if not (min_val <= val <= max_val):
                    is_in_range = False
                    limit_msg = f"Value must be between {min_val} and {max_val}"
                    lbl_item = self.item(row, 0)
                    row_label = lbl_item.text() if lbl_item else f"Row {row}"

        new_bg = QtGui.QColor(Color.white)
        new_fg = QtGui.QColor(Color.black)
        tooltip_msg = ""
        error_to_emit = None

        if is_empty:
            new_bg = Color.light_yellow
            tooltip_msg = "Field cannot be empty"
        elif not is_number:
            new_bg = Color.light_red
            new_fg = Color.light_yellow
            tooltip_msg = "Must be a number"
            error_to_emit = f"Validation Error ({row_label or row}): {tooltip_msg}"
        elif not is_in_range:
            new_bg = Color.light_red
            new_fg = Color.light_yellow
            tooltip_msg = limit_msg
            error_to_emit = f"Validation Error ({row_label}): {limit_msg}"

        self.blockSignals(True)
        try:
            if item.background().color() != new_bg:
                item.setBackground(QtGui.QBrush(new_bg))
            if item.foreground().color() != new_fg:
                item.setForeground(QtGui.QBrush(new_fg))
            item.setToolTip(tooltip_msg)
        finally:
            self.blockSignals(False)

        if error_to_emit:
            Log.e(self.TAG, f"Error during item change: {error_to_emit}")
            self.validationFailed.emit(error_to_emit)
            self._show_tooltip_popup(item, error_to_emit)

        self.clearSelection()

    def _show_tooltip_popup(
        self, item: QtWidgets.QTableWidgetItem, message: str
    ) -> None:
        """Forces a tooltip to appear immediately at the item's location.

        Args:
            item (QtWidgets.QTableWidgetItem): The item to show the tooltip for.
            message (str): The message to display.
        """
        rect = self.visualItemRect(item)
        viewport = self.viewport()

        if viewport:
            global_pos = viewport.mapToGlobal(rect.bottomLeft())
            QtWidgets.QToolTip.showText(global_pos, message, self)

    def _on_combo_change(self, idx: int, row: int) -> None:
        """Handles dependencies between dropdowns and row states.

        Specific behaviors:
        1. When Protein Type changes, it attempts to set the Protein Class.
        2. When "None" (index 0) is selected in certain rows, it may disable
           subsequent input rows (e.g., concentration).

        Args:
            idx (int): The new index of the combobox.
            row (int): The row index of the combobox.
        """
        # Update Protein Class if Protein Type changed
        if row == self.PROTEIN_TYPE_ROW:
            type_combo = self.cellWidget(row, 1)
            class_combo = self.cellWidget(self.PROTEIN_CLASS_ROW, 1)

            if type_combo and class_combo:
                p_type = type_combo.currentText().casefold()
                p_class = self._protein_type_to_class.get(p_type, "none").casefold()

                # Attempt to find the class in the second dropdown
                found_idx = -1
                for i in range(class_combo.count()):
                    if class_combo.itemText(i).casefold() == p_class:
                        found_idx = i
                        break

                if found_idx >= 0:
                    class_combo.setCurrentIndex(found_idx)
                else:
                    # Fallback logic: check for 'other' or reset
                    Log.w(self.TAG, f'Entry "{p_class}" is not a known Protein Class.')
                    fallback_idx = -1
                    for i in range(class_combo.count()):
                        if class_combo.itemText(i).casefold() == "other":
                            fallback_idx = i
                            break
                    class_combo.setCurrentIndex(fallback_idx)

        # Logic to enable/disable specific concentration rows based on selection
        conc_item = self.item(row + 1, 1)
        if conc_item and row != self.PROTEIN_CLASS_ROW:

            is_none_selected = idx == 0
            is_exempt_row = row in [self.PROTEIN_TYPE_ROW, self.BUFFER_TYPE_ROW]

            if is_none_selected and not is_exempt_row:
                conc_item.setText("0")
                flags = conc_item.flags() & ~(
                    QtCore.Qt.ItemFlag.ItemIsSelectable
                    | QtCore.Qt.ItemFlag.ItemIsEditable
                )
                conc_item.setFlags(QtCore.Qt.ItemFlags(flags))  # type: ignore
            else:
                flags = conc_item.flags() | (
                    QtCore.Qt.ItemFlag.ItemIsSelectable
                    | QtCore.Qt.ItemFlag.ItemIsEditable
                )
                conc_item.setFlags(QtCore.Qt.ItemFlags(flags))  # type: ignore
                conc_item.setText("")
