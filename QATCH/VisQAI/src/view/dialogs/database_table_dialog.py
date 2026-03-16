"""
database_table_dialog.py

Provides a generic, reusable dialog for displaying and managing database records.

This module contains the DatabaseTableDialog, a versatile QDialog that wraps a
QTableWidget with built-in support for searching, row deletion via context
menus, and interactive checkbox columns.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt


class DatabaseTableDialog(QtWidgets.QDialog):
    """A generic dialog for tabular data display with filtering and callbacks.

    This dialog simplifies the process of showing database rows by providing
    automatic table population, a search bar for filtering, and hooks for
    deleting records or toggling boolean states (checkboxes).

    Attributes:
        delete_callback (callable, optional): A function called when a row is deleted.
            Expected signature: `callback(record_id) -> bool`.
        check_col_idx (int, optional): The index of the column that should be
            rendered as a checkbox.
        check_callback (callable, optional): A function called when a checkbox
            is toggled. Expected signature: `callback(record_id, is_checked)`.
        search_bar (QtWidgets.QLineEdit): The input field used for table filtering.
        table (QtWidgets.QTableWidget): The central widget displaying the data.
    """

    def __init__(
        self,
        title,
        headers,
        data_rows,
        parent=None,
        delete_callback=None,
        check_col_idx=None,
        check_callback=None,
        export_callback=None,
    ):
        """Initializes the dialog and builds the table interface.

        Args:
            title (str): The window title.
            headers (list[str]): List of column header labels.
            data_rows (list[list]): A list of rows, where each row is a list of
                values matching the headers.
            parent (QWidget, optional): The parent widget. Defaults to None.
            delete_callback (callable, optional): Logic to execute on record
                deletion. Defaults to None.
            check_col_idx (int, optional): Index for a checkable column.
                Defaults to None.
            check_callback (callable, optional): Logic to execute on checkbox
                toggle. Defaults to None.
            export_callback (callable, optional): Logic to execute when the
                Export button is clicked. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        self.delete_callback = delete_callback
        self.check_col_idx = check_col_idx
        self.check_callback = check_callback

        layout = QtWidgets.QVBoxLayout(self)

        # Search / Filter
        self.search_bar = QtWidgets.QLineEdit()
        self.search_bar.setPlaceholderText("Filter...")
        self.search_bar.textChanged.connect(self.filter_table)
        layout.addWidget(self.search_bar)

        # Table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)

        # Connect signals
        if self.delete_callback:
            self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.table.customContextMenuRequested.connect(self.show_context_menu)
        if self.check_callback:
            self.table.itemChanged.connect(self.on_item_changed)
        self.populate_table(data_rows)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)

        if export_callback:
            export_btn = QtWidgets.QPushButton("Export All to CSV")
            export_btn.clicked.connect(export_callback)

            btn_box.addButton(export_btn, QtWidgets.QDialogButtonBox.ActionRole)

        layout.addWidget(btn_box)

    def populate_table(self, data_rows):
        """Fills the table with provided data rows.

        Handles the logic for converting raw data into QTableWidgetItems,
        including special handling for checkbox columns.

        Args:
            data_rows (list[list]): The data to display.
        """
        self.table.blockSignals(True)  # Prevent itemChanged firing during setup
        self.table.setRowCount(len(data_rows))

        for r_idx, row_data in enumerate(data_rows):
            for c_idx, val in enumerate(row_data):
                if self.check_col_idx is not None and c_idx == self.check_col_idx:
                    item = QtWidgets.QTableWidgetItem("")
                    item.setFlags(
                        QtCore.Qt.ItemIsUserCheckable
                        | QtCore.Qt.ItemIsEnabled
                        | QtCore.Qt.ItemIsSelectable
                    )
                    is_checked = str(val).lower() in ("true", "1", "yes")
                    item.setCheckState(
                        QtCore.Qt.Checked if is_checked else QtCore.Qt.Unchecked
                    )
                    item.setData(QtCore.Qt.UserRole, val)
                else:
                    item = QtWidgets.QTableWidgetItem(str(val))
                self.table.setItem(r_idx, c_idx, item)
        self.table.blockSignals(False)

    def on_item_changed(self, item):
        """Internal handler for checkbox state changes.

        Identifies the record ID from column 0 of the affected row and triggers
        the `check_callback`.

        Args:
            item (QTableWidgetItem): The table item that was changed.
        """
        if self.check_col_idx is not None and item.column() == self.check_col_idx:
            row = item.row()
            id_item = self.table.item(row, 0)
            if id_item and self.check_callback:
                record_id = id_item.text()
                is_checked = item.checkState() == QtCore.Qt.Checked
                self.check_callback(record_id, is_checked)

    def show_context_menu(self, pos):
        """Displays a context menu for row-level actions.

        Currently supports a 'Delete' action which is only enabled if at
        least one item is selected.

        Args:
            pos (QPoint): The local position where the menu was requested.
        """
        menu = QtWidgets.QMenu()
        delete_action = menu.addAction("Delete")

        if not self.table.selectedItems():
            delete_action.setEnabled(False)

        action = menu.exec_(self.table.mapToGlobal(pos))

        if action == delete_action:
            self.handle_delete()

    def handle_delete(self):
        """Identifies selected rows and triggers the deletion callback.

        Rows are processed in reverse order to ensure indices remain valid
        as rows are removed from the table widget.
        """
        rows = sorted(
            set(index.row() for index in self.table.selectedIndexes()), reverse=True
        )
        for row in rows:
            id_item = self.table.item(row, 0)
            if id_item:
                record_id = id_item.text()
                if self.delete_callback(record_id):
                    self.table.removeRow(row)

    def filter_table(self, text):
        """ "Performs a case-insensitive search across all table cells.

        Hides rows that do not contain the filter text in at least one column.

        Args:
            text (str): The search string provided by the user.
        """
        text = text.lower()
        for i in range(self.table.rowCount()):
            match = False
            for j in range(self.table.columnCount()):
                item = self.table.item(i, j)
                if item:
                    content = item.text().lower()
                    if not content and item.flags() & QtCore.Qt.ItemIsUserCheckable:
                        pass
                    elif text in content:
                        match = True
                        break
            self.table.setRowHidden(i, not match)
