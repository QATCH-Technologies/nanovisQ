from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt


class DatabaseTableDialog(QtWidgets.QDialog):
    """
    Generic Dialog to display database records in a table.
    Supports:
      - Search/Filter
      - Row Deletion (via delete_callback)
      - Checkbox Column (via check_col_idx + check_callback)
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
    ):
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

        # Populate
        self.populate_table(data_rows)

        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        # Close Button
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def populate_table(self, data_rows):
        self.table.blockSignals(True)  # Prevent itemChanged firing during setup
        self.table.setRowCount(len(data_rows))

        for r_idx, row_data in enumerate(data_rows):
            for c_idx, val in enumerate(row_data):

                # Checkbox Handling for specific column
                if self.check_col_idx is not None and c_idx == self.check_col_idx:
                    item = QtWidgets.QTableWidgetItem("")
                    item.setFlags(
                        QtCore.Qt.ItemIsUserCheckable
                        | QtCore.Qt.ItemIsEnabled
                        | QtCore.Qt.ItemIsSelectable
                    )

                    # Interpret value as boolean
                    is_checked = str(val).lower() in ("true", "1", "yes")
                    item.setCheckState(
                        QtCore.Qt.Checked if is_checked else QtCore.Qt.Unchecked
                    )

                    # Store original ID in data for easy access if needed, though we use col 0 lookup
                    item.setData(QtCore.Qt.UserRole, val)
                else:
                    item = QtWidgets.QTableWidgetItem(str(val))

                self.table.setItem(r_idx, c_idx, item)

        self.table.blockSignals(False)

    def on_item_changed(self, item):
        """Handle checkbox toggles."""
        if self.check_col_idx is not None and item.column() == self.check_col_idx:
            row = item.row()
            # Assume ID is always in column 0
            id_item = self.table.item(row, 0)
            if id_item and self.check_callback:
                record_id = id_item.text()
                is_checked = item.checkState() == QtCore.Qt.Checked
                self.check_callback(record_id, is_checked)

    def show_context_menu(self, pos):
        """Shows context menu with Delete option."""
        menu = QtWidgets.QMenu()
        delete_action = menu.addAction("Delete")

        if not self.table.selectedItems():
            delete_action.setEnabled(False)

        action = menu.exec_(self.table.mapToGlobal(pos))

        if action == delete_action:
            self.handle_delete()

    def handle_delete(self):
        """Processes deletion for selected rows."""
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
        """Simple case-insensitive row filtering."""
        text = text.lower()
        for i in range(self.table.rowCount()):
            match = False
            for j in range(self.table.columnCount()):
                item = self.table.item(i, j)
                # Check text or check state representation
                if item:
                    content = item.text().lower()
                    if not content and item.flags() & QtCore.Qt.ItemIsUserCheckable:
                        # Allow searching for 'checked' status if desired, or skip
                        pass
                    elif text in content:
                        match = True
                        break
            self.table.setRowHidden(i, not match)
