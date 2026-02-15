from PyQt5 import QtCore, QtWidgets


class FilterMenuButton(QtWidgets.QPushButton):
    """Button that opens a checkbox menu for multi-selection."""

    selectionChanged = QtCore.pyqtSignal()

    def __init__(self, title, items, parent=None):
        super().__init__(title, parent)
        self.items_map = {}  # name -> Action
        self.menu = QtWidgets.QMenu(self)
        self.menu.setStyleSheet("QMenu { menu-scrollable: 1; }")

        # 'All' Action
        self.act_all = self.menu.addAction("Select All")
        self.act_all.triggered.connect(self.select_all)
        self.menu.addSeparator()

        for item in items:
            name = item.name if hasattr(item, "name") else str(item)
            act = self.menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(True)  # Default to all selected
            act.toggled.connect(self._on_item_toggled)
            self.items_map[name] = act

        self.setMenu(self.menu)
        self.update_text()

        # Style
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            """
            QPushButton {
                text-align: left;
                padding: 5px 10px;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                background-color: white;
            }
            QPushButton::menu-indicator {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                padding-right: 10px; 
            }
        """
        )

    def _on_item_toggled(self):
        self.update_text()
        self.selectionChanged.emit()

    def select_all(self):
        self.menu.blockSignals(True)
        for act in self.items_map.values():
            act.setChecked(True)
        self.menu.blockSignals(False)
        self._on_item_toggled()

    def update_text(self):
        selected = [name for name, act in self.items_map.items() if act.isChecked()]
        total = len(self.items_map)

        if len(selected) == 0:
            self.setText("None selected")
            self.setStyleSheet(
                self.styleSheet().replace(
                    "border: 1px solid #d1d5db", "border: 1px solid #e57373"
                )
            )  # Red border warning
        elif len(selected) == total:
            self.setText("All selected")
            self.setStyleSheet(
                self.styleSheet().replace(
                    "border: 1px solid #e57373", "border: 1px solid #d1d5db"
                )
            )
        else:
            self.setText(f"{len(selected)} selected")
            self.setStyleSheet(
                self.styleSheet().replace(
                    "border: 1px solid #e57373", "border: 1px solid #d1d5db"
                )
            )

    def get_selected_items(self):
        return [name for name, act in self.items_map.items() if act.isChecked()]
