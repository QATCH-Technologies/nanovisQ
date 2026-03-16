"""
filter_menu_button.py

Provides a specialized QPushButton for multi-selection filtering.

This module contains the FilterMenuButton, which integrates a QMenu with
checkable actions. it is designed for scenarios where a user needs to
filter a dataset by multiple categories, providing visual feedback on
the current selection state directly on the button text.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

from PyQt5 import QtCore, QtWidgets


class FilterMenuButton(QtWidgets.QPushButton):
    """A button that triggers a multi-select checkbox menu.

    The button text updates dynamically based on how many items are selected
    (e.g., "All selected", "None selected", or "3 selected"). It also
    visually alerts the user with a red border if no filters are active.

    Attributes:
        selectionChanged (QtCore.pyqtSignal): Signal emitted whenever the
            state of any checkbox in the menu changes.
        items_map (dict): A mapping of item names (str) to their
            corresponding QAction objects.
        menu (QtWidgets.QMenu): The dropdown menu containing filter options.
        act_all (QtWidgets.QAction): The 'Select All' convenience action.
    """

    selectionChanged = QtCore.pyqtSignal()

    def __init__(self, title, items, parent=None):
        """Initializes the FilterMenuButton with a list of items.

        Args:
            title (str): Initial text for the button.
            items (list): A list of objects to be turned into filter options.
                If objects have a 'name' attribute, it is used; otherwise,
                str(item) is used.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(title, parent)
        self.items_map = {}
        self.menu = QtWidgets.QMenu(self)
        self.menu.setStyleSheet("QMenu { menu-scrollable: 1; }")

        self.act_all = self.menu.addAction("Select All")
        self.act_all.triggered.connect(self.select_all)
        self.menu.addSeparator()

        for item in items:
            name = item.name if hasattr(item, "name") else str(item)
            act = self.menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(True)
            act.toggled.connect(self._on_item_toggled)
            self.items_map[name] = act

        self.setMenu(self.menu)
        self.update_text()
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
        """Internal handler for checkbox toggle events.

        Updates the button text and emits the selectionChanged signal.
        """
        self.update_text()
        self.selectionChanged.emit()

    def select_all(self):
        """Check all items in the filter menu simultaneously.

        This method blocks signals temporarily to ensure update_text is only
        called once after all items are checked.
        """
        self.menu.blockSignals(True)
        for act in self.items_map.values():
            act.setChecked(True)
        self.menu.blockSignals(False)
        self._on_item_toggled()

    def update_text(self):
        """Updates the button label and border color based on selection count.

        Sets the border to red if no items are selected to indicate an invalid
        filter state.
        """
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
        """Returns the names of all currently checked items.

        Returns:
            list[str]: A list of names of the selected items.
        """
        return [name for name, act in self.items_map.items() if act.isChecked()]
