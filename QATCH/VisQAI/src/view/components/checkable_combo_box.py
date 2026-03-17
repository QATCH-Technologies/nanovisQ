"""
checkable_combo_box.py

This module provides :class:`CheckableComboBox`, a PyQt5 combo box widget
where each item in the dropdown carries an independent checkmark.  The
displayed text is kept in sync with the currently checked selection and the
widget can optionally be made user-editable (e.g. to accept free-text filter
expressions alongside the list entries).

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

from PyQt5 import QtCore, QtGui, QtWidgets


class CheckableComboBox(QtWidgets.QComboBox):
    """A QComboBox where every item has an associated checkbox.

    The combo box uses a :class:`~PyQt5.QtGui.QStandardItemModel` so that
    each row can store a ``Qt.CheckState``.  When the user clicks an item in
    the drop-down view its check state is toggled and the line-edit area of
    the combo box is updated to show a semicolon-separated list of all
    currently checked items.

    When *editable* mode is disabled (the default) the line-edit is made
    temporarily editable only when at least one item is checked, so that the
    synthesised label string can be displayed.  When *editable* mode is
    enabled the line-edit remains writable and shows a placeholder hint.

    Attributes:
        _editable (bool): Whether the widget should allow free-text input in
            addition to list selection.  Controlled via :meth:`setEditable`.
    """

    def __init__(self, parent=None):
        """Initialise the combo box and wire up internal signals.

        Args:
            parent (QtWidgets.QWidget, optional): Parent widget. Defaults to
                ``None``.
        """
        super(CheckableComboBox, self).__init__(parent)
        self.view().pressed.connect(self.handle_item_pressed)
        self.currentTextChanged.connect(self.check_items)
        self.setModel(QtGui.QStandardItemModel(self))
        self._editable = False

    def addItems(self, texts):
        """Add multiple items and initialise each one as unchecked.

        Args:
            texts (Iterable[str]): Display strings to add to the combo box.
        """
        super().addItems(texts)
        for i in range(self.count()):
            self.model().item(i, 0).setCheckState(QtCore.Qt.Unchecked)

    def getItems(self):
        """Return the display text of every item in the combo box.

        Returns:
            list[str]: Ordered list of item label strings.
        """
        items = []
        for i in range(self.count()):
            items.append(self.model().item(i, 0).text())
        return items

    def setEditable(self, editable: bool):
        """Control whether the line-edit accepts free-text input.

        When *editable* is ``False`` the internal ``QLineEdit`` is kept
        read-only (unless items are checked, in which case it is temporarily
        unlocked to display the generated label).  When ``True`` the
        line-edit is always writable and shows a ``<val>; <min>-<max>``
        placeholder.

        Args:
            editable (bool): ``True`` to allow free-text entry, ``False`` for
                display-only behaviour.
        """
        self._editable = editable

    def handle_item_pressed(self, index):
        """Toggle the check state of the item at *index* when it is pressed.

        Connected to the ``pressed`` signal of the internal list view.  After
        toggling, :meth:`check_items` is called to refresh the label.

        Args:
            index (QtCore.QModelIndex): Model index of the pressed item.
        """
        item = self.model().itemFromIndex(index)
        if item.checkState() == QtCore.Qt.Checked:
            item.setCheckState(QtCore.Qt.Unchecked)
        else:
            item.setCheckState(QtCore.Qt.Checked)
        self.check_items()

    def item_checked(self, index):
        """Return whether the item at a given row index is checked.

        Args:
            index (int): Zero-based row index of the item to query.

        Returns:
            bool: ``True`` if the item's check state is
            ``Qt.Checked``, ``False`` otherwise.
        """
        item = self.model().item(index, 0)
        return item.checkState() == QtCore.Qt.Checked

    def check_items(self):
        """Collect checked row indices and refresh the display label.

        Iterates over all rows, builds a list of indices whose check state is
        ``Qt.Checked``, then delegates label construction to
        :meth:`update_label`.
        """
        checkedItems = []
        for i in range(self.count()):
            if self.item_checked(i):
                checkedItems.append(i)
        self.update_label(checkedItems)

    def update_label(self, item_list):
        """Update the combo box display text from a list of checked row indices.

        Builds a semicolon-separated string from the labels of *item_list* and
        applies it to the line-edit area.  Editability of the underlying
        ``QLineEdit`` is adjusted as follows:

        * **Non-editable mode, items checked** - the base class is temporarily
          made editable so that :meth:`~PyQt5.QtWidgets.QComboBox.setCurrentText`
          can display the generated label, then the line-edit is locked
          read-only.
        * **Non-editable mode, nothing checked** - the base class reverts to
          non-editable and the selection is cleared.
        * **Editable mode** - the base class is always editable and the
          line-edit remains writable with a placeholder hint.

        Args:
            item_list (list[int]): Zero-based row indices of all currently
                checked items.

        Note:
            To display text different from the items in the list when a model
            is set, the ``QComboBox`` must be made editable.  If it is not
            editable, ``setCurrentText()`` will only succeed if the provided
            string exactly matches an existing item's display text in the
            model.
        """
        n = ""
        count = 0
        for i in item_list:
            text_label = self.model().item(i, 0).text()
            if count == 0:
                n += "% s" % text_label
            else:
                n += "; % s" % text_label
            count += 1

        # NOTE: To display text different from the items in the list
        # when a model is set, the QComboBox must be made editable.
        # If it is not editable, setCurrentText() will only succeed in
        # changing the displayed text if the provided string is an
        # exact match for an existing item's display text in the model.

        if not self._editable:

            # if items are checked, set checked text to combo box (see note above)
            if count > 0 and self.currentText() != n:
                super().setEditable(True)
                self.setCurrentText(n)

            # if no items are checked, set the combo box to be uneditable and blank
            if count == 0 and self.currentText() != n:
                super().setEditable(False)
                self.setCurrentIndex(-1)

        else:
            super().setEditable(True)
        line_edit = self.lineEdit()
        if line_edit:
            if self._editable:
                line_edit.setPlaceholderText("<val>; <min>-<max>")
                if line_edit.isReadOnly() == True:
                    line_edit.setReadOnly(False)
            else:
                if line_edit.isReadOnly() == False:
                    line_edit.setReadOnly(True)
