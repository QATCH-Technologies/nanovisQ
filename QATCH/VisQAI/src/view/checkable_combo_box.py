from PyQt5 import QtCore, QtGui, QtWidgets


class CheckableComboBox(QtWidgets.QComboBox):
    # create checkable combo box class
    def __init__(self, parent=None):
        super(CheckableComboBox, self).__init__(parent)
        self.view().pressed.connect(self.handle_item_pressed)
        self.currentTextChanged.connect(self.check_items)
        self.setModel(QtGui.QStandardItemModel(self))
        self._editable = False  # store editable state

    def addItems(self, texts):
        super().addItems(texts)

        # uncheck all items
        for i in range(self.count()):
            self.model().item(i, 0).setCheckState(QtCore.Qt.Unchecked)

    def getItems(self):
        items = []
        for i in range(self.count()):
            items.append(self.model().item(i, 0).text())
        return items

    def setEditable(self, editable: bool):

        # override setEditable to prevent user from making it editable
        # just store the value
        self._editable = editable

    # when any item get pressed
    def handle_item_pressed(self, index):

        # getting which item is pressed
        item = self.model().itemFromIndex(index)

        # make it check if unchecked and vice-versa
        if item.checkState() == QtCore.Qt.Checked:
            item.setCheckState(QtCore.Qt.Unchecked)
        else:
            item.setCheckState(QtCore.Qt.Checked)

        # calling method
        self.check_items()

    # method called by check_items
    def item_checked(self, index):

        # getting item at index
        item = self.model().item(index, 0)

        # return true if checked else false
        return item.checkState() == QtCore.Qt.Checked

    # method to check which items are checked
    def check_items(self):
        # blank list
        checkedItems = []

        # traversing the items
        for i in range(self.count()):

            # if item is checked add it to the list
            if self.item_checked(i):
                checkedItems.append(i)

        # call this method
        self.update_label(checkedItems)

    # method to update the label
    def update_label(self, item_list):

        n = ''
        count = 0

        # traversing the list
        for i in item_list:

            # getting label
            text_label = self.model().item(i, 0).text()

            # if count value is 0 don't add comma
            if count == 0:
                n += '% s' % text_label
            # else value is greater than 0
            # add semicolon
            else:
                n += '; % s' % text_label

            # increment count
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

        # Get the line edit and modify its properties to read-only
        line_edit = self.lineEdit()
        if line_edit:
            if self._editable:  # set by CheckableComboBox.setEditable()
                line_edit.setPlaceholderText("<val>; <min>-<max>")
                if line_edit.isReadOnly() == True:
                    line_edit.setReadOnly(False)
            else:
                if line_edit.isReadOnly() == False:
                    line_edit.setReadOnly(True)
