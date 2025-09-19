from PyQt5 import QtCore, QtGui, QtWidgets


class Color:
    black = QtGui.QColor(0, 0, 0)
    light_red = QtGui.QColor(255, 127, 127)
    light_yellow = QtGui.QColor(255, 255, 127)
    white = QtGui.QColor(255, 255, 255)


class TableView(QtWidgets.QTableWidget):

    # Define named constants for special row indices
    PROTEIN_TYPE_ROW = 0
    PROTEIN_CLASS_ROW = 2
    BUFFER_TYPE_ROW = 6

    def __init__(self, data, *args):
        QtWidgets.QTableWidget.__init__(self, *args)

        # Lookup table for mapping protein names to class types
        # i.e. {"Protein #1": "None", "Protein #2": "Other"}
        self._protein_type_to_class = {}

        self.itemChanged.connect(self._on_item_changed)
        self.setData(data)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.verticalHeader().setVisible(False)

    def clear(self):
        super().clear()
        self._is_empty = True
        

    def setData(self, data: dict[str, str]) -> None:
        self.data = data
        self.clear()
        horHeaders = []
        for n, key in enumerate(self.data.keys()):
            horHeaders.append(key)
            for m, item in enumerate(self.data[key]):
                if n == 0 or n == 2:
                    # always treat first and last cols as uneditable text, not a QComboBox
                    newitem = QtWidgets.QTableWidgetItem(str(item))
                elif len(str(item)) == 0:
                    # if item is blank (empty string)
                    newitem = QtWidgets.QTableWidgetItem(str(item))
                elif self._is_number(item):
                    # if item is a number, we want to format it
                    # as a string with at most 2 decimal places
                    # but removing any trailing zeros
                    # (ex: 1.234 -> "1.23", 1.200 -> "1.2", 1.000 -> "1")
                    round_item = f"{float(item):.2f}".rstrip("0").rstrip(".")
                    newitem = QtWidgets.QTableWidgetItem(round_item)
                else:
                    newitem = QtWidgets.QComboBox()
                    # newitem.addItem("add new...")
                    if isinstance(item, dict):
                        choices, selected = list(item.values())
                        # use a local copy to avoid mutating source choices too
                        local_choices = list(choices)
                        # skip Protein Type/Class and Buffer Type rows
                        if m not in [self.PROTEIN_TYPE_ROW, self.PROTEIN_CLASS_ROW, self.BUFFER_TYPE_ROW]:
                            if not any(str(c).casefold() == "none" for c in local_choices):
                                local_choices.insert(0, "None")
                        newitem.addItems(local_choices)
                        if len(selected):
                            try:
                                newitem.setCurrentIndex(
                                    [newitem.itemText(i).casefold() for i in range(newitem.count())]
                                    .index(str(selected).casefold()))  # case-insensitive matching
                            except ValueError:
                                print(f"WARNING: Entry \"{str(selected)}\" is not a known type!")
                                newitem.setCurrentText(str(selected))
                            self.data["Units"][m] = ""  # clear flag
                        else:
                            newitem.setCurrentIndex(-1)  # no selection
                            self.data["Units"][m] = "\u2190"  # unicode left arrow
                            newitem.currentIndexChanged.connect(
                                lambda idx, row=m: self._row_combo_set(row))
                    elif isinstance(item, list):
                        # skip Protein Type/Class and Buffer Type rows
                        if m not in [self.PROTEIN_TYPE_ROW, self.PROTEIN_CLASS_ROW, self.BUFFER_TYPE_ROW]:
                            newitem.addItem("None")
                        newitem.addItems(item)
                        self.data["Units"][m] = "\u2190"  # unicode left arrow
                        newitem.currentIndexChanged.connect(
                            lambda idx, row=m: self._row_combo_set(row))
                        newitem.setCurrentIndex(-1)  # no selection
                    else:  # str
                        newitem.addItem(item)
                        self.data["Units"][m] = ""  # clear flag
                    # add `_on_combo_change` handler to all QComboBox widgets
                    newitem.currentIndexChanged.connect(
                        lambda idx, row=m: self._on_combo_change(idx, row))
                # disable 1st and last column items (not selectable or editable)
                if n == 0 or n == 2:
                    if n == 0:  # bold 1st column items (headers)
                        font = newitem.font()
                        font.setBold(True)
                        newitem.setFont(font)
                    newitem.setFlags(newitem.flags() &
                                     ~(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEditable))
                if isinstance(newitem, QtWidgets.QWidget):
                    self.setCellWidget(m, n, newitem)
                else:
                    self.setItem(m, n, newitem)
                # Unhide a row if it was hidden
                if self.isRowHidden(m):
                    self.showRow(m)
        self.setHorizontalHeaderLabels(horHeaders)
        header = self.horizontalHeader()
        header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        self._is_empty = False

    def allSet(self) -> bool:
        for n, key in enumerate(self.data.keys()):
            for m, _ in enumerate(self.data[key]):
                if self.isRowHidden(m):
                    continue  # do not check hidden rows
                item = self.item(m, n)
                if item is None:
                    continue  # QComboBox will return a None item
                if item.background().color().name() in [Color.light_yellow.name(), Color.light_red.name()]:
                    return False
        return True
    
    def setProteinsByClass(self, proteins_by_class: dict):        
        result = {}
        for p_class, types in proteins_by_class.items():
            for type in types:
                p_type = str(type).casefold()
                if p_type in result:
                    print(f"Warning: '{type}' appears multiple times")
                    print(f"         '{type}' was '{result[p_type]}'; now '{p_class}'.")
                result[p_type] = str(p_class)
        self._protein_type_to_class = result
    

    def isEmpty(self) -> bool:
        return self._is_empty

    def _row_combo_set(self, idx):
        item = self.item(idx, 2)
        if item is not None:
            self.blockSignals(True)  # prevent recursion
            item.setBackground(QtGui.QBrush(Color.white))
            self.item(idx, 2).setText("")
            self.blockSignals(False)

    def _is_number(self, s: str):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem):
        row, col, text = item.row(), item.column(), item.text()
        print(f"Cell ({row}, {col}) changed to: {text}")

        if col == 2 and text == "\u2190":  # unicode left arrow
            item.setBackground(QtGui.QBrush(Color.light_yellow))

        if not (item.flags() & QtCore.Qt.ItemFlag.ItemIsEditable):
            # print("skip, disabled")
            return

        now_bg = item.background()
        now_fg = item.foreground()
        new_bg = QtGui.QBrush(now_bg.color())
        new_fg = QtGui.QBrush(now_fg.color())

        if len(text) == 0:
            new_bg.setColor(Color.light_yellow)
            new_fg.setColor(Color.black)
        elif not self._is_number(text):
            new_bg.setColor(Color.light_red)
            new_fg.setColor(Color.light_yellow)
        else:
            new_bg.setColor(Color.white)
            new_fg.setColor(Color.black)

        self.blockSignals(True)  # prevent recursion
        if new_bg.color().name() != now_bg.color().name():
            item.setBackground(new_bg)
        if new_fg.color().name() != now_fg.color().name():
            item.setForeground(new_fg)
        self.blockSignals(False)

        self.clearSelection()  # unselect on item change

    def _on_combo_change(self, idx: int, row: int):
        # self.blockSignals(True)
        if row == self.PROTEIN_TYPE_ROW:
            # Update protein class type to match protein name
            protein_type = self.cellWidget(row, 1).currentText().casefold()
            protein_class: str = self._protein_type_to_class.get(protein_type, "none")
            class_item: QtWidgets.QComboBox = self.cellWidget(self.PROTEIN_CLASS_ROW, 1)
            try:
                if class_item is not None:
                    class_item.setCurrentIndex(
                        [class_item.itemText(i).casefold() for i in range(class_item.count())]
                        .index(protein_class.casefold()))  # case-insensitive matching
            except ValueError:
                print(f"WARNING: Entry \"{protein_class}\" is not a known Protein Class! Using \"Other\" instead.")
                try:
                    class_item.setCurrentIndex(
                        [class_item.itemText(i).casefold() for i in range(class_item.count())]
                        .index("other"))
                except ValueError:
                    class_item.setCurrentIndex(-1)
            # NOTE: Do NOT return here, Protein Type has an associated Protein Concentration field
        conc_item = self.item(row+1, 1)  # concentration item
        if conc_item is None or row == self.PROTEIN_CLASS_ROW:
            return  # no concentration item to change
        # "None" selected for non-Protein/Buffer
        if idx == 0 and row not in [self.PROTEIN_TYPE_ROW, self.BUFFER_TYPE_ROW]:
            # If user selects "None" for any Type other than Protein or Buffer,
            # we assume they do not want to set a concentration for that item,
            # then disable the concentration item and set it to zero
            conc_item.setText("0")
            conc_item.setFlags(conc_item.flags() &
                               ~(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEditable))
        else:
            # If the user selects any other item, enable the concentration
            # value and set it to the default value (blank, missing input)
            conc_item.setFlags(conc_item.flags() |
                               (QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEditable))
            conc_item.setText("")
        # NOTE: By not blocking signals here, we allow the `itemChanged` signal
        #       to propagate and set/clear the background color based on text.
        #       This is important to ensure the UI reflects the current state.
        # self.blockSignals(False)
