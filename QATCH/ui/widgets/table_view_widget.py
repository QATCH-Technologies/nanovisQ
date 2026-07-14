from PyQt5 import QtCore, QtGui, QtWidgets


class TableView(QtWidgets.QTableWidget):

    def __init__(self, data, *args):
        QtWidgets.QTableWidget.__init__(self, *args)
        self.setData(data)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def setData(self, data):
        self.data = data
        self.clear()
        horHeaders = []
        for n, key in enumerate(self.data.keys()):
            horHeaders.append(key)
            for m, item in enumerate(self.data[key]):
                error_cell = False
                if str(item).startswith("*") and str(item).endswith("*"):
                    item = str(item)[1:-1]
                    error_cell = True
                if item == str(item):
                    newitem = QtWidgets.QTableWidgetItem(item)
                else:
                    newitem = QtWidgets.QTableWidgetItem(f"{item:2.2f}")
                newitem.setFlags(QtCore.Qt.ItemIsEnabled)
                if error_cell:
                    newitem.setForeground(QtGui.QBrush(QtGui.QColor(255, 127, 127)))
                self.setItem(m, n, newitem)
        self.setHorizontalHeaderLabels(horHeaders)
        header = self.horizontalHeader()
        header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )  # refactored for Python 3.11: was setResizeMode()
        header.setStretchLastSection(False)
        # for i in range(len(horHeaders)):
        #     header.setResizeMode(i, QtWidgets.QHeaderView.ResizeToContents
        #               if i < 3 else QtWidgets.QHeaderView.Stretch)
