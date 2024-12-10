from PyQt5 import QtGui
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QMessageBox, QDesktopWidget
from string import ascii_uppercase as auc
import sys
import json
import os

try:
    from QATCH.common.logger import Logger as Log
    from QATCH.common.architecture import Architecture
except:
    class Log():
        @staticmethod
        def d(s):
            print(s)

    class Architecture():
        @staticmethod
        def get_path():
            return os.getcwd()


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.wellWidget = WellPlate(6, 4, 4)


class WellPlate(QWidget):
    pressPos = None
    clicked = pyqtSignal(int, int)

    def __init__(self, well_cols=4, well_rows=1, num_devs_available=0):
        super().__init__()
        self.title = "Plate Configuration"
        self.well_diameter = 50
        self.well_spacing = 10
        self.padding = 50
        self.well_cols = well_cols
        self.well_rows = well_rows
        self.default_selected = num_devs_available
        self.width = 2 * self.padding + self.well_cols * \
            self.well_diameter + (self.well_cols - 1) * self.well_spacing
        self.height = 4 * self.padding + self.well_rows * \
            self.well_diameter + (self.well_rows - 1) * self.well_spacing
        self.setFixedSize(self.width, self.height)
        self.move(QDesktopWidget().availableGeometry().center() -
                  self.frameGeometry().center())
        self.InitWindow()

    def InitWindow(self):
        self.icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/advanced.png')
        self.setWindowIcon(QtGui.QIcon(self.icon_path))  # .png
        self.setWindowTitle(self.title)
        self.clicked.connect(self.clickEvent)
        self.createWells()
        self.show()
        vbox = QVBoxLayout()
        hbox_help = QHBoxLayout()
        self.button_help = QPushButton("?")
        # button_help.height(), button_help.height())
        self.button_help.setFixedSize(34, 34)
        self.button_help.setStyleSheet(
            "border-radius: 17px; border: 2px solid black; font-weight: bold;")
        self.button_help.clicked.connect(self.show_help)
        hbox_help.addWidget(self.button_help)
        hbox_help.addStretch()
        vbox.addLayout(hbox_help)
        vbox.addStretch()
        hbox = QHBoxLayout()
        hbox.addStretch()
        button_all = QPushButton("All")
        button_none = QPushButton("None")
        button_invert = QPushButton("Invert")
        button_all.clicked.connect(self.selectAll)
        button_none.clicked.connect(self.selectNone)
        button_invert.clicked.connect(self.selectInverse)
        hbox.addWidget(button_all)
        hbox.addWidget(button_none)
        hbox.addWidget(button_invert)
        hbox.addStretch()
        vbox.addLayout(hbox)
        save_cancel = QHBoxLayout()
        save = QPushButton("Save")
        cancel = QPushButton("Cancel")
        save.clicked.connect(self.save)
        cancel.clicked.connect(self.close)
        save.setDefault(True)
        save_cancel.addStretch()
        save_cancel.addWidget(save)
        save_cancel.addWidget(cancel)
        save_cancel.addStretch()
        vbox.addLayout(save_cancel)
        self.setLayout(vbox)

    def show_help(self):
        if not hasattr(self, 'msg'):
            self.msg = QMessageBox()
            self.msg.setWindowTitle("Help: Plate Configuration")
            self.msg.setText(
                "Use this window to select the wells on the plate that you'd like to use:")
            self.msg.setInformativeText("- Click on a well to toggle that well's selection state (selected = 'gold')\n" +
                                        "- Click on a header label ('A', '1', etc.) to toggle the entire row/column\n" +
                                        "- Click \"All\", \"None\" or \"Invert\" buttons to adjust selection accordingly\n\n" +
                                        "When ready, click \"Save\" to store your plate configuration for the next run.")
            self.msg.setWindowIcon(QtGui.QIcon(self.icon_path))  # .png
            self.msg.setIcon(QMessageBox.Question)
            self.msg.setStandardButtons(QMessageBox.Ok)
        if self.msg.isHidden():
            self.msg.move(self.button_help.mapToGlobal(QPoint(-12, -12)))
            self.msg.show()

    def save(self):
        msg = QMessageBox()
        msg.setWindowIcon(QtGui.QIcon(self.icon_path))  # .png
        msg.setStandardButtons(QMessageBox.Ok)
        if self.wells_selected > self.default_selected:
            msg.setWindowTitle("WARN: Plate Configuration")
            msg.setText(
                "You cannot select more wells on the plate than are currently detected.")
            msg.setInformativeText(f"Number of selected wells:\t{self.wells_selected}\n" +
                                   f"Number of detected wells:\t{self.default_selected} (channel count)\n\n" +
                                   "To re-detect well count, click \"Reset\" on the main \"Run\" mode window.")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()
        elif self.wells_selected == 0:
            msg.setWindowTitle("WARN: Plate Configuration")
            msg.setText(
                "Please select at least one well to save a valid plate configuration.")
            msg.setInformativeText(f"Number of selected wells:\t{self.wells_selected}\n" +
                                   f"Number of detected wells:\t{self.default_selected} (channel count)\n\n" +
                                   "Click on \"?\" for help with managing your plate configuration.")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()
        else:
            try:
                with open("plate-config.json", 'w') as f:
                    json.dump(self.well_states, f)
                msg.setWindowTitle("Saved: Plate Configuration")
                msg.setText("Your plate configuration was saved successfully.")
                msg.setInformativeText(
                    f"Selected: {self.wells_selected} out of {self.default_selected} wells")
                msg.setIcon(QMessageBox.NoIcon)
                msg.exec_()
                self.close()
            except Exception as e:
                msg.setWindowTitle("ERROR: Plate Configuration")
                msg.setText(
                    "An error occurred while trying to save the plate configuration.")
                msg.setInformativeText("ERROR: " + str(e))
                msg.setIcon(QMessageBox.Critical)
                msg.exec_()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.pressPos = event.pos()

    def mouseReleaseEvent(self, event):
        # ensure that the left button was pressed *and* released within the
        # geometry of the widget; if so, emit the signal;
        if (self.pressPos is not None and
            event.button() == Qt.LeftButton and
                event.pos() in self.rect()):
            self.clicked.emit(self.pressPos.x(), self.pressPos.y())
        self.pressPos = None

    def clickEvent(self, click_x, click_y):
        Log.d(f"Click event: {click_x}, {click_y}")
        for x in range(self.well_cols):
            for y in range(self.well_rows):
                left = self.padding + x * \
                    (self.well_diameter + self.well_spacing)
                right = left + self.well_diameter
                top = self.padding + y*(self.well_diameter + self.well_spacing)
                bottom = top + self.well_diameter
                if click_x in range(left, right) and click_y in range(top, bottom):
                    Log.d(f"Well @ ({x},{y}) was clicked!")
                    self.toggleWellSelection(x, y)
                    self.repaint()
                    # there can only be one click for a given coord(x,y)
                    return
                 # check for row headers (A,B,C,D)
                if x == 0 and click_x in range(0, left) and click_y in range(top, bottom):
                    for i in range(self.well_cols):
                        self.toggleWellSelection(i, y)  # toggle entire row
                    self.repaint()
                    # there can only be one click for a given coord(x,y)
                    return
            # check for col headers (1,2,3,4)
            if click_x in range(left, right) and click_y in range(0, top):
                for i in range(self.well_rows):
                    self.toggleWellSelection(x, i)  # toggle entire col
                self.repaint()
                return  # there can only be one click for a given coord(x,y)

    def createWells(self):
        self.well_states = []
        self.wells_selected = 0
        self._last_well_count = 0
        for x in range(self.well_cols):
            self.well_states.append([])
            for y in range(self.well_rows):
                self.well_states[x].append(False)
        for y in range(self.well_rows):
            for x in range(self.well_cols):
                if self.wells_selected < self.default_selected:
                    self.well_states[x][y] = True
                    self.wells_selected += 1
        try:
            with open("plate-config.json", 'r') as f:
                saved_states = json.load(f)
            if len(saved_states) == self.well_cols and len(saved_states[0]) == self.well_rows:
                self.well_states = saved_states
                self.wells_selected = 0
                for x in range(self.well_cols):
                    for y in range(self.well_rows):
                        # and self.wells_selected < self.default_selected:
                        if self.well_states[x][y]:
                            self.wells_selected += 1
                Log.d(
                    "Loaded \"plate-config.json\" successfully. Plate dimensions match.")
            else:
                Log.d("Failed to load \"plate-config.json\". Plate dimensions mismatch.")
        except:
            Log.d(
                "Failed to load \"plate-config.json\". File may not exist or an error occurred.")
            pass

    def toggleWellSelection(self, x, y):
        if self.well_states[x][y]:
            self.wells_selected -= 1
        self.well_states[x][y] = not self.well_states[x][y]
        if self.well_states[x][y]:
            self.wells_selected += 1
        # caller must call repaint()

    def selectAll(self):
        self.wells_selected = 0
        for x in range(self.well_cols):
            for y in range(self.well_rows):
                self.well_states[x][y] = True
                self.wells_selected += 1
        self.repaint()

    def selectNone(self):
        self.wells_selected = 0
        for x in range(self.well_cols):
            for y in range(self.well_rows):
                self.well_states[x][y] = False
        self.repaint()

    def selectInverse(self):
        self.wells_selected = 0
        for x in range(self.well_cols):
            for y in range(self.well_rows):
                self.well_states[x][y] = not self.well_states[x][y]
                if self.well_states[x][y]:
                    self.wells_selected += 1
        self.repaint()

    def paintEvent(self, event):
        if self.wells_selected != self._last_well_count:
            self._last_well_count = self.wells_selected
            Log.d(f"# selected wells: {self.wells_selected}")
        qatchGold = QColor(237, 177, 32)
        qatchBlue = QColor(77, 144, 238)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.black,  2, Qt.SolidLine))
        font = painter.font()
        font.setPointSize(font.pointSize() + 5)
        painter.setFont(font)
        for x in range(self.well_cols):
            painter.drawText(
                int(x*(self.well_diameter + self.well_spacing) +
                    self.well_diameter / 2 + self.padding - 5),
                int(self.padding / 2) + 10,
                str(x+1))
            for y in range(self.well_rows):
                if x == 0:
                    painter.drawText(
                        int(self.padding / 2) - 5,
                        int(y*(self.well_diameter + self.well_spacing) +
                            self.well_diameter / 2 + self.padding + 10),
                        auc[y])
                painter.setBrush(
                    QBrush(qatchGold if self.well_states[x][y] else qatchBlue, Qt.SolidPattern))
                painter.drawEllipse(
                    self.padding + x*(self.well_diameter + self.well_spacing),
                    self.padding + y*(self.well_diameter + self.well_spacing),
                    self.well_diameter,
                    self.well_diameter)
        painter.drawText(
            int(self.padding / 2) - 5,
            self.height - self.well_diameter - self.padding - 5,
            f"Selected Well Count: {self.wells_selected}")
        painter.end()


if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = Window()
    sys.exit(App.exec())
