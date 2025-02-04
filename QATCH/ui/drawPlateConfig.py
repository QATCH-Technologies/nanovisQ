from PyQt5 import QtGui
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QMessageBox, QDesktopWidget
import json
import os
import sys
from string import ascii_uppercase as auc  # Add this import statement
# Logger and Architecture setup with fallback definitions
try:
    from QATCH.common.logger import Logger as Log
    from QATCH.common.architecture import Architecture
except ImportError:
    class Log:
        @staticmethod
        def d(message):
            print(message)

    class Architecture:
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
            (self.well_diameter + self.well_spacing)
        self.height = 4 * self.padding + self.well_rows * \
            (self.well_diameter + self.well_spacing)
        self.setFixedSize(self.width, self.height)
        self.move(QDesktopWidget().availableGeometry().center() -
                  self.frameGeometry().center())
        self._init_window()

    def _init_window(self):
        self.icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/advanced.png')
        self.setWindowIcon(QtGui.QIcon(self.icon_path))
        self.setWindowTitle(self.title)
        self.clicked.connect(self.clickEvent)
        self.create_wells()
        self._create_ui()
        self.show()

    def _create_ui(self):
        vbox = QVBoxLayout()
        self._add_help_button(vbox)
        self._add_well_buttons(vbox)
        self._add_save_cancel_buttons(vbox)
        self.setLayout(vbox)

    def _add_help_button(self, layout):
        hbox_help = QHBoxLayout()
        self.button_help = QPushButton("?")
        self.button_help.setFixedSize(34, 34)
        self.button_help.setStyleSheet(
            "border-radius: 17px; border: 2px solid black; font-weight: bold;")
        self.button_help.clicked.connect(self.show_help)
        hbox_help.addWidget(self.button_help)
        hbox_help.addStretch()
        layout.addLayout(hbox_help)
        layout.addStretch()

    def _add_well_buttons(self, layout):
        hbox = QHBoxLayout()
        hbox.addStretch()
        button_all = QPushButton("All")
        button_none = QPushButton("None")
        button_invert = QPushButton("Invert")
        button_all.clicked.connect(self.select_all)
        button_none.clicked.connect(self.select_none)
        button_invert.clicked.connect(self.select_inverse)
        hbox.addWidget(button_all)
        hbox.addWidget(button_none)
        hbox.addWidget(button_invert)
        hbox.addStretch()
        layout.addLayout(hbox)

    def _add_save_cancel_buttons(self, layout):
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
        layout.addLayout(save_cancel)

    def show_help(self):
        if not hasattr(self, 'msg'):
            self.msg = QMessageBox()
            self.msg.setWindowTitle("Help: Plate Configuration")
            self.msg.setText(
                "Use this window to select the wells on the plate that you'd like to use:")
            self.msg.setInformativeText("- Click on a well to toggle its selection\n"
                                        "- Click on a header label to toggle an entire row/column\n"
                                        "- Use \"All\", \"None\", or \"Invert\" to adjust selection\n"
                                        "Click \"Save\" to store the configuration.")
            self.msg.setWindowIcon(QtGui.QIcon(self.icon_path))
            self.msg.setIcon(QMessageBox.Question)
            self.msg.setStandardButtons(QMessageBox.Ok)
        if self.msg.isHidden():
            self.msg.move(self.button_help.mapToGlobal(QPoint(-12, -12)))
            self.msg.show()

    def save(self):
        msg = QMessageBox()
        msg.setWindowIcon(QtGui.QIcon(self.icon_path))
        msg.setStandardButtons(QMessageBox.Ok)

        if self.wells_selected > self.default_selected:
            msg.setWindowTitle("WARN: Plate Configuration")
            msg.setText(
                "You cannot select more wells than currently detected.")
            msg.setInformativeText(
                f"Selected wells: {self.wells_selected}\nDetected wells: {self.default_selected}")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()
        elif self.wells_selected == 0:
            msg.setWindowTitle("WARN: Plate Configuration")
            msg.setText("Please select at least one well.")
            msg.setInformativeText("Click \"?\" for help.")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()
        else:
            try:
                with open("plate-config.json", 'w') as f:
                    json.dump(self.well_states, f)
                msg.setWindowTitle("Saved: Plate Configuration")
                msg.setText("Plate configuration saved successfully.")
                msg.setInformativeText(
                    f"Selected: {self.wells_selected} out of {self.default_selected} wells")
                msg.setIcon(QMessageBox.NoIcon)
                msg.exec_()
                self.close()
            except Exception as e:
                msg.setWindowTitle("ERROR: Plate Configuration")
                msg.setText(f"Error saving configuration: {e}")
                msg.setIcon(QMessageBox.Critical)
                msg.exec_()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.pressPos = event.pos()

    def mouseReleaseEvent(self, event):
        if (self.pressPos and event.button() == Qt.LeftButton and event.pos() in self.rect()):
            self.clicked.emit(self.pressPos.x(), self.pressPos.y())
        self.pressPos = None

    def clickEvent(self, click_x, click_y):
        Log.d(f"Click event: {click_x}, {click_y}")
        for x in range(self.well_cols):
            for y in range(self.well_rows):
                left = self.padding + x * \
                    (self.well_diameter + self.well_spacing)
                right = left + self.well_diameter
                top = self.padding + y * \
                    (self.well_diameter + self.well_spacing)
                bottom = top + self.well_diameter
                if click_x in range(left, right) and click_y in range(top, bottom):
                    Log.d(f"Well @ ({x},{y}) was clicked!")
                    self.toggle_well_selection(x, y)
                    self.repaint()
                    return
                if x == 0 and click_x in range(0, left) and click_y in range(top, bottom):
                    for i in range(self.well_cols):
                        self.toggle_well_selection(i, y)
                    self.repaint()
                    return
            if click_x in range(left, right) and click_y in range(0, top):
                for i in range(self.well_rows):
                    self.toggle_well_selection(x, i)
                self.repaint()
                return

    def create_wells(self):
        self.well_states = [[False for _ in range(
            self.well_rows)] for _ in range(self.well_cols)]
        self.wells_selected = 0
        for x in range(self.well_cols):
            for y in range(self.well_rows):
                if self.wells_selected < self.default_selected:
                    self.well_states[x][y] = True
                    self.wells_selected += 1
        try:
            with open("plate-config.json", 'r') as f:
                saved_states = json.load(f)
            if len(saved_states) == self.well_cols and len(saved_states[0]) == self.well_rows:
                self.well_states = saved_states
                self.wells_selected = sum(sum(row) for row in self.well_states)
                Log.d("Loaded plate-config.json successfully.")
            else:
                Log.d("Plate config mismatch or load failed.")
        except Exception as e:
            Log.d(f"Failed to load plate-config.json: {e}")

    def toggle_well_selection(self, x, y):
        self.well_states[x][y] = not self.well_states[x][y]
        self.wells_selected += 1 if self.well_states[x][y] else -1
        self.repaint()

    def select_all(self):
        self.wells_selected = self.well_cols * self.well_rows
        for x in range(self.well_cols):
            for y in range(self.well_rows):
                self.well_states[x][y] = True
        self.repaint()

    def select_none(self):
        self.wells_selected = 0
        for x in range(self.well_cols):
            for y in range(self.well_rows):
                self.well_states[x][y] = False
        self.repaint()

    def select_inverse(self):
        self.wells_selected = 0
        for x in range(self.well_cols):
            for y in range(self.well_rows):
                self.well_states[x][y] = not self.well_states[x][y]
                self.wells_selected += 1 if self.well_states[x][y] else -1
        self.repaint()

    def pain_event(self, event):
        qatchGold = QColor(237, 177, 32)
        qatchBlue = QColor(77, 144, 238)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        font = painter.font()
        font.setPointSize(font.pointSize() + 5)
        painter.setFont(font)
        for x in range(self.well_cols):
            painter.drawText(int(x * (self.well_diameter + self.well_spacing) + self.well_diameter / 2 + self.padding - 5),
                             int(self.padding / 2) + 10, str(x + 1))
            for y in range(self.well_rows):
                if x == 0:
                    painter.drawText(int(self.padding / 2) - 5,
                                     int(y * (self.well_diameter + self.well_spacing) +
                                         self.well_diameter / 2 + self.padding + 10),
                                     auc[y])
                painter.setBrush(
                    QBrush(qatchGold if self.well_states[x][y] else qatchBlue, Qt.SolidPattern))
                painter.drawEllipse(self.padding + x * (self.well_diameter + self.well_spacing),
                                    self.padding + y *
                                    (self.well_diameter + self.well_spacing),
                                    self.well_diameter, self.well_diameter)
        painter.drawText(int(self.padding / 2) - 5,
                         self.height - self.well_diameter - self.padding - 5,
                         f"Selected Well Count: {self.wells_selected}")
        painter.end()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec())
