import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QLineEdit, QPushButton, QWidget, QVBoxLayout, QMessageBox, QMenu
)
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QDrag


# Define tags and delimiters
tags = ["%username%", "%initials%", "%device%",
        "%runname%", "%date%", "%time%", "%port%"]
delimiters = {"Underscore": "_", "Hyphen": "-", "Space": " "}


class DraggableButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setFixedSize(150, 30)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def mouseMoveEvent(self, event):
        if event.buttons() != Qt.LeftButton:
            return

        mime_data = QMimeData()
        mime_data.setText(self.text())

        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec_(Qt.MoveAction)

    def show_context_menu(self, pos):
        menu = QMenu(self)
        remove_action = menu.addAction("Remove Tag")
        action = menu.exec_(self.mapToGlobal(pos))
        if action == remove_action:
            self.parent().remove_widget(self)


class DroppableArea(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFixedHeight(150)
        self.setStyleSheet("border: 1px solid black;")
        self.layout = QHBoxLayout(self)
        self.layout.setAlignment(Qt.AlignLeft)
        self.setLayout(self.layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            # Change background color on hover
            self.setStyleSheet(
                "background-color: lightblue; border: 1px solid black;")
            event.accept()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        # Reset the background color when dragging leaves the area
        self.setStyleSheet("background-color: none; border: 1px solid black;")

    def dropEvent(self, event):
        # Reset the background color when the drop occurs
        self.setStyleSheet("background-color: none; border: 1px solid black;")

        # Check if the drop is outside the droppable area
        if not self.geometry().contains(event.pos()):
            return  # Do not accept the drop if it's inside the box

        # Proceed if the dropped data contains text
        if event.mimeData().hasText():
            text = event.mimeData().text()

            # Check if the tag is already in the Set Tag Order box
            if any(widget.text() == text for widget in self.layout.children()):
                return  # Don't add the tag if it's already in the box

            # Add the tag button to the layout if it's not already present
            self.add_button_to_layout(text)
            event.accept()

    def add_button_to_layout(self, text):
        tag_button = DraggableButton(text)
        tag_button.setParent(self)
        tag_button.setAcceptDrops(True)
        self.layout.addWidget(tag_button)

    def remove_widget(self, widget):
        self.layout.removeWidget(widget)
        widget.deleteLater()

    def dragMoveEvent(self, event):
        # Detect which button is being dragged over
        for widget in self.layout.children():
            if widget.geometry().contains(event.pos()):
                widget.setStyleSheet("border: 1px solid red;")
            else:
                widget.setStyleSheet("")

        # Reorder the widget if dragged over another
        current_dragged_widget = self.childAt(event.pos())
        if current_dragged_widget:
            dragged_widget_text = event.mimeData().text()

            if current_dragged_widget and current_dragged_widget.text() != dragged_widget_text:
                # Move widget to the new position
                for i in range(self.layout.count()):
                    widget = self.layout.itemAt(i).widget()
                    if widget.text() == dragged_widget_text:
                        # Remove the dragged widget from its original position
                        self.layout.removeWidget(current_dragged_widget)
                        # Insert it at the new position
                        self.layout.insertWidget(i, current_dragged_widget)
                        break

            self.layout.update()


class FileNamingUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Naming Preferences")
        self.setGeometry(100, 100, 800, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.tag_selection_layout = QHBoxLayout()
        self.tag_label = QLabel("Available Tags:")
        self.tag_selection_layout.addWidget(self.tag_label)

        self.buttons = []
        for tag in tags:
            button = DraggableButton(tag)
            self.buttons.append(button)
            self.tag_selection_layout.addWidget(button)

        self.layout.addLayout(self.tag_selection_layout)

        # Select Delimiter layout
        self.delimiter_layout = QHBoxLayout()
        self.delimiter_label = QLabel("Select Delimiter:")
        self.delimiter_dropdown = QComboBox()
        self.delimiter_dropdown.addItems(delimiters.keys())
        self.delimiter_layout.addWidget(self.delimiter_label)
        self.delimiter_layout.addWidget(self.delimiter_dropdown)

        self.layout.addLayout(self.delimiter_layout)

        # Set Tag Order layout
        self.order_layout = QVBoxLayout()
        self.order_label = QLabel("Set Tag Order (Drag and Drop Tags):")
        self.order_layout.addWidget(self.order_label)

        self.droppable_area = DroppableArea()
        self.order_layout.addWidget(self.droppable_area)

        self.layout.addLayout(self.order_layout)

        # Preview Layout
        self.preview_layout = QVBoxLayout()
        self.preview_label = QLabel("Preview File Name:")
        self.preview_line = QLineEdit()
        self.preview_line.setReadOnly(True)
        self.preview_layout.addWidget(self.preview_label)
        self.preview_layout.addWidget(self.preview_line)

        self.layout.addLayout(self.preview_layout)

        # Buttons Layout
        self.buttons_layout = QHBoxLayout()
        self.preview_button = QPushButton("Generate Preview")
        self.save_button = QPushButton("Save Preferences")
        self.buttons_layout.addWidget(self.preview_button)
        self.buttons_layout.addWidget(self.save_button)

        self.layout.addLayout(self.buttons_layout)

        self.central_widget.setLayout(self.layout)

        # Signals
        self.preview_button.clicked.connect(self.generate_preview)
        self.save_button.clicked.connect(self.save_preferences)

    def generate_preview(self):
        tag_order = [self.droppable_area.layout.itemAt(i).widget().text()
                     for i in range(self.droppable_area.layout.count())]
        delimiter = delimiters[self.delimiter_dropdown.currentText()]

        if not tag_order:
            QMessageBox.warning(
                self, "No Tags in Order", "Please add at least one tag to the order to generate a preview.")
            return

        preview_name = delimiter.join(tag_order)
        self.preview_line.setText(preview_name)

    def save_preferences(self):
        tag_order = [self.droppable_area.layout.itemAt(i).widget().text()
                     for i in range(self.droppable_area.layout.count())]
        delimiter = delimiters[self.delimiter_dropdown.currentText()]

        if not tag_order:
            QMessageBox.warning(
                self, "No Tags in Order", "Please add at least one tag to the order to save preferences.")
            return

        preferences = {
            "tags": tag_order,
            "delimiter": delimiter
        }

        QMessageBox.information(self, "Preferences Saved",
                                f"Preferences saved successfully:\n{preferences}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FileNamingUI()
    window.show()
    sys.exit(app.exec_())
