from QATCH.common.logger import Logger as Log
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QVBoxLayout, QComboBox,
                             QLabel, QLineEdit, QPushButton, QMessageBox, QFormLayout, QHBoxLayout, QButtonGroup, QRadioButton
                             )
from PyQt5.QtCore import QDateTime
from datetime import datetime
import json
TAG = '[Configure Data]'
PATH_DELIMITERS = {"Underscore": "_", "Hyphen": "-",  "Space": " "}
DATE_TIME_FORMATS = {"YYYY-MM-DD": "%Y-%m-%d",
                     "DD-MM-YYYY": "%d-%m-%Y", "MM-DD-YYYY": "%m-%d-%Y"}

# TODO: Prevent user from typing in filename and folder format text boxes.
# TODO: if the box is cleared, the list of keywords coresponding to each box should also be cleared.
# Maybe look at using drag and drop option instead?
# TODO: If the delimiter is set to something else all previous delimiters should be updated.
# TODO: Link to UserProfiles to retrieve username, initials
# TODO: Link to FileStorage to get Device ID
# TODO: Make system follow user preferences.


class UIConfigureData(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Configure Data")
        self.layout = QVBoxLayout()
        self.last_focused_input = None  # Track the last focused input field
        self.current_delimiter = list(PATH_DELIMITERS.values())[0]
        # Keywords and their descriptions
        self.keywords = {
            "Username": "User's full name",
            "Initials": "User's initials",
            "Device ID": "Device ID",
            "Run Name": "Run name",
            "Date": "Date",
            "Time": "Time"
        }
        self.keywords_for_input = {
            "folder_format_input": [],
            "filename_format_input": []
        }

        # Folder and filename format sections
        self.folder_format_label = QLabel("Folder Format")
        self.folder_format_input = QLineEdit()
        self.folder_format_input.focusInEvent = self.set_last_focused_input(
            self.folder_format_input)

        self.filename_format_label = QLabel("Filename Format")
        self.filename_format_input = QLineEdit()
        self.filename_format_input.focusInEvent = self.set_last_focused_input(
            self.filename_format_input)

        # Delimiter selection as radio buttons
        self.delimiter_label = QLabel(
            "Delimiter (used to separate keywords in path)")

        # Create a group to manage radio buttons
        self.delimiter_group = QButtonGroup()

        # Create radio buttons for each delimiter option
        self.radio_underscore = QRadioButton("Underscore (_)")
        self.radio_underscore.clicked.connect(
            lambda: self.set_delimiter(PATH_DELIMITERS.get("Underscore")))
        self.delimiter_group.addButton(self.radio_underscore)

        self.radio_hyphen = QRadioButton("Hyphen (-)")
        self.radio_hyphen.clicked.connect(
            lambda: self.set_delimiter(PATH_DELIMITERS.get("Hyphen")))
        self.delimiter_group.addButton(self.radio_hyphen)

        self.radio_space = QRadioButton("Space")
        self.radio_space.clicked.connect(
            lambda: self.set_delimiter(PATH_DELIMITERS.get("Space")))
        self.delimiter_group.addButton(self.radio_space)

        # Set default delimiter to underscore
        self.radio_underscore.setChecked(True)

        # Layout for delimiter radio buttons
        self.delimiter_buttons_layout = QVBoxLayout()
        self.delimiter_buttons_layout.addWidget(self.radio_underscore)
        self.delimiter_buttons_layout.addWidget(self.radio_hyphen)
        self.delimiter_buttons_layout.addWidget(self.radio_space)

        # Buttons for inserting keywords into format fields
        self.keyword_buttons_label = QLabel("Insert Keywords")
        self.keyword_buttons_layout = QHBoxLayout()

        for key, desc in self.keywords.items():
            button = QPushButton(key)
            button.setToolTip(desc)
            button.clicked.connect(lambda _, k=key: self.insert_keyword(k))
            self.keyword_buttons_layout.addWidget(button)

        # Dropdown menus for date and time formats
        self.date_format_label = QLabel("Date Format")
        self.date_format_dropdown = QComboBox()
        self.date_format_dropdown.addItems(list(DATE_TIME_FORMATS.keys()))
        self.date_format_dropdown.setCurrentText(
            list(DATE_TIME_FORMATS.keys())[0])
        self.date_format_dropdown.currentIndexChanged.connect(
            self.generate_preview)

        self.time_format_label = QLabel("Time Format")
        self.time_format_dropdown = QComboBox()
        self.time_format_dropdown.addItems(
            ["HH:mm:ss", "hh:mm:ss A", "HH:mm", "hh:mm A"])
        self.time_format_dropdown.setCurrentText("HH:mm:ss")
        self.time_format_dropdown.currentIndexChanged.connect(
            self.generate_preview)

        # Output preview
        self.preview_label = QLabel("Output Preview")
        self.preview_output = QLabel("")
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.generate_preview)

        # Save button
        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)

        # Form layout
        form_layout = QFormLayout()
        form_layout.addRow(self.folder_format_label, self.folder_format_input)
        form_layout.addRow(self.filename_format_label,
                           self.filename_format_input)
        form_layout.addRow(self.delimiter_label, self.delimiter_buttons_layout)
        form_layout.addRow(self.date_format_label, self.date_format_dropdown)
        form_layout.addRow(self.time_format_label, self.time_format_dropdown)

        # Add components to main layout
        self.layout.addLayout(form_layout)
        self.layout.addWidget(self.keyword_buttons_label)
        self.layout.addLayout(self.keyword_buttons_layout)
        self.layout.addWidget(self.preview_label)
        self.layout.addWidget(self.preview_output)
        self.layout.addWidget(self.preview_button)
        self.layout.addWidget(self.save_button)
        self.setLayout(self.layout)

    def set_delimiter(self, delimiter):
        self.current_delimiter = delimiter

    def set_last_focused_input(self, input_field):
        """Helper function to update the last focused input field."""
        def focus_event(event):
            self.last_focused_input = input_field
            # Preserve default behavior
            QLineEdit.focusInEvent(input_field, event)
        return focus_event

    def insert_keyword(self, keyword):
        """Insert the selected keyword into the last focused input field."""

        if self.last_focused_input:
            # Insert the keyword into the input field
            # Add the delimiter only if there is more than one keyword and it's not the last one
            input_name = "folder_format_input" if self.last_focused_input == self.folder_format_input else "filename_format_input"

            if len(self.keywords_for_input[input_name]) > 0:
                self.last_focused_input.insert(self.current_delimiter)
                self.keywords_for_input[input_name].append(
                    self.current_delimiter)

            self.last_focused_input.insert(keyword)

            # Track the keywords in the specific input box
            self.keywords_for_input[input_name].append(keyword)

        else:
            QMessageBox.warning(
                self, "Error", "Please click on a format field before inserting a keyword.")

    def get_folder_format(self):
        return self.keywords_for_input.get("folder_format_input")

    def get_filename_format(self):
        return self.keywords_for_input.get("filename_format_input")

    def generate_preview(self):
        """Generate a preview of the folder and file structure based on example data."""
        # Example data for preview
        example_data = {
            "Username": "Paul MacNichol",
            "Initials": "PEM",
            "Device ID": "12345678",
            "Run Name": "Test Run",
            "Date": datetime.now().strftime(DATE_TIME_FORMATS.get(self.date_format_dropdown.currentText())),
            "Time": QDateTime.currentDateTime().toString(self.time_format_dropdown.currentText())
        }

        # Folder and file formats
        folder_format = self.folder_format_input.text()
        filename_format = self.filename_format_input.text()
        delimiter = self.current_delimiter

        # Replace keywords in preview
        def update_format(format_keywords):
            update_string = ""
            for keyword in format_keywords:
                if keyword not in PATH_DELIMITERS.values():
                    update_string += example_data.get(keyword)
                else:
                    update_string += keyword
            return update_string

        folder_format = update_format(self.get_folder_format())
        filename_format = update_format(self.get_filename_format())
        date_format = example_data.get("Date")
        time_format = example_data.get("Time")
        # Set preview output
        self.preview_output.setText(
            f"Folder: {folder_format}\nFile: {filename_format}\nDate: {date_format}\nTime: {time_format}")

    def save_settings(self):
        """Save the user-defined settings to a file."""
        folder_format = self.folder_format_input.text()
        filename_format = self.filename_format_input.text()
        delimiter = self.current_delimiter
        date_format = self.date_format_dropdown.currentText()
        time_format = self.time_format_dropdown.currentText()

        # Simple validation
        if not folder_format or not filename_format:
            QMessageBox.warning(
                self, "Error", "Folder and Filename formats cannot be empty!")
            return

        settings = {
            "folder_format": folder_format,
            "filename_format": filename_format,
            "delimiter": delimiter,
            "date_format": date_format,
            "time_format": time_format
        }

        try:
            # Save settings to preferences.json
            with open("file-preferences.json", "w") as f:
                json.dump(settings, f, indent=4)

            QMessageBox.information(
                self, "Saved", "Settings have been saved successfully!")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save settings: {e}")
