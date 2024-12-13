import sys
from QATCH.common.logger import Logger as Log
from QATCH.common.fileStorage import FileStorage
from QATCH.common.userProfiles import UserProfiles
from QATCH.core.constants import Constants
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabWidget, QComboBox, QPushButton, QHBoxLayout, QLabel, QCheckBox
from PyQt5.QtCore import Qt

TAG = '[Preferences]'
SELECT_TAG_PROMPT = '-- Select Tag --'


class PreferencesUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('File Naming Preferences')

        # Initialize the _updating flag to avoid recursion
        self._updating = False  # Initialize the _updating flag

        # Set fixed size for the window (width, height) based on the desired size
        # Adjust as needed for the window size in your screenshot
        self.setFixedSize(600, 400)

        # Layout for the main window
        main_layout = QVBoxLayout()

        # Tab widget to contain both tabs
        tab_widget = QTabWidget()

        # Date and time preferences tab
        date_time_tab = QWidget()
        date_time_layout = QVBoxLayout()

        # Date format dropdown
        date_format_layout = QHBoxLayout()  # Horizontal layout for date format
        date_format_label = QLabel("Date Format:")
        date_format_layout.addWidget(date_format_label)
        self.date_format_combo = QComboBox()
        self.date_format_combo.addItems(
            Constants.date_formats)
        date_format_layout.addWidget(self.date_format_combo)
        date_time_layout.addLayout(date_format_layout)

        # Time format dropdown
        time_format_layout = QHBoxLayout()  # Horizontal layout for time format
        time_format_label = QLabel("Time Format:")
        time_format_layout.addWidget(time_format_label)
        self.time_format_combo = QComboBox()
        self.time_format_combo.addItems(
            Constants.time_formats)
        time_format_layout.addWidget(self.time_format_combo)
        date_time_layout.addLayout(time_format_layout)

        # Preview button for Date and Time format
        self.preview_date_time_button = QPushButton(
            "Preview Date & Time Format")
        self.preview_date_time_button.clicked.connect(
            self.preview_date_time_format)
        self.preview_date_time_label = QLabel("Preview will appear here.")
        preview_layout = QVBoxLayout()  # Layout for preview button and label
        preview_layout.addWidget(self.preview_date_time_button)
        preview_layout.addWidget(self.preview_date_time_label)
        date_time_layout.addLayout(preview_layout)

        date_time_tab.setLayout(date_time_layout)

        # File and folder format preferences tab
        file_folder_tab = QWidget()
        file_folder_layout = QVBoxLayout()

        # Label for file and folder format instructions
        format_label = QLabel("Select tags for the file and folder format:")
        file_folder_layout.addWidget(format_label)

        # Tags available
        self.tags = Constants.valid_tags

        # Initialize the list of selected tags
        self.selected_tags = []

        # File Format Section
        file_format_label = QLabel("File Format:")
        file_folder_layout.addWidget(file_format_label)

        # Create a container to hold the file format dropdowns horizontally
        self.file_format_container = QHBoxLayout()
        self.file_format_combos = []  # List to keep track of file format combo boxes
        self.add_dropdown(self.file_format_container)  # Add the first dropdown
        file_folder_layout.addLayout(self.file_format_container)

        # File Format Button and Delimiter
        file_button_layout = QHBoxLayout()
        file_button_layout.setAlignment(
            Qt.AlignLeft)  # Align buttons to the left
        add_file_button = QPushButton("+")
        add_file_button.setFixedSize(40, 30)  # Set fixed size for the button
        add_file_button.clicked.connect(
            lambda: self.add_dropdown(self.file_format_container))
        file_button_layout.addWidget(add_file_button)

        remove_file_button = QPushButton("-")
        # Set fixed size for the button
        remove_file_button.setFixedSize(40, 30)
        remove_file_button.clicked.connect(
            lambda: self.remove_last_dropdown(self.file_format_container))
        file_button_layout.addWidget(remove_file_button)

        # Delimiter selection for file format
        self.file_delimiter_combo = QComboBox()
        # Set fixed size for the delimiter dropdown
        self.file_delimiter_combo.setFixedSize(40, 30)
        self.file_delimiter_combo.addItems(Constants.path_delimiters)
        file_button_layout.addWidget(self.file_delimiter_combo)

        file_folder_layout.addLayout(file_button_layout)

        # Folder Format Section
        folder_format_label = QLabel("Folder Format:")
        file_folder_layout.addWidget(folder_format_label)

        # Create a container to hold the folder format dropdowns horizontally
        self.folder_format_container = QHBoxLayout()
        self.folder_format_combos = []  # List to keep track of folder format combo boxes
        # Add the first dropdown
        self.add_dropdown(self.folder_format_container)
        file_folder_layout.addLayout(self.folder_format_container)

        # Folder Format Button and Delimiter
        folder_button_layout = QHBoxLayout()
        folder_button_layout.setAlignment(
            Qt.AlignLeft)  # Align buttons to the left
        add_folder_button = QPushButton("+")
        add_folder_button.setFixedSize(40, 30)  # Set fixed size for the button
        add_folder_button.clicked.connect(
            lambda: self.add_dropdown(self.folder_format_container))
        folder_button_layout.addWidget(add_folder_button)

        remove_folder_button = QPushButton("-")
        # Set fixed size for the button
        remove_folder_button.setFixedSize(40, 30)
        remove_folder_button.clicked.connect(
            lambda: self.remove_last_dropdown(self.folder_format_container))
        folder_button_layout.addWidget(remove_folder_button)

        # Delimiter selection for folder format
        self.folder_delimiter_combo = QComboBox()
        # Set fixed size for the delimiter dropdown
        self.folder_delimiter_combo.setFixedSize(40, 30)
        self.folder_delimiter_combo.addItems(
            Constants.path_delimiters)
        folder_button_layout.addWidget(self.folder_delimiter_combo)

        file_folder_layout.addLayout(folder_button_layout)

        # Preview button and label
        self.preview_button = QPushButton("Preview Format")
        self.preview_button.clicked.connect(self.preview_format)
        self.preview_label = QLabel("Preview will appear here.")
        file_folder_layout.addWidget(self.preview_button)
        file_folder_layout.addWidget(self.preview_label)

        file_folder_tab.setLayout(file_folder_layout)

        # Add tabs to the tab widget
        tab_widget.addTab(date_time_tab, "Date and Time Preferences")
        tab_widget.addTab(file_folder_tab, "File and Folder Preferences")

        # Add the tab widget to the main layout
        main_layout.addWidget(tab_widget)
        # Label to display status of global preferences
        self.global_pref_label = QLabel("Use global preferences: OFF")
        main_layout.addWidget(self.global_pref_label)

        # Toggle checkbox for global preferences
        self.global_pref_toggle = QCheckBox("Use global preferences", self)
        self.global_pref_toggle.stateChanged.connect(
            self.toggle_global_preferences)
        main_layout.addWidget(self.global_pref_toggle)
        # Submit button
        submit_button = QPushButton('Save Preferences')
        submit_button.clicked.connect(self.save_preferences)
        main_layout.addWidget(submit_button)

        # Set the layout for the window
        self.setLayout(main_layout)

    def add_dropdown(self, layout):
        """Add a new dropdown to the layout."""
        # Determine the correct layout based on the section (file or folder)
        if layout == self.file_format_container:
            combo_list = self.file_format_combos
        else:
            combo_list = self.folder_format_combos

        if len(combo_list) < len(self.tags):  # Ensure no more than 7 dropdowns
            combo = QComboBox()
            # Add the "Select tag" placeholder as the first item
            combo.addItem(SELECT_TAG_PROMPT)
            # Filter out already selected tags from the available options
            available_tags = [
                tag for tag in self.tags if tag not in self.selected_tags]
            combo.addItems(available_tags)

            layout.addWidget(combo)
            combo_list.append(combo)  # Add the combo to the respective list

    def remove_last_dropdown(self, layout):
        """Remove the last dropdown from the given layout, if there is more than one."""
        # Ensure there's more than one dropdown in the layout before allowing removal
        if layout == self.file_format_container and len(self.file_format_combos) > 1:
            self.remove_dropdown(self.file_format_combos[-1], layout)
        elif layout == self.folder_format_container and len(self.folder_format_combos) > 1:
            self.remove_dropdown(self.folder_format_combos[-1], layout)

    def remove_dropdown(self, combo, layout):
        """Remove a dropdown and its corresponding remove button."""
        # Determine the correct combo list based on the section (file or folder)
        if layout == self.file_format_container:
            combo_list = self.file_format_combos
        else:
            combo_list = self.folder_format_combos

        # Remove the combo from the list and layout
        if combo in combo_list:
            combo_list.remove(combo)
            layout.removeWidget(combo)

            # Delete the combo box widget itself
            combo.deleteLater()

            # Update the selected tags list to allow the tag to be selected again
            current_tag = combo.currentText()
            if current_tag != SELECT_TAG_PROMPT and current_tag in self.selected_tags:
                self.selected_tags.remove(current_tag)

    def toggle_global_preferences(self):
        if self.global_pref_toggle.isChecked():
            self.global_pref_label.setText("Use global preferences: ON")
            # Load and apply global preferences here
            self.load_global_preferences()
        else:
            self.global_pref_label.setText("Use global preferences: OFF")
            # Allow user-defined preferences
            self.load_user_preferences()

    def preview_date_time_format(self):
        """Preview the date and time format based on selected options."""
        date_format = self.date_format_combo.currentText()
        time_format = self.time_format_combo.currentText()

        # Example date and time string to preview
        from datetime import datetime
        current_datetime = datetime.now()

        formatted_date = current_datetime.strftime(date_format)
        formatted_time = current_datetime.strftime(time_format)

        preview_text = f"Date: {formatted_date}\nTime: {formatted_time}"
        self.preview_date_time_label.setText(preview_text)

    def load_global_preferences(self):
        UserProfiles.user_preferences.set_use_global(use_global=True)
        UserProfiles.user_preferences.set_preferences()

    def load_user_preferences(self):
        UserProfiles.user_preferences.set_use_global(use_global=False)
        UserProfiles.user_preferences.set_preferences()

    def preview_format(self):
        """Preview the file and folder format based on selected tags."""
        file_format = [combo.currentText()
                       for combo in self.file_format_combos]
        folder_format = [combo.currentText()
                         for combo in self.folder_format_combos]
        file_delimiter = self.file_delimiter_combo.currentText()
        folder_delimiter = self.folder_delimiter_combo.currentText()
        device_id = FileStorage.DEV_get_active(0)
        if device_id == "":
            device_id = "12345678"
        port_id = 1
        # Generate a preview string based on the selected format
        file_preview = UserProfiles.user_preferences._build_save_path(
            file_format, "Runname", file_delimiter, device_id, port_id)
        folder_preview = UserProfiles.user_preferences._build_save_path(
            folder_format, "Runname", folder_delimiter, device_id, port_id)

        preview_text = f"File Format Preview: {file_preview}\nFolder Format Preview: {folder_preview}"
        self.preview_label.setText(preview_text)

    def save_preferences(self):
        date_format = self.date_format_combo.currentText()
        time_format = self.time_format_combo.currentText()

        # Get file and folder formats
        file_format = [combo.currentText()
                       for combo in self.file_format_combos]
        folder_format = [combo.currentText()
                         for combo in self.folder_format_combos]
        file_delimiter = self.file_delimiter_combo.currentText()
        folder_delimiter = self.folder_delimiter_combo.currentText()

        file_format_pattern = ""
        folder_format_pattern = ""
        for i, tok in enumerate(file_format):
            file_format_pattern = file_format_pattern + tok
            if i < len(file_format) - 1:
                file_format_pattern = file_format_pattern + file_delimiter
        for i, tok in enumerate(folder_format):
            folder_format_pattern = folder_format_pattern + tok
            if i < len(folder_format) - 1:
                folder_format_pattern = folder_format_pattern + folder_delimiter

        UserProfiles.user_preferences._set_date_format(date_format=date_format)
        UserProfiles.user_preferences._set_time_format(time_format=time_format)
        UserProfiles.user_preferences._set_file_delimiter(
            file_delimiter=file_delimiter)
        UserProfiles.user_preferences._set_folder_delimiter(
            folder_delimiter=folder_delimiter)
        UserProfiles.user_preferences._set_file_format_pattern(
            file_format_pattern=file_format_pattern)
        UserProfiles.user_preferences._set_folder_format_pattern(
            folder_format_pattern=folder_format_pattern)
        UserProfiles.user_preferences.write_user_preferences()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PreferencesUI()
    window.show()
    sys.exit(app.exec_())
