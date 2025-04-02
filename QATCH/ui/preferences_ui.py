from QATCH.common.logger import Logger as Log
from QATCH.common.fileStorage import FileStorage
from QATCH.common.userProfiles import UserProfiles, UserRoles, UserPreferences
from QATCH.core.constants import Constants
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QTabWidget, QComboBox, QPushButton, QHBoxLayout, QLabel, QCheckBox, QLineEdit, QFileDialog, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import os
TAG = '[Preferences]'
SELECT_TAG_PROMPT = Constants.select_tag_prompt
SUBFOLDER_FIELD = Constants.subfolder_field


class PreferencesUI(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        # Assume non-admin user role until proven otherwise
        self._is_admin = False  # call 'check_user_role()' to set

        # Create default user preferences object (initially "global" only)
        UserProfiles.user_preferences = UserPreferences(
            UserProfiles.get_session_file())
        UserProfiles.user_preferences.set_preferences()

        self.setWindowTitle('Preferences')
        self.setWindowIcon(QIcon(r"QATCH\icons\preferences_icon.png"))

        # Initialize the _updating flag to avoid recursion
        self._updating = False  # Initialize the _updating flag

        # Set minimum size for the window (width, height) based on the desired size
        self.setMinimumSize(550, 475)

        # Layout for the main window
        main_layout = QVBoxLayout()

        # Tab widget to contain both tabs
        self.tab_widget = QTabWidget()

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
            "Preview Date && Time Format")
        self.preview_date_time_button.clicked.connect(
            self.preview_date_time_format)
        self.preview_date_time_label = QLabel("Preview will appear here.")
        preview_layout = QVBoxLayout()  # Layout for preview button and label
        preview_layout.addWidget(self.preview_date_time_button)
        preview_layout.addStretch()
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
        self.file_delimiter_combo.setFixedSize(40, 28)
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
        self.preview_button = QPushButton("Preview File && Folder Format")
        self.preview_button.clicked.connect(self.preview_format)
        self.preview_label = QLabel("Preview will appear here.")
        file_folder_layout.addWidget(self.preview_button)
        file_folder_layout.addStretch()
        file_folder_layout.addWidget(self.preview_label)

        file_folder_tab.setLayout(file_folder_layout)
        # Default Data Path Tab
        default_data_tab = QWidget()
        # Main layout
        default_data_layout = QVBoxLayout()

        # Load directory section
        load_label = QLabel("Select or input a load directory:")
        load_label.setAlignment(Qt.AlignLeft)
        default_data_layout.addWidget(load_label)

        load_directory_layout = QHBoxLayout()
        self.load_directory_input = QLineEdit()
        load_directory_layout.addWidget(self.load_directory_input)

        load_browse_button = QPushButton("Browse")
        load_browse_button.clicked.connect(self.open_load_file_dialog)
        load_directory_layout.addWidget(load_browse_button)

        default_data_layout.addLayout(load_directory_layout)

        # Write directory section
        write_directory_layout_1 = QHBoxLayout()
        write_label = QLabel("Select or input a write directory:")
        write_label.setAlignment(Qt.AlignLeft)
        write_directory_layout_1.addWidget(write_label)

        self.sync_write_with_load = QCheckBox("Same as load directory")
        self.sync_write_with_load.stateChanged.connect(self.toggle_folder_sync)
        if True:  # 2 layout options available (use one or the other)
            write_directory_layout_1.addWidget(
                self.sync_write_with_load, alignment=Qt.AlignRight)
        else:
            self.sync_write_with_load.setLayoutDirection(Qt.RightToLeft)
            write_directory_layout_1.addWidget(self.sync_write_with_load)

        default_data_layout.addLayout(write_directory_layout_1)

        write_directory_layout_2 = QHBoxLayout()
        self.write_directory_input = QLineEdit()
        write_directory_layout_2.addWidget(self.write_directory_input)

        self.write_browse_button = QPushButton("Browse")
        self.write_browse_button.clicked.connect(self.open_write_file_dialog)
        write_directory_layout_2.addWidget(self.write_browse_button)

        default_data_layout.addLayout(write_directory_layout_2)

        # Manually fire 'stateChanged' event on UI initialization
        # NOTE: This must be after ALL write layout objects created
        self.toggle_folder_sync(self.sync_write_with_load.checkState())

        # Spacer to push elements down if necessary
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        default_data_layout.addWidget(spacer)

        default_data_tab.setLayout(default_data_layout)

        # Add tabs to the tab widget
        self.tab_widget.addTab(date_time_tab, "Date and Time Preferences")
        self.tab_widget.addTab(file_folder_tab, "File and Folder Preferences")
        self.tab_widget.addTab(default_data_tab, "Default Data Paths")
        # Connect tab change signal to handler
        self.tab_widget.currentChanged.connect(self.handle_tab_change)

        # Add the tab widget to the main layout
        main_layout.addWidget(self.tab_widget)
        # Label to display status of global preferences
        self.global_pref_label = QLabel("Use global preferences: OFF")
        main_layout.addWidget(self.global_pref_label)

        # Toggle checkbox for global preferences
        self.global_pref_toggle = QCheckBox("Use global preferences", self)
        self.global_pref_toggle.stateChanged.connect(
            self.toggle_global_preferences)
        main_layout.addWidget(self.global_pref_toggle)
        # Submit button
        self.submit_button = QPushButton('Save Preferences')
        self.submit_button.clicked.connect(self.save_preferences)
        main_layout.addWidget(self.submit_button)
        reset_button = QPushButton('Reset to Default Preferences')
        reset_button.clicked.connect(self.reset_to_default_preferences)
        main_layout.addWidget(reset_button)
        # Set the layout for the window
        self.setLayout(main_layout)

    def showNormal(self, tab_idx=0):
        super(PreferencesUI, self).hide()
        super(PreferencesUI, self).showNormal()
        self.resize(self.minimumSize())

        # Reset labels to un-previewed states
        self.preview_date_time_label.setText("Preview will appear here.")
        self.preview_label.setText("Preview will appear here.")

        self.check_user_role()  # updates self._is_admin
        if UserProfiles().count() > 0 and self._is_admin is not None:
            if not self.global_pref_toggle.isChecked():
                # Toggle on first, forcing change handler to emit
                self.global_pref_toggle.setChecked(True)
            self.global_pref_toggle.setChecked(False)
            self.global_pref_toggle.setEnabled(True)
        else:  # is None:
            if self.global_pref_toggle.isChecked():
                # Toggle off first, forcing change handler to emit
                self.global_pref_toggle.setChecked(False)
            self.global_pref_toggle.setChecked(True)
            self.global_pref_toggle.setEnabled(False)

        self.tab_widget.setCurrentIndex(tab_idx)

    def check_user_role(self):
        self._is_admin = UserProfiles.check(
            self.parent.userrole, UserRoles.ADMIN)

    def handle_tab_change(self, index):
        """Handle tab change and load preferences if needed."""
        # Check if the file and folder preferences tab is selected
        if self.global_pref_toggle.isChecked():
            self.load_global_preferences()
        elif hasattr(UserProfiles.user_preferences, "_user_preferences_path"):
            self.load_user_preferences()

    def open_load_file_dialog(self):
        # Open file dialog for directory selection
        selected_directory = QFileDialog.getExistingDirectory(
            self, "Select Directory")
        if selected_directory:
            # Set the selected directory path to the input field
            self.load_directory_input.setText(selected_directory)
            # Sync with write folder (if sync is checked)
            if self.sync_write_with_load.isChecked():
                self.write_directory_input.setText(selected_directory)

    def open_write_file_dialog(self):
        # Open file dialog for directory selection
        selected_directory = QFileDialog.getExistingDirectory(
            self, "Select Directory")
        if selected_directory:
            # Set the selected directory path to the input field
            self.write_directory_input.setText(selected_directory)

    def toggle_folder_sync(self, state):
        is_synced = True if (state == Qt.CheckState.Checked) else False
        self.write_directory_input.setEnabled(not is_synced)
        self.write_browse_button.setEnabled(not is_synced)
        if is_synced:
            self.write_directory_input.setText(
                self.load_directory_input.text())
        if self.sync_write_with_load.checkState() != state:
            self.sync_write_with_load.setCheckState(state)

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
            # # Add "Subfolder" field as the 2nd item for folder format only
            # if combo_list != self.file_format_combos:
            #     combo.addItem(SUBFOLDER_FIELD)
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
        is_checked = self.global_pref_toggle.isChecked()
        self.check_user_role()  # updates self._is_admin

        # Update the label
        self.global_pref_label.setText(
            "Use global preferences: ON" if is_checked else "Use global preferences: OFF")

        # Load preferences
        if is_checked:
            self.load_global_preferences()
            if self._is_admin is None:
                self.submit_button.setEnabled(False)
            else:
                # Allows global pref file write when there are no user profiles in the system
                # thanks to the '_is_admin' flag returning True when no user profiles exist
                self.submit_button.setEnabled(True)
        else:  # not checked
            if hasattr(UserProfiles.user_preferences, "_user_preferences_path"):
                self.load_user_preferences()
            self.submit_button.setEnabled(True)

        return  # skip the rest of this method, we only disable the submit button now

        if not self._is_admin:
            # Disable/Enable Date and Time dropdowns
            self.date_format_combo.setEnabled(not is_checked)
            self.time_format_combo.setEnabled(not is_checked)

            # Disable/Enable File Format Section
            self.file_delimiter_combo.setEnabled(not is_checked)
            for combo in self.file_format_combos:
                # Disable all file format dropdowns
                combo.setEnabled(not is_checked)

            # Disable/Enable Folder Format Section
            self.folder_delimiter_combo.setEnabled(not is_checked)
            for combo in self.folder_format_combos:
                # Disable all folder format dropdowns
                combo.setEnabled(not is_checked)

            # Explicitly disable add/remove buttons
            self.disable_add_remove_buttons(is_checked)

            # Ensure all dropdowns in the layouts are disabled dynamically
            self.disable_all_combos_in_layout(
                self.file_format_container, not is_checked)
            self.disable_all_combos_in_layout(
                self.folder_format_container, not is_checked)

    def disable_add_remove_buttons(self, disable):
        """Disable add and remove buttons for file and folder format layouts."""
        for button in self.findChildren(QPushButton):
            if button.text() in ["+", "-"]:
                button.setEnabled(not disable)

    def disable_all_combos_in_layout(self, layout, enable):
        """Disable all QComboBox widgets in a layout."""
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item:
                widget = item.widget()
                if isinstance(widget, QComboBox):
                    widget.setEnabled(not enable)
                elif isinstance(widget, QWidget) and widget.layout():
                    # Recursively disable combo boxes in nested layouts
                    self.disable_all_combos_in_layout(widget.layout(), enable)

    def preview_date_time_format(self):
        """Preview the date and time format based on selected options."""
        date_format = self.date_format_combo.currentText()
        time_format = self.time_format_combo.currentText()

        # Example date and time string to preview
        from datetime import datetime
        current_datetime = datetime.now()

        formatted_date = current_datetime.strftime(
            Constants.date_conversion.get(date_format))
        formatted_time = current_datetime.strftime(
            Constants.time_conversion.get(time_format))

        preview_text = f"Date: {formatted_date}\nTime: {formatted_time}"
        self.preview_date_time_label.setText(preview_text)

    def load_global_preferences(self):
        UserProfiles.user_preferences.set_use_global(use_global=True)
        global_preferences = UserProfiles.user_preferences.load_global_preferences()
        UserProfiles.user_preferences.set_preferences()
        # Update the folder sync state based on the dictionary
        paths_synced = Qt.CheckState.Checked if (
            global_preferences["load_data_path"] == global_preferences["write_data_path"]
        ) else Qt.CheckState.Unchecked
        self.toggle_folder_sync(paths_synced)
        # Reset the load and write paths based on the dictionary
        self.load_directory_input.setText(global_preferences["load_data_path"])
        self.write_directory_input.setText(
            global_preferences["write_data_path"])
        # Reset the date and time format dropdowns based on the dictionary
        self.date_format_combo.setCurrentText(
            global_preferences["date_format"])
        self.time_format_combo.setCurrentText(
            global_preferences["time_format"])

        # Reset file format
        file_format = global_preferences["filename_format"].split(
            global_preferences["filename_format_delimiter"])
        self.set_file_format_dropdowns(
            file_format, global_preferences["filename_format_delimiter"])

        # Reset folder format
        folder_format = global_preferences["folder_format"].split(
            global_preferences["folder_format_delimiter"])
        self.set_folder_format_dropdowns(
            folder_format, Constants.default_preferences["folder_format_delimiter"])

        # Reset delimiters
        self.file_delimiter_combo.setCurrentText(
            global_preferences["filename_format_delimiter"])
        self.folder_delimiter_combo.setCurrentText(
            global_preferences["folder_format_delimiter"])
        # Reset global preferences toggle
        self.global_pref_toggle.setChecked(True)
        self.global_pref_label.setText("Use global preferences: ON")

    def load_user_preferences(self):
        UserProfiles.user_preferences.set_use_global(use_global=False)
        user_preferences = UserProfiles.user_preferences.load_user_preferences()
        UserProfiles.user_preferences.set_preferences()
        # Update the folder sync state based on the dictionary
        paths_synced = Qt.CheckState.Checked if (
            user_preferences["load_data_path"] == user_preferences["write_data_path"]
        ) else Qt.CheckState.Unchecked
        self.toggle_folder_sync(paths_synced)
        # Reset the load and write paths based on the dictionary
        self.load_directory_input.setText(user_preferences["load_data_path"])
        self.write_directory_input.setText(
            user_preferences["write_data_path"])
        # Reset the date and time format dropdowns based on the dictionary
        self.date_format_combo.setCurrentText(
            user_preferences["date_format"])
        self.time_format_combo.setCurrentText(
            user_preferences["time_format"])

        # Reset file format
        file_format = user_preferences["filename_format"].split(
            user_preferences["filename_format_delimiter"])
        self.set_file_format_dropdowns(
            file_format, user_preferences["filename_format_delimiter"])

        # Reset folder format
        folder_format = user_preferences["folder_format"].split(
            user_preferences["folder_format_delimiter"])
        self.set_folder_format_dropdowns(
            folder_format, Constants.default_preferences["folder_format_delimiter"])

        # Reset delimiters
        self.file_delimiter_combo.setCurrentText(
            user_preferences["filename_format_delimiter"])
        self.folder_delimiter_combo.setCurrentText(
            user_preferences["folder_format_delimiter"])

        # Reset global preferences toggle
        self.global_pref_toggle.setChecked(False)
        self.global_pref_label.setText("Use global preferences: OFF")

    def show_error_dialog(self, title, message):
        """
        Show an error dialog with the specified title and message.
        """
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle(title)
        error_dialog.setText(message)
        error_dialog.exec_()

    def preview_format(self):
        """Preview the file and folder format based on selected tags."""
        file_format = [combo.currentText()
                       for combo in self.file_format_combos]
        folder_format = [combo.currentText()
                         for combo in self.folder_format_combos]
        file_delimiter = self.file_delimiter_combo.currentText()
        folder_delimiter = self.folder_delimiter_combo.currentText()
        port_id = 0
        device_id = FileStorage.DEV_get_active(0)
        if device_id == "" or device_id is None:
            device_id = "12345678"
            Log.w(TAG,
                  f"Failed to retrieve active 'Device'. Using ID \"{device_id}\" as an example.")
        if "_" in device_id:  # for multiplex devices, parse port_id from device_id
            port_id, device_id = device_id.split("_", 1)
        port_id = int(str(port_id), 16)
        if port_id == "" or port_id is None:
            Log.w(TAG,
                  f"Failed to retrieve active 'Port ID'. Using ID \"{port_id}\" as an example.")
            port_id = "1"
        if port_id != port_id % 9:  # 4x6 system detected, PID A-D, not 1-4
            # convert int(10) -> int(161)
            # where int(10) refers to Port A, assuming active channel 1 of 6
            port_id = (port_id << 4) + 0x01

        # Generate a preview string based on the selected format
        file_preview = UserProfiles.user_preferences._build_save_path(
            file_format, "Runname", file_delimiter, device_id, port_id)
        folder_preview = UserProfiles.user_preferences._build_save_path(
            folder_format, "Runname", folder_delimiter, device_id, port_id)
        file_preview = file_preview.strip("-_ ")
        folder_preview = folder_preview.strip("-_ ")
        preview_text = f"File Format Preview: {file_preview}\nFolder Format Preview: {folder_preview}"
        self.preview_label.setText(preview_text)

    def save_preferences(self):
        load_data_path = self.load_directory_input.text()
        write_data_path = self.write_directory_input.text()
        self.check_user_role()  # updates self._is_admin

        # Check if load path exists
        if not os.path.exists(load_data_path):
            self.show_error_dialog(
                "Load Path Error", f"The specified load path does not exist:\n{load_data_path}")
            return
        # Check if write path exists
        if not os.path.exists(write_data_path):
            self.show_error_dialog(
                "Write Path Error", f"The specified write path does not exist:\n{write_data_path}")
            return
        # Check if paths are directories
        if not os.path.isdir(load_data_path):
            self.show_error_dialog(
                "Load Path Error", f"The specified load path is not a directory:\n{load_data_path}")
            return
        if not os.path.isdir(write_data_path):
            self.show_error_dialog(
                "Write Path Error", f"The specified write path is not a directory:\n{write_data_path}")
            return

        date_format = self.date_format_combo.currentText()
        time_format = self.time_format_combo.currentText()

        # Get file and folder formats
        file_format = [combo.currentText()
                       for combo in self.file_format_combos]
        folder_format = [combo.currentText()
                         for combo in self.folder_format_combos]
        file_delimiter = self.file_delimiter_combo.currentText()
        folder_delimiter = self.folder_delimiter_combo.currentText()

        # Check "Port" exists in format dropdowns
        if not "Port" in file_format:
            self.show_error_dialog(
                "File Format Error",
                "The \"Port\" tag must exist in the file format to create unique paths for multiplex runs.\n" +
                "For single runs, the tag's value will be blank (unused).")
            return
        # if not "Port" in folder_format:
        #     self.show_error_dialog(
        #         "Folder Format Error",
        #         "The \"Port\" tag must exist in the folder format to create unique paths for multiplex runs.\n" +
        #         "For single runs, the tag's value will be blank (unused).")
        #     return

        # Check tag placeholder text does not exist in format dropdowns
        if Constants.select_tag_prompt in file_format:
            self.show_error_dialog(
                "File Format Error",
                f"The \"{Constants.select_tag_prompt}\" placeholder is not a valid tag.\n" +
                "Please remove it from the file format and try again.")
            return
        if Constants.select_tag_prompt in folder_format:
            self.show_error_dialog(
                "Folder Format Error",
                f"The \"{Constants.select_tag_prompt}\" placeholder is not a valid tag.\n" +
                "Please remove it from the folder format and try again.")
            return

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

        # Save preferences
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
        UserProfiles.user_preferences._set_load_data_path(
            load_data_path=load_data_path)
        UserProfiles.user_preferences._set_write_data_path(
            write_data_path=write_data_path)

        write_globals = False
        if hasattr(UserProfiles.user_preferences, "_user_preferences_path"):
            UserProfiles.user_preferences.write_user_preferences()
            if self._is_admin and self.global_pref_toggle.isChecked():
                # Admin user is modifying global prefs, update both files
                write_globals = True
        elif self._is_admin:
            # No user profiles exist, read/write global pref file only
            write_globals = True
        if write_globals:
            UserProfiles.user_preferences.write_global_preferences()

        # Show a popup window to confirm preferences were saved
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Preferences Saved")
        msg_box.setText("Your preferences have been successfully saved.")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def reset_to_default_preferences(self):
        """Reset preferences to their default values based on a dictionary."""
        # Update folder sync state based on the dictionary
        paths_synced = Qt.CheckState.Checked if (
            Constants.default_preferences["load_data_path"] == Constants.default_preferences["write_data_path"]
        ) else Qt.CheckState.Unchecked
        self.toggle_folder_sync(paths_synced)
        # Reset the load and write paths based on the dictionary
        self.load_directory_input.setText(
            Constants.default_preferences["load_data_path"])
        self.write_directory_input.setText(
            Constants.default_preferences["write_data_path"])
        # Reset the date and time format dropdowns based on the dictionary
        self.date_format_combo.setCurrentText(
            Constants.default_preferences["date_format"])
        self.time_format_combo.setCurrentText(
            Constants.default_preferences["time_format"])

        # Reset file format
        file_format = Constants.default_preferences["filename_format"].split(
            Constants.default_preferences["filename_format_delimiter"])
        self.set_file_format_dropdowns(
            file_format, Constants.default_preferences["filename_format_delimiter"])

        # Reset folder format
        folder_format = Constants.default_preferences["folder_format"].split(
            Constants.default_preferences["folder_format_delimiter"])
        self.set_folder_format_dropdowns(
            folder_format, Constants.default_preferences["folder_format_delimiter"])

        # Reset delimiters
        self.file_delimiter_combo.setCurrentText(
            Constants.default_preferences["filename_format_delimiter"])
        self.folder_delimiter_combo.setCurrentText(
            Constants.default_preferences["folder_format_delimiter"])

    def set_file_format_dropdowns(self, file_format, delimiter):
        """Sets the file format dropdowns based on the provided file format list."""
        # Remove all existing dropdowns
        for combo in self.file_format_combos:
            combo.deleteLater()
        self.file_format_combos.clear()
        for i, format_item in enumerate(file_format):
            if i >= len(self.file_format_combos):
                self.add_dropdown(self.file_format_container)
            self.file_format_combos[i].setCurrentText(format_item)
        # Ensure delimiters are correctly set
        self.file_delimiter_combo.setCurrentText(delimiter)

    def set_folder_format_dropdowns(self, folder_format, delimiter):
        """Sets the folder format dropdowns based on the provided folder format list."""
        for combo in self.folder_format_combos:
            combo.deleteLater()
        self.folder_format_combos.clear()
        for i, format_item in enumerate(folder_format):
            if i >= len(self.folder_format_combos):
                self.add_dropdown(self.folder_format_container)
            self.folder_format_combos[i].setCurrentText(format_item)
        # Ensure delimiters are correctly set
        self.folder_delimiter_combo.setCurrentText(delimiter)
