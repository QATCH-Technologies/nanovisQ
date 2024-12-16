from QATCH.common.logger import Logger as Log
from QATCH.common.fileStorage import FileStorage
from QATCH.common.userProfiles import UserProfiles
from QATCH.core.constants import Constants
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QVBoxLayout, QComboBox,
                             QLabel, QLineEdit, QPushButton, QMessageBox, QFormLayout, QHBoxLayout
                             )
from PyQt5.QtCore import QDateTime, Qt
from PyQt5.QtGui import QIcon, QPixmap
from datetime import datetime
import json
import re
import serial.tools.list_ports
import os
TAG = '[Configure Data]'


class UIConfigureData(QtWidgets.QWidget):
    """
    The UI class for managing and saving user preferences for folder and filename formats,
    date/time formats, and related settings. This class provides methods for dynamically
    adjusting and saving preferences, including handling changes to the folder format,
    filename format, delimiters, and date/time formats.

    Attributes:
        folder_format_input (QLineEdit): The input field for the folder format.
        filename_format_input (QLineEdit): The input field for the filename format.
        date_format_dropdown (QComboBox): The dropdown for selecting the date format.
        time_format_dropdown (QComboBox): The dropdown for selecting the time format.
        folder_delimiter_dropdown (QComboBox): The dropdown for selecting the folder delimiter.
        filename_delimiter_dropdown (QComboBox): The dropdown for selecting the filename delimiter.
        keywords_for_input (dict): A dictionary storing keywords for input fields, including delimiters and keywords for formats.
        preview_output (QLabel): The label that displays the folder/filename preview.
        updating_text (bool): Flag to prevent recursive updates when modifying the text fields.
        last_focused_input (QLineEdit): The last focused input field to assist with user input handling.

    Methods:
        set_last_focused_input(input_field):
            Updates the `last_focused_input` attribute whenever a text field gains focus.

        get_folder_format():
            Retrieves the folder format keywords from the input fields.

        get_filename_format():
            Retrieves the filename format keywords from the input fields.

        generate_preview():
            Generates and displays a preview of the folder and file structure based on example data and user inputs.

        handle_text_change(input_name):
            Handles changes to the input fields, updates the `keywords_for_input` dictionary, and splits text based on delimiters.

        validate_format(folder_format, filename_format):
            Validates the folder and filename formats using regular expressions to ensure they conform to a specific pattern.

        save_preferences(default=False):
            Saves the user-defined preferences to a JSON file or loads default preferences if `default` is `True`.
    """

    def __init__(self):
        """
        Initializes the Configure Data window for setting folder/filename formats, date/time formats, and previewing the output.

        Sets up the GUI components, including:
        - Folder and filename format input fields with delimiter selection.
        - Date and time format selection dropdowns.
        - Preview area to show how the selected formats would appear.
        - Buttons to insert predefined keywords and backspace to remove the last keyword.
        - Button to save user preferences, as well as a button to reset to default preferences.

        Attributes:
            layout (QVBoxLayout): The main layout of the window containing all components.
            last_focused_input (QLineEdit): Tracks the last focused input field for convenience.
            keywords (dict): A dictionary mapping valid tags to their descriptions, used for inserting predefined keywords.
            keywords_for_input (dict): Stores keyword data related to folder and filename input fields.
            folder_format_label (QLabel): Label for the folder format input field.
            folder_format_input (QLineEdit): Input field for specifying the folder format.
            filename_format_label (QLabel): Label for the filename format input field.
            filename_format_input (QLineEdit): Input field for specifying the filename format.
            folder_delimiter_dropdown (QComboBox): Dropdown menu for selecting folder format delimiter.
            filename_delimiter_dropdown (QComboBox): Dropdown menu for selecting filename format delimiter.
            folder_backspace_button (QPushButton): Button to remove the last inserted keyword from the folder format.
            filename_backspace_button (QPushButton): Button to remove the last inserted keyword from the filename format.
            date_format_label (QLabel): Label for the date format dropdown.
            date_format_dropdown (QComboBox): Dropdown menu for selecting the date format.
            time_format_label (QLabel): Label for the time format dropdown.
            time_format_dropdown (QComboBox): Dropdown menu for selecting the time format.
            preview_label (QLabel): Label for the output preview section.
            preview_output (QLabel): Displays the output preview based on the current selections.
            preview_button (QPushButton): Button to generate the preview of the output.
            keyword_buttons_label (QLabel): Label for the section where keyword insertion buttons are displayed.
            keyword_buttons_layout (QHBoxLayout): Layout containing buttons for each predefined keyword.
            save_button (QPushButton): Button to save the user preferences.
            default_preferences_button (QPushButton): Button to restore the default preferences.

        Signals:
            focusInEvent (QLineEdit): Triggered when a user focuses on the folder or filename input field.
            textChanged (QLineEdit): Triggered when text is changed in the folder or filename input fields.
            currentIndexChanged (QComboBox): Triggered when the date or time format dropdown selection is changed.
            clicked (QPushButton): Triggered when the preview or save buttons are clicked.
        """
        super().__init__()
        self.setWindowTitle("Configure Data")
        self.layout = QVBoxLayout()
        self.last_focused_input = None  # Track the last focused input field
        # Keywords and their descriptions
        self.keywords = {
            Constants.valid_tags[0]: "User's name",
            Constants.valid_tags[1]: "User's initials",
            Constants.valid_tags[2]: "Device ID",
            Constants.valid_tags[3]: "Run name",
            Constants.valid_tags[4]: "Date",
            Constants.valid_tags[5]: "Time",
            Constants.valid_tags[6]: "COM Port"
        }

        self.keywords_for_input = {
            "folder_format_input": {"delimiter": list(Constants.path_delimiters.values())[0], "keywords": []},
            "filename_format_input": {"delimiter": list(Constants.path_delimiters.values())[0], "keywords": []}
        }

        user_info = UserProfiles.get_session_file()
        user_preferences_path = os.path.join(Constants.local_app_data_path, "profiles/users",
                                             f"{user_info}-file-format-preferences.json")
        global_preferences_path = os.path.join(
            Constants.local_app_data_path, "file-format-preferences.json")
        self.user_preferences = None
        self.global_preferences = None
        if os.path.exists(global_preferences_path):
            with open(global_preferences_path, 'r') as preferences_file:
                self.global_preferences = json.load(preferences_file)
        else:
            Log.e(
                tag=TAG, msg="No global file, writing default preferences and using.")
            FileStorage.DEV_write_default_preferences(global_preferences_path)
            with open(global_preferences_path, 'r') as preferences_file:
                self.global_preferences = json.load(preferences_file)

        if os.path.exists(user_preferences_path):
            Log.d(TAG, 'Using User Preferences')
            with open(user_preferences_path, 'r') as preferences_file:
                self.user_preferences = json.load(preferences_file)
        else:
            Log.e(TAG, "No user preferences file, writing default preferences and using.")
            FileStorage.DEV_write_default_preferences(user_preferences_path)
            with open(user_preferences_path, 'r') as preferences_file:
                self.user_preferences = json.load(preferences_file)

        self.folder_format_label = QLabel("Folder Format")
        self.folder_format_input = QLineEdit()
        self.folder_format_input.focusInEvent = self.set_last_focused_input(
            self.folder_format_input)
        self.folder_format_input.setText(
            self.user_preferences.get("folder_format"))
        self.folder_format_input.setReadOnly(True)

        self.filename_format_label = QLabel("Filename Format")
        self.filename_format_input = QLineEdit()
        self.filename_format_input.focusInEvent = self.set_last_focused_input(
            self.filename_format_input)
        self.filename_format_input.setText(
            self.user_preferences.get("filename_format"))
        self.filename_format_input.setReadOnly(True)

        # Delimiter selection for Folder Format
        self.folder_delimiter_dropdown = QComboBox()
        self.folder_delimiter_dropdown.addItems(
            list(Constants.path_delimiters.values()))
        self.folder_delimiter_dropdown.setCurrentText(
            self.user_preferences.get("folder_format_delimiter"))
        self.folder_delimiter_dropdown.currentIndexChanged.connect(
            lambda: self.set_delimiter(
                "folder_format_input", self.folder_delimiter_dropdown.currentText())
        )

        # Delimiter selection for Filename Format
        self.filename_delimiter_dropdown = QComboBox()
        self.filename_delimiter_dropdown.addItems(
            list(Constants.path_delimiters.values()))
        self.filename_delimiter_dropdown.setCurrentText(
            self.user_preferences.get("file_format_delimiter"))
        self.filename_delimiter_dropdown.currentIndexChanged.connect(
            lambda: self.set_delimiter(
                "filename_format_input", self.filename_delimiter_dropdown.currentText())
        )

        # Backspace buttons for Folder Format and Filename Format
        pixmap = QPixmap(r'QATCH\icons\backspace.png')
        pixmap = pixmap.scaled(10, 10)
        self.folder_backspace_button = QPushButton()
        self.folder_backspace_button.clicked.connect(lambda:
                                                     self.remove_last_keyword_folder('folder_format_input'))
        self.folder_backspace_button.setIcon(QIcon(pixmap))
        self.folder_backspace_button.setIconSize(
            pixmap.size())

        self.filename_backspace_button = QPushButton()
        self.filename_backspace_button.clicked.connect(lambda:
                                                       self.remove_last_keyword_file('filename_format_input'))

        self.filename_backspace_button.setIcon(QIcon(pixmap))
        self.filename_backspace_button.setIconSize(
            pixmap.size())

        # Layout for Folder Format and Filename Format inputs with delimiter dropdown
        folder_format_layout = QHBoxLayout()
        folder_format_layout.addWidget(self.folder_format_input)
        folder_format_layout.addWidget(self.folder_delimiter_dropdown)
        folder_format_layout.addWidget(self.folder_backspace_button)

        filename_format_layout = QHBoxLayout()
        filename_format_layout.addWidget(self.filename_format_input)
        filename_format_layout.addWidget(self.filename_delimiter_dropdown)
        filename_format_layout.addWidget(self.filename_backspace_button)

        self.updating_text = True
        self.folder_format_input.textChanged.connect(
            lambda: self.handle_text_change("folder_format_input"))
        self.updating_text = False
        self.updating_text = True
        self.filename_format_input.textChanged.connect(
            lambda: self.handle_text_change("filename_format_input"))
        self.updating_text = False

        # Date and Time format components
        self.date_format_label = QLabel("Date Format")
        self.date_format_dropdown = QComboBox()
        self.date_format_dropdown.addItems(list(Constants.date_formats.keys()))
        self.date_format_dropdown.setCurrentText(
            self.user_preferences.get("date_format"))
        self.date_format_dropdown.currentIndexChanged.connect(
            self.generate_preview)

        self.time_format_label = QLabel("Time Format")
        self.time_format_dropdown = QComboBox()
        self.time_format_dropdown.addItems(list(
            Constants.time_formats.keys()))
        self.time_format_dropdown.setCurrentText(
            self.user_preferences.get("time_format"))

        # Output preview section
        self.preview_label = QLabel("Output Preview")
        self.preview_output = QLabel("")
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.generate_preview)

        # Buttons for inserting keywords into format fields
        self.keyword_buttons_label = QLabel("Insert Keywords")
        self.keyword_buttons_layout = QHBoxLayout()

        for key, desc in self.keywords.items():
            button = QPushButton(key)
            button.setToolTip(desc)
            button.clicked.connect(lambda _, k=key: self.insert_keyword(k))
            self.keyword_buttons_layout.addWidget(button)

        # Form layout
        form_layout = QFormLayout()
        form_layout.addRow(self.folder_format_label, folder_format_layout)
        form_layout.addRow(self.filename_format_label, filename_format_layout)
        form_layout.addRow(self.date_format_label, self.date_format_dropdown)
        form_layout.addRow(self.time_format_label, self.time_format_dropdown)

        # Save button
        self.save_button = QPushButton("Save Preferences")
        self.save_button.clicked.connect(self.save_preferences)
        # Default preferences button
        self.default_preferences_button = QPushButton("Default Preferences")
        self.default_preferences_button.clicked.connect(
            lambda: self.save_preferences(default=True))

        # Preview alignment
        self.preview_output.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        self.preview_output.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.preview_output.setWordWrap(True)

        # Add components to main layout
        self.layout.addLayout(form_layout)
        self.layout.addWidget(self.keyword_buttons_label)
        self.layout.addLayout(self.keyword_buttons_layout)
        self.layout.addWidget(self.preview_label)
        self.layout.addWidget(self.preview_output)
        self.layout.addWidget(self.preview_button)
        self.layout.addWidget(self.save_button)
        self.layout.addWidget(self.default_preferences_button)
        self.setLayout(self.layout)

    def remove_last_keyword_folder(self, input_name: str = 'folder_format_input'):
        """
        Removes the last inserted keyword and/or delimiter from the folder format input field.

        This method removes the last keyword and its associated delimiter from the input field
        based on the provided `input_name`. If there are multiple keywords and delimiters, both
        are removed; otherwise, only the keyword is removed.

        Args:
            input_name (str): The name of the input field (either "folder_format_input" or
                            "filename_format_input") from which the last keyword and delimiter
                            should be removed.

        Notes:
            - The method checks if there are any keywords to remove before proceeding.
            - If the list of keywords has more than one item, both the keyword and the delimiter
            are removed. If only one keyword remains, only the keyword is removed.
            - The `folder_format_input` field is updated with the removed keyword(s) and delimiter(s).

        Example:
            If the input field contains a string like "%username%_%device%" and the last inserted
            keyword is "%device%" with the delimiter "_", calling this method would remove
            "%device%" and the "_" from the input field.

        """
        try:
            if self.keywords_for_input[input_name]["keywords"]:
                if len(self.keywords_for_input[input_name]["keywords"]) > 1:
                    removed_keyword = self.keywords_for_input[input_name]["keywords"].pop(
                    )
                    removed_delimiter = self.keywords_for_input[input_name]["keywords"].pop(
                    )
                    self.folder_format_input.setText(
                        self.folder_format_input.text().replace(removed_keyword, "", 1))
                    self.folder_format_input.setText(
                        self.folder_format_input.text().replace(removed_delimiter, "", 1))
                else:
                    removed_keyword = self.keywords_for_input[input_name]["keywords"].pop(
                    )
                    self.folder_format_input.setText(
                        self.folder_format_input.text().replace(removed_keyword, "", 1))
        except Exception as e:
            Log.e(
                tag=TAG, msg=f'Error while removing last folder format tag; message={e}')

    def remove_last_keyword_file(self, input_name: str = 'filename_format_input'):
        """
        Removes the last inserted keyword and/or delimiter from the filename format input field.

        This method removes the last keyword and its associated delimiter from the input field
        based on the provided `input_name`. If there are multiple keywords and delimiters, both
        are removed; otherwise, only the keyword is removed.

        Args:
            input_name (str): The name of the input field (either "folder_format_input" or
                            "filename_format_input") from which the last keyword and delimiter
                            should be removed.

        Notes:
            - The method checks if there are any keywords to remove before proceeding.
            - If the list of keywords has more than one item, both the keyword and the delimiter
            are removed. If only one keyword remains, only the keyword is removed.
            - The `filename_format_input` field is updated with the removed keyword(s) and delimiter(s).

        Example:
            If the input field contains a string like "%username%-%device%" and the last inserted
            keyword is "%device%" with the delimiter "-", calling this method would remove
            "%device%" and the "-" from the input field.

        """
        try:
            if self.keywords_for_input[input_name]["keywords"]:
                if len(self.keywords_for_input[input_name]["keywords"]) > 1:
                    removed_keyword = self.keywords_for_input[input_name]["keywords"].pop(
                    )
                    removed_delimiter = self.keywords_for_input[input_name]["keywords"].pop(
                    )
                    self.filename_format_input.setText(
                        self.filename_format_input.text().replace(removed_keyword, "", 1))
                    self.filename_format_input.setText(
                        self.filename_format_input.text().replace(removed_delimiter, "", 1))
                else:
                    removed_keyword = self.keywords_for_input[input_name]["keywords"].pop(
                    )
                    self.filename_format_input.setText(
                        self.filename_format_input.text().replace(removed_keyword, "", 1))
        except Exception as e:
            Log.e(
                tag=TAG, msg=f'Error while removing last file format tag; message={e}')

    def get_delimiter_key(self, input_name: str):
        """
        Retrieves the delimiter key from the PATH_DELIMITERS dictionary based on the current delimiter value.

        This method checks the current delimiter value associated with the given `input_name`
        and returns the corresponding key from the `PATH_DELIMITERS` dictionary. If no matching
        delimiter is found, it returns the first key from the dictionary as the default.

        Args:
            input_name (str): The name of the input field (either "folder_format_input" or
                            "filename_format_input") whose delimiter key needs to be retrieved.

        Returns:
            str: The key corresponding to the current delimiter value in `PATH_DELIMITERS`.

        Example:
            If the current delimiter is "_", this method will return "Underscore".

        Notes:
            - This method assumes that the current delimiter value exists in the `PATH_DELIMITERS` dictionary.
            - If no match is found, the method returns the first key in the dictionary as a fallback.

        """
        current_delimiter = self.keywords_for_input[input_name]["delimiter"]
        for key, value in Constants.path_delimiters.items():
            if value == current_delimiter:
                return key
        return list(Constants.path_delimiters.keys())[0]

    def set_delimiter(self, input_name: str, delimiter: str):
        """
        Updates the delimiter for the specified input field and adjusts the list of keywords accordingly.

        This method modifies the delimiter for a specified input (either folder or filename format)
        and updates the list of keywords used in that input. It replaces any existing delimiter
        in the keyword list with the new delimiter and updates the corresponding input field's text.

        Args:
            input_name (str): The name of the input field ("folder_format_input" or "filename_format_input")
                            whose delimiter needs to be updated.
            delimiter (str): The new delimiter to be set for the specified input.

        Returns:
            None

        Example:
            set_delimiter("folder_format_input", "_")

        Notes:
            - This method directly modifies the keyword list associated with the input field.
            - The input field's text is updated to reflect the new delimiter in the format string.
        """
        # Update the delimiter for the specific input
        self.keywords_for_input[input_name]["delimiter"] = delimiter

        # Replace all existing delimiters in the keywords list with the new delimiter
        for i, keyword in enumerate(self.keywords_for_input[input_name]["keywords"]):
            if keyword in Constants.path_delimiters.values():
                self.keywords_for_input[input_name]["keywords"][i] = delimiter
        keywords = self.keywords_for_input[input_name]["keywords"]
        updated_text = "".join(keywords)

        # Update the respective input field
        if input_name == "folder_format_input":
            self.folder_format_input.clear()
            self.folder_format_input.setText(updated_text)
        elif input_name == "filename_format_input":
            self.filename_format_input.clear()
            self.filename_format_input.setText(updated_text)

    def set_last_focused_input(self, input_field: str):
        """
        Helper function to update the last focused input field.

        This method returns a function that, when called, updates the `last_focused_input`
        attribute with the provided input field. The returned function preserves the default
        behavior of the focus event for the input field.

        Args:
            input_field (QLineEdit): The input field (either folder or filename format input)
                                    to track as the last focused input.

        Returns:
            function: A function that handles the focus event and updates the `last_focused_input`.

        Example:
            set_last_focused_input(self.folder_format_input)

        Notes:
            - The returned function is intended to be used as an event handler for the `focusInEvent`
            of a `QLineEdit`.
        """
        def focus_event(event):
            self.last_focused_input = input_field
            # Preserve default behavior
            QLineEdit.focusInEvent(input_field, event)
        return focus_event

    def insert_keyword(self, keyword: str):
        """
        Inserts the selected keyword into the last focused input field.

        This method inserts a specified keyword into either the folder or filename format input field.
        If a delimiter exists, it is inserted before the keyword. The method ensures that the keyword
        is placed in the correct input field and provides an error message if no field is selected.

        Args:
            keyword (str): The keyword to be inserted into the last focused input field.

        Returns:
            None

        Raises:
            QMessageBox warning: If no input field is focused when the method is called,
                                a warning message is displayed.

        Example:
            insert_keyword("UserName")

        Notes:
            - The method assumes that either `folder_format_input` or `filename_format_input` is focused.
            - The delimiter is inserted before the keyword if the list of keywords is not empty.
            - If no field is focused, a warning message is shown to the user.
        """
        # TODO: make sure PORT tag is present before saving and runname
        if self.last_focused_input:
            input_name = "folder_format_input" if self.last_focused_input == self.folder_format_input else "filename_format_input"

            if len(self.keywords_for_input[input_name]["keywords"]) > 0:
                # Insert the delimiter only if it's not the last one
                self.last_focused_input.insert(
                    self.keywords_for_input[input_name]["delimiter"])
                self.keywords_for_input[input_name]["keywords"].append(
                    self.keywords_for_input[input_name]["delimiter"])

            self.last_focused_input.insert(keyword)
        else:
            Log.e(tag=TAG, msg='No input box selected to format.')
            QMessageBox.warning(
                self, "Error", "Please click on a format field before inserting a keyword.")

    def get_folder_format(self):
        """
        Retrieve the keywords associated with the folder format input.

        This method returns the list of keywords that have been added to the folder format input field.
        It retrieves the value from the `keywords_for_input` dictionary.

        Returns:
            list: A list of keywords (strings) associated with the folder format input.

        Example:
            folder_keywords = get_folder_format(self)

        Notes:
            - The returned list contains the keywords set for the folder format input field, which are
            part of the `keywords_for_input` attribute.
        """
        return self.keywords_for_input["folder_format_input"]['keywords']

    def get_filename_format(self):
        """
        Retrieve the keywords associated with the filename format input.

        This method returns the list of keywords that have been added to the filename format input field.
        It retrieves the value from the `keywords_for_input` dictionary.

        Returns:
            list: A list of keywords (strings) associated with the filename format input.

        Example:
            filename_keywords = get_filename_format(self)

        Notes:
            - The returned list contains the keywords set for the filename format input field, which are
            part of the `keywords_for_input` attribute.
        """
        return self.keywords_for_input["filename_format_input"]["keywords"]

    def generate_preview(self):
        """
        Generate a preview of the folder and file structure based on example data.

        This method constructs a preview of the folder and file names by replacing valid tags with example data.
        The preview is based on the current session information, device details, and date/time formats.

        The following tags are replaced in the format:
        - `%username%`: The current username.
        - `%initials%`: The current user's initials.
        - `%device%`: The active device ID or placeholder if unavailable.
        - `%runname%`: Placeholder for the run name.
        - `%date%`: The current date in the selected date format.
        - `%time%`: The current time in the selected time format.

        The method updates the folder and filename format inputs by replacing keywords with appropriate values
        from the example data. It then displays the formatted preview in the output field.

        Example:
            generate_preview(self)

        Notes:
            - The preview is generated based on the currently active session and device.
            - The `folder_format_input` and `filename_format_input` text fields should contain the tags
            to be replaced.
        """
        # Example data for preview
        device_preview = FileStorage.DEV_get_active(0)
        if device_preview == "":
            device_preview = "[DEVICEID]"

        valid, infos = UserProfiles.session_info()
        if valid:
            username = infos[0]
            initials = infos[1]
        else:
            username = '[USERNAME]'
            initials = '[INITIALS]'
        example_data = {
            Constants.valid_tags[0]: username,
            Constants.valid_tags[1]: initials,
            Constants.valid_tags[2]: device_preview,
            Constants.valid_tags[3]: "[RUNNAME]",
            Constants.valid_tags[4]: datetime.now().strftime(Constants.date_formats.get(self.date_format_dropdown.currentText())),
            Constants.valid_tags[5]: QDateTime.currentDateTime().toString(
                self.time_format_dropdown.currentText()),
            Constants.valid_tags[6]: "[COM PORT]"
        }

        # Folder and file formats
        folder_format = self.folder_format_input.text()
        filename_format = self.filename_format_input.text()
        # Replace keywords in preview

        def update_format(format_keywords):
            update_string = ""
            for keyword in format_keywords:
                if keyword not in Constants.path_delimiters.values():
                    update_string += example_data.get(keyword)
                else:
                    update_string += keyword
            return update_string

        folder_format = update_format(self.get_folder_format())
        filename_format = update_format(self.get_filename_format())

        date_format = example_data.get("%date%")
        time_format = example_data.get("%time%")
        # Set preview output
        self.preview_output.setText(
            f"Folder: {folder_format}\nFile: {filename_format}\nDate: {date_format}\nTime: {time_format}")

    def handle_text_change(self, input_name: str):
        """
        Handle manual changes to the input fields and update `keywords_for_input` accordingly.

        This method is called when there is a change in the folder or filename format input fields.
        It processes the current text in the respective input field, splits the text by the current delimiter,
        and updates the `keywords_for_input` dictionary to reflect the changes made by the user.
        It prevents recursive updates by checking the `updating_text` flag.

        Args:
            input_name (str): The name of the input field that triggered the change.
                            Should be either "folder_format_input" or "filename_format_input".

        Returns:
            None

        Raises:
            ValueError: If the `input_name` is not valid (i.e., not one of "folder_format_input" or "filename_format_input").

        Notes:
            - The method uses the current delimiter to split the text input into components.
            - The updated keywords list is stored in the `keywords_for_input` dictionary under the corresponding `input_name`.
            - Avoids recursive updates by checking the `updating_text` flag.
        """
        if self.updating_text:
            return  # Prevent recursive updates

        # Get the current text from the input field
        if input_name == "folder_format_input":
            text = self.folder_format_input.text()
        elif input_name == "filename_format_input":
            text = self.filename_format_input.text()
        else:
            return

        # Get the current delimiter for this input
        delimiter = self.keywords_for_input[input_name]["delimiter"]

        # Split the text based on the delimiter
        parts = text.split(delimiter)

        # Rebuild the keywords list with delimiters
        new_keywords = []
        for i, part in enumerate(parts):
            part = part.strip()
            if part:  # Avoid adding empty strings
                new_keywords.append(part)
                if i < len(parts) - 1:
                    new_keywords.append(delimiter)
        # Update the keywords_for_input list
        self.keywords_for_input[input_name]["keywords"] = new_keywords

    def validate_format(self, folder_format: str, filename_format: str):
        """
        Validate the folder and filename format strings to ensure they follow a correct pattern of tags and delimiters.

        This method checks if the provided `folder_format` and `filename_format` strings match a predefined
        regex pattern. The pattern ensures that the formats contain valid tags (such as `%username%`, `%date%`,
        etc.) and proper delimiters (such as dashes, underscores, or spaces). If the formats do not conform
        to the expected pattern, a `ValueError` is raised.

        Args:
            folder_format (str): The format string for the folder path. Should consist of tags and delimiters.
            filename_format (str): The format string for the filename. Should consist of tags and delimiters.

        Returns:
            None

        Raises:
            ValueError: If either `folder_format` or `filename_format` do not match the expected format.

        Notes:
            - The regex pattern used checks for the following valid tags: `%username%`, `%initials%`, `%device%`,
            `%runname%`, `%date%`, and `%time%`.
            - The delimiters allowed in the formats are `-`, `_`, or spaces.
            - The format must not have any leading or trailing delimiters.
        """
        # Define a regex pattern for the format
        # Pattern: tag, delimiter, tag, ..., with no trailing delimiter or leading/trailing spaces
        tag_pattern = r"(%username%|%initials%|%device%|%runname%|%date%|%time%|%port%)"
        delimiter_pattern = r"[-_\s]"
        valid_format_regex = re.compile(
            fr"^{tag_pattern}({delimiter_pattern}{tag_pattern})*$"
        )

        # Helper function to validate a single format list
        def is_valid_format(format_str):
            if not valid_format_regex.match(format_str):
                return False
            return True

        # Validate folder_format and filename_format
        if not is_valid_format(folder_format):
            Log.e(tag=TAG, msg='Invalid folder format tag pattern detected.')
            raise ValueError(
                "Invalid folder_format: Ensure it follows the correct tag-delimiter pattern.")
        if not is_valid_format(filename_format):
            Log.e(tag=TAG, msg='Invalid filename format tag pattern detected.')
            raise ValueError(
                "Invalid filename_format: Ensure it follows the correct tag-delimiter pattern.")

    def get_serial_devices(self):
        ports = serial.tools.list_ports.comports()
        port_devices = {}
        if ports:
            for port in ports:
                print(f"- {port.device}: {port.description}")
                port_devices[port.device] = port.description
        else:
            print("No serial ports found.")

        return port_devices

    def save_preferences(self, default: bool = False):
        """
        Save the user-defined settings to a file, either loading default settings or saving the current settings.

        This method handles saving the user's folder format, filename format, delimiters, and date/time format
        to a preferences file (`file-preferences.json`). It also allows for resetting to default settings if
        specified. If the `default` flag is set to `True`, the method will load default preferences; otherwise,
        it will save the current user preferences.

        Args:
            default (bool): Flag indicating whether to load default settings (`True`) or save current settings (`False`).
                            Default is `False`, which saves the current settings.

        Returns:
            None

        Raises:
            ValueError: If folder format or filename format is empty.
            Exception: If there is an error during the saving process.

        Notes:
            - This method performs simple validation to ensure that folder format and filename format are not empty.
            - Preferences are saved to the `file-preferences.json` file.
            - Default preferences are defined in the `DEFAULT_PREFERNCES` dictionary.
        """
        """Save the user-defined settings to a file."""

        # TODO: On-saving, update the UserPreferences in UserProfiles to reflect changes to preferences.
        try:
            if default:
                self.folder_format_input.setText(
                    Constants.default_preferences["folder_format"])
                folder_format = Constants.default_preferences["folder_format"]
                self.filename_format_input.setText(
                    Constants.default_preferences["filename_format"])
                filename_format = Constants.default_preferences["filename_format"]
                date_format = self.date_format_dropdown.setCurrentText(
                    Constants.default_preferences["date_format"])
                time_format = self.date_format_dropdown.setCurrentText(
                    Constants.default_preferences["time_format"])
                folder_delimiter = Constants.default_preferences["folder_format_delimiter"]
                filename_delimiter = Constants.default_preferences["filename_format_delimiter"]
                self.folder_delimiter_dropdown.setCurrentText(
                    folder_delimiter)
                self.filename_delimiter_dropdown.setCurrentText(
                    filename_delimiter)
                self.time_format_dropdown.setCurrentText(date_format)
                self.time_format_dropdown.setCurrentText(time_format)
            else:

                folder_format = "".join(
                    self.keywords_for_input['folder_format_input']['keywords'])
                if folder_format is None or folder_format == "":
                    folder_format = self.folder_format_input.text()
                print(f"Folder format on save: {folder_format}")
                filename_format = "".join(
                    self.keywords_for_input['filename_format_input']['keywords'])
                if filename_format is None or filename_format == "":
                    filename_format = self.filename_format_input.text()
                print(f"File format on save: {filename_format}")

                # Requires  filename format to contain a %port% tag
                if Constants.valid_tags[6] not in filename_format:
                    Log.w(
                        tag=TAG, msg=f"Filename must contain a '{Constants.valid_tags[6]}' tag.")
                    QMessageBox.warning(
                        self, "Error", f"Filename must contain a '{Constants.valid_tags[6]}' tag.")
                    return
                # Requires  filename format to contain a %runname% tag
                if Constants.valid_tags[3] not in filename_format:
                    Log.w(
                        tag=TAG, msg=f"Filename must contain a '{Constants.valid_tags[3]}' tag.")
                    QMessageBox.warning(
                        self, "Error", f"Filename must contain a '{Constants.valid_tags[3]}' tag.")
                    return
                folder_delimiter = self.keywords_for_input['folder_format_input']['delimiter']
                filename_delimiter = self.keywords_for_input['filename_format_input']['delimiter']
            date_format = self.date_format_dropdown.currentText()
            time_format = self.time_format_dropdown.currentText()

            if not folder_format or not filename_format:
                Log.w(tag=TAG, msg="Folder and Filename formats cannot be empty!")
                QMessageBox.warning(
                    self, "Error", "Folder and Filename formats cannot be empty!")
                return
            self.validate_format(folder_format, filename_format)

            preferences = {
                "folder_format": folder_format,
                "filename_format": filename_format,
                "folder_format_delimiter": folder_delimiter,
                "filename_format_delimiter": filename_delimiter,
                "date_format": date_format,
                "time_format": time_format
            }

            # Save settings to preferences.json
            user_info = UserProfiles.get_session_file()
            user_info = user_info.split(".xml")[0]
            if user_info:
                save_path = os.path.join(Constants.local_app_data_path, "profiles/users",
                                         f"{user_info}-file-format-preferences.json")
            else:
                save_path = os.path.join(
                    Constants.local_app_data_path, "file-format-preferences.json")

            with open(save_path, "w") as f:
                json.dump(preferences, f, indent=4)
            if default:
                Log.i(tag=TAG, msg="Restored to default prefences.")
                QMessageBox.information(
                    self, "Saved", "Restored to default prefences.")
            else:
                Log.i(tag=TAG, msg="Successfully saved prefences.")
                QMessageBox.information(
                    self, "Saved", "Successfully saved prefences.")

            # Update user Preferences attribute to new preferences.
            UserProfiles.user_preferences.set_preferences()
        except Exception as e:
            Log.e(tag=TAG, msg=f"Failed to save settings: {e}")
            QMessageBox.warning(self, "Error", f"Failed to save settings: {e}")
