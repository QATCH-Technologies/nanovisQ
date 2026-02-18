from PyQt5 import QtWidgets

try:
    from src.models.ingredient import Buffer
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.ingredient import Buffer

_STYLE_UNSET = (
    "border: 1.5px solid #e53935; "
    "border-radius: 4px; "
    "padding: 6px; "
    "min-height: 24px; "
    "background-color: #fff5f5;"
)
_STYLE_NORMAL = (
    "border: 1px solid #d1d5db; "
    "border-radius: 4px; "
    "padding: 6px; "
    "min-height: 24px; "
    "background-color: #ffffff;"
)


class BufferConfigDialog(QtWidgets.QDialog):
    """Dialog for configuring buffer ingredients."""

    # pH of None / missing is tracked via self._ph_unset rather than a numeric
    # sentinel.  This keeps the full 0–14 range typeable in the spinbox.

    def __init__(self, ing_ctrl, existing_buffer=None, parent=None):
        super().__init__(parent)
        self.controller = ing_ctrl
        self.existing_buffer = existing_buffer
        self.result_ingredient = None
        is_edit = existing_buffer is not None
        self.setWindowTitle("Edit Buffer" if is_edit else "Add New Buffer")
        self.resize(350, 230)
        self.setModal(True)

        self.setStyleSheet(
            """
            QDialog { background-color: #ffffff; }
            QLabel { color: #333; }
            QLineEdit, QDoubleSpinBox { 
                border: 1px solid #d1d5db; 
                border-radius: 4px; 
                padding: 6px; 
                min-height: 24px;
                background-color: #ffffff;
            }
            QLineEdit:focus, QDoubleSpinBox:focus {
                border: 1px solid #00adee;
            }
        """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        lbl_header = QtWidgets.QLabel("Buffer Properties")
        lbl_header.setStyleSheet("font-weight: bold; font-size: 11pt; color: #00adee;")
        layout.addWidget(lbl_header)

        # Incomplete-fields banner
        self.lbl_incomplete = QtWidgets.QLabel(
            "\u26a0  Some fields are required. Please fill in the highlighted fields below."
        )
        self.lbl_incomplete.setStyleSheet(
            "color: #b71c1c; background-color: #ffebee; "
            "border: 1px solid #ef9a9a; border-radius: 4px; padding: 6px;"
        )
        self.lbl_incomplete.setWordWrap(True)
        self.lbl_incomplete.setVisible(False)
        layout.addWidget(self.lbl_incomplete)

        # Form
        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)

        self.edit_name = QtWidgets.QLineEdit()
        self.edit_name.setPlaceholderText("e.g., Phosphate")
        form_layout.addRow("Name*:", self.edit_name)

        self.spin_ph = QtWidgets.QDoubleSpinBox()
        self.spin_ph.setRange(0, 14)
        self.spin_ph.setDecimals(2)
        form_layout.addRow("pH*:", self.spin_ph)

        layout.addLayout(form_layout)

        # Clear the red style as soon as the user moves away from the sentinel
        self._ph_unset = True  # True until the user provides or loads a real pH
        self.spin_ph.valueChanged.connect(lambda v: self._on_ph_changed(v))

        # Load existing data
        if existing_buffer:
            self._populate_fields(existing_buffer)

        # Buttons
        layout.addStretch()
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        btn_save = QtWidgets.QPushButton("Save")
        btn_save.setDefault(True)
        btn_save.setStyleSheet(
            """
            QPushButton {
                background-color: #00adee;
                color: white;
                border: none;
                padding: 6px 20px;
                border-radius: 4px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0096d1;
            }
        """
        )
        btn_save.clicked.connect(self.save_and_accept)
        btn_layout.addWidget(btn_save)

        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def buffer_needs_completion(buffer) -> bool:
        """Return True if *buffer* has an unset pH.

        Can be called by the card widget to decide whether to highlight
        the edit-pen button in red without opening the dialog.
        """
        if not isinstance(buffer, Buffer):
            return False
        return buffer.pH is None

    def has_incomplete_fields(self) -> bool:
        """Return True if pH has not yet been provided."""
        return self._ph_unset

    def _mark_unset(self) -> None:
        self.spin_ph.setStyleSheet(_STYLE_UNSET)
        self.lbl_incomplete.setVisible(True)

    def _mark_set(self) -> None:
        self.spin_ph.setStyleSheet(_STYLE_NORMAL)
        self.lbl_incomplete.setVisible(False)

    def _on_ph_changed(self, value: float) -> None:
        # Any explicit user interaction clears the unset flag
        self._ph_unset = False
        self._mark_set()

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def _populate_fields(self, data):
        """Populate fields from either a Buffer object or a dictionary.

        A None pH is flagged as unset with red styling.
        A real pH value (including 0.0) is filled normally.
        """
        name = ""
        ph = None

        if isinstance(data, Buffer):
            name = data.name or ""
            ph = data.pH  # may be None
        elif isinstance(data, dict):
            name = data.get("name", "")
            ph = data.get("ph")  # intentionally no default

        self.edit_name.setText(name)

        if ph is None:
            self._ph_unset = True
            self.spin_ph.setValue(0.0)
            self._mark_unset()
        else:
            self._ph_unset = False
            self.spin_ph.setValue(float(ph))
            self._mark_set()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_and_accept(self):
        name = self.edit_name.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Name is required.")
            return

        if self.has_incomplete_fields():
            self._mark_unset()
            QtWidgets.QMessageBox.warning(
                self,
                "Incomplete Fields",
                "Please provide a pH value for this buffer.",
            )
            return

        try:
            if self.existing_buffer and isinstance(self.existing_buffer, Buffer):
                # UPDATE
                self.existing_buffer.name = name
                self.existing_buffer.pH = self.spin_ph.value()
                self.controller.update(self.existing_buffer.id, self.existing_buffer)
                self.result_ingredient = self.existing_buffer
            else:
                # ADD NEW
                new_buffer = Buffer(enc_id=-1, name=name, pH=self.spin_ph.value())
                self.result_ingredient = self.controller.add(new_buffer)

            self.accept()

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Database Error", f"Failed to save Buffer:\n{str(e)}"
            )

    def get_data(self):
        """Returns the buffer configuration as a dictionary."""
        return {
            "name": self.edit_name.text().strip(),
            "ph": self.spin_ph.value() if not self.has_incomplete_fields() else None,
        }
