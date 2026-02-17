from PyQt5 import QtWidgets

try:
    from src.models.ingredient import Buffer
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.ingredient import Buffer


class BufferConfigDialog(QtWidgets.QDialog):
    """Dialog for configuring buffer ingredients."""

    def __init__(self, ing_ctrl, existing_buffer=None, parent=None):
        super().__init__(parent)
        self.controller = ing_ctrl
        self.existing_buffer = existing_buffer
        self.result_ingredient = None
        is_edit = existing_buffer is not None
        self.setWindowTitle("Edit Buffer" if is_edit else "Add New Buffer")
        self.resize(350, 200)
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

        # Form
        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)

        self.edit_name = QtWidgets.QLineEdit()
        self.edit_name.setPlaceholderText("e.g., Phosphate")
        form_layout.addRow("Name*:", self.edit_name)

        self.spin_ph = QtWidgets.QDoubleSpinBox()
        self.spin_ph.setRange(0, 14)
        self.spin_ph.setDecimals(2)
        self.spin_ph.setValue(7.4)
        form_layout.addRow("pH:", self.spin_ph)

        layout.addLayout(form_layout)

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

    def _populate_fields(self, data):
        """Populate fields from either a Buffer object or a dictionary."""
        name = ""
        ph = 7.4

        if isinstance(data, Buffer):
            name = data.name
            ph = data.pH if data.pH is not None else 7.4
        elif isinstance(data, dict):
            name = data.get("name", "")
            ph = data.get("ph", 7.4)

        self.edit_name.setText(name)
        self.spin_ph.setValue(ph)

    def save_and_accept(self):
        name = self.edit_name.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Name is required.")
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
        return {"name": self.edit_name.text().strip(), "ph": self.spin_ph.value()}
