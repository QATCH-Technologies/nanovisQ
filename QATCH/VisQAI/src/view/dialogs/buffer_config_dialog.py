from PyQt5 import QtWidgets


class BufferConfigDialog(QtWidgets.QDialog):
    """Dialog for configuring buffer ingredients."""

    def __init__(self, existing_buffer=None, parent=None):
        super().__init__(parent)
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

        # Load existing data if provided
        if existing_buffer:
            self.edit_name.setText(existing_buffer.get("name", ""))
            self.spin_ph.setValue(existing_buffer.get("ph", 7.4))

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
        btn_save.clicked.connect(self.accept)
        btn_layout.addWidget(btn_save)

        layout.addLayout(btn_layout)

    def get_data(self):
        """Returns the buffer configuration as a dictionary."""
        return {"name": self.edit_name.text().strip(), "ph": self.spin_ph.value()}
