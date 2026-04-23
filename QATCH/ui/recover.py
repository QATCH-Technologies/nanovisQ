from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QListWidget,
    QWidget,
    QLabel,
    QPushButton,
    QFormLayout,
    QFrame,
    QProgressBar,
    QTextEdit,
    QInputDialog,
    QMessageBox,
    QApplication,
    QListWidgetItem,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from datetime import datetime


class UnnamedRun:
    """Data class to hold unnamed run information"""

    def __init__(self, timestamp, duration_seconds, num_points, ruling, file_size_mb):
        self.timestamp = timestamp
        self.duration_seconds = duration_seconds
        self.num_points = num_points
        self.ruling = ruling  # "Good" or "Bad"
        self.file_size_mb = file_size_mb


class RecoveryWorker(QThread):
    """Worker thread to handle recovery process without blocking UI"""

    progress_updated = pyqtSignal(int)  # Progress percentage
    status_updated = pyqtSignal(str)  # Status message
    recovery_complete = pyqtSignal(bool, str)  # Success flag, message

    def __init__(self, run_data, run_name):
        super().__init__()
        self.run_data = run_data
        self.run_name = run_name
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        """Perform the actual recovery work"""
        try:
            # Step 1: Parse raw data
            self.status_updated.emit("Parsing raw data...")
            self.progress_updated.emit(10)
            if self._is_cancelled:
                return
            # Your actual parsing logic here
            self.msleep(500)  # Simulated work

            # Step 2: Generate viscosity profile
            self.status_updated.emit("Generating viscosity profile...")
            self.progress_updated.emit(35)
            if self._is_cancelled:
                return
            # Your profile generation logic here
            self.msleep(800)

            # Step 3: Compute statistics
            self.status_updated.emit("Computing statistics...")
            self.progress_updated.emit(60)
            if self._is_cancelled:
                return
            # Your statistics computation here
            self.msleep(600)

            # Step 4: Generate missing metadata
            self.status_updated.emit("Generating missing metadata...")
            self.progress_updated.emit(80)
            if self._is_cancelled:
                return
            # Your metadata generation here
            self.msleep(400)

            # Step 5: Write file
            self.status_updated.emit("Writing recovered file...")
            self.progress_updated.emit(95)
            if self._is_cancelled:
                return
            # Your file writing logic here
            self.msleep(300)

            # Complete
            self.progress_updated.emit(100)
            self.status_updated.emit("Recovery complete!")
            self.recovery_complete.emit(
                True, f"Successfully recovered run as '{self.run_name}'"
            )

        except Exception as e:
            self.recovery_complete.emit(False, f"Recovery failed: {str(e)}")


class RecoveryProgressDialog(QDialog):
    """Modal dialog showing recovery progress"""

    def __init__(self, parent, run_data, run_name):
        super().__init__(parent)
        self.run_data = run_data
        self.run_name = run_name
        self.worker = None
        self.setup_ui()
        self.start_recovery()

    def setup_ui(self):
        self.setWindowTitle("Recovering Run")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)

        layout = QVBoxLayout()

        # Header showing run name
        header_label = QLabel(f'Recovering: "{self.run_name}"')
        header_font = QFont()
        header_font.setPointSize(11)
        header_font.setBold(True)
        header_label.setFont(header_font)
        layout.addWidget(header_label)

        # Progress section
        progress_label = QLabel("Progress:")
        layout.addWidget(progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        layout.addSpacing(10)

        # Status section
        status_label = QLabel("Status:")
        layout.addWidget(status_label)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        layout.addWidget(self.status_text)

        layout.addStretch()

        # Button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel_clicked)
        layout.addWidget(self.cancel_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def start_recovery(self):
        """Start the recovery worker thread"""
        self.worker = RecoveryWorker(self.run_data, self.run_name)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.status_updated.connect(self.update_status)
        self.worker.recovery_complete.connect(self.on_recovery_complete)
        self.worker.start()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def update_status(self, message):
        """Append status message"""
        self.status_text.append(message)
        # Auto-scroll to bottom
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum()
        )

    def on_recovery_complete(self, success, message):
        """Handle recovery completion"""
        if success:
            self.cancel_button.setText("Close")
            self.cancel_button.clicked.disconnect()
            self.cancel_button.clicked.connect(self.accept)
            QMessageBox.information(self, "Recovery Complete", message)
        else:
            self.cancel_button.setText("Close")
            self.cancel_button.clicked.disconnect()
            self.cancel_button.clicked.connect(self.reject)
            QMessageBox.critical(self, "Recovery Failed", message)

    def on_cancel_clicked(self):
        """Handle cancel button click"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Cancel Recovery",
                "Are you sure you want to cancel the recovery process?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.worker.cancel()
                self.worker.wait()  # Wait for thread to finish
                self.reject()
        else:
            self.reject()


class RecoverUnnamedRunsDialog(QDialog):
    """Main dialog for recovering unnamed runs"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.unnamed_runs = []  # Will be populated with UnnamedRun objects
        self.selected_run = None
        self.setup_ui()
        self.load_unnamed_runs()

    def setup_ui(self):
        self.setWindowTitle("Recover Unnamed Runs")
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)

        main_layout = QVBoxLayout()

        # Create horizontal splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - List of unnamed runs
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)

        list_label = QLabel("Unnamed Runs")
        list_label_font = QFont()
        list_label_font.setBold(True)
        list_label.setFont(list_label_font)
        left_layout.addWidget(list_label)

        self.runs_list = QListWidget()
        self.runs_list.currentItemChanged.connect(self.on_selection_changed)
        left_layout.addWidget(self.runs_list)

        left_panel.setLayout(left_layout)

        # Right panel - Details view
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)

        details_label = QLabel("Run Details")
        details_label_font = QFont()
        details_label_font.setBold(True)
        details_label.setFont(details_label_font)
        right_layout.addWidget(details_label)

        # Details frame with form layout
        self.details_frame = QFrame()
        self.details_frame.setFrameShape(QFrame.StyledPanel)
        details_form = QFormLayout()

        self.detail_datetime = QLabel("—")
        self.detail_duration = QLabel("—")
        self.detail_points = QLabel("—")
        self.detail_ruling = QLabel("—")
        self.detail_filesize = QLabel("—")

        details_form.addRow("Date/Time:", self.detail_datetime)
        details_form.addRow("Duration:", self.detail_duration)
        details_form.addRow("Data Points:", self.detail_points)
        details_form.addRow("Ruling:", self.detail_ruling)
        details_form.addRow("File Size:", self.detail_filesize)

        self.details_frame.setLayout(details_form)
        right_layout.addWidget(self.details_frame)

        # Optional: Preview graph placeholder
        # self.preview_widget = QWidget()
        # self.preview_widget.setMinimumHeight(200)
        # right_layout.addWidget(self.preview_widget)

        right_layout.addStretch()

        # Recover button
        self.recover_button = QPushButton("Recover This Run")
        self.recover_button.setEnabled(False)
        self.recover_button.clicked.connect(self.on_recover_clicked)
        right_layout.addWidget(self.recover_button)

        right_panel.setLayout(right_layout)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)  # Left panel
        splitter.setStretchFactor(1, 2)  # Right panel gets more space

        main_layout.addWidget(splitter)

        # Status bar at bottom
        self.status_label = QLabel("Loading unnamed runs...")
        self.status_label.setStyleSheet(
            "QLabel { padding: 5px; background-color: #f0f0f0; }"
        )
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)

    def load_unnamed_runs(self):
        """Load unnamed runs from your data source"""
        # TODO: Replace this with your actual data loading logic
        # This is sample data for demonstration
        sample_runs = [
            UnnamedRun("2026-04-21 14:32:15", 245, 1024, "Good", 2.3),
            UnnamedRun("2026-04-21 09:15:43", 312, 2048, "Good", 3.1),
            UnnamedRun("2026-04-20 16:48:22", 189, 512, "Bad", 1.2),
            UnnamedRun("2026-04-19 11:05:10", 267, 1536, "Good", 2.8),
        ]

        self.unnamed_runs = sample_runs

        # Populate list widget
        for run in self.unnamed_runs:
            item = QListWidgetItem(run.timestamp)
            item.setData(Qt.UserRole, run)  # Store run object with item
            self.runs_list.addItem(item)

        # Update status
        count = len(self.unnamed_runs)
        self.status_label.setText(
            f"Status: {count} unnamed run{'s' if count != 1 else ''} found"
        )

    def on_selection_changed(self, current, previous):
        """Handle selection change in list"""
        if current is None:
            self.clear_details()
            self.recover_button.setEnabled(False)
            self.selected_run = None
            return

        # Get the UnnamedRun object from the item
        run = current.data(Qt.UserRole)
        self.selected_run = run

        # Update details panel
        self.detail_datetime.setText(run.timestamp)
        self.detail_duration.setText(f"{run.duration_seconds} seconds")
        self.detail_points.setText(f"{run.num_points:,}")

        # Color-code ruling
        ruling_text = run.ruling
        if run.ruling == "Good":
            self.detail_ruling.setText(
                f'<span style="color: green; font-weight: bold;">{ruling_text}</span>'
            )
        else:
            self.detail_ruling.setText(
                f'<span style="color: orange; font-weight: bold;">{ruling_text}</span>'
            )

        self.detail_filesize.setText(f"{run.file_size_mb} MB")

        # Enable recover button
        self.recover_button.setEnabled(True)

    def clear_details(self):
        """Clear the details panel"""
        self.detail_datetime.setText("—")
        self.detail_duration.setText("—")
        self.detail_points.setText("—")
        self.detail_ruling.setText("—")
        self.detail_filesize.setText("—")

    def on_recover_clicked(self):
        """Handle recover button click"""
        if self.selected_run is None:
            return

        # Prompt for run name using QInputDialog
        run_name, ok = QInputDialog.getText(
            self,
            "Recover Run",
            f"Enter name for run from {self.selected_run.timestamp}:",
            text="",
        )

        # User clicked OK and provided a name
        if ok and run_name.strip():
            # Validate name
            if self.validate_run_name(run_name):
                # Launch recovery dialog
                recovery_dialog = RecoveryProgressDialog(
                    self, self.selected_run, run_name
                )
                result = recovery_dialog.exec_()

                if result == QDialog.Accepted:
                    # Recovery successful - remove from list
                    current_item = self.runs_list.currentItem()
                    row = self.runs_list.row(current_item)
                    self.runs_list.takeItem(row)
                    self.unnamed_runs.remove(self.selected_run)

                    # Update status
                    count = len(self.unnamed_runs)
                    self.status_label.setText(
                        f"Status: {count} unnamed run{'s' if count != 1 else ''} found"
                    )

                    # Clear selection
                    self.clear_details()
                    self.recover_button.setEnabled(False)
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Name",
                    "Run name already exists or contains invalid characters.",
                )

    def validate_run_name(self, name):
        """Validate the run name"""
        # TODO: Implement your actual validation logic
        # Check for:
        # - Empty name
        # - Invalid characters
        # - Duplicate names
        # - Length restrictions

        if not name.strip():
            return False

        # Example: Check for invalid characters
        invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        if any(char in name for char in invalid_chars):
            return False

        # Example: Check for duplicate (you'd check against your actual database)
        # existing_names = self.get_existing_run_names()
        # if name in existing_names:
        #     return False

        return True


# Example usage
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    dialog = RecoverUnnamedRunsDialog()
    dialog.show()

    sys.exit(app.exec_())
