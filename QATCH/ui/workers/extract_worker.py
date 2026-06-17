"""
extract_worker.py

This module defines the ExtractWorker class, which handles asynchronous ZIP
extraction to prevent main UI thread blocking. It includes built-in safeguards
for signal throttling, error handling, cancellation, and Zip-Slip vulnerability prevention.

Author(s)
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-06-17
"""

import os
import time
import shutil
import pyzipper
from PyQt5 import QtCore


class ExtractWorker(QtCore.QThread):
    """A threaded worker for extracting ZIP archives with UI progress reporting.

    This thread safely handles extracting files while emitting throttled signals
    to update the UI. It safeguards against path traversal attacks and cleans up
    nested directory structures upon completion.

    Attributes:
        label_text (QtCore.pyqtSignal): Signal emitting the current status or file name (str).
        set_range (QtCore.pyqtSignal): Signal emitting the min and max range for a progress bar (int, int).
        progress (QtCore.pyqtSignal): Signal emitting the current progress value (int).
        finished (QtCore.pyqtSignal): Signal emitted when the thread completes, regardless of success or failure.
        error (QtCore.pyqtSignal): Signal emitting an error message if the extraction fails (str).
    """

    label_text = QtCore.pyqtSignal(str)
    set_range = QtCore.pyqtSignal(int, int)
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, save_to: str, new_install_path: str) -> None:
        """Initializes the ExtractWorker thread.

        Args:
            save_to (str): The absolute file path to the source ZIP archive.
            new_install_path (str): The absolute directory path where the contents should be extracted.
        """
        super().__init__()
        self.save_to = save_to
        self.new_install_path = new_install_path
        self._is_cancelled = False

    def cancel(self) -> None:
        """Flags the extraction process to be aborted.

        This method is thread-safe and designed to be called from the main UI
        thread. The worker will halt extraction on the next file iteration.
        """
        self._is_cancelled = True

    def run(self) -> None:
        """Executes the thread's main operation.

        Wraps the core extraction logic in a safety block to ensure that any
        unhandled exceptions are caught and reported to the UI, and guarantees
        that the finished signal is always emitted.
        """
        try:
            self._execute_extraction()
        except Exception as e:
            self.error.emit(f"Extraction failed: {str(e)}")
        finally:
            # Always ensure the UI knows the thread is done, even on failure
            self.finished.emit()

    def _execute_extraction(self) -> None:
        """Performs the internal ZIP extraction and file manipulation logic.

        Reads the ZIP file and extracts contents iteratively, checking for
        cancellation and path traversal (Zip-Slip) attempts. Throttles UI updates
        to ensure smooth performance. Finalizes the setup by resolving incorrectly
        nested root directories and attempting to delete the source archive.

        Raises:
            Exception: If a file inside the archive attempts to extract outside
                the designated target directory.
            OSError: If file system operations during cleanup fail.
        """
        zip_filename = os.path.basename(self.save_to)[:-4]
        target_path = self.new_install_path

        if os.path.basename(target_path) != zip_filename:
            target_path = os.path.join(target_path, zip_filename)

        os.makedirs(target_path, exist_ok=True)
        absolute_target = os.path.abspath(target_path)

        with pyzipper.AESZipFile(self.save_to, "r") as zf:
            file_list = zf.namelist()
            total = len(file_list)
            self.set_range.emit(0, total)

            last_emit_time = 0
            emit_threshold = 0.05  # Only emit signals every 50ms to prevent UI freezing

            for i, file in enumerate(file_list):
                if self._is_cancelled:
                    self.error.emit("Extraction cancelled by user.")
                    return

                extracted_path = os.path.abspath(os.path.join(absolute_target, file))
                if not extracted_path.startswith(absolute_target):
                    raise Exception(f"Attempted path traversal detected: {file}")

                zf.extract(file, target_path)
                current_time = time.time()
                if (current_time - last_emit_time > emit_threshold) or (i == total - 1):
                    # Show only the file name, not the whole path, for a cleaner UI
                    self.label_text.emit(f"Extracting: {os.path.basename(file)}")
                    self.progress.emit(i + 1)
                    last_emit_time = current_time

        self.label_text.emit("Finalizing setup...")
        nested_path_wrong = os.path.join(target_path, zip_filename)

        if os.path.exists(nested_path_wrong) and os.path.isdir(nested_path_wrong):
            temp_path = target_path + "_temp"
            shutil.move(nested_path_wrong, temp_path)
            shutil.rmtree(target_path)  # Clear the now-empty parent
            os.rename(temp_path, target_path)

            # Relocate original zip if it happened to be inside the target path
            expected_zip_loc = os.path.join(target_path, zip_filename + ".zip")
            if os.path.dirname(self.save_to) == self.new_install_path and os.path.exists(
                expected_zip_loc
            ):
                self.save_to = expected_zip_loc

        if os.path.exists(self.save_to):
            try:
                os.remove(self.save_to)
            except OSError as e:
                self.label_text.emit(f"Note: Could not delete temporary zip: {e}")
