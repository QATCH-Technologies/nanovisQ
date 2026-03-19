"""
import_worker.py

Provides a background worker for asynchronous XML data importation.

This module contains the ImportWorker class, which handles the recursive
searching, parsing, and database synchronization of XML files. By running
these operations in a separate QThread, the main GUI remains responsive
during large batch imports.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

import os

from PyQt5.QtCore import QThread, pyqtSignal

try:
    TAG = "[ImportWorker (HEADLESS)]"

    class Log:
        @staticmethod
        def d(TAG, msg=""):
            print("DEBUG:", TAG, msg)

        @staticmethod
        def i(TAG, msg=""):
            print("INFO:", TAG, msg)

        @staticmethod
        def w(TAG, msg=""):
            print("WARNING:", TAG, msg)

        @staticmethod
        def e(TAG, msg=""):
            print("ERROR:", TAG, msg)

    from src.controller.formulation_controller import FormulationController
    from src.db.db import Database
    from src.io.parser import Parser


except (ImportError, ModuleNotFoundError):
    TAG = "[ImportWorker]"
    from QATCH.VisQAI.src.controller.formulation_controller import FormulationController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.io.parser import Parser
    from QATCH.common.logger import Logger as Log


class ImportWorker(QThread):
    """A worker thread for recursive XML discovery and database synchronization.

    This worker flattens a list of input directories and files, parses
    discovered XMLs into Formulation objects, and syncs them with the local
    database using a FormulationController. It emits signals to update the UI
    on progress, status messages, and errors.

    Attributes:
        progress_changed (pyqtSignal): Emits an integer (0-100) representing
            the percentage of files processed.
        status_changed (pyqtSignal): Emits a string message describing the
            current operation or warnings (e.g., missing fields).
        import_finished (pyqtSignal): Emits a list of successfully imported or
            updated Formulation objects upon completion.
        import_error (pyqtSignal): Emits a string describing a critical error
            that halted the import.
        input_paths (list[str]): The initial list of files or directories to
            search for XML data.
        _is_running (bool): Internal flag used to support graceful cancellation
            of the thread.
    """

    progress_changed = pyqtSignal(int)  # Progress value (0-100)
    status_changed = pyqtSignal(str)  # Status message
    import_finished = pyqtSignal(list)  # Successful results
    import_error = pyqtSignal(str)  # Error message

    def __init__(self, file_paths):
        """Initializes the worker with target paths.

        Args:
            file_paths (list[str]): A list of file or directory paths to process.
        """
        super().__init__()
        self.input_paths = file_paths
        self._is_running = True

    def run(self):
        """Executes the import logic in a background thread.

        The process follows two phases:
        1. Discovery: Recursively crawls `input_paths` for .xml files.
        2. Processing: Initializes a thread-local database connection, parses
           each file, and performs a sync (add or update) with the database.

        Metadata such as 'icl' and 'last_model' are preserved when updating
        existing formulations.
        """
        imported_data = []
        database = None
        try:
            self.status_changed.emit("Scanning directories for runs...")
            xml_files_to_process = self._discover_xml_files(self.input_paths)
            total_files = len(xml_files_to_process)

            if total_files == 0:
                self.import_error.emit("No XML files found in the selected directories.")
                return

            database = Database(parse_file_key=True)
            controller = FormulationController(database)
            parser = Parser()

            for i, fname in enumerate(xml_files_to_process):
                if not self._is_running:
                    break

                filename = os.path.basename(fname)
                self.status_changed.emit(f"Importing {filename}...")

                # Parse specific file
                parsed_forms = parser.parse(fname)

                if parsed_forms:
                    for formulation in parsed_forms:
                        if not self._is_running:
                            break
                        if hasattr(formulation, "missing_fields") and formulation.missing_fields:
                            missing_str = ", ".join(formulation.missing_fields)
                            self.status_changed.emit(f"Warning: {filename} missing {missing_str}")

                        # Database Sync Logic
                        signature = formulation.signature
                        final_formulation = None

                        if signature:
                            existing_form = controller.get_formulation_by_signature(signature)

                            if existing_form:
                                if existing_form != formulation:
                                    formulation.icl = getattr(existing_form, "icl", True)
                                    formulation.last_model = getattr(
                                        existing_form, "last_model", None
                                    )

                                    try:
                                        # Update the formulation in the database
                                        final_formulation = controller.update_formulation(
                                            existing_form.id, formulation
                                        )
                                    except Exception as e:
                                        Log.e(
                                            TAG,
                                            f"Failed to update existing formulation {filename}: {e}",
                                        )
                                        final_formulation = existing_form
                                else:
                                    final_formulation = existing_form

                                # Attach any missing fields for UI notification
                                if hasattr(formulation, "missing_fields"):
                                    final_formulation.missing_fields = formulation.missing_fields
                            else:
                                success = controller.add_formulation(formulation)
                                if success:
                                    final_formulation = formulation
                        else:
                            controller.add_formulation(formulation)
                            final_formulation = formulation

                        if final_formulation:
                            imported_data.append(final_formulation)

                # Update progress
                self._emit_progress(i, total_files)

            if self._is_running:
                self.import_finished.emit(imported_data)

        except ImportError as e:
            self.import_error.emit(f"Dependency Import Error: {e}. Check project structure.")
        except Exception as e:
            self.import_error.emit(f"Error during import: {str(e)}")
        finally:
            if database:
                try:
                    database.close()
                except Exception as e:
                    Log.e(TAG, f"Failed to close database with error: `{e}`")

    def _discover_xml_files(self, paths):
        """Recursively finds all .xml files in the given paths.

        Args:
            paths (list[str]): List of filesystem paths to search.

        Returns:
            list[str]: Absolute paths to all discovered XML files.
        """
        found_files = []
        for path in paths:
            if os.path.isfile(path) and path.lower().endswith(".xml"):
                found_files.append(path)
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith(".xml"):
                            found_files.append(os.path.join(root, file))
        return found_files

    def _emit_progress(self, current_index, total_files):
        """Calculates and emits the current progress percentage.

        Args:
            current_index (int): The index of the file just processed.
            total_files (int): The total count of files to process.
        """
        progress = int(((current_index + 1) / total_files) * 100)
        self.progress_changed.emit(progress)

    def stop(self):
        """Signals the worker to stop processing and exit the run loop."""
        self._is_running = False
