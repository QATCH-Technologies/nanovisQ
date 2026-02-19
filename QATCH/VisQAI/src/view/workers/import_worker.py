import os

from PyQt5.QtCore import QThread, pyqtSignal

# Import Parser
try:
    from src.io.parser import Parser
except (ImportError, ModuleNotFoundError):
    from QATCH.VisQAI.src.io.parser import Parser

# Import Database and Controller
try:
    from src.controller.formulation_controller import FormulationController
    from src.db.db import Database
except (ImportError, ModuleNotFoundError):
    from QATCH.VisQAI.src.controller.formulation_controller import FormulationController
    from QATCH.VisQAI.src.db.db import Database


class ImportWorker(QThread):
    """
    Worker thread to recursively find and parse XML files from directories.
    """

    # Signals to communicate with the main UI thread
    progress_changed = pyqtSignal(int)  # Progress value (0-100)
    status_changed = pyqtSignal(str)  # Status message
    import_finished = pyqtSignal(list)  # Successful results
    import_error = pyqtSignal(str)  # Error message

    def __init__(self, file_paths):
        super().__init__()
        self.input_paths = file_paths
        self._is_running = True

    def run(self):
        imported_data = []

        # --- THREAD-SAFE DATABASE INITIALIZATION ---
        database = None

        try:
            self.status_changed.emit("Scanning directories for runs...")

            # --- PHASE 1: DISCOVERY ---
            # Flatten inputs into a list of specific XML files
            xml_files_to_process = self._discover_xml_files(self.input_paths)
            total_files = len(xml_files_to_process)

            if total_files == 0:
                self.import_error.emit(
                    "No XML files found in the selected directories."
                )
                return

            # --- PHASE 2: PROCESSING ---
            database = Database(parse_file_key=True)
            controller = FormulationController(database)
            parser = Parser()

            for i, fname in enumerate(xml_files_to_process):
                if not self._is_running:
                    break

                filename = os.path.basename(fname)
                self.status_changed.emit(f"Importing {filename}...")

                # Parse specific file
                # parser.parse now handles list or str, returns list of formulations
                parsed_forms = parser.parse(fname)

                if parsed_forms:
                    for formulation in parsed_forms:
                        if not self._is_running:
                            break

                        # --- Missing Fields Check ---
                        if (
                            hasattr(formulation, "missing_fields")
                            and formulation.missing_fields
                        ):
                            missing_str = ", ".join(formulation.missing_fields)
                            self.status_changed.emit(
                                f"Warning: {filename} missing {missing_str}"
                            )

                        # Database Sync Logic
                        signature = formulation.signature
                        final_formulation = None

                        if signature:
                            existing_form = controller.get_formulation_by_signature(
                                signature
                            )

                            if existing_form:
                                # Check if the parsed formulation differs from the stored one
                                if existing_form != formulation:
                                    # Preserve local metadata (like UI toggles) before updating
                                    formulation.icl = getattr(
                                        existing_form, "icl", True
                                    )
                                    formulation.last_model = getattr(
                                        existing_form, "last_model", None
                                    )

                                    try:
                                        # Update the formulation in the database
                                        final_formulation = (
                                            controller.update_formulation(
                                                existing_form.id, formulation
                                            )
                                        )
                                    except Exception as e:
                                        print(
                                            f"Failed to update existing formulation {filename}: {e}"
                                        )
                                        final_formulation = existing_form
                                else:
                                    final_formulation = existing_form

                                # Attach any missing fields for UI notification
                                if hasattr(formulation, "missing_fields"):
                                    final_formulation.missing_fields = (
                                        formulation.missing_fields
                                    )
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
            self.import_error.emit(
                f"Dependency Import Error: {e}. Check project structure."
            )
        except Exception as e:
            self.import_error.emit(f"Error during import: {str(e)}")
        finally:
            if database:
                try:
                    database.close()
                except:
                    pass

    def _discover_xml_files(self, paths):
        """Recursively finds all .xml files in the given paths."""
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
        """Helper to calculate and emit progress."""
        progress = int(((current_index + 1) / total_files) * 100)
        self.progress_changed.emit(progress)

    def stop(self):
        self._is_running = False
