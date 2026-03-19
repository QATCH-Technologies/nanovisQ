"""
sample_generation_worker.py

Background worker for automated formulation sample generation.

This module provides the SampleGenerationWorker class, which handles the
asynchronous generation of design-of-experiment (DoE) samples. It utilizes
active learning (UCB) and user-defined constraints to suggest new formulation
candidates for testing.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

import os

from PyQt5 import QtCore

try:
    TAG = "[SampleGenerationWorker (HEADLESS)]"
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.managers.version_manager import VersionManager
    from src.processors.sampler import Sampler
    from src.utils.constraints import Constraints

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

except (ModuleNotFoundError, ImportError):
    TAG = "[SampleGenerationWorker]"
    from QATCH.common.logger import Logger as Log
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.managers.version_manager import VersionManager
    from QATCH.VisQAI.src.processors.sampler import Sampler
    from QATCH.VisQAI.src.utils.constraints import Constraints


class SampleGenerationWorker(QtCore.QThread):
    """A background thread that generates suggested formulation samples.

    This worker initializes a local database connection, builds a Constraints
    object from UI data, and uses a Sampler to suggest the next best
    experiments. It communicates progress and results back to the UI via
    Qt signals.

    Attributes:
        progress_update (QtCore.pyqtSignal): Emits (int, str) representing the
            completion percentage and a status message.
        generation_complete (QtCore.pyqtSignal): Emits a list of dictionaries
            containing the generated formulation data.
        generation_error (QtCore.pyqtSignal): Emits a string error message if
            the generation fails.
        num_samples (int): The number of samples to generate.
        model_path (str): The full path to the `.visq` model file used to guide
            sampling. The file will be committed to the version repository if not
            already tracked, and the resulting SHA is passed to the Sampler.
        constraints_data (list[dict]): Raw constraint definitions from the UI.
        _is_running (bool): Flag to support graceful termination of the thread.
    """

    progress_update = QtCore.pyqtSignal(int, str)
    generation_complete = QtCore.pyqtSignal(list)
    generation_error = QtCore.pyqtSignal(str)

    def __init__(self, num_samples, model_path, constraints_data, parent=None):
        """Initializes the worker with generation parameters.

        Args:
            num_samples (int): Count of formulations to suggest.
            model_path (str): Full path to the `.visq` model file. The file is
                committed to the version repository (idempotently) on first use
                so its SHA-256 digest can be resolved for the Sampler.
            constraints_data (list[dict]): Constraint configurations including
                ingredient, attribute, condition, and values.
            parent (QObject, optional): Parent object. Defaults to None.
        """
        super().__init__(parent)
        self.num_samples = num_samples
        self.model_path = model_path
        self.constraints_data = constraints_data
        self._is_running = True

    def run(self):
        """Executes the sample generation logic.

        The worker performs the following steps:
            1. Fetches all ingredients from the database locally in the
               background thread to avoid cross-thread issues.
            2. Translates UI constraints (e.g., Protein Class) into specific
               ingredient choices or numeric ranges.
            3. Commits the `.visq` model file to the version repository
               (idempotently) to obtain its SHA-256 digest, then initializes
               the `Sampler` with that digest.
            4. Iteratively generates formulations using Upper Confidence
               Bound (UCB) sampling.
            5. Packages each result into a `card_data` dictionary for the UI.

        The `card_data` schema for each sample:
            * name (str): "Generated Sample X"
            * measured (bool): False
            * model (str): The model filename
            * temperature (float): Formulation temperature
            * ingredients (dict): Mappings of ingredient types to detailed
              component data
        """
        db = None
        try:
            db = Database(parse_file_key=True)
            ing_ctrl = IngredientController(db)
            local_ingredients_by_type = {
                "Protein": ing_ctrl.get_all_proteins(),
                "Buffer": ing_ctrl.get_all_buffers(),
                "Salt": ing_ctrl.get_all_salts(),
                "Surfactant": ing_ctrl.get_all_surfactants(),
                "Stabilizer": ing_ctrl.get_all_stabilizers(),
                "Excipient": ing_ctrl.get_all_excipients(),
            }

            self.progress_update.emit(0, "Building constraints...")
            constraints = Constraints(db)

            for item in self.constraints_data:
                ingredient = item["ingredient"]
                attr = item["attribute"]
                cond = item["condition"]
                values = item["values"]

                if attr == "Class" and ingredient == "Protein":
                    proteins = local_ingredients_by_type.get("Protein", [])
                    matched_names = []
                    for p in proteins:
                        if hasattr(p, "class_type") and p.class_type:
                            c_val = str(
                                getattr(
                                    p.class_type,
                                    "value",
                                    getattr(p.class_type, "name", str(p.class_type)),
                                )
                            )
                            if c_val in values:
                                matched_names.append(p.name)
                    attr = "Type"
                    values = matched_names

                feature_key = f"{ingredient}_{attr[:4].lower()}"

                if feature_key in Constraints._CATEGORICAL:
                    choices = []
                    negate = cond == "is not"

                    all_ings = local_ingredients_by_type.get(ingredient, [])

                    for ing in all_ings:
                        match = ing.name in values
                        if (not negate and match) or (negate and not match):
                            choices.append(ing)
                    if not choices and not negate:
                        Log.w(
                            TAG,
                            f"Categorical constraint '{feature_key} is {values}' matched no "
                            f"ingredients in the database. Constraint will be ignored.",
                        )
                        continue

                    constraints.add_choices(feature=feature_key, choices=choices)

                elif feature_key in Constraints._NUMERIC:
                    v = float(values)
                    if cond == ">":
                        constraints.add_range(
                            feature=feature_key, low=v + 0.001, high=10000.0
                        )
                    elif cond == ">=":
                        constraints.add_range(feature=feature_key, low=v, high=10000.0)
                    elif cond == "=":
                        constraints.add_range(feature=feature_key, low=v, high=v)
                    elif cond == "<=":
                        constraints.add_range(feature=feature_key, low=0.0, high=v)
                    elif cond == "<":
                        constraints.add_range(
                            feature=feature_key, low=0.0, high=max(0.0, v - 0.001)
                        )
                    elif cond == "!=":
                        low_range = (0.0, max(0.0, v - 0.001))
                        high_range = (v + 0.001, 10000.0)
                        if (high_range[1] - high_range[0]) >= (
                            low_range[1] - low_range[0]
                        ):
                            constraints.add_range(
                                feature=feature_key,
                                low=high_range[0],
                                high=high_range[1],
                            )
                        else:
                            constraints.add_range(
                                feature=feature_key, low=low_range[0], high=low_range[1]
                            )

            self.progress_update.emit(5, "Initializing prediction engine...")
            base_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))
            repo_dir = os.path.join(project_root, "repo")
            vm = VersionManager(repo_dir=repo_dir)
            sha = vm.commit(self.model_path)
            sampler = Sampler(
                sha=sha, database=db, repo_dir=repo_dir, constraints=constraints
            )

            generated_cards_data = []
            for i in range(self.num_samples):
                if not self._is_running:
                    break

                progress_val = int(5 + ((i / self.num_samples) * 90))
                self.progress_update.emit(
                    progress_val, f"Generating sample {i + 1} of {self.num_samples}..."
                )

                new_formulation = sampler.get_next_sample(use_ucb=True)

                if new_formulation:
                    sampler.add_sample(new_formulation)
                    ingredients_map = {}
                    attr_map = {
                        "protein": "Protein",
                        "buffer": "Buffer",
                        "surfactant": "Surfactant",
                        "stabilizer": "Stabilizer",
                        "excipient": "Excipient",
                        "salt": "Salt",
                    }

                    for model_attr, ui_type in attr_map.items():
                        comp = getattr(new_formulation, model_attr, None)
                        if comp and comp.ingredient:
                            ingredients_map[ui_type] = {
                                "name": comp.ingredient.name,
                                "component": comp.ingredient.name,
                                "concentration": comp.concentration,
                                "units": comp.units,
                            }

                    card_data = {
                        "name": f"Generated Sample {i + 1}",
                        "measured": False,
                        "model": os.path.basename(self.model_path),
                        "temperature": getattr(new_formulation, "temperature", 25.0),
                        "ingredients": ingredients_map,
                    }
                    generated_cards_data.append(card_data)

            self.progress_update.emit(100, "Finalizing UI...")
            if self._is_running:
                self.generation_complete.emit(generated_cards_data)

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.generation_error.emit(str(e))
        finally:
            if db is not None:
                db.close()

    def stop(self):
        """Signals the worker thread to stop and terminate its loop."""
        self._is_running = False
