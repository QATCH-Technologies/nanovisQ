from PyQt5 import QtCore

try:
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.processors.sampler import Sampler
    from src.utils.constraints import Constraints
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.processors.sampler import Sampler
    from QATCH.VisQAI.src.utils.constraints import Constraints


class SampleGenerationWorker(QtCore.QThread):
    progress_update = QtCore.pyqtSignal(int, str)
    generation_complete = QtCore.pyqtSignal(list)
    generation_error = QtCore.pyqtSignal(str)

    def __init__(self, num_samples, model_file, constraints_data, parent=None):
        super().__init__(parent)
        self.num_samples = num_samples
        self.model_file = model_file
        self.constraints_data = constraints_data
        self._is_running = True

    def run(self):

        db = None
        try:
            db = Database(parse_file_key=True)

            # --- THE FIX: Fetch ingredients locally in the background thread ---
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

                    # Use the locally fetched ingredients, NOT the main thread's objects
                    all_ings = local_ingredients_by_type.get(ingredient, [])

                    for ing in all_ings:
                        match = ing.name in values
                        if (not negate and match) or (negate and not match):
                            choices.append(ing)

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
            asset_name = self.model_file.replace(".visq", "")
            sampler = Sampler(
                asset_name=asset_name, database=db, constraints=constraints
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
                        "model": self.model_file,
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
        self._is_running = False
