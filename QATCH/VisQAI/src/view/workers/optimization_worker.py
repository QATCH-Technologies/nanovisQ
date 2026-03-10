"""
optimization_worker.py

Background QThread that builds constraints, loads a predictor, constructs
a target ViscosityProfile, runs the differential-evolution Optimizer, and
emits the result as a card-data dict ready to be inserted by the dashboard.

Signals
-------
progress_update(int, str)           – percent (0-100) + status message
optimization_complete(dict)         – card_data dict (see run() docstring)
optimization_error(str)             – human-readable error message
"""

import os

import numpy as np
import pandas as pd
from PyQt5 import QtCore

try:
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.models.formulation import ViscosityProfile
    from src.models.predictor import Predictor
    from src.processors.optimizer import Optimizer
    from src.utils.constraints import Constraints
    from src.view.architecture import Architecture
except (ModuleNotFoundError, ImportError):
    from QATCH.common.architecture import Architecture
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.models.formulation import ViscosityProfile
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.processors.optimizer import Optimizer
    from QATCH.VisQAI.src.utils.constraints import Constraints


class OptimizationWorker(QtCore.QThread):
    """
    Runs Optimizer in a background thread and emits card_data on success.

    card_data keys
    --------------
    name             str   "Optimized Formulation"
    optimized        bool  True
    model            str   filename of the .visq asset
    temperature      float
    ingredients      dict  {type: {name, component, concentration, units}}
    estimated_profile dict {shear_rates: list, viscosities: list}
    targets          list  [{shear_rate, viscosity}, ...]  — the user's inputs
    """

    progress_update = QtCore.pyqtSignal(int, str)
    optimization_complete = QtCore.pyqtSignal(dict)
    optimization_error = QtCore.pyqtSignal(str)

    def __init__(self, model_file, targets, constraints_data, maxiter=100, parent=None):
        super().__init__(parent)
        self.model_file = model_file
        self.targets = targets  # list[{shear_rate, viscosity}]
        self.constraints_data = (
            constraints_data  # same schema as SampleGenerationWorker
        )
        self.maxiter = maxiter
        self._is_running = True

    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        db = None
        try:
            db = Database(parse_file_key=True)

            # ── 1. Fetch ingredients ───────────────────────────────────────────
            self.progress_update.emit(2, "Fetching ingredient database...")
            ing_ctrl = IngredientController(db)
            local_by_type = {
                "Protein": ing_ctrl.get_all_proteins(),
                "Buffer": ing_ctrl.get_all_buffers(),
                "Salt": ing_ctrl.get_all_salts(),
                "Surfactant": ing_ctrl.get_all_surfactants(),
                "Stabilizer": ing_ctrl.get_all_stabilizers(),
                "Excipient": ing_ctrl.get_all_excipients(),
            }

            # ── 2. Build constraints ───────────────────────────────────────────
            self.progress_update.emit(8, "Building constraints...")
            constraints = Constraints(db)

            for item in self.constraints_data:
                ingredient = item["ingredient"]
                attr = item["attribute"]
                cond = item["condition"]
                values = item["values"]

                # Translate Class → Type (same logic as SampleGenerationWorker)
                if attr == "Class" and ingredient == "Protein":
                    matched = []
                    for p in local_by_type.get("Protein", []):
                        if hasattr(p, "class_type") and p.class_type:
                            c_val = str(
                                getattr(
                                    p.class_type,
                                    "value",
                                    getattr(p.class_type, "name", str(p.class_type)),
                                )
                            )
                            if c_val in values:
                                matched.append(p.name)
                    attr = "Type"
                    values = matched

                feature_key = f"{ingredient}_{attr[:4].lower()}"

                if feature_key in Constraints._CATEGORICAL:
                    negate = cond == "is not"
                    all_ings = local_by_type.get(ingredient, [])

                    # Case-insensitive, whitespace-stripped comparison.
                    # The widget builds its list from ingredients_by_type (passed
                    # at creation time) while the worker uses a fresh DB query.
                    # If the DB name has trailing spaces, different casing, or
                    # unicode quirks, a plain `in` check silently skips the
                    # constraint and allows ALL ingredients — the root cause of
                    # the categorical constraint not being respected.
                    values_norm = {
                        str(v).strip().lower()
                        for v in (values if isinstance(values, list) else [values])
                    }
                    choices = [
                        ing
                        for ing in all_ings
                        if (not negate and ing.name.strip().lower() in values_norm)
                        or (negate and ing.name.strip().lower() not in values_norm)
                    ]

                    # Guard: if "is X" matched nothing, raise — a silent skip
                    # means the constraint is ignored entirely which is worse
                    # than an explicit error the user can act on.
                    if not choices and not negate:
                        raise ValueError(
                            f"Categorical constraint '{feature_key} is {values}' "
                            f"matched no ingredients in the database. "
                            f"Check that the selected name exactly matches a DB entry."
                        )
                    print(
                        f"[Optimizer] Constraint applied: {feature_key} "
                        f"{'is NOT' if negate else 'is'} {[c.name for c in choices]}"
                    )
                    constraints.add_choices(feature=feature_key, choices=choices)

                elif feature_key in Constraints._NUMERIC:
                    v = float(values)
                    if cond == ">":
                        constraints.add_range(feature_key, v + 0.001, 10_000.0)
                    elif cond == ">=":
                        constraints.add_range(feature_key, v, 10_000.0)
                    elif cond == "=":
                        constraints.add_range(feature_key, v, v)
                    elif cond == "<=":
                        constraints.add_range(feature_key, 0.0, v)
                    elif cond == "<":
                        constraints.add_range(feature_key, 0.0, max(0.0, v - 0.001))
                    elif cond == "!=":
                        low_r = (0.0, max(0.0, v - 0.001))
                        high_r = (v + 0.001, 10_000.0)
                        if (high_r[1] - high_r[0]) >= (low_r[1] - low_r[0]):
                            constraints.add_range(feature_key, high_r[0], high_r[1])
                        else:
                            constraints.add_range(feature_key, low_r[0], low_r[1])

            # ── 3. Load model ──────────────────────────────────────────────────
            self.progress_update.emit(15, "Loading prediction model...")
            # Resolve the .visq file directly from the assets directory — the
            # same path that OptimizeWidget (and GenerateSampleWidget) use when
            # populating the model combo.  No AssetManager required.
            assets_dir = os.path.join(
                Architecture.get_path(), "QATCH", "VisQAI", "assets"
            )
            asset_zip = os.path.join(assets_dir, self.model_file)
            if not os.path.isfile(asset_zip):
                raise FileNotFoundError(
                    f"Model file '{self.model_file}' not found in '{assets_dir}'."
                )
            predictor = Predictor(zip_path=asset_zip)

            # ── 4. Build target ViscosityProfile ──────────────────────────────
            self.progress_update.emit(20, "Building target viscosity profile...")
            shear_rates = [t["shear_rate"] for t in self.targets]
            target_viscs = [t["viscosity"] for t in self.targets]
            target_profile = ViscosityProfile(
                shear_rates=shear_rates,
                viscosities=target_viscs,
            )

            # ── 5. Run differential evolution ─────────────────────────────────
            self.progress_update.emit(
                25, "Starting differential evolution optimizer..."
            )

            def _progress_cb(status):
                if not self._is_running:
                    return
                frac = status.iteration / max(status.num_iterations, 1)
                pct = int(25 + frac * 68)
                self.progress_update.emit(
                    pct,
                    f"Optimizing… iteration {status.iteration}/{status.num_iterations}"
                    f"  ·  best loss: {status.best_value:.4f}",
                )

            optimizer = Optimizer(
                constraints=constraints,
                predictor=predictor,
                target=target_profile,
                maxiter=self.maxiter,
                popsize=15,
                seed=42,
            )
            best_formulation = optimizer.optimize(progress_callback=_progress_cb)

            # Diagnostic: log what the optimizer actually chose for each type
            for _attr in (
                "protein",
                "buffer",
                "salt",
                "surfactant",
                "stabilizer",
                "excipient",
            ):
                _comp = getattr(best_formulation, _attr, None)
                if _comp and getattr(_comp, "ingredient", None):
                    print(
                        f"[Optimizer] Best {_attr}: {_comp.ingredient.name} @ {_comp.concentration}"
                    )

            if not self._is_running:
                return

            # ── 6. Predict estimated profile for the result ───────────────────
            self.progress_update.emit(94, "Computing estimated viscosity profile...")
            pred_df = best_formulation.to_dataframe(encoded=False, training=False)
            pred_raw = predictor.predict(pred_df)
            estimated_shear = [100, 1_000, 10_000, 100_000, 15_000_000]

            # predictor.predict() returns a DataFrame, not a numpy array.
            # Extract the five viscosity columns in canonical order.
            _visc_cols = [
                "Viscosity_100",
                "Viscosity_1000",
                "Viscosity_10000",
                "Viscosity_100000",
                "Viscosity_15000000",
            ]
            if isinstance(pred_raw, pd.DataFrame):
                row = pred_raw.iloc[0] if len(pred_raw) > 0 else pd.Series(dtype=float)
                estimated_visc = []
                for col in _visc_cols:
                    v = row.get(col, row.get(f"Pred_{col}", float("nan")))
                    try:
                        estimated_visc.append(float(v))
                    except (TypeError, ValueError):
                        estimated_visc.append(float("nan"))
            else:
                estimated_visc = [
                    float(v) for v in np.asarray(pred_raw, dtype=float).flatten()
                ]

            # ── 7. Pack result as card_data ────────────────────────────────────
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
                comp = getattr(best_formulation, model_attr, None)
                if comp and getattr(comp, "ingredient", None):
                    ingredients_map[ui_type] = {
                        "name": comp.ingredient.name,
                        "component": comp.ingredient.name,
                        "concentration": comp.concentration,
                        "units": comp.units,
                    }

            card_data = {
                "name": "Optimized Formulation",
                "measured": False,
                "optimized": True,
                "model": self.model_file,
                "temperature": getattr(best_formulation, "temperature", 25.0),
                "ingredients": ingredients_map,
                "estimated_profile": {
                    "shear_rates": estimated_shear,
                    "viscosities": estimated_visc,
                },
                "targets": self.targets,
            }

            self.progress_update.emit(100, "Optimization complete.")
            if self._is_running:
                self.optimization_complete.emit(card_data)

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.optimization_error.emit(str(e))
        finally:
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass

    def stop(self):
        self._is_running = False
