from typing import Dict, Any, List, Tuple, Type
import numpy as np
from scipy.optimize import differential_evolution
import math

try:
    from src.models.formulation import ViscosityProfile, Formulation
    from src.models.predictor import Predictor
    from src.controller.ingredient_controller import IngredientController
    from src.models.ingredient import Ingredient, Protein, Buffer, Salt, Stabilizer, Surfactant
    from src.db.db import Database
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.formulation import ViscosityProfile, Formulation
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.models.ingredient import Protein, Buffer, Salt, Stabilizer, Surfactant
    from QATCH.VisQAI.src.db.db import Database


class Constraints:
    _NUMERIC: List[str] = [
        "Protein_conc",
        "Temperature",
        "Buffer_conc",
        "Salt_conc",
        "Stabilizer_conc",
        "Surfactant_conc",
    ]
    _CATEGORICAL: List[str] = [
        "Protein_type",
        "Buffer_type",
        "Salt_type",
        "Stabilizer_type",
        "Surfactant_type",
    ]
    _FEATURE_CLASS: Dict[str, Type[Ingredient]] = {
        "Protein_type": Protein,
        "Buffer_type": Buffer,
        "Salt_type": Salt,
        "Stabilizer_type": Stabilizer,
        "Surfactant_type": Surfactant,
    }
    _DEFAULT_RANGES = {
        "Protein_conc": (0, 1000),
        "Temperature": (24, 25),
        "Buffer_conc": (0, 100),
        "Salt_conc": (0, 200),
        "Stabilizer_conc": (0, 1),
        "Surfactant_conc": (0, 1), }

    def __init__(self, db: Database):
        self._db = db
        self._ingredient_ctrl = IngredientController(db=self._db)
        self._ranges: Dict[str, Tuple[float, float]] = {}
        self._choices: Dict[str, List[Ingredient]] = {}

    def add_range(self, feature: str, low: float, high: float) -> None:
        if feature not in self._NUMERIC:
            raise ValueError(
                f"Unknown numeric feature '{feature}'.  Only {self._NUMERIC} are allowed in add_range()."
            )
        if feature != 'Temperature' and (low < 0.0 or high < 0.0):
            raise ValueError(
                f"Negative values are not allowed for numeric feature {feature}"
            )
        self._ranges[feature] = (float(low), float(high))

    def add_choices(self, feature: str, choices: List[Ingredient]) -> None:
        if feature not in self._CATEGORICAL:
            raise ValueError(
                f"Unknown categorical feature '{feature}'.  Only {self._CATEGORICAL} are allowed in add_choices()."
            )
        for c in choices:
            if not isinstance(c, Ingredient):
                raise TypeError(
                    f"All choices for '{feature}' must be Ingredient instances; got {c!r} of type {type(c).__name__}"
                )
            if not self._ingredient_ctrl.get_by_name(name=c.name, ingredient=c):
                raise ValueError(
                    f"`{c.name}` has not been added to persistent store yet.")
        self._choices[feature] = list(choices)

    def build(self) -> Tuple[
        List[Tuple[float, float]],
        List[Dict[str, Any]]
    ]:
        bounds: List[Tuple[float, float]] = []
        encoding: List[Dict[str, Any]] = []

        all_ingredients = self._ingredient_ctrl.get_all_ingredients()
        for ing in all_ingredients:
            print(ing.to_dict())
        all_features = self._CATEGORICAL + self._NUMERIC

        for feat in all_features:
            if feat in self._CATEGORICAL:
                chosen = self._choices.get(feat)
                if chosen is None:
                    cls = self._FEATURE_CLASS[feat]
                    chosen = [
                        ing for ing in all_ingredients if isinstance(ing, cls)]
                if not chosen:
                    raise ValueError(f"No choices available for '{feat}'.")

                names = [ing.name for ing in chosen]
                encoding.append({
                    "feature": feat,
                    "type": "cat",
                    "choices": names
                })
                bounds.append((0.0, float(len(names) - 1)))

            elif feat in self._NUMERIC:
                if feat in self._ranges:
                    low, high = self._ranges[feat]
                else:
                    low, high = self._DEFAULT_RANGES[feat]
                if not (math.isfinite(low) and math.isfinite(high)):
                    raise ValueError(
                        f"Bounds for '{feat}' must be finite. Got ({low}, {high})."
                    )
                encoding.append({"feature": feat, "type": "num"})
                bounds.append((low, high))

            else:
                raise ValueError(f"Unknown feature '{feat}' in build()")

        return bounds, encoding


class Optimizer:
    def __init__(
        self,
        constraints: Constraints,
        predictor: Predictor,
        target: ViscosityProfile,
        maxiter: int = 100,
        popsize: int = 15,
        tol: float = 1e-6,
        seed: int = None,
    ):
        self.constraints = constraints
        self.predictor = predictor
        self.target = target
        self.maxiter = maxiter
        self.popsize = popsize
        self.tol = tol
        self.seed = seed
        self.bounds, self.encoding = self.constraints.build()
        all_ings = self.constraints._ingredient_ctrl.get_all_ingredients()
        self.cat_choices: Dict[str, List[Ingredient]] = {}
        for enc in self.encoding:
            if enc["type"] == "cat":
                feat = enc["feature"]
                choices = self.constraints._choices.get(feat)
                if choices is None:
                    cls = self.constraints._FEATURE_CLASS[feat]
                    choices = [ing for ing in all_ings if isinstance(ing, cls)]
                self.cat_choices[feat] = choices

    def _decode(self, x: np.ndarray) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for xi, enc in zip(x, self.encoding):
            feat = enc["feature"]
            if enc["type"] == "cat":
                idx = int(np.clip(round(xi), 0, len(
                    self.cat_choices[feat]) - 1))
                out[feat] = self.cat_choices[feat][idx]
            else:
                out[feat] = float(xi)
        return out

    def _build_formulation(self, feat_dict: Dict[str, Any]) -> Formulation:
        form = Formulation()
        form.set_buffer(
            buffer=feat_dict["Buffer_type"],
            concentration=feat_dict["Buffer_conc"],
            units="mM"
        )
        form.set_protein(
            protein=feat_dict["Protein_type"],
            concentration=feat_dict["Protein_conc"],
            units="mg/mL"
        )
        form.set_stabilizer(
            stabilizer=feat_dict["Stabilizer_type"],
            concentration=feat_dict["Stabilizer_conc"],
            units="M"
        )
        form.set_salt(
            salt=feat_dict["Salt_type"],
            concentration=feat_dict["Salt_conc"],
            units="mM"
        )
        form.set_surfactant(
            surfactant=feat_dict["Surfactant_type"],
            concentration=feat_dict["Surfactant_conc"],
            units="%w"
        )
        form.set_temperature(
            temp=feat_dict["Temperature"]
        )
        return form

    def _mse_loss(self, prof1: ViscosityProfile, prof2: ViscosityProfile) -> float:
        v1 = np.interp(prof2.shear_rates, prof1.shear_rates, prof1.viscosities)
        return float(np.mean((v1 - prof2.viscosities) ** 2))

    def _objective(self, x: np.ndarray) -> float:
        feat_dict = self._decode(x)
        formulation = self._build_formulation(feat_dict)
        pred = self.predictor.predict(data=formulation.to_dataframe())
        print(pred)
        pred_vp = ViscosityProfile(
            shear_rates=[100, 1000, 10000, 100000, 15000000], viscosities=pred.flatten().tolist())
        print(self.target)

        return self._mse_loss(pred_vp, self.target)

    def optimize(self) -> Formulation:
        result = differential_evolution(
            func=self._objective,
            bounds=self.bounds,
            maxiter=self.maxiter,
            popsize=self.popsize,
            tol=self.tol,
            seed=self.seed,
            polish=True,
        )
        best_feats = self._decode(result.x)
        return self._build_formulation(best_feats)
