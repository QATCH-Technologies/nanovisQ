from typing import Dict, Any, List, Tuple, Type
import numpy as np
from scipy.optimize import differential_evolution
import os

# try both src/ and QATCH.VisQAI paths
try:
    from src.models.formulation import ViscosityProfile, Formulation
    from src.models.predictor import Predictor
    from src.controller.asset_controller import AssetController
    from src.controller.ingredient_controller import IngredientController
    from src.models.ingredient import Protein, Buffer, Salt, Stabilizer, Surfactant
    from src.db.db import Database
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.formulation import ViscosityProfile, Formulation
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.controller.asset_controller import AssetController
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.models.ingredient import Protein, Buffer, Salt, Stabilizer, Surfactant
    from QATCH.VisQAI.src.db.db import Database


FEATURES: List[str] = [
    "Protein_type", "MW", "PI_mean", "PI_range", "Protein_conc",
    "Temperature", "Buffer_type", "Buffer_pH", "Buffer_conc",
    "Salt_type", "Salt_conc", "Stabilizer_type", "Stabilizer_conc",
    "Surfactant_type", "Surfactant_conc",
]

CATEGORICAL: Dict[str, Type] = {
    "Protein_type": Protein,
    "Buffer_type": Buffer,
    "Salt_type": Salt,
    "Stabilizer_type": Stabilizer,
    "Surfactant_type": Surfactant,
}
IMMUTABLE_FEATURES = {"MW", "PI_mean", "PI_range", "Buffer_pH"}


class Constraints:
    def __init__(
        self,
        ingredient_ctrl: IngredientController,
        fixed_properties: Dict[str, float]
    ):
        # must include values for ALL of IMMUTABLE_FEATURES
        missing = IMMUTABLE_FEATURES - fixed_properties.keys()
        if missing:
            raise ValueError(f"Must supply fixed_properties for {missing}")
        self._ingredient_ctrl = ingredient_ctrl
        self._fixed_props = fixed_properties
        self._ranges: Dict[str, Tuple[float, float]] = {}
        self._choices: Dict[str, List[Any]] = {}

    def add_range(self, feature: str, low: float, high: float) -> None:
        if feature in CATEGORICAL:
            raise ValueError(f"'{feature}' is categorical—use add_choices().")
        if feature in IMMUTABLE_FEATURES:
            raise ValueError(f"'{feature}' is immutable and cannot be ranged.")
        if feature not in FEATURES:
            raise KeyError(f"Unknown feature '{feature}'.")
        self._ranges[feature] = (float(low), float(high))

    def add_choices(self, feature: str, choices: List[Any]) -> None:
        if feature not in CATEGORICAL:
            raise ValueError(f"'{feature}' is numeric—use add_range().")
        self._choices[feature] = list(choices)

    def build(self) -> Tuple[List[Tuple[float, float]], List[Dict[str, Any]]]:
        bounds: List[Tuple[float, float]] = []
        encoding: List[Dict[str, Any]] = []

        all_ings = self._ingredient_ctrl.get_all_ingredients()

        for feat in FEATURES:
            if feat in IMMUTABLE_FEATURES:
                val = self._fixed_props[feat]
                encoding.append({"type": "fixed", "value": val})
                bounds.append((val, val))

            elif feat in CATEGORICAL:
                choices = self._choices.get(feat)
                if choices is None:
                    cls = CATEGORICAL[feat]
                    choices = [
                        ing.name for ing in all_ings if isinstance(ing, cls)]
                if not choices:
                    raise ValueError(f"No choices available for '{feat}'.")
                encoding.append({"type": "cat", "choices": choices})
                bounds.append((0.0, float(len(choices) - 1)))

            else:
                low, high = self._ranges.get(
                    feat, (float("-inf"), float("inf")))
                encoding.append({"type": "num"})
                bounds.append((low, high))

        return bounds, encoding


class Optimizer:
    def __init__(self, asset_name: str):
        self.asset_ctrl = AssetController(os.path.join("assets"))
        asset_path = self.asset_ctrl.get_asset_path(
            logical_name=asset_name, extensions=[".zip"]
        )
        self.predictor = Predictor(zip_path=asset_path)
        self.db = Database()
        self.ingredient_ctrl = IngredientController(self.db)

    def optimize(
        self,
        target_profile: ViscosityProfile,
        spec: Constraints,
        max_iter: int = 100,
    ) -> Formulation:

        bounds, encoding = spec.build()

        result = differential_evolution(
            func=self._objective,
            bounds=bounds,
            args=(encoding, target_profile),
            maxiter=max_iter,
            polish=True,
        )

        best_dict = self._decode(result.x, encoding)
        form = Formulation()
        form.from_dict(best_dict)
        return form

    def _decode(
        self,
        x: np.ndarray,
        encoding: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        formulation: Dict[str, Any] = {}
        for xi, feat, meta in zip(x, FEATURES, encoding):
            if meta["type"] == "cat":
                idx = int(round(xi))
                formulation[feat] = meta["choices"][idx]
            else:
                formulation[feat] = float(xi)
        return formulation

    def _objective(
        self,
        x: np.ndarray,
        encoding: List[Dict[str, Any]],
        target_profile: ViscosityProfile,
    ) -> float:
        form_dict = self._decode(x, encoding)
        formulation = Formulation()
        formulation.from_dict(form_dict)
        df = formulation.to_dataframe(encoded=False)
        pred = self.predictor.predict(df)

        if isinstance(pred, ViscosityProfile):
            pred_vals = np.array(pred.viscosities)
        else:
            pred_vals = np.array(pred)

        target_vals = np.array(target_profile.viscosities)
        return float(np.linalg.norm(pred_vals - target_vals))


if __name__ == "__main__":
    opt = Optimizer(asset_name="VisQAI-base")

    vp = ViscosityProfile(
        shear_rates=[100, 1_000, 10_000, 100_000, 15_000_000],
        viscosities=[10, 10, 10, 10, 10],
        units="cP"
    )
    vp.is_measured = False

    fixed_properties = {
        "MW": 300.0,
        "PI_mean": 6.5,
        "PI_range": 1.0,
        "Buffer_pH": 7.0,
    }

    spec = Constraints(opt.ingredient_ctrl, fixed_properties)

    spec.add_range("Protein_conc", 1.0, 100.0)
    spec.add_range("Temperature", 2.0, 25.0)
    spec.add_range("Buffer_conc", 10.0, 200.0)
    spec.add_range("Salt_conc", 0.0, 150.0)
    spec.add_range("Stabilizer_conc", 0.0, 50.0)
    spec.add_range("Surfactant_conc", 0.0, 20.0)
    spec.add_choices("Protein_type", ["BSA"])
    spec.add_choices("Buffer_type", ["Histidine", "PBS", "HEPES"])
    spec.add_choices("Salt_type", ["NaCl"])
    spec.add_choices("Stabilizer_type", ["Glycerol", "Trehalose"])
    spec.add_choices("Surfactant_type", ["Tween20", "Tween80"])

    formulation = opt.optimize(vp, spec, max_iter=100)

    print("Suggested formulation:")
    for feat, val in formulation.items():
        print(f"  {feat}: {val}")
