from typing import Dict, Any, List, Tuple, Type
import numpy as np
from scipy.optimize import differential_evolution
import math

try:
    from src.models.formulation import ViscosityProfile, Formulation
    from src.models.predictor import Predictor
    from src.models.ingredient import Ingredient
    from src.utils.constraints import Constraints
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.formulation import ViscosityProfile, Formulation
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.models.ingredient import Ingredient
    from QATCH.VisQAI.src.utils.constraints import Constraints


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
