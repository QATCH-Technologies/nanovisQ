import os
from typing import Dict, Union, List
import numpy as np

try:
    from src.db.db import Database
    from src.models.predictor import Predictor
    from src.models.formulation import Formulation, ViscosityProfile
    from src.controller.ingredient_controller import IngredientController
    from src.controller.formulation_controller import FormulationController
    from src.controller.asset_controller import AssetController, AssetError
    from src.utils.constraints import Constraints
    from src.models.ingredient import Ingredient
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.models.formulation import Formulation, ViscosityProfile
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.controller.formulation_controller import FormulationController
    from QATCH.VisQAI.src.controller.asset_controller import AssetController, AssetError
    from QATCH.VisQAI.src.utils.constraints import Constraints
    from QATCH.VisQAI.src.models.ingredient import Ingredient


class Sampler:
    def __init__(
        self,
        asset_name: str,
        database: Database,
        constraints: Constraints = None,
        seed: int = None
    ):
        self.database = database
        self.form_ctrl = FormulationController(db=database)
        self.ing_ctrl = IngredientController(db=database)

        # load model asset
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(
            os.path.join(base_dir, os.pardir, os.pardir))
        assets_dir = os.path.join(project_root, "assets")
        self.asset_ctrl = AssetController(assets_dir=assets_dir)
        if not self.asset_ctrl.asset_exists(asset_name, [".zip"]):
            raise AssetError(f"Asset `{asset_name}` not found.")
        asset_zip = self.asset_ctrl.get_asset_path(asset_name, [".zip"])
        self.predictor = Predictor(zip_path=asset_zip)

        # configure constraints
        if constraints is None:
            self.constraints = Constraints(db=database)
            hist = self.form_ctrl.get_all_as_dataframe(encoded=False)
            for feat in self.constraints._NUMERIC:
                if feat in hist.columns:
                    lo, hi = hist[feat].min(), hist[feat].max()
                    self.constraints.add_range(feat, float(lo), float(hi))
        else:
            self.constraints = constraints

        self._bounds, self._encoding = self.constraints.build()

        if seed is not None:
            np.random.seed(seed)

        self._current_uncertainty = np.array([])
        self._current_viscosity = None
        self._last_formulation = None

    def add_sample(self, formulation: Formulation) -> None:
        df = formulation.to_dataframe(encoded=False)
        vis, unc = self.predictor.predict_uncertainty(df)
        self._current_viscosity = self._make_viscosity_profile(vis)
        self._current_uncertainty = unc
        self._last_formulation = formulation

    def get_next_sample(self, use_ucb: bool = True, kappa: float = 2.0) -> Formulation:
        candidates: List[tuple] = []
        base_unc = np.nanmean(
            self._current_uncertainty) if self._current_uncertainty.size > 0 else float("inf")
        n_global = 20 if base_unc < 0.05 else 5

        for form in self._generate_random_samples(n_global):
            vis, unc = self.predictor.predict_uncertainty(
                form.to_dataframe(encoded=False))
            score = self._acquisition_ucb(
                vis, unc, kappa) if use_ucb else np.nanmean(unc)
            candidates.append((form, score))

        if self._last_formulation is not None:
            for form in self._perturb_formulation(self._last_formulation, base_unc):
                vis, unc = self.predictor.predict_uncertainty(
                    form.to_dataframe(encoded=False))
                score = self._acquisition_ucb(
                    vis, unc, kappa) if use_ucb else np.nanmean(unc)
                candidates.append((form, score))

        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0] if candidates else None

    def _round_suggestion(self, feat: str, val: float) -> float:
        if feat in ("Stabilizer_conc", "Surfactant_conc"):
            return float(round(min(max(val, 0.0), 1.0) / 0.05) * 0.05)
        else:
            return float(round(val / 5.0) * 5.0)

    def _generate_random_samples(self, n: int) -> List[Formulation]:
        samples: List[Formulation] = []

        for _ in range(n):
            suggestions: Dict[str, Union[str, float]] = {}
            for (low, high), enc in zip(self._bounds, self._encoding):
                feat = enc["feature"]
                if enc["type"] == "cat":
                    choices = enc.get("choices", [])
                    if not choices:
                        raise RuntimeError(
                            f"No choices for categorical feature {feat}")
                    suggestions[feat] = np.random.choice(choices)
                else:
                    raw = float(np.random.uniform(low, high))
                    suggestions[feat] = self._round_suggestion(feat, raw)

            samples.append(self._build_formulation(suggestions))
        return samples

    def _perturb_formulation(
        self,
        formulation: Formulation,
        base_uncertainty: float,
        max_uncertainty: float = 1.0,
        n: int = 5,
    ) -> List[Formulation]:
        noise_scale = min(1.0, base_uncertainty / max_uncertainty) * 0.2
        base_df = formulation.to_dataframe(encoded=False)
        perturbed: List[Formulation] = []

        for _ in range(n):
            sug: Dict[str, Union[str, float]] = {}
            for (low, high), enc in zip(self._bounds, self._encoding):
                feat = enc["feature"]
                val = base_df[feat].iloc[0]
                if enc["type"] == "num":
                    nv = val * (1 + np.random.normal(scale=noise_scale))
                    nv = float(np.clip(nv, low, high))
                    sug[feat] = self._round_suggestion(feat, nv)
                else:
                    sug[feat] = val
            perturbed.append(self._build_formulation(sug))
        return perturbed

    def _make_viscosity_profile(self, viscosities: np.ndarray) -> ViscosityProfile:
        return ViscosityProfile(
            shear_rates=[100, 1000, 10000, 100000, 15000000],
            viscosities=viscosities.flatten().tolist()
        )

    def _acquisition_ucb(
        self,
        viscosity: dict,
        uncertainty: np.ndarray,
        kappa: float = 2.0,
        reference_shear_rate: float = 10.0
    ) -> float:
        try:
            srs = np.array([float(sr) for sr in viscosity.keys()])
            vis = np.array([float(v) for v in viscosity.values()])
            idx = np.abs(srs - reference_shear_rate).argmin()
            mu = vis[idx]
        except Exception:
            mu = np.mean(viscosity)
        sigma = np.nanmean(uncertainty)
        return mu + kappa * sigma

    def _build_formulation(self, suggestions: Dict[str, Union[str, float]]) -> Formulation:
        form = Formulation()

        if (val := suggestions.get("Protein_type")):
            prot = val if isinstance(
                val, Ingredient) else self.ing_ctrl.get_protein_by_name(val)
            form.set_protein(prot,
                             float(suggestions.get("Protein_conc", 0.0)),
                             "mg/mL")

        if (val := suggestions.get("Buffer_type")):
            buff = val if isinstance(
                val, Ingredient) else self.ing_ctrl.get_buffer_by_name(val)
            form.set_buffer(buff,
                            float(suggestions.get("Buffer_conc", 0.0)),
                            "mM")

        if (val := suggestions.get("Salt_type")):
            salt = val if isinstance(
                val, Ingredient) else self.ing_ctrl.get_salt_by_name(val)
            form.set_salt(salt,
                          float(suggestions.get("Salt_conc", 0.0)),
                          "mM")

        if (val := suggestions.get("Stabilizer_type")):
            stab = val if isinstance(
                val, Ingredient) else self.ing_ctrl.get_stabilizer_by_name(val)
            form.set_stabilizer(stab,
                                float(suggestions.get(
                                    "Stabilizer_conc", 0.0)),
                                "M")

        if (val := suggestions.get("Surfactant_type")):
            surf = val if isinstance(
                val, Ingredient) else self.ing_ctrl.get_surfactant_by_name(val)
            form.set_surfactant(surf,
                                float(suggestions.get(
                                    "Surfactant_conc", 0.0)),
                                "%w")

        form.set_temperature(float(suggestions.get("Temperature", 25.0)))
        return form
