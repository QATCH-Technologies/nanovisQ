
"""
sampler.py

Module for sampling bioformulation candidates using an uncertainty-based acquisition function.

This module defines the `Sampler` class, which interfaces with a trained predictor model,
applies user-defined or historical constraints to generate candidate formulations,
and selects new samples based on either Upper Confidence Bound (UCB) or direct
uncertainty metrics.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-07-09

Version:
    2.1
"""

import os
from typing import Dict, Union, List
import numpy as np

try:
    from src.db.db import Database
    from src.models.predictor import Predictor
    from src.models.formulation import Formulation, ViscosityProfile
    from src.controller.ingredient_controller import IngredientController
    from src.controller.formulation_controller import FormulationController
    from src.managers.asset_manager import AssetManager, AssetError
    from src.utils.constraints import Constraints
    from src.models.ingredient import Ingredient
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.models.formulation import Formulation, ViscosityProfile
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.controller.formulation_controller import FormulationController
    from QATCH.VisQAI.src.managers.asset_manager import AssetManager, AssetError
    from QATCH.VisQAI.src.utils.constraints import Constraints
    from QATCH.VisQAI.src.models.ingredient import Ingredient


class Sampler:
    """
    Generates and evaluates bioformulation candidates based on predictive uncertainty.

    Attributes:
        database (Database): Database connection instance.
        form_ctrl (FormulationController): Controller for handling formulation records.
        ing_ctrl (IngredientController): Controller for ingredient lookups.
        asset_ctrl (AssetManager): Controller for loading model assets.
        predictor (Predictor): Predictor instance for model inference.
        constraints (Constraints): Constraint definitions for features.
        _bounds (List[Tuple[float, float]]): Numeric bounds for sampling.
        _encoding (List[Dict]): Encoding metadata for each feature.
        _current_uncertainty (np.ndarray): Last predicted uncertainties.
        _current_viscosity (ViscosityProfile): Last predicted viscosity profile.
        _last_formulation (Formulation): Last sampled formulation.
    """

    def __init__(
        self,
        asset_name: str,
        database: Database,
        constraints: Constraints = None,
        seed: int = None
    ):
        """
        Initializes the Sampler with model assets, controllers, and constraints.

        Args:
            asset_name (str): Name of the predictor asset (zip file without extension).
            database (Database): Database instance for controllers.
            constraints (Constraints, optional): Predefined constraints. If None, uses
                historical data to infer numeric ranges. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Raises:
            AssetError: If the specified asset is not found under the assets directory.
        """
        self.database = database
        self.form_ctrl = FormulationController(db=database)
        self.ing_ctrl = IngredientController(db=database)

        # Load model asset
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(
            os.path.join(base_dir, os.pardir, os.pardir))
        assets_dir = os.path.join(project_root, "assets")
        self.asset_ctrl = AssetManager(assets_dir=assets_dir)
        if not self.asset_ctrl.asset_exists(asset_name, ['.zip']):
            raise AssetError(f"Asset `{asset_name}` not found.")
        asset_zip = self.asset_ctrl.get_asset_path(asset_name, ['.zip'])
        self.predictor = Predictor(zip_path=asset_zip)

        # Configure constraints
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
        """
        Adds a new formulation sample and updates the internal state with its predictions.

        Args:
            formulation (Formulation): The candidate formulation to evaluate.
        """
        df = formulation.to_dataframe(encoded=False)
        vis, unc = self.predictor.predict_uncertainty(df)
        self._current_viscosity = self._make_viscosity_profile(vis)
        self._current_uncertainty = unc
        self._last_formulation = formulation

    def get_next_sample(
        self,
        use_ucb: bool = True,
        kappa: float = 2.0
    ) -> Formulation:
        """
        Generates and selects the next candidate formulation.

        Args:
            use_ucb (bool): If True, use the UCB acquisition function; otherwise, rank by mean uncertainty.
            kappa (float): Exploration-exploitation trade-off parameter for UCB.

        Returns:
            Formulation: The next selected formulation, or None if no candidates.
        """
        candidates: List[tuple] = []
        base_unc = np.nanmean(
            self._current_uncertainty) if self._current_uncertainty.size > 0 else float('inf')
        n_global = 20 if base_unc < 0.05 else 5

        # Generate global random candidates
        for form in self._generate_random_samples(n_global):
            vis, unc = self.predictor.predict_uncertainty(
                form.to_dataframe(encoded=False))
            score = self._acquisition_ucb(
                vis, unc, kappa) if use_ucb else np.nanmean(unc)
            candidates.append((form, score))

        # Generate local perturbations around the last formulation
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
        """
        Rounds a suggested numeric value according to feature-specific rules.

        Args:
            feat (str): Feature name (e.g., 'Stabilizer_conc').
            val (float): Raw suggested value.

        Returns:
            float: Rounded value within valid bounds.
        """
        if feat in ('Stabilizer_conc', 'Surfactant_conc'):
            rounded = float(round(min(max(val, 0.0), 1.0) / 0.05) * 0.05)
            return rounded if rounded > 0 else 0.05
        else:
            rounded = float(round(val / 5.0) * 5.0)
            return rounded if rounded > 0 else 5.0

    def _generate_random_samples(self, n: int) -> List[Formulation]:
        """
        Generates random formulations within defined constraints.

        Args:
            n (int): Number of random samples to generate.

        Returns:
            List[Formulation]: List of randomly generated formulations.
        """
        samples: List[Formulation] = []

        for _ in range(n):
            suggestions: Dict[str, Union[str, float]] = {}
            for (low, high), enc in zip(self._bounds, self._encoding):
                feat = enc['feature']
                if enc['type'] == 'cat':
                    choices = enc.get('choices', [])
                    if not choices:
                        raise RuntimeError(
                            f"No choices for categorical feature {feat}")
                    suggestions[feat] = np.random.choice(choices)
                else:
                    ing_type = suggestions.get(
                        feat.replace('_conc', '_type'), None)
                    if str(ing_type).lower() == 'none':
                        suggestions[feat] = 0.0
                    else:
                        # Generate a random float within the bounds
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
        """
        Creates perturbed variants of a given formulation based on uncertainty scale.

        Args:
            formulation (Formulation): Base formulation to perturb.
            base_uncertainty (float): Reference uncertainty for scaling noise.
            max_uncertainty (float): Upper bound for uncertainty scaling. Defaults to 1.0.
            n (int): Number of perturbations to generate. Defaults to 5.

        Returns:
            List[Formulation]: Perturbed formulations.
        """
        noise_scale = min(1.0, base_uncertainty / max_uncertainty) * 0.2
        base_df = formulation.to_dataframe(encoded=False)
        perturbed: List[Formulation] = []

        for _ in range(n):
            sug: Dict[str, Union[str, float]] = {}
            for (low, high), enc in zip(self._bounds, self._encoding):
                feat = enc['feature']
                val = base_df[feat].iloc[0]
                if enc['type'] == 'num':
                    nv = val * (1 + np.random.normal(scale=noise_scale))
                    nv = float(np.clip(nv, low, high))
                    sug[feat] = self._round_suggestion(feat, nv)
                else:
                    sug[feat] = val
            perturbed.append(self._build_formulation(sug))
        return perturbed

    def _make_viscosity_profile(self, viscosities: np.ndarray) -> ViscosityProfile:
        """
        Constructs a ViscosityProfile from predicted viscosity values.

        Args:
            viscosities (np.ndarray): Array of viscosity predictions.

        Returns:
            ViscosityProfile: Profile at predefined shear rates.
        """
        return ViscosityProfile(
            shear_rates=[100, 1000, 10000, 100000, 15000000],
            viscosities=viscosities.flatten().tolist()
        )

    def _acquisition_ucb(
        self,
        viscosity: np.ndarray,
        uncertainty: np.ndarray,
        kappa: float = 2.0,
        reference_shear_rate: float = 10_000
    ) -> float:
        """
        Computes Upper Confidence Bound (UCB) score for acquisition.

        Args:
            viscosity (dict): Mapping of shear rates to viscosity values.
            uncertainty (np.ndarray): Uncertainty estimates from the predictor.
            kappa (float): Exploration–exploitation trade-off parameter.
            reference_shear_rate (float): Shear rate to evaluate mean viscosity at.

        Returns:
            float: UCB score = mu + kappa * sigma.
        """
        try:
            srs = np.array([100, 1_000, 10_000, 100_000, 15_000_000])
            vis = viscosity.flatten()
            idx = np.abs(srs - reference_shear_rate).argmin()
            mu = vis[idx]
        except Exception:
            mu = np.mean(list(viscosity.values()))
        sigma = np.nanmean(uncertainty)
        return mu + kappa * sigma

    def _build_formulation(self, suggestions: Dict[str, Union[str, float]]) -> Formulation:
        """
        Builds a Formulation object from feature suggestions.

        Args:
            suggestions (Dict[str, Union[str, float]]): Feature-value mapping.

        Returns:
            Formulation: Constructed formulation with components set.
        """
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
