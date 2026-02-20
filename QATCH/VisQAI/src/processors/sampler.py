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
    2025-10-22

Version:
    2.4
"""

import os
from typing import Dict, List, Union

import numpy as np

try:
    import logging

    from src.controller.formulation_controller import FormulationController
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.managers.asset_manager import AssetError, AssetManager
    from src.models.formulation import Formulation, ViscosityProfile
    from src.models.ingredient import Ingredient
    from src.models.predictor import Predictor
    from src.utils.constraints import Constraints

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    class Log:
        """Logging utility for standardized log messages."""

        _logger = logging.getLogger("Predictor")

        @classmethod
        def i(cls, msg: str) -> None:
            """Log an informational message."""
            cls._logger.info(msg)

        @classmethod
        def w(cls, msg: str) -> None:
            """Log a warning message."""
            cls._logger.warning(msg)

        @classmethod
        def e(cls, msg: str) -> None:
            """Log an error message."""
            cls._logger.error(msg)

        @classmethod
        def d(cls, msg: str) -> None:
            """Log a debug message."""
            cls._logger.debug(msg)

except (ModuleNotFoundError, ImportError):
    from QATCH.common.logger import Logger as Log
    from QATCH.VisQAI.src.controller.formulation_controller import FormulationController
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.managers.asset_manager import AssetError, AssetManager
    from QATCH.VisQAI.src.models.formulation import Formulation, ViscosityProfile
    from QATCH.VisQAI.src.models.ingredient import Ingredient
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.utils.constraints import Constraints


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
        seed: int = None,
    ):
        self.database = database
        self.form_ctrl = FormulationController(db=database)
        self.ing_ctrl = IngredientController(db=database)

        # Load model asset
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))
        assets_dir = os.path.join(project_root, "assets")
        self.asset_ctrl = AssetManager(assets_dir=assets_dir)
        if not self.asset_ctrl.asset_exists(asset_name, [".visq"]):
            raise AssetError(f"Asset `{asset_name}` not found.")
        asset_zip = self.asset_ctrl.get_asset_path(asset_name, [".visq"])
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

        # Apply absolute constraints/caps
        for i, enc in enumerate(self._encoding):
            if enc["type"] == "num":
                feat = enc["feature"]
                low, high = self._bounds[i]

                if feat == "Protein_conc":
                    high = min(high, 600.0)
                elif feat == "Surfactant_conc":
                    high = min(high, 0.3)
                elif feat == "Stabilizer_conc":
                    high = min(high, 0.5)
                elif feat == "Buffer_conc":
                    high = min(high, 50.0)
                elif feat == "Salt_conc":
                    high = min(high, 150.0)
                elif feat == "Excipient_conc":
                    high = min(high, 500.0)

                # Ensure low does not exceed capped high
                low = min(low, high)
                self._bounds[i] = (low, high)

        # Warm start predictor
        if hasattr(self.predictor, "learn"):
            try:
                df_train = self.form_ctrl.get_all_as_dataframe(encoded=False)
                if df_train is not None and not df_train.empty:
                    Log.i(
                        "Warm starting predictor with historical database to saturate..."
                    )
                    self.predictor.learn(df_train)
            except Exception as e:
                Log.w(f"Failed to warm start predictor: {e}")

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
        df = formulation.to_dataframe(encoded=False, training=False)
        vis, unc_dict = self.predictor.predict_uncertainty(df)
        unc = unc_dict["std"] if isinstance(unc_dict, dict) else unc_dict
        self._current_viscosity = self._make_viscosity_profile(vis)
        self._current_uncertainty = unc
        self._last_formulation = formulation

    def get_next_sample(self, use_ucb: bool = True, kappa: float = 2.0) -> Formulation:
        """
        Generates and selects the next candidate formulation.

        Args:
            use_ucb (bool): If True, use the UCB acquisition function; otherwise, rank by mean uncertainty.
            kappa (float): Exploration-exploitation trade-off parameter for UCB.

        Returns:
            Formulation: The next selected formulation, or None if no candidates.
        """
        candidates: List[tuple] = []
        base_unc = (
            np.nanmean(self._current_uncertainty)
            if self._current_uncertainty.size > 0
            else float("inf")
        )
        n_global = 20 if base_unc < 0.05 else 5

        for form in self._generate_random_samples(n_global):
            vis, unc_dict = self.predictor.predict_uncertainty(
                form.to_dataframe(encoded=False, training=False)
            )
            unc = unc_dict["std"] if isinstance(unc_dict, dict) else unc_dict
            score = (
                self._acquisition_ucb(vis, unc, kappa) if use_ucb else np.nanmean(unc)
            )
            candidates.append((form, score))

        if self._last_formulation is not None:
            for form in self._perturb_formulation(self._last_formulation, base_unc):
                vis, unc_dict = self.predictor.predict_uncertainty(
                    form.to_dataframe(encoded=False, training=False)
                )
                unc = unc_dict["std"] if isinstance(unc_dict, dict) else unc_dict
                score = (
                    self._acquisition_ucb(vis, unc, kappa)
                    if use_ucb
                    else np.nanmean(unc)
                )
                candidates.append((form, score))

        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0] if candidates else None

    def _round_suggestion(
        self, feat: str, val: float, low: float, high: float
    ) -> float:
        """
        Rounds a suggested numeric value according to feature-specific rules and bounds.
        """
        if feat in ("Stabilizer_conc", "Surfactant_conc"):
            rounded = float(round(max(val, 0.0) / 0.05) * 0.05)
            min_step = 0.05
        else:
            rounded = float(round(max(val, 0.0) / 5.0) * 5.0)
            min_step = 5.0

        if rounded <= 0.0 and high > 0.0:
            rounded = min_step

        return float(np.clip(rounded, low, high))

    def _enforce_none_concentrations(
        self, suggestions: Dict[str, Union[str, float]]
    ) -> None:
        """
        Post-processing step that ensures if a categorical feature type is selected as 'None',
        its paired concentration feature is explicitly zeroed out.
        """
        for key, val in list(suggestions.items()):
            if key.endswith("_type") and str(val).lower() == "none":
                conc_key = key.replace("_type", "_conc")
                suggestions[conc_key] = 0.0

    def _generate_random_samples(self, n: int) -> List[Formulation]:
        """
        Generates random formulations within defined constraints.
        """
        samples: List[Formulation] = []

        for _ in range(n):
            suggestions: Dict[str, Union[str, float]] = {}
            for (low, high), enc in zip(self._bounds, self._encoding):
                feat = enc["feature"]
                if enc["type"] == "cat":
                    choices = enc.get("choices", [])
                    if not choices:
                        suggestions[feat] = "None"
                    else:
                        suggestions[feat] = np.random.choice(choices)
                else:
                    raw = float(np.random.uniform(low, high))
                    suggestions[feat] = self._round_suggestion(feat, raw, low, high)

            # Guarantee that 'None' selections dictate a 0.0 concentration
            self._enforce_none_concentrations(suggestions)

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
        """
        noise_scale = min(1.0, base_uncertainty / max_uncertainty) * 0.2
        base_df = formulation.to_dataframe(encoded=False, training=False)
        perturbed: List[Formulation] = []

        for _ in range(n):
            sug: Dict[str, Union[str, float]] = {}
            for (low, high), enc in zip(self._bounds, self._encoding):
                feat = enc["feature"]
                val = base_df[feat].iloc[0]
                if enc["type"] == "num":
                    nv = val * (1 + np.random.normal(scale=noise_scale))
                    nv = float(np.clip(nv, low, high))
                    sug[feat] = self._round_suggestion(feat, nv, low, high)
                else:
                    sug[feat] = val

            # Guarantee that 'None' selections dictate a 0.0 concentration
            self._enforce_none_concentrations(sug)

            perturbed.append(self._build_formulation(sug))
        return perturbed

    def _make_viscosity_profile(self, viscosities: np.ndarray) -> ViscosityProfile:
        """
        Constructs a ViscosityProfile from predicted viscosity values.
        """
        return ViscosityProfile(
            shear_rates=[100, 1000, 10000, 100000, 15000000],
            viscosities=viscosities.flatten().tolist(),
        )

    def _acquisition_ucb(
        self,
        viscosity: np.ndarray,
        uncertainty: np.ndarray,
        kappa: float = 2.0,
        reference_shear_rate: float = 10_000,
    ) -> float:
        """
        Computes Upper Confidence Bound (UCB) score for acquisition.
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

    def _resolve_ingredient(
        self, val: Union[str, Ingredient, None], get_method
    ) -> Union[Ingredient, None]:
        """Resolves an ingredient lookup safely and skips 'None' text matches."""
        if val is None or str(val).strip().lower() == "none":
            return None
        if isinstance(val, Ingredient):
            return val
        return get_method(val)

    def _build_formulation(
        self, suggestions: Dict[str, Union[str, float]]
    ) -> Formulation:
        """
        Builds a Formulation object from feature suggestions safely.
        """
        form = Formulation()

        prot = self._resolve_ingredient(
            suggestions.get("Protein_type"), self.ing_ctrl.get_protein_by_name
        )
        if prot:
            form.set_protein(prot, float(suggestions.get("Protein_conc", 0.0)), "mg/mL")

        buff = self._resolve_ingredient(
            suggestions.get("Buffer_type"), self.ing_ctrl.get_buffer_by_name
        )
        if buff:
            form.set_buffer(buff, float(suggestions.get("Buffer_conc", 0.0)), "mM")

        salt = self._resolve_ingredient(
            suggestions.get("Salt_type"), self.ing_ctrl.get_salt_by_name
        )
        if salt:
            form.set_salt(salt, float(suggestions.get("Salt_conc", 0.0)), "mM")

        stab = self._resolve_ingredient(
            suggestions.get("Stabilizer_type"), self.ing_ctrl.get_stabilizer_by_name
        )
        if stab:
            form.set_stabilizer(
                stab, float(suggestions.get("Stabilizer_conc", 0.0)), "M"
            )

        surf = self._resolve_ingredient(
            suggestions.get("Surfactant_type"), self.ing_ctrl.get_surfactant_by_name
        )
        if surf:
            form.set_surfactant(
                surf, float(suggestions.get("Surfactant_conc", 0.0)), "%w"
            )

        excip = self._resolve_ingredient(
            suggestions.get("Excipient_type"), self.ing_ctrl.get_excipient_by_name
        )
        if excip:
            form.set_excipient(
                excip, float(suggestions.get("Excipient_conc", 0.0)), "mM"
            )

        form.set_temperature(float(suggestions.get("Temperature", 25.0)))
        return form
