import sys
import os
import math
from typing import Dict, Union

import numpy as np
import pandas as pd

try:
    from src.db.db import Database
    from src.models.predictor import Predictor
    from src.models.formulation import (
        Formulation,
        ViscosityProfile,
        Protein,
        Buffer,
        Salt,
        Stabilizer,
        Surfactant,
    )
    from src.controller.ingredient_controller import IngredientController
    from src.controller.formulation_controller import FormulationController
    from src.controller.asset_controller import AssetController, AssetError
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.models.formulation import (
        Formulation,
        ViscosityProfile,
        Protein,
        Buffer,
        Salt,
        Stabilizer,
        Surfactant,
    )
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.controller.formulation_controller import FormulationController
    from QATCH.VisQAI.src.controller.asset_controller import AssetController, AssetError


class Sampler:
    _SPECIAL_NUMERIC_RANGES = {
        "Stabilizer_conc": (0.1, 1.0),
        "Surfactant_conc": (0.0, 0.1),
    }

    _SPECIAL_NUMERIC_STEPS = {
        "Stabilizer_conc": 0.1,
        "Surfactant_conc": 0.05,
    }

    _DEFAULT_NUMERIC_STEP = 5.0
    _DEFAULT_DECIMALS = 0
    _VALID_FEATURES = [
        "Protein_type",
        "MW",
        "PI_mean",
        "PI_range",
        "Protein_conc",
        "Temperature",
        "Buffer_type",
        "Buffer_pH",
        "Buffer_conc",
        "Salt_type",
        "Salt_conc",
        "Stabilizer_type",
        "Stabilizer_conc",
        "Surfactant_type",
        "Surfactant_conc",
    ]
    _CATEGORICAL = {
        "Protein_type",
        "Buffer_type",
        "Salt_type",
        "Stabilizer_type",
        "Surfactant_type",
    }
    _NUMERIC = set(_VALID_FEATURES) - _CATEGORICAL

    def __init__(self, asset_name: str, database: Database):
        self.database = database
        self.form_ctrl = FormulationController(db=database)
        self.ing_ctrl = IngredientController(db=database)
        base_dir = os.path.dirname(os.path.abspath(
            __file__))
        project_root = os.path.abspath(
            os.path.join(base_dir, os.pardir, os.pardir))
        assets_dir = os.path.join(project_root, "assets")
        self.asset_ctrl = AssetController(assets_dir=assets_dir)
        if not self.asset_ctrl.asset_exists(asset_name, [".zip"]):
            raise AssetError(f"Asset with name `{asset_name}` does not exist.")
        asset_zip_path = self.asset_ctrl.get_asset_path(asset_name, [".zip"])
        self.predictor = Predictor(zip_path=asset_zip_path)

        self._all_data: pd.DataFrame = self.form_ctrl.get_all_as_dataframe(
            encoded=False)

        self._current_uncertainty: np.ndarray = np.array([])
        self._current_viscosity: ViscosityProfile = None

    def add_sample(self, formulation: Formulation) -> None:
        form_df = formulation.to_dataframe()
        viscosity, uncertainty = self.predictor.predict_uncertainty(
            form_df)
        self._current_viscosity = self._make_viscosity_profile(viscosity)
        self._current_uncertainty = uncertainty
        self._last_formulation = formulation

    def get_next_sample(self, use_ucb: bool = True, kappa: float = 2.0) -> Formulation:
        candidates = []

       # 1. Global exploration
        current_unc = np.nanmean(
            self._current_uncertainty) if self._current_uncertainty.size > 0 else float("inf")
        n_global = 20 if current_unc < 0.05 else 5

        global_samples = self._generate_random_samples(n=n_global)
        for form in global_samples:
            viscosity, unc = self.predictor.predict_uncertainty(
                form.to_dataframe())
            score = self._acquisition_ucb(
                viscosity, unc, kappa) if use_ucb else np.nanmean(unc)
            candidates.append((form, score))

        # 2. Local (if applicable)
        if hasattr(self, "_last_formulation") and self._last_formulation is not None:
            local_samples = self._perturb_formulation(
                self._last_formulation, base_uncertainty=current_unc)
            for form in local_samples:
                viscosity, unc = self.predictor.predict_uncertainty(
                    form.to_dataframe())
                score = self._acquisition_ucb(
                    viscosity, unc, kappa) if use_ucb else np.nanmean(unc)
                candidates.append((form, score))

        # 3. Select best
        candidates.sort(key=lambda tup: -tup[1])  # descending by score
        return candidates[0][0] if candidates else None

    def _generate_random_samples(self, n: int = 20) -> list[Formulation]:
        samples = []
        cat_choices = {
            col: self._all_data[col].dropna().unique().tolist()
            for col in self._CATEGORICAL
            if col in self._all_data.columns
        }
        num_ranges = {
            col: (
                self._all_data[col].min(),
                self._all_data[col].max()
            )
            for col in self._NUMERIC
            if col in self._all_data.columns
        }

        for _ in range(n):
            suggestions = {}
            for col in self._VALID_FEATURES:
                if col in self._CATEGORICAL and cat_choices.get(col):
                    suggestions[col] = np.random.choice(cat_choices[col])

                elif col in self._NUMERIC:
                    if col in self._SPECIAL_NUMERIC_RANGES:
                        low, high = self._SPECIAL_NUMERIC_RANGES[col]
                    elif num_ranges.get(col):
                        low, high = num_ranges[col]
                    else:
                        low, high = 0.0, 0.0

                    raw = np.random.uniform(low, high)
                    clamped = float(np.clip(raw, low, high))
                    if col in self._SPECIAL_NUMERIC_STEPS:
                        step = self._SPECIAL_NUMERIC_STEPS[col]
                        decimals = 1
                    else:
                        step = self._DEFAULT_NUMERIC_STEP
                        decimals = self._DEFAULT_DECIMALS

                    stepped = round(clamped/step) * step
                    value = round(stepped, decimals)

                    suggestions[col] = float(
                        f"{value:.{decimals}f}") if decimals else int(value)
                else:
                    suggestions[col] = "Unknown"
            samples.append(self._build_formulation(suggestions))
        return samples

    def _perturb_formulation(
        self,
        formulation: Formulation,
        base_uncertainty: float,
        max_uncertainty: float = 1.0,
        n: int = 5,
    ) -> list[Formulation]:
        """
        Perturbs numeric values based on how uncertain the last sample was.
        More uncertain -> larger noise scale.

        Args:
            formulation: The base formulation to perturb.
            base_uncertainty: The mean uncertainty of the last prediction.
            max_uncertainty: The assumed maximum model uncertainty.
            n: Number of perturbed samples to generate.
        """
        noise_scale = min(1.0, base_uncertainty / max_uncertainty) * \
            0.2  # scale between 0–20%
        base_df = formulation.to_dataframe()
        perturbed = []

        for _ in range(n):
            perturbed_features = {}
            for col in self._VALID_FEATURES:
                val = base_df[col].values[0]
                if col in self._NUMERIC:
                    perturb = np.random.normal(loc=0.0, scale=noise_scale)
                    new_val = val * (1.0 + perturb)

                    if col in self._SPECIAL_NUMERIC_RANGES:
                        low, high = self._SPECIAL_NUMERIC_RANGES[col]
                        new_val = float(np.clip(new_val, low, high))

                    if col in self._SPECIAL_NUMERIC_STEPS:
                        step = self._SPECIAL_NUMERIC_STEPS[col]
                        decimals = 1
                    else:
                        step = self._DEFAULT_NUMERIC_STEP
                        decimals = self._DEFAULT_DECIMALS

                    stepped = round(new_val/step) * step
                    rounded = round(stepped, decimals)
                    perturbed_features[col] = float(
                        f"{rounded:.{decimals}f}") if decimals else int(rounded)
                else:
                    perturbed_features[col] = val
            perturbed.append(self._build_formulation(perturbed_features))
        return perturbed

    def _make_viscosity_profile(self, viscosity: dict) -> ViscosityProfile:
        """
        Convert a dict { shear_rate -> viscosity_value } into a ViscosityProfile.
        """
        shear_rates = []
        viscosities = []
        for shear_rate, vis in viscosity.items():
            shear_rates.append(float(shear_rate))
            viscosities.append(float(vis))
        profile = ViscosityProfile(
            shear_rates=shear_rates, viscosities=viscosities)
        return profile

    def _acquisition_ucb(
        self,
        viscosity: dict,
        uncertainty: np.ndarray,
        kappa: float = 2.0,
        reference_shear_rate: float = 10.0
    ) -> float:
        """
        Compute the UCB score for a formulation.

        Args:
            viscosity: Dict[shear_rate] -> viscosity.
            uncertainty: np.ndarray of uncertainties.
            kappa: Exploration-exploitation trade-off.
            reference_shear_rate: Target rate to evaluate viscosity at.

        Returns:
            UCB score (higher is better).
        """
        try:
            shear_rates = np.array([float(sr) for sr in viscosity.keys()])
            viscosities = np.array([float(v) for v in viscosity])
            idx = np.abs(shear_rates - reference_shear_rate).argmin()
            mu = viscosities[idx]
        except Exception:
            mu = np.mean(list(viscosity))

        sigma = np.nanmean(uncertainty)
        return mu + kappa * sigma

    def _build_formulation(self, suggestions: Dict[str, Union[str, float]]) -> Formulation:
        form = Formulation()
        try:
            if (name := suggestions.get("Protein_type")) != "Unknown":
                form.set_protein(self.ing_ctrl.get_protein_by_name(
                    name), float(suggestions["Protein_conc"]), "mg/mL")
            if (name := suggestions.get("Buffer_type")) != "Unknown":
                form.set_buffer(self.ing_ctrl.get_buffer_by_name(
                    name), float(suggestions["Buffer_conc"]), "mM")
            if (name := suggestions.get("Salt_type")) != "Unknown":
                form.set_salt(self.ing_ctrl.get_salt_by_name(name),
                              float(suggestions["Salt_conc"]), "mM")
            if (name := suggestions.get("Stabilizer_type")) != "Unknown":
                form.set_stabilizer(self.ing_ctrl.get_stabilizer_by_name(
                    name), float(suggestions["Stabilizer_conc"]), "M")
            if (name := suggestions.get("Surfactant_type")) != "Unknown":
                form.set_surfactant(self.ing_ctrl.get_surfactant_by_name(
                    name), float(suggestions["Surfactant_conc"]), "%w")
            form.set_temperature(float(suggestions.get("Temperature", 25.0)))
        except Exception as e:
            raise RuntimeError(f"Failed to build formulation: {e}")
        return form
