import os
import math
from typing import Dict, Union

import numpy as np
import pandas as pd

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


class Sampler:
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
        self.asset_ctrl = AssetController(
            assets_dir=os.path.join("VisQAI", "assets")
        )
        if not self.asset_ctrl.asset_exists(asset_name, [".zip"]):
            raise AssetError(f"Asset with name `{asset_name}` does not exist.")
        asset_zip_path = self.asset_ctrl.get_asset_path(asset_name, [".zip"])
        self.predictor = Predictor(source=asset_zip_path)

        self._all_data: pd.DataFrame = self.form_ctrl.get_all_as_dataframe(
            encoded=False)

        self._current_uncertainty: np.ndarray = np.array([])
        self._current_viscosity: ViscosityProfile = None

    def add_sample(self, formulation: Formulation) -> None:
        """
        Given a Formulation, run predictor.predict(...) to get both viscosities
        and uncertainty. Store them so get_next_sample() can refer to the latest.
        """
        form_df = formulation.to_dataframe()
        preds: Dict[str, Union[list, Dict[float, float]]
                    ] = self.predictor.predict(form_df)
        viscosity = preds.get("viscosities", {})
        self._current_viscosity = self._make_viscosity_profile(viscosity)
        self._current_uncertainty = np.array(
            preds.get("uncertainty", []), dtype=float)

    def update_model(self, formulation: Formulation):
        pass

    def get_next_sample(self) -> Formulation:

        df = self._all_data
        if df is None or df.empty:
            raise RuntimeError(
                "No historical data available to base suggestions on.")

        present = [col for col in self._VALID_FEATURES if col in df.columns]
        if not present:
            raise RuntimeError(
                "None of the valid features are present in the DataFrame.")

        if self._current_uncertainty.size > 0:
            mean_uncert = float(np.nanmean(self._current_uncertainty))
        else:
            # If we never called add_sample(), force "explore"
            mean_uncert = math.inf

        # TODO: tune this to model's output scale
        uncertainty_threshold = 0.1
        high_uncertainty = (mean_uncert > uncertainty_threshold)

        suggestions: Dict[str, Union[int, float, str]] = {}

        # pick the least‐frequent category
        for col in present:
            if col in self._CATEGORICAL:
                counts = df[col].value_counts(dropna=False)
                least_freq = counts.idxmin()
                suggestions[col] = least_freq

        # find largest gap (if high_uncertainty) or smallest gap (if low)
        for col in present:
            if col in self._NUMERIC:
                vals = df[col].dropna().astype(float).to_numpy()
                if vals.size == 0:
                    # No history default to 0.0
                    suggestions[col] = 0.0
                    continue

                unique_vals = np.unique(vals)
                sorted_vals = np.sort(unique_vals)

                if sorted_vals.size == 1:
                    base = sorted_vals[0]
                    if high_uncertainty:
                        # "Explore" by bumping +10%
                        x = base * 1.10
                    else:
                        # "Exploit" by staying at the same number
                        x = base
                else:
                    diffs = np.diff(sorted_vals)
                    max_gap_idx = int(np.argmax(diffs))
                    gap_left = sorted_vals[max_gap_idx]
                    gap_right = sorted_vals[max_gap_idx + 1]
                    midpoint_max = float((gap_left + gap_right) / 2.0)

                    if high_uncertainty:
                        # Explore pick largest gap’s midpoint
                        x = midpoint_max
                    else:
                        # Exploit pick smallest gap’s midpoint
                        min_gap_idx = int(np.argmin(diffs))
                        small_left = sorted_vals[min_gap_idx]
                        small_right = sorted_vals[min_gap_idx + 1]
                        x = float((small_left + small_right) / 2.0)

                suggestions[col] = x

        # Handle any completely missing valid features
        for col in self._VALID_FEATURES:
            if col not in suggestions:
                if col in self._CATEGORICAL:
                    # If that "type" was never in the DataFrame, assign "Unknown"
                    suggestions[col] = "Unknown"
                else:
                    # If a numeric was never in the DataFrame, pick 0.0
                    suggestions[col] = 0.0

        new_form = Formulation()
        protein_name = suggestions["Protein_type"]
        if protein_name != "Unknown":
            protein_obj: Protein = self.ing_ctrl.get_protein_by_name(
                protein_name
            )
            prot_conc = float(suggestions["Protein_conc"])
            new_form.set_protein(protein_obj, prot_conc, units="mg/mL")

        buffer_name = suggestions["Buffer_type"]
        if buffer_name != "Unknown":
            buffer_obj: Buffer = self.ing_ctrl.get_buffer_by_name(
                buffer_name
            )
            buf_conc = float(suggestions["Buffer_conc"])
            new_form.set_buffer(buffer_obj, buf_conc, units="mg/mL")

        salt_name = suggestions["Salt_type"]
        if salt_name != "Unknown":
            salt_obj: Salt = self.form_ctrl.ingredient_controller.get_salt_by_name(
                salt_name
            )
            salt_conc = float(suggestions["Salt_conc"])
            new_form.set_salt(salt_obj, salt_conc, units="mg/mL")

        stab_name = suggestions["Stabilizer_type"]
        if stab_name != "Unknown":
            stab_obj: Stabilizer = self.form_ctrl.ingredient_controller.get_stabilizer_by_name(
                stab_name
            )
            stab_conc = float(suggestions["Stabilizer_conc"])
            new_form.set_stabilizer(stab_obj, stab_conc, units="M")

        surf_name = suggestions["Surfactant_type"]
        if surf_name != "Unknown":
            surf_obj: Surfactant = self.form_ctrl.ingredient_controller.get_surfactant_by_name(
                surf_name
            )
            surf_conc = float(suggestions["Surfactant_conc"])
            new_form.set_surfactant(surf_obj, surf_conc, units="%w")

        temp_val = float(suggestions["Temperature"])
        new_form.set_temperature(temp_val)
        return new_form

    def _make_viscosity_profile(self, viscosity: dict) -> ViscosityProfile:
        """
        Convert a dict { shear_rate → viscosity_value } into a ViscosityProfile.
        """
        profile = ViscosityProfile()
        for shear_rate, vis in viscosity.items():
            profile.shear_rates.append(float(shear_rate))
            profile.viscosities.append(float(vis))
        return profile
