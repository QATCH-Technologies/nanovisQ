"""
sampler.py

This module provides a framework for exploring the bioformulation design space
by identifying candidates that maximize the information gain of predictive
models. It specifically focuses on "Active Learning" strategies, where
candidates are selected not just for performance, but to challenge and
improve the model's accuracy in unknown regions.

The module is centered around the `Sampler` class, which integrates several
key architectural patterns:

- Active Learning via Acquisition
- Hybrid Search Strategy:
    Candidates are generated through a dual-phase approach:
    - Global Exploration: Uses Monte Carlo sampling across the entire
      constrained space to find broad gaps in knowledge.
    - Local Exploitation: Perturbs existing "interesting" formulations to
      fine-tune results in specific high-value neighborhoods.
- Physical & Practical Constraints

Example:
    >>> sampler = Sampler(asset_name="v3_model", database=db)
    >>> # Suggest the next best formulation to test in the lab
    >>> next_form = sampler.get_next_sample(use_ucb=True, kappa=2.0)
    >>> print(f"Suggested Protein: {next_form.protein.name}")

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    2.1
"""

import os
from typing import Dict, List, Union

import numpy as np

try:
    TAG = "[Sampler (HEADLESS)]"
    from src.controller.formulation_controller import FormulationController
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.managers.asset_manager import AssetError, AssetManager
    from src.models.formulation import Formulation, ViscosityProfile
    from src.models.ingredient import Ingredient
    from src.models.predictor import Predictor
    from src.utils.constraints import Constraints

    class Log:
        @staticmethod
        def d(TAG, msg=""):
            print("DEBUG:", TAG, msg)

        @staticmethod
        def i(TAG, msg=""):
            print("INFO:", TAG, msg)

        @staticmethod
        def w(TAG, msg=""):
            print("WARNING:", TAG, msg)

        @staticmethod
        def e(TAG, msg=""):
            print("ERROR:", TAG, msg)

except (ModuleNotFoundError, ImportError):
    TAG = "[Sampler]"
    from QATCH.VisQAI.src.controller.formulation_controller import FormulationController
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.managers.asset_manager import AssetError, AssetManager
    from QATCH.VisQAI.src.models.formulation import Formulation, ViscosityProfile
    from QATCH.VisQAI.src.models.ingredient import Ingredient
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.utils.constraints import Constraints
    from QATCH.common.logger import Logger as Log


class Sampler:
    """Generates and evaluates bioformulation candidates based on predictive uncertainty.

    This class acts as an active learning engine that explores the formulation
    design space. It leverages a trained `Predictor` to identify regions of high
    uncertainty and utilizes `Constraints` to ensure that generated candidates
    are physically and chemically plausible.

    The sampler integrates database records for "warm-starting" the model and
    uses a internal controller ecosystem to resolve ingredient identities and
    manage model assets.

    Attributes:
        database (Database): Instance for persistent storage connectivity.
        form_ctrl (FormulationController): Interface for managing formulation
            lifecycle and historical data retrieval.
        ing_ctrl (IngredientController): Logic layer for looking up specific
            biochemical properties of ingredients.
        asset_ctrl (AssetManager): Manager for loading and verifying `.visq`
            model packages from the assets directory.
        predictor (Predictor): The inference engine used to estimate viscosity
            and prediction uncertainty.
        constraints (Constraints): Definition of the search space, including
            allowed ingredient types and concentration ranges.
        _bounds (List[Tuple[float, float]]): The finalized numeric search
            boundaries used by sampling algorithms.
        _encoding (List[Dict]): Metadata describing feature types and
            categorical choices for translation.
        _current_uncertainty (np.ndarray): Variance or standard deviation
            arrays from the most recent prediction cycle.
        _current_viscosity (Optional[ViscosityProfile]): The predicted
            viscosity results from the most recent candidate evaluation.
        _last_formulation (Optional[Formulation]): The most recently
            constructed Formulation domain object.
    """

    def __init__(
        self,
        asset_name: str,
        database: Database,
        constraints: Constraints = None,
        seed: int = None,
    ):
        """Initializes the Sampler with model assets and constrained design space.

        The initialization process performs several critical setup steps:
        - Asset Loading
        - Constraint Resolution
        - Physical Capping
        - -Warm Starting

        Args:
            asset_name: The name of the predictive model asset to load.
            database: An active Database connection for record lookups.
            constraints: Optional pre-defined Constraints. If None,
                constraints are derived from database history.
            seed: Optional integer seed for reproducibility in stochastic
                sampling operations.

        Raises:
            AssetError: If the specified `asset_name` cannot be found or is
                incompatible with the required `.visq` format.
        """
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
                        TAG,
                        "Warm starting predictor with contextual database to saturate.",
                    )
                    self.predictor.learn(df_train)
            except Exception as e:
                Log.w(TAG, f"Failed to warm start predictor: {e}")

        if seed is not None:
            np.random.seed(seed)

        self._current_uncertainty = np.array([])
        self._current_viscosity = None
        self._last_formulation = None

    def add_sample(self, formulation: Formulation) -> None:
        """Evaluates a formulation candidate and updates internal prediction state.

        This method takes a structured `Formulation` object, converts it into a
        computational format (DataFrame), and passes it through the `Predictor`.
        It captures both the estimated viscosity and the associated prediction
        uncertainty (standard deviation), updating the sampler's state to
        reflect the most recent evaluation.

        Args:
            formulation: The candidate formulation instance to be evaluated by
                the predictive model.

        Note:
            Uncertainty extraction defaults to the 'std' key from the
            predictor's output dictionary. Ensure the loaded model asset
            supports uncertainty quantification.
        """
        df = formulation.to_dataframe(encoded=False, training=False)
        vis, unc_dict = self.predictor.predict_uncertainty(df)
        unc = unc_dict["std"] if isinstance(unc_dict, dict) else unc_dict
        self._current_viscosity = self._make_viscosity_profile(vis)
        self._current_uncertainty = unc
        self._last_formulation = formulation

    def get_next_sample(self, use_ucb: bool = True, kappa: float = 2.0) -> Formulation:
        """Generates and selects the optimal next candidate formulation using active learning.

        This method identifies the next formulation to evaluate by searching the design
        space through a two-pronged approachof global exploration and local exploitation.

        The selection is guided by an acquisition function—either Upper Confidence
        Bound (UCB), which balances predicted viscosity against uncertainty, or
        pure uncertainty maximization.

        Args:
            use_ucb: If True, uses the Upper Confidence Bound acquisition function
                to balance exploration and exploitation. If False, ranks candidates
                strictly by maximum mean uncertainty. Defaults to True.
            kappa: The trade-off parameter for UCB. A higher kappa encourages
                broader exploration of high-uncertainty regions, while a lower
                kappa favors exploiting known regions. Defaults to 2.0.

        Returns:
            The selected Formulation object with the highest acquisition score,
            or None if no valid candidates could be generated.

        Note:
            The method dynamically adjusts the number of global samples (`n_global`)
            based on the `base_unc` of the previous evaluation. If the model is
            highly confident (low uncertainty), it increases the sample count
            to search for hidden gaps in the design space.
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
            score = self._acquisition_ucb(vis, unc, kappa) if use_ucb else np.nanmean(unc)
            candidates.append((form, score))

        if self._last_formulation is not None:
            for form in self._perturb_formulation(self._last_formulation, base_unc):
                vis, unc_dict = self.predictor.predict_uncertainty(
                    form.to_dataframe(encoded=False, training=False)
                )
                unc = unc_dict["std"] if isinstance(unc_dict, dict) else unc_dict
                score = self._acquisition_ucb(vis, unc, kappa) if use_ucb else np.nanmean(unc)
                candidates.append((form, score))

        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0] if candidates else None

    def _round_suggestion(self, feat: str, val: float, low: float, high: float) -> float:
        """Rounds a numeric suggestion to the nearest physically meaningful increment.

        This method applies domain-specific quantization to candidate values. For
        example, protein and buffer concentrations are rounded to the nearest 5.0 units,
        while low-concentration excipients (Stabilizers/Surfactants) are rounded
        to the nearest 0.05 units.

        The rounding logic includes a "minimum presence" rule: if a value is
        rounded down to 0.0 but the feature's upper bound is greater than zero,
        the method forces the value up to the minimum increment step (e.g., 5.0 or 0.05).
        Finally, the value is clipped to ensure it stays strictly within the
        active search bounds.

        Args:
            feat: The name of the feature being rounded (e.g., 'Salt_conc').
            val: The raw, unrounded numeric suggestion from the sampler.
            low: The lower boundary for the feature.
            high: The upper boundary for the feature.

        Returns:
            float: The quantized and clipped numeric value.

        Note:
            Quantization helps align AI suggestions with practical lab equipment
            limitations (e.g., pipetting resolution or scale precision).
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

    def _enforce_none_concentrations(self, suggestions: Dict[str, Union[str, float]]) -> None:
        """Enforces logical consistency between 'None' ingredients and their concentrations.

        This post-processing step ensures that the formulation remains physically
        valid. If the sampler selects a "None" placeholder for a categorical
        ingredient type (e.g., 'Salt_type' = 'None'), this method identifies
        the corresponding concentration key (e.g., 'Salt_conc') and explicitly
        sets it to 0.0.

        This prevents "ghost ingredients" where a concentration might exist
        without a defined chemical species, which would otherwise lead to
        errors during model inference or database entry.

        Args:
            suggestions: A dictionary of formulated features where keys are
                feature names (types and concentrations) and values are the
                proposed sampling results.
        """
        for key, val in list(suggestions.items()):
            if key.endswith("_type") and str(val).lower() == "none":
                conc_key = key.replace("_type", "_conc")
                suggestions[conc_key] = 0.0

    def _generate_random_samples(self, n: int) -> List[Formulation]:
        """Generates a batch of randomized formulations within established constraints.

        This method performs Monte Carlo sampling across the multidimensional
        design space defined by `_bounds` and `_encoding`. For each candidate, it
        randomly selects categorical ingredients and continuous concentrations.

        To ensure the samples are high-quality and lab-ready, the method applies
        three layers of post-processing to every raw suggestion.

        Args:
            n: The number of random formulation candidates to generate.

        Returns:
            List[Formulation]: A list of length `n` containing fully validated
                and initialized Formulation objects.

        Note:
            This method is used as the 'Exploration' phase of the active learning
            cycle to discover new areas of the design space that may contain
            high predictive uncertainty.
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
        """Creates a set of localized variations of an existing formulation.

        This method acts as a local exploitation strategy. It takes a successful
        or interesting baseline formulation and "jitters" its parameters to
        explore the immediate neighborhood.

        The intensity of the perturbation (the `noise_scale`) is dynamically
        linked to the `base_uncertainty`. If the model is highly uncertain
        about the current point, it takes larger steps to find the "edge" of
        its knowledge; if the model is confident, it takes smaller steps to
        fine-tune the suggestion.

        Args:
            formulation: The baseline Formulation object to perturb.
            base_uncertainty: The current prediction uncertainty for the baseline.
                Used to scale the magnitude of the Gaussian noise.
            max_uncertainty: The normalization constant for the noise scale.
                Defaults to 1.0.
            n: The number of perturbed variations to generate. Defaults to 5.

        Returns:
            List[Formulation]: A list of `n` new Formulation objects that are
                structurally similar to the baseline but with varied
                concentrations and/or ingredient types.

        Note:
            Numerical features are perturbed using a normal distribution centered
            around the original value, while categorical features are
            randomly resampled from their allowed choices.
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

                    choices = enc.get("choices", [])
                    if choices:
                        sug[feat] = np.random.choice(choices)
                    else:
                        sug[feat] = val

            self._enforce_none_concentrations(sug)
            perturbed.append(self._build_formulation(sug))
        return perturbed

    def _make_viscosity_profile(self, viscosities: np.ndarray) -> ViscosityProfile:
        """Constructs a ViscosityProfile object from a set of predicted values.

        This helper method acts as the final step in the prediction pipeline.
        It maps a raw numerical array of viscosities—typically extracted from
        the predictor—back to a structured `ViscosityProfile`.

        The shear rates are fixed to a standard five-point log-scale profile
        ranging from low shear (100 s 1/s) to ultra-high shear (1.5e7 1/s).
        This standardization ensures that all samples generated by the
        Sampler can be compared consistently.

        Args:
            viscosities: A 1D NumPy array containing the predicted viscosity
                values in cP.

        Returns:
            ViscosityProfile: A structured domain object containing the
                paired shear rate and viscosity data.
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
        """Computes the Upper Confidence Bound (UCB) score for a candidate sample.

        The UCB used is:
        Score = mu + kappa x sigma

        Args:
            viscosity: A 1D array of predicted viscosity values corresponding
                to the standard shear rate profile.
            uncertainty: A 1D array of prediction uncertainties (standard
                deviations) for each shear rate.
            kappa: The 'Exploration' coefficient. Higher values prioritize
                sampling in regions where the model is less certain. Defaults to 2.0.
            reference_shear_rate: The specific shear rate (in s 1/s) used to
                extract the mean prediction (mu). Defaults to 10,000.

        Returns:
            float: The calculated UCB score. Higher scores indicate more
                desirable candidates for the next sampling iteration.
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
        """Resolves an ingredient lookup safely by filtering null values and strings.

        This utility provides a robust bridge between raw input (often from
        user interfaces or dictionaries) and formal `Ingredient` domain objects.

        Args:
            val: The value to resolve. Can be a string name, an existing
                `Ingredient` instance, or `None`.
            get_method: A callable (usually a database or controller method)
                that takes a string and returns an `Ingredient` or `None`.

        Returns:
            Optional[Ingredient]: The resolved ingredient object, or `None` if
                the input was empty, "None", or the lookup failed.
        """
        if val is None or str(val).strip().lower() == "none":
            return None
        if isinstance(val, Ingredient):
            return val
        return get_method(val)

    def _build_formulation(self, suggestions: Dict[str, Union[str, float]]) -> Formulation:
        """Assembles a valid Formulation object from a dictionary of sampled features.

        This method acts as the bridge between raw sampling results and the
        formal domain model. It iterates through the standard formulation
        components (Protein, Buffer, Salt, Stabilizer, Surfactant, and Excipient),
        resolving each categorical name into a concrete `Ingredient` object
        via the `IngredientController`.

        Args:
            suggestions: A dictionary containing the proposed ingredient types
                (strings or objects) and their associated concentrations (floats).

        Returns:
            Formulation: A fully populated domain object ready for
                viscosity prediction or database persistent storage.
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
            form.set_stabilizer(stab, float(suggestions.get("Stabilizer_conc", 0.0)), "M")

        surf = self._resolve_ingredient(
            suggestions.get("Surfactant_type"), self.ing_ctrl.get_surfactant_by_name
        )
        if surf:
            form.set_surfactant(surf, float(suggestions.get("Surfactant_conc", 0.0)), "%w")

        excip = self._resolve_ingredient(
            suggestions.get("Excipient_type"), self.ing_ctrl.get_excipient_by_name
        )
        if excip:
            form.set_excipient(excip, float(suggestions.get("Excipient_conc", 0.0)), "mM")

        form.set_temperature(float(suggestions.get("Temperature", 25.0)))
        return form
