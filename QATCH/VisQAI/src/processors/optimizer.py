"""
optimizer.py

This module provides a specialized framework for solving inverse-formulation
problems in biopharmaceutical development. Specifically, it uses Differential
Evolution (DE) to identify ingredient combinations and concentrations that
yield a target rheological profile.

The module is organized into three primary components:

- Telemetry & Progress (OptimizationStatus & OptimizationProgressTracker):
    Captures and accumulates per-generation metrics including best fitness,
    stagnation counts, and improvement rates. This allows for real-time
    monitoring of convergence and supports early-stopping hooks for
    high-throughput optimization tasks.

- State Mapping (Encoding & Decoding):
    Handles the transformation between the optimizer's continuous search
    vectors and the discrete, structured domain of 'Formulation' objects.
    It manages categorical ingredient alignment with backend databases and
    enforces physical concentration constraints.

- The Optimization Engine (Optimizer):
    A hybrid global-to-local search engine. It uses Scrambled Latin Hypercube
    Sampling (LHS) for initial population seeding, Differential Evolution for
    global exploration of categorical and numerical spaces, and L-BFGS-B
    polishing for high-precision refinement of continuous concentration values.


Example:
    >>> optimizer = Optimizer(constraints, predictor, target_profile)
    >>> tracker = OptimizationProgressTracker()
    >>> best_formulation = optimizer.optimize(progress_callback=tracker)
    >>> print(f"Optimized with improvement rate: {tracker.get_improvement_rate()}")

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    2.1
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

try:
    TAG = "[Optimizer (HEADLESS)]"
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
    TAG = "[Optimizer]"
    from QATCH.VisQAI.src.models.formulation import Formulation, ViscosityProfile
    from QATCH.VisQAI.src.models.ingredient import Ingredient
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.utils.constraints import Constraints


_PRED_SHEAR_RATES: List[float] = [100.0, 1_000.0, 10_000.0, 100_000.0, 15_000_000.0]
_LOG_PRED_SHEAR = np.log10(_PRED_SHEAR_RATES)

_EPS = 1e-9
_LARGE_LOSS = 1e6


class OptimizationStatus:
    """Snapshot of optimizer state emitted once per Differential Evolution (DE) generation.

    This data class encapsulates the metrics of a specific point in the
    optimization process, providing insights into the current best fitness value,
    population characteristics, and numerical convergence. It is intended to
    be used for logging, UI progress updates, or early-stopping telemetry.

    Attributes:
        iteration (int): The current generation index.
        num_iterations (int): The maximum number of generations planned for
            the optimization run.
        best_value (float): The objective function value of the best-performing
            individual in the current population.
        population_size (int): The total number of individuals in the population.
        convergence (float): A measure of the population's stability or
            homogeneity (e.g., standard deviation of fitness or parameter spread).
        progress_percent (float): The linear completion percentage of the
            optimization run based on iterations.
    """

    def __init__(
        self,
        iteration: int,
        num_iterations: int,
        best_value: float,
        population_size: int,
        convergence: float,
    ) -> None:
        """Initializes the optimization status snapshot.

        Args:
            iteration: The current iteration/generation number.
            num_iterations: Total expected iterations.
            best_value: Best objective value found in this generation.
            population_size: Number of candidate solutions in the current population.
            convergence: Numerical indicator of how close the population is to
                settling on a solution.
        """
        self.iteration = iteration
        self.num_iterations = num_iterations
        self.best_value = best_value
        self.population_size = population_size
        self.convergence = convergence
        self.progress_percent = (
            iteration / num_iterations * 100 if num_iterations > 0 else 0.0
        )

    def __repr__(self) -> str:
        """Returns a string representation of the optimization status."""
        return (
            f"OptimizationStatus(iteration={self.iteration}, "
            f"best_value={self.best_value:.6f}, "
            f"progress={self.progress_percent:.1f}%)"
        )


class OptimizationProgressTracker:
    """Accumulates per-iteration status objects and monitors optimizer performance.

    This class serves as a central repository for the telemetry data emitted
    during an optimization run. It stores a chronological history of
    `OptimizationStatus` snapshots and maintains flattened lists of key metrics
    (iterations and best values) for rapid access during logging or visualization.

    Attributes:
        history (List[OptimizationStatus]): A chronological list of all status
            snapshots received during the optimization.
        iterations (List[int]): A flattened list of generation indices,
            useful for plotting the x-axis of convergence curves.
        best_values (List[float]): A flattened list of the best objective
            function values found at each iteration.
        start_value (Optional[float]): The best objective value found in the
            initial generation (iteration 0).
        best_value (Optional[float]): The most recent (and presumably lowest/highest)
            best objective value recorded.
    """

    def __init__(self) -> None:
        """Initializes an empty tracker with initialized history containers."""
        self.history: List[OptimizationStatus] = []
        self.iterations: List[int] = []
        self.best_values: List[float] = []
        self.start_value: Optional[float] = None
        self.best_value: Optional[float] = None

    def __call__(self, status: OptimizationStatus) -> None:
        """Updates the tracker with the latest optimization snapshot and logs progress.

        This method acts as the primary callback handler. It appends the current
        status to the internal history, updates the running best value, and
        calculates the total improvement since the initial generation. Finally,
        it formats and emits a structured log message to the system log.

        Args:
            status: An OptimizationStatus object containing the telemetry
                data for the current generation.

        Note:
            The improvement is calculated as a positive delta representing the
            reduction in the objective function value from the `start_value`.
        """
        self.history.append(status)
        self.iterations.append(status.iteration)
        self.best_values.append(status.best_value)

        if self.start_value is None:
            self.start_value = status.best_value
        if self.best_value is None or status.best_value < self.best_value:
            self.best_value = status.best_value

        improvement = (
            (self.start_value - status.best_value) if self.start_value else 0.0
        )
        Log.i(
            TAG,
            f"[{status.iteration:3d}/{status.num_iterations}] "
            f"Best: {status.best_value:10.6f} | "
            f"Improvement: {improvement:10.6f} | "
            f"Progress: {status.progress_percent:6.1f}%",
        )

    def get_improvement_rate(self) -> float:
        """Calculates the average reduction in the objective value per generation.

        This metric represents the 'velocity' of the optimization process by
        measuring the total improvement from the initial generation to the
        most recent generation, normalized by the total number of recorded
        iterations. It is a useful indicator for assessing the efficiency
        of the current optimization parameters.

        Returns:
            float: The average improvement per generation. Returns 0.0 if
                fewer than two snapshots have been recorded or if the
                initial value is unavailable.

        Note:
            The calculation is based on the difference between the
            `start_value` and the current `best_value`, divided by the
            total count of snapshots in the history.
        """
        if len(self.history) < 2 or self.start_value is None:
            return 0.0
        return (self.start_value - self.best_value) / len(self.history)

    def get_stagnation_iterations(self, threshold: int = 10) -> int:
        """Counts the number of consecutive recent generations without improvement.

        This method examines the tail of the optimization history to determine
        how many generations the 'best_value' has remained unchanged. It is
        primarily used for implementing early-stopping criteria or adjusting
        optimizer hyperparameters (like mutation rate) when a plateau is detected.

        Args:
            threshold: The maximum number of recent generations to inspect.
                The search will not exceed this depth. Defaults to 10.

        Returns:
            int: The count of consecutive generations (starting from the most
                recent) where the best value is identical to the current
                minimum. Returns 0 if the history is shorter than the
                specified threshold.

        Note:
            The count is determined by iterating backwards from the latest
            snapshot and incrementing until a superior (lower) best value
            is encountered.
        """
        if len(self.history) < threshold:
            return 0
        recent = self.history[-threshold:]
        best_recent = min(s.best_value for s in recent)
        count = 0
        for s in reversed(recent):
            if s.best_value == best_recent:
                count += 1
            else:
                break
        return count

    def get_plot_data(self) -> Tuple[List[int], List[float]]:
        """Retrieves a copy of the iteration and best value history for visualization.

        This method provides the raw numerical data required to generate
        convergence curves or progress plots. By returning copies of the
        internal lists, it ensures that external plotting libraries or
        manipulations do not inadvertently modify the tracker's primary
        historical record.

        Returns:
            Tuple[List[int], List[float]]: A tuple containing two lists:
                - The first list contains the generation indices (iterations).
                - The second list contains the best objective function values
                  recorded at each of those iterations.
        """
        return self.iterations.copy(), self.best_values.copy()


class Optimizer:
    """
    Mixed discrete-continuous formulation optimizer.

    Parameters
    ----------
    constraints : Constraints
        Defines allowable ingredient choices and numeric ranges.
    predictor : Predictor
        Loaded CNP/Ensemble model for viscosity inference.
    target : ViscosityProfile
        Target (shear_rate, viscosity) pairs the optimizer tries to match.
    maxiter : int
        Maximum DE generations.
    popsize : int
        DE population multiplier (total population = popsize x n_variables).
    tol : float
        DE convergence tolerance.
    seed : int, optional
        Random seed for reproducibility.
    target_weights : sequence of float, optional
        Per-point loss weights, aligned with ``target.shear_rates``.
        Defaults to uniform.  Weights are normalised internally.
    lambda_unc : float
        Uncertainty-penalty coefficient λ ≥ 0.
    history_df : pd.DataFrame, optional
        Historical formulation data in encoded=False format used to
        warm-start the CNP context.  Obtained from
        ``FormulationController.get_all_as_dataframe(encoded=False)``.
    polish_continuous : bool
        Run a L-BFGS-B pass on the continuous variables after DE finishes,
        with all categorical indices held fixed.  Default True.
    """

    def __init__(
        self,
        constraints: Constraints,
        predictor: Predictor,
        target: ViscosityProfile,
        maxiter: int = 100,
        popsize: int = 15,
        tol: float = 1e-6,
        seed: Optional[int] = None,
        *,
        target_weights: Optional[Sequence[float]] = None,
        lambda_unc: float = 0.0,
        history_df: Optional[pd.DataFrame] = None,
        polish_continuous: bool = True,
        early_stopping_rounds: int = 20,
        improvement_tol: float = 1e-8,
    ) -> None:
        """Initializes the Optimizer for Differential Evolution (DE).

        This constructor prepares the search space by aligning constraints with
        ingredient databases, setting up categorical encodings, and pre-computing
        target viscosity profiles in log-space. It also optimizes the search
        dimensionality by identifying fixed categorical features and adjusting
        concentration bounds to prevent physically impossible formulations.

        Args:
            constraints: A Constraints object defining the allowed ranges and
                ingredient choices.
            predictor: The machine learning model used to predict viscosity
                from formulation parameters.
            target: The desired ViscosityProfile to optimize toward.
            maxiter: Maximum number of generations for the DE algorithm.
                Defaults to 100.
            popsize: Multiplier for setting the total population size
                (total pop = popsize * number of variables). Defaults to 15.
            tol: Relative tolerance for convergence. Defaults to 1e-6.
            seed: Random seed for reproducibility. Defaults to None.
            target_weights: Optional importance weights for specific shear rates
                in the target profile. Defaults to None (equal weighting).
            lambda_unc: Uncertainty penalty weight. Increasing this favors
                formulations where the model has higher confidence. Defaults to 0.0.
            history_df: Optional historical data for "warm-starting" the
                predictor's context. Defaults to None.
            polish_continuous: If True, performs a local search refinement
                following the DE global optimization. Defaults to True.
            early_stopping_rounds: Number of generations to wait for improvement
                before terminating. Defaults to 20.
            improvement_tol: Minimum change in objective value to qualify as
                an improvement for early stopping. Defaults to 1e-8.

        Raises:
            ValueError: If categorical ingredients cannot be aligned with the
                database or if target_weights dimensions do not match the target.
        """
        self.constraints = constraints
        self.predictor = predictor
        self.target = target
        self.maxiter = maxiter
        self.popsize = popsize
        self.tol = tol
        self.seed = seed
        self.lambda_unc = lambda_unc
        self.polish_continuous = polish_continuous
        self.early_stopping_rounds = early_stopping_rounds
        self.improvement_tol = improvement_tol
        self.bounds, self.encoding = self.constraints.build()
        all_ings = self.constraints._ingredient_ctrl.get_all_ingredients()
        self.cat_choices: Dict[str, List[Ingredient]] = {}

        for enc in self.encoding:
            if enc["type"] != "cat":
                continue
            feat = enc["feature"]
            pool = self.constraints._choices.get(feat)
            if not pool:
                cls = self.constraints._FEATURE_CLASS[feat]
                pool = [ing for ing in all_ings if isinstance(ing, cls)]
            name_to_ing: Dict[str, Ingredient] = {ing.name.lower(): ing for ing in pool}
            aligned: List[Ingredient] = []
            for name in enc["choices"]:
                ing = name_to_ing.get(name.lower())
                if ing is not None:
                    aligned.append(ing)

            if not aligned:
                raise ValueError(
                    "Could not align any ingredients for categorical feature."
                )

            self.cat_choices[feat] = aligned
        self._fixed_cats: Dict[str, Any] = {}
        variable_bounds: List[Tuple[float, float]] = []
        variable_encoding: List[Dict[str, Any]] = []
        for b, enc in zip(self.bounds, self.encoding):
            feat = enc["feature"]
            if enc["type"] == "cat" and len(self.cat_choices.get(feat, [])) == 1:
                self._fixed_cats[feat] = self.cat_choices[feat][0]
            else:
                variable_bounds.append(b)
                variable_encoding.append(enc)
        self.bounds = variable_bounds
        self.encoding = variable_encoding

        _CAT_TO_CONC: Dict[str, str] = {
            "Protein_type": "Protein_conc",
            "Buffer_type": "Buffer_conc",
            "Salt_type": "Salt_conc",
            "Stabilizer_type": "Stabilizer_conc",
            "Surfactant_type": "Surfactant_conc",
            "Excipient_type": "Excipient_conc",
        }
        _MIN_CONC: Dict[str, float] = {
            "Protein_conc": 1.0,
            "Buffer_conc": 1.0,
            "Salt_conc": 1.0,
            "Stabilizer_conc": 0.001,
            "Surfactant_conc": 0.001,
            "Excipient_conc": 1.0,
        }
        for cat_feat, _ing in self._fixed_cats.items():
            if self._is_none_ingredient(_ing):
                continue
            conc_feat = _CAT_TO_CONC.get(cat_feat)
            if conc_feat is None:
                continue
            min_c = _MIN_CONC.get(conc_feat, 1.0)
            for j, enc in enumerate(self.encoding):
                if enc["feature"] == conc_feat:
                    lo, hi = self.bounds[j]
                    if lo < min_c:
                        self.bounds[j] = (min_c, max(hi, min_c))
                    break

        self._cat_idx = [i for i, e in enumerate(self.encoding) if e["type"] == "cat"]
        self._num_idx = [i for i, e in enumerate(self.encoding) if e["type"] == "num"]

        # Pre-compute log-space targets
        self._target_log_shear = np.log10(
            np.array(target.shear_rates, dtype=float) + _EPS
        )
        self._target_log_visc = np.log10(
            np.array(target.viscosities, dtype=float) + _EPS
        )

        if target_weights is not None:
            w = np.asarray(target_weights, dtype=float)
            if w.shape[0] != len(target.shear_rates):
                raise ValueError(
                    f"target_weights length ({len(w)}) must match the number "
                    f"of target shear rates ({len(target.shear_rates)})."
                )
            self._target_weights = w / w.sum()
        else:
            n = len(target.shear_rates)
            self._target_weights = np.ones(n, dtype=float) / n

        # Warm-start the CNP predictor with historical context
        if history_df is not None and not history_df.empty:
            if hasattr(self.predictor, "learn"):
                try:
                    self.predictor.learn(history_df)
                except Exception as e:
                    Log.w(
                        TAG,
                        f"Optimizer: predictor warm-start failed - {e}. "
                        "Continuing with base-model context.",
                    )

    def _decode(self, x: np.ndarray) -> Dict[str, Any]:
        """Translates a continuous vector from the optimizer into a structured formulation.

        Differential Evolution operates on continuous floating-point vectors. This
        method maps those values back to the original feature space defined by
        the encoding. For categorical features, it uses rounding and clipping
        to map the float to the nearest valid ingredient index. It also merges
        in any 'fixed' categorical features that were removed from the search
        space during initialization to reconstruct a complete formulation.

        Args:
            x: A 1D NumPy array representing a candidate solution in the
                optimizer's search space.

        Returns:
            Dict[str, Any]: A dictionary representing the formulation, where keys
                are feature names and values are either floats (for concentrations)
                or Ingredient objects (for types).

        Note:
            Categorical mapping uses a 'nearest-neighbor' approach:
            index = round(value). A final validation pass ensures that all
            categorical values are members of their respective allowed choices,
            defaulting to the first available choice if an inconsistency is detected.
        """
        out: Dict[str, Any] = dict(self._fixed_cats)
        for xi, enc in zip(x, self.encoding):
            feat = enc["feature"]
            if enc["type"] == "cat":
                choices = self.cat_choices[feat]
                idx = int(np.clip(round(xi), 0, len(choices) - 1))
                out[feat] = choices[idx]
            else:
                out[feat] = float(xi)
        for feat, choices in self.cat_choices.items():
            val = out.get(feat)
            if val is not None and val not in choices:
                out[feat] = choices[0]

        return out

    @staticmethod
    def _is_none_ingredient(ing: Any) -> bool:
        """Determines if an ingredient object represents a null or 'None' entry.

        This helper handles two cases for 'empty' ingredients:
        - The object is a literal `None`.
        - The object is an Ingredient instance (or similar) whose `name`
           attribute is explicitly set to a variation of "None".

        This is used during optimization to skip constraints or logic
        associated with active ingredients when a "None" placeholder
        is selected for a specific feature slot (e.g., no stabilizer).

        Args:
            ing: The ingredient object or value to check.

        Returns:
            bool: True if the ingredient is considered a null placeholder;
                False otherwise.
        """
        if ing is None:
            return True
        name = getattr(ing, "name", None)
        return name is not None and str(name).strip().lower() == "none"

    def _build_formulation(self, feat_dict: Dict[str, Any]) -> Formulation:
        """Assembles a valid Formulation object from a dictionary of features.

        This method maps raw ingredient types and concentrations back into the
        domain-specific `Formulation` model. It iterates through a predefined
        schema of components, applying the appropriate setter methods and units
        (e.g., mg/mL for protein, mM for buffers).

        The method performs basic validation during assembly:
        - It skips ingredients identified as "None" via `_is_none_ingredient`.
        - It skips ingredients with a concentration of 0.0 to ensure the
           resulting model only contains active components.

        Args:
            feat_dict: A dictionary containing the decoded features. Keys should
                match the expected feature names (e.g., 'Protein_type',
                'Temperature'), and values should be the corresponding
                ingredient objects or floats.

        Returns:
            Formulation object
        """
        form = Formulation()

        _COMPONENTS = [
            ("Protein_type", "Protein_conc", "set_protein", "mg/mL"),
            ("Buffer_type", "Buffer_conc", "set_buffer", "mM"),
            ("Salt_type", "Salt_conc", "set_salt", "mM"),
            ("Stabilizer_type", "Stabilizer_conc", "set_stabilizer", "M"),
            ("Surfactant_type", "Surfactant_conc", "set_surfactant", "%w"),
            ("Excipient_type", "Excipient_conc", "set_excipient", "mM"),
        ]

        for type_key, conc_key, setter, units in _COMPONENTS:
            ing = feat_dict.get(type_key)
            conc = float(feat_dict.get(conc_key, 0.0))

            if self._is_none_ingredient(ing):
                continue

            if conc == 0.0:
                continue

            getattr(form, setter)(ing, conc, units)

        form.set_temperature(temp=feat_dict.get("Temperature", 25.0))
        return form

    _VISC_COLS = [
        "Viscosity_100",
        "Viscosity_1000",
        "Viscosity_10000",
        "Viscosity_100000",
        "Viscosity_15000000",
    ]

    def _extract_pred_visc(self, raw) -> np.ndarray:
        """Extracts and cleans predicted viscosity values from raw predictor output.

        This method handles the transition from the predictor's raw output format
        (often a pandas DataFrame with specific column naming conventions) to a
        standardized NumPy array used for objective function calculations. It
        prioritizes columns defined in `_VISC_COLS` and looks for 'Pred_'
        prefixed alternatives if the standard names are missing.

        To maintain stability during optimization, any non-finite values
        (NaN, Inf) are replaced with a pre-defined `_LARGE_LOSS` penalty,
        effectively steering the Differential Evolution algorithm away from
        unstable regions of the search space.

        Args:
            raw: The raw output from a predictor. This can be a pandas DataFrame
                containing viscosity columns, a NumPy array, or a list of
                numerical values.

        Returns:
            np.ndarray: A 1D array of extracted viscosity values corresponding
                to the optimizer's shear rate profile.
        """
        if isinstance(raw, pd.DataFrame):
            row = raw.iloc[0] if len(raw) > 0 else pd.Series(dtype=float)
            vals = []
            for col in self._VISC_COLS:
                v = row.get(col, row.get(f"Pred_{col}", np.nan))
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = np.nan
                vals.append(v)
            arr = np.array(vals, dtype=float)
        else:
            arr = np.asarray(raw, dtype=float).flatten()
        arr = np.where(np.isfinite(arr), arr, _LARGE_LOSS)
        return arr

    def _log_mse_loss(self, pred_viscosities: np.ndarray) -> float:
        """Calculates the weighted Mean Squared Error (MSE) in log10-log10 space.

        This method computes the loss between the predicted viscosity profile
        and the target profile. By operating in log-space for both shear rate
        (x-axis) and viscosity (y-axis), the optimizer treats a 10% error at
        1 cP with the same mathematical weight as a 10% error at 100 cP.

        The method interpolates the predicted values—which are generated at
        fixed shear rates—onto the specific log-shear rates of the target
        profile before calculating the squared error and applying importance
        weights.

        Args:
            pred_viscosities: A 1D array of predicted viscosities corresponding
                to the standard optimizer shear rate profile.

        Returns:
            float: The weighted average of the squared log-errors.

        Note:
            Predictions are clipped to `_EPS` (a small positive constant)
            prior to log transformation to avoid mathematical errors with
            zero or negative values.
        """
        log_pred = np.log10(np.clip(pred_viscosities.astype(float), _EPS, None))
        log_interp = np.interp(self._target_log_shear, _LOG_PRED_SHEAR, log_pred)
        sq_err = (log_interp - self._target_log_visc) ** 2
        return float(np.dot(sq_err, self._target_weights))

    def _objective(self, x: np.ndarray) -> float:
        """The core cost function minimized by the Differential Evolution algorithm.

        This method evaluates a single candidate solution by performing a
        full round-trip through the optimization pipeline including decoding, assembly,
        prediction and scoring.

        Args:
            x: A 1D NumPy array representing a candidate formulation in
                the continuous search space.

        Returns:
            float: The total calculated loss. A lower value indicates a
                formulation that better matches the target profile with
                higher model confidence.

        Note:
            If uncertainty prediction fails for any reason, the method
            gracefully falls back to a standard point-prediction to ensure
            the optimization loop is not interrupted.
        """
        feat_dict = self._decode(x)
        formulation = self._build_formulation(feat_dict)
        df = formulation.to_dataframe(encoded=False, training=False)

        unc_penalty = 0.0
        if self.lambda_unc > 0.0:
            try:
                mean_pred, unc = self.predictor.predict_with_uncertainty(df)
                pred_visc = self._extract_pred_visc(mean_pred)
                unc_penalty = float(np.nanmean(unc.get("std_log10", [0.0])))
            except Exception:
                pred_visc = self._extract_pred_visc(self.predictor.predict(df))
        else:
            pred_visc = self._extract_pred_visc(self.predictor.predict(df))

        loss = self._log_mse_loss(pred_visc)
        if self.lambda_unc > 0.0:
            loss += self.lambda_unc * unc_penalty
        return loss

    def _build_initial_population(self, popsize_total: int) -> np.ndarray:
        """Constructs the starting population for the Differential Evolution algorithm.

        This method initializes the search space using a hybrid sampling strategy
        to ensure broad coverage and minimize initial clustering utilizing the following
        stragegies:

        1. Continuous Variables: Uses Latin Hypercube Sampling (LHS)
           approach. By dividing the range into strata and ensuring exactly one
           sample per stratum, the optimizer starts with a more uniform
           distribution than standard random sampling, reducing 'blind spots'
           in the search space.
        2. Categorical Variables: Uses a discrete uniform distribution to
           assign initial ingredient indices, ensuring that all valid choices
           for a category are likely represented in the first generation.

        Args:
            popsize_total: The total number of individuals (candidate formulations)
                to generate for the population.

        Returns:
            np.ndarray: A 2D array of shape `(popsize_total, n_vars)` containing
                the initialized continuous vectors ready for the first
                generation of DE.
        """
        rng = np.random.default_rng(self.seed)
        n_vars = len(self.bounds)
        pop = np.empty((popsize_total, n_vars), dtype=float)

        # Continuous: scrambled Latin Hypercube
        if self._num_idx:
            n_cont = len(self._num_idx)
            strata = (
                np.arange(popsize_total)[:, None] + rng.random((popsize_total, n_cont))
            ) / popsize_total
            for j in range(n_cont):
                rng.shuffle(strata[:, j])
            for k, i in enumerate(self._num_idx):
                lo, hi = self.bounds[i]
                pop[:, i] = lo + strata[:, k] * (hi - lo)

        # Categorical: discrete uniform over valid indices
        for i in self._cat_idx:
            lo, hi = self.bounds[i]
            n_choices = int(round(hi)) + 1
            pop[:, i] = rng.integers(
                0, n_choices, size=popsize_total, dtype=int
            ).astype(float)

        return pop

    def _polish_continuous(self, x_best: np.ndarray) -> Tuple[np.ndarray, float]:
        """Performs local refinement of continuous parameters while keeping categories fixed.

        Differential Evolution is excellent at global search and identifying the
        correct categorical 'buckets' (e.g., the right Protein or Buffer type),
        but it can be less efficient at converging on the exact decimal-point
        optimum for concentrations.

        This method "polishes" the best solution found by:
        1. Freezing all categorical indices to their nearest integer values.
        2. Isolating the continuous (numerical) variables.
        3. Executing a high-precision, gradient-based local search (L-BFGS-B).

        This two-stage approach ensures that the final formulation is not only in
        the right design region but is also mathematically optimized for its
        specific ingredients.

        Args:
            x_best: The best-performing continuous vector identified by the
                global optimizer.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing:
                - x_polished: The refined vector with optimized continuous values.
                - loss: The final objective function value after polishing.

        Note:
            If the optimization space contains no continuous variables, the
            method returns the input vector and its objective value immediately.
        """
        if not self._num_idx:
            return x_best.copy(), self._objective(x_best)
        frozen: Dict[int, float] = {i: float(round(x_best[i])) for i in self._cat_idx}
        x0_cont = x_best[self._num_idx]
        bounds_cont = [self.bounds[i] for i in self._num_idx]

        def _cont_obj(x_cont: np.ndarray) -> float:
            """Internal objective wrapper for local continuous search."""
            x_full = x_best.copy()
            for j, i in enumerate(self._num_idx):
                x_full[i] = x_cont[j]
            for i, v in frozen.items():
                x_full[i] = v
            return self._objective(x_full)

        result = minimize(
            _cont_obj,
            x0_cont,
            method="L-BFGS-B",
            bounds=bounds_cont,
            options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-8},
        )

        x_polished = x_best.copy()
        for j, i in enumerate(self._num_idx):
            x_polished[i] = result.x[j]
        for i, v in frozen.items():
            x_polished[i] = v

        return x_polished, float(result.fun)

    def optimize(
        self,
        strategy: str = "best1bin",
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        atol: float = 0.0,
        workers: int = 1,
        progress_callback: Optional[Callable[[OptimizationStatus], None]] = None,
        early_stopping_rounds: Optional[int] = None,
        improvement_tol: Optional[float] = None,
    ) -> Formulation:
        """Executes the global-to-local optimization pipeline to find an ideal formulation.

        This method coordinates the Differential Evolution (DE) search to minimize
        the difference between predicted and target viscosity profiles. It handles
        population initialization, real-time progress tracking, early stopping
        logic, and an optional gradient-based local "polishing" step.

        The process follows three main phases:
        1. Seeding: Generates an initial population using Latin Hypercube Sampling.
        2. Global Search: Iterates through generations using the DE algorithm.
           In each generation, a callback updates an internal `OptimizationProgressTracker`
           and evaluates early stopping criteria.
        3. Polishing: If `polish_continuous` is enabled, performs a high-precision
           local search on the numerical parameters of the best categorical solution.

        Args:
            strategy: The DE differential evolution strategy to use (e.g., 'best1bin',
                'rand1exp'). Defaults to "best1bin".
            mutation: The mutation constant (F). A tuple (min, max) indicates
                dithering. Defaults to (0.5, 1.0).
            recombination: The recombination constant (CR). Range [0, 1].
                Defaults to 0.7.
            atol: Absolute tolerance for convergence. Defaults to 0.0.
            workers: Number of CPU workers for parallel objective evaluation.
                Use -1 for all available cores. Defaults to 1.
            progress_callback: An optional function that receives an
                `OptimizationStatus` object after every generation.
            early_stopping_rounds: Number of generations with no improvement
                before stopping. Overrides class default if provided.
            improvement_tol: The minimum change required to count as an
                improvement. Overrides class default if provided.

        Returns:
            Formulation: The optimized Formulation object containing the
                ingredients, concentrations, and temperature that best match
                the target profile.

        Notes:
            The early stopping logic relies on a closure-based callback that
            communicates with the `OptimizationProgressTracker`. This allows
            real-time monitoring of the 'stagnation' count.
        """
        _es_rounds = (
            early_stopping_rounds
            if early_stopping_rounds is not None
            else self.early_stopping_rounds
        )
        _imp_tol = (
            improvement_tol if improvement_tol is not None else self.improvement_tol
        )

        tracker = OptimizationProgressTracker()
        best_obj_value = [float("inf")]
        stagnant_gens = [0]  # mutable counter for the closure
        popsize_total = self.popsize * len(self.bounds)
        init_pop = self._build_initial_population(popsize_total)

        def _callback(xk: np.ndarray, convergence: float = 0.0) -> bool:
            current = self._objective(xk)
            improved = current < best_obj_value[0] - _imp_tol
            if improved:
                best_obj_value[0] = current
                stagnant_gens[0] = 0
            else:
                stagnant_gens[0] += 1

            status = OptimizationStatus(
                iteration=len(tracker.history) + 1,
                num_iterations=self.maxiter,
                best_value=best_obj_value[0],
                population_size=self.popsize,
                convergence=convergence,
            )
            tracker(status)
            if progress_callback:
                progress_callback(status)

            # Return True to ask DE to stop; only when early stopping is on
            if _es_rounds > 0 and stagnant_gens[0] >= _es_rounds:
                return True
            return False

        result = differential_evolution(
            func=self._objective,
            bounds=self.bounds,
            maxiter=self.maxiter,
            popsize=self.popsize,
            tol=self.tol,
            seed=self.seed,
            polish=False,
            disp=False,
            workers=int(workers),
            atol=atol,
            recombination=recombination,
            mutation=mutation,
            strategy=strategy,
            init=init_pop,
            callback=_callback,
        )

        x_best = result.x
        if self.polish_continuous:
            x_polished, loss_polished = self._polish_continuous(x_best)
            if loss_polished < self._objective(x_best):
                x_best = x_polished

        return self._build_formulation(self._decode(x_best))
