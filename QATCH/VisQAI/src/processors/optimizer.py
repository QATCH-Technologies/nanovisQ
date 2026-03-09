"""
optimizer.py

Constrained formulation optimizer using Differential Evolution with a
correct mixed discrete-continuous search strategy.

Bug Fixes vs v1.2
─────────────────
1.  encoded=False on to_dataframe()
        The old optimizer called formulation.to_dataframe(training=False),
        which defaults to encoded=True and passes integer enc_ids to the
        predictor.  Every other prediction call in the codebase (sampler.py
        ×5) uses encoded=False.  The CNP engine expects human-readable
        ingredient names; enc_ids silently produce garbage predictions.

2.  Categorical choice–index alignment
        constraints.build() sorts categorical names alphabetically via
        ListUtils.unique_case_insensitive_sort and sets bounds=(0, n-1)
        against that sorted order.  The old optimizer built cat_choices from
        the unsorted constraints._choices pool, so index 0 decoded to the
        wrong ingredient.  Fixed by aligning cat_choices to match the sorted
        names list returned in the encoding.

3.  polish=True destroys categoricals
        differential_evolution(polish=True) runs L-BFGS-B on the full
        vector, including categorical dimensions.  L-BFGS-B treats them as
        continuous, drifts them off-integer, and the final _decode(result.x)
        rounds them to a different (often worse) formulation than DE found.
        Fixed with polish=False in DE and a separate continuous-only L-BFGS-B
        pass that freezes all categoricals at their DE-best integer values.

4.  Linear-space MSE loss
        Viscosity spans 4+ orders of magnitude (cP at 100 s⁻¹ can be 50×
        larger than cP at 15 M s⁻¹).  Linear MSE means absolute errors at
        high-viscosity low-shear points completely dominate; the optimizer
        effectively ignores accuracy at the shear rates the user cares about.
        Replaced with weighted MSE in log₁₀ space with log-shear interpolation
        so every target point contributes proportionally to its weight
        regardless of absolute scale.

5.  Zero-shot predictions (no warm-start)
        The CNP engine is a conditional neural process whose few-shot
        accuracy depends on context vectors loaded via predictor.learn().
        Sampler always calls this before predicting; the optimizer never did,
        running in zero-shot mode.  Fixed by accepting an optional history_df
        and calling predictor.learn() once before the first objective
        evaluation.

6.  LHS initialises categoricals as continuous floats
        init='latinhypercube' generates values like 2.71 for a categorical
        with 5 choices, so almost no initial candidates land at valid integer
        corners and DE wastes early generations correcting them.  Fixed with a
        custom discrete-aware initialiser: Latin Hypercube for continuous
        variables, discrete uniform integer sampling for categoricals.

New Features (non-breaking, all keyword-only)
─────────────────────────────────────────────
  target_weights     Per-target loss weights (length = len(target.shear_rates)).
                     Default uniform.  Increase weight for operating points that
                     matter most (e.g. 10 000 s⁻¹ for SC injections).

  lambda_unc         Uncertainty-penalty coefficient λ.  When > 0 the objective
                     adds λ × mean(std_log₁₀), steering the search toward
                     formulations the model predicts confidently.

  history_df         Historical formulation DataFrame (encoded=False) used to
                     warm-start the CNP context via predictor.learn().

  polish_continuous  Run a L-BFGS-B stage on continuous variables after DE,
                     with all categorical indices frozen.  Default True.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-09

Version:
    2.0
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

try:
    from src.models.formulation import Formulation, ViscosityProfile
    from src.models.ingredient import Ingredient
    from src.models.predictor import Predictor
    from src.utils.constraints import Constraints
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.formulation import Formulation, ViscosityProfile
    from QATCH.VisQAI.src.models.ingredient import Ingredient
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.utils.constraints import Constraints


# Canonical shear-rate grid the predictor outputs (must match inference engine)
_PRED_SHEAR_RATES: List[float] = [100.0, 1_000.0, 10_000.0, 100_000.0, 15_000_000.0]
_LOG_PRED_SHEAR = np.log10(_PRED_SHEAR_RATES)

_EPS = 1e-9  # guard against log10(0)
_LARGE_LOSS = 1e6  # finite penalty returned for invalid / None predictions


# ══════════════════════════════════════════════════════════════════════════════
# Progress helpers  (public API backward-compatible with v1.2)
# ══════════════════════════════════════════════════════════════════════════════


class OptimizationStatus:
    """Snapshot of optimizer state emitted once per DE generation."""

    def __init__(
        self,
        iteration: int,
        num_iterations: int,
        best_value: float,
        population_size: int,
        convergence: float,
    ) -> None:
        self.iteration = iteration
        self.num_iterations = num_iterations
        self.best_value = best_value
        self.population_size = population_size
        self.convergence = convergence
        self.progress_percent = (
            iteration / num_iterations * 100 if num_iterations > 0 else 0.0
        )

    def __repr__(self) -> str:
        return (
            f"OptimizationStatus(iteration={self.iteration}, "
            f"best_value={self.best_value:.6f}, "
            f"progress={self.progress_percent:.1f}%)"
        )


class OptimizationProgressTracker:
    """Accumulates per-iteration status objects and prints a progress line."""

    def __init__(self) -> None:
        self.history: List[OptimizationStatus] = []
        self.iterations: List[int] = []
        self.best_values: List[float] = []
        self.start_value: Optional[float] = None
        self.best_value: Optional[float] = None

    def __call__(self, status: OptimizationStatus) -> None:
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
        print(
            f"[{status.iteration:3d}/{status.num_iterations}] "
            f"Best: {status.best_value:10.6f} | "
            f"Improvement: {improvement:10.6f} | "
            f"Progress: {status.progress_percent:6.1f}%"
        )

    def get_improvement_rate(self) -> float:
        """Average objective improvement per generation."""
        if len(self.history) < 2 or self.start_value is None:
            return 0.0
        return (self.start_value - self.best_value) / len(self.history)

    def get_stagnation_iterations(self, threshold: int = 10) -> int:
        """Count consecutive tail generations with no improvement."""
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
        return self.iterations.copy(), self.best_values.copy()


# ══════════════════════════════════════════════════════════════════════════════
# Core Optimizer
# ══════════════════════════════════════════════════════════════════════════════


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
        DE population multiplier (total population = popsize × n_variables).
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

        # ── 1. Build bounds + encoding from constraints ────────────────────────
        self.bounds, self.encoding = self.constraints.build()

        # ── 2. Align cat_choices to the sorted names in each encoding entry ───
        #
        #   constraints.build() sorts each categorical's names alphabetically
        #   (via ListUtils.unique_case_insensitive_sort) and derives bounds from
        #   that sorted list.  We MUST build cat_choices in the identical order,
        #   otherwise index 0 in the vector decodes to the wrong ingredient.
        #   (This was Bug #2 in v1.2 — the old code used the unsorted pool.)
        #
        all_ings = self.constraints._ingredient_ctrl.get_all_ingredients()
        self.cat_choices: Dict[str, List[Ingredient]] = {}

        for enc in self.encoding:
            if enc["type"] != "cat":
                continue
            feat = enc["feature"]

            # Ingredient pool — respect any add_choices() constraint
            pool = self.constraints._choices.get(feat)
            if not pool:
                cls = self.constraints._FEATURE_CLASS[feat]
                pool = [ing for ing in all_ings if isinstance(ing, cls)]

            # Case-insensitive name → Ingredient lookup over the pool
            name_to_ing: Dict[str, Ingredient] = {ing.name.lower(): ing for ing in pool}

            # Reconstruct the list in the same sorted order build() used.
            # enc["choices"] is the already-sorted names list from build().
            aligned: List[Ingredient] = []
            for name in enc["choices"]:
                ing = name_to_ing.get(name.lower())
                if ing is not None:
                    aligned.append(ing)

            if not aligned:
                raise ValueError(
                    f"Could not align any ingredients for categorical feature "
                    f"'{feat}'.  This is a bug — the pool should never be "
                    f"empty at this point."
                )

            self.cat_choices[feat] = aligned

        # ── 3. Pull single-choice categoricals OUT of the DE search space ──────
        #
        #   When bounds = (0, 0) SciPy's DE normalises with (hi - lo) = 0 in
        #   the denominator, producing NaN for that dimension.  NaN propagates
        #   through mutation and the decoded index becomes undefined — the
        #   constrained protein (or other ingredient) ends up wrong in the output.
        #
        #   Fix: remove fixed categoricals from bounds / encoding entirely and
        #   store them in _fixed_cats.  _decode() injects them back before
        #   building each candidate formulation.
        #
        self._fixed_cats: Dict[str, Any] = {}
        variable_bounds: List[Tuple[float, float]] = []
        variable_encoding: List[Dict[str, Any]] = []

        for b, enc in zip(self.bounds, self.encoding):
            feat = enc["feature"]
            if enc["type"] == "cat" and len(self.cat_choices.get(feat, [])) == 1:
                # Only one valid choice → not a variable, it's a constant.
                self._fixed_cats[feat] = self.cat_choices[feat][0]
            else:
                variable_bounds.append(b)
                variable_encoding.append(enc)

        self.bounds = variable_bounds
        self.encoding = variable_encoding

        # ── 4. Enforce minimum concentration for constrained ingredient types ──
        #
        #   If the user constrained "Protein Type is Adalimumab" but the
        #   protein_conc lower bound is 0, DE can still set conc=0, making
        #   _build_formulation skip the protein entirely.  Lift the lower
        #   bound to a small positive value so DE always uses the ingredient.
        #
        _CAT_TO_CONC: Dict[str, str] = {
            "Protein_type": "Protein_conc",
            "Buffer_type": "Buffer_conc",
            "Salt_type": "Salt_conc",
            "Stabilizer_type": "Stabilizer_conc",
            "Surfactant_type": "Surfactant_conc",
            "Excipient_type": "Excipient_conc",
        }
        # Sensible positive minimums (well below typical experimental ranges)
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
                continue  # "None" type — don't force non-zero conc
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

        # ── 5. Index helpers for the categorical / continuous split ────────────
        self._cat_idx = [i for i, e in enumerate(self.encoding) if e["type"] == "cat"]
        self._num_idx = [i for i, e in enumerate(self.encoding) if e["type"] == "num"]

        # ── 6. Pre-compute log₁₀-space targets ────────────────────────────────
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

        # ── 7. Warm-start the CNP predictor with historical context ───────────
        #
        #   Sampler always calls predictor.learn(history_df) before predicting
        #   because the CNP engine is a conditional neural process; its accuracy
        #   depends on having context vectors loaded.  The old optimizer never
        #   did this, running in zero-shot mode.  (Bug #5 fix.)
        #
        if history_df is not None and not history_df.empty:
            if hasattr(self.predictor, "learn"):
                try:
                    self.predictor.learn(history_df)
                except Exception as exc:
                    warnings.warn(
                        f"Optimizer: predictor warm-start failed — {exc}. "
                        "Continuing with base-model context.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _decode(self, x: np.ndarray) -> Dict[str, Any]:
        """Map a continuous DE vector → {feature: value} dict.

        Fixed categoricals (single-choice constraints) are injected directly
        from _fixed_cats — they are not present in the DE vector at all.
        Variable categoricals are rounded to the nearest integer and clamped
        before indexing into cat_choices.

        A final hard-clamp pass ensures that every categorical in the decoded
        dict is within its allowed set, regardless of floating-point drift or
        any upstream ordering inconsistency.
        """
        # Seed with fixed categoricals so callers always get a complete dict
        out: Dict[str, Any] = dict(self._fixed_cats)
        for xi, enc in zip(x, self.encoding):
            feat = enc["feature"]
            if enc["type"] == "cat":
                choices = self.cat_choices[feat]
                idx = int(np.clip(round(xi), 0, len(choices) - 1))
                out[feat] = choices[idx]
            else:
                out[feat] = float(xi)

        # Hard-clamp: verify every decoded categorical is in its allowed pool.
        # This catches any edge case where floating-point drift or index
        # mis-alignment would produce a value outside the constraint set.
        for feat, choices in self.cat_choices.items():
            val = out.get(feat)
            if val is not None and val not in choices:
                out[feat] = choices[0]  # fall back to first (lowest-index) valid choice

        return out

    @staticmethod
    def _is_none_ingredient(ing: Any) -> bool:
        """Return True when an ingredient should be treated as absent.

        Covers two cases:
          - The ingredient object itself is None (missing from the pool).
          - The ingredient's name is the sentinel string "None" / "none",
            which some databases use to represent an optional component that
            was not selected.
        """
        if ing is None:
            return True
        name = getattr(ing, "name", None)
        return name is not None and str(name).strip().lower() == "none"

    def _build_formulation(self, feat_dict: Dict[str, Any]) -> Formulation:
        """Construct a Formulation from a decoded feature dict.

        Safety rule: if an ingredient is absent (None) or carries the name
        "None", its concentration is forced to 0.0 and the component is not
        set on the formulation.  This prevents the engine from receiving a
        formulation where, e.g., Salt_type=None but Salt_conc=75, which is
        physically meaningless and causes prediction errors.
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
                # Ingredient absent or named "None" — skip entirely.
                # Concentration is implicitly 0; no component is attached.
                continue

            if conc == 0.0:
                # Zero concentration with a real ingredient: valid (e.g. a
                # stabilizer that wasn't needed).  Skip to keep the formulation
                # lean, consistent with how Sampler handles this case.
                continue

            getattr(form, setter)(ing, conc, units)

        form.set_temperature(temp=feat_dict.get("Temperature", 25.0))
        return form

    # Ordered viscosity column names produced by to_dataframe / the engine
    _VISC_COLS = [
        "Viscosity_100",
        "Viscosity_1000",
        "Viscosity_10000",
        "Viscosity_100000",
        "Viscosity_15000000",
    ]

    def _extract_pred_visc(self, raw) -> np.ndarray:
        """Convert whatever the engine returns into a clean float64 array.

        ``predictor.predict()`` returns a DataFrame whose columns may be
        named ``Viscosity_*`` or ``Pred_Viscosity_*``.  Blindly calling
        ``np.asarray(df).flatten()`` produces an object-dtype array whenever
        any cell is None / pd.NA, which makes np.clip crash with:
            TypeError: '>=' not supported between instances of 'NoneType' and 'float'

        This helper extracts the five viscosity values in the canonical shear-
        rate order and converts them to float64, replacing any NaN/None with
        the large-penalty sentinel ``_LARGE_LOSS`` so the candidate is
        penalised rather than crashing.
        """
        if isinstance(raw, pd.DataFrame):
            # Try canonical column names first, then Pred_* variants
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
            # Already array-like (e.g. from predict_with_uncertainty mean)
            arr = np.asarray(raw, dtype=float).flatten()

        # Replace NaN / ±inf with a large-but-finite penalty value so
        # _log_mse_loss never receives bad input.
        arr = np.where(np.isfinite(arr), arr, _LARGE_LOSS)
        return arr

    def _log_mse_loss(self, pred_viscosities: np.ndarray) -> float:
        """Weighted MSE in log₁₀ space between predictions and targets.

        Interpolation is performed on the log₁₀(shear_rate) axis so that
        targets at shear rates between the five predictor output points are
        handled correctly.

        Parameters
        ----------
        pred_viscosities : float64 array of shape (5,)
            Predicted viscosities at _PRED_SHEAR_RATES in cP.

        Returns
        -------
        float
            Weighted mean squared error in log₁₀(cP) units.
        """
        # Ensure strictly positive before log; _extract_pred_visc should have
        # removed NaN/None already, but clip defensively.
        log_pred = np.log10(np.clip(pred_viscosities.astype(float), _EPS, None))
        log_interp = np.interp(self._target_log_shear, _LOG_PRED_SHEAR, log_pred)
        sq_err = (log_interp - self._target_log_visc) ** 2
        return float(np.dot(sq_err, self._target_weights))

    def _objective(self, x: np.ndarray) -> float:
        """Evaluate the loss for a DE candidate vector.

        Uses predict_with_uncertainty when lambda_unc > 0, falling back to
        predict() on failure to keep evaluation cost low in the common case.
        """
        feat_dict = self._decode(x)
        formulation = self._build_formulation(feat_dict)

        # Bug #1 fix: encoded=False — the engine expects human-readable names,
        # not integer enc_ids.  The old call was to_dataframe(training=False)
        # which defaulted to encoded=True.
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
        """Build a discrete-aware initial population.

        Categorical variables are sampled as uniform integers over
        [0, n_choices).  Continuous variables use a scrambled Latin Hypercube
        for good space coverage.  This fixes Bug #6 where standard LHS
        initialised categoricals as floats like 2.71 and wasted early
        generations correcting them.
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
        """Freeze categoricals and run L-BFGS-B on continuous variables only.

        This is the correct approach for mixed-integer polishing.  Bug #3
        fix: the old polish=True ran L-BFGS-B over the full vector including
        categoricals.  L-BFGS-B drifted them off-integer and the rounded
        result was often worse than what DE had found.

        Parameters
        ----------
        x_best : np.ndarray
            Best solution vector from DE.

        Returns
        -------
        (x_polished, loss_polished)
        """
        if not self._num_idx:
            return x_best.copy(), self._objective(x_best)

        # Round and freeze all categorical indices
        frozen: Dict[int, float] = {i: float(round(x_best[i])) for i in self._cat_idx}
        x0_cont = x_best[self._num_idx]
        bounds_cont = [self.bounds[i] for i in self._num_idx]

        def _cont_obj(x_cont: np.ndarray) -> float:
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

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

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
        """Run DE and return the best formulation found.

        The ``init`` keyword from v1.2 has been removed; population
        initialisation is now always discrete-aware.

        Parameters
        ----------
        strategy : str
            DE mutation strategy.
        mutation : (float, float)
            Dithering bounds for the mutation constant F.
        recombination : float
            Crossover probability [0, 1].
        atol : float
            Absolute convergence tolerance.
        workers : int
            Parallel workers.  Use 1 when the predictor holds GPU tensors
            (pickling may fail with >1).
        progress_callback : callable, optional
            Called once per generation with an OptimizationStatus object.
        early_stopping_rounds : int, optional
            Override the instance-level setting for this call.  Stop DE
            after this many consecutive generations with no improvement
            greater than improvement_tol.  0 disables early stopping.
        improvement_tol : float, optional
            Minimum absolute decrease in the best objective value that
            counts as meaningful progress.  Default 1e-8.

        Returns
        -------
        Formulation
        """
        # Resolve per-call overrides vs instance defaults
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
            polish=False,  # Bug #3 fix: we handle polish correctly below
            disp=False,
            workers=int(workers),
            atol=atol,
            recombination=recombination,
            mutation=mutation,
            strategy=strategy,
            init=init_pop,  # Bug #6 fix: discrete-aware initialisation
            callback=_callback,
        )

        x_best = result.x

        # Stage 2: continuous-only L-BFGS-B with categoricals frozen
        if self.polish_continuous:
            x_polished, loss_polished = self._polish_continuous(x_best)
            if loss_polished < self._objective(x_best):
                x_best = x_polished

        return self._build_formulation(self._decode(x_best))

    # ── Backward-compatible shim ───────────────────────────────────────────────

    def _mse_loss(self, prof1: ViscosityProfile, prof2: ViscosityProfile) -> float:
        """Legacy linear-space MSE between two ViscosityProfile objects.

        Retained for external callers that depended on this method in v1.2.
        New internal code uses _log_mse_loss exclusively.
        """
        v1 = np.interp(prof2.shear_rates, prof1.shear_rates, prof1.viscosities)
        return float(np.mean((v1 - np.array(prof2.viscosities)) ** 2))
