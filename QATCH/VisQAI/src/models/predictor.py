"""
predictor.py

Provides the Predictor class for loading a packaged viscosity model
and performing inference.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-26

Version:
    6.1
"""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    TAG = "[PredictorWrapper]"
    from QATCH.common.logger import Logger as Log
except ImportError:
    TAG = "[PredictorWrapper (HEADLESS)]"

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


try:
    from .secure_loader import (
        SecurePackageLoader,
        SecurityError,
        create_secure_loader_for_extracted_package,
    )

    SECURITY_AVAILABLE = True
except ImportError:
    try:
        from QATCH.VisQAI.src.io.secure_loader import (
            SecurePackageLoader,
            SecurityError,
            create_secure_loader_for_extracted_package,
        )

        SECURITY_AVAILABLE = True
    except ImportError:
        SECURITY_AVAILABLE = False
        SecurityError = Exception


class Predictor:
    """High-level interface for loading and running VisQ packaged viscosity models.

    Handles extraction of ``.visq`` zip packages, secure manifest loading,
    engine instantiation, inference, context-based adaptation, and persistence.

    Attributes:
        zip_path (Path): Path to the ``.visq`` package file.
        extracted_path (Path): Temporary directory where the package is extracted.
        manifest (dict): Parsed package manifest loaded from the ``.visq`` file.
        model_type (str): Architecture identifier (e.g. ``"CNP"``, ``"unknown"``).
        engine: The instantiated inference engine object.
    """

    def __init__(self, zip_path: str, temp_dir: Optional[str] = None):
        """Initialize the Predictor by extracting and loading the model package.

        Args:
            zip_path (str): Path to the ``.visq`` model package to load.
            temp_dir (str, optional): Directory in which to create the temporary
                extraction folder. Defaults to the system temp directory.

        Raises:
            FileNotFoundError: If ``zip_path`` does not exist on disk.
            NotImplementedError: If the package architecture is unsupported.
            Exception: Propagated from the secure loader on signature or manifest errors.
        """
        self.zip_path = Path(zip_path)
        self._tmpdir = tempfile.TemporaryDirectory(dir=temp_dir)
        self.extracted_path = Path(self._tmpdir.name)

        self.manifest = {}
        self.model_type = "unknown"
        self.engine = None

        self._load_package()

    def _load_package(self):
        """Unzip the package and dynamically load the inference engine.

        Loading strategy (in priority order):

        1. **Manifest-driven (v2):** If the manifest contains a ``modules``
           section, all declared code modules are loaded in the specified
           order and the entry-point class is resolved from the manifest.
        2. **Secure-loader legacy (v1):** Falls back to scanning ``files``
           for a single ``inference_code`` entry.
        3. **Hardcoded legacy:** Last-resort for very old packages that
           pre-date the secure loader.

        Raises:
            FileNotFoundError: If ``self.zip_path`` does not exist.
            NotImplementedError: If the package architecture is not supported.
            Exception: Propagated from the secure loader on any loading failure.
        """
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Package not found: {self.zip_path}")

        Log.i(TAG, f"Extracting package: {self.zip_path}")
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            zf.extractall(self.extracted_path)
        try:
            import sklearn.compose._column_transformer

            if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):

                class _RemainderColsList(list):
                    """Stub for legacy pickle compatibility (sklearn < 1.2)."""

                    pass

                sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
                Log.w("Applied sklearn._RemainderColsList for legacy model support.")
        except ImportError:
            pass
        except Exception as e:
            Log.w(TAG, f"Failed to apply sklearn patch: {e}")

        if not SECURITY_AVAILABLE:
            Log.e("SecurePackageLoader unavailable - cannot load package.")
            return

        try:
            loader = create_secure_loader_for_extracted_package(
                self.extracted_path, enforce_signatures=True
            )
            self.manifest = loader.load_manifest()
            arch = self.manifest.get("architecture", "LegacyEnsemble")

            if loader.has_module_manifest:
                self._load_from_module_manifest(loader)
            elif arch == "CrossSampleCNP":
                self._load_single_inference_module(loader)
            else:
                raise NotImplementedError(
                    "Model package is not supported.  Only compatible `.visq` packages"
                    "are supported."
                )

        except Exception as e:
            Log.e(TAG, f"Loader Error: {e}")
            raise

    def _load_from_module_manifest(self, loader: "SecurePackageLoader"):
        """Load all modules declared in the manifest and instantiate the entry-point engine.

        The manifest's ``modules.entry_point`` may contain an ``init_kwargs`` dict.
        The special token ``"$PACKAGE_DIR"`` is substituted with the actual extracted
        path at runtime before the engine class is constructed.

        Args:
            loader (SecurePackageLoader): A loader whose manifest has already been loaded.
        """
        loaded = loader.load_modules()
        Log.i(TAG, f"Loaded {len(loaded)} module(s) from package manifest.")

        engine_cls = loader.get_engine_class()

        # Resolve init kwargs from manifest (supports $PACKAGE_DIR token)
        modules_section = self.manifest.get("modules", {})
        raw_kwargs = modules_section.get("entry_point", {}).get("init_kwargs", {})
        init_kwargs = {
            k: (str(self.extracted_path) if v == "$PACKAGE_DIR" else v)
            for k, v in raw_kwargs.items()
        }

        # Default: pass model_dir if no explicit kwargs
        if not init_kwargs:
            init_kwargs = {"model_dir": str(self.extracted_path)}

        self.engine = engine_cls(**init_kwargs)
        self.model_type = self.manifest.get("architecture", "unknown")
        Log.i(TAG, f"{engine_cls.__name__} engine initialized " f"(arch={self.model_type}).")

    def _load_single_inference_module(self, loader: "SecurePackageLoader"):
        """Load a single inference module using the legacy v1 path.

        Looks for ``ViscosityPredictorCNP`` by convention inside the loaded module
        and initializes it with the extracted package directory.

        Args:
            loader (SecurePackageLoader): A loader whose manifest has already been loaded.
        """
        mod = loader.load_inference_module()
        self.model_type = "CNP"
        self.engine = mod.ViscosityPredictorCNP(model_dir=str(self.extracted_path))
        Log.i(TAG, "Engine initialized successfully (legacy single-module path).")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run point-estimate inference on the given input features.

        Args:
            df (pd.DataFrame): Input features. Must not contain target viscosity
                columns — strip them with ``prepare_prediction_input`` first.

        Returns:
            pd.DataFrame: Predicted viscosity values with one column per shear rate.

        Raises:
            RuntimeError: If the engine has not been initialized.
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized")
        return self.engine.predict(df)

    def predict_with_uncertainty(
        self,
        df: pd.DataFrame,
        n_samples: int = 50,
        ci_range: Tuple[float, float] = (2.5, 97.5),
        k: int = 8,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Run inference with Monte-Carlo uncertainty estimation.

        Performs ``n_samples`` stochastic forward passes and aggregates statistics
        over the draws. Results are returned in both log10 (model-native) and linear
        cP spaces.

        The stats dict exposes two spellings of each key for backward compatibility:
        ``lower_ci`` / ``lower``, ``upper_ci`` / ``upper``, ``std_log10`` / ``std``.

        Args:
            df (pd.DataFrame): Input features (target columns must be removed).
            n_samples (int): Number of Monte-Carlo draws. Defaults to 50.
            ci_range (Tuple[float, float]): Lower and upper percentile bounds for
                the confidence interval (e.g. ``(2.5, 97.5)`` for 95 % CI).
                Defaults to ``(2.5, 97.5)``.
            k (int): Context subset size per draw; should match the few-shot elbow
                found during evaluation. Defaults to 8.

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: A 2-tuple of:
                - **mean_pred** – mean predictions in linear cP.
                - **stats** – dict containing:
                    - ``mean_log10``: mean prediction in log10 units.
                    - ``std_log10`` / ``std``: std in log10 units (0.1 ≈ ±26 % error).
                    - ``lower_ci`` / ``lower``: lower CI bound in linear cP.
                    - ``upper_ci`` / ``upper``: upper CI bound in linear cP.

        Raises:
            RuntimeError: If the engine has not been initialized.
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized")
        mean_pred, stats = self.engine.predict_with_uncertainty(
            df, n_samples=n_samples, ci_range=ci_range, k=k
        )

        # Normalise stats keys so callers that used the old 'lower'/'upper'
        # names still work — expose both spellings.
        stats.setdefault("lower", stats.get("lower_ci"))
        stats.setdefault("upper", stats.get("upper_ci"))
        stats.setdefault("std", stats.get("std_log10"))

        return mean_pred, stats

    def predict_uncertainty(self, *args, **kwargs):
        """Alias for :meth:`predict_with_uncertainty`.

        All positional and keyword arguments are forwarded unchanged.

        Args:
            *args: Positional arguments passed to :meth:`predict_with_uncertainty`.
            **kwargs: Keyword arguments passed to :meth:`predict_with_uncertainty`.

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: Same as
                :meth:`predict_with_uncertainty`.
        """
        return self.predict_with_uncertainty(*args, **kwargs)

    def evaluate(
        self,
        eval_data: pd.DataFrame,
        targets: list[str],
        n_samples: int | None = None,
        fine_tune: bool = False,
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Evaluate the model and return per-sample, per-shear-rate metrics.

        Supports optional per-protein-type fine-tuning using ``history_df``.
        When ``fine_tune`` is ``True`` and conditions are met, the engine is
        adapted for each unique ``Protein_type`` in turn, then reset to its
        base state before proceeding to the next type.

        Args:
            eval_data (pd.DataFrame): DataFrame containing input features and
                ground-truth target viscosity values.
            targets (list[str]): List of target column names (e.g.
                ``['Viscosity_100', 'Viscosity_1000']``).  Currently overridden
                internally to the full five-shear-rate set.
            n_samples (int | None): Accepted for UI signature compatibility;
                not used internally.
            fine_tune (bool): If ``True``, adapt the engine per ``Protein_type``
                before predicting that subset. Defaults to ``False``.
            history_df (pd.DataFrame | None): Historical labelled data used as
                adaptation context when ``fine_tune`` is ``True``. Must contain
                a ``Protein_type`` column. Defaults to ``None``.

        Returns:
            pd.DataFrame: Long-format results with columns ``sample_idx``,
                ``shear_rate``, ``actual``, ``predicted``, ``pct_error``,
                ``abs_error``, ``lower_ci``, ``upper_ci``.  Returns an empty
                DataFrame with those columns if no predictions were produced.
        """
        shear_rates = [100, 1000, 10000, 100000, 15000000]

        targets = [
            "Viscosity_100",
            "Viscosity_1000",
            "Viscosity_10000",
            "Viscosity_100000",
            "Viscosity_15000000",
        ]

        def calc_metrics(actuals_df, pred_df, unc_dict):
            """Compute per-row, per-target error metrics.

            Args:
                actuals_df (pd.DataFrame): Ground-truth values; must contain the
                    target columns defined in the enclosing scope.
                pred_df (pd.DataFrame): Predicted values with matching columns and
                    row alignment to ``actuals_df``.
                unc_dict (dict): Uncertainty statistics from
                    :meth:`predict_with_uncertainty`.  May be empty.

            Returns:
                pd.DataFrame: Long-format DataFrame with columns ``sample_idx``,
                    ``shear_rate``, ``actual``, ``predicted``, ``pct_error``,
                    ``abs_error``, ``lower_ci``, ``upper_ci``.
            """
            results = []
            for idx, target in enumerate(targets):
                if target not in actuals_df.columns:
                    continue
                if target not in pred_df.columns:
                    continue
                for row_idx in range(len(actuals_df)):
                    actual = actuals_df.iloc[row_idx][target]
                    predicted = pred_df.iloc[row_idx][target]
                    abs_err = abs(predicted - actual)
                    pct_err = abs_err / abs(actual) * 100 if abs(actual) > 1e-9 else 0.0
                    lower = (
                        unc_dict.get("lower_ci", [None] * len(actuals_df))[row_idx]
                        if "lower_ci" in unc_dict
                        else None
                    )
                    upper = (
                        unc_dict.get("upper_ci", [None] * len(actuals_df))[row_idx]
                        if "upper_ci" in unc_dict
                        else None
                    )
                    if isinstance(lower, np.ndarray):
                        lower = lower[idx] if idx < len(lower) else None
                    if isinstance(upper, np.ndarray):
                        upper = upper[idx] if idx < len(upper) else None

                    results.append(
                        {
                            "sample_idx": row_idx,
                            "shear_rate": shear_rates[idx],
                            "actual": actual,
                            "predicted": predicted,
                            "pct_error": pct_err,
                            "abs_error": abs_err,
                            "lower_ci": lower,
                            "upper_ci": upper,
                        }
                    )
            return pd.DataFrame(results)

        def prepare_prediction_input(df, targets):
            """Strip target viscosity columns from a DataFrame before prediction.

            Args:
                df (pd.DataFrame): Input DataFrame that may contain target columns.
                targets (list[str]): Unused; target columns are removed by hard-coded
                    names for safety.

            Returns:
                pd.DataFrame: Copy of ``df`` with all viscosity target columns removed.
            """
            df_out = df.copy()
            df_out.drop(
                columns=[
                    "Viscosity_100",
                    "Viscosity_1000",
                    "Viscosity_10000",
                    "Viscosity_100000",
                    "Viscosity_15000000",
                ],
                inplace=True,
                errors="ignore",
            )
            return df_out

        results_list = []

        should_fine_tune = (
            fine_tune
            and history_df is not None
            and not history_df.empty
            and "Protein_type" in history_df.columns
        )

        if should_fine_tune and "Protein_type" not in eval_data.columns:
            Log.w(
                TAG,
                "Calibration requested but 'Protein_type' missing in eval_data. Using base model.",
            )
            should_fine_tune = False

        if should_fine_tune:
            processed_indices = set()
            unique_types = eval_data["Protein_type"].unique()
            eval_types = (
                eval_data["Protein_type"].unique() if "Protein_type" in eval_data.columns else []
            )
            all_types = set(unique_types).union(eval_types)

            Log.i(TAG, f"Starting evaluation with fine-tuning on {len(all_types)} types.")

            for p_type in all_types:
                hist_subset = history_df[history_df["Protein_type"] == p_type]
                eval_subset = eval_data[eval_data["Protein_type"] == p_type]

                if eval_subset.empty:
                    continue

                Log.i(
                    TAG,
                    f"Fine-tuning on protein type: {p_type} (History: {len(hist_subset)} + Eval: {len(eval_subset)})",
                )

                combined_learning_set = pd.concat([hist_subset, eval_subset], ignore_index=True)
                context_for_learn = combined_learning_set.copy()
                if self.model_type == "CNP" and hasattr(self.engine, "_select_diverse_context"):
                    context_for_learn = self.engine._select_diverse_context(
                        context_for_learn, max_k=15
                    )
                    Log.d(
                        TAG,
                        f"  Diverse context for {p_type}: "
                        f"{len(combined_learning_set)} → {len(context_for_learn)} samples",
                    )
                self.learn(context_for_learn)

                try:
                    pred_input = prepare_prediction_input(eval_subset, targets)
                    preds = self.predict(pred_input)

                    try:
                        _, unc_dict = self.predict_with_uncertainty(pred_input)
                    except Exception:
                        unc_dict = {}

                    metrics = calc_metrics(eval_subset, preds, unc_dict)
                    results_list.append(metrics)
                    processed_indices.update(eval_subset.index)
                except Exception as e:
                    Log.e(TAG, f"Prediction failed for type {p_type}: {e}")

                self.reset()

            remaining_eval = eval_data.drop(index=list(processed_indices), errors="ignore")
            if not remaining_eval.empty:
                Log.i(TAG, f"Predicting remaining {len(remaining_eval)} samples with base model.")
                pred_input = prepare_prediction_input(remaining_eval, targets)
                preds = self.predict(pred_input)
                try:
                    _, unc_dict = self.predict_with_uncertainty(pred_input)
                except Exception:
                    unc_dict = {}
                results_list.append(calc_metrics(remaining_eval, preds, unc_dict))

        else:
            # Standard evaluation (no fine-tuning)
            pred_input = prepare_prediction_input(eval_data, targets)
            preds = self.predict(pred_input)
            try:
                _, unc_dict = self.predict_with_uncertainty(pred_input)
            except Exception:
                unc_dict = {}

            results_list.append(calc_metrics(eval_data, preds, unc_dict))

        if not results_list:
            return pd.DataFrame(
                columns=[
                    "sample_idx",
                    "shear_rate",
                    "actual",
                    "predicted",
                    "pct_error",
                    "abs_error",
                    "lower_ci",
                    "upper_ci",
                ]
            )

        return pd.concat(results_list, ignore_index=True)

    def learn(
        self,
        df: pd.DataFrame,
        lr: float = 1e-3,
        steps: int = 50,
        n_draws: int = 20,
        k: int = 8,
    ):
        """Adapt the model to a new protein group by encoding its context.

        For CNP engines, delegates to the engine's encode-only ``learn()`` — no
        gradient updates are performed on the model weights.  ``steps`` and ``lr``
        are accepted for backward compatibility but are ignored by CNP engines.
        Multi-draw averaging is controlled by ``n_draws`` and ``k``.

        For non-CNP engines this is a no-op; legacy models do not support
        in-session adaptation.

        Args:
            df (pd.DataFrame): Labelled samples representing the new protein group.
                Must include both feature columns and target viscosity columns.
            lr (float): Learning rate placeholder (ignored for CNP). Defaults to 1e-3.
            steps (int): Optimisation steps placeholder (ignored for CNP). Defaults to 50.
            n_draws (int): Number of Monte-Carlo draws used when encoding context.
                Defaults to 20.
            k (int): Context subset size per draw. Defaults to 8.
        """
        if self.model_type != "CNP":
            return

        Log.i(TAG, f"Learning from {len(df)} new samples ")
        self.engine.learn(df, steps=steps, lr=lr, n_draws=n_draws, k=k)

    def reset(self):
        """Reset the model to its base state, discarding any adaptation from :meth:`learn`.

        For CNP engines, restores the pristine model weights from the in-memory
        snapshot saved at load time and clears the cached memory/context vectors.
        This operation is instant and does not touch the filesystem.

        For non-CNP or uninitialized engines, falls back to a full package reload
        (re-extract and re-init) for safety.
        """
        if (
            self.model_type == "CNP"
            and self.engine is not None
            and hasattr(self.engine, "_original_state")
        ):
            import copy

            Log.i(TAG, "Resetting engine state (fast in-place restore)...")
            self.engine.model.load_state_dict(copy.deepcopy(self.engine._original_state))
            self.engine.memory_vector = None
            self.engine.context_t = None
            Log.i(TAG, "Engine reset complete.")
            return

        # Legacy / non-CNP fallback: full package reload
        Log.i(TAG, "Resetting model state via full package reload...")
        if self.extracted_path.exists():
            for child in self.extracted_path.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)

        self.engine = None
        self._load_package()

    def save(self, output_path: str):
        """Persist the current model state (including any learned adaptations) to disk.

        Calls the engine's own ``save()`` to flush updated weights into the
        extracted directory, then re-zips the directory as a ``.visq`` file at
        ``output_path``.

        Args:
            output_path (str): Destination path for the saved package. The
                ``.visq`` extension is applied automatically regardless of the
                suffix provided.

        Raises:
            NotImplementedError: If ``self.model_type`` is not ``"CNP"``.
        """
        if self.model_type != "CNP":
            raise NotImplementedError(
                "Model package is not supported.  Only compatible `.visq` packages" "are supported."
            )

        if hasattr(self.engine, "save"):
            self.engine.save(output_dir=str(self.extracted_path))

        # We manually zip the EXTRACTED folder, which contains updated weights
        output = Path(output_path).with_suffix(".visq")
        Log.i(TAG, f"Saving updated model state to {output}...")

        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(self.extracted_path):
                for file in files:
                    abs_path = Path(root) / file
                    arc_name = abs_path.relative_to(self.extracted_path)
                    zf.write(abs_path, arc_name)
        Log.i(TAG, "Save complete.")

    def cleanup(self):
        """Release the temporary extraction directory.

        Safe to call multiple times; subsequent calls after the first are no-ops.
        Automatically called by :meth:`__exit__` when used as a context manager.
        """
        if self._tmpdir:
            try:
                self._tmpdir.cleanup()
            except Exception:
                pass
            self._tmpdir = None

    def __enter__(self):
        """Support use as a context manager.

        Returns:
            Predictor: The predictor instance itself.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources on context-manager exit.

        Delegates to :meth:`cleanup`. Exceptions from the body of the ``with``
        block are not suppressed.

        Args:
            exc_type (type | None): Exception type, or ``None`` if no exception.
            exc_val (BaseException | None): Exception instance, or ``None``.
            exc_tb (traceback | None): Traceback object, or ``None``.
        """
        self.cleanup()
