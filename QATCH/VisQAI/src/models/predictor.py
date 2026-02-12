"""
predictor.py

Provides the Predictor class for loading a packaged viscosity model
and performing inference.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-02-09

Version:
    6.5 (Relative Path Search Fix)
"""

import datetime
import importlib.util
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ==========================================
# Logger Setup
# ==========================================
try:
    from QATCH.common.logger import Logger as Log
except (ImportError, ModuleNotFoundError):
    import logging

    logging.basicConfig(level=logging.INFO)

    class Log:
        @classmethod
        def i(cls, msg):
            logging.info(msg)

        @classmethod
        def w(cls, msg):
            logging.warning(msg)

        @classmethod
        def e(cls, msg):
            logging.error(msg)

        @classmethod
        def d(cls, msg):
            logging.debug(msg)


# ==========================================
# Security Imports
# ==========================================
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
    """
    High-level interface for loading VisQ models.
    """

    def __init__(self, zip_path: str, temp_dir: Optional[str] = None):
        self.zip_path = Path(zip_path)
        self._tmpdir = tempfile.TemporaryDirectory(dir=temp_dir)
        self.extracted_path = Path(self._tmpdir.name)

        self.manifest = {}
        self.model_type = "unknown"
        self.engine = None

        self._load_package()

    def _load_package(self):
        """Unzips and dynamically loads the engine."""
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Package not found: {self.zip_path}")

        Log.i(f"Extracting package: {self.zip_path}")
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            zf.extractall(self.extracted_path)

        # ---------------------------------------------------------
        # 1. ROBUST PATH SEARCH for Local Code
        # ---------------------------------------------------------
        # Instead of guessing "src/...", we look relative to THIS file (predictor.py).
        current_dir = Path(__file__).resolve().parent

        candidate_paths = [
            # 1. Same directory (e.g. if everything is in src/)
            current_dir / "inference_o_net.py",
            # 2. Parent directory (e.g. if predictor is in src/io/)
            current_dir.parent / "inference_o_net.py",
            # 3. Two levels up + src (e.g. if predictor is in src/utils/)
            current_dir.parent.parent / "src" / "inference_o_net.py",
            # 4. Standard Project Root assumption
            Path("src/inference_o_net.py").resolve(),
        ]

        local_code_path = None
        for p in candidate_paths:
            if p.exists():
                local_code_path = p
                break

        # Define where the ZIPPED (stale) code lives
        extracted_code_path = self.extracted_path / "inference_o_net.py"
        inference_module = None

        if local_code_path and local_code_path.exists():
            inference_module = self._dynamic_load_from_path(
                "visq_inference_local", local_code_path
            )
        elif extracted_code_path.exists():
            inference_module = self._dynamic_load_from_path(
                "visq_inference_pkg", extracted_code_path
            )
        else:
            Log.w("No inference logic found. Attempting legacy fallback.")

        # 2. Initialize Engine
        if inference_module and hasattr(inference_module, "ViscosityPredictorCNP"):
            self.model_type = "CNP"
            # Initialize the class with the path to the WEIGHTS (extracted zip)
            # This uses the NEW code (inference_module) with the OLD weights (extracted_path)
            self.engine = inference_module.ViscosityPredictorCNP(
                model_dir=str(self.extracted_path)
            )
            Log.i("CNP Engine initialized successfully.")
            return

        # 3. Fallback to Secure Loader / Legacy if manual dynamic load failed/skipped
        if SECURITY_AVAILABLE:
            try:
                loader = create_secure_loader_for_extracted_package(
                    self.extracted_path, enforce_signatures=True
                )
                self.manifest = loader.load_manifest()
                arch = self.manifest.get("architecture", "LegacyEnsemble")

                if arch == "CrossSampleCNP":
                    # We already tried loading above, but if secure loader does something special:
                    mod = loader.load_inference_module()
                    self.model_type = "CNP"
                    self.engine = mod.ViscosityPredictorCNP(
                        model_dir=str(self.extracted_path)
                    )
                else:
                    self._load_legacy_engine(loader)
            except Exception as e:
                Log.e(f"Loader Error: {e}")
                raise e
        else:
            Log.e("Could not load inference engine.")

    def _dynamic_load_from_path(self, module_name, file_path):
        """Helper to load a python source file by absolute path."""
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(file_path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module
        except Exception as e:
            Log.e(f"Failed to dynamically load {file_path}: {e}")
        return None

    def _load_legacy_engine(self, loader):
        self.model_type = "Ensemble"
        # ... legacy code ...
        pass

    # ==========================================
    # Public API
    # ==========================================
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.engine is None:
            raise RuntimeError("Engine not initialized")
        return self.engine.predict(df)

    def predict_with_uncertainty(
        self,
        df: pd.DataFrame,
        n_samples: int = 20,
        ci_range: Tuple[float, float] = (2.5, 97.5),
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if self.engine is None:
            raise RuntimeError("Engine not initialized")

        return self.engine.predict_with_uncertainty(
            df, n_samples=n_samples, ci_range=ci_range
        )

    def predict_uncertainty(self, *args, **kwargs):
        return self.predict_with_uncertainty(*args, **kwargs)

    def evaluate(
        self,
        eval_data: pd.DataFrame,
        targets: list[str],
        n_samples: int = None,
        fine_tune: bool = False,
        history_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Evaluates the model on the provided data and returns a DataFrame of metrics.
        Supports optional fine-tuning per Protein_type using history_df.

        Args:
            eval_data: DataFrame containing input features and actual target values.
            targets: List of column names representing the targets (e.g. 'Viscosity_100').
            n_samples: Optional argument for consistency with UI signature.
            fine_tune: If True, performs fine-tuning per Protein_type found in history_df.
            history_df: DataFrame containing historical data for fine-tuning. Must contain
                        'Protein_type' column if fine_tune is True.

        Returns:
            pd.DataFrame: Long-format DataFrame containing 'sample_idx', 'shear_rate',
                          'actual', 'predicted', 'pct_error', 'abs_error',
                          'lower_ci', 'upper_ci'.
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized")

        # --- Helper for Metrics Calculation ---
        def calc_metrics(sub_df, sub_preds, sub_unc):
            # 1. Map canonical targets (e.g. 'Viscosity_100') to Prediction columns
            #    (e.g. 'Pred_Viscosity_100')
            target_map = {}  # Canonical Name -> Prediction Column Name

            for t in targets:
                if t in sub_preds.columns:
                    target_map[t] = t
                elif f"Pred_{t}" in sub_preds.columns:
                    target_map[t] = f"Pred_{t}"
                elif f"Predicted_{t}" in sub_preds.columns:
                    target_map[t] = f"Predicted_{t}"

            # The list of targets we can actually evaluate (canonical names)
            valid_targets = list(target_map.keys())

            # Base columns required by UI
            base_cols = [
                "sample_idx",
                "shear_rate",
                "actual",
                "predicted",
                "pct_error",
                "abs_error",
                "lower_ci",
                "upper_ci",
            ]

            if not valid_targets:
                return pd.DataFrame(columns=base_cols)

            # 2. Process Actuals (Ground Truth)
            # Use valid_targets to pull from input data (sub_df)
            y_true = sub_df.reindex(columns=valid_targets + ["sample_idx"])
            y_true["sample_idx"] = sub_df.index

            y_true_melt = y_true.melt(
                id_vars=["sample_idx"],
                value_vars=valid_targets,
                var_name="shear_rate",
                value_name="actual",
            )

            # 3. Process Predictions
            # Extract specific prediction columns and RENAME them to canonical names
            pred_cols = [target_map[t] for t in valid_targets]
            y_pred = sub_preds[pred_cols].copy()

            # Create rename dictionary: {'Pred_Viscosity_100': 'Viscosity_100'}
            rename_dict = {v: k for k, v in target_map.items()}
            y_pred = y_pred.rename(columns=rename_dict)

            y_pred["sample_idx"] = sub_preds.index
            y_pred_melt = y_pred.melt(
                id_vars=["sample_idx"], var_name="shear_rate", value_name="predicted"
            )

            # 4. Merge Actuals and Predictions
            # Use 'right' merge to keep predictions even if actuals are missing
            results = pd.merge(
                y_true_melt, y_pred_melt, on=["sample_idx", "shear_rate"], how="right"
            )

            # 5. Process Uncertainty (Lower/Upper CI)
            if "lower" in sub_unc and "upper" in sub_unc:
                try:
                    # Uncertainty arrays usually match prediction column structure
                    lower_df = pd.DataFrame(
                        sub_unc["lower"],
                        index=sub_preds.index,
                        columns=sub_preds.columns,
                    )
                    upper_df = pd.DataFrame(
                        sub_unc["upper"],
                        index=sub_preds.index,
                        columns=sub_preds.columns,
                    )

                    # Rename uncertainty columns to match canonical names using same map
                    lower_df = lower_df.rename(columns=rename_dict)
                    upper_df = upper_df.rename(columns=rename_dict)

                    # Filter to valid targets only
                    lower_df = lower_df.reindex(columns=valid_targets)
                    upper_df = upper_df.reindex(columns=valid_targets)

                    lower_df["sample_idx"] = lower_df.index
                    upper_df["sample_idx"] = upper_df.index

                    lower_melt = lower_df.melt(
                        id_vars=["sample_idx"],
                        var_name="shear_rate",
                        value_name="lower_ci",
                    )
                    upper_melt = upper_df.melt(
                        id_vars=["sample_idx"],
                        var_name="shear_rate",
                        value_name="upper_ci",
                    )

                    results = pd.merge(
                        results, lower_melt, on=["sample_idx", "shear_rate"], how="left"
                    )
                    results = pd.merge(
                        results, upper_melt, on=["sample_idx", "shear_rate"], how="left"
                    )

                except Exception as e:
                    Log.w(f"Failed to process uncertainty arrays: {e}")
                    results["lower_ci"] = np.nan
                    results["upper_ci"] = np.nan
            else:
                results["lower_ci"] = np.nan
                results["upper_ci"] = np.nan

            # 6. Calculate Errors
            results["error"] = results["predicted"] - results["actual"]
            results["abs_error"] = results["error"].abs()

            with np.errstate(divide="ignore", invalid="ignore"):
                results["pct_error"] = (
                    results["abs_error"] / np.abs(results["actual"])
                ) * 100

            results["pct_error"] = (
                results["pct_error"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            )

            # Ensure all required columns exist
            for col in base_cols:
                if col not in results.columns:
                    results[col] = np.nan

            return results

        # --- Helper: Prepare Prediction Input (Mask Targets) ---
        def prepare_prediction_input(df_in, target_cols):
            """Creates a copy of input and masks target columns with NaN."""
            df_out = df_in.copy()
            df_out.drop(
                columns=[
                    "Viscosity_100",
                    "Viscosity_1000",
                    "Viscosity_10000",
                    "Viscosity_100000",
                    "Viscosity_15000000",
                ],
                inplace=True,
            )
            return df_out

        # --- Main Evaluation Logic ---
        results_list = []

        should_fine_tune = (
            fine_tune
            and history_df is not None
            and not history_df.empty
            and "Protein_type" in history_df.columns
        )

        if should_fine_tune:
            if "Protein_type" not in eval_data.columns:
                Log.w(
                    "Fine-tune requested but 'Protein_type' missing in eval_data. Using base model."
                )
                should_fine_tune = False

        if should_fine_tune:
            processed_indices = set()
            unique_types = eval_data["Protein_type"].unique()
            eval_types = (
                eval_data["Protein_type"].unique()
                if "Protein_type" in eval_data.columns
                else []
            )
            all_types = set(unique_types).union(eval_types)

            Log.i(f"Starting evaluation with fine-tuning on {len(all_types)} types.")

            for p_type in all_types:
                hist_subset = history_df[history_df["Protein_type"] == p_type]
                eval_subset = eval_data[eval_data["Protein_type"] == p_type]

                if eval_subset.empty:
                    continue

                Log.i(
                    f"Fine-tuning on protein type: {p_type} (History: {len(hist_subset)} + Eval: {len(eval_subset)})"
                )

                # -------------------------------------------------------------
                # UPDATE: Merge history and eval samples for learning context
                # -------------------------------------------------------------
                combined_learning_set = pd.concat(
                    [hist_subset, eval_subset], ignore_index=True
                )
                self.learn(combined_learning_set.copy())
                # -------------------------------------------------------------

                try:
                    # For prediction, we MUST mask the targets
                    pred_input = prepare_prediction_input(eval_subset, targets)
                    preds = self.predict(pred_input)

                    try:
                        _, unc_dict = self.predict_with_uncertainty(pred_input)
                    except Exception:
                        unc_dict = {}

                    # Use original eval_subset (with answers) for metric calc
                    metrics = calc_metrics(eval_subset, preds, unc_dict)
                    results_list.append(metrics)
                    processed_indices.update(eval_subset.index)
                except Exception as e:
                    Log.e(f"Prediction failed for type {p_type}: {e}")

                self.reset()

            remaining_eval = eval_data.drop(
                index=list(processed_indices), errors="ignore"
            )
            if not remaining_eval.empty:
                Log.i(
                    f"Predicting remaining {len(remaining_eval)} samples with base model."
                )
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

    def learn(self, df: pd.DataFrame, fine_tune: bool = True, steps: int = 50):
        if self.model_type != "CNP":
            return
        Log.i(f"Learning from {len(df)} new samples...")
        self.engine.learn(df, steps=50, lr=1e-3)

    def reset(self):
        """
        Resets the model to its original state from the zip package.
        This effectively discards any fine-tuning performed via learn().
        """
        Log.i("Resetting model state to original package version...")
        if self.extracted_path.exists():
            for child in self.extracted_path.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)

        self.engine = None
        self._load_package()

    def save(self, output_path: str):
        if self.model_type != "CNP":
            raise NotImplementedError("Only CNP supports save.")

        if hasattr(self.engine, "save"):
            self.engine.save(output_dir=str(self.extracted_path))

        # We manually zip the EXTRACTED folder, which contains updated weights
        output = Path(output_path).with_suffix(".visq")
        Log.i(f"Saving updated model state to {output}...")

        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(self.extracted_path):
                for file in files:
                    abs_path = Path(root) / file
                    arc_name = abs_path.relative_to(self.extracted_path)
                    zf.write(abs_path, arc_name)
        Log.i("Save complete.")

    def cleanup(self):
        if self._tmpdir:
            try:
                self._tmpdir.cleanup()
            except Exception:
                pass
            self._tmpdir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
