"""
prediction_worker.py

Background worker for machine learning inference and In-Context Learning (ICL).

This module contains the PredictionThread class, which handles the complex
lifecycle of a VisQAI prediction. This includes resolving model assets,
fetching historical context for ICL, executing model inference with
uncertainty estimation, and interpolating results to match UI requirements.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

import os
import traceback

from PyQt5 import QtCore
import numpy as np
import pandas as pd

try:
    TAG = "[PredictionWorker (HEADLESS)]"

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

    from architecture import Architecture
    from src.controller.formulation_controller import FormulationController
    from src.db.db import Database
    from src.models.predictor import Predictor
except (ImportError, ModuleNotFoundError):
    TAG = "[PredictionWorker]"
    from QATCH.VisQAI.src.controller.formulation_controller import FormulationController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.common.architecture import Architecture
    from QATCH.common.logger import Logger as Log


class PredictionThread(QtCore.QThread):
    """Handles VisQ model loading, optional ICL, and inference in a background thread.

    This thread prepares formulation data, optionally queries the database for
    similar historical records to perform In-Context Learning, and runs
    prediction with confidence intervals. Results are interpolated to provide
    a smooth curve for the UI.

    Attributes:
        data_ready (QtCore.pyqtSignal): Emits a dictionary containing processed
            prediction data (x, y, upper, lower, etc.).
        error_occurred (QtCore.pyqtSignal): Emits a string error message if
            the inference fails.
        config (dict): Configuration mapping containing model info,
            formulation objects, ICL settings, and UI metadata.
        _is_running (bool): Flag used to support graceful thread termination.
    """

    data_ready = QtCore.pyqtSignal(dict)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, config):
        """Initializes the prediction thread with a configuration dictionary.

        Args:
            config (dict): Must contain 'model' (filename), 'formulation_object',
                and optionally 'icl_filter', 'ci', and 'shear_rates'.
        """
        super().__init__()
        self.config = config
        self._is_running = True

    def run(self):
        """Executes the prediction logic.

        The execution flow involves:
            1. Resolving the absolute path for the .visq model asset.
            2. Converting the input formulation into a model-ready DataFrame.
            3. Fetching historical context from the database if ICL is enabled.
            4. Initializing the `Predictor` and optionally running `predictor.learn()`.
            5. Performing inference to get mean viscosities and uncertainty bounds.
            6. Interpolating the 5-point standard output to the user's requested
               shear rates.
            7. Emitting the processed data package to the UI.
        """
        if not self._is_running:
            return

        db_conn = None
        try:
            model_filename = self.config.get("model")
            assets_path = os.path.join(Architecture.get_path(), "QATCH", "VisQAI", "assets")
            model_path = os.path.join(assets_path, model_filename)

            if not os.path.exists(model_path):
                Log.w(TAG, f"Model file not found: {model_path}.")
                return
            formulation = self.config.get("formulation_object")
            if not formulation:
                raise ValueError("Formulation object missing from configuration.")
            df_input = formulation.to_dataframe(encoded=False, training=False)

            # Determine Confidence Interval Parameters
            ci_percent = self.config.get("ci", 95)
            alpha = (100.0 - ci_percent) / 2.0
            ci_range = (alpha, 100.0 - alpha)

            # Check if ICL is requested and filters are provided
            icl_filter = self.config.get("icl_filter")
            learn_df = None
            Log.i(TAG, f"ICL Filter was: {icl_filter}")
            if icl_filter and icl_filter.get("fields"):
                try:
                    db_conn = Database(parse_file_key=True)
                    form_ctrl = FormulationController(db_conn)
                    learn_df = self._fetch_icl_context(form_ctrl, formulation, icl_filter)
                except Exception as e:
                    Log.e(TAG, f"ICL Data Fetch Error: {e}")
                    learn_df = None
                finally:
                    if db_conn:
                        db_conn.close()

            # Run Inference using Predictor context manager
            with Predictor(model_path) as predictor:
                if not self._is_running:
                    return

                # ICL Step
                if learn_df is not None and not learn_df.empty:
                    Log.i(TAG, f"Learning from {len(learn_df)} context samples...")
                    steps = self.config.get("steps", 50)
                    lr = self.config.get("lr", 0.01)
                    try:
                        predictor.learn(learn_df, steps=steps, lr=lr)
                    except Exception as e:
                        Log.e(TAG, f"ICL failed: {e}")

                # Predict Step
                means, unc_dict = predictor.predict_with_uncertainty(
                    df_input, n_samples=50, ci_range=ci_range
                )
                std_shear_rates = np.array([100, 1000, 10000, 100000, 15000000], dtype=float)
                y_pred_full = np.asanyarray(means).flatten()
                if y_pred_full.size >= 5:
                    y_pred = y_pred_full[:5]
                else:
                    Log.w(
                        TAG,
                        f"Model returned {y_pred_full.size} points. Expected 5.",
                    )
                    y_pred = np.resize(y_pred_full, 5)
                lower, upper = None, None

                def get_flat_slice(d, key):
                    arr = d.get(key)
                    if arr is None:
                        return None
                    flat = np.asanyarray(arr).flatten()
                    return flat[:5] if flat.size >= 5 else np.resize(flat, 5)

                if "lower" in unc_dict:
                    lower = get_flat_slice(unc_dict, "lower")
                elif "lower_ci" in unc_dict:
                    lower = get_flat_slice(unc_dict, "lower_ci")

                if "upper" in unc_dict:
                    upper = get_flat_slice(unc_dict, "upper")
                elif "upper_ci" in unc_dict:
                    upper = get_flat_slice(unc_dict, "upper_ci")

                if lower is None:
                    lower = y_pred
                if upper is None:
                    upper = y_pred
                requested_shear = self.config.get("shear_rates")

                if requested_shear is not None and len(requested_shear) > 0:
                    final_x = np.array(requested_shear, dtype=float)
                    std_shear_float = np.array(std_shear_rates, dtype=float)
                    y_pred_float = np.array(y_pred, dtype=float)
                    lower_float = np.array(lower, dtype=float)
                    upper_float = np.array(upper, dtype=float)

                    final_y = np.interp(final_x, std_shear_float, y_pred_float)
                    final_lower = np.interp(final_x, std_shear_float, lower_float)
                    final_upper = np.interp(final_x, std_shear_float, upper_float)
                else:
                    final_x = np.array(std_shear_rates, dtype=float)
                    final_y = np.array(y_pred, dtype=float)
                    final_lower = np.array(lower, dtype=float)
                    final_upper = np.array(upper, dtype=float)
                data_package = {
                    "x": final_x.tolist(),
                    "y": final_y.tolist(),
                    "upper": final_upper.tolist(),
                    "lower": final_lower.tolist(),
                    "measured_y": None,
                    "config_name": self.config.get("name", "Unknown"),
                    "color": self.config.get("color"),
                }

                if self._is_running:
                    self.data_ready.emit(data_package)

        except Exception as e:
            Log.e(TAG, f"Error: {e}")
            traceback.print_exc()
            self.error_occurred.emit(str(e))
        finally:
            if db_conn:
                try:
                    db_conn.close()
                except Exception:
                    pass

    def _fetch_icl_context(self, form_ctrl, current_formulation, icl_filter):
        """Fetches historical records for In-Context Learning based on filter criteria.

        Args:
            form_ctrl (FormulationController): Controller to fetch formulations.
            current_formulation (Formulation): The formulation being predicted.
            icl_filter (dict): Dictionary containing 'fields' (list of columns)
                and 'logic' ('AND'/'OR').

        Returns:
            pd.DataFrame or None: A DataFrame of historical records suitable for
                the `predictor.learn` method, or None if no matches are found.
        """
        filter_fields = icl_filter.get("fields", [])
        if not filter_fields:
            return None

        logic = icl_filter.get("logic", "AND")  # AND / OR
        all_forms = form_ctrl.get_all_formulations()
        if not all_forms:
            return None
        df_curr_readable = current_formulation.to_dataframe(encoded=False, training=False)

        matching_forms = []
        ref_values = {}
        for field in filter_fields:
            if field in df_curr_readable.columns:
                ref_values[field] = df_curr_readable.iloc[0][field]
            else:
                ref_values[field] = None

        for f in all_forms:
            if not getattr(f, "_icl", False):
                continue
            if not f.viscosity_profile or not f.viscosity_profile.viscosities:
                continue
            f_df = f.to_dataframe(encoded=False, training=False)

            matches = []
            for field in filter_fields:
                if field not in f_df.columns:
                    matches.append(False)
                    continue
                val = f_df.iloc[0][field]
                ref = ref_values.get(field)
                matches.append(str(val) == str(ref))
            if not matches:
                continue
            is_match = False
            if logic == "AND":
                is_match = all(matches)
            elif logic == "OR":
                is_match = any(matches)

            if is_match:
                matching_forms.append(f)

        if not matching_forms:
            Log.w(TAG, "No contextual data found matching ICL filter and flag.")
            return None

        dfs = []
        for f in matching_forms:
            try:
                d = f.to_dataframe(encoded=False, training=True)
                dfs.append(d)
            except Exception as e:
                Log.e(TAG, "Skipped malformed historical record: {e}")

        if not dfs:
            return None

        final_learn_df = pd.concat(dfs, ignore_index=True)
        return final_learn_df

    def stop(self):
        """Safely stops the background thread by setting the running flag."""
        self._is_running = False
        self.quit()
        self.wait(1000)
