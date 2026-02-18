import os
import traceback

import numpy as np
import pandas as pd
from PyQt5 import QtCore

# --- IMPORTS ---
try:
    from architecture import Architecture
    from src.controller.formulation_controller import FormulationController

    # Imports for ICL Data Fetching
    from src.db.db import Database

    # Import Predictor from the location indicated by your file structure
    from src.models.predictor import Predictor
except (ImportError, ModuleNotFoundError):
    from QATCH.common.architecture import Architecture
    from QATCH.VisQAI.src.controller.formulation_controller import (
        FormulationController,
    )

    # Imports for ICL Data Fetching (Fallback)
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.models.predictor import Predictor


class PredictionThread(QtCore.QThread):
    """
    Background worker that loads the selected VisQ model, optionally performs
    In-Context Learning (ICL) on filtered historical data, and runs inference.
    """

    data_ready = QtCore.pyqtSignal(dict)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._is_running = True

    def run(self):
        """
        Executes the prediction logic.
        """
        if not self._is_running:
            return

        db_conn = None
        try:
            # 1. Resolve Model Path
            model_filename = self.config.get("model")
            assets_path = os.path.join(
                Architecture.get_path(), "QATCH", "VisQAI", "assets"
            )
            model_path = os.path.join(assets_path, model_filename)

            if not os.path.exists(model_path):
                print(f"[Worker] Model file not found: {model_path}.")
                return

            # 2. Prepare Input Data
            formulation = self.config.get("formulation_object")
            if not formulation:
                raise ValueError("Formulation object missing from configuration.")

            # Convert formulation to DataFrame (encoded=True for model input)
            df_input = formulation.to_dataframe(encoded=False, training=False)

            # 3. Determine Confidence Interval Parameters
            ci_percent = self.config.get("ci", 95)
            alpha = (100.0 - ci_percent) / 2.0
            ci_range = (alpha, 100.0 - alpha)

            # 4. In-Context Learning (ICL) Preparation
            # Check if ICL is requested and filters are provided
            icl_filter = self.config.get("icl_filter")
            learn_df = None
            print(f"ICL Filter was: {icl_filter}")
            if icl_filter and icl_filter.get("fields"):
                try:
                    # Create a temporary DB connection for this thread
                    db_conn = Database(parse_file_key=True)
                    form_ctrl = FormulationController(db_conn)

                    # Fetch and filter data
                    learn_df = self._fetch_icl_context(
                        form_ctrl, formulation, icl_filter
                    )
                except Exception as e:
                    print(f"[Worker] ICL Data Fetch Error: {e}")
                    # Don't fail the whole prediction if learning data fails
                    learn_df = None
                finally:
                    if db_conn:
                        db_conn.close()

            # 5. Run Inference using Predictor Context Manager
            with Predictor(model_path) as predictor:
                if not self._is_running:
                    return

                # --- STEP: LEARN ---
                if learn_df is not None and not learn_df.empty:
                    print(f"[Worker] Learning from {len(learn_df)} context samples...")
                    steps = self.config.get("steps", 50)
                    lr = self.config.get("lr", 0.01)
                    try:
                        predictor.learn(learn_df, steps=steps, lr=lr)
                    except Exception as e:
                        print(f"[Worker] Learning failed: {e}")
                else:
                    print(f"[Worker] Learning failed data was `{learn_df}`")

                # --- STEP: PREDICT ---
                means, unc_dict = predictor.predict_with_uncertainty(
                    df_input, n_samples=50, ci_range=ci_range
                )

                # 6. Process Results
                std_shear_rates = np.array(
                    [100, 1000, 10000, 100000, 15000000], dtype=float
                )

                # Flatten to ensure 1D array
                y_pred_full = np.asanyarray(means).flatten()

                # Take first 5 points (assuming single sample input)
                if y_pred_full.size >= 5:
                    y_pred = y_pred_full[:5]
                else:
                    print(
                        f"[Worker] Warning: Model returned {y_pred_full.size} points. Expected >= 5."
                    )
                    y_pred = np.resize(y_pred_full, 5)

                # Flatten Uncertainty Arrays
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

                # 7. Alignment & Interpolation
                requested_shear = self.config.get("shear_rates")

                if requested_shear is not None and len(requested_shear) > 0:
                    final_x = np.array(requested_shear, dtype=float)
                    final_y = np.interp(final_x, std_shear_rates, y_pred)
                    final_lower = np.interp(final_x, std_shear_rates, lower)
                    final_upper = np.interp(final_x, std_shear_rates, upper)
                else:
                    final_x = std_shear_rates
                    final_y = y_pred
                    final_lower = lower
                    final_upper = upper

                # 8. Package Data
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
            print(f"[Worker] Error: {e}")
            traceback.print_exc()
            self.error_occurred.emit(str(e))
        finally:
            if db_conn:
                try:
                    db_conn.close()
                except Exception:
                    pass

    def _fetch_icl_context(self, form_ctrl, current_formulation, icl_filter):
        """
        Fetches all formulations, compares them against the current formulation
        using the specified filter criteria, and returns a DataFrame of
        encoded matches suitable for learning.
        """
        filter_fields = icl_filter.get("fields", [])
        if not filter_fields:
            return None

        logic = icl_filter.get("logic", "AND")  # AND / OR

        # 1. Fetch all formulations from DB
        all_forms = form_ctrl.get_all_formulations()
        if not all_forms:
            return None

        # 2. Generate Human-Readable DataFrames for Filtering
        # We use encoded=False so we can compare strings like 'mAb', 'Histidine'
        # rather than opaque one-hot vectors.

        # Current formulation (Reference)
        df_curr_readable = current_formulation.to_dataframe(
            encoded=False, training=False
        )

        # History formulations (Candidates)
        # Note: We filter FIRST before converting everything to encoded,
        # because encoded conversion might be expensive or opaque.

        # We'll iterate objects for filtering to avoid creating a massive DF first
        matching_forms = []

        # Extract reference values from current formulation
        ref_values = {}
        for field in filter_fields:
            if field in df_curr_readable.columns:
                ref_values[field] = df_curr_readable.iloc[0][field]
            else:
                ref_values[field] = None

        for f in all_forms:
            # Skip if no viscosity profile (cannot learn without targets)
            if not f.viscosity_profile or not f.viscosity_profile.viscosities:
                continue

            # Quick conversion for comparison
            # (In a production system, you might cache these or query SQL directly)
            f_df = f.to_dataframe(encoded=False, training=False)

            matches = []
            for field in filter_fields:
                if field not in f_df.columns:
                    matches.append(False)
                    continue

                val = f_df.iloc[0][field]
                ref = ref_values.get(field)

                # Loose equality check (string comparison)
                matches.append(str(val) == str(ref))

            if not matches:
                continue

            # Apply Logic
            is_match = False
            if logic == "AND":
                is_match = all(matches)
            elif logic == "OR":
                is_match = any(matches)

            if is_match:
                matching_forms.append(f)

        if not matching_forms:
            print("[Worker] No historical data found matching ICL filter.")
            return None

        # 3. Convert Matches to Encoded Training Data
        # Now we need encoded=True and training=True (to include Viscosity targets)
        dfs = []
        for f in matching_forms:
            try:
                # training=True ensures target columns (Viscosity_100, etc.) are present
                d = f.to_dataframe(encoded=False, training=True)
                dfs.append(d)
            except Exception as e:
                print(f"[Worker] Skipped malformed historical record: {e}")

        if not dfs:
            return None

        final_learn_df = pd.concat(dfs, ignore_index=True)
        return final_learn_df

    def stop(self):
        """Stops the thread safely."""
        self._is_running = False
        self.quit()
        self.wait(1000)
