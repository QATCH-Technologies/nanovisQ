#!/usr/bin/env python3
"""
q_model_predictor.py

Provides the QModelPredictor class for predicting Points of Interest (POIs) in dissipation
data using a pre-trained XGBoost booster and an sklearn scaler pipeline. Includes methods for
file validation, feature extraction, probability formatting, and bias correction for refined
POI selection.

Author: Paul MacNichol (paul.macnichol@qatchtech.com)
Date: 04-24-2025
Version: QModel.Ver3.1
"""

import xgboost as xgb

from sklearn.pipeline import Pipeline
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from QATCH.common.logger import Logger as Log
from QATCH.QModel.src.models.static_v3.q_model_data_processor import QDataProcessor
from QATCH.models.ModelData import ModelData
from QATCH.QModel.src.models.static_v2.q_image_clusterer import QClusterer
from QATCH.QModel.src.models.static_v2.q_multi_model import QPredictor
from QATCH.common.architecture import Architecture


class QModelPredictor:
    """
    Predictor for POI indices based on dissipation data using a pre-trained XGBoost booster and scaler.

    Attributes:
        _booster (xgb.Booster): Loaded XGBoost model for prediction.
        _scaler (Pipeline): Scaler pipeline for feature normalization.
    """

    def __init__(self, booster_path: str, scaler_path: str) -> None:
        """
        Initialize the QModelPredictor with model and scaler paths.

        Args:
            booster_path (str): Filesystem path to the XGBoost booster model (JSON).
            scaler_path (str): Filesystem path to the pickled scaler pipeline.

        Raises:
            Logging errors if provided paths are invalid or loading fails.
        """
        if booster_path is None or booster_path == "" or not os.path.exists(booster_path):
            Log.e(
                f'Booster path `{booster_path}` is empty string or does not exist.')
        if scaler_path is None or scaler_path == "" or not os.path.exists(scaler_path):
            Log.e(
                f'Scaler path `{scaler_path}` is empty string or does not exist.')

        self._booster: xgb.Booster = xgb.Booster()
        self._scaler: Pipeline = None
        try:
            self._booster.load_model(fname=booster_path)
            Log.i(f'Booster loaded from path `{booster_path}`.')
        except Exception as e:
            Log.e(
                f'Error loading booster from path `{booster_path}` with exception: `{e}`')
        try:
            self._scaler: Pipeline = self._load_scaler(scaler_path=scaler_path)
            Log.i(f'Scaler loaded from path `{scaler_path}`.')
        except Exception as e:
            Log.e(
                f'Error loading model from path `{scaler_path}` with exception: `{e}`')

    def _load_scaler(self, scaler_path: str) -> Pipeline:
        """
        Load a scaler object from a pickle file.

        Args:
            scaler_path (str): Path to the pickled scaler file.

        Returns:
            Pipeline: The loaded sklearn Pipeline scaler.

        Raises:
            IOError: If the scaler could not be loaded or is None.
        """
        scaler = None
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        if scaler is None:
            raise IOError("Scaler could not be loaded from specified path.")
        return scaler

    def _filter_labels(self, predicted_labels: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
        """
        Filter outlier class labels based on median absolute deviation (MAD).

        Args:
            predicted_labels (np.ndarray): Array of integer class labels.
            multiplier (float): Factor to scale the MAD threshold (default: 1.5).

        Returns:
            np.ndarray: Filtered labels with outliers set to 0.
        """
        filtered_labels = predicted_labels.copy()
        for cls in range(1, 7):
            class_indices = np.where(predicted_labels == cls)[0]
            if len(class_indices) == 0:
                continue
            centroid = np.median(class_indices)
            distances = np.abs(class_indices - centroid)
            mad = np.median(distances)
            threshold = multiplier * (mad if mad > 0 else 1)
            invalid_indices = class_indices[distances > threshold]
            filtered_labels[invalid_indices] = 0
        return filtered_labels

    def _get_model_data_predictions(self, file_buffer: str):
        """
        Obtain initial POI point predictions from the ModelData module.

        Args:
            file_buffer (str or file-like): Path to CSV file or file-like buffer.

        Returns:
            List[int]: List of POI indices predicted by ModelData.
        """
        model = ModelData()
        if isinstance(file_buffer, str):
            model_data_predictions = model.IdentifyPoints(file_buffer)
        else:
            file_buffer = self._reset_file_buffer(file_buffer)
            header = next(file_buffer)
            if isinstance(header, bytes):
                header = header.decode()
            csv_cols = (2, 4, 6, 7) if "Ambient" in header else (2, 3, 5, 6)
            file_data = np.loadtxt(
                file_buffer.readlines(), delimiter=",", usecols=csv_cols)
            relative_time = file_data[:, 0]
            resonance_frequency = file_data[:, 2]
            data = file_data[:, 3]
            model_data_predictions = model.IdentifyPoints(
                data_path="QModel Passthrough",
                times=relative_time,
                freq=resonance_frequency,
                diss=data
            )
        model_data_points = []
        if isinstance(model_data_predictions, list):
            for pt in model_data_predictions:
                if isinstance(pt, int):
                    model_data_points.append(pt)
                elif isinstance(pt, list) and pt:
                    model_data_points.append(max(pt, key=lambda x: x[1])[0])
        return model_data_points

    def _get_qmodel_v2_predictions(self, file_buffer: str):

        qmodel_v2_points = []
        cluster_model_path = os.path.join(
            Architecture.get_path(),
            "QATCH", "QModel", "SavedModels", "qmodel_v2",
            "cluster.joblib"
        )
        self.QModel_clusterer = QClusterer(
            model_path=cluster_model_path)

        predict_model_path = os.path.join(
            Architecture.get_path(),
            "QATCH", "QModel", "SavedModels", "qmodel_v2",
            "QMultiType_{}.json",
        )

        clusterer = QClusterer(model_path=cluster_model_path)

        try:
            label = clusterer.predict_label(file_buffer)
        except Exception as e:
            label = 0
            Log.w(
                f'Failed to retrieve clustering in static_v3, using classifier (0): {e}.')
        file_buffer.seek(0)
        QModel_v2_predictor = None
        try:
            if label == 0:
                QModel_v2_predictor = QPredictor(
                    model_path=predict_model_path.format(0)
                )
            elif label == 1:
                QModel_v2_predictor = QPredictor(
                    model_path=predict_model_path.format(1)
                )

            else:
                QModel_v2_predictor = QPredictor(
                    model_path=predict_model_path.format(2)
                )

        except:
            Log.w(
                f'Failed to set predictor in static_v3, using classifier (0): {e}.')
            QModel_v2_predictor = QPredictor(
                model_path=predict_model_path.format(0)
            )

        try:
            print(file_buffer)
            candidates = QModel_v2_predictor.predict(
                file_buffer=file_buffer, run_type=label)
            for c in candidates:
                qmodel_v2_points.append(c[0][0])
            return qmodel_v2_points
        except Exception as e:
            Log.w(
                f'Failed to predict in static_v3, returning ModelData positions: {e}.')
            return self._get_model_data_predictions(file_buffer=file_buffer)

    def _reset_file_buffer(self, file_buffer: str):
        """
        Reset a file-like buffer to its beginning if seekable.

        Args:
            file_buffer (str or file-like): The file path or buffer to reset.

        Returns:
            file_buffer: The reset buffer or original string path.

        Raises:
            Exception: If the buffer is not seekable.
        """
        if isinstance(file_buffer, str):
            return file_buffer
        if hasattr(file_buffer, "seekable") and file_buffer.seekable():
            file_buffer.seek(0)
            return file_buffer
        else:
            raise Exception(
                "Cannot `seek` stream prior to passing to processing.")

    def _validate_file_buffer(self, file_buffer: str) -> pd.DataFrame:
        """
        Validate and parse CSV data from a file buffer into a DataFrame.

        Args:
            file_buffer (str or file-like): Path or buffer containing CSV data.

        Returns:
            pd.DataFrame: Parsed DataFrame with required columns.

        Raises:
            ValueError: If the buffer is invalid, empty, or missing required columns.
        """
        try:
            file_buffer = self._reset_file_buffer(file_buffer=file_buffer)
        except Exception as e:
            raise ValueError(
                f"File buffer must be a non-empty string containing CSV data.")

        try:
            df = pd.read_csv(file_buffer)
        except pd.errors.EmptyDataError:
            raise ValueError("The provided data file is empty.")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing data file: `{str(e)}`")
        except Exception as e:
            raise ValueError(
                f"An unexpected error occurred while reading the data file: `{str(e)}`")

        if df.empty:
            raise ValueError("The data file does not contain any data.")

        required_columns = {"Dissipation",
                            "Resonance_Frequency", "Relative_time"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(
                f"Data file missing required columns: `{', '.join(missing)}`.")
        return df

    def _extract_predictions(self, predicted_probablities: np.ndarray, model_data_labels: np.ndarray) -> dict:
        """
        Convert model output probabilities into POI index candidates with confidences.

        Args:
            predicted_probablities (np.ndarray): Array of shape (n_samples, n_classes).
            model_data_labels (np.ndarray): Ground-truth label indices from ModelData.

        Returns:
            dict: Mapping of POI keys to dicts with 'indices' and 'confidences'.
        """
        poi_results = {}
        predicted_labels = np.argmax(predicted_probablities, axis=1)
        for poi in range(1, 7):
            key = f"POI{poi}"
            indices = np.where(predicted_labels == poi)[0]
            if indices.size == 0:
                poi_results[key] = {"indices": [
                    model_data_labels[poi - 1]], "confidences": [0]}
            else:
                confidences = predicted_probablities[indices, poi]
                sort_order = np.argsort(-confidences)
                sorted_indices = indices[sort_order].tolist()
                sorted_confidences = confidences[sort_order].tolist()

                poi_results[key] = {"indices": sorted_indices,
                                    "confidences": sorted_confidences}
        return poi_results

    def _correct_bias(self, dissipation: np.ndarray, relative_time: np.ndarray, candidates: list) -> int:
        """
        Adjust candidate selection by identifying steep slope onset and choosing nearest base.

        Args:
            dissipation (np.ndarray): Dissipation time series data.
            relative_time (np.ndarray): Corresponding time points.
            candidates (list): Candidate indices to choose from.

        Returns:
            int: Index of the best candidate after bias correction.
        """
        t = relative_time
        # 1) compute slope
        slope = np.gradient(dissipation, t)

        # 2) pick a threshold for “steep” — e.g. 90th percentile of slopes
        thresh = np.percentile(slope, 95)

        # 3) find peaks in the slope (these mark the on‑ramps of your jumps)
        peaks, props = find_peaks(slope, height=thresh)

        # 4) for each slope‑peak, walk backwards to find where slope ≺ 0 (the base)
        bases = []
        for p in peaks:
            j = p
            # walk back until slope is non‑positive (or you hit the start)
            while j > 0 and slope[j] > 0:
                j -= 1
            bases.append(j)

        if not bases:
            Log.w(
                "No steep jump detected!  Try lowering your threshold.")
            return candidates[0]
        if len(candidates) <= 0:
            Log.w(
                "Candidates list is empty")
            return candidates

        def min_dist_to_bases(cand):
            return min(abs(cand - b) for b in bases)

        best_idx = min(candidates, key=min_dist_to_bases)
        return best_idx

    def _select_best_predictions(
        self,
        predictions: dict,
        ground_truth: list,
        relative_time: np.ndarray,
        dissipation: np.ndarray,
    ) -> dict:
        """
        Refine predictions for POI4-6 by timing windows and bias correction.

        Args:
            predictions (dict): Initial POI prediction candidates.
            ground_truth_predictions (list): Ground-truth indices from ModelData or qmodel_v2.
            relative_time (np.ndarray): Time points for the data.
            dissipation (np.ndarray): Dissipation values over time.

        Returns:
            dict: Updated POI predictions with ordered 'indices' lists.
        """
        # make a shallow copy so we don't clobber the original dict
        best_positions = {k: {"indices": v["indices"][:], "confidences": v.get("confidences")}
                          for k, v in predictions.items()}

        # 1) define your [POI3, POI6] window and its thirds
        t0 = None
        if len(ground_truth) > 2:
            idx3 = ground_truth[2]
            if isinstance(idx3, (int, np.integer)) and 0 <= idx3 < len(relative_time):
                t0 = relative_time[idx3]

        # Fallback: first POI3 candidate
        if t0 is None:
            cand3 = predictions.get("POI3", {}).get("indices", [])
            if cand3 and isinstance(cand3[0], (int, np.integer)) and 0 <= cand3[0] < len(relative_time):
                t0 = relative_time[cand3[0]]

        # Last‐resort: start of the curve
        if t0 is None:
            t0 = relative_time[0]

        # --- SAFE t3 (POI6) ---
        # Collect all valid sources, then pick the earliest (min) time
        times = []

        # 1) model_data_labels[5]
        if len(ground_truth) > 5:
            idx6 = ground_truth[5]
            if isinstance(idx6, (int, np.integer)) and 0 <= idx6 < len(relative_time):
                times.append(relative_time[idx6])

        # 2) first POI6 candidate
        cand6 = predictions.get("POI6", {}).get("indices", [])
        if cand6 and isinstance(cand6[0], (int, np.integer)) and 0 <= cand6[0] < len(relative_time):
            times.append(relative_time[cand6[0]])

        # 3) fallback to end of curve if nothing valid
        t3 = min(times) if times else relative_time[-1]
        delta = abs(t3 - t0)
        part = delta / 3.0
        cut1, cut2 = t0 + part, t0 + 2*part

        # 2) window‑filter each POI’s candidates
        q4 = [i for i in best_positions["POI4"]["indices"]
              if relative_time[i] <= cut1]
        q5 = [i for i in best_positions["POI5"]["indices"]
              if cut1 <= relative_time[i] <= cut2]
        q6 = [i for i in best_positions["POI6"]["indices"]
              if relative_time[i] >= cut2]
        if len(ground_truth) > 3:
            q4.append(ground_truth[3])
        if len(ground_truth) > 4:
            q5.append(ground_truth[4])
        if len(ground_truth) > 5:
            q6.append(ground_truth[5])

        def choose_and_insert(lst):
            if not lst:
                return lst
            best = self._correct_bias(dissipation, relative_time, lst)
            # coerce to scalar int if needed
            if isinstance(best, (list, np.ndarray)):
                best = int(np.array(best).flat[0])
            else:
                best = int(best)
            return [best] + [i for i in lst if i != best]

        q4 = choose_and_insert(q4)
        q5 = choose_and_insert(q5)
        # q6 = choose_and_insert(q6)

        best_positions["POI4"]["indices"] = q4
        best_positions["POI5"]["indices"] = q5
        best_positions["POI6"]["indices"] = q6

        # 4) SPECIAL -1 HANDLING:
        #    If a POI’s list is *only* [-1,…], fall back to the ground‑truth label.
        #    Additionally, if POI6’s *first* candidate is -1, use 2%‐from‐end.
        for poi_num in (4, 5, 6):
            key = f"POI{poi_num}"
            lst = best_positions[key]["indices"]
            if not lst:
                continue

            # 4a) all -1’s → rollback to model_data_label
            if all(i == -1 for i in lst):
                fallback = ground_truth[poi_num - 1]
                best_positions[key]["indices"] = [fallback]
                continue

            # 4b) POI6 partial -1 in position 0 → 2% from end
            if key == "POI6" and lst[0] == -1:
                fallback = int(len(relative_time) * 0.98)
                best_positions[key]["indices"][0] = fallback

        # 5) FINAL ORDERING CHECK:
        #    Ensure each POI’s top candidate strictly follows the prior one,
        #    else fall back to its model_data_label.
        for poi_num in range(2, 7):  # POI2 … POI6
            key = f"POI{poi_num}"
            prev_key = f"POI{poi_num - 1}"

            # get the list (may be missing or empty)
            inds = best_positions.get(key, {}).get("indices", [])

            # 5a) if empty, seed with model_data_label
            if not inds:
                best_positions[key]["indices"] = [
                    ground_truth[poi_num - 1]]
                inds = best_positions[key]["indices"]

            # determine the “previous” index
            if best_positions.get(prev_key, {}).get("indices"):
                prev_idx = best_positions[prev_key]["indices"][0]
            else:
                prev_idx = ground_truth[poi_num - 2]

            # 5b) enforce strict ordering: current > previous
            if inds[0] <= prev_idx:
                best_positions[key]["indices"][0] = ground_truth[poi_num - 1]
        if len(ground_truth) > 1:
            best_positions["POI1"]["indices"].insert(0, ground_truth[0])
        if len(ground_truth) > 4:
            best_positions["POI4"]["indices"].insert(0, ground_truth[3])
        for key, data in best_positions.items():
            inds = data.get("indices", [])
            confs = data.get("confidences")
            if confs is not None:
                # truncate or leave as-is if shorter
                data["confidences"] = confs[: len(inds)]
        return best_positions

    def predict(self, file_buffer: str, forecast_start: int = -1, forecast_end: int = -1, actual_poi_indices: np.ndarray = None, plotting: bool = True) -> dict:
        """
        Predict POI indices from dissipation CSV data using the loaded model and scaler.

        Args:
            file_buffer (str or file-like): Path or buffer containing dissipation CSV data.
            forecast_start (int): Starting index for forecasting (default: -1).
            forecast_end (int): Ending index for forecasting (default: -1).
            actual_poi_indices (np.ndarray, optional): Ground-truth POI indices for comparison.

        Returns:
            dict: Final POI prediction dictionary with ordered 'indices' and confidences.
        """
        try:
            df = self._validate_file_buffer(file_buffer=file_buffer)
        except Exception as e:
            Log.e(
                f"File buffer `{file_buffer}` could not be validated because of error: `{e}`.")
            return
        qmodel_v2_labels = self._get_qmodel_v2_predictions(
            file_buffer=file_buffer)
        self._qmodel_v2_labels = qmodel_v2_labels
        model_data_labels = self._get_model_data_predictions(
            file_buffer=file_buffer)
        self._model_data_labels = model_data_labels

        ###
        # Sanity check to avoid throwing an `IndexError` later on
        if len(model_data_labels) == 0:
            # End error message with a colon as the next logged message is an error reason
            Log.e("Model generated no model data labels. No predictions can be made:")
            return
        ###
        file_buffer = self._reset_file_buffer(file_buffer=file_buffer)
        feature_vector = QDataProcessor.process_data(
            file_buffer=file_buffer, live=True)
        transformed_feature_vector = self._scaler.transform(
            feature_vector.values)
        ddata = xgb.DMatrix(transformed_feature_vector)
        predicted_probabilites = self._booster.predict(ddata)
        extracted_predictions = self._extract_predictions(
            predicted_probabilites, model_data_labels)
        final_predictions = self._select_best_predictions(
            predictions=extracted_predictions,
            ground_truth=qmodel_v2_labels,
            relative_time=df["Relative_time"].values,
            dissipation=feature_vector["Dissipation_smooth"].values)
        if plotting:
            plt.figure(figsize=(10, 6))

            plt.plot(df["Relative_time"],
                     df["Dissipation"],
                     linewidth=1.5,
                     label="Dissipation")

            # Overlay each POI’s candidate points
            cmap = plt.get_cmap("tab10")
            for i, (poi_name, poi_info) in enumerate(final_predictions.items()):
                idxs = poi_info["indices"]
                times = df["Relative_time"].iloc[idxs]
                values = df["Dissipation"].iloc[idxs]

                plt.scatter(times,
                            values,
                            marker="x",
                            s=100,
                            color=cmap(i),
                            label=poi_name)

            plt.xlabel("Relative time")
            plt.ylabel("Dissipation")
            plt.title("Dissipation curve with candidate POIs")
            plt.legend()
            plt.tight_layout()
            plt.show()
        return final_predictions


if __name__ == "__main__":
    booster_path = os.path.join(
        "QModel", "SavedModels", "qmodel_v3", "qmodel_v3_booster.json")
    scaler_path = os.path.join(
        "QModel", "SavedModels", "qmodel_v3", "qmodel_v3_scaler.pkl")
    test_dir = os.path.join('content', 'static', 'test')
    test_content = QDataProcessor.load_content(data_dir=test_dir)
    qmp = QModelPredictor(booster_path=booster_path, scaler_path=scaler_path)
    import random
    random.shuffle(test_content)
    for i, (data_file, poi_file) in enumerate(test_content):
        Log.i(f"Predicting on data file `{data_file}`.")
        poi_indices = pd.read_csv(poi_file, header=None)
        qmp.predict(file_buffer=data_file,
                    actual_poi_indices=poi_indices.values)
