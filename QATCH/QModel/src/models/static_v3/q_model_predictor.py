#!/usr/bin/env python3
"""
q_model_predictor.py

Provides the QModelPredictor class for predicting Points of Interest (POIs) in dissipation
data using a pre-trained XGBoost booster and an sklearn scaler pipeline. Includes methods for
file validation, feature extraction, probability formatting, and bias correction for refined
POI selection.

Author: Paul MacNichol (paul.macnichol@qatchtech.com)
Date: 04-30-2025
Version: QModel.Ver3.12
"""

import xgboost as xgb
from sklearn.pipeline import Pipeline
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Union, Optional
from scipy.signal import find_peaks, savgol_filter
from QATCH.common.logger import Logger as Log
from QATCH.QModel.src.models.static_v3.q_model_data_processor import QDataProcessor
from QATCH.models.ModelData import ModelData
from QATCH.QModel.src.models.static_v2.q_image_clusterer import QClusterer
from QATCH.QModel.src.models.static_v2.q_multi_model import QPredictor
from QATCH.common.architecture import Architecture
POI_1_OFFSET = 2
POI_2_OFFSET = -2

TAG = ["QModel V3"]


class QModelPredictor:
    """Predict points-of-interest (POIs) in dissipation time-series data.

    This class orchestrates the full pipeline for identifying six key POI indices
    using:
      - QModel v2 clustering to choose a prediction model,
      - An XGBoost booster for initial probability predictions,
      - Post-processing filters including baseline filtering (POI1),
        window slicing (POI4-6), bias correction, and strict ordering,
      - Confidence sorting and fallback mechanisms to ensure robust outputs.
    """

    def __init__(self, booster_path: str, scaler_path: str) -> None:
        """Initialize the model loader by loading the booster and scaler.

        Attempts to load an XGBoost booster model and a preprocessing scaler pipeline
        from the given file paths. If a path is invalid or loading fails, an error
        will be logged.

        Args:
            booster_path (str): Path to the XGBoost booster model file.
            scaler_path (str): Path to the serialized scaler pipeline file.

        Returns:
            None
        """
        if booster_path is None or booster_path == "" or not os.path.exists(booster_path):
            Log.e(TAG,
                  f'Booster path `{booster_path}` is empty string or does not exist.')
        if scaler_path is None or scaler_path == "" or not os.path.exists(scaler_path):
            Log.e(TAG,
                  f'Scaler path `{scaler_path}` is empty string or does not exist.')

        self._booster: xgb.Booster = xgb.Booster()
        self._scaler: Pipeline = None

        try:
            self._booster.load_model(fname=booster_path)
            Log.i(TAG, f'Booster loaded from path `{booster_path}`.')
        except Exception as e:
            Log.e(TAG,
                  f'Error loading booster from path `{booster_path}` with exception: `{e}`')

        try:
            self._scaler = self._load_scaler(scaler_path=scaler_path)
            Log.i(TAG, f'Scaler loaded from path `{scaler_path}`.')
        except Exception as e:
            Log.e(TAG,
                  f'Error loading scaler from path `{scaler_path}` with exception: `{e}`')

    def _load_scaler(self, scaler_path: str) -> Pipeline:
        """Load a serialized preprocessing scaler pipeline from a pickle file.

        Opens the file at `scaler_path` in binary mode and deserializes it into
        a scikit-learn `Pipeline` object. If loading fails or the result is `None`,
        an `IOError` is raised.

        Args:
            scaler_path (str): Filesystem path to the pickled scaler pipeline.

        Returns:
            Pipeline: The deserialized scaler pipeline ready for use.

        Raises:
            IOError: If the scaler could not be deserialized or is `None`.
        """
        scaler: Pipeline = None
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        if scaler is None:
            raise IOError(
                f"Scaler could not be loaded from path `{scaler_path}`.")

        return scaler

    def _filter_labels(self, predicted_labels: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
        """Filter outlier class labels based on median absolute deviation in index space.

        For each class label in 1–6, computes the median index of occurrences
        and removes any occurrences whose distance from that median exceeds
        `multiplier` × MAD (median absolute deviation). Outliers are replaced with 0.

        Args:
            predicted_labels (np.ndarray): 1D array of integer class labels.
                Valid labels are non-negative integers; 0 is reserved for “filtered out.”
            multiplier (float, optional): Scaling factor applied to the MAD to
                determine the outlier threshold. Must be positive. Defaults to 1.5.

        Returns:
            np.ndarray: A copy of `predicted_labels` where outlier labels have been
            set to 0.

        Raises:
            TypeError: If `predicted_labels` is not a NumPy array or if `multiplier`
                is not a float or int.
            ValueError: If `predicted_labels` is not one-dimensional, if it contains
                non-integer values, or if `multiplier` is not positive.
        """
        # Input validation
        if not isinstance(predicted_labels, np.ndarray):
            raise TypeError("`predicted_labels` must be a numpy.ndarray.")
        if predicted_labels.ndim != 1:
            raise ValueError("`predicted_labels` must be one-dimensional.")
        if not issubclass(predicted_labels.dtype.type, np.integer):
            raise ValueError(
                "`predicted_labels` array must contain integer values.")
        if not isinstance(multiplier, (float, int)):
            raise TypeError("`multiplier` must be a float or int.")
        multiplier = float(multiplier)
        if multiplier <= 0:
            raise ValueError("`multiplier` must be greater than zero.")

        filtered = predicted_labels.copy()

        # Process each class label 1 through 6
        for cls in range(1, 7):
            indices = np.where(predicted_labels == cls)[0]
            if indices.size == 0:
                continue

            # Compute median index and MAD
            median_idx = np.median(indices)
            deviations = np.abs(indices - median_idx)
            mad = np.median(deviations)
            threshold = multiplier * (mad if mad > 0 else 1)

            # Identify and filter outliers
            outlier_mask = deviations > threshold
            outlier_indices = indices[outlier_mask]
            filtered[outlier_indices] = 0

        return filtered

    def _get_model_data_predictions(self, file_buffer: str):
        """Load model-based point predictions from a file path or CSV buffer.

        This method uses the `ModelData` class to identify key point indices
        (POIs) in dissipation data. If `file_buffer` is a string, it is treated
        as a filesystem path and passed directly to `ModelData.IdentifyPoints`.
        Otherwise, the buffer is reset and read as CSV text: the header line
        determines which columns to load for time, frequency, and dissipation,
        and the data is parsed with `numpy.loadtxt`. Raw predictions from
        `IdentifyPoints` (either integers or lists of `(index, score)` pairs)
        are normalized into a flat list of integer indices by selecting each
        integer directly or, for lists, choosing the index with the highest score.

        Args:
            file_buffer (str or file-like):
                - If `str`, the path to a CSV file containing time, frequency,
                and dissipation data.
                - Otherwise, an open file-like object yielding CSV lines;
                the first line (header) is inspected to choose column indices:
                    - `(2, 4, 6, 7)` if "Ambient" appears in the header
                    - `(2, 3, 5, 6)` otherwise

        Returns:
            List[int]: A list of integer point indices as predicted by the model.
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

    def _get_qmodel_v2_predictions(self, file_buffer: str) -> List[int]:
        """Generate point-of-interest predictions using QModel v2.

        This method first uses a clustering model to assign the input data
        (`file_buffer`) to one of several clusters. Based on the cluster label,
        it selects a corresponding QPredictor model to generate candidate POI
        indices. The first index of each returned candidate is collected and
        returned as a list. If clustering or prediction fails at any stage,
        a warning is logged and a fallback prediction is returned.

        Args:
            file_buffer (str or file-like): Either a filesystem path to the data
                file or an open file-like buffer containing CSV lines of dissipation
                data.

        Returns:
            List[int]: A list of predicted POI indices. If prediction fails,
            falls back to `_get_model_data_predictions`.

        """
        qmodel_v2_points: List[int] = []

        # Paths for clustering and prediction models
        base_path = os.path.join(Architecture.get_path(
        ), "QATCH", "QModel", "SavedModels", "qmodel_v2")
        cluster_model_path = os.path.join(base_path, "cluster.joblib")
        predict_model_template = os.path.join(base_path, "QMultiType_{}.json")

        # Initialize clusterer
        clusterer = QClusterer(model_path=cluster_model_path)
        try:
            label = clusterer.predict_label(file_buffer)
        except Exception as e:
            label = 0
            Log.w(
                f"Failed clustering in QModel v2, defaulting to label 0: {e}")

        # Reset buffer for predictor
        try:
            file_buffer.seek(0)
        except Exception:
            pass

        # Select predictor based on cluster label
        try:
            model_path = predict_model_template.format(
                label if label in (0, 1) else 2)
            qpredictor = QPredictor(model_path=model_path)
        except Exception as e:
            Log.w(
                f"Failed to load QModel v2 predictor for label {label}, defaulting to 0: {e}")
            qpredictor = QPredictor(
                model_path=predict_model_template.format(0))

        # Generate predictions
        try:
            candidates = qpredictor.predict(
                file_buffer=file_buffer, run_type=label)
            for c in candidates:
                qmodel_v2_points.append(c[0][0])
            return qmodel_v2_points
        except Exception as e:
            Log.w(
                f"Failed QModel v2 prediction, falling back to ModelData: {e}")
            return self._get_model_data_predictions(file_buffer)

    def _reset_file_buffer(self, file_buffer: str):
        """Ensure the file buffer is positioned at its beginning for reading.

        If `file_buffer` is a file path (string), it is returned unchanged.
        If it is a seekable file-like object, its position is reset to zero.
        Otherwise, an exception is raised.

        Args:
            file_buffer (str or file-like): Either a filesystem path (string)
                or a file-like object supporting `seek`.

        Returns:
            str or file-like: The original `file_buffer`, with its read pointer
            reset if applicable.

        Raises:
            Exception: If `file_buffer` is a non-seekable stream and thus cannot
                be rewound.
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
        """Load and validate a CSV data file into a pandas DataFrame.

        This method ensures the file buffer is reset or accepts a string path,
        then attempts to read it as CSV. It handles common read errors, checks
        for emptiness, and verifies the presence of required columns.

        Args:
            file_buffer (str or file-like): Either a filesystem path pointing to
                a CSV file, or a file-like object containing CSV data.

        Returns:
            pd.DataFrame: A DataFrame containing the CSV data with at least the
            columns "Dissipation", "Resonance_Frequency", and "Relative_time".

        Raises:
            ValueError: If the buffer cannot be reset, the file is empty,
                        parsing fails, the resulting DataFrame is empty, or
                        required columns are missing.
        """
        # Reset buffer if necessary
        try:
            file_buffer = self._reset_file_buffer(file_buffer=file_buffer)
        except Exception:
            raise ValueError(
                "File buffer must be a non-empty string containing CSV data.")

        # Read CSV into DataFrame
        try:
            df = pd.read_csv(file_buffer)
        except pd.errors.EmptyDataError:
            raise ValueError("The provided data file is empty.")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing data file: `{e}`")
        except Exception as e:
            raise ValueError(
                f"An unexpected error occurred while reading the data file: `{e}`")

        # Validate DataFrame contents
        if df.empty:
            raise ValueError("The data file does not contain any data.")

        required_columns = {"Dissipation",
                            "Resonance_Frequency", "Relative_time"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Data file missing required columns: `{', '.join(missing)}`.")

        return df

    def _extract_predictions(
        self,
        predicted_probabilities: np.ndarray,
        model_data_labels: np.ndarray
    ) -> Dict[str, Dict[str, List[float]]]:
        """Extract POI indices and confidence scores from predicted probabilities.

        This method converts a 2D array of class probability predictions into a
        structured dictionary of point-of-interest (POI) results. For each POI class
        (1=>6), it identifies all sample indices where that POI was predicted,
        sorts them by descending confidence, and returns both the indices and
        their corresponding confidence scores. If no samples are predicted for a
        given POI, the ground truth index from `model_data_labels` is used with
        a confidence of 0.

        Args:
            predicted_probabilities (np.ndarray):
                A 2D array of shape `(n_samples, n_classes)` containing the
                predicted probability of each class at each sample index.
            model_data_labels (np.ndarray):
                A 1D array of length ≥6 containing ground-truth POI indices
                for classes 1 through 6. Used as a fallback when no predictions
                exist for a class.

        Returns:
            Dict[str, Dict[str, List[int] or List[float]]]:
                A dictionary mapping each POI key (`"POI1"` ... `"POI6"`) to:
                    - `"indices"`: list of sample indices sorted by confidence (desc),
                    - `"confidences"`: list of corresponding probability scores.

        Raises:
            TypeError:
                If `predicted_probabilities` or `model_data_labels` is not a
                numpy ndarray.
            ValueError:
                If `predicted_probabilities` is not 2D, if its second dimension
                is less than 7 (to cover classes 0–6), or if `model_data_labels`
                has fewer than 6 elements.
        """
        # Input validation
        if not isinstance(predicted_probabilities, np.ndarray):
            raise TypeError(
                "`predicted_probabilities` must be a numpy.ndarray.")
        if predicted_probabilities.ndim != 2:
            raise ValueError("`predicted_probabilities` must be a 2D array.")
        n_classes = predicted_probabilities.shape[1]
        if n_classes < 7:
            raise ValueError(
                "`predicted_probabilities` must have at least 7 columns for classes 0–6."
            )

        if not isinstance(model_data_labels, list):
            raise TypeError("`model_data_labels` must be a list")
        poi_results: Dict[str, Dict[str, List[float]]] = {}
        # Determine the most likely class at each sample index
        predicted_labels = np.argmax(predicted_probabilities, axis=1)

        # Process POIs 1 through 6
        for poi in range(1, 7):
            key = f"POI{poi}"
            # Find all indices predicted as this POI
            indices = np.where(predicted_labels == poi)[0]
            if indices.size == 0:
                # Fallback to the ground truth label index with zero confidence
                poi_results[key] = {
                    "indices": [int(model_data_labels[poi - 1])],
                    "confidences": [0.0]
                }
            else:
                # Gather confidence scores and sort descending
                confidences = predicted_probabilities[indices, poi]
                order = np.argsort(-confidences)
                sorted_indices = indices[order].tolist()
                sorted_confidences = confidences[order].tolist()
                poi_results[key] = {
                    "indices": sorted_indices,
                    "confidences": sorted_confidences
                }

        return poi_results

    def _correct_bias(
        self,
        dissipation: np.ndarray,
        relative_time: np.ndarray,
        candidates: list,
        feature_vector: pd.DataFrame,
        poi: str,
        ground_truth: list
    ) -> int:
        """Adjust a candidate point by bias-correcting toward steep slope onsets.

        Iteratively lowers a slope percentile threshold from 100 down toward 90 (in
        0.5% steps) to find peaks in the dissipation slope within the candidate
        range. For each peak, finds the left-hand base (where slope ≤ 0) and then
        selects the candidate closest to one of those bases. If no valid pick is
        found by the 90th percentile, falls back to 90% directly. Special handling
        for POI6 and a slip-check for POI4 are applied.

        Args:
            dissipation (np.ndarray): 1D array of dissipation values.
            relative_time (np.ndarray): 1D array of time values, same length as
                `dissipation`.
            candidates (list[int]): List of integer indices representing candidate
                points of interest.
            feature_vector (pd.DataFrame): DataFrame containing at least the columns
                `"Detrend_Difference"` and `"Resonance_Frequency_smooth"`.
            poi (str): One of `"POI1"` … `"POI6"`, determining special-case logic.
            ground_truth (list[int]): List of ground-truth POI indices for classes
                1=>6; used as a fallback for POI4 slip-check.

        Returns:
            int: The selected, bias-corrected index.

        Raises:
            TypeError: If any input is of incorrect type.
            ValueError: If array lengths mismatch, `poi` is invalid, or no
                candidates are provided.
        """
        # --- Input validation ---
        if not isinstance(dissipation, np.ndarray):
            raise TypeError("`dissipation` must be a numpy.ndarray.")
        if dissipation.ndim != 1:
            raise ValueError("`dissipation` must be 1-dimensional.")
        if not isinstance(relative_time, np.ndarray):
            raise TypeError("`relative_time` must be a numpy.ndarray.")
        if relative_time.ndim != 1:
            raise ValueError("`relative_time` must be 1-dimensional.")
        if dissipation.shape[0] != relative_time.shape[0]:
            raise ValueError(
                "`dissipation` and `relative_time` must have the same length.")
        if not isinstance(feature_vector, pd.DataFrame):
            raise TypeError("`feature_vector` must be a pandas DataFrame.")
        if poi not in {f"POI{i}" for i in range(1, 7)}:
            raise ValueError("`poi` must be one of 'POI1' … 'POI6'.")
        if not isinstance(ground_truth, list):
            raise TypeError("`ground_truth` must be a list of integers.")

        # --- Core logic ---
        cmin, cmax = min(candidates), max(candidates)
        chosen_perc = None
        valid_peaks = np.array([], dtype=int)
        valid_bases = []
        valid_filtered = []
        min_iter = 50.0 if poi == "POI6" else 89.6

        # For POI6, use detrended-difference column
        if poi == "POI6":
            if "Detrend_Difference" not in feature_vector:
                raise ValueError(
                    "`feature_vector` missing 'Detrend_Difference' column for POI6.")
            diss = feature_vector["Detrend_Difference"].values
        else:
            diss = dissipation

        # Compute slope
        slope = np.gradient(diss, relative_time)

        # Try lowering threshold percentiles from 100 → min_iter in 0.5 steps
        for perc in np.arange(100.0, min_iter, -0.5):
            thresh = np.percentile(slope, perc)
            peaks, _ = find_peaks(slope, height=thresh)
            peaks = peaks[(peaks >= cmin) & (peaks <= cmax)]
            if peaks.size == 0:
                continue

            # Find left-hand bases (first non-positive slope before each peak)
            bases = []
            for p in peaks:
                j = p
                while j > 0 and slope[j] > 0:
                    j -= 1
                bases.append(j)

            # Distance function: candidate to nearest base of a peak to its right
            def dist_if_left(c):
                valid = [(p_i, b_i)
                         for p_i, b_i in zip(peaks, bases) if p_i >= c]
                if not valid:
                    return np.inf
                pk, bs = min(valid, key=lambda x: abs(x[0] - c))
                return abs(c - bs)

            filtered = [c for c in candidates]
            if filtered:
                chosen_perc = perc
                valid_peaks, valid_bases, valid_filtered = peaks, bases, filtered
                break

        # Fallback at min_iter (e.g. 90th percentile)
        if chosen_perc is None:
            thresh = np.percentile(slope, min_iter)
            peaks, _ = find_peaks(slope, height=thresh)
            valid_peaks = peaks[(peaks >= cmin) & (peaks <= cmax)]
            valid_bases = []
            for p in valid_peaks:
                j = p
                while j > 0 and slope[j] > 0:
                    j -= 1
                valid_bases.append(j)
            valid_filtered = candidates

            def dist_if_left(c):
                valid = [(p_i, b_i) for p_i, b_i in zip(
                    valid_peaks, valid_bases) if p_i >= c]
                if not valid:
                    return np.inf
                pk, bs = min(valid, key=lambda x: abs(x[0] - c))
                return abs(c - bs)

        # Select best candidate by proximity to a base
        best_idx = min(valid_filtered, key=dist_if_left)

        # Special handling for POI6: slide to max dissipation then base of nearest RF peak
        if poi == "POI6" and valid_filtered:
            start_idx, end_idx = cmin, cmax
            knee_point = self._poi_6_knee_point(
                start_of_segment=start_idx,
                end_of_segment=end_idx,
                feature_vector=feature_vector,
                relative_time=relative_time
            )
            segment = diss[start_idx: end_idx + 1]
            peak_rel = int(np.argmax(segment))
            candidate_idx = start_idx + peak_rel

            best_idx = max(knee_point, candidate_idx)

            rf = feature_vector.get("Resonance_Frequency_smooth")
            if rf is None:
                raise ValueError(
                    "`feature_vector` missing 'Resonance_Frequency_smooth' for POI6.")
            rf_vals = rf.values
            peaks_rf, _ = find_peaks(rf_vals)
            proximity = 5
            close = peaks_rf[np.abs(peaks_rf - best_idx) <= proximity]
            if close.size > 0:
                peak_idx = close[np.argmin(np.abs(close - best_idx))]
                troughs, _ = find_peaks(-rf_vals)
                left_troughs = troughs[troughs < peak_idx]
                if left_troughs.size > 0:
                    best_idx = int(left_troughs[-1])
                else:
                    local_seg = rf_vals[start_idx: peak_idx + 1]
                    base_rel = int(np.argmin(local_seg))
                    best_idx = start_idx + base_rel

        # Slip-check for POI4: ensure bias correction didn't drift toward POI3
        if poi == "POI4" and len(ground_truth) >= 4:
            poi3_gt, poi4_gt = ground_truth[2], ground_truth[3]
            if abs(best_idx - poi3_gt) < abs(best_idx - poi4_gt):
                Log.w(
                    "Bias correction for POI4 slipped toward POI3; skipping adjustment.")
                best_idx = poi4_gt

        return best_idx

    def _poi_2_offset(self,
                      positions: Dict[str, Dict[str, List[int]]],
                      feature_vector: pd.DataFrame,
                      relative_time: np.ndarray) -> int:
        """Compute the adjusted index for POI2 based on the “knee” of the Difference_smooth curve.

        This method locates the point along the Difference_smooth signal—between the first
        candidate for POI1 and the last candidate for POI3—where the rate of increase
        (slope) falls below a fraction of its maximum (i.e., where the curve starts to flatten).
        It then returns that point’s index plus a fixed offset (POI_2_OFFSET).

        The algorithm is:
        1. Validate that `positions` contains dict entries for "POI1" and "POI3", each with
        an integer `indices` list.
        2. If POI3 indices are missing or empty, fall back to the model’s stored POI3 label.
        3. If POI1 indices are missing or empty, simply return two steps before the earliest
        POI3 candidate.
        4. Otherwise, slice the Difference_smooth curve between the first POI1 index and the
        last POI3 index.
        5. Compute the instantaneous slope via `np.gradient`.
        6. Find the first point after the peak slope where the slope drops below 20% of its max.
        If none is found, use the end of the segment.
        7. Return that flattened “knee” index plus POI_2_OFFSET.

        Args:
            positions: Mapping of POI keys ("POI1", "POI3", etc.) to dicts with an "indices"
                key containing candidate integer indices.
            feature_vector: DataFrame containing a 'Difference_smooth' column of smoothed
                difference values.
            relative_time: 1D array of time points corresponding to feature_vector rows.

        Returns:
            The integer index of the detected flattening point plus the constant POI_2_OFFSET.

        Raises:
            TypeError: If `positions` is not a dict, or if any indices list contains non-int
                entries.
            ValueError: If entries for 'POI1' or 'POI3' are missing or not dicts.
        """
        # Input validation
        if not isinstance(positions, dict):
            raise TypeError(
                "`positions` must be a dict mapping POI keys to dicts.")

        poi3_entry = positions.get("POI3")
        if not isinstance(poi3_entry, dict):
            raise ValueError(
                "Missing or invalid entry for 'POI3' in positions.")
        poi1_entry = positions.get("POI1")
        if not isinstance(poi1_entry, dict):
            raise ValueError(
                "Missing or invalid entry for 'POI1' in positions.")
        poi2_entry = positions.get("POI2")
        if not isinstance(poi2_entry, dict):
            raise ValueError(
                "Missing or invalid entry for 'POI2' in positions.")

        poi3_indices = poi3_entry.get("indices")
        if not isinstance(poi3_indices, (list, tuple)) or not poi3_indices:
            poi3_indices = [self._model_data_labels[2]]
        if not all(isinstance(idx, int) for idx in poi3_indices):
            raise TypeError("All entries in 'POI3' indices must be integers.")
        poi2_indices = poi2_entry.get("indices")
        if not isinstance(poi3_indices, (list, tuple)) or not poi3_indices:
            poi2_indices = [self._model_data_labels[1]]
        if not all(isinstance(idx, int) for idx in poi3_indices):
            raise TypeError("All entries in 'POI3' indices must be integers.")

        # --- 2) Compute candidates ---
        base2 = max(poi2_indices) + POI_2_OFFSET
        base3 = min(poi3_indices) + POI_2_OFFSET
        first3 = poi3_indices[0]

        # --- 3) Log everything so you can trace the decision ---
        Log.d(f"POI2 max(poi2)={max(poi2_indices)}, OFFSET={base2}")
        Log.d(f"POI3 min(poi3)={min(poi3_indices)}, OFFSET={base3}")
        Log.d(f"First POI3 index: {first3}")
        # --- 4) Make the final pick ---
        if base2 < first3:
            result = max(base3, base2)
            Log.d(f"Choosing max(base3, base2) {result}")
        else:
            result = base3
            Log.d(f"base2 >= first3; falling back to base3 {result}")

        return result

    def _poi_6_knee_point(self,
                          start_of_segment: int,
                          end_of_segment: int,
                          feature_vector: pd.DataFrame,
                          relative_time: np.ndarray) -> int:

        n = abs(start_of_segment - end_of_segment)
        if n < 3:
            diff = feature_vector['Detrend_Difference'].values
        else:
            w = max(3, int(n * 0.05))
            if w % 2 == 0:
                w += 1
            w = min(w, n if n % 2 == 1 else n - 1)
            polyorder = 1
            po = min(polyorder, w - 1)
            diff = savgol_filter(
                feature_vector['Detrend_Difference'].values, window_length=w, polyorder=po)
        seg_d = diff[start_of_segment:end_of_segment+1]
        seg_t = relative_time[start_of_segment:end_of_segment+1]
        slope = np.gradient(seg_d, seg_t)
        valley_rel = int(np.argmin(slope))
        peaks, _ = find_peaks(seg_d)
        pre_peaks = peaks[peaks < valley_rel]

        if pre_peaks.size > 0:
            knee_rel = int(pre_peaks[-1])
        else:
            knee_rel = int(np.argmax(seg_d[:valley_rel+1]))

        knee_idx = start_of_segment + knee_rel
        return knee_idx

    def _choose_and_insert(
        self,
        indices: List[int],
        dissipation: np.ndarray,
        relative_time: np.ndarray,
        feature_vector: pd.DataFrame,
        poi: str,
        ground_truth: List[int]
    ) -> List[int]:
        """Select and prioritize the best candidate index for a POI, optionally bias-corrected.

        For POI4, POI5, and POI6, applies bias correction to choose the best index.
        For other POIs, selects the first candidate as the best. Returns a list
        with the chosen index first, followed by the remaining candidates.

        Args:
            indices (List[int]): Candidate indices for the point of interest.
            dissipation (np.ndarray): 1D array of dissipation values.
            relative_time (np.ndarray): 1D array of time values (same length as `dissipation`).
            feature_vector (pd.DataFrame): DataFrame containing columns needed for
                bias correction (e.g., "Detrend_Difference", "Resonance_Frequency_smooth").
            poi (str): Name of the POI, one of "POI1" … "POI6".
            ground_truth (List[int]): Ground-truth POI indices used for bias logic.

        Returns:
            List[int]: Ordered list of indices, with the selected best index first.

        Raises:
            TypeError: If inputs are of incorrect types.
            ValueError: If `indices` is empty, arrays have mismatched lengths,
                        or `poi` is not in the expected set.
        """
        # --- Convert and validate indices ---
        if not isinstance(indices, (list, tuple)):
            raise TypeError("`indices` must be a list or tuple of integers.")
        try:
            indices = [int(i) for i in indices]
        except Exception:
            raise TypeError(
                "All entries in `indices` must be convertible to int.")

        # --- Convert and validate ground_truth ---
        if not isinstance(ground_truth, (list, tuple)):
            raise TypeError(
                "`ground_truth` must be a list or tuple of integers.")
        try:
            ground_truth = [int(gt) for gt in ground_truth]
        except Exception:
            raise TypeError(
                "All entries in `ground_truth` must be convertible to int.")

        # --- Validate arrays ---
        if not isinstance(dissipation, np.ndarray) or dissipation.ndim != 1:
            raise TypeError("`dissipation` must be a 1D numpy.ndarray.")
        if not isinstance(relative_time, np.ndarray) or relative_time.ndim != 1:
            raise TypeError("`relative_time` must be a 1D numpy.ndarray.")
        if dissipation.shape[0] != relative_time.shape[0]:
            raise ValueError(
                "`dissipation` and `relative_time` must have the same length.")

        # --- Validate DataFrame and poi ---
        if not isinstance(feature_vector, pd.DataFrame):
            raise TypeError("`feature_vector` must be a pandas DataFrame.")
        valid_pois = {f"POI{i}" for i in range(1, 7)}
        if poi not in valid_pois:
            raise ValueError(f"`poi` must be one of {sorted(valid_pois)}.")

        # Determine best index
        if poi in ("POI4", "POI5", "POI6"):
            best = self._correct_bias(
                dissipation, relative_time, indices, feature_vector, poi, ground_truth
            )
        else:
            best = indices[0]

        # Normalize to single integer
        if isinstance(best, (list, np.ndarray)):
            # Flatten and take first element
            best_idx = int(np.array(best).flat[0])
        else:
            best_idx = int(best)

        # Reorder: best first, then the rest
        remaining = [i for i in indices if i != best_idx]
        return [best_idx] + remaining

    def _filter_poi1_baseline(
        self,
        indices: List[Union[int, np.integer]],
        dissipation: np.ndarray,
        relative_time: np.ndarray
    ) -> List[int]:
        """Filter POI1 candidates using an early-run baseline and IQR outlier removal.

        Computes a baseline dissipation over the first 1–3% of the run. Any candidate
        whose dissipation at its index does not exceed this baseline is discarded.
        Remaining candidates are then filtered by the interquartile range (IQR) of
        their dissipation values. Finally, the leftmost candidate is offset by
        `POI_1_OFFSET` and placed first in the returned list.

        Args:
            indices (List[int or np.integer]):
                Candidate sample indices for POI1.
            dissipation (np.ndarray):
                1D array of dissipation values.
            relative_time (np.ndarray):
                1D array of time values (same length as `dissipation`).

        Returns:
            List[int]: Filtered and reordered list of POI1 indices.

        Raises:
            TypeError: If input types are incorrect or not convertible to required types.
            ValueError: If array lengths mismatch, indices are out of bounds, or
                        `relative_time` is too short.
        """
        # Validate dissipation and relative_time arrays
        if not isinstance(dissipation, np.ndarray) or dissipation.ndim != 1:
            raise TypeError("`dissipation` must be a 1D numpy.ndarray.")
        if not isinstance(relative_time, np.ndarray) or relative_time.ndim != 1:
            raise TypeError("`relative_time` must be a 1D numpy.ndarray.")
        if dissipation.shape[0] != relative_time.shape[0]:
            raise ValueError(
                "`dissipation` and `relative_time` must have the same length.")
        if dissipation.size < 2:
            raise ValueError(
                "`dissipation` must contain at least two data points.")
        if any(i < 0 or i >= dissipation.size for i in indices):
            raise ValueError(
                "All `indices` must be within the valid range of the data arrays.")

        # Baseline computation (first 1–3% of run)
        t0, tN = float(relative_time[0]), float(relative_time[-1])
        duration = tN - t0
        if duration <= 0:
            raise ValueError("`relative_time` must be strictly increasing.")
        low_t, high_t = t0 + 0.01 * duration, t0 + 0.03 * duration
        mask = (relative_time >= low_t) & (relative_time <= high_t)
        if mask.any():
            baseline = float(np.mean(dissipation[mask]))
        else:
            n = max(1, int(0.03 * dissipation.size))
            baseline = float(np.mean(dissipation[:n]))

        # Filter by baseline
        filtered = [i for i in indices if dissipation[i] > baseline]

        # IQR outlier removal
        if filtered:
            vals = np.array([dissipation[i] for i in filtered], dtype=float)
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            filtered = [
                idx for idx, val in zip(filtered, vals)
                if lower <= val <= upper
            ]

        # Reorder with leftmost + offset first
        if filtered:
            leftmost = min(filtered) + POI_1_OFFSET
            leftmost = int(leftmost)
            others = [idx for idx in filtered if idx != leftmost]
            filtered = [leftmost] + others

        return filtered

    def _sort_by_confidence(
        self,
        positions: Dict[str, Dict[str, List[Union[int, float]]]],
        poi_key: str
    ) -> None:
        """Sort the indices and confidences for a given POI by descending confidence.

        Ensures that for the specified `poi_key`, the list of indices and their
        corresponding confidences are both present, of equal length, and sortable.
        Converts any numpy types to native Python `int` and `float` before sorting.

        Args:
            positions (Dict[str, Dict[str, List[int or float]]]):
                Mapping from POI keys (e.g. "POI1"… "POI6") to dictionaries
                containing:
                - 'indices': List of sample indices (int or np.integer).
                - 'confidences': List of confidence scores (float or np.floating).
            poi_key (str): The POI key whose values should be sorted.

        Returns:
            None: The `positions` dict is modified in place.

        Raises:
            TypeError: If `positions` is not a dict, `poi_key` not a str, the entry
                under `poi_key` is not a dict, or elements in 'indices'/'confidences'
                cannot be converted to int/float.
            ValueError: If 'indices' and 'confidences' are of mismatched lengths.
        """
        # Validate inputs
        if not isinstance(positions, dict):
            raise TypeError(
                "`positions` must be a dict mapping POI keys to dicts.")
        if not isinstance(poi_key, str):
            raise TypeError("`poi_key` must be a string.")
        if poi_key not in positions:
            return

        entry = positions[poi_key]
        if not isinstance(entry, dict):
            raise TypeError(f"Entry for '{poi_key}' must be a dict.")

        inds = entry.get('indices')
        confs = entry.get('confidences')
        if not isinstance(inds, (list, tuple)) or not isinstance(confs, (list, tuple)):
            return
        if len(inds) != len(confs):
            raise ValueError(
                f"'indices' and 'confidences' must have the same length for '{poi_key}'.")

        # Convert types
        try:
            inds = [int(i) for i in inds]
        except Exception:
            raise TypeError(
                "All entries in 'indices' must be convertible to int.")
        try:
            confs = [float(c) for c in confs]
        except Exception:
            raise TypeError(
                "All entries in 'confidences' must be convertible to float.")

        # Sort by confidence descending
        pairs = list(zip(confs, inds))
        pairs.sort(key=lambda x: x[0], reverse=True)

        # Unzip back to sorted lists
        if pairs:
            sorted_confs, sorted_inds = zip(*pairs)
            entry['confidences'] = list(sorted_confs)
            entry['indices'] = list(sorted_inds)
        else:
            entry['confidences'] = []
            entry['indices'] = []

    def _filter_poi6_near_ground_truth(
        self,
        positions: Dict[str, Any],
        ground_truth: List[Union[int, np.integer]],
        relative_time: np.ndarray,
        pct: float = 0.1
    ) -> None:
        """Filter POI6 candidates to those within a time window around the ground-truth.

        Removes any candidate index for POI6 whose timestamp deviates by more than
        `pct * total_duration` from the ground-truth POI6 time.

        Args:
            positions (Dict[str, Any]):
                Mapping of POI keys to dicts containing 'indices' and optional 'confidences'.
            ground_truth (List[int or np.integer]):
                List of ground-truth POI indices; index 5 (the sixth element) is used for POI6.
            relative_time (np.ndarray):
                1D array of time values corresponding to dissipation measurements.
            pct (float, optional):
                Fraction of the total run duration defining the allowed deviation
                window. Must be between 0 and 1. Defaults to 0.1.

        Returns:
            None: Modifies `positions['POI6']` in place.

        Raises:
            TypeError: If input types are incorrect or not convertible.
            ValueError: If required data is missing, or `pct` is out of range,
                        or `relative_time` has insufficient length.
        """
        # Validate inputs
        if not isinstance(positions, dict):
            raise TypeError(
                "`positions` must be a dict mapping POI keys to dicts.")
        key = "POI6"
        if key not in positions:
            return
        entry = positions[key]
        if not isinstance(entry, dict):
            raise TypeError(f"Entry for '{key}' must be a dict.")
        if not isinstance(ground_truth, (list, tuple)):
            raise TypeError(
                "`ground_truth` must be a list or tuple of integers.")
        if len(ground_truth) <= 5:
            return
        try:
            gt6 = int(ground_truth[5])
        except Exception:
            raise TypeError(
                "Ground-truth POI6 index must be convertible to int.")
        if not isinstance(relative_time, np.ndarray) or relative_time.ndim != 1:
            raise TypeError("`relative_time` must be a 1D numpy.ndarray.")
        if relative_time.size < 2:
            raise ValueError(
                "`relative_time` must contain at least two time points.")
        if not (0 < pct <= 1):
            raise ValueError(
                "`pct` must be a float between 0 (exclusive) and 1 (inclusive).")
        if not self._valid_index(gt6, relative_time):
            return

        # Retrieve and validate candidate lists
        inds = entry.get("indices")
        confs = entry.get("confidences")
        if not isinstance(inds, (list, tuple)) or not inds or confs is None:
            return
        try:
            inds = [int(i) for i in inds]
        except Exception:
            raise TypeError(
                "All entries in 'indices' must be convertible to int.")
        try:
            confs = [float(c) for c in confs]
        except Exception:
            raise TypeError(
                "All entries in 'confidences' must be convertible to float.")
        if len(inds) != len(confs):
            raise ValueError(
                "'indices' and 'confidences' must have the same length.")

        # Compute time window around ground-truth
        total_duration = float(relative_time[-1] - relative_time[0])
        threshold = pct * total_duration

        # Filter candidates by time proximity
        new_inds = []
        new_confs = []
        gt_time = float(relative_time[gt6])
        for idx, conf in zip(inds, confs):
            if idx < 0 or idx >= relative_time.size:
                continue
            if abs(relative_time[idx] - gt_time) <= threshold:
                new_inds.append(int(idx))
                new_confs.append(float(conf))

        # Update in-place
        positions[key]["indices"] = new_inds
        positions[key]["confidences"] = new_confs

    def _copy_predictions(self, predictions):
        """Create a shallow copy of the predictions mapping.

        Each POI key in the `predictions` dict is copied to a new dict where the
        'indices' list is shallow-copied and the 'confidences' list (if present)
        is preserved.

        Args:
            predictions (Dict[str, Dict[str, Any]]): Mapping from POI keys (e.g.
                "POI1"… "POI6") to dicts containing:
                    - 'indices': List of integer indices.
                    - 'confidences' (optional): List of float confidences.

        Returns:
            Dict[str, Dict[str, Any]]: A new dict with the same keys as
            `predictions`, where each value dict has its 'indices' list copied
            and 'confidences' passed through.
        """
        return {
            k: {"indices": v["indices"][:],
                "confidences": v.get("confidences")}
            for k, v in predictions.items()
        }

    def _determine_t0(self, ground_truth, predictions, relative_time):
        """Determine the start time (t0) for windowing based on POI3.

        Prefers the ground-truth index for POI3 if available and valid; otherwise
        falls back to the first predicted index for POI3; if neither is available,
        returns the first timestamp.

        Args:
            ground_truth (List[int]): Ground-truth POI indices for classes 1–6.
            predictions (Dict[str, Dict[str, List[int]]]): Predicted POI mappings.
            relative_time (np.ndarray): 1D array of time values.

        Returns:
            float: The timestamp corresponding to the chosen POI3 index, or
            `relative_time[0]` if no valid POI3 index is found.
        """
        if len(ground_truth) > 2:
            idx = ground_truth[2]
            if self._valid_index(idx, relative_time):
                return relative_time[idx]
        cand = predictions.get("POI3", {}).get("indices", [])
        if cand and self._valid_index(cand[0], relative_time):
            return relative_time[cand[0]]
        return relative_time[0]

    def _determine_t3(self, ground_truth, predictions, relative_time):
        """Determine the end time (t3) for windowing based on POI6.

        Uses the ground-truth POI6 timestamp if available and valid, then any
        predicted POI6 index; returns the earliest of these, or the final
        timestamp if none are valid.

        Args:
            ground_truth (List[int]): Ground-truth POI indices for classes 1–6.
            predictions (Dict[str, Dict[str, List[int]]]): Predicted POI mappings.
            relative_time (np.ndarray): 1D array of time values.

        Returns:
            float: The timestamp corresponding to the chosen POI6 index, or
            `relative_time[-1]` if no valid POI6 index is found.
        """
        times = []
        if len(ground_truth) > 5 and self._valid_index(ground_truth[5], relative_time):
            times.append(relative_time[ground_truth[5]])
        cand = predictions.get("POI6", {}).get("indices", [])
        if cand and self._valid_index(cand[0], relative_time):
            times.append(relative_time[cand[0]])
        return min(times) if times else relative_time[-1]

    def _compute_cuts(self, t0, t3):
        """Compute two cut times dividing the interval [t0, t3] into thirds.

        Args:
            t0 (float): The start time.
            t3 (float): The end time.

        Returns:
            Tuple[float, float]: A pair `(cut1, cut2)` where
            `cut1 = t0 + (t3 - t0) / 3` and `cut2 = t0 + 2*(t3 - t0) / 3`.
        """
        delta = abs(t3 - t0)
        part = delta / 3.0
        return t0 + part, t0 + 2 * part

    def _filter_windows(
        self,
        positions,
        ground_truth,
        relative_time,
        cut1,
        cut2,
    ):
        """Partition POI4–POI6 candidates into time-based windows.

        Uses `cut1` and `cut2` to assign:
        - POI4 candidates occurring on or before `cut1`,
        - POI5 candidates between `cut1` and `cut2`,
        - POI6 candidates on or after `cut2`.

        Args:
            positions (Dict[str, Dict[str, List[int]]]): Mapping of POI keys to
                dicts with 'indices' lists.
            ground_truth (List[int]): Ground-truth POI indices (unused here).
            relative_time (np.ndarray): 1D array of time values.
            cut1 (float): First cut timestamp.
            cut2 (float): Second cut timestamp.

        Returns:
            Dict[str, List[int]]: A dict with keys 'POI4', 'POI5', 'POI6', each
            mapping to the list of candidate indices falling in the respective
            time window.
        """
        windows = {
            'POI4': [
                i for i in positions['POI4']['indices']
                if relative_time[i] <= cut1
            ],
            'POI5': [
                i for i in positions['POI5']['indices']
                if cut1 <= relative_time[i] <= cut2
            ],
            'POI6': [
                i for i in positions['POI6']['indices']
                if relative_time[i] >= cut2
            ]
        }
        return windows

    def _update_positions(
        self,
        positions: Dict[str, Dict[str, List[int]]],
        windows: Dict[str, List[int]]
    ) -> Dict[str, Dict[str, List[int]]]:
        """Update POI position indices in place based on filtered windows.

        Args:
            positions (dict): Mapping from POI keys ("POI4","POI5","POI6") to dicts
                containing at least an 'indices' list.
            windows (dict): Mapping from the same POI keys to the new list of indices.

        Returns:
            dict: The updated `positions` dict (same object as passed in).

        Raises:
            TypeError: If inputs are not dicts or the values are not lists of ints.
            KeyError: If a POI key in `windows` is missing from `positions`.
        """
        if not isinstance(positions, dict) or not isinstance(windows, dict):
            raise TypeError("`positions` and `windows` must be dicts.")
        for poi, inds in windows.items():
            if poi not in positions:
                raise KeyError(f"POI key '{poi}' not found in positions.")
            if not isinstance(inds, (list, tuple)):
                raise TypeError(
                    f"Window indices for '{poi}' must be a list of ints.")
            positions[poi]['indices'] = [int(i) for i in inds]
        return positions

    def _handle_negatives(
        self,
        positions: Dict[str, Dict[str, List[int]]],
        ground_truth: List[int],
        relative_time: np.ndarray
    ) -> None:
        """Replace invalid (-1) POI indices with fallbacks.

        For POI4–POI6:
        - If all indices are -1, replace with the ground-truth index.
        - For POI6, if the first index is -1, replace it with 98% of the run length.

        Args:
            positions (dict): Mapping from POI keys to dicts with 'indices' lists.
            ground_truth (list[int]): List of true POI indices (length ≥ 6).
            relative_time (np.ndarray): 1D array of time values.

        Raises:
            TypeError: If inputs are of incorrect types.
            ValueError: If `ground_truth` has fewer than 6 elements.
        """
        if not isinstance(positions, dict):
            raise TypeError("`positions` must be a dict.")
        if not isinstance(ground_truth, (list, tuple)) or len(ground_truth) < 6:
            raise ValueError("`ground_truth` must be a list of ≥6 ints.")
        if not isinstance(relative_time, np.ndarray) or relative_time.ndim != 1:
            raise TypeError("`relative_time` must be a 1D numpy.ndarray.")

        for num in (4, 5, 6):
            key = f"POI{num}"
            entry = positions.get(key)
            if not entry or 'indices' not in entry:
                continue
            lst = entry['indices']
            if not lst:
                continue
            # All-invalid case
            if all(int(i) == -1 for i in lst):
                entry['indices'] = [int(ground_truth[num - 1])]
                continue
            # POI6 special single-index case
            if key == 'POI6' and int(lst[0]) == -1:
                entry['indices'][0] = int(len(relative_time) * 0.98)

    def _enforce_strict_ordering(
        self,
        positions: Dict[str, Dict[str, List[int]]],
        ground_truth: List[int]
    ) -> None:
        """Ensure POI indices strictly increase from POI1 through POI6.

        For each POI2–POI6, if its first index ≤ the previous POI's first index,
        reset it to the corresponding ground-truth index.

        Args:
            positions (dict): Mapping from POI keys to dicts with 'indices' lists.
            ground_truth (list[int]): List of true POI indices (length ≥ 6).

        Raises:
            TypeError: If inputs are of incorrect types.
            ValueError: If `ground_truth` has fewer than 6 elements.
        """
        if not isinstance(positions, dict):
            raise TypeError("`positions` must be a dict.")
        if not isinstance(ground_truth, (list, tuple)) or len(ground_truth) < 6:
            raise ValueError("`ground_truth` must be a list of ≥6 ints.")

        for num in range(2, 7):
            key = f"POI{num}"
            prev_key = f"POI{num - 1}"
            inds = positions.get(key, {}).get('indices', [])
            if not inds:
                positions[key]['indices'] = [int(ground_truth[num - 1])]
                continue
            # Determine previous index
            prev_inds = positions.get(prev_key, {}).get('indices')
            prev_idx = int(prev_inds[0]) if prev_inds else int(
                ground_truth[num - 2])
            # Enforce ordering
            if int(inds[0]) <= prev_idx:
                positions[key]['indices'][0] = int(ground_truth[num - 1])

    def _truncate_confidences(
        self,
        positions: Dict[str, Dict[str, List[Any]]]
    ) -> None:
        """Truncate each confidence list to match its indices list length.

        Ensures that `positions[key]['confidences']` has at most as many entries
        as `positions[key]['indices']`.

        Args:
            positions (dict): Mapping from POI keys to dicts with 'indices' and
                optional 'confidences' lists.

        Raises:
            TypeError: If `positions` is not a dict or entries are malformed.
        """
        if not isinstance(positions, dict):
            raise TypeError("`positions` must be a dict.")
        for key, data in positions.items():
            if not isinstance(data, dict):
                continue
            inds = data.get('indices')
            confs = data.get('confidences')
            if confs is not None:
                if not isinstance(inds, (list, tuple)) or not isinstance(confs, (list, tuple)):
                    raise TypeError(
                        f"Invalid 'indices' or 'confidences' for '{key}'.")
                data['confidences'] = confs[:len(inds)]

    @staticmethod
    def _valid_index(idx: Any, arr: np.ndarray) -> bool:
        """Check whether idx is a valid array index into arr.

        Args:
            idx: Candidate index (int or numpy integer).
            arr (np.ndarray): Array to index.

        Returns:
            bool: True if idx is an integer 0 ≤ idx < len(arr), else False.
        """
        return isinstance(idx, (int, np.integer)) and 0 <= idx < len(arr)

    def _select_best_predictions(
        self,
        predictions: Dict[str, Dict[str, Any]],
        model_data_labels: List[int],
        ground_truth: List[int],
        relative_time: np.ndarray,
        feature_vector: pd.DataFrame,
        raw_vector: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Refine and select the best point-of-interest (POI) predictions.

        Applies a sequence of filters, windowing, and bias corrections to the
        raw `predictions` mapping. Ensures each POI (1–6) ends up with an ordered
        list of candidate indices and matching confidences.

        Workflow:
        1. Baseline-filter POI1 using raw dissipation.
        2. Deep-copy predictions to avoid side-effects.
        3. Boost and window-filter POI6 around ground-truth.
        4. Determine time anchors t0 (POI3) and t3 (POI6).
        5. Split POI4-6 into thirds of the run.
        6. Apply bias correction to POI4, POI5, and POI6.
        7. Handle invalid (-1) indices, insert POI2, enforce ordering.
        8. Truncate confidences to match indices.
        9. Guarantee each POI has ≥1 index and ≥1 confidence.

        Args:
            predictions (Dict[str, Dict[str, Any]]):
                Raw POI predictions mapping each `"POI{i}"` to a dict with:
                - `'indices'`: List of integer-like candidate indices.
                - `'confidences'` (optional): List of float-like scores.
            model_data_labels (List[int]):
                Ground-truth POI indices used for fallback injection and POI6 boosting.
                Must have length ≥ 6.
            ground_truth (List[int]):
                True POI indices for POI1…POI6. Must have length ≥ 6.
            relative_time (np.ndarray):
                1D array of timestamps (strictly increasing).
            feature_vector (pd.DataFrame):
                Must contain `"Dissipation_smooth"` and `"Resonance_Frequency_smooth"`.
            raw_vector (pd.DataFrame):
                Must contain raw `"Dissipation"` values.

        Returns:
            Dict[str, Dict[str, Any]]: Refined mapping of each POI to:
                - `'indices'`: List[int]
                - `'confidences'`: List[float]

        Raises:
            TypeError: If inputs are of incorrect types.
            ValueError: If required columns are missing, array lengths mismatch,
                        or ground_truth/model_data_labels lengths < 6.
        """
        dissipation = feature_vector["Dissipation_smooth"].values
        dissipation_raw = raw_vector["Dissipation"].values
        """
        Refine POI predictions by applying baseline filtering, timing windows, and bias correction.
        """
        filtered_poi1 = self._filter_poi1_baseline(
            predictions.get('POI1', {}).get('indices', []),
            dissipation_raw,
            relative_time
        )
        predictions['POI1'] = {
            'indices': filtered_poi1,
            'confidences': predictions.get('POI1', {}).get('confidences')
        }
        # copy inputs to avoid side-effects
        best_positions = self._copy_predictions(predictions)
        #
        #  Boost POI6 confidences toward ground truth without overriding original confidences
        if len(ground_truth) >= 6 and len(model_data_labels) >= 6:
            ground_truth[5] = max(ground_truth[5], model_data_labels[5])
        self._filter_poi6_near_ground_truth(
            best_positions, ground_truth, relative_time)
        self._sort_by_confidence(best_positions, 'POI6')

        # determine time anchors
        t0 = self._determine_t0(ground_truth, best_positions, relative_time)
        t3 = self._determine_t3(ground_truth, best_positions, relative_time)
        cut1, cut2 = self._compute_cuts(t0, t3)

        # filter POI4-6 windows
        windows = self._filter_windows(
            best_positions, ground_truth, relative_time, cut1, cut2)

        # apply bias correction and ranking
        for poi in ("POI4", "POI5", "POI6"):
            windows[poi] = self._choose_and_insert(
                windows[poi], dissipation, relative_time, feature_vector, poi=poi, ground_truth=ground_truth)

        # update positions with new windows
        best_positions = self._update_positions(best_positions, windows)

        # handle special -1 cases
        self._handle_negatives(best_positions, ground_truth, relative_time)
        if len(model_data_labels) >= 2:
            best_positions["POI2"]["indices"].insert(0, ground_truth[1])

        # enforce strict ordering across POIs
        best_poi2 = self._poi_2_offset(
            positions=best_positions, feature_vector=feature_vector, relative_time=relative_time)
        best_positions["POI2"]['indices'].insert(0, best_poi2)
        self._enforce_strict_ordering(best_positions, ground_truth)
        self._truncate_confidences(best_positions)

        # make sure best_positions has entries for all POIs
        for i in range(1, 7):
            poi = f"POI{i}"
            # ensure the sub‐dict exists
            entry = best_positions.setdefault(poi, {})
            # ensure both lists exist
            inds = entry.setdefault("indices", [])
            confs = entry.setdefault("confidences", [])

            # if no indices but we have a ground_truth for this POI, inject it
            if not inds and len(ground_truth) >= i:
                inds.append(int(model_data_labels[i - 1]))
                confs.append(1.0)
            # otherwise, if we somehow have indices but no confidences, fill them with 1.0
            elif inds and not confs:
                entry["confidences"] = [1.0] * len(inds)

        return best_positions

    def predict(self,
                file_buffer: str,
                forecast_start: int = -1,
                forecast_end: int = -1,
                actual_poi_indices: Optional[np.ndarray] = None,
                plotting: bool = False) -> Dict[str, Any]:
        """Load data, run QModel v2 clustering + XGBoost prediction, and refine POIs.

        This method validates the input buffer or path, extracts raw and QModel v2
        labels, runs the XGBoost model to get probability predictions, and refines
        them into final POI indices with confidences. Optionally, it can plot the
        dissipation curve and the selected POIs.

        Args:
            file_buffer (str or file-like): Filesystem path to a CSV file containing
                'Relative_time' and 'Dissipation' columns, or a file-like object.
            forecast_start (int): (Unused) Starting index for forecast; must be ≥ -1.
            forecast_end (int): (Unused) Ending index for forecast; must be ≥ -1.
            actual_poi_indices (np.ndarray, optional): Array of true POI indices for
                potential use in downstream evaluation or plotting.
            plotting (bool): If True, displays a matplotlib plot of the dissipation
                curve with POI markers.

        Returns:
            Dict[str, Dict[str, Any]]: A mapping for each "POI1"… "POI6" to a dict with:
                - "indices": List[int] of selected indices.
                - "confidences": List[float] of corresponding confidences.

        Raises:
            TypeError: If arguments are of incorrect types.
            ValueError: If data validation fails, required columns are missing,
                        or no model_data_labels are generated.
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
            Log.e(
                TAG, "Model generated no model data labels. No predictions can be made:")
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
            model_data_labels=model_data_labels,
            relative_time=df["Relative_time"].values,
            feature_vector=feature_vector,
            raw_vector=df)
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

                plt.axvline(df["Relative_time"].iloc[idxs[0]], color=cmap(i))

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
