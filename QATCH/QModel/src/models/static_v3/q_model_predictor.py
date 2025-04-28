#!/usr/bin/env python3
"""
q_model_predictor.py

Provides the QModelPredictor class for predicting Points of Interest (POIs) in dissipation
data using a pre-trained XGBoost booster and an sklearn scaler pipeline. Includes methods for
file validation, feature extraction, probability formatting, and bias correction for refined
POI selection.

Author: Paul MacNichol (paul.macnichol@qatchtech.com)
Date: 04-18-2025
Version: QModel.Ver3.6
"""

import xgboost as xgb

from sklearn.pipeline import Pipeline
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
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

    def _correct_bias(self, dissipation: np.ndarray, relative_time: np.ndarray, candidates: list, poi: str, ground_truth: list) -> int:
        """
        Adjust candidate selection by:
        • iteratively lowering the slope threshold from 100→90th percentile in 0.5 steps,
        • keeping only peaks within [min(candidates), max(candidates)],
        • picking the candidate on the left base of the nearest valid peak,
        • falling back on 90th percentile if needed,
        • and plotting dissipation & slope with annotations.
        """
        t = relative_time
        slope = np.gradient(dissipation, t)
        cmin, cmax = min(candidates), max(candidates)
        chosen_perc = None
        valid_peaks = np.array([], dtype=int)
        valid_bases = []
        valid_filtered = []
        min_iter = 50 if poi == "POI6" else 89.6
        # 1) Try percentiles 100→90 in 0.5 steps
        for perc in np.arange(100.0, min_iter, -0.5):
            thresh = np.percentile(slope, perc)
            peaks, _ = find_peaks(slope, height=thresh)

            # **re-apply the candidate‐range filter**
            peaks = peaks[(peaks >= cmin) & (peaks <= cmax)]
            if peaks.size == 0:
                continue

            # compute bases for each in‐range peak
            bases = []
            for p in peaks:
                j = p
                while j > 0 and slope[j] > 0:
                    j -= 1
                bases.append(j)

            # helper: distance if candidate ≤ at least one peak
            def dist_if_left(c):
                valid = [(p_i, b_i)
                         for p_i, b_i in zip(peaks, bases) if p_i >= c]
                if not valid:
                    return np.inf
                pk, bs = min(valid, key=lambda x: abs(x[0] - c))
                return abs(c - bs)

            filtered = [c for c in candidates if c < np.inf]
            if filtered:
                chosen_perc = perc
                valid_peaks, valid_bases = peaks, bases
                valid_filtered = filtered
                break

        # 2) Fallback at 90th percentile
        if chosen_perc is None:
            Log.w(
                "No valid pick found down to the 90th percentile; falling back on 90th.")
            chosen_perc = min_iter
            thresh = np.percentile(slope, chosen_perc)
            peaks, _ = find_peaks(slope, height=thresh)
            peaks = peaks[(peaks >= cmin) & (peaks <= cmax)]
            valid_peaks = peaks
            valid_bases = []
            for p in peaks:
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

        # 3) Pick best candidate
        best_idx = min(valid_filtered, key=dist_if_left)
        # POI 4 slip-check ---
        if poi == "POI4" and ground_truth is not None and len(ground_truth) >= 4:
            poi3_gt = ground_truth[2]
            poi4_gt = ground_truth[3]
            if abs(best_idx - poi3_gt) < abs(best_idx - poi4_gt):
                Log.w(
                    "Bias correction for POI4 slipped back toward POI3; skipping adjustment.")
                nearest_to_poi4 = min(
                    candidates, key=lambda c: abs(c - poi4_gt))
                best_idx = nearest_to_poi4
        # 4) Plotting
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        # # Dissipation
        # ax1.plot(t, dissipation, lw=1.5, label="Dissipation")
        # ax1.scatter(t[valid_peaks], dissipation[valid_peaks], marker="^", s=100,
        #             c="red", label="Slope Peaks")
        # ax1.scatter(t[valid_bases], dissipation[valid_bases], marker="o", s=80,
        #             c="blue", label="Bases")
        # ax1.scatter(t[candidates], dissipation[candidates], marker="o", s=60,
        #             edgecolors="orange", facecolors="none", label="Candidates")
        # ax1.scatter(t[best_idx], dissipation[best_idx], marker="*", s=200,
        #             c="green", label="Chosen")
        # ax1.set_ylabel("Dissipation")
        # ax1.set_title(
        #     f"Bias Correction (threshold: {chosen_perc:.1f}th percentile)")
        # ax1.legend(loc="upper left")

        # # Slope
        # ax2.plot(t, slope, lw=1.5, label="Slope")
        # ax2.axhline(np.percentile(slope, chosen_perc), color="gray", ls="--",
        #             label=f"{chosen_perc:.1f}th %ile")
        # ax2.scatter(t[valid_peaks], slope[valid_peaks], marker="^", s=100,
        #             c="red", label="Slope Peaks")
        # ax2.scatter(t[valid_bases], slope[valid_bases], marker="o", s=80,
        #             c="blue", label="Bases")
        # ax2.scatter(t[candidates], slope[candidates], marker="o", s=60,
        #             edgecolors="orange", facecolors="none", label="Candidates")
        # ax2.scatter(t[best_idx], slope[best_idx], marker="*", s=200,
        #             c="green", label="Chosen")
        # ax2.set_xlabel("Time")
        # ax2.set_ylabel("Slope")
        # ax2.legend(loc="upper left")

        # plt.tight_layout()
        # plt.show()

        return best_idx

    def _select_best_predictions(
        self,
        predictions: Dict[str, Dict[str, Any]],
        model_data_labels: List[int],
        ground_truth: List[int],
        relative_time: np.ndarray,
        dissipation: np.ndarray,
        dissipation_raw: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Refine POI predictions by applying baseline filtering, timing windows, and bias correction.
        """
        filtered_poi1 = self._filter_poi1_baseline(
            predictions.get('POI1', {}).get('indices', []),
            predictions.get('POI1', {}).get('confidences', []),
            dissipation_raw,
            relative_time
        )
        predictions['POI1'] = {
            'indices': filtered_poi1,
            'confidences': predictions.get('POI1', {}).get('confidences')
        }
        # 1) copy inputs to avoid side-effects
        best_positions = self._copy_predictions(predictions)
        # 1.1) boost POI6 confidences toward ground truth without overriding original confidences
        if len(ground_truth) >= 6 and len(model_data_labels) >= 6:
            ground_truth[5] = max(ground_truth[5], model_data_labels[5])
        self._filter_poi6_near_ground_truth(
            best_positions, ground_truth, relative_time)
        self._sort_by_confidence(best_positions, 'POI6')
        # 2) determine time anchors
        t0 = self._determine_t0(ground_truth, best_positions, relative_time)
        t3 = self._determine_t3(ground_truth, best_positions, relative_time)
        cut1, cut2 = self._compute_cuts(t0, t3)
        # 3) filter POI4-6 windows
        windows = self._filter_windows(
            best_positions, ground_truth, relative_time, cut1, cut2)
        # 4) apply bias correction and ranking
        for poi in ("POI4", "POI5", "POI6"):
            windows[poi] = self._choose_and_insert(
                windows[poi], dissipation, relative_time, poi=poi, ground_truth=ground_truth)
        # 5) update positions with new windows
        best_positions = self._update_positions(best_positions, windows)
        # 6) handle special -1 cases
        self._handle_negatives(best_positions, ground_truth, relative_time)
        # if len(model_data_labels) >= 2:
        #     best_positions["POI2"]["indices"].insert(0, model_data_labels[1])
        # 7) enforce strict ordering across POIs
        self._enforce_strict_ordering(best_positions, ground_truth)
        # 8) truncate confidences to match indices length
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

    def _filter_poi1_baseline(
        self,
        indices: List[int],
        confidences: List[float],
        dissipation: np.ndarray,
        relative_time: np.ndarray
    ) -> List[int]:
        """
        Remove POI1 candidates that lie within baseline dissipation (first 1-3% of run).
        Also plots the baseline window, original candidates, filtered candidates, and best.
        """
        if len(relative_time) < 2:
            return indices
        # define baseline window
        t_start, t_end = relative_time[0], relative_time[-1]
        duration = t_end - t_start
        low_t = t_start + 0.01 * duration
        high_t = t_start + 0.03 * duration
        mask = (relative_time >= low_t) & (relative_time <= high_t)
        if mask.any():
            baseline = float(np.mean(dissipation[mask]))
        else:
            n = max(1, int(0.03 * len(dissipation)))
            baseline = float(np.mean(dissipation[:n]))
        filtered = [i for i in indices if dissipation[i] > baseline]
        return filtered

    def _sort_by_confidence(
        self,
        positions: Dict[str, Any],
        poi_key: str
    ):
        """
        Sort the indices and confidences for a given POI by descending confidence.
        """
        if poi_key not in positions:
            return
        inds = positions[poi_key].get('indices', [])
        confs = positions[poi_key].get('confidences')
        if not inds or confs is None:
            return
        # sort pairs by confidence desc
        pairs = sorted(zip(confs, inds), key=lambda x: x[0], reverse=True)
        sorted_confs, sorted_inds = zip(*pairs)
        positions[poi_key]['indices'] = list(sorted_inds)
        positions[poi_key]['confidences'] = list(sorted_confs)

    def _filter_poi6_near_ground_truth(
        self,
        positions: Dict[str, Any],
        ground_truth: List[int],
        relative_time: np.ndarray,
        pct: float = 0.05
    ):
        """
        Remove POI6 candidates whose time is farther than pct*total_duration from ground-truth POI6.
        """
        key = 'POI6'
        if key not in positions or len(ground_truth) <= 5:
            return
        inds = positions[key]['indices']
        confs = positions[key].get('confidences')
        if confs is None or not inds:
            return
        gt6 = ground_truth[5]
        if not self._valid_index(gt6, relative_time):
            return

        total_duration = relative_time[-1] - relative_time[0]
        threshold = pct * total_duration

        new_inds: List[int] = []
        new_confs: List[float] = []
        for i, c in zip(inds, confs):
            if abs(relative_time[i] - relative_time[gt6]) <= threshold:
                new_inds.append(i)
                new_confs.append(c)

        positions[key]['indices'] = new_inds
        positions[key]['confidences'] = new_confs

    def _copy_predictions(self, predictions):
        return {
            k: {"indices": v["indices"][:], "confidences": v.get(
                "confidences")}  # type: ignore
            for k, v in predictions.items()
        }

    def _determine_t0(self, ground_truth, predictions, relative_time):
        if len(ground_truth) > 2:
            idx = ground_truth[2]
            if self._valid_index(idx, relative_time):
                return relative_time[idx]
        cand = predictions.get("POI3", {}).get("indices", [])
        if cand and self._valid_index(cand[0], relative_time):
            return relative_time[cand[0]]
        return relative_time[0]

    def _determine_t3(self, ground_truth, predictions, relative_time):
        times = []
        if len(ground_truth) > 5 and self._valid_index(ground_truth[5], relative_time):
            times.append(relative_time[ground_truth[5]])
        cand = predictions.get("POI6", {}).get("indices", [])
        if cand and self._valid_index(cand[0], relative_time):
            times.append(relative_time[cand[0]])
        return min(times) if times else relative_time[-1]

    def _compute_cuts(self, t0, t3):
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
    ) -> Dict[str, List[int]]:
        windows = {
            'POI4': [i for i in positions['POI4']['indices'] if relative_time[i] <= cut1],
            'POI5': [i for i in positions['POI5']['indices'] if cut1 <= relative_time[i] <= cut2],
            'POI6': [i for i in positions['POI6']['indices'] if relative_time[i] >= cut2]
        }
        if len(ground_truth) > 3:
            windows['POI4'].append(ground_truth[3])
        if len(ground_truth) > 4:
            windows['POI5'].append(ground_truth[4])
        if len(ground_truth) > 5:
            windows['POI6'].append(ground_truth[5])
        return windows

    def _choose_and_insert(self, indices, dissipation, relative_time, poi, ground_truth):
        if not indices:
            return []
        best = self._correct_bias(
            dissipation, relative_time, indices, poi, ground_truth)
        best_idx = int(np.array(best).flat[0]) if isinstance(
            best, (list, np.ndarray)) else int(best)
        return [best_idx] + [i for i in indices if i != best_idx]

    def _update_positions(self, positions, windows):
        for poi, inds in windows.items():
            positions[poi]['indices'] = inds
        return positions

    def _handle_negatives(self, positions, ground_truth, relative_time):
        for num in (4, 5, 6):
            key = f"POI{num}"
            lst = positions[key]['indices']
            if not lst:
                continue
            if all(i == -1 for i in lst):
                positions[key]['indices'] = [ground_truth[num - 1]]
                continue
            if key == 'POI6' and lst[0] == -1:
                positions[key]['indices'][0] = int(len(relative_time) * 0.98)

    def _enforce_strict_ordering(self, positions, ground_truth):
        for num in range(2, 7):
            key = f"POI{num}"
            prev_key = f"POI{num - 1}"
            inds = positions.get(key, {}).get('indices', [])
            if not inds:
                positions[key]['indices'] = [ground_truth[num - 1]]
                continue
            prev_idx = (
                positions[prev_key]['indices'][0]
                if positions.get(prev_key, {}).get('indices')
                else ground_truth[num - 2]
            )
            if inds[0] <= prev_idx:
                positions[key]['indices'][0] = ground_truth[num - 1]

    def _truncate_confidences(self, positions):
        for data in positions.values():
            inds = data.get('indices', [])
            confs = data.get('confidences')
            if confs is not None:
                data['confidences'] = confs[:len(inds)]

    @staticmethod
    def _valid_index(idx, arr):
        return isinstance(idx, (int, np.integer)) and 0 <= idx < len(arr)

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
            model_data_labels=model_data_labels,
            relative_time=df["Relative_time"].values,
            dissipation=feature_vector["Dissipation_smooth"].values,
            dissipation_raw=df["Dissipation"].values)
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
        print(final_predictions)
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
