"""
q_forecaster.py

This module provides the core classes for data processing and prediction for forecasting the number of
channels filled during a NanovisQ run.

This system is designed to handle raw dissipation and relative time signal data, compute additional features,
and generate predictions using an XGBoost model combined with Viterbi decoding for robust state sequence estimation.

Classes:
    QForecasterDataprocessor:
        - Provides static methods to convert raw data buffers into a pandas DataFrame.
        - Computes additional features from the 'Dissipation' signal, including smoothed signals,
          rolling statistics, derivative features, and envelope extraction using the Hilbert transform.
        - Implements a Viterbi decoding algorithm to infer the most likely sequence of states.
        - Identifies the initial fill point based on baseline noise analysis.

    QForecasterPredictor:
        - Manages model loading, data accumulation, and prediction workflow.
        - Applies preprocessors to the input data, generates predictions using the XGBoost model,
          and refines predictions via Viterbi decoding.
        - Maintains prediction history to assess the stability of predictions over time.
        - Resets accumulated data and handles batch prediction updates.

Module-Level Constants:
    FEATURES (List[str]): List of feature names used in data processing.
    TARGET (str): Target variable name for predictions.
    BASELINE_WINDOW (int): Window size for computing baseline noise to detect fill initiation.

Usage Example:s
    from q_forecaster import QForecasterPredictor, QForecasterDataprocessor

    predictor = QForecasterPredictor(save_dir="/path/to/models")
    predictor.load_models()
    new_data = QForecasterDataProcessor.convert_to_dataframe(worker: QATCH.core.worker.Worker)
    predicted_labels = predictor.update_predictions(new_data)

This module is an integral part of the QForecaster system and is intended to be used in real-time, batch
dissipation data processing.
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple
from scipy.signal import hilbert, savgol_filter
import numpy as np
import pandas as pd
import xgboost as xgb
from QATCH.common.logger import Logger as Log

TAG = '[QForecaster]'
""" Features the model was trained on. """
FEATURES = [
    'Relative_time',
    'Dissipation',
    'Dissipation_rolling_mean',
    'Dissipation_rolling_median',
    'Dissipation_ewm',
    'Dissipation_rolling_std',
    'Dissipation_diff',
    'Dissipation_pct_change',
    'Dissipation_ratio_to_mean',
    'Dissipation_ratio_to_ewm',
    'Dissipation_envelope'
]
""" The target value to make predictions for. """
TARGET = "Fill"

""" The window size to estimate baseline noise. """
BASELINE_WINDOW = 100


class QForecasterDataprocessor:
    """Data processing utilities for QForecaster.

    This class provides static methods to convert raw data buffer to a DataFrame, compute additional
    features on the 'Dissipation' signal, decode hidden state sequences using the Viterbi algorithm,
    and determine the fill point based on a baseline threshold.
    """

    @staticmethod
    def convert_to_dataframe(worker: Any) -> pd.DataFrame:
        """Converts raw buffer data from a worker into a pandas DataFrame.

        Retrieves the relative time and dissipation buffers from the worker, truncates them to the same
        length, and constructs a DataFrame. The DataFrame contains the following columns:
          - 'Relative_time': The relative time data.
          - 'Resonance_Frequency': A duplicate of the dissipation data (placeholder).
          - 'Dissipation': The dissipation data.

        Args:
            worker (QATCH.core.worker.Worker): A Worker that provides buffer data through the methods
                          `get_t1_buffer(index: int)` and `get_d2_buffer(index: int)` where `t1` is the time
                          buffer and `d2` is the dissipation buffer.

        Returns:
            pd.DataFrame: A DataFrame with the columns 'Relative_time', 'Resonance_Frequency', and 'Dissipation'.
        """
        relative_time = worker.get_t1_buffer(0)
        resonance_frequency = worker.get_d1_buffer(0)
        dissipation = worker.get_d2_buffer(0)

        # Determine the minimum length from both buffers
        min_length = min(len(relative_time), len(
            dissipation), len(resonance_frequency))
        # Truncate buffers to the minimum length
        relative_time_truncated = relative_time[:min_length]
        dissipation_truncated = dissipation[:min_length]
        resonance_frequency_truncated = resonance_frequency[:min_length]

        df = pd.DataFrame({
            'Relative_time': relative_time_truncated,
            'Resonance_Frequency': resonance_frequency_truncated,
            'Dissipation': dissipation_truncated
        })

        return df

    @staticmethod
    def compute_additional_features(df: pd.DataFrame) -> pd.DataFrame:
        """Computes additional features based on the 'Dissipation' column.

        Applies a Savitzky-Golay filter to smooth the 'Dissipation' data and then computes a variety of
        rolling and exponential moving average features, along with derivatives and ratios. Finally, the
        signal envelope is extracted using the Hilbert transform, and the 'Resonance_Frequency' column is
        removed if present. Any infinite values are replaced and missing data is filled with zeros.

        Args:
            df (pd.DataFrame): A DataFrame that contains at least the 'Dissipation' column.

        Returns:
            pd.DataFrame: The modified DataFrame with new feature columns including:
                - Dissipation_rolling_mean
                - Dissipation_rolling_median
                - Dissipation_ewm
                - Dissipation_rolling_std
                - Dissipation_diff
                - Dissipation_pct_change
                - Dissipation_ratio_to_mean
                - Dissipation_ratio_to_ewm
                - Dissipation_envelope
        """
        window = 10
        span = 10
        run_length = len(df)
        # Calculate initial window_length as 1% of run_length
        window_length = int(np.ceil(0.01 * run_length))

        # Ensure window_length is odd and at least 3
        if window_length % 2 == 0:
            window_length += 1
        if window_length < 3:
            window_length = 3

        # Adjust window_length if it's greater than the size of the 'Dissipation' data.
        diss_length = len(df['Dissipation'])
        if window_length > diss_length:
            # Set to diss_length, ensuring it's odd.
            window_length = diss_length if diss_length % 2 == 1 else diss_length - 1

        polyorder = 2 if window_length > 2 else 1
        df['Dissipation'] = savgol_filter(df['Dissipation'].values,
                                          window_length=window_length,
                                          polyorder=polyorder)
        df['Dissipation_rolling_mean'] = df['Dissipation'].rolling(
            window=window, min_periods=1).mean()
        df['Dissipation_rolling_median'] = df['Dissipation'].rolling(
            window=window, min_periods=1).median()
        df['Dissipation_ewm'] = df['Dissipation'].ewm(
            span=span, adjust=False).mean()
        df['Dissipation_rolling_std'] = df['Dissipation'].rolling(
            window=window, min_periods=1).std()
        df['Dissipation_diff'] = df['Dissipation'].diff()
        df['Dissipation_pct_change'] = df['Dissipation'].pct_change()
        df['Dissipation_ratio_to_mean'] = df['Dissipation'] / \
            df['Dissipation_rolling_mean']
        df['Dissipation_ratio_to_ewm'] = df['Dissipation'] / df['Dissipation_ewm']
        df['Dissipation_envelope'] = np.abs(hilbert(df['Dissipation'].values))

        # if 'Resonance_Frequency' in df.columns:
        #     df.drop(columns=['Resonance_Frequency'], inplace=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        return df

    @staticmethod
    def viterbi_decode(prob_matrix: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
        """Decodes the most probable sequence of states using the Viterbi algorithm.

        Given an observation probability matrix and a state transition matrix, this method computes the most
        likely sequence of hidden states over time.

        Args:
            prob_matrix (np.ndarray): A 2D array of shape (T, N) representing the observation probabilities,
                                      where T is the number of time steps and N is the number of states.
            transition_matrix (np.ndarray): A 2D array of shape (N, N) representing the transition probabilities
                                            between states.

        Returns:
            np.ndarray: A 1D array of length T representing the most likely sequence of states (as integer indices).
        """
        T, N = prob_matrix.shape
        dp = np.full((T, N), -np.inf)
        backpointer = np.zeros((T, N), dtype=int)
        dp[0, 0] = np.log(prob_matrix[0, 0])
        for t in range(1, T):
            for j in range(N):
                allowed_prev = [0] if j == 0 else [j - 1, j]
                best_state = allowed_prev[0]
                best_score = dp[t - 1, best_state] + \
                    np.log(transition_matrix[best_state, j])
                for i in allowed_prev:
                    if transition_matrix[i, j] <= 0:
                        continue
                    score = dp[t - 1, i] + np.log(transition_matrix[i, j])
                    if score > best_score:
                        best_score = score
                        best_state = i
                dp[t, j] = np.log(prob_matrix[t, j]) + best_score
                backpointer[t, j] = best_state
        best_path = np.zeros(T, dtype=int)
        best_path[T - 1] = np.argmax(dp[T - 1])
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]
        return best_path

    @staticmethod
    def init_fill_point(
        df: pd.DataFrame, baseline_window: int = 10, threshold_factor: float = 1.0
    ) -> int:
        """Identifies the initial fill point based on baseline noise.

        Scans the 'Dissipation' column of the DataFrame to find the first index at which the value exceeds
        a computed threshold. The threshold is defined as the mean of the first `baseline_window` data points
        plus `threshold_factor` times their standard deviation. If the DataFrame does not contain enough data
        or no value exceeds the threshold, -1 is returned.

        Args:
            df (pd.DataFrame): A DataFrame containing at least a 'Dissipation' column.
            baseline_window (int, optional): The number of initial data points used to compute the baseline
                                             mean and standard deviation. Defaults to 10.
            threshold_factor (float, optional): The multiplier for the baseline standard deviation to set the threshold.
                                                Defaults to 3.0.

        Returns:
            int: The index of the first occurrence where the 'Dissipation' value exceeds the threshold,
                 or -1 if no such point is found or if the DataFrame length is insufficient.

        Raises:
            ValueError: If the 'Dissipation' column is not found in the DataFrame.
        """
        if 'Resonance_Frequency' not in df.columns:
            raise ValueError(
                "Resonance_Frequency column not found in DataFrame.")
        if len(df) < baseline_window:
            return -1

        baseline_values = df['Resonance_Frequency'].iloc[:baseline_window]
        baseline_mean = baseline_values.mean()
        baseline_std = baseline_values.std()
        threshold = baseline_mean - threshold_factor * baseline_std
        dissipation = df['Resonance_Frequency'].values

        for idx, value in enumerate(dissipation):
            if value < threshold:
                return idx
        return -1


class QForecasterPredictor:
    """Predictor class for QForecaster that handles model loading, data accumulation, and prediction.

    This class loads a pre-trained XGBoost model along with its associated preprocessors and transition
    matrix. It provides methods to apply preprocessing to input data, generate predictions using the
    model and Viterbi decoding, and maintain a history of predictions to assess stability.
    """

    def __init__(
        self,
        numerical_features: List[str] = FEATURES,
        target: str = 'Fill',
        save_dir: Optional[str] = None,
        batch_threshold: int = 60
    ) -> None:
        """Initializes the QForecasterPredictor.

        Args:
            numerical_features (List[str], optional): List of feature names used for numerical processing.
                Defaults to FEATURES.
            target (str, optional): The target variable name. Defaults to 'Fill'.
            save_dir (Optional[str], optional): Directory path from which to load model files and preprocessors.
                Defaults to None.
            batch_threshold (int, optional): The rate at which to process batches of data. Defaults to 60.
        """
        self.numerical_features: List[str] = numerical_features
        self.target: str = target
        self.save_dir: Optional[str] = save_dir
        self.batch_threshold: int = batch_threshold

        self.model: Optional[xgb.Booster] = None
        self.preprocessors: Optional[Dict[str, Any]] = None
        self.transition_matrix: Optional[np.ndarray] = None

        self.accumulated_data: Optional[pd.DataFrame] = None
        self.batch_num: int = 0
        self.prediction_history: Dict[int, List[Tuple[int, int]]] = {}

    def load_models(self) -> None:
        """Loads the XGBoost model, preprocessors, and transition matrix from the specified directory.

        The method expects to find:
          - "model.json" for the XGBoost model,
          - "preprocessors.pkl" for the preprocessing objects, and
          - "transition_matrix.pkl" for the transition matrix.

        Raises:
            ValueError: If `save_dir` is not specified.
        """
        if not self.save_dir:
            raise ValueError("Save directory not specified.")

        model_path = os.path.join(self.save_dir, "model.json")
        self.model = xgb.Booster()
        self.model.load_model(model_path)

        preprocessors_path = os.path.join(self.save_dir, "preprocessors.pkl")
        with open(preprocessors_path, "rb") as f:
            self.preprocessors = pickle.load(f)

        transition_matrix_path = os.path.join(
            self.save_dir, "transition_matrix.pkl")
        with open(transition_matrix_path, "rb") as f:
            self.transition_matrix = pickle.load(f)

        Log.i(TAG, "All models and associated objects have been loaded successfully.")

    def _apply_preprocessors(
        self, X: pd.DataFrame, preprocessors: Dict[str, Any]
    ) -> np.ndarray:
        """Applies the loaded preprocessors to the input data.

        The method copies the input DataFrame, extracts the numerical features, applies the numerical
        imputer, and then scales the features.

        Args:
            X (pd.DataFrame): Input DataFrame containing the numerical features.
            preprocessors (Dict[str, Any]): Dictionary of preprocessor objects. Must include:
                - 'num_imputer': for imputing missing values.
                - 'scaler': for scaling the data.

        Returns:
            np.ndarray: The preprocessed numerical features as a NumPy array.
        """
        X_copy = X.copy()
        X_num = preprocessors['num_imputer'].transform(
            X_copy[self.numerical_features])
        X_num = preprocessors['scaler'].transform(X_num)
        return X_num

    def predict_native(
        self, model: xgb.Booster, preprocessors: Dict[str, Any], X: pd.DataFrame
    ) -> np.ndarray:
        """Generates predictions using the provided XGBoost model.

        The input DataFrame is filtered to contain only numerical features, preprocessed, converted to an
        XGBoost DMatrix, and then used to produce predictions.

        Args:
            model (xgb.Booster): The pre-trained XGBoost model.
            preprocessors (Dict[str, Any]): Dictionary containing preprocessing objects.
            X (pd.DataFrame): Input DataFrame from which predictions are to be generated.

        Returns:
            np.ndarray: An array of prediction probabilities or scores.
        """
        X = X[self.numerical_features]
        X_processed = self._apply_preprocessors(X, preprocessors)
        dmatrix = xgb.DMatrix(X_processed)
        return model.predict(dmatrix)

    def reset_accumulator(self) -> None:
        """Resets the accumulated data and batch counter.

        This method clears the internal DataFrame that accumulates new data and resets the batch number to zero.
        """
        self.accumulated_data = None
        self.batch_num = 0

    def is_prediction_stable(
        self,
        pred_list: List[Tuple[int, int]],
        stability_window: int = 5,
        frequency_threshold: float = 0.8,
        location_tolerance: int = 0
    ) -> Tuple[bool, Optional[int]]:
        """Determines if the prediction is stable based on its occurrence frequency and location consistency.

        Each element in `pred_list` is a tuple consisting of (location, prediction). A prediction is deemed
        stable if, among the last `stability_window` entries:
          - It appears with a frequency at least equal to `frequency_threshold`, and
          - The difference between the maximum and minimum locations does not exceed `location_tolerance`.

        Args:
            pred_list (List[Tuple[int, int]]): List of tuples where each tuple is (location, prediction).
            stability_window (int, optional): Number of recent predictions to consider for stability. Defaults to 5.
            frequency_threshold (float, optional): Required frequency fraction for stability. Defaults to 0.8.
            location_tolerance (int, optional): Maximum allowed difference between locations for stability.
                Defaults to 0.

        Returns:
            Tuple[bool, Optional[int]]:
                - A boolean indicating if a stable prediction was found.
                - The stable prediction class if stable, otherwise None.
        """
        if len(pred_list) < stability_window:
            return False, None

        recent = pred_list[-stability_window:]
        groups: Dict[int, Dict[str, Any]] = {}
        for loc, pred in recent:
            if pred not in groups:
                groups[pred] = {'count': 0, 'locations': []}
            groups[pred]['count'] += 1
            groups[pred]['locations'].append(loc)

        for pred, data in groups.items():
            frequency = data['count'] / stability_window
            if frequency >= frequency_threshold:
                if max(data['locations']) - min(data['locations']) <= location_tolerance:
                    return True, pred
        return False, None

    def update_predictions(
        self,
        new_data: pd.DataFrame,
        stability_window: int = 5,
        frequency_threshold: float = 0.8,
        confidence_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """Accumulates new data, generates predictions, and assesses prediction stability.

        This method accumulates the new batch of data, computes additional features, identifies the initial
        fill point using the QForecasterDataprocessor methods, and runs predictions on the live data (i.e.,
        data after the fill point). Predictions before the fill point are set to a no-fill class (0). The raw
        predictions are then offset, and Viterbi decoding is applied. Finally, the prediction history is updated,
        and stable predictions are determined based on recent history.

        Args:
            new_data (pd.DataFrame): New batch of data to be accumulated.
            stability_window (int, optional): Number of recent predictions considered for stability. Defaults to 5.
            frequency_threshold (float, optional): Required frequency fraction for a prediction to be stable.
                Defaults to 0.8.
            confidence_threshold (float, optional): Confidence threshold used as location tolerance in the stability
                check. Defaults to 0.6.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "status" (str): "waiting" if the fill has not yet started, "completed" otherwise.
                - "pred" (np.ndarray): Array of predictions.
                - "conf" (np.ndarray): Array of prediction confidences.
                - "stable_predictions" (Dict[int, int]): Mapping from data index to stable prediction class.
                - "accumulated_data" (pd.DataFrame): The accumulated DataFrame with computed features.
                - "accumulated_count" (int): The total number of accumulated entries.
        """
        # Accumulate new data.
        if self.accumulated_data is None:
            self.accumulated_data = pd.DataFrame(columns=new_data.columns)
        self.accumulated_data = pd.concat(
            [self.accumulated_data, new_data], ignore_index=True)
        current_count = len(self.accumulated_data)

        # Compute additional features.
        self.accumulated_data = QForecasterDataprocessor.compute_additional_features(
            self.accumulated_data)

        # Identify the initial fill point.
        init_fill_point = QForecasterDataprocessor.init_fill_point(
            self.accumulated_data, baseline_window=100)

        # If fill hasn't started, report all data as no_fill (0).
        if init_fill_point == -1:
            Log.d(TAG, "Fill not yet started; reporting all data as no_fill (0).")
            predictions = np.zeros(current_count, dtype=int)
            conf = np.zeros(current_count)
            for i in range(current_count):
                if i not in self.prediction_history:
                    self.prediction_history[i] = []
                self.prediction_history[i].append((i, 0))
            return {
                "status": "waiting",
                "pred": predictions,
                "conf": conf,
                "stable_predictions": {},
                "accumulated_data": self.accumulated_data,
                "accumulated_count": current_count
            }

        # For data after the init_fill point, run prediction using Viterbi decoding.
        predictions = np.zeros(current_count, dtype=int)
        conf = np.zeros(current_count)

        # Data before init_fill_point are set to no_fill (0).
        for i in range(init_fill_point):
            if i not in self.prediction_history:
                self.prediction_history[i] = []
            self.prediction_history[i].append((i, 0))

        # Run prediction on live data (after init_fill_point).
        X_live = self.accumulated_data[self.numerical_features].iloc[init_fill_point:]
        prob_matrix = self.predict_native(
            self.model, self.preprocessors, X_live)

        # Perform Viterbi decoding.
        ml_pred = QForecasterDataprocessor.viterbi_decode(
            prob_matrix, self.transition_matrix)
        ml_pred_offset = ml_pred + 1  # Offset raw predictions if needed.

        predictions[init_fill_point:] = ml_pred_offset
        ml_conf = np.array([prob_matrix[i, ml_pred[i]]
                           for i in range(len(ml_pred))])
        conf[init_fill_point:] = ml_conf

        # Update prediction history and check stability.
        for idx, p in enumerate(ml_pred_offset, start=init_fill_point):
            if idx not in self.prediction_history:
                self.prediction_history[idx] = []
            self.prediction_history[idx].append((idx, p))
        stable_predictions: Dict[int, int] = {}
        for idx, history in self.prediction_history.items():
            stable, stable_class = self.is_prediction_stable(
                history, stability_window, frequency_threshold, int(confidence_threshold))
            if stable:
                stable_predictions[idx] = stable_class

        self.batch_num += 1
        Log.d(TAG, f"Running predictions on batch {self.batch_num} with {current_count} entries. "
              f"Predictions are offset by 1 and data before init_fill are marked as no_fill (0).")

        return {
            "status": "completed",
            "pred": predictions,
            "conf": conf,
            "stable_predictions": stable_predictions,
            "accumulated_data": self.accumulated_data,
            "accumulated_count": current_count
        }
