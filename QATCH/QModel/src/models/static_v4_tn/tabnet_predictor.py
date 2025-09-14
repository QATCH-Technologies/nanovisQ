import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Optional, Union
import os

from QATCH.QModel.src.models.static_v4_tn.op_dp import DP
from QATCH.common.logger import Logger as Log
TAG = "[QModel4 (TN)]"


class TabNetPOIPredictor:
    """
    Standalone predictor class for POI detection using trained TabNet model.

    This class handles:
    - Model and scaler loading
    - Feature extraction and preprocessing
    - Window creation with temporal and context features
    - Prediction with temporal smoothing
    - Result formatting
    """

    def __init__(self,
                 model_path: str = './qmodel_tn_clf.zip',
                 scaler_path: str = './qmodel_tn_clf_scaler.joblib',
                 window_size: int = 256,
                 stride: int = 2,
                 tolerance: int = 32,
                 context_size: int = 64,
                 sigma: int = 16,
                 smoothing_window: int = 3,
                 confidence_threshold: float = 0.1):
        """
        Initialize the predictor with model paths and parameters.

        Args:
            model_path: Path to saved TabNet model
            scaler_path: Path to saved scaler
            window_size: Size of sliding window
            stride: Stride for sliding window
            tolerance: Tolerance for POI detection
            context_size: Size of context windows
            sigma: Sigma for soft labels
            smoothing_window: Window size for temporal smoothing
            confidence_threshold: Minimum confidence to report a detection
        """
        self.window_size = window_size
        self.stride = stride
        self.tolerance = tolerance
        self.context_size = context_size
        self.sigma = sigma
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold

        # Load model and scaler
        self.model = self._load_model(model_path)
        self.scaler = self._load_scaler(scaler_path)
        Log.i(
            TAG, f"Successfully loaded TN Model! {self.model}, {self.scaler}")

    def _load_model(self, model_path: str) -> TabNetClassifier:
        """Load the trained TabNet model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = TabNetClassifier()
        model.load_model(model_path)
        return model

    def _load_scaler(self, scaler_path: str):
        """Load the fitted scaler."""
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        return joblib.load(scaler_path)

    @staticmethod
    def gen_clf_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate classification features from raw data.
        This should match the DP.gen_clf_features() method from training.

        Note: You'll need to implement this based on your DP class.
        For now, this is a placeholder that returns the dataframe as-is.
        """
        df = DP.gen_clf_features(df)
        return df

    def create_windows(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Create windows with temporal features and context for prediction.

        Args:
            features_df: DataFrame with extracted features

        Returns:
            Array of windowed features
        """
        features = features_df.values
        n_samples = len(features)
        n_features = features.shape[1]
        windows = []

        # Calculate valid range considering context
        start_idx = self.context_size
        end_idx = n_samples - self.window_size - self.context_size

        # If we don't have enough samples for context, adjust
        if end_idx <= start_idx:
            start_idx = 0
            end_idx = n_samples - self.window_size
            context_size = 0
        else:
            context_size = self.context_size

        if end_idx <= start_idx:
            # Not enough data for even one window
            return np.array([])

        total_windows = len(range(start_idx, end_idx, self.stride))

        for window_idx, i in enumerate(range(start_idx, end_idx, self.stride)):
            # Main window
            window = features[i:i + self.window_size]

            # Flatten main window
            window_flat = window.flatten()

            # === TEMPORAL FEATURES ===
            # 1. Temporal position encoding
            temporal_position = window_idx / max(total_windows - 1, 1)

            # 2. Window statistics
            window_mean = np.mean(window, axis=0)
            window_std = np.std(window, axis=0)
            window_max = np.max(window, axis=0)
            window_min = np.min(window, axis=0)

            # 3. Rate of change features
            if i > 0:
                prev_center_idx = max(
                    0, i - self.stride + self.window_size // 2)
                curr_center_idx = i + self.window_size // 2
                if prev_center_idx < n_samples and curr_center_idx < n_samples:
                    diff_features = features[curr_center_idx] - \
                        features[prev_center_idx]
                else:
                    diff_features = np.zeros(n_features)
            else:
                diff_features = np.zeros(n_features)

            # 4. Trend features
            x_vals = np.arange(self.window_size)
            trend_features = np.zeros(n_features)
            for feat_idx in range(n_features):
                if np.std(window[:, feat_idx]) > 1e-6:
                    coef = np.polyfit(x_vals, window[:, feat_idx], 1)[0]
                    trend_features[feat_idx] = coef

            # === CONTEXT FEATURES ===
            if context_size > 0:
                # Pre-context
                pre_start = max(0, i - context_size)
                pre_end = i
                pre_context = features[pre_start:pre_end]

                if len(pre_context) > 0:
                    pre_mean = np.mean(pre_context, axis=0)
                    pre_std = np.std(pre_context, axis=0)
                    pre_to_window_change = window_mean - pre_mean
                else:
                    pre_mean = np.zeros(n_features)
                    pre_std = np.zeros(n_features)
                    pre_to_window_change = np.zeros(n_features)

                # Post-context
                post_start = i + self.window_size
                post_end = min(n_samples, post_start + context_size)
                post_context = features[post_start:post_end]

                if len(post_context) > 0:
                    post_mean = np.mean(post_context, axis=0)
                    post_std = np.std(post_context, axis=0)
                    window_to_post_change = post_mean - window_mean
                else:
                    post_mean = np.zeros(n_features)
                    post_std = np.zeros(n_features)
                    window_to_post_change = np.zeros(n_features)

                context_features = np.concatenate([
                    pre_mean,
                    pre_std,
                    post_mean,
                    post_std,
                    pre_to_window_change,
                    window_to_post_change
                ])
            else:
                context_features = np.array([])

            # === SPECTRAL FEATURES ===
            spectral_features = []
            for feat_idx in range(min(n_features, 3)):
                fft_vals = np.fft.fft(window[:, feat_idx])
                fft_abs = np.abs(fft_vals[:self.window_size//2])

                if len(fft_abs) > 0:
                    dominant_freq_idx = np.argmax(fft_abs[1:]) + 1
                    dominant_freq = dominant_freq_idx / self.window_size
                    dominant_magnitude = fft_abs[dominant_freq_idx]
                    spectral_energy = np.sum(fft_abs**2)
                else:
                    dominant_freq = 0
                    dominant_magnitude = 0
                    spectral_energy = 0

                spectral_features.extend(
                    [dominant_freq, dominant_magnitude, spectral_energy])

            spectral_features = np.array(spectral_features)

            # === COMBINE ALL FEATURES ===
            enhanced_window = np.concatenate([
                window_flat,
                [temporal_position],
                window_mean,
                window_std,
                window_max,
                window_min,
                diff_features,
                trend_features,
                context_features,
                spectral_features
            ])

            windows.append(enhanced_window)

        return np.array(windows) if windows else np.array([])

    def temporal_smoothing(self, predictions: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to predictions."""
        smoothed = np.zeros_like(predictions)
        for i in range(predictions.shape[1]):
            smoothed[:, i] = gaussian_filter1d(
                predictions[:, i],
                sigma=self.smoothing_window/3
            )
        return smoothed

    def get_window_centers(self, n_windows: int, signal_length: int) -> List[int]:
        """Calculate the center positions of all windows in the original signal."""
        window_centers = []
        for i in range(n_windows):
            window_start = i * self.stride + self.context_size
            window_center = window_start + self.window_size // 2
            window_centers.append(window_center)
        return window_centers

    def predict(self,
                data: pd.DataFrame,
                return_probabilities: bool = False) -> Dict[str, Dict[str, List]]:
        """
        Make predictions on new data.

        Args:
            data: Raw data DataFrame (will be processed with gen_clf_features)
            return_probabilities: If True, also return probability matrix

        Returns:
            Dictionary with POI predictions in format:
            {
                "POI1": {"indices": [idx1, idx2, ...], "confidences": [conf1, conf2, ...]},
                "POI2": {"indices": [...], "confidences": [...]},
                ...
                "POI6": {"indices": [...], "confidences": [...]}
            }
        """
        # Extract features
        features_df = self.gen_clf_features(data)

        # Create windows
        X = self.create_windows(features_df)

        if len(X) == 0:
            # No valid windows could be created
            return {f"POI{i}": {"indices": [], "confidences": []} for i in range(1, 7)}

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get predictions
        probs = self.model.predict_proba(X_scaled)

        # Apply temporal smoothing
        probs_smoothed = self.temporal_smoothing(probs)

        # Get window centers
        window_centers = self.get_window_centers(len(X), len(features_df))

        # Prepare results dictionary
        results = {}

        # Process each POI (classes 1-6, skipping background class 0)
        for poi_num in range(1, 7):
            poi_name = f"POI{poi_num}"
            poi_class = poi_num  # Class index in probability array

            # Get probabilities for this POI
            poi_probs = probs_smoothed[:, poi_class]

            # Find peaks above threshold
            indices = []
            confidences = []

            # Simple peak detection: find local maxima above threshold
            for i in range(1, len(poi_probs) - 1):
                if (poi_probs[i] > self.confidence_threshold and
                    poi_probs[i] >= poi_probs[i-1] and
                        poi_probs[i] >= poi_probs[i+1]):

                    if i < len(window_centers):
                        indices.append(int(window_centers[i]))
                        confidences.append(float(poi_probs[i]))

            # Alternative: Take top N predictions for this POI
            if not indices:  # If no peaks found, take the highest probability point
                top_idx = np.argmax(poi_probs)
                if poi_probs[top_idx] > self.confidence_threshold and top_idx < len(window_centers):
                    indices.append(int(window_centers[top_idx]))
                    confidences.append(float(poi_probs[top_idx]))

            results[poi_name] = {
                "indices": indices,
                "confidences": confidences
            }

        if return_probabilities:
            return results, probs_smoothed

        return results

    def predict_from_file(self,
                          csv_path: str,
                          return_probabilities: bool = False) -> Dict[str, Dict[str, List]]:
        """
        Make predictions from a CSV file.

        Args:
            csv_path: Path to CSV file
            return_probabilities: If True, also return probability matrix

        Returns:
            Dictionary with POI predictions
        """
        df = pd.read_csv(csv_path, engine='pyarrow')
        return self.predict(df, return_probabilities)

    def _reset_file_buffer(self, file_buffer: Union[str, object]) -> Union[str, object]:
        """Resets the file buffer to the beginning for reading.

        This method ensures that a file-like object is positioned at the start
        so it can be read from the beginning. If the input is a file path
        (string), it is returned unchanged.

        Args:
            file_buffer (Union[str, object]): A file path or file-like object
                supporting `seek()`.

        Returns:
            Union[str, object]: The same file path or the reset file-like object.

        Raises:
            Exception: If the file-like object does not support seeking.
        """
        """Ensure the file buffer is positioned at its beginning for reading."""
        if isinstance(file_buffer, str):
            return file_buffer
        if hasattr(file_buffer, "seekable") and file_buffer.seekable():
            file_buffer.seek(0)
            return file_buffer
        else:
            raise Exception(
                "Cannot seek stream prior to passing to processing.")

    def _validate_file_buffer(self, file_buffer: Union[str, object]) -> pd.DataFrame:
        """Loads and validates CSV data from a file or file-like object.

        This method reads CSV data into a pandas DataFrame, ensures it contains
        required columns, and checks that it is not empty. It first resets the
        buffer position if a file-like object is provided.

        Args:
            file_buffer (Union[str, object]): Path to a CSV file or a file-like
                object containing CSV data.

        Returns:
            pd.DataFrame: The loaded and validated DataFrame containing the CSV data.

        Raises:
            ValueError: If the file buffer cannot be read, the CSV is empty, or
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
            raise ValueError(f"Error parsing data file: {e}")
        except Exception as e:
            raise ValueError(
                f"An unexpected error occurred while reading the data file: {e}")

        # Validate DataFrame contents
        if df.empty:
            raise ValueError("The data file does not contain any data.")

        required_columns = {"Dissipation",
                            "Resonance_Frequency", "Relative_time"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Data file missing required columns: {', '.join(missing)}.")

        return df

    def _get_default_predictions(self) -> Dict[str, Dict[str, List]]:
        """Returns a default POI prediction dictionary with placeholder values.

        All POIs are set to `-1` for both indices and confidences scores,
        indicating that no prediction was made.

        Returns:
            Dict[str, Dict[str, List]]: Dictionary of default POI predictions with
            'indices' and 'confidences' set to [-1] for each POI.
        """
        return {
            "POI1": {"indices": [-1], "confidences": [-1]},
            "POI2": {"indices": [-1], "confidences": [-1]},
            "POI3": {"indices": [-1], "confidences": [-1]},
            "POI4": {"indices": [-1], "confidences": [-1]},
            "POI5": {"indices": [-1], "confidences": [-1]},
            "POI6": {"indices": [-1], "confidences": [-1]}
        }

    def get_top_predictions(self,
                            file_buffer,
                            top_n: int = 3) -> Dict[str, Dict[str, List]]:
        """
        Get top N predictions for each POI based on confidence.

        Args:
            data: Raw data DataFrame
            top_n: Number of top predictions to return per POI

        Returns:
            Dictionary with top N predictions for each POI
        """
        if file_buffer is not None:
            try:
                data = self._validate_file_buffer(file_buffer=file_buffer)
            except Exception as e:
                Log.d(f"File buffer could not be validated: {e}")
                return self._get_default_predictions()
        elif data is None:
            raise ValueError(
                "Either file_buffer or dataframe must be provided")

        # Get all predictions with probabilities
        _, probs_smoothed = self.predict(data, return_probabilities=True)

        # Extract features to get signal length
        features_df = self.gen_clf_features(data)

        # Get window centers
        window_centers = self.get_window_centers(
            len(probs_smoothed), len(features_df))

        results = {}

        for poi_num in range(1, 7):
            poi_name = f"POI{poi_num}"
            poi_class = poi_num

            # Get probabilities for this POI
            poi_probs = probs_smoothed[:, poi_class]

            # Get top N indices
            top_indices = np.argsort(poi_probs)[-top_n:][::-1]

            indices = []
            confidences = []

            for idx in top_indices:
                if poi_probs[idx] > self.confidence_threshold and idx < len(window_centers):
                    indices.append(int(window_centers[idx]))
                    confidences.append(float(poi_probs[idx]))

            results[poi_name] = {
                "indices": indices,
                "confidences": confidences
            }

        return results


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    # Initialize predictor
    predictor = TabNetPOIPredictor(
        model_path='./qmodel_tn_clf.zip',
        scaler_path='./qmodel_tn_clf_scaler.joblib',
        confidence_threshold=0.1
    )

    # Load your data
    df = pd.read_csv(
        'content/valid/02428/M240913W11_10CP_I9_3rd.csv', engine='pyarrow')

    # # Make predictions
    # predictions = predictor.predict(df)

    # # Or predict from file directly
    # predictions = predictor.predict_from_file(
    #     'content/valid/00026/M240625W10B_DMSO_B1_3rd.csv')
    predictions = predictor.get_top_predictions(df, top_n=3)

    # --- Plot Dissipation Curve ---
    plt.figure(figsize=(12, 6))
    plt.plot(df["Dissipation"].values, color="black",
             label="Dissipation Curve")

    # --- Overlay Predictions ---
    colors = plt.cm.tab10.colors  # cycle through distinct colors
    for i, (poi, data) in enumerate(predictions.items()):
        indices = np.array(data["indices"])
        confidences = np.array(data["confidences"])

        # Plot vertical lines for predicted POIs
        plt.vlines(
            indices,
            ymin=min(df["Dissipation"].values),
            ymax=max(df["Dissipation"].values),
            colors=colors[i % len(colors)],
            linestyles="dashed",
            alpha=0.7,
            label=f"{poi} (mean conf={confidences.mean():.2f})"
        )

        # Optional: Scatter markers at predicted points
        plt.scatter(
            indices,
            df["Dissipation"].values[indices],
            color=colors[i % len(colors)],
            s=50,
            edgecolor="k",
            zorder=5
        )

    # --- Labels & Legend ---
    plt.xlabel("Sample Index")
    plt.ylabel("Dissipation")
    plt.title("Dissipation Curve with Predicted POIs")
    plt.legend()
    plt.tight_layout()
    plt.show()
