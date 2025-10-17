import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Union, List, Any, Optional
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import json

warnings.filterwarnings('ignore')

try:
    from QATCH.common.logger import Logger as Log
    from QATCH.QModel.src.models.static_v4_fusion.v4_fusion_dataprocessor import FusionDataprocessor
    from QATCH.models.ModelData import ModelData
    from QATCH.QModel.src.models.static_v4_fusion.v4_fusion_partial_fill import PartialFillDetector
except (ImportError, ModuleNotFoundError):
    from v4_fusion_dataprocessor import FusionDataprocessor
    from ModelData import ModelData

    class Log:
        @staticmethod
        def d(tag: str = "", message: str = ""):
            print(f"{tag} [DEBUG] {message}")

        @staticmethod
        def i(tag: str = "", message: str = ""):
            print(f"{tag} [INFO] {message}")

        @staticmethod
        def w(tag: str = "", message: str = ""):
            print(f"{tag} [WARNING] {message}")

        @staticmethod
        def e(tag: str = "", message: str = ""):
            print(f"{tag} [ERROR] {message}")


# ============================================================================
# REGRESSION MODELS (for POI 1 and 2)
# ============================================================================

class RegModel(nn.Module):
    """Feed-Forward Network for regression."""

    def __init__(self, window_size: int, feature_dim: int, config: Dict[str, Any]):
        super(RegModel, self).__init__()
        input_dim = window_size * feature_dim
        hidden_1 = config.get('hidden_1', 256)
        hidden_2 = config.get('hidden_2', 128)
        hidden_3 = config.get('hidden_3', 64)
        dropout = config.get('dropout', 0.3)

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, hidden_3),
            nn.BatchNorm1d(hidden_3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze()


class RegPredictor:
    """Predictor for regression-based POI detection (POI 1 and 2)."""

    def __init__(self, model_path: str, batch_size: int = 512, poi_name: str = "POI"):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None
        self._scaler = StandardScaler()
        self._config = None
        self._window_size = None
        self._stride = None
        self._tolerance = None
        self._feature_dim = None
        self._gaussian_sigma = 2.0
        self._peak_threshold = 0.3
        self._batch_size = batch_size
        self.poi_name = poi_name

        # Visualization data
        self._last_predictions = None
        self._last_smoothed = None
        self._last_window_positions = None
        self._last_data = None

        self._load(model_path=model_path)

    def _load(self, model_path: str):
        """Load the trained regression model."""
        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False)

            self._config = checkpoint['config']
            self._window_size = checkpoint['window_size']
            self._stride = checkpoint['stride']
            self._tolerance = checkpoint['tolerance']
            self._feature_dim = checkpoint['feature_dim']
            self._gaussian_sigma = checkpoint.get('gaussian_sigma', 2.0)
            self._peak_threshold = checkpoint.get('peak_threshold', 0.3)

            self._scaler.mean_ = checkpoint['scaler_mean']
            self._scaler.scale_ = checkpoint['scaler_scale']

            self._model = RegModel(
                self._window_size, self._feature_dim, self._config
            ).to(self.device)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.eval()

            Log.d("RegPredictor", f"Model loaded from {model_path}")
        except Exception as e:
            Log.e("RegPredictor", f"Failed to load model: {e}")
            raise

    def predict(self, data: pd.DataFrame, progress_signal=None, visualize: bool = False, poi2_idx: int = -1) -> Tuple[int, float]:
        """
        Predict POI location from data.

        Returns:
            Tuple of (predicted_index, confidence_score)
        """
        if self._model is None:
            raise ValueError("Model not loaded")

        # Store data for visualization
        self._last_data = data.copy()

        # Generate features
        features_df = FusionDataprocessor.get_reg_features(data)
        result = self._predict_from_features(
            features_df, progress_signal, visualize, poi2_idx=poi2_idx)

        if visualize:
            self.visualize()

        return result

    def _predict_from_features(self, features_df: pd.DataFrame, progress_signal=None, visualize: bool = False, poi2_idx: int = -1) -> Tuple[int, float]:
        """
        Predict POI location from pre-computed features.
        This allows feature reuse between POI1 and POI2 models.

        Args:
            poi2_idx: If not -1, only peaks to the left of (less than) this index are considered

        Returns:
            Tuple of (predicted_index, confidence_score)
        """
        if self._model is None:
            raise ValueError("Model not loaded")
        features = features_df.values
        n_samples = len(features)
        # Create sliding windows
        windows = []
        window_positions = []
        for i in range(0, n_samples - self._window_size + 1, self._stride):
            window = features[i:i + self._window_size]
            windows.append(window)
            window_center = i + self._window_size // 2
            window_positions.append(window_center)
        # Handle tail
        last_start = (n_samples - self._window_size + 1)
        if last_start % self._stride != 0:
            start_idx = n_samples - self._window_size
            tail = features[start_idx:]
            pad_len = self._window_size - len(tail)
            pad_values = np.repeat(tail[-1][None, :], pad_len, axis=0)
            window = np.vstack([tail, pad_values])
            windows.append(window)
            window_center = start_idx + self._window_size // 2
            window_positions.append(window_center)
        if len(windows) == 0:
            return -1, -1
        windows = np.array(windows)
        window_positions = np.array(window_positions)
        # Normalize
        flat = windows.reshape(-1, self._feature_dim)
        flat = self._scaler.transform(flat)
        windows = flat.reshape(windows.shape)
        # Predict with batching
        op_predictions = self._predict_batched(windows)
        # Apply smoothing
        # op_smoothed = gaussian_filter1d(
        #     op_predictions, sigma=self._gaussian_sigma)
        op_smoothed = op_predictions
        # Store for visualization
        self._last_predictions = op_predictions
        self._last_smoothed = op_smoothed
        self._last_window_positions = window_positions
        # Find peaks
        peaks, _ = find_peaks(
            op_smoothed,
            height=self._peak_threshold,
            distance=self._tolerance // self._stride
        )
        if len(peaks) > 0:
            # Filter peaks based on poi2_idx constraint if specified
            if poi2_idx != -1:
                # Keep only peaks that are to the left of (less than) poi2_idx
                valid_peaks = peaks[window_positions[peaks] < poi2_idx]
                if len(valid_peaks) == 0:
                    Log.e("No valid peaks found before poi2_idx in REG prediction.")
                    return -1, -1
                peaks = valid_peaks

            # Select the most prominent valid peak
            max_idx = peaks[np.argmax(op_smoothed[peaks])]
            pred_index = int(window_positions[max_idx])
            confidence = float(op_smoothed[max_idx])
        else:
            Log.e("No significant peaks found in in REG prediction.")
            return -1, -1
        return pred_index, confidence

    def _predict_batched(self, windows: np.ndarray) -> np.ndarray:
        """
        Predict with automatic batching and OOM handling.

        Args:
            windows: Array of shape (n_windows, window_size, n_features)

        Returns:
            Array of predictions
        """
        self._model.eval()
        predictions = []
        batch_size = self._batch_size
        n_windows = len(windows)

        with torch.no_grad():
            i = 0
            while i < n_windows:
                try:
                    # Get batch
                    batch_end = min(i + batch_size, n_windows)
                    batch = windows[i:batch_end]

                    # Convert to tensor and predict
                    batch_tensor = torch.FloatTensor(batch).to(self.device)
                    batch_preds = self._model(batch_tensor).cpu().numpy()

                    predictions.append(batch_preds)

                    # Clear GPU memory
                    del batch_tensor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    i = batch_end

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Reduce batch size and retry
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        batch_size = max(1, batch_size // 2)
                        Log.w("RegPredictor",
                              f"OOM error, reducing batch size to {batch_size}")

                        if batch_size < 1:
                            raise RuntimeError(
                                "Cannot reduce batch size further. Out of memory.")
                    else:
                        raise

        return np.concatenate(predictions) if len(predictions) > 1 else predictions[0]

    def visualize(self, figsize: Tuple[int, int] = (14, 8), save_path: Optional[str] = None):
        """
        Visualize the regression prediction results.

        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
        """
        if self._last_data is None or self._last_predictions is None:
            raise ValueError(
                "No prediction data available. Run predict() first.")

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Get dissipation data
        dissipation = self._last_data['Dissipation'].values
        time_indices = np.arange(len(dissipation))

        # Find predicted POI
        if self._last_smoothed is not None:
            max_idx = np.argmax(self._last_smoothed)
            pred_index = int(self._last_window_positions[max_idx])
            confidence = float(self._last_smoothed[max_idx])
        else:
            pred_index = -1
            confidence = 0.0

        # Plot 1: Dissipation signal with POI marker
        axes[0].plot(time_indices, dissipation, 'b-', alpha=0.7, linewidth=1)
        if pred_index != -1:
            axes[0].axvline(pred_index, color='red', linestyle='--', linewidth=2,
                            label=f'{self.poi_name} (conf={confidence:.3f})')
            axes[0].plot(pred_index, dissipation[pred_index], 'r*',
                         markersize=15, markeredgewidth=2, markerfacecolor='none')
        axes[0].set_ylabel('Dissipation', fontsize=11, fontweight='bold')
        axes[0].set_title(
            f'{self.poi_name} Detection - Dissipation Signal', fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Raw predictions
        axes[1].plot(self._last_window_positions, self._last_predictions,
                     'g-', alpha=0.6, linewidth=1.5, label='Raw Predictions')
        axes[1].axhline(self._peak_threshold, color='orange', linestyle=':',
                        linewidth=1, label=f'Threshold={self._peak_threshold}')
        if pred_index != -1:
            axes[1].axvline(pred_index, color='red',
                            linestyle='--', linewidth=2, alpha=0.5)
        axes[1].set_ylabel('Confidence', fontsize=11, fontweight='bold')
        axes[1].set_title('Model Raw Output', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])

        # Plot 3: Smoothed predictions
        axes[2].plot(self._last_window_positions, self._last_smoothed,
                     'purple', linewidth=2, label=f'Smoothed (σ={self._gaussian_sigma})')
        axes[2].axhline(self._peak_threshold, color='orange', linestyle=':',
                        linewidth=1, label=f'Threshold={self._peak_threshold}')

        # Mark peaks
        peaks, _ = find_peaks(
            self._last_smoothed,
            height=self._peak_threshold,
            distance=self._tolerance // self._stride
        )
        if len(peaks) > 0:
            peak_positions = self._last_window_positions[peaks]
            peak_values = self._last_smoothed[peaks]
            axes[2].plot(peak_positions, peak_values, 'r^',
                         markersize=10, label='Detected Peaks')

        if pred_index != -1:
            axes[2].axvline(pred_index, color='red',
                            linestyle='--', linewidth=2, alpha=0.5)
            axes[2].plot(pred_index, confidence, 'r*',
                         markersize=15, markeredgewidth=2, markerfacecolor='none')

        axes[2].set_ylabel('Confidence', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Sample Index', fontsize=11, fontweight='bold')
        axes[2].set_title(
            'Smoothed Model Output with Peak Detection', fontsize=12, fontweight='bold')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            Log.i("RegPredictor", f"Visualization saved to {save_path}")

        plt.show()


# ============================================================================
# CLASSIFICATION MODELS (for POI 3, 4, 5, 6)
# ============================================================================

class ClfModel(nn.Module):
    """Multi-label classification model for POI detection."""

    def __init__(self, window_size: int, feature_dim: int,
                 conv_filters_1: int = 96, conv_filters_2: int = 256,
                 kernel_size: int = 3, lstm_units_1: int = 256,
                 lstm_units_2: int = 96, dense_units: int = 256,
                 dropout_1: float = 0.3, dropout_2: float = 0.3,
                 dropout_3: float = 0.2):
        super(ClfModel, self).__init__()

        self.conv1 = nn.Conv1d(feature_dim, conv_filters_1,
                               kernel_size=kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(conv_filters_1)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(conv_filters_1, conv_filters_2,
                               kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(conv_filters_2)
        self.pool2 = nn.MaxPool1d(2)

        self.lstm1 = nn.LSTM(conv_filters_2, lstm_units_1,
                             batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_1)

        self.lstm2 = nn.LSTM(lstm_units_1 * 2, lstm_units_2,
                             batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_2)

        self.fc1 = nn.Linear(lstm_units_2 * 2, dense_units)
        self.bn3 = nn.BatchNorm1d(dense_units)
        self.dropout3 = nn.Dropout(dropout_3)

        self.fc2 = nn.Linear(dense_units, 4)  # Non-POI, POI4, POI5, POI6

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = x.transpose(1, 2)

        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x = x[:, -1, :]

        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


class ClfPredictor:
    """Predictor for classification-based POI detection (POI 4, 5, 6)."""

    def __init__(self, model_path: str, batch_size: int = 256):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None
        self._scaler = StandardScaler()
        self._window_size = None
        self._stride = None
        self._tolerance = None
        self._feature_dim = None
        self._best_params = None
        self._batch_size = batch_size

        # Visualization data
        self._last_predictions = None
        self._last_window_positions = None
        self._last_data = None
        self._last_results = None

        self._load(model_path=model_path)

    def _load(self, model_path: str):
        """Load the trained classification model."""
        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False)
            self._window_size = checkpoint['window_size']
            self._stride = checkpoint['stride']
            self._tolerance = checkpoint['tolerance']
            self._feature_dim = checkpoint['feature_dim']
            self._best_params = checkpoint['best_params']

            self._scaler.mean_ = checkpoint['scaler_mean']
            self._scaler.scale_ = checkpoint['scaler_scale']

            if self._best_params:
                self._model = ClfModel(
                    window_size=self._window_size,
                    feature_dim=self._feature_dim,
                    **{k: v for k, v in self._best_params.items() if k not in ['learning_rate', 'focal_alpha', 'focal_gamma']}
                ).to(self.device)
            else:
                self._model = ClfModel(
                    window_size=self._window_size,
                    feature_dim=self._feature_dim
                ).to(self.device)

            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.eval()

            Log.d("ClfPredictor", f"Model loaded from {model_path}")
        except Exception as e:
            Log.e("ClfPredictor", f"Failed to load model: {e}")
            raise

    def predict(self, data: pd.DataFrame, visualize: bool = False) -> Dict[int, Tuple[int, float]]:
        """
        Predict POI locations from data.
        Returns:
            Dictionary mapping POI number (4, 5, 6) to (index, confidence)
        """
        if self._model is None:
            raise ValueError("Model not loaded")

        # Store data for visualization
        self._last_data = data.copy()

        # Generate features
        features_df = FusionDataprocessor.get_clf_features(data)
        features = features_df.values

        # Create sliding windows
        windows = []
        window_positions = []
        n = len(features)

        for i in range(0, n, self._stride):
            window = features[i:i + self._window_size]
            if len(window) < self._window_size:
                last_val = features[-1]
                pad_len = self._window_size - len(window)
                pad = np.tile(last_val, (pad_len, 1))
                window = np.vstack([window, pad])
            windows.append(window)
            window_positions.append(min(i + self._window_size // 2, n - 1))

        windows = np.array(windows)
        window_positions = np.array(window_positions)

        # Normalize
        windows_reshaped = windows.reshape(-1, windows.shape[-1])
        windows_reshaped = self._scaler.transform(windows_reshaped)
        windows = windows_reshaped.reshape(windows.shape)

        # Predict with batching
        predictions = self._predict_batched(windows)

        # Store for visualization
        self._last_predictions = predictions
        self._last_window_positions = window_positions

        # Extract POI locations using most prominent peak
        poi_indices = {4: 1, 5: 2, 6: 3}  # Map POI number to prediction index
        poi_locations = {}

        for poi_num, pred_idx in poi_indices.items():
            poi_probs = predictions[:, pred_idx]
            # Find the window with maximum probability
            best_idx = np.argmax(poi_probs)
            poi_locations[poi_num] = (
                int(window_positions[best_idx]),
                float(poi_probs[best_idx])
            )

        self._last_results = poi_locations

        if visualize:
            self.visualize()

        return poi_locations

    def _predict_batched(self, windows: np.ndarray) -> np.ndarray:
        """
        Predict with automatic batching and OOM handling.

        Args:
            windows: Array of shape (n_windows, window_size, n_features)

        Returns:
            Array of predictions
        """
        self._model.eval()
        predictions = []
        batch_size = self._batch_size
        n_windows = len(windows)

        with torch.no_grad():
            i = 0
            while i < n_windows:
                try:
                    # Get batch
                    batch_end = min(i + batch_size, n_windows)
                    batch = windows[i:batch_end]

                    # Convert to tensor and predict
                    batch_tensor = torch.FloatTensor(batch).to(self.device)
                    batch_preds = self._model(batch_tensor).cpu().numpy()

                    predictions.append(batch_preds)

                    # Clear GPU memory
                    del batch_tensor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    i = batch_end

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Reduce batch size and retry
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        batch_size = max(1, batch_size // 2)
                        Log.w("ClfPredictor",
                              f"OOM error, reducing batch size to {batch_size}")

                        if batch_size < 1:
                            raise RuntimeError(
                                "Cannot reduce batch size further. Out of memory.")
                    else:
                        raise

        return np.concatenate(predictions) if len(predictions) > 1 else predictions[0]

    def _non_maximum_suppression(self, probs: np.ndarray, mask: np.ndarray,
                                 nms_window: int) -> np.ndarray:
        """Apply non-maximum suppression."""
        peaks = []
        probs_copy = probs.copy()

        while True:
            valid_indices = np.where(mask & (probs_copy > 0))[0]
            if len(valid_indices) == 0:
                break

            max_idx = valid_indices[np.argmax(probs_copy[valid_indices])]
            peaks.append(max_idx)

            start = max(0, max_idx - nms_window // 2)
            end = min(len(probs_copy), max_idx + nms_window // 2 + 1)
            probs_copy[start:end] = 0

        return np.array(peaks)

    def visualize(self, figsize: Tuple[int, int] = (14, 10), save_path: Optional[str] = None):
        """
        Visualize the classification prediction results.

        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
        """
        if self._last_data is None or self._last_predictions is None:
            raise ValueError(
                "No prediction data available. Run predict() first.")

        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

        # Get dissipation data
        dissipation = self._last_data['Dissipation'].values
        time_indices = np.arange(len(dissipation))

        # Define POI colors and thresholds
        poi_info = {
            4: {'color': 'orange', 'threshold': 0.7, 'pred_idx': 1},
            5: {'color': 'green', 'threshold': 0.6, 'pred_idx': 2},
            6: {'color': 'purple', 'threshold': 0.65, 'pred_idx': 3}
        }

        # Plot 1: Dissipation signal with all POI markers
        axes[0].plot(time_indices, dissipation, 'b-', alpha=0.7, linewidth=1)

        for poi_num, info in poi_info.items():
            if poi_num in self._last_results:
                idx, conf = self._last_results[poi_num]
                axes[0].axvline(idx, color=info['color'], linestyle='--', linewidth=2,
                                label=f'POI{poi_num} (conf={conf:.3f})')
                axes[0].plot(idx, dissipation[idx], color=info['color'], marker='*',
                             markersize=15, markeredgewidth=2, markerfacecolor='none')

        axes[0].set_ylabel('Dissipation', fontsize=11, fontweight='bold')
        axes[0].set_title('POI Detection - Dissipation Signal',
                          fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Plot 2-4: Individual POI confidence scores
        for plot_idx, poi_num in enumerate([4, 5, 6], start=1):
            info = poi_info[poi_num]
            pred_idx = info['pred_idx']
            poi_probs = self._last_predictions[:, pred_idx]

            axes[plot_idx].plot(self._last_window_positions, poi_probs,
                                color=info['color'], linewidth=2, label=f'POI{poi_num} Confidence')
            axes[plot_idx].axhline(info['threshold'], color='red', linestyle=':',
                                   linewidth=1, label=f"Threshold={info['threshold']}")

            # Mark detected POI
            if poi_num in self._last_results:
                idx, conf = self._last_results[poi_num]
                axes[plot_idx].axvline(idx, color=info['color'], linestyle='--',
                                       linewidth=2, alpha=0.5)
                axes[plot_idx].plot(idx, conf, color=info['color'], marker='*',
                                    markersize=12, markeredgewidth=2, markerfacecolor='none')

            axes[plot_idx].set_ylabel(
                'Confidence', fontsize=10, fontweight='bold')
            axes[plot_idx].set_title(
                f'POI{poi_num} Detection Output', fontsize=11, fontweight='bold')
            axes[plot_idx].legend(loc='upper right', fontsize=9)
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].set_ylim([0, 1])

        axes[-1].set_xlabel('Sample Index', fontsize=11, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            Log.i("ClfPredictor", f"Visualization saved to {save_path}")

        plt.show()


# ============================================================================
# FUSION PREDICTOR
# ============================================================================

class QModelV4Fusion:
    """Main fusion predictor combining regression and classification models."""

    TAG = "V4Fusion"

    def __init__(self, reg_path_1: str, reg_path_2: str, clf_path: str,
                 reg_batch_size: int = 512, clf_batch_size: int = 256):
        """
        Initialize V4Fusion predictor.

        Args:
            reg_path_1: Path to POI1 regression model
            reg_path_2: Path to POI2 regression model
            clf_path: Path to classification model (POI4, 5, 6)
            reg_batch_size: Batch size for regression models (default: 512)
                           Larger batches are faster but use more GPU memory
            clf_batch_size: Batch size for classification model (default: 256)
                           Smaller due to more complex model architecture

        Note:
            - Batch sizes automatically reduce if GPU runs out of memory
            - POI1 and POI2 share feature computation for efficiency
            - GPU memory is cleared between model predictions
        """
        self.poi_1_model = RegPredictor(
            model_path=reg_path_1, batch_size=reg_batch_size, poi_name="POI1")
        self.poi_2_model = RegPredictor(
            model_path=reg_path_2, batch_size=reg_batch_size, poi_name="POI2")
        self.poi_345_model = ClfPredictor(
            model_path=clf_path, batch_size=clf_batch_size)

        # Store last prediction data for visualization
        self._last_data = None
        self._last_predictions = None

        Log.d(
            self.TAG, f"Initialized with batch sizes: reg={reg_batch_size}, clf={clf_batch_size}")

    def _get_default_predictions(self) -> Dict[str, Dict[str, List]]:
        """Returns default POI prediction dictionary with placeholder values."""
        return {
            "POI1": {"indices": [-1], "confidences": [-1]},
            "POI2": {"indices": [-1], "confidences": [-1]},
            "POI3": {"indices": [-1], "confidences": [-1]},
            "POI4": {"indices": [-1], "confidences": [-1]},
            "POI5": {"indices": [-1], "confidences": [-1]},
            "POI6": {"indices": [-1], "confidences": [-1]}
        }

    def _format_output(
        self,
        final_positions: Dict[int, int],
        confidence_scores: Dict[int, float]
    ) -> Dict[str, Dict[str, List[float]]]:
        """Format predictions into output dictionary."""
        poi_map = {1: 'POI1', 2: 'POI2', 3: 'POI3',
                   4: 'POI4', 5: 'POI5', 6: 'POI6'}

        output = {}
        for poi_num, poi_name in poi_map.items():
            if poi_num in final_positions:
                idx = final_positions[poi_num]
                conf = confidence_scores.get(poi_num, 0.0)
                output[poi_name] = {
                    'indices': [idx],
                    'confidences': [conf]
                }
            else:
                output[poi_name] = {
                    'indices': [-1],
                    'confidences': [-1]
                }

        # Sort by confidence
        for poi_name in output:
            zipped = list(
                zip(output[poi_name]['indices'],
                    output[poi_name]['confidences'])
            )
            zipped.sort(key=lambda x: x[1], reverse=True)
            output[poi_name]['indices'] = [z[0] for z in zipped]
            output[poi_name]['confidences'] = [z[1] for z in zipped]

        return output

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

    def _reset_file_buffer(self, file_buffer: Union[str, object]) -> Union[str, object]:
        """Resets the file buffer to the beginning for reading."""
        if isinstance(file_buffer, str):
            return file_buffer
        if hasattr(file_buffer, "seekable") and file_buffer.seekable():
            file_buffer.seek(0)
            return file_buffer
        else:
            raise Exception(
                "Cannot seek stream prior to passing to processing.")

    def _validate_file_buffer(self, file_buffer: Union[str, object]) -> pd.DataFrame:
        """Loads and validates CSV data from a file or file-like object."""
        try:
            file_buffer = self._reset_file_buffer(file_buffer=file_buffer)
        except Exception:
            raise ValueError(
                "File buffer must be a valid file path or seekable file-like object.")

        try:
            df = pd.read_csv(file_buffer)
        except pd.errors.EmptyDataError:
            raise ValueError("The provided data file is empty.")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing data file: {e}")
        except Exception as e:
            raise ValueError(
                f"An unexpected error occurred while reading the data file: {e}")

        if df.empty:
            raise ValueError("The data file does not contain any data.")

        required_columns = {"Dissipation",
                            "Resonance_Frequency", "Relative_time"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Data file missing required columns: {', '.join(missing)}.")

        return df

    def predict(self, progress_signal: Any = None, file_buffer: Any = None,
                df: pd.DataFrame = None, visualize: bool = False, use_partial_fills: bool = True):
        """
        Predict all POI locations from input data.

        Args:
            progress_signal: Optional signal for progress updates
            file_buffer: File path or file-like object containing CSV data
            df: Optional pre-loaded DataFrame
            visualize: Whether to show visualization after prediction

        Returns:
            Dictionary with POI predictions
        """

        if progress_signal:
            total_progress_steps = 7
            this_progress_step = 0

        # Validate input
        if file_buffer is not None:
            try:
                df = self._validate_file_buffer(file_buffer=file_buffer)
            except Exception as e:
                Log.e(self.TAG, f"File buffer could not be validated: {e}")
                return self._get_default_predictions()
        elif df is None:
            raise ValueError("Either file_buffer or df must be provided")

        # Store data for visualization
        self._last_data = df.copy()

        Log.d(self.TAG, "Predicting using QModelV4Fusion.")

        final_positions = {}
        confidence_scores = {}

        # Step 1: Predict POI 1
        if progress_signal:
            this_progress_step += 1
            progress_signal.emit(
                int(100 * this_progress_step / total_progress_steps),
                "Step 1/7: Detecting initial fill..."
            )

        try:
            poi2_idx, poi2_conf = self.poi_2_model.predict(df, poi2_idx=-1)
            if poi2_idx != -1:
                final_positions[2] = poi2_idx
                confidence_scores[2] = poi2_conf
            Log.d(
                self.TAG, f"End-of-fill: index={poi2_idx}, confidence={poi2_conf:.3f}")
        except Exception as e:
            Log.e(self.TAG, f"Error detecting end-of-fill: {e}")
        try:
            poi1_idx, poi1_conf = self.poi_1_model.predict(
                df, poi2_idx=poi2_idx)
            if poi1_idx != -1:
                final_positions[1] = poi1_idx
                confidence_scores[1] = poi1_conf
            Log.d(
                self.TAG, f"Initial fill: index={poi1_idx}, confidence={poi1_conf:.3f}")
        except Exception as e:
            Log.e(self.TAG, f"Error predicting initial fill: {e}")

        # Step 2: Predict POI 2
        if progress_signal:
            this_progress_step += 1
            progress_signal.emit(
                int(100 * this_progress_step / total_progress_steps),
                "Step 2/7: End-of-fill..."
            )

        # Step 3-6: Predict POI 4, 5, 6 (classification model)
        if progress_signal:
            this_progress_step += 1
            progress_signal.emit(
                int(100 * this_progress_step / total_progress_steps),
                "Step 3/7: Detecting channels 1, 2, & 3 ..."
            )

        try:
            clf_results = self.poi_345_model.predict(df)

            for poi_num, (idx, conf) in clf_results.items():
                final_positions[poi_num] = idx
                confidence_scores[poi_num] = conf
                Log.d(
                    self.TAG, f"POI{poi_num}: index={idx}, confidence={conf:.3f}")
        except Exception as e:
            Log.e(self.TAG, f"Error pDetecting channels 1, 2, & 3: {e}")

        # Note: POI 3 is not currently predicted by either model
        # This could be added in future iterations

        # Step 7: Format output
        if progress_signal:
            this_progress_step += 1
            progress_signal.emit(
                int(100 * this_progress_step / total_progress_steps),
                "Step 7/7: Formatting results..."
            )
        if use_partial_fills:
            # Use partial fill detectors
            detector = PartialFillDetector()
            pf = detector.process_predictions(
                df=df, clf_results=clf_results, poi2_idx=poi2_idx)

            # Update positions with partial fills (only keep if confidence >= 0.02)
            for k, v in pf.items():
                if k in final_positions and v[1] >= 0.05:
                    final_positions[k] = v[0]
                    confidence_scores[k] = v[1]
                elif k in final_positions:
                    del final_positions[k]
                    del confidence_scores[k]

            # Collapse to lowest available keys
            for target, sources in [(4, [5, 6]), (5, [6])]:
                if target not in final_positions:
                    for source in sources:
                        if source in final_positions:
                            final_positions[target] = final_positions.pop(
                                source)
                            confidence_scores[target] = confidence_scores.pop(
                                source)
                            break

        output = self._format_output(final_positions, confidence_scores)
        self._last_predictions = output

        if progress_signal:
            progress_signal.emit(100, "Auto-fit complete!")

        if visualize:
            self.visualize()

        return output

    def visualize(self, figsize: Tuple[int, int] = (16, 12), save_path: Optional[str] = None):
        """
        Visualize the fusion prediction results.

        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
        """
        if self._last_data is None or self._last_predictions is None:
            raise ValueError(
                "No prediction data available. Run predict() first.")

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)

        # Get dissipation data
        dissipation = self._last_data['Dissipation'].values
        time_indices = np.arange(len(dissipation))

        # Define POI info with colors
        poi_colors = {
            1: 'red', 2: 'blue', 3: 'cyan',
            4: 'orange', 5: 'green', 6: 'purple'
        }

        # Main plot: Dissipation with all POIs (spans both columns at top)
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.plot(time_indices, dissipation, 'k-', alpha=0.5,
                     linewidth=1, label='Dissipation')

        # Plot all detected POIs
        for poi_num in range(1, 7):
            poi_name = f'POI{poi_num}'
            if poi_name in self._last_predictions:
                idx = self._last_predictions[poi_name]['indices'][0]
                conf = self._last_predictions[poi_name]['confidences'][0]

                if idx != -1 and conf != -1:
                    color = poi_colors[poi_num]
                    ax_main.axvline(idx, color=color,
                                    linestyle='--', linewidth=1.5, alpha=0.7)
                    ax_main.plot(idx, dissipation[idx], color=color, marker='*',
                                 markersize=12, markeredgewidth=2, markerfacecolor='none',
                                 label=f'{poi_name} ({conf:.3f})')

        ax_main.set_ylabel('Dissipation', fontsize=12, fontweight='bold')
        ax_main.set_title('V4 Fusion - All POI Detections',
                          fontsize=14, fontweight='bold')
        ax_main.legend(loc='upper right', ncol=3, fontsize=9)
        ax_main.grid(True, alpha=0.3)

        # POI 1 and 2 regression outputs
        for col_idx, (poi_num, model) in enumerate([(1, self.poi_1_model), (2, self.poi_2_model)]):
            if model._last_smoothed is not None:
                ax = fig.add_subplot(gs[1, col_idx])

                ax.plot(model._last_window_positions, model._last_smoothed,
                        color=poi_colors[poi_num], linewidth=2, label=f'POI{poi_num} Confidence')
                ax.axhline(model._peak_threshold, color='red', linestyle=':',
                           linewidth=1, label=f'Threshold={model._peak_threshold}')

                poi_name = f'POI{poi_num}'
                if poi_name in self._last_predictions:
                    idx = self._last_predictions[poi_name]['indices'][0]
                    conf = self._last_predictions[poi_name]['confidences'][0]
                    if idx != -1:
                        ax.axvline(
                            idx, color=poi_colors[poi_num], linestyle='--', linewidth=2, alpha=0.5)
                        ax.plot(idx, conf, color=poi_colors[poi_num], marker='*',
                                markersize=12, markeredgewidth=2, markerfacecolor='none')

                ax.set_ylabel('Confidence', fontsize=10, fontweight='bold')
                ax.set_title(f'POI{poi_num} Regression Output',
                             fontsize=11, fontweight='bold')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1])
                ax.set_xlabel('Sample Index', fontsize=10)

        # POI 4, 5, 6 classification outputs
        if self.poi_345_model._last_predictions is not None:
            poi_info = {
                4: {'pred_idx': 1, 'threshold': 0.7, 'row': 2, 'col': 0},
                5: {'pred_idx': 2, 'threshold': 0.6, 'row': 2, 'col': 1},
                6: {'pred_idx': 3, 'threshold': 0.65, 'row': 3, 'col': 0}
            }

            for poi_num, info in poi_info.items():
                ax = fig.add_subplot(gs[info['row'], info['col']])

                pred_idx = info['pred_idx']
                poi_probs = self.poi_345_model._last_predictions[:, pred_idx]

                ax.plot(self.poi_345_model._last_window_positions, poi_probs,
                        color=poi_colors[poi_num], linewidth=2, label=f'POI{poi_num} Confidence')
                ax.axhline(info['threshold'], color='red', linestyle=':',
                           linewidth=1, label=f"Threshold={info['threshold']}")

                poi_name = f'POI{poi_num}'
                if poi_name in self._last_predictions:
                    idx = self._last_predictions[poi_name]['indices'][0]
                    conf = self._last_predictions[poi_name]['confidences'][0]
                    if idx != -1:
                        ax.axvline(
                            idx, color=poi_colors[poi_num], linestyle='--', linewidth=2, alpha=0.5)
                        ax.plot(idx, conf, color=poi_colors[poi_num], marker='*',
                                markersize=12, markeredgewidth=2, markerfacecolor='none')

                ax.set_ylabel('Confidence', fontsize=10, fontweight='bold')
                ax.set_title(f'POI{poi_num} Classification Output',
                             fontsize=11, fontweight='bold')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1])
                ax.set_xlabel('Sample Index', fontsize=10)

        # Summary statistics in bottom right
        ax_summary = fig.add_subplot(gs[3, 1])
        ax_summary.axis('off')

        summary_text = "Detection Summary\n" + "="*30 + "\n"
        for poi_num in range(1, 7):
            poi_name = f'POI{poi_num}'
            if poi_name in self._last_predictions:
                idx = self._last_predictions[poi_name]['indices'][0]
                conf = self._last_predictions[poi_name]['confidences'][0]

                if idx != -1 and conf != -1:
                    status = "✓ Detected"
                    summary_text += f"\n{poi_name}: {status}\n"
                    summary_text += f"  Index: {idx}\n"
                    summary_text += f"  Confidence: {conf:.3f}\n"
                else:
                    summary_text += f"\n{poi_name}: ✗ Not detected\n"

        ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Overall title
        fig.suptitle('V4 Fusion System - Complete POI Detection Analysis',
                     fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            Log.i(self.TAG, f"Visualization saved to {save_path}")

        plt.show()


# Example usage
if __name__ == "__main__":

    fusion = QModelV4Fusion(
        reg_path_1="poi_model_mini_window_1600_0.pth",
        reg_path_2="poi_model_mini_window_1600_1.pth",
        clf_path="v4_model_pytorch.pth",
        reg_batch_size=2048,
        clf_batch_size=1024
    )

    # Predict on a test file with visualization
    test_file = "content/PROTEIN/01798/M250505W3_IGG05SUC_20C_3rd.csv"
    predictions = fusion.predict(file_buffer=test_file, visualize=True)

    print("\nPredictions:")
    for poi_name, result in predictions.items():
        idx = result['indices'][0]
        conf = result['confidences'][0]
        if idx != -1:
            print(f"{poi_name}: index={idx}, confidence={conf:.3f}")
        else:
            print(f"{poi_name}: not detected")

    # You can also save the visualization
    # predictions = fusion.predict(file_buffer=test_file, visualize=False)
    # fusion.visualize(save_path="poi_detection_results.png")
