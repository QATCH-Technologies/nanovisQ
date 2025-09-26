import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Union, List
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import joblib
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


try:
    from QATCH.common.logger import Logger as Log
    from QATCH.QModel.src.models.static_v4_fusion.v4_fusion_dataprocessor import FusionDataprocessor

except (ImportError, ModuleNotFoundError):
    from QATCH.QModel.src.models.static_v4_fusion.v4_fusion_dataprocessor import FusionDataprocessor

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


class V4RegModel(nn.Module):
    def __init__(self, window_size: int, feature_dim: int, config: Dict[str, any]):
        super(V4RegModel, self).__init__()
        conv_filters = config.get('conv_filters', 64)
        kernel_size = config.get('kernel_size', 5)
        conv_filters_2 = config.get('conv_filters_2', 128)
        lstm_units = config.get('lstm_units', 128)
        lstm_units_2 = config.get('lstm_units_2', 64)
        dense_units = config.get('dense_units', 128)
        dropout_1 = config.get('dropout_1', 0.3)
        dropout_2 = config.get('dropout_2', 0.3)
        dropout_3 = config.get('dropout_3', 0.3)
        self.conv1 = nn.Conv1d(feature_dim, conv_filters,
                               kernel_size=kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(conv_filters, conv_filters_2,
                               kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(conv_filters_2)
        self.pool2 = nn.MaxPool1d(2)
        self.lstm1 = nn.LSTM(conv_filters_2, lstm_units,
                             batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_1)

        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units_2,
                             batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_2)
        self.fc1 = nn.Linear(lstm_units_2 * 2, dense_units)
        self.bn3 = nn.BatchNorm1d(dense_units)
        self.dropout3 = nn.Dropout(dropout_3)
        self.output = nn.Linear(dense_units, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]
        x = torch.relu(self.bn3(self.fc1(x)))
        x = self.dropout3(x)
        x = torch.sigmoid(self.output(x))
        return x.squeeze()


class V4RegPredictor:
    TAG = "[QModelv4.4 (Fusion REG)]"

    def __init__(self,
                 model_path: str,
                 poi_type: str = None,
                 device: str = None):
        # Set device
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        Log.d(self.TAG, f"Using device: {self.device}")
        if poi_type is None:
            # Try to extract from filename (e.g., "poi_model_small_window_0.pth" -> POI1)
            path_stem = Path(model_path).stem
            if path_stem[-1].isdigit():
                poi_idx = int(path_stem[-1])
                poi_map = {0: 'POI1', 1: 'POI2',
                           3: 'POI4', 4: 'POI5', 5: 'POI6'}
                self.poi_type = poi_map.get(poi_idx, f'POI{poi_idx}')
            else:
                self.poi_type = 'POI'
        else:
            self.poi_type = poi_type
        self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        self.window_size = checkpoint['window_size']
        self.stride = checkpoint['stride']
        self.tolerance = checkpoint['tolerance']
        self.feature_dim = checkpoint['feature_dim']
        self.gaussian_sigma = checkpoint.get('gaussian_sigma', 2.0)
        self.peak_threshold = checkpoint.get('peak_threshold', 0.3)

        self.scaler = StandardScaler()
        self.scaler.mean_ = checkpoint['scaler_mean']
        self.scaler.scale_ = checkpoint['scaler_scale']
        self.model = V4RegModel(
            self.window_size, self.feature_dim, self.config
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        Log.d(
            self.TAG, f"Model loaded from {model_path}, POI Type: {self.poi_type},Window size: {self.window_size}, Stride: {self.stride}")

    def predict(self,
                df: pd.DataFrame = None,
                apply_smoothing: bool = True,
                custom_threshold: float = None) -> Dict[str, any]:
        features_df = FusionDataprocessor.get_features(df)
        features = features_df.values
        n_samples = len(features)
        windows = []
        window_positions = []
        for i in range(0, n_samples - self.window_size + 1, self.stride):
            window = features[i:i + self.window_size]
            windows.append(window)
            window_center = i + self.window_size // 2
            window_positions.append(window_center)

        if len(windows) == 0:
            Log.w(self.TAG,
                  f"File too short (need at least {self.window_size} samples)")
            return {
                'poi_position': None,
                'poi_confidence': 0.0,
                'op_values': np.array([]),
                'op_smoothed': np.array([]),
                'window_positions': np.array([]),
                'all_peaks': [],
                'dataframe': df
            }

        windows = np.array(windows)
        window_positions = np.array(window_positions)
        flat = windows.reshape(-1, self.feature_dim)
        flat = self.scaler.transform(flat)
        windows_normalized = flat.reshape(windows.shape)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(windows_normalized).to(self.device)
            op_values = self.model(X_tensor).cpu().numpy()

        if apply_smoothing:
            op_smoothed = gaussian_filter1d(
                op_values, sigma=self.gaussian_sigma)
        else:
            op_smoothed = op_values.copy()
        threshold = custom_threshold if custom_threshold is not None else self.peak_threshold
        peaks, properties = find_peaks(
            op_smoothed,
            height=threshold,
            distance=self.tolerance // self.stride
        )

        if len(peaks) > 0:
            peak_heights = op_smoothed[peaks]
            best_peak_idx = np.argmax(peak_heights)
            poi_position = window_positions[peaks[best_peak_idx]]
            poi_confidence = float(peak_heights[best_peak_idx])
            all_peaks = [(window_positions[p], float(op_smoothed[p]))
                         for p in peaks]
        else:
            poi_position = None
            poi_confidence = 0.0
            all_peaks = []

        if poi_position is not None:
            Log.d(
                self.TAG, f"POI detected at position: {poi_position} with confidence {poi_confidence:.3f}.")
            if len(all_peaks) > 1:
                Log.d(
                    self.TAG, f"Additional peaks found: {len(all_peaks) - 1}")
        else:
            Log.d(self.TAG, "No POI detected")
            max_op = np.max(op_smoothed) if len(op_smoothed) > 0 else 0
            Log.d(
                self.TAG, f"Max OP value: {max_op:.3f} (threshold: {threshold:.3f})")

        return {
            'poi_position': poi_position,
            'poi_confidence': poi_confidence,
            'op_values': op_values,
            'op_smoothed': op_smoothed if apply_smoothing else None,
            'window_positions': window_positions,
            'all_peaks': all_peaks,
            'dataframe': df
        }

    def visualize(self,
                  prediction_result: Dict,
                  save_path: str = None,
                  show_plot: bool = True,
                  figsize: Tuple[int, int] = (15, 10)) -> None:
        df = prediction_result['dataframe']
        poi_position = prediction_result['poi_position']
        poi_confidence = prediction_result['poi_confidence']
        op_values = prediction_result['op_values']
        op_smoothed = prediction_result['op_smoothed']
        window_positions = prediction_result['window_positions']
        all_peaks = prediction_result['all_peaks']
        if 'Dissipation' in df.columns:
            dissipation = df['Dissipation'].values
        elif 'dissipation' in df.columns:
            dissipation = df['dissipation'].values
        else:
            dissipation = df.iloc[:, 0].values
        n_plots = 3 if op_smoothed is not None else 2
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize,
                                 height_ratios=[3] + [1] * (n_plots - 1))

        # Plot 1: Signal with POI marker
        axes[0].plot(dissipation, 'b-', linewidth=1,
                     alpha=0.7, label='Dissipation')
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Dissipation')
        axes[0].set_title(
            f'{self.poi_type} Detection Results - Regression Method')
        axes[0].grid(True, alpha=0.3)
        if poi_position is not None:
            axes[0].axvline(x=poi_position, color='red', linestyle='--',
                            linewidth=2, alpha=0.8, label=f'{self.poi_type} Detected')
            axes[0].text(poi_position, axes[0].get_ylim()[1] * 0.95,
                         f'{self.poi_type}\n{poi_confidence:.2f}',
                         rotation=0, ha='center', va='top',
                         bbox=dict(boxstyle='round',
                                   facecolor='white', alpha=0.8),
                         fontsize=10, color='red')
            shade_start = max(0, poi_position - self.tolerance)
            shade_end = min(len(dissipation), poi_position + self.tolerance)
            axes[0].axvspan(shade_start, shade_end, color='red', alpha=0.1)
        for i, (pos, conf) in enumerate(all_peaks[1:] if poi_position else all_peaks):
            axes[0].axvline(x=pos, color='orange', linestyle=':',
                            linewidth=1, alpha=0.5)
            axes[0].text(pos, axes[0].get_ylim()[0] +
                         (axes[0].get_ylim()[1] -
                          axes[0].get_ylim()[0]) * 0.05,
                         f'{conf:.2f}', rotation=90, va='bottom',
                         fontsize=8, color='orange')

        axes[0].legend(loc='upper right')

        # Plot 2: Raw OP values
        axes[1].plot(window_positions, op_values, 'g-', linewidth=1.5,
                     alpha=0.7, label='Raw OP values')
        axes[1].set_xlabel('Window Position')
        axes[1].set_ylabel('OP Value')
        axes[1].set_title('Overlap Parameter (OP) Predictions - Raw')
        axes[1].set_ylim([0, 1.05])
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=self.peak_threshold, color='red', linestyle='--',
                        alpha=0.5, label=f'Threshold ({self.peak_threshold:.2f})')
        axes[1].legend(loc='upper right')

        # Plot 3: Smoothed OP values with peaks (if smoothing applied)
        if op_smoothed is not None and n_plots > 2:
            axes[2].plot(window_positions, op_smoothed, 'b-', linewidth=1.5,
                         label=f'Smoothed OP (σ={self.gaussian_sigma})')
            for pos, conf in all_peaks:
                idx = np.argmin(np.abs(window_positions - pos))
                axes[2].scatter(pos, op_smoothed[idx], color='red' if pos == poi_position else 'orange',
                                s=50, zorder=5)

            axes[2].axhline(y=self.peak_threshold, color='red', linestyle='--',
                            alpha=0.5, label=f'Threshold ({self.peak_threshold:.2f})')
            axes[2].set_xlabel('Window Position')
            axes[2].set_ylabel('OP Value')
            axes[2].set_title('Smoothed OP Predictions with Peak Detection')
            axes[2].set_ylim([0, 1.05])
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, FusionDataprocessori=150,
                        bbox_inches='tight')
        if show_plot:
            plt.show()


class V4ClfModel(nn.Module):
    """PyTorch model for POI detection - inference version."""

    def __init__(self,
                 window_size: int,
                 feature_dim: int,
                 conv_filters_1: int = 64,
                 conv_filters_2: int = 128,
                 kernel_size: int = 3,
                 lstm_units_1: int = 128,
                 lstm_units_2: int = 64,
                 dense_units: int = 128,
                 dropout_rate: float = 0.3):
        super(V4ClfModel, self).__init__()

        self.window_size = window_size
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv1d(feature_dim, conv_filters_1,
                               kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(conv_filters_1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(
            conv_filters_1, conv_filters_2, 3, padding='same')
        self.bn2 = nn.BatchNorm1d(conv_filters_2)
        self.pool2 = nn.MaxPool1d(2)
        self.lstm1 = nn.LSTM(conv_filters_2, lstm_units_1,
                             batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.lstm2 = nn.LSTM(lstm_units_1 * 2, lstm_units_2,
                             batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(lstm_units_2 * 2, dense_units)
        self.bn3 = nn.BatchNorm1d(dense_units)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(dense_units, 6)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]
        x = F.relu(self.bn3(self.dense(x)))
        x = self.dropout3(x)
        x = torch.sigmoid(self.output(x))
        return x


class V4ClfPredictor:
    TAG = "[QModelv4.4 (Fusion CLF)]"

    def __init__(self,
                 model_path: str = 'v4_model_pytorch.pth',
                 scaler_path: str = 'v4_scaler_pytorch.joblib',
                 config_path: str = 'v4_config_pytorch.json',
                 device: str = None):

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        Log.d(self.TAG, f"Using device: {self.device}")
        self.load_model(model_path, scaler_path, config_path)

    def load_model(self, model_path: str, scaler_path: str, config_path: str) -> None:
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.window_size = config['window_size']
        self.stride = config['stride']
        self.tolerance = config['tolerance']
        self.feature_dim = config['feature_dim']
        self.best_params = config.get('best_params')
        self.scaler = joblib.load(scaler_path)
        checkpoint = torch.load(model_path, map_location=self.device)
        if self.best_params:
            model_params = {k: v for k, v in self.best_params.items()
                            if k not in ['learning_rate', 'batch_size']}
            self.model = self._create_model(**model_params)
        else:
            self.model = self._create_model()

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        Log.d(self.TAG, f"Model loaded from {model_path}")
        Log.d(self.TAG, f"Scaler loaded from {scaler_path}")
        Log.d(self.TAG, f"Configuration loaded from {config_path}")

    def _create_model(self, **params) -> V4ClfModel:
        return V4ClfModel(
            window_size=self.window_size,
            feature_dim=self.feature_dim,
            **params
        ).to(self.device)

    def predict(self,
                df: pd.DataFrame = None,
                threshold: float = 0.5,
                adaptive_thresholds: Dict[int, float] = None,
                enforce_constraints: bool = True) -> Dict[str, any]:
        features_df = FusionDataprocessor.gen_features(df)
        features = features_df.values
        windows = []
        window_positions = []
        n = len(features)
        w = self.window_size
        s = self.stride
        for i in range(0, n, s):
            end = i + w
            if end <= n:
                window = features[i:end]
            else:
                pad_len = end - n
                last_val = features[-1]
                pad = np.repeat(last_val[np.newaxis, :], pad_len, axis=0)
                window = np.vstack([features[i:n], pad])
            windows.append(window)
            center = min(i + w // 2, n - 1)
            window_positions.append(center)

        if len(windows) == 0:
            Log.w(self.TAG, "Not enough data for prediction")
            return {
                'poi_locations': {},
                'poi_count': 0,
                'poi_binary': np.zeros(5),
                'probabilities': None,
                'dataframe': df
            }

        windows = np.array(windows)
        windows_reshaped = windows.reshape(-1, windows.shape[-1])
        windows_reshaped = self.scaler.transform(windows_reshaped)
        windows = windows_reshaped.reshape(windows.shape)
        with torch.no_grad():
            windows_tensor = torch.FloatTensor(windows).to(self.device)
            predictions = self.model(windows_tensor).cpu().numpy()

        # Set adaptive thresholds
        if adaptive_thresholds is None:
            adaptive_thresholds = {
                1: 0.5,   # POI-1
                2: 0.5,   # POI-2
                4: 0.01,   # POI-4
                5: 0.1,   # POI-5
                6: 0.1   # POI-6
            }

        # Find POI candidates
        poi_locations = self._find_poi_candidates(
            predictions, window_positions, adaptive_thresholds)

        # Apply constraints if requested
        if enforce_constraints:
            poi_locations = self._enforce_sequential_constraints(poi_locations)
            if 'Relative_time' in df.columns:
                poi_locations = self._enforce_relative_gap_constraints(
                    poi_locations, df)

        # Create binary output
        poi_binary = np.zeros(5)
        for poi_num in [1, 2, 4, 5, 6]:
            if poi_num in poi_locations:
                idx = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4}[poi_num]
                poi_binary[idx] = 1

        # Calculate POI count
        poi_count = len(poi_locations)
        Log.d(self.TAG, f"POIs detected: {poi_count}")
        Log.d(self.TAG, f"POI locations: {poi_locations}")
        Log.d(self.TAG,
              f"Binary output [POI1, POI2, POI4, POI5, POI6]: {poi_binary.astype(int)}")

        return {
            'poi_locations': poi_locations,
            'poi_count': poi_count,
            'poi_binary': poi_binary,
            'probabilities': predictions,
            'window_positions': window_positions,
            'dataframe': df
        }

    def _find_poi_candidates(self, predictions: np.ndarray,
                             window_positions: List[int],
                             thresholds: Dict[int, float]) -> Dict[int, int]:
        poi_locations = {}
        poi_indices = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5}

        for poi_num, pred_idx in poi_indices.items():
            poi_probs = predictions[:, pred_idx]
            poi_threshold = thresholds.get(poi_num, 0.5)

            above_threshold = poi_probs > poi_threshold
            if np.any(above_threshold):
                # Find peak
                peak_indices = self._find_peaks(poi_probs, above_threshold)

                if len(peak_indices) > 0:
                    best_idx = peak_indices[np.argmax(poi_probs[peak_indices])]
                    poi_locations[poi_num] = window_positions[best_idx]

        return poi_locations

    def _find_peaks(self, probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        peaks = []
        for i in range(1, len(probs) - 1):
            if mask[i] and probs[i] >= probs[i-1] and probs[i] >= probs[i+1]:
                peaks.append(i)

        if len(mask) > 0:
            if mask[0] and (len(probs) == 1 or probs[0] >= probs[1]):
                peaks.insert(0, 0)
            if mask[-1] and probs[-1] >= probs[-2]:
                peaks.append(len(probs) - 1)

        return np.array(peaks) if peaks else np.array([])

    def _enforce_sequential_constraints(self, poi_candidates: Dict[int, int]) -> Dict[int, int]:
        validated_pois = {}

        # POI1 and POI2 can exist independently
        if 1 in poi_candidates:
            validated_pois[1] = poi_candidates[1]
        if 2 in poi_candidates:
            validated_pois[2] = poi_candidates[2]

        # POI4 requires both POI1 and POI2
        if 4 in poi_candidates:
            if 1 in validated_pois and 2 in validated_pois:
                if poi_candidates[4] > validated_pois[2]:
                    validated_pois[4] = poi_candidates[4]

        # POI5 requires POI4
        if 5 in poi_candidates:
            if 4 in validated_pois:
                if poi_candidates[5] > validated_pois[4]:
                    validated_pois[5] = poi_candidates[5]

        # POI6 requires POI5
        if 6 in poi_candidates:
            if 5 in validated_pois:
                if poi_candidates[6] > validated_pois[5]:
                    validated_pois[6] = poi_candidates[6]

        return validated_pois

    def _enforce_relative_gap_constraints(self, poi_candidates: Dict[int, int],
                                          df: pd.DataFrame) -> Dict[int, int]:
        """Apply relative time gap constraints."""
        refined_pois = poi_candidates.copy()

        if 'Relative_time' not in df.columns:
            return refined_pois

        def get_time(idx):
            if idx < len(df):
                return df['Relative_time'].iloc[idx]
            return None

        # Check gap constraints
        if 1 in refined_pois and 2 in refined_pois:
            time1 = get_time(refined_pois[1])
            time2 = get_time(refined_pois[2])

            if time1 is not None and time2 is not None:
                gap_1_2 = abs(time2 - time1)

                # Check POI4
                if 4 in refined_pois:
                    time4 = get_time(refined_pois[4])
                    if time4 is not None:
                        gap_2_4 = abs(time4 - time2)
                        if gap_2_4 < gap_1_2:
                            del refined_pois[4]

        # Remove dependent POIs if parent removed
        if 5 in refined_pois and 4 not in refined_pois:
            del refined_pois[5]
        if 6 in refined_pois and 5 not in refined_pois:
            del refined_pois[6]

        return refined_pois

    def visualize(self,
                  prediction_result: Dict,
                  save_path: str = None,
                  show_plot: bool = True) -> None:
        df = prediction_result['dataframe']
        poi_locations = prediction_result['poi_locations']
        probabilities = prediction_result['probabilities']
        window_positions = prediction_result['window_positions']
        use_time = 'Relative_time' in df.columns
        x_values = df['Relative_time'].values if use_time else np.arange(
            len(df))
        x_label = 'Relative Time' if use_time else 'Sample Index'
        window_x_values = []
        for pos in window_positions:
            if use_time and pos < len(df):
                window_x_values.append(df['Relative_time'].iloc[pos])
            else:
                window_x_values.append(pos)
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        poi_colors = {1: 'red', 2: 'orange',
                      4: 'yellow', 5: 'green', 6: 'blue'}

        # Plot 1: Dissipation
        if 'Dissipation' in df.columns:
            axes[0].plot(x_values, df['Dissipation'].values,
                         'b-', alpha=0.7, lw=0.5)
            axes[0].set_ylabel('Dissipation')
            axes[0].grid(True, alpha=0.3)

        # Plot 2: Resonance Frequency
        if 'Resonance_Frequency' in df.columns:
            axes[1].plot(x_values, df['Resonance_Frequency'].values,
                         'g-', alpha=0.7, lw=0.5)
            axes[1].set_ylabel('Resonance Frequency')
            axes[1].grid(True, alpha=0.3)

        # Plot 3: POI Probabilities
        if probabilities is not None:
            poi_indices = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5}
            for poi_num, idx in poi_indices.items():
                axes[2].plot(window_x_values, probabilities[:, idx],
                             color=poi_colors[poi_num], alpha=0.6,
                             label=f'POI-{poi_num}', lw=1)

            axes[2].set_ylabel('POI Probabilities')
            axes[2].legend(loc='upper right')
            axes[2].grid(True, alpha=0.3)
            axes[2].axhline(0.5, color='black', linestyle=':', alpha=0.5)

            # Any POI confidence
            any_conf = 1 - probabilities[:, 0]
            axes[3].plot(window_x_values, any_conf, 'r-', alpha=0.7, lw=1)
            axes[3].set_ylabel('Any POI Confidence')
            axes[3].grid(True, alpha=0.3)
        for poi_num, idx in poi_locations.items():
            if use_time and idx < len(df):
                x_pos = df['Relative_time'].iloc[idx]
            else:
                x_pos = idx

            for ax in axes:
                ax.axvline(x_pos, color=poi_colors.get(poi_num, 'black'),
                           linestyle='--', alpha=0.8, label=f'POI-{poi_num}' if ax == axes[0] else '')
        half_win = 64
        for poi_num, idx in poi_locations.items():
            start_idx = max(0, idx - half_win)
            end_idx = min(len(df) - 1, idx + half_win)

            if use_time:
                x_start = df['Relative_time'].iloc[start_idx]
                x_end = df['Relative_time'].iloc[end_idx]
            else:
                x_start, x_end = start_idx, end_idx

            for ax in axes:
                ax.axvspan(x_start, x_end, color=poi_colors.get(poi_num, 'gray'),
                           alpha=0.1)
        axes[0].set_title(
            f'POI Detection Results - {len(poi_locations)} POIs Detected')
        axes[-1].set_xlabel(x_label)

        if poi_locations:
            axes[0].legend(loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        if show_plot:
            plt.show()
        for poi_num in sorted(poi_locations.keys()):
            idx = poi_locations[poi_num]
            if use_time and idx < len(df):
                time_val = df['Relative_time'].iloc[idx]


class QModelV4Fusion:
    TAG = "[QModelv4.4 (Fusion MAIN)]"

    def __init__(self,
                 classification_model_path: str,
                 classification_scaler_path: str,
                 classification_config_path: str,
                 regression_model_paths: Dict[str, str],
                 device: str = None):
        Log.d(self.TAG, "Initializing QModelV4Fusion.")

        Log.d(self.TAG, "Loading classification head.")
        self.clf_predictor = V4ClfPredictor(
            model_path=classification_model_path,
            scaler_path=classification_scaler_path,
            config_path=classification_config_path,
            device=device
        )

        Log.d(self.TAG, "Loading regression heads.")
        self.reg_predictors = {}
        for poi_name, model_path in regression_model_paths.items():
            Log.d(self.TAG, f"Loading {poi_name} regression head.")
            self.reg_predictors[poi_name] = V4RegPredictor(
                model_path=model_path,
                poi_type=poi_name,
                device=device
            )

        Log.d(self.TAG, "QModelV4Fusion predictor initialized successfully!")

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

    def _format_output(
        self,
        final_positions: Dict[int, int],
        confidence_scores: Dict[int, float]
    ) -> Dict[str, Dict[str, List[float]]]:
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
        for poi_name in output:
            zipped = list(
                zip(output[poi_name]['indices'],
                    output[poi_name]['confidences'])
            )
            zipped.sort(key=lambda x: x[1], reverse=True)
            output[poi_name]['indices'] = [z[0] for z in zipped]
            output[poi_name]['confidences'] = [z[1] for z in zipped]

        return output

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

    def predict(self,
                file_buffer: str = None,
                window_margin: int = 100,
                use_regression_threshold: float = 0.5,
                enforce_constraints: bool = True,
                format_output: bool = False) -> Dict[str, any]:

        if file_buffer is not None:
            try:
                df = self._validate_file_buffer(file_buffer=file_buffer)
            except Exception as e:
                Log.d(f"File buffer could not be validated: {e}")
                return self._get_default_predictions()
        elif df is None:
            raise ValueError("Either file_buffer or df must be provided")

        Log.d(self.TAG, "Predicting using QModelV4Fusion.")
        Log.d(self.TAG, "Classification head prediction active.")
        clf_results = self.clf_predictor.predict(
            df=df,
            enforce_constraints=enforce_constraints
        )

        clf_positions = clf_results['poi_locations']
        clf_probabilities = clf_results['probabilities']
        window_positions = clf_results['window_positions']

        Log.d(
            self.TAG, f"Classification found {len(clf_positions)} POIs: {list(clf_positions.keys())}")

        Log.d(self.TAG, "Regression head predictions active.")
        final_positions = {}
        methods_used = {}
        regression_refinements = {}
        confidence_scores = {}
        all_regression_results = {}
        poi_map = {1: 'POI1', 2: 'POI2', 4: 'POI4', 5: 'POI5', 6: 'POI6'}
        for poi_name, predictor in self.reg_predictors.items():
            Log.d(self.TAG, f"Executing {poi_name} regression head.")
            reg_result = predictor.predict(
                df=df,
                apply_smoothing=True
            )
            all_regression_results[poi_name] = reg_result

        Log.d(self.TAG, "Regression peak search active.")
        final_positions = {}
        methods_used = {}
        confidence_scores = {}

        for poi_num, clf_position in clf_positions.items():
            poi_name = poi_map.get(poi_num)

            if poi_name and poi_name in all_regression_results:
                reg_result = all_regression_results[poi_name]
                poi1_final = final_positions.get(
                    1, None) if poi_num == 2 else None

                best_peak = None
                best_distance = float('inf')

                for peak_pos, peak_conf in reg_result.get('all_peaks', []):
                    distance = abs(peak_pos - clf_position)
                    if distance <= window_margin and peak_conf >= use_regression_threshold:
                        if poi_num == 2 and poi1_final is not None and peak_pos <= poi1_final:
                            continue
                        if distance < best_distance:
                            best_peak = (peak_pos, peak_conf)
                            best_distance = distance
                if best_peak:
                    final_positions[poi_num] = best_peak[0]
                    methods_used[poi_num] = 'regression'
                    confidence_scores[poi_num] = best_peak[1]
                else:
                    # fallback chain for POI2
                    if poi_num == 2 and poi1_final is not None:
                        # 1) classification after POI1?
                        if clf_position > poi1_final:
                            final_positions[poi_num] = clf_position
                            methods_used[poi_num] = 'classification'
                        else:
                            # 2) right bound of POI1 classification window
                            right_bound = final_positions[1] + 64
                            final_positions[poi_num] = right_bound
                            methods_used[poi_num] = 'poi1_right_bound'
                    else:
                        # default classification fallback
                        final_positions[poi_num] = clf_position
                        methods_used[poi_num] = 'classification'
                poi_indices = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5}
                if clf_probabilities is not None and poi_num in poi_indices:
                    closest_window_idx = np.argmin(
                        np.abs(np.array(window_positions) - clf_position))
                    confidence_scores[poi_num] = float(
                        clf_probabilities[closest_window_idx, poi_indices[poi_num]])
                else:
                    confidence_scores[poi_num] = 0.5

        # binary output
        poi_binary = np.zeros(5)
        for poi_num in [1, 2, 4, 5, 6]:
            if poi_num in final_positions:
                idx = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4}[poi_num]
                poi_binary[idx] = 1
        # Print summary
        Log.d(self.TAG, "QModelV4Fusion Preidction Successful")
        Log.d(self.TAG, f"Total POIs detected: {len(final_positions)}")
        Log.d(
            self.TAG, f"Binarized output[POI1, POI2, POI4, POI5, POI6]: {poi_binary.astype(int)}")
        if format_output:
            return self._format_output(final_positions=final_positions, confidence_scores=confidence_scores)
        return {
            'final_positions': final_positions,
            'methods_used': methods_used,
            'classification_results': clf_results,
            'regression_refinements': regression_refinements,
            'confidence_scores': confidence_scores,
            'poi_binary': poi_binary,
            'poi_count': len(final_positions),
            'dataframe': df
        }

    def visualize(self,
                  prediction_result: Dict,
                  save_path: str = None,
                  show_plot: bool = True,
                  figsize: Tuple[int, int] = (16, 14)) -> None:
        df = prediction_result['dataframe']
        final_positions = prediction_result['final_positions']
        methods_used = prediction_result['methods_used']
        clf_results = prediction_result['classification_results']
        confidence_scores = prediction_result['confidence_scores']
        use_time = 'Relative_time' in df.columns
        x_values = df['Relative_time'].values if use_time else np.arange(
            len(df))
        x_label = 'Relative Time' if use_time else 'Sample Index'
        fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
        poi_colors = {1: 'red', 2: 'orange', 4: 'gold', 5: 'green', 6: 'blue'}

        # Plot 1: Dissipation with final POI positions
        if 'Dissipation' in df.columns:
            axes[0].plot(x_values, df['Dissipation'].values,
                         'b-', alpha=0.7, lw=0.5)
        elif 'dissipation' in df.columns:
            axes[0].plot(x_values, df['dissipation'].values,
                         'b-', alpha=0.7, lw=0.5)
        axes[0].set_ylabel('Dissipation')
        axes[0].set_title('Hybrid POI Detection - Final Results')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Resonance Frequency
        if 'Resonance_Frequency' in df.columns:
            axes[1].plot(x_values, df['Resonance_Frequency'].values,
                         'g-', alpha=0.7, lw=0.5)
            axes[1].set_ylabel('Resonance Frequency')
            axes[1].grid(True, alpha=0.3)

        # Plot 3: Classification probabilities
        if clf_results['probabilities'] is not None:
            window_positions = clf_results['window_positions']
            probabilities = clf_results['probabilities']

            # Convert window positions to x-axis values
            window_x_values = []
            for pos in window_positions:
                if use_time and pos < len(df):
                    window_x_values.append(df['Relative_time'].iloc[pos])
                else:
                    window_x_values.append(pos)

            poi_indices = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5}
            for poi_num, idx in poi_indices.items():
                axes[2].plot(window_x_values, probabilities[:, idx],
                             color=poi_colors[poi_num], alpha=0.6,
                             label=f'POI-{poi_num}', lw=1)

            axes[2].set_ylabel('Classification Prob.')
            axes[2].legend(loc='upper right', ncol=5, fontsize=8)
            axes[2].grid(True, alpha=0.3)
            axes[2].axhline(0.5, color='black', linestyle=':', alpha=0.5)

        # Plot 4: Method comparison
        axes[3].set_ylabel('Detection Method')
        axes[3].set_ylim([-0.5, 1.5])
        axes[3].set_yticks([0, 1])
        axes[3].set_yticklabels(['Classification', 'Regression'])
        axes[3].grid(True, alpha=0.3)

        # Plot 5: Confidence scores
        axes[4].set_ylabel('Confidence')
        axes[4].set_ylim([0, 1.1])
        axes[4].grid(True, alpha=0.3)
        axes[4].set_xlabel(x_label)
        for poi_num, position in final_positions.items():
            method = methods_used[poi_num]
            confidence = confidence_scores.get(poi_num, 0)
            color = poi_colors.get(poi_num, 'black')
            if use_time and position < len(df):
                x_pos = df['Relative_time'].iloc[position]
            else:
                x_pos = position
            for ax in axes[:3]:
                ax.axvline(x_pos, color=color, linestyle='--' if method == 'classification' else '-',
                           alpha=0.6, linewidth=1.5)
            half_win = 64
            start_idx = max(0, position - half_win)
            end_idx = min(len(df) - 1, position + half_win)

            if use_time:
                x_start = df['Relative_time'].iloc[start_idx]
                x_end = df['Relative_time'].iloc[end_idx]
            else:
                x_start, x_end = start_idx, end_idx
            axes[0].axvspan(x_start, x_end, color=color, alpha=0.1)
            axes[0].text(x_pos, axes[0].get_ylim()[1] * 0.95,
                         f'POI{poi_num}\n{method[0].upper()}',
                         ha='center', va='top', fontsize=8,
                         bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
            y_val = 1 if method == 'regression' else 0
            axes[3].scatter(x_pos, y_val, color=color, s=100, zorder=5)
            axes[3].text(x_pos, y_val + 0.15, f'POI{poi_num}',
                         ha='center', fontsize=8, color=color)
            axes[4].bar(x_pos, confidence, width=20 if use_time else 50,
                        color=color, alpha=0.6, label=f'POI{poi_num}')
            axes[4].text(x_pos, confidence + 0.02, f'{confidence:.2f}',
                         ha='center', fontsize=8)

        # Add legends
        axes[0].text(0.02, 0.98, 'Solid line = Regression refined\nDashed line = Classification only',
                     transform=axes[0].transAxes, fontsize=8, va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show_plot:
            plt.show()
        clf_positions = clf_results['poi_locations']
        for poi_num in sorted(final_positions.keys()):
            if poi_num in clf_positions:
                clf_pos = clf_positions[poi_num]
                final_pos = final_positions[poi_num]
                diff = final_pos - clf_pos
                method = methods_used[poi_num]


# Example usage
if __name__ == "__main__":
    # Define paths to models
    classification_paths = {
        'model': 'QModel/SavedModels/qmodel_v4_fusion/v4_model_pytorch.pth',
        'scaler': 'QModel/SavedModels/qmodel_v4_fusion/v4_scaler_pytorch.joblib',
        'config': 'QModel/SavedModels/qmodel_v4_fusion/v4_config_pytorch.json'
    }

    regression_paths = {
        'POI1': 'QModel/SavedModels/qmodel_v4_fusion/poi_model_mini_window_0.pth',
        'POI2': 'QModel/SavedModels/qmodel_v4_fusion/poi_model_small_window_1.pth',
        'POI4': 'QModel/SavedModels/qmodel_v4_fusion/poi_model_med_window_3.pth',
        'POI5': 'QModel/SavedModels/qmodel_v4_fusion/poi_model_large_window_4.pth',
        'POI6': 'QModel/SavedModels/qmodel_v4_fusion/poi_model_small_window_5.pth'
    }

    # Initialize hybrid predictor
    predictor = QModelV4Fusion(
        classification_model_path=classification_paths['model'],
        classification_scaler_path=classification_paths['scaler'],
        classification_config_path=classification_paths['config'],
        regression_model_paths=regression_paths
    )

    content = FusionDataprocessor.load_content('content/PROTEIN')
    for data, poi in content:
        result = predictor.predict(
            file_buffer=data,
            window_margin=64,
            use_regression_threshold=0.25,
            enforce_constraints=False
        )

        # Visualize results
        predictor.visualize(result)

        # Access results
        print(f"\nFinal POI positions: {result['final_positions']}")
        print(f"Methods used: {result['methods_used']}")
        print(f"Confidence scores: {result['confidence_scores']}")
