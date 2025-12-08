import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import warnings
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import median_filter, gaussian_filter1d
from sklearn.preprocessing import RobustScaler
from typing import Dict, Tuple, Union, List, Any, Optional
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

# ==========================================
# LOGGING UTILS (Matching v4_fusion)
# ==========================================


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

# ==========================================
# MAIN QMODEL CLASS
# ==========================================

class FastFeatures:
    @staticmethod
    def generate(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)

        # 1. Aggressive Pre-Smoothing (The Fix for Barcode Noise)
        # Sigma=5 removes high-freq electrical noise before we calculate slopes
        for col in ["Dissipation", "Resonance_Frequency"]:
            if col in df.columns:
                df[col] = gaussian_filter1d(df[col].values, sigma=5.0)

        # 2. Compute "Difference" (Vectorized)
        if hasattr(FastFeatures, '_compute_difference_vectorized'):
            df["Difference"] = FastFeatures._compute_difference_vectorized(df)

        # 3. Compute DoG (Difference of Gaussians) features
        diss_vals = df['Dissipation'].values
        diff_vals = df['Difference'].values if "Difference" in df else np.zeros_like(
            diss_vals)

        # DoG is just Gaussian(sigma2) - Gaussian(sigma1), but since we already smoothed,
        # we can just apply another filter to get the lower frequency component
        df['Dissipation_DoG'] = gaussian_filter1d(diss_vals, sigma=2, order=1)
        df['Difference_DoG'] = gaussian_filter1d(diff_vals, sigma=2, order=1)

        # 4. Anomaly Score
        df['Dissipation_DoG_Anomaly_Score'] = FastFeatures._fast_anomaly_score(
            df['Dissipation_DoG'].values
        )

        # 5. Derivatives (Slopes)
        time_vals = df['Relative_time'].values
        dt = np.diff(time_vals, prepend=time_vals[0])
        # Prevent division by zero or tiny dt
        dt[dt <= 0] = np.nanmedian(dt) if len(dt) > 0 else 1.0

        # Calculate Delta using the SMOOTHED signals
        diss_diff = np.diff(diss_vals, prepend=diss_vals[0])
        rf_diff = np.diff(df['Resonance_Frequency'].values,
                          prepend=df['Resonance_Frequency'].values[0])

        # Normalized slopes
        diss_slope = diss_diff / dt
        rf_slope = rf_diff / dt

        # 6. Slope Ratio (Clipped for stability)
        # Avoid division by very small numbers
        rf_slope_safe = np.where(
            np.abs(rf_slope) < 1e-5, np.sign(rf_slope) * 1e-5, rf_slope)
        rf_slope_safe = np.where(rf_slope_safe == 0, 1e-5, rf_slope_safe)
        df['slope_ratio'] = np.clip(diss_slope / rf_slope_safe, -100, 100)

        # 7. Rolling Standard Deviation
        df['Diss_roll_std'] = df['Dissipation'].rolling(
            window=50, min_periods=1).std().fillna(0)

        # 8. Save Deltas
        df['Dissipation_Delta'] = diss_slope
        df['Resonance_Delta'] = rf_slope

        # 9. Clean up Time and Types
        df['Relative_time'] = df['Relative_time'].astype(np.float32)

        df.fillna(0, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)

        return df

    @staticmethod
    def _compute_difference_vectorized(df, difference_factor=2):
        xs = df["Relative_time"].values
        res_freq = df["Resonance_Frequency"].values
        diss = df["Dissipation"].values

        # Simple mask for baseline (assuming early part of file is baseline)
        mask = (xs > xs.min()) & (xs <= (xs.min() + (xs.max()-xs.min())*0.1))
        if mask.sum() < 2:
            avg_res = np.mean(res_freq[:10])
            avg_diss = np.mean(diss[:10])
        else:
            avg_res = np.nanmean(res_freq[mask])
            avg_diss = np.nanmean(diss[mask])

        if np.isnan(avg_res) or np.isnan(avg_diss):
            return np.zeros(len(df))

        ys_diss = (diss - avg_diss) * avg_res / 2
        ys_freq = avg_res - res_freq
        return ys_freq - difference_factor * ys_diss

    @staticmethod
    def _fast_anomaly_score(signal, window=100):
        n = len(signal)
        if n < window:
            window = max(10, n // 2)

        cumsum = np.cumsum(np.insert(signal, 0, 0))
        cumsum_sq = np.cumsum(np.insert(signal**2, 0, 0))
        pad = window // 2
        scores = np.zeros(n, dtype=np.float32)

        for i in range(n):
            left = max(0, i - pad)
            right = min(n, i + pad + 1)
            local_sum = cumsum[right] - cumsum[left]
            local_sum_sq = cumsum_sq[right] - cumsum_sq[left]
            count = right - left

            if count > 0:
                local_mean = local_sum / count
                local_var = (local_sum_sq / count) - (local_mean ** 2)
                local_std = np.sqrt(max(local_var, 1e-10))
                scores[i] = abs(signal[i] - local_mean) / local_std

        scores = scores - scores.min()
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score
        return scores


# ==========================================
# 3. Model Architecture
# ==========================================

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SqueezeExcitation(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet1D(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv1d(
            input_channels, 32, kernel_size=11, stride=2, padding=5, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Dilated convolutions included
        self.layer1 = self._make_layer(32, 2, stride=1, dilation=1)
        self.layer2 = self._make_layer(64, 2, stride=2, dilation=2)
        self.layer3 = self._make_layer(128, 2, stride=2, dilation=4)
        self.layer4 = self._make_layer(256, 2, stride=2, dilation=8)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.rnn = nn.LSTM(input_size=256, hidden_size=128,
                           num_layers=2, batch_first=True, bidirectional=True)

        # Standard Sigmoid output
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 2, num_classes),
            nn.Sigmoid()
        )

        # Initialize weights for stability
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def _make_layer(self, out_channels, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels,
                          1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = [ResidualBlock(
            self.in_channels, out_channels, stride=stride, downsample=downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(
                out_channels, out_channels, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = x.permute(0, 2, 1)

        # LSTM is unstable in float16, force float32
        x_float = x.float()
        x_float, _ = self.rnn(x_float)
        x = x_float[:, -1, :]

        return self.fc(x)

class QModelV5Resnet:
    """
    QModel wrapper for ResNet1D inference system.
    Implements sliding window voting, resampling, and peak/slope detection.
    """
    TAG = "ResNetSystem"

    # Default Configuration from infer_2.py
    CONFIG = {
        'window_size': 1024,
        'hidden_dim': 128,
        'poi_config': [
            {'file_idx': 0, 'label': 'POI1'},
            {'file_idx': 1, 'label': 'POI2'},
            {'file_idx': 3, 'label': 'POI3'},
            {'file_idx': 4, 'label': 'POI4'},
            {'file_idx': 5, 'label': 'POI5'},
        ],
        'feature_cols': [
            'Relative_time', 'Resonance_Frequency', 'Dissipation',
            'Dissipation_Delta', 'Resonance_Delta', 'Difference',
            'Dissipation_DoG', 'Difference_DoG', 'slope_ratio',
            'Dissipation_DoG_Anomaly_Score', 'Diss_roll_std'
        ],
        'inference_stride': 16,
    }

    def __init__(self, model_path: str, scaler_path: str = None, device_str: str = None):
        """
        Args:
            model_path: Path to the .pth model file.
            scaler_path: Path to the .pkl scaler file.
            device_str: 'cuda' or 'cpu'. Auto-detects if None.
        """
        self.device = torch.device(
            device_str if device_str else (
                "cuda" if torch.cuda.is_available() else "cpu")
        )

        self.num_classes = len(self.CONFIG['poi_config'])
        self._model = None
        self._scaler = None

        # State for visualization
        self._last_original_df = None
        self._last_resampled_df = None
        self._last_probs = None
        self._last_predictions = None

        self._load(model_path, scaler_path)

    def _load(self, model_path: str, scaler_path: str):
        """Load Model and Scaler."""
        try:
            # 1. Load Model
            self._model = ResNet1D(
                len(self.CONFIG['feature_cols']), self.num_classes).to(self.device)
            self._model.load_state_dict(torch.load(
                model_path, map_location=self.device))
            self._model.eval()
            Log.i(self.TAG, f"Model loaded from {model_path}")

            # 2. Load Scaler
            if scaler_path and os.path.exists(scaler_path):
                self._scaler = joblib.load(scaler_path)
                Log.i(self.TAG, f"Scaler loaded from {scaler_path}")
            else:
                Log.w(
                    self.TAG, "Scaler path invalid or not provided. Will fit temporary scaler at runtime.")
                self._scaler = None

        except Exception as e:
            Log.e(self.TAG, f"Failed to load resources: {e}")
            raise

    # ==========================================
    # DATA PRE-PROCESSING
    # ==========================================

    def _resample_to_constant_rate(self, df: pd.DataFrame, cols_to_interp: list):
        """
        Resamples dataframe to a constant sampling rate based on median dt.
        Returns: new_df, target_dt
        """
        times = df['Relative_time'].values
        if len(times) < 100:
            return df, 1.0

        # Calculate target dt from first 100 samples
        dt_initial = np.diff(times[:100])
        target_dt = np.median(dt_initial)

        if target_dt <= 0 or np.isnan(target_dt):
            return df, 1.0

        t_start = times[0]
        t_end = times[-1]
        new_time_grid = np.arange(t_start, t_end, target_dt)

        new_df = pd.DataFrame({'Relative_time': new_time_grid})

        for col in cols_to_interp:
            if col in df.columns:
                new_df[col] = np.interp(new_time_grid, times, df[col].values)

        return new_df, target_dt

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean, resample and feature gen."""
        # 1. Basic Cleaning
        cols_needed = ["Relative_time", "Dissipation", "Resonance_Frequency"]
        for c in cols_needed:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df.dropna(subset=['Relative_time'], inplace=True)
        df.sort_values('Relative_time', inplace=True)

        # 2. Resample (Crucial step from infer_2.py)
        resampled_df, _ = self._resample_to_constant_rate(
            df, ["Dissipation", "Resonance_Frequency"]
        )

        # 3. Generate Features
        if FastFeatures is None:
            raise ImportError(
                "FastFeatures class missing (train_2.py not found).")

        final_df = FastFeatures.generate(resampled_df)
        return final_df

    # ==========================================
    # CORE INFERENCE (Sliding Window Voting)
    # ==========================================

    def _run_inference_loop(self, df_feats: pd.DataFrame) -> np.ndarray:
        """
        Runs the model with sliding windows and accumulates probabilities (voting).
        """
        # Prepare Feature Matrix
        features = df_feats[self.CONFIG['feature_cols']].values

        # Handle Scaling
        if self._scaler:
            features = self._scaler.transform(features)
        else:
            # Fallback: fit on the fly if no global scaler provided
            temp_scaler = RobustScaler()
            features = temp_scaler.fit_transform(features)

        features = features.astype(np.float32)
        features = np.clip(features, -10, 10)
        features = np.nan_to_num(features, nan=0.0)

        # Padding
        w_size = self.CONFIG['window_size']
        pad_size = w_size // 2
        start_pad = features[:pad_size][::-1]
        end_pad = features[-pad_size:][::-1]
        padded_feats = np.vstack([start_pad, features, end_pad])

        stride = self.CONFIG['inference_stride']
        indices = range(0, len(padded_feats) - w_size, stride)

        # Accumulators
        probabilities = np.zeros((len(df_feats), self.num_classes))
        counts = np.zeros((len(df_feats), self.num_classes))

        batch_size = 64
        current_batch = []
        current_map_indices = []

        Log.d(self.TAG, "Running sliding window inference...")

        with torch.no_grad():
            for i in indices:
                window = padded_feats[i: i + w_size].T
                current_batch.append(window)
                current_map_indices.append(i)

                if len(current_batch) == batch_size:
                    self._process_batch(
                        current_batch, current_map_indices, probabilities, counts, stride, len(df_feats))
                    current_batch = []
                    current_map_indices = []

            # Process remaining
            if current_batch:
                self._process_batch(
                    current_batch, current_map_indices, probabilities, counts, stride, len(df_feats))

        counts[counts == 0] = 1
        raw_probs = probabilities / counts

        # Post-Processing Smoothing
        # 1. Median Filter (remove barcode noise)
        smooth_probs = median_filter(raw_probs, size=(51, 1))
        # 2. Gaussian Filter (smooth transitions)
        smooth_probs = gaussian_filter1d(smooth_probs, sigma=10, axis=0)

        return smooth_probs

    def _process_batch(self, batch, map_indices, probs_arr, counts_arr, stride, total_len):
        x = torch.tensor(np.array(batch), dtype=torch.float32).to(self.device)
        out = self._model(x).cpu().numpy()

        for idx, pred in zip(map_indices, out):
            start_p = max(0, idx - stride//2)
            end_p = min(total_len, idx + stride//2 + 1)
            probs_arr[start_p:end_p] += pred
            counts_arr[start_p:end_p] += 1
    def _enforce_sequential_constraints(self, candidates_map: Dict[str, List[Tuple[int, float]]]) -> Dict[str, Dict]:
        """
        Post-processing to fix order.
        Rule: If POI[i] is missing, check if POI[i+1] has multiple peaks. 
        If so, and the earlier peak fits the sequence, reclassify it to POI[i].
        """
        final_output = {}
        poi_configs = self.CONFIG['poi_config']
        
        # We need a list of labels to iterate by index
        labels = [c['label'] for c in poi_configs]
        
        # Track the index of the previously confirmed POI to ensure time linearity
        last_confirmed_idx = 0 

        for i in range(len(labels)):
            current_label = labels[i]
            current_candidates = candidates_map.get(current_label, [])
            
            # 1. Check if current is missing or empty
            if not current_candidates:
                # LOOK AHEAD: Check if the NEXT label has extra peaks to spare
                if i + 1 < len(labels):
                    next_label = labels[i+1]
                    next_candidates = candidates_map.get(next_label, [])
                    
                    # Heuristic: The next class has > 1 peak, or we are desperate
                    if len(next_candidates) >= 2:
                        # Sort by time
                        next_candidates.sort(key=lambda x: x[0])
                        
                        earliest_next_peak = next_candidates[0] # The "First Yellow Peak"
                        
                        # VALIDATION: Is this stolen peak actually after the previous POI?
                        if earliest_next_peak[0] > last_confirmed_idx:
                            Log.i(self.TAG, f"Reclassification: Stealing {next_label} peak at {earliest_next_peak[0]} for missing {current_label}")
                            
                            # Assign to current
                            current_candidates = [earliest_next_peak]
                            
                            # Remove from next (so we don't detect it twice)
                            candidates_map[next_label].pop(0)

            # 2. Select the Best Candidate for the current Class
            # We want the highest confidence that is ALSO after last_confirmed_idx
            best_candidate = (-1, -1.0)
            
            # Filter candidates that are temporally valid
            valid_candidates = [c for c in current_candidates if c[0] > last_confirmed_idx]
            
            if valid_candidates:
                # Heuristic: Usually pick the highest confidence among valid ones
                # OR pick the earliest one if we want strict left-to-right filling. 
                # Let's pick highest confidence to be safe, or earliest if confidences are similar.
                # Given the user issue, picking the EARLIEST valid one is safer for sequential logic.
                valid_candidates.sort(key=lambda x: x[0]) 
                best_candidate = valid_candidates[0]
                
                # Update constraint for next loop
                last_confirmed_idx = best_candidate[0]
            
            final_output[current_label] = {
                'resampled_index': best_candidate[0],
                'confidence': best_candidate[1]
            }
            
        return final_output
    # ==========================================
    # PEAK LOGIC (Hill vs Cliff)
    # ==========================================
    def _get_predictions_from_probs(self, probs: np.ndarray) -> Dict[str, List[Tuple[int, float]]]:
        """
        Modified to return ALL candidate peaks for each class, sorted by confidence.
        Returns: Dict { 'POI1': [(idx, conf), (idx, conf)...] }
        """
        candidates_map = {}

        for i in range(self.num_classes):
            label = self.CONFIG['poi_config'][i]['label']
            p_curve = probs[:, i]
            
            # Smooth lightly
            p_curve = gaussian_filter1d(p_curve, sigma=5)

            candidates = []

            # 1. Find all peaks with decent height
            peaks, props = find_peaks(p_curve, height=0.1, distance=50) # Distance 50 prevents getting the same peak twice
            
            if len(peaks) > 0:
                for p_idx, p_height in zip(peaks, props['peak_heights']):
                    candidates.append((int(p_idx), float(p_height)))
            else:
                # Fallback: Look for Cliff/Slope if no peaks found
                slope = np.gradient(p_curve)
                slope_peaks, slope_props = find_peaks(slope, height=0.005, distance=50)
                for p_idx, p_height in zip(slope_peaks, slope_props['peak_heights']):
                     # Use raw prob as confidence even if slope detected
                    candidates.append((int(p_idx), float(p_curve[p_idx])))

            # Sort candidates by time (index) initially
            candidates.sort(key=lambda x: x[0]) 
            candidates_map[label] = candidates

        return candidates_map   
    # ==========================================
    # HELPERS
    # ==========================================

    def _map_resampled_to_original(self, resampled_idx: int, resampled_df: pd.DataFrame, original_df: pd.DataFrame) -> int:
        """
        Maps an index from the resampled (interpolated) dataframe back to the closest index 
        in the original raw dataframe based on 'Relative_time'.
        """
        if resampled_idx == -1 or resampled_idx >= len(resampled_df):
            return -1

        target_time = resampled_df['Relative_time'].iloc[resampled_idx]

        # Find index in original_df closest to target_time
        # searchsorted requires sorted array
        times = original_df['Relative_time'].values
        idx = np.searchsorted(times, target_time)

        if idx == 0:
            return 0
        if idx == len(times):
            return len(times) - 1

        # Check which neighbor is closer
        if abs(times[idx] - target_time) < abs(times[idx-1] - target_time):
            return int(idx)
        else:
            return int(idx-1)

    def _reset_file_buffer(self, file_buffer: Union[str, object]) -> Union[str, object]:
        if isinstance(file_buffer, str):
            return file_buffer
        if hasattr(file_buffer, "seekable") and file_buffer.seekable():
            file_buffer.seek(0)
            return file_buffer
        else:
            raise Exception("Cannot seek stream prior to processing.")

    def _validate_file_buffer(self, file_buffer: Union[str, object]) -> pd.DataFrame:
        try:
            file_buffer = self._reset_file_buffer(file_buffer=file_buffer)
            df = pd.read_csv(file_buffer)
        except Exception as e:
            raise ValueError(f"Error reading CSV: {e}")

        required = {"Dissipation", "Resonance_Frequency", "Relative_time"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns. Need {required}")
        return df

    def _get_default_output(self):
        output = {}
        for conf in self.CONFIG['poi_config']:
            output[conf['label']] = {'indices': [-1], 'confidences': [-1]}
        return output

    # ==========================================
    # PUBLIC API (predict & visualize)
    # ==========================================

    def predict(self, progress_signal: Any = None, file_buffer: Any = None,
                df: pd.DataFrame = None, visualize: bool = False):
        """
        Standard QModel prediction entry point.
        """
        if self._model is None:
            raise ValueError("Model not loaded")

        # 1. Load Data
        if df is None and file_buffer is not None:
            try:
                df = self._validate_file_buffer(file_buffer)
            except Exception as e:
                Log.e(self.TAG, str(e))
                return self._get_default_output()
        elif df is None:
            raise ValueError("No data provided")

        self._last_original_df = df.copy()

        # 2. Preprocess
        resampled_df = self._preprocess(df.copy())
        self._last_resampled_df = resampled_df

        # 3. Inference
        probs = self._run_inference_loop(resampled_df)
        self._last_probs = probs

        # 4. Peak Analysis (UPDATED)
        # Step A: Get all candidates
        candidate_map = self._get_predictions_from_probs(probs)
        
        # Step B: Run the sequential fixer (The "Yellow Peak" logic)
        raw_preds = self._enforce_sequential_constraints(candidate_map)

        # 5. Format Output (Map back to original indices)
        final_output = {}

        for label, data in raw_preds.items():
            r_idx = data['resampled_index']
            conf = data['confidence']

            if r_idx != -1:
                orig_idx = self._map_resampled_to_original(
                    r_idx, resampled_df, df)
                final_output[label] = {
                    'indices': [orig_idx],
                    'confidences': [conf]
                }
            else:
                final_output[label] = {
                    'indices': [-1],
                    'confidences': [-1]
                }

        self._last_predictions = final_output

        if visualize:
            self.visualize()

        return final_output

    def visualize(self, figsize: Tuple[int, int] = (15, 10), save_path: Optional[str] = None):
        """
        Reproduces plot_results from infer_2.py using class state.
        """
        if self._last_resampled_df is None or self._last_probs is None:
            print("No prediction data to visualize.")
            return

        df = self._last_resampled_df
        probs = self._last_probs
        preds = self._last_predictions

        time_axis = df['Relative_time'].values

        plt.figure(figsize=figsize)

        # 1. Dissipation Curve
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(time_axis, df['Dissipation'],
                 color='black', alpha=0.6, label='Dissipation')
        ax1.set_ylabel('Dissipation (Resampled)')
        ax1.set_title('ResNet System Predictions')

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        y_min, y_max = ax1.get_ylim()

        # Plot Predictions
        # Note: preds contains ORIGINAL indices. We need RESAMPLED indices for this plot
        # because we are plotting against the resampled time axis/dataframe.
        # We need to reverse map or just look up the time from original and find it here.
        # Simpler approach for Vis: Use the stored `_last_probs` logic which operated on resampled data.

        # Let's re-extract the resampled indices from logic for plotting accuracy
        # or find the time in resampled df that matches the prediction time

        for i, config in enumerate(self.CONFIG['poi_config']):
            label = config['label']
            color = colors[i % len(colors)]

            if label in preds and preds[label]['indices'][0] != -1:
                # Get original index
                orig_idx = preds[label]['indices'][0]
                # Get time from original df
                orig_time = self._last_original_df['Relative_time'].iloc[orig_idx]

                ax1.vlines(orig_time, y_min, y_max, colors=color,
                           linestyles='solid', linewidth=2)
                ax1.plot(orig_time, df['Dissipation'].iloc[min(len(df)-1, int(orig_idx * (len(df)/len(self._last_original_df))))],  # Approx Y pos
                         'o', color=color, label=f'Pred {label}')

        ax1.legend(loc='upper right')

        # 2. Probability Curves
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        for i in range(self.num_classes):
            label = self.CONFIG["poi_config"][i]["label"]
            color = colors[i % len(colors)]
            ax2.plot(time_axis, probs[:, i], color=color,
                     alpha=0.8, label=f'{label} Prob')

        ax2.set_ylabel('Model Probability')
        ax2.set_xlabel('Relative Time')
        ax2.set_ylim(0, 1.1)
        ax2.legend(loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            Log.i(self.TAG, f"Saved plot to {save_path}")

        plt.show()
