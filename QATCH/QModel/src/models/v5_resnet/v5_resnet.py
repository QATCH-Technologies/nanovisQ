import os
import glob
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
# LOGGING UTILS
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
# FEATURE ENGINEERING
# ==========================================


class FastFeatures:
    @staticmethod
    def generate(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)

        # 1. Aggressive Pre-Smoothing
        for col in ["Dissipation", "Resonance_Frequency"]:
            if col in df.columns:
                df[col] = gaussian_filter1d(df[col].values, sigma=5.0)

        # 2. Compute "Difference" (Vectorized)
        if hasattr(FastFeatures, '_compute_difference_vectorized'):
            df["Difference"] = FastFeatures._compute_difference_vectorized(df)

        # 3. DoG Features
        diss_vals = df['Dissipation'].values
        diff_vals = df['Difference'].values if "Difference" in df else np.zeros_like(
            diss_vals)

        df['Dissipation_DoG'] = gaussian_filter1d(diss_vals, sigma=2, order=1)
        df['Difference_DoG'] = gaussian_filter1d(diff_vals, sigma=2, order=1)

        # 4. Anomaly Score
        df['Dissipation_DoG_Anomaly_Score'] = FastFeatures._fast_anomaly_score(
            df['Dissipation_DoG'].values
        )

        # 5. Derivatives
        time_vals = df['Relative_time'].values
        dt = np.diff(time_vals, prepend=time_vals[0])
        dt[dt <= 0] = np.nanmedian(dt) if len(dt) > 0 else 1.0

        diss_diff = np.diff(diss_vals, prepend=diss_vals[0])
        rf_diff = np.diff(df['Resonance_Frequency'].values,
                          prepend=df['Resonance_Frequency'].values[0])

        diss_slope = diss_diff / dt
        rf_slope = rf_diff / dt

        # 6. Slope Ratio
        rf_slope_safe = np.where(
            np.abs(rf_slope) < 1e-5, np.sign(rf_slope) * 1e-5, rf_slope)
        rf_slope_safe = np.where(rf_slope_safe == 0, 1e-5, rf_slope_safe)
        df['slope_ratio'] = np.clip(diss_slope / rf_slope_safe, -100, 100)

        # 7. Rolling Std
        df['Diss_roll_std'] = df['Dissipation'].rolling(
            window=50, min_periods=1).std().fillna(0)

        # 8. Save Deltas
        df['Dissipation_Delta'] = diss_slope
        df['Resonance_Delta'] = rf_slope

        # 9. Clean up
        df['Relative_time'] = df['Relative_time'].astype(np.float32)
        df.fillna(0, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)

        return df

    @staticmethod
    def _compute_difference_vectorized(df, difference_factor=2):
        xs = df["Relative_time"].values
        res_freq = df["Resonance_Frequency"].values
        diss = df["Dissipation"].values

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
# MODEL ARCHITECTURE
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

        self.layer1 = self._make_layer(32, 2, stride=1, dilation=1)
        self.layer2 = self._make_layer(64, 2, stride=2, dilation=2)
        self.layer3 = self._make_layer(128, 2, stride=2, dilation=4)
        self.layer4 = self._make_layer(256, 2, stride=2, dilation=8)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.rnn = nn.LSTM(input_size=256, hidden_size=128,
                           num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 2, num_classes),
            nn.Sigmoid()
        )
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
        x_float = x.float()
        x_float, _ = self.rnn(x_float)
        x = x_float[:, -1, :]
        return self.fc(x)


# ==========================================
# MAIN PREDICTOR CLASS (ENSEMBLE)
# ==========================================

class QModelV5Resnet:
    """
    QModel wrapper for ResNet1D ENSEMBLE inference.
    """
    TAG = "EnsembleSystem"

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

    def __init__(self, model_pattern: str, scaler_path: str = None, device_str: str = None):
        self.device = torch.device(
            device_str if device_str else (
                "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.num_classes = len(self.CONFIG['poi_config'])
        self._models = []
        self._scaler = None

        # State
        self._last_original_df = None
        self._last_resampled_df = None
        self._last_mean_probs = None
        self._last_std_probs = None
        self._last_predictions = None

        self._load(model_pattern, scaler_path)

    def _load(self, model_pattern: str, scaler_path: str):
        try:
            model_files = glob.glob(model_pattern)
            if not model_files:
                raise FileNotFoundError(
                    f"No models found matching: {model_pattern}")

            Log.i(self.TAG, f"Found {len(model_files)} ensemble members.")
            for m_path in model_files:
                try:
                    model = ResNet1D(
                        len(self.CONFIG['feature_cols']), self.num_classes).to(self.device)
                    model.load_state_dict(torch.load(
                        m_path, map_location=self.device))
                    model.eval()
                    self._models.append(model)
                except Exception as e:
                    Log.e(self.TAG, f"Failed to load member {m_path}: {e}")

            if not self._models:
                raise ValueError("No valid models could be loaded.")

            if scaler_path and os.path.exists(scaler_path):
                self._scaler = joblib.load(scaler_path)
            else:
                Log.w(
                    self.TAG, "Scaler path invalid. Fitting temporary scaler at runtime.")
                self._scaler = None

        except Exception as e:
            Log.e(self.TAG, f"Failed to load resources: {e}")
            raise

    # ==========================================
    # DATA PRE-PROCESSING
    # ==========================================

    def _resample_to_constant_rate(self, df: pd.DataFrame, cols_to_interp: list):
        times = df['Relative_time'].values
        if len(times) < 100:
            return df, 1.0

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
        cols_needed = ["Relative_time", "Dissipation", "Resonance_Frequency"]
        for c in cols_needed:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df.dropna(subset=['Relative_time'], inplace=True)
        df.sort_values('Relative_time', inplace=True)

        resampled_df, _ = self._resample_to_constant_rate(
            df, ["Dissipation", "Resonance_Frequency"])
        final_df = FastFeatures.generate(resampled_df)
        return final_df

    # ==========================================
    # ENSEMBLE INFERENCE
    # ==========================================

    def _run_single_model_raw(self, model, padded_feats, total_len, stride, w_size):
        indices = range(0, len(padded_feats) - w_size, stride)
        probabilities = np.zeros((total_len, self.num_classes))
        counts = np.zeros((total_len, self.num_classes))

        batch_size = 64
        current_batch = []
        current_map_indices = []

        with torch.no_grad():
            for i in indices:
                window = padded_feats[i: i + w_size].T
                current_batch.append(window)
                current_map_indices.append(i)

                if len(current_batch) == batch_size:
                    self._process_batch(
                        model, current_batch, current_map_indices, probabilities, counts, stride, total_len)
                    current_batch = []
                    current_map_indices = []

            if current_batch:
                self._process_batch(
                    model, current_batch, current_map_indices, probabilities, counts, stride, total_len)

        counts[counts == 0] = 1
        return probabilities / counts

    def _process_batch(self, model, batch, map_indices, probs_arr, counts_arr, stride, total_len):
        x = torch.tensor(np.array(batch), dtype=torch.float32).to(self.device)
        out = model(x).cpu().numpy()

        for idx, pred in zip(map_indices, out):
            start_p = max(0, idx - stride//2)
            end_p = min(total_len, idx + stride//2 + 1)
            probs_arr[start_p:end_p] += pred
            counts_arr[start_p:end_p] += 1

    def _run_ensemble_inference(self, df_feats: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        features = df_feats[self.CONFIG['feature_cols']].values
        if self._scaler:
            features = self._scaler.transform(features)
        else:
            temp_scaler = RobustScaler()
            features = temp_scaler.fit_transform(features)

        features = features.astype(np.float32)
        features = np.clip(features, -10, 10)
        features = np.nan_to_num(features, nan=0.0)

        w_size = self.CONFIG['window_size']
        pad_size = w_size // 2
        start_pad = features[:pad_size][::-1]
        end_pad = features[-pad_size:][::-1]
        padded_feats = np.vstack([start_pad, features, end_pad])
        stride = self.CONFIG['inference_stride']

        all_probs = []
        Log.d(self.TAG, f"Running inference on {len(self._models)} models...")
        for model in self._models:
            raw = self._run_single_model_raw(
                model, padded_feats, len(df_feats), stride, w_size)
            all_probs.append(raw)

        all_probs = np.array(all_probs)
        mean_probs = np.mean(all_probs, axis=0)
        std_probs = np.std(all_probs, axis=0)

        # Smooth Mean
        mean_probs = median_filter(mean_probs, size=(51, 1))
        mean_probs = gaussian_filter1d(mean_probs, sigma=10, axis=0)

        return mean_probs, std_probs

    # ==========================================
    # LOGIC & CONSTRAINTS
    # ==========================================

    def _enforce_sequential_constraints(self, candidates_map: Dict[str, List[Tuple[int, float]]]) -> Dict[str, Dict]:
        """
        1. Steals peaks from next class if current is missing.
        2. Filters for temporal validity (must be after last_confirmed_idx).
        3. Sorts by CONFIDENCE (Highest first).
        """
        final_output = {}
        poi_configs = self.CONFIG['poi_config']
        labels = [c['label'] for c in poi_configs]
        last_confirmed_idx = 0

        for i in range(len(labels)):
            current_label = labels[i]
            current_candidates = candidates_map.get(current_label, [])

            # Steal peak logic
            if not current_candidates:
                if i + 1 < len(labels):
                    next_label = labels[i+1]
                    next_candidates = candidates_map.get(next_label, [])
                    if len(next_candidates) >= 2:
                        next_candidates.sort(key=lambda x: x[0])
                        earliest_next_peak = next_candidates[0]
                        if earliest_next_peak[0] > last_confirmed_idx:
                            current_candidates = [earliest_next_peak]
                            candidates_map[next_label].pop(0)

            best_candidate = (-1, -1.0)
            valid_candidates = [
                c for c in current_candidates if c[0] > last_confirmed_idx]

            if valid_candidates:
                # Pick Highest Confidence among valid
                valid_candidates.sort(key=lambda x: x[1], reverse=True)
                best_candidate = valid_candidates[0]
                last_confirmed_idx = best_candidate[0]

            final_output[current_label] = {
                'resampled_index': best_candidate[0],
                'confidence': best_candidate[1]
            }

        return final_output

    def _apply_temporal_constraints(self, preds: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Applies logic to remove POIs based on interval duration.
        Logic is applied on RESAMPLED INDICES (which correlates directly to time).

        Rules:
        1. If Delta(POI2->POI3) <= Delta(POI1->POI2): Drop POI3.
        2. If Delta(POI3->POI4) < Delta(POI2->POI3): Drop POI4.
        3. If Delta(POI4->POI5) < Delta(POI3->POI4): Drop POI5.

        Assuming cascading failure: if a previous node is dropped, the interval logic
        for the next node breaks, so it is also dropped.
        """

        def get_idx(label):
            return preds[label]['resampled_index']

        def drop(label, reason):
            Log.i(self.TAG, f"Dropping {label}: {reason}")
            preds[label]['resampled_index'] = -1
            preds[label]['confidence'] = -1

        idx1 = get_idx('POI1')
        idx2 = get_idx('POI2')

        # We can only perform checks if 1 and 2 exist
        if idx1 != -1 and idx2 != -1:
            delta_1_2 = idx2 - idx1

            # --- Check POI 3 ---
            idx3 = get_idx('POI3')
            if idx3 != -1:
                delta_2_3 = idx3 - idx2
                if delta_2_3 <= delta_1_2:
                    drop(
                        'POI3', f"Delta 2->3 ({delta_2_3}) <= Delta 1->2 ({delta_1_2})")
                    idx3 = -1  # Update for cascade

            # --- Check POI 4 ---
            # Requires valid POI2 and POI3
            idx4 = get_idx('POI4')
            if idx3 != -1 and idx4 != -1:
                delta_2_3 = idx3 - idx2  # Recalculate or reuse
                delta_3_4 = idx4 - idx3
                if delta_3_4 < delta_2_3:
                    drop(
                        'POI4', f"Delta 3->4 ({delta_3_4}) < Delta 2->3 ({delta_2_3})")
                    idx4 = -1
            elif idx4 != -1 and idx3 == -1:
                # Cascade: 3 is missing, so 4 is invalid
                drop('POI4', "Cascading failure: POI3 missing")
                idx4 = -1

            # --- Check POI 5 ---
            # Requires valid POI3 and POI4
            idx5 = get_idx('POI5')
            if idx3 != -1 and idx4 != -1 and idx5 != -1:
                delta_3_4 = idx4 - idx3
                delta_4_5 = idx5 - idx4
                if delta_4_5 < delta_3_4:
                    drop(
                        'POI5', f"Delta 4->5 ({delta_4_5}) < Delta 3->4 ({delta_3_4})")
            elif idx5 != -1 and (idx3 == -1 or idx4 == -1):
                # Cascade
                drop('POI5', "Cascading failure: Previous POIs missing")

        return preds

    def _get_predictions_from_probs(self, probs: np.ndarray) -> Dict[str, List[Tuple[int, float]]]:
        candidates_map = {}
        for i in range(self.num_classes):
            label = self.CONFIG['poi_config'][i]['label']
            p_curve = probs[:, i]
            p_curve = gaussian_filter1d(p_curve, sigma=5)

            candidates = []
            peaks, props = find_peaks(p_curve, height=0.1, distance=50)

            if len(peaks) > 0:
                for p_idx, p_height in zip(peaks, props['peak_heights']):
                    candidates.append((int(p_idx), float(p_height)))
            else:
                slope = np.gradient(p_curve)
                slope_peaks, slope_props = find_peaks(
                    slope, height=0.005, distance=50)
                for p_idx in slope_peaks:
                    candidates.append((int(p_idx), float(p_curve[p_idx])))

            # No sorting needed here, done in _enforce
            candidates_map[label] = candidates

        return candidates_map

    # ==========================================
    # HELPERS
    # ==========================================

    def _map_resampled_to_original(self, resampled_idx: int, resampled_df: pd.DataFrame, original_df: pd.DataFrame) -> int:
        if resampled_idx == -1 or resampled_idx >= len(resampled_df):
            return -1
        target_time = resampled_df['Relative_time'].iloc[resampled_idx]
        times = original_df['Relative_time'].values
        idx = np.searchsorted(times, target_time)
        if idx == 0:
            return 0
        if idx == len(times):
            return len(times) - 1
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
        raise Exception("Cannot seek stream.")

    def _validate_file_buffer(self, file_buffer: Union[str, object]) -> pd.DataFrame:
        file_buffer = self._reset_file_buffer(file_buffer=file_buffer)
        df = pd.read_csv(file_buffer)
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
    # PUBLIC API
    # ==========================================

    def predict(self, file_buffer: Any = None, df: pd.DataFrame = None, visualize: bool = False):
        if not self._models:
            raise ValueError("Models not loaded")

        if df is None and file_buffer is not None:
            try:
                df = self._validate_file_buffer(file_buffer)
            except Exception as e:
                Log.e(self.TAG, str(e))
                return self._get_default_output()
        elif df is None:
            raise ValueError("No data provided")

        self._last_original_df = df.copy()

        # Preprocess
        resampled_df = self._preprocess(df.copy())
        self._last_resampled_df = resampled_df

        # Ensemble Inference
        mean_probs, std_probs = self._run_ensemble_inference(resampled_df)
        self._last_mean_probs = mean_probs
        self._last_std_probs = std_probs

        # 1. Get raw candidates
        candidate_map = self._get_predictions_from_probs(mean_probs)

        # 2. Sequential Fix (Left-to-Right + Highest Confidence)
        raw_preds = self._enforce_sequential_constraints(candidate_map)

        # 3. Temporal Constraints (Relative Delta Logic)
        raw_preds = self._apply_temporal_constraints(raw_preds)

        # 4. Format Output
        final_output = {}
        for label, data in raw_preds.items():
            r_idx = data['resampled_index']
            conf = data['confidence']

            if r_idx != -1:
                orig_idx = self._map_resampled_to_original(
                    r_idx, resampled_df, df)
                unc = float(std_probs[r_idx, [c['label']
                            for c in self.CONFIG['poi_config']].index(label)])

                final_output[label] = {
                    'indices': [orig_idx],
                    'confidences': [conf],
                    'uncertainty': [unc]
                }
            else:
                final_output[label] = {
                    'indices': [-1],
                    'confidences': [-1],
                    'uncertainty': [-1]
                }

        self._last_predictions = final_output

        if visualize:
            self.visualize()

        return final_output

    def visualize(self, figsize: Tuple[int, int] = (15, 12), save_path: Optional[str] = None):
        if self._last_resampled_df is None or self._last_mean_probs is None:
            print("No prediction data to visualize.")
            return

        df = self._last_resampled_df
        mean_probs = self._last_mean_probs
        std_probs = self._last_std_probs
        preds = self._last_predictions
        time_axis = df['Relative_time'].values

        plt.figure(figsize=figsize)

        # 1. Dissipation Curve
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(time_axis, df['Dissipation'],
                 color='black', alpha=0.6, label='Dissipation')
        ax1.set_ylabel('Dissipation (Resampled)')
        ax1.set_title(f'Ensemble Predictions ({len(self._models)} Models)')

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        y_min, y_max = ax1.get_ylim()

        for i, config in enumerate(self.CONFIG['poi_config']):
            label = config['label']
            color = colors[i % len(colors)]
            if label in preds and preds[label]['indices'][0] != -1:
                orig_idx = preds[label]['indices'][0]
                orig_time = self._last_original_df['Relative_time'].iloc[orig_idx]
                ax1.vlines(orig_time, y_min, y_max, colors=color,
                           linestyles='solid', linewidth=2)
                ax1.plot(orig_time, df['Dissipation'].iloc[min(len(df)-1, int(orig_idx * (len(df)/len(self._last_original_df))))],
                         'o', color=color, label=f'Pred {label}')

        ax1.legend(loc='upper right')

        # 2. Probability Curves
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        for i in range(self.num_classes):
            label = self.CONFIG["poi_config"][i]["label"]
            color = colors[i % len(colors)]
            mu = mean_probs[:, i]
            sigma = std_probs[:, i]
            ax2.plot(time_axis, mu, color=color,
                     alpha=1.0, label=f'{label} Mean')
            upper = np.clip(mu + sigma, 0, 1)
            lower = np.clip(mu - sigma, 0, 1)
            ax2.fill_between(time_axis, lower, upper, color=color, alpha=0.2)

        ax2.set_ylabel('Ensemble Probability')
        ax2.set_xlabel('Relative Time')
        ax2.set_ylim(0, 1.1)
        ax2.legend(loc='upper right')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            Log.i(self.TAG, f"Saved plot to {save_path}")
        plt.show()
