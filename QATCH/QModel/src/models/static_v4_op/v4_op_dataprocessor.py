import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Union, Dict
from tqdm import tqdm
import numpy as np
import os
import random
from sklearn.svm import OneClassSVM
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from collections import defaultdict
import matplotlib.patches as mpatches
from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import peak_prominences
from scipy.stats import skew, kurtosis
from numpy.lib.stride_tricks import sliding_window_view


class OPDataProcessor:
    DROP = ["Date", "Time", "Ambient", "Peak Magnitude (RAW)", "Temperature"]
    BASELINE_WIN = 500
    ROLLING_WIN = 50

    @staticmethod
    def gen_features(df: pd.DataFrame) -> pd.DataFrame:

        import numpy as np
        import pandas as pd
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import peak_prominences
        from scipy.stats import skew, kurtosis

        # Work with numpy arrays for speed
        df = df.copy()

        # Drop unnecessary columns
        df = df.drop(
            columns=[col for col in OPDataProcessor.DROP if col in df.columns], errors='ignore')

        # Extract arrays once - avoid repeated DataFrame access
        time = df['Relative_time'].values.astype(np.float64)
        diss = df['Dissipation'].values.astype(np.float64)
        rf = df['Resonance_Frequency'].values.astype(np.float64)

        n = len(time)

        # Handle non-monotonic time by sorting if needed
        if not np.all(np.diff(time) >= 0):
            sort_idx = np.argsort(time)
            time = time[sort_idx]
            diss = diss[sort_idx]
            rf = rf[sort_idx]
            # We'll need to unsort at the end
            unsort_idx = np.empty_like(sort_idx)
            unsort_idx[sort_idx] = np.arange(n)
            needs_unsorting = True
        else:
            needs_unsorting = False

        dt_median = np.median(np.diff(time))

        # ========== 1. BASELINE CORRECTION (VECTORIZED) ==========
        baseline_time = time[0] + OPDataProcessor.BASELINE_WIN * dt_median
        baseline_mask = time <= baseline_time

        base_d = np.median(diss[baseline_mask]) if np.any(
            baseline_mask) else np.median(diss[:10])
        base_rf = np.median(rf[baseline_mask]) if np.any(
            baseline_mask) else np.median(rf[:10])

        diss_corrected = diss - base_d
        rf_corrected = -(rf - base_rf)

        # Pre-allocate result arrays
        results = {}

        # ========== 2. MULTI-SCALE SMOOTHING (BATCH PROCESSING) ==========
        sigmas = np.array([1, 2, 4, 8])
        for sigma in sigmas:
            results[f'Diss_gauss_{sigma}'] = gaussian_filter1d(
                diss_corrected, sigma=sigma)
            results[f'RF_gauss_{sigma}'] = gaussian_filter1d(
                rf_corrected, sigma=sigma)

        # ========== 3. GRADIENT FEATURES (VECTORIZED) ==========
        # Use numpy's efficient gradient
        diss_grad1 = np.gradient(diss_corrected, time)
        rf_grad1 = np.gradient(rf_corrected, time)
        diss_grad2 = np.gradient(diss_grad1, time)
        rf_grad2 = np.gradient(rf_grad1, time)

        # Smooth gradients
        smooth_diss = gaussian_filter1d(diss_corrected, sigma=2)
        smooth_rf = gaussian_filter1d(rf_corrected, sigma=2)
        diss_smooth_grad1 = np.gradient(smooth_diss, time)
        rf_smooth_grad1 = np.gradient(smooth_rf, time)

        results.update({
            'Dissipation_corrected_grad1': diss_grad1,
            'Dissipation_corrected_grad2': diss_grad2,
            'RF_corrected_grad1': rf_grad1,
            'RF_corrected_grad2': rf_grad2,
            'Dissipation_corrected_smooth_grad1': diss_smooth_grad1,
            'Dissipation_corrected_smooth_grad2': np.gradient(diss_smooth_grad1, time),
            'RF_corrected_smooth_grad1': rf_smooth_grad1,
            'RF_corrected_smooth_grad2': np.gradient(rf_smooth_grad1, time)
        })

        # ========== 4. OPTIMIZED ROLLING OPERATIONS (NO PANDAS) ==========
        def fast_rolling_stats(signal, time_arr, window_time):
            """Compute rolling mean and std without pandas"""
            n = len(signal)
            means = np.zeros(n)
            stds = np.zeros(n)

            for i in range(n):
                current_time = time_arr[i]
                half_window = window_time / 2

                # Find points in window
                mask_start = max(0, i - 100)
                mask_end = min(n, i + 100)

                values = []
                for j in range(mask_start, mask_end):
                    if abs(time_arr[j] - current_time) <= half_window:
                        values.append(signal[j])

                if len(values) > 0:
                    values_arr = np.array(values)
                    means[i] = np.mean(values_arr)
                    stds[i] = np.std(values_arr) if len(values) > 1 else 0.0
                else:
                    means[i] = signal[i]
                    stds[i] = 0.0

            return means, stds

        # Compute rolling stats for all windows
        for window_points in [10, 20, 50]:
            window_time = window_points * dt_median

            diss_mean, diss_std = fast_rolling_stats(
                diss_corrected, time, window_time)
            rf_mean, rf_std = fast_rolling_stats(
                rf_corrected, time, window_time)

            results[f'Diss_roll_mean_{window_points}'] = diss_mean
            results[f'Diss_roll_std_{window_points}'] = diss_std
            results[f'RF_roll_mean_{window_points}'] = rf_mean
            results[f'RF_roll_std_{window_points}'] = rf_std

            # Z-scores
            results[f'Diss_zscore_{window_points}'] = (
                diss_corrected - diss_mean) / (diss_std + 1e-8)
            results[f'RF_zscore_{window_points}'] = (
                rf_corrected - rf_mean) / (rf_std + 1e-8)

        # ========== 5. CHANGE SCORE (VECTORIZED) ==========
        def fast_change_score(signal, time_arr, window_time):
            n = len(signal)
            result = np.zeros(n)

            for i in range(n):
                current_time = time_arr[i]

                left_vals = []
                right_vals = []

                # Limited search range for efficiency
                for j in range(max(0, i-50), min(n, i+50)):
                    t = time_arr[j]
                    if current_time - window_time <= t < current_time:
                        left_vals.append(signal[j])
                    elif current_time < t <= current_time + window_time:
                        right_vals.append(signal[j])

                if len(left_vals) > 0 and len(right_vals) > 0:
                    result[i] = abs(
                        np.mean(np.array(right_vals)) - np.mean(np.array(left_vals)))

            return result

        window_time = 20 * dt_median
        results['Diss_change_score'] = fast_change_score(
            diss_corrected, time, window_time)
        results['RF_change_score'] = fast_change_score(
            rf_corrected, time, window_time)

        # ========== 6. PEAK PROMINENCE (OPTIMIZED) ==========
        avg_dt = np.mean(np.diff(time))
        wlen_points = max(3, int(50 * dt_median / avg_dt))
        wlen_points = min(wlen_points, n // 2)  # Ensure wlen is not too large

        peaks = np.arange(n)
        diss_prom, _, _ = peak_prominences(
            diss_corrected, peaks, wlen=wlen_points)
        rf_prom, _, _ = peak_prominences(rf_corrected, peaks, wlen=wlen_points)

        results['Diss_prominence'] = diss_prom
        results['RF_prominence'] = rf_prom

        # ========== 7. CURVATURE (VECTORIZED) ==========
        results['Diss_curvature'] = diss_grad2 / \
            (1 + np.abs(diss_grad1) ** 0.5)
        results['RF_curvature'] = rf_grad2 / (1 + np.abs(rf_grad1) ** 0.5)

        # ========== 8. ENERGY (ULTRA-FAST CONVOLUTION) ==========
        # Use convolution for windowed energy calculation - much faster!
        for window_points in [10, 30]:
            window_samples = int(window_points * dt_median / avg_dt)
            window_samples = max(3, min(window_samples, n // 4))

            # Create window kernel
            kernel = np.ones(window_samples) / window_samples

            # Compute energy using convolution (squared signal)
            diss_energy = np.convolve(diss_corrected**2, kernel, mode='same')
            rf_energy = np.convolve(rf_corrected**2, kernel, mode='same')

            results[f'Diss_energy_{window_points}'] = diss_energy * \
                window_samples
            results[f'RF_energy_{window_points}'] = rf_energy * window_samples

        # ========== 9. CROSS-SIGNAL (VECTORIZED) ==========
        results['Diss_RF_product'] = diss_corrected * rf_corrected
        results['Diss_RF_ratio'] = diss_corrected / \
            (np.abs(rf_corrected) + 1e-8)
        results['gradient_alignment'] = diss_grad1 * rf_grad1
        results['Diss_RF_phase_diff'] = np.angle(
            diss_corrected + 1j * rf_corrected)

        # ========== 10. ANOMALY SCORE (REUSE ROLLING STATS) ==========
        # Use pre-computed rolling stats
        window_points = 50
        if f'Diss_roll_mean_{window_points}' in results:
            diss_mean = results[f'Diss_roll_mean_{window_points}']
            diss_std = results[f'Diss_roll_std_{window_points}']
            rf_mean = results[f'RF_roll_mean_{window_points}']
            rf_std = results[f'RF_roll_std_{window_points}']
        else:
            # Fallback computation
            window_time = window_points * dt_median
            diss_mean, diss_std = fast_rolling_stats(
                diss_corrected, time, window_time)
            rf_mean, rf_std = fast_rolling_stats(
                rf_corrected, time, window_time)

        diss_anomaly = np.abs((diss_corrected - diss_mean) / (diss_std + 1e-8))
        rf_anomaly = np.abs((rf_corrected - rf_mean) / (rf_std + 1e-8))

        # Normalize using min-max scaling
        diss_min, diss_max = diss_anomaly.min(), diss_anomaly.max()
        rf_min, rf_max = rf_anomaly.min(), rf_anomaly.max()

        results['Diss_anomaly_smooth'] = (
            diss_anomaly - diss_min) / (diss_max - diss_min + 1e-8)
        results['RF_anomaly_smooth'] = (
            rf_anomaly - rf_min) / (rf_max - rf_min + 1e-8)

        # ========== 11. TEMPORAL CONTEXT (VECTORIZED) ==========
        time_range = time[-1] - time[0]
        results['relative_position'] = (time - time[0]) / (time_range + 1e-8)
        results['time_from_start'] = time - time[0]

        # Sampling rate with edge handling
        sampling_rate = np.ones(n) / dt_median
        if n > 1:
            sampling_rate[1:] = 1.0 / np.maximum(np.diff(time), 1e-8)
            sampling_rate[0] = sampling_rate[1]

        results['local_sampling_rate'] = gaussian_filter1d(
            sampling_rate, sigma=min(5, n//10))

        # ========== 12. STATISTICAL MOMENTS (BROADCAST) ==========
        results['Diss_skew_global'] = np.full(n, skew(diss_corrected))
        results['Diss_kurtosis_global'] = np.full(n, kurtosis(diss_corrected))
        results['RF_skew_global'] = np.full(n, skew(rf_corrected))
        results['RF_kurtosis_global'] = np.full(n, kurtosis(rf_corrected))

        # ========== 13. DIFFERENCE FEATURES (VECTORIZED) ==========
        total_time = time[-1] - time[0]
        start_time = time[0] + total_time / 10
        end_time = time[0] + total_time / 5
        mask = (time >= start_time) & (time <= end_time)

        if np.any(mask):
            avg_freq = np.median(rf_corrected[mask])
            avg_diss = np.median(diss_corrected[mask])
        else:
            # Fallback if no points in range
            avg_freq = np.median(rf_corrected[:n//5])
            avg_diss = np.median(diss_corrected[:n//5])

        diff_robust = rf_corrected - 2 * diss_corrected * 0.5
        results['Difference_robust'] = diff_robust
        results['Difference_smooth'] = gaussian_filter1d(
            diff_robust, sigma=min(3, n//20))

        # Add corrected base signals
        results['Dissipation_corrected'] = diss_corrected
        results['RF_corrected'] = rf_corrected

        # ========== 14. UNSORT IF NEEDED ==========
        if needs_unsorting:
            for key in results:
                results[key] = results[key][unsort_idx]

        # ========== 15. BUILD FINAL DATAFRAME ==========
        # Add original columns that aren't in results
        for col in df.columns:
            if col not in results:
                results[col] = df[col].values

        # Create DataFrame
        df_result = pd.DataFrame(results)

        # ========== 16. CLEAN UP (VECTORIZED) ==========
        # Fast vectorized cleanup
        numeric_cols = df_result.select_dtypes(include=[np.number]).columns
        exclude_cols = {'Relative_time', 'Sample_number'}

        for col in numeric_cols:
            if col not in exclude_cols:
                values = df_result[col].values

                # Replace inf with finite values
                mask_inf = np.isinf(values)
                if np.any(mask_inf):
                    finite_vals = values[~mask_inf]
                    if len(finite_vals) > 0:
                        values[mask_inf] = np.median(finite_vals)
                    else:
                        values[mask_inf] = 0

                # Fill NaN
                mask_nan = np.isnan(values)
                if np.any(mask_nan):
                    values[mask_nan] = 0

                # Clip outliers
                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    if std_val > 0:
                        values = np.clip(values, mean_val -
                                         5*std_val, mean_val + 5*std_val)

                df_result[col] = values

        return df_result
