import os
import logging
import random
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, peak_prominences, peak_widths

from QATCH.common.logger import Logger

TAG = "[PartialFill]"


class PFDataProcessor:
    SAMPLE_FACTOR = 400
    """
    Processor for PF run data. Handles loading raw content files and generating
    a comprehensive set of statistical, spectral, and temporal features.

    Methods:
        load_content: Discover and return pairs of data and POI file paths.
        generate_features: Build a feature DataFrame from a single run's data.
    """

    @staticmethod
    def load_content(
        data_dir: str,
        num_datasets: Union[int, float] = np.inf
    ) -> List[Tuple[str, str]]:
        """
        Walk `data_dir` recursively to pair each CSV with its POI counterpart.

        Args:
            data_dir: Path to directory containing run CSV files.
            num_datasets: Maximum number of file pairs to return. Use np.inf for all.

        Returns:
            List of tuples (data_csv_path, poi_csv_path).

        Raises:
            NotADirectoryError: If `data_dir` does not exist or is not a directory.
            ValueError: If `num_datasets` is not a positive integer or np.inf.
        """
        # Validate inputs
        if not isinstance(data_dir, str):
            raise TypeError("`data_dir` must be a string path.")
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"Data directory not found: {data_dir}")
        if not (isinstance(num_datasets, (int, float)) and (num_datasets > 0 or num_datasets == np.inf)):
            raise ValueError(
                "`num_datasets` must be a positive int or np.inf.")

        loaded_content: List[Tuple[str, str]] = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if not f.endswith('.csv') or f.endswith(('_poi.csv', '_lower.csv')):
                    continue
                data_path = os.path.join(root, f)
                poi_path = data_path.replace('.csv', '_poi.csv')
                if not os.path.isfile(poi_path):
                    Logger.w("Missing POI file for %s", data_path)
                    continue
                loaded_content.append((data_path, poi_path))

        if not loaded_content:
            Logger.w("No valid data/POI pairs found in %s", data_dir)

        random.shuffle(loaded_content)
        if num_datasets == np.inf:
            return loaded_content
        return loaded_content[:int(num_datasets)]

    @staticmethod
    def _curve_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute basic statistics (mean, std, min, max, skew, kurtosis) for each column.

        Args:
            df: Short DataFrame of numeric curves.

        Returns:
            Single-row DataFrame with stats per column.

        Raises:
            ValueError: If `df` is empty or non-numeric.
        """
        if df.empty:
            raise ValueError("Input DataFrame for _curve_stats is empty.")
        stats = {}
        for col in df.columns:
            vals = df[col].to_numpy(dtype=float)
            stats[f"{col}_mean"] = np.mean(vals)
            stats[f"{col}_std"] = np.std(vals)
            stats[f"{col}_min"] = np.min(vals)
            stats[f"{col}_max"] = np.max(vals)
            stats[f"{col}_skew"] = skew(vals)
            stats[f"{col}_kurtosis"] = kurtosis(vals)
        return pd.DataFrame([stats])

    @staticmethod
    def _peak_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract peak counts, average prominences, and widths for each column.

        Args:
            df: DataFrame of curves.

        Returns:
            Single-row DataFrame with peak features.
        """
        if df.empty:
            raise ValueError("Input DataFrame for _peak_features is empty.")
        results = {}
        for col in df.columns:
            curve = df[col].to_numpy(dtype=float)
            peaks, _ = find_peaks(curve)
            if peaks.size == 0:
                results[f"{col}_peak_count"] = 0.0
                results[f"{col}_mean_prominence"] = 0.0
                results[f"{col}_mean_width"] = 0.0
            else:
                prom = peak_prominences(curve, peaks)[0]
                widths = peak_widths(curve, peaks, rel_height=0.5)[0]
                mask = widths > 0
                results[f"{col}_peak_count"] = float(mask.sum())
                results[f"{col}_mean_prominence"] = float(
                    np.mean(prom[mask])) if mask.any() else 0.0
                results[f"{col}_mean_width"] = float(
                    np.mean(widths[mask])) if mask.any() else 0.0
        return pd.DataFrame([results])

    @staticmethod
    def _curve_dynamics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute slope statistics and area under curve for each column.

        Args:
            df: DataFrame of curves.

        Returns:
            Single-row DataFrame with dynamic features.
        """
        if df.empty:
            raise ValueError("Input DataFrame for _curve_dynamics is empty.")
        results = {}
        for col in df.columns:
            vals = df[col].to_numpy(dtype=float)
            diffs = np.diff(vals)
            results[f"{col}_slope_max"] = float(
                np.max(diffs)) if diffs.size > 0 else 0.0
            results[f"{col}_slope_min"] = float(
                np.min(diffs)) if diffs.size > 0 else 0.0
            results[f"{col}_slope_mean"] = float(
                np.mean(diffs)) if diffs.size > 0 else 0.0
            results[f"{col}_auc"] = float(np.trapz(vals))
        return pd.DataFrame([results])

    @staticmethod
    def _fft_features(
        df: pd.DataFrame,
        sampling_rate: float = 1.0
    ) -> pd.DataFrame:
        """
        Compute FFT-based features: dominant frequency and spectral centroid.

        Args:
            df: DataFrame of curves.
            sampling_rate: Sampling frequency (Hz).

        Returns:
            Single-row DataFrame with FFT features.

        Raises:
            ValueError: If `sampling_rate` is not positive.
        """
        if sampling_rate <= 0:
            raise ValueError("`sampling_rate` must be > 0.")
        if df.empty:
            raise ValueError("Input DataFrame for _fft_features is empty.")
        results = {}
        for col in df.columns:
            vals = df[col].to_numpy(dtype=float)
            n = vals.size
            if n < 2:
                results[f"{col}_fft_dominant_freq"] = 0.0
                results[f"{col}_fft_spectral_centroid"] = 0.0
            else:
                freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
                mags = np.abs(np.fft.rfft(vals - np.mean(vals)))
                idx = np.argmax(mags)
                total = mags.sum()
                centroid = float((freqs * mags).sum() /
                                 total) if total > 0 else 0.0
                results[f"{col}_fft_dominant_freq"] = float(freqs[idx])
                results[f"{col}_fft_spectral_centroid"] = centroid
        return pd.DataFrame([results])

    @staticmethod
    def _cross_corr_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cross-correlation max and Pearson corrcoef for each pair of columns.

        Args:
            df: DataFrame of curves.

        Returns:
            Single-row DataFrame with cross-correlation features.
        """
        if df.shape[1] < 2:
            return pd.DataFrame()
        results = {}
        cols = df.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1 = df[cols[i]].to_numpy(dtype=float)
                c2 = df[cols[j]].to_numpy(dtype=float)
                corr_full = np.correlate(
                    c1 - c1.mean(), c2 - c2.mean(), mode='full')
                results[f"{cols[i]}_vs_{cols[j]}_corr_max"] = float(
                    np.max(corr_full))
                results[f"{cols[i]}_vs_{cols[j]}_corrcoef"] = float(
                    np.corrcoef(c1, c2)[0, 1])
        return pd.DataFrame([results])

    @staticmethod
    def _sample_interval_features(
        df: pd.DataFrame,
        intervals: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Sample each column at specified fractional intervals.

        Args:
            df: DataFrame of curves.
            intervals: List of fractions in [0, 1]. Defaults to 0.0,0.1,...,1.0.

        Returns:
            Single-row DataFrame with interval samples.
        """
        if df.empty:
            raise ValueError(
                "Input DataFrame for _sample_interval_features is empty.")
        if intervals is None:
            intervals = [i / 10.0 for i in range(11)]
        if any((p < 0 or p > 1) for p in intervals):
            raise ValueError("All intervals must be between 0 and 1.")
        n = len(df)
        samples: Dict[str, float] = {}
        for col in df.columns:
            for frac in intervals:
                idx = min(int(round(frac * (n - 1))), n - 1)
                samples[f"{col}_p{int(frac*100)}"] = float(df[col].iloc[idx])
        return pd.DataFrame([samples])

    @staticmethod
    def generate_features(
        dataframe: pd.DataFrame,
        sampling_rate: float = 1.0
    ) -> pd.DataFrame:
        """
        Generate a full feature set for a single PF run.

        Args:
            dataframe: Raw run DataFrame containing at least:
                       'Dissipation', 'Resonance_Frequency', 'Relative_time'.
            sampling_rate: Sampling frequency of the data.
        Returns:
            DataFrame with one row of concatenated features.

        Raises:
            ValueError: If required columns are missing or inputs invalid.
        """
        feat_columns = ["Dissipation", "Resonance_Frequency"]
        dataframe = dataframe[feat_columns].copy()
        feats = pd.DataFrame()
        # feats["run_length"] = [len(dataframe)]

        n = len(dataframe)
        win = max(3, int(np.ceil(0.05 * n)))
        if win % 2 == 0:
            win += 1
        diss = dataframe["Dissipation"].values
        rf = dataframe["Resonance_Frequency"].values
        # diss = savgol_filter(
        #     diss, window_length=win, polyorder=2)
        # rf = savgol_filter(
        #     rf, window_length=win, polyorder=2)

        n_baseline = max(1, int(len(dataframe) * 0.05))
        baseline_diss = diss[:n_baseline].mean()
        baseline_rf = rf[:n_baseline].mean()
        diss_re = diss - baseline_diss
        rf_re = rf - baseline_rf
        rf_flipped = -rf_re
        num_points = PFDataProcessor.SAMPLE_FACTOR
        idxs = np.linspace(0, n-1, num=num_points, dtype=int)
        diss_sel = diss_re[idxs]
        rf_sel = rf_flipped[idxs]
        d_min, d_max = diss_sel.min(),    diss_sel.max()
        r_min, r_max = rf_sel.min(),      rf_sel.max()

        diss_scaled = (diss_sel - d_min) / (d_max - d_min)
        rf_scaled = (rf_sel - r_min) / (r_max - r_min)

        sampled_df = pd.DataFrame({
            "Dissipation":         diss_scaled,
            "Resonance_Frequency": rf_scaled
        })

        # plt.figure(figsize=(8, 4))
        # plt.plot(diss_scaled,
        #          linestyle='-', label="Diss [200 pts]")
        # plt.plot(rf_scaled,
        #          linestyle='-', label="RF   [200 pts]")
        # plt.legend(loc="upper right")
        # plt.xlabel("Sample # (evenly spaced)")
        # plt.ylabel("Scaled & rebaselined value")
        # plt.title("200-Point Representation of Each Curve")
        # plt.tight_layout()
        # plt.show()
        feats = pd.concat(
            [feats, PFDataProcessor._curve_stats(sampled_df)], axis=1)
        feats = pd.concat(
            [feats, PFDataProcessor._peak_features(sampled_df)], axis=1)
        feats = pd.concat(
            [feats, PFDataProcessor._curve_dynamics(sampled_df)], axis=1)
        feats = pd.concat([feats, PFDataProcessor._fft_features(
            sampled_df, sampling_rate)], axis=1)
        feats = pd.concat(
            [feats, PFDataProcessor._cross_corr_features(sampled_df)], axis=1)

        return feats


if __name__ == '__main__':
    test_dir = os.path.join('content', 'static', 'test')
    try:
        pairs = PFDataProcessor.load_content(test_dir)
        for data_f, poi_f in pairs:
            df = pd.read_csv(data_f)
            pois = pd.read_csv(poi_f, header=None).values
            feats = PFDataProcessor.generate_features(
                df, detected_poi1=pois[0]
            )
            print(feats.head())
    except Exception as e:
        Logger.e("Failed processing: %s", e)
