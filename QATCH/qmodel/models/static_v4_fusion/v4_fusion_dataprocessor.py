import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Dict
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import random
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.svm import OneClassSVM
import os
import tempfile
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import glob


class FusionDataprocessor:
    DROP = ["Date", "Time", "Ambient", "Peak Magnitude (RAW)", "Temperature"]
    BASELINE_WIN = 500
    ROLLING_WIN = 50

    @staticmethod
    def load_balanced_content(data_dir: str, num_datasets: int, num_viscosity_bins: int = 5,
                              remove_outliers: bool = True, outlier_method: str = 'iqr'):
        """
        Returns a balanced list of file pairs (data_file, poi_file) based on
        viscosity profile. Extreme outliers are removed before binning.

        Args:
            data_dir: Directory containing the data files
            num_datasets: Number of datasets to return
            num_viscosity_bins: Number of bins for viscosity stratification (default: 5)
            remove_outliers: Whether to remove extreme outliers (default: True)
            outlier_method: Method for outlier detection - 'iqr' or 'percentile' (default: 'iqr')

        Returns:
            List of tuples: [(data_file_path, poi_file_path), ...]
        """
        if not isinstance(data_dir, str) or not data_dir.strip():
            raise ValueError("data_dir must be a non-empty string.")
        if not os.path.isdir(data_dir):
            raise ValueError(f"Directory '{data_dir}' does not exist.")

        # Phase 1: Collect all file pairs with metadata
        file_metadata = []

        for root, _, files in tqdm(os.walk(data_dir), desc='Collecting file metadata'):
            for f in files:
                # Process CSV data files
                if f.endswith(".csv") and not f.endswith("_poi.csv") and not f.endswith("_lower.csv"):
                    data_file = os.path.join(root, f)
                    poi_file = os.path.join(
                        root, f.replace(".csv", "_poi.csv"))

                    # Validate POI file exists and has unique values
                    if not os.path.exists(poi_file):
                        continue

                    try:
                        poi_df = pd.read_csv(poi_file, header=None)
                        poi_values = poi_df.values.flatten()
                        if len(poi_values) != len(np.unique(poi_values)):
                            continue
                    except Exception:
                        continue

                    matches = glob.glob(os.path.join(root, "analyze-*.zip"))

                    if not matches:
                        continue

                    analyze_zip = matches[0]
                    if not os.path.exists(analyze_zip):
                        continue

                    # Extract metadata from ZIP
                    metadata = FusionDataprocessor._extract_metadata(
                        analyze_zip)
                    if metadata:
                        file_metadata.append({
                            'data_file': data_file,
                            'poi_file': poi_file,
                            **metadata
                        })

        if len(file_metadata) == 0:
            print("Warning: No valid file pairs found.")
            return []

        print(f"Found {len(file_metadata)} valid file pairs.")

        # Phase 2: Remove outliers
        if remove_outliers:
            file_metadata = FusionDataprocessor._remove_outliers(
                file_metadata, method=outlier_method)
            print(
                f"After removing outliers: {len(file_metadata)} file pairs remain.")

            if len(file_metadata) == 0:
                print("Warning: No file pairs remain after outlier removal.")
                return []

        # Phase 3: Balanced selection
        selected_pairs = FusionDataprocessor._balanced_selection(
            file_metadata, num_datasets, num_viscosity_bins)

        return selected_pairs

    @staticmethod
    def _extract_metadata(analyze_zip_file: str) -> dict:
        """Extract viscosity metadata from analysis ZIP file."""
        metadata = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                with zipfile.ZipFile(analyze_zip_file, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)

                # Extract viscosity data from CSV
                for extracted_file in os.listdir(tmpdir):
                    if extracted_file.endswith("_out.csv"):
                        csv_path = os.path.join(tmpdir, extracted_file)
                        df = pd.read_csv(csv_path)
                        if 'viscosity_avg' in df.columns and 'percent_error' in df.columns:
                            metadata['avg_viscosity'] = df['viscosity_avg'].mean()
                            metadata['avg_percent_error'] = df['percent_error'].mean()
                            break  # Found viscosity data, no need to continue

                # Only return if we have viscosity data
                if 'avg_viscosity' in metadata:
                    return metadata

            except Exception as e:
                pass

        return {}

    @staticmethod
    def _remove_outliers(file_metadata: list, method: str = 'iqr', iqr_multiplier: float = 1.5) -> list:
        """
        Remove extreme outliers from the dataset based on viscosity values.

        Args:
            file_metadata: List of file metadata dictionaries
            method: Outlier detection method ('iqr' or 'percentile')
            iqr_multiplier: Multiplier for IQR method (default: 1.5, use 3.0 for extreme outliers only)

        Returns:
            Filtered list of file metadata without outliers
        """
        if not file_metadata:
            return []

        viscosities = np.array([item['avg_viscosity']
                               for item in file_metadata])

        if method == 'iqr':
            # IQR (Interquartile Range) method
            q1 = np.percentile(viscosities, 25)
            q3 = np.percentile(viscosities, 99)
            iqr = q3 - q1

            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            print(
                f"\nOutlier detection (IQR method, multiplier={iqr_multiplier}):")
            print(f"  Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
            print(f"  Valid range: [{lower_bound:.2f}, {upper_bound:.2f}]")

        elif method == 'percentile':
            # Simple percentile-based method (remove top and bottom 5%)
            lower_bound = np.percentile(viscosities, 5)
            upper_bound = np.percentile(viscosities, 95)

            print(f"\nOutlier detection (percentile method):")
            print(f"  Valid range: [{lower_bound:.2f}, {upper_bound:.2f}]")

        # Filter outliers
        filtered = [
            item for item in file_metadata
            if lower_bound <= item['avg_viscosity'] <= upper_bound
        ]

        num_removed = len(file_metadata) - len(filtered)
        print(
            f"  Removed {num_removed} outliers ({num_removed/len(file_metadata)*100:.1f}%)")

        return filtered

    @staticmethod
    def _balanced_selection(file_metadata: list, num_datasets: int, num_viscosity_bins: int) -> list:
        """
        Perform stratified balanced selection across viscosity ranges.
        """
        # Stratify by viscosity
        bins = FusionDataprocessor._stratify_by_viscosity(
            file_metadata, num_viscosity_bins)

        print(f"\nViscosity distribution across {len(bins)} bins:")
        for bin_idx, bin_items in bins.items():
            visc_range = [item['avg_viscosity'] for item in bin_items]
            print(
                f"  Bin {bin_idx}: {len(bin_items)} samples "
                f"(viscosity: {min(visc_range):.2f} - {max(visc_range):.2f})")

        # Balanced sampling across viscosity bins
        selected = []
        samples_per_bin = num_datasets // len(bins)
        remainder = num_datasets % len(bins)

        for bin_idx, (bin_id, bin_items) in enumerate(bins.items()):
            # Allocate extra samples to first few bins
            n_samples = samples_per_bin + (1 if bin_idx < remainder else 0)
            n_samples = min(n_samples, len(bin_items))

            # Random sampling without replacement
            if n_samples > 0:
                sampled = np.random.choice(
                    len(bin_items),
                    size=n_samples,
                    replace=False
                )
                selected.extend([bin_items[i] for i in sampled])

        # Shuffle final selection
        np.random.shuffle(selected)

        print(f"\nSelected {len(selected)} samples across viscosity bins")

        # Return file pairs
        return [(item['data_file'], item['poi_file']) for item in selected]

    @staticmethod
    def _stratify_by_viscosity(items: list, num_bins: int) -> dict:
        """Bin items by viscosity into equal-width bins."""
        if not items:
            return {}

        viscosities = np.array([item['avg_viscosity'] for item in items])
        min_visc, max_visc = viscosities.min(), viscosities.max()

        # Handle edge case of identical viscosities
        if min_visc == max_visc:
            return {0: items}

        # Create bins
        bin_edges = np.linspace(min_visc, max_visc, num_bins + 1)
        bins = defaultdict(list)

        for item in items:
            visc = item['avg_viscosity']
            # Assign to bin (rightmost bin is inclusive on both ends)
            bin_idx = np.searchsorted(bin_edges[1:-1], visc, side='left')
            bins[bin_idx].append(item)

        return dict(bins)

    @staticmethod
    def load_content(data_dir: str, num_datasets: Union[int, float] = np.inf) -> List[Tuple[str, str]]:
        """Load dataset pairs from directory."""
        if not isinstance(data_dir, str) or not data_dir.strip():
            raise ValueError("data_dir must be a non-empty string.")
        if not os.path.isdir(data_dir):
            raise ValueError(f"Directory '{data_dir}' does not exist.")

        # Collect all valid datasets with their POI characteristics
        dataset_poi_info: Dict[str, Dict] = {}

        for root, _, files in tqdm(os.walk(data_dir), desc='Scanning files...'):
            for f in files:
                if f.endswith(".csv") and not f.endswith("_poi.csv") and not f.endswith("_lower.csv"):
                    poi_file = f.replace(".csv", "_poi.csv")
                    poi_path = os.path.join(root, poi_file)
                    if not os.path.exists(poi_path):
                        continue
                    try:
                        poi_df = pd.read_csv(poi_path, header=None)
                    except Exception:
                        continue

                    poi_values = poi_df.values.flatten()
                    if len(poi_values) != len(np.unique(poi_values)):
                        continue

                    # Store dataset info with POI characteristics
                    data_path = os.path.join(root, f)
                    dataset_poi_info[data_path] = {
                        'poi_path': poi_path,
                        'poi_positions': poi_values,
                        'poi_stats': {
                            'mean': np.mean(poi_values),
                            'std': np.std(poi_values),
                            'min': np.min(poi_values),
                            'max': np.max(poi_values),
                            'median': np.median(poi_values),
                            'count': len(poi_values)
                        }
                    }

        if not dataset_poi_info:
            return []

        loaded_content = [(path, info['poi_path'])
                          for path, info in dataset_poi_info.items()]
        random.shuffle(loaded_content)
        if num_datasets != np.inf:
            loaded_content = loaded_content[:int(num_datasets)]
        return loaded_content

    @staticmethod
    def compute_difference_curve(df: pd.DataFrame, difference_factor: int = 2) -> pd.Series:
        """Compute difference curve from resonance frequency and dissipation."""
        required_cols = ["Relative_time", "Resonance_Frequency", "Dissipation"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' is missing from DataFrame.")

        xs = df["Relative_time"].values

        # Vectorized search for indices
        i = np.searchsorted(xs, 0.5)
        j = np.searchsorted(xs, 2.5)
        if i == j:
            j = np.searchsorted(xs, xs[j] + 2.0)

        # Use numpy for faster computation
        avg_res_freq = df["Resonance_Frequency"].iloc[i:j].mean()
        avg_diss = df["Dissipation"].iloc[i:j].mean()

        ys_diss = (df["Dissipation"].values - avg_diss) * avg_res_freq / 2
        ys_freq = avg_res_freq - df["Resonance_Frequency"].values

        return pd.Series(ys_freq - difference_factor * ys_diss, index=df.index)

    @staticmethod
    def compute_time_aware_baseline(df: pd.DataFrame, baseline_duration: float = 3.0):
        """Compute baseline using time duration instead of sample count."""
        time = df["Relative_time"].values
        baseline_mask = time <= baseline_duration

        base_d = df["Dissipation"].values[baseline_mask].mean()
        base_rf = df["Resonance_Frequency"].values[baseline_mask].mean()

        return base_d, base_rf

    @staticmethod
    def weighted_smooth(values: np.ndarray, time: np.ndarray, window_size: float = 0.1) -> np.ndarray:
        """Apply weighted smoothing based on time proximity."""
        smoothed = np.zeros_like(values)
        half_window = window_size / 2
        quarter_window = window_size / 4

        for i, t in enumerate(time):
            mask = np.abs(time - t) <= half_window
            if np.sum(mask) > 0:
                distances = np.abs(time[mask] - t)
                weights = np.exp(-distances / quarter_window)
                weights /= weights.sum()
                smoothed[i] = np.average(values[mask], weights=weights)
            else:
                smoothed[i] = values[i]
        return smoothed

    @staticmethod
    def compute_ocsvm_score(shift_series: pd.Series) -> np.ndarray:
        """Compute One-Class SVM anomaly scores."""
        if shift_series.empty:
            raise ValueError("shift_series is empty.")

        X = shift_series.values.reshape(-1, 1)
        ocsvm = OneClassSVM(nu=0.05, kernel='rbf',
                            gamma='scale', shrinking=False)
        ocsvm.fit(X)

        # Get raw decision scores and flip so negative spikes become positive
        scores = -ocsvm.decision_function(X)
        # Baseline at zero
        scores = scores - np.min(scores)
        return scores

    @staticmethod
    def compute_DoG(series: pd.Series, sigma: float = 2) -> pd.Series:
        """Compute Difference of Gaussians (first derivative of Gaussian)."""
        if sigma <= 0:
            raise ValueError("sigma must be a positive number.")
        result = gaussian_filter1d(series.values, sigma=sigma, order=1)
        return pd.Series(result, index=series.index)

    @staticmethod
    def compute_rolling_baseline_and_shift(dog_series: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
        """Compute rolling baseline and shift from baseline."""
        if window <= 0:
            raise ValueError("window must be a positive integer.")
        baseline = dog_series.rolling(
            window=window, center=True, min_periods=1).median()
        shift = dog_series - baseline
        return baseline, shift

    @staticmethod
    def process_dog_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Process DoG features for specified columns."""
        for col in columns:
            # Compute DoG
            df[f'{col}_DoG'] = FusionDataprocessor.compute_DoG(df[col])

            # Compute baseline and shift
            baseline_window = max(3, int(np.ceil(0.05 * len(df))))
            df[f'{col}_DoG_baseline'], df[f'{col}_DoG_shift'] = \
                FusionDataprocessor.compute_rolling_baseline_and_shift(
                    df[f'{col}_DoG'], baseline_window
            )

            # Compute SVM scores
            df[f'{col}_DoG_SVM_Score'] = FusionDataprocessor.compute_ocsvm_score(
                df[f'{col}_DoG_shift']
            )

        return df

    @staticmethod
    def apply_baseline_correction(df: pd.DataFrame, use_time_aware: bool = True) -> pd.DataFrame:
        """Apply baseline correction to the dataframe."""
        if use_time_aware:
            baseline_duration = FusionDataprocessor.BASELINE_WIN * 0.008  # convert to seconds
            base_d, base_rf = FusionDataprocessor.compute_time_aware_baseline(
                df, baseline_duration)
        else:
            base_d = df["Dissipation"].iloc[:FusionDataprocessor.BASELINE_WIN].mean()
            base_rf = df["Resonance_Frequency"].iloc[:FusionDataprocessor.BASELINE_WIN].mean(
            )

        df["Dissipation"] = df["Dissipation"] - base_d
        df["Resonance_Frequency"] = -(df["Resonance_Frequency"] - base_rf)

        return df, base_d, base_rf

    @staticmethod
    def get_reg_features(df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for regression tasks."""
        df = df.copy()
        df = df.drop(columns=FusionDataprocessor.DROP)

        # Apply baseline correction (time-aware)
        df, base_d, base_rf = FusionDataprocessor.apply_baseline_correction(
            df, use_time_aware=True)

        # Compute difference with regularized data
        df["Difference"] = - \
            FusionDataprocessor.compute_difference_curve(
                df, difference_factor=2)

        # Apply weighted smoothing
        time_values = df["Relative_time"].values
        for col in ["Dissipation", "Resonance_Frequency", "Difference"]:
            df[col] = FusionDataprocessor.weighted_smooth(
                df[col].values, time_values, window_size=0.05
            )

        # Process DoG features for all three columns
        df = FusionDataprocessor.process_dog_features(
            df, ["Dissipation", "Resonance_Frequency", "Difference"]
        )

        return df

    @staticmethod
    def get_clf_features(df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for classification tasks."""
        df = df.copy()
        df = df.drop(columns=FusionDataprocessor.DROP)

        # Compute smoothing window
        smooth_win = int(0.005 * len(df["Relative_time"]))
        smooth_win = smooth_win + 1 if smooth_win % 2 == 0 else smooth_win
        smooth_win = max(2, smooth_win)
        polyorder = 3

        # Compute difference curve first
        df["Difference"] = FusionDataprocessor.compute_difference_curve(
            df, difference_factor=2)

        # Apply baseline correction (sample-based)
        base_d = df["Dissipation"].iloc[:FusionDataprocessor.BASELINE_WIN].mean()
        base_rf = df["Resonance_Frequency"].iloc[:FusionDataprocessor.BASELINE_WIN].mean()
        base_diff = df["Difference"].iloc[:FusionDataprocessor.BASELINE_WIN].mean()

        df["Dissipation"] = df["Dissipation"] - base_d
        df["Difference"] = df["Difference"] - base_diff
        df["Resonance_Frequency"] = -(df["Resonance_Frequency"] - base_rf)

        # Apply Savitzky-Golay smoothing
        for col in ["Difference", "Dissipation", "Resonance_Frequency"]:
            df[col] = savgol_filter(df[col], smooth_win, polyorder)

        # Process DoG features
        df = FusionDataprocessor.process_dog_features(
            df, ["Dissipation", "Resonance_Frequency", "Difference"]
        )

        # Compute additional classification features efficiently
        df = FusionDataprocessor._add_classification_features(df)

        # Fill NaN values
        df.fillna(0, inplace=True)

        return df

    @staticmethod
    def _add_classification_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add additional features for classification."""
        # Precompute common values
        dt = df['Relative_time'].diff().replace(0, np.nan)
        diss_values = df['Dissipation'].values
        rf_values = df['Resonance_Frequency'].values
        time_values = df['Relative_time'].values

        # Compute slopes
        df['Diss_slope'] = df['Dissipation'].diff() / dt
        df['RF_slope'] = df['Resonance_Frequency'].diff() / dt
        df[['Diss_slope', 'RF_slope']] = df[[
            'Diss_slope', 'RF_slope']].fillna(0)

        # Rolling aggregates - use numpy for efficiency
        roll_win = FusionDataprocessor.ROLLING_WIN

        # Dissipation rolling stats
        for stat_name, stat_func in [('mean', 'mean'), ('std', 'std'), ('min', 'min'), ('max', 'max')]:
            df[f'Diss_roll_{stat_name}'] = df['Dissipation'].rolling(
                window=roll_win, min_periods=1
            ).agg(stat_func)

        # Resonance Frequency rolling stats
        for stat_name, stat_func in [('mean', 'mean'), ('std', 'std'), ('min', 'min'), ('max', 'max')]:
            df[f'RF_roll_{stat_name}'] = df['Resonance_Frequency'].rolling(
                window=roll_win, min_periods=1
            ).agg(stat_func)

        # Fill NaN in std columns
        df['Diss_roll_std'] = df['Diss_roll_std'].fillna(0)
        df['RF_roll_std'] = df['RF_roll_std'].fillna(0)

        # Vectorized feature computations
        df['Diss_x_RF'] = diss_values * rf_values
        df['slope_DxRF'] = df['Diss_slope'] * df['RF_slope']
        df['rollmean_DxrollRF'] = df['Diss_roll_mean'] * df['RF_roll_mean']
        df['Diss_over_RF'] = diss_values / (rf_values + 1e-6)
        df['slope_ratio'] = df['Diss_slope'] / (df['RF_slope'].values + 1e-6)
        df['rollstd_ratio'] = df['Diss_roll_std'] * 0  # placeholder
        df['time_x_Diss'] = time_values * diss_values
        df['time_x_slope_sum'] = time_values * \
            (df['Diss_slope'] + df['RF_slope'])

        # Shifted features
        df['Diss_t_x_RF_t1'] = diss_values * np.roll(rf_values, 1)
        df['Diss_t1_x_RF_t'] = np.roll(diss_values, 1) * rf_values

        # Range and area features (computed once for entire window)
        diss_range = diss_values.max() - diss_values.min()
        rf_range = rf_values.max() - rf_values.min()
        df['range_Dx_range_RF'] = diss_range * rf_range
        df['area_Dx_area_RF'] = diss_values.sum() * rf_values.sum()

        return df


if __name__ == "__main__":
    data_dir = "content/train"
    num_datasets = 10
    file_pairs = FusionDataprocessor.load_content(
        data_dir, num_datasets=num_datasets)

    for data_file, _ in tqdm(file_pairs):
        df = pd.read_csv(data_file, engine="pyarrow")
        df = FusionDataprocessor.get_reg_features(df)
        plt.figure()
        plt.plot(df["Difference"])
        plt.show()
