"""
qmodel_v4_dataprocessor.py

Provides the QModelDataProcessor for generating features and santizing input to QModel V4.x

Author: 
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date: 
    2025-08-19

Version: 
    QModel.Ver4.1
"""
import pandas as pd
from typing import Union, Tuple, List
from tqdm import tqdm
import numpy as np
import os
import random
from sklearn.svm import OneClassSVM
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


class DataProcessorV4:
    """Columns to ignore from incomming data."""
    DROP = ["Date", "Time", "Ambient", "Peak Magnitude (RAW)", "Temperature"]

    """Default baseline window size."""
    BASELINE_WIN = 500

    """Default rolling window size."""
    ROLLING_WIN = 50

    @staticmethod
    def load_content(data_dir: str, num_datasets: Union[int, float] = np.inf) -> List[Tuple[str, str]]:
        """Load CSV files and their corresponding POI annotation files from a directory.

        This method recursively scans `data_dir` for CSV files, ensuring each has a
        matching POI file (`*_poi.csv`) and no duplicate POI values. Files ending with
        `_poi.csv` or `_lower.csv` are ignored. The resulting valid pairs are shuffled
        and optionally truncated to `num_datasets`.

        Args:
            data_dir (str): Path to the directory containing CSV files and POI files.
            num_datasets (Union[int, float], optional): Maximum number of dataset pairs
                to return. Defaults to np.inf (all datasets).

        Returns:
            List[Tuple[str, str]]: List of tuples `(data_csv_path, poi_csv_path)` containing
                the full paths of valid data files and their corresponding POI annotation files.

        Raises:
            ValueError: If `data_dir` is not a non-empty string or does not exist.
        """
        if not isinstance(data_dir, str) or not data_dir.strip():
            raise ValueError("data_dir must be a non-empty string.")
        if not os.path.isdir(data_dir):
            raise ValueError(f"Directory '{data_dir}' does not exist.")
        loaded_content: List[Tuple[str, str]] = []

        for root, _, files in tqdm(os.walk(data_dir), desc='Loading files...'):
            for f in files:
                if f.endswith(".csv") and not f.endswith("_poi.csv") and not f.endswith("_lower.csv"):
                    poi_file = f.replace(".csv", "_poi.csv")
                    poi_path = os.path.join(root, poi_file)
                    if not os.path.exists(poi_path):
                        continue
                    try:
                        poi_df = pd.read_csv(poi_path, header=None)
                    except Exception as e:
                        continue
                    poi_values = poi_df.values
                    if len(poi_values) != len(np.unique(poi_values)):
                        continue
                    else:
                        loaded_content.append(
                            (os.path.join(root, f), poi_path))
        random.shuffle(loaded_content)
        if num_datasets == np.inf:
            return loaded_content
        return loaded_content[:int(num_datasets)]

    @staticmethod
    def gen_features(df: pd.DataFrame):
        """Generate additional features for POI detection from a DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame containing raw time-series data. 
                Expected to have numerical columns suitable for feature extraction.

        Returns:
            (implementation-specific): Returns a DataFrame with additional features
                computed for each row, suitable for POI prediction.

        Raises:
            ValueError: If the input DataFrame is empty or does not contain required columns.
        """
        def compute_ocsvm_score(shift_series: pd.Series):
            if shift_series.empty:
                raise ValueError("shift_series is empty.")
            X = shift_series.values.reshape(-1, 1)
            ocsvm = OneClassSVM(nu=0.05, kernel='rbf',
                                gamma='scale', shrinking=False)
            ocsvm.fit(X)
            # get raw decision scores and flip so negative spikes become positive
            scores = ocsvm.decision_function(X)
            scores = -scores
            # baseline at zero
            scores = scores - np.min(scores)
            return scores

        def compute_DoG(df: pd.DataFrame, col: str, sigma: float = 2):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
            if sigma <= 0:
                raise ValueError("sigma must be a positive number.")
            result = gaussian_filter1d(df[col], sigma=sigma, order=1)
            return pd.Series(result, index=df.index)

        def compute_rolling_baseline_and_shift(dog_series: pd.Series, window: int):
            if window <= 0:
                raise ValueError("window must be a positive integer.")
            baseline = dog_series.rolling(
                window=window, center=True, min_periods=1).median()
            shift = dog_series - baseline
            return baseline, shift

        def compute_difference_curve(df: pd.DataFrame, difference_factor: int = 2) -> pd.Series:
            required_cols = ["Relative_time",
                             "Resonance_Frequency", "Dissipation"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(
                        f"Column '{col}' is missing from DataFrame.")

            xs = df["Relative_time"]
            i = next((x for x, t in enumerate(xs) if t > 0.5), None)
            j = next((x for x, t in enumerate(xs) if t > 2.5), None)

            avg_res_freq = df["Resonance_Frequency"].iloc[i:j].mean()
            avg_diss = df["Dissipation"].iloc[i:j].mean()
            ys_diss = (df["Dissipation"] - avg_diss) * avg_res_freq / 2
            ys_freq = avg_res_freq - df["Resonance_Frequency"]
            return ys_freq - difference_factor * ys_diss

        df = df.copy()
        df = df.drop(columns=DataProcessorV4.DROP)
        smooth_win = int(
            0.005 * len(df["Relative_time"].values))
        if smooth_win % 2 == 0:
            smooth_win += 1
        if smooth_win <= 1:
            smooth_win = 2
        polyorder = 3
        # Rebaseline
        df["Difference"] = compute_difference_curve(df, difference_factor=2)
        base_d = df["Dissipation"].iloc[:DataProcessorV4.BASELINE_WIN].mean()
        base_rf = df["Resonance_Frequency"].iloc[:DataProcessorV4.BASELINE_WIN].mean()
        base_diff = df["Difference"].iloc[:DataProcessorV4.BASELINE_WIN].mean()
        df["Dissipation"] = df["Dissipation"] - base_d
        df["Difference"] = df["Difference"] - base_diff
        df["Resonance_Frequency"] = -(df["Resonance_Frequency"] - base_rf)

        # Smooth and compute difference
        df["Difference"] = savgol_filter(
            df["Difference"], smooth_win, polyorder)
        df["Dissipation"] = savgol_filter(
            df["Dissipation"], smooth_win, polyorder)
        df["Resonance_Frequency"] = savgol_filter(
            df["Resonance_Frequency"], smooth_win, polyorder)

        # `Dissipation` DoG processing
        df['Dissipation_DoG'] = compute_DoG(
            df, col='Dissipation')
        baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        df['Dissipation_DoG_baseline'], df['Dissipation_DoG_shift'] = compute_rolling_baseline_and_shift(
            df['Dissipation_DoG'], baseline_window
        )
        df['Dissipation_DoG_SVM_Score'] = compute_ocsvm_score(
            df['Dissipation_DoG_shift'])

        # `Resonance_Frequency` DoG processing
        df['Resonance_Frequency_DoG'] = compute_DoG(
            df, col='Resonance_Frequency')
        baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        df['Resonance_Frequency_DoG_baseline'], df['Resonance_Frequency_DoG_shift'] = compute_rolling_baseline_and_shift(
            df['Resonance_Frequency_DoG'], baseline_window
        )
        df['Resonance_Frequency_DoG_SVM_Score'] = compute_ocsvm_score(
            df['Resonance_Frequency_DoG_shift'])

        # `Difference` DoG processing
        df['Difference_DoG'] = compute_DoG(
            df, col='Difference')
        baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        df['Difference_DoG_baseline'], df['Difference_DoG_shift'] = compute_rolling_baseline_and_shift(
            df['Difference_DoG'], baseline_window
        )
        df['Difference_DoG_SVM_Score'] = compute_ocsvm_score(
            df['Difference_DoG_shift'])

        # 1) Compute slopes as Δy/Δt
        dt = df['Relative_time'].diff().replace(0, np.nan)
        df['Diss_slope'] = df['Dissipation'].diff() / dt
        df['RF_slope'] = df['Resonance_Frequency'].diff() / dt
        df[['Diss_slope', 'RF_slope']] = df[[
            'Diss_slope', 'RF_slope']].fillna(0)
        df['Diss_roll_mean'] = df['Dissipation'].rolling(
            window=DataProcessorV4.ROLLING_WIN, min_periods=1).mean()
        df['Diss_roll_std'] = df['Dissipation'].rolling(
            window=DataProcessorV4.ROLLING_WIN, min_periods=1).std().fillna(0)
        df['Diss_roll_min'] = df['Dissipation'].rolling(
            window=DataProcessorV4.ROLLING_WIN, min_periods=1).min()
        df['Diss_roll_max'] = df['Dissipation'].rolling(
            window=DataProcessorV4.ROLLING_WIN, min_periods=1).max()

        # 3) Rolling aggregates for Resonance_Frequency
        df['RF_roll_mean'] = df['Resonance_Frequency'].rolling(
            window=DataProcessorV4.ROLLING_WIN, min_periods=1).mean()
        df['RF_roll_std'] = df['Resonance_Frequency'].rolling(
            window=DataProcessorV4.ROLLING_WIN, min_periods=1).std().fillna(0)
        df['RF_roll_min'] = df['Resonance_Frequency'].rolling(
            window=DataProcessorV4.ROLLING_WIN, min_periods=1).min()
        df['RF_roll_max'] = df['Resonance_Frequency'].rolling(
            window=DataProcessorV4.ROLLING_WIN, min_periods=1).max()

        df['Diss_x_RF'] = df['Dissipation'] * df['Resonance_Frequency']
        df['slope_DxRF'] = df['Diss_slope'] * df['RF_slope']
        df['rollmean_DxrollRF'] = df['Diss_roll_mean'] * df['RF_roll_mean']
        df['Diss_over_RF'] = df['Dissipation'] / \
            (df['Resonance_Frequency'] + 1e-6)
        df['slope_ratio'] = df['Diss_slope'] / (df['RF_slope'] + 1e-6)
        df['rollstd_ratio'] = df['Diss_roll_std'] * \
            0  # placeholder if you compute RF_roll_std
        df['time_x_Diss'] = df['Relative_time'] * df['Dissipation']
        df['time_x_slope_sum'] = df['Relative_time'] * \
            (df['Diss_slope'] + df['RF_slope'])
        df['Diss_t_x_RF_t1'] = df['Dissipation'] * \
            df['Resonance_Frequency'].shift(1)
        df['Diss_t1_x_RF_t'] = df['Dissipation'].shift(
            1) * df['Resonance_Frequency']

        window = df  # or the slice of df you’re summarizing
        df['range_Dx_range_RF'] = (window['Dissipation'].max() - window['Dissipation'].min()) \
            * (window['Resonance_Frequency'].max() - window['Resonance_Frequency'].min())
        # if you compute area under the curve:
        df['area_Dx_area_RF'] = (
            window['Dissipation'].sum()) * (window['Resonance_Frequency'].sum())

        df.fillna(0, inplace=True)

        return df
