#! q_forecast_data_processor
"""
This module contains the QForecastDataProcessor class that provides methods
for processing forecast data. It includes functionality to convert raw worker
buffers to a DataFrame, load and balance datasets, compute various signal
features (e.g., difference of Gaussians, spectral features), and generate features
for forecasting purposes.
Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    04-04-2025

Version:
    V2
"""

import os
import random
import logging
from typing import Any, List, Tuple, Union, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch, savgol_filter
from scipy.stats import skew, kurtosis, zscore
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from QATCH.common.logger import Logger

# Global constants
TAG: List[str] = ['QForecastDataProcessor']
TARGET: str = "Fill"
DOWNSAMPLE_FACTOR: int = 5
SPIN_UP_TIME: Tuple[float, float] = (1.2, 1.4)
BASELINE_WINDOW: int = 100
PLOTTING: bool = False


class QForecastDataProcessor:
    """
    Class for processing forecast data.

    This class provides static methods for converting raw worker data into a DataFrame,
    loading and balancing datasets from a directory, and computing various features and statistics
    from the data, including signal smoothing, spectral analysis, and anomaly detection.
    """

    @staticmethod
    def convert_to_dataframe(worker: Any) -> pd.DataFrame:
        """
        Convert raw buffer data from a worker into a pandas DataFrame.

        Retrieves the relative time, resonance frequency, and dissipation buffers from the worker,
        truncates them to the same length, and constructs a DataFrame with the columns:
          - 'Relative_time'
          - 'Resonance_Frequency'
          - 'Dissipation'

        Args:
            worker (Any): A worker object that provides buffer data through methods 
                          `get_t1_buffer(index: int)`, `get_d1_buffer(index: int)`, and 
                          `get_d2_buffer(index: int)`.

        Returns:
            pd.DataFrame: A DataFrame with the required columns.

        Raises:
            ValueError: If the worker does not have the required buffer methods.
        """
        required_methods = ['get_t1_buffer', 'get_d1_buffer', 'get_d2_buffer']
        for method in required_methods:
            if not hasattr(worker, method):
                raise ValueError(
                    f"Worker is missing required method: {method}")

        relative_time = worker.get_t1_buffer(0)
        resonance_frequency = worker.get_d1_buffer(0)
        dissipation = worker.get_d2_buffer(0)

        # Determine the minimum length among buffers
        min_length = min(len(relative_time), len(
            resonance_frequency), len(dissipation))
        if min_length == 0:
            raise ValueError("One or more buffers are empty.")

        # Truncate buffers to the minimum length
        relative_time_truncated = relative_time[:min_length]
        resonance_frequency_truncated = resonance_frequency[:min_length]
        dissipation_truncated = dissipation[:min_length]

        df = pd.DataFrame({
            'Relative_time': relative_time_truncated,
            'Resonance_Frequency': resonance_frequency_truncated,
            'Dissipation': dissipation_truncated
        })
        return df

    @staticmethod
    def load_content(data_dir: str, num_datasets: Union[int, float] = np.inf, column: str = 'Dissipation') -> List[Tuple[str, str]]:
        """
        Load content from CSV files in the specified directory.

        Walks through the directory and collects file paths for valid CSV files that have a corresponding
        POI file (ending with '_poi.csv'). Files ending with "_lower.csv" are ignored.

        Args:
            data_dir (str): Directory path to search for files.
            num_datasets (int or float, optional): Number of datasets to return. Defaults to np.inf.
            column (str, optional): Column name to consider in the CSV files (unused in current implementation).

        Returns:
            List[Tuple[str, str]]: A list of tuples, each containing the data file path and the corresponding POI file path.

        Raises:
            ValueError: If data_dir is not a non-empty string or does not exist.
        """
        if not isinstance(data_dir, str) or not data_dir.strip():
            raise ValueError("data_dir must be a non-empty string.")
        if not os.path.isdir(data_dir):
            raise ValueError(f"Directory '{data_dir}' does not exist.")

        Logger.i(f"Loading content from {data_dir}")
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
                        Logger.w(f"Error reading POI file {poi_path}: {e}")
                        continue
                    poi_values = poi_df.values
                    if len(poi_values) != len(np.unique(poi_values)):
                        Logger.w('POI file contains duplicate indices.')
                    else:
                        loaded_content.append(
                            (os.path.join(root, f), poi_path))
        random.shuffle(loaded_content)
        if num_datasets == np.inf:
            return loaded_content
        return loaded_content[:int(num_datasets)]

    @staticmethod
    def load_balanced_content(data_dir: str, num_datasets: Union[int, float] = np.inf, clusters: int = 3, opt: bool = False) -> Union[Tuple[List[Tuple[str, str]], int], List]:
        """
        Load and balance content by clustering feature vectors extracted from CSV files.

        The function loads content using `load_content`, computes feature vectors for each valid file,
        and uses KMeans clustering to balance the datasets. Optionally, the optimal number of clusters is computed.

        Args:
            data_dir (str): Directory path to search for files.
            num_datasets (int or float, optional): Desired number of datasets. Defaults to np.inf.
            clusters (int, optional): Number of clusters to use if not using optimal clustering. Defaults to 3.
            opt (bool, optional): If True, compute the optimal number of clusters. Defaults to False.

        Returns:
            Union[Tuple[List[Tuple[str, str]], int], List]: A tuple containing the balanced sample and the number of clusters,
                                                              or an empty list if no content is loaded.
        """
        loaded_content = QForecastDataProcessor.load_content(
            data_dir, num_datasets=np.inf)
        if not loaded_content:
            return []

        features = []
        valid_content = []
        for data_file, poi_file in loaded_content:
            try:
                df = pd.read_csv(data_file)
                poi_df = pd.read_csv(poi_file, header=None)
                if "Dissipation" not in df.columns or "Resonance_Frequency" not in df.columns:
                    continue
                poi_mean = np.mean(poi_df.values)
                poi_std = np.std(poi_df.values)
                mean_diss = df["Dissipation"].mean()
                std_diss = df["Dissipation"].std()
                mean_freq = df["Resonance_Frequency"].mean()
                std_freq = df["Resonance_Frequency"].std()
                max_r_time = df['Relative_time'].max(
                ) if 'Relative_time' in df.columns else 0
                feature_vec = [poi_mean, poi_std, mean_diss,
                               std_diss, mean_freq, std_freq, max_r_time]
                features.append(feature_vec)
                valid_content.append((data_file, poi_file))
            except Exception as e:
                Logger.w(f"Could not read file {data_file}: {e}")
                continue

        if not features:
            return []
        features_arr = np.array(features)
        if opt:
            n_clusters, silhouette_scores = QForecastDataProcessor.compute_optimal_clusters(
                features_arr, min_clusters=2, max_clusters=50)
            Logger.i(
                f"Optimal clusters: {n_clusters} with score {silhouette_scores}.")
        else:
            n_clusters = min(clusters, len(features_arr))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features_arr)

        cluster_dict: Dict[int, List[Tuple[str, str]]] = {}
        for label, file_pair in zip(labels, valid_content):
            cluster_dict.setdefault(label, []).append(file_pair)

        balanced_sample: List[Tuple[str, str]] = []
        if num_datasets == np.inf:
            min_count = min(len(files) for files in cluster_dict.values())
            for file_list in cluster_dict.values():
                balanced_sample.extend(random.sample(file_list, min_count))
        else:
            per_cluster = int(num_datasets) // n_clusters
            for file_list in cluster_dict.values():
                if len(file_list) <= per_cluster:
                    balanced_sample.extend(file_list)
                else:
                    balanced_sample.extend(
                        random.sample(file_list, per_cluster))
            remaining = int(num_datasets) - len(balanced_sample)
            if remaining > 0:
                extra_files = [fp for files in cluster_dict.values()
                               for fp in files if fp not in balanced_sample]
                if extra_files:
                    balanced_sample.extend(random.sample(
                        extra_files, min(remaining, len(extra_files))))
        random.shuffle(balanced_sample)

        if PLOTTING:
            plt.figure(figsize=(10, 6))
            plt.scatter(features_arr[:, 0], features_arr[:, 1],
                        c=labels, cmap='viridis', s=50, label='All files')
            selected_indices = [i for i, pair in enumerate(
                valid_content) if pair in balanced_sample]
            plt.scatter(features_arr[selected_indices, 0], features_arr[selected_indices, 1],
                        facecolors='none', edgecolors='red', s=100, label='Selected balanced sample')
            centers = kmeans.cluster_centers_
            plt.scatter(centers[:, 0], centers[:, 1], c='black',
                        s=200, marker='X', label='Cluster Centers')
            plt.xlabel("Mean Dissipation")
            plt.ylabel("Std. Dissipation")
            plt.title("Clustering of Data Files and Selected Balanced Sample")
            plt.legend()
            plt.grid(True)
            plt.show()

        return balanced_sample, n_clusters

    @staticmethod
    def compute_optimal_clusters(features: np.ndarray, min_clusters: int = 2, max_clusters: int = 10) -> Tuple[int, Dict[int, float]]:
        """
        Compute the optimal number of clusters for the given feature set using silhouette scores.

        Args:
            features (np.ndarray): Array of feature vectors.
            min_clusters (int, optional): Minimum number of clusters to consider. Defaults to 2.
            max_clusters (int, optional): Maximum number of clusters to consider. Defaults to 10.

        Returns:
            Tuple[int, Dict[int, float]]: The optimal number of clusters and a dictionary mapping cluster numbers to silhouette scores.

        Raises:
            ValueError: If features is empty.
        """
        if features.size == 0:
            raise ValueError("Features array is empty.")

        scores: Dict[int, float] = {}
        best_k: int = min_clusters
        best_score: float = -1.0
        max_possible = min(max_clusters, features.shape[0])
        for k in range(min_clusters, max_possible + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k

        if PLOTTING:
            plt.figure(figsize=(8, 4))
            plt.plot(list(scores.keys()), list(scores.values()), marker='o')
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel("Average Silhouette Score")
            plt.title("Silhouette Score vs. Number of Clusters")
            plt.grid(True)
            plt.show()

        return best_k, scores

    @staticmethod
    def find_sampling_shift(df: pd.DataFrame) -> int:
        """
        Find the index where a significant shift in sampling occurs.

        Computes the difference of the "Relative_time" column, compares it against a threshold,
        and returns the first index where the difference exceeds the threshold.

        Args:
            df (pd.DataFrame): Input DataFrame that must contain the "Relative_time" column.

        Returns:
            int: The index of the first significant change, or -1 if none is found.

        Raises:
            ValueError: If "Relative_time" column is missing.
        """
        if "Relative_time" not in df.columns:
            raise ValueError(
                "Input DataFrame must contain 'Relative_time' column.")

        time_df = pd.DataFrame()
        time_df["Delta"] = df["Relative_time"].diff()
        threshold = 0.032
        rolling_avg = time_df["Delta"].expanding(min_periods=2).mean()
        time_df["Significant_change"] = (
            time_df["Delta"] - rolling_avg).abs() > threshold
        change_indices = time_df.index[time_df["Significant_change"]].tolist()
        return change_indices[0] if change_indices else -1

    @staticmethod
    def compute_DoG(df: pd.DataFrame, col: str, sigma: float = 2) -> pd.Series:
        """
        Compute the Difference of Gaussians (DoG) for a specified column in the DataFrame.

        Applies a Gaussian filter to compute the first derivative of the specified column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            col (str): Column name on which to compute the DoG.
            sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to 2.

        Returns:
            pd.Series: Series containing the computed DoG.

        Raises:
            ValueError: If the column is not in the DataFrame or if sigma is non-positive.
        """
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if sigma <= 0:
            raise ValueError("sigma must be a positive number.")
        result = gaussian_filter1d(df[col], sigma=sigma, order=1)
        return pd.Series(result, index=df.index)

    @staticmethod
    def compute_rolling_baseline_and_shift(dog_series: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
        """
        Compute the rolling baseline and shift of a DoG series.

        Args:
            dog_series (pd.Series): The input DoG series.
            window (int): Window size for the rolling median.

        Returns:
            Tuple[pd.Series, pd.Series]: A tuple containing the rolling baseline and the difference (shift).

        Raises:
            ValueError: If window is not a positive integer.
        """
        if window <= 0:
            raise ValueError("window must be a positive integer.")
        baseline = dog_series.rolling(
            window=window, center=True, min_periods=1).median()
        shift = dog_series - baseline
        return baseline, shift

    @staticmethod
    def compute_ocsvm_score(shift_series: pd.Series) -> np.ndarray:
        """
        Compute anomaly scores using One-Class SVM on a shifted series.

        Args:
            shift_series (pd.Series): The series on which to compute the anomaly score.

        Returns:
            np.ndarray: Array of anomaly scores from the One-Class SVM.

        Raises:
            ValueError: If shift_series is empty.
        """
        if shift_series.empty:
            raise ValueError("shift_series is empty.")
        X = shift_series.values.reshape(-1, 1)
        ocsvm = OneClassSVM(nu=0.05, kernel='rbf',
                            gamma='scale', shrinking=False)
        ocsvm.fit(X)
        return ocsvm.decision_function(X)

    @staticmethod
    def compute_spectral_features(signal: np.ndarray, relative_time: pd.Series) -> Tuple[float, float]:
        """
        Compute spectral features for a signal using Welch's method.

        Calculates the spectral entropy and dominant frequency.

        Args:
            signal (np.ndarray): 1D array representing the signal.
            relative_time (pd.Series): Series representing relative time values.

        Returns:
            Tuple[float, float]: A tuple with spectral entropy and the dominant frequency.

        Raises:
            ValueError: If signal is empty or if relative_time does not have enough data.
        """
        if signal.size == 0:
            raise ValueError("Signal array is empty.")
        if relative_time.empty or len(relative_time) < 2:
            raise ValueError("Insufficient relative_time data.")
        fs = 1 / np.mean(np.diff(relative_time))
        f, Pxx = welch(signal, fs=fs, nperseg=min(256, len(signal)))
        if np.sum(Pxx) == 0:
            raise ValueError(
                "Power spectral density sum is zero, cannot compute features.")
        psd_norm = Pxx / np.sum(Pxx)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        dominant_frequency = f[np.argmax(Pxx)]
        return spectral_entropy, dominant_frequency

    @staticmethod
    def smooth_signal(signal: np.ndarray) -> np.ndarray:
        """
        Smooth a signal using the Savitzky-Golay filter.

        Args:
            signal (np.ndarray): Input 1D signal array.

        Returns:
            np.ndarray: Smoothed signal.
        """
        if len(signal) < 2:
            raise ValueError("Signal array is too short to smooth.")
        window_size = min(11, len(signal) - 1) if len(signal) > 11 else 5
        return savgol_filter(signal, window_size, polyorder=2)

    @staticmethod
    def compute_rolling_statistics(signal: pd.Series, window: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute rolling statistics (skewness, kurtosis, and standard deviation) for a signal.

        Args:
            signal (pd.Series): The input signal as a pandas Series.
            window (int): Rolling window size.

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: Series for skewness, kurtosis, and variability.

        Raises:
            ValueError: If window is not positive.
        """
        if window <= 0:
            raise ValueError("window must be a positive integer.")
        skew_series = signal.rolling(
            window, min_periods=1).apply(skew, raw=True)
        kurtosis_series = signal.rolling(
            window, min_periods=1).apply(kurtosis, raw=True)
        variability_series = signal.rolling(window, min_periods=1).std()
        return skew_series, kurtosis_series, variability_series

    @staticmethod
    def compute_anomaly_zscore(signal: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Compute the z-score for anomaly detection and flag anomalies.

        Args:
            signal (pd.Series): Input signal.

        Returns:
            Tuple[pd.Series, pd.Series]: A tuple containing the z-scores and anomaly flags (binary).
        """
        z_scores = pd.Series(
            zscore(signal, nan_policy="omit"), index=signal.index)
        anomaly = (np.abs(z_scores) > 2).astype(int)
        return z_scores, anomaly

    @staticmethod
    def cluster_anomalies(anomaly_flags: pd.Series) -> pd.Series:
        """
        Cluster anomalies using DBSCAN and flag the first occurrence in each cluster.

        Args:
            anomaly_flags (pd.Series): Series with binary anomaly flags.

        Returns:
            pd.Series: Series with clustered anomaly flags.
        """
        if anomaly_flags.empty:
            return anomaly_flags
        anomaly_indices = anomaly_flags[anomaly_flags == 1].index
        if len(anomaly_indices) > 0:
            dbscan = DBSCAN(eps=5, min_samples=2)
            labels = dbscan.fit_predict(anomaly_indices.values.reshape(-1, 1))
            clustered_anomalies = np.zeros(len(anomaly_flags))
            for cluster in set(labels):
                if cluster != -1:
                    first_index = anomaly_indices[labels == cluster][0]
                    clustered_anomalies[first_index] = 1
            return pd.Series(clustered_anomalies, index=anomaly_flags.index)
        return anomaly_flags

    @staticmethod
    def compute_difference_curve(df: pd.DataFrame) -> pd.Series:
        """
        Compute the difference curve from the DataFrame.

        Searches for indices in 'Relative_time' where the time exceeds given thresholds,
        then computes a difference curve using 'Resonance_Frequency' and 'Dissipation'.

        Args:
            df (pd.DataFrame): Input DataFrame that must contain 'Relative_time',
                               'Resonance_Frequency', and 'Dissipation' columns.

        Returns:
            pd.Series: The computed difference curve.

        Raises:
            ValueError: If required columns are missing or if thresholds cannot be determined.
        """
        required_cols = ["Relative_time", "Resonance_Frequency", "Dissipation"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' is missing from DataFrame.")

        xs = df["Relative_time"]
        i = next((x for x, t in enumerate(xs) if t > 0.5), None)
        j = next((x for x, t in enumerate(xs) if t > 2.5), None)

        if i is None or j is None:
            raise ValueError(
                "Unable to determine valid indices for computing difference curve.")
        if j <= i:
            raise ValueError("Invalid indices: 'j' must be greater than 'i'.")

        avg_res_freq = df["Resonance_Frequency"].iloc[i:j].mean()
        avg_diss = df["Dissipation"].iloc[i:j].mean()
        ys_diss = (df["Dissipation"] - avg_diss) * avg_res_freq / 2
        ys_freq = avg_res_freq - df["Resonance_Frequency"]
        difference_factor = 3
        return ys_freq - difference_factor * ys_diss

    @staticmethod
    def generate_features(df: pd.DataFrame, live: bool = True, start_idx: int = -1, start_time: float = -1) -> pd.DataFrame:
        """
        Generate features from the input DataFrame for forecasting.

        Depending on whether the data is live or not, required columns are checked. The method
        computes a difference curve, smooths signals, applies DoG, and computes additional features.

        Args:
            df (pd.DataFrame): Input DataFrame.
            live (bool, optional): Flag indicating live data processing. Defaults to True.
            start_idx (int, optional): Start index for feature generation. Defaults to -1.
            start_time (float, optional): Start time for feature generation. Defaults to -1.

        Returns:
            pd.DataFrame: DataFrame with generated features.

        Raises:
            ValueError: If the input DataFrame is empty or missing required columns.
        """
        if live:
            required_columns = ["Dissipation",
                                "Resonance_Frequency", "Relative_time"]
        else:
            required_columns = ["Dissipation",
                                "Resonance_Frequency", "Relative_time", "Fill"]

        if df.empty or not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"Input DataFrame must contain the following columns: {required_columns}")

        df = df[required_columns].copy()
        if df.empty:
            raise ValueError(
                "DataFrame is empty after selecting required columns.")

        slice_loc = int(len(df) * 0.01)
        if slice_loc == 0 or len(df) < 2 * slice_loc:
            raise ValueError(
                "DataFrame is too small to compute baseline adjustments.")

        df["Difference"] = QForecastDataProcessor.compute_difference_curve(df)
        df["Dissipation"] -= np.average(
            df["Dissipation"].iloc[slice_loc:2 * slice_loc])
        df["Resonance_Frequency"] -= np.average(
            df["Resonance_Frequency"].iloc[slice_loc:2 * slice_loc])
        df["Resonance_Frequency"] = df["Resonance_Frequency"].values * -1

        window_length = 100
        polyorder = 3
        df["Difference_smooth"] = savgol_filter(
            df["Difference"], window_length, polyorder)
        df["Dissipation_smooth"] = savgol_filter(
            df["Dissipation"], window_length, polyorder)
        df["Resonance_Frequency_smooth"] = savgol_filter(
            df["Resonance_Frequency"], window_length, polyorder)

        # Compute Time_since_start feature.
        if start_idx == -1 or start_time == -1:
            df["Time_since_start"] = -1
        else:
            df["Time_since_start"] = np.where(df.index >= start_idx,
                                              df["Relative_time"] - start_time,
                                              -1)
        # Process Dissipation using DoG
        df['Dissipation_DoG'] = QForecastDataProcessor.compute_DoG(
            df, col='Dissipation')
        # Process Resonance Frequency using DoG
        df['Resonance_Frequency_DoG'] = QForecastDataProcessor.compute_DoG(
            df, col='Resonance_Frequency')
        # Process Difference using DoG
        df['Difference_DoG'] = QForecastDataProcessor.compute_DoG(
            df, col='Difference')

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        return df

    @staticmethod
    def process_fill(poi_file: str, length_df: int) -> pd.DataFrame:
        """
        Process fill information from a POI file and generate a label DataFrame.

        Reads the POI CSV file to obtain fill positions and creates a DataFrame of labels for the data,
        based on specified fill positions.

        Args:
            poi_file (str): File path for the POI CSV file.
            length_df (int): The length of the DataFrame that will be labeled.

        Returns:
            pd.DataFrame: DataFrame containing fill labels.

        Raises:
            ValueError: If poi_file is not a valid non-empty string or if the POI file does not contain enough indices.
        """
        if not isinstance(poi_file, str) or not poi_file.strip():
            raise ValueError("poi_file must be a non-empty string.")
        if not os.path.exists(poi_file):
            raise ValueError(f"POI file '{poi_file}' does not exist.")
        if not isinstance(length_df, int) or length_df <= 0:
            raise ValueError("length_df must be a positive integer.")

        fill_df = pd.read_csv(poi_file, header=None)
        fill_positions = fill_df[0].astype(int).values
        if len(fill_positions) < 6:
            raise ValueError("POI file does not contain enough fill indices.")
        labels = np.empty(length_df, dtype=int)
        labels[:fill_positions[0]] = 0
        labels[fill_positions[0]:fill_positions[3]] = 1
        labels[fill_positions[3]:fill_positions[4]] = 1
        labels[fill_positions[4]:fill_positions[5]] = 1
        labels[fill_positions[5]:] = 1

        return pd.DataFrame(labels)

    @staticmethod
    def process_data(df: pd.DataFrame, live: bool = True, start_idx: int = -1, start_time: Union[int, float] = -1) -> pd.DataFrame:
        """
        Process the input DataFrame and generate features for forecasting.

        Validates that the DataFrame contains required columns, resets the index,
        generates features, and removes the "Relative_time" column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            live (bool, optional): Flag indicating live data processing. Defaults to True.
            start_idx (int, optional): Start index for feature generation. Defaults to -1.
            start_time (int or float, optional): Start time for feature generation. Defaults to -1.

        Returns:
            pd.DataFrame: Processed DataFrame with generated features.

        Raises:
            ValueError: If the DataFrame is empty or missing required columns.
        """
        if live:
            required_cols = ["Relative_time",
                             "Dissipation", "Resonance_Frequency"]
        else:
            required_cols = ["Relative_time", "Dissipation",
                             "Resonance_Frequency", "Fill"]
        if df.empty or not all(col in df.columns for col in required_cols):
            raise ValueError(
                "The dataset is empty or missing required columns.")
        df = df[required_cols].copy()
        if df.empty:
            raise ValueError(
                "DataFrame is empty after filtering required columns.")

        df = df.reset_index(drop=True)
        df = QForecastDataProcessor.generate_features(
            df, live=live, start_idx=start_idx, start_time=start_time)
        df.reset_index(drop=True, inplace=True)
        if "Relative_time" in df.columns:
            df.drop(columns=['Relative_time'], inplace=True)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return df
