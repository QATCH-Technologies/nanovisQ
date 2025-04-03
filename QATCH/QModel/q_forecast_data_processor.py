import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import os
import logging as logging
import pandas as pd
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch, savgol_filter
from scipy.stats import skew, kurtosis, zscore
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from typing import Any
import random
from QATCH.common.logger import Logger
TAG = ['QForecastDataProcessor']
TARGET = "Fill"
DOWNSAMPLE_FACTOR = 5
SPIN_UP_TIME = (1.2, 1.4)
BASELINE_WINDOW = 100

PLOTTING = False


class QForecastDataProcessor:
    @staticmethod
    def convert_to_dataframe(worker: Any) -> pd.DataFrame:
        """Converts raw buffer data from a worker into a pandas DataFrame.

        Retrieves the relative time and dissipation buffers from the worker, truncates them to the same
        length, and constructs a DataFrame. The DataFrame contains the following columns:
          - 'Relative_time': The relative time data.
          - 'Resonance_Frequency': A duplicate of the dissipation data (placeholder).
          - 'Dissipation': The dissipation data.

        Args:
            worker (QATCH.core.worker.Worker): A Worker that provides buffer data through the methods
                          `get_t1_buffer(index: int)` and `get_d2_buffer(index: int)` where `t1` is the time
                          buffer and `d2` is the dissipation buffer.

        Returns:
            pd.DataFrame: A DataFrame with the columns 'Relative_time', 'Resonance_Frequency', and 'Dissipation'.
        """
        relative_time = worker.get_t1_buffer(0)
        resonance_frequency = worker.get_d1_buffer(0)
        dissipation = worker.get_d2_buffer(0)

        # Determine the minimum length from both buffers
        min_length = min(len(relative_time), len(
            dissipation), len(resonance_frequency))
        # Truncate buffers to the minimum length
        relative_time_truncated = relative_time[:min_length]
        dissipation_truncated = dissipation[:min_length]
        resonance_frequency_truncated = resonance_frequency[:min_length]

        df = pd.DataFrame({
            'Relative_time': relative_time_truncated,
            'Resonance_Frequency': resonance_frequency_truncated,
            'Dissipation': dissipation_truncated
        })

        return df

    @staticmethod
    def load_content(data_dir: str, num_datasets: int = np.inf, column: str = 'Dissipation') -> list:
        Logger.i(f"Loading content from {data_dir}")
        loaded_content = []

        for root, _, files in tqdm(os.walk(data_dir), desc='Loading files...'):
            for f in files:
                if f.endswith(".csv") and not f.endswith("_poi.csv") and not f.endswith("_lower.csv"):
                    poi_file = f.replace(".csv", "_poi.csv")
                    if not os.path.exists(os.path.join(root, poi_file)):
                        continue
                    poi_df = pd.read_csv(os.path.join(
                        root, poi_file), header=None)
                    poi_values = poi_df.values
                    if len(poi_values) != len(np.unique(poi_values)):
                        Logger.w(
                            f'POI file contains duplicate indices.')
                    else:
                        loaded_content.append(
                            (os.path.join(root, f), os.path.join(root, poi_file))
                        )
        random.shuffle(loaded_content)
        if num_datasets == np.inf:
            return loaded_content

        return loaded_content[:num_datasets]

    @staticmethod
    def load_balanced_content(data_dir: str, num_datasets: int = np.inf, clusters: int = 3, opt: bool = False) -> list:

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
                max_r_time = df['Relative_time'].max()
                feature_vec = [poi_mean, poi_std, mean_diss, std_diss,
                               mean_freq, std_freq, max_r_time]
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

        cluster_dict = {}
        for label, file_pair in zip(labels, valid_content):
            cluster_dict.setdefault(label, []).append(file_pair)

        balanced_sample = []
        if num_datasets == np.inf:
            min_count = min(len(files) for files in cluster_dict.values())
            for file_list in cluster_dict.values():
                balanced_sample.extend(random.sample(file_list, min_count))
        else:
            per_cluster = num_datasets // n_clusters
            for file_list in cluster_dict.values():
                if len(file_list) <= per_cluster:
                    balanced_sample.extend(file_list)
                else:
                    balanced_sample.extend(
                        random.sample(file_list, per_cluster))
            remaining = num_datasets - len(balanced_sample)
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
    def compute_optimal_clusters(features, min_clusters=2, max_clusters=10):
        scores = {}
        best_k = min_clusters
        best_score = -1
        # Ensure we don't consider more clusters than available samples.
        max_possible = min(max_clusters, len(features))
        for k in range(min_clusters, max_possible + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k

        if PLOTTING:
            # Optional: Plot silhouette scores to visualize the optimum.
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
        return pd.Series(gaussian_filter1d(df[col], sigma=sigma, order=1),
                         index=df.index)

    @staticmethod
    def compute_rolling_baseline_and_shift(dog_series: pd.Series, window: int) -> tuple:
        baseline = dog_series.rolling(
            window=window, center=True, min_periods=1).median()
        shift = dog_series - baseline
        return baseline, shift

    @staticmethod
    def compute_ocsvm_score(shift_series: pd.Series) -> np.ndarray:
        X = shift_series.values.reshape(-1, 1)
        ocsvm = OneClassSVM(nu=0.05, kernel='rbf',
                            gamma='scale', shrinking=False)
        ocsvm.fit(X)
        return ocsvm.decision_function(X)

    @staticmethod
    def compute_spectral_features(signal: np.ndarray, relative_time: pd.Series) -> tuple:
        fs = 1 / np.mean(np.diff(relative_time))
        f, Pxx = welch(signal, fs=fs, nperseg=min(256, len(signal)))
        psd_norm = Pxx / np.sum(Pxx)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
        dominant_frequency = f[np.argmax(Pxx)]
        return spectral_entropy, dominant_frequency

    @staticmethod
    def smooth_signal(signal: np.ndarray) -> np.ndarray:
        window_size = min(11, len(signal) - 1) if len(signal) > 11 else 5
        return savgol_filter(signal, window_size, polyorder=2)

    @staticmethod
    def compute_rolling_statistics(signal: pd.Series, window: int) -> tuple:
        skew_series = signal.rolling(
            window, min_periods=1).apply(skew, raw=True)
        kurtosis_series = signal.rolling(
            window, min_periods=1).apply(kurtosis, raw=True)
        variability_series = signal.rolling(window, min_periods=1).std()
        return skew_series, kurtosis_series, variability_series

    @staticmethod
    def compute_anomaly_zscore(signal: pd.Series) -> tuple:
        z_scores = pd.Series(
            zscore(signal, nan_policy="omit"), index=signal.index)
        anomaly = (np.abs(z_scores) > 2).astype(int)
        return z_scores, anomaly

    @staticmethod
    def cluster_anomalies(anomaly_flags: pd.Series) -> pd.Series:
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

    def compute_difference_curve(df: pd.DataFrame) -> pd.Series:
        xs = df["Relative_time"]
        i = next((x for x, t in enumerate(xs) if t > 0.5), None)
        j = next((x for x, t in enumerate(xs) if t > 2.5), None)

        if i is not None and j is not None:
            avg_res_freq = df["Resonance_Frequency"][i:j].mean()
            avg_diss = df["Dissipation"][i:j].mean()
            ys_diss = (df["Dissipation"] - avg_diss) * avg_res_freq / 2
            ys_freq = avg_res_freq - df["Resonance_Frequency"]
            difference_factor = 3
            return ys_freq - difference_factor * ys_diss
        return pd.Series(0, index=df.index)

    @staticmethod
    def generate_features(df: pd.DataFrame, live=True, start_idx: int = -1, start_time: float = -1) -> pd.DataFrame:
        if live:
            required_columns = ["Dissipation",
                                "Resonance_Frequency", "Relative_time"]
        else:
            required_columns = ["Dissipation",
                                "Resonance_Frequency", "Relative_time", "Fill"]

        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"Input DataFrame must contain the following columns: {required_columns}"
            )

        df = df[required_columns].copy()
        slice_loc = int(len(df) * 0.01)
        df["Difference"] = QForecastDataProcessor.compute_difference_curve(df)
        df["Dissipation"] -= np.average(
            df["Dissipation"].iloc[slice_loc:2*slice_loc])
        df["Resonance_Frequency"] -= np.average(
            df["Resonance_Frequency"].iloc[slice_loc:2*slice_loc])
        df["Resonance_Frequency"] = df["Resonance_Frequency"].values * -1

        window_length = 100
        polyorder = 3
        df["Difference_smooth"] = savgol_filter(
            df["Difference"], window_length, polyorder)
        df["Dissipation_smooth"] = savgol_filter(
            df["Dissipation"], window_length, polyorder)
        df["Resonance_Frequency_smooth"] = savgol_filter(
            df["Resonance_Frequency"], window_length, polyorder)

        # Add the "Time_since_start" feature.
        # If either start_idx or start_time is -1 (i.e. no valid start), assign -1 for all rows.
        if start_idx == -1 or start_time == -1:
            df["Time_since_start"] = -1
        else:
            df["Time_since_start"] = np.where(
                df.index >= start_idx,
                df["Relative_time"] - start_time,
                -1
            )
        # ########################
        #  Dissipation Processing
        # ########################

        df['Dissipation_DoG'] = QForecastDataProcessor.compute_DoG(
            df, col='Dissipation')
        # baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        # df['Dissipation_DoG_baseline'], df['Dissipation_DoG_shift'] = DataProcessor.compute_rolling_baseline_and_shift(
        #     df['Dissipation_DoG'], baseline_window
        # )
        # df['Dissipation_DoG_SVM_Score'] = DataProcessor.compute_ocsvm_score(
        #     df['Dissipation_DoG_shift'])
        # df['Dissipation_DoG_SVM_Score_Smooth'] = gaussian_filter1d(
        #     df['Dissipation_DoG_SVM_Score'], sigma=5)

        # ################
        #  RF Processing
        # ################

        df['Resonance_Frequency_DoG'] = QForecastDataProcessor.compute_DoG(
            df, col='Resonance_Frequency')
        # baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        # df['Reonance_Frequency_DoG_baseline'], df['Resonance_Frequency_DoG_shift'] = DataProcessor.compute_rolling_baseline_and_shift(
        #     df['Resonance_Frequency_DoG'], baseline_window
        # )
        # df['Resonance_Frequency_DoG_SVM_Score'] = DataProcessor.compute_ocsvm_score(
        #     df['Resonance_Frequency_DoG_shift'])
        # df['Resonance_Frequency_DoG_SVM_Score_Smooth'] = gaussian_filter1d(
        #     df['Resonance_Frequency_DoG_SVM_Score'], sigma=5)
        # #######################
        #  Difference Processing
        # #######################
        df['Difference_DoG'] = QForecastDataProcessor.compute_DoG(
            df, col='Difference')
        # baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        # df['Difference_DoG_baseline'], df['Difference_DoG_shift'] = DataProcessor.compute_rolling_baseline_and_shift(
        #     df['Difference_DoG'], baseline_window
        # )
        # df['Difference_DoG_SVM_Score'] = DataProcessor.compute_ocsvm_score(
        #     df['Difference_DoG_shift'])
        # df['Difference_DoG_SVM_Score_Smooth'] = gaussian_filter1d(
        #     df['Difference_DoG_SVM_Score'], sigma=5)
        # z_scores, anomaly = DataProcessor.compute_anomaly_zscore(
        #     df["DoG_SVM_Score_Smooth"])
        # df["DoG_SVM_Zscore"] = z_scores
        # df["DoG_SVM_Anomaly"] = DataProcessor.cluster_anomalies(anomaly)

        # smooth_factor = 21
        # if len(df) > smooth_factor:
        #     smoothed_DoG = savgol_filter(
        #         df['DoG_SVM_Score'], window_length=smooth_factor, polyorder=3
        #     )
        # else:
        #     smoothed_DoG = df['DoG_SVM_Score'].values

        # if smoothed_DoG.size < 2:
        #     df['DoG_derivative'] = np.zeros_like(smoothed_DoG)
        # else:
        #     df['DoG_derivative'] = np.gradient(smoothed_DoG)

        # df['DoG_slope'] = pd.Series(smoothed_DoG).rolling(window=10).apply(
        #     lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
        # )

        # window_size = 20
        # df['DoG_win_mean'] = pd.Series(
        #     smoothed_DoG).rolling(window=window_size).mean()
        # df['DoG_win_std'] = pd.Series(
        #     smoothed_DoG).rolling(window=window_size).std()
        # df['DoG_win_min'] = pd.Series(
        #     smoothed_DoG).rolling(window=window_size).min()
        # df['DoG_win_max'] = pd.Series(
        #     smoothed_DoG).rolling(window=window_size).max()

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        return df

    @staticmethod
    def process_fill(poi_file: str, length_df: int) -> pd.DataFrame:
        fill_df = pd.read_csv(poi_file, header=None)
        fill_positions = fill_df[0].astype(int).values
        labels = np.empty(length_df, dtype=int)
        labels[:fill_positions[0]] = 0
        labels[fill_positions[0]:fill_positions[3]] = 1
        labels[fill_positions[3]:fill_positions[4]] = 1
        labels[fill_positions[4]:fill_positions[5]] = 1
        labels[fill_positions[5]:] = 1

        return pd.DataFrame(labels)

    @staticmethod
    def process_data(df: pd.DataFrame, live: bool = True, start_idx: int = -1, start_time: int = -1):
        if live:
            required_cols = ["Relative_time", "Dissipation",
                             "Resonance_Frequency"]
        else:
            required_cols = ["Relative_time", "Dissipation",
                             "Resonance_Frequency", "Fill"]
        if df.empty or not all(col in df.columns for col in required_cols):
            raise ValueError(
                "The dataset is empty or missing required columns.")
        df = df[required_cols].copy()
        if df.empty:
            return None
        df = df.reset_index(drop=True)

        df = QForecastDataProcessor.generate_features(
            df, live=live, start_idx=start_idx, start_time=start_time)
        df.reset_index(drop=True, inplace=True)
        df.drop(columns=['Relative_time'], inplace=True)

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        return df
