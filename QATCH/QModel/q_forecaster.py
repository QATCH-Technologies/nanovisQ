
from scipy.signal import savgol_filter, hilbert
from scipy.signal import hilbert
from scipy.signal import savgol_filter
from scipy.signal import hilbert, savgol_filter
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.utils import resample
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
FEATURES = [
    'Relative_time',
    'Dissipation',
    'Dissipation_rolling_mean',
    'Dissipation_rolling_median',
    'Dissipation_ewm',
    'Dissipation_rolling_std',
    'Dissipation_diff',
    'Dissipation_pct_change',
    'Dissipation_ratio_to_mean',
    'Dissipation_ratio_to_ewm',
    'Dissipation_envelope'
]
NUM_CLASSES = 4
TARGET = "Fill"
DOWNSAMPLE_FACTOR = 5
SPIN_UP_TIME = (1.2, 1.4)
BASELINE_WINDOW = 100


class QForecasterDataprocessor:
    @staticmethod
    def convert_to_dataframe(worker) -> pd.DataFrame:
        relative_time = worker.get_t1_buffer(0)
        dissipation = worker.get_d2_buffer(0)

        # Determine the minimum length from both buffers
        min_length = min(len(relative_time), len(dissipation))

        # Truncate both buffers to the minimum length
        relative_time_truncated = relative_time[:min_length]
        dissipation_truncated = dissipation[:min_length]

        df = pd.DataFrame({
            'Relative_time': relative_time_truncated,
            'Resonance_Frequency': dissipation_truncated,
            'Dissipation': dissipation_truncated
        })

        return df

    @staticmethod
    def load_content(data_dir: str) -> list:
        """
        Walk through data_dir and return a list of tuples.
        Each tuple contains the path to a CSV file (excluding those ending in "_poi.csv" or "_lower.csv")
        and its corresponding POI file (with '_poi.csv' replacing '.csv').
        """
        print(f"[INFO] Loading content from {data_dir}")
        loaded_content = []
        for root, _, files in tqdm(os.walk(data_dir), desc='Loading files...'):
            for f in files:
                if f.endswith(".csv") and not f.endswith("_poi.csv") and not f.endswith("_lower.csv"):
                    poi_file = f.replace(".csv", "_poi.csv")
                    loaded_content.append(
                        (os.path.join(root, f), os.path.join(root, poi_file))
                    )
        return loaded_content

    @staticmethod
    def find_time_delta(df: pd.DataFrame) -> int:
        """
        Compute the first index at which the difference in Relative_time
        deviates significantly from its expanding rolling mean.
        Returns -1 if no significant change is found.
        """
        time_df = pd.DataFrame()
        time_df["Delta"] = df["Relative_time"].diff()
        threshold = 0.032
        rolling_avg = time_df["Delta"].expanding(min_periods=2).mean()
        time_df["Significant_change"] = (
            time_df["Delta"] - rolling_avg).abs() > threshold
        change_indices = time_df.index[time_df["Significant_change"]].tolist()
        return change_indices[0] if change_indices else -1

    @staticmethod
    def reassign_region(fill):
        """
        Reassign numeric fill values to string region labels.
        """
        if fill == 0:
            return 'no_fill'
        elif fill in [1, 2, 3]:
            return 'init_fill'
        elif fill == 4:
            return 'ch_1'
        elif fill == 5:
            return 'ch_2'
        elif fill == 6:
            return 'full_fill'
        else:
            return fill  # fallback if needed

    @staticmethod
    def compute_additional_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a series of additional features for the Dissipation column.
        """
        window = 10
        span = 10
        run_length = len(df)
        # Calculate initial window_length as 1% of run_length
        window_length = int(np.ceil(0.01 * run_length))

        # Ensure window_length is odd and at least 3
        if window_length % 2 == 0:
            window_length += 1
        if window_length < 3:
            window_length = 3

        # Adjust window_length if it's greater than the size of the 'Dissipation' data.
        diss_length = len(df['Dissipation'])
        if window_length > diss_length:
            # Set to diss_length, ensuring it's odd.
            window_length = diss_length if diss_length % 2 == 1 else diss_length - 1

        polyorder = 2 if window_length > 2 else 1
        df['Dissipation'] = savgol_filter(df['Dissipation'].values,
                                          window_length=window_length,
                                          polyorder=polyorder)
        df['Dissipation_rolling_mean'] = df['Dissipation'].rolling(
            window=window, min_periods=1).mean()
        df['Dissipation_rolling_median'] = df['Dissipation'].rolling(
            window=window, min_periods=1).median()
        df['Dissipation_ewm'] = df['Dissipation'].ewm(
            span=span, adjust=False).mean()
        df['Dissipation_rolling_std'] = df['Dissipation'].rolling(
            window=window, min_periods=1).std()
        df['Dissipation_diff'] = df['Dissipation'].diff()
        df['Dissipation_pct_change'] = df['Dissipation'].pct_change()
        df['Dissipation_ratio_to_mean'] = df['Dissipation'] / \
            df['Dissipation_rolling_mean']
        df['Dissipation_ratio_to_ewm'] = df['Dissipation'] / df['Dissipation_ewm']
        df['Dissipation_envelope'] = np.abs(hilbert(df['Dissipation'].values))

        if 'Resonance_Frequency' in df.columns:
            df.drop(columns=['Resonance_Frequency'], inplace=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        return df

    @staticmethod
    def _process_fill(df: pd.DataFrame, poi_file: str) -> pd.DataFrame:
        """
        Helper to process fill information from the poi_file.
        Reads the poi CSV (with no header) and adds a Fill column to df.
        If the poi file does not contain a header, the method treats the first column
        as change indices (adding 1 to Fill from that index onward).
        Optionally, if check_unique is True, the method returns None when the number
        of unique fill values is not 7.
        """
        fill_df = pd.read_csv(poi_file, header=None)
        if "Fill" in fill_df.columns:
            df["Fill"] = fill_df["Fill"]
        else:
            df["Fill"] = 0
            change_indices = sorted(fill_df.iloc[:, 0].values)
            for idx in change_indices:
                df.loc[idx:, "Fill"] += 1

        df["Fill"] = pd.Categorical(df["Fill"]).codes
        df["Fill"] = df["Fill"].apply(QForecasterDataprocessor.reassign_region)
        mapping = {'no_fill': 0, 'init_fill': 1,
                   'ch_1': 2, 'ch_2': 3, 'full_fill': 4}
        df["Fill"] = df["Fill"].map(mapping)
        return df

    def compute_and_plot_statistics(training_data):
        import seaborn as sns

        # Compute statistics for 'Dissipation'
        diss_min = training_data['Dissipation'].min()
        diss_max = training_data['Dissipation'].max()
        diss_range = diss_max - diss_min
        diss_mean = training_data['Dissipation'].mean()
        diss_std = training_data['Dissipation'].std()

        print("Dissipation Statistics:")
        print(f"Min: {diss_min}")
        print(f"Max: {diss_max}")
        print(f"Range: {diss_range}")
        print(f"Mean: {diss_mean}")
        print(f"Std Dev: {diss_std}")

        # Alternatively, get a full summary
        print("\nFull descriptive statistics:")
        print(training_data['Dissipation'].describe())

        # Plot a histogram of Dissipation values with a KDE overlay
        plt.figure(figsize=(10, 6))
        sns.histplot(training_data['Dissipation'], kde=True, bins=30)
        plt.title("Distribution of Dissipation Values")
        plt.xlabel("Dissipation")
        plt.ylabel("Frequency")
        plt.show()

        # Plot a boxplot to highlight outliers and distribution spread
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=training_data['Dissipation'])
        plt.title("Boxplot of Dissipation Values")
        plt.xlabel("Dissipation")
        plt.show()

    @staticmethod
    def load_and_preprocess_data(data_dir: str, num_datasets: int):
        runs = []
        content = QForecasterDataprocessor.load_content(data_dir)
        random.shuffle(content)

        if num_datasets < len(content):
            print(f'[INFO] {num_datasets} datasets loaded.')
            content = content[:num_datasets]
        else:
            print(f'[INFO] {len(content)} datasets loaded.')
        for file, poi_file in content:
            df = pd.read_csv(file)

            required_cols = ["Relative_time", "Dissipation"]
            if df.empty or not all(col in df.columns for col in required_cols):
                continue
            df = df[required_cols]
            try:
                df = QForecasterDataprocessor._process_fill(
                    df, poi_file=poi_file)
                poi_df = pd.read_csv(poi_file, header=None)
            except FileNotFoundError:
                df = None
            if df is None:
                continue
            df = df[df["Relative_time"] >= random.uniform(
                SPIN_UP_TIME[0], SPIN_UP_TIME[1])]
            df = df.iloc[::DOWNSAMPLE_FACTOR]
            df = df.reset_index()
            init_fill_point = poi_df.values[0][0] - 1
            df = df.iloc[init_fill_point:]
            if df is None or df.empty or len(df) <= 10:
                continue
            df.loc[df['Fill'] == 0, 'Fill'] = 1
            df['Fill'] -= 1
            df = QForecasterDataprocessor.compute_additional_features(df)
            df.reset_index(inplace=True)
            runs.append(df)

        training_data = pd.concat(runs).sort_values(
            "Relative_time").reset_index(drop=True)
        # training_data.drop(columns=['Relative_time'], inplace=True)

        # Balance the classes via upsampling
        training_data = QForecasterDataprocessor.balance_classes(
            training_data, target_col='Fill')

        return training_data

    @staticmethod
    def viterbi_decode(prob_matrix, transition_matrix):
        T, N = prob_matrix.shape
        dp = np.full((T, N), -np.inf)
        backpointer = np.zeros((T, N), dtype=int)
        dp[0, 0] = np.log(prob_matrix[0, 0])
        for t in range(1, T):
            for j in range(N):
                allowed_prev = [0] if j == 0 else [j-1, j]
                best_state = allowed_prev[0]
                best_score = dp[t-1, best_state] + \
                    np.log(transition_matrix[best_state, j])
                for i in allowed_prev:
                    if transition_matrix[i, j] <= 0:
                        continue
                    score = dp[t-1, i] + np.log(transition_matrix[i, j])
                    if score > best_score:
                        best_score = score
                        best_state = i
                dp[t, j] = np.log(prob_matrix[t, j]) + best_score
                backpointer[t, j] = best_state
        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(dp[T-1])
        for t in range(T-2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]
        return best_path

    @staticmethod
    def compute_dynamic_transition_matrix(training_data, num_states=5, smoothing=1e-6):
        states = training_data["Fill"].values
        transition_counts = np.zeros((num_states, num_states))
        for i in range(num_states):
            transition_counts[i, i] = smoothing
            if i + 1 < num_states:
                transition_counts[i, i+1] = smoothing
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            if next_state == current_state or next_state == current_state + 1:
                transition_counts[current_state, next_state] += 1
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return transition_counts / row_sums

    @staticmethod
    def init_fill_point(
        df: pd.DataFrame, baseline_window: int = 10, threshold_factor: float = 3.0
    ) -> int:
        """
        Identify the first index in the 'Dissipation' column where a significant increase
        occurs relative to the initial baseline noise.
        """
        if 'Dissipation' not in df.columns:
            raise ValueError("Dissipation column not found in DataFrame.")
        if len(df) < baseline_window:
            return -1

        baseline_values = df['Dissipation'].iloc[:baseline_window]
        baseline_mean = baseline_values.mean()
        baseline_std = baseline_values.std()
        threshold = baseline_mean + threshold_factor * baseline_std
        dissipation = df['Dissipation'].values

        for idx, value in enumerate(dissipation):
            if value > threshold:
                return idx
        return -1

    @staticmethod
    def balance_classes(df, target_col='Fill'):
        # Determine the maximum count among classes
        max_count = df[target_col].value_counts().max()
        balanced_dfs = []
        # For each class, upsample the minority classes
        for cls in df[target_col].unique():
            cls_df = df[df[target_col] == cls]
            # Resample with replacement to match max_count
            cls_upsampled = resample(cls_df,
                                     replace=True,
                                     n_samples=max_count,
                                     random_state=123)
            balanced_dfs.append(cls_upsampled)
        # Concatenate the balanced dataframes
        balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
        return balanced_df


class QForecasterPredictor:
    def __init__(self, numerical_features: list = FEATURES, target: str = 'Fill',
                 save_dir: str = None, batch_threshold: int = 60):
        self.numerical_features = numerical_features
        self.target = target
        self.save_dir = save_dir
        self.batch_threshold = batch_threshold

        self.model = None
        self.preprocessors = None
        self.transition_matrix = None

        self.accumulated_data = None
        self.batch_num = 0
        self.prediction_history = {}

    def load_models(self):
        if not self.save_dir:
            raise ValueError("Save directory not specified.")

        self.model = xgb.Booster()
        self.model.load_model(os.path.join(self.save_dir, "model.json"))
        with open(os.path.join(self.save_dir, "preprocessors.pkl"), "rb") as f:
            self.preprocessors = pickle.load(f)
        with open(os.path.join(self.save_dir, "transition_matrix.pkl"), "rb") as f:
            self.transition_matrix = pickle.load(f)

        print("[INFO] All models and associated objects have been loaded successfully.")

    def _apply_preprocessors(self, X, preprocessors):
        """
        Apply loaded preprocessors to input data.
        """
        X_copy = X.copy()
        X_num = preprocessors['num_imputer'].transform(
            X_copy[self.numerical_features])
        X_num = preprocessors['scaler'].transform(X_num)
        return X_num

    def predict_native(self, model, preprocessors, X):
        """
        Generate predictions using a loaded XGBoost model.
        """
        X = X[self.numerical_features]
        X_processed = self._apply_preprocessors(X, preprocessors)
        dmatrix = xgb.DMatrix(X_processed)
        return model.predict(dmatrix)

    def reset_accumulator(self):
        """
        Reset the internal accumulated dataframe and batch counter.
        """
        self.accumulated_data = None
        self.batch_num = 0

    def is_prediction_stable(self, pred_list, stability_window=5, frequency_threshold=0.8, location_tolerance=0):
        """
        Determine if a prediction is stable based solely on its location.

        Each element in pred_list is assumed to be a tuple: (location, prediction).

        A prediction is considered stable if, within the last `stability_window` entries:
        - It appears at least a fraction `frequency_threshold` of the time, and
        - The locations where it occurs differ by no more than `location_tolerance` (default 0, i.e. identical).
        """
        if len(pred_list) < stability_window:
            return False, None

        recent = pred_list[-stability_window:]
        groups = {}
        for loc, pred in recent:
            if pred not in groups:
                groups[pred] = {'count': 0, 'locations': []}
            groups[pred]['count'] += 1
            groups[pred]['locations'].append(loc)

        for pred, data in groups.items():
            frequency = data['count'] / stability_window
            if frequency >= frequency_threshold:
                # Check if the prediction consistently occurs at the same location.
                if max(data['locations']) - min(data['locations']) <= location_tolerance:
                    return True, pred
        return False, None

    def update_predictions(self, new_data, stability_window=5,
                           frequency_threshold=0.8, confidence_threshold=0.6):
        """
        Accumulate new data and run predictions in batches.
        - Data before the init_fill_point are set to no_fill (0).
        - Predictions are generated only after the init_fill_point is identified.
        - Raw ML predictions are obtained via Viterbi decoding.
        - Stable predictions are determined from prediction history.
        """
        # Accumulate new data.
        if self.accumulated_data is None:
            self.accumulated_data = pd.DataFrame(columns=new_data.columns)
        self.accumulated_data = pd.concat(
            [self.accumulated_data, new_data], ignore_index=True)
        current_count = len(self.accumulated_data)

        # Compute additional features.
        self.accumulated_data = QForecasterDataprocessor.compute_additional_features(
            self.accumulated_data)

        # Identify the initial fill point.
        init_fill_point = QForecasterDataprocessor.init_fill_point(
            self.accumulated_data, 100, threshold_factor=20)

        # If fill hasn't started, report all data as no_fill (0).
        if init_fill_point == -1:
            print("[INFO] Fill not yet started; reporting all data as no_fill (0).")
            predictions = np.zeros(current_count, dtype=int)
            conf = np.zeros(current_count)
            for i in range(current_count):
                if i not in self.prediction_history:
                    self.prediction_history[i] = []
                self.prediction_history[i].append((0, 1.0))
            return {
                "status": "waiting",
                "pred": predictions,
                "conf": conf,
                "stable_predictions": {},
                "accumulated_data": self.accumulated_data,
                "accumulated_count": current_count
            }

        # For data after the init_fill point, run prediction using Viterbi decoding.
        predictions = np.zeros(current_count, dtype=int)
        conf = np.zeros(current_count)

        # Data before init_fill_point are set to no_fill (0).
        for i in range(init_fill_point):
            if i not in self.prediction_history:
                self.prediction_history[i] = []
            # (location, no_fill prediction)
            self.prediction_history[i].append((i, 0))

        # Run prediction on live data (after init_fill_point).
        X_live = self.accumulated_data[self.numerical_features].iloc[init_fill_point:]
        prob_matrix = self.predict_native(
            self.model, self.preprocessors, X_live)

        # Perform Viterbi decoding.
        ml_pred = QForecasterDataprocessor.viterbi_decode(
            prob_matrix, self.transition_matrix)
        ml_pred_offset = ml_pred + 1  # Offset raw predictions if needed.

        predictions[init_fill_point:] = ml_pred_offset
        ml_conf = np.array([prob_matrix[i, ml_pred[i]]
                           for i in range(len(ml_pred))])
        conf[init_fill_point:] = ml_conf

        # Update prediction history and check stability.
        for idx, p in enumerate(ml_pred_offset, start=init_fill_point):
            if idx not in self.prediction_history:
                self.prediction_history[idx] = []
            self.prediction_history[idx].append((idx, p))
        stable_predictions = {}
        for idx, history in self.prediction_history.items():
            stable, stable_class = self.is_prediction_stable(
                history, stability_window, frequency_threshold, confidence_threshold)
            if stable:
                stable_predictions[idx] = stable_class

        self.batch_num += 1
        print(f"\n[INFO] Running predictions on batch {self.batch_num} with {current_count} entries. "
              f"Predictions are offset by 1 and data before init_fill are marked as no_fill (0).")

        return {
            "status": "completed",
            "pred": predictions,
            "conf": conf,
            "stable_predictions": stable_predictions,
            "accumulated_data": self.accumulated_data,
            "accumulated_count": current_count
        }
