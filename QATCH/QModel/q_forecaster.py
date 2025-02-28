
from scipy.signal import hilbert
from scipy.signal import savgol_filter
from scipy.signal import hilbert, savgol_filter

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
import pickle
from QATCH.core.worker import Worker
import matplotlib.pyplot as plt
FEATURES = [
    'Relative_time',
    'Dissipation',
    'Dissipation_rolling_mean',
    'Dissipation_rolling_median',
    'Dissipation_ewm',
    'Dissipation_rolling_std',
    'Dissipation_diff',
    'Dissipation_pct_change',
    'Dissipation_rate',
    'Dissipation_ratio_to_mean',
    'Dissipation_ratio_to_ewm',
    'Dissipation_envelope',
    'Time_shift'
]
TARGET = "Fill"
DATA_TO_LOAD = 50
IGNORE_BEFORE = 50


class QForecasterDataprocessor:
    @staticmethod
    def convert_to_dataframe(worker: Worker) -> pd.DataFrame:
        resonance_frequency = worker.get_value0_buffer(0)
        relative_time = worker.get_t1_buffer(0)
        dissipation = worker.get_d1_buffer(0)

        if not (len(relative_time) == len(dissipation)):
            raise ValueError("All buffers must have the same length.")

        df = pd.DataFrame({
            'Resonance_Frequency': dissipation,
            'Relative_time': relative_time,
            'Dissipation': dissipation
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
        Compute a series of additional features (e.g., rolling statistics,
        differences, ratios, and the signal envelope) for the Dissipation column.
        Also computes a 'Time_shift' column based on the first significant change in time.
        """
        window = 10
        span = 10
        run_length = len(df)
        window_length = int(np.ceil(0.01 * run_length))
        # Ensure window_length is odd and at least 3
        if window_length % 2 == 0:
            window_length += 1
        if window_length < 3:
            window_length = 3
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
        df['Relative_time_diff'] = df['Relative_time'].diff().replace(0, np.nan)
        df['Dissipation_rate'] = df['Dissipation_diff'] / df['Relative_time_diff']
        df['Dissipation_ratio_to_mean'] = df['Dissipation'] / \
            df['Dissipation_rolling_mean']
        df['Dissipation_ratio_to_ewm'] = df['Dissipation'] / df['Dissipation_ewm']
        df['Dissipation_envelope'] = np.abs(hilbert(df['Dissipation'].values))

        # Drop the Resonance_Frequency column if present.
        if 'Resonance_Frequency' in df.columns:
            df.drop(columns=['Resonance_Frequency'], inplace=True)

        t_delta = QForecasterDataprocessor.find_time_delta(df)
        if t_delta == -1:
            df['Time_shift'] = 0
        else:
            df.loc[t_delta:, 'Time_shift'] = 1

        # Replace any NaN or infinity values with 0
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

        return df

    @staticmethod
    def _process_fill(df: pd.DataFrame, poi_file: str, check_unique: bool = False, file_name: str = None) -> pd.DataFrame:
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
        # if check_unique:
        #     unique_fill = sorted(df["Fill"].unique())
        #     if len(unique_fill) != 7:
        #         print(f"[WARNING] File {file_name} does not have 7 unique Fill values; skipping."
        #               if file_name else "[WARNING] File does not have 7 unique Fill values; skipping.")
        #         return None

        df["Fill"] = df["Fill"].apply(QForecasterDataprocessor.reassign_region)
        mapping = {'no_fill': 0, 'init_fill': 1,
                   'ch_1': 2, 'ch_2': 3, 'full_fill': 4}
        df["Fill"] = df["Fill"].map(mapping)
        return df

    @staticmethod
    def load_and_preprocess_data_split(data_dir: str, required_runs: int = 20):
        """
        Load and preprocess data from all files in data_dir. Each file (and its matching
        POI file) is processed to compute additional features and fill information.
        Files are then categorized into 'short_runs' and 'long_runs' based on whether
        a significant time delta is detected. Before final concatenation, if the number
        of rows in a run exceeds IGNORE_BEFORE, the first IGNORE_BEFORE rows are dropped.
        Returns two DataFrames: one for short runs and one for long runs.
        """
        short_runs, long_runs = [], []
        content = QForecasterDataprocessor.load_content(data_dir)
        random.shuffle(content)

        for file, poi_file in content:
            # Exit early if both groups have reached the required number of runs.
            if len(short_runs) >= required_runs and len(long_runs) >= required_runs:
                break

            df = pd.read_csv(file)
            required_cols = ["Relative_time",
                             "Resonance_Frequency", "Dissipation"]
            if df.empty or not all(col in df.columns for col in required_cols):
                continue

            df = df[required_cols]
            df = QForecasterDataprocessor.compute_additional_features(df)
            df = QForecasterDataprocessor._process_fill(
                df, poi_file, check_unique=True, file_name=file)
            if df is None:
                continue

            if len(df) > IGNORE_BEFORE:
                df = df.iloc[IGNORE_BEFORE:]

            delta_idx = QForecasterDataprocessor.find_time_delta(df)
            if delta_idx == -1:
                if len(short_runs) < required_runs:
                    short_runs.append(df)
            else:
                if len(long_runs) < required_runs:
                    long_runs.append(df)

        if len(short_runs) < required_runs or len(long_runs) < required_runs:
            raise ValueError(f"Not enough runs found. Required: {required_runs} short and {required_runs} long, "
                             f"found: {len(short_runs)} short and {len(long_runs)} long.")

        training_data_short = pd.concat(short_runs).sort_values(
            "Relative_time").reset_index(drop=True)
        training_data_long = pd.concat(long_runs).sort_values(
            "Relative_time").reset_index(drop=True)
        return training_data_short, training_data_long

    @staticmethod
    def load_and_preprocess_single(data_file: str, poi_file: str):
        """
        Load and preprocess a single data file (and its corresponding POI file).
        The method ensures required sensor columns exist, processes the fill information,
        and prints a head sample of the resulting DataFrame.
        """
        df = pd.read_csv(data_file)
        required_cols = ["Relative_time", "Resonance_Frequency", "Dissipation"]
        if df.empty or not all(col in df.columns for col in required_cols):
            raise ValueError(
                "Data file is empty or missing required sensor columns.")
        df = df[required_cols]
        df = QForecasterDataprocessor._process_fill(df, poi_file)
        if df is None:
            raise ValueError("Error processing fill information.")
        print("[INFO] Preprocessed single-file data sample:")
        print(df.head())
        return df

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


###############################################################################
# Predictor Class: Handles generating predictions (including live predictions).
###############################################################################
class QForecasterPredictor:
    def __init__(self, numerical_features: list = FEATURES, target: str = TARGET,
                 save_dir: str = None, batch_threshold: int = 300):
        """
        Args:
            numerical_features (list): List of numerical feature names.
            categorical_features (list): List of categorical feature names.
            target (str): Target column name.
            save_dir (str): Directory from which to load saved objects.
            batch_threshold (int): Number of new entries to accumulate before running predictions.
        """
        self.numerical_features = numerical_features
        self.target = target
        self.save_dir = save_dir
        self.batch_threshold = batch_threshold

        # Placeholders for loaded objects.
        self.model_short = None
        self.model_long = None
        self.meta_clf = None
        self.preprocessors_short = None
        self.preprocessors_long = None
        self.transition_matrix_short = None
        self.transition_matrix_long = None
        self.meta_transition_matrix = None

        # Internal accumulator for live prediction updates.
        self.accumulated_data = None
        self.batch_num = 0
        self.fig = None
        self.axes = None

        # Prediction history for stability detection.
        self.prediction_history_short = {}
        self.prediction_history_long = {}

    def load_models(self):
        """
        Load base models, meta-classifier, and associated objects from disk.
        """
        if not self.save_dir:
            raise ValueError("Save directory not specified.")

        self.model_short = xgb.Booster()
        self.model_short.load_model(os.path.join(
            self.save_dir, "model_short.json"))
        with open(os.path.join(self.save_dir, "preprocessors_short.pkl"), "rb") as f:
            self.preprocessors_short = pickle.load(f)
        with open(os.path.join(self.save_dir, "transition_matrix_short.pkl"), "rb") as f:
            self.transition_matrix_short = pickle.load(f)

        self.model_long = xgb.Booster()
        self.model_long.load_model(os.path.join(
            self.save_dir, "model_long.json"))
        with open(os.path.join(self.save_dir, "preprocessors_long.pkl"), "rb") as f:
            self.preprocessors_long = pickle.load(f)
        with open(os.path.join(self.save_dir, "transition_matrix_long.pkl"), "rb") as f:
            self.transition_matrix_long = pickle.load(f)

        # Optionally, load the meta-classifier and its transition matrix if available.
        meta_clf_path = os.path.join(self.save_dir, "meta_clf.pkl")
        meta_trans_path = os.path.join(
            self.save_dir, "meta_transition_matrix.pkl")
        if os.path.exists(meta_clf_path):
            with open(meta_clf_path, "rb") as f:
                self.meta_clf = pickle.load(f)
        if os.path.exists(meta_trans_path):
            with open(meta_trans_path, "rb") as f:
                self.meta_transition_matrix = pickle.load(f)

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
        X_processed = self._apply_preprocessors(X, preprocessors)
        dmatrix = xgb.DMatrix(X_processed)
        return model.predict(dmatrix)

    def reset_accumulator(self):
        """
        Reset the internal accumulated dataframe and batch counter.
        """
        self.accumulated_data = None
        self.batch_num = 0

    def is_prediction_stable(self, pred_list, stability_window=5, frequency_threshold=0.8, confidence_threshold=0.9):
        """
        Determine if the predictions in pred_list are stable.

        Args:
            pred_list (list): A list of tuples (prediction, confidence) for recent updates.
            stability_window (int): The number of most recent predictions to consider.
            frequency_threshold (float): Fraction of predictions that must agree.
            confidence_threshold (float): Minimum average confidence required.

        Returns:
            (bool, stable_class): Tuple where bool indicates stability and stable_class is the converged prediction.
        """
        if len(pred_list) < stability_window:
            return False, None

        # Consider only the last 'stability_window' predictions.
        recent = pred_list[-stability_window:]
        counts = {}
        confidences = {}
        for pred, conf in recent:
            counts[pred] = counts.get(pred, 0) + 1
            confidences.setdefault(pred, []).append(conf)

        for pred, count in counts.items():
            if count / stability_window >= frequency_threshold:
                avg_conf = np.mean(confidences[pred])
                if avg_conf >= confidence_threshold:
                    return True, pred
        return False, None

    # --- Methods for model selection using the meta model ---
    def adjust_confidence(self, base_confidence, transition_matrix, current_state):
        """
        Adjust the model's base confidence using the dynamic transition matrix.

        Args:
            base_confidence (float): The original confidence score.
            transition_matrix (np.array): The dynamic transition matrix for the model.
            current_state (int): Current state index (as determined by your system logic).

        Returns:
            float: Adjusted confidence score.
        """
        # Example: Multiply base confidence by a weight derived from the transition matrix.
        weight = transition_matrix[current_state].max()
        return base_confidence * weight

    def extract_meta_features(self, X, conf_short, conf_long, additional_info):
        """
        Extract meta features from the input data and confidence scores.
        For model selection, we aggregate metrics across the batch into a single feature vector.

        Args:
            X (pd.DataFrame): Input data.
            conf_short (float): Aggregate confidence score from the short model.
            conf_long (float): Aggregate confidence score from the long model.
            additional_info (dict): Additional aggregated statistics.

        Returns:
            np.array: A feature vector of shape (1, n_features) for meta model prediction.
        """
        features = np.array([[conf_short, conf_long,
                              additional_info.get('stat_short', 0),
                              additional_info.get('stat_long', 0)]])
        return features

    def select_model(self, X, current_state=None):
        """
        Use the meta-model to decide which base model (short or long) should be used.
        Returns 0 for short model and 1 for long model.
        """
        # Override: if the time delta condition is met, choose the long model.
        if QForecasterDataprocessor.find_time_delta(X) >= 0:
            return 1  # Always select long model

        # 1. Get predictions for both models.
        prob_matrix_short = self.predict_native(
            self.model_short, self.preprocessors_short, X)
        prob_matrix_long = self.predict_native(
            self.model_long, self.preprocessors_long, X)

        # 2. Compute aggregate confidence scores.
        conf_short = np.mean(np.max(prob_matrix_short, axis=1))
        conf_long = np.mean(np.max(prob_matrix_long, axis=1))

        # Adjust based on the current state if provided.
        if current_state is not None:
            conf_short = self.adjust_confidence(
                conf_short, self.transition_matrix_short, current_state)
            conf_long = self.adjust_confidence(
                conf_long, self.transition_matrix_long, current_state)

        additional_info = {
            'stat_short': np.std(prob_matrix_short, axis=1).mean(),
            'stat_long': np.std(prob_matrix_long, axis=1).mean()
        }
        features = self.extract_meta_features(
            X, conf_short, conf_long, additional_info)

        model_choice = self.meta_clf.predict(features)[0]
        return model_choice

    def update_predictions(self, new_data, ignore_before=0, stability_window=5,
                           frequency_threshold=0.8, confidence_threshold=0.9, current_state=None):
        if self.accumulated_data is None:
            self.accumulated_data = pd.DataFrame(columns=new_data.columns)
            self.last_valid_index = 0
            self.last_seen_row = 0
            self.last_seen_time = new_data.iloc[self.last_seen_row]['Relative_time']

        matching_indices = new_data.index[new_data['Relative_time']
                                          == self.last_seen_time]
        if len(matching_indices) > 0:
            self.next_row = matching_indices[0]
        else:
            self.next_row = len(new_data)

        valid_new_data = new_data.iloc[self.last_seen_row:self.next_row]
        valid_new_data = valid_new_data.iloc[::-1]
        self.accumulated_data = pd.concat(
            [self.accumulated_data, valid_new_data], ignore_index=True)

        if self.next_row < len(new_data):
            self.last_seen_row = self.next_row
            self.last_seen_time = new_data.iloc[self.last_seen_row]['Relative_time']

        current_count = len(self.accumulated_data)
        self.batch_num += 1
        if self.batch_num == 1 and current_count > ignore_before:
            self.accumulated_data = self.accumulated_data.iloc[ignore_before:]
        self.accumulated_data.drop(
            columns=['Resonance_Frequency'], inplace=True)

        if current_count < self.last_valid_index + self.batch_threshold:
            print(
                f"[INFO] Accumulated {current_count} entries so far; waiting for exactly {self.batch_threshold} entries to run predictions.")
            return {
                "status": "waiting",
                "accumulated_count": current_count,
                "accumulated_data": self.accumulated_data,
                "predictions": None
            }
        self.accumulated_data = QForecasterDataprocessor.compute_additional_features(
            self.accumulated_data)
        plt.figure()
        plt.plot(self.accumulated_data['Relative_time'].values, label='R-time')
        # plt.plot(self.accumulated_data["Dissipation"], label='Dissipation')
        plt.legend()
        plt.show()
        plt.waitforbuttonpress()
        print(
            f"\n[INFO] Running predictions on batch {self.batch_num} with {current_count} entries.")

        # [The remainder of your existing prediction code follows here...]
        features = self.numerical_features
        X_live = self.accumulated_data[features]
        prob_matrix_short = self.predict_native(
            self.model_short, self.preprocessors_short, X_live)
        pred_short = QForecasterDataprocessor.viterbi_decode(
            prob_matrix_short, self.transition_matrix_short)
        prob_matrix_long = self.predict_native(
            self.model_long, self.preprocessors_long, X_live)
        pred_long = QForecasterDataprocessor.viterbi_decode(
            prob_matrix_long, self.transition_matrix_long)

        # Compute confidences
        conf_short = np.array([prob_matrix_short[i, pred_short[i]]
                               for i in range(len(pred_short))])
        conf_long = np.array([prob_matrix_long[i, pred_long[i]]
                              for i in range(len(pred_long))])

        # Update prediction histories
        for i, (p_s, p_l, c_s, c_l) in enumerate(zip(pred_short, pred_long, conf_short, conf_long)):
            self.prediction_history_short.setdefault(i, []).append((p_s, c_s))
            self.prediction_history_long.setdefault(i, []).append((p_l, c_l))

        # Use meta-model to select the model.
        selected_model = self.select_model(X_live, current_state=current_state)

        # Check for stable predictions.
        stable_predictions_short = {}
        for idx, history in self.prediction_history_short.items():
            stable, stable_class = self.is_prediction_stable(
                history, stability_window, frequency_threshold, confidence_threshold)
            if stable:
                stable_predictions_short[idx] = stable_class

        stable_predictions_long = {}
        for idx, history in self.prediction_history_long.items():
            stable, stable_class = self.is_prediction_stable(
                history, stability_window, frequency_threshold, confidence_threshold)
            if stable:
                stable_predictions_long[idx] = stable_class

        # Capture the accumulated data for plotting before resetting.
        data_for_plot = self.accumulated_data.copy()
        return {
            "status": "completed",
            "selected_model": selected_model,  # 0 for short, 1 for long
            "pred_short": pred_short,
            "pred_long": pred_long,
            "conf_short": conf_short,
            "conf_long": conf_long,
            "stable_predictions_short": stable_predictions_short,
            "stable_predictions_long": stable_predictions_long,
            "accumulated_data": data_for_plot,
            "accumulated_count": current_count
        }
