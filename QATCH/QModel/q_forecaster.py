
from scipy.signal import hilbert
from scipy.signal import savgol_filter
from scipy.signal import hilbert, savgol_filter
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from QATCH.core.worker import Worker
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
        relative_time = worker.get_t1_buffer(0)
        dissipation = worker.get_d2_buffer(0)

        if not (len(relative_time) == len(dissipation)):
            raise ValueError("All buffers must have the same length.")

        df = pd.DataFrame({
            'Resonance_Frequency': dissipation,
            'Relative_time': relative_time,
            'Dissipation': dissipation
        })

        return df

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


class QForecasterPredictor:
    def __init__(self, numerical_features: list, target: str = 'Fill',
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
        self.fig = None
        self.axes = None

        self.prediction_history = {}

    def load_models(self):
        if not self.save_dir:
            raise ValueError("Save directory not specified.")

        self.model = xgb.Booster()
        self.model.load_model(os.path.join(
            self.save_dir, "model.json"))
        with open(os.path.join(self.save_dir, "preprocessors.pkl"), "rb") as f:
            self.preprocessors = pickle.load(f)
        with open(os.path.join(self.save_dir, "transition_matrix.pkl"), "rb") as f:
            self.transition_matrix = pickle.load(f)

        print("[INFO] All models and associated objects have been loaded successfully.")

    def _apply_preprocessors(self, X, preprocessors):
        X_copy = X.copy()
        X_num = preprocessors['num_imputer'].transform(
            X_copy[self.numerical_features])
        X_num = preprocessors['scaler'].transform(X_num)
        return X_num

    def predict_native(self, model, preprocessors, X):
        X = X[FEATURES]
        X_processed = self._apply_preprocessors(X, preprocessors)
        dmatrix = xgb.DMatrix(X_processed)
        return model.predict(dmatrix)

    def reset_accumulator(self):
        self.accumulated_data = None
        self.batch_num = 0

    def is_prediction_stable(self, pred_list, stability_window=5, frequency_threshold=0.8, confidence_threshold=0.9):
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

    def update_predictions(self, new_data, ignore_before=0, stability_window=5,
                           frequency_threshold=0.8, confidence_threshold=0.9):
        # Initialize accumulator if needed.
        if self.accumulated_data is None:
            self.accumulated_data = pd.DataFrame(columns=new_data.columns)
            self.num_valid_entries = 0
            self.last_seen_row = 0
            self.last_seen_time = 0.0
        if self.last_seen_time == 0.0 and not new_data.empty:
            self.last_seen_time = new_data['Relative_time'].values[-1]

        matching_indices = new_data.index[new_data['Relative_time']
                                          == self.last_seen_time]
        if len(matching_indices) > 0:
            self.next_row = int(matching_indices[0])
        else:
            self.next_row = len(new_data)

        valid_new_data = new_data.iloc[self.last_seen_row:self.next_row]
        valid_new_data = valid_new_data.iloc[::-1]
        self.accumulated_data = pd.concat(
            [self.accumulated_data, new_data], ignore_index=True)
        current_count = len(self.accumulated_data)
        self.accumulated_data = QForecasterDataprocessor.compute_additional_features(
            self.accumulated_data)
        if current_count < self.batch_threshold:
            print(
                f"[INFO] Accumulated {current_count} entries so far; waiting for exactly {self.batch_threshold} entries to run predictions.")
            return {
                "status": "waiting",
                "accumulated_count": current_count,
                "accumulated_data": self.accumulated_data,
                "predictions": None
            }

        self.batch_num += 1
        print(
            f"\n[INFO] Running predictions on batch {self.batch_num} with {current_count} entries.")

        # Optionally ignore initial rows for stabilization (only on the first run).
        if self.batch_num == 1 and current_count > ignore_before:
            self.accumulated_data = self.accumulated_data.iloc[ignore_before:]

        features = self.numerical_features
        X_live = self.accumulated_data[features]

        prob_matrix = self.predict_native(
            self.model, self.preprocessors, X_live)
        pred = QForecasterDataprocessor.viterbi_decode(
            prob_matrix, self.transition_matrix)

        conf = np.array([prob_matrix[i, pred[i]]
                         for i in range(len(pred))])

        for i, (p_s, c_s) in enumerate(zip(pred, conf)):
            if i not in self.prediction_history:
                self.prediction_history[i] = []
            self.prediction_history[i].append((p_s, c_s))

        stable_predictions = {}
        for idx, history in self.prediction_history.items():
            stable, stable_class = self.is_prediction_stable(
                history, stability_window, frequency_threshold, confidence_threshold)
            if stable:
                stable_predictions[idx] = stable_class

        # Capture the accumulated data for plotting before resetting.
        data_for_plot = self.accumulated_data.copy()
        return {
            "status": "completed",
            "pred": pred,
            "conf": conf,
            "stable_predictions": stable_predictions,
            "accumulated_data": data_for_plot,
            "accumulated_count": current_count
        }
