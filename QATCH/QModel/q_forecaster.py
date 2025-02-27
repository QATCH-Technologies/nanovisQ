from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.linear_model import LogisticRegression
from scipy.signal import hilbert
from scipy.signal import savgol_filter
from scipy.signal import hilbert, savgol_filter
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
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
DATA_TO_LOAD = 50
IGNORE_BEFORE = 50


class QForecasterDataprocessor:
    def convert_to_dataframe(worker: Worker) -> pd.DataFrame:
        # Retrieve the three buffers from the worker
        resonance_frequency = worker.get_value0_buffer(0)
        relative_time = worker.get_d1_buffer(0)
        dissipation = worker.get_d2_buffer(0)

        # Optional: check that all buffers have the same length
        if not (len(resonance_frequency) == len(relative_time) == len(dissipation)):
            raise ValueError("All buffers must have the same length.")

        # Create the DataFrame with the specified column headers
        df = pd.DataFrame({
            'Resonance_Frequency': resonance_frequency,
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
# Trainer Class: Handles model training, hyperparameter tuning, and saving.
###############################################################################

class QForecasterTrainer:
    def __init__(self, numerical_features, categorical_features=None, target='Fill', save_dir=None):
        """
        Args:
            numerical_features (list): List of numerical feature names.
            categorical_features (list): List of categorical feature names.
            target (str): Target column name.
            save_dir (str): Directory to save trained objects.
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.target = target
        self.save_dir = save_dir

        # Placeholders for trained objects.
        self.model_short = None
        self.model_long = None

        self.preprocessors_short = None
        self.preprocessors_long = None

        self.params_short = None
        self.params_long = None

        self.transition_matrix_short = None
        self.transition_matrix_long = None

    def _build_preprocessors(self, X):
        """Fit and return preprocessors for numerical and categorical data."""
        num_imputer = SimpleImputer(strategy='mean')
        X_num = num_imputer.fit_transform(X[self.numerical_features])
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
        preprocessors = {'num_imputer': num_imputer, 'scaler': scaler}

        if self.categorical_features:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_cat = cat_imputer.fit_transform(X[self.categorical_features])
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            X_cat = encoder.fit_transform(X_cat)
            preprocessors.update(
                {'cat_imputer': cat_imputer, 'encoder': encoder})
            X_processed = np.hstack([X_num, X_cat])
        else:
            X_processed = X_num

        return X_processed, preprocessors

    def _tune_parameters(self, dtrain, base_params, max_evals=10):
        """Tune XGBoost hyperparameters using Hyperopt."""
        space = {
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1)
        }

        def objective(hyperparams):
            params = base_params.copy()
            params['max_depth'] = int(hyperparams['max_depth'])
            params['min_child_weight'] = int(hyperparams['min_child_weight'])
            params['learning_rate'] = hyperparams['learning_rate']
            params['subsample'] = hyperparams['subsample']
            params['colsample_bytree'] = hyperparams['colsample_bytree']

            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=200,
                nfold=5,
                metrics={'aucpr'},
                early_stopping_rounds=10,
                seed=42,
                verbose_eval=False
            )
            best_score = cv_results['test-aucpr-mean'].max()
            best_rounds = len(cv_results)
            return {'loss': -best_score, 'status': STATUS_OK, 'num_rounds': best_rounds}

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest,
                    max_evals=max_evals, trials=trials)
        best['max_depth'] = int(best['max_depth'])
        best['min_child_weight'] = int(best['min_child_weight'])
        params = base_params.copy()
        params.update(best)
        best_trial = min(trials.results, key=lambda x: x['loss'])
        optimal_rounds = best_trial['num_rounds']
        return params, optimal_rounds

    def train_model_native(self, training_data, tune=True):
        """
        Train an XGBoost model on the provided training data.

        Returns:
            model: Trained XGBoost model.
            preprocessors (dict): Fitted preprocessors.
            params (dict): Model parameters.
        """
        features = self.numerical_features + self.categorical_features
        X = training_data[features].copy()
        y = training_data[self.target].values

        X_processed, preprocessors = self._build_preprocessors(X)
        dtrain = xgb.DMatrix(X_processed, label=y)

        base_params = {
            'objective': 'multi:softprob',
            'num_class': 5,
            'eval_metric': 'aucpr',
            'seed': 42,
            'device': 'cuda',
        }

        if tune:
            params, optimal_rounds = self._tune_parameters(dtrain, base_params)
        else:
            params = base_params.copy()
            params.update({'max_depth': 5, 'learning_rate': 0.1})
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=200,
                nfold=5,
                metrics={'aucpr'},
                early_stopping_rounds=10,
                seed=42,
                verbose_eval=False
            )
            optimal_rounds = len(cv_results)

        model = xgb.train(params, dtrain, num_boost_round=optimal_rounds)
        return model, preprocessors, params

    def predict_native(self, model: xgb.Booster, preprocessors, X):
        """
        Helper method to generate predictions; used when training the meta-classifier.
        """
        X_copy = X.copy()

        # Ensure feature names are set on the model.
        if model.feature_names is None:
            # Assume that the DataFrame columns are the correct feature names.
            model.feature_names = list(X_copy.columns)

        print("Model feature names:", model.feature_names)
        print("Preprocessors:", preprocessors)

        X_num = preprocessors['num_imputer'].transform(
            X_copy[model.feature_names])
        X_num = preprocessors['scaler'].transform(X_num)

        if self.categorical_features:
            X_cat = preprocessors['cat_imputer'].transform(
                X_copy[self.categorical_features])
            X_cat = preprocessors['encoder'].transform(X_cat)
            X_processed = np.hstack([X_num, X_cat])
        else:
            X_processed = X_num

        dmatrix = xgb.DMatrix(X_processed)
        return model.predict(dmatrix)

    def train_base_models(self, training_data_short, training_data_long, tune=True):
        """
        Train two base models (short and long) and compute their dynamic transition matrices.
        """
        self.transition_matrix_short = QForecasterDataprocessor.compute_dynamic_transition_matrix(
            training_data_short)
        self.transition_matrix_long = QForecasterDataprocessor.compute_dynamic_transition_matrix(
            training_data_long)

        self.model_short, self.preprocessors_short, self.params_short = self.train_model_native(
            training_data_short, tune=tune)
        self.model_long, self.preprocessors_long, self.params_long = self.train_model_native(
            training_data_long, tune=tune)
        print("[INFO] Base model training complete.")

    def save_models(self):
        """
        Save base models, meta-classifier, preprocessors, parameters, and transition matrices.
        """
        if not self.save_dir:
            raise ValueError("Save directory not specified.")
        os.makedirs(self.save_dir, exist_ok=True)

        # Save short model and its objects.
        self.model_short.save_model(os.path.join(
            self.save_dir, "model_short.json"))
        with open(os.path.join(self.save_dir, "preprocessors_short.pkl"), "wb") as f:
            pickle.dump(self.preprocessors_short, f)
        with open(os.path.join(self.save_dir, "params_short.pkl"), "wb") as f:
            pickle.dump(self.params_short, f)
        with open(os.path.join(self.save_dir, "transition_matrix_short.pkl"), "wb") as f:
            pickle.dump(self.transition_matrix_short, f)

        # Save long model and its objects.
        self.model_long.save_model(os.path.join(
            self.save_dir, "model_long.json"))
        with open(os.path.join(self.save_dir, "preprocessors_long.pkl"), "wb") as f:
            pickle.dump(self.preprocessors_long, f)
        with open(os.path.join(self.save_dir, "params_long.pkl"), "wb") as f:
            pickle.dump(self.params_long, f)
        with open(os.path.join(self.save_dir, "transition_matrix_long.pkl"), "wb") as f:
            pickle.dump(self.transition_matrix_long, f)

        print("[INFO] All models and associated objects have been saved successfully.")

    def load_models(self):
        """
        Optionally, the trainer can also load models from disk.
        """
        if not self.save_dir:
            raise ValueError("Save directory not specified.")

        self.model_short = xgb.Booster()
        self.model_short.load_model(os.path.join(
            self.save_dir, "model_short.json"))
        with open(os.path.join(self.save_dir, "preprocessors_short.pkl"), "rb") as f:
            self.preprocessors_short = pickle.load(f)
        with open(os.path.join(self.save_dir, "params_short.pkl"), "rb") as f:
            self.params_short = pickle.load(f)
        with open(os.path.join(self.save_dir, "transition_matrix_short.pkl"), "rb") as f:
            self.transition_matrix_short = pickle.load(f)

        self.model_long = xgb.Booster()
        self.model_long.load_model(os.path.join(
            self.save_dir, "model_long.json"))
        with open(os.path.join(self.save_dir, "preprocessors_long.pkl"), "rb") as f:
            self.preprocessors_long = pickle.load(f)
        with open(os.path.join(self.save_dir, "params_long.pkl"), "rb") as f:
            self.params_long = pickle.load(f)
        with open(os.path.join(self.save_dir, "transition_matrix_long.pkl"), "rb") as f:
            self.transition_matrix_long = pickle.load(f)

        print("[INFO] All models and associated objects have been loaded successfully.")


###############################################################################
# Predictor Class: Handles generating predictions (including live predictions).
###############################################################################


class QForecasterPredictor:
    def __init__(self, numerical_features: list = FEATURES, categorical_features: list = None, target: str = 'Fill', save_dir: str = None, batch_threshold: int = 300):
        """
        Args:
            numerical_features (list): List of numerical feature names.
            categorical_features (list): List of categorical feature names.
            target (str): Target column name.
            save_dir (str): Directory from which to load saved objects.
            batch_threshold (int): Number of new entries to accumulate before running predictions.
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features if categorical_features is not None else []
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
        self.prediction_history = {}

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

        with open(os.path.join(self.save_dir, "meta_classifier.pkl"), "rb") as f:
            self.meta_clf = pickle.load(f)
        with open(os.path.join(self.save_dir, "meta_transition_matrix.pkl"), "rb") as f:
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
        if self.categorical_features:
            X_cat = preprocessors['cat_imputer'].transform(
                X_copy[self.categorical_features])
            X_cat = preprocessors['encoder'].transform(X_cat)
            X_processed = np.hstack([X_num, X_cat])
        else:
            X_processed = X_num
        return X_processed

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

        # Count frequency for each prediction.
        counts = {}
        confidences = {}
        for pred, conf in recent:
            counts[pred] = counts.get(pred, 0) + 1
            confidences.setdefault(pred, []).append(conf)

        # Check if any prediction meets the criteria.
        for pred, count in counts.items():
            if count / stability_window >= frequency_threshold:
                avg_conf = np.mean(confidences[pred])
                if avg_conf >= confidence_threshold:
                    return True, pred
        return False, None

    def update_predictions(self, new_data, ignore_before=0, stability_window=5, frequency_threshold=0.8, confidence_threshold=0.9):
        """
        Update internal accumulated data with a new batch of data and compute predictions only when the number
        of accumulated rows reaches the batch_threshold.

        Args:
            new_data (DataFrame): New incoming batch of data.
            ignore_before (int): Number of initial rows to ignore (e.g., for stabilization).
            stability_window (int): Number of recent predictions to check for stability.
            frequency_threshold (float): Required frequency of a class for it to be considered stable.
            confidence_threshold (float): Minimum average confidence required for stability.

        Returns:
            dict: A dictionary with prediction results or a status message indicating waiting for more data.
        """
        # Initialize accumulator if needed.
        if self.accumulated_data is None:
            self.accumulated_data = pd.DataFrame(columns=new_data.columns)

        # Append the new data.
        self.accumulated_data = pd.concat(
            [self.accumulated_data, new_data], ignore_index=True)
        current_count = len(self.accumulated_data)

        # Check if we have collected enough new entries.
        if current_count < self.batch_threshold:
            print(
                f"[INFO] Accumulated {current_count} entries so far; waiting for {self.batch_threshold} entries to run predictions.")
            return {
                "status": "waiting",
                "accumulated_count": current_count,
                "predictions": None
            }

        # Enough data accumulated; proceed with predictions.
        self.batch_num += 1
        print(
            f"\n[INFO] Running predictions on batch {self.batch_num} with {current_count} entries.")

        # Optionally ignore initial rows for stabilization (only on the first run).
        if self.batch_num == 1 and current_count > ignore_before:
            self.accumulated_data = self.accumulated_data.iloc[ignore_before:]

        features = self.numerical_features + self.categorical_features
        X_live = self.accumulated_data[features]

        # Get predictions from both base models.
        prob_matrix_short = self.predict_native(
            self.model_short, self.preprocessors_short, X_live)
        pred_short = QForecasterDataprocessor.viterbi_decode(
            prob_matrix_short, self.transition_matrix_short)

        prob_matrix_long = self.predict_native(
            self.model_long, self.preprocessors_long, X_live)
        pred_long = QForecasterDataprocessor.viterbi_decode(
            prob_matrix_long, self.transition_matrix_long)

        # Compute confidences for base models.
        conf_short = np.array([prob_matrix_short[i, pred_short[i]]
                              for i in range(len(pred_short))])
        conf_long = np.array([prob_matrix_long[i, pred_long[i]]
                             for i in range(len(pred_long))])

        # Combine outputs for the meta classifier.
        meta_features = np.hstack([prob_matrix_short, prob_matrix_long])
        meta_prob_matrix = self.meta_clf.predict_proba(meta_features)
        final_preds = QForecasterDataprocessor.viterbi_decode(
            meta_prob_matrix, self.meta_transition_matrix)
        final_conf = np.array([meta_prob_matrix[i, final_preds[i]]
                              for i in range(len(final_preds))])

        # Update prediction history for stability checking.
        for idx, (pred, conf) in enumerate(zip(final_preds, final_conf)):
            if idx not in self.prediction_history:
                self.prediction_history[idx] = []
            self.prediction_history[idx].append((pred, conf))

        # Check which predictions are stable.
        stable_predictions = {}
        for idx, history in self.prediction_history.items():
            stable, stable_class = self.is_prediction_stable(
                history, stability_window, frequency_threshold, confidence_threshold)
            if stable:
                stable_predictions[idx] = stable_class

        # Optionally reset the accumulator after running predictions so the next prediction is only on new data.
        self.reset_accumulator()

        return {
            "status": "completed",
            "pred_short": pred_short,
            "pred_long": pred_long,
            "final_preds": final_preds,
            "conf_short": conf_short,
            "conf_long": conf_long,
            "final_conf": final_conf,
            "stable_predictions": stable_predictions,
            "accumulated_data": self.accumulated_data
        }

# -----------------------------
# Helper: Simulate Serial Stream
# -----------------------------


def simulate_serial_stream_from_loaded(loaded_data, batch_size=100):
    """
    Yields sequential batches of rows from the loaded data.
    """
    num_rows = len(loaded_data)
    for start_idx in range(0, num_rows, batch_size):
        yield loaded_data.iloc[start_idx:start_idx+batch_size]

# -----------------------------
# Live Prediction Simulator Class
# -----------------------------


def simulate_serial_stream_from_loaded(loaded_data):
    """
    Simulate a serial data stream by yielding random chunks from loaded_data.

    Each chunk will have a random size between 100 and 200 rows.

    Args:
        loaded_data (pd.DataFrame): The loaded dataset to stream.

    Yields:
        pd.DataFrame: A chunk of data with a random number of rows.
    """
    num_rows = len(loaded_data)
    start_idx = 0
    while start_idx < num_rows:
        batch_size = random.randint(100, 200)
        yield loaded_data.iloc[start_idx:start_idx + batch_size]
        start_idx += batch_size


class QForecasterSimulator:
    def __init__(self, predictor: QForecasterPredictor, dataset: pd.DataFrame, poi_file: pd.DataFrame = None, ignore_before=0, delay=1.0):
        """
        Simulator class to stream data in random chunk sizes and update predictions live.
        """
        self.predictor = predictor
        self.dataset = dataset
        self.ignore_before = ignore_before
        self.delay = delay

        # If a POI file is provided, extract the actual indices.
        if poi_file is not None:
            self.actual_poi_indices = poi_file.values
        else:
            self.actual_poi_indices = np.array([])

        # Setup a live plot with a single axis.
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.ion()  # Enable interactive mode.
        plt.show()

    def run(self):
        """
        Runs the simulation by iterating through the dataset in random chunk sizes,
        updating predictions, and plotting the current state.
        """
        self.predictor.reset_accumulator()
        batch_number = 0
        for batch_data in simulate_serial_stream_from_loaded(self.dataset):
            batch_number += 1
            print(
                f"[INFO] Processing batch {batch_number} with {len(batch_data)} rows.")

            # Update predictions using the new batch.
            results = self.predictor.update_predictions(
                batch_data, ignore_before=self.ignore_before)

            # Update the live plot.
            self.plot_results(results, batch_number=batch_number)

            # Simulate processing delay.
            plt.pause(self.delay)

        plt.ioff()  # Turn off interactive mode.
        plt.show()

    def plot_results(self, results, batch_number):
        """
        Updates the live plot with the normalized dissipation curve overlaid
        with background shading corresponding to contiguous predicted class regions,
        using channel fill names in the legend. Additionally, regions where predictions
        are stable are overlaid with a darker shading. The plot is styled with a clean,
        minimalist aesthetic.
        """
        self.ax.cla()  # Clear previous plot

        # Apply minimalist styling.
        self.ax.set_facecolor('white')
        self.fig.patch.set_facecolor('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # Plot the dissipation curve with a clean dark line.
        accumulated_data = results["accumulated_data"]
        x_dissipation = np.arange(len(accumulated_data))
        self.ax.plot(x_dissipation, accumulated_data["Dissipation"],
                     linestyle=':', color='#333333', linewidth=1.5, label='Dissipation Curve')

        # Define mappings for class IDs to names and colors.
        class_names = {
            0: "no_fill",
            1: "initial_fill",
            2: "channel_1",
            3: "channel_2",
            4: "full_fill"
        }
        class_colors = {
            0: "#d3d3d3",    # light grey for no_fill
            1: "#a6cee3",    # pastel blue for initial_fill
            2: "#b2df8a",    # pastel green for channel_1
            3: "#fdbf6f",    # pastel orange for channel_2
            4: "#fb9a99"     # pastel red/pink for full_fill
        }

        # Overlay shaded regions for predicted class regions.
        preds = np.array(results["final_preds"])
        if len(preds) > 0:
            already_labeled = {}
            current_class = preds[0]
            start_idx = 0
            for i in range(1, len(preds)):
                if preds[i] != current_class:
                    end_idx = i - 1
                    label = class_names[current_class] if current_class not in already_labeled else None
                    already_labeled[current_class] = True
                    self.ax.axvspan(start_idx, end_idx, color=class_colors.get(current_class, '#cccccc'),
                                    alpha=0.3, label=label)
                    current_class = preds[i]
                    start_idx = i
            # Shade the final segment.
            end_idx = len(preds) - 1
            label = class_names[current_class] if current_class not in already_labeled else None
            self.ax.axvspan(start_idx, end_idx, color=class_colors.get(current_class, '#cccccc'),
                            alpha=0.3, label=label)

        # Scatter the actual POI indices on the dissipation curve.
        if self.actual_poi_indices.size > 0:
            valid_actual_indices = [
                int(idx) for idx in self.actual_poi_indices if idx < len(accumulated_data)]
            y_actual = accumulated_data["Dissipation"].iloc[valid_actual_indices]
            self.ax.scatter(valid_actual_indices, y_actual,
                            color='#2ca02c', marker='o', s=50, label='Actual POI')

        # Overlay stable prediction regions as darker shaded areas.
        if "stable_predictions" in results and results["stable_predictions"]:
            stable_dict = results["stable_predictions"]
            stable_idxs = sorted(stable_dict.keys())
            if stable_idxs:
                segments = []
                current_class = stable_dict[stable_idxs[0]]
                seg_start = stable_idxs[0]
                prev = stable_idxs[0]
                # Group contiguous indices with the same stable class.
                for idx in stable_idxs[1:]:
                    if idx == prev + 1 and stable_dict[idx] == current_class:
                        prev = idx
                    else:
                        segments.append((seg_start, prev, current_class))
                        seg_start = idx
                        prev = idx
                        current_class = stable_dict[idx]
                segments.append((seg_start, prev, current_class))
                already_labeled_stable = {}
                for seg_start, seg_end, cls in segments:
                    label = f"Stable: {class_names[cls]}" if cls not in already_labeled_stable else None
                    already_labeled_stable[cls] = True
                    self.ax.axvspan(seg_start, seg_end, color=class_colors.get(cls, '#000000'),
                                    alpha=0.6, label=label)

        self.ax.set_title(
            f'Normalized Dissipation Curve (Batch {batch_number})', fontsize=14, weight='medium')
        self.ax.set_xlabel('Data Index', fontsize=12)
        self.ax.set_ylabel(self.predictor.target, fontsize=12)
        self.ax.tick_params(axis='both', which='major', labelsize=10)
        self.ax.legend(frameon=False, fontsize=10)

        # Redraw the figure.
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


TESTING = True
TRAINING = True

# Main execution block.
if __name__ == '__main__':
    # Load your dataset (update the path and parameters as needed).
    if TRAINING:
        train_dir = r'content\training_data\full_fill'
        training_data_short, training_data_long = QForecasterDataprocessor.load_and_preprocess_data_split(
            train_dir, required_runs=140)
        qft = QForecasterTrainer(
            FEATURES, None, "Fill", r'QModel\SavedModels\forecaster')
        # qft.train_base_models(training_data_short=training_data_short,
        #                       training_data_long=training_data_long, tune=True)
        qft.load_models()
        meta_data = pd.concat(
            [training_data_short, training_data_long], axis=1)
        qft.train_meta_model(training_data_short)
        print(qft.meta_transition_matrix)
        qft.save_models()
    if TESTING:
        test_dir = r"content\test_data"
        test_content = QForecasterDataprocessor.load_content(test_dir)
        random.shuffle(test_content)
        for data_file, poi_file in test_content:
            dataset = pd.read_csv(data_file)
            end = np.random.randint(0, len(dataset))
            random_slice = dataset.iloc[0:end]
            poi_file = pd.read_csv(poi_file, header=None)

            # Create an instance of the predictor.
            predictor = QForecasterPredictor(
                FEATURES, target='Fill', save_dir='QModel/SavedModels/forecaster/')
            # Ensure models and preprocessors are loaded.
            predictor.load_models()

            delay = dataset['Relative_time'].max() / len(random_slice)

            # Create the simulator, now passing the poi_file.
            simulator = QForecasterSimulator(
                predictor,
                random_slice,
                poi_file=poi_file,
                ignore_before=50,
                delay=delay
            )

            # Run the simulation.
            simulator.run()
