import sys
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.early_stop import no_progress_loss
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
import pickle
from scipy.signal import find_peaks, argrelextrema
from scipy.interpolate import interp1d

# from ModelData import ModelData

Architecture_found = False
try:
    if not Architecture_found:
        from QATCH.common.architecture import Architecture
    Architecture_found = True
except:
    Architecture_found = False
    # Not finding this if OK: will use 'cwd' as 'root' path

QConstants_found = False
try:
    if not QConstants_found:
        from QConstants import *
    QConstants_found = True
except:
    QConstants_found = False
try:
    if not QConstants_found:
        from QATCH.QModel.QConstants import *
    QConstants_found = True
except:
    QConstants_found = False
if not QConstants_found:
    raise ImportError("Cannot find 'QConstants' in any expected location.")

ModelData_found = False
try:
    if not ModelData_found:
        from ModelData import ModelData
    ModelData_found = True
except:
    ModelData_found = False
try:
    if not ModelData_found:
        from QATCH.models.ModelData import ModelData
    ModelData_found = True
except:
    ModelData_found = False
if not ModelData_found:
    raise ImportError("Cannot find 'ModelData' in any expected location.")

QDataPipeline_found = False
try:
    if not QDataPipeline_found:
        from q_data_pipeline import QDataPipeline
    QDataPipeline_found = True
except:
    QDataPipeline_found = False
try:
    if not QDataPipeline_found:
        from QATCH.QModel.q_data_pipeline import QDataPipeline
    QDataPipeline_found = True
except:
    QDataPipeline_found = False
if not QDataPipeline_found:
    raise ImportError("Cannot find 'QDataPipeline' in any expected location.")


class QMultiModel:
    """
    A class used to represent and manage an XGBoost model for multi-target classification tasks.

    This class handles the initialization, training, hyperparameter tuning, and management of an XGBoost model.
    It supports multi-target classification with specific hyperparameters and uses cross-validation to optimize
    the model's performance.

    Attributes:
        __params__ (dict): A dictionary of hyperparameters used for training the XGBoost model.
        __train_df__ (pd.DataFrame): Training subset of the dataset.
        __valid_df__ (pd.DataFrame): Validation subset of the dataset.
        __test_df__ (pd.DataFrame): Test subset of the dataset.
        __dtrain__ (xgb.DMatrix): DMatrix object for the training data.
        __dvalid__ (xgb.DMatrix): DMatrix object for the validation data.
        __dtest__ (xgb.DMatrix): DMatrix object for the test data.
        __watchlist__ (list): List of DMatrix objects to monitor during training, containing tuples of the form (DMatrix, "name").
        __model__ (xgb.Booster or None): The trained XGBoost model, initially set to None.

    Methods:
        __init__(self, dataset, predictors, target_features):
            Initializes the XGBoost model with the specified dataset, predictors, and target features.

        train_model(self):
            Trains the multi-target XGBoost model using the training dataset.

        objective(self, params):
            Evaluates the performance of the XGBoost model using cross-validation and returns the best AUC score.

        tune(self, evaluations=250):
            Tunes the XGBoost model's hyperparameters using Bayesian optimization with Tree-structured Parzen Estimator (TPE).

        save_model(self, model_name="QMultiModel"):
            Saves the trained XGBoost model to a specified file.

        get_model(self):
            Retrieves the trained XGBoost model.
    """

    def __init__(
        self,
        dataset: pd.DataFrame = None,
        predictors: list = None,
        target_features: str = "Class",
    ) -> None:
        """
        Initializes the XGBoost model with the specified dataset, predictors, and target features.

        Args:
            dataset (pd.DataFrame): The complete dataset containing both predictors and target features.
            predictors (list[str]): A list of column names in `dataset` that will be used as features for model training.
            target_features (list[str]): A list of column names in `dataset` that will be used as target variables for the model.

        Attributes:
            __params__ (dict): A dictionary of hyperparameters used for training the XGBoost model. These include:
                - objective (str): The learning task and objective ("multi:softprob" for multi-class classification).
                - eval_metric (str): Evaluation metric ("auc" for Area Under the Curve).
                - eta (float): Step size shrinkage used in updates to prevent overfitting (0.175).
                - max_depth (int): Maximum depth of a tree (5).
                - min_child_weight (float): Minimum sum of instance weight (hessian) needed in a child (4.0).
                - subsample (float): Subsample ratio of the training instances (0.6).
                - colsample_bytree (float): Subsample ratio of columns when constructing each tree (0.75).
                - gamma (float): Minimum loss reduction required to make a further partition on a leaf node (0.8).
                - nthread (int): Number of threads used for training (NUM_THREADS).
                - booster (str): Type of booster to use ("gbtree").
                - device (str): Device to run on ("cuda").
                - tree_method (str): Tree construction algorithm ("auto").
                - sampling_method (str): Sampling method ("gradient_based").
                - seed (int): Random seed for reproducibility (SEED).
                - num_class (int): Number of classes (7).

            __train_df__ (pd.DataFrame): Training subset of the dataset.
            __valid_df__ (pd.DataFrame): Validation subset of the dataset.
            __test_df__ (pd.DataFrame): Test subset of the dataset.

            __dtrain__ (xgb.DMatrix): DMatrix object for the training data.
            __dvalid__ (xgb.DMatrix): DMatrix object for the validation data.
            __dtest__ (xgb.DMatrix): DMatrix object for the test data.

            __watchlist__ (list): List of DMatrix objects to watch during training, containing tuples of the form (DMatrix, "name").

            __model__ (xgb.Booster or None): The trained XGBoost model, initially set to None.
        """
        self.__params__ = {
            "objective": "multi:softprob",
            "eval_metric": "auc",
            "eta": 0.175,
            "max_depth": 5,
            "min_child_weight": 4.0,
            "subsample": 0.6,
            "colsample_bytree": 0.75,
            "gamma": 0.8,
            "nthread": NUM_THREADS,
            "booster": "gbtree",
            "device": "cuda",
            "tree_method": "auto",
            "sampling_method": "gradient_based",
            "seed": SEED,
            # "multi_strategy": "multi_output_tree",
            "num_class": 7,
        }
        self.__train_df__, self.__test_df__ = train_test_split(
            dataset, test_size=TEST_SIZE, random_state=SEED, shuffle=True
        )
        self.__train_df__, self.__valid_df__ = train_test_split(
            self.__train_df__,
            test_size=VALID_SIZE,
            random_state=SEED,
            shuffle=True,
        )

        self.__dtrain__ = xgb.DMatrix(
            data=self.__train_df__[predictors],
            label=self.__train_df__[target_features].values,
        )
        self.__dvalid__ = xgb.DMatrix(
            data=self.__valid_df__[predictors],
            label=self.__valid_df__[target_features].values,
        )
        self.__dtest__ = xgb.DMatrix(
            data=self.__test_df__[predictors],
            label=self.__test_df__[target_features].values,
        )

        self.__watchlist__ = [
            (self.__dtrain__, "train"),
            (self.__dvalid__, "valid"),
        ]

        self.__model__ = None

    def train_model(self) -> None:
        """
        Trains the multi-target XGBoost model using the training dataset.

        This method initializes the training process for the XGBoost model with the previously defined parameters and datasets.
        The model is trained over a number of rounds with early stopping if the performance does not improve on the validation set.

        During the training process, the model's performance is evaluated on both the training and validation datasets,
        and the best model based on the validation performance is saved.

        Prints a status message indicating the start of the training process.

        Attributes:
            __model__ (xgb.Booster): The trained XGBoost model after completion of the training process.

        Raises:
            ValueError: If any of the necessary parameters or datasets have not been initialized prior to calling this method.
        """
        print(f"[STATUS] Training multi-target model")
        self.__model__ = xgb.train(
            self.__params__,
            self.__dtrain__,
            MAX_ROUNDS,
            evals=self.__watchlist__,
            early_stopping_rounds=EARLY_STOP,
            maximize=True,
            verbose_eval=VERBOSE_EVAL,
        )

    def update_model(
        self,
        new_df: pd.DataFrame = None,
        predictors: list = None,
        target_features: str = "Class",
    ) -> None:
        print(f"[STATUS] Updating multi-target model")
        d_new = xgb.DMatrix(
            new_df[predictors],
            new_df[target_features].values,
        )
        self.__model__ = xgb.train(
            self.__params__,
            d_new,
            MAX_ROUNDS,
            evals=self.__watchlist__,
            early_stopping_rounds=EARLY_STOP,
            maximize=True,
            verbose_eval=VERBOSE_EVAL,
            xgb_model=self.__model__,
        )

    def objective(self, params: dict = None) -> None:
        """
        Evaluates the performance of the XGBoost model using cross-validation and returns the best AUC score.

        This method performs k-fold cross-validation on the training dataset using the provided hyperparameters.
        The method returns the best AUC score from the cross-validation process, which can be used as the objective function
        in hyperparameter optimization.

        Args:
            params (dict): Dictionary of hyperparameters to be used in the XGBoost model during cross-validation.

        Returns:
            dict: A dictionary containing the following keys:
                - "loss" (float): The best mean AUC score on the test set obtained during cross-validation.
                - "status" (str): The status of the evaluation, typically "ok".

        Raises:
            ValueError: If any of the necessary parameters or datasets have not been initialized prior to calling this method.
        """
        results = xgb.cv(
            params,
            self.__dtrain__,
            MAX_ROUNDS,
            nfold=NUMBER_KFOLDS,
            stratified=True,
            early_stopping_rounds=20,
            metrics=["auc"],
            verbose_eval=VERBOSE_EVAL,
        )
        best_score = results["test-auc-mean"].max()
        return {"loss": best_score, "status": STATUS_OK}

    def tune(self, evaluations: int = 250) -> dict:
        """
        Tunes the XGBoost model's hyperparameters using Bayesian optimization with Tree-structured Parzen Estimator (TPE).

        This method runs the hyperparameter tuning process for a specified number of iterations.
        It searches for the best hyperparameters within the defined search space by minimizing the loss function.
        The search is based on the performance of the model as evaluated by cross-validation.

        Args:
            evaluations (int, optional): The maximum number of iterations for hyperparameter optimization. Default is 250.

        Returns:
            dict: A dictionary of the best hyperparameters found during the tuning process.

        Raises:
            ValueError: If any of the necessary parameters or datasets have not been initialized prior to calling this method.

        Example:
            best_hyperparams = self.tune(evaluations=300)
        """
        print(f"[STATUS] Running model tuning for {evaluations} max iterations")
        space = {
            "max_depth": hp.choice("max_depth", np.arange(1, 20, 1, dtype=int)),
            "eta": hp.uniform("eta", 0, 1),
            "gamma": hp.uniform("gamma", 0, 10e1),
            "reg_alpha": hp.uniform("reg_alpha", 10e-7, 10),
            "reg_lambda": hp.uniform("reg_lambda", 0, 1),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            "colsample_bynode": hp.uniform("colsample_bynode", 0.5, 1),
            "colsample_bylevel": hp.uniform("colsample_bylevel", 0.5, 1),
            "min_child_weight": hp.choice(
                "min_child_weight", np.arange(1, 10, 1, dtype="int")
            ),
            "max_delta_step": hp.choice(
                "max_delta_step", np.arange(1, 10, 1, dtype="int")
            ),
            "subsample": hp.uniform("subsample", 0.5, 1),
            "eval_metric": "auc",
            "objective": "multi:softprob",
            "nthread": NUM_THREADS,
            "booster": "gbtree",
            "device": "cuda",
            "tree_method": "auto",
            "sampling_method": "gradient_based",
            "seed": SEED,
            # "multi_strategy": "multi_output_tree",
            "num_class": 7,
        }
        trials = Trials()
        best_hyperparams = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=evaluations,
            trials=trials,
            return_argmin=False,
            early_stop_fn=no_progress_loss(10),
        )

        self.__params__ = best_hyperparams.copy()
        if "eval_metric" in self.__params__:
            self.__params__ = {
                key: self.__params__[key]
                for key in self.__params__
                if key != "eval_metric"
            }

        print(f"-- best parameters, \n\t{best_hyperparams}")

        return best_hyperparams

    def save_model(self, model_name: str = "QMultiModel") -> None:
        """
        Saves the trained XGBoost model to a specified file.

        This method saves the current XGBoost model in JSON format to the specified location.
        The model is saved in the "QModel/SavedModels/" directory with the provided model name.

        Args:
            model_name (str, optional): The name of the model file to be saved. Default is "QMultiModel".

        Returns:
            None

        Example:
            self.save_model(model_name="MyModel")
        """
        filename = f"QModel/SavedModels/{model_name}.json"
        print(f"[INFO] Saving model {model_name}")
        self.__model__.save_model(filename)

    def get_model(self) -> xgb.Booster:
        """
        Retrieves the trained XGBoost model.

        This method returns the trained XGBoost model, which can be used for further predictions or analysis.

        Returns:
            xgb.Booster: The trained XGBoost model.

        Example:
            model = self.get_model()
        """
        return self.__model__


class QPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            raise ValueError("[QModelPredict __init__()] No model path given")

        # Load the pre-trained models from the specified paths
        self.__model__ = xgb.Booster()

        self.__model__.load_model(model_path)
        # with open("QModel/SavedModels/label_0.pkl", "rb") as file:
        #     self.__label_0__ = pickle.load(file)
        # with open("QModel/SavedModels/label_1.pkl", "rb") as file:
        #     self.__label_1__ = pickle.load(file)
        # with open("QModel/SavedModels/label_2.pkl", "rb") as file:
        #     self.__label_2__ = pickle.load(file)
        # Find the pickle files dynamically, using Architecture path (if available)
        if Architecture_found:
            relative_root = os.path.join(Architecture.get_path(), "QATCH")
        else:
            relative_root = os.getcwd()
        pickle_path = os.path.join(relative_root, "QModel/SavedModels/label_{}.pkl")
        for i in range(3):
            with open(pickle_path.format(i), "rb") as file:
                setattr(self, f"__label_{i}__", pickle.load(file))

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def compute_bounds(self, indices):
        bounds = []
        start = indices[0]
        end = indices[0]

        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                end = indices[i]
            else:
                bounds.append((start, end))
                start = indices[i]
                end = indices[i]

        # Append the last run
        bounds.append((start, end))

        return bounds

    def extract_results(self, results):

        num_indices = len(results[0])
        extracted = [[] for _ in range(num_indices)]

        for sublist in results:
            for idx in range(num_indices):
                extracted[idx].append(sublist[idx])

        return extracted

    def adjust_predictions(self, prediction, rel_time, poi_num, type, i, j):
        rel_time = rel_time[i:j]
        rel_time_norm = self.normalize(rel_time)
        bounds = None

        if type == 0:
            bounds = self.__label_0__["Class_" + str(poi_num)]
        elif type == 1:
            bounds = self.__label_1__["Class_" + str(poi_num)]
        elif type == 2:
            bounds = self.__label_2__["Class_" + str(poi_num)]

        lq = bounds["lq"]
        uq = bounds["uq"]
        adj = np.where((rel_time_norm >= lq) & (rel_time_norm <= uq), 1, 0)
        adjustment = np.concatenate(
            (np.zeros(i), np.array(adj), (np.zeros(len(prediction) - j)))
        )
        if len(prediction) == len(adjustment):
            adj_prediction = prediction * adjustment
            lq_idx = next((i for i, x in enumerate(adj) if x == 1), -1) + i
            uq_idx = (
                next((i for i, x in reversed(list(enumerate(adj))) if x == 1), -1) + i
            )
            return adj_prediction, (lq_idx, uq_idx)

        lq_idx = next((i for i, x in enumerate(adj) if x == 1), -1) + i
        uq_idx = next((i for i, x in reversed(list(enumerate(adj))) if x == 1), -1) + i
        return prediction, (lq_idx, uq_idx)

    def find_and_sort_peaks(self, signal):
        # Find peaks
        peaks, properties = find_peaks(signal)
        # Get the peak heights
        peak_heights = []
        for p in peaks:
            peak_heights.append(signal[p])

        # Sort peaks by height in descending order
        sorted_indices = np.argsort(peak_heights)[::-1]
        sorted_peaks = peaks[sorted_indices]
        return sorted_peaks

    def peak_density(self, signal, peaks, segment_le):
        density = len(peaks) / len(signal)
        return density

    def find_dynamic_increase(
        self, data, target_points=5, multiplier=3.0, reduction_factor=0.5, window_size=3
    ):
        if window_size < 2 or window_size >= len(data):
            return -1

        # Calculate the first derivative (differences between consecutive points)
        diff = np.diff(data)
        slope_change = []
        for i in range(len(diff) - window_size + 1):
            # Calculate the slope change over the current window
            window_slope_change = np.mean(np.diff(diff[i : i + window_size]))
            slope_change.append(window_slope_change)

        slope_change = np.array(slope_change)

        # Calculate baseline threshold (std deviation of slope changes in the sliding windows)
        baseline = np.std(slope_change)

        significant_points = []
        current_multiplier = multiplier

        # Iteratively reduce the threshold until enough significant points are found
        while len(significant_points) < target_points and current_multiplier > 0:
            # Calculate the current dynamic threshold
            dynamic_threshold = current_multiplier * baseline

            # Find new points where the change in slope exceeds the dynamic threshold
            new_points = np.where(slope_change > dynamic_threshold)[0]

            # Add new significant points to the list, ensuring no duplicates
            for point in new_points:
                if point + window_size - 1 not in [
                    p for p in significant_points
                ]:  # Adjust for window size
                    significant_points.append(
                        point + window_size - 1
                    )  # Store index and slope change
                    if len(significant_points) >= target_points:
                        break

            # Reduce the multiplier for the next iteration
            current_multiplier *= reduction_factor

        return significant_points[:target_points]

    def find_zero_slope_regions(self, data, threshold=1e-3, min_region_length=1):
        # Calculate the first derivative (approximate slope)
        slopes = np.diff(data)

        # Identify regions where the slope is approximately zero
        zero_slope_mask = np.abs(slopes) < threshold

        # Find consecutive regions of near-zero slope
        regions = []
        start_idx = None

        for i, is_zero_slope in enumerate(zero_slope_mask):
            if is_zero_slope and start_idx is None:
                start_idx = i
            elif not is_zero_slope and start_idx is not None:
                if i - start_idx >= min_region_length:
                    # Store the region as (start, end)
                    regions.append((start_idx, i))
                start_idx = None

        # Handle case where the last region extends to the end of the array
        if start_idx is not None and len(data) - start_idx - 1 >= min_region_length:
            regions.append((start_idx, len(data) - 1))

        return regions

    def adjustment_poi_1(self, guess, diss_raw):

        zero_slope = self.find_zero_slope_regions(self.normalize(diss_raw), 0.0075, 100)
        adjusted_guess = guess

        if len(zero_slope) >= 2:
            l = zero_slope[0][1]
            r = zero_slope[1][0]
            peaks_between, _ = find_peaks(diss_raw[l:r])
            between = []
            for p in peaks_between:
                between.append(p + l)
            # if emp_guess < l:
            #     adjusted_guess = r

            if between is not None and len(between) > 0:
                between = np.array(between)
                distances = np.abs(between - guess)
                if adjusted_guess >= r:
                    furthest_index = np.argmax(distances)
                    distances[furthest_index] = min(distances)
                    second_furthest_index = np.argmax(distances)
                    adjusted_guess = between[second_furthest_index]
                else:
                    nearest_index = np.argmin(distances)
                    distances[nearest_index] = np.argmin(distances)
                    adjusted_guess = between[nearest_index]

        else:
            peaks, _ = find_peaks(diss_raw)
            distances = np.abs(peaks - adjusted_guess)
            nearest_peak_index = peaks[np.argmin(distances)]
            adjusted_guess = nearest_peak_index

        # if abs(adjusted_guess - actual) > 5:
        #     fig, ax = plt.subplots()
        #     ax.plot(diss_raw, color="grey")
        #     for l, r in zero_slope:
        #         ax.fill_between((l, r), max(diss_raw), alpha=0.5)
        #     ax.scatter(between, diss_raw[between])
        #     ax.axvline(guess, color="green", linestyle="dotted", label="guess")

        #     ax.axvline(adjusted_guess, color="brown", label="adjusted")
        #     ax.axvline(actual, color="orange", linestyle="--", label="actual")
        #     plt.legend()

        #     plt.show()

        return adjusted_guess

    def adjustment_poi_2(self, guess, diss_raw, actual, bounds, poi_1_guess):

        diss_raw = self.normalize(diss_raw)
        diss_invert = -diss_raw
        start = bounds[0]
        stop = bounds[1]
        if guess < bounds[0]:
            start = guess

        if guess > bounds[1]:
            stop = guess
        if start < poi_1_guess:
            start = poi_1_guess

        if stop < poi_1_guess:
            delta = stop - start
            start = poi_1_guess
            stop = start + delta
        peaks, _ = find_peaks(diss_invert)
        # peaks = peaks + bounds[0]
        if len(peaks) == 0:
            return guess
        valid_peaks = peaks[peaks < guess]
        distances = np.abs(valid_peaks - guess)

        # Find the closest RF point to the initial guess, considering weights
        closest_idx = np.argmin(distances)
        adjustment = peaks[closest_idx]
        if adjustment < poi_1_guess:
            return guess
        points = np.array([adjustment, guess])
        weights = np.array([0.25, 0.75])
        weighted_mean = np.average(points, weights=weights)
        adjustment = int(weighted_mean)
        if abs(guess - adjustment) > 75:
            adjustment = guess
        # if abs(actual - adjustment) > 5:
        #     fig, ax = plt.subplots()
        #     ax.plot(diss_raw, color="grey")
        #     ax.fill_betweenx(
        #         [0, max(diss_raw)], bounds[0], bounds[1], color=f"yellow", alpha=0.5
        #     )
        #     ax.axvline(guess, color="green", linestyle="dotted", label="guess")

        #     ax.scatter(peaks, diss_raw[peaks])
        #     ax.axvline(adjustment, color="brown", label="adjusted")
        #     ax.axvline(actual[0], color="tan",
        #                linestyle="--", label="actual 1")
        #     ax.axvline(actual[1], color="orange",
        #                linestyle="--", label="actual 2")
        #     ax.scatter(
        #         peaks[closest_idx],
        #         diss_raw[peaks[closest_idx]],
        #         color="red",
        #         marker="x",
        #     )
        #     plt.legend()

        #     plt.show()

        return adjustment

    def adjustmet_poi_4(self, df, candidates, guess, actual, bounds):
        diss = df["Dissipation"]
        rf = df["Resonance_Frequency"]
        diff = df["Difference"]
        diss_peaks, _ = find_peaks(diss)
        rf_peaks, _ = find_peaks(rf)
        diff_peaks, _ = find_peaks(diff)
        initial_guess = np.array(guess)
        candidate_points = np.array(candidates)
        rf_points = np.array(rf_peaks)
        diss_points = np.array(diss_peaks)
        diff_points = np.array(diff_peaks)
        x_min, x_max = bounds
        candidates = np.append(candidates, guess)
        candidate_density = len(candidates) / (x_max - x_min)
        if candidate_density < 0.02 or (guess < x_min or guess > x_max):
            # Filter RF points within the bounds
            rf_points = np.concatenate((rf_points, diss_points))
            within_bounds = (rf_points >= x_min) & (rf_points <= x_max)
            filtered_rf_points = rf_points[within_bounds]

            # If no RF points within the bounds, return None or handle accordingly
            if filtered_rf_points.size == 0:

                return guess

            # Calculate proximity weight for each RF point based on diff and diss points
            def calculate_weight(rf_point):
                multiplier = 2
                diff_in_proximity = np.sum(
                    np.abs(np.array(diff_points) - rf_point) < 0.02 * len(rf)
                )
                weight = diff_in_proximity

                # Apply multiplier if a diss point is nearby
                if np.any(np.abs(np.array(diss_points) - rf_point) < 0.02 * len(rf)):
                    weight *= multiplier
                return weight

            epsilon = 1e-10

            # Calculate weights for filtered RF points
            weights = np.array(
                [calculate_weight(rf_point) for rf_point in filtered_rf_points]
            )

            # Calculate weighted distances from initial guess to each filtered RF point
            distances_to_rf = np.abs(filtered_rf_points - initial_guess)

            # Prevent division by zero by replacing zero weights with epsilon
            weighted_distances = distances_to_rf / np.where(
                weights == 0, epsilon, weights
            )

            # Find the closest RF point to the initial guess, considering weights
            closest_rf_idx = np.argmin(weighted_distances)
            closest_rf_point = filtered_rf_points[closest_rf_idx]

            # Calculate distances from candidate points to the closest RF point
            distances_to_closest_rf = np.abs(candidate_points - closest_rf_point)

            # Find the closest candidate point to the closest RF point
            closest_candidate_idx = np.argmin(distances_to_closest_rf)
            adjusted_point = candidate_points[closest_candidate_idx]
            # fig, ax = plt.subplots()
            # ax.plot(diss, color="grey")
            # ax.fill_betweenx(
            #     [0, max(diss)], bounds[0], bounds[1], color=f"yellow", alpha=0.5
            # )
            # # ax.scatter(diss_peaks, diss[diss_peaks], color="red", label="diss peaks")
            # ax.scatter(diff_peaks, diss[diff_peaks], color="green", label="diff peaks")
            # ax.scatter(rf_points, diss[rf_points], color="blue", label="rf peaks")
            # ax.scatter(candidates, diss[candidates], color="black", label="candidates")
            # ax.axvline(guess, color="orange", label="guess")
            # ax.axvline(actual, color="orange", linestyle="--", label="actual")
            # ax.axvline(adjusted_point, color="brown", label="adjusted")
            # plt.legend()
            # plt.show()
            return adjusted_point
        else:
            # fig, ax = plt.subplots()
            # ax.plot(diss, color="grey")
            # ax.fill_betweenx(
            #     [0, max(diss)], bounds[0], bounds[1], color=f"yellow", alpha=0.5
            # )
            # # ax.scatter(diss_peaks, diss[diss_peaks], color="red", label="diss peaks")
            # ax.scatter(diff_peaks, diss[diff_peaks], color="green", label="diff peaks")
            # ax.scatter(rf_peaks, diss[rf_peaks], color="blue", label="rf peaks")
            # ax.scatter(candidates, diss[candidates], color="black", label="candidates")
            # ax.axvline(guess, color="orange", label="guess")
            # ax.axvline(actual, color="orange", linestyle="--", label="actual")
            # plt.legend()
            # plt.show()
            return guess

    def adjustmet_poi_5(self, df, candidates, emp_guess, guess, actual, bounds):
        diss = df["Dissipation"]
        rf = df["Resonance_Frequency"]

        diff = df["Difference"]

        diss_peaks, _ = find_peaks(diss)
        rf_peaks, _ = find_peaks(rf)
        diff_peaks, _ = find_peaks(diff)
        initial_guess = np.array(guess)
        candidate_points = np.array(candidates)
        rf_points = np.array(rf_peaks)

        diss_points = np.array(diss_peaks)
        diff_points = np.array(diff_peaks)
        np.concatenate((rf_points, diff_points, diss_points))
        x_min, x_max = bounds
        candidate_density = len(candidates) / (x_max - x_min)
        if candidate_density < 0.01:
            zero_slope = self.find_zero_slope_regions(rf)
            # Filter RF points within the bounds
            # rf_points = np.concatenate((rf_points, diss_points))
            candidates = np.append(candidates, guess)
            within_bounds = (rf_points >= x_min) & (rf_points <= x_max)
            filtered_rf_points = rf_points[within_bounds]

            for l, r in zero_slope:
                np.append(rf_points, l)
                np.append(rf_points, r)
            # If no RF points within the bounds, return None or handle accordingly
            if filtered_rf_points.size == 0:
                return guess

            # Calculate proximity weight for each RF point based on diff and diss points
            def calculate_weight(rf_point):
                # multiplier = 2
                # diff_in_proximity = np.sum(
                #     np.abs(np.array(diff_points) - rf_point) < 0.02 * len(rf)
                # )
                # weight = diff_in_proximity

                # # Apply multiplier if a diss point is nearby
                # if np.any(np.abs(np.array(diss_points) - rf_point) < 0.02 * len(rf)):
                #     weight *= multiplier
                return 1

            # Calculate weights for filtered RF points
            weights = np.array(
                [calculate_weight(rf_point) for rf_point in filtered_rf_points]
            )

            # Calculate weighted distances from initial guess to each filtered RF point
            distances_to_rf = np.abs(filtered_rf_points - initial_guess)
            weighted_distances = distances_to_rf / weights  # Adjust distance by weight

            # Find the closest RF point to the initial guess, considering weights
            closest_rf_idx = np.argmin(weighted_distances)
            closest_rf_point = filtered_rf_points[closest_rf_idx]

            # Calculate distances from candidate points to the closest RF point
            distances_to_closest_rf = np.abs(candidate_points - closest_rf_point)

            # Find the closest candidate point to the closest RF point
            closest_candidate_idx = np.argmin(distances_to_closest_rf)
            adjusted_point = candidate_points[closest_candidate_idx]
            # fig, ax = plt.subplots()
            # ax.plot(diss, color="grey")
            # ax.fill_betweenx(
            #     [0, max(diss)], bounds[0], bounds[1], color=f"yellow", alpha=0.5
            # )
            # ax.scatter(diss_peaks, diss[diss_peaks],
            #            color="red", label="diss peaks")
            # ax.scatter(diff_peaks, diss[diff_peaks],
            #            color="green", label="diff peaks")
            # ax.scatter(rf_points, diss[rf_points],
            #            color="blue", label="rf peaks")
            # ax.scatter(
            #     emp_guess, diss[emp_guess], color="pink", marker="x", label="emp guess"
            # )
            # ax.scatter(candidates, diss[candidates],
            #            color="black", label="candidates")
            # ax.axvline(guess, color="purple",
            #            linestyle='dotted', label="guess")
            # ax.axvline(actual, color="orange", linestyle="--", label="actual")
            # ax.axvline(adjusted_point, color="brown", label="adjusted")
            # plt.legend()
            # plt.show()
            return adjusted_point
        else:
            return guess

    def adjustment_poi_6(self, guess, signal, diff, diss, actual, poi_5_guess):

        combo = self.normalize(signal[poi_5_guess:]) + self.normalize(
            diff[poi_5_guess:]
        )
        adjustment = np.argmax(combo) + poi_5_guess

        def nearest_peak(peaks, point):
            nearest_peak_idx = np.argmin(np.abs(peaks - point))
            return peaks[nearest_peak_idx]

        def envelope(signal):
            data = np.array(signal)
            maxima_indices = argrelextrema(data, np.greater)[0]
            upper_envelope = np.interp(
                np.arange(len(data)), maxima_indices, data[maxima_indices]
            )

            return upper_envelope, maxima_indices

        # if abs(adjustment - actual[5]) > 300:
        #     fig, ax = plt.subplots()
        #     ax.plot(self.normalize(diff), label="diff", color="brown")
        #     ax.plot(self.normalize(diss), label="diss", color="grey")
        #     ax.plot(combo, label="combo", color="black")
        #     ax.axvline(adjustment, label="Adjustment", color="red")
        #     ax.axvline(actual[5], label="Actual", color="red", linestyle="--")
        #     plt.legend()
        #     plt.show()
        return adjustment

    def predict(self, file_buffer, type=-1, start=-1, stop=-1, act=[None] * 6):
        # Load CSV data and drop unnecessary columns
        df = pd.read_csv(file_buffer)
        columns_to_drop = ["Date", "Time", "Ambient", "Temperature"]
        if not all(col in df.columns for col in columns_to_drop):
            raise ValueError(
                f"[QModelPredict predict()]: Input data must contain the following columns: {columns_to_drop}"
            )

        df = df.drop(columns=columns_to_drop)

        if not isinstance(file_buffer, str):
            if hasattr(file_buffer, "seekable") and file_buffer.seekable():
                # reset ByteIO buffer to beginning of stream
                file_buffer.seek(0)

            csv_headers = next(file_buffer)

            if isinstance(csv_headers, bytes):
                csv_headers = csv_headers.decode()

            if "Ambient" in csv_headers:
                csv_cols = (2, 4, 6, 7)
            else:
                csv_cols = (2, 3, 5, 6)

            file_data = np.loadtxt(
                file_buffer.readlines(), delimiter=",", skiprows=0, usecols=csv_cols
            )
            data_path = "QModel Passthrough"
            relative_time = file_data[:, 0]
            # temperature = file_data[:,1]
            resonance_frequency = file_data[:, 2]
            data = file_data[:, 3]

            emp_predictions = ModelData().IdentifyPoints(
                data_path=data_path,
                times=relative_time,
                freq=resonance_frequency,
                diss=data,
            )
        else:
            emp_predictions = ModelData().IdentifyPoints(file_buffer)
        emp_points = []
        start_bound = -1
        if isinstance(emp_predictions, list):
            for pt in emp_predictions:
                if isinstance(pt, int):
                    emp_points.append(pt)
                elif isinstance(pt, list):
                    max_pair = max(pt, key=lambda x: x[1])
                    emp_points.append(max_pair[0])
            start_bound = emp_points[0]

        ############################
        # MAIN PREDICTION PIPELINE #
        ############################

        file_buffer_2 = file_buffer
        if not isinstance(file_buffer_2, str):
            if hasattr(file_buffer_2, "seekable") and file_buffer_2.seekable():
                # reset ByteIO buffer to beginning of stream
                file_buffer_2.seek(0)
            else:
                # ERROR: 'file_buffer_2' must be 'BytesIO' type here, but it's not seekable!
                raise Exception(
                    "Cannot 'seek' stream prior to passing to 'QDataPipeline'."
                )
        else:
            # Assuming 'file_buffer_2' is a string to a file path, this will work fine as-is
            pass

        # Process data using QDataPipeline
        qdp = QDataPipeline(file_buffer_2)
        diss_raw = qdp.__dataframe__["Dissipation"]
        rel_time = qdp.__dataframe__["Relative_time"]
        qdp.preprocess(poi_filepath=None)
        diff_raw = qdp.__difference_raw__
        df = qdp.get_dataframe()
        f_names = self.__model__.feature_names
        df = df[f_names]
        d_data = xgb.DMatrix(df)

        results = self.__model__.predict(d_data)
        normalized_results = self.normalize(results)
        extracted_results = self.extract_results(normalized_results)

        # extracted_1 = emp_points[0]
        extracted_1 = np.argmax(extracted_results[1])
        extracted_2 = np.argmax(extracted_results[2])
        extracted_3 = np.argmax(extracted_results[3])
        extracted_4 = np.argmax(extracted_results[4])
        extracted_5 = np.argmax(extracted_results[5])
        extracted_6 = np.argmax(extracted_results[6])

        if not isinstance(emp_predictions, list):
            extracted_1 = np.argmax(extracted_results[1])

        if start > -1:
            extracted_1 = start
        if stop > -1:
            extracted_6 = stop

        if len(emp_points) <= 0:
            start_1 = extracted_1
            start_2 = extracted_2
            start_3 = extracted_3
            start_4 = extracted_4
            start_5 = extracted_5
            start_6 = extracted_6
        else:
            start_1 = emp_points[0]
            start_2 = emp_points[1]
            start_3 = emp_points[2]
            start_4 = emp_points[3]
            start_5 = emp_points[4]
            start_6 = emp_points[5]
        adj_1 = start_1
        poi_1 = self.adjustment_poi_1(guess=start_1, diss_raw=diss_raw)
        adj_2, bounds_2 = self.adjust_predictions(
            prediction=extracted_results[2],
            rel_time=rel_time,
            poi_num=2,
            type=type,
            i=poi_1,
            j=extracted_6,
        )
        adj_3, bounds_3 = self.adjust_predictions(
            prediction=extracted_results[3],
            rel_time=rel_time,
            poi_num=3,
            type=type,
            i=poi_1,
            j=extracted_6,
        )
        adj_4, bounds_4 = self.adjust_predictions(
            prediction=extracted_results[4],
            rel_time=rel_time,
            poi_num=4,
            type=type,
            i=poi_1,
            j=extracted_6,
        )
        adj_5, bounds_5 = self.adjust_predictions(
            prediction=extracted_results[5],
            rel_time=rel_time,
            poi_num=5,
            type=type,
            i=poi_1,
            j=extracted_6,
        )
        adj_6 = extracted_results[6]
        candidates_1 = self.find_and_sort_peaks(extracted_results[1])
        candidates_2 = self.find_and_sort_peaks(adj_2)
        candidates_3 = self.find_and_sort_peaks(adj_3)
        candidates_4 = self.find_and_sort_peaks(adj_4)
        candidates_5 = self.find_and_sort_peaks(adj_5)
        candidates_6 = self.find_and_sort_peaks(adj_6)

        poi_2 = self.adjustment_poi_2(
            guess=start_2,
            diss_raw=diss_raw,
            actual=act[1],
            bounds=bounds_2,
            poi_1_guess=poi_1,
        )
        # poi_2 = emp_points[1]
        poi_4 = self.adjustmet_poi_4(df, candidates_4, extracted_4, act[3], bounds_4)
        poi_5 = self.adjustmet_poi_5(
            df, candidates_5, extracted_5, start_5, act[4], bounds_5
        )
        if poi_1 >= poi_2:
            poi_1 = adj_1
        poi_3 = np.argmax(adj_3)

        # skip adjustment of point 6 when inverted (drop applied to outlet)
        if diff_raw.mean() < 0:
            poi_6 = start_6
        else:
            poi_6 = self.adjustment_poi_6(
                np.argmax(adj_6),
                adj_6,
                df["Difference"],
                df["Dissipation"],
                act[5],
                poi_5,
            )

        def sort_and_remove_point(arr, point):
            arr = np.array(arr)
            if len(arr) > MAX_GUESSES - 1:
                arr = arr[:MAX_GUESSES - 1]
            arr.sort()
            return arr[arr != point]

        candidates_1 = sort_and_remove_point(candidates_1, poi_1)
        candidates_2 = sort_and_remove_point(candidates_2, poi_2)
        candidates_3 = sort_and_remove_point(candidates_3, poi_3)
        candidates_4 = sort_and_remove_point(candidates_4, poi_4)
        candidates_5 = sort_and_remove_point(candidates_5, poi_5)
        candidates_6 = sort_and_remove_point(candidates_6, poi_6)

        candidates_1 = np.insert(candidates_1, 0, poi_1)
        candidates_2 = np.insert(candidates_2, 0, poi_2)
        candidates_3 = np.insert(candidates_3, 0, poi_3)
        candidates_4 = np.insert(candidates_4, 0, poi_4)
        candidates_5 = np.insert(candidates_5, 0, poi_5)
        candidates_6 = np.insert(candidates_6, 0, poi_6)

        confidence_1 = np.array(extracted_results[1])[candidates_1]
        confidence_2 = np.array(adj_2)[candidates_2]
        confidence_3 = np.array(adj_3)[candidates_3]
        confidence_4 = np.array(adj_4)[candidates_4]
        confidence_5 = np.array(adj_5)[candidates_5]
        confidence_6 = np.array(adj_6)[candidates_6]

        # TODO: Adjust 1st confidence to be better than 2nd guess

        candidates = [
            (candidates_1, confidence_1),
            (candidates_2, confidence_2),
            (candidates_3, confidence_3),
            (candidates_4, confidence_4),
            (candidates_5, confidence_5),
            (candidates_6, confidence_6),
        ]

        return candidates
