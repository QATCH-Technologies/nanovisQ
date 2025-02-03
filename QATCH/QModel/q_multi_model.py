import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.early_stop import no_progress_loss
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
import pickle
from scipy.signal import find_peaks
from scipy.stats import linregress
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
        print(
            f"[STATUS] Running model tuning for {evaluations} max iterations")
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
        pickle_path = os.path.join(
            relative_root, "QModel/SavedModels/label_{}.pkl")
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
        uq_idx = next((i for i, x in reversed(
            list(enumerate(adj))) if x == 1), -1) + i
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
            window_slope_change = np.mean(np.diff(diff[i: i + window_size]))
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

    def adjustment_poi_1(self, initial_guess, dissipation_data):
        """Adjusts the point of interest (POI) based on the nearest zero-slope region or peak.

        This method refines the initial guess of the point of interest by finding
        nearby regions with zero slope and selecting the most suitable peak. If no
        valid zero-slope region is found, it adjusts the guess to the nearest peak
        in the dissipation data.

        Args:
            initial_guess (int): Initial point of interest guess, typically an index within the dataset.
            dissipation_data (np.ndarray): 1D array of dissipation values.

        Returns:
            int: Adjusted index of the point of interest, either at or near a zero-slope
                region or closest peak.
        """
        zero_slope_regions = self.find_zero_slope_regions(
            self.normalize(dissipation_data), threshold=0.0075, min_region_length=100
        )
        adjusted_guess = initial_guess

        if len(zero_slope_regions) >= 2:
            left_bound = zero_slope_regions[0][1]
            right_bound = zero_slope_regions[1][0]

            peaks_between, _ = find_peaks(
                dissipation_data[left_bound:right_bound])
            peaks_indices = [peak + left_bound for peak in peaks_between]

            if peaks_indices:
                peaks_array = np.array(peaks_indices)
                distances = np.abs(peaks_array - initial_guess)

                if adjusted_guess >= right_bound:
                    furthest_index = np.argmax(distances)
                    distances[furthest_index] = np.min(
                        distances
                    )  # Ignore the furthest peak for the second furthest
                    second_furthest_index = np.argmax(distances)
                    adjusted_guess = peaks_array[second_furthest_index]
                else:
                    nearest_index = np.argmin(distances)
                    adjusted_guess = peaks_array[nearest_index]

        else:
            all_peaks, _ = find_peaks(dissipation_data)
            distances_to_peaks = np.abs(all_peaks - adjusted_guess)
            nearest_peak = all_peaks[np.argmin(distances_to_peaks)]
            adjusted_guess = nearest_peak

        return adjusted_guess

    def adjustment_poi_2(self, initial_guess, dissipation_data, bounds, poi_1_estimate):
        """Adjusts the point of interest (POI) within specified bounds, influenced by another POI.

        This method refines the initial guess for a point of interest based on specified
        bounds and a previous POI estimate. It calculates the nearest peak within the bounds,
        weighted toward the initial guess, and returns an adjusted POI if valid.

        Args:
            initial_guess (int): Initial guess for the point of interest, typically an index.
            dissipation_data (np.ndarray): 1D array of dissipation values to be processed.
            bounds (tuple[int, int]): Start and stop bounds within which to adjust the POI.
            poi_1_estimate (int): Estimate of a prior POI that affects the adjustment range.

        Returns:
            int: Adjusted index of the point of interest within the specified bounds.
        """
        dissipation_data = self.normalize(dissipation_data)
        dissipation_inverted = -dissipation_data
        start_bound, stop_bound = bounds

        if initial_guess < start_bound:
            start_bound = initial_guess
        elif initial_guess > stop_bound:
            stop_bound = initial_guess

        if start_bound < poi_1_estimate:
            start_bound = poi_1_estimate + 1
        if stop_bound < poi_1_estimate:
            range_delta = stop_bound - start_bound
            start_bound = poi_1_estimate
            stop_bound = start_bound + range_delta - 1

        peaks, _ = find_peaks(dissipation_inverted)
        if len(peaks) == 0:
            return initial_guess

        valid_peaks = peaks[peaks < initial_guess]
        distances_to_guess = np.abs(valid_peaks - initial_guess)
        closest_peak_idx = np.argmin(distances_to_guess)
        adjusted_guess = valid_peaks[closest_peak_idx]

        if adjusted_guess < poi_1_estimate:
            return initial_guess

        points = np.array([adjusted_guess, initial_guess])
        # Biases adjustment toward the initial guess
        weights = np.array([0.25, 0.75])
        weighted_mean = np.average(points, weights=weights)
        final_adjustment = int(weighted_mean)

        if abs(initial_guess - final_adjustment) > 75:
            final_adjustment = initial_guess

        return final_adjustment

    def adjustmet_poi_4(self, df, candidates, guess, actual, bounds, poi_1_guess):
        dissipation = self.normalize(df["Dissipation"])
        rf = self.normalize(df["Resonance_Frequency"])
        difference = self.normalize(df["Difference"])
        candidates = np.append(candidates, guess)

        # Temporary variable for the adjusted point.
        adjusted_point = -1

        def find_zero_slope_regions(data, threshold=0.1):
            # Calculate the slope (difference between consecutive points)
            slopes = np.diff(data)

            # Identify indices where the slope is within the threshold
            zero_slope_indices = np.where(np.abs(slopes) < threshold)[0]

            # Group indices into continuous regions
            zero_slope_regions = []
            if zero_slope_indices.size > 0:
                current_region = [zero_slope_indices[0]]

                for i in range(1, len(zero_slope_indices)):
                    if zero_slope_indices[i] == zero_slope_indices[i - 1] + 1:
                        current_region.append(zero_slope_indices[i])
                    else:
                        zero_slope_regions.append(current_region)
                        current_region = [zero_slope_indices[i]]

                # Append the last region
                zero_slope_regions.append(current_region)

            return zero_slope_regions

        peaks, _ = find_peaks(rf)

        # Check if there is a filtering of peaks such that they appear within our predefined, bounded
        # region.  If not, just report the guessed POI.
        filtered_peaks = [
            point for point in peaks if bounds[0] <= point <= bounds[1]]

        if len(filtered_peaks) == 0:
            # TODO: Adjust POI if there are no Resonance frequency peaks in the bounded region.  Potentially look at something like
            # Difference peaks or dissipation valleys and try to correlate.
            adjusted_point = guess
        else:
            # filtered_peaks = [point for point in peaks if bounds[0] <= point <= bounds[1]]
            filtered_peaks = [point for point in peaks if poi_1_guess <= point]
            # Interpolate between peaks to create the upper envelope
            t = np.arange(len(rf))
            envelope_interpolator = interp1d(
                t[peaks], rf[peaks], kind="linear", fill_value="extrapolate"
            )
            upper_envelope = envelope_interpolator(t)

            # Compute regions of approximately zero slope in the RF curve.
            zsr = find_zero_slope_regions(upper_envelope, threshold=0.00001)
            moved = False
            for region in zsr:
                lb = region[0]
                rb = region[-1]
                if lb <= filtered_peaks[0] <= rb:
                    adjusted_point = filtered_peaks[1]
                    moved = True
                    break
            if not moved:
                adjusted_point = filtered_peaks[0]
        # if abs(actual[3] - adjusted_point) > 50:
        #     fig, ax = plt.subplots()
        #     ax.plot(dissipation, label="Dissipation", color="black")
        #     ax.plot(difference, label="Difference", color="tan")
        #     ax.fill_betweenx(
        #         [0, max(dissipation)], bounds[0], bounds[1], color=f"yellow", alpha=0.1
        #     )
        #     ax.plot(rf, label="rf", color="grey")
        #     ax.plot(upper_envelope, label="upper_envelope", color="brown")
        #     ax.scatter(actual, upper_envelope[actual], color="blue", label="Actual")
        #     ax.scatter(
        #         candidates,
        #         upper_envelope[candidates],
        #         color="red",
        #         label="Candidates",
        #         marker="x",
        #     )
        #     for region in zsr:
        #         ax.axvspan(
        #             region[0],
        #             region[-1],
        #             color="red",
        #             alpha=0.5,
        #             label="Zero Slope Region",
        #         )
        #     ax.scatter(
        #         filtered_peaks,
        #         upper_envelope[filtered_peaks],
        #         color="green",
        #         label="Peaks",
        #     )

        #     ax.axvline(adjusted_point, color="orange", label="adjusted_point")
        #     plt.legend()
        #     plt.show()
        return adjusted_point

    def adjustmet_poi_5(
        self, df, candidates, emp_guess, mm_guess, actual, bounds, poi_1_guess, poi_4_guess, poi_6_guess
    ):
        if emp_guess >= poi_6_guess or emp_guess <= poi_4_guess:
            diss = df["Dissipation"]
            rf = df["Resonance_Frequency"]
            diff = df["Difference"]

            diss_peaks, _ = find_peaks(diss)
            rf_peaks, _ = find_peaks(rf)
            diff_peaks, _ = find_peaks(diff)
            initial_guess = np.array(mm_guess)
            candidate_points = np.array(candidates)
            rf_points = np.array(rf_peaks)
            r_time_1 = df.at[poi_1_guess, 'Relative_time']
            r_time_4 = df.at[poi_4_guess, 'Relative_time']
            initial_fill = r_time_4 - r_time_1
            channel_2_fill_rtime = r_time_1 + (initial_fill * 2.8)
            channel_3_fill_rtime = r_time_1 + (initial_fill * 9)
            channel_2_fill_idx = (df['Relative_time'] -
                                  channel_2_fill_rtime).abs().idxmin()
            channel_3_fill_idx = (df['Relative_time'] -
                                  channel_3_fill_rtime).abs().idxmin()
            diss_points = np.array(diss_peaks)
            diff_points = np.array(diff_peaks)
            np.concatenate((rf_points, diff_points, diss_points))

            x_min, x_max = bounds
            if x_min < poi_4_guess:
                x_min = channel_2_fill_idx
            if x_max > poi_6_guess:
                x_max = channel_3_fill_idx
            candidate_density = len(candidates) / (x_max - x_min)
            adjusted_point = -1
            if candidate_density < 0.01:
                zero_slope = self.find_zero_slope_regions(rf)
                # Filter RF points within the bounds
                # rf_points = np.concatenate((rf_points, diss_points))
                candidates = np.append(candidates, mm_guess)
                within_bounds = (rf_points >= x_min) & (rf_points <= x_max)
                filtered_rf_points = rf_points[within_bounds]

                for l, r in zero_slope:
                    np.append(rf_points, l)
                    np.append(rf_points, r)
                # If no RF points within the bounds, return None or handle accordingly
                if filtered_rf_points.size == 0:
                    return mm_guess

                # Calculate weights for filtered RF points
                weights = np.array([1 for rf_point in filtered_rf_points])

                # Calculate weighted distances from initial guess to each filtered RF point
                distances_to_rf = np.abs(filtered_rf_points - initial_guess)
                weighted_distances = distances_to_rf / weights  # Adjust distance by weight

                # Find the closest RF point to the initial guess, considering weights
                closest_rf_idx = np.argmin(weighted_distances)
                closest_rf_point = filtered_rf_points[closest_rf_idx]

                # Calculate distances from candidate points to the closest RF point
                distances_to_closest_rf = np.abs(
                    candidate_points - closest_rf_point)

                # Find the closest candidate point to the closest RF point
                closest_candidate_idx = np.argmin(distances_to_closest_rf)
                adjusted_point = candidate_points[closest_candidate_idx]

            else:
                adjusted_point = mm_guess
        else:
            adjusted_point = emp_guess

        # if abs(adjusted_point - actual) > 50:
        #     fig, ax = plt.subplots()
        #     ax.plot(diss, color="grey")

        #     ax.fill_betweenx(
        #         [0, max(diss)], poi_1_guess, channel_2_fill_idx, color=f"orange", alpha=0.5
        #     )
        #     ax.fill_betweenx(
        #         [0, max(diss)], poi_1_guess, channel_3_fill_idx, color=f"yellow", alpha=0.5
        #     )
        #     ax.fill_betweenx(
        #         [0, max(diss)], poi_1_guess, poi_4_guess, color=f"red", alpha=0.5
        #     )
        #     ax.scatter(diss_peaks, diss[diss_peaks],
        #                color="red", label="diss peaks")
        #     ax.scatter(diff_peaks, diss[diff_peaks],
        #                color="green", label="diff peaks")
        #     ax.scatter(candidates, diss[candidates],
        #                color="black", label="candidates")
        #     ax.axvline(guess, color="purple",
        #                linestyle="dotted", label="guess")

        #     ax.axvline(adjusted_point, color="brown", label="adjusted")
        #     ax.axvline(actual, color="orange", linestyle="--", label="actual")
        #     ax.scatter(actual, diss[actual], color="orange", marker="*")
        #     plt.legend()
        #     plt.show()
        return adjusted_point

    def adjustment_poi_6(
        self,
        poi_6_guess,
        candidates,
        dissipation,
        difference,
        rf,
        signal,
        t_delta,
        actual,
    ):
        # The following are normalized datasets collected from the raw raun data.
        rf = self.normalize(rf)
        dissipation = self.normalize(dissipation)
        difference = self.normalize(difference)
        signal = self.normalize(signal)

        # Diff peaks are peaks in the difference curve
        diff_peaks, _ = find_peaks(difference)
        # The list of candidates appends the intial guess of POI6 to the list of potential candidates.
        candidates = np.append(candidates, [poi_6_guess])

        def classify_tail(data, a, b):
            slope = data[a] - data[b]
            if slope < 0:
                return "increasing"
            elif slope > 0:
                return "decreasing"
            else:
                # TODO: Implement a method for determining noise at the end of the signal.
                return "noisy"

        # Subtract the difference from dissipation curve to get segments where the normalized difference
        # signal interesects witht the noramlized dissipation signal.
        diff_signal = difference - dissipation
        crossings = np.where(np.diff(np.sign(diff_signal)))[0]

        # Take the last crossing in where the difference and dissipation signal instersect.
        nearest_crossing = max(crossings)

        # Next, get the peak in the difference curve that minimizes the distance between to the final
        # interesction.  This peak indicates the the peak of interest ending the run.
        distances = np.abs(diff_peaks - nearest_crossing)
        nearest_peak = diff_peaks[np.argmin(distances)]

        # The nearest peak and crossing can be out of order so the following ensures that these two points
        # are in ascending order.
        a, b = (
            (nearest_peak, nearest_crossing)
            if nearest_peak < nearest_crossing
            else (nearest_crossing, nearest_peak)
        )
        # Get the type of tail the last 10% of the run classifies as.  10% is an artbirary
        # fraction of the run and can be adjusted.
        tail_class = classify_tail(difference, a, b)
        if tail_class == "increasing":
            # For an increasing tail in the difference curve, currently no adjustment is provided.
            # TODO: For an increasing tail, implelment an adjustment.
            # I think this could be something along the lines of looking at nearest valley in both
            # difference and dissipation and moving poi guess to that but I am not sure.
            filtered_candidates = candidates
            adjusted_poi_6 = poi_6_guess
        elif tail_class == "decreasing":
            # For decreasing tails, there are 2 cases: (1) Between the nearest peak and crossing point,
            # there exists some candidates.  Pick the candidate closest to the base of the increase segment
            # in the dissipation curve. (2) There are no candidates, in which case, pick the point on the
            # dissipation curve which has the most significant change in slope over the baseline slope of
            # that region.
            filtered_candidates = [
                point for point in candidates if a <= point <= b]

            if len(filtered_candidates) > 0:
                adjusted_poi_6 = min(filtered_candidates,
                                     key=lambda x: abs(x - a))
                # adjusted_poi_6 = max(filtered_candidates,
                #                      key=lambda p: signal[p])
                tail_class = tail_class + "_A"
            else:
                slope = np.diff(dissipation[a:b])
                if len(slope) == 0:
                    adjusted_poi_6 = poi_6_guess
                else:
                    average_slope = np.mean(slope)
                    increasing_index = np.argmax(slope > average_slope)
                    significant_point = increasing_index + 1
                    adjusted_poi_6 = significant_point + a
            if crossings[crossings > adjusted_poi_6].size > 0 and t_delta > 0:
                filter_2 = crossings[crossings > adjusted_poi_6]
                adjusted_poi_6 = filter_2[0]
        else:
            # TODO: Noise case
            # The final case is instended to handle the case where the end of the run is noisy.
            filtered_candidates = candidates
            adjusted_poi_6 = poi_6_guess

        # if abs(actual[5] - adjusted_poi_6) > 30:
        #     plt.figure(figsize=(8, 8))
        #     plt.plot(dissipation, label="Dissipation", color="grey")
        #     plt.scatter(
        #         filtered_candidates,
        #         dissipation[filtered_candidates],
        #         color="red",
        #         label="Candidates",
        #         marker="x",
        #     )

        #     plt.plot(difference, label="Difference", color="brown")
        #     plt.plot(rf, label="Resonance frequency", color="tan")

        #     plt.scatter(actual, dissipation[actual],
        #                 label="actual", color="blue")
        #     # plt.scatter(rf_max, dissipation[rf_max], label="rf_peaks", color="green")
        #     plt.scatter(
        #         nearest_peak,
        #         dissipation[nearest_peak],
        #         label="Nearest Peak",
        #         color="yellow",
        #         marker="*",
        #     )
        #     plt.scatter(
        #         crossings,
        #         difference[crossings],
        #         color="red",
        #         label="Crossings",
        #         zorder=5,
        #     )
        #     plt.scatter(
        #         nearest_crossing,
        #         difference[nearest_crossing],
        #         color="yellow",
        #         marker="*",
        #         label="Nearest Crossing",
        #         zorder=5,
        #     )

        #     if t_delta > 0:
        #         plt.axvline(t_delta, label="t_delta",
        #                     color="black", linestyle="dotted")
        #     plt.axvline(adjusted_poi_6, label="poi_6", color="purple")
        #     plt.legend()
        #     plt.title(tail_class)
        #     plt.show()

        return adjusted_poi_6

    def predict(self, file_buffer, run_type=-1, start=-1, stop=-1, act=[None] * 6):
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
        poi_1 = self.adjustment_poi_1(
            initial_guess=start_1, dissipation_data=diss_raw)
        adj_6 = extracted_results[6]
        candidates_6 = self.find_and_sort_peaks(adj_6)

        if diff_raw.mean() < 0:
            poi_6 = start_6
        else:
            t_delta = qdp.find_time_delta()
            poi_6 = self.adjustment_poi_6(
                poi_6_guess=np.argmax(adj_6),
                candidates=candidates_6,
                dissipation=df["Dissipation"],
                difference=df["Difference"],
                rf=df["Resonance_Frequency"],
                signal=adj_6,
                t_delta=t_delta,
                actual=act,
            )
        adj_2, bounds_2 = self.adjust_predictions(
            prediction=extracted_results[2],
            rel_time=rel_time,
            poi_num=2,
            type=run_type,
            i=poi_1,
            j=poi_6,
        )
        adj_3, bounds_3 = self.adjust_predictions(
            prediction=extracted_results[3],
            rel_time=rel_time,
            poi_num=3,
            type=run_type,
            i=poi_1,
            j=poi_6,
        )
        adj_4, bounds_4 = self.adjust_predictions(
            prediction=extracted_results[4],
            rel_time=rel_time,
            poi_num=4,
            type=run_type,
            i=poi_1,
            j=poi_6,
        )
        adj_5, bounds_5 = self.adjust_predictions(
            prediction=extracted_results[5],
            rel_time=rel_time,
            poi_num=5,
            type=run_type,
            i=poi_1,
            j=poi_6,
        )

        candidates_1 = self.find_and_sort_peaks(extracted_results[1])
        candidates_2 = self.find_and_sort_peaks(adj_2)
        candidates_3 = self.find_and_sort_peaks(adj_3)
        candidates_4 = self.find_and_sort_peaks(adj_4)
        candidates_5 = self.find_and_sort_peaks(adj_5)

        # skip adjustment of point 6 when inverted (drop applied to outlet)

        poi_2 = self.adjustment_poi_2(
            initial_guess=start_2,
            dissipation_data=diss_raw,
            bounds=bounds_2,
            poi_1_estimate=poi_1,
        )
        poi_4 = self.adjustmet_poi_4(
            df=df,
            candidates=candidates_4,
            guess=extracted_4,
            actual=act,
            bounds=bounds_4,
            poi_1_guess=poi_1,
        )
        poi_5 = self.adjustmet_poi_5(
            df=df,
            candidates=candidates_5,
            emp_guess=start_5,
            mm_guess=extracted_5,
            actual=act[4],
            bounds=bounds_5,
            poi_1_guess=poi_1,
            poi_4_guess=poi_4,
            poi_6_guess=poi_6,
        )

        if poi_1 >= poi_2:
            poi_1 = adj_1
        poi_3 = np.argmax(adj_3)
        if poi_3 <= poi_2:
            poi_3 = poi_2 + 20

        def sort_and_remove_point(arr, point):
            arr = np.array(arr)
            if len(arr) > MAX_GUESSES - 1:
                arr = arr[: MAX_GUESSES - 1]
            arr.sort()

            return arr[arr != point]

        candidates_list = [
            candidates_1,
            candidates_2,
            candidates_3,
            candidates_4,
            candidates_5,
            candidates_6,
        ]
        poi_list = [poi_1, poi_2, poi_3, poi_4, poi_5, poi_6]
        extracted_confidences = extracted_results[1:7]

        candidates = []

        for i in range(len(poi_list)):

            # Sort and remove point
            candidates_i = sort_and_remove_point(
                candidates_list[i], poi_list[i])
            filtered_points = candidates_i
            if i < 3:
                mean = np.mean(candidates_i)
                std_dev = np.std(candidates_i)
                threshold = 2
                # Filter points within the specified threshold
                filtered_points = [point for point in candidates_i if abs(
                    point - mean) <= threshold * std_dev]

            # Extract and sort confidence
            confidence_i = np.sort(np.array(extracted_confidences[i])[filtered_points])[
                ::-1
            ]

            # Insert POI at the start, remove the last element
            if len(candidates_i) > 1:
                candidates_i = np.insert(candidates_i, 0, poi_list[i])[:-1]
            else:
                candidates_i = np.insert(candidates_i, 0, poi_list[i])

            # Append to candidates list
            candidates.append((candidates_i, confidence_i))
        return candidates
