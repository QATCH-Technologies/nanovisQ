import xgboost as xgb
import pandas as pd
import numpy as np

from scipy.signal import (
    find_peaks,
    peak_widths,
    peak_prominences,
    savgol_filter,
    butter,
    filtfilt,
)
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from hyperopt import STATUS_OK, fmin, hp, tpe

try:
    from QDataPipline import QDataPipeline
except:
    from QATCH.QModel.QDataPipline import QDataPipeline

""" The following are parameters for QModel to use during training time. """
""" The percentage of data to include in the validation set. """
VALID_SIZE = 0.20
""" The percentage of data to include in the test set. """
TEST_SIZE = 0.20
""" The number of folds to train. """
NUMBER_KFOLDS = 5
""" A random seed to set the state of QModel to. """
SEED = 2018
""" The number of rounds to boost for. """
MAX_ROUNDS = 1000
OPT_ROUNDS = 1000
""" Acceptable number of early stopping rounds. """
EARLY_STOP = 50
""" The number of rounds after which to print a verbose model evaluation. """
VERBOSE_EVAL = 50
""" The target supervision feature. """
TARGET_FEATURES = "Class"
""" Training features for the pooling model. """
POOLING_PREDICTORS = [
    "Relative_time",
    "Dissipation",
    "Resonance_Frequency",
    "Peak Magnitude (RAW)",
    "Difference",
]
""" Training features for the discriminator model. """
DISCRIMINATOR_PREDICTORS = [
    "Relative_time",
    "Dissipation",
    "Resonance_Frequency",
    "Peak Magnitude (RAW)",
    "Difference",
    "Pooling",
]


class QModel:
    """
    QModel representing a machine learning model with pooling and discriminator components
    trained using XGBoost.

    This class initializes with a dataset and sets up training, validation, and testing
    splits for both pooling and discriminator models. It allows for hyperparameter tuning
    using Bayesian optimization with Hyperopt, evaluates model performance, and facilitates
    training and scoring of the pooling and discriminator models.

    Attributes:
        __pooling_params__ (dict): Parameters for the pooling model.
        __discriminator_params__ (dict): Parameters for the discriminator model.
        __train_df__ (pd.DataFrame): Training subset of the dataset.
        __valid_df__ (pd.DataFrame): Validation subset of the dataset.
        __test_df__ (pd.DataFrame): Testing subset of the dataset.
        __p_dtrain__ (xgb.DMatrix): DMatrix for the pooling model's training data.
        __p_dvalid__ (xgb.DMatrix): DMatrix for the pooling model's validation data.
        __p_dtest__ (xgb.DMatrix): DMatrix for the pooling model's testing data.
        __d_dtrain__ (xgb.DMatrix): DMatrix for the discriminator model's training data.
        __d_dvalid__ (xgb.DMatrix): DMatrix for the discriminator model's validation data.
        __d_dtest__ (xgb.DMatrix): DMatrix for the discriminator model's testing data.
        __pooler_watchlist__ (list): List of DMatrix objects for pooling model's training
        and validation monitoring.
        __discriminator_watchlist__ (list): List of DMatrix objects for discriminator model's
        training and validation monitoring.
        __pooling_model__ (xgb.Booster or None): Trained pooling model, initially set to None.
        __discriminator_model__ (xgb.Booster or None): Trained discriminator model, initially
        set to None.

    Methods:
        __init__(self, dataset):
            Initializes the model with the provided dataset, setting up data splits and
            initializing model parameters.

        train_pooler(self):
            Trains the pooling model using the defined training data and parameters.

        train_discriminator(self):
            Trains the discriminator model using the defined training data and parameters.

        score_pooler(self, params):
            Trains the pooling model with specified parameters and evaluates its performance
            on the validation dataset.

        tune(self, model, evaluations=250):
            Tunes hyperparameters for either the pooling or discriminator model using Bayesian
            optimization with Hyperopt.

    Notes:
        - This class assumes the usage of XGBoost for model training.
        - Hyperparameters are tuned using Bayesian optimization to improve model performance.
        - Adjustments to default parameters may be necessary for different datasets.

    Example:
        >>> from your_module import YourClassName
        >>> model_instance = YourClassName(dataset)
        >>> model_instance.tune(model='p', evaluations=300)
        -- Best pooling parameters,
            {'eta': 0.3, 'max_depth': 5, 'min_child_weight': 4, ...}

    """

    def __init__(self, dataset):
        """
        Initializes the model with the given dataset, setting up training, validation,
        and testing data splits, and configuring model parameters for pooling and
        discriminator models.

        Args:
            dataset (pd.DataFrame): The dataset to be used for training and evaluation.
            It should contain all necessary features and the target variable.

        Attributes:
            __pooling_params__ (dict): Parameters for the pooling model.
            __discriminator_params__ (dict): Parameters for the discriminator model.
            __train_df__ (pd.DataFrame): Training subset of the dataset.
            __valid_df__ (pd.DataFrame): Validation subset of the dataset.
            __test_df__ (pd.DataFrame): Testing subset of the dataset.
            __p_dtrain__ (xgb.DMatrix): DMatrix for the pooling model's training data.
            __p_dvalid__ (xgb.DMatrix): DMatrix for the pooling model's validation data.
            __p_dtest__ (xgb.DMatrix): DMatrix for the pooling model's testing data.
            __d_dtrain__ (xgb.DMatrix): DMatrix for the discriminator model's training data.
            __d_dvalid__ (xgb.DMatrix): DMatrix for the discriminator model's validation data.
            __d_dtest__ (xgb.DMatrix): DMatrix for the discriminator model's testing data.
            __pooler_watchlist__ (list): List of DMatrix objects for pooling model's training
            and validation monitoring.
            __discriminator_watchlist__ (list): List of DMatrix objects for discriminator model's
            training and validation monitoring.
            __pooling_model__ (xgb.Booster or None): Trained pooling model, initially set to None.
            __discriminator_model__ (xgb.Booster or None): Trained discriminator model, initially
            set to None.

        Notes:
            (06-25-2024) The default parameters for both the pooling and discriminator models are optimized
            for the VOYAGER dataset. Adjustments may be necessary for other datasets.
        """
        self.__pooling_params__ = {
            "objective": "binary:logistic",
            "eta": 0.375,
            "max_depth": 7,
            "min_child_weight": 2.0,
            "subsample": 0.95,
            "colsample_bytree": 0.7,
            "gamma": 0.65,
            "eval_metric": "auc",
            "nthread": 12,
            "booster": "gbtree",
            "device": "cuda",
            "tree_method": "hist",
            "seed": SEED,
        }
        self.__discriminator_params__ = {
            "objective": "binary:logistic",
            "eta": 0.475,
            "max_depth": 7,
            "min_child_weight": 2.0,
            "subsample": 0.7,
            "colsample_bytree": 0.55,
            "gamma": 0.7,
            "eval_metric": "auc",
            "nthread": 12,
            "booster": "gbtree",
            "device": "cuda",
            "tree_method": "hist",
            "seed": SEED,
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

        self.__p_dtrain__ = xgb.DMatrix(
            self.__train_df__[POOLING_PREDICTORS],
            label=self.__train_df__[TARGET_FEATURES].values,
        )
        self.__p_dvalid__ = xgb.DMatrix(
            self.__valid_df__[POOLING_PREDICTORS],
            label=self.__valid_df__[TARGET_FEATURES].values,
        )
        self.__p_dtest__ = xgb.DMatrix(
            self.__test_df__[POOLING_PREDICTORS],
            label=self.__test_df__[TARGET_FEATURES].values,
        )
        self.__d_dtrain__ = xgb.DMatrix(
            self.__train_df__[DISCRIMINATOR_PREDICTORS],
            label=self.__train_df__[TARGET_FEATURES].values,
        )
        self.__d_dvalid__ = xgb.DMatrix(
            self.__valid_df__[DISCRIMINATOR_PREDICTORS],
            label=self.__valid_df__[TARGET_FEATURES].values,
        )
        self.__d_dtest__ = xgb.DMatrix(
            self.__test_df__[DISCRIMINATOR_PREDICTORS],
            label=self.__test_df__[TARGET_FEATURES].values,
        )

        self.__pooler_watchlist__ = [
            (self.__p_dtrain__, "train"),
            (self.__p_dvalid__, "valid"),
        ]
        self.__discriminator_watchlist__ = [
            (self.__d_dtrain__, "train"),
            (self.__d_dvalid__, "valid"),
        ]

        self.__pooling_model__ = None
        self.__discriminator_model__ = None

    def train_pooler(self):
        """
        Trains the pooling model using the training dataset and specified parameters.

        This method sets up and trains an XGBoost model with the parameters defined
        in `__pooling_params__`. The training process monitors the performance on the
        validation dataset to apply early stopping if necessary.

        Attributes:
            __pooling_model__ (xgb.Booster): The trained pooling model.

        Trains the pooling model using:
            - `__p_dtrain__` (xgb.DMatrix): Training data for the pooling model.
            - `__pooler_watchlist__` (list): List of DMatrix objects for training
            and validation monitoring.

        Training Parameters:
            - `MAX_ROUNDS` (int): Maximum number of boosting rounds.
            - `EARLY_STOP` (int): Rounds to perform early stopping if validation
            metric does not improve.
            - `VERBOSE_EVAL` (bool or int): Controls the verbosity of the training
            process, indicating how often to print evaluation messages.

        Returns:
            None
        """
        self.__pooling_model__ = xgb.train(
            self.__pooling_params__,
            self.__p_dtrain__,
            MAX_ROUNDS,
            self.__pooler_watchlist__,
            early_stopping_rounds=EARLY_STOP,
            maximize=True,
            verbose_eval=VERBOSE_EVAL,
        )

    def train_discriminator(self):
        """
        Trains the discriminator model using the training dataset and specified parameters.

        This method sets up and trains an XGBoost model with the parameters defined
        in `__discriminator_params__`. The training process monitors the performance on the
        validation dataset to apply early stopping if necessary.

        Attributes:
            __discriminator_model__ (xgb.Booster): The trained discriminator model.

        Trains the discriminator model using:
            - `__d_dtrain__` (xgb.DMatrix): Training data for the discriminator model.
            - `__discriminator_watchlist__` (list): List of DMatrix objects for training
            and validation monitoring.

        Training Parameters:
            - `MAX_ROUNDS` (int): Maximum number of boosting rounds.
            - `EARLY_STOP` (int): Rounds to perform early stopping if validation
            metric does not improve.
            - `VERBOSE_EVAL` (bool or int): Controls the verbosity of the training
            process, indicating how often to print evaluation messages.

        Returns:
            None
        """
        self.__discriminator_model__ = xgb.train(
            self.__discriminator_params__,
            self.__d_dtrain__,
            MAX_ROUNDS,
            self.__discriminator_watchlist__,
            early_stopping_rounds=EARLY_STOP,
            maximize=True,
            verbose_eval=VERBOSE_EVAL,
        )

    def score_pooler(self, params):
        """
        Trains the pooling model with given parameters and evaluates its performance on the
        validation dataset.

        This method trains an XGBoost model with the provided parameters, then makes predictions
        on the validation set. It calculates the ROC AUC score and returns the loss and status.

        Args:
            params (dict): Parameters for training the pooling model.

        Returns:
            dict: A dictionary containing the loss and status.
                - "loss" (float): The loss computed as 1 - ROC AUC score.
                - "status" (str): Status of the evaluation, typically "ok".

        Trains the pooling model using:
            - `params` (dict): Parameters provided for the pooling model training.
            - `__p_dtrain__` (xgb.DMatrix): Training data for the pooling model.
            - `__pooler_watchlist__` (list): List of DMatrix objects for training
            and validation monitoring.

        Evaluation:
            - Predictions are made on `__p_dvalid__` (xgb.DMatrix): Validation data for
            the pooling model.
            - `roc_auc_score` is used to compute the AUC score from the predictions.

        Training Parameters:
            - `MAX_ROUNDS` (int): Maximum number of boosting rounds.
            - `EARLY_STOP` (int): Rounds to perform early stopping if validation
            metric does not improve.
            - `VERBOSE_EVAL` (bool or int): Controls the verbosity of the training
            process, indicating how often to print evaluation messages.
        """
        self.__pooling_model__ = xgb.train(
            params,
            self.__p_dtrain__,
            MAX_ROUNDS,
            self.__pooler_watchlist__,
            early_stopping_rounds=EARLY_STOP,
            maximize=True,
            verbose_eval=VERBOSE_EVAL,
        )
        predictions = self.__pooling_model__.predict(
            self.__p_dvalid__,
            iteration_range=range(0, self.__pooling_model__.best_iteration + 1),
        )
        score = roc_auc_score(self.__p_dvalid__.get_label(), predictions)
        loss = 1 - score
        return {"loss": loss, "status": STATUS_OK}

    def score_discrimintator(self, params):
        """
        Trains the discriminator model with given parameters and evaluates its performance on the
        validation dataset.

        This method trains an XGBoost model with the provided parameters, then makes predictions
        on the validation set. It calculates the ROC AUC score and returns the loss and status.

        Args:
            params (dict): Parameters for training the discriminator model.

        Returns:
            dict: A dictionary containing the loss and status.
                - "loss" (float): The loss computed as 1 - ROC AUC score.
                - "status" (str): Status of the evaluation, typically "ok".

        Trains the discriminator model using:
            - `params` (dict): Parameters provided for the discriminator model training.
            - `__d_dtrain__` (xgb.DMatrix): Training data for the pooling model.
            - `__discriminator_watchlist` (list): List of DMatrix objects for training
            and validation monitoring.

        Evaluation:
            - Predictions are made on `__d_dvalid__` (xgb.DMatrix): Validation data for
            the discriminator model.
            - `roc_auc_score` is used to compute the AUC score from the predictions.

        Training Parameters:
            - `MAX_ROUNDS` (int): Maximum number of boosting rounds.
            - `EARLY_STOP` (int): Rounds to perform early stopping if validation
            metric does not improve.
            - `VERBOSE_EVAL` (bool or int): Controls the verbosity of the training
            process, indicating how often to print evaluation messages.
        """
        self.__discriminator_model__ = xgb.train(
            params,
            self.__d_dtrain__,
            MAX_ROUNDS,
            self.__discriminator_watchlist__,
            early_stopping_rounds=EARLY_STOP,
            maximize=True,
            verbose_eval=VERBOSE_EVAL,
        )
        predictions = self.__discriminator_model__.predict(
            self.__d_dvalid__,
            iteration_range=range(0, self.__discriminator_model__.best_iteration + 1),
        )
        score = roc_auc_score(self.__d_dvalid__.get_label(), predictions)
        loss = 1 - score
        return {"loss": loss, "status": STATUS_OK}

    def tune(self, model, evaluations=250):
        """
        Tunes hyperparameters for the specified model using Bayesian optimization with Hyperopt.

        This method searches for the best hyperparameters for either the pooling or
        discriminator model based on the provided model type. The optimization process
        uses the Tree-structured Parzen Estimator (TPE) algorithm to minimize the loss
        computed during model evaluation.

        Args:
            model (str): The model to tune. Should be 'p' for pooling or 'd' for discriminator.
            evaluations (int, optional): The number of evaluations for the Hyperopt optimization.
            Defaults to 250.

        Returns:
            dict: The best hyperparameters found during the tuning process.

        Raises:
            ValueError: If an invalid model type is provided.

        Hyperparameter Space:
            - eta: Learning rate, selected uniformly from [0.025, 0.5] with a step of 0.025.
            - max_depth: Maximum tree depth, selected from integers in the range [1, 13].
            - min_child_weight: Minimum sum of instance weight needed in a child, selected uniformly from [1, 6] with a step of 1.
            - subsample: Subsample ratio of the training instances, selected uniformly from [0.5, 1] with a step of 0.05.
            - gamma: Minimum loss reduction required to make a further partition, selected uniformly from [0.5, 1] with a step of 0.05.
            - colsample_bytree: Subsample ratio of columns when constructing each tree, selected uniformly from [0.5, 1] with a step of 0.05.
            - eval_metric: Evaluation metric used, set to "auc".
            - objective: Learning task and objective, set to "binary:logistic".
            - nthread: Number of parallel threads used to run XGBoost, set to 12.
            - booster: Boosting algorithm to use, set to "gbtree".
            - device: Device to use for computation, set to "cuda".
            - tree_method: Tree construction algorithm used, set to "hist".
            - seed: Random seed for reproducibility.

        Example:
            To tune the pooling model:
            >>> best_params = model_instance.tune(model='p', evaluations=300)
        """
        space = {
            "eta": hp.quniform("eta", 0.025, 0.5, 0.025),
            "max_depth": hp.choice("max_depth", np.arange(1, 14, dtype=int)),
            "min_child_weight": hp.quniform("min_child_weight", 1, 6, 1),
            "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
            "gamma": hp.quniform("gamma", 0.5, 1, 0.05),
            "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.05),
            "eval_metric": "auc",
            "objective": "binary:logistic",
            "nthread": 12,
            "booster": "gbtree",
            "device": "cuda",
            "tree_method": "hist",
            "seed": SEED,
        }
        if model == "p":
            best = fmin(
                self.score_pooler,
                space,
                algo=tpe.suggest,
                max_evals=evaluations,
            )
            self.__pooling_params__ = best
            print(f"-- best pooling parameters, \n\t{best}")
        if model == "d":
            best = fmin(
                self.score_discrimintator,
                space,
                algo=tpe.suggest,
                max_evals=evaluations,
            )
            self.__discriminator_params__ = best
            print(f"-- best discriminator parameters, \n\t{best}")
        return best

    def save_pooler(self, model_name="QModelPooler"):
        """
        Saves the trained model to a file.

        Args:
            model_name (str, optional): Name of the model file. Defaults to "QModel".
        """
        filename = f"QModel/SavedModels/{model_name}.json"
        self.__pooling_model__.save_model(filename)

    def save_discriminator(self, model_name="QModelDiscriminator"):
        """
        Saves the trained model to a file.

        Args:
            model_name (str, optional): Name of the model file. Defaults to "QModel".
        """
        filename = f"QModel/SavedModels/{model_name}.json"
        self.__discriminator_model__.save_model(filename)

    def get_pooler(self):
        """
        Returns the trained pooler model.

        Returns:
            Booster: The trained XGBoost model.
        """
        return self.__pooling_model__

    def get_discriminator(self):
        """
        Returns the trained discriminator model.

        Returns:
            Booster: The trained XGBoost model.
        """
        return self.__discriminator_model__


class QModelPredict:
    def __init__(self, pooler_path=None, discriminator_path=None):
        """
        Initializes the QModelPredict object with pre-trained models loaded from specified paths.

        Args:
            pooler_path (str, optional): Path to the pre-trained pooling model file. Defaults to None.
            discriminator_path (str, optional): Path to the pre-trained discriminator model file. Defaults to None.

        Raises:
            ValueError: If either `pooler_path` or `discriminator_path` is not provided.

        Attributes:
            __pooler__ (xgb.Booster): Pre-trained XGBoost pooling model.
            __discriminator__ (xgb.Booster): Pre-trained XGBoost discriminator model.

        Notes:
            - This class assumes that the models are saved using XGBoost's native model format.
            - Both `pooler_path` and `discriminator_path` must be valid paths to the model files.

        Example:
            >>> from your_module import QModelPredict
            >>> predictor = QModelPredict(pooler_path='path/to/pooler_model.json', discriminator_path='path/to/discriminator_model.json')

        """
        if pooler_path is None or discriminator_path is None:
            raise ValueError("[QModelPredict __init__()] No model path given")

        # Load the pre-trained models from the specified paths
        self.__pooler__ = xgb.Booster()
        self.__discriminator__ = xgb.Booster()

        self.__pooler__.load_model(pooler_path)
        self.__discriminator__.load_model(discriminator_path)

    def histogram_analysis(self, data, num_bins, threshold):
        """
        Performs histogram analysis on the given data to adjust the number of bins until a specified
        threshold of non-zero histogram counts is reached.

        Args:
            data (array-like): The data to analyze.
            num_bins (int): Initial number of bins for the histogram.
            threshold (int): Desired number of non-zero histogram counts.

        Returns:
            tuple: A tuple containing:
                - hist (np.ndarray): Histogram counts.
                - bins (np.ndarray): Bin edges.

        Raises:
            ValueError: If any of the following conditions are met:
                        - `num_bins` is less than or equal to 0.
                        - `threshold` is less than or equal to 0.
                        - `num_bins` is greater than the length of `data`.

        Notes:
            - Adjusts `num_bins` iteratively until `threshold` non-zero histogram counts are achieved.
            - Uses NumPy's histogram function to compute the histogram.

        Example:
            >>> from QModel import QPredict
            >>> analyzer = QPredict()
            >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> hist, bins = analyzer.histogram_analysis(data, num_bins=10, threshold=5)
        """
        # Validate inputs
        if num_bins <= 0:
            raise ValueError(
                "[QModelPredict histrogram_analysis()]: `num_bins` must be greater than zero."
            )
        if threshold <= 0:
            raise ValueError(
                "[QModelPredict histrogram_analysis()]: `threshold` must be greater than zero."
            )
        if num_bins > len(data):
            raise ValueError(
                "[QModelPredict histrogram_analysis()]: `num_bins` cannot be greater than the length of `data`."
            )

        # Make initial partition of data into bins.
        hist, bins = np.histogram(data, bins=num_bins)

        # While the number of bins is less than the threshold number of bins (typically 3), continue
        # creating partitions until there are at least a threshold number of bins with values in them.
        # If there the number of bins gets decremented <= 0, the current number of bins is returned to
        # the caller.
        while np.count_nonzero(hist) != threshold:
            num_bins -= 1
            if num_bins <= 0:
                break
            hist, bins = np.histogram(data, bins=num_bins)
        return hist, bins

    def normalize(self, data):
        """
        Normalizes the given data to the range [0, 1].

        Args:
            data (array-like): The data to be normalized.

        Returns:
            np.ndarray: The normalized data.

        Raises:
            ValueError: If `data` is not a valid array-like object.

        Notes:
            - Uses NumPy to compute the normalization.

        Example:
            >>> from QModel import QPredict
            >>> precitions = QPredict()
            >>> data = [1, 2, 3, 4, 5]
            >>> normalized_data = precitions.normalize(data)
        """
        try:
            data = np.array(data)  # Ensure data is converted to a NumPy array
            if data.size == 0:
                raise ValueError(
                    "[QModelPredict normalize()]: `data` cannot be an empty array."
                )
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            return normalized_data
        except Exception as e:
            raise ValueError(
                "[QModelPredict normalize()]: Invalid `data` input for normalization."
            ) from e

    def full_bins(self, bins, hist):
        """
        Retrieves bins with non-zero histogram counts from the given bins and histogram.

        Args:
            bins (np.ndarray): Array of bin edges.
            hist (np.ndarray): Array of histogram counts corresponding to `bins`.

        Returns:
            list: List of tuples containing (start_bin, end_bin) for bins with non-zero counts.

        Raises:
            ValueError: If `bins` and `hist` are not of compatible shapes or if `bins` or `hist` are empty.

        Notes:
            - Assumes `bins` and `hist` are numpy arrays of the same length.
            - Filters out bins where `hist` value is zero.

        Example:
            >>> from QModel import QPredict
            >>> analyzer = QPredict()
            >>> bins = np.array([0, 1, 2, 3])
            >>> hist = np.array([0, 5, 0])
            >>> non_zero_bins = analyzer.full_bins(bins, hist)
        """
        try:
            if not isinstance(bins, np.ndarray) or not isinstance(hist, np.ndarray):
                raise ValueError(
                    "[QModelPredict full_bins()]: `bins` and `hist` must be numpy arrays."
                )
            if bins.size == 0 or hist.size == 0:
                raise ValueError(
                    "[QModelPredict full_bins()]: `bins` and `hist` cannot be empty arrays."
                )

            full_bins = [
                (int(bins[i]), int(bins[i + 1]))
                for i, height in enumerate(hist)
                if height > 0
            ]
            return full_bins
        except Exception as e:
            raise ValueError(
                "[QModelPredict full_bins()]: Invalid `bins` or `hist` input for `full_bins` function."
            ) from e

    def find_max_indices(self, data, ranges):
        """
        Finds the indices of maximum values in sub-arrays defined by the given ranges within the data.

        Args:
            data (array-like): The data to search for maximum values.
            ranges (list of tuples): List of tuples (start, end) defining index ranges in `data`.

        Returns:
            list: List of indices of maximum values within each range.

        Raises:
            ValueError: If `data` is not a valid array-like object or if `ranges` contains invalid tuples.

        Notes:
            - Uses NumPy's argmax function to find the index of the maximum value in each sub-array.
            - Adjusts the index to be in the context of the original `data`.

        Example:
            >>> from QModel import QPredict
            >>> analyzer = QPredict()
            >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> ranges = [(0, 4), (5, 9)]
            >>> max_indices = analyzer.find_max_indices(data, ranges)
        """
        try:
            # Validate `data`
            data = np.array(data)
            if data.size == 0:
                raise ValueError(
                    "[QModelPredict find_max_indices()]: `data` cannot be an empty array."
                )

            # Validate `ranges`
            if not isinstance(ranges, list):
                raise ValueError(
                    "[QModelPredict find_max_indices()]: `ranges` must be a list of tuples."
                )
            for range_tuple in ranges:
                if not isinstance(range_tuple, tuple) or len(range_tuple) != 2:
                    raise ValueError(
                        "[QModelPredict find_max_indices()]: `ranges` must be a list of tuples (start, end)."
                    )
                start, end = range_tuple
                if not (
                    isinstance(start, (int, np.integer))
                    and isinstance(end, (int, np.integer))
                ):
                    raise ValueError(
                        "[QModelPredict find_max_indices()]: Values in each tuple of `ranges` must be integers."
                    )
                if start < 0 or end >= len(data) or start > end:
                    raise ValueError(
                        "[QModelPredict find_max_indices()]: Invalid range (start, end) in `ranges`."
                    )

            # Find max indices
            max_indices = []
            for range_tuple in ranges:
                start, end = range_tuple
                sub_array = data[start : end + 1]
                max_index = np.argmax(sub_array)
                max_index_in_data = start + max_index
                max_indices.append(max_index_in_data)

            return max_indices
        except Exception as e:
            raise ValueError(
                "[QModelPredict find_max_indices()]: Invalid input for `find_max_indices` function."
            ) from e

    def noise_filter(self, data):
        """
        Applies a Butterworth low-pass filter to the given data.

        Args:
            data (array-like): The data to filter.

        Returns:
            np.ndarray: Filtered data.

        Raises:
            ValueError: If `data` is not a valid array-like object.

        Notes:
            - Uses scipy's Butterworth filter implementation (`butter` and `filtfilt` functions).
            - Assumes `predictions` is a 1-dimensional array suitable for filtering.

        Example:
            >>> from QModel import QPredict
            >>> processor = QPredict()
            >>> predictions = [0.1, 0.2, 0.3, 0.4, 0.5]
            >>> filtered_predictions = processor.noise_filter(predictions)
        """
        try:
            # Validate `predictions`
            data = np.array(data)
            if data.size == 0:
                raise ValueError(
                    "[QModelPredict noise_filter()]: `predictions` cannot be an empty array."
                )

            # Butterworth low-pass filter parameters
            fs = 20
            normal_cutoff = 2 / (0.5 * fs)

            # Get the filter coefficients
            b, a = butter(2, normal_cutoff, btype="low", analog=False)

            # Apply the filter using filtfilt
            filtered_predictions = filtfilt(b, a, data)

            return filtered_predictions
        except Exception as e:
            raise ValueError(
                "[QModelPredict noise_filter()]: Invalid input for `noise_filter` function."
            ) from e

    def find_top_peaks(self, y, num_peaks=3, prominence=0.1, height=0.1):
        """
        Finds the top peaks in the given signal `y` based on their prominence and height.

        Args:
            y (array-like): The input signal.
            num_peaks (int, optional): Number of top peaks to find. Defaults to 3.
            prominence (float, optional): Minimum prominence of peaks. Defaults to 0.1.
            height (float, optional): Minimum height of peaks. Defaults to 0.1.

        Returns:
            tuple: A tuple containing:
                - top_peaks (np.ndarray): Indices of the top peaks in `y`.
                - top_properties (dict): Dictionary containing properties of the top peaks.

        Raises:
            ValueError: If any of the following conditions are met:
                        - `y` is not a valid array-like object.
                        - `num_peaks` is not a positive integer.
                        - `prominence` or `height` are not non-negative floats.

        Notes:
            - Uses scipy's `find_peaks` function to detect peaks in `y`.
            - Computes a combined score for each peak based on its height and prominence.
            - Returns the indices of the top `num_peaks` peaks along with their properties.

        Example:
            >>> from QModel import QPredict
            >>> analyzer = QPredict()
            >>> y = [0.1, 0.5, 0.2, 0.8, 0.3, 0.7]
            >>> top_peaks, top_properties = analyzer.find_top_peaks(y, num_peaks=2, prominence=0.2, height=0.2)
        """
        try:
            # Validate input signal `y`
            y = np.asarray(y)
            if y.size == 0:
                raise ValueError(
                    "[QModelPredict find_top_peaks()]: `y` cannot be an empty array."
                )

            # Validate `num_peaks`
            if not isinstance(num_peaks, int) or num_peaks <= 0:
                raise ValueError(
                    "[QModelPredict find_top_peaks()]: `num_peaks` must be a positive integer."
                )
            # Find peaks with their properties

            peaks, properties = find_peaks(y, prominence=prominence, height=height)
            while len(peaks) < num_peaks:
                height -= 0.01
                peaks, properties = find_peaks(y, prominence=prominence, height=height)

            prominences = peak_prominences(y, peaks)[0]
            widths = peak_widths(y, peaks, rel_height=0.5)[0]
            heights = y[peaks]

            # Define the weights for height, prominence, and width
            w_h = 0.5
            w_p = 0.25
            w_w = 0.25

            # Normalize the properties
            H_min, H_max = heights.min(), heights.max()
            P_min, P_max = prominences.min(), prominences.max()
            W_min, W_max = widths.min(), widths.max()

            # Compute the scores for each peak
            scores = (
                w_h * (heights - H_min) / (H_max - H_min)
                + w_p * (prominences - P_min) / (P_max - P_min)
                + w_w * (widths - W_min) / (W_max - W_min)
            )

            # Find the indices of the top `num_peaks` peaks with the highest scores
            if len(peaks) > 0:
                top_peak_indices = np.argsort(scores)[-min(num_peaks, len(peaks)) :]
            else:
                top_peak_indices = []

            # Extract the top peaks and their properties
            top_peaks = peaks[top_peak_indices]
            top_properties = {
                key: properties[key][top_peak_indices] for key in properties
            }

            return top_peaks, top_properties

        except Exception as e:
            raise ValueError(
                "[QModelPredict find_top_peaks()]: Invalid input for `find_top_peaks` function."
            ) from e

    def find_start_stop(self, data):
        """
        Finds the start and stop indices of a critical region in the normalized data.

        Args:
            data (array-like): The input data to analyze.

        Returns:
            tuple: A tuple containing the start and stop indices of the critical region.

        Raises:
            ValueError: If `data` is not a valid array-like object or if `data` is empty.

        Notes:
            - Normalizes `data` to facilitate slope calculations.
            - Uses a Savitzky-Golay filter for smoothing the normalized data.
            - Determines the start of the critical region based on significant positive slope changes.
            - Determines the stop of the critical region based on significant negative slope changes.

        Example:
            >>> from QModel import QModelPredict
            >>> analyzer = QModelPredict()
            >>> data = [0.1, 0.2, 0.3, 0.4, 0.5]
            >>> start_index, stop_index = analyzer.find_start_stop(data)
        """
        # Validate `data`
        data = np.asarray(data)
        if data.size == 0:
            raise ValueError(
                "[QModelPredict find_start_stop()]: `data` cannot be an empty array."
            )
        data = self.normalize(data)
        savgol_filter(data, int(len(data) * 0.01), 1)

        # The start of the critical region for initial fill.
        start = 0
        # Slopes to determine the start of this region.
        start_slopes = []

        # A buffer for the first 1% of the input data set.
        # Ensures any sensor interference does not affect the slope
        # calculation.
        buffer = int(len(data) * 0.01)
        start_slopes = np.where(
            np.arange(start + 1, len(data)) < buffer,
            0,
            (data[start + 1 :] - data[start]) / np.arange(start + 1, len(data) - start),
        )

        # Compute where there is a significant positive change in the start slopes.
        # This index gets returned as the start index of the critical region.
        start_slopes = self.normalize(start_slopes)
        start_tmp = 1
        for i in range(1, len(start_slopes)):
            if start_slopes[i] < start_slopes[start] + 0.1:
                start_tmp = i

        # Compute where there is a significant negative change in the stop slopes.
        # This index gets returned as the stop index of the critical region.
        stop_tmp = start_tmp + buffer
        stop_slopes = [0]
        for i in range(start_tmp, len(start_slopes)):
            if not (np.isnan(start_slopes[i]) or np.isnan(start_slopes[start_tmp])):
                stop_slopes.append(
                    (start_slopes[i] - start_slopes[start_tmp]) / (i - start_tmp)
                )
            if stop_slopes[-1] < stop_slopes[-2] - np.mean(stop_slopes):
                stop_tmp = i
                break
        print(start)
        return (start + start_tmp - buffer, start + stop_tmp)

    def predict(self, file_buffer):
        """
        Predicts and interprets points of interest (POI) in the data provided via file_buffer.

        Args:
            file_buffer (file-like object): A buffer containing the CSV data for prediction.

        Returns:
            tuple: A tuple containing:
                - pois (list): List of points of interest indices.
                - pooling_results (np.ndarray): Normalized and filtered pooling predictions.
                - predictions (np.ndarray): Combined normalized predictions from the pooler and discriminator.
                - stop_bound (int): Stop index of the critical region.
                - start_bound (int): Start index of the critical region.

        Raises:
            ValueError: If the required columns are missing in the input data.
            Exception: If any error occurs during the prediction process.

        Notes:
            - Reads CSV data from file_buffer, dropping unnecessary columns.
            - Uses QDataPipeline to compute the difference and get the processed dataframe.
            - Generates predictions using the pooler and discriminator models.
            - Identifies the critical region start and stop indices based on dissipation values.
            - Finds peaks to the right of the critical region and performs histogram analysis.
            - Combines peaks found in and around the critical region to identify points of interest.

        Example:
            >>> with open('data.csv', 'rb') as f:
            >>>     pois, pooling_results, predictions, stop_bound, start_bound = your_class_instance.predict(f)
        """
        try:
            # Load CSV data and drop unnecessary columns
            pooler_df = pd.read_csv(file_buffer)
            columns_to_drop = ["Date", "Time", "Ambient", "Temperature"]
            if not all(col in pooler_df.columns for col in columns_to_drop):
                raise ValueError(
                    f"[QModelPredict predict()]: Input data must contain the following columns: {columns_to_drop}"
                )

            pooler_df = pooler_df.drop(columns=columns_to_drop)

            file_buffer_2 = file_buffer
            if not isinstance(file_buffer_2, str):
                if hasattr(file_buffer_2, "seekable") and file_buffer_2.seekable():
                    file_buffer_2.seek(0)  # reset ByteIO buffer to beginning of stream
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
            qdp.compute_difference()
            pooler_df = qdp.get_dataframe()

            # Ensure feature names match for pooling predictions
            f_names = self.__pooler__.feature_names
            pooler_df = pooler_df[f_names]
            pooling_data = xgb.DMatrix(pooler_df)

            # Generate pooling predictions
            pooling_results = self.__pooler__.predict(pooling_data)
            pooling_results = self.normalize(pooling_results)
            pooling_results = self.noise_filter(pooling_results)

            # Prepare discriminator data
            discriminator_df = pooler_df.copy()
            discriminator_df["Pooling"] = pooling_results
            discriminator_data = xgb.DMatrix(discriminator_df)

            # Generate discriminator predictions
            discriminator_results = self.__discriminator__.predict(discriminator_data)
            discriminator_results = self.normalize(discriminator_results)

            # Combine predictions
            predictions = np.add(discriminator_results, pooling_results)

            # Find critical region start and stop bounds
            padding = int(len(pooler_df["Dissipation"].values) * 0.01)
            start_bound, stop_bound = self.find_start_stop(
                pooler_df["Dissipation"].values[padding:]
            )
            stop_bound += padding

            # Find peaks to the right of the critical region
            r_peaks, _ = find_peaks(
                predictions[stop_bound:], height=predictions[stop_bound:].mean()
            )
            r_peaks += stop_bound

            # Perform histogram analysis on peaks to get regions for POI
            hist, bins = self.histogram_analysis(
                data=r_peaks, num_bins=len(r_peaks), threshold=3
            )
            bin_edges = self.full_bins(bins, hist)
            # Fall back on KMeans if the bin edges cannot be idetified using a histogram.
            if len(bin_edges) < 3:
                kmeans = KMeans(n_clusters=3, random_state=42)
                kmeans.fit(r_peaks.reshape(-1, 1))
                labels = kmeans.labels_
                clusters = {
                    i: r_peaks[labels == i].flatten().tolist()
                    for i in range(kmeans.n_clusters)
                }
                bin_edges = [
                    (min(cluster), max(cluster)) for cluster in clusters.values()
                ]

            r_peaks = []
            for bin in bin_edges:
                peak, _ = self.find_top_peaks(
                    predictions[bin[0] : bin[1]],
                    num_peaks=1,
                    prominence=0,
                    height=predictions[bin[0] : bin[1]].mean(),
                )
                r_peaks.append(peak[0] + bin[0])

            # Find peaks within the critical region
            l_peaks, _ = self.find_top_peaks(
                y=predictions[start_bound : stop_bound + 1],
                height=predictions[start_bound : stop_bound + 1].mean(),
                prominence=0,
            )
            l_peaks += start_bound

            # Combine and sort peaks
            peaks = np.concatenate((l_peaks, r_peaks))
            pois = sorted(set(map(int, peaks.flatten())))

            return pois, pooling_results, predictions, stop_bound, start_bound

        except ValueError as ve:
            raise ve
        except Exception as e:
            raise Exception(
                "[QModelPredict find_top_peaks()]: An error occurred during the prediction process."
            ) from e
