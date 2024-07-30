import multiprocessing
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.early_stop import no_progress_loss
from scipy.signal import (
    butter,
    filtfilt,
    find_peaks,
    peak_prominences,
    peak_widths,
    savgol_filter,
)
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Get the number of threads
NUM_THREADS = multiprocessing.cpu_count()
print(f"[INFO] Available {NUM_THREADS} threads.")

np.set_printoptions(threshold=sys.maxsize)

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
        from QDataPipline import QDataPipeline
    QDataPipeline_found = True
except:
    QDataPipeline_found = False
try:
    if not QDataPipeline_found:
        from QModel.QDataPipline import QDataPipeline
    QDataPipeline_found = True
except:
    QDataPipeline_found = False
try:
    if not QDataPipeline_found:
        from QATCH.QModel.QDataPipline import QDataPipeline
    QDataPipeline_found = True
except:
    QDataPipeline_found = False
if not QDataPipeline_found:
    raise ImportError("Cannot find 'QDataPipeline' in any expected location.")

""" The following are parameters for QModel to use during training time. """
""" The percentage of data to include in the validation set. """
VALID_SIZE = 0.20
""" The percentage of data to include in the test set. """
TEST_SIZE = 0.20
""" The number of folds to train. """
NUMBER_KFOLDS = 5
""" A random seed to set the state of QModel to. """
SEED = 42
""" The number of rounds to boost for. """
MAX_ROUNDS = 1000
OPT_ROUNDS = 1000
""" Acceptable number of early stopping rounds. """
EARLY_STOP = 50
""" The number of rounds after which to print a verbose model evaluation. """
VERBOSE_EVAL = 50
""" The target supervision feature. """
DISTANCES = [1.15, 1.61, 2.17, 5.00, 10.90, 16.2]


class QModel:
    def __init__(self, dataset, predictors, target_features):
        self.__params__ = {
            "objective": "binary:logistic",
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

    def train_model(self):
        self.__model__ = xgb.train(
            self.__params__,
            self.__dtrain__,
            MAX_ROUNDS,
            evals=self.__watchlist__,
            early_stopping_rounds=EARLY_STOP,
            maximize=True,
            verbose_eval=VERBOSE_EVAL,
        )

    def objective(self, params):
        results = xgb.cv(
            params,
            self.__dtrain__,
            MAX_ROUNDS,
            nfold=NUMBER_KFOLDS,
            stratified=True,
            early_stopping_rounds=20,
            metrics=[
                "auc",
                # "aucpr",
            ],
        )
        best_score = results["test-auc-mean"].max()
        return {"loss": best_score, "status": STATUS_OK}

    def tune(self, evaluations=250):
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
            "objective": "binary:logistic",
            "nthread": NUM_THREADS,
            "booster": "gbtree",
            "device": "cuda",
            "tree_method": "auto",
            "sampling_method": "gradient_based",
            "seed": SEED,
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

    def save_model(self, model_name="QModel"):
        filename = f"QModel/SavedModels/{model_name}.json"
        print(f"[INFO] Saving model {model_name}")
        self.__model__.save_model(filename)

    def get_model(self):
        return self.__model__


class QPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            raise ValueError("[QModelPredict __init__()] No model path given")

        # Load the pre-trained models from the specified paths
        self.__model__ = xgb.Booster()

        self.__model__.load_model(model_path)

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

    def predict(self, file_buffer):
        # Load CSV data and drop unnecessary columns
        df = pd.read_csv(file_buffer)
        columns_to_drop = ["Date", "Time", "Ambient", "Temperature"]
        if not all(col in df.columns for col in columns_to_drop):
            raise ValueError(
                f"[QModelPredict predict()]: Input data must contain the following columns: {columns_to_drop}"
            )

        df = df.drop(columns=columns_to_drop)

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
        rel_time = qdp.__dataframe__["Relative_time"]
        qdp.preprocess(poi_file=None)
        df = qdp.get_dataframe()

        f_names = self.__model__.feature_names
        df = df[f_names]
        d_data = xgb.DMatrix(df)

        results = self.__model__.predict(
            d_data,
        )
        results = self.normalize(results)
        results = np.concatenate(
            (
                np.zeros(df.index.min()),
                results,
            )
        )

        indices = np.where(results == 1)[0]
        bounds = self.compute_bounds(indices)
        return results, bounds, rel_time


class QModelPredict:
    def __init__(
        self,
        predictor_path_1,
        predictor_path_2,
        predictor_path_3,
        predictor_path_4,
        predictor_path_5,
        predictor_path_6,
    ):
        # self.__data__ = QDataPipeline(data_file).preprocess()
        self.__p1__ = QPredictor(predictor_path_1)
        self.__p2__ = QPredictor(predictor_path_2)
        self.__p3__ = QPredictor(predictor_path_3)
        self.__p4__ = QPredictor(predictor_path_4)
        self.__p5__ = QPredictor(predictor_path_5)
        self.__p6__ = QPredictor(predictor_path_6)

    def normalize(self, data):
        return np.array(
            ((data - np.min(data)) / (np.max(data) - np.min(data))).tolist()
        ).astype(float)

    def normalize_gen(self, arr, t_min=0, t_max=1):
        norm_arr = arr.copy()
        try:
            diff = t_max - t_min
            diff_arr = max(arr) - min(arr)
            if diff_arr == 0:
                diff_arr = 1
            norm_arr -= min(arr)
            norm_arr *= diff
            norm_arr /= diff_arr
            norm_arr += t_min
        except Exception as e:
            print("ERROR:" + str(e))
        return norm_arr

    def generate_zone_probabilities(self, t):
        amplitude = 1.0
        poi4_min_val = 0.2
        periodicity4 = 9
        period_skip4 = False
        poi5_min_val = 0.3
        periodicity5 = 6
        period_skip5 = True
        t = self.normalize(t)

        signal_region_equation_POI4 = 1 - np.cos(periodicity4 * t * np.pi)
        signal_region_equation_POI4 = self.normalize_gen(
            signal_region_equation_POI4, poi4_min_val, amplitude
        )
        if period_skip4:
            signal_region_equation_POI4 = np.where(
                t < 2 / periodicity4, poi4_min_val, signal_region_equation_POI4
            )
            signal_region_equation_POI4 = np.where(
                t > 4 / periodicity4, poi4_min_val, signal_region_equation_POI4
            )
        else:
            signal_region_equation_POI4 = np.where(
                t > 2 / periodicity4, poi4_min_val, signal_region_equation_POI4
            )

        signal_region_equation_POI5 = 1 - np.cos(periodicity5 * t * np.pi)
        signal_region_equation_POI5 = self.normalize_gen(
            signal_region_equation_POI5, poi5_min_val, amplitude
        )
        if period_skip5:
            signal_region_equation_POI5 = np.where(
                t < 2 / periodicity5, poi5_min_val, signal_region_equation_POI5
            )
            signal_region_equation_POI5 = np.where(
                t > 4 / periodicity5, poi5_min_val, signal_region_equation_POI5
            )
        else:
            signal_region_equation_POI5 = np.where(
                t > 2 / periodicity5, poi5_min_val, signal_region_equation_POI5
            )

        signal_region_equation_POI4 = np.where(t < 0.03, 0, signal_region_equation_POI4)
        if period_skip5:
            signal_region_equation_POI4 = np.where(
                t > 2 / periodicity5, 0, signal_region_equation_POI4
            )
        if not period_skip4:
            signal_region_equation_POI5 = np.where(
                t < 2 / periodicity4, 0, signal_region_equation_POI5
            )
        signal_region_equation_POI5 = np.where(
            t > 0.75, poi4_min_val, signal_region_equation_POI5
        )
        signal_region_equation_POI5 = np.where(t > 0.90, 0, signal_region_equation_POI5)

        return [signal_region_equation_POI4, signal_region_equation_POI5]

    def top_k_peaks(self, signal, k):
        peaks, _ = find_peaks(signal)
        peak_heights = signal[peaks]
        top_k_indices = np.argsort(peak_heights)[-k:][::-1]
        top_k_peaks = peaks[top_k_indices]
        return top_k_peaks

    def find_closest_point(self, x, y, x_values, y_values, slope, intercept):
        distances = np.abs(y - (slope * np.log(x_values) + intercept))
        closest_index = np.argmin(distances)
        return x_values[closest_index], y_values[closest_index]

    def predict(self, data):
        if not isinstance(data, str):
            csv_headers = next(data)

            if isinstance(csv_headers, bytes):
                csv_headers = csv_headers.decode()

            if "Ambient" in csv_headers:
                csv_cols = (2,4,6,7)
            else:
                csv_cols = (2,3,5,6)

            file_data  = np.loadtxt(data.readlines(), delimiter = ',', skiprows = 0, usecols = csv_cols)
            data_path = "QModel Passthrough"
            relative_time = file_data[:,0]
            # temperature = file_data[:,1]
            resonance_frequency = file_data[:,2]
            dissipation = file_data[:,3]

            emp_predictions = ModelData().IdentifyPoints(data_path=data_path, 
                                                         times=relative_time,
                                                         freq=resonance_frequency,
                                                         diss=dissipation)
        else:    
            emp_predictions = ModelData().IdentifyPoints(data)
        emp_points = []
        for pt in emp_predictions:
            if isinstance(pt, int):
                emp_points.append(pt)
            elif isinstance(pt, list):
                max_pair = max(pt, key=lambda x: x[1])
                emp_points.append(max_pair[0])

        k = 10
        data = self.reset_file_buffer(data)
        results_1, bound_1, rel_time = self.__p1__.predict(data)
        data = self.reset_file_buffer(data)
        results_2, bound_2, rel_time = self.__p2__.predict(data)
        data = self.reset_file_buffer(data)
        results_3, bound_3, rel_time = self.__p3__.predict(data)
        data = self.reset_file_buffer(data)
        results_4, bound_4, rel_time = self.__p4__.predict(data)
        data = self.reset_file_buffer(data)
        results_5, bound_5, rel_time = self.__p5__.predict(data)
        data = self.reset_file_buffer(data)
        results_6, bound_6, rel_time = self.__p6__.predict(data)

        model_results = [
            emp_points[0],
            # bound_1[0][0],

            # emp_points[1],
            bound_2[0][0],

            # emp_points[2],
            bound_3[0][0],

            # emp_points[3],
            bound_4[0][0],

            # emp_points[4],
            bound_5[0][0],

            # emp_points[5],
            bound_6[0][0],
        ]
        approx_4, approx_5 = self.generate_zone_probabilities(
            rel_time[model_results[0] : model_results[5]]
        )

        approx_4 = np.concatenate(
            (
                np.zeros(model_results[0]),
                approx_4,
                np.zeros(len(results_5) - model_results[5]),
            )
        )
        approx_5 = np.concatenate(
            (
                np.zeros(model_results[0]),
                approx_5,
                np.zeros(len(results_5) - model_results[5]),
            )
        )
        model_results[3] = np.argmax(approx_4 * results_4)
        model_results[4] = np.argmax(approx_5 * results_5)
        # plt.figure()
        # plt.title("POI 4")
        # plt.axvline(x=model_results[3])
        # plt.plot(approx_4 * results_4, label="Results_4")
        # plt.plot(approx_4, label="approx_4")
        # plt.legend()
        # plt.show()
        # plt.figure()
        # plt.title("POI 5")
        # plt.axvline(x=model_results[4])
        # plt.plot(approx_5 * results_5, label="Results_5")
        # plt.plot(approx_5, label="approx_5")
        # plt.legend()
        # plt.show()

        return model_results

    def reset_file_buffer(self, data):
        if not isinstance(data, str):
            if hasattr(data, "seekable") and data.seekable():
                data.seek(0)  # reset ByteIO buffer to beginning of stream
            else:
                # ERROR: 'data' must be 'BytesIO' type here, but it's not seekable!
                raise Exception(
                    "Cannot 'seek' stream when attempting to reset file buffer."
                )
        return data