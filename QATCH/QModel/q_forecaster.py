import os
import numpy as np
import xgboost as xgb
import pickle
import pandas as pd
from enum import Enum
from sklearn.pipeline import Pipeline
from QATCH.common.logger import Logger
from QATCH.QModel.q_forecast_data_processor import QForecastDataProcessor


PREDICTION_THRESHOLD = 0.75
TAG = ['QForecastPredictor']


class AvailableBoosters(Enum):
    START = 0
    END = 1


class FillStatus(Enum):
    NO_FILL = 0
    FILLING = 1
    FULL_FILL = 2


class QForecastPredictor:
    def __init__(self, start_booster_path: str, end_booster_path: str, scaler_path: str):
        self._start_booster = self._load_model(start_booster_path)
        self._end_booster = self._load_model(end_booster_path)
        self._scaler: Pipeline = self._load_scaler(scaler_path)
        self._data = pd.DataFrame()
        self._prediction_buffer_size = 0
        self._fill_state = FillStatus.NO_FILL
        self._active_booster = AvailableBoosters.START
        self._start_loc = {'index': -1, 'time': -1.0}
        self._end_loc = {'index': -1, 'time': -1.0}

    def get_fill_status(self):
        return self._fill_state

    def update_predictions(self, new_data: pd.DataFrame, prediction_rate: int = 100):
        self._extend_buffer(new_data=new_data)
        if self._prediction_buffer_size >= prediction_rate:
            self._prediction_buffer_size = 0
            self._fill_state = self._predict()

    def _extend_buffer(self, new_data: pd.DataFrame):
        if self._data is None:
            self._data = pd.DataFrame(columns=new_data.columns)
        self._data = pd.concat([self._data, new_data], ignore_index=True)
        self._prediction_buffer_size += len(new_data)

    def _predict(self):
        if self._active_booster == AvailableBoosters.START:
            return self._process_prediction(
                booster=self._start_booster,
                loc=self._start_loc,
                next_active=AvailableBoosters.END,
                completion_status=FillStatus.FILLING,
                waiting_status=FillStatus.NO_FILL,
            )
        elif self._active_booster == AvailableBoosters.END:
            return self._process_prediction(
                booster=self._end_booster,
                loc=self._end_loc,
                next_active=AvailableBoosters.END,  # or update as needed
                completion_status=FillStatus.FULL_FILL,
                waiting_status=FillStatus.FILLING,
            )
        else:
            Logger.e(TAG, "No valid booster active.")
            raise Exception("No valid booster active.")

    def _process_prediction(self, booster: xgb.Booster, loc: dict, next_active: AvailableBoosters, completion_status: FillStatus, waiting_status: FillStatus):
        start_index = loc.get('index')
        start_time = loc.get('time')
        predictions = self._get_predictions(
            booster=booster, start_index=start_index, start_time=start_time)
        if (predictions == 1).any():
            new_index = int(np.argmax(predictions == 1))
            loc['index'] = new_index
            loc['time'] = self._data.loc[new_index, 'Relative_time']
            self._active_booster = next_active
            return completion_status
        return waiting_status

    def _get_predictions(self, booster: xgb.Booster, start_index: int, start_time: float):
        features = QForecastDataProcessor.process_data(
            self._data, live=True, start_idx=start_index, start_time=start_time)
        Logger.i(TAG, features.columns)
        transformed_features = self._scaler.transform(features)
        dfeatures = xgb.DMatrix(transformed_features)
        probabilities = booster.predict(dfeatures)
        predictions = (probabilities > PREDICTION_THRESHOLD).astype(int)
        return predictions

    def _load_model(self, booster_path: str):
        if not os.path.exists(booster_path):
            Logger.e(TAG, f"Booster path `{booster_path}` does not exist.")
            raise Exception(f"Booster path `{booster_path}` does not exist.")

        booster = xgb.Booster()
        try:
            booster.load_model(booster_path)
            Logger.i(TAG,
                     f'Booster successfully loaded from path `{booster_path}`.')
            return booster
        except Exception as e:
            Logger.e(TAG,
                     f'Error loading booster with path `{booster_path}`: {e}.')

    def _load_scaler(self, scaler_path: str):
        if not os.path.exists(scaler_path):
            Logger.e(TAG, f"Scaler path `{scaler_path}` does not exist.")
            raise Exception(f"Scaler path `{scaler_path}` does not exist.")
        try:
            scaler = None
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            Logger.i(TAG,
                     f"Scaler successfully loaded from path `{scaler_path}`.")
            return scaler
        except Exception as e:
            Logger.e(TAG,
                     f'Error loading scaler with path `{scaler_path}`: {e}.')
