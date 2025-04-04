# q_forecast_predictor.py
"""
This module provides the QForecastPredictor class, which uses XGBoost boosters 
and a scaler to process forecast data and update prediction states. The predictor 
maintains an internal data buffer and toggles between available boosters to signal 
the fill status based on a prediction threshold.

Author: 
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    04-04-2025

Version
    V2
"""

import os
import numpy as np
import xgboost as xgb
import pickle
import pandas as pd
from enum import Enum
from sklearn.pipeline import Pipeline
from QATCH.common.logger import Logger
from QATCH.QModel.q_forecast_data_processor import QForecastDataProcessor

# Global constants
PREDICTION_THRESHOLD: float = 0.75
TAG: list = ['QForecastPredictor']


class AvailableBoosters(Enum):
    """Enum for available boosters in the prediction process."""
    START = 0
    END = 1


class FillStatus(Enum):
    """Enum for fill statuses during prediction updates."""
    NO_FILL = 0
    FILLING = 1
    FULL_FILL = 2


class QForecastPredictor:
    """
    Class for forecasting predictions using XGBoost models and a scaler.

    This predictor maintains an internal buffer of data and alternates between two 
    boosters (start and end) to determine if the fill process is ongoing or complete.
    """

    def __init__(self, start_booster_path: str, end_booster_path: str, scaler_path: str) -> None:
        """
        Initialize a QForecastPredictor instance.

        Args:
            start_booster_path (str): File path for the start booster model.
            end_booster_path (str): File path for the end booster model.
            scaler_path (str): File path for the scaler used for transforming data.
        """
        self._start_booster: xgb.Booster = self._load_model(start_booster_path)
        self._end_booster: xgb.Booster = self._load_model(end_booster_path)
        self._scaler: Pipeline = self._load_scaler(scaler_path)
        self._data: pd.DataFrame = pd.DataFrame()
        self._prediction_buffer_size: int = 0
        self._fill_state: FillStatus = FillStatus.NO_FILL
        self._active_booster: AvailableBoosters = AvailableBoosters.START
        self._start_loc: dict = {'index': -1, 'time': -1.0}
        self._end_loc: dict = {'index': -1, 'time': -1.0}

    def get_fill_status(self) -> FillStatus:
        """
        Get the current fill status.

        Returns:
            FillStatus: The current fill state of the predictor.
        """
        return self._fill_state

    def update_predictions(self, new_data: pd.DataFrame, prediction_rate: int = 100) -> None:
        """
        Update the prediction buffer with new data and perform prediction if the buffer is full.

        Args:
            new_data (pd.DataFrame): New incoming data to add to the buffer.
            prediction_rate (int, optional): The number of data points required to trigger a prediction.
                                              Defaults to 100.
        """
        self._extend_buffer(new_data=new_data)
        if self._prediction_buffer_size >= prediction_rate:
            self._prediction_buffer_size = 0
            self._fill_state = self._predict()

    def _extend_buffer(self, new_data: pd.DataFrame) -> None:
        """
        Extend the internal data buffer with new data.

        Args:
            new_data (pd.DataFrame): DataFrame containing new data to be appended.
        """
        if self._data is None:
            self._data = pd.DataFrame(columns=new_data.columns)
        self._data = pd.concat([self._data, new_data], ignore_index=True)
        self._prediction_buffer_size += len(new_data)

    def _predict(self) -> FillStatus:
        """
        Execute the prediction process based on the active booster.

        Returns:
            FillStatus: The updated fill status after processing predictions.
        """
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

    def _process_prediction(
        self,
        booster: xgb.Booster,
        loc: dict,
        next_active: AvailableBoosters,
        completion_status: FillStatus,
        waiting_status: FillStatus
    ) -> FillStatus:
        """
        Process prediction using the specified booster and update the location marker.

        Args:
            booster (xgb.Booster): The XGBoost booster model to use for prediction.
            loc (dict): Dictionary holding the current index and time for the booster.
            next_active (AvailableBoosters): The next booster to be activated if prediction is successful.
            completion_status (FillStatus): The fill status to return if prediction is successful.
            waiting_status (FillStatus): The fill status to return if prediction is not successful.

        Returns:
            FillStatus: The updated fill status after processing prediction.
        """
        start_index: int = loc.get('index')
        start_time: float = loc.get('time')
        predictions: np.ndarray = self._get_predictions(
            booster=booster, start_index=start_index, start_time=start_time)
        if (predictions == 1).any():
            new_index: int = int(np.argmax(predictions == 1))
            loc['index'] = new_index
            loc['time'] = self._data.loc[new_index, 'Relative_time']
            self._active_booster = next_active
            return completion_status
        return waiting_status

    def _get_predictions(
        self,
        booster: xgb.Booster,
        start_index: int,
        start_time: float
    ) -> np.ndarray:
        """
        Get predictions from the booster for the current buffered data.

        Args:
            booster (xgb.Booster): The XGBoost booster model to use for predictions.
            start_index (int): The starting index from where to process data.
            start_time (float): The starting time corresponding to the data.

        Returns:
            np.ndarray: An array of binary predictions based on the probability threshold.
        """
        features: pd.DataFrame = QForecastDataProcessor.process_data(
            self._data, live=True, start_idx=start_index, start_time=start_time)
        Logger.i(TAG, features.columns)
        transformed_features = self._scaler.transform(features)
        dfeatures = xgb.DMatrix(transformed_features)
        probabilities: np.ndarray = booster.predict(dfeatures)
        predictions: np.ndarray = (
            probabilities > PREDICTION_THRESHOLD).astype(int)
        return predictions

    def _load_model(self, booster_path: str) -> xgb.Booster:
        """
        Load an XGBoost booster model from the specified file path.

        Args:
            booster_path (str): The file path to the booster model.

        Returns:
            xgb.Booster: The loaded XGBoost booster model.

        Raises:
            Exception: If the booster file does not exist or fails to load.
        """
        if not os.path.exists(booster_path):
            Logger.e(TAG, f"Booster path `{booster_path}` does not exist.")
            raise Exception(f"Booster path `{booster_path}` does not exist.")

        booster: xgb.Booster = xgb.Booster()
        try:
            booster.load_model(booster_path)
            Logger.i(
                TAG, f'Booster successfully loaded from path `{booster_path}`.')
            return booster
        except Exception as e:
            Logger.e(
                TAG, f'Error loading booster with path `{booster_path}`: {e}.')
            raise

    def _load_scaler(self, scaler_path: str) -> Pipeline:
        """
        Load a scaler (pipeline) from the specified file path.

        Args:
            scaler_path (str): The file path to the scaler.

        Returns:
            Pipeline: The loaded scaler pipeline.

        Raises:
            Exception: If the scaler file does not exist or fails to load.
        """
        if not os.path.exists(scaler_path):
            Logger.e(TAG, f"Scaler path `{scaler_path}` does not exist.")
            raise Exception(f"Scaler path `{scaler_path}` does not exist.")
        try:
            with open(scaler_path, "rb") as f:
                scaler: Pipeline = pickle.load(f)
            Logger.i(
                TAG, f"Scaler successfully loaded from path `{scaler_path}`.")
            return scaler
        except Exception as e:
            Logger.e(
                TAG, f'Error loading scaler with path `{scaler_path}`: {e}.')
            raise
