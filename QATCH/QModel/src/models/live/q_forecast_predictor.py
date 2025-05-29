#!/usr/bin/env python3
"""
q_forecast_predictor.py

This module provides the QForecastPredictor class, which uses XGBoost boosters
and a scaler to process forecast data and update prediction states. The predictor
maintains an internal data buffer and toggles between available boosters to signal
the fill status based on a prediction threshold.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    05-14-2025

Version:
    V2.0.1
"""

import os
import numpy as np
import xgboost as xgb
import pickle
import pandas as pd
import threading
from enum import Enum
from sklearn.pipeline import Pipeline
from QATCH.QModel.src.models.live.q_forecast_data_processor import QForecastDataProcessor
from QATCH.QModel.src.models.pf.pf_predictor import PFPredictor
from QATCH.common.logger import Logger as Log
from QATCH.common.architecture import Architecture


# Global constants
PREDICTION_THRESHOLD: float = 0.75
WAIT_TIME: int = 1
TAG: str = '[QForecastPredictor]'


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

        Raises:
            ValueError: If any of the provided paths is not a non-empty string.
        """
        for path, name in [(start_booster_path, "start_booster_path"),
                           (end_booster_path, "end_booster_path"),
                           (scaler_path, "scaler_path")]:
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"{name} must be a non-empty string.")

        self._start_booster: xgb.Booster = self._load_model(start_booster_path)
        self._end_booster: xgb.Booster = self._load_model(end_booster_path)
        self._scaler: Pipeline = self._load_scaler(scaler_path)
        self._data: pd.DataFrame = pd.DataFrame()
        self._prediction_buffer_size: int = 0
        self._fill_state: FillStatus = FillStatus.NO_FILL
        self._active_booster: AvailableBoosters = AvailableBoosters.START
        self._start_loc: dict = {'index': -1, 'time': -1.0}
        self._end_loc: dict = {'index': -1, 'time': -1.0}
        self._last_max_time: float = 0.5
        self._has_reported_full_fill = False
        self._is_waiting_for_full_fill = False
        self._no_wait = False
        model_dir = os.path.join(
            Architecture.get_path(),
            "QATCH", "QModel", "SavedModels", "pf")
        self._pf_predictor = PFPredictor(model_dir=model_dir)

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

        Raises:
            ValueError: If new_data is not a DataFrame or prediction_rate is not a positive integer.
        """
        if not isinstance(new_data, pd.DataFrame):
            raise ValueError("new_data must be a pandas DataFrame.")
        if not isinstance(prediction_rate, int) or prediction_rate <= 0:
            raise ValueError("prediction_rate must be a positive integer.")

        self._extend_buffer(new_data=new_data)
        if self._prediction_buffer_size >= prediction_rate:
            self._prediction_buffer_size = 0
            self._fill_state = self._predict()

    def get_start_time(self) -> float:
        """
        Retrieves the estimated start time of the fill.

        Returns:
            float: The estimated start time of the fill.
        """
        return self._start_loc['time']

    def get_end_time(self) -> float:
        """
        Retrieves the estimated end time of the fill.

        Returns:
            float: The estimated end time of the fill.
        """
        return self._end_loc['time']

    def no_wait(self) -> None:
        """
        Bypasses wait time for hysteresis on channel 3 exit.

        Returns:
            None
        """
        self._no_wait = True

    def _extend_buffer(self, new_data: pd.DataFrame) -> None:
        """
        Extend the internal data buffer with new data.

        Args:
            new_data (pd.DataFrame): DataFrame containing new data to be appended.

        Raises:
            ValueError: If new_data does not contain the 'Relative_time' column.
        """
        if not isinstance(new_data, pd.DataFrame):
            raise ValueError("new_data must be a pandas DataFrame.")
        if not new_data.empty and 'Relative_time' not in new_data.columns:
            raise ValueError(
                "new_data must contain the 'Relative_time' column.")
        # time = new_data['Relative_time']
        # Log.i(TAG, f'{time}')
        if self._data is None or self._data.empty:
            self._data = new_data.copy()
            self._prediction_buffer_size = len(self._data)
        else:
            new_data_filtered = new_data[new_data['Relative_time']
                                         > self._last_max_time]
            new_data_aligned = new_data_filtered.reindex(
                columns=self._data.columns)
            self._data = pd.concat(
                [self._data, new_data_aligned], ignore_index=True)
            self._prediction_buffer_size += len(new_data_filtered)

        if not self._data.empty:
            self._last_max_time = self._data['Relative_time'].max()
            self._data.sort_values(
                by='Relative_time', ascending=True, inplace=True)
            self._data.reset_index(drop=True, inplace=True)

    def _predict(self) -> FillStatus:
        """
        Execute the prediction process based on the active booster.

        Returns:
            FillStatus: The updated fill status after processing predictions.

        Raises:
            Exception: If no valid booster is active.
        """
        if self._active_booster == AvailableBoosters.START:
            Log.d(TAG, 'Using start booster.')
            return self._process_prediction(
                booster=self._start_booster,
                loc=self._start_loc,
                next_active=AvailableBoosters.END,
                completion_status=FillStatus.FILLING,
                waiting_status=FillStatus.NO_FILL,
            )
        elif self._active_booster == AvailableBoosters.END:
            Log.d(TAG, 'Using end booster.')
            # If FULL_FILL has already been finalized, return it immediately.
            if self._has_reported_full_fill:
                return FillStatus.FULL_FILL

            result = self._process_prediction(
                booster=self._end_booster,
                loc=self._end_loc,
                next_active=AvailableBoosters.END,
                completion_status=FillStatus.FULL_FILL,
                waiting_status=FillStatus.FILLING,
            )

            if result == FillStatus.FULL_FILL and self._no_wait == False:
                # If not already waiting, start a non-blocking timer.
                if not self._is_waiting_for_full_fill:
                    self._is_waiting_for_full_fill = True
                    timer = threading.Timer(WAIT_TIME, self._set_full_fill)
                    timer.start()
                # While waiting, return FILLING to the caller.
                return FillStatus.FILLING
            else:
                return result
        else:
            Log.e(TAG, "No valid booster active.")
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
        if not isinstance(loc, dict):
            raise ValueError("loc must be a dictionary.")

        start_index: int = loc.get('index')
        start_time: float = loc.get('time')
        predictions: np.ndarray = self._get_predictions(
            booster=booster, start_index=start_index, start_time=start_time)
        if predictions.size == 0:
            Log.e(TAG, "No predictions were generated.")
            raise ValueError("No predictions generated from the booster.")

        if (predictions == 1).any():
            num_pois = self._pf_predictor.predict(file_buffer=self._data)
            if completion_status == FillStatus.FULL_FILL and num_pois < 6:
                return waiting_status

            new_index: int = int(np.argmax(predictions == 1))
            loc['index'] = new_index
            try:
                loc['time'] = self._data.loc[new_index, 'Relative_time']
            except Exception as e:
                Log.e(
                    TAG, f"Error accessing 'Relative_time' at index {new_index}: {e}")
                raise
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

        Raises:
            ValueError: If the processed features DataFrame is empty.
        """
        features: pd.DataFrame = QForecastDataProcessor.process_data(
            self._data, live=True, start_idx=start_index, start_time=start_time)
        if features.empty:
            Log.e(
                TAG, "Processed features are empty; unable to perform predictions.")
            raise ValueError("Processed features are empty.")
        try:
            transformed_features = self._scaler.transform(features.to_numpy())

            # transformed_features = self._scaler.transform(features)
        except Exception as e:
            Log.e(TAG, f"Error transforming features: {e}")
            raise
        dfeatures = xgb.DMatrix(transformed_features)
        try:
            probabilities: np.ndarray = booster.predict(dfeatures)
            Log.d(
                TAG, f'Max prob {max(probabilities)} @ idx {np.argmax(probabilities)}, data length={len(self._data)}')
        except Exception as e:
            Log.e(TAG, f"Error during booster prediction: {e}")
            raise
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
            ValueError: If booster_path is not a non-empty string.
            Exception: If the booster file does not exist or fails to load.
        """
        if not isinstance(booster_path, str) or not booster_path.strip():
            raise ValueError("booster_path must be a non-empty string.")
        if not os.path.exists(booster_path):
            Log.e(TAG, f"Booster path `{booster_path}` does not exist.")
            raise Exception(f"Booster path `{booster_path}` does not exist.")

        booster: xgb.Booster = xgb.Booster()
        try:
            booster.load_model(booster_path)
            Log.i(
                TAG, f'Booster successfully loaded from path `{booster_path}`.')
            return booster
        except Exception as e:
            Log.e(
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
            ValueError: If scaler_path is not a non-empty string.
            Exception: If the scaler file does not exist or fails to load.
        """
        if not isinstance(scaler_path, str) or not scaler_path.strip():
            raise ValueError("scaler_path must be a non-empty string.")
        if not os.path.exists(scaler_path):
            Log.e(TAG, f"Scaler path `{scaler_path}` does not exist.")
            raise Exception(f"Scaler path `{scaler_path}` does not exist.")
        try:
            with open(scaler_path, "rb") as f:
                scaler: Pipeline = pickle.load(f)
            Log.i(
                TAG, f"Scaler successfully loaded from path `{scaler_path}`.")
            return scaler
        except Exception as e:
            Log.e(
                TAG, f'Error loading scaler with path `{scaler_path}`: {e}.')
            raise

    def _set_full_fill(self):
        """Callback for the timer to finalize FULL_FILL reporting."""
        self._has_reported_full_fill = True
        self._is_waiting_for_full_fill = False
