"""
onyx_yolo_dataprocessor.py

This module provides the data preprocessing logic required for the QModel Onyx
YOLO pipeline. It handles the transformation of raw sensor CSV data into
interpolated, median-filtered time-series data ready for the rendering
contracts in `onyx_render.py` and `onyx_fill_render.py`.

Dependencies:
- pandas, numpy
- scipy.signal (medfilt)
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import medfilt

try:
    from QATCH.common.logger import Logger as Log
except (ImportError, ModuleNotFoundError):

    class Log:
        @staticmethod
        def d(tag: str, msg: str):
            print(f"{tag} [DEBUG] {msg}")

        @staticmethod
        def i(tag: str, msg: str):
            print(f"{tag} [INFO] {msg}")

        @staticmethod
        def w(tag: str, msg: str):
            print(f"{tag} [WARNING] {msg}")

        @staticmethod
        def e(tag: str, msg: str):
            print(f"{tag} [ERROR] {msg}")

    Log.i(tag="[HEADLESS OPERATION]", msg="Running...")


class QModelOnyxDataProcessor:
    """
    A utility class for preprocessing raw sensor data for the Onyx pipeline.

    This class handles the data pipeline from raw CSV to render-ready frame:
    1. Cleaning and interpolating raw sensor data.
    2. Applying median filtering to smooth signal noise.

    Attributes:
        TAG (str): Log tag for the class.
        COL_TIME (str): Column name for relative time.
        COL_DISS (str): Column name for dissipation.
        COL_FREQ (str): Column name for resonance frequency.
        TIME_STEP (float): The time interval for interpolation grid (seconds).
        MEDIAN_KERNEL (int): Kernel size for median filtering.
    """

    TAG = "[QModelOnyxDataProcessor]"
    # Column Names
    COL_TIME = "Relative_time"
    COL_DISS = "Dissipation"
    COL_FREQ = "Resonance_Frequency"

    # Preprocessing Settings
    DROP_COLS = ["Date", "Time", "Ambient", "Peak Magnitude (RAW)", "Temperature"]
    TIME_STEP = 0.005
    MEDIAN_KERNEL = 5

    @staticmethod
    def convert_to_dataframe(worker: Any) -> pd.DataFrame:
        """
        Convert raw buffer data from a worker into a pandas DataFrame.

        Retrieves the relative time, resonance frequency, and dissipation buffers from the worker,
        truncates them to the same length, and constructs a DataFrame.

        Args:
            worker (Any): A worker object that provides buffer data through methods
                          `get_t1_buffer(index: int)`, `get_d1_buffer(index: int)`, and
                          `get_d2_buffer(index: int)`.

        Returns:
            pd.DataFrame: A DataFrame with columns 'Relative_time', 'Resonance_Frequency',
            'Dissipation'.

        Raises:
            ValueError: If the worker does not have the required buffer methods or buffers are
            empty.
        """
        required_methods = ["get_t1_buffer", "get_d1_buffer", "get_d2_buffer"]
        for method in required_methods:
            if not hasattr(worker, method):
                raise ValueError(f"Worker is missing required method: {method}")

        relative_time = np.array(worker.get_t1_buffer(0))
        resonance_frequency = np.array(worker.get_d1_buffer(0))
        dissipation = np.array(worker.get_d2_buffer(0))

        min_length = min(len(relative_time), len(resonance_frequency), len(dissipation))

        if min_length == 0:
            raise ValueError("One or more buffers are empty.")

        t_raw = relative_time[:min_length]
        freq_raw = resonance_frequency[:min_length]
        diss_raw = dissipation[:min_length]

        df = pd.DataFrame(
            {
                QModelOnyxDataProcessor.COL_TIME: t_raw,
                QModelOnyxDataProcessor.COL_FREQ: freq_raw,
                QModelOnyxDataProcessor.COL_DISS: diss_raw,
            }
        )

        return df

    @classmethod
    def preprocess_dataframe(
        cls, df: pd.DataFrame, baseline_freq: float = None, baseline_diss: float = None
    ) -> pd.DataFrame:
        """
        Cleans, interpolates, and smooths the raw sensor dataframe.

        Performs the following steps:
        1. Drops unnecessary columns (Ambient, Temperature, etc.).
        2. Reindexes the dataframe to a fixed time grid defined by `TIME_STEP`.
        3. Interpolates missing values.
        4. Applies a median filter to smooth numeric columns.

        Args:
            df (pd.DataFrame): The raw input dataframe containing sensor data.
            baseline_freq (float, optional): Accepted for caller compatibility;
                currently unused (preprocessing no longer derives a baseline-
                relative feature from it).
            baseline_diss (float, optional): See `baseline_freq`.

        Returns:
            pd.DataFrame: The processed dataframe with interpolated time and
            smoothed signals, or None if the required time column is missing.
        """
        cols_to_drop = [c for c in cls.DROP_COLS if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)
        if cls.COL_TIME not in df.columns:
            return None
        df.drop_duplicates(subset=[cls.COL_TIME], keep="first", inplace=True)
        t_min = df[cls.COL_TIME].min()
        t_max = df[cls.COL_TIME].max()
        new_time_grid = np.arange(t_min, t_max, cls.TIME_STEP)
        df = df.set_index(cls.COL_TIME)
        combined_index = df.index.union(new_time_grid).sort_values()
        df = df.reindex(combined_index).interpolate(method="index").loc[new_time_grid]
        df = df.reset_index().rename(columns={"index": cls.COL_TIME})
        for col in df.columns:
            if col != cls.COL_TIME and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = medfilt(df[col], kernel_size=cls.MEDIAN_KERNEL)
        return df
