# v6_yolo_live.py
"""
This module provides the infrastructure for running a YOLO-based fill classifier
in a live, multiprocessing environment. It includes a classification logic class
that manages data buffering and prediction, as well as a dedicated multiprocessing
wrapper to handle execution in a separate process, ensuring non-blocking performance
for the main application.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2026-01-12

Version:
    2.0.1
"""

import logging
import multiprocessing
import os
import sys
from logging.handlers import QueueHandler
from typing import Optional

import pandas as pd

from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.QModel.src.models.v6_yolo.v6_yolo import (
    QModelV6Config,
    QModelV6YOLO_FillClassifier,
)
from QATCH.QModel.src.models.v6_yolo.v6_yolo_dataprocessor import (
    QModelV6YOLO_DataProcessor,
)

TAG = "[QModelV6YOLO_LiveProcess]"


class QModelV6YOLO_Live(QModelV6YOLO_FillClassifier):
    """Manages data buffering and executes predictions for real-time fill classification.

    This class handles the accumulation of streaming sensor data, maintains a fixed-size
    sliding window buffer, and triggers inference using a YOLO-based model when conditions
    are met.

    Attributes:
        buffer_window_size (Optional[int]): The maximum number of rows to retain in the
            rolling buffer. If None, the buffer grows indefinitely.
        current_prediction (int): The most recent classification result class ID.
        STATUS_MAP (dict): Mapping of integer classification codes to human-readable strings.
    """

    STATUS_MAP = {
        -1: "No Fill",
        0: "Initial FIll",
        1: "1 Channel",
        2: "2 Channels",
        3: "3 Channels",
    }
    TAG = "[QModelV6YOLO_Live]"

    def __init__(self, model_path: str, buffer_window_size: Optional[int] = None):
        """Initializes the live fill classifier with a model and buffer settings.

        Args:
            model_path (str): The file path to the trained YOLO model weights.
            buffer_window_size (Optional[int]): The maximum number of data rows to keep
                in memory. Defaults to None.
        """
        super().__init__(model_path)
        self.buffer_window_size = buffer_window_size
        self.current_prediction = 0

        # New storage attributes
        self._data: Optional[pd.DataFrame] = None
        self._last_max_time = -float("inf")
        self._prediction_buffer_size = 0

        Log.i(self.TAG, "Initialized LiveFillClassifier.")

    def add_chunk(self, df_chunk: pd.DataFrame) -> None:
        """Ingests a new chunk of data into the rolling buffer.

        This is the public interface for data ingestion. It wraps the internal
        buffer extension logic and handles error logging if the update fails.

        Args:
            df_chunk (pd.DataFrame): A pandas DataFrame containing the new time-series
                data to append.
        """
        try:
            self._extend_buffer(df_chunk)
        except ValueError as e:
            Log.e(self.TAG, f"Failed to add chunk: {e}")

    def _extend_buffer(self, new_data: pd.DataFrame) -> None:
        """Extends the internal data buffer with new unique data points.

        Ensures time monotonicity by only adding rows with 'Relative_time' greater
        than the previously recorded maximum. Also handles sorting and trimming the
        buffer to the defined `buffer_window_size`.

        Args:
            new_data (pd.DataFrame): DataFrame containing new data to be appended.

        Raises:
            ValueError: If `new_data` is not a DataFrame or is missing the
                'Relative_time' column.
        """
        if not isinstance(new_data, pd.DataFrame):
            raise ValueError("new_data must be a pandas DataFrame.")

        if new_data.empty:
            return

        if "Relative_time" not in new_data.columns:
            raise ValueError("new_data must contain the 'Relative_time' column.")

        if self._data is None or self._data.empty:
            self._data = new_data.copy()
            self._prediction_buffer_size = len(self._data)
        else:
            new_data_filtered = new_data[
                new_data["Relative_time"] > self._last_max_time
            ]

            if not new_data_filtered.empty:
                new_data_aligned = new_data_filtered.reindex(columns=self._data.columns)
                self._data = pd.concat(
                    [self._data, new_data_aligned], ignore_index=True
                )
                self._prediction_buffer_size += len(new_data_filtered)
        if self._data is not None and not self._data.empty:
            self._last_max_time = self._data["Relative_time"].max()
            self._data.sort_values(by="Relative_time", ascending=True, inplace=True)
            self._data.reset_index(drop=True, inplace=True)
            if self.buffer_window_size and len(self._data) > self.buffer_window_size:
                self._data = self._data.iloc[-self.buffer_window_size :]
                self._data.reset_index(drop=True, inplace=True)

    def attempt_classification(self) -> int:
        """Executes the classification pipeline on the current buffered data.

        This method validates data length against `QModelV6Config.MIN_SLICE_LENGTH`,
        filters for valid time ranges (Relative_time > 0.05), applies preprocessing,
        and runs the model inference.

        Returns:
            int: The classification result class ID. Returns the previous `current_prediction`
            if the buffer is insufficient, empty, or if inference fails.
        """
        if self._data is None or len(self._data) < QModelV6Config.MIN_SLICE_LENGTH:
            return self.current_prediction

        try:
            processed_df = self._data.copy()

            mask = self._data["Relative_time"] > 0.05
            if mask.any():
                self._data = self._data.loc[mask]
                processed_df = QModelV6YOLO_DataProcessor.preprocess_dataframe(
                    self._data.copy()
                )
                if processed_df is not None and not processed_df.empty:
                    pred = self.predict(processed_df)
                    self.current_prediction = pred
                    return pred
                else:
                    Log.w(self.TAG, "Preprocessing returned empty/None DataFrame.")
            else:
                Log.d(self.TAG, "Waiting for > 0.05s of buffer data.")
        except Exception as e:
            Log.e(self.TAG, f"Inference failed: {str(e)}")

        return self.current_prediction

    def get_status_str(self) -> str:
        """Retrieves the human-readable string representation of the current prediction.

        Returns:
            str: The status label corresponding to `current_prediction` (e.g., "1 Channel",
            "No Fill"). Returns "Unknown" if the ID is not in `STATUS_MAP`.
        """
        return self.STATUS_MAP.get(self.current_prediction, "Unknown")


class QModelV6YOLO_LiveProcess(multiprocessing.Process):
    """
    Dedicated process for running real-time YOLO fill state predictions.

    This class wraps the `QModelV6YOLO_Live` classifier in a `multiprocessing.Process`.
    It continuously consumes raw worker data from an input queue, processes it,
    runs inference, and pushes the results to an output queue. This design ensures
    that computationally expensive inference does not block the main application loop.

    Attributes:
        _queueLog (multiprocessing.Queue): Queue for thread-safe logging.
        _exit (multiprocessing.Event): Event flag to signal process termination.
        _done (multiprocessing.Event): Event flag to signal process completion.
        _queue_in (multiprocessing.Queue): Input queue receiving raw worker data.
        _queue_out (multiprocessing.Queue): Output queue sending prediction results (int, str).
        model_path (str): Path to the YOLO model file.
        buffer_window_size (Optional[int]): Rolling window size for the model buffer.
        _classifier (Optional[QModelV6YOLO_Live]): The internal classifier instance (created in run).
    """

    TAG = "[QModelV6YOLO_LiveProcess]"

    def __init__(
        self,
        queue_log: multiprocessing.Queue,
        queue_in: multiprocessing.Queue,
        queue_out: multiprocessing.Queue,
        buffer_window_size: Optional[int] = None,
    ) -> None:
        """
        Initializes the LiveProcess configuration.

        Args:
            queue_log (multiprocessing.Queue): Queue for logging messages back to the main process.
            queue_in (multiprocessing.Queue): Input queue for receiving raw data from workers.
            queue_out (multiprocessing.Queue): Output queue for sending prediction tuples.
            model_path (str): The file path to the trained YOLO model.
            buffer_window_size (Optional[int], optional): The maximum size of the rolling
                data buffer. Defaults to None.
        """
        Log.d(self.TAG, "Starting multiprocess fill status")
        self._queueLog: multiprocessing.Queue = queue_log
        multiprocessing.Process.__init__(self)

        self._exit = multiprocessing.Event()
        self._done = multiprocessing.Event()
        self._queue_in: multiprocessing.Queue = queue_in
        self._queue_out: multiprocessing.Queue = queue_out

        # Store config to initialize model inside run()
        v6_base_path = os.path.join(
            Architecture.get_path(), "QATCH", "QModel", "SavedModels", "qmodel_v6_yolo"
        )
        type_cls_asset = os.path.join(v6_base_path, "type_cls", "weights", "best.pt")
        self.model_path = type_cls_asset
        self.buffer_window_size = buffer_window_size

        # Instance placeholder
        self._classifier: Optional[QModelV6YOLO_Live] = None

    def run(self) -> None:
        """
        Executes the main process loop.

        Steps:
        1. Redirects stdout/stderr to prevent console spam.
        2. Configures logging to use the main process's queue.
        3. Initializes the `QModelV6YOLO_Live` classifier (must happen here to avoid pickling).
        4. Enters a loop to consume data from `_queue_in`, convert it to DataFrames,
           update the classifier, and run inference.
        5. Puts results into `_queue_out`.
        """
        try:
            # Redirect stdout/stderr
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")

            # Configure Logging
            logger = logging.getLogger("QATCH.logger")
            logger.addHandler(QueueHandler(self._queueLog))
            logger.setLevel(logging.DEBUG)

            from multiprocessing.util import get_logger

            multiprocessing_logger = get_logger()
            if multiprocessing_logger.handlers:
                multiprocessing_logger.handlers[0].setStream(sys.stderr)
            multiprocessing_logger.setLevel(logging.WARNING)

            # Initialize the Model (Must be done inside run to avoid pickling issues)
            self._classifier = QModelV6YOLO_Live(
                model_path=self.model_path, buffer_window_size=self.buffer_window_size
            )
            Log.i(TAG, "YOLO Live Process Started and Model Loaded.")

            while not self._exit.is_set():

                # Idle wait if queue is empty
                while self._queue_in.empty() and not self._exit.is_set():
                    pass

                data_received = False
                while not self._queue_in.empty():
                    raw_data = self._queue_in.get()
                    df_chunk = QModelV6YOLO_DataProcessor.convert_to_dataframe(raw_data)
                    try:
                        if df_chunk is not None and not df_chunk.empty:
                            self._classifier.add_chunk(df_chunk)
                            data_received = True

                    except ValueError as ve:
                        Log.w(TAG, f"Skipping worker data chunk: {ve}")
                    except Exception as e:
                        Log.e(TAG, f"Error converting worker data: {e}")

                    if data_received:
                        pred_int = self._classifier.attempt_classification()
                        pred_str = self._classifier.get_status_str()
                        self._queue_out.put((pred_int, pred_str))

        except Exception:
            # Capture traceback
            limit: Optional[int] = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list += format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(TAG, line)

        finally:
            Log.d(TAG, "QModelV6YOLO_LiveProcess stopped.")
            self._done.set()

    def is_running(self) -> bool:
        """
        Checks if the process is currently running.

        Returns:
            bool: True if the process is active, False if it has finished.
        """
        return not self._done.is_set()

    def stop(self) -> None:
        """
        Signals the process to stop execution.

        Sets the exit event, which will cause the main loop in `run()` to terminate.
        """
        self._exit.set()
