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
from typing import List, Optional

import pandas as pd

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
    """
    Manages data buffering and executes predictions for real-time fill classification.

    This class extends the base YOLO fill classifier to handle streaming data.
    It maintains a rolling buffer of data chunks, consolidates them into a single
    DataFrame, and performs inference when sufficient data is available.

    Attributes:
        buffer_window_size (Optional[int]): The maximum number of rows to retain in
            the rolling buffer. If None, the buffer size is unlimited (not recommended).
        current_prediction (int): The most recent classification result index.
        _chunk_buffer (List[pd.DataFrame]): A list of incoming data chunks waiting
            to be consolidated.
        _cumulative_row_count (int): The total number of rows currently held in the
            unconsolidated chunks.
    """

    STATUS_MAP = {
        -1: "No Fill",
        0: "Initial/Unknown",
        1: "1 Channel",
        2: "2 Channels",
        3: "3 Channels",
    }
    TAG = "[QModelV6YOLO_Live]"

    def __init__(self, model_path: str, buffer_window_size: Optional[int] = None):
        """
        Initializes the live fill classifier.

        Args:
            model_path (str): The file path to the trained YOLO model.
            buffer_window_size (Optional[int], optional): The maximum size of the
                rolling data buffer in rows. Defaults to None.
        """
        super().__init__(model_path)
        self.buffer_window_size = buffer_window_size
        self.current_prediction = 0
        self._chunk_buffer: List[pd.DataFrame] = []
        self._cumulative_row_count = 0

        Log.i(self.TAG, "Initialized LiveFillClassifier.")

    def add_chunk(self, df_chunk: pd.DataFrame) -> None:
        """
        Adds a new chunk of data to the processing buffer.

        If the buffer exceeds twice the `buffer_window_size`, it triggers an
        automatic consolidation and truncation to manage memory usage.

        Args:
            df_chunk (pd.DataFrame): The new dataframe chunk to append.
        """
        if df_chunk is None or df_chunk.empty:
            return

        self._chunk_buffer.append(df_chunk)
        self._cumulative_row_count += len(df_chunk)
        if self.buffer_window_size and self._cumulative_row_count > (
            self.buffer_window_size * 2
        ):
            self._consolidate_and_truncate()

    def _consolidate_and_truncate(self) -> pd.DataFrame:
        """
        Consolidates buffered chunks and enforces the window size limit.

        Merges all dataframes in `_chunk_buffer` into a single DataFrame and
        truncates it to keep only the most recent `buffer_window_size` rows.

        Returns:
            pd.DataFrame: The consolidated and truncated dataframe.
        """
        if not self._chunk_buffer:
            return pd.DataFrame()
        full_df = pd.concat(self._chunk_buffer, ignore_index=True)
        if self.buffer_window_size and len(full_df) > self.buffer_window_size:
            full_df = full_df.iloc[-self.buffer_window_size :]
        self._chunk_buffer = [full_df]
        self._cumulative_row_count = len(full_df)
        return full_df

    def attempt_classification(self) -> int:
        """
        Attempts to classify the current state of the buffered data.

        Consolidates the buffer, preprocesses the data, and runs the YOLO
        prediction model. If the data is insufficient (less than MIN_SLICE_LENGTH)
        or preprocessing fails, the previous prediction is retained.

        Returns:
            int: The class index of the prediction (e.g., 1, 2, 3) or the
            previous prediction if inference could not be run.
        """
        current_df = self._consolidate_and_truncate()
        if len(current_df) < QModelV6Config.MIN_SLICE_LENGTH:
            return 0

        try:
            processed_df = QModelV6YOLO_DataProcessor.preprocess_dataframe(
                current_df.copy()
            )

            if processed_df is not None and not processed_df.empty:
                pred = self.predict(processed_df)
                self.current_prediction = pred
                return pred
            else:
                Log.w(self.TAG, "Preprocessing returned empty/None DataFrame.")

        except Exception as e:
            Log.e(self.TAG, f"Inference failed: {str(e)}")
        return self.current_prediction

    def get_status_str(self) -> str:
        """
        Retrieves the string representation of the current prediction.

        Returns:
            str: The human-readable status string (e.g., "1 Channel").
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

    def __init__(
        self,
        queue_log: multiprocessing.Queue,
        queue_in: multiprocessing.Queue,
        queue_out: multiprocessing.Queue,
        model_path: str,
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
        self._queueLog: multiprocessing.Queue = queue_log
        multiprocessing.Process.__init__(self)

        self._exit = multiprocessing.Event()
        self._done = multiprocessing.Event()
        self._queue_in: multiprocessing.Queue = queue_in
        self._queue_out: multiprocessing.Queue = queue_out

        # Store config to initialize model inside run()
        self.model_path = model_path
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
                    raw_worker_data = self._queue_in.get()

                    try:
                        df_chunk = QModelV6YOLO_DataProcessor.convert_to_dataframe(
                            raw_worker_data
                        )

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
