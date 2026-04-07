"""
v6_yolo_live.py

This module provides the infrastructure for running a YOLO-based fill classifier
in a live, multiprocessing environment. It includes a classification logic class
that manages data buffering and prediction, as well as a dedicated multiprocessing
wrapper to handle execution in a separate process, ensuring non-blocking performance
for the main application.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2026-04-07

Version:
    2.1.1
"""

import logging
import multiprocessing
import os
import sys
from logging.handlers import QueueHandler
from queue import Empty
from typing import Dict, Optional, Tuple

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

    Handles accumulation of streaming sensor data, maintains a fixed-size sliding window
    buffer, and triggers inference using a YOLO-based model when conditions are met.

    In addition to standard classification, this class tracks when each channel state is
    first confirmed (post-debounce) and applies duration thresholds to emit one-shot
    on-display messages and warning logs. These diagnostics are live-only and do not
    affect the underlying model or prediction values.

    Attributes:
        STATUS_MAP (dict): Mapping of integer classification codes to human-readable strings.
        TAG (str): Log tag prefix for this class.
        DEBOUNCE_THRESHOLD (int): Number of consecutive identical predictions required
            before a state change is accepted.
        DURATION_THRESHOLDS (dict): Per-channel fill duration thresholds in seconds above
            which a warning is logged and a display message is emitted. A ``None`` threshold
            means the message is always emitted on that channel's confirmation.
        buffer_window_size (Optional[int]): The maximum number of rows to retain in the
            rolling buffer. If None, the buffer grows indefinitely.
        current_prediction (int): The most recent classification result class ID.
    """

    STATUS_MAP = {
        -1: "No Fill",
        0: "Initial Fill",
        1: "1 Channel",
        2: "2 Channels",
        3: "3 Channels",
    }
    TAG = "[QModelV6YOLO_Live]"

    # Number of consecutive identical predictions required before accepting a state change.
    DEBOUNCE_THRESHOLD = 3

    # Per-channel fill duration rules applied at the moment debounce confirmation fires.
    # Key   : channel count (matches current_prediction after state change)
    # Value : (threshold_seconds, display_message)
    #   threshold_seconds = None  -> message fires unconditionally on confirmation.
    #   threshold_seconds = float -> message fires only if the run's Relative_time at
    #                               confirmation equals or exceeds the threshold.
    DURATION_THRESHOLDS: Dict[int, Tuple[Optional[float], str]] = {
        1: (120.0, "Data Ready, You Can Stop"),  # ≥ 2 minutes
        2: (240.0, "Data Ready, You Can Stop"),  # ≥ 4 minutes
        3: (None, "Complete, Press Stop"),  # always on 3-channel confirmation
    }

    def __init__(self, model_path: str, buffer_window_size: Optional[int] = None):
        """Initializes the live fill classifier with a model and buffer settings.

        Args:
            model_path (str): The file path to the trained YOLO model weights.
            buffer_window_size (Optional[int]): The maximum number of data rows to keep
                in memory. Defaults to None.
        """
        super().__init__(model_path)
        self.buffer_window_size = buffer_window_size
        self.current_prediction = -1

        # Core buffer attributes
        self._data: Optional[pd.DataFrame] = None
        self._last_max_time = -float("inf")
        self._prediction_buffer_size = 0

        # Debounce state: candidate and consecutive count
        self._debounce_candidate = -1
        self._debounce_count = 0

        # Records the Relative_time (seconds) at which each channel was first confirmed.
        self._channel_confirm_times: Dict[int, float] = {}
        # Holds the next on-display message to be consumed by the process layer.
        # Cleared to None immediately after being read via get_and_clear_display_message().
        self._pending_display_message: Optional[str] = None
        # Records the Relative_time (seconds) when Initial Fill (channel 0) is first
        # confirmed. All fill-duration thresholds are measured from this point, not
        # from Relative_time = 0 (run start), so pre-fill time is excluded.
        self._fill_epoch: Optional[float] = None

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
            new_data_filtered = new_data[new_data["Relative_time"] > self._last_max_time]

            if not new_data_filtered.empty:
                new_data_aligned = new_data_filtered.reindex(columns=self._data.columns)
                self._data = pd.concat([self._data, new_data_aligned], ignore_index=True)
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

        When debounce confirms a new channel state this method also:

        * Records the confirmation time in ``_channel_confirm_times``.
        * Compares the elapsed run time against ``DURATION_THRESHOLDS``.
        * Emits a ``Log.w`` warning if the fill exceeded the expected duration.
        * Stores a one-shot display message in ``_pending_display_message`` for the
          process layer to forward to the main UI.

        Returns:
            int: The classification result class ID. Returns the previous `current_prediction`
            if the buffer is insufficient, empty, or if inference fails.
        """
        if self._data is None or len(self._data) < QModelV6Config.MIN_SLICE_LENGTH:
            return self.current_prediction

        try:
            filtered = self._data[self._data["Relative_time"] > 0.05]
            if filtered.empty:
                Log.d(self.TAG, "Waiting for > 0.05s of buffer data.")
                return self.current_prediction

            processed_df = QModelV6YOLO_DataProcessor.preprocess_dataframe(filtered.copy())
            if processed_df is not None and not processed_df.empty:
                pred = self.predict(processed_df)
                if pred == self._debounce_candidate:
                    self._debounce_count += 1
                else:
                    self._debounce_candidate = pred
                    self._debounce_count = 1

                if self._debounce_count >= self.DEBOUNCE_THRESHOLD:
                    previous_prediction = self.current_prediction
                    self.current_prediction = pred

                    # Fire diagnostic logic only on genuine state transitions
                    if pred != previous_prediction:
                        self._on_channel_confirmed(pred)
            else:
                Log.w(self.TAG, "Preprocessing returned empty/None DataFrame.")

        except Exception as e:
            Log.e(self.TAG, f"Inference failed: {str(e)}")

        return self.current_prediction

    def _on_channel_confirmed(self, channel: int) -> None:
        """Handles side-effects triggered when a new channel state is debounce-confirmed.

        Records the confirmation time, evaluates fill-duration thresholds relative
        to the Initial Fill epoch, emits warning logs for extended fills, and sets
        the pending on-display message.

        The epoch is set on the first confirmation of channel 0 (Initial Fill).
        All duration thresholds for channels 1, 2, and 3 are measured from that
        epoch so that pre-fill dead time (device startup, baseline capture, etc.)
        is excluded from the comparison.

        This method is live-only; it does not alter prediction values or buffer state.

        Args:
            channel (int): The newly confirmed channel count (e.g., 0, 1, 2, 3).
        """
        confirm_time: float = max(self._last_max_time, 0.0)
        self._channel_confirm_times[channel] = confirm_time

        # Capture the epoch on Initial Fill confirmation - all thresholds are
        # measured from this moment forward.
        if channel == 0:
            if self._fill_epoch is None:
                self._fill_epoch = confirm_time
                Log.i(
                    self.TAG,
                    f"Initial Fill confirmed at {confirm_time:.1f} s - fill epoch set.",
                )
            else:
                Log.d(
                    self.TAG,
                    f"Initial Fill reconfirmed at {confirm_time:.1f} s - keeping original epoch "
                    f"{self._fill_epoch:.1f} s.",
                )  # No duration threshold applies to channel 0 itself.

        if channel not in self.DURATION_THRESHOLDS:
            return

        threshold_s, message = self.DURATION_THRESHOLDS[channel]

        # Compute elapsed time since Initial Fill was confirmed.
        if self._fill_epoch is not None:
            elapsed_s: float = confirm_time - self._fill_epoch
            epoch_note = f"{elapsed_s:.1f} s since Initial Fill"
        else:
            # Edge case: classifier jumped straight past channel 0 (e.g. model
            # started mid-fill). Fall back to absolute run time so the logic
            # still functions, but flag it clearly in the log.
            elapsed_s = confirm_time
            epoch_note = f"{elapsed_s:.1f} s (no Initial Fill epoch - using absolute time)"
            Log.w(
                self.TAG,
                f"Channel {channel} confirmed but no Initial Fill epoch was recorded. "
                "Duration threshold will be evaluated against absolute run time.",
            )

        elapsed_min: float = elapsed_s / 60.0
        Log.i(
            self.TAG,
            f"Channel {channel} confirmed - {epoch_note} ({elapsed_min:.2f} min).",
        )

        if threshold_s is None:
            # Unconditional - always emit (e.g. 3-channel complete).
            Log.i(
                self.TAG,
                f"Channel {channel} fill complete - displaying: '{message}'",
            )
            self._pending_display_message = message

        elif elapsed_s >= threshold_s:
            threshold_min = threshold_s / 60.0
            Log.w(
                self.TAG,
                f"Extended fill detected: channel {channel} took {elapsed_min:.2f} min "
                f"since Initial Fill (threshold {threshold_min:.0f} min). "
                f"Displaying: '{message}'",
            )
            self._pending_display_message = message

        else:
            Log.d(
                self.TAG,
                f"Channel {channel} fill within normal duration "
                f"({elapsed_min:.2f} min < {threshold_s / 60.0:.0f} min threshold). "
                "No display message emitted.",
            )

    def get_and_clear_display_message(self) -> Optional[str]:
        """Returns the pending on-display message and clears it atomically.

        The message is set at most once per state transition and is consumed by
        the process layer so the main UI displays it exactly once.

        Returns:
            Optional[str]: The pending display message, or ``None`` if no message
            is waiting to be displayed.
        """
        msg = self._pending_display_message
        self._pending_display_message = None
        return msg

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

    Output queue items are ``(int, str, Optional[str])`` tuples:

    * ``int``  - the raw prediction class ID (e.g., -1, 0, 1, 2, 3).
    * ``str``  - the human-readable status label (e.g., "2 Channels").
    * ``Optional[str]`` - a one-shot on-display message for the main UI, or ``None``
      if no message is pending. The message is emitted at most once per channel
      state transition and is consumed by the first call that reads it.

    Attributes:
        _queueLog (multiprocessing.Queue): Queue for thread-safe logging.
        _exit (multiprocessing.Event): Event flag to signal process termination.
        _done (multiprocessing.Event): Event flag to signal process completion.
        _queue_in (multiprocessing.Queue): Input queue receiving raw worker data.
        _queue_out (multiprocessing.Queue): Output queue sending
            ``(int, str, Optional[str])`` prediction tuples.
        model_path (str): Path to the YOLO model file.
        buffer_window_size (Optional[int]): Rolling window size for the model buffer.
        _classifier (Optional[QModelV6YOLO_Live]): The internal classifier instance
            (created inside ``run()``).
    """

    TAG = "[QModelV6YOLO_LiveProcess]"

    def __init__(
        self,
        queue_log: multiprocessing.Queue,
        queue_in: multiprocessing.Queue,
        queue_out: multiprocessing.Queue,
        buffer_window_size: Optional[int] = None,
    ) -> None:
        """Initializes the LiveProcess with queue handles and buffer configuration.

        The YOLO model is intentionally not loaded here; it is loaded inside ``run()``
        to avoid pickling errors when the process is spawned.

        Args:
            queue_log (multiprocessing.Queue): Queue for forwarding log records back to
                the main process.
            queue_in (multiprocessing.Queue): Input queue that delivers raw worker data
                chunks for inference.
            queue_out (multiprocessing.Queue): Output queue that receives
                ``(int, str, Optional[str])`` tuples produced after each inference batch.
                The third element is a one-shot display message for the main UI, or
                ``None`` when no message is pending.
            buffer_window_size (Optional[int]): Maximum number of rows to keep in the
                rolling data buffer. Defaults to None (unbounded).
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
        type_cls_asset = os.path.join(
            v6_base_path, "classifiers", "fill_classifier", "weights", "best.pt"
        )
        self.model_path = type_cls_asset
        self.buffer_window_size = buffer_window_size

        # Instance placeholder
        self._classifier: Optional[QModelV6YOLO_Live] = None

    def run(self) -> None:
        """Executes the main inference loop for the live fill classification process.

        This method is called automatically when the process is started via
        ``multiprocessing.Process.start()``. It performs the following steps in order:

        1. Redirects ``stdout`` and ``stderr`` to ``/dev/null`` to suppress console output
           from the YOLO runtime.
        2. Configures the process-local logger to forward records through ``_queueLog``
           so they appear in the main-process log.
        3. Initializes the ``QModelV6YOLO_Live`` classifier (must occur here to avoid
           pickling errors).
        4. Enters the main loop: blocks on ``_queue_in`` with a short timeout, drains
           any additional queued chunks, feeds them to the classifier, then runs a single
           inference pass.
        5. Publishes ``(pred_int, pred_str, display_message)`` to ``_queue_out`` after
           each inference batch.  ``display_message`` is the result of
           ``get_and_clear_display_message()`` - a non-``None`` value means the main UI
           should display it; subsequent puts will carry ``None`` until the next
           qualifying state transition.
        6. On exit (normal or exceptional), closes the devnull handle and sets
           ``_done`` to signal callers that the process has finished.

        Raises:
            Exception: Any unhandled exception is caught, logged line-by-line via
                ``Log.e``, and then the ``finally`` block cleans up.
        """
        devnull = open(os.devnull, "w")
        mp_devnull = None
        try:
            sys.stdout = sys.stderr = devnull

            logger = logging.getLogger("QATCH.logger")
            logger.addHandler(QueueHandler(self._queueLog))
            logger.setLevel(logging.DEBUG)

            from multiprocessing.util import get_logger

            mp_devnull = open(os.devnull, "w")
            mp_logger = get_logger()
            if mp_logger.handlers:
                mp_logger.handlers[0].setStream(mp_devnull)
            mp_logger.setLevel(logging.WARNING)

            self._classifier = QModelV6YOLO_Live(
                model_path=self.model_path,
                buffer_window_size=self.buffer_window_size,
            )
            Log.i(TAG, "YOLO Live Process Started and Model Loaded.")

            while not self._exit.is_set():
                try:
                    raw_data = self._queue_in.get(timeout=0.05)
                except Empty:
                    continue

                chunks = [raw_data]
                # Drain any additional queued items
                while True:
                    try:
                        chunks.append(self._queue_in.get_nowait())
                    except Empty:
                        break

                data_received = False
                for chunk in chunks:
                    df_chunk = QModelV6YOLO_DataProcessor.convert_to_dataframe(chunk)
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
                    # Consume the one-shot display message (None if nothing pending).
                    display_message: Optional[str] = (
                        self._classifier.get_and_clear_display_message()
                    )
                    self._queue_out.put((pred_int, pred_str, display_message))

        except Exception:
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
            if mp_devnull is not None:
                mp_devnull.close()
            devnull.close()
            self._done.set()

    def is_running(self) -> bool:
        """Checks whether the process is still executing.

        Returns:
            bool: ``True`` if the process has not yet set its completion event,
            ``False`` once ``run()`` has exited (successfully or otherwise).
        """
        return not self._done.is_set()

    def stop(self) -> None:
        """Signals the process to terminate gracefully.

        Sets the internal exit event. The main loop in ``run()`` checks this event
        on each iteration and will exit at the next opportunity without interrupting
        an in-progress inference call.
        """
        self._exit.set()
