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
    2026-04-10

Version:
    2.1.3
"""

import logging
import multiprocessing
import os
import sys
from logging.handlers import QueueHandler
from queue import Empty
from typing import Dict, NamedTuple, Optional, Tuple

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


class DropEpochSignal(NamedTuple):
    """Sentinel put into the forecaster input queue by the UI when the drop is
    detected ('Sample detected' state).  The ``relative_time`` value is the
    Relative_time (seconds) at that moment and is used to seed ``_fill_epoch``
    in ``QModelV6YOLO_Live`` so fill-duration timers start at drop application
    rather than at the first 'Filling started' model prediction.
    """

    relative_time: float


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

    # Per-channel fill duration rules for confirmed channels.
    # Timed thresholds are re-evaluated each debounce-satisfied cycle so
    # warnings can fire even when the channel remains stable.
    # Key   : channel count (matches current_prediction after state change)
    # Value : (threshold_seconds, display_message)
    #   threshold_seconds = None  -> message fires unconditionally on confirmation.
    #   threshold_seconds = float -> message fires only if the run's Relative_time at
    #                               confirmation equals or exceeds the threshold.
    DURATION_THRESHOLDS: Dict[int, Tuple[Optional[float], str]] = {
        0: (60.0, "Data Ready, You Can Stop"),  # >= 1 min at Initial Fill, no ch1 yet
        1: (120.0, "Data Ready, You Can Stop"),  # >= 2 min at ch1, no ch2 yet
        3: (None, "Data Ready, Stop"),  # always on 3-channel confirmation
    }
    INITIAL_FILL_TIMEOUT_S: float = 180.0

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
        # Records the fill epoch (Relative_time, seconds) used for duration thresholds.
        # Primary source is the UI-provided drop-applied timestamp; channel-0
        # confirmation only seeds this as a fallback if no drop epoch was provided.
        self._fill_epoch: Optional[float] = None
        self._fill_epoch_source: Optional[str] = None  # "drop_timestamp" or "channel_0_confirm"
        # Tracks which channels have already had their timed duration warning fired so
        # that _evaluate_duration_threshold never double-emits for the same channel.
        self._channel_warning_fired: Dict[int, bool] = {}
        # Tracks which channels have exceeded their duration threshold but are waiting
        # for the *next* channel to be confirmed before the display message is emitted.
        self._extended_fill_latched: Dict[int, bool] = {}
        self._no_fill_timeout_fired: bool = False
        self._cal_csv_written: bool = False
        self._cal_csv_path: str = os.path.join(
            Architecture.get_path(), "QATCH", "QModel", "logs", "fill_calibration_30s.csv"
        )
        Log.i(self.TAG, "Initialized LiveFillClassifier.")

    def set_drop_applied_timestamp(self, relative_time: float) -> None:
        """Seeds the fill epoch from the UI-detected drop-application timestamp.

        Called by :class:`QModelV6YOLO_LiveProcess` when a
        :class:`DropEpochSignal` arrives in the input queue.  Sets
        ``_fill_epoch`` immediately so that all duration thresholds are measured
        from the moment the drop was physically applied, not from the later point
        at which the model first predicts 'Filling started' (channel 0).

        The epoch is only set once; subsequent calls are silently ignored so that
        the channel-0 confirmation path cannot accidentally overwrite it.

        Args:
            relative_time: The Relative_time value (seconds) recorded by the UI
                at the instant the drop was detected.
        """
        if self._fill_epoch is None or self._fill_epoch_source == "channel_0_fallback":
            self._fill_epoch = relative_time
            self._fill_epoch_source = "drop_signal"
            Log.i(
                self.TAG,
                f"Fill epoch seeded from drop-applied timestamp: {relative_time:.1f} s.",
            )
        else:
            Log.d(
                self.TAG,
                f"Fill epoch already set ({self._fill_epoch:.1f} s) - ignoring "
                f"drop-applied timestamp {relative_time:.1f} s.",
            )

    def _check_initial_fill_timeout(self) -> None:
        """Fires a 'Data Ready, Stop' signal if 3 minutes elapse after Initial Fill
         is confirmed without 1 Channel being detected.

        Requires all three conditions to be true simultaneously:
        - Current prediction is 0 (Initial Fill confirmed, not yet ch1).
        - Channel 0 confirmation time is recorded.
        - Elapsed time since channel 0 confirmation >= INITIAL_FILL_TIMEOUT_S.

        Fires at most once per run, guarded by ``_initial_fill_timeout_fired``.
        """
        if self._initial_fill_timeout_fired:
            return
        if self.current_prediction < 1:
            return

        ch0_time = self._channel_confirm_times.get(0)
        if ch0_time is None:
            return

        elapsed_s: float = max(self._last_max_time, 0.0) - ch0_time
        if elapsed_s >= self.INITIAL_FILL_TIMEOUT_S:
            Log.w(
                self.TAG,
                f"Initial Fill (ch0) confirmed at {ch0_time:.1f} s but no 1-Channel "
                f"state detected after {elapsed_s:.1f} s "
                f"(threshold {self.INITIAL_FILL_TIMEOUT_S / 60:.0f} min). "
                f"Emitting 'Data Ready, Stop'.",
            )
            self._pending_display_message = "Data Ready, Stop"
            self._initial_fill_timeout_fired = True

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

                    # Fire one-time bookkeeping only on genuine state transitions.
                    if pred != previous_prediction:
                        self._on_channel_confirmed(pred)

                    # Re-evaluate timed thresholds every cycle so warnings fire even
                    # when the channel has been stable since first confirmation.
                    self._evaluate_duration_threshold(self.current_prediction)
                self._check_initial_fill_timeout()
            else:
                Log.w(self.TAG, "Preprocessing returned empty/None DataFrame.")

        except Exception as e:
            Log.e(self.TAG, f"Inference failed: {str(e)}")

        return self.current_prediction

    def _on_channel_confirmed(self, channel: int) -> None:
        """Handles one-time bookkeeping when a new channel state is debounce-confirmed.

        Records the confirmation timestamp, sets the fill epoch when channel 0
        (Initial Fill) is first confirmed, and immediately emits any
        unconditional display message (``DURATION_THRESHOLDS`` entries whose
        threshold is ``None``).

        Timed threshold evaluation (120 s / 240 s) is **not** performed here;
        it runs every classification cycle via
        :meth:`_evaluate_duration_threshold` so that warnings fire even when
        the channel has been stable since initial confirmation.

        This method is live-only; it does not alter prediction values or buffer
        state.

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
                self._fill_epoch_source = "channel_0_fallback"
                Log.i(
                    self.TAG,
                    f"Initial Fill confirmed at {confirm_time:.1f} s - fill epoch set.",
                )
            else:
                Log.d(
                    self.TAG,
                    f"Initial Fill reconfirmed at {confirm_time:.1f} s - keeping original epoch "
                    f"{self._fill_epoch:.1f} s.",
                )

        # Check whether the previous channel's extended-fill latch was armed.
        # If so, now that this channel has finally been confirmed, emit the message.
        # This must run before any early return so that channels not in
        # DURATION_THRESHOLDS (e.g. channel 2) still release the latch for
        # channel 1 when they are confirmed.
        prev_channel = channel - 1
        if self._extended_fill_latched.get(prev_channel, False):
            _, prev_message = self.DURATION_THRESHOLDS.get(prev_channel, (None, None))
            if prev_message is not None:
                Log.i(
                    self.TAG,
                    f"Extended fill latch for channel {prev_channel} released on "
                    f"channel {channel} confirmation - displaying: '{prev_message}'",
                )
                self._pending_display_message = prev_message
                # Consume the latch so it cannot fire again.
                self._extended_fill_latched[prev_channel] = False

        if channel not in self.DURATION_THRESHOLDS:
            return

        threshold_s, message = self.DURATION_THRESHOLDS[channel]

        if threshold_s is None:
            # emit immediately on confirmation (e.g. 3-channel complete).
            Log.i(
                self.TAG,
                f"Channel {channel} fill complete - displaying: '{message}'",
            )
            self._pending_display_message = message
            self._channel_warning_fired[channel] = True

    def _evaluate_duration_threshold(self, channel: int) -> None:
        """Evaluates timed fill-duration thresholds for the currently stable channel.

        Called on every classification cycle (not just at state transitions) so
        that 120 s / 240 s warnings fire even when the channel has been
        stable since its first confirmation.

        Only channels with a non-``None`` threshold in :attr:`DURATION_THRESHOLDS`
        are evaluated.  Each warning fires at most once per channel, guarded by
        :attr:`_channel_warning_fired`.

        Args:
            channel (int): The currently confirmed channel count.
        """
        if channel not in self.DURATION_THRESHOLDS:
            return

        threshold_s, message = self.DURATION_THRESHOLDS[channel]

        if threshold_s is None:
            return

        if self._channel_warning_fired.get(channel, False):
            return

        confirm_time = self._channel_confirm_times.get(channel)
        if confirm_time is None:
            return

        if self._fill_epoch is None:
            return  # drop epoch not yet established; cannot evaluate fill duration

        current_time: float = max(self._last_max_time, 0.0)
        elapsed_s: float = current_time - self._fill_epoch

        if elapsed_s >= threshold_s:
            threshold_min = threshold_s / 60.0
            elapsed_min = elapsed_s / 60.0
            Log.w(
                self.TAG,
                f"Extended fill detected: channel {channel} at {elapsed_s:.1f} s "
                f"since fill epoch (threshold {threshold_min:.0f} min, "
                f"elapsed {elapsed_min:.2f} min). Latching display message until "
                f"next channel is confirmed: '{message}'",
            )
            # Do NOT emit the display message here. Instead arm the latch so that
            # _on_channel_confirmed will emit it once the next channel is detected.
            self._extended_fill_latched[channel] = True
            self._channel_warning_fired[channel] = True

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
        type_cls_asset = os.path.join(v6_base_path, "classifiers", "fill_classifier", "type_cls.pt")
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
                    if isinstance(chunk, DropEpochSignal):
                        self._classifier.set_drop_applied_timestamp(chunk.relative_time)
                        continue
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
