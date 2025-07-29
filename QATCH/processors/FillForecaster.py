import os
import sys
import logging
import multiprocessing
from logging.handlers import QueueHandler
from typing import Any, Optional, Dict
from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log

from QATCH.QModel.src.models.live.stopnet import StopNet

TAG = "[StopNetForecaster]"

# Map your POI codes to state names
POI_STATE_MAP = {
    1: "initial_fill",
    4: "channel1_fill",
    5: "channel2_fill",
    6: "full_fill",
}


class StopNetForecasterProcess(multiprocessing.Process):
    """Process for handling real-time StopNet predictions as a state machine."""

    def __init__(
        self,
        queue_log: multiprocessing.Queue,
        queue_in:  multiprocessing.Queue,
        queue_out: multiprocessing.Queue,
    ) -> None:
        super().__init__()
        self._queueLog = queue_log
        self._queue_in = queue_in
        self._queue_out = queue_out
        self._exit = multiprocessing.Event()
        self._done = multiprocessing.Event()
        # track the last reported state to suppress duplicates
        self._current_state: Optional[str] = None

    def run(self) -> None:
        try:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

            # setup logging over the queue
            logger = logging.getLogger("QATCH.logger")
            logger.addHandler(QueueHandler(self._queueLog))
            logger.setLevel(logging.DEBUG)

            from multiprocessing.util import get_logger
            mp_logger = get_logger()
            mp_logger.handlers[0].setStream(sys.stderr)
            mp_logger.setLevel(logging.WARNING)

            # ─── Load StopNet + scalers ─────────────────────────────────────────
            base = Architecture.get_path()
            model_p = os.path.join(base, "QATCH", "QModel", "SavedModels",
                                   "forecaster_v2", "stopnet_model.h5")
            t_scale = os.path.join(base, "QATCH", "QModel", "SavedModels",
                                   "forecaster_v2", "time_scaler.pkl")
            rf_scale = os.path.join(base, "QATCH", "QModel", "SavedModels",
                                    "forecaster_v2", "rf_scaler.pkl")
            d_scale = os.path.join(base, "QATCH", "QModel", "SavedModels",
                                   "forecaster_v2", "diss_scaler.pkl")

            self._stopnet = StopNet(
                model_path=model_p,
                time_scaler_path=t_scale,
                rf_scaler_path=rf_scale,
                diss_scaler_path=d_scale,
            )
            Log.d(TAG, "StopNet loaded; entering main loop.")

            # ─── Main loop ────────────────────────────────────────────────────────
            while not self._exit.is_set():
                # block until data arrives
                new_data = self._queue_in.get()
                # drop older frames
                while not self._queue_in.empty():
                    new_data = self._queue_in.get()

                # ensure it's a DataFrame with rows
                if hasattr(new_data, "empty") and not new_data.empty:
                    for _, row in new_data.iterrows():
                        result = self._stopnet.add_data_point(
                            relative_time=row["Relative_time"],
                            dissipation=row["Dissipation"],
                            resonance_freq=row["Resonance_Frequency"],
                        )
                        if result is None:
                            continue

                        poi, _, _ = result
                        # only care about non-zero POIs
                        if poi in POI_STATE_MAP:
                            state = POI_STATE_MAP[poi]
                            # only emit if state changed
                            if state != self._current_state:
                                self._current_state = state
                                self._queue_out.put({"fill_state": state})

        except Exception:
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            Log.e("Traceback (most recent call last):")
            for line in format_tb(tb):
                Log.e(line.rstrip())
            Log.e(f"{t.__name__}: {v}")

        finally:
            Log.d(TAG, "StopNetForecasterProcess stopped.")
            self._done.set()

    def stop(self) -> None:
        """Signal the process to exit gracefully."""
        self._exit.set()

    def is_running(self) -> bool:
        """Returns False once the process has fully shut down."""
        return not self._done.is_set()
