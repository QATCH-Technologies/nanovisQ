import os
import sys
import logging
import multiprocessing
from logging.handlers import QueueHandler
from typing import Any, Optional, Dict
from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from enum import Enum

from QATCH.QModel.src.models.live.stopnet import StopNet

TAG = "[StopNetProcess]"


class FillStatus(Enum):
    """Enum for fill statuses during prediction updates."""
    NO_FILL = 0
    INITIAL_FILL = 1
    CHANNEL_1_FILL = 2
    CHANNEL_2_FILL = 3
    FULL_FILL = 4


# Map your POI codes to state names
POI_STATE_MAP = {
    0: FillStatus.NO_FILL,
    1: FillStatus.INITIAL_FILL,
    4: FillStatus.CHANNEL_1_FILL,
    5: FillStatus.CHANNEL_2_FILL,
    6: FillStatus.FULL_FILL,
}


class StopNetProcess(multiprocessing.Process):
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
        self._current_state: FillStatus = FillStatus.NO_FILL
        self._last_seen_max_time = -1

    def run(self) -> None:
        try:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            logger = logging.getLogger("QATCH.logger")
            logger.addHandler(QueueHandler(self._queueLog))
            logger.setLevel(logging.DEBUG)

            from multiprocessing.util import get_logger
            mp_logger = get_logger()
            mp_logger.handlers[0].setStream(sys.stderr)
            mp_logger.setLevel(logging.WARNING)

            # ─── Load StopNet + scalers ─────────────────────────────────────────
            base = Architecture.get_path()
            print(base)
            model_p = os.path.join(base, "QATCH", "QModel", "SavedModels",
                                   "stopnet_assets", "stopnet.h5")
            t_scale = os.path.join(base, "QATCH", "QModel", "SavedModels",
                                   "stopnet_assets", "stopnet_scalers", "Relative_time_scaler.pkl")
            rf_scale = os.path.join(base, "QATCH", "QModel", "SavedModels",
                                    "stopnet_assets", "stopnet_scalers", "Resonance_Frequency_scaler.pkl")
            d_scale = os.path.join(base, "QATCH", "QModel", "SavedModels",
                                   "stopnet_assets", "stopnet_scalers", "Dissipation_scaler.pkl")

            self._stopnet = StopNet(
                model_path=model_p,
                time_scaler_path=t_scale,
                rf_scaler_path=rf_scale,
                diss_scaler_path=d_scale,
            )
            Log.d(TAG, "StopNet loaded; entering main loop.")

            while not self._exit.is_set():
                new_data = self._queue_in.get()
                while not self._queue_in.empty():
                    new_data = self._queue_in.get()
                if hasattr(new_data, "empty") and not new_data.empty:
                    for _, row in new_data.iterrows():
                        r_time = row["Relative_time"].max()
                        if r_time > self._last_seen_max_time:
                            self._last_seen_max_time = r_time
                            result = self._stopnet.add_data_point(
                                relative_time=row["Relative_time"],
                                dissipation=row["Dissipation"],
                                resonance_freq=row["Resonance_Frequency"],
                            )
                        if result is None:
                            continue

                        poi, _, _ = result
                        new_state = POI_STATE_MAP[poi]
                        if new_state.value > self._current_state.value:
                            self._current_state = new_state
                            self._queue_out.put(new_state)
        except Exception:
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            Log.e("Traceback (most recent call last):")
            for line in format_tb(tb):
                Log.e(line.rstrip())
            Log.e(f"{t.__name__}: {v}")

        finally:
            Log.d(TAG, "StopNetForecasterProcess stopped.")
            self._current_state = FillStatus.NO_FILL
            self._done.set()

    def stop(self) -> None:
        """Signal the process to exit gracefully."""
        self._current_state = FillStatus.NO_FILL
        self._exit.set()

    def is_running(self) -> bool:
        """Returns False once the process has fully shut down."""
        return not self._done.is_set()
