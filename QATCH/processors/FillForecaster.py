"""
Forecaster for raw data gathered from the SerialProcess in parallel timing.

This module provides a multiprocessor process that handles non-blocking real-time
predictions to offer a live estimation of the current fill state during a run.
It utilizes the QForecasterPredictor model from QATCH.QModel.q_forecaster to update
predictions based on incoming data received via a multiprocessing queue.

Author(s):
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    04-07-2025

Version:
    V2
"""

import os
import sys
import logging
import multiprocessing
from logging.handlers import QueueHandler
from typing import Any, Optional, Dict
from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.QModel.src.models.live.q_forecast_predictor import QForecastPredictor, FillStatus

TAG = "[FillForecaster]"


class FillForecasterProcess(multiprocessing.Process):
    """Process for handling real-time fill state predictions.

    This multiprocess-based class continuously reads incoming data from a queue,
    processes it using a forecaster model, and sends completed prediction results
    to an output queue. It is designed to operate in parallel with the main
    application to provide non-blocking prediction updates.
    """

    def __init__(self,
                 queue_log: multiprocessing.Queue,
                 queue_in: multiprocessing.Queue,
                 queue_out: multiprocessing.Queue) -> None:
        """Initializes the FillForecasterProcess.

        Params:
            queue_log (multiprocessing.Queue): Queue used for logging messages.
            queue_in (multiprocessing.Queue): Queue from which raw data is received.
            queue_out (multiprocessing.Queue): Queue to output processed prediction results.

        Returns:
            None
        """
        self._queueLog: multiprocessing.Queue = queue_log
        multiprocessing.Process.__init__(self)
        self._exit = multiprocessing.Event()
        self._done = multiprocessing.Event()
        self._queue_in: multiprocessing.Queue = queue_in
        self._queue_out: multiprocessing.Queue = queue_out
        self.state: FillStatus = FillStatus.NO_FILL

    def run(self) -> None:
        """Runs the process to process incoming data and compute predictions.

        This method performs the following steps:
          1. Redirects stdout and stderr to suppress console output.
          2. Configures the logger for the process and sets up the multiprocessing logger.
          3. Initializes the QForecasterPredictor with a specified model path, loads the model,
             and sets an initial prediction status.
          4. Enters a loop that waits for incoming data on the input queue.
             - It processes only the most recent data available.
             - If the new data is valid and non-empty, it updates the prediction results.
          5. If a prediction is marked as 'completed', it puts the result into the output queue.
          6. Handles any exceptions by logging a detailed traceback.
          7. Upon termination, logs the process shutdown and sets the done event.
        """
        try:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

            logger = logging.getLogger("QATCH.logger")
            logger.addHandler(QueueHandler(self._queueLog))
            logger.setLevel(logging.DEBUG)

            from multiprocessing.util import get_logger
            multiprocessing_logger = get_logger()
            multiprocessing_logger.handlers[0].setStream(sys.stderr)
            multiprocessing_logger.setLevel(logging.WARNING)

            # Run once actions, on start of process:
            start_booster_path = os.path.join(Architecture.get_path(),
                                              r"QATCH\QModel\SavedModels\forecaster_v2", 'bff_trained_start.json')
            end_booster_path = os.path.join(Architecture.get_path(),
                                            r"QATCH\QModel\SavedModels\forecaster_v2", 'bff_trained_end.json')
            scaler_path = os.path.join(Architecture.get_path(),
                                       r"QATCH\QModel\SavedModels\forecaster_v2", 'scaler.pkl')
            self._forecaster = QForecastPredictor(
                start_booster_path=start_booster_path, end_booster_path=end_booster_path, scaler_path=scaler_path)
            self.state = FillStatus.NO_FILL
            while not self._exit.is_set():
                while self._queue_in.empty() and not self._exit.is_set():
                    pass

                new_data = None

                # Read all available data from the queue; process only the most recent one.
                while not self._queue_in.empty():
                    new_data = self._queue_in.get()

                # Process the new data if it exists and is not empty.
                if new_data is not None and not new_data.empty:
                    self._forecaster.update_predictions(
                        new_data=new_data)
                self._queue_out.put([
                    self._forecaster.get_fill_status(),
                    self._forecaster.get_start_time(),
                    self._forecaster.get_end_time()
                ])
        except Exception:
            # Capture and log the traceback in case of an exception.
            limit: Optional[int] = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            a_list = ['Traceback (most recent call last):']
            a_list += format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        finally:
            # Gracefully end the subprocess and log the shutdown.
            Log.d(TAG, "FillForecasterProcess stopped.")
            self._done.set()

    def is_running(self) -> bool:
        """Checks if the process is still running.

        Returns:
            bool: True if the process is active, False if it has completed or stopped.
        """
        return not self._done.is_set()

    def stop(self) -> None:
        """Signals the process to stop.

        Sets the exit event, which causes the main loop in run() to terminate gracefully.
        """
        self._exit.set()
