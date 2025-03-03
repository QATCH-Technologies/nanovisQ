# from QATCH.core.constants import Constants
# from QATCH.core.ringBuffer import RingBuffer
from QATCH.common.architecture import Architecture
# from QATCH.common.fileStorage import FileStorage
from QATCH.common.logger import Logger as Log
from QATCH.QModel.q_forecaster import QForecasterPredictor
# import numpy as np
import multiprocessing
from time import time
import logging
from logging.handlers import QueueHandler
import sys
import os

TAG = "[FillForecaster]"

################################################################################
# Forecaster for the raw data gathered from the SerialProcess in parallel timing
# Summary: This multiprocessor thread handles non-blocking real-time predictions
#          to provide the user a live estimation of the run's current fill state
# More Info: See QForecasterPredictor model in file QATCH\QModel\q_forecaster.py
################################################################################


class FillForecasterProcess(multiprocessing.Process):

    def __init__(self, queue_log, queue_in, queue_out):
        """
        :param parser_process: Reference to a ParserProcess instance.
        :type parser_process: ParserProcess.
        """
        self._queueLog = queue_log  # Log.create()

        multiprocessing.Process.__init__(self)
        self._exit = multiprocessing.Event()
        self._done = multiprocessing.Event()
        self._queue_in = queue_in
        self._queue_out = queue_out

    ###########################################################################
    # Processes incoming data and calculates outcoming data
    ###########################################################################

    def run(self):
        try:
            # Log.create()
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
            qfp_path = os.path.join(Architecture.get_path(),
                                    r"QATCH\QModel\SavedModels\forecaster")
            self._forecaster = QForecasterPredictor(
                batch_threshold=50, save_dir=qfp_path)
            self._forecaster.load_models()
            self.forecast_predictions = {'status': 'init'}

            while not self._exit.is_set():
                # Log.d"waiting for data!")
                while self._queue_in.empty() and not self._exit.is_set():
                    pass  # wait for queue to fill (or exit)

                new_data = None
                while not self._queue_in.empty():
                    # Read in all available datas in queue, only process most-recent one
                    new_data = self._queue_in.get()

                # Log.d("got data!")
                if new_data is not None and not new_data.empty:
                    # Log.d(in_q)

                    # call FillForecaster to process in-queued data
                    self.forecast_predictions = self._forecaster.update_predictions(
                        new_data=new_data, ignore_before=50)

                if self.forecast_predictions['status'] == 'completed':
                    # add data to FillForecasterProcess out queue (back to MainWindow) for downstream LoggerProcess
                    Log.d(
                        TAG, f"Accumulated Count: {self.forecast_predictions['accumulated_count']}")
                    if self.forecast_predictions['selected_model']:
                        Log.d(
                            TAG, f"Long Model: {self.forecast_predictions['pred_long'][-1]}")
                    else:
                        Log.d(
                            TAG, f"Short Model: {self.forecast_predictions['pred_short'][-1]}")
                    self._queue_out.put(self.forecast_predictions)

        except:
            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            a_list = ['Traceback (most recent call last):']
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        finally:
            Log.d(" FillForecasterProcess stopped.")

            # gracefully end subprocess
            self._done.set()

    def is_running(self):
        return not self._done.is_set()

    def stop(self):
        # Signals the process to stop when the parent process stops
        self._exit.set()
