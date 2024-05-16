import multiprocessing
from QATCH.common.logger import Logger as Log
import logging
from logging.handlers import QueueHandler


TAG = ""#"[Parser]"

###############################################################################
# Process to parse incoming data and distribute it to worker
###############################################################################
class ParserProcess(multiprocessing.Process):


    ###########################################################################
    # Initializing values for process
    ###########################################################################
    def __init__(self, queue_log,
                       data_queue0,
                       data_queue1,
                       data_queue2,
                       data_queue3,
                       data_queue4,
                       data_queue5,
                       data_queue6):
        """
        :param data_queue{i}: References to queue where processed data will be put.
        :type data_queue{i}: multiprocessing Queue.
        """
        self._queueLog = queue_log #Log.create()
        logger = logging.getLogger("QATCH.logger")
        logger.addHandler(QueueHandler(self._queueLog))
        logger.setLevel(logging.DEBUG)

        multiprocessing.Process.__init__(self)
        self._exit = multiprocessing.Event()

        self._out_queue0 = data_queue0
        self._out_queue1 = data_queue1
        self._out_queue2 = data_queue2
        self._out_queue3 = data_queue3
        self._out_queue4 = data_queue4
        self._out_queue5 = data_queue5
        self._out_queue6 = data_queue6

        #Log.d(TAG, "Process ready")

    ###########################################################################
    # Add new raw data and calculated data to the corresponding internal queue
    ###########################################################################
    def add0(self, data):
        """
        Adds new raw data to internal queue0 (serial data: frequency).
        :param data: Raw data coming from acquisition process.
        :type data: int.
        """
        self._out_queue0.put(data)

    def add1(self, data):
        """
        Adds new raw data to internal queue1 (serial data: amplitude).
        :param data: Raw data coming from acquisition process.
        :type data: float.
        """
        self._out_queue1.put(data)

    def add2(self, data):
        """
        Adds new raw data to internal queue2 (serial data: phase).
        :param data: Raw data coming from acquisition process.
        :type float: float.
        """
        self._out_queue2.put(data)

    def add3(self, data):
        """
        Adds new processed data to internal queue3 (Resonance frequency).
        :param data: Calculated data.
        :type data: float.
        """
        self._out_queue3.put(data)

    def add4(self, data):
        """
        Adds new processed data to internal queue4 (Q-factor/dissipation).
        :param data: Calculated data.
        :type data: float.
        """
        self._out_queue4.put(data)

    def add5(self, data):
        """
        Adds new processed data to internal queue5 (Temperature).
        :param data: Calculated data.
        :type data: float.
        """
        self._out_queue5.put(data)

    def add6(self, data):
        """
        Adds new processed data to internal queue6 (Progress).
        :param data: Calculated data.
        :type data: float.
        """
        self._out_queue6.put(data)

    def stop(self):
        """
        Signals the process to stop parsing data.
        :return:
        """
        Log.d(TAG, "Process finishing...")
        self._exit.set()
