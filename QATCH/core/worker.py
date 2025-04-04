import logging
import multiprocessing
from multiprocessing import Queue

from QATCH.core.constants import Constants, OperationType
from QATCH.processors.Parser import ParserProcess
from QATCH.processors.Serial import SerialProcess
from QATCH.processors.Calibration import CalibrationProcess
from QATCH.processors.FillForecaster import FillForecasterProcess
from QATCH.common.fileStorage import FileStorage
from QATCH.common.logger import Logger as Log
from QATCH.core.ringBuffer import RingBuffer
import numpy as np
from time import time

TAG = ""  # "[Worker]"

###############################################################################
# Service that creates and concentrates all processes to run the application
###############################################################################


class Worker:

    ###########################################################################
    # Initializer to configure all processes needed for a basic configuration
    # NOTE: Subsequent configurations should use the config() call instead!
    ###########################################################################
    def __init__(self,
                 QCS_on=None,
                 port=None,
                 speed=Constants.serial_default_overtone,
                 samples=Constants.argument_default_samples,
                 source=OperationType.measurement,
                 export_enabled=False,
                 freq_hopping=False,
                 reconstruct=False):

        # set up logger (only once)
        multiprocessing.log_to_stderr()
        logger = multiprocessing.get_logger()
        logger.setLevel(logging.INFO)

        # configure basic process
        self.config(QCS_on=QCS_on,
                    port=port,
                    speed=speed,
                    samples=samples,
                    source=source,
                    export_enabled=export_enabled,
                    freq_hopping=freq_hopping,
                    reconstruct=reconstruct)

    ###########################################################################
    # Reusable call to configure all processes for acquisition and processing
    ###########################################################################
    def config(self,
               QCS_on=None,
               port=None,
               pid=None,
               speed=Constants.serial_default_overtone,
               samples=Constants.argument_default_samples,
               source=OperationType.measurement,
               export_enabled=False,
               freq_hopping=False,
               reconstruct=False):
        """
        :param port: Port to open on start :type port: str.
        :param speed: Speed for the specified port :type speed: float.
        :param samples: Number of samples :type samples: int.
        :param source: Source type :type source: OperationType.
        :param export_enabled: If true, data will be stored or exported in a file :type export_enabled: bool.
        :param export_path: If specified, defines where the data will be exported :type export_path: str.
        """
        # data queues
        self._queueLog = Queue()
        self._queue0 = Queue()
        self._queue1 = Queue()
        self._queue2 = Queue()
        self._queue3 = Queue()
        self._queue4 = Queue()
        self._queue5 = Queue()
        self._queue6 = Queue()

        # data buffers
        self._data0_buffer = None
        self._data1_buffer = None
        self._data2_buffer = None
        self._d1_buffer = None
        self._d2_buffer = None
        self._d3_buffer = None
        self._d4_buffer = None
        self._t1_buffer = None
        self._t2_buffer = None
        self._t3_buffer = None
        self._ser_error1 = 0
        self._ser_error2 = 0
        self._ser_error3 = 0
        self._ser_error4 = 0
        self._ser_err_usb = 0
        self._control_k = 0

        # instances of the processes
        self._acquisition_process = None
        self._parser_process = None

        # live forecaster model
        self._forecaster_process = None
        self._forecaster_in = Queue()
        self._forecaster_out = Queue()

        # others
        self._QCS_on = QCS_on  # QCS installed on device (unused now)
        self._port = port     # dynamically select COM vs Ethernet
        # overtones (str) if 'serial' is called
        # QCS (str) if 'calibration' is called
        self._pid = pid
        self._speed = speed
        self._samples = samples
        self._source = source
        self._export = export_enabled

        # Supporting variables
        self._d1_store = None  # data storing
        self._d2_store = None  # data storing
        self._readFREQ = None  # frequency range
        self._fStep = None  # sample rate
        self._overtone_name = None  # fundamental/overtones name (str)
        self._overtone_value = None  # fundamental/overtones value(float)
        # self._count = 0 # sweep counter
        self._flag = True
        self._timestart = 0
        self._freq_hopping = freq_hopping
        self._reconstruct = reconstruct

    ###########################################################################
    # Starts all processes, based on configuration given in constructor.
    ###########################################################################

    def start(self):

        if self._source == OperationType.measurement:
            self._samples = Constants.argument_default_samples
        elif self._source == OperationType.calibration:
            self._samples = Constants.calibration_default_samples
            self._readFREQ = Constants.calibration_readFREQ
        # Setup/reset the internal buffers
        self.reset_buffers(self._samples)
        # Instantiates process
        self._parser_process = ParserProcess(
            self._queueLog, self._queue0, self._queue1, self._queue2, self._queue3, self._queue4, self._queue5, self._queue6)
        # Checks the type of source
        if self._source == OperationType.measurement:
            self._acquisition_process = SerialProcess(
                self._queueLog, self._parser_process, self._timestart, self._freq_hopping, self._export, self._reconstruct)
        elif self._source == OperationType.calibration:
            self._acquisition_process = CalibrationProcess(
                self._queueLog, self._parser_process)

        port_and_peak_check = self._acquisition_process.open(
            port=self._port, speed=self._speed, pid=self._pid)
        if port_and_peak_check == 1:
            if self._source == OperationType.measurement:
                # (self._overtone_name,self._overtone_value, self._fStep, self._readFREQ, SG_window_size, spline_points, spline_factor, _, _, _) = self._acquisition_process.get_frequencies(self._samples, 0)
                if self._freq_hopping:
                    (self._overtone_name, self._overtone_value, self._fStep, self._readFREQ, SG_window_size, spline_points, spline_factor, _, _, _,
                     self._overtone_name_up, self._overtone_value_up, self._fStep_up, self._readFREQ_up, SG_window_size_up, spline_points_up, spline_factor_up, _, _, _,
                     self._overtone_name_down, self._overtone_value_down, self._fStep_down, self._readFREQ_down, SG_window_size_down, spline_points_down, spline_factor_down, _, _, _) = self._acquisition_process.get_frequencies(self._samples, 0)
                else:
                    (self._overtone_name, self._overtone_value, self._fStep, self._readFREQ, SG_window_size,
                     spline_points, spline_factor, _, _, _) = self._acquisition_process.get_frequencies(self._samples, 0)

                # Create and start live forecaster
                self._forecaster_process = FillForecasterProcess(
                    self._queueLog,
                    self._forecaster_in,
                    self._forecaster_out)
                self._forecaster_process.start()

                # Prepopulate frequency buffer before starting
                for i in range(len(self._data0_buffer)):
                    self._data0_buffer[i] = self._readFREQ

                Log.i("")
                Log.i(TAG, "DATA MAIN INFORMATION")
                Log.i(TAG, "Selected frequency: {} - {} Hz".format(
                    self._overtone_name, int(self._overtone_value)))
                Log.i(TAG, "Frequency start: {} Hz".format(
                    int(self._readFREQ[0])))
                Log.i(TAG, "Frequency stop:  {} Hz".format(
                    int(self._readFREQ[-1])))
                Log.i(TAG, "Frequency range: {} Hz".format(
                    int(self._readFREQ[-1]-self._readFREQ[0])))
                Log.i(TAG, "Number of samples: {}".format(int(self._samples-1)))
                Log.i(TAG, "Sample rate: {} Hz".format(int(self._fStep)))
                Log.i(TAG, "History buffer size: 5 min\n")
                Log.i(TAG, "MAIN PROCESSING INFORMATION")
                Log.i(TAG, "Method for baseline estimation and correction:")
                Log.i(TAG, "Least Squares Polynomial Fit (LSP)")
                Log.i(TAG, "Degree of the fitting polynomial: 8")

            elif self._source == OperationType.calibration:
                Log.i("")
                Log.i(TAG, "MAIN CALIBRATION INFORMATION")
                Log.i(TAG, "Calibration frequency start:  {} Hz".format(
                    int(Constants.calibration_frequency_start)))
                Log.i(TAG, "Calibration frequency stop:  {} Hz".format(
                    int(Constants.calibration_frequency_stop)))
                Log.i(TAG, "Frequency range: {} Hz".format(
                    int(Constants.calibration_frequency_stop-Constants.calibration_frequency_start)))
                Log.i(TAG, "Number of samples: {}".format(
                    int(Constants.calibration_default_samples-1)))
                Log.i(TAG, "Sample rate: {} Hz".format(
                    int(Constants.calibration_fStep)))
            Log.i(TAG, 'Starting processes...\n')
            # Starts processes
            self._acquisition_process.start()
            self._parser_process.start()
            return port_and_peak_check
        elif port_and_peak_check == 0:
            Log.w(TAG, "Warning: Port is not available")
            return port_and_peak_check
        elif port_and_peak_check == -1:
            Log.w(TAG, "Warning: No peak magnitudes found. Rerun Initialize.")
            return port_and_peak_check

    ###########################################################################
    # Stops all running processes
    ###########################################################################

    def stop(self):
        Log.i(TAG, "Running processes stopped...")

        self._acquisition_process.stop()

        stop = time()
        waitFor = 10  # timeout delay (seconds)
        while time() - stop < waitFor:
            if not self._acquisition_process.is_running():
                break

        if time() - stop >= waitFor:
            Log.w("Threads failed to stop in a timely manner. Forcing termination...")

        self._acquisition_process.terminate()
        self._acquisition_process.join()

        self._parser_process.stop()
        self._parser_process.terminate()
        self._parser_process.join()

        if self._forecaster_process is not None:
            self._forecaster_process.stop()
            self._forecaster_process.terminate()
            self._forecaster_process.join()

        Log.i(TAG, "Processes finished")

    ###########################################################################
    # Empties the internal queues, updating data to consumers
    ###########################################################################

    def consume_logger(self):
        if not self._queueLog.empty():
            logger = logging.getLogger("QATCH")
            while not self._queueLog.empty():
                logger.handle(self._queueLog.get(False))

    def consume_queue0(self):
        # queue0 for serial data: frequency
        while not self._queue0.empty():
            self._queue_data0(self._queue0.get(False))

    def consume_queue1(self):
        # queue1 for serial data: amplitude
        while not self._queue1.empty():
            self._queue_data1(self._queue1.get(False))

    def consume_queue2(self):
        # queue2 for serial data: phase
        while not self._queue2.empty():
            self._queue_data2(self._queue2.get(False))

    def consume_queue3(self):
        # queue3 for elaborated data: resonance frequency
        while not self._queue3.empty():
            self._queue_data3(self._queue3.get(False))

    def consume_queue4(self):
        # queue4 for elaborated data: Q-factor/Dissipation
        while not self._queue4.empty():
            self._queue_data4(self._queue4.get(False))

    def consume_queue5(self):
        # queue5 for elaborated data: Temperature
        while not self._queue5.empty():
            self._queue_data5(self._queue5.get(False))

    def consume_queue6(self):
        # queue6 for elaborated data: errors
        while not self._queue6.empty():
            self._queue_data6(self._queue6.get(False))

    ###########################################################################
    # Adds data to internal buffers.
    ###########################################################################
    def _queue_data0(self, data):
        # :param data: values to add for serial data: frequency :type data: int.
        i = data[0]
        self._data0_buffer[i] = data[1]

    def _queue_data1(self, data):
        # :param data: values to add for serial data: amplitude :type data: float.
        i = data[0]
        self._data1_buffer[i] = data[1]

    #####
    def _queue_data2(self, data):
        # :param data: values to add for serial data phase :type data: float.
        i = data[0]
        self._data2_buffer[i] = data[1]
        # Additional function: exports calibration data in a file if export box is checked.
        '''
        self.store_data_calibration()
        '''
    #####

    def _queue_data3(self, data):
        # :param data: values to add for Resonance frequency :type data: float.
        i = data[0]
        self._t1_store = data[1]  # time (unused)
        self._d1_store = data[2]  # data
        self._t1_buffer[i].append(data[1])
        self._d1_buffer[i].append(data[2])

    #####
    def _queue_data4(self, data):
        # Additional function: exports processed data in a file if export box is checked.
        # :param data: values to add for Q-factor/dissipation :type data: float.
        i = data[0]
        self._t2_store = data[1]  # time (unused)
        self._d2_store = data[2]  # data
        self._t2_buffer[i].append(data[1])
        self._d2_buffer[i].append(data[2])

    #####
    def _queue_data5(self, data):
        # Additional function: exports processed data in a file if export box is checked.
        # :param data: values to add for temperature :type data: float.
        i = data[0]
        self._t3_store = data[1]  # time (unused)
        self._d3_store = data[2]  # data1
        self._d4_store = data[3]  # data2
        self._t3_buffer[i].append(data[1])
        self._d3_buffer[i].append(data[2])
        self._d4_buffer[i].append(data[3])
        # for storing relative time
        if self._flag and ~np.isnan(self._d3_store):
            self._timestart = time()
            self._flag = False
        # Data Storage in csv and/or txt file
        '''
        self.store_data() # Now handled by SerialProcess entirely
        '''

        #####
    def _queue_data6(self, data):
        # :param data: values to add for serial error :type data: float.
        self._ser_error1 = data[0]
        self._ser_error2 = data[1]
        self._ser_error3 = data[2]
        self._ser_error4 = data[3]
        self._control_k = data[4]
        self._ser_err_usb = data[5]

    ###########################################################################
    # Gets data buffers for plot (Amplitude,Phase,Frequency and Dissipation)
    ###########################################################################
    def get_value0_buffer(self, i):
        # :return: int list.
        return self._data0_buffer[i]

    def get_value1_buffer(self, i):
        # :return: float list.
        return self._data1_buffer[i]

    #####
    def get_value2_buffer(self, i):
        # :return: float list.
        return self._data2_buffer[i]

    #####
    def get_d1_buffer(self, i):
        # :return: float list.
        return self._d1_buffer[i].get_partial()

    # Gets time buffers
    def get_t1_buffer(self, i):
        # :return: float list.
        return self._t1_buffer[i].get_partial()

    #####
    def get_d2_buffer(self, i):
        # :return: float list.
        return self._d2_buffer[i].get_partial()

    # Gets time buffers
    def get_t2_buffer(self, i):
        # :return: float list.
        return self._t2_buffer[i].get_partial()

    #####
    def get_d3_buffer(self, i):
        # :return: float list.
        return self._d3_buffer[i].get_partial()

    # Gets ambient buffer
    def get_d4_buffer(self, i):
        # :return: float list.
        return self._d4_buffer[i].get_partial()

    # Gets time buffers
    def get_t3_buffer(self, i):
        # :return: float list.
        return self._t3_buffer[i].get_partial()

    # Gets serial error
    def get_ser_error(self):
        # :return: float list.
        return self._ser_error1, self._ser_error2, self._ser_error3, self._ser_error4, self._control_k, self._ser_err_usb

    ###########################################################################
    # Checks if processes are running
    ###########################################################################

    def is_running(self):
        # :return: True if a process is running :rtype: bool.
        return self._acquisition_process is not None and self._acquisition_process.is_alive()

    ###########################################################################
    # Gets the available ports for specified source
    ###########################################################################

    @staticmethod
    def get_source_ports(source):
        """
        :param source: Source to get available ports :type source: OperationType.
        :return: List of available ports :rtype: str list.
        """
        if Constants.serial_simulate_device:
            return ['/dev/simulator']
        elif source == OperationType.measurement:
            ports = SerialProcess.get_ports()
            port_names = []
            for i in range(len(ports)):
                try:
                    port_names.append(ports[i][0:ports[i].index(':')])
                except:
                    # ignore ValueError
                    port_names.append(ports[i])
            Log.i('Port connected:', port_names)
            return ports
        elif source == OperationType.calibration:
            ports = CalibrationProcess.get_ports()
            port_names = []
            for i in range(len(ports)):
                try:
                    port_names.append(ports[i][0:ports[i].index(':')])
                except:
                    # ignore ValueError
                    port_names.append(ports[i])
            Log.i('Port connected:', port_names)
            return ports
        else:
            Log.w(TAG, "Warning: Unknown source selected")
            return None

    ###########################################################################
    # Gets the available speeds for specified source
    ###########################################################################

    @staticmethod
    def get_source_speeds(source, i=0):
        """
        :param source: Source to get available speeds :type source: OperationType.
        :return: List of available speeds :rtype: str list.
        """
        if source == OperationType.measurement:
            return SerialProcess.get_speeds(i)
        elif source == OperationType.calibration:
            return CalibrationProcess.get_speeds()
        else:
            Log.w(TAG, "Unknown source selected")
            return None

    ###########################################################################
    # Setup/Clear the internal buffers
    ###########################################################################

    def reset_buffers(self, samples):
        # :param samples: Number of samples for the buffers :type samples: int.

        # Initialises data buffers
        self._data0_buffer = []  # frequency
        self._data1_buffer = []  # amplitude
        self._data2_buffer = []  # phase
        self._d1_buffer = []  # Resonance frequency
        self._d2_buffer = []  # Dissipation
        self._d3_buffer = []  # temperature
        self._d4_buffer = []  # ambient
        self._t1_buffer = []  # time (Resonance frequency)
        self._t2_buffer = []  # time (Dissipation)
        self._t3_buffer = []  # time (temperature)

        for i in range(4):
            self._data0_buffer.append(np.zeros(samples))  # frequency
            self._data1_buffer.append(np.zeros(samples))  # amplitude
            self._data2_buffer.append(np.zeros(samples))  # phase
            # Resonance frequency
            self._d1_buffer.append(RingBuffer(Constants.ring_buffer_samples))
            self._d2_buffer.append(RingBuffer(
                Constants.ring_buffer_samples))  # Dissipation
            self._d3_buffer.append(RingBuffer(
                Constants.ring_buffer_samples))  # temperature
            self._d4_buffer.append(RingBuffer(
                Constants.ring_buffer_samples))  # ambient
            # time (Resonance frequency)
            self._t1_buffer.append(RingBuffer(Constants.ring_buffer_samples))
            self._t2_buffer.append(RingBuffer(
                Constants.ring_buffer_samples))  # time (Dissipation)
            self._t3_buffer.append(RingBuffer(
                Constants.ring_buffer_samples))  # time (temperature)

        # Initialises supporting variables
        self._d1_store = 0
        self._d2_store = 0
        self._d3_store = 0
        self._d4_store = 0
        self._t1_store = 0
        self._t2_store = 0
        self._t3_store = 0
        self._ser_error1 = 0
        self._ser_error2 = 0
        self._ser_error3 = 0
        self._ser_error4 = 0
        self._ser_err_usb = 0
        # self._control_k = 0

        # Log.i(TAG, "Buffers cleared")

    ############################################################################
    # Gets frequency range
    ############################################################################

    def get_frequency_range(self):
        """
        :param samples: Number of samples for the buffers :type samples: int.
        :return: overtone :type overtone: float.
        :return: frequency range :type readFREQ: float list.
        """
        return self._readFREQ  # stale after peak tracking kicks in (use _data0_buffer instead)

    ############################################################################
    # Gets overtones name, value and frequency step
    ############################################################################

    def get_overtone(self):
        """
        :param samples: Number of samples for the buffers :type samples: int.
        :return: overtone :type overtone: float.
        :return: frequency range :type readFREQ: float list.
        """
        return self._overtone_name, self._overtone_value, self._fStep
