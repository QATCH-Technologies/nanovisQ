import multiprocessing
from QATCH.core.constants import Constants
from QATCH.common.fileStorage import FileStorage
from QATCH.common.findDevices import Discovery
from QATCH.common.logger import Logger as Log
from QATCH.common.switcher import Overtone_Switcher_5MHz, Overtone_Switcher_10MHz
from QATCH.processors.Elaborate import ElaborateProcess
from multiprocessing import Queue
import threading
from time import time, sleep
from serial.tools import list_ports
import numpy as np
from numpy import loadtxt
# from progressbar import Bar, Percentage, ProgressBar, RotatingMarker,Timer
import logging
from logging.handlers import QueueHandler
import sys
import os

if Constants.serial_simulate_device:
    from QATCH.processors.Simulator import serial  # simulator
else:
    from QATCH.processors.Device import serial  # real device hardware


TAG = ""  # "[Serial]"

###############################################################################
# Process for the serial package and the communication with the serial port
# Processes incoming data and calculates outgoing data by the algorithms
###############################################################################


class SerialProcess(multiprocessing.Process):

    ###########################################################################
    # BASELINE CORRECTION
    ###########################################################################
    def baseline_correction(self, x, y, poly_order):

        # Estimate Baseline with Least Squares Polynomial Fit (LSP)
        coeffs = np.polyfit(x, y, poly_order)
        # Evaluate a polynomial at specific values
        poly_fitted = np.polyval(coeffs, x)
        return poly_fitted, coeffs

    ###########################################################################
    # BASELINE - Evaluates polynomial coefficients - Sweep over all frequencies
    ###########################################################################
    def baseline_coeffs(self, i):

        # initializations
        self.polyfitted_all = None
        self.coeffs_all = None
        self.polyfitted_all_phase = None
        self.coeffs_all_phase = None

        # loads Calibration (baseline correction) from file
        (self.freq_all, self.mag_all, self.phase_all) = self.load_calibration_file(i)

        # Baseline correction: input signal Amplitude (sweep all frequencies)
        (self.polyfitted_all, self.coeffs_all) = self.baseline_correction(
            self.freq_all, self.mag_all, 8)
        self.mag_beseline_corrected_all = self.mag_all-self.polyfitted_all

        # Baseline correction: input signal Phase (sweep all frequencies)
        if not self.phase_all is None:
            (self.polyfitted_all_phase, self.coeffs_all_phase) = self.baseline_correction(
                self.freq_all, self.phase_all, 8)
            self.phase_beseline_corrected_all = self.phase_all-self.polyfitted_all_phase
        else:
            self.polyfitted_all_phase = self.coeffs_all_phase = self.phase_beseline_corrected_all = None

        return self.coeffs_all

    def _convertADCtoMagnitude(self, adc):
        return (adc * Constants.ADCtoVolt - Constants.VCP) / 0.029

    def _convertADCtoPhase(self, adc):
        return (adc * Constants.ADCtoVolt - Constants.VCP) / 0.01

    def _convertMagnitudeToADC(self, val):
        return (val * 0.029 + Constants.VCP) / Constants.ADCtoVolt

    def _convertPhaseToADC(self, val):
        return (val * 0.01 + Constants.VCP) / Constants.ADCtoVolt

    ###########################################################################
    # Initializing values for process
    ###########################################################################

    def __init__(self, queue_log, parser_process, time_start, freq_hopping, export_enabled=False, reconstruct=False, driedValue=None, appliedValue=None):
        """
        :param parser_process: Reference to a ParserProcess instance.
        :type parser_process: ParserProcess.
        """
        self._queueLog = queue_log  # Log.create()

        multiprocessing.Process.__init__(self)
        self._exit = multiprocessing.Event()
        self._done = multiprocessing.Event()

        self._time_start = time()

        # Instantiate a single ParserProcess class used by each communication channel
        self._parser = parser_process

        self._serial = []
        self._serial.append(serial.Serial())

        # check for freq _freq_hopping
        self._freq_hopping = freq_hopping
        self._base_overtone_freq = Constants.base_overtone_freq

        self._export = export_enabled
        self._reconstruct = reconstruct
        self._driedValue = driedValue
        self._appliedValue = appliedValue

        self.sensorDriedTime = 0.0
        self.dropAppliedTime = 0.0

    ###########################################################################
    # Opens a specified serial port
    ###########################################################################

    def open(self, port, pid,
             speed=Constants.serial_default_overtone,
             timeout=Constants.serial_timeout_ms,
             writeTimeout=Constants.serial_writetimeout_ms):
        """
        :param port: Serial port name :type port: str.
        :param speed: Overtone selected for the analysis :type speed: str.
        :param timeout: Sets current read timeout :type timeout: float (seconds).
        :param writetTimeout: Sets current write timeout :type writeTimeout: float (seconds).
        :return: True if the port is available :rtype: bool.
        """

        self._serial.clear()
        for i in range(len(port)):
            self._serial.append(serial.Serial())
            self._serial[i].port = port[i] if isinstance(port, list) else port
            self._serial[i].baudrate = Constants.serial_default_speed  # 115200
            self._serial[i].stopbits = serial.STOPBITS_ONE
            self._serial[i].bytesize = serial.EIGHTBITS
            self._serial[i].timeout = timeout
            self._serial[i].write_timeout = writeTimeout
            if not isinstance(port, list):
                break

        if isinstance(port, list):

            self._baselines = np.zeros(len(port), dtype=int)
            self._startFreqs = np.zeros(len(port), dtype=int)
            self._stopFreqs = np.zeros(len(port), dtype=int)

            if self._freq_hopping:
                self._baselines_up = np.zeros(len(port), dtype=int)
                self._startFreqs_up = np.zeros(len(port), dtype=int)
                self._stopFreqs_up = np.zeros(len(port), dtype=int)
                self._baselines_down = np.zeros(len(port), dtype=int)
                self._startFreqs_down = np.zeros(len(port), dtype=int)
                self._stopFreqs_down = np.zeros(len(port), dtype=int)

        all_peaks_mag = []
        try:
            for i in range(len(self._serial)):
                j = i
                if len(self._serial) > 1:
                    j += 1
                # Loads frequencies from file
                peaks_mag, _, _, _, _ = self.load_frequencies_file(j)
                all_peaks_mag.append(peaks_mag)
        except:
            Log.w('No peak magnitudes found. Rerun Initialize.')
            return -1

        # handles the exceptions
        try:
            self._overtone = float(speed)
        except:
            Log.w(TAG, "Warning: wrong frequency selection, set default to {} Hz Fundamental".format(
                peaks_mag[0]))
            self._overtone = peaks_mag[0]

        self._overtone_int = None
        for x in range(len(self._serial)):
            for y in range(len(all_peaks_mag[x])):
                if self._overtone == all_peaks_mag[x][y]:
                    self._overtone_int = y
            if self._overtone_int != None:
                break
        # Checks for correct frequency selection
        if self._overtone_int == None:
            Log.w(TAG, "Warning: wrong frequency selection, set default to {} Hz Fundamental".format(
                peaks_mag[0]))
            self._overtone_int = 0

        is_open = False  # default, fail
        num_ports = len(self._serial)
        if num_ports > 0:
            is_open = True
            for j in range(num_ports):
                try:
                    if self._is_port_available(self._serial[j].port):
                        self._serial[j].open()
                        # port is available, confirm it is open
                        is_open &= self._serial[j].is_open
                        self._serial[j].close()
                    else:
                        is_open = False  # port not available
                except:
                    is_open = False  # port failed to open, may be in-use
        else:
            is_open = False  # no ports available

        self._pid = pid

        return is_open

    ###########################################################################
    # Reads the serial port,processes and adds all the data to internal queues
    ###########################################################################

    def run(self):
        """
        The expected format is a buffer (sweep) and a new buffer as a new sweep.
        The method parses data, converts each value to float and adds to a queue.
        If incoming data can't be converted to float,the data will be discarded.
        """
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

            # initializations
            self._flag_error = 0
            self._flag_error_usb = 0
            self._err1 = 0
            self._err2 = 0

            # track serial messaging
            self._seq = -1
            self._ts = -1

            # CALLS baseline_coeffs method
            self._coeffs_all = []
            for i in range(len(self._serial)):
                j = i
                if len(self._serial) > 1:
                    j += 1
                coeffs_all = self.baseline_coeffs(j)
                self._coeffs_all.append(coeffs_all)

            # Checks if the serial port is currently connected
            is_available = True
            for i in range(len(self._serial)):
                is_available &= self._is_port_available(self._serial[i].port)
            if is_available:

                samples = Constants.argument_default_samples
                for i in range(len(self._serial)):
                    j = i
                    if len(self._serial) > 1:
                        j += 1
                    try:
                        # Calls get_frequencies method:
                        # ACQUIRES overtone, sets start and stop frequencies, the step and range frequency according to the number of samples
                        if self._freq_hopping:
                            (self._overtone_name, overtone_value, fStep, readFREQ, SG_window_size, Spline_points, Spline_factor, baseline, start, stop,
                             self._overtone_name_up, overtone_value_up, fStep_up, readFREQ_up, SG_window_size_up, Spline_points_up, Spline_factor_up, baseline_up, start_up, stop_up,
                             self._overtone_name_down, overtone_value_down, fStep_down, readFREQ_down, SG_window_size_down, Spline_points_down, Spline_factor_down, baseline_down, start_down, stop_down) = self.get_frequencies(samples, j)

                            if len(self._serial) > 1:
                                # Store these for later (when hopping)
                                self._baselines_up[i] = baseline_up
                                self._startFreqs_up[i] = start_up
                                self._stopFreqs_up[i] = stop_up
                                self._baselines_down[i] = baseline_down
                                self._startFreqs_down[i] = start_down
                                self._stopFreqs_down[i] = stop_down
                        else:
                            (self._overtone_name, overtone_value, fStep, readFREQ, SG_window_size, Spline_points,
                             Spline_factor, baseline, start, stop) = self.get_frequencies(samples, j)

                        if len(self._serial) > 1:
                            # Store these for later (all the time)
                            self._baselines[i] = baseline
                            self._startFreqs[i] = start
                            self._stopFreqs[i] = stop
                    except IOError:
                        Log.w('No peak magnitudes available. Rerun Initialize.')

                # Gets the state of the serial ports
                all_ports_open = True
                for i in range(len(self._serial)):
                    if not self._serial[i].is_open:
                        # OPENS the serial port
                        try:
                            self._serial[i].open()
                        except OSError as err:
                            Log.e(err)
                            self.stop()
                            self._done.set()
                            return
                    all_ports_open &= self._serial[i].is_open

                if all_ports_open:
                    if len(self._serial) == 1:
                        # START elaborate process (and associated queues)
                        self._elaborate_in_q = Queue()  # used to pass data to elaborate process
                        self._elaborate_out_q = Queue()  # used to get data from elaborate process
                        self._elaborate_process = ElaborateProcess(self._queueLog,
                                                                   self._parser,
                                                                   self._elaborate_in_q,
                                                                   self._elaborate_out_q,
                                                                   self._export,
                                                                   self._overtone_name,
                                                                   self._reconstruct,
                                                                   self._driedValue,
                                                                   self._appliedValue)
                        self._elaborate_process.start()

                    # Initializes the sweep counter
                    k = 0
                    if not self._exit.is_set():
                        Log.i(TAG, 'Capturing raw data...')
                        Log.i(TAG, 'Wait, processing early data...')

                    # creates a timestamp
                    timestamp = time()

                    # persistent data reset for each sweep
                    data_mag = np.linspace(0, 0, samples)
                    data_ph = np.linspace(0, 0, samples)

                    self._environment = Constants.environment

                    # Initializes the progress bar
                    # bar = ProgressBar(widgets=[TAG,' ', Bar(marker='>'),' ',Percentage(),' ', Timer()], maxval=self._environment).start() #
                    #### SWEEPS LOOP ####

                    pending_readFREQ = readFREQ
                    # added for freq hopping
                    if self._freq_hopping:
                        pending_readFREQ_up = readFREQ_up
                        pending_readFREQ_down = readFREQ_down
                        readFREQs = [pending_readFREQ,
                                     pending_readFREQ_up, pending_readFREQ_down]
                        startFreqs = [self._startFreq,
                                      self._startFreq_up, self._startFreq_down]
                        stopFreqs = [self._stopFreq,
                                     self._stopFreq_up, self._stopFreq_down]
                        fSteps = [fStep, fStep_up, fStep_down]
                        baselines = [baseline, baseline_up, baseline_down]
                        c = 0
                        base_overtone_counter = 0
                        base_overtones_per_cycle = self._base_overtone_freq

                    time_last_speed_adjust = timestamp
                    format = -1
                    streaming = False
                    speed = 0
                    overtone = 0
                    last_overtone = 0

                    while not self._exit.is_set():

                        try:
                            # CREATE command string
                            # added for freq hopping
                            if self._freq_hopping:
                                base_overtone_counter += 1
                                if base_overtone_counter > base_overtones_per_cycle:
                                    c += 1
                                    if c % 3 == 0:
                                        c = 0
                                        base_overtone_counter = 0

                                # safety/sanity check
                                if overtone > len(readFREQs):
                                    overtone = 0

                                # First time thru "c" will always be zero...
                                # but if NOT streaming, we must send "c" freqs here as first 3 elements
                                if streaming:
                                    c = overtone = 0
                                # readFREQ = readFREQs[c] if not streaming else readFREQs[overtone]
                                cmd = (str(int(startFreqs[c])) + ';' + str(int(stopFreqs[c])) + ';' + str(int(baselines[c])) + ';' +
                                       str(int(startFreqs[1])) + ';' + str(int(stopFreqs[1])) + ';' + str(int(baselines[1])) + ';' +
                                       str(int(startFreqs[2])) + ';' + str(int(stopFreqs[2])) + ';' + str(int(baselines[2])) + ';' +
                                       str(int(base_overtones_per_cycle)))
                            else:
                                cmd = (str(int(self._startFreq)) + ';' +
                                       str(int(self._stopFreq)) + ';' + str(int(baseline)))

                            # Add SPEED CMD to send to device
                            max_speed = Constants.max_speed_single if len(
                                self._serial) == 1 else Constants.max_speed_multi4
                            cmd = ('SPEED {:.0f}'.format(
                                max_speed) + '\n' + cmd)

                            # Add AVG CMD to send to device
                            cmd = ('AVG ' + str(int(Constants.avg_in)) + ';' + str(int(Constants.avg_out)) + ';' + str(int(Constants.step_size))
                                   + ';' + str(int(Constants.max_drift_l_hz)) +
                                   ';' + str(int(Constants.max_drift_r_hz))
                                   + ';' + "{:0.3f}".format(10**(Constants.track_width_db/20)) + '\n' + cmd)

                            # TODO: All the frequency sweeps set on the devices are for the primary, must reset CMD frequencies for each PID

                            if len(self._serial) > 1:
                                ### MULTIPLEX CODE START ###

                                self._maxDriftLeft = []
                                self._maxDriftRight = []
                                for i in range(len(self._serial)):
                                    self._maxDriftLeft.append(
                                        Constants.max_drift_l_hz)
                                    self._maxDriftRight.append(
                                        Constants.max_drift_r_hz)

                                try:
                                    # look for and avoid any sweep overlap between ports
                                    # strategy: move max drift to midpoint of center frequencies found during calibration
                                    freqRanges = []
                                    for x in range(len(self._serial)):
                                        freqRanges.append(range(int(self._startFreqs[x] - Constants.max_drift_l_hz),
                                                                int(self._stopFreqs[x] + Constants.max_drift_r_hz)))
                                    for x in range(len(self._serial)):
                                        for y in range(len(self._serial)):
                                            if x == y:
                                                continue
                                            start = int(
                                                self._startFreqs[y] - Constants.max_drift_l_hz)
                                            stop = int(
                                                self._stopFreqs[y] + Constants.max_drift_r_hz)
                                            if start in freqRanges[x]:
                                                midpt = int(np.average(
                                                    [self._startFreqs[y], self._stopFreqs[x]]))
                                                freqRanges[x] = range(
                                                    freqRanges[x][0], midpt - 1)
                                                freqRanges[y] = range(
                                                    midpt + 1, freqRanges[y][-1])
                                            if stop in freqRanges[x]:
                                                midpt = int(np.average(
                                                    [self._startFreqs[x], self._stopFreqs[y]]))
                                                freqRanges[x] = range(
                                                    midpt + 1, freqRanges[x][-1])
                                                freqRanges[y] = range(
                                                    freqRanges[y][0], midpt - 1)

                                    self._maxDriftLeft = []
                                    self._maxDriftRight = []
                                    peaks_too_close = False
                                    for i in range(len(self._serial)):
                                        self._maxDriftLeft.append(
                                            abs(self._startFreqs[i] - freqRanges[i][0]))
                                        self._maxDriftRight.append(
                                            abs(freqRanges[i][-1] - self._stopFreqs[i]))
                                        minLimit_L = int(
                                            Constants.max_drift_l_hz / 2)
                                        if self._maxDriftLeft[i] < minLimit_L:
                                            Log.w(f"Port {i}: (Left Drift) " +
                                                  f"Actual = {self._maxDriftLeft[i]}, Min-Limit = {minLimit_L}")
                                            self._maxDriftLeft[i] = minLimit_L
                                            peaks_too_close = True
                                        minLimit_R = int(
                                            Constants.max_drift_r_hz / 2)
                                        if self._maxDriftRight[i] < minLimit_R:
                                            Log.w(f"Port {i}: (Right Drift) " +
                                                  f"Actual = {self._maxDriftRight[i]}, Min-Limit = {minLimit_R}")
                                            self._maxDriftRight[i] = minLimit_R
                                            peaks_too_close = True

                                    if peaks_too_close:
                                        Log.w(
                                            "Peaks may be too close to reliably track.")
                                        Log.w(
                                            "If you experience tracking issues, please re-calibrate or try another crystal.")

                                except Exception as e:
                                    Log.e(
                                        "Error checking peak proximity and tracking reliability prior to start.")
                                    Log.w(
                                        "If you experience tracking issues, please re-calibrate or try another crystal.")

                                cmd_split = cmd.splitlines()
                                avg_parts = cmd_split[0].split(";")
                                avg_parts[3] = "{}"
                                avg_parts[4] = "{}"
                                cmd_split[0] = ";".join(avg_parts)
                                cmd_format_len = 10 if self._freq_hopping else 3
                                cmd_split[-1] = ";".join(
                                    ["{}" for i in range(cmd_format_len)])
                                cmd = "\n".join(cmd_split)

                                SerBuffer = Queue()
                                LogBuffer = Queue()
                                ExpBuffer = Queue()

                                for i in range(len(self._serial)):
                                    self._serial[i].open()
                                    Log.d("Port {} opened!".format(i+1))
                                    if i == 0:  # primary device must have MSGBOX cleared
                                        this_cmd = b"MSGBOX\n"
                                        Log.d("Port {} write: {}".format(
                                            i+1, this_cmd))
                                        self._serial[i].write(this_cmd)
                                        bs = self._serial[i].read_until(
                                        ).decode()  # byte stream
                                        Log.d(
                                            "Port {} read: {}!".format(i+1, bs))
                                    if self._freq_hopping:
                                        this_cmd = cmd.format(
                                            self._maxDriftLeft[i], self._maxDriftRight[i],
                                            self._startFreqs[i], self._stopFreqs[i], self._baselines[i],
                                            self._startFreqs_up[i], self._stopFreqs_up[i], self._baselines_up[i],
                                            self._startFreqs_down[i], self._stopFreqs_down[i], self._baselines_down[i],
                                            base_overtones_per_cycle).encode()
                                    else:
                                        this_cmd = cmd.format(self._maxDriftLeft[i], self._maxDriftRight[i],
                                                              self._startFreqs[i], self._stopFreqs[i], self._baselines[i]).encode()
                                    Log.d("Port {} write: {}".format(
                                        i+1, this_cmd))
                                    self._serial[i].write(this_cmd)
                                    bs = self._serial[i].read_until(
                                    ).decode()  # byte stream
                                    # SerBuffer.put([i, bs])
                                    Log.d("Port {} read: {}".format(i+1, bs))
                                    # sleep(0.25) # TODO: Testing only, remove ideally

                                def get_mode(freq):
                                    mode = 0
                                    if freq < 10000000:
                                        mode = 1  # 5k
                                    elif freq < 20000000:
                                        mode = 3  # 15k
                                    elif freq < 30000000:
                                        mode = 5  # 25k
                                    elif freq < 40000000:
                                        mode = 7  # 35k
                                    elif freq < 50000000:
                                        mode = 9  # 45k
                                    return mode

                                def serial_read(s, i):
                                    # TODO: Testing only, remove ideally
                                    sleep(0.25*i)
                                    send_stream_cmd = True
                                    while not self._exit.is_set():
                                        start = time()
                                        waitFor = 3
                                        now = time()
                                        while s[i].in_waiting == 0 and now - start < waitFor and not send_stream_cmd:
                                            now = time()
                                            pass
                                        if now - start >= waitFor:  # timeout occurred
                                            send_stream_cmd = True
                                        if send_stream_cmd:
                                            send_stream_cmd = False
                                            cmd = "STREAM\n" + \
                                                'SPEED {:.0f}'.format(
                                                    Constants.max_speed_multi4) + '\n'
                                            s[i].write(cmd.encode())
                                            Log.d(
                                                "Port {} streaming".format(i+1))
                                        if s[i].in_waiting != 0:
                                            SerBuffer.put(
                                                [i, s[i].read_until().decode()])
                                    # Thread ending...
                                    Log.d("Port {} stopping...".format(i+1))
                                    if s[i].is_open:
                                        # TODO: Testing only, remove ideally
                                        sleep(0.25*(4-i))
                                        s[i].write("STOP\n".encode())
                                        # s[i].flush() # does not exist
                                        s[i].close()
                                    Log.d("Port {} stopped".format(i+1))

                                def data_logger():
                                    while not self._exit.is_set() or not LogBuffer.empty():
                                        if not LogBuffer.empty():
                                            out = LogBuffer.get_nowait()
                                            i = out[0]
                                            k = out[1]
                                            overtone = out[2]
                                            w_time = out[3]
                                            temperature = out[4]
                                            peak_mag = out[5]
                                            peak_freq = out[6]
                                            dissipation = out[7]
                                            t_amb = out[8]

                                            # Read the shared values
                                            with self._driedValue.get_lock():
                                                if self.sensorDriedTime != self._driedValue.value:
                                                    self.sensorDriedTime = self._driedValue.value
                                                    Log.d("[SerialProcess]",
                                                          f"Sensor dried time = {self.sensorDriedTime}")
                                            with self._appliedValue.get_lock():
                                                if self.dropAppliedTime != self._appliedValue.value:
                                                    self.dropAppliedTime = self._appliedValue.value
                                                    Log.d("[SerialProcess]",
                                                          f"Drop applied time = {self.dropAppliedTime}")

                                            if overtone in (0, 255):
                                                filenameCSV = "{}_{}".format(
                                                    Constants.csv_filename, self._overtone_name.split(' ')[0])
                                                write_interval = 1000 if w_time < Constants.downsample_after else Constants.downsample_file_count
                                                FileStorage.CSVsave(i+1, filenameCSV, Constants.csv_export_path, w_time, temperature,
                                                                    peak_mag, peak_freq, dissipation, t_amb, (k % write_interval == 0))
                                            elif overtone == 1:
                                                write_interval = 1000 if w_time < Constants.downsample_after else Constants.downsample_file_count * \
                                                    Constants.base_overtone_freq
                                                FileStorage.CSVsave(i+1, "overtone_upper", Constants.csv_export_path, w_time, temperature,
                                                                    peak_mag, peak_freq, dissipation, t_amb, (k % write_interval < Constants.base_overtone_freq))
                                            elif overtone == 2:
                                                write_interval = 1000 if w_time < Constants.downsample_after else Constants.downsample_file_count * \
                                                    Constants.base_overtone_freq
                                                FileStorage.CSVsave(i+1, "overtone_lower", Constants.csv_export_path, w_time, temperature,
                                                                    peak_mag, peak_freq, dissipation, t_amb, (k % write_interval < Constants.base_overtone_freq))
                                    # Thread ending...
                                    FileStorage.CSVflush_all()
                                    Log.d("stopped thread 'data_logger'")

                                def export_swps():
                                    while not self._exit.is_set() or not ExpBuffer.empty():
                                        if not ExpBuffer.empty():
                                            out = ExpBuffer.get_nowait()
                                            i = out[0]
                                            k = out[1]
                                            overtone = out[2]
                                            readFREQ = out[3]
                                            peaks_mag = out[4]
                                            peak_freq = out[5]
                                            left = out[6]
                                            right = out[7]
                                            phase = None
                                            baseline_offset = min(self._convertMagnitudeToADC(
                                                np.polyval(self._coeffs_all[i], peak_freq)), peak_mag)
                                            mag_result_fit = ElaborateProcess.build_curve(
                                                readFREQ, peak_mag - baseline_offset, peak_freq, left, right)
                                            self._readFREQ[np.argmax(
                                                mag_result_fit)] = peak_freq
                                            mag_result_fit[np.argmax(
                                                mag_result_fit)] = peak_mag - baseline_offset
                                            mag_result_fit = self._convertADCtoMagnitude(
                                                mag_result_fit)
                                            # zero offset
                                            mag_result_fit -= self._convertADCtoMagnitude(
                                                0)
                                            filtered_mag = mag_result_fit
                                            # Storing acquired sweeps
                                            filename = "{}_{}_{}".format(
                                                Constants.csv_sweeps_filename, self._overtone_name, k)
                                            path = "{}_{}".format(
                                                Constants.csv_sweeps_export_path, self._overtone_name)
                                            path = FileStorage.DEV_populate_path(
                                                path, i+1)
                                            if not phase is None:
                                                FileStorage.TXT_sweeps_save(
                                                    i+1, filename, path, readFREQ, filtered_mag, phase, appendNameToPath=False)
                                            else:
                                                FileStorage.TXT_sweeps_save(
                                                    i+1, filename, path, readFREQ, filtered_mag, appendNameToPath=False)
                                    # Thread ending...
                                    Log.d("stopped thread 'export_swps'")

                                Log.d("starting serial threads...")
                                for i in range(len(self._serial)):
                                    threading.Thread(target=serial_read, args=(
                                        self._serial, i),).start()
                                # thread0 = threading.Thread(target=serial_read, args=(self._serial,0),).start()
                                # thread1 = threading.Thread(target=serial_read, args=(self._serial,1),).start()
                                # thread2 = threading.Thread(target=serial_read, args=(self._serial,2),).start()
                                # thread3 = threading.Thread(target=serial_read, args=(self._serial,3),).start()
                                thread4 = threading.Thread(
                                    target=data_logger,).start()
                                if self._export:
                                    thread5 = threading.Thread(
                                        target=export_swps,).start()
                                Log.d("threads started!")

                                self._minFREQ = Constants.calibration_frequency_stop
                                self._maxFREQ = Constants.calibration_frequency_start
                                if self._freq_hopping:
                                    self._minFREQ_up = Constants.calibration_frequency_stop
                                    self._maxFREQ_up = Constants.calibration_frequency_start
                                    self._minFREQ_down = Constants.calibration_frequency_stop
                                    self._maxFREQ_down = Constants.calibration_frequency_start

                                self._seq = np.zeros(
                                    len(self._serial), dtype=int)
                                self._ts = np.zeros(
                                    len(self._serial), dtype=int)

                                while not self._exit.is_set():  # break immediately upon stop request (no need to flush buffered serial data)
                                    if not SerBuffer.empty():
                                        out = SerBuffer.get_nowait()
                                        device = out[0]
                                        buffer = out[1]

                                        # if self._reconstruct:
                                        #     Log.d(f"[STREAM] {device} {buffer.strip()}")

                                        data_raw = buffer.split(';')
                                        length = len(data_raw)

                                        if length == 6 or length == 8:

                                            if length == 6:
                                                format = str(data_raw[0])
                                                sequence = int(data_raw[1])
                                                w_time = int(data_raw[2])
                                                peak_mag = int(data_raw[3])
                                                peak_freq = int(data_raw[4])
                                                tec_state = tec_temp = 0
                                                data_temp = float(data_raw[5])

                                            if length == 8:
                                                format = str(data_raw[0])
                                                sequence = int(data_raw[1])
                                                w_time = int(data_raw[2])
                                                peak_mag = int(data_raw[3])
                                                peak_freq = int(data_raw[4])
                                                tec_state = int(data_raw[5])
                                                tec_temp = float(data_raw[6])
                                                data_temp = float(data_raw[7])

                                            if self._freq_hopping:
                                                if self._startFreqs[device] - Constants.max_drift_l_hz < peak_freq < self._stopFreqs[device] + Constants.max_drift_r_hz:
                                                    overtone = 0
                                                elif self._startFreqs_up[device] - Constants.max_drift_l_hz < peak_freq < self._stopFreqs_up[device] + Constants.max_drift_r_hz:
                                                    overtone = 1
                                                elif self._startFreqs_down[device] - Constants.max_drift_l_hz < peak_freq < self._stopFreqs_down[device] + Constants.max_drift_r_hz:
                                                    overtone = 2
                                                else:
                                                    overtone = 0xFF
                                            else:
                                                overtone = 0xFF
                                        else:
                                            print(
                                                "Message response length incorrect!")
                                            continue

                                        # Track and detect USB serial stream anomalies
                                        # i.e. dropped, corrupt or out-of-order packets
                                        if self._freq_hopping and overtone == 0xFF:
                                            Log.d(TAG, "WARN: Received USB packet dev{} #{} @{} with out-of-range frequency ({})".format(
                                                int(device), sequence, w_time, peak_freq))
                                        if sequence != self._seq[device] + 1 and self._seq[device] != 0:
                                            Log.d(TAG, "WARN: Missing {:.0f} USB packets dev{} @{} (SEQ was {}, now {})".format(
                                                sequence - self._seq[device], int(device), w_time, self._seq[device], sequence))
                                        if sequence <= self._seq[device]:
                                            Log.d(TAG, "WARN: Malformed USB packet dev{} @{} (SEQ was {}, now {})".format(
                                                int(device), w_time, self._seq[device], sequence))
                                        if w_time <= self._ts[device] and overtone in [0, 0xFF] and sequence != 1:
                                            Log.d(TAG, "WARN: Malformed USB packet dev{} #{} (TS was {}, now {})".format(
                                                int(device), sequence, self._ts[device], w_time))
                                        self._seq[device] = sequence
                                        if overtone in [0, 0xFF]:
                                            self._ts[device] = w_time

                                        peak_dB = self._convertADCtoMagnitude(
                                            peak_mag)
                                        dissipation = 10.0 ** (-peak_dB / 20)

                                        self._mode = get_mode(peak_freq)
                                        if self._mode == 1:
                                            dissipation *= Constants.dissipation_factor_1st_mode
                                        if self._mode == 3:
                                            dissipation *= Constants.dissipation_factor_3rd_mode
                                        if self._mode == 5:
                                            dissipation *= Constants.dissipation_factor_5th_mode

                                        left = right = dissipation * peak_freq / 2
                                        start = peak_freq - (3 * left)
                                        stop = peak_freq + (3 * right)
                                        if overtone in (0, 255):
                                            if self._minFREQ > start:
                                                self._minFREQ = start
                                            if self._maxFREQ < stop:
                                                self._maxFREQ = stop
                                            _min = self._minFREQ
                                            _max = self._maxFREQ
                                        elif overtone == 1:
                                            if self._minFREQ_up > start:
                                                self._minFREQ_up = start
                                            if self._maxFREQ_up < stop:
                                                self._maxFREQ_up = stop
                                            _min = self._minFREQ_up
                                            _max = self._maxFREQ_up
                                        elif overtone == 2:
                                            if self._minFREQ_down > start:
                                                self._minFREQ_down = start
                                            if self._maxFREQ_down < stop:
                                                self._maxFREQ_down = stop
                                            _min = self._minFREQ_down
                                            _max = self._maxFREQ_down
                                        self._readFREQ = np.linspace(
                                            _min, _max, Constants.argument_default_samples)
                                        baseline_offset = min(self._convertMagnitudeToADC(
                                            np.polyval(self._coeffs_all[device], peak_freq)), peak_mag)
                                        mag_result_fit = ElaborateProcess.build_curve(
                                            self._readFREQ, peak_mag - baseline_offset, peak_freq, left, right)
                                        self._readFREQ[np.argmax(
                                            mag_result_fit)] = peak_freq
                                        mag_result_fit[np.argmax(
                                            mag_result_fit)] = peak_mag - baseline_offset
                                        mag_result_fit = self._convertADCtoMagnitude(
                                            mag_result_fit)
                                        # zero offset
                                        mag_result_fit -= self._convertADCtoMagnitude(
                                            0)
                                        filtered_mag = mag_result_fit

                                        # Make sure peaks on Amplitudes plot for MULTI are not clipped at edges
                                        if filtered_mag[0] != 0:
                                            self._minFREQ -= 1
                                        if filtered_mag[-1] != 0:
                                            self._maxFREQ += 1

                                        w_time /= 1e4

                                        LogBuffer.put(
                                            [device, sequence, overtone, w_time, data_temp, peak_mag, peak_freq, dissipation, tec_temp])
                                        if self._export:
                                            ExpBuffer.put(
                                                [device, sequence, overtone, self._readFREQ, peak_mag, peak_freq, left, right])

                                        k = sequence
                                        settle_samples = Constants.initial_settle_samples
                                        if overtone in (0, 255) and k > settle_samples and k % (4*1) == device:
                                            self._parser.add0(
                                                [device, self._readFREQ])
                                            self._parser.add1(
                                                [device, filtered_mag])
                                            self._parser.add2(
                                                [device, tec_state])
                                            self._parser.add3(
                                                [device, w_time, peak_freq])
                                            self._parser.add4(
                                                [device, w_time, dissipation])
                                            self._parser.add5(
                                                [device, w_time, data_temp, tec_temp])
                                            self._parser.add6(
                                                [False, False, False, False, np.sum(self._seq), False])
                                        if k <= settle_samples:
                                            self._minFREQ = Constants.calibration_frequency_stop
                                            self._maxFREQ = Constants.calibration_frequency_start
                                            self._minFREQ_up = Constants.calibration_frequency_stop
                                            self._maxFREQ_up = Constants.calibration_frequency_start
                                            self._minFREQ_down = Constants.calibration_frequency_stop
                                            self._maxFREQ_down = Constants.calibration_frequency_start

                                # Stopping...
                                Log.d("Waiting for logger threads to finish... ")
                                while not LogBuffer.empty() or not ExpBuffer.empty():
                                    pass

                                # gracefully end subprocess
                                # half second per device, total of 2 seconds for shutdown
                                sleep(3)
                                # Log.d("Waiting for serial ports to close... ")
                                # while True:
                                #     at_least_1_open = False
                                #     for i in range(len(self._serial)):
                                #         at_least_1_open |= self._serial[i].is_open
                                #     if not at_least_1_open:
                                #         break # all have closed

                                Log.d(
                                    "Logger threads and serial ports have closed. Task Finished!")
                                self._done.set()

                                return  # unwind process

                                ### MULTIPLEX CODE END ###

                            ### NOTE: If running in multiplex mode, the rest of run() will not be executed! ###

                            # prepend "MSGBOX" cmd for single devices to clear msgbox (as was set from cal) AND
                            # append "STREAM" cmd and newlines to end of command (it's prepared for sending now)
                            cmd = f"MSGBOX\n{cmd}\nSTREAM\n" if k == 0 else "\n"

                            # allow FW some time to stream before giving up on it
                            if k == 1:
                                start = time()
                                waitFor = 3
                                while time() - start < waitFor:  # delay timeout
                                    if self._serial[0].in_waiting:
                                        break

                            # if buffer is pre-filled (even just once) we don't need to send CMD again (ever, this RUN)
                            if self._serial[0].in_waiting:
                                streaming = True

                            if not streaming:
                                # WRITES encoded command to the serial port
                                self._serial[0].write(cmd.encode())
                                Log.d(
                                    TAG, "{} {} - Sending FREQ CMD {}".format(k, streaming, cmd))

                            # Dynamically adjust FW loop timing to match SW loop timing (sync)
                            # At k=  10: Take initial timestamp
                            # At k=  20: Calculate and set rough time with 10 samples elapsed
                            # At k= 120: Calculate and set good time with 100 samples elapsed
                            # At k=1000: Calculate and set better time with 880 samples elapsed
                            # At k%1000: Calculate and set best time with 1000 samples elapsed
                            if streaming and (k % 1000 == 0 or k == 120 or k == 20 or k == 10):
                                if k == 10:
                                    time_last_speed_adjust = time()
                                else:
                                    factor = (
                                        1000/10 if k == 20 else
                                        1000/100 if k == 120 else
                                        1000/880 if k == 1000 else
                                        1000/1000)
                                    last_speed = speed if not speed == 0 else 1000  # initial FW loop timing
                                    speed = (
                                        time() - time_last_speed_adjust)*1e3
                                    # only the first check (from k=10 to k=20) gets x100; all else 1:1
                                    speed *= factor
                                    if k <= 1000:
                                        # round down to nearest 100us
                                        speed -= (speed % 100)
                                    else:
                                        # round up to nearest 100us
                                        speed += (100 - speed % 100)
                                    time_last_speed_adjust = time()
                                    if self._err1 and self._err2:
                                        speed = 1000  # initial SW loop timing
                                    cmd = "SPEED {:.0f}\n".format(speed)
                                    # , end='\r')
                                    Log.d(TAG, "{}      ".format(cmd.strip()))
                                    if abs(speed - last_speed) > 200 and k <= 1000:
                                        # only for direct serial
                                        if self._serial[0].net_port == None:
                                            self._serial[0].write(cmd.encode())

                            # Initializes buffer and strs record
                            buffer = ''

                            # Read the leading 2 bytes on the serial response to decide the FW string format
                            # only recognize new formats (not legacy, for speed)
                            ignored = 0
                            start = time()
                            waitFor = 3  # timeout delay (seconds)
                            retry = True
                            rx_ex = False
                            while retry:
                                while time() - start < waitFor:
                                    while self._serial[0].in_waiting == 0 and time() - start < waitFor:
                                        pass
                                    if not time() - start < waitFor:
                                        break
                                    buffer = self._serial[0].read_until(
                                    ).decode()  # (1).hex()
                                    if '\n' in buffer:
                                        retry = False  # message RX'd, exit loop to process packet!
                                        break
                                    if buffer == "51":  # Q
                                        buffer = "Q"
                                        while self._serial[0].in_waiting == 0 and time() - start < waitFor:
                                            pass
                                        if not time() - start < waitFor:
                                            break
                                        buffer += self._serial[0].read(
                                            1).decode()
                                        retry = False  # message RX'd, exit loop to process packet!
                                        break
                                    else:
                                        ignored += 1
                                        Log.d(TAG, "E = {}, bad = {}".format(
                                            ignored, buffer))
                                if not ignored == 0:
                                    Log.d(
                                        TAG, "Bad or partial packet received! Threw away {} bytes...".format(ignored))

                                # Log.d(TAG, "E = {}, buff = {}".format(ignored, buffer))
                                # Log.d(TAG, "E = {}".format((time() - time_elaborate_begin)*1e3))

                                format = -1  # Unrecognized if all else fails
                                if buffer.startswith("Q"):
                                    # raw bytes w/ phase w/ temp
                                    if buffer.startswith("QA"):
                                        format = 1
                                    # raw bytes w/ phase w/o temp
                                    elif buffer.startswith("QB"):
                                        format = 2
                                    # raw bytes w/o phase w/ temp
                                    elif buffer.startswith("QC"):
                                        format = 3
                                    # raw bytes w/o phase w/o temp
                                    elif buffer.startswith("QD"):
                                        format = 4
                                    # raw deltas w/ phase w/ temp
                                    elif buffer.startswith("QE"):
                                        format = 5
                                    # raw deltas w/ phase w/o temp
                                    elif buffer.startswith("QF"):
                                        format = 6
                                    # raw deltas w/o phase w/ temp
                                    elif buffer.startswith("QG"):
                                        format = 7
                                    # raw deltas w/o phase w/o temp
                                    elif buffer.startswith("QH"):
                                        format = 8
                                    # raw deltas w/o phase w/o temp
                                    elif buffer.startswith("QI"):
                                        format = 9
                                    # QUACK! (timeout)
                                    elif buffer.startswith("QU"):
                                        Log.w(
                                            "Session auto-ended. Timeout due to lack of interaction.")
                                        # throw away "ACK!\r\n"
                                        self._serial[0].read(
                                            self._serial[0].in_waiting)
                                        self.stop()
                                        rx_ex = True
                                        break
                                elif not buffer == '':
                                    format = 0

                                # Timeout exception / connection lost
                                if time() - start >= waitFor:
                                    Log.e(TAG, "No serial response from device")
                                    self.stop()
                                    rx_ex = True
                                    break

                                # Unrecognized format
                                if format == -1:
                                    Log.d(
                                        TAG, "Unrecognized serial response \"{}\" format from device".format(buffer))
                                    retry = True

                                if format == 0:
                                    # silently ignore command responses that do not start with "Q"
                                    retry = True
                                    continue

                                if format != 9:
                                    Log.d(
                                        TAG, "Unsupported serial response \"{}\" format from device".format(buffer))
                                    retry = True

                            # If stop() called above, do not proceed to processing serial data
                            if rx_ex and self._exit.is_set():
                                Log.e(TAG, "Stopping, due to receive exception...")
                                break

                            # Original format (plain text)
                            if format == 9:

                                # READS and decodes sweep from the serial port
                                # start = time()
                                # while time() - start < waitFor: # delay timeout
                                #     buffer += self._serial[0].read_until().decode() # (1).decode(Constants.app_encoding)
                                #     if '\n' in buffer:
                                #       break
                                data_raw = buffer.split(';')
                                length = len(data_raw)

                                if length == 6 or length == 8:

                                    if length == 6:
                                        format = str(data_raw[0])
                                        sequence = int(data_raw[1])
                                        w_time = int(data_raw[2])
                                        peak_mag = int(data_raw[3])
                                        peak_freq = int(data_raw[4])
                                        left = right = 0
                                        data_temp = float(data_raw[5])

                                    if length == 8:
                                        format = str(data_raw[0])
                                        sequence = int(data_raw[1])
                                        w_time = int(data_raw[2])
                                        peak_mag = int(data_raw[3])
                                        peak_freq = int(data_raw[4])
                                        left = int(data_raw[5])
                                        right = float(data_raw[6])
                                        data_temp = float(data_raw[7])

                                    if self._freq_hopping:
                                        if startFreqs[0] - Constants.max_drift_l_hz < peak_freq < stopFreqs[0] + Constants.max_drift_r_hz:
                                            overtone = 0
                                        elif startFreqs[1] - Constants.max_drift_l_hz < peak_freq < stopFreqs[1] + Constants.max_drift_r_hz:
                                            overtone = 1
                                        elif startFreqs[2] - Constants.max_drift_l_hz < peak_freq < stopFreqs[2] + Constants.max_drift_r_hz:
                                            overtone = 2
                                        else:
                                            overtone = 0xFF
                                    else:
                                        overtone = 0xFF
                                else:
                                    Log.d("Message response length incorrect!")
                                    continue

                                # Track and detect USB serial stream anomalies
                                # i.e. dropped, corrupt or out-of-order packets
                                if self._freq_hopping and overtone == 0xFF:
                                    Log.w(TAG, "WARN: Received USB packet #{} @{} with out-of-range frequency ({})".format(
                                        sequence, w_time, peak_freq))
                                if sequence != self._seq + 1 and self._seq != 0:
                                    Log.w(TAG, "WARN: Missing {} USB packets @{} (SEQ was {}, now {})".format(
                                        sequence - self._seq, w_time, self._seq, sequence))
                                if sequence <= self._seq:
                                    Log.w(
                                        TAG, "WARN: Malformed USB packet @{} (SEQ was {}, now {})".format(w_time, self._seq, sequence))
                                if w_time <= self._ts and overtone in [0, 0xFF] and sequence != 1:
                                    Log.w(TAG, "WARN: Malformed USB packet #{} (TS was {}, now {})".format(
                                        sequence, self._ts, w_time))
                                self._seq = sequence
                                if overtone in [0, 0xFF]:
                                    self._ts = w_time

                            # this applies to every format
                            if self._freq_hopping:
                                # Update frequencies based on response type (with a safety/sanity check)
                                readFREQ = readFREQs[overtone] if overtone < len(
                                    readFREQs) else readFREQs[0]

                            # Timeout exception / connection lost
                            if time() - start >= 3:
                                Log.d(TAG, "buffer = {}, size = {}".format(
                                    buffer, len(buffer)))
                                raise TimeoutError(
                                    "No serial response from device")

                        # specify handlers for different exceptions
                        except ValueError:
                            # , end='\r')
                            Log.w(
                                TAG, "WARNING (ValueError): convert raw to float failed")
                            # Log.w(TAG, "Warning (ValueError): convert Raw to float failed")
                        except TimeoutError:
                            if self._flag_error_usb == 0:
                                Log.w(
                                    TAG, "WARNING: (TimeoutError): No serial response from device")
                            self._flag_error_usb += 1
                        except RuntimeError:
                            # , end='\r')
                            Log.w(
                                TAG, "WARNING (RuntimeError): Unrecognized serial response format")
                            # Log.d(TAG, "Please, make sure the device and software are both up-to-date!")
                        except IOError as e:
                            if self._flag_error_usb == 0:
                                Log.w(
                                    TAG, "WARNING (IOError): Device stopped responding\n{}".format(e))
                            self._flag_error_usb += 1
                        except:
                            # , end='\r')
                            Log.w(
                                TAG, "WARNING (Exception): convert raw to float failed")
                            # Log.w(TAG, "Warning (ValueError): convert Raw to float failed")

                        # Calls elaborate method to performs results
                        try:
                            # invoke elaborate queue
                            self._elaborate_in_q.put(
                                [k,
                                 sequence,
                                 w_time,
                                 peak_mag,
                                 peak_freq,
                                 left,
                                 right,
                                 data_temp,
                                 coeffs_all,
                                 overtone]
                            )
                        except ValueError:
                            self._flag_error = 1
                            if k > self._environment:
                                Log.w(TAG, "WARNING (ValueError): miscalculation")
                        except:
                            self._flag_error = 1
                            if k > self._environment:
                                Log.w(TAG, "WARNING (ValueError): miscalculation")

                        # Parse output variables from ElaborateProcess (if available)
                        if k > 0:
                            # Log.d("waiting for elaborate {}".format(k))
                            while self._elaborate_out_q.empty() and not self._exit.is_set():
                                pass  # wait for queue to fill (or exit)
                        if not self._elaborate_out_q.empty():
                            out_q = self._elaborate_out_q.get_nowait()
                            w_time = out_q[0]
                            temperature = out_q[1]
                            dissipation = out_q[2]
                            freq_peak = out_q[3]
                            freq_left = out_q[4]
                            freq_right = out_q[5]
                            self._err1 = out_q[6]
                            self._err2 = out_q[7]
                            self._err3 = out_q[8]
                            self._err4 = out_q[9]
                        else:
                            w_time = temperature = dissipation = 0
                            freq_left = freq_peak = freq_right = 0
                            self._err1 = self._err2 = self._err3 = self._err4 = 0
                            Log.i(TAG, "Waiting on Elaborate thread...")

                        last_overtone = overtone

                        self._parser.add6(
                            [self._err1, self._err2, self._err3, self._err4, k, self._flag_error_usb])
                        if k <= self._environment:
                            pass  # bar.update(k)
                        if k/50 == k//50:
                            if k == 1000:
                                # Log.i('\n')
                                Log.i(TAG, "sweep #{} in {}s              ".format(
                                    k, (time()-timestamp)))
                        # Increases sweep counter
                        k += 1

                    # Exiting...
                    if k == self._environment:
                        pass  # bar.finish()

                    #### END SWEEPS LOOP ####

                    try:  # skip these post actions if port is not available
                        # and self._is_port_available(self._serial[0].com_port):
                        if self._serial[0].is_open:

                            # only for direct or emulated serial (_com or _hid):
                            if streaming and self._serial[0].net_port == None:
                                # Log.i(TAG, "Stopping serial stream...") #, end = '\r')
                                # retry, with up to 3 attempts...
                                for x in range(3):
                                    self._serial[0].write("stop\n".encode())
                                    # wait for "STOP" reply
                                    stopped = 0
                                    stop = time()
                                    waitFor = 3  # timeout delay (seconds)
                                    while time() - stop < waitFor:
                                        while (time() - stop < waitFor and
                                               self._serial[0].in_waiting == 0):
                                            pass
                                        if time() - stop < waitFor:
                                            # "S"
                                            while not self._serial[0].read(1).hex() == "53":
                                                pass
                                            # "T"
                                            if self._serial[0].read(1).hex() == "54":
                                                # "O"
                                                if self._serial[0].read(1).hex() == "4f":
                                                    # "P"
                                                    if self._serial[0].read(1).hex() == "50":
                                                        # "\r"
                                                        if self._serial[0].read(1).hex() == "0d":
                                                            # "\n"
                                                            if self._serial[0].read(1).hex() == "0a":
                                                                # Log.i(TAG, "Serial stream stopped!   ")
                                                                stopped = 1
                                                                break
                                    if stopped == 1:
                                        break
                                    if x == 2:
                                        Log.e(
                                            TAG, "Failed to stop serial stream!")
                    except:
                        # failed to stop stream
                        Log.w(
                            TAG, "WARNING: Failed to stop device stream. Error parsing response.")
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
            if self._serial[0].is_open:
                # Propogate stop down to dependent processes.
                self._elaborate_process.stop()

                stop = time()
                waitFor = 1  # timeout delay (seconds)
                while time() - stop < waitFor:
                    if not self._elaborate_process.is_running():
                        break

                self._elaborate_process.terminate()
                self._elaborate_process.join()

                # CLOSES serial port
                for i in range(len(self._serial)):
                    self._serial[i].close()

            # gracefully end subprocess
            self._done.set()

    def is_running(self):
        return not self._done.is_set()

    ###########################################################################
    # Stops acquiring data
    ###########################################################################

    def stop(self):
        # Signals the process to stop acquiring data.
        self._exit.set()

    ###########################################################################
    # Automatically selects the serial ports for Teensy (macox/windows)
    ###########################################################################

    @staticmethod
    def get_ports():
        return serial.enumerate()
        from QATCH.common.architecture import Architecture, OSType
        if Architecture.get_os() is OSType.macosx:
            import glob
            return glob.glob("/dev/tty.usbmodem*")
        elif Architecture.get_os() is OSType.linux:
            import glob
            return glob.glob("/dev/ttyACM*")
        else:
            found_ports = []
            port_connected = []
            found = False
            ports_avaiable = list(list_ports.comports())
            for port in ports_avaiable:
                if port[2].startswith("USB VID:PID=16C0:0483"):
                    found = True
                    port_connected.append(port[0])
            if found:
                found_ports = port_connected
            return found_ports

    ###########################################################################
    # Gets a list of the Overtones reading from file
    ###########################################################################

    @staticmethod
    def get_speeds(i):
        # :return: List of the Overtones :rtype: str list.
        # Loads frequencies from  file (path: 'common\')
        try:
            path = Constants.cvs_peakfrequencies_path
            path = FileStorage.DEV_populate_path(path, i)
            data = loadtxt(path)
            peaks_mag = data[:, 0]
            reversed_peaks_mag = peaks_mag[::-1]
        except:
            reversed_peaks_mag = [0]
        return [str(int(v)) for v in reversed_peaks_mag]

    ###########################################################################
    # Checks if the serial port is currently connected
    ###########################################################################

    def _is_port_available(self, port):
        """
        :param port: Port name to be verified.
        :return: True if the port is connected to the host :rtype: bool.
        """
        if Constants.serial_simulate_device:
            return True

        dm = Discovery()
        if self._serial[0].net_port == None:
            net_exists = False
        else:
            net_exists = dm.ping(self._serial[0].net_port)
        for p in self.get_ports():
            if p == port:
                return True
        if port == None:
            if len(dm.doDiscover()) > 0:
                return net_exists
        return False

    ###########################################################################
    # Sets frequency range for the corresponding overtone
    ###########################################################################

    def get_frequencies(self, samples, i):
        """
        :param samples: Number of samples :type samples: int.
        :return: overtone :rtype: float.
        :return: fStep, frequency step  :rtype: float.
        :return: readFREQ, frequency range :rtype: float list.
        """
        # Loads frequencies from file
        peaks_mag, _, left_bounds, right_bounds, baselines = self.load_frequencies_file(
            i)

        # Checks QCS type 5Mhz or 10MHz
        # Sets start and stop frequencies for the corresponding overtone
        if (peaks_mag[0] > 4e+06 and peaks_mag[0] < 6e+06):
            switch = Overtone_Switcher_5MHz(
                peak_frequencies=peaks_mag, left_bounds=left_bounds, right_bounds=right_bounds)
            # 0=fundamental, 1=3rd overtone and so on
            (overtone_name, overtone_value, self._startFreq, self._stopFreq, SG_window_size,
             spline_factor) = switch.overtone5MHz_to_freq_range(self._overtone_int)

            # Added for freq hopping
            if self._freq_hopping:
                (overtone_name_up, overtone_value_up, self._startFreq_up, self._stopFreq_up, SG_window_size_up,
                 spline_factor_up) = switch.overtone5MHz_to_freq_range(min(self._overtone_int+1, len(peaks_mag)-1))
                (overtone_name_down, overtone_value_down, self._startFreq_down, self._stopFreq_down, SG_window_size_down,
                 spline_factor_down) = switch.overtone5MHz_to_freq_range(max(self._overtone_int-1, 0))
            else:
                self._startFreq_up = 0
                self._stopFreq_up = 0
                self._startFreq_down = 0
                self._stopFreq_down = 0
            ###################################################################
            Log.i(TAG, "QATCH Device setup: @5MHz")
        elif (peaks_mag[0] > 9e+06 and peaks_mag[0] < 11e+06):
            switch = Overtone_Switcher_10MHz(
                peak_frequencies=peaks_mag, left_bounds=left_bounds, right_bounds=right_bounds)
            (overtone_name, overtone_value, self._startFreq, self._stopFreq, SG_window_size,
             spline_factor) = switch.overtone10MHz_to_freq_range(self._overtone_int)
            Log.i(TAG, "QATCH Device setup: @10MHz")

        # Sets the frequency step
        fStep = (self._stopFreq-self._startFreq)/(samples-1)
        # Added for freq hopping
        if self._freq_hopping:
            fStep_up = (self._stopFreq_up-self._startFreq_up)/(samples-1)
            fStep_down = (self._stopFreq_down-self._startFreq_down)/(samples-1)

        # Handle step-size remainders and adjust stops accordingly
        d = fStep - int(fStep)
        if not d == 0:
            self._stopFreq -= d*(samples-1)
            fStep = int(fStep)
        # Added for freq hopping
        if self._freq_hopping:
            d_up = fStep_up - int(fStep_up)
            if not d_up == 0:
                self._stopFreq_up -= d*(samples-1)
                fStep_up = int(fStep_up)
            d_down = fStep_down - int(fStep_down)
            if not d_down == 0:
                self._stopFreq_down -= d*(samples-1)
                fStep_down = int(fStep_down)

        # Sets spline points for fitting
        spline_points = int((self._stopFreq-self._startFreq))+1
        # Added for freq hopping
        if self._freq_hopping:
            spline_points_up = int((self._stopFreq_up-self._startFreq_up))+1
            spline_points_down = int(
                (self._stopFreq_down-self._startFreq_down))+1

        # Sets the frequency range for the corresponding overtone
        readFREQ = np.arange(samples) * (fStep) + self._startFreq
        # Added for freq hopping
        if self._freq_hopping:
            readFREQ_up = np.arange(samples) * (fStep_up) + self._startFreq_up
            readFREQ_down = np.arange(
                samples) * (fStep_down) + self._startFreq_down

        # Sets the baseline counts for the corresponding overtone
        baseline = baselines[self._overtone_int]
        if self._freq_hopping:
            baseline_up = baselines[min(
                self._overtone_int+1, len(peaks_mag)-1)]
            baseline_down = baselines[max(self._overtone_int-1, 0)]

        # Added bottom 2 lines for freq hopping
        if self._freq_hopping:
            return (overtone_name, overtone_value, fStep, readFREQ, SG_window_size, spline_points, spline_factor, baseline, self._startFreq, self._stopFreq,
                    overtone_name_up, overtone_value_up, fStep_up, readFREQ_up, SG_window_size_up, spline_points_up, spline_factor_up, baseline_up, self._startFreq_up, self._stopFreq_up,
                    overtone_name_down, overtone_value_down, fStep_down, readFREQ_down, SG_window_size_down, spline_points_down, spline_factor_down, baseline_down, self._startFreq_down, self._stopFreq_down)
        else:
            return (overtone_name, overtone_value, fStep, readFREQ, SG_window_size, spline_points, spline_factor, baseline, self._startFreq, self._stopFreq)

    ###########################################################################
    # Loads Fundamental frequency and Overtones from file
    ###########################################################################

    def load_frequencies_file(self, i):
        path = Constants.cvs_peakfrequencies_path
        path = FileStorage.DEV_populate_path(path, i)
        data = loadtxt(path)
        peaks_mag = data[:, 0]
        peaks_phase = data[:, 1]  # unused at the moment

        if len(data[0]) >= 3:
            left_bounds = data[:, 2]
        else:
            left_bounds = None

        if len(data[0]) >= 4:
            right_bounds = data[:, 3]
        else:
            right_bounds = None

        if len(data[0]) >= 5:
            baselines = data[:, 4]
        else:
            baselines = None

        return peaks_mag, peaks_phase, left_bounds, right_bounds, baselines

    ###########################################################################
    # Loads Calibration (baseline correction) from file
    ###########################################################################

    def load_calibration_file(self, i):
        # Loads Fundamental frequency and Overtones from file
        peaks_mag, _, _, _, _ = self.load_frequencies_file(i)

        # Checks QCS type 5Mhz or 10MHz
        if (peaks_mag[0] > 4e+06 and peaks_mag[0] < 6e+06):
            filename = Constants.csv_calibration_path
        elif (peaks_mag[0] > 9e+06 and peaks_mag[0] < 11e+06):
            filename = Constants.csv_calibration_path10
        filename = FileStorage.DEV_populate_path(filename, i)

        data = loadtxt(filename)
        freq_all = data[:, 0]
        mag_all = data[:, 1]

        if len(data[0]) >= 3:
            phase_all = data[:, 2]
        else:
            phase_all = None

        return freq_all, mag_all, phase_all
