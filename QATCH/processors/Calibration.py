import multiprocessing
from QATCH.core.constants import Constants
from QATCH.common.fileStorage import FileStorage
from QATCH.common.findDevices import Discovery
from QATCH.common.logger import Logger as Log

# from progress.bar import Bar
# from progressbar import Bar, Percentage, ProgressBar, RotatingMarker, Timer

import sys
import os
from time import time
from serial.tools import list_ports
import numpy as np
# import scipy.signal # unused
from numpy import loadtxt
import logging
from logging.handlers import QueueHandler

if Constants.serial_simulate_device:
    from QATCH.processors.Simulator import serial  # simulator
else:
    from QATCH.processors.Device import serial  # real device hardware


TAG = ""  # "[Calibration]"

###############################################################################
# Process for the serial package and the communication with the serial port
# Processes incoming data and calculates outgoing data by the algorithms
###############################################################################


class CalibrationProcess(multiprocessing.Process):

    ###########################################################################
    # BASELINE ESTIMATION
    # Estimates Baseline with Least Squares Polynomial Fit (LSP)
    ###########################################################################
    def baseline_estimation(self, x, y, poly_order):
        # Least Squares Polynomial Fit (LSP)
        coeffs = np.polyfit(x, y, poly_order)
        # Evaluate a polynomial at specific values
        poly_fitted = np.polyval(coeffs, x)
        return poly_fitted, coeffs

    ###########################################################################
    # BASELINE CORRECTION
    # estimates signal-baseline for amplitude and phase
    ###########################################################################
    def baseline_correction(self, readFREQ, data_mag, data_ph=None):

        # input signal Amplitude
        (self._polyfitted_all, self._coeffs_all) = self.baseline_estimation(
            readFREQ, data_mag, 8)
        self._mag_baseline_corrected_all = data_mag - self._polyfitted_all

        # self._mag_baseline_corrected_all = self.fft_filter(readFREQ, self._mag_baseline_corrected_all)

        if data_ph is None:
            return self._mag_baseline_corrected_all

        # input signal Phase
        (self._polyfitted_all_phase, self._coeffs_all_phase) = self.baseline_estimation(
            readFREQ, data_ph, 8)
        self._phase_beseline_corrected_all = data_ph - self._polyfitted_all_phase
        return self._mag_baseline_corrected_all, self._phase_beseline_corrected_all

    def fft_filter(self, x, y):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.fftpack import rfft, irfft, fftfreq, fft

        # Number of samplepoints
        N = len(x)
        # sample spacing
        T = x[1]-x[0]

        yf = fft(y)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        # fft end

        f_signal = rfft(y)

        cut_f_signal = f_signal.copy()
        limit = np.amax(f_signal)/10
        for i in range(len(cut_f_signal)):
            if np.abs(cut_f_signal[i]) > limit:
                cut_f_signal[i] = 0

        cut_signal = irfft(cut_f_signal)

        do_debug = False
        if do_debug:
            # plot results
            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 3))
            axarr = (ax1, ax2, ax3, ax4)

            axarr[0].plot(x, y)
            axarr[1].plot(x, cut_signal)
            axarr[2].plot(x, f_signal)
            axarr[3].plot(x, cut_f_signal)

            f.show()

        return cut_signal

    ###########################################################################
    # PEAK DETECTION
    # Calculates the relative extrema of data using Signal Processing Toolbox
    ###########################################################################
    def FindPeak(self, freq, mag, phase=None, dist=1):
        # Do this MUCH faster than using the scipy library
        max_search_dist = 30
        padding_size_dB = 3.0
        max_freq_drift = Constants.calibration_fStep * dist / 2

        #######################################
        # START ARGRELEXTREMA() FOR MAGNITUDE #
        #######################################

        # sort frequencies of peaks in order of descending magnitude
        dtypes = [('freq', int), ('mag', float)]
        values = np.vstack((freq, mag)).T
        values = list(map(tuple, values))
        array = np.array(values, dtype=dtypes)
        sorted = np.sort(array, order='mag')[::-1]

        # find the relative extrema for magnitude
        tmp_max_mag = np.linspace(0, 0, 10)
        tmp_max_mag_i = np.linspace(0, 0, 10)
        numPeaks = 0

        # ex: 1 searches 10-19.99 MHz range, etc, etc
        peaks_to_find = [0, 1, 2, 3, 4]
        ignored_peaks = []
        ignored_count = 0
        max_peak_mag = 0
        for x in range(len(sorted)):
            this_freq = sorted[x][0]
            this_peak = int(this_freq / 10e6)
            if this_peak in peaks_to_find:
                allow_drift = (2.5 + 1*this_peak) * 1e6
                target_freq = (5 + 10*this_peak) * 1e6
                actual_drift = abs(target_freq - this_freq)
                if actual_drift < allow_drift:
                    peaks_to_find.remove(this_peak)
                    Log.i(TAG, "Found peak {} at {:8.0f} with drift (allowed = {:7.0f}, actual = {:7.0f}) Hz".format(
                        this_peak, this_freq, allow_drift, actual_drift))
                    tmp_max_mag[numPeaks] = this_freq
                    tmp_max_mag_i[numPeaks] = np.where(freq == this_freq)[0][0]
                    numPeaks += 1
                    if max_peak_mag == 0:
                        max_peak_mag = sorted[x][1]
                else:
                    # Log.d("*Ignored peak at {:8.0f} with drift (allowed = {:7.0f}, actual = {:7.0f}) Hz".format(this_freq, allow_drift, actual_drift))
                    if not this_peak in ignored_peaks:
                        ignored_peaks.append(this_peak)
                    ignored_count += 1
            if len(peaks_to_find) == 0:
                break
        if ignored_count != 0:
            Log.w(TAG, "WARNING: {} bigger peaks were ignored (too much drift) for the following overtone(s): {}".format(
                ignored_count, ignored_peaks))
            Log.w(
                TAG, "Please confirm these peaks were detected correctly. Repeat Initialize if needed.")
        Log.i(
            TAG, f"The largest peak magnitude detected is: {max_peak_mag} dB @ {tmp_max_mag[0]} Hz")
        # trigger failure if largest peak is too small (no crystal)
        if max_peak_mag < padding_size_dB:
            tmp_max_mag[numPeaks] = freq[0]
            tmp_max_mag_i[numPeaks] = 0
            numPeaks += 1

        # Find -3dB points (L/R bounds)
        mag_bounds_L = np.linspace(0, 0, numPeaks)
        mag_bounds_R = np.linspace(0, 0, numPeaks)
        for x in range(numPeaks):  # for each peak found
            i = int(tmp_max_mag_i[x])
            limit = mag[i] - padding_size_dB
            for y in range(1, max_search_dist+1):  # scan left and right from peak
                if i-y >= 0:
                    if (mag[i-y] < limit or y == max_search_dist) and mag_bounds_L[x] == 0:
                        mag_bounds_L[x] = freq[i-y]  # record L bound
                if i+y < len(mag):
                    if (mag[i+y] < limit or y == max_search_dist) and mag_bounds_R[x] == 0:
                        mag_bounds_R[x] = freq[i+y]  # record R bound
                if not (mag_bounds_L[x] == 0 or mag_bounds_R[x] == 0):
                    break  # continue to next peak

        # sort indices from smallest to largest
        tmp_max_mag_i = tmp_max_mag_i[0:numPeaks]
        tmp_max_mag_i.sort()
        mag_bounds_L.sort()
        mag_bounds_R.sort()
        # DO NOT USE tmp_max_mag array now that index associations are muddy...

        ###################################
        # START ARGRELEXTREMA() FOR PHASE #
        ###################################

        if not phase is None:
            # find the relative extrema for phase
            tmp_max_phase = np.linspace(0, 0, 10)
            tmp_max_phase_i = np.linspace(0, 0, 10)
            numPeaks = 0

            for x in range(10):  # scan tmp_max_phase array
                for y in range(len(phase) - 1):  # scan phase array`
                    if tmp_max_phase[x] <= phase[y]:  # find next maximum
                        isDup = 0
                        for z in range(x):
                            if abs(tmp_max_phase_i[z] - y) < dist:
                                isDup = 1

                        if not isDup:
                            if tmp_max_phase[x] == 0 and tmp_max_phase_i[x] == 0:
                                numPeaks += 1
                            tmp_max_phase[x] = phase[y]
                            tmp_max_phase_i[x] = y
                # if no match found this round, BAIL
                if tmp_max_phase[x] == 0 and tmp_max_phase_i[x] == 0:
                    break

            # Find -3dB points (L/R bounds)
            phase_bounds_L = np.linspace(0, 0, numPeaks)
            phase_bounds_R = np.linspace(0, 0, numPeaks)
            for x in range(numPeaks):  # for each peak found
                i = int(tmp_max_phase_i[x])
                limit = phase[i] - padding_size_dB
                for y in range(1, max_search_dist+1):  # scan left and right from peak
                    if i-y >= 0:
                        if (phase[i-y] < limit or y == max_search_dist) and phase_bounds_L[x] == 0:
                            phase_bounds_L[x] = freq[i-y]  # record L bound
                    if i+y < len(phase):
                        if (phase[i+y] < limit or y == max_search_dist) and phase_bounds_R[x] == 0:
                            phase_bounds_R[x] = freq[i+y]  # record R bound
                    if not (phase_bounds_L[x] == 0 or phase_bounds_R[x] == 0):
                        break  # continue to next peak

            # sort indices from smallest to largest
            tmp_max_phase_i = tmp_max_phase_i[0:numPeaks]
            tmp_max_phase_i.sort()
            phase_bounds_L.sort()
            phase_bounds_R.sort()
            # DO NOT USE tmp_max_phase array now that index associations are muddy...
        else:
            tmp_max_phase_i = np.linspace(0, 0, numPeaks)

        ################################
        # STORE OFF RESULTS AND RETURN #
        ################################

        self.max_indexes_mag = list(tmp_max_mag_i.astype('int32'))
        self.max_indexes_phase = list(tmp_max_phase_i.astype('int32'))

        # local maxima amplitude and baselines
        # freq[self.max_indexes_mag]
        self.max_freq_mag = np.asarray([freq[i] for i in self.max_indexes_mag])
        self.max_value_mag = np.asarray(
            [mag[i] for i in self.max_indexes_mag])  # mag[self.max_indexes_mag]
        polyfit_all = np.asarray([self._polyfitted_all[i]
                                 for i in self.max_indexes_mag])
        self.max_baselines = self._convertMagnitudeToADC(polyfit_all).astype(
            int)  # self._polyfitted_all[self.max_indexes_mag]).astype(int)

        Log.d(TAG, "size_L = {}".format(self.max_freq_mag - mag_bounds_L))
        Log.d(TAG, "size_R = {}".format(mag_bounds_R - self.max_freq_mag))
        Log.d(TAG, "baselines = {}".format(self.max_baselines))

        if phase is None:
            return self.max_freq_mag, self.max_value_mag, self.max_freq_mag-mag_bounds_L, mag_bounds_R-self.max_freq_mag, self.max_baselines

        phase_bounds_L = np.resize(phase_bounds_L, len(mag_bounds_L))
        phase_bounds_R = np.resize(phase_bounds_R, len(mag_bounds_R))

        # map window bounds to smallest/largest of found -3dB points
        # using min()/max() + map() + zip()
        # Minimum/Maximum index value
        test_list_min = np.array([mag_bounds_L, phase_bounds_L])
        mag_bounds_L = list(map(min, zip(*test_list_min)))
        test_list_max = np.array([mag_bounds_R, phase_bounds_R])
        mag_bounds_R = list(map(max, zip(*test_list_max)))

        Log.d(TAG, "size_L = {}".format(mag_bounds_L))
        Log.d(TAG, "size_R = {}".format(mag_bounds_R))

        # local maxima phase
        self.max_freq_phase = freq[self.max_indexes_phase]
        self.max_value_phase = phase[self.max_indexes_phase]

        return self.max_freq_mag, self.max_value_mag, self.max_freq_phase, self.max_value_phase, self.max_freq_mag-mag_bounds_L, mag_bounds_R-self.max_freq_mag

    def _convertADCtoMagnitude(self, adc):
        return (adc * Constants.ADCtoVolt - Constants.VCP) / 0.029

    def _convertADCtoPhase(self, adc):
        return (adc * Constants.ADCtoVolt - Constants.VCP) / 0.01

    def _convertMagnitudeToADC(self, val):
        return (val * 0.029 + Constants.VCP) / Constants.ADCtoVolt

    ###########################################################################
    # Initializing values for process
    ###########################################################################

    def __init__(self, queue_log, parser_process):
        """
        :param parser_process: Reference to a ParserProcess instance.
        :type parser_process: ParserProcess.
        """
        self._queueLog = queue_log  # Log.create()

        multiprocessing.Process.__init__(self)
        self._exit = multiprocessing.Event()
        self._done = multiprocessing.Event()

        # Instantiate a ParserProcess class for each communication channels
        self._parser = parser_process
        # self._parser1 = parser_process
        # self._parser2 = parser_process
        # self._parser3 = parser_process
        # self._parser4 = parser_process
        # self._parser5 = parser_process
        # self._parser6 = parser_process
        self._serial = []
        self._serial.append(serial.Serial())

    ###########################################################################
    # Opens a specified serial port
    ###########################################################################
    def open(self, port, pid,
             speed=Constants.serial_default_QCS,
             timeout=Constants.serial_timeout_ms,
             writeTimeout=Constants.serial_writetimeout_ms):
        """
        :param port: Serial port name :type port: str.
        :param speed: Baud rate, in bps, to connect to port :type speed: int.
        :param timeout: Sets current read timeout :type timeout: float (seconds).
        :param writeTimeout: Sets current write timeout :type writeTimeout: float (seconds).
        :return: True if the port is available :rtype: bool.
        """
        self._serial.clear()
        for j in range(len(port)):
            self._serial.append(serial.Serial())
            self._serial[j].port = port[j] if isinstance(port, list) else port
            self._serial[j].baudrate = Constants.serial_default_speed  # 115200
            self._serial[j].stopbits = serial.STOPBITS_ONE
            self._serial[j].bytesize = serial.EIGHTBITS
            self._serial[j].timeout = timeout
            self._serial[j].write_timeout = writeTimeout
            if not isinstance(port, list):
                break

        self._QCStype = speed

        # Checks QCStype to calibrate
        if self._QCStype == '@5MHz_QCM':
            self._QCStype_int = 0
        elif self._QCStype == '@10MHz_QCM':
            self._QCStype_int = 1
        Log.i(TAG, f"Selected Quartz Crystal Sensor: {self._QCStype}")

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

            # # initializations
            # self._polyfitted_all = None
            # self._coeffs_all = None
            # self._polyfitted_all_phase = None
            # self._coeffs_all_phase = None
            # self._mag_baseline_corrected_all = None
            # self._phase_beseline_corrected_all = None
            # self._flag = 0
            # self._flag2 = 0

            # # define missing vars to indicate error without exception
            # samples   = Constants.calibration_default_samples
            # data_mag_baseline = data_ph_baseline = np.linspace(0, 0, samples)
            # mult_mag_baseline = []
            # mult_ph_baseline = []
            # data_ph = None

            multi_mag_baseline = []
            multi_ph_baseline = []
            multi_flag = []
            multi_flag2 = []

            # Checks if the serial port is currently connected
            is_available = True
            for j in range(len(self._serial)):
                is_available &= self._is_port_available(self._serial[j].port)
            if is_available:

                # Sets start, stop, step and range frequencies
                startFreq = Constants.calibration_frequency_start
                stopFreq = Constants.calibration_frequency_stop
                samples = Constants.calibration_default_samples
                fStep = Constants.calibration_fStep
                readFREQ = Constants.calibration_readFREQ

                # Gets the state of the serial ports
                all_ports_open = True
                for j in range(len(self._serial)):
                    if not self._serial[j].is_open:
                        # OPENS the serial port
                        try:
                            self._serial[j].open()
                            self._serial[j].reset_input_buffer()
                            self._serial[j].reset_output_buffer()
                        except OSError as err:
                            Log.e(err)
                            self.stop()
                            self._flag = self._flag2 = 1
                            # return
                    all_ports_open &= self._serial[j].is_open

                if not all_ports_open:
                    # port already open
                    Log.w(TAG, "WARNING: Cannot connect! Serial port is already open.")
                    Log.w(TAG, "Please, repeat Initialize again!")
                    raise PermissionError(
                        "Cannot connect! Serial port is already open.")

                for j in range(len(self._serial)):

                    # initializations
                    self._polyfitted_all = None
                    self._coeffs_all = None
                    self._polyfitted_all_phase = None
                    self._coeffs_all_phase = None
                    self._mag_baseline_corrected_all = None
                    self._phase_beseline_corrected_all = None
                    self._flag = 0
                    self._flag2 = 0

                    # define missing vars to indicate error without exception
                    samples = Constants.calibration_default_samples
                    data_mag_baseline = data_ph_baseline = np.linspace(
                        0, 0, samples)
                    data_ph = None

                    # Initializes the sweep counter
                    k = 0
                    if not self._exit.is_set():
                        Log.i(TAG, 'Initialize Process Started')
                        Log.i(
                            TAG, 'The operation will take a few seconds to complete... please wait...')
                        # raise Exception("This is a dummy exception!")

                    format = -1
                    overtone = 0

                    #### SWEEPS LOOP ####
                    while not self._exit.is_set():
                        # Boolean variable to process exceptions
                        self._flag = 0
                        self._flag2 = 0
                        # data reset for new sweep
                        data_mag = np.linspace(0, 0, samples)
                        data_ph = np.linspace(0, 0, samples)
                        try:
                            # CREATE command string
                            cmd = str(startFreq) + ';' + \
                                str(stopFreq) + ';' + str(int(fStep))

                            # append newline to end of command (it's prepared for sending now)
                            cmd += "\n"

                            if j > 0:  # primary port already open, indicate initializing
                                self._serial[0].write(
                                    "MULTI INIT 1\n".encode())
                            else:
                                self._serial[j].write(b"MSGBOX\n")

                            # WRITES encoded command to the serial port
                            self._serial[j].write(cmd.encode())

                            # Initialize buffer
                            buffer = ''

                            # Initialize ProgressBar
                            # bar = ProgressBar(widgets=[TAG,' ', Bar(marker='>'),' ',Percentage(),' ', Timer()]).start()

                            # Read the leading 2 bytes on the serial response to decide the FW string format
                            # only recognize new formats (not legacy, for speed and logical simplicity)
                            ignored = 0
                            start = time()
                            waitFor = 3  # timeout delay (seconds)
                            while time() - start < waitFor:
                                while self._serial[j].in_waiting == 0 and time() - start < waitFor:
                                    pass
                                if not time() - start < waitFor:
                                    break
                                buffer = self._serial[j].read(1).hex()
                                if buffer == "51":  # Q
                                    buffer = "Q"
                                    while self._serial[j].in_waiting == 0 and time() - start < waitFor:
                                        pass
                                    if not time() - start < waitFor:
                                        break
                                    buffer += self._serial[j].read(1).decode()
                                    break
                                elif buffer == "53":  # S (shift ack)
                                    # command acknowledged (do nothing)
                                    pass
                                else:
                                    ignored += 1
                            if not ignored == 0:
                                Log.d(
                                    TAG, "Bad or partial packet received! Threw away {} bytes...".format(ignored))

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
                            elif not buffer == '':
                                format = 0

                            # Timeout exception / connection lost
                            if time() - start >= waitFor:
                                Log.e(TAG, "Stopping, due to receive exception...")
                                raise Exception()

                            # Unrecognized format
                            if format == -1:
                                raise RuntimeError(
                                    "Unrecognized serial response format from device")

                            # Original format (plain text)
                            if format == 0:
                                self.maxval = samples
                                # Initialize strs
                                strs = ["" for x in range(samples + 2)]

                                # READS and decodes sweep from the serial port
                                while 1:
                                    start = time()
                                    while self._serial[j].in_waiting == 0 and time() - start < waitFor:
                                        pass
                                    if not time() - start < waitFor:
                                        Log.e(
                                            TAG, "Stopping, due to missing data...")
                                        raise Exception()
                                    start = time()
                                    # Constants.app_encoding
                                    buffer += self._serial[j].read(
                                        self._serial[j].in_waiting).decode()
                                    len_buffer = len(buffer)
                                    self._parser.add6(
                                        [self._flag, self._flag2, self._flag2, len_buffer / 17, False, False])
                                    # bar.update(len_buffer / 17)
                                    if 's' in buffer:
                                        break

                                # from a full buffer to a list of string
                                data_raw = buffer.split('\n')
                                length = len(data_raw)

                                # PERFORMS split with the semicolon delimiter
                                for i in range(length):
                                    strs[i] = data_raw[i].split(';')

                                # converts the sweep samples before adding to queue
                                for i in range(length - 2):
                                    data_mag[i] = self._convertADCtoMagnitude(
                                        float(strs[i][0]))
                                    data_ph[i] = self._convertADCtoPhase(
                                        float(strs[i][1]))

                            # Faster format (raw bytes) w/ phase
                            if format == 1 or format == 2:
                                # a single line of data is 8 characters long (4 hex bytes) + 6 for "QA" and temp
                                self.maxval = samples

                                valsRxd = 0
                                head = 2  # ignore the header bytes
                                tail = 0
                                size = 4  # 2 bytes are parsed as 4 characters, i.e. "0123"

                                # READS and decodes sweep from the serial port simultaneously
                                while 1:
                                    start = time()
                                    while head == tail and self._serial[j].in_waiting == 0 and time() - start < waitFor:
                                        pass
                                    if not time() - start < waitFor:
                                        Log.e(
                                            TAG, "Stopping, due to missing data...")
                                        raise Exception()
                                    start = time()
                                    # Constants.app_encoding
                                    buffer += self._serial[j].read(
                                        self._serial[j].in_waiting).hex()
                                    tail = len(buffer)

                                    if head != tail:
                                        if (tail - head) >= size:
                                            i = int(valsRxd / 2)
                                            valsRxd += 1
                                            val = float(
                                                int(buffer[head:head+size], 16))
                                            head += size

                                            # end loop once all samples are populated (ignore temp for cal)
                                            if i >= samples:
                                                self._serial[j].reset_input_buffer(
                                                )
                                                break

                                            # converts the sweep samples before adding to queue
                                            if valsRxd % 2:
                                                data_mag[i] = self._convertADCtoMagnitude(
                                                    val)
                                            else:
                                                data_ph[i] = self._convertADCtoPhase(
                                                    val)

                                            # Update progress every 1% samples to focus on speed
                                            if valsRxd % int(self.maxval / 100) == 0:
                                                self._parser.add6(
                                                    [self._flag, self._flag2, self._flag2, i, False, False])
                                                # bar.update(i)

                            # Faster format (raw bytes) w/o phase
                            if format == 3 or format == 4:
                                self.maxval = samples
                                data_ph = None

                                i = -1
                                head = 2  # ignore the header bytes
                                tail = 0
                                size = 4  # 2 bytes are parsed as 4 characters, i.e. "0123"

                                # READS and decodes sweep from the serial port simultaneously
                                while 1:
                                    start = time()
                                    while head == tail and self._serial[j].in_waiting == 0 and time() - start < waitFor:
                                        pass
                                    if not time() - start < waitFor:
                                        Log.e(
                                            TAG, "Stopping, due to missing data...")
                                        raise Exception()
                                    start = time()
                                    # Constants.app_encoding
                                    buffer += self._serial[j].read(
                                        self._serial[j].in_waiting).hex()
                                    tail = len(buffer)

                                    if head != tail:
                                        if (tail - head) >= size:
                                            # i = int(valsRxd / 2)
                                            i += 1
                                            val = float(
                                                int(buffer[head:head+size], 16))
                                            head += size

                                            # end loop once all samples are populated (ignore temp for cal)
                                            if i >= samples:
                                                self._serial[j].reset_input_buffer(
                                                )
                                                break

                                            # converts the sweep samples before adding to queue
                                            data_mag[i] = self._convertADCtoMagnitude(
                                                val)

                                            # Update progress every 1% samples to focus on speed
                                            if i % int(self.maxval / 100) == 0:
                                                self._parser.add6(
                                                    [self._flag, self._flag2, self._flag2, i, False, False])
                                                # bar.update(i)

                            # Faster format (delta bytes) w/ phase
                            if format == 5 or format == 6:
                                self.maxval = samples

                                i = -1
                                valsRxd = 0
                                head = 2  # ignore the header bytes
                                tail = 0
                                size = 4  # 2 bytes are parsed as 4 characters, i.e. "0123"
                                lastVal_mag = 0
                                lastVal_ph = 0

                                # READS and decodes sweep from the serial port simultaneously
                                while 1:
                                    start = time()
                                    while head == tail and self._serial[j].in_waiting == 0 and time() - start < waitFor:
                                        pass
                                    if not time() - start < waitFor:
                                        Log.e(
                                            TAG, "Stopping, due to missing data...")
                                        raise Exception()
                                    start = time()
                                    if (format == 5 and tail <= samples*2 + 13) or (format == 6 and tail <= samples*2 + 9):
                                        while self._serial[j].in_waiting:
                                            # Constants.app_encoding
                                            buffer += self._serial[j].read(
                                                1).hex()
                                            tail = len(buffer)
                                            if (format == 5 and tail >= samples*2 + 13) or (format == 6 and tail >= samples*2 + 9):
                                                break

                                    if head != tail:
                                        if (tail - head) >= size:
                                            if i == -1:
                                                # get overtone indicator byte on first sample only
                                                overtone = int(
                                                    buffer[head:head+2], 16)
                                                head += 2

                                            i = int(valsRxd / 2)
                                            valsRxd += 1
                                            val = float(
                                                int(buffer[head:head+size], 16))
                                            head += size

                                            if valsRxd == 2:
                                                size = 1  # deltas are just 4-bits

                                            # end loop once all samples are populated (ignore temp for cal)
                                            if i >= samples:
                                                self._serial[j].reset_input_buffer(
                                                )
                                                break

                                            if not i == 0:
                                                val = val if val < 8 else val - 16  # wrap 4-bit val to negative
                                                # compression ratio (MUST MATCH FW)
                                                val *= 4

                                            # converts the sweep samples before adding to queue
                                            if valsRxd == 1:
                                                lastVal_mag = val  # store off ADC counts for next delta
                                                data_mag[i] = self._convertADCtoMagnitude(
                                                    val)
                                            elif valsRxd == 2:
                                                lastVal_ph = val  # store off ADC counts for next delta
                                                data_ph[i] = self._convertADCtoPhase(
                                                    val)
                                            elif (valsRxd-3) % 4 < 2:
                                                val += lastVal_mag
                                                lastVal_mag = val  # store off ADC counts for next delta
                                                i += (valsRxd-1) % 2
                                                data_mag[i] = self._convertADCtoMagnitude(
                                                    val)
                                            else:
                                                val += lastVal_ph
                                                lastVal_ph = val  # store off ADC counts for next delta
                                                i += (valsRxd-1) % 2 - 1
                                                data_ph[i] = self._convertADCtoPhase(
                                                    val)

                                            # Update progress every 1% samples to focus on speed
                                            if i % int(self.maxval / 100) == 0:
                                                self._parser.add6(
                                                    [self._flag, self._flag2, self._flag2, i, False, False])
                                                # bar.update(i)

                            # Faster format (delta bytes) w/o phase
                            if format == 7 or format == 8:
                                self.maxval = samples * len(self._serial)
                                data_ph = None

                                i = -1
                                head = 2  # ignore the header bytes
                                tail = 0
                                size = 4  # 2 bytes are parsed as 4 characters, i.e. "0123"
                                lastVal = 0

                                # READS and decodes sweep from the serial port simultaneously
                                while 1:
                                    start = time()
                                    while head == tail and self._serial[j].in_waiting == 0 and time() - start < waitFor:
                                        pass
                                    if not time() - start < waitFor:
                                        Log.e(
                                            TAG, "Stopping, due to missing data...")
                                        raise Exception()
                                    if (format == 7 and tail <= samples + 9) or (format == 8 and tail <= samples + 5):
                                        while self._serial[j].in_waiting:
                                            # ASCII format specifier: space character 0x20 (see FW code)
                                            if overtone == 0x20:
                                                # Constants.app_encoding
                                                buffer += self._serial[j].read(
                                                    1).decode()
                                            else:
                                                # Constants.app_encoding
                                                buffer += self._serial[j].read(
                                                    1).hex()
                                            tail = len(buffer)
                                            if head != tail:
                                                break
                                            if (format == 7 and tail >= samples + 9) or (format == 8 and tail >= samples + 5):
                                                break

                                    if head != tail:
                                        if (tail - head) >= size:
                                            if i == -1:
                                                # get overtone indicator byte on first sample only
                                                overtone = int(
                                                    buffer[head:head+2], 16)
                                                head += 2

                                            i += 1
                                            val = float(
                                                int(buffer[head:head+size], 16))
                                            head += size
                                            size = 1  # deltas are just 4-bits

                                            # end loop once all samples are populated (ignore temp for cal)
                                            if i >= samples:
                                                self._serial[j].reset_input_buffer(
                                                )
                                                break

                                            if not i == 0:
                                                val = val if val < 8 else val - 16  # wrap 4-bit val to negative
                                                # compression ratio (MUST MATCH FW)
                                                val *= 4
                                                val += lastVal
                                            lastVal = val  # store off ADC counts for next delta
                                            data_mag[i] = self._convertADCtoMagnitude(
                                                val)

                                            # Update progress every 1% samples to focus on speed
                                            if i % int(self.maxval / 100) == 0:
                                                self._parser.add6([self._flag, self._flag2, self._flag2, int(
                                                    (samples*j + i)/len(self._serial)), False, False])
                                                # bar.update(i)

                            # bar.finish()

                            # primary port already open, indicate idle (between ports)
                            if j > 0:
                                self._serial[0].write(
                                    "MULTI INIT 0\n".encode())

                            Log.i(TAG, "Signals acquired successfully\n")

                        # specify handlers for different exceptions
                        except ValueError:
                            Log.w(TAG, "WARNING: ValueError during calibration!")
                            Log.w(
                                TAG, "Please, repeat the calibration after disconnecting/reconnecting device!")
                            self._flag = 1
                            # Log.w(TAG, "Warning: ValueError during calibration!"))
                        except RuntimeError:
                            Log.w(
                                TAG, "WARNING: Unrecognized serial response format from device!")
                            Log.w(
                                TAG, "Please, make sure the device and software are both up-to-date!")
                            self._flag = 1
                        except:
                            # raise
                            Log.w(TAG, "WARNING: Exception during calibration!")
                            Log.w(
                                TAG, "Please, repeat the calibration after disconnecting/reconnecting device!")
                            self._flag = 1
                            # Log.w(TAG, "Warning (ValueError): convert Raw to float failed")

                        if False:  # Set True to enable debug code to read a custom file as device input for calibration
                            # update path to CAL file accordingly
                            path = "config/6628700/Calibration_5MHz.txt"
                            fake_data = loadtxt(path)
                            readFREQ = fake_data[:, 0]
                            data_mag = fake_data[:, 1]
                            Log.d(
                                "DEBUG ONLY: Parsing CAL data from '{}'\n".format(path))

                        # CALLS baseline_correction method
                        if data_ph is None:
                            data_mag_baseline = self.baseline_correction(
                                readFREQ, data_mag)
                            multi_mag_baseline.append(data_mag_baseline)
                        else:
                            (data_mag_baseline, data_ph_baseline) = self.baseline_correction(
                                readFREQ, data_mag, data_ph)
                            multi_mag_baseline.append(data_mag_baseline)
                            multi_ph_baseline.append(data_ph_baseline)

                        # STOPS acquiring data
                        if k == 0:
                            if j == len(self._serial) - 1:
                                self.stop()
                            break
                    #### END SWEEPS LOOP ####

                    #### STORING DATA TO FILE ###
                    # CHECKS QCM Sensor type for saving calibration
                    if self._QCStype_int == 0:
                        distance = Constants.dist5
                        path = Constants.cvs_peakfrequencies_path
                        path_calib = Constants.csv_calibration_path
                        filename_calib = Constants.csv_calibration_filename  #
                    elif self._QCStype_int == 1:
                        distance = Constants.dist10
                        path = Constants.cvs_peakfrequencies_path
                        path_calib = Constants.csv_calibration_path10
                        filename_calib = Constants.csv_calibration_filename10  #

                    # POPULATE [ACTIVE]
                    k = j
                    if len(self._serial) > 1:
                        k += 1
                    else:
                        k = self._pid
                    path = FileStorage.DEV_populate_path(path, k)
                    path_calib = FileStorage.DEV_populate_path(path_calib, k)

                    try:
                        if os.path.exists(path):
                            # guarantee stale calibration result if this peak detection fails
                            os.remove(path)
                    except Exception as e:
                        Log.e("Failed to delete existing calibration file.")
                        Log.d("ERROR msg:", str(e))

                    # CHECKS the exceptions
                    if self._flag == 0:
                        Log.i(TAG, "Peak Detection Process Started")
                        Log.i(TAG, "Finding peaks in acquired signals...")
                        try:
                            # CALLS FindPeak method
                            if data_ph is None:
                                (max_freq_mag, max_value_mag, left_bounds, right_bounds, max_baselines) = self.FindPeak(
                                    readFREQ, data_mag_baseline, dist=distance)
                            else:
                                (max_freq_mag, max_value_mag, max_freq_phase, max_value_phase, left_bounds, right_bounds) = self.FindPeak(
                                    readFREQ, data_mag_baseline, data_ph_baseline, dist=distance)

                            Log.i(TAG, "{} peaks were found at frequencies: {} Hz\n".format(
                                len(max_freq_mag), max_freq_mag))
                            if (
                                (len(max_freq_mag) == 2 and stopFreq < 25e6 and fStep < 500) or
                                (len(max_freq_mag) == 5 and (max_freq_mag[0] > 3e+06 and max_freq_mag[0] < 6e+06)) or
                                (len(max_freq_mag) == 3 and (
                                    max_freq_mag[0] > 9e+06 and max_freq_mag[0] < 11e+06))
                            ):
                                # SAVES independently of the state of the export box
                                Log.i(TAG, "Saving data in file...")

                                # Widen left bounds since FW v2.3x has dynamic scanning now
                                for x in range(len(left_bounds)):
                                    # offset (in Hz) between calibration and measurement modes (due to step_size differences)
                                    left_bounds[x] += 5000

                                # Narrow left and right bounds for MULTIPLEX system performance to avoid overscanning when peaks are too close
                                # NOTE: This performance modification only affects the calibration of MULTI systems
                                if len(self._serial) > 1:
                                    for x in range(len(left_bounds)):
                                        left_bounds[x] /= 3
                                    for x in range(len(right_bounds)):
                                        right_bounds[x] /= 3

                                if data_ph is None:
                                    FileStorage.TXT_sweeps_save(
                                        j+1, filename_calib, Constants.csv_calibration_export_path, readFREQ, data_mag)
                                    np.savetxt(path, np.column_stack(
                                        [max_freq_mag, max_freq_mag, left_bounds, right_bounds, max_baselines]))
                                else:
                                    FileStorage.TXT_sweeps_save(
                                        j+1, filename_calib, Constants.csv_calibration_export_path, readFREQ, data_mag, data_ph)
                                    np.savetxt(path, np.column_stack(
                                        [max_freq_mag, max_freq_phase, left_bounds, right_bounds]))

                                Log.i(TAG, "Peak frequencies for {} saved in: {}".format(
                                    self._QCStype, path))
                                Log.i(TAG, "Calibration for {} saved in: {}".format(
                                    self._QCStype, path_calib))
                            else:
                                Log.w(
                                    TAG, "WARNING: Error during peak detection, incompatible peaks number or frequencies!")
                                Log.w(TAG, "Please, repeat Initialize again!")
                                self._flag2 = 1
                        except Exception as e:
                            Log.e(
                                TAG, "WARNING: Error during peak detection, exception finding peaks number or frequencies!")
                            Log.e(TAG, "Please, repeat Initialize again!")
                            self._flag2 = 1

                    temp_fail = False
                    if self._flag == 0 and self._flag2 == 0:
                        Log.i(TAG, 'Initialize success for baseline correction!\n')

                        if j == 0:
                            # Read and show the TEC temp check from the device
                            Log.d(TAG, "Performing temperature check...")
                            for i in range(3):
                                log_at_level = Log.d if i == 0 else Log.w
                                log_at_level(f"Attempt #{i+1}...")

                                self._serial[j].write("temp check {}\n".format(
                                    str(int(max_freq_mag[1]))).encode())
                                timeoutAt = time() + 3
                                temp_reply = ""
                                lines_in_reply = 7
                                # timeout needed if old FW
                                while temp_reply.count('\n') < lines_in_reply and time() < timeoutAt:
                                    # timeout needed if old FW:
                                    while self._serial[j].in_waiting == 0 and time() < timeoutAt:
                                        pass
                                    temp_reply += self._serial[j].read(
                                        self._serial[j].in_waiting).decode()

                                if time() < timeoutAt:
                                    temp_reply = temp_reply.strip().split('\n')
                                    Log.d(TAG, temp_reply[-3])  # print result
                                    Log.d(TAG, temp_reply[-2])  # print result
                                    Log.d(TAG, temp_reply[-1])  # print result
                                    if 'nan' in temp_reply[-2]:
                                        self._flag2 = 1
                                        Log.e(
                                            TAG, "WARNING: Temperature sensor not working. Please verify hardware contacts.")
                                        Log.e(
                                            TAG, "Please, repeat Initialize again!")
                                    elif not "PASS" in temp_reply[-1]:
                                        self._flag2 = 1
                                        Log.e(
                                            TAG, "ERROR: Temperature check failed. Please verify hardware contacts.")
                                        Log.e(
                                            TAG, "Please, repeat Initialize again!")
                                    else:
                                        self._flag2 = 0
                                        plurality = '' if i == 0 else 's'
                                        Log.i(
                                            f"Temperature Check: PASS ({i+1} attempt{plurality})")
                                        break  # do not try check again
                                else:
                                    # not a cal failure, fw might not support cmd, but if an earlier attempt failed, it failed
                                    Log.w(TAG, "CHECK RESULT: TIMEOUT")
                                    Log.w(
                                        TAG, "WARNING: Temperature check timeout. Skipped verification of hardware contacts.")
                                continue  # try check again, up to 3 times

                            # if check pass,then  self._flag2 = 0
                            if not self._flag2 == 0:
                                temp_fail = True
                                # temp check failed 3 times in a row, something is definitely wrong
                                Log.e("Temperature Check: FAILED (3x in a row)")
                                try:
                                    Log.d(
                                        "Removing calibration file due to failed temp check. Please try again.")
                                    if os.path.exists(path):
                                        # guarantee stale calibration result
                                        os.remove(path)
                                except Exception as e:
                                    Log.e(
                                        "Failed to delete existing calibration file.")
                                    Log.d("ERROR msg:", str(e))
                        else:
                            Log.d("Skipping temperature check on secondary device.")

                    multi_flag.append(self._flag)
                    multi_flag2.append(self._flag2)

                    if j == 0:  # base device, not a PID 2-4 device
                        if self._flag == 0 and self._flag2 == 0:
                            # successful cal
                            self._serial[j].write(
                                b"msgbox pass:initialize pass;press start then apply drop\n")
                        elif temp_fail:
                            # temp sensor fail
                            self._serial[j].write(
                                b"msgbox error:initialize error;temperature sensor fault\n")
                        else:
                            # peak detect failed
                            self._serial[j].write(
                                b"msgbox fail:initialize failed;check sensor contact\n")

            else:
                # port not available
                Log.w(TAG, "WARNING: Cannot connect! Serial port is not available.")
                Log.w(TAG, "Please, repeat Initialize again!")

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
            for j in range(len(self._serial)):
                try:
                    # ADDS new serial data to internal queue
                    self._parser.add0([j, Constants.calibration_readFREQ])
                    if j < len(multi_mag_baseline):
                        self._parser.add1(
                            [j, np.clip(multi_mag_baseline[j], 0, None)])
                    if j < len(multi_ph_baseline) and not data_ph is None:
                        self._parser.add2([j, multi_ph_baseline[j]])
                    #### END CALIBRATION ####
                except Exception as e:
                    Log.e("Error showing cal results: " + str(e))

                if self._serial[j].is_open:
                    # Flush serial buffers first!
                    while not self._serial[j].in_waiting == 0:
                        self._serial[j].reset_input_buffer()
                        self._serial[j].reset_output_buffer()

                #### CLOSES serial port ####
                self._serial[j].close()

            # ADDS error flags to internal queue
            self._parser.add5([0, any(multi_flag), any(multi_flag2), None])

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
    # Gets a list of the common serial baud rates, in bps (only 115200 used)
    ###########################################################################

    @staticmethod
    def get_speeds():
        # :return: List of the common baud rates, in bps :rtype: str list.
        return [str(v) for v in ['@5MHz_QCM', '@10MHz_QCM']]

    ###########################################################################
    # Checks if the serial port is currently connected
    ###########################################################################

    def _is_port_available(self, port):
        """
        :param port: Port name to be verified.
        :return: True if the port is connected to the host :rtype: bool.
        """
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
