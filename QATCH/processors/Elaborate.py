from QATCH.core.constants import Constants
from QATCH.core.ringBuffer import RingBuffer
from QATCH.common.fileStorage import FileStorage
from QATCH.common.logger import Logger as Log
import numpy as np
# from scipy.interpolate import UnivariateSpline # unused
import multiprocessing
from time import time
import logging
from logging.handlers import QueueHandler
import sys
import os

TAG = "[Elaborate]"

###############################################################################
# Elaborate on the raw data gathered from the SerialProcess in parallel timing
###############################################################################


class ElaborateProcess(multiprocessing.Process):

    def __init__(self, queue_log, parser_process, queue_in, queue_out, export, overtone_name, reconstruct, driedValue, appliedValue):
        """
        :param parser_process: Reference to a ParserProcess instance.
        :type parser_process: ParserProcess.
        """
        self._queueLog = queue_log  # Log.create()

        multiprocessing.Process.__init__(self)
        self._exit = multiprocessing.Event()
        self._done = multiprocessing.Event()
        self._parser = parser_process
        self._queue_in = queue_in
        self._queue_out = queue_out
        self._export = export
        self._overtone_name = overtone_name
        self._reconstruct = reconstruct
        self._driedValue = driedValue
        self._appliedValue = appliedValue

        self._count = 0  # sweeps counter
        self.sensorDriedTime = 0.0
        self.dropAppliedTime = 0.0

        self._time_start = time()
        self._last_parser_add = time()
        self._environment = Constants.avg_out
        self._frequency_buffer = RingBuffer(self._environment)
        self._dissipation_buffer = RingBuffer(self._environment)
        self._temperature_buffer = RingBuffer(self._environment)

        self._startFreq = 0
        self._stopFreq = 0
        self._startFreq_up = 0
        self._stopFreq_up = 0
        self._startFreq_down = 0
        self._stopFreq_down = 0

        self._minFREQ = Constants.calibration_frequency_stop
        self._maxFREQ = Constants.calibration_frequency_start
        self._minFREQ_up = Constants.calibration_frequency_stop
        self._maxFREQ_up = Constants.calibration_frequency_start
        self._minFREQ_down = Constants.calibration_frequency_stop
        self._maxFREQ_down = Constants.calibration_frequency_start

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

            while not self._exit.is_set():
                # Log.d"waiting for data!")
                while self._queue_in.empty() and not self._exit.is_set():
                    pass  # wait for queue to fill (or exit)

                # Log.d"got data!")
                if not self._queue_in.empty():
                    in_q = self._queue_in.get_nowait()

                    # Log.d(in_q)
                    # decompose in-queue into elaborate params
                    k = in_q[0]
                    sequence = in_q[1]
                    timestamp = in_q[2]
                    peak_mag = in_q[3]
                    peak_freq = in_q[4]
                    left = in_q[5]
                    right = in_q[6]
                    data_temp = in_q[7]
                    baseline = in_q[8]
                    overtone = in_q[9]

                    # call elaborate to process in-queued data
                    self.elaborate(k,
                                   sequence,
                                   timestamp,
                                   peak_mag,
                                   peak_freq,
                                   left,
                                   right,
                                   data_temp,
                                   baseline,
                                   overtone)

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
            # Log.d("ElaborateProcess stopping...")
            FileStorage.CSVflush_all()
            Log.d(" ElaborateProcess stopped.")

            # gracefully end subprocess
            self._done.set()

    def is_running(self):
        return not self._done.is_set()

    def _convertADCtoMagnitude(self, adc):
        return (adc * Constants.ADCtoVolt - Constants.VCP) / 0.029

    def _convertMagnitudeToADC(self, val):
        return (val * 0.029 + Constants.VCP) / Constants.ADCtoVolt

    @staticmethod
    def gaussian(x, alpha, mu, sigma):
        return alpha * np.exp(-np.power(x - mu, 2.) / (2 * (sigma * sigma)))

    @staticmethod
    def build_curve(x, alpha, mu, sigma_left, sigma_right):
        return np.where(
            x < mu,
            ElaborateProcess.gaussian(x, alpha, mu, sigma_left),
            ElaborateProcess.gaussian(x, alpha, mu, sigma_right)
        )

    ###########################################################################
    # Savitzky-Golay (Smoothing/Denoising Filter)
    ###########################################################################
    def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
        """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,) the values of the time history of the signal.
        window_size : int the length of the window. Must be an odd integer number.
        order : int the order of the polynomial used in the filtering.
                Must be less then `window_size` - 1.
        deriv: int the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N) the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
           Data by Simplified Least Squares Procedures. Analytical
           Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
           W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
           Cambridge University Press ISBN-13: 9780521880688
        """
        import numpy as np
        from math import factorial
        try:
            window_size = np.abs(int(window_size))
            order = np.abs(int(order))
        except ValueError as msg:
            raise ValueError(
                "WARNING: window size and order have to be of type int!")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError(
                "WARNING: window size must be a positive odd number!")
        if window_size < order + 2:
            raise TypeError(
                "WARNING: window size is too small for the polynomials order!")
        order_range = range(order+1)
        half_window = (window_size - 1) // 2
        # precompute coefficients
        b = np.asmatrix([[k**i for i in order_range]
                        for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with values taken from the signal itself
        firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')

    ###########################################################################
    # Processes incoming data and calculates outcoming data
    ###########################################################################

    def elaborate(self, k, sequence, w_time, peak_mag, peak_freq, left, right, temperature, baseline_coeffs, overtone):

        if overtone in (0, 255):
            if self._startFreq == 0 and self._stopFreq == 0:
                self._startFreq = peak_freq - Constants.max_drift_l_hz
                self._stopFreq = peak_freq + Constants.max_drift_r_hz
        elif overtone == 1:
            if self._startFreq_up == 0 and self._stopFreq_up == 0:
                self._startFreq_up = peak_freq - Constants.max_drift_l_hz
                self._stopFreq_up = peak_freq + Constants.max_drift_r_hz
        elif overtone == 2:
            if self._startFreq_down == 0 and self._stopFreq_down == 0:
                self._startFreq_down = peak_freq - Constants.max_drift_l_hz
                self._stopFreq_down = peak_freq + Constants.max_drift_r_hz

        # Get last temperature from buffer if none was provided this sweep
        if temperature == None:
            temperature = self._temperature_buffer.get_newest()

        # Pass up TEC status for temp branch
        # overloaded meaning for left/right
        tec_op = left
        t_amb = right
        self._parser.add2([0, tec_op])

        # sweep counter
        self._k = sequence
        # frequency range, samples number
        if self._k == 0:
            self._minFREQ = self._maxFREQ = peak_freq
            self._readFREQ = np.arange(
                self._minFREQ, self._maxFREQ + Constants.argument_default_samples - 1, 1)

        peak_dB = self._convertADCtoMagnitude(peak_mag)
        dissipation = 10.0 ** (-peak_dB / 20)

        self._mode = self.get_mode(peak_freq)
        if self._mode == 1:
            dissipation *= Constants.dissipation_factor_1st_mode
        if self._mode == 3:
            dissipation *= Constants.dissipation_factor_3rd_mode
        if self._mode == 5:
            dissipation *= Constants.dissipation_factor_5th_mode

        # Display reconstructed curve in top-left
        if (overtone in (0, 255) and self._reconstruct) or self._export:
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
                _min, _max, Constants.argument_default_samples - 1)
            baseline_offset = min(self._convertMagnitudeToADC(
                np.polyval(baseline_coeffs, peak_freq)), peak_mag)
            mag_result_fit = ElaborateProcess.build_curve(
                self._readFREQ, peak_mag - baseline_offset, peak_freq, left, right)
            self._readFREQ[np.argmax(mag_result_fit)] = peak_freq
            mag_result_fit[np.argmax(mag_result_fit)
                           ] = peak_mag - baseline_offset
            mag_result_fit = self._convertADCtoMagnitude(mag_result_fit)
            mag_result_fit -= self._convertADCtoMagnitude(0)  # zero offset
            filtered_mag = mag_result_fit

        phase = None
        # define error vars (with no error indicated)
        self._err1 = self._err2 = self._err3 = self._err4 = 0

        if overtone in (0, 255):
            if peak_freq <= self._startFreq:
                self._err1 = 1
            if peak_freq >= self._stopFreq:
                self._err2 = 1
        elif overtone == 1:
            if peak_freq <= self._startFreq_up:
                self._err1 = 2
            if peak_freq >= self._stopFreq_up:
                self._err2 = 2
        elif overtone == 2:
            if peak_freq <= self._startFreq_down:
                self._err1 = 3
            if peak_freq >= self._stopFreq_down:
                self._err2 = 3

        #######################################################
        ##############
        w_time /= 1e4
        ##############

        # add data to ElaborateProcess out queue (back to SerialProcess) for downstream LoggerProcess
        self._queue_out.put(
            [w_time,
             temperature,
             dissipation,
             peak_freq,
             left,
             right,
             self._err1,
             self._err2,
             self._err3,
             self._err4]
        )

# TODO AJR: Consider moving file writing operations to a separate thread
        if self._export:
            # Storing acquired sweeps
            filename = "{}_{}_{}".format(
                Constants.csv_sweeps_filename, self._overtone_name, self._count)
            path = "{}_{}".format(
                Constants.csv_sweeps_export_path, self._overtone_name)
            path = FileStorage.DEV_populate_path(path, 0)
            if not phase is None:
                FileStorage.TXT_sweeps_save(
                    0, filename, path, self._readFREQ, filtered_mag, phase, appendNameToPath=False)
            else:
                FileStorage.TXT_sweeps_save(
                    0, filename, path, self._readFREQ, filtered_mag, appendNameToPath=False)
            self._count += 1

        # Read the shared values
        with self._driedValue.get_lock():
            if self.sensorDriedTime != self._driedValue.value:
                self.sensorDriedTime = self._driedValue.value
                Log.d("[ElaborateProcess]",
                      f"Sensor dried time = {self.sensorDriedTime}")
        with self._appliedValue.get_lock():
            if self.dropAppliedTime != self._appliedValue.value:
                self.dropAppliedTime = self._appliedValue.value
                Log.d("[ElaborateProcess]",
                      f"Drop applied time = {self.dropAppliedTime}")

        if overtone in (0, 255):
            if not (self._err1 and self._err2):
                self._frequency_buffer.append(peak_freq)
                if not dissipation == 0:
                    self._dissipation_buffer.append(dissipation)
            self._temperature_buffer.append(temperature)

            filenameCSV = "{}_{}".format(
                Constants.csv_filename, self._overtone_name.split(' ')[0])
            write_interval = 1000 if w_time < Constants.downsample_after else Constants.downsample_file_count

            FileStorage.CSVsave(0, filenameCSV, Constants.csv_export_path, w_time, temperature,
                                peak_mag, peak_freq, dissipation, t_amb, (k % write_interval == 0))

        elif overtone == 1:
            write_interval = 1000 if w_time < Constants.downsample_after else Constants.downsample_file_count * \
                Constants.base_overtone_freq
            FileStorage.CSVsave(0, "overtone_upper", Constants.csv_export_path, w_time, temperature,
                                peak_mag, peak_freq, dissipation, t_amb, (k % write_interval < Constants.base_overtone_freq))
            return
        elif overtone == 2:
            write_interval = 1000 if w_time < Constants.downsample_after else Constants.downsample_file_count * \
                Constants.base_overtone_freq
            FileStorage.CSVsave(0, "overtone_lower", Constants.csv_export_path, w_time, temperature,
                                peak_mag, peak_freq, dissipation, t_amb, (k % write_interval < Constants.base_overtone_freq))
            return

        # update buffers at most once every 50ms (20/s)
        if self._k >= self._environment and (time() - self._last_parser_add) > 0.050:
            self._last_parser_add = time()  # update
            # FREQUENCY
            vec_app1 = self.savitzky_golay(self._frequency_buffer.get_partial(
            ), window_size=Constants.SG_window_environment, order=Constants.SG_order_environment)
            freq_range_mean = np.average(vec_app1)
            # DISSIPATION
            vec_app1d = self.savitzky_golay(self._dissipation_buffer.get_partial(
            ), window_size=Constants.SG_window_environment, order=Constants.SG_order_environment)
            diss_mean = np.average(vec_app1d)
            # TEMPERATURE
            vec_app1t = self.savitzky_golay(self._temperature_buffer.get_partial(
            ), window_size=Constants.SG_window_environment, order=Constants.SG_order_environment)
            temperature_mean = np.average(vec_app1t)

            # ADDS new frequency domain data (mag and ph) to internal queues
            if overtone in (0, 255) and self._reconstruct:
                self._parser.add0([0, self._readFREQ])
                self._parser.add1([0, filtered_mag])
                if not phase is None:
                    self._parser.add2([0, phase])

            # place upper limits on dissipation data for each mode
            if self._mode == 1 and diss_mean > Constants.max_dissipation_1st_mode:
                diss_mean = Constants.max_dissipation_1st_mode
            if self._mode == 3 and diss_mean > Constants.max_dissipation_3rd_mode:
                diss_mean = Constants.max_dissipation_3rd_mode
            if self._mode == 5 and diss_mean > Constants.max_dissipation_5th_mode:
                diss_mean = Constants.max_dissipation_5th_mode

            # throw away first few samples to allow firmware to settle on peak
            settle_samples = Constants.initial_settle_samples
            if not (self._err1 and self._err2) and k > settle_samples:
                if w_time > Constants.downsample_after and k % Constants.downsample_plot_count > 0:
                    return  # do not show this sample on plot
                # Adds new time domain data (resonance, dissipation, temp) to internal queues
                # w_time - time in seconds
                self._parser.add3([0, w_time, freq_range_mean])
                # w_time - time in seconds
                self._parser.add4([0, w_time, diss_mean])
                # w_time - time in seconds
                self._parser.add5([0, w_time, temperature_mean, t_amb])
            if k <= settle_samples:
                self._minFREQ = Constants.calibration_frequency_stop
                self._maxFREQ = Constants.calibration_frequency_start
                self._minFREQ_up = Constants.calibration_frequency_stop
                self._maxFREQ_up = Constants.calibration_frequency_start
                self._minFREQ_down = Constants.calibration_frequency_stop
                self._maxFREQ_down = Constants.calibration_frequency_start

    def get_mode(self, freq):
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

    def stop(self):
        # Signals the process to stop when the parent process stops
        self._exit.set()
