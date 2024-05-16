import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from tsmoothie.smoother import *
TAG = "[ModelData]"
try:
    from QATCH.common.logger import Logger as Log
except:
    class Log():
        @staticmethod
        def d(t, s=""):
            print(t, s)
        def i(t, s=""):
            print(t, s)
        def w(t, s=""):
            print(t, s)
        def e(t, s=""):
            print(t, s)
try:
    from QATCH.core.constants import Constants
except:
    class Constants():
        sf_min = 0.25
        sf_max = 25.0
        sf_step = 1.0
        smooth_factor_ratio = 0.75
        super_smooth_factor_ratio = sf_max
        baseline_smooth = 9
        consider_points_above_pct = 0.60
        downsample_file_count = 20
        running_headless = True
class ModelData():
    def __init__(self):
        self._plt_init = False
        if hasattr(Constants, "running_headless"):
            self._headless = Constants.running_headless
        else:
            self._headless = False
        if self._headless:
            self.min_time = 9e9
            self.max_time = 0
            self.sum_time = 0
            self.num_time = 0
            self.min_freq = 9e9
            self.max_freq = 0
            self.sum_freq = 0
            self.num_freq = 0
            self.min_diss = 9e9
            self.max_diss = 0
            self.sum_diss = 0
            self.num_diss = 0
    def normalize(self, arr, t_min = 0, t_max = 1):
        norm_arr = arr.copy()
        try:
            diff = t_max - t_min
            diff_arr = max(arr) - min(arr)
            if diff_arr == 0:
                diff_arr = 1
            norm_arr -= min(arr)
            norm_arr *= diff
            norm_arr /= diff_arr
            norm_arr += t_min
        except Exception as e:
            Log.e("ERROR:" + str(e))
        return norm_arr
    def normalize_list_of_tuples(self, arr, t_min = 0, t_max = 1):
        mags = []
        for p in arr:
            mags.append(p[1])
        if len(mags) > 0:
            mags = self.normalize(mags)
        norm_arr = []
        for i in range(len(arr)):
            norm_arr.append((arr[i][0], mags[i]))
        return norm_arr
    def point_fit(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        m_slope = abs((y2 - y1) / (x2 - x1))
        b_offset = y1 - m_slope * x1
        return (m_slope, b_offset)
    def line_fit(self, point, slope):
        x1, y1 = point
        m = slope
        b = y1 - m * x1
        return b
    def find_intersect(self, line1, line2):
        m1, b1 = line1
        m2, b2 = line2
        if m1 == m2:
            Log.w("find_intersect(): Slopes are identical, no intersection possible")
            return (0,0)
        x_intersection = (b2 - b1) / (m1 - m2)
        y_intersection = m1 * x_intersection + b1
        return (x_intersection, y_intersection)
    def compare_slopes(self, fill_fit_slope, post_fit_slope, left_point_slope, right_point_slope, last_result = 0):
        fill_error_left = abs(fill_fit_slope - left_point_slope)
        fill_error_right = abs(fill_fit_slope - right_point_slope)
        post_error_left = abs(post_fit_slope - left_point_slope)
        post_error_right = abs(post_fit_slope - right_point_slope)
        left_is_closer_to_fill = True if fill_error_left < post_error_left else False
        right_is_closer_to_post = True if post_error_right < fill_error_right else False
        if left_is_closer_to_fill and right_is_closer_to_post:
            return 0
        elif fill_error_right < post_error_right:
            return 1 if last_result >= 0 else 0
        elif post_error_left < fill_error_left:
            return -1 if last_result <= 0 else 0
        else:
            Log.w("Something happend during comparison that was unexpected.")
            return last_result
    def IdentifyPoints(self, data_path = None, times = None, freq = None, diss = None, start_at = None, sf = Constants.super_smooth_factor_ratio, i = -1, actual=[]):
        xs = ys = [0]
        poi_vals = -1
        t_quads = []
        global_index = i
        debug_plots = False
        debug_plots2 = False
        try:
            if self._headless:
                Log.i("ModelData file = {}".format(data_path))
            data_title = "{}_{}".format(int(i/2)+1, os.path.splitext(os.path.basename(data_path))[0])
            if times is None or freq is None or diss is None:
                with open(data_path, 'r') as f:
                    csv_headers = next(f)
                    if isinstance(csv_headers, bytes):
                        csv_headers = csv_headers.decode()
                    if "Ambient" in csv_headers:
                        csv_cols = (2,4,6,7)
                    else:
                        csv_cols = (2,3,5,6)
                    data  = loadtxt(f.readlines(), delimiter = ',', skiprows = 0, usecols = csv_cols)
            else:
                temp = times
                data = np.column_stack((times,temp,freq,diss))
            relative_time = data[:,0]
            resonance_frequency = data[:,2]
            dissipation = data[:,3]
            t_last = 0
            rows_to_toss = []
            for x,t in enumerate(relative_time):
                if t < t_last:
                    rows_to_toss.append(x-1)
                t_last = t
            if len(rows_to_toss) > 0:
                Log.w(f"Warning: time jump(s) observed at the following indices: {rows_to_toss}")
                relative_time = np.delete(relative_time, rows_to_toss)
                resonance_frequency = np.delete(resonance_frequency, rows_to_toss)
                dissipation = np.delete(dissipation, rows_to_toss)
                Log.w("Time jumps removed from dataset for analysis purposes (original file unchanged)")
            is_good = True
            min_time = xs[0]
            max_time = relative_time[-1]
            delta_time = max_time - min_time
            min_freq = np.argmin(resonance_frequency)
            max_freq = np.argmax(resonance_frequency)
            delta_freq = resonance_frequency[max_freq] - resonance_frequency[min_freq]
            min_diss = np.argmin(dissipation)
            max_diss = np.argmax(dissipation)
            delta_diss = dissipation[max_diss] - dissipation[min_diss]
            if delta_time < 9.0:
                Log.e(TAG, "Bad run due to 'relative_time' duration being too short.")
                is_good = False
            if delta_freq < 200:
                Log.e(TAG, "Bad run due to 'resonance_frequency' delta being too small.")
                is_good = False
            if delta_diss < 1e-5:
                Log.e(TAG, "Bad run due to 'dissipation' delta being too small.")
                is_good = False
            if self._headless:
                self.sum_time += delta_time
                self.num_time += 1
                self.sum_freq += delta_freq
                self.num_freq += 1
                self.sum_diss += delta_diss
                self.num_diss += 1
                if delta_time < self.min_time:
                    self.min_time = delta_time
                if delta_time > self.max_time:
                    self.max_time = delta_time
                if delta_freq < self.min_freq:
                    self.min_freq = delta_freq
                if delta_freq > self.max_freq:
                    self.max_freq = delta_freq
                if delta_diss < self.min_diss:
                    self.min_diss = delta_diss
                if delta_diss > self.max_diss:
                    self.max_diss = delta_diss
            if not is_good:
                poi_vals = -1
                return poi_vals
            poi_vals = -1
            xs = relative_time
            ys = dissipation
            total_runtime = xs[-1]
            smooth_factor = total_runtime * sf
            smooth_factor = max(3, int(smooth_factor) + (int(smooth_factor + 1) % 2))
            if self._headless:
                Log.i(TAG, f"Total run time: {total_runtime} secs")
                Log.d(TAG, f"Smoothing: {smooth_factor}")
                Log.d(TAG, f"SF Ratio:  {smooth_factor / total_runtime}")
                Log.d(TAG, f"Applying smooth factor for first 90s ONLY.")
            t_first_90_split = len(xs) if total_runtime <= 90 else next(x for x,t in enumerate(xs) if t > 90)
            extend_data = True if total_runtime > 90 else False
            extend_smf = int(smooth_factor / 20)
            extend_smf += (int(extend_smf + 1) % 2)
            if extend_smf == 1: extend_smf = 3
            if self._headless:
                Log.d(TAG, f"SF Extend: {extend_smf}")
            if extend_data:
                upper_bound = min(len(xs)-1, t_first_90_split+20)
                break_point = (upper_bound - t_first_90_split)
                lower_bound = max(0, t_first_90_split-break_point)
                before = xs[t_first_90_split] - xs[lower_bound]
                after = xs[upper_bound] - xs[t_first_90_split]
                if after > 5 * before:
                    Log.d("Yes, this data is downsampled.")
                else:
                    Log.w("This data does not appear to be downsampled. Treating entire run as regular data, downsampling all of it.")
                    t_first_90_split = -1
                    extend_data = True
            if extend_data:
                xs_extend = xs.copy()
                ys_extend = ys.copy()
                xs = np.concatenate((xs[0:t_first_90_split:20], xs[t_first_90_split:]))
                ys = np.concatenate((ys[0:t_first_90_split:20], ys[t_first_90_split:]))
            num_datapoints = len(xs)
            smooth_points = 300
            dps = num_datapoints / total_runtime
            smooth_factor = total_runtime * dps / smooth_points
            smooth_factor = max(3, int(smooth_factor) + (int(smooth_factor + 1) % 2))
            extend_smf = smooth_factor
            if self._headless:
                Log.d(f"Global smooth factor: {smooth_factor}")
            ys_fit = savgol_filter(ys, extend_smf if extend_data else smooth_factor, 1)
            if total_runtime < 3:
                Log.e("ERROR: Data run must be at least 3 seconds in total runtime to analyze.")
                return -1
            t_0p5 = next(x+0 for x,t in enumerate(xs) if t > 0.5)
            t_1p0 = next(x+1 for x,t in enumerate(xs) if t > 2.0)
            avg = np.average(resonance_frequency[t_0p5:t_1p0])
            ys = ys * avg / 2
            ys_fit = ys_fit * avg / 2
            ys = ys - np.amin(ys_fit)
            ys_fit = ys_fit - np.amin(ys_fit)
            ys_freq = avg - resonance_frequency
            if extend_data:
                ys_freq_extend = ys_freq.copy()
                ys_freq = np.concatenate((ys_freq[0:t_first_90_split:20], ys_freq[t_first_90_split:]))
            baseline = np.average(dissipation[t_0p5:t_1p0])
            diff_factor = 2.0
            ys_diff = ys_freq - diff_factor*ys
            if self._headless:
                Log.d(f"Difference factor: {diff_factor:1.1f}x")
            ys_freq_fit = savgol_filter(ys_freq, extend_smf if extend_data else smooth_factor, 1)
            ys_diff_fit = savgol_filter(ys_diff, extend_smf if extend_data else smooth_factor, 1)
            if extend_data:
                ys_ext_fit = savgol_filter(ys_extend, smooth_factor, 1)
                ys_extend = ys_extend * avg / 2
                ys_ext_fit = ys_ext_fit * avg / 2
                ys_extend = ys_extend - np.amin(ys_ext_fit)
                ys_ext_fit = ys_ext_fit - np.amin(ys_ext_fit)
                ys_diff_extend = ys_freq_extend - diff_factor*ys_extend
                ys_freq_ext_fit = savgol_filter(ys_freq_extend, smooth_factor, 1)
                ys_diff_ext_fit = savgol_filter(ys_diff_extend, smooth_factor, 1)
            super_smoother = ConvolutionSmoother(window_len=10*smooth_factor, window_type='ones')
            super_smoother.smooth(ys)
            super_smooth_diss_1st = self.normalize(savgol_filter(super_smoother.smooth_data[0], 11, 1, 1), 0, 100)
            super_smooth_diss_1st = super_smoother.smooth(super_smooth_diss_1st).smooth_data[0]
            super_smooth_diss_2nd = savgol_filter(super_smooth_diss_1st, 11, 1, 1)
            super_smooth_diss_2nd = super_smoother.smooth(super_smooth_diss_2nd).smooth_data[0]
            super_smooth_freq_1st = self.normalize(savgol_filter(super_smoother.smooth(ys_freq).smooth_data[0], 11, 1, 1), 0, 100)
            super_smooth_freq_1st = super_smoother.smooth(super_smooth_freq_1st).smooth_data[0]
            super_smooth_freq_2nd = savgol_filter(super_smooth_freq_1st, 11, 1, 1)
            super_smooth_freq_2nd = super_smoother.smooth(super_smooth_freq_2nd).smooth_data[0]
            super_smooth_diff_1st = self.normalize(savgol_filter(super_smoother.smooth(ys_diff).smooth_data[0], 11, 1, 1), 0, 100)
            super_smooth_diff_1st = super_smoother.smooth(super_smooth_diff_1st).smooth_data[0]
            super_smooth_diff_2nd = savgol_filter(super_smooth_diff_1st, 11, 1, 1)
            super_smooth_diff_2nd = super_smoother.smooth(super_smooth_diff_2nd).smooth_data[0]
            min_idx_1st = argrelextrema(super_smooth_diss_1st, np.less)[0]
            max_idx_1st = argrelextrema(super_smooth_diss_1st, np.greater)[0]
            min_idx_2nd = argrelextrema(super_smooth_diss_2nd, np.less)[0]
            max_idx_2nd = argrelextrema(super_smooth_diss_2nd, np.greater)[0]
            avg_diss_1 = np.average(super_smooth_diss_1st) * 1.2
            if super_smooth_diss_1st[0] == 100.0:
                max_idx_1st = np.insert(max_idx_1st, 0, 0)
            if super_smooth_diss_2nd[0] == np.amax(super_smooth_diss_2nd):
                max_idx_2nd = np.insert(max_idx_2nd, 0, 0)
            try:
                max_val_1st = next(y for y in max_idx_1st if super_smooth_diss_1st[y] > avg_diss_1)
                min_val_2nd = next(max_idx_2nd[max(0,x-1)] for x,y in enumerate(max_idx_2nd) if y > max_val_1st)
                max_val_2nd = next(y for y in min_idx_2nd if y > max_val_1st)
                half_diss_1 = avg_diss_1 + (super_smooth_diss_1st[max_val_1st] - avg_diss_1) * 0.4
                min_val_1st = next(y for y in min_idx_1st if y > max_val_2nd and super_smooth_diss_1st[y] < half_diss_1)
            except Exception as e:
                max_val_1st = 0
                min_val_2nd = 0
                max_val_2nd = 0
                half_diss_1 = avg_diss_1
                min_val_1st = 0
                Log.e("ERROR: Something is off!", str(e))
            initial_time_delta1 = xs[max_val_2nd] - xs[min_val_2nd]
            initial_time_delta2 = xs[min_val_1st] - xs[max_val_1st]
            avg_start = int((max_val_1st + min_val_2nd) / 2)
            avg_delta = (initial_time_delta1 + initial_time_delta2) / 2
            if avg_delta <= 0:
                avg_delta = (xs[-1] - xs[avg_start]) / 20
            expected_end1 = min(xs[avg_start] + 8*avg_delta, xs[-1])
            if expected_end1 == xs[-1]:
                Log.w("This run looks like an incomplete fill based on the initial fill duration!")
                expected_end1 = xs[avg_start] + (xs[-1] - xs[avg_start]) / 2
            expected_end2 = min(xs[avg_start] + 35*avg_delta, xs[-1])
            if self._headless:
                Log.d(f"Plot Title: {data_title}")
                Log.d(f"avg_start = {xs[avg_start]:2.1f}")
                Log.d(f"avg_delta = {avg_delta:2.1f}")
                Log.d(f"expected_end1 = {expected_end1:2.1f}")
                Log.d(f"expected_end2 = {expected_end2:2.1f}")
            idx_end1 = next(x for x,y in enumerate(xs) if y >= expected_end1)
            idx_end2 = next(x for x,y in enumerate(xs) if y >= expected_end2)
            try:
                avg_stop1 = idx_end1 + np.argmax(np.abs(super_smooth_diff_2nd[idx_end1:idx_end2]))
            except:
                avg_stop1 = len(xs) - 1
            try:
                avg_stop2 = np.argmax(ys_freq_fit)
                if avg_stop2 in [0, len(xs)-1]:
                    avg_stop2 = avg_stop1
            except:
                avg_stop2 = len(xs) - 1
            try:
                avg_stop3 = np.argmax(ys_diff_fit)
                if avg_stop3 in [0, len(xs)-1]:
                    avg_stop3 = avg_stop1
            except:
                avg_stop3 = len(xs) - 1
            avg_stop = int((avg_stop1 + avg_stop2 + avg_stop3) / 3)
            fill_time = xs[avg_stop] - xs[avg_start]
            smooth_factor = fill_time * dps / smooth_points
            smooth_factor = max(3, int(smooth_factor) + (int(smooth_factor + 1) % 2))
            extend_smf = smooth_factor
            if self._headless:
                Log.d(f"Optimal smoothing: {smooth_factor}")
            ys_fit = savgol_filter(ys, extend_smf if extend_data else smooth_factor, 1)
            ys_freq_fit = savgol_filter(ys_freq, extend_smf if extend_data else smooth_factor, 1)
            ys_diff_fit = savgol_filter(ys_diff, extend_smf if extend_data else smooth_factor, 1)
            ys_diss_1st = savgol_filter(ys_fit, extend_smf if extend_data else smooth_factor, 1, 1)
            tsf = extend_smf if extend_data else smooth_factor
            deriv_smoother = ConvolutionSmoother(window_len=tsf, window_type='ones')
            deriv_smoother.smooth(ys_diss_1st)
            ys_diss_1st = np.abs(deriv_smoother.smooth_data[0])
            ys_diss_2nd = savgol_filter(ys_diss_1st, extend_smf if extend_data else smooth_factor, 1, 1)
            deriv_smoother.smooth(ys_diss_2nd)
            ys_diss_2nd = deriv_smoother.smooth_data[0]
            ys_diss_2nd_abs = np.abs(ys_diss_2nd)
            ys_diss_diff_avg = np.average(ys_diss_1st)
            ys_diss_diff_offset = ys_diss_1st - ys_diss_diff_avg
            zeros3 = np.where(np.diff(np.sign(ys_diss_diff_offset)))[0]
            ys_freq_1st = savgol_filter(ys_freq_fit, extend_smf if extend_data else smooth_factor, 1, 1)
            deriv_smoother.smooth(ys_freq_1st)
            ys_freq_1st = np.abs(deriv_smoother.smooth_data[0])
            ys_freq_2nd = savgol_filter(ys_freq_1st, extend_smf if extend_data else smooth_factor, 1, 1)
            deriv_smoother.smooth(ys_freq_2nd)
            ys_freq_2nd = np.abs(deriv_smoother.smooth_data[0])
            ys_diff = ys_freq - diff_factor*ys
            ys_diff_fit = savgol_filter(ys_diff, extend_smf if extend_data else smooth_factor, 1)
            ys_diff_1st = savgol_filter(ys_diff_fit, extend_smf if extend_data else smooth_factor, 1, 1)
            deriv_smoother.smooth(ys_diff_1st)
            ys_diff_1st = np.abs(deriv_smoother.smooth_data[0])
            ys_diff_2nd = savgol_filter(ys_diff_1st, extend_smf if extend_data else smooth_factor, 1, 1)
            deriv_smoother.smooth(ys_diff_2nd)
            ys_diff_2nd = np.abs(deriv_smoother.smooth_data[0])
            tsf = extend_smf if extend_data else smooth_factor
            data_smoother = ConvolutionSmoother(window_len=tsf, window_type='ones')
            data_smoother.smooth(self.normalize(ys))
            ys_smooth = data_smoother.smooth_data[0]
            low1, up1 = data_smoother.get_intervals('sigma_interval', n_sigma=2)
            data_smoother.smooth(self.normalize(ys_freq))
            ys_freq_smooth = data_smoother.smooth_data[0]
            low2, up2 = data_smoother.get_intervals('sigma_interval', n_sigma=1)
            data_smoother.smooth(self.normalize(ys_diff))
            ys_diff_smooth = data_smoother.smooth_data[0]
            low3, up3 = data_smoother.get_intervals('sigma_interval', n_sigma=1)
            if extend_data:
                data_smoother.smooth(self.normalize(ys_diff_extend))
                ys_diff_smooth_fill = data_smoother.smooth_data[0]
                low3_fill, up3_fill = data_smoother.get_intervals('sigma_interval', n_sigma=1)
            else:
                ys_diff_smooth_fill = ys_diff_smooth
                low3_fill, up3_fill = low3, up3
            adjusted_diss_1st = np.concatenate((np.zeros(t_1p0), ys_diss_1st[t_1p0:]))
            baseline_diss_1st = savgol_filter(adjusted_diss_1st, Constants.baseline_smooth, 1)
            adjusted_diss_1st = np.abs(adjusted_diss_1st - baseline_diss_1st)
            adjusted_freq_1st = np.concatenate((np.zeros(t_1p0), ys_freq_1st[t_1p0:]))
            baseline_freq_1st = savgol_filter(adjusted_freq_1st, Constants.baseline_smooth, 1)
            adjusted_freq_1st = np.abs(adjusted_freq_1st - baseline_freq_1st)
            adjusted_diff_1st = np.concatenate((np.zeros(t_1p0), ys_diff_1st[t_1p0:]))
            baseline_diff_1st = savgol_filter(adjusted_diff_1st, Constants.baseline_smooth, 1)
            adjusted_diff_1st = np.abs(adjusted_diff_1st - baseline_diff_1st)
            if debug_plots:
                plt.figure()
                plt.title("Dissipation 1st deriv. (adjusted)")
                plt.plot(xs, ys_diss_1st, alpha=0.5)
                plt.plot(xs, baseline_diss_1st, alpha=0.5)
                plt.plot(xs, adjusted_diss_1st)
                plt.figure()
                plt.title("Frequency 1st deriv. (adjusted)")
                plt.plot(xs, ys_freq_1st, alpha=0.5)
                plt.plot(xs, baseline_freq_1st, alpha=0.5)
                plt.plot(xs, adjusted_freq_1st)
                plt.figure()
                plt.title("Difference 1st deriv. (adjusted)")
                plt.plot(xs, ys_diff_1st, alpha=0.5)
                plt.plot(xs, baseline_diff_1st, alpha=0.5)
                plt.plot(xs, adjusted_diff_1st)
            accumulated_1st = self.normalize(adjusted_diss_1st) + self.normalize(adjusted_freq_1st) + self.normalize(adjusted_diff_1st)
            minima_idx_1 = argrelextrema(accumulated_1st, np.less)[0]
            minima_val_1 = accumulated_1st[minima_idx_1]
            minima_dict_1 = {minima_idx_1[i]: minima_val_1[i] for i in range(len(minima_idx_1))}
            minima_sort_1 = sorted(minima_dict_1.items(), key = lambda kv:(kv[1], kv[0]))
            maxima_idx_1 = argrelextrema(accumulated_1st, np.greater)[0]
            maxima_val_1 = accumulated_1st[maxima_idx_1]
            if False:
                for i in range(min(len(minima_idx_1), len(maxima_idx_1))):
                    maxima_val_1[i] -= minima_val_1[i]
            maxima_dict_1 = {maxima_idx_1[i]: maxima_val_1[i] for i in range(len(maxima_idx_1))}
            maxima_sort_1 = sorted(maxima_dict_1.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
            if debug_plots:
                plt.figure()
                plt.title("Accumulated 1st deriv. (adjusted)")
                plt.plot(xs, adjusted_diss_1st, alpha=0.5)
                plt.plot(xs, adjusted_freq_1st, alpha=0.5)
                plt.plot(xs, adjusted_diff_1st, alpha=0.5)
                plt.plot(xs, accumulated_1st)
                plt.plot(xs[maxima_idx_1], self.normalize(maxima_val_1), color="pink")
            adjusted_diss_2nd = np.concatenate((np.zeros(t_1p0), ys_diss_2nd_abs[t_1p0:]))
            baseline_diss_2nd = savgol_filter(adjusted_diss_2nd, Constants.baseline_smooth, 1)
            adjusted_diss_2nd = np.abs(adjusted_diss_2nd - baseline_diss_2nd)
            adjusted_freq_2nd = np.concatenate((np.zeros(t_1p0), ys_freq_2nd[t_1p0:]))
            baseline_freq_2nd = savgol_filter(adjusted_freq_2nd, Constants.baseline_smooth, 1)
            adjusted_freq_2nd = np.abs(adjusted_freq_2nd - baseline_freq_2nd)
            adjusted_diff_2nd = np.concatenate((np.zeros(t_1p0), ys_diff_2nd[t_1p0:]))
            baseline_diff_2nd = savgol_filter(adjusted_diff_2nd, Constants.baseline_smooth, 1)
            adjusted_diff_2nd = np.abs(adjusted_diff_2nd - baseline_diff_2nd)
            if debug_plots:
                plt.figure()
                plt.title("Dissipation 2nd deriv. (adjusted)")
                plt.plot(xs, ys_diss_2nd_abs, alpha=0.5)
                plt.plot(xs, baseline_diss_2nd, alpha=0.5)
                plt.plot(xs, adjusted_diss_2nd)
                plt.figure()
                plt.title("Frequency 2nd deriv. (adjusted)")
                plt.plot(xs, ys_freq_2nd, alpha=0.5)
                plt.plot(xs, baseline_freq_2nd, alpha=0.5)
                plt.plot(xs, adjusted_freq_2nd)
                plt.figure()
                plt.title("Difference 2nd deriv. (adjusted)")
                plt.plot(xs, ys_diff_2nd, alpha=0.5)
                plt.plot(xs, baseline_diff_2nd, alpha=0.5)
                plt.plot(xs, adjusted_diff_2nd)
            accumulated_2nd = accumulated_1st + self.normalize(adjusted_diss_2nd) + self.normalize(adjusted_freq_2nd) + self.normalize(adjusted_diff_2nd)
            minima_idx_2 = argrelextrema(accumulated_2nd, np.less)[0]
            minima_val_2 = accumulated_2nd[minima_idx_2]
            minima_dict_2 = {minima_idx_2[i]: minima_val_2[i] for i in range(len(minima_idx_2))}
            minima_sort_2 = sorted(minima_dict_2.items(), key = lambda kv:(kv[1], kv[0]))
            maxima_idx_2 = argrelextrema(accumulated_2nd, np.greater)[0]
            maxima_val_2 = accumulated_2nd[maxima_idx_2]
            if False:
                for i in range(min(len(minima_idx_2), len(maxima_idx_2))):
                    maxima_val_2[i] -= minima_val_2[i]
            maxima_dict_2 = {maxima_idx_2[i]: maxima_val_2[i] for i in range(len(maxima_idx_2))}
            maxima_sort_2 = sorted(maxima_dict_2.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
            if debug_plots:
                plt.figure()
                plt.title("Accumulated 2nd deriv. (adjusted)")
                plt.plot(xs, adjusted_diss_2nd, alpha=0.5)
                plt.plot(xs, adjusted_freq_2nd, alpha=0.5)
                plt.plot(xs, adjusted_diff_2nd, alpha=0.5)
                plt.plot(xs, accumulated_2nd)
                plt.plot(xs[maxima_idx_2], self.normalize(maxima_val_2), color="pink")
                plt.figure()
                plt.title("Combined Accumulations")
                plt.plot(xs, accumulated_1st)
                plt.plot(xs, accumulated_2nd)
                plt.plot(xs, self.normalize(accumulated_1st + accumulated_2nd, 0, 3))
            level_smooth_factor = dps / 2
            level_smooth_factor = max(3, int(level_smooth_factor) + (int(level_smooth_factor + 1) % 2))
            level_smoother = ConvolutionSmoother(window_len=5*level_smooth_factor, window_type='ones')
            super_smooth_diss_t1 = savgol_filter(level_smoother.smooth(ys_fit).smooth_data[0], 11, 1)
            super_smooth_freq_t1 = savgol_filter(level_smoother.smooth(ys_freq_fit).smooth_data[0], 11, 1)
            super_smooth_diff_t1 = savgol_filter(level_smoother.smooth(ys_diff_fit).smooth_data[0], 11, 1)
            if debug_plots:
                plt.figure()
                plt.title("Fitting of Difference")
                plt.plot(xs, ys_fit)
                plt.plot(xs, super_smooth_diss_t1)
            if debug_plots2:
                if not self._plt_init:
                    self.fig_dbg2 = plt.figure()
                    self.plt_dbg2 = self.fig_dbg2.add_subplot(111)
                else:
                    self.plt_dbg2.cla()
                self.plt_dbg2.set_title("Deviation of Difference")
                growth = []
                deviation = ys_fit - super_smooth_diss_t1
                for dev_pt in range(len(deviation)):
                    dev_diff = deviation[dev_pt] - deviation[dev_pt-1]
                    growth.append(dev_diff)
                self.plt_dbg2.plot(xs, deviation)
                self.plt_dbg2.plot(xs, growth)
            ys_diss_diff_avg = 0
            possible_points1 = []
            this_max_point = (0,0)
            for m in maxima_dict_1.items():
                if m[1] > ys_diss_diff_avg:
                    if m[0] < t_1p0: continue
                    if m[1] > this_max_point[1]:
                        this_max_point = m
                if True:
                    if this_max_point != (0,0):
                        possible_points1.append(this_max_point)
                        this_max_point = (0,0)
            ys_diss_2ndd_avg = 0
            possible_points2 = []
            this_max_point = (0,0)
            for m in maxima_dict_2.items():
                if m[0] < t_1p0: continue
                if m[1] > ys_diss_2ndd_avg:
                    if m[1] > this_max_point[1]:
                        this_max_point = m
                if True:
                    if this_max_point != (0,0):
                        possible_points2.append(this_max_point)
                        this_max_point = (0,0)
            sorted_possibilities = []
            sorted_possibilities.extend(self.normalize_list_of_tuples(possible_points2))
            sorted_possibilities = sorted(sorted_possibilities, key = lambda kv:(kv[1], kv[0]), reverse = True)
            sequential_possibilities = sorted(sorted_possibilities, key = lambda kv:(kv[0], kv[1]), reverse = False)
            if len(sorted_possibilities) > 4:
                start_stop = sorted(sorted_possibilities[0:4])
            else:
                start_stop = sorted(sorted_possibilities)
            t_start = xs[start_stop[0][0]]
            t_blip3 = xs[start_stop[-1][0]]
            if len(sequential_possibilities) > 0:
                extend_dist = 0
                first_loop = True
                timeout_at = time.time() + 10.0
                while True:
                    allow_break = False
                    min_allowed_start = 3
                    max_allowed_start = avg_start + int((avg_stop - avg_start) / 10)
                    min_allowed_stop = avg_stop
                    if t_start < min_allowed_start:
                        Log.w("WARN: It looks like the run may have started prior to 3 seconds (during baseline period).")
                    if min_allowed_start < t_start < xs[max_allowed_start] and t_blip3 > xs[min_allowed_stop]:
                        if first_loop:
                            break
                        allow_break = True
                    else:
                        first_loop = False
                    extend_dist += 1
                    if len(sorted_possibilities) < 4+extend_dist:
                        break
                    if time.time() > timeout_at:
                        Log.e("Model took too long to finish and was aborted. Please manually select points.")
                        break
                    start_stop = sorted_possibilities[0:4+extend_dist]
                    if allow_break:
                        if start_stop[-1][0] < start_stop[-2][0]:
                            break
                    t_start = xs[avg_start]
                    t_blip3 = xs[sorted(start_stop)[-1][0]]
            else:
                raise Exception("Too few possibilities to Analyze.")
            t_delta = (t_blip3 - t_start) / 16
            t_quad0 = t_start - (1 + 0) * t_delta
            t_quad1 = t_start + (00.50) * t_delta
            t_quad2 = t_start + (1 + 2) * t_delta
            t_quad3 = t_start + (1 + 3) * t_delta
            t_quad4 = t_start + (1 + 9) * t_delta
            t_quad5 = t_start + (1 + 10) * t_delta
            t_quad6 = t_start + (1 + 20) * t_delta
            if t_quad0 < 2.5:
                t_quad0 = 2.5
                if t_quad1 < 3.0:
                    t_quad1 = 3.0
            if extend_data:
                norm_ys_diss = self.normalize(ys_extend)
                norm_ys_freq = self.normalize(ys_freq_extend)
                norm_ys_diff = self.normalize(ys_diff_extend)
                norm_ys_diff_fit = self.normalize(ys_diff_ext_fit)
                idx_quad0 = next(x for x,y in enumerate(xs_extend) if y > t_quad0)
                idx_quad1 = next(x for x,y in enumerate(xs_extend) if y >= t_quad1)
            else:
                norm_ys_diss = self.normalize(ys)
                norm_ys_freq = self.normalize(ys_freq)
                norm_ys_diff = self.normalize(ys_diff)
                norm_ys_diff_fit = self.normalize(ys_diff_fit)
                idx_quad0 = next(x for x,y in enumerate(xs) if y > t_quad0)
                idx_quad1 = next(x for x,y in enumerate(xs) if y >= t_quad1)
            scan_start = idx_quad0
            best_start_idx = 0
            best_end_idx = 0
            best_post_idx = 0
            best_num_fill_pts = 0
            best_post_pointfit = (0,0)
            best_max_pointfit = (0,0)
            best_x_intersection = 0
            best_y_intersection = 0
            start_idx = scan_start
            end_idx = scan_start
            post_idx = scan_start
            this_idx = scan_start
            start_diss = scan_start
            start_freq = scan_start
            start_diff = scan_start
            diff_baseline = np.average(norm_ys_diff_fit[t_0p5:idx_quad0])
            diff_delta = norm_ys_diff_fit[idx_quad1] - norm_ys_diff_fit[idx_quad0]
            if diff_delta < 0:
                Log.w("WARN: Baseline DIFF is higher than at the end of fill period.")
                Log.d("This condition is very unusual and is indicative of bad data.")
                norm_ys_diff = 1 - norm_ys_diff
                norm_ys_diff_fit = 1 - norm_ys_diff_fit
                diff_baseline = 1 - diff_baseline
                ys_diff_smooth_fill = 1 - ys_diff_smooth_fill
                temp_up3 = up3_fill[0].copy()
                temp_low3 = low3_fill[0].copy()
                up3_fill[0] = 1 - temp_low3
                low3_fill[0] = 1 - temp_up3
            diff_midpoint = diff_baseline + abs((diff_delta / 3))
            this_idx = idx_quad1
            debug_point1 = idx_quad0
            debug_point2 = idx_quad1
            method = -1
            while True:
                error_at_this_point = (up3_fill[0][this_idx-10] - low3_fill[0][this_idx-10]) / 2
                this_baseline = norm_ys_diff_fit[this_idx-10] - error_at_this_point
                if this_baseline < norm_ys_diff_fit[this_idx] < diff_midpoint:
                    if norm_ys_diff_fit[this_idx-1] > norm_ys_diff_fit[this_idx]:
                        if start_diff == scan_start:
                            start_diff = this_idx
                            debug_point1 = start_diff
                            break
                else:
                    start_diff = scan_start
                if this_idx == idx_quad0:
                    break
                this_idx -= 1
            break_diff_low = ys_diff_smooth_fill[this_idx] - 1.5*(ys_diff_smooth_fill[this_idx] - low3_fill[0][this_idx])
            break_diff_high = ys_diff_smooth_fill[this_idx] + 3*(up3_fill[0][this_idx] - ys_diff_smooth_fill[this_idx])
            while True:
                if (norm_ys_diff[this_idx] > break_diff_high and start_diff < this_idx - 1):
                    start_argmin = np.argmin(norm_ys_diff[start_diff:this_idx - 1])
                    debug_point2 = this_idx
                    if norm_ys_diff[start_diff + start_argmin] < break_diff_low:
                        Log.d("NOTICE: DIFF curve drops from baseline at start of fill.")
                        method = 4
                        start_diff = start_diff + start_argmin
                    else:
                        while True:
                            if norm_ys_diff[this_idx - 1] > norm_ys_diff[this_idx]:
                                Log.d("NOTICE: Found start point using relative minima approximation")
                                method = 1
                                start_diff = this_idx
                                break
                            if norm_ys_diff[this_idx] < up3_fill[0][debug_point1]:
                                Log.d("NOTICE: Found start point at crossing of average baseline")
                                method = 2
                                start_diff = this_idx
                                break
                            if this_idx <= debug_point1:
                                Log.w("WARN: Points are ever increasing at baseline DIFF, using 'start-1'")
                                method = 3
                                start_diff = debug_point2 - 1
                                break
                            this_idx -= 1
                    break
                if this_idx >= idx_quad1:
                    break
                this_idx += 1
            start_idx = start_diff
            if not extend_data:
                xs_extend = xs
            num_attempts = 0
            while True:
                num_attempts += 1
                max_pointfit = (0,0)
                max_idx = scan_start
                while True:
                    if num_attempts > 1 and max_pointfit[0] == 0:
                        pass
                    this_idx += 1
                    fit_start = idx_quad0
                    x1 = xs_extend[fit_start]
                    y1 = norm_ys_diff_fit[fit_start]
                    x2 = xs_extend[this_idx]
                    y2 = norm_ys_diff_fit[this_idx]
                    post_pointfit = self.point_fit((x1,y1),(x2,y2))
                    if post_pointfit[0] > max_pointfit[0]:
                        max_pointfit = post_pointfit
                        max_idx = this_idx
                    if this_idx >= idx_quad1:
                            break
                min_slope = (np.inf, this_idx)
                max_slope = (0, this_idx)
                p1 = (xs_extend[start_idx], norm_ys_diff_fit[start_idx])
                for this_idx in range(idx_quad1, start_idx, -1):
                    p2 = (xs_extend[this_idx], norm_ys_diff_fit[this_idx])
                    m, b = self.point_fit(p1, p2)
                    if min_slope[0] > m:
                        min_slope = (m, this_idx)
                    if max_slope[0] < m:
                        max_slope = (m, this_idx)
                p1 = (xs_extend[start_idx], norm_ys_diff_fit[start_idx])
                m1 = max_slope[0]
                b1 = self.line_fit(p1, m1)
                max_pointfit = (m1, b1)
                p2 = (xs_extend[max_idx], norm_ys_diff_fit[max_idx])
                m2 = min_slope[0]
                b2 = self.line_fit(p2, m2)
                post_pointfit = (m2, b2)
                x_intersection, y_intersection = self.find_intersect(max_pointfit, post_pointfit)
                end_idx = next(x for x,y in enumerate(xs_extend) if y >= x_intersection)
                post_idx = max_idx
                num_fill_pts = end_idx - start_idx
                smf = max(3, int(num_fill_pts/10))
                if smf % 2 == 0: smf += 1
                if extend_data:
                    ys_diff_soft = savgol_filter(ys_diff_extend, smf, 1)
                else:
                    ys_diff_soft = savgol_filter(ys_diff, smf, 1)
                norm_ys_diff_soft = self.normalize(ys_diff_soft)
                end_compare_result = 0
                this_idx = end_idx
                while True:
                    num_fill_pts = end_idx - start_idx
                    window_size = 1
                    p4 = (xs_extend[this_idx-window_size], norm_ys_diff_soft[this_idx-window_size])
                    p5 = (xs_extend[this_idx],             norm_ys_diff_soft[this_idx])
                    p6 = (xs_extend[this_idx+window_size], norm_ys_diff_soft[this_idx+window_size])
                    left_slope, _ = self.point_fit(p4, p5)
                    right_slope, _ = self.point_fit(p5, p6)
                    end_compare_result = self.compare_slopes(m1, m2, left_slope, right_slope, end_compare_result)
                    if end_compare_result == 0:
                        end_idx = this_idx
                        break
                    this_idx += end_compare_result
                    if this_idx <= start_idx or this_idx > idx_quad1:
                        break
                    max_pointfit = self.point_fit(p1, p5)
                    m1 = max_pointfit[0]
                    m2 = 0
                if end_compare_result != 0 and num_attempts < 2:
                    Log.w("Failed to find matching slope point! Using approximated end-of-fill point.")
                if num_attempts >= 1:
                    if best_end_idx != 0:
                        this_diff_mag_delta = norm_ys_diff_fit[end_idx] - norm_ys_diff_fit[start_idx]
                        last_diff_mag_delta = norm_ys_diff_fit[best_end_idx] - norm_ys_diff_fit[best_start_idx]
                        if (end_idx - start_idx < best_end_idx - best_start_idx or
                            this_diff_mag_delta < last_diff_mag_delta):
                            pass
                        else:
                            best_start_idx = start_idx
                            best_end_idx = end_idx
                            best_post_idx = post_idx
                            best_num_fill_pts = num_fill_pts
                            best_post_pointfit = post_pointfit
                            best_max_pointfit = max_pointfit
                            best_x_intersection = x_intersection
                            best_y_intersection = y_intersection
                    else:
                        best_start_idx = start_idx
                        best_end_idx = end_idx
                        best_post_idx = post_idx
                        best_num_fill_pts = num_fill_pts
                        best_post_pointfit = post_pointfit
                        best_max_pointfit = max_pointfit
                        best_x_intersection = x_intersection
                        best_y_intersection = y_intersection
                    break
                best_start_idx = start_idx
                best_end_idx = end_idx
                best_post_idx = post_idx
                best_num_fill_pts = num_fill_pts
                best_post_pointfit = post_pointfit
                best_max_pointfit = max_pointfit
                best_x_intersection = x_intersection
                best_y_intersection = y_intersection
                scan_start = end_idx
                start_idx = scan_start
                end_idx = scan_start
                post_idx = scan_start
                this_idx = scan_start
            if extend_data:
                avg_start = next(x for x,y in enumerate(xs) if y >= xs_extend[best_start_idx])
            else:
                avg_start = best_start_idx
            t_start = xs[avg_start]
            t_max_search = min(xs[avg_stop], t_quad5)
            t_end_search = xs[-1] - 1.0
            i_end_search = next(x for x,y in enumerate(xs) if y >= t_end_search)
            if diff_delta < 0:
                temp_high = break_diff_high.copy()
                temp_low = break_diff_low.copy()
                break_diff_high = 1 - temp_low
                break_diff_low = 1 - temp_high
                ys_diff_smooth_fill = 1 - ys_diff_smooth_fill
                up3_fill[0] = temp_up3
                low3_fill[0] = temp_low3
            norm_freq_fit = self.normalize(ys_freq_fit)
            idx_max_freq = np.argmax(norm_freq_fit)
            end_min_freq = np.amin(low2[0][i_end_search:])
            end_max_freq = np.amax(up2[0][i_end_search:])
            delta_freq = 1.0 - (end_max_freq - end_min_freq)
            if len(norm_freq_fit[i_end_search:]) < 5: delta_freq = 0.95
            if t_max_search < xs[idx_max_freq] < t_quad6 and norm_freq_fit[-1] < norm_freq_fit[idx_max_freq]*delta_freq:
                pass
            else:
                idx_max_freq = -1
            norm_diff_fit = self.normalize(ys_diff_fit)
            idx_max_diff = np.argmax(norm_diff_fit)
            end_min_diff = np.amin(low3[0][i_end_search:])
            end_max_diff = np.amax(up3[0][i_end_search:])
            delta_diff = 1.0 - (end_max_diff - end_min_diff)
            if len(norm_diff_fit[i_end_search:]) < 5: delta_diff = 0.95
            if t_max_search < xs[idx_max_diff] < t_quad6 and norm_diff_fit[-1] < norm_diff_fit[idx_max_diff]*delta_diff:
                pass
            else:
                idx_max_diff = -1
            if idx_max_freq != -1 and idx_max_diff != -1:
                Log.d("Using FREQ and DIFF dropoff to determine stop point.")
                if norm_freq_fit[-1] < idx_max_diff:
                    t_blip3 = xs[idx_max_freq]
                else:
                    t_blip3 = xs[idx_max_diff]
            elif idx_max_freq != -1:
                Log.d("Using FREQ dropoff to determine stop point.")
                t_blip3 = xs[idx_max_freq]
            elif idx_max_diff != -1:
                Log.d("Using DIFF dropoff to determine stop point.")
                t_blip3 = xs[idx_max_diff]
            idx_max_diss = -1
            if start_at != None:
                if len(start_at) == 6:
                    if start_at[0] != None:
                        if extend_data:
                            avg_start = next(x for x,y in enumerate(xs) if y >= xs_extend[start_at[0]])
                        else:
                            avg_start = start_at[0]
                        t_start = xs[avg_start]
                        Log.d(f"Using explicit start reference: xs[{avg_start}] = {t_start}s")
                        idx_max_diff = 0
                    if start_at[-1] != None:
                        if extend_data:
                            avg_stop = next(x for x,y in enumerate(xs) if y >= xs_extend[start_at[-1]])
                        else:
                            avg_stop = start_at[-1]
                        t_blip3 = xs[avg_stop]
                        Log.d(f"Using explicit stop reference: xs[{avg_stop}] = {t_blip3}s")
                        idx_max_diff = 0
                else:
                    Log.w("Starting conditions (start_at) are the wrong length (!= 6). Ignoring them.")
            if idx_max_freq != -1 or idx_max_diff != -1 or idx_max_diss != -1:
                t_delta = (t_blip3 - t_start) / 16
                t_quad0 = t_start - (1 + 0) * t_delta
                t_quad1 = t_start + (00.50) * t_delta
                t_quad2 = t_start + (1 + 2) * t_delta
                t_quad3 = t_start + (1 + 3) * t_delta
                t_quad4 = t_start + (1 + 9) * t_delta
                t_quad5 = t_start + (1 + 10) * t_delta
                t_quad6 = t_start + (1 + 20) * t_delta
            if t_quad0 < 2.5:
                t_quad0 = 2.5
                if t_quad1 < 3.0:
                    t_quad1 = 3.0
            if t_quad6 > xs[-1]:
                t_quad6 = xs[-1]
            t_quads = [t_quad0, t_quad1, t_quad2, t_quad3, t_quad4, t_quad5, t_quad6]
            if extend_data:
                idx_quad0 = next(x for x,y in enumerate(xs_extend) if y > t_quad0)
                idx_quad1 = next(x for x,y in enumerate(xs_extend) if y >= t_quad1)
            else:
                idx_quad0 = next(x for x,y in enumerate(xs) if y > t_quad0)
                idx_quad1 = next(x for x,y in enumerate(xs) if y >= t_quad1)
            if best_start_idx < idx_quad0:
                best_start_idx = idx_quad0 + 1
            if best_end_idx < idx_quad0:
                best_end_idx = idx_quad0 + 2
            if best_post_idx < idx_quad0:
                best_post_idx = idx_quad0 + 3
            if best_start_idx > idx_quad1:
                best_start_idx = idx_quad1 - 3
            if best_end_idx > idx_quad1:
                best_end_idx = idx_quad1 -2
            if best_post_idx > idx_quad1:
                best_post_idx = idx_quad1 - 1
            zone1_possibilities = []
            zone2_possibilities = []
            zone3_possibilities = []
            zone4_possibilities = []
            for p in sequential_possibilities:
                t_now = xs[p[0]]
                if t_now > t_quad0 and t_now < t_quad1:
                    zone1_possibilities.append(p)
                if t_now > t_quad1 and t_now < t_quad2:
                    zone2_possibilities.append(p)
                if t_now > t_quad3 and t_now < t_quad4:
                    zone3_possibilities.append(p)
                if t_now > t_quad5 and t_now < t_quad6:
                    zone4_possibilities.append(p)
            idx_quad1 = next(x for x,y in enumerate(xs) if y >= t_quad1)
            idx_quad2 = next(x for x,y in enumerate(xs) if y >= t_quad2)
            min_diss_zone2_1 = np.argmin(super_smooth_diss_1st[idx_quad1:idx_quad2])
            min_freq_zone2_1 = np.argmin(super_smooth_freq_1st[idx_quad1:idx_quad2])
            min_diff_zone2_1 = np.argmin(super_smooth_diff_1st[idx_quad1:idx_quad2])
            if min_diss_zone2_1 in [0, idx_quad2-idx_quad1-1]:
                min_diss_zone2_1 = -1
            if min_freq_zone2_1 in [0, idx_quad2-idx_quad1-1]:
                min_freq_zone2_1 = -1
            if min_diff_zone2_1 in [0, idx_quad2-idx_quad1-1]:
                min_diff_zone2_1 = -1
            min_diss_zone2_2 = np.argmin(super_smooth_diss_2nd[idx_quad1:idx_quad2])
            min_freq_zone2_2 = np.argmin(super_smooth_freq_2nd[idx_quad1:idx_quad2])
            min_diff_zone2_2 = np.argmin(super_smooth_diff_2nd[idx_quad1:idx_quad2])
            if min_diss_zone2_2 in [0, idx_quad2-idx_quad1-1]:
                min_diss_zone2_2 = -1
            if min_freq_zone2_2 in [0, idx_quad2-idx_quad1-1]:
                min_freq_zone2_2 = -1
            if min_diff_zone2_2 in [0, idx_quad2-idx_quad1-1]:
                min_diff_zone2_2 = -1
            num_count = 0
            num_sum = 0
            for i in [min_diss_zone2_1, min_freq_zone2_1, min_diff_zone2_1]:
                if i != -1:
                    num_sum += i
                    num_count += 1
            if num_count > 0:
                min_peak_zone2 = int(num_sum / num_count)
            else:
                min_peak_zone2 = int(idx_quad2/2-idx_quad1/2)
            idx_quad3 = next(x for x,y in enumerate(xs) if y >= t_quad3)
            idx_quad4 = next(x for x,y in enumerate(xs) if y >= t_quad4)
            min_diss_zone3 = np.argmin(super_smooth_diss_1st[idx_quad3:idx_quad4])
            min_freq_zone3 = np.argmin(super_smooth_freq_1st[idx_quad3:idx_quad4])
            min_diff_zone3 = np.argmin(super_smooth_diff_1st[idx_quad3:idx_quad4])
            if min_diss_zone3 in [0, idx_quad4-idx_quad3-1]:
                min_diss_zone3 = int(idx_quad4/2-idx_quad3/2)
            if min_freq_zone3 in [0, idx_quad4-idx_quad3-1]:
                min_freq_zone3 = int(idx_quad4/2-idx_quad3/2)
            if min_diff_zone3 in [0, idx_quad4-idx_quad3-1]:
                min_diff_zone3 = int(idx_quad4/2-idx_quad3/2)
            min_peak_zone3 = int(np.average([min_diss_zone3, min_freq_zone3, min_diff_zone3]))
            idx_quad5 = next(x for x,y in enumerate(xs) if y >= t_quad5)
            try:
                idx_quad6 = next(x for x,y in enumerate(xs) if y >= t_quad6)
            except:
                idx_quad6 = len(xs)-1
            max_diss_zone4_1 = np.argmax(super_smooth_diss_1st[idx_quad5:idx_quad6])
            max_freq_zone4_1 = np.argmax(super_smooth_freq_1st[idx_quad5:idx_quad6])
            max_diff_zone4_1 = np.argmax(super_smooth_diff_1st[idx_quad5:idx_quad6])
            if max_diss_zone4_1 in [0, idx_quad6-idx_quad5-1]:
                max_diss_zone4_1 = -1
            if max_freq_zone4_1 in [0, idx_quad6-idx_quad5-1]:
                max_freq_zone4_1 = -1
            if max_diff_zone4_1 in [0, idx_quad6-idx_quad5-1]:
                max_diff_zone4_1 = -1
            max_diss_zone4_2 = np.argmax(super_smooth_diss_2nd[idx_quad5:idx_quad6])
            max_freq_zone4_2 = np.argmax(super_smooth_freq_2nd[idx_quad5:idx_quad6])
            max_diff_zone4_2 = np.argmax(super_smooth_diff_2nd[idx_quad5:idx_quad6])
            if max_diss_zone4_2 in [0, idx_quad6-idx_quad5-1]:
                max_diss_zone4_2 = -1
            if max_freq_zone4_2 in [0, idx_quad6-idx_quad5-1]:
                max_freq_zone4_2 = -1
            if max_diff_zone4_2 in [0, idx_quad6-idx_quad5-1]:
                max_diff_zone4_2 = -1
            min_diss_zone4_2 = np.argmin(super_smooth_diss_2nd[idx_quad5:idx_quad6])
            min_freq_zone4_2 = np.argmin(super_smooth_freq_2nd[idx_quad5:idx_quad6])
            min_diff_zone4_2 = np.argmin(super_smooth_diff_2nd[idx_quad5:idx_quad6])
            if min_diss_zone4_2 in [0, idx_quad6-idx_quad5-1] or min_diss_zone4_2 < max_diss_zone4_2:
                min_diss_zone4_2 = -1
            if min_freq_zone4_2 in [0, idx_quad6-idx_quad5-1] or min_freq_zone4_2 < max_freq_zone4_2:
                min_freq_zone4_2 = -1
            if min_diff_zone4_2 in [0, idx_quad6-idx_quad5-1] or min_diff_zone4_2 < max_diff_zone4_2:
                min_diff_zone4_2 = -1
            num_count = 0
            num_sum = 0
            for i in [max_diss_zone4_1, max_freq_zone4_1, max_diff_zone4_1,
                      max_diss_zone4_2, max_freq_zone4_2, max_diff_zone4_2]:
                if i != -1:
                    num_sum += i
                    num_count += 1
            if num_count > 0:
                max_peak_zone4 = int(num_sum / num_count)
            else:
                max_peak_zone4 = int(idx_quad6/2-idx_quad5/2)
            zone2l = np.linspace(0.25, 1.0, min_peak_zone2)
            zone2r = np.linspace(1.0, 0.25, idx_quad2-idx_quad1-min_peak_zone2)
            zone2_scaler = np.concatenate([zone2l, zone2r])
            zone3l = np.linspace(0.25, 1.0, min_peak_zone3)
            zone3r = np.linspace(1.0, 0.25, idx_quad4-idx_quad3-min_peak_zone3)
            zone3_scaler = np.concatenate([zone3l, zone3r])
            ideal_zone1 = avg_start
            ideal_zone2 = idx_quad1 + min_peak_zone2
            ideal_zone3 = idx_quad3 + min_peak_zone3
            if False:
                if avg_stop > idx_quad5 and avg_stop < idx_quad6:
                    ideal_zone4 = avg_stop
                else:
                    ideal_zone4 = idx_quad5 + max_peak_zone4
            else:
                ideal_zone4 = -1
            zone1_possibilities = self.normalize_list_of_tuples(zone1_possibilities)
            zone2_possibilities = self.normalize_list_of_tuples(zone2_possibilities)
            zone3_possibilities = self.normalize_list_of_tuples(zone3_possibilities)
            zone4_possibilities = self.normalize_list_of_tuples(zone4_possibilities)
            for zpi in range(len(zone2_possibilities)):
                pos, mag = zone2_possibilities[zpi]
                scale_by = zone2_scaler[pos-idx_quad1]
                zone2_possibilities[zpi] = (pos, scale_by*mag)
            for zpi in range(len(zone3_possibilities)):
                pos, mag = zone3_possibilities[zpi]
                scale_by = zone3_scaler[pos-idx_quad3]
                zone3_possibilities[zpi] = (pos, scale_by*mag)
            for zpi in range(len(zone4_possibilities)):
                pos, mag = zone4_possibilities[zpi]
                if mag == 1.0 and ideal_zone4 == -1:
                    ideal_zone4 = pos
            if start_at != None:
                if len(start_at) == 6:
                    if start_at[0] != None:
                        if extend_data:
                            ideal_zone1 = next(x for x,y in enumerate(xs) if y >= xs_extend[start_at[0]])
                        else:
                            ideal_zone1 = start_at[0]
                        Log.d(f"Ideal zone 1 point: xs[{ideal_zone1}] = {xs[ideal_zone1]}s")
                    if start_at[-3] != None:
                        if extend_data:
                            ideal_zone2 = next(x for x,y in enumerate(xs) if y >= xs_extend[start_at[-3]])
                        else:
                            ideal_zone2 = start_at[-3]
                        Log.d(f"Ideal zone 2 point: xs[{ideal_zone2}] = {xs[ideal_zone2]}s")
                    if start_at[-2] != None:
                        if extend_data:
                            ideal_zone3 = next(x for x,y in enumerate(xs) if y >= xs_extend[start_at[-2]])
                        else:
                            ideal_zone3 = start_at[-2]
                        Log.d(f"Ideal zone 3 point: xs[{ideal_zone3}] = {xs[ideal_zone3]}s")
                    if start_at[-1] != None:
                        if extend_data:
                            ideal_zone4 = next(x for x,y in enumerate(xs) if y >= xs_extend[start_at[-1]])
                        else:
                            ideal_zone4 = start_at[-1]
                        Log.d(f"Ideal zone 4 point: xs[{ideal_zone4}] = {xs[ideal_zone4]}s")
                else:
                    Log.w("Starting conditions (start_at) are the wrong length (!= 6). Ignoring them.")
            zone1_possibilities = self.normalize_list_of_tuples(zone1_possibilities)
            zone2_possibilities = self.normalize_list_of_tuples(zone2_possibilities)
            zone3_possibilities = self.normalize_list_of_tuples(zone3_possibilities)
            zone4_possibilities = self.normalize_list_of_tuples(zone4_possibilities)
            consider_points_above_pct = Constants.consider_points_above_pct
            zone1_possibilities.append((ideal_zone1, consider_points_above_pct))
            zone2_possibilities.append((ideal_zone2, consider_points_above_pct))
            zone3_possibilities.append((ideal_zone3, consider_points_above_pct))
            zone4_possibilities.append((ideal_zone4, consider_points_above_pct))
            sorted_zone1_possibilities = sorted(zone1_possibilities, key = lambda kv:(kv[1], kv[0]), reverse = True)
            sorted_zone2_possibilities = sorted(zone2_possibilities, key = lambda kv:(kv[1], kv[0]), reverse = True)
            sorted_zone3_possibilities = sorted(zone3_possibilities, key = lambda kv:(kv[1], kv[0]), reverse = True)
            sorted_zone4_possibilities = sorted(zone4_possibilities, key = lambda kv:(kv[1], kv[0]), reverse = True)
            if True or end_compare_result == 0:
                Log.d(f"Number of fill points: {best_num_fill_pts}")
                best_start_idx += min(10, int(best_num_fill_pts / 25))
                best_post_idx = best_end_idx + max(2, int(best_num_fill_pts / 10))
            t_fill_start = xs_extend[best_start_idx] if extend_data else xs[best_start_idx]
            t_fill_end = xs_extend[best_end_idx] if extend_data else xs[best_end_idx]
            t_fill_post = xs_extend[best_post_idx] if extend_data else xs[best_post_idx]
            Log.d(f"Start-of-fill: {t_fill_start}")
            Log.d(f"End-of-fill: {t_fill_end}")
            Log.d(f"Post-of-fill: {t_fill_post}")
            if len(sorted_zone1_possibilities) > 0:
                if False:
                    zone1_mags = []
                    for zp in zone1_possibilities:
                        zone1_mags.append(zp[1])
                    avg_zone1_mag = np.average(zone1_mags)
                    for p in zone1_possibilities[::-1]:
                        if p[1] > 0.25:
                            t_start = xs[p[0]]
                            break
                else:
                    min_dist_to_ideal = idx_quad2 - idx_quad1
                    min_zp_idx = ideal_zone1
                    for zp in sorted_zone1_possibilities:
                        dist_to_ideal = abs(zp[0] - ideal_zone1)
                        if dist_to_ideal == 0 and zp[1] == consider_points_above_pct:
                            if min_dist_to_ideal / (idx_quad2 - idx_quad1) > 0.25:
                                min_zp_idx = ideal_zone1
                            t_start = xs[min_zp_idx]
                            break
                        if dist_to_ideal < min_dist_to_ideal:
                            min_dist_to_ideal = dist_to_ideal
                            min_zp_idx = zp[0]
            if len(sorted_zone2_possibilities) > 0:
                if False:
                    last_dist_to_ideal = idx_quad2 - idx_quad1
                    last_zp_idx = ideal_zone2
                    for zp in sorted_zone2_possibilities:
                        dist_to_ideal = abs(zp[0] - ideal_zone2)
                        if dist_to_ideal > last_dist_to_ideal or dist_to_ideal == 0:
                            t_blip1 = xs[last_zp_idx]
                            break
                        last_dist_to_ideal = dist_to_ideal
                        last_zp_idx = zp[0]
                else:
                    min_dist_to_ideal = idx_quad2 - idx_quad1
                    min_zp_idx = ideal_zone2
                    for zp in sorted_zone2_possibilities:
                        dist_to_ideal = abs(zp[0] - ideal_zone2)
                        if dist_to_ideal == 0 and zp[1] == consider_points_above_pct:
                            if min_dist_to_ideal / (idx_quad2 - idx_quad1) > 0.25:
                                min_zp_idx = ideal_zone2
                            t_blip1 = xs[min_zp_idx]
                            break
                        if dist_to_ideal < min_dist_to_ideal:
                            min_dist_to_ideal = dist_to_ideal
                            min_zp_idx = zp[0]
            else:
                t_blip1 = (t_quad1 + t_quad2) / 2
            if len(sorted_zone3_possibilities) > 0:
                if False:
                    last_dist_to_ideal = idx_quad4 - idx_quad3
                    last_zp_idx = ideal_zone3
                    for zp in sorted_zone3_possibilities:
                        dist_to_ideal = abs(zp[0] - ideal_zone3)
                        if dist_to_ideal > last_dist_to_ideal or dist_to_ideal == 0:
                            t_blip2 = xs[last_zp_idx]
                            break
                        last_dist_to_ideal = dist_to_ideal
                        last_zp_idx = zp[0]
                else:
                    min_dist_to_ideal = idx_quad4 - idx_quad3
                    min_zp_idx = ideal_zone3
                    for zp in sorted_zone3_possibilities:
                        dist_to_ideal = abs(zp[0] - ideal_zone3)
                        if dist_to_ideal == 0 and zp[1] == consider_points_above_pct:
                            if min_dist_to_ideal / (idx_quad4 - idx_quad3) > 0.25:
                                min_zp_idx = ideal_zone3
                            t_blip2 = xs[min_zp_idx]
                            break
                        if dist_to_ideal < min_dist_to_ideal:
                            min_dist_to_ideal = dist_to_ideal
                            min_zp_idx = zp[0]
            else:
                t_blip2 = (t_quad3 + t_quad4) / 2
            if len(sorted_zone4_possibilities) > 0:
                if False:
                    zone4_mags = []
                    for zp in zone4_possibilities:
                        zone4_mags.append(zp[1])
                    avg_zone4_mag = np.average(zone4_mags)
                    for p in zone4_possibilities:
                        if p[1] > 0.75:
                            t_blip3 = xs[p[0]]
                            break
                elif idx_max_freq == -1 and idx_max_diff == -1 and idx_max_diss == -1:
                    min_dist_to_ideal = idx_quad6 - idx_quad5
                    min_zp_idx = ideal_zone4
                    for zp in sorted_zone4_possibilities:
                        dist_to_ideal = abs(zp[0] - ideal_zone4)
                        if dist_to_ideal == 0 and zp[1] == consider_points_above_pct:
                            if False:
                                min_zp_idx = ideal_zone4
                            t_blip3 = xs[min_zp_idx]
                            break
                        if dist_to_ideal < min_dist_to_ideal:
                            min_dist_to_ideal = dist_to_ideal
                            min_zp_idx = zp[0]
            options_to_show = 5
            n_bins_zone1 = 10
            n_bins_zone2 = 10
            n_bins_zone3 = 10
            n_bins_zone4 = 10
            idx_start = next(x for x,y in enumerate(xs) if y >= t_fill_start)
            idx_blip1 = next(x for x,y in enumerate(xs) if y >= t_blip1)
            idx_blip2 = next(x for x,y in enumerate(xs) if y >= t_blip2)
            idx_blip3 = next(x for x,y in enumerate(xs) if y >= t_blip3)
            zone1_possibilities[-1] = (idx_start, 1.0 + consider_points_above_pct)
            zone2_possibilities[-1] = (idx_blip1, 1.0 + consider_points_above_pct)
            zone3_possibilities[-1] = (idx_blip2, 1.0 + consider_points_above_pct)
            zone4_possibilities[-1] = (idx_blip3, 1.0 + consider_points_above_pct)
            x1, y1 = map(list, zip(*zone1_possibilities))
            x2, y2 = map(list, zip(*zone2_possibilities))
            x3, y3 = map(list, zip(*zone3_possibilities))
            x4, y4 = map(list, zip(*zone4_possibilities))
            hist_values, bin_edges = np.histogram(xs[x1], bins=n_bins_zone1-1, range=(t_quad0,t_quad1))
            x_bin_numbers = np.digitize(xs[x1], bin_edges)
            zone1_options = []
            zone1_confidence = []
            last_bin = -1
            last_max = -1
            max_bin = x1[np.argmax(y1)]
            bin_possibility_sum = 0
            for x_bin in np.unique(x_bin_numbers):
                bin_idx = np.argwhere(x_bin_numbers == x_bin)
                if len(bin_idx) == 0:
                    continue
                for i,idx in enumerate(bin_idx):
                    idx = idx[0]
                    if y1[idx] < consider_points_above_pct / 2:
                        pass
                    if x_bin != last_bin:
                        zone1_options.append(x1[idx])
                        zone1_confidence.append(y1[idx])
                        last_bin = x_bin
                        last_max = idx
                    else:
                        if y1[idx] > y1[last_max]:
                            zone1_options[-1] = x1[idx]
                            last_max = idx
                        zone1_confidence[-1] += y1[idx]
            sort_order = np.array(zone1_confidence).argsort()[::-1]
            zone1_options = np.array(zone1_options)[sort_order]
            zone1_confidence = np.array(zone1_confidence)[sort_order]
            if zone1_options[0] != max_bin:
                max_idx = np.argwhere(zone1_options == max_bin)[0][0]
                temp_option = zone1_options[max_idx]
                temp_confidence = zone1_confidence[max_idx]
                for i in range(max_idx,0,-1):
                    zone1_options[i] = zone1_options[i-1]
                    zone1_confidence[i] = zone1_confidence[i-1]
                zone1_options[0] = temp_option
                zone1_confidence[0] = temp_confidence
            if len(sort_order) > options_to_show:
                zone1_options = zone1_options[:options_to_show]
                zone1_confidence = zone1_confidence[:options_to_show]
            zone1_confidence /= np.sum(zone1_confidence)
            zone1_pois = list(zip(zone1_options, zone1_confidence))
            hist_values, bin_edges = np.histogram(xs[x2], bins=n_bins_zone2-1, range=(t_quad1,t_quad2))
            x_bin_numbers = np.digitize(xs[x2], bin_edges)
            zone2_options = []
            zone2_confidence = []
            last_bin = -1
            last_max = -1
            max_bin = x2[np.argmax(y2)]
            bin_possibility_sum = 0
            for x_bin in np.unique(x_bin_numbers):
                bin_idx = np.argwhere(x_bin_numbers == x_bin)
                if len(bin_idx) == 0:
                    continue
                for i,idx in enumerate(bin_idx):
                    idx = idx[0]
                    if y2[idx] < consider_points_above_pct / 2:
                        pass
                    if x_bin != last_bin:
                        zone2_options.append(x2[idx])
                        zone2_confidence.append(y2[idx])
                        last_bin = x_bin
                        last_max = idx
                    else:
                        if y2[idx] > y2[last_max]:
                            zone2_options[-1] = x2[idx]
                            last_max = idx
                        zone2_confidence[-1] += y2[idx]
            sort_order = np.array(zone2_confidence).argsort()[::-1]
            zone2_options = np.array(zone2_options)[sort_order]
            zone2_confidence = np.array(zone2_confidence)[sort_order]
            if zone2_options[0] != max_bin:
                max_idx = np.argwhere(zone2_options == max_bin)[0][0]
                temp_option = zone2_options[max_idx]
                temp_confidence = zone2_confidence[max_idx]
                for i in range(max_idx,0,-1):
                    zone2_options[i] = zone2_options[i-1]
                    zone2_confidence[i] = zone2_confidence[i-1]
                zone2_options[0] = temp_option
                zone2_confidence[0] = temp_confidence
            if len(sort_order) > options_to_show:
                zone2_options = zone2_options[:options_to_show]
                zone2_confidence = zone2_confidence[:options_to_show]
            zone2_confidence /= np.sum(zone2_confidence)
            zone2_pois = list(zip(zone2_options, zone2_confidence))
            hist_values, bin_edges = np.histogram(xs[x3], bins=n_bins_zone3-1, range=(t_quad3,t_quad4))
            x_bin_numbers = np.digitize(xs[x3], bin_edges)
            zone3_options = []
            zone3_confidence = []
            last_bin = -1
            last_max = -1
            max_bin = x3[np.argmax(y3)]
            bin_possibility_sum = 0
            for x_bin in np.unique(x_bin_numbers):
                bin_idx = np.argwhere(x_bin_numbers == x_bin)
                if len(bin_idx) == 0:
                    continue
                for i,idx in enumerate(bin_idx):
                    idx = idx[0]
                    if y3[idx] < consider_points_above_pct / 2:
                        pass
                    if x_bin != last_bin:
                        zone3_options.append(x3[idx])
                        zone3_confidence.append(y3[idx])
                        last_bin = x_bin
                        last_max = idx
                    else:
                        if y3[idx] > y3[last_max]:
                            zone3_options[-1] = x3[idx]
                            last_max = idx
                        zone3_confidence[-1] += y3[idx]
            sort_order = np.array(zone3_confidence).argsort()[::-1]
            zone3_options = np.array(zone3_options)[sort_order]
            zone3_confidence = np.array(zone3_confidence)[sort_order]
            if zone3_options[0] != max_bin:
                max_idx = np.argwhere(zone3_options == max_bin)[0][0]
                temp_option = zone3_options[max_idx]
                temp_confidence = zone3_confidence[max_idx]
                for i in range(max_idx,0,-1):
                    zone3_options[i] = zone3_options[i-1]
                    zone3_confidence[i] = zone3_confidence[i-1]
                zone3_options[0] = temp_option
                zone3_confidence[0] = temp_confidence
            if len(sort_order) > options_to_show:
                zone3_options = zone3_options[:options_to_show]
                zone3_confidence = zone3_confidence[:options_to_show]
            zone3_confidence /= np.sum(zone3_confidence)
            zone3_pois = list(zip(zone3_options, zone3_confidence))
            hist_values, bin_edges = np.histogram(xs[x4], bins=n_bins_zone4-1, range=(t_quad5,t_quad6))
            x_bin_numbers = np.digitize(xs[x4], bin_edges)
            zone4_options = []
            zone4_confidence = []
            last_bin = -1
            last_max = -1
            max_bin = x4[np.argmax(y4)]
            bin_possibility_sum = 0
            for x_bin in np.unique(x_bin_numbers):
                bin_idx = np.argwhere(x_bin_numbers == x_bin)
                if len(bin_idx) == 0:
                    continue
                for i,idx in enumerate(bin_idx):
                    idx = idx[0]
                    if y4[idx] < consider_points_above_pct / 2:
                        pass
                    if x_bin != last_bin:
                        zone4_options.append(x4[idx])
                        zone4_confidence.append(y4[idx])
                        last_bin = x_bin
                        last_max = idx
                    else:
                        if y4[idx] > y4[last_max]:
                            zone4_options[-1] = x4[idx]
                            last_max = idx
                        zone4_confidence[-1] += y4[idx]
            sort_order = np.array(zone4_confidence).argsort()[::-1]
            zone4_options = np.array(zone4_options)[sort_order]
            zone4_confidence = np.array(zone4_confidence)[sort_order]
            if zone4_options[0] != max_bin:
                max_idx = np.argwhere(zone4_options == max_bin)[0][0]
                temp_option = zone4_options[max_idx]
                temp_confidence = zone4_confidence[max_idx]
                for i in range(max_idx,0,-1):
                    zone4_options[i] = zone4_options[i-1]
                    zone4_confidence[i] = zone4_confidence[i-1]
                zone4_options[0] = temp_option
                zone4_confidence[0] = temp_confidence
            if len(sort_order) > options_to_show:
                zone4_options = zone4_options[:options_to_show]
                zone4_confidence = zone4_confidence[:options_to_show]
            zone4_confidence /= np.sum(zone4_confidence)
            zone4_pois = list(zip(zone4_options, zone4_confidence))
            if self._headless:
                Log.i(f"Compare calculated stop to actual stop: {t_blip3} -> {xs[list(maxima_dict_1.keys())[-1]]}")
            start_stop = [t_fill_start, t_fill_end, t_fill_post, t_blip1, t_blip2, t_blip3]
            ys_fit_data = ys_diff_ext_fit if extend_data else ys_diff_fit
            if start_at != None:
                if len(start_at) == 6:
                    if start_at[-3] != None:
                        curr_xs_idx = start_at[-3]
                        curr_xs_idx = next(x for x,y in enumerate(xs) if y >= xs_extend[curr_xs_idx])
                        zone2_pois[0] = (curr_xs_idx, 1.0)
                    if start_at[-2] != None:
                        curr_xs_idx = start_at[-2]
                        curr_xs_idx = next(x for x,y in enumerate(xs) if y >= xs_extend[curr_xs_idx])
                        zone3_pois[0] = (curr_xs_idx, 1.0)
                    if start_at[-1] != None:
                        curr_xs_idx = start_at[-1]
                        curr_xs_idx = next(x for x,y in enumerate(xs) if y >= xs_extend[curr_xs_idx])
                        zone4_pois[0] = (curr_xs_idx, 1.0)
                else:
                    Log.w("Starting conditions (start_at) are the wrong length (!= 6). Ignoring them.")
            for i in range(len(zone2_pois)):
                curr_xs_idx = zone2_pois[i][0]
                same_confid = zone2_pois[i][1]
                curr_xs_idx = next(x for x,y in enumerate(xs_extend) if y >= xs[curr_xs_idx])
                if curr_xs_idx in [-1, len(xs_extend)-1]:
                    curr_xs_idx = len(xs_extend)-2
                max_time_shift = 1.0
                min_time = xs_extend[curr_xs_idx] - max_time_shift
                max_time = xs_extend[curr_xs_idx] + max_time_shift
                start_x_idx = curr_xs_idx
                while True:
                    p1 = (xs_extend[curr_xs_idx-1], ys_fit_data[curr_xs_idx-1])
                    p2 = (xs_extend[curr_xs_idx+0], ys_fit_data[curr_xs_idx+0])
                    p3 = (xs_extend[curr_xs_idx+1], ys_fit_data[curr_xs_idx+1])
                    m0 = 0
                    m1, b1 = best_max_pointfit
                    m2, b2 = self.point_fit(p1, p2)
                    m3, b3 = self.point_fit(p2, p3)
                    m2, m3 = abs(m2), abs(m3)
                    end_compare_result = -1
                    if m2 < m3 and abs(m2-m0) < abs(m2-m1):
                        end_compare_result = 0
                    if min_time < xs_extend[curr_xs_idx] < max_time:
                        if end_compare_result == 0:
                            if i == 0:
                                Log.d(f"Moved blip1 by {curr_xs_idx-start_x_idx} points when fine-tuning.")
                            zone2_pois[i] = (curr_xs_idx, same_confid)
                            break
                    else:
                        if i == 0:
                            Log.d("Fine-tuning point for blip1 not found. Leaving it as-is.")
                        zone2_pois[i] = (start_x_idx, same_confid)
                        break
                    curr_xs_idx += end_compare_result
            for i in range(len(zone3_pois)):
                curr_xs_idx = zone3_pois[i][0]
                same_confid = zone3_pois[i][1]
                curr_xs_idx = next(x for x,y in enumerate(xs_extend) if y >= xs[curr_xs_idx])
                if curr_xs_idx in [-1, len(xs_extend)-1]:
                    curr_xs_idx = len(xs_extend)-2
                max_time_shift = 2.0
                min_time = xs_extend[curr_xs_idx] - max_time_shift
                max_time = xs_extend[curr_xs_idx] + max_time_shift
                start_x_idx = curr_xs_idx
                while True:
                    p1 = (xs_extend[curr_xs_idx-1], ys_fit_data[curr_xs_idx-1])
                    p2 = (xs_extend[curr_xs_idx+0], ys_fit_data[curr_xs_idx+0])
                    p3 = (xs_extend[curr_xs_idx+1], ys_fit_data[curr_xs_idx+1])
                    m0 = 0
                    m1, b1 = best_max_pointfit
                    m2, b2 = self.point_fit(p1, p2)
                    m3, b3 = self.point_fit(p2, p3)
                    m2, m3 = abs(m2), abs(m3)
                    end_compare_result = -1
                    if m2 < m3 and abs(m2-m0) < abs(m2-m1):
                        end_compare_result = 0
                    if min_time < xs_extend[curr_xs_idx] < max_time:
                        if end_compare_result == 0:
                            if i == 0:
                                Log.d(f"Moved blip2 by {curr_xs_idx-start_x_idx} points when fine-tuning.")
                            zone3_pois[i] = (curr_xs_idx, same_confid)
                            break
                    else:
                        if i == 0:
                            Log.d("Fine-tuning point for blip2 not found. Leaving it as-is.")
                        zone3_pois[i] = (start_x_idx, same_confid)
                        break
                    curr_xs_idx += end_compare_result
            for i in range(len(zone4_pois)):
                curr_xs_idx = zone4_pois[i][0]
                same_confid = zone4_pois[i][1]
                curr_xs_idx = next(x for x,y in enumerate(xs_extend) if y >= xs[curr_xs_idx])
                if curr_xs_idx in [-1, len(xs_extend)-1]:
                    curr_xs_idx = len(xs_extend)-2
                max_time_shift = 3.0
                min_time = xs_extend[curr_xs_idx] - max_time_shift
                max_time = xs_extend[curr_xs_idx] + max_time_shift
                start_x_idx = curr_xs_idx
                while True:
                    p1 = (xs_extend[curr_xs_idx-1], ys_fit_data[curr_xs_idx-1])
                    p2 = (xs_extend[curr_xs_idx+0], ys_fit_data[curr_xs_idx+0])
                    p3 = (xs_extend[curr_xs_idx+1], ys_fit_data[curr_xs_idx+1])
                    m0 = 0
                    m1, b1 = best_max_pointfit
                    m2, b2 = self.point_fit(p1, p2)
                    m3, b3 = self.point_fit(p2, p3)
                    m2, m3 = abs(m2), abs(m3)
                    end_compare_result = -1
                    if m2 < m3 and abs(m2-m0) < abs(m2-m1):
                        end_compare_result = 0
                    if min_time < xs_extend[curr_xs_idx] < max_time:
                        if end_compare_result == 0:
                            if i == 0:
                                Log.d(f"Moved blip3 by {curr_xs_idx-start_x_idx} points when fine-tuning.")
                            zone4_pois[i] = (curr_xs_idx, same_confid)
                            break
                    else:
                        if i == 0:
                            Log.d("Fine-tuning point for blip3 not found. Leaving it as-is.")
                        zone4_pois[i] = (start_x_idx, same_confid)
                        break
                    curr_xs_idx += end_compare_result
            try:
                xs_extend[-1]
            except:
                xs_extend = xs
            for i in range(len(start_stop)):
                try:
                    start_stop[i] = next(x for x,y in enumerate(xs_extend) if y >= start_stop[i])
                except:
                    start_stop[i] = -1
            poi_vals = start_stop
            poi_vals[0] = [(poi_vals[0],zone1_pois[0][1])]
            poi_vals[-3] = zone2_pois
            poi_vals[-2] = zone3_pois
            poi_vals[-1] = zone4_pois
            if start_at != None:
                if len(start_at) == 6:
                    for i in range(len(start_at)):
                        poi_idx = start_at[i]
                        if poi_idx != None and poi_idx in [-1, len(xs_extend)-1]:
                            Log.d(f"Not fine-tuning POI ID{i} as it is marked as not present in dataset.")
                            poi_vals[i] = len(xs_extend)-1
            min_freq = np.argmin(resonance_frequency[t_0p5:])
            max_diss = np.argmax(dissipation[t_0p5:])
            if t_blip3 - t_fill_start < 3.0:
                Log.e(TAG, "Bad run due to short duration between calculated start and stop times")
                is_good = False
            if min_freq < avg_start - t_0p5:
                Log.e(TAG, "Bad run due to 'resonance_frequency' minimum being after run start.")
                is_good = False
            if max_diss < avg_start - t_0p5:
                Log.e(TAG, "Bad run due to 'dissipation' minimum being after run start.")
                is_good = False
            if not is_good:
                poi_vals = -1
                return poi_vals
            fill_time_d = xs_extend[poi_vals[1]] - xs_extend[poi_vals[0][0][0]]
            exit_time_d = xs_extend[-1]- xs_extend[poi_vals[0][0][0]]
            xs_factor_d = int(exit_time_d / fill_time_d)
            Log.d(f"FILL_TIME: {fill_time_d}s")
            Log.d(f"EXIT_TIME: {exit_time_d}s")
            Log.d(f"XS_FACTOR: {xs_factor_d}x")
            if start_at != None:
                if len(start_at) == 6:
                    if start_at[-1] != None and start_at[-1] not in [len(xs_extend-1)-1, -1]:
                        if xs_factor_d < 75:
                            Log.w("Ignoring models partial fill assessment as user explictly provided an existing stop point.")
                            xs_factor_d = 100
            if xs_factor_d < 75:
                norm_smooth_diss_1st = self.normalize(super_smooth_diss_1st)
                norm_smooth_freq_1st = self.normalize(super_smooth_freq_1st)
                norm_smooth_diff_1st = self.normalize(super_smooth_diff_1st)
                norm_smooth_diss_2nd = self.normalize(super_smooth_diss_2nd)
                norm_smooth_freq_2nd = self.normalize(super_smooth_freq_2nd)
                norm_smooth_diff_2nd = self.normalize(super_smooth_diff_2nd)
                start_deriv = max(norm_smooth_diss_1st[avg_start], norm_smooth_freq_1st[avg_start], norm_smooth_diff_1st[avg_start],
                                norm_smooth_diss_2nd[avg_start], norm_smooth_freq_2nd[avg_start], norm_smooth_diff_2nd[avg_start])
                stop_deriv =  max(norm_smooth_diss_1st[avg_stop],  norm_smooth_freq_1st[avg_stop],  norm_smooth_diff_1st[avg_stop],
                                norm_smooth_diss_2nd[avg_stop],  norm_smooth_freq_2nd[avg_stop],  norm_smooth_diff_2nd[avg_stop])
                if stop_deriv < consider_points_above_pct and not poi_vals[-1] in [len(xs)-1, -1]:
                    Log.w("Partial fill run detected. Not all channels appear to be available in dataset. Please confirm points.")
                    poi_vals[-3] = poi_vals[-2].copy()
                    if True:
                        poi_vals[-2] = poi_vals[-1].copy()
                    else:
                        poi_vals[-2] = -1
                    poi_vals[-1] = -1
        except Exception as e:
            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            a_list = ['Traceback (most recent call last):']
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)
            raise e
        finally:
            if global_index == -1:
                return poi_vals
            self.model_result = poi_vals.copy()
            poi_vals.clear()
            if isinstance(self.model_result, list):
                self.model_select = []
                for point in self.model_result:
                    self.model_select.append(0)
                    if isinstance(point, list):
                        select_point = point[self.model_select[-1]]
                        select_index = select_point[0]
                        select_confidence = select_point[1]
                        poi_vals.append(select_index)
                    else:
                        poi_vals.append(point)
            elif self.model_result == -1:
                return -1
            if poi_vals[0] > len(xs) - 1 or xs[poi_vals[0]] != xs_extend[poi_vals[0]]:
                for i in range(len(poi_vals)):
                    poi_vals[i] = next(x for x,y in enumerate(xs) if y >= xs_extend[poi_vals[i]])
            if not self._plt_init:
                self._plt_init = True
                self.fig = plt.figure()
                self.ax1 = self.fig.add_subplot(111)
                self.ax2 = self.ax1
                self.ax3 = self.ax1
                mngr = plt.get_current_fig_manager()
                mngr.window.setGeometry(0,0,1920,1080)
            else:
                self.ax1.cla()
                self.ax2.cla()
                self.ax3.cla()
            self.ax1.set_title(data_title)
            self.ax1.plot(xs[poi_vals], self.normalize(ys[poi_vals]), 'go')
            for p in poi_vals:
                self.ax1.axvline(xs[p], color='blue', linestyle='-')
            self.ax1.plot(xs, self.normalize(ys), 'r.', alpha = 0.25)
            self.ax1.plot(xs, ys_smooth, linewidth=1, color='red')
            self.ax1.fill_between(xs, low1[0], up1[0], alpha=0.1, color = 'red')
            self.ax1.plot(xs, self.normalize(ys_freq), 'g.', alpha = 0.25)
            self.ax1.plot(xs, ys_freq_smooth, linewidth=1, color='green')
            self.ax1.fill_between(xs, low2[0], up2[0], alpha=0.1, color = 'green')
            self.ax1.plot(xs, self.normalize(ys_diff), 'b.', alpha = 0.25)
            self.ax1.plot(xs, ys_diff_smooth, linewidth=1, color='blue')
            self.ax1.fill_between(xs, low3[0], up3[0], alpha=0.1, color = 'blue')
            self.ax2.plot(xs, self.normalize(np.abs(accumulated_2nd), 0, 1), 'b-')
            self.ax3.fill_betweenx([0,1], t_quad0, t_quad1, alpha=0.1, color='yellow')
            self.ax3.fill_betweenx([0,1], t_quad1, t_quad2, alpha=0.1, color='red')
            self.ax3.fill_betweenx([0,1], t_quad3, t_quad4, alpha=0.1, color='green')
            self.ax3.fill_betweenx([0,1], t_quad5, t_quad6, alpha=0.1, color='blue')
            for tq in t_quads:
                self.ax3.axvline(tq, color='black', linestyle=":")
            zp = np.concatenate((zone1_possibilities, zone2_possibilities, zone3_possibilities, zone4_possibilities))
            for p in zp:
                self.ax3.plot(xs[int(p[0])], p[1], "b.")
            init_fill = False
            if not init_fill:
                self.ax3.plot(xs[:idx_quad5], self.normalize(super_smooth_diss_1st)[:idx_quad5], color='yellow', linestyle='solid')
                self.ax3.plot(xs[:idx_quad5], self.normalize(super_smooth_diss_2nd)[:idx_quad5], color='pink', linestyle='solid')
                self.ax3.plot(xs[:idx_quad5], self.normalize(super_smooth_freq_1st)[:idx_quad5], color='yellow', linestyle='dashed')
                self.ax3.plot(xs[:idx_quad5], self.normalize(super_smooth_freq_2nd)[:idx_quad5], color='pink', linestyle='dashed')
                self.ax3.plot(xs[:idx_quad5], self.normalize(super_smooth_diff_1st)[:idx_quad5], color='yellow', linestyle='dotted')
                self.ax3.plot(xs[:idx_quad5], self.normalize(super_smooth_diff_2nd)[:idx_quad5], color='pink', linestyle='dotted')
                self.ax3.plot(xs[idx_quad5:], self.normalize(np.abs(super_smooth_diss_1st))[idx_quad5:], color='yellow', linestyle='solid')
                self.ax3.plot(xs[idx_quad5:], self.normalize(np.abs(super_smooth_diss_2nd))[idx_quad5:], color='pink', linestyle='solid')
                self.ax3.plot(xs[idx_quad5:], self.normalize(np.abs(super_smooth_freq_1st))[idx_quad5:], color='yellow', linestyle='dashed')
                self.ax3.plot(xs[idx_quad5:], self.normalize(np.abs(super_smooth_freq_2nd))[idx_quad5:], color='pink', linestyle='dashed')
                self.ax3.plot(xs[idx_quad5:], self.normalize(np.abs(super_smooth_diff_1st))[idx_quad5:], color='yellow', linestyle='dotted')
                self.ax3.plot(xs[idx_quad5:], self.normalize(np.abs(super_smooth_diff_2nd))[idx_quad5:], color='pink', linestyle='dotted')
            self.ax3.axvline(xs[avg_start], color='yellow', linestyle='-')
            self.ax3.axvline(xs[avg_stop], color='yellow', linestyle='-')
            self.ax3.axhline(consider_points_above_pct, color='grey', linestyle=':', alpha=0.75)
            if idx_max_freq != -1:
                self.ax3.annotate("FREQ", (xs[idx_quad4], 1.00))
            if idx_max_diss != -1:
                self.ax3.annotate("DISS", (xs[idx_quad4], 0.99))
            if idx_max_diff != -1:
                self.ax3.annotate("DIFF", (xs[idx_quad4], 0.98))
            if init_fill:
                self.ax3.plot(xs_extend, norm_ys_diff)
                self.ax3.plot(xs_extend, norm_ys_diff_fit)
                self.ax3.plot(xs_extend, np.polyval(best_post_pointfit, xs_extend), color="black", linestyle="solid", alpha=0.5)
                self.ax3.plot(xs_extend, np.polyval(best_max_pointfit, xs_extend), color="black", linestyle="solid", alpha=0.5)
                self.ax3.plot(best_x_intersection, best_y_intersection, "x")
                self.ax3.axvline(xs_extend[debug_point1], color="black", alpha=0.5)
                self.ax3.axvline(xs_extend[debug_point2], color="black", alpha=0.5)
                self.ax3.axhline(break_diff_high, color="black", alpha=0.5)
                self.ax3.axhline(break_diff_low, color="black", alpha=0.5)
                self.ax3.annotate(str(int(method)), (xs_extend[debug_point1], break_diff_high), ha='left', va='top')
                self.ax3.annotate(str(best_num_fill_pts), (xs_extend[best_end_idx], up3_fill[0][best_end_idx]), ha='left', va='bottom')
                self.ax3.plot(xs_extend, norm_ys_diff_soft, color="black", linestyle='dotted')
                xlim_min = xs_extend[debug_point1] - 1
                xlim_max = xs_extend[best_post_idx] + 1
                ylim_min = -0.05
                ylim_max = np.amax(ys_diff_smooth_fill[debug_point1:best_post_idx]) + 0.05
                self.ax1.set_xlim(xlim_min, xlim_max)
                self.ax2.set_xlim(xlim_min, xlim_max)
                self.ax3.set_xlim(xlim_min, xlim_max)
                self.ax1.set_ylim(ylim_min, ylim_max)
                self.ax2.set_ylim(ylim_min, ylim_max)
                self.ax3.set_ylim(ylim_min, ylim_max)
            else:
                self.ax3.annotate(f"{xs_factor_d}x", (0.0, 1.5))
            try:
                Log.d("Actual POIs:", xs_extend[actual])
            except:
                Log.d("Actual POIs:", xs[actual])
            for p in actual:
                try:
                    self.ax1.axvline(xs_extend[p], color='red', linestyle='dotted')
                except:
                    self.ax1.axvline(xs[p], color='red', linestyle='dotted')
            run_model = False
            if run_model:
                plt.draw()
                plt.pause(0.05)
                if init_fill:
                    self.fig.savefig(f"model_init\{data_title}.png")
                else:
                    self.fig.savefig(f"model_run\{data_title}.png")
            else:
                input("Press any key to continue . . .")
        return poi_vals
    def run(self):
        plt.ion()
        path_root = "content/testing-2024-02-27/"
        bad_files = []
        with open("bad_files.txt", 'r') as f:
            bad_files = f.readlines()
        training_files = os.listdir(path_root)
        for i, f in enumerate(training_files):
            try:
                if f + '\n' in bad_files: raise ImportError()
                if "BAD" in f.upper(): raise ImportError()
                if f.endswith("_poi.csv"): raise ImportError()
                if f.endswith("_lower.csv"): raise ImportError()
                data_path = os.path.join(path_root, f)
                if os.path.isdir(data_path): raise ImportError()
                poi_path = os.path.join(path_root, training_files[i+1])
                actual_pois = np.loadtxt(poi_path).astype(int)
                last_pois = actual_pois
                stable_streak = 0
                for sf in np.arange(Constants.sf_min, Constants.sf_max, Constants.sf_step):
                    calculated_pois = self.IdentifyPoints(data_path, sf=sf, i=i, actual=actual_pois)
                    if not isinstance(calculated_pois, list):
                        print("Bad run!")
                        break
                    else:
                        print("Good run")
                        print(calculated_pois)
                        break
                    for p in range(len(calculated_pois)):
                        if abs(calculated_pois[p] - last_pois[p]) > 1.0:
                            stable_streak = 0
                            print("Streak reset!")
                            break
                    stable_streak += 1
                    print(f"Streak = {stable_streak}")
                    print(calculated_pois, actual_pois)
                    last_pois = calculated_pois
                    break
                    if stable_streak >= 5:
                        print("Points are stable. Moving on.")
                        break
            except ImportError:
                Log.w("Skipping BAD file:", f)
                continue
            except Exception as e:
                Log.e("ERROR:", e)
                raise e
        if self._headless:
            self.avg_time = self.sum_time / self.num_time
            self.avg_freq = self.sum_freq / self.num_freq
            self.avg_diss = self.sum_diss / self.num_diss
            Log.d("min_time:", self.min_time)
            Log.d("avg_time:", self.avg_time)
            Log.d("max_time:", self.max_time)
            Log.d("min_freq:", self.min_freq)
            Log.d("avg_freq:", self.avg_freq)
            Log.d("max_freq:", self.max_freq)
            Log.d("min_diss:", self.min_diss)
            Log.d("avg_diss:", self.avg_diss)
            Log.d("max_diss:", self.max_diss)
if __name__ == '__main__':
    ModelData().run()