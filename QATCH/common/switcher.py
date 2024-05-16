from QATCH.core.constants import Constants

TAG = "[Switcher]"

###############################################################################
# Switches from overtone frequency to frequency range for 10MHz QC Sensor
# Returns other parameters that are used for processing
###############################################################################
class Overtone_Switcher_10MHz:

    def __init__(self,peak_frequencies = None, left_bounds = None, right_bounds = None):
        self.peak_frequencies = peak_frequencies
        self.left_bounds = left_bounds
        self.right_bounds = right_bounds

    # from fundamental frequency to the 5th overtone
    def overtone10MHz_to_freq_range(self, argument):
        self.arg = argument # save this for later
        method_name = 'overtone_' + str(self.arg)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: None)
        # Call the method as we return it
        return method()

    def overtone_0(self):
        # fundamental frequency
        name = "fundamental"
        start = self.peak_frequencies[self.arg] - self.calc_distance_left(Constants.max_drift_l_hz)
        stop  = self.peak_frequencies[self.arg] + self.calc_distance_right(Constants.max_drift_r_hz)
        return name ,self.peak_frequencies[self.arg], start, stop, Constants.SG_window_size, Constants.Spline_factor

    def overtone_1(self):
        # 3rd Overtone
        name = "3rd Overtone"
        start = self.peak_frequencies[self.arg] - self.calc_distance_left(Constants.max_drift_l_hz)
        stop  = self.peak_frequencies[self.arg] + self.calc_distance_right(Constants.max_drift_r_hz)
        return name, self.peak_frequencies[self.arg], start, stop, Constants.SG_window_size, Constants.Spline_factor

    def overtone_2(self):
        # 5th Overtone
        name = "5th Overtone"
        start = self.peak_frequencies[self.arg] - self.calc_distance_left(Constants.max_drift_l_hz)
        stop  = self.peak_frequencies[self.arg] + self.calc_distance_right(Constants.max_drift_r_hz)
        return name, self.peak_frequencies[self.arg], start, stop, Constants.SG_window_size, Constants.Spline_factor

    # Helper functions to make code cleaner
    def calc_distance_left(self, left_max):
        return (self.left_bounds[self.arg] if not self.left_bounds is None and self.left_bounds[self.arg] < left_max else left_max)

    def calc_distance_right(self, right_max):
        return (self.right_bounds[self.arg] if not self.right_bounds is None and self.left_bounds[self.arg] < right_max else right_max)


###############################################################################
# Switches from overtone frequency to frequency range for 5MHz QC Sensor
###############################################################################
class Overtone_Switcher_5MHz:

    def __init__(self,peak_frequencies = None, left_bounds = None, right_bounds = None):
        self.peak_frequencies = peak_frequencies
        self.left_bounds = left_bounds
        self.right_bounds = right_bounds

    # from fundamental frequency to the 9th overtone
    def overtone5MHz_to_freq_range(self, argument):
        self.arg = argument # save this for later
        method_name = 'overtone_' + str(self.arg)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: None)
        # Call the method as we return it
        return method()

    def overtone_0(self):
        # fundamental frequency
        name = "fundamental"
        start = self.peak_frequencies[self.arg] - self.calc_distance_left(Constants.max_drift_l_hz)
        stop  = self.peak_frequencies[self.arg] + self.calc_distance_right(Constants.max_drift_r_hz)
        return name, self.peak_frequencies[self.arg], start, stop, Constants.SG_window_size, Constants.Spline_factor

    def overtone_1(self):
        # 3rd Overtone
        name = "3rd Overtone"
        start = self.peak_frequencies[self.arg] - self.calc_distance_left(Constants.max_drift_l_hz)
        stop  = self.peak_frequencies[self.arg] + self.calc_distance_right(Constants.max_drift_r_hz)
        return name, self.peak_frequencies[self.arg], start, stop, Constants.SG_window_size, Constants.Spline_factor

    def overtone_2(self):
        # 5th Overtone
        name = "5th Overtone"
        start = self.peak_frequencies[self.arg] - self.calc_distance_left(Constants.max_drift_l_hz)
        stop  = self.peak_frequencies[self.arg] + self.calc_distance_right(Constants.max_drift_r_hz)
        return name, self.peak_frequencies[self.arg],start,stop, Constants.SG_window_size, Constants.Spline_factor

    def overtone_3(self):
        # 7th Overtone
        name = "7th Overtone"
        start = self.peak_frequencies[self.arg] - self.calc_distance_left(Constants.max_drift_l_hz)
        stop  = self.peak_frequencies[self.arg] + self.calc_distance_right(Constants.max_drift_r_hz)
        return name, self.peak_frequencies[self.arg],start,stop, Constants.SG_window_size, Constants.Spline_factor

    def overtone_4(self):
        # 9th Overtone
        name = "9th Overtone"
        start = self.peak_frequencies[self.arg] - self.calc_distance_left(Constants.max_drift_l_hz)
        stop  = self.peak_frequencies[self.arg] + self.calc_distance_right(Constants.max_drift_r_hz)
        return name, self.peak_frequencies[self.arg], start, stop, Constants.SG_window_size, Constants.Spline_factor

    # Helper functions to make code cleaner
    def calc_distance_left(self, left_max):
        return (self.left_bounds[self.arg] if not self.left_bounds is None and self.left_bounds[self.arg] < left_max else left_max)

    def calc_distance_right(self, right_max):
        return (self.right_bounds[self.arg] if not self.right_bounds is None and self.left_bounds[self.arg] < right_max else right_max)
