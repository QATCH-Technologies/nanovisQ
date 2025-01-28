import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants

TAG = ["CurveOptimizer"]

""" The percentage of the run data to ignore from the head of a difference curve. """
HEAD_TRIM_PERCENTAGE = 0.05
""" The percentage of the run data to ignore from the tail of a difference curve. """
TAIL_TRIM_PERCENTAGE = 0.5
""" Sets the left bound of the region behind ROI. """
BASE_OFFSET = 0.005
""" Restricts the difference factor. """
DIFFERENCE_FACTOR_RESTRICTION = (0.5, 3.0)


class CurveOptimizer:
    def __init__(self, file_buffer, initial_diff_factor: float = Constants.default_diff_factor) -> None:
        """
        Initializes the optimizer utilities such as the data buffer, dataframe object, left and right ROI
        bounds, and initial difference factor.

        Given a file buffer and an initial difference factor (typically 2.0), the file buffer is initialized
        to a data buffer and data is loaded into a dataframe from this buffer.  If an error occurs, an exception
        is thrown at this point.  Initial difference curve is initialized to 'None' and and the left/right ROI
        time/index bounds are initialized to -1.  Optimal difference factor is set to None.

        Args:
            file_buffer: location to load data from.
            initial_diff_factor (float): initial difference factor to begin optimization at (Optional)

        Returns:
            None

        Raises:
            Exception if an error occurs during initailizing a file buffer.
        """
        self._file_buffer = file_buffer
        self._initial_diff_factor = initial_diff_factor
        try:
            self._data_buffer = self._initialize_file_buffer(file_buffer)
            self._dataframe = pd.read_csv(self._data_buffer)
            Log.d(
                TAG, f"Run data loaded successfully from {self._data_buffer}.")
        except Exception as e:
            Log.e(TAG, f"Failed to load data file: {e}")
            raise

        self._difference_curve = None
        self._left_bound = {"time": -1, "index": -1}
        self._right_bound = {"time": -1, "index": -1}
        self._optimal_difference_factor = None
        self._head_trim = -1
        self._set_bounds()

    def _initialize_file_buffer(self, file_buffer):
        """
        Initializes the input file buffer.

        Given a file buffer, this function checks if the file bufer has a sekable attribute
        and the file_buffer is seakable.  If it is, this function sets the data buffer to
        the head of the seekable object.  Otherwise, the function raises a value error stating
        the buffer is not seekable.  The file buffer is retruned on success.

        Args:
            file_buffer: the input data buffer to initailize

        Returns:
            file_buffer seekable object.

        Raises:
            ValueError: if the file buffer has not seek attribute or the object is not seekable.
        """
        Log.d(TAG, 'Initializing file buffer.')
        if not isinstance(file_buffer, str):
            if hasattr(file_buffer, "seekable") and file_buffer.seekable():
                file_buffer.seek(0)
            else:
                Log.e(TAG, "Cannot 'seek' file buffer stream.")
                raise ValueError("Cannot 'seek' file buffer stream.")
        return file_buffer

    @staticmethod
    def _normalize(data: np.ndarray) -> np.ndarray:
        """
        Utility function to normalize an np.ndarray object.

        This function performs min-max normalization to normalize the input
        array between 0 and 1.  If an error occurs durring normalization, the fuction raises
        the error to the caller.

        Args:
            data (np.ndarray): The 1xN input array to normalize.

        Returns:
            np.ndarray: Input array normalized between 0 and 1.

        Raises:
            Exception to caller if normalization fails.
        """
        try:
            min_val = np.min(data)
            max_val = np.max(data)
            normalized = (data - min_val) / (max_val - min_val)
            Log.d(TAG, "Data normalized successfully.")
            return normalized
        except Exception as e:
            Log.e(TAG, f"Error normalizing data: {e}")
            raise

    def _generate_curve(self, difference_factor: float) -> np.ndarray:
        """
        Function to compute the difference curve with a parameterized difference factor.

        Given a difference factor and a dataset containing Dissipation, Resonance_Frequency, and Relative_time
        data as a pandas dataframe, the function attempts to compute the difference curve between the Dissipation
        and Resonance_Frequency data using Relative_time as the X-Axis component.

        Args:
            difference_factor (float): A difference factor value used to compute the difference curve.

        Returns:
            np.ndarray: A 1xN array of difference curve Y values.

        Raises:
            ValueError: if the input dataframe does not contain the requeired columns "Dissipation",
                "Resonance_Frequency", and "Relative_time"
        """
        try:
            difference_factor = difference_factor or self._initial_diff_factor
            required_columns = ["Dissipation",
                                "Resonance_Frequency", "Relative_time"]
            if not all(column in self._dataframe.columns for column in required_columns):
                Log.e(
                    TAG, f"Input CSV must contain the following columns: {required_columns}")
                raise ValueError(
                    f"Input CSV must contain the following columns: {required_columns}")

            xs = self._dataframe["Relative_time"]
            i = next(x for x, t in enumerate(xs) if t > 0.5)
            j = next(x for x, t in enumerate(xs) if t > 2.5)

            avg_resonance_frequency = self._dataframe["Resonance_Frequency"][i:j].mean(
            )
            avg_dissipation = self._dataframe["Dissipation"][i:j].mean()

            self._dataframe["ys_diss"] = (
                self._dataframe["Dissipation"] - avg_dissipation) * avg_resonance_frequency / 2
            self._dataframe["ys_freq"] = avg_resonance_frequency - \
                self._dataframe["Resonance_Frequency"]
            self._dataframe["Difference"] = self._dataframe["ys_freq"] - \
                difference_factor * self._dataframe["ys_diss"]

            self._difference_curve = np.stack(
                (self._dataframe["Relative_time"].values, self._dataframe["Difference"].values), axis=1)
            Log.d(TAG, "Curve generated successfully.")
            return self._difference_curve
        except Exception as e:
            Log.e(TAG, f"Error generating curve: {e}")
            raise

    def _find_region(self, difference: np.ndarray, relative_time: pd.Series, mode: str) -> float:
        """
        Searches for the initial fill region to perform smoothness optimization at.

        Given a difference data as a 1xN dataset, a relative time series as a 1xN dataset and a mode of operation
        ('right' or 'left'), this function attempts to dynamically search for the initail fill region to optimize
        the smootheness of.  First, 5% of the head of the difference and relative time data is trimmed off followed
        by the last 50% of the difference and relative time data.  This avoids any unecessary noise in the run data
        as well as unimportant data after the initial fill region. Next, a trend is established between the maximal
        difference value and the length of the run.  This trend is applied to the difference data to down trend the
        entire dataset making the initial fill peak more prominent over the baseline or any later data.

        Operating in 'left' mode requires a right bound be already established, the left bound is established as follows:
        Difference data and relative time data is trimmed on the right side again by the right bound.  The trend is then
        re-established on the trimmed data and applied to the already adjusted difference curve to make the base of the
        initial fill region more prominent.  The left-bound is returned as the global minima of this adjusted data stepped
        back by a small offset (0.5% of the remaining data) to make sure that the entire initial fill trough is captured.

        Operating in 'right' mode sets the right bound as follows: The global maxima of the adjusted difference factor is
        returned as the right bound of this data set.

        Args:
            difference (np.ndarray): A 1xN array containing 'Difference' data.
            relative_time (pd.Series): A 1xN pandas series containing 'Relative_time' data.
            mode (str): A mode of operation with acceptable values of 'right' or 'left' (i.e. which bound to compute).

        Returns:
            tuple(float, int): a tuple containing the 'Relative_time' at the bound and the index of the bound.

        Raises
            ValueError: If the parameterized mode is not 'left' or 'right'.
        """
        try:
            # Trim head and tail off difference and relative time data
            head_trim = int(len(difference) * HEAD_TRIM_PERCENTAGE)
            self._head_trim = head_trim
            tail_trim = int(len(difference) * TAIL_TRIM_PERCENTAGE)
            relative_time = relative_time.iloc[head_trim:tail_trim]
            difference = difference[head_trim:tail_trim]

            # Determine a downward trend over the difference data
            trend = np.linspace(0, difference.max(), len(difference))

            # Apply the trend
            adjusted_difference = difference - trend

            if mode == "left":
                # NOTE: Requires right bound already be set.  Trim again at the right bound index.
                relative_time = relative_time.iloc[:self._right_bound["index"]]
                difference = difference[:self._right_bound["index"]]

                # Recompute trend over shortened data.
                trend = np.linspace(0, difference.max(), len(difference))

                # Reapply trend to shortned data
                adjusted_difference = difference - trend

                # Report global minima over shortened data.
                index = int(np.argmin(adjusted_difference) -
                            (len(relative_time) * BASE_OFFSET))
                Log.d(
                    TAG, f"Left bound found at time: {relative_time.iloc[index+ head_trim]}.")
                return relative_time.iloc[index + head_trim], index + head_trim
            elif mode == 'right':
                # Report global max from downtrended data.
                index = np.argmax(adjusted_difference)
                Log.d(
                    TAG, f"Right bound found at time: {relative_time.iloc[index+ head_trim]}.")
                return relative_time.iloc[index + head_trim], index + head_trim
            else:
                raise ValueError(f"Invalid search bound requested {mode}.")
        except Exception as e:
            Log.e(TAG, f"Error finding optimization region: {e}")
            raise

    def _set_bounds(self) -> None:
        Log.d(TAG, "Setting region bounds.")
        # Generate initial curve.
        self._generate_curve(self._initial_diff_factor)

        # Establish right bound
        # IMPORTANT: this must be done before the left bound is established.
        rb_time, rb_idx = self._find_region(
            self._dataframe["Difference"].values, self._dataframe["Relative_time"], mode="right")
        self._right_bound["time"] = rb_time
        self._right_bound["index"] = rb_idx

        # Establish left bound
        lb_time, lb_idx = self._find_region(
            self._dataframe["Difference"].values, self._dataframe["Relative_time"], mode="left")
        self._left_bound["time"] = lb_time
        self._left_bound["index"] = lb_idx


class DifferenceFactorOptimizer(CurveOptimizer):
    def __init__(self, file_buffer, initial_diff_factor: float = Constants.default_diff_factor):
        super().__init__(file_buffer=file_buffer, initial_diff_factor=initial_diff_factor)

    def _objective(self, difference_factor: float, left_bound: float, right_bound: float) -> float:
        """
        The target function to optimize.

        This function computes the smoothness over a left/right bounded region of difference data.
        Smoothness is a measure of the sum of the square of the slopes times the number of
        of time deltas in the left/right bounded region of difference data.

        Args:
            differnce_factor (float): The difference factor to compute the difference curve with.
            left_bound (float): A Relative_time left bound of the difference data.
            right_bound (float): A Relative_time right bound of the difference data.

        Returns:
            float: A measure of the smoothness over the region bounded by left_bound/right_bound

        Raises:
            Errors raised to caller.
        """
        try:
            # Generate the difference curve 1xN array.
            curve = self._generate_curve(difference_factor)

            # Mask of the Region of Interest (ROI) by the parameterized left and right bound.
            region_mask = (curve[:, 0] >= left_bound) & (
                curve[:, 0] <= right_bound)

            # Get the differnce values and time values from this ROI.
            differences = curve[region_mask, 1]
            time_points = curve[region_mask, 0]

            # Compute the slopes of the difference region.
            slopes = np.diff(differences) / np.diff(time_points)

            # Compute the deltas between time points.
            step_sizes = np.diff(time_points)

            # Compute the objective, smoothness metric.
            smoothness_metric = np.sum((slopes ** 2) * step_sizes)

            Log.d(TAG, f"Smoothness objective calculated: {smoothness_metric}")
            return smoothness_metric
        except Exception as e:
            Log.e(TAG, f"Error calculating smoothness objective: {e}")
            raise

    def _optimize_difference_factor(self) -> float:
        """
        The primary function to perform the data i/o, bounding, and optimization processes.

        This function generates and initial difference curve using the initial differnce factor.  Using
        this difference curve, left and right time and index bounds are established and the left_bound/right_bound
        attributes are set.  Next, the function uses scipy.optimize.minimize to find a maximally smooth differnce factor
        using the previously set time bounds, initial difference factor (typically 2.0).  The optimization method is Nelder-Mead
        (https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method).  The result is extracted from this minimization process.  If
        the optimal factor is between 0.0 and 3.0, it is accepted and returned to the caller.  Otherwise, the default difference
        factor of 2.0 is returned.

        TODO: Look into if we should return a minimal or maximal value of 0.0 or 3.0 instead of 2.0  default.

        Args:
            None

        Returns:
            float: An optimized difference factor if the result of optimization is within bounds.  Otherwise, the
                default difference factor of 2.0 is reported.

        Raises:
            Exceptions reaised to caller.
        """
        try:
            # Perform optimization
            result = minimize(
                self._objective,
                self._initial_diff_factor,
                args=(self._left_bound['time'], self._right_bound['time']),
                method="Nelder-Mead"
            )

            # Report restults to caller
            optimal_difference_factor = result.x[0]
            if DIFFERENCE_FACTOR_RESTRICTION[0] < optimal_difference_factor < DIFFERENCE_FACTOR_RESTRICTION[1]:
                Log.d(
                    TAG, f"Optimal difference factor: {self._optimal_difference_factor}")
                self._optimal_difference_factor = optimal_difference_factor
            else:
                self._optimal_difference_factor = (
                    DIFFERENCE_FACTOR_RESTRICTION[0]
                    if abs(optimal_difference_factor - DIFFERENCE_FACTOR_RESTRICTION[0])
                    < abs(optimal_difference_factor - DIFFERENCE_FACTOR_RESTRICTION[1])
                    else DIFFERENCE_FACTOR_RESTRICTION[1]
                )
                Log.w(
                    TAG, f"Optimal difference factor {optimal_difference_factor} out of range {DIFFERENCE_FACTOR_RESTRICTION}, using {self._optimal_difference_factor}.")
            return self._optimal_difference_factor
        except Exception as e:
            Log.e(TAG, f"Error optimizing difference factor: {e}")
            raise

    def optimize(self) -> tuple:
        """
        Main entry point for calling class (Anlayze.py)

        Runs the optimize differnece factor function and reports the results of the optimization
        along with the left and right time bounds of the optimziation region.

        Args:
            None

        Returns:
            tuple(float, float, float): Returns the optimizatized difference factor along
                with the the left and right time bounds of the optimization reigon

        Raises:
            Errors raised to caller.
        """
        try:
            result = self._optimize_difference_factor()
            # Repair later drop effects.
            if result:
                Log.d(
                    TAG, f"Run completed successfully. Results: {result}, left-bound: {self._left_bound['time']}, right-bound: {self._right_bound['time']}.")
                return result, self._left_bound['time'], self._right_bound['time']
            else:
                Log.d(
                    TAG, f"Run completed unsucessful. Results: {result}, left-bound: {self._left_bound['time']}, right-bound: {self._right_bound['time']}.")
                return Constants.default_diff_factor, self._left_bound['time'], self._right_bound['time']
        except Exception as e:
            Log.e(TAG, f"Run failed: {e}")
            raise


class DropEffectMitigation(CurveOptimizer):

    def __init__(self, file_buffer, initial_diff_factor: float = Constants.default_diff_factor):
        super().__init__(file_buffer=file_buffer, initial_diff_factor=initial_diff_factor)
        if "Dissipation" not in self._dataframe.columns:
            Log.e(TAG, "The dataframe does not contain a 'Dissipation' column.")
            raise ValueError(
                "The dataframe does not contain a 'Dissipation' column.")

        if not ("index" in self._left_bound and "index" in self._right_bound):
            Log.e(TAG, "Bounds must be properly defined with 'index' keys.")
            raise ValueError(
                "Bounds must be properly defined with 'index' keys.")

        if (self._left_bound['index'] < 0 or self._right_bound['index'] >= len(self._dataframe)):
            Log.e(
                TAG, f"Bounds are out of range of the dataframe or not initialized {self._left_bound['index']} to {self._right_bound['index']}.")
            raise ValueError(
                f"Bounds are out of range of the dataframe or not initialized {self._left_bound['index']} to {self._right_bound['index']}.")

    def _detect_drop_effect(self, std_threshold: int, window_size: int):
        # Extract data
        data = self._dataframe["Dissipation"].values

        # Filter data within the bounded region [lb, rb]
        bounded_data = data[self._left_bound['index']: self._right_bound['index'] + 1]
        if len(bounded_data) < window_size:
            Log.e(TAG, "Window size is larger than the bounded data region.")
            raise ValueError(
                "Window size is larger than the bounded data region.")

        # Calculate deltas over a sliding window
        deltas = []
        indices = []

        for i in range(len(bounded_data) - window_size):
            window = bounded_data[i:i + window_size]
            delta = np.abs(window[-1] - window[0])
            deltas.append(delta)
            indices.append(i + window_size // 2)

        deltas = np.array(deltas)
        indices = np.array(indices,  dtype=int)

        # Calculate the threshold for outliers and identify points outside of
        # the std_threshold.
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        threshold = mean_delta + (std_threshold * std_delta)
        outliers = deltas > threshold
        outlier_indices = indices[outliers]
        return outlier_indices, deltas[outlier_indices]

    def cancel_drop_effect(self, std_theshold: int = 3, window_size: int = 20) -> np.ndarray:
        if not isinstance(std_theshold, int) or std_theshold <= 0:
            Log.e(TAG, "std_theshold must be a positive integer.")
            raise ValueError("std_theshold must be a positive integer.")
        if not isinstance(window_size, int) or window_size <= 0:
            Log.e(TAG, "window_size must be a positive integer.")
            raise ValueError("window_size must be a positive integer.")
        data = self._dataframe["Dissipation"].values

        outlier_indices, deltas = self._detect_drop_effect(
            std_threshold=std_theshold, window_size=window_size)

        # TODO: Need some way of offesting the outlier_indices by the self._left_bound['index'] value
        # print(outlier_indices, len(data), deltas)
        max_delta = max(deltas[outlier_indices])
        max_delta_index = outlier_indices[deltas[outlier_indices].argmax()]

        for i, pt in enumerate(data):
            if i > max_delta_index:
                data[i] = data[i] - max_delta

        plt.figure()
        plt.plot(self._dataframe["Dissipation"],
                 color='grey', linestyle='dotted', label='Original data')
        plt.axvline(self._left_bound['index'], c='r', label='Left-bound')
        plt.axvline(self._right_bound['index'], c='b', label='Right-bound')
        plt.scatter(outlier_indices,
                    self._dataframe["Dissipation"].values[outlier_indices])
        self._dataframe["Dissipation"] = data
        plt.plot(self._dataframe["Dissipation"],
                 color='green', label='Canceled data')
        plt.legend()
        plt.title("Canceled drop effect")
        plt.show()
        return self._dataframe["Dissipation"].values

    def interpolate_drop_effect(self, std_theshold: int = 3, window_size: int = 20) -> np.ndarray:

        # Input validation
        if not isinstance(std_theshold, int) or std_theshold <= 0:
            Log.e(TAG, "std_theshold must be a positive integer.")
            raise ValueError("std_theshold must be a positive integer.")
        if not isinstance(window_size, int) or window_size <= 0:
            Log.e(TAG, "window_size must be a positive integer.")
            raise ValueError("window_size must be a positive integer.")

        # Extract data
        data = self._dataframe["Dissipation"].values

        # Filter data within the bounded region [lb, rb]
        bounded_data = data[self._left_bound['index']                            :self._right_bound['index']]
        outlier_indices, _ = self._detect_drop_effect(
            std_threshold=std_theshold, window_size=window_size)
        # Repair the outliers
        repaired_bounded_data = bounded_data.copy()

        x_bounded = np.arange(
            self._left_bound['index'], self._right_bound['index'] + 1)
        if len(outlier_indices) > 0:
            valid_indices = np.setdiff1d(
                np.arange(len(bounded_data)), outlier_indices)

            # Ensure there are enough valid points for interpolation
            if len(valid_indices) < 2:
                Log.e(
                    TAG, "Not enough valid points for interpolation after outlier removal.")
                raise ValueError(
                    "Not enough valid points for interpolation after outlier removal.")

            valid_x = x_bounded[valid_indices]
            valid_y = bounded_data[valid_indices]
            interp_func = interp1d(
                valid_x, valid_y, kind="linear", fill_value="extrapolate")
            repaired_bounded_data[outlier_indices] = interp_func(
                x_bounded[outlier_indices])

        # Update the original data with the repaired region
        data[self._left_bound['index']:self._right_bound['index']] = repaired_bounded_data
        plt.figure()
        plt.plot(self._dataframe["Dissipation"],
                 color='grey', linestyle='dotted', label='Original data')
        plt.axvline(self._left_bound['index'], c='r', label='Left-bound')
        plt.axvline(self._right_bound['index'], c='b', label='Right-bound')
        plt.scatter(outlier_indices,
                    self._dataframe["Dissipation"].values[outlier_indices])
        self._dataframe["Dissipation"] = data
        plt.plot(self._dataframe["Dissipation"],
                 color='green', label='Interpolated data')
        plt.legend()
        plt.title("Interpolated drop effect")
        plt.show()

        return self._dataframe["Dissipation"].values
