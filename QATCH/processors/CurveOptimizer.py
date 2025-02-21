import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants
import random
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
                return relative_time.iloc[index], index + head_trim
            elif mode == 'right':
                # Report global max from downtrended data.
                index = np.argmax(adjusted_difference)
                Log.d(
                    TAG, f"Right bound found at time: {relative_time.iloc[index+ head_trim]}.")
                return relative_time.iloc[index], index + head_trim
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


TAG = "DropEffectCorrection"


class DropEffectCorrection(CurveOptimizer):
    """
    Class to mitigate drop effects in dissipation curves by detecting and correcting anomalies.

    Attributes:
        file_buffer: Input buffer containing the dataset.
        initial_diff_factor (float): Initial difference factor for curve computation.

    Methods:
        cancel_drop_effect(): Detects and corrects drop effects in the dissipation curve.
    """

    def __init__(self, file_buffer, initial_diff_factor: float = Constants.default_diff_factor):
        """
        Initializes the DropEffectMitigation with the provided file buffer and difference factor.

        Args:
            file_buffer: The data buffer containing dissipation data.
            initial_diff_factor (float): The initial factor for calculating the difference curve.

        Raises:
            ValueError: If required columns are missing or bounds are not properly defined.
        """
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
                TAG, f"Bounds are out of range of the dataframe or not initialized: {self._left_bound['index']} to {self._right_bound['index']}."
            )
            raise ValueError(
                f"Bounds are out of range of the dataframe or not initialized: {self._left_bound['index']} to {self._right_bound['index']}."
            )

        if "Difference" not in self._dataframe.columns:
            Log.d(
                TAG, f"'Difference' column not found. Computing difference curve with factor {initial_diff_factor}."
            )
            self._generate_curve(initial_diff_factor)

    def _detect_drop_effects(self, diff_offset: int = 2, threshold_factor: float = 3) -> list:
        """
        Detects non-normal y-deltas in the dissipation curve within the defined bounds.
        Instead of simply picking the two largest gradients, it computes the difference
        (with an offset of diff_offset) for each point in the region and flags those
        deviating strongly from a robust baseline.

        Args:
            diff_offset (int): The number of points to offset when computing differences.
            threshold_factor (float): Factor to scale the robust measure for outlier detection.

        Returns:
            list of tuples: Each tuple contains (global index, dissipation delta, resonance frequency delta)
                            for a detected jump.

        Raises:
            ValueError: If there are not enough points to compute differences.
        """
        left_idx = self._left_bound['index']
        right_idx = self._right_bound['index']

        if right_idx - left_idx < diff_offset + 1:
            Log.e(
                TAG, f"Not enough points in the specified region to compute differences with offset {diff_offset}.")
            raise ValueError(
                "Not enough points in the specified region to compute differences.")

        region_slice = slice(left_idx, right_idx + 1)
        diss_values = self._dataframe['Dissipation'].values[region_slice]
        rf_values = self._dataframe['Resonance_Frequency'].values[region_slice]

        # Compute differences with the specified offset to smooth out noise.
        local_indices = np.arange(diff_offset, len(diss_values))
        diffs = diss_values[local_indices] - \
            diss_values[local_indices - diff_offset]

        # Compute robust statistics: median and MAD.
        median_diff = np.median(diffs)
        mad_diff = np.median(np.abs(diffs - median_diff))
        threshold = threshold_factor * \
            (mad_diff if mad_diff > 0 else np.std(diffs))

        drop_effects = []
        for local_idx in local_indices:
            current_diff = diss_values[local_idx] - \
                diss_values[local_idx - diff_offset]
            if np.abs(current_diff - median_diff) > threshold:
                # Map back to the full dataframe index.
                global_idx = local_idx + left_idx
                rf_delta = rf_values[local_idx] - \
                    rf_values[local_idx - diff_offset]
                drop_effects.append((global_idx, current_diff, rf_delta))

        Log.d(
            TAG, f"Detected drop effects at indices {[de[0] for de in drop_effects]} with deltas: {[(de[1], de[2]) for de in drop_effects]}")
        return drop_effects

    def correct_drop_effects(self,
                             baseline_diss: float = None,
                             baseline_rf: float = None,
                             diss_threshold_ratio: float = 0.01,
                             rf_threshold_ratio: float = 0.01,
                             plot_corrections: bool = True) -> tuple:
        # Save original data for plotting.
        original_diss = self._dataframe['Dissipation'].values.copy()
        original_rf = self._dataframe['Resonance_Frequency'].values.copy()

        # If baselines are not provided, use a default value (e.g., from index 300).
        if baseline_diss is None:
            baseline_diss = original_diss[300]
        if baseline_rf is None:
            baseline_rf = original_rf[300]

        # Make working copies for corrections.
        corrected_diss = original_diss.copy()
        corrected_rf = original_rf.copy()

        # Get the drop effects from the detection method.
        # Each tuple is: (global_index, diss_delta, rf_delta)
        drop_effects = self._detect_drop_effects()

        # Ensure drop effects are sorted by their global index.
        drop_effects.sort(key=lambda x: x[0])

        # Process each detected drop effect sequentially.
        for i, drop in enumerate(drop_effects):
            idx, diss_delta, rf_delta = drop
            # Skip if the drop effect is at the very beginning.
            if idx <= 0:
                continue

            # Compute offsets so that the value at the drop index matches the previous (good) value.
            offset_diss = corrected_diss[idx - 1] - original_diss[idx]
            offset_rf = corrected_rf[idx - 1] - original_rf[idx]

            # Determine the segment end: from the current drop effect until the next drop effect,
            # or to the end of the data if this is the last drop.
            if i + 1 < len(drop_effects):
                next_idx = drop_effects[i + 1][0]
            else:
                next_idx = len(corrected_diss)

            # Apply the offset correction to the segment.
            corrected_diss[idx:next_idx] += offset_diss
            corrected_rf[idx:next_idx] += offset_rf

        # Get the region of interest from the object's bounds.
        left_idx = self._left_bound['index']
        right_idx = self._right_bound['index']

        # Ensure the indices are within the bounds of the data arrays.
        left_idx = max(0, left_idx)
        right_idx = min(len(corrected_diss), right_idx)

        # For Dissipation: enforce a running maximum with a relative threshold.
        running_max = corrected_diss[left_idx]
        for i in range(left_idx, right_idx):

            if corrected_diss[i] < running_max:

                gap = running_max - corrected_diss[i]
                allowed_gap = running_max * diss_threshold_ratio
                print(f'{gap}, {allowed_gap}')
                # Only partially correct if the gap exceeds the allowed fraction of the current maximum.
                if gap > allowed_gap:
                    corrected_diss[i] = random.uniform(
                        running_max - allowed_gap, running_max)
            else:
                running_max = corrected_diss[i]

        # For Resonance_Frequency: enforce a running minimum with a relative threshold.
        running_min = corrected_rf[left_idx]
        for i in range(left_idx, right_idx):
            if corrected_rf[i] > running_min:
                gap = corrected_rf[i] - running_min
                allowed_gap = running_min * rf_threshold_ratio
                # Only partially correct if the gap exceeds the allowed fraction of the current minimum.
                if gap < allowed_gap:
                    corrected_rf[i] = running_min
            else:
                running_min = corrected_rf[i]

        if plot_corrections:
            self._plot_corrections(original_diss, original_rf,
                                   corrected_diss, corrected_rf)

        return (corrected_diss, corrected_rf)

    def _plot_corrections(self, original_diss, original_rf, corrected_diss, corrected_rf):
        """
        Plots the original and corrected data for Dissipation and Resonance Frequency.

        Args:
            original_diss (np.array): The original dissipation values.
            original_rf (np.array): The original resonance frequency values.
            corrected_diss (np.array): The corrected dissipation values.
            corrected_rf (np.array): The corrected resonance frequency values.
        """
        indices = np.arange(len(original_diss))
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot for Dissipation.
        axs[0].plot(indices, original_diss,
                    label='Original Dissipation', color='blue')
        axs[0].plot(indices, corrected_diss,
                    label='Corrected Dissipation', color='red', linestyle='--')
        axs[0].set_title('Dissipation Correction')
        axs[0].set_xlabel('Index')
        axs[0].set_ylabel('Dissipation')
        axs[0].legend()

        # Plot for Resonance Frequency.
        axs[1].plot(indices, original_rf,
                    label='Original Resonance Frequency', color='blue')
        axs[1].plot(indices, corrected_rf,
                    label='Corrected Resonance Frequency', color='red', linestyle='--')
        axs[1].set_title('Resonance Frequency Correction')
        axs[1].set_xlabel('Index')
        axs[1].set_ylabel('Resonance Frequency')
        axs[1].legend()

        plt.tight_layout()
        plt.show()
