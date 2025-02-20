import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
                    TAG, f"Left bound found at time: {relative_time.iloc[index]}.")
                return relative_time.iloc[index], index + head_trim
            elif mode == 'right':
                # Report global max from downtrended data.
                index = np.argmax(adjusted_difference)
                Log.d(
                    TAG, f"Right bound found at time: {relative_time.iloc[index]}.")
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


TAG = "DifferenceFactorOptimizer"


class DifferenceFactorOptimizer(CurveOptimizer):
    def __init__(self, file_buffer, initial_diff_factor: float = Constants.default_diff_factor):
        super().__init__(file_buffer=file_buffer, initial_diff_factor=initial_diff_factor)

    def _objective(self, difference_factor: float, left_bound: float, right_bound: float) -> float:
        """
        Computes a penalty metric based on sudden changes in the first 20% of the ROI of the difference
        curve computed using the provided difference factor.

        The metric:
        - Computes the first differences for the first 20% of the ROI.
        - Determines the "typical" change (median) and a robust measure of dispersion (MAD).
        - Penalizes any differences that deviate from the typical change by more than a tolerance.

        Args:
            difference_factor (float): The factor used to generate the difference curve.
            left_bound (float): The left (start) time bound for the ROI.
            right_bound (float): The right (end) time bound for the ROI.

        Returns:
            float: The penalty metric (lower is better).

        Raises:
            Exception: If any error is encountered during computation.
        """
        try:
            # Generate the difference curve.
            # Assumed: column 0 is time and column 1 is the signal value.
            curve = self._generate_curve(difference_factor)

            # Extract the ROI based on time bounds.
            region_mask = (curve[:, 0] >= left_bound) & (
                curve[:, 0] <= right_bound)
            values = curve[region_mask, 1]

            # Check that we have enough data.
            if len(values) < 2:
                Log.w(TAG, "ROI is too short; returning a high penalty.")
                return np.inf

            # Use only the first 20% of the ROI.
            # Ensure at least 2 points for diff
            subset_length = max(2, int(0.3 * len(values)))
            sub_values = values[:subset_length]

            # Compute first differences on the restricted subset.
            diffs = np.diff(sub_values)
            if len(diffs) == 0:
                Log.w(
                    TAG, "Not enough differences in the subset; returning a high penalty.")
                return np.inf

            # Calculate the typical delta using the median.
            typical_delta = np.median(diffs)

            # Calculate the median absolute deviation (MAD) as a robust dispersion measure.
            mad = np.median(np.abs(diffs - typical_delta))

            # Set a tolerance. You might choose to scale the MAD if needed.
            # For example, tolerance = mad (strict) or tolerance = 1.5 * mad (more lenient)
            # Ensure a minimum tolerance if mad is very low
            tolerance = mad if mad > 0.05 else 0.05

            # Calculate the excess deviation beyond the tolerance.
            excess = np.maximum(np.abs(diffs - typical_delta) - tolerance, 0)
            penalty = np.sum(excess ** 2)
            Log.d(
                TAG, f"Penalty computed for difference factor {difference_factor}: {penalty}")
            return penalty

        except Exception as e:
            Log.e(TAG, f"Error calculating penalty metric: {e}")
            raise

    def _optimize_difference_factor(self) -> float:
        """
        Optimizes the difference factor by minimizing the penalty metric (which now only considers the
        first 20% of the ROI) using the L-BFGS-B method. This method enforces that the difference factor
        remains within the bounds [0.5, 3.0].

        Returns:
            float: The optimized difference factor (between 0.5 and 3.0).

        Raises:
            Exception: Any errors encountered during optimization are re-raised.
        """
        try:
            bounds = [(0.5, 2.0)]
            result = minimize(
                self._objective,
                self._initial_diff_factor,
                args=(self._left_bound['time'], self._right_bound['time']),
                method="L-BFGS-B",
                bounds=bounds
            )

            optimal_difference_factor = result.x[0]

            if 0.5 <= optimal_difference_factor <= 3.0:
                self._optimal_difference_factor = optimal_difference_factor
                Log.d(
                    TAG, f"Optimal difference factor found: {self._optimal_difference_factor}")
            else:
                self._optimal_difference_factor = max(
                    0.5, min(optimal_difference_factor, 3.0))
                Log.w(
                    TAG,
                    f"Optimal difference factor {optimal_difference_factor} out of bounds [0.5, 3.0], "
                    f"using {self._optimal_difference_factor}."
                )

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

    def _detect_drop_effect(self):
        """
        Detects drop effects within the defined bounds of the dissipation curve.

        Returns:
            tuple: Index of the drop, dissipation delta, difference delta, and resonance frequency delta.

        Raises:
            ValueError: If there are not enough points to compute derivatives.
        """
        bounds = (self._left_bound['index'], self._right_bound['index'])
        indices = np.arange(len(self._dataframe))

        regions = {col: self._dataframe[col].values[(indices >= bounds[0]) & (indices <= bounds[1])]
                   for col in ['Dissipation', 'Difference', 'Resonance_Frequency']}

        if len(regions['Dissipation']) < 3:
            Log.e(
                TAG, "Not enough points in the specified region to compute derivatives.")
            raise ValueError(
                "Not enough points in the specified region to compute derivatives.")

        max_idx = np.argmax(np.gradient(regions['Dissipation']))
        deltas = {key: region[max_idx] - region[max_idx - 2]
                  for key, region in regions.items()}

        Log.d(
            TAG, f"Drop effect detected at index {max_idx + bounds[0]} with deltas: {deltas}")

        return max_idx + bounds[0], deltas['Dissipation'], deltas['Difference'], deltas['Resonance_Frequency']

    def correct_drop_effect(self) -> np.ndarray:
        """
        Corrects the detected drop effect by adjusting the dissipation, difference, and resonance frequency values.

        Returns:
            np.ndarray: Updated values for dissipation, difference, and resonance frequency.
        """
        idx, diss_delta, diff_delta, rf_delta = self._detect_drop_effect()
        Log.d(
            TAG, f'Correction applied at index {idx}: diss_delta={diss_delta}, diff_delta={diff_delta}, rf_delta={rf_delta}'
        )

        for col, delta in zip(['Dissipation', 'Difference', 'Resonance_Frequency'], [diss_delta, diff_delta, rf_delta]):
            self._dataframe[col].values[idx + 1:] -= delta

        Log.d(TAG, f'Drop effect correction completed for index {idx}.')

        return tuple(self._dataframe[col].values for col in ['Dissipation', 'Difference', 'Resonance_Frequency'])
