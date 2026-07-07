"""
curve_optimizer.py

This module provides utilities for analyzing, optimizing, and correcting curves derived from run data,
particularly focusing on dissipation and resonance frequency measurements. It is designed to work with CSV
input files and leverages popular scientific libraries such as pandas, numpy, matplotlib, and SciPy for data
processing, numerical operations, plotting, and optimization.

Author(s):
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-04-29

Version:
    v12.0.1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants

""" The percentage of the run data to ignore from the head of a difference curve. """
HEAD_TRIM_PERCENTAGE = 0.05
""" The percentage of the run data to ignore from the tail of a difference curve. """
TAIL_TRIM_PERCENTAGE = 0.5
""" Restricts the difference factor. """
DIFFERENCE_FACTOR_RESTRICTION = (0.5, 3.0)
""" The relative time (in seconds) for where to place the right bound of region. """
REGION_RIGHT_BOUND_SEC = 1.0

##########################################
# CHANGE THESE
##########################################
# This is a percentage setback from the initial drop application.  Increasing this value in the range (0,1) will
# move the left side of the correction zone further from the initial application of the drop as a percentage of the
# index where the drop is applied.
INIT_DROP_SETBACK = 0.000

# Adjust the detected left-bound index by subtracting a proportion (BASE_OFFSET) of the data length,
# effectively shifting the boundary left to better capture the true start of the drop.
BASE_OFFSET = 0.005

# This is the detection senstivity for the dissipation and RF drop effects.  Increasing these values should independently
# increase how large a delta needs to be in order to be counted as a drop effect essentially correcting fewer deltas resulting
# in a less aggressive correction.
STARTING_THRESHOLD_FACTOR = 50

##########################################
# CHANGE THESE
##########################################


class CurveOptimizer:

    TAG = "[CurveOptimizer]"

    def __init__(
        self,
        file_path: str,
        file_buffer,
        initial_diff_factor: float = Constants.default_diff_factor,
        bounds: list = [],
    ) -> None:
        """
        Initializes the optimizer utilities such as the data buffer, dataframe object, left and right ROI
        bounds, and initial difference factor.

        Given a file buffer and an initial difference factor (typically 2.0), the file buffer is initialized
        to a data buffer and data is loaded into a dataframe from this buffer.  If an error occurs, an exception
        is thrown at this point.  Initial difference curve is initialized to 'None' and and the left/right ROI
        time/index bounds are initialized to -1.  Optimal difference factor is set to None.

        Args:
            file_path: string of loaded data path.
            file_buffer: location to load data from or a byte buffer stream of the opened file buffer.
            initial_diff_factor (float): initial difference factor to begin optimization at (Optional)
            bounds (list): POI values given by QModel or user-input (Optional)

        Returns:
            None

        Raises:
            Exception if an error occurs during initailizing a file buffer.
        """
        self._file_path = file_path
        self._file_buffer = file_buffer
        self._initial_diff_factor = initial_diff_factor
        try:
            self._data_buffer = self._initialize_file_buffer(file_buffer)
            self._dataframe = pd.read_csv(self._data_buffer)
            Log.d(self.TAG, f"Run data loaded successfully from {self._strip_filename(file_path)}.")
        except Exception as e:
            Log.e(self.TAG, f"Failed to load data file: {e}")
            raise

        self._difference_curve = None
        self._left_bound = {"time": -1, "index": -1}
        self._right_bound = {"time": -1, "index": -1}
        self._optimal_difference_factor = None
        self._head_trim = -1
        self._set_bounds(bounds=bounds)

    def _strip_filename(self, full_path):
        # Convert a full file path to just the file name part.
        # Note: handles either separator, returns no extension
        # example: "/a/b/c/test.csv" --> "test"
        return full_path.split("\\")[-1].split("/")[-1].split(".")[0]

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
        Log.d(self.TAG, "Initializing file buffer.")
        if not isinstance(file_buffer, str):
            if hasattr(file_buffer, "seekable") and file_buffer.seekable():
                file_buffer.seek(0)
            else:
                Log.e(self.TAG, "Cannot 'seek' file buffer stream.")
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
            Log.d(CurveOptimizer.TAG, "Data normalized successfully.")
            return normalized
        except Exception as e:
            Log.e(CurveOptimizer.TAG, f"Error normalizing data: {e}")
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
            required_columns = ["Dissipation", "Resonance_Frequency", "Relative_time"]
            if not all(column in self._dataframe.columns for column in required_columns):
                Log.e(self.TAG, f"Input CSV must contain the following columns: {required_columns}")
                raise ValueError(
                    f"Input CSV must contain the following columns: {required_columns}"
                )

            xs = self._dataframe["Relative_time"]
            i = next((x for x, t in enumerate(xs) if t > 0.5), 0)
            j = next((x for x, t in enumerate(xs) if t > 2.5), 1)
            if i == j:
                j = next((x for x, t in enumerate(xs) if t > xs[j] + 2.0), j + 1)

            avg_resonance_frequency = self._dataframe["Resonance_Frequency"][i:j].mean()
            avg_dissipation = self._dataframe["Dissipation"][i:j].mean()

            self._dataframe["ys_diss"] = (
                (self._dataframe["Dissipation"] - avg_dissipation) * avg_resonance_frequency / 2
            )
            self._dataframe["ys_freq"] = (
                avg_resonance_frequency - self._dataframe["Resonance_Frequency"]
            )
            self._dataframe["Difference"] = (
                self._dataframe["ys_freq"] - difference_factor * self._dataframe["ys_diss"]
            )

            self._difference_curve = np.stack(
                (self._dataframe["Relative_time"].values, self._dataframe["Difference"].values),
                axis=1,
            )
            Log.d(self.TAG, "Curve generated successfully.")
            return self._difference_curve
        except Exception as e:
            Log.e(self.TAG, f"Error generating curve: {e}")
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
                relative_time = relative_time.iloc[: self._right_bound["index"]]
                difference = difference[: self._right_bound["index"]]

                # Recompute trend over shortened data.
                trend = np.linspace(0, difference.max(), len(difference))

                # Reapply trend to shortned data
                adjusted_difference = difference - trend

                # Report global minima over shortened data.
                index = int(np.argmin(adjusted_difference) + (len(relative_time) * BASE_OFFSET))
                import math

                index = index + math.floor(INIT_DROP_SETBACK * index)
                Log.d(self.TAG, f"Left bound found at time: {relative_time.iloc[index]}.")
                return relative_time.iloc[index], index + head_trim
            elif mode == "right":
                # Report global max from downtrended data.
                index = int(np.argmax(adjusted_difference) + 0.1 * np.argmax(adjusted_difference))
                Log.d(self.TAG, f"Right bound found at time: {relative_time.iloc[index]}.")
                return relative_time.iloc[index], index + head_trim
            else:
                raise ValueError(f"Invalid search bound requested {mode}.")
        except Exception as e:
            Log.e(self.TAG, f"Error finding optimization region: {e}")
            raise

    def _set_bounds(self, bounds: list = []) -> None:
        Log.d(self.TAG, "Setting region bounds.")

        # Generate initial curve.
        self._generate_curve(Constants.default_diff_factor)

        # Use the POI selection to set left/right bounds (if given).
        if bounds:
            left_index = bounds[0]
            self._left_bound["time"] = self._dataframe["Relative_time"].iloc[left_index]
            self._left_bound["index"] = left_index

            # Get index of first timestamp value > seconds.
            right_mask = (
                self._dataframe["Relative_time"] > REGION_RIGHT_BOUND_SEC + self._left_bound["time"]
            )

            # Check if there is a time that meets the criteria.
            if right_mask.any():
                right_index = right_mask.idxmax()
                self._right_bound["time"] = self._dataframe["Relative_time"].iloc[right_index]
                self._right_bound["index"] = right_index
                return

            # Warn user that there were no times large enough to calculate the right boundary.
            Log.d(
                self.TAG,
                f"Not using given `bounds`. No times greater than `START + {REGION_RIGHT_BOUND_SEC}s`. Attempting to detect region...",
            )

        # Else (no usable bounds), use default left/right boundary detection.

        # Establish right bound
        # IMPORTANT: this must be done before the left bound is established.
        rb_time, rb_idx = self._find_region(
            self._dataframe["Difference"].values, self._dataframe["Relative_time"], mode="right"
        )
        self._right_bound["time"] = rb_time
        self._right_bound["index"] = rb_idx

        # Establish left bound
        lb_time, lb_idx = self._find_region(
            self._dataframe["Difference"].values, self._dataframe["Relative_time"], mode="left"
        )
        self._left_bound["time"] = lb_time
        self._left_bound["index"] = lb_idx


class DifferenceFactorOptimizer(CurveOptimizer):

    TAG = "[DifferenceFactorOptimizer]"

    def __init__(
        self,
        file_path: str,
        file_buffer,
        initial_diff_factor: float = Constants.default_diff_factor,
    ):
        super().__init__(
            file_path=file_path, file_buffer=file_buffer, initial_diff_factor=initial_diff_factor
        )

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
            region_mask = (curve[:, 0] >= left_bound) & (curve[:, 0] <= right_bound)
            values = curve[region_mask, 1]

            # Check that we have enough data.
            if len(values) < 2:
                Log.d(self.TAG, "ROI is too short; returning a high penalty.")
                return np.inf

            # Use only the first 20% of the ROI.
            # Ensure at least 2 points for diff
            subset_length = max(2, int(0.3 * len(values)))
            sub_values = values[:subset_length]

            # Compute first differences on the restricted subset.
            diffs = np.diff(sub_values)
            if len(diffs) == 0:
                Log.d(self.TAG, "Not enough differences in the subset; returning a high penalty.")
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
            penalty = np.sum(excess**2)
            Log.d(
                self.TAG, f"Penalty computed for difference factor {difference_factor}: {penalty}"
            )
            return penalty

        except Exception as e:
            Log.e(self.TAG, f"Error calculating penalty metric: {e}")
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
                args=(self._left_bound["time"], self._right_bound["time"]),
                method="L-BFGS-B",
                bounds=bounds,
            )

            optimal_difference_factor = result.x[0]

            if 0.5 <= optimal_difference_factor <= 3.0:
                self._optimal_difference_factor = optimal_difference_factor
                Log.d(
                    self.TAG, f"Optimal difference factor found: {self._optimal_difference_factor}"
                )
            else:
                self._optimal_difference_factor = max(0.5, min(optimal_difference_factor, 3.0))
                Log.d(
                    self.TAG,
                    f"Optimal difference factor {optimal_difference_factor} out of bounds [0.5, 3.0], "
                    f"using {self._optimal_difference_factor}.",
                )

            return self._optimal_difference_factor

        except Exception as e:
            Log.e(self.TAG, f"Error optimizing difference factor: {e}")
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
                    self.TAG,
                    f"Run completed successfully. Results: {result}, left-bound: {self._left_bound['time']}, right-bound: {self._right_bound['time']}.",
                )
                return result, self._left_bound["time"], self._right_bound["time"]
            else:
                Log.d(
                    self.TAG,
                    f"Run completed unsucessful. Results: {result}, left-bound: {self._left_bound['time']}, right-bound: {self._right_bound['time']}.",
                )
                return (
                    Constants.default_diff_factor,
                    self._left_bound["time"],
                    self._right_bound["time"],
                )
        except Exception as e:
            Log.e(self.TAG, f"Run failed: {e}")
            raise


##########################################
# Additive-step drop-effect removal (replaces the earlier bridge-style inserts
# used in DropEffectCorrection.correct_drop_effects: cosine blend, then
# slope-matched PCHIP bridge).
#
# A drop effect is an ADDITIVE STEP ARTIFACT: the droplet permanently shifts
# the level of both channels, and everything recorded after the drop rides on
# the shifted level. Bridging over the region (however smoothly) preserves
# that shift as a bump in the series instead of removing it. The correct
# repair is:
#
#   1. Locate the drop instant (steepest sample-to-sample change in the core).
#   2. Estimate the step height `delta` as the gap between quadratic trend
#      extrapolations fitted to guard-banded flanks on either side of the
#      core, both evaluated at the drop instant. Evaluating both fits at the
#      same time point cancels the genuine fill trend, isolating the
#      artifact.
#   3. Subtract `delta` from EVERY sample at/after the end of the transient
#      core (this is the step removal -- the permanent shift is intended).
#   4. Patch the short transient core with a PCHIP interpolant between the
#      pre-flank trend and the now-shifted post-flank trend, add
#      flank-resampled noise texture, and feather the seams.
##########################################

# Samples excluded on each side of the core before fitting flank trends, so
# transient leakage (and savgol window bleed) doesn't contaminate the fits.
GUARD_SAMPLES = 3

# Flank window length (samples) for the quadratic trend fits.
FLANK_SAMPLES = 30

# Knots per flank for the core patch interpolant.
PATCH_KNOTS = 4

# Patch-knot flank reach (samples).
PATCH_FLANK = 12

# Seam crossfade length (samples).
FEATHER_SAMPLES = 3

# Scale on resampled flank residuals used as noise texture in the patch.
RESIDUAL_SCALE = 0.7


def _flank_fits(times, ysm, anchor_idx, post_idx):
    """Quadratic trend fits on guard-banded flanks either side of the core."""
    n = len(ysm)
    pre_sl = slice(
        max(0, anchor_idx - GUARD_SAMPLES - FLANK_SAMPLES), max(1, anchor_idx - GUARD_SAMPLES + 1)
    )
    post_sl = slice(
        min(n - 2, post_idx + GUARD_SAMPLES), min(n, post_idx + GUARD_SAMPLES + FLANK_SAMPLES + 1)
    )
    p_pre = np.polyfit(times[pre_sl], ysm[pre_sl], 2)
    p_post = np.polyfit(times[post_sl], ysm[post_sl], 2)
    return p_pre, p_post


def _drop_instant(raw, core_start, core_end):
    """Index of the steepest single-sample change inside the core."""
    d = np.abs(np.diff(raw[core_start : core_end + 1]))
    return core_start + int(np.argmax(d)) if d.size else core_start


def estimate_step_delta(times, raw, ysm, core_start, core_end):
    """Trend-compensated step height across the transient core.

    Both flank fits are evaluated at the drop instant; their gap is the step.
    Because both fits extrapolate the *genuine* trend to the same time point,
    the fill dynamics cancel and only the artifact height remains.
    """
    anchor_idx = max(0, core_start - 1)
    post_idx = min(core_end + 1, len(raw) - 1)
    p_pre, p_post = _flank_fits(times, ysm, anchor_idx, post_idx)
    td = times[_drop_instant(raw, core_start, core_end)]
    return float(np.polyval(p_post, td) - np.polyval(p_pre, td))


def remove_drop_step(
    times, raw, corrected, ysm, core_start, core_end, rng=None, min_significance=3.0
):
    """Remove one drop-effect step from `corrected` in place; returns delta.

    Args:
        times: full Relative_time array.
        raw: original (uncorrected) signal -- used only to find the drop instant.
        corrected: working corrected array; modified in place. The step
            estimate is computed on THIS array's smooth trend so multi-region
            (right-to-left) processing chains correctly.
        ysm: savgol-smoothed version of `corrected` (window ~11, poly 3).
            Recompute between regions when processing multiple drops.
        core_start, core_end: tight detected core of the transient (use the
            detection streak, NOT a padded region -- padding was a bridge
            concept; step removal wants the shortest interval containing the
            discontinuity so trend extrapolation distances stay small).
        rng: numpy Generator (seed for reproducible reanalysis).
        min_significance: skip correction if |delta| is below this multiple of
            the pre-flank residual noise sigma (guards against "correcting"
            regions that are noise-level, which would inject a spurious step).

    Returns:
        float: applied delta (0.0 if skipped as insignificant).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(corrected)
    anchor_idx = max(0, core_start - 1)
    post_idx = min(core_end + 1, n - 1)

    delta = estimate_step_delta(times, raw, ysm, core_start, core_end)

    # Significance gate: don't inject a step where none exists.
    resid_lo = max(0, anchor_idx - 4 * FLANK_SAMPLES)
    flank_resid = (corrected - ysm)[resid_lo:anchor_idx]
    noise_sigma = np.std(flank_resid) if flank_resid.size > 5 else 0.0
    if noise_sigma > 0 and abs(delta) < min_significance * noise_sigma:
        return 0.0

    # 1) Step removal: shift everything at/after the transient's end.
    corrected[post_idx:] -= delta
    sm_shift = ysm.copy()
    sm_shift[post_idx:] -= delta

    # 2) Patch the core: PCHIP between pre-flank trend and shifted post-flank.
    li = np.unique(
        np.linspace(max(0, anchor_idx - PATCH_FLANK), anchor_idx, PATCH_KNOTS).astype(int)
    )
    ri = np.unique(
        np.linspace(post_idx, min(n - 1, post_idx + PATCH_FLANK), PATCH_KNOTS).astype(int)
    )
    knot_t = np.concatenate([times[li], times[ri]])
    knot_v = np.concatenate([ysm[li], sm_shift[ri]])
    keep = np.concatenate([[True], np.diff(knot_t) > 0])
    patch = PchipInterpolator(knot_t[keep], knot_v[keep])(times[core_start : core_end + 1])

    # Noise texture from real pre-flank residuals.
    if flank_resid.size > 5:
        patch = patch + RESIDUAL_SCALE * rng.choice(flank_resid, size=patch.size, replace=True)

    corrected[core_start : core_end + 1] = patch

    # 3) Feather both seams (left into raw-level data, right into shifted data).
    f = min(FEATHER_SAMPLES, patch.size // 4)
    if f > 0:
        w = np.linspace(0.0, 1.0, f + 2)[1:-1]
        raw_left = raw[core_start : core_start + f]
        corrected[core_start : core_start + f] = (1 - w) * raw_left + w * corrected[
            core_start : core_start + f
        ]
        shifted_right = raw[core_end - f + 1 : core_end + 1] - delta
        corrected[core_end - f + 1 : core_end + 1] = (
            w[::-1] * corrected[core_end - f + 1 : core_end + 1] + (1 - w[::-1]) * shifted_right
        )

    return delta


class DropEffectCorrection(CurveOptimizer):
    """
    Class to mitigate drop effects in dissipation and resonance frequency curves
    by detecting and correcting anomalies independently.

    Attributes:
        file_buffer: Input buffer containing the dataset.
        initial_diff_factor (float): Initial difference factor for curve computation.

    Methods:
        correct_drop_effects(): Detects and corrects drop effects in both curves independently.
    """

    TAG = "[DropEffectCorrection]"

    def __init__(
        self,
        file_path: str,
        file_buffer,
        initial_diff_factor: float = Constants.default_diff_factor,
        bounds: list = [],
    ):
        """
        Initializes the DropEffectCorrection with the provided file buffer and difference factor.

        Args:
            file_path (str): The path to the loaded data file. Same folder used for saving figures.
            file_buffer: The data buffer containing dissipation data.
            initial_diff_factor (float): The initial factor for calculating the difference curve.
            bounds (list, optional): POI values from QModel or user-input.

        Raises:
            ValueError: If required columns are missing or bounds are not properly defined.
        """
        super().__init__(
            file_path=file_path,
            file_buffer=file_buffer,
            initial_diff_factor=initial_diff_factor,
            bounds=bounds,
        )

        self.loaded_datapath = file_path
        self.bounds = bounds

        if "Dissipation" not in self._dataframe.columns:
            Log.e(self.TAG, "The dataframe does not contain a 'Dissipation' column.")
            raise ValueError("The dataframe does not contain a 'Dissipation' column.")

        if "Resonance_Frequency" not in self._dataframe.columns:
            Log.e(self.TAG, "The dataframe does not contain a 'Resonance_Frequency' column.")
            raise ValueError("The dataframe does not contain a 'Resonance_Frequency' column.")

        if not ("index" in self._left_bound and "index" in self._right_bound):
            Log.e(self.TAG, "Bounds must be properly defined with 'index' keys.")
            raise ValueError("Bounds must be properly defined with 'index' keys.")

        if self._left_bound["index"] < 0 or self._right_bound["index"] >= len(self._dataframe):
            Log.e(
                self.TAG,
                f"Bounds are out of range of the dataframe or not initialized: {self._left_bound['index']} to {self._right_bound['index']}.",
            )
            raise ValueError(
                f"Bounds are out of range of the dataframe or not initialized: {self._left_bound['index']} to {self._right_bound['index']}."
            )

        if "Difference" not in self._dataframe.columns:
            Log.d(
                self.TAG,
                f"'Difference' column not found. Computing difference curve with factor {initial_diff_factor}.",
            )
            self._generate_curve(initial_diff_factor)

    def _detect_drop_effects_for_column(
        self, col_name: str, diff_offset: int = 2, starting_threshold_factor: float = 2
    ) -> tuple[list, list]:
        """
        Detects drop effects for a given column within the defined bounds by computing
        differences (with an offset) and flagging those points with a difference that
        deviates strongly from a robust baseline.

        Args:
            col_name (str): The column name to process (e.g., "Dissipation" or "Resonance_Frequency").
            diff_offset (int): The number of points to offset when computing differences.
            starting_threshold_factor (float): Sensitivity factor to begin outlier detection at.

        Returns:
            list: list of indices where drop effects need to be corrected.
        """
        left_idx = self._left_bound["index"]
        right_idx = self._right_bound["index"]

        if right_idx - left_idx < diff_offset + 1:
            Log.e(
                self.TAG,
                f"Not enough points in the specified region to compute differences with offset {diff_offset}.",
            )
            raise ValueError("Not enough points in the specified region to compute differences.")

        region_slice = slice(left_idx, right_idx + 1)
        values = self._dataframe[col_name].values[region_slice]
        values = savgol_filter(
            values, window_length=self._safe_savgol_win(len(values), 11, 3), polyorder=3
        )

        # Compute differences with the specified offset to smooth out noise.
        local_indices = np.arange(diff_offset, len(values))
        diffs = values[local_indices] - values[local_indices - diff_offset]

        first_region_found = []
        contiguous_region_found = False
        last_region_size = 0

        while True:
            # Compute robust statistics: median and MAD.
            median_diff = np.median(diffs)
            mad_diff = np.median(np.abs(diffs - median_diff))
            # Use the MAD if positive; otherwise fall back to standard deviation.
            threshold = starting_threshold_factor * (mad_diff if mad_diff > 0 else np.std(diffs))

            drop_effects = []
            for local_idx in local_indices:
                current_diff = values[local_idx] - values[local_idx - diff_offset]

                # Flag points with a difference that deviates too much from the median.
                if np.abs(current_diff - median_diff) > threshold:
                    drop_effects.append(local_idx)

            current_streak = []
            for idx in drop_effects:
                if current_streak and idx == current_streak[-1] + 1:
                    current_streak.append(idx)
                else:
                    current_streak = [idx]

                # Check if we have reached at least 3 contiguous points.
                if len(current_streak) > max(3, last_region_size):
                    if not first_region_found:
                        first_region_found = current_streak
                    contiguous_region_found = True
                    break

            if contiguous_region_found:
                if last_region_size == len(current_streak) or col_name == "Resonance_Frequency":
                    break  # wait for stable result; but skip this delay for rf data
            last_region_size = len(current_streak)
            starting_threshold_factor -= 1
            if starting_threshold_factor <= 0:
                Log.d(self.TAG, "Reverting to initial region. Reached min threshold factor.")
                current_streak = first_region_found
                break
        if diff_offset in current_streak:
            Log.d(self.TAG, "Reverting to initial region. Returned min diff offset.")
            current_streak = first_region_found

        # Work left from minimum time index, looking for an opposite direction shift prior to the big jump.
        if not current_streak:
            Log.d(self.TAG, f"No drop effects detected in {col_name}.")
            return ([], values, 0)
        min_count = len(current_streak)
        min_drop = min(current_streak)
        sign = 1 if col_name == "Dissipation" else -1
        if sign == 1:
            argidx = np.argmax(values[:min_drop])
        else:
            argidx = np.argmin(values[:min_drop])
        base_slope = (values[argidx] - values[0]) / min_drop
        window_size = 3
        while True:
            min_drop = min_drop - 1
            current_streak.append(min_drop)
            min_drop = min_drop - 1
            # Pretend this step never happened if it fails to find an acceptable edge before the left bound.
            if min_drop <= 0:
                Log.d(
                    self.TAG,
                    "Drop effect using original bounds, scanning left never reached an acceptable edge.",
                )
                current_streak = current_streak[:min_count]
                break
            # Check if this index's value contains an acceptable edge.
            if sign == 1:
                if values[min_drop] - values[min_drop - window_size] > base_slope:
                    break
            else:
                if values[min_drop] - values[min_drop - window_size] < base_slope:
                    break
            # If unacceptable, add it to the list for correction.
            current_streak.append(min_drop)

        # Extend the drop region rightward until the raw signal reconverges with the
        # smooth signal.  Applies to both Dissipation and Resonance_Frequency.
        #
        # Recovery criterion: |raw - smooth| < tolerance for NUM_PTS consecutive steps,
        # where tolerance is 2x the pre-drop standard deviation.  This is direction-
        # agnostic (works for upward spikes in Dissipation and downward drops in RF)
        # and scales automatically with signal variability.
        #
        # Cap: max(75, 3 x initial-streak-length) -- wide enough for complex multi-phase
        # events while still terminating quickly for narrow spikes.
        NUM_PTS = diff_offset
        max_drop = max(current_streak)
        if max_drop + 1 < len(values) - 1:
            # Estimate pre-drop variability from the smooth signal before the anomaly.
            pre_end = min(current_streak)
            pre_start = max(0, diff_offset)
            pre_slice = (
                values[pre_start:pre_end] if pre_end > pre_start else values[: max(1, pre_end)]
            )
            pre_std = np.std(pre_slice) if len(pre_slice) > 1 else np.std(values) * 0.05
            tolerance = max(2.0 * pre_std, 1e-12)  # guard against near-zero std

            raw_full = self._dataframe[col_name].values[region_slice]
            recovery_count = 0
            total_dist = 0
            max_total_dist = max(35, 2 * min_count)
            while True:
                total_dist += 1
                if max_drop + 1 >= len(values) - 1:
                    break
                max_drop += 1
                if abs(raw_full[max_drop] - values[max_drop]) < tolerance:
                    recovery_count += 1
                else:
                    recovery_count = 0
                if recovery_count > NUM_PTS or total_dist > max_total_dist:
                    break
                current_streak.append(max_drop)

        Log.d(
            self.TAG,
            f"Detected drop effects in {col_name} at indices {[int(str(de)) for de in current_streak]}",
        )

        # Map back to the full dataframe index.
        global_idx = [local_idx + left_idx for local_idx in current_streak]
        # min_count is the initial streak size BEFORE any left/right expansion.
        # Callers use it so the skip guard threshold is independent of how far
        # the region was subsequently expanded.
        return (global_idx, values, min_count)

    def _middle_slice_list(self, whole: list, fraction: float = 0.5) -> list:
        n = len(whole)
        count = int(n * fraction)
        start = (n - count) // 2
        end = start + count
        middle = whole[start:end]
        return middle

    def _safe_savgol_win(self, n: int, preferred: int, poly: int) -> int:
        """
        Choose robust window lengths (odd, <= series length, > polyorder)
        """
        # best odd <= n
        best = n if n % 2 == 1 else n - 1
        w = min(preferred, max(3, best))
        min_odd = (poly + 2) if ((poly + 2) % 2 == 1) else (poly + 3)
        if w < min_odd:
            w = min(min_odd, best)
        return min(n, max(w, poly + 1))  # ensure > poly and < n

    def correct_drop_effects(
        self,
        baseline_diss: list = None,
        baseline_rf: list = None,
        show_corrections: bool = False,  # Default to True for debug ONLY
        save_corrections: bool = False,
        export_corrections_csv: bool = False,
    ) -> tuple:
        """
        Corrects drop effects in dissipation and resonance frequency data.

        This method detects sudden drop effects in the 'Dissipation' column of the internal
        dataframe and applies corresponding offset corrections to both 'Dissipation' and
        'Resonance_Frequency' columns. A drop effect is defined as a sudden decrease in the value
        compared to the previous (good) data point. The correction adjusts the subsequent segment
        of data so that the drop point matches the preceding value. Optionally, the method can
        plot the original and corrected data for visualization.

        Args:
            baseline_diss (float, optional): The baseline dissipation region to use for correction.
                If not provided, the baseline is determined by taking the standard deviation of the
                dissipation values prior to index `self._left_bound['index']` from the original data.
                Defaults to None.
            baseline_rf (float, optional): The baseline resonance region to use for correction.
                If not provided, the baseline is determined by taking the standard deviation of the
                resonance values prior to index `self._left_bound['index']` from the original data.
                Defaults to None.
            show_corrections (bool, optional): If True, the method will generate a plot and show it
                 to visualize the corrections applied. Defaults to False.
            save_corrections (bool, optional): If True, the method will generate a plot and save it
                 to visualize the corrections applied. Defaults to False. NOTE: If this argument and
                 `show_corrections` are True, the same plot generation will be used for both actions.
            export_corrections_csv (bool, optional): If True, the method will export the corrected
                 dataframes for dissipation and rf to a CSV file with a file name of the format:
                 `{RUNNAME}_corrected.csv` to the working directory. Defaults to False.

        Returns:
            tuple: A tuple containing two numpy arrays:
                - The first array contains the corrected dissipation values.
                - The second array contains the corrected resonance frequency values.

        Example:
            >>> corrected_diss, corrected_rf = instance.correct_drop_effects(show_corrections=True)
        """
        # Save original data for plotting.
        relative_time = self._dataframe["Relative_time"].values.copy()
        original_diss = self._dataframe["Dissipation"].values.copy()
        original_rf = self._dataframe["Resonance_Frequency"].values.copy()

        # Determine baselines if not provided.
        if baseline_diss is None:
            baseline_diss = original_diss[: self._left_bound["index"]]
        base_diss_middle = self._middle_slice_list(baseline_diss)
        # base_diss_avg = np.average(base_diss_middle)
        base_diss_std = np.std(base_diss_middle)
        if baseline_rf is None:
            baseline_rf = original_rf[: self._left_bound["index"]]
        base_rf_middle = self._middle_slice_list(baseline_rf)
        # base_rf_avg = np.average(base_rf_middle)
        base_rf_std = np.std(base_rf_middle)

        # Make working copies.
        corrected_diss = original_diss.copy()
        corrected_rf = original_rf.copy()

        # Detect drop effects independently for each curve.
        drop_effects_diss, smooth_diss, init_count_diss = self._detect_drop_effects_for_column(
            "Dissipation", starting_threshold_factor=STARTING_THRESHOLD_FACTOR
        )
        drop_effects_rf, smooth_rf, init_count_rf = self._detect_drop_effects_for_column(
            "Resonance_Frequency", starting_threshold_factor=STARTING_THRESHOLD_FACTOR
        )
        drop_effects = list(set(drop_effects_diss + drop_effects_rf))

        # Sort detected regions by ascending time index.
        drop_effects_diss.sort()
        drop_effects_rf.sort()
        drop_effects.sort()

        # List to store the continuous regions of correction.
        # May be two discreet regions if diss and rf indicate differently.
        contiguous_regions = []

        # List to store the starting indices of corrections.
        correction_indices = []

        # If multiple regions found (separate for diss and rf) correct both.
        for idx in drop_effects:
            if contiguous_regions and idx - 1 in contiguous_regions[-1]:
                contiguous_regions[-1].append(idx)
            else:
                contiguous_regions.append([idx])

        # Check if two zones got merged into one; where one would get skipped but not the other
        if len(contiguous_regions) == 1:
            region = contiguous_regions[0]
            idx = region[0]
            if idx - self._left_bound["index"] < len(region) // 2:
                Log.d(self.TAG, "Trimming the left region to prevent it from being skipped.")
                start_at = max(drop_effects_diss[0], drop_effects_rf[0])
                end_at = max(drop_effects_diss[-1], drop_effects_rf[-1])
                contiguous_regions = [np.arange(start_at, end_at + 1, 1, dtype=int).tolist()]
                # if len(drop_effects_diss) < len(drop_effects_rf):
                #     contiguous_regions = [drop_effects_diss]
                # else:
                #     contiguous_regions = [drop_effects_rf]

        Log.d(
            self.TAG,
            f"Found {len(contiguous_regions)} contiguous drop effect region(s) to correct.",
        )
        Log.d(
            self.TAG,
            f"Raw drop effect regions: {[[int(i) for i in np.array(idx, dtype=int)] for idx in contiguous_regions]}",
        )

        if len(contiguous_regions) > 1:
            Log.d(
                self.TAG,
                "Processing regions in reverse order. Starting with the last region, working left.",
            )

            # Start with right-most, working left, to avoid baseline shift offsets.
            contiguous_regions.reverse()

        smooth_diss_full = savgol_filter(
            original_diss,
            window_length=self._safe_savgol_win(len(original_diss), 11, 3),
            polyorder=3,
        )
        super_smooth_diss_full = savgol_filter(
            original_diss,
            window_length=self._safe_savgol_win(len(original_diss), 69, 3),
            polyorder=3,
        )
        super_smooth_rf_full = savgol_filter(
            original_rf, window_length=self._safe_savgol_win(len(original_rf), 69, 3), polyorder=3
        )

        # Process each detected drop effect region for Dissipation and Resonance Frequency.
        # Skip guard threshold: use the pre-expansion initial streak size so that the
        # guard does not grow with the region and incorrectly reject legitimate mid-run drops.
        init_count = max(
            1,
            min(
                init_count_diss if init_count_diss else len(drop_effects_diss),
                init_count_rf if init_count_rf else len(drop_effects_rf),
            ),
        )
        for region in contiguous_regions:

            # Directionality check: skip regions with no meaningful upward excursion in
            # dissipation.  Compare the peak within the region against the lower of the two
            # endpoints rather than comparing endpoints directly.  This is robust to the
            # right expansion having extended the region past the spike peak into recovery
            # (where the endpoint value can drop below the start value even for a genuine
            # upward spike, causing the old endpoint comparison to wrongly skip the region).
            region_diss_smooth = smooth_diss_full[region[0] : region[-1] + 1]
            peak_diss = float(np.max(region_diss_smooth))
            baseline_diss = min(float(region_diss_smooth[0]), float(region_diss_smooth[-1]))
            full_range = float(np.max(smooth_diss_full) - np.min(smooth_diss_full))
            if full_range > 0 and (peak_diss - baseline_diss) / full_range < 0.05:
                Log.d(
                    self.TAG,
                    f"Skipping region {relative_time[region[0]]:.3f}–{relative_time[region[-1]]:.3f}: "
                    f"no significant dissipation excursion (peak-baseline = {peak_diss - baseline_diss:.3e}, "
                    f"full range = {full_range:.3e})",
                )
                continue

            # Skip if the drop starts too close to the start-of-fill boundary.
            # Uses the pre-expansion initial streak size so the threshold stays small
            # even when the right expansion has made the total region very wide.
            idx = region[0]
            if idx - self._left_bound["index"] < init_count // 2:
                Log.d(
                    self.TAG,
                    "Skipped correcting an early region that was too close to start-of-fill.",
                )
                continue

            # Record the indices where the correction is applied.
            correction_indices.extend(region)

            # Tight detected core of the transient -- NOT padded. The old bridge
            # approaches widened the region to give a smoothing kernel more room to
            # work; step removal instead wants the shortest interval containing the
            # discontinuity, so the flank trend fits used to estimate the step
            # height stay close to (and representative of) the drop itself.
            core_start, core_end = region[0], region[-1]
            anchor_idx = max(0, core_start - 1)

            # Only if end-of-fill is contained in the core: the drop and the
            # genuine fill-completion event overlap, so treating this as a
            # permanent additive step would blur over real fill dynamics. Fall
            # back to an anchor-aligned super-smooth patch with no trailing
            # shift, as before.
            used_smooth_override = (
                len(self.bounds) > 1
                and isinstance(self.bounds[1], int)
                and core_start < self.bounds[1] < core_end
            )  # indices, not timestamps
            if used_smooth_override:
                Log.d(
                    self.TAG,
                    "Correcting drop with smoothed data due to region conflicts with end-of-fill",
                )
                insert_diss = super_smooth_diss_full[core_start : core_end + 1].copy()
                insert_rf = super_smooth_rf_full[core_start : core_end + 1].copy()

                # Align the smoothed fallback data to our anchor point to prevent disjoint jumps
                insert_diss += corrected_diss[anchor_idx] - insert_diss[0]
                insert_rf += corrected_rf[anchor_idx] - insert_rf[0]

                # Add jitter based on the baseline stdev since the super-smooth
                # fallback carries no noise texture of its own. Reduced multiplier
                # from 2.0 to 0.5 to prevent excessive jitter.
                rng = np.random.default_rng()
                insert_diss = insert_diss + rng.normal(
                    0.0, 0.5 * base_diss_std, size=len(insert_diss)
                )
                insert_rf = insert_rf + rng.normal(0.0, 0.5 * base_rf_std, size=len(insert_rf))

                corrected_diss[core_start : core_end + 1] = insert_diss
                corrected_rf[core_start : core_end + 1] = insert_rf

                Log.d(
                    self.TAG,
                    f"Smoothed drop region [{core_start}, {core_end}] "
                    "(no permanent step shift; overlaps end-of-fill).",
                )
                continue

            # Additive-step removal: estimate the permanent level shift caused
            # by the drop and subtract it from every sample at/after the
            # transient, then patch the short transient core with a
            # trend-matched, noise-textured interpolant. The smoothed trend is
            # recomputed from the CORRECTED arrays each iteration (rather than
            # reused from smooth_diss_full/smooth_rf_full, which reflect only
            # the original data) so a region's step estimate reflects any
            # shifts already applied by later (already-processed) regions in
            # this right-to-left pass.
            win = self._safe_savgol_win(len(corrected_diss), 11, 3)
            ysm_diss = savgol_filter(corrected_diss, win, 3)
            ysm_rf = savgol_filter(corrected_rf, win, 3)

            rng = np.random.default_rng()
            delta_diss = remove_drop_step(
                relative_time,
                original_diss,
                corrected_diss,
                ysm_diss,
                core_start,
                core_end,
                rng=rng,
            )
            delta_rf = remove_drop_step(
                relative_time,
                original_rf,
                corrected_rf,
                ysm_rf,
                core_start,
                core_end,
                rng=rng,
            )

            Log.d(
                self.TAG,
                f"Step-removed drop region [{core_start}, {core_end}]: "
                f"delta_diss={delta_diss:.4e}, delta_rf={delta_rf:.2f} Hz",
            )

        if show_corrections or save_corrections:
            self._plot_corrections(
                show_corrections,
                save_corrections,
                correction_indices,
                relative_time,
                original_diss,
                original_rf,
                corrected_diss,
                corrected_rf,
                smooth_diss,
                smooth_rf,
            )

        # DEBUG: Write out corrected datas to file for external review.
        if export_corrections_csv:
            with open(f"{self._strip_filename(self._file_path)}_corrected.csv", "w") as f:
                f.write("corrected_diss,corrected_rf\n")
                for i in range(corrected_diss.size):
                    f.write(f"{corrected_diss[i]},{corrected_rf[i]}\n")

        return (corrected_diss, corrected_rf)

    def _plot_corrections(
        self,
        show,
        save,
        correction_indices,
        relative_time,
        original_diss,
        original_rf,
        corrected_diss,
        corrected_rf,
        smooth_diss,
        smooth_rf,
    ):
        """
        Plots the original and corrected data for Dissipation and Resonance Frequency,
        and marks the indices where corrections occurred.
        """

        indices = np.arange(len(original_diss))
        subplots: tuple[plt.Figure, tuple[plt.Axes, plt.Axes]] = plt.subplots(2, 1, figsize=(10, 8))
        fig, axs = subplots

        # zoom_xid = min(correction_indices) - 2*len(correction_indices)
        # zoom_yid = max(correction_indices) + 2*len(correction_indices)
        zoom_xid = self._left_bound["index"]
        zoom_yid = self._right_bound["index"]

        # Plot for Dissipation.
        axs[0].plot(
            relative_time[indices], original_diss, label="Original Dissipation", color="blue"
        )
        axs[0].plot(
            relative_time[indices],
            corrected_diss,
            label="Corrected Dissipation",
            color="red",
            linestyle="--",
        )
        axs[0].plot(
            relative_time[indices][zoom_xid : zoom_yid + 1],
            smooth_diss,
            label="Smoothed Dissipation",
            color="yellow",
            linestyle="--",
        )
        axs[0].axvline(relative_time[self._left_bound["index"]], color="gray", linestyle=":")
        axs[0].axvline(relative_time[self._right_bound["index"]], color="gray", linestyle=":")

        # Mark the indices where corrections were applied.
        for idx in correction_indices:
            axs[0].axvline(relative_time[idx], color="green", linestyle=":", alpha=0.7)
        axs[0].set_title("Dissipation Correction")
        axs[0].set_xlabel("Relative Time (sec)")
        axs[0].set_ylabel("Dissipation")
        axs[0].legend()

        # # Zoom to the region of interest around the correction.
        axs[0].set_xlim(relative_time[zoom_xid], relative_time[zoom_yid])
        axs[0].set_ylim(
            min(original_diss[zoom_xid:zoom_yid]), max(original_diss[zoom_xid:zoom_yid])
        )

        # Plot for Resonance Frequency.
        axs[1].plot(
            relative_time[indices], original_rf, label="Original Resonance Frequency", color="blue"
        )
        axs[1].plot(
            relative_time[indices],
            corrected_rf,
            label="Corrected Resonance Frequency",
            color="red",
            linestyle="--",
        )
        axs[1].plot(
            relative_time[indices][zoom_xid : zoom_yid + 1],
            smooth_rf,
            label="Smoothed Resonance Frequency",
            color="yellow",
            linestyle="--",
        )
        axs[1].axvline(relative_time[self._left_bound["index"]], color="gray", linestyle=":")
        axs[1].axvline(relative_time[self._right_bound["index"]], color="gray", linestyle=":")

        # Mark the same correction indices on the RF plot.
        for idx in correction_indices:
            axs[1].axvline(relative_time[idx], color="green", linestyle=":", alpha=0.7)
        axs[1].set_title("Resonance Frequency Correction")
        axs[1].set_xlabel("Relative Time (sec)")
        axs[1].set_ylabel("Resonance Frequency (Hz)")
        axs[1].legend()

        # # Zoom to the region of interest around the correction.
        axs[1].set_xlim(relative_time[zoom_xid], relative_time[zoom_yid])
        axs[1].set_ylim(min(original_rf[zoom_xid:zoom_yid]), max(original_rf[zoom_xid:zoom_yid]))

        fig.tight_layout()

        if save:
            # export figure to pdf
            export_path = self.loaded_datapath
            export_path = export_path.replace(".csv", Constants.export_file_format)
            export_path = export_path.replace("_fundamental", "")
            export_path = export_path.replace("_3rd", "")
            export_path = export_path.replace(".csv", "_0.pdf")
            Log.i(self.TAG, f"Exporting Figure to:\n\t{export_path}")
            fig.savefig(export_path)

        if show:
            fig.show()

        # Zoom out to the entire view for shown plot
        # axs[0].set_xlim(0, relative_time[-1])
        # axs[0].set_ylim(min(original_diss), max(original_diss))
        # axs[1].set_xlim(0, relative_time[-1])
        # axs[1].set_ylim(min(original_rf), max(original_rf))
