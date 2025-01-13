"""
This module contains the CurveOptimizer class for optimizing the difference factor for
the difference curve provided from the datastream.

Classes:
    CurveOptimizer: Provides static methods for difference factor optimization.

Constants:
    DEFAULT_FACTOR: Default difference factor for difference calculation.
    Z_FACTOR: Constant for modified Z-score calculation.
    BOUNDS: Bounds for optimization of the difference factor.
    MIN_NUM_SAMPLES: Minimum number of samples required for optimization.
    SIGMA_SMOOTHING: Smoothing factor for Gaussian filtering.
"""
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from QATCH.common.logger import Logger as Log

TAG = "[CurveOptimizer]"


class CurveOptimizer:
    """
    Provides static methods for optimizing difference factor for a given data stream.

    Methods:
        run(file_buffer, num_samples=10): Main entry point for optimizing the difference curve.
        _get_sample_points(difference_curve, num_sample_points): Samples points from the difference curve.
        _remove_outliers(sample_data, threshold): Removes outliers from sample data using a modified Z-score.
        _generate_difference_curve(data_frame, difference_factor): Generates the difference curve from a DataFrame.
        _find_run_bounds(data, smoothing_sigma): Identifies run bounds in the data.
        _slope_metric(difference_factor, data_frame, sampled_indices): Evaluates a slope metric for optimization.
        _find_optimal_difference_factor(data_frame, initial_guess, sampled_indices, bounds): Finds the optimal factor.
        _initialize_file_buffer(file_buffer): Prepares the file buffer for reading.
    """
    DEFAULT_FACTOR = 2.0
    Z_FACTOR = 0.6745
    BOUNDS = (1.0, 3.0)
    MIN_NUM_SAMPLES = 3
    SIGMA_SMOOTHING = 10

    @staticmethod
    def run(file_buffer, num_samples: int = 10, bounds: tuple = BOUNDS) -> tuple:
        """
        Optimizes the difference factor for a given file buffer.

        Args:
            file_buffer: A seekable file-like object or file path containing the CSV data.
            num_samples (int): Number of sample points for optimization.

        Returns:
            tuple: Optimal difference factor and the associated metric.

        Raises:
            ValueError: If the file cannot be read, or if num_samples is invalid.
        """
        # Error checking for number of points on difference curve to sample.
        if num_samples <= CurveOptimizer.MIN_NUM_SAMPLES:
            Log.e(
                TAG, f"num_samples must be a positive integer larger than {CurveOptimizer.MIN_NUM_SAMPLES}.")
            raise ValueError(
                f"num_samples must be a positive integer larger than {CurveOptimizer.MIN_NUM_SAMPLES}.")

        file_buffer = CurveOptimizer._initialize_file_buffer(file_buffer)

        # Error checking for parsibility of file_buffer.
        try:
            data_frame = pd.read_csv(file_buffer)
        except Exception as e:
            Log.e(TAG, f"Error reading CSV file: {e}")
            raise ValueError(f"Error reading CSV file: {e}")
        Log.i(
            TAG, f"Beginning difference curve optimization using {num_samples} sample points on file_buffer {file_buffer}")

        # Create the initial difference curve with the default difference factor.
        difference_curve = CurveOptimizer._generate_difference_curve(
            data_frame)

        # Select sample points from the difference curve to interpolate between.
        sampled_data = CurveOptimizer._get_sample_points(
            difference_curve, num_sample_points=num_samples
        )

        # Remove outliers from head and tail of sample set.
        filtered_data = CurveOptimizer._remove_outliers(sampled_data)

        if len(filtered_data) == 0:
            Log.w(
                TAG, f"No data points left after outlier removal. Using imperfect samples.")
            filtered_indices = sampled_data
        else:
            filtered_indices = [int(item) for item in filtered_data[:, 0]]

        # Perform optimization of difference factor.
        optimal_factor, optimal_metric = CurveOptimizer._find_optimal_difference_factor(
            data_frame, CurveOptimizer.DEFAULT_FACTOR, filtered_indices, bounds=bounds
        )
        Log.i(
            TAG, f"Difference curve optimization complete, reporting optimal difference factor {optimal_factor}, with minimal objective score {optimal_metric}.")
        return (optimal_factor, optimal_metric)

    @staticmethod
    def _get_sample_points(difference_curve: np.ndarray, num_sample_points: int) -> np.ndarray:
        """
        Samples points from the difference curve.

        Args:
            difference_curve (np.ndarray): 2D array of the difference curve.
            num_sample_points (int): Number of sample points.

        Returns:
            np.ndarray: Array of sampled points.

        Raises:
            ValueError: If the input is invalid.
        """
        Log.d(TAG, f"Getting sample points.")
        if not isinstance(difference_curve, np.ndarray) or difference_curve.ndim != 2:
            Log.e(TAG, "difference_curve must be a 2D numpy array.")
            raise ValueError("difference_curve must be a 2D numpy array.")
        if num_sample_points <= 0:
            Log.e(TAG, "num_sample_points must be a positive integer.")
            raise ValueError("num_sample_points must be a positive integer.")

        start_idx, end_idx = CurveOptimizer._find_run_bounds(
            difference_curve, CurveOptimizer.SIGMA_SMOOTHING)
        sampled_indices = np.linspace(
            start_idx, end_idx, num=num_sample_points, dtype=int
        )
        sampled_y_values = difference_curve[sampled_indices]
        return sampled_y_values

    @staticmethod
    def _remove_outliers(sample_data: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        """
        Removes outliers from sample data using a modified Z-score.

        Args:
            sample_data (np.ndarray): 2D array of sample data.
            threshold (float): Z-score threshold for identifying outliers.

        Returns:
            np.ndarray: Filtered sample data.

        Raises:
            ValueError: If the input is invalid.
        """
        Log.d(TAG, f"Removing outliers")
        if not isinstance(sample_data, np.ndarray) or sample_data.ndim != 2:
            Log.e(TAG, "sample_data must be a 2D numpy array.")
            raise ValueError("sample_data must be a 2D numpy array.")
        if sample_data.shape[1] < 2:
            Log.e(TAG, "sample_data must have at least two columns.")
            raise ValueError("sample_data must have at least two columns.")
        if threshold <= 0:
            Log.e(TAG, "threshold must be a positive number.")
            raise ValueError("threshold must be a positive number.")

        y_values = sample_data[:, 1]
        median_y = np.median(y_values)
        mad = np.median(np.abs(y_values - median_y))

        if mad == 0:
            return sample_data, np.array([])

        modified_z_scores = CurveOptimizer.Z_FACTOR * \
            (y_values - median_y) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        filtered_data = sample_data[~outlier_mask]

        return filtered_data

    @staticmethod
    def _generate_difference_curve(data_frame: pd.DataFrame, difference_factor: float = DEFAULT_FACTOR) -> np.ndarray:
        """
        Generates the difference curve from the input DataFrame.

        Args:
            data_frame (pd.DataFrame): Input DataFrame containing required columns.
            difference_factor (float): Scaling factor for the difference calculation.

        Returns:
            np.ndarray: Generated difference curve.

        Raises:
            ValueError: If the input DataFrame is invalid.
        """
        Log.d(TAG, f"Generating difference curve.")
        if not isinstance(data_frame, pd.DataFrame):
            Log.e(TAG, "data_frame must be a pandas DataFrame.")
            raise ValueError("data_frame must be a pandas DataFrame.")
        if difference_factor <= 0:
            Log.e(TAG, "difference_factor must be a positive number.")
            raise ValueError("difference_factor must be a positive number.")

        required_columns = ["Dissipation", "Resonance_Frequency"]
        if not all(column in data_frame.columns for column in required_columns):
            Log.e(
                TAG, f"Input DataFrame must contain the following columns: {required_columns}")
            raise ValueError(
                f"Input DataFrame must contain the following columns: {required_columns}"
            )

        xs = data_frame["Relative_time"]
        if len(xs) < 2:
            Log.e(
                TAG,  f"Relative_time column must contain at least two values; found {len(xs)}")
            raise ValueError(
                f"Relative_time column must contain at least two values found {len(xs)}")

        i = next((x for x, t in enumerate(xs) if t > 0.5), None)
        j = next((x for x, t in enumerate(xs) if t > 2.5), None)

        if i is None or j is None or i >= j:
            Log.e(
                TAG,  f"Invalid time range in Relative_time column. Bounds were '({i, j})'")
            raise ValueError(
                f"Invalid time range in Relative_time column. Bounds were '({i, j})'")

        avg_resonance_frequency = data_frame["Resonance_Frequency"][i:j].mean()
        avg_dissipation = data_frame["Dissipation"][i:j].mean()

        data_frame["ys_diss"] = (
            (data_frame["Dissipation"] - avg_dissipation)
            * avg_resonance_frequency
            / 2
        )
        data_frame["ys_freq"] = avg_resonance_frequency - \
            data_frame["Resonance_Frequency"]
        data_frame["Difference"] = data_frame["ys_freq"] - \
            difference_factor * data_frame["ys_diss"]

        return np.stack(
            (np.arange(len(data_frame["Difference"])),
             data_frame["Difference"].values.flatten()),
            axis=1,
        )

    @staticmethod
    def _find_run_bounds(data: np.ndarray, smoothing_sigma: int) -> tuple:
        """
        Identifies run bounds in the data using Gaussian smoothing.

        Args:
            data (np.ndarray): 2D array of data.
            smoothing_sigma (int): Sigma value for Gaussian smoothing.

        Returns:
            tuple: Start and end indices of the run.

        Raises:
            ValueError: If the input is invalid.
        """
        Log.d(TAG, f"Computing left/right run bounds.")
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            Log.e(TAG, "data_frame must be a pandas DataFrame.")
            raise ValueError("data must be a 2D numpy array.")
        if smoothing_sigma <= 0:
            Log.e(TAG, "smoothing_sigma must be a positive integer.")
            raise ValueError("smoothing_sigma must be a positive integer.")

        y = data[:, 1]
        if len(y) < 2:
            Log.e(TAG, "Data must contain at least two points.")
            raise ValueError("Data must contain at least two points.")

        smoothed_y = gaussian_filter1d(y, sigma=smoothing_sigma)
        peaks, _ = find_peaks(smoothed_y)
        troughs, _ = find_peaks(-smoothed_y)

        if len(peaks) == 0 or len(troughs) == 0:
            Log.e(TAG, "Cannot find sufficient peaks or troughs in the data.")
            raise ValueError(
                "Cannot find sufficient peaks or troughs in the data.")

        start_peak_idx = peaks[np.argmax(smoothed_y[peaks])]
        steep_descent = np.argmax(
            np.gradient(smoothed_y) < -
            np.percentile(np.gradient(smoothed_y), 95)
        )
        end_peak_idx = (
            steep_descent
            if steep_descent > start_peak_idx
            else troughs[np.argmin(smoothed_y[troughs])]
        )

        return start_peak_idx, end_peak_idx

    @staticmethod
    def _slope_metric(difference_factor: float, data_frame: pd.DataFrame, sampled_indices) -> float:
        """
        Evaluates a slope between sampled points for the given difference factor.

        Args:
            difference_factor (float): Scaling factor for the difference calculation.
            data_frame (pd.DataFrame): DataFrame containing the input data.
            sampled_indices (list or np.ndarray): Indices of sampled points.

        Returns:
            float: Calculated slope metric.

        Raises:
            ValueError: If the input is invalid.
        """
        Log.d(
            TAG, f"Evaluating sample indices {sampled_indices} with difference factor '{difference_factor}'")
        if difference_factor <= 0:
            Log.e(TAG, "difference_factor must be a positive number.")
            raise ValueError("difference_factor must be a positive number.")
        if not isinstance(sampled_indices, (list, np.ndarray)):
            Log.e(TAG, "sampled_indices must be a list or numpy array.")
            raise ValueError("sampled_indices must be a list or numpy array.")

        curve = CurveOptimizer._generate_difference_curve(
            data_frame, difference_factor)
        sampled_points = curve[sampled_indices]
        slopes = np.diff(sampled_points[:, 1]) / np.diff(sampled_points[:, 0])
        return np.sum(slopes**2)

    @staticmethod
    def _find_optimal_difference_factor(data_frame: pd.DataFrame, initial_guess: float, sampled_indices: np.ndarray, bounds: tuple):
        """
        Finds the optimal difference factor using numerical optimization.

        Args:
            data_frame (pd.DataFrame): DataFrame containing the input data.
            initial_guess (float): Initial guess for the optimization.
            sampled_indices (np.ndarray): Indices of sampled points.
            bounds (tuple): Bounds for the optimization.

        Returns:
            tuple: Optimal factor and associated metric.

        Raises:
            ValueError: If the input is invalid.
            RuntimeError: If optimization fails.
        """
        Log.d(
            TAG, "Searching for optimal difference factor.")
        if initial_guess <= 0:
            Log.e(TAG, "initial_guess must be a positive number.")
            raise ValueError("initial_guess must be a positive number.")
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            Log.e(TAG, "bounds must be a tuple of two numbers.")
            raise ValueError("bounds must be a tuple of two numbers.")

        def objective(factor):
            return CurveOptimizer._slope_metric(factor, data_frame, sampled_indices)

        result = minimize(objective, initial_guess, bounds=[bounds])
        if not result.success:
            Log.e(TAG, "Optimization failed.")
            raise RuntimeError("Optimization failed.")

        return result.x[0], result.fun

    @staticmethod
    def _initialize_file_buffer(file_buffer):
        """
        Prepares the file buffer for reading.

        Args:
            file_buffer: File-like object or file path.

        Returns:
            File-like object: Prepared file buffer.

        Raises:
            ValueError: If the file buffer is invalid.
        """
        Log.d(TAG, 'Initializing file buffer.')
        if not isinstance(file_buffer, str):
            if hasattr(file_buffer, "seekable") and file_buffer.seekable():
                file_buffer.seek(0)
            else:
                Log.e(TAG, "Cannot 'seek' file buffer stream.")
                raise ValueError("Cannot 'seek' file buffer stream.")
        return file_buffer
