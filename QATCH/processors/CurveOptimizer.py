import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from QATCH.common.logger import Logger as Log

TAG = ["CurveOptimizer"]


class CurveOptimizer:
    def __init__(self, file_buffer, initial_diff_factor: float = 2.0):
        self.file_buffer = file_buffer
        self.initial_diff_factor = initial_diff_factor
        try:
            self.filepath = self._initialize_file_buffer(file_buffer)
            self.df = pd.read_csv(self.filepath)
            Log.d(TAG, f"CSV file loaded successfully from {self.filepath}.")
        except Exception as e:
            Log.e(TAG, f"Failed to load CSV file: {e}")
            raise

        self.curve = None
        self.lb = {"time": -1, "index": -1}
        self.rb = {"time": -1, "index": -1}
        self.optimal_difference_factor = None

    def _initialize_file_buffer(self, file_buffer):
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

    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        try:
            min_val = np.min(data)
            max_val = np.max(data)
            normalized = (data - min_val) / (max_val - min_val)
            Log.d(TAG, "Data normalized successfully.")
            return normalized
        except Exception as e:
            Log.e(TAG, f"Error normalizing data: {e}")
            raise

    def generate_curve(self, difference_factor: float = None) -> np.ndarray:
        try:
            difference_factor = difference_factor or self.initial_diff_factor
            required_columns = ["Dissipation",
                                "Resonance_Frequency", "Relative_time"]
            if not all(column in self.df.columns for column in required_columns):
                Log.e(
                    TAG, f"Input CSV must contain the following columns: {required_columns}")
                raise ValueError(
                    f"Input CSV must contain the following columns: {required_columns}")

            xs = self.df["Relative_time"]
            i = next(x for x, t in enumerate(xs) if t > 0.5)
            j = next(x for x, t in enumerate(xs) if t > 2.5)

            avg_resonance_frequency = self.df["Resonance_Frequency"][i:j].mean(
            )
            avg_dissipation = self.df["Dissipation"][i:j].mean()

            self.df["ys_diss"] = (
                self.df["Dissipation"] - avg_dissipation) * avg_resonance_frequency / 2
            self.df["ys_freq"] = avg_resonance_frequency - \
                self.df["Resonance_Frequency"]
            self.df["Difference"] = self.df["ys_freq"] - \
                difference_factor * self.df["ys_diss"]

            self.curve = np.stack(
                (self.df["Relative_time"].values, self.df["Difference"].values), axis=1)
            Log.d(TAG, "Curve generated successfully.")
            return self.curve
        except Exception as e:
            Log.e(TAG, f"Error generating curve: {e}")
            raise

    def find_extreme(self, difference: np.ndarray, resonance_frequency: np.ndarray, dissipation: np.ndarray, time: pd.Series, mode: str = "top") -> float:
        try:
            ignore_index = int(len(difference) * 0.05)
            ignore_index_2 = int(len(difference) * 0.5)
            time = time.iloc[ignore_index:ignore_index_2]
            difference = difference[ignore_index:ignore_index_2]
            trend = np.linspace(0, difference.max(), len(difference))
            adjusted_y = difference - trend

            if mode == "base":
                time = time.iloc[:self.rb["index"]]
                difference = difference[:self.rb["index"]]
                trend = np.linspace(0, difference.max(), len(difference))
                adjusted_y = difference - trend
                index = int(np.argmin(adjusted_y) - (len(time) * 0.005))
                Log.d(TAG, f"Base extreme found at time: {time.iloc[index]}.")
                return time.iloc[index], index
            else:
                index = np.argmax(adjusted_y)
                Log.d(TAG, f"Top extreme found at time: {time.iloc[index]}.")
                return time.iloc[index], index
        except Exception as e:
            Log.e(TAG, f"Error finding extreme: {e}")
            raise

    def smoothness_objective(self, difference_factor: float, lb: float, rb: float) -> float:
        try:
            curve = self.generate_curve(difference_factor)
            region_mask = (curve[:, 0] >= lb) & (curve[:, 0] <= rb)
            differences = curve[region_mask, 1]
            time_points = curve[region_mask, 0]

            slopes = np.diff(differences) / np.diff(time_points)
            step_sizes = np.diff(time_points)
            smoothness_metric = np.sum((slopes ** 2) * step_sizes)

            Log.d(TAG, f"Smoothness objective calculated: {smoothness_metric}")
            return smoothness_metric
        except Exception as e:
            Log.e(TAG, f"Error calculating smoothness objective: {e}")
            raise

    def optimize_difference_factor(self) -> float:
        try:
            self.generate_curve(self.initial_diff_factor)

            rb_time, rb_idx = self.find_extreme(
                self.df["Difference"].values, self.df["Resonance_Frequency"].values, self.df["Dissipation"].values, self.df["Relative_time"], mode="top")
            self.rb["time"] = rb_time
            self.rb["index"] = rb_idx

            lb_time, lb_idx = self.find_extreme(
                self.df["Difference"].values, self.df["Resonance_Frequency"].values, self.df["Dissipation"].values, self.df["Relative_time"], mode="base")
            self.lb["time"] = lb_time
            self.lb["index"] = lb_idx

            result = minimize(
                self.smoothness_objective,
                self.initial_diff_factor,
                args=(self.lb['time'], self.rb['time']),
                method="Nelder-Mead"
            )

            self.optimal_difference_factor = result.x[0]
            Log.d(
                TAG, f"Optimal difference factor: {self.optimal_difference_factor}")
            return self.optimal_difference_factor
        except Exception as e:
            Log.e(TAG, f"Error optimizing difference factor: {e}")
            raise

    def run(self):
        try:
            result = self.optimize_difference_factor()
            Log.d(
                TAG, f"Run completed successfully. Results: {result}, lb: {self.lb['time']}, rb: {self.rb['time']}.")
            return result, self.lb['time'], self.rb['time']
        except Exception as e:
            Log.e(TAG, f"Run failed: {e}")
            raise
