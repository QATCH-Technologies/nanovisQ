import pandas as pd
import os
import csv
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt, detrend
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

# ModelData_found = False
# try:
#     if not ModelData_found:
#         from ModelData import ModelData
#     ModelData_found = True
# except:
#     ModelData_found = False
# try:
#     if not ModelData_found:
#         from QATCH.models.ModelData import ModelData
#     ModelData_found = True
# except:
#     ModelData_found = False
# if not ModelData_found:
#     raise ImportError("Cannot find 'ModelData' in any expected location.")

DOWNSAMPLE_AFTER = 90
DOWNSAMPLE_COUNT = 20
DOWNSAMPLE_THRESHOLD = 0.6

TARGET_ALL = "Class"
TARGET_1 = "Class_1"
TARGET_2 = "Class_2"
TARGET_3 = "Class_3"
TARGET_4 = "Class_4"
TARGET_5 = "Class_5"
TARGET_6 = "Class_6"


class QDataPipeline:
    def __init__(self, filepath=None, multi_class=False):
        if filepath is not None:
            self.__filepath__ = filepath
            self.__dataframe__ = pd.read_csv(self.__filepath__)
            self.pca_df = pd.DataFrame()
            self.__interpolation_size__ = 0
            self.__multi_class__ = multi_class
        else:
            raise ValueError(
                f"[QDataPipeline.__init__] filepath required, found {filepath}."
            )

    def preprocess(self, poi_file=None):
        self.__dataframe__.drop(
            columns=[
                "Date",
                "Time",
                "Ambient",
                "Temperature",
                "Peak Magnitude (RAW)",
            ],
            inplace=True,
        )

        # self.trim_head()
        self.compute_difference()
        opt_diss, opt_res, opt_diff = 0, 0, 0
        if poi_file is not None:
            self.add_class(poi_file)

        smooth_win = int(0.005 * len(self.__dataframe__["Relative_time"].values))
        if smooth_win % 2 == 0:
            smooth_win += 1
        if smooth_win <= 1:
            smooth_win = 2
        opt_diss, opt_res, opt_diff = smooth_win, smooth_win, smooth_win
        # print(f"Optimal smoothing, diff:{opt_diff}, diss:{opt_diss}, res:{opt_res}")

        # self.interpolate()
        self.super_gradient("Dissipation")
        self.super_gradient("Difference")
        self.super_gradient("Cumulative")
        self.super_gradient("Resonance_Frequency")

        self.compute_smooth(column="Dissipation", winsize=opt_diss, polyorder=1)
        self.compute_smooth(column="Difference", winsize=opt_diff, polyorder=1)
        self.compute_smooth(column="Resonance_Frequency", winsize=opt_res, polyorder=1)

        self.compute_gradient(column="Dissipation")
        self.compute_gradient(column="Difference")
        self.compute_gradient(column="Resonance_Frequency")

        self.noise_filter("Cumulative")
        self.noise_filter("Dissipation")
        self.noise_filter("Difference")
        self.noise_filter("Resonance_Frequency")
        self.noise_filter("Difference_gradient")
        self.noise_filter("Dissipation_gradient")
        self.noise_filter("Resonance_Frequency_gradient")

        self.remove_trend("Cumulative")
        self.remove_trend("Dissipation")
        self.remove_trend("Resonance_Frequency")
        self.remove_trend("Difference")
        # emp_predictor = ModelData()
        # emp_result = emp_predictor.IdentifyPoints(data_path=self.__filepath__)
        # emp_result = [
        #     item if isinstance(item, int) else [val[0] for val in item]
        #     for item in emp_result
        #     if isinstance(item, (int, list))
        # ]
        # emp_result = [
        #     item
        #     for sublist in emp_result
        #     for item in (sublist if isinstance(sublist, list) else [sublist])
        # ]
        # self.__dataframe__["EMP"] = 0
        # for idx in emp_result:
        #     self.__dataframe__.loc[idx, "EMP"] = 1
        # self.__dataframe__.reset_index()

    def apply_lda(self, target):
        pass

    def find_time_delta(self):
        time_df = pd.DataFrame()
        time_df["Delta"] = self.__dataframe__["Relative_time"].diff()

        threshold = 0.032

        rolling_avg = time_df["Delta"].expanding(min_periods=2).mean()

        time_df["Significant_change"] = (
            time_df["Delta"] - rolling_avg
        ).abs() > threshold
        change_indices = time_df.index[time_df["Significant_change"]].tolist()
        has_significant_change = len(change_indices) > 0
        idx = -1
        if has_significant_change:
            idx = change_indices[0]
        return idx

    def get_dataframe(self):
        """
        Returns the DataFrame containing the loaded data.

        Returns:
            DataFrame: The loaded data.
        """
        return self.__dataframe__

    def super_gradient(self, column):
        name = column + "_super"
        data = self.__dataframe__[column]
        window = int(len(data) * 0.01)
        if window % 2 == 0:
            window += 1
        if window <= 1:
            window = 3

        data = savgol_filter(x=data, window_length=window, polyorder=1, deriv=0)
        self.__dataframe__[name] = self.normalize_data(
            savgol_filter(x=data, window_length=window, polyorder=1, deriv=1)
        )

    def standardize(self, column):
        # define standard scaler
        scaler = StandardScaler()
        df = self.__dataframe__
        # transform data
        self.__dataframe__[column] = scaler.fit_transform(
            np.array(df[column]).reshape(-1, 1)
        )

    def remove_trend(self, column):
        name = column + "_detrend"
        self.__dataframe__[name] = detrend(data=self.__dataframe__[column].values)

    def interpolate(self, num_rows=20):
        if max(self.__dataframe__["Relative_time"].values) > 90:
            df = self.__dataframe__
            start_index = df.index[df["Relative_time"] > 90].tolist()[0]
            result = pd.DataFrame(columns=df.columns)

            for i in range(len(df) - 1):
                # Append the current row to the result DataFrame
                result = pd.concat(
                    [
                        result if not result.empty else None,
                        pd.DataFrame(df.iloc[i]).transpose(),
                    ],
                    ignore_index=True,
                )
                # If the current index is greater than or equal to start_index, interpolate rows
                if i >= start_index:
                    self.__interpolation_size__ += 1
                    start_row = df.iloc[i]
                    end_row = df.iloc[i + 1]

                    # Interpolate columns excluding "Class"
                    interpolated_rows = pd.DataFrame()

                    for col in df.columns:
                        if col == "Class_1" or col == "Class_2":
                            interpolated_rows[col] = np.repeat(
                                start_row[col], num_rows, axis=0
                            )
                            # Check if the original value is 0, then interpolate to 0
                            if start_row[col] == 0:
                                interpolated_rows[col] = np.repeat(0, num_rows, axis=0)
                            else:
                                interpolated_rows[col] = start_row[col]
                                np.concatenate(
                                    (
                                        interpolated_rows[col],
                                        np.repeat(0, num_rows - 1, axis=0),
                                    ),
                                )
                        else:
                            interpolated_rows[col] = np.linspace(
                                start_row[col], end_row[col], num_rows
                            )

                    result = pd.concat(
                        [result, interpolated_rows],
                        ignore_index=True,
                    )

            # Append the last row of the original DataFrame
            result = pd.concat(
                [
                    result if not result.empty else None,
                    pd.DataFrame(df.iloc[-1]).transpose(),
                ],
                ignore_index=True,
            )

            self.__dataframe__ = result

    def noise_filter(self, column):
        # Validate `predictions`
        data = self.__dataframe__[column]
        if data.size == 0:
            raise ValueError(
                "[QModelPredict noise_filter()]: `predictions` cannot be an empty array."
            )

        # Butterworth low-pass filter parameters
        fs = 300
        normal_cutoff = 2 / (0.5 * fs)
        # Get the filter coefficients
        b, a = butter(2, Wn=[normal_cutoff], btype="lowpass", analog=False)

        # Apply the filter using filtfilt
        filtered = filtfilt(b, a, data)

        self.__dataframe__[column] = filtered
        self.__dataframe__[column] = self.__dataframe__[column].apply(
            lambda x: max(0, x)
        )
        # abs(
        #     self.__dataframe__[column]
        #     - savgol_filter(self.__dataframe__[column], 25, 1)
        # )

    def save_dataframe(self, new_filepath=None):
        """
        Saves the DataFrame to a CSV file.

        Args:
            new_filepath (str, optional): Path to save the CSV file. Defaults to None.

        If no new file path is provided, it saves to the original file path.
        """
        if new_filepath:
            self.__dataframe__.to_csv(new_filepath, index=False)
        else:
            self.__dataframe__.to_csv(self.__filepath__, index=False)

    def optimal_smooth_values(self, column, poi_points):
        data = self.__dataframe__[column].values
        max_corr = -1

        best_window_length = None
        value = int(len(data) * 0.01)
        if value % 2 == 0:
            value += 1
        window_lengths = range(3, value, 2)  # Window lengths must be odd and at least 3
        correlations = []

        for window_length in window_lengths:
            if window_length <= len(data):
                smoothed_data = savgol_filter(data, window_length, polyorder=1)
                corr, _ = pearsonr(smoothed_data, poi_points)
                correlations.append(corr)
                if corr > max_corr:
                    max_corr = corr
                    best_window_length = window_length
        return best_window_length, max_corr

    def compute_smooth(self, column, winsize=5, polyorder=1):
        self.__dataframe__[column] = savgol_filter(
            self.__dataframe__[column], winsize, polyorder
        )

    def compute_difference(self):
        # Ensure the required columns are present
        required_columns = ["Dissipation", "Resonance_Frequency"]
        if not all(column in self.__dataframe__.columns for column in required_columns):
            raise ValueError(
                f"[QDataPipeline.compute_difference] Input CSV must contain the following columns: {required_columns}"
            )

        # Calculate the average value of 'Dissipation' and 'Resonance_Frequency' columns
        xs = self.__dataframe__["Relative_time"]
        i = next(x + 0 for x, t in enumerate(xs) if t > 0.5)
        j = next(x + 1 for x, t in enumerate(xs) if t > 2.5)
        avg_resonance_frequency = self.__dataframe__["Resonance_Frequency"][i:j].mean()
        avg_dissipation = self.__dataframe__["Dissipation"][i:j].mean()
        # Compute the ys_diss, ys_freq, and ys_diff
        self.__dataframe__["ys_diss"] = (
            (self.__dataframe__["Dissipation"] - avg_dissipation)
            * avg_resonance_frequency
            / 2
        )
        self.__dataframe__["ys_freq"] = (
            avg_resonance_frequency - self.__dataframe__["Resonance_Frequency"]
        )
        diff_factor = 1.5
        self.__dataframe__["Difference"] = (
            self.__dataframe__["ys_freq"] - diff_factor * self.__dataframe__["ys_diss"]
        )

        # Drop the intermediate columns if not needed
        self.__dataframe__["Resonance_Frequency"] = self.__dataframe__["ys_freq"]
        self.__dataframe__["Dissipation"] = self.__dataframe__["ys_diss"]
        self.__dataframe__["Cumulative"] = savgol_filter(
            self.__dataframe__["Dissipation"].values
            + self.__dataframe__["Resonance_Frequency"].values,
            25,
            1,
            1,
        )
        self.__dataframe__.drop(columns=["ys_diss", "ys_freq"], inplace=True)

    def normalize_df(self):
        self.__dataframe__ = (self.__dataframe__ - self.__dataframe__.min()) / (
            self.__dataframe__.max() - self.__dataframe__.min()
        )

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def fill_nan(
        self,
    ):
        self.__dataframe__.replace([np.inf, -np.inf], 0)
        self.__dataframe__.replace(np.nan, 0)

    def trim_head(self):
        data = self.__dataframe__["Dissipation"]
        factor = int(len(data) * 0.03)
        data = data.values[factor:]
        savgol_filter(data, factor, 1)

        # The start of the critical region for initial fill.
        start = 0
        # Slopes to determine the start of this region.
        start_slopes = []

        # A buffer for the first 1% of the input data set.
        # Ensures any sensor interference does not affect the slope
        # calculation.
        buffer = int(len(data) * 0.01)
        start_slopes = np.where(
            np.arange(start + 1, len(data)) < buffer,
            0,
            (data[start + 1 :] - data[start]) / np.arange(start + 1, len(data) - start),
        )

        # Compute where there is a significant positive change in the start slopes.
        # This index gets returned as the start index of the critical region.
        start_slopes = self.normalize_data(start_slopes)
        start_tmp = 1

        for i in range(1, len(start_slopes)):
            if start_slopes[i] < start_slopes[start] + 0.1:
                start_tmp = i
        self.__dataframe__ = self.__dataframe__.loc[start + start_tmp :]
        # self.__dataframe__ = self.__dataframe__.reset_index(drop=True)
        self.head_trim = start + start_tmp - factor

    def compute_gradient(self, column):
        # Compute the gradient of the column
        gradient = self.__dataframe__[column].diff()
        # Create the new column name
        gradient_column_name = column + "_gradient"

        # Add the gradient as a new column to the dataframe
        self.__dataframe__[gradient_column_name] = gradient

    def add_class(self, poi_filepath=None):
        if poi_filepath is None:
            raise ValueError(
                "[QDataPipeline.add_class] Adding class for training requires POI reference file, found None."
            )
        if os.path.exists(poi_filepath):
            # Read POI file to get the indices
            with open(poi_filepath, "r") as poi_csv:
                reader = csv.reader(poi_csv)
                indices = [int(row[0]) - 2 for row in reader]
            if self.__multi_class__:
                self.__dataframe__["Class"] = 0
                for poi, idx in enumerate(indices):
                    self.__dataframe__.loc[idx, "Class"] = poi + 1
            else:
                self.__dataframe__["Class_1"] = 0
                self.__dataframe__["Class_2"] = 0
                self.__dataframe__["Class_3"] = 0
                self.__dataframe__["Class_4"] = 0
                self.__dataframe__["Class_5"] = 0
                self.__dataframe__["Class_6"] = 0
                for poi, idx in enumerate(indices):
                    self.__dataframe__.loc[idx, "Class_" + str(poi + 1)] = 1

        else:
            raise ValueError(
                f"[QDataPipeline.add_class] POI reference file does not exist at {poi_filepath}."
            )
