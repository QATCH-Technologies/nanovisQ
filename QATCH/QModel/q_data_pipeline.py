"""
QDataPipeline

A class for processing and analyzing time-series data stored in CSV files. 
This class provides methods to preprocess the data, compute various features, 
and perform filtering, interpolation, and normalization.

Modules:
    - os: For file path operations.
    - csv: For reading CSV files.
    - pandas: For data manipulation and analysis.
    - numpy: For numerical operations.
    - scipy.signal: For signal processing functions.
    - sklearn.preprocessing: For data scaling and normalization.

Classes:
    - QDataPipeline: A class to handle data preprocessing, feature extraction, 
    and filtering for time-series data.

Methods:
    - __init__(self, data_filepath: str = None, multi_class: bool = False) -> None:
        Initializes the QDataPipeline object with the path to a CSV file and an 
        optional flag for multi-class classification.

    - preprocess(self, poi_filepath: str = None) -> None:
        Performs data preprocessing including column removal, difference computation, 
        smoothing, gradient computation, noise filtering, and detrending.

    - find_time_delta(self) -> int:
        Finds the index of the first significant change in the 'Relative_time' column.

    - get_dataframe(self) -> pd.DataFrame:
        Returns the DataFrame containing the loaded data.

    - super_gradient(self, column: str) -> None:
        Computes the smoothed first derivative (gradient) of the specified column 
        and stores it in a new column with the suffix '_super'.

    - standardize(self, column):
        Standardizes the specified column using a StandardScaler.

    - remove_trend(self, column: str) -> None:
        Removes the linear trend from the specified column and stores 
        the detrended data in a new column with the suffix '_detrend'.

    - interpolate(self, num_rows: int = 20) -> None:
        Interpolates rows between specific points if 'Relative_time'
        exceeds 90. (Note: This function is currently too costly to integrate.)

    - noise_filter(self, column: str) -> None:
        Applies a Butterworth low-pass filter to remove noise from the specified column.

    - save_dataframe(self, new_filepath: str = None) -> None:
        Saves the DataFrame to a CSV file. If no new file path is provided, 
        it saves to the original file path.

    - compute_smooth(self, column, winsize=5, polyorder=1):
        Computes a smoothed version of the specified column using 
        the Savitzky-Golay filter.

    - compute_difference(self):
        Computes the difference between the 'Dissipation' and 'Resonance_Frequency' 
        columns and updates the DataFrame with the new 'Difference' column.

    - normalize_df(self):
        Normalizes the entire DataFrame by scaling values to the range [0, 1].

    - normalize_data(self, data):
        Normalizes the given data to the range [0, 1].

    - fill_nan(self):
        Replaces infinite and NaN values in the DataFrame with 0.

    - compute_gradient(self, column):
        Computes the gradient of the specified column and stores it 
        in a new column with the suffix '_gradient'.

    - add_class(self, poi_filepath=None):
        Adds class labels to the DataFrame based on the provided POI reference file. 
        Raises a ValueError if the POI file path is not provided.

equirements:
    - pandas: For CSV processing.
    - numpy: For data manipulation.
    - scipy: For signal processing.
    - sklearn: For data scaling.

Author:
    Paul MacNichol (paulmacnichol@gmail.com)
"""

import os
import csv
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt, detrend
from sklearn.preprocessing import StandardScaler


M_TARGET = "Class"


class QDataPipeline:
    """
    A data pipeline class for preprocessing and analyzing data from CSV files.

    This class provides functionality for loading, preprocessing, analyzing,
    and saving datasets.
    It includes methods for data smoothing, gradient computation, noise filtering,
    and trend removal. The class also supports handling and adding class
    labels for training purposes, as well as interpolation
    of data when necessary.

    Attributes:
        __data_filepath (str): The path to the CSV file containing the dataset.
        __dataframe (pd.DataFrame): The DataFrame containing the loaded dataset.
        multi_class (bool): Flag indicating whether the classification problem is multi-class.

    Methods:
        __init__(data_filepath: str = None, multi_class: bool = False) -> None:
            Initializes the QDataPipeline object by loading data from a CSV file.

        preprocess(poi_filepath: str = None) -> None:
            Preprocesses the data by performing various operations including column removal,
            difference computation, smoothing, gradient computation, noise filtering,
            and trend removal.

        find_time_delta() -> int:
            Finds the index of the first significant change in time delta in the
            'Relative_time' column.

        get_dataframe() -> pd.DataFrame:
            Returns the DataFrame containing the loaded data.

        super_gradient(column: str) -> None:
            Computes and stores the smoothed first derivative (gradient)
            of the specified column.

        standardize(column):
            Standardizes the values of the specified column using StandardScaler.

        remove_trend(column: str) -> None:
            Removes the linear trend from the specified column and stores the
            detrended data in a new column.

        interpolate(num_rows: int = 20) -> None:
            Interpolates rows between specific points in the DataFrame if
            'Relative_time' exceeds 90.

        noise_filter(column: str) -> None:
            Applies a Butterworth low-pass filter to remove noise from the
            specified column.

        save_dataframe(new_filepath: str = None) -> None:
            Saves the DataFrame to a CSV file at the specified path or
            the original file path.

        compute_smooth(column, winsize=5, polyorder=1):
            Applies Savitzky-Golay smoothing to the specified column with
            given window size and polynomial order.

        compute_difference():
            Computes the difference between 'Dissipation' and 'Resonance_Frequency'
            and updates related columns.

        normalize_df():
            Normalizes the entire DataFrame by scaling values between 0 and 1.

        normalize_data(data):
            Normalizes the provided data array by scaling values between 0 and 1.

        fill_nan():
            Replaces infinite and NaN values in the DataFrame with 0.

        compute_gradient(column):
            Computes and stores the gradient of the specified column in a new column.

        add_class(poi_filepath=None):
            Adds class labels to the DataFrame based on the provided POI reference file.
    """

    def __init__(self, data_filepath: str = "", multi_class: bool = False) -> None:
        """
        Initializes the QDataPipeline object.

        Parameters:
        - data_filepath (str): The path to the CSV file containing the dataset.
        - multi_class (bool): Flag indicating whether the classification problem is multi-class.
        Defaults to False.

        Raises:
        - ValueError: If `data_filepath` is not provided.
        """
        if not data_filepath == "":
            self.__data_filepath__ = data_filepath
            self.__dataframe__ = pd.read_csv(
                self.__data_filepath__
            )  # Load the CSV into a DataFrame
            self.multi_class = multi_class  # Set the multi_class flag
        else:
            raise ValueError(
                f"[QDataPipeline.__init__] Filepath required, found {data_filepath}."
            )
        self.__difference_raw__ = None

    def preprocess(self, poi_filepath: str = None) -> None:
        """
        Preprocesses the data by performing a series of operations including column removal,
        difference computation, smoothing, gradient computation, noise filtering, and trend removal.

        This method performs the following steps:
        1. Removes unnecessary columns from the DataFrame.
        2. Computes the difference between 'Dissipation' and 'Resonance_Frequency' columns.
        3. Applies smoothing to the 'Dissipation' and 'Resonance_Frequency' columns.
        4. Computes the gradient of the 'Dissipation' column.
        5. Filters out noise from the 'Dissipation' column using a low-pass filter.
        6. Removes the linear trend from the 'Dissipation' column.
        7. Interpolates missing data if the 'Relative_time' exceeds 90.
        8. Optionally, adds class labels based on the provided POI reference file.

        Args:
            poi_filepath (str, optional): The path to the POI reference file for adding
            class labels.
                If None, class labels are not added. Defaults to None.

        Returns:
            None: This method updates the DataFrame in place.
        """
        # STEP 0
        # Drop unecessary columns.
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
        # STEP 1
        # Compute the difference curve of this dataframe.
        self.compute_difference()
        self.__difference_raw__ = self.__dataframe__["Difference"]
        # OPTIONAL
        # For training datasets, add the POI flags to the dataframe.
        if poi_filepath is not None:
            self.add_class(poi_filepath)

        # Compute the 0.5% smoothing quantity for this dataset.
        smooth_win = int(
            0.005 * len(self.__dataframe__["Relative_time"].values))
        if smooth_win % 2 == 0:
            smooth_win += 1
        if smooth_win <= 1:
            smooth_win = 2

        # STEP 3
        # Compute the super gradient of the dissipation, difference, cumulative,
        # and resonance frequency curves.
        self.super_gradient("Dissipation")
        self.super_gradient("Difference")
        self.super_gradient("Cumulative")
        self.super_gradient("Resonance_Frequency")

        # STEP 4
        # Smooth the dissipation, difference, and resonance frequency curves
        # using dynamic window size and polyorder 1
        self.compute_smooth(column="Dissipation",
                            winsize=smooth_win, polyorder=1)
        self.compute_smooth(column="Difference",
                            winsize=smooth_win, polyorder=1)
        self.compute_smooth(
            column="Resonance_Frequency", winsize=smooth_win, polyorder=1
        )

        # STEP 5
        # Compute the gradient of the smooothed dissipation, difference, and resonance
        # frequency curves.
        self.compute_gradient(column="Dissipation")
        self.compute_gradient(column="Difference")
        self.compute_gradient(column="Resonance_Frequency")

        # STEP 6
        # Filter for noise among all curves eliminating lowpass jitter in the signal
        # for each curve.
        self.noise_filter("Cumulative")
        self.noise_filter("Dissipation")
        self.noise_filter("Difference")
        self.noise_filter("Resonance_Frequency")
        self.noise_filter("Difference_gradient")
        self.noise_filter("Dissipation_gradient")
        self.noise_filter("Resonance_Frequency_gradient")

        # STEP 7
        # Detrend Cumulative, Dissipation, Resonance Frequency, and Difference curves.
        self.remove_trend("Cumulative")
        self.remove_trend("Dissipation")
        self.remove_trend("Resonance_Frequency")
        self.remove_trend("Difference")
        # t_delta = self.find_time_delta()
        # if t_delta > 0:

        #     downsampling_factor = self.get_downsampling_factor()
        #     print(f"[INFO] Applying downsampling with factor of {downsampling_factor}")
        #     self.downsample(k=t_delta, factor=downsampling_factor)

    def get_downsampling_factor(self) -> int:
        # Calculate time deltas at the start and end of the dataframe
        delta_start = (
            self.__dataframe__["Relative_time"].iloc[2]
            - self.__dataframe__["Relative_time"].iloc[1]
        )
        delta_end = (
            self.__dataframe__["Relative_time"].iloc[-2]
            - self.__dataframe__["Relative_time"].iloc[-3]
        )

        # Determine the larger time delta
        larger_delta = max(delta_start, delta_end)

        # Calculate how many indices fit within the larger delta at the end
        indices_in_larger_delta = (
            self.__dataframe__["Relative_time"].iloc[-1]
            - self.__dataframe__["Relative_time"].iloc[0]
        ) // larger_delta

        # Return the downsampling factor, which is the number of indices fitting into the larger delta
        downsampling_factor = len(
            self.__dataframe__) // indices_in_larger_delta
        # return int(downsampling_factor)
        return 20

    def find_time_delta(self) -> int:
        """
        Finds the first significant time delta change in the 'Relative_time' column of the
        dataframe.

        Returns:
        - idx (int): The index of the first significant change in time delta.
                    Returns -1 if no significant change is found.
        """
        # Calculate the time difference (delta) between consecutive rows in the
        # 'Relative_time' column
        time_df = pd.DataFrame()
        time_df["Delta"] = self.__dataframe__["Relative_time"].diff()

        # Define the threshold for detecting a significant change
        threshold = 0.032

        # Calculate the expanding (cumulative) mean of the time deltas
        rolling_avg = time_df["Delta"].expanding(min_periods=2).mean()

        # Identify where the absolute difference between the delta and its
        # rolling average exceeds the threshold
        time_df["Significant_change"] = (
            time_df["Delta"] - rolling_avg
        ).abs() > threshold

        # Get the indices where a significant change occurs
        change_indices = time_df.index[time_df["Significant_change"]].tolist()

        # Determine if there is any significant change
        has_significant_change = len(change_indices) > 0

        # If a significant change is found, return the first index; otherwise, return -1
        idx = change_indices[0] if has_significant_change else -1

        return idx

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the DataFrame containing the loaded data.

        Returns:
            pd.DataFrame: The loaded data.
        """
        return self.__dataframe__

    def super_gradient(self, column: str) -> None:
        """
        Computes the smoothed first derivative of the specified column using the
        Savitzky-Golay filter.

        The resulting gradient is normalized and stored in a new column with the
        suffix '_super'.

        Parameters:
            column (str): The name of the column to compute the gradient for.

        Modifies:
            Adds a new column to the dataframe with the name '<column>_super'.
        """
        # Generate the name for the new column where the gradient will be stored
        name = f"{column}_super"

        # Extract the data from the specified column
        data = self.__dataframe__[column]

        # Determine the window size for the Savitzky-Golay filter, ensuring
        # it's an odd number and at least 3
        window = int(len(data) * 0.01)
        if window % 2 == 0:
            window += 1
        if window <= 1:
            window = 3

        # Apply the Savitzky-Golay filter to smooth the data
        smoothed_data = savgol_filter(
            x=data, window_length=window, polyorder=1, deriv=0
        )

        # Calculate the first derivative (gradient) of the smoothed data
        gradient = savgol_filter(
            x=smoothed_data, window_length=window, polyorder=1, deriv=1
        )

        # Normalize the gradient and store it in the new column
        self.__dataframe__[name] = self.normalize_data(gradient)

    def standardize(self, column: str) -> None:
        """
        Standardizes the features of the DataFrame by applying z-score normalization.

        This method standardizes the features by subtracting the mean and dividing by the
        standard deviation for each feature column in the DataFrame. The standardization
        is applied to ensure that all features have a mean of 0 and a standard
        deviation of 1, which helps in improving the performance of
        machine learning algorithms that are sensitive to feature scaling.

        Args:
            column (str): The name of the column to standardize.

        Returns:
            None: This method updates the DataFrame in place.
        """
        # Define standard scaler
        scaler = StandardScaler()
        df = self.__dataframe__
        # Transform data
        self.__dataframe__[column] = scaler.fit_transform(
            np.array(df[column]).reshape(-1, 1)
        )

    def remove_trend(self, column: str) -> None:
        """
        Removes the linear trend from the specified column of the dataframe.

        The detrended data is stored in a new column with the suffix '_detrend'.

        Parameters:
            column (str): The name of the column from which the trend will be removed.

        Modifies:
            Adds a new column to the dataframe with the name '<column>_detrend'.
        """
        # Generate the name for the new column where the detrended data will be stored
        name = f"{column}_detrend"

        # Remove the linear trend from the specified column and store it in the new column
        self.__dataframe__[name] = detrend(
            data=self.__dataframe__[column].values)

    # def interpolate(self, num_rows: int = 20) -> None:
    #     """
    #     *** THIS FUNCTION IS CURRENTLY TOO COSTLY TO INTEGRATE
    #     Interpolates the dataframe rows between specific points if the 'Relative_time'
    #     exceeds 90.

    #     Parameters:
    #         num_rows (int): The number of rows to interpolate between each pair of points.
    #         Defaults to 20.

    #     Modifies:
    #         Updates the dataframe with the interpolated rows, adding them after the
    #         specified start index.
    #     """
    #     df = self.__dataframe__

    #     # Check if 'Relative_time' exceeds 90 and find the start index
    #     if max(df["Relative_time"].values) > 90:
    #         start_index = df.index[df["Relative_time"] > 90].tolist()[0]

    #         # Initialize an empty DataFrame to store the result
    #         result = pd.DataFrame(columns=df.columns)

    #         # Iterate through the dataframe to interpolate rows where needed
    #         for i in range(len(df) - 1):
    #             # Append the current row to the result DataFrame
    #             result = pd.concat(
    #                 [result, pd.DataFrame(df.iloc[i]).transpose()],
    #                 ignore_index=True,
    #             )

    #             # If the current index is greater than or equal to start_index, interpolate rows
    #             if i >= start_index:
    #                 start_row = df.iloc[i]
    #                 end_row = df.iloc[i + 1]

    #                 # Prepare a DataFrame to store interpolated rows
    #                 interpolated_rows = pd.DataFrame()

    #                 for col in df.columns:
    #                     if col in ["Class_1", "Class_2"]:
    #                         # If the original value is 0, interpolate to 0
    #                         if start_row[col] == 0:
    #                             interpolated_rows[col] = np.repeat(0, num_rows, axis=0)
    #                         else:
    #                             # Otherwise, repeat the start value and fill the remaining with 0
    #                             interpolated_rows[col] = np.concatenate(
    #                                 (
    #                                     np.repeat(start_row[col], 1, axis=0),
    #                                     np.repeat(0, num_rows - 1, axis=0),
    #                                 )
    #                             )
    #                     else:
    #                         # Linearly interpolate other columns
    #                         interpolated_rows[col] = np.linspace(
    #                             start_row[col], end_row[col], num_rows
    #                         )

    #                 # Append the interpolated rows to the result DataFrame
    #                 result = pd.concat([result, interpolated_rows], ignore_index=True)

    #         # Append the last row of the original DataFrame to the result
    #         result = pd.concat(
    #             [result, pd.DataFrame(df.iloc[-1]).transpose()],
    #             ignore_index=True,
    #         )

    #         # Update the original dataframe with the interpolated result
    #         self.__dataframe__ = result

    def noise_filter(self, column: str) -> None:
        """
        Applies a Butterworth low-pass filter to remove noise from the specified
        column of the dataframe.

        The filtered data replaces the original data in the specified column.

        Parameters:
            column (str): The name of the column to filter.

        Raises:
            ValueError: If the column data is empty.
        """
        # Extract the data from the specified column
        data = self.__dataframe__[column]

        # Validate that the column is not empty
        if data.size == 0:
            raise ValueError(
                "[QModelPredict noise_filter()]: `predictions` cannot be an empty array."
            )

        # Define Butterworth low-pass filter parameters
        fs = 300  # Sampling frequency
        normal_cutoff = 2 / (0.5 * fs)  # Normalize the cutoff frequency

        # Get the filter coefficients for a 2nd order low-pass Butterworth filter
        b, a = butter(2, Wn=normal_cutoff, btype="lowpass", analog=False)

        # Apply the filter to the data using filtfilt for zero-phase filtering
        filtered = filtfilt(b, a, data)

        # Update the dataframe with the filtered data
        self.__dataframe__[column] = filtered

        # Ensure that no values are negative by setting any negative values to 0
        self.__dataframe__[column] = self.__dataframe__[column].apply(
            lambda x: max(0, x)
        )

    def save_dataframe(self, new_filepath: str = None) -> None:
        """
        Saves the DataFrame to a CSV file.

        Parameters:
            new_filepath (str, optional): The path where the CSV file will be saved.
                                          If not provided, the DataFrame will be saved to
                                          the original file path.

        Raises:
            ValueError: If the original file path is not set and no new file path is provided.
        """
        # Determine the file path to save the DataFrame
        filepath = new_filepath if new_filepath else self.__data_filepath__

        # Check if the file path is valid
        if not filepath:
            raise ValueError(
                "[QDataPipeline save_dataframe()]: No file path provided to save the DataFrame."
            )

        # Save the DataFrame to the specified CSV file
        self.__dataframe__.to_csv(filepath, index=False)

    def compute_smooth(self, column: str, winsize: int = 5, polyorder: int = 1) -> None:
        """
        Computes the smoothed version of the DataFrame's data.

        This method applies a smoothing technique to the data in the DataFrame to reduce noise and
        highlight trends. The smoothing process typically involves averaging or other techniques
        to create a smoother representation of the data, which can be useful for
        visualizations and analysis.

        Args:
             column (str): The name of the column in the DataFrame to be smoothed.
            winsize (int, optional): The size of the window used for smoothing. Default is 5.
                This determines the number of points used for each smoothing calculation.
            polyorder (int, optional): The order of the polynomial used for smoothing. Default is 1.
                This determines the degree of the polynomial used in the smoothing process.

        Returns:
            None: This method updates the DataFrame in place with the smoothed data.
        """
        self.__dataframe__[column] = savgol_filter(
            self.__dataframe__[column], winsize, polyorder
        )

    def compute_difference(self) -> None:
        """
        Computes the difference between consecutive values in the specified column.

        This method calculates the difference between each consecutive pair of values in the
        specified column of the DataFrame. The resulting difference values are used to
        analyze trends, changes, or fluctuations within the data series.

        Args:
            None

        Returns:
            None: This method updates the DataFrame with a new column containing the
            computed differences.
        """
        # Ensure the required columns are present
        required_columns = ["Dissipation", "Resonance_Frequency"]
        if not all(column in self.__dataframe__.columns for column in required_columns):
            raise ValueError(
                "[QDataPipeline.compute_difference] Input CSV must contain the"
                + f" following columns: {required_columns}"
            )

        # Calculate the average value of 'Dissipation' and 'Resonance_Frequency' columns
        xs = self.__dataframe__["Relative_time"]
        i = next(x + 0 for x, t in enumerate(xs) if t > 0.5)
        j = next(x + 1 for x, t in enumerate(xs) if t > 2.5)
        avg_resonance_frequency = self.__dataframe__[
            "Resonance_Frequency"][i:j].mean()
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
            self.__dataframe__["ys_freq"] -
            diff_factor * self.__dataframe__["ys_diss"]
        )

        # Drop the intermediate columns if not needed
        self.__dataframe__[
            "Resonance_Frequency"] = self.__dataframe__["ys_freq"]
        self.__dataframe__["Dissipation"] = self.__dataframe__["ys_diss"]
        self.__dataframe__["Cumulative"] = savgol_filter(
            self.__dataframe__["Dissipation"].values
            + self.__dataframe__["Resonance_Frequency"].values,
            25,
            1,
            1,
        )
        self.__dataframe__.drop(columns=["ys_diss", "ys_freq"], inplace=True)

    def normalize_df(self) -> None:
        """
        Normalizes the DataFrame to a [0, 1] range.

        This method applies min-max normalization to the DataFrame. It scales the
        values in each column so that the minimum value becomes 0 and the
        maximum value becomes 1. The normalization formula
        used is: (value - min) / (max - min).

        Returns:
            None: This method updates the DataFrame in place.
        """
        self.__dataframe__ = (self.__dataframe__ - self.__dataframe__.min()) / (
            self.__dataframe__.max() - self.__dataframe__.min()
        )

    def normalize_data(self, data) -> None:
        """
        Normalizes the input data to a [0, 1] range.

        This method applies min-max normalization to the provided data. It scales the
        values so that the minimum value becomes 0 and the maximum value becomes 1.
        The normalization formula used is:
        (value - min) / (max - min).

        Args:
            data (np.ndarray or list-like): The data to be normalized. Must be a numeric
            array-like structure.

        Returns:
            np.ndarray: The normalized data, with values scaled to the range [0, 1].

        Examples:
            >>> data = np.array([1, 2, 3, 4, 5])
            >>> normalize_data(data)
            array([0. , 0.25, 0.5 , 0.75, 1. ])
        """
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def fill_nan(
        self,
    ):
        """
        Replaces NaN, positive infinity, and negative infinity values in the dataframe with zeros.

        This method updates the dataframe by replacing any occurrences of NaN (Not a Number),
        positive infinity (`np.inf`), and negative infinity (`-np.inf`) with zeros.
        This is useful for handling missing or infinite values before performing further
        data processing or analysis.

        Args:
            None

        Returns:
            None

        Examples:
            >>> df = pd.DataFrame({'A': [1, np.nan, np.inf], 'B': [-np.inf, 2, 3]})
            >>> pipeline.fill_nan()
            >>> df
            A  B
            0  1  0
            1  0  2
            2  0  3
        """
        self.__dataframe__.replace([np.inf, -np.inf], 0)
        self.__dataframe__.replace(np.nan, 0)

    def downsample(self, k: int, factor: int) -> None:
        """
        Downsample the dataframe up to position `k` by keeping every `factor`-th row.

        If there is a value other than 0 in the "Class" column, remap it to the nearest downsampled point.

        Parameters:
        k (int): Position up to which downsampling is performed.
        factor (int): Downsample by this factor.
        mode (int): Prediction mode (1) ignores class column.  Training mode (0) adds training column
            downsampling.
        """
        if k > len(self.__dataframe__):
            raise ValueError("k is out of bounds for the dataframe length.")

        original_df = self.__dataframe__
        # Select the part to downsample
        df_part = original_df.iloc[:k]
        # Select the rest of the dataframe
        df_rest = original_df.iloc[k:]
        resampled_df = pd.concat([df_part.iloc[::factor], df_rest])
        if "Class" in original_df.columns:
            nonzero_class_rows = original_df[original_df["Class"] != 0]
            resampled_df = pd.concat([resampled_df, nonzero_class_rows])
            resampled_df = resampled_df.sort_values(by="Relative_time")

            # import matplotlib.pyplot as plt
            # import seaborn as sns

            # nonzero_locations = resampled_df[resampled_df["Class"] != 0].index[:6]
            # palette = sns.color_palette("husl", 7)
            # plt.figure(figsize=(8, 8))
            # plt.plot(resampled_df["Dissipation"])
            # for i, loc in enumerate(nonzero_locations):
            #     plt.axvline(loc, label=f"POI {i+1}", color=palette[i])
            # plt.plot(resampled_df["Class"])
            # plt.legend()
            # plt.show()
        self.__dataframe__ = resampled_df.reset_index(drop=True)

    def compute_gradient(self, column: str) -> None:
        """
        Computes the gradient (difference) of the specified column and adds it as a new column.

        This method calculates the gradient of the specified column in the dataframe by computing
        the difference between consecutive values. The result is added to the dataframe as a new
        column with the name `<column>_gradient`.

        Args:
            column (str): The name of the column for which to compute the gradient.

        Returns:
            None

        Examples:
            >>> df = pd.DataFrame({'A': [1, 2, 4, 7]})
            >>> pipeline.compute_gradient('A')
            >>> df
            A  A_gradient
            0  1         NaN
            1  2         1.0
            2  4         2.0
            3  7         3.0
        """
        # Compute the gradient of the column
        gradient = self.__dataframe__[column].diff()
        # Create the new column name
        gradient_column_name = column + "_gradient"

        # Add the gradient as a new column to the dataframe
        self.__dataframe__[gradient_column_name] = gradient

    def add_class(self, poi_filepath: str = "") -> None:
        """
        Adds class labels to the dataframe based on the POI reference file.

        This method reads a POI (Point of Interest) reference file to obtain indices and
        updates the dataframe with class labels. If `multi_class` is True, a single target
        column is updated with integer class labels. Otherwise, binary columns for each
        class are created and updated.

        Args:
            poi_filepath (str, optional): Path to the POI reference file in CSV format.
                                        Each row should contain an index indicating
                                        where to add class labels. If None, a ValueError
                                        is raised.

        Raises:
            ValueError: If `poi_filepath` is None, or if the file does not exist at the
                        specified path.

        Returns:
            None

        Examples:
            >>> pipeline.add_class('path/to/poi_file.csv')
        """
        if poi_filepath == "":
            raise ValueError(
                "[QDataPipeline.add_class] Adding class for training requires POI"
                + "reference file, found None."
            )
        if os.path.exists(poi_filepath):
            # Read POI file to get the indices
            with open(poi_filepath, "r", encoding="utf-8") as poi_csv:
                reader = csv.reader(poi_csv)
                indices = [int(row[0]) - 2 for row in reader]
            if self.multi_class:
                self.__dataframe__[M_TARGET] = 0
                for poi, idx in enumerate(indices):
                    self.__dataframe__.loc[idx, M_TARGET] = poi + 1
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
