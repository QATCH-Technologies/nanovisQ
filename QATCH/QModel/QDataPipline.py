import pandas as pd
import os
import csv
import numpy as np


class QDataPipeline:
    """
    A class to handle data preprocessing for machine learning models.

    Attributes:
        __filepath__ (str): Path to the CSV file containing the data.
        __dataframe__ (DataFrame): DataFrame containing the loaded data.
    """

    def __init__(self, filepath=None):
        """
        Initializes the QDataPipeline with the given file path.

        Args:
            filepath (str, optional): Path to the input CSV file. Defaults to None.

        Raises:
            ValueError: If no file path is provided.
        """
        if filepath is not None:
            self.__filepath__ = filepath
            self.__dataframe__ = pd.read_csv(self.__filepath__)
        else:
            raise ValueError(
                f"[QDataPipeline.__init__] filepath required, found {filepath}."
            )

    def get_dataframe(self):
        """
        Returns the DataFrame containing the loaded data.

        Returns:
            DataFrame: The loaded data.
        """
        return self.__dataframe__

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

    def compute_difference(self):
        """
        Computes the 'Difference' column based on 'Dissipation' and 'Resonance_Frequency'.

        Raises:
            ValueError: If the required columns are not present in the DataFrame.
        """
        # Ensure the required columns are present
        required_columns = ["Dissipation", "Resonance_Frequency"]
        if not all(column in self.__dataframe__.columns for column in required_columns):
            raise ValueError(
                f"[QDataPipeline.compute_difference] Input CSV must contain the following columns: {required_columns}"
            )

        # Calculate the average value of 'Dissipation' and 'Resonance_Frequency' columns
        avg_resonance_frequency = self.__dataframe__["Resonance_Frequency"].mean()
        # Compute the ys_diss, ys_freq, and ys_diff
        self.__dataframe__["ys_diss"] = (
            self.__dataframe__["Dissipation"] * avg_resonance_frequency / 2
        )
        self.__dataframe__["ys_freq"] = (
            avg_resonance_frequency - self.__dataframe__["Resonance_Frequency"]
        )
        diff_factor = 2.0
        self.__dataframe__["Difference"] = (
            self.__dataframe__["ys_freq"] - diff_factor * self.__dataframe__["ys_diss"]
        )

        # Drop the intermediate columns if not needed
        self.__dataframe__.drop(columns=["ys_diss", "ys_freq"], inplace=True)

    def compute_gradient(self, column):
        self.__dataframe__[column.join("_gradient")] = np.gradient(
            self.__dataframe__[column]
        )

    def add_class(self, poi_filepath=None):
        """
        Adds a 'Class' column based on points of interest (POI) from a reference file.

        Args:
            poi_filepath (str, optional): Path to the POI CSV file. Defaults to None.

        Raises:
            ValueError: If no POI file path is provided or the file does not exist.
        """
        if poi_filepath is None:
            raise ValueError(
                "[QDataPipeline.add_class] Adding class for training requires POI reference file, found None."
            )
        if os.path.exists(poi_filepath):
            # Read POI file to get the indices
            with open(poi_filepath, "r") as poi_csv:
                reader = csv.reader(poi_csv)
                indices = [int(row[0]) - 2 for row in reader]

            # Add 'Class' column with 1 at specified indices, 0 otherwise
            self.__dataframe__["Class"] = 0
            self.__dataframe__.loc[indices, "Class"] = 1
        else:
            raise ValueError(
                f"[QDataPipeline.add_class] POI reference file does not exist at {poi_filepath}."
            )
