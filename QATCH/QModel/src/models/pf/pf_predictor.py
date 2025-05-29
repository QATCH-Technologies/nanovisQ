"""
pf_predictor.py

This module defines the PFPredictor class, which encapsulates loading
and applying a preprocessing scaler and an XGBoost booster model to predict
the number of points of interest (POIs) in partial-fill (PF) runs.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    05-14-2025

Version:
    V1.0.1
"""

import os
import pickle
from typing import Union, Any
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from QATCH.common.logger import Logger as Log
from QATCH.QModel.src.models.pf.pf_data_processor import PFDataProcessor

TAG = '[PartialFill]'


class PFPredictor:
    """
    Standalone predictor for partially filled runs. Handles loading model artifacts,
    preprocessing, and outputting POI count predictions for individual runs.

    Attributes:
        model_dir (str): Directory containing model artifacts.
        scaler (Any): Preprocessing scaler pipeline.
        booster (xgb.Booster): Trained XGBoost model.
    """

    def __init__(self, model_dir: str) -> None:
        """
        Initialize the PFPredictor by validating the model directory and loading
        the scaler and booster from disk.

        Args:
            model_dir (str): Path to the directory containing the trained model
                artifacts (scaler and booster).

        Raises:
            NotADirectoryError: If the specified model_dir does not exist or is
                not a directory.

        Attributes:
            model_dir (str): The validated path to the model directory.
            scaler (sklearn.preprocessing._data.StandardScaler): The loaded
                scaler used for feature normalization.
            booster (xgboost.core.Booster): The loaded XGBoost booster model for
                making predictions.
        """
        Log.d(TAG, f"Initializing PFPredictor with model_dir={model_dir}")
        if not os.path.isdir(model_dir):
            Log.e(TAG, f"Model directory does not exist: {model_dir}")
            raise NotADirectoryError(
                f"Model directory does not exist: {model_dir}")
        self.model_dir = model_dir
        self.scaler = self._load_scaler()
        self.booster = self._load_booster()
        Log.i(TAG, "PFPredictor initialized successfully.")

    def _load_scaler(self) -> Any:
        """
        Load the scaler pipeline from a pickle file stored in the model directory.

        Returns:
            Any: The deserialized scaler object used for feature normalization.

        Raises:
            FileNotFoundError: If the scaler pickle file does not exist at the expected path.
            Exception: If an error occurs during file opening or unpickling.
        """
        scaler_path = os.path.join(self.model_dir, 'pf_scaler.pkl')
        Log.d(TAG, f"Attempting to load scaler from {scaler_path}")
        if not os.path.isfile(scaler_path):
            Log.e(TAG, f"Scaler file not found: {scaler_path}")
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            Log.i(TAG, "Scaler loaded successfully.")
            return scaler
        except Exception as e:
            Log.e(TAG, f"Failed to load scaler: {e}")
            raise

    def _load_booster(self) -> xgb.Booster:
        """
        Load the XGBoost Booster model from a JSON file stored in the model directory.

        Returns:
            xgb.Booster: The loaded XGBoost booster model used for making predictions.

        Raises:
            FileNotFoundError: If the booster JSON file does not exist at the expected path.
            Exception: If an error occurs during booster instantiation or loading.
        """
        booster_path = os.path.join(self.model_dir, 'pf_booster.json')
        Log.d(TAG, f"Attempting to load booster from {booster_path}")
        if not os.path.isfile(booster_path):
            Log.e(TAG, f"Booster file not found: {booster_path}")
            raise FileNotFoundError(f"Booster file not found: {booster_path}")
        try:
            booster = xgb.Booster()
            booster.load_model(booster_path)
            Log.i(TAG, "Booster loaded successfully.")
            return booster
        except Exception as e:
            Log.e(TAG, f"Failed to load booster: {e}")
            raise

    def _reset_file_buffer(self, file_buffer: str):
        """Ensure the file buffer is positioned at its beginning for reading.

        If `file_buffer` is a file path (string), it is returned unchanged.
        If it is a seekable file-like object, its position is reset to zero.
        Otherwise, an exception is raised.

        Args:
            file_buffer (str or file-like): Either a filesystem path (string)
                or a file-like object supporting `seek`.

        Returns:
            str or file-like: The original `file_buffer`, with its read pointer
            reset if applicable.

        Raises:
            Exception: If `file_buffer` is a non-seekable stream and thus cannot
                be rewound.
        """
        if isinstance(file_buffer, str):
            return file_buffer
        if hasattr(file_buffer, "seekable") and file_buffer.seekable():
            file_buffer.seek(0)
            return file_buffer
        else:
            raise Exception(
                "Cannot `seek` stream prior to passing to processing.")

    def _validate_file_buffer(self, file_buffer: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Load and validate a CSV data file into a pandas DataFrame.

        This method ensures the file buffer is reset or accepts a string path,
        then attempts to read it as CSV. It handles common read errors, checks
        for emptiness, and verifies the presence of required columns.

        Args:
            file_buffer (str or file-like): Either a filesystem path pointing to
                a CSV file, or a file-like object containing CSV data.

        Returns:
            pd.DataFrame: A DataFrame containing the CSV data with at least the
            columns "Dissipation", "Resonance_Frequency", and "Relative_time".

        Raises:
            ValueError: If the buffer cannot be reset, the file is empty,
                        parsing fails, the resulting DataFrame is empty, or
                        required columns are missing.
        """
        # Reset buffer if necessary
        if isinstance(file_buffer, pd.DataFrame):
            return file_buffer
        try:
            file_buffer = self._reset_file_buffer(file_buffer=file_buffer)
        except Exception:
            raise ValueError(
                "File buffer must be a non-empty string containing CSV data.")

        # Read CSV into DataFrame
        try:
            df = pd.read_csv(file_buffer)
        except pd.errors.EmptyDataError:
            raise ValueError("The provided data file is empty.")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing data file: `{e}`")
        except Exception as e:
            raise ValueError(
                f"An unexpected error occurred while reading the data file: `{e}`")

        # Validate DataFrame contents
        if df.empty:
            raise ValueError("The data file does not contain any data.")

        required_columns = {"Dissipation",
                            "Resonance_Frequency", "Relative_time"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Data file missing required columns: `{', '.join(missing)}`.")

        return df

    def predict(
        self,
        file_buffer: Union[str, pd.DataFrame],
        show_plot: bool = False
    ) -> int:
        """
        Predict the number of points of interest (POIs) for a single PF run.

        Args:
            file_buffer (Union[str, pd.DataFrame]): File path or DataFrame containing
                the PF run data. Must include a 'Dissipation' column.
            show_plot (bool): Whether to plot the dissipation curve before prediction.
                Defaults to False.

        Returns:
            int: Predicted POI class label.

        Raises:
            ValueError: If input lacks the required 'Dissipation' column or if feature
                generation or scaling fails.
            RuntimeError: If the model prediction step encounters an error.
        """
        Log.d(TAG, "Entering predict()")
        df = self._validate_file_buffer(file_buffer=file_buffer)

        # Validate required column
        if 'Dissipation' not in df.columns:
            Log.e(TAG, "Missing 'Dissipation' column in DataFrame")
            raise ValueError("Input data must contain a 'Dissipation' column.")

        # Optional plotting
        if show_plot:
            Log.d(TAG, "Plotting dissipation curve")
            try:
                plt.figure()
                plt.plot(df['Dissipation'])
                plt.xlabel('Index')
                plt.ylabel('Dissipation')
                plt.title('Dissipation Curve')
                plt.show()
                Log.i(TAG, "Dissipation curve plotted")
            except Exception as e:
                Log.w(TAG, f"Unable to plot dissipation curve: {e}")

        # Feature generation
        if len(df) <= PFDataProcessor.SAMPLE_FACTOR:
            Log.i(
                TAG, f"Run contains less than {PFDataProcessor.SAMPLE_FACTOR:} samples.")
            return 0

        try:
            Log.d(TAG, "Generating features")

            features = PFDataProcessor.generate_features(
                dataframe=df,
                sampling_rate=1.0,
            )
            Log.d(TAG, f"Features generated: {features.shape[1]} columns")
        except Exception as e:
            Log.e(TAG, f"Feature generation failed: {e}")
            raise ValueError(
                "Failed to generate features from input data.") from e

        # Feature scaling
        try:
            Log.d(TAG, "Scaling features")
            X_scaled = self.scaler.transform(features)
            Log.d(TAG, f"Features scaled: {X_scaled.shape}")
        except Exception as e:
            Log.e(TAG, f"Scaling features failed: {e}")
            raise ValueError("Feature scaling failed.") from e

        # Model prediction
        try:
            Log.d(TAG, "Predicting probabilities with PF booster")
            dmatrix = xgb.DMatrix(X_scaled)
            probs = self.booster.predict(dmatrix)
            Log.d(TAG, f"Raw prediction output: {probs}")
            pred = int(np.argmax(probs, axis=1)[0])
            Log.d(TAG, f"Predicted class: {pred}")
        except Exception as e:
            Log.e(TAG, f"Prediction error: {e}")
            raise RuntimeError("Model prediction failed.") from e

        Log.d(TAG, "Exiting predict()")
        return pred
