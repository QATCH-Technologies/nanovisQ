import os
import pickle
from typing import Union, Optional, Any

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from QATCH.common.logger import Logger as Log
from QATCH.QModel.src.models.pf.pf_data_processor import PFDataProcessor

TAG = ['PartialFill']


class PFPredictor:
    """
    Standalone predictor for PF runs. Handles loading model artifacts,
    preprocessing, and outputting POI count predictions for individual runs.

    Attributes:
        model_dir (str): Directory containing model artifacts.
        scaler (Any): Preprocessing scaler pipeline.
        booster (xgb.Booster): Trained XGBoost model.
    """

    def __init__(self, model_dir: str) -> None:
        """
        Initialize PFPredictor by validating the model directory and loading
        the scaler and booster from disk.
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
        Load the scaler pipeline from a pickle file.
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
        Load the XGBoost Booster from a JSON file.
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

    def _validate_file_buffer(self, file_buffer: str) -> pd.DataFrame:
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
        detected_poi1: Optional[int] = None,
        show_plot: bool = False
    ) -> int:
        """
        Predict the number of POIs for a single PF run.
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

        # Validate detected_poi1
        if detected_poi1 is not None and not isinstance(detected_poi1, (int, np.integer)):
            Log.e(TAG, "Invalid detected_poi1 type")
            raise TypeError("`detected_poi1` must be an integer if provided.")
        Log.d(TAG, f"Using detected_poi1={detected_poi1}")

        # Feature generation
        try:
            Log.d(TAG, "Generating features")
            features = PFDataProcessor.generate_features(
                dataframe=df,
                sampling_rate=1.0,
                detected_poi1=detected_poi1
            )
            Log.i(TAG, f"Features generated: {features.shape[1]} columns")
        except Exception as e:
            Log.e(TAG, f"Feature generation failed: {e}")
            raise ValueError(
                "Failed to generate features from input data.") from e

        # Feature scaling
        try:
            Log.d(TAG, "Scaling features")
            X_scaled = self.scaler.transform(features)
            Log.i(TAG, f"Features scaled: {X_scaled.shape}")
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
