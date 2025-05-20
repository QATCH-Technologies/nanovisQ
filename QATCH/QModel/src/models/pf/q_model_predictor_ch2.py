import os
import pickle
from typing import Union
import pandas as pd
import numpy as np
import xgboost as xgb
from QATCH.QModel.src.models.static_v3.q_model_data_processor import QDataProcessor
from QATCH.common.logger import Logger as Log

TAG = '[QModel_Ch2]'


class QModelPredictorCh2:
    def __init__(self, model_dir: str):
        if not os.path.exists(model_dir):
            Log.e(TAG, f'Directory path `{model_dir}` does nto exist.')
            raise FileNotFoundError(
                f'Directory path `{model_dir}` does nto exist.')

        self.model_dir = model_dir
        self.scaler_path = os.path.join(
            self.model_dir, "qmodel_scaler_ch2.pkl")
        self.booster_path = os.path.join(self.model_dir, "qmodel_ch2.json")

        self._load_scaler()
        self._load_booster()

    def _load_scaler(self):
        if not os.path.exists(self.scaler_path):
            Log.e(TAG, f"Scaler not found at `{self.scaler_path}`")
            raise FileNotFoundError(
                f"Scaler not found at `{self.scaler_path}`")
        with open(self.scaler_path, "rb") as f:
            self._scaler = pickle.load(f)

    def _load_booster(self):
        if not os.path.exists(self.booster_path):
            Log.e(TAG, f"Booster not found at `{self.booster_path}`")
            raise FileNotFoundError(
                f"Booster not found at `{self.booster_path}`")
        self._booster = xgb.Booster()
        self._booster.load_model(self.booster_path)

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
        """Load and validate a CSV data file into a pandas DataFrame.

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

    def predict(self, file_buffer: Union[pd.DataFrame, str, np.ndarray]) -> np.ndarray:
        try:
            df = self._validate_file_buffer(file_buffer=file_buffer)
        except Exception as e:
            Log.e(
                f"File buffer `{file_buffer}` could not be validated because of error: `{e}`.")
            return
        # `file_buffer` passed to `process_data` must be seekable, but `df` is not seekable
        file_buffer = self._reset_file_buffer(file_buffer)
        X = QDataProcessor.process_data(file_buffer)
        X_scaled = self._scaler.transform(X)
        dmat = xgb.DMatrix(X_scaled)
        preds = self._booster.predict(dmat)
        if preds.ndim == 2:
            return np.argmax(preds, axis=1)
        else:
            return (preds > 0.5).astype(int)
