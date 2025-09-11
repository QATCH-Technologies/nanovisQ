

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')
try:
    from QATCH.common.logger import Logger as Log
    from QATCH.QModel.src.models.static_v4_op.v4_op_dataprocessor import OPDataProcessor
except (ModuleNotFoundError, ImportError):
    from v4_op_dataprocessor import OPDataProcessor

    class Log:
        @staticmethod
        def d(tag: str = "", message: str = ""):
            print(f"[DEBUG] {tag}{message}")

        @staticmethod
        def i(tag: str = "", message: str = ""):
            print(f"[INFO] {tag}{message}")

        @staticmethod
        def w(tag: str = "", message: str = ""):
            print(f"[WARNING] {tag}{message}")

        @staticmethod
        def e(tag: str = "", message: str = ""):
            print(f"[ERROR] {tag}{message}")

TAG = "[v4.x Predictor (OP)]"


class QModelPredictorV4OP:

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.models = {}
        self.poi_names = ['POI1', 'POI2', 'POI4', 'POI5', 'POI6']
        self.poi_colors = ['red', 'blue', 'green', 'orange', 'purple']
        self.poi_mapping = [0, 1, 3, 4, 5]
        self._load_all_models()

    def _reset_file_buffer(self, file_buffer: Union[str, object]) -> Union[str, object]:
        """Resets the file buffer to the beginning for reading.

        This method ensures that a file-like object is positioned at the start
        so it can be read from the beginning. If the input is a file path
        (string), it is returned unchanged.

        Args:
            file_buffer (Union[str, object]): A file path or file-like object
                supporting `seek()`.

        Returns:
            Union[str, object]: The same file path or the reset file-like object.

        Raises:
            Exception: If the file-like object does not support seeking.
        """
        """Ensure the file buffer is positioned at its beginning for reading."""
        if isinstance(file_buffer, str):
            return file_buffer
        if hasattr(file_buffer, "seekable") and file_buffer.seekable():
            file_buffer.seek(0)
            return file_buffer
        else:
            raise Exception(
                "Cannot seek stream prior to passing to processing.")

    def _validate_file_buffer(self, file_buffer: Union[str, object]) -> pd.DataFrame:
        """Loads and validates CSV data from a file or file-like object.

        This method reads CSV data into a pandas DataFrame, ensures it contains
        required columns, and checks that it is not empty. It first resets the
        buffer position if a file-like object is provided.

        Args:
            file_buffer (Union[str, object]): Path to a CSV file or a file-like
                object containing CSV data.

        Returns:
            pd.DataFrame: The loaded and validated DataFrame containing the CSV data.

        Raises:
            ValueError: If the file buffer cannot be read, the CSV is empty, or
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
            raise ValueError(f"Error parsing data file: {e}")
        except Exception as e:
            raise ValueError(
                f"An unexpected error occurred while reading the data file: {e}")

        # Validate DataFrame contents
        if df.empty:
            raise ValueError("The data file does not contain any data.")

        required_columns = {"Dissipation",
                            "Resonance_Frequency", "Relative_time"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Data file missing required columns: {', '.join(missing)}.")

        return df

    def _load_all_models(self) -> Dict:

        Log.i(TAG, "Loading XGBoost Ensemble Models")

        for poi_idx in range(5):
            poi_dir = os.path.join(self.models_dir, f'poi_{poi_idx}')

            if not os.path.exists(poi_dir):
                Log.w(
                    TAG, f"Warning: No model found for {self.poi_names[poi_idx]}")
                continue

            try:
                # Load model parameters
                params_path = os.path.join(poi_dir, 'parameters.joblib')
                params = joblib.load(params_path)

                # Load XGBoost model
                model_path = os.path.join(poi_dir, 'model.json')
                xgb_model = xgb.XGBRegressor()
                xgb_model.load_model(model_path)

                # Load scaler
                scaler_path = os.path.join(poi_dir, 'scaler.joblib')
                scaler = joblib.load(scaler_path)

                # Store all components
                self.models[poi_idx] = {
                    'model': xgb_model,
                    'scaler': scaler,
                    'params': params,
                    'window_size': params['window_size'],
                    'stride': params['stride'],
                    'tolerance': params['tolerance'],
                    # 'gaussian_sigma': params['gaussian_sigma'],
                    # 'peak_threshold': params['peak_threshold'],
                    # 'peak_distance': params['peak_distance'],
                    'gaussian_sigma': 1.0,
                    'peak_threshold': 0.1,
                    'peak_distance': 1,
                }

                Log.d(TAG, f"Loaded model for {self.poi_names[poi_idx]}")

            except Exception as e:
                Log.e(TAG, f"Error loading {self.poi_names[poi_idx]}: {e}")

        Log.i(TAG, f"Successfully loaded {len(self.models)}/5 models")
        return self.models

    def predict(self, file_buffer: str, gt: Optional[Union[str, np.ndarray]] = None, format_output: bool = True) -> Dict:
        if file_buffer is not None:
            try:
                df = self._validate_file_buffer(file_buffer=file_buffer)
            except Exception as e:
                Log.d(TAG, f"File buffer could not be validated: {e}")
                raise ValueError(f"File buffer could not be validated: {e}")
        elif df is None:
            Log.e(TAG, "Either file_buffer or dataframe must be provided")
            raise ValueError("Either file_buffer or df must be provided")
        self._df = df
        features_df = OPDataProcessor.gen_features(df)
        features = features_df.values
        n_samples = len(features)

        # Load ground truth if available
        ground_truth = None
        if gt and os.path.exists(gt):
            if type(ground_truth) is str:
                ground_truth = pd.read_csv(
                    gt, header=None).values.flatten()
            elif type(ground_truth) is np.ndarray:
                ground_truth = gt
            else:
                Log.e(TAG, f"Parameter gt must be a file path or np.ndarry")
                raise ValueError(
                    f"Parameter gt must be a file path or np.ndarry")
            Log.i(
                TAG, f"Ground truth POIs provided as {ground_truth}")

        # Store all predictions
        all_predictions = {
            'file': file_buffer,
            'n_samples': n_samples,
            'predictions': {},
            'ground_truth': ground_truth
        }

        # Make predictions for each POI
        detected_positions = []

        for poi_idx, model_data in self.models.items():
            poi_name = self.poi_names[poi_idx]

            # Create windows
            windows = []
            window_positions = []

            window_size = model_data['window_size']
            stride = model_data['stride']

            for i in range(0, n_samples - window_size, stride):
                window = features[i:i + window_size]
                windows.append(window.flatten())
                window_positions.append(i + window_size // 2)

            if len(windows) == 0:
                Log.i(TAG, f"{poi_name}: Insufficient data for prediction")
                continue

            windows = np.array(windows)
            window_positions = np.array(window_positions)

            # Normalize and predict
            X_normalized = model_data['scaler'].transform(windows)
            op_values = model_data['model'].predict(
                X_normalized).astype(np.float32)

            # Apply smoothing
            sigma = model_data['gaussian_sigma']
            if sigma > 0:
                op_smoothed = gaussian_filter1d(op_values, sigma=sigma)
            else:
                op_smoothed = op_values

            # Find peaks
            peaks, properties = find_peaks(
                op_smoothed,
                height=model_data['peak_threshold'],
                distance=model_data['peak_distance']
            )

            # Get event position (highest peak)
            if len(peaks) > 0:
                tallest_idx = np.argmax(properties["peak_heights"])
                tallest_peak = peaks[tallest_idx]
                event_position = tallest_peak * stride + window_size // 2
                detected_positions.append(event_position)
                print(f"  {poi_name}: Detected at position {event_position}")
            else:
                event_position = None
                print(f"  {poi_name}: No event detected")

            # Store predictions
            all_predictions['predictions'][poi_name] = {
                'op_values': op_smoothed,
                'window_positions': window_positions,
                'event_position': event_position,
                'peaks': peaks,
                'peak_heights': properties.get('peak_heights', []) if len(peaks) > 0 else []
            }
        # self.visualize_predictions(all_predictions)
        if format_output:
            return self._format_predictions(all_predictions)
        return all_predictions

    def _format_predictions(self, all_predictions: Dict) -> Dict:

        formatted_output = {}

        for idx, poi_name in enumerate(self.poi_names):
            pred_data = all_predictions['predictions'].get(poi_name, {})

            if pred_data and pred_data.get('event_position') is not None:
                event_pos = pred_data['event_position']

                # Find confidence (closest OP value to event position)
                window_positions = np.array(pred_data['window_positions'])
                op_values = np.array(pred_data['op_values'])
                idx_closest = np.argmin(np.abs(window_positions - event_pos))
                confidence = float(op_values[idx_closest])

                formatted_output[poi_name] = {
                    "indices": [int(event_pos)],
                    "confidences": [confidence]
                }
            else:
                formatted_output[poi_name] = {
                    "indices": [-1],
                    "confidences": [-1]
                }
        formatted_output["POI3"] = {
            "indices": [-1],
            "confidences": [-1]
        }

        return formatted_output

    def visualize_predictions(self, predictions: Dict, save_path: Optional[str] = None):
        # Load the original data for visualization

        # Get sensor data
        if 'Dissipation' in self._df.columns:
            sensor_data = self._df['Dissipation'].values
        else:
            numeric_cols = self._df.select_dtypes(include=[np.number]).columns
            sensor_data = self._df[numeric_cols[0]].values if len(
                numeric_cols) > 0 else np.zeros(len(self._df))

        # Create figure with subplots
        n_pois = len(self.poi_names)
        fig = plt.figure(figsize=(16, 12))

        # Create grid spec for better layout
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(
            n_pois + 2, 1, height_ratios=[1.5] + [1]*n_pois + [1.5])

        # Plot 1: Raw sensor data with all detections
        ax_raw = fig.add_subplot(gs[0])
        ax_raw.plot(sensor_data, 'k-', alpha=0.5,
                    linewidth=0.8, label='Sensor Data')
        ax_raw.set_title('Raw Sensor Data with All Detected Events',
                         fontsize=12, fontweight='bold')
        ax_raw.set_ylabel('Sensor Value')
        ax_raw.grid(True, alpha=0.3)

        # Add all detected events to raw data plot
        for idx, poi_name in enumerate(self.poi_names):
            if poi_name in predictions['predictions']:
                pred_data = predictions['predictions'][poi_name]
                if pred_data['event_position'] is not None:
                    pos = int(pred_data['event_position'])
                    ax_raw.axvline(x=pos, color=self.poi_colors[idx],
                                   alpha=0.6, linestyle='--', linewidth=1.5)

        # Add ground truth if available
        if predictions['ground_truth'] is not None:
            for idx, poi_name in enumerate(self.poi_names):
                actual_idx = self.poi_mapping[idx]
                gt_pos = predictions['ground_truth'][actual_idx]
                ax_raw.axvline(x=gt_pos, color=self.poi_colors[idx],
                               alpha=0.3, linestyle=':', linewidth=2)

        # Individual POI plots
        for idx, poi_name in enumerate(self.poi_names):
            ax = fig.add_subplot(gs[idx + 1])

            if poi_name in predictions['predictions']:
                pred_data = predictions['predictions'][poi_name]

                # Plot OP values
                ax.plot(pred_data['window_positions'], pred_data['op_values'],
                        color=self.poi_colors[idx], alpha=0.7, linewidth=1)
                ax.fill_between(pred_data['window_positions'], 0, pred_data['op_values'],
                                color=self.poi_colors[idx], alpha=0.2)

                # Mark detected event
                if pred_data['event_position'] is not None:
                    ax.axvline(x=pred_data['event_position'],
                               color=self.poi_colors[idx],
                               linestyle='--', alpha=0.8, linewidth=2,
                               label=f'Predicted: {int(pred_data["event_position"])}')

                # Add ground truth
                if predictions['ground_truth'] is not None:
                    actual_idx = self.poi_mapping[idx]
                    gt_pos = predictions['ground_truth'][actual_idx]
                    ax.axvline(x=gt_pos, color='black',
                               linestyle=':', alpha=0.5, linewidth=2,
                               label=f'Ground Truth: {int(gt_pos)}')

                ax.set_ylabel('OP', fontsize=9)
                ax.set_ylim([0, 1.1])
                ax.legend(loc='upper right', fontsize=8)

            ax.set_title(f'{poi_name}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Summary plot
        ax_summary = fig.add_subplot(gs[-1])
        ax_summary.plot(sensor_data, 'gray', alpha=0.3, linewidth=0.8)

        # Plot all predictions and ground truth
        for idx, poi_name in enumerate(self.poi_names):
            if poi_name in predictions['predictions']:
                pred_data = predictions['predictions'][poi_name]
                if pred_data['event_position'] is not None:
                    pos = int(pred_data['event_position'])
                    ax_summary.scatter(pos, sensor_data[min(pos, len(sensor_data)-1)],
                                       color=self.poi_colors[idx], s=100,
                                       marker='o', alpha=0.8, label=f'{poi_name} (Pred)',
                                       edgecolors='white', linewidth=1.5, zorder=5)

            # Add ground truth markers
            if predictions['ground_truth'] is not None:
                actual_idx = self.poi_mapping[idx]
                gt_pos = int(predictions['ground_truth'][actual_idx])
                ax_summary.scatter(gt_pos, sensor_data[min(gt_pos, len(sensor_data)-1)],
                                   color=self.poi_colors[idx], s=100,
                                   marker='x', alpha=0.8, linewidth=3,
                                   label=f'{poi_name} (GT)' if idx == 0 else '', zorder=4)

        ax_summary.set_title('Summary: Predicted vs Ground Truth Positions',
                             fontsize=12, fontweight='bold')
        ax_summary.set_xlabel('Sample Index')
        ax_summary.set_ylabel('Sensor Value')
        ax_summary.legend(loc='upper right', ncol=2, fontsize=8)
        ax_summary.grid(True, alpha=0.3)

        plt.suptitle(f'POI Detection',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to {save_path}")

        plt.show()
