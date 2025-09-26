"""
qmodel_v4_predictor.py

Provides the QModelPredictor class for version 4.x of QModel based on a CNN, it
supports partially filled runs as well as fully filled runs with high precision.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-09-09

Version:
    QModel.Ver4.2
"""
import matplotlib.pyplot as plt
from itertools import chain
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from typing import Dict, List, Tuple, Optional, Union
import warnings
import os
import json
from datetime import datetime
from QATCH.QModel.src.models.static_v4.qmodel_v4_post import V4PostProcess
warnings.filterwarnings('ignore')
TAG = "[QModelv4.x]"
# Try to use legacy Keras if available
try:
    import tf_keras
    keras = tf_keras
    print("Using tf_keras for compatibility")
except ImportError:
    from tensorflow import keras
    print("Using tensorflow.keras")

try:
    from QATCH.QModel.src.models.static_v4.qmodel_v4_dataprocessor import DataProcessorV4
    from QATCH.common.logger import Logger as Log
except (ImportError, ModuleNotFoundError):
    from QModel.src.models.static_v4.qmodel_v4_dataprocessor import DataProcessorV4

    class Log:
        @staticmethod
        def d(message):
            print(f"[DEBUG] {message}")

        @staticmethod
        def i(message):
            print(f"[INFO] {message}")

        @staticmethod
        def w(message):
            print(f"[WARNING] {message}")

        @staticmethod
        def e(message):
            print(f"[ERROR] {message}")


def plot_predictions_with_confidences(df, final_poi):
    """
    Plot all candidate predictions with different colors based on confidence levels.

    Args:
        df: DataFrame with the data (must contain 'Relative_time' and 'Dissipation')
        final_poi: Dictionary of POI predictions with indices and confidences
    """
    plt.figure(figsize=(12, 8))

    # Plot the main dissipation signal with time on x-axis
    plt.plot(df["Relative_time"].values, df["Dissipation"].values,
             color='grey', linewidth=1.5, label="Dissipation", alpha=0.7)

    # Colormap for confidence levels
    colors = plt.cm.RdYlBu_r  # Red=high, blue=low

    # Collect predictions and confidences
    all_confidences = []
    all_predictions = []

    for poi_num, poi_data in final_poi.items():
        indices = poi_data.get("indices", [])
        confidences = poi_data.get("confidences", [])

        for idx, conf in zip(indices, confidences):
            if 0 <= idx < len(df):
                all_predictions.append((poi_num, idx, conf))
                all_confidences.append(conf)

    if all_confidences:
        # Normalize confidences
        min_conf = min(all_confidences)
        max_conf = max(all_confidences)
        conf_range = max_conf - min_conf if max_conf > min_conf else 1

        # Plot each prediction
        for poi_num, idx, conf in all_predictions:
            # use time instead of index
            time_val = df["Relative_time"].iloc[idx]
            norm_conf = (conf - min_conf) / conf_range
            color = colors(norm_conf)

            # Vertical line at the time point
            plt.axvline(time_val, color=color, linewidth=2.5, alpha=0.8)

            # Annotation
            plt.text(time_val, plt.ylim()[1] * 0.9,
                     f'POI{poi_num}\n{conf:.3f}',
                     rotation=0, ha='center', va='top', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor=color, alpha=0.7))

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=colors,
                                   norm=plt.Normalize(vmin=min(all_confidences),
                                                      vmax=max(all_confidences)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8, aspect=20)
        cbar.set_label('Confidence Level', rotation=270, labelpad=20)

    plt.xlabel('Relative Time')
    plt.ylabel('Dissipation')
    plt.title('Prediction Results with Confidence Levels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


class QModelPredictorV4:
    """Predicts Points of Interest (POIs) from time-series sensor data.

    This class wraps a pre-trained deep learning model and provides methods
    to load and preprocess data, generate features, predict POIs, and
    apply sequential and temporal constraints. It also tracks predictions
    for evaluation and ensures they follow domain-specific rules.

    Features:
        - Load pre-trained model and scaler with multiple fallback strategies.
        - Preprocess input data and generate sliding window features.
        - Predict the best POI locations with confidences scores.
        - Apply sequential and temporal constraints to predictions.
        - Validate that predictions follow all POI dependencies.
        - Non-maximum suppression to reduce duplicate detections.
        - Filtering strategies for specific POIs (e.g., POI1).
        - Prediction tracking for metrics and analysis.

    Attributes:
        model_path (str): Path to the pre-trained model file (.h5).
        scaler_path (str): Path to the scaler file (.pkl) for feature normalization.
        window_size (int): Size of the sliding window for feature extraction.
        stride (int): Step size for sliding window.
        tolerance (int): Acceptable distance (in points) for POI detection.
        poi_names (Dict[int, str]): Mapping of POI numbers to human-readable names.
        poi_indices (Dict[int, int]): Mapping from POI number to model output index.
        default_thresholds (Dict[int, float]): Adaptive thresholds for POI detection.
        prediction_history (List[Dict]): Stores tracked predictions for analysis.
        model (keras.Model): Loaded deep learning model for POI prediction.
        scaler (sklearn.preprocessing.StandardScaler): Loaded scaler for feature normalization.

    Methods:
        __init__(...): Initializes the predictor with model, scaler, and optional config.
        _load_config(config_path): Load configuration from JSON file.
        _load_model(): Load the model with multiple fallback strategies.
        _load_scaler(): Load the scaler from file.
        _reset_file_buffer(file_buffer): Reset file buffer to beginning if possible.
        _validate_file_buffer(file_buffer): Validate and read CSV data into DataFrame.
        generate_features(df): Generate features from raw data.
        preprocess_data(df): Preprocess data and create sliding window arrays.
        predict_best(file_buffer=None, df=None, ...): Predict the single best POI locations.
        _get_default_predictions(): Return default placeholder predictions.
        _format_final_predictions(poi_predictions, force): Format predictions for output.
        _non_maximum_suppression(probs, mask, window): Apply non-maximum suppression.
        _apply_constraints_to_topk(predictions, df): Apply sequential/temporal constraints.
        apply_constraints(predictions, df, enforce_sequential=True, enforce_gaps=True): Apply constraints.
        _enforce_gap_constraints(predictions, df): Enforce relative time gaps.
        _track_prediction(prediction): Track prediction for metrics and analysis.
        validate_predictions(predictions): Validate that predictions follow all constraints.

    Example:
        predictor = POIPredictor(
            model_path="model.h5", scaler_path="scaler.pkl")
        predictions = predictor.predict_best(file_buffer="data.csv")
        is_valid = predictor.validate_predictions(predictions)
    """

    def __init__(self,
                 model_path: str,
                 scaler_path: str,
                 window_size: int = 128,
                 stride: int = 16,
                 tolerance: int = 64,
                 config_path: Optional[str] = None):
        """Initializes the V4 Predictor with a pre-trained model and scaler.

        This constructor loads the specified model and scaler, sets up sliding
        window parameters, and initializes POI (Point of Interest) configurations.
        If a configuration file is provided, additional parameters may be loaded.

        Args:
            model_path (str): Path to the pre-trained Keras model (.h5 file).
            scaler_path (str): Path to the saved scaler (.pkl file).
            window_size (int, optional): Size of the sliding window used for
                feature extraction. Defaults to 128.
            stride (int, optional): Step size for the sliding window movement.
                Defaults to 16.
            tolerance (int, optional): Acceptable distance (in points) for POI
                detection. Defaults to 64.
            config_path (Optional[str], optional): Path to an optional
                configuration file. If provided and exists, it will be loaded.

        Attributes:
            window_size (int): Sliding window size for feature extraction.
            stride (int): Step size for sliding window.
            tolerance (int): Tolerance for POI detection.
            model_path (str): Path to the model file.
            scaler_path (str): Path to the scaler file.
            poi_names (dict): Mapping from label indices to POI names.
            poi_indices (dict): Mapping from POI numbers to prediction indices.
            default_thresholds (dict): Default adaptive thresholds for POI
                detection.
            prediction_history (list): History of predictions made by the model.
        """
        self.window_size = window_size
        self.stride = stride
        self.tolerance = tolerance
        self.model_path = model_path
        self.scaler_path = scaler_path

        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

        # Load the model and scaler
        self._load_model()
        self._load_scaler()

        # POI configuration
        self.poi_names = {
            0: 'Non-POI',
            1: 'POI-1',
            2: 'POI-2',
            3: 'POI-3',  # Always returns -1
            4: 'POI-4',
            5: 'POI-5',
            6: 'POI-6'
        }

        # Mapping from POI number to prediction index
        self.poi_indices = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5}

        # Default adaptive thresholds
        self.default_thresholds = {
            1: 0.5,
            2: 0.5,
            4: 0.7,
            5: 0.6,
            6: 0.65
        }

        # Initialize metrics tracking
        self.prediction_history = []

        Log.d("Predictor initialized successfully!")

    def _load_config(self, config_path: str):
        """Loads predictor configuration parameters from a JSON file.

        The configuration file can override default values for window size,
        stride, tolerance, and adaptive thresholds.

        Args:
            config_path (str): Path to the JSON configuration file.

        Updates:
            window_size (int): Sliding window size if specified in config.
            stride (int): Step size for sliding window if specified in config.
            tolerance (int): POI detection tolerance if specified in config.
            default_thresholds (dict): Updated thresholds if provided in config.

        Raises:
            Exception: Logs a warning if the configuration file cannot be loaded.
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            self.window_size = config.get('window_size', self.window_size)
            self.stride = config.get('stride', self.stride)
            self.tolerance = config.get('tolerance', self.tolerance)

            if 'thresholds' in config:
                self.default_thresholds.update(config['thresholds'])

            Log.d(f"Configuration loaded from {config_path}")
        except Exception as e:
            Log.w(f"Failed to load config: {e}")

    def _load_model(self):
        """Loads the .h5 machine learning model with comprehensive fallback strategies.

        Handles compatibility issues between TensorFlow 2.10 (training) and 2.18 (deployment)
        for .h5 format models. Attempts multiple loading strategies optimized for exe compilation.

        The method attempts to load the model in the following order:
            1. Standard Keras .h5 loading with compatibility settings
            2. Legacy tf_keras compatibility mode  
            3. Custom object scope with common layer registrations
            4. Architecture + weights loading (separate files)
            5. HDF5 direct loading with manual reconstruction

        Attributes:
            model (keras.Model): The loaded Keras model.

        Raises:
            Exception: If the model cannot be loaded with any of the strategies.
        """
        import tensorflow as tf
        import os
        import warnings
        import h5py

        # Suppress compatibility warnings that don't affect functionality
        warnings.filterwarnings(
            'ignore', category=UserWarning, module='tensorflow')
        warnings.filterwarnings(
            'ignore', category=FutureWarning, module='tensorflow')

        Log.d(TAG, f"Loading .h5 model from {self.model_path}...")
        Log.d(TAG, f"TensorFlow version: {tf.__version__}")

        # Verify the file exists and is accessible
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if not self.model_path.endswith('.h5'):
            raise ValueError(f"Expected .h5 file, got: {self.model_path}")

        model_loaded = False
        load_errors = []

        # Strategy 1: Standard Keras .h5 loading with compatibility settings
        if not model_loaded:
            try:
                Log.d(TAG, "Attempting standard Keras .h5 model loading...")

                # Set compatibility options for TensorFlow 2.18
                tf.compat.v1.disable_eager_execution() if hasattr(
                    tf.compat.v1, 'disable_eager_execution') else None

                self.model = tf.keras.models.load_model(
                    self.model_path,
                    compile=False,
                    custom_objects=self._get_h5_custom_objects()
                )
                Log.d(TAG, "Model loaded successfully with standard Keras")
                model_loaded = True

            except Exception as e:
                error_msg = f"Standard Keras .h5 loading failed: {str(e)[:200]}..."
                Log.d(TAG, error_msg)
                load_errors.append(error_msg)

        # Strategy 2: Try tf_keras for legacy .h5 support
        if not model_loaded:
            try:
                import tf_keras
                Log.d(TAG, "Attempting to load .h5 with tf_keras...")

                self.model = tf_keras.models.load_model(
                    self.model_path,
                    compile=False,
                    custom_objects=self._get_h5_custom_objects()
                )
                Log.d(TAG, "Model loaded successfully with tf_keras")
                model_loaded = True

            except ImportError:
                error_msg = "tf_keras not available for .h5 loading"
                Log.d(TAG, error_msg)
                load_errors.append(error_msg)
            except Exception as e:
                error_msg = f"tf_keras .h5 loading failed: {str(e)[:200]}..."
                Log.d(TAG, error_msg)
                load_errors.append(error_msg)

        # Strategy 3: Custom object scope with H5-specific handling
        if not model_loaded:
            try:
                Log.d(TAG, "Attempting .h5 loading with enhanced custom object scope...")

                custom_objects = self._get_comprehensive_custom_objects()

                with tf.keras.utils.custom_object_scope(custom_objects):
                    self.model = tf.keras.models.load_model(
                        self.model_path,
                        compile=False
                    )
                Log.d(TAG, "Model loaded successfully with enhanced custom object scope")
                model_loaded = True

            except Exception as e:
                error_msg = f"Enhanced custom object scope .h5 loading failed: {str(e)[:200]}..."
                Log.d(TAG, error_msg)
                load_errors.append(error_msg)

        # Strategy 4: Try to extract and load architecture + weights separately
        if not model_loaded:
            try:
                Log.d(TAG, "Attempting architecture + weights separation loading...")

                # Read the .h5 file structure
                with h5py.File(self.model_path, 'r') as h5file:
                    # Check if we can extract model config
                    if 'model_config' in h5file.attrs:
                        import json
                        model_config = json.loads(
                            h5file.attrs['model_config'].decode('utf-8'))

                        # Reconstruct model from config
                        self.model = tf.keras.models.model_from_json(
                            json.dumps(model_config),
                            custom_objects=self._get_comprehensive_custom_objects()
                        )

                        # Load weights
                        self.model.load_weights(self.model_path)
                        Log.d(
                            TAG, "Model loaded successfully with architecture + weights separation")
                        model_loaded = True

            except Exception as e:
                error_msg = f"Architecture + weights separation failed: {str(e)[:200]}..."
                Log.d(TAG, error_msg)
                load_errors.append(error_msg)

        # Strategy 5: Manual HDF5 reconstruction (last resort)
        if not model_loaded:
            try:
                Log.d(TAG, "Attempting manual HDF5 reconstruction...")

                model_info = self._analyze_h5_structure()
                if model_info:
                    self.model = self._reconstruct_from_h5_analysis(model_info)
                    if self.model:
                        Log.d(
                            TAG, "Model loaded successfully with manual HDF5 reconstruction")
                        model_loaded = True

            except Exception as e:
                error_msg = f"Manual HDF5 reconstruction failed: {str(e)[:200]}..."
                Log.d(TAG, error_msg)
                load_errors.append(error_msg)

        if not model_loaded:
            error_summary = "\n".join(
                [f"  - {error}" for error in load_errors])
            raise Exception(
                f"Failed to load .h5 model with all strategies:\n{error_summary}")

        # Compile the model with exe-friendly settings
        try:
            if hasattr(self.model, 'compile') and callable(getattr(self.model, 'compile')):
                Log.d(TAG, "Compiling model...")
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=0.001),  # Explicit optimizer
                    loss='binary_crossentropy',
                    metrics=['accuracy'],
                    run_eagerly=False,  # Important for exe compatibility
                    steps_per_execution=1  # Helps with TF 2.18 compatibility
                )
                Log.d(TAG, "Model compiled successfully")
            else:
                Log.d(
                    TAG, "Model compilation skipped (not supported by this model type)")

        except Exception as e:
            Log.d(TAG, f"Warning: Could not compile model: {e}")
            Log.d(TAG, "Model will work for inference without compilation")

    def _get_h5_custom_objects(self):
        """Returns custom objects specifically for .h5 model compatibility."""
        import tensorflow as tf

        custom_objects = {}

        # Common activation functions that might have changed between versions
        activations = ['swish', 'mish', 'gelu']
        for activation in activations:
            try:
                if hasattr(tf.nn, activation):
                    custom_objects[activation] = getattr(tf.nn, activation)
                elif hasattr(tf.keras.activations, activation):
                    custom_objects[activation] = getattr(
                        tf.keras.activations, activation)
            except (AttributeError, ImportError):
                continue

        # Add legacy layer mappings
        try:
            custom_objects['BatchNormalization'] = tf.keras.layers.BatchNormalization
            custom_objects['LayerNormalization'] = tf.keras.layers.LayerNormalization
        except AttributeError:
            pass

        return custom_objects

    def _get_comprehensive_custom_objects(self):
        """Returns comprehensive custom objects for problematic .h5 models."""
        import tensorflow as tf

        custom_objects = self._get_h5_custom_objects()

        # Add common optimizers
        try:
            custom_objects['Adam'] = tf.keras.optimizers.Adam
            custom_objects['RMSprop'] = tf.keras.optimizers.RMSprop
            custom_objects['SGD'] = tf.keras.optimizers.SGD
        except AttributeError:
            try:
                # TF 2.18 might have different optimizer paths
                custom_objects['Adam'] = tf.keras.optimizers.legacy.Adam
            except AttributeError:
                pass

        # Add common loss functions
        try:
            custom_objects['binary_crossentropy'] = tf.keras.losses.binary_crossentropy
            custom_objects['categorical_crossentropy'] = tf.keras.losses.categorical_crossentropy
        except AttributeError:
            pass

        # Add common metrics
        try:
            custom_objects['accuracy'] = tf.keras.metrics.accuracy
            custom_objects['Accuracy'] = tf.keras.metrics.Accuracy
        except AttributeError:
            pass

        # Handle potential custom layers or functions you might have used
        custom_objects.update(self._get_project_specific_objects())

        return custom_objects

    def _get_project_specific_objects(self):
        """Add any custom objects specific to your model here."""
        # Replace this with any custom layers, functions, or objects your model uses
        return {}

    def _analyze_h5_structure(self):
        """Analyzes the .h5 file structure to understand the model."""
        try:
            import h5py
            import json

            with h5py.File(self.model_path, 'r') as h5file:
                model_info = {}

                # Extract model configuration if available
                if 'model_config' in h5file.attrs:
                    model_info['config'] = json.loads(
                        h5file.attrs['model_config'].decode('utf-8'))

                # Extract keras version info
                if 'keras_version' in h5file.attrs:
                    model_info['keras_version'] = h5file.attrs['keras_version'].decode(
                        'utf-8')

                # Extract layer information
                model_info['layer_names'] = []
                if 'model_weights' in h5file:
                    model_weights = h5file['model_weights']
                    if hasattr(model_weights, 'keys'):
                        model_info['layer_names'] = list(model_weights.keys())

                return model_info

        except Exception as e:
            Log.d(TAG, f"H5 structure analysis failed: {e}")
            return None

    def _reconstruct_from_h5_analysis(self, model_info):
        """Attempts to reconstruct model from H5 analysis (implement based on your model)."""
        # This is a placeholder - you would implement this based on your specific model architecture
        # For now, return None to indicate this strategy failed
        Log.d(TAG, "Manual reconstruction not implemented - would need specific model architecture")
        return None

    def _validate_model_loading(self):
        """Validates that the loaded model is functional."""
        if not hasattr(self, 'model') or self.model is None:
            return False

        try:
            # Try to get model summary (basic functionality check)
            if hasattr(self.model, 'summary'):
                # Capture summary to verify model structure
                import io
                import sys

                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()
                try:
                    self.model.summary()
                    summary = buffer.getvalue()
                    Log.d(TAG, f"Model validation successful - layers detected")
                    return True
                finally:
                    sys.stdout = old_stdout

            return True

        except Exception as e:
            Log.d(TAG, f"Model validation failed: {e}")
            return False

    def _load_scaler(self):
        """Loads the feature scaler from a file.

        This method loads a pre-fitted scaler (e.g., `StandardScaler`) using
        `joblib` from the specified path. The scaler is used to normalize or
        standardize features before making predictions.

        Attributes:
            scaler: The loaded scaler object (e.g., `StandardScaler`).

        Raises:
            Exception: If the scaler cannot be loaded from the specified path.
        """
        Log.d(f"Loading scaler from {self.scaler_path}...")
        try:
            self.scaler = joblib.load(self.scaler_path)
            Log.d("Scaler loaded successfully")
        except Exception as e:
            Log.e(f"Failed to load scaler: {e}")
            raise

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

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates features from raw input data.

        This method uses the `DataProcessorV4.gen_features` function to
        transform raw input data into feature representations suitable
        for the POI model.

        Args:
            df (pd.DataFrame): Raw input data containing columns such as
                'Dissipation', 'Resonance_Frequency', and 'Relative_time'.

        Returns:
            pd.DataFrame: A DataFrame containing the generated features ready
                for prediction.
        """
        return DataProcessorV4.gen_features(df)

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[int], List[float]]:
        """Preprocess data for prediction."""
        # Generate features
        features_df = self.generate_features(df)
        features = features_df.values
        self._features_df = features_df

        # Create sliding windows
        windows = []
        window_positions = []
        window_times = []

        for i in range(0, len(features) - self.window_size, self.stride):
            window = features[i:i + self.window_size]
            windows.append(window)

            # Store center position of window
            center_pos = i + self.window_size // 2
            window_positions.append(center_pos)

            # Store time if available
            if 'Relative_time' in df.columns and center_pos < len(df):
                window_times.append(df['Relative_time'].iloc[center_pos])
            else:
                window_times.append(center_pos)

        windows = np.array(windows)

        # Normalize windows
        original_shape = windows.shape
        windows_flat = windows.reshape(-1, windows.shape[-1])
        windows_normalized = self.scaler.transform(windows_flat)
        windows_normalized = windows_normalized.reshape(original_shape)

        return windows_normalized, window_positions, window_times

    def predict(self,
                file_buffer: Union[str, object] = None,
                df: pd.DataFrame = None,
                top_k: int = 3,
                min_confidence: float = 0.25,
                force: bool = False,
                apply_constraints: bool = True) -> Dict[str, Dict[str, List]]:
        """Preprocesses raw data into normalized sliding windows for prediction.

        This method generates features from the raw data, constructs sliding
        windows of a specified size and stride, and normalizes the windows using
        the loaded scaler. It also tracks the center positions and times of each
        window for later reference.

        Args:
            df (pd.DataFrame): Raw input data containing columns such as
                'Dissipation', 'Resonance_Frequency', and 'Relative_time'.

        Returns:
            Tuple[np.ndarray, List[int], List[float]]:
                windows_normalized: A NumPy array of normalized feature windows.
                window_positions: List of center positions (indices) for each window.
                window_times: List of corresponding times for each window. If the
                    'Relative_time' column exists, uses its values; otherwise uses
                    center positions.
        """
        # Handle input data
        if file_buffer is not None:
            try:
                df = self._validate_file_buffer(file_buffer=file_buffer)
            except Exception as e:
                Log.d(f"File buffer could not be validated: {e}")
                return self._get_default_predictions()
        elif df is None:
            raise ValueError("Either file_buffer or df must be provided")

        # Preprocess data
        windows, positions, times = self.preprocess_data(df)

        if len(windows) == 0:
            Log.w("No windows generated from data")
            return self._get_default_predictions()

        # Get model predictions
        predictions = self.model.predict(windows, verbose=0)

        # Collect predictions for each POI
        poi_predictions = {}

        for poi_num, pred_idx in self.poi_indices.items():
            poi_probs = predictions[:, pred_idx]

            # Find indices above minimum confidences
            if force:
                sorted_indices = np.argsort(poi_probs)[::-1][:top_k]
                valid_indices = sorted_indices
            else:
                valid_indices = np.where(poi_probs >= min_confidence)[0]
                if len(valid_indices) > 0:
                    sorted_indices = valid_indices[np.argsort(
                        poi_probs[valid_indices])[::-1]]
                    valid_indices = sorted_indices[:top_k]

            if len(valid_indices) > 0:
                indices_list = []
                confidences_list = []

                for idx in valid_indices:
                    indices_list.append(positions[idx])
                    confidences_list.append(float(poi_probs[idx]))

                poi_predictions[poi_num] = {
                    'indices': indices_list,
                    'confidences': confidences_list,
                    'times': [times[idx] for idx in valid_indices] if times else None
                }

        # Apply constraints if requested
        if apply_constraints:
            poi_predictions = self._apply_constraints_to_topk(
                poi_predictions, df)

        # Convert to final format
        final_poi = self._format_final_predictions(poi_predictions, force)

        # final_poi = V4PostProcess.reasign(
        #     final_poi, df["Relative_time"].values)

        # final_poi["POI1"]["indices"][0] = V4PostProcess.poi_1(
        #     final_poi["POI1"]["indices"], final_poi["POI1"]["confidences"], self._features_df, relative_time=df["Relative_time"].values)
        # if final_poi["POI4"].get("indices")[0] > -1:
        #     final_poi["POI5"]["indices"][0] = V4PostProcess.density_select(
        #         poi_indices=final_poi["POI5"]["indices"], relative_time=df["Relative_time"].values, confidences=final_poi["POI5"]["confidences"])
        # if final_poi["POI5"].get("indices")[0] > -1:
        #     final_poi["POI6"]["indices"][0] = V4PostProcess.density_select(
        #         poi_indices=final_poi["POI6"]["indices"], relative_time=df["Relative_time"].values, confidences=final_poi["POI6"]["confidences"])
        # Track prediction
        self._track_prediction(final_poi)
        # plot_predictions_with_confidences(df, final_poi)
        return final_poi

    def _get_default_predictions(self) -> Dict[str, Dict[str, List]]:
        """Returns a default POI prediction dictionary with placeholder values.

        All POIs are set to `-1` for both indices and confidences scores,
        indicating that no prediction was made.

        Returns:
            Dict[str, Dict[str, List]]: Dictionary of default POI predictions with
            'indices' and 'confidences' set to [-1] for each POI.
        """
        return {
            "POI1": {"indices": [-1], "confidences": [-1]},
            "POI2": {"indices": [-1], "confidences": [-1]},
            "POI3": {"indices": [-1], "confidences": [-1]},
            "POI4": {"indices": [-1], "confidences": [-1]},
            "POI5": {"indices": [-1], "confidences": [-1]},
            "POI6": {"indices": [-1], "confidences": [-1]}
        }

    def _format_final_predictions(self, poi_predictions: Dict, force: bool) -> Dict[str, Dict[str, List]]:
        """Formats POI predictions into a consistent output structure.

        Ensures that all six POIs are included in the output. POI3 is always
        returned with placeholder values `-1`. For other POIs, if no valid
        prediction is available, `-1` values are used. Also flattens lists
        and removes duplicate indices while keeping first occurrence.
        """
        final_poi = {}

        def _flatten_and_dedupe(indices, confidences):
            # flatten
            flat_indices = list(chain.from_iterable(indices)) if any(
                isinstance(i, list) for i in indices) else indices
            flat_confidences = list(chain.from_iterable(confidences)) if any(
                isinstance(c, list) for c in confidences) else confidences

            # dedupe while keeping first occurrence
            seen = set()
            dedup_indices, dedup_confidences = [], []
            for idx, conf in zip(flat_indices, flat_confidences):
                if idx not in seen:
                    seen.add(idx)
                    dedup_indices.append(idx)
                    dedup_confidences.append(conf)
            return dedup_indices, dedup_confidences

        # Always include POI3 with -1 values
        final_poi["POI3"] = {"indices": [-1], "confidences": [-1]}

        # Process other POIs
        for poi_num in [1, 2, 4, 5, 6]:
            if poi_num in poi_predictions:
                pred = poi_predictions[poi_num]
                if pred['indices'] and pred['confidences']:
                    indices, confidences = _flatten_and_dedupe(
                        pred['indices'], pred['confidences'])
                    if not indices:  # fallback if all got removed
                        indices, confidences = [-1], [-1]
                    final_poi[f"POI{poi_num}"] = {
                        "indices": indices,
                        "confidences": confidences
                    }
                else:
                    final_poi[f"POI{poi_num}"] = {
                        "indices": [-1], "confidences": [-1]}
            else:
                final_poi[f"POI{poi_num}"] = {
                    "indices": [-1], "confidences": [-1]}

        return final_poi

    def _apply_constraints_to_topk(self, predictions: Dict[int, Dict],
                                   df: pd.DataFrame) -> Dict[int, Dict]:
        """Applies sequential and temporal constraints to top-k POI predictions.

        Certain POIs depend on the presence of previous POIs. This method enforces
        these constraints:
            - POI1 and POI2 can exist independently.
            - POI4 requires both POI1 and POI2 to exist and occur before it.
            - POI5 requires POI4 to exist and occur before it.
            - POI6 requires POI5 to exist and occur before it.

        Args:
            predictions (Dict[int, Dict]): Dictionary of predicted top-k POIs. Keys
                are POI numbers, and values contain 'indices', 'confidences', and
                optionally 'times'.
            df (pd.DataFrame): Original input data (used for temporal references if needed).

        Returns:
            Dict[int, Dict]: Filtered predictions dictionary with constraints applied.
                Only POIs that satisfy sequential and temporal dependencies are retained.
        """
        validated = {}

        # POI1 and POI2 can exist independently
        if 1 in predictions:
            validated[1] = predictions[1]
        if 2 in predictions:
            validated[2] = predictions[2]

        # POI4 requires both POI1 and POI2
        if 4 in predictions:
            if 1 in validated and 2 in validated:
                if validated[2]['indices']:
                    max_poi2_idx = max(validated[2]['indices'])
                    valid_poi4_indices = []
                    valid_poi4_confidences = []

                    for i, idx in enumerate(predictions[4]['indices']):
                        if idx > max_poi2_idx:
                            valid_poi4_indices.append(idx)
                            valid_poi4_confidences.append(
                                predictions[4]['confidences'][i])

                    if valid_poi4_indices:
                        validated[4] = {
                            'indices': valid_poi4_indices,
                            'confidences': valid_poi4_confidences,
                            'times': predictions[4].get('times', [])
                        }

        # POI5 requires POI4
        if 5 in predictions:
            if 4 in validated and validated[4]['indices']:
                max_poi4_idx = max(validated[4]['indices'])
                valid_poi5_indices = []
                valid_poi5_confidences = []

                for i, idx in enumerate(predictions[5]['indices']):
                    if idx > max_poi4_idx:
                        valid_poi5_indices.append(idx)
                        valid_poi5_confidences.append(
                            predictions[5]['confidences'][i])

                if valid_poi5_indices:
                    validated[5] = {
                        'indices': valid_poi5_indices,
                        'confidences': valid_poi5_confidences,
                        'times': predictions[5].get('times', [])
                    }

        # POI6 requires POI5
        if 6 in predictions:
            if 5 in validated and validated[5]['indices']:
                max_poi5_idx = max(validated[5]['indices'])
                valid_poi6_indices = []
                valid_poi6_confidences = []

                for i, idx in enumerate(predictions[6]['indices']):
                    if idx > max_poi5_idx:
                        valid_poi6_indices.append(idx)
                        valid_poi6_confidences.append(
                            predictions[6]['confidences'][i])

                if valid_poi6_indices:
                    validated[6] = {
                        'indices': valid_poi6_indices,
                        'confidences': valid_poi6_confidences,
                        'times': predictions[6].get('times', [])
                    }

        return validated

    def _track_prediction(self, prediction: Dict):
        """Tracks a prediction for logging, metrics, or future analysis.

        Each prediction is timestamped and appended to the internal
        `prediction_history` list to enable monitoring of model outputs
        over time.

        Args:
            prediction (Dict): Dictionary of POI predictions, typically in the
                format returned by `predict_best` or `_format_final_predictions`.
        """
        self.prediction_history.append({
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction
        })

    def validate_predictions(self, predictions: Dict[str, Dict[str, List]]) -> bool:
        """Validates that predicted POIs follow all defined constraints.

        This method checks:
            - POI3 always has placeholder values `[-1]`.
            - Sequential dependencies between POIs are satisfied:
                * POI4 must occur after POI1 and POI2.
                * POI5 must occur after POI4.
                * POI6 must occur after POI5.

        Args:
            predictions (Dict[str, Dict[str, List]]): Dictionary of final POI predictions.
                Each key is a POI label (e.g., 'POI1') and each value contains:
                    - 'indices': List[int] of predicted indices.
                    - 'confidences': List[float] of prediction confidences.

        Returns:
            bool: True if predictions satisfy all constraints, False otherwise.
        """
        # Check POI3 is always -1
        if "POI3" in predictions:
            if predictions["POI3"]["indices"] != [-1] or predictions["POI3"]["confidences"] != [-1]:
                return False

        # Check sequential constraints
        poi_indices = {}
        for poi_name, data in predictions.items():
            if poi_name != "POI3" and data["indices"][0] != -1:
                poi_num = int(poi_name.replace("POI", ""))
                poi_indices[poi_num] = data["indices"][0]

        # Validate dependencies
        if 4 in poi_indices:
            if 1 not in poi_indices or 2 not in poi_indices:
                return False
            if poi_indices[4] <= poi_indices[2]:
                return False

        if 5 in poi_indices:
            if 4 not in poi_indices:
                return False
            if poi_indices[5] <= poi_indices[4]:
                return False

        if 6 in poi_indices:
            if 5 not in poi_indices:
                return False
            if poi_indices[6] <= poi_indices[5]:
                return False

        return True
