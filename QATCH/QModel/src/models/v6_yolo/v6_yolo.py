# module: v6_yolo.py

"""This module implements the QModel V6 YOLO pipeline for data analysis.

It orchestrates a "Reverse Cascading Detection" strategy, utilizing multiple YOLO object detectors to
identify points of interest (POIs) in viscosity data. The pipeline handles data preprocessing,
fill-type classification, and sequential slicing of the dataset to isolate specific channel events (Init, Ch1, Ch2, Ch3).

Key Components:
- QModelV6YOLO: The main controller class.
- QModelV6YOLO_Detector: A wrapper for individual YOLO model instances.
- QModelV6Config: Configuration constants for the pipeline.

Dependencies:
- ultralytics (YOLO)
- pandas, numpy, matplotlib
- QATCH internal modules (Logger, DataProcessor, FillClassifier)

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2026-01-09

Version:
    6.0.1
"""

import os
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from QATCH.common.logger import (
        Logger as Log,  # pyright: ignore[reportPrivateImportUsage]
    )
    from QATCH.QModel.src.models.v6_yolo.v6_yolo_dataprocessor import (
        QModelV6YOLO_DataProcessor,
    )

except (ImportError, ModuleNotFoundError):

    class Log:
        @staticmethod
        def d(tag: str, message: str):
            print(f"{tag} [DEBUG] {message}")

        @staticmethod
        def i(tag: str, message: str):
            print(f"{tag} [INFO] {message}")

        @staticmethod
        def w(tag: str, message: str):
            print(f"{tag} [WARNING] {message}")

        @staticmethod
        def e(tag: str, message: str):
            print(f"{tag} [ERROR] {message}")

    Log.i(tag="[HEADLESS OPERATION]", message="=== RUNNING HEADLESS ===")
    from v6_yolo_dataprocessor import QModelV6YOLO_DataProcessor

try:
    from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]
except ImportError:
    Log.e(
        tag="[QModelV6YOLO]",
        message="'ultralytics' not found. YOLO inference will fail.",
    )


# --- Configuration Constants ---
class QModelV6Config:
    """Configuration constants for the QModel V6 YOLO pipeline."""

    # --- Detector Settings ---
    IMG_WIDTH: int = 2560
    IMG_HEIGHT: int = 384
    MIN_SLICE_LENGTH: int = 20
    CONF_THRESHOLD: float = 0.01

    # --- Fill Classifier Settings ---
    FILL_INFERENCE_W: int = 224
    FILL_INFERENCE_H: int = 224
    FILL_GEN_W: int = 640
    FILL_GEN_H: int = 640

    # Maps YOLO classification labels to the number of channels to detect.
    # The Controller uses this Int to decide how many 'cuts' to make.
    FILL_CLASS_MAP: Dict[str, int] = {
        "no_fill": -1,
        "initial_fill": 0,
        "1ch": 1,
        "2ch": 2,
        "3ch": 3,
    }

    # Progress Signal Steps
    PROG_LOAD_DATA: int = 10
    PROG_CLASSIFY: int = 30
    PROG_CONFIG: int = 40
    PROG_COMPLETE: int = 100


class QModelV6YOLO_FillClassifier:
    """
    Handles the classification of the run state (e.g., no_fill, initial_fill, 1ch, 2ch, 3ch).

    This class loads a specific YOLO classification model to analyze the raw sensor data
    visuals and determine which "Fill State" the run belongs to. This classification
    dictates how many channels (if any) the subsequent detection pipeline should search for.
    """

    TAG = "[QModelV6YOLO_FillClassifier]"

    def __init__(self, model_path: str):
        """
        Initializes the Fill Classifier with the provided model weights.

        Args:
            model_path (str): The absolute path to the .pt classification model file.

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If the YOLO model fails to load (e.g., corrupted file).
        """
        if not os.path.exists(model_path):
            Log.e(self.TAG, f"Model not found at: {model_path}")
            raise FileNotFoundError(f"Model not found at: {model_path}")

        try:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

            Log.i(self.TAG, f"Loading Fill Classifier from {model_path}...")
            self.model = YOLO(model_path)

        except Exception as e:
            Log.e(self.TAG, f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def predict(self, df: pd.DataFrame) -> int:
        """
        Generates a visual representation of the run data and classifies its fill state.

        This method:
        1. Generates a 3-strip stacked image from the dataframe using `DataProcessor`.
        2. Resizes the image to the model's expected inference resolution.
        3. Runs YOLO classification.
        4. Maps the predicted class label to an integer channel count.

        Args:
            df (pd.DataFrame): The raw sensor data to classify.

        Returns:
            int: The number of channels to detect (0, 1, 2, or 3). Returns 0 on failure
            or if the classification result implies no channels (e.g., "no_fill").
        """
        if df is None or df.empty:
            Log.w(self.TAG, "Dataframe provided for prediction is empty.")
            return 0

        # Generate Image
        # We divide the target GEN_H by 3 because the processor stacks 3 strips
        strip_height = QModelV6Config.FILL_GEN_H // 3

        try:
            img_high_res = QModelV6YOLO_DataProcessor.generate_fill_cls(
                df, img_h=strip_height, img_w=QModelV6Config.FILL_GEN_W
            )
        except Exception as e:
            Log.e(self.TAG, f"Error generating signal image: {e}")
            return 0

        if img_high_res is None:
            Log.w(self.TAG, "Generated image is None.")
            return 0

        # Resize for Inference
        img_input = cv2.resize(
            img_high_res,
            (QModelV6Config.FILL_INFERENCE_W, QModelV6Config.FILL_INFERENCE_H),
            interpolation=cv2.INTER_AREA,
        )

        # Inference
        try:
            results = self.model(img_input, verbose=False)
            if not results:
                Log.w(self.TAG, "Model returned no results.")
                return 0

            probs = results[0].probs
            top1_index = probs.top1

            # Robustly get the label name
            pred_label = results[0].names[top1_index]
            confidence = probs.top1conf.item()

            Log.d(self.TAG, f"Prediction: '{pred_label}' ({confidence:.1%})")

            if confidence < 0.5:
                Log.w(
                    self.TAG,
                    f"Low confidence ({confidence:.2f}) for class: {pred_label}",
                )

            return self._map_label_to_channels(pred_label)

        except Exception as e:
            Log.e(self.TAG, f"Inference error: {e}")
            return 0

    def _map_label_to_channels(self, label: str) -> int:
        """
        Maps the string label from YOLO to the integer channel count.

        Args:
            label (str): The predicted class label (e.g., "3ch", "no_fill").

        Returns:
            int: The corresponding channel count constant, or 0 if unknown.
        """
        label_clean = str(label).strip().lower()

        if label_clean in QModelV6Config.FILL_CLASS_MAP:
            return QModelV6Config.FILL_CLASS_MAP[label_clean]
        if label_clean.isdigit():
            return int(label_clean)
        Log.w(self.TAG, f"Unknown label '{label}'. Defaulting to 0 channels.")
        return 0


class QModelV6YOLO_Detector:
    """
    Generic wrapper for a single YOLO detector instance.

    This class encapsulates the loading and inference logic for a specific YOLO model
    (e.g., Init, Ch1, Ch2, or Ch3). It handles the conversion of input DataFrame slices
    into model-compatible images, executes the inference, and maps the normalized
    bounding box coordinates back to the time domain of the provided data slice.
    """

    def __init__(self, model_path: str):
        """
        Initializes the detector with a specific YOLO model.

        Args:
            model_path (str): The file path to the .pt or .onnx model weights.

        Raises:
            FileNotFoundError: If the model file does not exist at the specified path.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Detector model not found: {model_path}")
        self.model = YOLO(model_path)

    def predict_single(
        self, df: pd.DataFrame, target_class_map: Optional[Dict[int, int]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Runs inference on a specific slice of the sensor data.

        This method generates a visual representation of the provided DataFrame slice,
        runs the YOLO object detection, and converts the resulting bounding box
        coordinates (normalized 0-1) into actual timestamps based on the time range
        of the slice.

        Args:
            df (pd.DataFrame): The dataframe slice to analyze. Must contain a time column
                (e.g., 'Relative_time'). Data slices smaller than `MIN_SLICE_LENGTH` are ignored.
            target_class_map (Optional[Dict[int, int]]): A mapping from the YOLO model's
                internal class IDs (keys) to the application's POI IDs (values).
                If provided, the output dictionary will use the POI IDs as keys.
                Defaults to None.

        Returns:
            Dict[int, Dict[str, Any]]: A dictionary of the best detection for each class.
                Structure:
                {
                    poi_id: {
                        "time": float,  # The detected timestamp in seconds
                        "conf": float   # The model confidence score (0.0 - 1.0)
                    },
                    ...
                }
                Note: This method does *not* calculate the absolute row index; that must
                be handled by the controller using the returned time.
        """
        if df is None or len(df) < QModelV6Config.MIN_SLICE_LENGTH:
            return {}
        img_base = QModelV6YOLO_DataProcessor.generate_channel_det(
            df, img_w=QModelV6Config.IMG_WIDTH, img_h=QModelV6Config.IMG_HEIGHT
        )
        results = self.model(
            img_base, verbose=False, conf=QModelV6Config.CONF_THRESHOLD
        )
        col_time = "Relative_time"
        if col_time not in df.columns:
            col_time = "time" if "time" in df.columns else df.columns[0]

        time_vals = df[col_time].to_numpy(dtype=float)
        x_min, x_max = time_vals.min(), time_vals.max()

        best_dets = {}

        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf.item()

                # Keep the highest confidence detection for each class found
                if cls_id not in best_dets or conf > best_dets[cls_id]["conf"]:
                    x_norm = box.xywhn[0][0].item()
                    t = x_norm * (x_max - x_min) + x_min
                    best_dets[cls_id] = {"time": t, "conf": conf}

        # Map to Output format
        final_results = {}
        if target_class_map:
            for yolo_id, poi_id in target_class_map.items():
                if yolo_id in best_dets:
                    data = best_dets[yolo_id]
                    final_results[poi_id] = {
                        "conf": data["conf"],
                        "time": data["time"],
                    }
        else:
            final_results = best_dets

        return final_results


class QModelV6YOLO:
    """
    Controller class for the QModel V6 YOLO .

    This class manages the various machine YOLO models used in the
    V6 pipeline. It handles the lazy loading of the fill classifier and specific channel
    detectors (Init, Ch1, Ch2, Ch3) to optimize memory usage, ensuring models are only
    loaded when required by the prediction logic.
    """

    TAG = "QModelV6YOLO"

    # Maps internal integer Class IDs to application-standard POI strings
    POI_MAP = {1: "POI1", 2: "POI2", 3: "POI3", 4: "POI4", 5: "POI5", 6: "POI6"}

    def __init__(self, model_assets: Dict[str, Any]):
        """
        Initializes the QModelV6YOLO controller.

        Args:
            model_assets (Dict[str, Any]): A dictionary containing paths to model weights.
                Expected structure:
                {
                    "fill_classifier": "path/to/classifier.pt",
                    "detectors": {
                        "init": "path/to/init.pt",
                        "ch1": "path/to/ch1.pt",
                        # ... etc
                    }
                }
        """
        self.model_assets = model_assets
        self._fill_classifier = None
        self._detectors: Dict[str, Any] = {
            "init": None,
            "ch1": None,
            "ch2": None,
            "ch3": None,
        }

    def _load_fill_cls(self) -> Any:
        """
        Lazy loads the Fill Classifier model.

        Checks if the classifier is already loaded; if not, attempts to load it using
        the path provided in `model_assets`.

        Returns:
            Any: The loaded `QModelV6YOLO_FillClassifier` instance, or None if loading failed
            or no path was provided.
        """
        if self._fill_classifier is None:
            model_path = self.model_assets.get("fill_classifier")
            if model_path:
                self._fill_classifier = QModelV6YOLO_FillClassifier(model_path)
        return self._fill_classifier

    def _load_detector_by_name(self, name: str) -> Any:
        """
        Lazy loads a specific YOLO detector by its shorthand name.

        Args:
            name (str): The key for the detector to load (e.g., "init", "ch1", "ch2", "ch3").

        Returns:
            Any: The loaded `QModelV6YOLO_Detector` instance, or None if the path is missing
            or loading fails.
        """
        if self._detectors.get(name) is None:
            detector_paths = self.model_assets.get("detectors", {})
            model_path = detector_paths.get(name)
            if model_path:
                try:
                    self._detectors[name] = QModelV6YOLO_Detector(model_path)
                except Exception as e:
                    Log.e(self.TAG, f"Error while loading detector: {e}")
                    return None
        return self._detectors.get(name)

    def _get_default_predictions(self) -> Dict[str, Dict[str, List]]:
        """
        Generates a default prediction dictionary initialized with placeholder values.

        This is typically used as a fallback return value when predictions fail or cannot
        be computed (e.g., due to errors or missing models). Every mapped POI name is
        initialized with an index of -1 and a confidence of -1 to indicate "no detection."

        Returns:
            Dict[str, Dict[str, List]]: A dictionary where keys are POI names (e.g., "POI1")
            and values are standard result dictionaries:
            {"indices": [-1], "confidences": [-1]}.
        """
        return {
            poi_name: {"indices": [-1], "confidences": [-1]}
            for poi_name in self.POI_MAP.values()
        }

    def _format_output(
        self, final_results: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Formats the raw detection results into the standardized output dictionary.

        Maps the internal integer POI IDs (e.g., 1-6) to their string representations
        (e.g., "POI1", "POI2") using the class `POI_MAP`. It ensures every expected POI
        is present in the output, filling missing ones with default placeholder values (-1).

        Args:
            final_results (Dict[int, Dict[str, Any]]): A dictionary containing successful
                detections, keyed by internal POI ID. Values should include "index" and "conf".

        Returns:
            Dict[str, Dict[str, List[float]]]: The formatted output dictionary structured as:
                {
                    "POI_NAME": {
                        "indices": [int],       # Row index (in list for compatibility)
                        "confidences": [float]  # Confidence score
                    },
                    ...
                }
        """
        output = {}
        for poi_num, poi_name in self.POI_MAP.items():
            if poi_num in final_results:
                data = final_results[poi_num]
                output[poi_name] = {
                    "indices": [data["index"]],
                    "confidences": [data["conf"]],
                }
            else:
                # Fill missing POIs with failure placeholders
                output[poi_name] = {"indices": [-1], "confidences": [-1]}
        return output

    def _validate_file_buffer(self, file_buffer: Union[str, Any]) -> pd.DataFrame:
        """
        Loads and validates CSV data from a file path or a file-like object.

        If a file-like object (buffer) is provided, this method attempts to reset the
        pointer to the beginning using `.seek(0)` to ensure a full read.

        Args:
            file_buffer (Union[str, Any]): A file path string or a file-like object
                (e.g., io.BytesIO, open file handle) containing the CSV data.

        Returns:
            pd.DataFrame: The loaded pandas DataFrame.

        Raises:
            Exception: If the file cannot be read or parsed by pandas.
        """
        try:
            if not isinstance(file_buffer, str):
                if hasattr(file_buffer, "seekable") and file_buffer.seekable():
                    file_buffer.seek(0)
            df = pd.read_csv(file_buffer)
        except Exception as e:
            raise e
        return df

    def _get_raw_index(self, raw_df: pd.DataFrame, target_time: float) -> int:
        """
        Maps a target time value back to its absolute row index in the original DataFrame.

        This method performs a nearest-neighbor search on the time column of the raw
        dataset to find the row corresponding to the detected time. This is critical for
        resolving index shifts that occur when the processing pipeline drops rows (e.g.,
        during cleaning or slicing), ensuring that the final output index aligns perfectly
        with the original user data.

        Args:
            raw_df (pd.DataFrame): The original, unprocessed DataFrame. Must contain a
                time column (e.g., "Relative_time").
            target_time (float): The time value (in seconds) associated with a detection.

        Returns:
            int: The absolute row index (from `df.index`) of the sample closest to the
            target time.
        """
        col_time = "Relative_time"
        if col_time not in raw_df.columns:
            col_time = "time" if "time" in raw_df.columns else raw_df.columns[0]
        times = raw_df[col_time].to_numpy(dtype=float)
        idx = (np.abs(times - target_time)).argmin()
        return int(raw_df.index[idx.item()])

    def _visualize(
        self,
        df: pd.DataFrame,
        results: dict,
        cut_history: list,
        save_path: str = "v6_debug.png",
    ) -> None:
        """
        Generates a debug plot illustrating the cascade detection process.

        Visualizes the raw sensor signal ('Dissipation' vs 'Relative_time'), overlays the
        final predicted POI positions as colored vertical lines, and highlights the
        data slicing steps performed during the reverse cascade (indicating which parts
        of the signal were "cut" or excluded for subsequent detectors).

        Args:
            df (pd.DataFrame): The master DataFrame containing the full sensor run data.
                Must contain 'Relative_time' and 'Dissipation' columns.
            results (Dict[int, Dict[str, Any]]): The dictionary of detection results,
                where keys are POI IDs (int) and values are dictionaries containing
                the 'time' (float) of the prediction.
            cut_history (List[Tuple[str, float]]): A list of tuples recording the slicing
                actions taken. Each tuple contains (cut_name, cut_time).
            save_path (str, optional): The file path where the plot image will be saved.
                Defaults to "v6_debug.png".
        """
        if df is None or df.empty:
            return
        time = df["Relative_time"].values
        signal = (
            df["Dissipation"].values
            if "Dissipation" in df.columns
            else df.iloc[:, 1].values
        )
        plt.figure(figsize=(12, 6))
        plt.plot(time, signal, color="gray", alpha=0.6, label="Raw Signal")

        colors = {1: "green", 2: "blue", 4: "orange", 5: "red", 6: "purple"}
        for poi_id, data in results.items():
            if data and "time" in data:
                t = data["time"]
                c = colors.get(poi_id, "black")
                name = self.POI_MAP.get(poi_id, f"POI{poi_id}")
                plt.axvline(x=t, color=c, linestyle="-", linewidth=2, label=f"{name}")

        for _, (_, cut_time) in enumerate(cut_history):
            plt.axvline(x=cut_time, color="red", linestyle="--", linewidth=1, alpha=0.5)
            plt.axvspan(cut_time, np.max(time), color="red", alpha=0.05)

        plt.title(f"Cascade Detection Debug - {len(cut_history)} Slices Applied")
        plt.savefig(save_path)
        plt.close()

    def predict(
        self,
        progress_signal: Any = None,
        file_buffer: Any = None,
        df: pd.DataFrame | None = None,
        visualize: bool = False,
        num_channels: int | None = None,
    ) -> Tuple[Dict[str, Dict[str, List]], int]:
        """
        Executes the QModel V6 YOLO prediction pipeline on the provided data.

        This method orchestrates the complete detection workflow. It handles data loading,
        preprocessing, and fill type classification. If a specific channel count is not
        provided, it runs the classifier. It then executes a "Reverse Cascading Detection"
        strategy (Ch3 -> Ch2 -> Ch1 -> Init), cutting the dataset at each detection to
        isolate the signal for the next detector in the sequence.

        Args:
            progress_signal (Any, optional): A signal object (e.g., PyQt Signal) used to emit
                progress updates. Expected to have an `.emit(int, str)` method.
                Defaults to None.
            file_buffer (Any, optional): A file path (str) or file-like object containing
                CSV data. Used if `df` is not provided. Defaults to None.
            df (pd.DataFrame, optional): A pre-loaded pandas DataFrame containing the
                sensor data. Ignored if `file_buffer` is provided. Defaults to None.
            visualize (bool, optional): If True, triggers the generation of a visualization
                plot showing detections and cut points. Defaults to False.
            num_channels (int, optional): The number of channels to enforce. If None,
                the fill classifier is used to automatically determine the channel count.
                Defaults to None.

        Returns:
            Tuple[Dict[str, Dict[str, List]], int]: A tuple containing:
                1. A dictionary of predictions mapping POI names to their results:
                   {
                       "POI_NAME": {
                           "indices": [int],       # Row indices in the raw DataFrame
                           "confidences": [float], # Model confidence scores
                           "time": [float]         # (Optional) Time values if retained
                       },
                       ...
                   }
                2. The integer number of channels detected (or enforced) for this run.

        Note:
            If an error occurs during execution (e.g., missing data, preprocessing failure),
            the method catches the exception, logs the error, and returns a default
            prediction dictionary filled with placeholder values (-1).
        """
        try:
            if file_buffer is not None:
                raw_df = self._validate_file_buffer(file_buffer)
            elif df is None:
                raise ValueError("No data provided")
            else:
                raw_df = df

            if progress_signal:
                progress_signal.emit(10, "Data Loaded")

            master_df = QModelV6YOLO_DataProcessor.preprocess_dataframe(raw_df.copy())
            if master_df is None or master_df.empty:
                raise ValueError("Preprocessing failed")

            # --- Fill Type Classification ---
            if num_channels is None:
                fill_cls = self._load_fill_cls()
                if fill_cls:
                    if progress_signal:
                        progress_signal.emit(30, "Classifying Run Type...")
                    num_channels = fill_cls.predict(master_df)
                else:
                    # Fallback default if classifier fails/missing
                    num_channels = 3

            # ensure int for comparison
            num_channels = int(num_channels)

            # If the classifier detected "no_fill"
            if num_channels == -1:
                if progress_signal:
                    progress_signal.emit(100, "No Fill Detected - Skipping Analysis")
                Log.i(
                    self.TAG,
                    "Run classified as 'no_fill'. Returning default placeholders.",
                )
                return self._get_default_predictions(), num_channels

            # --- Reverse Cascading Detection ---
            final_results = {}
            current_df = master_df.copy()  # Working slice
            col_time = (
                "Relative_time"
                if "Relative_time" in current_df.columns
                else current_df.columns[0]
            )
            cut_history = []

            # Helper to process results and map to RAW index
            def process_detection(res_dict, poi_id):
                if poi_id in res_dict:
                    t_det = res_dict[poi_id]["time"]
                    conf_det = res_dict[poi_id]["conf"]
                    raw_idx = self._get_raw_index(raw_df, t_det)
                    final_results[poi_id] = {
                        "index": raw_idx,
                        "conf": conf_det,
                        "time": t_det,
                    }
                    return t_det
                return None

            # 3rd Channel (POI 6) - Runs only if 3+ channels detected
            if num_channels >= 3:
                det_ch3 = self._load_detector_by_name("ch3")
                if det_ch3:
                    res = det_ch3.predict_single(current_df, target_class_map={0: 6})
                    cut_time = process_detection(res, 6)
                    if cut_time:
                        current_df = current_df[current_df[col_time] < cut_time]
                        cut_history.append(("CH3_Cut", cut_time))

            # 2nd Channel (POI 5) - Runs if 2+ channels detected
            if num_channels >= 2:
                det_ch2 = self._load_detector_by_name("ch2")
                if det_ch2:
                    res = det_ch2.predict_single(current_df, target_class_map={0: 5})
                    cut_time = process_detection(res, 5)
                    if cut_time:
                        current_df = current_df[current_df[col_time] < cut_time]
                        cut_history.append(("CH2_Cut", cut_time))

            # 1st Channel (POI 4) - Runs if 1+ channels detected
            if num_channels >= 1:
                det_ch1 = self._load_detector_by_name("ch1")
                if det_ch1:
                    res = det_ch1.predict_single(current_df, target_class_map={0: 4})
                    cut_time = process_detection(res, 4)
                    if cut_time:
                        current_df = current_df[current_df[col_time] < cut_time]
                        cut_history.append(("CH1_Cut", cut_time))

            # Initial Fill (POI 1 & 2) - Runs for 0, 1, 2, or 3 channels
            # Since we guarded against -1 above, this runs for 'initial_fill' (0) and up.
            det_init = self._load_detector_by_name("init")
            if det_init:
                res = det_init.predict_single(current_df, target_class_map={0: 1, 1: 2})
                process_detection(res, 1)
                process_detection(res, 2)

            if progress_signal:
                progress_signal.emit(100, "Inference Complete")

            if visualize:
                try:
                    self._visualize(master_df, final_results, cut_history)
                except Exception as e:
                    Log.w(self.TAG, f"Visualization failed: {e}")

            return self._format_output(final_results), num_channels

        except Exception as e:
            Log.e(self.TAG, f"Error during prediction: {e}")
            traceback.print_exc()
            return self._get_default_predictions(), 0
