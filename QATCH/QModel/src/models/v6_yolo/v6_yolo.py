import os
import traceback
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

# Try imports, handle missing dependencies for robust loading
try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: 'ultralytics' not found. YOLO inference will fail.")

try:
    from QATCH.common.logger import Logger as Log

    # Assuming DataProcessor is available
    from QATCH.QModel.src.models.v6_yolo.v6_yolo_dataprocessor import (
        QModelV6YOLO_DataProcessor,
    )
    from QATCH.QModel.src.models.v6_yolo.v6_yolo_fill_cls import (
        QModelV6YOLO_FillClassifier,
    )
except (ImportError, ModuleNotFoundError):
    # Stub logging and processor if running outside QATCH environment
    class Log:
        @staticmethod
        def d(tag: str = "", message: str = ""):
            print(f"{tag} [DEBUG] {message}")

        @staticmethod
        def i(tag: str = "", message: str = ""):
            print(f"{tag} [INFO] {message}")

        @staticmethod
        def w(tag: str = "", message: str = ""):
            print(f"{tag} [WARNING] {message}")

        @staticmethod
        def e(tag: str = "", message: str = ""):
            print(f"{tag} [ERROR] {message}")

    if "QModelV6YOLO_DataProcessor" not in globals():

        class QModelV6YOLO_DataProcessor:
            @staticmethod
            def preprocess_dataframe(df):
                return df  # Pass through for stub


class QModelV6YOLO_Detector:
    """
    Generic Wrapper for a single YOLO detector.
    Used for Init, Ch1, Ch2, and Ch3 specific models.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Detector model not found: {model_path}")
        self.model = YOLO(model_path)

    def predict_single(
        self, df: pd.DataFrame, target_class_map: Dict[int, int] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Runs inference on the provided dataframe slice.

        Args:
            df: The (potentially truncated) dataframe.
            target_class_map: Dict mapping YOLO Class ID -> App POI ID.
                              If None, returns raw class IDs.

        Returns:
            Dict {poi_id: {'index': int, 'conf': float, 'time': float}}
        """
        # 1. Generate Image for this specific slice
        # The DataProcessor should handle auto-scaling based on the df passed
        img_base = QModelV6YOLO_DataProcessor.generate_channel_det(
            df, img_w=2560, img_h=384
        )

        # 2. Inference (Low conf to catch candidates)
        results = self.model(img_base, verbose=False, conf=0.001)

        # 3. Parse Results
        col_time = "Relative_time"
        time_vals = df[col_time].values
        x_min, x_max = time_vals.min(), time_vals.max()

        best_dets = {}

        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf.item()

                # Keep highest confidence per class
                if cls_id not in best_dets or conf > best_dets[cls_id]["conf"]:
                    x_norm = box.xywhn[0][0].item()
                    t = x_norm * (x_max - x_min) + x_min

                    # Map to Nearest Index in THIS slice
                    # Note: We return the global index from the 'Relative_time' match
                    # assuming the df slice preserves values, but we need the index relative
                    # to the original DF if the caller needs absolute rows.
                    # However, typical usage is mapping time back to index later or
                    # trusting the processor returns a slice of the original with original index.

                    best_dets[cls_id] = {"time": t, "conf": conf}

        # 4. Map to Output format
        final_results = {}
        if target_class_map:
            for yolo_id, poi_id in target_class_map.items():
                if yolo_id in best_dets:
                    data = best_dets[yolo_id]
                    # Find nearest index in current df slice
                    idx = (np.abs(time_vals - data["time"])).argmin()
                    # If df is a slice, we need the original index if possible.
                    # If df was reset_index'd, this is local. If not, df.index[idx] is global.
                    real_index = df.index[idx]

                    final_results[poi_id] = {
                        "index": int(real_index),
                        "conf": data["conf"],
                        "time": data["time"],
                    }
        else:
            # Return raw if no map provided
            final_results = best_dets

        return final_results


class QModelV6YOLO:
    """
    Controller class for the QModel V6 YOLO pipeline.
    Implements the "Reverse Cascade" detection logic:
    3ch -> Cut -> 2ch -> Cut -> 1ch -> Cut -> Init Fill
    """

    TAG = "QModelV6YOLO"

    # Legacy Output Map
    POI_MAP = {1: "POI1", 2: "POI2", 3: "POI3", 4: "POI4", 5: "POI5", 6: "POI6"}

    def __init__(self, model_assets: Dict[str, Any]):
        """
        Args:
            model_assets (dict): Dictionary containing paths to model weights.
                Expected keys: 'fill_classifier', 'detectors': {'init', 'ch1', 'ch2', 'ch3'}
        """
        self.model_assets = model_assets
        self._fill_classifier = None

        # Cache for loaded models
        self._detectors = {"init": None, "ch1": None, "ch2": None, "ch3": None}

    def _load_fill_cls(self):
        """Lazy loads the fill classifier."""
        if self._fill_classifier is None:
            model_path = self.model_assets.get("fill_classifier")
            if model_path:
                Log.i(self.TAG, f"Loading Fill Classifier from {model_path}")
                self._fill_classifier = QModelV6YOLO_FillClassifier(model_path)
            else:
                Log.e(self.TAG, "No path provided for Fill Classifier.")
        return self._fill_classifier

    def _load_detector_by_name(self, name: str):
        """Lazy loads a specific detector by name (init, ch1, ch2, ch3)."""
        if self._detectors.get(name) is None:
            detector_paths = self.model_assets.get("detectors", {})
            model_path = detector_paths.get(name)

            if model_path:
                Log.i(self.TAG, f"Loading Detector '{name}' from {model_path}")
                try:
                    self._detectors[name] = QModelV6YOLO_Detector(model_path)
                except Exception as e:
                    Log.e(self.TAG, f"Failed to load detector '{name}': {e}")
                    return None
            else:
                # Warning only, as some runs might not need all detectors
                Log.w(self.TAG, f"No model path found for detector '{name}'.")

        return self._detectors.get(name)

    def _get_default_predictions(self) -> Dict[str, Dict[str, List]]:
        return {
            poi_name: {"indices": [-1], "confidences": [-1]}
            for poi_name in self.POI_MAP.values()
        }

    def _format_output(
        self, final_results: Dict[int, Dict]
    ) -> Dict[str, Dict[str, List[float]]]:
        """Format predictions into the standardized output dictionary."""
        output = {}
        for poi_num, poi_name in self.POI_MAP.items():
            if poi_num in final_results:
                data = final_results[poi_num]
                output[poi_name] = {
                    "indices": [data["index"]],
                    "confidences": [data["conf"]],
                }
            else:
                output[poi_name] = {"indices": [-1], "confidences": [-1]}
        return output

    def _validate_file_buffer(self, file_buffer: Union[str, object]) -> pd.DataFrame:
        """Standard CSV validation."""
        try:
            if hasattr(file_buffer, "seekable") and file_buffer.seekable():
                file_buffer.seek(0)
            df = pd.read_csv(file_buffer)
        except Exception as e:
            raise ValueError(f"Failed to read data file: {e}")

        required_columns = {"Dissipation", "Resonance_Frequency", "Relative_time"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        return df

    def predict(
        self,
        progress_signal: Any = None,
        file_buffer: Any = None,
        df: pd.DataFrame = None,
        visualize: bool = False,
        num_channels: int = None,
    ) -> Tuple[Dict[str, Dict[str, List]], int]:
        """
        Main entry point for prediction.
        """
        try:
            # --- 1. Data Loading ---
            if file_buffer is not None:
                df = self._validate_file_buffer(file_buffer)
            elif df is None:
                raise ValueError("No data provided")

            if progress_signal:
                progress_signal.emit(10, "Data Loaded")

            # --- 2. Preprocessing ---
            # We treat the input DF as the 'master' source.
            # DataProcessor is used inside the detectors on slices,
            # but we might need a global preprocess here if required.
            master_df = QModelV6YOLO_DataProcessor.preprocess_dataframe(df.copy())
            if master_df is None or master_df.empty:
                raise ValueError("Preprocessing failed")

            # --- 3. Classification ---
            if num_channels is None:
                fill_cls = self._load_fill_cls()
                if fill_cls:
                    if progress_signal:
                        progress_signal.emit(30, "Classifying Run Type...")
                    num_channels = fill_cls.predict(master_df)
                else:
                    num_channels = 3  # Default fallback

            num_channels = int(num_channels)
            if progress_signal:
                progress_signal.emit(40, f"Configured for {num_channels} Channels")

            # --- 4. Cascading Detection ---
            # Results storage: Key = Legacy POI ID (1-6)
            final_results = {}

            # Working copy that gets sliced
            current_df = master_df.copy()

            # --- Step A: 3rd Channel (Legacy POI 6) ---
            if num_channels >= 3:
                det_ch3 = self._load_detector_by_name("ch3")
                if det_ch3:
                    # Model Class 0 -> Legacy POI 6
                    res = det_ch3.predict_single(current_df, target_class_map={0: 6})
                    if 6 in res:
                        final_results[6] = res[6]
                        # CUT DATASET
                        cut_time = res[6]["time"]
                        current_df = current_df[current_df["Relative_time"] < cut_time]
                        Log.d(
                            self.TAG, f"Detected POI6 at {cut_time:.2f}s. Cutting data."
                        )

            # --- Step B: 2nd Channel (Legacy POI 5) ---
            if num_channels >= 2:
                det_ch2 = self._load_detector_by_name("ch2")
                if det_ch2:
                    # Model Class 0 -> Legacy POI 5
                    res = det_ch2.predict_single(current_df, target_class_map={0: 5})
                    if 5 in res:
                        final_results[5] = res[5]
                        # CUT DATASET
                        cut_time = res[5]["time"]
                        current_df = current_df[current_df["Relative_time"] < cut_time]
                        Log.d(
                            self.TAG, f"Detected POI5 at {cut_time:.2f}s. Cutting data."
                        )

            # --- Step C: 1st Channel (Legacy POI 4) ---
            if num_channels >= 1:
                det_ch1 = self._load_detector_by_name("ch1")
                if det_ch1:
                    # Model Class 0 -> Legacy POI 4
                    res = det_ch1.predict_single(current_df, target_class_map={0: 4})
                    if 4 in res:
                        final_results[4] = res[4]
                        # CUT DATASET
                        cut_time = res[4]["time"]
                        current_df = current_df[current_df["Relative_time"] < cut_time]
                        Log.d(
                            self.TAG, f"Detected POI4 at {cut_time:.2f}s. Cutting data."
                        )

            # --- Step D: Initial Fill (Legacy POI 1 & 2) ---
            # Always run on the remaining data (init phase)
            det_init = self._load_detector_by_name("init")
            if det_init:
                # Model Class 0 -> POI 1
                # Model Class 1 -> POI 2
                res = det_init.predict_single(current_df, target_class_map={0: 1, 1: 2})
                final_results.update(res)

            # --- Step E: Legacy Placeholder (POI 3) ---
            # Explicitly ensure POI 3 is missing (handled by _format_output as -1) or force it here
            # User requirement: "POI3 ... should just be filled to -1"
            # _format_output will handle keys missing from final_results by setting them to -1.

            if progress_signal:
                progress_signal.emit(100, "Inference Complete")

            # --- 5. Formatting ---
            predictions = self._format_output(final_results)
            return predictions, num_channels

        except Exception as e:
            Log.e(self.TAG, f"Error during prediction: {e}")
            traceback.print_exc()
            return self._get_default_predictions(), 0
