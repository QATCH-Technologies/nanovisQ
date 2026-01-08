import os
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

    # Assuming DataProcessor is available or using the stub below
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
        def d(tag: str = "", message: str = ""): print(f"{tag} [DEBUG] {message}")
        @staticmethod
        def i(tag: str = "", message: str = ""): print(f"{tag} [INFO] {message}")
        @staticmethod
        def w(tag: str = "", message: str = ""): print(f"{tag} [WARNING] {message}")
        @staticmethod
        def e(tag: str = "", message: str = ""): print(f"{tag} [ERROR] {message}")

    if "QModelV6YOLO_DataProcessor" not in globals():
        class QModelV6YOLO_DataProcessor:
            @staticmethod
            def preprocess_dataframe(df): return df # Pass through for stub

class QModelV6YOLO_Detector:
    """
    Wraps the YOLO detection logic, including the 3-channel forcing/repair algorithms.
    """
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Detector model not found: {model_path}")
        self.model = YOLO(model_path)

    def _repair_topology(self, best_dets: Dict, targets: List[int], x_min: float, x_max: float):
        """
        Forces the detection set to contain all target keys by interpolating/extrapolating.
        """
        sorted_targets = sorted(targets)
        
        # 1. Handle Empty Case (Synthetic distribution)
        valid_count = sum(1 for t in sorted_targets if best_dets[t] is not None)
        if valid_count == 0:
            step = (x_max - x_min) / (len(sorted_targets) + 1)
            for i, t in enumerate(sorted_targets):
                best_dets[t] = {
                    "time": x_min + (step * (i + 1)),
                    "conf": 0.0
                }
            return best_dets

        # 2. Iterative Fill
        while any(best_dets[t] is None for t in sorted_targets):
            for i, t in enumerate(sorted_targets):
                if best_dets[t] is not None:
                    continue

                # Find Neighbors
                prev_node = next((best_dets[sorted_targets[j]] for j in range(i-1, -1, -1) if best_dets[sorted_targets[j]]), None)
                next_node = next((best_dets[sorted_targets[j]] for j in range(i+1, len(sorted_targets)) if best_dets[sorted_targets[j]]), None)

                fill_time = None
                
                if prev_node and next_node:
                    fill_time = (prev_node["time"] + next_node["time"]) / 2
                elif prev_node:
                    step = (x_max - x_min) * 0.05
                    fill_time = min(prev_node["time"] + step, x_max)
                elif next_node:
                    step = (x_max - x_min) * 0.05
                    fill_time = max(next_node["time"] - step, x_min)

                if fill_time is not None:
                    best_dets[t] = {"time": fill_time, "conf": 0.0}

        return best_dets

    def predict(self, df: pd.DataFrame, num_channels: int) -> Tuple[Dict[int, int], Dict[int, float]]:
        """
        Runs inference and returns indices and confidences.
        """
        # 1. Generate Image
        img_base = QModelV6YOLO_DataProcessor.generate_channel_det(df, img_w=2560, img_h=384) # Uses imported visualizer
        
        # 2. Inference (Low conf to catch candidates)
        results = self.model(img_base, verbose=False, conf=0.001)

        # 3. Define Targets based on Channel Count
        # Map: 3ch -> POI 0,1,2,3,4 (internal IDs 0-4 for simplicity in logic, mapped to 1-5 later)
        # Note: Adjust logic if your model classes are 0-4 but POI_MAP is 1-5.
        # Assuming Model Classes 0..N map to POI 1..N+1
        if num_channels == 3:
            target_classes = [0, 1, 2, 3, 4]
        elif num_channels == 2:
            target_classes = [0, 1, 2, 3]
        elif num_channels == 1:
            target_classes = [0, 1, 2]
        else:
            target_classes = [0, 1] # Fallback/Init

        best_dets = {c: None for c in target_classes}
        
        # Time helpers
        col_time = "Relative_time"
        time_vals = df[col_time].values
        x_min, x_max = time_vals.min(), time_vals.max()

        # 4. Parse Boxes
        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in best_dets:
                    conf = box.conf.item()
                    # Logic: Take highest confidence
                    if best_dets[cls_id] is None or conf > best_dets[cls_id]["conf"]:
                        x_norm = box.xywhn[0][0].item()
                        t = x_norm * (x_max - x_min) + x_min
                        best_dets[cls_id] = {"time": t, "conf": conf}

        # 5. Repair Topology (Force 3ch logic)
        # We always repair for 3ch, optionally others
        if num_channels == 3:
            best_dets = self._repair_topology(best_dets, target_classes, x_min, x_max)

        # 6. Map Time -> Index
        final_positions = {}
        confidences = {}

        for cls_id, data in best_dets.items():
            if data:
                # Find nearest index for the predicted time
                # (abs(df['time'] - target)).argmin()
                idx = (np.abs(time_vals - data["time"])).argmin()
                
                # Map Model Class ID (0-based) to App POI ID (1-based)
                # Assuming Model Class 0 = POI1
                poi_id = cls_id + 1 
                
                final_positions[poi_id] = int(idx)
                confidences[poi_id] = float(data["conf"])

        return final_positions, confidences


class QModelV6YOLO:

    TAG = "[QModelv6YOLO]"

    # Map internal class IDs to Output POI names
    POI_MAP = {1: "POI1", 2: "POI2", 3: "POI3", 4: "POI4", 5: "POI5", 6: "POI6"}

    def __init__(self, model_assets: Dict[str, Any]):
        """
        Args:
            model_assets (dict): Dictionary containing paths to model weights.
        """
        self.model_assets = model_assets
        self._fill_classifier = None
        self._detectors = {}
    def _load_fill_cls(self):
        """Lazy loads the fill classifier using path from model_assets."""
        if self._fill_classifier is None:
            model_path = self.model_assets.get("fill_classifier")
            if model_path:
                Log.i(self.TAG, f"Loading Fill Classifier from {model_path}")
                self._fill_classifier = QModelV6YOLO_FillClassifier(model_path)
            else:
                Log.e(self.TAG, "No path provided for Fill Classifier.")
        return self._fill_classifier

    def _load_detector(self, detector_id: int):
        """Lazy loads the specific detector for the given channel count."""
        if detector_id not in self._detectors:
            detector_paths = self.model_assets.get("detectors", {})
            model_path = detector_paths.get(detector_id)

            if model_path:
                Log.i(self.TAG, f"Loading Detector for {detector_id} channels from {model_path}")
                try:
                    # Instantiate our new wrapper class
                    self._detectors[detector_id] = QModelV6YOLO_Detector(model_path)
                except Exception as e:
                    Log.e(self.TAG, f"Failed to load detector: {e}")
                    return None
            else:
                Log.e(self.TAG, f"No model path found for {detector_id} channel detector.")

        return self._detectors.get(detector_id)

    def _get_default_predictions(self) -> Dict[str, Dict[str, List]]:
        return {
            poi_name: {"indices": [-1], "confidences": [-1]}
            for poi_name in self.POI_MAP.values()
        }

    def _format_output(
        self, final_positions: Dict[int, int], confidence_scores: Dict[int, float]
    ) -> Dict[str, Dict[str, List[float]]]:
        """Format predictions into the standardized output dictionary."""
        output = {}
        for poi_num, poi_name in self.POI_MAP.items():
            if poi_num in final_positions:
                idx = final_positions[poi_num]
                conf = confidence_scores.get(poi_num, 0.0)
                output[poi_name] = {"indices": [idx], "confidences": [conf]}
            else:
                output[poi_name] = {"indices": [-1], "confidences": [-1]}
        return output

    def _reset_file_buffer(self, file_buffer: Union[str, object]) -> Union[str, object]:
        if isinstance(file_buffer, str):
            return file_buffer
        if hasattr(file_buffer, "seekable") and file_buffer.seekable():
            file_buffer.seek(0)
            return file_buffer
        raise ValueError("File buffer is not seekable.")

    def _validate_file_buffer(self, file_buffer: Union[str, object]) -> pd.DataFrame:
        try:
            file_buffer = self._reset_file_buffer(file_buffer)
            df = pd.read_csv(file_buffer)
        except Exception as e:
            raise ValueError(f"Failed to read data file: {e}")
        if df.empty:
            raise ValueError("The data file is empty.")
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
            processed_df = QModelV6YOLO_DataProcessor.preprocess_dataframe(df.copy())
            if processed_df is None or processed_df.empty:
                raise ValueError("Preprocessing failed")

            # --- 3. Classification ---
            if num_channels is None:
                fill_cls = self._load_fill_cls()
                if fill_cls:
                    if progress_signal:
                        progress_signal.emit(30, "Classifying Run Type...")
                    num_channels = fill_cls.predict(processed_df)
                else:
                    num_channels = 3
            num_channels = int(num_channels) if num_channels else 0
            if progress_signal:
                progress_signal.emit(50, f"Configured for {num_channels} Channels")

            # --- 4. Detection ---
            detector = self._load_detector(detector_id=num_channels)

            final_positions = {}
            confidences = {}

            if detector:
                if progress_signal:
                     progress_signal.emit(70, "Running Object Detection...")
                # CALL THE NEW DETECTOR LOGIC
                final_positions, confidences = detector.predict(processed_df, num_channels)
            else:
                Log.w(self.TAG, f"No detector found for ID {num_channels}")

            if progress_signal:
                progress_signal.emit(100, "Inference Complete")

            # --- 5. Formatting ---
            predictions = self._format_output(final_positions, confidences)
            return predictions, num_channels

        except Exception as e:
            Log.e(self.TAG, f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_predictions(), 0