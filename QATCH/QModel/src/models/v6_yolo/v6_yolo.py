import os
import traceback
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

# --- 1. Robust Imports ---
try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: 'ultralytics' not found. YOLO inference will fail.")

# Attempt to import internal modules, or fall back to stubs for standalone testing
try:
    from QATCH.common.logger import Logger as Log
    from QATCH.QModel.src.models.v6_yolo.v6_yolo_dataprocessor import (
        QModelV6YOLO_DataProcessor,
    )
    from QATCH.QModel.src.models.v6_yolo.v6_yolo_fill_cls import (
        QModelV6YOLO_FillClassifier,
    )
except (ImportError, ModuleNotFoundError):
    # --- Stub Classes for Standalone Context ---
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

    # Helper to generate the image if the main processor isn't loaded
    from src.visualization import generate_signal_image  # Assumes local src exists

    class QModelV6YOLO_DataProcessor:
        @staticmethod
        def preprocess_dataframe(df):
            # Basic stub or call your local preprocess
            return df

        @staticmethod
        def generate_channel_det(df, img_w=2560, img_h=384):
            # Wrapper for the visualization logic used in training
            return generate_signal_image(df, width=img_w, height=img_h)

    class QModelV6YOLO_FillClassifier:
        def __init__(self, path):
            pass

        def predict(self, df):
            return 3  # Stub default


class QModelV6YOLO_Detector:
    """
    Generic Wrapper for a single YOLO detector.
    Handles inference on a specific dataframe slice and maps results to App POI IDs.
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
            target_class_map: Dict mapping YOLO Class ID -> Legacy POI ID.

        Returns:
            Dict {poi_id: {'index': int, 'conf': float, 'time': float}}
        """
        if df is None or len(df) < 20:
            return {}

        # 1. Generate Image (Zoomed to the current slice)
        img_base = QModelV6YOLO_DataProcessor.generate_channel_det(
            df, img_w=2560, img_h=384
        )

        # 2. Inference (Low conf to catch candidates, similar to training/testing)
        # Using conf=0.01 to filter absolute noise but keep weak signals
        results = self.model(img_base, verbose=False, conf=0.01)

        # 3. Parse Results
        col_time = "Relative_time"
        # Safety check for column names
        if col_time not in df.columns:
            # Fallback for different CSV versions
            col_time = "time" if "time" in df.columns else df.columns[0]

        time_vals = df[col_time].values
        x_min, x_max = time_vals.min(), time_vals.max()

        best_dets = {}

        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf.item()

                # Strategy: Keep highest confidence per class
                if cls_id not in best_dets or conf > best_dets[cls_id]["conf"]:
                    x_norm = box.xywhn[0][0].item()

                    # Map Normalized X -> Real Time in this SLICE
                    t = x_norm * (x_max - x_min) + x_min

                    best_dets[cls_id] = {"time": t, "conf": conf}

        # 4. Map to Output format (YOLO Class -> App POI ID)
        final_results = {}
        if target_class_map:
            for yolo_id, poi_id in target_class_map.items():
                if yolo_id in best_dets:
                    data = best_dets[yolo_id]

                    # Find nearest index in the dataframe
                    # We use abs difference to find the closest sample index
                    idx_loc = (np.abs(time_vals - data["time"])).argmin()

                    # Get the REAL index from the dataframe (preserves original CSV row)
                    real_index = df.index[idx_loc]

                    final_results[poi_id] = {
                        "index": int(real_index),
                        "conf": data["conf"],
                        "time": data["time"],
                    }
        else:
            final_results = best_dets

        return final_results


class QModelV6YOLO:
    """
    Controller class for the QModel V6 YOLO pipeline.

    Architecture: Reverse Cascade
    1. Determine Num Channels
    2. Detect CH3 (POI6) -> Cut Data
    3. Detect CH2 (POI5) -> Cut Data
    4. Detect CH1 (POI4) -> Cut Data
    5. Detect Init Fill (POI1, POI2)
    """

    TAG = "QModelV6YOLO"

    # Legacy Output Map (POI3 is intentionally unused/placeholder)
    POI_MAP = {1: "POI1", 2: "POI2", 3: "POI3", 4: "POI4", 5: "POI5", 6: "POI6"}

    def __init__(self, model_assets: Dict[str, Any]):
        """
        Args:
            model_assets (dict): Paths to weights.
            Keys: 'fill_classifier', 'detectors': {'init', 'ch1', 'ch2', 'ch3'}
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
        """Lazy loads a specific detector by name."""
        if self._detectors.get(name) is None:
            detector_paths = self.model_assets.get("detectors", {})
            model_path = detector_paths.get(name)

            if model_path:
                Log.d(self.TAG, f"Loading Detector '{name}' from {model_path}")
                try:
                    self._detectors[name] = QModelV6YOLO_Detector(model_path)
                except Exception as e:
                    Log.e(self.TAG, f"Failed to load detector '{name}': {e}")
                    return None
            else:
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

        # Basic validation (adjust columns as needed for your specific CSVs)
        required = ["Relative_time"]
        if not any(col in df.columns for col in required):
            # Try fallback to 'time' or first column
            pass
        return df

    def _visualize(
        self,
        df: pd.DataFrame,
        results: Dict[int, Dict],
        cut_history: List[Tuple[str, float]],
        save_path: str = "debug_cascade.png",
    ):
        """
        Generates a debug plot showing the full signal, the detections,
        and the specific 'Cut Lines' used during the cascade.
        """
        import matplotlib.pyplot as plt

        # Ensure we have data
        if df is None or df.empty:
            return

        time = df["Relative_time"].values
        # Fallback if Dissipation is missing
        signal = (
            df["Dissipation"].values
            if "Dissipation" in df.columns
            else df.iloc[:, 1].values
        )

        plt.figure(figsize=(12, 6))

        # 1. Plot the Full Signal
        plt.plot(time, signal, color="gray", alpha=0.6, label="Raw Signal")

        # 2. Plot Detections
        # Map ID to Color
        colors = {
            1: "green",  # POI1
            2: "blue",  # POI2
            4: "orange",  # POI4 (CH1)
            5: "red",  # POI5 (CH2)
            6: "purple",  # POI6 (CH3)
        }

        for poi_id, data in results.items():
            if data and "time" in data:
                t = data["time"]
                c = colors.get(poi_id, "black")
                name = self.POI_MAP.get(poi_id, f"POI{poi_id}")

                plt.axvline(
                    x=t, color=c, linestyle="-", linewidth=2, label=f"{name} ({t:.2f}s)"
                )
                plt.text(
                    t,
                    np.max(signal),
                    f"{name}",
                    color=c,
                    rotation=90,
                    verticalalignment="top",
                )

        # 3. Plot Slicing History (The "Reverse Cascade")
        # We start with the earliest cut (last in history list) to show the progression
        for i, (stage_name, cut_time) in enumerate(cut_history):
            plt.axvline(x=cut_time, color="red", linestyle="--", linewidth=1, alpha=0.5)
            # Shade the area that was REMOVED by this cut
            plt.axvspan(cut_time, np.max(time), color="red", alpha=0.05)
            plt.text(
                cut_time,
                np.min(signal),
                f"Cut: {stage_name}",
                color="red",
                rotation=0,
                fontsize=8,
            )

        plt.title(f"Cascade Detection Debug - {len(cut_history)} Slices Applied")
        plt.xlabel("Time (s)")
        plt.ylabel("Signal")
        plt.legend(loc="upper right")
        plt.tight_layout()

        # Save or Show
        plt.savefig(save_path)
        Log.i(self.TAG, f"Debug visualization saved to {save_path}")
        plt.close()

    def predict(
        self,
        progress_signal: Any = None,
        file_buffer: Any = None,
        df: pd.DataFrame = None,
        visualize: bool = False,  # Set True to enable debug plot
        num_channels: int = None,
    ) -> Tuple[Dict[str, Dict[str, List]], int]:
        """
        Main entry point for prediction with Visualization support.
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
                    num_channels = 3

            num_channels = int(num_channels)
            if progress_signal:
                progress_signal.emit(40, f"Configured for {num_channels} Channels")

            # --- 4. Cascading Detection ---
            final_results = {}
            current_df = master_df.copy()
            col_time = (
                "Relative_time"
                if "Relative_time" in current_df.columns
                else current_df.columns[0]
            )

            # TRACKING FOR VISUALIZATION
            cut_history = []  # List of (StageName, Time)

            # === Step A: 3rd Channel (POI 6) ===
            if num_channels >= 3:
                det_ch3 = self._load_detector_by_name("ch3")
                if det_ch3:
                    res = det_ch3.predict_single(current_df, target_class_map={0: 6})
                    if 6 in res:
                        final_results[6] = res[6]
                        cut_time = res[6]["time"]

                        # Apply Slice
                        current_df = current_df[current_df[col_time] < cut_time]

                        # Log Slice
                        cut_history.append(("CH3_Cut", cut_time))
                        Log.d(self.TAG, f"Sliced at CH3: {cut_time:.4f}s")

            # === Step B: 2nd Channel (POI 5) ===
            if num_channels >= 2:
                det_ch2 = self._load_detector_by_name("ch2")
                if det_ch2:
                    res = det_ch2.predict_single(current_df, target_class_map={0: 5})
                    if 5 in res:
                        final_results[5] = res[5]
                        cut_time = res[5]["time"]

                        # Apply Slice
                        current_df = current_df[current_df[col_time] < cut_time]

                        # Log Slice
                        cut_history.append(("CH2_Cut", cut_time))
                        Log.d(self.TAG, f"Sliced at CH2: {cut_time:.4f}s")

            # === Step C: 1st Channel (POI 4) ===
            if num_channels >= 1:
                det_ch1 = self._load_detector_by_name("ch1")
                if det_ch1:
                    res = det_ch1.predict_single(current_df, target_class_map={0: 4})
                    if 4 in res:
                        final_results[4] = res[4]
                        cut_time = res[4]["time"]

                        # Apply Slice
                        current_df = current_df[current_df[col_time] < cut_time]

                        # Log Slice
                        cut_history.append(("CH1_Cut", cut_time))
                        Log.d(self.TAG, f"Sliced at CH1: {cut_time:.4f}s")

            # === Step D: Initial Fill (POI 1 & 2) ===
            det_init = self._load_detector_by_name("init")
            if det_init:
                res = det_init.predict_single(current_df, target_class_map={0: 1, 1: 2})
                final_results.update(res)

            if progress_signal:
                progress_signal.emit(100, "Inference Complete")

            # --- VISUALIZATION BLOCK ---
            if visualize:
                try:
                    self._visualize(master_df, final_results, cut_history)
                except Exception as e:
                    Log.w(self.TAG, f"Visualization failed: {e}")

            # --- 5. Formatting ---
            predictions = self._format_output(final_results)
            return predictions, num_channels

        except Exception as e:
            Log.e(self.TAG, f"Error during prediction: {e}")
            traceback.print_exc()
            return self._get_default_predictions(), 0
