import os

import cv2
import pandas as pd
from ultralytics import YOLO

try:
    from QATCH.common.logger import Logger as Log
    from QATCH.QModel.src.models.v6_yolo.v6_yolo_dataprocessor import (
        QModelV6YOLO_DataProcessor,
    )
except (ImportError, ModuleNotFoundError):

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


import os

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

try:
    from QATCH.common.logger import Logger as Log
    from QATCH.QModel.src.models.v6_yolo.v6_yolo_dataprocessor import (
        QModelV6YOLO_DataProcessor,
    )
except (ImportError, ModuleNotFoundError):
    if "QModelV6YOLO_DataProcessor" not in globals():

        class QModelV6YOLO_DataProcessor:
            @staticmethod
            def generate_fill_cls(df, img_h, img_w, scaling_limits=None):
                print("Warning: Using stub DataProcessor")
                return np.zeros((img_h * 3, img_w, 3), dtype=np.uint8)

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


class QModelV6YOLO_FillClassifier:
    """
    Handles the classification of the run state (e.g., no_fill, initial_fill, 1ch, 2ch, 3ch).
    Determines which detector (if any) should be run.
    """

    TAG = "QModelV6YOLO_FillClassifier"

    # --- Configuration ---
    INFERENCE_W = 224
    INFERENCE_H = 224
    GEN_W = 640
    GEN_H = 640
    # Returns 0 if no detection should occur yet.
    CLASS_MAP = {"no_fill": 0, "initial_fill": 0, "1ch": 1, "2ch": 2, "3ch": 3}

    def __init__(self, model_path: str):
        """
        Args:
            model_path: Absolute path to the .pt classification model file.
        """
        if not os.path.exists(model_path):
            Log.e(self.TAG, f"Model not found at: {model_path}")
            raise FileNotFoundError(f"Model not found at: {model_path}")

        try:
            # FIX: Windows OMP Error workaround
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            Log.i(self.TAG, f"Loading model from {model_path}...")
            self.model = YOLO(model_path)
        except Exception as e:
            Log.e(self.TAG, f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def predict(self, df: pd.DataFrame) -> int:
        """
        Generates the signal image from the dataframe and classifies the fill state.

        Returns:
            int: The number of channels to detect (0, 1, 2, or 3).
        """
        if df is None or df.empty:
            Log.w(self.TAG, "Dataframe provided for prediction is empty or None.")
            return 0

        # Generate Image
        # Note: img_h in generate_fill_cls is height per strip.
        # We pass GEN_H // 3 so the total stacked image is GEN_H.
        strip_height = self.GEN_H // 3

        try:
            img_high_res = QModelV6YOLO_DataProcessor.generate_fill_cls(
                df, img_h=strip_height, img_w=self.GEN_W
            )
        except Exception as e:
            Log.e(self.TAG, f"Error generating signal image: {e}")
            return 0

        if img_high_res is None:
            Log.w(self.TAG, "Generated image is None.")
            return 0

        # Resize for Inference (Model Specific Resolution)
        img_input = cv2.resize(
            img_high_res,
            (self.INFERENCE_W, self.INFERENCE_H),
            interpolation=cv2.INTER_AREA,
        )

        # Inference
        try:
            # verbose=False suppresses the YOLO console spam
            results = self.model(img_input, verbose=False)

            # Parse Results
            if not results:
                Log.w(self.TAG, "Model returned no results.")
                return 0

            probs = results[0].probs
            top1_index = probs.top1
            pred_label = results[0].names[top1_index]
            confidence = probs.top1conf.item()

            Log.d(self.TAG, f"Prediction: {pred_label} ({confidence:.1%})")

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
        """Maps the string label from YOLO to the integer channel count."""
        return self.CLASS_MAP.get(label, 0)
