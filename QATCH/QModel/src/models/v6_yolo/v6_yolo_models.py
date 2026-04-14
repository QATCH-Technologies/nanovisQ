# module: v6_yolo_models.py

"""QModel V6 YOLO pipeline — YOLO model wrappers.

``QModelV6YOLO_FillClassifier``
    Wraps a YOLO-cls model that maps a run snapshot to ``(num_channels, confidence)``.

``QModelV6YOLO_Detector``
    Wraps a YOLO-det model that maps a DataFrame slice to time-domain
    bounding-box detections.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2026-04-14
Version:
    7.0.0
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from QATCH.QModel.src.models.v6_yolo.v6_yolo_config import QModelV6Config, resolve_time_column
from QATCH.QModel.src.models.v6_yolo.v6_yolo_logging import Log, TAG_CLS, TAG_DET

try:
    from QATCH.QModel.src.models.v6_yolo.v6_yolo_dataprocessor import (
        QModelV6YOLO_DataProcessor,
    )
except (ImportError, ModuleNotFoundError):
    try:
        from v6_yolo.v6_yolo_dataprocessor import QModelV6YOLO_DataProcessor
    except ImportError:
        from v6_yolo_dataprocessor import QModelV6YOLO_DataProcessor  # type: ignore[no-redef]

try:
    from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]
except ImportError:
    Log.e(TAG_CLS, "'ultralytics' not found.  YOLO inference will fail.")


# ──────────────────────────────────────────────────────────────────────
# Fill classifier
# ──────────────────────────────────────────────────────────────────────


class QModelV6YOLO_FillClassifier:
    """Classifies the run state (no_fill, initial_fill, 1ch, 2ch, 3ch).

    Loads a YOLO classification model and maps its top-1 prediction to
    an integer channel count that drives the downstream detection cascade.
    """

    TAG = TAG_CLS

    def __init__(self, model_path: str) -> None:
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

    # ── Public API ────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> int:
        """Classify fill state and return channel count.  Confidence discarded."""
        channels, _ = self.predict_confidence(df)
        return channels

    def predict_confidence(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Classify fill state and return ``(num_channels, confidence)``.

        Used by ``FillReplaySimulator`` so the replay loop can apply a
        confidence gate.

        Returns:
            ``(0, 0.0)`` on any failure.
        """
        if df is None or df.empty:
            return 0, 0.0

        strip_height = QModelV6Config.FILL_GEN_H // 3
        try:
            img_high_res = QModelV6YOLO_DataProcessor.generate_fill_cls(
                df, img_h=strip_height, img_w=QModelV6Config.FILL_GEN_W
            )
        except Exception as e:
            Log.e(self.TAG, f"predict_confidence: image generation error: {e}")
            return 0, 0.0

        if img_high_res is None:
            return 0, 0.0

        img_input = cv2.resize(
            img_high_res,
            (QModelV6Config.FILL_INFERENCE_W, QModelV6Config.FILL_INFERENCE_H),
            interpolation=cv2.INTER_AREA,
        )

        try:
            results = self.model(img_input, verbose=False)
            if not results:
                return 0, 0.0

            probs = results[0].probs
            top1_index = probs.top1
            pred_label = results[0].names[top1_index]
            confidence = float(probs.top1conf.item())

            Log.d(self.TAG, f"Prediction: '{pred_label}' ({confidence:.1%})")
            if confidence < 0.5:
                Log.w(self.TAG, f"Low confidence ({confidence:.2f}) for class: {pred_label}")

            return self._map_label_to_channels(pred_label), confidence

        except Exception as e:
            Log.e(self.TAG, f"predict_confidence: inference error: {e}")
            return 0, 0.0

    # ── Internals ─────────────────────────────────────────────────────

    @staticmethod
    def _map_label_to_channels(label: str) -> int:
        label_clean = str(label).strip().lower()
        if label_clean in QModelV6Config.FILL_CLASS_MAP:
            return QModelV6Config.FILL_CLASS_MAP[label_clean]
        if label_clean.isdigit():
            return int(label_clean)
        Log.w(TAG_CLS, f"Unknown label '{label}'.  Defaulting to 0 channels.")
        return 0


# ──────────────────────────────────────────────────────────────────────
# Generic detector
# ──────────────────────────────────────────────────────────────────────


class QModelV6YOLO_Detector:
    """Generic wrapper for a single YOLO detector (Init, Ch1, Ch2, Ch3).

    Converts DataFrame slices to model-compatible images, runs inference,
    and maps normalised bounding-box x-coords back to the time domain.
    """

    TAG = TAG_DET

    def __init__(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Detector model not found: {model_path}")
        self.model = YOLO(model_path)

    def predict_single(
        self,
        df: pd.DataFrame,
        target_class_map: Optional[Dict[int, int]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Run inference on a data slice and return time-domain detections.

        Args:
            df: DataFrame slice.  Returns ``{}`` if shorter than
                ``MIN_SLICE_LENGTH``.
            target_class_map: Maps YOLO class IDs -> application POI IDs.

        Returns:
            ``{poi_id: {"time": float, "conf": float}}``
        """
        if df is None or len(df) < QModelV6Config.MIN_SLICE_LENGTH:
            return {}

        img_base = QModelV6YOLO_DataProcessor.generate_channel_det(
            df, img_w=QModelV6Config.IMG_WIDTH, img_h=QModelV6Config.IMG_HEIGHT
        )
        results = self.model(img_base, verbose=False, conf=QModelV6Config.CONF_THRESHOLD)

        col_time = resolve_time_column(df)
        time_vals = df[col_time].to_numpy(dtype=float)
        x_min, x_max = time_vals.min(), time_vals.max()

        best_dets: Dict[int, Dict[str, Any]] = {}
        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf.item()
                if cls_id not in best_dets or conf > best_dets[cls_id]["conf"]:
                    x_norm = box.xywhn[0][0].item()
                    t = x_norm * (x_max - x_min) + x_min
                    best_dets[cls_id] = {"time": t, "conf": conf}

        if not target_class_map:
            return best_dets

        return {
            poi_id: best_dets[yolo_id]
            for yolo_id, poi_id in target_class_map.items()
            if yolo_id in best_dets
        }
