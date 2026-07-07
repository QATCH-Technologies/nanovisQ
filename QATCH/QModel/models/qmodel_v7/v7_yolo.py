# module: v7_yolo.py

"""This module implements the QModel V7 YOLO pipeline for data analysis.

It orchestrates a "Reverse Cascading Detection" strategy, utilizing multiple YOLO object detectors to
identify points of interest (POIs) in viscosity data. The pipeline handles data preprocessing,
fill-type classification, and sequential slicing of the dataset to isolate specific channel events (Init, Ch1, Ch2, Ch3).

V7 adds, on top of the V6 cascade + configuration-prior decode:
- a v2 detection renderer (derivative-energy salience strip) via ``v7_render``, and
- a post-decode zoom-refinement stage (``_refine_with_zoom``) that re-detects
  each placed channel POI in a narrow window with a zoom-trained detector.

The interface mirrors V6: instantiate ``QModelV7(model_assets)`` and call
``.predict(...)`` -> ``(output_dict, num_channels)``. Class names use their
own V7 convention (``QModelV7``, ``QModelV7Config``, ``QModelV7Detector``,
``QModelV7FillClassifier``) so this package is distinct from, and can be
imported alongside, the V6 package.

Key Components:
- QModelV7: The main controller class.
- QModelV7Detector: A wrapper for individual YOLO model instances.
- QModelV7Config: Configuration constants for the pipeline.

Dependencies:
- ultralytics (YOLO)
- pandas, numpy, matplotlib
- QATCH internal modules (Logger, DataProcessor, FillClassifier)

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-07-06

Version:
    7.0.0
"""

import datetime
import os
import time
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


except (ImportError, ModuleNotFoundError):

    class Log:
        @staticmethod
        def d(tag: str, msg: str):
            print(f"{tag} [DEBUG] {msg}")

        @staticmethod
        def i(tag: str, msg: str):
            print(f"{tag} [INFO] {msg}")

        @staticmethod
        def w(tag: str, msg: str):
            print(f"{tag} [WARNING] {msg}")

        @staticmethod
        def e(tag: str, msg: str):
            print(f"{tag} [ERROR] {msg}")

    Log.i(tag="[HEADLESS OPERATION]", msg="Running...")

# Data processor (v7). Same class name as v6 (QModelV7DataProcessor);
# the v7 file only adds a headless Log fallback and defaulted baseline args.
try:
    from QATCH.QModel.models.qmodel_v7.v7_yolo_dataprocessor import (
        QModelV7DataProcessor,
    )
except (ImportError, ModuleNotFoundError):
    from v7_yolo_dataprocessor import QModelV7DataProcessor

try:
    # New project requirement as of 2026-01-12
    from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]
except ImportError:
    Log.e(
        tag="[QModelV7]",
        msg="'ultralytics' not found. YOLO inference will fail.",
    )

# Configuration-prior decode layer (optional). When unavailable the
# decode_config path in QModelV7.predict degrades to a no-op and the
# pipeline behaves exactly as before.
try:
    from QATCH.QModel.models.qmodel_v7.v7_decode import (
        Candidate,
        dp_decode,
        score_configuration,
    )
    from QATCH.QModel.models.qmodel_v7.v7_spacing_prior import SpacingPrior

    _DECODE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    try:
        from QATCH.QModel.models.qmodel_v7.v7_decode import (
            Candidate,
            dp_decode,
            score_configuration,
        )
        from QATCH.QModel.models.qmodel_v7.v7_spacing_prior import SpacingPrior

        _DECODE_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        _DECODE_AVAILABLE = False
        Log.e(
            tag="[QModelV7]",
            msg="Decode modules not found. Configuration decode disabled; falling back to cascade-only behavior.",
        )


# Version-2 detection renderer (optional). Falls back to the v1 renderer
# when unavailable so older deployments are unaffected.
try:
    from QATCH.QModel.models.qmodel_v7.v7_render import (
        generate_det_image as _gen_det_image,
    )

    _RENDER_V2_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    try:
        from v7_render import generate_det_image as _gen_det_image

        _RENDER_V2_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        _RENDER_V2_AVAILABLE = False
        Log.w(
            tag="[QModelV7]",
            msg="v7_render not found. RENDER_VERSION=2 will fall back to the v1 detection renderer.",
        )


# --- Configuration Constants ---
class QModelV7Config:
    """Configuration constants for the QModel V7 YOLO pipeline."""

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

    # --- Configuration-Prior Decode Settings ---
    # Weight of the spacing log-likelihood relative to detection confidence.
    # Scalar, or set DECODE_LAMBDA_PAIRS to weight edges individually — the
    # prior's value is not uniform across the chain: on sharp well-detected
    # events (POI2->POI3) a broad gap prior mostly drags correct detections,
    # while on ambiguous late events it is the main defence. Sweep with
    # sweep_decode.py --edge3-scales; do not hand-pick.
    DECODE_LAMBDA: float = 0.25
    # e.g. {"POI2->POI3": 0.5} — unlisted pairs default to DECODE_LAMBDA.
    DECODE_LAMBDA_PAIRS: Optional[Dict[str, float]] = {
        "POI1->POI2": 0.0,  # analytic init exclusion (unchanged)
        "POI2->POI3": 0.125,  # edge3_scale 0.5 x base lambda
        "POI3->POI4": 0.125,
    }
    # Weight on summed (clipped) detection confidence.
    DECODE_CONF_WEIGHT: float = 1.0
    # Multiplicative slack on the learned hard gap bounds.
    DECODE_FEAS_SLACK: float = 1.5
    # Per-POI cap (top-K by confidence) on the decode lattice width.
    DECODE_MAX_CANDIDATES: int = 10
    # --- Zoom Refinement Settings ---
    # Post-decode refinement: re-render a window around each decoded channel
    # POI and re-detect with a zoom-trained detector. Targets the 1-5 s
    # localization band (candidates exist near truth but the full-run render
    # is too coarse for slow transitions). No-ops when zoom detector assets
    # are absent.
    REFINE_WINDOW_S: float = 24.0
    REFINE_MIN_CONF: float = 0.20
    # Maximum move the refiner may apply, as a fraction of the window; moves
    # larger than this indicate the refiner latched onto a different event.
    REFINE_MAX_SHIFT_FRAC: float = 0.45

    # Detection-image renderer version. MUST match the render the deployed
    # detector weights were trained on: 1 = legacy (diss/freq/difference
    # strips), 2 = v7 (diss/freq/derivative-energy salience strips, from
    # v7_render). Weights and render version ship together.
    RENDER_VERSION: int = 2

    # Hysteresis ("the decode must earn the move"): the decoded configuration
    # is only accepted if its score beats the cascade configuration's score —
    # under the SAME objective — by at least this margin. 0.0 disables the
    # guard (always accept the decode optimum). Raising it trades a few
    # missed fixes for fewer regressions on runs where the cascade was
    # already right; tune it with sweep_decode.py, not by hand.
    DECODE_MIN_MARGIN: float = 0.25

    # Progress Signal Steps
    PROG_LOAD_DATA: int = 10
    PROG_CLASSIFY: int = 30
    PROG_CONFIG: int = 40
    PROG_COMPLETE: int = 100


class QModelV7FillClassifier:
    """
    Handles the classification of the run state (e.g., no_fill, initial_fill, 1ch, 2ch, 3ch).

    This class loads a specific YOLO classification model to analyze the raw sensor data
    visuals and determine which "Fill State" the run belongs to. This classification
    dictates how many channels (if any) the subsequent detection pipeline should search for.
    """

    TAG = "[QModelV7FillClassifier]"

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
        strip_height = QModelV7Config.FILL_GEN_H // 3

        try:
            img_high_res = QModelV7DataProcessor.generate_fill_cls(
                df, img_h=strip_height, img_w=QModelV7Config.FILL_GEN_W
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
            (QModelV7Config.FILL_INFERENCE_W, QModelV7Config.FILL_INFERENCE_H),
            interpolation=cv2.INTER_AREA,
        )
        self._last_image = img_input
        # # --- Debugging for live fill frames and type cls ---
        # debug_dir = os.path.join(os.getcwd(), "debug_frames")
        # os.makedirs(debug_dir, exist_ok=True)

        # # Use nanoseconds to ensure unique filenames in a fast loop
        # timestamp = time.time_ns()
        # save_path = os.path.join(debug_dir, f"input_{timestamp}.png")

        # cv2.imwrite(save_path, img_input)
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

            # Log.d(self.TAG, f"Prediction: '{pred_label}' ({confidence:.1%})")

            # if confidence < 0.5:
            #     Log.w(
            #         self.TAG,
            #         f"Low confidence ({confidence:.2f}) for class: {pred_label}",
            #     )

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

        if label_clean in QModelV7Config.FILL_CLASS_MAP:
            return QModelV7Config.FILL_CLASS_MAP[label_clean]
        if label_clean.isdigit():
            return int(label_clean)
        Log.w(self.TAG, f"Unknown label '{label}'. Defaulting to 0 channels.")
        return 0


class QModelV7Detector:
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
        if df is None or len(df) < QModelV7Config.MIN_SLICE_LENGTH:
            return {}
        if QModelV7Config.RENDER_VERSION >= 2 and _RENDER_V2_AVAILABLE:
            img_base = _gen_det_image(
                df,
                QModelV7Config.IMG_WIDTH,
                QModelV7Config.IMG_HEIGHT,
                version=QModelV7Config.RENDER_VERSION,
            )
        else:
            img_base = QModelV7DataProcessor.generate_channel_det(
                df, img_w=QModelV7Config.IMG_WIDTH, img_h=QModelV7Config.IMG_HEIGHT
            )
        results = self.model(img_base, verbose=False, conf=QModelV7Config.CONF_THRESHOLD)
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

    def predict_candidates(
        self, df: pd.DataFrame, target_class_map: Optional[Dict[int, int]] = None
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Runs inference and returns ALL candidate detections per class, not just
        the single highest-confidence box.

        This is the candidate-harvesting counterpart to `predict_single`. It is
        used by the configuration-prior decode layer, which chooses the jointly
        coherent configuration across stages rather than greedily taking the
        most confident box per stage. Image generation, time mapping and the
        confidence threshold are identical to `predict_single`, so the
        candidate set here is a superset that always contains `predict_single`'s
        pick (the max-confidence one).

        Args:
            df (pd.DataFrame): The dataframe slice to analyze. Slices smaller
                than `MIN_SLICE_LENGTH` yield an empty result.
            target_class_map (Optional[Dict[int, int]]): Mapping from the YOLO
                model's internal class IDs to application POI IDs. If provided,
                output is keyed by POI ID; otherwise by raw YOLO class ID.

        Returns:
            Dict[int, List[Dict[str, Any]]]: Mapping of class/POI id to a list
            of candidate dicts, each {"time": float, "conf": float}, sorted by
            descending confidence. Classes with no detections are omitted.
        """
        if df is None or len(df) < QModelV7Config.MIN_SLICE_LENGTH:
            return {}
        if QModelV7Config.RENDER_VERSION >= 2 and _RENDER_V2_AVAILABLE:
            img_base = _gen_det_image(
                df,
                QModelV7Config.IMG_WIDTH,
                QModelV7Config.IMG_HEIGHT,
                version=QModelV7Config.RENDER_VERSION,
            )
        else:
            img_base = QModelV7DataProcessor.generate_channel_det(
                df, img_w=QModelV7Config.IMG_WIDTH, img_h=QModelV7Config.IMG_HEIGHT
            )
        results = self.model(img_base, verbose=False, conf=QModelV7Config.CONF_THRESHOLD)
        col_time = "Relative_time"
        if col_time not in df.columns:
            col_time = "time" if "time" in df.columns else df.columns[0]

        time_vals = df[col_time].to_numpy(dtype=float)
        x_min, x_max = time_vals.min(), time_vals.max()

        all_dets: Dict[int, List[Dict[str, Any]]] = {}
        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf.item()
                x_norm = box.xywhn[0][0].item()
                t = x_norm * (x_max - x_min) + x_min
                all_dets.setdefault(cls_id, []).append({"time": t, "conf": conf})

        # Sort each class's candidates by descending confidence.
        for cls_id in all_dets:
            all_dets[cls_id].sort(key=lambda d: d["conf"], reverse=True)

        if not target_class_map:
            return all_dets

        mapped: Dict[int, List[Dict[str, Any]]] = {}
        for yolo_id, poi_id in target_class_map.items():
            if yolo_id in all_dets:
                mapped[poi_id] = all_dets[yolo_id]
        return mapped


class QModelV7:
    """
    Controller class for the QModel V6 YOLO .

    This class manages the various machine YOLO models used in the
    V6 pipeline. It handles the lazy loading of the fill classifier and specific channel
    detectors (Init, Ch1, Ch2, Ch3) to optimize memory usage, ensuring models are only
    loaded when required by the prediction logic.
    """

    TAG = "QModelV7"

    # Maps internal integer Class IDs to application-standard POI strings
    POI_MAP = {1: "POI1", 2: "POI2", 3: "POI3", 4: "POI4", 5: "POI5", 6: "POI6"}

    # Decoder name space (spacing_prior.POI_ORDER) excludes the legacy id-3
    # shim, so production ids map onto a dense 5-POI chain:
    #   id 1 -> POI1, id 2 -> POI2, id 4 -> POI3, id 5 -> POI4, id 6 -> POI5.
    # NOTE: these names intentionally differ from POI_MAP's output names; the
    # decode layer works internally in chain space and results are mapped back
    # to production ids before formatting.
    DECODE_ID_TO_NAME = {1: "POI1", 2: "POI2", 4: "POI3", 5: "POI4", 6: "POI5"}
    DECODE_NAME_TO_ID = {v: k for k, v in DECODE_ID_TO_NAME.items()}

    def __init__(self, model_assets: Dict[str, Any]):
        """
        Initializes the QModelV7 controller.

        Args:
            model_assets (Dict[str, Any]): A dictionary containing paths to model weights.
                Expected structure:
                {
                    "fill_classifier": "path/to/classifier.pt",
                    "detectors": {
                        "init": "path/to/init.pt",
                        "ch1": "path/to/ch1.pt",
                        "poi5_fine": "path/to/poi5_fine.pt", # Deprecated
                        # ... etc
                    }
                }
        """
        self.model_assets = model_assets
        self._fill_classifier = None
        self._spacing_prior = None
        self._detectors: Dict[str, Any] = {
            "init": None,
            "ch1": None,
            "ch2": None,
            "ch3": None,
            # Optional post-decode zoom refiners (see _refine_with_zoom).
            "ch1_zoom": None,
            "ch2_zoom": None,
            "ch3_zoom": None,
        }

    def _load_fill_cls(self) -> Any:
        """
        Lazy loads the Fill Classifier model.

        Checks if the classifier is already loaded; if not, attempts to load it using
        the path provided in `model_assets`.

        Returns:
            Any: The loaded `QModelV7FillClassifier` instance, or None if loading failed
            or no path was provided.
        """
        if self._fill_classifier is None:
            model_path = self.model_assets.get("fill_classifier")
            if model_path:
                self._fill_classifier = QModelV7FillClassifier(model_path)
        return self._fill_classifier

    def _load_detector_by_name(self, name: str) -> Any:
        """
        Lazy loads a specific YOLO detector by its shorthand name.

        Args:
            name (str): The key for the detector to load (e.g., "init", "ch1", "ch2", "ch3").

        Returns:
            Any: The loaded `QModelV7Detector` instance, or None if the path is missing
            or loading fails.
        """
        if self._detectors.get(name) is None:
            detector_paths = self.model_assets.get("detectors", {})
            model_path = detector_paths.get(name)
            if model_path:
                try:
                    self._detectors[name] = QModelV7Detector(model_path)
                except Exception as e:
                    Log.e(self.TAG, f"Error while loading detector '{name}': {e}")
                    return None
        return self._detectors.get(name)

    def _load_spacing_prior(self) -> Any:
        """
        Lazy loads the learned SpacingPrior used by the configuration decode.

        The path is taken from `model_assets["spacing_prior"]` (a JSON file
        produced by fit_prior.py). Returns None — and the decode path becomes
        a no-op — if the decode modules or the prior file are unavailable, so
        enabling `decode_config` can never break a deployment that lacks the
        asset.

        Returns:
            Any: The loaded `SpacingPrior` instance, or None.
        """
        if not _DECODE_AVAILABLE:
            return None
        if self._spacing_prior is None:
            prior_path = self.model_assets.get("spacing_prior")
            if prior_path and os.path.exists(prior_path):
                try:
                    self._spacing_prior = SpacingPrior.load(prior_path)
                except Exception as e:
                    Log.e(self.TAG, f"Error while loading spacing prior: {e}")
                    return None
            else:
                Log.w(self.TAG, "Spacing prior asset missing; decode_config disabled.")
        return self._spacing_prior

    def _decode_with_prior(
        self,
        final_results: Dict[int, Dict[str, Any]],
        harvested: Dict[int, List[Dict[str, Any]]],
        num_channels: int,
        raw_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Runs the joint configuration decode over harvested candidates and
        overwrites the cascade's greedy placements with the decoded ones.

        The decode works in chain space (DECODE_ID_TO_NAME). Present POIs are
        gated by `num_channels` exactly as the cascade gates its stages, so the
        decoder can never introduce a POI the fill classifier said is absent.
        The cascade's greedy picks are injected into the candidate pools
        defensively, so the decoder's solution space always contains current
        production behaviour: with `require_feasible` fallback inside
        dp_decode, the decode is never worse than greedy.

        Args:
            final_results: Cascade results keyed by production POI id; mutated
                in place with decoded placements.
            harvested: Candidate store keyed by production POI id, values are
                lists of {"time", "conf", "index"} dicts.
            num_channels: Channel count from the fill classifier (or enforced).
            raw_df: Original DataFrame, for time -> raw index resolution.

        Returns:
            Dict[str, Any]: Diagnostics for the `_decode` output key:
                used / reason, feasible, fallback, total_score,
                spacing_loglik, present, changed (chain-space names whose
                placement moved vs. the cascade), and greedy_times.
        """
        if not _DECODE_AVAILABLE:
            return {"used": False, "reason": "decode modules unavailable"}
        prior = self._load_spacing_prior()
        if prior is None:
            return {"used": False, "reason": "spacing prior unavailable"}

        present = ["POI1", "POI2"]
        if num_channels >= 1:
            present.append("POI3")
        if num_channels >= 2:
            present.append("POI4")
        if num_channels >= 3:
            present.append("POI5")

        cands: Dict[str, List[Candidate]] = {}
        for poi_id, name in self.DECODE_ID_TO_NAME.items():
            pool = [
                Candidate(time=float(d["time"]), conf=float(d["conf"]))
                for d in harvested.get(poi_id, [])
            ]
            greedy_pick = final_results.get(poi_id)
            if greedy_pick is not None and not any(
                abs(c.time - float(greedy_pick["time"])) < 1e-9 for c in pool
            ):
                pool.append(
                    Candidate(time=float(greedy_pick["time"]), conf=float(greedy_pick["conf"]))
                )
            if pool:
                cands[name] = pool

        if not any(name in cands for name in present):
            return {"used": False, "reason": "no candidates harvested"}

        greedy_times = {name: max(cs, key=lambda c: c.conf).time for name, cs in cands.items()}

        # Snapshot the cascade's (pre-decode) placements in chain space so a
        # single predict() call carries both A/B arms: callers (e.g. the
        # decode benchmark) can compare production-greedy vs decoded without
        # a second YOLO pass.
        cascade_snapshot = {
            name: {
                "time": float(final_results[poi_id]["time"]),
                "index": int(final_results[poi_id]["index"]),
                "conf": float(final_results[poi_id]["conf"]),
            }
            for poi_id, name in self.DECODE_ID_TO_NAME.items()
            if poi_id in final_results
        }

        try:
            lam_eff: Any = QModelV7Config.DECODE_LAMBDA
            if QModelV7Config.DECODE_LAMBDA_PAIRS:
                base = float(QModelV7Config.DECODE_LAMBDA)
                lam_eff = {p: QModelV7Config.DECODE_LAMBDA_PAIRS.get(p, base) for p in prior.pairs}
            result = dp_decode(
                cands,
                present,
                prior,
                lam=QModelV7Config.DECODE_LAMBDA,
                conf_weight=QModelV7Config.DECODE_CONF_WEIGHT,
                feas_slack=QModelV7Config.DECODE_FEAS_SLACK,
                max_candidates=QModelV7Config.DECODE_MAX_CANDIDATES,
            )
        except Exception as e:
            Log.e(self.TAG, f"Configuration decode failed: {e}")
            return {"used": False, "reason": f"decode error: {e}"}

        # ---- accept-margin (hysteresis). Score the cascade's own
        # configuration under the decode objective; only move off it when the
        # decoded configuration wins by DECODE_MIN_MARGIN. Skipped when the
        # decode places POIs the cascade missed (scores over different POI
        # sets are not comparable, and recovering a missed POI is always
        # worth taking).
        kept_cascade = False
        cascade_chosen = {
            name: Candidate(time=float(rec["time"]), conf=float(rec["conf"]))
            for name, rec in cascade_snapshot.items()
            if name in present
        }
        margin = QModelV7Config.DECODE_MIN_MARGIN
        if margin > 0 and result.chosen and set(result.chosen.keys()) == set(cascade_chosen.keys()):
            cascade_score = score_configuration(
                cascade_chosen,
                prior,
                lam=lam_eff,
                conf_weight=QModelV7Config.DECODE_CONF_WEIGHT,
            )
            if result.total_score < cascade_score + margin:
                kept_cascade = True

        changed: List[str] = []
        if kept_cascade:
            return {
                "used": True,
                "feasible": result.feasible,
                "fallback": result.fallback_used,
                "total_score": result.total_score,
                "spacing_loglik": result.spacing_loglik,
                "present": present,
                "changed": [],
                "kept_cascade": True,
                "greedy_times": greedy_times,
                "cascade": cascade_snapshot,
            }
        for name, cand in result.chosen.items():
            poi_id = self.DECODE_NAME_TO_ID[name]
            prev = final_results.get(poi_id)
            if prev is None or abs(float(prev["time"]) - cand.time) > 1e-9:
                changed.append(name)
            final_results[poi_id] = {
                "index": self._get_raw_index(raw_df, cand.time),
                "conf": cand.conf,
                "time": cand.time,
            }
        if changed:
            Log.d(self.TAG, f"Config decode moved {len(changed)} POI(s): {changed}")

        return {
            "used": True,
            "feasible": result.feasible,
            "fallback": result.fallback_used,
            "total_score": result.total_score,
            "spacing_loglik": result.spacing_loglik,
            "present": present,
            "changed": changed,
            "kept_cascade": False,
            "greedy_times": greedy_times,
            "cascade": cascade_snapshot,
        }

    # POI id -> zoom detector asset name.
    ZOOM_REFINE_MAP = {4: "ch1_zoom", 5: "ch2_zoom", 6: "ch3_zoom"}

    def _refine_with_zoom(
        self,
        final_results: Dict[int, Dict[str, Any]],
        master_df: pd.DataFrame,
        raw_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Post-decode zoom refinement of the channel POIs.

        For each placed channel POI, slices a REFINE_WINDOW_S window of the
        preprocessed signal around the decoded time, renders it with the
        production renderer (the zoom detectors are trained on exactly this
        window distribution by build_dataset.py), re-detects, and accepts
        the refined position when it is confident and within the window's
        trust region. Mutates final_results in place; returns diagnostics.
        """
        meta: Dict[str, Any] = {"used": False, "moved": {}}
        if not any(
            self.model_assets.get("detectors", {}).get(n) for n in self.ZOOM_REFINE_MAP.values()
        ):
            meta["reason"] = "no zoom detector assets"
            return meta
        col_time = "Relative_time" if "Relative_time" in master_df.columns else master_df.columns[0]
        tv = master_df[col_time].to_numpy(dtype=float)
        t_lo, t_hi = float(tv.min()), float(tv.max())
        half_w = QModelV7Config.REFINE_WINDOW_S / 2.0
        for poi_id, det_name in self.ZOOM_REFINE_MAP.items():
            rec = final_results.get(poi_id)
            if rec is None or rec.get("index", -1) < 0:
                continue
            detector = self._load_detector_by_name(det_name)
            if detector is None:
                continue
            t_c = float(rec["time"])
            w0, w1 = max(t_lo, t_c - half_w), min(t_hi, t_c + half_w)
            if w1 - w0 < 4.0:
                continue
            sl = master_df[(master_df[col_time] >= w0) & (master_df[col_time] < w1)]
            if len(sl) < QModelV7Config.MIN_SLICE_LENGTH:
                continue
            try:
                res = detector.predict_single(sl, target_class_map={0: poi_id})
            except Exception as e:
                Log.w(self.TAG, f"Zoom refine failed for {det_name}: {e}")
                continue
            det = res.get(poi_id) if res else None
            if not det or det.get("conf", 0.0) < QModelV7Config.REFINE_MIN_CONF:
                continue
            t_new = float(det["time"])
            if abs(t_new - t_c) > QModelV7Config.REFINE_MAX_SHIFT_FRAC * (w1 - w0):
                continue  # latched onto a different event; keep decode pick
            ordered_ids = [1, 2, 4, 5, 6]
            pos = ordered_ids.index(poi_id)
            prev_t = next(
                (
                    float(final_results[p]["time"])
                    for p in reversed(ordered_ids[:pos])
                    if p in final_results and "time" in final_results[p]
                ),
                None,
            )
            next_t = next(
                (
                    float(final_results[p]["time"])
                    for p in ordered_ids[pos + 1 :]
                    if p in final_results and "time" in final_results[p]
                ),
                None,
            )
            if (prev_t is not None and t_new <= prev_t) or (next_t is not None and t_new >= next_t):
                continue
            meta["used"] = True
            meta["moved"][self.POI_MAP[poi_id]] = {
                "from": t_c,
                "to": t_new,
                "delta_s": t_new - t_c,
                "conf": float(det["conf"]),
            }
            final_results[poi_id] = {
                "index": self._get_raw_index(raw_df, t_new),
                "conf": float(det["conf"]),
                "time": t_new,
            }
        return meta

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
            poi_name: {"indices": [-1], "confidences": [-1]} for poi_name in self.POI_MAP.values()
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

        # UPDATED: Added %f for microseconds to prevent overwrites in fast loops
        # Format example: _20231027_153045_123456
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        base_name, ext = os.path.splitext(save_path)
        final_save_path = f"{base_name}_{timestamp}{ext}"

        time = df["Relative_time"].values
        signal = df["Dissipation"].values if "Dissipation" in df.columns else df.iloc[:, 1].values

        plt.figure(figsize=(12, 6))
        plt.plot(time, signal, color="gray", alpha=0.6, label="Raw Signal")

        colors = {1: "green", 2: "blue", 4: "orange", 5: "red", 6: "purple"}
        for poi_id, data in results.items():
            if data and "time" in data:
                t = data["time"]
                c = colors.get(poi_id, "black")
                name = getattr(self, "POI_MAP", {}).get(poi_id, f"POI{poi_id}")
                plt.axvline(x=t, color=c, linestyle="-", linewidth=2, label=f"{name}")

        for _, (_, cut_time) in enumerate(cut_history):
            plt.axvline(x=cut_time, color="red", linestyle="--", linewidth=1, alpha=0.5)
            plt.axvspan(cut_time, np.max(time), color="red", alpha=0.05)

        plt.title(f"Cascade Detection Debug - {len(cut_history)} Slices Applied")
        plt.savefig(final_save_path)
        plt.close()
        # Optional: Print where it saved for easier tracking
        Log.i(self.TAG, f"Debug plot saved to: {final_save_path}")

    def predict(
        self,
        progress_signal: Any = None,
        file_buffer: Any = None,
        df: pd.DataFrame | None = None,
        visualize: bool = False,
        num_channels: int | None = None,
        avg_res_freq: Optional[float] = None,
        avg_diss: Optional[float] = None,
        harvest_candidates: bool = False,
        decode_config: bool = False,
        refine_pois: bool = True,
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
            avg_res_freq (Optional[float]): Pre-computed baseline mean of
                `Resonance_Frequency` from the pre-fill window. When provided together
                with `avg_diss`, the baseline window search inside `preprocess_dataframe`
                is bypassed. This is required for live inference once the rolling buffer
                has trimmed past the baseline window and the early reference data is no
                longer present in the DataFrame.
            avg_diss (Optional[float]): Pre-computed baseline mean of `Dissipation` from
                the pre-fill window. See `avg_res_freq` for details.
            harvest_candidates (bool, optional): If True, additionally harvest ALL
                candidate detections per stage (not just the greedy best) from the
                same in-distribution slice each detector sees, and attach them to
                the output dict under the reserved key "_candidates" as
                {POI_NAME: [{"time","conf","index"}, ...]}. This does NOT change
                cuts or predictions — the cascade proceeds on the greedy pick
                exactly as in production; harvesting only observes the runners-up
                for the downstream configuration-prior decode. Defaults to False.
            decode_config (bool, optional): If True (implies harvesting), runs the
                joint configuration decode (dp_decode + SpacingPrior) over the
                harvested candidates after the cascade completes, and REPLACES the
                greedy placements with the globally-coherent configuration. The
                decoder is gated by `num_channels` (it cannot add absent POIs) and
                falls back to the greedy picks when no feasible joint path exists,
                so it is never worse than current behaviour. Also widens the
                harvest slices (see in-line comments) so an early greedy cut
                cannot exclude true downstream candidates. Diagnostics attach to
                the output under the reserved key "_decode". Requires
                `model_assets["spacing_prior"]` to point at a fitted prior JSON;
                no-ops with a warning otherwise. Defaults to False.

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

            master_df = QModelV7DataProcessor.preprocess_dataframe(
                raw_df.copy(),
                baseline_freq=avg_res_freq,
                baseline_diss=avg_diss,
            )

            if progress_signal:
                progress_signal.emit(20, "Preprocessing Data...")

            if master_df is None or master_df.empty:
                raise ValueError("Preprocessing failed")

            if num_channels is None:
                if progress_signal:
                    progress_signal.emit(30, "Determining Channel Count...")

                fill_cls = self._load_fill_cls()
                num_channels = int(fill_cls.predict(master_df)) if fill_cls else 3

            if num_channels == -1:
                if progress_signal:
                    progress_signal.emit(100, "No channels detected!")
                return self._get_default_predictions(), num_channels

            final_results = {}
            current_df = master_df.copy()
            col_time = (
                "Relative_time" if "Relative_time" in current_df.columns else current_df.columns[0]
            )
            cut_history = []
            # decode_config consumes the harvest, so it implies harvesting.
            harvest_candidates = harvest_candidates or decode_config
            # Candidate harvest store: poi_id -> list of {"time","conf","index"}.
            # Populated only when harvest_candidates=True. Harvesting does NOT
            # change cuts or predictions: the cascade still proceeds on
            # predict_single's greedy pick exactly as in production.
            harvested: Dict[int, List[Dict[str, Any]]] = {}
            # The harvest runs on a PARALLEL slice chain (`harvest_df`) that is
            # cut at the LATEST harvested candidate of each stage instead of
            # the greedy pick. Rationale: if the greedy pick is too early, the
            # production cut excises the true downstream event before its
            # detector ever sees it — a candidate that is never generated can
            # never be recovered by the decoder, silently capping oracle
            # recall. Cutting at the latest candidate keeps the harvest slice
            # a superset of the production slice (slightly wider than the
            # training distribution in the worst case) while guaranteeing the
            # true event region survives whenever ANY candidate covers it.
            harvest_df = master_df.copy() if harvest_candidates else None

            def harvest_stage(detector, slice_df, class_map):
                """Harvest all candidates for one stage from `slice_df`.
                Returns the latest harvested candidate time (used to advance
                the conservative harvest cut chain), or None."""
                if not harvest_candidates or detector is None or slice_df is None:
                    return None
                latest: Optional[float] = None
                try:
                    cands = detector.predict_candidates(slice_df, target_class_map=class_map)
                except Exception as exc:
                    Log.w(self.TAG, f"Candidate harvest failed: {exc}")
                    return None
                for poi_id, lst in cands.items():
                    out_lst = []
                    for d in lst:
                        out_lst.append(
                            {
                                "time": d["time"],
                                "conf": d["conf"],
                                "index": self._get_raw_index(raw_df, d["time"]),
                            }
                        )
                        if latest is None or d["time"] > latest:
                            latest = d["time"]
                    # If a stage is revisited (e.g. POI6 coarse + fine), keep the
                    # union; later refinement appends rather than overwrites.
                    harvested.setdefault(poi_id, []).extend(out_lst)
                return latest

            def advance_harvest_cut(greedy_cut, latest_cand):
                """Advance the harvest slice chain past this stage. The cut is
                the LATEST of the production cut and the latest harvested
                candidate, so the harvest slice never loses signal that any
                candidate still claims."""
                nonlocal harvest_df
                if harvest_df is None:
                    return
                cuts = [t for t in (greedy_cut, latest_cand) if t is not None]
                if cuts:
                    harvest_df = harvest_df[harvest_df[col_time] < max(cuts)]

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

            if num_channels >= 3:
                if progress_signal:
                    progress_signal.emit(45, "Detecting Channel 3...")

                det_ch3 = self._load_detector_by_name("ch3")
                if det_ch3:
                    res = det_ch3.predict_single(current_df, target_class_map={0: 6})
                    h_latest = harvest_stage(det_ch3, harvest_df, {0: 6})
                    cut_time = process_detection(res, 6)
                    if cut_time:
                        current_df = current_df[current_df[col_time] < cut_time]
                        cut_history.append(("CH3_Cut", cut_time))
                    advance_harvest_cut(cut_time, h_latest)

            if num_channels >= 2:
                if progress_signal:
                    progress_signal.emit(60, "Detecting Channel 2...")

                det_ch2 = self._load_detector_by_name("ch2")
                if det_ch2:
                    res = det_ch2.predict_single(current_df, target_class_map={0: 5})
                    h_latest = harvest_stage(det_ch2, harvest_df, {0: 5})
                    cut_time = process_detection(res, 5)
                    if cut_time:
                        current_df = current_df[current_df[col_time] < cut_time]
                        cut_history.append(("CH2_Cut", cut_time))
                    advance_harvest_cut(cut_time, h_latest)

            if num_channels >= 1:
                if progress_signal:
                    progress_signal.emit(75, "Detecting Channel 1...")

                det_ch1 = self._load_detector_by_name("ch1")
                if det_ch1:
                    res = det_ch1.predict_single(current_df, target_class_map={0: 4})
                    h_latest = harvest_stage(det_ch1, harvest_df, {0: 4})
                    cut_time = process_detection(res, 4)
                    if cut_time:
                        current_df = current_df[current_df[col_time] < cut_time]
                        cut_history.append(("CH1_Cut", cut_time))
                    advance_harvest_cut(cut_time, h_latest)

            if progress_signal:
                progress_signal.emit(85, "Detecting Initialization Points...")

            det_init = self._load_detector_by_name("init")
            if det_init:
                res = det_init.predict_single(current_df, target_class_map={0: 1, 1: 2})
                harvest_stage(det_init, harvest_df, {0: 1, 1: 2})
                process_detection(res, 1)
                process_detection(res, 2)

            if num_channels >= 3 and 5 in final_results:
                det_fine = self._load_detector_by_name("poi5_fine")
                if det_fine:
                    if progress_signal:
                        progress_signal.emit(90, "Applying Fine Adjustment...")
                    anchor_time = final_results[5]["time"]
                    fine_slice = master_df[master_df[col_time] >= anchor_time]
                    res_fine = det_fine.predict_single(fine_slice, target_class_map={0: 6})
                    harvest_stage(det_fine, fine_slice, {0: 6})

                    if 6 in res_fine:
                        process_detection(res_fine, 6)

            if 3 in final_results:
                del final_results[3]

            # ---- Configuration-prior decode: replace the cascade's greedy
            # placements with the jointly-coherent configuration. Mutates
            # final_results in place; falls back to greedy internally, so the
            # production output contract and worst-case behaviour are intact.
            decode_meta: Optional[Dict[str, Any]] = None
            if decode_config:
                if progress_signal:
                    progress_signal.emit(95, "Decoding Configuration...")
                decode_meta = self._decode_with_prior(
                    final_results, harvested, num_channels, raw_df
                )

            # ---- Zoom refinement: re-detect each placed channel POI in a
            # narrow window at full image width (zoom-trained detectors).
            # Runs AFTER the decode so the global configuration picks the
            # basin and the refiner only sharpens within it. No-ops unless
            # zoom detector assets are configured.
            refine_meta: Optional[Dict[str, Any]] = None
            if refine_pois:
                if progress_signal:
                    progress_signal.emit(97, "Refining POIs...")
                refine_meta = self._refine_with_zoom(final_results, master_df, raw_df)

            if progress_signal:
                progress_signal.emit(100, "Complete!")

            if visualize:
                try:
                    self._visualize(master_df, final_results, cut_history)
                except Exception as e:
                    Log.w(self.TAG, f"Visualization failed: {e}")

            output = self._format_output(final_results)
            if harvest_candidates:
                # Attach candidates under a reserved, namespaced key so the
                # return type and all existing POI keys are unchanged. Callers
                # that don't request harvesting never see this. Mapped to POI
                # names via POI_MAP; the legacy id-3 shim is excluded.
                cand_out: Dict[str, List[Dict[str, Any]]] = {}
                for poi_id, lst in harvested.items():
                    if poi_id == 3:
                        continue
                    name = self.POI_MAP.get(poi_id, f"POI{poi_id}")
                    # de-dup and keep sorted by confidence desc
                    seen = set()
                    uniq = []
                    for d in sorted(lst, key=lambda x: x["conf"], reverse=True):
                        key = (round(d["time"], 6), round(d["conf"], 6))
                        if key in seen:
                            continue
                        seen.add(key)
                        uniq.append(d)
                    cand_out[name] = uniq
                output["_candidates"] = cand_out

            if decode_meta is not None:
                # Reserved, namespaced key (mirrors "_candidates"): existing
                # POI keys and the return type are unchanged; callers that
                # don't request decoding never see this.
                output["_decode"] = decode_meta

            if refine_meta is not None and refine_meta.get("used"):
                output["_refine"] = refine_meta

            return output, num_channels

        except Exception as e:
            Log.e(self.TAG, f"Error during prediction: {e}")
            traceback.print_exc()
            return self._get_default_predictions(), 0
