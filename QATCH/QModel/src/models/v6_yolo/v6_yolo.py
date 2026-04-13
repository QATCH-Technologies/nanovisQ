# module: v6_yolo.py

"""QModel V6 YOLO pipeline — core detection and extended fill analysis.

Merged compilation unit (formerly ``v6_yolo.py`` + ``v6_extended_fill_detection.py``).

Pipeline order
--------------
1. Load & preprocess data.
2. **Stage 1 (optional)** — ``FillReplaySimulator`` replays the fill classifier
   over growing snapshots to locate the fill-end row *before* any channel
   detection runs.  The cascade then starts from ``fill_end_idx_buffered`` so
   channel detectors never see the fill-transition region.
3. Fill classification — determine ``num_channels`` from the full run.
4. Reverse cascading detection (Ch3 → Ch2 → Ch1 → Init).  POI1 and POI2 are
   harvested from the cascade's init-detector step; no separate re-extraction
   pass is needed.
5. **Stage 3 (optional)** — ``ExtendedFillAnalyzer`` maps the POI1→POI2 row
   delta to a viscosity proxy and decides which downstream detector set to use.

``ExtendedFillResult`` implements ``__iter__`` so callers that unpack the
standard two-tuple ``(predictions, num_channels)`` keep working without change.

Key Components
--------------
- ``QModelV6Config``              — configuration constants and progress milestones.
- ``QModelV6YOLO_FillClassifier`` — YOLO-cls wrapper.
- ``QModelV6YOLO_Detector``       — YOLO-det wrapper.
- ``QModelV6YOLO``                — controller / public API.
- ``FillReplaySimulator``         — Stage 1: fill-end localisation.
- ``ExtendedFillAnalyzer``        — Stage 3: viscosity proxy + routing decision.
- ``ExtendedFillResult`` (+ supporting dataclasses) — unified result container.

Dependencies
------------
ultralytics (YOLO), pandas, numpy, matplotlib, cv2,
QATCH internal modules (Logger, DataProcessor).

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2026-04-13
Version:
    6.3.0
"""

from __future__ import annotations

import datetime
import os
import traceback
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging — single fallback definition for headless / standalone usage
# ---------------------------------------------------------------------------
try:
    from QATCH.common.logger import Logger as Log  # pyright: ignore[reportPrivateImportUsage]
    from QATCH.QModel.src.models.v6_yolo.v6_yolo_dataprocessor import (
        QModelV6YOLO_DataProcessor,
    )
except (ImportError, ModuleNotFoundError):

    class Log:  # type: ignore[no-redef]
        @staticmethod
        def d(tag: str, msg: str) -> None:
            print(f"{tag} [DEBUG] {msg}")

        @staticmethod
        def i(tag: str, msg: str) -> None:
            print(f"{tag} [INFO] {msg}")

        @staticmethod
        def w(tag: str, msg: str) -> None:
            print(f"{tag} [WARNING] {msg}")

        @staticmethod
        def e(tag: str, msg: str) -> None:
            print(f"{tag} [ERROR] {msg}")

    Log.i(tag="[HEADLESS OPERATION]", msg="Running...")

    try:
        from v6_yolo.v6_yolo_dataprocessor import QModelV6YOLO_DataProcessor
    except ImportError:
        from v6_yolo_dataprocessor import QModelV6YOLO_DataProcessor  # type: ignore[no-redef]

try:
    from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]
except ImportError:
    Log.e(tag="[QModelV6YOLO]", msg="'ultralytics' not found. YOLO inference will fail.")

# ---------------------------------------------------------------------------
# Internal log tags
# ---------------------------------------------------------------------------
_TAG_CTRL = "QModelV6YOLO"
_TAG_CLS = "[QModelV6YOLO_FillClassifier]"
_TAG_ERD = "[QModelV6YOLO (ERD)]"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class QModelV6Config:
    """Configuration constants for the QModel V6 YOLO pipeline."""

    # Detector settings
    IMG_WIDTH: int = 2560
    IMG_HEIGHT: int = 384
    MIN_SLICE_LENGTH: int = 20
    CONF_THRESHOLD: float = 0.01

    # Fill classifier settings
    FILL_INFERENCE_W: int = 224
    FILL_INFERENCE_H: int = 224
    FILL_GEN_W: int = 640
    FILL_GEN_H: int = 640

    # Maps YOLO classification labels → channel count used by the cascade.
    FILL_CLASS_MAP: Dict[str, int] = {
        "no_fill": -1,
        "initial_fill": 0,
        "1ch": 1,
        "2ch": 2,
        "3ch": 3,
    }

    # ---------------------------------------------------------------------------
    # Progress milestones (percent) for the two operating modes.
    #
    # STANDARD mode (extended analysis disabled):
    #   10  Data loaded
    #   20  Preprocessed
    #   30  Fill classified
    #   45  Ch3 done
    #   60  Ch2 done
    #   75  Ch1 done
    #   85  Init done
    #   92  Fine adjustment done
    #  100  Complete
    #
    # EXTENDED mode (Stage 1 runs before cascade):
    #    5  Data loaded
    #   10  Preprocessed
    #   10→45  Stage 1 replay (inner-loop updates via callback)
    #   48  Fill classified
    #   57  Ch3 done
    #   66  Ch2 done
    #   75  Ch1 done
    #   84  Init done
    #   89  Fine adjustment done
    #   94  Stage 3 classification done
    #  100  Complete
    # ---------------------------------------------------------------------------

    # Standard mode
    STD_LOAD = 10
    STD_PREP = 20
    STD_CLASSIFY = 30
    STD_CH3 = 45
    STD_CH2 = 60
    STD_CH1 = 75
    STD_INIT = 85
    STD_FINE = 92
    STD_DONE = 100

    # Extended mode
    EXT_LOAD = 5
    EXT_PREP = 10
    EXT_REPLAY_LO = 10  # Stage 1 start (same as PREP; no bar stall)
    EXT_REPLAY_HI = 45  # Stage 1 end
    EXT_CLASSIFY = 48
    EXT_CH3 = 57
    EXT_CH2 = 66
    EXT_CH1 = 75
    EXT_INIT = 84
    EXT_FINE = 89
    EXT_STAGE3 = 94
    EXT_DONE = 100


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FillEndResult:
    """Output of Stage 1 (``FillReplaySimulator``)."""

    fill_end_idx: int
    """Row index in master_df where fill is stably declared."""
    fill_end_idx_buffered: int
    """``fill_end_idx + buffer_rows`` — cascade starts here."""
    fill_end_time: float
    """Corresponding ``Relative_time`` value in seconds."""
    confidence_at_end: float
    """Top-1 classifier confidence at the stable crossing frame."""
    n_frames_evaluated: int
    """Total snapshots classified (diagnostic)."""
    all_confidences: list[float] = field(default_factory=list)


@dataclass
class POIResult:
    """A single detected POI harvested from ``predict_single`` output."""

    class_id: int
    time: float
    """Timestamp in seconds."""
    row_idx_absolute: int
    """Absolute row index resolved against raw_df."""
    confidence: float


@dataclass
class InitialFillPOIs:
    """POI1 / POI2 pair forwarded to Stage 3."""

    slice_start_idx: int
    """Row in master_df where the cascade slice began (= ``cascade_start_row``)."""
    poi1: Optional[POIResult]
    poi2: Optional[POIResult]

    @property
    def delta_rows(self) -> Optional[int]:
        """POI1→POI2 interval in raw_df rows; ``None`` if either POI is missing."""
        if self.poi1 is not None and self.poi2 is not None:
            return self.poi2.row_idx_absolute - self.poi1.row_idx_absolute
        return None

    @property
    def delta_time(self) -> Optional[float]:
        """POI1→POI2 interval in seconds; ``None`` if either POI is missing."""
        if self.poi1 is not None and self.poi2 is not None:
            return self.poi2.time - self.poi1.time
        return None

    @property
    def both_detected(self) -> bool:
        return self.poi1 is not None and self.poi2 is not None


@dataclass
class ExtendedFillResult:
    """Full output of the extended-fill pipeline.

    Wraps the standard ``(predictions, num_channels)`` return and adds the
    three extended-analysis fields.

    ``__iter__`` is implemented so callers that unpack the result as a two-tuple
    continue to work without modification::

        # backward-compatible — still valid
        predictions, num_channels = model.predict(...)

        # access extended fields
        result = model.predict(...)
        if isinstance(result, ExtendedFillResult):
            print(result.detector_set, result.eta_proxy)
    """

    predictions: dict
    num_channels: int
    fill_end: Optional[FillEndResult]
    pois: Optional[InitialFillPOIs]
    eta_proxy: Optional[float]
    """Viscosity proxy (row-delta for now; calibrate against known η values)."""
    is_extended_fill: bool
    detector_set: Literal["standard", "extended"]

    def __iter__(self) -> Iterator:
        """Yield ``(predictions, num_channels)`` for backward-compatible unpacking."""
        yield self.predictions
        yield self.num_channels

    @property
    def ready_for_extended_detectors(self) -> bool:
        return self.is_extended_fill and self.fill_end is not None


# ---------------------------------------------------------------------------
# Stage 1: Fill replay simulator
# ---------------------------------------------------------------------------


class FillReplaySimulator:
    """Replays the fill classifier over a fully-buffered run to pinpoint the
    first frame where fill is confidently and stably declared.

    Mirrors the live classifier loop using ``predict_confidence()`` on growing
    snapshots of ``master_df``.

    Fill-detected condition::

        top-1 class is not "no_fill"  AND  confidence >= threshold
        held for ``duration_threshold_frames`` consecutive frames.
    """

    def __init__(
        self,
        fill_cls_loader: Callable[[], "QModelV6YOLO_FillClassifier"],
        step_rows: int = 10,
        min_start_rows: int = 50,
    ) -> None:
        """
        Args:
            fill_cls_loader: Zero-argument callable returning the cached
                ``QModelV6YOLO_FillClassifier``.  Weight loading is deferred
                until ``find_fill_end()`` is first called, and the controller's
                cached instance is always reused.
            step_rows: Rows between classifier calls.  Match to the live cadence.
            min_start_rows: Skip classification until the snapshot has at least
                this many rows (avoids false positives on run-startup noise).
        """
        self._clf_loader = fill_cls_loader
        self._step = step_rows
        self._min_start = min_start_rows

    def find_fill_end(
        self,
        master_df: pd.DataFrame,
        confidence_threshold: float = 0.85,
        duration_threshold_frames: int = 5,
        buffer_rows: int = 50,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Optional[FillEndResult]:
        """Iterate through ``master_df`` and detect the first stable fill crossing.

        Args:
            master_df: Preprocessed run DataFrame.
            confidence_threshold: Minimum confidence required for "filled".
            duration_threshold_frames: Consecutive frames required to confirm.
            buffer_rows: Extra rows added past the confirmed fill-end row; the
                cascade will start from ``fill_end_idx + buffer_rows``.
            progress_callback: Optional ``(int) -> None`` receiving values in
                ``[0, 100]`` as the loop progresses; drives the UI progress bar
                between ``EXT_REPLAY_LO`` and ``EXT_REPLAY_HI``.

        Returns:
            ``FillEndResult`` if fill was stably detected; ``None`` otherwise.
        """
        col_time = "Relative_time" if "Relative_time" in master_df.columns else master_df.columns[0]
        clf = self._clf_loader()
        if clf is None:
            Log.e(_TAG_ERD, "Stage 1: fill classifier unavailable.")
            return None

        confidences: List[float] = []
        stable_count: int = 0
        fill_end_idx: Optional[int] = None
        fill_end_conf: float = 0.0

        snapshot_ends = list(range(self._min_start, len(master_df), self._step))
        total = len(snapshot_ends)

        for i, end_row in enumerate(snapshot_ends):
            if progress_callback is not None and total > 0:
                progress_callback(int(i / total * 100))

            snapshot = master_df.iloc[:end_row]
            try:
                num_ch, conf = clf.predict_confidence(snapshot)
            except Exception as exc:
                Log.w(_TAG_ERD, f"Classifier error at row {end_row}: {exc}")
                confidences.append(0.0)
                stable_count = 0
                continue

            is_filled = num_ch != -1
            effective_conf = conf if is_filled else 0.0
            confidences.append(effective_conf)

            if effective_conf >= confidence_threshold:
                stable_count += 1
                if stable_count >= duration_threshold_frames:
                    fill_end_idx = end_row
                    fill_end_conf = effective_conf
                    Log.d(
                        _TAG_ERD,
                        f"Fill end at row {fill_end_idx} "
                        f"(num_ch={num_ch}, conf={conf:.3f}, stable={stable_count}).",
                    )
                    break
            else:
                stable_count = 0

        if fill_end_idx is None:
            Log.w(_TAG_ERD, f"Fill not detected across {len(confidences)} snapshots.")
            return None

        buffered_idx = min(fill_end_idx + buffer_rows, len(master_df) - 1)
        fill_end_time = float(master_df.iloc[fill_end_idx][col_time])

        return FillEndResult(
            fill_end_idx=fill_end_idx,
            fill_end_idx_buffered=buffered_idx,
            fill_end_time=fill_end_time,
            confidence_at_end=fill_end_conf,
            n_frames_evaluated=len(confidences),
            all_confidences=confidences,
        )


# ---------------------------------------------------------------------------
# Stage 3: Viscosity proxy + routing decision
# ---------------------------------------------------------------------------


class ExtendedFillAnalyzer:
    """Maps the POI1→POI2 row delta to a viscosity proxy and decides which
    downstream detector set to route to.

    Stage 1 (fill replay) now runs inside ``QModelV6YOLO.predict()`` before the
    cascade, and POI1/POI2 are harvested directly from the cascade's init-detector
    step.  This class therefore implements Stage 3 only — the routing decision.
    """

    DEFAULT_EXTENDED_THRESHOLD_ROWS: int = 200

    def __init__(self, extended_threshold_rows: Optional[int] = None) -> None:
        """
        Args:
            extended_threshold_rows: POI1→POI2 row-delta threshold for routing.
                Defaults to ``DEFAULT_EXTENDED_THRESHOLD_ROWS``.
        """
        self._ext_thresh = (
            extended_threshold_rows
            if extended_threshold_rows is not None
            else self.DEFAULT_EXTENDED_THRESHOLD_ROWS
        )

    def build_result(
        self,
        fill_end: Optional[FillEndResult],
        pois: InitialFillPOIs,
        predictions: dict,
        num_channels: int,
    ) -> ExtendedFillResult:
        """Run Stage 3 and assemble the final ``ExtendedFillResult``.

        Args:
            fill_end: Stage 1 output (``None`` when replay found no fill end;
                this is non-fatal — Stage 3 continues with whatever POIs exist).
            pois: POI1/POI2 harvested from the cascade init-detector step.
            predictions: Formatted output dict from ``_format_output()``.
            num_channels: Channel count from the base cascade.

        Returns:
            Fully-populated ``ExtendedFillResult``.
        """
        eta_proxy, is_extended = self._classify(pois)
        detector_set: Literal["standard", "extended"] = "extended" if is_extended else "standard"

        fe_str = (
            f"fill_end_idx={fill_end.fill_end_idx} (t={fill_end.fill_end_time:.2f}s)"
            if fill_end is not None
            else "fill_end=N/A"
        )
        Log.i(
            _TAG_ERD,
            f"{fe_str} | "
            f"POI1={'N/A' if pois.poi1 is None else pois.poi1.row_idx_absolute} "
            f"POI2={'N/A' if pois.poi2 is None else pois.poi2.row_idx_absolute} | "
            f"delta={pois.delta_rows} rows | "
            f"eta={eta_proxy if eta_proxy is not None else float('nan'):.1f} | "
            f"set={detector_set}",
        )

        return ExtendedFillResult(
            predictions=predictions,
            num_channels=num_channels,
            fill_end=fill_end,
            pois=pois,
            eta_proxy=eta_proxy,
            is_extended_fill=is_extended,
            detector_set=detector_set,
        )

    def _classify(self, pois: InitialFillPOIs) -> Tuple[Optional[float], bool]:
        """Map the POI1→POI2 row delta to a viscosity proxy and decide routing.

        Currently a simple threshold.  Replace with a calibrated regression or
        lightweight classifier once labeled data is available.

        Returns:
            ``(eta_proxy, is_extended_fill)``
        """
        if not pois.both_detected or pois.delta_rows is None:
            Log.w(_TAG_ERD, "Incomplete POIs — cannot classify, defaulting to standard.")
            return None, False

        eta_proxy = float(pois.delta_rows)
        return eta_proxy, (pois.delta_rows >= self._ext_thresh)


# ---------------------------------------------------------------------------
# Core pipeline classes
# ---------------------------------------------------------------------------


class QModelV6YOLO_FillClassifier:
    """Classifies the run state (no_fill, initial_fill, 1ch, 2ch, 3ch).

    Loads a YOLO classification model and maps its top-1 prediction to an integer
    channel count that drives the downstream detection cascade.
    """

    TAG = _TAG_CLS

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

    def predict(self, df: pd.DataFrame) -> int:
        """Classify fill state and return channel count.  Confidence discarded."""
        channels, _ = self.predict_confidence(df)
        return channels

    def predict_confidence(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Classify fill state and return ``(num_channels, confidence)``.

        Used by ``FillReplaySimulator`` so the replay loop can apply a confidence
        gate rather than just a channel-count threshold.

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

    def _map_label_to_channels(self, label: str) -> int:
        label_clean = str(label).strip().lower()
        if label_clean in QModelV6Config.FILL_CLASS_MAP:
            return QModelV6Config.FILL_CLASS_MAP[label_clean]
        if label_clean.isdigit():
            return int(label_clean)
        Log.w(self.TAG, f"Unknown label '{label}'. Defaulting to 0 channels.")
        return 0


class QModelV6YOLO_Detector:
    """Generic wrapper for a single YOLO detector (Init, Ch1, Ch2, Ch3).

    Converts DataFrame slices to model-compatible images, runs inference, and
    maps normalized bounding box x-coords back to the time domain.
    """

    def __init__(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Detector model not found: {model_path}")
        self.model = YOLO(model_path)

    def predict_single(
        self, df: pd.DataFrame, target_class_map: Optional[Dict[int, int]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Run inference on a data slice and return time-domain detections.

        Args:
            df: DataFrame slice.  Returns ``{}`` if shorter than ``MIN_SLICE_LENGTH``.
            target_class_map: Maps YOLO class IDs → application POI IDs.

        Returns:
            ``{poi_id: {"time": float, "conf": float}}``
        """
        if df is None or len(df) < QModelV6Config.MIN_SLICE_LENGTH:
            return {}

        img_base = QModelV6YOLO_DataProcessor.generate_channel_det(
            df, img_w=QModelV6Config.IMG_WIDTH, img_h=QModelV6Config.IMG_HEIGHT
        )
        results = self.model(img_base, verbose=False, conf=QModelV6Config.CONF_THRESHOLD)

        col_time = "Relative_time"
        if col_time not in df.columns:
            col_time = "time" if "time" in df.columns else df.columns[0]

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


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class QModelV6YOLO:
    """Controller for the QModel V6 YOLO pipeline.

    All model weights are lazy-loaded and cached on first use.  Stage 1
    (``FillReplaySimulator``) and Stage 3 (``ExtendedFillAnalyzer``) access the
    same cached instances via callable references — no weights file is ever
    loaded twice, regardless of whether extended analysis is enabled.

    Extended-mode execution order in ``predict()``
    -----------------------------------------------
    1. Preprocess → **Stage 1 replay** → fill classification → cascade → **Stage 3**.

    Stage 1 runs before the cascade so channel detectors never see the
    fill-transition region.  POI1/POI2 are harvested from the cascade's
    init-detector step and forwarded to Stage 3 — no separate extraction pass.
    """

    TAG = _TAG_CTRL
    POI_MAP = {1: "POI1", 2: "POI2", 3: "POI3", 4: "POI4", 5: "POI5", 6: "POI6"}

    def __init__(
        self,
        model_assets: Dict[str, Any],
        extended_fill_analysis: bool = True,
        fill_confidence_threshold: float = 0.85,
        fill_duration_threshold_frames: int = 5,
        fill_buffer_rows: int = 50,
        extended_threshold_rows: int = 200,
        step_rows: int = 10,
        min_start_rows: int = 50,
    ) -> None:
        """
        Args:
            model_assets: Dict of model weight paths::

                {
                    "fill_classifier": "path/to/classifier.pt",
                    "detectors": {
                        "init":      "path/to/init.pt",
                        "ch1":       "path/to/ch1.pt",
                        "ch2":       "path/to/ch2.pt",
                        "ch3":       "path/to/ch3.pt",
                        "poi5_fine": "path/to/poi5_fine.pt",
                    }
                }

            extended_fill_analysis: Enable Stage 1 + Stage 3.  No weights load
                at construction — loading is deferred to first ``predict()`` call.
            fill_confidence_threshold: Min confidence to declare fill end.
            fill_duration_threshold_frames: Consecutive frames needed to confirm.
            fill_buffer_rows: Extra rows after fill end; cascade starts here.
            extended_threshold_rows: POI1→POI2 row-delta threshold for routing.
            step_rows: Rows between classifier calls during Stage 1.
            min_start_rows: Minimum rows before Stage 1 begins classifying.
        """
        self.model_assets = model_assets
        self._fill_classifier: Optional[QModelV6YOLO_FillClassifier] = None
        self._detectors: Dict[str, Any] = {
            "init": None,
            "ch1": None,
            "ch2": None,
            "ch3": None,
            "poi5_fine": None,
        }

        # Stage 1 components + stored params (used in predict())
        self._fill_replay: Optional[FillReplaySimulator] = None
        self._replay_conf_thresh = fill_confidence_threshold
        self._replay_dur_thresh = fill_duration_threshold_frames
        self._replay_buffer = fill_buffer_rows

        # Stage 3
        self._extended_analyzer: Optional[ExtendedFillAnalyzer] = None

        if extended_fill_analysis:
            self.configure_extended_analysis(
                enabled=True,
                step_rows=step_rows,
                min_start_rows=min_start_rows,
                fill_confidence_threshold=fill_confidence_threshold,
                fill_duration_threshold_frames=fill_duration_threshold_frames,
                fill_buffer_rows=fill_buffer_rows,
                extended_threshold_rows=extended_threshold_rows,
            )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_extended_analysis(
        self,
        enabled: bool = True,
        step_rows: int = 10,
        min_start_rows: int = 50,
        fill_confidence_threshold: float = 0.85,
        fill_duration_threshold_frames: int = 5,
        fill_buffer_rows: int = 50,
        extended_threshold_rows: int = 200,
    ) -> None:
        """Enable or disable extended fill analysis.

        No weights are loaded here.  ``FillReplaySimulator`` receives a callable
        reference to ``_load_fill_cls`` and resolves the actual instance on first
        use, reusing whatever is already cached.

        Args:
            enabled: ``False`` disables Stage 1 + Stage 3 and restores the
                standard ``(predictions, num_channels)`` return from ``predict()``.
            step_rows: Rows between classifier calls during Stage 1.
            min_start_rows: Minimum rows before Stage 1 begins classifying.
            fill_confidence_threshold: Minimum top-1 confidence for "filled".
            fill_duration_threshold_frames: Consecutive frames needed to confirm.
            fill_buffer_rows: Extra rows after fill end; cascade starts here.
            extended_threshold_rows: POI1→POI2 row-delta threshold for routing.
        """
        if not enabled:
            self._fill_replay = None
            self._extended_analyzer = None
            Log.i(self.TAG, "Extended fill analysis disabled.")
            return

        self._replay_conf_thresh = fill_confidence_threshold
        self._replay_dur_thresh = fill_duration_threshold_frames
        self._replay_buffer = fill_buffer_rows

        self._fill_replay = FillReplaySimulator(
            fill_cls_loader=self._load_fill_cls,
            step_rows=step_rows,
            min_start_rows=min_start_rows,
        )
        self._extended_analyzer = ExtendedFillAnalyzer(
            extended_threshold_rows=extended_threshold_rows,
        )
        Log.i(self.TAG, "Extended fill analysis enabled (weights load on first predict()).")

    # ------------------------------------------------------------------
    # Lazy model loaders
    # ------------------------------------------------------------------

    def _load_fill_cls(self) -> Optional[QModelV6YOLO_FillClassifier]:
        """Lazy-load and cache the fill classifier."""
        if self._fill_classifier is None:
            model_path = self.model_assets.get("fill_classifier")
            if model_path:
                self._fill_classifier = QModelV6YOLO_FillClassifier(model_path)
        return self._fill_classifier

    def _load_detector_by_name(self, name: str) -> Optional[QModelV6YOLO_Detector]:
        """Lazy-load and cache a YOLO detector by shorthand name."""
        if self._detectors.get(name) is None:
            model_path = self.model_assets.get("detectors", {}).get(name)
            if model_path:
                try:
                    self._detectors[name] = QModelV6YOLO_Detector(model_path)
                except Exception as e:
                    Log.e(self.TAG, f"Error loading detector '{name}': {e}")
                    return None
        return self._detectors.get(name)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _get_default_predictions(self) -> Dict[str, Dict[str, List]]:
        return {
            poi_name: {"indices": [-1], "confidences": [-1]} for poi_name in self.POI_MAP.values()
        }

    def _format_output(
        self, final_results: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Dict[str, List[float]]]:
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

    def _validate_file_buffer(self, file_buffer: Union[str, Any]) -> pd.DataFrame:
        try:
            if not isinstance(file_buffer, str):
                if hasattr(file_buffer, "seekable") and file_buffer.seekable():
                    file_buffer.seek(0)
            return pd.read_csv(file_buffer)
        except Exception as e:
            raise e

    def _get_raw_index(self, raw_df: pd.DataFrame, target_time: float) -> int:
        """Nearest-neighbour lookup: time value → absolute row index in raw_df."""
        col_time = "Relative_time"
        if col_time not in raw_df.columns:
            col_time = "time" if "time" in raw_df.columns else raw_df.columns[0]
        times = raw_df[col_time].to_numpy(dtype=float)
        idx = int(np.abs(times - target_time).argmin())
        return int(raw_df.index[idx])

    @staticmethod
    def _make_poi_result(
        final_results: Dict[int, Dict[str, Any]], poi_id: int
    ) -> Optional[POIResult]:
        """Build a ``POIResult`` from a ``final_results`` entry, or ``None``."""
        if poi_id not in final_results:
            return None
        data = final_results[poi_id]
        return POIResult(
            class_id=poi_id,
            time=data["time"],
            row_idx_absolute=data["index"],
            confidence=data["conf"],
        )

    def _visualize(
        self,
        df: pd.DataFrame,
        results: dict,
        cut_history: list,
        save_path: str = "v6_debug.png",
    ) -> None:
        if df is None or df.empty:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name, ext = os.path.splitext(save_path)
        final_save_path = f"{base_name}_{timestamp}{ext}"

        time_col = df["Relative_time"].values
        signal = df["Dissipation"].values if "Dissipation" in df.columns else df.iloc[:, 1].values

        plt.figure(figsize=(12, 6))
        plt.plot(time_col, signal, color="gray", alpha=0.6, label="Raw Signal")

        colors = {1: "green", 2: "blue", 4: "orange", 5: "red", 6: "purple"}
        for poi_id, data in results.items():
            if data and "time" in data:
                c = colors.get(poi_id, "black")
                name = self.POI_MAP.get(poi_id, f"POI{poi_id}")
                plt.axvline(x=data["time"], color=c, linestyle="-", linewidth=2, label=name)

        for _, cut_time in cut_history:
            plt.axvline(x=cut_time, color="red", linestyle="--", linewidth=1, alpha=0.5)
            plt.axvspan(cut_time, np.max(time_col), color="red", alpha=0.05)

        plt.title(f"Cascade Detection Debug — {len(cut_history)} slices applied")
        plt.savefig(final_save_path)
        plt.close()
        Log.i(self.TAG, f"Debug plot saved to: {final_save_path}")

    # ------------------------------------------------------------------
    # Main prediction entry point
    # ------------------------------------------------------------------

    def predict(
        self,
        progress_signal: Any = None,
        file_buffer: Any = None,
        df: Optional[pd.DataFrame] = None,
        visualize: bool = False,
        num_channels: Optional[int] = None,
    ) -> Union[Tuple[Dict, int], ExtendedFillResult]:
        """Execute the QModel V6 YOLO prediction pipeline.

        Extended-mode execution order
        ------------------------------
        1. Load & preprocess.
        2. **Stage 1** — fill replay pinpoints ``fill_end_idx_buffered``.
        3. Fill classification (``num_channels``).
        4. Cascade from ``cascade_start_row`` (= ``fill_end_idx_buffered`` or 0).
           Init-detector step produces POI1 and POI2 in the correct region.
        5. **Stage 3** — build ``ExtendedFillResult`` using cascade POI1/POI2.

        Progress bar
        ------------
        Two milestone sets from ``QModelV6Config`` cover the full 0–100 range
        without overlap or stalling.  In extended mode Stage 1 occupies 10→45 %
        with per-iteration updates via ``progress_callback``; the cascade and
        Stage 3 fill 45→100 %.

        Args:
            progress_signal: PyQt-style signal with ``.emit(int, str)``.
            file_buffer: File path or file-like CSV source.
            df: Pre-loaded DataFrame; ignored when ``file_buffer`` is provided.
            visualize: Save a debug plot of detections and cut points.
            num_channels: Force channel count; bypasses fill classification.

        Returns:
            Standard mode: ``Tuple[Dict, int]``.
            Extended mode: ``ExtendedFillResult`` (also iterable as a two-tuple).
        """
        try:
            has_ext = self._fill_replay is not None
            C = QModelV6Config

            # Select milestone set
            P_LOAD = C.EXT_LOAD if has_ext else C.STD_LOAD
            P_PREP = C.EXT_PREP if has_ext else C.STD_PREP
            P_CLASSIFY = C.EXT_CLASSIFY if has_ext else C.STD_CLASSIFY
            P_CH3 = C.EXT_CH3 if has_ext else C.STD_CH3
            P_CH2 = C.EXT_CH2 if has_ext else C.STD_CH2
            P_CH1 = C.EXT_CH1 if has_ext else C.STD_CH1
            P_INIT = C.EXT_INIT if has_ext else C.STD_INIT
            P_FINE = C.EXT_FINE if has_ext else C.STD_FINE
            P_DONE = C.EXT_DONE if has_ext else C.STD_DONE

            def emit(pct: int, msg: str) -> None:
                if progress_signal is not None:
                    progress_signal.emit(pct, msg)

            # ----------------------------------------------------------
            # 1. Load
            # ----------------------------------------------------------
            if file_buffer is not None:
                raw_df = self._validate_file_buffer(file_buffer)
            elif df is not None:
                raw_df = df
            else:
                raise ValueError("No data provided: supply either file_buffer or df.")

            emit(P_LOAD, "Data Loaded")

            # ----------------------------------------------------------
            # 2. Preprocess
            # ----------------------------------------------------------
            master_df = QModelV6YOLO_DataProcessor.preprocess_dataframe(raw_df.copy())
            emit(P_PREP, "Preprocessing Data...")

            if master_df is None or master_df.empty:
                raise ValueError("Preprocessing failed — master_df is empty.")

            # ----------------------------------------------------------
            # 3. Stage 1: Fill replay (extended mode, BEFORE cascade)
            # ----------------------------------------------------------
            fill_end_result: Optional[FillEndResult] = None
            cascade_start_row: int = 0

            if has_ext and self._fill_replay is not None:
                emit(C.EXT_REPLAY_LO, "Replaying Fill Classifier...")

                def _replay_cb(pct: int) -> None:
                    """Map [0,100] → [EXT_REPLAY_LO, EXT_REPLAY_HI]."""
                    scaled = C.EXT_REPLAY_LO + int(pct / 100 * (C.EXT_REPLAY_HI - C.EXT_REPLAY_LO))
                    emit(scaled, "Replaying Fill Classifier...")

                fill_end_result = self._fill_replay.find_fill_end(
                    master_df=master_df,
                    confidence_threshold=self._replay_conf_thresh,
                    duration_threshold_frames=self._replay_dur_thresh,
                    buffer_rows=self._replay_buffer,
                    progress_callback=_replay_cb,
                )

                if fill_end_result is not None:
                    cascade_start_row = fill_end_result.fill_end_idx_buffered
                    Log.i(
                        self.TAG,
                        f"Stage 1: cascade starts at row {cascade_start_row} "
                        f"(fill end t={fill_end_result.fill_end_time:.2f}s).",
                    )
                else:
                    Log.w(self.TAG, "Stage 1: no fill end found — cascade starts at row 0.")

            # ----------------------------------------------------------
            # 4. Fill classification
            # ----------------------------------------------------------
            if num_channels is None:
                emit(P_CLASSIFY, "Determining Channel Count...")
                fill_cls = self._load_fill_cls()
                num_channels = int(fill_cls.predict(master_df)) if fill_cls else 3

            if num_channels == -1:
                return self._get_default_predictions(), num_channels

            # ----------------------------------------------------------
            # 5. Reverse cascading detection
            #    Starts from cascade_start_row (0 in standard mode or when
            #    Stage 1 finds nothing; fill_end_idx_buffered otherwise).
            # ----------------------------------------------------------
            final_results: Dict[int, Dict[str, Any]] = {}
            current_df = (
                master_df.iloc[cascade_start_row:].copy()
                if cascade_start_row > 0
                else master_df.copy()
            )
            col_time = (
                "Relative_time" if "Relative_time" in current_df.columns else current_df.columns[0]
            )
            cut_history: List[Tuple[str, float]] = []

            def process_detection(res_dict: dict, poi_id: int) -> Optional[float]:
                if poi_id in res_dict:
                    t_det = res_dict[poi_id]["time"]
                    c_det = res_dict[poi_id]["conf"]
                    raw_idx = self._get_raw_index(raw_df, t_det)
                    final_results[poi_id] = {"index": raw_idx, "conf": c_det, "time": t_det}
                    return t_det
                return None

            if num_channels >= 3:
                emit(P_CH3, "Detecting Channel 3...")
                det_ch3 = self._load_detector_by_name("ch3")
                if det_ch3:
                    res = det_ch3.predict_single(current_df, target_class_map={0: 6})
                    cut_time = process_detection(res, 6)
                    if cut_time:
                        current_df = current_df[current_df[col_time] < cut_time]
                        cut_history.append(("CH3_Cut", cut_time))

            if num_channels >= 2:
                emit(P_CH2, "Detecting Channel 2...")
                det_ch2 = self._load_detector_by_name("ch2")
                if det_ch2:
                    res = det_ch2.predict_single(current_df, target_class_map={0: 5})
                    cut_time = process_detection(res, 5)
                    if cut_time:
                        current_df = current_df[current_df[col_time] < cut_time]
                        cut_history.append(("CH2_Cut", cut_time))

            if num_channels >= 1:
                emit(P_CH1, "Detecting Channel 1...")
                det_ch1 = self._load_detector_by_name("ch1")
                if det_ch1:
                    res = det_ch1.predict_single(current_df, target_class_map={0: 4})
                    cut_time = process_detection(res, 4)
                    if cut_time:
                        current_df = current_df[current_df[col_time] < cut_time]
                        cut_history.append(("CH1_Cut", cut_time))

            emit(P_INIT, "Detecting Initialization Points...")
            det_init = self._load_detector_by_name("init")
            if det_init:
                res = det_init.predict_single(current_df, target_class_map={0: 1, 1: 2})
                process_detection(res, 1)
                process_detection(res, 2)

            if num_channels >= 3 and 5 in final_results:
                det_fine = self._load_detector_by_name("poi5_fine")
                if det_fine:
                    emit(P_FINE, "Applying Fine Adjustment...")
                    anchor_time = final_results[5]["time"]
                    fine_slice = master_df[master_df[col_time] >= anchor_time]
                    res_fine = det_fine.predict_single(fine_slice, target_class_map={0: 6})
                    if 6 in res_fine:
                        process_detection(res_fine, 6)

            # POI3 is internal-only — strip before formatting
            final_results.pop(3, None)

            if visualize:
                try:
                    self._visualize(master_df, final_results, cut_history)
                except Exception as e:
                    Log.w(self.TAG, f"Visualization failed: {e}")

            formatted = self._format_output(final_results)

            # ----------------------------------------------------------
            # 6. Stage 3: viscosity proxy + routing (extended mode only)
            #    POI1/POI2 come directly from the cascade above.
            # ----------------------------------------------------------
            if has_ext and self._extended_analyzer is not None:
                emit(C.EXT_STAGE3, "Classifying Fill Duration...")

                poi1 = self._make_poi_result(final_results, poi_id=1)
                poi2 = self._make_poi_result(final_results, poi_id=2)

                if poi1 is None:
                    Log.w(self.TAG, "Stage 3: POI1 not detected in cascade.")
                if poi2 is None:
                    Log.w(self.TAG, "Stage 3: POI2 not detected in cascade.")

                pois = InitialFillPOIs(
                    slice_start_idx=cascade_start_row,
                    poi1=poi1,
                    poi2=poi2,
                )

                try:
                    result = self._extended_analyzer.build_result(
                        fill_end=fill_end_result,
                        pois=pois,
                        predictions=formatted,
                        num_channels=num_channels,
                    )
                    emit(P_DONE, "Complete!")
                    return result
                except Exception as e:
                    Log.e(self.TAG, f"Stage 3 failed: {e}")
                    # Fall through to standard return so the caller always gets
                    # a usable result.

            emit(P_DONE, "Complete!")
            return formatted, num_channels

        except Exception as e:
            Log.e(self.TAG, f"Error during prediction: {e}")
            traceback.print_exc()
            return self._get_default_predictions(), 0
