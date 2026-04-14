# module: v6_yolo.py

"""QModel V6 YOLO pipeline — controller / public API.

Pipeline order (extended mode)
-------------------------------
1.  Load & preprocess data.
2.  **Stage 1** — ``FillReplaySimulator`` replays the fill classifier to
    locate ``fill_end_time``.  A *time-based* buffer (seconds, not rows)
    produces ``fill_end_time_buffered``; the cascade starts at the first
    row past that time.
3.  **Stage 2** — ``POIScout`` creates a bounded ``[fill_end_time,
    fill_end_time + scout_window]`` slice, runs the init detector to get
    early POI1/POI2 readings, and decides **standard vs. ERD weights**
    based on ``POI2.time − POI1.time``.
4.  Fill classification — determine ``num_channels``.
5.  Reverse cascading detection (Ch3 -> Ch2 -> Ch1 -> Init) using the
    **routed weight bank** chosen in Stage 2.
6.  **Stage 3** — ``ExtendedFillAnalyzer`` packages everything into an
    ``ExtendedFillResult``.

Standard mode (``extended_fill_analysis=False``) skips Stages 1–3 and
returns the classic ``(predictions, num_channels)`` tuple.

Key Components (sub-modules)
-----------------------------
- ``v6_yolo_config``          — ``QModelV6Config``, ``resolve_time_column``
- ``v6_yolo_results``         — dataclasses (``FillEndResult``, ``POIResult``,
                                ``POIScoutResult``, ``ExtendedFillResult``)
- ``v6_yolo_models``          — ``QModelV6YOLO_FillClassifier``,
                                ``QModelV6YOLO_Detector``
- ``v6_yolo_fill_replay``     — Stage 1: ``FillReplaySimulator``
- ``v6_yolo_poi_scout``       — Stage 2: ``POIScout``
- ``v6_yolo_routing``         — Stage 3: ``ExtendedFillAnalyzer``
- ``v6_yolo_visualization``   — debug plotting

Dependencies
------------
ultralytics (YOLO), pandas, numpy, matplotlib, cv2,
QATCH internal modules (Logger, DataProcessor).

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2026-04-14
Version:
    7.0.0
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ── Sub-module imports ────────────────────────────────────────────────
from QATCH.QModel.src.models.v6_yolo.v6_yolo_config import QModelV6Config, resolve_time_column
from QATCH.QModel.src.models.v6_yolo.v6_yolo_fill_replay import FillReplaySimulator
from QATCH.QModel.src.models.v6_yolo.v6_yolo_logging import Log, TAG_CTRL
from QATCH.QModel.src.models.v6_yolo.v6_yolo_models import (
    QModelV6YOLO_Detector,
    QModelV6YOLO_FillClassifier,
)
from QATCH.QModel.src.models.v6_yolo.v6_yolo_poi_scout import POIScout
from QATCH.QModel.src.models.v6_yolo.v6_yolo_results import (
    ExtendedFillResult,
    FillEndResult,
    POIResult,
    POIScoutResult,
)
from QATCH.QModel.src.models.v6_yolo.v6_yolo_routing import ExtendedFillAnalyzer
from QATCH.QModel.src.models.v6_yolo.v6_yolo_visualization import save_debug_plot

try:
    from QATCH.QModel.src.models.v6_yolo.v6_yolo_dataprocessor import (
        QModelV6YOLO_DataProcessor,
    )
except (ImportError, ModuleNotFoundError):
    try:
        from v6_yolo.v6_yolo_dataprocessor import QModelV6YOLO_DataProcessor
    except ImportError:
        from v6_yolo_dataprocessor import QModelV6YOLO_DataProcessor  # type: ignore[no-redef]

# ── Re-export public names so callers importing from v6_yolo still work ──
__all__ = [
    "QModelV6Config",
    "QModelV6YOLO",
    "QModelV6YOLO_FillClassifier",
    "QModelV6YOLO_Detector",
    "FillEndResult",
    "POIResult",
    "POIScoutResult",
    "ExtendedFillResult",
    "FillReplaySimulator",
    "POIScout",
    "ExtendedFillAnalyzer",
]


# ──────────────────────────────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────────────────────────────


class QModelV6YOLO:
    """Controller for the QModel V6 YOLO pipeline.

    All model weights are lazy-loaded and cached on first use.  Two
    independent weight caches exist — ``_detectors`` (standard) and
    ``_detectors_erd`` (extended / ERD) — and the cascade dynamically
    resolves which bank to use based on the Stage 2 routing decision.

    Extended-mode execution order in ``predict()``
    -----------------------------------------------
    1.  Preprocess -> **Stage 1** replay -> **Stage 2** POI scout (routes
        to standard or ERD weights) -> fill classification -> cascade
        (using routed weights) -> **Stage 3** result assembly.
    """

    TAG = TAG_CTRL
    POI_MAP = {1: "POI1", 2: "POI2", 3: "POI3", 4: "POI4", 5: "POI5", 6: "POI6"}

    def __init__(
        self,
        model_assets: Dict[str, Any],
        extended_fill_analysis: bool = True,
        fill_confidence_threshold: float = 0.85,
        fill_duration_threshold_frames: int = 5,
        fill_buffer_seconds: float = 5.0,
        scout_window_seconds: float = 60.0,
        extended_threshold_seconds: float = 15.0,
        step_rows: int = 10,
        min_start_rows: int = 50,
    ) -> None:
        """
        Args:
            model_assets: Dict of model weight paths.  For ERD routing
                supply a second detector bank under ``"detectors_erd"``::

                    {
                        "fill_classifier": "path/to/classifier.pt",
                        "detectors": {
                            "init": "...", "ch1": "...", "ch2": "...",
                            "ch3": "...", "poi5_fine": "...",
                        },
                        "detectors_erd": {          # optional
                            "init": "...", "ch1": "...", ...
                        },
                    }

                If ``"detectors_erd"`` is absent the standard bank is
                used for both routing outcomes (graceful no-op).

            extended_fill_analysis: Enable Stages 1–3.
            fill_confidence_threshold: Min confidence to declare fill end.
            fill_duration_threshold_frames: Consecutive frames to confirm.
            fill_buffer_seconds: Time (seconds) added past fill end;
                the cascade starts at the first row past this time.
            scout_window_seconds: Length of the Relative_time window
                for Stage 2 POI scouting.
            extended_threshold_seconds: POI1->POI2 time-delta (seconds)
                threshold for the standard / ERD routing decision.
            step_rows: Rows between classifier calls during Stage 1.
            min_start_rows: Minimum rows before Stage 1 begins.
        """
        self.model_assets = model_assets

        # ── Weight caches ─────────────────────────────────────────────
        self._fill_classifier: Optional[QModelV6YOLO_FillClassifier] = None
        self._detectors: Dict[str, Optional[QModelV6YOLO_Detector]] = {
            "init": None,
            "ch1": None,
            "ch2": None,
            "ch3": None,
            "poi5_fine": None,
        }
        self._detectors_erd: Dict[str, Optional[QModelV6YOLO_Detector]] = {
            "init": None,
            "ch1": None,
            "ch2": None,
            "ch3": None,
            "poi5_fine": None,
        }

        # ── Stage components ──────────────────────────────────────────
        self._fill_replay: Optional[FillReplaySimulator] = None
        self._poi_scout: Optional[POIScout] = None
        self._extended_analyzer: Optional[ExtendedFillAnalyzer] = None

        # Stored Stage 1 params (forwarded in predict())
        self._replay_conf_thresh = fill_confidence_threshold
        self._replay_dur_thresh = fill_duration_threshold_frames
        self._replay_buffer_sec = fill_buffer_seconds

        if extended_fill_analysis:
            self.configure_extended_analysis(
                enabled=True,
                step_rows=step_rows,
                min_start_rows=min_start_rows,
                fill_confidence_threshold=fill_confidence_threshold,
                fill_duration_threshold_frames=fill_duration_threshold_frames,
                fill_buffer_seconds=fill_buffer_seconds,
                scout_window_seconds=scout_window_seconds,
                extended_threshold_seconds=extended_threshold_seconds,
            )

    # ══════════════════════════════════════════════════════════════════
    # Configuration
    # ══════════════════════════════════════════════════════════════════

    def configure_extended_analysis(
        self,
        enabled: bool = True,
        step_rows: int = 10,
        min_start_rows: int = 50,
        fill_confidence_threshold: float = 0.85,
        fill_duration_threshold_frames: int = 5,
        fill_buffer_seconds: float = 5.0,
        scout_window_seconds: float = 60.0,
        extended_threshold_seconds: float = 15.0,
    ) -> None:
        """Enable or disable extended fill analysis (Stages 1–3).

        No weights are loaded here.  All loaders receive callable
        references and resolve actual instances on first use, reusing
        whatever is already cached.

        Args:
            enabled: ``False`` tears down Stages 1–3 and restores the
                standard ``(predictions, num_channels)`` return.
            step_rows: Rows between fill classifier calls (Stage 1).
            min_start_rows: Rows before Stage 1 begins classifying.
            fill_confidence_threshold: Min confidence for "filled".
            fill_duration_threshold_frames: Consecutive frames needed.
            fill_buffer_seconds: Time buffer after fill end (seconds).
            scout_window_seconds: Stage 2 scout window length (seconds).
            extended_threshold_seconds: POI time-delta routing threshold.
        """
        if not enabled:
            self._fill_replay = None
            self._poi_scout = None
            self._extended_analyzer = None
            Log.i(self.TAG, "Extended fill analysis disabled.")
            return

        self._replay_conf_thresh = fill_confidence_threshold
        self._replay_dur_thresh = fill_duration_threshold_frames
        self._replay_buffer_sec = fill_buffer_seconds

        self._fill_replay = FillReplaySimulator(
            fill_cls_loader=self._load_fill_cls,
            step_rows=step_rows,
            min_start_rows=min_start_rows,
        )
        self._poi_scout = POIScout(
            init_detector_loader=lambda: self._load_detector_by_name("init"),
            scout_window_seconds=scout_window_seconds,
            extended_threshold_seconds=extended_threshold_seconds,
        )
        self._extended_analyzer = ExtendedFillAnalyzer()

        Log.i(
            self.TAG,
            "Extended fill analysis enabled "
            f"(buffer={fill_buffer_seconds}s, "
            f"scout_window={scout_window_seconds}s, "
            f"threshold={extended_threshold_seconds}s).  "
            "Weights load on first predict().",
        )

    # ══════════════════════════════════════════════════════════════════
    # Lazy model loaders
    # ══════════════════════════════════════════════════════════════════

    def _load_fill_cls(self) -> Optional[QModelV6YOLO_FillClassifier]:
        """Lazy-load and cache the fill classifier."""
        if self._fill_classifier is None:
            model_path = self.model_assets.get("fill_classifier")
            if model_path:
                self._fill_classifier = QModelV6YOLO_FillClassifier(model_path)
        return self._fill_classifier

    def _load_detector_by_name(
        self,
        name: str,
        bank: str = "standard",
    ) -> Optional[QModelV6YOLO_Detector]:
        """Lazy-load and cache a YOLO detector from the specified bank.

        Args:
            name: Detector shorthand (``"init"``, ``"ch1"``, etc.).
            bank: ``"standard"`` or ``"extended"``; selects the weight
                source and the corresponding cache dict.

        Returns:
            Cached ``QModelV6YOLO_Detector`` or ``None``.
        """
        if bank == "extended":
            cache = self._detectors_erd
            assets_key = "detectors_erd"
        else:
            cache = self._detectors
            assets_key = "detectors"

        if cache.get(name) is None:
            model_path = self.model_assets.get(assets_key, {}).get(name)
            # Fall back to standard weights when ERD weights are absent.
            if model_path is None and bank == "extended":
                model_path = self.model_assets.get("detectors", {}).get(name)
                if model_path:
                    Log.w(
                        self.TAG,
                        f"ERD weights missing for '{name}' — " f"falling back to standard.",
                    )
            if model_path:
                try:
                    cache[name] = QModelV6YOLO_Detector(model_path)
                except Exception as e:
                    Log.e(self.TAG, f"Error loading detector '{name}' ({bank}): {e}")
                    return None
        return cache.get(name)

    # ══════════════════════════════════════════════════════════════════
    # Output helpers
    # ══════════════════════════════════════════════════════════════════

    def _get_default_predictions(self) -> Dict[str, Dict[str, list]]:
        return {
            poi_name: {"indices": [-1], "confidences": [-1]} for poi_name in self.POI_MAP.values()
        }

    def _format_output(
        self, final_results: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Dict[str, list]]:
        output: Dict[str, Dict[str, list]] = {}
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

    @staticmethod
    def _validate_file_buffer(file_buffer) -> pd.DataFrame:
        if not isinstance(file_buffer, str):
            if hasattr(file_buffer, "seekable") and file_buffer.seekable():
                file_buffer.seek(0)
        return pd.read_csv(file_buffer)

    @staticmethod
    def _get_raw_index(raw_df: pd.DataFrame, target_time: float) -> int:
        """Nearest-neighbour lookup: time value -> absolute row index."""
        col_time = resolve_time_column(raw_df)
        times = raw_df[col_time].to_numpy(dtype=float)
        idx = int(np.abs(times - target_time).argmin())
        return int(raw_df.index[idx])

    @staticmethod
    def _make_poi_result(
        final_results: Dict[int, Dict[str, Any]], poi_id: int
    ) -> Optional[POIResult]:
        if poi_id not in final_results:
            return None
        data = final_results[poi_id]
        return POIResult(
            class_id=poi_id,
            time=data["time"],
            row_idx_absolute=data["index"],
            confidence=data["conf"],
        )

    # ══════════════════════════════════════════════════════════════════
    # Main prediction entry point
    # ══════════════════════════════════════════════════════════════════

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
        2. **Stage 1** — fill replay -> ``fill_end_time``
           (time-based buffer).
        3. **Stage 2** — POI scout -> ``detector_set``
           (``"standard"`` or ``"extended"``).
        4. Fill classification (``num_channels``).
        5. Cascade from ``cascade_start_row`` using the **routed**
           weight bank.
        6. **Stage 3** — assemble ``ExtendedFillResult``.

        Args:
            progress_signal: PyQt-style signal with ``.emit(int, str)``.
            file_buffer: File path or file-like CSV source.
            df: Pre-loaded DataFrame; ignored when ``file_buffer`` given.
            visualize: Save a debug plot of detections and cut points.
            num_channels: Force channel count; bypasses classification.

        Returns:
            Standard mode: ``Tuple[Dict, int]``.
            Extended mode: ``ExtendedFillResult`` (iterable as two-tuple).
        """
        try:
            has_ext = self._fill_replay is not None
            C = QModelV6Config

            # ── Select milestone set ──────────────────────────────────
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

            # ══════════════════════════════════════════════════════════
            # 1. Load
            # ══════════════════════════════════════════════════════════
            if file_buffer is not None:
                raw_df = self._validate_file_buffer(file_buffer)
            elif df is not None:
                raw_df = df
            else:
                raise ValueError("No data provided: supply file_buffer or df.")

            emit(P_LOAD, "Data Loaded")

            # ══════════════════════════════════════════════════════════
            # 2. Preprocess
            # ══════════════════════════════════════════════════════════
            master_df = QModelV6YOLO_DataProcessor.preprocess_dataframe(raw_df.copy())
            emit(P_PREP, "Preprocessing Data...")

            if master_df is None or master_df.empty:
                raise ValueError("Preprocessing failed — master_df is empty.")

            # ══════════════════════════════════════════════════════════
            # 3. Stage 1: Fill replay (extended mode only)
            # ══════════════════════════════════════════════════════════
            fill_end_result: Optional[FillEndResult] = None
            scout_result: Optional[POIScoutResult] = None
            cascade_start_row: int = 0
            detector_bank: str = "standard"

            if has_ext and self._fill_replay is not None:
                emit(C.EXT_REPLAY_LO, "Replaying Fill Classifier...")

                def _replay_cb(pct: int) -> None:
                    scaled = C.EXT_REPLAY_LO + int(pct / 100 * (C.EXT_REPLAY_HI - C.EXT_REPLAY_LO))
                    emit(scaled, "Replaying Fill Classifier...")

                fill_end_result = self._fill_replay.find_fill_end(
                    master_df=master_df,
                    confidence_threshold=self._replay_conf_thresh,
                    duration_threshold_frames=self._replay_dur_thresh,
                    buffer_seconds=self._replay_buffer_sec,
                    progress_callback=_replay_cb,
                )

                if fill_end_result is not None:
                    cascade_start_row = fill_end_result.fill_end_idx_buffered
                    Log.i(
                        self.TAG,
                        f"Stage 1: cascade will start at row {cascade_start_row} "
                        f"(fill end t={fill_end_result.fill_end_time:.2f}s, "
                        f"buffered t={fill_end_result.fill_end_time_buffered:.2f}s).",
                    )
                else:
                    Log.w(self.TAG, "Stage 1: no fill end found — cascade starts at row 0.")

            # ══════════════════════════════════════════════════════════
            # 4. Stage 2: POI scout (extended mode, after Stage 1)
            # ══════════════════════════════════════════════════════════
            if has_ext and self._poi_scout is not None and fill_end_result is not None:
                emit(C.EXT_SCOUT_LO, "Scouting POI1/POI2...")

                def _scout_cb(pct: int) -> None:
                    scaled = C.EXT_SCOUT_LO + int(pct / 100 * (C.EXT_SCOUT_HI - C.EXT_SCOUT_LO))
                    emit(scaled, "Scouting POI1/POI2...")

                scout_result = self._poi_scout.scout(
                    master_df=master_df,
                    raw_df=raw_df,
                    fill_end=fill_end_result,
                    progress_callback=_scout_cb,
                )
                detector_bank = scout_result.detector_set
                Log.i(
                    self.TAG,
                    f"Stage 2: routed to '{detector_bank}' weight bank.",
                )

            # ══════════════════════════════════════════════════════════
            # 5. Fill classification
            # ══════════════════════════════════════════════════════════
            if num_channels is None:
                emit(P_CLASSIFY, "Determining Channel Count...")
                fill_cls = self._load_fill_cls()
                num_channels = int(fill_cls.predict(master_df)) if fill_cls else 3

            if num_channels == -1:
                return self._get_default_predictions(), num_channels

            # ══════════════════════════════════════════════════════════
            # 6. Reverse cascading detection
            #    Uses the ROUTED weight bank from Stage 2.
            # ══════════════════════════════════════════════════════════
            final_results: Dict[int, Dict[str, Any]] = {}
            current_df = (
                master_df.iloc[cascade_start_row:].copy()
                if cascade_start_row > 0
                else master_df.copy()
            )
            col_time = resolve_time_column(current_df)
            cut_history: List[Tuple[str, float]] = []

            def process_detection(res_dict: dict, poi_id: int) -> Optional[float]:
                if poi_id in res_dict:
                    t_det = res_dict[poi_id]["time"]
                    c_det = res_dict[poi_id]["conf"]
                    raw_idx = self._get_raw_index(raw_df, t_det)
                    final_results[poi_id] = {
                        "index": raw_idx,
                        "conf": c_det,
                        "time": t_det,
                    }
                    return t_det
                return None

            if num_channels >= 3:
                emit(P_CH3, "Detecting Channel 3...")
                det = self._load_detector_by_name("ch3", bank=detector_bank)
                if det:
                    res = det.predict_single(current_df, target_class_map={0: 6})
                    cut_time = process_detection(res, 6)
                    if cut_time:
                        current_df = current_df[current_df[col_time] < cut_time]
                        cut_history.append(("CH3_Cut", cut_time))

            if num_channels >= 2:
                emit(P_CH2, "Detecting Channel 2...")
                det = self._load_detector_by_name("ch2", bank=detector_bank)
                if det:
                    res = det.predict_single(current_df, target_class_map={0: 5})
                    cut_time = process_detection(res, 5)
                    if cut_time:
                        current_df = current_df[current_df[col_time] < cut_time]
                        cut_history.append(("CH2_Cut", cut_time))

            if num_channels >= 1:
                emit(P_CH1, "Detecting Channel 1...")
                det = self._load_detector_by_name("ch1", bank=detector_bank)
                if det:
                    res = det.predict_single(current_df, target_class_map={0: 4})
                    cut_time = process_detection(res, 4)
                    if cut_time:
                        current_df = current_df[current_df[col_time] < cut_time]
                        cut_history.append(("CH1_Cut", cut_time))

            emit(P_INIT, "Detecting Initialization Points...")
            det_init = self._load_detector_by_name("init", bank=detector_bank)
            if det_init:
                res = det_init.predict_single(current_df, target_class_map={0: 1, 1: 2})
                process_detection(res, 1)
                process_detection(res, 2)

            if num_channels >= 3 and 5 in final_results:
                det_fine = self._load_detector_by_name("poi5_fine", bank=detector_bank)
                if det_fine:
                    emit(P_FINE, "Applying Fine Adjustment...")
                    anchor_time = final_results[5]["time"]
                    fine_slice = master_df[master_df[col_time] >= anchor_time]
                    res_fine = det_fine.predict_single(fine_slice, target_class_map={0: 6})
                    if 6 in res_fine:
                        process_detection(res_fine, 6)

            # POI3 is internal-only — strip before formatting.
            final_results.pop(3, None)

            if visualize:
                try:
                    save_debug_plot(
                        master_df,
                        final_results,
                        cut_history,
                        fill_end=fill_end_result,
                        scout=scout_result,
                    )
                except Exception as e:
                    Log.w(self.TAG, f"Visualization failed: {e}")

            formatted = self._format_output(final_results)

            # ══════════════════════════════════════════════════════════
            # 7. Stage 3: assemble ExtendedFillResult (extended only)
            # ══════════════════════════════════════════════════════════
            if has_ext and self._extended_analyzer is not None:
                emit(C.EXT_STAGE3, "Packaging Results...")

                try:
                    result = self._extended_analyzer.build_result(
                        fill_end=fill_end_result,
                        scout=scout_result,
                        predictions=formatted,
                        num_channels=num_channels,
                    )
                    emit(P_DONE, "Complete!")
                    return result
                except Exception as e:
                    Log.e(self.TAG, f"Stage 3 failed: {e}")
                    # Fall through to standard return.

            emit(P_DONE, "Complete!")
            return formatted, num_channels

        except Exception as e:
            Log.e(self.TAG, f"Error during prediction: {e}")
            traceback.print_exc()
            return self._get_default_predictions(), 0
