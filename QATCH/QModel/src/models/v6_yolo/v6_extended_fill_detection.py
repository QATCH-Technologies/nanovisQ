# module: extended_fill_analyzer.py

"""Post-hoc extended fill analysis pipeline for QModel V6 YOLO.

Designed as a preprocessing pass over a fully-buffered run — does not touch the
live collection process.  Extends the output of ``QModelV6YOLO.predict()`` with:

- ``fill_end_idx``   — row in master_df where fill is first confidently declared.
- ``poi1``, ``poi2`` — detected via the init.pt detector on the post-fill slice.
- ``eta_proxy``      — viscosity surrogate derived from the POI1→POI2 interval.
- ``is_extended_fill`` / ``detector_set`` — the routing decision for Stage 4.

All three stage classes are injected with objects that already exist inside
``QModelV6YOLO``, so no new model files are required for Stages 1-3.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Avoid a hard import cycle — v6_yolo imports us, we only need these for
    # type hints inside TYPE_CHECKING so they're erased at runtime.
    from v6_yolo import QModelV6YOLO_FillClassifier, QModelV6YOLO_Detector
try:
    from QATCH.common.logger import (
        Logger as Log,  # pyright: ignore[reportPrivateImportUsage]
    )

except ImportError:

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


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
TAG = "[QModelV6YOLO (ERD)]"


@dataclass
class FillEndResult:
    """Output of Stage 1 (FillReplaySimulator)."""

    fill_end_idx: int  # Row index in master_df where fill is stably declared
    fill_end_idx_buffered: int  # fill_end_idx + buffer — start of the Stage 2 slice
    fill_end_time: float  # Corresponding Relative_time value (for diagnostics)
    confidence_at_end: float  # Classifier confidence at the stable crossing frame
    n_frames_evaluated: int  # How many snapshots were classified (diagnostic)
    all_confidences: list[float] = field(default_factory=list)


@dataclass
class POIResult:
    """A single detected POI returned by the init detector."""

    class_id: int
    time: float  # Timestamp in seconds (from predict_single)
    row_idx_absolute: int  # Absolute row index resolved in raw_df
    confidence: float


@dataclass
class InitialFillPOIs:
    """Output of Stage 2 (InitialFillSlicer)."""

    slice_start_idx: int  # Row in master_df where the slice began

    poi1: Optional[POIResult]  # Mapped from YOLO class 0 → POI ID 1
    poi2: Optional[POIResult]  # Mapped from YOLO class 1 → POI ID 2

    @property
    def delta_rows(self) -> Optional[int]:
        """POI1→POI2 interval in raw_df rows. None if either POI is missing."""
        if self.poi1 is not None and self.poi2 is not None:
            return self.poi2.row_idx_absolute - self.poi1.row_idx_absolute
        return None

    @property
    def delta_time(self) -> Optional[float]:
        """POI1→POI2 interval in seconds. None if either POI is missing."""
        if self.poi1 is not None and self.poi2 is not None:
            return self.poi2.time - self.poi1.time
        return None

    @property
    def both_detected(self) -> bool:
        return self.poi1 is not None and self.poi2 is not None


@dataclass
class ExtendedFillResult:
    """
    Full output of ExtendedFillAnalyzer.

    Wraps the existing ``predict()`` return values and adds the three
    extended-fill analysis stages so callers receive everything in one object.
    """

    # Passthrough from predict()
    predictions: dict
    num_channels: int

    # Stage 1
    fill_end: Optional[FillEndResult]

    # Stage 2
    pois: Optional[InitialFillPOIs]

    # Stage 3
    eta_proxy: Optional[float]  # Viscosity proxy (row-delta, or calibrated units)
    is_extended_fill: bool
    detector_set: Literal["standard", "extended"]

    @property
    def ready_for_extended_detectors(self) -> bool:
        return self.is_extended_fill and self.fill_end is not None


# ---------------------------------------------------------------------------
# Stage 1: Fill replay simulator
# ---------------------------------------------------------------------------


class FillReplaySimulator:
    """
    Replays the fill classifier over a fully-buffered run to locate the first
    frame where fill is confidently and stably declared.

    Uses ``QModelV6YOLO_FillClassifier.predict_confidence()`` on growing
    snapshots of ``master_df``, mirroring the live classifier loop without
    altering the live predictor state.

    The "fill detected" condition mirrors the live logic:
        top-1 class is not "no_fill"  AND  confidence >= threshold
        held for ``duration_threshold_frames`` consecutive frames.
    """

    def __init__(
        self,
        fill_classifier: "QModelV6YOLO_FillClassifier",
        step_rows: int = 10,
        min_start_rows: int = 50,
    ):
        """
        Args:
            fill_classifier: Loaded ``QModelV6YOLO_FillClassifier`` instance —
                the same object used by ``QModelV6YOLO.predict()``.
            step_rows: Number of new rows between classifier calls.  Set this
                to match the live update cadence so the replay sees the same
                sequence of snapshots.
            min_start_rows: Skip classification until master_df has at least
                this many rows (avoids false positives on empty-run noise).
        """
        self._clf = fill_classifier
        self._step = step_rows
        self._min_start = min_start_rows

    def find_fill_end(
        self,
        master_df: pd.DataFrame,
        confidence_threshold: float = 0.85,
        duration_threshold_frames: int = 5,
        buffer_rows: int = 50,
    ) -> Optional[FillEndResult]:
        """
        Iterates through ``master_df`` in steps, calls the fill classifier on
        each growing snapshot, and detects the first stable fill crossing.

        Args:
            master_df: Preprocessed run DataFrame (output of
                ``QModelV6YOLO_DataProcessor.preprocess_dataframe()``).
            confidence_threshold: Minimum classifier confidence for "filled".
            duration_threshold_frames: Consecutive frames required to confirm.
            buffer_rows: Extra rows added past the confirmed fill end before
                the Stage 2 slice begins — lets the fill transition settle.

        Returns:
            ``FillEndResult``, or ``None`` if fill was never confidently detected.
        """
        col_time = "Relative_time" if "Relative_time" in master_df.columns else master_df.columns[0]

        confidences: list[float] = []
        stable_count: int = 0
        fill_end_idx: Optional[int] = None
        fill_end_conf: float = 0.0

        snapshot_ends = range(self._min_start, len(master_df), self._step)

        for end_row in snapshot_ends:
            snapshot = master_df.iloc[:end_row]

            try:
                num_ch, conf = self._clf.predict_confidence(snapshot)
            except Exception as exc:
                Log.w(TAG, f"Classifier error at row {end_row}: {exc}")
                confidences.append(0.0)
                stable_count = 0
                continue

            # "Filled" = any class other than no_fill (channel count != -1)
            is_filled = num_ch != -1
            effective_conf = conf if is_filled else 0.0
            confidences.append(effective_conf)

            if effective_conf >= confidence_threshold:
                stable_count += 1
                if stable_count >= duration_threshold_frames:
                    fill_end_idx = end_row
                    fill_end_conf = effective_conf
                    Log.d(
                        TAG,
                        f"Fill end at row {fill_end_idx} "
                        f"(num_ch={num_ch}, conf={conf}, stable={stable_count} frames).",
                    )
                    break
            else:
                stable_count = 0

        if fill_end_idx is None:
            Log.w(TAG, f"Fill not detected across {len(confidences)} frames.")
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
# Stage 2: Initial fill slicer + POI extractor
# ---------------------------------------------------------------------------


class InitialFillSlicer:
    """
    Slices ``master_df`` starting at ``fill_end_idx_buffered`` and runs the
    init.pt detector to extract POI1 and POI2 positions.

    Mirrors the live call exactly::

        res = det_init.predict_single(current_df, target_class_map={0: 1, 1: 2})
        process_detection(res, 1)
        process_detection(res, 2)

    ``predict_single`` returns time-domain detections; absolute row indices are
    resolved against ``raw_df`` using the same nearest-neighbour logic as
    ``QModelV6YOLO._get_raw_index()``.
    """

    _TARGET_CLASS_MAP = {0: 1, 1: 2}

    def __init__(self, det_init: "QModelV6YOLO_Detector"):
        self._det = det_init

    def extract_pois(
        self,
        master_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        fill_end_result: FillEndResult,
    ) -> InitialFillPOIs:
        """
        Slices master_df from fill_end_idx_buffered onward and runs the init
        detector.  POI positions are returned as absolute row indices in raw_df.

        Args:
            master_df: Preprocessed run DataFrame.
            raw_df: Original unprocessed DataFrame — used for index resolution.
            fill_end_result: Output of Stage 1.
        """
        start = fill_end_result.fill_end_idx_buffered
        slice_df = master_df.iloc[start:].reset_index(drop=True)

        if slice_df.empty:
            Log.w(TAG, f"empty slice after buffered fill end (row={start}).")
            return InitialFillPOIs(slice_start_idx=start, poi1=None, poi2=None)

        # Same call as the live predict() cascade
        res = self._det.predict_single(slice_df, target_class_map=self._TARGET_CLASS_MAP)

        poi1 = self._resolve_poi(res, class_id=1, raw_df=raw_df)
        poi2 = self._resolve_poi(res, class_id=2, raw_df=raw_df)

        if poi1 is None:
            Log.w(TAG, "POI1 not detected in slice.")
        if poi2 is None:
            Log.w(TAG, "POI2 not detected in slice.")

        return InitialFillPOIs(slice_start_idx=start, poi1=poi1, poi2=poi2)

    @staticmethod
    def _resolve_poi(
        det_result: dict,
        class_id: int,
        raw_df: pd.DataFrame,
    ) -> Optional[POIResult]:
        """
        Extracts the detection for ``class_id`` from a ``predict_single`` result
        and resolves its timestamp to an absolute row index in ``raw_df``.

        ``predict_single`` already maps normalized box x-coords to timestamps via
        ``x_norm * (x_max - x_min) + x_min`` — we just need the nearest-neighbour
        lookup that ``_get_raw_index`` performs, replicated here without needing
        a reference back to the controller instance.
        """
        if class_id not in det_result:
            return None

        t = float(det_result[class_id]["time"])
        conf = float(det_result[class_id]["conf"])

        # Replicate QModelV6YOLO._get_raw_index() logic
        col_time = (
            "Relative_time"
            if "Relative_time" in raw_df.columns
            else ("time" if "time" in raw_df.columns else raw_df.columns[0])
        )
        times = raw_df[col_time].to_numpy(dtype=float)
        row_abs = int(raw_df.index[int(np.abs(times - t).argmin())])

        return POIResult(
            class_id=class_id,
            time=t,
            row_idx_absolute=row_abs,
            confidence=conf,
        )


# ---------------------------------------------------------------------------
# Stage 3 + Orchestrator
# ---------------------------------------------------------------------------


class ExtendedFillAnalyzer:
    """
    Orchestrates Stages 1-3 as a post-hoc pass over a completed run.

    All injected objects (``fill_classifier``, ``det_init``) are the same
    instances already held by ``QModelV6YOLO`` — no new files are loaded.

    Typical usage (from inside ``QModelV6YOLO.predict()``)::

        extended = self._extended_analyzer.analyze(
            raw_df=raw_df,
            master_df=master_df,
            predictions=formatted_output,
            num_channels=num_channels,
        )
        return extended
    """

    DEFAULT_EXTENDED_THRESHOLD_ROWS: int = 200

    def __init__(
        self,
        fill_classifier: "QModelV6YOLO_FillClassifier",
        det_init: "QModelV6YOLO_Detector",
        # Stage 1 tuning
        step_rows: int = 10,
        min_start_rows: int = 50,
        fill_confidence_threshold: float = 0.85,
        fill_duration_threshold_frames: int = 5,
        fill_buffer_rows: int = 50,
        # Stage 3 tuning
        extended_threshold_rows: Optional[int] = None,
    ):
        self._replay = FillReplaySimulator(
            fill_classifier=fill_classifier,
            step_rows=step_rows,
            min_start_rows=min_start_rows,
        )
        self._slicer = InitialFillSlicer(det_init=det_init)

        self._fill_conf_thresh = fill_confidence_threshold
        self._fill_dur_thresh = fill_duration_threshold_frames
        self._fill_buffer = fill_buffer_rows
        self._ext_thresh = (
            extended_threshold_rows
            if extended_threshold_rows is not None
            else self.DEFAULT_EXTENDED_THRESHOLD_ROWS
        )

    def analyze(
        self,
        raw_df: pd.DataFrame,
        master_df: pd.DataFrame,
        predictions: dict,
        num_channels: int,
    ) -> ExtendedFillResult:
        """
        Runs all three stages and returns a fully-populated ``ExtendedFillResult``.

        Args:
            raw_df: Original unprocessed DataFrame — used for index resolution.
            master_df: Preprocessed DataFrame — used for classifier replay and slicing.
            predictions: The formatted output dict from ``QModelV6YOLO._format_output()``.
            num_channels: The channel count determined by the base predict() call.

        Returns:
            ``ExtendedFillResult`` with all stage outputs populated.
        """

        def _default(reason: str) -> ExtendedFillResult:
            Log.w(TAG, f"{reason} - defaulting to standard.")
            return ExtendedFillResult(
                predictions=predictions,
                num_channels=num_channels,
                fill_end=None,
                pois=None,
                eta_proxy=None,
                is_extended_fill=False,
                detector_set="standard",
            )

        # Stage 1 -------------------------------------------------------
        fill_end = self._replay.find_fill_end(
            master_df=master_df,
            confidence_threshold=self._fill_conf_thresh,
            duration_threshold_frames=self._fill_dur_thresh,
            buffer_rows=self._fill_buffer,
        )

        if fill_end is None:
            return _default("Stage 1 failed to detect fill end")

        # Stage 2 -------------------------------------------------------
        pois = self._slicer.extract_pois(
            master_df=master_df,
            raw_df=raw_df,
            fill_end_result=fill_end,
        )

        # Stage 3 -------------------------------------------------------
        eta_proxy, is_extended = self._classify(pois)

        detector_set: Literal["standard", "extended"] = "extended" if is_extended else "standard"

        Log.i(
            TAG,
            f"fill_end_idx={fill_end.fill_end_idx} (t={fill_end.fill_end_time}s) | "
            f"POI1={fill_end.fill_end_idx if pois.poi1 is None else pois.poi1.row_idx_absolute,} "
            f"POI2={'N/A' if pois.poi2 is None else pois.poi2.row_idx_absolute} | "
            f"delta={pois.delta_rows} rows | "
            f"eta={eta_proxy if eta_proxy is not None else float('nan')} "
            f"| set={detector_set}",
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

    def _classify(self, pois: InitialFillPOIs) -> tuple[Optional[float], bool]:
        """
        Stage 3: maps the POI1→POI2 row delta to a viscosity proxy and decides
        whether this is an extended fill scenario.

        Currently a simple threshold on ``delta_rows``.  Replace with a
        calibrated regression or lightweight classifier once labeled data
        has been analysed.

        Returns:
            ``(eta_proxy, is_extended_fill)``
        """
        if not pois.both_detected or pois.delta_rows is None:
            Log.w(TAG, "Incomplete POIs - cannot classify, defaulting standard.")
            return None, False

        eta_proxy = float(pois.delta_rows)  # placeholder; calibrate against known η values
        return eta_proxy, (pois.delta_rows >= self._ext_thresh)
