# module: v6_yolo_poi_scout.py

"""QModel V6 YOLO pipeline — Stage 2: early POI1/POI2 detection.

After Stage 1 locates the fill-end time, ``POIScout`` creates a bounded
``Relative_time`` window guaranteed to contain POI1 and POI2 for normal
fills, runs the init detector on that window, and measures the POI1->POI2
time delta to decide which weight bank (standard vs. ERD) the cascade
should load.

This runs *before* the cascade so the routing decision is available
before any channel detectors fire.

Window construction
-------------------
The scout window spans::

    [fill_end_time,  fill_end_time + scout_window_seconds]

``scout_window_seconds`` should be set large enough that POI2 is
captured even for the slowest normal-fill runs.  For extended-fill
(high-viscosity) samples POI2 will still be captured — its late arrival
is exactly what triggers ERD routing.

If the window extends past the run, the slice is clamped to the last
available row.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2026-04-14
Version:
    7.0.0
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from QATCH.QModel.src.models.v6_yolo.v6_yolo_config import resolve_time_column
from QATCH.QModel.src.models.v6_yolo.v6_yolo_logging import Log, TAG_SCOUT
from QATCH.QModel.src.models.v6_yolo.v6_yolo_results import FillEndResult, POIResult, POIScoutResult

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from v6_yolo_models import QModelV6YOLO_Detector


class POIScout:
    """Early POI1/POI2 detection on a time-bounded post-fill window.

    The scout slice is built from ``Relative_time`` boundaries — not
    row indices — so the window adapts naturally to different sampling
    rates.

    Routing rule (simple threshold; replace with a calibrated classifier
    when labelled data is available)::

        delta_time = POI2.time − POI1.time
        detector_set = "extended"  if delta_time ≥ threshold  else "standard"

    If either POI is missing, routing defaults to ``"standard"``.
    """

    TAG = TAG_SCOUT

    def __init__(
        self,
        init_detector_loader: Callable[[], Optional["QModelV6YOLO_Detector"]],
        scout_window_seconds: float = 60.0,
        extended_threshold_seconds: float = 15.0,
    ) -> None:
        """
        Args:
            init_detector_loader: Zero-arg callable returning the cached
                init detector instance.  Deferred so no weights load
                until ``scout()`` is called.
            scout_window_seconds: Length of the ``Relative_time`` window
                after ``fill_end_time`` to slice for POI detection.
            extended_threshold_seconds: POI1->POI2 time-delta threshold
                for the standard / extended routing decision.
        """
        self._det_loader = init_detector_loader
        self._window_sec = scout_window_seconds
        self._ext_thresh_sec = extended_threshold_seconds

    # ── Public API ────────────────────────────────────────────────────

    def scout(
        self,
        master_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        fill_end: FillEndResult,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> POIScoutResult:
        """Run early POI1/POI2 detection and return routing decision.

        Args:
            master_df: Preprocessed DataFrame (full run).
            raw_df: Original unprocessed DataFrame (for absolute index
                resolution).
            fill_end: Stage 1 output — provides ``fill_end_time``.
            progress_callback: ``(int) -> None`` receiving 0–100.

        Returns:
            ``POIScoutResult`` with detected POIs and ``detector_set``.
        """
        if progress_callback is not None:
            progress_callback(0)

        col_time = resolve_time_column(master_df)
        start_time = fill_end.fill_end_time
        end_time = start_time + self._window_sec

        # ── Build time-bounded slice ──────────────────────────────────
        time_vals = master_df[col_time].to_numpy(dtype=float)
        mask = (time_vals >= start_time) & (time_vals <= end_time)
        scout_slice = master_df.loc[mask].copy()

        if scout_slice.empty or len(scout_slice) < 20:
            Log.w(
                self.TAG,
                f"Scout window [{start_time:.2f}s, {end_time:.2f}s] "
                f"contains only {len(scout_slice)} rows — too short.",
            )
            if progress_callback is not None:
                progress_callback(100)
            return self._empty_result(start_time, end_time)

        Log.i(
            self.TAG,
            f"Scout window: [{start_time:.2f}s, {end_time:.2f}s]  " f"({len(scout_slice)} rows)",
        )

        if progress_callback is not None:
            progress_callback(30)

        # ── Run init detector on the scout slice ──────────────────────
        det = self._det_loader()
        if det is None:
            Log.e(self.TAG, "Init detector unavailable for scouting.")
            if progress_callback is not None:
                progress_callback(100)
            return self._empty_result(start_time, end_time)

        res: Dict[int, Dict[str, Any]] = det.predict_single(
            scout_slice, target_class_map={0: 1, 1: 2}
        )

        if progress_callback is not None:
            progress_callback(70)

        # ── Build POIResults ──────────────────────────────────────────
        poi1 = self._build_poi(res, poi_id=1, raw_df=raw_df)
        poi2 = self._build_poi(res, poi_id=2, raw_df=raw_df)

        # ── Routing decision ──────────────────────────────────────────
        detector_set = self._route(poi1, poi2)

        if progress_callback is not None:
            progress_callback(100)

        Log.i(
            self.TAG,
            f"POI1={'N/A' if poi1 is None else f'{poi1.time:.2f}s'}  "
            f"POI2={'N/A' if poi2 is None else f'{poi2.time:.2f}s'}  "
            f"delta={self._delta_time(poi1, poi2)}  "
            f"-> {detector_set}",
        )

        return POIScoutResult(
            slice_start_time=start_time,
            slice_end_time=end_time,
            poi1=poi1,
            poi2=poi2,
            detector_set=detector_set,
        )

    # ── Internals ─────────────────────────────────────────────────────

    def _build_poi(
        self,
        res: Dict[int, Dict[str, Any]],
        poi_id: int,
        raw_df: pd.DataFrame,
    ) -> Optional[POIResult]:
        """Convert a ``predict_single`` entry into a ``POIResult``."""
        if poi_id not in res:
            return None
        entry = res[poi_id]
        raw_idx = self._get_raw_index(raw_df, entry["time"])
        return POIResult(
            class_id=poi_id,
            time=entry["time"],
            row_idx_absolute=raw_idx,
            confidence=entry["conf"],
        )

    def _route(
        self,
        poi1: Optional[POIResult],
        poi2: Optional[POIResult],
    ) -> str:
        """Decide ``"standard"`` or ``"extended"`` based on time delta."""
        if poi1 is None or poi2 is None:
            Log.w(
                self.TAG,
                "Incomplete POIs — cannot route, defaulting to standard.",
            )
            return "standard"

        delta = poi2.time - poi1.time
        if delta >= self._ext_thresh_sec:
            return "extended"
        return "standard"

    @staticmethod
    def _delta_time(poi1: Optional[POIResult], poi2: Optional[POIResult]) -> str:
        if poi1 is not None and poi2 is not None:
            return f"{poi2.time - poi1.time:.2f}s"
        return "N/A"

    @staticmethod
    def _get_raw_index(raw_df: pd.DataFrame, target_time: float) -> int:
        col_time = resolve_time_column(raw_df)
        times = raw_df[col_time].to_numpy(dtype=float)
        idx = int(np.abs(times - target_time).argmin())
        return int(raw_df.index[idx])

    def _empty_result(self, start_time: float, end_time: float) -> POIScoutResult:
        return POIScoutResult(
            slice_start_time=start_time,
            slice_end_time=end_time,
            poi1=None,
            poi2=None,
            detector_set="standard",
        )
