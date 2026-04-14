# module: v6_yolo_fill_replay.py

"""QModel V6 YOLO pipeline — Stage 1: fill-end localisation.

``FillReplaySimulator`` replays the fill classifier over growing
snapshots of ``master_df`` to pinpoint the transition row where the
sensor is stably filled, before the cascade ever runs.

Key fixes over the previous implementation
-------------------------------------------
1.  **Correct fill-end index**: records the *first* frame of the stable
    confidence run, not the *last*.  The old code set
    ``fill_end_idx = end_row`` at the moment the stable count crossed
    the threshold, which was ``(duration_threshold - 1) * step_rows``
    past the actual transition.

2.  **Time-based buffer**: ``buffer_seconds`` (in ``Relative_time``
    units) replaces the former ``buffer_rows``.  The buffered row index
    is resolved by finding the nearest row whose ``Relative_time`` is
    ≥ ``fill_end_time + buffer_seconds``.  This is invariant to
    sampling-rate changes across different runs.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2026-04-14
Version:
    7.0.0
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from QATCH.QModel.src.models.v6_yolo.v6_yolo_config import resolve_time_column
from QATCH.QModel.src.models.v6_yolo.v6_yolo_logging import Log, TAG_REPLAY
from QATCH.QModel.src.models.v6_yolo.v6_yolo_results import FillEndResult

# Forward reference — resolved at runtime via the loader callable.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from v6_yolo_models import QModelV6YOLO_FillClassifier


class FillReplaySimulator:
    """Replays the fill classifier over a buffered run to pinpoint the
    first frame where fill is confidently and stably declared.

    Mirrors the live classifier loop using ``predict_confidence()`` on
    growing snapshots of ``master_df``.

    Fill-detected condition::

        top-1 class ≠ "no_fill"  AND  confidence ≥ threshold
        held for ``duration_threshold_frames`` consecutive snapshots.

    The *first* snapshot in that stable run is recorded as
    ``fill_end_idx`` (not the last, which was the previous bug).
    """

    TAG = TAG_REPLAY

    def __init__(
        self,
        fill_cls_loader: Callable[[], Optional["QModelV6YOLO_FillClassifier"]],
        step_rows: int = 10,
        min_start_rows: int = 50,
    ) -> None:
        """
        Args:
            fill_cls_loader: Zero-argument callable returning the cached
                ``QModelV6YOLO_FillClassifier``.  Weight loading is
                deferred until ``find_fill_end()`` is first called.
            step_rows: Rows between classifier evaluations.
            min_start_rows: Skip classification until the snapshot has at
                least this many rows (avoids startup noise).
        """
        self._clf_loader = fill_cls_loader
        self._step = step_rows
        self._min_start = min_start_rows

    # ── Public API ────────────────────────────────────────────────────

    def find_fill_end(
        self,
        master_df: pd.DataFrame,
        confidence_threshold: float = 0.85,
        duration_threshold_frames: int = 5,
        buffer_seconds: float = 5.0,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Optional[FillEndResult]:
        """Iterate through ``master_df`` and detect the first stable fill.

        Args:
            master_df: Preprocessed run DataFrame.
            confidence_threshold: Minimum top-1 confidence for "filled".
            duration_threshold_frames: Consecutive snapshots required.
            buffer_seconds: Time (in ``Relative_time`` seconds) added
                past the confirmed fill-end row.  The cascade starts
                from the row nearest to
                ``fill_end_time + buffer_seconds``.
            progress_callback: ``(int) -> None`` receiving 0–100.

        Returns:
            ``FillEndResult`` if fill was stably detected; ``None``
            otherwise.
        """
        col_time = resolve_time_column(master_df)
        clf = self._clf_loader()
        if clf is None:
            Log.e(self.TAG, "Stage 1: fill classifier unavailable.")
            return None

        confidences: List[float] = []
        stable_count: int = 0
        stable_start_row: Optional[int] = None  # first row of current stable run
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
                Log.w(self.TAG, f"Classifier error at row {end_row}: {exc}")
                confidences.append(0.0)
                stable_count = 0
                stable_start_row = None
                continue

            is_filled = num_ch != -1
            effective_conf = conf if is_filled else 0.0
            confidences.append(effective_conf)

            if effective_conf >= confidence_threshold:
                if stable_count == 0:
                    # ── Mark the START of the stable run ──
                    stable_start_row = end_row
                stable_count += 1

                if stable_count >= duration_threshold_frames:
                    # Record the first frame of the run, not the last.
                    fill_end_idx = stable_start_row
                    fill_end_conf = effective_conf
                    Log.d(
                        self.TAG,
                        f"Fill end at row {fill_end_idx} "
                        f"(num_ch={num_ch}, conf={conf:.3f}, "
                        f"stable_run={stable_count}).",
                    )
                    break
            else:
                stable_count = 0
                stable_start_row = None

        if fill_end_idx is None:
            Log.w(
                self.TAG,
                f"Fill not detected across {len(confidences)} snapshots.",
            )
            return None

        # ── Time-based buffer ─────────────────────────────────────────
        fill_end_time = float(master_df.iloc[fill_end_idx][col_time])
        buffered_time = fill_end_time + buffer_seconds
        time_vals = master_df[col_time].to_numpy(dtype=float)

        # Find the first row whose Relative_time ≥ buffered_time.
        candidates = np.where(time_vals >= buffered_time)[0]
        if len(candidates) > 0:
            buffered_idx = int(candidates[0])
        else:
            buffered_idx = len(master_df) - 1

        Log.i(
            self.TAG,
            f"fill_end_time={fill_end_time:.2f}s  "
            f"buffer={buffer_seconds}s  "
            f"buffered_time={buffered_time:.2f}s  "
            f"buffered_idx={buffered_idx}",
        )

        return FillEndResult(
            fill_end_idx=fill_end_idx,
            fill_end_idx_buffered=buffered_idx,
            fill_end_time=fill_end_time,
            fill_end_time_buffered=buffered_time,
            confidence_at_end=fill_end_conf,
            n_frames_evaluated=len(confidences),
            all_confidences=confidences,
        )
