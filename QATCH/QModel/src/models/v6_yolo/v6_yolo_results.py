# module: v6_yolo_results.py

"""QModel V6 YOLO pipeline — result dataclasses.

Every structured output produced by the pipeline stages is defined here
so that the individual stage modules have no circular imports.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2026-04-14
Version:
    7.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal, Optional


# ── Stage 1: Fill Replay ──────────────────────────────────────────────


@dataclass
class FillEndResult:
    """Output of Stage 1 (``FillReplaySimulator``)."""

    fill_end_idx: int
    """Row index where fill was *first* stably declared (start of stable run)."""
    fill_end_idx_buffered: int
    """Row corresponding to ``fill_end_time + buffer_seconds``."""
    fill_end_time: float
    """``Relative_time`` (seconds) at ``fill_end_idx``."""
    fill_end_time_buffered: float
    """``fill_end_time + buffer_seconds`` — cascade starts here."""
    confidence_at_end: float
    """Top-1 classifier confidence at the stable crossing frame."""
    n_frames_evaluated: int
    """Total snapshots classified (diagnostic)."""
    all_confidences: list[float] = field(default_factory=list)


# ── Stage 2: POI Scout ────────────────────────────────────────────────


@dataclass
class POIResult:
    """A single detected POI harvested from ``predict_single`` output."""

    class_id: int
    time: float
    """Timestamp in seconds (Relative_time domain)."""
    row_idx_absolute: int
    """Absolute row index resolved against ``raw_df``."""
    confidence: float


@dataclass
class POIScoutResult:
    """Output of Stage 2 (``POIScout``) — early POI1/POI2 detection.

    Contains the scouted POIs, the time-based slice boundaries, and the
    routing decision (standard vs. extended detector weights).
    """

    slice_start_time: float
    """``Relative_time`` where the scout window begins (= ``fill_end_time``)."""
    slice_end_time: float
    """``Relative_time`` where the scout window ends."""
    poi1: Optional[POIResult]
    poi2: Optional[POIResult]
    detector_set: Literal["standard", "extended"]
    """Which weight bank the cascade should use."""

    @property
    def delta_time(self) -> Optional[float]:
        """POI1->POI2 interval in seconds; ``None`` if either is missing."""
        if self.poi1 is not None and self.poi2 is not None:
            return self.poi2.time - self.poi1.time
        return None

    @property
    def both_detected(self) -> bool:
        return self.poi1 is not None and self.poi2 is not None


# ── Stage 3: Assembled result ─────────────────────────────────────────


@dataclass
class ExtendedFillResult:
    """Full output of the extended-fill pipeline.

    Wraps the standard ``(predictions, num_channels)`` return and adds
    extended-analysis fields.

    ``__iter__`` is implemented so callers that unpack a two-tuple keep
    working without modification::

        # backward-compatible
        predictions, num_channels = model.predict(...)

        # extended access
        result = model.predict(...)
        if isinstance(result, ExtendedFillResult):
            print(result.detector_set, result.eta_proxy)
    """

    predictions: dict
    num_channels: int
    fill_end: Optional[FillEndResult]
    scout: Optional[POIScoutResult]
    eta_proxy: Optional[float]
    """Viscosity proxy (POI1->POI2 time-delta in seconds)."""
    is_extended_fill: bool
    detector_set: Literal["standard", "extended"]

    def __iter__(self) -> Iterator:
        """Yield ``(predictions, num_channels)`` for backward-compat unpacking."""
        yield self.predictions
        yield self.num_channels

    @property
    def ready_for_extended_detectors(self) -> bool:
        return self.is_extended_fill and self.fill_end is not None
