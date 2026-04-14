# module: v6_yolo_routing.py

"""QModel V6 YOLO pipeline — Stage 3: result assembly.

``ExtendedFillAnalyzer`` packages the outputs of Stages 1–2 and the
cascade into a single ``ExtendedFillResult``.

The actual routing decision (standard vs. extended weights) is now made
in Stage 2 (``POIScout``) *before* the cascade runs.  This module
simply carries the decision forward and assembles the unified result.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2026-04-14
Version:
    7.0.0
"""

from __future__ import annotations

from typing import Literal, Optional

from QATCH.QModel.src.models.v6_yolo.v6_yolo_logging import Log, TAG_ROUTE
from QATCH.QModel.src.models.v6_yolo.v6_yolo_results import (
    ExtendedFillResult,
    FillEndResult,
    POIScoutResult,
)


class ExtendedFillAnalyzer:
    """Assembles the final ``ExtendedFillResult`` from pipeline outputs.

    The routing decision (``detector_set``) arrives from
    ``POIScoutResult``; this class adds the viscosity proxy and produces
    a loggable summary.
    """

    TAG = TAG_ROUTE

    def build_result(
        self,
        fill_end: Optional[FillEndResult],
        scout: Optional[POIScoutResult],
        predictions: dict,
        num_channels: int,
    ) -> ExtendedFillResult:
        """Package all stage outputs into an ``ExtendedFillResult``.

        Args:
            fill_end: Stage 1 output (may be ``None``).
            scout: Stage 2 output (may be ``None`` if scouting failed).
            predictions: Formatted output dict from the cascade.
            num_channels: Channel count from fill classification.

        Returns:
            Fully-populated ``ExtendedFillResult``.
        """
        # Pull routing and eta from the scout result.
        if scout is not None:
            detector_set: Literal["standard", "extended"] = (
                "extended" if scout.detector_set == "extended" else "standard"
            )
            eta_proxy = scout.delta_time  # seconds-based (or None)
            is_extended = scout.detector_set == "extended"
        else:
            detector_set = "standard"
            eta_proxy = None
            is_extended = False

        # ── Diagnostic log ────────────────────────────────────────────
        fe_str = (
            f"fill_end_idx={fill_end.fill_end_idx} " f"(t={fill_end.fill_end_time:.2f}s)"
            if fill_end is not None
            else "fill_end=N/A"
        )
        poi1_str = (
            f"{scout.poi1.time:.2f}s" if scout is not None and scout.poi1 is not None else "N/A"
        )
        poi2_str = (
            f"{scout.poi2.time:.2f}s" if scout is not None and scout.poi2 is not None else "N/A"
        )
        Log.i(
            self.TAG,
            f"{fe_str} | "
            f"POI1={poi1_str} POI2={poi2_str} | "
            f"eta={eta_proxy if eta_proxy is not None else float('nan'):.2f}s | "
            f"set={detector_set}",
        )

        return ExtendedFillResult(
            predictions=predictions,
            num_channels=num_channels,
            fill_end=fill_end,
            scout=scout,
            eta_proxy=eta_proxy,
            is_extended_fill=is_extended,
            detector_set=detector_set,
        )
