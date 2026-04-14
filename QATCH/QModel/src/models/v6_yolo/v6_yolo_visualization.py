# module: v6_yolo_visualization.py

"""QModel V6 YOLO pipeline — debug visualisation helpers.

Extracted from the monolith so the controller stays lean.  Produces
timestamped PNG debug plots of the cascade detections, cut points,
and (optionally) the fill-end and scout-window markers.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2026-04-14
Version:
    7.0.0
"""

from __future__ import annotations

import datetime
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from QATCH.QModel.src.models.v6_yolo.v6_yolo_logging import Log, TAG_VIS
from QATCH.QModel.src.models.v6_yolo.v6_yolo_results import FillEndResult, POIScoutResult

# POI-ID -> display name (mirrors QModelV6YOLO.POI_MAP)
_POI_MAP = {1: "POI1", 2: "POI2", 3: "POI3", 4: "POI4", 5: "POI5", 6: "POI6"}
_POI_COLORS = {1: "green", 2: "blue", 4: "orange", 5: "red", 6: "purple"}


def save_debug_plot(
    df: pd.DataFrame,
    results: Dict[int, Dict[str, Any]],
    cut_history: List[Tuple[str, float]],
    save_path: str = "v6_debug.png",
    fill_end: Optional[FillEndResult] = None,
    scout: Optional[POIScoutResult] = None,
) -> None:
    """Save a timestamped debug plot of cascade detections.

    Args:
        df: Preprocessed master DataFrame (full run).
        results: ``{poi_id: {"time": float, "conf": float, ...}}``.
        cut_history: ``[(label, cut_time), ...]`` from the cascade.
        save_path: Base file path; a timestamp suffix is appended.
        fill_end: Stage 1 output — adds fill-end + buffer markers.
        scout: Stage 2 output — adds scout-window shading.
    """
    if df is None or df.empty:
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name, ext = os.path.splitext(save_path)
    final_save_path = f"{base_name}_{timestamp}{ext}"

    time_col = df["Relative_time"].values
    signal = df["Dissipation"].values if "Dissipation" in df.columns else df.iloc[:, 1].values

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_col, signal, color="gray", alpha=0.6, label="Raw Signal")

    # ── Fill end + buffer markers ─────────────────────────────────────
    if fill_end is not None:
        ax.axvline(
            x=fill_end.fill_end_time,
            color="cyan",
            linestyle="--",
            linewidth=1.5,
            label=f"Fill End ({fill_end.fill_end_time:.1f}s)",
        )
        ax.axvline(
            x=fill_end.fill_end_time_buffered,
            color="cyan",
            linestyle=":",
            linewidth=1,
            alpha=0.6,
            label=f"Buffer End ({fill_end.fill_end_time_buffered:.1f}s)",
        )

    # ── Scout window shading ──────────────────────────────────────────
    if scout is not None:
        ax.axvspan(
            scout.slice_start_time,
            scout.slice_end_time,
            color="yellow",
            alpha=0.08,
            label=f"Scout Window [{scout.slice_start_time:.1f}–{scout.slice_end_time:.1f}s]",
        )

    # ── POI markers ───────────────────────────────────────────────────
    for poi_id, data in results.items():
        if data and "time" in data:
            c = _POI_COLORS.get(poi_id, "black")
            name = _POI_MAP.get(poi_id, f"POI{poi_id}")
            ax.axvline(x=data["time"], color=c, linestyle="-", linewidth=2, label=name)

    # ── Cascade cut shading ───────────────────────────────────────────
    for _, cut_time in cut_history:
        ax.axvline(x=cut_time, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax.axvspan(cut_time, np.max(time_col), color="red", alpha=0.05)

    ax.set_title(f"Cascade Detection Debug — {len(cut_history)} slices applied")
    ax.set_xlabel("Relative_time (s)")
    ax.set_ylabel("Signal")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(final_save_path, dpi=150)
    plt.close(fig)
    Log.i(TAG_VIS, f"Debug plot saved to: {final_save_path}")
