# module: v6_yolo_config.py

"""QModel V6 YOLO pipeline — configuration constants.

Centralises image dimensions, classifier settings, fill class mappings,
and progress-bar milestone tables used by the controller and sub-modules.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2026-04-14
Version:
    7.0.0
"""

from __future__ import annotations

from typing import Dict


class QModelV6Config:
    """Configuration constants for the QModel V6 YOLO pipeline."""

    # ── Detector image settings ────────────────────────────────────────
    IMG_WIDTH: int = 2560
    IMG_HEIGHT: int = 384
    MIN_SLICE_LENGTH: int = 20
    CONF_THRESHOLD: float = 0.01

    # ── Fill classifier image settings ─────────────────────────────────
    FILL_INFERENCE_W: int = 224
    FILL_INFERENCE_H: int = 224
    FILL_GEN_W: int = 640
    FILL_GEN_H: int = 640

    # Maps YOLO classification labels -> channel count.
    FILL_CLASS_MAP: Dict[str, int] = {
        "no_fill": -1,
        "initial_fill": 0,
        "1ch": 1,
        "2ch": 2,
        "3ch": 3,
    }

    # ── Progress-bar milestones (percent) ──────────────────────────────
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
    # EXTENDED mode (Stage 1 + Stage 2 run before cascade):
    #    5  Data loaded
    #   10  Preprocessed
    #   10->35  Stage 1 replay
    #   35->45  Stage 2 POI scout
    #   48  Fill classified
    #   57  Ch3 done
    #   66  Ch2 done
    #   75  Ch1 done
    #   84  Init done
    #   89  Fine adjustment done
    #   94  Stage 3 result packaging
    #  100  Complete

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
    EXT_REPLAY_LO = 10
    EXT_REPLAY_HI = 35
    EXT_SCOUT_LO = 35
    EXT_SCOUT_HI = 45
    EXT_CLASSIFY = 48
    EXT_CH3 = 57
    EXT_CH2 = 66
    EXT_CH1 = 75
    EXT_INIT = 84
    EXT_FINE = 89
    EXT_STAGE3 = 94
    EXT_DONE = 100


# ── Shared time-column resolver ───────────────────────────────────────


def resolve_time_column(df) -> str:
    """Return the best available time-column name from *df*."""
    if "Relative_time" in df.columns:
        return "Relative_time"
    if "time" in df.columns:
        return "time"
    return df.columns[0]
