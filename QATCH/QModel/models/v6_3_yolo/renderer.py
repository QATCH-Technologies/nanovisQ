"""
renderer.py
===========

Part of the QModel V6 YOLO predictor subpackage (v6.3.0).
Formerly ``v7_render.py``. (The "v7"/"v2" here refers to the detection-image
RENDER VERSION dispatched by ``generate_det_image``, not a module version.)

Version-2 detection-image renderer, addressing the two representation
failures the v7 training run exposed:

  1. LATE-EVENT FLATTENING. Per-strip global percentile normalization makes
     the early fill step own the entire dynamic range; late POIs in long
     viscous runs become featureless flat plateaus — the regions where ch2
     trained to recall 0.63 and ch3 collapsed. The fix is not a cleverer
     amplitude normalization (any global value scaling keeps the step
     dominant): it is to render what the detector is actually supposed to
     find. POIs are TRANSITIONS, so the third strip is replaced with a
     DERIVATIVE-ENERGY trace: the combined, robustly-scaled, log-compressed
     |d/dt| of dissipation and resonance frequency. Events appear as bright
     vertical ridges with near-uniform salience regardless of where in the
     run they occur or how large the absolute amplitude change is. The
     Difference curve it replaces is a linear combination of the two value
     strips and carried little independent information.

  2. The dissipation (R) and resonance (G) value strips are kept exactly as
     v1 renders them (same percentile normalization, fill + white outline),
     so global fill-context cues the detectors already exploit are
     preserved.

Train/inference contract: this module is used by BOTH build_dataset.py and
the production predictor (QModelV6Config.RENDER_VERSION). The render the
weights were trained on MUST be the render they see at inference; the
version flag exists precisely so old (v1-trained) weights keep working
while v2-trained weights roll out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cv2

try:
    from .dataprocessor import DataProcessor as DP
except ImportError:  # flat / headless execution
    from dataprocessor import DataProcessor as DP

COL_TIME = "Relative_time"
COL_DISS = "Dissipation"
COL_FREQ = "Resonance_Frequency"

PADDING = DP.PADDING
IMG_CHANNELS = 3
COLOR_WHITE = (255, 255, 255)

# Derivative-energy parameters.
DERIV_SCALES_S = (0.25, 1.0, 4.0)  # half-window timescales for transitions
DERIV_SMOOTH_S = 0.15  # post-smoothing of the salience trace
DERIV_UPPER_PCT = 99.8  # robust ceiling for ridge normalization
DERIV_EPS = 1e-12


def _robust_mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _scaled_curv(x: np.ndarray, w: int) -> np.ndarray:
    """|x[i+w] - 2x[i] + x[i-w]|: windowed second difference, same length
    (edges clamped). Fires on steps AND on slope changes, and is inherently
    insensitive to a linear trend — which is what distinguishes a late-fill
    POI (a bend on a monotone background) from the background itself."""
    n = len(x)
    out = np.zeros(n)
    if n <= 2 * w:
        return out
    out[w:-w] = np.abs(x[2 * w :] - 2.0 * x[w:-w] + x[: -2 * w])
    out[:w] = out[w]
    out[-w:] = out[-w - 1]
    return out


def derivative_energy(df: pd.DataFrame) -> np.ndarray:
    """Multi-scale transition salience: for each signal and each timescale
    (~0.25 s / 1 s / 4 s), the windowed SECOND difference is normalized by
    its own MAD; the per-sample maximum across signals and scales is
    log-compressed and lightly smoothed. Fast init events and slow viscous
    transitions both appear as ridges of comparable salience, because each
    scale is normalized against ITS OWN noise floor — the property a
    single-sample gradient lacks (at the 5 ms resample grid it is pure
    noise) and a global value normalization lacks (the early fill step owns
    the dynamic range)."""
    n = len(df)
    if n < 16:
        return np.zeros(n)
    t = pd.to_numeric(df[COL_TIME], errors="coerce").to_numpy(dtype=float)
    dt = float(np.nanmedian(np.diff(t))) or 0.005
    sal = np.zeros(n)
    for col in (COL_DISS, COL_FREQ):
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
        for scale_s in DERIV_SCALES_S:
            w = int(round(scale_s / dt))
            w = max(2, min(w, (n - 1) // 2 - 1))
            if w < 2:
                continue
            d = _scaled_curv(x, w)
            s = _robust_mad(d)
            if s <= 0:
                s = float(np.std(d)) or 1.0
            sal = np.maximum(sal, d / (s + DERIV_EPS))
    e = np.log1p(sal)
    k = max(3, int(round(DERIV_SMOOTH_S / dt)) | 1)
    if n > k:
        e = np.convolve(e, np.ones(k) / k, mode="same")
    return e


def _strip_points(
    values: np.ndarray,
    img_w: int,
    strip_h: int,
    strip_idx: int,
    p_lower: float = 1.0,
    p_upper: float = 99.0,
) -> np.ndarray:
    """Same normalization/geometry contract as the v1 renderer's
    _get_signal_points (percentile clip -> strip pixel band)."""
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return None
    lo, hi = np.percentile(finite, [p_lower, p_upper])
    diff = hi - lo
    if diff <= 0:
        diff = 1.0
    norm = np.clip((values - lo) / diff, 0, 1)
    x = np.linspace(0, img_w - 1, len(values)).astype(np.int32)
    draw_h = strip_h - 2 * PADDING
    y_off = strip_idx * strip_h + PADDING
    y = (y_off + (1.0 - norm) * draw_h).astype(np.int32)
    return np.stack((x, y), axis=1)


def generate_detection_image_v2(df: pd.DataFrame, img_w: int, img_h: int) -> np.ndarray:
    """v2 detection render: R=dissipation, G=resonance frequency (both as in
    v1), B=derivative-energy ridge trace."""
    img = np.zeros((img_h, img_w, IMG_CHANNELS), dtype=np.uint8)
    if df is None or df.empty or len(df) < 2:
        return img
    strip_h = img_h // 3

    traces = [
        (
            (
                pd.to_numeric(df.get(COL_DISS), errors="coerce").to_numpy(dtype=float)
                if COL_DISS in df.columns
                else None
            ),
            0,
            2,
        ),  # red channel (BGR idx 2)
        (
            (
                pd.to_numeric(df.get(COL_FREQ), errors="coerce").to_numpy(dtype=float)
                if COL_FREQ in df.columns
                else None
            ),
            1,
            1,
        ),  # green
        (derivative_energy(df), 2, 0),  # blue: event ridges
    ]
    for values, strip_idx, ch_idx in traces:
        if values is None:
            continue
        p_hi = DERIV_UPPER_PCT if strip_idx == 2 else 99.0
        pts = _strip_points(values, img_w, strip_h, strip_idx, p_upper=p_hi)
        if pts is None:
            continue
        strip_bottom = (strip_idx + 1) * strip_h - PADDING
        poly = np.concatenate([pts, [[pts[-1][0], strip_bottom]], [[pts[0][0], strip_bottom]]])
        color = [0, 0, 0]
        color[ch_idx] = 255
        cv2.fillPoly(img, [poly], tuple(color))
        cv2.polylines(
            img,
            [pts.reshape((-1, 1, 2))],
            isClosed=False,
            color=COLOR_WHITE,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return img


def generate_detection_image(
    df: pd.DataFrame, img_w: int, img_h: int, version: int = 2
) -> np.ndarray:
    """Version dispatch used by both training (build_dataset.py) and
    inference (v6_yolo, via QModelV6Config.RENDER_VERSION)."""
    if version == 1:
        return DP.generate_detection_image(df, img_w=img_w, img_h=img_h)
    return generate_detection_image_v2(df, img_w=img_w, img_h=img_h)
