"""
QATCH.QModel.models.qmodel_onyx.onyx_fill_render


Fill-CLASSIFICATION renderer for the Onyx pipeline, sharing the one train/
deploy input contract with the fill classifier and the streaming path.

Why the classifier needs the derivative-energy strip more than the detectors
----------------------------------------------------------------------------
The fill classifier's job is literally COUNTING transitions: the class IS the
number of fill events visible so far. The legacy third strip is the Difference
curve - a linear combination of the two value strips that carries almost no
independent information - and all three strips share the v1 percentile-value
normalization whose failure mode the onyx detector work exposed: in a long
viscous run the early fill step owns the entire dynamic range, so the late
transitions that distinguish 2ch from 3ch flatten into featureless plateaus.
That is exactly the confusion boundary a channel counter cannot afford to lose.

The v2 classification render mirrors the onyx detector render (`onyx_render`):

  * Strips 0/1 (dissipation red, resonance green) are kept EXACTLY as the v1
    classifier render draws them - same percentile normalization, color fill,
    +50 edge highlight - preserving the global fill-context cues (step
    position, plateau levels, fill fraction of frame) the current model
    already exploits. Those cues are what separate no_fill / initial_fill,
    where value shape matters more than transition count.
  * Strip 2 replaces the Difference curve with the DERIVATIVE-ENERGY trace
    from `onyx_render`: multi-scale, per-scale-MAD-normalized, log-compressed
    curvature salience. Every transition appears as a ridge of comparable
    height regardless of where in the run it occurs.

Version 3 (step-coincidence energy) additionally replaces that curvature
strip with a matched STEP filter combined across signals by geometric mean;
see the block comment on `step_coincidence_energy`.

Train/deploy contract
---------------------
`prepare_cls_input` is the ONE function that turns a preprocessed dataframe
into the 224x224 tensor-ready image; the dataset builder saves its exact
output and the predictor feeds its exact output, so training and inference
images are bit-identical by construction. `FILL_RENDER_VERSION` (in `onyx`)
dispatches v1 (legacy weights) vs v2 vs v3 - weights and render version
travel together, the same roll-out mechanism as
`QModelOnyxConfig.RENDER_VERSION` on the detector side.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from QATCH.QModel.models.qmodel_onyx.onyx_dataprocessor import (
    QModelOnyxDataProcessor as DP,
)
from QATCH.QModel.models.qmodel_onyx.onyx_render import (
    DERIV_UPPER_PCT,
    _robust_mad,
    derivative_energy,
)

COL_TIME = DP.COL_TIME
COL_DISS = DP.COL_DISS
COL_FREQ = DP.COL_FREQ

PADDING = DP.PADDING
IMG_CHANNELS = 3

# Generation and inference geometry - identical to the classifier path
# (QModelOnyxConfig.FILL_GEN_* / FILL_INFERENCE_*), restated here so the
# render contract is self-contained.
FILL_GEN_W = 640
FILL_GEN_H = 640
FILL_INFERENCE_W = 224
FILL_INFERENCE_H = 224

# Visual language of the classification render (matches v1 generate_fill_cls:
# colored fills with a +50 edge highlight, NOT the detector's channel masks).
# (values_key, BGR fill color, upper percentile)
STRIP_SPEC = (
    ("diss", (0, 0, 255), 99.0),
    ("freq", (0, 255, 0), 99.0),
    ("energy", (255, 0, 0), DERIV_UPPER_PCT),
)

# --- Version 3: step-coincidence energy parameters ---
# Absolute scales plus SPAN-RELATIVE scales: the fill classifier sees whole
# runs from 25 s to 750 s; fixed absolute scales cannot serve both ends, and
# measured transition extents ran ~6-9% of span, which span/12..span/32
# brackets.
STEP_ABS_SCALES_S = (0.5, 2.0, 8.0)
STEP_REL_SCALES = (1.0 / 32.0, 1.0 / 12.0)
STEP_SMOOTH_S = 0.15


def _strip_points(
    values: np.ndarray,
    img_w: int,
    strip_h: int,
    strip_idx: int,
    p_lower: float = 1.0,
    p_upper: float = 99.0,
):
    """Percentile clip -> strip pixel band. Same normalization/geometry
    contract as the dataprocessor's _get_signal_points."""
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


def _series(df: pd.DataFrame, col: str) -> np.ndarray | None:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _draw_strips(df: pd.DataFrame, img_w: int, img_h: int, energy: np.ndarray) -> np.ndarray:
    """Shared draw path for v2/v3: value strips + an energy strip. The only
    difference between the two versions is which energy trace is passed in."""
    img = np.zeros((img_h, img_w, IMG_CHANNELS), dtype=np.uint8)
    if df is None or df.empty or len(df) < 2:
        return img
    strip_h = img_h // 3
    series = {
        "diss": _series(df, COL_DISS),
        "freq": _series(df, COL_FREQ),
        "energy": energy,
    }
    for strip_idx, (key, color, p_hi) in enumerate(STRIP_SPEC):
        values = series[key]
        if values is None:
            continue
        pts = _strip_points(values, img_w, strip_h, strip_idx, p_upper=p_hi)
        if pts is None:
            continue
        strip_bottom = (strip_idx + 1) * strip_h - PADDING
        poly = np.concatenate([pts, [[pts[-1][0], strip_bottom]], [[pts[0][0], strip_bottom]]])
        cv2.fillPoly(img, [poly], color)
        edge_color = tuple(min(c + 50, 255) for c in color)
        cv2.polylines(
            img,
            [pts.reshape((-1, 1, 2))],
            isClosed=False,
            color=edge_color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return img


def generate_fill_cls_v2(
    df: pd.DataFrame, img_w: int = FILL_GEN_W, img_h: int = FILL_GEN_H
) -> np.ndarray:
    """v2 classification render: dissipation (red) + resonance (green) value
    strips exactly as v1, derivative-energy ridge strip (blue) replacing the
    Difference curve. Takes TOTAL image dimensions."""
    return _draw_strips(df, img_w, img_h, derivative_energy(df))


def _step_response(x: np.ndarray, w: int) -> np.ndarray:
    """|mean(x[i+1..i+w]) - mean(x[i-w..i])|, normalized by the interior MAD
    of its own response - a matched filter for level shifts at scale ~w
    samples. O(n) via cumulative sums."""
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x, nan=float(np.nanmedian(x))))])
    i = np.arange(n)
    hi = np.minimum(i + w, n - 1)
    lo = np.maximum(i - w, 0)
    mean_r = (cs[hi + 1] - cs[i + 1]) / np.maximum(hi - i, 1)
    mean_l = (cs[i + 1] - cs[lo]) / np.maximum(i - lo + 1, 1)
    d = np.abs(mean_r - mean_l)
    interior = d[w : n - w] if n > 2 * w else d
    s = _robust_mad(interior)
    if s <= 0:
        s = float(np.std(d)) or 1.0
    return d / s


def step_coincidence_energy(df: pd.DataFrame) -> np.ndarray:
    """Multi-scale, cross-signal-coincident step salience.

    A STEP filter (difference of adjacent window MEANS) carries the full step
    amplitude as signal while noise shrinks ~sqrt(w/dt), so slow LATE
    transitions register where the v2 windowed second difference (three
    sampled points, noise never averaging down) washes them out. Physics says
    a genuine fill transition moves dissipation AND resonance frequency
    together, so per-scale responses are combined by GEOMETRIC MEAN - keeping
    coordinated events and suppressing single-channel drift excursions that
    fuel 2ch->3ch over-counts. Returns a per-sample trace; drawn as the energy
    strip by the v3 render.
    """
    n = len(df)
    if n < 16 or COL_TIME not in df.columns:
        return np.zeros(n)
    t = pd.to_numeric(df[COL_TIME], errors="coerce").to_numpy(dtype=float)
    dt = float(np.nanmedian(np.diff(t))) or DP.TIME_STEP
    span = float(t[-1] - t[0])
    scales = sorted(set(list(STEP_ABS_SCALES_S) + [max(0.5, span * r) for r in STEP_REL_SCALES]))
    have = [c for c in (COL_DISS, COL_FREQ) if c in df.columns]
    if not have:
        return np.zeros(n)
    sal = np.zeros(n)
    for scale_s in scales:
        w = int(round(scale_s / dt))
        w = max(3, min(w, (n - 1) // 2 - 1))
        if w < 3:
            continue
        resp = [_step_response(_series(df, c), w) for c in have]
        per_scale = np.sqrt(resp[0] * resp[1]) if len(resp) == 2 else resp[0]
        sal = np.maximum(sal, per_scale)
    e = np.log1p(sal)
    k = max(3, int(round(STEP_SMOOTH_S / dt)) | 1)
    if n > k:
        e = np.convolve(e, np.ones(k) / k, mode="same")
    return e


def generate_fill_cls_v3(
    df: pd.DataFrame, img_w: int = FILL_GEN_W, img_h: int = FILL_GEN_H
) -> np.ndarray:
    """v3 render: identical to v2 except the energy strip is the
    step-coincidence energy instead of the curvature-based derivative
    energy."""
    return _draw_strips(df, img_w, img_h, step_coincidence_energy(df))


def generate_fill_image(df: pd.DataFrame, version: int = 2) -> np.ndarray:
    """Version dispatch. version=1 reproduces the legacy classifier render
    (diss/freq/Difference at FILL_GEN geometry) so old type_cls weights keep
    working; version=2 is the derivative-energy render; version>=3 swaps in
    the step-coincidence energy (retrain required - weights and version
    travel together)."""
    if version == 1:
        # v1 generate_fill_cls takes PER-STRIP height.
        return DP.generate_fill_cls(df, img_h=FILL_GEN_H // 3, img_w=FILL_GEN_W)
    if version >= 3:
        return generate_fill_cls_v3(df, FILL_GEN_W, FILL_GEN_H)
    return generate_fill_cls_v2(df, FILL_GEN_W, FILL_GEN_H)


def prepare_cls_input(df: pd.DataFrame, version: int = 2) -> np.ndarray:
    """THE train/deploy contract: preprocessed dataframe -> the exact
    224x224 BGR uint8 image the classifier consumes. The dataset builder
    saves this image; the predictor feeds this image. INTER_AREA matches the
    existing inference path."""
    img = generate_fill_image(df, version=version)
    return cv2.resize(img, (FILL_INFERENCE_W, FILL_INFERENCE_H), interpolation=cv2.INTER_AREA)
