"""
QATCH.QModel.models.qmodel_onyx.onyx_render.py

Generates image-based representations viscosity data for model
and inference.

This module converts time-series sensor signals into multi-channel images used by
the Onyx detection model. It supports multiple rendering implementations through
a versioned dispatch interface, allowing training and inference code to use the
same rendering pipeline while maintaining backward compatibility.

Version 2 extends the original renderer by replacing the third channel with a
multi-scale derivative-energy representation that emphasizes rapid transitions
and gradual inflection points. This salience trace is computed from windowed
second differences across multiple temporal scales, normalized by a robust noise
estimate, and rendered as a dedicated image channel to improve point-of-interest
(POI) detection.

The generated images consist of vertically stacked signal strips:

* Red channel: Dissipation signal.
* Green channel: Resonance frequency signal.
* Blue channel (v2): Multi-scale derivative-energy salience.
* White polylines: Signal outlines drawn for improved edge definition.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-07-09
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

try:
    from QATCH.QModel.models.qmodel_onyx.onyx_dataprocessor import (
        QModelOnyxDataProcessor as DP,
    )
except (ImportError, ModuleNotFoundError):
    from QATCH.QModel.models.qmodel_onyx.onyx_dataprocessor import (
        QModelOnyxDataProcessor as DP,
    )

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
    """Computes the robust median absolute deviation (MAD) of an array.

    The MAD is scaled by 1.4826 so that, for normally distributed data, it is
    a consistent estimator of the standard deviation. Non-finite values are
    ignored before computation. If fewer than eight finite samples remain,
    `0.0` is returned to indicate that a reliable noise estimate cannot be
    computed.

    Args:
        x: Input array containing the values to analyze.

    Returns:
        The scaled median absolute deviation of the finite values, or `0.0`
        if there are fewer than eight finite samples.
    """
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _scaled_curv(x: np.ndarray, w: int) -> np.ndarray:
    """Computes the absolute windowed second difference of a signal.

    This function estimates local curvature by evaluating the magnitude of the
    second finite difference over a symmetric window. Unlike a first derivative,
    the second difference is insensitive to linear trends while responding
    strongly to steps, inflection points, and changes in slope. The output has
    the same length as the input, with edge values replicated to avoid boundary
    artifacts.

    Args:
        x: One-dimensional input signal.
        w: Half-window size, in samples, used to compute the second difference.

    Returns:
        An array of the same length as `x` containing the absolute windowed
        second difference. Returns an array of zeros if the signal is too short
        for the specified window size.
    """
    n = len(x)
    out = np.zeros(n)
    if n <= 2 * w:
        return out
    out[w:-w] = np.abs(x[2 * w :] - 2.0 * x[w:-w] + x[: -2 * w])
    out[:w] = out[w]
    out[-w:] = out[-w - 1]
    return out


def derivative_energy(df: pd.DataFrame) -> np.ndarray:
    """Computes a multi-scale transition salience trace from sensor signals.

    The salience trace is constructed by computing the windowed second
    difference of the dissipation and resonance frequency signals over multiple
    temporal scales. At each scale, the response is normalized by its own
    median absolute deviation (MAD) to compensate for scale-dependent noise.
    The maximum normalized response across all signals and scales is then
    log-compressed and lightly smoothed to produce a single one-dimensional
    feature emphasizing transitions.

    This representation highlights both abrupt events and slower changes in
    slope while remaining relatively insensitive to linear trends and the
    differing dynamic ranges of the input signals.

    Args:
        df: DataFrame containing the input time-series data. Expected columns
            include `Relative_time`, `Dissipation`, and
            `Resonance_Frequency`.

    Returns:
        A one-dimensional NumPy array containing the derivative-energy
        salience trace. Returns an array of zeros if the input is too short
        or does not contain a valid time column.
    """
    n = len(df)
    if n < 16:
        return np.zeros(n)
    if COL_TIME not in df.columns:
        return np.zeros(n)
    t = pd.to_numeric(df[COL_TIME], errors="coerce").to_numpy(dtype=float)
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    dt = float(np.median(diffs)) if len(diffs) else DP.TIME_STEP
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
    """Converts a one-dimensional signal into image coordinates for rendering.

    The input signal is robustly normalized using lower and upper percentile
    clipping, then mapped into the vertical extent of a single horizontal strip
    within the output image. Non-finite values are replaced with the median of
    the finite samples before normalization.

    Args:
        values: One-dimensional signal to convert into image coordinates.
        img_w: Width of the output image in pixels.
        strip_h: Height of the strip allocated to the signal in pixels.
        strip_idx: Zero-based index of the strip within the output image.
        p_lower: Lower percentile used for normalization.
        p_upper: Upper percentile used for normalization.

    Returns:
        An `(N, 2)` array of `(x, y)` pixel coordinates suitable for
        rendering with OpenCV, where `N` is the number of input samples.
        Returns `None` if fewer than two finite values are available.
    """
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    if len(finite) < 2:
        return None
    if not np.all(finite_mask):
        values = np.where(finite_mask, values, np.median(finite))
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


def generate_channel_det_v2(df: pd.DataFrame, img_w: int, img_h: int) -> np.ndarray:
    """Generates a version 2 detection image from Onyx sensor data.

    The output image is composed of three vertically stacked signal strips,
    each rendered into a separate color channel:

    * Red: Dissipation signal.
    * Green: Resonance frequency signal.
    * Blue: Multi-scale derivative-energy salience trace.

    Each signal is robustly normalized, converted to image coordinates, and
    rendered as a filled polygon with a white anti-aliased outline. The
    derivative-energy channel emphasizes transition regions that are useful for
    point-of-interest (POI) detection.

    Args:
        df: DataFrame containing the input sensor data.
        img_w: Width of the output image in pixels.
        img_h: Height of the output image in pixels.

    Returns:
        A three-channel `uint8` image of shape `(img_h, img_w, 3)` suitable
        for use as input to the Onyx detection model. If the input DataFrame is
        `None`, empty, or contains fewer than two samples, a blank image is
        returned.
    """
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


def generate_det_image(
    df: pd.DataFrame,
    img_w: int,
    img_h: int,
    version: int = 2,
) -> np.ndarray:
    """Generates a detection image using the requested rendering version.

    This function provides a common entry point for both model training and
    inference, dispatching to the appropriate rendering implementation based on
    the specified version number. Version 1 uses the legacy renderer provided
    by the data processor, while version 2 uses the enhanced derivative-energy
    renderer implemented in this module.

    Args:
        df: DataFrame containing the input sensor data.
        img_w: Width of the output image in pixels.
        img_h: Height of the output image in pixels.
        version: Rendering implementation to use. Supported values are `1`
            (legacy renderer) and `2` (derivative-energy renderer).

    Returns:
        A three-channel `uint8` image suitable for model training or
        inference.

    Raises:
        ValueError: If `version` is not a supported rendering version.
    """
    """Version dispatch used by both training (build_dataset.py) and
    inference (qmodel_onyx, via QModelOnyxConfig.RENDER_VERSION)."""
    if version == 1:
        return DP.generate_channel_det(df, img_w=img_w, img_h=img_h)
    if version == 2:
        return generate_channel_det_v2(df, img_w=img_w, img_h=img_h)
    raise ValueError(f"Unsupported render version: {version!r}")
