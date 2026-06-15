"""
QModel v6.2 vs v6.3 — cross-version A/B benchmark
=================================================

Loads BOTH model iterations and their assets, runs each on the SAME corpus,
and produces a side-by-side performance analysis of the two production
pipelines' FINAL output (i.e. whatever each version actually returns to a
caller — for v6.2 that's cascade + optional config-decode; for v6.3 that's
cascade + config-decode + zoom refinement + the v2 renderer).

This mirrors benchmark_decode.py's reporting, but where that script pairs
"greedy vs decoded" inside ONE version, this one pairs "v6.2 vs v6.3" across
versions. The two arms therefore differ in everything that changed between
the releases (renderer v1->v2, the zoom-refine stage, the decode config
constants), so the deltas attribute end-to-end release improvement rather
than any single component.

Pairing is per-(run, POI): each run is fed identically to both controllers,
the final POI placements are read back from each version's output contract
(`POIx.indices[0]` resolved against the run's own time axis), and the
absolute time error vs ground truth is compared.

Headline outputs (mirrors benchmark_decode.py):
  * per-POI global aggregate, side by side (v6.2 vs v6.3), with paired
    deltas, win/loss/tie counts and gross failures FIXED vs INTRODUCED
  * per-viscosity-tier breakdown so the high-cP tail (slow fills) is visible
  * miss accounting: how often each version returned no placement for a POI
    that is present in ground truth (a -1 index), since the versions can
    differ in coverage as well as accuracy

Outputs written to ``--output``:
  * ``ab_metrics.csv``           — per-POI global aggregate, both versions
  * ``ab_metrics_by_tier.csv``   — per-POI per-tier breakdown, both versions
  * ``per_run_results.csv``      — one row per (run, POI) with both errors
  * ``regressions.csv``          — cases v6.3 made grossly worse than v6.2
  * ``regression_plots/``        — diagnostic figure per regression run

Viscosity tiers and corpus discovery are taken verbatim from
benchmark_decode.py so the two reports are directly comparable.

Usage
-----
    python benchmark_v62_v63.py \
        --raw-root path/to/data/raw \
        --assets-v62 assets_v62.json \
        --assets-v63 assets_v63.json \
        [--v62-module-dir path/containing/v6_yolo.py] \
        [--v63-module-dir path/containing/controller.py] \
        [--n-runs 300] [--gross-threshold 2.0] [--output benchmark_v62_v63_out]

    python benchmark_v62_v63.py --selftest
        (synthetic corpus + two mocked controllers exercising the REAL
         aggregation / pairing / reporting plumbing without model weights)

Asset JSON format (per version)
-------------------------------
Each ``--assets-*`` file is the ``model_assets`` dict the corresponding
controller's ``__init__`` expects, e.g.::

    {
      "fill_classifier": "/abs/path/fill_cls.pt",
      "detectors": {
        "init": "/abs/path/init.pt",
        "ch1":  "/abs/path/ch1.pt",
        "ch2":  "/abs/path/ch2.pt",
        "ch3":  "/abs/path/ch3.pt",
        "poi5_fine": "/abs/path/poi5_fine.pt",
        "ch1_zoom": "/abs/path/ch1_zoom.pt",   # v6.3 zoom refine
        "ch2_zoom": "/abs/path/ch2_zoom.pt",
        "ch3_zoom": "/abs/path/ch3_zoom.pt"
      },
      "spacing_prior": "/abs/path/spacing_prior.json"
    }

The ``spacing_prior`` key may also be supplied via ``--prior-v62`` /
``--prior-v63`` (overrides whatever is in the asset JSON).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display server
import matplotlib.pyplot as plt

LOG = logging.getLogger("qmodel.benchmark_v62_v63")

# ===========================================================================
#  Corpus layout (identical to benchmark_decode.py / fit_prior.py)
# ===========================================================================

# Chain-space POI order (the 5 detected POIs; the legacy row-2 shim is skipped).
POI_KEYS = ["POI1", "POI2", "POI3", "POI4", "POI5"]

# poi-csv row -> chain-space POI name (row 2 is the legacy shim, skipped).
POI_ROW = {"POI1": 0, "POI2": 1, "POI3": 3, "POI4": 4, "POI5": 5}

# chain name -> production output name (the POI_MAP id space both controllers
# emit). Both versions share POI_MAP {1:POI1,2:POI2,3:POI3,4:POI4,5:POI5,6:POI6}
# and DECODE_ID_TO_NAME {1:POI1,2:POI2,4:POI3,5:POI4,6:POI5}, so the chain
# names map onto the SAME production output keys for both releases.
CHAIN_TO_PROD = {"POI1": "POI1", "POI2": "POI2", "POI3": "POI4", "POI4": "POI5", "POI5": "POI6"}

TIER_EDGES = [2.66, 6.16, 18.14, 73.4]
TIER_LABELS = ["<2.66 cP", "2.66-6.16 cP", "6.16-18.14 cP", "18.14-73.4 cP", "73.4+ cP", "unknown"]


def viscosity_tier(cp: Optional[float]) -> int:
    if cp is None or not np.isfinite(cp):
        return len(TIER_EDGES) + 1  # "unknown"
    for i, edge in enumerate(TIER_EDGES):
        if cp < edge:
            return i
    return len(TIER_EDGES)


@dataclass
class RunRecord:
    run_id: str
    csv_path: Path
    poi_times: Dict[str, float]  # chain-space truth (prefix on partial fills)
    viscosity_cP: Optional[float]


def _truth_times(poi_path: Path, time_axis: np.ndarray) -> Dict[str, float]:
    """Chain-space ground-truth times; strictly-ascending, non-tail prefix."""
    try:
        raw_idx = pd.to_numeric(
            pd.read_csv(poi_path, header=None).iloc[:, 0], errors="coerce"
        ).to_numpy()
    except Exception:
        return {}
    n_rows = len(time_axis)
    last_idx = n_rows - 1
    tail_tol = max(2, int(0.001 * n_rows))
    out: Dict[str, float] = {}
    prev_t = -np.inf
    for name in POI_KEYS:
        row = POI_ROW[name]
        if row >= len(raw_idx) or np.isnan(raw_idx[row]):
            break
        idx = int(raw_idx[row])
        if idx >= last_idx - tail_tol or idx < 0 or idx > last_idx:
            break
        t = float(time_axis[idx])
        if t <= prev_t:
            break
        out[name] = t
        prev_t = t
    return out


def _viscosity_from_frame(df: pd.DataFrame) -> Optional[float]:
    df.columns = [str(c).lstrip("# ").strip() for c in df.columns]
    if "viscosity_avg" in df.columns:
        v = pd.to_numeric(df["viscosity_avg"], errors="coerce").dropna()
        if len(v):
            return float(v.mean())
    return None


def _run_viscosity(run_dir: Path) -> Optional[float]:
    """Mean viscosity_avg from a run's analyze output (loose CSV, then zips)."""
    for p in run_dir.glob("*analyze_out*.csv"):
        try:
            cp = _viscosity_from_frame(pd.read_csv(p))
            if cp is not None:
                return cp
        except Exception:
            continue
    import io
    import zipfile

    for z in sorted(run_dir.glob("analyze-*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                for name in zf.namelist():
                    if "analyze_out" in name.lower() and name.lower().endswith(".csv"):
                        with zf.open(name) as fh:
                            cp = _viscosity_from_frame(pd.read_csv(io.BytesIO(fh.read())))
                        if cp is not None:
                            return cp
        except Exception:
            continue
    return None


def discover_runs(raw_root: Path, time_col: str = "Relative_time") -> List[RunRecord]:
    runs: List[RunRecord] = []
    for d in sorted(Path(raw_root).iterdir()):
        if not d.is_dir():
            continue
        cands = [
            p
            for p in d.glob("*.csv")
            if not p.name.lower().endswith("_poi.csv") and "analyze_out" not in p.name.lower()
        ]
        poi_files = list(d.glob("*_poi.csv"))
        if not cands or not poi_files:
            continue
        try:
            data = pd.read_csv(cands[0])
        except Exception:
            continue
        tcol = time_col if time_col in data.columns else data.columns[0]
        ta = pd.to_numeric(data[tcol], errors="coerce").to_numpy()
        if len(ta) < 2 or np.isnan(ta).all():
            continue
        truth = _truth_times(poi_files[0], ta)
        if not truth:
            continue
        runs.append(
            RunRecord(
                run_id=d.name,
                csv_path=cands[0],
                poi_times=truth,
                viscosity_cP=_run_viscosity(d),
            )
        )
    return runs


# ===========================================================================
#  Metric accumulators
# ===========================================================================


@dataclass
class _POIMetrics:
    """Per-POI time-error accumulator (same fields as benchmark_decode.py)."""

    time_errs: List[float] = field(default_factory=list)
    n: int = 0
    n_miss: int = 0  # POI present in truth but version returned no placement
    mae: float = float("nan")
    rmse: float = float("nan")
    median_ae: float = float("nan")
    bias: float = float("nan")
    max_ae: float = float("nan")
    gross_failure_rate: float = float("nan")

    def record(self, err: float) -> None:
        self.time_errs.append(err)

    def summarize(self, gross_threshold: float) -> None:
        if not self.time_errs:
            return
        e = np.array(self.time_errs)
        ae = np.abs(e)
        self.n = len(e)
        self.mae = float(np.mean(ae))
        self.rmse = float(np.sqrt(np.mean(e**2)))
        self.median_ae = float(np.median(ae))
        self.bias = float(np.mean(e))
        self.max_ae = float(np.max(ae))
        self.gross_failure_rate = float(np.mean(ae > gross_threshold))


@dataclass
class _PairedCounts:
    """Paired A/B counters for one POI (or one POI x tier cell). 'win' means
    v6.3 strictly better than v6.2 (beyond the tie band)."""

    wins: int = 0  # v6.3 strictly better
    losses: int = 0  # v6.3 strictly worse
    ties: int = 0
    gross_fixed: int = 0  # v6.2 gross, v6.3 fine
    gross_introduced: int = 0  # v6.2 fine, v6.3 gross
    miss_fixed: int = 0  # v6.2 missed, v6.3 placed
    miss_introduced: int = 0  # v6.2 placed, v6.3 missed


# ===========================================================================
#  Pretty printing
# ===========================================================================


def _print_global(
    a: Dict[str, _POIMetrics],
    b: Dict[str, _POIMetrics],
    paired: Dict[str, _PairedCounts],
    gross_threshold: float,
    n_runs: int,
    label_a: str,
    label_b: str,
    output_dir: Path,
) -> None:
    HDR = (
        f"{'POI':<6} {'N':>5}  {'MAE_2':>8} {'MAE_3':>8} {'dMAE':>8}  "
        f"{'Med_2':>7} {'Med_3':>7}  {'Fail%_2':>8} {'Fail%_3':>8}  "
        f"{'Fixed':>5} {'Intro':>5}  {'W/L/T':>12}  {'Miss23':>8}"
    )
    FMT = (
        "{:<6} {:>5d}  {:>8.3f} {:>8.3f} {:>+8.3f}  {:>7.3f} {:>7.3f}  "
        "{:>7.1%} {:>8.1%}  {:>5d} {:>5d}  {:>12}  {:>8}"
    )
    SEP = "-" * len(HDR)
    BAR = "=" * (len(HDR) + 4)

    print(f"\n{BAR}")
    print(f"  QModel cross-version A/B Benchmark   ({label_a} vs {label_b})")
    print(f"  {n_runs} runs  |  gross > {gross_threshold} s  |  paired, same corpus")
    print(f"  Output -> {output_dir}")
    print(f"  {SEP}")
    print(f"  {HDR}")
    print(f"  {SEP}")
    for poi in POI_KEYS:
        ma, mb, pc = a.get(poi), b.get(poi), paired.get(poi)
        if ma is None or ma.n == 0:
            print(f"  {poi:<6} {'-':>5}")
            continue
        wlt = f"{pc.wins}/{pc.losses}/{pc.ties}"
        miss = f"{ma.n_miss}/{mb.n_miss}"
        print(
            "  "
            + FMT.format(
                poi,
                ma.n,
                ma.mae,
                mb.mae,
                mb.mae - ma.mae,
                ma.median_ae,
                mb.median_ae,
                ma.gross_failure_rate,
                mb.gross_failure_rate,
                pc.gross_fixed,
                pc.gross_introduced,
                wlt,
                miss,
            )
        )
    print(f"  {SEP}")
    print(
        f"  Columns: _2 = {label_a}, _3 = {label_b}.  dMAE<0 favours {label_b}.  "
        f"Fixed/Intro = gross failures {label_b} fixed / introduced."
    )
    print(f"  Miss23 = (# {label_a} misses)/(# {label_b} misses) among POIs present in truth.")
    print(f"{BAR}\n")


def _print_tier(
    a: Dict[str, Dict[int, _POIMetrics]],
    b: Dict[str, Dict[int, _POIMetrics]],
    gross_threshold: float,
    label_a: str,
    label_b: str,
) -> None:
    HDR = (
        f"{'POI':<6} {'Tier':<14} {'N':>5}  {'MAE_2':>8} {'MAE_3':>8} {'dMAE':>8}  "
        f"{'Fail%_2':>8} {'Fail%_3':>8}"
    )
    FMT = "{:<6} {:<14} {:>5d}  {:>8.3f} {:>8.3f} {:>+8.3f}  {:>7.1%} {:>8.1%}"
    SEP = "-" * len(HDR)
    BAR = "=" * (len(HDR) + 4)

    print(f"\n{BAR}")
    print(f"  Per-Tier A/B Breakdown   ({label_a} vs {label_b})   gross > {gross_threshold} s")
    print(f"  {SEP}")
    print(f"  {HDR}")
    print(f"  {SEP}")
    for poi in POI_KEYS:
        any_printed = False
        for tier_idx, tier_label in enumerate(TIER_LABELS):
            ma = a.get(poi, {}).get(tier_idx)
            mb = b.get(poi, {}).get(tier_idx)
            if ma is None or ma.n == 0:
                continue
            print(
                "  "
                + FMT.format(
                    poi,
                    tier_label,
                    ma.n,
                    ma.mae,
                    mb.mae,
                    mb.mae - ma.mae,
                    ma.gross_failure_rate,
                    mb.gross_failure_rate,
                )
            )
            any_printed = True
        if not any_printed:
            print(f"  {poi:<6} {'-':>14}")
        print(f"  {SEP}")
    print(f"{BAR}\n")


# ===========================================================================
#  Regression plotting (v6.3 made it grossly worse than v6.2)
# ===========================================================================

_PLOT_SIGNALS = [("Dissipation", "Dissipation"), ("Resonance_Frequency", "Resonance_Frequency")]
_POI_COLORS = {
    "POI1": "#e41a1c",
    "POI2": "#377eb8",
    "POI3": "#4daf4a",
    "POI4": "#984ea3",
    "POI5": "#ff7f00",
}


def _render_regression_plot(
    df_raw: pd.DataFrame,
    run_id: str,
    tier_label: str,
    regressions: Dict[str, Dict[str, float]],
    out_path: Path,
    label_a: str,
    label_b: str,
) -> bool:
    """One figure per run v6.3 regressed: truth solid, v6.2 dashed,
    v6.3 dotted, per failing POI."""
    tcol = "Relative_time" if "Relative_time" in df_raw.columns else df_raw.columns[0]
    t = pd.to_numeric(df_raw[tcol], errors="coerce").to_numpy()
    fig, axes = plt.subplots(len(_PLOT_SIGNALS), 1, figsize=(13, 6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (col, nice) in zip(axes, _PLOT_SIGNALS):
        if col in df_raw.columns:
            ax.plot(t, pd.to_numeric(df_raw[col], errors="coerce").to_numpy(), color="0.25", lw=0.9)
        ax.set_ylabel(nice, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        for poi, info in sorted(regressions.items()):
            c = _POI_COLORS.get(poi, "#000000")
            ax.axvline(info["true_t"], color=c, lw=1.6, alpha=0.9)
            ax.axvline(info["a_t"], color=c, lw=1.4, ls="--", alpha=0.9)
            ax.axvline(info["b_t"], color=c, lw=1.4, ls=":", alpha=0.9)
    handles = [
        plt.Line2D([0], [0], color="0.25", lw=1.6, label="truth (solid)"),
        plt.Line2D([0], [0], color="0.25", lw=1.6, ls="--", label=f"{label_a} (dashed)"),
        plt.Line2D([0], [0], color="0.25", lw=1.6, ls=":", label=f"{label_b} (dotted)"),
    ]
    for poi in sorted(regressions):
        handles.append(plt.Line2D([0], [0], color=_POI_COLORS.get(poi, "#000"), lw=2.4, label=poi))
    axes[0].legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.85)
    axes[-1].set_xlabel("Relative_time (s)", fontsize=9)
    fig.suptitle(
        f"{label_b} regression vs {label_a} — run {run_id}   (tier {tier_label})", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


# ===========================================================================
#  Per-run extraction from a controller's output contract
# ===========================================================================


def _times_from_output(output: Dict[str, Any], df_raw: pd.DataFrame) -> Dict[str, float]:
    """Chain-space final placements from one predict() call's output dict.

    Reads the standard ``{POIx: {"indices":[i], "confidences":[c]}}`` contract
    that BOTH versions emit, resolves the index against the run's own time
    axis, and keys the result in chain space. POIs the version did not place
    (index < 0) are omitted, so callers can distinguish a miss from a hit."""
    tcol = "Relative_time" if "Relative_time" in df_raw.columns else df_raw.columns[0]
    tv = pd.to_numeric(df_raw[tcol], errors="coerce").to_numpy()
    n = len(tv)

    times: Dict[str, float] = {}
    for chain, prod in CHAIN_TO_PROD.items():
        rec = output.get(prod, {})
        idxs = rec.get("indices", [-1])
        if not idxs or idxs[0] is None:
            continue
        i = int(idxs[0])
        if 0 <= i < n:
            times[chain] = float(tv[i])
    return times


# ===========================================================================
#  Benchmark driver
# ===========================================================================


def run_benchmark(
    controller_a: Any,
    controller_b: Any,
    runs: List[RunRecord],
    output_dir: Path,
    label_a: str = "v6.2",
    label_b: str = "v6.3",
    predict_kwargs_a: Optional[Dict[str, Any]] = None,
    predict_kwargs_b: Optional[Dict[str, Any]] = None,
    n_runs: Optional[int] = None,
    gross_threshold: float = 2.0,
    tie_band: float = 0.05,
    seed: int = 1337,
) -> Dict[str, Any]:
    """Run the paired cross-version benchmark and print/persist metrics.

    Both controllers must expose ``predict(df=...) -> (output_dict, n_ch)``
    with the shared QModel output contract. Each is called once per run with
    its own ``predict_kwargs_*`` (so v6.3 can be run with refine/decode on and
    v6.2 with decode on, reflecting each release's intended production mode).
    Returns the global summary dict (useful for asserting in tests / CI).
    """
    predict_kwargs_a = dict(predict_kwargs_a or {})
    predict_kwargs_b = dict(predict_kwargs_b or {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "regression_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if n_runs is not None and len(runs) > n_runs:
        rng = np.random.default_rng(seed)
        runs = list(runs)
        rng.shuffle(runs)
        runs = runs[:n_runs]
        LOG.info("Benchmark: capped to %d runs", len(runs))

    a_metrics = {p: _POIMetrics() for p in POI_KEYS}
    b_metrics = {p: _POIMetrics() for p in POI_KEYS}
    a_tier = {p: {t: _POIMetrics() for t in range(len(TIER_LABELS))} for p in POI_KEYS}
    b_tier = {p: {t: _POIMetrics() for t in range(len(TIER_LABELS))} for p in POI_KEYS}
    paired = {p: _PairedCounts() for p in POI_KEYS}

    per_run_rows: List[Dict[str, Any]] = []
    regression_rows: List[Dict[str, Any]] = []
    n_processed = n_regression_plots = 0
    n_fail_a = n_fail_b = 0

    for run in runs:
        try:
            df_raw = pd.read_csv(run.csv_path)
        except Exception as exc:
            LOG.warning("Benchmark: read failed for %s (%s)", run.csv_path, exc)
            continue

        try:
            out_a, _ = controller_a.predict(df=df_raw, **predict_kwargs_a)
        except Exception as exc:
            LOG.warning("Benchmark: %s predict failed for run %s (%s)", label_a, run.run_id, exc)
            n_fail_a += 1
            continue
        try:
            out_b, _ = controller_b.predict(df=df_raw, **predict_kwargs_b)
        except Exception as exc:
            LOG.warning("Benchmark: %s predict failed for run %s (%s)", label_b, run.run_id, exc)
            n_fail_b += 1
            continue

        a_t = _times_from_output(out_a, df_raw)
        b_t = _times_from_output(out_b, df_raw)
        tier = viscosity_tier(run.viscosity_cP)
        run_regressions: Dict[str, Dict[str, float]] = {}

        for poi in POI_KEYS:
            true_t = run.poi_times.get(poi)
            if true_t is None:
                continue
            ta, tb = a_t.get(poi), b_t.get(poi)
            a_hit, b_hit = ta is not None, tb is not None

            # ---- miss accounting (POI present in truth)
            if not a_hit:
                a_metrics[poi].n_miss += 1
            if not b_hit:
                b_metrics[poi].n_miss += 1
            if not a_hit and b_hit:
                paired[poi].miss_fixed += 1
            if a_hit and not b_hit:
                paired[poi].miss_introduced += 1

            # error metrics only when BOTH placed the POI (paired comparison)
            if not (a_hit and b_hit):
                per_run_rows.append(
                    dict(
                        run_id=run.run_id,
                        poi=poi,
                        viscosity_cP=run.viscosity_cP,
                        tier=TIER_LABELS[tier],
                        true_t=true_t,
                        a_t=ta,
                        b_t=tb,
                        a_err_s=(ta - true_t) if a_hit else None,
                        b_err_s=(tb - true_t) if b_hit else None,
                        a_hit=a_hit,
                        b_hit=b_hit,
                    )
                )
                continue

            ea, eb = ta - true_t, tb - true_t
            a_metrics[poi].record(ea)
            b_metrics[poi].record(eb)
            a_tier[poi][tier].record(ea)
            b_tier[poi][tier].record(eb)

            delta = abs(ea) - abs(eb)  # >0 => v6.3 better
            if delta > tie_band:
                paired[poi].wins += 1
            elif delta < -tie_band:
                paired[poi].losses += 1
            else:
                paired[poi].ties += 1

            a_gross, b_gross = abs(ea) > gross_threshold, abs(eb) > gross_threshold
            if a_gross and not b_gross:
                paired[poi].gross_fixed += 1
            if b_gross and not a_gross:
                paired[poi].gross_introduced += 1
                regression_rows.append(
                    dict(
                        run_id=run.run_id,
                        poi=poi,
                        viscosity_cP=run.viscosity_cP,
                        tier=TIER_LABELS[tier],
                        true_t=true_t,
                        a_t=ta,
                        b_t=tb,
                        a_err_s=ea,
                        b_err_s=eb,
                    )
                )
                run_regressions[poi] = {"true_t": true_t, "a_t": ta, "b_t": tb}

            per_run_rows.append(
                dict(
                    run_id=run.run_id,
                    poi=poi,
                    viscosity_cP=run.viscosity_cP,
                    tier=TIER_LABELS[tier],
                    true_t=true_t,
                    a_t=ta,
                    b_t=tb,
                    a_err_s=ea,
                    b_err_s=eb,
                    a_hit=True,
                    b_hit=True,
                )
            )

        if run_regressions:
            try:
                if _render_regression_plot(
                    df_raw,
                    run.run_id,
                    TIER_LABELS[tier],
                    run_regressions,
                    plot_dir / f"{run.run_id}.png",
                    label_a,
                    label_b,
                ):
                    n_regression_plots += 1
            except Exception as exc:
                LOG.warning("Benchmark: regression plot failed for %s (%s)", run.run_id, exc)

        n_processed += 1
        if n_processed % 100 == 0:
            LOG.info("Benchmark: processed %d / %d", n_processed, len(runs))

    for poi in POI_KEYS:
        a_metrics[poi].summarize(gross_threshold)
        b_metrics[poi].summarize(gross_threshold)
        for t in range(len(TIER_LABELS)):
            a_tier[poi][t].summarize(gross_threshold)
            b_tier[poi][t].summarize(gross_threshold)

    _print_global(
        a_metrics, b_metrics, paired, gross_threshold, n_processed, label_a, label_b, output_dir
    )
    _print_tier(a_tier, b_tier, gross_threshold, label_a, label_b)

    if n_fail_a or n_fail_b:
        print(
            f"  NOTE: predict() failed on {n_fail_a} run(s) for {label_a}, "
            f"{n_fail_b} for {label_b} (runs skipped from both arms).\n"
        )

    # ------------------------------------------------------------- CSVs
    pd.DataFrame(
        [
            dict(
                poi=poi,
                n=a_metrics[poi].n,
                n_miss_a=a_metrics[poi].n_miss,
                n_miss_b=b_metrics[poi].n_miss,
                mae_a_s=a_metrics[poi].mae,
                mae_b_s=b_metrics[poi].mae,
                delta_mae_s=b_metrics[poi].mae - a_metrics[poi].mae,
                median_ae_a_s=a_metrics[poi].median_ae,
                median_ae_b_s=b_metrics[poi].median_ae,
                rmse_a_s=a_metrics[poi].rmse,
                rmse_b_s=b_metrics[poi].rmse,
                bias_a_s=a_metrics[poi].bias,
                bias_b_s=b_metrics[poi].bias,
                max_ae_a_s=a_metrics[poi].max_ae,
                max_ae_b_s=b_metrics[poi].max_ae,
                gross_rate_a=a_metrics[poi].gross_failure_rate,
                gross_rate_b=b_metrics[poi].gross_failure_rate,
                gross_fixed=paired[poi].gross_fixed,
                gross_introduced=paired[poi].gross_introduced,
                miss_fixed=paired[poi].miss_fixed,
                miss_introduced=paired[poi].miss_introduced,
                wins=paired[poi].wins,
                losses=paired[poi].losses,
                ties=paired[poi].ties,
            )
            for poi in POI_KEYS
        ]
    ).to_csv(output_dir / "ab_metrics.csv", index=False)

    tier_rows = []
    for poi in POI_KEYS:
        for t in range(len(TIER_LABELS)):
            ma, mb = a_tier[poi][t], b_tier[poi][t]
            if ma.n == 0:
                continue
            tier_rows.append(
                dict(
                    poi=poi,
                    tier_index=t,
                    tier_label=TIER_LABELS[t],
                    n=ma.n,
                    mae_a_s=ma.mae,
                    mae_b_s=mb.mae,
                    delta_mae_s=mb.mae - ma.mae,
                    gross_rate_a=ma.gross_failure_rate,
                    gross_rate_b=mb.gross_failure_rate,
                )
            )
    if tier_rows:
        pd.DataFrame(tier_rows).to_csv(output_dir / "ab_metrics_by_tier.csv", index=False)
    if per_run_rows:
        pd.DataFrame(per_run_rows).to_csv(output_dir / "per_run_results.csv", index=False)
    if regression_rows:
        pd.DataFrame(regression_rows).to_csv(output_dir / "regressions.csv", index=False)
        LOG.info(
            "Benchmark: %d %s regression(s) -> %s (%d plot(s))",
            len(regression_rows),
            label_b,
            output_dir / "regressions.csv",
            n_regression_plots,
        )

    LOG.info("Benchmark complete. Metrics -> %s", output_dir)
    return dict(
        n_runs=n_processed,
        n_fail_a=n_fail_a,
        n_fail_b=n_fail_b,
        global_a={p: a_metrics[p] for p in POI_KEYS},
        global_b={p: b_metrics[p] for p in POI_KEYS},
        paired=paired,
    )


# ===========================================================================
#  Dynamic loading of the two controller modules
# ===========================================================================


def _install_qatch_stub() -> None:
    """Both controllers try `from QATCH...` imports first and fall back to a
    headless path on ImportError. v6.3's controller, however, does
    `from QATCH.ui.drawPlateConfig import Architecture` at module top level
    OUTSIDE any try/except, and its __init__ calls Architecture.get_path().
    So we can't simply let QATCH be absent — that hard-crashes the import.

    The fix: register a minimal fake `QATCH` package tree in sys.modules whose
    submodules raise ModuleNotFoundError on attribute access for the data /
    decode / renderer modules (so the controllers fall through to their
    headless sibling-import fallbacks), while providing a working
    `QATCH.ui.drawPlateConfig.Architecture` with a no-op get_path(). The
    detector paths v6.3's __init__ builds from get_path() are never read
    (model_assets["detectors"] overrides them), so the value is irrelevant —
    it only needs to not crash.

    Idempotent: safe to call more than once.
    """
    import types

    if "QATCH" in sys.modules and getattr(sys.modules["QATCH"], "_benchmark_stub", False):
        return

    def _pkg(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__path__ = []  # mark as a package so submodule imports are attempted
        m._benchmark_stub = True
        sys.modules[name] = m
        return m

    qatch = _pkg("QATCH")
    ui = _pkg("QATCH.ui")
    qatch.ui = ui

    draw = types.ModuleType("QATCH.ui.drawPlateConfig")
    draw._benchmark_stub = True

    class _Architecture:
        @staticmethod
        def get_path() -> str:
            # Never actually used for asset resolution (asset JSON overrides),
            # but must return a string so os.path.join doesn't blow up.
            return os.getcwd()

    draw.Architecture = _Architecture
    ui.drawPlateConfig = draw
    sys.modules["QATCH.ui.drawPlateConfig"] = draw

    # IMPORTANT: do NOT create QATCH.common / QATCH.qmodel submodules. Leaving
    # them absent makes `from QATCH.common.logger import Logger` and
    # `from QATCH.qmodel.models... import ...` raise ModuleNotFoundError, which
    # is exactly what the controllers' try/except headless fallbacks expect.


def _load_controller_module(module_path: Path, package_name: str, module_stem: str) -> Any:
    """Load a controller that lives inside a real package directory, importing
    it AS a submodule of that package so both its relative imports
    (`from .dataprocessor import ...`) and its bare-sibling fallbacks
    (`from dataprocessor import ...`) resolve.

    `module_path`  : full path to the controller .py (e.g. v6_2_yolo/v6_yolo.py)
    `package_name` : the package the file lives in (its parent dir name,
                     e.g. "v6_2_yolo")
    `module_stem`  : the controller file's stem (e.g. "v6_yolo" / "controller")

    The package's PARENT is placed on sys.path so `import <package_name>`
    works, and the package directory itself is also added so the controllers'
    last-resort bare imports succeed.

    Some controllers (notably v6.2's v6_yolo.py) have NO bare-name fallback for
    their sibling modules — their only headless path is
    `from QATCH.qmodel.models.<pkg>.<sibling> import ...`. To make that resolve
    without a real QATCH install, the real package directory is also aliased
    under the QATCH.qmodel.models.<pkg> dotted name (its __path__ points at the
    real dir), so those imports load the actual sibling files.
    """
    module_path = Path(module_path).resolve()
    pkg_dir = module_path.parent
    parent_dir = pkg_dir.parent

    for p in (str(parent_dir), str(pkg_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Ensure the package object exists (import it; create a namespace package
    # if it has no __init__.py or its __init__ imports something unavailable).
    if package_name not in sys.modules:
        try:
            importlib.import_module(package_name)
        except Exception as exc:
            LOG.warning(
                "Package %s did not import cleanly (%s); using a namespace shim.",
                package_name,
                exc,
            )
            import types

            shim = types.ModuleType(package_name)
            shim.__path__ = [str(pkg_dir)]
            sys.modules[package_name] = shim

    # Alias the real package dir under QATCH.qmodel.models.<package_name> so the
    # controllers' `from QATCH.qmodel.models...` headless fallbacks resolve to
    # the actual sibling files on disk.
    _alias_under_qatch(pkg_dir, package_name)

    fq_name = f"{package_name}.{module_stem}"
    spec = importlib.util.spec_from_file_location(fq_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    # Register under the fully-qualified name BEFORE exec so relative imports
    # inside the module resolve against the (real or shim) package.
    sys.modules[fq_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _alias_under_qatch(pkg_dir: Path, package_name: str) -> None:
    """Register namespace packages QATCH.qmodel, QATCH.qmodel.models, and
    QATCH.qmodel.models.<package_name> whose leaf __path__ points at the real
    package directory, so `from QATCH.qmodel.models.<package_name>.<sibling>
    import ...` loads the on-disk sibling modules."""
    import types

    def _ensure_ns(name: str, path: Optional[str] = None) -> types.ModuleType:
        if name in sys.modules:
            m = sys.modules[name]
        else:
            m = types.ModuleType(name)
            m.__path__ = []
            m._benchmark_stub = True
            sys.modules[name] = m
        if path is not None:
            m.__path__ = [path]
        return m

    qmodel = _ensure_ns("QATCH.qmodel")
    models = _ensure_ns("QATCH.qmodel.models")
    leaf = _ensure_ns(f"QATCH.qmodel.models.{package_name}", str(pkg_dir))
    # wire attribute access (QATCH.qmodel.models.<pkg>)
    qatch = sys.modules.get("QATCH")
    if qatch is not None:
        qatch.qmodel = qmodel
    qmodel.models = models
    setattr(models, package_name, leaf)


def _instantiate_controller(module: Any, class_name: str, assets: Dict[str, Any]) -> Any:
    cls = getattr(module, class_name, None)
    if cls is None:
        # fall back: find the controller class heuristically
        for cand in ("QModelV6YOLO", "QModel"):
            cls = getattr(module, cand, None)
            if cls is not None:
                break
    if cls is None:
        raise AttributeError(
            f"no controller class '{class_name}' (or known alias) in {module.__name__}"
        )
    return cls(assets)


# ===========================================================================
#  Self-test: synthetic corpus + two mocked controllers, REAL aggregation
# ===========================================================================


def _selftest(tmp_root: Path, output_dir: Path, n_runs: int = 80, seed: int = 11) -> None:
    """Builds a synthetic corpus, then two fake controllers that emit the
    REAL output contract: version A (stand-in for v6.2) is noisier and misses
    a channel POI occasionally; version B (stand-in for v6.3) is tighter and
    fixes some of A's gross errors but occasionally regresses one. Pushes both
    through the FULL benchmark aggregation to verify pairing / tiers / miss
    accounting / CSVs end-to-end, without model weights."""
    rng = np.random.default_rng(seed)
    tmp_root.mkdir(parents=True, exist_ok=True)

    # ---- corpus
    for i in range(n_runs):
        scale = rng.choice([0.5, 1.0, 2.5, 6.0])  # fast .. very slow fills
        t0 = rng.uniform(3, 12)
        gaps = scale * np.exp(rng.normal(np.log([4, 6, 8, 10]), 0.4))
        truth = np.concatenate([[t0], t0 + np.cumsum(gaps)])  # 5 POIs
        end = truth[-1] * 1.3
        t_axis = np.arange(0, end, 0.02)
        idxs = [int(np.searchsorted(t_axis, tt)) for tt in truth]
        run_dir = tmp_root / f"synth_{i:04d}"
        run_dir.mkdir(exist_ok=True)
        diss = np.cumsum(rng.normal(0, 1e-7, len(t_axis))) + 3e-5
        pd.DataFrame(
            {
                "Relative_time": t_axis,
                "Dissipation": diss,
                "Resonance_Frequency": 1.5e7 - np.linspace(0, 500, len(t_axis)),
            }
        ).to_csv(run_dir / f"synth_{i:04d}.csv", index=False)
        # poi rows 0..5 with the legacy row-2 shim duplicated next to row 1
        rows = [idxs[0], idxs[1], idxs[1] + 2, idxs[2], idxs[3], idxs[4]]
        pd.Series(rows).to_csv(run_dir / f"synth_{i:04d}_poi.csv", index=False, header=False)
        cp = {0.5: 3.0, 1.0: 15.0, 2.5: 80.0, 6.0: 220.0}[float(scale)] * rng.uniform(0.8, 1.2)
        pd.DataFrame({"shear_rate": [100.0], "viscosity_avg": [cp]}).to_csv(
            run_dir / f"synth_{i:04d}_analyze_out.csv", index=False
        )

    runs = discover_runs(tmp_root)
    assert runs, "selftest corpus discovery failed"
    truth_by_path = {str(r.csv_path): r.poi_times for r in runs}

    POI_MAP = {1: "POI1", 2: "POI2", 3: "POI3", 4: "POI4", 5: "POI5", 6: "POI6"}
    CHAIN_TO_ID = {"POI1": 1, "POI2": 2, "POI3": 4, "POI4": 5, "POI5": 6}

    class _MockController:
        """Emits the real {POIx:{indices,confidences}} contract from a noise
        model over ground truth. `noise_s` controls accuracy; `miss_p` the
        per-channel-POI miss probability; `gross_p` injects occasional gross
        errors; `seed_off` decorrelates the two versions."""

        def __init__(self, noise_s, miss_p, gross_p, seed_off):
            self.noise_s = noise_s
            self.miss_p = miss_p
            self.gross_p = gross_p
            self.rng = np.random.default_rng(seed + seed_off)

        def predict(self, df=None, **kw):
            tcol = "Relative_time"
            tv = df[tcol].to_numpy(dtype=float)
            n = len(tv)
            # recover this run's truth via the time axis match
            truth = None
            for r in runs:
                if (
                    abs(pd.read_csv(r.csv_path)[tcol].to_numpy()[0] - tv[0]) < 1e-12
                    and len(pd.read_csv(r.csv_path)) == n
                ):
                    truth = r.poi_times
                    break
            if truth is None:
                truth = {}
            out = {name: {"indices": [-1], "confidences": [-1]} for name in POI_MAP.values()}
            for chain, true_t in truth.items():
                prod = POI_MAP[CHAIN_TO_ID[chain]]
                # channel POIs (POI3/4/5 chain) can miss; init POIs never do
                is_channel = chain in ("POI3", "POI4", "POI5")
                if is_channel and self.rng.random() < self.miss_p:
                    continue
                if self.rng.random() < self.gross_p:
                    t_pred = true_t + self.rng.choice([-1, 1]) * self.rng.uniform(2.5, 6.0)
                else:
                    t_pred = true_t + self.rng.normal(0, self.noise_s)
                i = int(np.clip(np.searchsorted(tv, t_pred), 0, n - 1))
                out[prod] = {"indices": [i], "confidences": [float(self.rng.uniform(0.4, 0.95))]}
            return out, max(0, len(truth) - 2)

    # A ~ v6.2 (noisier, more misses, more gross); B ~ v6.3 (tighter)
    ctl_a = _MockController(noise_s=0.6, miss_p=0.10, gross_p=0.14, seed_off=1)
    ctl_b = _MockController(noise_s=0.3, miss_p=0.04, gross_p=0.06, seed_off=2)

    summary = run_benchmark(
        ctl_a,
        ctl_b,
        runs,
        output_dir,
        label_a="v6.2(mock)",
        label_b="v6.3(mock)",
        gross_threshold=2.0,
    )
    # plumbing assertions
    assert summary["n_runs"] == len(runs)
    total_n = sum(summary["global_a"][p].n for p in POI_KEYS)
    assert total_n > 0, "no paired comparisons recorded"
    expected = {"ab_metrics.csv", "ab_metrics_by_tier.csv", "per_run_results.csv"}
    present = {p.name for p in Path(output_dir).iterdir()}
    assert expected <= present, f"missing outputs: {expected - present}"
    print("SELFTEST OK — cross-version pairing, tiers, miss accounting and CSVs verified.")


# ===========================================================================
#  CLI
# ===========================================================================


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--raw-root", type=Path, default=Path("data"), help="data root of run directories"
    )
    ap.add_argument(
        "--assets-v62",
        type=Path,
        default=Path("6_2_assets.json"),
        help="model_assets JSON for v6.2 (v6_yolo.py)",
    )
    ap.add_argument(
        "--assets-v63",
        type=Path,
        default=Path("6_3_assets.json"),
        help="model_assets JSON for v6.3 (controller.py)",
    )
    ap.add_argument(
        "--prior-v62", type=Path, default=None, help="override spacing_prior path for v6.2"
    )
    ap.add_argument(
        "--prior-v63", type=Path, default=None, help="override spacing_prior path for v6.3"
    )
    ap.add_argument(
        "--v62-module",
        type=Path,
        default=Path("v6_2_yolo/v6_yolo.py"),
        help="path to the v6.2 controller module (inside the v6_2_yolo package)",
    )
    ap.add_argument(
        "--v63-module",
        type=Path,
        default=Path("v6_3_yolo/controller.py"),
        help="path to the v6.3 controller module (inside the v6_3_yolo package)",
    )
    ap.add_argument("--v62-class", type=str, default="QModelV6YOLO")
    ap.add_argument("--v63-class", type=str, default="QModel")
    ap.add_argument("--output", type=Path, default=Path("benchmark_v62_v63_out"))
    ap.add_argument("--n-runs", type=int, default=None)
    ap.add_argument("--gross-threshold", type=float, default=2.0)
    ap.add_argument("--tie-band", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    # Per-version predict() modes. Each release is benchmarked in its intended
    # production configuration by default: both decode the config; only v6.3
    # has the zoom-refine stage.
    ap.add_argument("--v62-decode", dest="v62_decode", action="store_true", default=True)
    ap.add_argument("--no-v62-decode", dest="v62_decode", action="store_false")
    ap.add_argument("--v63-decode", dest="v63_decode", action="store_true", default=True)
    ap.add_argument("--no-v63-decode", dest="v63_decode", action="store_false")
    ap.add_argument("--v63-refine", dest="v63_refine", action="store_true", default=True)
    ap.add_argument("--no-v63-refine", dest="v63_refine", action="store_false")
    ap.add_argument("--selftest", action="store_true", help="run synthetic end-to-end self-test")
    args = ap.parse_args()

    if args.selftest:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            _selftest(Path(td) / "corpus", args.output / "selftest")
        return

    if not args.assets_v62 or not args.assets_v63:
        ap.error("--assets-v62 and --assets-v63 are required (or use --selftest)")

    assets_a = json.loads(Path(args.assets_v62).read_text())
    assets_b = json.loads(Path(args.assets_v63).read_text())
    if args.prior_v62:
        assets_a["spacing_prior"] = str(args.prior_v62)
    if args.prior_v63:
        assets_b["spacing_prior"] = str(args.prior_v63)

    # Make the controllers' `from QATCH...` top-level imports survive outside a
    # QATCH install, so their headless fallbacks fire (see _install_qatch_stub).
    _install_qatch_stub()

    LOG.info("Loading v6.2 controller from %s", args.v62_module)
    mod_a = _load_controller_module(
        args.v62_module, package_name="v6_2_yolo", module_stem=args.v62_module.stem
    )
    LOG.info("Loading v6.3 controller from %s", args.v63_module)
    mod_b = _load_controller_module(
        args.v63_module, package_name="v6_3_yolo", module_stem=args.v63_module.stem
    )

    controller_a = _instantiate_controller(mod_a, args.v62_class, assets_a)
    controller_b = _instantiate_controller(mod_b, args.v63_class, assets_b)

    runs = discover_runs(args.raw_root)
    if not runs:
        LOG.error("No runs found under %s", args.raw_root)
        return
    LOG.info("Discovered %d runs", len(runs))

    # v6.2's predict() has no refine_pois kwarg; v6.3 adds it.
    pk_a = dict(decode_config=args.v62_decode)
    pk_b = dict(decode_config=args.v63_decode, refine_pois=args.v63_refine)

    run_benchmark(
        controller_a,
        controller_b,
        runs,
        args.output,
        label_a="v6.2",
        label_b="v6.3",
        predict_kwargs_a=pk_a,
        predict_kwargs_b=pk_b,
        n_runs=args.n_runs,
        gross_threshold=args.gross_threshold,
        tie_band=args.tie_band,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
