"""
spacing_prior.py
================

Part of the QModel V6 YOLO predictor subpackage (v6.3.0).

A flat, pairwise configuration prior over POI positions, learned from
complete-fill ground-truth configurations. This is the model the EDA selected:
flat (not run-conditional) and pairwise (not autoregressive), because on the
real data those additions hurt rather than helped.

What it encodes
---------------
For each consecutive POI pair (POI_k -> POI_{k+1}) it learns the distribution
of the GAP between them, in a scale that the EDA showed is stable. Two gap
parameterisations are supported and blended:

  * absolute gap in seconds (good where event timing is physically anchored)
  * gap as a fraction of the run's total POI span (scale-free; helps across
    runs of very different duration)

At decode time the prior scores a candidate configuration by the summed
log-likelihood of its gaps, plus hard feasibility (monotonic order, learned
min/max gap bounds). The score combines with YOLO detection confidence in the
DP decoder (see decode.py).

It is deliberately simple and interpretable: each gap is modelled as a
log-normal (gaps are positive and right-skewed), which the DP turns into an
additive quadratic-in-log penalty. Nothing here needs the raw signal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# POI order; consecutive pairs define the gaps the prior models.
POI_ORDER = ["POI1", "POI2", "POI3", "POI4", "POI5"]


@dataclass
class GapStat:
    """Log-normal stats for one consecutive-gap, in seconds and span-fraction."""

    # log-seconds
    log_mu_sec: float
    log_sd_sec: float
    # log-fraction-of-span
    log_mu_frac: float
    log_sd_frac: float
    # hard feasibility bounds (seconds), from robust percentiles
    min_gap_sec: float
    max_gap_sec: float
    n: int


@dataclass
class SpacingPrior:
    pairs: List[str]  # e.g. ["POI1->POI2", ...]
    gap: Dict[str, GapStat] = field(default_factory=dict)
    # blend weight between seconds-based and fraction-based log-likelihood.
    # 0 = pure seconds, 1 = pure span-fraction. Default mixes both.
    frac_blend: float = 0.5
    # feasibility bound percentiles used when fitting.
    bound_lo_pct: float = 0.5
    bound_hi_pct: float = 99.5

    # ------------------------------------------------------------------ fit
    @staticmethod
    def fit(
        configs_sec: np.ndarray,
        frac_blend: float = 0.5,
        bound_lo_pct: float = 0.5,
        bound_hi_pct: float = 99.5,
    ) -> "SpacingPrior":
        """Fit from complete configurations.

        configs_sec: (N, P) array of POI times in seconds, strictly ascending
        rows (complete fills only). P must equal len(POI_ORDER).
        """
        N, P = configs_sec.shape
        assert P == len(POI_ORDER), f"expected {len(POI_ORDER)} POIs, got {P}"
        span = configs_sec[:, -1] - configs_sec[:, 0]
        span = np.where(span < 1e-9, np.nan, span)
        pairs = [f"{POI_ORDER[i]}->{POI_ORDER[i+1]}" for i in range(P - 1)]
        prior = SpacingPrior(
            pairs=pairs, frac_blend=frac_blend, bound_lo_pct=bound_lo_pct, bound_hi_pct=bound_hi_pct
        )
        for i in range(P - 1):
            g_sec = configs_sec[:, i + 1] - configs_sec[:, i]
            g_sec = g_sec[np.isfinite(g_sec) & (g_sec > 0)]
            g_frac = (configs_sec[:, i + 1] - configs_sec[:, i]) / span
            g_frac = g_frac[np.isfinite(g_frac) & (g_frac > 0)]
            ls = np.log(g_sec)
            lf = np.log(g_frac)
            prior.gap[pairs[i]] = GapStat(
                log_mu_sec=float(ls.mean()),
                log_sd_sec=float(ls.std() + 1e-6),
                log_mu_frac=float(lf.mean()),
                log_sd_frac=float(lf.std() + 1e-6),
                min_gap_sec=float(np.percentile(g_sec, bound_lo_pct)),
                max_gap_sec=float(np.percentile(g_sec, bound_hi_pct)),
                n=int(len(g_sec)),
            )
        return prior

    # ----------------------------------------------------- composed gaps
    def composed_stat(self, i: int, j: int) -> GapStat:
        """Stats for the composed gap POI_ORDER[i] -> POI_ORDER[j] (j > i),
        built by composing the consecutive-gap log-normals: medians add in
        linear space, variances add in log space (a standard log-normal-sum
        approximation). Exact (the fitted stat) for j == i + 1.

        Used when two *present* POIs are not globally adjacent, so the gap
        between them spans one or more absent POIs and must be scored against
        the composition of the intervening fitted gaps, not against the first
        gap's stats alone.
        """
        if j <= i:
            raise ValueError(f"composed_stat requires j > i, got ({i}, {j})")
        if j == i + 1:
            return self.gap[self.pairs[i]]
        cache = getattr(self, "_composed_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_composed_cache", cache)
        key = (i, j)
        if key in cache:
            return cache[key]
        ks = range(i, j)
        med_sec = sum(np.exp(self.gap[self.pairs[k]].log_mu_sec) for k in ks)
        var_sec = sum(self.gap[self.pairs[k]].log_sd_sec ** 2 for k in ks)
        med_frac = sum(np.exp(self.gap[self.pairs[k]].log_mu_frac) for k in ks)
        var_frac = sum(self.gap[self.pairs[k]].log_sd_frac ** 2 for k in ks)
        gs = GapStat(
            log_mu_sec=float(np.log(max(med_sec, 1e-9))),
            log_sd_sec=float(np.sqrt(var_sec) + 1e-6),
            log_mu_frac=float(np.log(min(max(med_frac, 1e-9), 0.999))),
            log_sd_frac=float(np.sqrt(var_frac) + 1e-6),
            min_gap_sec=float(sum(self.gap[self.pairs[k]].min_gap_sec for k in ks)),
            max_gap_sec=float(sum(self.gap[self.pairs[k]].max_gap_sec for k in ks)),
            n=int(min(self.gap[self.pairs[k]].n for k in ks)),
        )
        cache[key] = gs
        return gs

    # -------------------------------------------------------------- scoring
    def _stat_loglik(self, gs: GapStat, gap_sec: float, span_sec: float) -> float:
        if gap_sec <= 0:
            return -1e9
        # seconds log-normal log-density (drop constants; keep shape)
        z_sec = (np.log(gap_sec) - gs.log_mu_sec) / gs.log_sd_sec
        ll_sec = -0.5 * z_sec * z_sec - np.log(gs.log_sd_sec)
        # fraction log-normal
        if span_sec and span_sec > 0:
            frac = gap_sec / span_sec
            if frac <= 0:
                ll_frac = -1e9
            else:
                z_f = (np.log(frac) - gs.log_mu_frac) / gs.log_sd_frac
                ll_frac = -0.5 * z_f * z_f - np.log(gs.log_sd_frac)
        else:
            ll_frac = ll_sec
        return float((1 - self.frac_blend) * ll_sec + self.frac_blend * ll_frac)

    def gap_loglik(self, pair_idx: int, gap_sec: float, span_sec: float) -> float:
        """Log-likelihood of one consecutive gap (higher = more plausible).
        Blends the seconds and span-fraction log-normal densities. Pass
        span_sec <= 0 to disable the fraction component (falls back to the
        seconds density), e.g. on partial fills where the fitted span
        semantics (POI1..POI5 of complete fills) do not apply."""
        return self._stat_loglik(self.gap[self.pairs[pair_idx]], gap_sec, span_sec)

    def gap_loglik_between(self, i: int, j: int, gap_sec: float, span_sec: float) -> float:
        """Log-likelihood of the gap between global POI indices i and j
        (j > i). Uses the fitted stat when adjacent, the composed stat when
        the pair spans absent POIs."""
        return self._stat_loglik(self.composed_stat(i, j), gap_sec, span_sec)

    def gap_loglik_scoped(
        self, i: int, j: int, gap_sec: float, span_sec: float, span_lo: int, span_hi: int
    ) -> float:
        """Log-likelihood of the gap POI_ORDER[i] -> POI_ORDER[j], with the
        span-fraction component re-referenced to the span between global POI
        indices span_lo .. span_hi (the first/last *placed* POIs at decode
        time).

        The fitted frac stats assume span = t(POI_last) - t(POI_first) of a
        COMPLETE fill. On a prefix (partial) fill the decode-time span covers
        fewer gaps, so the fitted frac medians are systematically too small.
        This method re-derives the frac location from the seconds medians:

            log_mu_frac(i->j | span_lo..span_hi)
                = log( median_sec(i->j) / sum_k median_sec(k),  k in [span_lo, span_hi) )

        and keeps the fitted frac dispersion (the best available estimate of
        relative-scale spread). When the scope IS the full chain, the fitted
        frac stats are used unchanged. This is what lets the prior reason
        about viscosity: the observed early gaps anchor the run's scale, and
        every other gap must be proportionate to that scale — a compressed
        (fast-fill-shaped) decoy configuration makes the fixed early gaps an
        implausibly large fraction of the span and is heavily penalised,
        without making the prior run-conditional.
        """
        full = span_lo == 0 and span_hi == len(POI_ORDER) - 1
        base = self.composed_stat(i, j)
        if full or span_sec <= 0:
            return self._stat_loglik(base, gap_sec, span_sec)
        p_med = sum(
            float(np.exp(self.gap[self.pairs[k]].log_mu_sec)) for k in range(span_lo, span_hi)
        )
        if p_med <= 0:
            return self._stat_loglik(base, gap_sec, 0.0)
        scoped = GapStat(
            log_mu_sec=base.log_mu_sec,
            log_sd_sec=base.log_sd_sec,
            log_mu_frac=float(base.log_mu_sec - np.log(p_med)),
            log_sd_frac=base.log_sd_frac,
            min_gap_sec=base.min_gap_sec,
            max_gap_sec=base.max_gap_sec,
            n=base.n,
        )
        return self._stat_loglik(scoped, gap_sec, span_sec)

    def gap_feasible(self, pair_idx: int, gap_sec: float, slack: float = 1.5) -> bool:
        """Hard feasibility: gap within learned [min,max] bounds, with slack
        (multiplicative) so we don't reject borderline-but-valid gaps."""
        gs = self.gap[self.pairs[pair_idx]]
        return (
            gap_sec > 0 and gap_sec >= gs.min_gap_sec / slack and gap_sec <= gs.max_gap_sec * slack
        )

    def gap_feasible_between(self, i: int, j: int, gap_sec: float, slack: float = 1.5) -> bool:
        """Hard feasibility between global POI indices i and j (j > i),
        composing the intervening bounds when the pair is non-adjacent."""
        gs = self.composed_stat(i, j)
        return (
            gap_sec > 0 and gap_sec >= gs.min_gap_sec / slack and gap_sec <= gs.max_gap_sec * slack
        )

    # ----------------------------------------------------------- whole config
    def config_loglik(self, times_sec: List[float]) -> float:
        """Total spacing log-likelihood of a full ordered configuration."""
        span = times_sec[-1] - times_sec[0]
        total = 0.0
        for i in range(len(times_sec) - 1):
            total += self.gap_loglik(i, times_sec[i + 1] - times_sec[i], span)
        return total

    # --------------------------------------------------------------- persist
    def save(self, path: Path) -> None:
        d = {
            "pairs": self.pairs,
            "frac_blend": self.frac_blend,
            "bound_lo_pct": self.bound_lo_pct,
            "bound_hi_pct": self.bound_hi_pct,
            "gap": {k: asdict(v) for k, v in self.gap.items()},
        }
        Path(path).write_text(json.dumps(d, indent=2))

    @staticmethod
    def load(path: Path) -> "SpacingPrior":
        d = json.loads(Path(path).read_text())
        p = SpacingPrior(
            pairs=d["pairs"],
            frac_blend=d["frac_blend"],
            bound_lo_pct=d["bound_lo_pct"],
            bound_hi_pct=d["bound_hi_pct"],
        )
        p.gap = {k: GapStat(**v) for k, v in d["gap"].items()}
        return p
