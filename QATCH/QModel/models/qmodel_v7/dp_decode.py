"""
dp_decode.py
============

Joint configuration decode over YOLO candidate detections.

Problem
-------
The production cascade keeps only the single highest-confidence detection per
POI (``best_dets``) and cuts the signal at it. That greedy, per-POI choice is
exactly what produces locally-plausible-but-globally-illogical placements, and
because the cascade *cuts* on each pick, one bad early detection corrupts every
later one.

This module replaces the greedy pick with a global decode: given several
CANDIDATE detections per present POI (time + confidence), choose one candidate
per POI such that the configuration is

    (a) strictly time-ordered (hard),
    (b) feasible under the learned per-gap bounds (hard, with slack),
    (c) maximal in   sum(detection confidence)  +  lambda * spacing log-lik,

where the spacing log-likelihood comes from the learned SpacingPrior. This is
the EDA-selected design: flat, pairwise prior + monotonic constraint, solved
exactly by dynamic programming over the candidate lattice.

Span handling (important)
-------------------------
The prior's span-fraction component was fitted on COMPLETE fills with
span = t(POI5) - t(POI1). Two consequences honoured here:

  * On partial (prefix) fills the fraction component is RE-REFERENCED to
    the span of the present prefix via SpacingPrior.gap_loglik_scoped (the
    fitted frac medians assume the complete-fill span and would otherwise be
    systematically biased). This keeps the scale-coupling — the observed
    early gaps anchor the run's scale and later gaps must be proportionate —
    active on partial fills, where it is the main defence against
    fast-fill-shaped decoy configurations on slow (high-viscosity) runs.
    With fewer than three placed POIs there is no ratio information and the
    density is seconds-only.
  * The span is not known until the configuration is chosen, and a
    per-edge "span so far" proxy breaks DP optimality. But the span is
    FULLY DETERMINED by the first and last placed candidates, so the decode
    conditions on that pair: for each (first, last) candidate combination it
    runs an exact DP with the span fixed, and keeps the best consistently
    re-scored configuration. With the per-POI candidate cap this is at most
    K^2 small DPs and yields the exact argmax of the blended objective —
    no fixed-point approximation that could converge to the wrong basin
    (e.g. a compressed fast-fill-shaped decoy on a slow run).

Non-adjacent present POIs (e.g. POI2 and POI4 present with POI3 unplaceable)
are scored against the COMPOSITION of the intervening fitted gaps via
SpacingPrior.composed_stat, not against the first gap's stats.

It only decodes the POIs that are actually present (the fill-count gate /
type_cls has already decided how many channels exist), so partial fills are
handled by passing fewer POIs — no spurious late POIs are introduced.

Inputs / outputs are plain dicts so this drops in alongside the existing
predictor without depending on its internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from QATCH.QModel.models.qmodel_v7.spacing_prior import SpacingPrior, POI_ORDER
except (ImportError, ModuleNotFoundError):
    from spacing_prior import SpacingPrior, POI_ORDER


@dataclass
class Candidate:
    time: float  # detected timestamp (seconds)
    conf: float  # YOLO confidence in [0,1]


@dataclass
class DecodeResult:
    chosen: Dict[str, Candidate]  # poi_name -> chosen candidate
    total_score: float
    spacing_loglik: float
    conf_sum: float
    feasible: bool  # False if no fully-feasible path existed
    fallback_used: bool  # True if we relaxed to greedy/partial


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _prep_candidates(
    candidates: Dict[str, List[Candidate]],
    present: List[str],
    min_conf: float,
    max_candidates: int,
) -> Dict[str, List[Candidate]]:
    """Filter, dedupe, cap and time-sort candidate lists per present POI."""
    cand: Dict[str, List[Candidate]] = {}
    for p in present:
        cs = [c for c in candidates.get(p, []) if c.conf >= min_conf]
        if not cs:
            cs = list(candidates.get(p, []))  # don't drop a POI entirely
        # dedupe identical timestamps (coarse+fine stages can re-emit the same
        # box); keep the max-confidence instance.
        by_t: Dict[float, Candidate] = {}
        for c in cs:
            key = round(c.time, 6)
            if key not in by_t or c.conf > by_t[key].conf:
                by_t[key] = c
        cs = list(by_t.values())
        # cap to top-K by confidence (keeps the lattice small and prunes
        # noise-floor boxes), then sort by time for the DP.
        cs.sort(key=lambda c: c.conf, reverse=True)
        cs = cs[: max(1, max_candidates)]
        cs.sort(key=lambda c: c.time)
        cand[p] = cs
    return cand


def _score_config(
    chosen: Dict[str, Candidate],
    placeable: List[str],
    prior: SpacingPrior,
    lam: float,
    conf_weight: float,
    use_frac: bool,
) -> Tuple[float, float, float]:
    """Consistently (re-)score a configuration. Returns (total, spacing_ll, conf_sum)."""
    g_index = {POI_ORDER[i]: i for i in range(len(POI_ORDER))}
    placed = [p for p in placeable if p in chosen]
    conf_sum = conf_weight * sum(_clip01(chosen[p].conf) for p in placed)
    times = [chosen[p].time for p in placed]
    span = (max(times) - min(times)) if (use_frac and len(times) > 1) else 0.0
    span_lo = g_index[placed[0]] if placed else 0
    span_hi = g_index[placed[-1]] if placed else 0
    spacing_ll = 0.0
    for a, b in zip(placed[:-1], placed[1:]):
        spacing_ll += prior.gap_loglik_scoped(
            g_index[a], g_index[b], chosen[b].time - chosen[a].time, span, span_lo, span_hi
        )
    return conf_sum + lam * spacing_ll, spacing_ll, conf_sum


def _dp_pass(
    cand: Dict[str, List[Candidate]],
    placeable: List[str],
    prior: SpacingPrior,
    lam: float,
    conf_weight: float,
    feas_slack: float,
    require_feasible: bool,
    span_for_frac: float,
) -> Optional[Dict[str, Candidate]]:
    """One exact DP over the lattice. span_for_frac <= 0 disables the
    fraction component (seconds-only). Returns the argmax configuration or
    None if no admissible ordered path exists."""
    g_index = {POI_ORDER[i]: i for i in range(len(POI_ORDER))}
    span_lo, span_hi = g_index[placeable[0]], g_index[placeable[-1]]
    P = len(placeable)
    dp: List[List[float]] = [[-1e18] * len(cand[placeable[j]]) for j in range(P)]
    back: List[List[int]] = [[-1] * len(cand[placeable[j]]) for j in range(P)]

    for k, c in enumerate(cand[placeable[0]]):
        dp[0][k] = conf_weight * _clip01(c.conf)

    for j in range(1, P):
        a, b = placeable[j - 1], placeable[j]
        gi, gj = g_index[a], g_index[b]
        for k, cb in enumerate(cand[b]):
            best = -1e18
            best_prev = -1
            for kp, ca in enumerate(cand[a]):
                if dp[j - 1][kp] <= -1e17:
                    continue
                gap = cb.time - ca.time
                if gap <= 0:
                    continue  # hard: strict ordering
                if require_feasible and not prior.gap_feasible_between(gi, gj, gap, feas_slack):
                    continue  # hard: gap bounds
                ll = prior.gap_loglik_scoped(gi, gj, gap, span_for_frac, span_lo, span_hi)
                score = dp[j - 1][kp] + lam * ll + conf_weight * _clip01(cb.conf)
                if score > best:
                    best = score
                    best_prev = kp
            dp[j][k] = best
            back[j][k] = best_prev

    last = P - 1
    if not dp[last]:
        return None
    best_k = int(np.argmax(dp[last]))
    if dp[last][best_k] <= -1e17:
        return None

    chosen: Dict[str, Candidate] = {}
    k = best_k
    for j in range(last, -1, -1):
        if k < 0:
            return None  # defensive: broken backtrack chain
        chosen[placeable[j]] = cand[placeable[j]][k]
        k = back[j][k] if j > 0 else -1
    return chosen


def _span_conditioned_decode(
    cand: Dict[str, List[Candidate]],
    placeable: List[str],
    prior: SpacingPrior,
    lam: float,
    conf_weight: float,
    feas_slack: float,
    require_feasible: bool,
) -> Optional[Dict[str, Candidate]]:
    """Exact decode of the blended (seconds + span-fraction) objective.

    The span-fraction denominators depend only on the first and last placed
    candidates, so we enumerate those pairs, fix the span, and run an exact
    DP for each. Returns the best configuration under consistent re-scoring,
    or None if no admissible path exists for any pair."""
    first, last = placeable[0], placeable[-1]
    best_chosen: Optional[Dict[str, Candidate]] = None
    best_score = -1e18
    for f in cand[first]:
        for l in cand[last]:
            span = l.time - f.time
            if span <= 0:
                continue
            sub = dict(cand)
            sub[first] = [f]
            sub[last] = [l]
            ch = _dp_pass(
                sub,
                placeable,
                prior,
                lam,
                conf_weight,
                feas_slack,
                require_feasible,
                span_for_frac=span,
            )
            if ch is None:
                continue
            s = _score_config(ch, placeable, prior, lam, conf_weight, True)[0]
            if s > best_score:
                best_score = s
                best_chosen = ch
    return best_chosen


def dp_decode(
    candidates: Dict[str, List[Candidate]],
    present_pois: Sequence[str],
    prior: SpacingPrior,
    lam: float = 1.0,
    conf_weight: float = 1.0,
    feas_slack: float = 1.5,
    min_conf: float = 0.0,
    require_feasible: bool = True,
    max_candidates: int = 10,
) -> DecodeResult:
    """Decode the optimal ordered configuration.

    candidates:   poi_name -> list of Candidate (any order); only present_pois
                  are used. Each present POI must have >=1 candidate.
    present_pois: ordered subset of POI_ORDER that type_cls says are present.
    lam:          weight on the spacing prior relative to confidence.
    conf_weight:  weight on summed detection confidence.
    feas_slack:   multiplicative slack on learned gap bounds.
    min_conf:     drop candidates below this confidence before decoding.
    require_feasible: if True, only fully gap-feasible paths are allowed; if no
                  such path exists we fall back (see fallback_used).
    max_candidates: per-POI cap (top-K by confidence) on the lattice width.

    Returns the best DecodeResult. Spacing terms are applied between
    consecutive *present* POIs using the global POI_ORDER indices, composing
    intervening fitted gaps when the pair is not globally adjacent. The
    span-fraction component of the prior is used whenever at least three
    POIs are placeable, with prefix-scoped span semantics (see module doc).
    """
    present = [p for p in POI_ORDER if p in present_pois]
    if not present:
        return DecodeResult({}, -1e18, 0.0, 0.0, False, True)

    cand = _prep_candidates(candidates, present, min_conf, max_candidates)
    placeable = [p for p in present if cand[p]]
    if not placeable:
        return DecodeResult({}, -1e18, 0.0, 0.0, False, True)

    # The frac component needs at least two gaps to carry ratio information
    # (with a single gap, gap/span == 1 identically). Prefix-fill span
    # semantics are handled by gap_loglik_scoped, so partial fills are
    # eligible too.
    use_frac = len(placeable) >= 3 and prior.frac_blend > 0

    # ---- decode: exact under the full objective. With frac active the
    # decode is span-conditioned (see _span_conditioned_decode); otherwise a
    # single seconds-only DP is already exact.
    if use_frac:
        chosen1 = _span_conditioned_decode(
            cand, placeable, prior, lam, conf_weight, feas_slack, require_feasible
        )
    else:
        chosen1 = _dp_pass(
            cand,
            placeable,
            prior,
            lam,
            conf_weight,
            feas_slack,
            require_feasible,
            span_for_frac=0.0,
        )

    if chosen1 is None and require_feasible:
        # Relax feasibility (keep only strict ordering) and decode once more.
        relaxed = _dp_pass(cand, placeable, prior, lam, conf_weight, 1e9, False, span_for_frac=0.0)
        if relaxed is None or len(relaxed) < len(placeable):
            # Even strict ordering has no complete path -> production-safe
            # floor: per-POI greedy, never worse than current behaviour.
            return _greedy_result(cand, placeable, prior, lam, conf_weight)
        total, sll, csum = _score_config(relaxed, placeable, prior, lam, conf_weight, use_frac)
        return DecodeResult(relaxed, total, sll, csum, False, True)

    if chosen1 is None:
        return DecodeResult({}, -1e18, 0.0, 0.0, False, True)

    total, sll, csum = _score_config(chosen1, placeable, prior, lam, conf_weight, use_frac)
    return DecodeResult(chosen1, total, sll, csum, True, False)


def _greedy_result(
    cand: Dict[str, List[Candidate]],
    placeable: List[str],
    prior: SpacingPrior,
    lam: float,
    conf_weight: float,
) -> DecodeResult:
    """Per-POI greedy selection (highest confidence), used as the production-safe
    floor when no ordered/feasible joint path exists. Never drops a POI that has
    candidates, so the decoder is never worse than current behaviour."""
    chosen = {p: max(cand[p], key=lambda c: c.conf) for p in placeable if cand[p]}
    total, sll, csum = _score_config(chosen, placeable, prior, lam, conf_weight, False)
    return DecodeResult(chosen, total, sll, csum, False, True)


def score_configuration(
    chosen: Dict[str, Candidate],
    prior: SpacingPrior,
    lam: float = 1.0,
    conf_weight: float = 1.0,
) -> float:
    """Score an arbitrary configuration under the SAME objective and frac
    rules dp_decode uses, so external configurations (e.g. the production
    cascade's picks) are directly comparable to DecodeResult.total_score.
    Used by the controller's accept-margin (hysteresis) test."""
    placeable = [p for p in POI_ORDER if p in chosen]
    if not placeable:
        return -1e18
    use_frac = len(placeable) >= 3 and prior.frac_blend > 0
    return _score_config(chosen, placeable, prior, lam, conf_weight, use_frac)[0]


def greedy_baseline(
    candidates: Dict[str, List[Candidate]], present_pois: Sequence[str]
) -> Dict[str, Candidate]:
    """The current production behaviour: pick the highest-confidence candidate
    per POI independently (no joint reasoning). Used for A/B comparison."""
    out = {}
    for p in present_pois:
        cs = candidates.get(p, [])
        if cs:
            out[p] = max(cs, key=lambda c: c.conf)
    return out
