"""
decode.py
=========

Part of the QModel V6 YOLO predictor subpackage (v6.4.0).
Formerly ``dp_decode.py`` (v6.3).

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

where the spacing log-likelihood comes from the learned SpacingPrior.

What changed in v6.4
--------------------
The spacing term is now the JOINT scale-shape log-likelihood (one-factor
log-normal over the gaps; see spacing_prior.py), not a sum of independent
per-gap log-normals. The decoder detects a joint-capable prior
(``prior.version >= 2``) and, if present, scores each configuration by its
exact joint density. This is what fixes the non-Newtonian regressions: a
uniformly slow (viscous) run is no longer penalised for having large absolute
gaps, only for having the WRONG SHAPE, so the decode stops losing to greedy on
slow fills while still rejecting genuinely incoherent (shape-broken) decoys
harder than the old prior did.

Exactness
---------
For the one-factor joint model the configuration density factorises by the
chain rule into per-edge CONDITIONALS, and the conditional of the next gap
given all placed gaps depends on the history ONLY through the posterior of the
single shared latent — two scalars (a precision and a precision-weighted
mean). So the left-to-right DP carries those two scalars in its state and
remains EXACT: accumulating conditionals telescopes to the true joint density.
As in v6.3 the span (needed only for the legacy frac component / flat priors)
is fixed by conditioning on the (first,last) candidate pair, so the whole
decode is at most K^2 small exact DPs.

Backward compatibility
-----------------------
A flat (v6.3) prior still decodes exactly as before via the seconds/frac
``gap_loglik_scoped`` path — the joint path is only taken when the loaded
prior carries the joint parameters. Inputs/outputs are unchanged plain dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .spacing_prior import SpacingPrior, POI_ORDER
except ImportError:  # flat / headless execution (not imported as a package)
    from spacing_prior import SpacingPrior, POI_ORDER

_FLOOR = -1e18


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
        by_t: Dict[float, Candidate] = {}
        for c in cs:
            key = round(c.time, 6)
            if key not in by_t or c.conf > by_t[key].conf:
                by_t[key] = c
        cs = list(by_t.values())
        cs.sort(key=lambda c: c.conf, reverse=True)
        cs = cs[: max(1, max_candidates)]
        cs.sort(key=lambda c: c.time)
        cand[p] = cs
    return cand


# ---------------------------------------------------------------------------
#  Joint one-factor edge scoring with a 2-scalar latent-posterior DP state.
#
#  Model (per gap, log-space):  log g_k = mu_k + beta_k * s + eps_k,
#  s ~ N(0,1), eps_k ~ N(0, psi_k). After observing gaps with composed params
#  (mu_a, beta_a, psi_a) and values x_a = log(gap_a), the latent posterior is
#  Gaussian with precision  P = 1 + sum beta_a^2/psi_a  and natural mean
#  h = sum beta_a (x_a - mu_a)/psi_a, so  E[s|.] = h/P, Var = 1/P.
#
#  The conditional density of a NEW gap (mu_q, beta_q, psi_q) with value x_q:
#     pred_mean = mu_q + beta_q * (h/P)
#     pred_var  = psi_q + beta_q^2 * (1/P + 1)   ... wait: marginal of eps plus
#                 latent contribution. Derivation below in code comments.
# ---------------------------------------------------------------------------


def _gap_blocks(placed_global: List[int]) -> List[Tuple[int, int]]:
    """Map a list of global POI indices (ascending) to the composed gap blocks
    [lo, hi) for each consecutive present pair."""
    return [(placed_global[i], placed_global[i + 1]) for i in range(len(placed_global) - 1)]


# ---------------------------------------------------------------------------
#  EXACT one-factor joint DP via Sherman-Morrison + theta-linearisation.
#
#  With Sigma = beta beta^T + diag(psi), the Mahalanobis term factorises
#  (Sherman-Morrison):
#
#     (x-mu)^T Sigma^-1 (x-mu)
#         = sum_i (x_i-mu_i)^2/psi_i  -  A^2 / Pden ,
#     A    = sum_i beta_i (x_i - mu_i)/psi_i        (ADDITIVE across edges)
#     Pden = 1 + sum_i beta_i^2/psi_i               (const for a fixed present set)
#
#  Every term is additive per edge EXCEPT -A^2/Pden. Linearise it:
#
#     -A^2/Pden = min_theta ( theta^2 - 2 theta A ) / Pden     (opt at theta=A)
#
#  For a FIXED theta the full objective is additive per edge, so an exact DP
#  finds the best configuration; sweeping theta on a grid spanning the feasible
#  A-range and keeping the best configuration recovers the exact joint argmax
#  (each theta gives a valid configuration whose TRUE joint score we then
#  compute; the max over the grid is the global optimum once the grid brackets
#  A* — which it does, since A is bounded by the candidate lattice).
# ---------------------------------------------------------------------------


def _edge_additive_terms(prior, lo, hi, gap_sec):
    """Per-edge pieces for the additive (theta-fixed) objective.
    Returns (q_i, a_i, logdet_i) where:
      q_i      = (x-mu)^2/psi                  (the diagonal Mahalanobis part)
      a_i      = beta (x-mu)/psi               (the A accumulator contribution)
      logdet_i = log(psi)                      (per-edge log-det piece)
    plus the constants Pden contribution beta^2/psi via a_coef.
    """
    mu, beta, psi = prior._composed_log_gap_model(lo, hi)
    if gap_sec <= 0:
        return None
    x = float(np.log(gap_sec))
    d = x - mu
    q = d * d / psi
    a = beta * d / psi
    pden_inc = beta * beta / psi
    return q, a, float(np.log(psi)), pden_inc


def _dp_pass_joint(
    cand: Dict[str, List[Candidate]],
    placeable: List[str],
    prior: SpacingPrior,
    lam: float,
    conf_weight: float,
    feas_slack: float,
    require_feasible: bool,
    theta_grid: int = 11,
) -> Optional[Dict[str, Candidate]]:
    """Exact DP under the JOINT objective via theta-linearisation.

    For each theta on a grid we run a standard additive DP whose edge cost is

        conf(cb) - 0.5 * lam * ( q_edge - 2*theta*a_edge / Pden )

    (the per-edge logdet and the +theta^2/Pden and -lam*0.5*const pieces are
    configuration-independent and folded in at the end / are constant across
    candidates, so they don't affect the per-theta argmax). Each theta yields a
    concrete configuration; we re-score every candidate configuration under the
    TRUE joint objective (via _score_config) and keep the global best. Because
    -A^2/Pden = min_theta(theta^2-2 theta A)/Pden is tight at theta=A and the
    grid brackets the feasible A-range, the best-over-grid configuration is the
    exact joint argmax."""
    g_index = {POI_ORDER[i]: i for i in range(len(POI_ORDER))}
    global_idx = [g_index[p] for p in placeable]
    blocks = _gap_blocks(global_idx)
    P = len(placeable)

    # Pden is constant for this present set.
    Pden = 1.0
    for lo, hi in blocks:
        _, beta, psi = prior._composed_log_gap_model(lo, hi)
        Pden += beta * beta / psi

    # Estimate the A-range to bracket the theta grid: A = sum a_edge, each
    # a_edge over the candidate gaps. Use min/max achievable per edge.
    a_lo_total = 0.0
    a_hi_total = 0.0
    for j in range(1, P):
        lo, hi = blocks[j - 1]
        a, b = placeable[j - 1], placeable[j]
        edge_as = []
        for ca in cand[a]:
            for cb in cand[b]:
                gap = cb.time - ca.time
                if gap <= 0:
                    continue
                terms = _edge_additive_terms(prior, lo, hi, gap)
                if terms is None:
                    continue
                edge_as.append(terms[1])
        if edge_as:
            a_lo_total += min(edge_as)
            a_hi_total += max(edge_as)
    if a_hi_total < a_lo_total:
        a_lo_total, a_hi_total = a_hi_total, a_lo_total
    pad = 0.5 * (abs(a_hi_total) + abs(a_lo_total) + 1.0)
    thetas = np.linspace(a_lo_total - pad, a_hi_total + pad, max(3, theta_grid))

    best_chosen: Optional[Dict[str, Candidate]] = None
    best_true = -1e18

    def _true_score(chosen) -> float:
        return _score_config(chosen, placeable, prior, lam, conf_weight, False)[0]

    def _solve(theta):
        return _dp_pass_joint_fixed_theta(
            cand,
            placeable,
            prior,
            lam,
            conf_weight,
            feas_slack,
            require_feasible,
            blocks,
            g_index,
            Pden,
            float(theta),
        )

    _solve_memo: Dict[int, Optional[Dict[str, Candidate]]] = {}

    def _solve_cached(theta):
        key = int(round(theta * 1e4))
        if key not in _solve_memo:
            _solve_memo[key] = _solve(theta)
        return _solve_memo[key]

    def _A_of(chosen) -> float:
        placed = [p for p in placeable if p in chosen]
        g_idx = [g_index[p] for p in placed]
        A = 0.0
        for (lo, hi), (a, b) in zip(_gap_blocks(g_idx), zip(placed[:-1], placed[1:])):
            gap = chosen[b].time - chosen[a].time
            terms = _edge_additive_terms(prior, lo, hi, gap)
            if terms is not None:
                A += terms[1]
        return A

    # For a FIXED configuration the joint score is -(theta-A)^2/Pden + const, a
    # downward parabola in theta peaking at theta=A. The decode maximises the
    # upper envelope of these parabolas over configurations. From each grid
    # seed we run the fixed-point map theta -> A(argmax_config(theta)) to its
    # fixed point; every parabola's apex is reachable from a seed in its basin,
    # so seeding the whole bracketed range and iterating to convergence yields
    # the exact global argmax even with a coarse grid.
    converged: set = set()
    for theta0 in thetas:
        theta = float(theta0)
        seen = set()
        for _ in range(6):
            tkey = int(round(theta * 1e4))
            if tkey in converged:
                break
            chosen = _solve_cached(theta)
            if chosen is None:
                break
            ts = _true_score(chosen)
            if ts > best_true:
                best_true = ts
                best_chosen = chosen
            A = _A_of(chosen)
            key = round(A, 6)
            if key in seen or abs(A - theta) < 1e-6:
                converged.add(int(round(A * 1e4)))
                break
            seen.add(key)
            theta = A
    return best_chosen


def _dp_pass_joint_fixed_theta(
    cand,
    placeable,
    prior,
    lam,
    conf_weight,
    feas_slack,
    require_feasible,
    blocks,
    g_index,
    Pden,
    theta,
) -> Optional[Dict[str, Candidate]]:
    """Additive exact DP for one fixed theta."""
    P = len(placeable)
    dp: List[List[float]] = [[-1e18] * len(cand[placeable[j]]) for j in range(P)]
    back: List[List[int]] = [[-1] * len(cand[placeable[j]]) for j in range(P)]

    for k, c in enumerate(cand[placeable[0]]):
        dp[0][k] = conf_weight * _clip01(c.conf)

    for j in range(1, P):
        lo, hi = blocks[j - 1]
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
                    continue
                if require_feasible and not prior.gap_feasible_between(gi, gj, gap, feas_slack):
                    continue
                terms = _edge_additive_terms(prior, lo, hi, gap)
                if terms is None:
                    continue
                q, a_e, logdet, _ = terms
                # per-edge contribution to -0.5*lam*(quad + logdet) with the
                # theta-linearised cross term:  -A^2/Pden -> (theta^2-2 theta A)/Pden
                edge_quad = q - 2.0 * theta * a_e / Pden
                edge_cost = -0.5 * lam * (edge_quad + logdet + np.log(2.0 * np.pi))
                # soft absolute-scale anchor lives on the FIRST placed gap only;
                # fold it into edge 0 so the DP optimum matches the true score.
                if j == 1 and prior.frac_blend > 0:
                    edge_cost += lam * prior.frac_blend * prior.anchor_loglik(gap)
                score = dp[j - 1][kp] + edge_cost + conf_weight * _clip01(cb.conf)
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
            return None
        chosen[placeable[j]] = cand[placeable[j]][k]
        k = back[j][k] if j > 0 else -1
    return chosen


def _score_config(
    chosen: Dict[str, Candidate],
    placeable: List[str],
    prior: SpacingPrior,
    lam: float,
    conf_weight: float,
    use_frac: bool,
) -> Tuple[float, float, float]:
    """Consistently (re-)score a configuration. Returns (total, spacing_ll, conf_sum).

    Uses the JOINT density when the prior is joint-capable; otherwise the
    legacy seconds/frac blend (identical to v6.3)."""
    g_index = {POI_ORDER[i]: i for i in range(len(POI_ORDER))}
    placed = [p for p in placeable if p in chosen]
    conf_sum = conf_weight * sum(_clip01(chosen[p].conf) for p in placed)
    if len(placed) < 2:
        return conf_sum, 0.0, conf_sum

    if prior._has_joint():
        global_idx = [g_index[p] for p in placed]
        blocks = _gap_blocks(global_idx)
        times = [chosen[p].time for p in placed]
        spacing_ll = prior.config_loglik_joint(times, gap_index=blocks)
        # soft absolute-scale anchor on the first placed gap (weak; weighted
        # by frac_blend) so the scale-free joint can't drift to absurd T.
        if prior.frac_blend > 0:
            anchor_gap = chosen[placed[1]].time - chosen[placed[0]].time
            spacing_ll += prior.frac_blend * prior.anchor_loglik(anchor_gap)
        return conf_sum + lam * spacing_ll, spacing_ll, conf_sum

    # ---- legacy flat path (v6.3) ----
    times = [chosen[p].time for p in placed]
    span = (max(times) - min(times)) if (use_frac and len(times) > 1) else 0.0
    span_lo = g_index[placed[0]]
    span_hi = g_index[placed[-1]]
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
    """One exact DP over the lattice for the LEGACY (flat) objective.
    span_for_frac <= 0 disables the fraction component (seconds-only)."""
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
                    continue
                if require_feasible and not prior.gap_feasible_between(gi, gj, gap, feas_slack):
                    continue
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
            return None
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
    """Exact decode of the LEGACY blended (seconds + span-fraction) objective.
    Enumerates (first,last) candidate pairs to fix the span, runs an exact DP
    per pair, keeps the best consistently re-scored configuration."""
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

    Joint-capable priors (version >= 2) are scored by the exact joint
    scale-shape density (the v6.4 path); flat priors fall back to the v6.3
    seconds/span-fraction objective. Both are exact argmaxes of their
    respective objectives. See module docstring.
    """
    present = [p for p in POI_ORDER if p in present_pois]
    if not present:
        return DecodeResult({}, -1e18, 0.0, 0.0, False, True)

    cand = _prep_candidates(candidates, present, min_conf, max_candidates)
    placeable = [p for p in present if cand[p]]
    if not placeable:
        return DecodeResult({}, -1e18, 0.0, 0.0, False, True)

    joint = prior._has_joint()
    use_frac = (not joint) and len(placeable) >= 3 and prior.frac_blend > 0

    if joint:
        chosen1 = _dp_pass_joint(
            cand,
            placeable,
            prior,
            lam,
            conf_weight,
            feas_slack,
            require_feasible,
        )
    elif use_frac:
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
        if joint:
            relaxed = _dp_pass_joint(cand, placeable, prior, lam, conf_weight, 1e9, False)
        else:
            relaxed = _dp_pass(
                cand, placeable, prior, lam, conf_weight, 1e9, False, span_for_frac=0.0
            )
        if relaxed is None or len(relaxed) < len(placeable):
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
    """Per-POI greedy selection (highest confidence), production-safe floor."""
    chosen = {p: max(cand[p], key=lambda c: c.conf) for p in placeable if cand[p]}
    total, sll, csum = _score_config(chosen, placeable, prior, lam, conf_weight, False)
    return DecodeResult(chosen, total, sll, csum, False, True)


def score_configuration(
    chosen: Dict[str, Candidate],
    prior: SpacingPrior,
    lam: float = 1.0,
    conf_weight: float = 1.0,
) -> float:
    """Score an arbitrary configuration under the SAME objective dp_decode
    uses (joint when available, else legacy), so external configurations (the
    cascade's picks) are directly comparable to DecodeResult.total_score.
    Used by the controller's accept-margin (hysteresis) test."""
    placeable = [p for p in POI_ORDER if p in chosen]
    if not placeable:
        return -1e18
    use_frac = (not prior._has_joint()) and len(placeable) >= 3 and prior.frac_blend > 0
    return _score_config(chosen, placeable, prior, lam, conf_weight, use_frac)[0]


def greedy_baseline(
    candidates: Dict[str, List[Candidate]], present_pois: Sequence[str]
) -> Dict[str, Candidate]:
    """Current production behaviour: highest-confidence candidate per POI."""
    out = {}
    for p in present_pois:
        cs = candidates.get(p, [])
        if cs:
            out[p] = max(cs, key=lambda c: c.conf)
    return out
