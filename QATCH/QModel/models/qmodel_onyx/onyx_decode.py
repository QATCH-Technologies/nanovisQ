"""
QATCH.QModel.models.qmodel_onyx.onyx_decode

Joint configuration decoder for Onyx POI candidate detections.

This module performs global decoding over multiple candidate detections per
point-of-interest (POI) to replace the previous greedy, per-POI selection
strategy. Given candidate timestamps and confidence scores, the decoder selects
the most likely ordered POI configuration using a combination of detection
confidence and learned temporal spacing constraints.

The greedy approach can select individually plausible detections that form an
invalid overall configuration. Because downstream processing depends on the
selected timestamps, an incorrect early selection can propagate errors through
the remainder of the fill sequence. This module instead solves the selection
problem jointly using dynamic programming over the candidate lattice.

The optimization objective is:

    confidence score + lambda * spacing log-likelihood

subject to:

* Candidate timestamps are strictly ordered.
* Inter-POI gaps satisfy learned feasibility bounds (when enabled).
* Gap likelihoods follow the learned :class:`OnyxSpacingPrior`.

The spacing prior combines absolute gap duration statistics and normalized
span-fraction statistics. For complete fills, the span is defined as the time
between the first and last POIs. For partial fills, the span-fraction component
is re-referenced using the currently placed POIs through scoped likelihood
evaluation. This preserves scale relationships while avoiding bias caused by
using complete-fill statistics on incomplete sequences.

Because the span-fraction component depends on the final selected first and
last candidates, span-aware decoding conditions on possible endpoint pairs and
runs exact dynamic programming for each fixed span. This preserves the global
optimum of the blended objective rather than relying on an approximate
fixed-point solution.

Non-adjacent present POIs are evaluated using composed spacing statistics from
the intervening fitted gaps. For example, if an intermediate POI is absent, the
spacing between the surrounding detected POIs is scored against the combined
distribution of the missing intervals rather than a single adjacent gap model.

The decoder only operates on POIs already identified as present by upstream
classification logic. It does not introduce additional POIs, allowing partial
fills to be handled naturally by decoding smaller subsets of the global POI
sequence.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-07-09
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from QATCH.QModel.models.qmodel_onyx.onyx_spacing_prior import (
        POI_ORDER,
        OnyxSpacingPrior,
    )
except (ImportError, ModuleNotFoundError):
    from QATCH.QModel.models.qmodel_onyx.onyx_spacing_prior import (
        POI_ORDER,
        OnyxSpacingPrior,
    )


@dataclass
class Candidate:
    """Represents a single candidate POI detection.

    Stores the detected timestamp and confidence score produced by the object
    detection stage. Multiple candidates may exist for the same POI, and the
    joint decoder evaluates these candidates together to determine the most
    globally consistent configuration.

    Attributes:
        time: Detected POI timestamp in seconds.
        conf: Detection confidence score, typically in the range `[0, 1]`.
    """

    time: float  # detected timestamp (seconds)
    conf: float  # YOLO confidence in [0,1]


@dataclass
class DecodeResult:
    """Stores the result of a joint POI configuration decode.

    Contains the selected candidate for each decoded POI along with the
    components of the optimization objective and status flags describing
    whether the result was obtained through a fully feasible decode or a
    fallback path.

    Attributes:
        chosen: Mapping of POI name to the selected `Candidate` detection.
        total_score: Final objective score combining detection confidence and
            spacing prior likelihood.
        spacing_loglik: Unweighted spacing log-likelihood contribution from the
            selected configuration.
        conf_sum: Total weighted detection confidence contribution from the
            selected candidates.
        feasible: Indicates whether a fully feasible path satisfying all hard
            spacing constraints was found.
        fallback_used: Indicates whether the result required relaxing
            constraints or using a fallback strategy such as greedy selection.
    """

    chosen: Dict[str, Candidate]  # poi_name -> chosen candidate
    total_score: float
    spacing_loglik: float
    conf_sum: float
    feasible: bool  # False if no fully-feasible path existed
    fallback_used: bool  # True if we relaxed to greedy/partial


Lam = Union[float, Dict[str, float]]


def _clip01(x: float) -> float:
    """Clips a value to the normalized confidence range `[0, 1]`.

    Ensures that confidence values used during scoring remain within the
    expected range, preventing invalid or out-of-range values from affecting
    the decode objective.

    Args:
        x: Input value to clamp.

    Returns:
        The input value constrained to the interval `[0.0, 1.0]`.
    """
    return max(0.0, min(1.0, x))


def _lam_for_edge(lam: Lam, gi: int, gj: int) -> float:
    """Resolves the spacing-prior weight for a POI transition.

    Determines the lambda weighting applied to the spacing prior contribution
    for the edge from `POI_ORDER[gi]` to `POI_ORDER[gj]`. The weight may be
    provided as a single scalar applied uniformly to all transitions or as a
    mapping of adjacent POI pair names to individual weights.

    For non-adjacent transitions, which occur when one or more intermediate
    POIs are absent, the effective weight is computed as the mean of the
    lambda values for the intervening adjacent POI pairs. This mirrors the
    composition of gap statistics performed by the spacing prior.

    Args:
        lam (Lam): Spacing-prior weight configuration. May be a scalar value applied
            to all edges or a dictionary keyed by adjacent POI pair names
            (for example, `"POI1->POI2"`).
        gi (int): Global index of the starting POI in `POI_ORDER`.
        gj (int): Global index of the ending POI in `POI_ORDER`. Must be greater
            than `gi` for a non-adjacent transition.

    Returns:
        The resolved spacing-prior weight for the specified POI transition.
        Returns `0.0` if an empty lambda mapping is provided.
    """
    if not isinstance(lam, dict):
        return float(lam)
    if not lam:
        return 0.0
    default = sum(lam.values()) / len(lam)
    vals = [float(lam.get(f"{POI_ORDER[k]}->{POI_ORDER[k + 1]}", default)) for k in range(gi, gj)]
    return sum(vals) / len(vals) if vals else default


def _prep_candidates(
    candidates: Dict[str, List[Candidate]],
    present: List[str],
    min_conf: float,
    max_candidates: int,
) -> Dict[str, List[Candidate]]:
    """Filters and prepares candidate detections for dynamic programming.

    Cleans the raw candidate lists for each present POI by applying confidence
    filtering, removing duplicate timestamps, limiting the number of candidates,
    and sorting the remaining candidates chronologically. This preprocessing
    reduces the size of the candidate lattice while preserving the strongest
    plausible detections for joint decoding.

    Candidate lists are never allowed to become empty solely because of the
    confidence threshold. If all candidates for a POI are filtered out, the
    original candidate list is retained to avoid removing a POI from the
    decode problem.

    Duplicate detections at the same timestamp are merged by keeping the
    highest-confidence candidate. This handles cases where multiple detection
    stages produce the same bounding box.

    Args:
        candidates (Dict[str, List[Candidate]]): Mapping of POI names to lists of candidate detections.
        present (List[str]): Ordered list of POI names to include in decoding.
        min_conf (float): Minimum confidence threshold used to filter candidates.
        max_candidates (int): Maximum number of candidates retained per POI after
            confidence ranking.

    Returns:
        A dictionary mapping each present POI to its cleaned, capped, and
        time-sorted candidate list.
    """
    cand: Dict[str, List[Candidate]] = {}
    for p in present:
        cs = [c for c in candidates.get(p, []) if c.conf >= min_conf]
        if not cs:
            cs = list(candidates.get(p, []))  # don't drop a POI entirely
        # dedupe identical timestamps
        by_t: Dict[float, Candidate] = {}
        for c in cs:
            key = round(c.time, 6)
            if key not in by_t or c.conf > by_t[key].conf:
                by_t[key] = c
        cs = list(by_t.values())
        # cap to top-K by confidence then sort by time for the DP.
        cs.sort(key=lambda c: c.conf, reverse=True)
        cs = cs[: max(1, max_candidates)]
        cs.sort(key=lambda c: c.time)
        cand[p] = cs
    return cand


def _score_config(
    chosen: Dict[str, Candidate],
    placeable: List[str],
    prior: OnyxSpacingPrior,
    lam: Lam,
    conf_weight: float,
    use_frac: bool,
) -> Tuple[float, float, float]:
    """Computes the objective score for a decoded POI configuration.

    Recalculates the configuration score using the same scoring logic applied
    during decoding. The score combines weighted detection confidence with the
    weighted spacing-prior log-likelihood between consecutive present POIs.

    For partial configurations or configurations with missing intermediate POIs,
    spacing likelihoods are evaluated using the appropriate composed gap
    statistics from the spacing prior. When enabled, span-fraction likelihoods
    are calculated using the selected configuration span and scoped span
    semantics.

    Args:
        chosen (Dict[str, Candidate]): Mapping of POI names to their selected candidate detections.
        placeable (List[str]): Ordered list of POIs considered during decoding.
        prior (OnyxSpacingPrior): Learned POI spacing prior used to evaluate temporal consistency.
        lam (Lam): Spacing-prior weighting configuration. May be a scalar or a
            per-transition mapping.
        conf_weight (float): Weight applied to the summed detection confidence term.
        use_frac (bool): Whether span-fraction likelihoods should be included in
            spacing evaluation.

    Returns:
        A tuple containing:
            * Total objective score (confidence contribution plus weighted
              spacing likelihood).
            * Unweighted spacing log-likelihood.
            * Weighted confidence sum.
    """
    g_index = {POI_ORDER[i]: i for i in range(len(POI_ORDER))}
    placed = [p for p in placeable if p in chosen]
    conf_sum = conf_weight * sum(_clip01(chosen[p].conf) for p in placed)
    times = [chosen[p].time for p in placed]
    span = (max(times) - min(times)) if (use_frac and len(times) > 1) else 0.0
    span_lo = g_index[placed[0]] if placed else 0
    span_hi = g_index[placed[-1]] if placed else 0
    spacing_ll = 0.0
    weighted_spacing_ll = 0.0
    for a, b in zip(placed[:-1], placed[1:]):
        gi, gj = g_index[a], g_index[b]
        ll = prior.gap_loglik_scoped(
            gi, gj, chosen[b].time - chosen[a].time, span, span_lo, span_hi
        )
        spacing_ll += ll
        weighted_spacing_ll += _lam_for_edge(lam, gi, gj) * ll
    return conf_sum + weighted_spacing_ll, spacing_ll, conf_sum


def _dp_pass(
    cand: Dict[str, List[Candidate]],
    placeable: List[str],
    prior: OnyxSpacingPrior,
    lam: Lam,
    conf_weight: float,
    feas_slack: float,
    require_feasible: bool,
    span_for_frac: float,
) -> Optional[Dict[str, Candidate]]:
    """Runs one exact dynamic-programming pass over the candidate lattice.

    Evaluates all valid ordered candidate paths through the available POI
    candidates and returns the configuration with the maximum objective score.
    The optimization combines candidate confidence with weighted spacing-prior
    likelihood while enforcing strict temporal ordering and, optionally,
    learned gap feasibility constraints.

    The dynamic program operates on adjacent present POIs in the decode order.
    Missing global POIs are handled through the spacing prior's composed gap
    statistics. The span-fraction component of the prior can be disabled by
    passing `span_for_frac <= 0`, causing the pass to use seconds-based
    likelihoods only.

    Args:
        cand (Dict[str, List[Candidate]]): Mapping of POI names to prepared candidate lists.
        placeable (List[str]): Ordered list of POIs included in the decode path.
        prior (OnyxSpacingPrior): Learned spacing prior used for gap feasibility and likelihood
            evaluation.
        lam (Lam): Spacing-prior weight configuration. May be a scalar value or a
            mapping of POI pair names to individual weights.
        conf_weight (float): Weight applied to candidate confidence contributions.
        feas_slack (float): Multiplicative slack applied to learned gap feasibility
            bounds.
        require_feasible (bool): If `True`, discard paths containing gaps outside
            learned feasibility bounds.
        span_for_frac (float): Span duration used by the span-fraction likelihood
            calculation. Values less than or equal to zero disable the
            fraction component.

    Returns:
        A dictionary mapping POI names to their selected candidates for the
        highest-scoring valid path, or `None` if no admissible ordered path
        exists.

    Notes:
        The returned path is reconstructed using back pointers after the final
        DP stage. A missing back pointer during reconstruction is treated as a
        failed decode and returns `None` defensively.
    """
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
                score = (
                    dp[j - 1][kp] + _lam_for_edge(lam, gi, gj) * ll + conf_weight * _clip01(cb.conf)
                )
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
    prior: OnyxSpacingPrior,
    lam: Lam,
    conf_weight: float,
    feas_slack: float,
    require_feasible: bool,
) -> Optional[Dict[str, Candidate]]:
    """Performs exact decoding with a fixed span-fraction reference.

    The span-fraction component of the spacing prior depends only on the first
    and last selected POI candidates. Since those endpoints determine the
    configuration span, this method enumerates all possible first/last
    candidate combinations, fixes the span for each combination, and performs
    an exact dynamic-programming decode under that condition.

    Each resulting configuration is consistently rescored using the full
    blended objective (confidence plus weighted spacing likelihood), and the
    highest-scoring valid configuration is returned.

    This approach preserves the exact optimum of the blended objective while
    avoiding the loss of optimality that would occur if span were estimated
    incrementally during the dynamic-programming pass.

    Args:
        cand (Dict[str, List[Candidate]]): Mapping of POI names to prepared candidate lists.
        placeable (List[str]): Ordered list of POIs included in the decode path.
        prior (OnyxSpacingPrior): Learned spacing prior used for gap likelihood and feasibility
            evaluation.
        lam (Lam): Spacing-prior weight configuration. May be a scalar value or a
            mapping of POI pair names to individual weights.
        conf_weight (float): Weight applied to candidate confidence contributions.
        feas_slack (float): Multiplicative slack applied to learned gap feasibility
            bounds.
        require_feasible (bool): If `True`, only configurations satisfying hard gap
            feasibility constraints are considered.

    Returns:
        The highest-scoring decoded configuration across all valid endpoint
        span conditions, or `None` if no admissible configuration exists.
    """
    first, last = placeable[0], placeable[-1]
    best_chosen: Optional[Dict[str, Candidate]] = None
    best_score = -1e18
    for f in cand[first]:
        for last_cand in cand[last]:
            span = last_cand.time - f.time
            if span <= 0:
                continue
            sub = dict(cand)
            sub[first] = [f]
            sub[last] = [last_cand]
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
    prior: OnyxSpacingPrior,
    lam: Lam = 1.0,
    conf_weight: float = 1.0,
    feas_slack: float = 1.5,
    min_conf: float = 0.0,
    require_feasible: bool = True,
    max_candidates: int = 10,
) -> DecodeResult:
    """Decodes the optimal globally consistent POI configuration.

    Selects one candidate detection per present POI by optimizing the combined
    detection confidence and learned spacing-prior likelihood objective. The
    decoder replaces independent per-POI selection with a joint optimization
    that enforces temporal ordering and, when enabled, learned gap feasibility
    constraints.

    Candidate detections are first filtered, deduplicated, capped, and
    time-sorted to form the dynamic-programming lattice. The decoder then uses
    either a standard seconds-based dynamic program or a span-conditioned
    dynamic program when the span-fraction component of the spacing prior is
    active.

    If a fully feasible solution cannot be found and feasibility enforcement is
    enabled, the decoder retries with relaxed gap constraints. If no complete
    ordered solution exists, it falls back to the independent highest-
    confidence selection strategy to preserve production behavior.

    Spacing likelihoods are evaluated between consecutive present POIs using
    their positions in the global POI ordering. When present POIs are not
    globally adjacent, composed gap statistics from the spacing prior are used.
    Span-fraction likelihoods are enabled only when sufficient POIs are present
    to provide meaningful ratio information.

    Args:
        candidates (Dict[str, List[Candidate]]): Mapping of POI names to candidate detections. Candidate
            lists may be in any order. Only POIs listed in `present_pois` are
            decoded.
        present_pois (Sequence[str]): Ordered subset of POIs identified as present by upstream
            classification logic.
        prior (OnyxSpacingPrior): Learned spacing prior used for temporal likelihood scoring and
            feasibility checks.
        lam (float): Weight applied to spacing-prior likelihood contributions. May be a
            scalar value applied globally or a mapping of POI pair names to
            individual weights.
        conf_weight (float): Weight applied to candidate confidence contributions.
        feas_slack (float): Multiplicative slack applied to learned gap feasibility
            bounds.
        min_conf (float): Minimum candidate confidence threshold applied before
            decoding. If filtering removes all candidates for a POI, the
            original candidates are retained.
        require_feasible (bool): If `True`, only configurations satisfying learned
            gap feasibility constraints are accepted during the initial decode.
            A relaxed fallback decode is attempted if none exist.
        max_candidates (int): Maximum number of candidates retained per POI after
            confidence ranking.

    Returns:
        A `DecodeResult` containing the selected candidate configuration,
        objective score components, and flags describing whether the solution
        was fully feasible or required fallback handling.

    NOTE:
        The decoder only operates on POIs already determined to be present and
        never introduces additional detections. This allows partial fills to be
        handled naturally by decoding shorter POI sequences.
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

    # Decode exact under the full objective. With frac active the
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
        relaxed = (
            _span_conditioned_decode(
                cand, placeable, prior, lam, conf_weight, feas_slack, require_feasible=False
            )
            if use_frac
            else _dp_pass(cand, placeable, prior, lam, conf_weight, 1e9, False, span_for_frac=0.0)
        )
        if relaxed is None or len(relaxed) < len(placeable):
            # Even strict ordering has no complete path
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
    prior: OnyxSpacingPrior,
    lam: Lam,
    conf_weight: float,
) -> DecodeResult:
    """Creates a fallback decode using independent confidence selection.

    Selects the highest-confidence candidate independently for each placeable
    POI. This strategy is used as the production-safe fallback when no fully
    ordered or gap-feasible joint configuration can be found by the dynamic
    programming decoder.

    The fallback preserves existing production behavior by ensuring that any
    POI with available candidates is still assigned a detection rather than
    being removed from the result. The resulting configuration is rescored
    using the same objective components as normal decoding, although without
    the span-fraction spacing component.

    Args:
        cand: Mapping of POI names to prepared candidate lists.
        placeable: Ordered list of POIs that can be assigned candidates.
        prior: Learned spacing prior used to evaluate the resulting
            configuration.
        lam: Spacing-prior weight configuration. May be a scalar or a mapping
            of POI pair names to individual weights.
        conf_weight: Weight applied to candidate confidence contributions.

    Returns:
        A `DecodeResult` containing the greedily selected candidates, the
        resulting score components, and flags indicating that the result was
        not fully feasible and required fallback handling.
    """
    chosen = {p: max(cand[p], key=lambda c: c.conf) for p in placeable if cand[p]}
    total, sll, csum = _score_config(chosen, placeable, prior, lam, conf_weight, False)
    return DecodeResult(chosen, total, sll, csum, False, True)


def score_configuration(
    chosen: Dict[str, Candidate],
    prior: OnyxSpacingPrior,
    lam: Lam = 1.0,
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
    """Selects the highest-confidence candidate independently for each POI.

    Implements the legacy production selection behavior used before joint
    configuration decoding. Each present POI is evaluated independently, and
    the candidate with the highest detection confidence is selected without
    considering temporal ordering, spacing constraints, or interactions with
    other POIs.

    This function is provided as a baseline for A/B comparisons against the
    joint decoder.

    Args:
        candidates: Mapping of POI names to candidate detections.
        present_pois: Sequence of POI names to evaluate.

    Returns:
        A mapping of POI names to their highest-confidence candidate. POIs with
        no available candidates are omitted from the result.
    """
    out = {}
    for p in present_pois:
        cs = candidates.get(p, [])
        if cs:
            out[p] = max(cs, key=lambda c: c.conf)
    return out
