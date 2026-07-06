"""
v6_yolo_decode.py

Joint configuration decode over YOLO candidate detections.

This module replaces the greedy pick with a global decode: given several
CANDIDATE detections per present POI (time + confidence), choose one candidate
per POI such that the configuration is

    (a) strictly time-ordered (hard),
    (b) feasible under the learned per-gap bounds (hard, with slack),
    (c) maximal in   sum(detection confidence)  +  lambda * spacing log-lik,

where the spacing log-likelihood comes from the learned QModelV6YOLO_SpacingPrior. This is
the EDA-selected design: flat, pairwise prior + monotonic constraint, solved
exactly by dynamic programming over the candidate lattice.

Non-adjacent present POIs (e.g. POI2 and POI4 present with POI3 unplaceable)
are scored against the COMPOSITION of the intervening fitted gaps via
QModelV6YOLO_SpacingPrior.composed_stat, not against the first gap's stats.

It only decodes the POIs that are actually present (the fill-count gate /
type_cls has already decided how many channels exist), so partial fills are
handled by passing fewer POIs — no spurious late POIs are introduced.

Inputs / outputs are plain dicts so this drops in alongside the existing
predictor without depending on its internals.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-06-11

Version:
    6.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from QATCH.QModel.models.v6_yolo.v6_yolo_spacing_prior import (
        QModelV6YOLO_SpacingPrior,
        POI_ORDER,
    )
except (ImportError, ModuleNotFoundError):
    from QATCH.QModel.models.v6_yolo.v6_yolo_spacing_prior import (
        QModelV6YOLO_SpacingPrior,
        POI_ORDER,
    )


@dataclass
class Candidate:
    """Represents a detected event candidate in time.

    A lightweight structure used to store a single detection output,
    typically produced by a model such as YOLO. Each candidate encodes
    the detected timestamp and its associated confidence score.

    Attributes:
        time: Detection timestamp in seconds. Represents the absolute
            position of the candidate in the signal or sequence.
        conf: Model confidence score in the range [0, 1], where higher
            values indicate greater likelihood that the detection is
            valid.
    """

    time: float  # detected timestamp (seconds)
    conf: float  # YOLO confidence in [0,1]


@dataclass
class DecodeResult:
    """Result of a decoded POI configuration.

    Encapsulates the output of a decoding procedure that selects one
    candidate per POI and evaluates both model confidence and spacing
    consistency.

    Attributes:
        chosen: Mapping from POI name to the selected :class:`Candidate`
            representing the final decoded configuration.
        total_score: Combined objective score used for decoding, typically
            aggregating spacing likelihood, confidence, and any additional
            penalties or heuristics.
        spacing_loglik: Total spacing log-likelihood of the chosen
            configuration under the spacing prior.
        conf_sum: Sum of confidence scores of all selected candidates.
        feasible: Whether a fully feasible configuration was found under
            hard spacing constraints. If False, no valid global solution
            satisfied all constraints.
        fallback_used: Indicates whether the decoder had to relax
            constraints (e.g., greedy decoding or partial relaxation) to
            produce a result.
    """

    chosen: Dict[str, Candidate]  # poi_name -> chosen candidate
    total_score: float
    spacing_loglik: float
    conf_sum: float
    feasible: bool  # False if no fully-feasible path existed
    fallback_used: bool  # True if we relaxed to greedy/partial


def _clip01(x: float) -> float:
    """Clamp a scalar value to the unit interval [0, 1].

    Ensures that an input value is restricted to the valid probability
    range. Values below 0.0 are mapped to 0.0, and values above 1.0 are
    mapped to 1.0.

    Args:
        x: Input scalar value.

    Returns:
        The value clipped to the range [0.0, 1.0].

    Notes:
        This is a utility function commonly used to enforce valid bounds
        for probabilities, confidence scores, or normalized weights.
    """
    return max(0.0, min(1.0, x))


def _prep_candidates(
    candidates: Dict[str, List[Candidate]],
    present: List[str],
    min_conf: float,
    max_candidates: int,
) -> Dict[str, List[Candidate]]:
    """Preprocess candidate detections for dynamic programming decoding.

    Filters, deduplicates, and orders candidate detections for each
    requested POI before they are used in downstream decoding.

    The preprocessing pipeline performs the following steps per POI:

    1. Confidence filtering using ``min_conf``.
    2. Fallback retention of all candidates if filtering removes all
       detections (ensures no POI is dropped entirely).
    3. Deduplication of near-identical timestamps (rounded to 1e-6
       precision), keeping the highest-confidence instance.
    4. Confidence-based pruning to keep only the top-K candidates.
    5. Time-based sorting to support sequential decoding.

    Args:
        candidates: Mapping from POI name to a list of raw detection
            :class:`Candidate` objects.
        present: Ordered list of POIs to be included in decoding.
        min_conf: Minimum confidence threshold for retaining detections.
        max_candidates: Maximum number of candidates to keep per POI after
            pruning.

    Returns:
        A dictionary mapping each POI in ``present`` to a cleaned and
        time-sorted list of :class:`Candidate` objects ready for decoding.

    Notes:
        Deduplication is performed using a coarse time key
        (``round(c.time, 6)``) to merge near-identical detections
        originating from different model stages (e.g., coarse + fine
        inference passes).
    """
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
    prior: QModelV6YOLO_SpacingPrior,
    lam: float,
    conf_weight: float,
    use_frac: bool,
) -> Tuple[float, float, float]:
    """Score a decoded POI configuration using confidence and spacing priors.

    Computes a joint objective for a candidate configuration by combining:

    * A confidence term derived from selected detections.
    * A spacing likelihood term from the learned prior.

    The function returns all intermediate components to support debugging
    and alternative decoding strategies.

    Args:
        chosen: Mapping from POI name to selected :class:`Candidate`.
        placeable: Ordered list of POIs that are eligible for scoring.
        prior: Learned spacing prior used to compute log-likelihoods.
        lam: Weight applied to the spacing log-likelihood term.
        conf_weight: Weight applied to summed detection confidences.
        use_frac: If True, enables span-aware (fraction-based) likelihood
            normalization in the spacing model.

    Returns:
        A tuple of:

        * total score (confidence + λ * spacing likelihood)
        * spacing log-likelihood component
        * confidence sum component

    Notes:
        Only POIs present in both ``placeable`` and ``chosen`` are scored.
        Confidence values are clipped to [0, 1] via :func:`_clip01`.

        The spacing term is accumulated over consecutive POI pairs using
        :meth:`QModelV6YOLO_SpacingPrior.gap_loglik_scoped`, ensuring that partial
        configurations are scored consistently under the same model family.
    """
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
    prior: QModelV6YOLO_SpacingPrior,
    lam: float,
    conf_weight: float,
    feas_slack: float,
    require_feasible: bool,
    span_for_frac: float,
) -> Optional[Dict[str, Candidate]]:
    """Run dynamic programming over the candidate lattice to select a valid configuration.

    Performs an exact Viterbi-style search over ordered POIs to find the
    highest-scoring assignment of one candidate per POI. The score combines
    detection confidence and spacing consistency under the learned prior.

    The DP enforces temporal ordering and optionally enforces hard feasibility
    constraints on inter-POI gaps. It also evaluates spacing likelihood using
    a scoped prior that adapts to partial or full-span configurations.

    Args:
        cand: Mapping from POI name to a list of preprocessed
            :class:`Candidate` objects (already filtered and sorted).
        placeable: Ordered list of POIs to be included in decoding.
        prior: Learned spacing prior
            (:class:`QModelV6YOLO_SpacingPrior`) used for likelihood and
            feasibility evaluation.
        lam: Weight applied to spacing log-likelihood in the total score.
        conf_weight: Weight applied to candidate confidence scores.
        feas_slack: Multiplicative slack factor applied to feasibility
            bounds, making hard constraints more permissive.
        require_feasible: If True, enforces hard feasibility checks via
            :meth:`SpacingPrior.gap_feasible_between`.
        span_for_frac: Span used for fraction-based normalization in the
            spacing model. If <= 0, fraction-based terms are disabled.

    Returns:
        A dictionary mapping POI names to selected :class:`Candidate`
        objects representing the best-scoring configuration, or ``None``
        if no valid monotonic path exists.

    Notes:
        This is a Viterbi-style DP over a fully connected bipartite layer
        graph between consecutive POIs in ``placeable``.

        Transitions enforce:
            * Strict temporal ordering (gap > 0)
            * Optional hard feasibility constraints
            * Soft spacing likelihood via scoped prior
            * Per-node confidence reward

        The algorithm backtracks from the best-scoring terminal state to
        reconstruct the optimal configuration.

    Warnings:
        The DP assumes candidate lists are pre-sorted by time within each
        POI. Violating this assumption may degrade correctness or performance.
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
    prior: QModelV6YOLO_SpacingPrior,
    lam: float,
    conf_weight: float,
    feas_slack: float,
    require_feasible: bool,
) -> Optional[Dict[str, Candidate]]:
    """Perform exact decoding with span-conditioned fraction normalization.

    This decoder resolves a key coupling in the spacing model: the
    span-fraction likelihood depends on the total span between the first
    and last selected POIs. Because this span is not known a priori, the
    algorithm explicitly enumerates candidate endpoint pairs and conditions
    the entire decoding process on each resulting span.

    For each feasible choice of first and last candidate:

    1. The configuration span is fixed as:
       ``span = t(last) - t(first)``
    2. Endpoint candidates are pinned (single-choice constraints).
    3. A constrained DP (:meth:`_dp_pass`) is executed under that span.
    4. The resulting configuration is re-scored under a consistent
       span-aware objective (:meth:`_score_config`).
    5. The best-scoring configuration across all endpoint pairs is returned.

    Args:
        cand: Mapping from POI name to list of preprocessed
            :class:`Candidate` objects.
        placeable: Ordered list of POIs to decode.
        prior: Learned spacing prior used for scoring and constraints.
        lam: Weight applied to spacing log-likelihood.
        conf_weight: Weight applied to candidate confidence scores.
        feas_slack: Slack factor applied to hard feasibility bounds.
        require_feasible: Whether to enforce hard feasibility constraints
            during DP.

    Returns:
        The highest-scoring configuration satisfying a consistent
        span-conditioned objective, or ``None`` if no valid configuration
        exists for any endpoint pairing.

    Notes:
        This procedure resolves a structural issue in span-normalized
        models: the fraction likelihood is not independent of endpoint
        selection. By conditioning on endpoints explicitly, the decoder
        ensures consistency between:

        * span definition
        * DP transitions
        * final scoring pass

        This makes the objective globally coherent rather than
        implicitly self-referential.
    """
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
    prior: QModelV6YOLO_SpacingPrior,
    lam: float = 1.0,
    conf_weight: float = 1.0,
    feas_slack: float = 1.5,
    min_conf: float = 0.0,
    require_feasible: bool = True,
    max_candidates: int = 10,
) -> DecodeResult:
    """Decode the optimal ordered POI configuration under the full spacing model.

    This is the top-level inference routine that selects exactly one candidate
    per active POI by optimizing a joint objective combining:

    * Detection confidence (YOLO scores)
    * Learned spacing prior (log-likelihood over POI gaps)
    * Optional hard feasibility constraints on gap bounds

    The decoder operates over a restricted subset of POIs (`present_pois`)
    and produces a globally consistent ordered configuration aligned with
    ``POI_ORDER``.

    Two decoding regimes are supported:

    1. **Seconds-only DP**:
       Used when span-fraction modeling is disabled or insufficient (fewer
       than 3 POIs). This yields a standard Viterbi-style solution over
       independent gap likelihoods.

    2. **Span-conditioned DP**:
       Used when span-fraction modeling is active. Because the fraction
       likelihood depends on the unknown global span, decoding becomes
       conditional on endpoint selection. The algorithm therefore enumerates
       endpoint spans and invokes :meth:`_span_conditioned_decode`.

    If no fully feasible configuration is found under strict constraints,
    the decoder falls back to a relaxed feasibility pass or a greedy
    construction as a last resort.

    Args:
        candidates: Mapping from POI name to raw candidate detections.
        present_pois: Ordered subset of POIs considered present in the
            current inference instance.
        prior: Learned spacing prior
            (:class:`QModelV6YOLO_SpacingPrior`) used for scoring.
        lam: Weight applied to spacing log-likelihood.
        conf_weight: Weight applied to summed detection confidence scores.
        feas_slack: Multiplicative relaxation factor applied to hard gap
            feasibility bounds.
        min_conf: Minimum confidence threshold used during preprocessing.
        require_feasible: If True, enforces hard feasibility constraints
            during decoding; otherwise only soft scoring is used.
        max_candidates: Maximum number of candidates retained per POI.

    Returns:
        A :class:`DecodeResult` containing:

        * selected candidates per POI
        * total objective score
        * spacing log-likelihood component
        * confidence sum
        * feasibility status
        * fallback usage flag

    Notes:
        This function is the main entry point for inference. It orchestrates:

        * candidate preprocessing (:meth:`_prep_candidates`)
        * DP-based optimization (:meth:`_dp_pass`)
        * span-conditioned decoding (:meth:`_span_conditioned_decode`)
        * fallback strategies (relaxed DP or greedy decoding)

        The model enforces strict temporal ordering and optionally hard
        physical plausibility constraints while still allowing probabilistic
        ranking via the spacing prior.

    Warnings:
        This decoder assumes ``POI_ORDER`` defines a strict global ordering
        of all possible POIs. All indexing and span logic depends on this
        invariant.
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

    # Exact under the full objective. With frac active the
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
        if use_frac:
            relaxed = _span_conditioned_decode(
                cand,
                placeable,
                prior,
                lam,
                conf_weight,
                feas_slack,
                False,
            )
        else:
            relaxed = _dp_pass(
                cand,
                placeable,
                prior,
                lam,
                conf_weight,
                1e9,
                False,
                span_for_frac=0.0,
            )
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
    prior: QModelV6YOLO_SpacingPrior,
    lam: float,
    conf_weight: float,
) -> DecodeResult:
    """Construct a fallback greedy decoding result using per-POI confidence maximization.

    Produces a valid (though potentially suboptimal) configuration by selecting,
    independently for each POI, the candidate with the highest confidence score.
    This is used as a safety fallback when no globally consistent solution can
    be found under the DP formulation or feasibility constraints.

    Unlike the DP-based decoders, this method does not enforce temporal
    ordering constraints beyond the implicit ordering of ``placeable``, and it
    does not guarantee global optimality under the spacing prior. However, it
    guarantees that every POI with available candidates contributes exactly one
    selected detection, ensuring a complete output whenever possible.

    Args:
        cand: Mapping from POI name to list of :class:`Candidate` objects.
        placeable: Ordered list of POIs eligible for decoding.
        prior: Learned spacing prior. Included for compatibility with scoring,
            but not actively optimized in the greedy selection step.
        lam: Spacing weight (used only during scoring, not selection).
        conf_weight: Weight applied to confidence scores in final evaluation.

    Returns:
        A :class:`DecodeResult` containing the greedy selection, its scored
        objective components, and flags indicating that this result was
        produced via fallback rather than full optimization.

    Notes:
        This method is intentionally simple:

        * No DP or joint optimization is performed
        * Each POI is treated independently
        * Only confidence is used for selection

        It is designed as a robustness floor: the decoder should always
        return at least this quality of result, even in degenerate cases
        where global constraints cannot be satisfied.
    """
    chosen = {p: max(cand[p], key=lambda c: c.conf) for p in placeable if cand[p]}
    total, sll, csum = _score_config(chosen, placeable, prior, lam, conf_weight, False)
    return DecodeResult(chosen, total, sll, csum, False, True)


def score_configuration(
    chosen: Dict[str, Candidate],
    prior: QModelV6YOLO_SpacingPrior,
    lam: float = 1.0,
    conf_weight: float = 1.0,
) -> float:
    """Compute a comparable objective score for an externally provided configuration.

    Evaluates an arbitrary POI configuration under the same scoring function
    used by :func:`dp_decode`, ensuring consistency between production-selected
    configurations and DP-optimized solutions.

    This is primarily used for:
        * comparing cascade outputs against decoded solutions
        * hysteresis / accept-margin decision logic in the controller
        * external evaluation of model-selected configurations

    The scoring function mirrors the internal DP objective:

        total = confidence_term + λ * spacing_loglikelihood

    where the spacing term uses the same rules as decoding, including
    span-conditioned fraction logic when applicable.

    Args:
        chosen: Mapping from POI name to selected :class:`Candidate`.
        prior: Learned spacing prior used to compute spacing likelihoods.
        lam: Weight applied to spacing log-likelihood.
        conf_weight: Weight applied to summed confidence scores.

    Returns:
        A scalar objective score compatible with :attr:`DecodeResult.total_score`.
        Higher values indicate better agreement with both confidence and
        spacing priors.

    Notes:
        The function internally reconstructs the ordered subset of POIs
        present in the configuration and applies the same fraction-mode
        condition used in decoding:

            use_frac = len(placeable) >= 3 and prior.frac_blend > 0

        This ensures that externally evaluated configurations remain
        directly comparable to DP outputs.
    """
    placeable = [p for p in POI_ORDER if p in chosen]
    if not placeable:
        return -1e18
    use_frac = len(placeable) >= 3 and prior.frac_blend > 0
    return _score_config(chosen, placeable, prior, lam, conf_weight, use_frac)[0]


def greedy_baseline(
    candidates: Dict[str, List[Candidate]], present_pois: Sequence[str]
) -> Dict[str, Candidate]:
    """Construct the production baseline configuration via independent selection.

    Implements the current production inference strategy, which selects the
    highest-confidence candidate independently for each POI without modeling
    temporal dependencies or joint spacing constraints.

    This baseline serves as a reference point for evaluating the benefit of
    the full joint decoding system (:func:`dp_decode`), and is used in
    A/B comparisons and regression testing.

    Args:
        candidates: Mapping from POI name to list of :class:`Candidate`
            detections.
        present_pois: Ordered sequence of POIs considered present in the
            current inference instance.

    Returns:
        A dictionary mapping each POI to its highest-confidence
        :class:`Candidate`. POIs with no available candidates are omitted.

    Notes:
        This method is intentionally non-structured:

        * No temporal ordering constraints are enforced
        * No spacing or feasibility priors are used
        * Each POI is treated independently

        As a result, it may produce temporally inconsistent configurations,
        but it reflects the exact behavior of the current production system.
    """
    out = {}
    for p in present_pois:
        cs = candidates.get(p, [])
        if cs:
            out[p] = max(cs, key=lambda c: c.conf)
    return out
