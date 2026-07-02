"""
v6_qmodel_yolo_spacing_prior.py

A flat, pairwise configuration prior over POI positions, learned from
complete-fill ground-truth configurations. This is the model the EDA selected:
flat (not run-conditional) and pairwise (not autoregressive), because on the
real data those additions hurt rather than helped.

At decode time the prior scores a candidate configuration by the summed
log-likelihood of its gaps, plus hard feasibility (monotonic order, learned
min/max gap bounds). The score combines with YOLO detection confidence in the
DP decoder (see dp_decode.py).

It is deliberately simple and interpretable: each gap is modelled as a
log-normal (gaps are positive and right-skewed), which the DP turns into an
additive quadratic-in-log penalty. Nothing here needs the raw signal.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-06-11

Version:
    6.2.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# POI order
POI_ORDER = ["POI1", "POI2", "POI3", "POI4", "POI5"]


@dataclass
class GapStat:
    """Statistical model of a consecutive gap distribution.

    Stores parameters for a log-normal representation of gap durations and
    normalized gap fractions, along with feasibility bounds derived from
    robust percentile estimates.

    Attributes:
        log_mu_sec: Mean of the natural logarithm of gap durations in seconds.
        log_sd_sec: Standard deviation of the natural logarithm of gap
            durations in seconds.
        log_mu_frac: Mean of the natural logarithm of gap durations expressed
            as a fraction of the total span.
        log_sd_frac: Standard deviation of the natural logarithm of gap
            fractions.
        min_gap_sec: Lower feasibility bound for gap duration in seconds,
            typically derived from a lower robust percentile.
        max_gap_sec: Upper feasibility bound for gap duration in seconds,
            typically derived from an upper robust percentile.
        n: Number of observations used to estimate the statistics.
    """

    # Log-gap duration statistics (seconds)
    log_mu_sec: float
    log_sd_sec: float

    # Log-gap duration statistics (fraction of total span)
    log_mu_frac: float
    log_sd_frac: float

    # Feasibility bounds (seconds)
    min_gap_sec: float
    max_gap_sec: float

    # Sample count
    n: int


@dataclass
class QModelV6YOLO_SpacingPrior:
    """Statistical prior over consecutive POI spacing relationships.

    This model captures the expected temporal spacing between adjacent points
    of interest (POIs) using log-normal distributions. Each consecutive POI
    pair is modeled in both absolute time (seconds) and relative time
    (fraction of the total configuration span), allowing likelihood
    calculations that are robust to variations in overall process duration.

    Attributes:
        pairs: Ordered list of consecutive POI transition identifiers,
            such as `["POI1->POI2", "POI2->POI3"]`.
        gap: Mapping from POI transition identifiers to their fitted
            :class:`GapStat` distributions.
        frac_blend: Weight used when combining seconds-based and
            span-fraction-based log-likelihoods. A value of `0.0` uses
            only absolute gap durations, while `1.0` uses only relative
            gap fractions.
        bound_lo_pct: Lower percentile used to estimate minimum feasible
            gap durations during fitting.
        bound_hi_pct: Upper percentile used to estimate maximum feasible
            gap durations during fitting.
    """

    pairs: List[str]  # e.g. ["POI1->POI2", ...]
    gap: Dict[str, GapStat] = field(default_factory=dict)

    # Blend weight between seconds-based and fraction-based likelihoods.
    # 0 = pure seconds, 1 = pure span-fraction.
    frac_blend: float = 0.5

    # Percentiles used to estimate feasibility bounds during fitting.
    bound_lo_pct: float = 0.5
    bound_hi_pct: float = 99.5

    @staticmethod
    def fit(
        configs_sec: np.ndarray,
        frac_blend: float = 0.5,
        bound_lo_pct: float = 0.5,
        bound_hi_pct: float = 99.5,
    ) -> "QModelV6YOLO_SpacingPrior":
        """Fit a spacing prior from complete POI configurations.

        Consecutive POI gaps are extracted from each configuration and
        modeled using log-normal statistics in both absolute time and
        normalized span-fraction space. Feasibility bounds are estimated
        from robust gap percentiles.

        Args:
            configs_sec: Array of shape `(N, P)` containing complete POI
                configurations in seconds. Each row must be strictly
                increasing and contain all POIs in `POI_ORDER`.
            frac_blend: Weight used to combine fraction-based and
                seconds-based likelihoods. `0.0` uses only seconds,
                `1.0` uses only normalized fractions.
            bound_lo_pct: Lower percentile used when estimating minimum
                feasible gap durations.
            bound_hi_pct: Upper percentile used when estimating maximum
                feasible gap durations.

        Returns:
            A fitted `QModelV6YOLO_SpacingPrior` instance containing
            spacing distributions for each consecutive POI pair.

        Raises:
            AssertionError: If the number of POIs in `configs_sec` does
                not match `len(POI_ORDER)`.
        """
        N, P = configs_sec.shape
        assert P == len(POI_ORDER), f"expected {len(POI_ORDER)} POIs, got {P}"
        span = configs_sec[:, -1] - configs_sec[:, 0]
        span = np.where(span < 1e-9, np.nan, span)
        pairs = [f"{POI_ORDER[i]}->{POI_ORDER[i+1]}" for i in range(P - 1)]
        prior = QModelV6YOLO_SpacingPrior(
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

    def composed_stat(self, i: int, j: int) -> GapStat:
        """Compute statistics for a composed gap spanning multiple POIs.

        Constructs an approximate :class:`GapStat` for the interval
        `POI_ORDER[i] -> POI_ORDER[j]` by composing the fitted statistics of
        the consecutive gaps between them. This allows spacing likelihoods to be
        evaluated when one or more intermediate POIs are absent.

        The composition uses a standard log-normal sum approximation:

        * Gap medians are converted to linear space and summed.
        * Log-space variances are assumed independent and summed.
        * Feasibility bounds are summed across component gaps.
        * Sample count is taken as the minimum count among component gaps.

        For directly adjacent POIs (`j == i + 1`), the originally fitted
        consecutive-gap statistic is returned unchanged.

        Results are cached to avoid repeatedly recomputing composed
        distributions for the same POI pair.

        Args:
            i: Index of the starting POI in `POI_ORDER`.
            j: Index of the ending POI in `POI_ORDER`. Must satisfy
                `j > i`.

        Returns:
            A `GapStat` representing the approximate distribution of the
            composed gap spanning all consecutive intervals between `i` and
            `j`.

        Raises:
            ValueError: If `j <= i`.
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

    def _stat_loglik(self, gs: GapStat, gap_sec: float, span_sec: float) -> float:
        """Compute the blended spacing log-likelihood for an observed gap.

        Evaluates how well an observed gap matches a fitted gap distribution
        using both absolute duration and normalized span-fraction statistics.
        The resulting score is a weighted combination of two log-normal
        log-likelihood terms:

        * A seconds-based likelihood using the observed gap duration.
        * A fraction-based likelihood using the gap as a fraction of the
        total span.

        The blend weight is controlled by `self.frac_blend`. A value of
        `0.0` uses only absolute timing information, while `1.0` uses
        only relative spacing information.

        The returned value is a relative log-likelihood score intended for
        ranking or comparing candidate configurations. Constant terms from
        the log-normal density are omitted because only relative scores are
        required.

        Args:
            gs: Fitted gap statistics describing the expected distribution of
                the gap.
            gap_sec: Observed gap duration in seconds.
            span_sec: Total span duration in seconds used to compute the
                normalized gap fraction.

        Returns:
            A blended log-likelihood score. Larger values indicate better
            agreement with the fitted spacing distribution.

        Notes:
            Invalid or non-positive gaps receive a large negative penalty.
            If `span_sec` is not positive, the seconds-based likelihood is
            used for both components of the blend.
        """
        if not np.isfinite(gap_sec) or gap_sec <= 0:
            return -1e9
        z_sec = (np.log(gap_sec) - gs.log_mu_sec) / gs.log_sd_sec
        ll_sec = -0.5 * z_sec * z_sec - np.log(gs.log_sd_sec)
        # fraction log-normal
        if span_sec and span_sec > 0 and np.isfinite(span_sec):
            frac = gap_sec / span_sec
            if not np.isfinite(frac) or frac <= 0:
                ll_frac = -1e9
            else:
                z_f = (np.log(frac) - gs.log_mu_frac) / gs.log_sd_frac
                ll_frac = -0.5 * z_f * z_f - np.log(gs.log_sd_frac)
        else:
            ll_frac = ll_sec
        return float((1 - self.frac_blend) * ll_sec + self.frac_blend * ll_frac)

    def gap_loglik(self, pair_idx: int, gap_sec: float, span_sec: float) -> float:
        """Compute the plausibility of an observed consecutive POI gap.

        Evaluates an observed gap against the fitted spacing distribution for
        a specific consecutive POI pair. The score is computed as a blended
        log-likelihood using both:

        * The absolute gap duration in seconds.
        * The gap duration normalized by the total configuration span.

        The relative contribution of each component is controlled by
        `self.frac_blend`.

        Args:
            pair_idx: Index of the consecutive POI pair within `self.pairs`.
            gap_sec: Observed gap duration in seconds.
            span_sec: Total configuration span in seconds. If non-positive,
                the fraction-based likelihood is disabled and only the
                seconds-based component is used. This is useful for partial
                configurations where the span semantics of the fitted model
                do not apply.

        Returns:
            A relative log-likelihood score, where larger values indicate
            that the observed gap is more consistent with the fitted spacing
            distribution.

        Notes:
            This method is intended for directly adjacent POI pairs. For
            gaps spanning one or more missing intermediate POIs, use
            `composed_stat()` to obtain an appropriate composed
            distribution before evaluating the likelihood.
        """
        return self._stat_loglik(self.gap[self.pairs[pair_idx]], gap_sec, span_sec)

    def gap_loglik_between(self, i: int, j: int, gap_sec: float, span_sec: float) -> float:
        """Compute the plausibility of a gap between two global POI indices.

        Evaluates an observed gap against the expected spacing distribution
        between `POI_ORDER[i]` and `POI_ORDER[j]`.

        For adjacent POIs (`j == i + 1`), the likelihood is computed using
        the directly fitted gap statistics. For non-adjacent POIs, a composed
        spacing distribution is constructed from the intervening consecutive
        gaps and used instead. This allows meaningful scoring when one or more
        intermediate POIs are absent.

        The score is a blended log-likelihood combining absolute gap duration
        and normalized span-fraction information according to
        `self.frac_blend`.

        Args:
            i: Index of the starting POI in `POI_ORDER`.
            j: Index of the ending POI in `POI_ORDER`. Must satisfy
                `j > i`.
            gap_sec: Observed gap duration in seconds between the two POIs.
            span_sec: Total configuration span in seconds. If non-positive,
                the fraction-based likelihood component is disabled and only
                the seconds-based component is used.

        Returns:
            A relative log-likelihood score, where larger values indicate
            that the observed gap is more consistent with the expected spacing
            between the specified POIs.

        Raises:
            ValueError: If `j <= i`.
        """
        return self._stat_loglik(self.composed_stat(i, j), gap_sec, span_sec)

    def gap_loglik_scoped(
        self, i: int, j: int, gap_sec: float, span_sec: float, span_lo: int, span_hi: int
    ) -> float:
        """Compute a scoped spacing likelihood for a POI gap.

        Evaluates the plausibility of the gap between `POI_ORDER[i]` and
        `POI_ORDER[j]` while accounting for the fact that the observed span
        may cover only a subset of the full POI sequence.

        The fraction-based component of the spacing prior is normally defined
        relative to the span of a complete configuration:

        `t(POI_last) - t(POI_first)`

        During decoding, however, only a prefix or partial configuration may
        be available. In that case, the observed span covers fewer gaps than
        the complete-fill span used during fitting, causing the fitted
        fraction statistics to systematically underestimate expected gap
        fractions.

        To compensate, this method re-scopes the fraction model to the
        decode-time span defined by `span_lo` and `span_hi`. The expected
        fraction location is recomputed from the fitted seconds medians while
        preserving the fitted fraction dispersion.

        where the denominator is the expected span between the placed POIs
        `span_lo` and `span_hi`.

        This allows the prior to enforce proportional spacing relationships
        even when only part of the POI sequence is present. Early observed
        gaps effectively anchor the run's temporal scale, and candidate
        configurations are scored according to how well their remaining gaps
        match that scale.

        When the scoped span corresponds to the complete POI chain, the
        original fitted fraction statistics are used unchanged.

        Args:
            i: Index of the starting POI in `POI_ORDER`.
            j: Index of the ending POI in `POI_ORDER`. Must satisfy
                `j > i`.
            gap_sec: Observed gap duration in seconds.
            span_sec: Observed span duration in seconds between the currently
                placed boundary POIs.
            span_lo: Index of the first placed POI defining the scoped span.
            span_hi: Index of the last placed POI defining the scoped span.
                The span is interpreted as covering the gaps
                `[span_lo, span_hi)`.

        Returns:
            A relative log-likelihood score, where larger values indicate
            better agreement with the expected spacing pattern under the
            current decode-time span.

        Notes:
            If the scoped span corresponds to the full POI chain or
            `span_sec <= 0`, this method falls back to
            :meth:`_stat_loglik` using the original fitted statistics.

            For non-adjacent POIs, the likelihood is evaluated using a
            composed gap distribution obtained from :meth:`composed_stat`.
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
        """Check whether an observed gap satisfies learned feasibility bounds.

        Applies a hard validity check using the robust minimum and maximum gap
        durations learned during fitting. The bounds are expanded by a
        multiplicative slack factor to avoid rejecting gaps that are close to
        the empirical limits but still plausible.

        This method is intended as a coarse feasibility filter rather than a
        probabilistic score. Unlike :meth:`gap_loglik`, it produces a binary
        decision indicating whether the gap falls within an acceptable range.

        Args:
            pair_idx: Index of the consecutive POI pair within `self.pairs`.
            gap_sec: Observed gap duration in seconds.
            slack: Multiplicative tolerance applied to the learned bounds.
                The effective feasibility interval becomes::

                    [min_gap_sec / slack, max_gap_sec * slack]

                Values greater than `1.0` make the check more permissive.

        Returns:
            `True` if the gap is positive and falls within the slack-adjusted
            feasibility bounds; otherwise `False`.

        Notes:
            The underlying bounds are derived from robust percentiles of the
            training data rather than the absolute minimum and maximum observed
            gaps. This helps reduce sensitivity to outliers while still
            preventing physically implausible configurations.
        """
        gs = self.gap[self.pairs[pair_idx]]
        return (
            gap_sec > 0 and gap_sec >= gs.min_gap_sec / slack and gap_sec <= gs.max_gap_sec * slack
        )

    def gap_feasible_between(self, i: int, j: int, gap_sec: float, slack: float = 1.5) -> bool:
        """Check whether a gap between two global POIs is feasible.

        Applies a hard feasibility test to the observed gap between
        `POI_ORDER[i]` and `POI_ORDER[j]`.

        For adjacent POIs (`j == i + 1`), the learned feasibility bounds
        for the corresponding consecutive gap are used directly. For
        non-adjacent POIs, feasibility bounds are composed across the
        intervening gaps using :meth:`composed_stat`, allowing the check to
        remain valid when one or more intermediate POIs are absent.

        The resulting bounds are expanded by a multiplicative slack factor to
        reduce sensitivity to borderline cases.

        Args:
            i: Index of the starting POI in `POI_ORDER`.
            j: Index of the ending POI in `POI_ORDER`. Must satisfy
                `j > i`.
            gap_sec: Observed gap duration in seconds between the two POIs.
            slack: Multiplicative tolerance applied to the learned bounds.
                The effective feasibility interval becomes::

                    [min_gap_sec / slack, max_gap_sec * slack]

                Values greater than `1.0` make the feasibility check more
                permissive.

        Returns:
            `True` if the gap is positive and falls within the
            slack-adjusted feasibility bounds; otherwise `False`.

        Raises:
            ValueError: If `j <= i`.

        Notes:
            This method performs a hard feasibility check and does not
            measure relative plausibility. For probabilistic scoring of
            adjacent or non-adjacent POI gaps, use
            :meth:`gap_loglik_between`.
        """
        gs = self.composed_stat(i, j)
        return (
            gap_sec > 0 and gap_sec >= gs.min_gap_sec / slack and gap_sec <= gs.max_gap_sec * slack
        )

    def config_loglik(self, times_sec: List[float]) -> float:
        """Compute the total spacing likelihood of a complete POI configuration.

        Evaluates a fully specified, ordered POI configuration by summing the
        spacing log-likelihoods of all consecutive gaps. Each gap is scored
        against its corresponding fitted spacing distribution using both
        absolute timing and relative span-fraction information.

        The resulting score represents the overall agreement between the
        observed configuration and the learned spacing prior. Larger values
        indicate that the configuration exhibits more plausible temporal
        spacing patterns.

        Args:
            times_sec: Ordered POI times in seconds. The list is expected to
                contain a complete configuration matching the POI sequence
                used when fitting the spacing prior.

        Returns:
            The summed spacing log-likelihood across all consecutive POI
            gaps.

        Notes:
            The score is a relative likelihood measure rather than a
            normalized probability. It is most useful for comparing
            candidate configurations, ranking alternatives, or combining
            with other model components in a larger scoring framework.

            The span used by each gap likelihood is defined as::

                times_sec[-1] - times_sec[0]

            which corresponds to the complete configuration span assumed
            during fitting.
        """
        span = times_sec[-1] - times_sec[0]
        total = 0.0
        for i in range(len(times_sec) - 1):
            total += self.gap_loglik(i, times_sec[i + 1] - times_sec[i], span)
        return total

    def save(self, path: Path) -> None:
        """Serialize the spacing prior to disk as JSON.

        Stores all learned parameters of the spacing prior, including POI pair
        structure, blending configuration, percentile bounds, and all fitted
        :class:`GapStat` objects. The resulting file can later be reloaded to
        reconstruct the exact same model state.

        Args:
            path: Destination file path where the JSON representation of the
                model will be written.

        Notes:
            The serialization uses :func:`dataclasses.asdict` to convert each
            :class:`GapStat` into a JSON-compatible dictionary. No lossy
            transformations are applied beyond standard JSON encoding.

            The saved structure includes:

            * `pairs`: Ordered POI transition identifiers.
            * `frac_blend`: Blend weight between seconds and fraction models.
            * `bound_lo_pct` / `bound_hi_pct`: Percentile bounds used
            during fitting.
            * `gap`: Mapping from POI transitions to fitted statistics.

            This method does not store raw training data, only the learned
            statistical parameters.
        """
        d = {
            "pairs": self.pairs,
            "frac_blend": self.frac_blend,
            "bound_lo_pct": self.bound_lo_pct,
            "bound_hi_pct": self.bound_hi_pct,
            "gap": {k: asdict(v) for k, v in self.gap.items()},
        }
        Path(path).write_text(json.dumps(d, indent=2))

    @staticmethod
    def load(path: Path) -> "QModelV6YOLO_SpacingPrior":
        """Load a serialized spacing prior from disk.

        Deserializes a previously saved :class:`QModelV6YOLO_SpacingPrior`
        instance from a JSON file produced by :meth:`save`.

        The method reconstructs both the model configuration and all fitted
        :class:`GapStat` objects, restoring the full spacing prior used for
        likelihood evaluation.

        Args:
            path: Path to the JSON file containing the serialized model.

        Returns:
            A fully reconstructed `QModelV6YOLO_SpacingPrior` instance with
            identical parameters and fitted gap statistics.

        Notes:
            This method assumes the input file exactly matches the schema
            produced by :meth:`save`. No schema migration or validation is
            performed. If the underlying dataclass definitions have changed,
            loading may raise a `TypeError` or silently produce incorrect
            behavior.

            Each entry in the stored `gap` dictionary is expanded using:

                GapStat(**v)

            so all keys in the JSON must match the fields of :class:`GapStat`.
        """
        d = json.loads(Path(path).read_text())
        p = QModelV6YOLO_SpacingPrior(
            pairs=d["pairs"],
            frac_blend=d["frac_blend"],
            bound_lo_pct=d["bound_lo_pct"],
            bound_hi_pct=d["bound_hi_pct"],
        )
        p.gap = {k: GapStat(**v) for k, v in d["gap"].items()}
        return p
