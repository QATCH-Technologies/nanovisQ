"""
QATCH.QModel.models.qmodel_onyx.onyx_spacing_prior.py

A flat, pairwise configuration prior over POI positions, learned from
complete-fill ground-truth configurations. This is the model the EDA selected:
flat (not run-conditional) and pairwise (not autoregressive), because on the
real data those additions hurt rather than helped.

At decode time the prior scores a candidate configuration by the summed
log-likelihood of its gaps, plus hard feasibility (monotonic order, learned
min/max gap bounds). The score combines with detection confidence in the
`onyx_decoder.py`

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-07-09
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np

# POI order; consecutive pairs define the gaps the prior models.
POI_ORDER = ["POI1", "POI2", "POI3", "POI4", "POI5"]


@dataclass
class GapStat:
    """Stores statistical models for a consecutive point-of-interest (POI) gap.

    Each instance contains the parameters of log-normal distributions describing
    the duration of a gap in both absolute time and as a fraction of the total
    run span. It also records robust minimum and maximum feasible gap durations
    derived from the training data.

    Attributes:
        log_mu_sec: Mean of the natural logarithm of the gap duration, in
            seconds.
        log_sd_sec: Standard deviation of the natural logarithm of the gap
            duration, in seconds.
        log_mu_frac: Mean of the natural logarithm of the gap expressed as a
            fraction of the total run duration.
        log_sd_frac: Standard deviation of the natural logarithm of the gap
            fraction.
        min_gap_sec: Minimum feasible gap duration, in seconds, estimated from
            robust percentiles.
        max_gap_sec: Maximum feasible gap duration, in seconds, estimated from
            robust percentiles.
        n: Number of samples used to estimate the statistics.
    """

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
class OnyxSpacingPrior:
    """Stores statistical priors describing the expected spacing between POIs.

    The spacing prior models the expected temporal separation between
    consecutive points of interest (POIs). Each consecutive POI pair is
    associated with a fitted `GapStat` describing the distribution of gap
    durations observed during training. The prior can evaluate candidate POI
    sequences using both absolute time and normalized run duration.

    Attributes:
        pairs: Ordered list of consecutive POI pair identifiers (for example,
            `["POI1->POI2", "POI2->POI3"]`).
        gap: Mapping from POI pair identifier to its corresponding
            `GapStat`.
        frac_blend: Weight used to blend the fraction-based and
            seconds-based log-likelihoods. A value of `0.0` uses only
            absolute gap duration, while `1.0` uses only the gap expressed
            as a fraction of the total run duration.
        bound_lo_pct: Lower percentile used when fitting robust feasibility
            bounds for gap durations.
        bound_hi_pct: Upper percentile used when fitting robust feasibility
            bounds for gap durations.
    """

    pairs: List[str]  # e.g. ["POI1->POI2", ...]
    gap: Dict[str, GapStat] = field(default_factory=dict)

    # blend weight between seconds-based and fraction-based log-likelihood.
    # 0 = pure seconds, 1 = pure span-fraction. Default mixes both.
    frac_blend: float = 0.5

    # feasibility bound percentiles used when fitting.
    bound_lo_pct: float = 0.5
    bound_hi_pct: float = 99.5

    @staticmethod
    def fit(
        configs_sec: np.ndarray,
        frac_blend: float = 0.5,
        bound_lo_pct: float = 0.5,
        bound_hi_pct: float = 99.5,
    ) -> "OnyxSpacingPrior":
        """Fits a spacing prior from complete POI configurations.

        The input consists of complete, strictly increasing POI timestamps for
        multiple runs. For each consecutive POI pair, the method estimates a
        log-normal distribution for both the absolute gap duration (seconds) and
        the gap normalized by the total run duration. Robust lower and upper
        feasibility bounds are also computed from the specified percentiles.

        Args:
            configs_sec: A `(N, P)` array containing POI timestamps in seconds,
                where `N` is the number of runs and `P` equals
                `len(POI_ORDER)`. Each row must contain finite, strictly
                increasing timestamps.
            frac_blend: Weight used to blend fraction-based and seconds-based
                log-likelihoods when evaluating candidate sequences.
            bound_lo_pct: Lower percentile used to estimate the minimum feasible
                gap duration.
            bound_hi_pct: Upper percentile used to estimate the maximum feasible
                gap duration.

        Returns:
            An `OnyxSpacingPrior` populated with fitted gap statistics for each
            consecutive POI pair.

        Raises:
            ValueError: If `configs_sec` does not have the expected shape,
                contains non-finite values, contains non-increasing timestamps,
                or does not contain any valid positive gaps for a POI pair.
        """
        configs_sec = np.asarray(configs_sec, dtype=float)
        if configs_sec.ndim != 2 or configs_sec.shape[1] != len(POI_ORDER):
            raise ValueError(
                f"expected (N, {len(POI_ORDER)}) POI configurations, got {configs_sec.shape}"
            )
        if not np.all(np.isfinite(configs_sec)):
            raise ValueError("configs_sec must contain only finite values")
        if not np.all(np.diff(configs_sec, axis=1) > 0):
            raise ValueError("configs_sec rows must be strictly ascending")
        _, P = configs_sec.shape
        span = configs_sec[:, -1] - configs_sec[:, 0]
        span = np.where(span < 1e-9, np.nan, span)
        pairs = [f"{POI_ORDER[i]}->{POI_ORDER[i+1]}" for i in range(P - 1)]
        prior = OnyxSpacingPrior(
            pairs=pairs, frac_blend=frac_blend, bound_lo_pct=bound_lo_pct, bound_hi_pct=bound_hi_pct
        )
        for i in range(P - 1):
            g_sec = configs_sec[:, i + 1] - configs_sec[:, i]
            g_sec = g_sec[np.isfinite(g_sec) & (g_sec > 0)]
            g_frac = (configs_sec[:, i + 1] - configs_sec[:, i]) / span
            g_frac = g_frac[np.isfinite(g_frac) & (g_frac > 0)]
            if g_sec.size == 0 or g_frac.size == 0:
                raise ValueError(f"no valid positive gaps found for {pairs[i]}")
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
        """Computes composed gap statistics between non-adjacent POIs.

        Builds the expected spacing statistics for a gap spanning multiple
        consecutive POI intervals by composing the fitted log-normal gap models
        between the intermediate POIs. Consecutive gap statistics are returned
        directly, while longer gaps are approximated by combining the component
        distributions: median gap durations are summed in linear space and
        variances are accumulated in log space.

        This method is used when two detected POIs are present but one or more
        globally ordered POIs between them are missing. In this case, the observed
        spacing should be evaluated against the composed distribution of the
        intervening gaps rather than the statistics of a single consecutive pair.

        Args:
            i (int): Index of the starting POI in `POI_ORDER`.
            j (int): Index of the ending POI in `POI_ORDER`. Must be greater than
                `i`.

        Returns:
            A `GapStat` instance representing the composed spacing distribution
            between POI `i` and POI `j`.

        Raises:
            ValueError: If `j` is less than or equal to `i`.

        NOTE:
            Computed composed statistics are cached on the instance to avoid
            repeated composition of the same POI pair.
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

    def _stat_loglik(
        self,
        gs: GapStat,
        gap_sec: float,
        span_sec: float,
    ) -> float:
        """Computes the log-likelihood of an observed POI gap.

        Evaluates how well an observed gap duration matches the expected spacing
        distribution represented by `GapStat`. The score combines two
        log-normal likelihoods: one based on the absolute gap duration in seconds
        and one based on the gap normalized by the total run span. The relative
        contribution of these two terms is controlled by `frac_blend`.

        The calculation omits constant terms from the normal probability density
        function and retains only the components that affect the relative score.

        Args:
            gs (GapStat): Gap statistics describing the expected spacing distribution.
            gap_sec (float): Observed gap duration between POIs, in seconds.
            span_sec (float): Total duration of the run, in seconds, used to normalize the
                gap for the span-fraction likelihood.

        Returns:
            The blended log-likelihood score for the observed gap. Returns a large
            negative value for invalid (non-positive) gap durations.
        """
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

    def gap_loglik(
        self,
        pair_idx: int,
        gap_sec: float,
        span_sec: float,
    ) -> float:
        """Computes the log-likelihood of a consecutive POI gap.

        Evaluates the plausibility of an observed gap between two consecutive POIs
        using the fitted spacing statistics for the specified POI pair. The score
        combines absolute time-based and span-normalized log-normal likelihoods
        according to the configured blend weight.

        A non-positive `span_sec` disables the span-fraction component and falls
        back to evaluating only the absolute gap duration. This is useful for
        partial fills where the total run span does not correspond to the complete
        POI sequence used during prior fitting.

        Args:
            pair_idx (int): Index of the consecutive POI pair in `self.pairs`.
            gap_sec (float): Observed gap duration between the POIs, in seconds.
            span_sec (float): Total run duration, in seconds. Values less than or equal to
                zero disable span-fraction scoring.

        Returns:
            The log-likelihood score for the observed gap. Higher values indicate
            that the gap is more consistent with the fitted spacing prior.
        """
        return self._stat_loglik(self.gap[self.pairs[pair_idx]], gap_sec, span_sec)

    def gap_loglik_between(
        self,
        i: int,
        j: int,
        gap_sec: float,
        span_sec: float,
    ) -> float:
        """Computes the log-likelihood of a gap between arbitrary POI indices.

        Evaluates the plausibility of the observed spacing between two POIs in the
        global POI ordering. If the POIs are consecutive, the fitted gap statistics
        for that pair are used directly. If intermediate POIs are absent, the
        expected spacing is evaluated using a composed gap distribution that
        combines the statistics of the intervening consecutive gaps.

        Args:
            i (int): Index of the starting POI in the global POI ordering.
            j (int): Index of the ending POI in the global POI ordering. Must be greater
                than `i`.
            gap_sec (float): Observed gap duration between the two POIs, in seconds.
            span_sec (float): Total run duration, in seconds, used for span-fraction
                likelihood evaluation. Values less than or equal to zero disable
                the fraction component.

        Returns:
            The log-likelihood score for the observed gap. Higher values indicate
            greater agreement with the expected POI spacing distribution.

        Raises:
            ValueError: If `j` is not greater than `i`.
        """
        """Log-likelihood of the gap between global POI indices i and j
        (j > i). Uses the fitted stat when adjacent, the composed stat when
        the pair spans absent POIs."""
        return self._stat_loglik(self.composed_stat(i, j), gap_sec, span_sec)

    def gap_loglik_scoped(
        self,
        i: int,
        j: int,
        gap_sec: float,
        span_sec: float,
        span_lo: int,
        span_hi: int,
    ) -> float:
        """Computes a scoped log-likelihood for a POI gap within a partial fill.

        Evaluates the likelihood of the gap between two global POI indices while
        adjusting the span-fraction component to account for the subset of POIs
        currently placed during decoding.

        The fitted fraction statistics are based on complete fills, where the span
        is defined as the time between the first and last POIs in the complete
        sequence. For partial fills, the observed span contains fewer gaps and the
        fitted fraction means may no longer represent the expected proportions.
        This method re-references the fraction mean using the expected absolute
        gap durations within the active POI span while preserving the fitted
        relative-scale variance.

        When the active span covers the complete POI chain, the original fitted
        fraction statistics are used without modification.

        Args:
            i (int): Index of the starting POI in the global POI ordering.
            j (int): Index of the ending POI in the global POI ordering. Must be greater
                than `i`.
            gap_sec (float): Observed gap duration between POIs, in seconds.
            span_sec (float): Observed duration of the active POI span, in seconds.
            span_lo (int): Index of the first placed POI defining the active span.
            span_hi (int): Index of the last placed POI defining the active span.

        Returns:
            The scoped log-likelihood score for the observed gap. Higher values
            indicate greater agreement with the expected spacing prior.

        NOTE:
            The scoped fraction mean is derived from the expected seconds-based
            gap medians:

            `log_mu_frac = log_mu_sec(gap) - log(sum(expected_gap_seconds))`

            This allows the prior to adapt to partial fills without making the
            model explicitly dependent on individual runs.
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
        """Checks whether a gap duration is within learned feasibility bounds.

        Applies a hard constraint using the robust minimum and maximum gap
        durations learned for a consecutive POI pair. A multiplicative slack factor
        is applied to avoid rejecting borderline values that may still represent
        valid observations.

        Args:
            pair_idx (int): Index of the consecutive POI pair in `self.pairs`.
            gap_sec (float): Observed gap duration between the POIs, in seconds.
            slack (float): Multiplicative tolerance applied to the learned bounds. Values
                greater than `1.0` widen the acceptable range.

        Returns:
            `True` if the gap duration is positive and falls within the adjusted
            feasible range, otherwise `False`.
        """

        gs = self.gap[self.pairs[pair_idx]]
        return (
            gap_sec > 0 and gap_sec >= gs.min_gap_sec / slack and gap_sec <= gs.max_gap_sec * slack
        )

    def gap_feasible_between(
        self,
        i: int,
        j: int,
        gap_sec: float,
        slack: float = 1.5,
    ) -> bool:
        """Checks whether a gap between arbitrary POI indices is feasible.

        Applies a hard feasibility constraint to the observed spacing between two
        POIs in the global ordering. Consecutive POI pairs use their fitted gap
        bounds directly, while non-adjacent pairs use bounds derived from the
        composed statistics of the intervening gaps.

        A multiplicative slack factor is applied to the learned bounds to allow
        borderline values that may still represent valid observations.

        Args:
            i (int): Index of the starting POI in the global POI ordering.
            j (int): Index of the ending POI in the global POI ordering. Must be greater
                than `i`.
            gap_sec (float): Observed gap duration between the POIs, in seconds.
            slack (float): Multiplicative tolerance applied to the learned minimum and
                maximum bounds. Values greater than `1.0` widen the feasible
                range.

        Returns:
            `True` if the gap duration is positive and falls within the adjusted
            feasible range, otherwise `False`.

        Raises:
            ValueError: If `j` is not greater than `i`.
        """
        gs = self.composed_stat(i, j)
        return (
            gap_sec > 0 and gap_sec >= gs.min_gap_sec / slack and gap_sec <= gs.max_gap_sec * slack
        )

    def config_loglik(self, times_sec: List[float]) -> float:
        """Computes the total spacing log-likelihood of an ordered POI sequence.

        Evaluates the overall plausibility of a complete POI configuration by
        summing the individual gap log-likelihoods between consecutive POIs. Each
        gap is scored using the learned spacing prior, with the total configuration
        span used for the span-fraction likelihood component.

        Args:
            times_sec (List[float]): Ordered list of POI timestamps in seconds. Values must
                correspond to the expected global POI ordering.

        Returns:
            The summed log-likelihood score for the complete POI configuration.
            Higher values indicate that the observed spacing is more consistent
            with the learned prior.
        """
        span = times_sec[-1] - times_sec[0]
        total = 0.0
        for i in range(len(times_sec) - 1):
            total += self.gap_loglik(i, times_sec[i + 1] - times_sec[i], span)
        return total

    def save(self, path: str) -> None:
        """Serializes the spacing prior to a JSON file.

        Converts the prior configuration, fitting parameters, and gap statistics
        into a JSON-compatible dictionary and writes the resulting representation
        to disk. The serialized file can later be restored using
        :meth:`load`.

        Args:
            path (str): Destination file path for the serialized spacing prior.
        """
        d = {
            "pairs": self.pairs,
            "frac_blend": self.frac_blend,
            "bound_lo_pct": self.bound_lo_pct,
            "bound_hi_pct": self.bound_hi_pct,
            "gap": {k: asdict(v) for k, v in self.gap.items()},
        }
        with open(path, "w") as f:
            f.write(json.dumps(d, indent=2))

    @staticmethod
    def load(path: str) -> "OnyxSpacingPrior":
        """Loads a spacing prior from a JSON file.

        Reconstructs an `OnyxSpacingPrior` instance from a previously serialized
        JSON representation, restoring both the global fitting parameters and the
        per-pair `GapStat` objects.

        Args:
            path (str): Path to the JSON file containing a serialized spacing prior.

        Returns:
            A reconstructed `OnyxSpacingPrior` instance.
        """
        with open(path, "r") as f:
            d = json.load(f)
        p = OnyxSpacingPrior(
            pairs=d["pairs"],
            frac_blend=d["frac_blend"],
            bound_lo_pct=d["bound_lo_pct"],
            bound_hi_pct=d["bound_hi_pct"],
        )
        p.gap = {k: GapStat(**v) for k, v in d["gap"].items()}
        return p
