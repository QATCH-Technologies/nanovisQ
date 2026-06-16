"""
spacing_prior.py
================

Part of the QModel V6 YOLO predictor subpackage (v6.4.0).

A JOINT configuration prior over POI positions, learned from complete-fill
ground-truth configurations. This supersedes the v6.3 flat/pairwise
independent-gap prior, which the EDA-era assumptions made adequate for
Newtonian (constant-viscosity) fills but which fails on non-Newtonian
solutions: there the *absolute* gaps stretch run-to-run while their *shape*
(the ratios between gaps) stays stable, and an independent per-gap
log-normal cannot represent "same shape, different scale" without paying a
large likelihood penalty — so the decode loses to greedy on exactly those
runs.

The model
---------
Work in log-space on the P-1 consecutive gaps  g_k = t(POI_{k+1}) - t(POI_k).
The anchor gap  T = g_1 = gap(POI1->POI2)  defines the run's characteristic
timescale; every later gap is, physically, a ratio of T that drifts smoothly
with viscosity / shear-thinning. We capture that with a one-factor Gaussian
on the log-gaps:

    log g  ~  N( mu , Sigma ),     Sigma = beta beta^T + diag(psi)

  * mu     : mean log-gap vector (the canonical "shape")
  * beta   : per-gap loading on a single shared latent slowness factor s~N(0,1)
             (a slow/viscous run moves ALL gaps together along beta)
  * psi    : independent per-gap residual log-variance

Equivalently  log g_k = mu_k + beta_k s + eps_k . Because the latent s is
shared, a UNIFORMLY stretched run (all log-gaps shifted along beta) costs
almost nothing — only deviations in SHAPE are penalised. That is precisely
the scale-invariance the old prior lacked.

Ratio nuance (the "T-ratio" structure you asked for)
----------------------------------------------------
The conditional mean of any gap GIVEN the anchor (and any other observed
gaps) is linear in the observed log-gaps — i.e. once T is known, each later
gap has an *expected ratio* to T, and that ratio is itself nudged by how
much T deviates from its own mean (a longer-than-typical T predicts a
longer-than-typical later gap, with a learned, non-constant slope). This is
the "more nuanced" T-ratio the flat prior could not express, and it falls
straight out of the Gaussian conditionals (see ``cond_gap_loglik``).

Interface
---------
This class is a DROP-IN superset of the v6.3 SpacingPrior: every method the
v6.3 decode.py called still exists with identical semantics
(``gap_loglik``, ``gap_loglik_scoped``, ``gap_loglik_between``,
``gap_feasible``, ``gap_feasible_between``, ``composed_stat``,
``config_loglik``, ``frac_blend``, ``save``/``load``). New joint methods
(``config_loglik_joint``, ``cond_gap_loglik``, ``anchor_loglik``) power the
v6.4 joint decode. ``load`` transparently reads BOTH the old flat JSON and
the new joint JSON, and ``fit`` produces the joint model; a flat-only file
loads as a degenerate joint model (zero latent loading) so old assets keep
working unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# POI order; consecutive pairs define the gaps the prior models.
POI_ORDER = ["POI1", "POI2", "POI3", "POI4", "POI5"]

_LOG2PI = float(np.log(2.0 * np.pi))
_FLOOR = -1e9


# ---------------------------------------------------------------------------
#  Backward-compatible per-gap stat (kept so old JSON round-trips and so the
#  composed_stat / feasibility helpers the decode relies on still work).
# ---------------------------------------------------------------------------
@dataclass
class GapStat:
    """Log-normal stats for one consecutive-gap, in seconds and span-fraction."""

    log_mu_sec: float
    log_sd_sec: float
    log_mu_frac: float
    log_sd_frac: float
    min_gap_sec: float
    max_gap_sec: float
    n: int


def _fit_one_factor(
    L: np.ndarray, iters: int = 200, tol: float = 1e-7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit log g ~ N(mu, beta beta^T + diag(psi)) by EM (one-factor analysis).

    L : (N, D) matrix of log-gaps. Returns (mu, beta, psi).

    A one-factor model is the right capacity here: D is small (4), the
    dominant correlation across gaps is the single shared "slowness" mode
    (viscosity / shear-thinning stretches the whole tail together), and one
    factor keeps Sigma well-conditioned and the fit stable on the modest
    complete-fill sample. EM is closed-form per step and converges in a few
    dozen iterations.
    """
    N, D = L.shape
    mu = L.mean(axis=0)
    Lc = L - mu
    S = (Lc.T @ Lc) / max(N, 1)  # empirical covariance
    # init: leading eigenvector scaled by sqrt(eigval); psi = residual diag
    w, V = np.linalg.eigh(S)
    lead = V[:, -1] * np.sqrt(max(w[-1], 1e-8))
    beta = lead.copy()
    psi = np.clip(np.diag(S) - beta**2, 1e-6, None)

    prev = -np.inf
    for _ in range(iters):
        # E-step: posterior of latent s given each obs (factor analysis)
        psi_inv = 1.0 / psi
        # m = beta^T psi^-1 beta  (scalar, one factor)
        bpb = float(beta @ (psi_inv * beta))
        cond_var = 1.0 / (1.0 + bpb)  # Var(s | x)
        # E[s | x_i] = cond_var * beta^T psi^-1 x_i
        Ez = cond_var * (Lc @ (psi_inv * beta))  # (N,)
        Ezz = cond_var + Ez**2  # E[s^2 | x_i]
        # M-step
        # beta = (sum x_i E[s|x_i]) / (sum E[s^2|x_i])
        beta = (Lc.T @ Ez) / max(float(Ezz.sum()), 1e-12)
        psi = np.clip(
            np.diag(S) - 2.0 * beta * ((Lc.T @ Ez) / N) + beta**2 * float(Ezz.mean()),
            1e-6,
            None,
        )
        # log-likelihood (for convergence check)
        Sigma = np.outer(beta, beta) + np.diag(psi)
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            break
        Sinv = np.linalg.inv(Sigma)
        ll = -0.5 * N * (D * _LOG2PI + logdet) - 0.5 * float(np.sum(Lc * (Lc @ Sinv)))
        if abs(ll - prev) < tol * max(1.0, abs(prev)):
            prev = ll
            break
        prev = ll
    return mu, beta, psi


@dataclass
class SpacingPrior:
    """Joint scale-shape spacing prior. See module docstring."""

    pairs: List[str]  # ["POI1->POI2", ...]
    # --- joint log-gap model (the new core) ---
    mu: List[float] = field(default_factory=list)  # (D,) mean log-gap
    beta: List[float] = field(default_factory=list)  # (D,) latent loading
    psi: List[float] = field(default_factory=list)  # (D,) residual var
    # --- legacy per-gap stats (composition / feasibility / flat fallback) ---
    gap: Dict[str, GapStat] = field(default_factory=dict)
    # blend weight retained for interface compatibility; in the joint model it
    # mixes the joint (shape, scale-free) term with the absolute-seconds anchor
    # term. 0 = ignore the absolute anchor scale (pure shape); 1 = legacy-like
    # weighting toward absolute seconds. Default leans on shape, which is what
    # generalises to non-Newtonian runs.
    frac_blend: float = 0.5
    bound_lo_pct: float = 0.5
    bound_hi_pct: float = 99.5
    # marginal of the anchor gap T = g_1, in log-seconds (for anchor_loglik)
    anchor_log_mu: float = 0.0
    anchor_log_sd: float = 1.0
    version: int = 2  # 2 = joint; 1 = flat (legacy)

    # ------------------------------------------------------------------ fit
    @staticmethod
    def fit(
        configs_sec: np.ndarray,
        frac_blend: float = 0.5,
        bound_lo_pct: float = 0.5,
        bound_hi_pct: float = 99.5,
    ) -> "SpacingPrior":
        """Fit the joint model from complete configurations.

        configs_sec : (N, P) POI times in seconds, strictly ascending rows
        (complete fills only). P must equal len(POI_ORDER).
        """
        N, P = configs_sec.shape
        assert P == len(POI_ORDER), f"expected {len(POI_ORDER)} POIs, got {P}"
        D = P - 1
        pairs = [f"{POI_ORDER[i]}->{POI_ORDER[i+1]}" for i in range(D)]

        gaps = np.diff(configs_sec, axis=1)  # (N, D) seconds
        # keep only physically valid rows (all gaps positive)
        good = np.all(np.isfinite(gaps) & (gaps > 0), axis=1)
        gaps = gaps[good]
        if len(gaps) < D + 2:
            raise SystemExit(
                f"Too few valid complete-fill configs ({len(gaps)}) to fit a " f"{D}-d joint prior."
            )
        L = np.log(gaps)  # (N, D) log-gaps
        mu, beta, psi = _fit_one_factor(L)

        prior = SpacingPrior(
            pairs=pairs,
            mu=[float(x) for x in mu],
            beta=[float(x) for x in beta],
            psi=[float(x) for x in psi],
            frac_blend=frac_blend,
            bound_lo_pct=bound_lo_pct,
            bound_hi_pct=bound_hi_pct,
            anchor_log_mu=float(L[:, 0].mean()),
            anchor_log_sd=float(L[:, 0].std() + 1e-6),
            version=2,
        )

        # legacy per-gap stats: marginal log-normal per gap (Sigma diagonal),
        # plus span-fraction stats and robust seconds bounds. These power
        # composed_stat / feasibility and the flat fallback paths.
        span = configs_sec[good, -1] - configs_sec[good, 0]
        span = np.where(span < 1e-9, np.nan, span)
        Sigma_diag = beta**2 + psi
        for i in range(D):
            g_sec = gaps[:, i]
            g_frac = g_sec / span
            g_frac = g_frac[np.isfinite(g_frac) & (g_frac > 0)]
            lf = np.log(g_frac)
            prior.gap[pairs[i]] = GapStat(
                log_mu_sec=float(mu[i]),
                log_sd_sec=float(np.sqrt(Sigma_diag[i]) + 1e-6),
                log_mu_frac=float(lf.mean()) if len(lf) else float(mu[i]),
                log_sd_frac=float(lf.std() + 1e-6) if len(lf) else 1.0,
                min_gap_sec=float(np.percentile(g_sec, bound_lo_pct)),
                max_gap_sec=float(np.percentile(g_sec, bound_hi_pct)),
                n=int(len(g_sec)),
            )
        return prior

    # --------------------------------------------------- numpy views (cached)
    def _vec(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache = getattr(self, "_vec_cache", None)
        if cache is None:
            mu = np.asarray(self.mu, dtype=float)
            beta = np.asarray(self.beta, dtype=float)
            psi = np.asarray(self.psi, dtype=float)
            cache = (mu, beta, psi)
            object.__setattr__(self, "_vec_cache", cache)
        return cache

    @property
    def D(self) -> int:
        return len(self.pairs)

    def _has_joint(self) -> bool:
        return self.version >= 2 and len(self.mu) == self.D and self.D > 0

    # =====================================================================
    #  JOINT model: full multivariate-normal log-gap density and its
    #  conditionals. These are the new core used by the v6.4 joint decode.
    # =====================================================================
    def _Sigma(self, idx: Sequence[int]) -> np.ndarray:
        mu, beta, psi = self._vec()
        b = beta[list(idx)]
        return np.outer(b, b) + np.diag(psi[list(idx)])

    def _mvn_loglik(self, idx: Sequence[int], log_gaps: np.ndarray) -> float:
        """log N(log_gaps ; mu[idx], Sigma[idx]) for a SUBSET of consecutive
        gaps identified by their global gap indices ``idx``. Used when only a
        subset of gaps is observed (partial fills / non-adjacent present POIs
        handled by composition upstream)."""
        mu, _, _ = self._vec()
        idx = list(idx)
        d = len(idx)
        if d == 0:
            return 0.0
        x = np.asarray(log_gaps, dtype=float) - mu[idx]
        Sig = self._Sigma(idx)
        sign, logdet = np.linalg.slogdet(Sig)
        if sign <= 0:
            return _FLOOR
        sol = np.linalg.solve(Sig, x)
        quad = float(x @ sol)
        return -0.5 * (d * _LOG2PI + logdet + quad)

    def config_loglik_joint(
        self, times_sec: Sequence[float], gap_index: Optional[Sequence[int]] = None
    ) -> float:
        """Joint shape log-likelihood of a configuration.

        times_sec  : ordered POI times of the PRESENT pois.
        gap_index  : the global gap index of each consecutive present-pair,
                     i.e. for present global POI indices p_0<p_1<...<p_m the
                     gap (p_a -> p_{a+1}) is a COMPOSED gap spanning
                     [p_a, p_{a+1}); see ``_composed_log_gap_model``. When the
                     present pois are globally contiguous starting at POI1,
                     this is just range(len(times)-1).

        For contiguous present prefixes this is the exact one-factor MVN
        density on the observed log-gaps (scale-invariant in the shared
        latent). For non-contiguous present sets the intervening gaps are
        marginalised by composing the latent loadings and residual variances
        (sums of the bridged dimensions), which preserves the latent coupling
        across the bridge — the property the old per-gap composition lacked.
        """
        t = np.asarray(times_sec, dtype=float)
        if len(t) < 2:
            return 0.0
        if not self._has_joint():
            # flat fallback: independent marginals (legacy behaviour)
            return self.config_loglik(list(t))
        seg = self._segment_model(t, gap_index)
        if seg is None:
            return _FLOOR
        mu_s, Sig_s, log_gaps = seg
        x = log_gaps - mu_s
        sign, logdet = np.linalg.slogdet(Sig_s)
        if sign <= 0:
            return _FLOOR
        sol = np.linalg.solve(Sig_s, x)
        return -0.5 * (len(x) * _LOG2PI + logdet + float(x @ sol))

    def _composed_log_gap_model(self, lo: int, hi: int) -> Tuple[float, float, float]:
        """Latent model for the COMPOSED gap spanning global gap dims
        [lo, hi) (i.e. POI_ORDER[lo] -> POI_ORDER[hi]).

        Composition rule, derived from the additive log-normal-sum
        approximation used by the legacy composed_stat but carried through to
        the latent parameters so the shared factor survives the bridge:
        the composed gap's log-mean is log(sum exp(mu_k)); its latent loading
        and residual variance are the SUMS over the bridged dims (a stretch
        of the run stretches every bridged sub-gap, so loadings add). Returns
        (mu_comp, beta_comp, psi_comp).
        """
        mu, beta, psi = self._vec()
        ks = list(range(lo, hi))
        med = float(np.sum(np.exp(mu[ks])))
        mu_comp = float(np.log(max(med, 1e-9)))
        beta_comp = float(np.sum(beta[ks]))
        psi_comp = float(np.sum(psi[ks]))
        return mu_comp, beta_comp, psi_comp

    def _segment_model(
        self, times_sec: np.ndarray, gap_index: Optional[Sequence[int]]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Build (mu, Sigma, observed_log_gaps) for the present configuration.

        Each present consecutive-pair maps to a composed gap over a contiguous
        block of global gap dims; we compose the latent params per pair and
        assemble the one-factor Sigma = beta beta^T + diag(psi) across pairs,
        so the shared latent still couples the (possibly bridged) pairs.
        """
        t = np.asarray(times_sec, dtype=float)
        g = np.diff(t)
        if np.any(g <= 0):
            return None
        log_g = np.log(g)
        m = len(g)
        if gap_index is None:
            blocks = [(i, i + 1) for i in range(m)]  # contiguous from POI1
        else:
            gi = list(gap_index)
            # gap_index gives, per present pair, the (lo,hi) block via the
            # global indices of the bracketing present pois.
            blocks = gi  # already (lo,hi) tuples
        mu_s = np.empty(m)
        beta_s = np.empty(m)
        psi_s = np.empty(m)
        for a, blk in enumerate(blocks):
            lo, hi = blk if isinstance(blk, (tuple, list)) else (blk, blk + 1)
            mu_s[a], beta_s[a], psi_s[a] = self._composed_log_gap_model(lo, hi)
        Sig = np.outer(beta_s, beta_s) + np.diag(psi_s)
        return mu_s, Sig, log_g

    def cond_gap_loglik(
        self,
        i: int,
        j: int,
        gap_sec: float,
        observed: Optional[Sequence[Tuple[int, int, float]]] = None,
    ) -> float:
        """Log-density of the gap POI_ORDER[i]->POI_ORDER[j] CONDITIONED on
        already-observed gaps.

        observed : list of (lo, hi, gap_sec) for gaps already placed in this
                   configuration (each a composed block [lo,hi)). When non-empty,
                   the conditional mean of the queried gap is shifted toward the
                   value implied by the shared latent — this is the "longer T
                   predicts a longer later gap" T-ratio coupling. When empty it
                   reduces to the queried gap's marginal log-normal density.

        This is what the DP uses to score one edge while remaining consistent
        with the full joint: scoring edges by their conditionals telescopes to
        the exact joint density (chain rule), so a left-to-right DP that
        accumulates conditionals is scoring the true MVN — no independence
        assumption.
        """
        if not self._has_joint():
            return self.gap_loglik_between(i, j, gap_sec, 0.0)
        if gap_sec <= 0:
            return _FLOOR
        mu_q, beta_q, psi_q = self._composed_log_gap_model(i, j)
        x_q = float(np.log(gap_sec))
        if not observed:
            var_q = beta_q * beta_q + psi_q
            z = (x_q - mu_q) / np.sqrt(var_q)
            return float(-0.5 * (_LOG2PI + np.log(var_q) + z * z))
        # Build joint over [observed..., queried] using the one-factor structure
        obs = list(observed)
        mus, betas, psis, xs = [], [], [], []
        for lo, hi, gs in obs:
            if gs <= 0:
                return _FLOOR
            mc, bc, pc = self._composed_log_gap_model(lo, hi)
            mus.append(mc)
            betas.append(bc)
            psis.append(pc)
            xs.append(np.log(gs))
        mus.append(mu_q)
        betas.append(beta_q)
        psis.append(psi_q)
        mu_v = np.asarray(mus)
        beta_v = np.asarray(betas)
        psi_v = np.asarray(psis)
        xs = np.asarray(xs)
        k = len(obs)
        Sig = np.outer(beta_v, beta_v) + np.diag(psi_v)
        Soo = Sig[:k, :k]
        Soq = Sig[:k, k]
        Sqq = float(Sig[k, k])
        sol = np.linalg.solve(Soo, (xs - mu_v[:k]))
        cond_mu = mu_v[k] + float(Soq @ sol)
        cond_var = Sqq - float(Soq @ np.linalg.solve(Soo, Soq))
        cond_var = max(cond_var, 1e-9)
        z = (x_q - cond_mu) / np.sqrt(cond_var)
        return float(-0.5 * (_LOG2PI + np.log(cond_var) + z * z))

    def anchor_loglik(self, T_sec: float) -> float:
        """Absolute-scale log-density of the anchor gap T = gap(POI1->POI2).

        The joint shape term is (deliberately) scale-free in the shared latent,
        so a run can stretch freely along the slowness axis. This term re-adds
        a soft, broad preference for physically plausible absolute T, weighted
        by ``frac_blend`` at scoring time. It is intentionally weak: its job is
        to break ties between wildly different absolute scales, NOT to pull
        non-Newtonian runs back toward the Newtonian median (that was the old
        prior's failure)."""
        if T_sec <= 0:
            return _FLOOR
        z = (np.log(T_sec) - self.anchor_log_mu) / self.anchor_log_sd
        return float(-0.5 * (_LOG2PI + 2 * np.log(self.anchor_log_sd) + z * z))

    # =====================================================================
    #  LEGACY interface (unchanged semantics) — composition, feasibility,
    #  per-gap log-likelihood, flat config_loglik. Kept so the v6.3 decode
    #  still runs and so old flat JSON assets load.
    # =====================================================================
    def composed_stat(self, i: int, j: int) -> GapStat:
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
        if gap_sec <= 0:
            return _FLOOR
        z_sec = (np.log(gap_sec) - gs.log_mu_sec) / gs.log_sd_sec
        ll_sec = -0.5 * z_sec * z_sec - np.log(gs.log_sd_sec)
        if span_sec and span_sec > 0:
            frac = gap_sec / span_sec
            if frac <= 0:
                ll_frac = _FLOOR
            else:
                z_f = (np.log(frac) - gs.log_mu_frac) / gs.log_sd_frac
                ll_frac = -0.5 * z_f * z_f - np.log(gs.log_sd_frac)
        else:
            ll_frac = ll_sec
        return float((1 - self.frac_blend) * ll_sec + self.frac_blend * ll_frac)

    def gap_loglik(self, pair_idx: int, gap_sec: float, span_sec: float) -> float:
        return self._stat_loglik(self.gap[self.pairs[pair_idx]], gap_sec, span_sec)

    def gap_loglik_between(self, i: int, j: int, gap_sec: float, span_sec: float) -> float:
        return self._stat_loglik(self.composed_stat(i, j), gap_sec, span_sec)

    def gap_loglik_scoped(
        self, i: int, j: int, gap_sec: float, span_sec: float, span_lo: int, span_hi: int
    ) -> float:
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
        gs = self.gap[self.pairs[pair_idx]]
        return (
            gap_sec > 0 and gap_sec >= gs.min_gap_sec / slack and gap_sec <= gs.max_gap_sec * slack
        )

    def gap_feasible_between(self, i: int, j: int, gap_sec: float, slack: float = 1.5) -> bool:
        gs = self.composed_stat(i, j)
        return (
            gap_sec > 0 and gap_sec >= gs.min_gap_sec / slack and gap_sec <= gs.max_gap_sec * slack
        )

    def config_loglik(self, times_sec: List[float]) -> float:
        span = times_sec[-1] - times_sec[0]
        total = 0.0
        for i in range(len(times_sec) - 1):
            total += self.gap_loglik(i, times_sec[i + 1] - times_sec[i], span)
        return total

    # --------------------------------------------------------------- persist
    def save(self, path: Path) -> None:
        d = {
            "version": self.version,
            "pairs": self.pairs,
            "mu": self.mu,
            "beta": self.beta,
            "psi": self.psi,
            "frac_blend": self.frac_blend,
            "bound_lo_pct": self.bound_lo_pct,
            "bound_hi_pct": self.bound_hi_pct,
            "anchor_log_mu": self.anchor_log_mu,
            "anchor_log_sd": self.anchor_log_sd,
            "gap": {k: asdict(v) for k, v in self.gap.items()},
        }
        Path(path).write_text(json.dumps(d, indent=2))

    @staticmethod
    def load(path: Path) -> "SpacingPrior":
        d = json.loads(Path(path).read_text())
        version = int(d.get("version", 1))
        p = SpacingPrior(
            pairs=d["pairs"],
            frac_blend=d.get("frac_blend", 0.5),
            bound_lo_pct=d.get("bound_lo_pct", 0.5),
            bound_hi_pct=d.get("bound_hi_pct", 99.5),
            version=version,
        )
        p.gap = {k: GapStat(**v) for k, v in d.get("gap", {}).items()}
        if version >= 2 and "mu" in d:
            p.mu = list(map(float, d["mu"]))
            p.beta = list(map(float, d["beta"]))
            p.psi = list(map(float, d["psi"]))
            p.anchor_log_mu = float(d.get("anchor_log_mu", 0.0))
            p.anchor_log_sd = float(d.get("anchor_log_sd", 1.0))
        else:
            # legacy flat JSON: synthesise a degenerate joint model (no latent
            # coupling) from the per-gap marginals so joint methods still work.
            p.version = 1
            if p.gap:
                p.mu = [p.gap[pr].log_mu_sec for pr in p.pairs]
                p.beta = [0.0 for _ in p.pairs]
                p.psi = [max(p.gap[pr].log_sd_sec ** 2, 1e-6) for pr in p.pairs]
                p.anchor_log_mu = p.gap[p.pairs[0]].log_mu_sec
                p.anchor_log_sd = p.gap[p.pairs[0]].log_sd_sec
        return p
