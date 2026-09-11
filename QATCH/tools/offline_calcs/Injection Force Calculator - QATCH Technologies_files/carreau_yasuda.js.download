function log10(x) {
  return Math.log(x) / Math.LN10;
}

function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr) {
  const m = mean(arr);
  return Math.sqrt(mean(arr.map(x => (x - m) ** 2)));
}

function min(arr) {
  return Math.min(...arr);
}

function max(arr) {
  return Math.max(...arr);
}

function percentile(arr, p) {
  if (arr.length === 0) return NaN;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const w = idx - lo;
  return sorted[lo] * (1 - w) + sorted[hi] * w;
}

function logspace(startExp, endExp, n) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    out.push(10 ** (startExp + t * (endExp - startExp)));
  }
  return out;
}

function seededRandom(seed = 1) {
  let s = seed >>> 0;
  return function () {
    s += 0x6D2B79F5;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randUniform(rng, lo, hi) {
  return lo + (hi - lo) * rng();
}

function randLogUniform(rng, lo, hi) {
  return 10 ** randUniform(rng, log10(lo), log10(hi));
}

function carreauYasuda(gamma, eta0, etainf, n, gammaC, a = 2.0) {
  return etainf + (eta0 - etainf) *
    (1 + (gamma / gammaC) ** a) ** ((n - 1) / a);
}

function predictArray(gammaArr, fit, a = 2.0) {
  if (fit.model_type === "Newtonian") {
    return gammaArr.map(() => fit.eta0);
  }
  return gammaArr.map(g => carreauYasuda(g, fit.eta0, fit.etainf, fit.n, fit.gamma_c, a));
}

function linearSlope(x, y) {
  const mx = mean(x);
  const my = mean(y);
  let num = 0;
  let den = 0;
  for (let i = 0; i < x.length; i++) {
    num += (x[i] - mx) * (y[i] - my);
    den += (x[i] - mx) ** 2;
  }
  return den === 0 ? 0 : num / den;
}

function newtonianResult(gamma, eta, a, reason) {
  const etaMean = mean(eta);
  const etaStd  = std(eta);
  const gLow = Math.max(1, Math.min(...gamma) / 3);
  const gammaGrid = logspace(Math.log10(gLow), Math.log10(20e6), 300);
  // Median is the mean; ±2σ band captures measurement scatter.
  // Clamped to 1% of etaMean so the lower band never goes negative on log scale.
  const medBand  = gammaGrid.map(() => etaMean);
  const lowBand  = gammaGrid.map(() => Math.max(etaMean - 2 * etaStd, etaMean * 0.01));
  const highBand = gammaGrid.map(() => etaMean + 2 * etaStd);
  const rmseLog = Math.sqrt(mean(eta.map(v => (log10(etaMean) - log10(v)) ** 2)));

  const fit = {
    eta0: etaMean,
    etainf: etaMean,
    n: 1.0,
    gamma_c: null,
    a: a,
    rmse_log: rmseLog,
    model_type: "Newtonian"
  };

  return {
    best_fit: fit,
    accepted_fits: [fit],
    n_accepted_fits: 1,
    gamma_grid: gammaGrid,
    lower_band:  lowBand,
    median_band: medBand,
    upper_band:  highBand,
    measured_shear_rates: gamma,
    measured_viscosities: eta,
    diagnostics: {
      confidence: "Good",
      model_type: "Newtonian",
      warnings: [
        reason,
        "For a Newtonian profile, <em>n</em> and &gamma;<sub>c</sub> are not independently meaningful. <em>n</em> is set to 1 and &gamma;<sub>c</sub> is reported as None."
      ],
      eta_mean: etaMean,
      eta_cv: std(eta) / etaMean,
      eta_range_relative_to_mean: (max(eta) - min(eta)) / etaMean,
      shear_span_decades: log10(max(gamma)) - log10(min(gamma))
    }
  };
}

function robustNewtonianPrecheck(gamma, eta, a, opts = {}) {
  const maxCv = opts.max_cv ?? 0.08;
  const maxRange = opts.max_range ?? 0.15;
  const maxAbsLogSlope = opts.max_abs_log_slope ?? 0.025;
  const minShearSpanDecades = opts.min_shear_span_decades ?? 3.0;
  const minMaxShearRate = opts.min_max_shear_rate ?? 1e5;

  const etaMean = mean(eta);
  const etaCv = std(eta) / etaMean;
  const etaRange = (max(eta) - min(eta)) / etaMean;
  const shearSpan = log10(max(gamma)) - log10(min(gamma));

  const x = gamma.map(log10);
  const y = eta.map(log10);
  const slope = linearSlope(x, y);

  const broad = shearSpan >= minShearSpanDecades;
  const reachesHighShear = max(gamma) >= minMaxShearRate;
  const lowScatter = etaCv <= maxCv && etaRange <= maxRange;
  const nearZeroSlope = Math.abs(slope) <= maxAbsLogSlope;

  if (broad && reachesHighShear && lowScatter && nearZeroSlope) {
    return newtonianResult(
      gamma,
      eta,
      a,
      "Newtonian behavior detected: viscosity scatter is small with no meaningful shear-rate trend across at least 3 decades, including high shear-rate coverage above 1e5 s⁻¹."
    );
  }
  return null;
}

function rmseLogForParams(gamma, eta, params, a) {
  const [eta0, etainf, n, gammaC] = params;
  if (!(eta0 > etainf) || n < 0.2 || n > 1 || gammaC <= 0) return Infinity;

  let sum = 0;
  for (let i = 0; i < gamma.length; i++) {
    const pred = carreauYasuda(gamma[i], eta0, etainf, n, gammaC, a);
    if (!isFinite(pred) || pred <= 0) return Infinity;
    const err = log10(pred) - log10(eta[i]);
    sum += err * err;
  }
  return Math.sqrt(sum / gamma.length);
}

function makeFit(params, gamma, eta, a) {
  const [eta0, etainf, n, gammaC] = params;
  if (!(eta0 > etainf) || n < 0.2 || n > 1 || gammaC <= 0) return null;

  const preds = gamma.map(g => carreauYasuda(g, eta0, etainf, n, gammaC, a));
  if (preds.some(p => !isFinite(p) || p <= 0)) return null;
  for (let i = 1; i < preds.length; i++) {
    if (preds[i] > preds[i - 1]) return null;
  }

  const rmse = rmseLogForParams(gamma, eta, params, a);

  return {
    eta0: eta0,
    etainf: etainf,
    n: n,
    gamma_c: gammaC,
    a: a,
    rmse_log: rmse,
    model_type: "Carreau-Yasuda"
  };
}

function fitsPoints(fit, gamma, eta, a, pointTolerance) {
  const preds = predictArray(gamma, fit, a);
  const relErrors = preds.map((p, i) => Math.abs(p - eta[i]) / eta[i]);
  const fractionGood = relErrors.filter(e => e <= pointTolerance).length / relErrors.length;
  const rmse = Math.sqrt(mean(preds.map((p, i) => (log10(p) - log10(eta[i])) ** 2)));

  return (
    fractionGood >= 0.75 &&
    rmse <= 0.10 &&
    max(relErrors) <= 0.40
  );
}

function coordinateRefine(params, gamma, eta, a, lower, upper, iterations = 80) {
  let x = [
    log10(params[0]),
    log10(params[1]),
    params[2],
    log10(params[3])
  ];

  const lo = [log10(lower.eta0), log10(lower.etainf), lower.n, log10(lower.gamma_c)];
  const hi = [log10(upper.eta0), log10(upper.etainf), upper.n, log10(upper.gamma_c)];
  let step = [0.3, 0.3, 0.08, 0.5];

  function toParams(z) {
    return [10 ** z[0], 10 ** z[1], z[2], 10 ** z[3]];
  }

  let bestScore = rmseLogForParams(gamma, eta, toParams(x), a);

  for (let iter = 0; iter < iterations; iter++) {
    let improved = false;

    for (let j = 0; j < 4; j++) {
      for (const dir of [-1, 1]) {
        const trial = [...x];
        trial[j] = Math.min(hi[j], Math.max(lo[j], trial[j] + dir * step[j]));
        const score = rmseLogForParams(gamma, eta, toParams(trial), a);

        if (score < bestScore) {
          x = trial;
          bestScore = score;
          improved = true;
        }
      }
    }

    if (!improved) {
      step = step.map(s => s * 0.65);
      if (max(step) < 1e-4) break;
    }
  }

  return toParams(x);
}

function isDuplicate(candidate, fits, tolerance = 0.08) {
  const c = [
    log10(candidate.eta0),
    log10(candidate.etainf),
    candidate.n,
    log10(candidate.gamma_c)
  ];

  for (const fit of fits) {
    if (fit.gamma_c == null) continue;

    const f = [
      log10(fit.eta0),
      log10(fit.etainf),
      fit.n,
      log10(fit.gamma_c)
    ];

    const dist = Math.sqrt(c.reduce((s, v, i) => s + (v - f[i]) ** 2, 0));
    if (dist < tolerance) return true;
  }
  return false;
}

function deduplicateFits(fits, maxAccepted, tolerance = 0.08) {
  const kept = [];
  for (const fit of fits) {
    if (kept.length >= maxAccepted) break;
    if (!isDuplicate(fit, kept, tolerance)) kept.push(fit);
  }
  return kept;
}

function diagnoseFit(gamma, eta, fits) {
  const warnings = [];
  const shearSpan = log10(max(gamma)) - log10(min(gamma));
  const viscosityDropFraction = (max(eta) - min(eta)) / max(eta);

  const gammaCValues = fits.filter(f => f.gamma_c != null).map(f => f.gamma_c);
  const eta0Values = fits.map(f => f.eta0);
  const etainfValues = fits.map(f => f.etainf);

  const fractionCutoffInside = gammaCValues.length
    ? gammaCValues.filter(g => g >= min(gamma) && g <= max(gamma)).length / gammaCValues.length
    : null;

  if (shearSpan < 2) warnings.push("Narrow shear-rate coverage.");
  if (viscosityDropFraction < 0.15) warnings.push("Weak shear-thinning signal.");
  if (fractionCutoffInside !== null && fractionCutoffInside < 0.5) warnings.push("Transition region poorly constrained.");
  if (fits.length < 10) warnings.push("Few distinct plausible fits were found.");

  const eta0Ratio = percentile(eta0Values, 95) / percentile(eta0Values, 5);
  const etainfRatio = percentile(etainfValues, 95) / percentile(etainfValues, 5);

  if (eta0Ratio > 5) warnings.push("Low-shear plateau viscosity is poorly constrained.");
  if (etainfRatio > 5) warnings.push("High-shear limiting viscosity is poorly constrained.");

  let confidence = "Good";
  if (warnings.length > 0 && warnings.length <= 2) confidence = "Moderate";
  if (warnings.length > 2) confidence = "Weak";

  return {
    confidence,
    model_type: "Carreau-Yasuda",
    warnings,
    shear_span_decades: shearSpan,
    viscosity_drop_fraction: viscosityDropFraction,
    fraction_cutoff_inside: fractionCutoffInside,
    eta0_95_to_5_ratio: eta0Ratio,
    etainf_95_to_5_ratio: etainfRatio
  };
}

function fitCarreauYasudaEnsemble(shearRates, viscosities, options = {}) {
  const a = options.a ?? 2.0;
  const nStarts = options.n_starts ?? 40;
  const nRandomCandidates = options.n_random_candidates ?? 3000;
  const maxAccepted = options.max_accepted ?? 200;
  const pointTolerance = options.point_tolerance ?? 0.25;
  const randomSeed = options.random_seed ?? 1;
  const eta0EtainfRelativeThreshold = options.eta0_etainf_relative_threshold ?? 0.08;

  if (shearRates.length < 2) throw new Error("At least 2 shear-rate measurements are required.");
  if (shearRates.length !== viscosities.length) throw new Error("Shear rates and viscosities must have the same length.");
  if (shearRates.some(x => x <= 0) || viscosities.some(x => x <= 0)) {
    throw new Error("All shear rates and viscosities must be positive.");
  }

  // Rough-fit padding: the solver needs >= 4 points, so if the user supplies
  // only 2 or 3 we cycle through their points to fill out to 4. The fit is
  // then under-constrained on purpose and will report low confidence.
  if (shearRates.length < 4) {
    const n0 = shearRates.length;
    const padG = shearRates.slice();
    const padE = viscosities.slice();
    for (let i = 0; padG.length < 4; i++) {
      padG.push(shearRates[i % n0]);
      padE.push(viscosities[i % n0]);
    }
    shearRates = padG;
    viscosities = padE;
  }

  const pairs = shearRates.map((g, i) => [Number(g), Number(viscosities[i])]).sort((a, b) => a[0] - b[0]);
  const gamma = pairs.map(p => p[0]);
  const eta = pairs.map(p => p[1]);

  const precheck = robustNewtonianPrecheck(gamma, eta, a);
  if (precheck) return precheck;

  const rng = seededRandom(randomSeed);
  const etaMin = min(eta);
  const etaMax = max(eta);

  const lower = {
    eta0: etaMax * 0.8,
    etainf: etaMin * 0.001,
    n: 0.2,
    gamma_c: 1.0
  };

  const upper = {
    eta0: etaMax * 200.0,
    etainf: etaMin * 1.2,
    n: 1.0,
    gamma_c: 1e8
  };

  const starts = [
    [etaMax * 1.1, etaMin * 0.8, 0.8, median(gamma)],
    [etaMax * 1.5, etaMin * 0.5, 0.6, median(gamma)],
    [etaMax * 3.0, etaMin * 0.2, 0.4, min(gamma) * 10],
    [etaMax * 10.0, etaMin * 0.05, 0.25, max(gamma) * 10],
    [etaMax, etaMin, 1.0, median(gamma)]
  ];

  while (starts.length < nStarts) {
    starts.push([
      randLogUniform(rng, lower.eta0, upper.eta0),
      randLogUniform(rng, lower.etainf, upper.etainf),
      randUniform(rng, lower.n, upper.n),
      randLogUniform(rng, lower.gamma_c, upper.gamma_c)
    ]);
  }

  const allFits = [];

  // Optimized/refined candidates
  for (const start of starts) {
    const refined = coordinateRefine(start, gamma, eta, a, lower, upper, 80);
    const fit = makeFit(refined, gamma, eta, a);
    if (fit && fitsPoints(fit, gamma, eta, a, pointTolerance)) allFits.push(fit);
  }

  // Random exploratory candidates
  for (let i = 0; i < nRandomCandidates; i++) {
    const params = [
      randLogUniform(rng, lower.eta0, upper.eta0),
      randLogUniform(rng, lower.etainf, upper.etainf),
      randUniform(rng, lower.n, upper.n),
      randLogUniform(rng, lower.gamma_c, upper.gamma_c)
    ];

    const fit = makeFit(params, gamma, eta, a);
    if (fit && fitsPoints(fit, gamma, eta, a, pointTolerance)) allFits.push(fit);
  }

  if (allFits.length === 0) {
    throw new Error("No valid fits found. Try increasing point_tolerance or n_random_candidates.");
  }

  allFits.sort((a, b) => a.rmse_log - b.rmse_log);
  const best = allFits[0];

  // Only collapse to a Newtonian label when the data actually justifies it:
  // the run must span >= 3 decades of shear AND reach high shear (>= 1e5 s^-1).
  // Without that coverage we keep the Carreau-Yasuda fit rather than ASSUME Newtonian.
  const newtShearSpan = log10(max(gamma)) - log10(min(gamma));
  const newtReachesHighShear = max(gamma) >= 1e5;
  const newtBroad = newtShearSpan >= 3.0;
  const etaGap = Math.abs(best.eta0 - best.etainf) / Math.max(best.eta0, best.etainf);
  if (etaGap <= eta0EtainfRelativeThreshold && newtBroad && newtReachesHighShear) {
    return newtonianResult(
      gamma,
      eta,
      a,
      "Newtonian behavior detected: fitted eta0 and etainf are practically indistinguishable across at least 3 decades of shear with coverage above 1e5 s⁻¹, so n and &gamma;<sub>c</sub> are not meaningful."
    );
  }

  const fits = deduplicateFits(allFits, maxAccepted, 0.08);
  const gLow = Math.max(1, Math.min(...shearRates) / 3);
  const gammaGrid = logspace(Math.log10(gLow), Math.log10(20e6), 300);
  const predictionMatrix = fits.map(f => predictArray(gammaGrid, f, a));

  const lowerBand = [];
  const medianBand = [];
  const upperBand = [];

  for (let j = 0; j < gammaGrid.length; j++) {
    const values = predictionMatrix.map(row => row[j]);
    lowerBand.push(percentile(values, 5));
    medianBand.push(percentile(values, 50));
    upperBand.push(percentile(values, 95));
  }

  return {
    best_fit: best,
    accepted_fits: fits,
    n_accepted_fits: fits.length,
    gamma_grid: gammaGrid,
    lower_band: lowerBand,
    median_band: medianBand,
    upper_band: upperBand,
    measured_shear_rates: gamma,
    measured_viscosities: eta,
    diagnostics: diagnoseFit(gamma, eta, fits)
  };
}

function median(arr) {
  return percentile(arr, 50);
}

// Browser global
window.fitCarreauYasudaEnsemble = fitCarreauYasudaEnsemble;
window.carreauYasuda = carreauYasuda;

function _logInterp(grid, band, x) {
  const n = grid.length;
  if (x <= grid[0]) return band[0];
  if (x >= grid[n - 1]) return band[n - 1];
  const lx = Math.log(x);
  for (let i = 1; i < n; i++) {
    if (grid[i] >= x) {
      const l0 = Math.log(grid[i - 1]);
      const l1 = Math.log(grid[i]);
      const w = (lx - l0) / (l1 - l0);
      // interpolate in log space for smooth positive values
      return Math.exp(Math.log(band[i - 1]) * (1 - w) + Math.log(band[i]) * w);
    }
  }
  return band[n - 1];
}

/*
  Returns the viscosity (and 5/50/95 ensemble band values) at a given
  shear rate for a fit object returned by fitCarreauYasudaEnsemble.
  All viscosities are in the same units the data were supplied in (cP).
*/
function viscosityAtShear(result, gamma, a = 2.0) {
  const fit = result.best_fit;
  let best;
  if (fit.model_type === "Newtonian") {
    best = fit.eta0;
  } else {
    best = carreauYasuda(gamma, fit.eta0, fit.etainf, fit.n, fit.gamma_c, a);
  }
  return {
    best: best,
    lower: _logInterp(result.gamma_grid, result.lower_band, gamma),
    median: _logInterp(result.gamma_grid, result.median_band, gamma),
    upper: _logInterp(result.gamma_grid, result.upper_band, gamma)
  };
}

window.viscosityAtShear = viscosityAtShear;