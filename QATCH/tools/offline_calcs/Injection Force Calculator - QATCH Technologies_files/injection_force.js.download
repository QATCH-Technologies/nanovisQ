'use strict';

let chart = null;          // Chart.js instance
let lastFit = null;        // last viscosity fit result
let lastGammaEff = null;   // last computed effective shear rate (for chart marker)

/* small DOM helpers */
const $ = id => document.getElementById(id);
const fv = id => { const v = parseFloat($(id).value); return Number.isFinite(v) ? v : 0; };
function showErr(boxId, msg) { const b = $(boxId); b.textContent = 'Error: ' + msg; b.style.display = 'block'; }
function clearErr(boxId) { $(boxId).style.display = 'none'; }
function fmt(v, d = 2) { if (!Number.isFinite(v)) return '-'; return (Math.abs(v) < 1e-12 ? 0 : v).toFixed(d); }
function fmtSci(v) {
  if (!Number.isFinite(v)) return '-';
  if (v === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(v)));
  if (e >= -1 && e < 5) return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  const m = v / Math.pow(10, e);
  return m.toFixed(2) + '\u00d710' + sup(e);
}
function sup(n) {
  const map = { '-': '\u207b', '0': '\u2070', '1': '\u00b9', '2': '\u00b2', '3': '\u00b3', '4': '\u2074',
                '5': '\u2075', '6': '\u2076', '7': '\u2077', '8': '\u2078', '9': '\u2079' };
  return String(n).split('').map(c => map[c] || c).join('');
}

/* ────────────────────────────────────────────────────────────────
   nanovisQ referral link — shown when the force estimate is too
   uncertain to be useful. Swap in the exact product page if desired.
   ──────────────────────────────────────────────────────────────── */
const NANOVIS_URL = 'https://qatchtech.com/';
const NANOVIS_BLOG_URL = 'https://qatchtech.com/blog/f/why-shear-rate-range-matters-for-injectability-prediction';

/* If the predicted plunger-force band spans more than this fraction of
   its central value, the estimate is flagged as too uncertain to use. */
const FORCE_UNCERTAINTY_LIMIT = 0.50;   // 50%

/* Needle gauge → nominal INNER diameter (mm), regular-wall hypodermic.
   Values are diameters; the math halves them to get the lumen radius.
   27 G (0.21 mm ID → 0.105 mm radius) matches the prior default. */
const NEEDLE_ID_MM = {
  18: 0.838, 19: 0.686, 20: 0.603, 21: 0.514, 22: 0.413, 23: 0.337,
  24: 0.311, 25: 0.260, 26: 0.260, 27: 0.210, 28: 0.184, 29: 0.184,
  30: 0.159, 31: 0.133, 32: 0.108, 33: 0.108, 34: 0.0826
};

/* Resolve the needle field to an inner diameter in mm. Reads the gauge
   dropdown; when "Custom…" is selected it reads the custom box, which
   accepts either a gauge number (7-34) or a raw inner diameter in mm. */
function needleDiamMM() {
  const sel = $('needleGauge');
  const selVal = sel ? String(sel.value || '').trim() : '';
  if (!selVal) throw new Error('Select a needle gauge.');

  if (selVal === 'custom') {
    const customVal = String($('needleCustom').value || '').trim();
    if (!customVal) throw new Error(
      needleCustomMode === 'gauge'
        ? 'Enter a gauge number (e.g. 25).'
        : 'Enter an inner diameter in mm (e.g. 0.210).'
    );
    const num = parseFloat(customVal);
    if (!Number.isFinite(num) || num <= 0) throw new Error('Enter a valid positive number.');
    if (needleCustomMode === 'gauge') {
      const g = Math.round(num);
      if (NEEDLE_ID_MM[g] != null) return NEEDLE_ID_MM[g];
      throw new Error('Enter a gauge between 18 and 34.');
    } else {
      if (num >= 5) throw new Error('Inner diameter seems too large — enter in mm (e.g. 0.210 for 27 G).');
      return num;
    }
  }

  // Standard dropdown gauge
  const num = parseFloat(selVal);
  if (!Number.isFinite(num)) throw new Error('Could not read the needle gauge.');
  const g = Math.round(num);
  if (NEEDLE_ID_MM[g] != null) return NEEDLE_ID_MM[g];
  throw new Error('No inner-diameter on file for ' + g + ' G.');
}
/* Show/hide the custom needle box when the dropdown changes. */
function onGaugeChange() {
  const isCustom = $('needleGauge').value === 'custom';
  $('needleCustomWrap').style.display = isCustom ? '' : 'none';
  if (isCustom) {
    needleCustomMode = 'gauge';
    setNeedleCustomMode('gauge');
    const c = $('needleCustom'); if (c) c.focus();
  }
  computeDerived();
}

function makeViscRow(shear = '', visc = '') {
  const row = document.createElement('div');
  row.className = 'vg-row';
  row.innerHTML =
    '<input type="number" class="vg-shear" min="0" step="any" inputmode="decimal" placeholder="e.g. 1000">' +
    '<input type="number" class="vg-visc"  min="0" step="any" inputmode="decimal" placeholder="e.g. 45">' +
    '<button type="button" class="vg-del" title="Remove row" onclick="delViscRow(this)">\u00d7</button>';
  if (shear !== '' && shear != null) row.querySelector('.vg-shear').value = shear;
  if (visc  !== '' && visc  != null) row.querySelector('.vg-visc').value = visc;
  return row;
}

function addViscRow(shear = '', visc = '') {
  const host = $('viscRows');
  if (!host) return;
  const row = makeViscRow(shear, visc);
  host.appendChild(row);
  if (shear === '') row.querySelector('.vg-shear').focus();
}

function delViscRow(btn) {
  const host = $('viscRows');
  const row = btn.closest('.vg-row');
  if (!row || !host) return;
  row.remove();
  if (!host.querySelector('.vg-row')) addViscRow();
}

function setViscRows(pairs) {
  const host = $('viscRows');
  if (!host) return;
  host.innerHTML = '';
  if (!pairs || !pairs.length) { addViscRow(); return; }
  pairs.forEach(([g, e]) => host.appendChild(makeViscRow(g, e)));
}

/* Read the grid into the {shearRates, viscosities} shape the fitter expects. */
function getViscData() {
  const shearRates = [], viscosities = [];
  document.querySelectorAll('#viscRows .vg-row').forEach(row => {
    const g = Number(row.querySelector('.vg-shear').value);
    const e = Number(row.querySelector('.vg-visc').value);
    if (Number.isFinite(g) && Number.isFinite(e) &&
        row.querySelector('.vg-shear').value.trim() !== '' &&
        row.querySelector('.vg-visc').value.trim() !== '') {
      shearRates.push(g);
      viscosities.push(e);
    }
  });
  return { shearRates, viscosities };
}

/* viscosity-mode toggle (Newtonian | Custom fit) */
let viscMode = 'custom';

/* custom needle input mode ('gauge' | 'id') */
let needleCustomMode = 'gauge';
function setNeedleCustomMode(m) {
  needleCustomMode = m;
  $('ncmGauge').classList.toggle('active', m === 'gauge');
  $('ncmId').classList.toggle('active', m === 'id');
  const inp  = $('needleCustom');
  const hint = $('needleCustomHint');
  if (inp) {
    if (m === 'gauge') {
      inp.placeholder = 'e.g. 25';
      inp.min  = '18';
      inp.max  = '34';
      inp.step = '1';
      if (hint) hint.textContent = 'Valid range: 18-34 G';
    } else {
      inp.placeholder = 'e.g. 0.210';
      inp.min  = '0.01';
      inp.max  = '4.99';
      inp.step = 'any';
      if (hint) hint.textContent = 'Inner diameter in mm (0.01 - 4.99)';
    }
    inp.value = '';
    inp.classList.remove('field-invalid');
  }
  computeDerived();
}

function validateNeedleCustom() {
  const inp = $('needleCustom');
  if (!inp) return;
  const val = inp.value.trim();
  if (!val) { inp.classList.remove('field-invalid'); return; }
  const num = parseFloat(val);
  let valid;
  if (needleCustomMode === 'gauge') {
    const g = Math.round(num);
    valid = Number.isFinite(num) && g >= 18 && g <= 34 && NEEDLE_ID_MM[g] != null;
  } else {
    valid = Number.isFinite(num) && num > 0 && num < 5;
  }
  inp.classList.toggle('field-invalid', !valid);
}

function setViscMode(m) {
  viscMode = m;
  $('vmNewt').classList.toggle('active', m === 'newt');
  $('vmCustom').classList.toggle('active', m === 'custom');
  $('newtInputs').style.display = m === 'newt' ? '' : 'none';
  $('customInputs').style.display = m === 'custom' ? '' : 'none';
}

/* injection-spec toggle (Time | Flow Q | Velocity) */
let injMode = 'time';
function setInjMode(m) {
  injMode = m;
  ['imTime', 'imFlow', 'imVel'].forEach(id => $(id).classList.remove('active'));
  ({ time: 'imTime', flow: 'imFlow', velocity: 'imVel' })[m] && $(({ time: 'imTime', flow: 'imFlow', velocity: 'imVel' })[m]).classList.add('active');
  $('timeField').style.display = m === 'time' ? '' : 'none';
  $('flowField').style.display = m === 'flow' ? '' : 'none';
  $('velField').style.display = m === 'velocity' ? '' : 'none';
  computeDerived();
}

/* ── geometry + injection kinematics */
function getSyringe() {
  const Rb = fv('barrelR') / 1000;            // mm -> m
  const Rn = (needleDiamMM() / 2) / 1000;     // gauge inner Ø (mm) -> radius (m)
  const L = fv('needleL') / 1000;             // mm -> m
  const V = fv('injVol') * 1e-6;              // mL -> m^3
  const Abarrel = Math.PI * Rb * Rb;          // m^2

  if (Rb <= 0) throw new Error('Barrel inner radius must be > 0.');
  if (Rn <= 0) throw new Error('Needle inner radius must be > 0.');
  if (L <= 0) throw new Error('Needle length must be > 0.');

  let Q;   // m^3/s
  if (injMode === 'time') {
    const t = fv('injTime');
    if (V <= 0) throw new Error('Injection volume must be > 0 for time-based rate.');
    if (t <= 0) throw new Error('Injection time must be > 0.');
    Q = V / t;
  } else if (injMode === 'flow') {
    const Qml = fv('flowQ');
    if (Qml <= 0) throw new Error('Flow rate Q must be > 0.');
    Q = Qml * 1e-6;                      // mL/s -> m^3/s
  } else { // velocity
    const vmm = fv('plungerV');
    if (vmm <= 0) throw new Error('Plunger velocity must be > 0.');
    Q = (vmm / 1000) * Abarrel;          // mm/s -> m/s × m^2 = m^3/s
  }

  const velocity = Q / Abarrel;          // plunger velocity m/s
  const t = V > 0 ? V / Q : null;        // implied injection time (s)
  return { Rb, Rn, L, V, Q, velocity, t, Abarrel };
}

/* live readout of derived Q / velocity / time chips */
function computeDerived() {
  try {
    const s = getSyringe();
    $('dQ').innerHTML = fmt(s.Q * 1e6, 3) + ' <small>mL/s</small>';
    $('dVel').innerHTML = fmt(s.velocity * 1000, 2) + ' <small>mm/s</small>';
    $('dTime').innerHTML = s.t ? fmt(s.t, 1) + ' <small>s</small>' : '-';
  } catch (e) {
    $('dQ').innerHTML = '-';
    $('dVel').innerHTML = '-';
    $('dTime').innerHTML = '-';
  }
}

/* Fit the viscosity profile */
function fitViscosity() {
  clearErr('viscError');
  try {
    const a = 2.0;   // Yasuda transition width (fixed)
    let result;

    if (viscMode === 'newt') {
      const eta = fv('newtVisc');
      if (eta <= 0) throw new Error('Enter a viscosity greater than 0 cP.');
      const grid = [];
      for (let i = 0; i < 300; i++) grid.push(Math.pow(10, (i / 299) * Math.log10(2e7)));
      const flat = grid.map(() => eta);
      result = {
        best_fit: { eta0: eta, etainf: eta, n: 1.0, gamma_c: null, a, rmse_log: 0, model_type: 'Newtonian' },
        accepted_fits: [], n_accepted_fits: 1,
        gamma_grid: grid, lower_band: flat, median_band: flat, upper_band: flat,
        measured_shear_rates: [], measured_viscosities: [],
        diagnostics: { confidence: 'Good', model_type: 'Newtonian',
          warnings: ['User-specified Newtonian fluid: viscosity is held constant at all shear rates (n = 1).'] }
      };
    } else {
      const { shearRates, viscosities } = getViscData();
      if (shearRates.length < 2) throw new Error('Custom fit needs at least 2 (shear rate, viscosity) rows. With 2-3 points the data is duplicated for a rough, low-confidence fit.');
      result = window.fitCarreauYasudaEnsemble(shearRates, viscosities, {
        a,
        n_random_candidates: 3000,
        n_starts: 40,
        point_tolerance: 0.25,
        max_accepted: 200,
        random_seed: 1
      });
    }

    lastFit = result;
    lastGammaEff = null;
    renderRheology(result);
    drawChart(result, null);
    $('emptyState').style.display = 'none';
    $('rheologyBlock').style.display = 'block';
    $('rheologyBlock').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    return result;
  } catch (e) {
    const msg = e.message && e.message.startsWith('No valid fits found')
      ? 'Nofit is available for this data. Try adding more measurements or a wider shear-rate range.'
      : e.message;
    showErr('viscError', msg);
    return null;
  }
}

function renderRheology(r) {
  const f = r.best_fit, d = r.diagnostics;
  const isNewt = f.model_type === 'Newtonian';
  const confCls = { Good: 'conf-good', Moderate: 'conf-moderate', Weak: 'conf-weak' }[d.confidence] || 'conf-good';

  let metrics;
  if (isNewt) {
    metrics = `
      <div class="metric-card highlight">
        <div class="metric-label">Viscosity η</div>
        <div class="metric-value">${fmt(f.eta0, 2)} <small>cP</small></div>
        <div class="metric-delta neutral">Newtonian)</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">POWER-LAW n</div>
        <div class="metric-value">1.00</div>
        <div class="metric-delta neutral">no shear-thinning</div>
      </div>`;
  } else {
    metrics = `
      <div class="metric-card highlight">
        <div class="metric-label">η<sub>0</sub> (ZERO-SHEAR)</div>
        <div class="metric-value">${fmt(f.eta0, 2)} <small>cP</small></div>
        <div class="metric-delta neutral">low-shear plateau</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">η<sub>∞</sub> (HIGH-SHEAR)</div>
        <div class="metric-value">${fmt(f.etainf, 2)} <small>cP</small></div>
        <div class="metric-delta neutral">limiting plateau</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">POWER-LAW n</div>
        <div class="metric-value">${fmt(f.n, 3)}</div>
        <div class="metric-delta ${f.n < 1 ? 'down' : 'neutral'}">${f.n < 1 ? 'shear-thinning' : 'Newtonian'}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">γ<sub>c</sub> (CRITICAL)</div>
        <div class="metric-value">${fmtSci(f.gamma_c)} <small>s⁻¹</small></div>
        <div class="metric-delta neutral">transition</div>
      </div>`;
  }

  const warnHtml = (d.warnings && d.warnings.length)
    ? `<div class="warn-list">${d.warnings.map(w => `
        <div class="warn-item">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
          <span>${w}</span></div>`).join('')}</div>`
    : `<div class="ok-note">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
        Fit is well constrained across the supplied shear-rate range.</div>`;

  const fitCount = isNewt ? '' :
    `<span class="metric-ci" style="display:block;margin:-8px 2px 14px">Ensemble of <b>${r.n_accepted_fits}</b> plausible fits · best RMSE(log) = <b>${fmt(f.rmse_log, 4)}</b></span>`;

  $('rheologyBlock').innerHTML = `
    <div class="block-head">
      <span class="bh-icon">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M20 8c-3 0-3 6-6 6s-3-4-6-4-3 4-5 4"/></svg>
      </span>
      <span class="bh-title">Viscosity Profile</span>
      <span class="bh-badge ${isNewt ? 'badge-newt' : 'badge-cy'}">${isNewt ? 'Newtonian' : 'Carreau-Yasuda'}</span>
    </div>
    <div class="metric-grid ${isNewt ? '' : 'four'}">${metrics}</div>
    ${fitCount}
    <div class="chart-card">
      <div class="chart-quality">
        <span class="cq-label">Fit Quality</span>
        <span class="bh-badge ${confCls}">${d.confidence}</span>
      </div>
      <div class="chart-wrap"><canvas id="fitChart"></canvas></div>
      <div class="chart-legend">
        <div class="legend-item" style="display:flex;align-items:center;gap:6px"><span class="legend-swatch" style="background:rgba(23,169,212,0.22)"></span>90% ensemble band (5-95%)</div>
        <div class="legend-item" style="display:flex;align-items:center;gap:6px"><span class="legend-line" style="border-color:#0d8ab3"></span>Median fit</div>
        ${r.measured_shear_rates.length ? '<div class="legend-item" style="display:flex;align-items:center;gap:6px"><span class="legend-dot2" style="background:#1a2a3a"></span>Measured data</div>' : ''}
        <div class="legend-item" id="needleLegend" style="display:none;align-items:center;gap:6px"><span class="legend-line" style="border-color:#e8962a;border-top-style:dashed"></span>NEEDLE SHEAR γ<sub>eff</sub></div>
      </div>
    </div>
    ${warnHtml}`;
}

/* Injection force */
function calcForce() {
  clearErr('forceError');
  if (!lastFit) {
    const ok = fitViscosity();
    if (!ok) { showErr('forceError', 'Fix the viscosity inputs above first - the force model needs a viscosity profile.'); return; }
  }
  try {
    const s = getSyringe();
    const phi = (() => { const v = parseFloat($('phi').value); return Number.isFinite(v) && v > 0 ? v : 1; })();
    const fric = (() => { const v = parseFloat($('friction').value); return Number.isFinite(v) && v >= 0 ? v : 1.5; })();

    // Effective needle shear rate uses the FITTED power-law index n (n = 1 for Newtonian)
    const n = lastFit.best_fit.n;
    const a = lastFit.best_fit.a || 2.0;
    const gammaEff = (2 * s.Q / (Math.PI * Math.pow(s.Rn, 3))) * ((3 * n + 1) / (2 * n + 1));
    lastGammaEff = gammaEff;

    // Viscosity (+ 5/50/95 ensemble band) at that shear rate, in cP
    const v = window.viscosityAtShear(lastFit, gammaEff, a);

    // Hagen-Poiseuille hydrodynamic force (η in Pa·s -> N); π cancels
    const Fh = etaCP => 8 * (etaCP * 1e-3) * s.L * s.Q * s.Rb * s.Rb / Math.pow(s.Rn, 4);
    const totalOf = etaCP => phi * Fh(etaCP) + fric;

    const hydroMed = Fh(v.median), hydroMod = phi * hydroMed;
    const Fmed = totalOf(v.median);
    const Flow = totalOf(v.lower);
    const Fhigh = totalOf(v.upper);

    const isNewt = lastFit.best_fit.model_type === 'Newtonian';
    const extrap = lastFit.measured_shear_rates.length
      ? gammaEff > Math.max(...lastFit.measured_shear_rates) * 1.05
      : false;

    renderForce({ s, phi, fric, n, gammaEff, v, hydroMed, hydroMod, Fmed, Flow, Fhigh, isNewt, extrap });
    drawChart(lastFit, gammaEff);   // redraw with the needle-shear marker
    $('forceBlock').style.display = 'block';
    $('forceBlock').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (e) {
    showErr('forceError', e.message);
  }
}

function renderForce(o) {
  // Relative width of the plunger-force confidence band. If it exceeds the
  // limit, the estimate is too uncertain to act on and we steer to nanovisQ.
  const hasBand = Math.abs(o.Fhigh - o.Flow) > 0.01;
  const forceRange = (hasBand && o.Fmed > 0) ? (o.Fhigh - o.Flow) / o.Fmed : 0;
  const cyFitWeak = !o.isNewt && lastFit?.diagnostics?.confidence === 'Weak';
  const tooUncertain = (forceRange > FORCE_UNCERTAINTY_LIMIT) || cyFitWeak;
  const uncertainHtml = tooUncertain ? `
    <div class="uncertainty-warn">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <div class="uw-body">
        <div class="uw-title">Measurement uncertainty is too large</div>
        <div class="uw-text">The viscosity profile is under-constrained, leading to a large injection force range. Expanding shear-rate coverage with direct measurements can substantially improve prediction confidence.</div>
        <a class="uw-link" href="${NANOVIS_BLOG_URL}" target="_blank" rel="noopener noreferrer">Learn more about shear-rate coverage →</a>
      </div>
    </div>` : '';

  const ciVisc = hasBand ?
    `<div class="metric-ci">90% CI <b>${fmt(o.v.lower, 2)}</b> - <b>${fmt(o.v.upper, 2)}</b> cP</div>` : '';
  const ciForce = hasBand ?
    `<div class="metric-ci">90% CI <b>${fmt(o.Flow, 1)}</b> - <b>${fmt(o.Fhigh, 1)}</b> N</div>` : '';

  const extrapNote = o.extrap
    ? `<div class="warn-item" style="margin-bottom:14px">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        <span>The needle shear rate (${fmtSci(o.gammaEff)} s⁻¹) is above your highest measured point, so viscosity here is extrapolated from the fitted model - exactly the high-shear regime this model is designed to project. Treat the band as the confidence range.</span></div>`
    : '';

  $('forceBlock').innerHTML = `
    <div class="block-head">
      <span class="bh-icon">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v6"/><path d="M9 5h6"/><rect x="8" y="8" width="8" height="9" rx="1"/><path d="M10 17v3a2 2 0 002 2 2 2 0 002-2v-3"/></svg>
      </span>
      <span class="bh-title">Injection Force</span>
      <span class="bh-badge badge-cy">φ = ${fmt(o.phi, 2)}</span>
    </div>
    ${extrapNote}
    <div class="metric-grid four">
      <div class="metric-card">
        <div class="metric-label">Needle shear γ<sub>eff</sub></div>
        <div class="metric-value">${fmtSci(o.gammaEff)} <small>s⁻¹</small></div>
        <div class="metric-delta neutral">effective, n = ${fmt(o.n, 2)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">η AT NEEDLE SHEAR</div>
        <div class="metric-value">${fmt(o.v.median, 2)} <small>cP</small></div>
        ${ciVisc || '<div class="metric-delta neutral">from profile</div>'}
      </div>
      <div class="metric-card">
        <div class="metric-label">HYDRODYNAMIC FORCE</div>
        <div class="metric-value">${fmt(o.hydroMod, 1)} <small>N</small></div>
        <div class="metric-delta neutral">φ · Hagen-Poiseuille</div>
      </div>
      <div class="metric-card highlight">
        <div class="metric-label">TOTAL PLUNGER FORCE</div>
        <div class="metric-value">${fmt(o.Fmed, 1)} <small>N</small></div>
        ${ciForce || '<div class="metric-delta neutral">hydro + friction</div>'}
      </div>
    </div>

    ${uncertainHtml}
    ${buildGauge(o.Fmed, o.Flow, o.Fhigh, hasBand)}

    <div class="breakdown">
      <div class="bd-row">
        <span class="bd-name">Hydrodynamic force (η, geometry, flow)</span>
        <span class="bd-val">${fmt(o.hydroMed, 2)} N</span>
      </div>
      <div class="bd-row">
        <span class="bd-name"><span class="op">×</span>Shape / tip correction φ</span>
        <span class="bd-val">${fmt(o.phi, 2)}</span>
      </div>
      <div class="bd-row">
        <span class="bd-name"><span class="op">+</span>Stopper / barrel friction</span>
        <span class="bd-val">${fmt(o.fric, 2)} N</span>
      </div>
      <div class="bd-row total">
        <span class="bd-name">Estimated total injection force</span>
        <span class="bd-val">${fmt(o.Fmed, 2)} N</span>
      </div>
    </div>`;
}

/* force "scale": zoned gauge with CI span + pointer + verdict */
function buildGauge(Fmed, Flow, Fhigh, hasBand) {
  const zones = [
    { to: 10,       cls: 'z-easy', label: 'Low',       col: 'rgba(23,169,212,0.55)' },
    { to: 20,       cls: 'z-mod',  label: 'Moderate',  col: 'rgba(102,187,106,0.55)' },
    { to: 30,       cls: 'z-firm', label: 'High',      col: 'rgba(232,150,42,0.6)' },
    { to: Infinity, cls: 'z-high', label: 'Very high', col: 'rgba(217,79,79,0.62)' }
  ];
  let scaleMax = Math.max(40, Math.ceil((Fhigh * 1.2) / 10) * 10);
  const pct = x => Math.max(0, Math.min(100, (x / scaleMax) * 100));
  let prev = 0, zoneHtml = '';
  for (const z of zones) {
    const top = Math.min(z.to, scaleMax);
    if (top <= prev) continue;
    const w = ((top - prev) / scaleMax) * 100;
    zoneHtml += `<div class="gauge-zone ${z.cls}" style="width:${w}%"></div>`;
    prev = top;
    if (top >= scaleMax) break;
  }

  const ciLeft = pct(Flow), ciW = Math.max(0.5, pct(Fhigh) - pct(Flow));
  const ciSpan = hasBand ?
    `<div class="gauge-ci" style="left:${ciLeft}%;width:${ciW}%"></div>` : '';

  let verdict, vCol;
  if      (Fmed < 10) { verdict = 'low — comfortable for most adults to inject manually';         vCol = 'rgba(23,169,212,1)'; }
  else if (Fmed < 20) { verdict = 'moderate — generally manageable for healthy adults';            vCol = 'rgba(102,187,106,1)'; }
  else if (Fmed < 30) { verdict = 'high — may be difficult for users with reduced dexterity';     vCol = 'rgba(232,150,42,1)'; }
  else                { verdict = 'very high — consider an autoinjector, larger needle, or a lower-viscosity formulation'; vCol = 'rgba(217,79,79,1)'; }

  return `
    <div class="gauge-card">
      <div class="gauge-readout">
        <span class="g-value">${fmt(Fmed, 1)}</span><span class="g-unit">N</span>
        ${hasBand ? `<span class="g-range">${fmt(Flow, 1)} – ${fmt(Fhigh, 1)} N<span>90% confidence</span></span>` : ''}
      </div>
      <div class="gauge-track">
        ${zoneHtml}
        ${ciSpan}
        <div class="gauge-marker" style="left:${pct(Fmed)}%"></div>
      </div>
      <div class="gauge-scale"><span>0 N</span><span>${(scaleMax/2)} N</span><span>${scaleMax} N</span></div>
      <div class="gauge-labels">
        <span class="gauge-tag"><span class="gt-dot" style="background:rgba(23,169,212,0.55)"></span>Low &lt;10 N</span>
        <span class="gauge-tag"><span class="gt-dot" style="background:rgba(102,187,106,0.55)"></span>Moderate 10-20 N</span>
        <span class="gauge-tag"><span class="gt-dot" style="background:rgba(232,150,42,0.6)"></span>High 20-30 N</span>
        <span class="gauge-tag"><span class="gt-dot" style="background:rgba(217,79,79,0.62)"></span>Very high &gt;30 N</span>
      </div>
      <div class="gauge-verdict">
        At <b>${fmt(Fmed, 1)} N</b>, the predicted injection force is <b style="color:${vCol}">${verdict}</b>.
      </div>
    </div>`;
}

/* Chart.js render (log-log; CI band, median, measured, marker) */
function drawChart(result, gammaEff) {
  const canvas = $('fitChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const upper = result.gamma_grid.map((x, i) => ({ x, y: result.upper_band[i] }));
  const lower = result.gamma_grid.map((x, i) => ({ x, y: result.lower_band[i] }));
  const median = result.gamma_grid.map((x, i) => ({ x, y: result.median_band[i] }));
  const measured = result.measured_shear_rates.map((x, i) => ({ x, y: result.measured_viscosities[i] }));

  const allY = [...result.upper_band, ...result.lower_band, ...result.measured_viscosities].filter(v => v > 0);
  const yMin = Math.max(0.05, Math.min(...allY) * 0.6);
  const yMax = Math.max(...allY) * 1.6;

  const allX = [...result.gamma_grid, ...result.measured_shear_rates].filter(v => v > 0);
  const xMin = Math.max(0.5, Math.min(...allX) * 0.8);

  const datasets = [
    { label: 'Upper', data: upper, borderColor: 'rgba(23,169,212,0.0)', borderWidth: 0, pointRadius: 0, fill: false, tension: 0.1 },
    { label: 'CI band', data: lower, borderColor: 'rgba(23,169,212,0.0)', backgroundColor: 'rgba(23,169,212,0.18)', borderWidth: 0, pointRadius: 0, fill: '-1', tension: 0.1 },
    { label: 'Median fit', data: median, borderColor: '#0d8ab3', borderWidth: 2.4, pointRadius: 0, fill: false, tension: 0.1 }
  ];
  if (measured.length) {
    datasets.push({ label: 'Measured', data: measured, type: 'scatter', borderColor: '#1a2a3a', backgroundColor: '#1a2a3a', pointRadius: 4.5, pointHoverRadius: 6 });
  }
  if (gammaEff) {
    datasets.push({ label: 'Needle γ<sub>eff</sub>', data: [{ x: gammaEff, y: yMin }, { x: gammaEff, y: yMax }], borderColor: '#e8962a', borderWidth: 2, borderDash: [6, 4], pointRadius: 0, fill: false });
    const nl = $('needleLegend'); if (nl) nl.style.display = 'flex';
  }

  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false, parsing: false, animation: { duration: 500 },
      interaction: { mode: 'nearest', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(26,42,58,0.92)', padding: 10, cornerRadius: 8, displayColors: false,
          callbacks: {
            title: items => 'γ̇ = ' + Number(items[0].parsed.x).toLocaleString(undefined, { maximumFractionDigits: 0 }) + ' s⁻¹',
            label: item => item.dataset.label + ': ' + Number(item.parsed.y).toFixed(2) + ' cP'
          }
        }
      },
      scales: {
        x: {
          type: 'logarithmic', min: xMin, max: 2e7,
          title: { display: true, text: 'Shear rate  γ̇  (s⁻¹)', color: '#4a6070', font: { family: 'Montserrat', size: 11, weight: '600' } },
          grid: { color: 'rgba(23,169,212,0.08)' }, ticks: { color: '#8aa0b0', font: { family: 'Roboto', size: 10 } }
        },
        y: {
          type: 'logarithmic', min: yMin, max: yMax,
          title: { display: true, text: 'Viscosity  η  (cP)', color: '#4a6070', font: { family: 'Montserrat', size: 11, weight: '600' } },
          grid: { color: 'rgba(23,169,212,0.08)' }, ticks: { color: '#8aa0b0', font: { family: 'Roboto', size: 10 } }
        }
      }
    }
  });
}

/* Info drawer / accordion / info-pops / contact (from Donnan) */
function openDrawer() { $('infoDrawer').classList.add('open'); $('drawerBackdrop').classList.add('open'); }
function closeDrawer() { $('infoDrawer').classList.remove('open'); $('drawerBackdrop').classList.remove('open'); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });

function toggleAccordion(btn) {
  const body = btn.nextElementSibling;
  const open = body.classList.contains('open');
  body.classList.toggle('open', !open);
  btn.classList.toggle('active', !open);
}
function toggleInfo(id) { $(id).classList.toggle('open'); }

function submitContact(e) {
  e.preventDefault();
  const name = $('contactName').value.trim();
  const email = $('contactEmail').value.trim();
  const msg = $('contactMsg').value.trim();
  const subject = encodeURIComponent('QATCH Injection Force Calculator - Inquiry');
  const body = encodeURIComponent('Name: ' + name + '\nEmail: ' + email + '\n\nMessage:\n' + msg);
  window.location.href = 'mailto:info@qatchtech.com?subject=' + subject + '&body=' + body;
  $('contactConfirm').style.display = 'block';
}

/* init */
window.addEventListener('DOMContentLoaded', () => {
  addViscRow();
  addViscRow();
  setViscMode('custom');
  setInjMode('time');
  ['barrelR', 'injVol', 'injTime', 'flowQ', 'plungerV'].forEach(id => {
    const el = $(id); if (el) el.addEventListener('input', computeDerived);
  });
  const ncInp = $('needleCustom');
  if (ncInp) ncInp.addEventListener('input', () => { validateNeedleCustom(); computeDerived(); });
  computeDerived();
});

function clearViscRows() {
  const host = $('viscRows');
  if (!host) return;
  host.innerHTML = '';
  addViscRow();
  addViscRow();
  lastFit = null;
}