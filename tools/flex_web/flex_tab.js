/* "Scoring" tab of the flex maps (US map, city science maps): the normal map (same page, tiles and tabs) whose scoring tab (in place of the map's own
 * "Scoring funcs") recomputes the level of service in the browser with any scoring variant (fitted presets, or curves edited by hand) and recolours the H3 cells.
 *
 * Data: one folder per area (a US state, a city; transitlos.flex.export) under FLEX_CFG.base: station profiles (stop_*.bin), the
 * stations of every cell (cell_station.bin / cell_dist.bin), cell centres (cell_geom.bin) and counts (cell_counts.bin, population first).
 * Scoring: FlexScore (score.js, the same formulas as transitlos.flex.scoring). Cells: the area's own resolution R (meta.cell_res: 10 in the US,
 * 11 in the cities; res 8 for far cells); the map's other resolutions: finer cells take their res-R / res-8 parent's value, coarser ones the
 * population-weighted mean of the cells inside.
 * Only the cells drawn on screen are recoloured (after every move), through the map's own `access_override` feature-state.
 */
(function () {
  const CFG = window.FLEX_CFG || { base: '/us_transit_los_flex/', presets: 'flex_presets.json', maxStates: 3, minZoom: 6 };
  const STOPS = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'];
  const DIV = ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b'];
  let INDEX = null, PRESETS = [], params = null, defaults = null, mode = 'score', states = {}, order = [], painted = new Map(), busy = false, statusEl = null;
  const $ = (id) => document.getElementById(id);
  function hex2rgb(h) { h = h.replace('#', ''); return [0, 2, 4].map((i) => parseInt(h.substr(i, 2), 16)); }
  function ramp(stops, t) { t = Math.max(0, Math.min(1, isFinite(t) ? t : 0)); const x = t * (stops.length - 1), i = Math.min(Math.floor(x), stops.length - 2), f = x - i;
    const a = hex2rgb(stops[i]), b = hex2rgb(stops[i + 1]); return 'rgb(' + [0, 1, 2].map((k) => Math.round(a[k] + (b[k] - a[k]) * f)).join(',') + ')'; }
  const colour = (v, d) => mode === 'diff' ? ramp(DIV, 0.5 - Math.max(-30, Math.min(30, v - d)) / 60) : ramp(STOPS, v / 100);
  async function bin(path, T) { const r = await fetch(path); if (!r.ok) throw new Error(path + ' ' + r.status); return new T(await r.arrayBuffer()); }
  function status(t) { if (statusEl) statusEl.textContent = t; }

  async function loadState(st) {
    const p = CFG.base + st.path, meta = await (await fetch(p + 'meta.json')).json();
    const [md, hh, vv, np, cst, cdist, geom, counts] = await Promise.all([bin(p + 'stop_mode.bin', Int8Array), bin(p + 'stop_h.bin', Float32Array), bin(p + 'stop_v.bin', Float32Array),
      bin(p + 'stop_np.bin', Int32Array), bin(p + 'cell_station.bin', Int32Array), bin(p + 'cell_dist.bin', Uint16Array), bin(p + 'cell_geom.bin', Float32Array),
      meta.count_columns && meta.count_columns.length ? bin(p + 'cell_counts.bin', Float32Array) : Promise.resolve(new Float32Array(0))]);
    const nc = meta.nc, ncol = (meta.count_columns || []).length, ids = new Array(nc), res = new Int8Array(nc), pop = new Float32Array(nc);
    const cx = new Float32Array(nc), cy = new Float32Array(nc);   // cell centres (the map editor places new stations' cells with them)
    const R = meta.cell_res || 10;
    for (let c = 0; c < nc; c++) {   // H3 id from the cell centre; resolution from the centre-vertex distance (res 10 ~ 75 m, res 11 ~ 28 m, res 8 ~ 530 m)
      const lon = geom[c * 14], lat = geom[c * 14 + 1], dx = (geom[c * 14 + 2] - lon) * Math.cos(lat * Math.PI / 180) * 111320, dy = (geom[c * 14 + 3] - lat) * 110574;
      res[c] = Math.hypot(dx, dy) > 200 ? 8 : R; ids[c] = h3.latLngToCell(lat, lon, res[c]); pop[c] = ncol ? counts[c * ncol] : 1; cx[c] = lon; cy[c] = lat;
    }
    const D = { meta, R, md, hh, vv, np, cst, cdist, nc, ids, res, pop, cx, cy, path: p, S: meta.S, P: meta.P, K: meta.K, ss: new Float32Array(meta.S * meta.P), los: new Float32Array(nc), def: new Float32Array(nc) };
    if (meta.bg_ids) {   // census units under the cells (block group, tract; counties and the state from the block-group ids)
      const [cbg, ctr] = await Promise.all([bin(p + 'cell_bg.bin', Int32Array), bin(p + 'cell_tract.bin', Int32Array)]);
      D.cbg = cbg; D.ctr = ctr; D.bgIds = meta.bg_ids; D.trIds = meta.tract_ids;
    }
    if (!defaults) { defaults = JSON.parse(JSON.stringify(meta.params)); if (!params) params = JSON.parse(JSON.stringify(defaults)); buildEditor(); }
    score(D, defaults, D.def);
    return D;
  }
  function score(D, p, out) {
    const S = FlexScore.prepare(p);
    FlexScore.stopScores(S, D.md, D.hh, D.vv, D.S * D.P, D.ss);
    FlexScore.cellScores(S, D.cst, D.cdist, D.nc, D.K, D.P, D.np, D.md, D.ss, out, null);
    if (out === D.los && D.edit) D.edit.afterCells(S, out);   // the map editor's new stations
  }
  /* per area: value of every map cell (res r) -> {v, d}: res >= R from the res-R (else res-8) parent, coarser from the population-weighted
   * mean of the flex cells inside. Rebuilt after every recompute. */
  function aggregate(D) {
    D.vR = new Map(); D.v8 = new Map(); D.agg = {};
    for (let c = 0; c < D.nc; c++) (D.res[c] === D.R ? D.vR : D.v8).set(D.ids[c], c);
    for (let r = 3; r < D.R; r++) {
      const acc = new Map();
      for (let c = 0; c < D.nc; c++) {
        if (D.res[c] < r) continue;
        const k = D.res[c] === r ? D.ids[c] : h3.cellToParent(D.ids[c], r), w = D.pop[c] > 0 ? D.pop[c] : 1e-6;
        const a = acc.get(k) || [0, 0, 0]; a[0] += w * D.los[c]; a[1] += w * D.def[c]; a[2] += w; acc.set(k, a);
      }
      D.agg[r] = acc;
    }
    D.census = {};
    if (D.cbg) {
      const add = (lvl, k, c) => { if (!k) return; const m = D.census[lvl] || (D.census[lvl] = new Map()); const a = m.get(k) || [0, 0, 0], w = D.pop[c] > 0 ? D.pop[c] : 1e-6;
        a[0] += w * D.los[c]; a[1] += w * D.def[c]; a[2] += w; m.set(k, a); };
      for (let c = 0; c < D.nc; c++) {
        const g = D.cbg[c] >= 0 ? D.bgIds[D.cbg[c]] : null, t = D.ctr[c] >= 0 ? D.trIds[D.ctr[c]] : null;
        add('blockgroup', g, c); add('tract', t, c); if (g) { add('county', g.slice(0, 5), c); add('state', g.slice(0, 2), c); }
      }
    }
  }
  function censusValue(level, id) {
    for (const name of order) {
      const D = states[name]; const m = D && D.census && D.census[level]; const a = m && m.get(id);
      if (a && a[2] > 0) return [a[0] / a[2], a[1] / a[2]];
    }
    return null;
  }
  function valueOf(id, r) {
    for (const name of order) {
      const D = states[name]; if (!D || !D.vR) continue;
      if (r >= D.R) { let c = D.vR.get(r === D.R ? id : h3.cellToParent(id, D.R)); if (c === undefined) c = D.v8.get(h3.cellToParent(id, 8)); if (c !== undefined) return [D.los[c], D.def[c]]; continue; }
      const a = D.agg[r] && D.agg[r].get(id);
      if (a && a[2] > 0) return [a[0] / a[2], a[1] / a[2]];
      if (r > 8) { const c = D.v8.get(h3.cellToParent(id, 8)); if (c !== undefined) return [D.los[c], D.def[c]]; }
    }
    return null;
  }
  function cellLayers(m) { return (m.getStyle().layers || []).filter((l) => /^(hexagons|circles):h3_\d+|^census:census_(blockgroup|tract|county|state)/.test(l.source || '') && (l.type === 'fill' || l.type === 'circle')).map((l) => l.id); }
  function clearPaint(m) { painted.forEach((v, k) => { const [src, sl, id] = k.split('\u0001'); try { m.removeFeatureState({ source: src, sourceLayer: sl, id }, 'access_override'); } catch (e) {} }); painted.clear(); }
  function paint() {
    const m = window.__mainMap || window.__map; if (!m) return;
    if (mode === 'map') { clearPaint(m); return; }
    const layers = cellLayers(m); if (!layers.length) return;
    let n = 0;
    for (const f of m.queryRenderedFeatures({ layers })) {
      if (f.id === undefined || f.id === null) continue;
      const cm = /^census:census_(blockgroup|tract|county|state)/.exec(f.source);
      const val = cm ? censusValue(cm[1], String(f.id)) : valueOf(String(f.id), +(/h3_(\d+)/.exec(f.source) || [])[1]);
      if (!val) continue;
      const k = f.source + '\u0001' + (f.sourceLayer || '') + '\u0001' + f.id, col = colour(val[0], val[1]);
      if (painted.get(k) === col) continue;
      m.setFeatureState({ source: f.source, sourceLayer: f.sourceLayer, id: f.id }, { access_override: col }); painted.set(k, col); n++;
    }
    if (window.__flexRepaintBuildings) window.__flexRepaintBuildings();
    if (order.length) status('Recomputed: ' + order.join(', ') + ' · ' + painted.size + ' cells recoloured');   // every repaint (tiles arrive after the data)
    return n;
  }
  window.__flexCellColour = (id, r) => { if (mode === 'map') return null; const v = valueOf(String(id), r); return v ? colour(v[0], v[1]) : null; };   // the cell's colour, for the buildings
  async function ensureStates() {
    const m = window.__mainMap || window.__map; if (!m || !INDEX || busy) return;
    if (m.getZoom() < CFG.minZoom) { status('Zoom in (zoom ' + CFG.minZoom + '+) to recompute the cells.'); return; }
    const b = m.getBounds(), want = INDEX.states.filter((s) => !(s.bbox[0] > b.getEast() || s.bbox[2] < b.getWest() || s.bbox[1] > b.getNorth() || s.bbox[3] < b.getSouth())).map((s) => s.name);
    const missing = want.filter((n) => !states[n]).slice(0, CFG.maxStates);
    if (missing.length) {
      busy = true; status('Loading ' + missing.join(', ') + '…');
      try {
        for (const n of missing) { const st = INDEX.states.find((s) => s.name === n); states[n] = await loadState(st); order.push(n); recomputeOne(states[n]); }
        while (order.length > CFG.maxStates) { const drop = order.find((n) => want.indexOf(n) < 0); if (!drop) break; order.splice(order.indexOf(drop), 1); delete states[drop]; }
      } catch (e) { status('Could not load the flexible-scoring data: ' + e.message); busy = false; return; }
      busy = false;
    }
    paint();
  }
  function recomputeOne(D) { const t = performance.now(); score(D, params, D.los); aggregate(D); D.ms = performance.now() - t; }
  let timer = 0;
  function recompute() { clearTimeout(timer); timer = setTimeout(() => { order.forEach((n) => recomputeOne(states[n])); painted.clear(); paint(); }, 150); }
  function buildEditor() {
    const el = $('flexEditors'); if (!el || !params || !window.FlexEditor) return;
    FlexEditor.build(el, params, () => { $('flexPreset').value = 'custom'; recompute(); });
  }
  function addStyle() {
    if (document.getElementById('flexTabStyle')) return;
    const st = document.createElement('style'); st.id = 'flexTabStyle';
    st.textContent = '#statsTab_flex{--bg:#fff;--fg:#222;--mut:#666;--line:#ddd;--card:#f7f7f7;--acc:#2a6fdb}' +
      '#statsTab_flex .ed{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:7px 8px;margin:8px 0} #statsTab_flex .edt{font-weight:600;margin-bottom:3px}' +
      '#statsTab_flex .sr{display:block;color:var(--mut);margin:3px 0 0;font-size:12px} #statsTab_flex .hint{color:var(--mut);font-size:11px;margin-top:4px}' +
      '#statsTab_flex .srow{display:flex;align-items:center;justify-content:space-between;gap:6px} #statsTab_flex .sname{flex:1} #statsTab_flex .snum{width:64px;font-size:12px;padding:1px 3px}' +
      '#statsTab_flex input[type=range]{width:100%;margin:0 0 4px} #statsTab_flex select{width:100%;margin:2px 0}' +
      '#statsTab_flex .plot{display:block;margin-top:4px;max-width:100%} #statsTab_flex .pbg{fill:var(--bg);stroke:var(--line)} #statsTab_flex .grid{stroke:var(--line);stroke-dasharray:2 3}' +
      '#statsTab_flex .tick{fill:var(--mut);font-size:9px} #statsTab_flex .curve{stroke:var(--acc);stroke-width:2;fill:none} #statsTab_flex .refcurve{stroke:var(--mut);stroke-width:1;stroke-dasharray:3 3;opacity:.7;fill:none}' +
      '#statsTab_flex .pt{fill:var(--acc);stroke:var(--bg);stroke-width:1.5;cursor:grab}' +
      '.flexHide{display:none !important} ' +
      'html.mapdark #statsTab_flex{--bg:#171717;--fg:#e6e6e6;--mut:#999;--line:#333;--card:#1f1f1f;--acc:#6aa0ff}';
    document.head.appendChild(st);
  }
  function addTab() {
    addStyle();
    const tabs = document.querySelector('#statsPanel .statsTabBtn'); if (!tabs) return false;
    const bar = tabs.parentNode; if (document.getElementById('statsTab_flex')) return true;
    const btn = document.createElement('button'); btn.className = 'statsTabBtn'; btn.setAttribute('data-tab', 'flex'); btn.textContent = 'Scoring';
    // one scoring tab: the flex options take the place of the map's own scoring-functions tab
    const own = bar.querySelector('.statsTabBtn[data-tab="discretization"]');
    if (own) { bar.insertBefore(btn, own); own.remove(); const op = document.getElementById('statsTab_discretization'); if (op) op.style.display = 'none'; } else bar.appendChild(btn);
    const panel = document.createElement('div'); panel.id = 'statsTab_flex'; panel.className = 'statsTabPanel'; panel.style.display = 'none';
    panel.innerHTML = '<div style="font:12px/1.4 sans-serif;color:#333;">' +
      '<div style="display:flex;gap:6px;margin-bottom:6px;"><label style="flex:2;">Scoring<select id="flexPreset" style="width:100%;"></select></label>' +
      '<label style="flex:1;">Colour<select id="flexMode" style="width:100%;"><option value="score">Recomputed score</option><option value="diff">Change vs default</option><option value="map">Published map</option></select></label></div>' +
      '<div id="flexLegend" style="display:none;margin-bottom:6px;"><div style="height:10px;border-radius:2px;background:linear-gradient(to right,#b2182b,#ef8a62,#fddbc7,#f7f7f7,#d1e5f0,#67a9cf,#2166ac);"></div>' +
      '<div style="display:flex;justify-content:space-between;color:#666;font-size:11px;"><span>&minus;30</span><span>0</span><span>+30</span></div></div>' +
      '<div id="flexStatus" style="color:#666;margin-bottom:6px;"></div><div id="flexEditors"></div></div>';
    bar.parentNode.appendChild(panel);
    // the census controls above the tabs (level, compare with, extended census) do not apply to scoring: hidden while it is open
    bar.addEventListener('click', (e) => { const t = e.target.closest('.statsTabBtn'); if (!t) return;
      ['scopeBar', 'extCensusRow', 'statsAreaBtnRow'].forEach((id) => { const el = $(id); if (el) el.classList.toggle('flexHide', t.dataset.tab === 'flex'); }); });
    btn.addEventListener('click', () => {
      window.__statsTab = 'flex';
      document.querySelectorAll('.statsTabBtn').forEach((b) => b.classList.remove('active')); btn.classList.add('active');
      document.querySelectorAll('.statsTabPanel').forEach((p) => { p.style.display = 'none'; }); panel.style.display = 'block';
      ensureStates();
    });
    statusEl = $('flexStatus');
    const sel = $('flexPreset');
    sel.innerHTML = '<option value="custom" hidden>Custom (edited)</option><option value="default">Default score (published map)</option>' + PRESETS.map((p, i) => '<option value="p' + i + '">' + p.name + '</option>').join('');
    sel.value = 'default';
    sel.onchange = () => { params = JSON.parse(JSON.stringify(sel.value === 'default' ? defaults : PRESETS[+sel.value.slice(1)].params)); buildEditor(); recompute(); };
    $('flexMode').onchange = (e) => { mode = e.target.value; $('flexLegend').style.display = mode === 'diff' ? 'block' : 'none'; painted.clear(); const m = window.__mainMap || window.__map; if (m) clearPaint(m); paint(); };
    return true;
  }
  async function init() {
    try { INDEX = await (await fetch(CFG.base + 'index.json')).json(); } catch (e) { INDEX = null; }
    try { PRESETS = await (await fetch(CFG.presets)).json(); } catch (e) { PRESETS = []; }
    const tryTab = () => { if (!addTab()) setTimeout(tryTab, 500); }; tryTab();
    const hook = () => { const m = window.__mainMap || window.__map; if (!m) { setTimeout(hook, 500); return; }
      m.on('moveend', () => { if (window.__statsTab === 'flex' || painted.size || window.__flexEditOn) ensureStates(); });
      m.on('idle', () => { if (painted.size || window.__statsTab === 'flex' || window.__flexEditOn) paint(); }); };
    hook();
  }
  window.__flexDebug = { states: () => states, valueOf, paint };   // for the page tests
  // for the map editor (flex_edit.js): load the area, recompute and recolour after an edit, the scoring in use, the colour mode
  window.__flexApi = {
    load: async () => { await ensureStates(); return order.length ? states[order[0]] : null; },
    recompute: () => { order.forEach((n) => recomputeOne(states[n])); painted.clear(); paint(); },
    // after an edit: every stop score (cheap), then only the cells that list a changed station or a new station's cell, and the coarser
    // levels updated by the change of those cells
    recomputeEdit: (info) => {
      const D = order.length ? states[order[0]] : null; if (!D || !D.agg) return;
      const S = FlexScore.prepare(params);
      FlexScore.stopScores(S, D.md, D.hh, D.vv, D.S * D.P, D.ss);
      if (!D.st2c) {   // station -> cells listing it (CSR), once
        const n = D.meta.S, cnt = new Int32Array(n + 1);
        for (let i = 0; i < D.cst.length; i++) { const s = D.cst[i]; if (s >= 0 && s < n) cnt[s + 1]++; }
        for (let s = 0; s < n; s++) cnt[s + 1] += cnt[s];
        const pos = cnt.slice(0, n), cells = new Int32Array(cnt[n]);
        for (let i = 0; i < D.cst.length; i++) { const s = D.cst[i]; if (s >= 0 && s < n) cells[pos[s]++] = (i / D.K) | 0; }
        D.st2c = { off: cnt, cells };
      }
      const set = new Set(info.cells || []);
      for (const s of info.stations || []) if (s < D.meta.S) for (let i = D.st2c.off[s]; i < D.st2c.off[s + 1]; i++) set.add(D.st2c.cells[i]);
      const cells = Int32Array.from(set), old = new Float32Array(cells.length);
      cells.forEach((c, i) => { old[i] = D.los[c]; });
      FlexScore.cellScoresAt(S, D.cst, D.cdist, cells, D.K, D.P, D.np, D.md, D.ss, D.los);
      if (D.edit) D.edit.afterCells(S, D.los);
      cells.forEach((c, i) => {
        const d = D.los[c] - old[i]; if (!d) return;
        const w = D.pop[c] > 0 ? D.pop[c] : 1e-6;
        for (let r = 3; r < D.R; r++) { if (D.res[c] < r) continue; const a = D.agg[r].get(D.res[c] === r ? D.ids[c] : h3.cellToParent(D.ids[c], r)); if (a) a[0] += w * d; }
        if (D.cbg) {
          const g = D.cbg[c] >= 0 ? D.bgIds[D.cbg[c]] : null, t = D.ctr[c] >= 0 ? D.trIds[D.ctr[c]] : null;
          const upd = (lvl, k) => { const m = k && D.census[lvl]; const a = m && m.get(k); if (a) a[0] += w * d; };
          upd('blockgroup', g); upd('tract', t); if (g) { upd('county', g.slice(0, 5)); upd('state', g.slice(0, 2)); }
        }
      });
      painted.clear(); paint();
      return cells.length;
    },
    params: () => params, defaults: () => defaults,
    setColour: (m) => { mode = m; const s = $('flexMode'); if (s) s.value = m; const lg = $('flexLegend'); if (lg) lg.style.display = m === 'diff' ? 'block' : 'none';
      painted.clear(); const mm = window.__mainMap || window.__map; if (mm) clearPaint(mm); paint(); },
    colour: () => mode,
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
