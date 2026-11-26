/* Map editor of the flex maps: the map's "Draw a new line" button becomes "Edit". In edit mode routes and stations are drawn from the flex data
 * (transitlos.flex.export.export_routes) and can be changed; the flexible score is recomputed in the browser (FlexEdit, edit.js) and the cells
 * recoloured by the Scoring tab (flex_tab.js, window.__flexApi).
 *
 *   hover             route / station names; the route under the mouse is highlighted
 *   click a route     its editor (a list to choose from when several routes share the clicked line): mode, headway, speed; add or remove
 *                     stops (then click stations or the map); delete / restore; a new line's name and colour
 *   click a station   its routes as a table (one row per route: mode, headway, speed km/h, score), editable, filterable by mode; the first row
 *                     changes all the shown routes together; a row's x takes the stop off that route; a route's badge opens the route;
 *                     delete / restore the station
 *   New line          click the map to place its stations (within 150 m of a station: that station), shift-click for a bend without a stop;
 *                     Enter finishes, Backspace removes the last point, Esc cancels
 *   Ctrl+Z            undo; edits are kept in the browser (per map) and can be exported / imported as a file
 * Every change is a factor on whole routes (edit.js). Deleted routes and stations stay on the map, faded, to be restored.
 */
(function () {
  const MODES = ['bus', 'tram', 'rail'], MODE_COL = ['#8b0000', '#b38f00', '#006400'], WIDTH = [2, 3.2, 4.4];
  let map = null, api = null, D = null, E = null, on = false, tool = 'select', toolRoute = -1, popup = null, popupFor = null, draft = null;
  let sel = null, hidden = [], panel = null, tip = null, history = [];
  const $ = (id) => document.getElementById(id);
  const fmt = (x, d = 1) => (x == null || !isFinite(x) ? '' : (+x).toFixed(d));
  const esc = (t) => String(t == null ? '' : t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const LAYERS = ['fe-sel', 'fe-hov', 'fe-lines-del', 'fe-lines-case', 'fe-lines', 'fe-lines-add', 'fe-lines-hit', 'fe-st-del', 'fe-st', 'fe-st-ring', 'fe-draft-l', 'fe-draft-p'];

  function style() {
    if ($('flexEditStyle')) return;
    const st = document.createElement('style'); st.id = 'flexEditStyle';
    st.textContent = '#flexEditPanel{position:absolute;z-index:1001;background:#fff;border:1px solid #ccc;border-radius:6px;box-shadow:0 2px 10px rgba(0,0,0,.25);' +
      'padding:8px;width:260px;font:12px/1.35 system-ui,-apple-system,Segoe UI,sans-serif;color:#1d2733;max-height:78vh;overflow-y:auto}' +
      '#flexEditPanel .fe-row{display:flex;gap:4px;margin:4px 0;align-items:center} #flexEditPanel button{flex:1;padding:4px 6px;border:1px solid #bbb;border-radius:4px;background:#f6f7f9;cursor:pointer;font-size:12px}' +
      '#flexEditPanel button.on{background:#2b6cb0;color:#fff;border-color:#2b6cb0} #flexEditPanel button:disabled{opacity:.45;cursor:default} #flexEditPanel button.fe-i{flex:none;width:30px}' +
      '#flexEditPanel label{display:flex;justify-content:space-between;align-items:center;gap:6px;margin:3px 0} #flexEditPanel input,#flexEditPanel select{width:120px;font-size:12px}' +
      '#flexEditPanel .fe-t{font-weight:700;margin:8px 0 2px} #flexEditPanel .fe-list>div{display:flex;justify-content:space-between;align-items:center;gap:6px;padding:2px 0;border-bottom:1px solid #eee;cursor:pointer}' +
      '#flexEditPanel .fe-list>div:hover{background:#f3f6fa} .fe-x{flex:none;cursor:pointer;color:#a00;border:0 !important;background:none !important;padding:0 3px !important;font-size:13px}' +
      '.fe-pop{font:12px/1.35 system-ui,-apple-system,Segoe UI,sans-serif;color:#1d2733} .fe-pop table{border-collapse:collapse;margin-top:4px} .fe-pop td,.fe-pop th{padding:2px 4px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap}' +
      '.fe-pop th{font-weight:600;color:#555} .fe-pop td:first-child,.fe-pop th:first-child{text-align:left} .fe-pop input[type=number]{width:54px;font-size:12px} .fe-pop select{font-size:12px}' +
      '.fe-b{display:inline-block;padding:0 5px;border-radius:3px;color:#fff;font-weight:700;font-size:11px;margin-right:4px;cursor:pointer;white-space:nowrap}' +
      '.fe-pop button{font-size:11px;padding:2px 6px;cursor:pointer;margin:4px 4px 0 0} .fe-pop .fe-all td{background:#f3f6fa;font-weight:600} .fe-pop .fe-ed td{background:#fff8e6}' +
      '.fe-pop .fe-off td{color:#999;text-decoration:line-through} .fe-pop .fe-off td:last-child{text-decoration:none} .fe-pick div{cursor:pointer;padding:3px 0;border-bottom:1px solid #eee} .fe-pick div:hover{background:#f3f6fa}' +
      '.fe-head{display:flex;align-items:center;gap:6px;margin:0 18px 2px 0;font-weight:700;font-size:13px} .fe-pop .fe-name{width:120px;font-size:12px}' +
      '#flexEditTip{position:absolute;z-index:1002;pointer-events:none;background:rgba(255,255,255,.95);border:1px solid #ccc;border-radius:4px;padding:2px 5px;font:11px system-ui,sans-serif;color:#1d2733;max-width:260px}' +
      'html.mapdark #flexEditPanel,html.mapdark #flexEditTip{filter:invert(.92) hue-rotate(180deg)}';
    document.head.appendChild(st);
  }

  /* ---------------------------------------------------------------- route look */
  function light(c) {   // a colour too pale to draw on the light basemap
    const m = /^#?([0-9a-f]{6})$/i.exec(c || ''); if (!m) return true;
    const n = parseInt(m[1], 16);
    return 0.2126 * (n >> 16) + 0.7152 * ((n >> 8) & 255) + 0.0722 * (n & 255) > 215;
  }
  function curMode(r) { const e = E.route[r], ri = E.routeInfo(r); return e && e.mode !== null ? e.mode : (ri ? ri.mode : 0); }
  function lineColour(r) {   // the route's colour, else its text colour, else its mode's
    const ri = E.routeInfo(r);
    if (ri && ri.color && !light(ri.color)) return ri.color;
    if (ri && ri.text && !light(ri.text)) return ri.text;
    return MODE_COL[curMode(r)];
  }
  const routeName = (r) => { const ri = E.routeInfo(r); return (ri && (ri.name || ri.long)) || '?'; };
  const badge = (r) => '<span class="fe-b" data-open="' + r + '" style="background:' + lineColour(r) + '">' + esc(routeName(r)) + '</span>';
  const isDel = (r) => { const e = E.route[r]; return !!(e && e.del) || (r >= E.routes.length && E.newRoutes[r - E.routes.length].gone); };
  function modeSelect(val, attrs) { return '<select ' + attrs + '>' + MODES.map((m, i) => '<option value="' + i + '"' + (i === val ? ' selected' : '') + '>' + m + '</option>').join('') + '</select>'; }

  /* ---------------------------------------------------------------- map layers */
  function lineFeatures() {
    const out = [], push = (r, coords, extra) => out.push({ type: 'Feature', geometry: { type: 'LineString', coordinates: coords },
      properties: Object.assign({ r, w: WIDTH[curMode(r)], c: lineColour(r), ed: E.isEdited(r) ? 1 : 0, del: isDel(r) ? 1 : 0, add: 0 }, extra || {}) });
    for (const f of E.lines.features) push(f.properties.r, f.geometry.coordinates);
    E.newRoutes.forEach((nr, k) => { if (!nr.gone && nr.coords.length > 1) push(E.routes.length + k, nr.coords); });
    for (const r in E.route) {   // stops added to a route: a link from its nearest stop
      const e = E.route[r];
      if (!e.add.length || +r >= E.routes.length) continue;
      for (const s of e.add) {
        const p = E.lonlatOf(s); let best = null, bd = Infinity;
        for (const t of E.routes[r].st) { const q = E.lonlatOf(t), d = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2; if (d < bd) { bd = d; best = q; } }
        if (best) push(+r, [best, p], { add: 1 });
      }
    }
    return { type: 'FeatureCollection', features: out };
  }
  function stationFeatures() {
    const out = [], onSel = new Set(), offSel = new Set();
    if (sel && sel.type === 'route') {
      for (const s of E.stationsOf(sel.r)) (E.skips(s, sel.r) ? offSel : onSel).add(s);
    }
    for (let s = 0; s < D.S; s++) {
      const del = E.stDel.has(s);
      if (!del && !D.np[s]) continue;
      if (del && s >= E.S0 && !E.routesAt(s).length) continue;
      let m = 0;
      for (let j = 0; j < D.np[s]; j++) m = Math.max(m, D.md[s * D.P + j]);
      out.push({ type: 'Feature', geometry: { type: 'Point', coordinates: E.lonlatOf(s) },
                 properties: { s, c: MODE_COL[m], del: del ? 1 : 0, k: offSel.has(s) ? 2 : (onSel.has(s) ? 1 : 0), sel: sel && sel.type === 'station' && sel.s === s ? 1 : 0 } });
    }
    return { type: 'FeatureCollection', features: out };
  }
  function draftFeatures() {
    const f = [];
    if (draft && draft.pts.length) {
      if (draft.pts.length > 1) f.push({ type: 'Feature', geometry: { type: 'LineString', coordinates: draft.pts.map((p) => p.ll) }, properties: {} });
      draft.pts.forEach((p) => f.push({ type: 'Feature', geometry: { type: 'Point', coordinates: p.ll }, properties: { stop: p.stop ? 1 : 0 } }));
    }
    return { type: 'FeatureCollection', features: f };
  }
  function addLayers() {
    if (!map.getSource('fe-lines')) {
      map.addSource('fe-lines', { type: 'geojson', data: lineFeatures() });
      map.addSource('fe-st', { type: 'geojson', data: stationFeatures() });
      map.addSource('fe-draft', { type: 'geojson', data: draftFeatures() });
      const live = ['all', ['==', ['get', 'del'], 0], ['==', ['get', 'add'], 0]];
      map.addLayer({ id: 'fe-sel', type: 'line', source: 'fe-lines', filter: ['==', ['get', 'r'], -1], layout: { 'line-cap': 'round', 'line-join': 'round' },
                     paint: { 'line-color': '#ffcc00', 'line-width': ['+', ['get', 'w'], 9], 'line-opacity': 0.75 } });
      map.addLayer({ id: 'fe-hov', type: 'line', source: 'fe-lines', filter: ['==', ['get', 'r'], -1], layout: { 'line-cap': 'round', 'line-join': 'round' },
                     paint: { 'line-color': '#2b6cb0', 'line-width': ['+', ['get', 'w'], 6], 'line-opacity': 0.35 } });
      map.addLayer({ id: 'fe-lines-del', type: 'line', source: 'fe-lines', filter: ['==', ['get', 'del'], 1], paint: { 'line-color': '#888', 'line-width': 1.5, 'line-dasharray': [2, 2], 'line-opacity': 0.7 } });
      map.addLayer({ id: 'fe-lines-case', type: 'line', source: 'fe-lines', filter: live, paint: { 'line-color': '#fff', 'line-width': ['+', ['get', 'w'], 2], 'line-opacity': 0.8 } });
      map.addLayer({ id: 'fe-lines', type: 'line', source: 'fe-lines', filter: live, layout: { 'line-sort-key': ['get', 'w'] },
                     paint: { 'line-color': ['get', 'c'], 'line-width': ['get', 'w'], 'line-opacity': ['case', ['==', ['get', 'ed'], 1], 1, 0.75] } });
      map.addLayer({ id: 'fe-lines-add', type: 'line', source: 'fe-lines', filter: ['all', ['==', ['get', 'add'], 1], ['==', ['get', 'del'], 0]],
                     paint: { 'line-color': ['get', 'c'], 'line-width': ['get', 'w'], 'line-dasharray': [1.5, 1] } });
      map.addLayer({ id: 'fe-lines-hit', type: 'line', source: 'fe-lines', paint: { 'line-color': '#000', 'line-width': 12, 'line-opacity': 0.001 } });
      const rad = ['interpolate', ['linear'], ['zoom'], 9, 2, 13, 4, 16, 7];
      map.addLayer({ id: 'fe-st-del', type: 'circle', source: 'fe-st', filter: ['==', ['get', 'del'], 1],
                     paint: { 'circle-color': 'rgba(0,0,0,0)', 'circle-stroke-color': '#888', 'circle-stroke-width': 1.5, 'circle-radius': rad } });
      map.addLayer({ id: 'fe-st', type: 'circle', source: 'fe-st', filter: ['==', ['get', 'del'], 0],
                     paint: { 'circle-color': ['case', ['==', ['get', 'k'], 2], '#fff', ['get', 'c']], 'circle-stroke-color': ['case', ['==', ['get', 'k'], 2], '#c00', '#fff'],
                              'circle-stroke-width': ['case', ['>', ['get', 'k'], 0], 2, 1.2],
                              'circle-radius': ['interpolate', ['linear'], ['zoom'], 9, ['case', ['>', ['get', 'k'], 0], 4, 2], 13, ['case', ['>', ['get', 'k'], 0], 6, 4], 16, ['case', ['>', ['get', 'k'], 0], 9, 7]] } });
      map.addLayer({ id: 'fe-st-ring', type: 'circle', source: 'fe-st', filter: ['==', ['get', 'sel'], 1],
                     paint: { 'circle-color': 'rgba(0,0,0,0)', 'circle-stroke-color': '#ffcc00', 'circle-stroke-width': 3, 'circle-radius': ['interpolate', ['linear'], ['zoom'], 9, 7, 13, 9, 16, 12] } });
      map.addLayer({ id: 'fe-draft-l', type: 'line', source: 'fe-draft', filter: ['==', ['geometry-type'], 'LineString'], paint: { 'line-color': '#e6194b', 'line-width': 3, 'line-dasharray': [2, 1] } });
      map.addLayer({ id: 'fe-draft-p', type: 'circle', source: 'fe-draft', filter: ['==', ['geometry-type'], 'Point'],
                     paint: { 'circle-color': ['case', ['==', ['get', 'stop'], 1], '#e6194b', '#fff'], 'circle-radius': ['case', ['==', ['get', 'stop'], 1], 5, 3],
                              'circle-stroke-color': '#e6194b', 'circle-stroke-width': 1.5 } });
    }
    for (const id of LAYERS) map.setLayoutProperty(id, 'visibility', 'visible');
  }
  function refreshLayers() {
    if (!map || !map.getSource('fe-lines')) return;
    map.getSource('fe-lines').setData(lineFeatures()); map.getSource('fe-st').setData(stationFeatures()); map.getSource('fe-draft').setData(draftFeatures());
    map.setFilter('fe-sel', ['==', ['get', 'r'], sel && sel.type === 'route' ? sel.r : -1]);
  }
  /* the map's own stops and routes are hidden while editing (the map's zoom logic shows them again: re-hidden after every move) */
  function hideBase() {
    if (!on) return;
    for (const l of (map.getStyle().layers || [])) {
      if (!/^__(routes|stops)_/.test(l.id) || map.getLayoutProperty(l.id, 'visibility') === 'none') continue;
      map.setLayoutProperty(l.id, 'visibility', 'none');
      if (hidden.indexOf(l.id) < 0) hidden.push(l.id);
    }
  }
  function showBase() {
    hidden.forEach((id) => { if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', 'visible'); });
    hidden = [];
    map.fire('zoomend');   // the map's own visibility rules (by zoom, by the layer checkboxes) apply again
  }

  /* ---------------------------------------------------------------- edits, history, saving */
  const storeKey = () => 'flexEdits:' + location.pathname + ':' + D.path;
  function remember() { history.push(JSON.stringify(E.serialize())); if (history.length > 60) history.shift(); }
  async function changed() {
    status('…');
    api.recomputeEdit(await E.apply());
    try { if (E.hasEdits()) localStorage.setItem(storeKey(), JSON.stringify(E.serialize())); else localStorage.removeItem(storeKey()); } catch (e) { /* storage off */ }
    refreshLayers(); renderPanel();
    if (popupFor) await popupFor.render();
    status('');
  }
  async function edit(fn) { remember(); await fn(); await changed(); }   // every change: an undo point, apply, save
  async function undo() {
    if (!history.length) return;
    if (popup) popup.remove();
    sel = null;
    E.restore(JSON.parse(history.pop()));
    await changed();
  }
  function setMode(r, m) { const e = E.edit(r), ri = E.routeInfo(r); e.mode = (ri && m === ri.mode && r < E.routes.length) ? null : m; }
  function routeMedian(r) {
    const hs = [], vs = [];
    for (const s of E.stationsOf(r)) { if (E.skips(s, r) || E.stDel.has(s)) continue; const c = E.stats(s, [r], true); if (c) { if (isFinite(c.h)) hs.push(c.h); if (isFinite(c.v)) vs.push(c.v); } }
    const med = (a) => { if (!a.length) return NaN; a.sort((x, y) => x - y); return a[Math.floor(a.length / 2)]; };
    return { h: med(hs), v: med(vs) };
  }
  function scale(rs, key, s, val) {   // a value typed at a station (or a route's median) -> a factor on each of the routes
    const cur = s == null ? routeMedian(rs[0]) : E.stats(s, rs, true);
    const base = cur && cur[key];
    if (!(val > 0) || !(base > 0)) return;
    const f = val / base;
    rs.forEach((r) => { E.edit(r)[key] *= f; });
  }

  /* ---------------------------------------------------------------- popups */
  function bind(el, handler) {
    el.querySelectorAll('[data-k]').forEach((x) => {
      const ev = x.tagName === 'BUTTON' ? 'click' : 'change';
      x.addEventListener(ev, (e) => { e.stopPropagation(); handler(x.dataset.k, x); });
      if (x.tagName === 'INPUT' && x.type === 'number') x.addEventListener('keydown', (e) => { if (e.key === 'Enter') x.blur(); });
    });
    el.querySelectorAll('[data-open]').forEach((x) => x.addEventListener('click', (e) => { e.stopPropagation(); routePopup(+x.dataset.open, popup.getLngLat()); }));
  }
  async function stationPopup(s, lngLat) {
    sel = { type: 'station', s }; refreshLayers();
    await E.loadStations([s]);
    let filt = -1;
    const view = {
      render: async () => {
        const all = E.routesAt(s), modes = [...new Set(all.map(curMode))].sort();
        const shown = all.filter((r) => filt < 0 || curMode(r) === filt);
        const live = shown.filter((r) => !isDel(r) && !E.skips(s, r));
        const dead = E.stDel.has(s);
        let h = '<div class="fe-pop"><div class="fe-head">' + esc(E.nameOf(s) || 'Station') + '</div>';
        if (modes.length > 1) h += '<select data-k="filt"><option value="-1">all modes</option>' + modes.map((i) => '<option value="' + i + '"' + (i === filt ? ' selected' : '') + '>' + MODES[i] + '</option>').join('') + '</select>';
        if (!dead && shown.length) {
          h += '<table><tr><th></th><th>mode</th><th>headway (min)</th><th>speed (km/h)</th><th>score</th><th></th></tr>';
          if (live.length > 1) {
            const a = E.stats(s, live, true, api.params());
            if (a) h += '<tr class="fe-all"><td>' + (filt < 0 ? 'all' : MODES[filt]) + '</td><td>' + modeSelect(a.mode, 'data-k="mode" data-r="all"') + '</td>' +
              '<td><input type="number" min="0.5" step="0.5" data-k="h" data-r="all" value="' + fmt(a.h) + '"></td><td><input type="number" min="1" step="1" data-k="v" data-r="all" value="' + fmt(a.v) + '"></td>' +
              '<td>' + fmt(a.score) + '</td><td></td></tr>';
          }
          for (const r of shown) {
            const off = isDel(r) || E.skips(s, r), c = E.stats(s, [r], !off, api.params());
            h += '<tr' + (off ? ' class="fe-off"' : (E.isEdited(r) ? ' class="fe-ed"' : '')) + '><td>' + badge(r) + '</td>';
            if (off || !c) h += '<td>' + MODES[curMode(r)] + '</td><td>' + fmt(c && c.h) + '</td><td>' + fmt(c && c.v) + '</td><td></td>';
            else h += '<td>' + modeSelect(c.mode, 'data-k="mode" data-r="' + r + '"') + '</td><td><input type="number" min="0.5" step="0.5" data-k="h" data-r="' + r + '" value="' + fmt(c.h) + '"></td>' +
              '<td><input type="number" min="1" step="1" data-k="v" data-r="' + r + '" value="' + fmt(c.v) + '"></td><td>' + fmt(c.score) + '</td>';
            h += '<td>' + (isDel(r) ? '<button class="fe-x" data-k="undel" data-r="' + r + '" title="restore the route">↺</button>'
              : E.skips(s, r) ? '<button class="fe-x" data-k="unskip" data-r="' + r + '" title="the route stops here again">↺</button>'
              : '<button class="fe-x" data-k="skip" data-r="' + r + '" title="the route no longer stops here">✕</button>') + '</td></tr>';
          }
          h += '</table>';
        }
        h += '<div>' + (dead ? '<button data-k="undelst">Restore station</button>' : '<button data-k="delst">Delete station</button>') + '</div></div>';
        popup.setHTML(h);
        const el = popup.getElement(); if (!el) return;
        bind(el, (k, x) => {
          if (k === 'filt') { filt = +x.value; return view.render(); }
          const rr = x.dataset.r, targets = rr === 'all' ? live : (rr != null ? [+rr] : []);
          edit(() => {
            if (k === 'delst') E.stDel.add(s);
            else if (k === 'undelst') E.stDel.delete(s);
            else if (k === 'skip') skipStop(s, +rr);
            else if (k === 'unskip') { const e = E.edit(+rr); e.skip = e.skip.filter((x2) => x2 !== s); }
            else if (k === 'undel') undelete(+rr);
            else if (k === 'mode') targets.forEach((r) => setMode(r, +x.value));
            else if (k === 'h' || k === 'v') scale(targets, k, s, +x.value);
          });
        });
      },
    };
    openPopup(lngLat, view);
  }
  function skipStop(s, r) {   // r no longer stops at s (a stop added to it is simply taken off again)
    const e = E.edit(r), i = e.add.indexOf(s);
    if (i >= 0) e.add.splice(i, 1);
    else if (r >= E.routes.length) { const nr = E.newRoutes[r - E.routes.length], j = nr.st.indexOf(s); if (j >= 0) nr.st.splice(j, 1); }
    else if (e.skip.indexOf(s) < 0) e.skip.push(s);
  }
  function insertPoint(coords, p) {   // a new line's new stop goes where it lengthens the line least (an end, or between two points)
    const d = (a, c) => Math.hypot((a[0] - c[0]) * Math.cos(a[1] * Math.PI / 180), a[1] - c[1]);
    let at = coords.length, best = coords.length ? d(coords[coords.length - 1], p) : 0;
    if (coords.length && d(coords[0], p) < best) { best = d(coords[0], p); at = 0; }
    for (let i = 0; i + 1 < coords.length; i++) { const x = d(coords[i], p) + d(p, coords[i + 1]) - d(coords[i], coords[i + 1]); if (x < best) { best = x; at = i + 1; } }
    coords.splice(at, 0, p);
  }
  function undelete(r) { if (r >= E.routes.length) E.newRoutes[r - E.routes.length].gone = false; E.edit(r).del = false; }
  async function routePopup(r, lngLat) {
    sel = { type: 'route', r }; refreshLayers();
    await E.loadStations(E.stationsOf(r));
    const view = {
      render: async () => {
        const ri = E.routeInfo(r), isNew = r >= E.routes.length, dead = isDel(r), m = routeMedian(r);
        let h = '<div class="fe-pop"><div class="fe-head">' + badge(r) + '<span style="font-weight:600;font-size:12px">' + esc(ri.long || '') + '</span></div>';
        if (isNew && !dead) h += '<div style="display:flex;gap:6px;align-items:center;margin-top:4px"><input class="fe-name" data-k="name" value="' + esc(ri.name) + '">' +
          '<input type="color" data-k="color" value="' + (ri.color || '#e6194b') + '"></div>';
        if (!dead) h += '<table><tr><th>mode</th><th>headway (min)</th><th>speed (km/h)</th></tr><tr><td>' + modeSelect(curMode(r), 'data-k="mode"') + '</td>' +
          '<td><input type="number" min="0.5" step="0.5" data-k="h" value="' + fmt(m.h) + '"></td><td><input type="number" min="1" step="1" data-k="v" value="' + fmt(m.v) + '"></td></tr></table>';
        h += '<div>' + (dead ? '<button data-k="undel">Restore route</button>' : '<button data-k="stops">Add or remove stops</button><button data-k="del">Delete route</button>') +
          (!dead && !isNew && E.isEdited(r) ? '<button data-k="orig" title="back to the original route">↺ Original</button>' : '') + '</div></div>';
        popup.setHTML(h);
        const el = popup.getElement(); if (!el) return;
        bind(el, (k, x) => {
          if (k === 'stops') { setTool('stops', r); return; }
          edit(() => {
            if (k === 'del') { if (isNew) E.newRoutes[r - E.routes.length].gone = true; else E.edit(r).del = true; }
            else if (k === 'undel') undelete(r);
            else if (k === 'orig') delete E.route[r];
            else if (k === 'mode') setMode(r, +x.value);
            else if (k === 'name') ri.name = x.value;
            else if (k === 'color') ri.color = x.value;
            else scale([r], k, null, +x.value);
          });
        });
      },
    };
    openPopup(lngLat, view);
  }
  function pickRoute(rs, lngLat) {
    const h = '<div class="fe-pop fe-pick" style="margin-right:14px">' + rs.map((r) => '<div data-r="' + r + '">' + badge(r) + esc(E.routeInfo(r).long || '') +
      (isDel(r) ? ' <span style="color:#999">✕</span>' : '') + '</div>').join('') + '</div>';
    if (popup) popup.remove();
    popupFor = null;
    popup = new maplibregl.Popup({ closeOnClick: false, maxWidth: '420px' }).setLngLat(inView(lngLat)).setHTML(h).addTo(map);
    popup.on('close', () => map.setFilter('fe-hov', ['==', ['get', 'r'], -1]));
    popup.getElement().querySelectorAll('[data-r]').forEach((x) => {
      x.addEventListener('mouseenter', () => map.setFilter('fe-hov', ['==', ['get', 'r'], +x.dataset.r]));
      x.addEventListener('click', (e) => { e.stopPropagation(); routePopup(+x.dataset.r, lngLat); });
    });
  }
  function inView(lngLat) {   // a popup anchored off screen opens at the centre instead
    const ll = maplibregl.LngLat.convert(lngLat);
    return map.getBounds().contains(ll) ? ll : map.getCenter();
  }
  function openPopup(lngLat, view) {
    lngLat = inView(lngLat);
    popupFor = null;   // closing the previous popup must not clear the new selection
    if (popup) popup.remove();
    popup = new maplibregl.Popup({ closeOnClick: false, maxWidth: '560px' }).setLngLat(lngLat).setHTML('<div class="fe-pop">…</div>').addTo(map);
    popupFor = view;
    popup.on('close', () => { if (popupFor === view) { popupFor = null; if (tool === 'select') { sel = null; refreshLayers(); } } });
    view.render();
  }

  /* ---------------------------------------------------------------- map events */
  function hits(point) {
    const bb = [[point.x - 6, point.y - 6], [point.x + 6, point.y + 6]];
    const st = map.queryRenderedFeatures(bb, { layers: ['fe-st', 'fe-st-del'] });
    const rs = [...new Set(map.queryRenderedFeatures(bb, { layers: ['fe-lines-hit'] }).map((f) => f.properties.r))];
    return { st: st.length ? st[0] : null, rs };
  }
  function onMove(e) {
    if (!on || !tip) return;
    if (tool === 'draw') { map.getCanvas().style.cursor = 'crosshair'; tip.style.display = 'none'; return; }
    const h = hits(e.point);
    let text = '';
    if (h.st) text = esc(E.nameOf(h.st.properties.s) || 'Station');
    else if (h.rs.length) text = h.rs.slice(0, 4).map(badge).join('') + (h.rs.length > 4 ? ' +' + (h.rs.length - 4) : '');
    map.setFilter('fe-hov', ['==', ['get', 'r'], !h.st && h.rs.length === 1 ? h.rs[0] : -1]);
    map.getCanvas().style.cursor = tool === 'stops' ? 'copy' : (text ? 'pointer' : '');
    if (!text) { tip.style.display = 'none'; return; }
    tip.innerHTML = text; tip.style.display = 'block';
    const r = map.getContainer().getBoundingClientRect(), hr = tip.parentNode.getBoundingClientRect();
    tip.style.left = (r.left - hr.left + e.point.x + 14) + 'px'; tip.style.top = (r.top - hr.top + e.point.y + 10) + 'px';
  }
  function onClick(e) {
    if (!on) return;
    if (tool === 'draw') {
      draft.pts.push({ ll: [e.lngLat.lng, e.lngLat.lat], stop: !(e.originalEvent && e.originalEvent.shiftKey) });
      refreshLayers(); renderPanel(); return;
    }
    const h = hits(e.point);
    if (tool === 'stops') {   // the chosen route: one of its stops is taken off / put back; any other station or place becomes a stop of it
      const r = toolRoute;
      return edit(() => {
        const s = h.st ? h.st.properties.s : E.addStation(e.lngLat.lng, e.lngLat.lat, routeName(r));
        const e2 = E.edit(r), isNew = r >= E.routes.length, nr = isNew ? E.newRoutes[r - E.routes.length] : null;
        const own = isNew ? nr.st.indexOf(s) >= 0 : E.routes[r].st.indexOf(s) >= 0;
        if (isNew) { if (own) nr.st.splice(nr.st.indexOf(s), 1); else { nr.st.push(s); insertPoint(nr.coords, E.lonlatOf(s)); } }
        else if (own) { const i = e2.skip.indexOf(s); if (i >= 0) e2.skip.splice(i, 1); else e2.skip.push(s); }
        else { const i = e2.add.indexOf(s); if (i >= 0) e2.add.splice(i, 1); else e2.add.push(s); }
      });
    }
    if (h.st) return stationPopup(h.st.properties.s, h.st.geometry.coordinates);
    if (h.rs.length === 1) return routePopup(h.rs[0], e.lngLat);
    if (h.rs.length > 1) return pickRoute(h.rs, e.lngLat);
    if (popup) popup.remove();
    sel = null; refreshLayers();
  }
  function onKey(e) {
    if (!on || /^(INPUT|SELECT|TEXTAREA)$/.test((e.target && e.target.tagName) || '')) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); return; }
    if (tool === 'draw') {
      if (e.key === 'Enter') finishDraft();
      else if (e.key === 'Backspace') { e.preventDefault(); draft.pts.pop(); refreshLayers(); renderPanel(); }
      else if (e.key === 'Escape') setTool('select');
      return;
    }
    if (e.key === 'Escape') { if (tool === 'stops') setTool('select', -1, true); else if (popup) popup.remove(); }
  }
  async function finishDraft() {
    const stops = draft.pts.filter((p) => p.stop);
    if (stops.length < 2) return;
    remember();
    const st = stops.map((p) => E.addStation(p.ll[0], p.ll[1], draft.name));
    const r = E.addRoute({ name: draft.name, long: '', mode: draft.mode, h: draft.h, v: draft.v, color: draft.color, text: '#fff', st: [...new Set(st)],
                           coords: draft.pts.map((p) => p.ll) });
    const at = draft.pts[Math.floor(draft.pts.length / 2)].ll;
    draft = null; tool = 'select';
    map.doubleClickZoom.enable(); map.boxZoom.enable();
    await changed();
    routePopup(r, at);
  }
  function setTool(t, r, reopen) {
    const prev = toolRoute;
    tool = t; toolRoute = r == null ? -1 : r;
    if (t === 'draw') {   // shift-click is a bend of the line, not the map's box zoom; double clicks do not zoom
      if (!draft) draft = { pts: [], name: 'New line', mode: 0, h: 10, v: 20, color: '#e6194b' };
      if (popup) popup.remove(); map.doubleClickZoom.disable(); map.boxZoom.disable();
    } else { draft = null; map.doubleClickZoom.enable(); map.boxZoom.enable(); }
    if (t === 'stops') { if (popup) popup.remove(); sel = { type: 'route', r }; }
    if (t === 'select' && !reopen) sel = null;
    refreshLayers(); renderPanel();
    if (reopen && prev >= 0) { const st = E.stationsOf(prev); routePopup(prev, st.length ? E.lonlatOf(st[Math.floor(st.length / 2)]) : map.getCenter()); }
  }

  /* ---------------------------------------------------------------- panel */
  function status(t) { const el = $('feStatus'); if (el) el.textContent = t; }
  function editSummary(r) {
    const e = E.route[r], p = [];
    if (e.del) return 'deleted';
    if (e.mode !== null) p.push(MODES[e.mode]);
    if (e.h !== 1) p.push('headway ×' + fmt(e.h, 2));
    if (e.v !== 1) p.push('speed ×' + fmt(e.v, 2));
    if (e.add.length) p.push('+' + e.add.length + ' stops');
    if (e.skip.length) p.push('−' + e.skip.length + ' stops');
    return p.join(', ');
  }
  function zoomTo(list) {
    const pts = list.map(E.lonlatOf); if (!pts.length) return;
    const b = new maplibregl.LngLatBounds(pts[0], pts[0]); pts.forEach((p) => b.extend(p));
    map.fitBounds(b, { padding: 80, maxZoom: 15, duration: 600 });
  }
  function renderPanel() {
    if (!panel) return;
    let h = '<div class="fe-row"><button data-a="select" class="' + (tool === 'select' ? 'on' : '') + '">Edit</button>' +
      '<button data-a="draw" class="' + (tool === 'draw' ? 'on' : '') + '">New line</button>' +
      '<button data-a="undo" class="fe-i" title="undo (Ctrl+Z)"' + (history.length ? '' : ' disabled') + '>↶</button></div>';
    if (tool === 'draw') {
      const n = draft.pts.filter((p) => p.stop).length;
      h += '<label>name <input data-f="name" value="' + esc(draft.name) + '"></label><label>mode ' + modeSelect(draft.mode, 'data-f="mode"') + '</label>' +
        '<label>headway (min) <input type="number" min="0.5" step="0.5" data-f="h" value="' + draft.h + '"></label>' +
        '<label>speed (km/h) <input type="number" min="1" step="1" data-f="v" value="' + draft.v + '"></label>' +
        '<label>colour <input type="color" data-f="color" value="' + draft.color + '"></label>' +
        '<div class="fe-row"><button data-a="back" class="fe-i" title="remove the last point (Backspace)"' + (draft.pts.length ? '' : ' disabled') + '>⌫</button>' +
        '<button data-a="finish" title="Enter"' + (n > 1 ? '' : ' disabled') + '>Finish (' + n + ')</button><button data-a="cancel" title="Esc">Cancel</button></div>';
    }
    if (tool === 'stops') h += '<div class="fe-row"><span style="flex:1">' + badge(toolRoute) + '</span><button data-a="done" title="Esc">Done</button></div>';
    h += '<label>colour <select data-a="colour"><option value="score">recomputed score</option><option value="diff">change vs original</option></select></label>';
    const rows = [];
    for (const r in E.route) if (E.isEdited(+r) && +r < E.routes.length) rows.push('<div data-go="r' + r + '"><span>' + badge(+r) + esc(editSummary(+r)) + '</span><button class="fe-x" data-u="' + r + '" title="back to the original route">↺</button></div>');
    E.newRoutes.forEach((nr, k) => { if (!nr.gone) rows.push('<div data-go="r' + (E.routes.length + k) + '"><span>' + badge(E.routes.length + k) + 'new</span><button class="fe-x" data-n="' + (E.routes.length + k) + '" title="delete">✕</button></div>'); });
    E.stDel.forEach((s) => rows.push('<div data-go="s' + s + '"><span>' + esc(E.nameOf(s) || 'station') + ': deleted</span><button class="fe-x" data-s="' + s + '" title="restore">↺</button></div>'));
    if (rows.length) h += '<div class="fe-t">Edits</div><div class="fe-list">' + rows.join('') + '</div>';
    h += '<div class="fe-row">' + (rows.length ? '<button data-a="reset">Undo all</button>' : '') +
      '<button data-a="export" class="fe-i" title="save the edits to a file"' + (rows.length ? '' : ' disabled') + '>⤓</button>' +
      '<button data-a="import" class="fe-i" title="load edits from a file">⤒</button></div><div id="feStatus" style="color:#666;min-height:14px"></div>';
    panel.innerHTML = h;
    const cs = panel.querySelector('[data-a="colour"]'); cs.value = api.colour() === 'diff' ? 'diff' : 'score';
    cs.addEventListener('change', () => api.setColour(cs.value));
    panel.querySelectorAll('button[data-a]').forEach((b) => b.addEventListener('click', async () => {
      const a = b.dataset.a;
      if (a === 'select' || a === 'cancel' || a === 'done') setTool('select', -1, a === 'done');
      else if (a === 'draw') setTool('draw');
      else if (a === 'undo') undo();
      else if (a === 'back') { draft.pts.pop(); refreshLayers(); renderPanel(); }
      else if (a === 'finish') finishDraft();
      else if (a === 'reset') { remember(); if (popup) popup.remove(); sel = null; api.recomputeEdit(await E.reset()); try { localStorage.removeItem(storeKey()); } catch (e) { /* */ } refreshLayers(); renderPanel(); }
      else if (a === 'export') {
        const blob = new Blob([JSON.stringify(E.serialize(), null, 1)], { type: 'application/json' }), u = URL.createObjectURL(blob), l = document.createElement('a');
        l.href = u; l.download = (D.meta.name || 'map') + '_edits.json'; l.click(); setTimeout(() => URL.revokeObjectURL(u), 1000);
      } else if (a === 'import') {
        const inp = document.createElement('input'); inp.type = 'file'; inp.accept = '.json,application/json';
        inp.onchange = async () => { const f = inp.files[0]; if (!f) return; try { const o = JSON.parse(await f.text()); remember(); E.restore(o); await changed(); } catch (err) { status('✕'); } };
        inp.click();
      }
    }));
    panel.querySelectorAll('[data-f]').forEach((x) => x.addEventListener('change', () => { const f = x.dataset.f; draft[f] = (f === 'name' || f === 'color') ? x.value : +x.value; }));
    panel.querySelectorAll('[data-u]').forEach((b) => b.addEventListener('click', (e) => { e.stopPropagation(); edit(() => { delete E.route[+b.dataset.u]; }); }));
    panel.querySelectorAll('[data-n]').forEach((b) => b.addEventListener('click', (e) => { e.stopPropagation(); edit(() => E.removeNewRoute(+b.dataset.n)); }));
    panel.querySelectorAll('[data-s]').forEach((b) => b.addEventListener('click', (e) => { e.stopPropagation(); edit(() => { E.stDel.delete(+b.dataset.s); }); }));
    panel.querySelectorAll('[data-go]').forEach((x) => x.addEventListener('click', () => {
      const g = x.dataset.go, id = +g.slice(1);
      if (g[0] === 'r') { const st = E.stationsOf(id); zoomTo(st); routePopup(id, E.lonlatOf(st[Math.floor(st.length / 2)])); }
      else { zoomTo([id]); stationPopup(id, E.lonlatOf(id)); }
    }));
  }

  /* ---------------------------------------------------------------- on / off */
  async function enter(btn) {
    map = window.__mainMap || window.__map; api = window.__flexApi;
    if (!map || !api || !window.FlexEdit) return;
    btn.classList.add('active');
    window.__flexEditOn = true; window.__routeEditorMode = 'flexedit';   // the map's cell popups stay closed while editing
    style();
    const host = map.getContainer().parentNode || document.body, r = btn.getBoundingClientRect(), hr = host.getBoundingClientRect();
    panel = document.createElement('div'); panel.id = 'flexEditPanel';
    panel.style.left = (r.left - hr.left) + 'px'; panel.style.top = (r.bottom - hr.top + 6) + 'px';
    host.appendChild(panel);
    tip = document.createElement('div'); tip.id = 'flexEditTip'; tip.style.display = 'none'; host.appendChild(tip);
    panel.innerHTML = '<div id="feStatus">…</div>';
    D = await api.load();
    if (!D) { status(''); return exit(btn); }
    E = D.edit || (D.edit = FlexEdit.create(D, D.path, (D.meta && D.meta.build_params) || api.defaults()));
    await E.load();
    if (api.colour() === 'map') api.setColour('score');
    on = true; tool = 'select'; sel = null;
    hideBase(); addLayers();
    if (!E.hasEdits()) {   // the edits of the last visit
      try { const o = JSON.parse(localStorage.getItem(storeKey()) || 'null'); if (o) { E.restore(o); api.recomputeEdit(await E.apply()); } } catch (err) { /* ignore */ }
    }
    refreshLayers();
    map.on('click', onClick); map.on('mousemove', onMove); map.on('zoomend', hideBase); map.on('moveend', hideBase); map.on('idle', hideBase);
    document.addEventListener('keydown', onKey);
    renderPanel();
  }
  function exit(btn) {
    on = false; draft = null; sel = null; tool = 'select';
    btn.classList.remove('active');
    window.__routeEditorMode = 'idle';
    window.__flexEditOn = !!(E && E.hasEdits());   // the recomputed colours stay while there are edits
    if (map) {
      map.off('click', onClick); map.off('mousemove', onMove); map.off('zoomend', hideBase); map.off('moveend', hideBase); map.off('idle', hideBase);
      map.doubleClickZoom.enable(); map.boxZoom.enable(); map.getCanvas().style.cursor = '';
      for (const id of LAYERS) if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', 'none');
      showBase();
    }
    document.removeEventListener('keydown', onKey);
    if (popup) popup.remove();
    if (panel) panel.remove();
    if (tip) tip.remove();
    panel = null; tip = null;
  }
  function install() {
    const old = $('routeEditToggleBtn');
    if (!old) return false;
    const btn = old.cloneNode(true);   // the map's own draw-a-line handlers stay on the old node
    btn.textContent = 'Edit'; btn.title = 'Edit routes and stops, add lines';
    old.parentNode.replaceChild(btn, old);
    const tb = $('route-toolbar'); if (tb) tb.style.display = 'none';
    btn.addEventListener('click', (ev) => { ev.stopPropagation(); if (on || panel) exit(btn); else enter(btn); });
    return true;
  }
  const tryInstall = () => { if (!install()) setTimeout(tryInstall, 400); };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', tryInstall); else tryInstall();
  window.__flexEditDebug = { engine: () => E, state: () => ({ tool, toolRoute, sel, history: history.length }) };
})();
