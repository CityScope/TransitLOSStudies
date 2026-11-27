/* 3D view and the "Buildings" geometry of a flex map (only when window.FLEX_CFG.buildings is set: the folder of the buildings' PMTiles).
 *
 *   Buildings geometry   a radio besides hexagons / circles / census (from zoom 14; zoomed out it shows the census geometry): every building footprint coloured by the score of its cell (the cell's
 *                        value is the one the Scoring tab computes, through the same `access_override` feature-state of the other geometries).
 *   3D                   a button: terrain (mountains) extruded from elevation tiles, the map tilted; with the Buildings geometry the buildings
 *                        are extruded to their height. Off: flat map, the pitch back to 0.
 * Terrain tiles: the city's own terrarium elevation tiles (CFG.terrain, written by flex/build_terrain.py), else the public AWS ones (elevation only,
 * no background map: the basemap stays the OpenMapTiles one).
 * Colours: one `match` expression on the building's cell (`h3_11`) for the cells on screen, set on the layer after every move: no per-building state.
 */
(function () {
  const CFG = window.FLEX_CFG || {};
  if (!CFG.buildings) return;
  const GROUP = 'input[name="__base_group"]';
  const OTHER = /^(hexagons|circles|census):/;
  let m = null, on3d = false, building = false;
  
  function layersOf(re) { return (m.getStyle().layers || []).filter((l) => re.test(l.source || '')).map((l) => l.id); }
  const BUILDING_ZOOM = 14;   // buildings are drawn from this zoom; below it the census geometry is shown
  const saved = new Map();    // layer -> its visibility before the buildings geometry took over
  function applyLevel() {
    if (!building) return;
    const low = m.getZoom() < BUILDING_ZOOM;
    for (const id of layersOf(OTHER)) {
      if (!saved.has(id)) saved.set(id, m.getLayoutProperty(id, 'visibility') || 'visible');
      const census = /^census:/.test(m.getLayer(id).source || '');
      m.setLayoutProperty(id, 'visibility', census && low ? 'visible' : 'none');
    }
  }
  function setBuilding(flag) {
    building = flag;
    if (flag) applyLevel();
    else { saved.forEach((v, id) => { if (m.getLayer(id)) m.setLayoutProperty(id, 'visibility', v); }); saved.clear(); }
    refreshBuildings();
  }
  let tm = 0;
  async function colourBuildings() {
    if (!building || !m.getLayer('bld-2d') || !window.__flexCellColour) return;
    if (window.__flexApi && window.__flexApi.load) await window.__flexApi.load();   // the scores of the cells under the buildings (the Scoring tab loads them only when it is open)
    if (!building) return;
    const cells = new Set();
    for (const f of m.queryRenderedFeatures({ layers: [on3d ? 'bld-3d' : 'bld-2d'] })) if (f.properties && f.properties.h3_11) cells.add(f.properties.h3_11);
    const expr = ['match', ['get', 'h3_11']];
    for (const c of cells) { const col = window.__flexCellColour(c, 11); if (col) expr.push(c, col); }
    expr.push('#c9c9c9');
    const key = expr.length > 3 ? expr : '#c9c9c9';
    m.setPaintProperty('bld-2d', 'fill-color', key); m.setPaintProperty('bld-3d', 'fill-extrusion-color', key);
  }
  function schedule() { clearTimeout(tm); tm = setTimeout(colourBuildings, 200); }
  window.__flexRepaintBuildings = schedule;
  function refreshBuildings() {
    if (!m.getLayer('bld-2d')) return;
    m.setLayoutProperty('bld-2d', 'visibility', building && !on3d ? 'visible' : 'none');
    m.setLayoutProperty('bld-3d', 'visibility', building && on3d ? 'visible' : 'none');
    schedule();
  }
  let prevGeom = null;   // the geometry chosen before 3D switched to the buildings
  function chooseGeometry(v) {
    const sel = document.getElementById('baseGroupSelect');
    if (sel) { sel.value = v; sel.dispatchEvent(new Event('change', { bubbles: true })); return; }
    const r = document.querySelector(GROUP + '[value="' + v + '"]'); if (r && !r.checked) { r.checked = true; r.dispatchEvent(new Event('change', { bubbles: true })); }
  }
  function set3d(flag) {
    on3d = flag;
    if (flag && !building && CFG.buildings) { const sel = document.getElementById('baseGroupSelect'); const c = document.querySelector(GROUP + ':checked'); prevGeom = sel ? sel.value : (c && c.value); chooseGeometry('building'); }   // 3D shows the buildings extruded
    else if (!flag && prevGeom) { const p = prevGeom; prevGeom = null; if (building) chooseGeometry(p); }
    if (flag) {
      m.setTerrain({ source: 'flex-dem', exaggeration: 1.3 });
      m.jumpTo({ pitch: 60 });   // an instant tilt (an animated one left only the basemap drawn in tests)
    } else {
      m.setTerrain(null);
      m.jumpTo({ pitch: 0 });
    }
    refreshBuildings();
    const b = document.getElementById('flex3dButton'); if (b) { b.style.background = flag ? '#2b6cb0' : '#fff'; b.style.color = flag ? '#fff' : '#222'; }
  }
  function init() {
    m = window.__mainMap || window.__map; if (!m) return setTimeout(init, 300);
    const go = () => {
      if (m.getSource('buildings')) return;
      m.addSource('buildings', { type: 'vector', tiles: ['pmtiles://' + CFG.buildings + '/buildings.pmtiles/{z}/{x}/{y}'], minzoom: 14, maxzoom: 15 });
      const col = '#c9c9c9';   // replaced by colourBuildings(): one match expression on the cell of the building, not a state per building
      m.addLayer({ id: 'bld-2d', type: 'fill', source: 'buildings', 'source-layer': 'buildings', minzoom: 14, layout: { visibility: 'none' }, paint: { 'fill-color': col, 'fill-opacity': 0.9 } });
      m.addLayer({ id: 'bld-3d', type: 'fill-extrusion', source: 'buildings', 'source-layer': 'buildings', minzoom: 14, layout: { visibility: 'none' },
                  paint: { 'fill-extrusion-color': col, 'fill-extrusion-height': ['get', 'height_m'], 'fill-extrusion-base': 0, 'fill-extrusion-opacity': 0.92 } });
      const tm = CFG.terrainMeta;   // {bounds, minzoom, maxzoom} of the city's own elevation tiles: the page asks for none outside them
      m.addSource('flex-dem', Object.assign({ type: 'raster-dem', tiles: [CFG.terrain ? CFG.terrain + '/{z}/{x}/{y}.png' : 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
                                              encoding: 'terrarium', tileSize: 256, maxzoom: tm ? tm.maxzoom : 14 }, tm ? { bounds: tm.bounds, minzoom: tm.minzoom } : {}));
      // the radio besides the other geometries
      const census = document.querySelector(GROUP + '[value="census"]') || document.querySelector(GROUP);   // (a city without census geometry has the other radios)
      if (census) {
        const lab = census.closest('label'), nl = document.createElement('label'); nl.style.display = 'none';   // the radio stays hidden and drives the map; the page's dropdown gets the entry
        nl.innerHTML = '<input type="radio" name="__base_group" value="building"> buildings';
        lab.parentNode.insertBefore(nl, lab.nextSibling);
        const addOption = () => {   // the page turns its radios into one dropdown (baseGroupSelect): the new geometry is an option of it
          const sel = document.getElementById('baseGroupSelect');
          if (!sel) return false;
          if (!sel.dataset.flexBuilding) {   // the page's own handler only knows the radios it was built with: this one drives the buildings radio
            sel.dataset.flexBuilding = '1';
            sel.addEventListener('change', () => {
              const r = document.querySelector(GROUP + '[value="building"]');
              if (sel.value === 'building' && r && !r.checked) { r.checked = true; r.dispatchEvent(new Event('change', { bubbles: true })); }
            });
          }
          if (![...sel.options].some((o) => o.value === 'building')) { const o = document.createElement('option'); o.value = 'building'; o.textContent = '🏢 buildings'; sel.insertBefore(o, sel.querySelector('option[value="none"]')); }
          return true;
        };
        let tries = 0; const t = setInterval(() => { if (addOption() || ++tries > 400) clearInterval(t); }, 500);
      }
      document.querySelectorAll(GROUP).forEach((el) => el.addEventListener('change', () => setTimeout(() => { const c = document.querySelector(GROUP + ':checked'); setBuilding(!!c && c.value === 'building'); }, 0)));
      m.on('zoom', applyLevel); m.on('moveend', schedule); m.on('sourcedata', (e) => { if (e.sourceId === 'buildings' && e.isSourceLoaded) schedule(); });
      const btn = document.createElement('button'); btn.id = 'flex3dButton'; btn.type = 'button'; btn.textContent = '3D'; btn.title = '3D view';
      btn.style.cssText = 'border:1px solid #c9ced6;background:#fff;color:#222;border-radius:6px;cursor:pointer;font:600 12px sans-serif;padding:0 6px;margin-right:4px;height:22px;line-height:20px;';
      btn.onclick = () => set3d(!on3d);
      // beside the dark-mode button, in the legend's header (the dark button is placed by the page a moment later)
      const place = () => {
        const head = document.getElementById('layer-switcher-head'); if (!head) return false;
        const ref = document.getElementById('mapDarkBtn') || head.querySelector('#layer-switcher-toggle');
        head.insertBefore(btn, ref || null); return !!document.getElementById('mapDarkBtn');
      };
      let pt = 0; const pti = setInterval(() => { if (place() || ++pt > 40) clearInterval(pti); }, 500); place();
    };
    if (m.isStyleLoaded()) go(); else { const t = setInterval(() => { if (m.isStyleLoaded()) { clearInterval(t); go(); } }, 500); }   // (the map's `load` may already have fired)
  }
  init();
})();
