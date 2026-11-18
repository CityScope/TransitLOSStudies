# -*- coding: utf-8 -*-
"""Client-side number formatting + native-resolution notes of the Spain map (inlined by ``i18n_inject`` next to the language runtime).

``transitlos.map.build`` has no formatting hook, so the page's own global functions are wrapped (same approach the city studies use for
``_control_panel_html``): ``__shapePopupNum(v, key)`` (popup rows, KPI values, legends, stats panel) formats by column family and falls
back to the builder's formatter for anything it does not know; ``__isPercentField(key)`` answers False for ``ine_incomeP80P20Ratio``
(a plain ratio such as 3.4, not a percentage) which also removes the " (%)" suffix the builder adds to its label in JS-built lists.
Percent columns arrive as 0-1 fractions (``census_spain.map_columns`` rescales the sources that publish 0-100).
Separators follow the page language (en: 1,234.5; es: 1.234,5).
"""

FORMAT_JS = r"""
(function () {
  function loc() { return (window.__mapLanguage && window.__mapLanguage() === 'es') ? 'es-ES' : 'en-US'; }
  function nf(v, dec) {
    var o = {minimumFractionDigits: dec, maximumFractionDigits: dec};
    try { o.useGrouping = 'always'; return Number(v).toLocaleString(loc(), o); }
    catch (e) { o.useGrouping = true; return Number(v).toLocaleString(loc(), o); }
  }
  function es() { return loc() === 'es-ES'; }
  var COUNT = /votes|electors|population|households|dwellings|buildings|building_units|residents|laborforce|places|deposits|netmig|n_vc|workers|foreign|poverty|education/i;
  function spainFmt(v, key) {
    if (v == null || v === '' || isNaN(v)) return null;
    var k = String(key || ''), x = Number(v);
    if (!k) return null;
    if (k === 'level_of_service') return nf(x, 1);
    if (/P80P20Ratio$/i.test(k)) return nf(x, 2);
    if (/(share|rate|ratio|pct|percent)$/i.test(k)) return nf(x * 100, 1) + (es() ? ' %' : '%');
    if (/rent_m2|eur_m2/i.test(k)) return nf(x, /valor_tasado/i.test(k) ? 0 : 2) + ' €/m²';
    if (/area_m2|AvgUsefulArea|m2_total/i.test(k)) return nf(x, 0) + ' m²';
    if (/income|rent_total|rent_monthly/i.test(k)) return nf(x, 0) + ' €';
    if (/ageMean|AvgAge/i.test(k)) return nf(x, 1) + ' ' + (es() ? 'años' : 'years');
    if (/year_built/i.test(k)) return String(Math.round(x));
    if (/height_mean_m/i.test(k)) return nf(x, 1) + '\u00a0m';
    if (/per_building/i.test(k)) return nf(x, 2);
    if (/householdSizeMean/i.test(k)) return nf(x, 2);
    if (/giniIndex/i.test(k)) return nf(x, 1);
    if (COUNT.test(k)) return nf(x, 0);
    var a = Math.abs(x);
    return nf(x, a >= 100 ? 0 : (a >= 10 ? 1 : 2));
  }
  window.__spainFmt = spainFmt;
  var num0 = window.__shapePopupNum;
  window.__shapePopupNum = function (v, key) { var s = spainFmt(v, key); return s !== null ? s : (num0 ? num0(v, key) : String(v)); };
  var pct0 = window.__isPercentField;
  window.__isPercentField = function (key) { return /P80P20Ratio$/i.test(String(key || '')) ? false : (pct0 ? pct0(key) : false); };

  // Native-resolution notes under the popup's statistics table (values of a whole municipality / census district repeat on each cell).
  var NOTES = [
    [/(municipality|municipio)\)/i, 'Values marked (municipality) describe the whole municipality and repeat on each of its sections and cells.'],
    [/(district|distrito)\)/i, 'Values marked (district) come from the census district and repeat on each of its sections and cells.']
  ];
  function addNotes(root) {
    var wraps = root.querySelectorAll ? root.querySelectorAll('.cpTableWrap') : [];
    wraps.forEach(function (w) {
      if (w.querySelector('.cpNote')) return;
      var labels = Array.prototype.map.call(w.querySelectorAll('tbody td:first-child'), function (td) { return td.textContent; }).join('\n');
      NOTES.forEach(function (n) {
        if (!n[0].test(labels)) return;
        var d = document.createElement('div');
        d.className = 'cpNote';
        d.style.cssText = 'font-size:10px;color:#778;padding:4px 10px;border-top:1px solid #eef0f3;';
        d.textContent = n[1];
        w.appendChild(d);
      });
    });
  }
  new MutationObserver(function (muts) {
    muts.forEach(function (m) { m.addedNodes.forEach(function (nd) { if (nd.nodeType === 1 && nd.querySelector && nd.querySelector('.cpTableWrap')) setTimeout(function () { addNotes(nd); }, 120); }); });   // after the language layer has labelled the rows
  }).observe(document.body, {childList: true, subtree: true});
})();
"""
