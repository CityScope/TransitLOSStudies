# -*- coding: utf-8 -*-
"""Bilingual (ES/EN) layer for the generated map.html.

``transitlos.map.build.build_city_map`` has NO i18n / extra-html hook (checked: no `i18n`, `extra_html`, `lang` parameter), so, like usa_study's
``_center_on_contiguous_us``, the finished map.html is post-processed: one ``<script>`` (dictionary from ``code/i18n.py`` + runtime below) is
inserted before ``</body>``. The runtime translates every text node and the placeholder/title/aria-label attributes (a MutationObserver
covers popups, legends and panels built later by the map's own JS) by looking the ORIGINAL English text up in the dictionary; switching back
to English restores the originals. Column names are translated through ``COLUMN_LABELS`` (ES) and ``COLUMN_LABELS_EN``.

Language choice: ``?lang=es|en`` forces one; else the visitor's saved choice (localStorage, try/catch); else the browser language
(navigator.languages[0] / navigator.language: Spanish if it starts with ``es``, else English). No geo-IP or any other network lookup.
Control: a small low-contrast text box at the top right showing ``ES`` or ``EN``; click opens a two-item list (ES, EN). No icons or emoji.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import i18n
from .format_inject import FORMAT_JS

MARK = "<!-- spain-i18n -->"

RUNTIME = r"""
(function () {
  var D = window.__I18N;
  var KEY = 'spain_map_lang';
  var lang = 'en';
  var store = new WeakMap();      // text node / [element, attr] -> {orig, shown}
  var busy = false;

  function lsGet(k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } }
  function lsSet(k, v) { try { window.localStorage.setItem(k, v); } catch (e) {} }

  function norm(s) { return s.replace(/\s+/g, ' ').trim(); }
  function tr(text) {
    var t = norm(text);
    if (!t) return null;
    var dict = lang === 'es' ? D.es : D.en;
    var m = dict[t];
    if (m !== undefined) return m;
    if (lang === 'es') {
      for (var i = 0; i < D.patterns.length; i++) {
        var p = D.patterns[i], re = p._re || (p._re = new RegExp(p[0]));
        if (re.test(t)) return t.replace(re, p[1]);
      }
    }
    var c = t.indexOf(': ');          // "Label: value" -> translate the label
    if (c > 0) { var l = dict[t.slice(0, c)]; if (l !== undefined) return l + t.slice(c); }
    return null;
  }
  function applyText(node) {
    var rec = store.get(node);
    if (rec && node.nodeValue !== rec.shown) rec = null;     // changed by the page's own code: new original
    var orig = rec ? rec.orig : node.nodeValue;
    var out = orig, w = orig.match(/^(\s*)([\s\S]*?)(\s*)$/);
    var t = tr(w[2]);
    if (t !== null) out = w[1] + t + w[3];
    if (out !== node.nodeValue) node.nodeValue = out;
    store.set(node, {orig: orig, shown: node.nodeValue});
  }
  var ATTRS = ['placeholder', 'title', 'aria-label', 'alt'];
  function applyAttr(el, a) {
    var v = el.getAttribute(a);
    if (v === null) return;
    var map = store.get(el) || {};
    var rec = map[a];
    if (rec && v !== rec.shown) rec = null;
    var orig = rec ? rec.orig : v, out = orig;
    var t = tr(orig); if (t !== null) out = t;
    if (out !== v) el.setAttribute(a, out);
    map[a] = {orig: orig, shown: out};
    store.set(el, map);
  }
  function skip(el) { return el && (el.nodeName === 'SCRIPT' || el.nodeName === 'STYLE' || el.id === 'lang-switch' || (el.closest && el.closest('#lang-switch'))); }
  function walk(root) {
    if (!root) return;
    if (root.nodeType === 3) { if (!skip(root.parentNode)) applyText(root); return; }
    if (root.nodeType !== 1 || skip(root)) return;
    ATTRS.forEach(function (a) { if (root.hasAttribute(a)) applyAttr(root, a); });
    var tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT, null);
    var n;
    while ((n = tw.nextNode())) {
      if (n.nodeType === 3) { if (!skip(n.parentNode)) applyText(n); }
      else if (!skip(n)) ATTRS.forEach(function (a) { if (n.hasAttribute(a)) applyAttr(n, a); });
    }
  }
  var obs = new MutationObserver(function (muts) {
    if (busy) return;
    busy = true;
    try {
      muts.forEach(function (m) {
        if (m.type === 'childList') m.addedNodes.forEach(walk);
        else if (m.type === 'characterData') { if (!skip(m.target.parentNode)) applyText(m.target); }
        else if (m.type === 'attributes') applyAttr(m.target, m.attributeName);
      });
    } finally { busy = false; }
  });
  function translateAll() {
    busy = true;
    try { walk(document.body); document.title = tr(window.__origTitle) || window.__origTitle; } finally { busy = false; }
  }
  function setLang(l, remember) {
    lang = (l === 'es') ? 'es' : 'en';
    document.documentElement.setAttribute('lang', lang);
    if (remember) lsSet(KEY, lang);
    translateAll();
    var b = document.getElementById('lang-current'); if (b) b.textContent = lang.toUpperCase();
    ['es', 'en'].forEach(function (x) { var o = document.getElementById('lang-opt-' + x); if (o) o.setAttribute('aria-selected', String(x === lang)); });
    try { window.dispatchEvent(new CustomEvent('spain-lang-change', {detail: lang})); } catch (e) {}
  }
  window.__setMapLanguage = setLang;
  window.__mapLanguage = function () { return lang; };
  window.__i18n = function (en) { var t = D.es[norm(en)]; return (lang === 'es' && t !== undefined) ? t : en; };   // for scripts that build strings later

  function buildSwitch() {
    var css = document.createElement('style');
    css.textContent =
      '.maplibregl-ctrl-top-right{margin-top:18px}#lang-switch{position:fixed;top:2px;right:4px;z-index:100000;font:11px/1 system-ui,sans-serif;letter-spacing:.04em;opacity:.5;transition:opacity .15s}' +
      '#lang-switch:hover,#lang-switch:focus-within{opacity:.95}' +
      '#lang-current{background:rgba(255,255,255,.55);color:#445;border:1px solid rgba(0,0,0,.12);border-radius:3px;padding:3px 6px;cursor:pointer;font:inherit}' +
      '#lang-current:after{content:"";display:inline-block;margin-left:5px;vertical-align:middle;border:3px solid transparent;border-top-color:currentColor;border-bottom:0}' +
      '#lang-list{display:none;position:absolute;right:0;top:100%;margin:2px 0 0;padding:2px 0;list-style:none;min-width:100%;background:rgba(255,255,255,.95);border:1px solid rgba(0,0,0,.15);border-radius:3px}' +
      '#lang-switch.open #lang-list{display:block}' +
      '#lang-list li{padding:4px 8px;color:#445;cursor:pointer;text-align:left}' +
      '#lang-list li:hover,#lang-list li[aria-selected=true]{background:rgba(0,0,0,.07)}' +
      '@media (prefers-color-scheme:dark){#lang-current{background:rgba(30,30,30,.55);color:#bbc;border-color:rgba(255,255,255,.15)}#lang-list{background:rgba(30,30,30,.95);border-color:rgba(255,255,255,.2)}#lang-list li{color:#bbc}#lang-list li:hover,#lang-list li[aria-selected=true]{background:rgba(255,255,255,.1)}}';
    document.head.appendChild(css);
    var box = document.createElement('div');
    box.id = 'lang-switch';
    box.innerHTML = '<button id="lang-current" type="button" aria-haspopup="listbox" aria-expanded="false" aria-label="Language / Idioma">EN</button>' +
      '<ul id="lang-list" role="listbox"><li id="lang-opt-es" role="option" data-l="es">ES</li><li id="lang-opt-en" role="option" data-l="en">EN</li></ul>';
    document.body.appendChild(box);
    var btn = box.querySelector('#lang-current');
    function toggle(open) { box.classList.toggle('open', open); btn.setAttribute('aria-expanded', String(open)); }
    btn.addEventListener('click', function (e) { e.stopPropagation(); toggle(!box.classList.contains('open')); });
    box.querySelectorAll('li').forEach(function (li) {
      li.addEventListener('click', function (e) { e.stopPropagation(); setLang(li.getAttribute('data-l'), true); toggle(false); });
    });
    document.addEventListener('click', function () { toggle(false); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') toggle(false); });
  }

  function navLang() {
    var n = ((navigator.languages && navigator.languages[0]) || navigator.language || 'en').toLowerCase();
    return n.indexOf('es') === 0 ? 'es' : 'en';
  }
  function start() {
    window.__origTitle = document.title;
    buildSwitch();
    obs.observe(document.body, {childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ATTRS});
    var q = (new URLSearchParams(location.search).get('lang') || '').toLowerCase();
    var saved = lsGet(KEY);
    if (q === 'es' || q === 'en') return setLang(q, false);
    if (saved === 'es' || saved === 'en') return setLang(saved, false);
    setLang(navLang(), false);              // browser language only: no network lookup, no visitor data leaves the page
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
"""


def dictionary() -> dict:
    patterns = [[rx, re.sub(r"\\(\d)", r"$\1", repl)] for rx, repl in i18n.PATTERNS_ES]   # python \1 -> JS $1
    return {"es": i18n.UI_ES, "patterns": patterns, "columns_es": i18n.COLUMN_LABELS_ES, "columns_en": i18n.COLUMN_LABELS_EN}


def runtime_script(columns=()) -> str:
    d = dictionary()
    extra_en, extra_es = i18n.labels_for(columns)
    # column labels are plain English-label -> Spanish entries for the translator; English overrides rewrite the humanized English label
    es = dict(d["es"])
    es.update(d["columns_es"])
    es.update(extra_es)
    es.update(getattr(i18n, "EXTRA_ES", {}))
    payload = {"es": es, "en": {**d["columns_en"], **extra_en}, "patterns": d["patterns"]}
    return "<script>window.__I18N=" + json.dumps(payload, ensure_ascii=False) + ";" + FORMAT_JS + RUNTIME + "</script>"


def inject_language_switch(html_path, columns=()) -> None:
    path = Path(html_path)
    text = path.read_text(encoding="utf-8")
    if MARK in text:
        text = re.sub(re.escape(MARK) + r"<script>.*?</script>", "", text, flags=re.S)
    block = MARK + runtime_script(columns)
    idx = text.rfind("</body>")
    text = text[:idx] + block + text[idx:] if idx >= 0 else text + block
    path.write_text(text, encoding="utf-8")
