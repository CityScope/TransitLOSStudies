"""Headless-browser test of the bilingual map.

    python tests/test_language_switch.py data/PaisVasco/map.html

Checks: the dropdown exists (text only, ES/EN), opens on click and lists ES and EN; picking each option switches panel labels, popup and column names
and back; the choice is stored in localStorage and survives a reload; the default follows the browser language (es* -> Spanish, else English);
?lang= forces a language; no request is ever made to a geo-IP service.
"""
import json
import tempfile
import re
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HTML = Path(sys.argv[1]).resolve()
PORT = 8791
srv = subprocess.Popen([sys.executable, str(Path(__file__).with_name("rangeserver.py")), str(HTML.parent), str(PORT)])
time.sleep(1.0)
URL = f"http://127.0.0.1:{PORT}/{HTML.name}"
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra and not cond else ""))
    if not cond:
        fails.append(name)


GEO_CALLS = []


def geo(country):
    """Route handler: records (and aborts) any call to a geo-IP service; there must be none."""
    def handler(route):
        GEO_CALLS.append(route.request.url)
        route.abort()
    return handler


def page(browser, country, locale):
    ctx = browser.new_context(locale=locale, viewport={"width": 1400, "height": 900})
    ctx.route(re.compile(r"api\.country\.is|geojs\.io"), geo(country))
    pg = ctx.new_page()
    pg.goto(URL)
    pg.wait_for_selector("#lang-current")
    pg.wait_for_timeout(2500)
    return ctx, pg


def body(pg):
    return pg.evaluate("document.body.innerText")


try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        # --- defaults ---
        ctx, pg = page(b, None, "es-ES")
        check("Spanish browser -> Spanish", pg.inner_text("#lang-current") == "ES" and "Leyenda" in body(pg), pg.inner_text("#lang-current"))
        ctx.close()
        ctx, pg = page(b, None, "en-US")
        check("English browser -> English", pg.inner_text("#lang-current") == "EN" and "Legend" in body(pg))
        ctx.close()
        ctx, pg = page(b, None, "fr-FR")
        check("other browser language -> English", pg.inner_text("#lang-current") == "EN" and "Legend" in body(pg))
        ctx.close()
        ctx, pg = page(b, None, "es-MX")
        check("es-MX -> Spanish", pg.inner_text("#lang-current") == "ES")
        ctx.close()
        # --- dropdown ---
        ctx, pg = page(b, "US", "en-US")
        sw = pg.locator("#lang-switch")
        check("switch exists, top-right, text only", sw.count() == 1 and not re.search(r"[^\x00-\x7f]", sw.inner_text().replace("▾", "")), sw.inner_text())
        box = sw.bounding_box()
        check("switch is small and in the top strip (the map renderer places it next to the top-left controls)", box["y"] < 40 and box["width"] < 80 and box["height"] < 40, str(box))
        check("list closed initially", not pg.locator("#lang-list").is_visible())
        pg.click("#lang-current")
        opts = [t.strip() for t in pg.locator("#lang-list li").all_inner_texts()]
        check("click opens a dropdown listing ES and EN", pg.locator("#lang-list").is_visible() and opts == ["ES", "EN"], str(opts))
        pg.click("#lang-opt-es")
        pg.wait_for_timeout(500)
        t = body(pg)
        check("ES: current shows ES, list closed", pg.inner_text("#lang-current") == "ES" and not pg.locator("#lang-list").is_visible())
        check("ES: panel labels translated", all(s in t for s in ("Leyenda", "Capas", "Nivel de servicio", "Mapa base")), t[:300])
        check("ES: no English panel labels left", "Legend" not in t and "Basemap opacity" not in t)
        check("ES: html lang", pg.evaluate("document.documentElement.lang") == "es")
        # popup in Spanish
        pg.click("text=censo")
        pg.wait_for_timeout(2500)
        pg.evaluate("window.__mainMap.jumpTo({center: [-1.98, 43.31], zoom: 14})")   # a known populated spot, whatever the map's start view
        pg.wait_for_timeout(2500)
        pg.evaluate("window.__openCellPopup(-1.98, 43.31, {})")
        pg.wait_for_timeout(2500)
        t = pg.evaluate("[...document.querySelectorAll('.maplibregl-popup')].map(e=>e.innerText).join('|')")
        check("ES: popup header and fields translated", all(x in t.lower() for x in ("hexágono", "celda h3", "nivel de servicio", "población")) and any(x in t.lower() for x in ("área urbana", "españa")), t[:200])
        check("ES: popup 'show all' button translated", "Mostrar las" in t and "estadísticas" in t, t[:300])
        pg.click(".cpBtn")
        pg.wait_for_timeout(600)
        t = pg.evaluate("[...document.querySelectorAll('.maplibregl-popup')].map(e=>e.innerText).join('|')")
        check("ES: census column names translated", "Población masculina" in t and "Hogares" in t and "Male population" not in t, t[:300])
        # back to English
        pg.click("#lang-current")
        pg.click("#lang-opt-en")
        pg.wait_for_timeout(500)
        t = body(pg)
        check("EN: everything restored", pg.inner_text("#lang-current") == "EN" and "Legend" in t and "Leyenda" not in t and "Población" not in t, t[:200])
        stored = pg.evaluate("localStorage.getItem('spain_map_lang')")
        check("choice stored in localStorage", stored == "en", str(stored))
        # remembered choice beats the browser language (English browser, user picked ES)
        pg.click("#lang-current")
        pg.click("#lang-opt-es")
        pg.wait_for_timeout(300)
        pg.reload()
        pg.wait_for_selector("#lang-current")
        pg.wait_for_timeout(2500)
        check("saved choice survives reload and beats the browser language", pg.inner_text("#lang-current") == "ES" and "Leyenda" in body(pg))
        pg.screenshot(path=str(Path(tempfile.gettempdir()) / "language_switch_test.png"))
        ctx.close()
        # --- localStorage blocked: must still work ---
        ctx = b.new_context(locale="en-US", viewport={"width": 1400, "height": 900})
        ctx.route(re.compile(r"api\.country\.is|geojs\.io"), geo(None))
        ctx.add_init_script("Object.defineProperty(window,'localStorage',{get(){throw new Error('blocked')}})")
        pg = ctx.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(URL)
        pg.wait_for_selector("#lang-current")
        pg.wait_for_timeout(2500)
        check("works with localStorage blocked (browser language, no page error)", "Legend" in body(pg) and not [e for e in errs if "blocked" in e], str(errs))
        ctx.close()
        ctx, pg = page(b, None, "en-US")
        pg.goto(URL + "?lang=es"); pg.wait_for_selector("#lang-current"); pg.wait_for_timeout(2500)
        check("?lang=es forces Spanish on an English browser", pg.inner_text("#lang-current") == "ES")
        ctx.close()
        check("no request to any geo-IP service", not GEO_CALLS, str(GEO_CALLS))
        b.close()
finally:
    srv.terminate()
print("\nFAILED: " + ", ".join(fails) if fails else "\nALL PASSED")
sys.exit(1 if fails else 0)
