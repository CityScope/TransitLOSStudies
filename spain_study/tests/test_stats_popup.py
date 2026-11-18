"""Headless check of the stats panel and the click popup, EN and ES.

    python tests/test_stats_popup.py data/PaisVasco/map.html
"""
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HTML = Path(sys.argv[1]).resolve()
PORT = 8792
srv = subprocess.Popen([sys.executable, str(Path(__file__).with_name("rangeserver.py")), str(HTML.parent), str(PORT)])
time.sleep(1.0)
URL = f"http://127.0.0.1:{PORT}/{HTML.name}"
CENTER = [-1.98, 43.31]
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra and not cond else ""))
    if not cond:
        fails.append(name)


def options(pg, sel):
    return pg.evaluate("(s)=>[...document.querySelectorAll('#'+s+' option')].map(o=>o.text.trim())", sel)


try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        for lang in ("en", "es"):
            pg = b.new_page(viewport={"width": 1400, "height": 900})
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)[:150]))
            pg.goto(f"{URL}?lang={lang}")
            pg.wait_for_selector("#lang-current")
            pg.wait_for_timeout(9000)
            pg.evaluate(f"window.__mainMap.jumpTo({{center:{CENTER},zoom:14}})")
            pg.wait_for_timeout(4000)
            L = lang.upper()
            check(f"{L} stats button", pg.locator("#statsButton").count() == 1)
            pg.click("#statsButton")
            pg.wait_for_timeout(1500)
            dist, reg = options(pg, "distMainSelect"), options(pg, "regFieldSelect")
            check(f"{L} distribution list has the census counts", len(dist) >= 20, str(len(dist)))
            check(f"{L} regression list has the relative census columns", len(reg) >= 60, str(len(reg)))
            check(f"{L} municipality natives are regression variables, not distribution bars",
                  any(re.search(r"turístic|tourist", o, re.I) for o in reg) and not any(re.search(r"turístic|tourist|fianza|deposit|migrat|tasado|apprais", o, re.I) for o in dist))
            check(f"{L} stats labels translated", (any("Renta" in o for o in reg) if lang == "es" else any("income" in o.lower() for o in reg)), str(reg[:3]))
            check(f"{L} P80/P20 label has no percent sign", not any(re.search(r"P80/P20.*%", o) for o in reg))
            pg.click("#statsButton")
            pg.wait_for_timeout(500)
            circle = options(pg, "circleFieldSelect")
            check(f"{L} no duplicate circle entries", len(circle) == len(set(circle)), str(circle))
            check(f"{L} no municipality natives as circle size", not any(re.search(r"turístic|tourist|fianza|deposit|migrat|tasado|apprais", o, re.I) or "EMAL" in o for o in circle), str(circle))
            check(f"{L} clear SERPAVI contracts label", any(("Rental contracts in SERPAVI sample, flats, 2024" if lang == "en" else "Contratos de alquiler en la muestra SERPAVI, pisos, 2024") == o for o in circle), str(circle))
            pg.evaluate(f"window.__openCellPopup({CENTER[0]}, {CENTER[1]}, {{}})")   # a populated spot, whatever the start view
            pg.wait_for_timeout(5000)
            pg.click(".cpBtn")
            pg.wait_for_timeout(800)
            txt = pg.evaluate("document.querySelector('.cellPopup') ? document.querySelector('.cellPopup').innerText : ''")
            check(f"{L} popup shows many statistics", len(re.findall(r"\n", txt)) > 40, txt[:200])
            check(f"{L} population is rounded", not re.search(r"\d\.\d{4,}", txt))
            check(f"{L} percentages and euros", "%" in txt and "€" in txt, txt[:300])
            ratio = re.search(r"(?:P80/P20)[^\n]*\n?\s*([0-9][0-9.,]*)([^\n]*)", txt)
            check(f"{L} P80/P20 shown as a plain ratio", bool(ratio) and "%" not in ratio.group(2)[:3] and float(ratio.group(1).replace(",", ".")) < 10, ratio.group(0) if ratio else "row missing")
            note = ("Values marked (municipality)" if lang == "en" else "Los valores marcados (municipio)")
            check(f"{L} native-resolution note", note in txt, txt[-300:])
            check(f"{L} thousands separator", re.search(r"\d[.,   ]\d{3}\b", txt) is not None)
            pg.screenshot(path=str(Path(tempfile.gettempdir()) / f"stats_popup_{lang}.png"))
            check(f"{L} no page errors", not errs, str(errs))
            pg.close()
        b.close()
finally:
    srv.terminate()
print("FAILED: " + ", ".join(fails) if fails else "ALL PASSED")
sys.exit(1 if fails else 0)
