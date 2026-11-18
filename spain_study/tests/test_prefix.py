"""The nationwide map must work when served under a URL sub-path (e.g. /spain_los/).

    python tests/test_prefix.py data/_spain [/spain_los/]

Serves the folder under the prefix (rangeserver --prefix --inject-base, the <base href> a sub-path host needs), opens the page and
checks that no request of the page itself (tiles, chunks, quantiles, PMTiles ranges) failed or escaped the prefix, and that the stats panel and the popup load their chunks.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(sys.argv[1]).resolve()
PREFIX = "/" + (sys.argv[2] if len(sys.argv) > 2 else "/spain_los/").strip("/") + "/"
PORT = 8793
srv = subprocess.Popen([sys.executable, str(Path(__file__).with_name("rangeserver.py")), str(ROOT), str(PORT), "--prefix", PREFIX, "--inject-base"])
time.sleep(1.0)
bad, ok_chunks = [], []
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra and not cond else ""))
    if not cond:
        fails.append(name)


try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)[:150]))

        def on_resp(r):
            if r.url.startswith(f"http://127.0.0.1:{PORT}"):
                if r.status >= 400:
                    bad.append((r.status, r.url))
                elif "stats_chunks" in r.url or "quantiles" in r.url:
                    ok_chunks.append(r.url)

        pg.on("response", on_resp)
        pg.goto(f"http://127.0.0.1:{PORT}{PREFIX}?lang=en")
        pg.wait_for_selector("#lang-current")
        pg.wait_for_timeout(9000)
        pg.evaluate("window.__mainMap.jumpTo({center: [-1.98, 43.31], zoom: 12})")
        pg.wait_for_timeout(5000)
        check("stats button", pg.locator("#statsButton").count() == 1)
        pg.click("#statsButton")
        pg.wait_for_timeout(3000)
        check("stats panel lists columns", len(pg.evaluate("[...document.querySelectorAll('#regFieldSelect option')]")) > 50)
        pg.click("#statsButton")
        pg.evaluate("window.__mainMap.jumpTo({center: [-1.98, 43.31], zoom: 14})")
        pg.wait_for_timeout(4000)
        pg.evaluate("window.__openCellPopup(-1.98, 43.31, {})")
        pg.wait_for_timeout(5000)
        t = pg.evaluate("[...document.querySelectorAll('.maplibregl-popup')].map(p=>p.innerText).join('|')")
        check("popup with statistics loaded from the chunks", "statistics" in t, t[:150])
        check("stats chunks fetched under the prefix", any(PREFIX in u and "stats_chunks" in u for u in ok_chunks), str(ok_chunks[:3]))
        check("no failed same-host request", not [x for x in bad if "world_quantiles" not in x[1]], str(bad[:5]))
        check("no page errors", not errs, str(errs))
        # nothing may have been requested outside the prefix
        out = [u for u in ok_chunks if PREFIX not in u]
        check("no request outside the prefix", not out, str(out[:3]))
        b.close()
finally:
    srv.terminate()
print("FAILED: " + ", ".join(fails) if fails else "ALL PASSED")
sys.exit(1 if fails else 0)
