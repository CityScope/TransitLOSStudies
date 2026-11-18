"""Test a built map before deploying it: serve its folder locally, open map.html in Chromium, fail on page errors, console errors or failed
requests (404/5xx) other than the known-optional ones, and on a blank map (no canvas drawn). Saves a screenshot.

    python3 test_map.py SERVE_ROOT PAGE_PATH OUT_PNG [--wait 25]
    e.g. test_map.py .../city_science_network andorra/map.html /tmp/andorra.png
Prints MAP_TEST_OK or MAP_TEST_FAIL with the reasons; exit code 0 / 1.
"""
import functools
import http.server
import socket
import sys
import threading
import time

from playwright.sync_api import sync_playwright

root, page_path, out_png = sys.argv[1], sys.argv[2], sys.argv[3]
wait = float(sys.argv[sys.argv.index("--wait") + 1]) if "--wait" in sys.argv else 25.0
OPTIONAL = ("world_quantiles.json", "favicon.ico", "combined_map", ".build", "/fonts/")   # site-root files / basemap glyphs   # served from the site root / optional


class Quiet(http.server.SimpleHTTPRequestHandler):
    """Static server with HTTP Range support (PMTiles read byte ranges)."""

    def log_message(self, *a):
        pass

    def send_head(self):
        rng = self.headers.get("Range")
        if not rng or not rng.startswith("bytes="):
            return super().send_head()
        import os
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        a, _, z = rng[6:].split(",")[0].partition("-")
        start = int(a) if a else max(0, size - int(z))
        end = min(int(z), size - 1) if (a and z) else size - 1
        f = open(path, "rb"); f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        n = getattr(self, "_remaining", None)
        if n is None:
            return super().copyfile(source, outputfile)
        while n > 0:
            b = source.read(min(1 << 20, n))
            if not b:
                break
            outputfile.write(b); n -= len(b)
        self._remaining = None


s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Quiet, directory=root))
threading.Thread(target=srv.serve_forever, daemon=True).start()

errors, failed, console = [], [], []
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": 1500, "height": 950})
    pg.on("pageerror", lambda e: errors.append(str(e)[:300]))
    pg.on("console", lambda m: console.append(m.text[:300]) if m.type == "error" else None)
    pg.on("response", lambda r: failed.append(f"{r.status} {r.url}") if r.status >= 400 and not any(o in r.url for o in OPTIONAL) else None)
    # an aborted request of the page itself is its one self-reload (adds the build id ?v=), not a failure
    pg.on("requestfailed", lambda r: failed.append(f"failed {r.url}") if not any(o in r.url for o in OPTIONAL) and not r.url.split("?")[0].endswith(".html")
          and "ERR_ABORTED" not in (r.failure or "") else None)
    pg.goto(f"http://127.0.0.1:{port}/{page_path}", wait_until="load", timeout=180000)
    time.sleep(wait)
    drawn = pg.evaluate("""() => { const c = document.querySelector('canvas'); if (!c) return -1;
        try { const g = document.createElement('canvas'); g.width = 64; g.height = 64; const x = g.getContext('2d'); x.drawImage(c, 0, 0, 64, 64);
              const d = x.getImageData(0, 0, 64, 64).data; let v = new Set(); for (let i = 0; i < d.length; i += 4) v.add(d[i] + ',' + d[i+1] + ',' + d[i+2]);
              return v.size; } catch (e) { return -2; } }""")
    pg.screenshot(path=out_png)
    b.close()
srv.shutdown()
console = [c for c in console if not any(o in c for o in OPTIONAL) and "Failed to load resource" not in c]
bad = errors + console + failed
from PIL import Image
im = Image.open(out_png).convert("RGB").resize((300, 190))
px = list(im.getdata())
vivid = sum(1 for r, g, b2 in px if max(r, g, b2) - min(r, g, b2) > 60)   # LOS colours (the basemap is grey)
if drawn == -1:
    bad.append("no map canvas")
if vivid < 40:
    bad.append(f"map looks empty ({vivid} coloured pixels in the screenshot)")
print(f"coloured pixels {vivid}")
print(f"canvas colours {drawn}; page errors {len(errors)}; console errors {len(console)}; failed requests {len(failed)}")
for x in bad[:15]:
    print("  ", x)
print("MAP_TEST_OK" if not bad else "MAP_TEST_FAIL")
sys.exit(0 if not bad else 1)
