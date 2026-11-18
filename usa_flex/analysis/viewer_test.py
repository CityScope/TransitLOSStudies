"""End-to-end test of the map (headless Chromium + WebGL): load a state, drive the scoring editors, compare every score and catchment with Python, click a stop and a cell.

    python analysis/viewer_test.py Vermont
"""
import json
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import os as _os
WEB = HERE / ("web_buf" if _os.environ.get("FLEX_VARIANT") == "buf" else "web")   # FLEX_VARIANT=buf: the deployed stop-buffer map
import run_county as rc  # noqa: E402
from transitlos.flex import core, scoring as sc, shapes as sh, catchment as ct  # noqa: E402


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a, **k):
        pass


def python_side(name):
    web = WEB / name
    meta = json.loads((web / "meta.json").read_text())
    S, P, K, nc = meta["S"], meta["P"], meta["K"], meta["nc"]
    mode = np.fromfile(web / "stop_mode.bin", np.int8).reshape(S, P); h = np.fromfile(web / "stop_h.bin", np.float32).reshape(S, P); v = np.fromfile(web / "stop_v.bin", np.float32).reshape(S, P)
    prof = core.Profiles(mode, h, v)
    cs = np.fromfile(web / "cell_station.bin", np.int32).reshape(nc, K); cd = np.fromfile(web / "cell_dist.bin", np.uint16).reshape(nc, K)
    counts = np.fromfile(web / "cell_counts.bin", np.float32).reshape(nc, len(meta["count_columns"]))
    return meta, prof, cs, cd, counts


def main(name="Vermont"):
    meta, prof, cs, cd, counts = python_side(name)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(WEB)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    from playwright.sync_api import sync_playwright
    ok = True
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 1500, "height": 900})
        errs, missing = [], []
        pg.on("pageerror", lambda e: errs.append(str(e))); pg.on("response", lambda r: missing.append(r.url) if r.status == 404 else None); pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        t0 = time.time()
        pg.goto(f"http://127.0.0.1:{srv.server_port}/viewer.html?state={name}")
        pg.evaluate("window.__ready")
        print(f"loaded {name}: {meta['S']} stations, {meta['nc']} cells in {time.time()-t0:.1f}s; JS errors so far: {errs}")

        def check(label, params_py, apply_js):
            nonlocal ok
            pg.evaluate(apply_js)
            js = np.array(pg.evaluate("__api.los()"), dtype=np.float64)
            los, slot, _ = core.recompute_cells_fast(cs, cd, prof, params_py)
            d = float(np.abs(js - los).max())
            win = ct.winner_station(cs, slot)
            tot = ct.stop_catchment(win, counts, prof.S)
            jt = np.array(pg.evaluate("__api.tot()")).reshape(prof.S, -1)
            dt = float(np.abs(jt - tot).max() / max(tot.max(), 1))
            same_win = bool((np.array(pg.evaluate("__api.slot()")) == slot).mean() > 0.9999)
            good = d < 1e-3 and dt < 1e-6 and same_win
            ok &= good
            print(f"  {'OK ' if good else 'BAD'} {label:60s} max|score diff| {d:.1e}   catchment rel diff {dt:.1e}   same winners {same_win}")

        # 1. default
        check("default scoring", sc.DEFAULT, "__api.setParams(JSON.parse(JSON.stringify(__api.meta().params)))")
        # 2. drive the editors like a user: pick a family from each drop-down, parameters at their slider defaults
        pg.evaluate("__api.buildEditors()")
        sels = pg.query_selector_all("#editors .ed select")
        print("  editor drop-downs found:", len(sels))
        def pick(i, value):
            sels[i].select_option(value); pg.wait_for_timeout(120)
        pick(1, "exponential")        # headway curve family (index 0 = aggregation)
        py = sc.DEFAULT.with_(headway_curve=sh.headway_exponential(5, 40))
        pg.evaluate("__api.schedule()")
        check("headway: exponential (editor drop-down)", py, "0")
        pick(2, "logistic")           # speed curve
        py = py.with_(speed_curve=sh.speed_logistic(20, 0.2, 1.0)); pg.evaluate("__api.schedule()")
        check("+ speed: logistic", py, "0")
        pick(3, "gaussian")           # walk curve
        py = py.with_(walk_curve=sh.walk_gaussian(1.0)); pg.evaluate("__api.schedule()")
        check("+ walk: gaussian", py, "0")
        sels[0].select_option("power:-1"); pg.wait_for_timeout(120)
        py = py.with_(agg="power_mean", agg_p=-1.0); pg.evaluate("__api.schedule()")
        check("+ harmonic aggregation", py, "0")
        # weights: set mode weight to 0 (axis off)
        pg.evaluate("(()=>{const p=__api.getParams(); p.w_mode=0; __api.setParams(p)})()")
        check("+ mode switched OFF (weight 0)", py.with_(w_mode=0.0), "0")
        # 3. custom points: edit through the API-compatible path (drag simulated by changing the family's points)
        pg.evaluate("__api.setParams(JSON.parse(JSON.stringify(__api.meta().params)))")
        # 4. clicks
        # population each station wins under the default scoring, computed here (the buffer variant writes no stop_catchment.parquet)
        _los, _slot, _ = core.recompute_cells_fast(cs, cd, prof, sc.DEFAULT)
        cat = pd.DataFrame({"population": ct.stop_catchment(ct.winner_station(cs, _slot), counts[:, 0], prof.S)[:, 0]})
        s = int(cat["population"].to_numpy().argmax())
        pg.evaluate("__api.setSize(0)")
        x, y = pg.evaluate(f"__api.stopXY({s})")
        pg.evaluate(f"__api.click({x}, {y})")
        txt = pg.evaluate("__api.popup()")
        want = f"{int(round(cat['population'].iloc[s])):,}"
        hit = pg.evaluate("__api.popupKind()") == f"stop:{s}" and want in txt
        ok &= hit
        print(f"  {'OK ' if hit else 'BAD'} click on the biggest stop (#{s}): popup shows population {want}: {'yes' if want in txt else 'NO'}")
        print("    popup:", txt.replace("\n", " | ")[:260])
        c = int(np.argmax(counts[:, 0]))
        x, y = pg.evaluate(f"__api.cellXY({c})"); pg.evaluate(f"__api.click({x}, {y})")
        print("    cell popup:", pg.evaluate("__api.popup()").replace("\n", " | ")[:200])
        print("    legend:", pg.evaluate("__api.legend()").replace("\n", " | ")[:200])
        print("    stats:", pg.evaluate("__api.stats()").replace("\n", " | "))
        pg.screenshot(path=str(WEB / f"screenshot_{name}.png"))
        print("404:", missing); print("JS errors:", errs, "| ALL OK" if ok and not errs else "| PROBLEMS")
        b.close()
    srv.shutdown()


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["Vermont"]))
