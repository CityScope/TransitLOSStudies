"""Small sample of the US for the web map: every scoring edit re-estimates how well the score explains the ACS transit share (R^2 and loss) for the nation and for a metro.

    python fit/build_sample.py

Writes web/sample/ (same binary layout as a state, plus block-group tables) and web/sample/presets.json (literature, US fitted, one fit per metro).
Block groups are drawn at random, stratified (by state for the nation, by CBSA for each metro); every cell of a drawn block group is kept, so its
score is sum(cell population x cell score) / census population (cells out of reach count 0).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import fit2  # noqa: E402
import fit_scoring as fs  # noqa: E402
import run_county as rc  # noqa: E402
from transitlos.flex import scoring as sc  # noqa: E402

OUT = ROOT / "web" / "sample"
K, CELLS, N_NATION, N_METRO = 10, None, 3000, 300   # CELLS None: every cell of a sampled block group (its score is then the exact population-weighted mean)
NATIONAL_FITS = [HERE / "data" / f for f in ("g_param_nested_s0.json", "g_param_nested_s1.json", "w_param_nested_s0.json", "w_param_nested_s1.json")]   # analytic, nested: editable in the map
METRO_FIT = HERE / "data" / ("g_metros_param_nested.json" if (HERE / "data" / "g_metros_param_nested.json").exists() else "m_param_nested_wls.json")


def main():
    if "--presets-only" in sys.argv:
        return presets(json.load(open(METRO_FIT)))
    rng = np.random.default_rng(7)
    OUT.mkdir(parents=True, exist_ok=True)
    bgt = pd.read_parquet(HERE / "data" / "blockgroups.parquet")
    states = [a for a in rc.STATE_DIRS if (ROOT / "web" / rc.STATE_DIRS[a] / "meta.json").is_file()]
    ncell = np.zeros(len(bgt), np.int64)
    cellpop = np.zeros(len(bgt))
    for a in states:
        z = np.load(HERE / "data" / f"cells_{rc.STATE_DIRS[a]}.npz")
        ok = z["bg"] >= 0
        ncell += np.bincount(z["bg"][ok], minlength=len(bgt))
        cellpop += np.bincount(z["bg"][ok], weights=z["pop"][ok].astype(float), minlength=len(bgt))
    pop = bgt.population.fillna(0).to_numpy().astype(float)
    wk = bgt.acs5_workers.fillna(0).to_numpy().astype(float)
    tr = bgt.publicTransportCommuters.to_numpy().astype(float)
    elig = (ncell > 0) & (pop > 0) & (wk > 0) & ~np.isnan(tr)
    print("eligible block groups", int(elig.sum()), "of", len(bgt), flush=True)

    # ------------------------------------------------------------------ sets: (name, cbsa, rows, weights)
    sets = []
    nat_rows, nat_w = [], []
    stt = bgt.state.to_numpy()
    for a in sorted(set(stt[elig])):
        rows = np.flatnonzero(elig & (stt == a))
        n = int(min(len(rows), max(15, round(N_NATION * len(rows) / elig.sum()))))
        pick = rng.choice(rows, n, replace=False)
        nat_rows.append(pick)
        nat_w.append(np.full(n, len(rows) / n))
    sets.append({"name": "United States", "cbsa": "", "rows": np.concatenate(nat_rows), "w": np.concatenate(nat_w)})
    mf = json.load(open(METRO_FIT))
    cb = bgt.cbsa.fillna("").to_numpy()
    for m in mf["metros"]:
        rows = np.flatnonzero(elig & (cb == str(m["cbsa"])))
        n = min(len(rows), N_METRO)
        pick = rng.choice(rows, n, replace=False)
        sets.append({"name": m["name"], "cbsa": str(m["cbsa"]), "rows": pick, "w": np.full(n, len(rows) / n)})
    used = np.unique(np.concatenate([s["rows"] for s in sets]))
    print("block groups in the sample", len(used), flush=True)
    row_of = np.full(len(bgt), -1, np.int64)
    row_of[used] = np.arange(len(used))

    # ------------------------------------------------------------------ cells of those block groups, state by state
    cells = {k: [] for k in ("bg", "pop", "st", "dist", "lon", "lat", "state")}
    prof_parts, stn_off = [], 0
    for a in states:
        name = rc.STATE_DIRS[a]
        z = np.load(HERE / "data" / f"cells_{name}.npz")
        bgc = z["bg"]
        sel = np.flatnonzero((bgc >= 0) & (row_of[np.maximum(bgc, 0)] >= 0))
        if len(sel) == 0:
            continue
        sel = sel[np.argsort(bgc[sel], kind="stable")]
        if CELLS:   # optional cap (random cells per block group)
            order = rng.permutation(len(sel)); sel = sel[order]; sel = sel[np.argsort(bgc[sel], kind="stable")]
            b = bgc[sel]
            start = np.r_[0, np.flatnonzero(np.diff(b)) + 1]
            rank = np.arange(len(sel)) - np.repeat(start, np.diff(np.r_[start, len(sel)]))
            sel = sel[rank < CELLS]
        st = fs.State(a, bgt)
        meta = json.load(open(ROOT / "web" / name / "meta.json"))
        geom = np.fromfile(ROOT / "web" / name / "cell_geom.bin", np.float32).reshape(meta["nc"], 14)
        cst, cd = st.cst[sel], st.cdist[sel].astype(np.int64)
        o = np.argsort(np.where(cst >= 0, cd, 70000), axis=1, kind="stable")[:, :K]
        cst, cd = np.take_along_axis(cst, o, 1), np.take_along_axis(cd, o, 1)
        usedst = np.unique(cst[cst >= 0])
        remap = np.full(meta["S"], -1, np.int64)
        remap[usedst] = stn_off + np.arange(len(usedst))
        cst = np.where(cst >= 0, remap[np.maximum(cst, 0)], -1)
        npp = np.fromfile(ROOT / "web" / name / "stop_np.bin", np.int32)
        prof_parts.append((st.prof.mode[usedst], st.prof.h[usedst], st.prof.v[usedst], npp[usedst]))
        stn_off += len(usedst)
        cells["bg"].append(row_of[bgc[sel]]); cells["pop"].append(z["pop"][sel]); cells["st"].append(cst.astype(np.int32))
        cells["dist"].append(np.minimum(cd, 65535).astype(np.uint16)); cells["lon"].append(geom[sel, 0]); cells["lat"].append(geom[sel, 1])
        print(a, len(sel), "cells", flush=True)
    bg = np.concatenate(cells["bg"]); o = np.argsort(bg, kind="stable")
    cat = lambda k: np.concatenate(cells[k])[o]
    bg, cpop, cst, cdist, lon, lat = bg[o], cat("pop").astype(np.float32), cat("st"), cat("dist"), cat("lon"), cat("lat")
    mode = np.concatenate([p[0] for p in prof_parts]).astype(np.int8)
    h = np.concatenate([p[1] for p in prof_parts]).astype(np.float32)
    v = np.concatenate([p[2] for p in prof_parts]).astype(np.float32)
    npr = np.concatenate([p[3] for p in prof_parts]).astype(np.int32)
    nc, S, P = len(bg), len(npr), mode.shape[1]
    print("sample cells", nc, "stations", S, flush=True)

    # block-group table: population, workers, transit commuters, coverage (people in cells / population)
    cov = np.minimum(cellpop[used] / np.maximum(pop[used], 1.0), 1.5)
    bgtab = np.column_stack([pop[used], wk[used], tr[used], cov]).astype(np.float32)
    set_bg = np.concatenate([row_of[s["rows"]] for s in sets]).astype(np.int32)
    set_w = np.concatenate([s["w"] for s in sets]).astype(np.float32)
    off = np.r_[0, np.cumsum([len(s["rows"]) for s in sets])]

    mode.tofile(OUT / "stop_mode.bin"); h.tofile(OUT / "stop_h.bin"); v.tofile(OUT / "stop_v.bin"); npr.tofile(OUT / "stop_np.bin")
    cst.astype(np.int32).tofile(OUT / "cell_station.bin"); cdist.tofile(OUT / "cell_dist.bin"); cpop.tofile(OUT / "cell_pop.bin"); bg.astype(np.int32).tofile(OUT / "cell_bg.bin")
    bgtab.tofile(OUT / "bg.bin"); set_bg.tofile(OUT / "set_bg.bin"); set_w.tofile(OUT / "set_w.bin")

    # metro extent: the middle 96% of the sampled cell centres
    def bbox(rows):
        m = np.isin(bg, row_of[rows])
        return [float(np.percentile(lon[m], 2)), float(np.percentile(lat[m], 2)), float(np.percentile(lon[m], 98)), float(np.percentile(lat[m], 98))]
    meta = {"nc": nc, "S": S, "P": P, "K": K, "nbg": len(used),
            "sets": [{"name": s["name"], "cbsa": s["cbsa"], "o": int(off[i]), "n": int(len(s["rows"])), **({"bbox": bbox(s["rows"])} if i else {})} for i, s in enumerate(sets)]}
    (OUT / "meta.json").write_text(json.dumps(meta))

    presets(mf)
    print("done", sum(f.stat().st_size for f in OUT.iterdir()) / 1e6, "MB", flush=True)


def presets(mf):
    def entry(name, model, x, extra=None):
        p = model.params(np.asarray(x, float))
        return {"name": name, "params": json.loads(p.to_json()), "gamma": model.gamma(np.asarray(x, float)), **(extra or {})}
    M = fit2.Model("param", "nested")
    best = min((r for f in NATIONAL_FITS if f.exists() for r in json.load(open(f))["runs"]), key=lambda r: r["loss"])
    print("US fitted: full-sample loss", best["loss"], "| metro fits from", METRO_FIT.name, flush=True)
    presets = [{"name": "Literature (default)", "params": json.loads(sc.DEFAULT.to_json()), "gamma": 1.0},
               entry("US fitted", M, best["x"])]
    for i, m in enumerate(mf["metros"]):
        presets.append(entry(m["name"], M, m["x"], {"metro": i + 1, "own_r2": 1 - m["loss_own_fit"] / m["null"]}))
    (OUT / "presets.json").write_text(json.dumps(presets))
    import time
    (OUT / "version.txt").write_text(time.strftime("%Y%m%d%H%M%S"))   # the browser fetches this one uncached and puts it in the URLs of the sample files
    print("presets written", flush=True)


if __name__ == "__main__":
    main()
