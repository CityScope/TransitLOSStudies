"""Flat (cell, station, service pattern) items for the GPU fits (fit/torch_fit.py), every cell with residents or jobs of every block group in the US.

    python fit/build_items.py
Each item: cell, station, mode (0 bus, 1 tram, 2 rail), headway (min), speed (km/h), walking distance (m). The K=15 nearest stations of a cell come with their network
walking distance. A rail or tram station beyond those 15 (a dense bus grid hides it) is added too, with the straight-line distance times the detour factor of that state
(the median network / straight-line ratio of the cells' own items), up to 3 km, so long walks to rail and tram are not lost.
Far access (park-and-ride, feeder buses, bikes): in each band of default station score (best profile's default stop score, FAR_BANDS, any mode) the
nearest station up to FAR_MAX_M away (straight line x detour) is reachable too -- a strong station, and also the weaker ones park-and-ride riders often
use (an hourly commuter-rail stop scores low by default). From every stored cell (bands' stations not among its own K), and from the populated cells no
stop reaches on foot (no stored cell; built here from the census blocks, written to fit/data/far_cells_<State>.npz so that fit/build_blocks.py points
their blocks at them). The walking curve then decides how much a 1-20 km trip to each is worth (beyond ~2 km it stands for driving,
feeder buses and bikes too, not walking).
Writes fit/data/torch/{cells,items}.npz."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import run_county as rc  # noqa: E402

import paths as PATHS  # noqa: E402
WEB = PATHS.WEB
OUT = PATHS.DATA / "torch"
EXTRA_MAX_M = 3000.0
FAR_BANDS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 9.0)]   # default stop score: 0.5 ~ the median US station, >= 0.8 the top ~1%
FAR_MAX_M = 20000.0   # park-and-ride reach: 1 km knots to 5 km, then 5 km knots (torch_fit.D_KNOTS)
STUDY = HERE.parents[1] / "usa_study" / "data"


def xy(lon, lat, lat0):
    return np.column_stack([np.radians(lon) * 6371000.0 * np.cos(np.radians(lat0)), np.radians(lat) * 6371000.0])


def one_state(name):
    import json
    w = WEB / name
    if not (w / "meta.json").is_file():
        return None
    meta = json.load(open(w / "meta.json"))
    S, P, nc, K = meta["S"], meta["P"], meta["nc"], meta["K"]
    mode = np.fromfile(w / "stop_mode.bin", np.int8).reshape(S, P)
    h = np.fromfile(w / "stop_h.bin", np.float32).reshape(S, P)
    v = np.fromfile(w / "stop_v.bin", np.float32).reshape(S, P)
    cst = np.fromfile(w / "cell_station.bin", np.int32).reshape(nc, K)
    cd = np.fromfile(w / "cell_dist.bin", np.uint16).reshape(nc, K).astype(np.float32)
    geom = np.fromfile(w / "cell_geom.bin", np.float32).reshape(nc, 14)
    sll = np.fromfile(w / "stop_lonlat.bin", np.float32).reshape(S, 2)
    z = np.load(PATHS.DATA / f"cells_{name}.npz")
    bg, pop = z["bg"], z["pop"].astype(np.float32)
    jobs = np.zeros(nc, np.float32)
    if "lodes_wac_jobsTotal" in meta.get("count_columns", []):   # jobs per cell (web/<state>/cell_counts.bin): destination access is job-weighted
        cc = np.fromfile(w / "cell_counts.bin", np.float32).reshape(nc, len(meta["count_columns"]))
        jobs = cc[:, meta["count_columns"].index("lodes_wac_jobsTotal")]
    keep = (bg >= 0) & ((pop > 0) | (jobs > 0))
    cells = np.flatnonzero(keep)
    lon, lat = geom[cells, 0], geom[cells, 1]
    lat0 = float(np.median(lat)) if len(lat) else 40.0
    # items from the K nearest stations
    cs, dd = cst[cells], cd[cells]
    ci, kk = np.nonzero(cs >= 0)
    st = cs[ci, kk]; dist = dd[ci, kk]
    pm = mode[st] >= 0                                     # (n, P) patterns that exist
    ii, jj = np.nonzero(pm)
    it_cell, it_st, it_j, it_d = ci[ii], st[ii], jj, dist[ii]
    # detour factor: network / straight line on these items (distances above 100 m)
    pc = xy(lon, lat, lat0); ps = xy(sll[:, 0], sll[:, 1], lat0)
    straight = np.linalg.norm(pc[ci] - ps[st], axis=1)
    okr = (dist > 100) & (straight > 50)
    detour = float(np.median(dist[okr] / straight[okr])) if okr.any() else 1.3
    extra = []
    for m in (1, 2):   # tram, rail
        has = (mode == m).any(1)
        sid = np.flatnonzero(has)
        if not len(sid):
            continue
        tree = cKDTree(ps[sid])
        dst, ix = tree.query(pc, k=1, distance_upper_bound=EXTRA_MAX_M / detour)
        okc = np.isfinite(dst)
        cand_cell = np.flatnonzero(okc); cand_st = sid[ix[okc]]
        inK = (cs[cand_cell] == cand_st[:, None]).any(1)
        cand_cell, cand_st, cand_d = cand_cell[~inK], cand_st[~inK], dst[okc][~inK] * detour
        pmm = mode[cand_st] == m
        a, j = np.nonzero(pmm)
        extra.append((cand_cell[a], cand_st[a], j, cand_d[a].astype(np.float32)))
    # far access to the stations of very high default score
    from transitlos.flex import scoring as sc
    with np.errstate(invalid="ignore"):
        pss = np.where(mode >= 0, sc.stop_score(np.maximum(mode, 0), np.maximum(h, 0.5), np.maximum(v, 0.1)), 0.0)
    best_ss = pss.max(1)
    bands = [np.flatnonzero((best_ss >= lo) & (best_ss < hi)) for lo, hi in FAR_BANDS]
    hi_st = np.flatnonzero(best_ss >= FAR_BANDS[0][0])
    far_ids = np.zeros(0, np.uint64); far = dict(bg=np.zeros(0, np.int32), pop=np.zeros(0, np.float32), lon=np.zeros(0, np.float32), lat=np.zeros(0, np.float32))
    n_far_items = 0
    if len(hi_st):
        trees = [(cKDTree(ps[b]), b) for b in bands if len(b)]
        def far_items(cell_rows, pts, own=None):
            out = []
            for tree, b in trees:   # the nearest station of each score band
                dst, ix = tree.query(pts, k=1, distance_upper_bound=FAR_MAX_M / detour)
                r = np.flatnonzero(np.isfinite(dst)); stn = b[ix[r]]; dd = dst[r]
                if own is not None:   # skip stations already among the cell's own items (its K nearest, by network distance)
                    inK = (own[r] == stn[:, None]).any(1); r, stn, dd = r[~inK], stn[~inK], dd[~inK]
                a, j = np.nonzero(mode[stn] >= 0)
                out.append((cell_rows[r[a]], stn[a], j, (dd[a] * detour).astype(np.float32)))
            return tuple(np.concatenate([o[i] for o in out]) for i in range(4))
        if PATHS.BUF:   # buffer variant: far access by road (run_state_buffered.py far.parquet: band x bus/tram, rail; up to 15 per cell)
            extra.append(far_from_table(name, cells, mode, pss))
        else:
            extra.append(far_items(np.arange(len(cells)), pc, own=cs))
        n_far_items += len(extra[-1][0])
        fc = None if PATHS.BUF else far_cells(name, w, nc, cells, geom, hi_st, ps, lat0, detour)   # buffer variant: every populated cell is stored already
        if fc is not None:
            far_ids, far = fc
            fpc = xy(far["lon"], far["lat"], lat0)
            e = far_items(np.arange(len(far_ids)) + len(cells), fpc)
            extra.append(e); n_far_items += len(e[0])
    np.savez(PATHS.DATA / f"far_cells_{name}.npz", h3=far_ids)
    n_extra = 0
    for e in extra:
        it_cell = np.r_[it_cell, e[0]]; it_st = np.r_[it_st, e[1]]; it_j = np.r_[it_j, e[2]]; it_d = np.r_[it_d, e[3]]; n_extra += len(e[0])
    with np.errstate(divide="ignore"):
        freq = np.where(mode >= 0, 60.0 / np.maximum(h, 0.5), 0.0).sum(1)          # departures per hour of all patterns of the station
    hc = np.where(freq > 0, 60.0 / np.maximum(freq, 1e-9), 1440.0).astype(np.float32)
    return {"name": name, "bg": np.r_[bg[cells].astype(np.int32), far["bg"]], "hc": hc[it_st], "pop": np.r_[pop[cells], far["pop"]],
            "jobs": np.r_[jobs[cells], np.zeros(len(far_ids), np.float32)], "lon": np.r_[lon.astype(np.float32), far["lon"]], "lat": np.r_[lat.astype(np.float32), far["lat"]],
            "cell": it_cell.astype(np.int32), "st": it_st.astype(np.int32), "mode": mode[it_st, it_j].astype(np.int8), "h": h[it_st, it_j].astype(np.float32),
            "v": v[it_st, it_j].astype(np.float32), "d": it_d.astype(np.float32), "S": S, "detour": detour, "n_extra": n_extra - n_far_items,
            "n_far_cells": len(far_ids), "n_far_items": n_far_items, "n_hi": len(hi_st)}


_BG_ROWS = None


def far_from_table(name, cells, mode, pss):
    """(cell row, station, pattern, distance) far items from data_buf/<State>/far.parquet (rows aligned with the web cells): for each far
    station, the pattern of its group (0 bus / tram, 1 rail) with the best default stop score `pss` [S, P], at the road distance (one item per
    station and group keeps the national item count within the GPU)."""
    import pandas as pd
    f = pd.read_parquet(PATHS.FLEX_DATA / name / "far.parquet")
    K = sum(1 for c in f.columns if c.startswith("far_stop_"))
    st = f[[f"far_stop_{k}" for k in range(K)]].to_numpy()[cells]
    dd = f[[f"far_dist_{k}" for k in range(K)]].to_numpy()[cells].astype(np.float32)
    gg = f[[f"far_group_{k}" for k in range(K)]].to_numpy()[cells]
    r, k = np.nonzero(st >= 0)
    s_, d_, g_ = st[r, k], dd[r, k], gg[r, k]
    ok = (mode[s_] >= 0) & np.where(g_[:, None] == 1, mode[s_] == 2, np.isin(mode[s_], (0, 1)))   # (n, P): the group's patterns
    j = np.where(ok, pss[s_], -1.0).argmax(1)
    has = ok.any(1)
    return r[has], s_[has], j[has], d_[has]


def far_cells(name, w, nc, cells, geom, hi_st, ps, lat0, detour):
    """Populated res-10 cells with no stored cell (no stop within walking reach) within FAR_MAX_M of a station of the score bands, from the census
    blocks: (sorted h3 ids, {bg row, population, lon, lat}). Each cell's block group is the one with most of its residents."""
    global _BG_ROWS
    import geopandas as gpd
    import h3
    import pandas as pd
    from h3ronpy.vector import coordinates_to_cells
    if _BG_ROWS is None:
        g = pd.read_parquet(PATHS.DATA / "blockgroups.parquet", columns=["GEOID"]).GEOID
        _BG_ROWS = pd.Series(np.arange(len(g)), index=g)
    web_ids = np.asarray(coordinates_to_cells(geom[:, 1].astype(np.float64), geom[:, 0].astype(np.float64), 10)).astype(np.uint64)
    stored = np.sort(web_ids[cells])
    parts = []
    for f in sorted((STUDY / name / "counties").glob("*/block.geoparquet")):
        b = gpd.read_parquet(f, columns=["GEOID", "population", "geometry"])
        b["population"] = pd.to_numeric(b["population"], errors="coerce").fillna(0)
        b = b[b.population > 0]
        if b.empty:
            continue
        rp = b.geometry.to_crs(4326).representative_point()
        parts.append(pd.DataFrame({"bg": b["GEOID"].astype(str).str[:12].to_numpy(), "pop": b.population.to_numpy(np.float32), "lon": rp.x.to_numpy(), "lat": rp.y.to_numpy()}))
    if not parts:
        return None
    bl = pd.concat(parts, ignore_index=True)
    bl["cell"] = np.asarray(coordinates_to_cells(bl.lat.to_numpy(), bl.lon.to_numpy(), 10)).astype(np.uint64)
    pos = np.minimum(np.searchsorted(stored, bl.cell.to_numpy()), max(len(stored) - 1, 0))
    bl = bl[~((len(stored) > 0) & (stored[pos] == bl.cell.to_numpy()))]           # only blocks without a stored cell
    d, _ = cKDTree(ps[hi_st]).query(xy(bl.lon.to_numpy(), bl.lat.to_numpy(), lat0), k=1, distance_upper_bound=FAR_MAX_M / detour)
    bl = bl[np.isfinite(d)]
    if bl.empty:
        return None
    top = bl.sort_values("pop", ascending=False).drop_duplicates("cell").set_index("cell")["bg"]
    agg = bl.groupby("cell")["pop"].sum().sort_index()
    ids = agg.index.to_numpy(np.uint64)
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in ids])
    bgr = _BG_ROWS.reindex(top.reindex(agg.index).to_numpy()).fillna(-1).to_numpy(np.int64)
    ok = bgr >= 0
    return ids[ok], {"bg": bgr[ok].astype(np.int32), "pop": agg.to_numpy(np.float32)[ok], "lon": ll[ok, 1].astype(np.float32), "lat": ll[ok, 0].astype(np.float32)}


def main():
    names = [rc.STATE_DIRS[a] for a in rc.STATE_DIRS]
    OUT.mkdir(parents=True, exist_ok=True)
    cells, items, coff, soff = [], [], 0, 0
    with ProcessPoolExecutor(6) as ex:
        for r in ex.map(one_state, names):
            if r is None:
                continue
            n = len(r["bg"])
            cells.append((r["bg"], r["pop"], r["lon"], r["lat"], r["jobs"]))
            items.append((r["cell"] + coff, r["st"] + soff, r["mode"], r["h"], r["v"], r["d"], r["hc"]))
            print(f"{r['name']}: cells {n:,} (far cells {r['n_far_cells']:,}) items {len(r['cell']):,} (extra rail/tram {r['n_extra']:,}, far {r['n_far_items']:,} "
                  f"to the nearest of {r['n_hi']:,} stations of score >= {FAR_BANDS[0][0]} per band) detour {r['detour']:.2f}", flush=True)
            coff += n; soff += r["S"]
    cat = lambda L, i: np.concatenate([x[i] for x in L])  # noqa: E731
    np.savez(OUT / "cells.npz", bg=cat(cells, 0), pop=cat(cells, 1), lon=cat(cells, 2), lat=cat(cells, 3), jobs=cat(cells, 4))
    np.savez(OUT / "items.npz", cell=cat(items, 0), st=cat(items, 1), mode=cat(items, 2), h=cat(items, 3), v=cat(items, 4), d=cat(items, 5), hc=cat(items, 6))
    print("total cells", coff, "items", sum(len(x[0]) for x in items), flush=True)


if __name__ == "__main__":
    main()
