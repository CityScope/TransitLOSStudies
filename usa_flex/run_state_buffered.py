"""Flexible-scoring map, stop-buffer method: one run per state, census data only (blocks, block groups, LODES jobs; no WorldPop / GBA).

    python run_state_buffered.py VT NY ...       # or no arguments: every state; --workers N (states in parallel, largest first)

Processing area = the state. Rules (constants below, documented here):
  * Walk zone: a WALK_BUFFER_M buffer around every station (the state's and its neighbours' feeds, as the original study). Only there are roads
    loaded (from the state's Geofabrik extract), kept connected per buffer polygon (`crop_by_aoi_connected`), and only there are the census BLOCKS
    used, on H3 res FINE_RES cells (every res-10 child of the res-COARSE_RES cells the buffer touches, centre inside the state).
  * Everywhere else in the state: census BLOCK GROUPS only, on res-COARSE_RES cells (no stops within walking reach: level of service 0).
  * Each fine cell is joined to its nearest road (straight line from the cell centre to the nearest point of the nearest edge); its walking
    distance to a station = that straight line + the network distance from that point to the station (`core.process_cells` with the cell
    radius set to 0, the straight line added after). Stations beyond MAX_WALK_M in total are dropped from the cell.
  * Blocks reach their res-10 cells by centroid, split by NSI floor area (residential for residents, other for jobs).
  * A res-10 cell with fewer than MIN_DENSITY_KM2 residents + jobs per km2 is dropped (res-8 cells only when empty); its counts go back into the kept cells of the state
    with one factor per group (residents' columns by the population factor, workplace columns by the jobs factor, so the shares inside a cell
    do not change). Logged, WARNING when more than RESCALE_WARN of a group is rescaled.
Outputs: data_buf/<State>/{stop_profiles.parquet, cells.parquet, summary.json}, web_buf/<State>/ (the viewer files, transitlos.flex.export) and
web_buf/index.json. The existing data/ and web/ (and the fit data built from them) are not touched.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "data_buf"
WEB = HERE / "web_buf"
OSM = HERE.parent / "cache" / "osm"

WALK_BUFFER_M = 2000.0    # walk zone around every station (= the walking cap of the fits)
MAX_WALK_M = 2000.0       # longest walk (straight line to the road + network) kept for a cell
FAR_BUFFER_M = 15000.0    # roads loaded once within this distance of every station: the walk search and the drive (far-access) search share them
FAR_K = 15                # far stations kept per cell (2-15 km by road)
FAR_BANDS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 9.0)]   # default stop score of the station's best pattern
FAR_GROUPS = {0: (0, 1), 1: (2,)}                                # 0 = bus / tram, 1 = rail
FINE_RES, COARSE_RES = 10, 8
MIN_DENSITY_KM2 = 10.0    # a res-10 cell needs >= 10 residents + jobs per km2 (~0.15 people); res-8 cells: anyone
RESCALE_WARN = 0.05
SIMPLIFY_M = 10.0
CODE_VERSION = 8   # 8: station speed = straight-line distance 15 min before / after the station (was the whole trip); 7: far access by road 2-15 km (band x bus/tram, rail; 15 per cell), roads loaded once to 15 km; 6: NSI floor-area weights; density rule (10 /km2) on res-10 cells only; 5: inhabited = >= 50 residents + jobs per km2; 4: walk-zone blocks by centroid, split equally (totals kept); 3: every block's res-8 cell exists; 2: NaN block counts -> 0; outside the walk zones the blocks summed onto res-8 cells (jobs kept)
POP_GROUP = ["population", "under18Population", "over65Population", "commutersResidents", "commuteMinutesResidents", "commuteKmResidents"]


def log(abbr, msg, t0=[time.time()]):
    print(f"[{abbr} {time.time() - t0[0]:7.0f}s] {msg}", flush=True)


def _cells_in(geom_4326, res):
    """uint64 H3 cells whose centre is inside a (multi)polygon."""
    import h3ronpy.vector as hv
    import shapely
    if geom_4326 is None or geom_4326.is_empty:
        return np.zeros(0, np.uint64)
    parts = list(geom_4326.geoms) if geom_4326.geom_type == "MultiPolygon" else [geom_4326]
    out = [np.asarray(hv.geometry_to_cells(p, res)).astype(np.uint64) for p in parts]
    return np.unique(np.concatenate(out)) if out else np.zeros(0, np.uint64)


def _centres(cells):
    import h3ronpy.vector as hv
    r = hv.cells_to_coordinates(np.asarray(cells, np.uint64))
    return np.asarray(r.column("lat")), np.asarray(r.column("lng"))


def _far_search(abbr, net, stations, prof, params, cells):
    """Far access by road for every cell: in each default-score band (FAR_BANDS) and group (bus/tram, rail), the stations no nearer station
    of the same band and group dominates (as frequent and as fast), by road distance = straight line from the cell centre to its nearest road
    + network distance from that point. Up to FAR_K per cell: first the nearest of every band and group, then the next ranks by distance.
    Stations within WALK (MAX_WALK_M) are left to the walk search. Returns (station [nc, FAR_K] int32 -1 empty, dist uint16 65535 empty,
    group int8 -1 empty)."""
    import shapely
    from pyproj import Transformer
    from transitlos.flex import core, scoring as sc
    nc = len(cells)
    lat, lon = _centres(cells)
    x, y = Transformer.from_crs(4326, net.crs, always_xy=True).transform(lon, lat)
    pts = shapely.points(x, y)
    ei = shapely.STRtree(net.geom).nearest(pts)
    near = net.geom[ei]
    off = shapely.distance(near, pts).astype(np.float32)
    pos = np.clip(shapely.line_locate_point(near, pts), 0, net.el[ei]).astype(np.float32)
    rec_off = np.arange(nc + 1, dtype=np.int32)
    with np.errstate(invalid="ignore"):
        pss = np.where(prof.mode >= 0, sc.stop_score(np.maximum(prof.mode, 0), np.maximum(prof.h, 0.5), np.maximum(prof.v, 0.1)), 0.0)
    best = pss.max(1)
    cand = []   # (station, dist, group, rank) per cell, from every band x group search
    cfg = core.Config(max_walk_m=FAR_BUFFER_M, cell_radius_m=0.0, K=FAR_K, Kc=24, Kcand=96)
    for lo, hi in FAR_BANDS:
        for g, modes in FAR_GROUPS.items():
            ingrp = np.isin(prof.mode, modes)
            sub = np.flatnonzero((best >= lo) & (best < hi) & ingrp.any(1))
            if not len(sub):
                continue
            sp = core.Profiles(np.where(ingrp[sub], prof.mode[sub], -1), prof.h[sub], prof.v[sub])
            sn, so, ss, _ = net.snap(stations.lat.to_numpy()[sub], stations.lon.to_numpy()[sub])
            lab = core.multilabel_search(net.indptr, net.adj, net.w, sn, so, ss, sp, cfg)[:4]
            cs, cd, _, _ = core.process_cells(rec_off, net.eu[ei], net.ev[ei], pos, pos, net.el[ei], lab, sp, cfg, params)
            tot = np.where(cd == 65535, np.inf, cd.astype(np.float32) + off[:, None])
            ok = (cs >= 0) & (tot > MAX_WALK_M) & (tot <= FAR_BUFFER_M)
            r, k = np.nonzero(ok)
            cand.append(pd.DataFrame({"c": r, "st": sub[cs[r, k]], "d": tot[r, k], "g": g, "rank": k}))
    st_out = np.full((nc, FAR_K), -1, np.int32); d_out = np.full((nc, FAR_K), 65535, np.uint16); g_out = np.full((nc, FAR_K), -1, np.int8)
    if cand:
        c = pd.concat(cand, ignore_index=True).sort_values(["c", "st", "d"]).drop_duplicates(["c", "st", "g"])
        c = c.sort_values(["c", "rank", "d"])
        c["slot"] = c.groupby("c").cumcount()
        c = c[c["slot"] < FAR_K]
        st_out[c.c.to_numpy(), c.slot.to_numpy()] = c.st.to_numpy()
        d_out[c.c.to_numpy(), c.slot.to_numpy()] = np.minimum(np.round(c.d.to_numpy()), 65534)
        g_out[c.c.to_numpy(), c.slot.to_numpy()] = c.g.to_numpy()
    log(abbr, f"far access: {int((st_out[:, 0] >= 0).sum()):,} of {nc:,} cells reach a station 2-15 km by road; median road offset {np.median(off):.0f} m")
    return st_out, d_out, g_out


NSI_OK = [True]


def _nsi_floor_area(abbr, cells):
    """Residential and non-residential floor area (m2, USACE NSI structures by their point) per cell of `cells` (sorted res-10 ids);
    (None, None) when NSI is unavailable (the split then falls back to equal shares)."""
    try:
        import us
        import h3ronpy.vector as hv
        from pycensus.cadaster.usa.nsi.loader import state as nsi_state
        g = nsi_state(us.states.lookup(abbr).fips, HERE.parent / "cache")
    except Exception as exc:  # noqa: BLE001
        log(abbr, f"NSI floor areas not available ({type(exc).__name__}: {str(exc)[:120]}): equal split")
        NSI_OK[0] = False
        return None, None
    c = np.asarray(hv.coordinates_to_cells(g.geometry.y.to_numpy(), g.geometry.x.to_numpy(), FINE_RES)).astype(np.uint64)
    pos = np.searchsorted(cells, c); hit = (pos < len(cells)) & (cells[np.minimum(pos, len(cells) - 1)] == c)
    ba = np.nan_to_num(pd.to_numeric(g["built_area_m2"], errors="coerce").to_numpy(dtype=float))
    res = g["use_class"].isin(["residential", "mixed"]).to_numpy()
    w_res = np.zeros(len(cells)); w_non = np.zeros(len(cells))
    np.add.at(w_res, pos[hit & res], ba[hit & res]); np.add.at(w_non, pos[hit & ~res], ba[hit & ~res])
    log(abbr, f"NSI: {int(hit.sum()):,} structures in the walk zone, {w_res.sum() / 1e6:,.1f} km2 residential and {w_non.sum() / 1e6:,.1f} km2 other floor area")
    return w_res, w_non


def _blocks_by_centroid(blocks, cells, cols, w_res=None, w_non=None) -> pd.DataFrame:
    """Counts per cell (cells sorted): each cell takes data from the one block holding its centre; a block's counts are split over those
    cells by built floor area (`w_res` for the residents' columns, `w_non` -- non-residential floor area -- for the workplace columns; each
    falling back to the total floor area, then to an equal split, when the block's cells have none). The cells' floor areas are rescaled
    so they sum to the block's (the weights of a block sum to 1): every block's total is kept exactly. A block holding no cell centre goes
    whole to the cell of its representative point."""
    import geopandas as gpd
    import h3ronpy.vector as hv
    out = np.zeros((len(cells), len(cols)))
    if not len(cells) or blocks is None or not len(blocks):
        return pd.DataFrame(out, columns=cols)
    lat, lon = _centres(cells)
    bb = blocks[cols + ["geometry"]].reset_index(drop=True)
    j = gpd.sjoin(gpd.GeoDataFrame({"ci": np.arange(len(cells))}, geometry=gpd.points_from_xy(lon, lat), crs=4326), bb, how="inner", predicate="within")
    j = j[~j["ci"].duplicated()]
    bi, ci = j["index_right"].to_numpy(), j["ci"].to_numpy()
    n = np.bincount(bi, minlength=len(bb))
    vals = bb[cols].to_numpy(dtype=float)
    eq = 1.0 / n[bi]

    def share(w):
        if w is None:
            return eq
        wc = w[ci]
        tot = np.bincount(bi, weights=wc, minlength=len(bb))[bi]
        return np.where(tot > 0, wc / np.where(tot > 0, tot, 1.0), np.nan)

    w_all = None if w_res is None else w_res + w_non
    s_all = share(w_all)
    s_all = np.where(np.isnan(s_all), eq, s_all)
    s_res, s_non = share(w_res), share(w_non)
    s_res = np.where(np.isnan(s_res), s_all, s_res); s_non = np.where(np.isnan(s_non), s_all, s_non)
    for k, c in enumerate(cols):
        np.add.at(out[:, k], ci, vals[bi, k] * (s_res if c in POP_GROUP else s_non))
    lone = np.flatnonzero(n == 0)
    if len(lone):
        rp = bb.geometry.iloc[lone].representative_point()
        lc = np.asarray(hv.coordinates_to_cells(rp.y.to_numpy(), rp.x.to_numpy(), FINE_RES)).astype(np.uint64)
        pos = np.searchsorted(cells, lc); hit = (pos < len(cells)) & (cells[np.minimum(pos, len(cells) - 1)] == lc)
        for k in range(len(cols)):
            np.add.at(out[:, k], pos[hit], vals[lone[hit], k])
    return pd.DataFrame(out, columns=cols)


def _rescale(counts: pd.DataFrame, keep: np.ndarray, abbr: str) -> pd.DataFrame:
    """Drop the uninhabited rows and put their counts back with one factor per group (see the module docstring)."""
    jobs_cols = [c for c in counts.columns if c not in POP_GROUP]
    out = counts[keep].copy()
    for grp, key in ((POP_GROUP, "population"), (jobs_cols, "lodes_wac_jobsTotal")):
        cols = [c for c in grp if c in out.columns]
        if key not in counts.columns or not cols:
            continue
        total, kept = float(counts[key].sum()), float(out[key].sum())
        if total <= 0 or kept <= 0:
            continue
        f = total / kept
        share = (total - kept) / total
        msg = f"[rescale] {key}: {total - kept:,.0f} of {total:,.0f} ({share:.1%}) in cells below {MIN_DENSITY_KM2:g} residents + jobs per km2 -- factor {f:.4f} on {len(cols)} columns"
        log(abbr, ("WARNING " if share > RESCALE_WARN else "") + msg)
        out[cols] = out[cols] * f
    return out


def run_state(abbr: str) -> dict:
    import geopandas as gpd
    import shapely
    import run_county as rc
    import build_web as bw
    from transitlos.flex import core, catchment as ct, scoring as sc
    from transitlos.flex.cells import CellStops
    from transitlos.flex.export import export_for_browser
    from transitlos.network import prepare_street_network
    from UrbanAccessAnalyzer.api import AreaOfInterest
    from UrbanAccessAnalyzer.csr_network import CsrNetwork
    from UrbanAccessAnalyzer.osm_io import download_geofabrik

    t0 = time.time()
    name = rc.STATE_DIRS[abbr]
    sdir = rc.STUDY / "data" / name
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = core.Config(max_walk_m=MAX_WALK_M, cell_radius_m=0.0)
    params = sc.DEFAULT

    # -- stations (candidates cached by the existing flex run are reused)
    old = HERE / "data" / name
    from transitlos.scoring import direction_subset as ds
    tag = f"{ds.SPEED_METHOD}{ds.SPEED_WINDOW_MIN:g}{ds.SPEED_WINDOW_DIRECTION}"
    for f in (f"candidates_{tag}.parquet", f"stops_raw_{tag}.parquet"):
        if (old / f).is_file() and not (out / f).is_file():
            shutil.copy2(old / f, out / f)
    rc.QUIET[0] = True
    stations, prof, cut = rc.state_profiles(sdir, out / "x", cfg, params)
    rc.write_stations(out, stations, prof, cut)
    log(abbr, f"{len(stations):,} stations")

    # -- zones
    state = gpd.read_parquet(sdir / "state.geoparquet")[["geometry"]].to_crs(4326)
    state_geom = state.union_all()
    st_pts = gpd.GeoSeries(gpd.points_from_xy(stations.lon, stations.lat), crs=4326).to_crs(5070)
    buf = gpd.GeoSeries([st_pts.buffer(WALK_BUFFER_M, resolution=8).union_all()], crs=5070).to_crs(4326).iloc[0]
    import h3ronpy as h3r
    coarse_all = _cells_in(state_geom, COARSE_RES)
    fine_parents = np.unique(np.asarray(h3r.change_resolution(_cells_in(buf.intersection(state_geom), FINE_RES), COARSE_RES)).astype(np.uint64))
    fine = np.asarray(h3r.change_resolution(fine_parents, FINE_RES)).astype(np.uint64)
    flat, flon = _centres(fine)
    fine = np.unique(fine[shapely.contains_xy(state_geom, flon, flat)])
    coarse = np.setdiff1d(coarse_all, fine_parents)
    log(abbr, f"walk zone: {len(fine):,} res-{FINE_RES} cells ({len(fine_parents):,} res-{COARSE_RES} areas); elsewhere {len(coarse):,} res-{COARSE_RES} cells")

    # -- census: blocks in the walk zone, block groups elsewhere (same count columns and commute helpers as build_web)
    bgc = bw._bg_commute()
    blocks, bgs = [], []
    for cd in sorted(p for p in (sdir / "counties").iterdir() if p.is_dir()):
        if (cd / "block.geoparquet").is_file():
            blocks.append(gpd.read_parquet(cd / "block.geoparquet"))
        if (cd / "blockgroup.geoparquet").is_file():
            bgs.append(gpd.read_parquet(cd / "blockgroup.geoparquet"))
    cols = list(bw.COUNT_COLUMNS)
    b = bw.add_commute_counts(pd.concat(blocks, ignore_index=True), bgc).to_crs(4326) if blocks else None
    g = bw.add_commute_counts(pd.concat(bgs, ignore_index=True), bgc).to_crs(4326) if bgs else None
    if b is not None:   # a block without a LODES / commute value has 0 of it (NaN would poison every sum it enters)
        for c in [c for c in cols if c in b.columns]:
            b[c] = pd.to_numeric(b[c], errors="coerce").fillna(0.0)
        import h3ronpy.vector as h3v
        rp = b.geometry.representative_point()
        bcell8 = np.asarray(h3v.coordinates_to_cells(rp.y.to_numpy(), rp.x.to_numpy(), COARSE_RES)).astype(np.uint64)
    if b is not None and len(fine):
        b_in = b[np.isin(bcell8, fine_parents)]          # blocks of the walk zone (by representative point)
        w_res, w_non = _nsi_floor_area(abbr, fine)
        cnt_f = _blocks_by_centroid(b_in, fine, [c for c in cols if c in b_in.columns], w_res, w_non).reindex(columns=cols, fill_value=0.0)
    else:
        cnt_f = pd.DataFrame(0.0, index=range(len(fine)), columns=cols)
    # outside the walk zones: the blocks summed onto their res-COARSE_RES cell (block groups carry no LODES jobs; blocks place people better).
    # Every block's cell exists, also where the cell centre is just across the state line (else those people would be lost)
    if b is not None:
        coarse = np.union1d(coarse, np.setdiff1d(np.unique(bcell8), fine_parents))
    coarse = np.sort(coarse)
    cnt_c = pd.DataFrame(0.0, index=range(len(coarse)), columns=cols)
    if b is not None and len(coarse):
        b_out = b[~np.isin(bcell8, fine_parents)]
        c8 = bcell8[~np.isin(bcell8, fine_parents)]
        pos = np.searchsorted(coarse, c8); hit = (pos < len(coarse)) & (coarse[np.minimum(pos, len(coarse) - 1)] == c8)
        for c in [c for c in cols if c in b_out.columns]:
            v = np.zeros(len(coarse)); np.add.at(v, pos[hit], b_out[c].to_numpy(dtype=float)[hit]); cnt_c[c] = v

    # -- roads of the walk zone (connected per buffer polygon), nearest road of every fine cell, network search
    pbf = download_geofabrik(state, output_folder=str(OSM))
    buf_far = gpd.GeoSeries([st_pts.buffer(FAR_BUFFER_M, resolution=8).union_all()], crs=5070).to_crs(4326).iloc[0]
    net_sn = prepare_street_network(AreaOfInterest(gpd.GeoDataFrame(geometry=[buf_far], crs=4326)), cache_dir=str(out / "_network15k"), cluster_distance=SIMPLIFY_M,
                                    pbf_path=pbf, network_type="all", ignore_oneway=True)
    net = CsrNetwork(net_sn.nodes, net_sn.edges, net_sn.crs)
    del net_sn
    log(abbr, f"network: {net.n:,} nodes, {len(net.eu):,} edges")
    from pyproj import Transformer
    flat, flon = _centres(fine)
    x, y = Transformer.from_crs(4326, net.crs, always_xy=True).transform(flon, flat)
    pts = shapely.points(x, y)
    ei = shapely.STRtree(net.geom).nearest(pts)
    near = net.geom[ei]
    off = shapely.distance(near, pts).astype(np.float32)
    pos = np.clip(shapely.line_locate_point(near, pts), 0, net.el[ei]).astype(np.float32)
    sn, so, ss, ok = net.snap(stations.lat.to_numpy(), stations.lon.to_numpy())
    ls, ld, ln, lc, stats = core.multilabel_search(net.indptr, net.adj, net.w, sn, so, ss, prof, cfg)
    rec_off = np.arange(len(fine) + 1, dtype=np.int32)
    cs, cdist, cf, cr = core.process_cells(rec_off, net.eu[ei], net.ev[ei], pos, pos, net.el[ei], (ls, ld, ln, lc), prof, cfg, params)
    tot = cdist.astype(np.float32) + off[:, None]
    bad = (cdist == 65535) | (tot > MAX_WALK_M)
    cs = np.where(bad, -1, cs).astype(np.int32)
    cdist = np.where(bad, 65535, np.minimum(np.round(tot), 65534)).astype(np.uint16)
    log(abbr, f"search: {int(stats[0]):,} labels; {int((cs[:, 0] >= 0).sum()):,} of {len(fine):,} cells reach a station; median road offset {np.median(off):.0f} m")

    # -- far access (drive, 2-15 km) for every cell, on the same network
    far_st, far_d, far_g = _far_search(abbr, net, stations, prof, params, np.r_[fine, coarse])

    # -- all cells, the >= MIN_DENSITY_KM2 rule with rescaling
    K = cs.shape[1]
    cells = np.r_[fine, coarse]
    station = np.vstack([cs, np.full((len(coarse), K), -1, np.int32)])
    dist = np.vstack([cdist, np.full((len(coarse), K), 65535, np.uint16)])
    counts = pd.concat([cnt_f, cnt_c], ignore_index=True)
    import h3ronpy
    area_km2 = np.asarray(h3ronpy.cells_area_km2(cells)).astype(float)
    act = counts["population"].to_numpy() + counts.get("lodes_wac_jobsTotal", pd.Series(0.0, index=counts.index)).to_numpy()
    finest = np.r_[np.ones(len(fine), bool), np.zeros(len(coarse), bool)]
    keep = np.where(finest, act / area_km2 >= MIN_DENSITY_KM2, act > 0)   # the density rule on the finest (res-10) cells only
    counts = _rescale(counts, keep, abbr).reset_index(drop=True)
    cells, station, dist = cells[keep], station[keep], dist[keep]
    far_st, far_d, far_g = far_st[keep], far_d[keep], far_g[keep]
    order = np.argsort(cells)
    cells, station, dist, counts = cells[order], station[order], dist[order], counts.iloc[order].reset_index(drop=True)
    far_st, far_d, far_g = far_st[order], far_d[order], far_g[order]
    pd.DataFrame({"h3": [f"{int(c):x}" for c in cells], **{f"far_stop_{k}": far_st[:, k] for k in range(FAR_K)},
                  **{f"far_dist_{k}": far_d[:, k] for k in range(FAR_K)}, **{f"far_group_{k}": far_g[:, k] for k in range(FAR_K)}}).to_parquet(out / "far.parquet")
    res_flag = np.where(np.isin(cells, fine), FINE_RES, COARSE_RES).astype(np.uint8)

    # -- block group under every cell centre (block-group / tract views), census population of the block groups
    clat, clon = _centres(cells)
    bg_pop, gid = {}, np.full(len(cells), "", dtype=object)
    if g is not None:
        gg = g[["GEOID", "population", "geometry"]].reset_index(drop=True)
        j = gpd.sjoin(gpd.GeoDataFrame({"_i": np.arange(len(cells))}, geometry=gpd.points_from_xy(clon, clat), crs=4326), gg, how="left", predicate="within")
        j = j.drop_duplicates("_i").sort_values("_i")
        gid = j["GEOID"].fillna("").astype(str).to_numpy()
        bg_pop = dict(zip(gg["GEOID"].astype(str), pd.to_numeric(gg["population"], errors="coerce").fillna(0).astype(float)))

    # -- outputs
    pd.DataFrame({"h3": [f"{int(c):x}" for c in cells], "res": res_flag, "bg": gid, **{c: counts[c].to_numpy() for c in counts.columns},
                  **{f"stop_{k}": station[:, k] for k in range(K)}, **{f"dist_{k}": dist[:, k] for k in range(K)}}).to_parquet(out / "cells.parquet")
    st = pd.read_parquet(out / "stop_profiles.parquet")
    resobj = CellStops(cells, station, dist, np.zeros(len(cells), np.uint8), np.zeros(len(cells), np.uint8), {})
    web = WEB / name
    meta = export_for_browser(web, st.lat.to_numpy(), st.lon.to_numpy(), prof, resobj, params, counts=counts, count_labels=bw.COUNT_COLUMNS,
                              extra_meta={"name": name, "method": "stop buffers", "walk_buffer_m": WALK_BUFFER_M, "min_density_km2": MIN_DENSITY_KM2,
                                          "hidden_columns": [cols.index(c) for c in bw.HIDDEN_COLUMNS],
                                          "mean_columns": [{"label": l, "num": cols.index(n), "den": cols.index(d), "scale": k} for l, n, d, k in bw.MEAN_COLUMNS]},
                              points=pd.read_parquet(out / "stops_by_id.parquet") if (out / "stops_by_id.parquet").is_file() else None)
    bw.write_census_units(web, gid, bg_pop)
    los, slot, _ = core.recompute_cells_fast(station, dist, prof, params)
    pop = counts["population"].to_numpy()
    summ = {"name": name, "path": f"{name}/", "bbox": meta.get("bbox"), "stations": int(prof.S), "cells": int(len(cells)),
            "cells_fine": int((res_flag == FINE_RES).sum()), "cells_coarse": int((res_flag == COARSE_RES).sum()),
            "population": float(pop.sum()), "population_in_reached_cells": float(pop[slot >= 0].sum()),
            "network_edges": int(len(net.eu)), "seconds": round(time.time() - t0, 1), "code_version": CODE_VERSION, "nsi_ok": NSI_OK[0]}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    log(abbr, f"done: {summ['cells']:,} cells, population {summ['population']:,.0f} ({summ['population_in_reached_cells'] / max(summ['population'], 1):.1%} reach a station)")
    return summ


def _one(abbr):
    try:
        return abbr, run_state(abbr), None
    except Exception:  # noqa: BLE001
        return abbr, None, traceback.format_exc()[-1500:]


def main():
    import run_county as rc
    from transitlos.flex.export import publish_frontend
    import build_web as bw

    if "--index-only" in sys.argv:   # web_buf/index.json from every finished state (after run_us_buffered.py)
        idx = {"states": sorted((json.loads(f.read_text()) for f in OUT.glob("*/summary.json")), key=lambda s: s["name"])}
        WEB.mkdir(parents=True, exist_ok=True); (WEB / "index.json").write_text(json.dumps(idx))
        print("states", len(idx["states"]), "data version", bw.write_index_version(WEB), flush=True)
        publish_frontend(WEB)
        return
    if "--one" in sys.argv:   # one state, no index (run by run_us_buffered.py under the memory scheduler)
        a = [x for x in sys.argv[1:] if not x.startswith("--")][0]
        r = run_state(a)
        print("ok", r["name"], flush=True)
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    abbrs = [a for a in args if a.isalpha()] or [a for a, n in rc.STATE_DIRS.items() if (rc.STUDY / "data" / n / "counties").is_dir()]
    size = {a: sum(f.stat().st_size for f in (rc.STUDY / "data" / rc.STATE_DIRS[a] / "counties").glob("*/block.geoparquet")) for a in abbrs}
    abbrs.sort(key=lambda a: -size[a])   # largest first: the long ones start early
    WEB.mkdir(parents=True, exist_ok=True)
    idx_path = WEB / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.is_file() else {"states": []}
    failed = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, a): a for a in abbrs}
        for f in as_completed(futs):
            a, r, err = f.result()
            if err:
                failed.append(a); print(f"{a} FAILED\n{err}", flush=True); continue
            idx["states"] = sorted([s for s in idx["states"] if s["name"] != r["name"]] + [r], key=lambda s: s["name"])
            idx_path.write_text(json.dumps(idx))
    print("data version", bw.write_index_version(WEB), "failed:", failed, flush=True)
    publish_frontend(WEB)


if __name__ == "__main__":
    main()
