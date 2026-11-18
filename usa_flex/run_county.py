"""One US county end to end for the flexible scoring (`transitlos.flex`): station profiles -> street-network search -> cell/stations table -> default scores
-> files (+ validation against the original study's h3_res10).

    python run_county.py VT 50007            # Chittenden County, Vermont

Reads (never writes) the original study: `usa_study/data/<State>/{gtfs, state.geoparquet, counties/<GEOID>/_network_cache_b1500, counties/<GEOID>/h3_res10.geoparquet}`.
Writes only under `usa_flex/data/<State>/<GEOID>/`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / "usa_study"
sys.path.insert(0, str(HERE.parent / "tools"))   # the shared study code (studytools)
from transitlos.flex import core, profiles as pr, scoring as sc   # noqa: E402
from transitlos.flex.cells import compute_cell_stops              # noqa: E402
from UrbanAccessAnalyzer.csr_network import CsrNetwork as Net     # noqa: E402

import us as _us
STATE_DIRS = {s.abbr: s.name.replace(' ', '') for s in _us.states.STATES_AND_TERRITORIES}   # same folder naming as usa_study (e.g. NewYork)


QUIET = [False]


def log(msg, t0=[time.time()]):
    if not QUIET[0]:
        print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def state_profiles(state_dir: Path, out_dir: Path, cfg: core.Config, params=sc.DEFAULT, county_dir: Path | None = None, buffer_m: float = 1500.0):
    """Station profiles (cached next to the results: candidates are the slow part). With `county_dir` only the feeds' stops within that county plus
    `buffer_m` are scored (large states, e.g. New York: the whole state's 85 feeds are not needed to test one county)."""
    import geopandas as gpd
    # the speed settings are part of the cache name (2026-11-17: speed = straight-line distance 15 min before / after the station), so a
    # cache built with another speed method is never reused
    from transitlos.scoring import direction_subset as ds
    tag = f"{ds.SPEED_METHOD}{ds.SPEED_WINDOW_MIN:g}{ds.SPEED_WINDOW_DIRECTION}"
    cache = out_dir.parent / (f"candidates_{tag}.parquet" if county_dir is None else f"candidates_{county_dir.name}_{tag}.parquet")
    if cache.is_file():
        cand = pd.read_parquet(cache)
        log(f"candidates: {len(cand)} rows (cache)")
    else:
        dirs = sorted(p for p in (state_dir / "gtfs").iterdir() if p.is_dir() and not p.name.startswith("_"))
        from studytools.us_feeds import neighbour_feed_dirs

        dirs += [d for d in neighbour_feed_dirs(state_dir) if d not in dirs]   # service across the state line (the study's rule)
        if county_dir is None:
            aoi = gpd.read_parquet(state_dir / "state.geoparquet")[["geometry"]].to_crs(4326)
        else:
            c = gpd.read_parquet(county_dir / "county.geoparquet")[["geometry"]].to_crs(4326)
            aoi = c.to_crs(c.estimate_utm_crs()).buffer(buffer_m).to_crs(4326).to_frame("geometry").set_geometry("geometry")
        sink: list = []
        cand = pr.state_candidates(dirs, aoi, log=log, stops_sink=sink)
        cand.to_parquet(cache)
        if sink:   # all real stops (own location, name, service weight) + the station each belongs to: the map shows these, not one point per station
            pd.concat(sink, ignore_index=True).drop_duplicates(["stop_id", "station_lat", "station_lon"]).to_parquet(cache.with_name(cache.name.replace("candidates", "stops_raw")))
        log(f"candidates: {len(cand)} rows from {len(dirs)} feeds")
    res = pr.select(cand, cfg, params)
    raw = cache.with_name(cache.name.replace("candidates", "stops_raw"))
    if raw.is_file():
        stations = res[0]
        key = {(a, b): i for i, (a, b) in enumerate(zip(stations.lat.round(6), stations.lon.round(6)))}
        r = pd.read_parquet(raw)
        r["station"] = [key.get((a, b), -1) for a, b in zip(r.station_lat.round(6), r.station_lon.round(6))]
        r = r[r.station >= 0].drop(columns=["station_lat", "station_lon"]).reset_index(drop=True)
        r["stop_name"] = r["stop_name"].astype(str)
        r.to_parquet(cache.with_name(cache.name.replace("candidates", "stops_by_id")))
    return res


def run(state: str, geoid: str, cfg: core.Config = core.Config(), params=sc.DEFAULT, out_root: Path | None = None, validate: bool = True, county_feeds_only: bool = False, shared=None, quiet: bool = False) -> dict:
    state_dir = STUDY / "data" / STATE_DIRS[state]
    county_dir = state_dir / "counties" / geoid
    out_dir = (out_root or HERE / "data") / STATE_DIRS[state] / geoid
    out_dir.mkdir(parents=True, exist_ok=True)
    QUIET[0] = quiet
    res: dict = {"state": state, "county": geoid, "config": cfg.__dict__}

    # `shared` = (stations, prof, cut) already built for the whole state (run_us_state.py): the county table then refers to the state's station index
    stations, prof, cut = shared if shared is not None else state_profiles(state_dir, out_dir, cfg, params, county_dir if county_feeds_only else None)
    log(f"stations kept {len(stations)} (profiles per station {stations.n_profiles.value_counts().sort_index().to_dict()})")
    res["stations"] = int(len(stations))

    t = time.time()
    net = Net.load(county_dir / "_network_cache_b1500")
    res["t_load_network_s"] = round(time.time() - t, 2)
    log(f"network: {net.n} nodes, {len(net.eu)} edges")

    t = time.time()
    sn, so, ss, ok = net.snap(stations.lat.to_numpy(), stations.lon.to_numpy())
    res["t_snap_s"] = round(time.time() - t, 2)
    log(f"snapped {int(ok.sum())}/{len(stations)} stations")

    t = time.time()
    ls, ld, ln, lc, stats = core.multilabel_search(net.indptr, net.adj, net.w, sn, so, ss, prof, cfg)
    res["t_search_s"] = round(time.time() - t, 2)
    res["search_stats"] = {"labels": int(stats[0]), "dropped_by_dominance": int(stats[1]), "node_cap_hits": int(stats[2])}
    log(f"search: {stats[0]} labels, {stats[1]} dominated, {stats[2]} cap hits ({res['t_search_s']}s)")

    t = time.time()
    reached = ln > 0
    cells, rec_off, ru, rv, ra, rb, rl = net.edge_cell_records(10, edge_mask=reached[net.eu] | reached[net.ev])
    res["t_edge_cells_s"] = round(time.time() - t, 2)
    log(f"cells touched by streets: {len(cells)} ({res['t_edge_cells_s']}s)")

    t = time.time()
    cs, cd, cf, cr = core.process_cells(rec_off, ru, rv, ra, rb, rl, (ls, ld, ln, lc), prof, cfg, params)
    res["t_cells_s"] = round(time.time() - t, 2)
    served = (cs >= 0).any(1)
    res["cells"] = int(len(cells)); res["cells_with_stops"] = int(served.sum())
    res["cells_capped"] = int((cf & 1).sum()); res["cells_candidates_cut"] = int((cf & 2).sum()); res["cells_search_cap"] = int((cf & 4).sum())
    res["cells_with_cropped"] = int((cr > 0).sum()); res["stations_cropped_total"] = int(cr.sum()); res["max_cropped_in_a_cell"] = int(cr.max()) if len(cr) else 0
    res["stations_with_cropped_profiles"] = int((stations.dropped > 0).sum()); res["profiles_cropped_total"] = int(stations.dropped.sum())
    log(f"cells with stops {served.sum()}, capped {res['cells_capped']} ({res['t_cells_s']}s)")

    t = time.time()
    los, slot, pidx = core.recompute_cells(cs, cd, prof, params)
    res["t_recompute_s"] = round(time.time() - t, 4)
    log(f"recompute {len(cells)} cells: {res['t_recompute_s']}s")

    # ---- files
    P, K = cfg.P, cfg.K
    if shared is None:
        write_stations(out_dir, stations, prof, cut)
    import h3
    cell_ids = np.array([f"{int(c):x}" for c in cells])
    cdf = pd.DataFrame({"h3": cell_ids})
    for k in range(K):
        cdf[f"stop_{k}"] = cs[:, k]; cdf[f"dist_{k}"] = cd[:, k]
    cdf["flag"] = cf; cdf["cropped"] = cr
    cdf.to_parquet(out_dir / "cell_stops.parquet")
    (out_dir / "params.json").write_text(json.dumps({"score": json.loads(params.to_json()), "config": cfg.__dict__}, indent=1))
    sc_df = pd.DataFrame({"h3": cell_ids, "los": los, "slot": slot, "profile": pidx})
    sc_df.to_parquet(out_dir / "scores_default.parquet")

    if validate:
        res["validation"] = validate_against_original(sc_df, county_dir)
        log("validation: " + json.dumps(res["validation"]))
    (out_dir / "run.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def write_stations(folder: Path, stations, prof, cut) -> None:
    """stop_profiles.parquet: one row per station (lat, lon, profiles cropped by the cap, then mode_j / h_j / v_j / cut_j for the P profile slots)."""
    P = prof.P
    st = pd.DataFrame({"lat": stations.lat, "lon": stations.lon, "cropped": stations.dropped.to_numpy().astype(np.uint8)})
    for j in range(P):
        st[f"mode_{j}"] = prof.mode[:, j]; st[f"h_{j}"] = prof.h[:, j]; st[f"v_{j}"] = prof.v[:, j]; st[f"cut_{j}"] = cut[:, j]
    st.to_parquet(Path(folder) / "stop_profiles.parquet")


def ceil10(x):
    return np.ceil(np.asarray(x) / 10.0 - 1e-9) * 10.0


def validate_against_original(sc_df: pd.DataFrame, county_dir: Path) -> dict:
    orig = pd.read_parquet(county_dir / "h3_res10.geoparquet", columns=["h3", "population", "level_of_service"])
    m = orig.merge(sc_df, on="h3", how="left")
    m["los"] = m["los"].fillna(0.0)
    mine, theirs = ceil10(m.los.to_numpy()), m.level_of_service.fillna(0).to_numpy()
    w = m.population.fillna(0).to_numpy()
    out = {"cells_compared": int(len(m)),
           "equal_after_rounding": round(float((mine == theirs).mean()), 4),
           "within_10_points": round(float((np.abs(mine - theirs) <= 10).mean()), 4),
           "mean_abs_diff": round(float(np.abs(mine - theirs).mean()), 3),
           "pop_weighted_mean_ours": round(float(np.average(m.los, weights=w)) if w.sum() else 0, 3),
           "pop_weighted_mean_original": round(float(np.average(theirs, weights=w)) if w.sum() else 0, 3),
           "corr": round(float(np.corrcoef(m.los, theirs)[0, 1]), 4) if m.los.std() > 0 and theirs.std() > 0 else None,
           "original_zero_ours_positive": round(float(((theirs == 0) & (m.los > 0)).mean()), 4),
           "ours_zero_original_positive": round(float(((m.los == 0) & (theirs > 0)).mean()), 4)}
    return out


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])


def load_run(out_dir: Path):
    """Read a saved run back: (cell ids, station[nc,K], dist[nc,K], core.Profiles, params, config dict)."""
    out_dir = Path(out_dir)
    st = pd.read_parquet(out_dir / "stop_profiles.parquet" if (out_dir / "stop_profiles.parquet").is_file() else out_dir.parent / "stop_profiles.parquet")
    cdf = pd.read_parquet(out_dir / "cell_stops.parquet")
    meta = json.loads((out_dir / "params.json").read_text())
    P = sum(1 for c in st.columns if c.startswith("mode_")); K = sum(1 for c in cdf.columns if c.startswith("stop_"))
    prof = core.Profiles(np.column_stack([st[f"mode_{j}"] for j in range(P)]), np.column_stack([st[f"h_{j}"] for j in range(P)]), np.column_stack([st[f"v_{j}"] for j in range(P)]))
    cs = np.column_stack([cdf[f"stop_{k}"] for k in range(K)]).astype(np.int32); cd = np.column_stack([cdf[f"dist_{k}"] for k in range(K)]).astype(np.uint16)
    return cdf["h3"].to_numpy(), cs, cd, prof, sc.ScoreParams.from_json(json.dumps(meta["score"])), meta["config"]
