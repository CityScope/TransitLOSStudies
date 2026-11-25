"""Flexible scoring (`transitlos.flex`) for one city science map: station profiles -> street-network search -> cell / station table -> default
scores, on the city's own feeds and streets.

    python flex/run_city_flex.py <city> [<city> ...]      # e.g. andorra boston

Settings (user, 2026-11-21): H3 resolution 11 cells, up to 20 service options (profiles) per stop, up to 15 stops per cell; the rest as the US
flex map (`usa_flex/run_county.py`). Reads (never writes) the city pipeline's `<city>/gtfs/` and `<city>/streets/`; writes `<city>/flex/`:
    stop_profiles.parquet   one row per station: lat, lon, profiles cropped, mode_j / h_j / v_j / cut_j for the P slots
    route_departures_*.parquet, routes_*.parquet, route_shapes_*.parquet   departures per station and route, the routes and their shapes
                            (for the map editor: build_city_flex_web.py exports them with transitlos.flex.export.export_routes)
    stops_by_id.parquet     the real stops of every station (the map draws these)
    cell_stops.parquet      h3 (res 11), stop_k / dist_k (m) for the K slots, flags
    scores_default.parquet  h3, default score, winning slot and profile
    params.json, run.json   scoring, configuration, timings and counts
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = 11
P_PROFILES, K_STOPS = 20, 15


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def config():
    """The flex configuration of the city maps: res-11 cells, 20 profiles per station, 15 stations per cell."""
    import h3

    from transitlos.flex import core

    return core.Config(P=P_PROFILES, K=K_STOPS, cell_radius_m=float(h3.average_hexagon_edge_length(RES, unit="m")))


def profiles(city: str, out: Path, cfg, params):
    """Stations and their profiles from the city's feeds inside its AOI (candidates cached: they are the slow part)."""
    import geopandas as gpd

    from transitlos.flex import profiles as pr
    from transitlos.scoring import direction_subset as ds

    tag = f"{ds.SPEED_METHOD}{ds.SPEED_WINDOW_MIN:g}{ds.SPEED_WINDOW_DIRECTION}"
    cache = out / f"candidates_{tag}.parquet"
    raw = out / f"stops_raw_{tag}.parquet"
    rfiles = [out / f"route_departures_{tag}.parquet", out / f"routes_{tag}.parquet", out / f"route_shapes_{tag}.parquet"]
    dirs = sorted(p for p in (HERE / city / "gtfs").iterdir() if p.is_dir() and not p.name.startswith("_"))
    # unzipped files keep the dates of the archive: the folders' own dates (set when a feed is unpacked) count too
    newest = max([d.stat().st_mtime for d in dirs] + [f.stat().st_mtime for d in dirs for f in d.iterdir() if f.is_file()], default=0.0)
    fresh = cache.is_file() and cache.stat().st_mtime >= newest and json.loads((out / "run.json").read_text()).get("feeds") == [d.name for d in dirs] \
        if (out / "run.json").is_file() else False
    fresh = fresh and all(p.is_file() and p.stat().st_mtime >= newest for p in rfiles)
    if fresh:   # the cache is kept only while the city's feeds are the same folders and none changed since
        cand = pd.read_parquet(cache)
        log(f"candidates: {len(cand)} rows (cache)")
    else:
        aoi = gpd.read_file(HERE / city / "aoi.gpkg")[["geometry"]].to_crs(4326)
        sink: list = []
        rsink: list = []
        cand = pr.state_candidates(dirs, aoi, log=log, stops_sink=sink, routes_sink=rsink)
        cand.to_parquet(cache)
        save_routes(rsink, rfiles)
        if sink:
            pd.concat(sink, ignore_index=True).drop_duplicates(["stop_id", "station_lat", "station_lon"]).to_parquet(raw)
        log(f"candidates: {len(cand)} rows from {len(dirs)} feeds")
    stations, prof, cut = pr.select(cand, cfg, params)
    if raw.is_file():   # the real stops of each kept station
        key = {(a, b): i for i, (a, b) in enumerate(zip(stations.lat.round(6), stations.lon.round(6)))}
        r = pd.read_parquet(raw)
        r["station"] = [key.get((a, b), -1) for a, b in zip(r.station_lat.round(6), r.station_lon.round(6))]
        r = r[r.station >= 0].drop(columns=["station_lat", "station_lon"]).reset_index(drop=True)
        r["stop_name"] = r["stop_name"].astype(str)
        r.to_parquet(out / "stops_by_id.parquet")
    return stations, prof, cut


def save_routes(parts: list, files: list) -> None:
    """The editor's route data of every feed group (profiles.state_candidates routes_sink) as three parquet files."""
    import geopandas as gpd

    dep = pd.concat([p[0] for p in parts], ignore_index=True) if parts else pd.DataFrame(
        columns=["departure_time", "route_id", "sector", "rank", "dist_m", "time_s", "station_lat", "station_lon"])
    rt = pd.concat([p[1] for p in parts], ignore_index=True) if parts else pd.DataFrame(columns=["route_id", "mode"])
    sh = [p[2] for p in parts if len(p[2])]
    shp = gpd.GeoDataFrame(pd.concat(sh, ignore_index=True), geometry="geometry", crs=4326) if sh else gpd.GeoDataFrame(
        {"route_id": [], "n_trips": []}, geometry=[], crs=4326)
    dep.to_parquet(files[0]); rt.to_parquet(files[1]); shp.to_parquet(files[2])
    log(f"route data: {len(rt)} routes, {len(dep)} departures, {len(shp)} shapes")


def route_parts(city: str) -> list:
    """The saved route data of a city as one `export_routes` part (empty when run() has not written them)."""
    import geopandas as gpd
    from transitlos.scoring import direction_subset as ds

    out = HERE / city / "flex"
    tag = f"{ds.SPEED_METHOD}{ds.SPEED_WINDOW_MIN:g}{ds.SPEED_WINDOW_DIRECTION}"
    files = [out / f"route_departures_{tag}.parquet", out / f"routes_{tag}.parquet", out / f"route_shapes_{tag}.parquet"]
    if not all(p.is_file() for p in files):
        return []
    return [(pd.read_parquet(files[0]), pd.read_parquet(files[1]), gpd.read_parquet(files[2]))]


def run(city: str) -> dict:
    """One city end to end (see the module docstring).

    Args:
        city: City folder name.

    Returns:
        The run summary (also written to ``<city>/flex/run.json``).
    """
    from UrbanAccessAnalyzer.csr_network import CsrNetwork as Net

    from transitlos.flex import core, scoring as sc

    out = HERE / city / "flex"
    out.mkdir(parents=True, exist_ok=True)
    cfg, params = config(), sc.DEFAULT
    res: dict = {"city": city, "resolution": RES, "config": cfg.__dict__,
                 "feeds": sorted(p.name for p in (HERE / city / "gtfs").iterdir() if p.is_dir() and not p.name.startswith("_"))}

    stations, prof, cut = profiles(city, out, cfg, params)
    res["stations"] = int(len(stations))
    log(f"stations {len(stations)} (profiles per station {stations.n_profiles.value_counts().sort_index().to_dict()})")

    t = time.time()
    net = Net.load(HERE / city / "streets")
    log(f"network: {net.n} nodes, {len(net.eu)} edges ({time.time() - t:.0f}s)")
    sn, so, ss, ok = net.snap(stations.lat.to_numpy(), stations.lon.to_numpy())
    res["stations_snapped"] = int(ok.sum())
    log(f"snapped {int(ok.sum())}/{len(stations)} stations")

    t = time.time()
    ls, ld, ln, lc, stats = core.multilabel_search(net.indptr, net.adj, net.w, sn, so, ss, prof, cfg)
    res["t_search_s"] = round(time.time() - t, 1)
    res["search_stats"] = {"labels": int(stats[0]), "dropped_by_dominance": int(stats[1]), "node_cap_hits": int(stats[2])}
    log(f"search: {stats[0]} labels, {stats[1]} dominated, {stats[2]} cap hits ({res['t_search_s']}s)")

    t = time.time()
    reached = ln > 0
    cells, rec_off, ru, rv, ra, rb, rl = net.edge_cell_records(RES, edge_mask=reached[net.eu] | reached[net.ev])
    log(f"res-{RES} cells touched by reached streets: {len(cells)} ({time.time() - t:.0f}s)")

    t = time.time()
    cs, cd, cf, cr = core.process_cells(rec_off, ru, rv, ra, rb, rl, (ls, ld, ln, lc), prof, cfg, params)
    served = (cs >= 0).any(1)
    res.update(cells=int(len(cells)), cells_with_stops=int(served.sum()), cells_capped=int((cf & 1).sum()), t_cells_s=round(time.time() - t, 1),
               stations_with_cropped_profiles=int((stations.dropped > 0).sum()), profiles_cropped_total=int(stations.dropped.sum()))
    log(f"cells with stops {served.sum()}, capped {res['cells_capped']} ({res['t_cells_s']}s)")

    los, slot, pidx = core.recompute_cells(cs, cd, prof, params)

    st = pd.DataFrame({"lat": stations.lat, "lon": stations.lon, "cropped": stations.dropped.to_numpy().astype(np.uint8)})
    for j in range(prof.P):
        st[f"mode_{j}"] = prof.mode[:, j]; st[f"h_{j}"] = prof.h[:, j]; st[f"v_{j}"] = prof.v[:, j]; st[f"cut_{j}"] = cut[:, j]
    st.to_parquet(out / "stop_profiles.parquet")
    ids = np.array([f"{int(c):x}" for c in cells])
    cdf = pd.DataFrame({"h3": ids})
    for k in range(cfg.K):
        cdf[f"stop_{k}"] = cs[:, k]; cdf[f"dist_{k}"] = cd[:, k]
    cdf["flag"] = cf; cdf["cropped"] = cr
    cdf.to_parquet(out / "cell_stops.parquet")
    pd.DataFrame({"h3": ids, "los": los, "slot": slot, "profile": pidx}).to_parquet(out / "scores_default.parquet")
    (out / "params.json").write_text(json.dumps({"score": json.loads(params.to_json()), "config": cfg.__dict__, "resolution": RES}, indent=1))
    (out / "run.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"{city}: done")
    return res


if __name__ == "__main__":
    for c in sys.argv[1:]:
        run(c)
