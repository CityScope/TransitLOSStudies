"""Per-state browser data for the flexible-scoring map: station profiles + cell/station tables + CENSUS COUNTS + cell geometry.

    python build_web.py [VT NY ...]          # default: every state that has a summary.json

For each state, reads data/<State>/{stop_profiles.parquet, <GEOID>/cell_stops.parquet} (this prototype's results) and the ORIGINAL study's county polygons and census
blocks (read-only: population, jobs (LODES), age groups), keeps each cell only in the county that contains its centre (the networks overlap at borders),
spreads the block counts over the cells (transitlos.flex.catchment), computes the default catchment of every stop, and writes
    web/<State>/...                          (transitlos.flex.export.export_for_browser) and web/index.json
    data/<State>/stop_catchment.parquet      stop lat/lon + the summed counts of the cells each stop wins + number of cells won
Open the map with:  python -m http.server -d web 8000   ->   http://localhost:8000/viewer.html   (viewer.html + scripts are copied into web/)
"""
import json
import shutil
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / "usa_study"
sys.path.insert(0, str(HERE))
import run_county as rc  # noqa: E402
from transitlos.flex import core, scoring as sc, catchment as ct  # noqa: E402
from transitlos.flex.cells import CellStops  # noqa: E402
from transitlos.flex.export import export_for_browser, publish_frontend, FRONTEND  # noqa: E402

COUNT_COLUMNS = {   # block column -> label
    "population": "Population",
    "lodes_wac_jobsTotal": "Jobs (LODES workplace)",
    "lodes_wac_jobsIncomeLow": "Jobs, low wage",
    "lodes_wac_jobsIncomeMid": "Jobs, mid wage",
    "lodes_wac_jobsIncomeHigh": "Jobs, high wage",
    "under18Population": "Population under 18",
    "over65Population": "Population 65+",
    "commutersResidents": "Commuters (residents, not working at home)",
    "commuteMinutesResidents": "Commute minutes (residents)",
    "commuteMinutesWorkers": "Commute minutes (workers)",
    "commuteKmResidents": "Commute km (residents)",
    "commuteKmWorkers": "Commute km (workers)",
}
#: Shown as means (scale x sum of the numerator / sum of the denominator over a cell or a stop's catchment), never as sizes: the helper sums are hidden.
MEAN_COLUMNS = [("Mean commute time of residents (min)", "commuteMinutesResidents", "commutersResidents", 1.0),
                ("Mean commute time of workers (min)", "commuteMinutesWorkers", "lodes_wac_jobsTotal", 1.0),
                ("Mean commute distance of residents (km)", "commuteKmResidents", "commutersResidents", 1.0),
                ("Mean commute distance of workers (km)", "commuteKmWorkers", "lodes_wac_jobsTotal", 1.0),
                ("Mean commute speed of residents (km/h)", "commuteKmResidents", "commuteMinutesResidents", 60.0),
                ("Mean commute speed of workers (km/h)", "commuteKmWorkers", "commuteMinutesWorkers", 60.0)]
HIDDEN_COLUMNS = ["commutersResidents", "commuteMinutesResidents", "commuteMinutesWorkers", "commuteKmResidents", "commuteKmWorkers"]
BG_TABLE = HERE.parent / "cache" / "derived" / "bg_table.parquet"


def _bg_commute():
    """Block-group commute times (build_bg_table.py) for the block-level helper counts."""
    t = pd.read_parquet(BG_TABLE, columns=["GEOID", "population", "commuters_not_wfh", "commute_time_residents_min", "commute_time_workers_min",
                                           "commute_km_residents", "commute_km_workers"]).set_index("GEOID")
    return t


def add_commute_counts(b, bgc):
    """Block-level commuters, residents' commute minutes (block share of its block group's commuters x the block group's mean) and workers' commute minutes (block jobs x mean)."""
    g = b["GEOID"].astype(str).str[:12]
    pop = pd.to_numeric(b["population"], errors="coerce").fillna(0).to_numpy() if "population" in b.columns else np.zeros(len(b))
    bgpop = g.map(bgc.population).astype(float).to_numpy()
    commuters = g.map(bgc.commuters_not_wfh).astype(float).to_numpy() * np.where(bgpop > 0, pop / np.where(bgpop > 0, bgpop, 1), 0.0)
    tres = g.map(bgc.commute_time_residents_min).astype(float).to_numpy()
    twp = g.map(bgc.commute_time_workers_min).astype(float).to_numpy()
    jobs = pd.to_numeric(b["lodes_wac_jobsTotal"], errors="coerce").fillna(0).to_numpy() if "lodes_wac_jobsTotal" in b.columns else np.zeros(len(b))
    kres = g.map(bgc.commute_km_residents).astype(float).to_numpy()
    kwp = g.map(bgc.commute_km_workers).astype(float).to_numpy()
    ok_r = np.isfinite(tres) & np.isfinite(commuters) & np.isfinite(kres)   # time, distance and speed all over the same commuters
    ok_w = np.isfinite(twp) & np.isfinite(kwp)
    b = b.copy()
    b["commutersResidents"] = np.where(ok_r, commuters, 0.0)
    b["commuteMinutesResidents"] = np.where(ok_r, commuters * np.nan_to_num(tres), 0.0)
    b["commuteKmResidents"] = np.where(ok_r, commuters * np.nan_to_num(kres), 0.0)
    b["commuteMinutesWorkers"] = np.where(ok_w, jobs * np.nan_to_num(twp), 0.0)
    b["commuteKmWorkers"] = np.where(ok_w, jobs * np.nan_to_num(kwp), 0.0)
    return b


def write_census_units(web: Path, cell_gid, bg_pop: dict) -> None:
    """cell_bg.bin / cell_tract.bin (int32 local index per cell, -1 = none), bg_pop.bin / tract_pop.bin (census population) and their GEOIDs in meta.json:
    the viewer's block-group and tract views take sum(cell population x cell score) / census population of the unit (cells out of reach count 0)."""
    gid = np.asarray(cell_gid, dtype=object)
    bgs = sorted({g for g in gid if g})
    bi = {g: i for i, g in enumerate(bgs)}
    tracts = sorted({g[:11] for g in bgs})
    ti = {t: i for i, t in enumerate(tracts)}
    np.array([bi.get(g, -1) for g in gid], dtype=np.int32).tofile(web / "cell_bg.bin")
    np.array([ti.get(g[:11], -1) if g else -1 for g in gid], dtype=np.int32).tofile(web / "cell_tract.bin")
    bp = np.array([bg_pop.get(g, 0.0) for g in bgs], dtype=np.float32); bp.tofile(web / "bg_pop.bin")
    tp = np.zeros(len(tracts), np.float32)
    np.add.at(tp, np.array([ti[g[:11]] for g in bgs], dtype=np.int64), bp); tp.tofile(web / "tract_pop.bin")
    meta = json.loads((web / "meta.json").read_text())
    meta.update({"bg_ids": bgs, "tract_ids": tracts})
    (web / "meta.json").write_text(json.dumps(meta))


def build_state(abbr: str, web_root: Path) -> dict:
    name = rc.STATE_DIRS[abbr]
    out_state = HERE / "data" / name
    st = pd.read_parquet(out_state / "stop_profiles.parquet")
    P = sum(1 for c in st.columns if c.startswith("mode_"))
    prof = core.Profiles(np.column_stack([st[f"mode_{j}"] for j in range(P)]), np.column_stack([st[f"h_{j}"] for j in range(P)]), np.column_stack([st[f"v_{j}"] for j in range(P)]))
    cell_parts, station_parts, dist_parts, count_parts, cols_used = [], [], [], [], None
    bg_parts, bg_pop = [], {}
    bgc = _bg_commute()
    for cdir in sorted(p for p in out_state.iterdir() if p.is_dir() and p.name.isdigit() and (p / "cell_stops.parquet").is_file()):
        cs = pd.read_parquet(cdir / "cell_stops.parquet")
        if cs.empty:
            continue
        cells = np.array([int(h, 16) for h in cs["h3"]], dtype=np.uint64)
        county = gpd.read_parquet(STUDY / "data" / name / "counties" / cdir.name / "county.geoparquet")[["geometry"]].to_crs(4326)
        lat, lon = ct.cell_centres(cells)
        inside = gpd.sjoin(gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326), county, how="left", predicate="within")["index_right"].notna().to_numpy()
        inside = inside[: len(cells)] if len(inside) >= len(cells) else np.r_[inside, np.zeros(len(cells) - len(inside), bool)]
        cells, cs = cells[inside], cs[inside].reset_index(drop=True)
        if not len(cells):
            continue
        K = sum(1 for c in cs.columns if c.startswith("stop_"))
        bpath = STUDY / "data" / name / "counties" / cdir.name / "block.geoparquet"
        if bpath.is_file():
            b = add_commute_counts(gpd.read_parquet(bpath), bgc)
            use = [c for c in COUNT_COLUMNS if c in b.columns]
            cnt = ct.block_counts_to_cells(b, cells, use)
            cnt = cnt.reindex(columns=list(COUNT_COLUMNS), fill_value=0.0).fillna(0.0)
        else:
            cnt = pd.DataFrame(0.0, index=range(len(cells)), columns=list(COUNT_COLUMNS))
        # the census block group under every cell centre (block-group / tract view of the map: census-population-weighted means of the cell scores)
        bgp = STUDY / "data" / name / "counties" / cdir.name / "blockgroup.geoparquet"
        gid = np.full(len(cells), "", dtype=object)
        if bgp.is_file():
            bgs = gpd.read_parquet(bgp, columns=["GEOID", "population", "geometry"]).to_crs(4326).reset_index(drop=True)
            clat, clon = ct.cell_centres(cells)
            j = gpd.sjoin(gpd.GeoDataFrame({"_i": np.arange(len(cells))}, geometry=gpd.points_from_xy(clon, clat), crs=4326), bgs, how="left", predicate="within")
            j = j.drop_duplicates("_i").sort_values("_i")
            gid = j["GEOID"].fillna("").astype(str).to_numpy()
            bg_pop.update(dict(zip(bgs["GEOID"].astype(str), pd.to_numeric(bgs["population"], errors="coerce").fillna(0).astype(float))))
        bg_parts.append(gid)
        cell_parts.append(cells)
        station_parts.append(np.column_stack([cs[f"stop_{k}"] for k in range(K)]).astype(np.int32))
        dist_parts.append(np.column_stack([cs[f"dist_{k}"] for k in range(K)]).astype(np.uint16))
        count_parts.append(cnt)
    cells = np.concatenate(cell_parts)
    order = np.argsort(cells, kind="stable")
    cells, station, dist = cells[order], np.vstack(station_parts)[order], np.vstack(dist_parts)[order]
    counts = pd.concat(count_parts, ignore_index=True).iloc[order].reset_index(drop=True)
    cell_gid = np.concatenate(bg_parts)[order]
    _, first = np.unique(cells, return_index=True)               # a cell can sit in one county only; guard anyway
    cells, station, dist, counts = cells[first], station[first], dist[first], counts.iloc[first].reset_index(drop=True)
    cell_gid = cell_gid[first]
    res = CellStops(cells, station, dist, np.zeros(len(cells), np.uint8), np.zeros(len(cells), np.uint8), {})
    web = web_root / name
    meta = export_for_browser(web, st.lat.to_numpy(), st.lon.to_numpy(), prof, res, sc.DEFAULT, counts=counts, count_labels=COUNT_COLUMNS,
                              extra_meta={"name": name, "hidden_columns": [list(COUNT_COLUMNS).index(c) for c in HIDDEN_COLUMNS],
                                          "mean_columns": [{"label": l, "num": list(COUNT_COLUMNS).index(n), "den": list(COUNT_COLUMNS).index(d), "scale": k}
                                                           for l, n, d, k in MEAN_COLUMNS]},
                              points=pd.read_parquet(out_state / "stops_by_id.parquet") if (out_state / "stops_by_id.parquet").is_file() else None)
    write_census_units(web, cell_gid, bg_pop)
    # default catchment of every stop
    los, slot, _ = core.recompute_cells_fast(station, dist, prof, sc.DEFAULT)
    winner = ct.winner_station(station, slot)
    tot = ct.stop_catchment(winner, counts.to_numpy(), prof.S)
    cat = pd.DataFrame(tot, columns=list(counts.columns)); cat.insert(0, "lat", st.lat.to_numpy()); cat.insert(1, "lon", st.lon.to_numpy()); cat["cells_won"] = ct.cells_won(winner, prof.S)
    cat.to_parquet(out_state / "stop_catchment.parquet")
    pop = counts["population"].to_numpy()
    served = slot >= 0
    return {"name": name, "path": f"{name}/", "bbox": meta.get("bbox"), "stations": int(prof.S), "cells": int(len(cells)), "population_in_reached_cells": float(pop[served].sum()),
            "population_attributed_to_stops": float(tot[:, list(counts.columns).index("population")].sum())}


def write_index_version(web_root: Path) -> str:
    """index.json['version']: a hash of every state's meta.json and file sizes. The viewer puts it in the URL of all state data, so changed data is never read from an older cache."""
    import hashlib
    idx_path = web_root / "index.json"; idx = json.loads(idx_path.read_text()); h = hashlib.sha1()
    for s_ in idx["states"]:
        d = web_root / s_["path"]
        h.update((d / "meta.json").read_bytes())
        for f in sorted(d.iterdir()):
            h.update(f"{f.name}:{f.stat().st_size}".encode())
    idx["version"] = h.hexdigest()[:12]
    idx_path.write_text(json.dumps(idx))
    return idx["version"]


def main():
    abbrs = [a for a in sys.argv[1:] if a.isalpha()] or [a for a, n in rc.STATE_DIRS.items() if (HERE / "data" / n / "summary.json").is_file()]
    web_root = HERE / "web"; web_root.mkdir(exist_ok=True)
    idx_path = web_root / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.is_file() else {"states": []}
    for a in abbrs:
        t = time.time()
        try:
            r = build_state(a, web_root)
        except Exception as exc:
            print(a, "FAILED", repr(exc), flush=True); continue
        idx["states"] = [s for s in idx["states"] if s["name"] != r["name"]] + [r]
        idx["states"].sort(key=lambda s: s["name"])
        idx_path.write_text(json.dumps(idx))
        print(f"{a}: {r['stations']} stations, {r['cells']} cells, population attributed {r['population_attributed_to_stops']:.0f} of {r['population_in_reached_cells']:.0f}  ({time.time()-t:.0f}s)", flush=True)
    print("data version", write_index_version(web_root), flush=True)
    publish_frontend(web_root)


if __name__ == "__main__":
    main()
