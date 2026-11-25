"""Fit data of one city science map for the flex fits (`fit_tests.py`): outcome units, their populated res-11 cells and the cells' service items.

    python flex/build_fit_data.py [city ...]          # every city with an outcome table by default

Reads `<city>/results/flex_outcomes.parquet` (outcomes.py), `<city>/results/downloads/h3_res11.parquet` (population and census columns per
res-11 cell) and `<city>/flex/` (run_city_flex.py: station profiles and the 15 nearest stations of every cell). Writes `<city>/flex/fit/`:
    units.parquet   one row per outcome unit: GEOID, y / n (transit and transit + car, the counts the likelihood sees), weight (people the unit
                    stands for), fold (spatial: the unit's res-7 parent hexagon), the raw control columns (CONTROLS), population, cells,
                    los_default (the map's default score, population-weighted over the unit's cells)
    items.npz       it_cell, it_mode (0 bus, 1 tram, 2 rail), it_h (min), it_v (km/h, clipped per mode as the US fit), it_d (m, street network);
                    cell_unit, cell_pop (cells of the units; cells without stops have no items and access 0)

An item is one service option of one of the cell's stations. Options of the same mode with the same headway and speed at several stations
(one line stopping at several nearby stations) are kept once, at the nearest station: with curves falling in distance the farther ones can
never be the cell's best.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
CELL_RES, FOLD_RES, N_FOLDS = 11, 7, 5
SPEED_CLIP = {0: (5.0, 70.0), 1: (5.0, 60.0), 2: (10.0, 130.0)}   # bus, tram, rail (km/h): as the US fit (FIT4_SPEED_CLIP)
# control -> (candidate columns of the res-11 table, transform). Car ownership is left out (it is an outcome of the service, as in the US fit).
CONTROLS = {
    "log_income": (["medianHouseholdIncome"], "log"),
    "poverty_share": (["poverty_share"], ""),
    "university_share": (["university_share"], ""),
    "basic_education_share": (["education_basic_or_less_share"], ""),
    "unemployment_rate": (["unemployment_rate"], ""),
    "under18_share": (["under18_share"], ""),
    "over65_share": (["over65_share"], ""),
    "single_person_share": (["singlePersonHouseholds/households"], ""),
    "renter_share": (["renter_households_share", "acs_renter_rate", "statcan_renter_rate", "cbs_renter_rate"], ""),
    "foreign_born_share": (["foreign_born_share"], ""),
    "log_density_1km": (["pop_density_1km"], "log1p"),
    "left_share": (["elec_national_left_share"], ""),
    "old_housing_share": (["dwellings_built_before_1945_share"], ""),
}
POP = ("census_residents", "population")


def _cells(city: str) -> pd.DataFrame:
    """Populated res-11 cells with their population and the raw control columns."""
    import pyarrow.parquet as pq

    f = HERE / city / "results" / "downloads" / "h3_res11.parquet"
    names = set(pq.read_schema(f).names)
    pop = next(c for c in POP if c in names)
    want = {pop}
    for cands, _ in CONTROLS.values():
        for c in cands:
            want.update(c.split("/"))
    t = pd.read_parquet(f, columns=["h3_cell", *sorted(want & names)]).drop_duplicates("h3_cell")
    t = t.rename(columns={pop: "pop"})
    t["pop"] = pd.to_numeric(t["pop"], errors="coerce").fillna(0.0)
    t = t[t["pop"] > 0].reset_index(drop=True)
    out = pd.DataFrame({"h3": t["h3_cell"].astype(str), "pop": t["pop"].astype(float)})
    for name, (cands, tr) in CONTROLS.items():
        src = next((c for c in cands if all(p in t.columns for p in c.split("/"))), None)
        if src is None:
            continue
        if "/" in src:
            a, b = src.split("/")
            v = pd.to_numeric(t[a], errors="coerce") / pd.to_numeric(t[b], errors="coerce").where(lambda s: s > 0)
        else:
            v = pd.to_numeric(t[src], errors="coerce")
        if tr == "log":
            v = np.log(v.where(v > 0))
        elif tr == "log1p":
            v = np.log1p(v.clip(lower=0))
        out[name] = v.astype(float)
    return out


def _utm(g) -> str:
    """UTM zone of the area's centre (units of the outer region can span two zones, where geopandas' estimate gives up)."""
    b = g.to_crs(4326).total_bounds
    lon, lat = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return "EPSG:%d" % ((32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1)


def _assign(cells: pd.DataFrame, units) -> np.ndarray:
    """Unit index of each cell (its centre in the unit's polygon), -1 outside every unit."""
    import geopandas as gpd
    import h3

    ll = np.array([h3.cell_to_latlng(h) for h in cells["h3"]])
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(ll[:, 1], ll[:, 0]), crs=4326)
    j = gpd.sjoin(pts, units[["geometry"]].reset_index(drop=True).reset_index(names="unit"), how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    u = j["unit"].fillna(-1).astype(int).reindex(range(len(cells))).to_numpy().copy()
    miss = np.nonzero(u < 0)[0]
    if len(miss):   # units drawn without the streets (Chile's manzanas): a cell centre on the street goes to the nearest unit within 100 m
        crs = _utm(units)
        q = gpd.sjoin_nearest(pts.iloc[miss].to_crs(crs), units[["geometry"]].reset_index(drop=True).reset_index(names="unit").to_crs(crs),
                              how="left", max_distance=100.0)
        q = q[~q.index.duplicated(keep="first")]
        u[q.index.to_numpy()] = q["unit"].fillna(-1).astype(int).to_numpy()
    return u


def _items(city: str, cell_h3: np.ndarray):
    """Service items of the given cells (see the module docstring)."""
    src = HERE / city / "flex"
    st = pd.read_parquet(src / "stop_profiles.parquet")
    P = sum(1 for c in st.columns if c.startswith("mode_"))
    mode = np.column_stack([st[f"mode_{j}"] for j in range(P)]).astype(np.int16)
    hh = np.column_stack([st[f"h_{j}"] for j in range(P)]).astype(np.float32)
    vv = np.column_stack([st[f"v_{j}"] for j in range(P)]).astype(np.float32)
    cs = pd.read_parquet(src / "cell_stops.parquet")
    K = sum(1 for c in cs.columns if c.startswith("stop_"))
    pos = pd.Series(np.arange(len(cs)), index=cs["h3"].astype(str))
    ci = pos.reindex(cell_h3).to_numpy()
    has = np.isfinite(ci)
    rows = ci[has].astype(np.int64)
    cell_idx = np.nonzero(has)[0]
    S = np.column_stack([cs[f"stop_{k}"].to_numpy() for k in range(K)])[rows]
    D = np.column_stack([cs[f"dist_{k}"].to_numpy() for k in range(K)])[rows].astype(np.float32)
    c_rep, k_rep = np.nonzero(S >= 0)
    stn, dist, cell = S[c_rep, k_rep], D[c_rep, k_rep], cell_idx[c_rep]
    out = []
    for j in range(P):
        m = mode[stn, j]
        ok = (m >= 0) & (hh[stn, j] > 0)
        out.append(pd.DataFrame({"cell": cell[ok], "mode": m[ok], "h": hh[stn[ok], j], "v": vv[stn[ok], j], "d": dist[ok]}))
    it = pd.concat(out, ignore_index=True)
    n0 = len(it)
    it = it.sort_values("d", kind="stable").drop_duplicates(["cell", "mode", "h", "v"], keep="first").sort_values(["cell", "mode"], kind="stable")
    lo = it["mode"].map({k: v[0] for k, v in SPEED_CLIP.items()}).to_numpy(np.float32)
    hi = it["mode"].map({k: v[1] for k, v in SPEED_CLIP.items()}).to_numpy(np.float32)
    it["v"] = np.clip(it["v"].to_numpy(np.float32), lo, hi)
    return it.reset_index(drop=True), n0


def build(city: str) -> dict:
    """Writes `<city>/flex/fit/` (see the module docstring).

    Args:
        city: City folder name.

    Returns:
        Counts of the build.
    """
    import geopandas as gpd
    import h3

    units = gpd.read_parquet(HERE / city / "results" / "flex_outcomes.parquet").to_crs(4326).reset_index(drop=True)
    units = units[np.isfinite(units["share_tc"])].reset_index(drop=True)
    cells = _cells(city)
    cells["unit"] = _assign(cells, units)
    cells = cells[cells["unit"] >= 0].reset_index(drop=True)
    it, n0 = _items(city, cells["h3"].to_numpy())
    out = HERE / city / "flex" / "fit"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "items.npz", it_cell=it["cell"].to_numpy(np.int32), it_mode=it["mode"].to_numpy(np.int8),
                        it_h=it["h"].to_numpy(np.float32), it_v=it["v"].to_numpy(np.float32), it_d=it["d"].to_numpy(np.float32),
                        cell_unit=cells["unit"].to_numpy(np.int32), cell_pop=cells["pop"].to_numpy(np.float32))
    # unit table: outcome counts, people represented, spatial fold, controls (population-weighted means over the unit's cells)
    share = units["share_tc"].to_numpy(float)
    n_s = pd.to_numeric(units["n_sample_tc"], errors="coerce").to_numpy(float)
    n_e = pd.to_numeric(units["n_tc"], errors="coerce").to_numpy(float)
    n_lik = np.where(np.isfinite(n_s) & (n_s > 0), n_s, n_e)          # the people actually counted / surveyed
    pop_u = np.bincount(cells["unit"], weights=cells["pop"], minlength=len(units))
    weight = np.where(np.isfinite(n_e) & (n_e > 0), n_e, pop_u)           # the people a unit stands for
    cen = units.geometry.to_crs(_utm(units)).centroid.to_crs(4326)
    parent = [h3.latlng_to_cell(p.y, p.x, FOLD_RES) for p in cen]
    rng = np.random.default_rng(7)
    up = sorted(set(parent))
    fold_of = dict(zip(up, rng.integers(0, N_FOLDS, len(up))))
    tab = pd.DataFrame({"GEOID": units["GEOID"].astype(str), "share": share, "n": n_lik, "y": share * n_lik, "weight": weight,
                        "pop": pop_u, "cells": np.bincount(cells["unit"], minlength=len(units)), "fold": [int(fold_of[p]) for p in parent],
                        "basis": units["basis"].astype(str)})
    wpop = cells["pop"].to_numpy(float)
    # the map's default score (transitlos.flex DEFAULT) per unit, population-weighted over its cells (0 where no stop is reached)
    sc = pd.read_parquet(HERE / city / "flex" / "scores_default.parquet")
    los = pd.Series(sc["los"].to_numpy(float), index=sc["h3"].astype(str)).reindex(cells["h3"]).fillna(0.0).to_numpy()
    tab["los_default"] = np.bincount(cells["unit"], weights=los * wpop, minlength=len(units)) / np.maximum(pop_u, 1e-9)
    for name in CONTROLS:
        if name not in cells.columns:
            continue
        v = cells[name].to_numpy(float)
        ok = np.isfinite(v)
        num = np.bincount(cells["unit"][ok], weights=(v * wpop)[ok], minlength=len(units))
        den = np.bincount(cells["unit"][ok], weights=wpop[ok], minlength=len(units))
        tab[name] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    tab.to_parquet(out / "units.parquet")
    res = {"city": city, "units": len(tab), "units_with_cells": int((tab["cells"] > 0).sum()), "cells": len(cells),
           "items_raw": int(n0), "items": len(it), "controls": [c for c in CONTROLS if c in tab.columns]}
    print(res, flush=True)
    return res


if __name__ == "__main__":
    cities = sys.argv[1:] or sorted(p.parents[1].name for p in HERE.glob("*/results/flex_outcomes.parquet"))
    for c in cities:
        build(c)
