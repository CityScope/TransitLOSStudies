"""Per-state data for the map's stats panels (distribution / regression / census), so every tab can show the nation, one
state, one county or one place -- and compare any two of them.

`stats_chunks/<State>.json.gz`: column-oriented res-7 cells of one state (`h3_cell`, `level_of_service`, `population`,
every census / ACS / jobs / election variable) plus `county` / `place` = index into `counties` / `places` (-1 = none).
`stats_chunks/index.json`: per state its name, bounding box, simplified outline (to find which state is on screen) and the
names of its counties / places (so the pickers work before any state file is fetched). The browser fetches a state file only
when that state is selected or on screen.
"""
import gzip
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STATS_CHUNK_DIRNAME = "stats_chunks"
CENSUS_NAMES_URL = "https://api.census.gov/data/2020/dec/pl"


def geo_names(cache_path: Path) -> dict:
    """`{"state": {fips: name}, "county": {geoid: name}, "place": {geoid: name}}` from the Census API (cached)."""
    if cache_path.is_file():
        return json.loads(cache_path.read_text())
    import requests

    from pycensus.api_keys import resolve_api_key

    key = resolve_api_key("us_census", "US_CENSUS_API_KEY")

    def rows(**q):
        r = requests.get(CENSUS_NAMES_URL, params={"get": "NAME", "key": key, **q}, timeout=120)
        r.raise_for_status()
        return r.json()[1:]

    names = {"state": {}, "county": {}, "place": {}}
    for name, st in rows(**{"for": "state:*"}):
        names["state"][st] = name
    for name, st, co in rows(**{"for": "county:*"}):
        names["county"][st + co] = name.split(",")[0]
    for st in list(names["state"]):
        try:
            for name, _st, pl in rows(**{"for": "place:*", "in": f"state:{st}"}):
                names["place"][st + pl] = name.split(",")[0]
        except Exception as exc:  # a state without places
            logger.warning(f"No place names for state {st}: {exc}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(names))
    return names


def _state_chunk(args) -> dict:
    state_dir, out_dir, names = args
    from .census_h3 import add_census_variables

    state_dir, out_dir = Path(state_dir), Path(out_dir)
    res7 = pd.read_parquet(state_dir / "h3_res7.geoparquet")
    res7 = add_census_variables(res7, state_dir)
    import h3

    cells = res7["h3"].tolist()
    ll = [h3.cell_to_latlng(c) for c in cells]
    pts = gpd.GeoDataFrame({"_i": np.arange(len(cells))}, geometry=gpd.points_from_xy([x[1] for x in ll], [x[0] for x in ll]), crs=4326)

    def assign(file, ids_names):
        p = state_dir / file
        if not p.is_file():
            return np.full(len(cells), -1), []
        g = gpd.read_parquet(p)[["GEOID", "geometry"]]
        j = gpd.sjoin(pts, g, how="left", predicate="within").drop_duplicates("_i").sort_values("_i")
        used = sorted(set(j["GEOID"].dropna()), key=lambda x: (ids_names.get(x, x), x))
        index = {gid: i for i, gid in enumerate(used)}
        idx = np.array([index.get(x, -1) if isinstance(x, str) else -1 for x in j["GEOID"]])
        pops = res7["population"].fillna(0).groupby(idx).sum()
        return idx, [{"id": gid, "name": ids_names.get(gid, gid), "pop": int(pops.get(index[gid], 0))} for gid in used]

    county_idx, counties = assign("county.geoparquet", names["county"])
    place_idx, places = assign("place.geoparquet", names["place"])

    cols = {"h3_cell": cells}
    for c in res7.columns:
        if c != "h3" and pd.api.types.is_numeric_dtype(res7[c]):
            v = res7[c].to_numpy(float).round(2)
            cols[c] = [None if not np.isfinite(x) else float(x) for x in v]
    cols["county"] = county_idx.tolist()
    cols["place"] = place_idx.tolist()
    out_dir.mkdir(parents=True, exist_ok=True)
    key = state_dir.name
    with gzip.open(out_dir / f"{key}.json.gz", "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(cols, f, separators=(",", ":"))

    st = gpd.read_parquet(state_dir / "state.geoparquet")
    fips = str(st["GEOID"].iloc[0])[:2]
    geom = st.geometry.union_all().simplify(0.02)
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    rings = [[[round(x, 3), round(y, 3)] for x, y in p.exterior.coords] for p in polys]
    return {
        "key": key, "name": names["state"].get(fips, re.sub(r"(?<=[a-z])(?=[A-Z])", " ", key).replace("of", " of ").replace("  ", " ")),
        "bbox": [round(v, 3) for v in st.total_bounds], "rings": rings, "counties": counties, "places": places, "n": len(cells), "pop": int(res7["population"].fillna(0).sum()),
    }


def build_stats_chunks(state_dirs, out_dir: Path, names_cache: Path, workers: int = 10) -> dict:
    out = Path(out_dir) / STATS_CHUNK_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    names = geo_names(names_cache)
    with ProcessPoolExecutor(workers) as ex:
        states = list(ex.map(_state_chunk, [(str(d), str(out), names) for d in state_dirs]))
    index = {"states": sorted(states, key=lambda s: s["name"])}
    (out / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    logger.info(f"Stats chunks: {len(states)} states, {sum(s['n'] for s in states)} cells -> {out}")
    return index


NATION_SAMPLE_CELLS = 100_000
NATION_SEED = 7


def build_nation_sample(res7: pd.DataFrame, out_dir: Path, n_cells: int = NATION_SAMPLE_CELLS, seed: int = NATION_SEED) -> dict:
    """`stats_chunks/Nation.json.gz`: the nation-level stats data at the SAME resolution as the state files (res 7).

    The full national res-7 table (~1.3M cells x ~70 variables) is too big for a browser, so the nation scope uses a seeded
    uniform random sample of its cells. Count columns (population, jobs, votes, ...) are multiplied by 1/fraction, so totals
    and every population-weighted statistic stay unbiased; shares / medians / densities / level of service are unchanged, so the
    unweighted regression and median splits are unbiased estimates of the full-grid result. `factor` is stored so the page can scale
    the effect of a drawn line the same way. `res7` = the merged national res-7 table (column `h3` + variables).
    """
    from .state_pipeline import _column_agg_map

    try:
        agg_map = _column_agg_map()
    except Exception:
        agg_map = {}
    is_mean = lambda c: agg_map.get(c) == "mean" or c.endswith(("_share", "_rate")) or any(  # noqa: E731
        m in c.lower() for m in ("median", "mean_", "_mean", "density", "_index", "_ratio"))
    df = (res7[res7["population"].fillna(0) > 0] if "population" in res7.columns else res7).reset_index(drop=True)
    pop = df["population"].fillna(0).to_numpy(float)
    los = df["level_of_service"].to_numpy(float)
    # Strata: population bucket x level-of-service bucket. A uniform fraction is drawn inside each stratum (so unweighted
    # statistics such as the regression stay unbiased) and the count columns are rescaled PER STRATUM so every stratum keeps
    # its exact population: weighted means / histograms then match the full grid almost exactly.
    pop_bucket = pd.cut(pop, np.quantile(pop, [0, .5, .7, .8, .9, .95, .98, .99, .995, 1]) + np.arange(10) * 1e-9, labels=False, include_lowest=True)
    los_bucket = np.digitize(np.nan_to_num(los, nan=-1), [0.0001] + list(range(5, 100, 5)))  # no data / 0 / 5-point steps
    df["_stratum"] = pop_bucket * 100 + los_bucket
    frac = min(n_cells, len(df)) / len(df)
    rng = np.random.default_rng(seed)
    keep = []
    for _, idx in df.groupby("_stratum").indices.items():
        k = max(1, int(round(len(idx) * frac)))
        keep.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    sample = df.iloc[np.concatenate(keep)].sort_values("h3")
    full_pop = df.groupby("_stratum")["population"].sum()
    samp_pop = sample.groupby("_stratum")["population"].sum()
    scale = (full_pop / samp_pop.where(samp_pop > 0)).fillna(1.0)
    row_scale = sample["_stratum"].map(scale).to_numpy(float)
    n = len(sample)
    factor = len(df) / n
    cols = {"h3_cell": sample["h3"].tolist()}
    for c in sample.columns:
        if c in ("h3", "_stratum") or not pd.api.types.is_numeric_dtype(sample[c]):
            continue
        v = sample[c].to_numpy(float)
        if c != "level_of_service" and not is_mean(c):
            v = v * row_scale
        v = v.round(2)
        cols[c] = [None if not np.isfinite(x) else float(x) for x in v]
    cols["factor"] = factor
    out = Path(out_dir) / STATS_CHUNK_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "Nation.json.gz", "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(cols, f, separators=(",", ":"))
    logger.info(f"Nation stats sample: {n} of {len(df)} res-7 cells (x{factor:.2f}) -> {out / 'Nation.json.gz'}")
    return {"cells": n, "of": len(df), "factor": factor}
