"""Neighbourhood densities (residents per km2 within 500 m / 1 / 2 / 3 km: transitlos.neighbourhood.DENSITY_RADII) on every h3_res*
layer and every census-unit layer of the study, without rerunning the roll-ups (the pipeline writes the H3 columns from now on).
Census layers: residents-weighted mean of the cells' densities, cells assigned by centre, finest resolution first (11 -> 8), then the
nearest res-9 cell for polygons with no populated cell centre.

    python add_densities.py [State ...]      (one State per process: run under memsched for parallelism)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from transitlos.neighbourhood import DENSITY_COLUMNS, add_densities, densities_to_polygons, nearest_cell_density

DATA = Path(__file__).resolve().parent / "data"
SKIP = {"street", "stops", "routes", "routes_lines"}
COLS = list(DENSITY_COLUMNS)


def layers(sd: Path):
    for f in sorted(sd.glob("*.geoparquet")) + sorted(sd.glob("counties/*/*.geoparquet")):
        if f.stem.startswith("h3_") or f.stem in SKIP:
            continue
        yield f


def one(sd: Path) -> str:
    t = time.time()
    cells = {}
    for f in sorted(sd.glob("h3_res*.geoparquet")):
        d = pd.read_parquet(f)
        if "population" not in d.columns or "h3" not in d.columns:
            continue
        d = add_densities(d, "h3", "population")
        d.to_parquet(f)
        r = int(f.stem.replace("h3_res", ""))
        if r in (8, 9, 10, 11):
            cells[r] = d[["h3", "population"] + COLS]
    cells = dict(sorted(cells.items(), reverse=True))   # finest first
    n = 0
    for f in layers(sd):
        g = gpd.read_parquet(f)
        if "population" not in g.columns or "geometry" not in g.columns:
            continue
        g = g.drop(columns=COLS, errors="ignore")
        val = None
        for r, c in cells.items():   # polygons holding no populated centre take the next coarser resolution's
            v = densities_to_polygons(g[["geometry"]], c, COLS)[COLS].to_numpy(dtype=float)
            val = v if val is None else np.where(np.isnan(val), v, val)
            if not np.isnan(val).any():
                break
        miss = np.isnan(val).any(axis=1)
        if miss.any() and 9 in cells:   # no populated cell centre inside at any resolution: the nearest res-9 cell's values
            val[miss] = nearest_cell_density(g.loc[miss, ["geometry"]], cells[9], dens_col=COLS)
        for i, c in enumerate(COLS):
            g[c] = val[:, i]
        g.to_parquet(f)
        n += 1
    return f"{sd.name}: {len(cells)} h3 + {n} census layers in {time.time() - t:.0f}s"


if __name__ == "__main__":
    want = sys.argv[1:]
    for sd in sorted(d for d in DATA.iterdir() if d.is_dir() and not d.name.startswith("_") and (d / "h3_res11.geoparquet").is_file()):
        if not want or sd.name in want:
            print(one(sd), flush=True)
    print("DENSITIES_DONE")
