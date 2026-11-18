"""`pop_density_1km` on every census-unit layer of the US study (state-wide layers and each county's block / block group / tract /
county files): residents-weighted mean of the state's res-11 cells' 1 km density, cells assigned by centre (transitlos.neighbourhood).

    python add_density_1km_census.py [State ...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

from transitlos.neighbourhood import density_1km_to_polygons, nearest_cell_density

DATA = Path(__file__).resolve().parent / "data"
SKIP = {"street", "stops", "routes", "routes_lines"}


def layers(sd: Path):
    for f in sorted(sd.glob("*.geoparquet")) + sorted(sd.glob("counties/*/*.geoparquet")):
        if f.stem.startswith("h3_") or f.stem in SKIP:
            continue
        yield f


def one_state(sd: Path) -> str:
    t = time.time()
    cells = {r: pd.read_parquet(sd / f"h3_res{r}.geoparquet", columns=["h3", "population", "pop_density_1km"])
             for r in (11, 10, 9, 8) if (sd / f"h3_res{r}.geoparquet").is_file()}
    n = 0
    for f in layers(sd):
        g = gpd.read_parquet(f)
        if "population" not in g.columns or "geometry" not in g.columns:
            continue
        g = g.drop(columns=["pop_density_1km"], errors="ignore")
        val = None
        for r, c in cells.items():   # finest cells first; polygons holding no populated centre take the next coarser resolution's
            v = density_1km_to_polygons(g[["geometry"]], c)["pop_density_1km"].to_numpy()
            val = v if val is None else pd.Series(val).fillna(pd.Series(v)).to_numpy()
            if not pd.isna(val).any():
                break
        miss = pd.isna(val)
        if miss.any() and 9 in cells:   # no populated cell centre inside at any resolution: the nearest res-9 cell's value
            val = pd.Series(val).to_numpy(dtype=float, copy=True)
            val[miss] = nearest_cell_density(g.loc[miss, ["geometry"]], cells[9])
        g["pop_density_1km"] = val
        g.to_parquet(f)
        n += 1
    return f"{sd.name}: {n} layers in {time.time() - t:.0f}s"


if __name__ == "__main__":
    want = sys.argv[1:]
    for sd in sorted(d for d in DATA.iterdir() if d.is_dir() and not d.name.startswith("_") and (d / "h3_res11.geoparquet").is_file()):
        if not want or sd.name in want:
            print(one_state(sd), flush=True)
    print("DENSITY_CENSUS_DONE")
