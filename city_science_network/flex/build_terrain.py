"""Elevation tiles of a city, served with its map (no public server needed at view time): `<city>/map_tiles/terrain/{z}/{x}/{y}.png`.

    python flex/build_terrain.py <city> [<city> ...]

One request per city to OpenTopography (Copernicus GLO-30, 30 m; the key is "opentopography" in ~/.config/ptlos/api_keys.json), cut into
terrarium-encoded PNG tiles, zoom 7 to 11 (about 55 m per pixel at mid latitudes). The 3D view of the flex map uses them for mountains and terrain.
Cities are done one after another with a pause: the server must never see a burst of requests.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).resolve().parents[1]
PAUSE_S = 5.0   # between cities


def build(city: str, zmin: int = 7, zmax: int = 11, margin: float = 0.15) -> dict:
    from pycensus.geography import dem as pdem

    core = gpd.read_file(HERE / city / "core.gpkg")
    dem = pdem.load(core, margin=margin)   # pycensus.dem: one request, cached
    w, s, e, n = core.to_crs(4326).total_bounds
    dx, dy = (e - w) * margin, (n - s) * margin
    out = HERE / city / "map_tiles" / "terrain"
    shutil.rmtree(out, ignore_errors=True)
    return {"city": city, "dem_mb": round(dem.stat().st_size / 1e6, 1), **pdem.write_terrarium_tiles(dem, out, (w - dx, s - dy, e + dx, n + dy), zmin, zmax)}


if __name__ == "__main__":
    for i, c in enumerate(sys.argv[1:]):
        if i:
            time.sleep(PAUSE_S)
        print(build(c), flush=True)
