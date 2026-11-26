"""Building footprints and heights of a city, as map tiles: `<city>/map_tiles/buildings/buildings.pmtiles` (3D view and the "Buildings" geometry).

    python flex/build_buildings.py <city> [<city> ...]

The footprints come from the most local source covering the city (pycensus.cadaster: an official cadaster or building dataset where there is
one, GlobalBuildingAtlas elsewhere). The area is the city's core polygon (`<city>/core.gpkg`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).resolve().parents[1]
ISO3 = {"andorra": "AND", "beerseba": "ISR", "boston": "USA", "concepcion": "CHL", "gipuzkoa": "ESP", "guadalajara": "MEX", "hamburg": "DEU",
        "san_francisco": "USA", "shanghai": "CHN", "taipei": "TWN", "toronto": "CAN"}


def build(city: str) -> dict:
    from geohierarchy.maps.buildings import write_building_tiles
    from pycensus import cadaster

    aoi = gpd.read_file(HERE / city / "core.gpkg")
    # the US source (NSI) is a point per structure, not a footprint: US cities use GlobalBuildingAtlas footprints
    b = cadaster.load_buildings(ISO3[city], aoi, sources=[] if ISO3[city] == "USA" else None)
    out = write_building_tiles(b, HERE / city / "map_tiles" / "buildings", min_zoom=14)
    return {"city": city, "buildings": len(b), "height_known": float(b["height_m"].notna().mean()) if "height_m" in b else 0.0, "file": str(out),
            "mb": round(out.stat().st_size / 1e6, 1)}


if __name__ == "__main__":
    for c in sys.argv[1:]:
        print(build(c), flush=True)
