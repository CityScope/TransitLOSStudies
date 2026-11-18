"""Automatic check for missing routes / frequencies of a city-science city (wrapper around `pyGTFSHandler.downloaders.utils.coverage_check`).

    python -m code.check_missing_transit guadalajara concepcion          # one or several cities; no argument = all eleven
    python -m code.check_missing_transit shanghai --no-osm               # feeds only (no Overpass)

For every city it checks the feeds in `<city>/gtfs/*` inside the official metro area (`<city>/metro_official.gpkg`):
flagged routes (no service / 1-2 trips / metro-tram-rail-bus slower than plausible), OSM route relations absent from the feeds, OSM stations
without a feed stop, OSM `interval` tags shorter than the feed headway, and the large low-score clusters of the last run
(`<city>/results/downloads/h3_res7.parquet`) with the OSM context of each. Output: `<city>/checks/missing_transit.json`
from the city's downloaded OSM extract (`streets/<pbf_filename>`, fetched from Geofabrik when missing; never Overpass) (a work list: see `pyGTFSHandler/docs/methodology/new_area_gtfs_agent_guide.md, section 7` for what to do with each finding).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CITIES = ["andorra", "beerseba", "boston", "concepcion", "gipuzkoa", "guadalajara", "hamburg", "san_francisco", "shanghai", "taipei", "toronto"]


def check_city(city: str, use_osm: bool = True, cache: str | None = None) -> dict:
    from code.city_config import CITY_CONFIGS
    from code.gtfs_fetch import GEOFABRIK_PBF
    import geopandas as gpd
    import h3
    import numpy as np
    import pandas as pd
    from pyGTFSHandler.downloaders.utils import coverage_check as cc
    from pyGTFSHandler.downloaders.utils.osm_extract import OsmData, download_extract

    folder = ROOT / city
    feeds = sorted(p for p in (folder / "gtfs").iterdir() if (p / "routes.txt").is_file())
    area = gpd.read_file(folder / "metro_official.gpkg").to_crs(4326)
    w, s, e, n = area.total_bounds
    osm_pbf = None
    if use_osm:
        pbf_name = CITY_CONFIGS[city].pbf_filename
        osm_pbf = ROOT / "streets" / pbf_name
        if not osm_pbf.is_file():   # the same extract the street network uses
            download_extract(GEOFABRIK_PBF[pbf_name], osm_pbf)
    rep = cc.run_check(feeds, (s, w, n, e), osm_pbf=osm_pbf, cache=cache)
    cells_file = folder / "results" / "downloads" / "h3_res7.parquet"
    if cells_file.is_file():
        d = pd.read_parquet(cells_file)
        ll = np.array([h3.cell_to_latlng(x) for x in d.h3_cell])
        d["lat"], d["lon"] = ll[:, 0], ll[:, 1]
        pts = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.lon, d.lat), crs=4326)
        pts = pts[pts.within(area.geometry.union_all())]
        pts["score"] = pts["level_of_service"]
        clusters = cc.underserved_clusters(pts, stops=cc.feed_stops(feeds))
        if osm_pbf:
            osm = OsmData(osm_pbf, (s, w, n, e), cache)   # cached by run_check above
            for c in clusters:
                c["osm"] = cc.cluster_context(c["bbox"], osm)
        rep["underserved_clusters"] = clusters
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cities", nargs="*")
    ap.add_argument("--no-osm", action="store_true")
    ap.add_argument("--cache", default=None, help="parsed-OSM cache folder (default: the shared TransitLOSStudies/cache/osm/coverage_check)")
    a = ap.parse_args()
    cache = a.cache or str(ROOT.parent / "cache" / "osm" / "coverage_check")
    for c in a.cities or CITIES:
        rep = check_city(c, not a.no_osm, cache)
        out = ROOT / c / "checks"
        out.mkdir(exist_ok=True)
        (out / "missing_transit.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        print(c, rep["summary"], "clusters:", len(rep.get("underserved_clusters", [])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
