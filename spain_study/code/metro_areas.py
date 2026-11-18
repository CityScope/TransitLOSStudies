"""Spanish metropolitan areas: municipalities assigned to Eurostat GISCO URAU 2024 functional urban areas (FUA) by centroid-in-polygon.

`build_metro_map(data_dir, out_json)` writes {"areas": {code: name}, "muni": {municipality INE id: code}}; the polygons are cached in the
shared TransitLOSStudies/cache/urau folder.
"""
from __future__ import annotations

import glob
import json
import os
import urllib.request

import geopandas as gpd

URL = "https://gisco-services.ec.europa.eu/distribution/v2/urau/geojson/URAU_RG_100K_2024_4326.geojson"
CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "cache", "urau")


def build_metro_map(data_dir: str, out_json: str) -> dict:
    os.makedirs(CACHE, exist_ok=True)
    src = os.path.join(CACHE, "URAU_RG_100K_2024_4326.geojson")
    if not os.path.isfile(src):
        urllib.request.urlretrieve(URL, src)
    fua = gpd.read_file(src)
    fua = fua[(fua["CNTR_CODE"] == "ES") & (fua["URAU_CATG"] == "F")][["URAU_CODE", "URAU_NAME", "geometry"]].to_crs(4326)
    muni = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*", "municipality.geoparquet"))):
        g = gpd.read_parquet(f)
        idc = "GEOID" if "GEOID" in g.columns else "id"
        g = g[[idc, "geometry"]].rename(columns={idc: "id"}).to_crs(3035)
        g["geometry"] = g.geometry.representative_point()
        muni.append(g.to_crs(4326))
    muni = gpd.GeoDataFrame(muni[0].__class__(__import__("pandas").concat(muni, ignore_index=True)), crs=4326)
    j = gpd.sjoin(muni, fua, how="inner", predicate="within")
    out = {"areas": dict(zip(fua["URAU_CODE"], fua["URAU_NAME"])), "muni": dict(zip(j["id"].astype(str), j["URAU_CODE"]))}
    json.dump(out, open(out_json, "w"), ensure_ascii=False, separators=(",", ":"))
    return out


def build_metro_assets(data_dir: str, out_dir: str) -> dict:
    """Everything the map needs for the Eurostat metropolitan areas: `stats_chunks/metro_map.json`, one stats chunk per area (+ index entries)
    and `metro_lines.geojson` (the dotted boundary layer), written under `out_dir` (the map's folder)."""
    import sys
    from pathlib import Path

    sys.path[:0] = [str(Path(__file__).resolve().parents[3] / "transitLOS" / "src")]
    from transitlos.map.metro_areas import build_metro_chunks, metro_lines_geojson

    out = Path(out_dir)
    m = build_metro_map(data_dir, str(out / "stats_chunks" / "metro_map.json"))
    fua = gpd.read_file(os.path.join(CACHE, "URAU_RG_100K_2024_4326.geojson"))
    fua = fua[(fua["CNTR_CODE"] == "ES") & (fua["URAU_CATG"] == "F")][["URAU_CODE", "URAU_NAME", "geometry"]].rename(columns={"URAU_NAME": "name"})
    used = set(m["muni"].values())
    fua = fua[fua["URAU_CODE"].isin(used)]
    lines = metro_lines_geojson(fua, "name")
    (out / "metro_lines.geojson").write_text(json.dumps(lines, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    metros = build_metro_chunks(str(out / "stats_chunks"), m["areas"], m["muni"], "place")
    return {"areas": len(metros), "lines": len(lines["features"])}
