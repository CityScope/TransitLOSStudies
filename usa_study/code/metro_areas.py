"""US metropolitan areas (Census CBSAs, metropolitan only): members = counties; stats chunks, index entries and dotted boundary lines for the map."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CBSA_FILE = ROOT / "TransitLOSStudies" / "cache" / "census_cbsa" / "list1.xlsx"


def cbsa_members() -> tuple[dict, dict]:
    """``({cbsa code: title}, {county fips: cbsa code})`` of the metropolitan statistical areas (micropolitan areas are left out)."""
    d = pd.read_excel(CBSA_FILE, header=2, dtype=str).dropna(subset=["CBSA Code", "FIPS State Code", "FIPS County Code"])
    d = d[d["Metropolitan/Micropolitan Statistical Area"].str.startswith("Metro")]
    d["fips"] = d["FIPS State Code"] + d["FIPS County Code"]
    return dict(zip(d["CBSA Code"], d["CBSA Title"])), dict(zip(d["fips"], d["CBSA Code"]))


def build_metro_assets(data_dir: str, out_dir: str) -> dict:
    """`stats_chunks/metro_<code>.json.gz` + index entries and `metro_lines.geojson` under `out_dir` (the nationwide map's folder)."""
    import sys

    sys.path[:0] = [str(ROOT / "transitLOS" / "src")]
    from transitlos.map.metro_areas import build_metro_chunks, metro_lines_geojson

    areas, members = cbsa_members()
    out = Path(out_dir)
    counties = [gpd.read_parquet(p)[["GEOID", "geometry"]].to_crs(4326) for p in sorted(Path(data_dir).glob("*/county.geoparquet"))]
    c = pd.concat(counties, ignore_index=True)
    c["cbsa"] = c["GEOID"].map(members)
    polys = gpd.GeoDataFrame(c.dropna(subset=["cbsa"]).dissolve(by="cbsa").reset_index(), geometry="geometry", crs=4326)
    polys["name"] = polys["cbsa"].map(areas)
    lines = metro_lines_geojson(polys, "name", tolerance_deg=0.004)
    (out / "metro_lines.geojson").write_text(json.dumps(lines, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    metros = build_metro_chunks(str(out / "stats_chunks"), areas, members, "county")
    return {"areas": len(metros), "lines": len(lines["features"])}
