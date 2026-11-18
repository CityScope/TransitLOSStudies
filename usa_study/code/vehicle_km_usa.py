"""Daily vehicle-kilometres of a state's transit service inside every county and every place (`<state>/vehicle_km.parquet`).

Uses the shared `transitlos.vehicle_km` calculation: trips running on the analysis day x frequency runs x the length of each trip's shape
clipped to the area (stop-to-stop lines when a trip has no shape). Every downloaded feed of the state is used on the same date its stops were
scored on: `ANALYSIS_DATE` for feeds with service that day, the feed's own representative date for the expired ones (recorded in
`stops.geoparquet` as `source_feed` / `representative_date`).
Rows: `level` ("county" / "place"), `GEOID`, `vehicle_km`, `trip_runs`. A state's total is the sum of its counties.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyproj
import shapely

from .gtfs_stage import ANALYSIS_DATE

logger = logging.getLogger(__name__)
VEHICLE_KM_FILE = "vehicle_km.parquet"


def compute_state_vehicle_km(state_dir: Path, analysis_date: date = ANALYSIS_DATE) -> pd.DataFrame:
    from transitlos.vehicle_km import metric_crs_for, vehicle_km_by_area

    state_dir = Path(state_dir)
    parts = []
    for level, file in (("county", "county.geoparquet"), ("place", "place.geoparquet")):
        p = state_dir / file
        if p.is_file():
            g = gpd.read_parquet(p)[["GEOID", "geometry"]].to_crs(4326)
            g["level"] = level
            parts.append(g)
    if not parts:
        raise ValueError(f"{state_dir}: no county / place boundaries")
    areas = pd.concat(parts, ignore_index=True)
    centre = shapely.union_all(areas.geometry.values).centroid
    to_metric = pyproj.Transformer.from_crs(4326, metric_crs_for(centre.x, centre.y), always_xy=True)
    metric = gpd.GeoSeries([shapely.ops.transform(to_metric.transform, g) for g in areas.geometry])

    fallback = {}
    stops = state_dir / "stops.geoparquet"
    if stops.is_file():
        s = pd.read_parquet(stops)
        if {"source_feed", "representative_date"} <= set(s.columns):
            for feed, d in s[["source_feed", "representative_date"]].dropna().drop_duplicates().itertuples(index=False):
                fallback[Path(str(feed)).name] = date.fromisoformat(str(d)[:10])
    km = pd.Series(0.0, index=areas.index)
    runs = pd.Series(0.0, index=areas.index)
    from pyGTFSHandler.utils.feed_dedupe import drop_duplicate_feeds

    feed_dirs = drop_duplicate_feeds(sorted(p for p in (state_dir / "gtfs").iterdir() if p.is_dir() and (p / "stops.txt").is_file()))
    for d in feed_dirs:
        day = fallback.get(d.name, analysis_date)
        try:
            k, r = vehicle_km_by_area(d, day, metric, to_metric)
        except Exception as exc:  # noqa: BLE001 -- one malformed feed must not discard the others
            logger.warning(f"{state_dir.name}/{d.name}: vehicle-km skipped ({type(exc).__name__}: {str(exc)[:80]})")
            continue
        km += k
        runs += r
    out = pd.DataFrame({"level": areas["level"], "GEOID": areas["GEOID"], "vehicle_km": km.round(1), "trip_runs": runs.round(1)})
    out.to_parquet(state_dir / VEHICLE_KM_FILE)
    return out
