"""Daily vehicle-kilometres of a city's transit service inside its AOI (metro) and city core.

`vehicle_km = sum over trips running on the analysis day of  (runs of the trip) x (distance of the trip inside the area)`:
  * trips running that day: `calendar.txt` / `calendar_dates.txt` services active on the picked date (same date as the stop
    scoring, `picked_dates`);
  * runs: 1 for a normal trip; for a trip in `frequencies.txt` every headway window gives `(end - start) // headway` runs (a trip every
    5 minutes is 12 trips an hour);
  * distance inside the area: the length of the trip's `shapes.txt` line clipped to the area (metric UTM); a trip without a shape
    uses the straight lines between its stops.
Writes `<city>/results/vehicle_km.json`, read by the Place-rank tab of the map (population, vehicle-km, vehicle-km per person).
"""
from __future__ import annotations

import json
import pathlib
from datetime import date

import geopandas as gpd
import numpy as np


import pyproj
import shapely

def compute_vehicle_km(city_dir, aoi: gpd.GeoDataFrame, core: gpd.GeoDataFrame | None = None, dates: dict | None = None, extra: dict | None = None, stops_per_feed: bool = False,
                      day_type: str = "weekday") -> dict:
    from transitlos.vehicle_km import metric_crs_for, vehicle_km_by_area

    city_dir = pathlib.Path(city_dir)
    aoi4326 = aoi.to_crs(4326)
    centre = aoi4326.union_all().centroid
    to_metric = pyproj.Transformer.from_crs(4326, metric_crs_for(centre.x, centre.y), always_xy=True)
    names = ["metro"] + (["core"] if core is not None and len(core) else [])
    geoms = [aoi4326.union_all()] + ([core.to_crs(4326).union_all()] if "core" in names else [])
    for n_, g_ in (extra or {}).items():   # further analysis areas (official metro area, municipality)
        names.append(n_); geoms.append(g_.to_crs(4326).union_all())
    areas = gpd.GeoSeries([shapely.ops.transform(to_metric.transform, g) for g in geoms])
    dates = dict(dates or picked_dates(city_dir, aoi4326, stops_per_feed, day_type))
    # the date each feed was really scored on (a feed without service on the shared date is scored alone on its own representative
    # day: results/stops.parquet records it per `source_feed`) -- vehicle-km on that same day, never a different one
    try:
        import pandas as pd

        sp = city_dir / "results" / "stops.parquet"
        if sp.is_file():
            st = pd.read_parquet(sp, columns=["source_feed", "analysis_date"])
            for f_, d_ in st.dropna().drop_duplicates("source_feed").itertuples(index=False):
                if f_ in dates:
                    dates[f_] = date.fromisoformat(str(d_)[:10])
    except Exception:  # noqa: BLE001 -- older stops tables without the columns
        pass
    total = {k: {"vehicle_km": 0.0, "trip_runs": 0.0} for k in names}
    used = {}
    for d in sorted(p for p in (city_dir / "gtfs").iterdir() if p.is_dir() and (p / "stops.txt").is_file()):
        day = dates.get(d.name)
        if day is None:
            continue
        km, runs = vehicle_km_by_area(d, day, areas, to_metric)
        if runs.sum() > 0 or km.sum() > 0:
            used[d.name] = day.isoformat()
        for n, name in enumerate(names):
            total[name]["vehicle_km"] += float(km[n])
            total[name]["trip_runs"] += float(runs[n])
    out = {"city": city_dir.name, "dates": used, **{k: total[k] for k in names},
           "method": "daily vehicle-km on the analysis date inside the area: active trips x frequency runs x shape length clipped to the area"}
    for k in names:
        if k in out:
            out[k] = {a: round(b, 1) for a, b in out[k].items()}
    (city_dir / "results").mkdir(parents=True, exist_ok=True)
    (city_dir / "results" / "vehicle_km.json").write_text(json.dumps(out, indent=1))
    return out


def picked_dates(city_dir, aoi=None, stops_per_feed: bool = False, day_type: str = "weekday") -> dict[str, date]:
    """`{feed folder name: analysis date}` exactly as the stop scoring picks them (one shared date, or one per feed for `stops_per_feed`).

    Cached in `<city>/results/analysis_dates.json` (the pick is deterministic given the feeds, and loading the feeds is the slow part).
    """
    import json

    from pyGTFSHandler.feed import Feed
    from transitlos import stops as tstops

    city_dir = pathlib.Path(city_dir)
    cache = city_dir / "results" / ("analysis_dates.json" if day_type == "weekday" else f"analysis_dates_{day_type}.json")
    dirs = [p for p in sorted((city_dir / "gtfs").iterdir()) if p.is_dir() and (p / "stops.txt").is_file()]
    if cache.is_file():
        cached = json.loads(cache.read_text())
        if set(cached) == {d.name for d in dirs}:
            return {k: date.fromisoformat(v) for k, v in cached.items()}
    if aoi is None and (city_dir / "aoi.gpkg").is_file():
        aoi = gpd.read_file(city_dir / "aoi.gpkg").to_crs(4326)
    groups = [[d] for d in dirs] if stops_per_feed else [dirs]
    out: dict[str, date] = {}
    for group in groups:
        try:
            picked = tstops.pick_analysis_date(Feed(gtfs_dirs=[str(d) for d in group], aoi=aoi), day_type=day_type)
        except Exception:  # feed with nothing inside the AOI (the pipeline skips it too)
            continue
        for d in group:
            out[d.name] = picked
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({k: v.isoformat() for k, v in out.items()}))
    return out
