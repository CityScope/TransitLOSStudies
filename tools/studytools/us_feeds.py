"""US feeds shared by the US studies (usa_study, usa_flex)."""

from __future__ import annotations

from pathlib import Path
from typing import List


def neighbour_feed_dirs(state_dir: Path, min_stops: int = 1) -> List[Path]:
    """Feed folders of the NEIGHBOURING states that have stops inside this state.

    Each state's feeds are clipped to its own boundary, and an agency's feed is usually downloaded only by the state NTD lists it under, so
    service across a state line was missing on the other side (2026-11 audit: MBTA commuter rail in Rhode Island, SEPTA in Delaware, NJ
    Transit / Metro-North stations in Rockland and Orange NY, St. Louis MetroLink and MetroBus in Illinois, Metrorail in Maryland and
    Virginia). These folders are loaded with the state's own feeds (and clipped to the state); a copy of a feed the state already has is
    removed by the duplicate check. Cached in `<state>/neighbour_feeds.json` (recomputed when a neighbour's feed list changes)."""
    import json

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely import contains_xy

    state_dir = Path(state_dir); data_dir = state_dir.parent
    me = gpd.read_parquet(state_dir / "state.geoparquet").to_crs(4326).geometry.union_all()
    near = []
    for other in sorted(data_dir.iterdir()):
        if other == state_dir or not (other / "state.geoparquet").is_file() or not (other / "gtfs").is_dir():
            continue
        og = gpd.read_parquet(other / "state.geoparquet").to_crs(4326).geometry.union_all()
        if og.distance(me) < 0.05:   # degrees (~5 km): states that touch (water borders included)
            near.append(other)
    folders = [f for o in near for f in sorted((o / "gtfs").iterdir()) if f.is_dir() and not f.name.startswith("_") and (f / "stops.txt").is_file()]
    key = {str(f): int((f / "stops.txt").stat().st_mtime) for f in folders}
    cache = state_dir / "neighbour_feeds.json"
    if cache.is_file():
        c = json.loads(cache.read_text())
        if c.get("key") == key:
            return [Path(p) for p in c["inside"]]
    minx, miny, maxx, maxy = me.bounds
    inside = []
    for f in folders:
        try:
            st = pd.read_csv(f / "stops.txt", dtype=str, keep_default_na=False, encoding_errors="replace", usecols=lambda c: c.strip().lstrip("﻿") in ("stop_lat", "stop_lon"))
        except Exception:  # noqa: BLE001
            continue
        st.columns = [c.strip().lstrip("﻿") for c in st.columns]
        la = pd.to_numeric(st.get("stop_lat"), errors="coerce").to_numpy(); lo = pd.to_numeric(st.get("stop_lon"), errors="coerce").to_numpy()
        ok = np.isfinite(la) & np.isfinite(lo) & (lo >= minx) & (lo <= maxx) & (la >= miny) & (la <= maxy)
        if ok.sum() >= min_stops and contains_xy(me, lo[ok], la[ok]).sum() >= min_stops:
            inside.append(str(f))
    cache.write_text(json.dumps({"key": key, "inside": inside}, indent=1))
    return [Path(p) for p in inside]
