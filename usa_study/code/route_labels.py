"""Route NAMES for the scored stops, as the map's stop popups and route badges expect them.

The scorer keeps `route_ids` (GTFS route_ids joined with ';', possibly suffixed `_file_N` when feeds were merged). The map shows the
route name (`route_short_name`, else `route_long_name`, else the id -- the label of the route lines, so badge colours match) in
`all_routes` / `stop_routes` / `route_label`, joined with ','.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Union

import geopandas as gpd
import pandas as pd


def fix_station_names(stops: gpd.GeoDataFrame, gtfs_dirs: Sequence[Union[str, Path]]) -> gpd.GeoDataFrame:
    """Replace a station name that is just the parent_station id ("STN_A04", "place-coecl", "1:934") by the name in the feeds' stops.txt."""
    import re

    if stops is None or stops.empty or "parent_station" not in stops.columns or "stop_name" not in stops.columns:
        return stops
    names: Dict[str, str] = {}
    for d in gtfs_dirs:
        p = Path(d) / "stops.txt"
        if not p.is_file():
            continue
        try:
            t = pd.read_csv(p, dtype=str, keep_default_na=False, encoding_errors="replace", usecols=lambda c: c.strip().lstrip("\ufeff") in ("stop_id", "stop_name"))
        except Exception:  # noqa: BLE001
            continue
        t.columns = [c.strip().lstrip("\ufeff") for c in t.columns]
        for sid, nm in zip(t["stop_id"].str.strip(), t["stop_name"].str.strip()):
            if nm:
                names.setdefault(sid, nm)
    if not names:
        return stops
    out = stops.copy()
    same = out["stop_name"].astype(str) == out["parent_station"].astype(str)
    fixed = [names.get(re.sub(r"_file_\d+$", "", str(p).strip())) for p in out.loc[same, "parent_station"]]
    out.loc[same, "stop_name"] = [f if f else old for f, old in zip(fixed, out.loc[same, "stop_name"])]
    return out


def add_route_labels(stops: gpd.GeoDataFrame, gtfs_dirs: Sequence[Union[str, Path]]) -> gpd.GeoDataFrame:
    """Add the route-name columns to `stops` (no-op if already there)."""
    import re

    if stops is None or stops.empty or "route_ids" not in stops.columns or "all_routes" in stops.columns:
        return stops
    per_feed: Dict[str, Dict[str, str]] = {}
    union: Dict[str, str] = {}
    for d in gtfs_dirs:
        p = Path(d) / "routes.txt"
        if not p.is_file():
            continue
        try:
            r = pd.read_csv(p, dtype=str, keep_default_na=False, encoding_errors="replace")
        except Exception:  # noqa: BLE001
            continue
        r.columns = [c.strip().lstrip("﻿") for c in r.columns]
        if "route_id" not in r.columns:
            continue
        labels = {}
        for _, row in r.iterrows():
            rid = str(row["route_id"]).strip()
            name = (row.get("route_short_name", "") or "").strip() or (row.get("route_long_name", "") or "").strip() or rid
            labels[rid] = name
        per_feed[Path(d).name] = labels
        for rid, name in labels.items():
            union.setdefault(rid, name)

    def convert(ids, feed):
        own = per_feed.get(feed, {}) if feed else {}
        out = []
        for rid in str(ids or "").split(";"):
            rid = rid.strip()
            if not rid:
                continue
            raw = re.sub(r"_file_\d+$", "", rid)
            name = own.get(rid) or own.get(raw) or union.get(rid) or union.get(raw) or raw
            if name not in out:
                out.append(name)
        return ",".join(out)

    feeds = stops["source_feed"] if "source_feed" in stops.columns else [None] * len(stops)
    joined = [convert(i, f if isinstance(f, str) else None) for i, f in zip(stops["route_ids"], feeds)]
    stops = stops.copy()
    stops["all_routes"] = joined
    stops["stop_routes"] = joined
    stops["route_label"] = joined
    return fix_station_names(stops, gtfs_dirs)
