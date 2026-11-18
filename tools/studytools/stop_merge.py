"""Merge rows of the same physical stop scored more than once (several feeds or days): shared by the city and US studies."""

from __future__ import annotations

import logging

import geopandas as gpd

logger = logging.getLogger(__name__)


def merge_same_location_stops(stops: gpd.GeoDataFrame, decimals: int = 6) -> gpd.GeoDataFrame:
    """One row per location: of all rows at the same coordinates keep the best `stop_score`, with the route lists joined.

    Feeds scored separately (each day / each fallback feed gets its own `Feed` object) can produce the same physical stop
    several times; the combined table must not carry the same stop more than once. Rows at DIFFERENT locations are never
    merged here. `route_ids` of the kept row becomes the union (in order of appearance, de-duplicated) of the group's.
    """
    if stops.empty:
        return stops
    key = stops["stop_lat"].round(decimals).astype(str) + "," + stops["stop_lon"].round(decimals).astype(str)
    ranked = stops.assign(_loc=key).sort_values("stop_score", ascending=False, na_position="last", kind="stable")
    if "route_ids" in ranked.columns:
        joined = ranked.groupby("_loc", sort=False)["route_ids"].agg(
            lambda col: ";".join(dict.fromkeys(r for v in col.dropna() for r in str(v).split(";") if r))
        )
    best = ranked.drop_duplicates("_loc", keep="first").copy()
    if "route_ids" in best.columns:
        best["route_ids"] = best["_loc"].map(joined)
    merged = best.drop(columns="_loc").sort_index().reset_index(drop=True)
    logger.info(f"Merged same-location stops: {len(stops):,} -> {len(merged):,} rows.")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=stops.crs)
