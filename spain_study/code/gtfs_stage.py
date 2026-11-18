# -*- coding: utf-8 -*-
"""Stage 1 (Spain): GTFS feeds from the NAP -> scored stops of one province (``stops.geoparquet``).

Same design as usa_study's ``gtfs_stage`` (copied pattern, nothing imported from it):
  1. every feed of the province is loaded together as one multi-dir ``pyGTFSHandler.Feed`` over a shared window, ALWAYS with
     ``aoi=province`` so cross-border / nationwide feeds (Renfe, Euskotren, ...) only keep stops inside this province;
  2. a feed without service on ``ANALYSIS_DATE`` is scored alone on its own representative weekday (fallback);
  3. best mode-tier / min-speed subset per station (``best_subset_scoring``), results unioned, same-location stops merged.
Modes: the NAP has no NTD-style mode table, so each route's mode comes from its GTFS ``route_type``
(``transitlos.scoring.mrc.mode_category``: rail / tram / bus; metro, funicular, cable car and ferry follow that function).
"""
from __future__ import annotations

import logging
from datetime import date, time
from pathlib import Path
from typing import List, Optional, Sequence

import geopandas as gpd
import pandas as pd
import polars as pl

from pyGTFSHandler.feed import Feed

logger = logging.getLogger(__name__)

#: Shared window of the combined load (two weeks starting Mon 2026-11-02, inside the validity of the Basque feeds) and the ONE scored day.
DEFAULT_WINDOW = (date(2026, 11, 2), date(2026, 11, 13))
ANALYSIS_DATE = date(2026, 11, 4)   # Wednesday
ANALYSIS_START_TIME = time(6, 0)
ANALYSIS_END_TIME = time(22, 0)
STOP_GROUP_DISTANCE_M = 150.0
_CANON = {"bus": 3, "tram": 0, "rail": 2}


def resolve_feed_modes(gtfs_dir: Path) -> pl.DataFrame:
    """``route_id, route_type, scoring_mode`` of one feed from its own GTFS route_type."""
    from transitlos.scoring.mrc import mode_category

    routes = pl.read_csv(Path(gtfs_dir) / "routes.txt", infer_schema_length=None).with_columns(pl.col("route_id").cast(pl.Utf8))
    if "route_type" not in routes.columns:
        routes = routes.with_columns(pl.lit(3).alias("route_type"))
    modes = []
    for rt in routes["route_type"].to_list():
        try:
            modes.append(mode_category(int(rt)))
        except (TypeError, ValueError):
            modes.append("bus")
    return routes.select(["route_id", "route_type"]).with_columns(pl.Series("scoring_mode", modes), pl.Series("canonical_route_type", [_CANON[m] for m in modes]))


def _route_modes_for(dirs: Sequence[Path]) -> pl.DataFrame:
    frames = []
    for file_id, d in enumerate(dirs):   # file_id = position in the Feed's gtfs_dirs list
        try:
            frames.append(resolve_feed_modes(d).with_columns(pl.lit(file_id, dtype=pl.Int64).alias("file_id")))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Feed '{d}': could not resolve modes: {exc}")
    if not frames:
        return pl.DataFrame(schema={"route_id": pl.Utf8, "scoring_mode": pl.Utf8})
    return pl.concat(frames, how="diagonal_relaxed")


def split_feeds_by_service(dirs: Sequence[Path], aoi, stop_group_distance: float = STOP_GROUP_DISTANCE_M):
    """``(in_window, fallback)``: feeds with / without any trip on ``ANALYSIS_DATE`` inside the AOI."""
    ok, fb = [], []
    for d in dirs:
        try:
            feed = Feed(gtfs_dirs=[str(d)], aoi=aoi, start_date=ANALYSIS_DATE, end_date=ANALYSIS_DATE, stop_group_distance=stop_group_distance)
            has = feed.trips.lf.limit(1).collect().height > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Feed '{d}': service check failed, treating as fallback: {exc}")
            has = False
        if has:   # a holiday or a normal day, never a mix: a reduced / holiday service on the shared date -> the feed's own typical day
            from transitlos.day_check import day_volume_ratio

            try:
                r = day_volume_ratio(d, ANALYSIS_DATE)
            except Exception:  # noqa: BLE001
                r = None
            if r is not None and not (0.8 <= r <= 1.25):
                logger.warning(f"Feed '{d}': {ANALYSIS_DATE} runs {r:.0%} of its usual trips for that weekday -- scored on its own typical day")
                has = False
        (ok if has else fb).append(d)
    return ok, fb


def merge_same_location_stops(stops: gpd.GeoDataFrame, decimals: int = 6) -> gpd.GeoDataFrame:
    """One row per location: best ``stop_score`` kept, route lists joined."""
    if stops.empty:
        return stops
    key = stops["stop_lat"].round(decimals).astype(str) + "," + stops["stop_lon"].round(decimals).astype(str)
    ranked = stops.assign(_loc=key).sort_values("stop_score", ascending=False, na_position="last", kind="stable")
    if "route_ids" in ranked.columns:
        joined = ranked.groupby("_loc", sort=False)["route_ids"].agg(lambda col: ";".join(dict.fromkeys(r for v in col.dropna() for r in str(v).split(";") if r)))
    best = ranked.drop_duplicates("_loc", keep="first").copy()
    if "route_ids" in best.columns:
        best["route_ids"] = best["_loc"].map(joined)
    merged = best.drop(columns="_loc").sort_index().reset_index(drop=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=stops.crs)


def score_province_stops(gtfs_dirs: Sequence[Path], aoi: gpd.GeoDataFrame, window: tuple = DEFAULT_WINDOW,
                         stop_group_distance: float = STOP_GROUP_DISTANCE_M, stop_table_sink: Optional[list] = None) -> gpd.GeoDataFrame:
    """Scored stops of one province (see module docstring). ``aoi``: province polygon, EPSG:4326."""
    from .best_subset_scoring import best_subset_stop_scores

    dirs = [Path(d) for d in gtfs_dirs]
    in_window, fallback = split_feeds_by_service(dirs, aoi, stop_group_distance)
    frames = []

    def _score(group):
        feed = Feed(gtfs_dirs=[str(d) for d in group], aoi=aoi, start_date=window[0], end_date=window[1], stop_group_distance=stop_group_distance)
        scored = best_subset_stop_scores(feed, ANALYSIS_DATE, _route_modes_for(group), ANALYSIS_START_TIME, ANALYSIS_END_TIME)
        if scored.empty:
            return None
        scored = feed.add_stop_coords(scored)
        if stop_table_sink is not None:
            from transitlos.stop_table import stop_table_from_feed

            stop_table_sink.append(stop_table_from_feed(feed, scored))
        return scored

    if in_window:
        try:
            r = _score(in_window)
            if r is not None:
                frames.append(r)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Combined load failed ({exc}); retrying feeds one by one.")
            for d in in_window:
                try:
                    r = _score([d])
                except Exception as exc2:  # noqa: BLE001
                    logger.warning(f"Feed '{d}': skipped: {exc2}")
                    continue
                if r is not None:
                    frames.append(r)
    for d in fallback:
        try:
            feed = Feed(gtfs_dirs=[str(d)], aoi=aoi, stop_group_distance=stop_group_distance)
            rep = feed.get_representative_date(date_type="weekday")
            if rep is None:
                continue
            sc = best_subset_stop_scores(feed, rep, _route_modes_for([d]), ANALYSIS_START_TIME, ANALYSIS_END_TIME)
            if sc.empty:
                continue
            sc = feed.add_stop_coords(sc)
            sc["source_feed"] = str(d)
            sc["representative_date"] = str(rep)
            if stop_table_sink is not None:
                from transitlos.stop_table import stop_table_from_feed

                t = stop_table_from_feed(feed, sc)
                t["source_feed"] = str(d)
                t["representative_date"] = str(rep)
                stop_table_sink.append(t)
            frames.append(sc)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Feed '{d}': fallback scoring failed: {exc}")
    if not frames:
        return gpd.GeoDataFrame(columns=["parent_station", "geometry"], geometry="geometry", crs=4326)
    allst = pl.concat([pl.from_pandas(f) if isinstance(f, pd.DataFrame) else f for f in frames], how="diagonal_relaxed").to_pandas()
    geom = gpd.points_from_xy(allst["stop_lon"], allst["stop_lat"])
    return merge_same_location_stops(gpd.GeoDataFrame(allst, geometry=geom, crs=4326))
