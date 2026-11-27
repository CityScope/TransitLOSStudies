"""Stop scoring of every area = the same scoring as the US study (`best_subset_scoring`, `stop_merge`):

  * `best_subset_stop_scores`: each station gets its BEST achievable score over mode tiers (bus+tram+rail / tram+rail / rail only)
    crossed with minimum-speed cutoffs, headways re-derived per subset (never a naive recombination), 06:00-22:00 window;
  * strict 150 m station grouping (`GTFS_STOP_GROUPING=complete`: no two stops of a group are farther apart than that);
  * feeds without service on the analysis date (expired / not started) are NOT combined with the others: each is scored alone on its
    own representative date and every group's stops are concatenated;
  * `merge_same_location_stops`: one row per location, best score kept, route lists joined.

Only the pieces that are specific to the US study are replaced: the route mode (bus/tram/rail) comes from each route's GTFS `route_type`
(no NTD), and the analysis date from `transitlos.stops.pick_analysis_date` (automatic, per feed or shared) or an explicit pin.
"""
from __future__ import annotations

import logging
import os
from datetime import date, time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import geopandas as gpd
import pandas as pd
import polars as pl

os.environ["GTFS_STOP_GROUPING"] = "complete"  # read by pyGTFSHandler at Feed construction

logger = logging.getLogger(__name__)
STOP_GROUP_DISTANCE_M = 150.0
_CANONICAL_ROUTE_TYPE = {"bus": 3, "tram": 0, "rail": 2}


def _route_modes(dirs: Sequence[Path]) -> pl.DataFrame:
    """`route_id, scoring_mode` ('bus' / 'tram' / 'rail') from each route's GTFS route_type."""
    from transitlos.scoring.mrc import mode_category

    frames = []
    for file_id, d in enumerate(dirs):   # file_id = position in the Feed's gtfs_dirs list
        p = d / "routes.txt"
        if not p.is_file():
            continue
        try:
            routes = pl.read_csv(p, infer_schema_length=0, truncate_ragged_lines=True)
            routes = routes.rename({c: c.strip().lstrip("﻿") for c in routes.columns})
            if "route_type" in routes.columns:
                modes = [mode_category(int(v) if v not in (None, "") and str(v).strip().lstrip("-").isdigit() else None)
                         for v in routes["route_type"].to_list()]
            else:
                modes = ["bus"] * len(routes)
            frames.append(pl.DataFrame({"route_id": routes["route_id"].cast(pl.Utf8), "scoring_mode": modes, "file_id": [file_id] * len(routes)}))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{d}: could not read route modes ({exc}); treating its routes as bus")
    if not frames:
        return pl.DataFrame(schema={"route_id": pl.Utf8, "scoring_mode": pl.Utf8})
    return pl.concat(frames, how="diagonal_relaxed").unique(["route_id", "file_id"], keep="first")


_HOLIDAYS: dict = {}   # (country, region, year) -> {iso date: name}, looked up once per area


def _holiday_name(day: date, aoi) -> Optional[str]:
    """The public holiday the date falls on in the area's country and region (None when it is none, or the holiday service is unreachable)."""
    try:
        from datetime import timedelta

        from pyGTFSHandler.utils import date_parsing, geocoding

        c = aoi.to_crs(4326).union_all().centroid
        key = (round(c.y, 1), round(c.x, 1), day.year)
        if key not in _HOLIDAYS:
            country, region = geocoding.get_country_region(c.y, c.x)
            _HOLIDAYS[key] = {str(date(1970, 1, 1) + timedelta(days=int(n))): name
                              for n, name in date_parsing.get_holidays(day.year, country, region).select("date", "name").iter_rows()}
        return _HOLIDAYS[key].get(day.isoformat())
    except Exception:  # noqa: BLE001
        return None


def day_label(day: date, aoi=None) -> str:
    """`Tuesday 2026-12-01` plus `, public holiday: <name>` when it is one."""
    h = _holiday_name(day, aoi) if aoi is not None else None
    return f"{day:%A} {day.isoformat()}" + (f", public holiday: {h}" if h else "")


def _service_check(d: Path, day: date, aoi, group_m: float, strict: bool = False) -> tuple:
    """`(fits, why not)`: whether the feed can be scored on `day`, and if not what is wrong with the date for it."""
    from pyGTFSHandler.feed import Feed

    none = "it has no service that day (outside its calendar, or no trips that day of the week)"
    try:
        feed = Feed(gtfs_dirs=[str(d)], aoi=aoi, start_date=day, end_date=day, stop_group_distance=group_m)
        if feed.trips.lf.limit(1).collect().height == 0:
            return False, none
    except Exception as exc:  # noqa: BLE001 -- the feed library raises when a date has no trips at all
        if "no trips found" in str(exc).lower():
            return False, none
        raise
    if strict:   # the date was chosen by the user: used as it is
        return True, ""
    from transitlos.day_check import day_volume_ratio

    try:
        r = day_volume_ratio(d, day)
    except Exception:  # noqa: BLE001
        r = None
    if r is not None and not (0.8 <= r <= 1.25):
        return False, f"it runs {r:.0%} of its usual trips for that day of the week"
    return True, ""


def _has_service_on(d: Path, day: date, aoi, group_m: float, strict: bool = False) -> bool:
    from pyGTFSHandler.feed import Feed

    feed = Feed(gtfs_dirs=[str(d)], aoi=aoi, start_date=day, end_date=day, stop_group_distance=group_m)
    if feed.trips.lf.limit(1).collect().height == 0:
        return False
    if strict:   # the date was chosen by the user: used as it is
        return True
    # a holiday or a normal day, never a mix: reduced / holiday service on the shared date -> scored alone on the feed's own typical day
    from transitlos.day_check import day_volume_ratio

    try:
        r = day_volume_ratio(d, day)
    except Exception:  # noqa: BLE001
        r = None
    if r is not None and not (0.8 <= r <= 1.25):
        logger.warning(f"feed {d.name}: {day} runs {r:.0%} of its usual trips for that weekday -- scored on its own typical day")
        return False
    return True


def score_city_stops(
    gtfs_dirs: Sequence[Union[str, Path]],
    aoi: gpd.GeoDataFrame,
    per_feed: bool = False,
    date_range: Optional[Union[Tuple, Dict[str, Tuple]]] = None,
    start_time: time = time(6, 0),
    end_time: time = time(22, 0),
    stop_group_distance: float = STOP_GROUP_DISTANCE_M,
    stop_table_sink: Optional[list] = None,
    day_type: str = "weekday",
    strict_date: bool = False,
) -> gpd.GeoDataFrame:
    """Scored stops of one city (the `stops.parquet` content): one row per station with `stop_score`, sub-scores, `mode_category`.
    `day_type`: the analysis day is a working "weekday", a "saturday" or a "sunday". `strict_date`: the pinned `date_range` date is used
    as it is: no swap to a feed's own typical day when its volume that day differs (a date chosen by the user)."""
    from pyGTFSHandler.feed import Feed
    from transitlos.stops import pick_analysis_date as _pick

    def pick_analysis_date(feed):
        return _pick(feed, day_type=day_type)

    from .best_subset_scoring import best_subset_stop_scores
    from .stop_merge import merge_same_location_stops
    dirs = [Path(d) for d in gtfs_dirs if (Path(d) / "stops.txt").is_file()]

    def pinned(d: Path) -> Optional[date]:
        r = date_range.get(d.name) if isinstance(date_range, dict) else date_range
        return None if r is None else (r[0].date() if hasattr(r[0], "date") else r[0])

    def score(group: Sequence[Path], day: date, feed=None):
        feed = feed or Feed(gtfs_dirs=[str(d) for d in group], aoi=aoi, start_date=day, end_date=day, stop_group_distance=stop_group_distance)
        scored = best_subset_stop_scores(feed, day, _route_modes(group), start_time, end_time)
        if scored.empty:
            return None
        scored = feed.add_stop_coords(scored)
        scored["analysis_date"] = day.isoformat()
        if stop_table_sink is not None:   # every real stop_id at its own location, with its own routes / mode (the map draws these)
            from transitlos.stop_table import stop_table_from_feed

            tbl = stop_table_from_feed(feed, scored)
            tbl["analysis_date"] = day.isoformat()
            stop_table_sink.append(tbl)
        return scored

    frames = []

    def alone(d: Path, requested: Optional[date] = None, reason: str = ""):
        try:
            day = pinned(d)
            requested = requested or day
            feed = Feed(gtfs_dirs=[str(d)], aoi=aoi, stop_group_distance=stop_group_distance)
            if day is None:
                day = pick_analysis_date(feed)
            else:
                from transitlos.day_check import day_volume_ratio

                try:
                    _r = day_volume_ratio(d, day)
                except Exception:  # noqa: BLE001
                    _r = None
                if _r is not None and not (0.8 <= _r <= 1.25) and not strict_date:
                    day = pick_analysis_date(feed)
            try:
                single = Feed(gtfs_dirs=[str(d)], aoi=aoi, start_date=day, end_date=day, stop_group_distance=stop_group_distance)
                out = score([d], day, single)
            except Exception as exc_day:  # noqa: BLE001 -- a pinned / shared date outside this feed's calendar (expired feed): its own date
                own = pick_analysis_date(feed)
                if own == day:
                    raise
                logger.warning(f"feed {d.name}: no service on {day} ({str(exc_day)[:80]}); scored on its own date {own}")
                print(f"[stop-scoring] feed {d.name}: no service on {day}; scored on its own date {own}", flush=True)
                day = own
                single = Feed(gtfs_dirs=[str(d)], aoi=aoi, start_date=day, end_date=day, stop_group_distance=stop_group_distance)
                out = score([d], day, single)
        except Exception as exc:  # noqa: BLE001 -- e.g. nothing of this feed inside the AOI
            logger.warning(f"feed {d.name}: skipped ({type(exc).__name__}: {str(exc)[:100]})")
            print(f"[stop-scoring] feed {d.name}: skipped ({type(exc).__name__}: {str(exc)[:100]})", flush=True)
            return
        if requested is not None and day != requested:   # the date asked for (chosen or shared) did not fit this feed: say which date it got
            print(f"[stop-scoring] WARNING feed {d.name}: the date {day_label(requested, aoi)} does not fit it ({reason or 'no service or an unusual day for this feed'}); "
                  f"scored on its own typical day {day_label(day, aoi)}", flush=True)
        if out is not None:
            out["source_feed"] = d.name
            frames.append(out)

    if per_feed:
        for d in dirs:
            alone(d)
    else:
        shared = next((pinned(d) for d in dirs if pinned(d) is not None), None)
        if shared is None:
            shared = pick_analysis_date(Feed(gtfs_dirs=[str(d) for d in dirs], aoi=aoi, stop_group_distance=stop_group_distance))
        in_window, expired, reasons = [], [], {}
        for d in dirs:
            try:
                ok, why = _service_check(d, shared, aoi, stop_group_distance, strict_date)
                (in_window if ok else expired).append(d)
                reasons[d.name] = why
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"feed {d.name}: could not check service on {shared}: {exc}")
                expired.append(d)
                reasons[d.name] = f"its service on that date could not be checked ({str(exc)[:60]})"
        print(f"[stop-scoring] shared analysis date {shared}: {len(in_window)} feeds with service, {len(expired)} scored on their own dates", flush=True)
        if in_window:
            try:
                out = score(in_window, shared)
                if out is not None:
                    frames.append(out)
            except Exception as exc:  # noqa: BLE001 -- one bad feed must not discard the others
                logger.warning(f"combined load failed ({exc}); scoring the feeds one by one")
                for d in in_window:
                    try:
                        out = score([d], shared)
                    except Exception as exc2:  # noqa: BLE001
                        logger.warning(f"feed {d.name}: skipped ({exc2})")
                        continue
                    if out is not None:
                        out["source_feed"] = d.name
                        frames.append(out)
        for d in expired:
            alone(d, requested=shared, reason=reasons.get(d.name, ""))

    if not frames:
        raise ValueError(f"score_city_stops: no feed produced any scored stop inside the AOI ({[d.name for d in dirs][:5]}...)")
    all_stops = pl.concat([pl.from_pandas(f) if isinstance(f, pd.DataFrame) else f for f in frames], how="diagonal_relaxed").to_pandas()
    gdf = gpd.GeoDataFrame(all_stops, geometry=gpd.points_from_xy(all_stops["stop_lon"], all_stops["stop_lat"]), crs=4326)
    gdf = merge_same_location_stops(gdf)
    gdf["mode_category"] = gdf["scoring_mode"]
    gdf["route_type"] = gdf["scoring_mode"].map(_CANONICAL_ROUTE_TYPE)
    return gdf



def add_route_labels(stops: gpd.GeoDataFrame, gtfs_dirs: Sequence[Union[str, Path]]) -> gpd.GeoDataFrame:
    """Route names for the stop popups / route badges (`route_labels`, shared with the US study)."""
    from .route_labels import add_route_labels as _add

    return _add(stops, gtfs_dirs)


def build_stop_table_file(city_dir: Path, config, force: bool = False) -> gpd.GeoDataFrame:
    """`<city>/results/stops_by_id.parquet`: every real stop_id at its own location with ITS routes / mode + its station's scores.

    The map draws this table (the scored stations of `stops.parquet` are one row per parent station). Re-scores like the pipeline did
    (same Feed groups, dates from `results/analysis_dates.json`) without touching the stored scores or the street level of service.
    """
    import json
    from datetime import date

    out = Path(city_dir) / "results" / "stops_by_id.parquet"
    if out.is_file() and not force:
        return gpd.read_parquet(out)
    aoi = gpd.read_file(Path(city_dir) / "aoi.gpkg")
    dates_file = Path(city_dir) / "results" / "analysis_dates.json"
    date_range = None
    if dates_file.is_file():
        date_range = {k: (date.fromisoformat(v), date.fromisoformat(v)) for k, v in json.loads(dates_file.read_text()).items()}
    sink: list = []
    score_city_stops(sorted(str(p) for p in (Path(city_dir) / "gtfs").iterdir() if p.is_dir()), aoi, per_feed=config.stops_per_feed,
                     date_range=date_range, stop_table_sink=sink)
    table = pd.concat(sink, ignore_index=True) if sink else pd.DataFrame()
    if table.empty:
        return gpd.GeoDataFrame(table)
    gdf = gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(table["stop_lon"], table["stop_lat"]), crs=4326)
    gdf.to_parquet(out)
    return gdf
