# -*- coding: utf-8 -*-
"""Stage 1: per-state GTFS load + stop scoring -> `stops.geoparquet`.

Design confirmed with Miguel (2026-09-29), see
`~/.claude/.../memory/usa_study_pipeline_progress.md`:

1. Load every downloaded feed for a state together, as one multi-dir
   `pyGTFSHandler.Feed`, windowed to a shared date range (Oct 1-15 2026 by
   default) -- **always** with `aoi=state_aoi`, so a cross-border agency
   (NY/NJ/CT, DC-area, ...) only keeps the stops/routes actually inside
   this state.
2. Any individual feed with zero trips in that shared window is pulled out
   onto a separate "fallback" list and loaded/scored on its OWN, since by
   definition the shared window can't score it.
3. Each fallback feed loads as its own single-feed `Feed` (same
   `stop_group_distance`, same `aoi=state_aoi`), then
   `feed.get_representative_date(date_type="weekday")` picks a day from
   its own full service range.
4. Stop score is computed for both the combined graph and every fallback
   feed, and the results are unioned into one `stops.geoparquet` per state.

Mode resolution (NTD mode_name default, GTFS route_type fallback for
feeds where NTD's single-mode-per-URL claim is contradicted, manual
review for ambiguous BRT cases) follows
`~/.claude/.../memory/usa_study_ntd_mode_resolution.md`; see
`resolve_stop_modes` below for exactly how that's wired into scoring.
"""

from __future__ import annotations

import logging
from datetime import date, time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import geopandas as gpd
import pandas as pd
import polars as pl

from pyGTFSHandler.feed import Feed
from pyGTFSHandler.downloaders.usa.ntd import NTD_MODE_TO_SCORING_MODE
from pyGTFSHandler.downloaders.usa.brt import BRT_ROUTE_TYPE

logger = logging.getLogger(__name__)

#: Shared analysis window for every state's combined GTFS load (see module
#: docstring). A feed with zero trips inside this window becomes a
#: fallback feed instead.
DEFAULT_WINDOW = (date(2026, 10, 1), date(2026, 10, 15))

#: The ONE day every in-window feed is scored on (a mid-window Wednesday: 1,063 of 1,225 feeds run on it, vs 1,049 on
#: Oct 1). A feed without service on this day -- even if it runs on another day of the window, like Bi-State, whose
#: calendar starts Oct 2 -- cannot be scored with the others (no trips = no headway = its stops silently vanished); it is
#: scored alone on its own typical weekday instead.
ANALYSIS_DATE = date(2026, 10, 7)

#: Daily analysis window for headway/speed (study-wide, same as `city_science_network`'s
#: `analysis_start_time`/`analysis_end_time`). It MUST NOT be the full day: headway is
#: sum(gap^2)/window-length with the overnight gap added as one gap, so a 24 h window
#: turns every route's night break into 30-90 min of 'headway' (BART Embarcadero 5 min -> 30 min).
ANALYSIS_START_TIME = time(6, 0)
ANALYSIS_END_TIME = time(22, 0)

#: Stops without a parent station are grouped when within this distance, and NO two stops of a group are farther apart
#: than this (complete linkage: `GTFS_STOP_GROUPING=complete`, set in main.py). The library's default grouping chains
#: (neighbour-of-neighbour), which merged whole downtown streets into one "station" (CTA Loop: 395 stops -> 35
#: stations, clusters up to 2.9 km), each scored from a single mean location.
STOP_GROUP_DISTANCE_M = 150.0

#: `transitlos.scoring.mrc.mode_category`'s own {rail, tram} route_type
#: sets (everything else, including our BRT marker `702`, falls to "bus"
#: -- see that module). This maps OUR resolved bus/tram/rail category back
#: onto a representative literal `route_type` value that `mode_category`
#: is guaranteed to classify the same way, since `compute_stop_scores`
#: derives mode from a stop's `route_type` column, not from any NTD
#: field. Without this adapter, a BRT route (marked `702` by
#: `pyGTFSHandler.downloaders.usa.brt`) would silently score as plain bus
#: instead of tram (this project's agreed BRT->tram mapping, see
#: `NTD_MODE_TO_SCORING_MODE`), since `mode_category(702)` isn't in either
#: of `mrc.py`'s recognized sets.
_SCORING_MODE_TO_CANONICAL_ROUTE_TYPE = {"bus": 3, "tram": 0, "rail": 2}


def resolve_feed_modes(gtfs_dir: Path, ntd_rows: List[Dict[str, object]]) -> pl.DataFrame:
    """Resolve each route's scoring mode (bus/tram/rail) for one downloaded feed.

    Implements the rule agreed 2026-09-29 (see module docstring):

    1. Default: every route in the feed gets NTD's own `mode_name` (mapped
       via `NTD_MODE_TO_SCORING_MODE`) -- correct whenever NTD's
       single-mode-per-feed-URL claim actually holds.
    2. If the feed's real GTFS `route_type` is NOT uniform across its
       routes (a contradiction with NTD listing one mode for this whole
       feed URL), fall back to mapping each route's OWN `route_type` via
       `transitlos.scoring.mrc.mode_category` instead, logged as a
       flagged contradiction.
    3. Exception, still following the agreed rule: a route already marked
       `BRT_ROUTE_TYPE` (702) by `pyGTFSHandler.downloaders.usa.brt` is
       always "tram" regardless of case 1/2, since that marker only ever
       gets written after a specific, either-automatic-safe or
       human-verified BRT determination -- it should never be
       second-guessed by the generic route_type fallback.

    Args:
        gtfs_dir: Path to the extracted GTFS feed directory (its
            `routes.txt` is read for real `route_type` values).
        ntd_rows: Every NTD row whose `download_url` points at this feed
            (i.e. every `mode_name` NTD associates with it).

    Returns:
        Polars DataFrame with `route_id`, `route_type` (real GTFS value),
        `scoring_mode` (`"bus"`/`"tram"`/`"rail"`), and
        `canonical_route_type` (the representative value to feed
        `compute_stop_scores` so `mode_category` reproduces
        `scoring_mode` exactly -- see
        `_SCORING_MODE_TO_CANONICAL_ROUTE_TYPE`).
    """
    from transitlos.scoring.mrc import mode_category

    routes_path = gtfs_dir / "routes.txt"
    # `route_id` is int in some real feeds' routes.txt, string in others --
    # cast to Utf8 so `resolve_feed_modes` results from different feeds can
    # be `pl.concat`-ed together in `_attach_scoring_mode`.
    routes = pl.read_csv(routes_path, infer_schema_length=None).with_columns(
        pl.col("route_id").cast(pl.Utf8)
    )

    ntd_modes = {r.get("mode_name") for r in ntd_rows}
    if "route_type" not in routes.columns:
        # Some feeds (Metra) ship a routes.txt without route_type: take the mode from NTD (rail / tram / bus).
        ntd_mode = NTD_MODE_TO_SCORING_MODE.get(next(iter(ntd_modes))) if len(ntd_modes) == 1 else None
        fill = {"rail": 2, "tram": 0, "bus": 3}.get(ntd_mode, 3)
        logger.warning(f"Feed '{gtfs_dir}': routes.txt has no route_type -- using NTD mode {ntd_mode!r} (route_type {fill}).")
        routes = routes.with_columns(pl.lit(fill).alias("route_type"))
    unique_types = routes["route_type"].unique().to_list()
    default_mode = None
    if len(ntd_modes) == 1:
        default_mode = NTD_MODE_TO_SCORING_MODE.get(next(iter(ntd_modes)))

    contradiction = len(unique_types) > 1 and default_mode is not None
    if contradiction:
        logger.warning(
            f"Feed '{gtfs_dir}': NTD says single mode {ntd_modes!r} but real "
            f"route_type is non-unique ({unique_types}) -- falling back to "
            "per-route route_type for scoring mode, per the agreed rule."
        )

    def _row_mode(route_type: int) -> str:
        if route_type == BRT_ROUTE_TYPE:
            return "tram"
        if contradiction or default_mode is None:
            return mode_category(route_type)
        return default_mode

    scoring_modes = [_row_mode(rt) for rt in routes["route_type"].to_list()]
    return routes.select(["route_id", "route_type"]).with_columns(
        pl.Series("scoring_mode", scoring_modes),
        pl.Series(
            "canonical_route_type",
            [_SCORING_MODE_TO_CANONICAL_ROUTE_TYPE[m] for m in scoring_modes],
        ),
    )


def _feed_has_service_in_window(
    gtfs_dir: Path, aoi: gpd.GeoDataFrame, window: tuple, stop_group_distance: float
) -> bool:
    """Whether a single feed has any trip active inside `window`.

    Builds a lightweight single-feed `Feed` filtered to `window` and
    checks for at least one trip -- reuses `Feed`'s own calendar/
    calendar_dates date-filtering logic (respecting exceptions), rather
    than a hand-rolled `calendar.txt` range check.
    """
    feed = Feed(
        gtfs_dirs=[str(gtfs_dir)],
        aoi=aoi,
        start_date=ANALYSIS_DATE,
        end_date=ANALYSIS_DATE,
        stop_group_distance=stop_group_distance,
    )
    return feed.trips.lf.limit(1).collect().height > 0


def split_feeds_by_window(
    gtfs_dirs: Sequence[Path],
    aoi: gpd.GeoDataFrame,
    window: tuple = DEFAULT_WINDOW,
    stop_group_distance: float = STOP_GROUP_DISTANCE_M,
) -> tuple[List[Path], List[Path]]:
    """Partition a state's downloaded feeds into "has service in `window`" / fallback.

    Args:
        gtfs_dirs: Extracted GTFS feed directories for one state.
        aoi: State AOI (see module docstring -- always applied).
        window: Shared analysis date range.
        stop_group_distance: Forwarded to the per-feed `Feed` check.

    Returns:
        `(in_window, fallback)` -- `fallback` feeds have zero trips
        anywhere in `window` and must be scored individually with their
        own representative date instead.
    """
    in_window, fallback = [], []
    for gtfs_dir in gtfs_dirs:
        try:
            has_service = _feed_has_service_in_window(gtfs_dir, aoi, window, stop_group_distance)
        except Exception as exc:  # pragma: no cover - real feeds vary widely
            logger.warning(f"Feed '{gtfs_dir}': could not check service window, treating as fallback: {exc}")
            has_service = False
        (in_window if has_service else fallback).append(gtfs_dir)
    return in_window, fallback


def _route_modes_for(gtfs_dirs: Sequence[Path], ntd_rows_by_dir: Dict[Path, List[Dict]]) -> pl.DataFrame:
    """Concatenate `resolve_feed_modes` results across several feed directories."""
    mode_frames = []
    for gtfs_dir in gtfs_dirs:
        ntd_rows = ntd_rows_by_dir.get(gtfs_dir, [])
        try:
            mode_frames.append(resolve_feed_modes(gtfs_dir, ntd_rows))
        except Exception as exc:  # pragma: no cover - malformed routes.txt varies
            logger.warning(f"Feed '{gtfs_dir}': could not resolve modes: {exc}")
    if not mode_frames:
        return pl.DataFrame(schema={"route_id": pl.Utf8, "scoring_mode": pl.Utf8})
    return pl.concat(mode_frames, how="diagonal_relaxed")


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


def score_state_stops(
    gtfs_dirs: Sequence[Path],
    ntd_rows_by_dir: Dict[Path, List[Dict[str, object]]],
    state_aoi: gpd.GeoDataFrame,
    window: tuple = DEFAULT_WINDOW,
    stop_group_distance: float = STOP_GROUP_DISTANCE_M,
) -> gpd.GeoDataFrame:
    """Build one state's scored `stops` table (the `stops.geoparquet` content).

    Args:
        gtfs_dirs: Every downloaded, extracted GTFS feed directory for
            this state (already BRT-overridden by
            `pyGTFSHandler.downloaders.usa.ntd.NTDDownloader.download_feeds`).
        ntd_rows_by_dir: Maps each `gtfs_dirs` entry to the NTD row(s)
            whose `download_url` produced it (for mode resolution).
        state_aoi: State boundary, EPSG:4326. Passed as `aoi=` to every
            `Feed`/`download_and_prepare_stops` call in this function --
            never omitted, so cross-border agencies only keep this
            state's own stops/routes.
        window: Shared analysis window for the combined multi-feed load.
        stop_group_distance: Meters; forwarded to every `Feed`.

    Returns:
        GeoDataFrame with one row per `parent_station` (`stop_lat`/
        `stop_lon`/`geometry`, `route_ids`, `scoring_mode` -- the winning
        mode tier, see `best_subset_scoring`, `stop_score`, and every
        sub-score column `best_subset_stop_scores` returns), pooling the
        combined-window load and every fallback feed's own
        individually-scored stops.
    """
    from .best_subset_scoring import best_subset_stop_scores

    in_window, fallback = split_feeds_by_window(gtfs_dirs, state_aoi, window, stop_group_distance)
    frames = []

    # Feeds with service in the window are loaded together (one stop clustering, headways combined across them).
    # Feeds WITHOUT service in the window (expired calendars) are NOT combined with anything: each is scored alone
    # on its own representative date. Every group's stops are kept (even at the same place) and concatenated, so
    # isochrones are computed for all of them and headways are never combined across groups; the LOS step then takes
    # the best access per street edge.
    if in_window:
        def _score_in_window(dirs):
            feed = Feed(
                gtfs_dirs=[str(d) for d in dirs],
                aoi=state_aoi,
                start_date=window[0],
                end_date=window[1],
                stop_group_distance=stop_group_distance,
            )
            route_modes = _route_modes_for(dirs, ntd_rows_by_dir)
            scored = best_subset_stop_scores(feed, ANALYSIS_DATE, route_modes, ANALYSIS_START_TIME, ANALYSIS_END_TIME)
            if scored.empty:
                return None
            return feed.add_stop_coords(scored)

        try:
            combined = _score_in_window(in_window)
            if combined is not None:
                frames.append(combined)
        except Exception as exc:
            # One bad feed must not discard the others: retry each alone.
            logger.warning(f"Combined in-window load failed ({exc}); retrying feeds one by one.")
            for d in in_window:
                try:
                    single = _score_in_window([d])
                except Exception as exc2:
                    logger.warning(f"Feed '{d}': could not load -- skipping: {exc2}")
                    continue
                if single is not None:
                    frames.append(single)
    else:
        logger.warning(f"No feeds have service in {window} -- every feed will use its own fallback date.")

    for gtfs_dir in fallback:
        try:
            feed = Feed(gtfs_dirs=[str(gtfs_dir)], aoi=state_aoi, stop_group_distance=stop_group_distance)
            rep_date = feed.get_representative_date(date_type="weekday")
        except Exception as exc:  # e.g. feed has no stops in this state, or is malformed
            logger.warning(f"Feed '{gtfs_dir}': could not load -- skipping: {exc}")
            continue
        if rep_date is None:
            logger.warning(f"Feed '{gtfs_dir}': no representative date found at all -- skipping.")
            continue
        route_modes = _route_modes_for([gtfs_dir], ntd_rows_by_dir)
        try:
            fallback_stops = best_subset_stop_scores(feed, rep_date, route_modes, ANALYSIS_START_TIME, ANALYSIS_END_TIME)
        except Exception as exc:  # pragma: no cover - real feeds vary widely
            logger.warning(f"Feed '{gtfs_dir}': fallback scoring failed on {rep_date}: {exc}")
            continue
        if fallback_stops.empty:
            continue
        fallback_stops = feed.add_stop_coords(fallback_stops)
        fallback_stops["source_feed"] = str(gtfs_dir)
        fallback_stops["representative_date"] = str(rep_date)
        frames.append(fallback_stops)

    if not frames:
        return gpd.GeoDataFrame(columns=["parent_station", "geometry"], geometry="geometry", crs=4326)

    # `how="diagonal_relaxed"`: fallback frames carry extra columns
    # (`source_feed`/`representative_date`) the combined-window frame
    # doesn't, and dtypes can differ slightly across feeds -- diagonal
    # concat fills missing columns with null instead of requiring an
    # identical schema everywhere, and `_relaxed` widens mismatched dtypes
    # instead of raising.
    all_stops = pl.concat(
        [pl.from_pandas(f) if isinstance(f, pd.DataFrame) else f for f in frames], how="diagonal_relaxed"
    ).to_pandas()

    geometry = gpd.points_from_xy(all_stops["stop_lon"], all_stops["stop_lat"])
    return merge_same_location_stops(gpd.GeoDataFrame(all_stops, geometry=geometry, crs=4326))
