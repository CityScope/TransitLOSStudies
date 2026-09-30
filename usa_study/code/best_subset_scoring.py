# -*- coding: utf-8 -*-
"""Best mode-tier / min-speed subset search per parent_station.

Design confirmed with Miguel (2026-09-29): a stop's final `stop_score`
should be the BEST achievable one, not just the one combining every route
serving it as-is. Two knobs are searched per `parent_station`, crossed:

1. **Minimum mode tier** -- exactly 3 choices, not a per-route powerset:
   - "bus" tier: keep every route (bus + tram + rail, if any at the stop).
   - "tram" tier: drop bus routes, keep tram + rail.
   - "rail" tier: keep only rail.
   A tier with zero remaining routes at a given stop is skipped.
2. **Minimum speed cutoff**: candidates are that mode tier's own route
   speeds, ROUNDED TO THE NEAREST 2 KM/H bucket (Miguel's simplification,
   2026-09-29 -- avoids a near-infinite set of near-duplicate raw-speed
   cutoffs; the final isochrone/PTLOS output is itself ceiling-discretized
   onto a finite grid, so this bounded set of candidates already covers
   every combination that could change the final result). A route is
   "kept" under a cutoff if its own speed >= that cutoff.

**Headway is NOT naively recombined from precomputed per-route numbers**
(Miguel, 2026-09-29: "headway is computed for the complete stop so it
cannot be aggregated by computing headway of individual routes" -- a flat
harmonic sum across routes, ignoring direction, is NOT the same quantity
as the real combined headway). But re-deriving it from the raw feed via a
fresh `_get_headway_at_stops` call per (stop, subset) trial turned out
far too slow at real multi-feed scale (didn't finish for NM's 10 combined
feeds within several minutes -- every trial re-ran the expensive base
date/time filtering + frequency expansion from scratch).

The actual fix, reading `pyGTFSHandler.analysis.stops._get_headway_at_stops`'s
own source: its `how="add"` result is ITSELF built by combining the exact
same per-`(parent_station, route_id, direction_id)` headway values
`how="all"` returns -- group by `direction_id`, harmonic-sum
(`1/sum(1/h)`) WITHIN each direction, then keep whichever direction's
combined headway is lowest (`mix_directions=False`, the default). So
this fetches that atomic `how="all"` table ONCE per feed (the one
expensive pass), then replicates that exact same group-by-direction/
harmonic-sum/pick-best-direction algorithm itself, in memory, restricted
to each candidate subset's `route_id`s -- reproducing the REAL production
formula exactly (not an approximation, and not the earlier, wrong,
direction-blind flat harmonic sum), while being cheap to repeat per
subset since no raw-feed re-filtering happens after the first fetch.

Speed is different: `get_speed_at_stops(how="mean")`'s aggregation is a
plain trip-count-weighted average of each route's `distance_weight`/
`time_weight` (`sum(distance)/sum(time)`, linear and associative over
whichever rows are included), so recombining a subset of ALREADY-FETCHED
per-route `distance_weight`/`time_weight` pairs by summing them
reproduces the exact same value `get_speed_at_stops` would compute
directly on that subset -- no direction-grouping/edge-case structure is
lost the way it is for headway, so no filtered re-fetch is needed there.

The combination with the highest resulting `stop_score` (via
`transitlos.scoring`'s own `mrc_score`/`speed_score`/`frequency_score`/
`stop_score`, reusing the exact same literature-grounded functions
`transitlos.stop_scores.compute_stop_scores` does, just per-combination
instead of once) wins.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Optional, Union

import pandas as pd
import polars as pl

from pyGTFSHandler.feed import Feed
from transitlos.scoring.frequency import frequency_score
from transitlos.scoring.mrc import mrc_score
from transitlos.scoring.speed import speed_score
from transitlos.scoring.stop_score import stop_score as compute_final_stop_score

#: Ordered coarsest (most inclusive) to finest (least inclusive) -- see
#: module docstring. `higher_tiers[mode]` is which `scoring_mode` values
#: stay in scope once that tier is the minimum.
_MODE_TIERS: dict[str, set[str]] = {
    "bus": {"bus", "tram", "rail"},
    "tram": {"tram", "rail"},
    "rail": {"rail"},
}

#: Feeds `mrc_score`/`compute_stop_scores` a representative literal
#: `route_type` for a resolved scoring_mode -- see
#: `gtfs_stage._SCORING_MODE_TO_CANONICAL_ROUTE_TYPE` (kept in sync;
#: duplicated here to avoid a circular import between the two modules).
_SCORING_MODE_TO_CANONICAL_ROUTE_TYPE = {"bus": 3, "tram": 0, "rail": 2}

#: Speed-cutoff discretization bucket (km/h) -- Miguel's simplification,
#: see module docstring.
_SPEED_BUCKET_KMH = 2.0


def _combined_headway_from_subset(rows: list[dict]) -> Optional[float]:
    """Replicate `_get_headway_at_stops(how="add", mix_directions=False)` on a route subset.

    `rows` are `(route_id, direction_id, headway_minutes)` entries for ONE
    `parent_station`, already restricted to a candidate subset of
    `route_id`s (the caller's job). Groups by `direction_id`, harmonic-sums
    headways WITHIN each direction, then returns whichever direction's
    combined headway is lowest (best) -- exactly the real algorithm's
    `mix_directions=False` path, just run here in memory on however many
    of a stop's routes the caller chose to keep, instead of on all of them.
    """
    by_direction: dict[object, list[float]] = {}
    for r in rows:
        h = r["headway_minutes"]
        if h is None or h <= 0:
            continue
        by_direction.setdefault(r["direction_id"], []).append(h)

    best_headway = None
    for headways in by_direction.values():
        combined = 1.0 / sum(1.0 / h for h in headways)
        if best_headway is None or combined < best_headway:
            best_headway = combined
    return best_headway


def _weighted_speed(distances: list[float], times: list[float]) -> Optional[float]:
    """`sum(distance) / sum(time) * 3.6` (km/h), matching `get_speed_at_stops(how="mean")`."""
    pairs = [(d, t) for d, t in zip(distances, times) if d is not None and t is not None and t > 0]
    if not pairs:
        return None
    total_d = sum(d for d, _ in pairs)
    total_t = sum(t for _, t in pairs)
    return (total_d / 1000.0) / (total_t / 3600.0)


def best_subset_stop_scores(
    feed: Feed,
    analysis_date: Union[date, datetime],
    route_modes: pl.DataFrame,
    start_time: Union[datetime, time] = time.min,
    end_time: Union[datetime, time] = time.max,
    region: str = "global",
) -> pd.DataFrame:
    """Search mode-tier x min-speed combinations per parent_station, keep the best.

    Args:
        feed: An already-constructed `pyGTFSHandler.feed.Feed` (its own
            `aoi`/date filters already applied).
        analysis_date: Single service date to analyze (same convention as
            `transitlos.stops.download_and_prepare_stops`'s `date_range[0]`
            -- this function does not itself pick/average across a range).
        route_modes: Per-route `scoring_mode` (`"bus"`/`"tram"`/`"rail"`),
            as returned by `gtfs_stage.resolve_feed_modes` (columns
            `route_id`, `scoring_mode`).
        start_time: Forwarded to `get_headway_at_stops`/`get_speed_at_stops`.
        end_time: Forwarded to `get_headway_at_stops`/`get_speed_at_stops`.
        region: Forwarded to every `transitlos.scoring` call.

    Returns:
        pandas DataFrame with one row per `parent_station`: `parent_station`,
        `route_ids` (every route kept in the winning combination),
        `scoring_mode` (winning mode tier), `headway_minutes`,
        `avg_speed_kmh`, `min_speed_cutoff_kmh`, `mrc_score`, `speed_score`,
        `frequency_score`, `stop_score`. A `parent_station` with no valid
        combination at all (e.g. every route has null headway/speed) is
        omitted.
    """
    # One expensive pass each: every route's own per-direction headway
    # (the atomic values the real "add" algorithm combines -- see module
    # docstring), and every route's speed components (for candidate
    # cutoffs; recombining these is exact, see module docstring).
    headway = feed.get_headway_at_stops(
        analysis_date, start_time, end_time, by="route_id", at="parent_station", how="all"
    )
    if isinstance(headway, pl.LazyFrame):
        headway = headway.collect()
    headway = headway.select(["parent_station", "route_id", "direction_id", "headway"]).rename(
        {"headway": "headway_minutes"}
    )
    headway = headway.with_columns(pl.col("route_id").cast(pl.Utf8))

    speed = feed.get_speed_at_stops(
        analysis_date, start_time, end_time, by="route_id", at="parent_station", how="mean"
    )
    speed = speed.select(["parent_station", "route_id", "distance_weight", "time_weight"])
    speed = speed.with_columns(pl.col("route_id").cast(pl.Utf8))
    # One row per (parent_station, route_id) -- distinct from `headway`,
    # which has one row per (parent_station, route_id, direction_id). Kept
    # separate so summing speed components for a kept-route subset never
    # double-counts a route with more than one direction.
    speed = speed.join(
        route_modes.select(["route_id", "scoring_mode"]).with_columns(pl.col("route_id").cast(pl.Utf8)),
        on="route_id",
        how="left",
    )
    speed = speed.with_columns(pl.col("scoring_mode").fill_null("bus"))
    speed = speed.with_columns(
        (pl.col("distance_weight") / 1000.0 / (pl.col("time_weight") / 3600.0)).alias("route_speed_kmh")
    )

    headway_by_station = {
        station[0] if isinstance(station, tuple) else station: group.to_dicts()
        for station, group in headway.group_by("parent_station")
    }

    results = []
    for parent_station, group in speed.group_by("parent_station"):
        parent_station = parent_station[0] if isinstance(parent_station, tuple) else parent_station
        rows = group.to_dicts()
        station_headway_rows = headway_by_station.get(parent_station, [])
        best = None
        # Memoize by the actual kept-route-id set within this stop: several
        # (tier, cutoff) pairs can produce an identical subset, and this
        # dedupes the (already-cheap, but not free) recombination.
        headway_cache: dict[frozenset, Optional[float]] = {}

        for tier_name, tier_modes in _MODE_TIERS.items():
            tier_rows = [r for r in rows if r["scoring_mode"] in tier_modes]
            if not tier_rows:
                continue

            bucketed_speeds = sorted(
                {
                    round(r["route_speed_kmh"] / _SPEED_BUCKET_KMH) * _SPEED_BUCKET_KMH
                    for r in tier_rows
                    if r["route_speed_kmh"] is not None
                }
            )
            # A tier with no real speed data anywhere still deserves one
            # trial (cutoff 0.0 keeps everything) rather than being skipped
            # outright.
            cutoffs = bucketed_speeds or [0.0]

            for cutoff in cutoffs:
                kept = [
                    r
                    for r in tier_rows
                    if r["route_speed_kmh"] is None or r["route_speed_kmh"] >= cutoff
                ]
                if not kept:
                    continue

                kept_ids = [str(r["route_id"]) for r in kept]
                cache_key = frozenset(kept_ids)
                if cache_key not in headway_cache:
                    subset_headway_rows = [r for r in station_headway_rows if r["route_id"] in cache_key]
                    headway_cache[cache_key] = _combined_headway_from_subset(subset_headway_rows)
                combined_headway = headway_cache[cache_key]
                combined_speed = _weighted_speed(
                    [r["distance_weight"] for r in kept], [r["time_weight"] for r in kept]
                )
                if combined_headway is None or combined_speed is None:
                    continue

                canonical_route_type = _SCORING_MODE_TO_CANONICAL_ROUTE_TYPE[tier_name]
                mrc = mrc_score(route_type=canonical_route_type, headway_cv=None, region=region)
                sp = speed_score(combined_speed, region=region)
                fr = frequency_score(combined_headway, region=region)
                score = compute_final_stop_score(mrc, sp, fr, region=region)

                if best is None or score > best["stop_score"]:
                    best = {
                        "parent_station": parent_station,
                        "route_ids": ";".join(sorted(kept_ids)),
                        "scoring_mode": tier_name,
                        "headway_minutes": combined_headway,
                        "avg_speed_kmh": combined_speed,
                        "min_speed_cutoff_kmh": cutoff,
                        "mrc_score": mrc,
                        "speed_score": sp,
                        "frequency_score": fr,
                        "stop_score": score,
                    }

        if best is not None:
            results.append(best)

    return pd.DataFrame(results)
