# -*- coding: utf-8 -*-
"""Stage 2: per-county street network + LOS + census/h3 resampling.

Design (per `usa_study/prompt.txt` part 2, and Miguel's explicit
2026-09-29 instruction: "as states are very [varied in size] the
isochrone process should happen per county and then join all"):

1. Per state: download (or reuse a cached) state-level `.osm.pbf` via
   Geofabrik under `usa_study/osm_files/` (`UrbanAccessAnalyzer.osm_io
   .download_geofabrik` already picks the smallest Geofabrik region that
   fully contains a given AOI -- for a US state AOI that's Geofabrik's own
   state-level extract, so no separate "best-matching file" logic is
   needed here).
2. Per county: load the street network from that PBF cropped to the
   county, simplify (10m clusters), and connectivity-repair
   (`transitlos.network.prepare_street_network` already does load +
   simplify + `crop_by_aoi_connected` in one call -- this satisfies the
   prompt's "simplify + connected components pass to delete unconnected
   edges" step directly).
3. Filter the state's `stops.geoparquet` to this county + a 1500m buffer,
   then crop the network again to the buffered union of those stops (the
   prompt's "delete streets outside [max isochrone distance] of any
   stop" step) -- this second crop is a plain spatial clip, not another
   connectivity repair, since fragmentation into separate per-stop
   islands is now expected and correct, not a defect to fix.
4. Run `transitlos.level_of_service.compute_level_of_service` with the
   pre-built network/stops (1500m max walk distance) to get a
   street-edges GeoDataFrame with `level_of_service`.
5. Fetch census levels NATIVELY per
   `~/.claude/.../memory/usa_study_native_level_rule.md` (DHC block/
   blockgroup/tract/county, ACS5 blockgroup/tract/county -- never
   resampled between two natively-covered levels), join `level_of_service`
   onto each via `UrbanAccessAnalyzer.geometry_levels.edges_to_level`
   (length-weighted line-to-polygon aggregation).
6. Build h3 res 11/9 grids over the county, join `level_of_service` the
   same way.
7. Save per-county: `street.geoparquet` (score > 0 only, clipped to the
   county boundary itself, not the buffered stop area), and one
   geoparquet per census/h3 level (population/jobs > 0 only).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

import h3
import numpy as np
import shapely
from geohierarchy import h3_cells
from geohierarchy.utils import h3_cells_from_ids
from geohierarchy.aggregation import Mean
from transitlos.level_of_service import compute_level_of_service
from transitlos.network import prepare_street_network
from UrbanAccessAnalyzer import geometry_levels
from UrbanAccessAnalyzer.api import AreaOfInterest
from UrbanAccessAnalyzer.osm_io import download_geofabrik

logger = logging.getLogger(__name__)

#: Prompt's own max isochrone distance -- used both for `compute_level_of_service`'s
#: `max_walk_distance_m` and for how far the stops buffer (step 3 above)
#: reaches around each stop.
MAX_WALK_DISTANCE_M = 1500.0

#: Street-network simplification cluster distance (prompt's "edge length 10m").
SIMPLIFY_DISTANCE_M = 10.0

#: h3 resolutions to build per county (prompt's "maximum h3 level... res 11) and res 9").
H3_RESOLUTIONS = (11, 9, 7)

#: Same for every county: res 7 covers all populated area; finer
#: resolutions (> `H3_COARSE_RESOLUTION`) are only generated inside census
#: tracts denser than this (people per km2).
H3_COARSE_RESOLUTION = 7
H3_FINEST_RESOLUTION = 11
H3_SCHEME_MARKER = "h3_scheme_v4.txt"
#: Buffer (m) around LOS > 0 street edges when selecting res 11 cells.
H3_LOS_REGION_BUFFER_M = 40.0
H3_DENSE_TRACT_PER_KM2 = 1000.0


def get_state_pbf(state_aoi: gpd.GeoDataFrame, osm_files_dir: Path) -> str:
    """Download (or reuse) the smallest Geofabrik region covering a state.

    Args:
        state_aoi: State boundary, EPSG:4326.
        osm_files_dir: `usa_study/osm_files/` -- shared across states/
            counties (Geofabrik regions commonly cover more than one
            county, so this is cached at the `osm_files/` level, not
            per-county or even always per-state, exactly matching the
            prompt's "check if the file already exists to avoid
            redownloading").

    Returns:
        Path to the downloaded/cached `.osm.pbf`.
    """
    os.makedirs(osm_files_dir, exist_ok=True)
    # `download_geofabrik` itself already checks for an existing file with
    # the region's expected name before downloading -- see its docstring/
    # implementation in `UrbanAccessAnalyzer.osm_io`.
    return download_geofabrik(state_aoi, output_folder=str(osm_files_dir))


def _to_wkb_polars(gdf: gpd.GeoDataFrame):
    """Convert a plain GeoDataFrame to the WKB-geometry Polars table `edges_to_level` expects."""
    import polars as pl
    import shapely

    wkb = shapely.to_wkb(gdf.geometry.to_numpy())
    df = pl.from_pandas(gdf.drop(columns="geometry").reset_index(drop=True))
    return df.with_columns(pl.Series("geometry", wkb, dtype=pl.Binary))


def _buffer_m(gdf: gpd.GeoDataFrame, distance_m: float) -> gpd.GeoDataFrame:
    """Buffer a EPSG:4326 GeoDataFrame by `distance_m`, via a local UTM CRS."""
    utm = gdf.estimate_utm_crs()
    buffered = gdf.to_crs(utm)
    buffered["geometry"] = buffered.geometry.buffer(distance_m)
    return buffered.to_crs(4326)


STOPS_SIGNATURE_FILE = "stops_signature.txt"


def county_stops_signature(county_gdf: gpd.GeoDataFrame, stops_gdf: gpd.GeoDataFrame) -> str:
    """Fingerprint of the stops that can affect a county (those within walking distance of it): location and score.

    Stored with the county's outputs; when the stops are rescored later, only counties whose fingerprint changed are
    recomputed (scores compared at 0.01, locations at ~1 m).
    """
    import hashlib

    buf = _buffer_m(county_gdf, MAX_WALK_DISTANCE_M).union_all()
    sel = stops_gdf[stops_gdf.intersects(buf)]
    rows = sorted(zip(sel["stop_lat"].round(5), sel["stop_lon"].round(5), sel["stop_score"].round(2)))
    return hashlib.sha1(repr(rows).encode()).hexdigest()


def process_county(
    county_gdf: gpd.GeoDataFrame,
    county_geoid: str,
    state_pbf_path: str,
    stops_gdf: gpd.GeoDataFrame,
    county_dir: Path,
    network_cache_dir: Optional[Path] = None,
    census_cache_dir: Optional[Path] = None,
) -> dict:
    """Run the full per-county isochrone/LOS pipeline, saving outputs into `county_dir`.

    Args:
        county_gdf: Single-row GeoDataFrame of this county's boundary,
            EPSG:4326 (from the state's `county` level, native to DHC).
        county_geoid: This county's `GEOID`, for logging/filenames.
        state_pbf_path: Path to the state's downloaded `.osm.pbf` (see
            `get_state_pbf`).
        stops_gdf: The state's full scored `stops.geoparquet`
            (`gtfs_stage.score_state_stops`'s output), EPSG:4326.
        county_dir: Output directory for this county (created if missing).
        network_cache_dir: Optional cache dir for the simplified/cropped
            street network (forwarded to `prepare_street_network`).
        census_cache_dir: pyCensus parquet cache directory -- pass the
            SAME path across every county in a state (and across states,
            if convenient) so census fetches are cached/reused rather
            than re-downloaded per county. Defaults to
            `pycensus.cache.default_cache_dir()` (pyCensus's own default)
            if not given.

    Returns:
        Dict summary: `n_stops`, `n_edges_scored`, `n_edges_total`, and
        which output files were written (empty results are skipped, not
        written as empty files).
    """
    os.makedirs(county_dir, exist_ok=True)
    county_aoi = AreaOfInterest(county_gdf)

    county_buffer = _buffer_m(county_gdf, MAX_WALK_DISTANCE_M)
    stops_in_county = stops_gdf[stops_gdf.intersects(county_buffer.union_all())].reset_index(drop=True)
    n_stops = len(stops_in_county)

    written = {}
    scored_edges = []
    edges_gdf = None
    edges_wkb = None
    scored_wkb = None
    street_geoms = None
    n_network_edges = n_cropped_edges = 0

    if n_stops > 0:
        # The network covers the county PLUS the walking distance around it: a stop just outside the border reaches
        # streets inside it only through streets outside it. Results are still restricted to the county below.
        network = prepare_street_network(
            AreaOfInterest(county_buffer),
            cache_dir=str(network_cache_dir) if network_cache_dir else None,
            cluster_distance=SIMPLIFY_DISTANCE_M,
            pbf_path=state_pbf_path,
            network_type="all",
            ignore_oneway=True,
        )
        stops_union_aoi = AreaOfInterest(_buffer_m(stops_in_county, MAX_WALK_DISTANCE_M))
        # Plain spatial clip, NOT another connectivity repair -- see module
        # docstring step 3. `StreetNetwork.crop`'s own `crop_by_aoi_connected`
        # still runs, but with a buffer of 0 it degenerates to an exact clip
        # per real (now-expected-to-be-many) connected piece.
        # Street PRESENCE (every street in the county, scored or not) decides, per h3 cell, between
        # "has streets but no transit access" (LOS 0) and "no streets at all" (no score).
        street_geoms = gpd.GeoSeries(
            shapely.from_wkb(network.edges["geometry_wkb"].to_numpy()), crs=network.crs
        ).to_crs(4326)
        n_network_edges = network.edges.height
        network = network.crop(stops_union_aoi, crop_buffer_m=0.0)
        n_cropped_edges = network.edges.height

        edges_gdf = compute_level_of_service(
            aoi=county_aoi,
            network=network,
            stops=stops_in_county,
            max_walk_distance_m=MAX_WALK_DISTANCE_M,
        )

        if len(edges_gdf) == 0:  # no streets near the stops: same as no transit (LOS 0)
            edges_gdf = None
    if edges_gdf is not None:
        county_geom = county_gdf.to_crs(edges_gdf.crs).union_all()
        edges_in_county = edges_gdf[edges_gdf.intersects(county_geom)]
        scored_edges = edges_in_county[edges_in_county["level_of_service"] > 0]
        if len(scored_edges) > 0:
            street_path = county_dir / "street.geoparquet"
            scored_edges.to_crs(4326).to_parquet(street_path)
            written["street"] = str(street_path)

        # `edges_to_level` labels its input WKB with whatever `crs` it's given
        # rather than reprojecting -- so reproject to WGS84 here FIRST, the
        # same pattern `UrbanAccessAnalyzer.api.AccessibilityAnalyzer.to_h3`/
        # `.to_level` use (both reproject edges to WGS84 before ever calling
        # `edges_to_level`, since h3/census grids are natively WGS84 too).
        # Only edges with LOS > 0 feed the h3 means (same inputs as `rebuild_county_h3`, which only has these).
        scored_wgs = edges_in_county[edges_in_county["level_of_service"] > 0].to_crs(4326)
        scored_wkb = _to_wkb_polars(scored_wgs) if len(scored_wgs) else None
        # Census levels (block ... county) average over ALL of the county's edges, zeros included (not the buffer ring
        # around it); h3 uses the scored ones.
        edges_wkb = _to_wkb_polars(edges_in_county.to_crs(4326)) if len(edges_in_county) else None
    else:
        # No public transport (or street network) reaches this county: LOS is 0 everywhere, so
        # the street network is not downloaded/built at all.
        logger.warning(f"County {county_geoid}: no stops within {MAX_WALK_DISTANCE_M}m -- skipping street network, LOS = 0.")

    census_written = _join_census_and_h3(
        county_gdf, county_geoid, edges_wkb, county_dir, census_cache_dir, street_geoms=street_geoms,
        scored_edges=scored_wkb,
    )
    written.update(census_written)

    (county_dir / STOPS_SIGNATURE_FILE).write_text(county_stops_signature(county_gdf, stops_gdf))
    return {
        "n_stops": n_stops,
        "n_network_edges": n_network_edges,  # cached county+buffer network (never cropped to stops)
        "n_cropped_edges": n_cropped_edges,  # the part within walking range of the stops
        "n_edges_total": 0 if edges_gdf is None else len(edges_gdf),
        "n_edges_scored": len(scored_edges),
        "written": written,
    }


#: (level, source module attribute path) -- every source natively covering
#: that level is merged onto it, per the native-level rule
#: ([[usa_study_native_level_rule]]): never resample a level a source
#: already natively publishes.
_CENSUS_ADMIN_LEVELS = ("block", "blockgroup", "tract", "county")

#: Matches every `{year}_{office}_...` election column
#: (`_add_elections`/`elections.schema`'s own naming convention).
_ELECTION_COL_RE = re.compile(r"^\d{4}_(president|senate)_")


def _add_elections(county_gdf: gpd.GeoDataFrame, state_fips: str, cache_dir: str) -> gpd.GeoDataFrame:
    """Join every implemented MEDSL office/year onto county-level rows.

    County is elections' own native level here (see
    `pycensus.countries.usa.elections.loader`'s module docstring for
    exactly what's implemented -- House and most non-2024 non-presidential
    years are documented gaps, not silently produced as zero/missing).
    One column pair per `(year, office)`: `{year}_{office}_total_votes`
    and `{year}_{office}_{party}_share` for each major party (democrat/
    republican) -- a two-party-style summary rather than one raw column
    per individual candidate, since candidate names/count vary by race
    and this is the figure a chloropleth map actually wants.

    Args:
        county_gdf: County-level rows with a real `GEOID` column.
        state_fips: 2-digit state FIPS.
        cache_dir: pyCensus cache dir (also used for the elections
            source's own downloaded-file cache).

    Returns:
        `county_gdf` with elections columns added (left join -- a county
        with no election data available just gets nulls, not dropped).
    """
    from pycensus.countries.usa import elections

    result = county_gdf.copy()
    for office, year in elections.AVAILABLE:
        try:
            votes = elections.load(office, year, state_fips=state_fips, cache_dir=cache_dir)
        except Exception as exc:  # pragma: no cover - real network/source availability varies
            logger.warning(f"Elections {office}/{year} failed for state {state_fips}: {exc}")
            continue
        if votes.empty:
            continue

        prefix = f"{year}_{office}"
        totals = votes.groupby("GEOID")["votes"].sum().rename(f"{prefix}_total_votes")
        result = result.merge(totals, on="GEOID", how="left")

        # MEDSL's own column naming is inconsistent across sources:
        # 2000-2016 president uses "party", 2024 senate uses
        # "party_simplified" -- confirmed live 2026-09-29.
        party_col = next((c for c in ("party", "party_simplified") if c in votes.columns), None)
        if party_col:
            for party in ("democrat", "republican", "DEMOCRAT", "REPUBLICAN"):
                party_votes = votes[votes[party_col].astype(str).str.lower() == party.lower()]
                if party_votes.empty:
                    continue
                party_key = party.lower()
                share_col = f"{prefix}_{party_key}_share"
                if share_col in result.columns:
                    continue  # already added (avoid duplicate DEMOCRAT/democrat casing pass)
                party_sum = party_votes.groupby("GEOID")["votes"].sum()
                result = result.merge(party_sum.rename("_party_votes"), on="GEOID", how="left")
                result[share_col] = (result["_party_votes"] / result[f"{prefix}_total_votes"].replace(0, pd.NA)).fillna(0.0)
                result = result.drop(columns="_party_votes")

    return result


def _disaggregate_elections_onto(
    target_gdf: gpd.GeoDataFrame, county_election_row: Optional[pd.Series], pop_col: str = "population"
) -> gpd.GeoDataFrame:
    """Resample county-native election columns DOWN onto a finer level (block/blockgroup/tract/h3).

    Elections are only ever fetched natively at county (see `_add_elections`
    -- MEDSL's real published granularity for the offices/years this
    project implements). Per Miguel's explicit direction (2026-09-29:
    "download the election data at the election census levels and then
    resample to blockgroup, tract, and all the other levels and h3
    cells"), every finer level should still carry election data, resampled
    down rather than left blank.

    Count columns (`*_total_votes`) are apportioned by each target row's
    share of the county's total population (mass-conserving: summing
    every target row's apportioned value reproduces the county total).
    Share/rate columns (`*_..._share`) are copied through UNCHANGED --
    this data has no finer-than-county breakdown of who voted for whom,
    so the honest disaggregation is "this share applies uniformly across
    the whole county," not a fabricated finer pattern.

    Args:
        target_gdf: Block/blockgroup/tract/h3-cell rows, already carrying
            their own `population` column (used as the apportionment
            weight).
        county_election_row: The single county row's election columns
            (a `pd.Series`, e.g. `county_gdf.iloc[0]`), or `None` if this
            county has no election data at all (nothing added).

    Returns:
        `target_gdf` with every `{year}_{office}_...` column added.
    """
    if county_election_row is None or pop_col not in target_gdf.columns:
        return target_gdf

    election_cols = [c for c in county_election_row.index if _ELECTION_COL_RE.match(c)]
    if not election_cols:
        return target_gdf

    target_gdf = target_gdf.copy()
    county_population = pd.to_numeric(target_gdf[pop_col], errors="coerce").fillna(0).sum()
    pop_share = (
        pd.to_numeric(target_gdf[pop_col], errors="coerce").fillna(0) / county_population
        if county_population > 0
        else 0.0
    )

    for col in election_cols:
        value = county_election_row[col]
        if pd.isna(value):
            continue
        if col.endswith("_share"):
            target_gdf[col] = value
        else:
            target_gdf[col] = value * pop_share

    return target_gdf


def _fetch_native_census_level(level: str, county_gdf: gpd.GeoDataFrame, state_fips: str, cache_dir: str):
    """Fetch one admin level, merging every source that natively covers it.

    Args:
        level: One of `_CENSUS_ADMIN_LEVELS`.
        county_gdf: This county's boundary (used as the `aoi` clip).
        state_fips: State FIPS/abbreviation (`dhc.load`/`acs5.load`'s
            `states` argument).
        cache_dir: pyCensus parquet cache directory.

    Returns:
        A merged GeoDataFrame (real TIGER geometry from whichever source
        is present; DHC preferred when both are native at this level,
        since it's a full count, not a 5-year estimate), or `None` if no
        source natively covers this level.
    """
    from pycensus.countries.usa import acs5, dhc, lodes_wac
    from pycensus.countries.usa.acs5.schema import NATIVE_LEVELS as ACS5_NATIVE
    from pycensus.countries.usa.dhc.schema import NATIVE_LEVELS as DHC_NATIVE
    from pycensus.countries.usa.lodes_wac.schema import NATIVE_LEVELS as LODES_NATIVE

    frames = []
    if level in DHC_NATIVE:
        gdf = dhc.load(aoi=county_gdf, states=[state_fips], level=level, cache_dir=cache_dir)
        assert gdf.empty or gdf.attrs.get("native_level"), f"DHC {level} fetch was not native."
        if not gdf.empty:
            frames.append(gdf)
    if level in ACS5_NATIVE:
        gdf = acs5.load(aoi=county_gdf, states=[state_fips], level=level, cache_dir=cache_dir)
        assert gdf.empty or gdf.attrs.get("native_level"), f"ACS5 {level} fetch was not native."
        if not gdf.empty:
            frames.append(gdf)
    if level in LODES_NATIVE:
        gdf = lodes_wac.load_wac(aoi=county_gdf, states=[state_fips], level=level, cache_dir=cache_dir)
        assert gdf.empty or gdf.attrs.get("native_level"), f"LODES WAC {level} fetch was not native."
        if not gdf.empty:
            frames.append(gdf)

    if not frames:
        return None

    merged = frames[0]
    for extra in frames[1:]:
        extra_cols = [c for c in extra.columns if c not in ("geometry",) and c not in merged.columns]
        merged = merged.merge(extra[["GEOID", *extra_cols]], on="GEOID", how="outer")
    return merged


def _county_election_row(county_gdf, county_geoid, state_fips, cache_dir, county_centroid_geom):
    """Fetch this county's native county-level row (+ elections); returns (county_level_gdf, election_row)."""
    county_election_row: Optional[pd.Series] = None
    county_level_gdf = _fetch_native_census_level("county", county_gdf, state_fips, cache_dir)
    if county_level_gdf is not None and not county_level_gdf.empty:
        county_level_gdf = county_level_gdf[
            county_level_gdf.geometry.representative_point().within(county_centroid_geom)  # centroid can lie outside island / multi-part counties
        ].reset_index(drop=True)
        county_level_gdf = _add_elections(county_level_gdf, state_fips, cache_dir)
        if len(county_level_gdf) == 1:
            county_election_row = county_level_gdf.iloc[0]
        elif len(county_level_gdf) > 1:
            # A county-level fetch should return exactly this county's own
            # row; if more than one survived the centroid filter, match by
            # GEOID rather than guessing which row is "ours".
            matches = county_level_gdf[county_level_gdf["GEOID"] == county_geoid]
            if not matches.empty:
                county_election_row = matches.iloc[0]
    return county_level_gdf, county_election_row


def _join_census_and_h3(
    county_gdf: gpd.GeoDataFrame,
    county_geoid: str,
    edges_gdf: gpd.GeoDataFrame,
    county_dir: Path,
    census_cache_dir: Optional[Path] = None,
    street_geoms=None,
    scored_edges=None,
) -> dict:
    """Join `level_of_service` + population/jobs onto every census/h3 level, save geoparquets.

    Args:
        county_gdf: This county's boundary, EPSG:4326.
        county_geoid: This county's `GEOID` (for `states=` -- the first 2
            digits are the state FIPS).
        edges_gdf: `compute_level_of_service`'s scored street edges
            (projected CRS).
        county_dir: Output directory.
        census_cache_dir: See `process_county`'s own docstring.

    Returns:
        Dict of written file paths, one key per level (levels with no
        native census data, or where nothing survives the
        population/jobs > 0 filter, are simply absent).
    """
    state_fips = county_geoid[:2]
    if census_cache_dir is not None:
        cache_dir = str(census_cache_dir)
    else:
        from pycensus.cache import default_cache_dir

        cache_dir = default_cache_dir()
    written = {}
    county_centroid_geom = county_gdf.to_crs(4326).union_all()

    # Elections are only ever fetched natively at "county" ([[usa_study_native_level_rule]]);
    # fetch that row up front so block/blockgroup/tract/h3 can each get it
    # disaggregated down via `_disaggregate_elections_onto`, regardless of
    # `_CENSUS_ADMIN_LEVELS`' own iteration order.
    county_level_gdf, county_election_row = _county_election_row(
        county_gdf, county_geoid, state_fips, cache_dir, county_centroid_geom
    )

    for level in _CENSUS_ADMIN_LEVELS:
        if level == "county" and county_level_gdf is not None:
            gdf = county_level_gdf
        else:
            gdf = _fetch_native_census_level(level, county_gdf, state_fips, cache_dir)
        if gdf is None or gdf.empty:
            continue

        # "use centroid to check" (prompt.txt) -- keep only polygons whose
        # centroid actually falls in this county (a level fetched with
        # `aoi=county_gdf` can still return neighboring polygons that
        # merely intersect the clip bounds). Already applied above for
        # the reused `county_level_gdf`.
        if level != "county":
            centroids = gdf.geometry.representative_point()  # always inside the polygon (a centroid can fall outside it)
            gdf = gdf[centroids.within(county_centroid_geom)].reset_index(drop=True)
        if gdf.empty:
            continue

        # `edges_to_level` returns ONLY `id_col` + geometry + the
        # requested columns -- merge its `level_of_service` back onto the
        # original `gdf` (which carries every census column) rather than
        # replacing `gdf` with that slim result.
        if edges_gdf is None:
            gdf["level_of_service"] = 0.0
        else:
            los = geometry_levels.edges_to_level(
                edges_gdf, gdf, id_col="GEOID", columns=["level_of_service"], agg=Mean(), geometry_col="geometry"
            )
            gdf = gdf.merge(los[["GEOID", "level_of_service"]], on="GEOID", how="left")
        gdf["level_of_service"] = gdf["level_of_service"].fillna(0.0)

        pop_col = next((c for c in ("population", "dhc_population", "acs5_population") if c in gdf.columns), None)
        if level != "county" and pop_col is not None:
            gdf = _disaggregate_elections_onto(gdf, county_election_row, pop_col=pop_col)

        jobs_col = "lodes_wac_jobsTotal" if "lodes_wac_jobsTotal" in gdf.columns else None
        has_activity = pd.Series(False, index=gdf.index)
        if pop_col is not None:
            has_activity |= pd.to_numeric(gdf[pop_col], errors="coerce").fillna(0) > 0
        if jobs_col is not None:
            has_activity |= pd.to_numeric(gdf[jobs_col], errors="coerce").fillna(0) > 0
        gdf = gdf[has_activity]
        if gdf.empty:
            continue

        path = county_dir / f"{level}.geoparquet"
        gdf.to_crs(4326).to_parquet(path)
        written[level] = str(path)

    written.update(
        _write_h3_levels(
            county_gdf, county_geoid, scored_edges, county_dir, cache_dir, county_election_row, county_centroid_geom,
            street_geoms=street_geoms,
        )
    )
    return written


def _write_h3_levels(
    county_gdf, county_geoid, edges_gdf, county_dir, cache_dir, county_election_row, county_centroid_geom, street_geoms=None
) -> dict:
    """Write this county's h3 files: res 11 from the street edges, coarser levels derived from it.

    Res 11 exists wherever LOS > 0 (any population density) plus inside dense
    tracts (> `H3_DENSE_TRACT_PER_KM2`). Res 9 and 7 are then resampled from
    res 11 (population-weighted LOS). Population stays complete at every
    level because it comes from census blocks, not from res 11.
    """
    state_fips = county_geoid[:2]
    written = {}
    block_gdf = _fetch_native_census_level("block", county_gdf, state_fips, cache_dir)
    dense_region = _dense_tract_region(county_gdf, state_fips, cache_dir, county_centroid_geom)
    res11 = None
    for resolution in H3_RESOLUTIONS:  # finest first
        out = _join_h3_level(
            county_gdf, county_geoid, edges_gdf, block_gdf, resolution, county_dir, county_election_row,
            dense_region=dense_region, res11=res11, street_geoms=street_geoms,
        )
        if out is None:
            continue
        path, df = out
        written[f"h3_res{resolution}"] = path
        if resolution == H3_FINEST_RESOLUTION:
            res11 = df
    (county_dir / H3_SCHEME_MARKER).write_text(
        "res11: LOS>0 or dense tracts, mean of scored edges, 0 if streets but no score, null if no streets; coarser resampled\n"
    )
    return written


def _dense_tract_region(county_gdf, state_fips, cache_dir, county_geom):
    """Union (EPSG:4326) of this county's tracts denser than `H3_DENSE_TRACT_PER_KM2`, or None."""
    tracts = _fetch_native_census_level("tract", county_gdf, state_fips, cache_dir)
    if tracts is None or tracts.empty or "population" not in tracts.columns:
        return None
    tracts = tracts[tracts.geometry.centroid.within(county_geom)]
    return dense_union(tracts)


def dense_union(tracts: gpd.GeoDataFrame):
    """Union of tracts with population density above `H3_DENSE_TRACT_PER_KM2` (None if none)."""
    if tracts.empty:
        return None
    tracts = tracts.to_crs(4326)
    area_km2 = tracts.to_crs(tracts.estimate_utm_crs()).area / 1e6
    pop = pd.to_numeric(tracts["population"], errors="coerce").fillna(0)
    dense = tracts[(pop / area_km2.where(area_km2 > 0)) > H3_DENSE_TRACT_PER_KM2]
    if dense.empty:
        return None
    return dense.union_all()


def county_already_processed(county_dir: Path) -> bool:
    """A county counts as done once it has its county file or any h3 output.

    (Some counties have no native census county row, e.g. Connecticut's
    planning regions, so `county.geoparquet` alone can't be the marker.)
    """
    return (county_dir / "county.geoparquet").is_file() or any(county_dir.glob("h3_res*.geoparquet"))


def rebuild_county_h3(county_gdf: gpd.GeoDataFrame, county_dir: Path, census_cache_dir: Optional[Path] = None) -> bool:
    """Rebuild a finished county's h3 files under the current scheme, without recomputing LOS.

    Uses the county's saved scored street edges (`street.geoparquet`, LOS > 0
    only) for the cell means and the cached county street network
    (`_network_cache`) for street presence -- the same inputs a fresh run
    uses, so the result is identical to a full recomputation.

    Returns:
        True if rebuilt, False if already current / the county has no output.
    """
    if (county_dir / H3_SCHEME_MARKER).is_file() or not county_already_processed(county_dir):
        return False
    geoid = county_dir.name
    cache_dir = str(census_cache_dir) if census_cache_dir is not None else None
    if cache_dir is None:
        from pycensus.cache import default_cache_dir

        cache_dir = default_cache_dir()
    centroid_geom = county_gdf.to_crs(4326).union_all()
    _, election_row = _county_election_row(county_gdf, geoid, geoid[:2], cache_dir, centroid_geom)

    edges_wkb = None
    street_path = county_dir / "street.geoparquet"
    if street_path.is_file():
        edges_wkb = _to_wkb_polars(gpd.read_parquet(street_path).to_crs(4326))
    street_geoms = None
    net_edges = county_dir / "_network_cache_b1500" / "edges.parquet"
    if net_edges.is_file():
        import polars as pl

        crs = (county_dir / "_network_cache_b1500" / "meta.txt").read_text().strip()
        street_geoms = gpd.GeoSeries(
            shapely.from_wkb(pl.read_parquet(net_edges, columns=["geometry_wkb"])["geometry_wkb"].to_numpy()), crs=crs
        ).to_crs(4326)
    for old in county_dir.glob("h3_res*.geoparquet"):
        old.unlink()
    _write_h3_levels(county_gdf, geoid, edges_wkb, county_dir, cache_dir, election_row, centroid_geom, street_geoms=street_geoms)
    return True


def _los_region_4326(edges_gdf):
    """Union (EPSG:4326) of the buffered street edges with LOS > 0, or None.

    `edges_gdf` is the polars WKB edge table (EPSG:4326) passed through the pipeline.
    """
    if edges_gdf is None or edges_gdf.height == 0:
        return None
    scored = edges_gdf.filter(edges_gdf["level_of_service"] > 0)
    if scored.height == 0:
        return None
    geoms = gpd.GeoSeries(shapely.from_wkb(scored["geometry"].to_numpy()), crs=4326)
    utm = geoms.estimate_utm_crs()
    region = geoms.to_crs(utm).buffer(H3_LOS_REGION_BUFFER_M).union_all()
    return gpd.GeoSeries([region], crs=utm).to_crs(4326).iloc[0]


def _scored_edge_cell_means(edges_wkb, resolution: int, step_m: float = 10.0) -> pd.Series:
    """Mean LOS of the scored street edges in each h3 cell, from points sampled every `step_m` along every edge.

    `geohierarchy.edges_to_level` length-weights edge overlaps but drops cells where an edge only grazes a small part of
    its length -- at res 11 (~30 m cells, edges of 100+ m) about a third of the cells that really touch a scored street
    came out as 0. Sampling points along the (densified) lines and binning them into cells is exact for such small
    cells and is length-weighted by construction.
    """
    from pyproj import Transformer

    if edges_wkb is None or edges_wkb.height == 0:
        return pd.Series(dtype=float)
    geoms = shapely.from_wkb(edges_wkb["geometry"].to_numpy())
    los = edges_wkb["level_of_service"].to_numpy().astype(float)
    series = gpd.GeoSeries(geoms, crs=4326)
    utm = series.estimate_utm_crs()
    dense = shapely.segmentize(series.to_crs(utm).values, step_m)
    xy, idx = shapely.get_coordinates(dense, return_index=True)
    lon, lat = Transformer.from_crs(utm, 4326, always_xy=True).transform(xy[:, 0], xy[:, 1])
    cells = [h3.latlng_to_cell(la, lo, resolution) for la, lo in zip(lat, lon)]
    return pd.DataFrame({"h3": cells, "los": los[idx]}).groupby("h3")["los"].mean()


def _join_h3_level(
    county_gdf: gpd.GeoDataFrame,
    county_geoid: str,
    edges_gdf,
    block_gdf: Optional[gpd.GeoDataFrame],
    resolution: int,
    county_dir: Path,
    county_election_row: Optional[pd.Series] = None,
    dense_region=None,
    res11: Optional[pd.DataFrame] = None,
    street_geoms=None,
):
    """Build one h3 resolution for a county, join LOS + population, save.

    Res 11 (`H3_FINEST_RESOLUTION`): cells over (buffered LOS > 0 streets) union
    (dense tracts); LOS is the mean of the street edges' LOS in each cell.
    Coarser: cells covering res 11's footprint (res 9) or the whole county
    (res 7); LOS is the population-weighted mean of the res 11 children,
    with population taken from the blocks (so people outside res 11, who have
    LOS 0, still count in the denominator).

    Returns:
        `(path, DataFrame)` or None if nothing to write.
    """
    county_geom = county_gdf.to_crs(4326).union_all()

    if resolution == H3_FINEST_RESOLUTION:
        parts = [g for g in (dense_region, _los_region_4326(edges_gdf)) if g is not None]
        if not parts:
            return None
        region_gdf = gpd.GeoDataFrame(geometry=[shapely.union_all(parts)], crs=4326)
        grid = h3_cells(region_gdf, resolution)
    elif resolution == 9:
        if res11 is None or res11.empty:
            return None
        parents = sorted({h3.cell_to_parent(c, resolution) for c in res11["h3"]})
        grid = h3_cells_from_ids(parents)
    else:  # res 7: whole county (cells without population are dropped below)
        grid = h3_cells(county_gdf, resolution)

    grid = grid[grid.geometry.centroid.within(county_geom)].reset_index(drop=True)
    if grid.empty:
        return None

    if resolution == H3_FINEST_RESOLUTION:
        # Mean of the SCORED street edges (LOS > 0) in the cell; NaN where none touch it.
        grid["level_of_service"] = grid["h3"].map(_scored_edge_cell_means(edges_gdf, resolution))
        # No scored edge: LOS 0 if the cell touches any street, else no score (None, ignored when resampling).
        # `street_geoms=None` means street presence is unknown (no network was loaded): assume streets.
        unscored = grid["level_of_service"].isna()
        if unscored.any():
            if street_geoms is None:
                grid.loc[unscored, "level_of_service"] = 0.0
            else:
                cand = grid.loc[unscored, ["h3", "geometry"]]
                hit = gpd.sjoin(
                    cand.set_crs(4326, allow_override=True),
                    gpd.GeoDataFrame(geometry=street_geoms.values, crs=4326),
                    how="inner", predicate="intersects",
                )["h3"].unique()
                grid.loc[unscored & grid["h3"].isin(hit), "level_of_service"] = 0.0

    # Real decennial block population is the finest native source
    # available (see [[usa_study_native_level_rule]]) -- area-weighted
    # onto h3 cells since blocks and h3 cells don't nest.
    grid = _add_area_weighted(grid, block_gdf, ["population"], id_col="h3")

    if resolution != H3_FINEST_RESOLUTION:
        grid["level_of_service"] = _resample_los_from_res11(grid, res11, resolution)

    if "population" in grid.columns:
        grid = _disaggregate_elections_onto(grid, county_election_row, pop_col="population")

    activity_col = "population" if "population" in grid.columns else None
    if activity_col is None or grid[activity_col].fillna(0).sum() == 0:
        return None
    grid = grid[pd.to_numeric(grid[activity_col], errors="coerce").fillna(0) > 0]
    if grid.empty:
        return None

    # h3 cells save cell id only, no geometry (prompt.txt: "For h3 no need
    # to save geometry if not needed only save the cell id").
    out = grid.drop(columns="geometry")
    path = county_dir / f"h3_res{resolution}.geoparquet"
    out.to_parquet(path)
    return str(path), out


def _resample_los_from_res11(grid: pd.DataFrame, res11: Optional[pd.DataFrame], resolution: int) -> pd.Series:
    """Population-weighted LOS of each coarser cell from its res 11 children.

    Children with no score (no streets) are ignored: their population is removed from the
    denominator. People of the coarse cell outside any res 11 cell count as LOS 0.
    A cell whose entire population is unscored gets no score (NaN).
    """
    if res11 is None or res11.empty:
        return pd.Series(0.0, index=grid.index)
    kids = res11[["h3", "level_of_service", "population"]].copy()
    kids["_parent"] = [h3.cell_to_parent(c, resolution) for c in kids["h3"]]
    pop_k = pd.to_numeric(kids["population"], errors="coerce").fillna(0)
    scored = kids["level_of_service"].notna()
    numer = (pop_k * kids["level_of_service"]).where(scored, 0.0).groupby(kids["_parent"]).sum()
    unscored_pop = pop_k.where(~scored, 0.0).groupby(kids["_parent"]).sum()
    pop = pd.to_numeric(grid["population"], errors="coerce").fillna(0)
    denom = (pop - grid["h3"].map(unscored_pop).fillna(0.0)).clip(lower=0)
    los = grid["h3"].map(numer).fillna(0.0) / denom.where(denom > 0)
    los = los.clip(upper=float(kids["level_of_service"].max()))
    # all of the cell's population unscored -> no score; no children at all -> 0
    has_kids = grid["h3"].isin(numer.index)
    los = los.where(~(has_kids & ~(denom > 0)), np.nan)
    return los.where(has_kids, 0.0)


def _add_area_weighted(
    grid: gpd.GeoDataFrame, source_gdf: Optional[gpd.GeoDataFrame], columns: list, id_col: str
) -> gpd.GeoDataFrame:
    """Area-weighted apportionment of `columns` from `source_gdf` onto `grid`'s polygons.

    Each source polygon's value is split across every grid cell it
    intersects, proportional to the intersection's share of the source
    polygon's own area (mass-conserving: the sum across all grid cells a
    source polygon touches reproduces that polygon's original value).
    """
    if source_gdf is None or source_gdf.empty:
        for col in columns:
            grid[col] = 0.0
        return grid

    available = [c for c in columns if c in source_gdf.columns]
    if not available:
        for col in columns:
            grid[col] = 0.0
        return grid

    src = source_gdf.to_crs(grid.crs) if source_gdf.crs != grid.crs else source_gdf
    src = src[["GEOID", "geometry", *available]].copy()
    src["_src_area"] = src.geometry.area

    joined = gpd.overlay(grid[[id_col, "geometry"]], src, how="intersection")
    joined["_isect_area"] = joined.geometry.area
    joined["_share"] = joined["_isect_area"] / joined["_src_area"].replace(0, pd.NA)

    for col in available:
        joined[col] = pd.to_numeric(joined[col], errors="coerce") * joined["_share"]

    agg = joined.groupby(id_col)[available].sum(numeric_only=True)
    grid = grid.merge(agg, on=id_col, how="left")
    for col in available:
        grid[col] = grid[col].fillna(0.0)
    return grid

