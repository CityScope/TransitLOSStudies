# -*- coding: utf-8 -*-
"""usa_study orchestrator -- run the whole pipeline for one, several, or all US states.

Stages, per `prompt.txt` (see `code/gtfs_stage.py`, `code/county_pipeline.py`,
`code/state_pipeline.py`, `code/build_state_map.py`, and
`pycensus.countries.usa.elections` for each stage's own module docstring --
this file just wires them together):

1. Download that state's NTD GTFS feeds (`pyGTFSHandler.downloaders.usa
   .NTDDownloader`), check service windows, score stops
   (`gtfs_stage.score_state_stops`) -> `stops.geoparquet`.
2. Per county: street network + LOS + census/h3 (`county_pipeline
   .process_county`) -> `counties/<GEOID>/*.geoparquet`.
3. State rollup: `county.geoparquet`, coarser h3 resolutions, state
   census districts (`state_pipeline.merge_counties`,
   `.merge_and_resample_h3`, `.add_state_districts`).
4. Map: `code.build_state_map.build_state_map` -> `<state_dir>/map.html`
   (reuses `transitlos.map.build.build_city_map` directly -- see that
   module's own docstring for the one documented UI simplification: place/
   congressional/state-legislative/school-district levels share the same
   one census dropdown as the real admin hierarchy, not a separate
   selector).

Elections (`pycensus.countries.usa.elections`) are joined onto the county
level only for now -- see that module's own docstring for exactly what
years/offices are implemented and why the rest is a documented gap.

Usage:
    python main.py MA
    python main.py MA RI CT           # several states, one process each
    python main.py ALL                # every US state + DC
    python main.py MA --force         # redo even if state_dir already has output
    python main.py MA --skip-map      # stages 1-3 only, no map.html
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyGTFSHandler"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyCensus" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "transitLOS" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "UrbanAccessAnalyzer"))

from code.gtfs_stage import DEFAULT_WINDOW, score_state_stops
from code.county_pipeline import get_state_pbf, process_county
from code.state_pipeline import add_state_districts, add_state_native_elections, merge_and_resample_h3, merge_counties
from pyGTFSHandler.downloaders.usa.ntd import NTDDownloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("usa_study.main")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OSM_FILES_DIR = BASE_DIR / "osm_files"
PYCENSUS_CACHE_DIR = DATA_DIR / "pycensus_cache"


def sanitize_state_name(state_name: str) -> str:
    """State display name -> folder-safe name (no spaces/special chars, per prompt.txt)."""
    return "".join(c for c in state_name if c.isalnum())


def run_state(state_abbr: str, force: bool = False, skip_map: bool = False) -> None:
    """Run the whole pipeline for one state."""
    from pycensus.countries.usa import geography

    state_gdf = geography.load_boundaries("state", state_abbr, 2023)
    state_fips = state_gdf["GEOID"].iloc[0] if "GEOID" in state_gdf.columns else None
    import us

    state_info = us.states.lookup(state_abbr)
    state_name = sanitize_state_name(state_info.name)
    state_dir = DATA_DIR / state_name
    gtfs_dir = state_dir / "gtfs"
    counties_dir = state_dir / "counties"
    counties_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== {state_info.name} ({state_abbr}) -- state FIPS {state_fips} ===")

    # --- Stage 0: download + validate NTD feeds ---
    stops_path = state_dir / "stops.geoparquet"
    if force or not stops_path.is_file():
        logger.info("Stage 1: downloading NTD feeds...")
        downloader = NTDDownloader()
        feeds = downloader.search_feeds(state=state_abbr)
        working, broken = downloader.check_feed_links(feeds)
        if broken:
            logger.warning(f"{len(broken)} NTD feed link(s) broken for {state_abbr}: "
                            f"{[f.raw['ntd'].get('agency_name') for f in broken]}")
        gtfs_dir.mkdir(parents=True, exist_ok=True)
        paths = downloader.download_feeds(working, str(gtfs_dir))
        gtfs_dirs = [Path(p) for p in paths]

        ntd_rows_by_dir = {}
        for feed, path in zip(working, paths):
            ntd_rows_by_dir.setdefault(Path(path), []).append(feed.raw["ntd"])

        logger.info(f"Stage 1: scoring stops for {len(gtfs_dirs)} feeds...")
        stops_gdf = score_state_stops(gtfs_dirs, ntd_rows_by_dir, state_gdf, window=DEFAULT_WINDOW)
        stops_gdf.to_parquet(stops_path)
        logger.info(f"Stage 1 done: {len(stops_gdf)} stops -> {stops_path}")
    else:
        logger.info(f"Stage 1: reusing existing {stops_path}")
        stops_gdf = gpd.read_parquet(stops_path)

    # --- Stage 2: per-county isochrones/LOS/census ---
    logger.info("Stage 2: downloading state OSM extract...")
    pbf_path = get_state_pbf(state_gdf, OSM_FILES_DIR)

    from pycensus.countries.usa import dhc

    counties_gdf = dhc.load(aoi=state_gdf, states=[state_abbr], level="county", cache_dir=str(PYCENSUS_CACHE_DIR))
    logger.info(f"Stage 2: processing {len(counties_gdf)} counties...")

    for _, county_row in counties_gdf.iterrows():
        county_geoid = county_row["GEOID"]
        county_folder = counties_dir / county_geoid
        if not force and (county_folder / "county.geoparquet").is_file():
            logger.info(f"County {county_geoid}: already processed, skipping.")
            continue

        county_gdf_single = gpd.GeoDataFrame([county_row], geometry="geometry", crs=counties_gdf.crs)
        try:
            result = process_county(
                county_gdf_single,
                county_geoid,
                pbf_path,
                stops_gdf,
                county_folder,
                network_cache_dir=county_folder / "_network_cache",
                census_cache_dir=PYCENSUS_CACHE_DIR,
            )
            logger.info(f"County {county_geoid}: {result}")
        except Exception:
            logger.error(f"County {county_geoid} FAILED, skipping:\n{traceback.format_exc()}")

    # --- Stage 3: state rollup ---
    logger.info("Stage 3: state rollup...")
    try:
        merge_counties(state_dir)
    except Exception:
        logger.error(f"State county rollup failed:\n{traceback.format_exc()}")
    try:
        merge_and_resample_h3(state_dir)
    except Exception:
        logger.error(f"State h3 rollup failed:\n{traceback.format_exc()}")
    resolved_state_fips = state_fips[:2] if state_fips else us.states.lookup(state_abbr).fips
    try:
        add_state_districts(state_dir, resolved_state_fips)
    except Exception:
        logger.error(f"State districts failed:\n{traceback.format_exc()}")
    try:
        add_state_native_elections(state_dir, resolved_state_fips, state_abbr, cache_dir=PYCENSUS_CACHE_DIR)
    except Exception:
        logger.error(f"State-native elections failed:\n{traceback.format_exc()}")

    # --- Stage 4: map ---
    if not skip_map:
        logger.info("Stage 4: building map.html...")
        try:
            from code.build_state_map import build_state_map

            map_path = build_state_map(state_dir)
            logger.info(f"Stage 4 done: {map_path}")
        except Exception:
            logger.error(f"Map build failed:\n{traceback.format_exc()}")

    logger.info(f"=== {state_info.name} done -- output under {state_dir} ===")


#: Every US state + DC (per prompt.txt: "Then launch it only for
#: massachussets state" implies the full run is over this complete
#: roster -- `python main.py ALL` expands to exactly this list).
ALL_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
    "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
]


def main():
    parser = argparse.ArgumentParser(description="Run the usa_study pipeline for one or more states.")
    parser.add_argument(
        "states", nargs="+", help="State abbreviation(s), e.g. MA, or ALL for every US state + DC."
    )
    parser.add_argument("--force", action="store_true", help="Redo stages even if output already exists.")
    parser.add_argument("--skip-map", action="store_true", help="Skip stage 4 (map.html build).")
    args = parser.parse_args()

    if len(args.states) == 1 and args.states[0].upper() == "ALL":
        states = ALL_STATES
    else:
        states = args.states

    logger.info(f"Running usa_study for {len(states)} state(s): {states}")
    failed = []
    for state_abbr in states:
        try:
            run_state(state_abbr, force=args.force, skip_map=args.skip_map)
        except Exception:
            # One state's hard failure (e.g. a bad NTD/census API response
            # that isn't already caught inside run_state's own per-stage
            # try/excepts) should never take down a multi-state/ALL run --
            # "fail safely, skip", same convention every stage inside
            # run_state already follows per-county.
            logger.error(f"State {state_abbr} FAILED entirely, skipping:\n{traceback.format_exc()}")
            failed.append(state_abbr)

    if failed:
        logger.warning(f"{len(failed)} state(s) failed entirely: {failed}")
    logger.info("usa_study run complete.")


if __name__ == "__main__":
    main()
