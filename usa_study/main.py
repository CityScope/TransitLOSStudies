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
    python main.py ALL                # whole experiment: every state, then the nationwide map (stage 5)
    # every run ends by writing data/ptlos_by_state.csv, data/ptlos_usa.json (the PTLOS score) and data/run_manifest.json
    python main.py --usa-map-only     # only rebuild the nationwide map from existing state outputs
    (see README.md for all flags and the final experiment design)
"""

from __future__ import annotations

import os

# Continent-scale maps: generate vector tiles natively only down to z14 (MapLibre overzooms past it).
os.environ.setdefault("GEOHIERARCHY_MAX_NATIVE_TILE_ZOOM", "14")
# Stop grouping: strict maximum group diameter (no chaining), see code/gtfs_stage.py STOP_GROUP_DISTANCE_M.
os.environ.setdefault("GTFS_STOP_GROUPING", "complete")

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyGTFSHandler"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pyCensus" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "transitLOS" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "UrbanAccessAnalyzer"))

from code.gtfs_stage import DEFAULT_WINDOW, score_state_stops
from code.county_pipeline import STOPS_SIGNATURE_FILE, county_already_processed, county_stops_signature, get_state_pbf, rebuild_county_h3, process_county
from code.state_pipeline import add_state_districts, add_state_native_elections, merge_and_resample_h3, merge_counties
from pyGTFSHandler.downloaders.usa.ntd import NTDDownloader

# One single log file for the whole experiment (appended across runs), next to the console output.
(Path(__file__).parent / "data").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(Path(__file__).parent / "data" / "usa_study.log")],
)
logger = logging.getLogger("usa_study.main")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OSM_FILES_DIR = BASE_DIR / "osm_files"
PYCENSUS_CACHE_DIR = DATA_DIR / "pycensus_cache"


def sanitize_state_name(state_name: str) -> str:
    """State display name -> folder-safe name (no spaces/special chars, per prompt.txt)."""
    return "".join(c for c in state_name if c.isalnum())


def run_state(state_abbr: str, force: bool = False, skip_map: bool = False, refresh_stops: bool = False) -> None:
    """Run the whole pipeline for one state.

    `refresh_stops`: redo stage 1 (feed download + stop scoring) even though stops exist, e.g. after
    improving the downloader; the county outputs are only invalidated if the stops actually changed.
    """
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
    old_stops = None
    if refresh_stops and stops_path.is_file():
        old_stops = gpd.read_parquet(stops_path)
    if force or refresh_stops or not stops_path.is_file():
        logger.info("Stage 1: downloading NTD feeds...")
        from code.feed_stage import acquire_state_feeds

        gtfs_dirs, ntd_rows_by_dir = acquire_state_feeds(
            state_abbr, state_info.name, state_gdf, state_dir, gtfs_dir, DATA_DIR
        )

        logger.info(f"Stage 1: scoring stops for {len(gtfs_dirs)} feeds...")
        stops_gdf = score_state_stops(gtfs_dirs, ntd_rows_by_dir, state_gdf, window=DEFAULT_WINDOW)
        def _signature(g):
            # Compared by what the stops ARE (location + score), not by their station names: those are not
            # stable between runs, which used to make every refresh look like a change and wipe finished counties.
            return sorted(zip(g["stop_lat"].round(6), g["stop_lon"].round(6), g["stop_score"].round(4)))

        stops_gdf.to_parquet(stops_path)
        # No state-wide wipe: each county keeps a fingerprint of the stops around it (`stops_signature.txt`) and is
        # recomputed below only if that fingerprint changed (its `_network_cache_b1500` is reused).
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

    def _process_one(county_row) -> bool:
        """Process (or skip / h3-rebuild) one county; True on success, False if it failed."""
        county_geoid = county_row["GEOID"]
        county_folder = counties_dir / county_geoid
        if not force and county_already_processed(county_folder):
            sig_file = county_folder / STOPS_SIGNATURE_FILE
            current = county_stops_signature(
                gpd.GeoDataFrame([county_row], geometry="geometry", crs=counties_gdf.crs), stops_gdf
            )
            needs_recompute = sig_file.is_file() and sig_file.read_text().strip() != current
            if needs_recompute:
                logger.info(f"County {county_geoid}: stops around it changed -- recomputing.")
                for stale in county_folder.glob("*"):
                    if stale.is_file():
                        stale.unlink()
            elif not sig_file.is_file():
                sig_file.write_text(current)  # first run with fingerprints: adopt the current stops as the baseline
            try:
                if not needs_recompute and rebuild_county_h3(
                    gpd.GeoDataFrame([county_row], geometry="geometry", crs=counties_gdf.crs),
                    county_folder, PYCENSUS_CACHE_DIR,
                ):
                    logger.info(f"County {county_geoid}: h3 rebuilt under current scheme (res 11 from LOS>0/dense tracts, coarser resampled).")
            except Exception:
                logger.error(f"County {county_geoid} h3 migration FAILED:\n{traceback.format_exc()}")
                return False
            if not needs_recompute:
                logger.info(f"County {county_geoid}: already processed, skipping.")
                return True

        county_gdf_single = gpd.GeoDataFrame([county_row], geometry="geometry", crs=counties_gdf.crs)
        try:
            result = process_county(
                county_gdf_single,
                county_geoid,
                pbf_path,
                stops_gdf,
                county_folder,
                network_cache_dir=county_folder / NETWORK_CACHE_NAME,
                census_cache_dir=PYCENSUS_CACHE_DIR,
            )
            logger.info(f"County {county_geoid}: {result}")
            return True
        except Exception:
            logger.error(f"County {county_geoid} FAILED, skipping:\n{traceback.format_exc()}")
            return False

    pending = [row for _, row in counties_gdf.iterrows() if not _process_one(row)]

    # Automatic reruns: temporary failures (Census API busy / "maintenance", dropped connections,
    # bad downloads) usually clear on their own, so retry the failed counties after a pause.
    for attempt, wait_s in enumerate(COUNTY_RETRY_WAITS_S, start=1):
        if not pending:
            break
        logger.warning(
            f"{len(pending)} county(ies) failed; automatic retry {attempt}/{len(COUNTY_RETRY_WAITS_S)} "
            f"in {wait_s}s: {[r['GEOID'] for r in pending]}"
        )
        time.sleep(wait_s)
        pending = [row for row in pending if not _process_one(row)]
    if pending:
        logger.error(f"{len(pending)} county(ies) still failing after retries: {[r['GEOID'] for r in pending]}")

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
#: Pauses (seconds) before each automatic retry of the counties that failed in a state's first pass.
#: Per-county street network cache (county + walking-distance buffer). The name changed with the buffer so older,
#: county-only caches are never reused.
NETWORK_CACHE_NAME = "_network_cache_b1500"
COUNTY_RETRY_WAITS_S = (60, 180, 600)
#: Same for states that failed entirely (retried after the whole pass).
STATE_RETRY_WAITS_S = (300, 900)

ALL_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
    "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
]


def main():
    parser = argparse.ArgumentParser(description="Run the usa_study pipeline for one or more states.")
    parser.add_argument(
        "states", nargs="*", help="State abbreviation(s), e.g. MA, or ALL for every US state + DC."
    )
    parser.add_argument("--force", action="store_true", help="Redo stages even if output already exists.")
    parser.add_argument("--skip-map", action="store_true", help="Skip every map build (state and national).")
    parser.add_argument(
        "--state-maps", action="store_true",
        help="Also build each state's own map.html (default: only for runs that are not ALL; "
        "an ALL run builds the national map instead).",
    )
    parser.add_argument(
        "--refresh-stops", action="store_true",
        help="Redo stage 1 (GTFS download + stop scoring) for the given states; counties are recomputed "
        "only where the stops changed.",
    )
    parser.add_argument("--no-streets", action="store_true", help="Leave the street edges out of the national map.")
    parser.add_argument(
        "--usa-map-only", action="store_true",
        help="Skip the state runs; only (re)build the national map from the existing state outputs.",
    )
    args = parser.parse_args()

    is_all = len(args.states) == 1 and args.states[0].upper() == "ALL"
    if is_all or args.usa_map_only:
        states = ALL_STATES
    else:
        states = args.states
    if not states and not args.usa_map_only:
        parser.error("give at least one state, ALL, or --usa-map-only")

    failed = []
    if not args.usa_map_only:
        logger.info(f"Running usa_study for {len(states)} state(s): {states}")
        state_map = (not args.skip_map) and (args.state_maps or not is_all)
        for state_abbr in states:
            try:
                run_state(state_abbr, force=args.force, skip_map=not state_map, refresh_stops=args.refresh_stops)
            except Exception:
                # One state's hard failure (e.g. a bad NTD/census API response
                # that isn't already caught inside run_state's own per-stage
                # try/excepts) should never take down a multi-state/ALL run --
                # "fail safely, skip", same convention every stage inside
                # run_state already follows per-county.
                logger.error(f"State {state_abbr} FAILED entirely, skipping:\n{traceback.format_exc()}")
                failed.append(state_abbr)

        # Automatic rerun of states that failed entirely (transient network/API errors), after a pause.
        for attempt, wait_s in enumerate(STATE_RETRY_WAITS_S, start=1):
            if not failed:
                break
            logger.warning(f"Retrying {len(failed)} failed state(s) in {wait_s}s ({attempt}/{len(STATE_RETRY_WAITS_S)}): {failed}")
            time.sleep(wait_s)
            still = []
            for state_abbr in failed:
                try:
                    run_state(state_abbr, force=args.force, skip_map=not state_map, refresh_stops=args.refresh_stops)
                except Exception:
                    logger.error(f"State {state_abbr} FAILED again:\n{traceback.format_exc()}")
                    still.append(state_abbr)
            failed = still

    # Stage 5: the nationwide map (ALL runs, or --usa-map-only).
    if (is_all or args.usa_map_only) and not args.skip_map:
        try:
            from build_usa_map import build_usa_map

            logger.info("Stage 5: building the nationwide map (data/_usa/map.html)...")
            build_usa_map(include_streets=not args.no_streets)
        except Exception:
            logger.error(f"National map build failed:\n{traceback.format_exc()}")

    try:
        from code.feed_fallback import write_feed_exceptions_summary

        write_feed_exceptions_summary(DATA_DIR)
    except Exception:
        logger.error(f"Feed exceptions summary failed:\n{traceback.format_exc()}")

    try:
        from code.summary import write_ptlos_summary, write_run_manifest
        from code.gtfs_stage import ANALYSIS_DATE, ANALYSIS_START_TIME, ANALYSIS_END_TIME, STOP_GROUP_DISTANCE_M
        from code.county_pipeline import H3_RESOLUTIONS, H3_DENSE_TRACT_PER_KM2

        usa = write_ptlos_summary(DATA_DIR)
        logger.info(f"USA PTLOS = {usa['ptlos']:.2f} ({usa['served_share']:.1%} of people served; {usa['states']} states, "
                    f"{usa['counties']} counties, {usa['stops']} stops) -> {DATA_DIR / 'ptlos_by_state.csv'}")
        write_run_manifest(DATA_DIR, Path(__file__).parent.parent.parent, {
            "analysis_date": str(ANALYSIS_DATE), "window": [str(ANALYSIS_START_TIME), str(ANALYSIS_END_TIME)],
            "stop_group_distance_m": STOP_GROUP_DISTANCE_M, "h3_resolutions": list(H3_RESOLUTIONS),
            "h3_dense_tract_per_km2": H3_DENSE_TRACT_PER_KM2, "network_cache": NETWORK_CACHE_NAME,
            "env": {k: os.environ.get(k) for k in ("GTFS_STOP_GROUPING", "GEOHIERARCHY_MAX_NATIVE_TILE_ZOOM")},
            "argv": sys.argv[1:],
        })
    except Exception:
        logger.error(f"PTLOS summary failed:\n{traceback.format_exc()}")

    if failed:
        logger.warning(f"{len(failed)} state(s) failed entirely: {failed}")
    logger.info("usa_study run complete.")


if __name__ == "__main__":
    main()
