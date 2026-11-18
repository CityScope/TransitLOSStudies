# -*- coding: utf-8 -*-
"""spain_study orchestrator -- the usa_study pipeline for Spain (52 provinces, 19 autonomous communities).

Stages (units: PROVINCE = usa county, AUTONOMOUS COMMUNITY = usa state):
1. GTFS from the NAP (``nap_fetch.py``; every feed lives in the ONE common folder ``data/gtfs``, ``code/gtfs_store.py``): the feeds touching a province are extracted and scored
   (``code.gtfs_stage``) -> ``<ccaa>/provinces/<NN>/stops.geoparquet``.
2. Per province: street network + level of service + census sections (``code.census_spain``, ``ine_`` columns) + H3 res 11..7
   (``code.province_pipeline``).
3. Community roll-up: province / municipality / section / ccaa levels, coarser H3 3-5 (``code.ccaa_pipeline``), merged stops.
4. Map of the community: ``code.build_region_map`` -> ``data/<Community>/map.html`` (bilingual ES/EN, ``code.i18n_inject``).
5. Nationwide map with lazy stats chunks (``build_spain_map.py``), ALL runs or ``--spain-map-only``.

    python main.py 20                  # Gipuzkoa (province code), then its community's roll-up and map
    python main.py 20 48 01            # several provinces (grouped by community)
    python main.py ccaa:16             # every province of the Basque Country (INE community code)
    python main.py ALL                 # all 52 provinces + nationwide map (heavy: do not run casually)
    python main.py 20 --force --skip-map --refresh-stops
    python main.py --spain-map-only
"""
from __future__ import annotations

import os

os.environ.setdefault("GEOHIERARCHY_MAX_NATIVE_TILE_ZOOM", "14")
os.environ.setdefault("GTFS_STOP_GROUPING", "complete")

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

import geopandas as gpd
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent.parent          # .../PTLOS
sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(DATA_DIR / "spain_study.log")])
logger = logging.getLogger("spain_study.main")

from code.study_cache import apply_env, resolve_cache_dir  # noqa: E402

from code.study_cache import osm_dir, pycensus_dir, spain_raw_dir  # noqa: E402

CACHE_DIR = resolve_cache_dir()  # shared TransitLOSStudies/cache (default); --cache-dir / STUDY_CACHE_DIR override
SPAIN_DATA_DIR = str(spain_raw_dir(CACHE_DIR))       # raw census downloads (pycensus data_dir)
PYCENSUS_CACHE_DIR = str(pycensus_dir(CACHE_DIR))
apply_env(CACHE_DIR)
from code.gtfs_store import default_gtfs_dir  # noqa: E402

GTFS_DIR = default_gtfs_dir(DATA_DIR)          # ONE common folder of GTFS zips for all of Spain (code/gtfs_store.py)
OSM_FILES_DIR = osm_dir(CACHE_DIR)
PROVINCE_RETRY_WAITS_S = (60, 300)


def _constants():
    from pycensus.countries.spain import constants

    return constants


def ccaa_of_province(code: str) -> str:
    for ccaa, provs in _constants().AUTONOMOUS_COMMUNITY_PROVINCES.items():
        if code in provs:
            return ccaa
    raise KeyError(code)


def sanitize(name: str) -> str:
    import unicodedata

    base = unicodedata.normalize("NFKD", name.split("(")[0]).encode("ascii", "ignore").decode()
    return "".join(c for c in base if c.isalnum())


def ccaa_dir_of(ccaa: str) -> Path:
    return DATA_DIR / sanitize(_constants().AUTONOMOUS_COMMUNITY_CODES[ccaa])


def all_provinces():
    return sorted(_constants().PROVINCE_CODES)


def expand(tokens):
    c = _constants()
    out = []
    for t in tokens:
        if t.upper() == "ALL":
            out += all_provinces()
        elif t.lower().startswith("ccaa:"):
            out += c.AUTONOMOUS_COMMUNITY_PROVINCES[t.split(":")[1].zfill(2)]
        else:
            out.append(t.zfill(2))
    seen = []
    for p in out:
        if p not in c.PROVINCE_CODES:
            raise SystemExit(f"unknown province code {p}")
        if p not in seen:
            seen.append(p)
    return seen


def run_province(code: str, force: bool = False, refresh_stops: bool = False) -> None:
    from code import census_spain
    from code.gtfs_store import province_feed_dirs, write_province_feeds
    from code.gtfs_stage import score_province_stops
    from code.province_pipeline import (NETWORK_CACHE_NAME, STOPS_SIGNATURE_FILE, get_pbf, process_province, province_already_processed,
                                        province_stops_signature)

    ccaa = ccaa_of_province(code)
    cdir = ccaa_dir_of(ccaa)
    pdir = cdir / "provinces" / code
    pdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== province {code} {_constants().PROVINCE_CODES[code]} (community {ccaa} -> {cdir.name}) ===")

    raw = pdir / "_sections_raw.parquet"
    if raw.is_file() and not force:
        sections = gpd.read_parquet(raw)
    else:
        sections = census_spain.load_province_sections(code, SPAIN_DATA_DIR, PYCENSUS_CACHE_DIR)
        sections.to_parquet(raw)
    prov_gdf = gpd.GeoDataFrame({"GEOID": [code]}, geometry=[sections.to_crs(4326).union_all().buffer(0)], crs=4326)

    # --- stage 1 ---
    stops_path = pdir / "stops.geoparquet"
    if force or refresh_stops or not stops_path.is_file():
        logger.info("Stage 1: NAP feeds -> scored stops")
        dirs = province_feed_dirs(prov_gdf, GTFS_DIR)
        write_province_feeds(pdir, dirs)
        sink: list = []
        stops = score_province_stops(dirs, prov_gdf, stop_table_sink=sink) if dirs else gpd.GeoDataFrame(
            {"stop_lat": [], "stop_lon": [], "stop_score": []}, geometry=[], crs=4326)
        if sink:
            pd.concat(sink, ignore_index=True).to_parquet(pdir / "stops_by_id.parquet")
        stops.to_parquet(stops_path)
        logger.info(f"Stage 1 done: {len(stops)} stops")
    else:
        stops = gpd.read_parquet(stops_path)

    # --- stage 2 ---
    if not force and province_already_processed(pdir):
        sig = pdir / STOPS_SIGNATURE_FILE
        if sig.is_file() and len(stops) and sig.read_text().strip() != province_stops_signature(prov_gdf, stops):
            logger.info("stops around the province changed -- recomputing")
            force = True
        else:
            logger.info("Stage 2: already processed, skipping")
            return
    logger.info("Stage 2: street network, LOS, census, H3")
    pbf = get_pbf(prov_gdf, OSM_FILES_DIR)
    res = process_province(sections, code, pbf, stops, pdir, network_cache_dir=pdir / NETWORK_CACHE_NAME, data_dir=SPAIN_DATA_DIR)
    logger.info(f"Province {code}: {res['n_stops']} stops, {res['n_edges_scored']} scored street edges, wrote {sorted(res['written'])}")


def run_ccaa(ccaa: str, provinces, force=False, refresh_stops=False, skip_map=False) -> None:
    from code.ccaa_pipeline import merge_and_resample_h3, merge_levels
    import code.gtfs_stage as gs

    cdir = ccaa_dir_of(ccaa)
    failed = []
    for p in provinces:
        try:
            run_province(p, force, refresh_stops)
        except Exception:
            logger.error(f"Province {p} FAILED:\n{traceback.format_exc()}")
            failed.append(p)
    for wait in PROVINCE_RETRY_WAITS_S:
        if not failed:
            break
        logger.warning(f"retrying {failed} in {wait}s")
        time.sleep(wait)
        failed = [p for p in failed if not _try(run_province, p, force, refresh_stops)]
    logger.info("Stage 3: community roll-up")
    names = {p: n for p, n in _constants().PROVINCE_CODES.items()}
    names["ccaa"] = _constants().AUTONOMOUS_COMMUNITY_CODES[ccaa]
    try:
        merge_levels(cdir, names)
        merge_and_resample_h3(cdir)
        _merge_stops(cdir)
    except Exception:
        logger.error(f"Roll-up failed:\n{traceback.format_exc()}")
        return
    if not skip_map:
        logger.info("Stage 4: building map.html")
        try:
            from code.build_region_map import build_region_map

            logger.info(f"Stage 4 done: {build_region_map(cdir)}")
        except Exception:
            logger.error(f"Map build failed:\n{traceback.format_exc()}")


def _try(fn, *a) -> bool:
    try:
        fn(*a)
        return True
    except Exception:
        logger.error(traceback.format_exc())
        return False


def _merge_stops(cdir: Path) -> None:
    from code.gtfs_stage import merge_same_location_stops

    fr = [gpd.read_parquet(p / "stops.geoparquet") for p in sorted((cdir / "provinces").glob("*")) if (p / "stops.geoparquet").is_file()]
    fr = [f for f in fr if len(f)]
    if fr:
        merge_same_location_stops(gpd.GeoDataFrame(pd.concat(fr, ignore_index=True), crs=4326)).to_parquet(cdir / "stops.geoparquet")
    by = [pd.read_parquet(p / "stops_by_id.parquet") for p in sorted((cdir / "provinces").glob("*")) if (p / "stops_by_id.parquet").is_file()]
    if by:
        t = pd.concat(by, ignore_index=True)
        key = [c for c in ("stop_id", "stop_lat", "stop_lon") if c in t.columns]
        t.drop_duplicates(key).to_parquet(cdir / "stops_by_id.parquet")


def write_ptlos_summary() -> None:
    """``data/ptlos_by_province.csv``: population-weighted mean LOS and served share per finished province."""
    rows = []
    for f in sorted(DATA_DIR.glob("*/provinces/*/h3_res8.geoparquet")):
        d = pd.read_parquet(f, columns=["population", "level_of_service"])
        pop = d["population"].fillna(0)
        if pop.sum() > 0:
            rows.append({"province": f.parent.name, "population": pop.sum(), "ptlos": float((pop * d["level_of_service"]).sum() / pop.sum()),
                         "served_share": float(pop[d["level_of_service"] > 0].sum() / pop.sum())})
    if rows:
        pd.DataFrame(rows).to_csv(DATA_DIR / "ptlos_by_province.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("units", nargs="*", help="province codes (e.g. 20), ccaa:<code> (e.g. ccaa:16), or ALL")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-map", action="store_true")
    ap.add_argument("--refresh-stops", action="store_true", help="redo stage 1 (feeds + scoring); provinces recompute only if the stops changed")
    ap.add_argument("--no-streets", action="store_true", help="leave street edges out of the nationwide map")
    ap.add_argument("--spain-map-only", action="store_true")
    ap.add_argument("--community-maps", action="store_true", help="with ALL: also build every community's own map.html")
    ap.add_argument("--cache-dir", help="shared study cache folder (default TransitLOSStudies/cache; also STUDY_CACHE_DIR)")
    ap.add_argument("--gtfs-dir", help="common folder with all GTFS zips (default data/gtfs or env SPAIN_GTFS_DIR)")
    a = ap.parse_args()
    if a.gtfs_dir:
        global GTFS_DIR
        GTFS_DIR = Path(a.gtfs_dir)
        os.environ["SPAIN_GTFS_DIR"] = str(GTFS_DIR)
    is_all = len(a.units) == 1 and a.units[0].upper() == "ALL"
    if not a.units and not a.spain_map_only:
        ap.error("give province codes, ccaa:<code>, ALL, or --spain-map-only")
    if not a.spain_map_only:
        provs = expand(a.units)
        groups = {}
        for p in provs:
            groups.setdefault(ccaa_of_province(p), []).append(p)
        for ccaa, ps in groups.items():
            run_ccaa(ccaa, ps, a.force, a.refresh_stops, skip_map=a.skip_map or (is_all and not a.community_maps))
    if (is_all or a.spain_map_only) and not a.skip_map:
        try:
            from build_spain_map import build_spain_map

            logger.info("Stage 5: nationwide map")
            build_spain_map(include_streets=not a.no_streets)
        except Exception:
            logger.error(f"Nationwide map failed:\n{traceback.format_exc()}")
    write_ptlos_summary()
    logger.info("spain_study run complete.")


if __name__ == "__main__":
    main()
