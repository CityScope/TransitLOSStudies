"""Stage 1a: get every usable GTFS feed of a state on disk (shared by `main.py` and `fetch_missing_feeds.py`).

Order of sources for each NTD agency:
  1. its own NTD weblink (browser User-Agent, retries, SSL / legacy-TLS fallbacks in `robust_get`);
  2. a download that is not static GTFS (GTFS-RT endpoint, web page, empty folder) counts as broken;
  3. broken ones: the Mobility Database catalog (pinned `KNOWN_FEED_OVERRIDES`, then name matching, archived copies);
  4. still missing: Transitland (agency search + archived latest feed version; needs the key in
     `~/.pygtfshandler/api_keys.json` or `TRANSITLAND_API_KEY`).
Every state that had broken feeds gets a `feed_report.json` (agency, reason, resolution).
"""
import logging
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


GTFS_FILES = {
    "agency", "stops", "routes", "trips", "stop_times", "calendar", "calendar_dates", "shapes", "frequencies",
    "transfers", "feed_info", "fare_attributes", "fare_rules", "pathways", "levels", "translations", "attributions",
}
_OPTIONAL_WHEN_EMPTY = {"shapes", "frequencies", "calendar_dates", "transfers", "feed_info", "fare_attributes", "fare_rules"}


def sanitize_feed_folder(folder: Path, dry_run: bool = False) -> List[str]:
    """Fix the packaging quirks that make the GTFS loader skip a feed; returns what was (or would be) changed.

    - the feed lives in one nested sub-folder (zip with a top-level directory, e.g. `google_transit.zip/`): its files
      are moved up to `folder`;
    - file names are normalised to lower-case `.txt` (`Stops.txt`, or `.csv` extensions);
    - empty optional files (a zero-byte `shapes.txt` crashes the reader) are removed.
    """
    import shutil

    changes: List[str] = []
    if not folder.is_dir():
        return changes

    def has_core(d: Path) -> bool:
        names = {p.name.lower() for p in d.iterdir() if p.is_file()}
        return any(n in names for n in ("stops.txt", "stops.csv")) and any(n in names for n in ("trips.txt", "trips.csv"))

    if not has_core(folder):
        # candidates: directories (at any depth) that hold the core files
        nested = [d for d in sorted(folder.rglob("*")) if d.is_dir() and has_core(d)]
        if nested:
            src = nested[0]
            changes.append(f"flatten {src.relative_to(folder)}")
            if not dry_run:
                for item in list(src.iterdir()):
                    target = folder / item.name
                    if not target.exists():
                        shutil.move(str(item), str(target))
    for f in sorted(p for p in folder.iterdir() if p.is_file()):
        stem, ext = f.stem, f.suffix.lower()
        if stem.lower() in GTFS_FILES and ext in (".txt", ".csv"):
            canon = folder / f"{stem.lower()}.txt"
            if f != canon and not canon.exists():
                changes.append(f"rename {f.name} -> {canon.name}")
                if not dry_run:
                    f.rename(canon)
                    f = canon
        if f.exists() and f.stem.lower() in _OPTIONAL_WHEN_EMPTY and f.stat().st_size == 0:
            changes.append(f"remove empty {f.name}")
            if not dry_run:
                f.unlink()
    # header fields padded with spaces (`route_id, route_type` -- Metra) make columns unfindable: strip them
    for f in sorted(p for p in folder.iterdir() if p.is_file() and p.suffix == ".txt" and p.stem in GTFS_FILES):
        try:
            with open(f, "rb") as fh:
                first = fh.readline()
        except OSError:
            continue
        text = first.decode("utf-8-sig", errors="replace").rstrip("\r\n")
        parts = text.split(",")
        if any(x != x.strip() for x in parts):
            changes.append(f"strip header spaces in {f.name}")
            if not dry_run:
                clean = ",".join(x.strip() for x in parts)
                tmp = f.with_suffix(".tmp")
                with open(f, "rb") as src, open(tmp, "wb") as dst:
                    src.readline()
                    dst.write((clean + "\n").encode("utf-8"))
                    shutil.copyfileobj(src, dst, 1 << 20)
                tmp.replace(f)
    return changes


def is_static_gtfs(folder: Path) -> bool:
    """True if the folder (possibly with a nested sub-folder) holds stops/trips/stop_times files."""
    names = {p.name.lower() for p in folder.rglob("*") if p.is_file()}
    return all(any(n in names for n in (f"{base}.txt", f"{base}.csv")) for base in ("stops", "trips", "stop_times"))


#: Systems the NTD links / fallbacks could not deliver (checked 2026-10): official static GTFS zips added to the state's feeds.
#: `state abbreviation -> [(folder name, url)]`. The folder is created under `<state>/gtfs/` once; later runs reuse it.
MANUAL_FEEDS_BY_STATE: Dict[str, List[Tuple[str, str]]] = {
    # MTA New York City Transit subway (28 lines) + Staten Island Railway. The NTD row for it (20099) was matched to the Staten Island
    # Ferry feed, so the whole subway was missing from the map.
    # The NTD row 20008 only delivered the Manhattan bus feed: Staten Island, Brooklyn, Queens and Bronx NYCT buses were missing.
    "NY": [("manual-nyct-subway", "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip"),
           ("manual-nyct-bus-staten-island", "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_si.zip"),
           ("manual-nyct-bus-brooklyn", "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_b.zip"),
           ("manual-nyct-bus-queens", "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_q.zip"),
           ("manual-nyct-bus-bronx", "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_bx.zip")],
    # Sound Transit Link light rail + Sounder: the NTD rows were replaced by bus-only Pierce / Community Transit feeds.
    "WA": [("manual-soundtransit-rail", "https://www.soundtransit.org/GTFS-rail/40_gtfs.zip")],
    # 2026-11 audit (riders without stops per county, ACS bus / rail commuters against the scored stops). Mobility Database copies, no key:
    # WMATA Metrobus: its own link needs an API key; the Mobility Database copy was REJECTED (matched to the Metrorail NTD row, bus-only
    # routes): all of Metrobus was missing. WMATA serves DC, Maryland and Virginia, but each state keeps only the stops inside its own
    # boundary, so the WMATA feeds are added to all three (Metrorail in Montgomery, Prince George's, Arlington, Fairfax... was missing).
    "DC": [("manual-wmata-bus", "https://files.mobilitydatabase.org/mdb-1846/latest.zip")],
    # MDOT MTA: MARC and Light RailLink were rejected (matched to the bus NTD row), Metro SubwayLink was never downloaded.
    "MD": [("manual-wmata-bus", "https://files.mobilitydatabase.org/mdb-1846/latest.zip"),
           ("manual-wmata-rail", "https://files.mobilitydatabase.org/mdb-1847/latest.zip"),
           ("manual-mta-marc", "https://files.mobilitydatabase.org/mdb-468/latest.zip"),
           ("manual-mta-lightrail", "https://files.mobilitydatabase.org/mdb-469/latest.zip"),
           ("manual-mta-metro-subway", "https://files.mobilitydatabase.org/mdb-470/latest.zip")],
    "VA": [("manual-wmata-bus", "https://files.mobilitydatabase.org/mdb-1846/latest.zip"),
           ("manual-wmata-rail", "https://files.mobilitydatabase.org/mdb-1847/latest.zip")],
    # PSTA (Pinellas County, the St. Petersburg half of the Tampa metro): rejected as a match for another NTD row.
    "FL": [("manual-psta", "https://files.mobilitydatabase.org/mdb-326/latest.zip")],
}
# NFTA Metro buses (Buffalo): the kept NFTA feed holds only Metro Rail; the bus feed was rejected.
MANUAL_FEEDS_BY_STATE["NY"].append(("manual-nfta-bus", "https://files.mobilitydatabase.org/tld-401/latest.zip"))
# Local systems with riders but no stops in the 2026-11 audit (no NTD feed delivered, or a broken one: Chatham Area Transit's had 3 stops).
# Feeds without service on the analysis date are scored on their own representative weekday, as every fallback feed.
_MDB = "https://files.mobilitydatabase.org/{}/latest.zip"
for _st, _name, _id in [
    ("GA", "manual-cat-savannah", "mdb-2080"),           # Chatham Area Transit (Savannah)
    ("PA", "manual-rrta-lancaster", "mdb-2000"),         # Red Rose Transit (Lancaster)
    ("PA", "manual-mcta-monroe", "mdb-3460"),            # Monroe County Transportation Authority
    ("OH", "manual-sarta-canton", "mdb-2001"),           # SARTA (Canton)
    ("AL", "manual-wave-mobile", "tld-6773"),            # The Wave (Mobile)
    ("MA", "manual-brta-berkshire", "mdb-537"),          # BRTA / Link413 (Pittsfield)
    ("MA", "manual-wrta-worcester", "mdb-432"),          # WRTA (Worcester)
    ("NY", "manual-beeline-westchester", "mdb-2353"),    # Westchester County Bee-Line
    ("NY", "manual-lhvt", "mdb-915"),                    # Lower Hudson Valley Transit
    ("MD", "manual-sts-st-marys", "mdb-2268"),           # St. Mary's Transit System
]:
    MANUAL_FEEDS_BY_STATE.setdefault(_st, []).append((_name, _MDB.format(_id)))


def neighbour_feed_dirs(state_dir: Path, min_stops: int = 1) -> List[Path]:
    """Feed folders of the NEIGHBOURING states that have stops inside this state.

    Each state's feeds are clipped to its own boundary, and an agency's feed is usually downloaded only by the state NTD lists it under, so
    service across a state line was missing on the other side (2026-11 audit: MBTA commuter rail in Rhode Island, SEPTA in Delaware, NJ
    Transit / Metro-North stations in Rockland and Orange NY, St. Louis MetroLink and MetroBus in Illinois, Metrorail in Maryland and
    Virginia). These folders are loaded with the state's own feeds (and clipped to the state); a copy of a feed the state already has is
    removed by the duplicate check. Cached in `<state>/neighbour_feeds.json` (recomputed when a neighbour's feed list changes)."""
    import json

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely import contains_xy

    state_dir = Path(state_dir); data_dir = state_dir.parent
    me = gpd.read_parquet(state_dir / "state.geoparquet").to_crs(4326).geometry.union_all()
    near = []
    for other in sorted(data_dir.iterdir()):
        if other == state_dir or not (other / "state.geoparquet").is_file() or not (other / "gtfs").is_dir():
            continue
        og = gpd.read_parquet(other / "state.geoparquet").to_crs(4326).geometry.union_all()
        if og.distance(me) < 0.05:   # degrees (~5 km): states that touch (water borders included)
            near.append(other)
    folders = [f for o in near for f in sorted((o / "gtfs").iterdir()) if f.is_dir() and not f.name.startswith("_") and (f / "stops.txt").is_file()]
    key = {str(f): int((f / "stops.txt").stat().st_mtime) for f in folders}
    cache = state_dir / "neighbour_feeds.json"
    if cache.is_file():
        c = json.loads(cache.read_text())
        if c.get("key") == key:
            return [Path(p) for p in c["inside"]]
    minx, miny, maxx, maxy = me.bounds
    inside = []
    for f in folders:
        try:
            st = pd.read_csv(f / "stops.txt", dtype=str, keep_default_na=False, encoding_errors="replace", usecols=lambda c: c.strip().lstrip("﻿") in ("stop_lat", "stop_lon"))
        except Exception:  # noqa: BLE001
            continue
        st.columns = [c.strip().lstrip("﻿") for c in st.columns]
        la = pd.to_numeric(st.get("stop_lat"), errors="coerce").to_numpy(); lo = pd.to_numeric(st.get("stop_lon"), errors="coerce").to_numpy()
        ok = np.isfinite(la) & np.isfinite(lo) & (lo >= minx) & (lo <= maxx) & (la >= miny) & (la <= maxy)
        if ok.sum() >= min_stops and contains_xy(me, lo[ok], la[ok]).sum() >= min_stops:
            inside.append(str(f))
    cache.write_text(json.dumps({"key": key, "inside": inside}, indent=1))
    return [Path(p) for p in inside]


def ensure_manual_feeds(state_abbr: str, gtfs_dir: Path) -> List[Path]:
    """Download (once) and unpack the manual feeds of `state_abbr`; returns their folders."""
    import io
    import zipfile

    import requests

    out: List[Path] = []
    for name, url in MANUAL_FEEDS_BY_STATE.get(state_abbr, []):
        folder = gtfs_dir / name
        if not (folder.is_dir() and is_static_gtfs(folder)):
            try:
                resp = requests.get(url, timeout=180)
                resp.raise_for_status()
                folder.mkdir(parents=True, exist_ok=True)
                zipfile.ZipFile(io.BytesIO(resp.content)).extractall(folder)
                sanitize_feed_folder(folder)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"manual feed {name} ({url}) failed: {exc}")
                continue
        if folder.is_dir() and is_static_gtfs(folder):
            out.append(folder)
    return out


def acquire_state_feeds(
    state_abbr: str, state_name: str, state_gdf, state_dir: Path, gtfs_dir: Path, data_dir: Path
) -> Tuple[List[Path], Dict[Path, List[dict]]]:
    """Download/validate/repair a state's feeds; returns `(gtfs_dirs, ntd_rows_by_dir)` ready for scoring."""
    from pyGTFSHandler.downloaders.usa.ntd import NTDDownloader
    from pyGTFSHandler.downloaders.utils.naming import build_feed_filename

    downloader = NTDDownloader()
    feeds = downloader.search_feeds(state=state_abbr)
    working, broken = downloader.check_feed_links(feeds)
    if broken:
        logger.warning(
            f"{len(broken)} NTD feed link(s) broken for {state_abbr}: "
            f"{[f.raw['ntd'].get('agency_name') for f in broken]}"
        )
    gtfs_dir.mkdir(parents=True, exist_ok=True)
    downloader.download_feeds(working, str(gtfs_dir))
    for folder in sorted(p for p in gtfs_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        fixed = sanitize_feed_folder(folder)
        if fixed:
            logger.info(f"Sanitized feed folder {folder.name}: {fixed}")

    # Map every feed to ITS folder by the downloader's own naming (not by position: a skipped download would shift a
    # positional zip) and validate it: some NTD "weblinks" are not static GTFS at all.
    gtfs_dirs: List[Path] = []
    ntd_rows_by_dir: Dict[Path, List[dict]] = {}
    unusable = []
    from .feed_fallback import REPLACE_NTD_AGENCIES

    for feed in list(working):
        if str(feed.raw["ntd"].get("ntd_id")) in REPLACE_NTD_AGENCIES:
            feed.raw["broken_reason"] = "partial NTD feed (a route or mode only); replaced by the full feed"
            unusable.append(feed)
    working = [f for f in working if f not in unusable]
    for feed in working:
        folder = gtfs_dir / build_feed_filename(feed.id, feed.name or "", feed.provider or "")
        if folder.is_dir() and is_static_gtfs(folder):
            gtfs_dirs.append(folder)
            ntd_rows_by_dir.setdefault(folder, []).append(feed.raw["ntd"])
        else:
            # A download that fails once is often fine on the next try (the MTA subway zip did exactly that): retry before giving up.
            for attempt in range(2):
                try:
                    shutil.rmtree(folder, ignore_errors=True)
                    downloader.download_feeds([feed], str(gtfs_dir))
                    sanitize_feed_folder(folder)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"retry {attempt + 1} of NTD feed {folder.name} failed: {exc}")
                if folder.is_dir() and is_static_gtfs(folder):
                    break
            if folder.is_dir() and is_static_gtfs(folder):
                gtfs_dirs.append(folder)
                ntd_rows_by_dir.setdefault(folder, []).append(feed.raw["ntd"])
                continue
            feed.raw["broken_reason"] = "download is not a static GTFS feed (no stops/trips/stop_times)"
            unusable.append(feed)
    if unusable:
        logger.warning(
            f"{len(unusable)} NTD download(s) for {state_abbr} are not static GTFS: "
            f"{[f.raw['ntd'].get('agency_name') for f in unusable]}"
        )
        broken = [*broken, *unusable]
        working = [f for f in working if f not in unusable]

    if broken:
        try:
            from .feed_fallback import (
                download_fallbacks, find_fallbacks, load_catalog, transitland_fallbacks, write_feed_report,
            )

            catalog = load_catalog(data_dir / "_cache" / "mdb_feeds.csv")
            matches = find_fallbacks(broken, working, state_name, catalog, tuple(state_gdf.total_bounds))
            from .feed_validate import filter_downloaded

            downloaded = filter_downloaded(download_fallbacks(matches, gtfs_dir))   # a ferry feed can never stand for a subway
            for folder, rows in downloaded.items():
                if folder not in gtfs_dirs:
                    gtfs_dirs.append(folder)
                ntd_rows_by_dir.setdefault(folder, []).extend(rows)

            # Still missing after the Mobility Database: Transitland (only if a key is configured).
            resolved = {row.get("agency_name") for rows in downloaded.values() for row in rows}
            leftover = [f for f in broken if f.raw["ntd"].get("agency_name") not in resolved]
            tl_downloaded = filter_downloaded(transitland_fallbacks(leftover, state_name, gtfs_dir)) if leftover else {}
            for folder, rows in tl_downloaded.items():
                if folder not in gtfs_dirs:
                    gtfs_dirs.append(folder)
                ntd_rows_by_dir.setdefault(folder, []).extend(rows)

            # Last resort: fuzzy name + spatial search verified against the candidates' own stops.
            from .feed_match import match_unresolved

            resolved |= {row.get("agency_name") for rows in tl_downloaded.values() for row in rows}
            covered = {f.raw["ntd"].get("agency_name") for f in working}
            todo = [f.raw["ntd"] for f in broken
                    if f.raw["ntd"].get("agency_name") not in resolved | covered]
            fuzzy_ok, fuzzy_report = match_unresolved(todo, state_abbr, catalog, gtfs_dir) if todo else ({}, {})
            fuzzy_ok = filter_downloaded(fuzzy_ok)
            for folder, rows in fuzzy_ok.items():
                if folder not in gtfs_dirs:
                    gtfs_dirs.append(folder)
                ntd_rows_by_dir.setdefault(folder, []).extend(rows)
                logger.info(f"Fuzzy-matched feed {folder.name} for {[r.get('agency_name') for r in rows]}")

            write_feed_report(state_dir, state_abbr, working, broken, matches, downloaded, tl_downloaded,
                              fuzzy_ok, fuzzy_report)
        except Exception:
            logger.error(f"Feed fallback failed:\n{traceback.format_exc()}")
    for folder in ensure_manual_feeds(state_abbr, gtfs_dir):
        if folder not in gtfs_dirs:
            gtfs_dirs.append(folder)
            ntd_rows_by_dir.setdefault(folder, [])
    for folder in neighbour_feed_dirs(state_dir):   # the neighbouring states' feeds with stops in this state (clipped to it when scored)
        if folder not in gtfs_dirs:
            gtfs_dirs.append(folder)
            ntd_rows_by_dir.setdefault(folder, [])
    # fallback feeds (Mobility Database / Transitland) get the same treatment
    for folder in gtfs_dirs:
        fixed = sanitize_feed_folder(folder)
        if fixed:
            logger.info(f"Sanitized feed folder {folder.name}: {fixed}")
    # The same service published twice (an NTD row per mode with one weblink, an NTD feed plus the Mobility Database / Transitland copy of
    # the same agency, ...) would count every trip twice: ~22% of the national stop rows were such copies. Keep the larger feed; the
    # NTD rows of a dropped copy move to the feed that replaces it, so the audit still sees them resolved.
    from pyGTFSHandler.utils.feed_dedupe import find_duplicate_feeds

    from .gtfs_stage import ANALYSIS_DATE

    dup = find_duplicate_feeds(gtfs_dirs, service_date=ANALYSIS_DATE)   # of two copies, the one running on the analysis date
    if dup:
        logger.info(f"{state_abbr}: duplicate feeds dropped: {[f'{a.name} (= {b.name})' for a, b in dup.items()]}")
        for gone, keep in dup.items():
            ntd_rows_by_dir.setdefault(keep, []).extend(ntd_rows_by_dir.pop(gone, []))
        gtfs_dirs = [d for d in gtfs_dirs if d not in dup]
    try:   # every NTD (agency, mode) row must end with a valid feed; the rest is logged as an ERROR and kept in feed_audit.json
        import json

        from .feed_validate import audit_state

        problems = audit_state(state_abbr, [f.raw["ntd"] for f in feeds], gtfs_dirs, ntd_rows_by_dir)
        (state_dir / "feed_audit.json").write_text(json.dumps({"unresolved_ntd_rows": problems}, indent=1))
    except Exception:
        logger.error(f"feed audit failed:\n{traceback.format_exc()}")
    return gtfs_dirs, ntd_rows_by_dir
