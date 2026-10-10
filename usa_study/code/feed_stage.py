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
            downloaded = download_fallbacks(matches, gtfs_dir)
            for folder, rows in downloaded.items():
                if folder not in gtfs_dirs:
                    gtfs_dirs.append(folder)
                ntd_rows_by_dir.setdefault(folder, []).extend(rows)

            # Still missing after the Mobility Database: Transitland (only if a key is configured).
            resolved = {row.get("agency_name") for rows in downloaded.values() for row in rows}
            leftover = [f for f in broken if f.raw["ntd"].get("agency_name") not in resolved]
            tl_downloaded = transitland_fallbacks(leftover, state_name, gtfs_dir) if leftover else {}
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
            for folder, rows in fuzzy_ok.items():
                if folder not in gtfs_dirs:
                    gtfs_dirs.append(folder)
                ntd_rows_by_dir.setdefault(folder, []).extend(rows)
                logger.info(f"Fuzzy-matched feed {folder.name} for {[r.get('agency_name') for r in rows]}")

            write_feed_report(state_dir, state_abbr, working, broken, matches, downloaded, tl_downloaded,
                              fuzzy_ok, fuzzy_report)
        except Exception:
            logger.error(f"Feed fallback failed:\n{traceback.format_exc()}")
    # fallback feeds (Mobility Database / Transitland) get the same treatment
    for folder in gtfs_dirs:
        fixed = sanitize_feed_folder(folder)
        if fixed:
            logger.info(f"Sanitized feed folder {folder.name}: {fixed}")
    return gtfs_dirs, ntd_rows_by_dir
