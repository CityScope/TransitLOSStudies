"""Safeguards against a wrong or missing feed ending up in a state's data.

Why this exists (2026-10): the NTD row of the MTA New York City subway (heavy rail) had a download that failed once, the Mobility Database
name matcher then picked the Staten Island FERRY feed for "Staten Island Rapid Transit Operating Authority", and nothing noticed: the whole
subway was missing from the New York map. Three guards, all cheap:

1. `replacement_problem`: a replacement feed must contain routes of the kind NTD says the row operates (rail-like for heavy / light /
   commuter rail, streetcar...; bus-like for bus; ferry for ferries). A ferry feed can never replace a subway.
2. `download_with_retries` (feed_stage): a failed NTD download is retried before the row is called broken (the subway zip that failed once
   downloads fine).
3. `audit_state` / `audit_known_systems`: after a state is acquired, every NTD (agency, mode) must have a valid feed, and well-known systems
   (`KNOWN_SYSTEMS`) must have stops where their stations are; anything else is logged as an ERROR and written to `feed_audit.json`.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: GTFS route_type families (basic + extended types).
RAIL_LIKE = set(range(0, 3)) | {5, 6, 7, 11, 12} | set(range(100, 200)) | set(range(400, 500)) | set(range(900, 1000)) | set(range(1300, 1400))
BUS_LIKE = {3, 11} | set(range(200, 300)) | set(range(700, 900))
FERRY_LIKE = {4} | set(range(1000, 1100))

#: NTD `mode_name` -> kinds of routes a feed for that mode must contain at least one of.
NTD_MODE_KINDS: Dict[str, Set[str]] = {
    "Heavy Rail": {"rail"}, "Commuter Rail": {"rail"}, "Light Rail": {"rail"}, "Hybrid Rail": {"rail"}, "Streetcar Rail": {"rail"},
    "Monorail/Automated Guideway": {"rail"}, "Alaska Railroad": {"rail"}, "Cable Car": {"rail"}, "Inclined Plane": {"rail"},
    "Aerial Tramway": {"rail"},
    "Bus": {"bus"}, "Commuter Bus": {"bus"}, "Trolleybus": {"bus", "rail"}, "Bus Rapid Transit": {"bus", "rail"},
    "Ferryboat": {"ferry"},
}


def feed_route_types(folder: Path) -> Set[int]:
    """The set of GTFS route_type values of a feed folder (empty if unreadable)."""
    p = Path(folder) / "routes.txt"
    if not p.is_file():
        return set()
    try:
        r = pd.read_csv(p, dtype=str, keep_default_na=False, encoding_errors="replace", usecols=lambda c: c.strip().lstrip("﻿") == "route_type")
    except Exception:  # noqa: BLE001
        return set()
    r.columns = ["route_type"]
    return {int(v) for v in pd.to_numeric(r["route_type"].str.strip(), errors="coerce").dropna().astype(int)}


def kinds_of(types: Iterable[int]) -> Set[str]:
    out = set()
    for t in types:
        if t in RAIL_LIKE:
            out.add("rail")
        if t in BUS_LIKE:
            out.add("bus")
        if t in FERRY_LIKE:
            out.add("ferry")
    return out


def expected_kinds(ntd_rows: Sequence[dict]) -> Set[str]:
    """Union of the kinds the NTD rows served by one feed need (empty = no expectation, e.g. a manual feed)."""
    out: Set[str] = set()
    for r in ntd_rows:
        out |= NTD_MODE_KINDS.get(str(r.get("mode_name") or r.get("mode") or ""), set())
    return out


def replacement_problem(folder: Path, ntd_rows: Sequence[dict]) -> Optional[str]:
    """Why `folder` cannot stand for the NTD rows it was matched to (None = fine)."""
    need = expected_kinds(ntd_rows)
    if not need:
        return None
    have = kinds_of(feed_route_types(folder))
    if not have:
        return "feed has no usable routes"
    if not (need & have):
        modes = sorted({str(r.get("mode_name") or r.get("mode") or "?") for r in ntd_rows})
        return f"NTD says {modes} ({'/'.join(sorted(need))}) but the feed only has {'/'.join(sorted(have))} routes"
    return None


def reject_folder(folder: Path, reason: str) -> None:
    """Move a refused feed out of the way (`_rejected_<name>`, ignored by every stage) and say so loudly."""
    logger.error(f"REJECTED feed {folder.name}: {reason}")
    target = folder.with_name("_rejected_" + folder.name)
    try:
        if target.exists():
            import shutil

            shutil.rmtree(target)
        folder.rename(target)
    except OSError as exc:  # noqa: BLE001
        logger.warning(f"could not move {folder} aside: {exc}")


def filter_downloaded(downloaded: Dict[Path, List[dict]]) -> Dict[Path, List[dict]]:
    """Drop (and move aside) every downloaded replacement that fails `replacement_problem`."""
    ok: Dict[Path, List[dict]] = {}
    for folder, rows in downloaded.items():
        why = replacement_problem(folder, rows)
        if why:
            reject_folder(folder, why)
        else:
            ok[folder] = rows
    return ok


# ---- audits -----------------------------------------------------------------------------------------------------------------------------
def audit_state(state_abbr: str, ntd_rows: Sequence[dict], gtfs_dirs: Sequence[Path], ntd_rows_by_dir: Dict[Path, List[dict]]) -> List[dict]:
    """NTD (agency, mode) rows that end without any valid feed. Logged as errors; returned for `feed_audit.json`."""
    served: Set[Tuple[str, str]] = set()
    for d in gtfs_dirs:
        rows = ntd_rows_by_dir.get(d, [])
        if replacement_problem(d, rows) is None:
            for r in rows:
                served.add((str(r.get("agency_name")), str(r.get("mode_name") or r.get("mode"))))
    problems = []
    seen: Set[Tuple[str, str]] = set()
    for r in ntd_rows:
        key = (str(r.get("agency_name")), str(r.get("mode_name") or r.get("mode")))
        if key in seen or key in served:
            continue
        seen.add(key)
        problems.append({"state": state_abbr, "agency": key[0], "mode": key[1], "ntd_id": r.get("ntd_id")})
        logger.error(f"FEED AUDIT {state_abbr}: NTD agency '{key[0]}' mode '{key[1]}' has NO valid feed -- its service is missing from the map")
    return problems


#: Rail / tram systems that must be present in the downloaded feeds after a full run: (state folder, name, lat, lon, agency keywords).
#: The check (`audit_known_systems`) needs a feed whose agency name contains one of the keywords, with rail-like routes, and a stop of it
#: within 600 m of the station. A system missing from the feeds (e.g. a ferry feed matched instead of the subway) is logged as an ERROR.
KNOWN_SYSTEMS: List[Tuple[str, str, float, float, Tuple[str, ...]]] = [
    ("NewYork", "NYC subway - Times Sq-42 St", 40.7553, -73.9872, ("new york city transit", "nyct")),
    ("NewYork", "NYC subway - 125 St (Harlem)", 40.8039, -73.9376, ("new york city transit", "nyct")),
    ("NewYork", "Staten Island Railway - Tottenville", 40.5126, -74.2514, ("new york city transit", "nyct", "staten island")),
    ("NewYork", "LIRR - Jamaica", 40.6999, -73.8084, ("long island",)),
    ("NewYork", "Metro-North - Grand Central", 40.7527, -73.9772, ("metro-north", "metro north")),
    ("Illinois", "Chicago L - Clark/Lake", 41.8857, -87.6310, ("chicago transit", "cta")),
    ("Illinois", "Metra - Union Station", 41.8786, -87.6403, ("metra", "commuter rail division")),
    ("DistrictofColumbia", "WMATA Metrorail - Metro Center", 38.8983, -77.0280, ("wmata", "washington metropolitan")),
    ("Massachusetts", "MBTA - Park Street", 42.3564, -71.0624, ("mbta", "massachusetts bay")),
    ("California", "BART - 16th St Mission", 37.7650, -122.4196, ("bart", "bay area rapid")),
    ("California", "LA Metro - Union Station", 34.0560, -118.2368, ("los angeles county metropolitan", "la metro", "metro")),
    ("California", "Muni Metro - Castro", 37.7625, -122.4350, ("san francisco municipal", "muni", "sfmta")),
    ("California", "Metrolink - Union Station", 34.0560, -118.2368, ("southern california regional rail", "metrolink")),
    ("Pennsylvania", "SEPTA - 30th Street", 39.9557, -75.1822, ("septa", "southeastern pennsylvania")),
    ("Pennsylvania", "Pittsburgh T - Steel Plaza", 40.4393, -79.9959, ("pittsburgh", "port authority", "prt")),
    ("Washington", "Sound Transit Link - Westlake", 47.6113, -122.3375, ("sound transit",)),
    ("Georgia", "MARTA - Five Points", 33.7538, -84.3915, ("marta", "metropolitan atlanta")),
    ("Maryland", "Baltimore Metro/Light Rail - Charles Center", 39.2897, -76.6143, ("maryland transit", "mta", "mdot")),
    ("Florida", "Miami Metrorail - Government Center", 25.7752, -80.1960, ("miami-dade", "miami dade")),
    ("Texas", "DART - Union Station", 32.7757, -96.8087, ("dallas area rapid", "dart")),
    ("Colorado", "Denver RTD - Union Station", 39.7547, -105.0035, ("regional transportation district", "rtd")),
    ("Arizona", "Valley Metro Rail - Central Ave/Washington", 33.4484, -112.0735, ("valley metro",)),
    ("Minnesota", "Metro Transit - Target Field", 44.9818, -93.2776, ("metro transit",)),
    ("Oregon", "TriMet MAX - Pioneer Square", 45.5188, -122.6794, ("trimet", "tri-county")),
    ("Utah", "UTA TRAX - Temple Square", 40.7704, -111.8940, ("utah transit", "uta")),
    ("NewJersey", "NJ Transit - Newark Penn", 40.7346, -74.1645, ("nj transit", "njt", "new jersey transit")),
    ("Louisiana", "New Orleans streetcar - Canal St", 29.9547, -90.0759, ("new orleans", "rta")),
]


def _rail_agency_names(folder: Path) -> str:
    """Lower-case names of the agencies that operate the feed's RAIL-like routes (a bus feed listing Sound Transit as an agency does not count)."""
    ag = pd.read_csv(folder / "agency.txt", dtype=str, keep_default_na=False, encoding_errors="replace")
    ag.columns = [c.strip().lstrip("\ufeff") for c in ag.columns]
    r = pd.read_csv(folder / "routes.txt", dtype=str, keep_default_na=False, encoding_errors="replace")
    r.columns = [c.strip().lstrip("\ufeff") for c in r.columns]
    t = pd.to_numeric(r["route_type"].str.strip(), errors="coerce")
    rail = r[t.isin(RAIL_LIKE)]
    if "agency_id" in r.columns and "agency_id" in ag.columns and len(ag) > 1:
        ids = set(rail["agency_id"].str.strip())
        names = ag.loc[ag["agency_id"].str.strip().isin(ids), "agency_name"]
        if len(names):
            return " ".join(names).lower()
    return " ".join(ag["agency_name"]).lower()


def audit_known_systems(data_dir: Path, radius_m: float = 600.0) -> List[str]:
    """Known systems missing from the downloaded feeds (see `KNOWN_SYSTEMS`). Written to `data/feed_audit_known_systems.json`."""
    missing: List[str] = []
    cache: Dict[str, List[Tuple[str, Set[int], pd.DataFrame]]] = {}
    for state, name, lat, lon, keywords in KNOWN_SYSTEMS:
        sd = Path(data_dir) / state / "gtfs"
        if not sd.is_dir():
            continue
        if state not in cache:
            feeds = []
            for d in sorted(sd.iterdir()):
                if not d.is_dir() or d.name.startswith("_") or not (d / "stops.txt").is_file():
                    continue
                rail = feed_route_types(d) & RAIL_LIKE
                if not rail:
                    continue
                try:
                    agency = _rail_agency_names(d)
                    st = pd.read_csv(d / "stops.txt", dtype=str, keep_default_na=False, encoding_errors="replace", usecols=lambda c: c.strip() in ("stop_lat", "stop_lon"))
                    st.columns = [c.strip() for c in st.columns]
                    st = st.apply(pd.to_numeric, errors="coerce").dropna()
                except Exception:  # noqa: BLE001
                    continue
                feeds.append((agency, rail, st))
            cache[state] = feeds
        found = False
        for agency, _rail, st in cache[state]:
            if not any(k in agency for k in keywords):
                continue
            d = np.hypot((st["stop_lat"].to_numpy() - lat) * 111_000, (st["stop_lon"].to_numpy() - lon) * 111_000 * np.cos(np.radians(lat)))
            if (d <= radius_m).any():
                found = True
                break
        if not found:
            missing.append(f"{state}: {name}")
            logger.error(f"KNOWN SYSTEM MISSING: {name} ({state}) -- no downloaded feed of {keywords} with rail routes has a stop within {radius_m:.0f} m of ({lat}, {lon})")
    Path(data_dir, "feed_audit_known_systems.json").write_text(json.dumps({"missing": missing, "checked": len(KNOWN_SYSTEMS)}, indent=1))
    return missing
