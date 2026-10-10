"""Keyless fallback for NTD feeds whose own weblink is dead / blocked / needs an API key.

The NTD weblink for an agency is sometimes unusable (403/406 bot filters, 401 API-key
walls such as WMATA, dead links). The Mobility Database (MobilityData) publishes a public
catalog (`feeds_v2.csv`) and keeps an archived copy of each feed at
`https://files.mobilitydatabase.org/<id>/latest.zip` -- no key or token needed. For each
broken NTD feed we look for the same agency in that catalog (same state, similar name)
and download the archived copy instead.

Matching is deliberately conservative: normalized-name token overlap within the same
state, and agencies that already have a working NTD feed in the state are skipped so
the same service is never loaded twice.
"""
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

CATALOG_URL = "https://files.mobilitydatabase.org/feeds_v2.csv"
CATALOG_MAX_AGE_S = 7 * 24 * 3600
MIN_SCORE = 0.5

#: Known exceptions: NTD `agency_name` -> Mobility Database feed ids to use when its own NTD weblink
#: is unusable (401 API-key wall, 403/406 bot filter, dead link). Checked before the fuzzy name match.
#: Seeded from the 2026-10 national run; extend it as `data/feed_exceptions.csv` shows new cases.
KNOWN_FEED_OVERRIDES: Dict[str, List[str]] = {
    # id -> Mobility Database / Transitland-sourced feed ids, checked by hand (same agency; archived copy
    # downloads; schedule not older than ~2024 unless noted). Catalog status "inactive"/"deprecated" only
    # means the agency's own URL stopped being tracked, the archived last version is still the right network.
    "Washington Metropolitan Area Transit Authority": ["mdb-1846", "mdb-1847"],  # WMATA: NTD link needs an API key (401)
    "Rhode Island Public Transit Authority": ["mdb-624"],  # RIPTA: site answers 403 to scripts; catalog lists it under MA
    "Transportation District Commission of Hampton Roads": ["mdb-473"],  # HRT
    "Alameda-Contra Costa Transit District": ["mdb-2455"],  # AC Transit
    # California
    "Sacramento Regional Transit District": ["mdb-2137"],  # SacRT (schedule ends 2025-04)
    "North County Transit District": ["mdb-3093"],  # NCTD: own link needs legacy TLS (handled by robust_get)
    "City of Culver City": ["mdb-38"],
    "Livermore / Amador Valley Transit Authority": ["mdb-64"],  # LAVTA / Wheels
    "The Eastern Contra Costa Transit Authority": ["mdb-1974"],  # Tri Delta Transit
    "Butte County Association of Governments": ["ntd-90208"],  # Butte Regional Transit (B-Line)
    "City of Glendora": ["mdb-3097"],
    "City of Ridgecrest": ["mdb-2279"],
    "City of Rio Vista": ["mdb-81"],  # Delta Breeze
    # Florida
    "Pinellas Suncoast Transit Authority": ["mdb-326"],  # PSTA
    "Broward County Board of County Commissioners": ["mdb-330"],  # Broward County Transit
    "County of Citrus": ["mdb-2257"],
    "City of Ocala, Florida": ["mdb-343"],  # SunTran (Marion County)
    # Georgia / South Carolina / Tennessee / West Virginia / Wisconsin
    "Augusta Richmond County Transit Department": ["tld-5780"],
    "Macon-Bibb County Transit Authority": ["tld-7846"],
    "City of Rock Hill": ["tld-6741"],  # My Ride Rock Hill
    "City of Clarksville": ["tld-4131"],  # Clarksville Transit
    "Monongalia County Urban Mass Transit Authority": ["mdb-407"],  # Mountain Line Transit (Morgantown)
    "Milwaukee County": ["mdb-2127"],  # MCTS (schedule ends 2025-08)
    # second review (matched by NTD city + service name; all feeds download and are the same agency)
    "Eagle Valley Transportation Authority": ["mdb-2045"],  # ECO Transit
    "East Alabama Regional Planning and Development Commission": ["mdb-2211"],  # Areawide Community Transportation
    "Four Rivers Resource Services": ["ntd-50276"],  # Ride Solution (Boonville IN)
    "Senior Citizens Council": ["tld-4694"],  # Goose Creek Transit (Sheridan WY)
    "Hood River County Transportation District": ["mdb-260"],  # Columbia Area Transit
    "OCCK, Inc.": ["mdb-1867"],  # 81 Connection / CityGo / KanConnect (Salina KS)
    # third review (web research of the agencies' own data pages + catalog checks)
    "Regional Transit Service": ["mdb-533"],  # Rochester RGRTA / RTS (1,885 stops, 9,054 trips, valid to 2027-01)
    "San Diego Metropolitan Transit System": ["mdb-13"],  # MTS (4,359 stops, 25,609 trips; the Transitland copy gave the trolley a 67 min headway)
    # partial NTD feeds replaced (see REPLACE_NTD_AGENCIES)
    "Denver Regional Transportation District": ["mdb-178"],  # RTD (the NTD link is one 7-stop route)
    "Cambria County Transit Authority": ["tld-4671"],
    # found with the authenticated Mobility Database search: Linn Shuttle serves Sweet Home and Lebanon (70 stops, valid to 2026-11)
    "Senior Citizens of Sweet Home, Inc.": ["mdb-137"],
    "Oregon|City of Lebanon": ["mdb-137"],  # state-aware key: other states have a "City of Lebanon" too
}

#: NTD agencies (by NTD id) whose NTD weblink is a small PARTIAL feed (a route or a mode, found by comparing weekday trips
#: with fleet size): their NTD feeds are treated as broken so the override feeds replace them.
REPLACE_NTD_AGENCIES = {
    "80006": "Denver Regional Transportation District",  # 1,053 vehicles, feed has 72 weekday trips
    "30034": "Maryland Transit Administration",  # 1,814 vehicles, feed has 394 weekday trips
    "40027": "Pinellas Suncoast Transit Authority",  # 532 vehicles, 30 trips
    "42000": "Atlanta-Region Transit Link Authority",  # 265 vehicles, 123 trips
    "30012": "Cambria County Transit Authority",  # 117 vehicles, 74 trips
    "60091": "Hill Country Transit District",  # 113 vehicles, 30 trips
}

#: Reviewed and deliberately NOT matched (kept as `unresolved` in the feed report):
#:  - blocked at the agency's own site, no archived copy: City of Torrance, City of Lompoc, City of West Hollywood,
#:    City of Glendale (Beeline), County of Hawaii (Hele-On)  -> need a manual GTFS zip or a Transitland key
#:  - schedule too old: City of Calabasas (2021), City of Jonesboro (2017)
#:  - empty stops.txt: Anaheim Transportation Network
#:  - not in the catalog (or only a wrong-agency name match): San Diego MTS, South Gate, Yurok Tribe, County of
#:    Sacramento Municipal Services Agency, Okaloosa County, Coconut Creek, Pembroke Pines, Murfreesboro,
#:    Central West Virginia Transit Authority

_STOP = {
    "of", "the", "and", "city", "county", "town", "transit", "transportation", "authority", "district",
    "department", "dept", "board", "commissioners", "public", "system", "services", "service", "agency",
    "regional", "metropolitan", "metro", "corporation", "corp", "inc", "co", "dba", "area", "urban",
    "mass", "government", "commission", "rapid", "state", "dot", "llc", "benefit", "special",
}


def _tokens(name: str) -> set:
    words = re.findall(r"[a-z0-9]+", (name or "").lower())
    return {w for w in words if w not in _STOP and len(w) > 1}


def _score(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def load_catalog(cache_path: Path) -> pd.DataFrame:
    """Active US GTFS rows of the Mobility Database catalog (cached on disk, refreshed weekly)."""
    from pyGTFSHandler.downloaders.utils.download import robust_get

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.is_file() or time.time() - cache_path.stat().st_mtime > CATALOG_MAX_AGE_S:
        try:
            with robust_get(CATALOG_URL, timeout=120) as r:
                cache_path.write_bytes(r.content)
        except Exception as exc:
            logger.warning(f"Could not refresh the Mobility Database catalog: {exc}")
            if not cache_path.is_file():
                return pd.DataFrame()
    df = pd.read_csv(cache_path, low_memory=False)
    df = df[(df["location.country_code"] == "US") & (df["data_type"] == "gtfs")].copy()
    # Archived copy if the catalog has one, else the agency's own direct link.
    df["urls.latest"] = df["urls.latest"].fillna(df["urls.direct_download"])
    return df[df["urls.latest"].notna()].reset_index(drop=True)


def find_fallbacks(
    broken: List, working: List, state_name: str, catalog: pd.DataFrame, state_bounds=None
) -> List[Tuple[str, str, dict]]:
    """For each broken NTD feed, the best catalog matches: `(catalog_id, latest_url, ntd_row)` tuples.

    A catalog row is a candidate if its subdivision is the state OR its bounding box overlaps
    `state_bounds` (`(minx, miny, maxx, maxy)`): the catalog's subdivision is sometimes wrong
    (RIPTA is listed under Massachusetts).
    """
    if catalog.empty:
        return []
    mask = catalog["location.subdivision_name"].fillna("").str.lower() == state_name.lower()
    if state_bounds is not None:
        minx, miny, maxx, maxy = state_bounds
        mask |= (
            (catalog["location.bounding_box.maximum_longitude"] >= minx)
            & (catalog["location.bounding_box.minimum_longitude"] <= maxx)
            & (catalog["location.bounding_box.maximum_latitude"] >= miny)
            & (catalog["location.bounding_box.minimum_latitude"] <= maxy)
        )
    in_state = catalog[mask & (catalog["status"] == "active")]
    working_names = {f.raw["ntd"].get("agency_name", "") for f in working}
    out: List[Tuple[str, str, dict]] = []
    seen_ids = set()
    for feed in broken:
        ntd = feed.raw["ntd"]
        agency = ntd.get("agency_name", "")
        if agency in working_names:
            continue  # the agency already has a working feed (another NTD row)
        direct = DIRECT_URL_OVERRIDES.get(agency)
        if direct and ("url-" + re.sub(r"[^a-z0-9]+", "-", agency.lower()).strip("-")) not in seen_ids:
            cid = "url-" + re.sub(r"[^a-z0-9]+", "-", agency.lower()).strip("-")
            seen_ids.add(cid)
            out.append((cid, direct, ntd))
            logger.info(f"Direct URL override for broken NTD feed '{agency}': {direct}")
            continue
        forced = KNOWN_FEED_OVERRIDES.get(f"{state_name}|{agency}") or KNOWN_FEED_OVERRIDES.get(agency)
        if forced:
            known = catalog[catalog["id"].isin(forced)]
            for _, row in known.iterrows():
                if row["id"] not in seen_ids:
                    seen_ids.add(row["id"])
                    out.append((row["id"], row["urls.latest"], ntd))
                    logger.info(f"Known override for broken NTD feed '{agency}': {row['id']} '{row['provider']}'")
            if len(known):
                continue
        scores = in_state["provider"].map(lambda p: _score(agency, str(p)))
        if scores.empty or scores.max() < MIN_SCORE:
            continue
        for idx in scores[scores >= max(MIN_SCORE, scores.max() - 0.1)].index:
            row = in_state.loc[idx]
            if row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            out.append((row["id"], row["urls.latest"], ntd))
            logger.info(f"Fallback for broken NTD feed '{agency}': {row['id']} '{row['provider']}'")
    return out


def download_fallbacks(matches: List[Tuple[str, str, dict]], gtfs_dir: Path) -> Dict[Path, List[dict]]:
    """Download + unzip each matched archived feed into `gtfs_dir`; returns `{folder: [ntd_row]}`."""
    from pyGTFSHandler.downloaders.utils.download import robust_get
    from pyGTFSHandler.utils import io

    result: Dict[Path, List[dict]] = {}
    for cid, url, ntd in matches:
        folder = gtfs_dir / f"{cid}_fallback"
        if not folder.is_dir():
            zip_path = gtfs_dir / f"{cid}_fallback.zip"
            try:
                with robust_get(url, timeout=180) as r, open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                extracted = Path(io.unzip(str(zip_path)))
                zip_path.unlink(missing_ok=True)
                if extracted != folder and extracted.is_dir():
                    extracted.rename(folder)
            except Exception as exc:
                logger.warning(f"Fallback download of {cid} failed: {exc}")
                zip_path.unlink(missing_ok=True)
                continue
        if folder.is_dir():
            result.setdefault(folder, []).append(ntd)
    return result


def write_feed_report(
    state_dir: Path, state_abbr: str, working: List, broken: List, matches, downloaded, transitland=None,
    fuzzy=None, fuzzy_report=None
) -> Path:
    """Log every feed exception of one state to `<state_dir>/feed_report.json`.

    Per broken NTD feed: agency, feed id, why it broke, and whether a Mobility Database copy replaced it.
    """
    import json

    replaced = {}
    for cid, _url, ntd in matches:
        replaced.setdefault(ntd.get("agency_name", ""), []).append(cid)
    ok_ids = {folder.name.replace("_fallback", "") for folder in downloaded}
    tl_agencies = {row.get("agency_name", "") for rows in (transitland or {}).values() for row in rows}
    fz_agencies = {r.get("agency_name", "") for rs in (fuzzy or {}).values() for r in rs}
    working_agencies = {f.raw["ntd"].get("agency_name", "") for f in working}
    entries = []
    for feed in broken:
        agency = feed.raw["ntd"].get("agency_name", "")
        ids = [c for c in replaced.get(agency, []) if c in ok_ids]
        entries.append(
            {
                "state": state_abbr,
                "ntd_feed_id": feed.id,
                "agency": agency,
                "reason": feed.raw.get("broken_reason", ""),
                "url": feed.download_url,
                "resolution": "mobility_database" if ids else (
                    "transitland" if agency in tl_agencies else "fuzzy_match" if agency in fz_agencies else (
                        "covered_by_working_feed" if agency in working_agencies else "unresolved"
                    )
                ),
                "mobility_database_ids": ids,
                "candidates": (fuzzy_report or {}).get(agency, []),
            }
        )
    path = state_dir / "feed_report.json"
    path.write_text(json.dumps({"working_feeds": len(working), "broken": entries}, indent=1))
    return path


def write_feed_exceptions_summary(data_dir: Path) -> Optional[Path]:
    """Aggregate every state's `feed_report.json` into `data/feed_exceptions.csv` (one row per broken feed)."""
    import json

    rows = []
    for report in sorted(data_dir.glob("*/feed_report.json")):
        try:
            rows.extend(json.loads(report.read_text())["broken"])
        except Exception:
            continue
    if not rows:
        return None
    out = data_dir / "feed_exceptions.csv"
    pd.DataFrame(rows).sort_values(["resolution", "state", "agency"]).to_csv(out, index=False)
    unresolved = sum(1 for r in rows if r["resolution"] == "unresolved")
    logger.info(f"Feed exceptions: {len(rows)} broken NTD feeds, {unresolved} unresolved -> {out}")
    return out


TRANSITLAND_BASE = "https://transit.land/api/v2/rest"


def _transitland_key() -> Optional[str]:
    """Transitland API key via pyGTFSHandler's lookup: `TRANSITLAND_API_KEY`, else `api_keys.json` (see its config module)."""
    from pyGTFSHandler.downloaders.utils.config import get_api_key

    return get_api_key("transitland", None, "TRANSITLAND_API_KEY") or None


_SKIP_WORDS = {"of", "the", "and", "for", "dba", "inc", "llc", "corp", "co"}

#: NTD agency name -> names Transitland uses for it (when neither words nor initials match).
TRANSITLAND_ALIASES = {
    "Central Puget Sound Regional Transit Authority": ["Sound Transit"],
    "Los Angeles County Metropolitan Transportation Authority": ["Metro - Los Angeles", "LA Metro"],
    "Southeastern Pennsylvania Transportation Authority": ["SEPTA"],
    "Metropolitan Atlanta Rapid Transit Authority": ["MARTA"],
}
#: Transitland names/feeds that look like a match by words but are a different (or nationwide) service.
_WRONG_KIND = ("airport", "university", "college", "campus", "megabus", "flixbus", "greyhound", "amtrak", "intercity bus", "bay~area~rg")


def _begins_subsequence(word: str, chunk: str) -> bool:
    """`chunk` starts with the word's first letter and its letters occur in `word` in order."""
    if not word or word[0] != chunk[0]:
        return False
    it = iter(word)
    return all(c in it for c in chunk)


def _acronym_match(name: str, acr: str) -> bool:
    """True if `acr` (e.g. 'SEPTA', 'MBTA') can be written from the words of `name`: each significant word contributes
    its first 1-3 letters, in order (Southeastern Pennsylvania Transportation Authority -> SE+P+T+A)."""
    words = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in _SKIP_WORDS]
    acr = re.sub(r"[^a-z0-9]", "", acr.lower())
    if len(acr) < 3 or not words:
        return False

    def go(wi: int, ai: int) -> bool:
        if wi == len(words):
            return ai == len(acr)
        for k in (1, 2, 3):
            chunk = acr[ai : ai + k]
            if len(chunk) == k and _begins_subsequence(words[wi], chunk) and go(wi + 1, ai + k):
                return True
        return False

    return go(0, 0)


def _match_score(ntd_name: str, tl_name: str) -> float:
    low_t, low_n = (tl_name or "").lower(), ntd_name.lower()
    if any(w in low_t and w not in low_n for w in _WRONG_KIND):
        return 0.0
    base = _score(ntd_name, tl_name)
    if _acronym_match(ntd_name, tl_name):
        base = max(base, 0.9)
    # camel-case brand names that contain a distinctive word of the official name (GoTriangle ~ Research Triangle ...)
    t_low = re.sub(r"[^a-z0-9]", "", (tl_name or "").lower())
    if any(len(w) >= 6 and (w in t_low or t_low in w) and len(t_low) >= 6 for w in _tokens(ntd_name)):
        base = max(base, 0.6)
    return base


#: Hand-verified Transitland matches for NTD agencies whose official names share no word with the name Transitland
#: uses: NTD agency -> (Transitland agency name, optional exact feed onestop id when the name is ambiguous).
TRANSITLAND_OVERRIDES = {
    "Research Triangle Regional Public Transportation Authority": ("GoTriangle", None),
    "City of Albuquerque": ("ABQ RIDE", None),
    "City and County of Honolulu": ("TheBus", None),
    "City of Memphis": ("MATA", None),
    "Interurban Transit Partnership": ("The Rapid", None),
    "Rock Island County Metropolitan Mass Transit District": ("MetroLINK", None),
    "City of Galveston": ("Island Transit", "f-galveston~tx~us"),
    "Ames Transit Agency": ("CyRide", None),
    "City of Dubuque": ("The Jule", None),
    "City of Columbia": ("GoCOMO", None),
    "New York City Economic Development Corporation": ("NYC Ferry", None),
    "Lawrence County Port Authority": ("Lawrence County Transit", None),
    "People for People Yakima": ("People for People", None),
    "Metro-North Commuter Railroad Company, dba: MTA Metro-North Railroad": ("Metro-North Railroad", "f-dr7-mtanyc~metro~north"),
    "Metro Transit": ("Metro Transit", "f-9zv-twin~cities~minnesota"),
    "King County": ("Metro Transit", "f-c23-metrokingcounty"),
    "Fairfax County, VA": ("Fairfax Connector", None),
    "Regional Public Transportation Authority": ("Valley Metro", "f-9tb-valleymetro"),
    "Valley Metro Rail, Inc.": ("Valley Metro", "f-9tb-valleymetro"),
    "Buncombe County": ("Mountain Mobility", None),
    "Board of County Commissioners of Calvert County, Maryland": ("Calvert County Public Transportation", None),
    "Mayor and City Council Town of Ocean City": ("Ocean City Transportation", None),
    "City of Las Cruces NM": ("Roadrunner Transit", None),
    "Carson Area Metropolitan Planning Organization": ("Jump Around Carson", None),
    "Arrowhead Economic Opportunity Agency, Inc.": ("Arrowhead Transit", None),
    # second review
    "San Diego Metropolitan Transit System": ("MTS", "f-9mu-mts"),  # the MTS feed (agencies: MTS + airport shuttle)
    "County of Hawaii": ("Hele On Bus", "f-hele~on~hi"),
    "Central Oklahoma Transportation and Parking Authority": ("EMBARK", "f-9y69-embark"),
    "Okaloosa County Board of County Commissioners": ("Emerald Coast Rider", "f-dj6m-okaloosacountytransit~fl~us"),
    "City of Evansville": ("METS", "f-evansville~in~us"),
    "Michiana Area Council of Governments": ("Interurban Trolley", "f-macog"),
    "Virginia|City of Fredericksburg": ("FXBGO!", "f-fredericksburg~regional~transit"),
    "McLennan County": ("Waco Transit", "f-waco~transit"),
    "Umpqua Public Transportation District": ("UTrans", "f-9r8m-umpquatransit~or~us"),
    "Terrebonne Parish Consolidated Government": ("Good Earth Transit", None),
    "Stanislaus Regional Transit Authority": ("The S", "f-9q9-modesto~ca~us"),
    # fuzzy + spatial match, verified against the candidate's stops and routes
    "River Bend Transit": ("DAVENPORT CITIBUS", "f-citibus~ia~us"),  # Davenport IA: 11 routes, 565 stops, all within 15 km
    "Town of Wallkill": ("Middletown Area Transit", "f-middletown~ny~us"),  # Middletown NY: 4 routes, 26 stops
    "Arizona|City of Glendale": ("Valley Metro", "f-9tb-valleymetro"),  # (Glendale, CA's Beeline is a different agency)
    "Arizona|City of Scottsdale": ("Valley Metro", "f-9tb-valleymetro"),
    "Tennessee|Regional Transportation Authority": ("WeGo Public Transit", "f-dn6-regionaltransportationauthorityofmiddletennessee~nashville"),
    # partial NTD feeds replaced (see REPLACE_NTD_AGENCIES); a list = several feeds of one agency
    "Maryland Transit Administration": [
        ("Maryland Transit Administration", "f-dq-mtamaryland~bus"),
        ("Maryland Transit Administration", "f-dq-mtamaryland~subwaylink"),
        ("Maryland Transit Administration", "f-dq-mtamaryland~raillink"),
        ("Maryland Transit Administration", "f-dq-mtamaryland~commuterbus"),
    ],
    "Atlanta-Region Transit Link Authority": ("Xpress", "f-dnh-grtaxpress"),
    "Hill Country Transit District": ("Hill Country Transit District", "f-hctd~tx~us"),
}
#: NTD agency-name prefix -> same, for families of NTD rows (CTtransit divisions are listed as separate agencies).
TRANSITLAND_PREFIX_OVERRIDES = {
    "Connecticut Department of Transportation": ("CTtransit", "f-drk-cttransit~hartford"),
}
#: Official download URLs found on the agencies' own data pages, for agencies whose catalog copy is stale or missing.
DIRECT_URL_OVERRIDES = {
    "Loudoun County": "https://www.loudoun.gov/loudountransitgtfs",  # current feed; the catalog copy ended 2026-09-26
}

#: Reviewed and NOT matched (no usable feed found in the Mobility Database or Transitland, or only a wrong-agency name):
#:  East Alabama RPDC, Anaheim Transportation Network, South Gate, County of Sacramento MSA, San Diego MTS (only an airport
#:  feed), Stanislaus RTA, Eagle Valley TA, Glendale/Scottsdale AZ, Coconut Creek, Pembroke Pines, Tamarac, Davie, Okaloosa,
#:  Four Rivers, Michiana COG, Evansville, OCCK, Paducah, Terrebonne, Beverly MA, Mission Hill Link, City of Niles,
#:  Central Oklahoma TPA, Lebanon OR, Hood River, Sweet Home, Umpqua, Town of Wallkill, Regional Transit Service (NY),
#:  Suburban Transit Lines / OBC Lines / Community Transit Lines (NJ contractors), Loudoun County, City of Fredericksburg,
#:  Central West Virginia TA, Senior Citizens Council (WY), River Bend Transit, McLennan County, TN Regional Transportation
#:  Authority (WeGo feed exists but may duplicate Nashville service), County of Hawaii (own site blocks, no archive).


def _override_for(agency: str, state_name: str = ""):
    if f"{state_name}|{agency}" in TRANSITLAND_OVERRIDES:
        return TRANSITLAND_OVERRIDES[f"{state_name}|{agency}"]
    if agency in TRANSITLAND_OVERRIDES:
        return TRANSITLAND_OVERRIDES[agency]
    for prefix, val in TRANSITLAND_PREFIX_OVERRIDES.items():
        if agency.startswith(prefix) and "cttransit" in agency.lower().replace(" ", ""):
            return val
    return None


def transitland_feed_for_agency(agency: str, state_name: str, key: str) -> List[Tuple[str, str]]:
    """The best-matching Transitland agency's `(feed_onestop_id, agency_name)` for an NTD agency in a state.

    Several query variants are tried (full name, distinctive words, initials), because Transitland knows agencies by
    short names ("MBTA", "SEPTA") that share no word with NTD's official names. Candidates are scored by word overlap or
    acronym match; airports, universities and nationwide bus brands are rejected; only the best score is kept (if
    equally good, a non-"flex" feed is preferred, and distinct feeds are kept only when their names differ, e.g. a
    rail and a bus feed).
    """
    import requests

    forced = _override_for(agency, state_name)
    if isinstance(forced, list):  # several feeds of one agency, all with explicit ids
        return [(fid, name) for name, fid in forced if fid]
    if forced is not None:
        tl_name, fid = forced
        if fid:
            return [(fid, tl_name)]
        try:
            r = requests.get(
                f"{TRANSITLAND_BASE}/agencies",
                params={"search": tl_name, "adm1_name": state_name, "limit": 20, "apikey": key}, timeout=60,
            )
            r.raise_for_status()
            for a in r.json().get("agencies", []):
                f_id = ((a.get("feed_version") or {}).get("feed") or {}).get("onestop_id")
                if f_id and (a.get("agency_name") or "").strip().lower().startswith(tl_name.lower()):
                    return [(f_id, a["agency_name"])]
        except Exception as exc:
            logger.warning(f"Transitland override lookup for '{tl_name}' failed: {exc}")
        return []

    words = [w for w in re.findall(r"[a-z0-9]+", agency.lower()) if w not in _SKIP_WORDS]
    initials = "".join(w[0] for w in words)
    aliases = TRANSITLAND_ALIASES.get(agency, [])
    variants = list(dict.fromkeys([agency, " ".join(sorted(_tokens(agency))), initials.upper(), *aliases]))
    cands = {}
    for term in variants:
        if len(term) < 3:
            continue
        try:
            r = requests.get(
                f"{TRANSITLAND_BASE}/agencies",
                params={"search": term, "adm1_name": state_name, "limit": 20, "apikey": key},
                timeout=60,
            )
            r.raise_for_status()
        except Exception as exc:
            logger.warning(f"Transitland agency search for '{term}' failed: {exc}")
            continue
        for a in r.json().get("agencies", []):
            fid = ((a.get("feed_version") or {}).get("feed") or {}).get("onestop_id")
            name = a.get("agency_name") or ""
            if fid and fid not in cands:
                score = _match_score(agency, name)
                if any(name.lower() == al.lower() for al in aliases):
                    score = max(score, 0.95)
                cands[fid] = (score, name)
    good = {f: v for f, v in cands.items() if v[0] >= MIN_SCORE}
    if not good:
        return []
    top = max(v[0] for v in good.values())
    best = sorted(((f, v[1]) for f, v in good.items() if v[0] >= top - 1e-9), key=lambda x: ("flex" in x[0], x[0]))
    out, names = [], set()
    for f, n in best:
        if n.lower() in names:
            continue  # same agency name twice (duplicate feeds): keep one
        names.add(n.lower())
        out.append((f, n))
    return out


def download_transitland_feed(onestop_id: str, dst_zip: Path, key: str) -> bool:
    """Download Transitland's archived latest version of a feed; True if a valid zip with stops.txt was saved."""
    import io
    import zipfile

    import requests

    try:
        r = requests.get(
            f"{TRANSITLAND_BASE}/feeds/{onestop_id}/download_latest_feed_version", params={"apikey": key}, timeout=300
        )
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        if not any(n.lower().endswith("stops.txt") for n in z.namelist()):
            return False
        dst_zip.write_bytes(r.content)
        return True
    except Exception as exc:
        logger.warning(f"Transitland download of {onestop_id} failed: {exc}")
        return False


def transitland_fallbacks(unresolved: List, state_name: str, gtfs_dir: Path) -> Dict[Path, List[dict]]:
    """Second fallback, for agencies the Mobility Database could not provide: Transitland.

    Uses the Transitland API key (`TRANSITLAND_API_KEY` or `"transitland"` in `~/.pygtfshandler/api_keys.json`); without
    one it does nothing. For each unresolved NTD feed it finds the same agency in Transitland (same state, matching
    name), downloads Transitland's archived latest version of that agency's feed and returns `{folder: [ntd_row]}`.
    """
    from pyGTFSHandler.utils import io

    key = _transitland_key()
    if not key:
        logger.info("Transitland fallback skipped (no API key configured)")
        return {}

    result: Dict[Path, List[dict]] = {}
    done = set()
    for feed in unresolved:
        ntd = feed.raw["ntd"]
        agency = ntd.get("agency_name", "")
        for fid, name in transitland_feed_for_agency(agency, state_name, key):
            folder = gtfs_dir / f"{fid}_fallback"
            if fid in done:
                result.setdefault(folder, []).append(ntd)
                continue
            if not folder.is_dir():
                zip_path = gtfs_dir / f"{fid}_fallback.zip"
                if not download_transitland_feed(fid, zip_path, key):
                    zip_path.unlink(missing_ok=True)
                    continue
                try:
                    extracted = Path(io.unzip(str(zip_path)))
                    zip_path.unlink(missing_ok=True)
                    if extracted != folder and extracted.is_dir():
                        extracted.rename(folder)
                except Exception as exc:
                    logger.warning(f"Transitland feed {fid}: could not unzip: {exc}")
                    zip_path.unlink(missing_ok=True)
                    continue
            if folder.is_dir():
                done.add(fid)
                result.setdefault(folder, []).append(ntd)
                logger.info(f"Transitland fallback for broken NTD feed '{agency}': {fid} ('{name}')")
    return result
