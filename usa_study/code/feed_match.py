"""Last-resort feed discovery for agencies no pinned override / Mobility Database name match / Transitland name
match could resolve: fuzzy NAME + SPATIAL matching, followed by verification against the candidate's own stops.

1. Geocode the NTD agency city (Nominatim).
2. Candidates = Mobility Database feeds whose bounding box contains the point + Transitland agencies within 8 km,
   ranked by name similarity (acronym/token overlap + character ratio).
3. Each top candidate is downloaded and checked: stops within 5/15 km of the city, stop names containing the city,
   trips, service end date.
4. A candidate is ACCEPTED automatically only if it is verified (see `is_verified`); all others are kept as
   suggestions in `feed_report.json` for manual review.
"""
import csv
import difflib
import io
import math
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from .feed_fallback import _tokens, _transitland_key

TL = "https://transit.land/api/v2/rest"
UA = {"User-Agent": "ptlos-feed-match/1.0"}
MIN_SIMILARITY = 0.45


def geocode(city: str, state_abbr: str) -> Optional[Tuple[float, float]]:
    for attempt in range(3):
      try:
        r = requests.get("https://nominatim.openstreetmap.org/search", timeout=30, headers=UA,
                         params={"q": f"{city}, {state_abbr}, USA", "format": "json", "limit": 1}).json()
        time.sleep(1.1)  # Nominatim usage policy
        if r:
            return float(r[0]["lat"]), float(r[0]["lon"])
      except Exception:
        time.sleep(5)
    return None


def _km(a, b, c, d) -> float:
    p = math.pi / 180
    x = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(x))


def similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    jac = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    return max(jac, difflib.SequenceMatcher(None, a.lower(), str(b).lower()).ratio() * 0.8)


def spatial_candidates(agency: str, city: str, lat: float, lon: float, catalog) -> List[dict]:
    """Mobility Database (bbox contains point) + Transitland (8 km) candidates, best name similarity first."""
    cands = []
    if catalog is not None and len(catalog):
        import pandas as pd

        pre = "location.bounding_box."
        if pre + "minimum_latitude" in catalog.columns:
            num = {c: pd.to_numeric(catalog[pre + c], errors="coerce")
                   for c in ("minimum_latitude", "maximum_latitude", "minimum_longitude", "maximum_longitude")}
            inb = catalog[(num["minimum_latitude"] <= lat) & (num["maximum_latitude"] >= lat)
                          & (num["minimum_longitude"] <= lon) & (num["maximum_longitude"] >= lon)]
            for x in inb.itertuples():
                cands.append({"source": "mdb", "id": str(x.id), "name": f"{x.provider} {x.name}",
                              "url": f"https://files.mobilitydatabase.org/{x.id}/latest.zip",
                              "sim": similarity(f"{agency} {city}", f"{x.provider} {x.name}")})
    key = _transitland_key()
    if key:
        try:
            rows = requests.get(f"{TL}/agencies", timeout=40, headers=UA,
                                params={"lat": lat, "lon": lon, "radius": 8000, "limit": 15, "apikey": key}
                                ).json().get("agencies", [])
            for a in rows:
                fid = ((a.get("feed_version") or {}).get("feed") or {}).get("onestop_id")
                if fid:
                    cands.append({"source": "transitland", "id": fid, "name": a.get("agency_name") or "",
                                  "url": f"{TL}/feeds/{fid}/download_latest_feed_version?apikey={key}",
                                  "sim": similarity(f"{agency} {city}", a.get("agency_name") or "")})
        except Exception:
            pass
    seen, out = set(), []
    for c in sorted(cands, key=lambda c: -c["sim"]):
        if c["id"] not in seen:
            seen.add(c["id"])
            out.append(c)
    return out


def verify(content: bytes, lat: float, lon: float, city: str) -> dict:
    """Stop-based evidence that a downloaded zip really serves the agency's city."""
    z = zipfile.ZipFile(io.BytesIO(content))
    rd = lambda n: csv.DictReader(io.TextIOWrapper(z.open(n), encoding="utf-8-sig"))
    pts = [(float(x["stop_lat"]), float(x["stop_lon"]), x.get("stop_name", ""))
           for x in rd("stops.txt") if x.get("stop_lat") and x.get("stop_lon")]
    ends = [x["end_date"] for x in rd("calendar.txt")] if "calendar.txt" in z.namelist() else []
    return {
        "stops": len(pts),
        "within_5km": sum(_km(lat, lon, a, b) <= 5 for a, b, _ in pts),
        "within_15km": sum(_km(lat, lon, a, b) <= 15 for a, b, _ in pts),
        "name_hits": sum(city.lower() in (n or "").lower() for _, _, n in pts),
        "trips": sum(1 for _ in z.open("trips.txt")) - 1,
        "service_end": max(ends) if ends else "",
    }


def is_verified(cand: dict, ev: dict) -> bool:
    return (cand["sim"] >= MIN_SIMILARITY and ev["trips"] > 0 and ev["within_15km"] >= 3
            and (ev["name_hits"] > 0 or ev["within_5km"] >= 3))


def match_unresolved(unresolved: List[dict], state_abbr: str, catalog, gtfs_dir: Path,
                     top_n: int = 3) -> Tuple[Dict[Path, List[dict]], Dict[str, List[dict]]]:
    """`unresolved`: NTD rows (need agency_name, city). Returns ({folder: ntd_rows} accepted, {agency: evidence list})."""
    accepted: Dict[Path, List[dict]] = {}
    report: Dict[str, List[dict]] = {}
    for row in unresolved:
        agency, city = row.get("agency_name", ""), row.get("city", "")
        if not city:
            continue
        pt = geocode(city, state_abbr)
        if not pt:
            continue
        report[agency] = []
        for cand in spatial_candidates(agency, city, *pt, catalog)[:top_n]:
            try:
                content = requests.get(cand["url"], timeout=90, headers={"User-Agent": "Mozilla/5.0"}).content
                ev = verify(content, *pt, city)
            except Exception as e:
                report[agency].append({**cand, "url": None, "error": type(e).__name__})
                continue
            ok = is_verified(cand, ev)
            report[agency].append({**{k: v for k, v in cand.items() if k != "url"}, **ev, "accepted": ok})
            if ok and agency not in {r.get("agency_name") for rs in accepted.values() for r in rs}:
                folder = gtfs_dir / f"{cand['id'].replace('/', '_')}_match"
                folder.mkdir(parents=True, exist_ok=True)
                zipfile.ZipFile(io.BytesIO(content)).extractall(folder)
                accepted[folder] = [row]
    return accepted, report
