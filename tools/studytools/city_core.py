"""Resolve a city's "core" boundary, used to split results into core vs metro.

US cities: the Census `place` boundary matching the city name (via
`pycensus.countries.usa.constants.GEOMETRY_FUNCS["place"]`, i.e. `pygris.places`).
Non-US cities: the OSM administrative boundary for the city name, resolved
directly against Nominatim (see `_geocode_admin_polygon` below) rather than
through `UrbanAccessAnalyzer.api.AreaOfInterest.from_name`, which takes
Nominatim's single top-ranked hit unfiltered. For many city names that top
hit is a `class=place` *node* (a point marker for the city, no polygon) with
the actual `boundary=administrative` *relation* (the real polygon) ranked
lower -- e.g. Nominatim's top result for "Guadalajara, Jalisco, Mexico" is a
point; its administrative boundary relation is the second result. Taking the
point unfiltered silently produces a boundary that no H3 cell's centroid can
ever fall inside, so `core/h3_grid.parquet` comes out with zero rows instead
of erroring -- worth actively filtering for, not just hoping the top hit is
a polygon.
"""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import requests
from pycensus.countries.usa.constants import GEOMETRY_FUNCS
from shapely.geometry import shape


# geocoded boundaries, in the shared cache (`STUDY_CACHE_DIR`, default TransitLOSStudies/cache)
_CACHE_DIR = Path(os.environ.get("STUDY_CACHE_DIR") or Path(__file__).resolve().parents[2] / "cache") / "core_boundaries"


def _geocode_admin_polygon(query: str, limit: int = 10, retries: int = 4) -> gpd.GeoDataFrame:
    """Geocode `query` via Nominatim, keeping only the first Polygon/MultiPolygon hit.

    Mirrors `UrbanAccessAnalyzer.utils.get_city_geometry`'s request shape
    (same endpoint/params), but requests several candidates instead of just
    the top one and skips any that come back as a bare `Point` (a place
    marker, not a boundary) -- see module docstring for why that matters.

    A successful result is cached on disk (`core_boundaries/`) and retries are made on empty / point-only / failed responses
    (Nominatim is rate limited and sometimes answers differently under load): the core / metro split of a city must not change
    from one run to the next.
    """
    import hashlib
    import time

    cache = _CACHE_DIR / (hashlib.sha1(query.encode("utf-8")).hexdigest()[:16] + ".geojson")
    if cache.is_file():
        return gpd.read_file(cache)
    last: Exception = ValueError(f"No boundary polygon (only point/no results) found for {query!r}.")
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "jsonv2", "polygon_geojson": 1, "limit": limit},
                headers={"User-Agent": "transitLOS"},
                timeout=30,
            )
            response.raise_for_status()
            for result in response.json():
                geojson = result.get("geojson")
                if geojson and geojson.get("type") in ("Polygon", "MultiPolygon"):
                    gdf = gpd.GeoDataFrame({"display_name": [result["display_name"]]}, geometry=[shape(geojson)], crs="EPSG:4326")
                    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    gdf.to_file(cache, driver="GeoJSON")
                    return gdf
        except requests.RequestException as exc:
            last = exc
        time.sleep(5 * attempt)
    gdf = _overpass_admin_polygon(query)
    if gdf is not None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        gdf.to_file(cache, driver="GeoJSON")
        return gdf
    raise last


def _overpass_admin_polygon(query: str):
    """Fallback when Nominatim only knows the place as a point: the administrative boundary (admin_level 8, else 7 / 9) that
    contains the point and carries the same name, fetched through Overpass + Nominatim's relation lookup. None if not found."""
    import time

    for attempt in range(3):  # Overpass / Nominatim rate-limit now and then
        out = _overpass_admin_polygon_once(query)
        if out is not None:
            return out
        time.sleep(10 * (attempt + 1))
    return None


def _overpass_admin_polygon_once(query: str):
    try:
        hit = requests.get("https://nominatim.openstreetmap.org/search", params={"q": query, "format": "jsonv2", "limit": 1},
                           headers={"User-Agent": "transitLOS"}, timeout=30).json()
        if not hit:
            return None
        lat, lon, name = float(hit[0]["lat"]), float(hit[0]["lon"]), hit[0]["display_name"].split(",")[0].strip().lower()
        q = f'[out:json][timeout:60]; is_in({lat},{lon})->.a; rel(pivot.a)["boundary"="administrative"]; out tags;'
        rels = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=90,
                             headers={"User-Agent": "transitLOS"}).json()["elements"]
        named = [r for r in rels if r.get("tags", {}).get("name", "").lower() == name and str(r["tags"].get("admin_level")) in ("7", "8", "9")]
        if not named:
            return None
        rel = sorted(named, key=lambda r: abs(int(r["tags"]["admin_level"]) - 8))[0]
        geo = requests.post("https://overpass-api.de/api/interpreter", data={"data": f"[out:json][timeout:90]; rel({rel['id']}); out geom;"},
                            timeout=120, headers={"User-Agent": "transitLOS"}).json()["elements"][0]
        from shapely.geometry import LineString
        from shapely.ops import linemerge, polygonize, unary_union

        lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]]) for m in geo.get("members", [])
                 if m.get("type") == "way" and m.get("role") in ("outer", "") and len(m.get("geometry", [])) >= 2]
        polys = list(polygonize(linemerge(lines)))
        if polys:
            return gpd.GeoDataFrame({"display_name": [f"{rel['tags'].get('name')} (OSM relation {rel['id']})"]},
                                    geometry=[unary_union(polys)], crs="EPSG:4326")
    except Exception:  # noqa: BLE001 -- the caller raises its own error
        return None
    return None


def resolve_core_boundary(
    city_name: str,
    is_us: bool,
    state: str | None = None,
    year: int = 2023,
    geocode_name: str | None = None,
    core_file: "str | Path | None" = None,
) -> gpd.GeoDataFrame:
    """Resolve a city's core-boundary polygon.

    If `core_file` exists (the UN urban centre written by `scripts_un_areas.py`, or a hand-set core) it is returned as is, before any
    Census / Nominatim lookup.

    Args:
        city_name: For US cities, the plain place name (e.g. `"Boston"`) to
            match against `pygris.places`. Ignored for non-US cities when
            `geocode_name` is given.
        is_us: Whether to resolve via Census `place` boundaries (`True`) or
            OSM admin boundaries (`False`).
        state: US state abbreviation/name/FIPS forwarded to `pygris.places`
            as `state`. Required when `is_us=True`.
        year: Vintage year forwarded to `pygris.places`.
        geocode_name: For non-US cities, the geocode query sent to Nominatim
            (e.g. `"Guadalajara, Jalisco, Mexico"` -- `CityConfig.geocode_name`).
            Falls back to `city_name` if not given.

    Returns:
        A single-row (or matching subset) GeoDataFrame, EPSG:4326, of the
        city-core boundary.

    Raises:
        ValueError: If `is_us=True` and no `place` matches `city_name`, if
            `state` is missing for a US city, or if no polygon boundary can
            be geocoded for a non-US city.
    """
    if core_file is not None and Path(core_file).is_file():
        return gpd.read_file(core_file).to_crs(4326).reset_index(drop=True)
    if is_us:
        if state is None:
            raise ValueError("state is required to resolve a US city's Census place boundary.")
        places = GEOMETRY_FUNCS["place"](state=state, year=year)
        match = places[places["NAME"].str.lower() == city_name.strip().lower()]
        if match.empty:
            match = places[places["NAME"].str.lower().str.contains(city_name.strip().lower())]
        if match.empty:
            raise ValueError(f"No Census place found matching {city_name!r} in state {state!r}.")
        return match.to_crs(4326).reset_index(drop=True)

    return _geocode_admin_polygon(geocode_name or city_name).to_crs(4326)
