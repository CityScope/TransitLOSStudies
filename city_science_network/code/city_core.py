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

import geopandas as gpd
import requests
from pycensus.countries.usa.constants import GEOMETRY_FUNCS
from shapely.geometry import shape


def _geocode_admin_polygon(query: str, limit: int = 10) -> gpd.GeoDataFrame:
    """Geocode `query` via Nominatim, keeping only the first Polygon/MultiPolygon hit.

    Mirrors `UrbanAccessAnalyzer.utils.get_city_geometry`'s request shape
    (same endpoint/params), but requests several candidates instead of just
    the top one and skips any that come back as a bare `Point` (a place
    marker, not a boundary) -- see module docstring for why that matters.
    """
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "jsonv2", "polygon_geojson": 1, "limit": limit},
        headers={"User-Agent": "transitLOS"},
        timeout=15,
    )
    response.raise_for_status()
    results = response.json()
    for result in results:
        geojson = result.get("geojson")
        if geojson and geojson.get("type") in ("Polygon", "MultiPolygon"):
            geom = shape(geojson)
            return gpd.GeoDataFrame(
                {"display_name": [result["display_name"]]}, geometry=[geom], crs="EPSG:4326"
            )
    raise ValueError(f"No boundary polygon (only point/no results) found for {query!r}.")


def resolve_core_boundary(
    city_name: str,
    is_us: bool,
    state: str | None = None,
    year: int = 2023,
    geocode_name: str | None = None,
) -> gpd.GeoDataFrame:
    """Resolve a city's core-boundary polygon.

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
