"""Transit level of service for one area anywhere in the world, from public data.

Choose an area (a place name, your own file, or a shape drawn on a map), the day and the hours, and the transit feeds (found
online, or your own), run, and get one zip with:

* `streets.parquet`        every street segment with its transit level of service;
* `h3_res<N>.parquet`      H3 hexagons at every resolution, with population, level of service and the census columns;
* `census_<level>.parquet` the census at its finest level, with the same columns.

Every input comes from the most local source that covers the area (tables `*_BY_PLACE` below): the census of the region
or the country when pyCensus has one (e.g. the Basque Country, Spain, USA, Canada, Mexico, Germany, Israel, Andorra, Taiwan,
Chile; the results then have no WorldPop columns), otherwise WorldPop; transit feeds from the country's official portal (US
National Transit Database, Spain's National Access Point), otherwise the global catalogs (Mobility Database, Transitland);
municipal boundaries from the country (US Census places), otherwise OpenStreetMap. Streets come from OpenStreetMap (Geofabrik).

Run it with the settings below:

    python example.py

or from Python / the notebook:

    import example
    aoi = example.area_from_place("Cambridge, Massachusetts, USA", "municipality")
    zip_path = example.run(aoi, day="saturday", start_time="08:00", end_time="20:00")
"""

from __future__ import annotations

import json
import os
import math
import re
import shutil
import sys
import zipfile
from datetime import date as _date, time
from pathlib import Path
from typing import Optional, Sequence, Union

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))   # the shared study code (TransitLOSStudies/tools/studytools)

import geopandas as gpd  # noqa: E402

# ============================== SETTINGS ==============================
# The area
PLACE = "Cambridge, Massachusetts, USA"   # a place name, found with OpenStreetMap
AREA = "municipality"                      # "metro_area" (UN functional urban area), "urban_core" (UN urban centre) or "municipality"
AOI_FILE = None                        # your own area file (.gpkg, .geojson, .shp, ...): used instead of PLACE

# When
DAY = "weekday"                        # "weekday" (a working day), "saturday" or "sunday"
START_TIME = "06:00"                   # start of the hours analysed
END_TIME = "22:00"                     # end of the hours analysed

# Transit feeds
FEEDS = "download"                     # "download": find the area's feeds online
                                       # or a folder with your GTFS feeds (zips or unzipped folders)
                                       # or a list of feeds: zip files, folders or download links
FEED_SOURCES = None                    # where "download" looks; None = the most local sources (FEED_SOURCES_BY_PLACE),
                                       # then the global ones for the feeds they miss; e.g. ["mobility_database", "transitland"]

# Output
NAME = None                            # name of the results folder and zip; None = taken from the place or file name
OUTPUT_DIR = HERE / "areas"            # one folder per area
CACHE_DIR = None                       # downloads shared by every run; None = TransitLOSStudies/cache
BUILD_MAP = True                       # build the interactive map (map.html, finest hexagons, circles and census); False = only the zip
# ======================================================================

AREAS = ("municipality", "urban_core", "metro_area")
#: other names for the same area types ("functional area" is the UN name of the metro area)
AREA_ALIASES = {"functional_area": "metro_area", "functional_urban_area": "metro_area", "fua": "metro_area", "urban_centre": "urban_core",
                "urban_center": "urban_core", "city": "municipality"}
DAYS = ("weekday", "saturday", "sunday")
_UN_KIND = {"urban_core": "urban_centre", "metro_area": "functional_urban_area"}

#: Where each input comes from, most local first. Keys are an ISO 3166-2 region ("ES-PV"), a country ("ES") or "*" (anywhere). Census and
#: boundaries: the first source that covers the area. Feeds: every source in order, a later one only adding feeds the earlier ones miss
#: (a copy of an earlier feed is dropped). Add a better local source by putting its line above the more general ones.
FEED_SOURCES_BY_PLACE = [
    ("US", ["ntd"]),                                   # US National Transit Database (official, by state)
    ("ES", ["nap"]),                                   # Spain's National Access Point (official, by province; free key)
    ("*", ["mobility_database", "transitland"]),       # global catalogs
]
CENSUS_BY_PLACE = [
    ("ES-PV", "euskadi"),                              # Basque Country: Eustat, its own statistics office
    ("*", None),                                       # the country's census in pyCensus, if it has one; else WorldPop
]
MUNICIPALITY_BY_PLACE = [
    ("US", "us_census_place"),                         # US Census place boundaries (TIGER)
    ("*", "openstreetmap"),                            # OpenStreetMap administrative boundary
]

#: a feed whose coverage is wider than this (km across) is regional or intercity: not ticked by default in `find_feeds`
COVERAGE_MAX_KM = 250

#: API keys this study may ask for: (what needs it, where to sign up, environment variable)
API_KEYS = {
    "transitland": ("Transitland (transit feeds)", "https://www.transit.land (create an account, then an API key)", "TRANSITLAND_API_KEY"),
    "nap_es": ("Spain's National Access Point (transit feeds)", "https://nap.transportes.gob.es (register, then the API key in your profile)",
               "NAP_API_KEY"),
    "mobility_database": ("the Mobility Database (transit feeds)", "https://mobilitydatabase.org/sign-up (create an account, then copy your refresh token from the account page)",
                          "MOBILITY_DATABASE_REFRESH_TOKEN"),
    "us_census": ("the US Census Bureau API (US census data)", "https://api.census.gov/data/key_signup.html (a free key, sent by email)",
                  "US_CENSUS_API_KEY"),
    "destatis_genesis": ("the German census (Destatis GENESIS-Online)", "https://genesis.destatis.de/datenbank/online/statistics#modal=web-service-api (register a free GENESIS-Online account, then create a token)",
                         "DESTATIS_GENESIS_TOKEN"),
}


# ---------------------------------------------------------------- the area
def area_from_place(place: str, area: str = "municipality") -> gpd.GeoDataFrame:
    """The area of a place name.

    Args:
        place: Place name, e.g. "Cambridge, Massachusetts, USA".
        area: "municipality" (its administrative boundary: the most local source in `MUNICIPALITY_BY_PLACE`), "urban_core"
            (the UN urban centre that contains it) or "metro_area" (the UN functional urban area: the urban centre plus its
            commuting zone).
    """
    from studytools.city_core import _geocode_admin_polygon

    area = AREA_ALIASES.get(area.strip().lower().replace(" ", "_"), area.strip().lower().replace(" ", "_"))
    if area not in AREAS:
        raise ValueError(f"area must be one of {AREAS} (or {tuple(AREA_ALIASES)}), not {area!r}")
    osm = _geocode_admin_polygon(place).to_crs(4326)
    point = osm.union_all().representative_point()
    if area == "municipality":
        codes = place_codes(gpd.GeoDataFrame(geometry=[point], crs=4326))
        for source in by_place(MUNICIPALITY_BY_PLACE, codes):
            if source == "us_census_place":
                try:
                    from studytools.city_core import resolve_core_boundary

                    state = next(r.split("-")[1] for r in codes["regions"] if r.startswith("US-"))
                    return _one_area(resolve_core_boundary(place.split(",")[0].strip(), True, state=state), place)
                except Exception as exc:  # noqa: BLE001 -- not a Census place: the next source
                    print(f"[area] US Census place not found ({exc}); using OpenStreetMap", flush=True)
            elif source == "openstreetmap":
                return _one_area(osm, place)
    from pycensus.census.worldwide import un_metro_areas

    found = un_metro_areas.find(point=(point.x, point.y), kind=_UN_KIND[area])
    if found.empty:
        raise ValueError(f"No UN {area.replace('_', ' ')} contains {place!r}. Use area='municipality' instead.")
    return _one_area(found.head(1), f"{found['name'].iloc[0]} ({area.replace('_', ' ')})")


def area_from_file(path) -> gpd.GeoDataFrame:
    """The area in a file (.gpkg, .geojson, .shp, .kml, ...): all its shapes together."""
    return _one_area(gpd.read_file(path), Path(path).stem)


def area_from_geojson(geojson) -> gpd.GeoDataFrame:
    """The area of a GeoJSON geometry or feature (dict), e.g. a shape drawn on the notebook map."""
    from shapely.geometry import shape

    geom = geojson.get("geometry", geojson) if isinstance(geojson, dict) else geojson
    return _one_area(gpd.GeoDataFrame(geometry=[shape(geom)], crs=4326), "drawn area")


def _one_area(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    gdf = gdf.to_crs(4326) if gdf.crs else gdf.set_crs(4326)
    return gpd.GeoDataFrame({"name": [label]}, geometry=[gdf.union_all()], crs=4326)


def area_km2(aoi: gpd.GeoDataFrame) -> float:
    """Area in km²."""
    return float(aoi.to_crs("EPSG:6933").area.sum() / 1e6)


# ---------------------------------------------------------------- API keys
def api_key(name: str) -> Optional[str]:
    """The API key `name`; when it is missing, says where to sign up and asks for it (Enter skips). A key typed in is saved in
    `~/.config/ptlos/api_keys.json`, so it is asked only once."""
    import getpass

    if name == "destatis_genesis":   # census keys: pyCensus; transit-feed keys: pyGTFSHandler (each package has its own lookup)
        from pycensus.api_keys import STANDARD_PATH, resolve_api_key, save_api_key
    else:
        from pyGTFSHandler.downloaders.utils.config import STANDARD_PATH, save_api_key
        from pyGTFSHandler.downloaders.utils.config import get_api_key as _get

        def resolve_api_key(n, env):
            return _get(n, None, env)

    what, url, env = API_KEYS[name]
    key = resolve_api_key(name, env)
    if key:
        return key
    print(f"{what} needs an API key. Get one for free at {url}", flush=True)
    if not (sys.stdin and sys.stdin.isatty()) and "ipykernel" not in sys.modules:
        print(f"Not asked (no terminal): skipped. Add it to {STANDARD_PATH} or set the environment variable {env}.", flush=True)
        return None
    key = getpass.getpass("Paste the key here, or press Enter to skip: ").strip()
    if key:
        save_api_key(name, key)
        print(f"Saved in {STANDARD_PATH}", flush=True)
    return key or None


# ---------------------------------------------------------------- census and inputs
def check_census_api(country: str) -> None:
    """For the USA: stop early, with what to do, when the Census Bureau API does not answer with data. Without a (valid) key it answers with an
    HTML page, and the census would silently be left out of the results."""
    if country != "USA":
        return
    import requests
    from pycensus.api_keys import resolve_api_key

    key = resolve_api_key("us_census", "US_CENSUS_API_KEY")
    try:
        r = requests.get("https://api.census.gov/data/2023/acs/acs5", params={"get": "NAME", "for": "state:25", **({"key": key} if key else {})}, timeout=60)
        r.json()
    except Exception as exc:  # noqa: BLE001
        problem = "no key was found" if not key else "the key was not accepted" if "json" in type(exc).__name__.lower() else f"it could not be reached ({exc})"
        raise RuntimeError(
            f"The US Census Bureau API did not return census data: {problem}. Get a free key at https://api.census.gov/data/key_signup.html (it "
            f"arrives by email). In Colab add it as a Secret named US_CENSUS_API_KEY (key icon in the left bar, allow notebook access) and run "
            f"this step again; on your own machine set the environment variable US_CENSUS_API_KEY or save it in ~/.config/ptlos/api_keys.json "
            f"as \"us_census\".") from None


#: the keys each census source needs: (source key, country or None for any) -> key names of `API_KEYS`
SOURCE_KEYS = {("acs5", None): ["us_census"], ("dhc", None): ["us_census"], ("census", "DEU"): ["destatis_genesis"]}


def keys_needed(aoi: gpd.GeoDataFrame, sources: Sequence[str]) -> list:
    """The API keys the ticked census `sources` need, and whether each is already there: `[{name, what, url, env, have, sources}]`. The sources
    are the keys of `census_sources(aoi)`; the notebook asks for the missing ones."""
    from pycensus.api_keys import resolve_api_key

    iso3 = place_codes(aoi)["iso3"]
    need: dict = {}
    for (src, country), names in SOURCE_KEYS.items():
        if src in sources and country in (None, iso3):
            for n in names:
                need.setdefault(n, []).append(src)
    return [{"name": n, "what": API_KEYS[n][0], "url": API_KEYS[n][1].split(" ")[0], "help": API_KEYS[n][1], "env": API_KEYS[n][2],
             "have": bool(resolve_api_key(n, API_KEYS[n][2])), "sources": src, "optional": False, "required": True} for n, src in need.items()]


#: the key each feed source needs, and whether it is optional (the source then still works from its public catalog)
FEED_KEYS = {"nap": ("nap_es", False), "transitland": ("transitland", False), "mobility_database": ("mobility_database", True)}


def feed_keys_needed(aoi: gpd.GeoDataFrame, sources: Optional[Sequence[str]] = None) -> list:
    """The API keys of the transit-feed sources that will be searched for the area (`sources`, or those of `FEED_SOURCES_BY_PLACE`), and whether
    each is there: `[{name, what, url, help, env, have, sources, optional}]`. The notebook asks for the missing ones before the feeds are searched."""
    from pycensus.api_keys import resolve_api_key

    names = list(sources) if sources else [x for tier in by_place(FEED_SOURCES_BY_PLACE, place_codes(aoi)) for x in tier]
    out = []
    for src in dict.fromkeys(names):
        if src in FEED_KEYS:
            n, optional = FEED_KEYS[src]
            out.append({"name": n, "what": API_KEYS[n][0], "url": API_KEYS[n][1].split(" ")[0], "help": API_KEYS[n][1], "env": API_KEYS[n][2],
                        "have": bool(resolve_api_key(n, API_KEYS[n][2])), "sources": [src], "optional": optional, "required": False})
    return out


def save_key(name: str, key: str) -> dict:
    """Save a key the user typed in the notebook (`~/.config/ptlos/api_keys.json`, read by every package) after a quick check where one exists:
    `{ok, message}`."""
    from pycensus.api_keys import save_api_key

    key = key.strip()
    if not key:
        return {"ok": False, "message": "No key typed."}
    if name == "us_census":
        import requests

        try:
            requests.get("https://api.census.gov/data/2023/acs/acs5", params={"get": "NAME", "for": "state:25", "key": key}, timeout=60).json()
        except Exception:  # noqa: BLE001
            return {"ok": False, "message": "The Census Bureau did not accept this key (check it was copied whole and activated from the email)."}
    save_api_key(name, key)
    return {"ok": True, "message": "Saved."}


def place_codes(aoi: gpd.GeoDataFrame, samples: int = 9) -> dict:
    """Where the area is: `{"iso2", "iso3", "regions": ISO 3166-2 codes, "provinces": names}` from points spread over it."""
    import numpy as np
    from geohierarchy.geocoding import reverse_geocode
    from shapely import contains_xy

    geom = aoi.union_all()
    pts = [geom.representative_point()]
    minx, miny, maxx, maxy = geom.bounds
    n = int(np.ceil(np.sqrt(samples)))
    xs, ys = np.meshgrid(np.linspace(minx, maxx, n + 2)[1:-1], np.linspace(miny, maxy, n + 2)[1:-1])
    inside = contains_xy(geom, xs.ravel(), ys.ravel())
    from shapely.geometry import Point
    pts += [Point(x, y) for x, y in zip(xs.ravel()[inside], ys.ravel()[inside])]
    out = {"iso2": None, "iso3": None, "regions": [], "provinces": []}
    for pt in pts:
        a = reverse_geocode(pt.y, pt.x)
        if not a.get("country_code"):
            continue
        out["iso2"] = out["iso2"] or a["country_code"].upper()
        for k, key in (("regions", "ISO3166-2-lvl4"), ("provinces", "province")):
            v = a.get(key)
            if v and v not in out[k]:
                out[k].append(v)
    if out["iso2"]:
        import pycountry

        out["iso3"] = pycountry.countries.get(alpha_2=out["iso2"]).alpha_3
    return out


def by_place(table, codes: dict) -> list:
    """The values of `table` ([(place key, value), ...], most local first) that apply to the area, in order."""
    keys = set(codes["regions"]) | ({codes["iso2"]} if codes["iso2"] else set()) | {"*"}
    return [value for key, value in table if key in keys]


def area_config(aoi: gpd.GeoDataFrame, name: str, osm_folder: Path, worldpop="auto"):
    """`(configuration, place codes)` of the area: its country, which census to use (the most local source), its OpenStreetMap
    extract and WorldPop raster; and where it is (`place_codes`). `worldpop`: "auto" = WorldPop columns only where pyCensus has no census for the
    country; True = also next to a census (the real WorldPop raster is used then); False = never (a country without census still needs it)."""
    from studytools.config import CityConfig, _census_supported
    from studytools.inputs import osm_extract, worldpop_filename

    codes = place_codes(aoi)
    iso3 = codes["iso3"]
    if not iso3:
        raise ValueError("Could not find the country of the area.")
    census_module = next(iter(by_place(CENSUS_BY_PLACE, codes)), None)   # the most local census source (None = the country's)
    if iso3 == "DEU":
        api_key("destatis_genesis")
    states = _us_states(aoi) if iso3 == "USA" else None
    pbf = osm_extract(aoi, osm_folder)
    has_census = _census_supported(iso3, census_module)
    config = CityConfig(key=name, display_name=str(aoi["name"].iloc[0]), geocode_name=str(aoi["name"].iloc[0]), country=iso3,
                        census_states=states, pbf_filename=pbf.name, worldpop_filename=worldpop_filename(iso3),
                        census_module=census_module, worldpop_layers=(worldpop is True) or not has_census,   # with a census: no WorldPop columns unless asked
                        map_finest_only=True)   # the map: finest H3 resolution (hexagons, circles) and finest census level only
    return config, codes


def census_used(config) -> str:
    """Which population source the run will use, in words."""
    if config.uses_census:
        return f"census of {config.census_module or config.country}" + (f" ({', '.join(config.census_states)})" if config.census_states else "")
    return f"WorldPop (pyCensus has no census for {config.country})"


def _us_states(aoi: gpd.GeoDataFrame) -> list:
    """US states (two-letter codes) the area touches."""
    import pygris

    states = pygris.states(cb=True, year=2023, cache=True).to_crs(4326)
    return sorted(states.loc[states.intersects(aoi.union_all()), "STUSPS"].tolist())


# ---------------------------------------------------------------- transit feeds
def get_feeds(aoi: gpd.GeoDataFrame, gtfs_dir: Path, feeds: Union[str, Path, Sequence] = "download",
              sources: Optional[Sequence[str]] = None, place: Optional[dict] = None, states: Optional[Sequence[str]] = None,
              only: Optional[Sequence[str]] = None) -> list:
    """The GTFS feeds of the run in `gtfs_dir` (one folder per feed). Feeds without stops in the area and duplicate copies of the same
    feed are left out. Returns the feed folders.

    Args:
        aoi: The area.
        gtfs_dir: Folder the feeds go to.
        feeds: "download" (find them online), a folder with GTFS feeds (zips or unzipped folders), or a list of zip files, folders and
            download links.
        sources: For "download": the sources to use ("ntd", "nap", "mobility_database", "transitland"). None = the sources of
            `FEED_SOURCES_BY_PLACE`, most local first, each later one only adding the feeds the earlier ones miss.
        place: Where the area is (`place_codes`); found when missing.
        states: US states of the area (for "ntd").
        only: For "download": the keys (`find_feeds`) of the only feeds to download; None = all that the sources list.
    """
    gtfs_dir = Path(gtfs_dir)
    choice = (json.dumps(feeds if isinstance(feeds, str) else [str(f) for f in feeds]) + json.dumps(list(sources or [])) + "|local+missing+operators"
              + (json.dumps(sorted(only)) if only is not None else ""))
    marker = gtfs_dir / "_feeds_choice.json"
    if marker.is_file() and marker.read_text() == choice and any(gtfs_dir.glob("*/stops.txt")):
        return sorted(p.parent for p in gtfs_dir.glob("*/stops.txt"))   # same choice as the last run: reuse
    if gtfs_dir.exists():
        shutil.rmtree(gtfs_dir)
    gtfs_dir.mkdir(parents=True)
    if isinstance(feeds, str) and feeds == "download":
        place = place or place_codes(aoi)
        tiers = [list(sources)] if sources else by_place(FEED_SOURCES_BY_PLACE, place)
        kept, origin = [], {}
        for tier in tiers:   # most local first; each later (more general) source only adds the feeds the earlier ones miss
            added = _keep_feeds_in_area(download_feeds(aoi, gtfs_dir, tier, place, states, only), aoi, keep=kept)
            print(f"[gtfs] {', '.join(tier)}: {len(added)} feeds" + (" not already covered" if kept else ""), flush=True)
            origin.update({p.name: ", ".join(tier) for p in added})
            kept += added
        (gtfs_dir / "_feeds_sources.json").write_text(json.dumps(origin))
    else:
        items = sorted(Path(feeds).iterdir()) if isinstance(feeds, (str, Path)) else list(feeds)
        kept = _keep_feeds_in_area([f for f in (_add_feed(item, gtfs_dir) for item in items) if f is not None], aoi)
    marker.write_text(choice)
    print(f"[gtfs] {len(kept)} feeds with stops in the area: {', '.join(p.name for p in kept)}", flush=True)
    return kept


def _search_source(source: str, aoi: gpd.GeoDataFrame, place: dict, states: Optional[Sequence[str]]):
    """`(client, found)` of one feed source: its downloader and the feeds it lists for the area (metadata only, nothing downloaded), or None
    when it needs an API key that is missing."""
    from pyGTFSHandler.downloaders import MobilityDatabaseCatalog, TransitLandDownloader

    if source == "ntd":
        from pyGTFSHandler.downloaders.usa import NTDDownloader

        client = NTDDownloader()
        return client, client.search_feeds(state=list(states or [r.split("-")[1] for r in place.get("regions", []) if r.startswith("US-")]))
    if source == "nap":
        from pyGTFSHandler.downloaders.spain import NAPDownloader

        key = api_key("nap_es")
        if not key:
            return None
        client = NAPDownloader(api_key=key)
        return client, client.search_feeds(region=place.get("provinces", []), region_type=0)   # 0 = province
    if source == "mobility_database":
        client = MobilityDatabaseCatalog()
        return client, client.search_feeds(aoi=aoi)
    if source == "transitland":
        key = api_key("transitland")
        if not key:
            return None
        client = TransitLandDownloader(api_key=key)
        return client, client.search_feeds(aoi=aoi)
    raise ValueError(f"unknown feed source {source!r}: use 'ntd', 'nap', 'mobility_database' or 'transitland'")


def download_feeds(aoi: gpd.GeoDataFrame, gtfs_dir: Path, sources: Sequence[str], place: Optional[dict] = None,
                   states: Optional[Sequence[str]] = None, only: Optional[Sequence[str]] = None) -> list:
    """Download the area's feeds from `sources`. Returns the feed folders.

    Sources: "ntd" (US National Transit Database, the area's states), "nap" (Spain's National Access Point, the area's provinces;
    asks for a free key), "mobility_database" (Mobility Database catalog, no key), "transitland" (asks for a free key).
    `only`: keys (`"<source>:<feed id>"`, as `find_feeds` gives them) of the only feeds to download; None = all that the sources list.
    """
    place = place or {}
    only = None if only is None else set(only)
    folders = []
    for s in sources:
        try:
            searched = _search_source(s, aoi, place, states)
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 -- one source down must not stop the others
            print(f"[gtfs] {s}: search failed ({exc})", flush=True)
            continue
        if searched is None:
            continue
        client, found = searched
        if only is not None:
            found = [f for f in found if f"{s}:{f.id}" in only]
            if not found:
                continue
        print(f"[gtfs] {s}: {len(found)} feeds", flush=True)
        try:
            folders += [Path(p) for p in client.download_feeds(found, str(gtfs_dir))]
        except Exception:  # noqa: BLE001 -- one at a time, so one broken link does not stop the others
            for feed in found:
                try:
                    folders += [Path(p) for p in client.download_feeds([feed], str(gtfs_dir))]
                except Exception as exc:  # noqa: BLE001
                    print(f"[gtfs] {feed.id}: download failed ({exc})", flush=True)
    unique = list(dict.fromkeys(folders))   # a downloader can return one folder per row (NTD: one row per mode of a feed)
    return [f for f in unique if (f / "stops.txt").is_file()]


def _dig(record: dict, dotted: str, default=None):
    """`record["a.b.c"]` or `record["a"]["b"]["c"]`: the sources return their metadata flat (Mobility Database) or nested (Transitland)."""
    if dotted in record:
        return record[dotted]
    node = record
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _area_names(aoi: gpd.GeoDataFrame) -> set:
    """Lower-case names that identify the area: its place name and the name of the UN urban centre it lies in."""
    names = {str(aoi["name"].iloc[0]).split(",")[0].split("(")[0].strip().lower()}
    try:
        from pycensus.census.worldwide import un_metro_areas

        point = aoi.to_crs(4326).union_all().representative_point()
        found = un_metro_areas.find(point=(point.x, point.y), kind="urban_centre")
        if not found.empty:
            names.add(str(found["name"].iloc[0]).lower())
    except Exception:  # noqa: BLE001
        pass
    return {n for n in names if n}


FEED_DATABASES = {
    "ntd": ("US National Transit Database", "Official list of US transit agencies' feeds, by state.", "US"),
    "nap": ("Spain's National Access Point", "Official feeds of Spanish operators, by province (needs a free key).", "ES"),
    "mobility_database": ("Mobility Database", "Worldwide catalog of GTFS feeds (global).", None),
    "transitland": ("Transitland", "Worldwide catalog of GTFS feeds (global; needs a free key).", None),
}


def feed_databases(aoi: gpd.GeoDataFrame) -> list:
    """The feed databases to search, to choose BEFORE the search: the global ones and those of the area's country, all ticked by default, the
    country's first (most local first: a later one only adds the feeds the earlier ones miss). `[{key, title, description, scope, default}]`."""
    place = place_codes(aoi)
    local = [x for tier in by_place(FEED_SOURCES_BY_PLACE, place) for x in tier if FEED_DATABASES[x][2]]
    glob = [k for k, v in FEED_DATABASES.items() if v[2] is None]
    return [{"key": k, "title": FEED_DATABASES[k][0], "description": FEED_DATABASES[k][1], "scope": FEED_DATABASES[k][2] or "global", "default": True}
            for k in dict.fromkeys(local + glob)]


def _stop_time_rows(raw: dict) -> dict:
    """Row counts of the feed's files when the source's metadata lists them (Transitland does): `{stops, routes, stop_times, frequencies}` (missing
    = not listed). stop_times is trips x stops only for a feed without frequencies.txt: with one, a trip's rows stand for many departures, so the
    stop_times count is not used then (`service` shows the stops and routes)."""
    files = _dig(raw, "feed_state.feed_version.files") or _dig(raw, "feed_version.files") or []
    rows = {}
    for f in files if isinstance(files, list) else []:
        if isinstance(f, dict) and f.get("rows") is not None:
            rows[str(f.get("name", "")).lower().removesuffix(".txt")] = int(f["rows"])
    return {k: rows[k] for k in ("stops", "routes", "stop_times", "frequencies") if k in rows}


def find_feeds(aoi: gpd.GeoDataFrame, feeds: Union[str, Path, Sequence] = "download", sources: Optional[Sequence[str]] = FEED_SOURCES) -> list:
    """The feeds the sources list for the area, from their metadata only: nothing is downloaded, so you can choose before downloading.

    Returns one dict per feed (the same feed listed by several sources once, under the most local one): `key`, `keys` (to give to
    `list_feeds(selected=...)`), `source`, `provider`, `name`, `location`, `modes`, `updated`, `url`, `likely` (whether it probably serves the
    area: the source placed it there, or its urbanized area / city matches) and `why`. Empty for your own feeds (`feeds` other than
    "download"): there is nothing to choose before downloading.
    """
    if not (isinstance(feeds, str) and feeds == "download"):
        return []
    place = place_codes(aoi)
    states = _us_states(aoi) if place["iso3"] == "USA" else None
    tiers = [list(sources)] if sources else by_place(FEED_SOURCES_BY_PLACE, place)
    names, entries = _area_names(aoi), {}
    minx, miny, maxx, maxy = aoi.to_crs(4326).total_bounds
    for s in [x for tier in tiers for x in tier]:
        try:
            searched = _search_source(s, aoi, place, states)
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[gtfs] {s}: search failed ({exc})", flush=True)
            continue
        if searched is None:
            continue
        for f in searched[1]:
            raw = f.raw if isinstance(f.raw, dict) else {}
            ntd = raw.get("ntd") if isinstance(raw.get("ntd"), dict) else {}
            box = [_dig(raw, f"location.bounding_box.{k}") for k in ("minimum_longitude", "minimum_latitude", "maximum_longitude", "maximum_latitude")]
            box = [float(v) for v in box] if all(v not in (None, "") for v in box) else None
            location = ", ".join(str(x) for x in (ntd.get("city"), ntd.get("state"), _dig(raw, "location.municipality"),
                                                  _dig(raw, "location.subdivision_name")) if x)
            likely, why = True, f"listed by {s} for the area"
            if s == "ntd":
                uza, city = str(ntd.get("uza_name", "")), str(ntd.get("city", ""))
                likely = city.lower() in names
                same_uza = any(n in uza.lower() for n in names)
                why = (f"based in {city}, in the area" if likely else
                       f"based in {city}, same urbanized area ({uza}); may not serve the area" if same_uza else f"based in {city}, elsewhere")
            else:   # a coverage box or polygon: it must include the area, and not be regional or intercity in size
                shape_ = None
                if box:
                    from shapely.geometry import box as _box

                    shape_ = _box(*box)
                elif isinstance(_dig(raw, "feed_state.feed_version.geometry"), dict):
                    from shapely.geometry import shape as _shape

                    shape_ = _shape(_dig(raw, "feed_state.feed_version.geometry"))
                if shape_ is not None:
                    x0, y0, x1, y1 = shape_.bounds
                    span = ((x1 - x0) * 111.3 * math.cos(math.radians((y0 + y1) / 2))) ** 2
                    span = (span + ((y1 - y0) * 111.3) ** 2) ** 0.5
                    inside = shape_.intersects(aoi.to_crs(4326).union_all())
                    likely = inside and span <= COVERAGE_MAX_KM
                    why = (f"its coverage ({span:,.0f} km across) " + ("includes the area" if inside else "does not include the area")
                           + (" but is regional or intercity in size" if inside and not likely else ""))
            url = str(_dig(raw, "urls.direct_download") or f.download_url or "").strip()   # the agency's own link: the same feed from other sources matches
            key = f"{s}:{f.id}"
            entry = entries.get(url.lower()) if url else None
            if entry:   # the same feed from another source: kept under the first (most local) one, the others noted
                entry["keys"].append(key)
                entry["also"].append(s)
                if ntd.get("mode_name") and ntd["mode_name"] not in entry["modes"]:
                    entry["modes"].append(ntd["mode_name"])
                continue
            entries[url.lower() or key] = {
                "key": key, "keys": [key], "source": s, "also": [], "provider": str(f.provider or "") if str(f.provider) != "None" else "",
                "name": str(f.name or "") if str(f.name) not in ("None", "") else "", "location": location,
                "modes": [ntd["mode_name"]] if ntd.get("mode_name") else [], "url": url,
                "updated": str(ntd.get("new_modified_date") or _dig(raw, "feed_state.feed_version.fetched_at") or "")[:10],
                "likely": bool(likely), "why": why, "stop_times": _stop_time_rows(raw)}
    out = []
    for e in entries.values():
        e["provider"] = e["provider"] or e["name"] or e["key"].split(":", 1)[1]
        if e["provider"].startswith("f-"):   # a Transitland feed id (f-drt3-middlesex~ma~us): made readable
            e["provider"] = e["provider"][2:].replace("~", " ")
        e["modes"] = ", ".join(e["modes"])
        c = e.pop("stop_times")   # row counts of the files listed by the catalog: only stops and routes are shown (stop_times rows miss frequency-based service)
        e["service"] = ", ".join(x for x in (f"{c['stops']:,} stops" if c.get("stops") else "", f"{c['routes']:,} routes" if c.get("routes") else "") if x)
        e["source"] = ", ".join(dict.fromkeys([e["source"], *e.pop("also")]))
        out.append(e)
    return sorted(out, key=lambda e: (not e["likely"], e["provider"].lower()))


def _add_feed(item, gtfs_dir: Path) -> Optional[Path]:
    """Copy (or download) one feed into `gtfs_dir` as an unzipped folder."""
    from studytools.inputs import fetch_file

    s = str(item)
    if s.startswith(("http://", "https://")):
        name = _safe_name(re.sub(r"\.zip.*$", "", s.rstrip("/").split("/")[-1])) or "feed"
        zpath = gtfs_dir / f"{name}.zip"
        fetch_file(s, zpath)
    else:
        p = Path(s).expanduser()
        name = _safe_name(p.stem)
        if p.is_dir():
            if not (p / "stops.txt").is_file():
                print(f"[gtfs] {p}: no stops.txt, skipped", flush=True)
                return None
            shutil.copytree(p, gtfs_dir / name, dirs_exist_ok=True)
            return gtfs_dir / name
        if not (p.is_file() and p.suffix.lower() == ".zip"):
            print(f"[gtfs] {p}: not a GTFS zip or folder, skipped", flush=True)
            return None
        zpath = p
    out = gtfs_dir / name
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    inner = [d for d in out.iterdir() if d.is_dir() and (d / "stops.txt").is_file()]
    if not (out / "stops.txt").is_file() and len(inner) == 1:   # a zip with the feed inside one folder
        for f in inner[0].iterdir():
            shutil.move(str(f), out / f.name)
        inner[0].rmdir()
    if zpath.parent == gtfs_dir:
        zpath.unlink()
    return out if (out / "stops.txt").is_file() else None


def _keep_feeds_in_area(folders: Sequence[Path], aoi: gpd.GeoDataFrame, keep: Sequence[Path] = ()) -> list:
    """Of `folders`, the feeds with at least one stop inside the area that are not copies of a feed in `keep` (the feeds already taken
    from a more local source, which always win) nor of each other. The others are deleted."""
    import pandas as pd
    from pyGTFSHandler.utils.feed_dedupe import drop_duplicate_feeds, find_duplicate_feeds, same_operators
    from shapely import contains_xy

    geom = aoi.union_all()
    keep = list(keep)
    new = []
    for f in dict.fromkeys(folders):
        if f in keep:
            continue
        try:
            s = pd.read_csv(f / "stops.txt", usecols=["stop_lat", "stop_lon"]).dropna()
            inside = bool(contains_xy(geom, s["stop_lon"].to_numpy(float), s["stop_lat"].to_numpy(float)).any())
        except Exception:  # noqa: BLE001 -- unreadable feed
            inside = False
        (new.append(f) if inside else shutil.rmtree(f, ignore_errors=True))
    if keep and new:   # a copy of an earlier feed (same stops and routes, or the same operators) is dropped: the more local source wins
        dup = find_duplicate_feeds(keep + new)
        copies = {f for f in new if f in dup or any(dup.get(k) == f for k in keep) or same_operators(f, keep)}
        for f in copies:
            shutil.rmtree(f, ignore_errors=True)
        new = [f for f in new if f not in copies]
    unique = drop_duplicate_feeds(new)
    for f in set(new) - set(unique):
        shutil.rmtree(f, ignore_errors=True)
    return unique


# ---------------------------------------------------------------- feeds and service dates
def _area_stop_ids(folder: Path, geom) -> set:
    """Ids of the stops of the feed in `folder` that lie inside `geom`."""
    import pandas as pd
    from shapely import contains_xy

    s = pd.read_csv(folder / "stops.txt", usecols=["stop_id", "stop_lat", "stop_lon"], dtype={"stop_id": str}).dropna()
    return set(s.loc[contains_xy(geom, s["stop_lon"].to_numpy(float), s["stop_lat"].to_numpy(float)), "stop_id"])


def _service_table(folder: Path, stop_ids: set):
    """Per service_id of the feed: `trips` that call at a stop of the area and the `events` (stop calls in the area) of those trips."""
    import polars as pl

    trips = pl.scan_csv(folder / "trips.txt", infer_schema_length=0).select("trip_id", "service_id")
    calls = (pl.scan_csv(folder / "stop_times.txt", infer_schema_length=0).select("trip_id", "stop_id")
             .filter(pl.col("stop_id").is_in(list(stop_ids))).group_by("trip_id").len().rename({"len": "events"}))
    return calls.join(trips, on="trip_id").group_by("service_id").agg(pl.len().alias("trips"), pl.col("events").sum().alias("events")).collect()


def _service_days(folder: Path):
    """`(service_id, date)` of every day each service runs (calendar.txt weekdays and date range, then calendar_dates.txt exceptions)."""
    import polars as pl

    days = []
    cal = folder / "calendar.txt"
    if cal.is_file():
        c = pl.read_csv(cal, infer_schema_length=0).with_columns(pl.col("start_date", "end_date").str.strip_chars().str.to_date("%Y%m%d"))
        names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        c = c.with_columns(date=pl.date_ranges("start_date", "end_date")).explode("date").with_columns(dow=pl.col("date").dt.weekday() - 1)
        flags = pl.concat_list([pl.col(n) == "1" for n in names])
        days.append(c.filter(flags.list.get(pl.col("dow"))).select("service_id", "date"))
    exc = folder / "calendar_dates.txt"
    base = pl.concat(days) if days else pl.DataFrame(schema={"service_id": pl.Utf8, "date": pl.Date})
    if exc.is_file():
        e = pl.read_csv(exc, infer_schema_length=0).with_columns(pl.col("date").str.strip_chars().str.to_date("%Y%m%d"))
        base = pl.concat([base.join(e.filter(pl.col("exception_type") == "2").select("service_id", "date"), on=["service_id", "date"], how="anti"),
                          e.filter(pl.col("exception_type") == "1").select("service_id", "date")]).unique()
    return base


def list_feeds(aoi: gpd.GeoDataFrame, feeds: Union[str, Path, Sequence] = "download", name: Optional[str] = None,
               sources: Optional[Sequence[str]] = FEED_SOURCES, selected: Optional[Sequence[str]] = None) -> list:
    """Download the GTFS feeds of the area (as `get_feeds`; kept in `areas/<name>/gtfs_all`) and describe each, to choose from.

    `selected`: the `keys` of the feeds to download, chosen from `find_feeds` (their metadata, before any download); None = all that the
    sources list. Nothing else is downloaded here (the street data comes with `run`).

    Returns one dict per feed: `name` (folder), `agency`, `source`, `stops` (in the area), `trips` and `events` (stop calls) in the
    area per typical day of its calendar, `first_date` and `last_date` of its calendar.
    """
    from studytools import cache as study_cache

    name = _safe_name(name or str(aoi["name"].iloc[0]))
    cache = study_cache.resolve_cache_dir(CACHE_DIR)
    study_cache.apply_env(cache, "example")
    place = place_codes(aoi)
    states = _us_states(aoi) if place["iso3"] == "USA" else None
    gtfs_all = Path(OUTPUT_DIR) / name / "gtfs_all"
    folders = get_feeds(aoi, gtfs_all, feeds, sources, place=place, states=states, only=selected)
    origin = json.loads((gtfs_all / "_feeds_sources.json").read_text()) if (gtfs_all / "_feeds_sources.json").is_file() else {}
    geom = aoi.union_all()
    out = []
    for f in folders:
        info = {"name": f.name, "agency": "", "source": origin.get(f.name, "your feeds"), "stops": 0, "trips": 0, "events": 0,
                "first_date": None, "last_date": None}
        try:
            import polars as pl

            if (f / "agency.txt").is_file():
                names = pl.read_csv(f / "agency.txt", infer_schema_length=0)["agency_name"].drop_nulls().unique(maintain_order=True).to_list()
                info["agency"] = ", ".join(names[:3]) + (f" +{len(names) - 3}" if len(names) > 3 else "")
            ids = _area_stop_ids(f, geom)
            tbl = _service_table(f, ids)
            days = _service_days(f)
            per_day = days.join(tbl, on="service_id").group_by("date").agg(pl.col("trips").sum(), pl.col("events").sum())
            info.update(stops=len(ids), trips=int(per_day["trips"].median() or 0), events=int(per_day["events"].median() or 0),
                        first_date=str(days["date"].min()), last_date=str(days["date"].max()))
        except Exception as exc:  # noqa: BLE001 -- a feed that cannot be described is still offered
            print(f"[gtfs] {f.name}: could not describe ({type(exc).__name__}: {str(exc)[:80]})", flush=True)
        out.append(info)
    return out


def _holidays(aoi: gpd.GeoDataFrame, lo, hi) -> dict:
    """`{"YYYY-MM-DD": name}` of the public holidays of the area's country and region between the dates `lo` and `hi` (empty when the holiday
    service is unreachable)."""
    from datetime import timedelta

    from pyGTFSHandler.utils import date_parsing, geocoding

    centre = aoi.to_crs(4326).union_all().centroid
    try:
        country, region = geocoding.get_country_region(centre.y, centre.x)
    except Exception:  # noqa: BLE001
        return {}
    found = {}
    for year in range(lo.year, hi.year + 1):
        for days, label in date_parsing.get_holidays(year, country, region).select("date", "name").iter_rows():
            found[str(_date(1970, 1, 1) + timedelta(days=days))] = label
    return found


def service_dates(aoi: gpd.GeoDataFrame, feed_names: Sequence[str], day: str = DAY, name: Optional[str] = None) -> dict:
    """Service per date of the chosen feeds in the area, and the date to use.

    Returns `{"dates": [{"date", "weekday", "trips", "events"}...], "suggested", "day"}`: `trips` = trips that call at a stop of the
    area, `events` = stop calls in the area (stops x trips), summed over the feeds, for every date of their calendars within a year of
    today (or the last year of the calendar when it ended earlier), with the name of the public holiday on it, if any. `suggested`: like the
    automatic choice, the middle date of the `day` type (weekday, saturday, sunday), holidays excluded, with at least 95% of the busiest
    such date's `events`, within 60 days of today.
    """
    from datetime import date as _date, timedelta

    import polars as pl

    if day not in DAYS:
        raise ValueError(f"day must be one of {DAYS}, not {day!r}")
    name = _safe_name(name or str(aoi["name"].iloc[0]))
    gtfs_all, geom = Path(OUTPUT_DIR) / name / "gtfs_all", aoi.union_all()
    frames = []
    for f in feed_names:
        folder = gtfs_all / f
        days = _service_days(folder)
        frames.append(days.join(_service_table(folder, _area_stop_ids(folder, geom)), on="service_id")
                      .group_by("date").agg(pl.col("trips").sum(), pl.col("events").sum()))
    if not frames:
        raise ValueError("No feeds chosen.")
    table = pl.concat(frames).group_by("date").agg(pl.col("trips").sum(), pl.col("events").sum()).sort("date")
    today, last = _date.today(), table["date"].max()
    lo, hi = today - timedelta(days=365), today + timedelta(days=365)
    if last < lo:   # calendars that ended long ago: their last year
        lo, hi = last - timedelta(days=365), last
    table = table.filter(pl.col("date").is_between(lo, hi)).with_columns(wd=pl.col("date").dt.weekday() - 1)
    kind = {"weekday": [0, 1, 2, 3, 4], "saturday": [5], "sunday": [6]}[day]
    holidays = _holidays(aoi, table["date"].min(), table["date"].max())
    working = table.filter(~pl.col("date").cast(pl.Utf8).is_in(list(holidays)))   # public holidays are never suggested
    near = working.filter(pl.col("wd").is_in(kind) & pl.col("date").is_between(today - timedelta(days=60), today + timedelta(days=60)))
    pool = near if near.height else working.filter(pl.col("wd").is_in(kind))
    if pool.is_empty():
        raise ValueError(f"The chosen feeds have no {day} service.")
    top = pool.filter(pl.col("events") >= 0.95 * pool["events"].max()).sort("date")
    suggested = top["date"][top.height // 2]
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {"dates": [{"date": str(d), "weekday": names[w], "trips": int(t), "events": int(e), "holiday": holidays.get(str(d), "")}
                      for d, t, e, w in table.iter_rows()], "suggested": str(suggested), "day": day}



# ---------------------------------------------------------------- census columns
#: the census data sources, one group per download unit: key -> (title, what it holds, prefixes of its result columns). The groups of the
#: notebook's choosers (before the run, what to download; after it, which columns to keep).
SOURCES = {
    "acs5": ("ACS 5-year estimates (USA)", "US Census Bureau American Community Survey, block groups: population, age, income, housing, education, commuting, jobs, health.",
             ("acs5_", "acs_")),
    "dhc": ("Decennial census 2020 (USA)", "US Census Bureau, blocks: the population of every block (the base of the population map), race and ethnicity, age, housing units.",
            ("dhc_",)),
    "lodes": ("Jobs by workplace (LODES, USA)", "US LEHD LODES: jobs by workplace.", ("lodes_",)),
    "religion": ("Religion census (USA)", "US Religion Census 2020, counties: congregations and adherents by denomination.", ("usrc_",)),
    "housing_history": ("Housing, earlier periods (USA)", "Housing tenure, rent and home values for earlier census periods (tracts).", ("ushs_",)),
    "elections": ("Elections (USA)", "Election results: precinct results resampled to tracts, and the last national election by county.", ("elec_",)),
    "buildings": ("Buildings (GlobalBuildingAtlas)", "Building footprints, heights and built floor area, and the built-surface population estimate.", ("gba_",)),
    "worldpop_grid": ("WorldPop population grid", "WorldPop's modelled population on a 100 m grid, the base of the population map where the census is too coarse.", ()),
    "worldpop_layers": ("WorldPop age, sex and urban layers", "WorldPop age, sex and urban-share layers of the whole country.", ("worldpop",)),
    "census": ("National census", "The country's own census columns.", ()),
    "derived": ("Derived indicators", "Shares, rates and harmonised names made by this study from the sources above.", ()),
}
#: the USA sources that can be left out of the download (the buildings are used whenever needed)
_OPTIONAL_USA = ("dhc", "acs5", "lodes", "religion", "housing_history", "elections")


def census_sources(aoi: gpd.GeoDataFrame) -> list:
    """The census data sources for the area, to choose from BEFORE the run downloads them (unticked ones are never downloaded).

    Returns one dict per source: `key`, `title`, `description`, `default` (ticked at the start), `note`, and `forced` (a rule, never a lock: the
    source is ticked and cannot be unticked while the rule holds). Nothing is locked: the census population (decennial blocks in the USA, the
    national census elsewhere) is ticked by default where the census is sub-municipal, and when it is unticked WorldPop's population grid is
    ticked automatically (`replaces`); where the country has no census, WorldPop's grid is the only population source and stays ticked.
    The buildings (GlobalBuildingAtlas) are not listed: they are used whenever the run needs them (the built-surface population, resampling).
    """
    from studytools.config import _census_supported

    iso3 = place_codes(aoi)["iso3"]
    census_module = next(iter(by_place(CENSUS_BY_PLACE, place_codes(aoi))), None)
    has_census = _census_supported(iso3, census_module)

    def row(key, default=True, note="", **kw):
        return {"key": key, "title": SOURCES[key][0], "description": SOURCES[key][1], "default": default, "note": note, "required": False, **kw}

    rows = []
    if iso3 == "USA" and has_census:
        rows = [row("dhc"), row("acs5"), row("lodes"), row("religion"), row("housing_history"), row("elections")]
    elif has_census:
        rows = [row("census")]
    if has_census:
        pop = rows[0]["key"]
        rows.append(row("worldpop_grid", default=False, replaces=pop,
                        note="used for the population when the census population is unticked, and where the census areas are large (about 1.5 GB for the USA)"))
        rows.append(row("worldpop_layers", default=False, note="a one-time download of tens of GB for the USA"))
    else:
        rows.append(row("worldpop_grid", forced=True, note="the country has no census here: WorldPop is the population source"))
        rows.append(row("worldpop_layers", forced=True, note="no census here: WorldPop's layers give age, sex and urban share"))
    return rows


_census_catalog = census_sources   # the function under a name `run()` can still reach


def column_group(column: str, bare: dict) -> Optional[str]:
    """The source group (a key of `SOURCES`) of a result column, None when it is always kept (ids, geometry, population, level of service,
    densities). A column takes the source of its name's prefix; a bare name (`medianAge`) the source the run recorded for it (`bare`);
    the rest are derived indicators."""
    if column in ("geometry", "h3_cell", "GEOID", "population", "level_of_service", "area_m2") or column.startswith(("pop_", "population_density", "__")):
        return None
    for key, (_, _, prefixes) in SOURCES.items():
        if column.startswith(prefixes) if prefixes else False:
            return key
    if re.match(r"^\d{4}_", column):
        return "elections"
    if column in bare:
        for key, (_, _, prefixes) in SOURCES.items():
            if prefixes and bare[column].startswith(prefixes):
                return key
    return "derived"


def _metadata(area_dir: Path) -> dict:
    """`{column: entry}` of the run's `census_metadata.json` (name, source, level, unit, description of each census column)."""
    f = area_dir / "results" / "census_metadata.json"
    return {e["column"]: e for e in json.loads(f.read_text())} if f.is_file() else {}


def _words(column: str) -> str:
    """A readable name from a column name: `acs_car_free_rate` -> "Acs car free rate", `medianAge` -> "Median age"."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", column).replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def _short(text: str, n: int = 260) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "..."


def result_columns(aoi: gpd.GeoDataFrame, name: Optional[str] = None) -> dict:
    """The census columns of the results by source group, with what each is: `{group title: {"description", "columns": [{"column", "name",
    "description", "unit", "level", "source", "in"}]}}`. `level` = the census level the source publishes it at ("derived" for columns made
    by this study); `in` = the result tables that hold it. Descriptions: the run's census metadata, else the global schema's definition."""
    import pyarrow.parquet as pq

    area_dir = Path(OUTPUT_DIR) / _safe_name(name or str(aoi["name"].iloc[0]))
    definition = {e["feature_name"]: e.get("definition", "") for e in json.loads(
        (Path(__import__("pycensus").__file__).parent / "census" / "global_schema.json").read_text())}
    meta, tables = _metadata(area_dir), {}
    bare_file = area_dir / "results" / "bare_column_sources.json"
    bare = json.loads(bare_file.read_text()) if bare_file.is_file() else {}
    for arc, src in _result_tables(area_dir).items():
        if arc != "streets.parquet" and src.is_file():
            tables[arc] = pq.ParquetFile(src).schema_arrow.names
    h3 = sorted(int(re.match(r"h3_res(\d+)", t).group(1)) for t in tables if t.startswith("h3_res"))

    def where(c: str) -> str:   # the result tables that hold column c
        parts = []
        if any(c in tables[f"h3_res{r}.parquet"] for r in h3):
            parts.append("H3 res " + ", ".join(str(r) for r in h3 if c in tables[f"h3_res{r}.parquet"]))
        parts += [t[:-8].replace("_", " ") for t in tables if t.startswith("census_") and c in tables[t]]
        return "; ".join(parts)

    groups: dict = {}
    for c in dict.fromkeys(c for names in tables.values() for c in names):
        key = column_group(c, bare)
        if not key:
            continue
        m = meta.get(c, {})
        level = m.get("level") or ("derived" if c not in meta else "")
        title, text, _ = SOURCES[key]
        groups.setdefault(title, {"description": text, "key": key, "columns": []})["columns"].append(
            {"column": c, "name": m.get("name") or _words(c),
             "description": _short(m.get("description") or definition.get(c) or "A column derived or merged by this study; no further description is recorded."),
             "unit": m.get("unit") or "", "level": level, "source": m.get("source") or "", "in": where(c)})
    order = {SOURCES[k][0]: i for i, k in enumerate(SOURCES)}
    return dict(sorted(groups.items(), key=lambda kv: order.get(kv[0], 99)))


def export_results(aoi: gpd.GeoDataFrame, columns: Sequence[str], name: Optional[str] = None) -> Path:
    """Rewrite the results zip with only the census columns in `columns` (chosen from `result_columns`); everything else stays."""
    area_dir = Path(OUTPUT_DIR) / _safe_name(name or str(aoi["name"].iloc[0]))
    keep = set(columns)
    drop = {c["column"] for g in result_columns(aoi, name).values() for c in g["columns"] if c["column"] not in keep}
    return make_zip(area_dir, drop)


# ---------------------------------------------------------------- run
def _share_worldpop_downloads(area_dir: Path, shared: Path) -> None:
    """The country-wide WorldPop age and sex rasters (tens of GB) are kept once in `shared`, for every area: the area's
    `worldpop_demographic/raw_age_structures` becomes a link to it, and what an earlier run left there is moved over."""
    link = Path(area_dir) / "worldpop_demographic" / "raw_age_structures"
    shared.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        return
    if link.is_dir():
        for item in link.iterdir():
            target = shared / item.name
            if target.exists():   # the same country already shared: merge the folders, keeping what is there
                for f in item.rglob("*"):
                    if f.is_file() and not (shared / f.relative_to(link)).exists():
                        (shared / f.relative_to(link)).parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(f), shared / f.relative_to(link))
            else:
                shutil.move(str(item), target)
        shutil.rmtree(link)
    link.symlink_to(shared, target_is_directory=True)


def run(aoi: gpd.GeoDataFrame, name: Optional[str] = None, day: str = DAY, start_time: str = START_TIME, end_time: str = END_TIME,
        feeds=FEEDS, feed_sources: Optional[Sequence[str]] = FEED_SOURCES, output_dir: Path = OUTPUT_DIR,
        cache_dir: Optional[str] = CACHE_DIR, build_map: bool = BUILD_MAP, feed_names: Optional[Sequence[str]] = None,
        service_date: Optional[str] = None, census_sources: Optional[Sequence[str]] = None) -> Path:
    """Run the study for `aoi` and return the results zip.

    Args:
        aoi: The area (from `area_from_place`, `area_from_file` or `area_from_geojson`).
        name: Results folder / zip name; None = from the area's name.
        day: "weekday" (a working day), "saturday" or "sunday".
        start_time: Start of the hours analysed ("HH:MM").
        end_time: End of the hours analysed ("HH:MM").
        feeds: "download", a folder with GTFS feeds, or a list of feed zips / folders / links (see `get_feeds`).
        feed_names: The feeds to use out of those `list_feeds` got (their `name`); None = get the feeds with `feeds`.
        service_date: The analysis date ("YYYY-MM-DD", see `service_dates`); None = chosen automatically, like the US study.
        census_sources: The census sources to download (keys of `census_sources`); the others are never downloaded. None = each one's default.
        feed_sources: Sources "download" uses; None = the most local ones that have the area's feeds.
        output_dir: Folder that gets one sub-folder per area.
        cache_dir: Shared download cache; None = TransitLOSStudies/cache.
        build_map: Also build the interactive map (`map.html`).
    """
    import dataclasses

    from studytools import cache as study_cache
    from studytools.inputs import census_population_raster, worldpop_raster
    from studytools.params import GLOBAL_PARAMS
    from studytools.pipeline import run_city_study

    if day not in DAYS:
        raise ValueError(f"day must be one of {DAYS}, not {day!r}")
    t0, t1 = _clock(start_time), _clock(end_time)
    if t1 <= t0:
        raise ValueError(f"end_time ({end_time}) must be after start_time ({start_time})")
    params = dataclasses.replace(GLOBAL_PARAMS, analysis_day_type=day, analysis_start_time=t0, analysis_end_time=t1)

    name = _safe_name(name or str(aoi["name"].iloc[0]))
    cache = study_cache.resolve_cache_dir(cache_dir)
    study_cache.apply_env(cache, "example")
    area_dir = Path(output_dir) / name
    area_dir.mkdir(parents=True, exist_ok=True)
    aoi.to_file(area_dir / "aoi.gpkg", driver="GPKG")
    aoi.to_file(area_dir / "core.gpkg", driver="GPKG")   # the core = the whole area (the core / metro split is not used here)
    (area_dir / "results" / "pop_access_h3_checkpoint.parquet").unlink(missing_ok=True)   # never reuse a level of service of other settings
    print(f"[example] {name}: {area_km2(aoi):,.0f} km2, {day} {start_time}-{end_time}", flush=True)

    config, place = area_config(aoi, name, study_cache.osm_dir(cache))
    print(f"[example] population: {census_used(config)}", flush=True)
    catalog = _census_catalog(aoi)   # (the parameter `census_sources` hides the function of that name)
    chosen = {r["key"] for r in catalog if r["default"]} if census_sources is None else set(census_sources)
    for r in catalog:   # the rules: census population unticked -> WorldPop's grid; no census -> WorldPop is the only source
        if r.get("forced") or (r.get("replaces") and r["replaces"] not in chosen):
            chosen.add(r["key"])
    config = dataclasses.replace(
        config, worldpop_layers="worldpop_layers" in chosen or not config.uses_census,
        census_skip=tuple(k for k in _OPTIONAL_USA if k not in chosen) if config.country == "USA" else ())   # unchosen sources are never downloaded
    if config.census_skip:
        print(f"[example] census sources left out: {', '.join(config.census_skip)}", flush=True)
    if config.uses_census:
        check_census_api(config.country)   # before any long step: a missing key would leave the census out of the results
    census_dir = study_cache.pycensus_dir(cache, "example")
    population_dir = area_dir / "population"
    tif = population_dir / f"{name}_census_population.tif"
    if config.worldpop_layers:   # the age and sex rasters are shared by every area of the country
        _share_worldpop_downloads(area_dir, Path(study_cache.worldpop_dir(cache)) / "raw_age_structures")
    if config.uses_census and not config.worldpop_layers and "worldpop_grid" not in chosen and census_population_raster(config, aoi, census_dir, tif):   # a fine census: no WorldPop
        config, wp_root = dataclasses.replace(config, worldpop_filename=tif.name), population_dir
        os.environ["PYCENSUS_POPULATION_RASTER"] = str(tif)   # the census raster is also the surface the election results are spread over (no WorldPop)
    else:
        worldpop_raster(config.country, study_cache.worldpop_dir(cache))
        wp_root = study_cache.worldpop_dir(cache)
    if feed_names is not None:   # the feeds chosen from `list_feeds`
        shutil.rmtree(area_dir / "gtfs", ignore_errors=True)
        for f in feed_names:
            shutil.copytree(area_dir / "gtfs_all" / f, area_dir / "gtfs" / f)
        got = bool(feed_names)
    else:
        got = get_feeds(aoi, area_dir / "gtfs", feeds, feed_sources, place=place, states=config.census_states)
    if not got:
        raise RuntimeError("No GTFS feed with stops in this area. Give your own feeds with FEEDS (a folder or a list).")
    pinned = None if service_date is None else (_date.fromisoformat(str(service_date)),) * 2
    if pinned is not None:   # the stop table and the other steps that re-score the feeds read the date of the run from here (not a date of their own)
        (area_dir / "results").mkdir(parents=True, exist_ok=True)
        (area_dir / "results" / "analysis_dates.json").write_text(json.dumps({p.name: pinned[0].isoformat() for p in (area_dir / "gtfs").iterdir() if p.is_dir()}))

    run_city_study(city_dir=area_dir, config=config, params=params, streets_root=study_cache.osm_dir(cache),
                   worldpop_root=wp_root, census_root=census_dir,
                   build_map=build_map, date_range=pinned, strict_date=pinned is not None)
    return make_zip(area_dir)


def _result_tables(area_dir: Path) -> dict:
    """`{name in the zip: file}` of the results: the streets, the H3 tables and the census at its finest level."""
    res = Path(area_dir) / "results"
    dl = res / "metro" / "downloads"   # "metro" = the whole area
    if not dl.is_dir():
        dl = res / "downloads"
    manifest = json.loads((res / "downloads_manifest.json").read_text())
    files = {"streets.parquet": res / "access_edges.parquet"}
    files.update({f"h3_res{r}.parquet": dl / f"h3_res{r}.parquet" for r in manifest.get("h3_resolutions", [])})
    levels = manifest.get("census_levels") or []
    if levels:   # finest level last
        files[f"census_{levels[-1]}.parquet"] = dl / f"census_{levels[-1]}.parquet"
    return files


def make_zip(area_dir: Path, drop: Sequence[str] = ()) -> Path:
    """`<area>/<area>_results.zip`: the streets with their level of service, the H3 tables and the census at its finest level,
    without the columns in `drop`."""
    import tempfile

    import pyarrow.parquet as pq

    area_dir, drop = Path(area_dir), set(drop) | {"__index_level_0__"}   # (pandas' stray index column is never kept)
    out = area_dir / f"{area_dir.name}_results.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z, tempfile.TemporaryDirectory() as tmp:   # parquet is already compressed
        for arc, src in _result_tables(area_dir).items():
            if not src.is_file():
                continue
            if arc != "streets.parquet" and drop & set(pq.ParquetFile(src).schema_arrow.names):
                table = pq.read_table(src)
                pq.write_table(table.drop_columns([c for c in table.column_names if c in drop]), Path(tmp) / arc)
                src = Path(tmp) / arc
            z.write(src, arc)
    print(f"[example] results: {out} ({out.stat().st_size / 1e6:,.0f} MB)" + (f", {len(drop) - 1} columns left out" if len(drop) > 1 else ""), flush=True)
    return out


# ---------------------------------------------------------------- map of the results
def get_area(place: str, area: str = "municipality", file=None) -> gpd.GeoDataFrame:
    """The area of a place name, or of a file (a path) when `file` is given."""
    return area_from_file(file) if file else area_from_place(place, area)


def _clock(s: Union[str, time]) -> time:
    if isinstance(s, time):
        return s
    h, m = (int(x) for x in str(s).split(":")[:2])
    return time(h, m)


def _safe_name(s: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", s.split(",")[0].split("(")[0]).strip("_").lower()
    return s or "area"


if __name__ == "__main__":
    area = area_from_file(AOI_FILE) if AOI_FILE else area_from_place(PLACE, AREA)
    run(area, NAME)
