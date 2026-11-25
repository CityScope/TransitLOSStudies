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
import re
import shutil
import sys
import zipfile
from datetime import time
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

#: API keys this study may ask for: (what needs it, where to sign up, environment variable)
API_KEYS = {
    "transitland": ("Transitland (transit feeds)", "https://www.transit.land (create an account, then an API key)", "TRANSITLAND_API_KEY"),
    "nap_es": ("Spain's National Access Point (transit feeds)", "https://nap.transportes.gob.es (register, then the API key in your profile)",
               "NAP_API_KEY"),
    "destatis_genesis": ("the German census (Destatis GENESIS-Online)", "https://www-genesis.destatis.de (register, then create a token)",
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


def area_config(aoi: gpd.GeoDataFrame, name: str, osm_folder: Path):
    """`(configuration, place codes)` of the area: its country, which census to use (the most local source), its OpenStreetMap
    extract and WorldPop raster; and where it is (`place_codes`)."""
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
                        census_module=census_module, worldpop_layers=not has_census,   # with a census: no WorldPop columns
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
              sources: Optional[Sequence[str]] = None, place: Optional[dict] = None, states: Optional[Sequence[str]] = None) -> list:
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
    """
    gtfs_dir = Path(gtfs_dir)
    choice = json.dumps(feeds if isinstance(feeds, str) else [str(f) for f in feeds]) + json.dumps(list(sources or [])) + "|local+missing+operators"
    marker = gtfs_dir / "_feeds_choice.json"
    if marker.is_file() and marker.read_text() == choice and any(gtfs_dir.glob("*/stops.txt")):
        return sorted(p.parent for p in gtfs_dir.glob("*/stops.txt"))   # same choice as the last run: reuse
    if gtfs_dir.exists():
        shutil.rmtree(gtfs_dir)
    gtfs_dir.mkdir(parents=True)
    if isinstance(feeds, str) and feeds == "download":
        place = place or place_codes(aoi)
        tiers = [list(sources)] if sources else by_place(FEED_SOURCES_BY_PLACE, place)
        kept = []
        for tier in tiers:   # most local first; each later (more general) source only adds the feeds the earlier ones miss
            added = _keep_feeds_in_area(download_feeds(aoi, gtfs_dir, tier, place, states), aoi, keep=kept)
            print(f"[gtfs] {', '.join(tier)}: {len(added)} feeds" + (" not already covered" if kept else ""), flush=True)
            kept += added
    else:
        items = sorted(Path(feeds).iterdir()) if isinstance(feeds, (str, Path)) else list(feeds)
        kept = _keep_feeds_in_area([f for f in (_add_feed(item, gtfs_dir) for item in items) if f is not None], aoi)
    marker.write_text(choice)
    print(f"[gtfs] {len(kept)} feeds with stops in the area: {', '.join(p.name for p in kept)}", flush=True)
    return kept


def download_feeds(aoi: gpd.GeoDataFrame, gtfs_dir: Path, sources: Sequence[str], place: Optional[dict] = None,
                   states: Optional[Sequence[str]] = None) -> list:
    """Download the area's feeds from `sources`. Returns the feed folders.

    Sources: "ntd" (US National Transit Database, the area's states), "nap" (Spain's National Access Point, the area's provinces;
    asks for a free key), "mobility_database" (Mobility Database catalog, no key), "transitland" (asks for a free key).
    """
    from pyGTFSHandler.downloaders import MobilityDatabaseCatalog, TransitLandDownloader

    place = place or {}
    folders = []
    for s in sources:
        try:
            if s == "ntd":
                from pyGTFSHandler.downloaders.usa import NTDDownloader

                client = NTDDownloader()
                found = client.search_feeds(state=list(states or [r.split("-")[1] for r in place.get("regions", []) if r.startswith("US-")]))
            elif s == "nap":
                from pyGTFSHandler.downloaders.spain import NAPDownloader

                key = api_key("nap_es")
                if not key:
                    continue
                client = NAPDownloader(api_key=key)
                found = client.search_feeds(region=place.get("provinces", []), region_type=0)   # 0 = province
            elif s == "mobility_database":
                client = MobilityDatabaseCatalog()
                found = client.search_feeds(aoi=aoi)
            elif s == "transitland":
                key = api_key("transitland")
                if not key:
                    continue
                client = TransitLandDownloader(api_key=key)
                found = client.search_feeds(aoi=aoi)
            else:
                raise ValueError(f"unknown feed source {s!r}: use 'ntd', 'nap', 'mobility_database' or 'transitland'")
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 -- one source down must not stop the others
            print(f"[gtfs] {s}: search failed ({exc})", flush=True)
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


# ---------------------------------------------------------------- run
def run(aoi: gpd.GeoDataFrame, name: Optional[str] = None, day: str = DAY, start_time: str = START_TIME, end_time: str = END_TIME,
        feeds=FEEDS, feed_sources: Optional[Sequence[str]] = FEED_SOURCES, output_dir: Path = OUTPUT_DIR,
        cache_dir: Optional[str] = CACHE_DIR, build_map: bool = BUILD_MAP) -> Path:
    """Run the study for `aoi` and return the results zip.

    Args:
        aoi: The area (from `area_from_place`, `area_from_file` or `area_from_geojson`).
        name: Results folder / zip name; None = from the area's name.
        day: "weekday" (a working day), "saturday" or "sunday".
        start_time: Start of the hours analysed ("HH:MM").
        end_time: End of the hours analysed ("HH:MM").
        feeds: "download", a folder with GTFS feeds, or a list of feed zips / folders / links (see `get_feeds`).
        feed_sources: Sources "download" uses; None = the most local ones that have the area's feeds.
        output_dir: Folder that gets one sub-folder per area.
        cache_dir: Shared download cache; None = TransitLOSStudies/cache.
        build_map: Also build the interactive map (`map.html`).
    """
    import dataclasses

    from studytools import cache as study_cache
    from studytools.inputs import worldpop_raster
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
    worldpop_raster(config.country, study_cache.worldpop_dir(cache))
    if not get_feeds(aoi, area_dir / "gtfs", feeds, feed_sources, place=place, states=config.census_states):
        raise RuntimeError("No GTFS feed with stops in this area. Give your own feeds with FEEDS (a folder or a list).")

    run_city_study(city_dir=area_dir, config=config, params=params, streets_root=study_cache.osm_dir(cache),
                   worldpop_root=study_cache.worldpop_dir(cache), census_root=study_cache.pycensus_dir(cache, "example"),
                   build_map=build_map)
    return make_zip(area_dir)


def make_zip(area_dir: Path) -> Path:
    """`<area>/<area>_results.zip`: the streets with their level of service, the H3 tables and the census at its finest level."""
    area_dir = Path(area_dir)
    res = area_dir / "results"
    dl = res / "metro" / "downloads"   # "metro" = the whole area
    if not dl.is_dir():
        dl = res / "downloads"
    manifest = json.loads((res / "downloads_manifest.json").read_text())
    files = {"streets.parquet": res / "access_edges.parquet"}
    files.update({f"h3_res{r}.parquet": dl / f"h3_res{r}.parquet" for r in manifest.get("h3_resolutions", [])})
    levels = manifest.get("census_levels") or []
    if levels:   # finest level last
        files[f"census_{levels[-1]}.parquet"] = dl / f"census_{levels[-1]}.parquet"
    out = area_dir / f"{area_dir.name}_results.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z:   # parquet is already compressed
        for arc, src in files.items():
            if src.is_file():
                z.write(src, arc)
    print(f"[example] results: {out} ({out.stat().st_size / 1e6:,.0f} MB)", flush=True)
    return out


# ---------------------------------------------------------------- map of the results
def show_map(results, height: int = 760):
    """Show the interactive map of a run (`map.html`, with its distribution, regression, ANOVA, discretization and metadata tabs) in
    Jupyter or Colab. The map files are served from this machine by a small local server.

    Args:
        results: The results zip returned by `run`, or the area folder.
        height: Height of the map in pixels.
    """
    area_dir = Path(results).parent if str(results).endswith(".zip") else Path(results)
    if not (area_dir / "map.html").is_file():
        raise FileNotFoundError(f"No map in {area_dir}: run with build_map=True.")
    port = _serve(area_dir)
    try:
        from google.colab import output
    except ImportError:
        from IPython.display import IFrame

        return IFrame(f"http://localhost:{port}/map.html", width="100%", height=height)
    output.serve_kernel_port_as_iframe(port, path="/map.html", height=height)


_SERVERS: dict = {}


def _serve(folder: Path) -> int:
    """Serve `folder` over HTTP on a free local port (in a background thread) and return the port. Supports byte ranges, which the
    map's tile files (PMTiles) need."""
    import functools
    import http.server
    import threading

    folder = Path(folder).resolve()
    if folder in _SERVERS:
        return _SERVERS[folder]

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):   # quiet
            pass

        def send_head(self):
            rng = self.headers.get("Range")
            path = Path(self.translate_path(self.path))
            if not rng or not path.is_file():
                return super().send_head()
            size = path.stat().st_size
            first, _, last = rng.split("=", 1)[1].split(",")[0].strip().partition("-")
            if first:            # bytes=first-last or bytes=first-
                start, end = int(first), min(int(last), size - 1) if last else size - 1
            else:                # bytes=-n: the last n bytes
                start, end = max(0, size - int(last)), size - 1
            if start >= size:
                self.send_error(416)
                return None
            f = open(path, "rb")
            f.seek(start)
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self._remaining = end - start + 1
            return f

        def copyfile(self, source, outputfile):
            left = getattr(self, "_remaining", None)
            if left is None:
                return super().copyfile(source, outputfile)
            while left > 0:
                chunk = source.read(min(1 << 16, left))
                if not chunk:
                    break
                outputfile.write(chunk)
                left -= len(chunk)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(folder)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _SERVERS[folder] = server.server_address[1]
    return _SERVERS[folder]


# ---------------------------------------------------------------- notebook helpers
def draw_area(aoi: Optional[gpd.GeoDataFrame] = None):
    """A map with drawing tools (polygon, rectangle), opened on `aoi`. Returns (map, drawn): `drawn` collects the shapes drawn."""
    from ipyleaflet import DrawControl, Map

    c = aoi.union_all().centroid if aoi is not None else None
    drawn: list = []
    m = Map(center=(c.y, c.x) if c is not None else (20, 0), zoom=9 if c is not None else 2, scroll_wheel_zoom=True)
    ctl = DrawControl(polygon={"shapeOptions": {"color": "#d33"}}, rectangle={"shapeOptions": {"color": "#d33"}},
                      polyline={}, circlemarker={}, marker={}, circle={})
    ctl.on_draw(lambda control, action, geo_json: drawn.append(geo_json) if action == "created" else None)
    m.add(ctl)
    return m, drawn


def get_area(place: str, area: str = "municipality", file=None) -> gpd.GeoDataFrame:
    """The area of a place name, or of a file when `file` is given: a path, or "upload" to pick one from your computer (Colab)."""
    if not file:
        return area_from_place(place, area)
    if file == "upload":
        from google.colab import files

        file = next(iter(files.upload()))
    return area_from_file(file)


def area_from_drawn(drawn: list) -> gpd.GeoDataFrame:
    """The last shape drawn on the `draw_area` map."""
    if not drawn:
        raise RuntimeError("Nothing drawn: draw a polygon or rectangle on the map above, then run this cell and the ones below again.")
    return area_from_geojson(drawn[-1])


def show_area(aoi: gpd.GeoDataFrame):
    """The area on a light map (ipyleaflet, a simplified outline: folium chokes on big geometries), with its name and size."""
    import json

    import ipywidgets as widgets
    from ipyleaflet import GeoJSON, Map

    minx, miny, maxx, maxy = aoi.total_bounds
    outline = aoi.geometry.simplify(max(maxx - minx, maxy - miny) / 500)
    m = Map(center=((miny + maxy) / 2, (minx + maxx) / 2), scroll_wheel_zoom=True, layout=widgets.Layout(height="420px"))
    m.add(GeoJSON(data=json.loads(outline.to_json()), style={"color": "#d33", "fillOpacity": 0.1, "weight": 2}))
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    return widgets.VBox([widgets.HTML(f"<b>{aoi['name'].iloc[0]}</b>: {area_km2(aoi):,.0f} km²"), m])


def upload_feeds() -> list:
    """Pick GTFS zip files from your computer (Colab); returns their names, for `FEEDS`."""
    from google.colab import files

    return list(files.upload())


def download(path) -> None:
    """Download a file in Colab; elsewhere print where it is."""
    try:
        from google.colab import files
    except ImportError:
        print(path)
    else:
        files.download(str(path))


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
