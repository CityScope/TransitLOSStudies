"""Static configuration of one study area (`CityConfig`): name, country, census source, OSM extract, WorldPop raster.

The city study keeps its registry of cities in `city_science_network/code/city_config.py`; the example study builds one from an area.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional, Sequence


# `CityConfig.country` is an ISO-3166-1 alpha-3 code (e.g. "USA", "MEX"),
# but `pycensus`'s per-country submodules are lowercase full country names
# (`pycensus.countries.usa`, `pycensus.countries.mexico`, ... -- see that package's
# `src/pycensus/__init__.py`), not a lowercased alpha-3 code. Naively trying
# `pycensus.countries.{country.lower()}` therefore only accidentally works for USA
# ("usa" == "usa") and silently fails ImportError (-> `uses_census=False`)
# for every other alpha-3 code that doesn't literally match its module name,
# e.g. "MEX" -> "mex" (module is `mexico`), "ESP" -> "esp" (module is
# `spain`). This table is the explicit ISO3 -> pycensus-submodule mapping.
#
# Real bug fixed 2026-08-21: CAN/AND/TWN/CHL were missing here even though
# `pycensus` gained real Canada/Andorra/Taiwan/Chile modules earlier this
# session -- Toronto/Andorra/Taipei/Concepcion were silently falling back to
# `uses_census=False` (ImportError swallowed below) despite real census data
# being available for all four. `CHN` (Shanghai) is deliberately absent --
# there is no `pycensus.countries.china` module by explicit design (no
# China census implementation) -- Shanghai correctly stays WorldPop-only.
_ISO3_TO_PYCENSUS_MODULE: dict[str, str] = {
    "USA": "usa",
    "MEX": "mexico",
    "ESP": "spain",
    "DEU": "germany",
    "ISR": "israel",
    "CAN": "canada",
    "AND": "andorra",
    "TWN": "taiwan",
    "CHL": "chile",
}


def _census_supported(country: str, census_module: str | None = None) -> bool:
    """Whether `pycensus` implements national census loaders for `country`.

    `pycensus` organizes each country it supports as its own submodule
    (see `_ISO3_TO_PYCENSUS_MODULE` above for the ISO3 -> module-name
    mapping, since the two naming schemes don't line up) rather than
    exposing an explicit "supported countries" registry, so this checks
    for that submodule directly instead of hardcoding a country list here.
    Add a country to `pycensus` and it becomes usable from `CityConfig`
    once its ISO3 code is added to `_ISO3_TO_PYCENSUS_MODULE`.

    `census_module`, when given (see `CityConfig.census_module`), overrides
    `_ISO3_TO_PYCENSUS_MODULE[country]` entirely -- this is how a specific
    city routes to a *sub*-national census module instead of its country's
    generic one, e.g. Gipuzkoa (`country="ESP"`) routing to
    `pycensus.countries.euskadi` (the Basque Country's own statistical source)
    instead of `pycensus.countries.spain` (generic Spain/INE), which is wrong for a
    Basque city -- Euskadi collects and reports its own census data
    separately from INE.

    `pycensus.countries.worldwide` (WorldPop) is deliberately excluded: it is a
    global raster source keyed by whatever AOI the caller passes in, not
    a per-country census source, and every `CityConfig` already uses it
    for population regardless of `country`.
    """
    module_name = census_module if census_module is not None else _ISO3_TO_PYCENSUS_MODULE.get(country)
    if module_name is None:
        return False
    try:
        importlib.import_module(f"pycensus.countries.{module_name}")
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class CityConfig:
    """Static facts about one city study area.

    Attributes:
        key: Directory/city key (matches the `city_science_network/<key>/` folder).
        display_name: Human-readable name, used in figures/titles.
        geocode_name: Name/query passed to
            `UrbanAccessAnalyzer.api.AreaOfInterest.from_name` to resolve the
            non-US city-core boundary (ignored for cities `pycensus` has
            census support for).
        country: This city's country, spelled to match a `pycensus`
            per-country submodule name if one exists (e.g. `"USA"`).
            Drives `uses_census` below -- not itself a `pycensus` API call,
            just the key `_census_supported` checks for a matching module.
        census_states: US state name(s)/abbreviation(s) forwarded to
            `pycensus.countries.usa.acs5.load`/`.dhc.load` as `states`. `None` for
            cities outside `pycensus`'s census coverage.
        pbf_filename: Shared `.osm.pbf` file (under `city_science_network/streets/`)
            covering this city's metro AOI.
        worldpop_filename: Shared WorldPop raster (under
            `city_science_network/worldpop/`) covering this city's country.
        census_module: Optional override of the `pycensus` submodule name
            used for this city, bypassing `_ISO3_TO_PYCENSUS_MODULE[country]`.
            `None` (the default) keeps the plain country-level dispatch.
            Set this when a city's *region*, not its whole country, has its
            own separate census source -- e.g. Gipuzkoa (`country="ESP"`)
            sets `census_module="euskadi"` so it uses the Basque Country's
            own Eustat-sourced data instead of generic Spain/INE, while
            other Spanish cities (if any are added) keep using
            `pycensus.countries.spain` untouched.
    """

    key: str
    display_name: str
    geocode_name: str
    country: str
    census_states: Optional[Sequence[str]]
    pbf_filename: str
    worldpop_filename: str
    census_module: Optional[str] = None
    # When True, `download_and_prepare_stops` scores each `gtfs/` subdirectory
    # (agency feed) independently -- its own `Feed` load, its own
    # representative-date pick, its own stop scoring -- then concatenates the
    # resulting per-feed stop GeoDataFrames, instead of stacking every feed
    # into one `Feed` and picking a single shared representative date across
    # all of them. Real bug this fixes (Taipei, found 2026-08-24): its
    # `tdx_bus` feed's real `calendar.txt` validity is only 2026-05-01 to
    # 2026-06-30 (a narrow real-time TDX export snapshot, already stale
    # relative to any near-term analysis date), while `trtc_metro`'s spans
    # 2025-08-18 to 2026-12-31 -- no single date can be representative of
    # both feeds' real service at once. Only set this for a city with real,
    # verified per-feed calendar misalignment like this -- for every other
    # multi-feed city, one shared representative date across all stacked
    # feeds is both correct and cheaper (see `transitlos.stops` module
    # docstring for the general case).
    stops_per_feed: bool = False
    # When True, `_add_worldpop_gapfill` (pipeline.py) runs after the census
    # join: for every h3 cell where the real census join left NO real
    # population coverage (`cbs_population` -- or the country's equivalent
    # bare canonical population column, whichever the join uses -- null),
    # real WorldPop age/sex-structure layers are filled in under a
    # `worldpop_`-prefixed name, never overwriting/renaming into the
    # census-source namespace. Beersheba's real CBS locality coverage is
    # only ~51% of its real AOI by area -- the remaining ~49% is open Negev desert / unrecognized
    # Bedouin villages CBS deliberately never surveys (documented CBS
    # exclusion, verified via web search 2026-08-25) -- so without this,
    # roughly half the map's h3 cells get zero demographic detail even
    # though WorldPop's satellite-based raster genuinely covers them. Only
    # set this for a city with a similar real, documented census-coverage
    # gap -- for a city where the census join already covers the whole AOI,
    # this is a no-op (no cell will ever have a null population to fill).
    census_worldpop_gapfill: bool = False
    # GlobalBuildingAtlas buildings (`pycensus.countries.worldwide.gba`): building metrics (`gba_building*`, `gba_builtFloorArea*`)
    # on every h3 resolution and census level, and a population source from built floor area (`gba_builtSurfacePopulation`:
    # people per built m2 calibrated on the census / WorldPop, filling where WorldPop misses what is built -- see
    # `pipeline._gba_population_rasters`).
    gba_buildings: bool = True   # GlobalBuildingAtlas built-surface population + building metrics (every city)
    # False: no WorldPop columns in the results (no `worldpop_*` demographic layers, no WorldPop comparison columns). For an area with its
    # own census (the example study sets it so); WorldPop is then only the fine-grained distribution inside each census unit.
    worldpop_layers: bool = True
    # True: the map has only the finest H3 resolution (hexagons and circles) and the finest census level, no street or development
    # layers -- light enough to build and open anywhere (the example study, e.g. in Colab). The downloads keep every level.
    map_finest_only: bool = False
    # 2026-09-29, explicit user request ("any hexagon... that touches the
    # border of israel in beerseba should be deleted"): when True,
    # `_filter_cells_touching_country_border` (pipeline.py) drops every h3
    # cell whose geometry intersects the OSM/Nominatim-geocoded national
    # boundary line of `border_country_geocode_name` -- a cell straddling
    # (or exactly abutting) the line is dropped outright, not clipped to
    # the in-country portion, since a fractional cell can't honestly carry
    # a single equity flag/level_of_service value. Beersheba's AOI (a union
    # of Negev-region localities near the West Bank/Gaza/Egypt frontiers)
    # is the motivating case -- a border-touching cell there is politically
    # sensitive to show at all, not just a data-quality nuisance. Runs
    # after every census/WorldPop join (so the equity-flag regression and
    # every downstream stat/map layer never sees these cells), gated
    # separately from `census_worldpop_gapfill` since a future border city
    # might need one flag without the other.
    exclude_cells_touching_country_border: bool = False
    # Nominatim geocode query for the border-clip above (`country`'s ISO
    # code alone isn't a valid Nominatim query, so this is a separate plain-
    # text field, not derived from `country`). Always set this explicitly
    # whenever `exclude_cells_touching_country_border=True`.
    border_country_geocode_name: str | None = None

    @property
    def uses_census(self) -> bool:
        """Whether this city uses `pycensus` national census loaders
        (population/equity variables, and a Census `place` boundary for
        the city-core split) instead of WorldPop + an OSM admin boundary.

        Computed from whether `pycensus` actually implements `country`
        (see `_census_supported`), or `census_module` when that's set,
        rather than a per-city boolean, so this stays correct as
        `pycensus` gains countries.
        """
        return _census_supported(self.country, self.census_module)
