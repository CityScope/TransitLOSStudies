"""Per-city static configuration: name, country, US-ness, census states, WorldPop iso.

No logic lives here beyond a plain dataclass and a name-keyed registry --
every value is a fact about a city, not a computed default.
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


CITY_CONFIGS: dict[str, CityConfig] = {
    "boston": CityConfig(
        key="boston",
        display_name="Boston",
        geocode_name="Boston, Massachusetts, USA",
        country="USA",
        # The Boston-Cambridge-Newton CBSA is MA-NH, not MA-only -- Census coverage
        # limited to MA left every h3 cell on the NH side of the metro with zero acs_*
        # data at every level (confirmed: summed acs_population undercounted Boston's
        # real ~4.9M metro population by ~2.7M before this fix).
        census_states=("MA", "NH"),
        pbf_filename="us_massachusetts.osm.pbf",
        worldpop_filename="usa_pop_2025_CN_100m_R2025A_v1.tif",
    ),
    "san_francisco": CityConfig(
        key="san_francisco",
        display_name="San Francisco",
        geocode_name="San Francisco, California, USA",
        country="USA",
        census_states=("CA",),
        pbf_filename="northern_california.osm.pbf",
        worldpop_filename="usa_pop_2025_CN_100m_R2025A_v1.tif",
    ),
    "guadalajara": CityConfig(
        key="guadalajara",
        display_name="Guadalajara",
        geocode_name="Guadalajara, Jalisco, Mexico",
        country="MEX",
        census_states=None,
        pbf_filename="mexico.osm.pbf",
        worldpop_filename="mex_pop_2025_CN_100m_R2025A_v1.tif",
    ),
    "gipuzkoa": CityConfig(
        key="gipuzkoa",
        display_name="Gipuzkoa",
        # Metro AOI (gipuzkoa/aoi.gpkg) is the whole province (89 municipality
        # polygons). Core boundary must isolate just the San Sebastián /
        # Donostia municipality -- geocoding "Gipuzkoa, Spain" would resolve
        # the whole-province admin boundary again and make core == metro.
        geocode_name="San Sebastián, Gipuzkoa, Spain",
        country="ESP",
        census_states=None,
        pbf_filename="spain_paisvasco.osm.pbf",
        worldpop_filename="esp_pop_2025_CN_100m_R2025A_v1.tif",
        # Gipuzkoa IS INE province code "20" -- investigated 2026-08-22
        # whether Euskadi's own EUSTAT statistics office publishes richer
        # section-level (or finer) census data than generic Spain/INE:
        # EUSTAT's geo-cartography portals (opendata.euskadi.eus,
        # eustat.eus/geoservicios) remain genuinely unreachable (403/404) to
        # a scripted client, so pycensus.countries.euskadi has no boundary
        # source finer than municipality and only 3 fields (population,
        # areaKm2, populationDensity, all from GISCO/LAU 2021 -- the SAME
        # underlying INE-sourced municipal figures Spain/INE already
        # exposes, not independent EUSTAT data). Generic
        # pycensus.countries.spain, by contrast, now has a real, live
        # section-level (seccion censal) source from INE's own OGC API
        # Features service -- 544 real sections for Gipuzkoa (province
        # "20"), live-verified 2026-08-22 by summing section population for
        # Donostia (185,734 vs. real municipal total 188,102, 98.7% match)
        # and Irun (62,635 vs. 62,933, 99.5% match), plus 3 real fields
        # (population/malePopulation/femalePopulation) at that level. So
        # generic Spain/INE gives Gipuzkoa BOTH finer resolution and more
        # real fields than the Euskadi-specific module -- routing there
        # instead. Do not resurrect census_module="euskadi" without a real,
        # verified EUSTAT attribute/geometry source that's actually finer
        # than section-level (see pycensus.countries.euskadi.api's module
        # docstring: EUSTAT's Data Bank API IS real/reachable but not yet
        # wired to any loader).
    ),
    "toronto": CityConfig(
        key="toronto",
        display_name="Toronto",
        geocode_name="Toronto, Ontario, Canada",
        country="CAN",
        census_states=None,
        pbf_filename="canada_ontario.osm.pbf",
        worldpop_filename="can_pop_2025_CN_100m_R2025A_v1.tif",
    ),
    "andorra": CityConfig(
        key="andorra",
        display_name="Andorra",
        # Core-boundary geocode target is the capital, not the country -- the
        # metro-scale AOI (andorra/aoi.gpkg) is the whole country polygon, but
        # `resolve_core_boundary` uses this query to find the city-core split.
        geocode_name="Andorra la Vella, Andorra",
        country="AND",
        census_states=None,
        pbf_filename="andorra.osm.pbf",
        worldpop_filename="and_pop_2025_CN_100m_R2025A_v1.tif",
    ),
    "hamburg": CityConfig(
        key="hamburg",
        display_name="Hamburg",
        geocode_name="Hamburg, Germany",
        country="DEU",
        census_states=None,
        pbf_filename="germany.osm.pbf",
        worldpop_filename="deu_pop_2025_CN_100m_R2025A_v1.tif",
    ),
    "taipei": CityConfig(
        key="taipei",
        display_name="Taipei",
        # City-core resolution (see code/city_core.py) geocodes this name directly
        # against Nominatim, keeping the first Polygon/MultiPolygon admin-boundary
        # hit -- resolves to Taipei City specifically, not the wider metro AOI
        # (Taipei City + New Taipei City + Keelung + Taoyuan City) built separately
        # into aoi.gpkg.
        # "Taipei City, Taiwan" resolves to junk POIs on Nominatim (no
        # boundary=administrative hit) -- "Taipei, Taiwan" correctly ranks
        # the 臺北市 (Taipei City) admin-boundary relation first.
        geocode_name="Taipei, Taiwan",
        country="TWN",
        census_states=None,
        pbf_filename="taiwan.osm.pbf",
        worldpop_filename="twn_pop_2025_CN_100m_R2025A_v1.tif",
        # Real bug found 2026-08-24: `tdx_bus`'s calendar.txt validity is
        # only 2026-05-01 - 2026-06-30, while `trtc_metro`'s spans
        # 2025-08-18 - 2026-12-31 -- stacking both into one Feed and picking
        # a single shared representative date starves whichever feed's real
        # window the pick lands outside of (bounded-window search: mostly
        # missed tdx_bus's narrow window, 6,302 stops; old unbounded search:
        # happened to land inside tdx_bus's window but by luck, 51,864
        # stops). See `CityConfig.stops_per_feed`'s docstring.
        stops_per_feed=True,
    ),
    "shanghai": CityConfig(
        key="shanghai",
        display_name="Shanghai",
        # City-core resolution (see code/city_core.py) geocodes this name directly
        # against Nominatim, keeping the first Polygon/MultiPolygon admin-boundary
        # hit -- resolves to Shanghai municipality specifically, not the wider
        # metro AOI (Shanghai plus 13 neighboring Jiangsu/Zhejiang/Anhui
        # prefecture-level cities) built separately into aoi.gpkg.
        geocode_name="Shanghai, China",
        country="CHN",
        census_states=None,
        pbf_filename="china.osm.pbf",
        worldpop_filename="chn_pop_2025_CN_100m_R2025A_v1.tif",
    ),
    "beerseba": CityConfig(
        key="beerseba",
        display_name="Beersheba",
        geocode_name="Beersheba, Israel",
        country="ISR",
        census_states=None,
        pbf_filename="israel_and_palestine.osm.pbf",
        worldpop_filename="isr_pop_2025_CN_100m_R2025A_v1.tif",
        # aoi.gpkg is a union of ~25 separate localities (see
        # aoi_25locality_union_backup.gpkg) -- concave and gappy, under-
        # covering the metro's real footprint for both isochrones and the
        # census join. Previously widened to its convex hull
        # (`aoi_convex_hull`); removed per explicit user request (2026-08-30)
        # -- the real AOI (union of the 25 localities, no hull) is used
        # everywhere now. `census_worldpop_gapfill` below still fills any
        # h3 cell the real census join leaves uncovered.
        # Real CBS locality polygons cover only ~51% of this AOI (771 km2 of
        # 1503 km2) -- the rest is open Negev desert / unrecognized Bedouin
        # villages CBS deliberately never surveys. See
        # `CityConfig.census_worldpop_gapfill`'s docstring.
        census_worldpop_gapfill=True,
        # See `CityConfig.exclude_cells_touching_country_border`'s
        # docstring -- explicit user request, 2026-09-29.
        exclude_cells_touching_country_border=True,
        border_country_geocode_name="Israel",
    ),
    "concepcion": CityConfig(
        key="concepcion",
        display_name="Concepción",
        # Metro AOI (concepcion/aoi.gpkg) is Gran Concepción, Chile's
        # officially constituted metropolitan area (Supreme Decree n°326,
        # 2024-08-28) -- the union of its 11 constituent comunas
        # (Concepción, Talcahuano, San Pedro de la Paz, Chiguayante,
        # Hualpén, Coronel, Lota, Penco, Tomé, Hualqui, Santa Juana), each
        # geocoded individually against Nominatim and unioned (mirrors
        # gipuzkoa's/andorra's/taipei's/shanghai's wider-than-core aoi.gpkg
        # pattern). This geocode_name resolves just the Concepción comuna
        # itself for the core split.
        geocode_name="Concepción, Bío Bío, Chile",
        country="CHL",
        census_states=None,
        # Not yet fetched -- Chile isn't covered by the shared streets/worldpop
        # data this repo currently has on disk (see `main.py`'s CITY_KEYS note).
        # Named to match the convention other countries' files use
        # (e.g. `mexico.osm.pbf`, `chn_pop_...tif`) once fetched.
        pbf_filename="chile.osm.pbf",
        worldpop_filename="chl_pop_2025_CN_100m_R2025A_v1.tif",
    ),
}
