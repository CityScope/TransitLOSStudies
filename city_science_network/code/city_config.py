"""The cities of the city study: one `CityConfig` (see `studytools.config`) per city folder.

Every value is a fact about a city, not a computed default.
"""

from __future__ import annotations

from studytools.config import CityConfig, _census_supported  # noqa: F401  (one implementation, in tools/studytools)


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
        # Feeds' calendars are stale by different amounts: score each feed on its own auto-picked date.
        stops_per_feed=True,
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
        # Feeds' calendars are stale by different amounts: score each feed on its own auto-picked date.
        stops_per_feed=True,
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
