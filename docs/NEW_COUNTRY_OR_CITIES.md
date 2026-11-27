# Adding a source, a country or a list of cities

For a person or an AI agent who must (1) add a **data source** (transit feeds, census, buildings), (2) build the **fixed map and the
flexible-scoring map of a new country**, or (3) build a **new list of cities** like the city science study. It says what to read, in
which order, and which existing study to copy. The detail of each package lives in that package's own guides; this page ties them
together. Paths are relative to the root of each repository (`TransitLOSStudies`, `pyCensus`, `pyGTFSHandler`, `transitLOS`,
`UrbanAccessAnalyzer`, `geohierarchy`).

If you only need one area (a city, a metro area or a drawn shape), you need none of this: use
[`example_metro_area/example.ipynb`](https://colab.research.google.com/github/CityScope/TransitLOSStudies/blob/main/example_metro_area/example.ipynb).

## 0. Rules that apply to everything

1. **Never fabricate data.** Every number comes from a file or an API you opened. Check a new source against a known total (the agency's
   published population, the number of stops an agency reports) before using it. An HTTP 200 can be an HTML error page.
2. **Most local source first.** An input comes from the most local source that covers the area: city or region, then country, then the
   worldwide fallback (WorldPop, Mobility Database, Transitland). A worldwide source is never used where a local one exists.
3. **Sub-municipal data** where it exists (block, enumeration area, grid cell, statistical area). Municipality-only data is a last resort,
   and elections are always below the county (precinct or polling place), on the official electoral districts.
4. **Keep the original column names** of a source (`acs5_povertyBelow100`) and add the shared name next to it, not instead of it.
5. **Layers.** `geohierarchy` < `pyCensus` and `pyGTFSHandler` < `UrbanAccessAnalyzer` < `transitLOS` < the studies. A package depends only
   on the ones below it. Shared code goes in the lowest layer that needs it. A study never imports another study; what two studies share
   goes in `tools/studytools/`.
6. **Shared cache.** Downloads go to the shared cache (`cache/`, or `STUDY_CACHE_DIR`), never into a package or the repository.
7. **Public repository.** Commit code, docs and the smallest inputs that can no longer be downloaded. Never commit API keys, local paths,
   notes about a machine or about hosting (keep those as `*.local.md`, which `.gitignore` excludes), or large files (the pre-commit hook
   `tools/check_large_files.py` refuses over 10 MB).
8. **API keys** live in one file, `~/.config/ptlos/api_keys.json`, or in environment variables; each package resolves them
   (`pycensus.api_keys.resolve_api_key`, `pyGTFSHandler.downloaders.utils.config.get_api_key`). Add a key's name and where to sign up to
   that package's `api_keys.example.json`.
9. **Memory.** The builds are heavy. Run them in memory-capped processes (`memsched.py` schedules jobs by free memory and records each
   job's peak in a JSON file next to the study); a plain parallel pool can take the machine down.
10. Test without network (a saved or hand-made answer) and run the study's checks (section 6).

## 1. Adding a data source

### 1a. Transit feeds (pyGTFSHandler)

Read, in this order:

1. `docs/contributing_downloader.md`: the interface every downloader implements and where it goes.
2. `docs/methodology/new_area_gtfs_agent_guide.md`: how to find, check and add the feeds of a new area (official portals, what to do
   when feeds are missing, API-key table).
3. `pyGTFSHandler/downloaders/base.py` (`BaseGTFSDownloader`: `search_feeds` returns metadata only, `download_feeds` downloads) and
   `pyGTFSHandler/downloaders/utils/` (`config.py` keys, `aoi.py` area to bounding box, `models.py` `GTFSFeedMetadata`, `download.py`,
   `http.py`, `naming.py`, `coverage_check.py`, `dates.py`).
4. A worldwide catalog as a model: `downloaders/mobility_database_catalog.py` (no key) and `downloaders/transitland.py` (key).
   A national portal as a model: `downloaders/usa/ntd.py` (`NTDDownloader`, by state) and `downloaders/spain/nap.py` (`NAPDownloader`, by
   province, key).
5. For feed problems after download (missing service dates, frequencies, directions): `pyGTFSHandler/feed.py` and
   `pyGTFSHandler/models/` (`calendar.py`, `frequencies.py`, `stop_times.py`, `shapes.py`), and `docs/methodology/*.md`.

Then plug the downloader into the studies:

* One area (notebook and script): `example_metro_area/example.py`: add the source to `_search_source` and to `FEED_SOURCES_BY_PLACE`
  (the most local sources first; later ones only add feeds the earlier ones miss).
* A country study: the study's own GTFS stage (`usa_study/code/gtfs_stage.py`; Spain's `spain_study/nap_fetch.py` and
  `spain_study/code/gtfs_stage.py`, `gtfs_store.py`).
* A city list: `city_science_network/code/gtfs_fetch.py`, `gtfs_audit.py` (analysis dates), `check_missing_transit.py`, and the
  per-city folders `gtfs/`, `gtfs_archive/` (feeds that disappear online, archived by `archive_feeds.py`) and `assumed_feeds/` (feeds
  built by hand, each with an `ASSUMED.md` saying what was assumed).

### 1b. Census and other area data (pyCensus)

Read, in this order:

1. `docs/AGENT_GUIDE_NEW_CITY_OR_COUNTRY.md`: the hard rules, what exists, the resolution classes, the manual-download table, step by
   step for a new city or country.
2. `docs/CONTRIBUTING_SOURCES.md`: where a census, worldwide or building (cadaster) source goes and what its loader returns.
3. `docs/SCHEMAS.md`: what every schema file must contain. `docs/GLOBAL_CENSUS_COLUMNS.md`: every shared (global) column and which
   country source supplies it.
4. Code to copy from: a small country, `src/pycensus/census/countries/andorra/` or `taiwan/` (geography, constants, `schema/`, one
   loader per source); a big API-based one, `countries/usa/` (`acs5`, `dhc`, `lodes_wac`, `religion`, `housing`, `elections`,
   `elections_precinct`); a file-based one, `countries/spain/`; EU-wide grids in `countries/eu/`.
5. Shared machinery: `src/pycensus/census/global_schema.json` (shared columns), `reference/global_equivalents.json` (local to global names),
   `derived_columns.py`, `elections_common.py`, `resolution.py` (every source declares `low`, `normal` or `high`), `api_keys.py`, and the
   building sources in `src/pycensus/cadaster/` (`SOURCES` and `BY_COUNTRY` in its `__init__.py`).

Then wire it into the studies. This is the step most often forgotten, because the pipeline dispatches by country:

* `tools/studytools/config.py`: `CityConfig` (one per study area) and `_census_supported(country, census_module)`, which decides
  whether a country uses a census or only WorldPop.
* `tools/studytools/pipeline.py`: `_join_census` dispatches on the country (USA, Mexico, Chile, Canada, Germany, Israel, Taiwan, Basque
  Country ...), `_join_race` and `_join_jobs` add the decennial census and the jobs; each country has a `*_KEEP_COLUMNS` set and a
  loader function. Add your country's branch and its keep columns here, and check the census-population step
  (`_census_population_polygons`, `inputs.census_population_raster`): the finest level with a population count is what the built-surface
  population and the population map are calibrated to.
* A source that can be left out of a download (the notebook's "census sources" step) is a family in `CityConfig.census_skip`.

### 1c. Other inputs

Streets: OpenStreetMap extracts through `UrbanAccessAnalyzer` (`osm_io.py`, Geofabrik). Population fallback and buildings:
`pycensus.census.worldwide` (WorldPop, GlobalBuildingAtlas). Urban areas: `pycensus.census.worldwide.un_metro_areas` (UN functional
urban areas and urban centres). Scoring: `transitLOS` (`src/transitlos/scoring/`, `stop_scores.py`, `level_of_service.py`).

## 2. The fixed map of a new country (the Spain and US maps)

The fixed map scores every place with the standard transitLOS scoring and shows it on one interactive map. The two examples have the
same shape, with different names for the units:

| Level | US (`usa_study/`) | Spain (`spain_study/`) |
|---|---|---|
| unit processed on its own | county | province (52) |
| roll-up | state | autonomous community (19) |
| nation | `build_usa_map.py` | `build_spain_map.py` |

Start from the one nearer to your country (Spain for a country with one official feed portal and a national census with sections; the
US for many agencies and an API census). Read the orchestrator first (`spain_study/main.py` states its stages in its docstring; so does
`usa_study/main.py`), then:

1. **Feeds of a unit**: score the stops. Spain: `nap_fetch.py` and `code/gtfs_stage.py`, `code/gtfs_store.py`. US:
   `code/gtfs_stage.py` (`score_state_stops`) with `code/feed_stage.py`, `feed_fallback.py`, `feed_match.py` and `feed_validate.py`
   (several agencies per state, fallbacks, matching and validation of feeds; the feed lists are in `usa_study/feed_lists/`).
   Scoring is `tools/studytools/stop_scoring.py` (best mode and speed subset per station, 150 m grouping).
2. **Per unit**: street network, level of service, census, H3 resolutions 11 to 7. Spain: `code/province_pipeline.py`,
   `code/census_spain.py`. US: `code/county_pipeline.py`, `code/census_h3.py`.
3. **Roll-up**: the coarser levels (province and community, or county and state), coarser H3, merged stops. Spain:
   `code/ccaa_pipeline.py` (`merge_and_resample_h3`). US: `code/state_pipeline.py` (`merge_counties`, `merge_and_resample_h3`,
   `add_state_districts`), called by `main.py`.
4. **Map of a region**: Spain `code/build_region_map.py` (bilingual, `code/i18n*.py`); US `code/build_state_map.py`. Both call
   `transitlos.map.build` and `geohierarchy.maps`.
5. **Nationwide map**: `build_spain_map.py` / `build_usa_map.py` (lazy statistics and population chunks: `code/stats_chunks.py` and
   `pop_chunks.py` in Spain, `code/stats_chunks.py` and `pop_chunks_usa.py` in the US).
6. **Extra columns** added after the build, as separate scripts so they can be rerun: `add_mitma_mobility.py`, `add_densities.py`,
   `add_density_1km_census.py` (Spain); `add_acs_columns.py`, `add_commute_columns.py`, `add_housing_columns.py`,
   `add_od_los_columns.py`, `add_precinct_elections.py`, `add_voting_districts.py` (US). `rescale_factors_from_logs.py` collects the
   factors the build applied to the counts.
7. **`run.sh`** chains it all with memory caps and ends with the checks. Copy `spain_study/run.sh`.

To start a new country: copy `spain_study/` to `<country>_study/`, rename the units, replace the feed stage with your portal's
downloader (1a) and `code/census_spain.py` with your census loader (1b), set the country's `CityConfig` fields, and keep every other
file. Add the study's tests (see `spain_study/tests/`), a section in this repository's README (the table of studies and `run.sh`), and
run the checks.

## 3. The flexible-scoring map of a new country (the US `usa_flex/` map)

The flex map replaces the fixed score by a score computed in the browser from each station's service (headway by time of day, speed,
mode) and the walking distance to it, so that the user can change the curves or edit routes. It needs the fixed map of the same
country first (it reads its units, census blocks and street networks). The only country example is `usa_flex/`.

Read `transitLOS/src/transitlos/flex/` first: `profiles.py` (a station's service profile), `cells.py` and `core.py` (cells, stations and
walking distances), `catchment.py` (census counts spread over the cells), `scoring.py` (the score), `export.py` (the files the browser
reads), `frontend/` (the page and the route editor). Then the study side:

1. `usa_flex/run_state_buffered.py`: one run per region, with the rules in its docstring (a walking buffer around every station; roads and
   blocks only there; coarser census elsewhere; fine cells below a density are dropped and their counts returned with one factor).
2. `usa_flex/run_us_buffered.py`: all regions under the memory scheduler (`memsched.py`), one job per region.
3. `usa_flex/build_web.py`: the per-region files for the browser. `usa_flex/build_flex_page.py`: the "Flexible scoring" tab added to the
   fixed national map, from `tools/flex_web/flex_tab.js` (shared with the city flex maps) and the scoring variants.
4. `usa_flex/run.sh` chains these, then builds the data for fitting the score to observed transit use (`usa_flex/fit/`, which needs
   PyTorch and a GPU; `usa_flex/fit/score_study/` holds the studies).

For a new country: copy `usa_flex/` to `<country>_flex/`, point `STUDY` at the new fixed map's data, replace the census inputs of
`run_state_buffered.py` with the country's finest census (and its jobs, if any), and rename the regions. The fits are optional: a flex
map needs only steps 1 to 3.

## 4. A new list of cities (the city science study)

`city_science_network/` takes a list of cities, each from its feeds, census and OpenStreetMap, and builds a fixed map and a flex map
for each, plus a combined map. Each city is a folder with `aoi.gpkg` (the processing area), `core.gpkg` (the urban centre), `gtfs/`
(its feeds), `results/` (everything computed) and `map.html`.

To add a city:

1. **Area.** `scripts_un_areas.py` writes the metro area (`aoi.gpkg`) and core (`core.gpkg`) from the UN functional urban area and urban
   centre containing a point; add the city's point to `POINTS`. Where a city is better defined otherwise (a province, a country), keep your
   own file and list it in the script's keep lists.
2. **Config.** One `CityConfig` in `code/city_config.py`: `key`, `display_name`, `geocode_name`, `country` (ISO3), `census_states` (US),
   `pbf_filename` (the Geofabrik extract), `worldpop_filename`, and the flags the city needs (`stops_per_feed` when the feeds' calendars
   differ, `census_module` for a regional census, `exclude_cells_touching_country_border`...). Every value is a fact about the city.
3. **Feeds.** Download with the downloaders of 1a into `<city>/gtfs/`; archive feeds that may vanish (`archive_feeds.py`); build missing
   ones by hand under `assumed_feeds/` with an `ASSUMED.md`. `code/gtfs_audit.py` checks the analysis dates, `code/check_missing_transit.py`
   looks for transit the feeds miss.
4. **Census.** If the country is new to pyCensus, add it (1b) and wire it (section 1b, the second list).
5. **Run.** Add the city to `CITIES` in `run_all.py` and `run.sh`; `run.sh <city>` builds it under the memory scheduler and checks the map.
6. **Flex map.** `flex/run_city_flex.py <city>` (station profiles, street search, cell and station table of `transitlos.flex`) then
   `flex/build_city_flex_web.py <city>` (the map with its editor, from `tools/flex_web/`). Read `flex/README.md`.
7. **Observed transit use** (only to fit and test scores): add a loader to `flex/outcomes.py` that returns one row per unit of the finest
   published geography with the transit share, the commuters or trips behind it and the survey sample; then `flex/build_fit_data.py`
   and `flex/fit_cities.py` (the model is the library `transitlos.flex.fit`). Use units coarse enough for the survey (a few dozen surveyed
   people per unit at least): `flex/FIT_RESULTS.md` shows how much a score can be fitted at each unit size.

For a **different list of cities** that should not be mixed with these, copy `city_science_network/` to a new study folder, empty its
city folders, and change `CITIES`, `code/city_config.py` and the city points in `scripts_un_areas.py`. Do not import code from
`city_science_network`: what is general belongs in `tools/studytools/`.

## 5. Where things are, in one table

| Need | Read |
|---|---|
| a new feed source | `pyGTFSHandler/docs/contributing_downloader.md`, `downloaders/base.py`, `downloaders/usa/ntd.py` |
| a new census country or source | `pyCensus/docs/AGENT_GUIDE_NEW_CITY_OR_COUNTRY.md`, `CONTRIBUTING_SOURCES.md`, `SCHEMAS.md` |
| how a country's census reaches the pipeline | `tools/studytools/config.py`, `pipeline.py` (`_join_census`) |
| the per-area pipeline | `tools/studytools/pipeline.py` (`run_city_study`), `stop_scoring.py`, `params.py`, `inputs.py` |
| the scoring | `transitLOS/src/transitlos/scoring/`, `stop_scores.py`, `level_of_service.py` |
| a fixed country map | `spain_study/main.py`, `usa_study/main.py`, their `run.sh` |
| a flex country map | `transitLOS/src/transitlos/flex/`, `usa_flex/run_state_buffered.py`, `build_web.py`, `build_flex_page.py` |
| a list of cities | `city_science_network/code/city_config.py`, `run_all.py`, `run.sh`, `flex/README.md` |
| the map pages | `transitLOS/src/transitlos/map/`, `geohierarchy/src/geohierarchy/maps/maplibre/render.py`, `tools/flex_web/` |
| any area, no code | `example_metro_area/` (notebook and script) |

## 6. Before you call it done

* The package tests: `python -m pytest` in each package you touched (they run without network); in `tools/`, `city_science_network/`
  and `usa_study/` for the studies.
* A known-total check of every new source (population, households or stops against an independent figure).
* The study's `run.sh` ends with `checks/check_map_data.py` (the data files of a map) and `checks/test_map.py` (the page opens and
  draws, with a screenshot; needs Playwright). Both must pass.
* One memory-capped run of the smallest unit end to end before the whole country or list.
* Nothing in the commit that is a key, a local path, a machine or hosting note, or a file over 10 MB. Docs for a new package
  module build with `mkdocs build --strict` in that package.
