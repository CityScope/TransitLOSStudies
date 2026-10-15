# usa_study

Nationwide US transit level-of-service study. See `prompt.txt` for the
full original spec, and each `code/*.py` module's own docstring for how
each stage actually works.

## Setup on a new machine

This directory is not itself an installable package -- it's a script
that imports from 5 sibling repos, expected to sit next to it in the same
layout as this checkout (`main.py` resolves them via relative paths):

```
AccessibilityMIT/
├── pyGTFSHandler/
├── pyCensus/
├── transitLOS/
├── UrbanAccessAnalyzer/
├── geohierarchy/
└── TransitLOSStudies/
    └── usa_study/   <- this directory
```

Install all 5 in dependency order (each `pip install -e .` picks up its
own declared deps -- `us`/`pygris`/`h3`/etc. are already listed in each
package's own `pyproject.toml`, nothing extra to install by hand):

```bash
pip install -e geohierarchy
pip install -e pyCensus
pip install -e pyGTFSHandler
pip install -e UrbanAccessAnalyzer
pip install -e transitLOS
```

API keys: the **Census API key** is needed in practice (a 50-state run hits
the unauthenticated rate limit). Put it in the gitignored
`pyCensus/api_keys.json` as `{"us_census": "<key>"}` (or set
`US_CENSUS_API_KEY`); `pycensus/api_keys.py` resolves it. NTD (GTFS) needs
no key -- `pyGTFSHandler`'s NTD downloader works unauthenticated (a Socrata
app token via `SOCRATA_APP_TOKEN` only raises the throttling tier).

## Running

`main.py` is the single entry point for the whole experiment:

```bash
cd TransitLOSStudies/usa_study
python main.py ALL                  # THE FULL EXPERIMENT: 50 states + DC, then the nationwide map
python main.py MA                   # one state (+ its own map.html)
python main.py MA RI CT             # several states
python main.py ALL --force          # redo everything even if output exists
python main.py ALL --skip-map       # data stages only, no map at all
python main.py ALL --state-maps     # also build every state's own map.html
python main.py ALL --no-streets     # nationwide map without the street layer
python main.py --usa-map-only       # only (re)build the nationwide map from existing state outputs
python build_usa_map.py --states AL,DE --no-streets   # smoke-test the national map on a subset
```

Run it in the background (`nohup`/`tmux`): a full run takes about a day.
It is **resumable**: a state with `stops.geoparquet` skips stage 1, a county
with `county.geoparquet` is skipped (and its h3 files are rebuilt if they
predate the current h3 scheme, see below), stages 3-5 always rerun (cheap).
Per-state and per-county failures are logged and skipped, never fatal; the
log ends with the list of states that failed entirely, which a rerun of
`python main.py ALL` retries. Expect several hundred GB of output.

Stages: (1) NTD GTFS download + stop scoring, (2) per-county street network +
level of service (LOS) + census/h3, (3) state rollup, (4) per-state map
(optional), (5) the nationwide map.

Output, per state under `data/<StateName>/`:

- `stops.geoparquet` -- scored GTFS stops (stage 1).
- `counties/<GEOID>/` -- `street.geoparquet` (edges with LOS > 0 only),
  `block`/`blockgroup`/`tract`/`county.geoparquet`, `h3_res{7,9,11}.geoparquet`
  and the marker `h3_scheme_v2.txt` (stage 2).
- `county.geoparquet`, `h3_res{3,4,5,7,9,11}.geoparquet`, `place`/
  `congressional_district`/`state_legislative_district`/
  `school_district`/`state.geoparquet` -- state rollup (stage 3).
- `map.html` + `tiles/` -- only with `--state-maps` (or a non-ALL run).

Nationwide: `data/_usa/map.html` + `data/_usa/tiles/` (stage 5; serve the folder
over HTTP, the page loads PMTiles from `tiles/`).

Downloaded OSM extracts are cached under `osm_files/` (shared; downloads are
resumable and renamed only when complete). The pyCensus cache lives under
`data/pycensus_cache/`. Both are gitignored and regenerable.

## Experiment design (final)

**H3 layers (same rules in every county).**
- Res 11: cells wherever LOS > 0 (any population density, street edges buffered 40 m) plus every tract denser
  than 1,000 people/km2. A cell's LOS is the mean LOS of the *scored* street edges (LOS > 0) in it. A cell with
  no scored edge gets **0 if it touches any street** (street presence comes from the full county street network,
  cached in `_network_cache`) and **no score (null) if it touches no street at all**.
- Res 9 and 7: resampled from res 11 by population-weighted mean over the *scored* children only; cells with no
  score are ignored (their population leaves the denominator). Population comes from census blocks
  (area-weighted), so people outside res 11 count as LOS 0. Res 7 covers the whole county; res 9 the parents of
  res 11 cells.
- Res 5, 4, 3: the state rollup resamples up from res 7 (population-weighted means for LOS and share columns,
  sums for counts). In the nationwide map, border cells that appear in two states are merged at res 7 first.
- Counties without transit (or with no streets near the stops) skip the street network entirely: only census/h3
  data, LOS 0, street presence assumed.
- Counties finished under an older h3 scheme are rebuilt automatically (`rebuild_county_h3`, marker
  `h3_scheme_v3.txt`) from the saved scored edges + cached network -- the same inputs a fresh run uses, so the
  result is identical to a recomputation.
- Connecticut: TIGER 2022+ replaced the 8 counties with 9 planning regions (new codes) while census/ACS/election
  data stay keyed by the legacy counties, so Connecticut uses the last legacy TIGER vintage (2021) at every level.

**Maps (MapLibre + PMTiles).**
- Layers: h3 hexagons/circles (res 3, 4, 5, 7, 9, 11), census levels
  (county, tract, block group, block), "special" levels (places, congressional/
  state-legislative/school districts, states), streets, stops.
- Res 7 stays drawn under the finer levels (`set_fallback_level`, overzoomed,
  no extra tiling), so where res 9/11 don't exist you still see res 7.
- Special levels: zoomed out (< zoom 5, `special_group_fallback_zoom`) the
  states layer is shown even when another special level is picked; the picked
  level shows from that zoom up.
- Tiles are generated natively only down to zoom 14
  (`GEOHIERARCHY_MAX_NATIVE_TILE_ZOOM`, set by `main.py`/`build_usa_map.py`);
  MapLibre overzooms beyond it. Special levels are capped at native zoom 10.
  Without these caps a few huge polygons (Alaska blocks, state-sized districts)
  explode into millions of tile features.
- Nationwide map (`build_usa_map.py`, stage 5 of `ALL`): coarse levels (h3 3/4/5/7, county, tract, special levels,
  stops) are built in one process from every state's files (`code.build_state_map.assemble_map_inputs`). The heavy
  levels -- h3 res 9/11, census block groups/blocks, **streets** -- are tiled **one state at a time**
  (`code.external_tiles`) into `tiles/<kind>/<level>/chunk_<State>.pmtiles`; the page loads those chunks lazily, so
  streets and fine layers only download at high zoom and nothing nationwide has to fit in memory (a single process
  was OOM-killed at ~90-126 GB). The in-memory frames for those levels are small representative samples (schema +
  value domains).

## Stop scoring (stage 1) -- rules that matter

- **Analysis window 06:00-22:00 for every feed** (`ANALYSIS_START_TIME`/`ANALYSIS_END_TIME` in `code/gtfs_stage.py`,
  same as `city_science_network`). Headway is sum(gap^2)/window with the overnight break counted as one gap, so a
  24 h window turned every route's night break into 30-90 min of "headway" (BART Embarcadero: 5 min became 30 min;
  SF Muni median headway 81 min instead of 11). Never score on the full day.
- Feeds with service in the analysis dates (2026-10-01..10-15) are loaded together in one `Feed` (single stop
  clustering, headways combined across them). Feeds WITHOUT service in the window (expired calendars) are scored
  alone on their own representative weekday (checked against the feed's typical weekday: ratio ~1.00) and are NOT
  combined with anything; all groups' stop tables are concatenated even at the same location, so isochrones are
  computed for every stop and the LOS step takes the best access per street edge.
- A downloaded NTD "weblink" that is not static GTFS (a GTFS-RT endpoint such as Fairfax Connector's, a web page, a
  directory) counts as broken and goes through the Mobility Database fallback.
### Feed discovery order

For every NTD agency whose feed is broken/partial, `acquire_state_feeds` tries, in order: pinned overrides
(`KNOWN_FEED_OVERRIDES`, `TRANSITLAND_OVERRIDES`, `DIRECT_URL_OVERRIDES`) -> Mobility Database name match ->
Transitland name match -> **fuzzy name + spatial match** (`code/feed_match.py`): the agency city is geocoded,
Mobility Database feeds whose bounding box contains it and Transitland agencies within 8 km are ranked by name
similarity, the top 3 are downloaded and checked against their own stops (stops within 5/15 km, stop names
containing the city, trips, service end). A candidate is used automatically only when verified (similarity >= 0.45,
trips > 0, >= 3 stops within 15 km and city name hits or >= 3 stops within 5 km); every candidate and its evidence is
written to `feed_report.json` (`candidates`) for manual review. Resolution `fuzzy_match` appears in `feed_exceptions.csv`.
Requires the git-ignored `pyGTFSHandler/api_keys.json` (Transitland key); run `python main.py <STATES> --refresh-stops`
to re-acquire feeds for states that had unresolved agencies.


- **Stop grouping: 150 m, strict** (applies to every feed of every state: `Feed(stop_group_distance=150)` with `GTFS_STOP_GROUPING=complete`). Stops are grouped into stations by `GTFS_STOP_GROUPING=complete` (set in `main.py`): no two
  stops of a group are farther apart than 150 m (complete linkage), EXCEPT stops that share a declared `parent_station`
  (multi-entrance stations stay whole: such pairs count as distance 0). The library's default groups transitively (neighbour of neighbour), which merged whole downtown streets
  into one "station" scored from one mean location (CTA Loop: 395 stops -> 35 stations, clusters up to 2.9 km); strict
  grouping gives 155 stations there and a maximum in-group distance of 149.8 m.
- **Feed packaging** (`code/feed_stage.sanitize_feed_folder`): nested folders are flattened, file names normalised to
  lower-case `.txt`, padded header names stripped, empty optional files removed; partial NTD feeds of large agencies
  (`REPLACE_NTD_AGENCIES`, e.g. Denver RTD) are replaced by the full feed.

## Cell and census scoring

- Res 11 cell LOS = mean LOS of the scored street edges, computed from points sampled every 10 m along each edge
  (`_scored_edge_cell_means`); `geohierarchy.edges_to_level` is NOT used for 30 m cells because it drops cells an edge
  only grazes (about a third of the cells touching scored streets came out as 0).
- Census levels (block ... county) average LOS over ALL edges of the county (zeros included).

## Operational lessons built into the code

- `pycensus/api_keys.py` was gitignored by accident (`**/api_keys.py`); it is
  now tracked (`.gitignore` exception).
- Geofabrik's `-latest` link is sometimes missing; the download falls back to
  the newest dated file, retries, resumes and renames only on completion.
- A bad/duplicate GTFS input never fails a state: unreadable zips are skipped,
  duplicate `stop_id`s are resolved (same parent station: keep first; different
  parents: drop the stop and every trip through it), a failed combined feed
  load retries feed by feed.
- Census TIGER downloads: stub files (e.g. "Request Rejected" pages) are
  discarded and retried; if a year stays blocked the adjacent vintage is used
  (ND blocks 2023 -> 2024). Counties with a broken `areawater` file are skipped
  in the water erase (CA Mono). LODES years missing for a state fall back to
  the newest available (AK -> 2016).
- Transient Census API errors fail only that county: failed counties are retried automatically after 1, 3 and 10
  minutes (`COUNTY_RETRY_WAITS_S`), and states that fail entirely are retried after 5 and 15 minutes.
- Feed exceptions: NTD weblinks that are dead / blocked (401 API key, 403/406 bot filters, SSL) are retried with a
  browser User-Agent and an SSL fallback; still-broken ones are looked up in the Mobility Database public catalog
  (`code/feed_fallback.py`, archived copies at `files.mobilitydatabase.org/<id>/latest.zip`, no key needed; known
  agencies are pinned in `KNOWN_FEED_OVERRIDES`). Every state writes `feed_report.json` and the run writes
  `data/feed_exceptions.csv` (agency, reason, resolution); `python main.py <states> --refresh-stops` redoes stage 1
  after improving feeds and only recomputes counties whose stops changed.

## Known gaps / documented simplifications

See `usa_study_pipeline_progress.md` (Claude's own session memory, not
checked into this repo) for the full running log, but the load-bearing
ones:

- **Elections**: only President 2000-2016/2024 and Senate 2024 are
  implemented (`pycensus.countries.usa.elections.loader`'s own
  docstring has the full reasoning) -- House races and 2018/2020/2022
  aren't available from any source this project found that isn't gated
  behind a Harvard Dataverse Guestbook survey no scripted request can
  satisfy.
- **"Local district"** (city council wards / county commissioner
  districts) was dropped entirely as a geography level -- confirmed no
  national dataset exists for it.
- **Map selector**: place/congressional/state-legislative/school-district
  levels share the same one census dropdown as the real admin hierarchy
  (county/tract/blockgroup/block), not a separate selector tier -- see
  `code/build_state_map.py`'s own docstring.
- **BRT route_type marking**: GTFS has no route_type value for Bus Rapid
  Transit, so this project repurposes code `702` ("Express Bus Service")
  as a project-internal marker, written only into downloaded feed
  copies, never the source agency's own data (`pyGTFSHandler/downloaders
  /usa/brt.py`). Only a handful of agencies have a verified route-level
  mapping seeded in `KNOWN_BRT_ROUTES`; the rest get a manual-review
  warning logged rather than a guess.


## Outputs of a complete run

`python main.py ALL` writes one log (`data/usa_study.log`), `data/ptlos_by_state.csv` + `data/ptlos_usa.json` (the PTLOS score: population-weighted mean level of service over H3 res-7 cells), `data/feed_exceptions.csv`, and `data/run_manifest.json` (parameters, git HEAD of the six repos, fingerprint of every GTFS file used). Feeds already on disk are reused, so to reproduce a result exactly keep `data/*/gtfs`; a fresh machine re-downloads the latest feeds (the NTD/Mobility Database/Transitland sources change over time), so the manifest fingerprints show whether the same feed versions were used.
## Nationwide map internals (how it stays fast)

- `data/_usa/map.html` is ~16 MB: stops are **not** embedded. Each state's stops are `stops_chunks/<State>.json`, fetched when the
  view reaches zoom 7 and intersects the state (at most 10 states are kept in memory).
- The stats panel (distribution / regression / census) reads the H3 res-5 grid (~35k cells) embedded in the page.
- Drawn lines: **Compute** fetches only the `pop_chunks/` squares (0.04 deg, res-11 cells; res-7 cells where no res-11 exists, see
  `code/pop_chunks_usa.py`) around the line (~0.5 s) and recolors the res-11 hexagons. Census-polygon recolor needs `geoid_chunks/`
  (not built nationwide; the console shows a harmless 404).
- `python build_usa_map.py --page-only` rewrites only the page + chunks and reuses the tiles on disk (minutes instead of hours).

## Stats panels: scope, comparison, scenario, address search

- Every stats tab (distribution / regression / census) has **Level** and **Compare with** buttons. Level: Nation, State, County or
  Place (County/Place only after a state is picked, one of them at a time; clicking an active State/County/Place again deactivates it);
  type in the box to filter (best matches first, 8 shown with their population). Compare with adds None. Zooming out to a view that
  covers several states resets to the nation with no comparison; zooming into one state selects it automatically.
- Nation data = `stats_chunks/Nation.json.gz`: a seeded, stratified (population x level-of-service) sample of ~100k of the 1.3M
  res-7 cells with count columns rescaled per stratum, so it is at the SAME resolution as the state files, keeps the exact national
  population (327,267,727 vs 327,268,274) and weighted level of service (21.61 vs 21.60); the res-4 grid embedded in the page is only
  the instant fallback while it loads. State / county / place data = `stats_chunks/<State>.json.gz` (res-7 cells with
  every census / ACS / jobs / election variable, `code/stats_chunks.py` + `code/census_h3.py`), fetched on demand.
- A Scenario selector (Original / With edits) is next to the scope buttons and mirrors the top bar.
- The address box (top left) geocodes with OpenStreetMap (Nominatim) and flies to the point at zoom 6.
- "More housing / more transit" overlay (`code/equity.py`): one national regression of level of service on log population density
  (res-7 cells with LOS > 0); more housing = above +1 std, more transit = below -1 std and density > 1000 people/km2.

- **No edit mode at nation level**: while the view covers several states (or below zoom 5) the Draw button and line toolbar are
  hidden, 'With edits' is disabled and nation-scope plots never apply edits; zooming into a single state brings them back.

## Data notes and checks worth knowing

- County rows used to be dropped for counties whose geometric centre lies outside their own boundary (island / multi-part counties:
  Honolulu, San Francisco, Miami-Dade, Aleutians West, one Missouri county); `county_pipeline` now tests with a point guaranteed to
  be inside the polygon. Such counties need `python main.py <STATE> --skip-map` after clearing their folder to be rebuilt.
- pyCensus' `acs5_commuteTimeMean` is mapped to ACS B08303_001 = the NUMBER of commuters, not a mean. `code/census_h3.py` therefore builds
  `mean_commute_time_minutes` from the aggregate travel time B08013_001 (cached in `data/_cache/acs_commute_aggregate.json`) divided by it
  (national weighted mean 26.6 min). The pyCensus schema entry itself still needs fixing.
- Development overlay tiles are written per state (`code/external_tiles.py::make_dev_writer`, flagged cells only: 51 sources instead of
  5,000+) and the overlay only shows from zoom 8 (`dev_min_zoom` in `stats_scope_cfg`).
- `python build_usa_map.py --page-only --census-tiles` rewrites the page, the stats / population / stops chunks and the census county / tract
  tiles (use it after counties were reprocessed); without `--census-tiles` only the page and chunks.
- Each stats-panel state table is cached in the browser (at most 4 states + the nation); after 14 visited states the page holds ~400 MB
  of JS heap.
