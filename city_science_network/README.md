# city_science_network

A City Scope (CityScope org) case study: a multi-city transit
level-of-service (LOS) pipeline. For each city it downloads/loads GTFS
transit feeds, street networks, census/WorldPop population data, and an
area-of-interest boundary, computes stop- and edge-level transit LOS and
population-weighted accessibility on an H3 grid, and renders the result as
an interactive map plus regression/ANOVA figures comparing LOS against
demographics.

The pipeline logic lives in `code/` (`pipeline.py`, `city_config.py`,
`city_core.py`, `cli.py`, `params.py`, `stats.py`, `figures.py`,
`h3_population.py`, `combined_map.py`). It is built on several sibling
City Scope packages, declared as local path dependencies in
`pyproject.toml` (`[tool.uv.sources]`) — most importantly `transitlos`
(also a CityScope GitHub repo) for stop/LOS scoring and map rendering,
plus `UrbanAccessAnalyzer`, `geohierarchy`, and `pyCensus` for AOI
handling, H3 aggregation, and census/WorldPop ingestion respectively.
These siblings are expected to be checked out next to this repo (e.g.
`../transitLOS`) — they are not included here.

Each of the 11 study cities (`andorra/`, `beerseba/`, `boston/`,
`concepcion/`, `gipuzkoa/`, `guadalajara/`, `hamburg/`, `san_francisco/`,
`shanghai/`, `taipei/`, `toronto/`) has its own `run.py` entry point that
configures and runs the pipeline for that city. Everything else those
directories produce at runtime — `aoi.gpkg`, `gtfs/`, `streets/`,
`results/`, `figures/`, `map.html`, `map_tiles/`, `pop_chunks/`,
`worldpop_demographic/`, `logs/`, etc. — is generated output and is
**not** tracked in this repo (see `.gitignore`); running a city's
`run.py` regenerates it locally.

## Running a city

```bash
uv run python boston/run.py
```

Each city script writes to its own folder: `results/` (geoparquet edges +
core/metro H3 grids), `figures/` (regression and ANOVA jpgs), and
`map.html` + `map_tiles/` (the interactive map, see below).

`main.py` runs every city in sequence and then builds the cross-city
comparison figures in `city_science_network/figures/`.

## Viewing the map

Each city's `map.html` is a self-contained Leaflet page backed by a local
`map_tiles/` folder of vector tiles (`.pbf`), not embedded GeoJSON — this is
what keeps `map.html` itself tiny (tens of KB) even for a full metro-area H3
resolution-10 grid.

Because the page fetches tiles over `fetch()`, opening it directly from disk
(`file://`) will be blocked by the browser's local-file restrictions — serve
it over HTTP instead. From the city folder (e.g. `boston/`):

```bash
python -m http.server 8000
```

then open `http://localhost:8000/map.html` in a browser. `map.html` and its
`map_tiles/` folder must stay in the same directory.

## Boston GeoPackage export

Boston (the pipeline's smoke-test city) additionally gets its parquet result
layers exported to GeoPackage for viewing in QGIS/ArcGIS:

```bash
uv run python boston/run.py --export-gpkg
```

Writes `boston/results/access_edges.gpkg`, `boston/results/core/h3_grid.gpkg`,
and `boston/results/metro/h3_grid.gpkg` alongside the existing `.parquet`
files. Other cities only produce `.parquet` outputs.

## Running all cities / the combined map

`main.py` runs every city's pipeline in sequence (skipping/continuing past
per-city failures), rebuilds the cross-city comparison figures in
`figures/`, and rebuilds `combined_map.html` — a single map letting you
switch between cities — after each city, whether it succeeded or was
skipped. `scripts_relaunch_maps.sh` and `scripts_relaunch_maps_resume.sh`
are operational helpers for re-running/resuming the multi-city map rebuild
(e.g. after a crash) without redoing already-cached pipeline stages.

## Tests

`tests/` holds the pytest suite (`uv run pytest`).
