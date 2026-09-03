# TransitLOSStudies

A collection of transit level-of-service (LOS) studies, each in its own top-level
folder with the code needed to run it. This repo itself is not a Python package —
each study folder is its own project with its own `pyproject.toml`/dependencies.

## Studies

- **[city_science_network/](city_science_network/)** — the current, actively
  maintained multi-city transit-LOS study (City Scope / CityScope org), covering
  11 cities (Andorra, Beer Sheva, Boston, Concepción, Gipuzkoa, Guadalajara,
  Hamburg, San Francisco, Shanghai, Taipei, Toronto). Built on the sibling
  `transitLOS`, `UrbanAccessAnalyzer`, `geohierarchy`, and `pyCensus` packages.
  See its own README for how to run it.

Only each study's code is tracked here — the data each study downloads and the
results/maps it generates are gitignored (see `.gitignore`); running a study
regenerates them locally.

## Adding a new study

Add a new top-level folder with its own `pyproject.toml` (using `uv`) and code,
and list it above. If it shares data/results-style generated output, extend
`.gitignore` with the same ignore-everything-then-whitelist-code pattern used for
`city_science_network/`.
