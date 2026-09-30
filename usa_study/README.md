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

No API keys are required to run this study. `pyGTFSHandler`'s NTD
downloader works fully unauthenticated (see
`pyGTFSHandler/downloaders/usa/ntd.py`'s own docstring) -- a Socrata app
token only raises the throttling tier, it's not needed for correctness.

## Running

```bash
cd TransitLOSStudies/usa_study
python main.py MA              # one state
python main.py MA RI CT        # several states, one process each
python main.py ALL             # every US state + DC
python main.py MA --force      # redo stages even if output already exists
python main.py MA --skip-map   # stages 1-3 only, skip building map.html
```

Each state's output lands under `data/<StateName>/`:

- `stops.geoparquet` -- scored GTFS stops (stage 1).
- `counties/<GEOID>/` -- one folder per county: `street.geoparquet`,
  `block`/`blockgroup`/`tract`/`county.geoparquet`, `h3_res11`/
  `h3_res9.geoparquet` (stage 2).
- `county.geoparquet`, `h3_res{5,7,9,11}.geoparquet`, `place`/
  `congressional_district`/`state_legislative_district`/
  `school_district.geoparquet` -- state-wide rollup (stage 3).
- `map.html` + `tiles/` -- the interactive map (stage 4).

Downloaded state OSM extracts are cached under `osm_files/` (shared
across states/counties -- not deleted between runs). The pyCensus
parquet cache lives under `data/pycensus_cache/` (shared across every
state, so re-running a second state never re-fetches census data the
first state already pulled). Both are gitignored -- fully regenerable,
not source.

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
