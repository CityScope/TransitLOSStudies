"""Build `city_science_network/combined_map.html`: a multi-city wrapper around each city's own `map.html`.

Architecture (documented per the task spec, which asked to record this choice):

  * LEAST-INVASIVE, iframe-based wrapper. Each city already has its own
    fully-built, self-contained `map.html` (tiles, scenario editor, stats
    panel, everything). Rather than merging N cities' maps into one page
    (which would mean rearchitecting `transitlos.map.build`'s tile
    layering, scenario editor state, and stats panel to be city-aware),
    this wrapper just embeds the *currently selected* city's `map.html` in
    an `<iframe>` and swaps `iframe.src` when the user picks a different
    city. Every city's map keeps working exactly as it already does,
    completely unmodified.
  * Written as a **standalone generator script** (this file), not a new
    function inside `transitlos/map/build.py`. `build.py` is being
    concurrently edited by another agent (dual-dropdown/2nd-column
    stats-panel work in `_stats_panel_html`/`_stats_panel_js`/
    `getStatsData`); a combined-page template has no real reason to live
    in that module (it doesn't build a Folium map or touch tile layers at
    all) and keeping it here avoids any merge collision.
  * The city selector is a real DOM child INSIDE the iframed city's own
    `#top-center-bar` (the pill a real per-city `map.html` already shows,
    containing `#topBarScenarioSelect` -- the scenario dropdown -- and
    `#topBarResults` -- the population-weighted level of service readout,
    separated by a thin divider). `injectCitySelector()` (page template JS
    below) prepends a live `<select>` of the same city roster as this page's
    own dropdown, styled identically to `#topBarScenarioSelect` (transparent,
    borderless, same font), followed by the same divider style already used
    between that select and the results readout -- so City reads as one more
    segment of that SAME bar, immediately left of the scenario select, not a
    second box next to it. Selecting an option there directly changes
    `iframe.src` (a real interactive control, not an inert label). The
    wrapper's own `#cityDropdown` overlay/`#citySelectorBar` pill stay as a
    fallback, shown only if injection fails (same-origin access throws, e.g.
    this page opened over `file://`, or a future map build genuinely lacks
    `#top-center-bar`) -- kept reachable rather than erroring, but not the
    normal path.
  * The stats panel is the SAME `_stats_panel_html`/`_stats_panel_js`
    building blocks a per-city `map.html` uses for its own bottom-left 📊
    panel -- "Place rank" (cross-city) first, then Distribution/Regression/
    ANOVA/Discretization -- but computed over the CURRENTLY SELECTED place's
    own per-h3-cell data (`window.__statsData`, fetched fresh from that
    place's `results/stats_data.json` on every city switch by
    `__loadStatsDataForCity`) rather than a cross-city column average.

Manifest: any `<city_dir>/map.html` this script is told about (see
`CITY_MANIFEST` below) is offered in the selector. A place is included in
the stats panel's "Place rank" tab only if it also has a
`<city_dir>/results/summary.json` and `results/stats_data.json` (written by
`code.pipeline.write_city_summary`/`build_city_map`) -- a place missing
either shows as unranked ("--") rather than being silently dropped.

Usage: `uv run python code/combined_map.py` from `city_science_network/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # city_science_network/

try:
    from code.city_config import CITY_CONFIGS
except ImportError:
    # Allow running as a bare script (`python code/combined_map.py`) in
    # addition to the normal package import (`from code.combined_map import
    # build_combined_map`, used by `main.py`, or `python -m code.combined_map`
    # -- the latter is the recommended way to run this file standalone, see
    # `Usage` above). A bare script invocation puts this file's own
    # directory (city_science_network/code/) at sys.path[0], so `import code`
    # resolves to the stdlib `code` module (interactive-interpreter helper)
    # instead of this package, and caches that wrong module in
    # sys.modules['code'] -- hence the explicit `del` before retrying with
    # city_science_network/ correctly placed ahead of it on sys.path.
    sys.modules.pop("code", None)
    sys.path.insert(0, str(ROOT))
    from code.city_config import CITY_CONFIGS

from transitlos.map import build_download_panel_block
# `_stats_panel_html`/`_stats_panel_js` are the SAME building blocks a real
# per-city `map.html` uses for its own bottom-left stats panel (Place rank
# tab first, then Distribution/Regression/ANOVA/Discretization for whichever
# place's data happens to be loaded into `window.__statsData`). Reused here
# directly -- rather than through the now-removed `build_place_rank_panel_block`
# convenience wrapper, which hardcoded `place_rank_only=True` -- since the
# combined page now wants the FULL panel, just re-pointed at the currently
# selected city's own `results/stats_data.json` on every city switch (see
# `__loadStatsDataForCity` in the page template's JS below) instead of a
# fixed roster of per-city h3 data baked in at build time.
from transitlos.map.build import (
    _POPUP_ABS_TO_REL,
    _stats_panel_helper_js,
    _stats_panel_html,
    _stats_panel_js,
)

# (key, display_name, path-to-city-dir-relative-to-city_science_network/).
# Derived from `code.city_config.CITY_CONFIGS` (the same roster
# `main.py`'s `CITY_KEYS` walks) so every city `main.py` knows about is
# automatically offered here the moment its `map.html` exists on disk --
# `_available_cities` below filters out anything not yet built, so it's
# safe to always list the full roster rather than hand-editing this tuple
# as each city finishes. `boston` here means the metro study built by
# `city_science_network/boston/run.py`. `boston_city` -- a distinct, separately
# maintained study living as a sibling of `city_science_network/` entirely (see
# `TransitLOSStudies/boston_city/run.py`) -- is deliberately NOT included
# here (explicit user request, 2026-08-30: "boston city should not be part
# of combined map. Only the CS cities in the CS folder."), even though an
# earlier version of this file manually appended it as an extra entry.
CITY_MANIFEST = tuple(
    (key, cfg.display_name, key) for key, cfg in CITY_CONFIGS.items()
)


# Column vocabulary for the unified stats panel's Place-rank tab weight-column
# selector AND (doubling as) the Distribution tab's "Distribute by"/overlay
# column list -- both are the `count_fields` argument `_stats_panel_html`
# takes. Matches `code.pipeline.WEIGHT_COLUMN_CANDIDATES` exactly -- itself
# sourced from `pycensus.schemas.global_schema.json`'s `"type": "count"`
# entries, filtered to drop anything matching
# `code.pipeline._CENSUS_RATE_SUBSTRINGS` or already a
# `SHARE_COLUMNS`/`RATE_SOURCE_COLUMNS` key -- kept as a parallel constant
# here (rather than importing `code.pipeline`, a heavy geopandas/pandas
# dependency this page-generator script otherwise has no need for) since the
# two lists must simply agree on key->label, not share code;
# `transitlos.map.build`'s own `field_label` humanizes each key into the
# option text shown, so no separate label list is needed here beyond the key
# itself. The panel's own JS excludes, per place, whichever of these columns
# that place's `stats_data.json` doesn't actually carry, rather than
# defaulting to 0/1.
WEIGHT_COLUMN_CANDIDATES = (
    ("population", "Population"),
    ("households", "Households"),
    ("adultPopulation", "Adult population"),
    ("femalePopulation", "Female population"),
    ("malePopulation", "Male population"),
    ("under18Population", "Population under 18"),
    ("over65Population", "Population over 65"),
    ("cars", "Cars"),
    ("educationPrimaryOrLessPopulation", "Population, primary education or less"),
    ("educationSecondaryPopulation", "Population, secondary education"),
    ("educationUniversityPopulation", "Population, university education"),
    ("educationOtherTertiaryPopulation", "Population, other tertiary education"),
    ("laborForce", "Labor force"),
    ("employedResidents", "Employed residents"),
    ("unemployedResidents", "Unemployed residents"),
    ("workplaceJobs", "Workplace jobs"),
    ("carCommuters", "Car commuters"),
    ("publicTransportCommuters", "Public transport commuters"),
    ("walkCommuters", "Walk commuters"),
    ("bikeCommuters", "Bike commuters"),
    ("motorcycleHouseholds", "Motorcycle households"),
    ("bicycleHouseholds", "Bicycle households"),
    ("carHouseholds", "Car households"),
    ("twoCarHouseholds", "Two-car households"),
    ("povertyPopulation", "Poverty population"),
    ("foreignBornPopulation", "Foreign-born population"),
    ("minorityEthnicityPopulation", "Minority-ethnicity population"),
    ("religiousPopulation", "Religious population"),
    ("minorityReligionPopulation", "Minority-religion population"),
    ("populationAndJobs", "Population and jobs"),
)


# Column vocabulary for the unified stats panel's Regression/ANOVA tabs
# (the `regression_fields` argument `_stats_panel_html`/`_stats_panel_js`
# take), per the user's explicit ask ("relative columns for Anova and
# regression are with the .share or .density for population and population
# and jobs or jobs option"). Matches `code.pipeline.RELATIVE_STATS_BASE_COLUMNS`
# x {share, density} -- 3 base quantities x 2 relative modes = 6 options, kept
# as a parallel constant here (rather than importing `code.pipeline`, a heavy
# geopandas/pandas dependency this page-generator script otherwise has no
# need for) since the two lists must simply agree on key->label.
#
# See `code.pipeline.RELATIVE_STATS_BASE_COLUMNS`'s own comment for the full
# design-decision writeup on what ".share"/".density" mean at this
# cross-city level (within-city population-weighted-averaged relative
# values per city -- NOT a literal "share of all cities combined", since
# `pycensus.accessors`'s `.share`/`.density` are within-geography
# derivations with no natural cross-city analog).
# Real bug (2026-08-28, live "ANOVA tab is empty" report): these dotted keys
# were never anything `_stats_panel_js`'s `renderAnova`/`__anovaDiff` could
# actually read. `_stats_panel_js` (reused as-is below, the SAME per-city
# machinery `code.pipeline._stats_json_data` builds `results/stats_data.json`
# for) indexes `window.__statsData` by PLAIN per-h3-cell column name --
# `data[field]`, an array -- and that JSON is fetched straight from each
# city's own file, whose keys are exactly whatever real, undotted column
# names that city's `relative_fields(all_fields)` picked (see
# `code.pipeline._stats_panel_block`), e.g. `pop_density`/`pop_jobs_density`.
# No per-cell column named `"population.share"` (or any `.share`/`.density`
# suffixed key) is ever written there -- that `<col>.share`/`<col>.density`
# naming is `RELATIVE_STATS_BASE_COLUMNS`' cross-city SCALAR summary
# vocabulary (one number per city, `_relative_column_stats_for_grid` in
# pipeline.py, feeding the separate "Stats overview" city-ranking selector
# above), not a per-cell derivation anything computes on the fly here. Every
# `data[field]` lookup for these six keys was therefore always `undefined`,
# so `__anovaDiff` returned `null` for all six fields and `renderAnova`'s
# result list -- and therefore the whole tab -- was always empty, regardless
# of which city was selected. Swapped for the two real per-cell relative
# columns that DO land in every applicable city's `stats_data.json` (the
# same names `STATS_OVERVIEW_COLUMNS` documents): `pop_density` (present
# wherever `population` is) and `pop_jobs_density` (wherever a jobs source
# was joined; simply absent -- not erroring -- for a city without one, same
# graceful-exclusion contract every other field here already has).
# Item 4 additions (2026-08-30): `population_density`/`worldpop_population_density`/
# `jobs_density`/`jobs_and_population_density`/`jobs_share` -- the same five
# generically-named derived columns `code.pipeline._add_derived_density_columns`
# now adds to every census level and h3 resolution wherever their real inputs
# exist (mirrors `pop_density`/`pop_jobs_density` above: absent -- not
# erroring -- for a city without the underlying source, same graceful-
# exclusion contract). This static tuple is only ever the page's INITIAL
# (pre-fetch) state -- `__rebuildStatsFieldSelectors` below immediately
# overrides it, client-side, from whichever real relative-column keys the
# selected city's own `stats_data.json` actually carries (see that
# function's docstring) -- but keeping it in step avoids a one-frame flash
# of a stale/incomplete field list before that first fetch resolves.
ANOVA_REGRESSION_COLUMNS = (
    ("pop_density", "Population density"),
    ("pop_jobs_density", "Population + jobs density"),
    ("population_density", "Population density (census)"),
    ("worldpop_population_density", "Population density (WorldPop)"),
    ("jobs_density", "Jobs density"),
    ("jobs_and_population_density", "Jobs + population density"),
    ("jobs_share", "Jobs share"),
)


def _available_cities() -> list[dict]:
    """Cities from `CITY_MANIFEST` whose `map.html` actually exists on disk."""
    out = []
    for key, fallback_name, rel_dir in CITY_MANIFEST:
        city_dir = (ROOT / rel_dir).resolve()
        map_path = city_dir / "map.html"
        if not map_path.exists():
            continue
        # `display_name` is ALWAYS `CITY_MANIFEST`'s own name (`fallback_name`
        # above), never overridden from `summary.json` -- `CITY_MANIFEST` is
        # the one place that already knows about the whole roster (and, back
        # when a manually-appended `boston_city` entry lived alongside the
        # config-driven `boston` metro study, is what disambiguated them --
        # see git history/`CITY_MANIFEST`'s own comment; `boston_city` was
        # removed from this roster entirely per explicit user request,
        # 2026-08-30), so it's the only source of truth for display names
        # here rather than trusting each city's own `summary.json`.
        display_name = fallback_name
        metro_median = None
        core_median = None
        metro_median_by_weight: dict = {}
        core_median_by_weight: dict = {}
        column_stats = {"metro": {}, "core": {}}
        aoi_center = None
        summary_path = city_dir / "results" / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
                metro_median = summary.get("metro_median_access")
                core_median = summary.get("core_median_access")
                metro_median_by_weight = summary.get("metro_median_access_by_weight", {}) or {}
                core_median_by_weight = summary.get("core_median_access_by_weight", {}) or {}
                column_stats = summary.get("column_stats", column_stats)
                # [lon, lat] -- see `code.pipeline.write_city_summary`'s own
                # docstring. Used to place this city's marker on the
                # combined page's low-zoom "all cities" overview map (see
                # `_PAGE_TEMPLATE`'s overview-map JS below); `None` for a
                # city built before this field existed, which the overview
                # map simply skips (no marker) rather than plotting `[0,0]`.
                aoi_center = summary.get("aoi_center")
            except (json.JSONDecodeError, OSError):
                pass
        # Real total metro population (2026-09-02, explicit user request:
        # overview-map circles "depending of total population"), NOT
        # `column_stats.metro.population` above (a per-CELL
        # population-weighted MEAN -- e.g. ~670 for Guadalajara -- entirely
        # unusable for sizing a whole-city marker). `results/stats_data.json`
        # (already required on disk for the stats panel's Distribution/
        # Regression/ANOVA/Place-rank tabs -- see `write_city_summary`'s
        # sibling `_stats_json_data`) carries the real per-cell `population`
        # array this just sums; cheap (one small JSON parse, no geopandas/
        # parquet read) and reuses a file this function's caller already
        # needs to have deployed anyway for the stats panel to work at all.
        metro_total_population = None
        stats_data_path = city_dir / "results" / "stats_data.json"
        if stats_data_path.exists():
            try:
                stats_data = json.loads(stats_data_path.read_text())
                pop_arr = (stats_data.get("metro") or {}).get("population")
                if pop_arr:
                    metro_total_population = float(sum(v for v in pop_arr if v is not None))
            except (json.JSONDecodeError, OSError, TypeError):
                pass
        # `population` is always the default weight; older `summary.json`
        # files (written before `WEIGHT_COLUMN_CANDIDATES` existed) won't
        # have a `*_median_access_by_weight` entry for it yet, so backfill
        # from the plain `metro_median_access`/`core_median_access` fields
        # rather than losing the "Population" option for those cities.
        if "population" not in metro_median_by_weight and metro_median is not None:
            metro_median_by_weight = {**metro_median_by_weight, "population": metro_median}
        if "population" not in core_median_by_weight and core_median is not None:
            core_median_by_weight = {**core_median_by_weight, "population": core_median}
        # Download panel manifest (see `pipeline.py`'s `_finish_pipeline_stages`,
        # which writes this alongside the real `results/{metro,core}/downloads/
        # *.parquet` files it describes). Absent for a city built before this
        # feature existed -- that city is simply left out of the download
        # panel's place dropdown further down, not a hard error.
        downloads_manifest = None
        downloads_manifest_path = city_dir / "results" / "downloads_manifest.json"
        if downloads_manifest_path.exists():
            try:
                downloads_manifest = json.loads(downloads_manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        out.append(
            {
                "key": key,
                "display_name": display_name,
                # Relative path FROM combined_map.html (which lives at ROOT)
                # to that city's map.html, for the iframe `src`.
                "map_src": f"{rel_dir}/map.html",
                # Same relative-to-ROOT convention, one level up from
                # `map_src` -- the base the download panel's <a download>
                # links are built from (`{base_path}/{scope}/downloads/...`).
                "downloads_base_path": f"{rel_dir}/results",
                "downloads_manifest": downloads_manifest,
                "center": aoi_center,
                "metro_total_population": metro_total_population,
                "metro_median_access": metro_median,
                "core_median_access": core_median,
                "metro_median_access_by_weight": metro_median_by_weight,
                "core_median_access_by_weight": core_median_by_weight,
                "column_stats": column_stats,
            }
        )
    return out


_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>transitLOS -- combined city map</title>
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@5.13.0/dist/maplibre-gl.css">
<script src="https://unpkg.com/maplibre-gl@5.13.0/dist/maplibre-gl.js"></script>
<style>
  html, body {{ margin:0; padding:0; height:100%; font:13px/1.4 sans-serif; }}
  #cityIframe {{ position:absolute; top:0; left:0; right:0; bottom:0; width:100%; height:100%; border:0; }}
  /* All-cities low-zoom overview (2026-09-01, explicit user request: "on
     very low zoom levels 8 or lower show all cities on the map and if
     user zooms to another city change automatically the city"). Same
     fullscreen footprint as #cityIframe -- exactly one of the two is
     visible at a time, toggled by JS below (`switchToCity`/
     `switchToOverview`), never both. */
  #overviewMap {{ position:absolute; top:0; left:0; right:0; bottom:0; width:100%; height:100%; display:none; }}
  /* Small text label under each circle marker (2026-09-02: markers became
     population-sized/score-colored circles -- see `ensureOverviewMap`'s
     own comment -- with this label now BELOW the circle instead of being
     the whole marker itself, so a big circle doesn't force wide pill text
     on top of it). */
  .overviewMarker {{
    background:rgba(255,255,255,0.92); color:#222; font:600 11px sans-serif; white-space:nowrap;
    padding:2px 6px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.3);
    margin-top:3px;
  }}
  .overviewMarker:hover {{ background:#0d5bba; }}

  /* Fallback-only top-center pill, living in THIS (wrapper) document --
     hidden once `injectCitySelector()` (page template JS) successfully
     merges a real `<select>` into the iframed city's own `#top-center-bar`
     (immediately left of its scenario select), which is the normal path.
     This pill + `#cityDropdown` below only render if that injection fails
     (cross-origin iframe, or a map build genuinely missing
     `#top-center-bar`) -- kept reachable rather than erroring, not the
     primary UI. */
  #citySelectorBar {{
    display:block; position:absolute; top:10px; left:50%; transform:translateX(-50%); z-index:1301;
    background:white; border-radius:6px; box-shadow:0 1px 6px rgba(0,0,0,0.3);
    padding:8px 14px; font:13px sans-serif; cursor:pointer; user-select:none;
  }}
  #citySelectorBar:hover .v {{ text-decoration:underline; }}
  #citySelectorBar .k {{ color:#888; margin-right:4px; }}
  #citySelectorBar .caret {{ font-size:10px; color:#888; margin-left:4px; }}
  /* Dropdown LIST for the fallback pill above -- only ever shown alongside
     it (the primary `<select>` path opens/closes natively, no overlay
     needed). */
  #cityDropdown {{
    position:absolute; top:42px; left:50%; transform:translateX(-50%); z-index:1301;
    background:white; border-radius:6px; box-shadow:0 2px 10px rgba(0,0,0,0.35);
    min-width:180px; max-height:60vh; overflow:auto; display:none;
  }}
  #cityDropdown.open {{ display:block; }}
  #cityDropdown .opt {{
    padding:8px 12px; cursor:pointer; border-bottom:1px solid #eee;
  }}
  #cityDropdown .opt:last-child {{ border-bottom:none; }}
  #cityDropdown .opt:hover {{ background:#f0f4ff; }}
  #cityDropdown .opt.active {{ background:#e8edff; font-weight:600; }}
</style>
</head>
<body>
<iframe id="cityIframe" src="{initial_src}"></iframe>
<div id="overviewMap"></div>

<div id="citySelectorBar"><span class="k">City</span><span class="v" id="citySelectorLabel"></span><span class="caret">&#9662;</span></div>
<div id="cityDropdown">{dropdown_options}</div>

<!-- Shared "Place rank" popup, from `transitlos.map.build`'s own bottom-left
     📊 stats panel -- the SAME implementation a per-city map.html's own
     "Place rank" tab uses (see `build_place_rank_panel_block`), reduced to
     just that tab. Injected verbatim after `.format()` (below), not through
     it, since its own HTML/JS contains real unescaped `{{`/`}}`. -->
<!-- Unified stats panel: the SAME `_stats_panel_html`/`_stats_panel_js`
     building blocks a real per-city `map.html` uses for its own bottom-left
     📊 panel, so "Place rank" (cross-city) is the first tab and
     Distribution/Regression/ANOVA/Discretization follow it -- but computed
     over the CURRENTLY SELECTED place's own per-h3-cell data, not a
     cross-city average. `window.__statsData` starts empty and is populated
     (and re-populated on every city switch) by `__loadStatsDataForCity`
     below, which fetches that place's own `results/stats_data.json` -- the
     exact same payload a per-city map.html bakes inline for itself. -->
<script>window.__statsData = {{}};</script>
<!--STATS_PANEL-->

<!-- "Download data" panel -- same shared implementation a per-city map.html
     uses (`build_download_panel_block`), but with the place dropdown enabled
     (`place_options` below) since the combined page has no data of its own:
     each option points at that city's own `results/{{metro,core}}/downloads/
     ...` files via a per-city base path (`window.__downloadBasePaths`). -->
<!--DOWNLOAD_PANEL-->

<script>
var CITIES = {cities_json};
var currentKey = {initial_key_json};

function selectCity(key) {{
  var c = null;
  for (var i = 0; i < CITIES.length; i++) if (CITIES[i].key === key) c = CITIES[i];
  if (!c) return;
  // Only reassign `iframe.src` (a real navigation/reload) when the city
  // actually changed -- `switchToCity` below can call this for the SAME
  // city already loaded (e.g. zooming back into the city you just zoomed
  // out of, on the overview map), and reloading an already-correct iframe
  // would lose that map's own zoom/pan/scenario state for no reason.
  if (key !== currentKey || document.getElementById('cityIframe').getAttribute('src') !== c.map_src) {{
    document.getElementById('cityIframe').src = c.map_src;
  }}
  currentKey = key;
  document.getElementById('cityDropdown').classList.remove('open');
  renderDropdown();
  updateCitySelectorLabel();
  __loadStatsDataForCity(key);
}}

// -- Low-zoom "all cities" overview <-> single-city auto-switch --
// (2026-09-01, explicit user request: "combined map to automatically
// change cities and on very low zoom levels 8 or lower show all cities on
// the map and if user zooms to another city change automatically the
// city.") Two mutually-exclusive display modes: MODE_CITY (the existing
// `#cityIframe`, unchanged) and MODE_OVERVIEW (`#overviewMap`, a small
// standalone MapLibre map showing one marker per city from `CITIES[i].center`
// -- see `code.pipeline.write_city_summary`'s `aoi_center`). Zoom 8 is the
// literal threshold from the request in both directions, so neither mode
// can "trap" the other just above/below it.
var MODE_CITY = 'city', MODE_OVERVIEW = 'overview';
var __mode = MODE_CITY;
var __overviewMap = null;
// 2026-09-02 (explicit user request: "activate the world map with a lower
// zoom level") -- lowered from 8 so a city's own map takes over sooner
// (both directions: zooming in on the overview, and the point at which
// zooming OUT of a city's own map drops back to the overview).
var OVERVIEW_ZOOM_THRESHOLD = 5;

// Same red-yellow-green ramp convention every access-score display in this
// app already uses (0 = red, 1 = green). Top-level (not nested inside
// `ensureOverviewMap`) so `window.__onPlaceRankComputed` below can also
// reach it, independent of whether the overview map itself has been built
// yet.
function __scoreColor(v) {{
  if (v == null || isNaN(v)) return '#888';
  v = Math.max(0, Math.min(1, v));
  var stops = v < 0.5
    ? [[214, 47, 39], [255, 208, 66], v * 2]
    : [[255, 208, 66], [26, 152, 80], (v - 0.5) * 2];
  var a = stops[0], b = stops[1], t = stops[2];
  var rgb = [0, 1, 2].map(function(k) {{ return Math.round(a[k] + (b[k] - a[k]) * t); }});
  return 'rgb(' + rgb.join(',') + ')';
}}

// 2026-09-02 (live user report: "the numbers and colors of the global map
// do not coincide with place rank population weight so something is
// wrong") -- real bug: the overview map's circles used `c.metro_median_access`,
// baked into `results/summary.json` at that city's last FULL pipeline run
// (`code.pipeline.write_city_summary`), which a `--map-only` rebuild (the
// normal way this whole map gets refreshed -- see `hosting/README.md`)
// never touches -- only `results/stats_data.json` gets rewritten on that
// path. Meanwhile the Place-rank tab computes its own population-weighted
// median LIVE from that same freshly-rebuilt `stats_data.json`, so the two
// numbers drift apart the moment they were built at different times (which
// is always, for every city rebuilt today). `__overviewMarkerRefs` holds
// each city's marker DOM elements (populated once `ensureOverviewMap`
// actually builds them); `__latestScoreByKey` caches whichever value
// arrived MOST RECENTLY per city -- from `window.__onPlaceRankComputed`
// (the exact same live computation `renderPlaceRank` in `_stats_panel_js`
// already does, handed over via that hook) if it has run yet, else still
// the stale build-time fallback until it does. Markers are only ever
// created AFTER `switchToOverview()` -> `ensureOverviewMap()` has run
// (page load, before `renderPlaceRank`'s background fetches resolve), so
// this cache is what lets a LATER-arriving live value still reach
// already-built markers, not just markers built after the fact.
var __overviewMarkerRefs = {{}}; // key -> {{circleEl, labelEl}}
var __latestScoreByKey = {{}}; // key -> population-weighted median access (live, once known)
function __applyOverviewMarkerScore(key, score) {{
  var ref = __overviewMarkerRefs[key];
  if (!ref) return;
  ref.circleEl.style.background = __scoreColor(score);
  var c = null;
  for (var i = 0; i < CITIES.length; i++) if (CITIES[i].key === key) c = CITIES[i];
  ref.labelEl.textContent = (c ? c.display_name : key) + (score != null ? ' — ' + score.toFixed(2) : '');
}}
window.__onPlaceRankComputed = function(results) {{
  results.forEach(function(r) {{
    if (r.value == null) return; // still loading / genuinely unranked -- keep showing the last known value
    __latestScoreByKey[r.key] = r.value;
    __applyOverviewMarkerScore(r.key, r.value);
  }});
}};

function __nearestCity(lon, lat) {{
  var best = null, bestD = Infinity;
  for (var i = 0; i < CITIES.length; i++) {{
    var c = CITIES[i];
    if (!c.center) continue;
    var dx = c.center[0] - lon, dy = c.center[1] - lat;
    var d = dx * dx + dy * dy;
    if (d < bestD) {{ bestD = d; best = c; }}
  }}
  return best;
}}

function ensureOverviewMap() {{
  if (__overviewMap) return __overviewMap;
  var pts = [];
  for (var i = 0; i < CITIES.length; i++) if (CITIES[i].center) pts.push(CITIES[i]);
  __overviewMap = new maplibregl.Map({{
    container: 'overviewMap',
    // 2026-09-02 follow-up (explicit user request: "for the world map use
    // the same simple background map as for the individual places") -- a
    // per-city `map.html`'s own DEFAULT basemap is NOT CartoDB either (see
    // `geohierarchy.maps.maplibre.render.BASEMAP_STYLES`/`DEFAULT_BASEMAP`
    // -- CartoDB Positron/Dark Matter were already replaced there with the
    // OpenFreeMap equivalents for the same "no CARTO" reason this page's
    // own CartoDB raster tiles were dropped for originally). Using the
    // exact same style URL here (a full MapLibre vector style, not a plain
    // raster XYZ template like the earlier OSM-raster attempt) keeps the
    // world map and every individual city map visually consistent.
    style: 'https://tiles.openfreemap.org/styles/positron',
    center: pts.length ? pts[0].center : [0, 20],
    zoom: 2,
    maxZoom: OVERVIEW_ZOOM_THRESHOLD + 2,
  }});
  __overviewMap.addControl(new maplibregl.NavigationControl(), 'top-right');
  // 2026-09-02 (explicit user request: "some circle depending of total
  // population and circle with color depending on population weighted
  // level of service") -- radius scaled by sqrt(population), the standard
  // cartographic convention for a proportional-symbol map (area, not
  // radius, proportional to the quantity -- a plain linear radius scale
  // visually exaggerates a bigger city far more than its real population
  // ratio warrants). Domain is THIS PAGE'S OWN city roster's min/max, not
  // a fixed constant, so the ramp always uses the full visual range
  // regardless of how many/which cities are currently deployed.
  var __pops = pts.map(function(c) {{ return c.metro_total_population; }}).filter(function(v) {{ return v != null && v > 0; }});
  var __popMin = __pops.length ? Math.min.apply(null, __pops) : 0;
  var __popMax = __pops.length ? Math.max.apply(null, __pops) : 1;
  var MIN_R = 8, MAX_R = 32;
  function __popRadius(pop) {{
    if (pop == null || pop <= 0 || __popMax <= __popMin) return MIN_R;
    var t = (Math.sqrt(pop) - Math.sqrt(__popMin)) / (Math.sqrt(__popMax) - Math.sqrt(__popMin));
    return MIN_R + t * (MAX_R - MIN_R);
  }}
  __overviewMap.on('load', function() {{
    if (pts.length > 1) {{
      var bounds = new maplibregl.LngLatBounds();
      pts.forEach(function(c) {{ bounds.extend(c.center); }});
      __overviewMap.fitBounds(bounds, {{padding: 60, maxZoom: OVERVIEW_ZOOM_THRESHOLD - 1, duration: 0}});
    }}
    pts.forEach(function(c) {{
      // Prefer the LIVE population-weighted value (`__onPlaceRankComputed`,
      // fed by the exact same computation the Place-rank tab shows) once
      // known; `c.metro_median_access` (stale build-time `summary.json`
      // value -- see the big comment above `__overviewMarkerRefs`) is only
      // the initial placeholder until that arrives.
      var score = (c.key in __latestScoreByKey) ? __latestScoreByKey[c.key] : c.metro_median_access;
      var r = __popRadius(c.metro_total_population);
      // 2026-09-02 (live user report: "global map when zoomed out appear
      // sometimes very off in position") -- `wrap` used to stack the circle
      // AND the text label in a flex column, with `anchor: 'center'`
      // centering the WHOLE stack (circle + label + the gap between them)
      // on `c.center`. That puts the CIRCLE itself several pixels above
      // the real coordinate (roughly half the label's own height) -- a
      // fixed on-screen pixel offset that doesn't scale with zoom, so it
      // reads as "barely noticeable" zoomed into one city but "very off"
      // zoomed out to the whole world, where that same pixel gap is a
      // large fraction of the distance between nearby city markers. Fix:
      // `wrap` is now sized/anchored to the CIRCLE only (`anchor: 'center'`
      // now genuinely centers the circle on `c.center`); the label is
      // pulled OUT of layout flow (`position:absolute`, positioned below
      // the circle) so it can never shift the circle's own center.
      //
      // 2026-09-02 follow-up (live user report, same day: "cities appear in
      // very weird positions that change with the map zoom and do not
      // overlap with the background map") -- REAL bug in the fix above:
      // `wrap.style.cssText = 'position:relative;...'` sets `position` via
      // inline style, which wins over (overwrites, not merges with) the
      // `position:absolute` MapLibre's own `.maplibregl-marker` CSS class
      // gives every marker element -- `Marker` positions markers purely via
      // a CSS `transform: translate3d(...)` on this element, which only
      // lands correctly relative to the map when the element is genuinely
      // `position:absolute` (its containing block is then the map's own
      // positioned container, a fixed reference frame); `position:relative`
      // instead keeps it in NORMAL DOCUMENT FLOW, so its actual on-screen
      // position becomes whatever that flow happens to place it at (which
      // silently shifts as other DOM content changes, e.g. across a
      // re-render loop rebuilding every marker) with the transform then
      // applied on TOP of that wrong base position -- exactly "position
      // changes with zoom, doesn't match the basemap". Fix: don't set
      // `position` on `wrap` AT ALL -- leave MapLibre's own class in
      // control of it (still `position:absolute`, still a valid CSS
      // positioning ancestor for the label's own `position:absolute`
      // below), only set the size this element actually needs.
      var wrap = document.createElement('div');
      wrap.style.cssText = 'width:' + (r * 2) + 'px;height:' + (r * 2) + 'px;cursor:pointer;';
      var circle = document.createElement('div');
      circle.className = 'overviewCircle';
      circle.style.cssText = 'width:100%;height:100%;border-radius:50%;' +
        'background:' + __scoreColor(score) + ';border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,0.4);';
      var label = document.createElement('div');
      label.className = 'overviewMarker';
      label.style.cssText = 'position:absolute;top:100%;left:50%;transform:translateX(-50%);margin-top:3px;';
      label.textContent = c.display_name + (score != null ? ' — ' + score.toFixed(2) : '');
      wrap.appendChild(circle);
      wrap.appendChild(label);
      __overviewMarkerRefs[c.key] = {{circleEl: circle, labelEl: label}};
      // 2026-09-02 (explicit user request: "allow user to click on the
      // worldwide map on a city and this means zoom into the city") --
      // fly the overview map to that city rather than jumping to its
      // `map.html` instantly; the existing `zoomend` handler below (the
      // same one scroll-zooming already relies on) picks up the flight's
      // end and does the actual `switchToCity` once past the threshold,
      // so a click and a manual zoom-in now go through the exact same
      // path -- one behavior, two triggers.
      wrap.addEventListener('click', function(ev) {{
        ev.stopPropagation();
        __overviewMap.flyTo({{center: c.center, zoom: OVERVIEW_ZOOM_THRESHOLD + 1, duration: 1200}});
      }});
      new maplibregl.Marker({{element: wrap, anchor: 'center'}}).setLngLat(c.center).addTo(__overviewMap);
    }});
  }});
  // Zooming IN past the threshold on the overview map auto-picks whichever
  // city's marker is closest to the current view center and switches to it
  // -- "if user zooms to another city change automatically the city".
  __overviewMap.on('zoomend', function() {{
    if (__mode !== MODE_OVERVIEW) return;
    if (__overviewMap.getZoom() > OVERVIEW_ZOOM_THRESHOLD) {{
      var center = __overviewMap.getCenter();
      var nearest = __nearestCity(center.lng, center.lat);
      if (nearest) switchToCity(nearest.key);
    }}
  }});
  return __overviewMap;
}}

// 2026-09-02 (explicit user request: "still have the stats panel with the
// place rank (only place rank in the worldwide map and leave as is in each
// individual city)") -- Place rank is the one tab that's cross-city and
// makes sense with no single place selected; Distribution/Regression/ANOVA/
// Discretization/Metadata all read `window.__statsData`, a SINGLE place's
// per-cell data, which has no "worldwide" meaning at all. Rather than
// rebuilding the panel's HTML per mode, this just hides/shows the other
// tab buttons and forces "Place rank" active whenever entering overview
// mode -- switching back to a city (`switchToCity`) reveals them again
// exactly as `_stats_panel_html` originally built them ("leave as is").
function __updateStatsTabVisibility() {{
  var overview = __mode === MODE_OVERVIEW;
  document.querySelectorAll('.statsTabBtn').forEach(function(btn) {{
    if (btn.dataset.tab === 'placerank') return;
    btn.style.display = overview ? 'none' : '';
  }});
  if (overview) {{
    var placerankBtn = document.querySelector('.statsTabBtn[data-tab="placerank"]');
    if (placerankBtn && !placerankBtn.classList.contains('active')) placerankBtn.click();
  }}
}}

function switchToCity(key) {{
  __mode = MODE_CITY;
  document.getElementById('overviewMap').style.display = 'none';
  document.getElementById('cityIframe').style.display = 'block';
  selectCity(key);
  __updateStatsTabVisibility();
}}

function switchToOverview(fromLngLat) {{
  __mode = MODE_OVERVIEW;
  document.getElementById('cityIframe').style.display = 'none';
  var ov = ensureOverviewMap();
  document.getElementById('overviewMap').style.display = 'block';
  ov.resize();
  if (fromLngLat) ov.jumpTo({{center: fromLngLat, zoom: OVERVIEW_ZOOM_THRESHOLD - 1}});
  __updateStatsTabVisibility();
}}

// Zooming OUT past the threshold INSIDE the currently-shown city's own map
// auto-drops back to the overview -- the other half of the same request.
// `iframe.contentWindow.__mainMap` (see `geohierarchy.maps.maplibre.render`,
// which exposes it on `window` for exactly this) only exists once that
// city's MapLibre instance has finished constructing, so this polls (same
// pattern `injectCitySelector`'s own `setInterval` already uses for the
// same same-origin-iframe-not-ready-yet problem) rather than relying on a
// single 'load' event.
function __pollCityZoom() {{
  if (__mode !== MODE_CITY) return;
  try {{
    var win = document.getElementById('cityIframe').contentWindow;
    var m = win && win.__mainMap;
    if (!m) return;
    // One listener per (iframe document, not per poll tick) -- `_wiredFor`
    // is stamped with `currentKey` so a city switch (a real iframe
    // navigation, a new `window`/map instance) re-wires automatically
    // without ever double-attaching to the same still-live map.
    if (m.__combinedWiredFor !== currentKey) {{
      m.__combinedWiredFor = currentKey;
      m.on('zoomend', function() {{
        if (__mode !== MODE_CITY) return;
        if (m.getZoom() <= OVERVIEW_ZOOM_THRESHOLD) {{
          var c = m.getCenter();
          switchToOverview([c.lng, c.lat]);
        }}
      }});
    }}
  }} catch (e) {{}}
}}
setInterval(__pollCityZoom, 800);

function renderDropdown() {{
  var html = '';
  for (var i = 0; i < CITIES.length; i++) {{
    var c = CITIES[i];
    var cls = 'opt' + (c.key === currentKey ? ' active' : '');
    html += '<div class="' + cls + '" data-key="' + c.key + '">' + c.display_name + '</div>';
  }}
  document.getElementById('cityDropdown').innerHTML = html;
  var opts = document.querySelectorAll('#cityDropdown .opt');
  for (var j = 0; j < opts.length; j++) {{
    opts[j].addEventListener('click', function(ev) {{ selectCity(ev.currentTarget.getAttribute('data-key')); }});
  }}
}}

document.addEventListener('click', function() {{
  document.getElementById('cityDropdown').classList.remove('open');
}});

function currentCityLabel() {{
  for (var i = 0; i < CITIES.length; i++) if (CITIES[i].key === currentKey) return CITIES[i].display_name;
  return currentKey;
}}

// -- City selector: merged into the iframed city's own #top-center-bar --
//
// A real `<select id="clCitySelectorInline">`, one `<option>` per CITIES
// entry, is prepended as the FIRST child of the iframed page's
// `#top-center-bar` (a real DOM node living in the IFRAME's own document --
// same-origin, so this is plain cross-frame DOM access, no postMessage
// needed) -- immediately LEFT of `#topBarScenarioSelect` (the scenario
// dropdown), styled identically to it (transparent, borderless, same font)
// so it reads as one more segment of that SAME bar -- "[City v] | [Scenario
// v] | Access: 0.62 ..." -- not a second box next to it. A thin divider
// matching the bar's existing one (between the scenario select and the
// access-score readout) is inserted right after it. Picking an option fires
// `selectCity()` directly -- a real interactive control, not a static label.
//
// Falls back to the standalone `#citySelectorBar` top-center pill/
// `#cityDropdown` overlay (this wrapper document's own, hidden once
// injection succeeds) only if injection ever fails -- e.g. a city's
// `map.html` genuinely has no `#top-center-bar`, or the iframe is
// cross-origin (e.g. this page opened over `file://`, where
// `iframe.contentDocument` throws).
var CITY_SELECTOR_INLINE_ID = 'clCitySelectorInline';

function injectCitySelector() {{
  try {{
    var idoc = document.getElementById('cityIframe').contentDocument;
    if (!idoc) throw new Error('no iframe document');
    var bar = idoc.getElementById('top-center-bar');
    if (!bar) throw new Error('no top-center-bar in iframe');
    var sel = idoc.getElementById(CITY_SELECTOR_INLINE_ID);
    if (!sel) {{
      sel = idoc.createElement('select');
      sel.id = CITY_SELECTOR_INLINE_ID;
      sel.style.cssText = 'border:none;background:transparent;font:600 12px sans-serif;' +
        'color:#1a1a1a;padding:2px 4px;cursor:pointer;';
      var divider = idoc.createElement('span');
      divider.id = CITY_SELECTOR_INLINE_ID + 'Divider';
      divider.style.cssText = 'width:1px;background:#ddd;align-self:stretch;margin:0 10px;';
      bar.insertBefore(divider, bar.firstChild);
      bar.insertBefore(sel, divider);
      sel.addEventListener('change', function(ev) {{ selectCity(ev.target.value); }});
    }}
    var optsHtml = '';
    for (var i = 0; i < CITIES.length; i++) {{
      var c = CITIES[i];
      optsHtml += '<option value="' + c.key + '"' + (c.key === currentKey ? ' selected' : '') + '>' +
        c.display_name + '</option>';
    }}
    sel.innerHTML = optsHtml;
    document.getElementById('citySelectorBar').style.display = 'none';
    document.getElementById('cityDropdown').classList.remove('open');
    return true;
  }} catch (e) {{
    // Fallback: standalone top-center pill, own click handler wired once.
    var fallback = document.getElementById('citySelectorBar');
    fallback.style.display = 'block';
    if (!fallback.__wired) {{
      fallback.__wired = true;
      fallback.addEventListener('click', function(ev) {{
        ev.stopPropagation();
        document.getElementById('cityDropdown').classList.toggle('open');
      }});
    }}
    var dd = document.getElementById('cityDropdown');
    dd.style.left = '50%'; dd.style.top = '42px'; dd.style.transform = 'translateX(-50%)';
    return false;
  }}
}}

function updateCitySelectorLabel() {{
  document.getElementById('citySelectorLabel').textContent = currentCityLabel();
  try {{
    var idoc = document.getElementById('cityIframe').contentDocument;
    var sel = idoc && idoc.getElementById(CITY_SELECTOR_INLINE_ID);
    if (sel) sel.value = currentKey;
  }} catch (e) {{}}
}}

document.getElementById('cityIframe').addEventListener('load', function() {{
  injectCitySelector();
  __loadStatsDataForCity(currentKey);
}});
// `#top-center-bar` can be replaced wholesale without a navigation (e.g. its
// own script re-rendering the bar) -- a light poll keeps the inline select
// present without needing the iframe's own scripts to cooperate.
setInterval(injectCitySelector, 800);
injectCitySelector();
updateCitySelectorLabel();

// -- Per-place stats data, feeding the unified stats panel's Distribution/
// Regression/ANOVA/Discretization tabs (`_stats_panel_js`, embedded via the
// STATS_PANEL placeholder above). Its "Place rank" tab needs no per-place data
// (it fetches every city's own stats_data.json itself, cross-city); the
// other tabs read `window.__statsData` directly, which this keeps pointed
// at whichever place is currently selected.
var __statsDataCache = {{}}; // key -> fetched payload (or false on failure)

// Bug fix (2026-08-29, live user report): the ANOVA/Regression tabs used to
// offer only the two columns baked into `ANOVA_REGRESSION_COLUMNS` (a
// hardcoded Python tuple at the top of this file) regardless of which real
// relative (share/rate/density) columns the CURRENTLY SELECTED city's own
// `stats_data.json` actually carries -- e.g. Guadalajara's real
// `labor_force_rate`/`unemployment_rate`/`minority_ethnicity_share`/
// `inegi_*_share` columns were never offered, even though that SAME city's
// own per-city `map.html` ANOVA tab (built by `transitlos.map.build`, which
// computes `relative_fields(all_fields)` over the real GeoDataFrame at
// build time for that one city) shows all of them. A real per-city
// `map.html` can do that in Python because it only ever serves one city;
// this combined page switches cities via a live client-side `fetch()`
// (`window.__statsData` reassigned on every city switch, right below), so
// there is no single city's GeoDataFrame available at PYTHON build time to
// compute `relative_fields`/`absolute_fields` over here -- the fix has to run
// in JS, client-side, against whatever keys `window.__statsData` actually has
// after each fetch. Mirrors `is_relative_field`/`relative_fields`/
// `absolute_fields` in `transitlos/map/build.py` (`_RELATIVE_MARKERS`) --
// kept as a JS-side parallel constant for the same reason
// `WEIGHT_COLUMN_CANDIDATES`/`ANOVA_REGRESSION_COLUMNS` above are parallel
// Python constants rather than an import: no import path from this page's
// embedded JS back into that Python module.
var __RELATIVE_MARKERS = [
  'share', 'rate', 'ratio', 'density', 'pct', 'percent',
  'median', 'mean', 'avg', 'average', 'index', 'score',
  'per_capita', 'per_km', 'per_ha', 'per_1000', '_per_'
];
function __isRelativeFieldKey(key) {{
  var name = String(key).toLowerCase();
  for (var i = 0; i < __RELATIVE_MARKERS.length; i++) {{
    if (name.indexOf(__RELATIVE_MARKERS[i]) !== -1) return true;
  }}
  return false;
}}
// Keys present in every `stats_data.json` payload that are never real
// regression/ANOVA/distribution variables -- `level_of_service` is the FIXED
// y-axis every tab already correlates against (excluded the same way
// `_stats_field_candidates`'s `extra_exclude=["level_of_service"]` excludes it
// server-side), `h3_cell` is a join key for the scenario editor's recomputed
// overlay (see `getStatsData` in `_stats_panel_js`), not a field.
var __STATS_NON_FIELD_KEYS = {{level_of_service: 1, h3_cell: 1}};
function __rebuildOptionsFromKeys(selectEl, keys, preserveSelection) {{
  if (!selectEl) return;
  var prevValue = selectEl.value;
  var html = '';
  keys.forEach(function(k) {{ html += '<option value="' + k + '">' + __fieldLabel(k) + '</option>'; }});
  selectEl.innerHTML = html;
  if (preserveSelection && keys.indexOf(prevValue) !== -1) selectEl.value = prevValue;
  else if (keys.length) selectEl.value = keys[0];
}}
// Rebuilds every field-vocabulary-driven control (`#regFieldSelect` +
// `__anovaFields`'s Regression/ANOVA options, `#distMainSelect`/
// `#distOverlaySelect`'s Distribution options) from whichever keys `data`
// actually carries for the city just loaded. Run from `__loadStatsDataForCity`
// right after `window.__statsData` is reassigned, before the active tab
// re-renders.
function __rebuildStatsFieldSelectors(data) {{
  // `data` is the FULL fetched payload (`{{metro: {{col: [...]}}, core: {{col: [...]}}}}`),
  // not the column-array map itself -- the real per-column arrays this
  // function needs to inspect live one level down, under whichever area is
  // currently active (falling back to 'metro', the panel's own default,
  // if `window.__statsArea` isn't set yet). A prior version of this
  // function read `Object.keys(data)` directly, which only ever produced
  // `['metro', 'core']` -- neither of those values is itself an array, so
  // the `Array.isArray` filter below silently zeroed out every field list
  // (verified live: `window.__anovaFields` came back `[]`, and the ANOVA
  // tab quietly fell back to whatever it was before, e.g. one hardcoded
  // column -- no error, just wrong/empty data).
  var areaData = (data && data[window.__statsArea]) || (data && data.metro) || {{}};
  var keys = Object.keys(areaData).filter(function(k) {{
    return !__STATS_NON_FIELD_KEYS[k] && Array.isArray(areaData[k]);
  }});
  var relFields = keys.filter(__isRelativeFieldKey);
  var absFields = keys.filter(function(k) {{ return !__isRelativeFieldKey(k); }});
  if (!relFields.length) relFields = keys.indexOf('pop_density') !== -1 ? ['pop_density'] : [];
  // `__anovaFields` (the array `renderAnova`/`renderRegression`'s R^2 bars
  // actually iterate) lives inside `_stats_panel_js`'s own closure -- a
  // plain `window.__anovaFields = relFields` assignment here would just
  // create an unrelated same-named global property, never read by those
  // functions (a real bug found live: the ANOVA tab kept showing only the
  // original build-time column despite this looking right from outside).
  // `window.__setRegressionFields` is the sanctioned hook `_stats_panel_js`
  // exposes specifically so an external caller CAN reach that closure-local
  // variable -- it also keeps `#regFieldSelect`'s own `<option>`s in sync.
  if (window.__setRegressionFields) window.__setRegressionFields(relFields);
  __rebuildOptionsFromKeys(document.getElementById('distMainSelect'), absFields, true);
  var overlaySel = document.getElementById('distOverlaySelect');
  if (overlaySel) {{
    var prevOverlay = overlaySel.value;
    var overlayHtml = '<option value="none" selected>None</option>';
    absFields.forEach(function(k) {{ overlayHtml += '<option value="' + k + '">' + __fieldLabel(k) + '</option>'; }});
    overlaySel.innerHTML = overlayHtml;
    if (absFields.indexOf(prevOverlay) !== -1) overlaySel.value = prevOverlay;
  }}
}}

// -- Metadata tab (2026-09-02, explicit user request: the combined page's
// Metadata tab was entirely absent -- see `_stats_panel_html`'s
// `metadata_rows` docstring). A real per-city `map.html` computes this
// table once in Python from its own GeoDataFrame (`_column_metadata_rows`
// in transitlos/map/build.py); this page has no such GeoDataFrame at
// build time (it only ever has whichever place's `stats_data.json` was
// last fetched), so it recomputes the same total/average/share logic here
// in JS, against `window.__statsData`, the same per-cell payload already
// driving Distribution/Regression/ANOVA -- same pattern
// `__rebuildStatsFieldSelectors` already established for those tabs'
// field lists. Source/level/description columns (real per-city maps show
// pyCensus's own `global_schema.json` definitions there) are left blank
// here rather than reimplemented -- that lookup lives in `pycensus`
// (Python-only, no JS port), and Column/Total/Average/Share/Unit already
// carry the actual data value.
//
// `__ABS_TO_REL`: the exact same absolute-column -> relative-field-key
// pairing table `_shape_popup_js`'s click popup uses for a per-city map
// (`_POPUP_ABS_TO_REL` in build.py) -- imported directly rather than
// re-derived, so "which share pairs with which total" never drifts
// between the popup and this tab.
var __ABS_TO_REL = {abs_to_rel_json};

function __weightedAvg(values, weights) {{
  var wSum = 0, wvSum = 0, plainSum = 0, n = 0;
  for (var i = 0; i < values.length; i++) {{
    var v = values[i];
    if (v == null || isNaN(v)) continue;
    n++;
    plainSum += v;
    var w = (weights && weights[i] != null && !isNaN(weights[i])) ? weights[i] : 0;
    wSum += w;
    wvSum += v * w;
  }}
  if (n === 0) return null;
  return {{sum: plainSum, avg: wSum > 0 ? (wvSum / wSum) : (plainSum / n), n: n}};
}}

function __renderMetadataTab() {{
  var body = document.getElementById('metadataTabBody');
  if (!body) return;
  var area = window.__statsArea === 'core' ? 'core' : 'metro';
  var data = (window.__statsData && window.__statsData[area]) || (window.__statsData && window.__statsData.metro) || {{}};
  var keys = Object.keys(data).filter(function(k) {{
    return !__STATS_NON_FIELD_KEYS[k] && Array.isArray(data[k]);
  }});
  var weights = data.population || null;
  var consumedShare = {{}};
  keys.forEach(function(k) {{
    var relKey = __ABS_TO_REL[k];
    if (relKey && keys.indexOf(relKey) !== -1) consumedShare[relKey] = true;
  }});
  var rows = '';
  keys.forEach(function(k) {{
    if (consumedShare[k]) return; // shown as a share on its numerator's row instead
    var stat = __weightedAvg(data[k], weights);
    if (!stat) return;
    var isRel = __isRelativeFieldKey(k);
    var totalStr = isRel ? '' : stat.sum.toLocaleString(undefined, {{maximumFractionDigits: 0}});
    var avgStr = stat.avg.toLocaleString(undefined, {{maximumFractionDigits: 2}});
    var shareStr = '';
    var relKey = __ABS_TO_REL[k];
    if (relKey && keys.indexOf(relKey) !== -1) {{
      var relStat = __weightedAvg(data[relKey], weights);
      if (relStat) shareStr = (relStat.avg * 100).toFixed(1) + '%';
    }}
    // Mirrors `field_unit_suffix` in transitlos/map/build.py exactly (a
    // "mean"/"median"/"score" relative field -- e.g. `educationMeanGrade`
    // -- isn't a 0..1 share, so it gets no unit at all, not a wrong "%").
    var kLower = k.toLowerCase();
    var unitStr = '';
    if (kLower.indexOf('density') !== -1) unitStr = 'pop/km²';
    else if (['share', 'rate', 'pct', 'percent', 'ratio'].some(function(m) {{ return kLower.indexOf(m) !== -1; }})) unitStr = '%';
    else if (!isRel) unitStr = 'count';
    rows += '<tr style="border-top:1px solid #eee;">' +
      '<td style="font-weight:600;">' + __fieldLabel(k) + '</td>' +
      '<td>--</td><td>--</td>' +
      '<td style="text-align:right;">' + totalStr + '</td>' +
      '<td style="text-align:right;">' + avgStr + '</td>' +
      '<td style="text-align:right;">' + shareStr + '</td>' +
      '<td>' + unitStr + '</td>' +
      '<td>' + (weights ? 'population' : '(unweighted)') + '</td>' +
      '<td></td></tr>';
  }});
  body.innerHTML = rows || '<tr><td colspan="9" style="color:#888;">No data loaded for this place yet.</td></tr>';
}}
window.__renderMetadataTab = __renderMetadataTab;

function __loadStatsDataForCity(key) {{
  if (Object.prototype.hasOwnProperty.call(__statsDataCache, key)) {{
    if (key === currentKey) {{
      window.__statsData = __statsDataCache[key] || {{}};
      __rebuildStatsFieldSelectors(window.__statsData);
      __refreshActiveStatsTab();
    }}
    return;
  }}
  var c = null;
  for (var i = 0; i < CITIES.length; i++) if (CITIES[i].key === key) c = CITIES[i];
  if (!c) return;
  fetch(c.downloads_base_path + '/stats_data.json')
    .then(function(resp) {{ return resp.ok ? resp.json() : {{}}; }})
    .catch(function() {{ return {{}}; }})
    .then(function(data) {{
      __statsDataCache[key] = data;
      if (key === currentKey) {{
        window.__statsData = data || {{}};
        __rebuildStatsFieldSelectors(window.__statsData);
        __refreshActiveStatsTab();
      }}
    }});
}}
function __refreshActiveStatsTab() {{
  var activeBtn = document.querySelector('.statsTabBtn.active');
  if (activeBtn) activeBtn.click();
}}

// Test/automation hooks.
window.__selectCity = selectCity;
window.__cities = CITIES;
window.__loadStatsDataForCity = __loadStatsDataForCity;

renderDropdown();
// 2026-09-02 (explicit user request: "combined map start should be the
// complete world so that user can see all cities") -- open on the
// all-cities overview rather than dropping straight into `cities[0]`'s
// own map. `currentKey`/the iframe's `src` still point at the initial
// city underneath (unchanged) so the very first zoom-in or manual city
// pick is instant, not a fresh navigation.
switchToOverview();
// 2026-09-02 (live user report -- hosted map: "the panel works but no
// data"): a real race, not new today but only just made visible. This
// script's own top-level code runs as soon as the browser parses it
// (script tags execute in document order, well before the page's `load`
// event) and immediately kicks off `__loadStatsDataForCity(currentKey)`'s
// `fetch()` for the initial city. `_stats_panel_js` (embedded above, via
// the STATS_PANEL placeholder) defines `window.__setRegressionFields` --
// the hook `__rebuildStatsFieldSelectors` needs to correct `#regFieldSelect`'s
// options down to the columns THIS city's `stats_data.json` actually has --
// but only INSIDE its own `window.addEventListener('load', ...)` callback.
// On a fast same-origin fetch (typical for this page, and apparently
// consistently the case on the real hosted server), that fetch's `.then()`
// can resolve BEFORE the page's `load` event fires, meaning
// `window.__setRegressionFields` doesn't exist yet -- `__rebuildStatsFieldSelectors`'s
// `if (window.__setRegressionFields)` guard then just silently skips the
// correction (by design, to not throw), leaving `#regFieldSelect` stuck on
// its build-time hardcoded `ANOVA_REGRESSION_COLUMNS` list -- fields a
// given city's real data may not actually carry at all (e.g. `pop_density`,
// deliberately dropped from a city's own `stats_data.json` once a real
// `population_density`/`worldpop_population_density` column replaces it --
// see `_numeric_field_candidates`'s docstring in transitlos/map/build.py).
// Selecting/defaulting to one of those stale, nonexistent options makes
// `renderRegression`'s `data[field].map(...)` throw on `undefined` -- the
// exact "panel works, no data" symptom (tabs/dropdowns render, but the
// chart never does because the render call threw). Registering THIS
// city-load kickoff inside its OWN `load` listener guarantees it runs
// strictly after `_stats_panel_js`'s own `load` listener above (same-event
// listeners fire in registration order), so `window.__setRegressionFields`
// is always defined in time, on every load -- not just on a slow-fetch
// browser/connection where the race happened not to bite.
window.addEventListener('load', function() {{
  __loadStatsDataForCity(currentKey);
}});
</script>
</body>
</html>
"""


def build_combined_map(out_path: Path | None = None) -> Path:
    """Write `city_science_network/combined_map.html` from the currently-available cities.

    Returns the written path. Safe to re-run any time -- it only reads
    each city's `map.html`/`results/summary.json` and always overwrites
    the combined page from scratch.
    """
    cities = _available_cities()
    if not cities:
        raise RuntimeError("No city map.html found under CITY_MANIFEST -- nothing to combine.")
    # Ordered best-to-worst by `metro_median_access` (population-weighted
    # median level of service, metro scope -- the same headline figure the
    # Place-rank tab ranks by) so both the dropdown list AND the page's
    # default-selected/initial city (`cities[0]`) are the best-scoring place,
    # per explicit user request. A city with no `metro_median_access` yet
    # (summary.json missing/not yet built) sorts to the very end rather than
    # being treated as a 0 -- it simply hasn't been ranked, not the worst.
    cities.sort(
        key=lambda c: (c["metro_median_access"] is None, -(c["metro_median_access"] or 0))
    )

    dropdown_options = "".join(
        f'<div class="opt{" active" if i == 0 else ""}" data-key="{c["key"]}">{c["display_name"]}</div>'
        for i, c in enumerate(cities)
    )
    # The unified stats panel -- SAME `_stats_panel_html`/`_stats_panel_js`
    # building blocks a per-city map.html uses for its own bottom-left 📊
    # panel: "Place rank" (cross-city, from every place's own
    # `results/summary.json`/`results/stats_data.json`) first, then
    # Distribution/Regression/ANOVA/Discretization computed over whichever
    # place is currently selected (`window.__statsData`, kept in sync by
    # `__loadStatsDataForCity` in the page template's own JS). Wired to this
    # page's REAL city roster (`cities`, from `_available_cities()`) rather
    # than build.py's own hand-maintained fallback manifest, and to a
    # `fetch_prefix=""` since `combined_map.html` lives AT `city_science_network/`
    # itself (one level up from where a per-city `map.html` sits, which is
    # why that one still uses its own default `"../"` prefix).
    # `count_fields`/`regression_fields` reuse the same curated
    # `WEIGHT_COLUMN_CANDIDATES`/`ANOVA_REGRESSION_COLUMNS` vocabularies the
    # old cross-city-only panel used -- absolute counts and .share/.density
    # relative columns respectively -- so the Distribution tab's "Distribute
    # by"/weight-column options and the Regression/ANOVA tabs' variable
    # options stay the same curated list a user of the old panel already
    # knew, just now computed over real per-h3-cell data for one place
    # instead of a cross-city population-weighted average.
    # `_stats_panel_js`'s Place-rank tab (and its "Compare with place"
    # sub-selects) fetch each manifest entry's data from `<key>/results/...`,
    # relative to this page -- true for every entry in `cities`, since
    # `CITY_MANIFEST` (and therefore `cities`) is city_science_network-only (no
    # `boston_city`-style sibling-directory entry, per explicit user
    # request, 2026-08-30), so no path-prefix exclusion is needed here.
    place_rank_manifest = [{"key": c["key"], "label": c["display_name"]} for c in cities]
    count_fields = [key for key, _label in WEIGHT_COLUMN_CANDIDATES]
    regression_fields = [key for key, _label in ANOVA_REGRESSION_COLUMNS]
    stats_panel_html = _stats_panel_html(
        regression_fields, count_fields,
        enable_place_comparison=True, has_multi_area=True, place_rank_only=False,
        # `[]`, not `None`: renders the Metadata tab's button + empty table
        # shell (see `_stats_panel_html`'s own docstring on the `is not None`
        # gate) -- this page has no per-city GeoDataFrame at Python build
        # time to fill real rows from, so `#metadataTabBody` is populated
        # live in JS instead (`__renderMetadataTab` below, wired via the
        # `window.__renderMetadataTab` hook `__renderActiveTab` checks).
        metadata_rows=[],
        # 2026-09-02 (explicit user request: "on global map compare with
        # should allow to select city core/metro, scenario and place so
        # that cities can be compared between them in regression anova
        # distribution etc") -- same manifest already handed to
        # `_stats_panel_js` below for the Place-rank tab, reused here for
        # each OTHER tab's own new "Compare with" place row (see
        # `_stats_panel_html`'s own docstring).
        place_manifest=place_rank_manifest,
    )
    stats_panel_js = _stats_panel_js(
        regression_fields, is_us=False, region="global",
        enable_place_comparison=True, has_multi_area=True,
        place_rank_only=False, place_rank_fetch_prefix="",
        place_manifest=place_rank_manifest,
    )
    stats_panel_block_html = (
        # `_stats_panel_js` calls a handful of helpers (`__fieldLabel` and
        # its dependencies) that a real per-city `map.html` gets for free
        # from OTHER injected scripts this lightweight page-generator never
        # includes -- without this, every stats tab that labels a field
        # (Regression/ANOVA/Distribution) throws "__fieldLabel is not
        # defined" and silently renders nothing (verified live via headless
        # console errors: `window.__statsTab` updated correctly on tab
        # click, but the exception aborted `renderAnova` before it wrote
        # anything to `#anovaBars`). See `_stats_panel_helper_js`'s own
        # docstring -- same fix `_inject_maplibre_stats_panel_into_saved_html`
        # needed for MapLibre's own per-city pages.
        _stats_panel_helper_js()
        + stats_panel_html
        + "<script>\nwindow.addEventListener('load', function() {\n"
        + stats_panel_js
        + "\n});\n</script>\n"
    )
    # "Download data" panel -- only offers cities that actually have a real
    # `downloads_manifest.json` (see `_available_cities()`; absent for a
    # city built before this feature existed). If NONE do yet, the panel is
    # simply omitted rather than shown with an empty/broken place list.
    download_cities = [c for c in cities if c["downloads_manifest"]]
    download_panel_html = ""
    if download_cities:
        download_place_options = [
            {"value": c["key"], "label": c["display_name"], "base_path": c["downloads_base_path"]}
            for c in download_cities
        ]
        download_manifests_json = json.dumps({c["key"]: c["downloads_manifest"] for c in download_cities})
        download_panel_html = build_download_panel_block(
            download_manifests_json, place_options=download_place_options,
        )
    html = _PAGE_TEMPLATE.format(
        initial_src=cities[0]["map_src"],
        initial_key_json=json.dumps(cities[0]["key"]),
        dropdown_options=dropdown_options,
        cities_json=json.dumps(cities),
        abs_to_rel_json=json.dumps(_POPUP_ABS_TO_REL),
    )
    # Injected by plain string replacement, NOT through `.format()` above --
    # `stats_panel_block_html`/`download_panel_html` are real HTML/JS from
    # `transitlos.map.build` containing their own unescaped `{`/`}` (object
    # literals, CSS rules), which `.format()` would choke on.
    html = html.replace("<!--STATS_PANEL-->", stats_panel_block_html)
    html = html.replace("<!--DOWNLOAD_PANEL-->", download_panel_html)
    out_path = out_path or (ROOT / "combined_map.html")
    out_path.write_text(html)
    return out_path


if __name__ == "__main__":
    path = build_combined_map()
    print(f"[combined_map] wrote {path}")
