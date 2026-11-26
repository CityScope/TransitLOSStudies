"""Orchestrates one city's transit-LOS study end to end.

`run_city_study` wires together `transitlos` (stops, network, LOS),
`studytools.h3_population` (WorldPop -> H3), US Census joins (for US cities),
`studytools.stats` (regression/ANOVA/equity flag), `studytools.city_core` (core/metro
split), `studytools.figures`, and `transitlos.map` into one call per city.

Note on race: ACS has no race/ethnicity breakdown at all, so race is joined
separately from decennial DHC (`_join_race`, full-count data -- more
accurate than an ACS sample estimate would be anyway) rather than folded
into `EQUITY_VAR_CANDIDATES`. `poverty_share` remains one of USA's four
equity-split variables alongside `car_free_share`, `renter_share`, and
`rent_burden_share`; `EQUITY_VAR_CANDIDATES` also carries real,
country-specific candidates for every other wired country (see that dict's
own docstring/comments) so their stats panels aren't limited to the
population-density ANOVA alone.
"""

from __future__ import annotations

import gc
import json
import dataclasses
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import polars as pl
import shapely
from geohierarchy import GeoHierarchy
from geohierarchy.aggregation import Mean, Sum
from transitlos.level_of_service import compute_level_of_service
from transitlos.network import prepare_street_network
from transitlos.stops import download_and_prepare_stops, download_and_prepare_stops_per_feed
from transitlos.stop_scores import compute_stop_scores
from transitlos.scoring import mode_category
from UrbanAccessAnalyzer import h3_ops
from UrbanAccessAnalyzer.api import AreaOfInterest

from .config import CityConfig
from .city_core import resolve_core_boundary
from .figures import access_distribution_bar, anova_diff_bar, percent_area_bar, regression_scatter
from .h3_population import population_and_access_to_h3
from .params import StudyParams, apply_memory_dial
from .stats import (
    access_distribution,
    discretize_score,
    equity_flag,
    equity_flag_from_thresholds,
    equity_flag_thresholds,
    linreg,
    weighted_mean,
    weighted_median,
    weighted_median_split_anova,
)

def _h3_cell_centroids(h3_cells) -> np.ndarray:
    """Analytic H3-cell centroids as a numpy array of shapely Points (EPSG:4326).

    Every row of ``h3_grid``/``stats_grid`` is one H3 cell (its ``h3_cell``
    column), so its true centroid is exactly ``h3.cell_to_latlng`` -- no need
    to materialize the hexagon polygon and run a shapely polygon-centroid
    algorithm over it (or reproject that polygon first). H3's lat/lng is
    always WGS84 regardless of the GeoDataFrame's nominal CRS, and
    `h3_ops.to_gdf` builds these grids with ``crs="EPSG:4326"`` already, so
    the points below are directly comparable to `core_union` (also 4326)
    with no `.to_crs()` needed.

    Mirrors `UrbanAccessAnalyzer.h3_ops._cell_polygons`'s preference for
    `h3ronpy`'s Arrow-native vectorized path when available, falling back to
    the always-available `h3` package's per-cell loop otherwise.
    """
    cells = pd.Series(h3_cells).astype(str) if not isinstance(h3_cells, pd.Series) else h3_cells.astype(str)
    try:
        import warnings

        import h3ronpy
        import h3ronpy.vector as h3ronpy_vector
        import pyarrow as pa

        wkb = h3ronpy_vector.cells_to_wkb_points(h3ronpy.cells_parse(pa.array(cells)))
        with warnings.catch_warnings():
            # See `h3_ops._cell_polygons`: h3ronpy hands back a
            # `geoarrow.wkb`-typed Arrow array that polars falls back to
            # plain binary storage for (with a warning) -- exactly what's
            # wanted here since the bytes go straight into `shapely.from_wkb`.
            warnings.simplefilter("ignore", UserWarning)
            return shapely.from_wkb(pl.Series(wkb).to_numpy())
    except ImportError:
        lats_lngs = [h3.cell_to_latlng(cell) for cell in cells]
        return shapely.points([(lng, lat) for lat, lng in lats_lngs])


def polars_vertex_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Vertex-mean centroid for *any* polygon, computed in Polars over a metric CRS.

    A representative point per polygon -- deliberately the mean of the
    exterior ring's vertex X's and mean of its vertex Y's, not the true
    area-weighted geometric centroid -- good enough for core/metro
    membership tests, cheap to get: no per-row Python `.centroid` or
    `.exterior.coords` loop, no shapely polygon-centroid math.

    Reprojects to the same per-geometry UTM zone `estimate_utm_crs()` picks
    (the helper already used for `area_m2` elsewhere in this module) before
    averaging, so the mean is over real meters, not lat/lon degrees --
    important for polygons spanning enough latitude/longitude for degree
    averaging to visibly skew the result. Reprojects the resulting points
    back to `gdf`'s own CRS before returning.

    `shapely.get_coordinates(..., return_index=True)` extracts every vertex
    of every geometry in one vectorized call (fast at scale, unlike a
    per-geometry `.exterior.coords` Python loop), tagged with which source
    row it came from; a lazy Polars group-by-mean then reduces vertices to
    one (x, y) per row. For a MultiPolygon this pools every part's vertices
    into a single mean; for a Polygon with holes it also pools interior-ring
    vertices -- both an acceptable bias for a representative-point use case.
    """
    if gdf.empty:
        return gpd.GeoSeries([], crs=gdf.crs, index=gdf.index)

    src_crs = gdf.crs
    utm_crs = gdf.estimate_utm_crs()
    proj_geoms = gdf.geometry.to_crs(utm_crs).to_numpy()

    coords, row_idx = shapely.get_coordinates(proj_geoms, return_index=True)

    centroid_xy = (
        pl.LazyFrame({"row": row_idx, "x": coords[:, 0], "y": coords[:, 1]})
        .group_by("row", maintain_order=True)
        .agg(pl.col("x").mean(), pl.col("y").mean())
        .collect()
    )

    pts = gpd.GeoSeries(
        shapely.points(centroid_xy["x"].to_numpy(), centroid_xy["y"].to_numpy()),
        index=gdf.index[centroid_xy["row"].to_numpy()],
        crs=utm_crs,
    ).to_crs(src_crs)
    return pts.reindex(gdf.index)


# Denominators/numerators below use the BARE canonical name wherever
# `_rename_canonical_columns` renames that field (e.g. `acs5_households` ->
# `households`, `acs5_povertyPopulation` -> `povertyPopulation`,
# `acs5_publicTransportCommuters` -> `publicTransportCommuters`,
# `acs5_medianHouseholdIncome` -> `medianHouseholdIncome`) and the original
# `acs5_`-prefixed name for any field with no exact `global_schema.json`
# equivalent (`acs5_vehiclesHouseholds0`, `acs5_rentBurdened`/
# `acs5_rentBurdenPopulation`, `acs5_workers`) -- see `_rename_canonical_columns`.
# --------------------------------------------------------------------------
# Country-generic equity/income/transit variable lookup for
# `_run_regressions_and_anovas`.
#
# These used to be hardcoded to USA-only ACS5 column names (`US_EQUITY_VARS`/
# `US_TRANSIT_VAR`/`US_INCOME_VAR`, kept below only in spirit -- every
# non-US city's stats panel silently only ever showed the population-density
# ANOVA). Generalized 2026-08-23: instead of a second layer of per-country
# constants (the same rot this is fixing), each entry below is an ordered
# list of CANDIDATE column names already materialized on `h3_grid` by
# `_add_share_columns` (`SHARE_COLUMNS`'s numerator/denominator ratios AND
# `RATE_SOURCE_COLUMNS`' already-published rates, both normalized onto the
# same 0-1 fraction convention) -- the first candidate present on a given
# study's grid wins, so one country's real data never collides with
# another's (the same one-country-per-study invariant the rest of this
# module relies on). USA's own candidates are listed first everywhere below,
# so USA's stats panel is byte-for-byte the same four equity vars + the same
# income + transit variables as before this refactor.
#
# `EQUITY_VAR_CANDIDATES` keys are independent equity dimensions (unlike the
# old dict, more than the historical 4 US keys) -- every key with a real
# candidate present on a given grid gets its own regression/ANOVA, so e.g.
# Israel can get `renter_share` AND `owner_share` AND `foreign_born_share`
# simultaneously, not just one "winner".
EQUITY_VAR_CANDIDATES: dict[str, list[str]] = {
    # USA (unchanged from the old `US_EQUITY_VARS`).
    "car_free_share": ["acs_car_free_rate", "cbs_no_vehicle_rate"],
    "renter_share": ["acs_renter_rate", "statcan_renter_rate", "cbs_renter_rate"],
    "poverty_share": ["acs_poverty_rate", "casen_poverty_rate"],
    "rent_burden_share": ["acs_rent_burden_rate"],
    # New, added generally (not per-country) 2026-08-23: real for Germany,
    # Taiwan (via `moi_foreignBornPopulation`-less path -- Taiwan has no
    # foreign-born field, uses `moi_minority_ethnicity_share` instead),
    # Canada, Israel, Spain, Andorra -- see each country's `*_KEEP_COLUMNS`.
    "foreign_born_share": ["foreign_born_share"],
    "owner_share": ["statcan_owner_rate", "cbs_owner_rate"],
    "minority_ethnicity_share": ["minority_ethnicity_share", "moi_minority_ethnicity_share"],
}

# Real total-commuters transit-mode-share pair/rate, checked in order;
# `transit_share` itself is a `SHARE_COLUMNS` entry with both a USA
# (`acs5_workers` denominator) and Canada (`statcan_commuteTotalPopulation`
# denominator) real candidate pair, so this list only needs the one
# already-materialized share name.
TRANSIT_VAR_CANDIDATES: list[str] = ["transit_share"]

# Real median-household-income-equivalent column, checked in order.
# `medianHouseholdIncome` is a canonical `global_schema.json` field, so
# `_rename_canonical_columns` renames BOTH USA's `acs5_medianHouseholdIncome`
# and Canada's `statcan_medianHouseholdIncome` onto this one bare name --
# already country-generic with no extra work. Israel's CBS has no household
# income field at all; its median annual WAGE fields
# (`cbs_selfEmployedMedianAnnualWage`/`cbs_employeesMedianAnnualWage`) are a
# real, imperfect but reasonable substitute (per-worker wage, not
# per-household income) -- listed last, after the two real household-income
# sources, and only used when neither is present. Spain currently has no
# real income field wired at all (investigated -- INE/Censo2021 don't expose
# one at section level) and gets no income ANOVA.
INCOME_VAR_CANDIDATES: list[str] = [
    "medianHouseholdIncome",
    "cbs_selfEmployedMedianAnnualWage",
    "cbs_employeesMedianAnnualWage",
]


def _first_present_column(gdf, candidates: list[str]) -> Optional[str]:
    """The first of `candidates` that's an actual column on `gdf`, or `None`."""
    for col in candidates:
        if col in gdf.columns:
            return col
    return None

# Relative (share/rate) versions of the key absolute counts kept in
# `ACS_KEEP_COLUMNS`/`DHC_KEEP_COLUMNS`, materialized as real columns by
# `_add_share_columns` so they're selectable in the map's generic per-cell
# numeric-field dropdown (regression/ANOVA tabs), not just usable inside the
# `EQUITY_VAR_CANDIDATES`/`TRANSIT_VAR_CANDIDATES` regressions above. Per the
# study plan: every equity dimension should be available both as an
# absolute count and a relative share, with relative as the one meant to be
# used by default (absolute stays available for anyone who wants raw
# totals, e.g. `acs_population` itself, which is inherently absolute and
# has no share form).
#
# NOTE (schema rename, added this session): every numerator/denominator below
# uses the BARE canonical name wherever `_rename_canonical_columns` (see
# above) renames that field after the census join -- e.g. `acs5_households`/
# `inegi_households` -> `households`, `dhc_malePopulation`/
# `inegi_malePopulation` -> `malePopulation`, `acs5_laborForce`/
# `inegi_laborForce` -> `laborForce`. `population` itself is the one
# canonical field EXCLUDED from that rename (see
# `_RENAME_EXCLUDED_CANONICAL_NAMES`), so every `*_population` denominator
# below (`acs5_population`, `dhc_population`, `inegi_population`) keeps its
# real source prefix. A field with no exact `global_schema.json` equivalent
# (`acs5_workers`, `acs5_rentBurdened`/`acs5_rentBurdenPopulation`,
# `inegi_populationCatholic`, ...) also keeps its prefixed name unchanged.
# Each value is a LIST of candidate `(numerator, denominator)` pairs, not a
# single pair: two countries can compute "the same" derived rate against a
# different real denominator (e.g. labor-force rate over 16+ in the US
# Census vs. over 18+/adult in INEGI) -- `_add_share_columns` tries each
# candidate in order and uses the first whose both columns are present, so
# only one country's data is ever live in a given study (the same
# one-country-per-study invariant `_rename_canonical_columns` relies on) and
# there is no ambiguity about which candidate "wins".
#
# UNIFICATION RULE (this session, per explicit user request: "always the
# same column names across countries"): a derived name is unified (its
# country prefix dropped) iff its NUMERATOR is itself a bare canonical
# `global_schema.json` field (i.e. already renamed by
# `_rename_canonical_columns`) -- that's the objective signal that the
# underlying concept is genuinely universal, not just similarly-named. A
# derived name whose numerator has no canonical equivalent (religion
# sub-breakdowns, dwelling amenities, healthcare, disability, `acs5_workers`-
# based rates, race/ethnicity categories, rent burden, education) keeps its
# source prefix -- unifying those would claim a cross-country
# comparability that doesn't exist.
def _share_pair_ok(total, num: str, den: str) -> bool:
    """A share candidate is usable when its numerator has data (not zero everywhere) and does not add up to more than its whole
    (2026-11-15: Gipuzkoa's all-zero employedResidents gave a 0 % unemployment rate, and a labor force from one source over a working-age
    population from another gave 100 % participation). `total(col)` = the column's sum over the grid (NaN skipped)."""
    try:
        n, d = float(total(num)), float(total(den))
    except Exception:  # noqa: BLE001
        return True
    return n > 0 and d > 0 and n <= 1.05 * d


def _pd_total(gdf):
    return lambda c: pd.to_numeric(gdf[c], errors="coerce").sum()


SHARE_COLUMNS: dict[str, list[tuple[str, str]]] = {
    # -- housing / tenure (no canonical numerator -- stay prefixed) --
    "acs_renter_rate": [("acs5_householdsRenter", "households")],
    "acs_owner_rate": [("acs5_householdsOwner", "households")],
    "acs_rent_burden_rate": [("acs5_rentBurdened", "acs5_rentBurdenPopulation")],
    "acs_car_free_rate": [("acs5_vehiclesHouseholds0", "households")],
    # -- income / poverty (no canonical numerator -- stay prefixed) --
    "acs_poverty_rate": [("acs5_povertyBelow100", "povertyPopulation")],
    # -- employment (canonical numerator -- unified name) --
    # US denominates labor-force rate over 16+, Mexico over adults (18+);
    # genuinely different real denominators for "the same" concept, so both
    # are listed as candidates under one unified name.
    "labor_force_rate": [
        ("laborForce", "acs5_population16plus"),
        ("eustat_laborForce", "eustat_population16plus"),   # 2026-11-15: numerator and denominator of one source before mixed ones
        ("laborForce", "adultPopulation"),
    ],
    "unemployment_rate": [("unemployedResidents", "laborForce"), ("eustat_unemployedResidents", "eustat_laborForce")],
    "employment_rate": [("employedResidents", "laborForce"), ("eustat_employedResidents", "eustat_laborForce")],
    # -- workers (commuters by residence) --
    # `acs_worker_rate` is denominated by total census population, answering
    # "how much of this cell is commuters at all" -- `acs5_workers` itself has
    # no canonical equivalent, so this stays prefixed. The mode-share numerators
    # (transit/car/walk/bike commuters) ARE canonical, so those unify; their
    # shared denominator (`acs5_workers`, "people who commute at all") has no
    # canonical equivalent either, but per the unification rule only the
    # NUMERATOR decides -- a mode's share of commuters is a universal-enough
    # concept even though `acs5_workers` itself stays a source-specific count.
    "acs_worker_rate": [("acs5_workers", "acs5_population")],
    # Canada's StatCan commute table has no `acs5_workers`-equivalent name --
    # its own real total-commuters denominator is `statcan_commuteTotalPopulation`
    # (see `CANADA_KEEP_COLUMNS`) -- added as a second real candidate so
    # `transit_share`/`car_commute_share`/etc. fire for Canada too, not just
    # the US ACS join.
    "transit_share": [
        ("publicTransportCommuters", "acs5_workers"),
        ("publicTransportCommuters", "statcan_commuteTotalPopulation"),
    ],
    "car_commute_share": [
        ("carCommuters", "acs5_workers"),
        ("carCommuters", "statcan_commuteTotalPopulation"),
    ],
    "walk_commute_share": [
        ("walkCommuters", "acs5_workers"),
        ("walkCommuters", "statcan_commuteTotalPopulation"),
    ],
    "bike_commute_share": [
        ("bikeCommuters", "acs5_workers"),
        ("bikeCommuters", "statcan_commuteTotalPopulation"),
    ],
    # -- commuting, Germany (BA Pendlerstatistik; added 2026-08-25 audit) --
    # No canonical equivalent exists for either concept, so both stay
    # prefixed. Denominated by `ba_jobsTotal` (AMK's workplace-jobs total,
    # already joined for every Germany city) rather than
    # `ba_pendler_pendlerJobsAtWorkplace` (that Pendlerstatistik column is
    # deliberately NOT kept on `h3_grid`/`census_gdf` -- see
    # `GERMANY_BA_PENDLER_KEEP_COLUMNS`'s comment -- and the two are
    # verified numerically identical for the same Stichtag anyway, see
    # `pycensus.census.countries.germany.ba_pendler.loader`'s module docstring).
    "commuter_in_share": [("ba_pendler_einpendler", "ba_jobsTotal")],
    "local_worker_share": [("ba_pendler_sameGemeindeWorkers", "ba_jobsTotal")],
    # -- education (no canonical numerator -- stay prefixed) --
    "acs_bachelors_rate": [("acs5_educationBachelorsPlus", "acs5_population25plus")],
    "acs_less_than_hs_rate": [("acs5_educationLessThanHs", "acs5_population25plus")],
    "acs_high_school_rate": [("acs5_educationHighSchool", "acs5_population25plus")],
    "acs_some_college_rate": [("acs5_educationSomeCollege", "acs5_population25plus")],
    # -- education, Canada (canonical numerators -- unified names; added
    # 2026-08-25 audit) -- `educationPrimaryOrLessPopulation`/
    # `educationSecondaryPopulation`/`educationUniversityPopulation`/
    # `educationOtherTertiaryPopulation` ARE canonical `global_schema.json`
    # fields (verified by reading it directly) and land bare after rename
    # for Canada (see `CANADA_KEEP_COLUMNS`) -- unlike the US's ACS
    # attainment buckets above (a different, non-equivalent categorization
    # with no canonical match), these had no share form at all despite
    # being real, already-fetched, already-unified columns. Denominated by
    # `statcan_educationTotalPopulation` (real, Ontario-only like the rest
    # of StatCan's education table -- see `CANADA_KEEP_COLUMNS`'s own
    # comment -- stays prefixed, no canonical equivalent). Spain has the
    # same four canonical numerators (`ine_education*Population`, minus
    # `OtherTertiary` -- see `SPAIN_KEEP_COLUMNS`'s own comment on why) but
    # no real matching "education total" denominator column, so it is
    # deliberately NOT added as a candidate here (would require inventing a
    # denominator).
    "education_primary_or_less_share": [("educationPrimaryOrLessPopulation", "statcan_educationTotalPopulation")],
    "education_secondary_share": [("educationSecondaryPopulation", "statcan_educationTotalPopulation")],
    "education_university_share": [("educationUniversityPopulation", "statcan_educationTotalPopulation")],
    "education_other_tertiary_share": [("educationOtherTertiaryPopulation", "statcan_educationTotalPopulation")],
    # -- nativity (canonical numerator -- unified name; added 2026-08-22
    # alongside `acs5_foreignBornPopulation`) --
    # `foreignBornPopulation` itself is canonical and renamed bare regardless
    # of source country (USA/Spain/Canada/Germany/Andorra/Israel all land it
    # under this exact name -- see each country's `*_KEEP_COLUMNS`), but its
    # real census-population denominator stays source-prefixed (`population`
    # itself is excluded from the canonical rename -- see
    # `_RENAME_EXCLUDED_CANONICAL_NAMES`), so every wired country's own
    # population column is listed as a candidate denominator here. This is
    # the single most widely-available equity dimension across countries
    # wired so far (added generally, not per-country, 2026-08-23).
    "foreign_born_share": [
        ("foreignBornPopulation", "acs5_population"),
        ("foreignBornPopulation", "ine_population"),
        ("foreignBornPopulation", "statcan_population"),
        ("foreignBornPopulation", "destatis_population"),
        ("foreignBornPopulation", "estadisticaad_population"),
        ("foreignBornPopulation", "cbs_population"),
    ],
    # -- Eustat municipal indicators, Gipuzkoa (real PXWeb BDE fields joined
    # at municipality resolution alongside INE section data -- see
    # `EUSTAT_MUNICIPAL_KEEP_COLUMNS`/`_eustat_municipal_loader`, added
    # 2026-08-25). No canonical `global_schema.json` equivalent for
    # establishments/births/deaths, so numerators stay `eustat_`-prefixed;
    # denominated over `ine_population` (the finer, more accurate real
    # source already joined for Spain -- see `spain.censo2021` module
    # docstring's accuracy comparison) rather than reinventing a
    # municipality-level population figure.
    "eustat_establishments_rate": [("eustat_establishments", "ine_population")],
    "eustat_birth_rate_2021_2025": [("eustat_birthsPeriod2021_2025", "ine_population")],
    "eustat_death_rate_2021_2025": [("eustat_deathsPeriod2021_2025", "ine_population")],
    # -- tenure, Canada (StatCan `statcan_renterHouseholds`/`statcan_ownerHouseholds`
    # -- fixed 2026-08-25 audit: `renterHouseholds`/`ownerHouseholds` are NOT
    # in `global_schema.json`'s canonical feature-name set (verified by
    # reading it directly), so `_rename_canonical_columns` never renames them
    # off their `statcan_` prefix -- the previous bare-name candidates here
    # could never match a real column, so `statcan_renter_rate`/
    # `statcan_owner_rate` silently never fired for Canada despite
    # `statcan_renterHouseholds`/`statcan_ownerHouseholds` being real,
    # already-fetched columns (confirmed live in `toronto/results/core/
    # h3_grid.parquet`). `households` (unlike `renterHouseholds`) IS
    # canonical and IS renamed bare -- `statcan_households` itself no longer
    # exists as a column post-rename, so the denominator is fixed to match.) --
    # (`ownerHouseholds` / `renterHouseholds` later became global columns: the loader now returns them bare, so both forms)
    "statcan_renter_rate": [("statcan_renterHouseholds", "households"), ("renterHouseholds", "households")],
    "statcan_owner_rate": [("statcan_ownerHouseholds", "households"), ("ownerHouseholds", "households")],
    # -- tenure/vehicle/foreign-resident, Israel (CBS already publishes these
    # as rates, not raw counts -- see `RATE_SOURCE_COLUMNS` below for those;
    # this entry is for the household-count-shaped `renterHouseholds`, not
    # present for Israel, kept only as a placeholder note) --
    # -- minority ethnicity, Taiwan (`moi_minorityEthnicityPopulation` is
    # NOT a canonical numerator -- MOI's aborigine-population concept has no
    # exact `global_schema.json` match -- so this stays its own prefixed
    # entry rather than folding into the Mexico-only `minority_ethnicity_share`
    # above, which is denominated over `inegi_population`) --
    "moi_minority_ethnicity_share": [("moi_minorityEthnicityPopulation", "moi_population")],
    # -- race / ethnicity (decennial DHC; see `_join_race`) --
    # The five race categories partition `dhc_population` (P3), so their
    # shares sum to ~1; `dhc_hispanic_share` is ethnicity (P5) and
    # deliberately overlaps them, as in the Census's own tabulation. None of
    # the five race/hispanic category names are exact `global_schema.json`
    # matches (only `minorityEthnicityPopulation` is, and DHC doesn't publish
    # that single aggregate), so they stay `dhc_`-prefixed.
    "dhc_white_share": [("dhc_whitePopulation", "dhc_population")],
    "dhc_black_share": [("dhc_blackPopulation", "dhc_population")],
    "dhc_asian_share": [("dhc_asianPopulation", "dhc_population")],
    "dhc_native_share": [("dhc_nativePopulation", "dhc_population")],
    "dhc_other_race_share": [("dhc_otherRacePopulation", "dhc_population")],
    "dhc_hispanic_share": [("dhc_hispanicPopulation", "dhc_population")],
    # DHC's own directly-published "any non-white" aggregate (P3) -- not a
    # sum of the five race categories above, kept prefixed like them (no
    # exact `global_schema.json` equivalent).
    "dhc_nonwhite_share": [("dhc_nonwhitePopulation", "dhc_population")],
    # Working-age (18-64) -- the natural complement of `children_share`/
    # `elderly_share` below; no canonical equivalent, stays prefixed.
    "dhc_working_age_share": [("dhc_age18to64Population", "dhc_population")],
    # -- gender / age (decennial DHC P12 + INEGI -- canonical numerators, unified) --
    # DHC denominates by `dhc_population`, INEGI by `inegi_population` (both
    # real source-specific population counts, deliberately excluded from the
    # canonical rename -- see `_RENAME_EXCLUDED_CANONICAL_NAMES`).
    # Added (2026-08-25 audit): Spain (ine_malePopulation/ine_femalePopulation),
    # Chile (ine_cl_malePopulation/ine_cl_femalePopulation), Taiwan
    # (moi_malePopulation/moi_femalePopulation), and Germany
    # (destatis_malePopulation/destatis_femalePopulation) all have the same
    # canonical male/femalePopulation numerator plus their own real
    # population denominator already in `*_KEEP_COLUMNS`, but were missing
    # from these candidate lists -- a real gap, not speculative (every
    # denominator added below is a real, already-fetched column).
    "female_share": [
        ("femalePopulation", "dhc_population"),
        ("femalePopulation", "inegi_population"),
        ("femalePopulation", "ine_population"),
        ("femalePopulation", "ine_cl_population"),
        ("femalePopulation", "moi_population"),
        ("femalePopulation", "destatis_population"),
    ],
    "male_share": [
        ("malePopulation", "dhc_population"),
        ("malePopulation", "inegi_population"),
        ("malePopulation", "ine_population"),
        ("malePopulation", "ine_cl_population"),
        ("malePopulation", "moi_population"),
        ("malePopulation", "destatis_population"),
    ],
    # Taiwan's `moi_under18Population`/`moi_over65Population` (MOI RIS
    # ODRP014 single-year-age sums, see `TAIWAN_KEEP_COLUMNS`) are the same
    # canonical under18/over65Population fields DHC provides for the US --
    # added as a second real candidate denominated over `moi_population`
    # (2026-08-25 audit; previously only DHC fired for these).
    "children_share": [("under18Population", "dhc_population"), ("under18Population", "moi_population")],
    "elderly_share": [("over65Population", "dhc_population"), ("over65Population", "moi_population")],
    # Taiwan's `moi_adultPopulation` (MOI RIS ODRP014, ages 18+ -- verified
    # to match global_schema.json's `adultPopulation` definition exactly,
    # same 18+ boundary as Mexico's INEGI `adultPopulation`) added as a
    # second real candidate (2026-08-25 audit).
    "adult_share": [("adultPopulation", "inegi_population"), ("adultPopulation", "moi_population")],
    "minority_ethnicity_share": [("minorityEthnicityPopulation", "inegi_population")],
    "religious_share": [("religiousPopulation", "inegi_population")],

    # -- Mexico (INEGI, `inegi_*` -- see `INEGI_KEEP_COLUMNS`) -- religion
    # sub-breakdowns, healthcare, disability, dwelling amenities: no
    # canonical numerator, stay prefixed. Mexico's census join lands only
    # absolute counts (same as ACS/DHC before this dict existed), so without
    # these Mexico's map "Circle size by"/"Opacity by"/regression/ANOVA
    # dropdowns only ever offered `pop_density` and `inegi_educationMeanGrade`
    # (the only two `inegi_*` columns whose raw name happens to match
    # `_RELATIVE_MARKERS`) even though the real join carries 25 genuine
    # INEGI fields; there is no Mexican equivalent of DHC race/ethnicity or
    # LODES jobs-by-workplace (see `_join_race`/`_join_jobs`'s country='MEX'
    # skip messages), so those two categories genuinely have no share form
    # here.
    "inegi_catholic_share": [("inegi_populationCatholic", "inegi_population")],
    "inegi_no_religion_share": [("inegi_populationNoReligion", "inegi_population")],
    # Added (2026-08-25 audit): the other two INEGI religion sub-breakdowns
    # already kept in `INEGI_KEEP_COLUMNS` (real counts, same denominator
    # convention as `inegi_catholic_share`/`inegi_no_religion_share`/
    # `religious_share` immediately above) had no share form and so never
    # appeared in the map's ANOVA/regression/opacity/circle-size dropdowns --
    # a real gap, not a speculative addition; the numerator/denominator pair
    # is exactly the same shape as the other two religion shares.
    "inegi_evangelical_share": [("inegi_populationEvangelicalProtestant", "inegi_population")],
    "inegi_other_religion_share": [("inegi_populationOtherReligion", "inegi_population")],
    "inegi_healthcare_coverage_rate": [("inegi_populationWithHealthcare", "inegi_population")],
    "inegi_disability_rate": [("inegi_populationDisability", "inegi_population")],
    "inegi_internet_rate": [("inegi_dwellingsInternet", "inegi_occupiedDwellings")],
    "inegi_computer_rate": [("inegi_dwellingsComputer", "inegi_occupiedDwellings")],
    "inegi_cellphone_rate": [("inegi_dwellingsCellphone", "inegi_occupiedDwellings")],
    # 2026-09-01 (live user report, popup): "Occupied dwellings" had no
    # share row at all -- unlike internet/computer/cellphone above (all
    # denominated by `inegi_occupiedDwellings` itself), occupancy's own
    # natural denominator is the TOTAL dwelling count, `inegi_dwellings`.
    "inegi_occupancy_rate": [("inegi_occupiedDwellings", "inegi_dwellings")],
    # `carHouseholds` is a canonical numerator (renamed from
    # `inegi_carHouseholds`), so this unifies despite currently being the
    # only country with car-ownership data.
    "car_ownership_rate": [("carHouseholds", "households")],

    # -- urban/rural, Chile (INE, `ine_cl_*` -- see `CHILE_KEEP_COLUMNS`) --
    # Added (2026-08-25 audit): `ine_cl_urbanPopulation`/`ine_cl_ruralPopulation`
    # are real, already-fetched INE fields (no exact `global_schema.json`
    # equivalent -- see `CHILE_KEEP_COLUMNS`'s own comment -- so they stay
    # prefixed) that partition `ine_cl_population` exactly like ACS's
    # renter/owner households pair above, but had no share form wired at
    # all.
    "ine_cl_urban_share": [("ine_cl_urbanPopulation", "ine_cl_population")],
    "ine_cl_rural_share": [("ine_cl_ruralPopulation", "ine_cl_population")],

    # -- religion, Israel (CBS, `cbs_*` -- see `ISRAEL_KEEP_COLUMNS`).
    # Added (2026-08-25): CBS reports religion as a single per-area
    # dominant-religion classification code, not a population breakdown
    # (see `pycensus.census.countries.israel.cbs.schema.cbs_schema.json`'s
    # "religionCode" feature) -- `cbs.loader.load()` reconstructs real
    # per-religion population counts from that classification (live-
    # verified 2026-08-25: Beersheba metro AOI sums to 948,640 Jewish +
    # 184,920 Muslim = 1,133,560, matching total population exactly; no
    # Christian/Druze/Other in this particular AOI, but the columns are
    # real for any Israeli metro that has them). Same numerator/denominator
    # shape as Mexico's religion shares above.
    "cbs_jewish_share": [("cbs_jewishPopulation", "cbs_population")],
    # (pycensus gives these two their canonical bare names, `muslimPopulation` / `christianPopulation`, so the shares need that numerator too)
    "cbs_muslim_share": [("cbs_muslimPopulation", "cbs_population"), ("muslimPopulation", "cbs_population")],
    "cbs_christian_share": [("cbs_christianPopulation", "cbs_population"), ("christianPopulation", "cbs_population")],
    "cbs_druze_share": [("cbs_druzePopulation", "cbs_population")],
    "cbs_other_religion_share": [("cbs_otherReligionPopulation", "cbs_population")],

    # -- religious observance ("degree of religiosity"), Israel (CBS,
    # `cbs_*`). Added (2026-08-25): same shape as the religion shares
    # above, from CBS's secular/traditional/religious/Haredi per-area
    # classification (live-verified: Beersheba locality sums to 165,960
    # secular + 48,420 traditional + 3,220 religious = 217,600, matching
    # total population exactly; no Haredi population in Beersheba proper).
    "cbs_secular_share": [("cbs_secularPopulation", "cbs_population")],
    "cbs_traditional_religiosity_share": [("cbs_traditionalReligiosityPopulation", "cbs_population")],
    "cbs_religious_observant_share": [("cbs_religiousObservantPopulation", "cbs_population")],
    "cbs_haredim_share": [("cbs_haredimPopulation", "cbs_population")],
    "cbs_other_religiosity_share": [("cbs_otherReligiosityPopulation", "cbs_population")],

    # -- WorldPop census-gap-fill age/sex breakdown (`worldpop_*` -- see
    # `CityConfig.census_worldpop_gapfill`/`_add_worldpop_gapfill`). Added
    # 2026-08-25 for Beersheba's real ~49%-by-area CBS coverage gap.
    # Denominated over WorldPop's own population (`pop`-family, already
    # covers the whole grid including gap cells), NEVER the map-facing
    # `population` column -- once `_finalize_population_columns` derives
    # `population` fresh per-city (which can be `census_residents` or even
    # `census_residents` + jobs for Boston/SF), using it here would silently
    # divide a WorldPop numerator by a non-WorldPop, non-comparable
    # denominator. `"population_worldpop"` (Hamburg's Zensus-replaces-
    # `population` stash -- see `_join_zensus_grid_population`) is checked
    # FIRST so Hamburg divides by its own real pre-replacement WorldPop
    # figure rather than the Zensus count that has, by this point, already
    # overwritten bare `population` in place; every other city falls
    # through to bare `population`, which -- at the point `_add_share_columns`
    # runs (before `_finalize_population_columns`) -- is still WorldPop's
    # raw, un-rederived count. `is_relative_field` (transitLOS/map/build.py)
    # picks these up via the "share" name marker like every other share
    # here, so they surface correctly in the map's ANOVA/opacity dropdowns.
    "worldpop_male_share": [("worldpop_malePopulation", "worldpop_residents"), ("worldpop_malePopulation", "population_worldpop"), ("worldpop_malePopulation", "population")],
    "worldpop_female_share": [("worldpop_femalePopulation", "worldpop_residents"), ("worldpop_femalePopulation", "population_worldpop"), ("worldpop_femalePopulation", "population")],
    "worldpop_children_share": [("worldpop_under18Population", "worldpop_residents"), ("worldpop_under18Population", "population_worldpop"), ("worldpop_under18Population", "population")],
    "worldpop_elderly_share": [("worldpop_over65Population", "worldpop_residents"), ("worldpop_over65Population", "population_worldpop"), ("worldpop_over65Population", "population")],
    "worldpop_adult_share": [("worldpop_adultPopulation", "worldpop_residents"), ("worldpop_adultPopulation", "population_worldpop"), ("worldpop_adultPopulation", "population")],
    "worldpop_urban_share": [("worldpop_urbanPopulation", "worldpop_residents"), ("worldpop_urbanPopulation", "population_worldpop"), ("worldpop_urbanPopulation", "population")],

    # -- WorldPop `urbanPopulation` (added 2026-08-25, see
    # `pycensus.census.worldwide.worldpop.loader`'s module docstring for
    # live verification) -- same bare-name shape as the other
    # `_join_worldpop_global_schema` features above (`worldpop_female_share`
    # etc. are for the gap-fill-prefixed columns; this is for Shanghai's
    # bare-name join, which already has "population" itself on `h3_grid`).
    # `births`/`pregnancies` were added alongside this then deliberately
    # dropped (see `GLOBAL_SCHEMA_FEATURES`'s docstring) -- both are fixed
    # at WorldPop's 2015 vintage, outside this project's 2020-2026 data-
    # recency requirement, unlike `urbanPopulation` which has real per-year
    # coverage through 2030.
    "urban_share": [("urbanPopulation", "population")],
}


# Some real census sources publish a rate/share DIRECTLY (no separate
# numerator/denominator count pair to divide) -- e.g. Israel's CBS
# (`cbs_householdsRenterRate`, `cbs_foreignResidentRate`, ...) and Chile's
# Casen SAE poverty estimate (`casen_cl_povertyRate`). `SHARE_COLUMNS` above
# only handles the "divide two counts" shape; this is the second shape, kept
# as its own registry since it needs no division -- just a name (matching
# `SHARE_COLUMNS`'s `_share`/`_rate` naming convention, so the map's generic
# relative-field classifier and `_run_regressions_and_anovas`' candidate
# lookups treat it identically to a computed share) and a scale factor to
# normalize onto SHARE_COLUMNS' 0-1 fraction convention.
#
# Scale factors are picked from each source's own verified real values, not
# guessed: CBS's own loader derives a population count as
# `rate / 100.0 * pop` (see `pycensus.census.countries.israel.cbs.loader.load`), so
# CBS `*Rate` columns are 0-100 percent -- divide by 100. Casen's own module
# docstring gives real verified values already as fractions (Santiago
# `povertyRate=0.038648`), so Casen needs no rescaling (divide by 1).
#
# Each value is a list of `(source_col, divisor)` candidates, same
# first-match-wins convention as `SHARE_COLUMNS`.
RATE_SOURCE_COLUMNS: dict[str, list[tuple[str, float]]] = {
    "cbs_renter_rate": [("cbs_householdsRenterRate", 100.0)],
    "cbs_owner_rate": [("cbs_householdsOwnerRate", 100.0)],
    "cbs_no_vehicle_rate": [("cbs_householdsNoVehicleRate", 100.0)],
    "cbs_foreign_resident_rate": [("cbs_foreignResidentRate", 100.0)],
    "cbs_employment_rate": [("cbs_employmentRate", 100.0)],
    "cbs_self_employed_rate": [("cbs_selfEmployedRate", 100.0)],
    "casen_poverty_rate": [("casen_cl_povertyRate", 1.0)],
    # Added (2026-08-25 audit): Chile Casen's own multidimensional-poverty
    # SAE estimate (`casen_cl_povertyRateMultidimensional`, present in
    # `CHILE_CASEN_KEEP_COLUMNS` alongside `casen_cl_povertyRate`, same
    # Fay-Herriot SAE methodology and already published as a 0-1 fraction --
    # see `pycensus.census.countries.chile.schema.casen_schema.json`'s
    # `povertyRateMultidimensional` feature, `unit: "share (0-1)"`, no
    # rescaling needed) had no `RATE_SOURCE_COLUMNS` entry at all -- a real
    # gap, not speculative.
    "casen_poverty_rate_multidimensional": [("casen_cl_povertyRateMultidimensional", 1.0)],
    # Added (2026-08-25 audit): the 12 real CBS `*Rate`/`populationShare*`
    # fields below are all already-fetched, already-published percentages
    # (0-100, verified live against the CBS ArcGIS layer's own `_pcnt`
    # fields per the module docstring cited above -- e.g.
    # `institutionalizedResidentRate` verified against `inst_pcnt=2.9` for
    # Beersheba) sitting unused in `ISRAEL_KEEP_COLUMNS`; only 6 of CBS's 18
    # real rate-shaped fields had a `RATE_SOURCE_COLUMNS` entry before this
    # audit, so these never appeared in the map's ANOVA/regression/opacity/
    # circle-size dropdowns despite being real, already-fetched columns --
    # same pattern as the Mexico religion-share gap fixed earlier today.
    "cbs_population_share_0_19": [("cbs_populationShare019", 100.0)],
    "cbs_population_share_20_64": [("cbs_populationShare2064", 100.0)],
    "cbs_population_share_65_plus": [("cbs_populationShare65plus", 100.0)],
    "cbs_work_outside_locality_rate": [("cbs_workOutsideLocalityRate", 100.0)],
    "cbs_academic_degree_rate": [("cbs_academicDegreeRate", 100.0)],
    "cbs_households_2plus_vehicle_rate": [("cbs_households2plusVehicleRate", 100.0)],
    "cbs_institutionalized_resident_rate": [("cbs_institutionalizedResidentRate", 100.0)],
    "cbs_labor_force_participation_rate": [("cbs_laborForceParticipationRate", 100.0)],
    "cbs_top_wage_decile_employees_rate": [("cbs_topWageDecileEmployeesRate", 100.0)],
    "cbs_households_with_young_children_rate": [("cbs_householdsWithYoungChildrenRate", 100.0)],
    "cbs_households_with_young_adults_rate": [("cbs_householdsWithYoungAdultsRate", 100.0)],
    "cbs_households_with_parking_rate": [("cbs_householdsWithParkingRate", 100.0)],
}


def _inject_core_outline(city_dir: Path, core_boundary: gpd.GeoDataFrame) -> None:
    """Dotted outline of the urban core (same dotted black line with white halo as the metropolitan areas of the Spain / US maps), own checkbox.

    Writes `results/core_lines.geojson` (shipped with the results folder) and adds the layer script to `map.html`. Non-fatal.
    """
    try:
        import json as _json

        from transitlos.map.metro_areas import lines_script, metro_lines_geojson

        gdf = core_boundary.to_crs(4326).copy()
        gdf["name"] = "Urban core"
        (city_dir / "results").mkdir(parents=True, exist_ok=True)
        (city_dir / "results" / "core_lines.geojson").write_text(_json.dumps(metro_lines_geojson(gdf, tolerance_deg=0.0003)), encoding="utf-8")
        html_path = city_dir / "map.html"
        html = html_path.read_text(encoding="utf-8")
        if "__core_dots" not in html:
            block = lines_script("results/core_lines.geojson", "Urban core", "core", 7.5, width=0.9)
            html = html.replace("</body>", block + "</body>", 1) if "</body>" in html else html + block
            html_path.write_text(html, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] urban-core outline skipped: {exc!r}")


AREA_FILES = {"metro": "un_metro.gpkg", "official": "metro_official.gpkg", "municipality": "municipality.gpkg", "core": "core.gpkg"}
AREA_LABELS = {"metro": "UN metro area", "official": "Metro area", "municipality": "Municipality", "core": "UN urban core"}


def _load_area_unions(city_dir: Path):
    """`{key: shapely geometry (EPSG:4326)}` of the four analysis areas (`scripts_four_areas.py`), or None when a city still has only metro + core."""
    if not all((city_dir / f).is_file() for f in AREA_FILES.values()):
        return None
    from .area_geom import one_piece
    return {k: one_piece(gpd.read_file(city_dir / f).to_crs(4326).union_all(), city_dir.name) for k, f in AREA_FILES.items()}


def _area_masks(h3_cells, unions: dict) -> dict:
    cents = _h3_cell_centroids(h3_cells)
    return {k: np.asarray(shapely.within(cents, g), dtype=bool) for k, g in unions.items()}


AREA_FLAG_COLUMNS = {"metro": "area_un_metro", "official": "area_official_metro", "municipality": "area_municipality", "core": "area_un_core"}


def _add_area_flags(df, unions, centroids=None):
    """`df` with one 0/1 column per analysis area (`AREA_FLAG_COLUMNS`): the data of the four areas is stored once, for the whole processing area."""
    cents = centroids if centroids is not None else _h3_cell_centroids(df["h3_cell"])
    out = df.copy()
    for k, g in unions.items():
        out[AREA_FLAG_COLUMNS[k]] = np.asarray(shapely.within(cents, g), dtype=np.int8)
    return out


def _set_area_labels(city_dir: Path) -> None:
    """Dropdown labels of the four areas from the names in the area files: a city whose official area is named after a region (Biobio, Gipuzkoa, Jalisco) shows that name for the
    official area and the city name for the other three (`Concepción (UN metro area)`); other cities keep the generic labels."""
    try:
        from transitlos.map import build as _b
        off = gpd.read_file(city_dir / AREA_FILES["official"]).name.iloc[0]
        if off == "Metro area":
            return
        city = gpd.read_file(city_dir / AREA_FILES["municipality"]).name.iloc[0]
        _b._AREA_LABELS_4.update({"official": off, "municipality": city, "metro": f"{city} (UN metro area)", "core": f"{city} (UN urban core)"})
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] area labels skipped: {exc!r}")


def _stats_by_area(city_dir: Path, stats_grid, stats_core_mask) -> dict:
    _set_area_labels(city_dir)
    """`stats_by_area` of the map: the four areas (UN metro = `metro`, official metro, municipality, UN core = `core`) or, for a city without the area files, metro + core."""
    unions = _load_area_unions(city_dir)
    if not unions:
        return {"metro": stats_grid, "core": stats_grid[stats_core_mask]}
    masks = _area_masks(stats_grid["h3_cell"], unions)
    return {k: stats_grid[masks[k]] for k in ("metro", "official", "municipality", "core")}


def _inject_area_outlines(city_dir: Path) -> None:
    """Outlines for the area selector: UN metro area (solid) + dotted UN core, and official metro (solid) + dotted municipality.

    Writes `results/area_lines_{un,official}.geojson` and adds `window.__areaOutlines` (switched by the map's Area selector). Non-fatal.
    """
    try:
        import json as _json

        from transitlos.map.metro_areas import lines_script, metro_lines_geojson

        res = city_dir / "results"
        res.mkdir(parents=True, exist_ok=True)
        for fname, groups in (("area_lines_un", (("metro", "UN metro area"), ("core", "UN urban core"))),
                              ("area_lines_official", (("official", "Metro area"), ("municipality", "Municipality")))):
            parts = []
            for key, name in groups:
                from .area_geom import one_piece
                g = gpd.GeoDataFrame({"name": [name]}, geometry=[one_piece(gpd.read_file(city_dir / AREA_FILES[key]).to_crs(4326).union_all(), city_dir.name)], crs=4326)
                parts.append(g[["name", "geometry"]])
            gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
            (res / f"{fname}.geojson").write_text(_json.dumps(metro_lines_geojson(gdf, tolerance_deg=0.0003)), encoding="utf-8")
        # display mask: the tiles hold the whole processing area once, so everything outside the displayed area (UN metro or official metro) is covered by an inverse polygon
        from shapely.geometry import box as _box, mapping as _mapping
        for key, fname in (("metro", "area_mask_un"), ("official", "area_mask_official")):
            from .area_geom import one_piece
            g_ = one_piece(gpd.read_file(city_dir / AREA_FILES[key]).to_crs(4326).union_all(), city_dir.name).simplify(0.0003)
            inv = _box(-179.9, -85.0, 179.9, 85.0).difference(g_)
            (res / f"{fname}.geojson").write_text(_json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": _mapping(inv)}]}), encoding="utf-8")
        html_path = city_dir / "map.html"
        html = html_path.read_text(encoding="utf-8")
        import re as _re
        html = _re.sub(r"<script>\s*\(function\(\) \{\s*// Selected area -> which boundaries are drawn.*?</script>\s*", "", html, flags=_re.S)   # an older outline script is replaced
        if "__areaOutlines" not in html:
            block = ("<script>\n" + AREA_OUTLINES_JS + "\n</script>\n")
            html = html.replace("</body>", block + "</body>", 1) if "</body>" in html else html + block
            html_path.write_text(html, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] area outlines skipped: {exc!r}")


AREA_OUTLINES_JS = r"""
(function() {
  // Selected area -> which boundaries are drawn: UN metro / UN core show the UN metro area (solid) and the UN core (dotted);
  // official metro / municipality show the official metro area (solid) and the municipality (dotted).
  // Everything outside the displayed area is hidden by a second, basemap-only map laid over the data and clipped (CSS clip-path) to the outside of the area,
  // so the data layers (hexagons, census, roads, stops, circles) show only inside the area while the background map still covers everything.
  var GROUPS = {un: 'results/area_lines_un.geojson', official: 'results/area_lines_official.geojson'};
  var MASKS = {un: 'results/area_mask_un.geojson', official: 'results/area_mask_official.geojson'};   // world box minus the displayed area: its rings are the area outlines
  var SOLID = {un: 'UN metro area', official: 'Metro area'}, DOT = {un: 'UN urban core', official: 'Municipality'};
  window.__areaOutlines = true;
  function group(area) { return (area === 'official' || area === 'municipality') ? 'official' : 'un'; }
  var ov = null, ovDiv = null, rings = {}, curGroup = 'un', raf = 0, lastStyle = '', restyleT = 0;

  function addLines(m, before) {
    Object.keys(GROUPS).forEach(function(g) {
      var id = '__ao_' + g;
      if (!m.getSource(id)) m.addSource(id, {type: 'geojson', data: GROUPS[g] + '?v=' + ((document.querySelector('meta[name="build-id"]') || {}).content || '')});   // MapLibre loads this URL itself (not through fetch): version it like every other data file
      var vis = {visibility: g === curGroup ? 'visible' : 'none'};
      function L(suf, name, layout, paint) { if (!m.getLayer(id + suf)) m.addLayer({id: id + suf, type: 'line', source: id, filter: ['==', ['get', 'name'], name], layout: Object.assign({}, layout, vis), paint: paint}, before); }
      L('_solid_halo', SOLID[g], {'line-join': 'round'}, {'line-color': '#ffffff', 'line-opacity': 0.8, 'line-width': 4});
      L('_solid', SOLID[g], {'line-join': 'round'}, {'line-color': '#1f2937', 'line-width': 1.6});
      L('_dot_halo', DOT[g], {'line-cap': 'round', 'line-join': 'round'}, {'line-color': '#ffffff', 'line-opacity': 0.85, 'line-width': 5});
      L('_dot', DOT[g], {'line-cap': 'round', 'line-join': 'round'}, {'line-color': '#000000', 'line-width': 2.6, 'line-dasharray': [0.01, 1.9]});
    });
  }
  function basemapStyle(map) {
    var st = map.getStyle(), src = {};
    function keep(id) { return id === 'basemap' || id.indexOf('__bm_') === 0; }
    Object.keys(st.sources || {}).forEach(function(k) { if (keep(k)) src[k] = st.sources[k]; });
    return {version: 8, glyphs: st.glyphs, sprite: st.sprite, sources: src, layers: (st.layers || []).filter(function(l) { return keep(l.id); })};
  }
  function loadRings(g) {
    if (rings[g]) return Promise.resolve(rings[g]);
    return fetch(MASKS[g]).then(function(r) { return r.json(); }).then(function(fc) {
      var out = [];
      fc.features.forEach(function(f) {
        var polys = f.geometry.type === 'Polygon' ? [f.geometry.coordinates] : f.geometry.coordinates;
        polys.forEach(function(poly) { poly.forEach(function(r) {
          if (Math.abs(r[0][0]) >= 179) return;   // the world box is replaced by the viewport rectangle
          var bb = [1e9, 1e9, -1e9, -1e9];
          r.forEach(function(p) { if (p[0] < bb[0]) bb[0] = p[0]; if (p[1] < bb[1]) bb[1] = p[1]; if (p[0] > bb[2]) bb[2] = p[0]; if (p[1] > bb[3]) bb[3] = p[1]; });
          out.push({pts: r, bb: bb});
        }); });
      });
      rings[g] = out; return out;
    });
  }
  function clip() {
    raf = 0;
    var map = window.__map, rs = rings[curGroup]; if (!ov || !map || !rs) return;
    var c = map.getCanvas(), W = c.clientWidth, H = c.clientHeight, b = map.getBounds();
    var w = b.getWest(), e = b.getEast(), s = b.getSouth(), n = b.getNorth(), d = 'M0 0H' + W + 'V' + H + 'H0Z';
    rs.forEach(function(r) {
      if (r.bb[2] < w || r.bb[0] > e || r.bb[3] < s || r.bb[1] > n) return;
      for (var i = 0; i < r.pts.length; i++) { var q = map.project(r.pts[i]); d += (i ? 'L' : 'M') + q.x.toFixed(1) + ' ' + q.y.toFixed(1); }
      d += 'Z';
    });
    ovDiv.style.clipPath = 'path(evenodd, "' + d + '")';
    ovDiv.style.visibility = 'visible';
  }
  function schedule() { if (!raf) raf = requestAnimationFrame(clip); }
  function sync() { var map = window.__map; if (!ov) return; ov.jumpTo({center: map.getCenter(), zoom: map.getZoom(), bearing: map.getBearing(), pitch: map.getPitch()}); schedule(); }
  function restyle() {
    var map = window.__map; if (!ov || !map) return;
    var st = basemapStyle(map), js = JSON.stringify(st);
    if (!st.layers.length || js === lastStyle) return;
    lastStyle = js; try { ov.setStyle(st); } catch (e) {}
  }
  function ensureOverlay(map) {
    if (ov) return true;
    var st = basemapStyle(map); if (!st.layers.length) return false;
    ovDiv = document.createElement('div');
    ovDiv.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;visibility:hidden;';
    var cont = map.getCanvasContainer(); cont.insertBefore(ovDiv, map.getCanvas().nextSibling);
    lastStyle = JSON.stringify(st);
    ov = new maplibregl.Map({container: ovDiv, style: st, center: map.getCenter(), zoom: map.getZoom(), bearing: map.getBearing(), pitch: map.getPitch(), interactive: false, attributionControl: false, fadeDuration: 0});
    ov.on('style.load', function() { try { addLines(ov); } catch (e) {} });
    map.on('move', sync); map.on('resize', function() { ov.resize(); sync(); });
    map.on('styledata', function() { clearTimeout(restyleT); restyleT = setTimeout(restyle, 250); });
    ['basemapSelect', 'basemapOpacitySlider', 'basemapBwCheckbox'].forEach(function(id) {
      var el = document.getElementById(id); if (el) { el.addEventListener('change', function() { setTimeout(restyle, 400); }); el.addEventListener('input', function() { setTimeout(restyle, 100); }); }
    });
    return true;
  }
  function apply(area) {
    var map = window.__map; if (!map || !map.getLayer) return;
    curGroup = group(area);
    [map, ov].forEach(function(m) {
      if (!m || !m.getLayer) return;
      Object.keys(GROUPS).forEach(function(g) {
        ['_solid_halo', '_solid', '_dot_halo', '_dot'].forEach(function(suf) {
          var id = '__ao_' + g + suf; if (m.getLayer(id)) { m.setLayoutProperty(id, 'visibility', g === curGroup ? 'visible' : 'none'); m.moveLayer(id); }
        });
      });
    });
    if (ensureOverlay(map)) loadRings(curGroup).then(function() { sync(); });
  }
  // The Gipuzkoa study is named "Gipuzkoa" only while the official (province) area is shown, "Donostia" in the other three areas (place-rank lists and titles).
  var RENAME = {official: ['Donostia', 'Gipuzkoa'], un: ['Gipuzkoa', 'Donostia']};
  function renameCity() {
    var r = RENAME[window.__areaGroupNow || 'un']; if (!r || !document.body) return;
    var w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null), n, hits = [];
    while ((n = w.nextNode())) { if (n.nodeValue.indexOf(r[0]) >= 0 && !/^\s*Donostia \(|^\s*Gipuzkoa \(/.test(n.nodeValue)) hits.push(n); }
    hits.forEach(function(t) { t.nodeValue = t.nodeValue.split(r[0]).join(r[1]); });
  }
  var __renaming = false, __mo = new MutationObserver(function() { if (__renaming) return; __renaming = true; try { renameCity(); } finally { __renaming = false; } });
  var __gip = /gipuzkoa|donostia/i.test(location.pathname + ' ' + document.title);
  if (__gip) try { __mo.observe(document.documentElement, {childList: true, subtree: true, characterData: true}); } catch (e) {}
  window.__applyAreaOutlines = function(area) { window.__areaGroupNow = group(area); apply(area); if (__gip) renameCity(); };
  var n = 0, t = setInterval(function() {
    var map = window.__map;
    if (map && map.getStyle && map.isStyleLoaded && map.isStyleLoaded()) {
      try { addLines(map, map.getLayer('__stops_cluster') ? '__stops_cluster' : undefined); window.__applyAreaOutlines(window.__currentArea || 'metro'); } catch (e) {}
      if (map.getLayer('__ao_un_dot') && ov) clearInterval(t);
    }
    if (++n > 240) clearInterval(t);
  }, 500);
})();
"""


def _load_aoi_gdf(city_dir: Path, config: "CityConfig") -> gpd.GeoDataFrame:
    """Loads `aoi.gpkg` as-is -- the real AOI geometry, no convex-hull
    widening (removed per explicit user request 2026-08-30; previously
    Beersheba's concave, gappy union of ~25 localities was widened to its
    convex hull via `config.aoi_convex_hull`, now deleted)."""
    return gpd.read_file(city_dir / "aoi.gpkg")


def _add_share_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add physical ratio columns for every `SHARE_COLUMNS`/`RATE_SOURCE_COLUMNS` pair present on `gdf`.

    The census join (`_join_polygon_stats`) only carries absolute counts --
    ratios were previously only computed ad hoc inside
    `_run_regressions_and_anovas`, invisible to the map's own field picker.
    Safe to call after any resample: it's a plain elementwise ratio of two
    already-resampled absolute columns, computed fresh each time, never
    itself resampled. `RATE_SOURCE_COLUMNS` needs no division -- just a
    rescale of an already-published rate onto the same 0-1 convention.
    """
    for share_name, candidates in SHARE_COLUMNS.items():
        for num_col, den_col in candidates:
            if num_col in gdf.columns and den_col in gdf.columns and _share_pair_ok(_pd_total(gdf), num_col, den_col):
                gdf[share_name] = (gdf[num_col] / gdf[den_col].replace(0, np.nan)).to_numpy(dtype=float)
                gdf[share_name] = gdf[share_name].clip(0.0, 1.0)   # a part never exceeds its whole (mismatched sources / suppression noise)
                break
    for share_name, candidates in RATE_SOURCE_COLUMNS.items():
        for src_col, divisor in candidates:
            if src_col in gdf.columns:
                gdf[share_name] = (gdf[src_col].astype(float) / divisor).to_numpy(dtype=float)
                # Drop the raw source column once its rescaled alias exists:
                # both are numeric and both match `is_relative_field`'s
                # "rate"/"share" markers, so leaving both on the grid made the
                # map's ANOVA/regression pick them up as two separate
                # variables carrying identical information (e.g.
                # `cbs_selfEmployedRate` and `cbs_self_employed_rate`).
                if src_col != share_name:
                    gdf = gdf.drop(columns=[src_col])
                break
    return gdf


def _share_column_sources(gdf) -> dict[str, tuple[str, str]]:
    """`share_col_name -> (numerator_col, denominator_col)` for every real `SHARE_COLUMNS` entry present on `gdf`.

    Re-derives (rather than records at compute time) which numerator/
    denominator candidate `_add_share_columns` actually used for each share
    column already materialized on `gdf` -- cheap (just re-checking column
    presence, no recomputation) and avoids threading extra state through
    every `_add_share_columns` call site. Used to fold a share column's
    metadata row into its numerator's row on the map's "Metadata" tab (see
    `transitlos.map.build._column_metadata_rows`) instead of showing them
    as two disconnected rows -- 2026-09-01 explicit user request ("men and
    total, average, share, so we do not need a separate share column").
    Only covers `SHARE_COLUMNS` (numerator/denominator pairs); a
    `RATE_SOURCE_COLUMNS` rescaled rate has no absolute numerator column of
    its own left on `gdf` (its raw source column is dropped once rescaled --
    see `_add_share_columns`), so those keep their own standalone row.
    """
    out: dict[str, tuple[str, str]] = {}
    for share_name, candidates in SHARE_COLUMNS.items():
        if share_name not in gdf.columns:
            continue
        for num_col, den_col in candidates:
            if num_col in gdf.columns and den_col in gdf.columns and _share_pair_ok(_pd_total(gdf), num_col, den_col):
                out[share_name] = (num_col, den_col)
                break
    return out


def _write_census_metadata_json(
    city_dir: Path,
    stats_grid,
    h3_by_resolution: dict,
    census_by_level: Optional[dict],
    country: Optional[str],
    share_source_map: dict,
) -> Path:
    """Write `results/census_metadata.json`: one entry per real census/WorldPop column.

    2026-09-22, explicit user request: every transit study should produce a
    standalone `census_metadata.json` (name, source, country, total, mean,
    share, share_column, native_levels, resampled_levels,
    resampling_up_method, resampling_down_method, description) alongside
    the same info already shown on the map's own "Metadata" tab -- reuses
    `transitlos.map.build._column_metadata_rows` (the exact function that
    builds that tab) so the file and the tab can never drift apart; this is
    a second, cheap call over already-in-memory data (no recomputation of
    anything expensive), not a separate code path.
    """
    from transitlos.map.build import _column_metadata_rows

    rows = _column_metadata_rows(
        stats_grid, share_source_map=share_source_map,
        h3_by_resolution=h3_by_resolution, census_by_level=census_by_level, country=country,
    )
    out_path = Path(city_dir) / "results" / "census_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    return out_path


# --------------------------------------------------------------------------
# Generic "population + jobs" rule.
#
# Any city whose census join happens to carry a *jobs / workplaces located in
# this geometry* count gets two derived columns; a city without such a column
# simply doesn't (nothing below is Boston- or US-specific, it is keyed purely
# on a column existing). For the US that column comes from LEHD LODES WAC
# (`C000`, total jobs by workplace location -- see `_join_jobs`); another
# country's study only has to land its own equivalent count under one of the
# names in `JOBS_COLUMN_CANDIDATES` (or a column ending in `_jobs_total`) to
# get the same two derived columns for free.
#
#   `pop_jobs_total`   -- absolute: residents + jobs in the cell. Named
#                         *without* a `_density`/`_share`/`_rate` substring so
#                         the map's naming-convention classifier treats it as
#                         an absolute (distribution chart only, never
#                         regression/ANOVA/circle-size/opacity).
#   `pop_jobs_density` -- relative: the same total per km2, i.e. "activity
#                         density". Named with `_density` so the same
#                         classifier treats it as relative, and it is what the
#                         map defaults to (see `_map_field_rank`) and what the
#                         development overlay is computed from (see
#                         `development_density_column`).
# --------------------------------------------------------------------------

POP_JOBS_TOTAL_COLUMN = "pop_jobs_total"
GLOBAL_WORKPLACES_COLUMN = "workplaceJobs"                 # pycensus global schema name of the workplaces count
GLOBAL_POP_AND_WORKPLACES_COLUMN = "populationAndJobs"     # pycensus global schema name of residents + workplaces
POP_JOBS_DENSITY_COLUMN = "pop_jobs_density"

# `census_residents` / `worldpop_residents` (2026-09-23, explicit user
# request): every city should carry two clearly-named, unambiguous
# population columns -- the real census-source resident count
# (`census_residents`, absent only for Shanghai, which has no real census
# join at all) and WorldPop's own modeled count (`worldpop_residents`,
# present everywhere). Both are materialized once, at the very end of grid
# preparation (`_finalize_population_columns`/`_finalize_population_columns_polars`,
# called from `prepare_grid_for_map`/`_prepare_grid_polars` and from
# `_census_geometries_with_score`), from whatever `_census_population_column`/
# `_worldpop_population_column` resolve to at that point -- see those two
# functions' docstrings for exactly which source column each name means per
# city/path. The map-facing `population` column is then (re)derived fresh
# from these two per `WORLDPOP_ONLY_POPULATION_CITIES`/jobs-presence (see
# `_finalize_population_columns`), so `population`'s meaning is never left
# to whichever raw source happened to land in that name first.
CENSUS_RESIDENTS_COLUMN = "census_residents"
WORLDPOP_RESIDENTS_COLUMN = "worldpop_residents"

# Cities where the map-facing `population` column should be a duplicate of
# `worldpop_residents` rather than `census_residents`, even though a real
# census join exists (Beersheba, Andorra) or doesn't (Shanghai, which would
# fall back to WorldPop here regardless -- included for clarity/robustness,
# not because it changes behavior). Every other `uses_census` city instead
# gets `population = census_residents` (Boston/San Francisco additionally
# add their jobs count -- see `_finalize_population_columns`), and every
# non-`uses_census`, non-Shanghai-like WorldPop-only city (none exist today,
# but the fallback below covers one anyway) gets `worldpop_residents` too.
WORLDPOP_ONLY_POPULATION_CITIES = frozenset({"beerseba", "shanghai", "andorra"})

# Checked in order; the first one present on the grid wins. `lodes_wac_*` is
# what `pycensus.census.countries.usa.lodes_wac` produces; the rest are the plain names a non-US
# study is likely to bring its own workplace/jobs count in under.
JOBS_COLUMN_CANDIDATES: tuple[str, ...] = (
    "lodes_wac_jobsTotal",
    "jobs_total",
    "total_jobs",
    "jobs",
    "workplaces",
)


def jobs_column(gdf) -> Optional[str]:
    """Return the grid's jobs/workplace-count column, or `None` if it has none.

    Generic on purpose: any study whose census join lands a total-jobs count
    under one of `JOBS_COLUMN_CANDIDATES` -- or under any column ending in
    `_jobs_total` -- gets `pop_jobs_total`/`pop_jobs_density` automatically,
    and any study without one keeps behaving exactly as before.
    """
    for col in JOBS_COLUMN_CANDIDATES:
        if col in gdf.columns:
            return col
    for col in gdf.columns:
        if isinstance(col, str) and (col.endswith("_jobs_total") or col.endswith("JobsTotal")):
            return col
    return None


# Every wired country's real `pycensus.census.countries.<x>.<source>.schema.SCHEMA
# (prefix=...)` -- the single source of truth for "which column prefixes are
# real census data" used across this module (resampling's `sum_cols`,
# population-column detection, and anywhere else that needs "every census
# column, whichever country this study is"). Kept as ONE tuple instead of
# separately hardcoded per use site: a stale/partial copy in just one of
# those use sites is exactly the bug this fixes (see `build_h3_by_resolution`'s
# `census_cols` docstring note) -- new countries only need adding here once.
CENSUS_COLUMN_PREFIXES: tuple[str, ...] = (
    "acs5_", "acs3_", "acs1_", "dhc_", "lodes_wac_", "lodes_rac_", "lodes_",
    "inegi_", "ine_cl_", "ine_", "cbs_", "eustat_", "statcan_", "moi_",
    "estadisticaad_", "destatis_", "ba_", "usrc_", "ushs_",
    "cbscommute_", "inegimov_",   # Israel census mode to work (sub-quarter / locality), INEGI Censo 2020 movilidad (municipality)
    # `_add_worldpop_gapfill`'s output columns (`worldpop_malePopulation`,
    # ...) -- included here so `_census_columns`'s stale-column cleanup
    # (used by `refresh_census_only`, see its own comment) drops and
    # re-derives them fresh on every re-run too, the same as every real
    # census-source prefix above.
    "worldpop_",
    "gba_",   # GlobalBuildingAtlas building metrics / built-surface population (`_gba_cell_population`, `_add_gba_columns`)
)

# `pyCensus/src/pycensus/census/global_schema.json`'s `feature_name` set,
# loaded once (never hardcoded -- see `AGENT_GUIDE_NEW_CITY_OR_COUNTRY.md`'s
# anti-fabrication rule) and memoized here. Resolved via `pycensus.__file__`
# rather than a path relative to this repo, so it works the same whether
# pyCensus is a normal site-packages install or (as in this dev checkout) an
# editable install from a sibling repo -- the same convention every other
# `pycensus.census.countries.<x>.<source>` import in this module already relies on
# implicitly (they all resolve through whatever `pycensus` is on `sys.path`).
_GLOBAL_SCHEMA_FEATURE_NAMES_CACHE: Optional[frozenset[str]] = None


def _global_schema_feature_names() -> frozenset[str]:
    """The canonical `feature_name` set every wired country's census join may rename onto."""
    global _GLOBAL_SCHEMA_FEATURE_NAMES_CACHE
    if _GLOBAL_SCHEMA_FEATURE_NAMES_CACHE is None:
        from pycensus.canonical import GLOBAL_SCHEMA_PATH as schema_path
        with open(schema_path, encoding="utf-8") as f:
            entries = json.load(f)
        _GLOBAL_SCHEMA_FEATURE_NAMES_CACHE = frozenset(entry["feature_name"] for entry in entries)
    return _GLOBAL_SCHEMA_FEATURE_NAMES_CACHE


# `population` is deliberately excluded from the rename, even though it is a
# real `global_schema.json` feature_name -- unlike every other canonical
# field, it collides with a column that already means something else on
# `h3_grid`: `_add_h3_grid`/`population_and_access_to_h3` land a
# WorldPop-derived `population` column *before* any census join runs (see
# `run_city_study`), and that is the figure `pop_density`, `filter_populated`
# and every apportionment weight in this module are built from. A census
# source's own population count is a different figure kept deliberately
# prefixed (`acs5_population`/`inegi_population`/...) so it stays usable as a
# `SHARE_COLUMNS` denominator (e.g. `acs_worker_rate`) without silently
# swapping which "population" the rest of the pipeline computes against.
# Renaming it only on `_census_geometries_with_score`'s census-polygon path
# (which has no pre-existing `population` to collide with) while leaving it
# prefixed on the `h3_grid` path would make the same `SHARE_COLUMNS` entry
# resolve under two different names depending on which path built the grid --
# so this field is excluded from the rename on *both* paths for consistency,
# not just where a literal collision happens to occur.
_RENAME_EXCLUDED_CANONICAL_NAMES = frozenset({"population"})


def _restore_population_prefix(gdf: gpd.GeoDataFrame, prefix: str) -> gpd.GeoDataFrame:
    """Re-prefix a country census loader's `population` column right after loading it.

    pyCensus's `apply_canonical_names` (added during its 2026-09 schema
    redesign) now runs unconditionally inside every country loader's own
    `load()`, renaming that source's population column down to the bare
    canonical name `"population"` before this pipeline ever sees it --
    directly at odds with the invariant documented just above
    (`_RENAME_EXCLUDED_CANONICAL_NAMES`): this file reserves bare
    `"population"` EXCLUSIVELY for WorldPop's own count, and expects every
    census source's own population under its source-prefixed name (e.g.
    `dhc_population`, `inegi_population`) so `SHARE_COLUMNS` denominators,
    `_interpolate_acs_to_dhc_blocks`, and `DHC_KEEP_COLUMNS`/-equivalents
    keep resolving it. Discovered live 2026-09-29 (`_interpolate_acs_to_
    dhc_blocks` crashed with `KeyError: 'dhc_population'` once
    `dhc.load()`'s real output was actually exercised end-to-end).

    Call this immediately after EVERY country census loader's `.load(...)`
    call, before the result reaches any other logic in this file -- fixing
    it at this one boundary point per source, rather than renaming the
    ~70 downstream `<prefix>_population` references, keeps every existing
    consumer correct without touching them.

    Args:
        gdf: A country loader's raw `load()` return value.
        prefix: That source's column prefix (e.g. `"dhc"`, `"inegi"`).

    Returns:
        `gdf` with `population` renamed to `f"{prefix}_population"`, or
        `gdf` unchanged if it has no bare `population` column (e.g.
        already prefixed, or empty) or already has the prefixed name.
    """
    if "population" in gdf.columns and f"{prefix}_population" not in gdf.columns:
        gdf = gdf.rename(columns={"population": f"{prefix}_population"})
    return gdf


def _rename_canonical_columns(gdf):
    """Rename every prefixed column whose de-prefixed name is a real `global_schema.json` feature to that bare name.

    This study never joins more than one country's census data at once (a
    per-city, per-run invariant confirmed for this pipeline), so a rename to
    a bare canonical name can't collide with *another country's* same-named
    column. See `_RENAME_EXCLUDED_CANONICAL_NAMES` for the one field
    (`population`) excluded from this rename on purpose. Any other would-be
    collision with an existing column is skipped (with a printed warning)
    rather than applied, as a defensive backstop -- not expected to trigger
    given today's column set, but data is never silently overwritten if it
    does.

    A column whose de-prefixed name has no EXACT match in
    `global_schema.json` (e.g. `inegi_populationCatholic`,
    `inegi_dwellingsInternet`) keeps its source-prefixed name unchanged --
    stripping it would falsely imply cross-country comparability that
    doesn't exist. Idempotent: a column with none of `CENSUS_COLUMN_PREFIXES`
    as a prefix (including one already renamed to bare) is left untouched.

    Call this exactly once, immediately after a country's raw census join
    lands its prefixed columns -- both `_join_census`/`_join_race`/
    `_join_jobs`'s h3-grid call sites, and `_census_geometries_with_score`'s
    separate per-level census-polygon path -- and before anything downstream
    reads those column names.
    """
    canonical = _global_schema_feature_names() - _RENAME_EXCLUDED_CANONICAL_NAMES
    rename_map: dict[str, str] = {}
    for col in gdf.columns:
        if not isinstance(col, str):
            continue
        prefix = next((p for p in CENSUS_COLUMN_PREFIXES if col.startswith(p)), None)
        if prefix is None:
            continue
        bare = col[len(prefix):]
        if bare not in canonical and not bare.startswith("elec_"):   # election columns are comparable across countries: always bare
            continue
        if bare in gdf.columns or bare in rename_map.values():
            print(
                f"[pipeline] _rename_canonical_columns: keeping {col!r} prefixed -- "
                f"{bare!r} already exists on this grid (expected for 'population': the grid's "
                "own WorldPop-derived population predates the census join)"
            )
            continue
        rename_map[col] = bare
    if rename_map:
        from transitlos.map import build as _mb

        for _old, _bare in rename_map.items():
            _mb.BARE_COLUMN_SOURCE_PREFIX[_bare] = _old[: len(_old) - len(_bare)]
        if _mb.BARE_COLUMN_SOURCE_FILE:
            Path(_mb.BARE_COLUMN_SOURCE_FILE).parent.mkdir(parents=True, exist_ok=True)
            Path(_mb.BARE_COLUMN_SOURCE_FILE).write_text(json.dumps(_mb.BARE_COLUMN_SOURCE_PREFIX))
    gdf = gdf.rename(columns=rename_map) if rename_map else gdf
    # A census column named differently from its global-schema twin (e.g. `acs5_povertyBelow100` / `povertyPopulation`) keeps its
    # local name AND gets the global-named copy; a column named the same exists once (the rename above).
    from pycensus.global_equivalents import add_global_equivalents

    return _sanitize_census_columns(add_global_equivalents(gdf))


DEVELOPMENT_H3_RESOLUTION = 8


def force_development_resolution(h3_by_resolution: dict, resolution: int = DEVELOPMENT_H3_RESOLUTION) -> None:
    """Development view (more housing / more transit) at ONE H3 resolution, 8: `equity_flag` is kept only on that resolution's grid (the map
    builder picks the resolution that carries it), so the overlay is one city-wide regression on res-8 cells. In place; the stored parquets
    keep the flags they have. No-op when the city has no grid at `resolution`."""
    if resolution not in h3_by_resolution or "equity_flag" not in h3_by_resolution[resolution].columns:
        return
    for res, g in list(h3_by_resolution.items()):
        if res != resolution and "equity_flag" in g.columns:
            h3_by_resolution[res] = g.drop(columns="equity_flag")


def _census_total_from_sexes(df):
    """Real census resident count for grids whose census source has no total-population column: male + female (both census columns).
    Returns None unless both exist and the sum varies against WorldPop (a copy of WorldPop is not a census)."""
    import numpy as np

    if df is None or "malePopulation" not in df.columns or "femalePopulation" not in df.columns:
        return None
    total = df["malePopulation"].to_numpy(dtype=float) + df["femalePopulation"].to_numpy(dtype=float)
    return total if np.isfinite(total).any() and np.nansum(total) > 0 else None


def _use_census_sexes_total(frames):
    """Where `census_residents` is missing or just a copy of `worldpop_residents`, replace it with male+female and recompute the four
    WorldPop-vs-census columns (overestimation = max(0, worldpop - census), underestimation = max(0, census - worldpop), shares)."""
    import numpy as np

    new = ("population_overestimation_vs_census", "population_underestimation_vs_census", "worldpop_overestimation_share", "worldpop_underestimation_share")
    for k, df in list(frames.items()):
        if df is None or "worldpop_residents" not in df.columns:
            continue
        wp = df["worldpop_residents"].to_numpy(dtype=float)
        cur = df["census_residents"].to_numpy(dtype=float) if "census_residents" in df.columns else None
        if cur is not None and not np.allclose(np.nan_to_num(cur), np.nan_to_num(wp)):
            continue
        census = _census_total_from_sexes(df)
        if census is None:
            continue
        out = df.copy()
        out["census_residents"] = census
        census = np.nan_to_num(np.asarray(census, dtype=float))   # no census value = census 0 for the difference columns
        over, under = np.maximum(0.0, wp - census), np.maximum(0.0, census - wp)
        out[new[0]], out[new[1]] = over, under
        out[new[2]] = np.where(wp != 0, over / np.where(wp != 0, wp, 1.0), 0.0)
        out[new[3]] = np.where(census != 0, under / np.where(census != 0, census, 1.0), 0.0)
        frames[k] = out


def _fill_worldpop_differences(h3_by_resolution, stats_grid):
    """Give every h3 grid the four WorldPop-vs-census columns: grids stored with only the finest resolution carrying them get them from
    the finest grid's `census_residents` summed to their own cells (each level then compares its own two populations)."""
    import h3
    import numpy as np
    import pandas as pd

    new = ("population_overestimation_vs_census", "population_underestimation_vs_census", "worldpop_overestimation_share", "worldpop_underestimation_share")
    src = next((f for f in h3_by_resolution.values() if f is not None and "census_residents" in f.columns and "worldpop_residents" in f.columns), None)
    if src is None:
        return stats_grid
    src_res = h3.get_resolution(str(src["h3_cell"].iloc[0]))

    def fill(df):
        if df is None or "worldpop_residents" not in df.columns or "worldpop_overestimation_share" in df.columns or "census_residents" in df.columns and "population_overestimation_vs_census" in df.columns:
            return df
        res = h3.get_resolution(str(df["h3_cell"].iloc[0]))
        if res > src_res:
            return df
        parent = pd.Series([h3.cell_to_parent(str(c), res) for c in src["h3_cell"]], index=src.index)
        # every cell is compared whole: a sub-cell without a census count counts as census 0 (WorldPop over-estimates there by its full count)
        cen = src["census_residents"].astype(float).fillna(0.0).groupby(parent.to_numpy()).sum()
        out = df.copy()
        census = np.nan_to_num(out["h3_cell"].astype(str).map(cen).to_numpy(dtype=float))
        wp = np.nan_to_num(out["worldpop_residents"].to_numpy(dtype=float))
        out["census_residents"] = census
        over, under = np.maximum(0.0, wp - census), np.maximum(0.0, census - wp)
        out[new[0]], out[new[1]] = over, under
        out[new[2]] = np.where(wp != 0, over / np.where(wp != 0, wp, 1.0), 0.0)
        out[new[3]] = np.where(census != 0, under / np.where(census != 0, census, 1.0), 0.0)
        return out

    for k, v in list(h3_by_resolution.items()):
        h3_by_resolution[k] = fill(v)
    return fill(stats_grid)


def _scale_worldpop_to_census(city_dir, h3_by_resolution, census_by_level, stats_grid):
    """WorldPop scaled to the census with ONE factor for the whole city (census total / WorldPop total over the finest grid's cells that
    have a census count); individual cells keep their own WorldPop-vs-census difference. Grids whose `worldpop_residents` was only a copy of
    the census (no raw WorldPop kept) get the raw WorldPop count re-read from the city's WorldPop raster. Then the four difference columns
    are recomputed on every grid: overestimation = max(0, worldpop - census), underestimation = max(0, census - worldpop), shares /worldpop
    and /census."""
    import glob

    import h3
    import numpy as np

    new = ("population_overestimation_vs_census", "population_underestimation_vs_census", "worldpop_overestimation_share", "worldpop_underestimation_share")
    src = next((f for r, f in sorted(h3_by_resolution.items(), key=lambda kv: -int(kv[0])) if f is not None and "census_residents" in f.columns and "worldpop_residents" in f.columns), None)
    if src is None:
        return stats_grid
    tifs = sorted(glob.glob(str(Path(city_dir) / "worldpop_demographic" / "clipped_*_pop_2025_CN_100m*.tif"))) or sorted(glob.glob(str(Path(city_dir) / "worldpop_demographic" / "*_pop_2025_CN_100m*.tif")))
    raw_cache = {}

    def degenerate(df):
        c, w = df["census_residents"].to_numpy(dtype=float), df["worldpop_residents"].to_numpy(dtype=float)
        return bool(np.allclose(np.nan_to_num(c), np.nan_to_num(w)))

    def raw_worldpop(df):
        res = h3.get_resolution(str(df["h3_cell"].iloc[0]))
        if res not in raw_cache:
            from pycensus.census.worldwide.worldpop.h3 import worldpop_raster_to_h3_fast

            layer = worldpop_raster_to_h3_fast(tifs[0], resolution=res, value_col="__wp_raw").to_pandas().set_index("h3_cell")["__wp_raw"]
            raw_cache[res] = layer
        return df["h3_cell"].astype(str).map(raw_cache[res]).to_numpy(dtype=float)

    # the raw WorldPop count of every grid, then the single city-wide factor
    raw = {}
    frames = dict(h3_by_resolution)
    frames["__stats"] = stats_grid
    for k, df in frames.items():
        if df is None or "worldpop_residents" not in df.columns or "h3_cell" not in df.columns:
            continue
        if "census_residents" in df.columns and degenerate(df):
            if not tifs:
                continue
            raw[k] = raw_worldpop(df)
        else:
            raw[k] = df["worldpop_residents"].to_numpy(dtype=float)
    key0 = next((k for k, f in frames.items() if f is src), None)
    if key0 not in raw:
        return stats_grid
    cen0, wp0 = src["census_residents"].to_numpy(dtype=float), raw[key0]
    ok = np.isfinite(cen0) & np.isfinite(wp0) & (wp0 > 0)
    if not ok.any() or wp0[ok].sum() <= 0:
        return stats_grid
    factor = float(cen0[ok].sum() / wp0[ok].sum())
    print(f"[pipeline] WorldPop scaled to the census total with one factor: x{factor:.4f}")

    def recompute(df, wp_raw):
        out = df.copy()
        wp = wp_raw * factor
        # a cell the census has no value for counts as census 0: the four difference columns always have a value (WorldPop over-estimates there by its whole count)
        census = np.nan_to_num(out["census_residents"].to_numpy(dtype=float)) if "census_residents" in out.columns else np.zeros(len(out))
        out["worldpop_residents"] = wp
        over, under = np.maximum(0.0, wp - census), np.maximum(0.0, census - wp)
        out[new[0]], out[new[1]] = over, under
        out[new[2]] = np.where(wp != 0, over / np.where(wp != 0, wp, 1.0), 0.0)
        out[new[3]] = np.where(census != 0, under / np.where(census != 0, census, 1.0), 0.0)
        return out

    for k, wp_raw in raw.items():
        if k == "__stats":
            stats_grid = recompute(stats_grid, wp_raw)
        else:
            h3_by_resolution[k] = recompute(h3_by_resolution[k], wp_raw)
    for k, df in list((census_by_level or {}).items()):   # census polygons: their WorldPop sum is raw, the census count is real
        if df is not None and "worldpop_residents" in df.columns and "census_residents" in df.columns:
            census_by_level[k] = recompute(df, df["worldpop_residents"].to_numpy(dtype=float))
    return stats_grid


def _zero_fill_worldpop_differences(frames):
    """Every cell gets a value for the four WorldPop-vs-census columns (same number of points as the rest of the stats):
    where no census count covers a cell there is no measured difference, so 0."""
    for k, df in list(frames.items()):
        if df is None or "worldpop_residents" not in df.columns:
            continue
        cols = [c for c in ("population_overestimation_vs_census", "population_underestimation_vs_census", "worldpop_overestimation_share", "worldpop_underestimation_share") if c in df.columns]
        if cols:
            out = df.copy()
            out[cols] = out[cols].fillna(0.0)
            frames[k] = out


GBA_VS_CENSUS_COLUMNS = ("gba_population_overestimation_vs_census", "gba_population_underestimation_vs_census",
                         "gba_overestimation_share", "gba_underestimation_share")


def _add_gba_vs_census(df):
    """The built-surface population (`GBA_POPULATION_COLUMN`) against the census, like WorldPop's four columns: overestimation =
    max(0, built-surface - census), underestimation = max(0, census - built-surface), shares / built-surface and / census (0 where the
    denominator is 0). A no-op without both columns."""
    if df is None or GBA_POPULATION_COLUMN not in df.columns:
        return df
    census_col = CENSUS_RESIDENTS_COLUMN if CENSUS_RESIDENTS_COLUMN in df.columns else _census_population_column(df)
    if census_col is None or census_col == "population":
        return df
    g = np.nan_to_num(pd.to_numeric(df[GBA_POPULATION_COLUMN], errors="coerce").to_numpy(dtype=float))
    c = np.nan_to_num(pd.to_numeric(df[census_col], errors="coerce").to_numpy(dtype=float))
    over, under = np.maximum(0.0, g - c), np.maximum(0.0, c - g)
    df = df.copy()
    df[GBA_VS_CENSUS_COLUMNS[0]], df[GBA_VS_CENSUS_COLUMNS[1]] = over, under
    df[GBA_VS_CENSUS_COLUMNS[2]] = np.where(g != 0, over / np.where(g != 0, g, 1.0), 0.0)
    df[GBA_VS_CENSUS_COLUMNS[3]] = np.where(c != 0, under / np.where(c != 0, c, 1.0), 0.0)
    return df


def filter_map_columns(city_dir, h3_by_resolution, census_by_level, stats_grid):
    """Map-only column filter: drops historic variants (older elections, older housing / rent / price periods) from what is handed to
    `build_city_map`; the stored parquet files keep everything. Rule + data-driven detection: `pycensus.latest_columns`.
    Writes `<city>/results/map_columns_dropped.json`; returns the (filtered) stats grid."""
    from pycensus.latest_columns import filter_map_frames, latest_column_filter

    from pycensus.global_equivalents import add_global_equivalents

    for frames in (h3_by_resolution, census_by_level or {}):
        for k, v in list(frames.items()):
            frames[k] = _add_gba_vs_census(v)
    stats_grid = _add_gba_vs_census(stats_grid)
    groups = {"h3": h3_by_resolution}
    if census_by_level:
        groups["census"] = census_by_level
    for frames in groups.values():   # grids built before the global names existed (map-only rebuild from the checkpoint) get them here
        for k, v in list(frames.items()):
            frames[k] = add_global_equivalents(v)
    if stats_grid is not None:
        stats_grid = add_global_equivalents(stats_grid)
    stats_grid = _fill_worldpop_differences(h3_by_resolution, stats_grid)
    stats_grid = _scale_worldpop_to_census(city_dir, h3_by_resolution, census_by_level, stats_grid)
    # Hamburg: the census is bare `population` until the steps above name it `census_residents`, so the first `_add_gba_vs_census` pass
    # skipped it: compare again wherever the columns are still missing
    for frames in (h3_by_resolution, census_by_level or {}):
        for k, v in list(frames.items()):
            if v is not None and GBA_VS_CENSUS_COLUMNS[0] not in v.columns:
                frames[k] = _add_gba_vs_census(v)
    if stats_grid is not None and GBA_VS_CENSUS_COLUMNS[0] not in stats_grid.columns:
        stats_grid = _add_gba_vs_census(stats_grid)
    _zero_fill_worldpop_differences(h3_by_resolution)
    if census_by_level:
        _zero_fill_worldpop_differences(census_by_level)
    if stats_grid is not None:
        _sg0 = {"g": stats_grid}
        _zero_fill_worldpop_differences(_sg0)
        stats_grid = _sg0["g"]
    _OLD_SHARE = {"population_overestimation_vs_census_share": "worldpop_overestimation_share", "population_underestimation_vs_census_share": "worldpop_underestimation_share"}
    for frames in groups.values():   # grids stored before the shares got their final names
        for k, v in list(frames.items()):
            if any(c in v.columns for c in _OLD_SHARE):
                frames[k] = v.drop(columns=[o for o, n in _OLD_SHARE.items() if o in v.columns and n in v.columns]).rename(columns=_OLD_SHARE)
    if stats_grid is not None and any(c in stats_grid.columns for c in _OLD_SHARE):
        stats_grid = stats_grid.drop(columns=[o for o, n in _OLD_SHARE.items() if o in stats_grid.columns and n in stats_grid.columns]).rename(columns=_OLD_SHARE)
    from pycensus.latest_columns import minor_party_columns

    _all = [df for fr in groups.values() for df in fr.values() if hasattr(df, "columns")]
    _party_frames = [df for df in _all if any("_party_" in str(c) for c in df.columns)]
    _minor = minor_party_columns(max(_party_frames, key=len)) if _party_frames else set()
    if stats_grid is not None and _minor:
        stats_grid = stats_grid.drop(columns=[c for c in stats_grid.columns if c in _minor])
    filter_map_frames(groups, report_path=Path(city_dir) / "results" / "map_columns_dropped.json")
    # 2026-11-13 (user request: only the latest election of each level, nothing repeated): the period-free election names
    # (elec_<scope>_<metric>, global twins of the latest dated columns) and the MEDSL-style `<year>_president|senate_*` copies are hidden
    # on the city maps whenever the dated elec_<scope>_<year>_* column is there; the stored grids keep them.
    def _elec_dups(cols):
        import re
        cols = set(map(str, cols)); out = set()
        for c in cols:
            m = re.match(r"^elec_(national|state|municipal)_((?:left|right|other|democrat|republican)_share|turnout_rate|total_votes)$", c)
            if m and any(re.match(rf"^elec_{m.group(1)}_\d{{4}}(_party)?_{m.group(2)}$", d) for d in cols):
                out.add(c)
            m = re.match(r"^(\d{4})_(president|senate)_(.+)$", c)
            if m and any(d.startswith(f"elec_{'national' if m.group(2) == 'president' else 'state'}_{m.group(1)}_") for d in cols):
                out.add(c)
        return out
    for frames in groups.values():
        for k, v in list(frames.items()):
            if hasattr(v, "columns"):
                dd = _elec_dups(v.columns)
                if dd:
                    frames[k] = v.drop(columns=list(dd))
    if stats_grid is not None:
        dd = _elec_dups(stats_grid.columns)
        if dd:
            stats_grid = stats_grid.drop(columns=list(dd))
    force_development_resolution(h3_by_resolution)
    if stats_grid is not None:
        from pycensus.year_built import drop_intermediate_bands

        _, dropped = latest_column_filter(stats_grid.columns)
        stats_grid = drop_intermediate_bands(stats_grid)
        if dropped:
            stats_grid = stats_grid.drop(columns=[c for c in dropped if c in stats_grid.columns])
    return stats_grid


def _register_relative_fields(city_dir) -> None:
    """Tell the map builder which census columns are intensive (prices, rents, incomes ... with no rate-like name), so they are never offered
    as counts (circle size, distribution). The census step records them in `results/rate_columns.json`; map-only rebuilds read that file."""
    from transitlos.map import build as _map_build

    f = Path(city_dir) / "results" / "rate_columns.json"
    try:   # name patterns of the columns the loaders add (valid for grids built before the registry file existed)
        import pyarrow.parquet as pq

        names = pq.read_schema(Path(city_dir) / "results" / "metro" / "h3_grid.parquet").names
    except Exception:
        names = []
    for c in names:
        low = c.lower()
        if (low.startswith("elec_") and not low.endswith("_total_votes")) or low.startswith("ushs_") \
                or ("serpavi" in low and "_n_" not in low) or any(k in low for k in ("tourist_", "netmig_", "valor_tasado", "ine_eustat_income")) \
                or ("_emal_" in low and "deposits" not in low) or "realpricesalem2" in low or "realpricepresalem2" in low or "realrentm2" in low \
                or low.startswith("ine_eustat_dwelling") and low.endswith(("share", "avgage", "avguseful", "avgusefularea")):
            _RATE_COLUMNS.add(c)
    if _RATE_COLUMNS:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(sorted(_RATE_COLUMNS)))
    elif f.is_file():
        _RATE_COLUMNS.update(json.loads(f.read_text()))
    _map_build.register_relative_fields(_RATE_COLUMNS)


def _sanitize_census_columns(gdf):
    """Drop census columns that came out entirely empty and blank impossible negative prices / incomes / rents (sentinel or resampling noise)."""
    if not hasattr(gdf, "isna"):   # a polars frame (some cities keep the h3 grid in polars): nothing to sanitize with the pandas API
        return gdf
    census = [c for c in _census_columns(gdf) if c in gdf.columns and c != "population"]
    empty = [c for c in census if gdf[c].isna().all()]
    if empty:
        print(f"[pipeline] dropping {len(empty)} all-empty census column(s): {empty[:8]}{' ...' if len(empty) > 8 else ''}")
        gdf = gdf.drop(columns=empty)
    for c in census:
        low = c.lower()
        if c in gdf.columns and any(k in low for k in ("income", "rent", "price", "value")) and not any(k in low for k in ("cagr", "change", "growth", "gini", "share", "ratio")):
            col = gdf[c]
            if pd.api.types.is_numeric_dtype(col) and (col < 0).any():
                gdf[c] = col.where(col >= 0)
    return gdf


def _set_bare_source_file(city_dir) -> None:
    """Point `transitlos.map.build` at this city's `results/bare_column_sources.json` (source prefix of census columns renamed to bare names)."""
    from transitlos.map import build as _mb

    _mb.BARE_COLUMN_SOURCE_PREFIX.clear()
    _mb.BARE_COLUMN_SOURCE_FILE = str(Path(city_dir) / "results" / "bare_column_sources.json")


def _stops_with_stop_table(city_dir: Path, config, stops):
    """The stops the map draws: every real stop_id at its own location (`results/stops_by_id.parquet`, rebuilt when missing/stale), plus any
    scored station the table lacks; falls back to the scored stations with route labels."""
    from .stop_scoring import add_route_labels, build_stop_table_file

    try:
        _stop_table = build_stop_table_file(city_dir, config)
    except Exception as _exc:  # noqa: BLE001 -- fall back to the scored stations rather than losing the map
        print(f"[pipeline:{config.key}] stop table failed ({type(_exc).__name__}: {str(_exc)[:100]}); drawing the scored stations")
        _stop_table = None
    if _stop_table is not None and len(_stop_table):
        # The table must describe the SAME stations as the scored `stops` (a stale table from other dates clusters stops differently):
        # rebuild it when it misses stations, and keep any still-missing station as its own point so no stop disappears.
        _missing = ~stops["parent_station"].astype(str).isin(set(_stop_table["parent_station"].astype(str)))
        if _missing.mean() > 0.02:
            print(f"[pipeline:{config.key}] stop table misses {int(_missing.sum())} of {len(stops)} stations -> rebuilding it")
            _stop_table = build_stop_table_file(city_dir, config, force=True)
            _missing = ~stops["parent_station"].astype(str).isin(set(_stop_table["parent_station"].astype(str)))
        _extra = add_route_labels(stops[_missing].copy(), _gtfs_dirs(city_dir / "gtfs")) if _missing.any() else None
        stops = pd.concat([_stop_table, _extra], ignore_index=True) if _extra is not None and len(_extra) else _stop_table
        if _extra is not None and len(_extra):
            stops = gpd.GeoDataFrame(stops, geometry="geometry", crs=4326)
    else:
        stops = add_route_labels(stops, _gtfs_dirs(city_dir / "gtfs"))
    return stops


def _census_columns(gdf) -> list[str]:
    """Every census-derived column on `gdf` -- still-prefixed AND renamed-to-bare-canonical alike.

    `_rename_canonical_columns` renames a real subset of a country's census
    columns to their bare `global_schema.json` name (e.g. `households`,
    `malePopulation`, `laborForce`), so a plain
    `col.startswith(CENSUS_COLUMN_PREFIXES)` check -- correct before that
    rename existed -- silently drops every renamed column from anything
    using it to decide "which columns are census data" (e.g.
    `build_h3_by_resolution`'s resampling `sum_cols`). This adds back every
    column matching a real canonical `feature_name`, except bare
    `population` itself: that column always exists (WorldPop) regardless of
    whether a census join ran at all, so it must stay handled by its own
    dedicated `"population"` entry in `_resample_h3`'s `keep_cols`, never
    folded into `sum_cols` a second time.
    """
    canonical = _global_schema_feature_names() - _RENAME_EXCLUDED_CANONICAL_NAMES
    return [
        c
        for c in gdf.columns
        if isinstance(c, str) and (c.startswith(CENSUS_COLUMN_PREFIXES) or c in canonical or c in US_LAST_ELECTION_COLUMNS or c.startswith("elec_"))
    ]


def _population_column(gdf) -> Optional[str]:
    """Which column holds the resident headcount `pop_density`/weighting was built from.

    NOTE (2026-09-23): despite the name, bare `"population"` is NEVER
    renamed onto by `_rename_canonical_columns` (`"population"` is the one
    name `_RENAME_EXCLUDED_CANONICAL_NAMES` deliberately excludes from that
    rename -- see its own docstring). This function is the generic
    "whatever headcount this grid currently has" lookup used BEFORE
    `_finalize_population_columns` has run (weighting resamples, the
    equity-flag split, `_add_pop_jobs_columns`'s no-op guard, ...): bare
    `population` when present (WorldPop's raw count pre-census-join, or --
    for Hamburg only -- the real Zensus grid count that replaces it in
    place) is checked first, with a `<prefix>population` fallback for any
    grid this runs on before a census join has landed at all (a non-US
    city's census-geometry map layer, or a future census-join site that
    forgets to call `_rename_canonical_columns`) -- so it still computes
    real density/equity stats instead of silently getting `pop_density=NaN`
    everywhere (the stale-hardcoded-column-list anti-pattern
    AGENT_GUIDE_NEW_CITY_OR_COUNTRY.md warns about). After
    `_finalize_population_columns` has run, bare `population` is instead the
    per-city DERIVED map-facing figure (see that function's docstring) --
    this function still resolves to it correctly since it's still the first
    bare-name check, it just no longer means "raw WorldPop" once that step
    has run.
    """
    for col in ("population", *(f"{prefix}population" for prefix in CENSUS_COLUMN_PREFIXES)):
        if col in gdf.columns:
            return col
    return None


def _area_km2(gdf) -> Optional[np.ndarray]:
    """Per-row area in km2, from `area_m2` if present, else from the H3 cell id."""
    if "area_m2" in gdf.columns:
        return gdf["area_m2"].to_numpy(dtype=float) / 1e6
    if "h3_cell" in gdf.columns:
        return np.array([_cell_area_m2(c) for c in gdf["h3_cell"]], dtype=float) / 1e6
    return None


def _add_pop_jobs_columns(gdf):
    """Add `pop_jobs_total` / `pop_jobs_density` when a jobs column is present.

    A no-op (returning `gdf` untouched) for any grid without a jobs count, so
    it is safe to call unconditionally from `prepare_grid_for_map`. Recomputed
    from scratch after every resample rather than resampled, exactly like the
    share columns: both inputs (`population`, the jobs count) are additive, so
    the totals sum correctly and the density is then a plain division by the
    coarser cell's own area.
    """
    jobs_col = jobs_column(gdf)
    pop_col = _population_column(gdf)
    if jobs_col is None or pop_col is None:
        return gdf
    population = gdf[pop_col].to_numpy(dtype=float)
    jobs = gdf[jobs_col].to_numpy(dtype=float)
    # A cell with population but no jobs record (or vice versa) should still
    # get the half it does have, not NaN.
    total = np.nansum(np.vstack([population, jobs]), axis=0)
    total[np.isnan(population) & np.isnan(jobs)] = np.nan
    gdf[POP_JOBS_TOTAL_COLUMN] = total
    # Global-schema names for the same figures (`workplaceJobs`, `populationAndJobs`): a second column only when the local name differs.
    if jobs_col != GLOBAL_WORKPLACES_COLUMN and GLOBAL_WORKPLACES_COLUMN not in gdf.columns:
        gdf[GLOBAL_WORKPLACES_COLUMN] = gdf[jobs_col].to_numpy(dtype=float)
    if GLOBAL_POP_AND_WORKPLACES_COLUMN not in gdf.columns:
        gdf[GLOBAL_POP_AND_WORKPLACES_COLUMN] = total

    area_km2 = _area_km2(gdf)
    if area_km2 is not None:
        gdf[POP_JOBS_DENSITY_COLUMN] = total / np.where(area_km2 > 0, area_km2, np.nan)
    return gdf


def _census_population_column(gdf) -> Optional[str]:
    """The census-source population column specifically -- never WorldPop's bare `population`.

    Mirrors `_population_column`'s prefixed fallbacks but skips the bare
    `"population"` name entirely: that name is reserved for WorldPop's own
    count (see `_population_column`'s docstring / `_RENAME_EXCLUDED_CANONICAL_NAMES`),
    so `population_density` (item 4, CENSUS population specifically) must
    never silently fall back to it.

    2026-09-22 bug fix: this docstring used to claim iterating
    `CENSUS_COLUMN_PREFIXES` (which includes `"worldpop_"`, for
    `_add_worldpop_gapfill`'s own `worldpop_malePopulation`-style columns)
    was harmless here because `"worldpop_population"` (the one name that
    prefix+`"population"` combination would produce) "is never a real
    column name" -- true when written, but a real `worldpop_population`
    column now exists on every map (added 2026-09-06, see that commit's
    "I want worldpop population column to appear on all maps" request),
    which made this function silently return WorldPop's OWN total as if it
    were a real census count -- live-caught via the `population_overestimation_vs_census`/
    `population_underestimation_vs_census` columns (see `_add_worldpop_outside_census_columns`)
    coming out as a suspicious flat zero on Shanghai, a country with no real
    census join at all. `"worldpop_"` is explicitly excluded from the
    prefix scan now.
    """
    # Hamburg special case: `_join_zensus_grid_population` replaces bare
    # `population` IN PLACE with the real Destatis Zensus grid count (a
    # strictly better census source than the admin-polygon `destatis_population`
    # coverage column that may also be present -- see that function's own
    # docstring) and stashes the pre-replacement WorldPop value under
    # `population_worldpop`. Whenever that stash exists, bare `population`
    # IS the real census figure, so it must be returned here instead of (or
    # in preference to) any `<prefix>population` column.
    if "population_worldpop" in gdf.columns:
        return "population"
    for prefix in CENSUS_COLUMN_PREFIXES:
        if prefix == "worldpop_":
            continue
        col = f"{prefix}population"
        if col in gdf.columns:
            return col
    return None


def _worldpop_population_column(gdf) -> Optional[str]:
    """The WorldPop-derived population column, if this grid carries one.

    On the h3-grid path bare `"population"` is *always* WorldPop's own count
    (see `_population_column`'s docstring: it predates any census join and is
    excluded from `_rename_canonical_columns`), so that is checked first --
    never a `<prefix>population` census column, which `_census_population_column`
    covers separately. `_census_geometries_with_score`'s census-polygon path
    instead lands the same real WorldPop measurement under
    `worldpop_population_map_source` (see that function's own comment) when
    bare `population` is already taken by a real census count.
    """
    for col in ("population_worldpop", "worldpop_population_map_source", "population"):
        if col in gdf.columns:
            return col
    return None


def _finalize_population_columns(gdf, uses_census: bool, worldpop_only_population: bool = False):
    """Materialize `census_residents`/`worldpop_residents` and (re)derive map-facing `population`.

    Call this LAST, after every other population-derived column
    (`_add_pop_jobs_columns`, `_add_derived_density_columns`,
    `_add_worldpop_outside_census_columns`) has already run and read
    whatever `_population_column`/`_census_population_column`/
    `_worldpop_population_column` resolved to at that point -- this function
    renames/overwrites those source columns, so anything reading them
    afterwards would get the wrong (already-finalized) value.

    - `worldpop_residents` = `_worldpop_population_column(gdf)`'s value
      (always present). The old duplicate-under-another-name columns that
      value used to live under (`population_worldpop` -- Hamburg's
      pre-Zensus WorldPop stash; `worldpop_population_map_source` -- the
      census-polygon path's WorldPop fallback aggregate) are dropped.
    - `census_residents` = `_census_population_column(gdf)`'s value, when
      `uses_census` and a real census source is present (absent for
      Shanghai, and for any city whose census join happened to fail for
      this AOI). The now-redundant source column (e.g. `cbs_population`,
      `inegi_population`) is dropped, EXCEPT when that source was bare
      `population` itself (Hamburg's Zensus-grid-replaces-`population`
      special case), since bare `population` is about to be overwritten
      below anyway.
    - `population` (overwritten in place, always ends up present) =
        - `worldpop_residents`, for `worldpop_only_population` cities
          (Beersheba/Andorra by explicit user request) or any city with no
          `census_residents` at all (Shanghai);
        - `census_residents`, for every other `uses_census` city (workplaces are NOT added: `population` is residents only and the
          residents + workplaces total is `pop_jobs_total`).
      `pop_density` (the plain, pre-existing people-per-km2 column, not
      `population_density`) is recomputed from this new value so it stays
      consistent with what `population` now means.
    """
    worldpop_col = _worldpop_population_column(gdf)
    if worldpop_col is not None:
        gdf[WORLDPOP_RESIDENTS_COLUMN] = gdf[worldpop_col].astype(float)

    census_col = _census_population_column(gdf) if uses_census else None
    if census_col is not None:
        gdf[CENSUS_RESIDENTS_COLUMN] = gdf[census_col].astype(float)
        if census_col != "population":
            gdf = gdf.drop(columns=[census_col])

    # `population` is ALWAYS the residential population. The residents + workplaces total has its own column (`pop_jobs_total`,
    # "Population and workplaces"), so workplaces are never mixed into `population`.
    if CENSUS_RESIDENTS_COLUMN in gdf.columns and not worldpop_only_population:
        new_population = gdf[CENSUS_RESIDENTS_COLUMN].to_numpy(dtype=float)
    elif WORLDPOP_RESIDENTS_COLUMN in gdf.columns:
        new_population = gdf[WORLDPOP_RESIDENTS_COLUMN].to_numpy(dtype=float)
    else:
        new_population = None

    if new_population is not None:
        gdf["population"] = new_population
        if "area_m2" in gdf.columns:
            area_km2 = gdf["area_m2"].to_numpy(dtype=float) / 1e6
            gdf["pop_density"] = gdf["population"].to_numpy(dtype=float) / np.where(area_km2 > 0, area_km2, np.nan)

    drop_leftover = [
        c for c in ("population_worldpop", "worldpop_population_map_source", "worldpop_population")
        if c in gdf.columns
    ]
    if drop_leftover:
        gdf = gdf.drop(columns=drop_leftover)
    return gdf


def _finalize_population_columns_polars(df: pl.DataFrame, uses_census: bool, worldpop_only_population: bool = False) -> pl.DataFrame:
    """Polars mirror of `_finalize_population_columns` -- see that function's docstring."""
    worldpop_col = _worldpop_population_column(df)
    if worldpop_col is not None:
        df = df.with_columns(pl.col(worldpop_col).cast(pl.Float64).alias(WORLDPOP_RESIDENTS_COLUMN))

    census_col = _census_population_column(df) if uses_census else None
    if census_col is not None:
        df = df.with_columns(pl.col(census_col).cast(pl.Float64).alias(CENSUS_RESIDENTS_COLUMN))
        if census_col != "population":
            df = df.drop(census_col)

    jobs_col = jobs_column(df)
    new_population_expr = None
    if CENSUS_RESIDENTS_COLUMN in df.columns and not worldpop_only_population:
        new_population_expr = pl.col(CENSUS_RESIDENTS_COLUMN).cast(pl.Float64)   # residents only; workplaces live in `pop_jobs_total`
    elif WORLDPOP_RESIDENTS_COLUMN in df.columns:
        new_population_expr = pl.col(WORLDPOP_RESIDENTS_COLUMN).cast(pl.Float64)

    if new_population_expr is not None:
        df = df.with_columns(new_population_expr.alias("population"))
        if "area_m2" in df.columns:
            safe_area = pl.Series(
                "__safe_area_km2", np.where(df["area_m2"].cast(pl.Float64).to_numpy() / 1e6 > 0, df["area_m2"].cast(pl.Float64).to_numpy() / 1e6, np.nan)
            )
            df = df.with_columns((pl.col("population").cast(pl.Float64) / safe_area).alias("pop_density"))

    drop_leftover = [
        c for c in ("population_worldpop", "worldpop_population_map_source", "worldpop_population")
        if c in df.columns
    ]
    if drop_leftover:
        df = df.drop(drop_leftover)
    return df


# --------------------------------------------------------------------------
# Five generically-named derived-density/share columns (map popups + ANOVA/
# Regression/Distribution tabs), computed fresh after every resample/level
# aggregation exactly like `_add_pop_jobs_columns`/`_add_share_columns` above
# -- never resampled themselves, since each is a plain ratio of already-
# resampled/aggregated absolute columns:
#   - `population_density`: CENSUS population (`_census_population_column`)
#     per km2 -- same formula/unit as the pre-existing `pop_density` (people
#     per km2, not m2 -- see `_add_h3_grid`'s docstring for why), just under
#     the literal name item 4 asks for and sourced from the real census count
#     specifically rather than whichever of census/WorldPop `_population_column`
#     happens to prefer.
#   - `worldpop_population_density`: same formula, WorldPop population
#     (`_worldpop_population_column`) specifically.
#   - `jobs_density`: jobs count (`jobs_column`) per km2.
#   - `jobs_and_population_density`: a same-named alias of the pre-existing
#     `pop_jobs_density` (`POP_JOBS_DENSITY_COLUMN`, from `_add_pop_jobs_columns`)
#     -- identical inputs/formula (population + jobs, summed, over area), so
#     this reuses that computation rather than re-deriving it.
#   - `jobs_share`: jobs / (population + jobs), zero-denominator guarded the
#     same `.replace(0, np.nan)` way `_add_share_columns` guards every other
#     ratio column.
# A no-op per column whenever its real inputs are missing (no jobs count, no
# census join, no area) -- exactly the same "generic, keyed only on column
# presence" rule `_add_pop_jobs_columns` already follows.
# --------------------------------------------------------------------------


def _add_derived_density_columns(gdf):
    """Add `population_density`/`worldpop_population_density`/`jobs_density`/`jobs_and_population_density`/`jobs_share` where their real inputs exist."""
    area_km2 = _area_km2(gdf)
    if area_km2 is None:
        return gdf
    safe_area = np.where(area_km2 > 0, area_km2, np.nan)

    census_pop_col = _census_population_column(gdf)
    if census_pop_col is not None:
        gdf["population_density"] = gdf[census_pop_col].to_numpy(dtype=float) / safe_area

    worldpop_col = _worldpop_population_column(gdf)
    if worldpop_col is not None:
        gdf["worldpop_population_density"] = gdf[worldpop_col].to_numpy(dtype=float) / safe_area
        # 2026-09-02 (live user report, Shanghai): a non-census city (no
        # `census_pop_col` above, e.g. Shanghai -- China has no pyCensus
        # module) never gets the bare `population_density` name at all,
        # only `worldpop_population_density` -- inconsistent with every
        # OTHER demographic breakdown for that same city
        # (`malePopulation`/`femalePopulation`/etc, landed under bare
        # names by `_join_worldpop_global_schema` since that function is
        # the CENSUS SUBSTITUTE for a non-census city, not a supplementary
        # source -- see its own docstring). Explicit user request: "I would
        # like all columns to be duplicated just population or
        # population_density etc ... and the same columns with worldpop in
        # the column name just duplicating the data" -- a plain duplicate
        # (not a rename) so `worldpop_population_density` keeps existing
        # too, matching every other worldpop-sourced demographic column's
        # bare-name/`worldpop_`-prefixed pair.
        if census_pop_col is None:
            gdf["population_density"] = gdf["worldpop_population_density"]

    # 2026-11-13/14 (user requests): density of the neighbourhood -- residents of every cell whose centre is within 500 m / 1 / 2 / 3 km of
    # this cell's centre, over those cells' area (transitlos.neighbourhood.density_within_multi); H3 grids only
    _pc = census_pop_col or worldpop_col
    if _pc is not None and "h3_cell" in gdf.columns and len(gdf):
        from transitlos.neighbourhood import DENSITY_RADII, density_within_multi

        _dv = density_within_multi(gdf["h3_cell"].astype(str).to_numpy(), gdf[_pc].to_numpy(dtype=float), [r for _, r in DENSITY_RADII])
        for (_dc, _), _v in zip(DENSITY_RADII, _dv):
            gdf[_dc] = _v

    jobs_col = jobs_column(gdf)
    if jobs_col is not None:
        jobs = gdf[jobs_col].to_numpy(dtype=float)
        gdf["jobs_density"] = jobs / safe_area
        pop_col = _population_column(gdf)
        if pop_col is not None:
            population = gdf[pop_col].to_numpy(dtype=float)
            denom = population + jobs
            gdf["jobs_share"] = np.where(denom > 0, jobs / denom, np.nan)

    if POP_JOBS_DENSITY_COLUMN in gdf.columns:
        gdf["jobs_and_population_density"] = gdf[POP_JOBS_DENSITY_COLUMN]

    return gdf


def _add_worldpop_outside_census_columns(gdf):
    """Add `population_overestimation_vs_census`/`population_underestimation_vs_census` (absolute) and their `_share` ratios.

    2026-09-22, explicit user request (original version): "add a new column
    to distribution worldpop outside census with the population (resampled)
    from worldpop per census polygon or h3 cell minus the census population
    value and for the share value use the census population value but allow
    negative or positive values... for anova and regression use this share
    value for distribution use the subtraction just. This column should not
    exist for maps only with worldpop data but no census." Later replaced,
    verbatim: "Instead of worldpop outside census do two columns: worldpop
    overestimation (more worldpop than census) and worldpop underestimation
    (less worldpop than census population). Both column with absolute
    values. And the relative version should be overestimation / worldpop
    and underestimation / census." A no-op (`gdf` returned unchanged) unless
    BOTH a real census-source population column (`_census_population_column`)
    and WorldPop's own (`_worldpop_population_column`) are present -- a
    WorldPop-only country (no census join at all) never has a
    `_census_population_column`, so it never gets any of these four
    columns, matching that request exactly. Same "recompute fresh after
    every resample, from already-resampled absolute columns" pattern as
    `_add_derived_density_columns`/`_add_pop_jobs_columns` (both plain
    elementwise math over two additive columns, never itself resampled).

    `population_overestimation_vs_census` = max(0, worldpop - census): how much MORE
    WorldPop's raster estimate counted than the real census, floored at
    zero (no `_RELATIVE_MARKERS` substring, so
    `transitlos.map.build.is_relative_field` naturally routes it to
    Distribution's absolute-fields list). `population_underestimation_vs_census` =
    max(0, census - worldpop): how much FEWER WorldPop counted than the
    real census, also floored at zero. Together they split the old signed
    `worldpop_outside_census` difference into two always-nonnegative
    magnitudes (one of the two is always exactly 0 for any given row).

    `worldpop_overestimation_share`/`worldpop_underestimation_share` (the
    `_share` suffix routes them to ANOVA/Regression/Opacity): the absolute
    columns divided by WorldPop's own count and the real census count
    respectively (NOT by each other's numerator's source, per the user's
    "overestimation / worldpop and underestimation / census"), NaN where
    the respective denominator is zero/missing -- same zero-denominator
    convention as the old `worldpop_outside_census_share`.
    """
    census_col = _census_population_column(gdf)
    worldpop_col = _worldpop_population_column(gdf)
    if census_col is None or worldpop_col is None:
        return gdf
    census_pop = gdf[census_col].to_numpy(dtype=float)
    worldpop_pop = gdf[worldpop_col].to_numpy(dtype=float)
    overestimation = np.maximum(0.0, worldpop_pop - census_pop)
    underestimation = np.maximum(0.0, census_pop - worldpop_pop)
    gdf["population_overestimation_vs_census"] = overestimation
    gdf["population_underestimation_vs_census"] = underestimation
    gdf["worldpop_overestimation_share"] = overestimation / np.where(worldpop_pop != 0, worldpop_pop, np.nan)
    gdf["worldpop_underestimation_share"] = underestimation / np.where(census_pop != 0, census_pop, np.nan)
    return gdf


def _add_derived_density_columns_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Polars mirror of `_add_derived_density_columns`."""
    if "area_m2" in df.columns:
        area_km2 = df["area_m2"].cast(pl.Float64).to_numpy() / 1e6
    elif "h3_cell" in df.columns:
        area_km2 = _cell_area_m2_series(df["h3_cell"]) / 1e6
    else:
        return df
    safe_area = pl.Series("__safe_area_km2", np.where(area_km2 > 0, area_km2, np.nan))

    exprs = []
    census_pop_col = _census_population_column(df)
    if census_pop_col is not None:
        exprs.append((pl.col(census_pop_col).cast(pl.Float64) / safe_area).alias("population_density"))

    worldpop_col = _worldpop_population_column(df)
    if worldpop_col is not None:
        exprs.append((pl.col(worldpop_col).cast(pl.Float64) / safe_area).alias("worldpop_population_density"))
        # 2026-09-02: mirrors `_add_derived_density_columns`'s own fix
        # (see its docstring) -- a non-census city (no `census_pop_col`
        # above) never gets bare `population_density` at all through this
        # polars path either, which is what a non-census, `native_is_polars`
        # city (Shanghai's scale-driven path -- see `build_h3_by_resolution`'s
        # `native_is_polars` branch) actually goes through for every
        # coarser resolution's own `_prepare_grid_polars` call, not the
        # pandas `_add_derived_density_columns` this mirrors. Missing this
        # meant Shanghai's `stats_data.json` (built from the STATS
        # resolution, resampled through this exact path) only ever carried
        # `worldpop_population_density`, not the requested duplicate.
        if census_pop_col is None:
            exprs.append(pl.col(worldpop_col).cast(pl.Float64).truediv(safe_area).alias("population_density"))

    jobs_col = jobs_column(df)
    pop_col = _population_column(df)
    if jobs_col is not None:
        exprs.append((pl.col(jobs_col).cast(pl.Float64) / safe_area).alias("jobs_density"))
        if pop_col is not None:
            denom = pl.col(pop_col).cast(pl.Float64) + pl.col(jobs_col).cast(pl.Float64)
            exprs.append(
                pl.when(denom > 0)
                .then(pl.col(jobs_col).cast(pl.Float64) / denom)
                .otherwise(None)
                .alias("jobs_share")
            )
    if exprs:
        df = df.with_columns(exprs)
    if POP_JOBS_DENSITY_COLUMN in df.columns:
        df = df.with_columns(pl.col(POP_JOBS_DENSITY_COLUMN).alias("jobs_and_population_density"))
    _pc = census_pop_col or worldpop_col
    if _pc is not None and "h3_cell" in df.columns and df.height:   # 500 m / 1 / 2 / 3 km neighbourhood densities (see the pandas version)
        from transitlos.neighbourhood import DENSITY_RADII, density_within_multi

        _dv = density_within_multi(df["h3_cell"].cast(pl.Utf8).to_numpy(), df[_pc].cast(pl.Float64).fill_null(0.0).to_numpy(), [r for _, r in DENSITY_RADII])
        df = df.with_columns([pl.Series(_dc, _v) for (_dc, _), _v in zip(DENSITY_RADII, _dv)])
    return df


def _add_worldpop_outside_census_columns_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Polars mirror of `_add_worldpop_outside_census_columns` -- see that function's docstring."""
    census_col = _census_population_column(df)
    worldpop_col = _worldpop_population_column(df)
    if census_col is None or worldpop_col is None:
        return df
    census_f = pl.col(census_col).cast(pl.Float64)
    worldpop_f = pl.col(worldpop_col).cast(pl.Float64)
    overestimation = (worldpop_f - census_f).clip(lower_bound=0.0)
    underestimation = (census_f - worldpop_f).clip(lower_bound=0.0)
    return df.with_columns(
        overestimation.alias("population_overestimation_vs_census"),
        underestimation.alias("population_underestimation_vs_census"),
    ).with_columns(
        pl.when(worldpop_f != 0)
        .then(pl.col("population_overestimation_vs_census") / worldpop_f)
        .otherwise(None)
        .alias("worldpop_overestimation_share"),
        pl.when(census_f != 0)
        .then(pl.col("population_underestimation_vs_census") / census_f)
        .otherwise(None)
        .alias("worldpop_underestimation_share"),
    )


def population_filter_column(gdf) -> Optional[str]:
    """Which column decides whether a cell/polygon is shown on the map at all.

    Population+jobs where the study has a jobs count (a downtown block group
    full of offices and almost no residents is a real, occupied place and
    must not be hidden), plain resident population otherwise. Returns `None`
    for a grid that carries neither, in which case `filter_populated` leaves
    it alone rather than silently emptying it.
    """
    if POP_JOBS_TOTAL_COLUMN in gdf.columns:
        return POP_JOBS_TOTAL_COLUMN
    return _population_column(gdf)


def filter_populated(gdf: gpd.GeoDataFrame, positive_only: bool = False) -> gpd.GeoDataFrame:
    """Keep only rows with real people or real jobs in them.

    The rule, per the study's explicit instruction, is that the
    hexagon/circle/census layers key their inclusion on *occupancy*, never on
    access:

      - population `> 1` OR any real jobs-count column `> 1` -> keep,
        **whatever** `level_of_service` says. An inhabited block with no
        transit at all has `level_of_service == 0` (a real, meaningful zero since
        `UrbanAccessAnalyzer`'s 2026-08-13 "unreached network scores 0, not
        null" change) and is precisely the most interesting thing an
        accessibility map can show; dropping it would draw a map of where
        transit already is instead of a map of who has it.
      - neither -> drop, even if it happens to carry a nonzero access value
        (an empty industrial strip next to a busy station adds no
        information and drags every median split).

    2026-08-30 fix (explicit user request: "always show any geometry h3
    cell or census polygon if any population or jobs column has a value
    above 1"): this used to key on `population_filter_column`'s single
    winning column -- `pop_jobs_total` (population+jobs summed) when a jobs
    count exists, else bare `population` -- thresholded at `> 0`. Two real
    gaps: (1) a `population_filter_column` grid that has *neither*
    `pop_jobs_total` nor `population` (e.g. it only carries a raw jobs
    count under one of `JOBS_COLUMN_CANDIDATES` some map paths never route
    through `_add_pop_jobs_columns`) fell through `population_filter_column`
    returning `None` and skipped filtering entirely rather than checking
    jobs directly; (2) the threshold itself was `> 0`, not the `> 1` the
    user asked for. Checking `population` and the real jobs column
    independently (whichever of the two -- or both -- exist) and requiring
    either to exceed 1 satisfies both asks directly, without relying on
    `pop_jobs_total` having been computed first.
    """
    cols = _occupancy_columns(gdf)
    if not cols:
        return gdf
    keep = np.zeros(len(gdf), dtype=bool)
    for col in cols:
        values = pd.to_numeric(gdf[col], errors="coerce").to_numpy(dtype=float)
        # h3 levels (`positive_only`): the native resolution already applied the density rule (`_native_density_rule`) and the coarser
        # levels are its sums, so every cell with anybody stays; census polygons: at least `OCCUPANCY_MIN`
        keep |= np.isfinite(values) & ((values > 0) if positive_only else (values >= OCCUPANCY_MIN))
    out = gdf[keep].copy()
    _rescale_uninhabited(gdf, out, cols)
    return out


def _rescale_uninhabited(before, after, cols) -> None:
    """The people / jobs of the dropped (uninhabited) rows go into the kept rows: each count column scaled by one factor so its total
    stays (`log_roadless`, warning above 5%). In place on `after` (pandas) -- see `_filter_populated_polars` for the polars twin."""
    from .h3_population import log_roadless

    for col in cols:
        total = float(np.nansum(pd.to_numeric(before[col], errors="coerce").to_numpy(dtype=float)))
        kept = float(np.nansum(pd.to_numeric(after[col], errors="coerce").to_numpy(dtype=float)))
        if total > 0 and kept > 0 and total - kept > 1e-9 * total:
            log_roadless(f"{len(before) - len(after):,} uninhabited rows", col, total, total - kept, 0.0, 0.0, 0.0, rescaled_to=total,
                         how="in rows with fewer than 1 resident and 1 job of any source")
            after[col] = pd.to_numeric(after[col], errors="coerce") * (total / kept)


#: A cell / polygon is inhabited (drawn) when ANY population or jobs source puts at least this many people or jobs in it (`filter_populated`).
OCCUPANCY_MIN = 1.0


def _occupancy_columns(gdf) -> list:
    """Every population / jobs count of `gdf`, one per source: the map-facing `population`, WorldPop's and the census' residents
    and the jobs count -- where the sources disagree, a cell only one of them fills is still an occupied cell."""
    cols = [c for c in (_population_column(gdf), WORLDPOP_RESIDENTS_COLUMN, CENSUS_RESIDENTS_COLUMN, GBA_POPULATION_COLUMN, jobs_column(gdf)) if c and c in gdf.columns]
    return list(dict.fromkeys(cols))


def development_density_column(gdf) -> str:
    """Which density the development-opportunity flag (`equity_flag`) splits on.

    Population+jobs density where a jobs count was available (a downtown full
    of offices but few residents is *not* an under-built place, and treating
    it as one is exactly what the population-only version got wrong), plain
    `pop_density` for any city without jobs data -- unchanged behaviour there.
    """
    if POP_JOBS_DENSITY_COLUMN in gdf.columns:
        return POP_JOBS_DENSITY_COLUMN
    return "pop_density"


# --------------------------------------------------------------------------
# Which columns the map's four generic variable selectors (stats-panel ANOVA,
# stats-panel regression, "Circle size by", "Opacity by") may offer, and in
# what order.
#
# `transitlos.map.build` builds all four lists the same way: every numeric
# column of the grid it is handed, minus a small `_FIELD_EXCLUDE` set, in
# column order -- and the first entry of that list is what both the
# regression dropdown and (absent `pop_density`) the circle-size dropdown
# preselect. So the study controls those selectors purely by choosing which
# columns exist and in what order they sit, which is what the two helpers
# below do. `build.py` itself is shared across studies and stays untouched.
# --------------------------------------------------------------------------

# `area_m2` is grid bookkeeping (never real census/WorldPop data), so it
# doesn't belong in any analysis/dropdown selector. The old
# `worldpop_population_map_source` provenance-flag column no longer exists
# at all (see `_finalize_population_columns`: its value is now carried under
# the real `worldpop_residents` column instead), so it's gone from this set.
#
# `population` USED to be excluded here too (a raw headcount reads the same
# as `pop_density` on an h3 grid, where every cell has equal area, but stops
# being comparable the moment the map switches resolution or draws census
# polygons of wildly different areas) -- but per explicit user request
# ("make sure all maps have a population column, a worldpop population
# column, all other census and worldpop columns" -- 2026-09-07), every
# count-type selector (circle-size, distribution, place-rank) must include
# `population` again. 2026-09-23: `population` is now a clean, always-
# derived column (see `_finalize_population_columns`) and `census_residents`/
# `worldpop_residents` are real, independently useful count columns too, so
# none of the three are excluded here -- all three are offered in every
# count-type selector alongside every other real count column. See
# `transitlos.map.build`'s `_numeric_field_candidates`/`absolute_fields`,
# which is what actually builds the canonical count-fields list every one of
# those selectors now shares.
MAP_FIELD_EXCLUDE = frozenset({"area_m2"})


def _map_field_rank(col: str) -> int:
    """Sort rank for `_order_map_fields` -- lower means offered/defaulted earlier."""
    if col == POP_JOBS_DENSITY_COLUMN:
        # Activity density (residents + jobs) is a strictly better description
        # of "how much is going on here" than residents alone, so wherever a
        # jobs count was available to compute it, it outranks `pop_density`
        # and becomes the default regression / ANOVA / circle-size variable.
        return 0
    if col == "pop_density":
        return 1
    if col.endswith(("_share", "_rate")) or col.endswith("_density"):
        return 2
    if _is_rate_col(col):  # income/rent medians and means
        return 3
    return 4  # absolute counts


def _order_map_fields(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reorder columns so *relative* variables come first in every map selector.

    The study's rule is that a share/rate/density is the default choice for
    ANOVA, regression, circle size and opacity, with the absolute count kept
    available but never preselected. Since `build.py` derives all four
    dropdowns from column order, ordering the columns *is* the policy:
    `pop_density` first (so it is the default regression/ANOVA and
    circle-size variable), then every `*_share`/`*_rate` column, then
    rate-like medians/means, then the raw counts last.

    Non-numeric and excluded columns (`geometry`, `h3_cell`, `population`,
    ...) keep their original relative order at the front; they are never
    offered anyway.
    """
    cols = list(gdf.columns)

    def _is_offered(col: str) -> bool:
        if col in MAP_FIELD_EXCLUDE:
            return False
        try:
            # Mirrors `build._numeric_field_candidates`, including its guard:
            # pandas extension dtypes (the string dtype `GEOID` arrives as)
            # raise rather than return False from `np.issubdtype`.
            return bool(np.issubdtype(gdf[col].dtype, np.number))
        except TypeError:
            return False

    ordered = sorted(
        cols,
        key=lambda c: ((_map_field_rank(c), cols.index(c)) if _is_offered(c) else (-1, cols.index(c))),
    )
    return gdf[ordered]


def _apply_map_field_policy() -> None:
    """Teach `transitlos.map.build` to keep bookkeeping columns out of the four selectors.

    Idempotent, and confined to this study package -- `build.py` is shared
    with the other city studies and a parallel effort, so nothing here edits
    it directly. `_FIELD_EXCLUDE` gains `MAP_FIELD_EXCLUDE` (just `area_m2`
    as of 2026-09-23 -- see that constant's own comment), which removes that
    bookkeeping column from the ANOVA, regression, circle-size, opacity-by,
    and distribution-tab lists.

    2026-09-07: this function used to ALSO exclude `population` and then
    monkeypatch `_stats_json_data`/`_stats_count_fields` to sneak it back in
    for weighting/distribution purposes specifically -- per explicit user
    request ("make sure all maps have a population column... all other
    census and worldpop columns"), `population` is a real count column like
    any other now, so `MAP_FIELD_EXCLUDE` no longer names it and those two
    compensating patches are gone; `transitlos.map.build`'s own canonical
    count-fields list (`absolute_fields(_numeric_field_candidates(...))`)
    already includes it, and every selector (circle-size, distribution,
    place-rank) shares that same list.
    """
    from transitlos.map import build as _map_build

    _map_build._FIELD_EXCLUDE = set(_map_build._FIELD_EXCLUDE) | set(MAP_FIELD_EXCLUDE)

    # 3. "Circle size by" default. The regression/ANOVA default is simply
    #    `regression_fields[0]`, so column order (`_order_map_fields`) already
    #    makes it `pop_jobs_density`; the circle-size default, by contrast, is
    #    hardcoded to `pop_density` inside `build.py`. Rather than editing that
    #    shared module, the two functions that emit the default (the panel's
    #    `selected` option and the JS `window.__circleField`) are wrapped here
    #    to prefer `pop_jobs_density` whenever it is one of the offered fields,
    #    leaving `pop_density` in the dropdown and leaving every non-jobs city
    #    on exactly its old default.
    _original_control_panel_html = _map_build._control_panel_html
    _original_control_panel_js = _map_build._control_panel_js
    _preferred: dict[str, Optional[str]] = {"circle_field": None}

    def _control_panel_html_preferring_pop_jobs(circle_fields, opacity_fields, default_circle_field=None):
        # (2026-11-10: the circle-size list now also offers relative columns, so pop_jobs_density is always in it; the default stays the build's own, population)
        _preferred["circle_field"] = default_circle_field
        return _original_control_panel_html(circle_fields, opacity_fields, default_circle_field)

    def _control_panel_js_preferring_pop_jobs(map_var, default_circle_field, *args, **kwargs):
        # Always paired with the call above (same `_inject_controls_into_saved_html`
        # invocation, html first), so this reuses whatever that one selected.
        if _preferred["circle_field"] is not None:
            default_circle_field = _preferred["circle_field"]
        return _original_control_panel_js(map_var, default_circle_field, *args, **kwargs)

    _map_build._control_panel_html = _control_panel_html_preferring_pop_jobs
    _map_build._control_panel_js = _control_panel_js_preferring_pop_jobs


def _ensure_worldpop_shares(h3_by_resolution) -> list:
    """Make sure every WorldPop count column has its share column on every resolution (maps built from older cached grids lack them).

    `SHARE_COLUMNS` already names each `worldpop_*_share` = numerator / population; a grid prepared before those definitions existed
    (e.g. Shanghai's cached one) never got them. Computed here in place wherever the numerator and denominator exist and the share
    does not. Returns the share names added.
    """
    added = set()
    for res, gdf in (h3_by_resolution or {}).items():
        for share, candidates in SHARE_COLUMNS.items():
            if not share.startswith("worldpop_") or share in gdf.columns:
                continue
            for num, den in candidates:
                if num in gdf.columns and den in gdf.columns:
                    d = pd.to_numeric(gdf[den], errors="coerce").to_numpy(dtype=float)
                    n = pd.to_numeric(gdf[num], errors="coerce").to_numpy(dtype=float)
                    gdf[share] = np.where(d > 0, n / np.where(d > 0, d, 1.0), np.nan)
                    added.add(share)
                    break
    return sorted(added)


def _exclude_worldpop_twins(h3_by_resolution) -> set:
    """Keep bare-named duplicates of WorldPop columns out of the map (selectors, popups, stats panel, metadata).

    A bare column (`malePopulation`, `population_density`, ...) whose values are identical to its `worldpop_<name>` twin adds nothing
    but a second row with another name, so only the `worldpop_` one stays. Bare columns that differ from the twin (a real census
    figure next to the WorldPop estimate) are kept. `pop_density` is the same density as `worldpop_population_density`
    when population is WorldPop's, and is dropped on the same rule. Idempotent; returns the excluded names.
    """
    from transitlos.map import build as _map_build

    if not h3_by_resolution:
        return set()
    grid = h3_by_resolution[max(h3_by_resolution)]
    cols = set(grid.columns)
    dropped = set()
    for twin in [c for c in cols if c.startswith("worldpop_")]:
        bare = twin[len("worldpop_"):]
        candidates = [bare] + (["pop_density"] if bare == "population_density" else [])
        for name in candidates:
            if name in cols and name != twin:
                try:
                    a = np.asarray(grid[name], dtype=float)
                    b = np.asarray(grid[twin], dtype=float)
                    same = a.shape == b.shape and np.allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)
                except Exception:  # noqa: BLE001 -- non-numeric columns are never twins
                    same = False
                if same:
                    dropped.add(name)
    _map_build._FIELD_EXCLUDE = set(_map_build._FIELD_EXCLUDE) | dropped
    return dropped


def prepare_grid_for_map(gdf: gpd.GeoDataFrame, uses_census: bool, worldpop_only_population: bool = False) -> gpd.GeoDataFrame:
    """Materialize share columns (census only) + population/jobs columns, relative first."""
    if "level_of_service" in gdf.columns:
        # Normalize negative zero. `discretize_score`'s round/clip can produce
        # `-0.0`, which is numerically identical to `0.0` everywhere but
        # renders as the nonsense string "-0.00" in a map popup
        # (JavaScript's `(-0).toFixed(2)` keeps the sign). Adding 0.0 maps
        # -0.0 -> 0.0 and leaves every other value untouched.
        gdf["level_of_service"] = gdf["level_of_service"].to_numpy(dtype=float) + 0.0
    if uses_census:
        gdf = _add_share_columns(gdf)
    # Not gated on `uses_census`: the rule is "if this grid has a jobs count", and
    # `_add_pop_jobs_columns` is a no-op when it doesn't.
    gdf = _add_pop_jobs_columns(gdf)
    gdf = _add_derived_density_columns(gdf)
    gdf = _add_worldpop_outside_census_columns(gdf)
    gdf = _finalize_population_columns(gdf, uses_census, worldpop_only_population)
    if GBA_VS_CENSUS_COLUMNS[0] not in gdf.columns:   # Hamburg: the census is bare `population` until finalized, so compare once it is `census_residents`
        gdf = _add_gba_vs_census(gdf)
    return _order_map_fields(gdf)


def read_h3_grid_table(path) -> pl.DataFrame:
    """Read a cached `h3_grid.parquet`'s **attribute** columns into Polars, skipping geometry.

    `gpd.read_parquet` on the Boston metro grid (1.6M resolution-11 cells,
    58 attribute columns) costs ~4.6 GB RSS: it decodes the stored WKB into
    1.6M individual shapely objects and lands the attributes in a pandas
    frame, both of which are far heavier than the data itself. The same
    attributes read straight into Polars cost ~1.0 GB, and the hexagon
    geometry is fully recoverable from the `h3_cell` ids at any point (that
    is how every *resampled* resolution's geometry has always been built --
    see `_add_h3_grid`), so nothing is lost by not reading it.

    Every consumer that needs geometry gets it from `h3_grid_to_gdf`; the
    resampling path (`_resample_h3`) needs no geometry at all and can work
    off this frame directly, which is what keeps the coarse resolutions
    nearly free.

    Args:
        path: Path to a `h3_grid.parquet` written by this pipeline.

    Returns:
        Polars DataFrame with every column except `geometry` (and pandas'
        `__index_level_0__` bookkeeping column, which carries no data).
    """
    import pyarrow.parquet as pq

    skip = {"geometry", "__index_level_0__"}
    columns = [c for c in pq.ParquetFile(str(path)).schema_arrow.names if c not in skip]
    return pl.read_parquet(str(path), columns=columns)


def h3_grid_to_gdf(df: pl.DataFrame) -> gpd.GeoDataFrame:
    """Materialize hexagon geometry for a Polars H3 table read by `read_h3_grid_table`."""
    return h3_ops.to_gdf(df, h3_column="h3_cell")


# --------------------------------------------------------------------------
# Polars mirrors of the map-preparation helpers above.
#
# Every one of these steps -- share ratios, population+jobs totals and
# densities, the occupancy filter, the map-field column ordering, the
# `equity_flag` median split -- is *pure tabular arithmetic*. None of them
# reads a polygon. Running them on a GeoDataFrame nonetheless made every one
# of them drag ~1.6M shapely polygons along: `_order_map_fields` reindexes
# (copying the whole frame), `filter_populated` masks and `.copy()`s it again,
# and each intermediate holds a second copy of the geometry column's object
# array. Doing the arithmetic in Polars first and materializing hexagons
# exactly once, at the end, is what keeps `build_h3_by_resolution`'s peak flat
# (measured on Boston's 1.6M-cell resolution 11: ~5.9 GB -> ~3.4 GB).
#
# `_prepare_grid_polars` is verified row-for-row and column-for-column against
# the GeoDataFrame path it mirrors (see the equivalence check in the session
# notes); the pandas helpers stay in place and stay authoritative -- census
# geometry levels, which are small, still go through them.
# --------------------------------------------------------------------------


def _nan_or_null(col: str) -> pl.Expr:
    """`True` where `col` is null *or* NaN.

    Parquet round-trips can land "missing" as either an Arrow null (which
    Polars exposes as null and pandas as NaN) or a literal NaN payload
    (which both keep as NaN), and the pandas helpers these mirror test with
    `np.isnan`/`isfinite`, which see both alike. Treating only nulls as
    missing here would silently disagree with them on exactly the cells that
    matter (empty ocean/forest cells).
    """
    return pl.col(col).is_null() | pl.col(col).is_nan()


def _add_share_columns_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Polars mirror of `_add_share_columns` (numerator / denominator, 0 denominator -> null; plus `RATE_SOURCE_COLUMNS` rescale)."""
    exprs = []
    for share, candidates in SHARE_COLUMNS.items():
        for num, den in candidates:
            if num in df.columns and den in df.columns and _share_pair_ok(lambda c: df[c].cast(pl.Float64, strict=False).sum(), num, den):
                ratio = pl.col(num) / pl.when(pl.col(den) == 0).then(None).otherwise(pl.col(den))
                ratio = ratio.clip(0.0, 1.0)
                exprs.append(ratio.alias(share))
                break
    drop_cols = []
    for share, candidates in RATE_SOURCE_COLUMNS.items():
        for src_col, divisor in candidates:
            if src_col in df.columns:
                exprs.append((pl.col(src_col).cast(pl.Float64) / divisor).alias(share))
                if src_col != share:
                    drop_cols.append(src_col)
                break
    if exprs:
        df = df.with_columns(exprs)
    if drop_cols:
        df = df.drop(drop_cols)
    return df


def _cell_area_m2_series(cells: pl.Series) -> np.ndarray:
    """Per-cell area in m2, straight from `h3.cell_area`.

    Deliberately *not* `h3ronpy.cells_area_m2`, which is vectorized and
    faster but disagrees with `h3.cell_area` in the ~8th significant digit.
    `area_m2` feeds `pop_jobs_density`, which feeds `equity_flag`'s median
    split -- a difference that small still flips the classification of a cell
    sitting exactly on the median, so this stays on the same implementation
    the pandas path has always used.
    """
    return np.fromiter((_cell_area_m2(c) for c in cells), dtype=float, count=len(cells))


def _add_pop_jobs_columns_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Polars mirror of `_add_pop_jobs_columns` (no-op without a jobs column)."""
    jobs_col, pop_col = jobs_column(df), _population_column(df)
    if jobs_col is None or pop_col is None:
        return df
    # nansum semantics: a cell with only one of the two still gets that one;
    # only "neither" is missing.
    total = (
        pl.when(_nan_or_null(pop_col) & _nan_or_null(jobs_col))
        .then(None)
        .otherwise(
            pl.when(_nan_or_null(pop_col)).then(0.0).otherwise(pl.col(pop_col))
            + pl.when(_nan_or_null(jobs_col)).then(0.0).otherwise(pl.col(jobs_col))
        )
    )
    df = df.with_columns(total.cast(pl.Float64).alias(POP_JOBS_TOTAL_COLUMN))
    if jobs_col != GLOBAL_WORKPLACES_COLUMN and GLOBAL_WORKPLACES_COLUMN not in df.columns:
        df = df.with_columns(pl.col(jobs_col).cast(pl.Float64).alias(GLOBAL_WORKPLACES_COLUMN))
    if GLOBAL_POP_AND_WORKPLACES_COLUMN not in df.columns:
        df = df.with_columns(pl.col(POP_JOBS_TOTAL_COLUMN).alias(GLOBAL_POP_AND_WORKPLACES_COLUMN))

    if "area_m2" in df.columns:
        area_km2 = df["area_m2"].cast(pl.Float64).to_numpy() / 1e6
    elif "h3_cell" in df.columns:
        area_km2 = _cell_area_m2_series(df["h3_cell"]) / 1e6
    else:
        return df
    area_km2 = np.where(area_km2 > 0, area_km2, np.nan)
    return df.with_columns(
        (pl.col(POP_JOBS_TOTAL_COLUMN) / pl.Series(POP_JOBS_DENSITY_COLUMN, area_km2)).alias(
            POP_JOBS_DENSITY_COLUMN
        )
    )


def _order_map_fields_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Polars mirror of `_order_map_fields` (relative variables first)."""
    cols = list(df.columns)

    def _is_offered(col: str) -> bool:
        return col not in MAP_FIELD_EXCLUDE and df.schema[col].is_numeric()

    ordered = sorted(
        cols,
        key=lambda c: ((_map_field_rank(c), cols.index(c)) if _is_offered(c) else (-1, cols.index(c))),
    )
    return df.select(ordered)


def _filter_populated_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Polars mirror of `filter_populated(positive_only=True)` (h3 levels only: any population / jobs source > 0, never access-based)."""
    cols = _occupancy_columns(df)
    if not cols:
        return df
    cond = None
    for col in cols:
        c = pl.col(col).cast(pl.Float64)
        c = c.is_finite() & (c > 0)
        cond = c if cond is None else (cond | c.fill_null(False))
    out = df.filter(cond.fill_null(False))
    from .h3_population import log_roadless

    for col in cols:   # the people / jobs of the dropped rows rescaled into the kept ones (see `_rescale_uninhabited`)
        total = float(df[col].cast(pl.Float64).fill_nan(None).sum() or 0.0)
        kept = float(out[col].cast(pl.Float64).fill_nan(None).sum() or 0.0)
        if total > 0 and kept > 0 and total - kept > 1e-9 * total:
            log_roadless(f"{df.height - out.height:,} uninhabited rows", col, total, total - kept, 0.0, 0.0, 0.0, rescaled_to=total,
                         how="in rows with fewer than 1 resident and 1 job of any source")
            out = out.with_columns(pl.col(col).cast(pl.Float64) * (total / kept))
    return out


def _add_h3_columns_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Polars mirror of `_add_h3_grid`'s non-geometry work (`area_m2`, `pop_density`).

    See `_add_h3_grid` for why `pop_density` is per *square kilometre* and why
    a null `level_of_service` is filled with a real 0 rather than left missing.
    """
    if "level_of_service" in df.columns:
        df = df.with_columns(pl.col("level_of_service").fill_null(0.0))
    area_m2 = _cell_area_m2_series(df["h3_cell"])
    return df.with_columns(
        pl.Series("area_m2", area_m2),
        (pl.col("population") / pl.Series("area_m2", area_m2 / 1e6)).alias("pop_density"),
    )


def _prepare_grid_polars(
    df: pl.DataFrame, uses_census: bool, equity_thresholds: Union[None, dict, str] = None,
    worldpop_only_population: bool = False,
) -> pl.DataFrame:
    """Polars mirror of `prepare_grid_for_map` + `filter_populated` + `equity_flag`.

    Returns the table a resolution's GeoDataFrame is built from -- every
    column final, every row already filtered -- so the caller can materialize
    hexagon geometry once, for exactly the rows that survive.

    Args:
        equity_thresholds: Controls how the `equity_flag` column is added --
            default `None` (unchanged from before this parameter existed):
            compute `equity_flag` from `equity_flag(density, access,
            population)` over just THIS call's own rows, as always.
            `"skip"`: don't add `equity_flag` at all -- used by
            `build_h3_by_resolution_chunked`'s first pass, where a single
            tile's rows are the wrong scope for the median split (it's a
            whole-*resolution* statistic, not a per-tile one -- see
            `studytools.stats.equity_flag_thresholds`'s docstring). A dict (as
            returned by `equity_flag_thresholds`, or `None` from that
            function when the whole resolution was too small to split):
            add `equity_flag` via `equity_flag_from_thresholds` using these
            already-computed, whole-resolution scalars instead of computing
            a new (locally-scoped, therefore wrong for a tile) split.
    """
    if "level_of_service" in df.columns:
        # Normalize -0.0 -> 0.0; see `prepare_grid_for_map`.
        df = df.with_columns(pl.col("level_of_service").cast(pl.Float64) + 0.0)
    if uses_census:
        df = _add_share_columns_polars(df)
    df = _add_pop_jobs_columns_polars(df)
    df = _add_derived_density_columns_polars(df)
    df = _add_worldpop_outside_census_columns_polars(df)
    df = _finalize_population_columns_polars(df, uses_census, worldpop_only_population)
    if GBA_VS_CENSUS_COLUMNS[0] not in df.columns and GBA_POPULATION_COLUMN in df.columns and CENSUS_RESIDENTS_COLUMN in df.columns:
        g = pl.col(GBA_POPULATION_COLUMN).cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
        c = pl.col(CENSUS_RESIDENTS_COLUMN).cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
        over, under = pl.max_horizontal(g - c, pl.lit(0.0)), pl.max_horizontal(c - g, pl.lit(0.0))
        df = df.with_columns(over.alias(GBA_VS_CENSUS_COLUMNS[0]), under.alias(GBA_VS_CENSUS_COLUMNS[1]))
        df = df.with_columns(pl.when(g != 0).then(pl.col(GBA_VS_CENSUS_COLUMNS[0]) / g).otherwise(0.0).alias(GBA_VS_CENSUS_COLUMNS[2]),
                             pl.when(c != 0).then(pl.col(GBA_VS_CENSUS_COLUMNS[1]) / c).otherwise(0.0).alias(GBA_VS_CENSUS_COLUMNS[3]))
    df = _order_map_fields_polars(df)
    df = _filter_populated_polars(df)
    if equity_thresholds == "skip":
        return df
    density = df[development_density_column(df)].cast(pl.Float64).to_numpy()
    access = df["level_of_service"].cast(pl.Float64).to_numpy()
    pop_col = _population_column(df)
    population = df[pop_col].cast(pl.Float64).to_numpy() if pop_col else np.zeros(df.height)
    if equity_thresholds is None:
        flags = equity_flag(density, access, population)
    else:
        flags = equity_flag_from_thresholds(density, access, population, equity_thresholds)
    return df.with_columns(pl.Series("equity_flag", flags))


def _native_density_rule(h3_grid, min_per_km2: float):
    """Native (finest) cells: keep a cell when, for any residents source (WorldPop, census, GBA built-surface estimate), its residents plus
    its jobs reach `min_per_km2` per km2 (jobs alone count too). The dropped cells' counts go back into the kept ones with ONE factor per
    group for the whole area -- WorldPop (its population and its other counts), census residents (every census count column), jobs (every
    jobs column), GBA population -- so the census keeps its totals and the shares inside a cell stay right (`transitlos.grid_rules`). WorldPop
    and the GBA estimate are scaled to the census total population (their own total without a census). Building-stock counts are not rescaled.
    Logged per group, WARNING above 5%. Polars or pandas in, same type out."""
    import h3ronpy

    from .h3_population import log_roadless

    is_pl = isinstance(h3_grid, pl.DataFrame)
    n = h3_grid.height if is_pl else len(h3_grid)
    if not n or "h3_cell" not in h3_grid.columns:
        return h3_grid

    def arr(c):
        v = h3_grid[c].cast(pl.Float64).to_numpy() if is_pl else pd.to_numeric(h3_grid[c], errors="coerce").to_numpy(dtype=float)
        return np.nan_to_num(np.asarray(v, dtype=float))

    cells = h3_grid["h3_cell"].to_list() if is_pl else h3_grid["h3_cell"].astype(str).tolist()
    area = np.asarray(h3ronpy.cells_area_km2(np.asarray(h3ronpy.cells_parse(cells)).astype(np.uint64))).astype(float)
    wp, cen = _worldpop_population_column(h3_grid), _census_population_column(h3_grid)
    gba = GBA_POPULATION_COLUMN if GBA_POPULATION_COLUMN in h3_grid.columns else None
    jobs = jobs_column(h3_grid)
    j = arr(jobs) if jobs else np.zeros(n)
    dens = j / area
    for src in (wp, cen, gba):
        if src:
            dens = np.maximum(dens, (arr(src) + j) / area)
    keep = dens >= min_per_km2
    census = [c for c in _census_columns(h3_grid) if not _is_rate_col(c) and not _is_building_stock_col(c)]
    jobs_cols = [c for c in census if c in JOBS_COLUMN_CANDIDATES or c.startswith("lodes_wac_") or "jobs" in c.lower()]
    groups = [("WorldPop", wp, [c for c in [wp, *[c for c in h3_grid.columns if isinstance(c, str) and c.startswith("worldpop_") and not _is_rate_col(c)]] if c]),
              ("census residents", cen, [c for c in census if c not in jobs_cols]),
              ("jobs", jobs, jobs_cols if jobs in jobs_cols else ([jobs] if jobs else [])),
              ("GBA", gba, [gba] if gba else [])]
    out = h3_grid.filter(pl.Series(keep)) if is_pl else h3_grid[keep].copy()
    census_total = float(arr(cen).sum()) if cen else 0.0   # the census residents on the grid = the census total of the area (`_join_polygon_stats`)
    for name, anchor, cols in groups:
        cols = [c for c in dict.fromkeys(cols) if c in h3_grid.columns]
        if not anchor or anchor not in cols:
            continue
        total = float(arr(anchor).sum()); kept = float(arr(anchor)[keep].sum())
        if name in ("WorldPop", "GBA") and census_total > 0:   # the estimates are scaled to the census total population (one factor)
            total = census_total
        if total <= 0 or kept <= 0 or abs(total - kept) <= 1e-9 * total:
            continue
        log_roadless(f"{n - int(keep.sum()):,} native cells below {min_per_km2:g} residents + jobs per km2", f"{name} ({anchor})", total, total - kept,
                     0.0, 0.0, 0.0, rescaled_to=total, how=f"factor {total / kept:.4f} on {len(cols)} columns"
                     + (" (to the census total population)" if name in ("WorldPop", "GBA") and census_total > 0 else ""),
                     columns=cols, step="active cells (density rule)")
        f = total / kept
        if is_pl:
            out = out.with_columns([(pl.col(c).cast(pl.Float64) * f).alias(c) for c in cols])
        else:
            for c in cols:
                out[c] = pd.to_numeric(out[c], errors="coerce") * f
    return out


def build_h3_by_resolution(
    h3_grid, params, uses_census: bool, worldpop_only_population: bool = False
) -> dict[int, gpd.GeoDataFrame]:
    """Build the `{resolution: grid}` map input, one entry per resolution the map needs.

    Shared by `run_city_study` and `boston/rebuild_map_only.py` so the fast
    map-only rebuild can never drift from the real pipeline on which columns
    each resolution carries (it previously resampled only `acs_*` at the
    stats resolution and *nothing* at the others, so the coarser levels lost
    every census variable and the shares were carried as summed ratios).

    Every non-native resolution is resampled from `h3_grid` carrying all
    `acs_*`/`dhc_*` columns, gets its shares recomputed from its own summed
    counts, and gets its own `equity_flag` -- the map's development overlay
    picks whichever resolution has that column nearest h3 9, so it has to
    exist away from the native resolution too.

    `h3_grid` may be either a GeoDataFrame or -- preferred, and much
    cheaper -- the geometry-free Polars table from `read_h3_grid_table`.
    Only the *native* resolution's entry needs hexagon geometry for the
    source grid (every other resolution rebuilds its own from the resampled
    cell ids), so with a Polars input the native grid's ~1.6M shapely
    polygons are materialized **last**, after every coarse resolution is
    already done -- the coarse resamples then run against a ~1 GB Polars
    frame instead of a ~4.5 GB GeoDataFrame, and no step ever holds both
    the full geometry and a pandas copy of the same attributes.
    """
    # Real bug fixed 2026-08-21: this only matched USA's own source prefixes,
    # so every OTHER country's census columns (e.g. Mexico's `inegi_*`) were
    # silently dropped by `_resample_h3`'s `sum_cols=` at every resolution
    # except the native one (`params.h3_resolution`, which skips resampling
    # entirely) -- including `stats_h3_resolution`, which feeds the map's
    # Regression/ANOVA/Distribution tabs. That's why a non-US city's stats
    # panel only ever offered `pop_density` (computed fresh from population/
    # area post-resample) and never any real census field, even though the
    # native-resolution hexagon/circle "Circle size by" dropdown -- built
    # from the un-resampled grid -- had them all along. `CENSUS_COLUMN_PREFIXES`
    # covers every wired country's real source prefix (see each one's own
    # `pycensus.census.countries.<x>.<source>.schema.SCHEMA(prefix=...)`).
    #
    # `_census_columns` (not a plain prefix check) because
    # `_rename_canonical_columns` has, by this point, already renamed a real
    # subset of these to their bare `global_schema.json` name (e.g.
    # `households`, `malePopulation`) -- a plain `startswith` check would
    # silently drop exactly those renamed columns from resampling.
    # the density rule on the native cells first: every coarser level is then the exact sum of the kept native cells
    h3_grid = _native_density_rule(h3_grid, getattr(params, "h3_occupancy_min_per_km2", 10.0))
    census_cols = _census_columns(h3_grid)
    # Hamburg: `population` is the Zensus grid only while `population_worldpop` exists (`_census_population_column`); without it the
    # coarser levels fell back to the admin-polygon `destatis_population` (5.41 M vs the grid's 5.35 M, 2026-11-21 map check)
    census_cols += [c for c in ("population_worldpop",) if c in h3_grid.columns and c not in census_cols]
    resolutions = list(
        dict.fromkeys([params.h3_resolution, params.stats_h3_resolution, *params.map_h3_resolutions])
    )
    native_is_polars = isinstance(h3_grid, pl.DataFrame)
    # Coarse-first when the source is geometry-free: see the docstring.
    order = (
        [r for r in resolutions if r != params.h3_resolution] + [params.h3_resolution]
        if native_is_polars
        else resolutions
    )

    by_resolution: dict[int, gpd.GeoDataFrame] = {}
    for res in order:
        if native_is_polars:
            # Explicit GC + malloc_trim before each resolution's own work --
            # same pattern already proven for a real leak in
            # `population_and_access_to_h3` (2026-09-01, `del cells,
            # cell_lat, cell_lng` + `gc.collect()` + `malloc_trim`). Live RSS
            # instrumentation showed the baseline "already resident" figure
            # right before the native (largest) resolution's own work never
            # drops between resolutions even though earlier resolutions'
            # large intermediates are no longer referenced -- glibc's
            # allocator doesn't return freed arena memory to the OS on its
            # own, and CPython's own GC doesn't run implicitly on a tight
            # timer, so reclaimable memory from res 7/5/9's processing was
            # very likely still sitting in this process's resident set,
            # eating into the headroom the native resolution's own
            # (already large, already chunked) geometry construction needs.
            import gc as _gc
            _gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
            # All-tabular path: derive every column, apply the occupancy
            # filter and take the `equity_flag` median split in Polars, then
            # build hexagon geometry exactly once, for exactly the surviving
            # rows. Nothing above this line ever touches a polygon.
            if res == params.h3_resolution:
                table = h3_grid
            elif getattr(params, "isochrone_chunk_h3_resolution", None) is not None:
                table = _add_h3_columns_polars(
                    _resample_h3_chunked(
                        h3_grid, target_resolution=res, sum_cols=census_cols,
                        chunk_h3_resolution=params.isochrone_chunk_h3_resolution,
                    )
                )
            else:
                table = _add_h3_columns_polars(
                    _resample_h3(h3_grid, target_resolution=res, sum_cols=census_cols)
                )
            prepared = _prepare_grid_polars(table, uses_census, worldpop_only_population=worldpop_only_population)
            by_resolution[res] = h3_grid_to_gdf(prepared)
            continue

        if res == params.h3_resolution:
            grid = prepare_grid_for_map(h3_grid, uses_census, worldpop_only_population=worldpop_only_population)
        else:
            grid = prepare_grid_for_map(
                _add_h3_grid(_resample_h3(h3_grid, target_resolution=res, sum_cols=census_cols)), uses_census,
                worldpop_only_population=worldpop_only_population,
            )
        # Occupancy filter *before* `equity_flag`, so its regression fit
        # (log density vs level_of_service) is taken over the cells the map
        # actually draws rather than over a sea of empty ocean/forest cells
        # that would distort the fit.
        grid = filter_populated(grid, positive_only=True)
        # Population+jobs density where jobs data exists, population density
        # otherwise -- see `development_density_column`. `population` is
        # accepted by `equity_flag` for signature compatibility but no
        # longer used by its regression-based method -- see that function's
        # own docstring (`code/stats.py`).
        pop_col = _population_column(grid)
        grid["equity_flag"] = equity_flag(
            grid[development_density_column(grid)].to_numpy(),
            grid["level_of_service"].to_numpy(),
            grid[pop_col].to_numpy() if pop_col else np.zeros(len(grid)),
        )
        by_resolution[res] = grid
    # Restore the canonical (native-first) key order regardless of the order
    # the entries were actually computed in, so callers that iterate the dict
    # see the same sequence they always have.
    return {res: by_resolution[res] for res in resolutions}


def _h3_tile_stage1_worker(
    table_ipc_bytes: bytes,
    tile_id: str,
    resolutions: List[int],
    native_resolution: int,
    census_cols: List[str],
    uses_census: bool,
    staging_dir: str,
    worldpop_only_population: bool = False,
) -> Dict[int, str]:
    """Stage 1 of the chunked path: resample + derive columns for ONE tile, all resolutions, minus `equity_flag`.

    Runs in a `ProcessPoolExecutor` worker (see `build_h3_by_resolution_chunked`)
    -- the tile's rows are handed over pre-serialized (Arrow IPC bytes) so
    nothing about the parent process's memory is shared or implicitly
    pickled. Mirrors `build_h3_by_resolution`'s per-resolution body
    (`_resample_h3` -> `_add_h3_columns_polars` -> `_prepare_grid_polars`)
    exactly, just scoped to this tile's rows only, so the two paths can
    never silently drift in what columns/values each resolution carries.

    `equity_flag` is deliberately NOT added here (`_prepare_grid_polars(...,
    equity_thresholds="skip")`): it's a whole-*resolution* population-
    weighted-median split (see `studytools.stats.equity_flag_thresholds`'s
    docstring), so it can't be computed correctly from one tile's rows
    alone. Stage 2 (`build_h3_by_resolution_chunked`, in the parent process)
    computes the split's five scalars from a cheap geometry-free scan across
    every tile's stage-1 output; stage 3 (`_h3_tile_stage2_worker`) applies
    them and builds each tile's final geometry.

    Writes one **tabular** (no geometry yet) staging Parquet per resolution
    -- `<staging_dir>/res<R>/_staging_tile_<id>.parquet` -- and returns their
    paths.

    Resampling a coarser resolution from *only this tile's* rows is exact
    (not an approximation) as long as `tile_id`'s resolution is <= every
    resolution in `resolutions`: every fine cell's ancestor at any coarser
    resolution up to and including the tile's own resolution is, by
    construction, this same tile -- H3's parent/child hierarchy nests
    cleanly, so no coarse cell's children can ever be split across two
    tiles. `build_h3_by_resolution_chunked` enforces that precondition
    before ever submitting tiles to this worker.
    """
    import io

    table = pl.read_ipc(io.BytesIO(table_ipc_bytes))
    paths: Dict[int, str] = {}
    for res in resolutions:
        if res == native_resolution:
            resampled = table
        else:
            resampled = _add_h3_columns_polars(
                _resample_h3(table, target_resolution=res, sum_cols=census_cols)
            )
        prepared = _prepare_grid_polars(
            resampled, uses_census, equity_thresholds="skip", worldpop_only_population=worldpop_only_population
        )
        res_dir = Path(staging_dir) / f"res{res}"
        res_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(res_dir / f"_staging_tile_{tile_id}.parquet")
        prepared.write_parquet(out_path)
        paths[res] = out_path
    return paths


def _h3_tile_stage2_worker(
    staging_path: str, tile_id: str, res: int, uses_census: bool, thresholds: Optional[dict], output_dir: str,
    densities_path: Optional[str] = None,
) -> str:
    """Stage 2 (final) of the chunked path: apply the whole-resolution `equity_flag` split + build geometry for ONE (tile, resolution).

    Runs in its own `ProcessPoolExecutor` worker, same isolation rationale as
    stage 1. Reads back stage 1's tabular staging file (never geometry, so
    still cheap), adds `equity_flag` from the already-computed whole-
    resolution `thresholds` (`studytools.stats.equity_flag_from_thresholds` --
    a pure per-row comparison against 5 scalars, correct no matter which
    tile's rows it's applied to), then builds this tile's hexagon geometry
    and writes the FINAL GeoParquet, deleting the staging file.
    """
    df = pl.read_parquet(staging_path)
    if densities_path:   # the neighbourhood densities of the whole resolution (stage 1 only saw this tile's cells)
        dens = pl.read_parquet(densities_path)
        cols = [c for c in dens.columns if c != "h3_cell"]
        df = df.drop([c for c in cols if c in df.columns]).join(dens.filter(pl.col("h3_cell").is_in(df["h3_cell"])), on="h3_cell", how="left")
    if "equity_flag" not in df.columns:
        density = df[development_density_column(df)].cast(pl.Float64).to_numpy()
        access = df["level_of_service"].cast(pl.Float64).to_numpy()
        pop_col = _population_column(df)
        population = df[pop_col].cast(pl.Float64).to_numpy() if pop_col else np.zeros(df.height)
        flags = equity_flag_from_thresholds(density, access, population, thresholds)
        df = df.with_columns(pl.Series("equity_flag", flags))
    gdf = h3_grid_to_gdf(df)
    res_dir = Path(output_dir) / f"res{res}"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(res_dir / f"tile_{tile_id}.parquet")
    gdf.to_parquet(out_path)
    try:
        os.remove(staging_path)
    except OSError:
        pass
    return out_path


def build_h3_by_resolution_chunked(
    h3_grid: pl.DataFrame,
    params,
    uses_census: bool,
    output_dir: str,
    max_workers: Optional[int] = None,
    worldpop_only_population: bool = False,
) -> Dict[int, List[str]]:
    """Chunked replacement for `build_h3_by_resolution`: never materializes a full-city grid.

    For `StudyParams.chunked_h3_output`-enabled cities ONLY (see that field's
    docstring). Where `build_h3_by_resolution` builds ONE GeoDataFrame per
    map/stats resolution for the WHOLE city (its geometry-free-Polars-input
    fast path still ends every resolution, including the native one, with a
    single full-city `h3_ops.to_gdf` call), this instead runs in three
    stages:

      1. Partitions `h3_grid` (the geometry-free Polars table -- see
         `read_h3_grid_table`) into `params.chunked_h3_output`-resolution
         tiles with `geohierarchy.chunked_h3_grid.partition_h3_table` (exact,
         no halo needed -- every row already IS one whole H3 cell, see that
         function's own docstring).
      2. Stage 1 (`_h3_tile_stage1_worker`, one `ProcessPoolExecutor` worker
         per tile): resample + derive every map/stats column for every
         target resolution, scoped to that tile's rows, EXCEPT
         `equity_flag` -- a whole-resolution population-weighted-median
         split (`studytools.stats.equity_flag_thresholds`) that a single tile's
         rows can't compute correctly on their own. Writes tabular (no
         geometry) staging Parquet.
      3. Between stages (this function, main process): for each resolution,
         a `pl.scan_parquet` lazy multi-file scan reads back just the three
         columns `equity_flag_thresholds` needs (density/access/population)
         across every tile's stage-1 output and collects them ONCE into
         `studytools.stats.equity_flag_thresholds` -- cheap (three float columns,
         no geometry, same order of magnitude as the existing
         `native_is_polars` fast path's own full-city table) even at
         Shanghai's scale, and gives the exact same five scalars the
         unchunked path would compute from the same rows.
      4. Stage 2 (`_h3_tile_stage2_worker`, one worker per (tile,
         resolution)): re-reads that tile's stage-1 staging file, applies
         the resolution's already-computed thresholds via
         `studytools.stats.equity_flag_from_thresholds` (a pure per-row
         comparison -- correct however the rows are grouped), builds this
         tile's hexagon geometry, and writes the FINAL GeoParquet --
         `<output_dir>/res<R>/tile_<id>.parquet`. Nothing is ever merged
         back into one in-memory table here; callers that need a full-
         resolution view read the file list with
         `geohierarchy.chunked_h3_grid.read_h3_grid_chunked_columns`
         (attributes only) or by loading + concatenating the small number of
         per-tile GeoParquet files they actually need.

    The key invariant this buys over `build_h3_by_resolution`: at no point,
    in any process, is there an in-memory object (Polars table or
    GeoDataFrame) holding EVERY row of the city's native-resolution grid at
    once -- `h3_grid` itself is still one full-city Polars table (the
    geometry-free input this function is handed, same as the existing
    `native_is_polars` fast path already accepts -- see `partition_h3_table`'s
    reasoning for why an id-only partition of a geometry-free table is cheap,
    ~1 GB-scale for Shanghai, not the ~20 GB+ a full-city GeoDataFrame needs),
    but every subsequent step -- resample, share/derived columns, hexagon
    geometry -- runs on one tile's worth of rows, in its own process, and is
    dropped (process exits) before the next tile starts. Peak RSS in any one
    process scales with one tile's row count, never the whole city's.

    Args:
        h3_grid: Geometry-free Polars H3 attribute table (same shape as
            `build_h3_by_resolution`'s `native_is_polars` branch requires --
            a GeoDataFrame input is not supported here, since the whole
            point is to never build one for the full city).
        params: `StudyParams`; `params.chunked_h3_output` is the tile
            resolution, `params.h3_resolution`/`stats_h3_resolution`/
            `map_h3_resolutions` are the target resolutions (same set
            `build_h3_by_resolution` uses).
        uses_census: Forwarded to `_prepare_grid_polars` (adds `acs_*`-style
            share columns when True).
        output_dir: Directory `res<R>/tile_<id>.parquet` files are written
            under (created if missing).
        max_workers: `ProcessPoolExecutor` worker count (default
            `os.cpu_count()`); pass 1 to run tiles serially in-process
            (matches `build_h3_grid_chunked`'s own `max_workers=1` escape
            hatch for tests/sandboxed CI without a `spawn` context).

    Returns:
        `{resolution: [tile_parquet_path, ...]}` for every resolution in
        `params.h3_resolution`/`stats_h3_resolution`/`map_h3_resolutions`
        (deduplicated, same as `build_h3_by_resolution`'s own `resolutions`
        list), one path per non-empty tile.
    """
    from geohierarchy.chunked_h3_grid import partition_h3_table

    if not isinstance(h3_grid, pl.DataFrame):
        raise TypeError(
            "build_h3_by_resolution_chunked requires a geometry-free Polars "
            "h3_grid table (e.g. read_h3_grid_table's output), not a GeoDataFrame -- "
            "materializing a GeoDataFrame for the whole city is exactly what this "
            "function exists to avoid."
        )
    tile_resolution = params.chunked_h3_output
    if tile_resolution is None:
        raise ValueError("params.chunked_h3_output must be set to use build_h3_by_resolution_chunked")

    census_cols = _census_columns(h3_grid)
    # Hamburg: `population` is the Zensus grid only while `population_worldpop` exists (`_census_population_column`); without it the
    # coarser levels fell back to the admin-polygon `destatis_population` (5.41 M vs the grid's 5.35 M, 2026-11-21 map check)
    census_cols += [c for c in ("population_worldpop",) if c in h3_grid.columns and c not in census_cols]
    resolutions = list(
        dict.fromkeys([params.h3_resolution, params.stats_h3_resolution, *params.map_h3_resolutions])
    )
    if any(tile_resolution > res for res in resolutions):
        raise ValueError(
            f"chunked_h3_output tile resolution ({tile_resolution}) must be <= every target "
            f"resolution {sorted(resolutions)} -- otherwise a single coarse output cell's "
            "children could straddle two tiles, and per-tile resampling would no longer be exact."
        )

    partitions = partition_h3_table(h3_grid, tile_resolution, h3_col="h3_cell")
    by_resolution: Dict[int, List[str]] = {res: [] for res in resolutions}
    if not partitions:
        return by_resolution

    def _to_ipc_bytes(table: pl.DataFrame) -> bytes:
        import io

        buf = io.BytesIO()
        table.write_ipc(buf)
        return buf.getvalue()

    workers = max_workers if max_workers is not None else os.cpu_count()
    tile_items = list(partitions.items())
    staging_dir = str(Path(output_dir) / "_staging")
    use_pool = bool(workers and workers > 1 and len(tile_items) > 1)

    # --- Stage 1: per-tile resample + derive columns, everything except equity_flag. ---
    if use_pool:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(
                    _h3_tile_stage1_worker,
                    _to_ipc_bytes(table), tile_id, resolutions, params.h3_resolution,
                    census_cols, uses_census, staging_dir, worldpop_only_population,
                )
                for tile_id, table in tile_items
            ]
            stage1_results = [f.result() for f in futures]
    else:
        stage1_results = [
            _h3_tile_stage1_worker(
                _to_ipc_bytes(table), tile_id, resolutions, params.h3_resolution,
                census_cols, uses_census, staging_dir, worldpop_only_population,
            )
            for tile_id, table in tile_items
        ]

    # {res: [(tile_id, staging_path), ...]}
    staging_by_res: Dict[int, List[tuple]] = {res: [] for res in resolutions}
    for (tile_id, _table), result in zip(tile_items, stage1_results):
        for res, path in result.items():
            staging_by_res[res].append((tile_id, path))

    # --- Between stages: one cheap whole-resolution lazy scan per resolution
    # for the equity_flag split's 5 scalars -- never geometry, never every
    # column, just the 3 that `equity_flag_thresholds` needs. ---
    thresholds_by_res: Dict[int, Optional[dict]] = {}
    for res in resolutions:
        paths = [p for _, p in staging_by_res[res]]
        if not paths:
            thresholds_by_res[res] = None
            continue
        one_tile = pl.read_parquet(paths[0])
        density_col = development_density_column(one_tile)
        pop_col = _population_column(one_tile)
        select_cols = [density_col, "level_of_service"] + ([pop_col] if pop_col else [])
        scanned = pl.scan_parquet(paths).select(select_cols).collect()
        density = scanned[density_col].cast(pl.Float64).to_numpy()
        access = scanned["level_of_service"].cast(pl.Float64).to_numpy()
        population = scanned[pop_col].cast(pl.Float64).to_numpy() if pop_col else np.zeros(scanned.height)
        thresholds_by_res[res] = equity_flag_thresholds(density, access, population)

    # --- Between stages: the 500 m / 1 / 2 / 3 km neighbourhood densities need every cell of the resolution (a tile's border cells
    # have neighbours in the next tile): computed once per resolution from two columns, then written into each tile in stage 2. ---
    from transitlos.neighbourhood import DENSITY_RADII, density_within_multi
    densities_by_res: Dict[int, Optional[str]] = {}
    for res in resolutions:
        paths = [p for _, p in staging_by_res[res]]
        densities_by_res[res] = None
        if not paths:
            continue
        one_tile = pl.read_parquet(paths[0], n_rows=1)
        pc = _census_population_column(one_tile) or _worldpop_population_column(one_tile)
        if pc is None or DENSITY_RADII[0][0] not in one_tile.columns:
            continue
        cells = pl.scan_parquet(paths).select(["h3_cell", pc]).collect()
        dv = density_within_multi(cells["h3_cell"].cast(pl.Utf8).to_numpy(), cells[pc].cast(pl.Float64).fill_null(0.0).to_numpy(),
                                  [r for _, r in DENSITY_RADII])
        out = str(Path(staging_dir) / f"densities_res{res}.parquet")
        cells.select("h3_cell").with_columns([pl.Series(c, v) for (c, _), v in zip(DENSITY_RADII, dv)]).write_parquet(out)
        densities_by_res[res] = out

    # --- Stage 2: apply the whole-resolution thresholds + build geometry, per (tile, resolution). ---
    stage2_jobs = [
        (tile_id, res, path)
        for res in resolutions
        for tile_id, path in staging_by_res[res]
    ]
    if use_pool:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(
                    _h3_tile_stage2_worker,
                    path, tile_id, res, uses_census, thresholds_by_res[res], output_dir, densities_by_res[res],
                )
                for tile_id, res, path in stage2_jobs
            ]
            final_paths = [f.result() for f in futures]
    else:
        final_paths = [
            _h3_tile_stage2_worker(path, tile_id, res, uses_census, thresholds_by_res[res], output_dir, densities_by_res[res])
            for tile_id, res, path in stage2_jobs
        ]

    for (tile_id, res, _path), final_path in zip(stage2_jobs, final_paths):
        by_resolution[res].append(final_path)

    try:
        import shutil

        shutil.rmtree(staging_dir, ignore_errors=True)
    except OSError:
        pass

    return by_resolution


def _concat_gdf_parquets(paths: list[str]) -> gpd.GeoDataFrame:
    """Merge one resolution's per-tile GeoParquet files back into one whole-city GeoDataFrame.

    Used by `_finish_pipeline_stages`'s `chunked_h3_output` branch to hand
    `build_h3_by_resolution_chunked`'s tile files to downstream code that
    still expects `build_h3_by_resolution`'s plain `{res: GeoDataFrame}`
    shape -- see the call site's own comment for why this is
    correctness-only, not (yet) a peak-memory win for chunked cities.
    """
    frames = [gpd.read_parquet(p) for p in paths]
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


# Coarsest -> finest census geometry levels for the map's "census" shape
# option (distinct from `params.census_levels`, which joins ACS attributes
# onto the h3 grid and isn't order-sensitive the same way).
# "block" IS included here (real, live-verified decennial DHC data -- see
# `_usa_map_census_loader`'s docstring) even though ACS5 itself can't
# publish below blockgroup -- the map overlay dispatches per-level to
# whichever real source actually has that level's data, not to ACS alone.
MAP_CENSUS_LEVELS: tuple[str, ...] = ("county", "tract", "blockgroup", "block")


def _usa_map_census_loader(aoi, states, level, cache_dir):
    """USA's map census-polygon overlay: ACS5 for county/tract/blockgroup, real decennial DHC for block.

    ACS 5-year estimates are never published at block level (a real Census
    Bureau publication limit -- see `StudyParams.census_levels`'s own note),
    so "block" polygons here carry DHC's real fields (population, race/
    ethnicity, gender, age -- see `DHC_KEEP_COLUMNS`) instead of ACS's
    (income, education, commute mode, ...) -- an honest reflection of what's
    actually published at that resolution, not a gap-filled/fabricated
    income figure at block level. `_census_geometries_with_score` (the
    caller) works with whatever real fields whichever source returns for
    each level; it doesn't require the same field set across levels.
    """
    from pycensus.census.countries.usa import acs5, dhc

    if level == "block":
        gdf = dhc.load(aoi=aoi, states=states, level=level, cache_dir=str(cache_dir))
    else:
        gdf = acs5.load(aoi=aoi, states=states, level=level, cache_dir=str(cache_dir))
    # Guard, not a behavior change: pyCensus (2026-09-29) now tags every USA
    # source load with `.attrs["native_level"]` (real per-source
    # native-vs-resampled metadata, see `pycensus.geometry_levels.
    # is_native_level`). This dispatch was ALREADY correct -- DHC for
    # block, ACS5 for everything coarser -- so this assertion should never
    # fire; it exists purely so a future edit that breaks the dispatch
    # (e.g. accidentally routing "block" to acs5) fails loudly here rather
    # than silently returning empty/wrong data.
    assert gdf.attrs.get("native_level"), (
        f"USA map census loader fetched '{level}' from a non-native source -- "
        "this should always be a level that source natively publishes."
    )
    gdf = _restore_population_prefix(gdf, "dhc" if level == "block" else "acs5")
    return gdf


def _gtfs_dirs(gtfs_root: Path) -> list[str]:
    """List every immediate subdirectory of `gtfs_root` (one per GTFS feed)."""
    return [str(p) for p in sorted(gtfs_root.iterdir()) if p.is_dir()]


def _cell_area_m2(cell: str) -> float:
    return h3.cell_area(cell, unit="m^2")


def _add_h3_grid(pop_h3: pl.DataFrame) -> gpd.GeoDataFrame:
    """Build hexagon polygons + population density (people/km2) for a `studytools.h3_population` table.

    `pop_density` is deliberately **per square kilometre**, not per square
    metre: it is one of the variables offered in the map's ANOVA /
    regression / circle-size / opacity-by selectors, where per-m2 values
    (Boston's core lands around 0.005 people/m2) are unreadable and easy to
    misread as "no people here". Per km2 puts the same cells at the several
    thousand people/km2 an urban core actually has. `equity_flag` and every
    regression are scale-invariant (median split / linear fit), so the unit
    change moves no result, only the printed numbers.
    """
    grid = h3_ops.to_gdf(pop_h3, h3_column="h3_cell")
    if "level_of_service" in grid.columns:
        # A cell that exists (it has geometry and it touches the street
        # network) but that no street-edge score reached is a cell with *no
        # transit access*, i.e. 0 -- not a cell with unknown access. Leaving
        # it null would make it vanish from every population-weighted mean,
        # median split and map layer downstream. (Distances, where "unknown"
        # is a genuinely different statement from "zero", are deliberately
        # not treated this way anywhere in this pipeline.)
        grid["level_of_service"] = grid["level_of_service"].fillna(0.0)
    grid["area_m2"] = grid["h3_cell"].map(_cell_area_m2)
    grid["pop_density"] = grid["population"] / (grid["area_m2"] / 1e6)
    return grid


# Case-insensitive: pyCensus's post-schema-redesign column names are camelCase
# (e.g. `acs5_medianHouseholdIncome`, `acs5_householdSizeMean`), not the old
# snake_case (`acs_income_median_household`) this used to match verbatim.
#
# `"rate"`/`"ratio"`/`"share"`/`"density"` added when wiring Israel/Spain in
# (2026-08-19): Israel's `pycensus.census.countries.israel.cbs` schema has real raw source
# columns of exactly this shape (`cbs_sexRatio`, `cbs_employmentRate`,
# `cbs_populationShare019`, `cbs_populationDensity`, ...) -- these are
# already-relative figures for their whole locality polygon, so they must be
# broadcast like `mean`/`median`, never apportioned by population share (that
# would silently corrupt them exactly like the round-6 camelCase mean/median
# bug this doc already fixed once). This also fixes the same latent bug for
# Euskadi's pre-existing `eustat_populationDensity` column, which was being
# mis-apportioned before this change (never previously caught: `"density"`
# wasn't in this tuple).
_CENSUS_RATE_SUBSTRINGS = ("mean", "median", "rate", "ratio", "share", "density")

# Columns that are intensive (averaged, never summed when cells merge) although their names carry no rate word (prices per m2, incomes,
# tenure percentages, ...). Filled from the pycensus schemas (`_register_schema_rates`) and by the loaders that add unschema'd columns.
_RATE_COLUMNS: set = set()


def _is_rate_col(col: str) -> bool:
    return col in _RATE_COLUMNS or any(s in col.lower() for s in _CENSUS_RATE_SUBSTRINGS)


# Building-stock counts (buildings, Catastro dwellings / building units, footprint and floor areas): physical stock, not people. They are never
# rescaled -- the stock in cells that are dropped (off-street, below the activity threshold) is simply not on the map.
import re as _re_bs
_BUILDING_STOCK_RE = _re_bs.compile(r"(catastro_(buildings|building_units|dwellings|footprint_m2_total|floor_area_m2_total)$|buildingCount$|"
                                r"buildingsBuilt|builtFloorArea$|buildingFootprintArea$|floor_area_m2_total$|footprint_m2_total$)")


def _is_building_stock_col(col) -> bool:
    return isinstance(col, str) and not _is_rate_col(col) and bool(_BUILDING_STOCK_RE.search(col))

# Curated "absolute weight" column vocabulary for the combined multi-city
# map's rank-panel weight-column selector (`combined_map.py`'s "City rank"
# tab, weighted-median level of service by an arbitrary weight column, default
# `population`). Sourced from `pycensus.schemas.global_schema.json`'s own
# entries with `"type": "count"` (its way of marking a real absolute count,
# as opposed to `"mean"`/`"median"` entries like `medianAge`), further
# filtered to drop anything matching `_CENSUS_RATE_SUBSTRINGS` and anything
# already a `SHARE_COLUMNS`/`RATE_SOURCE_COLUMNS` key -- belt-and-suspenders
# with the substring filter, since none of those keys happen to collide with
# a global_schema count name today, but a future share/rate addition
# shouldn't have to remember to also update this list. Kept as a parallel
# constant (not a live `pycensus` import) for the same reason
# `STATS_OVERVIEW_COLUMNS` is: this module has no other need for pycensus's
# schema loader. Per-city weighted medians are computed only for whichever
# of these columns actually exist on that city's own `h3_grid` (most cities
# only have a handful -- see `write_city_summary`'s
# `_median_access_by_weight_for_grid`); a column with zero cities carrying it
# simply never appears as a selector option client-side.
WEIGHT_COLUMN_CANDIDATES: tuple[tuple[str, str], ...] = (
    # The weight column of the cross-city ranking: global-schema names only (comparable between cities). A city whose census lacks a column
    # simply does not appear when that weight is selected. The WorldPop demographic columns exist for every city.
    ("population", "Population"),
    ("households", "Households"),
    ("adultPopulation", "Adult population"),
    ("femalePopulation", "Female population"),
    ("malePopulation", "Male population"),
    ("under18Population", "Population under 18"),
    ("over65Population", "Population over 65"),
    ("urbanPopulation", "Urban population"),
    ("cars", "Cars"),
    ("educationPrimaryOrLessPopulation", "Population, primary education or less"),
    ("educationSecondaryPopulation", "Population, secondary education"),
    ("educationUniversityPopulation", "Population, university education"),
    ("educationOtherTertiaryPopulation", "Population, other tertiary education"),
    ("laborForce", "Labor force"),
    ("employedResidents", "Employed residents"),
    ("unemployedResidents", "Unemployed residents"),
    ("workplaceJobs", "Workplaces"),
    ("carCommuters", "Car commuters"),
    ("publicTransportCommuters", "Public transport commuters"),
    ("walkCommuters", "Walk commuters"),
    ("bikeCommuters", "Bike commuters"),
    ("motorcycleCommuters", "Motorcycle commuters"),
    ("motorcycleHouseholds", "Motorcycle households"),
    ("bicycleHouseholds", "Bicycle households"),
    ("carHouseholds", "Car households"),
    ("twoCarHouseholds", "Two-car households"),
    ("povertyPopulation", "Poverty population"),
    ("foreignBornPopulation", "Foreign-born population"),
    ("minorityEthnicityPopulation", "Minority-ethnicity population"),
    ("religiousPopulation", "Religious population"),
    ("minorityReligionPopulation", "Minority-religion population"),
    ("populationAndJobs", "Population and workplaces"),
    ("worldpop_malePopulation", "Male population (WorldPop)"),
    ("worldpop_femalePopulation", "Female population (WorldPop)"),
    ("worldpop_under18Population", "Population under 18 (WorldPop)"),
    ("worldpop_over65Population", "Population over 65 (WorldPop)"),
    ("worldpop_adultPopulation", "Adult population (WorldPop)"),
    ("worldpop_urbanPopulation", "Urban population (WorldPop)"),
    ("worldpop_residents", "Residents (WorldPop)"),
)
assert all(
    not any(s in col.lower() for s in _CENSUS_RATE_SUBSTRINGS)
    and col not in SHARE_COLUMNS
    and col not in RATE_SOURCE_COLUMNS
    for col, _label in WEIGHT_COLUMN_CANDIDATES
), "WEIGHT_COLUMN_CANDIDATES must only contain real absolute counts, never a rate/share column"

# Kept ACS variables -- a curated subset, not pycensus's full ~17-column ACS schema.
# Fewer columns means smaller map vector tiles (every kept column is embedded in every
# feature's tile properties) and a shorter, more legible stats-panel variable list.
# Covers population/housing, income, car/transit access, and poverty -- the ones an
# equity-focused transit-LOS study actually uses in the regression/ANOVA/map tabs.
#
# Names match pyCensus's current `acs5_`-prefixed, camelCase schema (see
# `pyCensus/src/pycensus/usa/acs5/schema.json` / `ColumnSchema(prefix="acs5")`)
# -- NOT the pre-schema-redesign `acs_`-prefixed snake_case names this used to
# have, which silently matched zero columns after that rename (2026-08-19).
ACS_KEEP_COLUMNS = {
    "acs5_population",
    "acs5_households",
    "acs5_householdsRenter",
    "acs5_householdsOwner",  # B25003_002 -- denominator-sharing complement of renter
    "acs5_medianHouseholdIncome",  # B19013_001 (rate-like: broadcast, never apportioned) -- renamed from incomeMedianHousehold to match global_schema.json's canonical name
    "acs5_incomeMeanCapita",  # B19301_001 -- per-capita income, the relative income measure
    "acs5_povertyBelow100",
    "acs5_povertyPopulation",
    # Employment (B23025) -- the "workers" equity dimension needs a labor-force
    # denominator to have a relative form at all.
    "acs5_population16plus",
    "acs5_laborForce",
    "acs5_unemployedResidents",  # renamed from unemployed to match global_schema.json's canonical name
    # Commute mode (B08301) -- transit was already here; car/walk/bike added so
    # every mode share in `SHARE_COLUMNS` has its numerator. Renamed from
    # workersTransit/workersCar/workersWalk/workersBike to match
    # global_schema.json's canonical publicTransportCommuters/carCommuters/
    # walkCommuters/bikeCommuters.
    "acs5_workers",
    "acs5_publicTransportCommuters",
    "acs5_carCommuters",
    "acs5_walkCommuters",
    "acs5_bikeCommuters",
    "acs5_vehiclesHouseholds0",
    # Rent burden (B25070, >=30% of income on rent) -- a standard housing-cost
    # equity indicator, previously missing entirely from this schema.
    "acs5_rentBurdened",
    "acs5_rentBurdenPopulation",
    "acs5_rentMedianGross",  # B25064_001 (rate-like: broadcast, never apportioned)
    # Educational attainment (B15003, population 25+) -- previously missing.
    "acs5_population25plus",
    "acs5_educationBachelorsPlus",
    "acs5_educationLessThanHs",
    "acs5_educationHighSchool",
    "acs5_educationSomeCollege",
    # Remaining real fields from `acs5_schema.json` that were still missing
    # from this curated subset (audited 2026-08-22 against every real
    # `feature_name` the schema declares -- see the schema-redesign memory
    # doc): household size, nativity, commute time, and median age are all
    # real ACS5 fields with no equity-study reason to leave out.
    "acs5_householdSizeMean",  # B25010_001 (rate-like: broadcast, never apportioned)
    "acs5_foreignBornPopulation",  # B05002 -- canonical global_schema.json field
    "acs5_commuteTimeMean",  # B08303 (rate-like: broadcast, never apportioned)
    "acs5_medianAge",  # B01002 (rate-like: broadcast, never apportioned) -- canonical global_schema.json field
}

# DHC (decennial) race/ethnicity variables -- ACS doesn't publish a race breakdown at
# all (see `pycensus.census.countries.usa.acs5.schema.SCHEMA`), decennial DHC is the only source for it
# (and is a full count, not a sample estimate, which is arguably better anyway).
#
# Names match pyCensus's current camelCase `dhc_` schema (see
# `pyCensus/src/pycensus/usa/dhc/schema.json`) -- NOT the pre-schema-redesign
# snake_case names (`dhc_population_white`) this used to have.
DHC_KEEP_COLUMNS = {
    "dhc_population",  # denominator for the race/ethnicity shares below
    "dhc_whitePopulation",
    "dhc_blackPopulation",
    "dhc_asianPopulation",
    "dhc_nativePopulation",
    "dhc_otherRacePopulation",
    "dhc_hispanicPopulation",
    # Gender (P12, sex-by-age) -- numerators for the unified `female_share`/
    # `male_share` (see `SHARE_COLUMNS`); renamed bare (`malePopulation`/
    # `femalePopulation`) by `_rename_canonical_columns` before those run.
    "dhc_malePopulation",
    "dhc_femalePopulation",
    # Children (P12, sex-by-age, under-18 bins summed by pycensus) --
    # numerator for the unified `children_share`.
    "dhc_under18Population",
    # Elderly (P12, sex-by-age, 65+ bins summed by pycensus) -- numerator
    # for the unified `elderly_share`.
    "dhc_over65Population",
    # Remaining real fields from `dhc_schema.json` that were still missing
    # from this curated subset (audited 2026-08-22): working-age population
    # (18-64, the natural third bin alongside under18/over65), total
    # housing units, and the "any non-white" aggregate DHC publishes
    # directly (P3) rather than requiring it to be derived by summing the
    # five race categories.
    "dhc_age18to64Population",
    "dhc_housingUnits",
    "dhc_nonwhitePopulation",
}


# USA-only areal interpolation: ACS5's count-style fields (labor force,
# education attainment, commute mode, poverty, renter/owner, vehicle access
# -- everything in `ACS_INTERPOLATED_FIELDS` below) are only published down
# to blockgroup (a real Census Bureau publication limit -- ACS is a sample,
# not a full count, so the Bureau doesn't release it finer). Decennial DHC
# (`dhc_population`, a full count) is real down to block, genuinely finer
# than ACS5's blockgroup. `_interpolate_acs_to_dhc_blocks` spreads each
# ACS5 blockgroup count across the real DHC blocks inside it, weighted by
# each block's *real* DHC population share of the blockgroup's total DHC
# population (equal split if the blockgroup's DHC population is zero) --
# the same population-weighted areal-interpolation technique
# `_join_polygon_stats` already uses to apportion a census polygon's count
# columns across h3 cells by WorldPop population share (see that function's
# docstring), just applied polygon-to-polygon (blockgroup -> block) instead
# of polygon-to-h3-cell, and using a real published population (DHC) as the
# weight instead of a modeled one (WorldPop).
#
# Deliberately NOT applied to ACS5's rate/mean/median fields
# (`medianHouseholdIncome`, `incomeMeanCapita`, `rentMedianGross`,
# `householdSizeMean`, `commuteTimeMean`, `medianAge`): a rate is already
# representative of the whole blockgroup, and "interpolating" it by
# population share would just copy the same value onto every block inside
# the blockgroup -- no different from the plain broadcast
# `_join_polygon_stats` already does for rate columns, so there is nothing
# for population-weighted interpolation to add for those fields.
#
# Output columns are suffixed `_interpolated` (e.g.
# `acs5_laborForce_interpolated`) -- a real but *approximated* figure
# (estimated by spreading a real blockgroup total down onto finer real
# geometry, not itself a value the Census Bureau published at block
# resolution), kept in a column distinct from any real block-resolution
# figure so the two are never silently blended under one name -- same
# spirit as `pycensus.census.worldwide.worldpop.loader`'s documented
# bin-15 linear-interpolation columns for `under18Population`.
ACS_COUNT_FIELDS_FOR_INTERPOLATION = {
    "acs5_population",
    "acs5_households",
    "acs5_householdsRenter",
    "acs5_householdsOwner",
    "acs5_povertyBelow100",
    "acs5_povertyPopulation",
    "acs5_population16plus",
    "acs5_laborForce",
    "acs5_unemployedResidents",
    "acs5_workers",
    "acs5_publicTransportCommuters",
    "acs5_carCommuters",
    "acs5_walkCommuters",
    "acs5_bikeCommuters",
    "acs5_vehiclesHouseholds0",
    "acs5_rentBurdened",
    "acs5_rentBurdenPopulation",
    "acs5_population25plus",
    "acs5_educationBachelorsPlus",
    "acs5_educationLessThanHs",
    "acs5_educationHighSchool",
    "acs5_educationSomeCollege",
    "acs5_foreignBornPopulation",
}

# `_join_polygon_stats`'s `keep_columns` set for the interpolated join --
# every field in `ACS_COUNT_FIELDS_FOR_INTERPOLATION`, `_interpolated`-suffixed.
ACS_INTERPOLATED_KEEP_COLUMNS = {f"{c}_interpolated" for c in ACS_COUNT_FIELDS_FOR_INTERPOLATION}


def _schema_columns(country: str, source: str, prefix: str) -> set:
    """Every feature of a pycensus source schema as the pipeline names it (`prefix` + feature name): ALL the local census columns."""
    import json as _json

    from pycensus.schema import schema_dir

    f = schema_dir(country) / f"{source}_schema.json"
    try:
        return {prefix + e["feature_name"] for e in _json.loads(f.read_text(encoding="utf-8")).get("features", [])}
    except (OSError, ValueError, KeyError):
        return set()


def _register_schema_rates(country: str, source: str, prefix: str) -> None:
    """Mark every intensive feature (mean / median / index / percentage) of a pycensus schema as a rate column."""
    import json as _json

    from pycensus.schema import schema_dir

    f = schema_dir(country) / f"{source}_schema.json"
    try:
        feats = _json.loads(f.read_text(encoding="utf-8")).get("features", [])
    except (OSError, ValueError):
        return
    _RATE_COLUMNS.update(prefix + e["feature_name"] for e in feats if e.get("type") in ("mean", "median", "index", "percentage"))


# The maps and the stats panel offer ALL the local census columns of a source, not a curated subset (user, 2026-10).
ACS_KEEP_COLUMNS = set(ACS_KEEP_COLUMNS) | _schema_columns("usa", "acs5", "acs5_")


def _interpolate_acs_to_dhc_blocks(
    acs_gdf: gpd.GeoDataFrame,
    block_gdf: gpd.GeoDataFrame,
    count_fields: set = ACS_COUNT_FIELDS_FOR_INTERPOLATION,
) -> gpd.GeoDataFrame:
    """Population-weighted areal interpolation: ACS5 blockgroup counts -> real DHC blocks.

    For each real DHC census block, finds the ACS5 blockgroup polygon its
    centroid falls within, then apportions that blockgroup's count-style
    fields across every block inside it by the block's share of the
    blockgroup's total real `dhc_population` (equal split among blocks if
    the blockgroup's summed DHC population is zero). See the module-level
    comment above this function for why only count fields (not rate/mean/
    median fields) are interpolated, and why output columns are
    `_interpolated`-suffixed.

    Args:
        acs_gdf: ACS5 blockgroup-level GeoDataFrame (`acs5_*`-prefixed
            columns), as returned by `pycensus.census.countries.usa.acs5.load`.
        block_gdf: DHC block-level GeoDataFrame (`dhc_population` column,
            real per-block population), as returned by
            `pycensus.census.countries.usa.dhc.load`.
        count_fields: ACS5 column names (already `acs5_`-prefixed) to
            interpolate. Any name not present in `acs_gdf` is skipped.

    Returns:
        GeoDataFrame with block `GEOID`, block `geometry`, and one
        `<field>_interpolated` column per requested field. Blocks whose
        centroid doesn't fall inside any ACS5 blockgroup get `NaN`.
    """
    block_gdf = block_gdf.reset_index(drop=True)
    fields = [f for f in count_fields if f in acs_gdf.columns]
    out = gpd.GeoDataFrame(
        {"GEOID": block_gdf["GEOID"].to_numpy()}, geometry=block_gdf.geometry.to_numpy(), crs=block_gdf.crs
    )
    if not fields or acs_gdf.empty or block_gdf.empty:
        for field in fields:
            out[f"{field}_interpolated"] = np.nan
        return out

    block_pop = pd.to_numeric(block_gdf["dhc_population"], errors="coerce").fillna(0.0).to_numpy()
    acs_proj = acs_gdf.to_crs(block_gdf.crs) if acs_gdf.crs != block_gdf.crs else acs_gdf
    acs_proj = acs_proj.reset_index(drop=True)
    block_centroids = gpd.GeoDataFrame(geometry=block_gdf.geometry.centroid, crs=block_gdf.crs)

    joined = gpd.sjoin(block_centroids, acs_proj, how="left", predicate="within")
    bg_row = joined["index_right"].to_numpy()
    # A block can land in more than one row of `joined` only if blockgroup
    # polygons overlap (they shouldn't for real Census geography); take the
    # first match per block to stay one-row-per-block.
    joined = joined[~joined.index.duplicated(keep="first")]
    bg_row = joined.reindex(range(len(block_gdf)))["index_right"].to_numpy()

    valid = ~np.isnan(bg_row)
    block_idx = np.where(valid)[0]
    bg_row_valid = bg_row[valid].astype(int)

    bg_pop_sum = pd.Series(block_pop[block_idx]).groupby(bg_row_valid).transform("sum").to_numpy()
    n_in_bg = pd.Series(bg_row_valid).groupby(bg_row_valid).transform("size").to_numpy()
    share = np.where(
        bg_pop_sum > 0, block_pop[block_idx] / np.where(bg_pop_sum > 0, bg_pop_sum, 1.0), 1.0 / n_in_bg
    )

    for field in fields:
        col_values = pd.to_numeric(acs_proj[field], errors="coerce").to_numpy()[bg_row_valid]
        values = np.full(len(block_gdf), np.nan)
        values[block_idx] = col_values * share
        out[f"{field}_interpolated"] = values

    return out


TORONTO_PROFILE_KEEP_COLUMNS = set()   # filled below from the pycensus toronto schema


def _toronto_profile_levels(cache_dir) -> tuple:
    """Dissemination areas first when the Ontario Census Profile is on disk (manual download, see the pyCensus agent guide), then neighbourhoods."""
    f = Path(cache_dir) / "canada" / "raw" / "census_profile" / "98-401-X2021006_Ontario_eng_CSV.zip"
    return ("dissemination-area", "neighbourhood") if f.is_file() else ("neighbourhood",)


def _toronto_profile_loader(aoi, states, level, cache_dir):
    """`_join_polygon_stats`-shaped loader for the City of Toronto 2021 neighbourhood profile (see pycensus ``canada.toronto``)."""
    from pycensus.census.countries.canada import toronto

    gdf = toronto.load(aoi=aoi, level=level, data_dir=str(Path(cache_dir) / "toronto_profile_data"), canada_dir=str(Path(cache_dir) / "canada"))
    gdf = _restore_population_prefix(gdf, "statcan")
    if str(level).startswith(("dissem", "da")):   # federal 2025, Ontario 2025 and Toronto 2022 mayor results per dissemination area
        from pycensus.census.countries.canada import elections as el

        for scope in ("national", "state", "municipal"):
            try:
                # federal and Ontario: every polling division of the study area (2026-11-20: only the City of Toronto's before)
                t = el.load(scope, level="da", census_geometry={"da": gdf[["GEOID", "geometry"]]},
                            area=gdf.to_crs(4326).union_all() if scope != "municipal" else None)
                gdf = _merge_elections(gdf, t, TORONTO_PROFILE_KEEP_COLUMNS, "statcan_")
            except Exception as exc:  # pragma: no cover - network/data availability varies
                print(f"[pipeline] skipping Toronto election {scope}: {exc}")
    return gdf


US_RELIGION_KEEP_COLUMNS = set()   # filled below from the pycensus usa religion schema


def _us_religion_loader(aoi, states, level, cache_dir):
    """`_join_polygon_stats`-shaped loader: county polygons with the 2020 U.S. Religion Census columns (``usrc_*``)."""
    from pycensus.census.countries.usa import acs5
    from pycensus.census.countries.usa.religion import load_county_religion

    counties = acs5.load(aoi=aoi, states=list(states) if states else None, level="county", cache_dir=str(cache_dir))[["GEOID", "geometry"]]
    rel = load_county_religion(data_dir=str(Path(cache_dir) / "usa_religion"))
    return counties.merge(rel, on="GEOID", how="left")


US_HOUSING_KEEP_COLUMNS = set()   # filled below from the pycensus usa housing schema
US_PRECINCT_ELECTION_KEEP: set = set()


def _us_precinct_elections_loader(aoi, states, level, cache_dir):
    """`_join_polygon_stats`-shaped loader: 2020 tracts with precinct-resolution elections (2024 president, 2022 governor, latest mayor)."""
    from pycensus.census.countries.usa import acs5
    from pycensus.census.countries.usa.elections_precinct import load as load_prec

    tracts = acs5.load(aoi=aoi, states=list(states) if states else None, level="tract", cache_dir=str(cache_dir))[["GEOID", "geometry"]]
    fips = {str(g)[:2] for g in tracts["GEOID"]}
    city = "boston" if "25" in fips else "san_francisco" if "06" in fips else None
    if city is None:
        return tracts
    for scope in ("national", "state", "municipal"):
        try:
            t = load_prec(city, scope, level="tract", census_geometry={"tract": tracts, "blockgroup": tracts})
            cols = _election_columns(t, US_PRECINCT_ELECTION_KEEP)
            _RATE_COLUMNS.update(c for c in cols if not c.endswith("_total_votes"))
            tracts = tracts.merge(t.reset_index()[["GEOID", *cols]] if "GEOID" not in t.columns else t[["GEOID", *cols]], on="GEOID", how="left")
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping US precinct election {scope}: {exc}")
    # 2026-11-21: the city precinct files cover only Boston / San Francisco; the rest of the metro takes the 2024 presidential result of the
    # NYT nationwide precinct map (same columns; no turnout: the NYT files carry no registered voters). Governor / mayor stay city-only.
    try:
        tracts = _us_national_nyt_fill(tracts, aoi, states, cache_dir)
    except Exception as exc:  # pragma: no cover - network/data availability varies
        print(f"[pipeline] skipping NYT 2024 precinct fill: {exc}")
    return tracts


def _us_national_nyt_fill(tracts, aoi, states, cache_dir):
    """Tracts without a 2024 presidential value get it from the NYT 2024 precinct map (dasymetric by tract population)."""
    from pycensus.census.countries.usa import acs5
    from pycensus.census.countries.usa.elections_precinct import nyt2024

    tag = "elec_national_2024"
    have = f"{tag}_left_share"
    if have in tracts.columns and tracts[have].notna().all():
        return tracts
    pop = acs5.load(aoi=aoi, states=list(states) if states else None, level="tract", cache_dir=str(cache_dir))
    pc = next((c for c in ("population", "acs5_population") if c in pop.columns), None)
    if pc is None:
        return tracts
    miss = tracts[have].isna() if have in tracts.columns else pd.Series(True, index=tracts.index)
    tgt = tracts.loc[miss, ["GEOID", "geometry"]].merge(pop[["GEOID", pc]], on="GEOID", how="left").rename(columns={pc: "population"})
    parts = []
    for fips in sorted({str(g)[:2] for g in tgt["GEOID"]}):
        sub = tgt[tgt["GEOID"].astype(str).str[:2] == fips].reset_index(drop=True)
        prec = nyt2024.load_state_precincts(fips, cache_dir=str(Path(cache_dir) / "elections"))
        v = nyt2024.allocate_to_polygons(gpd.GeoDataFrame(sub, geometry="geometry", crs=tracts.crs), prec, pop_col="population")
        if v is None:
            continue
        v.index = sub["GEOID"].to_numpy()
        parts.append(v)
    if not parts:
        return tracts
    v = pd.concat(parts)
    tot = v["votes_total"].where(v["votes_total"] > 0)
    fill = pd.DataFrame({f"{tag}_total_votes": v["votes_total"], f"{tag}_left_share": v["votes_dem"] / tot, f"{tag}_right_share": v["votes_rep"] / tot,
                         f"{tag}_other_share": (v["votes_total"] - v["votes_dem"] - v["votes_rep"]).clip(lower=0) / tot,
                         f"{tag}_party_democrat_share": v["votes_dem"] / tot, f"{tag}_party_republican_share": v["votes_rep"] / tot,
                         f"{tag}_party_other_share": (v["votes_total"] - v["votes_dem"] - v["votes_rep"]).clip(lower=0) / tot})
    for c in fill.columns:
        new = tracts["GEOID"].map(fill[c])
        tracts[c] = tracts[c].fillna(new) if c in tracts.columns else new
        US_PRECINCT_ELECTION_KEEP.add(c)
        if not c.endswith("_total_votes"):
            _RATE_COLUMNS.add(c)
    print(f"[pipeline] NYT 2024 precinct fill: {int(fill[f'{tag}_left_share'].notna().sum())} tracts outside the city precinct files")
    return tracts


def _us_housing_loader(aoi, states, level, cache_dir):
    """`_join_polygon_stats`-shaped loader: 2020 tracts with home value / rent / tenure (ACS now, ~2016, 2000), FHFA price index and HUD rents (``ushs_*``)."""
    import us as _us
    from pycensus.census.countries.usa import acs5
    from pycensus.census.countries.usa.housing import load_tract_housing

    tracts = acs5.load(aoi=aoi, states=list(states) if states else None, level="tract", cache_dir=str(cache_dir))[["GEOID", "geometry"]]
    frames = []
    for fips in sorted({str(g)[:2] for g in tracts["GEOID"]}):
        st = _us.states.lookup(fips)
        h = load_tract_housing(fips, st.abbr if st else "", str(Path(cache_dir) / "usa_housing"))
        frames.append(h.reset_index() if "GEOID" not in h.columns else h)
    housing = pd.concat(frames, ignore_index=True)
    housing = housing.rename(columns={c: f"ushs_{c}" for c in housing.columns if c != "GEOID"})
    return tracts.merge(housing, on="GEOID", how="left")


US_LAST_ELECTION_COLUMNS = {"2024_president_total_votes", "2024_president_democrat_share", "2024_president_republican_share"}


def _us_last_election_loader(aoi, states, level, cache_dir):
    """`_join_polygon_stats`-shaped loader: county polygons with the 2024 presidential result (votes, two-party-style shares 0-1)."""
    import us as _us
    from pycensus.census.countries.usa import acs5
    from pycensus.census.countries.usa import elections

    counties = acs5.load(aoi=aoi, states=list(states) if states else None, level="county", cache_dir=str(cache_dir))[["GEOID", "geometry"]]
    frames = []
    for fips in sorted({str(g)[:2] for g in counties["GEOID"]}):
        st = _us.states.lookup(fips)
        votes = elections.load("president", 2024, "county", state_fips=fips, state_abbr=st.abbr if st else None, cache_dir=str(cache_dir))
        if votes.empty:
            continue
        party_col = next((c for c in ("party", "party_simplified") if c in votes.columns), None)
        total = votes.groupby("GEOID")["votes"].sum()
        out = pd.DataFrame({"GEOID": total.index.astype(str), "2024_president_total_votes": total.to_numpy(dtype=float)})
        for party in ("democrat", "republican"):
            pv = votes[votes[party_col].astype(str).str.lower() == party].groupby("GEOID")["votes"].sum() if party_col else pd.Series(dtype=float)
            out[f"2024_president_{party}_share"] = (out["GEOID"].map(pv) / total.reindex(out["GEOID"]).to_numpy()).fillna(0.0).to_numpy()
        frames.append(out)
    if not frames:
        return gpd.GeoDataFrame(columns=["GEOID", "geometry"], geometry="geometry", crs=4326)
    return counties.merge(pd.concat(frames, ignore_index=True), on="GEOID", how="left")


def _acs_interpolated_census_loader(aoi, states, level, cache_dir):
    """`_join_polygon_stats`-shaped loader: real ACS5 blockgroup counts interpolated onto real DHC blocks.

    Ignores `level` (always operates at block resolution) -- meant to be
    called with `levels=("block",)` only. Returns `h3_grid` untouched
    (empty GeoDataFrame) if either the ACS5 or DHC fetch fails for this
    AOI, same "fail safely, skip" behavior every other loader in this
    module follows.
    """
    from pycensus.census.countries.usa import acs5, dhc

    states_arg = list(states) if states else None
    acs_gdf = acs5.load(aoi=aoi, states=states_arg, level="blockgroup", cache_dir=str(cache_dir))
    block_gdf = dhc.load(aoi=aoi, states=states_arg, level="block", cache_dir=str(cache_dir))
    # Same guard as `_usa_map_census_loader`: both fetches are already at
    # each source's own native level (ACS5 blockgroup, DHC block) -- this
    # function's whole point is interpolating ACS5 DOWN to real DHC
    # blocks precisely because ACS5 has no native block-level data of its
    # own to fetch instead, so this should never fire.
    if not acs_gdf.empty:
        assert acs_gdf.attrs.get("native_level"), "ACS5 blockgroup fetch was not native."
    if not block_gdf.empty:
        assert block_gdf.attrs.get("native_level"), "DHC block fetch was not native."
    if acs_gdf.empty or block_gdf.empty:
        return gpd.GeoDataFrame(columns=["GEOID", "geometry"], geometry="geometry", crs=4326)
    acs_gdf = _restore_population_prefix(acs_gdf, "acs5")
    block_gdf = _restore_population_prefix(block_gdf, "dhc")
    return _interpolate_acs_to_dhc_blocks(acs_gdf, block_gdf)


def _join_polygon_stats_lightweight(join_fn, h3_grid, *args, **kwargs):
    """Run a `_join_census`/`_join_race`/`_join_jobs`-shaped join against a
    lightweight (`h3_cell`+`population`-only) projection of `h3_grid`, then
    merge the newly-added columns back onto the real (full-column)
    `h3_grid`.

    2026-09-05, real bug fix (live report: hexagonal density artifact at
    H3 res-5 chunk boundaries). This replaces `_chunked_h3_grid_join`'s
    per-chunk approach to these three joins: `_join_polygon_stats`
    normalizes each census polygon's population share using only the h3
    cells visible in whatever `h3_grid` it's given, so chunking it caused
    any polygon straddling a chunk boundary to have its FULL real
    population redistributed independently within EACH chunk that saw any
    of its cells -- its true total counted once per chunk touched
    (confirmed live: a 2-chunk-straddling polygon summed to ~2x its real
    value).

    The actual memory `_join_polygon_stats` needs scales with `h3_grid`'s
    COLUMN count (it copies/holds the whole frame it's given), not with
    the join computation itself -- that only ever reads `h3_cell`/
    `population` internally, deriving each cell's centroid analytically
    from `h3_cell` (`_h3_cell_centroids`), never a real geometry column.
    So passing a `h3_cell`+`population`-only projection through the join
    keeps peak memory bounded by that tiny projection's size regardless of
    how many other worldpop/census columns the real `h3_grid` carries,
    while staying exactly correct -- one single, global apportionment pass
    over the whole grid, no chunk-boundary double counting at all.

    Args:
        join_fn: `_join_census`, `_join_race`, or `_join_jobs`.
        h3_grid: The real, full-column grid. Never mutated in place except
            for the new columns actually added.
        *args, **kwargs: Forwarded to `join_fn` after the lightweight grid.

    Returns:
        `h3_grid` with every column `join_fn` added, merged in by index.
    """
    # Geometry column kept (not dropped): a single geometry column isn't
    # the memory driver (the many extra worldpop/census attribute columns
    # are), and dropping it would silently downgrade `light` from a
    # `GeoDataFrame` to a plain `DataFrame` (losing `.crs`, which
    # `_join_polygon_stats` reads) even though `light.geometry`'s actual
    # VALUES are never used there (centroids are derived analytically from
    # `h3_cell` -- see `_h3_cell_centroids`).
    light_cols = ["h3_cell", "population"]
    if h3_grid.geometry.name not in light_cols:
        light_cols.append(h3_grid.geometry.name)
    light = h3_grid[light_cols].copy()
    before_cols = set(light.columns)
    light = join_fn(light, *args, **kwargs)
    new_cols = [c for c in light.columns if c not in before_cols]
    for c in new_cols:
        h3_grid[c] = light[c]
    # 2026-09-07 bug fix (live Hamburg report: "destatis population column
    # and worldpop population seem too similar" -- Germany's real 100m
    # Zensus grid override never actually reached the final grid). `_join_census`'s
    # Germany branch overwrites `population` IN PLACE (`_join_zensus_grid_population`
    # replaces WorldPop's estimate with the real Zensus grid value on the
    # SAME column name) -- but `new_cols` above only ever catches columns
    # that didn't exist before the join, so an in-place modification to an
    # already-existing column (`population`) was silently discarded here,
    # leaving the real, full `h3_grid`'s `population` at its original
    # pre-join (WorldPop) value even though the lightweight `light` frame's
    # own `population` was correctly updated. Always copying `population`
    # back (not just genuinely new columns) fixes this for Germany and
    # costs nothing for every other country, where `join_fn` never touches it.
    h3_grid["population"] = light["population"]
    return h3_grid


def _scatter_by_cell(h3_row: np.ndarray, pair_values: np.ndarray, n_cells: int, rate: bool) -> np.ndarray:
    """Per-cell value of one column from (cell, value) pairs: the NaN-skipping SUM for counts (all-NaN stays NaN, like pandas `sum(min_count=1)`) or the LAST non-NaN value
    for rates (like groupby `last`). `bincount` instead of a pandas groupby: linear, several times faster, and on the GPU for very large joins (`geohierarchy._backend`)."""
    out = np.full(n_cells, np.nan)
    ok = ~np.isnan(pair_values)
    if rate:
        out[h3_row[ok]] = pair_values[ok]   # a repeated index keeps the last assigned value, i.e. the last non-NaN pair
        return out
    from geohierarchy._backend import array_module, to_numpy

    xp = array_module(len(h3_row))
    cells = xp.asarray(h3_row)
    summed = to_numpy(xp.bincount(cells, weights=xp.asarray(np.where(ok, pair_values, 0.0)), minlength=n_cells))
    seen = to_numpy(xp.bincount(cells, weights=xp.asarray(ok.astype(np.float64)), minlength=n_cells)) > 0
    return np.where(seen, summed, np.nan)


import time as _time_mod


_FALLBACK_MAX_M = 600.0   # a census polygon with no cell centroid inside is attributed to its nearest cell only within this distance
_GRID_RESCALE_WARN = 0.05   # = transitlos.grid_rules.GridRules.rescale_warn_share
#: built floor area per H3 cell of the current city (`_set_floor_area`): {"resolution", "cells" (sorted uint64), "res", "non" (m2)}
_FLOOR_AREA: dict = {}


def _set_floor_area(config, aoi_gdf, resolution: int, cache_dir) -> None:
    """Load the city's buildings (its country's cadaster, GlobalBuildingAtlas where none covers them; `pycensus.cadaster.load_buildings`) and
    keep their built floor area per res-`resolution` cell (residential / other) for the census spreading (`_join_polygon_stats`)."""
    from pycensus import cadaster
    from pycensus.cadaster.metrics import cell_floor_area

    _FLOOR_AREA.clear()
    try:
        b = cadaster.load_buildings(config.country, aoi_gdf, cache_dir=cache_dir)
        cells, wr, wn = cell_floor_area(b, resolution)
        print(f"[pipeline:{config.key}] floor areas: {len(b):,} buildings ({', '.join(f'{k} {v:,}' for k, v in b['source'].value_counts().items())}), "
              f"{(wr.sum() + wn.sum()) / 1e6:,.1f} km2 in {len(cells):,} res-{resolution} cells")
        if len(cells):
            _FLOOR_AREA.update({"resolution": resolution, "cells": cells, "res": wr, "non": wn})
    except Exception as exc:  # noqa: BLE001 -- without floor areas the census is spread by WorldPop (logged)
        print(f"[pipeline:{config.key}] WARNING floor areas not available ({type(exc).__name__}: {str(exc)[:200]}) -- census spread by WorldPop")


def _metric_distance(xy_a: np.ndarray, xy_b: np.ndarray, crs) -> np.ndarray:
    """Metres between paired points given in `crs` (degrees are scaled by the local latitude; projected units are taken as metres)."""
    d = xy_a - xy_b
    try:
        geographic = bool(crs is not None and gpd.GeoSeries([], crs=crs).crs.is_geographic)
    except Exception:  # noqa: BLE001
        geographic = False
    if geographic:
        lat = np.deg2rad((xy_a[:, 1] + xy_b[:, 1]) / 2)
        return np.hypot(d[:, 0] * 111_320.0 * np.cos(lat), d[:, 1] * 110_574.0)
    return np.hypot(d[:, 0], d[:, 1])


def _inside_aoi_fraction(census_gdf: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame) -> np.ndarray:
    """Share of each census polygon's area that lies inside the AOI (1 for a polygon fully inside, 0 for one outside).

    Only polygons that straddle the AOI edge need a real intersection: a prepared-geometry `contains` / `intersects` test (vectorised, microseconds each) settles the
    ones fully inside (1) and fully outside (0) first."""
    try:
        import shapely

        utm = census_gdf.geometry.iloc[:1].set_crs(census_gdf.crs).estimate_utm_crs() if census_gdf.crs is not None else None
        g = census_gdf.geometry.to_crs(utm) if utm is not None else census_gdf.geometry
        a = aoi.to_crs(utm).union_all() if utm is not None else aoi.union_all()
        geoms = g.to_numpy()
        shapely.prepare(a)
        inside = shapely.contains(a, geoms)
        frac = np.where(inside, 1.0, 0.0)
        partial = np.flatnonzero(~inside & shapely.intersects(a, geoms))
        if len(partial):
            gp = geoms[partial]
            area = shapely.area(gp)
            inter = shapely.area(shapely.intersection(gp, a))
            frac[partial] = np.where(area > 0, np.clip(inter / np.where(area > 0, area, 1.0), 0.0, 1.0), 1.0)
        return frac
    except Exception as exc:  # noqa: BLE001 -- never block the join
        print(f"[pipeline] AOI fraction of census polygons not computed: {type(exc).__name__}: {str(exc)[:100]}")
        return np.ones(len(census_gdf))


def _residual_backfill(current: np.ndarray, empty: np.ndarray, h3_row: np.ndarray, poly_row: np.ndarray, poly_total: np.ndarray, cell_pop: np.ndarray) -> np.ndarray:
    """Values for the still-empty cells from one coarser level: per polygon, its total minus what its cells already carry, shared by WorldPop among the empty cells.

    Integer-coded `bincount`s over the (cell, polygon) pairs (NumPy, or CuPy on a usable GPU for very large joins; see `geohierarchy._backend`): linear in the pairs,
    no pandas groupby."""
    from geohierarchy._backend import array_module, to_numpy

    import pandas as pd

    stride = int(poly_row.max()) + 1 if len(poly_row) else 1
    key = pd.unique(h3_row.astype(np.int64) * stride + poly_row.astype(np.int64))   # one entry per (cell, polygon), hash-based
    cell, poly = key // stride, key % stride
    xp = array_module(len(cell))
    cell_x, poly_x = xp.asarray(cell), xp.asarray(poly)
    n_poly = int(poly.max()) + 1 if len(poly) else 0
    empty_x = xp.asarray(empty)[cell_x]
    have = xp.where(empty_x, 0.0, xp.nan_to_num(xp.asarray(current)[cell_x]))
    pop = xp.where(empty_x, xp.nan_to_num(xp.asarray(cell_pop)[cell_x]), 0.0)
    placed = xp.bincount(poly_x, weights=have, minlength=n_poly)
    # An empty cell takes the polygon's total x its own weight share of ALL the polygon's cells (2026-11-20). Before, the empty cells shared
    # the whole residual (total - what the finer level placed), which also holds the finer polygons that held no cell centre: on a coast the
    # only empty cells are water, and Toronto's municipalities put 1.6 M residents (21 %) into Lake Ontario cells (one cell 27,590 people),
    # Boston 192 k into 6 empty cells. What no cell takes now goes to the AOI-wide rescale of `_join_polygon_stats` (one factor).
    w_all = xp.nan_to_num(xp.asarray(cell_pop)[cell_x])
    pop_all = xp.bincount(poly_x, weights=w_all, minlength=n_poly)
    n_all = xp.bincount(poly_x, minlength=n_poly).astype(xp.float64)
    total = xp.nan_to_num(xp.asarray(poly_total[:n_poly] if len(poly_total) >= n_poly else np.pad(poly_total, (0, n_poly - len(poly_total)))))
    resid = xp.maximum(total - placed, 0.0)
    share = xp.where(pop_all[poly_x] > 0, pop / xp.where(pop_all[poly_x] > 0, pop_all[poly_x], 1.0), 1.0 / xp.where(n_all[poly_x] > 0, n_all[poly_x], 1.0))
    share = xp.where(empty_x, share, 0.0)
    give = xp.bincount(poly_x, weights=share, minlength=n_poly) * total            # what the polygon's empty cells take
    cap = xp.where(give > resid, resid / xp.where(give > 0, give, 1.0), 1.0)         # never more than the residual (no mass twice)
    val = xp.where(empty_x, total[poly_x] * share * cap[poly_x], 0.0)
    per_cell = to_numpy(xp.bincount(cell_x, weights=val, minlength=len(current)))
    got = to_numpy(xp.bincount(cell_x, weights=empty_x.astype(xp.float64), minlength=len(current))) > 0
    return np.where(got, per_cell, np.nan)


def _pycensus_stamp() -> str:
    """Newest modification time under the installed pycensus package (code and bundled json): a cached loader result is only reused while the loaders are unchanged."""
    import pycensus

    root, newest, n = Path(pycensus.__file__).resolve().parent, 0.0, 0
    stack = [str(root)]
    while stack:
        for e in os.scandir(stack.pop()):
            if e.is_dir(follow_symlinks=False):
                if e.name != "__pycache__":
                    stack.append(e.path)
            elif e.name.endswith((".py", ".json")):
                newest, n = max(newest, e.stat().st_mtime), n + 1
    return f"{newest:.0f}-{n}"


def _cached_census_load(loader, aoi, states_arg, level, cache_dir, prefix):
    """`loader(aoi=..., level=...)` with its processed result kept as geoparquet under `<cache_dir>/_loader_cache`.

    A census loader re-reads and re-processes the raw polygons and tables on every run (minutes for Toronto's StatCan levels) although nothing changed between two rebuilds.
    The cache key covers the AOI geometry, level, states, the loader, the source file the loader is defined in (the loader adapters here call
    further helpers of this module, so a change to any of them must miss the cache) and the pycensus code state; set CS_LOADER_CACHE=0 to bypass
    it. Any cache problem falls back to the loader."""
    import hashlib
    import inspect

    if os.environ.get("CS_LOADER_CACHE", "1") == "0":
        return loader(aoi=aoi, states=states_arg, level=level, cache_dir=str(cache_dir))
    try:
        key = hashlib.sha1("|".join([prefix, str(level), str(states_arg), getattr(loader, "__module__", "") + "." + getattr(loader, "__qualname__", ""), _pycensus_stamp(),
                                     hashlib.sha1(Path(inspect.getsourcefile(loader)).read_bytes()).hexdigest(),
                                     hashlib.sha1(aoi.to_crs(4326).union_all().wkb).hexdigest()]).encode()).hexdigest()[:16]
        path = Path(cache_dir) / "_loader_cache" / f"{prefix}{level}_{key}.parquet"
        if path.is_file():
            gdf = gpd.read_parquet(path)
            print(f"[pipeline] {prefix}{level}: census loader result from cache ({len(gdf):,} rows)")
            return gdf
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] loader cache unavailable: {type(exc).__name__}: {str(exc)[:80]}")
        path = None
    gdf = loader(aoi=aoi, states=states_arg, level=level, cache_dir=str(cache_dir))
    if path is not None and gdf is not None and not gdf.empty:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_parquet(str(path) + ".tmp")
            os.replace(str(path) + ".tmp", path)
        except Exception as exc:  # noqa: BLE001
            print(f"[pipeline] loader result not cached: {type(exc).__name__}: {str(exc)[:80]}")
    return gdf


def _is_activity_column(keep_col: str, prefix: str) -> bool:
    """Is this joined column a population or jobs COUNT (a source of occupied cells)?"""
    base = keep_col[len(prefix):] if prefix and keep_col.startswith(prefix) else keep_col
    return base in ("population", "totalPopulation") or keep_col in JOBS_COLUMN_CANDIDATES or keep_col.endswith(("JobsTotal", "_jobs_total", "jobsTotal"))


def _log_census_roadless(census_gdf, source, prefix, keep_columns, inside_frac, unmatched_rows, keep_fb) -> None:
    """`log_roadless` for each population / jobs count of one census level: its polygons with no street cell inside (their people are
    not near a road of the grid) go whole to the nearest cell within `_FALLBACK_MAX_M`, or are dropped beyond it."""
    from .h3_population import log_roadless

    rows = np.asarray(unmatched_rows, dtype=int)
    for col in census_gdf.columns:
        keep_col = col if col.startswith(prefix) else f"{prefix}{col}"
        if col == "geometry" or keep_col not in keep_columns or not _is_activity_column(keep_col, prefix):
            continue
        v = pd.to_numeric(census_gdf[col], errors="coerce").fillna(0).to_numpy() * inside_frac
        roadless = float(v[rows].sum()) if len(rows) else 0.0
        dropped = float(v[rows[~keep_fb]].sum()) if len(rows) else 0.0
        log_roadless(source, keep_col, float(v.sum()), roadless, roadless - dropped, dropped, _FALLBACK_MAX_M)


def _join_polygon_stats(
    h3_grid: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    states: tuple[str, ...],
    levels: tuple[str, ...],
    cache_dir: Path,
    loader,
    keep_columns: set,
    prefix: str,
) -> gpd.GeoDataFrame:
    """Join count/rate columns from a census `loader` onto `h3_grid` by centroid-in-polygon.

    Shared by the ACS join (`_join_census`) and the DHC race join
    (`_join_race`) -- same apportionment/fallback logic, different source
    and column set. Never clips `aoi` to the city-core polygon -- always
    the full metro AOI, per the plan's explicit instruction, so coverage
    matches the metro-wide h3 grid regardless of which split (core/metro)
    is written later.

    Count-style columns (population, households, workers, ... -- anything
    whose name doesn't look like a rate/mean/median) are *apportioned*
    across every h3 cell within a census polygon by that cell's share of
    the polygon's total WorldPop `population` (equal split if the whole
    polygon has zero population), not simply copied onto every cell as-is.
    A census polygon is very often bigger than one h3 cell, so a naive
    broadcast copy would let `sum(acs_population)` across cells wildly
    exceed the real total (confirmed: Boston's metro `acs_population`
    summed to ~40M against a real ~5M metro population). Rate-style
    columns (income, commute time, ...) are still broadcast as-is --
    apportioning a *rate* by population share would be meaningless, a rate
    is already representative of the whole polygon.

    Args:
        loader: `pycensus.census.countries.usa.acs5.load`/`pycensus.census.countries.usa.dhc.load`-shaped
            callable: `(aoi, states, level, cache_dir) -> GeoDataFrame`.
        keep_columns: Column names (after `prefix`-renaming) to keep; every
            other column from the source is dropped before joining.
        prefix: Prefix the source's own columns already carry (`"acs5_"`/
            `"dhc_"`/`"lodes_wac_"`) -- used only to recognize which columns
            to consider.
    """
    # `h3_row`/`poly_row` below (and the final `values` arrays) are built and
    # consumed as *positional* (0..len-1) indices into plain numpy arrays
    # (`cell_population`, `values = np.full(len(h3_grid), ...)`), but they
    # originate from `centroids`/`joined`'s pandas *label* index. Those only
    # coincide when `h3_grid` has a clean default RangeIndex. A city's
    # on-disk `core` grid is a row subset of a larger `metro` grid (see
    # `city_config.py`/results split), so its saved index is a sparse subset
    # of a much larger range (confirmed for guadalajara: index up to 448013
    # over only 54181 rows) -- `cell_population[h3_row]` then raises
    # `IndexError` for any matched label past `len(h3_grid)`, and even where
    # it doesn't raise, a label that happens to be < len(h3_grid) silently
    # indexes the wrong row. Reset to a clean positional index for the
    # duration of this function and restore the caller's original index on
    # the way out so nothing downstream (which may key on it) is disturbed.
    original_index = h3_grid.index
    h3_grid = h3_grid.reset_index(drop=True)

    h3_grid = h3_grid.copy()
    import h3ronpy
    from h3ronpy.vector import cells_to_coordinates

    from .h3_population import log_roadless

    # Candidate cells (the normal-map rule, `transitlos.grid_rules`): the grid's street cells plus every cell holding a building
    # (`_FLOOR_AREA`, set by `_set_floor_area` for the AOI). A census polygon's counts go to the candidate cells whose centre it holds (one
    # polygon per cell), split by built floor area -- residential floor area for residents' columns, non-residential for jobs, each falling
    # back to the total floor area, then to an equal split; without floor areas, by WorldPop. Only the street cells are kept: the share of the
    # off-street buildings, and every polygon holding no candidate centre at all, go back into the kept cells with ONE factor per group
    # (residents' columns by the residents factor, jobs columns by the jobs factor) at the end, so totals and shares inside a cell stay right.
    grid_ids = np.asarray(h3ronpy.cells_parse(h3_grid["h3_cell"].astype(str).tolist())).astype(np.uint64)
    fa = _FLOOR_AREA if _FLOOR_AREA.get("resolution") is not None and len(grid_ids) and _FLOOR_AREA["resolution"] == int(h3ronpy.cells_resolution(grid_ids[:1])[0].as_py()) else {}
    if fa:
        universe = np.union1d(grid_ids, fa["cells"])
        w_res = np.zeros(len(universe)); w_non = np.zeros(len(universe))
        ix = np.searchsorted(universe, fa["cells"]); w_res[ix] = fa["res"]; w_non[ix] = fa["non"]
    else:
        universe = np.unique(grid_ids)
        wp = np.zeros(len(universe)); wp[np.searchsorted(universe, grid_ids)] = np.nan_to_num(pd.to_numeric(h3_grid["population"], errors="coerce").to_numpy(float))
        w_res = w_non = wp
    grid_row_of_uni = np.full(len(universe), -1, np.int64)
    grid_row_of_uni[np.searchsorted(universe, grid_ids)] = np.arange(len(grid_ids))
    co = cells_to_coordinates(universe)
    uni_pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.asarray(co.column("lng")), np.asarray(co.column("lat"))), crs=4326)
    w_tot = w_res + w_non if fa else w_res
    group_w = {"res": w_res, "non": w_non}

    def is_jobs(col):
        return col in JOBS_COLUMN_CANDIDATES or col.startswith("lodes_wac_") or col.endswith(("JobsTotal", "_jobs_total", "jobsTotal")) or "jobs" in col.lower()

    states_arg = list(states) if states else None
    targets: dict = {}   # count column -> total inside the AOI at the finest level that has it (what the cells must add up to)
    for level in levels:
        try:
            census_gdf = _cached_census_load(loader, aoi, states_arg, level, cache_dir, prefix)
        except Exception as exc:  # pragma: no cover - network/data availability varies per level
            print(f"[pipeline] skipping {prefix}level {level!r}: {exc}")
            continue
        if census_gdf.empty:
            continue
        census_gdf = census_gdf.reset_index(drop=True)
        pts = uni_pts.to_crs(census_gdf.crs)
        _tj = _time_mod.time()
        joined = gpd.sjoin(pts, census_gdf[["geometry"]], how="inner", predicate="within")
        joined = joined[~joined.index.duplicated()]
        print(f"[TIMING] census join {prefix}{level}: point-in-polygon sjoin of {len(pts):,} candidate cell centres x {len(census_gdf):,} polygons took {_time_mod.time()-_tj:.1f}s")
        _tc = _time_mod.time()
        uni_row = joined.index.to_numpy()
        poly_row = joined["index_right"].to_numpy().astype(int)
        n_poly = len(census_gdf)
        inside_frac = _inside_aoi_fraction(census_gdf, aoi)
        n_in = np.bincount(poly_row, minlength=n_poly)

        def share_of(w):
            wc = np.nan_to_num(w[uni_row])
            tot = np.bincount(poly_row, weights=wc, minlength=n_poly)[poly_row]
            return np.where(tot > 0, wc / np.where(tot > 0, tot, 1.0), np.nan)

        s_all = share_of(w_tot)
        s_all = np.where(np.isnan(s_all), 1.0 / n_in[poly_row], s_all)
        shares = {g: np.where(np.isnan(sg := share_of(w)), s_all, sg) for g, w in group_w.items()}
        g_row = grid_row_of_uni[uni_row]
        on_grid = g_row >= 0
        h3_row, p_row = g_row[on_grid], poly_row[on_grid]
        n_unmatched = int((n_in == 0).sum())

        for col in census_gdf.columns:
            # `pycensus` renames canonical columns to their bare name (e.g. `inegi_population` -> `population`); this module still works with the
            # prefixed names (`keep_columns`, `SHARE_COLUMNS`, ...), and the prefixed key also keeps a bare `population` from colliding with the
            # grid's own WorldPop `population` (see `_RENAME_EXCLUDED_CANONICAL_NAMES`).
            keep_col = col if col.startswith(prefix) else f"{prefix}{col}"
            if keep_col not in keep_columns:
                continue
            is_rate = _is_rate_col(keep_col)
            # `pd.to_numeric`: a source can publish `None` (object dtype) for some regions (Israel's small localities)
            poly_vals = pd.to_numeric(census_gdf[col], errors="coerce").to_numpy(dtype=float)
            if is_rate:
                pair_values = poly_vals[p_row]
            else:
                # a polygon reaching outside the AOI only brings the part inside it (its area share)
                sh = shares["non" if is_jobs(keep_col) else "res"][on_grid]
                pair_values = poly_vals[p_row] * sh * inside_frac[p_row]
                if keep_col not in targets:
                    targets[keep_col] = float(np.nansum(poly_vals * inside_frac))
                    if _is_activity_column(keep_col, prefix):
                        lost = float(np.nansum((poly_vals * inside_frac)[n_in == 0]))
                        off = float(np.nansum(poly_vals[poly_row] * shares["non" if is_jobs(keep_col) else "res"] * inside_frac[poly_row])) - float(np.nansum(pair_values))
                        if targets[keep_col] > 0:
                            log_roadless(f"{prefix}{level}", keep_col, targets[keep_col], lost + max(off, 0.0), 0.0, 0.0, 0.0, rescaled_to=targets[keep_col],
                                         how=f"in {n_unmatched} polygons holding no cell centre and in buildings away from the streets")
            # `min_count=1` on the sum path (in `_scatter_by_cell`): a level with no real value for this column (all NaN) must stay NaN, not 0,
            # so a coarser level can still fill it (Hamburg's district-only foreignBornPopulation, 2026-08-23).
            values = _scatter_by_cell(h3_row, pair_values, len(h3_grid), is_rate)
            if keep_col not in h3_grid.columns:
                h3_grid[keep_col] = values
            else:
                # cells a finer level left empty (NaN) take this coarser level's values; for counts only each polygon's residual (its total
                # inside the AOI minus what its cells already carry), shared by floor area among the empty cells -- never twice the mass
                missing = h3_grid[keep_col].isna().to_numpy()
                if not is_rate:
                    gw = group_w["non" if is_jobs(keep_col) else "res"]
                    cw = np.zeros(len(h3_grid)); cw[grid_row_of_uni[grid_row_of_uni >= 0]] = gw[grid_row_of_uni >= 0]
                    values = _residual_backfill(h3_grid[keep_col].to_numpy(dtype=float), missing, h3_row, p_row, poly_vals * inside_frac, cw)
                h3_grid.loc[missing, keep_col] = values[missing]
        print(f"[TIMING] census join {prefix}{level}: apportioning + per-column scatter took {_time_mod.time()-_tc:.1f}s")

    # one factor per group for the whole AOI: what no street cell received (off-street buildings, polygons holding no cell centre)
    count_cols = [c for c in targets if c in h3_grid.columns and not _is_building_stock_col(c)]
    for grp, cols in (("residents", [c for c in count_cols if not is_jobs(c)]), ("jobs", [c for c in count_cols if is_jobs(c)])):
        if not cols:
            continue
        anchor = max(cols, key=lambda c: targets[c])
        placed = float(np.nansum(h3_grid[anchor].to_numpy(dtype=float)))
        if targets[anchor] <= 0 or placed <= 0:
            continue
        f = targets[anchor] / placed
        share = (targets[anchor] - placed) / targets[anchor]
        msg = f"[rescale] {prefix}{grp}: {targets[anchor] - placed:,.0f} of {targets[anchor]:,.0f} {anchor} ({share:.1%}) not on a street cell -- factor {f:.4f} on {len(cols)} columns"
        print(("WARNING " if abs(share) > _GRID_RESCALE_WARN else "") + msg)
        from .h3_population import record_rescale

        record_rescale("census join (street cells)", f"{prefix}{grp}", anchor, targets[anchor], placed, f, cols)
        for c in cols:
            h3_grid[c] = h3_grid[c].to_numpy(dtype=float) * f
    h3_grid.index = original_index
    return h3_grid


# Mexico (INEGI) has no ACS/DHC/LODES equivalents -- it publishes population,
# household, religion, health, disability, education, and labor-force totals
# (see `pycensus.census.countries.mexico.inegi.schema.SCHEMA`) but nothing shaped like US
# race/ethnicity categories or LODES workplace-jobs flows. This is the
# population/household counterpart of `ACS_KEEP_COLUMNS` for `_join_census`
# only; `_join_race`/`_join_jobs` have no Mexico source at all and are
# skipped for `country="MEX"` (see their dispatch below) rather than faking
# data or crashing. Kept set is every real, verified field in
# `inegi/schema.json` (25 fields) -- not a hand-curated subset -- so the
# study gets the full column set INEGI actually publishes. Field names
# match `pycensus`'s `global_schema.json` canonical names wherever an exact
# semantic equivalent exists (`malePopulation`, `femalePopulation`,
# `adultPopulation`, `minorityEthnicityPopulation` -- Mexico's own
# indigenous-language-speaker count, `employedResidents`,
# `unemployedResidents`, `carHouseholds`, and the computed
# `religiousPopulation` = population - populationNoReligion); fields with no
# genuine canonical equivalent (religion sub-breakdowns, healthcare,
# disability, mean schooling grade, dwelling amenities) keep their real
# INEGI-specific names rather than being forced into a mismatched global one.
INEGI_KEEP_COLUMNS = {
    "inegi_population",
    "inegi_malePopulation",
    "inegi_femalePopulation",
    "inegi_population15plus",
    "inegi_adultPopulation",
    "inegi_minorityEthnicityPopulation",
    "inegi_populationCatholic",
    "inegi_populationEvangelicalProtestant",
    "inegi_populationOtherReligion",
    "inegi_populationNoReligion",
    "inegi_religiousPopulation",
    "inegi_laborForce",
    "inegi_employedResidents",
    "inegi_unemployedResidents",
    "inegi_populationWithHealthcare",
    "inegi_populationWithoutHealthcare",
    "inegi_populationDisability",
    "inegi_educationMeanGrade",
    "inegi_households",
    "inegi_dwellings",
    "inegi_occupiedDwellings",
    "inegi_dwellingsInternet",
    "inegi_dwellingsComputer",
    "inegi_dwellingsCellphone",
    "inegi_carHouseholds",
}

# INEGI's `inegi.load` implements "block" (urban manzana), "ageb" (urban
# AGEB, tract-equivalent), "municipality", and "state" levels (see
# `pycensus.census.countries.mexico.geography.IMPLEMENTED_LEVELS`) -- every other standard
# level falls back, via `pycensus.census.countries.mexico.constants.LEVEL_FALLBACK`, onto one
# of these four. Listed finest-first: `_join_polygon_stats` joins each level
# in order and only backfills h3 cells still missing a value at the next
# (coarser) level, so block gives the finest real resolution in urban AOI
# coverage, AGEB fills in urban gaps block doesn't cover, and
# municipality/state fill in the rural gaps AGEB/block don't cover.
INEGI_LEVELS: tuple[str, ...] = ("block", "ageb", "municipality", "state")


def _mexico_census_loader(aoi, states, level, cache_dir):
    """Adapt `pycensus.census.countries.mexico.inegi.load`'s signature to `_join_polygon_stats`'s.

    Explicitly forwards `data_dir` (INEGI's raw ITER/RESAGEBURB download
    cache) alongside `cache_dir` (the joined parquet cache) -- `inegi.load`
    treats these as two separate parameters and previously only received
    `cache_dir` here, so raw downloads silently fell back to `inegi`'s own
    CWD-relative default instead of living under the shared census cache.
    """
    from pycensus.census.countries.mexico import inegi

    gdf = inegi.load(
        aoi=aoi, states=states, level=level, cache_dir=str(cache_dir), data_dir=str(Path(cache_dir) / "mexico")
    )
    gdf = _restore_population_prefix(gdf, "inegi")
    if level in ("ageb", "municipality") and len(gdf):   # 2024 presidential result (casilla -> seccion -> H3 -> AGEB / municipality)
        # municipality: the fallback for the cells outside the urban AGEBs (rural areas had no election value, 2026-11-20)
        try:
            from pycensus.census.countries.mexico import elections as el

            t = el.load("national", level=level, census_geometry={level: gdf[["GEOID", "geometry"]]}, aoi=aoi)
            gdf = _merge_elections(gdf, t, INEGI_KEEP_COLUMNS, "inegi_")
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping Mexico election join: {exc}")
    return gdf


# Euskadi (Basque Country) municipal population, via pycensus.euskadi --
# real data sourced from Eurostat/GISCO's LAU 2021 dataset (see that
# module's docstrings for why: eustat.eus/opendata.euskadi.eus both 403/404
# a scripted client). Only population/area/density are real/verified so far
# (see pycensus/euskadi/eustat/schema.json) -- keep set matches exactly.
EUSKADI_KEEP_COLUMNS = {"eustat_population", "eustat_areaKm2", "eustat_populationDensity"}

# pycensus.euskadi implements only these three real levels (see
# pycensus.census.countries.euskadi.constants.LEVEL_FALLBACK) -- finer standard levels all
# fall back to "municipality" inside the loader itself, so requesting them
# here would just repeat the same fetch three times for no benefit.
EUSKADI_LEVELS: tuple[str, ...] = ("municipality", "territory")


def _euskadi_census_loader(aoi, states, level, cache_dir):
    """Adapt `pycensus.census.countries.euskadi.eustat.load`'s signature to `_join_polygon_stats`'s.

    Euskadi has no US-style `states` concept (`states` is ignored, unlike
    the USA/Mexico loaders) -- `regions` is derived from `aoi` instead, via
    `pycensus.census.countries.euskadi.geography.territories_in_aoi`.
    """
    from pycensus.census.countries.euskadi import eustat

    return _restore_population_prefix(eustat.load(aoi=aoi, level=level, cache_dir=str(cache_dir)), "eustat")


# Spain (INE Padron, `pycensus.census.countries.spain.ine`, plus INE's Censo 2021
# SDC21 dissemination API, `pycensus.census.countries.spain.censo2021`) -- real,
# verified population + education/employment/migration/household data (see
# the schema-redesign memory doc's "Spain rebuilt for real" round, and the
# 2026-08-22 census2021-column-coverage pass). Both sources' output columns
# are canonicalized to their BARE `global_schema.json` name by
# `pycensus.canonical.apply_canonical_names` (e.g. `population`, not
# `ine_population`) -- `_join_polygon_stats` reconstructs the `"ine_"`-
# prefixed form purely as this set's naming convention (see its own
# docstring/comment), so every entry here is `"ine_" + <bare column name>`
# regardless of which of the two real sources actually produced it.
# censo2021's `educationOtherTertiaryPopulation` and any income/commute-mode
# fields are NOT included: SDC21 has no way to split "other tertiary" from
# university (`ID_ESREAL_GR5` only has one tertiary bucket) and no income or
# commute-mode variable at all -- see
# `pycensus.census.countries.spain.censo2021.loader` module docstring.
SPAIN_KEEP_COLUMNS = {
    "ine_population",
    "ine_malePopulation",
    "ine_femalePopulation",
    "ine_households",
    "ine_educationPrimaryOrLessPopulation",
    "ine_educationSecondaryPopulation",
    "ine_educationUniversityPopulation",
    "ine_laborForce",
    "ine_employedResidents",
    "ine_unemployedResidents",
    "ine_foreignBornPopulation",
}

# `pycensus.census.countries.spain.geography.IMPLEMENTED_LEVELS` = (nation,
# autonomous_community, province, municipality, section) -- as of a
# 2026-08-2x rebuild INE's own OGC "Secciones censales" service is real and
# live-verified (see that module's docstring), and `spain.ine.loader.load`
# genuinely fetches real attribute data at `level="section"` (not just
# boundary geometry -- see `ine/loader.py`'s `_fetch_missing`, which hits
# the section-level Padron source when `level == "section"`), so "section"
# (roughly blockgroup-equivalent) is the finest real level, not
# municipality. `censo2021.load` shares the same real geography and also
# has genuine section-level attribute data (see its module docstring).
# district/block still have no verified cartography source and are
# correctly absent. Listed finest-first, same convention as `INEGI_LEVELS`.
SPAIN_LEVELS: tuple[str, ...] = ("section", "municipality", "province")


def _spain_census_loader(aoi, states, level, cache_dir):
    """Adapt Spain's two real census sources to `_join_polygon_stats`'s single-loader signature.

    Both `pycensus.census.countries.spain.ine.load` (Padron population) and
    `pycensus.census.countries.spain.censo2021.load` (Censo 2021 education/
    employment/migration/households) take `regions` (autonomous-community/
    province/municipality/section code), not `states` -- derived from `aoi`
    when omitted, so `states` is ignored here (same pattern as
    `_euskadi_census_loader`). The two sources share the same real
    boundaries (`pycensus.census.countries.spain.geography`), so their outputs are
    merged on `GEOID` (dropping censo2021's own geometry column, since
    it's identical to ine's for a shared `GEOID`). If censo2021's fetch
    fails (its API is a separate, less-established endpoint than ine's
    Padron), that's swallowed and only the Padron population columns are
    returned -- a partial-but-real join beats no join.
    """
    from pycensus.census.countries.spain import censo2021, ine

    data_dir = str(Path(cache_dir) / "spain")
    result = _restore_population_prefix(
        ine.load(aoi=aoi, level=level, cache_dir=str(cache_dir), data_dir=data_dir), "ine"
    )
    try:
        extra = censo2021.load(aoi=aoi, level=level, cache_dir=str(cache_dir), data_dir=data_dir)
        extra_cols = [c for c in extra.columns if c not in ("geometry",)]
        result = result.merge(extra[extra_cols], on="GEOID", how="left")
    except Exception as exc:  # pragma: no cover - network/data availability varies
        print(f"[pipeline] skipping Spain censo2021 join, keeping Padron population only: {exc}")
    # INE income atlas (ADRH): income, Gini, relative poverty, age/household/nationality indicators per section and municipality.
    if level in ("section", "municipality") and len(result):
        try:
            from pycensus.census.countries.spain import adrh

            provinces = sorted({str(g)[:2] for g in result["GEOID"]})
            attrs = adrh.load_attributes(provinces, level, data_dir=data_dir)
            if level == "section":   # Basque sections: Eustat's sex / age / nationality table (ADRH values win where both have a column)
                from pycensus.census.countries.spain.adrh.eustat_sections import load_sections

                eus = load_sections(provinces, data_dir=data_dir)
                if len(eus):
                    attrs = attrs.set_index("GEOID").combine_first(eus.set_index("GEOID")).reset_index()
            result = result.merge(attrs, on="GEOID", how="left")
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping Spain ADRH income join: {exc}")
    return _spain_housing_extras(result, level, data_dir)


_SPAIN_COUNT_EXTRAS = {"ine_serpavi_n_vc_2024", "ine_dwellings", "ine_eustat_dwellingsTotal", "ine_eustat_dwellingsFamily",
                       "ine_eustat_dwellingsMain", "ine_eustat_dwellingsOther", "ine_emal_deposits"}


def _spain_housing_extras(result, level, data_dir):
    """Housing prices / rents / tenure / tourism / migration / Basque income layered onto the Spanish section or municipality table.

    Sources (all pycensus): SERPAVI rent per m2 (section, flats `vc` and houses `vu`; 2011-2024, kept for 2024 / 2016 / 2011), INE tourist-dwelling
    statistic and net migration (municipality, broadcast to sections), MIVAU appraised value (municipality > 25k), Census 2021 dwelling use and
    district-level tenure / year built / floor-area / building shares, and for the Basque provinces Eustat income and dwellings per section plus
    EMAL rents per municipality. Each source is optional: a failure prints and keeps the rest.
    """
    import pandas as pd

    from pycensus.census.countries.spain import serpavi, valor_tasado
    from pycensus.census.countries.spain.ine_tables import load_migration_balance, load_tourist_dwellings

    if not len(result) or level not in ("section", "municipality"):
        return result
    provinces = sorted({str(g)[:2] for g in result["GEOID"]})
    mun_key = result["GEOID"].astype(str).str[:5]
    new_cols: list = []

    def attach(frame, mapping, on_mun=False):
        nonlocal result
        keep = frame[["GEOID", *[c for c in mapping if c in frame.columns]]].rename(columns=mapping)
        if on_mun:
            keep = keep.rename(columns={"GEOID": "_mun"})
            result = result.assign(_mun=mun_key.to_numpy()).merge(keep, on="_mun", how="left").drop(columns="_mun")
        else:
            result = result.merge(keep, on="GEOID", how="left")
        new_cols.extend(c for c in mapping.values() if c in result.columns)

    try:  # SERPAVI rent
        sp = serpavi.load_attributes(provinces, level, data_dir)
        mapping = {}
        for y in (2024, 2016, 2011):
            mapping[f"serpavi_rent_m2_median_vc_{y}"] = f"ine_serpavi_rent_m2_median_vc_{y}"
        for y in (2024, 2016):
            mapping[f"serpavi_rent_total_median_vc_{y}"] = f"ine_serpavi_rent_total_median_vc_{y}"
        for name in ("rent_m2_median_vu_2024", "rent_m2_p25_vc_2024", "rent_m2_p75_vc_2024", "area_m2_median_vc_2024", "n_vc_2024", "rent_m2_latest"):
            mapping[f"serpavi_{name}"] = f"ine_serpavi_{name}"
        attach(sp, mapping)
    except Exception as exc:  # pragma: no cover
        print(f"[pipeline] skipping SERPAVI rent join: {exc}")
    try:  # tourist dwellings, net migration, appraised value (municipality natives)
        tour = load_tourist_dwellings(provinces, data_dir)
        attach(tour, {f"ine_tourist_{k}_{t}": f"ine_tourist_{k}_{t}" for k in ("dwellings", "places", "dwellings_share") for t in ("2020M08", "2026M05")}, on_mun=True)
        mig = load_migration_balance(provinces, data_dir)
        attach(mig, {f"ine_netmig_{k}_2024": f"ine_netmig_{k}_2024" for k in ("total", "exterior", "interior")}, on_mun=True)
        vt = valor_tasado.load_attributes(data_dir)
        attach(vt, {f"valor_tasado_{k}_T2A{y}": f"ine_valor_tasado_{k}_eur_m2_{y}" for k in ("total", "new", "old") for y in (2026, 2016)}, on_mun=True)
    except Exception as exc:  # pragma: no cover
        print(f"[pipeline] skipping Spain municipal housing/tourism joins: {exc}")
    if level == "section":
        try:  # Census 2021 dwelling use + district-level tenure / year built / floor area / building type
            from pycensus.census.countries.spain.censo2021.housing import load_section_housing

            ch = load_section_housing(data_dir).drop(columns=["district", "household_size_mean"], errors="ignore")
            attach(ch, {c: f"ine_{c}" for c in ch.columns if c != "GEOID"})
        except Exception as exc:  # pragma: no cover
            print(f"[pipeline] skipping Census 2021 dwelling tenure join: {exc}")
    basque = [p for p in provinces if p in ("01", "20", "48")]
    if basque:
        try:
            from pycensus.census.countries.euskadi.eustat import housing_income as hi

            em = hi.load_emal_municipal(2025, data_dir, tuple(basque))
            em16 = hi.load_emal_municipal(2016, data_dir, tuple(basque)).rename(columns={"emalRentM2": "emalRentM2_2016"})
            em = em.merge(em16[["GEOID", "emalRentM2_2016"]], on="GEOID", how="left")
            attach(em, {"emalRentM2": "ine_emal_rent_m2_2025", "emalRentM2_2016": "ine_emal_rent_m2_2016", "emalRentMonthly": "ine_emal_rent_monthly_2025",
                        "emalDeposits": "ine_emal_deposits"}, on_mun=True)
            if level == "section":
                inc = hi.load_income_sections(tuple(basque), data_dir)
                attach(inc, {c: f"ine_eustat_{c}" for c in inc.columns if c.startswith("income")})
                if "20" in basque:
                    dw = hi.load_dwellings_sections(data_dir)
                    attach(dw, {c: f"ine_eustat_{c[:-3]}Share" if c.endswith("Pct") else f"ine_eustat_{c}" for c in dw.columns if c.startswith("dwelling")})
        except Exception as exc:  # pragma: no cover
            print(f"[pipeline] skipping Basque housing / income join: {exc}")
    if level == "section":   # latest national (Congreso 2023), municipal (2023) and regional (Parlamento Vasco 2024) elections per section
        try:
            from pycensus.census.countries.spain import elections as el

            for fn, kw in ((el.load_congreso_2023, {}), (el.load_municipal_2023, {}),
                           (el.load_parlamento_vasco_2024, {"provinces": basque})):
                if "provinces" in kw and not kw["provinces"]:
                    continue
                try:
                    t = fn(level="section", provinces=kw.get("provinces", provinces), data_dir=data_dir)
                    result = _merge_elections(result, t, SPAIN_KEEP_COLUMNS, "ine_")
                except Exception as exc:  # pragma: no cover
                    print(f"[pipeline] skipping Spain election {fn.__name__}: {exc}")
        except Exception as exc:  # pragma: no cover
            print(f"[pipeline] skipping Spain elections: {exc}")
    for c in new_cols:   # sources publishing 0-100 percentages -> 0-1 shares
        if c in result.columns and (c.startswith("ine_tourist_dwellings_share_") or (c.startswith("ine_eustat_dwellings") and c.endswith("Share"))):
            result[c] = pd.to_numeric(result[c], errors="coerce") / 100.0
    SPAIN_KEEP_COLUMNS.update(new_cols)
    _RATE_COLUMNS.update(c for c in new_cols if c not in _SPAIN_COUNT_EXTRAS)
    return result


# Eustat municipal indicators (real, live PXWeb "Banco de Datos" API,
# `pycensus.census.countries.euskadi.eustat.municipal_indicators` -- added
# 2026-08-25), layered ALONGSIDE the finer INE section-level join above, not
# replacing it. Eustat's own section-level population is measurably less
# accurate than INE's (see `euskadi/api.py`'s module docstring), but these
# three fields (business establishments, births, deaths) simply don't exist
# in either INE source at any resolution, so joining them at their real
# native municipality resolution is a genuine addition, not a downgrade.
# Joined by 5-digit municipality `GEOID` -- confirmed the same INE/LAU code
# space `spain.ine`/`spain.censo2021` section `GEOID[:5]` and
# `euskadi.geography`'s GISCO municipality boundaries already use (Donostia/
# San Sebastian = "20069" in both). Landed under `"eustat_"` (this pipeline's
# existing `EUSKADI_KEEP_COLUMNS` GISCO-population join also uses that
# prefix, for Euskadi-routed cities -- Gipuzkoa itself routes through
# `country == "ESP"`, not `census_module == "euskadi"`, so there is no
# actual column collision for Gipuzkoa's own map).
EUSTAT_MUNICIPAL_KEEP_COLUMNS = {
    "eustat_establishments",
    "eustat_birthsPeriod2021_2025",
    "eustat_deathsPeriod2021_2025",
}

EUSTAT_MUNICIPAL_LEVELS: tuple[str, ...] = ("municipality",)


def _eustat_municipal_loader(aoi, states, level, cache_dir):
    """Adapt `pycensus.census.countries.euskadi.eustat.municipal_indicators`' fetchers to `_join_polygon_stats`'s signature.

    Returns a municipality-polygon GeoDataFrame (real GISCO LAU boundaries,
    same source `euskadi.eustat.loader`/`euskadi.geography` already use)
    with the three real Eustat BDE indicator columns merged on, so
    `_join_polygon_stats`'s ordinary population-share apportionment logic
    applies unchanged. `states` is ignored (no US-style states concept here,
    same as `_euskadi_census_loader`); territories are derived from `aoi`.
    """
    import geopandas as gpd
    import pandas as pd

    from pycensus.census.countries.euskadi import geography as euskadi_geography
    from pycensus.census.countries.euskadi.eustat import municipal_indicators

    if level != "municipality":
        return gpd.GeoDataFrame({"GEOID": []}, geometry=gpd.GeoSeries([], crs=4326), crs=4326)

    territory_codes = euskadi_geography.territories_in_aoi(aoi, cache_dir=cache_dir)
    if not territory_codes:
        return gpd.GeoDataFrame({"GEOID": []}, geometry=gpd.GeoSeries([], crs=4326), crs=4326)

    boundaries = euskadi_geography.load_boundaries_multi(
        "municipality", regions=territory_codes, aoi=aoi, cache_dir=cache_dir
    )
    indicator_frames = [municipal_indicators.load_municipal_indicators(terr) for terr in territory_codes]
    indicators = pd.concat(indicator_frames, ignore_index=True) if indicator_frames else pd.DataFrame({"GEOID": []})
    # More Eustat municipal tables (age/sex, birthplace, nationality, education, activity, households, tenure, income): bare names for
    # pycensus global-schema features, `eustat_` for the rest.
    from pycensus.canonical import global_feature_names

    glob = global_feature_names()
    extra_frames = [municipal_indicators.load_extra_indicators(terr) for terr in territory_codes]
    extra = pd.concat(extra_frames, ignore_index=True) if extra_frames else pd.DataFrame({"GEOID": []})
    extra = extra.rename(columns={c: (c if c in glob or c == "GEOID" else f"eustat_{c}") for c in extra.columns})
    indicators = indicators.merge(extra, on="GEOID", how="outer") if len(indicators) else extra

    merged = boundaries.merge(indicators, on="GEOID", how="left")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=boundaries.crs)


# Israel (CBS census, `pycensus.census.countries.israel.cbs`) -- real, verified data (see the
# schema-redesign memory doc's "Israel rebuild" round). `cbs.loader.load`'s own
# docstring/`IMPLEMENTED_LEVELS` (audited 2026-08-22) confirm BOTH
# `"locality"` and `"statistical_area"` have a real, live-verified attribute
# source -- the same `SCHEMA` fields are fetched from a different ArcGIS
# FeatureServer layer at each level (`_ATTRIBUTE_SOURCE` in `loader.py`), and
# the module docstring cross-checks the two levels' totals for Beersheba
# (locality pop_approx=217580 vs. summed statistical_area pop_approx=217600,
# matching to rounding), so `statistical_area` is a genuinely finer level
# with real data, not just finer boundary geometry. Listed finest-first,
# same convention as `INEGI_LEVELS`. Prefix "cbs_"
# (`pycensus.census.countries.israel.cbs.schema.SCHEMA = ColumnSchema(prefix="cbs")`).
ISRAEL_KEEP_COLUMNS = {
    "cbs_population",
    "cbs_households",
    "cbs_populationDensity",
    "cbs_populationChangeSince2008",
    "cbs_sexRatio",
    "cbs_ageMedian",
    "cbs_populationShare019",
    "cbs_populationShare2064",
    "cbs_populationShare65plus",
    "cbs_dependencyRatio",
    "cbs_householdSizeMean",
    "cbs_employmentRate",
    "cbs_selfEmployedRate",
    "cbs_employeesMedianAnnualWage",
    "cbs_workOutsideLocalityRate",
    "cbs_academicDegreeRate",
    "cbs_householdsOwnerRate",
    "cbs_householdsRenterRate",
    "cbs_householdsNoVehicleRate",
    "cbs_households2plusVehicleRate",
    "cbs_foreignResidentRate",
    # Derived (round(Foreign_pcnt/100 * pop_approx)) -- CBS has no native
    # foreign-resident COUNT field, only the rate above; see
    # pycensus.census.countries.israel.schema.cbs_schema.json's
    # "foreignBornPopulation" feature and cbs/loader.py's `load()` for the
    # verified-consistent derivation. No real jobs-by-workplace source was
    # found for Israel (investigated 2026-08-23: CBS's own ArcGIS layers --
    # both census_2022_setl_all_2021 and the statistical-area layer -- were
    # enumerated live and have only residence-based employment fields
    # (Empl_pcnt, WrkOutLoc_pcnt, etc.), no workplace-location jobs count;
    # no live, geographically-broken-down National Insurance
    # Institute/Bituach Leumi workplace-jobs dataset was found either).
    "cbs_foreignBornPopulation",
    "cbs_institutionalizedResidentRate",
    "cbs_ageMedianMale",
    "cbs_ageMedianFemale",
    "cbs_laborForceParticipationRate",
    "cbs_workHoursMeanWeekly",
    "cbs_selfEmployedMedianAnnualWage",
    "cbs_topWageDecileEmployeesRate",
    "cbs_householdsWithYoungChildrenRate",
    "cbs_householdsWithYoungAdultsRate",
    "cbs_childrenBornMean",
    "cbs_householdsWithParkingRate",
    # Derived from CBS's per-area dominant-religion classification code
    # (see cbs_schema.json's "religionCode" feature and cbs/loader.py's
    # `load()`) -- real, live-verified population counts, not fabricated.
    # See `SHARE_COLUMNS`' cbs_jewish_share/cbs_muslim_share/etc. entries.
    "cbs_jewishPopulation",
    "cbs_muslimPopulation",
    "cbs_christianPopulation",
    "cbs_druzePopulation",
    "cbs_otherReligionPopulation",
    # Derived from CBS's per-area "degree of religiosity" classification
    # code (secular/traditional/religious/Haredi -- see cbs_schema.json's
    # "religiosityCode" feature and cbs/loader.py's `load()`). Real,
    # live-verified counts.
    "cbs_secularPopulation",
    "cbs_traditionalReligiosityPopulation",
    "cbs_religiousObservantPopulation",
    "cbs_haredimPopulation",
    "cbs_otherReligiosityPopulation",
}
ISRAEL_KEEP_COLUMNS = set(ISRAEL_KEEP_COLUMNS) | _schema_columns("israel", "cbs", "cbs_")

ISRAEL_LEVELS: tuple[str, ...] = ("statistical_area", "locality")


ISRAEL_COMMUTE_KEEP_COLUMNS: set = set()   # filled from the schema below (every feature of pycensus israel/cbs_commute)


def _israel_commute_loader(aoi, states, level, cache_dir):
    """Census 2022 means of transport to work (`pycensus.census.countries.israel.cbs_commute`), one level: `commute_area`."""
    from pycensus.census.countries.israel import cbs_commute

    return cbs_commute.load(aoi=aoi, level=level, cache_dir=str(cache_dir))


# Sub-municipal transit use from local travel surveys (2026-11-20, the finest published unit of each): Guadalajara EOD 2023 per AGEB of
# residence, Donostia neighbourhood sheets of the PMUS, Taipei districts (city DOT survey 2018), Hamburg Bezirke (MiD 2023).
IMEPLAN_EOD_KEEP_COLUMNS: set = set()
DONOSTIA_PMUS_KEEP_COLUMNS: set = set()
TAIPEI_MODE_USE_KEEP_COLUMNS: set = set()
MID_HAMBURG_KEEP_COLUMNS: set = set()
EM_CAPV_KEEP_COLUMNS: set = set()


def _em_capv_loader(aoi, states, level, cache_dir):
    """Basque mobility study 2021: trips by mode per comarca of origin, capitals apart (`pycensus...spain.mobility.em_capv`), ``emcapv_``."""
    from pycensus.census.countries.spain.mobility import em_capv

    return em_capv.load(aoi=aoi, cache_dir=str(cache_dir))


def _empty_units():
    return gpd.GeoDataFrame({"GEOID": []}, geometry=[], crs=4326)


def _imeplan_eod_loader(aoi, states, level, cache_dir):
    """Guadalajara EOD 2023 commuters by mode per AGEB of residence, then per survey microzone and centralidad for the AGEBs the survey did not
    sample (`pycensus...mexico.imeplan_eod`); empty outside its survey area."""
    from pycensus.census.countries.mexico import imeplan_eod

    return imeplan_eod.load(aoi=aoi, level=level, cache_dir=str(cache_dir), data_dir=str(Path(cache_dir) / "mexico"))


def _donostia_pmus_loader(aoi, states, level, cache_dir):
    """Donostia trips by mode per neighbourhood (`pycensus...spain.mobility.donostia_pmus`), prefixed ``donpmus_``."""
    from pycensus.census.countries.spain.mobility import donostia_pmus

    return donostia_pmus.load(aoi=aoi, cache_dir=str(cache_dir))


def _taipei_mode_use_loader(aoi, states, level, cache_dir):
    """Mode shares, prefixed ``tpemode_``: ``district`` = Taipei City's 12 districts (city DOT survey 2018, `taiwan.taipei_mode_use`);
    ``traffic_zone`` = New Taipei's 7 traffic zones (MOTC survey 2010 as tabulated by the Taipei DOT, `taiwan.shuangbei_commute`), for the
    cells outside Taipei City."""
    from pycensus.census.countries.taiwan import shuangbei_commute, taipei_mode_use

    if level == "traffic_zone":
        g = shuangbei_commute.load(cache_dir=str(cache_dir), data_dir=str(Path(cache_dir) / "taiwan"))
        g = g[g["county"] == "New Taipei City"].rename(columns={"sample_trips": "sample_commute_trips"})
    else:
        g = taipei_mode_use.load(cache_dir=str(cache_dir), data_dir=str(Path(cache_dir) / "taiwan"))
    g = g[g.intersects(aoi.to_crs(4326).union_all())] if aoi is not None else g
    keep = [c for c in g.columns if c not in ("GEOID", "geometry", "name", "year", "sample_residents", "zone", "districts", "county")]
    return g[["GEOID", "geometry", *keep]].rename(columns={c: "tpemode_" + c for c in keep}).reset_index(drop=True)


def _mid_hamburg_loader(aoi, states, level, cache_dir):
    """Trip shares by main mode, prefixed ``midhh_``: ``bezirk`` = Hamburg's 7 Bezirke (MiD 2023), ``kreis`` = the Kreise of the
    Metropolregion around it (MiD 2017 regional report) for the cells outside the city."""
    from pycensus.census.countries.germany import mid_hamburg

    g = mid_hamburg.kreise(data_dir=str(Path(cache_dir) / "germany")) if level == "kreis" else mid_hamburg.bezirke(cache_dir=str(cache_dir))
    g = g[g.intersects(aoi.to_crs(4326).union_all())] if aoi is not None else g
    keep = [c for c in g.columns if c.endswith("_share")]
    return g[["GEOID", "geometry", *keep]].rename(columns={c: "midhh_" + c for c in keep}).reset_index(drop=True)


def _join_survey_units(h3_grid, aoi, states, cache_dir, levels, loader, keep, prefix, label):
    """One travel-survey source on its own zoning (``levels``: one level or a finest-first tuple); skipped (map unchanged) when it fails or
    has no unit in the AOI."""
    try:
        return _join_polygon_stats(h3_grid, aoi, states, (levels,) if isinstance(levels, str) else tuple(levels), cache_dir, loader, keep, prefix)
    except Exception as exc:  # pragma: no cover - network/data availability varies
        print(f"[pipeline] skipping {label} join: {exc}")
        return h3_grid


def _israel_census_loader(aoi, states, level, cache_dir):
    """Adapt `pycensus.census.countries.israel.cbs.load`'s signature to `_join_polygon_stats`'s.

    Israel's `cbs.load` takes `localities`/`regions` (CBS/LAMAS locality
    code), not `states` -- derived from `aoi` when omitted, so `states` is
    ignored here (same pattern as `_euskadi_census_loader`/`_spain_census_loader`).
    """
    from pycensus.census.countries.israel import cbs

    gdf = _restore_population_prefix(cbs.load(aoi=aoi, level=level, cache_dir=str(cache_dir)), "cbs")
    if level == "statistical_area":   # Knesset 25 (2022) per statistical area (ballot boxes -> H3 -> areas; box locations geocoded from polling places)
        try:
            from pycensus.census.countries.israel.elections import load_knesset

            localities = sorted({int(str(g)[:4]) for g in gdf["GEOID"]})[:50]
            import pandas as pd

            parts = []
            for loc in localities:   # one locality at a time: a locality without ballot-box data must not sink the others
                try:
                    parts.append(load_knesset(25, localities=(loc,), data_dir=str(Path(cache_dir) / "israel_elections")).census)
                except Exception as exc:  # pragma: no cover
                    print(f"[pipeline] Israel election: locality {loc} skipped: {str(exc)[:80]}")
            t = pd.concat(parts)
            gdf = _merge_elections(gdf, t, ISRAEL_KEEP_COLUMNS, "cbs_")
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping Israel election join: {exc}")
        # statistical areas exist only in the larger localities: each locality minus its statistical areas becomes a remainder area
        # with the locality's counts minus theirs (`pycensus.remainder`), so this level covers every locality
        try:
            from pycensus.remainder import add_remainders

            loc = _restore_population_prefix(cbs.load(aoi=aoi, level="locality", cache_dir=str(cache_dir)), "cbs")
            counts = [c for c in gdf.columns if c not in ("GEOID", "geometry") and not _is_rate_col(c if c.startswith("cbs_") else f"cbs_{c}")
                      and pd.api.types.is_numeric_dtype(gdf[c])]
            n0 = len(gdf)
            gdf = add_remainders(gdf, loc, counts)
            print(f"[pipeline] Israel statistical areas: {n0} + {len(gdf) - n0} locality remainders")
        except Exception as exc:  # pragma: no cover
            print(f"[pipeline] Israel locality remainders not added: {type(exc).__name__}: {str(exc)[:120]}")
    if level == "locality" and len(gdf):   # Knesset 25 summed per locality: the localities without statistical areas (2026-11-20)
        try:
            from pycensus.census.countries.israel.elections import load_knesset_localities

            t = load_knesset_localities(25, localities=sorted({int(str(g)) for g in gdf["GEOID"] if str(g).isdigit()}),
                                        data_dir=str(Path(cache_dir) / "israel_elections"))
            gdf = _merge_elections(gdf, t, ISRAEL_KEEP_COLUMNS, "cbs_")
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping Israel locality election join: {exc}")
    return gdf


# Canada (StatCan, `pycensus.census.countries.canada.statcan`) -- real, verified data
# (see that module's own docstrings/tests: nationwide sums cross-checked
# against StatCan's own published 98100002.csv figures). `population`/
# `households`/`dwellings` are nationwide (all levels, all provinces).
# `householdSizeMean`/`medianHouseholdIncome`/`meanHouseholdIncome`/
# `ownerHouseholds`/`renterHouseholds`/education-breakdown/commute-mode
# fields (added 2026-08-23, catalogue 98-401-X2021006) are
# dissemination-area-level, ONTARIO ONLY -- NaN for every other province,
# never fabricated.
# Verified real values match `SCHEMA.prefixed(name)`/the reconstructed
# `keep_col` convention this join uses (see `_join_polygon_stats`'s
# `keep_col = col if col.startswith(prefix) else f"{prefix}{col}"`), not
# guessed.
CANADA_KEEP_COLUMNS = {
    "statcan_population",
    "statcan_households",
    "statcan_dwellings",
    "statcan_householdSizeMean",
    "statcan_medianHouseholdIncome",
    "statcan_meanHouseholdIncome",
    "statcan_ownerHouseholds",
    "statcan_renterHouseholds",
    "statcan_educationTotalPopulation",
    "statcan_educationPrimaryOrLessPopulation",
    "statcan_educationSecondaryPopulation",
    "statcan_educationOtherTertiaryPopulation",
    "statcan_educationUniversityPopulation",
    "statcan_commuteTotalPopulation",
    "statcan_carCommuters",
    "statcan_publicTransportCommuters",
    "statcan_walkCommuters",
    "statcan_bikeCommuters",
    # Real immigrant/foreign-born population count, dissemination-area
    # level, Ontario only (98-401-X2021006, CHARACTERISTIC_NAME
    # "Immigrants") -- live-verified 2026-08-23: Canada-wide total
    # 8,361,505 (23.0%) exactly matches StatCan's own published 2021
    # Census headline immigrant count/share.
    "statcan_foreignBornPopulation",
    # Real workplace-located jobs count -- Canada's analog to the USA's
    # LODES WAC (see LODES_KEEP_COLUMNS), municipality (CSD) level, ALL
    # of Canada (not Ontario-only). Sourced from StatCan PID 98-10-0459
    # ("Commuting flow from geography of residence to geography of
    # work"), summed per workplace CSD -- live-verified 2026-08-23:
    # Toronto=753,185, Montreal=649,595, Calgary=370,125, Ottawa=272,920,
    # national total 10,680,335 within ~1.5% of the companion table
    # 98-10-0456's published national "Usual place of work" total. This
    # measure EXCLUDES work-from-home/no-fixed-workplace-address workers
    # (a large share in the pandemic-affected 2021 Census), so it
    # UNDER-counts true total jobs-by-workplace vs. a LODES-style
    # complete jobs census -- not directly comparable to
    # lodes_wac_jobsTotal without accounting for that gap. See
    # pycensus.census.countries.canada.constants.WORKPLACE_JOBS_PID's docstring.
    "statcan_jobsTotal",
}

# Real attribute data is wired for all 4 native levels (province/municipality/
# dissemination-area/dissemination-block); listed finest-first for
# `_join_polygon_stats`'s coarser-fills-gaps backfill loop, same convention
# as `INEGI_LEVELS`. Dissemination-block (StatCan's actual finest unit) was
# live-verified 2026-08-22 -- real per-block population/dwelling counts
# from the 2021 Geographic Attribute File (catalogue 92-151-X), not merely
# a populated/unpopulated flag; block-level sums reproduce the DA-level
# PID 98-10-0015 totals exactly for all 3,743 real Toronto DAs (see
# pycensus.census.countries.canada.statcan.loader module docstring).
CANADA_LEVELS: tuple[str, ...] = ("dissemination-block", "dissemination-area", "municipality", "province")


def _canada_census_loader(aoi, states, level, cache_dir):
    """Adapt `pycensus.census.countries.canada.statcan.load`'s signature to `_join_polygon_stats`'s.

    Canada's `statcan.load` takes `provinces`, not `states` -- derived from
    `aoi` when omitted, so `states` is ignored here (same pattern as
    `_euskadi_census_loader`/`_israel_census_loader`).
    """
    from pycensus.census.countries.canada import statcan

    gdf = statcan.load(aoi=aoi, level=level, cache_dir=str(cache_dir), data_dir=str(Path(cache_dir) / "canada"))
    data_cols = [c for c in gdf.columns if c not in ("GEOID", "geometry")]
    if gdf.empty or not data_cols or not gdf[data_cols].notna().any().any():
        # StatCan's tables are unreachable (outage page): boundaries without any census value are no census level at all.
        raise ValueError(f"StatCan returned no census values for level {level!r}")
    return _restore_population_prefix(gdf, "statcan")


# Andorra (`pycensus.census.countries.andorra.estadisticaad`) -- real, live-verified
# data (module docstring: nationwide `pob_2023` cross-checked against
# Departament d'Estadistica's own published figure). Only real field is
# `population`, already canonical.
# estadisticaad_buildingCount / estadisticaad_activeBusinessCount added
# 2026-08-23 after live-verifying two more real, keyless, parish-level
# ArcGIS FeatureServer sources on estadistica.ad's Esri Hub portal
# (buildings: "Edificis 2019", 21,529 real building footprints, single
# vintage year 2019; active businesses: "REE_Ubicacio" business register,
# 8,142 real active businesses, live current-state snapshot) -- see
# pycensus.census.countries.andorra.api / estadisticaad.loader module docstrings
# for the exact URLs and verification numbers. Everything else from the
# outside-source lead (A047 income, A061/A062 building/dwelling
# characteristics beyond footprint counts, A231 territorial register,
# births/deaths, employment/unemployment, social-security affiliation,
# households) was checked against estadistica.ad's real portal search API
# and found to be either not discoverable as a queryable table, or (income)
# plausibly national-only -- not wired in, see the andorra estadisticaad
# README for the per-category breakdown.
#
# `foreignBornPopulation` added 2026-08-23: real, parish-level, but sourced
# from a bulk PDF bulletin (`api.fetch_population_by_nationality_and_parish`),
# not a FeatureServer -- the Departament d'Estadistica's regular "Nota de
# premsa" bulletin (section 2.2 "Poblacio per nacionalitat") is the only
# real, sourced place a nationality/foreign-resident breakdown was found for
# Andorra, and it is nationality-based (Estrangera = non-Andorran
# nationality), not literal place-of-birth. Verified live 2026-08-23:
# nationwide 48,008/87,486 = 54.9% foreign, matching Andorra's well-
# documented ~45-55% foreign-resident share. A single dated snapshot
# (28-Feb-2025), returned regardless of requested year, same as
# buildingCount/activeBusinessCount. Jobs-by-workplace was investigated the
# same night (REE_Ubicacio business register field-checked live for an
# employee-count field -- none exists; CASS re-checked for a parish-level
# affiliate table -- none found) and is NOT wired, see the README.
ANDORRA_KEEP_COLUMNS = {
    "estadisticaad_population",
    "estadisticaad_buildingCount",
    "estadisticaad_activeBusinessCount",
    "estadisticaad_foreignBornPopulation",
}

ANDORRA_LEVELS: tuple[str, ...] = ("parish", "nation")


def _election_columns(df, keep: set) -> list:
    """Election columns worth mapping (votes total, turnout, left/right/other share, per-party shares); registered in `keep`."""
    cols = [c for c in df.columns if c.startswith("elec_") and (c.endswith(("_total_votes", "_turnout_rate", "_left_share", "_right_share", "_other_share"))
                                                              or ("_party_" in c and c.endswith("_share")))]
    keep.update(cols)
    return cols


def _merge_elections(gdf, t, keep: set, prefix: str):
    """Merge a loader's election frame onto `gdf` by GEOID, naming the columns `<prefix>elec_*` so the source join keeps them
    (`_rename_canonical_columns` strips the prefix again: the election columns are bare and comparable across countries)."""
    if "GEOID" not in t.columns:
        t = t.reset_index()
    cols = _election_columns(t, set())
    ren = {c: prefix + c for c in cols}
    keep.update(ren.values())
    _RATE_COLUMNS.update(c for c in cols if not c.endswith("_total_votes"))
    _RATE_COLUMNS.update(v for k, v in ren.items() if not k.endswith("_total_votes"))
    return gdf.merge(t[["GEOID", *cols]].rename(columns=ren), on="GEOID", how="left")


GERMANY_ELECTION_KEEP: set = set()


def _hamburg_elections_loader(aoi, states, level, cache_dir):
    """`_join_polygon_stats`-shaped loader: Hamburg Wahlbezirk polygons with the latest Bundestag, Buergerschaft and Bezirksversammlung results."""
    from pycensus.census.countries.germany import elections as el
    from pycensus.census.countries.germany.elections.loader import load_wahlbezirke

    data_dir = str(Path(cache_dir) / "germany")
    out = load_wahlbezirke(data_dir)
    for fn in (el.load_bundestag_2025, el.load_buergerschaft_2025, el.load_bezirksversammlung_2024):
        try:
            t = fn(level="wahlbezirk", data_dir=data_dir)
            out = out.merge(t[["GEOID", *_election_columns(t, GERMANY_ELECTION_KEEP)]], on="GEOID", how="left")
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping Hamburg election {fn.__name__}: {exc}")
    return out


def _andorra_census_loader(aoi, states, level, cache_dir):
    """Adapt `pycensus.census.countries.andorra.estadisticaad.load`'s signature to `_join_polygon_stats`'s."""
    from pycensus.census.countries.andorra import estadisticaad

    gdf = estadisticaad.load(aoi=aoi, level=level, cache_dir=str(cache_dir))
    gdf = _restore_population_prefix(gdf, "estadisticaad")
    if level == "parish":   # Consell General 2023 and Comunals 2023 (parish = the finest published level)
        from pycensus.census.countries.andorra import elections as el

        for fn in (el.load_consell_general_2023, el.load_comunals_2023):
            try:
                t = fn(data_dir=str(Path(cache_dir) / "andorra"))
                gdf = _merge_elections(gdf, t, ANDORRA_KEEP_COLUMNS, "estadisticaad_")
            except Exception as exc:  # pragma: no cover
                print(f"[pipeline] skipping Andorra election {fn.__name__}: {exc}")
    return gdf


# Taiwan (MOI RIS, `pycensus.census.countries.taiwan.moi`) -- real, verified data
# (module docstring/tests: real MOI RIS API responses, real district codes).
# `households` (ODRP014, merged onto ODRP013 rows by district_code) and
# `minorityEthnicityPopulation` (ODRP013's real aborigine_plain_total_*/
# aborigine_mountain_total_* fields, summed) were added 2026-08-22 after
# live-verifying both against real, well-documented figures (Taipei ~1.08M
# households, ~19k indigenous residents; Hualien County ~94k indigenous
# residents -- see pycensus.census.countries.taiwan.api module docstring). All 5
# real fields already match global_schema.json canonical names.
TAIWAN_KEEP_COLUMNS = {
    "moi_population", "moi_malePopulation", "moi_femalePopulation",
    "moi_households", "moi_minorityEthnicityPopulation",
    # Age-bracket fields added 2026-08-23: real MOI RIS ODRP014 single-year
    # age columns (people_age_000_m/f .. people_age_100up_m/f, the top
    # bucket genuinely top-coded at 100+, not an approximation), aggregated
    # in `pycensus.census.countries.taiwan.moi.loader._add_derived_age_columns`.
    # `moi_under18Population`/`moi_adultPopulation`/`moi_over65Population`
    # are additive sums-of-sums (valid at any level); `moi_medianAge` is
    # computed once per requested level from that level's own summed age
    # distribution (never averaged from lower-level medians) and is a rate
    # column here (`_CENSUS_RATE_SUBSTRINGS` matches "median" -> broadcast,
    # not population-apportioned). Live-verified 2026-08-23 for Taipei City
    # (county_city "63000"): population=2,451,007, under18=356,134,
    # adult=2,094,873, over65=580,555 (23.7% share, plausible against
    # Taipei's well-documented, unusually old age structure), medianAge=47.0;
    # under18 + adult reproduces population exactly (no residual/fabricated
    # gap -- ODRP014's 202 age columns sum exactly to ODRP013's own
    # people_total for every village checked).
    "moi_under18Population", "moi_adultPopulation", "moi_over65Population", "moi_medianAge",
}

# `schema.json` lists village/district/county_city/nation. `village` (村里/li)
# was added 2026-08-22: OSM Overpass admin_level=9 relations carry MOI's own
# 11-digit district_code as their `nat_ref` tag directly, verified live to
# be an *exact* 456/456 match against MOI's own Taipei village list (see
# `pycensus.census.countries.taiwan.geography` module docstring / `constants.
# OVERPASS_API_URL` docstring) -- `pycensus.census.countries.taiwan.geography.
# load_boundaries`/`villages_in_aoi` re-verify this match (>=99% of MOI's
# live village list per county) on every fetch and raise loudly if it drops
# below threshold, rather than silently using unverified geometry. Only
# `nation` still has no real, verified boundary source (NLSC's official
# files all 403 from their TGOS file host) -- see
# `pycensus.census.countries.taiwan.geography.IMPLEMENTED_LEVELS`. Requesting
# "nation" directly would raise `ValueError`.
TAIWAN_LEVELS: tuple[str, ...] = ("village", "district", "county_city")


def _taiwan_census_loader(aoi, states, level, cache_dir):
    """Adapt `pycensus.census.countries.taiwan.moi.load`'s signature to `_join_polygon_stats`'s."""
    from pycensus.census.countries.taiwan import moi

    gdf = moi.load(aoi=aoi, level=level, cache_dir=str(cache_dir), data_dir=str(Path(cache_dir) / "taiwan"))
    gdf = _restore_population_prefix(gdf, "moi")
    gdf = _taiwan_realprice_extras(gdf, level, cache_dir)
    if level in ("village", "district", "county_city"):   # 2024 presidential + legislative, 2022 mayor + council (village-native)
        from pycensus.census.countries.taiwan import elections as el

        for fn, kw in ((el.load_presidential, 2024), (el.load_legislative, 2024), (el.load_mayor, 2022), (el.load_council, 2022)):
            try:
                # every county of the area (2026-11-21: Taipei City only before; the metro also covers New Taipei, Keelung, Taoyuan)
                t = fn(kw, counties=tuple(sorted({str(g)[:5] for g in gdf["GEOID"]})), level=level,
                       data_dir=str(Path(cache_dir) / "taiwan_elections"), with_h3=False).census
                gdf = _merge_elections(gdf, t, TAIWAN_KEEP_COLUMNS, "moi_")
            except Exception as exc:  # pragma: no cover - network/data availability varies
                print(f"[pipeline] skipping Taiwan election {fn.__name__}: {exc}")
    return gdf


def _taiwan_realprice_extras(gdf, level, cache_dir):
    """MOI real-price registry (sale / pre-sale price and rent per m2, native = district) joined onto every Taiwan level by GEOID prefix."""
    import pandas as pd

    from pycensus.census.countries.taiwan.moi.realprice import load_realprice

    width = {"village": 8, "district": 8, "county_city": 5}.get(level)
    if width is None or not len(gdf):
        return gdf
    # Registry county letters (a=Taipei, f=New Taipei, ...), derived from the loaded GEOIDs' 5-digit county codes via the package mapping when
    # available; Taipei and New Taipei are the cities of this study.
    from pycensus.census.countries.taiwan.constants import COUNTY_CODES as _CC
    from pycensus.census.countries.taiwan.moi.realprice import CITY_LETTERS

    names = {_CC.get(str(g)[:5]) for g in gdf["GEOID"]}   # every county of the area (Taipei + New Taipei only before 2026-11-21)
    letters = tuple(sorted(k for k, v in CITY_LETTERS.items() if v in names)) or ("a", "f")
    out = gdf
    new_cols: list = []
    for year in (2025, 2016, 2013):
        try:
            rp = load_realprice(year, "county_city" if level == "county_city" else "district", letters, str(Path(cache_dir) / "taiwan"))
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping Taiwan real-price {year}: {exc}")
            continue
        keep = ["realPriceSaleM2", "realPricePresaleM2", "realRentM2"] + (["realPriceSaleCount", "realPricePresaleCount", "realRentCount"] if year == 2025 else [])
        rp = rp[["GEOID", *[c for c in keep if c in rp.columns]]]
        rp = rp.rename(columns={c: f"moi_{c}_{year}" if c.endswith("M2") else f"moi_{c}" for c in rp.columns if c != "GEOID"})
        rp["_key"] = rp.pop("GEOID").astype(str).str[:width]
        out = out.assign(_key=out["GEOID"].astype(str).str[:width]).merge(rp, on="_key", how="left").drop(columns="_key")
        new_cols += [c for c in rp.columns if c != "_key"]
    TAIWAN_KEEP_COLUMNS.update(new_cols)
    _RATE_COLUMNS.update(c for c in new_cols if c.endswith(("M2_2025", "M2_2016", "M2_2013")))
    return out


# Chile (INE/DPA, `pycensus.census.countries.chile.ine`, prefix "ine_cl" -- distinct
# from Spain's "ine" to avoid a collision) -- real, verified data (module
# docstring/tests: nationwide sum cross-checked against Chile's published
# Censo 2017 total). All 6 real fields already match global_schema.json
# canonical names except `urbanPopulation`/`ruralPopulation` (no exact
# global-schema equivalent).
CHILE_KEEP_COLUMNS = {
    "ine_cl_population", "ine_cl_malePopulation", "ine_cl_femalePopulation",
    "ine_cl_urbanPopulation", "ine_cl_ruralPopulation", "ine_cl_households",
}

CHILE_LEVELS: tuple[str, ...] = ("distrito", "comuna", "provincia", "region")


def _chile_census_loader(aoi, states, level, cache_dir):
    """Adapt `pycensus.census.countries.chile.ine.load`'s signature to `_join_polygon_stats`'s."""
    from pycensus.census.countries.chile import ine

    gdf = ine.load(aoi=aoi, level=level, cache_dir=str(cache_dir), data_dir=str(Path(cache_dir) / "chile"))
    return _restore_population_prefix(gdf, "ine_cl")


CHILE_MANZANA_KEEP_COLUMNS = set()   # filled below from the pycensus chile ine_manzana schema


def _chile_manzana_loader(aoi, states, level, cache_dir):
    from pycensus.census.countries.chile.ine import manzana

    gdf = _restore_population_prefix(manzana.load(aoi=aoi, data_dir=str(Path(cache_dir) / "chile")), "ine_cl")
    try:   # 2025 presidential runoff (Servel preliminary; mesa -> local de votacion -> H3 -> manzana)
        from pycensus.census.countries.chile import elections as el

        wp_name = "chl_pop_2025_CN_100m_R2025A_v1.tif"
        wp = next((q for q in (Path(cache_dir) / "worldpop" / wp_name, Path(cache_dir).parent / "worldpop" / wp_name,
                               Path(cache_dir).parent.parent / "cache" / "worldpop" / wp_name, Path(cache_dir).parent / "cache" / "worldpop" / wp_name) if q.is_file()), Path(wp_name))
        t = el.load("national", level="manzana", aoi=aoi, census_geometry={"manzana": gdf[["GEOID", "geometry"]]}, cache_dir=str(Path(cache_dir) / "chile_elections"),
                    worldpop_path=str(wp) if wp.is_file() else None)
        gdf = _merge_elections(gdf, t, CHILE_MANZANA_KEEP_COLUMNS, "ine_cl_")
    except Exception as exc:  # pragma: no cover - network/data availability varies
        print(f"[pipeline] skipping Chile election join: {exc}")
    return gdf


# Chile Casen SAE comunal poverty (Ministerio de Desarrollo Social y Familia,
# `pycensus.census.countries.chile.casen`, prefix "casen_cl") -- real, government-
# published Small Area Estimation (Fay-Herriot model) comunal poverty rates,
# a genuinely distinct source from INE's Censo 2017 DPA data above (a real
# bulk .xlsx download, not the ArcGIS FeatureServer). Live-verified
# 2026-08-23 against the source's own published figures: comuna Santiago
# (13101) povertyRate=0.038648, Concepcion (08101) povertyRate=0.054067,
# Talcahuano (08110) povertyRate=0.050827; povertyRateMultidimensional
# Santiago=0.164654, Concepcion=0.114778. Comuna-only (no region/provincia/
# distrito table exists for this source) -- see
# `pycensus.census.countries.chile.casen.api` module docstring for the full
# provenance/verification notes, including what was actually tried for
# REDATAM (a live, session-stateful engine with no documented query API --
# not just "no API exists") and CASEN's Ministerio de Desarrollo Social
# publication as the real, verified alternative that was found instead.
CHILE_CASEN_KEEP_COLUMNS = {"casen_cl_povertyRate", "casen_cl_povertyRateMultidimensional"}

CHILE_CASEN_LEVELS: tuple[str, ...] = ("comuna",)


def _chile_casen_census_loader(aoi, states, level, cache_dir):
    """Adapt `pycensus.census.countries.chile.casen.load`'s signature to `_join_polygon_stats`'s."""
    from pycensus.census.countries.chile import casen

    return casen.load(aoi=aoi, level=level, cache_dir=str(cache_dir), data_dir=str(Path(cache_dir) / "chile"))


# Chile SII (Servicio de Impuestos Internos) comunal jobs-by-workplace
# (`pycensus.census.countries.chile.sii`, prefix "sii_cl") -- real, government-
# published dependent-worker (payroll job) counts attributed to the
# employer's registered comuna, i.e. jobs *located in* the comuna, not
# employed residents -- Chile's equivalent of USA's LODES WAC `jobsTotal`
# (see `LODES_KEEP_COLUMNS`/`_join_jobs` below). Source is a real, keyless
# bulk .xlsx download (`https://www.sii.cl/estadisticas/region/
# PUB_Reg_Com.xlsx`), comuna-only (SII's file has no finer geography).
# Live-verified 2026-08-23: comuna Concepcion (08101) = 154,065 jobs,
# comuna Santiago (13101) = 1,141,309 jobs -- both plausible given each
# comuna's known role as a regional/national job center (exceeding each
# comuna's own resident population). Data vintage is commercial year 2015
# (the file's most recent year for this series -- see
# `pycensus.census.countries.chile.sii.api` module docstring for why more recent
# years were not found), notably older than the Censo 2017 DPA population
# already wired via `CHILE_KEEP_COLUMNS`. Comuna-name matching against the
# real DPA boundary service is live (no hardcoded name table) and drops
# (rather than mismatches) comunas it can't confidently match -- 5 of
# Region Metropolitana's 52 comunas were dropped this way in verification
# (abbreviated/hyphenated SII names not yet handled: "EST CENTRAL",
# "TIL-TIL", "SAN PEDRO DE MELIPILLA", "P AGUIRRE CERDA", "SAN JOSE
# MAIPO") -- Region VIII (Bio-Bio, the Gran Concepcion study area) matched
# all 33/33 real comunas with no drops.
CHILE_SII_KEEP_COLUMNS = {"sii_cl_jobsTotal"}

CHILE_SII_LEVELS: tuple[str, ...] = ("comuna",)


def _chile_sii_census_loader(aoi, states, level, cache_dir):
    """Adapt `pycensus.census.countries.chile.sii.load`'s signature to `_join_polygon_stats`'s."""
    from pycensus.census.countries.chile import sii

    return sii.load(aoi=aoi, level=level, cache_dir=str(cache_dir), data_dir=str(Path(cache_dir) / "chile"))


# Germany (Destatis GENESIS-Online, `pycensus.census.countries.germany.destatis`)
# -- real, live-verified data (module docstring/tests: Germany 31.12.2022
# = 83,118,501, Bavaria 31.12.2022 = 13,105,221, both cross-checked against
# Destatis' own published figures). Only real field is `population`,
# already canonical (see `_join_polygon_stats`'s prefix-reconstruction
# docstring). `district` (Kreis/NUTS3) is now wired too -- a real
# AGS-to-NUTS3 crosswalk (germany/destatis/crosswalk.py) was built and
# live-verified 2026-08-22: all 400 real GENESIS district rows matched,
# and 6 districts across 6 different states (Muenchen, Stuttgart, Koeln,
# Hamburg, Berlin, Leipzig) were independently cross-checked against their
# real published Zensus 2022 populations. `municipality` (Gemeinde) is now
# wired too, live-verified 2026-08-22 -- sourced from Eurostat/GISCO's LAU
# dataset (not GENESIS, which has no Gemeinde-level table under statistic
# 12411), which carries a real population figure alongside the boundary.
# GISCO's LAU_ID for Germany IS the 8-digit AGS code, so no crosswalk is
# needed. Sum-to-parent check: Landkreis Starnberg's 15 real LAU
# municipalities (2021 vintage) summed to 136,610 vs. GENESIS' district
# total of 138,488 (31.12.2022) for the same Kreis -- +1.37% over ~1.5
# years, consistent with growth already verified at district level. Note
# the vintage caveat: LAU population is 2021 (registry-based), not the
# same 31.12.2022 Zensus-corrected vintage as district/state/nation.
# `destatis_malePopulation`/`destatis_femalePopulation` added 2026-08-22:
# real GENESIS "Bevoelkerung ... Geschlecht" sex-breakdown tables
# (12411-0003/0011/0016, nation/state/district -- see
# `pycensus.census.countries.germany.constants.GENESIS_SEX_TABLES`), live-verified
# to sum exactly to the already-wired `population` at every level (nation
# 40,919,705 + 42,198,796 = 83,118,501; Hamburg state/district
# 894,537 + 938,138 = 1,832,675). Not available for municipality (GISCO LAU
# carries no sex breakdown) -- absent there, not fabricated.
GERMANY_KEEP_COLUMNS = {
    "destatis_population",
    "destatis_malePopulation",
    "destatis_femalePopulation",
    # Real Auslaenderzentralregister-derived foreign-population figure
    # (statistic 12521, table 12521-0040) -- only wired at "district" level
    # (see pycensus.census.countries.germany.constants.GENESIS_FOREIGN_POPULATION_TABLES);
    # NaN at nation/state/municipality. Live-verified 2026-08-23: Muenchen
    # (Kreis AGS 09162), 31.12.2022 = 505,760.
    "destatis_foreignBornPopulation",
}

# Administrative (polygon-hierarchy) levels only -- used by `_join_census`'s
# h3-grid join, which goes through `_join_polygon_stats`'s
# centroid-in-polygon population apportionment. That apportionment logic is
# for a census polygon that's typically much bigger than one h3 cell (a
# municipality, district, ...); it's the wrong tool for the real Zensus
# *grid* (already H3-shaped 1:1 by `_germany_grid_census_loader` below, see
# `GERMANY_LEVELS`), so "grid" is deliberately excluded here.
GERMANY_ADMIN_LEVELS: tuple[str, ...] = ("municipality", "district", "state", "nation")

# Every level `_census_loader_and_levels_for_map` exposes on the map's census
# dropdown for Germany -- `GERMANY_ADMIN_LEVELS` plus the real Zensus 2022
# 100m grid, resampled onto H3 res 10 (see `_germany_grid_census_loader`),
# as the new finest option. NOT used for `_join_census`'s h3-grid join (see
# `GERMANY_ADMIN_LEVELS` above) -- only for the map-geometry dispatch.
GERMANY_LEVELS: tuple[str, ...] = ("grid", *GERMANY_ADMIN_LEVELS)


def _germany_census_loader(aoi, states, level, cache_dir):
    """Adapt `pycensus.census.countries.germany.destatis.load`'s signature to `_join_polygon_stats`'s."""
    from pycensus.census.countries.germany.destatis import loader as destatis_loader

    gdf = destatis_loader.load(
        aoi=aoi, level=level, cache_dir=str(cache_dir), data_dir=str(Path(cache_dir) / "germany")
    )
    return _restore_population_prefix(gdf, "destatis")


# Native cell size of Destatis' Zensus 2022 population grid (see
# `pycensus.census.countries.germany.zensus_grid.loader`'s module docstring) --
# passed to `pycensus.raster_to_h3.resample_grid_to_h3` so the H3 resolution
# it picks (res 10, ~0.015 km^2/cell -- the closest real match without going
# coarser than one real 100m x 100m = 0.01 km^2 source cell) is derived from
# H3's own average-cell-area table, not hardcoded here.
GERMANY_ZENSUS_GRID_CELL_SIZE_M = 100.0


def _germany_grid_census_loader(aoi, states, level, cache_dir):
    """`grid` census level for Germany: the real Zensus 2022 100m grid, as its own native rectangles.

    2026-08-30 (explicit user request, live Hamburg map: "for the german
    census pixels I would like rectangle layer and not h3"): this used to
    resample the grid onto H3 res-10 via `pycensus.raster_to_h3.
    resample_grid_to_h3` first. Two real problems with that, both reported
    live on Hamburg: (1) `resample_grid_to_h3` assigns each 100m cell by its
    *centroid* to a single nearest H3 cell (see that function's own
    docstring) -- since one 100m Zensus cell (0.01 km^2) is smaller than a
    res-10 hexagon (~0.015 km^2) but *larger* than several finer cells this
    map's own h3 grid sometimes renders at, and because only H3 cells that
    happen to contain a source centroid are ever emitted at all (a plain
    `groupby` over assigned cells, nothing else), the result is a sparse,
    checkerboard-looking H3 layer with real coverage gaps rather than the
    grid's true full coverage; (2) resampling at all was never necessary --
    the Zensus grid's own native 100m x 100m squares are real, exact
    geometry, and `_census_geometries_with_score` (this loader's only
    caller) already knows how to render an arbitrary polygon level via
    `GeoHierarchy.add_level`/its own population-weighted `level_of_service`
    aggregation -- the exact same generic mechanism used for every other
    country's real census admin polygons. Handing it the raw rectangles
    directly needs no new primitive and sidesteps the resampling (and its
    bug) entirely for this level. `GERMANY_ZENSUS_GRID_CELL_SIZE_M`/
    `pycensus.raster_to_h3` are no longer used by this loader (kept for any
    other caller that still wants an H3-resampled Zensus population, e.g. a
    future coarser fallback) -- see `_join_zensus_grid_population` for how
    the h3 GRID's own `population` column now gets the (still H3-shaped,
    but now area-weighted rather than nearest-centroid) equivalent.

    Returns an empty GeoDataFrame (not an error) for an AOI outside Germany,
    matching every other loader's "no data for this AOI" convention.
    """
    from pycensus.census.countries.germany.zensus_grid import loader as zensus_loader

    grid = zensus_loader.load(aoi=aoi, data_dir=str(Path(cache_dir) / "zensus_grid_data"))
    if grid.empty:
        return gpd.GeoDataFrame({"GEOID": [], "population": []}, geometry=[], crs=4326)
    return grid


def _germany_map_census_loader(aoi, states, level, cache_dir):
    """`_census_loader_and_levels_for_map`'s Germany loader: dispatches `grid` vs. admin levels.

    The map-only dispatcher needs one single `(loader, levels)` pair per
    country (see `_census_loader_and_levels_for_map`), but Germany's `grid`
    level is architecturally different from its admin levels (a resampled
    raster, not a GENESIS/GISCO polygon fetch) and needs a different real
    loader (`_germany_grid_census_loader`) -- this picks between the two by
    `level` so the single-loader contract still holds from the caller's side.
    """
    if level == "grid":
        return _germany_grid_census_loader(aoi, states, level, cache_dir)
    return _germany_census_loader(aoi, states, level, cache_dir)


def _join_zensus_grid_population(
    h3_grid: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame, cache_dir: Path
) -> gpd.GeoDataFrame:
    """Replace WorldPop-modeled `population` with real Destatis Zensus 2022 grid population.

    Mirrors the shape of `population_and_access_to_h3` (WorldPop raster ->
    H3), but the source here is `pycensus.census.countries.germany.zensus_grid`'s
    real, official 100m x 100m Destatis Zensus 2022 population grid --
    architecturally a point/cell grid like WorldPop, not an administrative
    polygon hierarchy (see that module's docstring), so it's joined here
    rather than through `_join_polygon_stats`.

    For Germany specifically, the real Zensus grid is a strictly better
    population source than WorldPop's modeled/interpolated raster, so this
    *replaces* `population` (and recomputes `pop_density`) for German
    cities rather than keeping both as equally-weighted options -- the
    original WorldPop-derived figure is kept alongside under
    `population_worldpop` for comparison/fallback, never discarded.

    2026-08-30 fix (live user report, Hamburg): this used to reduce each
    real 100m Zensus cell to its own centroid and assign it to the
    *nearest* H3 hexagon (`gpd.sjoin_nearest`) -- a real, still-broken
    "one pixel gets assigned to one cell" centroid join, the same class of
    bug already fixed today for WorldPop's raster resampler
    (`geohierarchy.raster_resample.raster_to_h3_tiled`, via
    `worldpop_raster_to_h3`) but never ported here. A 100m Zensus cell
    (0.01 km^2) is comparable to or larger than many of this pipeline's own
    finer H3 resolutions, so a centroid join routinely (a) leaves whole H3
    cells with zero population whenever no Zensus centroid happens to land
    inside them even though the cell plainly overlaps one or more real
    populated Zensus squares, producing the sparse/incomplete "checkerboard"
    of populated cells reported live, and (b) can pile an entire 100m cell's
    population onto a single nearest hexagon even when that cell actually
    straddles several. Replaced with `geohierarchy`'s generic
    geometry-to-geometry resampler (`GeoHierarchy.add_vector_data` +
    `Sum(geoweighted=True)`), the same exact-area-weighted, full-coverage,
    mass-conserving approach as the WorldPop fix: every H3 cell that
    geometrically intersects a Zensus square receives that square's
    population split in proportion to the true overlap area, so every
    touched H3 cell is guaranteed to appear (no gaps) and the total is
    conserved by construction (every square's population is fully
    distributed, no double counting).

    Returns `h3_grid` unchanged (after logging) if the Zensus loader
    returns zero cells for `aoi` (e.g. AOI outside Germany).
    """
    from pycensus.census.countries.germany.zensus_grid import loader as zensus_loader

    grid = zensus_loader.load(aoi=aoi, data_dir=str(Path(cache_dir) / "zensus_grid_data"))
    if grid.empty:
        print("[pipeline] Zensus grid returned 0 cells for this AOI -- keeping WorldPop-derived population")
        return h3_grid

    grid = grid[["population", "geometry"]].copy()
    grid["population"] = grid["population"].astype(float)
    grid["_zensus_idx"] = grid.index

    # 2026-09-06 bug fix (live user report, Hamburg): real area-weighted
    # overlap (`Sum(geoweighted=True)`) gives every H3 cell that so much as
    # SLIVER-touches a real populated 100m Zensus square a nonzero share of
    # its population -- including a neighboring cell that is genuinely
    # unpopulated (a park, water, a road) but happens to share a few
    # centimeters of boundary with a populated square due to ordinary
    # geometry/reprojection precision, not a real population split. A small
    # negative buffer on each Zensus square (shrinking a 100m square by 1m
    # on every side, ~4% of its area) removes exactly these boundary-sliver
    # artifacts while barely touching genuine substantial overlaps -- H3
    # res-11 cells (~2,150 m2, smaller than one 100m square) that legitimately
    # sit mostly inside a populated square keep almost all of their real
    # overlap area, so their population share is essentially unchanged.
    grid["geometry"] = grid.to_crs(grid.estimate_utm_crs()).geometry.buffer(-1.0).to_crs(grid.crs)
    grid = grid[~grid.geometry.is_empty]

    hierarchy = GeoHierarchy(crs=h3_grid.crs)
    hierarchy.add_level(
        "_zensus_h3", h3_grid[["h3_cell", "geometry"]].copy(), id_col="h3_cell"
    )
    hierarchy.add_vector_data(
        grid,
        level="_zensus_h3",
        columns=["population"],
        agg=Sum(geoweighted=True),
        fill_null=0.0,
    )
    zensus_by_cell = (
        hierarchy["_zensus_h3"][["h3_cell", "population"]]
        .set_index("h3_cell")["population"]
    )

    h3_grid = h3_grid.copy()
    h3_grid["population_worldpop"] = h3_grid["population"]
    h3_grid["population"] = h3_grid["h3_cell"].map(zensus_by_cell).fillna(0.0)
    if "area_m2" in h3_grid.columns:
        h3_grid["pop_density"] = h3_grid["population"] / (h3_grid["area_m2"] / 1e6)

    # The other Zensus 2022 100 m tables (age bands, birth country, household size, religion): same exact-area-weighted split.
    try:
        extra = zensus_loader.load_extra(aoi=aoi, data_dir=str(Path(cache_dir) / "zensus_grid_data"))
        extra_cols = [c for c in extra.columns if c not in ("GEOID", "geometry")]
        extra = extra[~extra.geometry.is_empty].copy()
        extra["geometry"] = extra.to_crs(extra.estimate_utm_crs()).geometry.buffer(-1.0).to_crs(extra.crs)
        extra = extra[~extra.geometry.is_empty]
        extra[extra_cols] = extra[extra_cols].fillna(0.0).astype(float)
        hierarchy.add_vector_data(extra[extra_cols + ["geometry"]], level="_zensus_h3", columns=extra_cols, agg=Sum(geoweighted=True), fill_null=0.0)
        by_cell = hierarchy["_zensus_h3"].set_index("h3_cell")
        for c in extra_cols:
            h3_grid[f"destatis_{c}"] = h3_grid["h3_cell"].map(by_cell[c]).fillna(0.0).to_numpy()
        print(f"[pipeline] Zensus grid extra tables joined: {len(extra_cols)} columns")
    except Exception as exc:  # noqa: BLE001 -- the population replacement above must survive a failed extra download
        print(f"[pipeline] Zensus grid extra tables skipped: {type(exc).__name__}: {str(exc)[:120]}")

    # Hamburg's Sozialmonitoring (social status / dynamics class per Statistisches Gebiet, ~900 areas): the class of the area holding the H3 cell's centre
    try:
        from pycensus.census.countries.germany.hamburg_sozialmonitoring import loader as soz_loader

        soz = soz_loader.load(aoi=aoi, data_dir=str(Path(cache_dir) / "zensus_grid_data"))
        cents = gpd.GeoDataFrame(geometry=_h3_cell_centroids(h3_grid["h3_cell"]), crs=4326)
        hit = gpd.sjoin(cents, soz[["socialStatusIndex", "socialDynamicsIndex", "geometry"]], how="left", predicate="within")
        hit = hit[~hit.index.duplicated(keep="first")]
        h3_grid["socialStatusIndex"] = hit["socialStatusIndex"].reindex(range(len(h3_grid))).to_numpy()
        h3_grid["socialDynamicsIndex"] = hit["socialDynamicsIndex"].reindex(range(len(h3_grid))).to_numpy()
        print(f"[pipeline] Hamburg Sozialmonitoring joined: {int(h3_grid['socialStatusIndex'].notna().sum()):,} cells")
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] Hamburg Sozialmonitoring skipped: {type(exc).__name__}: {str(exc)[:120]}")

    total_zensus_grid = float(grid["population"].sum())
    total_on_h3 = float(h3_grid["population"].sum())
    total_worldpop = float(h3_grid["population_worldpop"].sum())
    print(
        f"[pipeline] Zensus grid -> H3 population: {len(grid):,} real 100m cells, "
        f"{total_zensus_grid:,.0f} people -> {total_on_h3:,.0f} on {len(h3_grid):,} H3 cells "
        f"(WorldPop-derived total was {total_worldpop:,.0f}); replacing `population` "
        "(WorldPop kept as `population_worldpop`)"
    )
    return h3_grid


def _join_census(
    h3_grid: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    states: tuple[str, ...],
    census_levels: tuple[str, ...],
    cache_dir: Path,
    country: str = "USA",
    census_module: str | None = None,
    skip: tuple[str, ...] = (),
) -> gpd.GeoDataFrame:
    """Join a curated set of national census population/housing attributes onto `h3_grid`.

    `skip` (USA): source families not to download or join -- "acs5" (ACS 5-year, with its block interpolation), "religion", "housing_history",
    "elections" (`CityConfig.census_skip`).

    Dispatches to the pyCensus module for `country` (an ISO3 code, matching
    `CityConfig.country`) instead of hardcoding USA -- see `ACS_KEEP_COLUMNS`
    for the USA column set and `INEGI_KEEP_COLUMNS` for Mexico's.
    `census_module`, when set (see `CityConfig.census_module`), overrides
    the country-level dispatch entirely -- e.g. Gipuzkoa passes
    `census_module="euskadi"` so it joins Basque-specific population data
    (`EUSKADI_KEEP_COLUMNS`) instead of falling into the generic-Spain
    "not yet supported" branch below. Countries (or overrides) with no
    wired-up pyCensus join (Germany/Israel/generic Spain as of this
    writing -- their `pycensus` submodules exist but were not
    audited/verified this round, see the schema-redesign memory doc) log a
    clear "not yet supported" message and return `h3_grid` unchanged rather
    than silently degrading to zero columns or crashing.
    """
    if census_module == "euskadi":
        return _join_polygon_stats(
            h3_grid, aoi, states, EUSKADI_LEVELS, cache_dir, _euskadi_census_loader, EUSKADI_KEEP_COLUMNS, "eustat_"
        )
    if country == "USA":
        from pycensus.census.countries.usa import acs5 as acs

        if "acs5" not in skip:
            h3_grid = _join_polygon_stats(
                h3_grid, aoi, states, census_levels, cache_dir, acs.load, ACS_KEEP_COLUMNS, "acs5_"
            )
        # Additionally interpolate ACS5's blockgroup-only count fields down
        # onto real DHC blocks (see `_interpolate_acs_to_dhc_blocks`) and
        # join those at block resolution too, under `_interpolated`-suffixed
        # column names so they're never confused with the real
        # blockgroup-resolution `acs5_*` columns just joined above.
        try:
            if "acs5" in skip:
                raise RuntimeError("skipped (not chosen)")
            h3_grid = _join_polygon_stats(
                h3_grid,
                aoi,
                states,
                ("block",),
                cache_dir,
                _acs_interpolated_census_loader,
                ACS_INTERPOLATED_KEEP_COLUMNS,
                "acs5_",
            )
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping ACS->DHC-block interpolation: {exc}")
        # Religion: 2020 U.S. Religion Census adherents per county (the Census Bureau asks no religion question).
        try:
            if "religion" in skip:
                raise RuntimeError("skipped (not chosen)")
            h3_grid = _join_polygon_stats(
                h3_grid, aoi, states, ("county",), cache_dir, _us_religion_loader, US_RELIGION_KEEP_COLUMNS, "usrc_"
            )
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping US religion join: {exc}")
        # Housing prices / rents / tenure at tract level (ACS now, ~2016, 2000; FHFA index; HUD rents).
        try:
            if "housing_history" in skip:
                raise RuntimeError("skipped (not chosen)")
            h3_grid = _join_polygon_stats(
                h3_grid, aoi, states, ("tract",), cache_dir, _us_housing_loader, US_HOUSING_KEEP_COLUMNS, "ushs_"
            )
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping US housing join: {exc}")
        try:   # precinct-resolution elections (resampled precinct -> H3 -> tract)
            if "elections" in skip:
                raise RuntimeError("skipped (not chosen)")
            h3_grid = _join_polygon_stats(
                h3_grid, aoi, states, ("tract",), cache_dir, _us_precinct_elections_loader, US_PRECINCT_ELECTION_KEEP, ""
            )
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping US precinct elections join: {exc}")
        # The last national election (2024 presidential, county results from MIT MEDSL) as county-level columns.
        try:
            if "elections" in skip:
                raise RuntimeError("skipped (not chosen)")
            h3_grid = _join_polygon_stats(
                h3_grid, aoi, states, ("county",), cache_dir, _us_last_election_loader, US_LAST_ELECTION_COLUMNS, ""
            )
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping last-election join: {exc}")
        return h3_grid
    if country == "MEX":
        h3_grid = _join_polygon_stats(
            h3_grid, aoi, states, INEGI_LEVELS, cache_dir, _mexico_census_loader, INEGI_KEEP_COLUMNS, "inegi_"
        )
        return _join_survey_units(h3_grid, aoi, states, cache_dir, ("ageb", "microzone", "centralidad"), _imeplan_eod_loader, IMEPLAN_EOD_KEEP_COLUMNS, "imepeod_",
                                  "Guadalajara EOD 2023")
    if country == "ESP":
        h3_grid = _join_polygon_stats(
            h3_grid, aoi, states, SPAIN_LEVELS, cache_dir, _spain_census_loader, SPAIN_KEEP_COLUMNS, "ine_"
        )
        try:
            h3_grid = _join_polygon_stats(
                h3_grid,
                aoi,
                states,
                EUSTAT_MUNICIPAL_LEVELS,
                cache_dir,
                _eustat_municipal_loader,
                EUSTAT_MUNICIPAL_KEEP_COLUMNS,
                "eustat_",
            )
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping Eustat municipal indicators join: {exc}")
        h3_grid = _join_survey_units(h3_grid, aoi, states, cache_dir, "auzoa", _donostia_pmus_loader, DONOSTIA_PMUS_KEEP_COLUMNS, "donpmus_",
                                     "Donostia PMUS neighbourhoods")
        return _join_survey_units(h3_grid, aoi, states, cache_dir, "eustat_zone", _em_capv_loader, EM_CAPV_KEEP_COLUMNS, "emcapv_",
                                  "Basque mobility study 2021 comarcas")
    if country == "ISR":
        h3_grid = _join_polygon_stats(
            h3_grid, aoi, states, ISRAEL_LEVELS, cache_dir, _israel_census_loader, ISRAEL_KEEP_COLUMNS, "cbs_"
        )
        # Census 2022 main means of transport to work per sub-quarter of the large cities / per locality elsewhere (CBS public-use file).
        try:
            h3_grid = _join_polygon_stats(
                h3_grid, aoi, states, ("commute_area",), cache_dir, _israel_commute_loader, ISRAEL_COMMUTE_KEEP_COLUMNS, "cbscommute_"
            )
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping Israel census mode-to-work join: {exc}")
        return h3_grid
    if country == "CAN":
        h3_grid = _join_polygon_stats(
            h3_grid, aoi, states, CANADA_LEVELS, cache_dir, _canada_census_loader, CANADA_KEEP_COLUMNS, "statcan_"
        )
        # City of Toronto neighbourhood profiles (full 2021 Census profile, 158 neighbourhoods): ~330 more columns, and the census
        # source that still works when statcan.gc.ca refuses the connection.
        try:
            h3_grid = _join_polygon_stats(
                h3_grid, aoi, states, _toronto_profile_levels(cache_dir), cache_dir, _toronto_profile_loader, TORONTO_PROFILE_KEEP_COLUMNS, "statcan_"
            )
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping Toronto neighbourhood profile join: {exc}")
        return h3_grid
    if country == "AND":
        return _join_polygon_stats(
            h3_grid, aoi, states, ANDORRA_LEVELS, cache_dir, _andorra_census_loader, ANDORRA_KEEP_COLUMNS, "estadisticaad_"
        )
    if country == "TWN":
        h3_grid = _join_polygon_stats(
            h3_grid, aoi, states, TAIWAN_LEVELS, cache_dir, _taiwan_census_loader, TAIWAN_KEEP_COLUMNS, "moi_"
        )
        return _join_survey_units(h3_grid, aoi, states, cache_dir, ("district", "traffic_zone"), _taipei_mode_use_loader, TAIPEI_MODE_USE_KEEP_COLUMNS, "tpemode_",
                                  "Taipei district mode use")
    if country == "CHL":
        h3_grid = _join_polygon_stats(
            h3_grid, aoi, states, CHILE_LEVELS, cache_dir, _chile_census_loader, CHILE_KEEP_COLUMNS, "ine_cl_"
        )
        # Censo 2017 at block (manzana) level: age bands, immigrants, indigenous people, households, dwelling types.
        try:
            h3_grid = _join_polygon_stats(
                h3_grid, aoi, states, ("manzana",), cache_dir, _chile_manzana_loader, CHILE_MANZANA_KEEP_COLUMNS, "ine_cl_"
            )
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping Chile manzana join: {exc}")
        try:
            h3_grid = _join_polygon_stats(
                h3_grid,
                aoi,
                states,
                CHILE_CASEN_LEVELS,
                cache_dir,
                _chile_casen_census_loader,
                CHILE_CASEN_KEEP_COLUMNS,
                "casen_cl_",
            )
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping Chile Casen SAE comunal poverty join: {exc}")
        try:
            h3_grid = _join_polygon_stats(
                h3_grid,
                aoi,
                states,
                CHILE_SII_LEVELS,
                cache_dir,
                _chile_sii_census_loader,
                CHILE_SII_KEEP_COLUMNS,
                "sii_cl_",
            )
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping Chile SII comunal jobs-by-workplace join: {exc}")
        return h3_grid
    if country == "DEU":
        # 2026-09-07 bug fix (live report -- "destatis population column and
        # worldpop population seem too similar"): `_join_polygon_stats`
        # apportions each admin polygon's real total down to h3 cells in
        # proportion to `h3_grid["population"]` at the moment it runs (see
        # that function's own `cell_population`/`poly_population` weighting)
        # -- running the Zensus grid override FIRST means that weight is
        # the REAL 100m grid population, not WorldPop's raw estimate, so
        # `destatis_population`'s spatial pattern is disaggregated using the
        # real official grid (and cells with zero real Zensus coverage get
        # zero/near-zero apportioned `destatis_population`, not a
        # WorldPop-driven guess) instead of just reproducing WorldPop's own
        # density pattern rescaled to Destatis's total.
        try:
            h3_grid = _join_zensus_grid_population(h3_grid, aoi, cache_dir)
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping Zensus grid population replacement: {exc}")
        h3_grid = _join_polygon_stats(
            h3_grid, aoi, states, GERMANY_ADMIN_LEVELS, cache_dir, _germany_census_loader, GERMANY_KEEP_COLUMNS, "destatis_"
        )
        try:   # latest national / state / municipal elections per Wahlbezirk
            h3_grid = _join_polygon_stats(
                h3_grid, aoi, states, ("wahlbezirk",), cache_dir, _hamburg_elections_loader, GERMANY_ELECTION_KEEP, ""
            )
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] skipping Hamburg elections join: {exc}")
        return _join_survey_units(h3_grid, aoi, states, cache_dir, ("bezirk", "kreis"), _mid_hamburg_loader, MID_HAMBURG_KEEP_COLUMNS, "midhh_",
                                  "Hamburg MiD Bezirke")
    print(
        f"[pipeline] census join not yet supported for country={country!r} "
        "(pycensus module exists but is unaudited/unwired for this study) -- skipping, 0 census columns"
    )
    return h3_grid


def _join_race(
    h3_grid: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    states: tuple[str, ...],
    cache_dir: Path,
    country: str = "USA",
) -> gpd.GeoDataFrame:
    """Join decennial DHC race/ethnicity counts onto `h3_grid` (see `DHC_KEEP_COLUMNS`).

    ACS has no race/ethnicity breakdown at all -- decennial DHC is the only
    source available via `pycensus`, and it's USA-only: no other country's
    `pycensus` module publishes anything shaped like US race/ethnicity
    categories, so this is a no-op (h3_grid returned unchanged, no `dhc_*`
    columns) for every `country != "USA"`.
    """
    if country != "USA":
        print(f"[pipeline] no race/ethnicity census source for country={country!r} -- skipping race join")
        return h3_grid
    from pycensus.census.countries.usa import dhc

    # "block" is real and live-verified: decennial DHC (a full count, not a
    # sample) publishes real attribute data down to block level -- unlike
    # ACS5, which stops at blockgroup (a real Census Bureau publication
    # limit, not a pyCensus gap) -- see `dhc/loader.py`'s own `load()`
    # docstring ("is available down to 'block'"). Finest-first, same
    # convention every other country's `*_LEVELS` tuple uses.
    return _join_polygon_stats(
        h3_grid, aoi, states, ("block", "blockgroup", "tract", "place", "county"), cache_dir, dhc.load, DHC_KEEP_COLUMNS, "dhc_"
    )


# LODES WAC (Workplace Area Characteristics) -- jobs *located in* the geometry,
# which is what "how much activity happens here" needs (LODES RAC and ACS
# B08301 both count workers by where they *live*, the wrong end of the commute
# for this). Only `C000`/total jobs is kept: the income/age breakdowns would
# add four more columns to every vector tile for no use in this study.
LODES_KEEP_COLUMNS = {"lodes_wac_jobsTotal"}

# LODES publishes down to block level; blockgroup is the finest level pyCensus
# aggregates it to and the finest this study's h3 grid can use meaningfully.
LODES_LEVELS: tuple[str, ...] = ("blockgroup", "tract", "county")

# BA (Bundesagentur fuer Arbeit) "Arbeitsmarkt kommunal" (AMK) -- Germany's
# real, independent LODES-WAC equivalent, built from mandatory
# social-security registrations (sozialversicherungspflichtige Beschaeftigung)
# rather than Destatis/GENESIS census-style data. Only the "am Arbeitsort"
# (workplace) total is kept -- see `pycensus.census.countries.germany.ba.loader`'s
# module docstring for the live verification (Hamburg 2025: 1,084,974
# workplace jobs) and for why AMK's real granularity is Gemeinde
# (municipality), not Kreis (district) as first assumed.
GERMANY_BA_KEEP_COLUMNS = {"ba_jobsTotal"}
GERMANY_BA_LEVELS: tuple[str, ...] = ("municipality", "district")

# BA "Gemeindedaten aus der Beschaeftigungsstatistik" -- a real, separate BA
# product from AMK above (found via
# https://statistik.arbeitsagentur.de/DE/Navigation/Footer/Top-Produkte/
# Gemeindedaten-sozialversicherungspflichtig-Beschaeftigter-Nav.html), also
# keyless. Its own workplace-jobs figure (`ba_pendler_pendlerJobsAtWorkplace`)
# matches AMK's `ba_jobsTotal` exactly for the same Stichtag (Hamburg 2025:
# both 1,084,974 -- see `pycensus.census.countries.germany.ba_pendler.loader`'s
# module docstring), so it's kept anyway ONLY for the fields AMK doesn't
# have: the real Wohnort/Arbeitsort commuter cross-tab (Einpendler/
# Auspendler/same-Gemeinde workers) needed for `commuter_in_share`/
# `local_worker_share` below -- `pendlerJobsAtWorkplace` itself is dropped
# here to avoid a redundant near-duplicate jobs column on every feature.
GERMANY_BA_PENDLER_KEEP_COLUMNS = {
    "ba_pendler_sameGemeindeWorkers",
    "ba_pendler_einpendler",
    "ba_pendler_auspendler",
    "ba_pendler_residentWorkers",
}


def _join_jobs(
    h3_grid: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    states: tuple[str, ...],
    cache_dir: Path,
    country: str = "USA",
) -> gpd.GeoDataFrame:
    """Join a workplace-jobs source onto `h3_grid` (see `LODES_KEEP_COLUMNS`/`GERMANY_BA_KEEP_COLUMNS`).

    Same centroid-in-polygon + population-apportionment machinery as the ACS
    and DHC joins. Jobs are a count, so they are apportioned (not broadcast)
    across the h3 cells of each source polygon, and the metro-wide sum is
    preserved. Failure to fetch is non-fatal: the study then simply has no
    jobs column, and `_add_pop_jobs_columns` degrades to a no-op.

    LEHD LODES (USA) and BA's AMK (Germany, `country == "DEU"`) are the only
    two countries with a real workplace-jobs-flow source wired up here --
    every other country prints a clear "no LODES-equivalent" message and
    returns `h3_grid` unchanged.
    """
    if country == "USA":
        from pycensus.census.countries.usa import lodes_wac as lodes

        def _loader(aoi, states, level, cache_dir):
            return lodes.load_wac(aoi=aoi, states=states, level=level, cache_dir=cache_dir)

        return _join_polygon_stats(
            h3_grid, aoi, states, LODES_LEVELS, cache_dir, _loader, LODES_KEEP_COLUMNS, "lodes_wac_"
        )
    if country == "DEU":
        from pycensus.census.countries.germany import ba, ba_pendler

        def _loader(aoi, states, level, cache_dir):
            return ba.load(
                aoi=aoi, regions=states, level=level, cache_dir=cache_dir, data_dir=str(Path(cache_dir) / "germany")
            )

        h3_grid = _join_polygon_stats(
            h3_grid, aoi, states, GERMANY_BA_LEVELS, cache_dir, _loader, GERMANY_BA_KEEP_COLUMNS, "ba_"
        )

        # 2026-08-25 audit: real BA Pendlerstatistik (Einpendler/Auspendler/
        # same-Gemeinde workers), see `GERMANY_BA_PENDLER_KEEP_COLUMNS` above
        # and `pycensus.census.countries.germany.ba_pendler.loader`. Non-fatal like
        # the AMK join above -- a failed fetch just leaves the study without
        # `commuter_in_share`/`local_worker_share` (see `SHARE_COLUMNS`).
        def _pendler_loader(aoi, states, level, cache_dir):
            return ba_pendler.load(
                aoi=aoi, regions=states, level=level, cache_dir=cache_dir, data_dir=str(Path(cache_dir) / "germany")
            )

        try:
            h3_grid = _join_polygon_stats(
                h3_grid, aoi, states, GERMANY_BA_LEVELS, cache_dir, _pendler_loader,
                GERMANY_BA_PENDLER_KEEP_COLUMNS, "ba_pendler_",
            )
        except Exception as exc:  # pragma: no cover - level/state availability varies
            print(f"[pipeline] no BA Pendlerstatistik for country={country!r}: {exc}")
        return h3_grid
    print(f"[pipeline] no LODES-equivalent jobs source for country={country!r} -- skipping jobs join")
    return h3_grid


def _chunked_h3_grid_join(join_fn, h3_grid, aoi, chunk_h3_resolution: int, *args, **kwargs):
    """Memory-bounded wrapper for `_join_census`/`_join_race`/`_join_jobs`-shaped functions.

    Those functions all share the same shape: `(h3_grid, aoi, states, ...,
    cache_dir, ...) -> h3_grid` (with census attributes joined on), and
    internally call `_join_polygon_stats`, whose peak memory scales with
    `h3_grid`'s row count (`cell_population = h3_grid["population"].to_numpy()`,
    a full centroid `sjoin` against every downloaded census polygon, ...) --
    exactly the same "whole metro area's h3 grid in memory for one join"
    shape of problem the isochrone chunking fixes for the street network
    (see `docs/H3_CHUNKED_PIPELINE_DESIGN.md`).

    This partitions `h3_grid` by each row's H3 ancestor cell at
    `chunk_h3_resolution` and calls the given `join_fn` once per partition
    against the SAME full `aoi` (so each chunk's census loader call still
    downloads/queries the real polygon data covering that chunk's cells --
    only the row-count-driven apportionment/sjoin arrays shrink, not the
    census polygons considered), then concatenates the per-chunk results.

    Correctness caveat (read before enabling for a level with large
    polygons): `_join_polygon_stats`'s population-based apportionment for a
    census polygon divides that polygon's total count across every h3 cell
    matched to it (`cell_population[h3_row]` summed per `poly_row`, i.e. per
    polygon). When a polygon straddles a chunk boundary, each chunk only
    "sees" its own subset of that polygon's h3 cells, so the two chunks each
    apportion the polygon's *full* count across only their own local subset
    -- overcounting: the polygon's total ends up divided among each side of
    the boundary independently rather than once across the true combined
    cell set. This is *exact* for a census level whose polygons are smaller
    than a typical chunk (true for "block"/"blockgroup"/most "tract"
    geometries relative to a res-4, ~1,770 km^2 chunk) but only
    *approximate* for a coarse level whose polygons can be chunk-scale or
    larger (e.g. "county"/"place"/"nation") -- those levels' apportioned
    counts should be spot-checked against the unchunked path before trusting
    them at Boston/Shanghai scale. Rate-style columns (income, commute time,
    ...) are broadcast, not apportioned, and are unaffected by this caveat.

    Args:
        join_fn: `_join_census`, `_join_race`, or `_join_jobs` (or anything
            matching their `(h3_grid, aoi, states, cache_dir, ...) ->
            h3_grid` shape) -- called unchanged, once per chunk.
        h3_grid: Full input grid (GeoDataFrame or Polars DataFrame with an
            `h3_cell` column).
        aoi: Forwarded to `join_fn` unchanged for every chunk (see caveat
            above -- this is NOT itself clipped per chunk).
        chunk_h3_resolution: H3 resolution to partition by
            (`StudyParams.isochrone_chunk_h3_resolution`).
        *args, **kwargs: Forwarded to `join_fn` after `h3_grid, aoi`.

    Returns:
        Concatenated per-chunk result, same type/columns `join_fn` itself
        returns.
    """
    is_polars = isinstance(h3_grid, pl.DataFrame)
    cells = h3_grid["h3_cell"].to_list() if is_polars else list(h3_grid["h3_cell"])
    chunk_ids = [h3.cell_to_parent(c, chunk_h3_resolution) for c in cells]

    if is_polars:
        tagged = h3_grid.with_columns(pl.Series("_chunk_parent", chunk_ids))
        parents = tagged["_chunk_parent"].unique().to_list()
        parts = []
        for p in parents:
            sub = tagged.filter(pl.col("_chunk_parent") == p).drop("_chunk_parent")
            result = join_fn(sub, aoi, *args, **kwargs)
            parts.append(result)
        return pl.concat(parts, how="vertical")
    else:
        tagged = h3_grid.copy()
        tagged["_chunk_parent"] = chunk_ids
        parts = []
        for p in tagged["_chunk_parent"].unique():
            sub = tagged[tagged["_chunk_parent"] == p].drop(columns="_chunk_parent")
            result = join_fn(sub, aoi, *args, **kwargs)
            parts.append(result)
        return pd.concat(parts, ignore_index=False) if not isinstance(h3_grid, gpd.GeoDataFrame) else gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=False), crs=h3_grid.crs
        )


def _worldpop_year_from_filename(worldpop_filename: str) -> int:
    """Extract the WorldPop dataset year from a `CityConfig.worldpop_filename` like `chn_pop_2025_CN_100m_R2025A_v1.tif`."""
    import re

    m = re.search(r"_pop_(\d{4})_", worldpop_filename)
    if not m:
        raise ValueError(
            f"Could not extract a 4-digit year from worldpop_filename={worldpop_filename!r} "
            "(expected the '..._pop_<year>_...' pattern used by every CityConfig.worldpop_filename)."
        )
    return int(m.group(1))


def _join_worldpop_global_schema(
    h3_table: pl.DataFrame,
    aoi: gpd.GeoDataFrame,
    resolution: int,
    year: int,
    cache_dir: Path,
) -> pl.DataFrame:
    """Join every non-`population` WorldPop `GLOBAL_SCHEMA_FEATURES` layer onto `h3_table`, under bare canonical names.

    Takes and returns a lightweight, geometry-free `pl.DataFrame` (the same
    shape `studytools.h3_population.population_and_access_to_h3` returns, i.e.
    `h3_cell` + `population` + `level_of_service`/access columns) -- NOT
    `_add_h3_grid`'s geometry-bearing `gpd.GeoDataFrame` -- and the caller
    (`run_city_study`) must call this BEFORE `_add_h3_grid`, not after.

    2026-08-31 OOM fix: every feature in this join goes through
    `worldpop_raster_to_h3`/`geohierarchy.raster_resample.raster_to_h3_tiled`,
    which builds its OWN H3 cell geometry internally straight from each
    downloaded raster's own extent (`raster_to_h3_tiled`'s `h3_cells(...)`
    call) -- it never reads or needs the caller's H3 grid geometry at all.
    This function previously took/returned `_add_h3_grid`'s real
    hexagon-polygon `GeoDataFrame` and was called AFTER it, meaning
    Shanghai's full 25.7M-cell grid worth of real `shapely` `Polygon`
    objects sat fully resident in memory for the entire duration of this
    join, even though nothing in this join ever used them. Confirmed via a
    real, watchdog-guarded repro against the real cached (smaller, 13M-row)
    `shanghai/results/metro/h3_grid.parquet`: just building the real
    `geopandas.GeoDataFrame` (via `shapely.from_wkb` on the geometry
    column) OOM-killed a 10GB memory cgroup at 13M rows -- well under
    Shanghai's real 25.7M-cell grid, which would need on the order of
    20-30+GB for its geometry column alone. That dwarfed both this join's
    own `reset_index` copy (removed, see below) and
    `raster_to_h3_tiled`'s per-tile worker overhead, and fully explains why
    the run died immediately on the very first feature. Moving this join to
    run on the lightweight, geometry-free table -- and building the real
    hexagon geometry only once, after this join and all other tabular joins
    for the non-census branch are done -- keeps that one big allocation
    from ever coexisting with this join's downloads/raster resampling.

    This also drops the old pandas `reset_index(drop=True)`/restore-index
    dance entirely (replaced by a plain polars left-`join`): that dance
    made an extra full copy of the (then geometry-bearing) frame for no
    functional reason -- a polars `join` needs no positional/index
    bookkeeping at all.

    Census substitute for cities with `CityConfig.uses_census == False`
    (Shanghai as of this writing -- see `CityConfig.uses_census`/
    `_census_supported`; there is no China census implementation in
    `pycensus`). Downloads each layer via
    `pycensus.census.worldwide.worldpop.loader.download_worldpop_layer`
    (see that module's docstring for exactly how each feature is derived --
    exact for `malePopulation`/`femalePopulation`/`over65Population`, a
    documented linear-within-bin approximation for `under18Population`/
    `adultPopulation`) and resamples each raster onto `h3_grid`'s H3 cells
    with the same `UrbanAccessAnalyzer.h3_ops.from_raster_centroid` machinery
    `population_and_access_to_h3` already uses for the base `population`
    raster.

    `population` itself is deliberately excluded from `GLOBAL_SCHEMA_FEATURES`
    here -- it is already on `h3_grid`, landed by `population_and_access_to_h3`
    from a *different*, independently-sourced WorldPop 'pop'-family raster
    (see `download_worldpop_layer`'s module docstring for why the two aren't
    guaranteed to sum identically), and that existing column is the one
    `pop_density`/`filter_populated`/every apportionment weight in this
    module is built from -- re-joining WorldPop's `population` under the
    same bare name would silently clobber it (same collision concern
    `_RENAME_EXCLUDED_CANONICAL_NAMES` documents for the census path).

    Landed directly under bare canonical names (`malePopulation`,
    `femalePopulation`, ...) rather than source-prefixed like the census
    joins: unlike the multi-country census path, only one non-census,
    WorldPop-based city configuration is ever active in a given run, so
    there is no cross-source ambiguity a prefix would need to resolve, and
    `global_schema.json` names already *are* these features' real final
    names here. Any bare name already present on `h3_grid` (defensive
    backstop; not expected given today's column set) is skipped with a
    printed warning rather than overwritten.
    """
    from pycensus.census.worldwide.worldpop.loader import GLOBAL_SCHEMA_FEATURES, download_worldpop_layer

    # Real OOM incidents, 2026-08-21: `under18Population`/`adultPopulation`/
    # `over65Population` each required `pycensus`'s `_weighted_sum_rasters` to
    # download and hold multiple whole-country age-bin rasters at once (up
    # to ~10 for under18/adult; 12 -- `_OVER65_BINS`'s 6 bins x 2 sexes --
    # for over65), unlike the single-raster `malePopulation`/
    # `femalePopulation`. For a China-wide raster this OOM-killed the entire
    # `main.py` run THREE separate times live (first on `under18Population`,
    # then again on the same feature after a per-feature `gc.collect()`
    # partial fix, then on `over65Population` specifically once the first
    # two were skipped -- each crash one feature further than the last,
    # confirming it's a genuine per-feature multi-raster memory cost, not
    # one bad feature). Fixed 2026-08-25: `_weighted_sum_rasters` now does
    # windowed/chunked raster I/O (reads/writes fixed-size pixel blocks
    # across all input rasters at a time instead of whole rasters), verified
    # to produce byte-for-byte identical output to the old whole-raster
    # approach on synthetic multi-source rasters. All non-`population`
    # `GLOBAL_SCHEMA_FEATURES` are now joined for Shanghai; the per-feature
    # `gc.collect()` below is kept as a still-cheap extra safety margin.
    #
    # Added 2026-08-25: `births`/`pregnancies`/`urbanPopulation` (see
    # `pycensus.census.worldwide.worldpop.loader`'s module docstring for
    # live verification) join through this exact same loop -- `births`/
    # `pregnancies` are single-file downloads (no multi-raster band math,
    # so no extra OOM exposure), and `urbanPopulation`'s one internal
    # reproject+multiply (`_derive_urban_population`) is windowed the same
    # way `_weighted_sum_rasters` is. `births`/`pregnancies` raise
    # `ValueError` for a country WorldPop has no data for -- not expected
    # to be hit for Shanghai (China has real WorldPop `births`/`pregnancies`
    # coverage, confirmed live), but if it ever is, that propagates as a
    # hard failure of this whole join rather than silently skipping the
    # feature, same as any other genuine WorldPop lookup failure here.
    features = [f for f in GLOBAL_SCHEMA_FEATURES if f != "population"]
    collisions = [f for f in features if f in h3_table.columns]
    if collisions:
        print(
            f"[pipeline] WorldPop global-schema join: {collisions} already present on h3_table -- "
            "skipping those to avoid overwriting existing data"
        )
        features = [f for f in features if f not in collisions]

    cache_dir.mkdir(parents=True, exist_ok=True)

    import gc

    for feature in features:
        print(f"[pipeline] downloading/deriving WorldPop '{feature}' raster (year={year})")
        tif_path = download_worldpop_layer(aoi, year, feature, folder=str(cache_dir))
        # 2026-08-30: area-weighted, mass-conserving resample (replaces the
        # old centroid-based `h3_ops.from_raster_centroid`, which silently
        # dropped a pixel's value entirely into whichever single h3 cell its
        # centroid fell in -- wrong for any pixel straddling a cell boundary,
        # and left cells with no pixel centroid inside them empty even when
        # a pixel's area genuinely overlapped them). See
        # `geohierarchy.raster_resample`/`pycensus...worldpop.h3` for the
        # conservation guarantee.
        from pycensus.census.worldwide.worldpop.h3 import worldpop_raster_to_h3_fast as worldpop_raster_to_h3
        layer_h3 = worldpop_raster_to_h3(tif_path, resolution=resolution, value_col=feature)
        # Plain polars left-join, not the old pandas `.map()` on a
        # `.to_pandas().set_index(...)` Series -- see this function's
        # docstring for why `h3_table` is now geometry-free polars, not a
        # `GeoDataFrame`, and why this avoids an extra full-frame copy.
        h3_table = h3_table.join(layer_h3, on="h3_cell", how="left").with_columns(
            pl.col(feature).fill_null(0.0)
        )
        # A whole-country raster (e.g. Shanghai's China-wide WorldPop tif) is
        # large enough that 5 features' worth of decoded arrays/polars frames
        # left for the next GC cycle to reclaim -- rather than freed
        # immediately -- was enough to OOM-kill the whole `main.py` run
        # (confirmed live 2026-08-21: killed mid-way through this exact
        # loop, having already completed `malePopulation`). Deleting the
        # large intermediates and forcing a collection before starting the
        # next feature's download keeps peak memory to roughly one
        # feature's worth at a time instead of accumulating.
        del layer_h3
        gc.collect()

    # 2026-09-02 (live user report, Shanghai): every feature this function
    # joins lands ONLY under its bare canonical name (see this function's
    # own docstring on why -- a non-census city has no cross-source
    # ambiguity a prefix would need to resolve), which made this city's
    # popup/metadata table show "Male population"/"Female population"/etc
    # with no `worldpop_`-prefixed counterpart, while its OWN population
    # density row showed the OPPOSITE pattern (only `worldpop_population_density`,
    # no bare `population_density`, before the matching fix in
    # `_add_derived_density_columns`) -- inconsistent labeling for data that
    # is, in both cases, entirely WorldPop-sourced. Explicit user request:
    # "I would like all columns to be duplicated ... no worldpop in the
    # column name and the same columns with worldpop in the column name
    # just duplicating the data." A plain column alias (`pl.col(f).alias(...)`,
    # not a second raster download/resample) -- cheap, and correctness-safe
    # since both names point at literally the same values.
    h3_table = h3_table.with_columns([
        pl.col(f).alias(f"worldpop_{f}") for f in features if f in h3_table.columns
    ])

    return h3_table


# Every `uses_census`-capable country's own real census-join population
# column -- the exact column `_join_polygon_stats` leaves null (via its
# `missing = h3_grid[keep_col].isna()` backfill logic) on any h3 cell its
# census source found no real polygon for at any level. This is the single
# "does this cell have real census coverage?" signal used two ways:
#
#   - `_add_worldpop_gapfill` (opt-in via `CityConfig.census_worldpop_gapfill`,
#     currently only Beersheba/Israel): fills WorldPop demographic data ONLY
#     onto the cells where this column IS null (real coverage genuinely
#     absent, e.g. Negev desert/unrecognized villages CBS never surveys).
#   - `_add_worldpop_demographic_layers` (every other `uses_census` city,
#     2026-08-29 fix): the opposite restriction -- fills WorldPop demographic
#     data ONLY onto the cells where this column is NOT null (real census
#     coverage present), so a cell genuinely outside every census polygon
#     (e.g. open country/water/uninhabited land the census source correctly
#     never covers) doesn't get a fabricated WorldPop estimate presented as
#     equally real. Before this fix, `_add_worldpop_demographic_layers` filled
#     `worldpop_*` unconditionally onto the FULL grid of every city -- wrong
#     per explicit user instruction: WorldPop should only ever fill cells a
#     real census polygon actually covers, except Beersheba, whose
#     `census_worldpop_gapfill` opt-in already has a documented, motivated
#     reason to go further (see `_add_worldpop_gapfill`'s docstring).
#
# Each entry is the BARE prefixed column name (survives `_rename_canonical_columns`
# unrenamed -- see that function's `bare in gdf.columns` collision-avoidance:
# `population` always already exists on `h3_grid` from WorldPop before any
# census join runs, so every country's own `<prefix>population` column is left
# prefixed rather than renamed away, and stays usable here). Verified against
# `_join_census`'s actual per-country dispatch (which loader + which prefix
# each country's `KEEP_COLUMNS` set actually uses) rather than assumed from
# `CENSUS_COLUMN_PREFIXES` alone, since some countries run MULTIPLE prefixed
# joins (e.g. Chile's `ine_cl_` + `casen_cl_` + `sii_cl_`) and only one of
# them actually carries `population`:
#   USA -> acs5_population (ACS5, `_join_census`'s `country == "USA"` branch)
#   MEX -> inegi_population
#   ESP -> ine_population (generic Spain/INE; Gipuzkoa's `census_module="euskadi"`
#          override is not currently used by any live `CityConfig`, see
#          `city_config.py`'s Gipuzkoa entry -- add an override here too if
#          that ever changes, since Euskadi's join uses "eustat_" instead)
#   DEU -> destatis_population (the ADMIN-POLYGON join's own coverage column;
#          `_join_zensus_grid_population` separately REPLACES bare
#          `population` with real Zensus grid data for the whole grid, but
#          that's an unrelated, independent mechanism -- `destatis_population`
#          still reflects real admin-polygon coverage correctly)
#   ISR -> cbs_population (the ORIGINAL entry this dict has always had --
#          Beersheba uses this for the opposite, gap-fill purpose above)
#   CAN -> statcan_population
#   AND -> estadisticaad_population
#   TWN -> moi_population
#   CHL -> ine_cl_population (the primary Chile join; Casen/SII's own
#          `casen_cl_`/`sii_cl_` joins don't carry a population field)
#
# `CHN` (Shanghai) is deliberately absent: it has no real census join at all
# (`uses_census=False`), so it never reaches either the gap-fill or the
# restricted-fill code path -- it stays on `_join_worldpop_global_schema`'s
# separate always-on-full-grid, bare-named mechanism instead.
CENSUS_WORLDPOP_GAPFILL_POPULATION_COLUMN: dict[str, str] = {
    "USA": "acs5_population",
    "MEX": "inegi_population",
    "ESP": "ine_population",
    "DEU": "destatis_population",
    "ISR": "cbs_population",
    "CAN": "statcan_population",
    "AND": "estadisticaad_population",
    "TWN": "moi_population",
    "CHL": "ine_cl_population",
}


def _filter_cells_touching_country_border(
    h3_grid: gpd.GeoDataFrame,
    geocode_name: str,
    cache_dir: Path,
) -> gpd.GeoDataFrame:
    """Drop every h3 cell whose geometry touches `geocode_name`'s national boundary line.

    Opt-in (`CityConfig.exclude_cells_touching_country_border`) -- see that
    flag's docstring for the motivating case (Beersheba, near the West
    Bank/Gaza/Egypt frontiers). Geocodes the country once via
    `_geocode_admin_polygon` (same Nominatim helper `resolve_core_boundary`
    uses for non-US city-core boundaries), caching the polygon to
    `cache_dir/country_boundary.gpkg` so a re-run never re-hits Nominatim.
    A cell is dropped whenever it INTERSECTS the boundary LINE (the
    country polygon's `.boundary`, not its interior) -- i.e. it straddles
    or exactly abuts the border -- never clipped to its in-country portion,
    since a fractional cell can't honestly carry one equity flag/
    level_of_service value. Cells entirely inside or entirely outside the
    country are both left untouched (the AOI itself already keeps this
    pipeline from generating cells far outside the country in practice).
    """
    from .city_core import _geocode_admin_polygon

    cache_path = cache_dir / "country_boundary.gpkg"
    if cache_path.exists():
        boundary_gdf = gpd.read_file(cache_path)
    else:
        boundary_gdf = _geocode_admin_polygon(geocode_name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        boundary_gdf.to_file(cache_path, driver="GPKG")

    boundary_line = boundary_gdf.to_crs(h3_grid.crs).union_all().boundary
    touches_border = h3_grid.geometry.intersects(boundary_line)
    n_dropped = int(touches_border.sum())
    print(
        f"[pipeline] border clip ({geocode_name}): dropping {n_dropped} of {len(h3_grid)} "
        "h3 cell(s) that touch the national border"
    )
    return h3_grid[~touches_border].reset_index(drop=True)


def _add_worldpop_gapfill(
    h3_grid: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    resolution: int,
    year: int,
    cache_dir: Path,
    population_column: str,
) -> gpd.GeoDataFrame:
    """Fill real WorldPop age/sex-structure data onto h3 cells the real census join left uncovered.

    Opt-in (`CityConfig.census_worldpop_gapfill`) -- see that flag's
    docstring for the motivating case (Beersheba: real CBS locality
    polygons cover only ~51% of the metro AOI by area; the rest is open
    Negev desert / unrecognized Bedouin villages CBS deliberately never
    surveys, a documented exclusion, not a data gap in this pipeline).

    A cell counts as "uncovered" when `population_column` (the census
    source's own bare population count, e.g. `cbs_population` -- left null
    by `_join_polygon_stats` for any cell no real census polygon at any
    level matched, see that function's `missing = h3_grid[keep_col].isna()`
    backfill logic) is null, or -- defensively -- when `population_column`
    isn't on `h3_grid` at all (the census join found nothing whatsoever).

    Landed under a distinctly `worldpop_`-prefixed name
    (`worldpop_malePopulation`, `worldpop_femalePopulation`,
    `worldpop_under18Population`, `worldpop_over65Population`,
    `worldpop_adultPopulation`) so real `cbs_*`/other census-source columns
    are never overwritten or renamed into -- real census data always wins
    where it exists; this only ever writes into cells that were null. Every
    `worldpop_*` value is set ONLY on gap cells (`np.nan` elsewhere on the
    grid, never a duplicate of an already-covered cell's real census
    figures), so summing a `worldpop_*` column and a `cbs_*`/other
    census-source column together (a citywide total) can never double-count
    a cell -- each cell contributes real data from at most one source per
    demographic breakdown.

    Deliberately does NOT add a `worldpop_population` column: `h3_grid`
    already carries a bare `population` column (from
    `population_and_access_to_h3`, a WorldPop `pop`-family raster) that
    covers the WHOLE grid, gap cells included -- that existing coverage is
    the whole reason a gap-fill for population count specifically would be
    a pure duplicate. `population` is also already what `pop_density`/every
    apportionment weight/the map's default population figure is built from
    (see `_RENAME_EXCLUDED_CANONICAL_NAMES`), so reusing it instead of a new
    `worldpop_population` also means the `worldpop_*_share` columns below
    denominate against a real, already-covered-everywhere figure rather
    than a new column that would just equal a subset of it.
    """
    from pycensus.census.worldwide.worldpop.loader import GLOBAL_SCHEMA_FEATURES, download_worldpop_layer

    if population_column not in h3_grid.columns:
        print(
            f"[pipeline] WorldPop gap-fill: {population_column!r} not present on h3_grid at all -- "
            "treating every cell as uncovered"
        )
        missing_mask = np.ones(len(h3_grid), dtype=bool)
    else:
        missing_mask = h3_grid[population_column].isna().to_numpy()

    n_missing = int(missing_mask.sum())
    print(
        f"[pipeline] WorldPop gap-fill: {n_missing}/{len(h3_grid)} h3 cells have no real "
        f"{population_column!r} coverage -- filling those with real WorldPop age/sex layers"
    )
    if n_missing == 0:
        return h3_grid

    cache_dir.mkdir(parents=True, exist_ok=True)
    original_index = h3_grid.index
    h3_grid = h3_grid.reset_index(drop=True)  # `missing_mask` above is positional, computed before this reset

    features = [f for f in GLOBAL_SCHEMA_FEATURES if f != "population"]

    import gc

    for feature in features:
        col = f"worldpop_{feature}"
        if col in h3_grid.columns:
            print(f"[pipeline] WorldPop gap-fill: {col!r} already present on h3_grid -- skipping")
            continue
        print(f"[pipeline] WorldPop gap-fill: downloading/deriving '{feature}' raster (year={year})")
        tif_path = download_worldpop_layer(aoi, year, feature, folder=str(cache_dir))
        # 2026-08-30: area-weighted, mass-conserving resample -- see the
        # matching comment in `_join_worldpop_global_schema` above.
        from pycensus.census.worldwide.worldpop.h3 import worldpop_raster_to_h3_fast as worldpop_raster_to_h3
        layer_h3 = worldpop_raster_to_h3(tif_path, resolution=resolution, value_col=col)
        layer_series = layer_h3.to_pandas().set_index("h3_cell")[col]
        values = h3_grid["h3_cell"].map(layer_series).to_numpy(dtype=float)
        # Same OOM lesson as `_join_worldpop_global_schema` (2026-08-21
        # incident) -- delete each feature's large intermediates and force a
        # collection before starting the next feature's download rather than
        # letting them accumulate for the next GC cycle.
        h3_grid[col] = np.where(missing_mask, values, np.nan)
        del layer_h3, layer_series, values
        gc.collect()

    h3_grid.index = original_index
    return h3_grid


def _add_worldpop_demographic_layers(
    h3_grid: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    resolution: int,
    year: int,
    cache_dir: Path,
    restrict_to_population_column: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Join every non-`population` WorldPop `GLOBAL_SCHEMA_FEATURES` layer onto `h3_grid`'s real-census-covered cells.

    Generalization (2026-08-25) of the two existing WorldPop demographic
    mechanisms to run on every city, not just the two special cases:

      - `_join_worldpop_global_schema` only runs for Shanghai (no real
        China census implementation in `pycensus` at all) and lands its
        columns under BARE canonical names (`malePopulation`, ...) since
        WorldPop is the only source there.
      - `_add_worldpop_gapfill` only runs for `census_worldpop_gapfill`
        cities (Beersheba) and only fills the h3 cells a real census join
        left uncovered, under `worldpop_`-prefixed names, leaving every
        other cell `np.nan`.

    This function is the additive third case: a second, independently-sourced
    real WorldPop demographic estimate, always under `worldpop_`-prefixed
    names so it can never collide with or overwrite a real
    `cbs_`/`ine_`/`inegi_`/etc. census column -- this is extra real data,
    never a replacement. Not called for Shanghai (out of scope; it already
    gets the same features under bare names via `_join_worldpop_global_schema`,
    and re-adding a `worldpop_`-prefixed duplicate of China's already-huge
    whole-country raster join would just double the OOM exposure for zero new
    information).

    `restrict_to_population_column` (2026-08-29, fixing an explicit user-
    reported bug): when given, a cell is only filled if that column is
    non-null on it -- i.e. only cells with REAL census-polygon coverage (see
    `CENSUS_WORLDPOP_GAPFILL_POPULATION_COLUMN` for the per-country column and
    the reasoning) get a `worldpop_*` value; every other cell is left
    `np.nan`. Before this parameter existed, this function filled the WHOLE
    grid unconditionally regardless of census coverage, which silently
    presented WorldPop's modeled estimate for genuinely-uncovered land (open
    country, water, land outside any census polygon) as if it were real data.
    Pass `None` (the default) to keep the original full-grid behavior --
    used ONLY for Beersheba (`CityConfig.census_worldpop_gapfill`), whose
    `_add_worldpop_gapfill` call already has a documented, motivated reason
    to cover cells beyond real census polygons (~49% of the metro AOI is
    real, deliberately-uncovered Negev desert/unrecognized villages, not a
    data gap -- see that function's docstring), so restricting THIS function
    too for Beersheba would just make those legitimately-filled gap cells
    lose the top-up they already correctly receive below.

    If a `worldpop_<feature>` column already exists on `h3_grid` (e.g.
    Beersheba's `_add_worldpop_gapfill`, which only wrote gap cells and left
    the rest `np.nan`), only the still-null cells (further narrowed by
    `restrict_to_population_column` when given) are filled -- so a real
    gap-fill value already computed for a cell is never overwritten.

    Same windowed raster I/O + per-feature `gc.collect()` memory discipline
    as `_join_worldpop_global_schema`/`_add_worldpop_gapfill` (see those
    docstrings for the live 2026-08-21 OOM incident this avoids).
    """
    from pycensus.census.worldwide.worldpop.loader import GLOBAL_SCHEMA_FEATURES, download_worldpop_layer

    features = [f for f in GLOBAL_SCHEMA_FEATURES if f != "population"]

    cache_dir.mkdir(parents=True, exist_ok=True)
    original_index = h3_grid.index
    h3_grid = h3_grid.reset_index(drop=True)

    if restrict_to_population_column is None:
        covered_mask = None
    elif restrict_to_population_column not in h3_grid.columns:
        print(
            f"[pipeline] WorldPop demographic layer: restrict column "
            f"{restrict_to_population_column!r} not present on h3_grid at all -- "
            "filling every cell (no census coverage to restrict to)"
        )
        # No census figure exists at all (e.g. the census download was refused): nothing to restrict to, WorldPop fills every cell.
        covered_mask = None
    else:
        covered_mask = h3_grid[restrict_to_population_column].notna().to_numpy()
        print(
            f"[pipeline] WorldPop demographic layer: restricting fill to "
            f"{int(covered_mask.sum())}/{len(h3_grid)} h3 cells with real "
            f"{restrict_to_population_column!r} census coverage"
        )

    import gc

    from pycensus.census.worldwide.worldpop.h3 import worldpop_raster_to_h3_fast as worldpop_raster_to_h3
    from concurrent.futures import ThreadPoolExecutor

    cells_all = h3_grid["h3_cell"]

    def _one_layer(feature):
        """The h3 values of one WorldPop layer: independent of the other layers, so they run side by side (raster reads and polars release the GIL)."""
        col = f"worldpop_{feature}"
        print(f"[pipeline] WorldPop demographic layer: downloading/deriving '{feature}' raster (year={year})", flush=True)
        tif_path = download_worldpop_layer(aoi, year, feature, folder=str(cache_dir))
        # 2026-08-30: area-weighted, mass-conserving resample -- see the
        # matching comment in `_join_worldpop_global_schema` above.
        layer_h3 = worldpop_raster_to_h3(tif_path, resolution=resolution, value_col=col)
        layer_series = layer_h3.to_pandas().set_index("h3_cell")[col]
        return col, cells_all.map(layer_series).to_numpy(dtype=float)

    workers = max(1, min(len(features), int(os.environ.get("CS_LAYER_WORKERS", "4"))))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_one_layer, features))
    for col, values in results:
        if covered_mask is not None:
            values = np.where(covered_mask, values, np.nan)
        if col in h3_grid.columns:
            existing = h3_grid[col].to_numpy(dtype=float)
            h3_grid[col] = np.where(np.isnan(existing), values, existing)
        else:
            h3_grid[col] = values
    del results
    gc.collect()

    h3_grid.index = original_index
    return h3_grid


def _sjoin_nearest_chunked(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    id_col: str,
    chunk_size: int = 2_000,
    pad_m: float = 5_000.0,
) -> gpd.GeoDataFrame:
    """`gpd.sjoin_nearest(left, right)`, but without paying for all of `right` at once.

    `left` (e.g. census polygons with no h3 centroid strictly within them) is
    typically small -- a handful up to a few thousand rows -- but `right` can
    be the FULL h3 grid: for a metro the size of Boston that is millions of
    rows. `gpd.sjoin_nearest` builds its candidate spatial index over the
    *entire* right-hand argument regardless of how few points are actually
    being queried, and `_census_geometries_with_score` calls this once per
    census level (block, blockgroup, tract, county) -- handing it the whole
    grid every time is exactly the trap `h3_population.py`'s
    `population_and_access_to_h3` docstring warns about for
    `edges_to_h3_by_distance`: re-testing every candidate against work that
    only ever touches a local geographic extent. This chunks `left` and
    restricts each chunk's `right` candidates to that chunk's own padded
    bounding box, so a geographically small batch of query points only pays
    for nearby candidates, not the whole grid.

    Both `left` and `right` must already share a projected (metric) CRS --
    `pad_m` is a distance in that CRS's units.
    """
    if len(left) <= chunk_size:
        return gpd.sjoin_nearest(left, right, how="left")

    right_xy = np.column_stack([right.geometry.x.to_numpy(), right.geometry.y.to_numpy()])
    parts = []
    for start in range(0, len(left), chunk_size):
        chunk = left.iloc[start : start + chunk_size]
        minx, miny, maxx, maxy = chunk.total_bounds
        mask = (
            (right_xy[:, 0] >= minx - pad_m)
            & (right_xy[:, 0] <= maxx + pad_m)
            & (right_xy[:, 1] >= miny - pad_m)
            & (right_xy[:, 1] <= maxy + pad_m)
        )
        local_right = right.loc[mask]
        if local_right.empty:
            # Pad too small for this chunk (a sparse patch of the grid) --
            # fall back to the full candidate set rather than silently
            # dropping the match.
            local_right = right
        parts.append(gpd.sjoin_nearest(chunk, local_right, how="left"))
    return pd.concat(parts, ignore_index=False)


def _census_loader_and_levels_for_map(
    country: str, census_module: str | None
) -> tuple[Callable[..., gpd.GeoDataFrame], tuple[str, ...]] | None:
    """Resolve `(loader, levels)` for the map's census-geometry overlay, or `None` if unsupported.

    Mirrors `_join_census`'s per-country dispatch table (same modules, same
    `*_LEVELS` tuples) so the map's "Census geometries" shape option covers
    every country `_join_census` does, not just USA -- before this, this
    function was hardcoded to `pycensus.census.countries.usa.acs5` unconditionally, so every
    non-US city's `_census_geometries_with_score` call failed immediately
    (`states=None` -> `list(None)` -> `TypeError`, swallowed by the
    per-level `except`) and silently produced an empty `census_by_level`.
    """
    if census_module == "euskadi":
        return _euskadi_census_loader, EUSKADI_LEVELS
    if country == "USA":
        # `MAP_CENSUS_LEVELS` is coarse-to-fine (what the caller needs
        # directly), but this dispatcher's `levels` return value is always
        # read back through `_census_geometries_with_score`'s automatic
        # `reversed(levels)` step (every OTHER country's `*_LEVELS` tuple is
        # finest-first, for `_join_polygon_stats`'s unrelated backfill loop)
        # -- so passing `MAP_CENSUS_LEVELS` as-is here would get reversed
        # into finest-first by that shared step and end up backwards. Passing
        # the already-reversed (finest-first) tuple keeps it consistent with
        # every other country and lands back coarse-to-fine after that step.
        return _usa_map_census_loader, tuple(reversed(MAP_CENSUS_LEVELS))
    if country == "MEX":
        return _mexico_census_loader, INEGI_LEVELS
    if country == "ESP":
        return _spain_census_loader, SPAIN_LEVELS
    if country == "ISR":
        return _israel_census_loader, ISRAEL_LEVELS
    if country == "CAN":
        return _canada_census_loader, CANADA_LEVELS
    if country == "AND":
        return _andorra_census_loader, ANDORRA_LEVELS
    if country == "TWN":
        return _taiwan_census_loader, TAIWAN_LEVELS
    if country == "CHL":
        return _chile_census_loader, CHILE_LEVELS
    if country == "DEU":
        return _germany_map_census_loader, GERMANY_LEVELS
    return None


def _census_polygon_agg_chunked(
    pts: gpd.GeoDataFrame,
    pts_h3_cell,
    census_gdf: gpd.GeoDataFrame,
    chunk_h3_resolution: int,
) -> pd.DataFrame:
    """Memory-bounded equivalent of `_census_geometries_with_score`'s two
    `add_vector_data` calls (population-weighted `level_of_service` mean +
    area-weighted `population` sum).

    2026-09-04, explicit user request (live Boston OOM, 3 attempts): "use
    res 5 chunks... process each chunk completely independently... [for]
    population, census, streets, etc. resampling." `isochrone_chunk_h3_resolution`
    already chunked isochrones/the h3-grid-onto-census join/H3 resampling
    (`_chunked_h3_grid_join`, `_resample_h3_chunked`,
    `compute_node_access_chunked`) -- this was the one remaining whole-
    metro-at-once stage, and the actual crash site in all three of Boston's
    2026-09-04 attempts (confirmed live: identical `level_core_mask`
    traceback location each time, right before this function's
    `add_vector_data` calls).

    `pts` (one row per POPULATED h3 cell) is the row-count-driven memory
    cost -- millions of rows for a Boston-scale metro, vs. `census_gdf`'s
    thousands of polygons even at "block" level. Partitioning `pts` by each
    cell's `chunk_h3_resolution`-ancestor and processing one chunk's `pts`
    subset against the SAME (small, unchunked) `census_gdf` at a time
    bounds peak memory to one chunk's cell count.

    Exact, not approximate, unlike `_join_polygon_stats_chunked`'s
    boundary-polygon caveat -- both quantities are re-expressed as plain
    sums before chunking:

    - `level_of_service` (population-weighted mean over touching cells,
      `intersection_mode="intersects"`) = `sum(level_of_service *
      population) / sum(population)` over the matched cell set. Both the
      numerator and denominator are linear sums over disjoint per-chunk
      cell subsets (chunking partitions `pts` exactly, no cell counted
      twice or skipped), so summing each chunk's own partial numerator/
      denominator and dividing once at the end reproduces the EXACT same
      ratio the unchunked single-pass `Mean(weight_column="population")`
      would compute.
    - `population` (`Sum(geoweighted=True)`, exact area-fraction weighting,
      `intersection_mode="exact"`) is already a plain sum per polygon --
      each row's own area-fraction contribution doesn't depend on which
      other rows are present, so summing per-chunk partial sums is exact
      by the same linearity argument regardless of geoweighting.

    Returns a DataFrame with columns `_geo_idx`, `level_of_service`,
    `_h3_fallback_population` -- the same shape the unchunked path's own
    `agg_result` has, so the caller's merge/nearest-cell-fallback logic
    downstream is unchanged either way.
    """
    from geohierarchy import GeoHierarchy
    from geohierarchy.aggregation import Sum

    chunk_ids = np.array([h3.cell_to_parent(c, chunk_h3_resolution) for c in pts_h3_cell])
    census_mini = census_gdf[["_geo_idx", "geometry"]].copy()

    wlos_parts: list[pd.DataFrame] = []
    wpop_parts: list[pd.DataFrame] = []
    exact_pop_parts: list[pd.DataFrame] = []
    for parent in pd.unique(chunk_ids):
        pts_chunk = pts.loc[chunk_ids == parent]
        if pts_chunk.empty:
            continue
        pts_chunk = pts_chunk.copy()
        pts_chunk["_weighted_los"] = pts_chunk["level_of_service"].to_numpy(dtype=float) * pts_chunk[
            "population"
        ].to_numpy(dtype=float)

        hierarchy = GeoHierarchy(crs=census_gdf.crs)
        hierarchy.add_level("_census_map_level", census_mini, id_col="_geo_idx")
        hierarchy.add_vector_data(
            pts_chunk,
            level="_census_map_level",
            columns=["_weighted_los", "population"],
            agg={
                "_weighted_los": Sum(geoweighted=False),
                "population": Sum(geoweighted=False),
            },
            fill_null=0.0,
            intersection_mode="intersects",
        )
        _hier_id_col = hierarchy.id_cols["_census_map_level"]
        part = hierarchy["_census_map_level"][[_hier_id_col, "_weighted_los", "population"]]
        wlos_parts.append(part[[_hier_id_col, "_weighted_los"]].rename(columns={_hier_id_col: "_geo_idx"}))
        wpop_parts.append(
            part[[_hier_id_col, "population"]].rename(columns={_hier_id_col: "_geo_idx", "population": "_wpop"})
        )

        hierarchy2 = GeoHierarchy(crs=census_gdf.crs)
        hierarchy2.add_level("_census_map_level", census_mini, id_col="_geo_idx")
        hierarchy2.add_vector_data(
            pts_chunk,
            level="_census_map_level",
            columns=["population"],
            agg={"population": Sum(geoweighted=True)},
            fill_null=0.0,
            intersection_mode="exact",
        )
        _hier_id_col2 = hierarchy2.id_cols["_census_map_level"]
        exact_pop_parts.append(
            hierarchy2["_census_map_level"][[_hier_id_col2, "population"]].rename(
                columns={_hier_id_col2: "_geo_idx", "population": "_exact_pop"}
            )
        )

    result = census_gdf[["_geo_idx"]].copy()
    if wlos_parts:
        wlos_total = pd.concat(wlos_parts, ignore_index=True).groupby("_geo_idx", as_index=False)["_weighted_los"].sum()
        wpop_total = pd.concat(wpop_parts, ignore_index=True).groupby("_geo_idx", as_index=False)["_wpop"].sum()
        exact_pop_total = pd.concat(exact_pop_parts, ignore_index=True).groupby("_geo_idx", as_index=False)["_exact_pop"].sum()
        result = result.merge(wlos_total, on="_geo_idx", how="left")
        result = result.merge(wpop_total, on="_geo_idx", how="left")
        result = result.merge(exact_pop_total, on="_geo_idx", how="left")
    else:
        result["_weighted_los"] = np.nan
        result["_wpop"] = np.nan
        result["_exact_pop"] = np.nan
    result["_weighted_los"] = result["_weighted_los"].fillna(0.0)
    result["_wpop"] = result["_wpop"].fillna(0.0)
    result["_exact_pop"] = result["_exact_pop"].fillna(0.0)
    result["level_of_service"] = np.where(
        result["_wpop"].to_numpy() > 0, result["_weighted_los"] / result["_wpop"].replace(0, np.nan), np.nan
    )
    result = result.rename(columns={"_exact_pop": "_h3_fallback_population"})
    return result[["_geo_idx", "level_of_service", "_h3_fallback_population"]]


_RESAMPLE_MAX_CELLS = 3_000_000
_RESAMPLE_SKIP = {
    "GEOID", "_geo_idx", "geometry", "area_m2", "level_of_service", "h3_cell", "equity_flag",
    # population bookkeeping: reconciled per polygon by _finalize_population_columns /
    # _add_worldpop_outside_census_columns, which must keep seeing the census value
    "population", "census_residents", "worldpop_residents", "population_overestimation_vs_census", "population_underestimation_vs_census",
}


def _resample_h3_density_1km(census_gdf: gpd.GeoDataFrame, h3_grid: gpd.GeoDataFrame, populated_mask) -> gpd.GeoDataFrame:
    """The neighbourhood densities (500 m / 1 / 2 / 3 km, transitlos.neighbourhood.DENSITY_COLUMNS) on census polygons: the residents-weighted
    mean of their H3 cells' densities (the densities the polygon's residents live in), from area-weighted sums of density x residents and of
    residents. Columns the layer already has are kept."""
    from transitlos.neighbourhood import DENSITY_COLUMNS

    cols = [c for c in DENSITY_COLUMNS if c in h3_grid.columns and c not in census_gdf.columns]
    if not cols:
        return census_gdf
    pc = _census_population_column(h3_grid) or _worldpop_population_column(h3_grid)
    if pc is None:
        return census_gdf
    cells = h3_grid.loc[populated_mask, cols + [pc, "geometry"]].copy()
    if len(cells) > _RESAMPLE_MAX_CELLS or not len(cells):
        return census_gdf
    w = pd.to_numeric(cells[pc], errors="coerce").fillna(0.0)
    data = {f"_dw{i}": pd.to_numeric(cells[c], errors="coerce").fillna(0.0) * w for i, c in enumerate(cols)}
    data["_d1p"] = w
    cells = gpd.GeoDataFrame(data, geometry=cells.geometry.to_numpy(), crs=h3_grid.crs).to_crs(census_gdf.crs)
    hierarchy = GeoHierarchy(crs=census_gdf.crs)
    hierarchy.add_level("_census_map_level", census_gdf[["_geo_idx", "geometry"]].copy(), id_col="_geo_idx")
    sums = list(data)
    hierarchy.add_vector_data(cells, level="_census_map_level", columns=sums,
                              agg={k: Sum(geoweighted=True) for k in sums}, fill_null=None, intersection_mode="exact")
    id_col = hierarchy.id_cols["_census_map_level"]
    r = hierarchy["_census_map_level"][[id_col] + sums].rename(columns={id_col: "_geo_idx"})
    for i, c in enumerate(cols):
        r[c] = np.where(r["_d1p"] > 0, r[f"_dw{i}"] / r["_d1p"].where(r["_d1p"] > 0), np.nan)
    print(f"[pipeline] {', '.join(cols)} on {int(r[cols[0]].notna().sum()):,} census polygons (residents-weighted mean of their cells)")
    return census_gdf.merge(r[["_geo_idx"] + cols], on="_geo_idx", how="left")


def _resample_missing_h3_counts(
    census_gdf: gpd.GeoDataFrame, h3_grid: gpd.GeoDataFrame, populated_mask
) -> gpd.GeoDataFrame:
    """Resample absolute-count columns that exist on the H3 grid but not on this census layer.

    Some sources are only published at one level (e.g. Eustat establishments/births/deaths at
    municipality), so the polygon layers of the other levels would lack them. Every such count is
    area-weighted summed from the H3 cells onto the polygons (geohierarchy `Sum(geoweighted=True)`,
    exact intersection), so each census geometry carries every count column. Native columns are
    never overwritten. Skipped for grids above `_RESAMPLE_MAX_CELLS` populated cells.
    """
    from transitlos.map.build import absolute_fields

    census_gdf = _resample_h3_density_1km(census_gdf, h3_grid, populated_mask)
    # 2026-11-15: a column the layer has but leaves blank everywhere (an ACS table not published at this level, e.g. nativity on block
    # groups) is filled from the cells too -- they hold the coarser level's values
    blank = [c for c in census_gdf.columns if c in h3_grid.columns and pd.to_numeric(census_gdf[c], errors="coerce").isna().all()]
    missing = [
        c for c in absolute_fields(list(h3_grid.columns))
        if (c not in census_gdf.columns or c in blank) and c not in _RESAMPLE_SKIP
        and pd.api.types.is_numeric_dtype(h3_grid[c]) and h3_grid[c].notna().any()
    ]
    if not missing:
        return census_gdf
    census_gdf = census_gdf.drop(columns=[c for c in missing if c in census_gdf.columns])
    cells = h3_grid.loc[populated_mask, [*missing, "geometry"]]
    if len(cells) > _RESAMPLE_MAX_CELLS:
        print(f"[pipeline] skipping h3->census resampling of {len(missing)} count columns: {len(cells)} cells > {_RESAMPLE_MAX_CELLS}")
        return census_gdf
    cells = gpd.GeoDataFrame(cells, geometry="geometry", crs=h3_grid.crs).to_crs(census_gdf.crs)
    hierarchy = GeoHierarchy(crs=census_gdf.crs)
    hierarchy.add_level("_census_map_level", census_gdf[["_geo_idx", "geometry"]].copy(), id_col="_geo_idx")
    hierarchy.add_vector_data(
        cells,
        level="_census_map_level",
        columns=missing,
        agg={c: Sum(geoweighted=True) for c in missing},
        fill_null=None,
        intersection_mode="exact",
    )
    id_col = hierarchy.id_cols["_census_map_level"]
    resampled = hierarchy["_census_map_level"][[id_col, *missing]].rename(columns={id_col: "_geo_idx"})
    print(f"[pipeline] resampled {len(missing)} h3 count columns onto census polygons: {missing}")
    return census_gdf.merge(resampled, on="_geo_idx", how="left")


def _census_geometries_with_score(
    h3_grid: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    states: tuple[str, ...],
    census_levels: tuple[str, ...],
    cache_dir: Path,
    country: str = "USA",
    census_module: str | None = None,
    chunk_h3_resolution: Optional[int] = None,
    worldpop_only_population: bool = False,
) -> dict[str, gpd.GeoDataFrame]:
    """Aggregate `level_of_service`/`pop_density` onto real census polygons, per level.

    Used only for the map's "Census geometries" shape option -- unlike
    `_join_census` (which pulls census attributes *onto* the h3 grid), this
    goes the other direction: h3 cells' level_of_service is population-weighted
    up onto each census polygon so the map can render LOS directly on
    census boundaries rather than the h3 grid. Each source's own attribute
    columns are already present on `census_gdf` from its `load()` and kept
    as-is. Dispatches per `country`/`census_module` (see
    `_census_loader_and_levels_for_map`) the same way `_join_census` does;
    `country`/`census_module` not covered there print a clear message and
    return an empty dict rather than crashing.
    """
    resolved = _census_loader_and_levels_for_map(country, census_module)
    if resolved is None:
        print(
            f"[pipeline] census-geometry map overlay not supported for country={country!r} -- "
            "skipping, map falls back to the h3 grid for this shape option"
        )
        return {}
    loader, levels = resolved
    if levels is not None:
        # `INEGI_LEVELS`/`EUSKADI_LEVELS`/`SPAIN_LEVELS` are all defined
        # finest-first (for `_join_polygon_stats`'s own coarser-fills-gaps
        # backfill loop, a completely different consumer -- see each
        # constant's docstring). This function's own caller
        # (`build_city_map` in build.py, via `_level_zoom_bands`) requires
        # the OPPOSITE: coarsest-to-finest, so each level gets progressively
        # more of the zoom range as you zoom in. Reversing here (rather than
        # reordering the shared *_LEVELS tuples, which would break the
        # backfill loop) means the returned `census_by_level` dict is always
        # coarse->fine regardless of country -- the real bug this fixes: for
        # every non-US country the map only ever showed the COARSEST level
        # (e.g. Mexico's "state") at high zoom and the finest (e.g. "ageb",
        # roughly blockgroup-equivalent) only in a sliver near zoom 0, where
        # nobody looks -- it read as "no block/blockgroup/tract polygons,
        # just big municipality/state shapes" even though the finer data was
        # fetched and present in `census_by_level` all along. USA's
        # `MAP_CENSUS_LEVELS` (the `levels is None` branch below) is already
        # coarse->fine and is left untouched.
        census_levels = tuple(reversed(levels))

    # H3-native analytic centroid (see `_h3_cell_centroids`) -- kept only for
    # the nearest-cell fallback below (a "closest real measurement" search
    # needs a point, not a polygon). The actual per-polygon aggregation now
    # uses each cell's real hexagon geometry (`h3_grid.geometry`), not this
    # centroid -- see the `_geo_idx` sjoin below for why.
    centroids = pd.Series(_h3_cell_centroids(h3_grid["h3_cell"]), index=h3_grid.index)
    result: dict[str, gpd.GeoDataFrame] = {}

    # Countries whose `load()` derives regions from `aoi` instead of an
    # explicit US-style state list (see each `_*_census_loader`'s own
    # docstring) accept `states=None` directly -- unlike USA's `acs.load`,
    # which needs a real list. `list(None)` used to be called unconditionally
    # here and raised for every non-US call, silently swallowed by the
    # per-level `except` below.
    states_arg = list(states) if states else None
    for level in census_levels:
        import time as _time
        _t0 = _time.time()
        if country == "CAN" and level in ("dissemination-area", "dissemination-block", "municipality", "province"):
            # StatCan polygons come from its ArcGIS service while the file site is down (self-touching rings break the area overlay);
            # their values are already aggregated into the H3 grid, only the map polygon layer is skipped.
            print(f"[pipeline] skipping StatCan census level {level!r} for the map polygons")
            continue
        try:
            census_gdf = loader(aoi=aoi, states=states_arg, level=level, cache_dir=str(cache_dir))
        except Exception as exc:  # pragma: no cover - network/data availability varies per level
            print(f"[pipeline] skipping census level {level!r} for map: {exc}")
            continue
        print(f"[TIMING] {level}: loader took {_time.time()-_t0:.1f}s, rows={len(census_gdf)}")
        if census_gdf.empty:
            continue
        census_gdf = census_gdf.reset_index(drop=True)
        census_gdf["_geo_idx"] = census_gdf.index
        # Perf fix (2026-08-30, live: Toronto's "province" level -- a SINGLE
        # polygon, Ontario's real government boundary -- made the join below
        # hang for 30+ minutes against only 1.3M h3 cells). Root cause,
        # confirmed by direct profiling: the raw polygon has ~3.8 MILLION
        # vertices (full province boundary at Great-Lakes-coastline detail).
        # A single complex destination polygon gets zero benefit from the
        # join's spatial index (which only helps disambiguate BETWEEN
        # multiple candidate geometries by bounding box; with one candidate,
        # every point anywhere near the whole province is a "candidate"),
        # so every one of the 1.3M point-in-polygon tests paid the full cost
        # of walking a 3.8M-vertex boundary. `simplify()` ALONE wasn't
        # enough -- it also has to walk the same 3.8M vertices and itself
        # timed out (verified: >120s with no result). The actual fix is to
        # CLIP each level's geometry to the AOI's own bounding box (plus a
        # generous pad) BEFORE anything else: this map only ever needs the
        # sliver of a province/state/country-level polygon that overlaps
        # the metro AOI, and clipping is cheap (a bbox-rect clip, not a
        # boundary walk) -- confirmed live: clip 3.8M vertices -> 144K in
        # 0.03s, and *then* `simplify()` (previously the same tolerance
        # already used elsewhere in this module for the AOI-gap-fallback
        # geometry) finishes in ~0.1s down to ~9K vertices, after which the
        # actual join drops from an unbounded hang to ~50s. Applied to
        # every level, not just the coarsest, since any level's polygon can
        # in principle carry excess detail outside the AOI.
        _minx, _miny, _maxx, _maxy = aoi.to_crs(census_gdf.crs).total_bounds
        _pad = 0.05
        census_gdf["geometry"] = census_gdf.geometry.clip_by_rect(
            _minx - _pad, _miny - _pad, _maxx + _pad, _maxy + _pad
        )
        census_gdf = census_gdf[~census_gdf.geometry.is_empty].reset_index(drop=True)
        census_gdf["_geo_idx"] = census_gdf.index
        census_gdf["geometry"] = census_gdf.geometry.simplify(0.00003, preserve_topology=True)
        # 2026-09-01 (live user report): the bbox `clip_by_rect` above is a
        # rectangular clip -- any polygon (e.g. a state/province/municipality
        # whose real boundary extends past the AOI's own shape but still
        # within its rectangular bounding box + pad) gets a hard straight
        # edge wherever it crossed the rectangle, which reads as "half the
        # shape is a giant square" on the map. It's still needed as the FIRST
        # step (a rect clip is ~free; it's what brought Toronto's 3.8M-vertex
        # province polygon down to a join-able ~144K vertices before anything
        # else could touch it -- see the fix above). This second pass
        # intersects against the real, UN-padded AOI polygon (not its
        # bounding box), replacing the straight bbox edges with the actual
        # AOI outline -- cheap now that `simplify()` above already brought
        # vertex counts down to the thousands, not millions.
        #
        # 2026-09-01 follow-up (live user report, same day): this originally
        # buffered the AOI polygon by `_pad` (0.05deg, ~5.5km) before
        # intersecting, on the theory that boundary-straddling census units
        # shouldn't be sliced off -- but a buffer here means the DISPLAYED
        # geometry visibly bulges past the real AOI edge by that same ~5.5km
        # everywhere, which is exactly the "buffer around the AOI on the map"
        # the user then reported. Any buffer is a processing concern (the
        # `_join_census`/`_join_polygon_stats` paths already handle their own
        # padding independently for exactly that purpose); what's DISPLAYED
        # must be clipped to the real, unpadded AOI shape. `_geo_idx` was
        # already reset after the bbox-rect clip above, well before any
        # h3-cell join happens, so an exact (non-buffered) AOI clip here
        # cannot itself drop cells from the aggregation -- it only trims the
        # render geometry.
        # Bug fix (2026-09-03, live user report -- Boston: a tiny border
        # block reporting population in the tens of thousands, next to
        # normal ~300-person interior blocks; explicit follow-up: "use for
        # any population and anything the real aoi not the aoi with
        # buffer... crop the geometries of the census by the aoi and do an
        # area weighted resampling of all absolute columns"). Aggregation
        # is against this real, unpadded AOI-clipped geometry (not a wider
        # unclipped/padded one -- the real AOI is what should define the
        # study area for population purposes; a buffer is only for
        # avoiding a stop-cropping boundary effect elsewhere, not this).
        # The border-inflation bug itself is fixed a few lines below by
        # switching `population`'s aggregation to real area-fraction
        # weighting (`intersection_mode="exact"`) instead of unweighted
        # `intersects` membership, which is what let a hairline-thin
        # boundary sliver (confirmed live: 202 Boston blocks with literal
        # `area_m2 == 0` but nonzero population) inherit each merely-
        # touching h3 cell's FULL population.
        #
        # That fixes the h3-DERIVED fallback population, but a REAL
        # census-source population (joined by GEOID a few lines below, from
        # `loader()`'s own columns -- already present on `census_gdf` before
        # any clipping happens here) is a per-block constant, independent of
        # geometry: it does NOT shrink just because this polygon's DISPLAY
        # geometry gets clipped down to a small AOI-boundary fragment. A
        # real block mostly outside the AOI, with only a sliver of its true
        # extent overlapping, still reports its FULL population on that
        # sliver otherwise (confirmed live: San Francisco blockgroup
        # 060855046011, area_m2=3.0, population=1185 -> ~394M/km2).
        # `_orig_area_m2` (this block's true area, from the cheap
        # generous-pad bbox clip above, NOT the expensive full unclipped
        # geometry -- Toronto's province-level polygon is 3.8M vertices
        # before that clip; computing area on it directly here would
        # reintroduce the exact perf problem that clip exists to avoid)
        # is snapshotted now, before the real-AOI intersection shrinks
        # `census_gdf.geometry`, so every absolute column can be scaled by
        # the fraction of this block's true area that actually survives the
        # clip -- applied a bit further down, once every absolute column
        # (real census join, LODES jobs, DHC race, h3 fallback) exists.
        census_gdf["_orig_area_m2"] = census_gdf.to_crs(census_gdf.estimate_utm_crs()).geometry.area
        _aoi_union = shapely.make_valid(aoi.to_crs(census_gdf.crs).geometry.union_all())
        try:
            census_gdf["geometry"] = census_gdf.geometry.intersection(_aoi_union)
        except shapely.errors.GEOSException:   # a self-touching census polygon: repair it (buffer(0)) and clip again
            census_gdf["geometry"] = census_gdf.geometry.buffer(0).intersection(_aoi_union)
        # Bug fix (2026-09-04, live: Boston's "tract" level crashed the new
        # area-weighted population join with "df2 contains mixed geometry
        # types"). A polygon clipped along just its boundary EDGE (not
        # interior) by the intersection above can come back as a
        # non-empty-but-degenerate LineString/GeometryCollection instead of
        # a Polygon -- `~geometry.is_empty` doesn't catch this (it's
        # genuinely non-empty, just zero-area and the wrong dimensionality).
        # `gpd.overlay`'s exact-area-fraction join (used by the population
        # aggregation below) can't handle a mix of Polygon and
        # non-Polygon geometry in the same call. These slivers have no real
        # area to represent anyway, so they're dropped here the same way a
        # genuinely empty intersection already was.
        census_gdf = census_gdf[census_gdf.geom_type.isin(("Polygon", "MultiPolygon"))]
        census_gdf = census_gdf[~census_gdf.geometry.is_empty].reset_index(drop=True)
        census_gdf["_geo_idx"] = census_gdf.index

        # Bug fix (2026-08-29, live user report -- Guadalajara): this used to
        # join on each h3 cell's CENTROID with predicate="within", i.e. "does
        # this cell's centroid point fall strictly inside the polygon". For a
        # small/oddly-shaped census block (an AGEB, a US block, ...) that is
        # smaller than or comparable to a single res-11 hexagon (~2,150 m^2),
        # it's common for every cell that actually overlaps the block to have
        # its centroid fall just outside it -- leaving zero matched cells (or,
        # worse, only the low/zero-population edge cells matching while the
        # real high-access, high-population cells' centroids land a few
        # meters over the line) even though the hexagon grid visibly covers
        # the block with real access-score data. That's exactly the "block
        # shows 0/null while its own underlying hexagons show high access
        # scores" pattern reported live: a zero-population matched set drives
        # the weighted average to NaN, which the `fillna(0.0)` below then
        # paints as a hard zero instead of falling back to the (populated)
        # nearby cells the unmatched-fallback branch was meant to catch.
        # Joining on each cell's real hexagon geometry (`h3_grid.geometry`)
        # with predicate="intersects" instead fixes both failure modes at
        # once: every res-11 cell that spatially touches the polygon at all
        # -- not just the ones whose single point happens to land inside --
        # contributes to the population-weighted average, per the user's own
        # spec ("population weighted average of all hexagon cells of highest
        # resolution touching that geometry").
        # Perf fix (2026-08-29, live: Toronto's dissemination-block level --
        # ~2M+ res-11 hexagons x tens of thousands of small blocks -- never
        # finished this join, still running after 2+ hours of CPU time with
        # the naive "every hexagon x every polygon" intersects join below).
        # A cell with zero population contributes exactly zero to the
        # population-weighted average (`_weighted = level_of_service *
        # population` is 0, and it adds 0 to both the numerator and
        # `grouped["population"]`'s denominator) -- excluding it from the
        # join changes nothing about the result, only how many candidate
        # pairs `gpd.sjoin`'s spatial index has to exact-test. Real metros
        # (Toronto included) have large shares of their finest h3 grid over
        # parks/water/industrial/unpopulated land that this drops for free.
        populated_mask = h3_grid["population"].to_numpy(dtype=float) > 0
        pts = gpd.GeoDataFrame(
            {
                # `.to_numpy()` (not the Series itself) would drop h3_grid's
                # index; GeoDataFrame's constructor then assigns default
                # RangeIndex labels to these columns while `h3_grid.geometry`
                # below keeps h3_grid's *real* (possibly non-contiguous -- e.g.
                # after `filter_populated`'s boolean-mask row drops upstream)
                # index, and geopandas attaches a `geometry=` Series by index
                # label, not position. That silently scrambles which
                # level_of_service/population pairs with which cell geometry
                # instead of erroring, corrupting the population-weighted
                # aggregation below. Keeping these as Series (h3_grid's own
                # index) makes every column -- including the geometry -- align
                # on the same labels, position-for-position.
                "level_of_service": h3_grid.loc[populated_mask, "level_of_service"],
                "population": h3_grid.loc[populated_mask, "population"],
            },
            geometry=h3_grid.loc[populated_mask].geometry,
            crs=h3_grid.crs,
        ).to_crs(census_gdf.crs)
        # 2026-08-30: replaced the ad-hoc `gpd.sjoin(..., predicate="intersects")`
        # + manual weighted-average `groupby` with `geohierarchy`'s generic
        # geometry-to-geometry resampler (`GeoHierarchy.add_vector_data`) +
        # schema-driven aggregation strategy (`Mean(weight_column="population")`).
        # This reproduces the same population-weighted average of every h3
        # cell touching a polygon (see spec comment above). `Sum(...)` for
        # `population` reproduces the old `grouped["population"]` h3-population
        # fallback below unchanged. `fill_null=None` leaves genuinely
        # unmatched polygons null so the existing nearest-cell fallback below
        # (unchanged) still catches them -- `add_vector_data` itself only
        # offers a flat scalar/dict fill, not a "borrow the closest real
        # measurement" fallback, so that part of the old logic is kept as-is
        # rather than replaced.
        #
        # `geoweighted=False` here, deliberately, NOT the more area-accurate
        # `geoweighted=True`: an earlier version of this code used
        # `geoweighted=True` (weight each straddling cell's contribution by
        # its true overlap-area FRACTION with each polygon, on top of the
        # population weight), which is more geometrically faithful for a
        # cell split across two polygons but requires computing an exact
        # intersection geometry/area for every overlapping (cell, polygon)
        # pair -- real overlay work, not just a boolean membership test.
        # Reproduced live on Toronto's dissemination-block level (1.31M h3
        # cells x tens of thousands of small polygons): `geoweighted=True`
        # ran for the full 30-minute verification window without finishing.
        # Plain population weighting (this line) only needs `intersects`
        # membership, matching the user's own spec ("population weighted
        # average of all hexagon cells ... touching that geometry" -- no
        # area-fraction requirement was asked for), and is dramatically
        # cheaper at this scale. If a future correctness need for exact
        # area-fraction weighting arises, it should be reintroduced with a
        # chunked/bounded-candidate strategy, not a flat `geoweighted=True`
        # over the whole grid again.
        if chunk_h3_resolution is not None:
            # Memory-bounded path (2026-09-04, live Boston OOM x3 -- see
            # `_census_polygon_agg_chunked`'s docstring for the full
            # rationale/correctness argument). Mathematically exact, not an
            # approximation of the unchunked path below -- both quantities
            # are re-expressed as sums that are linear in `pts`'s rows, so
            # chunking `pts` and combining per-chunk partial sums reproduces
            # the identical result the single whole-grid `add_vector_data`
            # calls below would.
            _t1 = _time.time()
            agg_result = _census_polygon_agg_chunked(
                pts, h3_grid.loc[populated_mask, "h3_cell"], census_gdf, chunk_h3_resolution
            )
            print(f"[TIMING] {level}: chunked add_vector_data (level_of_service + population) took {_time.time()-_t1:.1f}s, pts rows={len(pts)}")
        else:
            _t1 = _time.time()
            hierarchy = GeoHierarchy(crs=census_gdf.crs)
            hierarchy.add_level(
                "_census_map_level", census_gdf[["_geo_idx", "geometry"]].copy(), id_col="_geo_idx"
            )
            print(f"[TIMING] {level}: add_level took {_time.time()-_t1:.1f}s")
            _t2 = _time.time()
            hierarchy.add_vector_data(
                pts,
                level="_census_map_level",
                columns=["level_of_service"],
                agg={
                    "level_of_service": Mean(weight_column="population", geoweighted=False),
                },
                fill_null=None,
                # Bug fix (2026-09-01, live user report -- Guadalajara: blocks
                # showing 0 access despite the real hexagons underneath having
                # real access scores). `geoweighted=False` above (correctly
                # asking for NO area-fraction weighting) also silently defaulted
                # `get_id_mapping`'s matching itself down to "centroid" mode --
                # a hexagon only counts toward a polygon if its CENTROID falls
                # inside it, not whether the hexagon's real shape overlaps the
                # polygon at all. For a small/oddly-shaped block comparable in
                # size to a single hexagon (exactly the case this function's own
                # `unmatched`/nearest-cell fallback below was written to catch
                # for polygons with ZERO matched cells) this silently dropped
                # individual straddling hexagons from polygons that still had
                # other matched cells, so the fallback never triggered but the
                # population-weighted average was computed over the wrong
                # (incomplete) set of cells. `intersection_mode="intersects"` is
                # a real geometry-touches test -- same cost as "centroid" (an
                # indexed `sjoin`, not `gpd.overlay`), just without the
                # centroid substitution. Kept for `level_of_service` (not an
                # absolute/summed quantity -- see the area-weighted `population`
                # join right below, added for exactly that distinction).
                intersection_mode="intersects",
            )
            print(f"[TIMING] {level}: add_vector_data (level_of_service) took {_time.time()-_t2:.1f}s, pts rows={len(pts)}")
            # Bug fix (2026-09-03, live user report -- Boston border-sliver
            # population inflation; explicit follow-up: "do an area weighted
            # resampling of all absolute columns"). `population` is an
            # absolute/summed quantity, unlike `level_of_service` above --
            # `intersection_mode="exact"` computes each matched h3 cell's real
            # overlap-AREA FRACTION with this polygon and weights its
            # contribution by that fraction, instead of counting every merely-
            # touching cell's full population once. Real geometric overlay
            # (`gpd.overlay`), not the cheap indexed `sjoin` the
            # `level_of_service` join above uses -- a separate, smaller call
            # (`columns=["population"]` only) rather than folding into the call
            # above, since mixing an "exact" column into that call would force
            # the same expensive path onto `level_of_service` too.
            _t3 = _time.time()
            hierarchy.add_vector_data(
                pts,
                level="_census_map_level",
                columns=["population"],
                agg={"population": Sum(geoweighted=True)},
                fill_null=None,
                intersection_mode="exact",
            )
            print(f"[TIMING] {level}: add_vector_data (population, area-weighted) took {_time.time()-_t3:.1f}s")
            _hier_id_col = hierarchy.id_cols["_census_map_level"]
            agg_result = hierarchy["_census_map_level"][[_hier_id_col, "level_of_service", "population"]]
            agg_result = agg_result.rename(
                columns={_hier_id_col: "_geo_idx", "population": "_h3_fallback_population"}
            )
        grouped_index = agg_result.loc[agg_result["level_of_service"].notna(), "_geo_idx"]

        census_gdf = census_gdf.merge(agg_result, on="_geo_idx", how="left")
        # Bug fix (2026-08-29, live user report): a census polygon with no
        # real census-source population match (e.g. a gap polygon outside
        # INEGI/ACS/etc coverage that the map still draws) used to show a
        # null "Population" in the click popup even though the h3 grid this
        # polygon was built from has a real, never-null `population` column
        # (from `population_and_access_to_h3`'s WorldPop base raster -- see
        # `_population_column`'s docstring). `_h3_fallback_population` above is
        # already the sum of that bare h3 `population` over every h3 cell
        # touching this polygon -- carry it along as a fallback so
        # whichever column `_population_column` picks below (the real
        # census-source one, renamed to bare `population` later by
        # `_rename_canonical_columns`) gets backfilled instead of staying
        # null. Country-agnostic: no `inegi_population`-specific special
        # case, just the same `_population_column` lookup used everywhere
        # else.
        # A polygon with no h3 centroid strictly *within* it -- e.g. a small
        # or narrow block group whose every candidate centroid happens to
        # fall a few meters over the line into a neighbor, which real Boston
        # data shows for ~8% of block groups (disproportionately small,
        # densely-populated ones -- ~490k people's worth) -- still has real
        # transit access; it's the *point-in-polygon test* that came up
        # empty, not the area's LOS. Falling back straight to 0.0 here (as a
        # previous version of this function did) paints every one of those
        # block groups as "zero access" regardless of their true score,
        # which is exactly the "solid zero patches in otherwise well-served
        # areas" pattern reported from the first live look at Boston's
        # census layer. Nearest-h3-centroid is the same "no access
        # information -> borrow the closest real measurement" fallback
        # `_join_polygon_stats` already uses elsewhere in this module.
        unmatched = census_gdf.loc[~census_gdf["_geo_idx"].isin(grouped_index)]
        if not unmatched.empty:
            # `sjoin_nearest` needs a projected (metric) CRS -- both `pts` and
            # `census_gdf` are lat/lon here, and nearest-in-degrees is skewed
            # enough at Boston's latitude to occasionally pick a farther-away
            # h3 cell over a closer one, undermining the whole point of the
            # fallback. `estimate_utm_crs()` picks one appropriate zone for
            # this AOI (same helper already used for `area_m2` above).
            utm_crs = census_gdf.estimate_utm_crs()
            unmatched_pts = gpd.GeoDataFrame(
                {"_geo_idx": unmatched["_geo_idx"].to_numpy()},
                geometry=unmatched.geometry.representative_point(),
                crs=census_gdf.crs,
            ).to_crs(utm_crs)
            # Nearest-cell fallback uses each cell's true centroid (not its
            # hexagon polygon, which `pts.geometry` now holds for the
            # intersects-based aggregation above) -- a point-to-point nearest
            # search is the correct "closest real measurement" semantics and
            # matches this function's pre-existing behavior here.
            centroid_pts_utm = gpd.GeoDataFrame(
                {"level_of_service": h3_grid["level_of_service"], "population": h3_grid["population"]},
                geometry=centroids,
                crs=h3_grid.crs,
            ).to_crs(utm_crs)
            nearest = _sjoin_nearest_chunked(unmatched_pts, centroid_pts_utm, id_col="_geo_idx")
            nearest = nearest.drop_duplicates(subset="_geo_idx").set_index("_geo_idx")
            census_gdf = census_gdf.merge(
                nearest[["level_of_service", "population"]].rename(
                    columns={"level_of_service": "_nearest_level_of_service", "population": "_nearest_population"}
                ),
                left_on="_geo_idx",
                right_index=True,
                how="left",
            )
            census_gdf["level_of_service"] = census_gdf["level_of_service"].fillna(census_gdf["_nearest_level_of_service"])
            # Same gap-polygon fallback as the `grouped["population"]` merge
            # above, for polygons that had no h3 centroid within them at all
            # (so no `grouped` row to fall back on) -- use the single nearest
            # h3 cell's own `population` as a last-resort estimate rather
            # than leaving this polygon's population null too.
            census_gdf["_h3_fallback_population"] = census_gdf["_h3_fallback_population"].fillna(
                census_gdf["_nearest_population"]
            )
            census_gdf = census_gdf.drop(columns=["_nearest_level_of_service", "_nearest_population"])
        # A block group with no h3 centroid within it *and* no h3 grid to fall
        # back on at all (e.g. an empty study) has no access information --
        # but "no reachable transit" is exactly what 0 means on this scale,
        # and a null here would knock the polygon out of the map's color
        # scale, its popup and every aggregate. See `_add_h3_grid`.
        census_gdf["level_of_service"] = census_gdf["level_of_service"].fillna(0.0)
        census_gdf["area_m2"] = census_gdf.to_crs(census_gdf.estimate_utm_crs()).geometry.area
        # Backfill whichever population column this polygon's own
        # `_population_column` will resolve to (the real census-source
        # column if one exists for this country/level, else none) with the
        # h3-derived fallback computed above, so every polygon -- gap cells
        # included -- has a real, non-null population figure. Create a bare
        # `population` column outright when this source has no population
        # column of its own at all.
        pop_col_for_fill = _population_column(census_gdf)
        if pop_col_for_fill is None:
            census_gdf["population"] = census_gdf["_h3_fallback_population"]
        else:
            census_gdf[pop_col_for_fill] = census_gdf[pop_col_for_fill].fillna(
                census_gdf["_h3_fallback_population"]
            )
        # Item 4 (`worldpop_population_density`): `_h3_fallback_population` is
        # the real population-weighted-aggregation SUM of the h3 grid's own
        # WorldPop-derived `population` over every cell touching this
        # polygon -- a genuine WorldPop measurement, independent of whether a
        # real census population column exists for this level/country. Kept
        # under its own name (`worldpop_population_map_source`, checked by
        # `_worldpop_population_column` alongside bare `population`) rather
        # than written to bare `population` itself: `_population_column`
        # checks bare `population` FIRST, so writing it here would make every
        # OTHER use of `_population_column` on `census_gdf` (`pop_density`,
        # `equity_flag`, `filter_populated`, ...) silently prefer WorldPop
        # over the real census count whenever both exist -- a behavior change
        # far beyond this item's scope.
        census_gdf["worldpop_population_map_source"] = census_gdf["_h3_fallback_population"]
        census_gdf = census_gdf.drop(columns=["_h3_fallback_population"])
        # people/km2, matching `_add_h3_grid` -- the census-polygon shape option
        # feeds the same map field selectors as the h3 grids, so the two must
        # not disagree on units.
        pop_col_for_density = _population_column(census_gdf)
        census_gdf["pop_density"] = (
            census_gdf[pop_col_for_density] if pop_col_for_density else np.nan
        ) / (census_gdf["area_m2"] / 1e6)
        # Same jobs columns as the h3 grids carry, so the "census geometries"
        # shape option offers the same (default) variable rather than falling
        # back to `pop_density` alone. Joined by GEOID -- these *are* the
        # LODES/BA geometries, no apportionment needed. USA (LODES) and
        # Germany (BA's AMK, see `_join_jobs`) are the only two countries
        # wired here with a workplace-jobs-flow source.
        try:
            if country == "USA":
                from pycensus.census.countries.usa import lodes_wac as lodes

                jobs_gdf = lodes.load_wac(
                    aoi=aoi, states=states_arg, level=level, cache_dir=str(cache_dir)
                )
                jobs_keep_columns = LODES_KEEP_COLUMNS
            elif country == "DEU":
                from pycensus.census.countries.germany import ba

                jobs_gdf = ba.load(aoi=aoi, regions=states_arg, level=level, cache_dir=str(cache_dir))
                jobs_keep_columns = GERMANY_BA_KEEP_COLUMNS
            else:
                raise ValueError(f"no LODES-equivalent jobs source for country={country!r}")
            census_gdf = census_gdf.merge(
                jobs_gdf[["GEOID", *sorted(jobs_keep_columns)]], on="GEOID", how="left"
            )
        except Exception as exc:  # pragma: no cover - level/state availability varies
            print(f"[pipeline] no LODES/BA jobs for census level {level!r}: {exc}")
        # 2026-08-25 audit: real BA Pendlerstatistik commuter cross-tab, same
        # "census geometries" shape option as the jobs merge just above --
        # see `GERMANY_BA_PENDLER_KEEP_COLUMNS`/`_join_jobs`. Germany-only,
        # joined by GEOID (no apportionment needed, same as BA's AMK).
        if country == "DEU":
            try:
                from pycensus.census.countries.germany import ba_pendler

                pendler_gdf = ba_pendler.load(
                    aoi=aoi, regions=states_arg, level=level, cache_dir=str(cache_dir)
                )
                pendler_cols = [c for c in GERMANY_BA_PENDLER_KEEP_COLUMNS if c in pendler_gdf.columns]
                census_gdf = census_gdf.merge(
                    pendler_gdf[["GEOID", *sorted(pendler_cols)]], on="GEOID", how="left"
                )
            except Exception as exc:  # pragma: no cover - level/state availability varies
                print(f"[pipeline] no BA Pendlerstatistik for census level {level!r}: {exc}")
        # Decennial DHC race/gender/age data, the same way the h3-grid side of
        # the pipeline gets it via `_join_race` -- without this merge, every
        # `dhc_*_share` field (white/black/asian/native/other_race/hispanic/
        # female/male/children/elderly) is entirely ABSENT from every
        # census-polygon feature's properties (not degenerate -- genuinely
        # missing), so `build.py`'s opacity/color-by-field selectors silently
        # fall back to a fixed value whenever a `dhc_*` field is picked on the
        # "Census geometries" shape option. Joined by GEOID, same as LODES
        # jobs just above -- these are the DHC geometries too, no apportionment
        # needed. Originally found and patched locally in
        # `TransitLOSStudies/boston_city/rebuild_map_only.py`; ported here so
        # every US city gets it automatically. USA-only: no country wired here
        # has a decennial-race-equivalent source (see `_join_race`).
        try:
            if country != "USA":
                raise ValueError(f"no race/ethnicity census source for country={country!r}")
            from pycensus.census.countries.usa import dhc

            dhc_gdf = dhc.load(aoi=aoi, states=states_arg, level=level, cache_dir=str(cache_dir))
            # Same native-level guard as `_usa_map_census_loader`/`_join_race`
            # (2026-09-29) -- DHC is native at every `MAP_CENSUS_LEVELS` entry
            # this loop visits, so this should never fire; caught by the
            # surrounding `try` like any other failure here, degrading to
            # "no DHC data for this level" rather than crashing the study.
            assert dhc_gdf.empty or dhc_gdf.attrs.get("native_level"), (
                f"DHC fetch at level {level!r} was not native."
            )
            dhc_gdf = _restore_population_prefix(dhc_gdf, "dhc")
            dhc_cols = [c for c in DHC_KEEP_COLUMNS if c in dhc_gdf.columns]
            census_gdf = census_gdf.merge(dhc_gdf[["GEOID", *sorted(dhc_cols)]], on="GEOID", how="left")
        except Exception as exc:  # pragma: no cover - network/data availability varies
            print(f"[pipeline] no DHC race/gender/age data for census level {level!r}: {exc}")
        # Rename every canonical field to its bare `global_schema.json` name
        # now that every source (the country's own loader, plus LODES/DHC for
        # USA) has landed its prefixed columns on `census_gdf` -- see
        # `_rename_canonical_columns`. This is the census-polygon overlay's
        # own independent join path (distinct from `_join_census`'s h3-grid
        # path above), so it needs its own call.
        census_gdf = _rename_canonical_columns(census_gdf)
        # Bug fix (2026-09-04, live: San Francisco blockgroup 060855046011,
        # area_m2=3.0, population=1185 -> ~394M people/km2 -- explicit
        # follow-up request: "do an area weighted resampling of all
        # absolute columns"). Every ABSOLUTE column joined onto this
        # polygon so far (real census-source population, LODES jobs, DHC
        # race/gender/age counts, housing units, ...) is a per-block
        # constant from `loader()`'s own data -- unrelated to the display
        # geometry -- so none of it shrinks on its own just because this
        # polygon got clipped down to a small AOI-boundary fragment above.
        # Scaling every one by `_orig_area_m2`'s survival fraction
        # apportions each block's attributes across its clipped pieces the
        # standard dasymetric way (uniform-density assumption): a fragment
        # keeping 10% of the block's true area gets 10% of its population/
        # jobs/etc, and (population/area) -- true density -- comes out
        # unchanged, exactly the "no free lunch, no over-count" behavior
        # meant here. Must happen BEFORE `_add_pop_jobs_columns`/
        # `_add_derived_density_columns` below, which derive `pop_density`/
        # `jobs_density`/`pop_jobs_density`/etc from these same absolute
        # columns and `area_m2` -- computing them from already-scaled
        # absolutes (over the already-small clipped `area_m2`) is what
        # makes the resulting density come out correctly UNCHANGED, instead
        # of needing a second manual recompute here.
        _orig_area = census_gdf["_orig_area_m2"].to_numpy(dtype=float)
        _area_fraction = np.where(_orig_area > 0, census_gdf["area_m2"].to_numpy(dtype=float) / _orig_area, 1.0)
        _area_fraction = np.clip(_area_fraction, 0.0, 1.0)
        from transitlos.map.build import absolute_fields as _absolute_fields

        _non_scalable = {
            "GEOID", "_geo_idx", "geometry", "area_m2", "_orig_area_m2",
            "level_of_service", "h3_cell", "equity_flag",
        }
        _scale_cols = [
            c for c in _absolute_fields(list(census_gdf.columns))
            if c not in _non_scalable and pd.api.types.is_numeric_dtype(census_gdf[c])
        ]
        for c in _scale_cols:
            census_gdf[c] = census_gdf[c].to_numpy(dtype=float) * _area_fraction
        census_gdf = census_gdf.drop(columns=["_orig_area_m2"])
        census_gdf = _resample_missing_h3_counts(census_gdf, h3_grid, populated_mask)
        census_gdf = _add_share_columns(census_gdf)
        census_gdf = _add_pop_jobs_columns(census_gdf)
        census_gdf = _add_derived_density_columns(census_gdf)
        census_gdf = _add_worldpop_outside_census_columns(census_gdf)
        census_gdf = _finalize_population_columns(census_gdf, True, worldpop_only_population)
        if GBA_VS_CENSUS_COLUMNS[0] not in census_gdf.columns:
            census_gdf = _add_gba_vs_census(census_gdf)
        census_gdf = _order_map_fields(census_gdf)
        # Same occupancy rule as the h3 grids (`filter_populated`): a census
        # polygon with people and/or jobs is drawn regardless of its access
        # score, an empty one is not drawn regardless of it.
        census_gdf = filter_populated(census_gdf)
        # The development-opportunity flag, computed on the polygon's *own*
        # density and level of service. For a US city this is what the map's
        # development overlay is rendered from (see
        # `development_gdf_for_map`); the h3 version stays as the fallback
        # for a study with no census geometry at all.
        pop_col = _population_column(census_gdf)
        census_gdf["equity_flag"] = equity_flag(
            census_gdf[development_density_column(census_gdf)].to_numpy(),
            census_gdf["level_of_service"].to_numpy(),
            census_gdf[pop_col].to_numpy() if pop_col else np.zeros(len(census_gdf)),
        )
        # Real, live-reproduced OOM cause (2026-08-22): Boston's aoi.gpkg
        # covers the entire state of Massachusetts (its AOI spans RI to NH),
        # and real US Census "block" boundaries (added tonight, see
        # `MAP_CENSUS_LEVELS`) follow actual street segments -- ~103,000 real
        # MA blocks vs. only ~5,100 blockgroups, each with far more vertices
        # per polygon than a blockgroup/tract/county boundary. This is a ~20x
        # jump in both polygon COUNT and per-polygon geometric complexity
        # landing on the tile-build step right as it's already tight on
        # memory (see the `RAYON_NUM_THREADS`/column-pruning fixes earlier
        # tonight in `write_level_tiles`, which reduced but didn't eliminate
        # the crash). `simplify(preserve_topology=True)` at a tolerance well
        # below real TIGER/Line boundary accuracy (~0.5m) trims excess
        # vertices before tiling ever sees the geometry -- the same
        # "precision the rendered map can't show anyway" tradeoff already
        # applied to tile properties (float32 downcast) this session, just
        # for geometry instead of attributes. ~0.00003 deg is ~3m at these
        # latitudes -- far finer than a census block's own real positional
        # accuracy, so no visible boundary distortion at the zoom levels
        # these tiles actually render at.
        census_gdf["geometry"] = census_gdf.geometry.simplify(0.00003, preserve_topology=True)
        result[level] = census_gdf.drop(columns=["_geo_idx"])

    return result


# --------------------------------------------------------------------------
# Development-opportunity overlay geometry.
#
# `transitlos.map.build.build_city_map` renders that overlay from whichever
# h3 resolution carries `equity_flag`, because h3 is the only geometry it can
# assume every study has. Where real census block groups *are* available they
# are strictly better: they are the geometry the population/jobs counts are
# actually published on, so a block group's density is a measured quantity
# rather than an apportioned one, and its boundaries are the ones a planner
# reads a "this area is under-built / under-served" claim against.
#
# `build.py` is shared with the other city studies (and, today, with a
# parallel effort), so rather than edit it, the overlay's *tiles* are
# regenerated from the census polygons immediately after `build_city_map`
# returns, under the exact level name and tile directory the generated HTML
# already points at. Nothing about `_development_style_js`'s rendering is
# hexagon-specific -- it keys purely on `properties.equity_flag` and draws a
# border, so arbitrary polygons render identically.
# --------------------------------------------------------------------------

DEVELOPMENT_CENSUS_LEVEL = "blockgroup"
#: Development view is always the H3 res-8 overlay (one regression per city, `force_development_resolution`); True restores the census-polygon overlay.
DEVELOPMENT_FROM_CENSUS = False


def development_gdf_for_map(
    census_by_level: Optional[dict[str, gpd.GeoDataFrame]]
) -> Optional[gpd.GeoDataFrame]:
    """The census polygons the development overlay should be drawn on, if any.

    Returns `None` for a study with no census geometry (non-US, or a US city
    whose census fetch failed), which leaves the h3-hexagon overlay
    `build_city_map` builds by itself in place -- the documented fallback.
    """
    if not census_by_level or not DEVELOPMENT_FROM_CENSUS:
        return None
    gdf = census_by_level.get(DEVELOPMENT_CENSUS_LEVEL)
    if gdf is None:
        # `DEVELOPMENT_CENSUS_LEVEL` ("blockgroup") is USA-shaped and won't
        # be a key for any other country (Mexico has "ageb"/"municipality"/
        # "state", Israel has "locality", etc). `census_by_level` is built by
        # `_census_geometries_with_score`, which now always returns it
        # coarsest-to-finest regardless of country (see that function's
        # `reversed(levels)` fix) -- so the LAST value is the finest real
        # level actually available for THIS country -- used here instead of
        # re-checking the USA-only `MAP_CENSUS_LEVELS` tuple, which
        # previously left every non-US city's development overlay falling
        # back to h3 hexagons even when real census polygons (e.g. Mexico's
        # AGEBs) were fetched successfully just above.
        values = list(census_by_level.values())
        gdf = values[-1] if values else None
    if gdf is None or gdf.empty or "equity_flag" not in gdf.columns:
        return None
    return gdf


def _development_level_name(h3_by_resolution: dict[int, gpd.GeoDataFrame]) -> str:
    """Reproduce `build_city_map`'s own choice of development-overlay level name.

    Must stay in step with `transitlos.map.build.build_city_map`: the level
    name is baked into the tile URL template of the HTML it generates, so the
    replacement tiles have to land under exactly that name.
    """
    resolutions = sorted(h3_by_resolution)
    candidates = [r for r in resolutions if "equity_flag" in h3_by_resolution[r].columns] or resolutions
    le9 = [r for r in candidates if r <= 9]
    return f"h3_{max(le9) if le9 else min(candidates)}"


def rebuild_development_tiles_from_census(
    development_gdf: gpd.GeoDataFrame,
    h3_by_resolution: dict[int, gpd.GeoDataFrame],
    tiles_dir: Path,
    out_dir: Path,
) -> None:
    """Redraw the development overlay's vector tiles from census polygons.

    Call *after* `build_city_map` (which always rebuilds these tiles itself
    from h3). Wipes and rewrites `tiles_dir/development/<level>` so no stale
    hexagon tile survives in an area the census polygons don't cover.
    """
    import shutil

    from geohierarchy import Max
    from geohierarchy.core import GeoHierarchy
    from geohierarchy.maps.folium.render import HierarchyMap

    level = _development_level_name(h3_by_resolution)
    dev_dir = Path(tiles_dir) / "development"
    shutil.rmtree(dev_dir, ignore_errors=True)
    dev_dir.mkdir(parents=True, exist_ok=True)

    gdf = development_gdf.to_crs(4326).reset_index(drop=True).copy()
    # `build_city_map` configures this level with `id_col="h3_cell"`; the id is
    # only ever used as a feature key, so the block group's own GEOID serves.
    gdf["h3_cell"] = gdf["GEOID"].astype(str) if "GEOID" in gdf.columns else gdf.index.astype(str)

    hierarchy = GeoHierarchy(crs=4326)
    hierarchy.add_level(level, gdf, id_col="h3_cell", agg=Max())
    dev_map = HierarchyMap(hierarchy, levels=[level], tiles_dir=str(Path(tiles_dir) / "development"))
    from transitlos.map import build as _map_build

    dev_map.configure_level(
        level,
        style_js=_map_build._development_style_js(),
        popup_fields=["equity_flag", "level_of_service"],
    )
    dev_map.set_resolution(level, min_zoom=0, max_zoom=25)
    import contextlib

    with contextlib.chdir(out_dir):
        dev_map.build()
    print(f"[pipeline] development overlay rebuilt from {len(gdf)} census polygons -> {dev_dir}/{level}")


def _resample_h3(h3_grid, target_resolution: int, sum_cols: list[str]) -> pl.DataFrame:
    """Resample an H3 grid to a coarser resolution.

    `population` and the additive columns in `sum_cols` (e.g. Census counts)
    get summed. `level_of_service` is aggregated as a population-weighted mean,
    not a plain per-cell average -- a plain mean would let a near-empty cell
    swing the coarser cell's score as much as a densely-populated one; the
    rate-like `sum_cols` (income/rent medians and means) get the same
    population-weighted treatment, and share columns are left out to be
    recomputed by `_add_share_columns`.

    Args:
        h3_grid: Grid with `h3_cell`, `population`, `level_of_service`, and
            (optionally) `sum_cols` columns. Either a GeoDataFrame or a
            Polars DataFrame (`read_h3_grid_table`); geometry is never read
            here, so the Polars form is both cheaper and preferred -- a
            pandas source has to be copied column-subset-wise and converted,
            which on the Boston metro grid is ~700 MB of transient garbage
            *per resolution*.
        target_resolution: Coarser H3 resolution to resample onto (must be
            <= the grid's current resolution).
        sum_cols: Additional census columns (e.g. `acs_*`) to carry through,
            each aggregated according to its kind (see below).

    Returns:
        Polars DataFrame with `h3_cell`, `population`, `level_of_service`, and
        the non-share subset of `sum_cols`, one row per resampled cell.
    """
    # Three kinds of column need three different aggregations, and summing
    # them all (as this used to) is wrong for two of them:
    #  - counts        -> sum (population, households, workers, ...)
    #  - rate-like     -> population-weighted mean. Summing a *median* across
    #                     child cells is meaningless and inflates it by
    #                     roughly the number of children (Boston's resampled
    #                     `acs_income_median_household` was reaching the
    #                     millions before this).
    #  - share columns -> dropped here entirely and recomputed downstream by
    #                     `_add_share_columns` from the freshly summed
    #                     numerator/denominator, which is exact; aggregating
    #                     the ratio itself never is.
    share_cols = [c for c in sum_cols if c in SHARE_COLUMNS]
    rate_cols = [c for c in sum_cols if c not in share_cols and _is_rate_col(c)]
    count_cols = [c for c in sum_cols if c not in share_cols and c not in rate_cols]

    keep_cols = ["h3_cell", "level_of_service", "population"] + count_cols + rate_cols
    df = h3_grid.select(keep_cols) if isinstance(h3_grid, pl.DataFrame) else pl.from_pandas(h3_grid[keep_cols].copy())
    df = df.with_columns(
        (pl.col("level_of_service").fill_null(0.0) * pl.col("population")).alias("_weighted_access"),
        # Plain (unweighted) sum + count, used only as the fallback for a
        # coarse cell whose children all have zero population: the
        # population-weighted mean is 0/0 there, and the answer to "what is
        # this cell's level of service" is still a number (the plain mean of its
        # children), never null -- see `_add_h3_grid`'s note on why a null
        # level of service is never acceptable.
        pl.col("level_of_service").fill_null(0.0).alias("_access_sum"),
        pl.lit(1.0).alias("_access_n"),
    )
    # A rate column that is null on a cell must contribute neither value nor
    # weight, so each one carries its own population-weight companion.
    for col in rate_cols:
        df = df.with_columns(
            (pl.col(col) * pl.col("population")).alias(f"_weighted_{col}"),
            pl.when(pl.col(col).is_not_null()).then(pl.col("population")).otherwise(0.0).alias(f"_weight_{col}"),
        )
    additive_cols = (
        ["population", "_weighted_access", "_access_sum", "_access_n"]
        + count_cols
        + [f"_weighted_{c}" for c in rate_cols]
        + [f"_weight_{c}" for c in rate_cols]
    )
    resampled = h3_ops.resample(
        df.drop(rate_cols), target_resolution=target_resolution, columns=additive_cols, method="sum"
    )
    resampled = resampled.with_columns(
        pl.when(pl.col("population") > 0)
        .then(pl.col("_weighted_access") / pl.col("population"))
        .when(pl.col("_access_n") > 0)
        .then(pl.col("_access_sum") / pl.col("_access_n"))
        .otherwise(0.0)
        .alias("level_of_service")
    ).drop("_weighted_access", "_access_sum", "_access_n")
    for col in rate_cols:
        resampled = resampled.with_columns(
            pl.when(pl.col(f"_weight_{col}") > 0)
            .then(pl.col(f"_weighted_{col}") / pl.col(f"_weight_{col}"))
            .otherwise(None)
            .alias(col)
        ).drop(f"_weighted_{col}", f"_weight_{col}")
    resampled = _clamp_density_to_children_bounds(df, resampled, ["population"] + count_cols)
    return resampled


def _clamp_density_to_children_bounds(
    fine: pl.DataFrame, coarse: pl.DataFrame, count_cols: list[str]
) -> pl.DataFrame:
    """Clamp each coarse cell's count columns so their density stays within their children's density range.

    2026-09-05, explicit user spec (live report: Boston's H3 res-5 tiles
    showing anomalously high population/density right at the AOI
    boundary): "when resampling count values... measure the max and min
    density values... from all the raster or geometries that are parent or
    influenced the value of each cell. If the cell density is away from
    the max or min density set the nearest bound." A coarse (parent) H3
    cell's count is a sum of its fine (child) cells' counts -- physically,
    that sum's own density (`count / parent_area`) should never fall
    outside the range of densities its own children actually had; if it
    does, something about the resampling (e.g. only a fragment of the
    parent's true area actually has data, so `count / parent_area`
    underestimates OR the parent cell straddles a data-availability edge)
    has distorted the figure, and this clamps it back into the physically
    plausible range implied by the real children.

    Args:
        fine: The pre-resample (child-resolution) grid, with `h3_cell` and
            every column in `count_cols`.
        coarse: `h3_ops.resample`'s already-summed output, with `h3_cell`
            and every column in `count_cols`.
        count_cols: Additive/count-like column names to clamp (density has
            no meaning for `level_of_service` or a rate column, so those
            are left untouched).

    Returns:
        `coarse` with each `count_cols` entry clamped in place.
    """
    import h3ronpy

    if coarse.is_empty() or not count_cols:
        return coarse

    fine_cells = fine["h3_cell"].to_list()
    coarse_resolution = h3.get_resolution(coarse["h3_cell"][0])
    fine_parent = pl.Series("_parent", [h3.cell_to_parent(c, coarse_resolution) for c in fine_cells])
    fine_area = np.asarray(h3ronpy.cells_area_m2(h3ronpy.cells_parse(fine_cells)))
    # 2026-09-06 bug fix: these H3 grids only ever contain occupied/populated
    # cells, never a full tessellation -- a coarse cell's real children
    # (`fine_parent`) typically cover only a fraction of its full geometric
    # area (e.g. one res-7 cell here can have just 3 of its ~2401 possible
    # res-11 children present). Comparing `count / full_geometric_area`
    # against density bounds computed from each CHILD's own (much smaller,
    # undiluted) area systematically floors the coarse density up to `dmin`
    # for every sparse cell -- inflating population 2x+ across whole cities
    # (confirmed on Concepcion: 1,071,925 -> 2,307,067, dragging the
    # population-weighted median level_of_service from 0.95 down to 0.0).
    # Using the SUM of the real children's own areas as the coarse area puts
    # both sides of the comparison on the same basis the bounds were
    # actually computed on.
    effective_area = (
        pl.DataFrame({"_parent": fine_parent, "_area": fine_area})
        .group_by("_parent")
        .agg(pl.col("_area").sum().alias("_effective_area"))
    )
    coarse_area_map = dict(zip(effective_area["_parent"].to_list(), effective_area["_effective_area"].to_list()))
    full_area = np.asarray(h3ronpy.cells_area_m2(h3ronpy.cells_parse(coarse["h3_cell"].to_list())))
    full_area_map = dict(zip(coarse["h3_cell"].to_list(), full_area))

    for col in count_cols:
        if col not in fine.columns or col not in coarse.columns:
            continue
        fine_density = fine[col].cast(pl.Float64).to_numpy() / fine_area
        # the parent's density over the children that HAVE a value (2026-11-12 fix: the area of children without one -- e.g. cells no census
        # polygon covers -- counted in the denominator but not in the bounds, so the parent looked too sparse and was pushed UP to the lowest
        # child density, adding people that exist nowhere; with this area the clamp never moves an exact sum and totals are kept)
        bounds = (
            pl.DataFrame({"_parent": fine_parent, "_density": fine_density, "_area": fine_area})
            .filter(pl.col("_density").is_finite())
            .group_by("_parent")
            .agg(pl.col("_density").min().alias("_dmin"), pl.col("_density").max().alias("_dmax"), pl.col("_area").sum().alias("_varea"))
        )
        joined = coarse.select(["h3_cell", col]).join(bounds, left_on="h3_cell", right_on="_parent", how="left")
        cells = joined["h3_cell"].to_list()
        varea = joined["_varea"].cast(pl.Float64).to_numpy()
        c_area = np.array([coarse_area_map.get(c, full_area_map[c]) for c in cells], dtype=float)
        c_area = np.where(np.isfinite(varea) & (varea > 0), varea, c_area)
        cur_density = joined[col].cast(pl.Float64).to_numpy() / c_area
        dmin = joined["_dmin"].cast(pl.Float64).to_numpy()
        dmax = joined["_dmax"].cast(pl.Float64).to_numpy()
        has_bounds = np.isfinite(dmin) & np.isfinite(dmax)
        clamped_density = np.where(
            has_bounds, np.clip(cur_density, dmin, dmax), cur_density
        )
        clamped_value = clamped_density * c_area
        coarse = coarse.with_columns(pl.Series(col, clamped_value))
    return coarse


def _resample_h3_chunked(
    h3_grid, target_resolution: int, sum_cols: list[str], chunk_h3_resolution: int
) -> pl.DataFrame:
    """Memory-bounded `_resample_h3`, chunked by H3 res-N ancestor cell.

    `_resample_h3` is a pure group-by aggregation (sum/population-weighted
    mean per target cell) with no cross-row global statistic -- unlike the
    isochrone step, there is no "buffer zone" ambiguity to resolve here: an
    H3 cell's ancestor at any coarser resolution is a single, well-defined
    cell (H3's hierarchy is strictly nested), so partitioning `h3_grid`'s
    rows by their `chunk_h3_resolution`-ancestor before resampling and
    concatenating the per-chunk results afterward is *exact*, not an
    approximation -- every source row is aggregated by exactly one chunk, no
    row is ever split or double-counted, as long as `chunk_h3_resolution` is
    coarser than or equal to `target_resolution` (true for every resolution
    this study resamples to: 5/7/9/11 vs. a chunk resolution of 4).

    Args:
        h3_grid: See `_resample_h3`.
        target_resolution: See `_resample_h3`. Must be a resolution finer
            than (numerically >=) `chunk_h3_resolution`, or a target cell
            could itself span more than one chunk (not the case for any of
            this study's resolutions vs. the default chunk resolution 4).
        sum_cols: See `_resample_h3`.
        chunk_h3_resolution: H3 resolution to partition the input by
            (`StudyParams.isochrone_chunk_h3_resolution`).

    Returns:
        Same shape as `_resample_h3`.
    """
    if chunk_h3_resolution > target_resolution:
        raise ValueError(
            f"_resample_h3_chunked: chunk_h3_resolution ({chunk_h3_resolution}) must be "
            f"<= target_resolution ({target_resolution}) -- a target cell must nest "
            "inside exactly one chunk cell, or resampling would need cross-chunk merging."
        )
    cells = h3_grid["h3_cell"].to_list()
    parents = pl.Series("_chunk_parent", [h3.cell_to_parent(c, chunk_h3_resolution) for c in cells])
    tagged = h3_grid.with_columns(parents) if isinstance(h3_grid, pl.DataFrame) else pl.from_pandas(h3_grid).with_columns(parents)

    chunks = []
    for parent in tagged["_chunk_parent"].unique().to_list():
        sub = tagged.filter(pl.col("_chunk_parent") == parent).drop("_chunk_parent")
        resampled = _resample_h3(sub, target_resolution=target_resolution, sum_cols=sum_cols)
        if resampled.height:
            chunks.append(resampled)
    if not chunks:
        return _resample_h3(h3_grid.head(0) if isinstance(h3_grid, pl.DataFrame) else pl.from_pandas(h3_grid.head(0)), target_resolution, sum_cols)
    return pl.concat(chunks, how="vertical")


# "All count columns" offered by the cross-city stats-overview panel
# (`combined_map.html`'s "Compare column" selector). Deliberately a small,
# curated allowlist rather than every numeric column in `h3_grid` -- these
# are the columns that exist (or are meaningfully absent) uniformly across
# the whole city roster, matching the global pyCensus schema's canonical
# vocabulary (`pyCensus/src/pycensus/global_schema.json`) where a census
# equivalent exists:
#   - "population": present for every city (built from a general population
#     source, not census-dependent) -- the required default column.
#   - "lodes_wac_jobs_total" (displayed as "jobs"): USA-only (LODES WAC),
#     present wherever `_add_pop_jobs_columns` found a jobs column.
#   - "pop_jobs_total" / "pop_jobs_density": the population+jobs combined
#     column already computed by `_add_pop_jobs_columns`/
#     `_add_pop_jobs_columns_polars` (sum of population + jobs, absolute
#     and per-km2) -- a no-op/absent when no jobs column was found (e.g.
#     Mexico/Spain/Germany/Israel cities before their pipeline wiring is
#     done, or any city with no LODES jobs data).
# A city missing a given column from its `h3_grid` simply gets no entry for
# that key in `column_stats` -- `combined_map.py`'s selector reads this and
# excludes the city from that column's ranking, per the graceful-exclusion
# rule the stats-overview panel implements.
STATS_OVERVIEW_COLUMNS: tuple[tuple[str, str], ...] = (
    ("population", "Population"),
    ("lodes_wac_jobs_total", "Jobs"),
    ("pop_jobs_total", "Population + jobs"),
    ("pop_jobs_density", "Population + jobs density"),
)


# --------------------------------------------------------------------------
# Cross-city relative columns for ANOVA/regression (`combined_map.py`'s
# "Stats overview" -> ANOVA/Regression sub-tabs).
#
# The user asked for the ANOVA/regression comparison columns to be ".share"
# or ".density" of population / population+jobs / jobs, mirroring the
# `pycensus.accessors` `ColumnView.share`/`ColumnView.density` derivations
# built in the pyCensus schema redesign (`pyCensus/src/pycensus/accessors.py`).
#
# Those accessors are defined for WITHIN-geography use on schema-registered,
# source-prefixed columns (e.g. `gdf.lodes["jobs_total"].share` divides by
# that row's registered `relative_to` total column, typically a containing
# geography's own total -- "this tract's population as a share of its
# containing city"). `h3_grid` here does not go through a registered
# pycensus source accessor at all: `population`/`lodes_wac_jobs_total`/
# `pop_jobs_total` are plain, unprefixed columns assembled by this
# pipeline's own `_add_pop_jobs_columns` (see above) from whichever
# per-city source produced them (LODES jobs, ACS/DHC population, or a
# non-US city's own population source) -- there is no single
# `ColumnSchema`/`NewColumnSchema` shared across all cities in the roster,
# and no cross-city "containing geography" for `.share` to divide by (there
# is no supra-city total population registered anywhere).
#
# DESIGN DECISION (documented per the task's own instruction to flag
# ambiguity rather than silently pick something arbitrary): at the
# cross-city stats-overview level, "share"/"density" are reinterpreted as
# the PER-CELL, WITHIN-CITY relative value, population-weighted-averaged up
# to one number per city -- i.e. exactly the same computation the
# `.share`/`.density` accessors would produce if this grid *were* a
# registered pycensus source with `relative_to` pointing at the grid's own
# city-wide total:
#   - `<col>.density` = value / cell area (km2) -- identical formula to
#     `ColumnView.density` (value / geometry area), using this file's own
#     `_area_km2` helper (already used for `pop_jobs_density`) rather than
#     `geohierarchy.utils.area` directly, since `_area_km2` already handles
#     the `area_m2`-column-vs-h3-cell-id fallback this pipeline needs and
#     avoids a redundant CRS reprojection at this call site.
#   - `<col>.share` = value / (grid[col].sum() for this scope) -- i.e. "this
#     cell's share of the CITY'S OWN metro-or-core total for that column",
#     the direct within-city analog of `ColumnView.share`'s
#     "value / relative_to total" where the `relative_to` total is the
#     grid's own city-wide sum (the closest thing to a "containing
#     geography total" that exists at this level).
# This is genuinely a re-derivation of the accessor's formula, not a call
# into `pycensus.accessors` itself -- calling the real accessor would
# require registering a `ColumnSchema`/`NewColumnSchema` for this ad hoc,
# cross-city-roster column set, which does not exist and would be
# artificial to invent. If the user intended a literal supra-city
# "share of all cities combined" instead, that would need a second pass
# over every city's summary (not just one grid) -- flagged here for the
# user to weigh in on if this interpretation isn't what they had in mind.
RELATIVE_STATS_BASE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("population", "Population"),
    ("lodes_wac_jobs_total", "Jobs"),
    ("pop_jobs_total", "Population + jobs"),
)


def _relative_column_stats_for_grid(grid, population: np.ndarray) -> dict[str, float]:
    """Population-weighted mean of `.share`/`.density` for each `RELATIVE_STATS_BASE_COLUMNS` entry.

    See the design-decision comment above `RELATIVE_STATS_BASE_COLUMNS` for
    what "share"/"density" mean at this cross-city level. Keys are
    `"<col>.share"` / `"<col>.density"`, merged into the same
    `column_stats[scope]` dict `_column_stats_for_grid` already populates --
    a city/scope missing the base column (or with an all-zero/NaN total for
    `.share`, or ungeometry-able rows for `.density`) simply gets no entry
    for that key, same graceful-exclusion contract as the existing columns.
    """
    out: dict[str, float] = {}
    if grid is None:
        return out
    area_km2 = _area_km2(grid)
    for col, _label in RELATIVE_STATS_BASE_COLUMNS:
        if col not in grid.columns:
            continue
        values = pd.to_numeric(grid[col], errors="coerce").to_numpy(dtype=float)
        if values.size == 0:
            continue

        def _weighted_mean(per_cell: np.ndarray, key: str) -> None:
            if per_cell.size != population.size:
                return
            mask = np.isfinite(per_cell) & np.isfinite(population)
            weight_sum = float(np.nansum(population[mask])) if mask.any() else 0.0
            if weight_sum > 0:
                mean = float(np.average(per_cell[mask], weights=population[mask]))
            elif np.any(mask):
                mean = float(np.nanmean(per_cell[mask]))
            else:
                return
            if np.isnan(mean):
                return
            out[key] = mean

        # .density -- value / cell area (km2), same formula as
        # `ColumnView.density` (value / geometry area).
        if area_km2 is not None and area_km2.size == values.size:
            with np.errstate(divide="ignore", invalid="ignore"):
                density = np.where(area_km2 > 0, values / area_km2, np.nan)
            _weighted_mean(density, f"{col}.density")

        # .share -- value / this grid's own total for the column, the
        # within-city analog of `ColumnView.share`'s `relative_to` division.
        total = float(np.nansum(values))
        if total > 0:
            share = values / total
            _weighted_mean(share, f"{col}.share")
    return out


def _column_stats_for_grid(grid, population: np.ndarray) -> dict[str, float]:
    """Population-weighted mean of every `STATS_OVERVIEW_COLUMNS` column actually present in `grid`.

    Population-weighted mean (not median) is used here -- unlike the
    existing `metro_median_access`/`core_median_access` fields, which are
    specifically the median *level of service* -- because the stats-overview
    panel is meant to compare cities by an "average score" of an arbitrary
    count column (the user's own phrasing), and a population-weighted mean
    is the standard way to roll a per-cell count/density column up to a
    single city-level number without over/under-weighting sparsely
    populated cells. Falls back to a plain (unweighted) mean when the
    weighted mean can't be computed (e.g. population is all-zero/NaN) so a
    real city with a real column doesn't silently disappear from the
    overview just because its population column happens to sum to zero.
    """
    out: dict[str, float] = {}
    for col, _label in STATS_OVERVIEW_COLUMNS:
        if col not in grid.columns:
            continue
        values = pd.to_numeric(grid[col], errors="coerce").to_numpy(dtype=float)
        if values.size == 0:
            continue
        weight_sum = np.nansum(population) if population.size == values.size else 0.0
        if weight_sum and weight_sum > 0:
            mask = ~np.isnan(values) & ~np.isnan(population)
            if mask.any():
                mean = float(np.average(values[mask], weights=population[mask]))
            else:
                continue
        else:
            mean = float(np.nanmean(values)) if np.any(~np.isnan(values)) else None
            if mean is None:
                continue
        if np.isnan(mean):
            continue
        out[col] = mean
    return out


def _median_access_by_weight_for_grid(grid, access: np.ndarray) -> dict[str, float]:
    """Weighted-median `level_of_service`, keyed by each `WEIGHT_COLUMN_CANDIDATES` column actually present on `grid`.

    Mirrors `weighted_median(access, population)` (the existing
    `metro_median_access`/`core_median_access` computation) but repeated once
    per candidate weight column instead of hardcoding `population` -- powers
    the combined multi-city map's rank-panel weight-column selector. Most
    cities only carry a handful of the ~29 candidates on their own
    `h3_grid`; a city missing a given weight column simply gets no entry for
    that key -- the client-side rank table excludes it from that column's
    ranking rather than defaulting to 0/1, same graceful-exclusion contract
    as `_column_stats_for_grid`.
    """
    out: dict[str, float] = {}
    if grid is None:
        return out
    for col, _label in WEIGHT_COLUMN_CANDIDATES:
        if col not in grid.columns:
            continue
        weights = pd.to_numeric(grid[col], errors="coerce").to_numpy(dtype=float)
        if weights.size != access.size:
            continue
        # 2026-09-23, explicit user request (item 4): every cross-city
        # ranking aggregate is now a weighted MEAN, not a median -- the
        # JSON key names (`*_median_access*`) are left unchanged (renaming
        # them means touching every `combined_map.py`/client-JS reader of
        # this field, out of scope for this pass), but the *value* under
        # them is a weighted mean as of this date. See `weighted_mean` in
        # `studytools.stats`.
        mean = weighted_mean(access, weights)
        if not np.isnan(mean):
            out[col] = mean
    return out


def write_city_summary(
    city_dir: Path,
    display_name: str,
    *,
    metro_access: np.ndarray,
    metro_population: np.ndarray,
    core_access: np.ndarray,
    core_population: np.ndarray,
    metro_grid=None,
    core_grid=None,
) -> dict:
    """Write `results/summary.json`: this city's population-weighted median level of service.

    Cheap (two `weighted_median` calls) and independent of the rest of the
    map-building step, so it's computed right after the metro/core parquet
    grids are finalized. Consumed by the combined multi-city map page
    (`city_science_network/combined_map.html`) to rank all processed cities without
    opening their (much heavier) `map.html`/parquet outputs.

    Shape written to `results/summary.json`::

        {
          "display_name": "Boston",
          "metro_median_access": 0.42,
          "core_median_access": 0.61,
          "column_stats": {
            "metro": {"population": 1234.5, "pop_jobs_total": 2200.1,
                      "population.share": 0.0021, "population.density": 812.3},
            "core": {"population": 1500.2}
          }
        }

    `metro_median_access`/`core_median_access` are `null` when the score
    couldn't be computed (e.g. an empty grid) -- readers should skip the
    city for ranking purposes in that case, not treat it as 0.

    `metro_grid`/`core_grid` (optional, the same `h3_grid`/`h3_grid[core_mask]`
    GeoDataFrames the caller already has on hand) drive `column_stats`: a
    population-weighted mean per `STATS_OVERVIEW_COLUMNS` entry that's
    actually present on that grid, plus `.share`/`.density` variants of
    `RELATIVE_STATS_BASE_COLUMNS` (population / jobs / population+jobs) for
    the cross-city ANOVA/regression sub-tabs -- see the design-decision
    comment above `RELATIVE_STATS_BASE_COLUMNS` for what "share"/"density"
    mean at this cross-city level. Omitted (`None`, the default) for
    backward compatibility with any older call site -- `column_stats` is
    then just `{"metro": {}, "core": {}}`, which the stats-overview panel
    treats identically to "this city has no data for any column".
    """
    # 2026-09-23, explicit user request (item 4): population-weighted MEAN,
    # not median (see `_median_access_by_weight_for_grid`'s matching note --
    # the `metro_median`/`metro_median_access` names are kept as-is, the
    # values are means now).
    metro_median = weighted_mean(metro_access, metro_population)
    core_median = weighted_mean(core_access, core_population)
    # `aoi_center`: cheap bounding-box center (not a real weighted
    # centroid -- `metro_grid` can be millions of rows, and this only
    # needs to be good enough to place one marker pin on the combined
    # map's low-zoom overview, see `code.combined_map`) of the metro h3
    # grid's own geometry, in lon/lat. `None` when `metro_grid` wasn't
    # passed (older call sites) or is empty -- readers should just omit
    # that city's marker rather than plotting `[0, 0]`.
    # 2026-09-02 (live user report, combined map: markers looking "off
    # position", not aligned with the real city): used to be the metro
    # AOI's plain bounding-box midpoint -- fine for a roughly-square,
    # evenly-populated AOI, but wrong-looking for a lopsided or partially-
    # empty one (e.g. Beersheba's own AOI is only ~51% real coverage, the
    # rest open Negev desert on one side -- see `_add_worldpop_gapfill`'s
    # docstring -- so its bbox midpoint sits measurably off from where the
    # city and its people actually are). A population-weighted centroid of
    # the real h3 cells is what a person actually means by "where is this
    # city" -- the center of mass of where people live, not of the AOI
    # rectangle's corners.
    aoi_center = None
    if metro_grid is not None and len(metro_grid) > 0:
        try:
            metro_4326 = metro_grid.to_crs(4326)
            pts = metro_4326.geometry.centroid
            weights = metro_4326["population"].to_numpy(dtype=float)
            weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
            if weights.sum() > 0:
                cx = float(np.average(pts.x.to_numpy(), weights=weights))
                cy = float(np.average(pts.y.to_numpy(), weights=weights))
            else:
                minx, miny, maxx, maxy = metro_4326.total_bounds
                cx, cy = float((minx + maxx) / 2), float((miny + maxy) / 2)
            aoi_center = [cx, cy]
        except Exception:
            aoi_center = None
    summary = {
        "display_name": display_name,
        "aoi_center": aoi_center,
        "metro_median_access": None if np.isnan(metro_median) else metro_median,
        "core_median_access": None if np.isnan(core_median) else core_median,
        # Weighted median access, keyed by weight column -- powers the
        # combined map's selectable-weight rank table (default `population`,
        # which duplicates `metro_median_access`/`core_median_access` above
        # under the `"population"` key so the client doesn't need two code
        # paths for the default vs. a chosen weight column).
        "metro_median_access_by_weight": _median_access_by_weight_for_grid(metro_grid, metro_access),
        "core_median_access_by_weight": _median_access_by_weight_for_grid(core_grid, core_access),
        "column_stats": {
            "metro": {
                **(_column_stats_for_grid(metro_grid, metro_population) if metro_grid is not None else {}),
                **(_relative_column_stats_for_grid(metro_grid, metro_population) if metro_grid is not None else {}),
            },
            "core": {
                **(_column_stats_for_grid(core_grid, core_population) if core_grid is not None else {}),
                **(_relative_column_stats_for_grid(core_grid, core_population) if core_grid is not None else {}),
            },
        },
    }
    out_path = city_dir / "results" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def _run_regressions_and_anovas(
    h3_grid: gpd.GeoDataFrame, uses_census: bool, out_dir: Path, city_label: str, area_label: str
) -> dict:
    """Run every regression + population-weighted median-split ANOVA, saving figures for one area scope.

    Args:
        h3_grid: The h3 grid already filtered to the area being analyzed
            (metro or city-core).
        uses_census: Whether to also run the census-derived variables.
        out_dir: City directory (figures saved under `out_dir/figures`).
        city_label: Display name, used in plot titles.
        area_label: `"metro"` or `"core"` -- appended to every filename and
            plot title so both scopes' figures coexist.

    Returns:
        A small summary dict (used for the equity flag + cross-city
        comparison figures).
    """
    figures_dir = out_dir / "figures"
    score = h3_grid["level_of_service"].to_numpy()
    density = h3_grid["pop_density"].to_numpy()
    population = h3_grid["population"].to_numpy()
    area_title = f"{city_label} ({area_label})"

    summary: dict = {}
    anovas: dict[str, dict] = {}

    density_fit = linreg(density, score, weights=population)
    summary["r2_density"] = density_fit.r2
    regression_scatter(
        density, score, density_fit, "population density (people/km²)", "level_of_service",
        f"{area_title}: LOS vs population density", str(figures_dir / f"regression_density_{area_label}.jpg"),
    )

    anova_density = weighted_median_split_anova(density, score, population)
    anovas["pop density"] = anova_density
    summary["anova_density"] = anova_density

    if uses_census:
        for var_name, candidates in EQUITY_VAR_CANDIDATES.items():
            col = _first_present_column(h3_grid, candidates)
            if col is None:
                continue
            share = h3_grid[col].to_numpy(dtype=float)
            anova = weighted_median_split_anova(share, score, population)
            anovas[var_name] = anova
            summary[f"anova_{var_name}"] = anova

        income_col = _first_present_column(h3_grid, INCOME_VAR_CANDIDATES)
        if income_col is not None:
            income = h3_grid[income_col].to_numpy(dtype=float)
            anova_income = weighted_median_split_anova(income, score, population)
            anovas["income"] = anova_income
            summary["anova_income"] = anova_income

        transit_col = _first_present_column(h3_grid, TRANSIT_VAR_CANDIDATES)
        if transit_col is not None:
            transit_share = h3_grid[transit_col].to_numpy(dtype=float)
            transit_fit = linreg(transit_share, score, weights=population)
            summary["r2_transit_share"] = transit_fit.r2
            regression_scatter(
                transit_share, score, transit_fit, "% commuting by transit", "level_of_service",
                f"{area_title}: LOS vs transit commute share",
                str(figures_dir / f"regression_transit_share_{area_label}.jpg"),
            )
            anova_transit = weighted_median_split_anova(transit_share, score, population)
            anovas["transit commute share"] = anova_transit
            summary["anova_transit_share"] = anova_transit

    anova_diff_bar(
        anovas, "level_of_service", f"{area_title}: population-weighted LOS difference by median-split group",
        str(figures_dir / f"anova_summary_{area_label}.jpg"),
    )

    overlay_col = (
        _first_present_column(h3_grid, TRANSIT_VAR_CANDIDATES) if uses_census else None
    )
    dist = access_distribution(
        score, population, h3_grid[overlay_col].to_numpy(dtype=float) if overlay_col else None
    )
    access_distribution_bar(
        dist, overlay_col, f"{area_title}: level_of_service distribution",
        str(figures_dir / f"distribution_{area_label}.jpg"),
    )

    return summary


import contextlib


@contextlib.contextmanager
def _tile_worker_cap(tile_workers: Optional[int]):
    """Temporarily cap tile-build parallelism for the duration of a vector-tile build.

    Historically this only patched `os.cpu_count()`, on the assumption that
    `geohierarchy.maps.folium.render` read it to size a `ProcessPoolExecutor`.
    That's no longer true: `HierarchyMap.build()` now generates tiles via
    `write_level_tiles` -> `freestiler.freestile_file`, a Rust extension that
    never calls back into Python's `os.cpu_count()` -- so the patch was a
    silent no-op against the actual tile-writing path. This was confirmed
    live: Boston (a modestly sized city, ~1.6M H3 cells) was OOM-killed by
    the kernel at `tile_workers=1` in almost exactly the same time as at
    `tile_workers=2`, which is only possible if the cap was never reaching
    the process actually doing the work.

    freestiler's Rust engine uses `rayon` for its own internal parallelism,
    which sizes its global thread pool from the `RAYON_NUM_THREADS`
    environment variable (read once, at the pool's first use) rather than
    any Python-visible CPU count -- so that's the env var that actually
    throttles freestiler's worker threads (and therefore its peak memory,
    each thread holding its own working buffers). The `os.cpu_count()` patch
    is kept too, harmlessly, in case any remaining code path still reads it.
    Shared by `run_city_study`, `rebuild_map_only`, and `refresh_census_only`
    so the cap applies identically on every entry point that builds tiles,
    not just the full pipeline.
    """
    if tile_workers is None:
        yield
        return
    _n = max(1, int(tile_workers))
    _orig = os.cpu_count
    os.cpu_count = lambda _n=_n: _n
    _orig_rayon = os.environ.get("RAYON_NUM_THREADS")
    os.environ["RAYON_NUM_THREADS"] = str(_n)
    print(f"[pipeline] tile pool capped at {_n} worker(s) (RAYON_NUM_THREADS={_n})")
    try:
        yield
    finally:
        os.cpu_count = _orig
        if _orig_rayon is None:
            os.environ.pop("RAYON_NUM_THREADS", None)
        else:
            os.environ["RAYON_NUM_THREADS"] = _orig_rayon


def _map_inputs_checkpoint_dir(city_dir: Path) -> Path:
    return Path(city_dir) / "results" / "map_inputs_checkpoint"


def _write_map_inputs_checkpoint(
    city_dir: Path,
    h3_by_resolution: dict[int, gpd.GeoDataFrame],
    census_by_level: Optional[dict[str, gpd.GeoDataFrame]],
    routes_lines_gdf: gpd.GeoDataFrame,
    stats_h3_resolution: int,
    stats_core_mask,
) -> None:
    """Cache every expensive `rebuild_map_only` input so a pure style/HTML
    change can skip straight to `build_city_map` (see `style_only=True`).

    2026-09-22, explicit user request ("I dont want to recompute tiles if I
    only want some minor color changes or map html style changes... right
    now it seems it is not even with the map only flag"): `--map-only`
    already skips the GTFS/network/isochrone/WorldPop/census-JOIN stages
    (real, expensive, network-bound work), but it still re-does three
    substantial CPU-bound stages from the cached `h3_grid.parquet` every
    single time -- H3 resampling to every map resolution
    (`build_h3_by_resolution`), population-weighted score aggregation onto
    every real census polygon level (`_census_geometries_with_score`), and
    rebuilding every GTFS route's line geometry (`build_route_lines`) --
    none of which a style/color/HTML change needs at all, since none of
    them depend on anything `build_city_map` doesn't already receive as a
    plain argument. Caching their OUTPUT (not the H3 grid Parquet itself,
    which is unchanged) as geoparquet lets a later `style_only=True` call
    just read these files back (pure I/O, no resampling/aggregation/GTFS
    parsing) and jump straight to HTML/tile generation.

    Written by every real (non-style-only) `rebuild_map_only`/full-pipeline
    call, so the NEXT `--style-only` call always has a fresh checkpoint --
    if the underlying data changes (a real reprocess), the checkpoint gets
    overwritten in the same run that changed it, never silently stale.

    `stats_core_mask` (boolean array aligned with `h3_by_resolution[stats_h3_resolution]`)
    is folded in as a `_stats_core` column on that resolution's own file
    rather than a separate artifact -- one fewer thing to keep in sync.
    """
    out_dir = _map_inputs_checkpoint_dir(city_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):
        old.unlink()

    for res, gdf in h3_by_resolution.items():
        gdf = gdf.copy()
        if res == stats_h3_resolution:
            gdf["_stats_core"] = np.asarray(stats_core_mask, dtype=bool)
        gdf.to_parquet(out_dir / f"h3_res{res}.parquet")

    if census_by_level:
        for level, gdf in census_by_level.items():
            gdf.to_parquet(out_dir / f"census_{level}.parquet")

    routes_lines_gdf.to_parquet(out_dir / "routes_lines.parquet")
    (out_dir / "stats_h3_resolution.txt").write_text(str(stats_h3_resolution))


def _read_map_inputs_checkpoint(
    city_dir: Path,
) -> Optional[tuple[dict[int, gpd.GeoDataFrame], Optional[dict[str, gpd.GeoDataFrame]], gpd.GeoDataFrame, gpd.GeoDataFrame, "np.ndarray"]]:
    """Load `_write_map_inputs_checkpoint`'s output, or `None` if it doesn't exist yet.

    Returns `(h3_by_resolution, census_by_level, stats_grid, stats_core_mask)`
    -- `stats_grid`/`stats_core_mask` are just `h3_by_resolution[stats_h3_resolution]`
    and its `_stats_core` column, split back out (that column is dropped
    from the returned `h3_by_resolution` entry so it doesn't leak into the
    map as a fake extra field).
    """
    out_dir = _map_inputs_checkpoint_dir(city_dir)
    res_marker = out_dir / "stats_h3_resolution.txt"
    if not res_marker.is_file():
        return None

    stats_h3_resolution = int(res_marker.read_text().strip())
    h3_by_resolution: dict[int, gpd.GeoDataFrame] = {}
    for p in sorted(out_dir.glob("h3_res*.parquet")):
        res = int(p.stem.removeprefix("h3_res"))
        h3_by_resolution[res] = gpd.read_parquet(p)

    if stats_h3_resolution not in h3_by_resolution:
        return None

    stats_grid = h3_by_resolution[stats_h3_resolution]
    if "_stats_core" not in stats_grid.columns:
        return None
    stats_core_mask = stats_grid["_stats_core"].to_numpy()
    stats_grid = stats_grid.drop(columns=["_stats_core"])
    h3_by_resolution[stats_h3_resolution] = stats_grid

    census_by_level: Optional[dict[str, gpd.GeoDataFrame]] = None
    census_paths = sorted(out_dir.glob("census_*.parquet"))
    if census_paths:
        census_by_level = {p.stem.removeprefix("census_"): gpd.read_parquet(p) for p in census_paths}

    routes_path = out_dir / "routes_lines.parquet"
    if not routes_path.is_file():
        return None
    routes_lines_gdf = gpd.read_parquet(routes_path)

    return h3_by_resolution, census_by_level, stats_grid, routes_lines_gdf, stats_core_mask


def _map_routes_gdf_for_display(routes_lines_gdf):
    """Pick the route rows the MAP's own line layer + route/mode checkboxes should use.

    Normally this is just the `has_real_shape` subset (2026-09-01
    user-requested default: don't draw fabricated straight-line-between-
    stops geometry on the map). But Andorra's GTFS feed
    (`andorra/gtfs/ena_manual_reconstruction`, a manually reconstructed
    feed with no `shapes.txt` at all) has ZERO routes with real shape
    geometry -- filtering to `has_real_shape` there doesn't just hide a few
    fabricated lines, it empties the route layer completely, which in turn
    made `transitlos.map`'s `has_routes` check (derived from "is the passed
    routes_gdf empty") hide the entire route/mode (bus/tram/rail) checkbox
    UI, not merely the map lines -- a real, user-visible bug (checkboxes
    "don't render at all" on Andorra, confirmed present on Boston, which
    has real shapes.txt and hits the normal case fine).

    The `has_real_shape` policy was never meant to cover "this whole feed
    has no shape data" -- that's a data-completeness gap, not a
    should-we-fabricate-a-line judgment call, and the fix that respects the
    original request without regressing this case is: fall back to the
    full (straight-line-geometry) route set only when filtering to real
    shapes would leave nothing to draw at all. Any city with at least one
    real-shape route keeps exactly the prior behavior byte-for-byte.
    """
    if "has_real_shape" not in routes_lines_gdf.columns:
        return routes_lines_gdf
    real_shape_gdf = routes_lines_gdf[routes_lines_gdf["has_real_shape"]]
    if real_shape_gdf.empty and not routes_lines_gdf.empty:
        return routes_lines_gdf
    return real_shape_gdf


def _export_downloads_single(city_dir, h3_by_resolution, census_by_level, stops, routes_lines_gdf, core_crs, area_unions) -> dict:
    """Downloads of a four-area city: ONE file set for the whole processing area in `results/downloads/`, every row flagged with the areas it belongs to
    (`area_un_metro`, `area_official_metro`, `area_municipality`, `area_un_core`), instead of one overlapping copy per area."""
    downloads_dir = city_dir / "results" / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    for old in ("metro", "official", "municipality", "core"):   # copies of the earlier per-area layout
        shutil.rmtree(city_dir / "results" / old / "downloads", ignore_errors=True)
    from concurrent.futures import ThreadPoolExecutor

    def _h3_file(item):
        res, res_gdf = item
        _add_area_flags(res_gdf, area_unions).drop(columns="area_m2", errors="ignore").to_parquet(downloads_dir / f"h3_res{res}.parquet")

    def _census_file(item):
        level, level_gdf = item
        cen = (level_gdf.geometry.centroid if level_gdf.crs == core_crs else level_gdf.to_crs(core_crs).geometry.centroid)
        cen4326 = gpd.GeoSeries(cen, crs=core_crs).to_crs(4326)
        flagged = level_gdf.copy()
        for k, g in area_unions.items():
            flagged[AREA_FLAG_COLUMNS[k]] = np.asarray(shapely.within(cen4326.to_numpy(), g), dtype=np.int8)
        flagged.to_parquet(downloads_dir / f"census_{level}.parquet")
        del cen, cen4326, flagged

    # The files are independent and the writes (pyarrow, GEOS) release the GIL: a few at a time, the biggest first (memory stays bounded by CS_EXPORT_WORKERS).
    jobs = [(_h3_file, it) for it in sorted(h3_by_resolution.items(), key=lambda kv: -len(kv[1]))] + [(_census_file, it) for it in (census_by_level or {}).items()]
    with ThreadPoolExecutor(max_workers=max(1, int(os.environ.get("CS_EXPORT_WORKERS", "3")))) as pool:
        for fut in [pool.submit(fn, it) for fn, it in jobs]:
            fut.result()
    gc.collect()
    manifest = {
        "scopes": ["all"],
        "h3_resolutions": sorted(h3_by_resolution.keys()),
        "census_levels": list(census_by_level.keys()) if census_by_level else [],
        "has_stops": stops is not None and len(stops) > 0,
        "has_routes": routes_lines_gdf is not None and len(routes_lines_gdf) > 0,
    }
    if manifest["has_stops"]:
        stops.to_parquet(downloads_dir / "stops.parquet")
    if manifest["has_routes"]:
        routes_lines_gdf.to_parquet(downloads_dir / "routes.parquet")
    (city_dir / "results" / "downloads_manifest.json").write_text(json.dumps(manifest))
    return manifest


def _export_downloads(
    city_dir: Path,
    h3_by_resolution: dict,
    census_by_level: Optional[dict],
    stops,
    routes_lines_gdf,
    core_union,
    core_crs,
    area_unions=None,
) -> dict:
    """Write the real standalone parquet files the map's "Download data" panel points at.

    Shared by `_finish_pipeline_stages` (full pipeline run) and
    `rebuild_map_only` (fast map-only re-entry point) so both write
    identically shaped output -- reuses data both callers already have in
    memory (`h3_by_resolution`/`census_by_level`/`stops`/`routes_lines_gdf`
    are all already computed for the map build itself), no recomputation.

    Per-resolution h3 grids come from `h3_by_resolution` (already resampled
    for the map); per-level census geometries from `census_by_level`
    (already scored for the map). Both get a metro/core split via
    `core_union` -- h3 rows via centroid-in-polygon on `h3_cell` (same
    convention the pipeline's own metro/core split uses elsewhere), census
    rows via centroid-in-polygon on their own geometry (reprojected to
    `core_crs` first if needed, since census geometries aren't guaranteed
    to already share the h3 grid's CRS). Stops/routes are geographic
    point/line layers with no "scope" of their own (a stop or route either
    is or isn't in the study's GTFS extent), so they're written once,
    outside the metro/core split.

    Returns the `downloads_manifest` dict also written to
    `results/downloads_manifest.json` and baked into `map.html` for the
    download panel's JS to read.
    """
    if area_unions:
        return _export_downloads_single(city_dir, h3_by_resolution, census_by_level, stops, routes_lines_gdf, core_crs, area_unions)
    scopes = ["metro", "official", "municipality", "core"] if area_unions else ["metro", "core"]
    downloads_manifest = {
        "scopes": scopes,
        "h3_resolutions": sorted(h3_by_resolution.keys()),
        "census_levels": list(census_by_level.keys()) if census_by_level else [],
        "has_stops": stops is not None and len(stops) > 0,
        "has_routes": routes_lines_gdf is not None and len(routes_lines_gdf) > 0,
    }
    for scope in scopes:
        (city_dir / "results" / scope / "downloads").mkdir(parents=True, exist_ok=True)
    unions = dict(area_unions) if area_unions else {"core": core_union}   # legacy: metro = everything
    for res, res_gdf in h3_by_resolution.items():
        cents = _h3_cell_centroids(res_gdf["h3_cell"])
        for scope in scopes:
            part = res_gdf if scope not in unions or (not area_unions and scope == "metro") else res_gdf[np.asarray(shapely.within(cents, unions[scope]), dtype=bool)]
            part.drop(columns="area_m2", errors="ignore").to_parquet(city_dir / "results" / scope / "downloads" / f"h3_res{res}.parquet")
            del part
    if census_by_level:
        for level, level_gdf in census_by_level.items():
            # Boston OOM fix (2026-08-26): reproject only what is needed and free it at once (see git history); the same now for every scope.
            cen = level_gdf.geometry.centroid if level_gdf.crs == core_crs else level_gdf.to_crs(core_crs).geometry.centroid
            for scope in scopes:
                if scope == "metro" and not area_unions:
                    level_gdf.to_parquet(city_dir / "results" / scope / "downloads" / f"census_{level}.parquet")
                    continue
                u = unions[scope]
                if area_unions and gpd.GeoSeries([u], crs=4326).crs != core_crs:
                    u = gpd.GeoSeries([u], crs=4326).to_crs(core_crs).iloc[0]
                level_gdf[cen.within(u)].to_parquet(city_dir / "results" / scope / "downloads" / f"census_{level}.parquet")
            del cen
            gc.collect()
    downloads_dir = city_dir / "results" / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    if downloads_manifest["has_stops"]:
        stops.to_parquet(downloads_dir / "stops.parquet")
    if downloads_manifest["has_routes"]:
        routes_lines_gdf.to_parquet(downloads_dir / "routes.parquet")
    # Also written as its own small JSON file (not just baked into
    # map.html's JS) so `combined_map.py` can read it per-city without
    # having to scrape `<script>` contents out of each city's own
    # map.html -- see that module's own download-panel wiring.
    (city_dir / "results" / "downloads_manifest.json").write_text(json.dumps(downloads_manifest))
    return downloads_manifest


GBA_POPULATION_COLUMN = "gba_builtSurfacePopulation"


def _gba_cache_root(census_dir: Path) -> Path:
    """The shared cache folder (`TransitLOSStudies/cache`): GBA tiles live in `<it>/gba`, shared by every study."""
    from .cache import resolve_cache_dir

    try:
        return Path(resolve_cache_dir())
    except Exception:  # noqa: BLE001
        return Path(census_dir).parent


def _gba_buildings(census_dir: Path, aoi_gdf):
    from pycensus.census.worldwide import gba

    return gba.fill_heights(gba.load_buildings(aoi_gdf, cache_dir=_gba_cache_root(census_dir)))


def _census_population_polygons(config, aoi_gdf, census_dir: Path):
    """(polygons, population column) of the country's finest census level with a usable population count over the AOI, or (None, None)
    (no census for the country): the census the built-surface population is calibrated to. Same loaders as the census-geometry layers
    (`_census_loader_and_levels_for_map`, finest level first); a level is used when at least 95% of its polygons carry a population.
    Israel: statistical areas + locality remainders (`_israel_census_loader`)."""
    if config.country == "ISR":   # statistical areas with the locality remainders: complete and finer than the localities
        g = _israel_census_loader(aoi_gdf, None, "statistical_area", census_dir)
        col = next((c for c in ("cbs_population", "population") if c in g.columns), None)
        if col:
            return g, col
    resolved = _census_loader_and_levels_for_map(config.country, getattr(config, "census_module", None)) if config.uses_census else None
    if resolved is None:
        return None, None
    loader, levels = resolved
    states = list(config.census_states) if config.census_states else None
    for level in levels:
        try:
            g = loader(aoi_gdf, states, level, census_dir)
        except Exception as exc:  # noqa: BLE001 -- a level a source does not publish: try the next one
            print(f"[pipeline:{config.key}] GBA calibration: census level {level!r} not loaded ({type(exc).__name__}: {str(exc)[:120]})")
            continue
        if g is None or g.empty:
            continue
        cands = [c for c in g.columns if isinstance(c, str) and not c.startswith("worldpop") and (c in ("population", "totalPopulation") or c.endswith("_population"))]
        cands.sort(key=lambda c: (c != "population", c != "totalPopulation", len(c)))
        for col in cands:
            v = pd.to_numeric(g[col], errors="coerce")
            if v.notna().mean() >= 0.95 and v.sum() > 0:
                print(f"[pipeline:{config.key}] GBA calibration: census level {level!r} ({len(g):,} polygons, {col} {v.sum():,.0f})")
                return g.reset_index(drop=True), col
    return None, None


def _write_rescale_factors(city_dir: Path) -> None:
    """`results/rescale_factors.json`: every factor applied to the cells' counts (`h3_population.RESCALE_FACTORS`), merged with the file's
    earlier entries (a re-entry run only redoes the later steps). Original value of a column = mapped value / the product of its factors."""
    import json as _json

    from .h3_population import RESCALE_FACTORS

    f = Path(city_dir) / "results" / "rescale_factors.json"
    old = []
    if f.is_file():
        try:
            old = _json.loads(f.read_text()).get("factors", [])
        except Exception:  # noqa: BLE001
            old = []
    keys = {(r["step"], r["source"], r["what"]) for r in RESCALE_FACTORS}
    rows = [r for r in old if (r["step"], r["source"], r["what"]) not in keys] + list(RESCALE_FACTORS)
    per_col: dict = {}
    for r in rows:
        for c in r.get("columns") or []:
            per_col[c] = per_col.get(c, 1.0) * r["factor"]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(_json.dumps({"note": "original = value / column_factor", "factors": rows, "column_factor": per_col}, indent=1))


def _gba_cell_population(config, city_dir: Path, aoi_gdf, census_dir: Path, cells, resolution: int, worldpop=None) -> np.ndarray:
    """Built-surface population of every native h3 cell in `cells` (aligned), the map's rule:

      * GBA built area of a building = footprint x height / 3 (`gba.fill_heights`);
      * in each census polygon of the finest usable level (`_census_population_polygons`): rate = census population / built area of the
        buildings whose centroid lies inside it (all of them, roads or not);
      * a cell takes rate(polygon holding its centre) x its own built area. A cell outside every polygon takes the built-area-weighted rate
        of the 5 nearest polygons; without any census, one rate for the whole area (WorldPop people / built area, the cells' own).
    Not normalised here: `_native_density_rule` scales the active cells' total to the census total (one factor)."""
    import json as _json

    import h3ronpy
    from pycensus.census.worldwide import gba
    from scipy.spatial import cKDTree

    cells = list(cells)
    b = _gba_buildings(census_dir, aoi_gdf)
    print(f"[pipeline:{config.key}] GBA: {len(b):,} buildings, built floor area {b['built_m2'].sum() / 1e6:,.2f} km2 "
          f"({b['height_filled'].mean():.1%} heights filled)")
    bc = gba.h3_metrics(b, resolution)["builtFloorArea"]
    B = pd.Series(cells).map(bc).astype(float).fillna(0.0).to_numpy()
    polys, col = _census_population_polygons(config, aoi_gdf, census_dir)
    info = {"cells": len(cells), "built_m2_cells": float(B.sum()), "built_m2_total": float(b["built_m2"].sum())}
    if polys is None or not len(polys):   # no census: one rate, the WorldPop people of these cells per built m2
        wp = np.nan_to_num(np.asarray(worldpop, dtype=float)) if worldpop is not None else np.zeros(len(cells))
        info["rate"] = float(wp.sum() / B.sum()) if B.sum() > 0 else None
        rate = np.full(len(cells), info["rate"] if info["rate"] is not None else np.nan)
    else:
        P = polys[[col, "geometry"]].to_crs(4326).reset_index(drop=True)
        C = pd.to_numeric(P[col], errors="coerce").fillna(0.0).to_numpy()
        pts = gpd.GeoDataFrame(geometry=b.geometry.centroid.to_crs(4326) if b.crs else b.geometry.centroid, crs=4326)
        j = gpd.sjoin(pts, P[["geometry"]], predicate="within")
        j = j[~j.index.duplicated()]
        sumB = np.bincount(j["index_right"].to_numpy(), weights=b.loc[j.index, "built_m2"].to_numpy(float), minlength=len(P))
        r = np.where(sumB > 0, C / np.where(sumB > 0, sumB, 1.0), np.nan)
        cen = gpd.GeoDataFrame(geometry=_h3_cell_centroids(cells), crs=4326)
        jc = gpd.sjoin(cen, P[["geometry"]], predicate="within")
        jc = jc[~jc.index.duplicated()]
        rate = np.full(len(cells), np.nan)
        rate[jc.index.to_numpy()] = r[jc["index_right"].to_numpy()]
        ok = np.isfinite(r)
        miss = np.isnan(rate)
        if miss.any() and ok.any():
            rp = P.geometry.representative_point()
            tree = cKDTree(np.column_stack([rp.x.to_numpy()[ok], rp.y.to_numpy()[ok]]))
            k = min(5, int(ok.sum()))
            xy = np.column_stack([np.array([g.x for g in cen.geometry[miss]]), np.array([g.y for g in cen.geometry[miss]])])
            _, nn = tree.query(xy, k=k)
            nn = nn.reshape(len(xy), k)
            rr, ww = r[ok], sumB[ok]
            rate[miss] = (rr[nn] * ww[nn]).sum(1) / np.maximum(ww[nn].sum(1), 1e-9)
        info.update(census_level_column=col, census_polygons=len(P), census_total=float(C.sum()),
                    cells_outside_census=int(miss.sum()), rate_median=float(np.nanmedian(r)) if ok.any() else None)
    est = np.nan_to_num(rate * B)
    info["estimate_cells"] = float(est.sum())
    out_dir = city_dir / "results" / "gba"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "built_surface_population.json").write_text(_json.dumps(info, indent=1, default=str))
    print(f"[pipeline:{config.key}] GBA built-surface population (before the census-total factor): " + ", ".join(f"{k} {v}" for k, v in info.items()))
    return est


def _add_gba_columns(config, census_dir: Path, aoi_gdf, h3_by_resolution: dict | None = None, census_by_level: dict | None = None,
                     h3_grid=None) -> None:
    """GBA building metrics (`gba_` + `pycensus.census.worldwide.gba.METRIC_COLUMNS`) computed directly on every h3 resolution and
    census level (max / median cannot be resampled), in place. Buildings by centroid."""
    if not getattr(config, "gba_buildings", False):
        return
    from pycensus.census.worldwide import gba

    b = _gba_buildings(census_dir, aoi_gdf)
    cols = [f"gba_{c}" for c in gba.METRIC_COLUMNS]
    for res, grid in (h3_by_resolution or {}).items():
        m = gba.h3_metrics(b, res).add_prefix("gba_")
        for c in cols:
            grid[c] = grid["h3_cell"].map(m[c]).astype(float).fillna(0.0 if c in ("gba_buildingCount", "gba_buildingFootprintArea", "gba_builtFloorArea") else np.nan).to_numpy()
    for level, gdf in (census_by_level or {}).items():
        if "GEOID" not in gdf.columns:
            continue
        m = gba.polygon_metrics(b, gdf, "GEOID").add_prefix("gba_")
        for c in cols:
            gdf[c] = gdf["GEOID"].map(m[c]).astype(float).to_numpy()
        if h3_grid is not None and GBA_POPULATION_COLUMN in h3_grid.columns:   # the built-surface population of the finest cells inside
            pts = gpd.GeoDataFrame({"v": h3_grid[GBA_POPULATION_COLUMN].to_numpy(dtype=float)},
                                   geometry=_h3_cell_centroids(h3_grid["h3_cell"]), crs=4326).to_crs(gdf.crs)
            j = gpd.sjoin(pts, gdf[["GEOID", "geometry"]], predicate="within")
            gdf[GBA_POPULATION_COLUMN] = gdf["GEOID"].map(j.groupby("GEOID")["v"].sum()).fillna(0.0).to_numpy()


def _map_layers(config, h3_by_resolution, census_by_level, access_gdf, development_gdf):
    """What the map draws: everything, or with `config.map_finest_only` only the finest H3 resolution, the finest census level and no
    street / development layers."""
    if not getattr(config, "map_finest_only", False):
        return h3_by_resolution, census_by_level, access_gdf, development_gdf
    finest = max(h3_by_resolution)
    levels = census_by_level or {}
    census = {k: levels[k] for k in [max(levels, key=lambda k: len(levels[k]))]} if levels else {}   # the finest = the most units
    edges = gpd.read_parquet(access_gdf) if isinstance(access_gdf, (str, Path)) else access_gdf
    return {finest: h3_by_resolution[finest]}, census, edges.iloc[:0], None


def _finish_pipeline_stages(
    city_dir: Path,
    config: CityConfig,
    params: StudyParams,
    census_dir: Path,
    aoi_gdf,
    h3_grid,
    stops,
    access_gdf,
    tile_workers: Optional[int] = 4,
    run_stats: bool = True,
    _lap=lambda label: None,
    build_map: bool = True,
) -> None:
    """Shared tail of the pipeline: resample -> core/metro split -> stats -> map (`build_map=False`: stop after the downloads).

    Factored out of `run_city_study` so the fast re-entry points
    (`refresh_census_only`, and `run_city_study` itself) share exactly one
    implementation of "what happens once the h3 grid has population + access
    + (optionally) census columns on it" -- avoids the two ever drifting.
    `access_gdf` may be a `Path` (lazy parquet, loaded here) or an already
    loaded GeoDataFrame, matching `run_city_study`'s `reuse_cached_los` path.
    """
    h3_grid = _sanitize_census_columns(h3_grid)   # derived shares can still be entirely empty (e.g. a missing denominator source)
    if getattr(params, "chunked_h3_output", None) is not None:
        # Opt-in chunked path (`StudyParams.chunked_h3_output`): resample every
        # resolution via per-tile worker processes instead of one whole-city
        # `build_h3_by_resolution` call. NOTE this is only half of the memory
        # win the chunked architecture is designed for: `build_h3_by_resolution_chunked`
        # itself never materializes a full-city GeoDataFrame (see its own
        # docstring), but every downstream consumer in the rest of this
        # function (core/metro split, `_run_regressions_and_anovas`, map
        # building, the plain `h3_grid.parquet` writes below) still expects
        # one whole-city `gpd.GeoDataFrame` per resolution, exactly like
        # `build_h3_by_resolution` returns -- so the tile files are
        # immediately re-concatenated into that same shape here. That
        # reassembly is correctness-preserving (proven equivalent to the
        # unchunked path cell-for-cell by
        # `tests/test_h3_by_resolution_chunked.py`) and lets a
        # `chunked_h3_output` city produce byte-identical downstream results,
        # but it does NOT by itself lower this function's own peak memory --
        # doing that for real needs the downstream stages (stats, map
        # building) rewritten to read the tile files lazily instead of one
        # merged GeoDataFrame, which is deferred (see the chunked-h3
        # migration notes handed to the orchestrating session).
        print(f"[pipeline:{config.key}] resampling every map/stats H3 resolution (chunked, tile_resolution={params.chunked_h3_output})")
        chunk_paths_by_res = build_h3_by_resolution_chunked(
            h3_grid, params, config.uses_census,
            output_dir=str(city_dir / "results" / "metro" / "h3_by_resolution_chunks"),
            worldpop_only_population=config.key in WORLDPOP_ONLY_POPULATION_CITIES,
        )
        h3_by_resolution = {res: _concat_gdf_parquets(paths) for res, paths in chunk_paths_by_res.items()}
    else:
        print(f"[pipeline:{config.key}] resampling every map/stats H3 resolution")
        h3_by_resolution = build_h3_by_resolution(
        h3_grid, params, config.uses_census, worldpop_only_population=config.key in WORLDPOP_ONLY_POPULATION_CITIES
    )
    _add_gba_columns(config, census_dir, aoi_gdf, h3_by_resolution=h3_by_resolution)
    _write_rescale_factors(city_dir)
    h3_grid = h3_by_resolution[params.h3_resolution]
    stats_grid = h3_by_resolution[params.stats_h3_resolution]

    _lap("resample to every h3 resolution")
    print(f"[pipeline:{config.key}] resolving city-core boundary and splitting core/metro")
    core_boundary = resolve_core_boundary(
        config.display_name,
        config.country == "USA",
        state=config.census_states[0] if config.census_states else None,
        geocode_name=config.geocode_name,
        core_file=city_dir / "core.gpkg",
    )
    core_union = core_boundary.union_all()
    try:  # daily vehicle-km inside the AOI / core, shown on the Place-rank tab (non-fatal if it fails)
        from .vehicle_km import compute_vehicle_km

        _extra = {n: gpd.read_file(city_dir / f) for n, f in (("official", "metro_official.gpkg"), ("municipality", "municipality.gpkg")) if (city_dir / f).is_file()}
        _au = _load_area_unions(city_dir)   # 2026-11-13: the very areas of the Place-rank tab (its population is summed over them)
        if _au:
            _g = lambda k: gpd.GeoDataFrame(geometry=[_au[k]], crs=4326)  # noqa: E731
            _vkm = compute_vehicle_km(city_dir, _g("metro"), _g("core"), extra={"official": _g("official"), "municipality": _g("municipality")}, stops_per_feed=config.stops_per_feed, day_type=params.analysis_day_type)
        else:
            _vkm = compute_vehicle_km(city_dir, aoi_gdf, core_boundary, extra=_extra, stops_per_feed=config.stops_per_feed, day_type=params.analysis_day_type)
        print(f"[pipeline:{config.key}] vehicle-km/day: metro {_vkm['metro']['vehicle_km']:,.0f}"
              + (f", core {_vkm['core']['vehicle_km']:,.0f}" if "core" in _vkm else ""))
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline:{config.key}] WARNING: vehicle-km not computed: {type(exc).__name__}: {exc}")
    core_mask = pd.Series(
        shapely.within(_h3_cell_centroids(h3_grid["h3_cell"]), core_union), index=h3_grid.index
    )
    stats_core_mask = pd.Series(
        shapely.within(_h3_cell_centroids(stats_grid["h3_cell"]), core_union), index=stats_grid.index
    )
    area_unions = _load_area_unions(city_dir)
    if area_unions:   # four analysis areas: `metro` is the UN functional urban area, `core` the UN urban centre
        core_mask = pd.Series(_area_masks(h3_grid["h3_cell"], {"core": area_unions["core"]})["core"], index=h3_grid.index)
        summary_metro_mask = pd.Series(_area_masks(h3_grid["h3_cell"], {"metro": area_unions["metro"]})["metro"], index=h3_grid.index)
    else:
        summary_metro_mask = pd.Series(True, index=h3_grid.index)

    _lap("core/metro boundary split")
    if run_stats:
        print(f"[pipeline:{config.key}] running regressions + population-weighted median-split ANOVAs (metro + core)")
        if area_unions:
            stats_masks = _area_masks(stats_grid["h3_cell"], area_unions)
            for _scope in ("metro", "official", "municipality", "core"):
                _run_regressions_and_anovas(stats_grid[stats_masks[_scope]], config.uses_census, city_dir, config.display_name, _scope)
        else:
            _run_regressions_and_anovas(stats_grid, config.uses_census, city_dir, config.display_name, "metro")
            _run_regressions_and_anovas(stats_grid[stats_core_mask], config.uses_census, city_dir, config.display_name, "core")
        _lap("regressions + ANOVAs")

        shares = h3_grid["equity_flag"].value_counts(normalize=True, dropna=True).to_dict()
        percent_area_bar(
            {k: v for k, v in shares.items() if k}, f"{config.display_name}: equity flag area share",
            str(city_dir / "figures" / "equity_area_share.jpg"),
        )

        if area_unions:
            # one grid for the whole processing area (the union of the four areas) with a 0/1 membership column per area; not one copy per area
            (city_dir / "results" / "metro").mkdir(parents=True, exist_ok=True)
            _add_area_flags(h3_grid, area_unions).drop(columns="area_m2", errors="ignore").to_parquet(city_dir / "results" / "metro" / "h3_grid.parquet")
            for _old in ("official", "municipality", "core"):
                shutil.rmtree(city_dir / "results" / _old, ignore_errors=True)
        else:
            h3_grid.drop(columns="area_m2", errors="ignore").to_parquet(city_dir / "results" / "metro" / "h3_grid.parquet")
            h3_grid[core_mask].drop(columns="area_m2", errors="ignore").to_parquet(city_dir / "results" / "core" / "h3_grid.parquet")

        # Opt-in chunked-mode output (`StudyParams.chunked_h3_output`, see its
        # docstring): additionally write one standalone GeoParquet per H3
        # tile-resolution ancestor, built in its own worker process, so
        # DOWNSTREAM consumers (future map-tiling/stats passes) never need
        # `h3_grid` in memory at all -- they can read these tile files
        # instead. IMPORTANT CAVEAT (not yet resolved -- see the report to
        # the orchestrating session): by this point in `run_city_study`,
        # `h3_grid` has ALREADY been fully materialized as one in-memory
        # GeoDataFrame by `build_h3_by_resolution`/`_add_h3_grid` upstream,
        # so writing chunks here is additive/forward-looking -- it does NOT
        # by itself avoid the upstream single-GeoDataFrame OOM that is
        # Shanghai's actual remaining failure. Making `build_h3_by_resolution`
        # itself skip whole-grid materialization for chunked-mode cities
        # (partitioning `read_h3_grid_table`'s raw Polars output BEFORE
        # `h3_ops.to_gdf`/`prepare_grid_for_map` ever run, one tile at a
        # time) is the follow-up piece a Shanghai run would actually need;
        # this call site proves the per-tile GeoParquet writer works
        # end-to-end and gives every other stage (map tiling, stats) a real
        # file set to build against in the meantime. This does NOT change or
        # remove the whole-grid `h3_grid.parquet` write above; both
        # currently coexist so every other consumer of the metro
        # `h3_grid.parquet` file keeps working unmodified.
        if getattr(params, "chunked_h3_output", None) is not None:
            from geohierarchy.chunked_h3_grid import build_h3_grid_chunked

            print(
                f"[pipeline:{config.key}] writing chunked h3 grid "
                f"(tile_resolution={params.chunked_h3_output})"
            )
            chunk_paths = build_h3_grid_chunked(
                pl.from_pandas(pd.DataFrame(h3_grid.drop(columns=["geometry", "area_m2"], errors="ignore"))).with_columns(
                    pl.Series("h3_cell", h3_grid["h3_cell"].to_numpy())
                ),
                str(city_dir / "results" / "metro" / "h3_grid_chunks"),
                tile_resolution=params.chunked_h3_output,
            )
            print(f"[pipeline:{config.key}] wrote {len(chunk_paths)} h3 grid chunk files")
            _lap("chunked h3 grid output")

        write_city_summary(
            city_dir,
            config.display_name,
            metro_access=h3_grid.loc[summary_metro_mask, "level_of_service"].to_numpy(),
            metro_population=h3_grid.loc[summary_metro_mask, "population"].to_numpy(),
            core_access=h3_grid.loc[core_mask, "level_of_service"].to_numpy(),
            core_population=h3_grid.loc[core_mask, "population"].to_numpy(),
            metro_grid=h3_grid.loc[summary_metro_mask],
            core_grid=h3_grid.loc[core_mask],
        )
        _lap("stats: regressions/ANOVAs + summary + h3 grid export")

    print(f"[pipeline:{config.key}] building map")
    if isinstance(access_gdf, Path):
        access_gdf = gpd.read_parquet(access_gdf)
    census_by_level = None
    if config.uses_census:
        print(f"[pipeline:{config.key}] aggregating level_of_service onto census geometries for the map")
        census_by_level = _census_geometries_with_score(
            h3_grid, aoi_gdf, config.census_states, MAP_CENSUS_LEVELS, census_dir,
            country=config.country, census_module=config.census_module,
            chunk_h3_resolution=params.isochrone_chunk_h3_resolution,
            worldpop_only_population=config.key in WORLDPOP_ONLY_POPULATION_CITIES,
        )
        _add_gba_columns(config, census_dir, aoi_gdf, census_by_level=census_by_level, h3_grid=h3_grid)
        _lap("map: census geometries + score aggregation")

        census_out_dir = city_dir / "results" / "census"
        census_out_dir.mkdir(parents=True, exist_ok=True)
        for level, level_gdf in census_by_level.items():
            level_gdf.to_parquet(census_out_dir / f"{level}.parquet")
        _lap("results: per-census-level parquet export")
        # Boston OOM fix (2026-08-26): `_census_geometries_with_score` for a
        # statewide, block-level US city (Boston: ~103k real TIGER block
        # polygons, MA-wide AOI) builds several large intermediate frames per
        # level (sjoin results, jobs/DHC merges, simplified geometry copies)
        # that Python's refcounting mostly frees as the loop moves on -- but
        # nothing forces a *collection* between here and the equally large
        # `h3_by_resolution` dict (every map/stats h3 resolution, up to
        # millions of rows for Boston) that's been resident the whole time.
        # An explicit collect right after the per-level parquet export (the
        # last point `census_by_level`'s per-level frames are rewritten/
        # touched before being read again in `_export_downloads` and
        # `build_city_map`) reclaims that headroom before the next stage
        # adds `routes_lines_gdf`/`development_gdf` on top.
        gc.collect()

    from transitlos.map import build_city_map, build_route_lines

    # `exclude_no_shape_routes=False` here (not the function's own default):
    # this `routes_lines_gdf` also feeds the downloadable `routes.parquet`
    # export just below, which should keep every real route (each row still
    # carries `has_real_shape` so a downstream consumer can filter/label
    # them) -- only the MAP's own line layer should actually hide
    # straight-line-fallback routes (2026-09-01, user-requested default:
    # "if no shapes.txt or for the routes or trips that do not have shapes
    # ... not to display the route shape on the map"), via `map_routes_gdf`
    # below.
    from .stop_scoring import add_route_labels

    # The map draws EVERY stop_id at its own location (own mode, own routes, + its station's scores), not one point per station.
    from .stop_scoring import build_stop_table_file

    stops = _stops_with_stop_table(city_dir, config, stops)
    routes_lines_gdf = build_route_lines(_gtfs_dirs(city_dir / "gtfs"), aoi=aoi_gdf, exclude_no_shape_routes=False)
    map_routes_gdf = _map_routes_gdf_for_display(routes_lines_gdf)

    _apply_map_field_policy()
    _ensure_worldpop_shares(h3_by_resolution)
    _exclude_worldpop_twins(h3_by_resolution)
    development_gdf = development_gdf_for_map(census_by_level)
    gc.collect()

    for _frames in (h3_by_resolution, census_by_level or {}):   # the built-surface vs census columns in the downloads too
        for _k in list(_frames):
            _frames[_k] = _add_gba_vs_census(_frames[_k])
    if not config.worldpop_layers:   # an area with its own census: no WorldPop columns in the results
        for _frames in (h3_by_resolution, census_by_level or {}):
            for _k in list(_frames):
                _frames[_k] = _frames[_k].drop(columns=[c for c in _frames[_k].columns if str(c).startswith("worldpop_")])
    downloads_manifest = _export_downloads(
        city_dir, h3_by_resolution, census_by_level, stops, routes_lines_gdf, core_union, core_boundary.crs,
        area_unions=area_unions,
    )
    _lap("results: per-resolution/per-level/stops/routes download parquet export")
    if not build_map:
        print(f"[pipeline:{config.key}] done (no map: build_map=False)")
        return
    # Boston OOM fix (2026-08-26): this is the peak of the pre-tiling
    # sequence -- `h3_by_resolution` (all resolutions), `census_by_level`
    # (all levels, block included), `development_gdf` (a further
    # census-polygon copy for the equity overlay) and `routes_lines_gdf` are
    # all simultaneously alive right here, one call before
    # `build_city_map`/`write_level_tiles` even starts -- i.e. before the
    # existing H3-res4 chunked-tiling safeguard (`H3_CHUNK_ROW_THRESHOLD` in
    # geohierarchy) is ever reached. A collect here is the last chance to
    # shed any garbage from the exports above before the tile build's own
    # (already memory-tight, per the `RAYON_NUM_THREADS`/worker-cap fixes
    # elsewhere in this function) allocations stack on top.
    gc.collect()

    stats_grid = filter_map_columns(city_dir, h3_by_resolution, census_by_level, stats_grid)
    _register_relative_fields(city_dir)
    with _tile_worker_cap(tile_workers):
        _map_h3, _map_census, _map_edges, _map_dev = _map_layers(config, h3_by_resolution, census_by_level, access_gdf, development_gdf)
        build_city_map(
            h3_by_resolution=_map_h3,
            edges_gdf=_map_edges,
            tiles_dir=str(city_dir / "map_tiles"),
            out_html=str(city_dir / "map.html"),
            census_by_level=_map_census or None,
            stats_by_area=_stats_by_area(city_dir, stats_grid, stats_core_mask),
            is_us=config.uses_census,
            stops_gdf=stops,
            routes_gdf=map_routes_gdf,
            region=params.region,
            development_gdf=_map_dev,
            include_streets=not getattr(config, "map_finest_only", False),
            include_development=not getattr(config, "map_finest_only", False),
            renderer=os.environ.get("TRANSITLOS_RENDERER", "maplibre"),
            downloads_manifest=downloads_manifest,
            share_source_map=_share_column_sources(stats_grid),
            country=config.country,
        )
    _write_census_metadata_json(city_dir, stats_grid, h3_by_resolution, census_by_level, config.country, _share_column_sources(stats_grid))
    _lap("map: build_city_map (tile generation + HTML, incl. development overlay)")

    print(f"[pipeline:{config.key}] building H3 res-11 population chunks for the map editor")
    try:
        from transitlos.map.pop_chunks import build_population_chunks_from_parquet

        pop_chunks_summary = build_population_chunks_from_parquet(
            str(city_dir / "results" / "metro" / "h3_grid.parquet") if run_stats
            else str(city_dir / "results" / "metro" / "h3_grid.parquet"),
            str(city_dir / "pop_chunks"),
        )
        print(f"[pipeline:{config.key}]  -> {pop_chunks_summary}")
    except Exception as exc:  # pragma: no cover - never let this abort an otherwise-finished run
        print(f"[pipeline:{config.key}] WARNING: population chunk generation failed: {exc}")
    _lap("map: population chunks for editor")
    if _load_area_unions(city_dir):
        _inject_area_outlines(city_dir)
    else:
        _inject_core_outline(city_dir, core_boundary)
    print(f"[pipeline:{config.key}] done")


def refresh_census_only(
    city_dir: Path,
    config: CityConfig,
    params: StudyParams,
    census_root: Optional[Path] = None,
    tile_workers: Optional[int] = 4,
) -> None:
    """Re-run just the census join + map/stats rebuild, reusing cached pipeline output.

    Fast re-entry point for the case the user asked for explicitly: a city
    was first run WorldPop-only (its `CityConfig.uses_census` was False, or
    its `census_module` wasn't wired up yet -- e.g. Hamburg pending Germany
    GENESIS API credentials), and census support later becomes available.
    Re-running the *entire* pipeline just to pick up census columns would
    redo the expensive GTFS load, street network build, and isochrone/LOS
    computation for no reason -- all of that is untouched by census
    availability. This instead:

      1. Loads the cached `results/stops.parquet`, `results/access_edges.parquet`
         (untouched, reused as-is), and `results/metro/h3_grid.parquet`
         (the pre-census-join, population+access h3 grid the earlier
         WorldPop-only run produced -- see `run_city_study`, which always
         saves this same grid to this same path).
      2. Re-joins ACS/DHC/LODES census attributes onto it (`_join_census`,
         `_join_race`, `_join_jobs`) -- this is the one stage that couldn't
         run before.
      3. Re-runs everything downstream that depends on census columns:
         resample-to-every-resolution, core/metro split, regressions/ANOVAs,
         `results/census/*.parquet` export, and the map rebuild -- via the
         same `_finish_pipeline_stages` helper `run_city_study` itself uses,
         so this can't silently drift from a real end-to-end run.

    Raises `FileNotFoundError` if the cached stops/access-edges/h3-grid
    aren't there yet -- this is a re-entry point, not a substitute for a
    first full `run_city_study` run.
    """
    city_dir = Path(city_dir)
    census_dir = Path(census_root) if census_root is not None else city_dir / "uscensus"
    census_dir.mkdir(parents=True, exist_ok=True)

    if not config.uses_census:
        print(
            f"[pipeline:{config.key}] refresh_census_only: config.uses_census is still False "
            "(census module not wired up / not supported for this country yet) -- nothing to refresh"
        )
        return

    _set_bare_source_file(city_dir)
    stops_path = city_dir / "results" / "stops.parquet"
    access_path = city_dir / "results" / "access_edges.parquet"
    h3_grid_path = city_dir / "results" / "metro" / "h3_grid.parquet"
    for p in (stops_path, access_path, h3_grid_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"refresh_census_only requires a prior full run's cached output at {p}, "
                "but it doesn't exist -- run the full pipeline (run.py with no flags) first."
            )

    aoi_gdf = _load_aoi_gdf(city_dir, config)
    print(f"[pipeline:{config.key}] refresh_census_only: loading cached stops + h3 grid (skipping GTFS/network/isochrones)")
    stops = gpd.read_parquet(stops_path)
    access_gdf = access_path  # lazy, matches run_city_study's reuse_cached_los path
    h3_grid = gpd.read_parquet(h3_grid_path)

    # Real bug fixed 2026-08-23: `h3_grid_path` is BOTH this function's input
    # AND the exact path `_finish_pipeline_stages` overwrites it with at the
    # end (see its own docstring/call site) -- so re-running
    # `refresh_census_only` a second time (e.g. to pick up a pyCensus/join
    # fix, which is the entire point of this re-entry point existing) loads
    # a grid that already carries the PREVIOUS run's census columns, both
    # still-prefixed (`statcan_foreignBornPopulation`) and any renamed-bare
    # canonical ones (`foreignBornPopulation`) `_rename_canonical_columns`
    # produced last time. Two compounding failures followed for Toronto:
    # (1) `_join_polygon_stats`'s `keep_col not in h3_grid.columns` check
    # saw the prefixed column already existed (from the previous run) and
    # took its "only backfill real NaN" branch instead of writing fresh
    # values -- and since a false non-NaN 0.0 (or any stale non-NaN value)
    # was already sitting there from an earlier, differently-broken run,
    # `missing = h3_grid[keep_col].isna()` never fired and the fresh, now-
    # correct join value was silently discarded every single re-run,
    # forever. (2) `_rename_canonical_columns` saw the bare canonical name
    # already existed too (left over from the previous run's own rename)
    # and skipped renaming the fresh prefixed column into it, so even a
    # correctly-refetched value would've been stranded under the prefixed
    # name while every downstream consumer read the bare (stale) one.
    # Reproduced live 2026-08-23: re-running `_join_census` on Toronto's
    # on-disk `h3_grid.parquet` (which already carried a stale, all-0.0
    # `statcan_foreignBornPopulation` from an earlier run) reproduced the
    # exact 0.0 sum; dropping every census-derived column first and
    # re-running the identical join produced the correct 3,013,955. Fixed
    # by dropping every census-derived column (prefixed and
    # renamed-to-bare-canonical alike, via `_census_columns` -- excludes
    # bare `population`, which is WorldPop's and must never be touched)
    # before the joins below, so each `refresh_census_only` call always
    # re-joins from a clean slate instead of partially, permanently
    # blocking on its own previous output.
    stale_census_cols = [c for c in _census_columns(h3_grid) if c in h3_grid.columns]
    if stale_census_cols:
        print(
            f"[pipeline:{config.key}] refresh_census_only: dropping {len(stale_census_cols)} stale "
            "census column(s) from a previous run before re-joining fresh"
        )
        h3_grid = h3_grid.drop(columns=stale_census_cols)

    print(f"[pipeline:{config.key}] joining census attributes for country={config.country!r} ({len(params.census_levels)} levels)")
    # Bug fix (2026-09-05, live report -- hexagonal density artifact at
    # H3 res-5 tile/chunk boundaries): `_join_polygon_stats` (via
    # `_join_census`) normalizes each census polygon's population share
    # using ONLY the h3 cells visible in its current call -- when chunked,
    # a polygon straddling a chunk boundary gets its FULL real population
    # redistributed independently within EACH chunk that sees any of its
    # cells, so its true total is counted once per chunk it touches
    # (confirmed live: a 2-chunk-straddling polygon's population summed to
    # ~2x its real value). This join was never the actual OOM crash site in
    # this study (isochrones and the census-geometry map aggregation were,
    # both separately and correctly chunked) -- always run it unchunked to
    # avoid this real correctness bug, regardless of `isochrone_chunk_h3_resolution`.
    _set_floor_area(config, aoi_gdf, params.h3_resolution, _gba_cache_root(census_dir))
    h3_grid = _join_polygon_stats_lightweight(
        _join_census, h3_grid, aoi_gdf, config.census_states, params.census_levels, census_dir,
        country=config.country, census_module=config.census_module,
    )
    print(f"[pipeline:{config.key}] joining decennial race/ethnicity attributes")
    # See the matching comment above `_join_census` a few lines up --
    # same chunk-boundary double-counting bug, always run unchunked.
    h3_grid = _join_polygon_stats_lightweight(_join_race, h3_grid, aoi_gdf, config.census_states, census_dir, country=config.country)
    print(f"[pipeline:{config.key}] joining LODES WAC jobs-by-workplace counts")
    try:
        # Same chunk-boundary double-counting bug as `_join_census`/
        # `_join_race` above -- always run unchunked.
        h3_grid = _join_polygon_stats_lightweight(_join_jobs, h3_grid, aoi_gdf, config.census_states, census_dir, country=config.country)
    except Exception as exc:  # pragma: no cover - LODES availability varies by state/year
        print(f"[pipeline] skipping LODES jobs join: {exc}")
    h3_grid = _rename_canonical_columns(h3_grid)

    if config.exclude_cells_touching_country_border:
        h3_grid = _filter_cells_touching_country_border(
            h3_grid, config.border_country_geocode_name, city_dir / "border_clip"
        )

    if config.census_worldpop_gapfill:
        gapfill_pop_col = CENSUS_WORLDPOP_GAPFILL_POPULATION_COLUMN.get(config.country)
        if gapfill_pop_col is None:
            print(
                f"[pipeline:{config.key}] census_worldpop_gapfill is set but country={config.country!r} "
                "has no entry in CENSUS_WORLDPOP_GAPFILL_POPULATION_COLUMN -- skipping"
            )
        else:
            print(f"[pipeline:{config.key}] WorldPop census gap-fill enabled -- filling uncovered h3 cells")
            h3_grid = _add_worldpop_gapfill(
                h3_grid, aoi_gdf, params.h3_resolution,
                _worldpop_year_from_filename(config.worldpop_filename),
                city_dir / "worldpop_gapfill", gapfill_pop_col,
            )

    if os.environ.get("CS_TRANSITLOS_SKIP_WORLDPOP") == "1" or not config.worldpop_layers:
        print(f"[pipeline:{config.key}] skipping WorldPop demographic layers")
    else:
        # Restricted to real census-covered cells for every country EXCEPT
        # Beersheba (`config.census_worldpop_gapfill`) -- see
        # `_add_worldpop_demographic_layers`'s `restrict_to_population_column`
        # docstring for why Beersheba is the one deliberate exception.
        # 2026-11-10 (explicit user request: "I want all WorldPop data"): the WorldPop demographic layers fill EVERY cell WorldPop covers, census or not
        demographic_restrict_col = None
        print(
            f"[pipeline:{config.key}] joining WorldPop demographic layers (worldpop_* columns), "
            f"restricted to real census coverage via {demographic_restrict_col!r}"
            if demographic_restrict_col is not None
            else f"[pipeline:{config.key}] joining WorldPop demographic layers (worldpop_* columns) onto full grid"
        )
        h3_grid = _add_worldpop_demographic_layers(
            h3_grid, aoi_gdf, params.h3_resolution,
            _worldpop_year_from_filename(config.worldpop_filename),
            city_dir / "worldpop_demographic",
            restrict_to_population_column=demographic_restrict_col,
        )

    _finish_pipeline_stages(
        city_dir, config, params, census_dir, aoi_gdf, h3_grid, stops, access_gdf,
        tile_workers=tile_workers, run_stats=True,
    )


def rebuild_map_only(
    city_dir: Path,
    config: CityConfig,
    params: StudyParams,
    census_root: Optional[Path] = None,
    rebuild_tiles: bool = False,
    tile_workers: Optional[int] = 4,
    build_pop_chunks: bool = False,
    use_pmtiles: bool = True,
    enable_place_comparison: Optional[bool] = None,
    style_only: bool = False,
) -> None:
    """Fast map/tiles-only rebuild, reusing cached `results/*.parquet`.

    Consolidates what used to be separate per-city `rebuild_map_only.py`
    scripts (`boston/`, `guadalajara/`, `boston_city/`) into one shared,
    city-agnostic implementation, driven by a `--map-only`/`--tiles-only`
    flag on `run.py` instead. Skips the GTFS/network/isochrone/WorldPop/
    census-join stages entirely and skips the stats/regression/ANOVA stage
    too (unlike `refresh_census_only`, which keeps stats) -- this is purely
    for iterating on map HTML/JS/CSS or, with `rebuild_tiles=True`, redrawing
    tiles from already-computed `level_of_service` values.

    `style_only` (2026-09-22, explicit user request -- "I dont want to
    recompute tiles if I only want some minor color changes or map html
    style changes... right now it seems it is not even with the map only
    flag"): even a plain `--map-only` still redoes three substantial
    CPU-bound stages every time -- H3 resampling to every map resolution,
    population-weighted score aggregation onto every real census polygon,
    and rebuilding every GTFS route's line geometry -- none of which a
    style/color/HTML change needs, since none of them depend on anything
    `build_city_map` doesn't already receive as a plain argument. When
    `style_only=True` and a prior real (non-style-only) run already wrote
    `_write_map_inputs_checkpoint`'s cache, those three stages (plus the
    downloads re-export, another real cost for a large city) are skipped
    entirely in favor of a pure-I/O reload -- see that function's own
    docstring. Falls back to the normal full computation (and writes a
    fresh checkpoint) if no checkpoint exists yet, so this flag is always
    safe to pass even before the first real `--map-only` run.
    """
    _set_bare_source_file(city_dir)
    city_dir = Path(city_dir)
    census_dir = Path(census_root) if census_root is not None else city_dir / "uscensus"

    h3_grid_path = city_dir / "results" / "metro" / "h3_grid.parquet"
    access_path = city_dir / "results" / "access_edges.parquet"
    stops_path = city_dir / "results" / "stops.parquet"
    for p in (h3_grid_path, access_path, stops_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"rebuild_map_only requires a prior full run's cached output at {p}, "
                "but it doesn't exist -- run the full pipeline (run.py with no flags) first."
            )

    if build_pop_chunks:
        from transitlos.map.pop_chunks import build_population_chunks_from_parquet

        print(f"[pipeline:{config.key}] building H3 res-11 population chunks for the editor")
        summary = build_population_chunks_from_parquet(str(h3_grid_path), str(city_dir / "pop_chunks"))
        print(f"[pipeline:{config.key}]  -> {summary}")

    from transitlos.map import build_city_map

    access_gdf = gpd.read_parquet(access_path, columns=["level_of_service", "geometry"])
    stops = gpd.read_parquet(stops_path)
    stops = _stops_with_stop_table(city_dir, config, stops)

    checkpoint = _read_map_inputs_checkpoint(city_dir) if style_only else None
    downloads_manifest_path = city_dir / "results" / "downloads_manifest.json"
    if checkpoint is not None and downloads_manifest_path.is_file():
        print(f"[pipeline:{config.key}] style-only: reusing cached H3-resample/census-geometry/routes/downloads (pure I/O, no recompute)")
        h3_by_resolution, census_by_level, stats_grid, routes_lines_gdf, stats_core_mask = checkpoint
        map_routes_gdf = _map_routes_gdf_for_display(routes_lines_gdf)
        downloads_manifest = json.loads(downloads_manifest_path.read_text())
    else:
        if style_only:
            print(f"[pipeline:{config.key}] style-only: no checkpoint yet -- falling back to a full (slower) rebuild, which will write one for next time")
        print(f"[pipeline:{config.key}] loading cached results")
        h3_table = read_h3_grid_table(h3_grid_path)
        aoi_gdf = _load_aoi_gdf(city_dir, config)

        print(f"[pipeline:{config.key}] resampling H3 resolutions (cheap, from cache)")
        h3_by_resolution = build_h3_by_resolution(
        h3_table, params, config.uses_census, worldpop_only_population=config.key in WORLDPOP_ONLY_POPULATION_CITIES
    )
        _write_rescale_factors(city_dir)
        del h3_table
        import gc as _gc
        _gc.collect()
        h3_grid = h3_by_resolution[params.h3_resolution]
        stats_grid = h3_by_resolution[params.stats_h3_resolution]

        core_boundary = resolve_core_boundary(
            config.display_name, config.country == "USA",
            state=config.census_states[0] if config.census_states else None,
            geocode_name=config.geocode_name,
            core_file=city_dir / "core.gpkg",
        )
        core_union = core_boundary.union_all()
        stats_core_mask = stats_grid.geometry.centroid.to_crs(4326).within(core_union)

        census_by_level = None
        if config.uses_census:
            print(f"[pipeline:{config.key}] aggregating level_of_service onto census geometries")
            census_by_level = _census_geometries_with_score(
                h3_grid, aoi_gdf, config.census_states, MAP_CENSUS_LEVELS, census_dir,
                country=config.country, census_module=config.census_module,
                chunk_h3_resolution=params.isochrone_chunk_h3_resolution,
                worldpop_only_population=config.key in WORLDPOP_ONLY_POPULATION_CITIES,
            )

        from transitlos.map import build_route_lines

        print(f"[pipeline:{config.key}] building transit-line geometry from GTFS")
        # `exclude_no_shape_routes=False` -- see the matching comment in
        # `run_city_study` above: `routes_lines_gdf` also feeds the
        # downloadable `routes.parquet` export a few lines down and should keep
        # every real route; only the map's own line layer (`map_routes_gdf`)
        # hides straight-line-fallback routes.
        routes_lines_gdf = build_route_lines(
            [p for p in sorted((city_dir / "gtfs").iterdir()) if p.is_dir()], aoi=aoi_gdf, exclude_no_shape_routes=False,
        )
        map_routes_gdf = _map_routes_gdf_for_display(routes_lines_gdf)
        print(f"[pipeline:{config.key}]  -> {len(routes_lines_gdf)} route lines")

        for _frames in (h3_by_resolution, census_by_level or {}):   # the built-surface vs census columns in the downloads too
            for _k in list(_frames):
                _frames[_k] = _add_gba_vs_census(_frames[_k])
        downloads_manifest = _export_downloads(
            city_dir, h3_by_resolution, census_by_level, stops, routes_lines_gdf, core_union, core_boundary.crs,
            area_unions=_load_area_unions(city_dir),
        )
        _write_map_inputs_checkpoint(
            city_dir, h3_by_resolution, census_by_level, routes_lines_gdf, params.stats_h3_resolution, stats_core_mask,
        )

    print(
        f"[pipeline:{config.key}] rebuilding tiles + HTML" if rebuild_tiles
        else f"[pipeline:{config.key}] building map HTML only (skip_tile_build=True)"
    )

    _apply_map_field_policy()
    _ensure_worldpop_shares(h3_by_resolution)
    _exclude_worldpop_twins(h3_by_resolution)
    development_gdf = development_gdf_for_map(census_by_level)
    stats_grid = filter_map_columns(city_dir, h3_by_resolution, census_by_level, stats_grid)
    _register_relative_fields(city_dir)
    with _tile_worker_cap(tile_workers):
        _map_h3, _map_census, _map_edges, _map_dev = _map_layers(config, h3_by_resolution, census_by_level, access_gdf, development_gdf)
        build_city_map(
            h3_by_resolution=_map_h3,
            edges_gdf=_map_edges,
            tiles_dir=str(city_dir / "map_tiles"),
            out_html=str(city_dir / "map.html"),
            census_by_level=_map_census or None,
            stats_by_area=_stats_by_area(city_dir, stats_grid, stats_core_mask),
            is_us=config.uses_census,
            stops_gdf=stops,
            routes_gdf=map_routes_gdf,
            region=params.region,
            skip_tile_build=not rebuild_tiles,
            development_gdf=_map_dev,
            include_streets=not getattr(config, "map_finest_only", False),
            include_development=not getattr(config, "map_finest_only", False),
            renderer=os.environ.get("TRANSITLOS_RENDERER", "maplibre"),
            use_pmtiles=use_pmtiles,
            enable_place_comparison=(
                enable_place_comparison if enable_place_comparison is not None else False
            ),
            downloads_manifest=downloads_manifest,
            share_source_map=_share_column_sources(stats_grid),
            country=config.country,
        )
    _write_census_metadata_json(city_dir, stats_grid, h3_by_resolution, census_by_level, config.country, _share_column_sources(stats_grid))
    if _load_area_unions(city_dir):
        _inject_area_outlines(city_dir)
    else:
        _inject_core_outline(city_dir, core_boundary)
    print(f"[pipeline:{config.key}] done -> {city_dir / 'map.html'}")


def rebuild_development_tiles_only(
    city_dir: Path,
    config: CityConfig,
    params: StudyParams,
    census_root: Optional[Path] = None,
) -> None:
    """Redraw just the development-opportunity overlay's tiles from census polygons.

    Consolidates the former `boston/rebuild_development_only.py`. Targeted
    repair for the case `build_city_map`'s h3-hexagon development tiles (its
    unconditional first pass) survive because a `rebuild_map_only` run was
    interrupted before the census redraw that normally replaces them ran.
    Touches only `map_tiles/development/`, never `map.html` or any other
    layer's tiles.
    """
    city_dir = Path(city_dir)
    census_dir = Path(census_root) if census_root is not None else city_dir / "uscensus"
    h3_grid_path = city_dir / "results" / "metro" / "h3_grid.parquet"
    if not h3_grid_path.is_file():
        raise FileNotFoundError(f"rebuild_development_tiles_only requires cached {h3_grid_path}")

    print(f"[pipeline:{config.key}] loading cached h3 grid + AOI")
    h3_grid = gpd.read_parquet(h3_grid_path)
    aoi_gdf = _load_aoi_gdf(city_dir, config)

    h3_by_resolution = build_h3_by_resolution(
        h3_grid, params, config.uses_census, worldpop_only_population=config.key in WORLDPOP_ONLY_POPULATION_CITIES
    )
    h3_grid = h3_by_resolution[params.h3_resolution]
    _write_rescale_factors(city_dir)

    if not config.uses_census:
        print(f"[pipeline:{config.key}] not a census city -- h3 development overlay is correct, nothing to do")
        return

    census_by_level = _census_geometries_with_score(
        h3_grid, aoi_gdf, config.census_states, MAP_CENSUS_LEVELS, census_dir,
        country=config.country, census_module=config.census_module,
        chunk_h3_resolution=params.isochrone_chunk_h3_resolution,
        worldpop_only_population=config.key in WORLDPOP_ONLY_POPULATION_CITIES,
    )
    development_gdf = development_gdf_for_map(census_by_level)
    if development_gdf is None:
        print(f"[pipeline:{config.key}] no census development geometry available -- leaving h3 overlay in place")
        return

    rebuild_development_tiles_from_census(development_gdf, h3_by_resolution, city_dir / "map_tiles", city_dir)
    print(f"[pipeline:{config.key}] done -- map.html and all other tile layers untouched")


def export_results_gpkg(city_dir: Path) -> list[Path]:
    """Export a city's parquet result layers to GeoPackage for GIS-desktop viewing.

    Consolidates the former per-city `export_gpkg.py`. Returns the list of
    `.gpkg` paths written (skipping any parquet that doesn't exist yet).
    """
    city_dir = Path(city_dir)
    parquet_files = [
        city_dir / "results" / "access_edges.parquet",
        city_dir / "results" / "core" / "h3_grid.parquet",
        city_dir / "results" / "metro" / "h3_grid.parquet",
    ]
    written = []
    for parquet_path in parquet_files:
        if not parquet_path.is_file():
            print(f"[pipeline] skip (missing): {parquet_path}")
            continue
        gdf = gpd.read_parquet(parquet_path)
        gpkg_path = parquet_path.with_suffix(".gpkg")
        gdf.to_file(gpkg_path, driver="GPKG")
        written.append(gpkg_path)
        print(f"[pipeline] wrote {gpkg_path}")
    return written


def run_city_study(
    city_dir: Path,
    config: CityConfig,
    params: StudyParams,
    streets_root: Path,
    worldpop_root: Path,
    census_root: Optional[Path] = None,
    date_range: Optional[Union[tuple, dict]] = None,
    strict_date: bool = False,
    reuse_cached_los: bool = False,
    tile_workers: Optional[int] = 4,
    build_map: bool = True,
) -> None:
    """Run the full per-city pipeline: stops -> LOS -> H3 population -> stats -> core/metro -> figures/map.

    Memory dial (set by `run_all.py` / `memsched.py`, no code change per city): the environment variable `STUDY_CHUNK_H3_RES` (3..6) makes every
    stage that is chunked (isochrones, census join, H3 resampling, map-build census geometry) process the area in H3 cells of that resolution
    (res 4 ~ 1,770 km2, 5 ~ 250 km2, 6 ~ 36 km2, i.e. county / municipality sized pieces), each with its halo buffer (`STUDY_CHUNK_BUFFER_M`, default
    the params' buffer), and join the results; a smaller piece needs less memory. A city whose estimate does not fit the machine, or that ran out of
    memory, is rerun by the scheduler with the next finer resolution.

    Args:
        city_dir: City folder (e.g. `city_science_network/boston`), holding
            `aoi.gpkg`/`gtfs/` already, and where `streets/`,
            `worldpop/`, `results/`, `figures/`, `map.html` are (re)created.
        config: Static city facts (`code.city_config.CityConfig`).
        params: Study-wide parameters (`studytools.params.StudyParams`), identical
            across every city.
        streets_root: Directory holding the shared `.osm.pbf` files
            (`TransitLOSStudies/streets`, shared by every study, not just
            `city_science_network`'s cities).
        worldpop_root: Directory holding the shared WorldPop rasters
            (`TransitLOSStudies/worldpop`, likewise shared).
        census_root: Directory the census loaders (ACS/DHC/LODES via
            `pyCensus`) cache fetched data under (`TransitLOSStudies/census`,
            shared across every study -- two different cities in the same
            country/state share
            most of their block-group-level fetches, so a shared cache
            avoids redundant Census API calls, not just redundant disk
            usage). Defaults to `city_dir / "uscensus"` (the old, per-city
            location) when not given, for any caller that hasn't been
            updated to pass it explicitly.
        strict_date: With a pinned `date_range`, use that date as it is (no swap to a feed's own typical day): a date chosen by the user.
        date_range: Optional `(start_date, end_date)` forwarded to GTFS
            loading; strongly recommended once a valid service date is
            known (skips an expensive auto-detection fallback). When
            `config.stops_per_feed` is set, this may instead be a
            `{feed_directory_name: (start_date, end_date)}` dict pinning
            only specific feeds (see `taipei/run.py`'s `TAIPEI_DATE_RANGE`
            for a real example) -- see
            `transitlos.stops.download_and_prepare_stops_per_feed`'s
            docstring.
        reuse_cached_los: Skip the GTFS load, stop scoring and isochrone
            computation, loading `results/stops.parquet` +
            `results/access_edges.parquet` from a previous run instead and
            resuming at the WorldPop/H3 stage. Purely an operational escape
            hatch for restarting a run that died in a *later* stage (as one
            did on 2026-08-13, OOM-killed during the H3 join) without paying
            for the earlier ones again; it changes no results, and it falls
            back to computing them normally if either file is missing. Never
            set it after changing anything that feeds `level_of_service`.
        tile_workers: Cap on `geohierarchy`'s vector-tile-build process pool
            (default 4). `HierarchyMap.build`'s workers are *forked*
            subprocesses that inherit a copy-on-write copy of this entire
            process's memory at fork time -- not just the (often small)
            slice of data a given tile job needs -- so an unbounded pool
            sized to `os.cpu_count()` has repeatedly OOM-crashed this
            machine even on modest datasets (measured: a single-level,
            3,661-feature development overlay alone drove 16 default
            workers to ~9-15 GB RSS each). Pass `None` to leave
            `os.cpu_count()` alone (full parallelism, only safe on a machine
            with the RAM to match), or a higher explicit number to trade
            some of that safety margin back for speed.
        build_map: False = stop once `results/downloads/` is written (no `map.html`, no tiles).
    """
    params = apply_memory_dial(params)
    if os.environ.get("STUDY_CHUNK_H3_RES"):
        print(f"[pipeline] memory dial: chunks of H3 res {params.isochrone_chunk_h3_resolution}, buffer {params.isochrone_chunk_buffer_m:.0f} m", flush=True)
    _set_bare_source_file(city_dir)
    import time as _time
    _t0 = [_time.time()]

    def _lap(label: str) -> None:
        now = _time.time()
        print(f"[pipeline:{config.key}][TIMING] {label}: {now - _t0[0]:.1f}s")
        _t0[0] = now

    city_dir = Path(city_dir)
    census_dir = Path(census_root) if census_root is not None else city_dir / "uscensus"
    for sub in ("streets", "results", "figures"):
        (city_dir / sub).mkdir(parents=True, exist_ok=True)
    census_dir.mkdir(parents=True, exist_ok=True)
    (city_dir / "results" / "core").mkdir(parents=True, exist_ok=True)
    (city_dir / "results" / "metro").mkdir(parents=True, exist_ok=True)

    aoi_gdf = _load_aoi_gdf(city_dir, config)
    aoi = AreaOfInterest(aoi_gdf)

    stops_path = city_dir / "results" / "stops.parquet"
    access_path = city_dir / "results" / "access_edges.parquet"
    resuming = reuse_cached_los and stops_path.is_file() and access_path.is_file()

    if resuming:
        print(f"[pipeline:{config.key}] reuse_cached_los: loading cached stops + access edges")
        stops = gpd.read_parquet(stops_path)
        # `access_gdf` itself isn't loaded eagerly here: it's only actually
        # needed later, for `build_city_map`'s `edges_gdf=` well after the
        # H3/census/jobs join stage. Passing `access_path` straight through
        # to `population_and_access_to_h3` lets its own edge_chunk_size
        # chunking cover the *initial* parquet read too, instead of this
        # eagerly materializing all ~4.9M edges' Shapely geometry (~3GB on
        # Boston) up front, alongside the H3 join's own peak memory -- see
        # `population_and_access_to_h3`'s `streets_edges` docstring.
        access_gdf = access_path
        print(f"[pipeline:{config.key}]  -> {len(stops)} stops (scored edges loaded lazily/chunked)")
        _lap("load cached stops + level of service")
    else:
        print(f"[pipeline:{config.key}] preparing street network")
        network = prepare_street_network(
            aoi, cache_dir=str(city_dir / "streets"), pbf_path=str(streets_root / config.pbf_filename),
            cluster_distance=params.street_simplify_distance,
        )

        _lap("street network")
        print(f"[pipeline:{config.key}] loading + scoring stops")
        # `config.stops_per_feed` (real cities needing it: Taipei -- see its
        # `CityConfig` entry's docstring) scores each `gtfs/` subdirectory
        # independently and concatenates, instead of stacking every feed
        # into one `Feed` and picking a single shared representative date
        # -- needed when different feeds' real calendar validity windows
        # don't overlap well enough for one date to represent all of them.
        # Scoring = the USA study's (best mode-tier x speed-cutoff subset per station, strict 150 m grouping, expired feeds scored
        # alone, same-location merge): see code/stop_scoring.py. `config.stops_per_feed` scores each feed on its own date.
        from .stop_scoring import score_city_stops

        stops = score_city_stops(
            _gtfs_dirs(city_dir / "gtfs"), aoi_gdf, per_feed=config.stops_per_feed, date_range=date_range, strict_date=strict_date,
            start_time=params.analysis_start_time, end_time=params.analysis_end_time, day_type=params.analysis_day_type,
        )
        stops.to_parquet(stops_path)

        _lap("stops load + scoring")
        print(f"[pipeline:{config.key}] computing street-level level of service")
        access_gdf = compute_level_of_service(
            aoi, network=network, stops=stops, region=params.region,
            walk_distance_steps=params.walk_distance_steps, score_bins=params.score_bins,
            max_walk_distance_m=params.walk_distance_steps[-1],
            # Opt-in memory-bounded isochrone chunking (see
            # `StudyParams.isochrone_chunk_h3_resolution`'s docstring) --
            # `None` (the default for every city not explicitly overridden)
            # forwards straight through to `AccessibilityAnalyzer.run` as
            # `chunk_h3_resolution=None`, the original whole-network path,
            # byte-for-byte unaffected.
            chunk_h3_resolution=params.isochrone_chunk_h3_resolution,
            chunk_buffer_m=params.isochrone_chunk_buffer_m,
        )
        access_gdf["level_of_service"] = discretize_score(access_gdf["level_of_service"].to_numpy(), params.score_bin_width)
        access_gdf.to_parquet(access_path)
        # The full-AOI street network is the single biggest object in the run
        # and nothing after this point uses it; dropping it here frees several
        # GB before the (memory-hungry) H3 join.
        del network

        _lap("level of service (isochrones)")

    # Checkpoint (2026-09-01, user-requested): for a non-census city, the
    # sequence from here through `_join_worldpop_global_schema` is the single
    # most expensive, most failure-prone stretch of the whole pipeline --
    # ~90-110 minutes on Shanghai's real scale (10.25M street edges, 25.7M
    # H3 cells), and it has to be redone byte-for-byte on every retry even
    # though it's fully deterministic given `access_gdf`/the AOI. Cache its
    # output (`pop_access_h3`, still geometry-free/lightweight at this
    # point) to parquet right after it's computed, and reuse that cache on
    # a later run instead of recomputing -- so a later-stage OOM (e.g. in
    # `_add_h3_grid`'s geometry construction, downstream of this) doesn't
    # cost another 90+ minutes to even get back to the point of failure.
    # Census cities skip this (their own per-source joins are comparatively
    # cheap and already resumable via `reuse_cached_los`'s stops/access
    # caching above).
    pop_access_checkpoint_path = city_dir / "results" / "pop_access_h3_checkpoint.parquet"
    if not config.uses_census and pop_access_checkpoint_path.is_file():
        print(f"[pipeline:{config.key}] loading cached pop_access_h3 checkpoint (skipping edge join + worldpop global-schema join)")
        pop_access_h3 = pl.read_parquet(pop_access_checkpoint_path)
        if getattr(config, "gba_buildings", False):   # the built-surface estimate is cheap: always recomputed (the checkpoint may hold an older rule)
            pop_access_h3 = pop_access_h3.with_columns(pl.Series(GBA_POPULATION_COLUMN, _gba_cell_population(
                config, city_dir, aoi_gdf, census_dir, pop_access_h3["h3_cell"].to_list(), params.h3_resolution,
                worldpop=pop_access_h3["population"].to_numpy() if "population" in pop_access_h3.columns else None)))
        _lap("worldpop -> h3 + street access join + worldpop global-schema join (from checkpoint)")
    else:
        print(f"[pipeline:{config.key}] resampling WorldPop to H3 + assigning level_of_service (distance-based touching)")
        # every cell within `h3_street_max_dist_m` of a street, whatever its population
        pop_access_h3 = population_and_access_to_h3(
            str(worldpop_root / config.worldpop_filename), aoi_gdf, access_gdf, resolution=params.h3_resolution,
            street_max_dist_m=params.h3_street_max_dist_m,
        )
        if getattr(config, "gba_buildings", False):   # built-surface population on the street cells (census rate x cell built area)
            pop_access_h3 = pop_access_h3.with_columns(pl.Series(GBA_POPULATION_COLUMN, _gba_cell_population(
                config, city_dir, aoi_gdf, census_dir, pop_access_h3["h3_cell"].to_list(), params.h3_resolution,
                worldpop=pop_access_h3["population"].to_numpy() if "population" in pop_access_h3.columns else None)))

        _lap("worldpop -> h3 + street access join")
        if not config.uses_census:
            print(
                f"[pipeline:{config.key}] no census support for country={config.country!r} -- "
                "joining WorldPop global-schema layers instead (see _join_worldpop_global_schema)"
            )
            pop_access_h3 = _join_worldpop_global_schema(
                pop_access_h3,
                aoi_gdf,
                params.h3_resolution,
                _worldpop_year_from_filename(config.worldpop_filename),
                city_dir / "worldpop_global_schema",
            )
            _lap("worldpop global-schema join")
            pop_access_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            pop_access_h3.write_parquet(pop_access_checkpoint_path)
            print(f"[pipeline:{config.key}] wrote pop_access_h3 checkpoint -> {pop_access_checkpoint_path}")

    # 2026-09-23 (superseding the 2026-09-06 `worldpop_population` alias
    # this comment used to describe): the map-facing WorldPop count now
    # lives under `worldpop_residents`, materialized once at the very end of
    # grid preparation by `_finalize_population_columns`/
    # `_finalize_population_columns_polars` (called from
    # `prepare_grid_for_map`/`_prepare_grid_polars`) from whatever
    # `_worldpop_population_column` resolves to AFTER resampling -- not
    # snapshotted here, pre-census-join/pre-resample, under a second name.
    # `population` itself stays bare WorldPop through the rest of this
    # function (census joins below read/overwrite it per country -- see
    # `_RENAME_EXCLUDED_CANONICAL_NAMES`), and is itself re-derived to its
    # final per-city meaning (census residents, WorldPop residents, or
    # census + jobs) by that same finalize step.

    if config.uses_census:
        h3_grid = _add_h3_grid(pop_access_h3)
        # built floor area per native cell (cadaster, GBA where none): the census counts are spread by it (`_join_polygon_stats`)
        _set_floor_area(config, aoi_gdf, params.h3_resolution, _gba_cache_root(census_dir))
        _lap("building floor areas")
        print(f"[pipeline:{config.key}] joining census attributes for country={config.country!r} ({len(params.census_levels)} levels)")
        # Bug fix (2026-09-05, live report -- hexagonal density artifact at
        # H3 res-5 tile/chunk boundaries): `_join_polygon_stats` normalizes
        # each census polygon's population share using ONLY the h3 cells
        # visible in its current call -- when chunked (the
        # `isochrone_chunk_h3_resolution` knob), a polygon straddling a
        # chunk boundary gets its FULL real population redistributed
        # independently within EACH chunk that sees any of its cells, so
        # its true total is counted once per chunk it touches (confirmed
        # live: a 2-chunk-straddling polygon's population summed to ~2x its
        # real value). This join was never the actual OOM crash site in
        # this study (isochrones and the census-geometry map aggregation
        # were, both separately and correctly chunked) -- always run it
        # unchunked to avoid this real correctness bug.
        h3_grid = _join_polygon_stats_lightweight(
            _join_census,
            h3_grid,
            aoi_gdf,
            config.census_states,
            params.census_levels,
            census_dir,
            country=config.country,
            census_module=config.census_module,
            skip=config.census_skip,
        )
        _lap("census ACS join")
        print(f"[pipeline:{config.key}] joining decennial race/ethnicity attributes")
        if "dhc" not in config.census_skip:
            h3_grid = _join_polygon_stats_lightweight(_join_race, h3_grid, aoi_gdf, config.census_states, census_dir, country=config.country)
        _lap("census DHC race join")
        print(f"[pipeline:{config.key}] joining LODES WAC jobs-by-workplace counts")
        try:
            if "lodes" in config.census_skip:
                raise RuntimeError("skipped (not chosen)")
            h3_grid = _join_polygon_stats_lightweight(_join_jobs, h3_grid, aoi_gdf, config.census_states, census_dir, country=config.country)
        except Exception as exc:  # pragma: no cover - LODES availability varies by state/year
            print(f"[pipeline] skipping LODES jobs join: {exc}")
        _lap("LODES jobs join")
        # Rename every canonical field (see `_rename_canonical_columns`) now
        # that ACS/DHC/LODES (or the equivalent non-US source) have all
        # landed their prefixed columns on `h3_grid`.
        h3_grid = _rename_canonical_columns(h3_grid)
        _lap("canonical column rename")

        if config.exclude_cells_touching_country_border:
            h3_grid = _filter_cells_touching_country_border(
                h3_grid, config.border_country_geocode_name, city_dir / "border_clip"
            )
            _lap("border clip")

        if config.census_worldpop_gapfill:
            gapfill_pop_col = CENSUS_WORLDPOP_GAPFILL_POPULATION_COLUMN.get(config.country)
            if gapfill_pop_col is None:
                print(
                    f"[pipeline:{config.key}] census_worldpop_gapfill is set but country={config.country!r} "
                    "has no entry in CENSUS_WORLDPOP_GAPFILL_POPULATION_COLUMN -- skipping"
                )
            else:
                print(f"[pipeline:{config.key}] WorldPop census gap-fill enabled -- filling uncovered h3 cells")
                h3_grid = _add_worldpop_gapfill(
                    h3_grid, aoi_gdf, params.h3_resolution,
                    _worldpop_year_from_filename(config.worldpop_filename),
                    city_dir / "worldpop_gapfill", gapfill_pop_col,
                )
            _lap("worldpop census gap-fill")
    else:
        # The edge join + `_join_worldpop_global_schema` join themselves
        # (and their checkpointing) already ran above, before this
        # `if config.uses_census:` split -- see the checkpoint block's
        # comment.
        #
        # Shanghai OOM fix (2026-09-01): this used to eagerly call
        # `_add_h3_grid(pop_access_h3)` here, building real hexagon-polygon
        # geometry for the FULL, native-resolution grid (25,732,258 rows for
        # Shanghai) before `_finish_pipeline_stages` -> `build_h3_by_resolution`
        # ever ran -- even though `build_h3_by_resolution` already has its
        # own "preferred, much cheaper" geometry-free Polars fast path
        # (`native_is_polars`, see its docstring), specifically designed to
        # defer building the native resolution's geometry until *after*
        # every coarser map/stats resolution has already been resampled from
        # cheap tabular data. Passing an already-materialized GeoDataFrame
        # here (as `_add_h3_grid` would produce) skips that fast path
        # entirely and forces every downstream resample to carry the full
        # geometry + a pandas copy of every attribute column along with it --
        # confirmed live to still OOM-kill Shanghai's run at this exact next
        # step even after `_add_h3_grid`/`to_gdf` itself were made
        # memory-bounded. Passing the lightweight Polars table straight
        # through instead (nothing between here and `_finish_pipeline_stages`
        # needs real geometry -- verified by reading the code in between)
        # lets `build_h3_by_resolution` take its own cheap path exactly as
        # designed. Still need `_add_h3_columns_polars` (the polars mirror
        # of `_add_h3_grid`'s NON-geometry work -- `area_m2`/`pop_density`)
        # though: `build_h3_by_resolution`'s native-resolution branch uses
        # its Polars input as-is with no further column derivation (every
        # OTHER resolution gets this same call already, via its own
        # resample step -- only the native resolution was implicitly
        # assuming it had already been run, which was true for existing
        # Polars-input callers like `refresh_census_only` but not for this
        # new one). Missing this raised `ColumnNotFoundError: "pop_density"`
        # live on the very next run after the geometry-deferral fix above.
        h3_grid = _add_h3_columns_polars(pop_access_h3)

    if config.uses_census:
        # Additive real WorldPop demographic layers for every census city --
        # see `_add_worldpop_demographic_layers`. Not run for Shanghai
        # (the `else` branch above): it already gets these exact features
        # under bare names via `_join_worldpop_global_schema`.
        if os.environ.get("CS_TRANSITLOS_SKIP_WORLDPOP") == "1" or not config.worldpop_layers:
            print(f"[pipeline:{config.key}] skipping WorldPop demographic layers")
        else:
            # Restricted to real census-covered cells for every country EXCEPT
            # Beersheba (`config.census_worldpop_gapfill`) -- see
            # `_add_worldpop_demographic_layers`'s `restrict_to_population_column`
            # docstring for why Beersheba is the one deliberate exception.
            # 2026-11-10 (explicit user request: "I want all WorldPop data"): the WorldPop demographic layers fill EVERY cell WorldPop covers, census or not
            demographic_restrict_col = None
            print(
                f"[pipeline:{config.key}] joining WorldPop demographic layers (worldpop_* columns), "
                f"restricted to real census coverage via {demographic_restrict_col!r}"
                if demographic_restrict_col is not None
                else f"[pipeline:{config.key}] joining WorldPop demographic layers (worldpop_* columns) onto full grid"
            )
            h3_grid = _add_worldpop_demographic_layers(
                h3_grid, aoi_gdf, params.h3_resolution,
                _worldpop_year_from_filename(config.worldpop_filename),
                city_dir / "worldpop_demographic",
                restrict_to_population_column=demographic_restrict_col,
            )
            _lap("worldpop demographic layers")

    # Resample -> core/metro split -> stats -> map: shared with the fast
    # re-entry points (`refresh_census_only`) via `_finish_pipeline_stages`,
    # so a full run and a census-refresh run can never silently diverge on
    # this tail.
    _finish_pipeline_stages(
        city_dir, config, params, census_dir, aoi_gdf, h3_grid, stops, access_gdf,
        tile_workers=tile_workers, run_stats=True, _lap=_lap, build_map=build_map,
    )


# ALL the local census columns of every source (user, 2026-10): each KEEP set also takes every feature of its pycensus schema.
# Updated in place so structures that captured the sets earlier see the additions.
for _keep, _args in (
    (DHC_KEEP_COLUMNS, ("usa", "dhc", "dhc_")),
    (INEGI_KEEP_COLUMNS, ("mexico", "inegi", "inegi_")),
    (EUSKADI_KEEP_COLUMNS, ("euskadi", "eustat", "eustat_")),
    (EUSTAT_MUNICIPAL_KEEP_COLUMNS, ("euskadi", "eustat", "eustat_")),
    (EUSTAT_MUNICIPAL_KEEP_COLUMNS, ("euskadi", "eustat_municipal", "eustat_")),
    (SPAIN_KEEP_COLUMNS, ("spain", "censo2021", "censo2021_")),
    (SPAIN_KEEP_COLUMNS, ("spain", "ine", "ine_")),
    (SPAIN_KEEP_COLUMNS, ("spain", "adrh", "ine_")),
    (CANADA_KEEP_COLUMNS, ("canada", "statcan", "statcan_")),
    (ANDORRA_KEEP_COLUMNS, ("andorra", "estadisticaad", "estadisticaad_")),
    (TAIWAN_KEEP_COLUMNS, ("taiwan", "moi", "moi_")),
    (CHILE_KEEP_COLUMNS, ("chile", "ine", "ine_cl_")),
    (CHILE_CASEN_KEEP_COLUMNS, ("chile", "casen", "casen_cl_")),
    (ISRAEL_COMMUTE_KEEP_COLUMNS, ("israel", "cbs_commute", "cbscommute_")),
    (IMEPLAN_EOD_KEEP_COLUMNS, ("mexico", "imeplan_eod", "imepeod_")),
    (DONOSTIA_PMUS_KEEP_COLUMNS, ("spain", "donostia_pmus", "donpmus_")),
    (TAIPEI_MODE_USE_KEEP_COLUMNS, ("taiwan", "taipei_mode_use", "tpemode_")),
    (MID_HAMBURG_KEEP_COLUMNS, ("germany", "mid_hamburg", "midhh_")),
    (EM_CAPV_KEEP_COLUMNS, ("spain", "em_capv", "emcapv_")),
    (GERMANY_KEEP_COLUMNS, ("germany", "destatis", "destatis_")),
    (GERMANY_BA_PENDLER_KEEP_COLUMNS, ("germany", "ba_pendler", "ba_pendler_")),
    (TORONTO_PROFILE_KEEP_COLUMNS, ("canada", "toronto", "statcan_")),
    (US_RELIGION_KEEP_COLUMNS, ("usa", "religion", "usrc_")),
    (US_HOUSING_KEEP_COLUMNS, ("usa", "housing", "ushs_")),
    (CHILE_MANZANA_KEEP_COLUMNS, ("chile", "ine_manzana", "ine_cl_")),
):
    _keep.update(_schema_columns(*_args))
    _register_schema_rates(*_args)


# `cbs_population` is renamed to `census_residents` (and `population`) once the grid is finalized, so the Israeli religion / religiosity
# shares also fall back to those denominators (they stayed absent from the maps otherwise).
for _share, _cands in SHARE_COLUMNS.items():
    if _share.startswith("cbs_") and _cands and _cands[0][1] == "cbs_population":
        _cands.extend([(_cands[0][0], "census_residents"), (_cands[0][0], "population")])
