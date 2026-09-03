"""Tests for `code.pipeline._run_regressions_and_anovas`'s country-generic
equity/income/transit variable lookup (`EQUITY_VAR_CANDIDATES`/
`TRANSIT_VAR_CANDIDATES`/`INCOME_VAR_CANDIDATES`), added after the user
flagged that every non-US city's stats panel only ever showed the
population-density ANOVA (the old `US_EQUITY_VARS`/`US_TRANSIT_VAR`/
`US_INCOME_VAR` were hardcoded to USA-only ACS5 column names). Verifies:
USA's behavior is unchanged, and a Germany-like / Canada-like synthetic grid
(with real per-country column names, pre-`_add_share_columns`) now produces
more than just the population-density ANOVA.
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from code.pipeline import _add_share_columns, _run_regressions_and_anovas


def _square(i):
    return Polygon([(i * 2000, 0), (i * 2000 + 1000, 0), (i * 2000 + 1000, 1000), (i * 2000, 1000)])


def _base_grid(n=8):
    rng = np.random.default_rng(0)
    geoms = [_square(i) for i in range(n)]
    population = rng.uniform(100, 1000, n)
    level_of_service = rng.uniform(0, 1, n)
    return gpd.GeoDataFrame(
        {
            "h3_cell": [f"cell{i}" for i in range(n)],
            "population": population,
            "pop_density": population / 1.0,
            "level_of_service": level_of_service,
            "area_m2": [1_000_000.0] * n,
            "geometry": geoms,
        },
        crs="EPSG:32619",
    )


def test_usa_grid_still_produces_all_four_legacy_equity_vars_plus_income_and_transit(tmp_path):
    grid = _base_grid()
    n = len(grid)
    rng = np.random.default_rng(1)
    grid["households"] = rng.uniform(50, 500, n)
    grid["povertyPopulation"] = rng.uniform(50, 500, n)
    grid["acs5_vehiclesHouseholds0"] = rng.uniform(0, grid["households"])
    grid["acs5_householdsRenter"] = rng.uniform(0, grid["households"])
    grid["acs5_povertyBelow100"] = rng.uniform(0, grid["povertyPopulation"])
    grid["acs5_rentBurdened"] = rng.uniform(0, 100, n)
    grid["acs5_rentBurdenPopulation"] = rng.uniform(100, 200, n)
    grid["acs5_workers"] = rng.uniform(50, 500, n)
    grid["publicTransportCommuters"] = rng.uniform(0, grid["acs5_workers"])
    grid["medianHouseholdIncome"] = rng.uniform(30000, 120000, n)

    grid = _add_share_columns(grid)

    summary = _run_regressions_and_anovas(grid, True, tmp_path, "Test USA City", "metro")

    for key in (
        "anova_density",
        "anova_car_free_share",
        "anova_renter_share",
        "anova_poverty_share",
        "anova_rent_burden_share",
        "anova_income",
        "anova_transit_share",
    ):
        assert key in summary, f"missing {key} in USA summary"

    # No non-US-only equity vars should spuriously appear.
    assert "anova_foreign_born_share" not in summary
    assert "anova_owner_share" not in summary


def test_germany_like_grid_gets_foreign_born_share_anova_beyond_population_density(tmp_path):
    grid = _base_grid()
    n = len(grid)
    rng = np.random.default_rng(2)
    grid["destatis_population"] = grid["population"]
    grid["foreignBornPopulation"] = rng.uniform(0, grid["destatis_population"])

    grid = _add_share_columns(grid)
    assert "foreign_born_share" in grid.columns

    summary = _run_regressions_and_anovas(grid, True, tmp_path, "Test Germany City", "metro")

    assert "anova_density" in summary
    assert "anova_foreign_born_share" in summary
    # Confirms this isn't just population-density: at least one *other*
    # ANOVA/regression summary key beyond `anova_density`/`r2_density`.
    extra_keys = set(summary) - {"anova_density", "r2_density"}
    assert extra_keys, "Germany-like grid should get more than just population-density"

    # No USA-only or Canada-only vars should spuriously appear.
    assert "anova_car_free_share" not in summary
    assert "anova_owner_share" not in summary
    assert "anova_income" not in summary


def test_canada_like_grid_gets_income_owner_and_transit_share(tmp_path):
    grid = _base_grid()
    n = len(grid)
    rng = np.random.default_rng(3)
    grid["statcan_population"] = grid["population"]
    grid["statcan_households"] = rng.uniform(50, 500, n)
    grid["ownerHouseholds"] = rng.uniform(0, grid["statcan_households"])
    grid["renterHouseholds"] = grid["statcan_households"] - grid["ownerHouseholds"]
    grid["statcan_commuteTotalPopulation"] = rng.uniform(50, 500, n)
    grid["publicTransportCommuters"] = rng.uniform(0, grid["statcan_commuteTotalPopulation"])
    grid["medianHouseholdIncome"] = rng.uniform(30000, 150000, n)
    grid["foreignBornPopulation"] = rng.uniform(0, grid["statcan_population"])

    grid = _add_share_columns(grid)
    for col in ("statcan_owner_rate", "statcan_renter_rate", "transit_share", "foreign_born_share"):
        assert col in grid.columns, f"expected {col} to be materialized"

    summary = _run_regressions_and_anovas(grid, True, tmp_path, "Test Canada City", "metro")

    for key in (
        "anova_density",
        "anova_owner_share",
        "anova_renter_share",
        "anova_income",
        "anova_transit_share",
        "anova_foreign_born_share",
        "r2_transit_share",
    ):
        assert key in summary, f"missing {key} in Canada summary"


def test_no_census_data_only_produces_population_density(tmp_path):
    grid = _base_grid()
    summary = _run_regressions_and_anovas(grid, False, tmp_path, "Test No-Census City", "metro")
    assert set(summary) == {"anova_density", "r2_density"}
