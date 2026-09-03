"""Tests for the cross-city ANOVA/regression relative columns
(`code.pipeline.RELATIVE_STATS_BASE_COLUMNS`/`_relative_column_stats_for_grid`),
added per the user's ask that ANOVA/regression comparison columns be
`.share`/`.density` variants of population / jobs / population+jobs. See the
design-decision comment above `RELATIVE_STATS_BASE_COLUMNS` in
`code/pipeline.py` for what those mean at this cross-city, per-city-summary
level (population-weighted-averaged within-city relative value, not a
literal cross-city total).
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from code.pipeline import (
    RELATIVE_STATS_BASE_COLUMNS,
    _relative_column_stats_for_grid,
    write_city_summary,
)


def _square(x0, y0, side=0.01):
    return Polygon([(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)])


def _make_grid(n=4, with_jobs=True):
    """A small synthetic h3-like grid: n equal-size square cells in a projected CRS.

    Uses a local UTM-like projected CRS (EPSG:32619, meters) directly so
    `geohierarchy.utils.area` doesn't need to reproject anything -- each
    cell is a 1000m x 1000m square, area = 1e6 m2 = 1 km2 exactly, which
    makes `.density` values trivial to hand-check (value / 1 km2 == value).
    """
    geoms = [Polygon([(i * 2000, 0), (i * 2000 + 1000, 0), (i * 2000 + 1000, 1000), (i * 2000, 1000)]) for i in range(n)]
    data = {
        "population": [100.0, 200.0, 300.0, 400.0][:n],
        # `_area_km2` (reused here for `.density`, same helper `pop_jobs_density`
        # already uses) reads `area_m2` directly rather than recomputing from
        # geometry -- each square is exactly 1000m x 1000m = 1e6 m2 = 1 km2.
        "area_m2": [1_000_000.0] * n,
        "geometry": geoms,
    }
    if with_jobs:
        data["lodes_wac_jobs_total"] = [10.0, 20.0, 30.0, 40.0][:n]
        data["pop_jobs_total"] = [110.0, 220.0, 330.0, 440.0][:n]
    gdf = gpd.GeoDataFrame(data, crs="EPSG:32619")
    return gdf


def test_relative_stats_base_columns_are_population_jobs_and_combined():
    keys = [k for k, _label in RELATIVE_STATS_BASE_COLUMNS]
    assert keys == ["population", "lodes_wac_jobs_total", "pop_jobs_total"]


def test_density_is_value_over_cell_area_km2():
    grid = _make_grid(with_jobs=False)
    population = grid["population"].to_numpy(dtype=float)
    out = _relative_column_stats_for_grid(grid, population)
    # Each cell is exactly 1 km2 (1000m x 1000m), so density == raw value,
    # population-weighted mean of [100,200,300,400] weighted by itself:
    # sum(w*w)/sum(w) = (100^2+200^2+300^2+400^2)/(100+200+300+400)
    expected = (100**2 + 200**2 + 300**2 + 400**2) / (100 + 200 + 300 + 400)
    assert out["population.density"] == pytest_approx(expected)


def test_share_is_value_over_grid_total():
    grid = _make_grid(with_jobs=False)
    population = grid["population"].to_numpy(dtype=float)
    out = _relative_column_stats_for_grid(grid, population)
    total = 100.0 + 200.0 + 300.0 + 400.0
    shares = np.array([100, 200, 300, 400]) / total
    expected = float(np.average(shares, weights=population))
    assert out["population.share"] == pytest_approx(expected)


def test_missing_jobs_column_gracefully_omitted():
    grid = _make_grid(with_jobs=False)
    population = grid["population"].to_numpy(dtype=float)
    out = _relative_column_stats_for_grid(grid, population)
    assert "lodes_wac_jobs_total.share" not in out
    assert "lodes_wac_jobs_total.density" not in out
    assert "pop_jobs_total.share" not in out
    # population is present, so its keys must exist.
    assert "population.share" in out
    assert "population.density" in out


def test_all_three_base_columns_present_when_jobs_data_exists():
    grid = _make_grid(with_jobs=True)
    population = grid["population"].to_numpy(dtype=float)
    out = _relative_column_stats_for_grid(grid, population)
    for col, _label in RELATIVE_STATS_BASE_COLUMNS:
        assert f"{col}.share" in out, f"missing {col}.share"
        assert f"{col}.density" in out, f"missing {col}.density"


def test_write_city_summary_merges_relative_columns_into_column_stats(tmp_path):
    grid = _make_grid(with_jobs=True)
    population = grid["population"].to_numpy(dtype=float)
    access = np.array([0.5, 0.6, 0.7, 0.8])
    summary = write_city_summary(
        tmp_path,
        "Test City",
        metro_access=access,
        metro_population=population,
        core_access=access[:2],
        core_population=population[:2],
        metro_grid=grid,
        core_grid=grid.iloc[:2],
    )
    metro_stats = summary["column_stats"]["metro"]
    # Plain (absolute) columns from the existing STATS_OVERVIEW_COLUMNS logic
    # still present (no regression on the previously-built feature).
    assert "population" in metro_stats
    assert "pop_jobs_total" in metro_stats
    # New relative columns present alongside them.
    assert "population.share" in metro_stats
    assert "population.density" in metro_stats
    assert "lodes_wac_jobs_total.share" in metro_stats
    assert "pop_jobs_total.density" in metro_stats

    out_path = tmp_path / "results" / "summary.json"
    assert out_path.exists()


def pytest_approx(x):
    import pytest

    return pytest.approx(x, rel=1e-6)
