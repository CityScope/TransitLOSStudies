"""Regression tests for `pipeline._interpolate_acs_to_dhc_blocks`.

Population-weighted areal interpolation of ACS5's blockgroup-only count
fields down onto real DHC blocks: each block should get a share of its
blockgroup's ACS count proportional to the block's own real `dhc_population`
share of the blockgroup's total DHC population.
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from code.pipeline import ACS_COUNT_FIELDS_FOR_INTERPOLATION, _interpolate_acs_to_dhc_blocks


def _make_blockgroup(geoid, x0, laborforce, x1=None, x1_offset=1.0):
    x1 = x0 + x1_offset if x1 is None else x1
    return {"GEOID": geoid, "geometry": box(x0, 0, x1, 1), "acs5_laborForce": laborforce}


def test_blockgroup_count_apportioned_by_real_block_population_share():
    # One blockgroup covering [0, 1] x [0, 1], ACS5 laborForce = 100.
    acs_gdf = gpd.GeoDataFrame(
        [_make_blockgroup("BG1", 0.0, 100)],
        crs=4326,
    )

    # Two real DHC blocks splitting that blockgroup 3:1 by population.
    block_gdf = gpd.GeoDataFrame(
        {
            "GEOID": ["B1", "B2"],
            "dhc_population": [30, 10],
        },
        geometry=[box(0.0, 0.0, 0.5, 1.0), box(0.5, 0.0, 1.0, 1.0)],
        crs=4326,
    )

    out = _interpolate_acs_to_dhc_blocks(acs_gdf, block_gdf, {"acs5_laborForce"})

    assert set(out.columns) == {"GEOID", "geometry", "acs5_laborForce_interpolated"}
    values = dict(zip(out["GEOID"], out["acs5_laborForce_interpolated"]))
    assert values["B1"] == pytest_approx(75.0)
    assert values["B2"] == pytest_approx(25.0)
    # Interpolated total must equal the real ACS5 blockgroup total exactly
    # (population-weighted apportionment never creates or destroys count).
    assert values["B1"] + values["B2"] == pytest_approx(100.0)


def test_zero_population_blockgroup_splits_equally():
    acs_gdf = gpd.GeoDataFrame([_make_blockgroup("BG1", 0.0, 40)], crs=4326)
    block_gdf = gpd.GeoDataFrame(
        {"GEOID": ["B1", "B2"], "dhc_population": [0, 0]},
        geometry=[box(0.0, 0.0, 0.5, 1.0), box(0.5, 0.0, 1.0, 1.0)],
        crs=4326,
    )
    out = _interpolate_acs_to_dhc_blocks(acs_gdf, block_gdf, {"acs5_laborForce"})
    values = dict(zip(out["GEOID"], out["acs5_laborForce_interpolated"]))
    assert values["B1"] == pytest_approx(20.0)
    assert values["B2"] == pytest_approx(20.0)


def test_block_outside_every_blockgroup_gets_nan():
    acs_gdf = gpd.GeoDataFrame([_make_blockgroup("BG1", 0.0, 40)], crs=4326)
    block_gdf = gpd.GeoDataFrame(
        {"GEOID": ["B1", "FAR"], "dhc_population": [10, 5]},
        geometry=[box(0.0, 0.0, 1.0, 1.0), box(100.0, 100.0, 101.0, 101.0)],
        crs=4326,
    )
    out = _interpolate_acs_to_dhc_blocks(acs_gdf, block_gdf, {"acs5_laborForce"})
    values = dict(zip(out["GEOID"], out["acs5_laborForce_interpolated"]))
    assert values["B1"] == pytest_approx(40.0)
    assert np.isnan(values["FAR"])


def test_only_count_fields_are_interpolated_never_rate_fields():
    """`ACS_COUNT_FIELDS_FOR_INTERPOLATION` must not contain any rate/mean/
    median field -- interpolating those would be a meaningless constant
    broadcast, and `pipeline._join_census`'s USA branch relies on this set
    being count-only so it never produces a misleading `*Mean_interpolated`/
    `*Median_interpolated` column.
    """
    rate_substrings = ("mean", "median", "rate", "ratio", "share", "density")
    for field in ACS_COUNT_FIELDS_FOR_INTERPOLATION:
        assert not any(s in field.lower() for s in rate_substrings), field


def pytest_approx(x):
    import pytest

    return pytest.approx(x)
