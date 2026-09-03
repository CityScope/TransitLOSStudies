"""`_finish_pipeline_stages` must dispatch to the chunked h3-by-resolution path
only when `StudyParams.chunked_h3_output` is set, and never change behavior
for the (default) unchunked case.

`_finish_pipeline_stages` itself is a large integration function (core/metro
split, stats, map building, file writes) that isn't worth fully mocking here.
Instead this pins down exactly the two things step 3 of the chunked-h3 wiring
needs: (1) the dispatch itself picks the right function based on the flag,
and (2) `_concat_gdf_parquets` (the shim that lets the chunked path's tile
files feed the same downstream code the unchunked path already does)
reassembles tile files into a GeoDataFrame identical to the unchunked one.
"""

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import polars as pl
import pytest
import shapely

import code.pipeline as P
from code.params import StudyParams


class _Abort(Exception):
    """Raised right after the h3-by-resolution dispatch to short-circuit the
    rest of `_finish_pipeline_stages` (core boundary resolution, stats, map
    building) -- none of that is what this test is checking."""


def _synthetic_grid(n=40):
    start = h3.latlng_to_cell(37.7749, -122.4194, 9)
    cells = list(h3.grid_disk(start, 3))[:n]
    rng = np.random.default_rng(0)
    return pl.DataFrame({
        "h3_cell": cells,
        "population": rng.uniform(0, 500, len(cells)).round(1),
        "level_of_service": rng.uniform(0, 1, len(cells)).round(3),
        "jobs": rng.uniform(0, 300, len(cells)).round(1),
    })


@pytest.mark.parametrize("chunked", [False, True])
def test_finish_pipeline_stages_dispatches_on_chunked_h3_output_flag(monkeypatch, tmp_path, chunked):
    calls = {"chunked": False, "unchunked": False}

    real_chunked = P.build_h3_by_resolution_chunked
    real_unchunked = P.build_h3_by_resolution

    def spy_chunked(*args, **kwargs):
        calls["chunked"] = True
        return real_chunked(*args, **kwargs)

    def spy_unchunked(*args, **kwargs):
        calls["unchunked"] = True
        return real_unchunked(*args, **kwargs)

    monkeypatch.setattr(P, "build_h3_by_resolution_chunked", spy_chunked)
    monkeypatch.setattr(P, "build_h3_by_resolution", spy_unchunked)

    def abort(*a, **k):
        raise _Abort()

    monkeypatch.setattr(P, "resolve_core_boundary", abort)

    df = _synthetic_grid()
    params = StudyParams(
        h3_resolution=9, stats_h3_resolution=7, map_h3_resolutions=(7, 9),
        chunked_h3_output=4 if chunked else None,
    )

    class FakeConfig:
        key = "testcity"
        uses_census = False
        display_name = "Test City"
        country = "USA"
        census_states = None
        geocode_name = None

    (tmp_path / "results" / "metro").mkdir(parents=True)

    with pytest.raises(_Abort):
        P._finish_pipeline_stages(
            city_dir=tmp_path,
            config=FakeConfig(),
            params=params,
            census_dir=tmp_path,
            aoi_gdf=None,
            h3_grid=df,
            stops=None,
            access_gdf=None,
        )

    assert calls["chunked"] is chunked
    assert calls["unchunked"] is (not chunked)


def test_concat_gdf_parquets_matches_source(tmp_path):
    cells = list(h3.grid_disk(h3.latlng_to_cell(37.7749, -122.4194, 9), 2))
    rng = np.random.default_rng(1)
    n = len(cells)
    df = pd.DataFrame({
        "h3_cell": cells,
        "population": rng.uniform(0, 500, n).round(1),
    })
    geometry = [shapely.Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in cells]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    half = n // 2
    p1, p2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    gdf.iloc[:half].to_parquet(p1)
    gdf.iloc[half:].to_parquet(p2)

    result = P._concat_gdf_parquets([str(p1), str(p2)])
    result_sorted = result.sort_values("h3_cell").reset_index(drop=True)
    expected_sorted = gdf.sort_values("h3_cell").reset_index(drop=True)

    assert result_sorted["h3_cell"].tolist() == expected_sorted["h3_cell"].tolist()
    np.testing.assert_allclose(
        result_sorted["population"].to_numpy(), expected_sorted["population"].to_numpy()
    )
    assert result.crs == gdf.crs
