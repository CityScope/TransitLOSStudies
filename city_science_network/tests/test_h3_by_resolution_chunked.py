"""Correctness test: `build_h3_by_resolution_chunked` must match unchunked `build_h3_by_resolution`.

The chunked path exists so a huge city (Shanghai) never has to materialize
one full-city GeoDataFrame per map/stats resolution (see
`build_h3_by_resolution_chunked`'s own docstring). Like
`test_h3_resample_chunked.py`'s `_resample_h3_chunked` check, this is a pure
partition + independent per-tile aggregation with no buffer/overlap concept,
so the two paths' outputs must be *exactly* equal cell-for-cell (row order
and file-vs-in-memory representation aside), not just close.
"""

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import polars as pl
import pytest

from code.params import StudyParams
from code.pipeline import build_h3_by_resolution, build_h3_by_resolution_chunked


def _real_cells_across_multiple_tile_parents(tile_res=4, n_per_parent=8, n_parents=3, base_res=9):
    """Real H3 cells spread across several distinct `tile_res`-resolution parent cells."""
    start_tile = h3.latlng_to_cell(37.7749, -122.4194, tile_res)
    tile_cells = list(h3.grid_disk(start_tile, 2))[:n_parents]

    cells = []
    for parent in tile_cells:
        frontier = [parent]
        res = tile_res
        while res < base_res:
            frontier = [c for cell in frontier for c in h3.cell_to_children(cell, res + 1)]
            res += 1
        cells.extend(frontier[:n_per_parent])
    return cells


def _synthetic_grid(cells):
    rng = np.random.default_rng(0)
    n = len(cells)
    return pl.DataFrame(
        {
            "h3_cell": cells,
            "population": rng.uniform(0, 500, n).round(1),
            "level_of_service": rng.uniform(0, 1, n).round(3),
            "jobs": rng.uniform(0, 300, n).round(1),
        }
    )


def _sorted_frame(gdf: gpd.GeoDataFrame, columns) -> gpd.GeoDataFrame:
    return gdf.sort_values("h3_cell").reset_index(drop=True)[list(columns)]


@pytest.mark.parametrize("max_workers", [1, 2])
def test_chunked_matches_unchunked_for_every_resolution(tmp_path, max_workers):
    base_res = 9
    tile_res = 4
    cells = _real_cells_across_multiple_tile_parents(tile_res=tile_res, base_res=base_res)
    df = _synthetic_grid(cells)

    params = StudyParams(
        h3_resolution=base_res,
        stats_h3_resolution=7,
        map_h3_resolutions=(4, 7, 9),
        chunked_h3_output=tile_res,
    )

    unchunked = build_h3_by_resolution(df, params, uses_census=False)
    chunked_paths = build_h3_by_resolution_chunked(
        df, params, uses_census=False, output_dir=str(tmp_path / "chunks"), max_workers=max_workers
    )

    resolutions = list(dict.fromkeys([params.h3_resolution, params.stats_h3_resolution, *params.map_h3_resolutions]))
    assert set(chunked_paths.keys()) == set(resolutions)

    for res in resolutions:
        expected = unchunked[res]
        assert chunked_paths[res], f"no tile files written for resolution {res}"
        actual = pd_concat_gdf(chunked_paths[res])

        common_cols = [c for c in expected.columns if c != "geometry"]
        a = _sorted_frame(expected, common_cols)
        b = _sorted_frame(actual, common_cols)

        assert a["h3_cell"].tolist() == b["h3_cell"].tolist(), f"resolution {res}: cell set mismatch"
        for col in common_cols:
            if col == "h3_cell":
                continue
            if not pd.api.types.is_numeric_dtype(a[col]):
                a_vals = [None if pd.isna(v) else v for v in a[col].tolist()]
                b_vals = [None if pd.isna(v) else v for v in b[col].tolist()]
                assert a_vals == b_vals, f"resolution {res}, column {col}"
            else:
                np.testing.assert_allclose(
                    a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float),
                    equal_nan=True, err_msg=f"resolution {res}, column {col}",
                )

        # Geometry must also agree cell-for-cell (same hexagon each cell).
        expected_sorted = expected.sort_values("h3_cell").reset_index(drop=True)
        actual_sorted = actual.sort_values("h3_cell").reset_index(drop=True)
        assert all(
            expected_sorted.geometry.iloc[i].equals_exact(actual_sorted.geometry.iloc[i], tolerance=1e-9)
            for i in range(len(expected_sorted))
        ), f"resolution {res}: geometry mismatch"


def pd_concat_gdf(paths):
    import pandas as pd

    frames = [gpd.read_parquet(p) for p in paths]
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def test_rejects_geodataframe_input(tmp_path):
    params = StudyParams(chunked_h3_output=4)
    with pytest.raises(TypeError):
        build_h3_by_resolution_chunked(
            gpd.GeoDataFrame({"h3_cell": []}), params, uses_census=False, output_dir=str(tmp_path)
        )


def test_rejects_tile_resolution_finer_than_a_target_resolution(tmp_path):
    df = _synthetic_grid(_real_cells_across_multiple_tile_parents(tile_res=4, base_res=9))
    params = StudyParams(
        h3_resolution=9, stats_h3_resolution=7, map_h3_resolutions=(7, 9),
        chunked_h3_output=8,  # finer than stats_h3_resolution=7 -- must be rejected
    )
    with pytest.raises(ValueError):
        build_h3_by_resolution_chunked(df, params, uses_census=False, output_dir=str(tmp_path))
