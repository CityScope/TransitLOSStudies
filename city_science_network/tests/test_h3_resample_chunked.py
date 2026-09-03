"""Correctness test: `_resample_h3_chunked` must match unchunked `_resample_h3`.

Part of the H3-chunked pipeline (see `docs/H3_CHUNKED_PIPELINE_DESIGN.md`):
`_resample_h3_chunked` partitions the input grid by each row's H3 res-4
ancestor before resampling, to bound peak memory on oversized AOIs
(Boston/Shanghai). Unlike the isochrone chunking, this is a pure group-by
aggregation with no buffer/overlap concept, so chunked and unchunked results
must be *exactly* equal (row order aside), not just close.
"""

import h3
import numpy as np
import polars as pl

from code.pipeline import _resample_h3, _resample_h3_chunked


def _real_cells_across_multiple_res4_parents(n_per_parent=6, n_parents=3, base_res=9):
    """Pick real H3 cells spread across several distinct res-4 parent cells.

    Starts from a real res-4 cell (San Francisco-ish) and walks to
    neighboring res-4 cells via `grid_disk` so the test actually exercises
    multiple chunks, then, for each, takes several of its real res-`base_res`
    descendants (`cell_to_children` walked down one level at a time).
    """
    start_res4 = h3.latlng_to_cell(37.7749, -122.4194, 4)
    res4_cells = list(h3.grid_disk(start_res4, 2))[:n_parents]

    cells = []
    for parent in res4_cells:
        frontier = [parent]
        res = 4
        while res < base_res:
            frontier = [c for cell in frontier for c in h3.cell_to_children(cell, res + 1)]
            res += 1
        cells.extend(frontier[:n_per_parent])
    return cells


def test_resample_h3_chunked_matches_unchunked():
    cells = _real_cells_across_multiple_res4_parents()
    rng = np.random.default_rng(0)
    n = len(cells)
    df = pl.DataFrame(
        {
            "h3_cell": cells,
            "population": rng.uniform(0, 500, n).round(1),
            "level_of_service": rng.uniform(0, 1, n).round(3),
            "acs5_income_median_household": rng.choice([np.nan, 40000.0, 60000.0, 80000.0], n),
            "acs5_households": rng.integers(0, 200, n).astype(float),
        }
    )
    sum_cols = ["acs5_income_median_household", "acs5_households"]

    target_res = 7  # coarser than base_res=9, finer than chunk_h3_resolution=4
    unchunked = _resample_h3(df, target_resolution=target_res, sum_cols=sum_cols)
    chunked = _resample_h3_chunked(df, target_resolution=target_res, sum_cols=sum_cols, chunk_h3_resolution=4)

    assert set(unchunked["h3_cell"].to_list()) == set(chunked["h3_cell"].to_list())

    a = unchunked.sort("h3_cell")
    b = chunked.sort("h3_cell").select(unchunked.columns)
    np.testing.assert_allclose(a["population"].to_numpy(), b["population"].to_numpy())
    np.testing.assert_allclose(a["level_of_service"].to_numpy(), b["level_of_service"].to_numpy())
    np.testing.assert_allclose(a["acs5_households"].to_numpy(), b["acs5_households"].to_numpy())
    a_income = a["acs5_income_median_household"].to_numpy()
    b_income = b["acs5_income_median_household"].to_numpy()
    np.testing.assert_allclose(a_income, b_income, equal_nan=True)
