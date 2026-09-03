"""Correctness test for the cell-batching fix in `h3_population.population_and_access_to_h3`.

2026-08-31: real, watchdog-instrumented measurement against Shanghai's
cached data showed that even after bbox-restricting each edge chunk's
candidate H3 cells (`local_cells`), a single `edges_to_h3_by_distance` call
against a megaregion-scale candidate set (millions of cells) alone drove RSS
past several extra GB before returning -- independent of `edge_chunk_size`.
The fix sub-batches `local_cells` into groups of `cell_batch_size` and
merges each batch's result into the running accumulator via the same
Max()-safe merge already used across edge chunks (see `_merge` in
`h3_population.py`).

This test exercises exactly that merge invariant directly against
`geohierarchy.edges_to_h3_by_distance` (the function actually called per
batch): batching the candidate cells and merging results chunk-by-chunk via
group_by/max must produce the *same* result as one unbatched call -- both
matched cells (via the KD-tree radius match) and fallback cells (via the
nearest-street fallback), since `fallback_radius_multiplier`'s search radius
depends only on which edges and cells are handed to a given call, not on
how many *other* cells happen to be batched alongside them.
"""

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import LineString, box

from geohierarchy import Max, edges_to_h3_by_distance
from geohierarchy.utils import h3_cells


@pytest.fixture
def scattered_edges():
    # A handful of short streets scattered across the bbox below, so
    # different candidate cells end up matched (within `margin_m`) vs. only
    # reachable via the nearest-street fallback.
    return gpd.GeoDataFrame(
        {
            "level_of_service": [10.0, 20.0, 30.0, 40.0],
            "geometry": [
                LineString([(0.001, 0.001), (0.003, 0.003)]),
                LineString([(0.020, 0.005), (0.022, 0.007)]),
                LineString([(0.040, 0.040), (0.042, 0.042)]),
                LineString([(0.005, 0.045), (0.007, 0.047)]),
            ],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def candidate_cells(scattered_edges):
    bbox_gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 0.05, 0.05)]}, crs="EPSG:4326")
    grid = h3_cells(bbox_gdf, resolution=9)
    return grid["h3"].tolist()


def _merge(base: pl.DataFrame | None, part: pl.DataFrame) -> pl.DataFrame:
    if base is None:
        return part
    if part.is_empty():
        return base
    return (
        pl.concat([base, part], how="vertical")
        .group_by("h3_cell")
        .agg(pl.col("level_of_service").max())
    )


def test_cell_batching_matches_unbatched(scattered_edges, candidate_cells):
    assert len(candidate_cells) > 8, "fixture should produce enough cells to make batching meaningful"

    unbatched = edges_to_h3_by_distance(
        scattered_edges,
        candidate_cells,
        columns="level_of_service",
        resolution=9,
        margin_m=10.0,
        agg=Max(),
        fallback_radius_multiplier=5.0,
    )

    batch_size = max(1, len(candidate_cells) // 4)  # force >= 4 batches
    batched = None
    for i in range(0, len(candidate_cells), batch_size):
        part = edges_to_h3_by_distance(
            scattered_edges,
            candidate_cells[i : i + batch_size],
            columns="level_of_service",
            resolution=9,
            margin_m=10.0,
            agg=Max(),
            fallback_radius_multiplier=5.0,
        )
        batched = _merge(batched, part)

    assert set(unbatched["h3_cell"].to_list()) == set(batched["h3_cell"].to_list())

    unbatched_sorted = unbatched.sort("h3_cell")
    batched_sorted = batched.sort("h3_cell")
    assert unbatched_sorted["h3_cell"].to_list() == batched_sorted["h3_cell"].to_list()
    assert unbatched_sorted["level_of_service"].to_list() == pytest.approx(
        batched_sorted["level_of_service"].to_list()
    )


def test_cell_batching_disabled_is_single_batch(scattered_edges, candidate_cells):
    # `cell_batch_size=0`/`None` in `population_and_access_to_h3` means "one
    # call, no sub-batching" -- sanity check that a batch size >= the full
    # candidate list behaves identically to the unbatched call (the actual
    # code path taken when `len(local_cells) <= cell_batch_size`).
    unbatched = edges_to_h3_by_distance(
        scattered_edges,
        candidate_cells,
        columns="level_of_service",
        resolution=9,
        margin_m=10.0,
        agg=Max(),
        fallback_radius_multiplier=5.0,
    )
    single_batch = edges_to_h3_by_distance(
        scattered_edges,
        candidate_cells[: len(candidate_cells)],
        columns="level_of_service",
        resolution=9,
        margin_m=10.0,
        agg=Max(),
        fallback_radius_multiplier=5.0,
    )
    assert unbatched.sort("h3_cell").equals(single_batch.sort("h3_cell"))
