"""Tests for `code.pipeline._sjoin_nearest_chunked`, the bounding-box-chunked
`gpd.sjoin_nearest` wrapper added to fix a Boston-scale OOM in
`_census_geometries_with_score`'s unmatched-polygon fallback (see the
chunking-pattern docstring on `_sjoin_nearest_chunked` and, for the earlier
analogous fix this one is modeled on, `h3_population.py`'s `_edge_chunks`/
`population_and_access_to_h3`).

The key correctness property under test: chunking `left` and restricting each
chunk's `right` candidates to a padded local bounding box must produce the
exact same nearest match as one unchunked `gpd.sjoin_nearest(left, right)`
call, as long as the pad comfortably covers the true nearest distance.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from code.pipeline import _sjoin_nearest_chunked


def _make_points(n, spacing=100.0, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, spacing * n, size=n)
    ys = rng.uniform(0, spacing * n, size=n)
    return gpd.GeoDataFrame(
        {"value": np.arange(n, dtype=float)},
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs="EPSG:32619",
    )


def test_chunked_matches_unchunked_small_case_below_threshold():
    # len(left) <= chunk_size takes the direct, unchunked branch -- still
    # exercise it so a future change to the threshold doesn't silently break it.
    left = _make_points(10, seed=1)
    right = _make_points(50, seed=2)
    left["_geo_idx"] = np.arange(len(left))
    chunked = _sjoin_nearest_chunked(left, right, id_col="_geo_idx", chunk_size=2_000)
    direct = gpd.sjoin_nearest(left, right, how="left")
    assert sorted(chunked["value_right"].tolist()) == sorted(direct["value_right"].tolist())


def test_chunked_matches_unchunked_forces_chunking_branch():
    # A wide, dense grid of `right` points and a small chunk_size so the
    # bounding-box-restricted branch actually runs (not the len<=chunk_size
    # passthrough), with a generous pad so no true nearest match is missed.
    left = _make_points(500, spacing=50.0, seed=3)
    right = _make_points(3_000, spacing=50.0, seed=4)
    left["_geo_idx"] = np.arange(len(left))

    chunked = _sjoin_nearest_chunked(left, right, id_col="_geo_idx", chunk_size=50, pad_m=5_000.0)
    direct = gpd.sjoin_nearest(left, right, how="left")

    chunked_sorted = chunked.sort_values("_geo_idx")["value_right"].to_numpy()
    direct_sorted = direct.sort_values("_geo_idx")["value_right"].to_numpy()
    assert np.array_equal(chunked_sorted, direct_sorted)
    assert len(chunked) == len(left)


def test_chunked_falls_back_to_full_candidates_when_local_box_empty():
    # `right` is far outside any chunk's padded bounding box -- the "pad too
    # small" fallback to the full candidate set must still find it rather
    # than dropping the match.
    left = _make_points(10, spacing=10.0, seed=5)
    far_point = gpd.GeoDataFrame(
        {"value": [999.0]}, geometry=[Point(1_000_000.0, 1_000_000.0)], crs="EPSG:32619"
    )
    left["_geo_idx"] = np.arange(len(left))

    chunked = _sjoin_nearest_chunked(left, far_point, id_col="_geo_idx", chunk_size=1, pad_m=1.0)
    assert len(chunked) == len(left)
    assert (chunked["value_right"] == 999.0).all()


def test_memory_scales_with_chunk_size_not_dataset_size():
    """Peak candidate-set size per `gpd.sjoin_nearest` call should be bounded
    by the local bounding box, not the size of `right` -- the whole point of
    this fix. Monkeypatch `gpd.sjoin_nearest` to record how many candidate
    rows it was actually handed on each call.
    """
    import code.pipeline as pipeline_mod

    left = _make_points(1_000, spacing=20.0, seed=6)
    right = _make_points(20_000, spacing=20.0, seed=7)
    left["_geo_idx"] = np.arange(len(left))

    call_sizes = []
    real_sjoin_nearest = gpd.sjoin_nearest

    def _spy(left_df, right_df, **kwargs):
        call_sizes.append(len(right_df))
        return real_sjoin_nearest(left_df, right_df, **kwargs)

    orig = pipeline_mod.gpd.sjoin_nearest
    pipeline_mod.gpd.sjoin_nearest = _spy
    try:
        pipeline_mod._sjoin_nearest_chunked(left, right, id_col="_geo_idx", chunk_size=100, pad_m=200.0)
    finally:
        pipeline_mod.gpd.sjoin_nearest = orig

    assert call_sizes, "sjoin_nearest was never called"
    # Every chunk's candidate set should be a small local slice, nowhere near
    # the full 20,000-row `right` grid.
    assert max(call_sizes) < len(right) / 2
