"""Tests for `code.pipeline._join_zensus_grid_population`: WorldPop -> real Zensus grid replacement.

Technical aspects:
    Built 2026-08-22 alongside wiring `pycensus.countries.germany.zensus_grid`
    (a real, live-verified Destatis Zensus 2022 100m x 100m population grid,
    see that module's docstring) into Hamburg's pipeline. Offline: the real
    Destatis download is monkeypatched out via
    `pycensus.countries.germany.zensus_grid.loader.download_grid`, same
    pattern as `pyCensus/tests/test_germany_zensus_grid.py`.

    Uses real H3 hexagon geometry (`h3.grid_disk` + `UrbanAccessAnalyzer.
    h3_ops.to_gdf`) around a real Hamburg-area coordinate, not fabricated
    polygons, so the area-weighted join is exercised on realistic
    geometry/scale (100m Zensus cells against ~150m-wide H3 res-9 hexagons).

    2026-08-30 fix (live user report, Hamburg: "one pixel of worldpop gets
    assigned to one cell and this is wrong"): the nearest-centroid
    `gpd.sjoin_nearest` join this module used to do was replaced with
    `geohierarchy`'s exact area-weighted resampler (`GeoHierarchy.
    add_vector_data` + `Sum(geoweighted=True)`) -- the same approach as
    today's WorldPop raster fix. One real behavior change from that: a
    Zensus cell with genuinely ZERO geometric overlap with any H3 cell in
    the grid (e.g. the "FAR" cell below, placed well outside the whole
    res-9 disk used here) is no longer force-assigned to a "nearest anyway"
    hexagon -- it contributes nothing, which is the geometrically honest
    answer once population is split by real overlap area rather than
    "closest available point". In the real pipeline this never matters
    (the h3 grid already covers the whole AOI by construction), so this
    only changes the synthetic edge case exercised here.
"""

from __future__ import annotations

import geopandas as gpd
import h3
import polars as pl
import pytest
from shapely.geometry import box
from UrbanAccessAnalyzer import h3_ops

from code import pipeline as P


def _hamburg_h3_grid(resolution=9):
    center = h3.latlng_to_cell(53.55, 9.99, resolution)
    cells = list(h3.grid_disk(center, 1))
    df = pl.DataFrame(
        {
            "h3_cell": cells,
            "population": [100.0] * len(cells),  # WorldPop-derived placeholder
            "level_of_service": [1.0] * len(cells),
        }
    )
    grid = h3_ops.to_gdf(df, h3_column="h3_cell")
    grid["area_m2"] = 1_000_000.0
    grid["pop_density"] = grid["population"]
    return grid


@pytest.fixture
def fake_zensus_grid(monkeypatch):
    """Monkeypatch the real Zensus grid loader with real-shaped synthetic 100m cells."""

    def fake_load(aoi=None, data_dir=None):
        # Two real-shaped 100m cells placed near the Hamburg H3 grid's centre
        # (inside the res-9 hex used by `_hamburg_h3_grid`) and one placed
        # far outside every hexagon in that grid (near AOI edge), to
        # exercise the nearest-cell fallback instead of a strict `within`.
        cx, cy = 9.9906, 53.5492
        cells = gpd.GeoDataFrame(
            {
                "GEOID": ["A", "B", "FAR"],
                "population": [42, 8, 15],
            },
            geometry=[
                box(cx - 0.0005, cy - 0.0005, cx, cy),
                box(cx, cy, cx + 0.0005, cy + 0.0005),
                box(20.0, 53.5, 20.0005, 53.5005),  # far outside the AOI's H3 cells
            ],
            crs=4326,
        )
        return cells

    monkeypatch.setattr("pycensus.countries.germany.zensus_grid.loader.load", fake_load)
    return fake_load


class TestJoinZensusGridPopulation:
    def test_replaces_population_and_keeps_worldpop(self, fake_zensus_grid, tmp_path):
        h3_grid = _hamburg_h3_grid()
        aoi = gpd.GeoDataFrame(geometry=[box(9.9, 53.5, 10.1, 53.6)], crs=4326)

        result = P._join_zensus_grid_population(h3_grid, aoi, tmp_path)

        assert "population_worldpop" in result.columns
        assert (result["population_worldpop"] == 100.0).all()  # untouched original WorldPop values

        # Cells A (42) and B (8) both sit inside the res-9 hex disk and are
        # split across every H3 cell they geometrically overlap (area
        # weighted, mass conserving for cells actually touching the grid).
        # "FAR" (15) has zero overlap with any cell in this synthetic grid
        # and is correctly excluded, not force-assigned to a "nearest
        # anyway" hexagon (see module docstring) -- so the real, honestly
        # conserved total here is 42 + 8 = 50, not the full source sum.
        assert result["population"].sum() == pytest.approx(50.0)

        # pop_density must be recomputed from the new population, not stale.
        expected_density = result["population"] / (result["area_m2"] / 1e6)
        assert result["pop_density"].to_numpy() == pytest.approx(expected_density.to_numpy())

    def test_empty_zensus_grid_keeps_worldpop_population(self, monkeypatch, tmp_path):
        def fake_load_empty(aoi=None, data_dir=None):
            return gpd.GeoDataFrame({"GEOID": [], "population": []}, geometry=[], crs=4326)

        monkeypatch.setattr("pycensus.countries.germany.zensus_grid.loader.load", fake_load_empty)

        h3_grid = _hamburg_h3_grid()
        aoi = gpd.GeoDataFrame(geometry=[box(9.9, 53.5, 10.1, 53.6)], crs=4326)
        result = P._join_zensus_grid_population(h3_grid, aoi, tmp_path)

        # Unchanged (still the original WorldPop-derived population, no new columns).
        assert "population_worldpop" not in result.columns
        assert (result["population"] == 100.0).all()

    def test_wired_into_germany_census_dispatch(self, monkeypatch, tmp_path, capsys):
        # _join_census's country="DEU" branch must call
        # _join_zensus_grid_population after the polygon-stats join, and
        # must swallow (not propagate) any failure from it (network/data
        # availability varies, same pattern as the LODES jobs join).
        calls = {}

        def fake_join_polygon_stats(h3_grid, aoi, states, levels, cache_dir, loader, keep_columns, prefix):
            calls["polygon_stats_called"] = True
            return h3_grid

        def fake_join_zensus(h3_grid, aoi, cache_dir):
            calls["zensus_called"] = True
            raise RuntimeError("simulated Destatis outage")

        monkeypatch.setattr(P, "_join_polygon_stats", fake_join_polygon_stats)
        monkeypatch.setattr(P, "_join_zensus_grid_population", fake_join_zensus)

        result = P._join_census(
            h3_grid="fake_h3_grid",
            aoi="fake_aoi",
            states=None,
            census_levels=("district",),
            cache_dir=tmp_path,
            country="DEU",
            census_module=None,
        )

        assert calls == {"polygon_stats_called": True, "zensus_called": True}
        assert result == "fake_h3_grid"  # failure swallowed, h3_grid returned unchanged
        assert "skipping Zensus grid population replacement" in capsys.readouterr().out
