"""Tests for Germany's new `grid` census level: the real Zensus 100m grid, as native rectangles.

Technical aspects:
    Unlike `test_zensus_grid_population_join.py` (which covers
    `_join_zensus_grid_population`'s *replace-population-on-the-existing-h3-grid*
    path), this covers the *separate* map-level "grid" census level added
    to `GERMANY_LEVELS`: `code.pipeline._germany_grid_census_loader` and its
    dispatch through `_germany_map_census_loader`/
    `_census_loader_and_levels_for_map`.

    2026-08-30 (explicit user request: "for the german census pixels I
    would like rectangle layer and not h3"): `_germany_grid_census_loader`
    no longer resamples onto H3 at all -- it hands back the Zensus grid's
    own native 100m x 100m rectangles unchanged, which
    `_census_geometries_with_score` (the only caller) already knows how to
    render as an arbitrary polygon level via `GeoHierarchy.add_level`, the
    same generic mechanism used for every other country's real census admin
    polygons.

    Offline: the real Destatis download is monkeypatched out, same pattern
    as `test_zensus_grid_population_join.py`.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from code import pipeline as P


@pytest.fixture
def fake_zensus_grid(monkeypatch):
    """A small real-shaped synthetic 100m grid near Hamburg (3 cells, pop 42+8+15=65)."""

    def fake_load(aoi=None, data_dir=None):
        cx, cy = 9.9906, 53.5492
        cells = gpd.GeoDataFrame(
            {"GEOID": ["A", "B", "C"], "population": [42, 8, 15]},
            geometry=[
                box(cx - 0.0005, cy - 0.0005, cx, cy),
                box(cx, cy, cx + 0.0005, cy + 0.0005),
                box(cx + 0.0005, cy, cx + 0.001, cy + 0.0005),
            ],
            crs=4326,
        )
        return cells

    monkeypatch.setattr("pycensus.countries.germany.zensus_grid.loader.load", fake_load)
    return fake_load


class TestGermanyGridCensusLoader:
    def test_grid_is_finest_level_in_germany_levels(self):
        assert P.GERMANY_LEVELS[0] == "grid"
        assert "grid" not in P.GERMANY_ADMIN_LEVELS

    def test_join_census_h3_join_excludes_grid_level(self, monkeypatch, tmp_path):
        # The h3-grid-column join (`_join_census`) must NOT pass "grid" to
        # `_join_polygon_stats` -- that apportionment logic is for admin
        # polygons, and grid-level population replacement is handled
        # separately by `_join_zensus_grid_population`.
        seen_levels = {}

        def fake_join_polygon_stats(h3_grid, aoi, states, levels, cache_dir, loader, keep_columns, prefix):
            seen_levels["levels"] = levels
            return h3_grid

        def fake_join_zensus(h3_grid, aoi, cache_dir):
            return h3_grid

        monkeypatch.setattr(P, "_join_polygon_stats", fake_join_polygon_stats)
        monkeypatch.setattr(P, "_join_zensus_grid_population", fake_join_zensus)

        P._join_census(
            h3_grid="fake_h3_grid", aoi="fake_aoi", states=None, census_levels=("district",),
            cache_dir=tmp_path, country="DEU", census_module=None,
        )
        assert "grid" not in seen_levels["levels"]
        assert seen_levels["levels"] == P.GERMANY_ADMIN_LEVELS

    def test_census_loader_and_levels_for_map_includes_grid(self):
        loader, levels = P._census_loader_and_levels_for_map("DEU", None)
        assert loader is P._germany_map_census_loader
        assert levels == P.GERMANY_LEVELS
        assert "grid" in levels

    def test_grid_loader_returns_native_rectangles_not_h3(self, fake_zensus_grid, tmp_path):
        aoi = gpd.GeoDataFrame(geometry=[box(9.9, 53.5, 10.1, 53.6)], crs=4326)
        out = P._germany_grid_census_loader(aoi=aoi, states=None, level="grid", cache_dir=tmp_path)

        assert not out.empty
        # No H3 resampling at all -- the loader's own real GEOID/geometry
        # pass through unchanged, one row per source 100m cell (no grouping,
        # no h3_cell/resolution columns).
        assert "h3_cell" not in out.columns
        assert "resolution" not in out.columns
        assert list(out["GEOID"]) == ["A", "B", "C"]
        assert out["population"].sum() == pytest.approx(65.0)
        assert len(out) == 3
        # Real 100m squares, not H3 hexagons -- every geometry is a rectangle.
        assert all(geom.geom_type == "Polygon" for geom in out.geometry)

    def test_map_dispatch_routes_grid_level_to_grid_loader(self, fake_zensus_grid, tmp_path):
        aoi = gpd.GeoDataFrame(geometry=[box(9.9, 53.5, 10.1, 53.6)], crs=4326)
        out = P._germany_map_census_loader(aoi=aoi, states=None, level="grid", cache_dir=tmp_path)
        assert "h3_cell" not in out.columns
        assert out["population"].sum() == pytest.approx(65.0)

    def test_map_dispatch_routes_admin_level_to_destatis_loader(self, monkeypatch, tmp_path):
        calls = {}

        def fake_destatis_load(aoi=None, level=None, cache_dir=None):
            calls["level"] = level
            return gpd.GeoDataFrame({"population": [1.0]}, geometry=[box(0, 0, 1, 1)], crs=4326)

        monkeypatch.setattr(
            "pycensus.countries.germany.destatis.loader.load", fake_destatis_load
        )
        aoi = gpd.GeoDataFrame(geometry=[box(9.9, 53.5, 10.1, 53.6)], crs=4326)
        out = P._germany_map_census_loader(aoi=aoi, states=None, level="district", cache_dir=tmp_path)
        assert calls["level"] == "district"
        assert not out.empty

    def test_grid_loader_empty_for_aoi_outside_germany(self, monkeypatch, tmp_path):
        def fake_load_empty(aoi=None, data_dir=None):
            return gpd.GeoDataFrame({"GEOID": [], "population": []}, geometry=[], crs=4326)

        monkeypatch.setattr("pycensus.countries.germany.zensus_grid.loader.load", fake_load_empty)
        aoi = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=4326)
        out = P._germany_grid_census_loader(aoi=aoi, states=None, level="grid", cache_dir=tmp_path)
        assert out.empty
