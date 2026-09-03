"""Regression test for `_join_worldpop_global_schema`'s geometry-free rewrite.

2026-08-31 OOM fix: Shanghai (the only city on this non-census code path --
`CityConfig.uses_census == False`) died with an OOM kill immediately on this
join's first feature at its real ~25.7M-cell H3 grid scale. Root cause,
confirmed via a real watchdog-guarded repro against the actual cached
(smaller, 13M-row) `shanghai/results/metro/h3_grid.parquet`: `run_city_study`
called `_add_h3_grid` (which builds real `shapely` hexagon-polygon geometry
for every H3 cell) BEFORE this join ran, even though this join's own raster
resampling (`worldpop_raster_to_h3` / `raster_to_h3_tiled`) builds its OWN H3
cell geometry internally from each downloaded raster's extent and never
reads the caller's geometry at all -- so the real geometry-bearing
`GeoDataFrame` sat fully resident (confirmed to alone OOM-kill a 10GB cgroup
at 13M rows) for the whole duration of a join that never needed it.

The fix moved this join to run on the lightweight, geometry-free
`pl.DataFrame` `population_and_access_to_h3` returns, BEFORE `_add_h3_grid`
runs (see `run_city_study`'s non-census branch), and rewrote the join itself
from a pandas `.map()`-on-a-`.to_pandas()`-Series (plus an unneeded
`reset_index(drop=True)` full-frame copy) to a plain polars left-join.

This test locks in that the rewritten join:
  - accepts/returns a plain `pl.DataFrame` (no geometry column required),
  - left-joins each downloaded feature onto `h3_cell` and fills cells the
    raster didn't cover with 0.0 (never null), matching the old behavior,
  - keeps skipping any feature name already present on the input table
    (the pre-existing collision guard), and
  - does not require -- and does not choke on -- extra non-`h3_cell`
    columns already present on the input table (e.g. `population`,
    `level_of_service`), leaving them untouched.
"""

import polars as pl
import pytest

from code import pipeline as P


def test_join_worldpop_global_schema_is_geometry_free_polars_join(monkeypatch, tmp_path):
    h3_table = pl.DataFrame(
        {
            "h3_cell": ["cellA", "cellB", "cellC"],
            "population": [10.0, 20.0, 0.0],
            "level_of_service": [1.0, 2.0, 0.0],
        }
    )

    fake_features = ["malePopulation", "femalePopulation"]
    # Only cellA/cellB are "covered" by the fake raster for each feature --
    # cellC (no pixel touching it) must come back as a real 0.0, not null.
    fake_layer_values = {
        "malePopulation": pl.DataFrame({"h3_cell": ["cellA", "cellB"], "malePopulation": [4.0, 8.0]}),
        "femalePopulation": pl.DataFrame({"h3_cell": ["cellA", "cellB"], "femalePopulation": [6.0, 12.0]}),
    }

    monkeypatch.setattr(P, "gpd", P.gpd)  # sanity: module still has gpd for the `aoi` type hint

    import pycensus.countries.worldwide.worldpop.loader as loader_mod
    import pycensus.countries.worldwide.worldpop.h3 as worldpop_h3_mod

    monkeypatch.setattr(loader_mod, "GLOBAL_SCHEMA_FEATURES", ["population"] + fake_features, raising=False)
    monkeypatch.setattr(
        loader_mod, "download_worldpop_layer", lambda aoi, year, feature, folder: f"/fake/{feature}.tif"
    )
    monkeypatch.setattr(
        worldpop_h3_mod,
        "worldpop_raster_to_h3",
        lambda tif_path, resolution, value_col: fake_layer_values[value_col],
    )

    result = P._join_worldpop_global_schema(
        h3_table, aoi=None, resolution=9, year=2025, cache_dir=tmp_path
    )

    assert isinstance(result, pl.DataFrame)
    assert "geometry" not in result.columns
    # Original columns preserved untouched.
    assert result.sort("h3_cell")["population"].to_list() == [10.0, 20.0, 0.0]
    assert result.sort("h3_cell")["level_of_service"].to_list() == [1.0, 2.0, 0.0]
    # New features landed under bare names, 0.0 (not null) for the untouched cell.
    got = result.sort("h3_cell")
    assert got["malePopulation"].to_list() == [4.0, 8.0, 0.0]
    assert got["femalePopulation"].to_list() == [6.0, 12.0, 0.0]
    assert got["malePopulation"].null_count() == 0
    assert got["femalePopulation"].null_count() == 0


def test_join_worldpop_global_schema_skips_existing_columns(monkeypatch, tmp_path):
    h3_table = pl.DataFrame(
        {
            "h3_cell": ["cellA"],
            "population": [10.0],
            "malePopulation": [99.0],  # already present -- must be left alone
        }
    )

    import pycensus.countries.worldwide.worldpop.loader as loader_mod
    import pycensus.countries.worldwide.worldpop.h3 as worldpop_h3_mod

    monkeypatch.setattr(
        loader_mod, "GLOBAL_SCHEMA_FEATURES", ["population", "malePopulation", "femalePopulation"], raising=False
    )
    calls = []

    def _fake_download(aoi, year, feature, folder):
        calls.append(feature)
        return f"/fake/{feature}.tif"

    monkeypatch.setattr(loader_mod, "download_worldpop_layer", _fake_download)
    monkeypatch.setattr(
        worldpop_h3_mod,
        "worldpop_raster_to_h3",
        lambda tif_path, resolution, value_col: pl.DataFrame({"h3_cell": ["cellA"], value_col: [5.0]}),
    )

    result = P._join_worldpop_global_schema(h3_table, aoi=None, resolution=9, year=2025, cache_dir=tmp_path)

    # malePopulation already existed -- must be skipped (never re-downloaded, never overwritten).
    assert "malePopulation" not in calls
    assert result["malePopulation"].to_list() == [99.0]
    # femalePopulation is new -- must be joined in.
    assert "femalePopulation" in calls
    assert result["femalePopulation"].to_list() == [5.0]
