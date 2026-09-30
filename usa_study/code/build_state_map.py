# -*- coding: utf-8 -*-
"""Assemble one state's usa_study output into `transitlos.map.build.build_city_map`'s shapes.

`build_city_map` (12k+ lines, already built and battle-tested for
`city_science_network`) takes exactly the data shapes this study's
pipeline already produces -- `h3_by_resolution: Dict[int, gdf]`,
`edges_gdf`, `census_by_level: Dict[str, gdf]` (coarse to fine),
`stops_gdf` -- so this module is pure assembly/glue, not a
reimplementation of the map itself.

Per Miguel's 2026-09-29 clarification ("election levels and place are
special and have to be specially selected to be visualized" -- unlike
county/tract/blockgroup/block, which auto-select by zoom), these levels
are passed through `build_city_map`'s own `special_levels`/
`special_level_labels` params (a genuinely separate "special" selector
group, not folded into the zoom-banded `census_by_level` dict). Includes
`state` (this study's own `state.geoparquet`: state-NATIVE election
results + resampled-up census, from `state_pipeline
.add_state_native_elections`) alongside the 4 census-district special
levels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import h3
import pandas as pd
import shapely

logger = logging.getLogger(__name__)

#: Coarse-to-fine order for the real census admin hierarchy (per
#: `build_city_map`'s own docstring convention).
CENSUS_ADMIN_ORDER = ("county", "tract", "blockgroup", "block")

#: Special, explicitly-selected levels -- see module docstring. These are
#: NOT part of the auto-zoom-banded county/tract/blockgroup/block census
#: hierarchy, so they're assembled into their own dict and passed to
#: `build_city_map`'s `special_levels=`, never `census_by_level`.
CENSUS_SPECIAL_LEVELS = ("place", "congressional_district", "state_legislative_district", "school_district")

SPECIAL_LEVEL_LABELS = {
    "place": "Place",
    "congressional_district": "Congressional District",
    "state_legislative_district": "State Legislative District",
    "school_district": "School District",
    "state": "State (elections)",
}

H3_RESOLUTIONS = (5, 7, 9, 11)


def _h3_geometry(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Reconstruct real polygon geometry for an h3-cell-id-only DataFrame.

    The per-county/state h3 outputs deliberately save only the cell id
    (prompt.txt: "For h3 no need to save geometry if not needed") --
    `build_city_map` needs real geometry to build vector tiles, so this
    reconstructs it from each cell id via `h3.cell_to_boundary` at map-
    build time instead.
    """
    polys = [shapely.Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)]) for cell in df["h3"]]
    return gpd.GeoDataFrame(df, geometry=polys, crs=4326)


def _load_state_streets(state_dir: Path) -> gpd.GeoDataFrame:
    """Concatenate every county's own `street.geoparquet` into one state-wide edges GeoDataFrame."""
    frames = []
    for county_folder in sorted((state_dir / "counties").iterdir()):
        p = county_folder / "street.geoparquet"
        if p.is_file():
            frames.append(gpd.read_parquet(p))
    if not frames:
        raise ValueError(f"No street.geoparquet files found under {state_dir / 'counties'}")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)


def assemble_state_map_inputs(state_dir: Path) -> dict:
    """Build every `build_city_map` argument from one state's real usa_study output.

    Args:
        state_dir: State folder (containing `stops.geoparquet`,
            `county.geoparquet`, `h3_res*.geoparquet`, and a `counties/`
            subdirectory), as written by `main.py`/`state_pipeline.py`.

    Returns:
        Dict of keyword arguments ready to `**`-splat into
        `transitlos.map.build.build_city_map` (missing optional inputs,
        e.g. no state-district files yet built, are simply omitted).
    """
    h3_by_resolution: Dict[int, gpd.GeoDataFrame] = {}
    for res in H3_RESOLUTIONS:
        path = state_dir / f"h3_res{res}.geoparquet"
        if path.is_file():
            h3_by_resolution[res] = _h3_geometry(pd.read_parquet(path))
        else:
            logger.warning(f"No h3_res{res}.geoparquet under {state_dir} -- that resolution omitted from the map.")

    census_by_level: Dict[str, gpd.GeoDataFrame] = {}
    county_path = state_dir / "county.geoparquet"
    if county_path.is_file():
        census_by_level["county"] = gpd.read_parquet(county_path)
    for level in ("tract", "blockgroup", "block"):
        frames = []
        for county_folder in sorted((state_dir / "counties").iterdir()):
            p = county_folder / f"{level}.geoparquet"
            if p.is_file():
                frames.append(gpd.read_parquet(p))
        if frames:
            census_by_level[level] = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)

    special_levels: Dict[str, gpd.GeoDataFrame] = {}
    for level in (*CENSUS_SPECIAL_LEVELS, "state"):
        path = state_dir / f"{level}.geoparquet"
        if path.is_file():
            special_levels[level] = gpd.read_parquet(path)
        else:
            logger.warning(f"No {level}.geoparquet under {state_dir} -- omitted from the map's special selector.")

    edges_gdf = _load_state_streets(state_dir)

    stops_gdf = None
    stops_path = state_dir / "stops.geoparquet"
    if stops_path.is_file():
        stops_gdf = gpd.read_parquet(stops_path)
        # `build_city_map` expects `mode_category` (`"bus"`/`"tram"`/`"rail"`)
        # -- this study's own equivalent column is `scoring_mode` (see
        # `gtfs_stage.py`/`best_subset_scoring.py`), same 3-value
        # vocabulary, just named differently.
        if "scoring_mode" in stops_gdf.columns and "mode_category" not in stops_gdf.columns:
            stops_gdf = stops_gdf.rename(columns={"scoring_mode": "mode_category"})

    return {
        "h3_by_resolution": h3_by_resolution,
        "edges_gdf": edges_gdf,
        "census_by_level": census_by_level,
        "special_levels": special_levels,
        "special_level_labels": {k: v for k, v in SPECIAL_LEVEL_LABELS.items() if k in special_levels},
        "stops_gdf": stops_gdf,
        "is_us": True,
    }


def build_state_map(state_dir: Path, tiles_dir: Optional[Path] = None, out_html: Optional[Path] = None) -> str:
    """Build one state's real map.html from its usa_study output.

    Args:
        state_dir: See `assemble_state_map_inputs`.
        tiles_dir: Vector-tile output directory; defaults to
            `state_dir / "tiles"`.
        out_html: Output HTML path; defaults to `state_dir / "map.html"`.

    Returns:
        Absolute path to the written HTML file (see `build_city_map`).
    """
    from transitlos.map.build import build_city_map

    inputs = assemble_state_map_inputs(state_dir)
    return build_city_map(
        tiles_dir=str(tiles_dir or state_dir / "tiles"),
        out_html=str(out_html or state_dir / "map.html"),
        **inputs,
    )
