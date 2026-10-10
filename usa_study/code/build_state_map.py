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

H3_RESOLUTIONS = (3, 4, 5, 7, 9, 11)

#: Resolution covering all populated area; finer layers fall back to it where absent.
COARSE_RESOLUTION = 7


def _h3_geometry(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Reconstruct real polygon geometry for an h3-cell-id-only DataFrame.

    The per-county/state h3 outputs deliberately save only the cell id
    (prompt.txt: "For h3 no need to save geometry if not needed") --
    `build_city_map` needs real geometry to build vector tiles, so this
    reconstructs it from each cell id via `h3.cell_to_boundary` at map-
    build time instead.
    """
    polys = [shapely.Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)]) for cell in df["h3"]]
    # `build_city_map` keys h3 layers on `h3_cell`.
    return gpd.GeoDataFrame(df.rename(columns={"h3": "h3_cell"}), geometry=polys, crs=4326)


def _load_state_streets(state_dir: Path) -> gpd.GeoDataFrame:
    """Concatenate every county's own `street.geoparquet` into one state-wide edges GeoDataFrame."""
    frames = []
    for county_folder in sorted((state_dir / "counties").iterdir()):
        p = county_folder / "street.geoparquet"
        if p.is_file():
            frames.append(gpd.read_parquet(p))
    if not frames:
        logger.warning(f"No street.geoparquet files under {state_dir / 'counties'} (no scored streets) -- skipped.")
        return None
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)


#: Levels whose tiles are written per state in nationwide ("external") mode -- keep in sync with `code.external_tiles`.
EXTERNAL_H3 = (9, 11)  # sampled in memory (tiles per state); res 7 is tiled per state but kept in full
EXTERNAL_CENSUS = ("blockgroup", "block")


def _representative_sample(parts, n_per_state: int = 4000) -> pd.DataFrame:
    """A small stand-in for a huge per-state table: a random sample from each state plus every numeric
    column's min and max row, so schema and value domains (colour/opacity ranges) match the full data."""
    out = []
    for df in parts:
        if df.empty:
            continue
        pick = {*df.sample(min(len(df), n_per_state), random_state=0).index}
        for col in df.select_dtypes("number").columns:
            if df[col].notna().any():
                pick.add(df[col].idxmax())
                pick.add(df[col].idxmin())
        out.append(df.loc[sorted(pick)])
    return pd.concat(out, ignore_index=True)


def _concat_gdfs(frames):
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return None
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)


def assemble_state_map_inputs(state_dir: Path) -> dict:
    """Build every `build_city_map` argument from one state's real usa_study output (see `assemble_map_inputs`)."""
    return assemble_map_inputs([Path(state_dir)])


def assemble_map_inputs(state_dirs, include_streets: bool = True, external: bool = False) -> dict:
    """Build every `build_city_map` argument from one or MANY states' usa_study output.

    Args:
        state_dirs: State folders (each containing `stops.geoparquet`,
            `county.geoparquet`, `h3_res*.geoparquet` and a `counties/`
            subdirectory), as written by `main.py`/`state_pipeline.py`.
            Passing all of them gives the nationwide map inputs.
        include_streets: Whether to load the (very large) street edges.
        external: Nationwide mode: the heavy levels (h3 res 9/11, census block/blockgroup, streets) are NOT
            loaded -- only a representative sample (schema + value domains) is returned, because their tiles
            are written per state by `code.external_tiles` and never held in one process.

    Returns:
        Dict of keyword arguments ready to `**`-splat into
        `transitlos.map.build.build_city_map` (missing optional inputs,
        e.g. no state-district files yet built, are simply omitted).
    """
    state_dirs = [Path(d) for d in state_dirs]
    multi = len(state_dirs) > 1

    h3_tables: Dict[int, pd.DataFrame] = {}
    for res in H3_RESOLUTIONS:
        parts = [pd.read_parquet(d / f"h3_res{res}.geoparquet") for d in state_dirs if (d / f"h3_res{res}.geoparquet").is_file()]
        if parts and external and res in EXTERNAL_H3 and res != COARSE_RESOLUTION:
            h3_tables[res] = _representative_sample(parts)
        elif parts:
            h3_tables[res] = pd.concat(parts, ignore_index=True)
        else:
            logger.warning(f"No h3_res{res}.geoparquet under {[d.name for d in state_dirs][:3]}... -- that resolution omitted from the map.")

    if multi and COARSE_RESOLUTION in h3_tables:
        # A cell on a state border appears in both states with partial counts: merge duplicates
        # at the base resolution, then re-derive every coarser level from the merged table.
        from .state_pipeline import _resample_h3_up

        merged = _resample_h3_up(h3_tables[COARSE_RESOLUTION], COARSE_RESOLUTION)
        h3_tables[COARSE_RESOLUTION] = merged
        for res in H3_RESOLUTIONS:
            if res < COARSE_RESOLUTION:
                h3_tables[res] = _resample_h3_up(merged, res)

    # Note: res 7 stays in memory IN FULL even though its tiles are written per state (`code.external_tiles`): the
    # stats panel and the coarse levels are computed from it, and ~1.5M cells fit easily.

    h3_by_resolution = {res: _h3_geometry(df) for res, df in h3_tables.items()}

    # Res 9/11 only exist where LOS > 0 or in dense tracts; the map keeps res 7 drawn
    # underneath them (`set_fallback_level` in transitlos.map.build) for everywhere else.

    census_by_level: Dict[str, gpd.GeoDataFrame] = {}
    county = _concat_gdfs([gpd.read_parquet(d / "county.geoparquet") for d in state_dirs if (d / "county.geoparquet").is_file()])
    if county is not None:
        census_by_level["county"] = county
    for level in ("tract", "blockgroup", "block"):
        frames = []
        for d in state_dirs:
            files = [p for p in (cf / f"{level}.geoparquet" for cf in sorted((d / "counties").iterdir())) if p.is_file()]
            if external and level in EXTERNAL_CENSUS:
                files = files[:1]  # a sample county per state is enough for schema/domains
            for p in files:
                frames.append(gpd.read_parquet(p).head(3000) if external and level in EXTERNAL_CENSUS else gpd.read_parquet(p))
        gdf = _concat_gdfs(frames)
        if gdf is not None:
            census_by_level[level] = gdf

    special_levels: Dict[str, gpd.GeoDataFrame] = {}
    for level in (*CENSUS_SPECIAL_LEVELS, "state"):
        gdf = _concat_gdfs([gpd.read_parquet(d / f"{level}.geoparquet") for d in state_dirs if (d / f"{level}.geoparquet").is_file()])
        if gdf is not None:
            special_levels[level] = gdf
        else:
            logger.warning(f"No {level}.geoparquet found -- omitted from the map's special selector.")

    edges_gdf = None
    if include_streets and not external:
        edges_gdf = _concat_gdfs([_load_state_streets(d) for d in state_dirs if (d / "counties").is_dir()])

    if edges_gdf is None:
        # `build_city_map` requires an edges frame: hand it a small sample so the streets
        # overlay is effectively empty when streets are excluded.
        samples = []
        for d in state_dirs:
            sample = next(iter(sorted((d / "counties").glob("*/street.geoparquet"))), None)
            if sample is not None:
                samples.append(gpd.read_parquet(sample).head(300))
            if len(samples) >= 10:
                break
        edges_gdf = _concat_gdfs(samples)

    stops_gdf = _concat_gdfs([gpd.read_parquet(d / "stops.geoparquet") for d in state_dirs if (d / "stops.geoparquet").is_file()])
    if stops_gdf is not None:
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
