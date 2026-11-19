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

from .los_fill import fill_zero_los  # noqa: E402
logger = logging.getLogger(__name__)

#: Coarse-to-fine order for the real census admin hierarchy (per
#: `build_city_map`'s own docstring convention).
CENSUS_ADMIN_ORDER = ("county", "tract", "blockgroup", "block")

#: Special, explicitly-selected levels -- see module docstring. These are
#: NOT part of the auto-zoom-banded county/tract/blockgroup/block census
#: hierarchy, so they're assembled into their own dict and passed to
#: `build_city_map`'s `special_levels=`, never `census_by_level`.
CENSUS_SPECIAL_LEVELS = ("place", "congressional_district", "state_legislative_district", "school_district", "voting_district")

SPECIAL_LEVEL_LABELS = {
    "place": "Place",
    "congressional_district": "Congressional District",
    "state_legislative_district": "State Legislative District",
    "school_district": "School District",
    "voting_district": "Voting District",
    "state": "State (elections)",
}

H3_RESOLUTIONS = (3, 4, 5, 7, 8, 9, 10, 11)

#: Resolution covering all populated area; finer layers fall back to it where absent.
COARSE_RESOLUTION = 8


def _h3_geometry(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Reconstruct real polygon geometry for an h3-cell-id-only DataFrame.

    The per-county/state h3 outputs deliberately save only the cell id
    (prompt.txt: "For h3 no need to save geometry if not needed") --
    `build_city_map` needs real geometry to build vector tiles, so this
    reconstructs it from each cell id via `h3.cell_to_boundary` at map-
    build time instead.
    """
    from .external_tiles import h3_polygons

    polys = h3_polygons(df["h3"].tolist())   # vectorised (one shapely call per vertex count)
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
STATS_RESOLUTION = 4  # nation-level stats grid embedded in the page (res 4: ~6k cells; states load res 8 on demand)
EXTERNAL_H3 = (9, 10, 11)  # sampled in memory (tiles per state); res 8 is the stats resolution, kept in full
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


def _pmap(fn, items, max_workers: int = 12):
    """Ordered map over states in worker processes (spawn; workers die with the parent). One state or ASSEMBLE_WORKERS=1: in this process."""
    import os

    items = list(items)
    workers = int(os.environ.get("ASSEMBLE_WORKERS", max(1, min((os.cpu_count() or 2) - 2, max_workers))))
    if workers <= 1 or len(items) < 2:
        return [fn(x) for x in items]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from .external_tiles import _die_with_parent

    with ProcessPoolExecutor(min(workers, len(items)), mp_context=mp.get_context("spawn"), initializer=_die_with_parent) as ex:
        return list(ex.map(fn, items))


def _state_coarse_h3(d: Path) -> Optional[pd.DataFrame]:
    """One state's res-7 table with its census variables (a worker job)."""
    from .census_h3 import add_census_variables

    p = d / f"h3_res{COARSE_RESOLUTION}.geoparquet"
    return add_census_variables(pd.read_parquet(p), d) if p.is_file() else None


def _state_census_level(args) -> Optional[gpd.GeoDataFrame]:
    """One state's census level (every county, LOS gaps filled), or a sample of its first county (a worker job)."""
    d, level, sample = args
    files = [p for p in (cf / f"{level}.geoparquet" for cf in sorted((d / "counties").iterdir())) if p.is_file()]
    if sample:
        files = files[:1]
    frames = [fill_zero_los(gpd.read_parquet(p).head(3000) if sample else gpd.read_parquet(p), d) for p in files]
    return _concat_gdfs(frames)


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
    import time as _time
    _t = [_time.time()]

    def _stage(name):   # stage timings in the log, to see where the assembly goes
        now = _time.time(); logger.info(f"assemble: {name} {now - _t[0]:.0f}s"); _t[0] = now

    h3_tables: Dict[int, pd.DataFrame] = {}
    for res in H3_RESOLUTIONS:
        if res == COARSE_RESOLUTION:
            # Census / ACS / jobs variables (tract values) so the stats panel can use every variable, not only elections: one worker per state.
            parts = [x for x in _pmap(_state_coarse_h3, state_dirs) if x is not None]
        else:
            parts = [pd.read_parquet(d / f"h3_res{res}.geoparquet") for d in state_dirs if (d / f"h3_res{res}.geoparquet").is_file()]
        if parts and external and res in EXTERNAL_H3 and res != COARSE_RESOLUTION:
            h3_tables[res] = _representative_sample(parts)
        elif parts:
            h3_tables[res] = pd.concat(parts, ignore_index=True)
        else:
            logger.warning(f"No h3_res{res}.geoparquet under {[d.name for d in state_dirs][:3]}... -- that resolution omitted from the map.")

    _stage("h3 tables + census variables")
    if multi and COARSE_RESOLUTION in h3_tables:
        # A cell on a state border appears in both states with partial counts: merge duplicates
        # at the base resolution, then re-derive every coarser level from the merged table.
        from .state_pipeline import _resample_h3_up

        merged = _resample_h3_up(h3_tables[COARSE_RESOLUTION], COARSE_RESOLUTION)
        h3_tables[COARSE_RESOLUTION] = merged
        coarse = [res for res in H3_RESOLUTIONS if res < COARSE_RESOLUTION]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max(1, len(coarse))) as ex:   # independent levels; numpy's reductions release the GIL
            for res, table in zip(coarse, ex.map(lambda r: _resample_h3_up(merged, r), coarse)):
                h3_tables[res] = table

    # Note: res 7 stays in memory IN FULL even though its tiles are written per state (`code.external_tiles`): the
    # stats panel and the coarse levels are computed from it, and ~1.5M cells fit easily.

    _stage("border merge + coarse levels")
    if 8 in h3_tables:   # development view: one regression per COUNTY on res-8 cells (`code.equity`)
        from .equity import add_equity_flag

        h3_tables[8] = add_equity_flag(h3_tables[8], state_dirs)
    _stage("equity flag")
    h3_by_resolution = {res: _h3_geometry(df) for res, df in h3_tables.items()}
    _stage("h3 geometry")

    # Res 9/11 only exist where LOS > 0 or in dense tracts; the map keeps res 7 drawn
    # underneath them (`set_fallback_level` in transitlos.map.build) for everywhere else.

    census_by_level: Dict[str, gpd.GeoDataFrame] = {}
    # The state polygons are the coarsest census level: the map opens on them (51 features) and counties / tracts / ... take over on zoom.
    state_level = _concat_gdfs([gpd.read_parquet(d / "state.geoparquet") for d in state_dirs if (d / "state.geoparquet").is_file()])
    if state_level is not None:
        census_by_level["state"] = state_level
    county = _concat_gdfs([gpd.read_parquet(d / "county.geoparquet") for d in state_dirs if (d / "county.geoparquet").is_file()])
    if county is not None:
        census_by_level["county"] = county
    for level in ("tract", "blockgroup", "block"):
        sample = bool(external and level in EXTERNAL_CENSUS)   # a sample county per state is enough for schema/domains
        frames = _pmap(_state_census_level, [(d, level, sample) for d in state_dirs if (d / "counties").is_dir()])
        gdf = _concat_gdfs(frames)
        if gdf is not None:
            census_by_level[level] = gdf

    _stage("census levels")
    # Global-schema twins of the differently named ACS columns (`povertyPopulation` next to `acs5_povertyBelow100`, ...).
    from pycensus.global_equivalents import add_global_equivalents

    census_by_level = {k: add_global_equivalents(v) for k, v in census_by_level.items()}

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

    from studytools.route_labels import add_route_labels

    def _labelled(d: Path):
        by_id = d / "stops_by_id.parquet"
        if by_id.is_file():   # one row per real stop_id at its own location, with its own mode / routes / colours (`transitlos.stop_table`)
            t = pd.read_parquet(by_id)
            return gpd.GeoDataFrame(t, geometry=gpd.points_from_xy(t["stop_lon"], t["stop_lat"]), crs=4326)
        g = gpd.read_parquet(d / "stops.geoparquet")
        return add_route_labels(g, sorted(p for p in (d / "gtfs").iterdir() if p.is_dir()) if (d / "gtfs").is_dir() else [])

    _stage("global twins + special levels + streets sample")
    stops_gdf = _concat_gdfs([_labelled(d) for d in state_dirs if (d / "stops.geoparquet").is_file()]) if not multi else None   # nationwide the stops go to per-state chunks below
    if stops_gdf is not None:
        # `build_city_map` expects `mode_category` (`"bus"`/`"tram"`/`"rail"`)
        # -- this study's own equivalent column is `scoring_mode` (see
        # `gtfs_stage.py`/`best_subset_scoring.py`), same 3-value
        # vocabulary, just named differently.
        if "scoring_mode" in stops_gdf.columns and "mode_category" not in stops_gdf.columns:
            stops_gdf = stops_gdf.rename(columns={"scoring_mode": "mode_category"})

    # Nationwide: stops are written as one JSON per state and loaded on demand when zoomed in (the browser cannot hold
    # 300k clustered points); the stats panel / scenario editor read the res-5 grid (embedded, ~35k cells).
    stops_chunks = None
    stats_by_area = None
    if multi:
        stops_chunks = {}
        for d in state_dirs:
            p = d / "stops.geoparquet"
            if p.is_file():
                g = _labelled(d)
                if "scoring_mode" in g.columns and "mode_category" not in g.columns:
                    g = g.rename(columns={"scoring_mode": "mode_category"})
                stops_chunks[d.name] = g
        stops_gdf = None
        from .route_lines_usa import ROUTE_LINES_FILE, compute_state_route_lines

        routes_chunks = {}
        for d in state_dirs:
            f = d / ROUTE_LINES_FILE
            lines = gpd.read_parquet(f) if f.is_file() else (compute_state_route_lines(d) if (d / "gtfs").is_dir() else None)
            if lines is not None and len(lines):
                routes_chunks[d.name] = lines
        _stage("stops + route chunks")
        if STATS_RESOLUTION in h3_by_resolution:
            stats = h3_by_resolution[STATS_RESOLUTION].copy()
            num_cols = [c for c in stats.columns if c not in ("h3_cell", "geometry") and pd.api.types.is_numeric_dtype(stats[c])]
            stats[num_cols] = stats[num_cols].round(2)  # keeps the embedded page JSON small
            stats_by_area = {"metro": stats}

    out = {
        "h3_by_resolution": h3_by_resolution,
        "edges_gdf": edges_gdf,
        "census_by_level": census_by_level,
        "special_levels": special_levels,
        "special_level_labels": {k: v for k, v in SPECIAL_LEVEL_LABELS.items() if k in special_levels},
        "stops_gdf": stops_gdf,
        "is_us": True,
    }
    if stops_chunks is not None:
        out["stops_chunks"] = stops_chunks
        out["routes_chunks"] = routes_chunks
    if stats_by_area is not None:
        out["stats_by_area"] = stats_by_area
    filter_map_columns(out, report_path=Path(state_dirs[0]).parent / "map_columns_dropped.json" if multi else Path(state_dirs[0]) / "map_columns_dropped.json")
    return out


def filter_map_columns(inputs: dict, report_path=None) -> dict:
    """Map-only filter: keep the LATEST municipal/state/national election and the latest-period housing / rent / price columns (US ACS current
    period without `_2016`/`_2000`, FHFA latest, HUD latest FY ...) in what `build_city_map` receives; the stored parquet files keep all
    periods. Data-driven (`pycensus.latest_columns`); the dropped names per frame go to `report_path` (JSON)."""
    from pycensus import derived_columns
    from pycensus.latest_columns import filter_map_inputs

    derived_columns.SHARE_SCALE = 100.0   # the US H3 tables hold shares as percentages: derived / year-built shares on the census levels match
    return filter_map_inputs(inputs, report_path=report_path)


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
