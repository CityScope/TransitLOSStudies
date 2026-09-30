"""Study-wide parameters shared identically across every city.

Per the study's "standard parameters (same for all stops)" requirement, one
fixed `StudyParams` instance is used for every city regardless of continent
-- this deliberately does *not* switch `region` per city. **2026-09-28**:
`transitlos.scoring.parameters.REGIONS` no longer encodes genuine
region-specific literature defaults at all -- the finalized scoring model
(scoring.md Section 3.3) found no defensible region-specific override for
any of its parameters, so every `REGIONS` key ("global", "europe",
"north_america", "global_south") now maps to identical, universal values.
Using `region="global"` uniformly therefore no longer trades away any real
regional calibration accuracy; it is kept only for API-call-site stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Optional


@dataclass(frozen=True)
class StudyParams:
    """Parameters forwarded into every step of `code.pipeline.run_city_study`.

    Attributes:
        region: `transitlos.scoring.parameters.REGIONS` key used for every
            city's stop/LOS scoring. Fixed at `"global"` study-wide (see
            module docstring) so every city's `level_of_service` is on the same
            scale -- never set per-city. Kept for API-call-site stability
            only; every `REGIONS` key now resolves to identical parameters.
        walk_distance_steps: Isochrone distance thresholds (meters) forwarded
            to `transitlos.level_of_service.compute_level_of_service`. The
            literature-standard bands are 400/800/1200m (TCRP 95 /
            ARE-ÖROK); 2000m is appended as an explicit *extension* beyond
            the literature (an outer band using the same decay model, not a
            new literature claim) per this study's ask to also look further
            than the standard walk-catchment bands.
        h3_resolution: H3 resolution used for the population/equity grid
            (`code.h3_population`). Res 10 cells are ~65 m across -- fine
            enough to resolve individual street-adjacent blocks.
        map_h3_resolutions: H3 resolutions rendered as separate zoom-banded
            layers in each city's map (`transitlos.map`).
        census_levels: US Census geometry levels joined onto the population
            grid for US cities (`pycensus.countries.usa.constants.GEOMETRY_FUNCS`
            keys).
        score_bins: Distance-matrix score bucket count forwarded to
            `compute_level_of_service` (see that function's docstring for
            the speed/accuracy tradeoff table). Set to 30 (not the
            library's own "very high precision" default of 50): since
            `level_of_service` is discretized to 0.05-wide bins after this step
            anyway (`score_bin_width` below), only half-a-bin accuracy
            (0.025) is actually needed to guarantee correct bin assignment.
            `score_bins=30` has max error 0.017 (safely under 0.025, per
            the docstring's benchmark table) while being the *fastest*
            setting in that table (3.38x speedup vs `None`, vs 50's 2.86x)
            -- also means more stops share a tier (`UrbanAccessAnalyzer.
            isochrones.compute_node_access` runs one multi-source search
            per tier across every stop in it, not one search per stop), so
            fewer, larger multi-source searches.
        stats_h3_resolution: H3 resolution every regression and median-split
            ANOVA (`code.pipeline._run_regressions_and_anovas`) is run at.
            Coarser than `h3_resolution` on purpose -- res 10 cells are
            individually noisy (small/zero population, near-binary
            street-adjacency), so statistics are computed on a res-8
            resample instead (population/Census counts summed, level_of_service
            population-weighted-averaged; see `code.pipeline._resample_h3`).
        analysis_start_time: Start of the daily service window headway and
            speed are computed over (`transitlos.stops.
            download_and_prepare_stops`'s `start_time`). Fixed at 06:00
            study-wide so every city's `level_of_service` reflects daytime
            service only, not overnight/owl-service headways that would
            otherwise drag the score down uniformly and uninformatively.
        analysis_end_time: End of that window. Fixed at 22:00 study-wide.
        isochrone_chunk_h3_resolution: H3 resolution used to partition the
            street network into memory-bounded chunks for isochrone
            computation (`UrbanAccessAnalyzer.isochrones.compute_node_access_chunked`,
            wired through `transitlos.level_of_service.compute_level_of_service`),
            census join (`code.pipeline._join_polygon_stats_chunked`), and H3
            resampling (`code.pipeline._resample_h3_chunked`) -- one knob
            controls all three, since they're the same underlying "whole
            metro area in memory at once" OOM risk (see
            `docs/H3_CHUNKED_PIPELINE_DESIGN.md`). Default `None`: chunking
            is off, every step runs its original whole-AOI-at-once path,
            byte-for-byte identical to pre-chunking behavior. Set to e.g. `4`
            (~1,770 km^2/cell) per-city for an oversized AOI that OOMs
            unchunked (Boston, later Shanghai) -- this is deliberately NOT a
            study-wide default, since ordinary well-scoped AOIs (San
            Francisco, Guadalajara, ...) don't need it and chunking adds
            overhead (buffer overlap, per-chunk fixed costs) for no benefit
            on those cities.
        isochrone_chunk_buffer_m: Buffer (meters) added around each
            isochrone chunk before clipping the street network, only used
            when `isochrone_chunk_h3_resolution` is set. Must be >=
            `max(walk_distance_steps)` (currently 2000.0) or a stop's
            isochrone search gets truncated at the chunk boundary instead of
            reaching its true radius -- see
            `UrbanAccessAnalyzer.isochrones.compute_node_access_chunked`'s
            docstring for the full correctness argument. Default 1000.0
            matches the project's own initial proposal but is deliberately
            SMALLER than `max(walk_distance_steps)` -- any city actually
            enabling chunking must pass a larger value explicitly (e.g.
            2000.0) rather than silently inheriting an under-sized default;
            `compute_node_access_chunked` warns (not silently truncates) if
            this invariant is violated.
        census_chunk_buffer_m: Buffer (meters) added around each chunk for
            the census-join step (`_join_polygon_stats_chunked`) and used as
            the sub-AOI expansion for H3 resampling's chunk partitioning.
            Much smaller than `isochrone_chunk_buffer_m` is typically
            sufficient -- unlike a walk isochrone's multi-hundred-meter
            search radius, a census-polygon-to-h3-cell join is a purely
            local operation (only needs to cover a polygon/cell straddling a
            chunk boundary), similar in spirit to the small tile-blending
            buffer `tile_chunk_buffer_m` describes for the (separately
            owned) tiling stage. Default 200.0. NOTE: apportionment
            correctness for a census level whose polygons are themselves
            larger than this buffer (e.g. "county"/"place") is only
            approximate under chunking -- a polygon straddling a chunk
            boundary has its total apportioned using only the h3 cells
            visible to each chunk, not the polygon's true full cell set. See
            `_join_polygon_stats_chunked`'s docstring.
        tile_chunk_buffer_m: Buffer (meters) used by `geohierarchy`'s (a
            separate, sibling-owned package) chunked tile generation for
            per-H3-cell tile-boundary blending. Lives here (not in
            `geohierarchy`) only because the user asked for all chunk-size
            knobs to be configurable "in the main file" alongside the other
            H3-chunking parameters -- `city_science_network` itself does not read
            this field. Default 100.0: tile rendering only needs geometry
            visible near a cell edge for a clean visual transition, not a
            full isochrone- or census-join-scale buffer.
        street_simplify_distance: Cluster distance (meters) forwarded to
            `transitlos.network.prepare_street_network`'s `cluster_distance`
            (-> `StreetNetwork.simplify`). Fixed at 30.0 study-wide (up from
            the library's own 10.0 default): collapses node clusters within
            30m into one node, shrinking the graph before every downstream
            step (isochrones, H3 aggregation) without materially changing
            walk-accessibility at the 400m+ distance scales this study uses.
        score_bin_width: Width `level_of_service` is discretized to after
            `compute_level_of_service` (`code.stats.discretize_score`).
            Default 0.05 -> 20 bins covering `[0, 1]`, applied uniformly so
            every city/figure/map compares the same discrete scale rather
            than continuous floating-point scores.
        chunked_h3_output: Opt-in, per-city flag (default `None`/off, must
            be set explicitly on a `CityConfig`/`StudyParams` override --
            NEVER a study-wide default) enabling the "real chunked" H3 grid
            output path: instead of `_add_h3_grid`/`build_h3_by_resolution`
            materializing the WHOLE city's population/access/census grid as
            ONE in-memory `GeoDataFrame` before writing `h3_grid.parquet`,
            `geohierarchy.chunked_h3_grid.build_h3_grid_chunked` partitions
            the geometry-free grid table by each cell's ancestor at this H3
            resolution and writes ONE standalone GeoParquet file per tile
            (`results/metro/h3_grid_chunks/tile_<id>.parquet`), building
            each tile's hexagon geometry in its own `ProcessPoolExecutor`
            worker so peak memory is bounded by one tile's size, never the
            whole city's -- see that function's docstring for why no
            halo/buffer is needed here (unlike `isochrone_chunk_h3_resolution`'s
            street-network chunking) -- a pure attribute table partitioned
            by H3 ancestor id is an exact partition on its own. Distinct
            from `isochrone_chunk_h3_resolution` (which still merges its
            chunked isochrone/census-join results back into one in-memory
            table) -- this flag controls only the FINAL grid-materialization
            step, and can be set independently of that one. A value here
            (e.g. `5`, matching `raster_to_h3_tiled`'s and this module's own
            default) is what Shanghai's `run.py` opts into; every other
            city leaves this `None` and keeps the existing single-
            GeoDataFrame `h3_grid.parquet` output byte-for-byte unchanged.
            Per-tile pmtiles generation, multi-source map rendering, and a
            polars-lazy cross-chunk stats path
            (`geohierarchy.chunked_h3_grid.read_h3_grid_chunked_columns`)
            build on top of this same per-tile file set but are wired
            separately (see `docs/H3_CHUNKED_PIPELINE_DESIGN.md` for the
            fuller architecture and what remains to fully consume this flag
            downstream of the grid-write step).
    """

    region: str = "global"
    walk_distance_steps: tuple[float, ...] = (400.0, 800.0, 1200.0, 2000.0)
    # Res 11 (~2,150 m^2 avg cell, ~46m across) instead of res 10 (~15,047 m^2, ~140m
    # across) -- res 10 cells were noticeably bigger than a typical dense-urban census
    # block, causing more centroid-matching mismatch between h3 cells and the census
    # geometries mapped onto/from them than a closer size match would.
    h3_resolution: int = 11
    # Every other resolution (11/9/7/5), not every one from 5-11: one pmtiles
    # archive is built per entry here (see `build_h3_by_resolution`'s
    # docstring), and adjacent H3 resolutions differ in cell area by ~7x --
    # close enough that a skipped resolution's zoom band just gets absorbed
    # by its neighbor (`_level_zoom_bands` in transitlos.map.build) without
    # a visibly coarser jump, at roughly half the tile-build cost/count.
    map_h3_resolutions: tuple[int, ...] = (5, 7, 9, 11)
    # ACS 5-year estimates are not published at the block level (only block group and
    # coarser) -- "block" here would silently fail every fetch (confirmed via a real
    # Census API error, "unknown/unsupported geography hierarchy"), voiding coverage
    # for every state in one request whenever it wasn't already cache-hit by luck.
    census_levels: tuple[str, ...] = ("blockgroup", "tract", "place", "county")
    score_bins: int = 30
    stats_h3_resolution: int = 7
    analysis_start_time: time = time(6, 0)
    analysis_end_time: time = time(22, 0)
    score_bin_width: float = 0.05
    street_simplify_distance: float = 30.0
    isochrone_chunk_h3_resolution: Optional[int] = None
    isochrone_chunk_buffer_m: float = 1000.0
    census_chunk_buffer_m: float = 200.0
    tile_chunk_buffer_m: float = 100.0
    chunked_h3_output: Optional[int] = None


GLOBAL_PARAMS = StudyParams()
