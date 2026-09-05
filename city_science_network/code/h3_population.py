"""WorldPop raster -> H3 population + level_of_service grid.

Thin orchestration over two reusable, configurable functions in the sibling
packages (not custom logic of its own):

- `pycensus.countries.worldwide.worldpop.h3.worldpop_raster_to_h3` resamples
  the WorldPop raster onto H3 cells -- area-weighted and mass-conserving
  (built on `geohierarchy.raster_resample`), not the old centroid-snap
  method, so a pixel straddling a cell boundary splits its value correctly
  instead of landing entirely in whichever cell its centroid happens to fall
  in. The raster is clipped to this study's AOI first (`_clip_to_aoi`) since
  the input here is the shared, whole-country raw WorldPop file.
- `geohierarchy.edges_to_h3_by_distance` assigns each cell the
  `Max()`-aggregated `level_of_service` among street edges within a configurable
  distance of its centroid (again vectorized, no Shapely/GEOS overlay).

The only logic that belongs to this study specifically is the population
redistribution: touching a street is a *mandatory* condition for a hexagon
to exist in the grid at all (and therefore to be drawn on the map), so cells
with no street within range are dropped -- their population is moved to the
**nearest** surviving street-touching cell, but only within a real distance
cap (`max_reassign_dist_m`, 500m): a roadless cell further than that from any
served cell has its population dropped outright rather than force-attached
to a far-away cell it doesn't actually border. `sum(population)` is
unchanged EXCEPT for this excluded remainder (logged) -- a deliberate,
usually small reduction, not the unbounded pile-up the uncapped version
allowed (2026-09-04 fix, live: Boston border hexagons up to ~467,737 people
in a single ~2,200 m^2 cell, from far-away roadless donors all sharing the
same nearest-but-still-very-far recipient).

Nearest-neighbour, not proportional-to-the-whole-study-area (which is what
this did before 2026-08-13): a global proportional split takes the people
living in a roadless pocket and smears them across the entire metro in
proportion to existing population, which silently inflates dense downtown
cells with residents who physically live 50 km away. Attributing them to the
nearest cell that *does* have a street keeps them where they actually are, to
within one cell's distance of the road that serves them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from geohierarchy import Max, edges_to_h3_by_distance
from pycensus.countries.worldwide.worldpop.h3 import worldpop_raster_to_h3_fast as worldpop_raster_to_h3
from pycensus.countries.worldwide.worldpop.loader import _clip_to_aoi


def _num_edges(streets_edges: gpd.GeoDataFrame | str | Path) -> int:
    """Row count without materializing geometry, when given a parquet path."""
    if isinstance(streets_edges, (str, Path)):
        return pq.ParquetFile(str(streets_edges)).metadata.num_rows
    return len(streets_edges)


def _spatially_sorted_path(streets_edges_path: Path, bucket_m: float = 15_000.0) -> Path:
    """Return a cached, spatially-sorted copy of `streets_edges_path`.

    `_edge_chunks` reads consecutive row ranges as chunks, and
    `population_and_access_to_h3` restricts each chunk's candidate H3 cells
    to that chunk's own padded bounding box. That restriction only pays off
    if consecutive edges in the file are geographically close together --
    which turned out false for Shanghai's real `access_edges.parquet`
    (10,249,415 edges, no city grouping in file order, confirmed by
    inspecting row-group bounding boxes directly): row groups 0-1 are each
    ~210km x 175km (one city's worth), but from row group 2 onward every
    row group's bbox already spans ~460km x 590km -- essentially the
    *entire* 14-city megaregion. That makes the per-chunk bbox restriction
    a no-op from the 3rd chunk on, so every subsequent chunk pays the full
    cross-region candidate-cell cost the restriction was added to avoid --
    reproducing the exact OOM it was meant to prevent (the live run OOM'd
    at precisely this point: "edges 2,250,000/10,249,415").

    This sorts edges once by a coarse raster-scan key -- a Y-band (of
    `bucket_m` height) then X within the band -- computed from each edge's
    bounding-box centre, and caches the result next to the source file so
    every subsequent chunk of `edge_chunk_size` consecutive rows is
    geographically compact, making the bbox restriction actually restrict.

    Memory-bounded by design: geometry is decoded one parquet batch at a
    time purely to compute each edge's (cx, cy) bbox-centre floats (8
    bytes/edge kept; the decoded Shapely batch is dropped immediately
    after), and the final reorder+write operates on the still-WKB-encoded
    Arrow table (never decoding all 10M+ geometries into Shapely objects
    at once).
    """
    cache_path = streets_edges_path.with_name(streets_edges_path.stem + "_spatial_sorted.parquet")
    if cache_path.is_file() and cache_path.stat().st_mtime >= streets_edges_path.stat().st_mtime:
        return cache_path

    print(f"[h3_population] spatially sorting {streets_edges_path.name} for locality-bounded chunking", flush=True)
    pf = pq.ParquetFile(str(streets_edges_path))
    geo_meta = json.loads(pf.schema_arrow.metadata[b"geo"])
    geom_col = geo_meta["primary_column"]

    cx_parts, cy_parts = [], []
    for batch in pf.iter_batches(batch_size=200_000, columns=[geom_col]):
        geoms = shapely.from_wkb(batch.column(geom_col).to_numpy(zero_copy_only=False))
        minx, miny, maxx, maxy = shapely.bounds(geoms).T
        cx_parts.append(((minx + maxx) * 0.5).astype(np.float64))
        cy_parts.append(((miny + maxy) * 0.5).astype(np.float64))
    cx = np.concatenate(cx_parts)
    cy = np.concatenate(cy_parts)
    band = np.floor(cy / bucket_m).astype(np.int64)
    sort_idx = np.lexsort((cx, band))  # primary key `band` (last), secondary `cx`

    # WKB bytes only, never decoded into Shapely objects here -- comparable
    # in size to the file on disk (hundreds of MB), not the multi-GB a full
    # `gpd.read_parquet` materialization would cost.
    table = pq.read_table(str(streets_edges_path))
    table = table.take(pa.array(sort_idx))
    tmp_path = cache_path.with_suffix(".tmp.parquet")
    pq.write_table(table, str(tmp_path))
    tmp_path.replace(cache_path)
    print(f"[h3_population]  -> cached spatially-sorted copy at {cache_path.name}", flush=True)
    return cache_path


def _edge_chunks(streets_edges: gpd.GeoDataFrame | str | Path, chunk_size: int, n_total: int):
    """Yield `streets_edges` in chunks of at most `chunk_size` rows.

    When `streets_edges` is a path, chunks are pulled straight from the
    parquet file's row groups via pyarrow and only *that chunk's* WKB column
    is decoded into real Shapely geometry -- the whole file is never
    materialized as a GeoDataFrame at once. This is what `edge_chunk_size`
    already implied should happen; previously callers had to eagerly
    `gpd.read_parquet` the entire file (full WKB geometry, ~3GB extra peak
    RSS on Boston's 4.9M-edge `access_edges.parquet`) before this loop ever
    started, defeating the point of chunking. Passing an already-loaded
    GeoDataFrame still works exactly as before (simple `.iloc` slicing).
    """
    if isinstance(streets_edges, (str, Path)):
        pf = pq.ParquetFile(str(streets_edges))
        geo_meta = json.loads(pf.schema_arrow.metadata[b"geo"])
        geom_col = geo_meta["primary_column"]
        crs = geo_meta["columns"][geom_col].get("crs")
        for batch in pf.iter_batches(batch_size=chunk_size or n_total):
            df = batch.to_pandas()
            df[geom_col] = shapely.from_wkb(df[geom_col].to_numpy())
            yield gpd.GeoDataFrame(df, geometry=geom_col, crs=crs)
    else:
        step = chunk_size or n_total
        for start in range(0, n_total, step):
            yield streets_edges.iloc[start : start + step]


def population_and_access_to_h3(
    worldpop_tif: str,
    aoi: gpd.GeoDataFrame,
    streets_edges: gpd.GeoDataFrame | str | Path,
    access_col: str = "level_of_service",
    resolution: int = 10,
    margin_m: float = 10.0,
    edge_chunk_size: int = 750_000,
    fallback_radius_multiplier: float | None = 5.0,
    cell_batch_size: int = 300_000,
) -> pl.DataFrame:
    """Resample WorldPop onto H3, keeping only cells within `margin_m` of a street edge.

    Args:
        worldpop_tif: Path to a WorldPop population GeoTIFF (whole-country;
            cropped internally to `aoi`'s exact polygon).
        aoi: Area of interest used to crop the raster read.
        streets_edges: Street-edges GeoDataFrame carrying `access_col`
            (e.g. from `transitlos.level_of_service.compute_level_of_service`),
            OR a path/str to a parquet file of the same (e.g. a cached
            `results/access_edges.parquet`). Passing a path lets chunked
            reading (see `edge_chunk_size`) apply to the *initial load* too,
            instead of eagerly materializing every edge's Shapely geometry
            up front regardless of chunk size.
        access_col: Column on `streets_edges` to assign onto each touching
            H3 cell (its max among edges within range).
        resolution: H3 resolution for the population/access grid.
        margin_m: Extra distance beyond one cell diameter still counted as
            "touching" a street -- forwarded to `edges_to_h3_by_distance`.
        fallback_radius_multiplier: Forwarded to `edges_to_h3_by_distance`'s
            own nearest-street fallback (see its docstring) -- lets a cell
            with no street within `margin_m` still pick up its nearest
            street's score directly instead of only ever inheriting a
            neighbour's population via the roadless-cell reassignment below.
            `None` disables the fallback, matching the pre-fallback behavior.
        edge_chunk_size: How many street edges to hand `edges_to_h3_by_distance`
            at a time. Since 2026-08-13 the street-edges input covers the
            *entire* AOI network (`UrbanAccessAnalyzer` now scores unreached
            streets 0 instead of dropping them, which took Boston from 1.7M to
            4.9M edge rows), and that function's peak memory scales with the
            number of edges handed to it -- enough to OOM a 30 GB machine in
            one call. Chunking is exactly equivalent here because the
            aggregation is `Max()`: the max over the per-chunk maxima is the
            max over all edges. Set to `0`/`None` for a single unchunked call.
        cell_batch_size: How many candidate H3 cells to hand
            `edges_to_h3_by_distance` in one call, inside the chunked branch
            (see `edge_chunk_size`). Real, watchdog-instrumented measurement
            against Shanghai's cached data (2026-08-30/31) showed the bbox
            restriction on `local_cells` (see below) does NOT bring a single
            edge-chunk's candidate-cell count down anywhere near enough for
            a 14-city megaregion: with the real 13,055,113-cell Shanghai
            candidate set, just ONE `edges_to_h3_by_distance` call (already
            bbox-restricted to that chunk's own edges) drove RSS from 3.7 GB
            to beyond 7.3 GB *before finishing*, i.e. millions of local cells
            still reach that function in one call. Its internal
            `cKDTree.query_ball_point` returns a Python list with one NumPy
            array object per queried cell (large fixed per-object overhead
            at this cell count) and then explodes that into a long-format
            (cell, matched sample point) row per match -- both scale with
            the candidate cell count handed to it, independent of
            `edge_chunk_size`. Splitting `local_cells` into batches of this
            size and merging each batch's result into the running
            `access_h3` accumulator immediately (same Max()-safe
            merge-as-you-go already used across edge chunks) bounds that
            function's peak working set to a fixed size regardless of how
            large a single edge chunk's own bounding box turns out to be.
            Set to `0`/`None` to disable (one call per edge chunk, prior
            behavior).

    Returns:
        Polars DataFrame with `h3_cell`, `population`, and `access_col`,
        one row per H3 cell within `margin_m` of a street edge -- touching a
        street is mandatory for a cell to be in this grid at all. Population
        from cells outside that range is moved onto the *nearest* kept cell
        (by cell-centroid distance), so `sum(population)` is unchanged.
    """
    # 2026-08-30: area-weighted, mass-conserving resample (replaces the old
    # centroid-based `h3_ops.from_raster_centroid`, which dropped a pixel's
    # entire value into whichever single h3 cell its centroid fell in --
    # wrong for any pixel straddling a cell boundary). `worldpop_raster_to_h3`
    # itself has no `aoi` param (it processes whatever raster it's given in
    # full) since it's meant to run on an already-clipped file -- `worldpop_tif`
    # here is the SHARED, whole-country raw raster, so clip it to this city's
    # AOI first (same `_clip_to_aoi` helper the WorldPop download path
    # already uses -- a real download-time optimization, not a hack: WorldPop
    # rasters run tens of thousands of pixels per side per country, and
    # resampling that in full for a single metro AOI would be enormously
    # wasteful and slow).
    # The destination filename MUST be AOI-specific, not just derived from
    # `worldpop_tif`'s own (shared, per-country) stem: `worldpop_tif` is one
    # whole-country raster reused by every city in that country (e.g. every
    # USA city shares `usa_pop_2025_CN_100m_R2025A_v1.tif` out of the same
    # shared `worldpop/` directory), so a fixed "<stem>_clipped_for_h3.tif"
    # name collides across cities. Real live bug (2026-08-30, this v2
    # rebuild pass): Boston ran first and wrote its Boston-clipped raster to
    # that fixed name; San Francisco ran later, saw the file already existed
    # (`_clip_to_aoi(..., overwrite=False)`), and silently reused Boston's
    # clipped raster -- its `pop_h3` cells all came back centered on Boston,
    # MA (41-43degN) while its street edges were San Francisco's (37-38degN),
    # so the street-touching join matched zero cells and the population
    # conservation assert correctly caught it (8.4M -> 0.0). Hashing the
    # AOI's own bounds into the filename makes every distinct AOI -- even
    # sharing the same country raster and the same output directory -- get
    # its own clipped file, with no need to plumb a per-city cache directory
    # through this function's signature.
    import hashlib

    aoi_wgs84 = aoi.to_crs(4326) if aoi.crs is not None else aoi
    aoi_hash = hashlib.sha1(np.asarray(aoi_wgs84.total_bounds).round(6).tobytes()).hexdigest()[:10]
    clipped_tif = str(
        Path(worldpop_tif).with_name(f"{Path(worldpop_tif).stem}_clipped_for_h3_{aoi_hash}.tif")
    )
    clipped_tif = _clip_to_aoi(worldpop_tif, aoi, clipped_tif, overwrite=False)

    # Bug fix (2026-09-04, live user report -- large population circles
    # still showing right at the AOI boundary even after the cell-level
    # AOI filter below; confirmed live on Concepcion: 9 of the top 10
    # highest-population h3 cells sit within ~22m of the AOI boundary
    # line). `_clip_to_aoi` above is only ever a bounding-BOX crop (a cheap
    # rectangular window read -- see its own docstring), never a polygon
    # mask. `worldpop_raster_to_h3`'s resampling is real, area-weighted,
    # mass-conserving math over whatever pixels the raster hands it -- for
    # a hexagon straddling the AOI edge, that includes real population
    # from pixels OUTSIDE the true polygon (the bbox crop kept them), which
    # is exactly what inflated boundary cells even after cells outside the
    # AOI were dropped (that filter only removes whole out-of-AOI cells --
    # a KEPT cell that happens to straddle the edge still had its own
    # population value computed from a mix of inside- and outside-AOI
    # pixels). Masking the raster to the real, unpadded AOI polygon here --
    # setting every pixel outside it to nodata, not just to the bbox's
    # rectangular window -- ensures a straddling cell's resampled value
    # only ever reflects the portion of it that's actually inside the AOI.
    masked_tif = clipped_tif.replace("_clipped_for_h3_", "_masked_for_h3_")
    if not os.path.isfile(masked_tif):
        import rasterio
        import rasterio.mask

        with rasterio.open(clipped_tif) as _src:
            _aoi_native = aoi_wgs84.to_crs(_src.crs) if _src.crs is not None else aoi_wgs84
            _masked_data, _masked_transform = rasterio.mask.mask(
                _src, _aoi_native.geometry, crop=False, nodata=0, filled=True
            )
            _profile = _src.profile.copy()
            _profile.update(nodata=0)
        with rasterio.open(masked_tif, "w", **_profile) as _dst:
            _dst.write(_masked_data)
    pop_h3 = worldpop_raster_to_h3(masked_tif, resolution=resolution, value_col="population")

    # Bug fix (2026-09-04, live user report -- "all maps seem to have this
    # boundary issue... you might be doing bbox to crop but not setting to
    # 0/none all cells outside the AOI... use the real aoi without any
    # buffer"). Confirmed: `_clip_to_aoi` only crops the raster to a padded
    # bounding BOX (a cheap rectangular window read, not a polygon mask --
    # see that function's own docstring), so every h3 cell resampled from
    # it -- including ones geometrically outside this city's real AOI
    # polygon but inside the padded rectangle -- keeps its real WorldPop
    # population. Nothing downstream in this module ever re-filtered by the
    # true polygon (only by street proximity), so a border cell just
    # outside the intended study area could carry real population from
    # neighbouring, out-of-scope territory. Filtered here, against the
    # real (unpadded, unbuffered) AOI polygon -- the pad above is a
    # processing-only concern (avoiding a truncated raster read), never
    # what's actually counted.
    import h3ronpy as _h3ronpy
    import h3ronpy.vector as _h3v

    _cell_ids = _h3ronpy.cells_parse(pop_h3["h3_cell"].to_list())
    _latlng = _h3v.cells_to_coordinates(_cell_ids)
    _centroids = gpd.GeoDataFrame(
        {"h3_cell": pop_h3["h3_cell"].to_list()},
        geometry=gpd.points_from_xy(np.asarray(_latlng["lng"]), np.asarray(_latlng["lat"])),
        crs=4326,
    )
    _aoi_union = aoi_wgs84.geometry.union_all()
    _within_aoi = set(_centroids.loc[_centroids.geometry.within(_aoi_union), "h3_cell"])
    _n_before_aoi_filter = pop_h3.height
    pop_h3 = pop_h3.filter(pl.col("h3_cell").is_in(_within_aoi))
    print(
        f"[h3_population] real-AOI polygon filter (no buffer): kept {pop_h3.height:,}/"
        f"{_n_before_aoi_filter:,} cells",
        flush=True,
    )

    total_before = float(pop_h3["population"].sum())

    cells = pop_h3["h3_cell"].to_list()
    n_edges = _num_edges(streets_edges)
    if not edge_chunk_size or edge_chunk_size >= n_edges:
        whole = next(iter(_edge_chunks(streets_edges, 0, n_edges)))
        access_h3 = edges_to_h3_by_distance(
            whole, cells, access_col, resolution, margin_m, agg=Max(),
            fallback_radius_multiplier=fallback_radius_multiplier,
        )
        del cells  # dead from here on; see the matching `del` in the chunked branch below
    else:
        # Max() over chunks == Max() over everything; see `edge_chunk_size`.
        #
        # `edges_to_h3_by_distance` re-tests *every* candidate cell handed to
        # it against each chunk's edges (parsing/reprojecting the whole
        # `cells` list and running a KD-tree radius query over all of it),
        # even though a 750K-edge chunk only ever touches cells near that
        # chunk's own geographic extent. For a single-city study that waste
        # is small; for a multi-city megaregion like Shanghai (14 cities'
        # worth of H3 cells in `cells` at once) it means every chunk pays
        # the full cross-region candidate-cell cost for edges that can only
        # ever match a local fraction of it -- this is what balloons peak
        # memory well past what the edge count alone would predict. Bound
        # each chunk's candidate cells to its own bounding box (padded by
        # the same search radius `edges_to_h3_by_distance` uses) so a chunk
        # confined to one city only pays for that city's cells.
        import h3
        import h3ronpy
        import h3ronpy.vector as h3v

        # 2026-08-31: vectorized (h3ronpy) instead of a pure-Python
        # `[h3.cell_to_latlng(c) for c in cells]` loop -- real measurement
        # against Shanghai's actual 13,055,113-cell candidate set showed
        # that loop alone added ~2.1 GB RSS (1.6 GB -> 3.7 GB) building an
        # intermediate list of 13M+ Python tuples before the final
        # `np.array` conversion, purely to compute a one-time lat/lng
        # lookup this same package already does vectorized elsewhere (see
        # `edges_to_h3_by_distance`'s own `h3v.cells_to_coordinates` call).
        cell_ids = h3ronpy.cells_parse(cells)
        _latlng = h3v.cells_to_coordinates(cell_ids)
        cell_lat = np.asarray(_latlng["lat"])
        cell_lng = np.asarray(_latlng["lng"])
        edge_len_m = h3.average_hexagon_edge_length(resolution, unit="m")
        pad_deg = (margin_m + 2 * edge_len_m) / 111_000.0 + 0.01  # generous, cheap upper bound

        # The bbox restriction below only shrinks the candidate-cell set if
        # consecutive rows in `streets_edges` are geographically close --
        # true for a plain per-city file, but false for a scattered/
        # unsorted multi-city megaregion file (see `_spatially_sorted_path`
        # docstring: verified true of Shanghai's real `access_edges.parquet`,
        # where this was the actual cause of an OOM the bbox restriction was
        # supposed to prevent). Sort-once-and-cache when reading from a path.
        chunk_source = streets_edges
        if isinstance(streets_edges, (str, Path)):
            chunk_source = _spatially_sorted_path(Path(streets_edges))

        # Reduce (group_by/max) after every chunk instead of accumulating
        # every chunk's un-merged result in a `parts` list for one final
        # concat+group_by at the end. The megaregion's per-chunk cell-row
        # counts grow roughly linearly with edges processed (verified live:
        # ~490K cell rows/chunk, climbing to ~7.3M accumulated by edge
        # 7.5M/10.25M) because -- unlike a single-city study -- Shanghai's
        # AOI cells are shared across many nearby chunks, so held-but-
        # unmerged duplicates across chunks pile up for the entire run.
        # Merging as we go keeps resident memory bounded by the number of
        # *unique* touched cells seen so far (which converges well below
        # the full AOI cell count) rather than the sum across all chunks --
        # this is what OOM-killed the run at edge 7.5M/10.25M even after
        # the spatial-sort fix (15.0G RSS against a 20G cgroup cap).
        def _merge(base: pl.DataFrame | None, part: pl.DataFrame) -> pl.DataFrame:
            part = pl.from_pandas(part) if not isinstance(part, pl.DataFrame) else part
            if base is None:
                return part
            if part.is_empty():
                return base
            return (
                pl.concat([base, part], how="vertical")
                .group_by("h3_cell")
                .agg(pl.col(access_col).max())
            )

        access_h3: pl.DataFrame | None = None
        processed = 0
        for chunk in _edge_chunks(chunk_source, edge_chunk_size, n_edges):
            chunk_ll = chunk if chunk.crs is None or chunk.crs.to_epsg() == 4326 else chunk.to_crs(4326)
            minx, miny, maxx, maxy = chunk_ll.total_bounds
            mask = (
                (cell_lng >= minx - pad_deg)
                & (cell_lng <= maxx + pad_deg)
                & (cell_lat >= miny - pad_deg)
                & (cell_lat <= maxy + pad_deg)
            )
            local_cells = [c for c, m in zip(cells, mask) if m]
            # `fallback_radius_multiplier`'s search radius is already covered
            # by `pad_deg`'s own margin (`margin_m + 2 * edge_len_m`) only for
            # the default multiplier; a caller-supplied multiplier larger than
            # that isn't specially widened here, so an out-of-chunk fallback
            # match can be missed for a cell right at a chunk boundary -- it
            # simply won't get a fallback value from *this* chunk, same as if
            # it had no street nearby here at all. Harmless under Max(): if
            # another chunk's bbox does cover it, that chunk supplies the real
            # value; otherwise the cell falls back to the pre-existing
            # roadless-cell reassignment below, same as before this param
            # existed.
            #
            # 2026-08-31: `local_cells` (bbox-restricted to this one edge
            # chunk) is STILL far too large to hand `edges_to_h3_by_distance`
            # in one call for a megaregion like Shanghai -- real,
            # watchdog-instrumented measurement against Shanghai's cached
            # data showed a single such call (already bbox-restricted)
            # drove RSS from 3.7 GB past 7.3 GB before even returning,
            # because that function's internal `cKDTree.query_ball_point`
            # produces one Python/NumPy array object per candidate cell
            # (huge fixed overhead in the millions) and then explodes
            # matches into a long-format row per (cell, sample point) pair
            # -- both scale with candidate cell count, not `edge_chunk_size`.
            # Sub-batch `local_cells` too, merging each batch's result into
            # `access_h3` immediately via the same Max()-safe merge used
            # across edge chunks, so peak memory is bounded by
            # `cell_batch_size` regardless of how large a single edge
            # chunk's own bounding box (and therefore candidate-cell count)
            # turns out to be.
            if not cell_batch_size or len(local_cells) <= cell_batch_size:
                cell_batches = [local_cells]
            else:
                cell_batches = [
                    local_cells[i : i + cell_batch_size]
                    for i in range(0, len(local_cells), cell_batch_size)
                ]
            for cell_batch in cell_batches:
                part = edges_to_h3_by_distance(
                    chunk, cell_batch, access_col, resolution, margin_m, agg=Max(),
                    fallback_radius_multiplier=fallback_radius_multiplier,
                )
                access_h3 = _merge(access_h3, part)
            processed += len(chunk)
            print(
                f"[h3_population] edges {processed}/{n_edges} "
                f"-> {access_h3.height} unique cell rows so far "
                f"({len(local_cells):,} local candidate cells in {len(cell_batches)} batch(es))",
                flush=True,
            )
        # 2026-08-31: `cells` (an 11.5M+-element Python list of h3 id
        # strings, built once before the loop at the top of this function)
        # and `cell_lat`/`cell_lng` (the matching float64 coordinate arrays
        # used only for this loop's per-chunk bbox mask) are never used
        # again after the loop ends -- but nothing dropped the references,
        # so they stayed resident through the entire redistribution section
        # below on every previous run. Real, watchdog-instrumented
        # measurement against Shanghai's actual current candidate set
        # (11,510,545 cells, re-derived from today's real clipped WorldPop
        # raster) showed the redistribution section's OWN peak memory is
        # only ~5GB even fully unoptimized -- nowhere near enough to
        # explain the observed OOM on a 20GB-capped run by itself. That
        # means whatever OOM'd this run was memory already resident by the
        # time the loop finished, on top of which the redistribution
        # section's own allocations (chiefly the `pop_h3.join(...)` below,
        # the single largest new allocation measured, +~1.7GB) tipped it
        # over. Freeing this dead ~11.5M-string list + two float64 arrays
        # (a real, if partial, contributor -- call it low-single-digit-GB)
        # before that join runs is strictly correct (nothing below uses
        # them) and costs nothing.
        del cells, cell_lat, cell_lng
        import gc

        gc.collect()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass  # best-effort glibc arena release; harmless if unavailable
    # A Python `set` of every touching cell id (previously built via
    # `set(access_h3["h3_cell"].to_list())`) materializes one Python `str`
    # object plus set-entry overhead per row -- expensive at multi-million
    # row scale. A polars-native join stays in Arrow/polars memory the
    # whole time, which is dramatically more compact than a Python set of
    # that many strings.
    #
    # 2026-08-31: real, watchdog-instrumented measurement against
    # Shanghai's actual current candidate set (11,510,545 cells,
    # re-derived from today's real clipped WorldPop raster -- NOT the
    # unverified "25.7M accumulated access_h3 rows" figure quoted in an
    # earlier debugging pass, which predates today's AOI-hash clipped-
    # raster fix and could not be reproduced against today's real data)
    # showed this join is the single largest new allocation in this whole
    # post-loop section (+~1.7GB in the measured scenario). Using
    # semi/anti joins for `kept`/`dropped` instead of building a full
    # `on_street`-augmented copy of `pop_h3` and then running two separate
    # `.filter()` passes over it avoids materializing that extra
    # full-width intermediate frame at all -- polars can compute each
    # semi/anti join directly against `pop_h3` without an intermediate
    # "on_street" column.
    touching = access_h3.select("h3_cell").unique()
    kept = pop_h3.join(touching, on="h3_cell", how="semi")
    dropped = pop_h3.join(touching, on="h3_cell", how="anti").filter(pl.col("population") > 0)
    removed_total = float(dropped["population"].sum())
    # Set below (2026-09-04 fix) when a roadless cell has no street-touching
    # neighbour within `max_reassign_dist_m` -- its population is dropped,
    # not force-attached to a far-away cell, so the conservation check a
    # few lines down must expect the total to shrink by exactly this much
    # rather than staying byte-for-byte unchanged.
    excluded_total = 0.0

    if removed_total > 0.0 and not kept.is_empty():
        # Move each roadless cell's people to the nearest cell that does have
        # a street, rather than smearing them across the whole study area --
        # see the module docstring.
        import h3ronpy
        import h3ronpy.vector as h3v
        from scipy.spatial import cKDTree

        def _latlng(cells_list):
            # Vectorized (h3ronpy), not a pure-Python `h3.cell_to_latlng`
            # loop -- `kept_cells`/`dropped` can be single-digit millions of
            # rows at Shanghai's scale (real measured: pop_h3 11,510,545
            # rows total), and the same per-cell Python-loop cost already
            # confirmed live (+2.1GB baseline, see this function's
            # `cell_batch_size` fix above) applies here too.
            cell_ids = h3ronpy.cells_parse(list(cells_list))
            latlng = h3v.cells_to_coordinates(cell_ids)
            lat = np.asarray(latlng["lat"], dtype=float)
            lng = np.asarray(latlng["lng"], dtype=float)
            # Equirectangular projection about the grid's mean latitude: this
            # is only ever used for a *nearest-neighbour* query over cells a
            # few hundred metres apart, where the distortion is far below one
            # cell width, and it avoids a full geodesic index.
            lat0 = np.deg2rad(lat.mean()) if len(lat) else 0.0
            return np.column_stack([np.deg2rad(lng) * np.cos(lat0), np.deg2rad(lat)])

        kept_cells = kept["h3_cell"].to_list()
        tree = cKDTree(_latlng(kept_cells))
        dist, nearest = tree.query(_latlng(dropped["h3_cell"].to_list()))

        # Bug fix (2026-09-04, live user report -- Boston border cells
        # showing populations in the hundreds of thousands inside a single
        # ~2,200 m^2 hexagon, up to ~467,737 people/cell). The module's own
        # docstring says this should keep a roadless cell's people "to
        # within one cell's distance of the road that serves them" -- but
        # nothing in the code actually enforced a distance limit: EVERY
        # roadless cell, no matter how far from the served street network,
        # got folded into its single globally-nearest street-touching cell.
        # Cells beyond the edge of the real, connected street network (e.g.
        # rural/water/unreachable fringe near the AOI boundary) could all
        # share the same "nearest" recipient -- often itself a border cell,
        # since that's what's geographically closest to a large swath of
        # unserved land outside it -- so thousands of small, spread-out
        # donor populations piled onto one tiny hexagon.
        # `max_reassign_dist_m` caps this reassignment. A one-hexagon-width
        # cap (~57m at res-11, tried 2026-09-04) turned out too tight in
        # practice -- live on Concepcion, it dropped 96,424 people (8.7% of
        # the city's population) as "unreachable", cells whose real street
        # is just a bit further than one hexagon away in areas with sparser
        # OSM street coverage than Andorra's. Explicit follow-up: "assign
        # the nearest street up to 300m. But never do this with cells
        # outside the aoi" -- 300m is generous enough to cover normal OSM
        # coverage gaps without reintroducing the original far-away-donor
        # pile-up bug (this is still a per-cell single-nearest-neighbour
        # cap, nothing like the old fully-unbounded version), and the "never
        # outside the aoi" half is already guaranteed upstream (`pop_h3` is
        # filtered to the real AOI polygon before this function ever runs --
        # see the real-AOI polygon filter above). A roadless cell further
        # than 300m from any served cell has its population DROPPED (not
        # force-attached to a far-away cell it doesn't actually border) --
        # a deliberate, honest reduction in `sum(population)`, in exchange
        # for never fabricating an impossible single-hexagon population
        # again.
        max_reassign_dist_m = 300.0
        earth_radius_m = 6_371_000.0
        max_reassign_dist_rad = max_reassign_dist_m / earth_radius_m
        within_range = dist <= max_reassign_dist_rad

        dropped_pop = dropped["population"].to_numpy().astype(float)
        excluded_total = float(dropped_pop[~within_range].sum())
        excluded_count = int((~within_range).sum())

        gained = (
            pl.DataFrame(
                {
                    "h3_cell": [kept_cells[i] for i in nearest[within_range]],
                    "_gained": dropped_pop[within_range],
                }
            )
            .group_by("h3_cell")
            .agg(pl.col("_gained").sum())
        )
        kept = (
            kept.join(gained, on="h3_cell", how="left")
            .with_columns((pl.col("population") + pl.col("_gained").fill_null(0.0)).alias("population"))
            .drop("_gained")
        )
        moved_total = removed_total - excluded_total
        print(
            f"[h3_population] moved {moved_total:,.0f} people from {int(within_range.sum()):,} roadless "
            f"cells onto their nearest street-touching neighbour within {max_reassign_dist_m:.0f}m "
            f"({gained.height:,} recipients); dropped {excluded_total:,.0f} people from {excluded_count:,} "
            f"roadless cells with no street-touching neighbour that close (unreachable/rural fringe)",
            flush=True,
        )

    result = kept.join(access_h3, on="h3_cell", how="left")

    total_after = float(result["population"].sum()) if not result.is_empty() else 0.0
    expected_after = total_before - excluded_total
    if excluded_total > 0.0:
        print(
            f"[h3_population] population conserved (minus {excluded_total:,.2f} intentionally "
            f"dropped, see above): {total_before:,.2f} -> {total_after:,.2f} across "
            f"{result.height:,} street-touching cells",
            flush=True,
        )
    else:
        print(
            f"[h3_population] population conserved: {total_before:,.2f} -> {total_after:,.2f} "
            f"across {result.height:,} street-touching cells",
            flush=True,
        )
    assert abs(expected_after - total_after) < 1e-6 * max(total_before, 1.0), (
        f"Population redistribution changed the total by more than the intentionally-dropped "
        f"remainder: expected {expected_after} ({total_before} - {excluded_total} excluded), got {total_after}"
    )
    return result
