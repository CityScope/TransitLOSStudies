"""Per-state tile writers for the nationwide map's heavy levels.

A nationwide map cannot hold 45M h3 res-11 cells, 8M census blocks and ~50M street edges in one
process (it was OOM-killed at ~90-126 GB). Instead those levels are tiled ONE STATE AT A TIME into
`<tiles>/<kind>/<level>/chunk_<State>.pmtiles` (MVT layer name = level name); the MapLibre page
treats the chunk files of a level as one continuous layer and only fetches the ones in view -- which
also means streets and other fine layers load lazily at high zoom.

`transitlos.map.build.build_city_map(external_tile_writers=...)` calls these writers instead of
tiling the (placeholder) in-memory frames; they receive exactly the id/property columns and zoom
range the in-process path would have used, so the tiles are interchangeable with the normal ones.
"""
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import shapely

logger = logging.getLogger(__name__)

#: h3 resolutions / census levels whose tiles are written per state (everything finer than a county).
EXTERNAL_H3_RESOLUTIONS = (7, 8, 9, 10, 11)  # res 7 per state too: nationwide it split into 5,000+ chunk sources
EXTERNAL_CENSUS_LEVELS = ("blockgroup", "block")


from .los_fill import fill_zero_los  # noqa: E402

def _state_name(d: Path) -> str:
    return d.name


def h3_polygons(cells) -> np.ndarray:
    """Shapely polygons of H3 cells, built in one vectorised call per vertex count (6 for hexagons, 5 for pentagons, more where a cell crosses an icosahedron edge)."""
    rings = [h3.cell_to_boundary(c) for c in cells]
    out = np.empty(len(rings), dtype=object)
    sizes = np.fromiter((len(r) for r in rings), dtype=np.int32, count=len(rings))
    for n in np.unique(sizes):
        idx = np.flatnonzero(sizes == n)
        latlng = np.asarray([rings[i] for i in idx], dtype=np.float64)          # (m, n, 2) lat, lon
        out[idx] = shapely.polygons(latlng[:, :, ::-1])
    return out


def h3_points(cells):
    ll = np.asarray([h3.cell_to_latlng(c) for c in cells], dtype=np.float64).reshape(-1, 2)
    return shapely.points(ll[:, 1], ll[:, 0])


def _h3_frame(state_dir: Path, res: int, circles: bool) -> Optional[gpd.GeoDataFrame]:
    path = state_dir / f"h3_res{res}.geoparquet"
    if not path.is_file():
        return None
    df = pd.read_parquet(path).rename(columns={"h3": "h3_cell"})
    cells = df["h3_cell"].tolist()
    geoms = h3_points(cells) if circles else h3_polygons(cells)
    return gpd.GeoDataFrame(df, geometry=geoms, crs=4326)


def _dev_frame(state_dir: Path, offset: int) -> Optional[gpd.GeoDataFrame]:
    """One folder's flagged res-8 cells (equity_flag set, each against its OWN county's regression line) for the development overlay."""
    from .equity import state_flagged_cells

    df = state_flagged_cells(state_dir)
    if df.empty:
        return None
    df = df.rename(columns={"h3": "h3_cell"}).reset_index(drop=True)
    return gpd.GeoDataFrame(df[["h3_cell", "equity_flag", "level_of_service"]], geometry=h3_polygons(df["h3_cell"].tolist()), crs=4326)


def _census_frame(state_dir: Path, level: str) -> Optional[gpd.GeoDataFrame]:
    frames = [fill_zero_los(gpd.read_parquet(p), state_dir) for p in sorted((state_dir / "counties").glob(f"*/{level}.geoparquet"))]
    return pd.concat(frames, ignore_index=True) if frames else None


def _street_frame(state_dir: Path, offset: int) -> Optional[gpd.GeoDataFrame]:
    frames = [gpd.read_parquet(p) for p in sorted((state_dir / "counties").glob("*/street.geoparquet"))]
    if not frames:
        return None
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)
    gdf["edge_id"] = np.arange(offset, offset + len(gdf), dtype="int64")
    return gdf


def _write_chunk(gdf, level_name, id_col, property_cols, min_zoom, max_zoom, tiles_dir: Path, chunk: str) -> None:
    """Tile one state's rows of a level into `<tiles_dir>/<level>/chunk_<chunk>.pmtiles`."""
    from geohierarchy.maps.folium.fast_tiles import generate_pmtiles_from_geoparquet

    gdf = gdf.to_crs(4326) if gdf.crs is not None and gdf.crs.to_epsg() != 4326 else gdf
    # The hierarchy stores ids under an internal `_<level>_<orig>` name; keep both spellings.
    orig = id_col[len(level_name) + 2:] if id_col.startswith(f"_{level_name}_") else id_col
    if orig in gdf.columns:
        gdf = gdf.assign(**{id_col: gdf[orig]})
    keep = [c for c in dict.fromkeys([id_col, orig, *property_cols]) if c in gdf.columns]
    geom = gdf.geometry.name
    gdf = gdf[[*keep, geom]].copy()
    for c in keep:
        if gdf[c].dtype == "float64":
            gdf[c] = gdf[c].astype("float32")
    out_dir = tiles_dir / level_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(tiles_dir)) as tmp:
        src = Path(tmp) / "rows.parquet"
        gdf.to_parquet(src)
        generate_pmtiles_from_geoparquet(
            src, out_dir / f"chunk_{chunk}.pmtiles", layer_name=level_name,
            min_zoom=int(min_zoom), max_zoom=int(max_zoom), property_cols=[c for c in keep],
        )


#: Peak memory of one state's tiling job, as a multiple of its input parquet bytes (measured in the nationwide build: California's res-11 hexagons, 0.27 GB of
#: parquet, peaked at 12.8 GB; points and lines are lighter than polygons). The scheduler also checks the real free memory before every job.
MEMORY_FACTOR = {"edges": 30.0, "census_block": 14.0, "census_blockgroup": 14.0}   # streets: CA (1.7 GB parquet) peaked near 40 GB
DEFAULT_MEMORY_FACTOR = 48.0
RESERVE_BYTES = 6e9   # never start a job that would leave less than this free


def _input_files(job) -> List[Path]:
    kind, arg, d = job
    if kind == "h3":
        return [d / f"h3_res{arg[0]}.geoparquet"]
    if kind == "census":
        return list((d / "counties").glob(f"*/{arg}.geoparquet")) + [d / "h3_res11.geoparquet"]
    if kind == "streets":
        return list((d / "counties").glob("*/street.geoparquet"))
    return [d / "h3_res8.geoparquet"]


def _job_bytes(job) -> int:
    return sum(p.stat().st_size for p in _input_files(job) if p.is_file())


def _street_rows(d: Path) -> int:
    import pyarrow.parquet as pq

    return sum(pq.read_metadata(p).num_rows for p in sorted((d / "counties").glob("*/street.geoparquet")))


def _load(job, offset):
    kind, arg, d = job
    if kind == "h3":
        return _h3_frame(d, arg[0], arg[1])
    if kind == "census":
        return _census_frame(d, arg)
    if kind == "streets":
        return _street_frame(d, offset)
    return _dev_frame(d, offset)


def _run_job(job, offset, level_name, id_col, property_cols, min_zoom, max_zoom, tiles_dir):
    """One state's chunk of one level (runs in a worker process)."""
    d = job[2]
    try:
        gdf = _load(job, offset)
    except Exception as exc:  # noqa: BLE001
        return d.name, -1, f"failed to load ({exc})"
    if gdf is None or not len(gdf):
        return d.name, 0, ""
    _write_chunk(gdf, level_name, id_col, property_cols, min_zoom, max_zoom, Path(tiles_dir), _state_name(d))
    return d.name, len(gdf), ""


def _die_with_parent() -> None:
    """Pool initializer: on Linux the worker gets SIGTERM when its parent dies, so a killed build never leaves tilers running (and holding memory)."""
    try:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG = 1
    except Exception:  # noqa: BLE001 -- not Linux: nothing to do
        pass


def _workers_and_budget():
    import os

    import psutil

    cpus = os.cpu_count() or 2
    workers = int(os.environ.get("USA_TILE_WORKERS", max(1, min(cpus - 2, 12))))
    free = psutil.virtual_memory().available
    room = _cgroup_room()
    if room is not None:   # running in a memory-capped scope: the cap, not the machine, is the limit
        free = min(free, room)
    budget = float(os.environ.get("USA_TILE_MEMORY_GB", 0)) * 1e9 or 0.7 * free
    return workers, budget


def _cgroup_room():
    """Bytes left under this process's cgroup memory cap (cgroup v2 `memory.max` minus `memory.current`), or None when uncapped."""
    try:
        rel = next(line.split("::", 1)[1].strip() for line in open("/proc/self/cgroup") if line.startswith("0::"))
        d = Path("/sys/fs/cgroup") / rel.lstrip("/")
        cap = (d / "memory.max").read_text().strip()
        if cap == "max":
            return None
        return max(0, int(cap) - int((d / "memory.current").read_text()))
    except Exception:  # noqa: BLE001 -- no cgroup v2: the machine's free memory is the limit
        return None


def _make_writer(kind: str, arg, state_dirs: List[Path]):
    """Writer for one level: every state's chunk is tiled in a pool of worker processes, largest states first, never starting a job whose estimated peak memory would
    push the running total over the budget (70 % of the free memory, or USA_TILE_MEMORY_GB); USA_TILE_WORKERS=1 tiles them one after another as before."""
    def writer(level_name, id_col, property_cols, min_zoom, max_zoom, tiles_dir):
        import multiprocessing as mp
        import time
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

        import psutil

        level_dir = Path(tiles_dir) / level_name
        if level_dir.is_dir():
            shutil.rmtree(level_dir)  # drop chunks of a previous build
        jobs = [(kind, arg, d) for d in state_dirs]
        offsets, off = {}, 0
        for j in jobs:   # street edge ids continue from state to state, exactly as when the states were tiled in order
            offsets[j[2]] = off
            if kind == "streets":
                off += _street_rows(j[2])
        factor = MEMORY_FACTOR.get(level_name, DEFAULT_MEMORY_FACTOR)
        cost = {j[2]: _job_bytes(j) * factor for j in jobs}
        workers, budget = _workers_and_budget()
        t0 = time.time()
        args = (level_name, id_col, list(property_cols), min_zoom, max_zoom, str(tiles_dir))
        if workers <= 1 or len(jobs) == 1:
            for j in jobs:
                name, n, err = _run_job(j, offsets[j[2]], *args)
                (logger.error if err else logger.info)(f"{level_name}: {name} {err or f'tiled ({n:,} rows)'}")
            return
        pending = sorted(jobs, key=lambda j: -cost[j[2]])
        running, used = {}, 0.0
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_die_with_parent) as ex:
            while pending or running:
                while pending and len(running) < workers:
                    free = psutil.virtual_memory().available
                    j = next((j for j in pending if not running or (used + cost[j[2]] <= budget and cost[j[2]] + RESERVE_BYTES <= free)), None)
                    if j is None:
                        break
                    pending.remove(j)
                    running[ex.submit(_run_job, j, offsets[j[2]], *args)] = cost[j[2]]
                    used += cost[j[2]]
                done, _ = wait(list(running), return_when=FIRST_COMPLETED, timeout=5)   # wake up every few seconds: free memory changes while jobs run
                if not done:
                    continue
                for f in done:
                    used -= running.pop(f)
                    name, n, err = f.result()
                    (logger.error if err else logger.info)(f"{level_name}: {name} {err or f'tiled ({n:,} rows)'}")
        logger.info(f"{level_name}: {len(jobs)} states tiled in {time.time() - t0:.0f}s ({workers} workers)")

    return writer


def make_dev_writer(state_dirs: Iterable[Path]):
    """Writer for the development overlay: `tiles/development/h3_8/chunk_<State>.pmtiles` (51 sources, not thousands)."""
    return _make_writer("dev", None, [Path(d) for d in state_dirs])


def make_writers(state_dirs: Iterable[Path], include_streets: bool = True) -> Dict[Tuple[str, str], Callable]:
    """`{(map_kind, level_name): writer}` for `build_city_map(external_tile_writers=...)`."""
    dirs = [Path(d) for d in state_dirs]
    writers: Dict[Tuple[str, str], Callable] = {}
    for res in EXTERNAL_H3_RESOLUTIONS:
        writers[("hexagons", f"h3_{res}")] = _make_writer("h3", (res, False), dirs)
        writers[("circles", f"h3_{res}")] = _make_writer("h3", (res, True), dirs)
    for level in EXTERNAL_CENSUS_LEVELS:
        writers[("census", f"census_{level}")] = _make_writer("census", level, dirs)
    writers[("development", "h3_8")] = make_dev_writer(dirs)
    if include_streets:
        writers[("streets", "edges")] = _make_writer("streets", None, dirs)
    return writers
