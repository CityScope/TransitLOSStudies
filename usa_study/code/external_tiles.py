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
EXTERNAL_H3_RESOLUTIONS = (7, 9, 11)  # res 7 per state too: nationwide it split into 5,000+ chunk sources
EXTERNAL_CENSUS_LEVELS = ("blockgroup", "block")


def _state_name(d: Path) -> str:
    return d.name


def _h3_frame(state_dir: Path, res: int, circles: bool) -> Optional[gpd.GeoDataFrame]:
    path = state_dir / f"h3_res{res}.geoparquet"
    if not path.is_file():
        return None
    df = pd.read_parquet(path).rename(columns={"h3": "h3_cell"})
    cells = df["h3_cell"].tolist()
    if circles:
        geoms = shapely.points([h3.cell_to_latlng(c)[1] for c in cells], [h3.cell_to_latlng(c)[0] for c in cells])
    else:
        geoms = [shapely.Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(c)]) for c in cells]
    return gpd.GeoDataFrame(df, geometry=geoms, crs=4326)


def _dev_frame(state_dir: Path, offset: int) -> Optional[gpd.GeoDataFrame]:
    """One state's flagged res-7 cells (equity_flag set) for the development overlay -- unflagged cells draw nothing."""
    from .equity import flags_from_thresholds, load_thresholds

    path = state_dir / "h3_res7.geoparquet"
    if not path.is_file():
        return None
    df = pd.read_parquet(path, columns=["h3", "population", "level_of_service"])
    df["equity_flag"] = flags_from_thresholds(df, load_thresholds())
    df = df[df["equity_flag"].notna()].rename(columns={"h3": "h3_cell"}).reset_index(drop=True)
    geoms = [shapely.Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(c)]) for c in df["h3_cell"]]
    return gpd.GeoDataFrame(df[["h3_cell", "equity_flag", "level_of_service"]], geometry=geoms, crs=4326)


def make_dev_writer(state_dirs: Iterable[Path]):
    """Writer for the development overlay: `tiles/development/h3_7/chunk_<State>.pmtiles` (51 sources, not thousands)."""
    return _make_writer(_dev_frame, [Path(d) for d in state_dirs])


def _census_frame(state_dir: Path, level: str) -> Optional[gpd.GeoDataFrame]:
    frames = [gpd.read_parquet(p) for p in sorted((state_dir / "counties").glob(f"*/{level}.geoparquet"))]
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


def _make_writer(loader: Callable[[Path, int], Optional[gpd.GeoDataFrame]], state_dirs: List[Path]):
    def writer(level_name, id_col, property_cols, min_zoom, max_zoom, tiles_dir):
        level_dir = Path(tiles_dir) / level_name
        if level_dir.is_dir():
            shutil.rmtree(level_dir)  # drop chunks of a previous build
        offset = 0
        for d in state_dirs:
            try:
                gdf = loader(d, offset)
            except Exception as exc:
                logger.error(f"{level_name}: {d.name} failed to load ({exc}); skipped")
                continue
            if gdf is None or not len(gdf):
                continue
            offset += len(gdf)
            _write_chunk(gdf, level_name, id_col, property_cols, min_zoom, max_zoom, Path(tiles_dir), _state_name(d))
            logger.info(f"{level_name}: {d.name} tiled ({len(gdf):,} rows)")
            del gdf

    return writer


def make_writers(state_dirs: Iterable[Path], include_streets: bool = True) -> Dict[Tuple[str, str], Callable]:
    """`{(map_kind, level_name): writer}` for `build_city_map(external_tile_writers=...)`."""
    dirs = [Path(d) for d in state_dirs]
    writers: Dict[Tuple[str, str], Callable] = {}
    for res in EXTERNAL_H3_RESOLUTIONS:
        writers[("hexagons", f"h3_{res}")] = _make_writer(lambda d, off, r=res: _h3_frame(d, r, False), dirs)
        writers[("circles", f"h3_{res}")] = _make_writer(lambda d, off, r=res: _h3_frame(d, r, True), dirs)
    for level in EXTERNAL_CENSUS_LEVELS:
        writers[("census", f"census_{level}")] = _make_writer(lambda d, off, lv=level: _census_frame(d, lv), dirs)
    writers[("development", "h3_7")] = make_dev_writer(dirs)
    if include_streets:
        writers[("streets", "edges")] = _make_writer(_street_frame, dirs)
    return writers
