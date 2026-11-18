"""Census polygons with no transit score (0 / missing) take the mean score of the finest H3 cells (res 11, LOS > 0) inside them.

Streets-based block scores miss blocks whose streets are not in the scored network; the res-11 hexagons next to them do carry the score.
Polygons with no cell centre inside take the nearest cell within `MAX_GAP_M`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

MAX_GAP_M = 80.0


@lru_cache(maxsize=2)
def _cells(state_dir: str) -> gpd.GeoDataFrame | None:
    p = Path(state_dir) / "h3_res11.geoparquet"
    if not p.is_file():
        return None
    df = pd.read_parquet(p, columns=["h3", "level_of_service"])
    df = df[df["level_of_service"] > 0]
    if df.empty:
        return None
    ll = np.asarray([h3.cell_to_latlng(c) for c in df["h3"]], dtype=np.float64).reshape(-1, 2)
    return gpd.GeoDataFrame({"los": df["level_of_service"].to_numpy()}, geometry=gpd.points_from_xy(ll[:, 1], ll[:, 0]), crs=4326)


def fill_zero_los(gdf: gpd.GeoDataFrame, state_dir) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty or "level_of_service" not in gdf.columns:
        return gdf
    zero = ~(gdf["level_of_service"] > 0)
    if not zero.any():
        return gdf
    cells = _cells(str(state_dir))
    if cells is None:
        return gdf
    gdf = gdf.copy()
    sub = gpd.GeoDataFrame(geometry=gdf.geometry[zero].to_crs(4326))
    # only the state's cells near these polygons (bbox + a margin wider than MAX_GAP_M): the join is then local instead of scanning every res-11 cell of the state
    # once per county file (1.6 s per county in Texas before)
    import shapely

    x0, y0, x1, y1 = sub.total_bounds
    pad = 0.004
    near_idx = cells.sindex.query(shapely.box(x0 - pad, y0 - pad, x1 + pad, y1 + pad))
    cells = cells.iloc[near_idx]
    if cells.empty:
        return gdf
    hit = gpd.sjoin(cells, sub, how="inner", predicate="within").groupby("index_right")["los"].mean()
    gdf.loc[hit.index, "level_of_service"] = hit.to_numpy()
    left = sub[~sub.index.isin(hit.index)]
    if len(left):
        near = gpd.sjoin_nearest(left.to_crs(5070), cells.to_crs(5070), how="inner", max_distance=MAX_GAP_M, distance_col="d")
        near = near[~near.index.duplicated()]
        gdf.loc[near.index, "level_of_service"] = near["los"].to_numpy()
    return gdf
