# -*- coding: utf-8 -*-
"""Transit line geometry of a community (``routes_lines.parquet``) for the nationwide map (copied pattern of usa_study's route_lines_usa)."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)
ROUTE_LINES_FILE = "routes_lines.parquet"
SIMPLIFY_DEG = 0.0001


def compute_ccaa_route_lines(ccaa_dir: Path) -> gpd.GeoDataFrame:
    from transitlos.map.routes import build_route_lines

    ccaa_dir = Path(ccaa_dir)
    from .gtfs_store import ccaa_feed_dirs, default_gtfs_dir

    dirs = ccaa_feed_dirs(ccaa_dir, default_gtfs_dir(ccaa_dir.parent))
    aoi = gpd.read_parquet(ccaa_dir / "ccaa.geoparquet")
    frames = []
    for d in dirs:
        try:
            g = build_route_lines([d], aoi=aoi, exclude_no_shape_routes=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{ccaa_dir.name}/{d.name}: route lines skipped ({type(exc).__name__}: {str(exc)[:80]})")
            continue
        if g is not None and not g.empty:
            frames.append(g.to_crs(4326))
    if frames:
        out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)
        out["geometry"] = out.geometry.simplify(SIMPLIFY_DEG)
        out = out[~out.geometry.is_empty & out.geometry.notna()]
    else:
        out = gpd.GeoDataFrame({"route_label": [], "mode": [], "color": [], "geometry": []}, geometry="geometry", crs=4326)
    out.to_parquet(ccaa_dir / ROUTE_LINES_FILE)
    return out
