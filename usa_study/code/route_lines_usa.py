"""Transit line geometry of a state (`<state>/routes_lines.parquet`), drawn on the nationwide map.

One line per GTFS route (`transitlos.map.routes.build_route_lines`, the same builder as the city maps: the real `shapes.txt` line,
else none), clipped to the state, lightly simplified (~10 m) so a whole state's lines stay small. Routes without a real shape are
left out, like on the city maps.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

logger = logging.getLogger(__name__)
ROUTE_LINES_FILE = "routes_lines.parquet"
SIMPLIFY_DEG = 0.0001


def compute_state_route_lines(state_dir: Path) -> gpd.GeoDataFrame:
    from transitlos.map.routes import build_route_lines

    state_dir = Path(state_dir)
    dirs = sorted(p for p in (state_dir / "gtfs").iterdir() if p.is_dir() and (p / "stops.txt").is_file()) if (state_dir / "gtfs").is_dir() else []
    if dirs:   # one service published twice (own file + copy) would draw every line twice
        from pyGTFSHandler.utils.feed_dedupe import drop_duplicate_feeds

        dirs = drop_duplicate_feeds(dirs)
    state = gpd.read_parquet(state_dir / "state.geoparquet")
    frames = []
    for d in dirs:
        try:
            g = build_route_lines([d], aoi=state, exclude_no_shape_routes=True)
        except Exception as exc:  # noqa: BLE001 -- one malformed feed must not discard the others
            logger.warning(f"{state_dir.name}/{d.name}: route lines skipped ({type(exc).__name__}: {str(exc)[:80]})")
            continue
        if g is not None and not g.empty:
            frames.append(g.to_crs(4326))
    if frames:
        import pandas as pd

        out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)
        out["geometry"] = out.geometry.simplify(SIMPLIFY_DEG)
        out = out[~out.geometry.is_empty & out.geometry.notna()]
    else:
        out = gpd.GeoDataFrame({"route_label": [], "mode": [], "color": [], "geometry": []}, geometry="geometry", crs=4326)
    out.to_parquet(state_dir / ROUTE_LINES_FILE)
    return out
