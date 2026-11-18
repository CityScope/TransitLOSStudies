# -*- coding: utf-8 -*-
"""Assemble one autonomous community's (or the whole country's) Spain outputs into ``transitlos.map.build.build_city_map`` inputs.

Pure glue, like usa_study's ``build_state_map``: census hierarchy (coarse to fine) ``ccaa`` > ``province`` > ``municipality`` > ``section``,
H3 res 3..11, street edges, stops, plus the bilingual page patch (``i18n_inject``) applied to the written map.html.
"""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import h3
import pandas as pd
import shapely

logger = logging.getLogger(__name__)
H3_RESOLUTIONS = (3, 4, 5, 7, 8, 9, 10, 11)
CENSUS_ORDER = ("ccaa", "province", "municipality", "section")
COARSE_RESOLUTION = 8
STATS_RESOLUTION = 8
EXTERNAL_H3 = (9, 10, 11)
EXTERNAL_CENSUS = ("section",)


def _h3_geometry(df: pd.DataFrame) -> gpd.GeoDataFrame:
    polys = [shapely.Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(c)]) for c in df["h3"]]
    return gpd.GeoDataFrame(df.rename(columns={"h3": "h3_cell"}), geometry=polys, crs=4326)


def _concat(frames):
    frames = [f for f in frames if f is not None and len(f)]
    return None if not frames else gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)


def _streets(ccaa_dir: Path):
    return _concat([gpd.read_parquet(p) for p in sorted((ccaa_dir / "provinces").glob("*/street.geoparquet"))])


def _sample(parts, n=4000):
    out = []
    for df in parts:
        pick = {*df.sample(min(len(df), n), random_state=0).index}
        for col in df.select_dtypes("number").columns:
            if df[col].notna().any():
                pick.add(df[col].idxmax())
                pick.add(df[col].idxmin())
        out.append(df.loc[sorted(pick)])
    return pd.concat(out, ignore_index=True)


def assemble_map_inputs(ccaa_dirs, include_streets: bool = True, external: bool = False) -> dict:
    """Keyword arguments for ``build_city_map`` from one or many CCAA output folders (see ``main.py``)."""
    ccaa_dirs = [Path(d) for d in ccaa_dirs]
    multi = len(ccaa_dirs) > 1
    from .ccaa_pipeline import _resample_up

    from .census_spain import map_columns
    from .gtfs_store import ccaa_feed_dirs, default_gtfs_dir

    h3_tables: Dict[int, pd.DataFrame] = {}
    reports: Dict[str, dict] = {}
    for res in H3_RESOLUTIONS:
        parts = [pd.read_parquet(d / f"h3_res{res}.geoparquet") for d in ccaa_dirs if (d / f"h3_res{res}.geoparquet").is_file()]
        if parts and external and res in EXTERNAL_H3:
            h3_tables[res] = _sample(parts)
        elif parts:
            h3_tables[res] = pd.concat(parts, ignore_index=True)
        if res in h3_tables:
            h3_tables[res] = map_columns(h3_tables[res], reports.setdefault(f"h3_res{res}", {}))
    if multi and COARSE_RESOLUTION in h3_tables:
        merged = _resample_up(h3_tables[COARSE_RESOLUTION], COARSE_RESOLUTION)   # border cells shared by two communities
        h3_tables[COARSE_RESOLUTION] = merged
        for res in H3_RESOLUTIONS:
            if res < COARSE_RESOLUTION:
                h3_tables[res] = _resample_up(merged, res)
    if 8 in h3_tables:   # development view: one regression per PROVINCE on res-8 cells (`code.equity`)
        from .equity import add_equity_flag

        h3_tables[8] = add_equity_flag(h3_tables[8], ccaa_dirs)
    h3_by_resolution = {r: _h3_geometry(df) for r, df in h3_tables.items()}

    census: Dict[str, gpd.GeoDataFrame] = {}
    for lvl in CENSUS_ORDER:
        frames = [gpd.read_parquet(d / f"{lvl}.geoparquet") for d in ccaa_dirs if (d / f"{lvl}.geoparquet").is_file()]
        if external and lvl in EXTERNAL_CENSUS:
            frames = [f.head(3000) for f in frames]
        g = _concat([gpd.GeoDataFrame(map_columns(f, reports.setdefault(f"{lvl}/{i}", {})), geometry="geometry", crs=f.crs) for i, f in enumerate(frames)])
        if g is not None:
            census[lvl] = g
    if len(census.get("province", [])) <= 1 and "ccaa" in census:   # one-province community: the community level adds nothing
        census.pop("ccaa", None)
    try:
        from pycensus.global_equivalents import add_global_equivalents

        census = {k: add_global_equivalents(v) for k, v in census.items()}
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"global equivalents skipped: {exc}")

    edges = _streets_all(ccaa_dirs, include_streets and not external)
    stops_gdf, stops_chunks, routes_chunks = None, None, None
    from .route_labels import add_route_labels

    def labelled(d: Path):
        by_id = d / "stops_by_id.parquet"
        if by_id.is_file():
            t = pd.read_parquet(by_id)
            g = gpd.GeoDataFrame(t, geometry=gpd.points_from_xy(t["stop_lon"], t["stop_lat"]), crs=4326)
        else:
            g = add_route_labels(gpd.read_parquet(d / "stops.geoparquet"), ccaa_feed_dirs(d, default_gtfs_dir(d.parent)))
        return g.rename(columns={"scoring_mode": "mode_category"}) if "scoring_mode" in g.columns and "mode_category" not in g.columns else g

    have = [d for d in ccaa_dirs if (d / "stops.geoparquet").is_file() or (d / "stops_by_id.parquet").is_file()]
    if multi:
        stops_chunks = {d.name: labelled(d) for d in have}
        from .route_lines import ROUTE_LINES_FILE, compute_ccaa_route_lines

        routes_chunks = {}
        for d in have:
            f = d / ROUTE_LINES_FILE
            lines = gpd.read_parquet(f) if f.is_file() else (compute_ccaa_route_lines(d))
            if lines is not None and len(lines):
                routes_chunks[d.name] = lines
    else:
        stops_gdf = _concat([labelled(d) for d in have])
    _write_report(reports, ccaa_dirs)
    out = {"h3_by_resolution": h3_by_resolution, "edges_gdf": edges, "census_by_level": census, "stops_gdf": stops_gdf, "is_us": False, "country": "ESP"}
    if not multi and STATS_RESOLUTION in h3_by_resolution:   # the community map's own stats panel (distribution / regression / ANOVA), as the city maps
        out["stats_by_area"] = {"metro": h3_by_resolution[STATS_RESOLUTION]}
    if multi and 4 in h3_by_resolution:   # nation-level stats grid embedded in the page (res 4); communities load res 8 on demand
        st = h3_by_resolution[4].copy()
        num = [c for c in st.columns if c not in ("h3_cell", "geometry") and pd.api.types.is_numeric_dtype(st[c])]
        st[num] = st[num].round(2)
        out["stats_by_area"] = {"metro": st}
    if stops_chunks is not None:
        out["stops_chunks"], out["routes_chunks"] = stops_chunks, routes_chunks
    return out


def _write_report(reports: dict, ccaa_dirs) -> None:
    """Which census columns were dropped (all NaN) / clipped to [0, 1] / had negative values blanked: ``map_columns_report.json`` + log."""
    import json

    reports = {k: v for k, v in reports.items() if v}
    dropped = sorted({c for v in reports.values() for c in v.get("dropped", [])})
    logger.info(f"map columns dropped (entirely NaN) for {[d.name for d in ccaa_dirs][:3]}: {dropped}")
    for k, v in reports.items():
        if v.get("clipped") or v.get("negative"):
            logger.info(f"  {k}: clipped to [0,1] {v.get('clipped', {})}; negative blanked {v.get('negative', {})}")
    target = ccaa_dirs[0] / "map_columns_report.json" if len(ccaa_dirs) == 1 else ccaa_dirs[0].parent / "map_columns_report_national.json"
    try:
        target.write_text(json.dumps(reports, indent=1, default=str))
    except OSError:
        pass


def _streets_all(ccaa_dirs, include: bool):
    if include:
        e = _concat([_streets(d) for d in ccaa_dirs])
        if e is not None:
            return e
    samples = []
    for d in ccaa_dirs:
        s = next(iter(sorted((d / "provinces").glob("*/street.geoparquet"))), None)
        if s is not None:
            samples.append(gpd.read_parquet(s).head(300))
        if len(samples) >= 10:
            break
    return _concat(samples)


@contextlib.contextmanager
def spain_field_policy():
    """Selector policy of the shared builder, Spain edition (restored on exit; build.py itself is untouched).

    ``transitlos.map.build.is_relative_field`` decides which columns are counts (circle size, distribution) and which are relative
    (opacity, regression, ANOVA). Municipality natives repeat on every section, so they are treated as relative (see
    ``census_spain.with_native_columns_as_relative``).
    """
    from transitlos.map import build as _b

    from .census_spain import with_native_columns_as_relative

    original = _b.is_relative_field
    _b.is_relative_field = with_native_columns_as_relative(original)
    try:
        yield
    finally:
        _b.is_relative_field = original


def _data_columns(inputs: dict) -> list:
    cols = set()
    for g in (inputs.get("h3_by_resolution") or {}).values():
        cols.update(g.columns)
    for g in (inputs.get("census_by_level") or {}).values():
        cols.update(g.columns)
    return sorted(cols)


def build_region_map(ccaa_dir: Path, tiles_dir: Optional[Path] = None, out_html: Optional[Path] = None) -> str:
    """Build one community's map.html (bilingual, see ``i18n_inject``)."""
    from transitlos.map.build import build_city_map

    from .i18n_inject import inject_language_switch

    ccaa_dir = Path(ccaa_dir)
    inputs = assemble_map_inputs([ccaa_dir])
    with spain_field_policy():
        html = build_city_map(tiles_dir=str(tiles_dir or ccaa_dir / "tiles"), out_html=str(out_html or ccaa_dir / "map.html"), **inputs)
    inject_language_switch(html, columns=_data_columns(inputs))
    return html
