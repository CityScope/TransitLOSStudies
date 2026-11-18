# -*- coding: utf-8 -*-
"""Stage 2 (Spain): one province -> street network, level of service, census sections / municipalities, H3 res 11..7.

Mirrors usa_study's ``county_pipeline`` with these differences:
  * unit = province (INE 2-digit code); native census level = SECTION (``census_spain``), plus municipality (dissolved sections, municipality
    natives such as tourist dwellings and valor tasado are kept as native values);
  * OSM extract from Geofabrik (smallest region covering the province, e.g. ``spain/pais-vasco``);
  * H3 columns come from sections under the normal-map rules (``_write_h3``, ``study_params.GRID``): res 11 cells near streets, sections by
    centroid split by Catastro floor area, >= 10 residents per km2, one rescale factor per group for the province; coarser levels aggregated
    from res 11; census levels take the population-weighted LOS of their res 11 cells;
  * no elections.
Outputs under ``<province_dir>``: ``section.geoparquet``, ``municipality.geoparquet``, ``street.geoparquet`` (LOS > 0), ``h3_res{11,10,9,8,7}.geoparquet``
(cell id + columns, no geometry), ``column_kinds.json``, ``stops_signature.txt``.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import shapely
from transitlos.level_of_service import compute_level_of_service
from transitlos.network import prepare_street_network
from UrbanAccessAnalyzer.api import AreaOfInterest
from UrbanAccessAnalyzer.osm_io import download_geofabrik

from . import census_spain

logger = logging.getLogger(__name__)
MAX_WALK_DISTANCE_M = 1500.0
SIMPLIFY_DISTANCE_M = 10.0
H3_RESOLUTIONS = (11, 10, 9, 8, 7)
NETWORK_CACHE_NAME = "_network_cache_b1500"
STOPS_SIGNATURE_FILE = "stops_signature.txt"


def get_pbf(aoi: gpd.GeoDataFrame, osm_files_dir: Path) -> str:
    os.makedirs(osm_files_dir, exist_ok=True)
    return download_geofabrik(aoi, output_folder=str(osm_files_dir))


def _to_wkb_polars(gdf):
    import polars as pl

    wkb = shapely.to_wkb(gdf.geometry.to_numpy())
    return pl.from_pandas(gdf.drop(columns="geometry").reset_index(drop=True)).with_columns(pl.Series("geometry", wkb, dtype=pl.Binary))


def _buffer_m(gdf, d):
    utm = gdf.estimate_utm_crs()
    b = gdf.to_crs(utm)
    b["geometry"] = b.geometry.buffer(d)
    return b.to_crs(4326)


def province_stops_signature(prov_gdf, stops_gdf) -> str:
    buf = _buffer_m(prov_gdf, MAX_WALK_DISTANCE_M).union_all()
    if not len(stops_gdf) or "stop_lat" not in stops_gdf.columns:   # no stops around the province (e.g. Melilla)
        return hashlib.sha1(b"[]").hexdigest()
    sel = stops_gdf[stops_gdf.intersects(buf)]
    rows = sorted(zip(sel["stop_lat"].round(5), sel["stop_lon"].round(5), sel["stop_score"].round(2)))
    return hashlib.sha1(repr(rows).encode()).hexdigest()


def province_already_processed(d: Path) -> bool:
    return (d / "section.geoparquet").is_file() and any(d.glob("h3_res*.geoparquet"))


def _floor_area_cells(prov_code: Optional[str], prov_gdf, resolution: int):
    """(cells, residential, other floor area) of the province's buildings: Catastro (Gipuzkoa: b5m), GlobalBuildingAtlas where the cadaster
    has nothing (foral provinces, failed downloads). Empty arrays when no source is available (the split is then equal)."""
    from pycensus import cadaster
    from pycensus.cadaster.metrics import cell_floor_area

    from .study_params import FORAL_SOURCES

    cache = Path(__file__).resolve().parents[2] / "cache"
    try:
        b = cadaster.load_buildings("ESP", prov_gdf, cache_dir=cache, sources=FORAL_SOURCES.get(prov_code or "", ["es_catastro"]), gba_fill=False)
        if b.empty:
            logger.warning(f"Province {prov_code}: no cadaster buildings -- GlobalBuildingAtlas floor areas")
            b = cadaster.load_buildings("ESP", prov_gdf, cache_dir=cache, sources=[], gba_fill=True)
        logger.info(f"Province {prov_code}: {len(b):,} buildings, {pd.to_numeric(b['built_area_m2'], errors='coerce').sum() / 1e6:,.1f} km2 floor area")
        return cell_floor_area(b, resolution)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Province {prov_code}: floor areas not available ({type(exc).__name__}: {str(exc)[:160]}); equal split")
        return np.zeros(0, np.uint64), np.zeros(0), np.zeros(0)


def census_los_from_res11(polygons: gpd.GeoDataFrame, res11: Optional[pd.DataFrame]) -> np.ndarray:
    """Level of service of census polygons: the population-weighted mean of the res 11 cells whose centre they hold (plain mean where nobody
    lives; a polygon smaller than a cell takes the cell under it). No res 11 cell: 0."""
    if res11 is None or res11.empty or polygons.empty:
        return np.zeros(len(polygons))
    import h3ronpy
    from pycensus.dasymetric import cells_to_polygons_mean

    cells = np.asarray(h3ronpy.cells_parse(res11["h3"].tolist())).astype(np.uint64)
    w = [pd.to_numeric(res11["population"], errors="coerce").fillna(0).to_numpy(float)]
    return np.nan_to_num(cells_to_polygons_mean(polygons, cells, pd.to_numeric(res11["level_of_service"], errors="coerce").fillna(0).to_numpy(float), w), nan=0.0)


def _write_h3(prov_gdf, sections, scored_wkb, street_geoms, prov_dir: Path, prov_code: Optional[str] = None) -> dict:
    """H3 res 11..7 of one province under the normal-map rules (`study_params.GRID`, documented in `transitlos.grid_rules`):

      * res 11 candidate cells: every cell within `street_max_dist_m` of a street of the (connected) network, plus every cell holding a building;
      * sections -> cells by centroid (one section per cell), counts split by Catastro floor area (residential for residents' columns, total
        for the building-stock columns `ine_catastro_*`); rates are the section's value;
      * active = street cell with >= `occupancy_min_per_km2` residents per km2; everything else (buildings off the streets, near-empty cells,
        sections holding no cell centre) goes back into the active cells with ONE factor per group for the province (residents' columns by the
        population factor, building-stock columns by the floor-area factor), logged, WARNING above `rescale_warn_share`;
      * LOS = the best street edge within reach of the cell centre (nearest street as fallback);
      * res 10..7: exact aggregation of res 11 (sums; rates and LOS population-weighted).
    The returned dict carries the res 11 table under "_res11" (callers pop it)."""
    import h3ronpy
    from geohierarchy import h3_cells_near_lines
    from pycensus.dasymetric import polygon_values_at_cells, polygons_to_cells, rescale_to_totals

    from .h3_los_sampling import sample_cell_los
    from .study_params import GRID

    prov_geom = prov_gdf.to_crs(4326).union_all()
    rates, counts = census_spain.split_columns(sections)
    counts = ["population", *[c for c in counts if c != "population"]]
    written: dict = {"_res11": None}
    lines = street_geoms if street_geoms is not None and len(street_geoms) else (
        gpd.GeoSeries(shapely.from_wkb(scored_wkb["geometry"].to_numpy()), crs=4326) if scored_wkb is not None and scored_wkb.height else None)
    if lines is None:
        return written
    ids = []
    vals = lines.to_crs(4326).values
    for a in range(0, len(vals), 200_000):
        ids.append(np.asarray(h3_cells_near_lines(vals[a:a + 200_000], 11, GRID.street_max_dist_m, region=prov_geom)).astype(np.uint64))
    street = np.unique(np.concatenate(ids)) if ids else np.zeros(0, np.uint64)
    if not len(street):
        return written
    w_cells, w_res, w_non = _floor_area_cells(prov_code, prov_gdf, 11)
    universe = np.union1d(street, w_cells)
    wr = np.zeros(len(universe)); wn = np.zeros(len(universe))
    if len(w_cells):
        ix = np.searchsorted(universe, w_cells); wr[ix] = w_res; wn[ix] = w_non
    groups = {c: ("all" if c.startswith("ine_catastro_") else "res") for c in counts}
    cnt, _ = polygons_to_cells(sections, universe, counts, weights={"res": wr, "non": wn} if len(w_cells) else None, groups=groups)
    area = np.asarray(h3ronpy.cells_area_km2(universe)).astype(float)
    act = np.nan_to_num(cnt["population"].to_numpy(float))
    active = np.isin(universe, street) & (act / area >= GRID.occupancy_min_per_km2)
    totals = {c: float(np.nansum(pd.to_numeric(sections[c], errors="coerce").to_numpy(float))) for c in counts}
    # building stock (group "all": Catastro buildings, dwellings, floor area) is never rescaled: what lies in inactive cells is dropped
    anchor = {"res": "population"}
    factors: dict = {}
    cnt = rescale_to_totals(cnt, active, totals, groups, anchor, log=logger.info, warn_share=GRID.rescale_warn_share, label=f"province {prov_code}",
                            factors=factors)
    import json as _json

    (prov_dir / "rescale_factors.json").write_text(_json.dumps({"note": "original = value / factor (res 11 and every coarser level)", "province": prov_code,
                                                                 "factors": factors}, indent=1))
    keep = universe[active]
    if not len(keep):
        return written
    h3_ids = h3ronpy.cells_to_string(keep).to_pylist()
    sampled = sample_cell_los(scored_wkb, street_geoms, h3_ids, 11, max_dist_m=GRID.street_max_dist_m)
    rv = polygon_values_at_cells(sections, keep, rates)
    res11 = pd.concat([pd.DataFrame({"h3": h3_ids, "level_of_service": pd.Series(h3_ids).map(sampled).fillna(0.0).to_numpy()}),
                       cnt[counts].reset_index(drop=True), rv[rates].reset_index(drop=True)], axis=1)
    path = prov_dir / "h3_res11.geoparquet"
    res11.to_parquet(path)
    written["h3_res11"] = str(path)
    written["_res11"] = res11

    # coarser levels: exact aggregation of res 11
    pop = pd.to_numeric(res11["population"], errors="coerce").fillna(0).to_numpy(float)
    for res in [r for r in H3_RESOLUTIONS if r != 11]:
        k = pd.DataFrame({"_p": [h3.cell_to_parent(c, res) for c in res11["h3"]]})
        g = k.groupby("_p").indices
        parents = sorted(g)
        out = {"h3": parents}
        for c in counts:
            v = res11[c].to_numpy(float)
            s_ = pd.Series(v).groupby(k["_p"].to_numpy()).sum(min_count=1)
            out[c] = s_.reindex(parents).to_numpy()
        for c in [*rates, "level_of_service"]:
            v = res11[c].to_numpy(float); m = np.isfinite(v)
            num = pd.Series(np.where(m, v * pop, 0.0)).groupby(k["_p"].to_numpy()).sum()
            den = pd.Series(np.where(m, pop, 0.0)).groupby(k["_p"].to_numpy()).sum()
            plain = pd.Series(v).groupby(k["_p"].to_numpy()).mean()
            out[c] = np.where(den.reindex(parents).to_numpy() > 0, (num / den.where(den > 0)).reindex(parents).to_numpy(), plain.reindex(parents).to_numpy())
        tab = pd.DataFrame(out)
        tab = tab[pd.to_numeric(tab["population"], errors="coerce").fillna(0) > 0]
        if tab.empty:
            continue
        path = prov_dir / f"h3_res{res}.geoparquet"
        tab.reset_index(drop=True).to_parquet(path)
        written[f"h3_res{res}"] = str(path)
    return written


def municipality_level(sections: gpd.GeoDataFrame, names: Optional[pd.Series] = None) -> gpd.GeoDataFrame:
    """Dissolve sections to municipalities: counts summed, rates population-weighted (municipality natives are constant within one)."""
    rates, counts = census_spain.split_columns(sections)
    if "level_of_service" in sections.columns:
        rates = [*rates, "level_of_service"]
    s = sections.copy()
    pop = pd.to_numeric(s["population"], errors="coerce").fillna(0)
    agg = {c: "sum" for c in counts}
    for c in rates:
        v = pd.to_numeric(s[c], errors="coerce")
        s[f"_w_{c}"] = (v * pop).where(v.notna(), 0.0)
        s[f"_p_{c}"] = pop.where(v.notna(), 0.0)
        agg[f"_w_{c}"] = "sum"
        agg[f"_p_{c}"] = "sum"
    g = s.groupby("municipality_id")
    tab = g.agg(agg)
    for c in rates:
        tab[c] = tab[f"_w_{c}"] / tab[f"_p_{c}"].where(tab[f"_p_{c}"] > 0)
        # a municipality without population keeps the plain mean (natives are constant) so it is not lost
    tab = tab.drop(columns=[c for c in tab.columns if c.startswith(("_w_", "_p_"))])
    geom = s.dissolve(by="municipality_id").geometry
    out = gpd.GeoDataFrame(tab.join(geom), geometry="geometry", crs=sections.crs).reset_index().rename(columns={"municipality_id": "GEOID"})
    if "level_of_service_area" in out:
        out = out.drop(columns="level_of_service_area")
    if names is not None:
        out["name"] = out["GEOID"].map(names)
    out["province_id"] = out["GEOID"].str[:2]
    return out


def municipality_names(data_dir: str) -> pd.Series:
    from pycensus.countries.spain import geography

    m = geography._load_spain_municipalities(data_dir)
    return m.set_index("GEOID")["LAU_NAME"]


def process_province(sections: gpd.GeoDataFrame, prov_code: str, pbf_path: str, stops_gdf: gpd.GeoDataFrame, prov_dir: Path,
                     network_cache_dir: Optional[Path] = None, data_dir: str = "data/spain") -> dict:
    """Run the full per-province pipeline (``sections``: ``census_spain.load_province_sections``)."""
    os.makedirs(prov_dir, exist_ok=True)
    prov_geom = sections.to_crs(4326).union_all().buffer(0)
    prov_gdf = gpd.GeoDataFrame({"GEOID": [prov_code]}, geometry=[prov_geom], crs=4326)
    aoi = AreaOfInterest(prov_gdf)
    buf = _buffer_m(prov_gdf, MAX_WALK_DISTANCE_M)
    stops_in = stops_gdf[stops_gdf.intersects(buf.union_all())].reset_index(drop=True)
    n_stops = len(stops_in)
    edges_gdf = None
    street_geoms = None
    n_scored = 0
    if n_stops:
        network = prepare_street_network(AreaOfInterest(buf), cache_dir=str(network_cache_dir) if network_cache_dir else None,
                                         cluster_distance=SIMPLIFY_DISTANCE_M, pbf_path=pbf_path, network_type="all", ignore_oneway=True)
        street_geoms = gpd.GeoSeries(shapely.from_wkb(network.edges["geometry_wkb"].to_numpy()), crs=network.crs).to_crs(4326)
        network = network.crop(AreaOfInterest(_buffer_m(stops_in, MAX_WALK_DISTANCE_M)), crop_buffer_m=0.0)
        edges_gdf = compute_level_of_service(aoi=aoi, network=network, stops=stops_in, max_walk_distance_m=MAX_WALK_DISTANCE_M)
        if len(edges_gdf) == 0:
            edges_gdf = None
    else:
        logger.warning(f"Province {prov_code}: no stops within {MAX_WALK_DISTANCE_M} m -- LOS = 0.")
    written, scored_wkb = {}, None
    if edges_gdf is not None:
        inside = edges_gdf[edges_gdf.intersects(prov_geom if edges_gdf.crs.to_epsg() == 4326 else gpd.GeoSeries([prov_geom], crs=4326).to_crs(edges_gdf.crs).iloc[0])]
        scored = inside[inside["level_of_service"] > 0]
        n_scored = len(scored)
        if n_scored:
            scored.to_crs(4326).to_parquet(prov_dir / "street.geoparquet")
            written["street"] = str(prov_dir / "street.geoparquet")
            scored_wkb = _to_wkb_polars(scored.to_crs(4326))
    # --- h3 first: the census levels take their LOS from the res 11 cells (population-weighted mean) ---
    sec = sections.copy()
    h3_written = _write_h3(prov_gdf, sec, scored_wkb, street_geoms, prov_dir, prov_code)
    res11 = h3_written.pop("_res11", None)
    written.update(h3_written)
    # --- census levels ---
    sec["level_of_service"] = census_los_from_res11(sec, res11)
    sec = sec[pd.to_numeric(sec["population"], errors="coerce").fillna(0) > 0]
    sec.to_crs(4326).to_parquet(prov_dir / "section.geoparquet")
    written["section"] = str(prov_dir / "section.geoparquet")
    try:
        names = municipality_names(data_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"municipality names unavailable: {exc}")
        names = None
    mun = municipality_level(sec, names)
    mun["level_of_service"] = census_los_from_res11(mun, res11)
    mun.to_crs(4326).to_parquet(prov_dir / "municipality.geoparquet")
    written["municipality"] = str(prov_dir / "municipality.geoparquet")
    census_spain.write_column_kinds(sec, prov_dir / "column_kinds.json")
    (prov_dir / STOPS_SIGNATURE_FILE).write_text(province_stops_signature(prov_gdf, stops_gdf))
    prov_gdf.to_parquet(prov_dir / "province_boundary.geoparquet")
    return {"n_stops": n_stops, "n_edges_scored": n_scored, "written": written}


def rebuild_province_h3(prov_dir: Path) -> dict:
    """Re-derive one finished province's `h3_res*.geoparquet` under the current LOS-resampling scheme (`code.h3_los_sampling`) from its saved
    `section.geoparquet`, scored `street.geoparquet` and cached street network; no LOS recomputation, no network access.
    The community roll-up must be redone afterwards (`code.ccaa_pipeline.merge_and_resample_h3`)."""
    import polars as pl

    prov_dir = Path(prov_dir)
    sections = gpd.read_parquet(prov_dir / "section.geoparquet")
    prov_geom = sections.to_crs(4326).union_all().buffer(0)
    prov_gdf = gpd.GeoDataFrame({"GEOID": [prov_dir.name]}, geometry=[prov_geom], crs=4326)
    scored_wkb = _to_wkb_polars(gpd.read_parquet(prov_dir / "street.geoparquet").to_crs(4326)) if (prov_dir / "street.geoparquet").is_file() else None
    street_geoms = None
    net = prov_dir / "_network_cache_b1500"
    if (net / "edges.parquet").is_file():
        crs = (net / "meta.txt").read_text().strip()
        street_geoms = gpd.GeoSeries(shapely.from_wkb(pl.read_parquet(net / "edges.parquet", columns=["geometry_wkb"])["geometry_wkb"].to_numpy()), crs=crs).to_crs(4326)
    for old in prov_dir.glob("h3_res*.geoparquet"):
        old.unlink()
    written = _write_h3(prov_gdf, sections.drop(columns="level_of_service", errors="ignore"), scored_wkb, street_geoms, prov_dir, prov_dir.name)
    res11 = written.pop("_res11", None)
    for name in ("section", "municipality"):   # census levels: LOS from the new res 11 cells
        f = prov_dir / f"{name}.geoparquet"
        if f.is_file():
            g = gpd.read_parquet(f)
            g["level_of_service"] = census_los_from_res11(g, res11)
            g.to_parquet(f)
    return written
