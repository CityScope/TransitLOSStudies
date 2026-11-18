# -*- coding: utf-8 -*-
"""Stage 3 (Spain): roll the provinces of one autonomous community (CCAA) up.

Mirrors usa_study's ``state_pipeline``: ``province.geoparquet`` (dissolved sections, counts summed / rates population-weighted),
``municipality.geoparquet`` (concat), ``section.geoparquet`` (concat), ``ccaa.geoparquet`` (one row, the community itself), and the
H3 levels: res 7-11 are concatenated from the provinces (border cells merged), coarser res 3, 4, 5 resampled up from res 7 with
population weighting for rates and sums for counts (``census_spain.is_rate``).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import h3
import pandas as pd

from . import census_spain
from .province_pipeline import municipality_level

logger = logging.getLogger(__name__)
CCAA_H3_RESOLUTIONS: Sequence[int] = (3, 4, 5, 7, 8, 9, 10, 11)
BASE_RES = 7


def province_dirs(ccaa_dir: Path):
    return sorted(p for p in (Path(ccaa_dir) / "provinces").glob("*") if p.is_dir())


def _concat(ccaa_dir: Path, name: str):
    frames = [gpd.read_parquet(p / name) for p in province_dirs(ccaa_dir) if (p / name).is_file()]
    return None if not frames else gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)


def merge_levels(ccaa_dir: Path, names: dict | None = None) -> dict:
    """section / municipality / province / ccaa geoparquet files of the community."""
    ccaa_dir = Path(ccaa_dir)
    out = {}
    sec = _concat(ccaa_dir, "section.geoparquet")
    if sec is None:
        raise ValueError(f"no province outputs under {ccaa_dir}")
    mun = _concat(ccaa_dir, "municipality.geoparquet")
    prov_list = []
    for p in province_dirs(ccaa_dir):
        f = p / "section.geoparquet"
        if not f.is_file():
            continue
        s = gpd.read_parquet(f)
        s["municipality_id"] = p.name   # dissolve the whole province into one row
        row = municipality_level(s)
        row["GEOID"] = p.name
        row["name"] = (names or {}).get(p.name, p.name)
        row = row.drop(columns=["province_id"], errors="ignore")
        prov_list.append(row)
    prov = gpd.GeoDataFrame(pd.concat(prov_list, ignore_index=True), geometry="geometry", crs=4326)
    # a province's LOS is the mean of its sections' LOS weighted by population (rates)
    ccaa_row = municipality_level(sec.assign(municipality_id=ccaa_dir.name)).drop(columns=["province_id"], errors="ignore")
    ccaa_row["GEOID"] = ccaa_dir.name
    ccaa_row["name"] = (names or {}).get("ccaa", ccaa_dir.name)
    for lvl, g in (("section", sec), ("municipality", mun), ("province", prov), ("ccaa", ccaa_row)):
        if g is not None and len(g):
            g.to_crs(4326).to_parquet(ccaa_dir / f"{lvl}.geoparquet")
            out[lvl] = str(ccaa_dir / f"{lvl}.geoparquet")
    return out


def _resample_up(fine: pd.DataFrame, res: int) -> pd.DataFrame:
    df = fine.copy()
    df["_parent"] = [h3.cell_to_parent(c, res) for c in df["h3"]]
    num = [c for c in df.columns if c not in ("h3", "_parent", "level_of_service") and pd.api.types.is_numeric_dtype(df[c])]
    rates = [c for c in num if census_spain.is_rate(c)]
    counts = [c for c in num if c not in rates]
    pop = pd.to_numeric(df["population"], errors="coerce").fillna(0)
    agg = {c: "sum" for c in counts}
    cols_w = [*rates, "level_of_service"] if "level_of_service" in df.columns else rates
    for c in cols_w:
        v = df[c]
        df[f"_w_{c}"] = (v.fillna(0.0) * pop).where(v.notna(), 0.0)
        df[f"_p_{c}"] = pop.where(v.notna(), 0.0)
        agg[f"_w_{c}"] = "sum"
        agg[f"_p_{c}"] = "sum"
    g = df.groupby("_parent").agg(agg)
    for c in cols_w:
        g[c] = g[f"_w_{c}"] / g[f"_p_{c}"].where(g[f"_p_{c}"] > 0)
    g = g.drop(columns=[c for c in g.columns if c.startswith(("_w_", "_p_"))])
    return g.reset_index().rename(columns={"_parent": "h3"})


def merge_and_resample_h3(ccaa_dir: Path, resolutions: Sequence[int] = CCAA_H3_RESOLUTIONS) -> dict:
    ccaa_dir = Path(ccaa_dir)
    written = {}
    dirs = province_dirs(ccaa_dir)

    def cat(res):
        fr = [pd.read_parquet(p / f"h3_res{res}.geoparquet") for p in dirs if (p / f"h3_res{res}.geoparquet").is_file()]
        return pd.concat(fr, ignore_index=True) if fr else None

    base = cat(BASE_RES)
    if base is None:
        logger.warning("no h3 res 7 files -- skipping H3 roll-up")
        return written
    for res in sorted(resolutions, reverse=True):
        out = _resample_up(base, res) if res < BASE_RES else (cat(res) if res >= BASE_RES else None)
        if out is None or out.empty:
            continue
        if res >= BASE_RES and out["h3"].duplicated().any():   # cells shared by two provinces: merge as a resample onto themselves
            out = _resample_up(out, res)
        from transitlos.neighbourhood import add_densities

        out = add_densities(out, "h3", "population")   # 500 m / 1 / 2 / 3 km neighbourhood densities, from this level's own cells
        out.to_parquet(ccaa_dir / f"h3_res{res}.geoparquet")
        written[res] = str(ccaa_dir / f"h3_res{res}.geoparquet")
    return written
