"""Add the MITMA mobility columns (``ine_mitma_*``) to the finished community outputs without rerunning the transit pipeline.

    python add_mitma_mobility.py [Community ...]      # default: every finished community in data/

Sections get the district values; municipality / province / community rows are population-weighted means of their sections (counts are summed);
H3 cells take the value of the section holding the cell centre (rates only: counts are not apportioned to cells). Marker `mitma_mobility.done`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
sys.path.insert(0, str(BASE))
from code import census_spain  # noqa: E402

from code.study_cache import pycensus_dir, resolve_cache_dir  # noqa: E402

CACHE = str(pycensus_dir(resolve_cache_dir()))   # shared TransitLOSStudies/cache
MARK = "mitma_mobility.done"


def _is_count(c: str) -> bool:
    return c.startswith("ine_mitma_regular_flows_") and "share" not in c


def _write(df, path: Path) -> None:
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp)
    tmp.replace(path)


def _agg(sec: pd.DataFrame, key: pd.Series, cols) -> pd.DataFrame:
    w = pd.to_numeric(sec["population"], errors="coerce").fillna(0.0)
    out = {}
    for c in cols:
        v = pd.to_numeric(sec[c], errors="coerce")
        if _is_count(c):
            out[c] = v.groupby(key).sum(min_count=1)
        else:
            ok = v.notna()
            out[c] = (v.where(ok, 0) * w).groupby(key).sum() / w.where(ok, 0).groupby(key).sum().replace(0, np.nan)
    return pd.DataFrame(out)


def process(cdir: Path) -> str:
    if (cdir / MARK).exists():
        return "already done"
    t = census_spain.mitma_table(CACHE)
    cols = list(t.columns)
    sp = cdir / "section.geoparquet"
    sec = gpd.read_parquet(sp)
    sec = sec.drop(columns=[c for c in cols if c in sec.columns])
    sec = sec.merge(t, left_on=sec["GEOID"].astype(str).str.zfill(10).to_numpy(), right_index=True, how="left").drop(columns="key_0", errors="ignore")
    _write(sec, sp)
    g = sec["GEOID"].astype(str).str.zfill(10)
    for level, key in (("municipality", g.str[:5]), ("province", g.str[:2]), ("ccaa", pd.Series("all", index=sec.index))):
        path = cdir / f"{level}.geoparquet"
        if not path.is_file():
            continue
        df = gpd.read_parquet(path)
        a = _agg(sec, key, cols)
        gid = df["GEOID"].astype(str) if level != "ccaa" else pd.Series("all", index=df.index)
        df = df.drop(columns=[c for c in cols if c in df.columns])
        for c in cols:
            df[c] = gid.map(a[c]).to_numpy()
        _write(df, path)
    rates = [c for c in cols if not _is_count(c)]
    pts = gpd.GeoDataFrame(geometry=sec.geometry.make_valid().reset_index(drop=True), crs=sec.crs)
    rate_vals = sec[rates].reset_index(drop=True)
    for path in sorted(cdir.glob("h3_res*.geoparquet")):
        df = pd.read_parquet(path)
        cells = df["h3"].astype(str).to_numpy()
        ll = np.array([h3.cell_to_latlng(c) for c in cells])
        cp = gpd.GeoDataFrame(geometry=gpd.points_from_xy(ll[:, 1], ll[:, 0]), crs=4326)
        j = gpd.sjoin(cp, pts, how="left", predicate="within")
        j = j[~j.index.duplicated(keep="first")]
        idx = j["index_right"].to_numpy()
        df = df.drop(columns=[c for c in rates if c in df.columns])
        ok = ~np.isnan(idx.astype(float))
        for c in rates:
            col = np.full(len(df), np.nan)
            col[ok] = rate_vals[c].to_numpy()[idx[ok].astype(int)]
            df[c] = col
        _write(df, path)
    (cdir / MARK).write_text("ok\n")
    return f"ok ({len(sec)} sections, {len(cols)} columns)"


if __name__ == "__main__":
    names = sys.argv[1:] or [d.name for d in sorted((BASE / "data").iterdir()) if d.is_dir() and (d / "section.geoparquet").is_file() and not d.name.startswith("_")]
    for n in names:
        try:
            print(f"[mitma] {n}: {process(BASE / 'data' / n)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[mitma] {n}: FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
