"""Add the level of service at the other end of the commute to every census level of every finished state:

    losAtWorkplacesResidentsMean   (by home)        the mean level of service of the block groups where the area's working residents work
    losAtHomesWorkersMean          (by workplace)   the mean level of service of the block groups where the people who work in the area live

    python add_od_los_columns.py [StateDirName ...]
Each block group's level of service is its own `level_of_service` (the population-weighted mean of its cells); the means run over the LODES home -> work
flows between block groups up to 80 km apart (usa_flex/fit/data/torch/flows.npz), weighted by the workers of each flow. Block groups carry their own values,
blocks their block group's, tracts / counties / states the means of their block groups weighted by resident workers (first column) and by jobs (second),
h3 cells the values of the block group containing their centre (code/census_h3.BG_COLUMNS). Also writes TransitLOSStudies/cache/derived/bg_od_los.parquet
(the block-group values and weights, for the fits). Run after the counties are (re)computed: the values follow the level of service of the whole country.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
FLEX_FIT = next((p for p in (BASE.parent / "usa_flex" / "fit" / "data_buf", BASE.parent / "usa_flex" / "fit" / "data") if (p / "torch" / "flows.npz").is_file()), BASE.parent / "usa_flex" / "fit" / "data_buf")
OUT_TABLE = BASE.parent / "cache" / "derived" / "bg_od_los.parquet"
RES, WP = "losAtWorkplacesResidentsMean", "losAtHomesWorkersMean"
COLUMNS = [RES, WP]


def _bg_los_one(sd: Path) -> pd.Series:
    parts = []
    for f in sorted(sd.glob("counties/*/blockgroup.geoparquet")):
        b = pd.read_parquet(f, columns=["GEOID", "level_of_service"])
        parts.append(b)
    return pd.concat(parts).set_index("GEOID")["level_of_service"] if parts else pd.Series(dtype="float64")


def bg_values(states) -> pd.DataFrame:
    """Per block group: RES / WP and their weights (resident workers / jobs of the LODES flows that reach a block group with a level of service)."""
    with ProcessPoolExecutor(12) as ex:
        los = pd.concat(list(ex.map(_bg_los_one, states)))
    los = los[~los.index.duplicated()]
    geo = pd.read_parquet(FLEX_FIT / "blockgroups.parquet", columns=["GEOID"]).GEOID.astype(str).to_numpy()
    f = np.load(FLEX_FIT / "torch" / "flows.npz")
    h, w, n = f["h"].astype(np.int64), f["w"].astype(np.int64), f["n"].astype(np.float64)
    lv = pd.Series(geo).map(los).to_numpy(dtype="float64")   # level of service of every block group row
    out = {}
    for key, here, there in ((RES, h, w), (WP, w, h)):
        ok = np.isfinite(lv[there])
        num = np.bincount(here[ok], weights=n[ok] * lv[there][ok], minlength=len(geo))
        den = np.bincount(here[ok], weights=n[ok], minlength=len(geo))
        out[key] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        out[key + "_w"] = den
    t = pd.DataFrame(out, index=pd.Index(geo, name="GEOID"))
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    t.assign(level_of_service=pd.Series(geo).map(los).to_numpy()).to_parquet(OUT_TABLE)
    return t


def grouped(t: pd.DataFrame, key) -> pd.DataFrame:
    k = np.asarray(key)
    res = {}
    for c in COLUMNS:
        w = t[c + "_w"].where(t[c].notna(), 0).to_numpy()
        g = pd.DataFrame({"k": k, "v": t[c].fillna(0).to_numpy() * w, "w": w}).groupby("k").sum()
        res[c] = g.v / g.w.where(g.w > 0)
    return pd.DataFrame(res)


def write(path: Path, values: pd.DataFrame, key=None) -> None:
    g = gpd.read_parquet(path)
    k = g["GEOID"].astype(str).map(key) if callable(key) else g["GEOID"].astype(str)
    for c in COLUMNS:
        g[c] = k.map(values[c]).astype("float64").round(4).to_numpy()
    g.to_parquet(path)


def one_state(args) -> str:
    sd, t = args
    fips, n = None, 0
    for cd in sorted(p for p in (sd / "counties").iterdir() if p.is_dir()):
        fips = cd.name[:2]
        sub = t[t.index.str.startswith(cd.name)]
        if (cd / "blockgroup.geoparquet").is_file():
            write(cd / "blockgroup.geoparquet", sub)
        if (cd / "block.geoparquet").is_file():
            write(cd / "block.geoparquet", sub, lambda g: g[:12])
        if (cd / "tract.geoparquet").is_file():
            write(cd / "tract.geoparquet", grouped(sub, sub.index.str[:11]))
        if (cd / "county.geoparquet").is_file():
            write(cd / "county.geoparquet", grouped(sub, [cd.name] * len(sub)))
        n += 1
    st = t[t.index.str.startswith(fips)] if fips else t.iloc[:0]
    if (sd / "county.geoparquet").is_file():
        write(sd / "county.geoparquet", grouped(st, st.index.str[:5]))
    if (sd / "state.geoparquet").is_file():
        write(sd / "state.geoparquet", grouped(st, [fips] * len(st)), lambda g: g[:2])
    return f"{sd.name}: {n} counties"


def main():
    want = set(sys.argv[1:])
    alls = [d for d in sorted(DATA.iterdir()) if (d / "counties").is_dir() and not d.name.startswith("_")]
    t = bg_values(alls)   # always from the whole country: commutes cross state lines
    print(f"block groups with a value: residents {int(t[RES].notna().sum()):,}, workers {int(t[WP].notna().sum()):,}", flush=True)
    dirs = [d for d in alls if not want or d.name in want]
    with ProcessPoolExecutor(12) as ex:
        for r in ex.map(one_state, [(d, t) for d in dirs]):
            print(r, flush=True)


if __name__ == "__main__":
    main()
