"""Add the mean commute time, straight-line distance (LODES flows up to 80 km) and speed (distance / time, km/h) of residents (by home) and of workers
(by workplace) to every census level of every finished state.

    python add_commute_columns.py [StateDirName ...]
Source: TransitLOSStudies/cache/derived/bg_table.parquet (usa_flex/analysis/build_bg_table.py). Block groups carry their own values, blocks their block group's,
tracts the exact ACS tract mean for residents (B08013 / B08303) and the job-weighted mean of their block groups for workers, counties and states the
commuter-weighted (residents) and job-weighted (workers) means of their block groups."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
TABLE = BASE.parent / "cache" / "derived" / "bg_table.parquet"
RES, WP = "commuteTimeResidentsMean", "commuteTimeWorkersMean"
KRES, KWP = "commuteDistanceResidentsMean", "commuteDistanceWorkersMean"
SRES, SWP = "commuteSpeedResidentsMean", "commuteSpeedWorkersMean"
COLUMNS = [RES, WP, KRES, KWP, SRES, SWP]


def bg_values() -> pd.DataFrame:
    t = pd.read_parquet(TABLE, columns=["GEOID", "commute_time_residents_min", "commute_time_workers_min", "commuters_not_wfh", "jobs_lodes", "tract_commute_time_mean",
                                        "commute_km_residents", "commute_km_workers", "resident_workers_lodes"])
    t = t.rename(columns={"commute_time_residents_min": RES, "commute_time_workers_min": WP, "commute_km_residents": KRES, "commute_km_workers": KWP}).set_index("GEOID")
    t[SRES] = 60.0 * t[KRES] / t[RES].where(t[RES] > 0)   # km/h: straight-line distance over door-to-door time
    t[SWP] = 60.0 * t[KWP] / t[WP].where(t[WP] > 0)
    return t


def grouped(t: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
    """Commuter-weighted residents' means and job-weighted workers' means of the block groups grouped by `key`; speeds = mean distance / mean time."""
    k = key.to_numpy()

    def wmean(col, wcol):
        w = t[wcol].fillna(0).where(t[col].notna(), 0)
        g = pd.DataFrame({"k": k, "v": (t[col].fillna(0) * w).to_numpy(), "w": w.to_numpy()}).groupby("k").sum()
        return g.v / g.w.where(g.w > 0)
    out = pd.DataFrame({RES: wmean(RES, "commuters_not_wfh"), WP: wmean(WP, "jobs_lodes"), KRES: wmean(KRES, "resident_workers_lodes"), KWP: wmean(KWP, "jobs_lodes")})
    out[SRES] = 60.0 * out[KRES] / out[RES].where(out[RES] > 0)
    out[SWP] = 60.0 * out[KWP] / out[WP].where(out[WP] > 0)
    return out


def write(path: Path, values: pd.DataFrame, key) -> None:
    g = gpd.read_parquet(path)
    k = g["GEOID"].astype(str).map(key) if callable(key) else g["GEOID"].astype(str)
    for c in COLUMNS:
        g[c] = k.map(values[c]).astype("float64").round(2).to_numpy()
    g.to_parquet(path)


def one_state(sd: Path) -> str:
    t = bg_values()
    fips = None
    n = 0
    for cd in sorted(p for p in (sd / "counties").iterdir() if p.is_dir()):
        fips = cd.name[:2]
        sub = t[t.index.str.startswith(cd.name)]
        if (cd / "blockgroup.geoparquet").is_file():
            write(cd / "blockgroup.geoparquet", sub, None)
        if (cd / "block.geoparquet").is_file():
            write(cd / "block.geoparquet", sub, lambda g: g[:12])
        if (cd / "tract.geoparquet").is_file():
            tr = grouped(sub, sub.index.str[:11].to_series(index=sub.index))
            exact = sub.groupby(sub.index.str[:11]).tract_commute_time_mean.first()
            tr[RES] = exact.reindex(tr.index).where(lambda v: (v > 0) & (v < 180)).fillna(tr[RES])
            tr[SRES] = 60.0 * tr[KRES] / tr[RES].where(tr[RES] > 0)
            write(cd / "tract.geoparquet", tr, None)
        if (cd / "county.geoparquet").is_file():
            write(cd / "county.geoparquet", grouped(sub, pd.Series(cd.name, index=sub.index)), None)
        n += 1
    st = t[t.index.str.startswith(fips)] if fips else t.iloc[:0]
    if (sd / "county.geoparquet").is_file():
        write(sd / "county.geoparquet", grouped(st, st.index.str[:5].to_series(index=st.index)), None)
    if (sd / "state.geoparquet").is_file():
        write(sd / "state.geoparquet", grouped(st, pd.Series(fips, index=st.index)), lambda g: g[:2])
    return f"{sd.name}: {n} counties"


def main():
    want = set(sys.argv[1:])
    dirs = [d for d in sorted(DATA.iterdir()) if (d / "counties").is_dir() and not d.name.startswith("_") and (not want or d.name in want)]
    with ProcessPoolExecutor(12) as ex:
        for r in ex.map(one_state, dirs):
            print(r, flush=True)


if __name__ == "__main__":
    main()
