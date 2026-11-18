"""Extra block-group controls for the gen-5 fits (2026-11-15, user: "could you try more controls?") -> data_buf/bg_extra2.parquet.

From every state's census block-group layers (usa_study/data/<State>/counties/*/blockgroup.geoparquet): neighbourhood density at 500 m and
3 km, housing type (multifamily 10+ units, detached), housing age (built before 1950), race / ethnicity (Black, Hispanic, Asian), foreign born,
household size, households with children, people living alone. Car commute speed by tract: LODES mean commute distance of the tract's resident
workers / ACS mean car commute time (B08134 bands, pycensus usa.acs_commute_time). Raw values; torch_fit4 transforms and standardises them.

    python fit/build_extra_controls.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
T = HERE.parents[1]
US = T / "usa_study" / "data"
CACHE = T / "cache"
OUT = HERE / "data_buf" / "bg_extra2.parquet"
COLS = ["GEOID", "population", "households", "pop_density_500m", "pop_density_3km", "dhc_blackPopulation", "dhc_hispanicPopulation",
        "dhc_asianPopulation", "foreignBornPopulation", "acs5_householdSizeMean", "acs5_householdsWithChildren", "acs5_livingAloneHouseholds",
        "acs5_housingUnits", "acs5_unitsDetached1", "acs5_unitsMulti10to19", "acs5_unitsMulti20Plus", "acs5_dwellingsBuiltTotal",
        "acs5_dwellingsBuilt1939OrEarlier", "acs5_dwellingsBuilt1940To1949"]


def main():
    parts = []
    for f in sorted(glob.glob(str(US / "*" / "counties" / "*" / "blockgroup.geoparquet"))):
        if "/_" in f:
            continue
        have = [c for c in COLS if c in pq.read_schema(f).names]
        parts.append(pd.read_parquet(f, columns=have))
    b = pd.concat(parts, ignore_index=True).drop_duplicates("GEOID").set_index("GEOID")
    num = lambda c: pd.to_numeric(b[c], errors="coerce") if c in b else pd.Series(np.nan, index=b.index)  # noqa: E731
    pop, hh, hu = num("population").where(num("population") > 0), num("households").where(num("households") > 0), num("acs5_housingUnits").where(num("acs5_housingUnits") > 0)
    blt = num("acs5_dwellingsBuiltTotal").where(num("acs5_dwellingsBuiltTotal") > 0)
    out = pd.DataFrame(index=b.index)
    out["pop_density_500m"] = num("pop_density_500m"); out["pop_density_3km"] = num("pop_density_3km")
    out["black_share"] = num("dhc_blackPopulation") / pop; out["hispanic_share"] = num("dhc_hispanicPopulation") / pop
    out["asian_share"] = num("dhc_asianPopulation") / pop; out["foreign_born_share"] = num("foreignBornPopulation") / pop
    out["household_size"] = num("acs5_householdSizeMean"); out["children_hh_share"] = num("acs5_householdsWithChildren") / hh
    out["living_alone_share"] = num("acs5_livingAloneHouseholds") / hh
    out["multifamily10_share"] = (num("acs5_unitsMulti10to19").fillna(0) + num("acs5_unitsMulti20Plus").fillna(0)) / hu
    out["detached_share"] = num("acs5_unitsDetached1") / hu
    out["built_pre1950_share"] = (num("acs5_dwellingsBuilt1939OrEarlier").fillna(0) + num("acs5_dwellingsBuilt1940To1949").fillna(0)) / blt
    # car commute speed by tract
    from pycensus.census.countries.usa.acs_commute_time import tract_table

    bg = pd.read_parquet(T / "cache" / "derived" / "bg_table.parquet", columns=["GEOID", "commute_km_residents", "resident_workers_lodes"]).set_index("GEOID")
    tr = bg.assign(tract=bg.index.str[:11], w=pd.to_numeric(bg["resident_workers_lodes"], errors="coerce").fillna(0),
                   km=pd.to_numeric(bg["commute_km_residents"], errors="coerce"))
    tr = tr[np.isfinite(tr["km"]) & (tr["w"] > 0)]
    km_tract = (tr["km"] * tr["w"]).groupby(tr["tract"]).sum() / tr["w"].groupby(tr["tract"]).sum()
    times = pd.concat([tract_table(f, cache_dir=str(CACHE)) for f in sorted({g[:2] for g in out.index})], ignore_index=True).set_index("GEOID")
    sp = (km_tract / (times["car_time_min"].reindex(km_tract.index) / 60.0)).replace([np.inf, -np.inf], np.nan)
    out["car_speed_kmh"] = sp.reindex(out.index.str[:11]).to_numpy()
    out["car_time_min"] = times["car_time_min"].reindex(out.index.str[:11]).to_numpy()
    out.reset_index().rename(columns={"index": "GEOID"}).to_parquet(OUT)
    print(out.describe().T[["count", "mean", "50%"]].round(3).to_string())
    print("wrote", OUT, len(out))


if __name__ == "__main__":
    main()
