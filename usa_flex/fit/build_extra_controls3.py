"""More block-group controls for the control study (2026-11-16, user: "look at pycensus ... maybe you can find more controls")
-> data_buf/bg_extra3.parquet. From the pycensus ACS 5-year columns of every state's block-group layers (usa_study/data/<State>/counties/*/
blockgroup.geoparquet). Values are already transformed (shares, or log of money / counts); torch_fit4 standardises them (FIT4_CTRL_ADD).

    python fit/build_extra_controls3.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
US = HERE.parents[1] / "usa_study" / "data"
OUT = HERE / "data_buf" / "bg_extra3.parquet"
A = "acs5_"
RAW = ["households", "population", "incomeGini", "incomeHouseholdsUnder10k", "incomeHouseholds10kTo25k", "incomeHouseholds200kPlus", "snapHouseholds",
       "employedPopulation16plus", "jobsService", "jobsProductionTransportationMaterialMoving", "jobsManagementBusinessScienceArts",
       "employedFemalePopulation16plus", "disabilityUniverse", "disabilityPopulation", "healthInsuranceUniverse", "uninsuredPopulation",
       "limitedEnglishHouseholds", "nativityUniverse", "nonCitizenPopulation", "civilianPopulation18plusVeterans", "veteranPopulation",
       "noComputerHouseholds", "housingUnits", "vacantHousingUnits", "unitsMobileHome", "unitsSmallMulti2to9", "unitsAttached1", "roomsMedian",
       "homeValueMedian", "rentMedianGross", "rentBurdenMedian", "workers", "workFromHome",
       "yearBuiltMedian", "dwellingsBuiltTotal", "dwellingsBuilt1939OrEarlier", "dwellingsBuilt1940To1949", "dwellingsBuilt1950To1959",
       "dwellingsBuilt1960To1969", "dwellingsBuilt1970To1979", "dwellingsBuilt2000To2009", "dwellingsBuilt2010To2019", "dwellingsBuilt2020OrLater"]


def main():
    parts = []
    want = ["GEOID", "population", "households", "pop_density_2km", "pop_density_4km", "pop_density_5km"] + [A + c for c in RAW]
    for f in sorted(glob.glob(str(US / "*" / "counties" / "*" / "blockgroup.geoparquet"))):
        if "/_" in f:
            continue
        have = [c for c in want if c in pq.read_schema(f).names]
        parts.append(pd.read_parquet(f, columns=have))
    b = pd.concat(parts, ignore_index=True).drop_duplicates("GEOID").set_index("GEOID")

    def num(c):
        for k in (A + c, c):
            if k in b:
                return pd.to_numeric(b[k], errors="coerce")
        return pd.Series(np.nan, index=b.index)

    def share(a, u):
        den = num(u); return num(a) / den.where(den > 0) if isinstance(a, str) else a / den.where(den > 0)

    hh = "households"
    o0 = {}
    for r in ("2km", "4km", "5km"):   # neighbourhood density (user 2026-11-16: "Density include the 4 and 5km")
        o0[f"log_pop_density_{r}"] = np.log1p(num(f"pop_density_{r}"))
    # housing age beyond "built before 1950" (user 2026-11-16: "built before 1980 or a similar boundary too")
    blt = lambda *cs: sum(num(f"dwellingsBuilt{c}").fillna(0) for c in cs)  # noqa: E731
    tot = num("dwellingsBuiltTotal").where(num("dwellingsBuiltTotal") > 0)
    o0["built_pre1980_share"] = blt("1939OrEarlier", "1940To1949", "1950To1959", "1960To1969", "1970To1979") / tot
    o0["built_post2000_share"] = blt("2000To2009", "2010To2019", "2020OrLater") / tot
    o0["year_built_median"] = num("yearBuiltMedian").where(num("yearBuiltMedian") > 1800)
    o = pd.DataFrame(o0, index=b.index)
    # B19083 is published by tract only; the block-group layers hold it split over the tract's block groups like a count (the pieces sum
    # to the tract's index), so the tract sum is the block group's Gini
    g = num("incomeGini"); o["income_gini"] = g.groupby(b.index.str[:11]).transform(lambda x: x.sum(min_count=1))
    o["lowinc_share"] = share(num("incomeHouseholdsUnder10k").fillna(0) + num("incomeHouseholds10kTo25k").fillna(0), hh)
    o["highinc_share"] = share("incomeHouseholds200kPlus", hh)
    o["snap_share"] = share("snapHouseholds", hh)
    o["service_occ_share"] = share("jobsService", "employedPopulation16plus")
    o["production_occ_share"] = share("jobsProductionTransportationMaterialMoving", "employedPopulation16plus")
    o["management_occ_share"] = share("jobsManagementBusinessScienceArts", "employedPopulation16plus")
    o["female_worker_share"] = share("employedFemalePopulation16plus", "employedPopulation16plus")
    o["disability_share"] = share("disabilityPopulation", "disabilityUniverse")
    o["uninsured_share"] = share("uninsuredPopulation", "healthInsuranceUniverse")
    o["limited_english_share"] = share("limitedEnglishHouseholds", hh)
    o["noncitizen_share"] = share("nonCitizenPopulation", "nativityUniverse")
    o["veteran_share"] = share("veteranPopulation", "civilianPopulation18plusVeterans")
    o["no_computer_share"] = share("noComputerHouseholds", hh)
    o["vacancy_share"] = share("vacantHousingUnits", "housingUnits")
    o["mobile_home_share"] = share("unitsMobileHome", "housingUnits")
    o["small_multi_share"] = share("unitsSmallMulti2to9", "housingUnits")
    o["attached_share"] = share("unitsAttached1", "housingUnits")
    o["rooms_median"] = num("roomsMedian")
    o["log_home_value"] = np.log(num("homeValueMedian").where(num("homeValueMedian") > 0))
    o["log_rent"] = np.log(num("rentMedianGross").where(num("rentMedianGross") > 0))
    o["rent_burden_median"] = num("rentBurdenMedian")
    o["wfh_share"] = share("workFromHome", "workers")
    for c in o:
        o[c] = o[c].replace([np.inf, -np.inf], np.nan).astype("float64")
    o.reset_index().rename(columns={"index": "GEOID"}).to_parquet(OUT)
    print(o.describe().T[["count", "mean", "50%"]].round(3).to_string())
    print("wrote", OUT, len(o))


if __name__ == "__main__":
    main()
