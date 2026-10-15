"""Census / ACS / LODES variables on the H3 grid (for the map's stats panel: census ANOVA, regression, distribution).

The county pipeline writes H3 tables with population + election results only; the other census variables live on the
tract tables (`counties/<GEOID>/tract.geoparquet`). Here each res-7 cell takes the value of the tract containing its
centre, as intensive variables (shares, rates, medians, densities) -- raw counts are not comparable between cells.
Coarser levels are then population-weighted means of the res-7 cells (`state_pipeline._resample_h3_up`).
Names ending in `_share`/`_rate` show as "(%)" and `*_density` as "(pop/km2)" in the map.
"""
import logging
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# (output name, numerator column(s), denominator column) -- ratios as 0-100 where the name ends in _share / _rate.
SHARES = [
    ("nonwhite_share", "dhc_nonwhitePopulation", "population"),
    ("white_share", "dhc_whitePopulation", "population"),
    ("black_share", "dhc_blackPopulation", "population"),
    ("asian_share", "dhc_asianPopulation", "population"),
    ("native_share", "dhc_nativePopulation", "population"),
    ("hispanic_share", "dhc_hispanicPopulation", "population"),
    ("foreign_born_share", "foreignBornPopulation", "population"),
    ("female_share", "femalePopulation", "population"),
    ("under18_share", "under18Population", "population"),
    ("age18to64_share", "dhc_age18to64Population", "population"),
    ("over65_share", "over65Population", "population"),
    ("owner_households_share", "acs5_householdsOwner", "households"),
    ("renter_households_share", "acs5_householdsRenter", "households"),
    ("households_with_children_share", "acs5_householdsWithChildren", "households"),
    ("zero_vehicle_households_share", "acs5_vehiclesHouseholds0", "households"),
    ("one_vehicle_households_share", "acs5_vehiclesHouseholds1", "households"),
    ("two_vehicle_households_share", "acs5_vehiclesHouseholds2", "households"),
    ("three_plus_vehicle_households_share", "acs5_vehiclesHouseholds3Plus", "households"),
    ("poverty_share", "acs5_povertyBelow100", "povertyPopulation"),
    ("unemployment_rate", "unemployedResidents", "laborForce"),
    ("car_commute_share", "carCommuters", "acs5_workers"),
    ("transit_commute_share", "publicTransportCommuters", "acs5_workers"),
    ("walk_commute_share", "walkCommuters", "acs5_workers"),
    ("bike_commute_share", "bikeCommuters", "acs5_workers"),
    ("rent_burdened_share", "acs5_rentBurdened", "acs5_rentBurdenPopulation"),
    ("less_than_highschool_share", "acs5_educationLessThanHs", "acs5_population25plus"),
    ("highschool_share", "acs5_educationHighSchool", "acs5_population25plus"),
    ("some_college_share", "acs5_educationSomeCollege", "acs5_population25plus"),
    ("bachelors_plus_share", "acs5_educationBachelorsPlus", "acs5_population25plus"),
    ("income_under_10k_share", "acs5_incomeHouseholdsUnder10k", "households"),
    ("income_10k_25k_share", "acs5_incomeHouseholds10kTo25k", "households"),
    ("income_25k_50k_share", "acs5_incomeHouseholds25kTo50k", "households"),
    ("income_50k_100k_share", "acs5_incomeHouseholds50kTo100k", "households"),
    ("income_100k_200k_share", "acs5_incomeHouseholds100kTo200k", "households"),
    ("income_200k_plus_share", "acs5_incomeHouseholds200kPlus", "households"),
    ("occupation_management_share", "acs5_jobsManagementBusinessScienceArts", "acs5_employedPopulation16plus"),
    ("occupation_service_share", "acs5_jobsService", "acs5_employedPopulation16plus"),
    ("occupation_sales_office_share", "acs5_jobsSalesOffice", "acs5_employedPopulation16plus"),
    ("occupation_construction_share", "acs5_jobsNaturalResourcesConstructionMaintenance", "acs5_employedPopulation16plus"),
    ("occupation_production_transport_share", "acs5_jobsProductionTransportationMaterialMoving", "acs5_employedPopulation16plus"),
    ("jobs_income_low_share", "lodes_wac_jobsIncomeLow", "lodes_wac_jobsTotal"),
    ("jobs_income_mid_share", "lodes_wac_jobsIncomeMid", "lodes_wac_jobsTotal"),
    ("jobs_income_high_share", "lodes_wac_jobsIncomeHigh", "lodes_wac_jobsTotal"),
    ("jobs_age_young_share", "lodes_wac_jobsAgeYoung", "lodes_wac_jobsTotal"),
    ("jobs_age_prime_share", "lodes_wac_jobsAgePrime", "lodes_wac_jobsTotal"),
    ("jobs_age_older_share", "lodes_wac_jobsAgeOlder", "lodes_wac_jobsTotal"),
]
#: Already-intensive tract columns, renamed so the map recognises them as intensive (marker words: median/mean/index).
INTENSIVE = {
    "medianHouseholdIncome": "median_household_income",
    "acs5_incomeMeanCapita": "mean_income_per_capita",
    "medianAge": "median_age",
    "acs5_rentMedianGross": "median_gross_rent",
    "acs5_householdSizeMean": "mean_household_size",
    "acs5_incomeGini": "income_gini_index",
}
#: (output name, numerator column) -- per km2 of tract area.
DENSITIES = [
    ("population_density", "population"),
    ("jobs_density", "lodes_wac_jobsTotal"),
    ("housing_units_density", "dhc_housingUnits"),
]
OUTPUT_COLUMNS = [s[0] for s in SHARES] + list(INTENSIVE.values()) + [d[0] for d in DENSITIES] + ["mean_commute_time_minutes", "jobs_per_resident_ratio"]


ACS_YEAR = 2023  # vintage pyCensus' acs5 columns come from (verified: B08303_001 equals `acs5_commuteTimeMean` exactly)
_COMMUTE_CACHE = Path(__file__).resolve().parent.parent / "data" / "_cache" / "acs_commute_aggregate.json"


def commute_aggregates(state_fips: str) -> dict:
    """`{tract GEOID: aggregate travel time to work (minutes)}` (ACS B08013_001) for one state, cached on disk.

    pyCensus' `acs5_commuteTimeMean` is mapped to B08303_001, which is the NUMBER of commuters (not a mean); dividing the
    aggregate minutes by that count gives the real mean commute time.
    """
    import json

    cache = json.loads(_COMMUTE_CACHE.read_text()) if _COMMUTE_CACHE.is_file() else {}
    if state_fips not in cache:
        import requests

        from pycensus.api_keys import resolve_api_key

        r = requests.get(
            f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5",
            params={"get": "B08013_001E", "for": "tract:*", "in": f"state:{state_fips} county:*", "key": resolve_api_key("us_census", "US_CENSUS_API_KEY")},
            timeout=120,
        )
        r.raise_for_status()
        rows = r.json()[1:]
        cache[state_fips] = {st + co + tr: float(v) for v, st, co, tr in rows if v not in (None, "-666666666")}
        _COMMUTE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _COMMUTE_CACHE.write_text(json.dumps(cache))
    return cache[state_fips]


def tract_variables(state_dir: Path) -> gpd.GeoDataFrame:
    """One state's tracts with every derived variable (`OUTPUT_COLUMNS`), EPSG:4326."""
    frames = [gpd.read_parquet(p) for p in sorted((state_dir / "counties").glob("*/tract.geoparquet"))]
    if not frames:
        return gpd.GeoDataFrame()
    tr = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)
    num = lambda c: pd.to_numeric(tr[c], errors="coerce") if c in tr.columns else pd.Series(np.nan, index=tr.index)  # noqa: E731
    out = pd.DataFrame(index=tr.index)
    for name, top, bottom in SHARES:
        den = num(bottom)
        out[name] = (100.0 * num(top) / den.where(den > 0)).clip(0, 100)
    for col, name in INTENSIVE.items():
        v = num(col)
        out[name] = v.where(v > 0)  # Census sentinels / zeros mean "no data"
    area_km2 = tr.geometry.to_crs(5070).area / 1e6
    for name, col in DENSITIES:
        out[name] = num(col) / area_km2.where(area_km2 > 0)
    agg = tr["GEOID"].astype(str).map(commute_aggregates(str(tr["GEOID"].iloc[0])[:2]))
    commuters = num("acs5_commuteTimeMean")  # = B08303_001, workers who do not work at home
    out["mean_commute_time_minutes"] = (agg / commuters.where(commuters > 0)).where(lambda v: (v > 0) & (v < 180))
    out["jobs_per_resident_ratio"] = num("lodes_wac_jobsTotal") / num("population").where(num("population") > 0)
    return gpd.GeoDataFrame(out, geometry=tr.geometry.to_crs(4326).values, crs=4326)


def add_census_variables(res7: pd.DataFrame, state_dir: Path) -> pd.DataFrame:
    """`res7` (column `h3`) plus `OUTPUT_COLUMNS`, from the tract containing each cell's centre (NaN outside tracts)."""
    tracts = tract_variables(state_dir)
    if tracts.empty or res7.empty:
        return res7
    cells = res7["h3"].tolist()
    pts = gpd.GeoDataFrame(
        {"_i": np.arange(len(cells))}, geometry=gpd.points_from_xy(*zip(*[(h3.cell_to_latlng(c)[1], h3.cell_to_latlng(c)[0]) for c in cells])),
        crs=4326,
    )
    joined = gpd.sjoin(pts, tracts, how="left", predicate="within").drop_duplicates("_i").sort_values("_i")
    out = res7.reset_index(drop=True).copy()
    for col in OUTPUT_COLUMNS:
        out[col] = joined[col].to_numpy() if col in joined.columns else np.nan
    return out
