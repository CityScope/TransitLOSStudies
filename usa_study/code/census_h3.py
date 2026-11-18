"""Census / ACS / LODES variables on the H3 grid (for the map's stats panel: census ANOVA, regression, distribution).

The county pipeline writes H3 tables with population + election results only; the other census variables live on the
tract tables (`counties/<GEOID>/tract.geoparquet`). Here each res-7 cell takes the value of the tract containing its
centre, as intensive variables (shares, rates, medians, densities) -- raw counts are not comparable between cells.
Coarser levels are then population-weighted means of the res-7 cells (`state_pipeline._resample_h3_up`).
Names ending in `_share`/`_rate` show as "(%)" and `*_density` as "(pop/km2)" in the map.
"""
import functools
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
    ("male_share", "malePopulation", "population"),
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
    ("university_share", "acs5_educationBachelorsPlus", "acs5_population25plus"),   # template name of bachelors_plus_share
    ("education_basic_or_less_share", "acs5_educationLessThanHs", "acs5_population25plus"),   # template name of less_than_highschool_share
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
    # More ACS 5-year tables (2026-10): housing stock, household types, digital divide, health, nativity, commute detail, industry.
    ("vacant_housing_share", "acs5_vacantHousingUnits", "acs5_housingUnits"),
    ("detached_house_share", "acs5_unitsDetached1", "acs5_housingUnits"),
    ("attached_house_share", "acs5_unitsAttached1", "acs5_housingUnits"),
    ("small_multifamily_share", "acs5_unitsSmallMulti2to9", "acs5_housingUnits"),
    ("midsize_multifamily_share", "acs5_unitsMulti10to19", "acs5_housingUnits"),
    ("large_multifamily_share", "acs5_unitsMulti20Plus", "acs5_housingUnits"),
    ("mobile_home_share", "acs5_unitsMobileHome", "acs5_housingUnits"),
    ("family_households_share", "acs5_familyHouseholds", "households"),
    ("living_alone_households_share", "acs5_livingAloneHouseholds", "households"),
    ("snap_households_share", "acs5_snapHouseholds", "households"),
    ("broadband_households_share", "acs5_broadbandHouseholds", "households"),
    ("no_internet_households_share", "acs5_noInternetHouseholds", "households"),
    ("no_computer_households_share", "acs5_noComputerHouseholds", "households"),
    ("limited_english_households_share", "acs5_limitedEnglishHouseholds", "households"),
    ("car_households_share", "carHouseholds", "households"),
    ("two_plus_car_households_share", "twoCarHouseholds", "households"),
    ("veteran_share", "acs5_veteranPopulation", "acs5_civilianPopulation18plusVeterans"),
    ("disability_share", "acs5_disabilityPopulation", "acs5_disabilityUniverse"),
    ("uninsured_share", "acs5_uninsuredPopulation", "acs5_healthInsuranceUniverse"),
    ("non_citizen_share", "acs5_nonCitizenPopulation", "acs5_nativityUniverse"),
    ("naturalized_citizen_share", "acs5_naturalizedCitizenPopulation", "acs5_nativityUniverse"),
    ("employment_rate", "employedResidents", "acs5_population16plus"),
    ("drive_alone_commute_share", "acs5_driveAloneCommuters", "acs5_workers"),
    ("carpool_commute_share", "acs5_carpoolCommuters", "acs5_workers"),
    ("motorcycle_commute_share", "acs5_motorcycleCommuters", "acs5_workers"),
    ("work_from_home_share", "acs5_workFromHome", "acs5_workers"),
    ("commute_under_15_min_share", "acs5_commuteUnder15Minutes", "acs5_commuteTimeMean"),
    ("commute_60_plus_min_share", "acs5_commute60PlusMinutes", "acs5_commuteTimeMean"),
    ("industry_agriculture_mining_share", "acs5_industryAgriculture", "acs5_employedPopulation16plus"),
    ("industry_construction_share", "acs5_industryConstruction", "acs5_employedPopulation16plus"),
    ("industry_manufacturing_share", "acs5_industryManufacturing", "acs5_employedPopulation16plus"),
    ("industry_wholesale_share", "acs5_industryWholesale", "acs5_employedPopulation16plus"),
    ("industry_retail_share", "acs5_industryRetail", "acs5_employedPopulation16plus"),
    ("industry_transport_utilities_share", "acs5_industryTransportUtilities", "acs5_employedPopulation16plus"),
    ("industry_information_share", "acs5_industryInformation", "acs5_employedPopulation16plus"),
    ("industry_finance_real_estate_share", "acs5_industryFinanceRealEstate", "acs5_employedPopulation16plus"),
    ("industry_professional_services_share", "acs5_industryProfessionalServices", "acs5_employedPopulation16plus"),
    ("industry_education_health_share", "acs5_industryEducationHealth", "acs5_employedPopulation16plus"),
    ("industry_arts_food_share", "acs5_industryArtsFood", "acs5_employedPopulation16plus"),
    ("industry_other_services_share", "acs5_industryOtherServices", "acs5_employedPopulation16plus"),
    ("industry_public_admin_share", "acs5_industryPublicAdmin", "acs5_employedPopulation16plus"),
]
#: Already-intensive tract columns, renamed so the map recognises them as intensive (marker words: median/mean/index).
INTENSIVE = {
    "medianHouseholdIncome": "median_household_income",
    "acs5_incomeMeanCapita": "mean_income_per_capita",
    "medianAge": "median_age",
    "acs5_rentMedianGross": "median_gross_rent",
    "acs5_householdSizeMean": "mean_household_size",
    "acs5_incomeGini": "income_gini_index",
    "acs5_homeValueMedian": "median_home_value",
    "acs5_contractRentMedian": "median_contract_rent",
    "acs5_rentBurdenMedian": "median_rent_burden_percent",
    "acs5_yearBuiltMedian": "median_year_built",
    "acs5_roomsMedian": "median_rooms",
    "acs5_earningsMedianWorkers": "median_worker_earnings",
    "acs5_familyIncomeMedian": "median_family_income",
}
#: (output name, numerator column) -- per km2 of tract area.
DENSITIES = [
    ("population_density", "population"),
    ("jobs_density", "lodes_wac_jobsTotal"),
    ("housing_units_density", "dhc_housingUnits"),
]
#: County-level shares from the 2020 U.S. Religion Census (adherents as % of the population, by tradition) -> output name.
RELIGION_COLUMNS = {
    "usrc_adherents_share": "religious_adherents_share", "usrc_catholic_share": "catholic_share", "usrc_nondenominational_share": "nondenominational_christian_share",
    "usrc_southernBaptist_share": "southern_baptist_share", "usrc_lds_share": "latter_day_saints_share", "usrc_muslim_share": "muslim_share",
    "usrc_jewish_share": "jewish_share", "usrc_hindu_share": "hindu_share", "usrc_buddhist_share": "buddhist_share",
    "usrc_orthodoxChristian_share": "orthodox_christian_share", "usrc_congregations_per_100k_rate": "congregations_per_100k_rate",
}
#: Housing price / rent columns written onto the tract files by `add_housing_columns.py` (pycensus `usa.housing`); copied as they are (intensive).
HOUSING_COLUMNS = [
    f"{b}{s}" for s in ("", "_2016", "_2000") for b in (
        "home_value_median", "rent_median", "rent_burden_median", "owner_share", "renter_share", "rent_burdened_share", "rooms_median", "rent_per_room_mean")
    if not (b == "rent_burdened_share" and s == "_2000")
] + [f"house_price_index_fhfa{z}{s}" for z in ("", "_zip") for s in ("", "_2016", "_2006")] + ["house_price_cagr_2016_2025_rate", "listing_price_per_m2", "listing_median_area_m2"] + [
    f"hud_{k}_FY{fy}" for fy in (2019, 2022, 2026) for k in ("safmr_2br_rent_mean", "fmr_2br_county_rent_mean")
]
#: Year-built shares of dwellings (0-100 here like every other share of this table; the global schema's 0-1 is applied by the city pipeline): ACS B25034 bands
#: (`acs5_dwellingsBuilt*`, added to the tract files by `add_acs_columns.py`) -> `pycensus.year_built` (1945 splits the 1940-49 band 50/50).
YEAR_BUILT_COLUMNS = [
    "dwellings_built_before_1945_share", "dwellings_built_1945_1970_share", "dwellings_built_1970_2000_share", "dwellings_built_after_2000_share",
]
#: Block-group values (`add_commute_columns.py`): a cell takes those of the block group containing its centre.
BG_COLUMNS = ["commuteTimeResidentsMean", "commuteTimeWorkersMean", "commuteDistanceResidentsMean", "commuteDistanceWorkersMean",
              "commuteSpeedResidentsMean", "commuteSpeedWorkersMean",
              "losAtWorkplacesResidentsMean", "losAtHomesWorkersMean"]   # + `add_od_los_columns.py`: level of service at the other end of the commute
JOBS_COLUMNS = ["jobs", "pop_jobs_total", "pop_jobs_density"]   # jobs / population + jobs per CELL (LODES WAC tract density x cell area)
OUTPUT_COLUMNS = JOBS_COLUMNS + YEAR_BUILT_COLUMNS + HOUSING_COLUMNS + list(RELIGION_COLUMNS.values()) + [s[0] for s in SHARES] + list(INTENSIVE.values()) + [d[0] for d in DENSITIES] + ["mean_commute_time_minutes", "jobs_per_resident_ratio", "cars_per_household_ratio"] + BG_COLUMNS


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


@functools.lru_cache(maxsize=1)
def _county_religion() -> pd.DataFrame:
    """County religion table (parsed from the RCMS Excel file once per process: ~10 s each time otherwise)."""
    from pycensus.countries.usa.religion import load_county_religion

    return load_county_religion(data_dir=str(Path(__file__).resolve().parent.parent / "data" / "_cache" / "usa_religion")).set_index("GEOID")


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
    for name in HOUSING_COLUMNS:  # shares may be 0; prices / rents must be > 0
        v = num(name)
        out[name] = v if name.endswith(("_share", "_rate")) or "_share_" in name else v.where(v > 0)
    from pycensus.year_built import config, shares_from_bands

    bands = next(src["columns"] for src in config()["sources"] if src["name"] == "usa_acs5_B25034")
    yb = shares_from_bands(tr, bands, scale=100.0)   # NaN everywhere until `add_acs_columns.py` has added the B25034 bands
    for name in YEAR_BUILT_COLUMNS:
        out[name] = yb[name].to_numpy()
    area_km2 = tr.geometry.to_crs(5070).area / 1e6
    for name, col in DENSITIES:
        out[name] = num(col) / area_km2.where(area_km2 > 0)
    agg = tr["GEOID"].astype(str).map(commute_aggregates(str(tr["GEOID"].iloc[0])[:2]))
    commuters = num("acs5_commuteTimeMean")  # = B08303_001, workers who do not work at home
    out["mean_commute_time_minutes"] = (agg / commuters.where(commuters > 0)).where(lambda v: (v > 0) & (v < 180))
    out["jobs_per_resident_ratio"] = num("lodes_wac_jobsTotal") / num("population").where(num("population") > 0)
    rel = _county_religion()
    county = tr["GEOID"].astype(str).str[:5]
    for src, name in RELIGION_COLUMNS.items():
        out[name] = county.map(rel[src]).to_numpy()
    out["cars_per_household_ratio"] = (num("cars") / num("households").where(num("households") > 0)).where(lambda v: v < 10)
    return gpd.GeoDataFrame(out, geometry=tr.geometry.to_crs(4326).values, crs=4326)


def add_census_variables(res7: pd.DataFrame, state_dir: Path) -> pd.DataFrame:
    """`res7` (column `h3`) plus `OUTPUT_COLUMNS`, from the tract containing each cell's centre (NaN outside tracts)."""
    tracts = tract_variables(state_dir)
    if tracts.empty or res7.empty:
        return res7
    cells = res7["h3"].tolist()
    ll = np.asarray([h3.cell_to_latlng(c) for c in cells], dtype=np.float64).reshape(-1, 2)
    pts = gpd.GeoDataFrame({"_i": np.arange(len(cells))}, geometry=gpd.points_from_xy(ll[:, 1], ll[:, 0]), crs=4326)
    joined = gpd.sjoin(pts, tracts, how="left", predicate="within").drop_duplicates("_i").sort_values("_i")
    out = res7.reset_index(drop=True).copy()
    for col in OUTPUT_COLUMNS:
        if col not in BG_COLUMNS:
            out[col] = joined[col].to_numpy() if col in joined.columns else np.nan
    bg_files = sorted((state_dir / "counties").glob("*/blockgroup.geoparquet"))
    import pyarrow.parquet as pq

    bgs = [gpd.read_parquet(f, columns=["geometry", *BG_COLUMNS]) for f in bg_files if set(BG_COLUMNS) <= set(pq.read_schema(f).names)]
    if bgs:
        bg = gpd.GeoDataFrame(pd.concat(bgs, ignore_index=True), geometry="geometry", crs=bgs[0].crs).to_crs(4326)
        jb = gpd.sjoin(pts, bg, how="left", predicate="within").drop_duplicates("_i").sort_values("_i")
        for col in BG_COLUMNS:
            out[col] = jb[col].to_numpy()
    else:
        for col in BG_COLUMNS:
            out[col] = np.nan
    area = np.array([h3.cell_area(c, unit="km^2") for c in out["h3"]])
    dens = pd.to_numeric(joined["jobs_density"], errors="coerce").to_numpy() if "jobs_density" in joined.columns else np.full(len(out), np.nan)
    out["jobs"] = dens * area   # tract jobs density x this cell's area
    if "population" in out.columns:
        pop = pd.to_numeric(out["population"], errors="coerce").fillna(0).to_numpy()
        out["pop_jobs_total"] = pop + np.nan_to_num(out["jobs"].to_numpy())
        out["pop_jobs_density"] = out["pop_jobs_total"] / area
    return out
