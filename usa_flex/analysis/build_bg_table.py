"""One national block-group table: geometry facts, census covariates, ACS commute detail and LODES commute flows.

    python analysis/build_bg_table.py
Writes TransitLOSStudies/cache/derived/bg_table.parquet (one row per 2020 block group).

Commute time, residents (origin): ACS B08303 travel-time bins give a block-group mean (bin midpoints); it is rescaled so that the commuter-weighted mean of the block groups
of a tract equals the tract's exact mean (B08013 aggregate minutes / B08303 commuters), which is not published below tract level.
Commute time, workers (destination): LODES origin-destination flows (home block group -> work block group). Each origin's flows get a time proportional to
(8 min + straight-line km / 0.5 km per min), rescaled so that the origin's flow-weighted mean equals its residents' mean above; the destination mean is the flow-weighted
mean over its incoming flows. Flows over 80 km straight line (jobs coded at a distant head office) are left out. Also the flow-weighted straight-line commute distance (km) at origin and destination, and jobs per block group (LODES, all jobs)."""
import glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "usa_study" / "data"
CACHE = ROOT / "cache"
OUT = CACHE / "derived" / "bg_table.parquet"
KEEP = ["population", "households", "acs5_householdsOwner", "acs5_householdsRenter", "medianHouseholdIncome", "acs5_incomeMeanCapita", "povertyPopulation", "acs5_povertyBelow100",
        "acs5_workers", "carCommuters", "publicTransportCommuters", "walkCommuters", "bikeCommuters", "acs5_vehiclesHouseholds0", "acs5_vehiclesHouseholds1",
        "acs5_vehiclesHouseholds2", "acs5_vehiclesHouseholds3Plus", "acs5_population25plus", "acs5_educationBachelorsPlus", "under18Population", "over65Population",
        "acs5_population16plus", "laborForce", "unemployedResidents"]
MID = np.array([2.5, 7, 12, 17, 22, 27, 32, 37, 42, 52, 74.5, 105.0])


def one_county(f):
    g = gpd.read_parquet(f)
    cols = [c for c in KEEP if c in g.columns]
    out = pd.DataFrame(g[["GEOID", *cols]]).copy()
    p = g.geometry.to_crs(5070)
    out["area_km2"] = p.area.to_numpy() / 1e6
    c = p.representative_point().to_crs(4326)
    out["lon"], out["lat"] = c.x.to_numpy(), c.y.to_numpy()
    return out


def geo_table():
    files = sorted(glob.glob(str(STUDY / "*" / "counties" / "*" / "blockgroup.geoparquet")))
    with ProcessPoolExecutor(18) as ex:
        parts = list(ex.map(one_county, files, chunksize=8))
    t = pd.concat(parts, ignore_index=True).drop_duplicates("GEOID")
    for c in KEEP:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")
    return t


def acs_tables():
    bg = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(CACHE / "acs_bg_2023" / "bg_*.parquet")))], ignore_index=True)
    tr = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(CACHE / "acs_bg_2023" / "tract_*.parquet")))], ignore_index=True)
    bins = [f"B08303_{i:03d}E" for i in range(2, 14)]
    bb = bg[bins].to_numpy(float); bn = bb.sum(1)
    bg["commute_binmean"] = np.where(bn > 0, (bb * MID).sum(1) / np.maximum(bn, 1), np.nan)
    tb = tr[bins].to_numpy(float); tn = tb.sum(1)
    tr["tract_binmean"] = np.where(tn > 0, (tb * MID).sum(1) / np.maximum(tn, 1), np.nan)
    tr["tract_mean"] = np.where(tr["B08303_001E"] > 0, tr["B08013_001E"] / tr["B08303_001E"], np.nan)
    # the block groups' commuter-weighted bin mean in each tract (equals the tract bin mean up to rounding)
    bg["tract"] = bg.GEOID.str[:11]
    agg = bg.assign(m=bg.commute_binmean * bn).groupby("tract").agg(m=("m", "sum"), n=("B08303_001E", "sum"))
    bgm = (agg.m / agg.n.where(agg.n > 0)).rename("bg_binmean_tract")
    tr = tr.set_index("GEOID").join(bgm)
    fac = (tr.tract_mean / tr.bg_binmean_tract).where(lambda v: (v > 0.5) & (v < 2.0))
    bg["commute_time_residents_min"] = bg.commute_binmean * bg.tract.map(fac).fillna(1.0).to_numpy()
    bg.loc[bg["B08303_001E"].fillna(0) <= 0, "commute_time_residents_min"] = np.nan
    ren = {"B08303_001E": "commuters_not_wfh", "B08301_001E": "workers_b08301", "B08301_010E": "transit_commuters_b08301", "B08301_011E": "bus_commuters",
           "B08301_012E": "subway_commuters", "B08301_013E": "commuter_rail_commuters", "B08301_014E": "light_rail_commuters", "B08301_015E": "ferry_commuters",
           "B08301_021E": "work_from_home", "B08301_002E": "car_truck_van_commuters", "B08301_018E": "bike_commuters_b08301", "B08301_019E": "walk_commuters_b08301",
           "B25046_001E": "vehicles_aggregate", "B19013_001E": "median_hh_income_b19013", "C17002_001E": "poverty_universe", "C17002_002E": "poverty_below_50",
           "C17002_003E": "poverty_50_99"}
    keep = ["GEOID", "commute_time_residents_min", *ren] + bins
    out = bg[keep].rename(columns=ren)
    # tract-level means of transportation by vehicles available (B08141): transit commuters with no vehicle, etc.
    trk = tr[["tract_mean", "B08141_001E", "B08141_002E", "B08141_016E", "B08141_017E"]].rename(columns={
        "tract_mean": "tract_commute_time_mean", "B08141_001E": "tract_workers", "B08141_002E": "tract_workers_no_vehicle", "B08141_016E": "tract_transit_workers",
        "B08141_017E": "tract_transit_workers_no_vehicle"})
    return out, trk


def lodes_flows():
    files = sorted(glob.glob(str(CACHE / "lodes_od" / "*_od_*_JT00_*.csv.gz")))
    frames = []
    for f in files:
        d = (pl.read_csv(f, columns=["w_geocode", "h_geocode", "S000"], schema_overrides={"w_geocode": pl.Utf8, "h_geocode": pl.Utf8, "S000": pl.Int64})
             .with_columns(pl.col("w_geocode").str.slice(0, 12).alias("w"), pl.col("h_geocode").str.slice(0, 12).alias("h"))
             .group_by(["h", "w"]).agg(pl.col("S000").sum().alias("n")))
        frames.append(d)
    return pl.concat(frames).group_by(["h", "w"]).agg(pl.col("n").sum())


def main():
    geo = geo_table(); print("geometry", len(geo), flush=True)
    acs, trk = acs_tables(); print("acs", len(acs), flush=True)
    t = geo.merge(acs, on="GEOID", how="left")
    t = t.merge(trk, left_on=t.GEOID.str[:11], right_index=True, how="left").drop(columns="key_0", errors="ignore")
    fl = lodes_flows(); print("flows", fl.height, flush=True)
    xy = pl.DataFrame({"g": t.GEOID.to_numpy(), "lon": t.lon.to_numpy(), "lat": t.lat.to_numpy(), "tres": t.commute_time_residents_min.to_numpy()})
    f = (fl.join(xy.rename({"g": "h", "lon": "hlon", "lat": "hlat", "tres": "tres"}), on="h", how="inner")
           .join(xy.select(["g", "lon", "lat"]).rename({"g": "w", "lon": "wlon", "lat": "wlat"}), on="w", how="inner"))
    r = 6371.0
    f = f.with_columns((2 * r * (((pl.col("wlat") - pl.col("hlat")).radians() / 2).sin() ** 2 + (pl.col("hlat").radians().cos() * pl.col("wlat").radians().cos()
                       * (((pl.col("wlon") - pl.col("hlon")).radians() / 2).sin() ** 2))).sqrt().arcsin()).alias("km"))
    # LODES places some jobs at a distant head office (flows of thousands of km): only flows up to 80 km straight line count as daily commutes
    f = f.filter(pl.col("km") <= 80.0).with_columns(pl.max_horizontal(pl.col("km"), pl.lit(0.5)).alias("km")).with_columns((8.0 + pl.col("km") / 0.5).alias("g"))
    gbar = f.group_by("h").agg(((pl.col("g") * pl.col("n")).sum() / pl.col("n").sum()).alias("gbar"))
    f = f.join(gbar, on="h").with_columns((pl.col("tres") * pl.col("g") / pl.col("gbar")).alias("t"))
    wagg = f.group_by("w").agg(pl.col("n").sum().alias("jobs_lodes"),
                               ((pl.col("t") * pl.col("n")).filter(pl.col("t").is_not_nan() & pl.col("t").is_not_null()).sum()
                                / pl.col("n").filter(pl.col("t").is_not_nan() & pl.col("t").is_not_null()).sum()).alias("commute_time_workers_min"),
                               ((pl.col("km") * pl.col("n")).sum() / pl.col("n").sum()).alias("commute_km_workers"))
    hagg = f.group_by("h").agg(pl.col("n").sum().alias("resident_workers_lodes"), ((pl.col("km") * pl.col("n")).sum() / pl.col("n").sum()).alias("commute_km_residents"))
    t = t.merge(wagg.to_pandas().rename(columns={"w": "GEOID"}), on="GEOID", how="left").merge(hagg.to_pandas().rename(columns={"h": "GEOID"}), on="GEOID", how="left")
    t["jobs_lodes"] = t.jobs_lodes.fillna(0.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(OUT)
    d = t
    w = d.commuters_not_wfh.fillna(0)
    print("rows", len(d), "res mean (commuter weighted)", float((d.commute_time_residents_min * w).sum() / w[d.commute_time_residents_min.notna()].sum()),
          "| wp mean (job weighted)", float((d.commute_time_workers_min * d.jobs_lodes).sum() / d.jobs_lodes[d.commute_time_workers_min.notna()].sum()), flush=True)
    print(d[["commute_time_residents_min", "commute_time_workers_min", "commute_km_residents", "commute_km_workers"]].describe().T, flush=True)


if __name__ == "__main__":
    main()
