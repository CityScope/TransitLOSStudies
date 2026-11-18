"""Extra block-group columns for the transit-vs-car score fit (fit/torch_fit4.py), written to fit/data_buf/bg_extra.parquet:

  pop_density_1km   residents per km2 within 1 km, population-weighted mean over the block group's flex cells (res-10 walk-zone cells and the
                    res-8 cells outside, each res-8 cell spread evenly over its 49 res-10 children; transitlos.neighbourhood.density_within)
  fuel_usd_gal      mean retail motor-gasoline price of the state over the ACS 5-year period (EIA SEDS; pycensus usa.fuel_prices)
  region, division  US Census region / division of the state (the universality test fits each one alone)

    python fit/build_bg_extra.py [first_year last_year]      # ACS period, default 2019 2023
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_buf"
OUT = HERE / "data_buf" / "bg_extra.parquet"
CACHE = HERE.parents[1] / "cache"

REGION = {  # state FIPS -> (region, division)
    **{f: ("Northeast", "New England") for f in ("09", "23", "25", "33", "44", "50")},
    **{f: ("Northeast", "Middle Atlantic") for f in ("34", "36", "42")},
    **{f: ("Midwest", "East North Central") for f in ("17", "18", "26", "39", "55")},
    **{f: ("Midwest", "West North Central") for f in ("19", "20", "27", "29", "31", "38", "46")},
    **{f: ("South", "South Atlantic") for f in ("10", "11", "12", "13", "24", "37", "45", "51", "54")},
    **{f: ("South", "East South Central") for f in ("01", "21", "28", "47")},
    **{f: ("South", "West South Central") for f in ("05", "22", "40", "48")},
    **{f: ("West", "Mountain") for f in ("04", "08", "16", "30", "32", "35", "49", "56")},
    **{f: ("West", "Pacific") for f in ("02", "06", "15", "41", "53")},
}
FIPS_USPS = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
             "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
             "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
             "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
             "54": "WV", "55": "WI", "56": "WY"}


def state_density(cells: pd.DataFrame) -> pd.DataFrame:
    """bg, pop, pop_density_1km for every res-10 cell (res-8 cells expanded into their 49 children)."""
    from transitlos.neighbourhood import density_within

    fine = cells[cells["res"] == 10][["h3", "bg", "population"]]
    coarse = cells[cells["res"] == 8][["h3", "bg", "population"]]
    rows = [fine]
    if len(coarse):
        ch = [(c, b, p / 49.0) for h, b, p in coarse.itertuples(index=False) for c in h3.cell_to_children(str(h), 10)]
        rows.append(pd.DataFrame(ch, columns=["h3", "bg", "population"]))
    t = pd.concat(rows, ignore_index=True).drop_duplicates("h3")
    t["d1km"] = density_within(t["h3"].astype(str).to_numpy(), t["population"].to_numpy(dtype=float))
    return t


def main(y0: int = 2019, y1: int = 2023) -> None:
    from pycensus.countries.usa.fuel_prices import gasoline_price_mean

    fuel = gasoline_price_mean(y0, y1, str(CACHE))
    out = []
    for sd in sorted(p for p in DATA.iterdir() if (p / "cells.parquet").is_file()):
        t0 = time.time()
        c = pd.read_parquet(sd / "cells.parquet", columns=["h3", "res", "bg", "population"])
        c = c[c["bg"].notna()]
        t = state_density(c)
        w = t["population"].clip(lower=1e-6)
        g = (t.assign(wd=w * t["d1km"], w=w).groupby("bg")[["wd", "w"]].sum())
        part = pd.DataFrame({"GEOID": g.index.astype(str), "pop_density_1km": (g["wd"] / g["w"]).to_numpy()})
        out.append(part)
        print(f"{sd.name}: {len(part):,} block groups, {len(t):,} cells ({time.time() - t0:.0f}s)", flush=True)
    bg = pd.concat(out, ignore_index=True).drop_duplicates("GEOID")
    st = bg["GEOID"].str[:2]
    bg["fuel_usd_gal"] = st.map(FIPS_USPS).map(fuel)
    bg["region"] = st.map(lambda f: REGION.get(f, (None, None))[0])
    bg["division"] = st.map(lambda f: REGION.get(f, (None, None))[1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    bg.to_parquet(OUT)
    print(f"wrote {OUT}: {len(bg):,} block groups; fuel {bg['fuel_usd_gal'].min():.2f}-{bg['fuel_usd_gal'].max():.2f} $/gal; "
          f"1 km density median {bg['pop_density_1km'].median():,.0f}/km2", flush=True)


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:3]] if len(sys.argv) >= 3 else [2019, 2023]
    main(*a)
