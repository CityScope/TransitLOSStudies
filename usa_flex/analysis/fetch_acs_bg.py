"""ACS 2023 5-year tables at block group (and tract) level for every state, cached in TransitLOSStudies/cache/acs_bg_2023/.

    python analysis/fetch_acs_bg.py
B08303 travel time bins, B08301 means of transportation (bus / subway / commuter rail / light rail...), B25044 tenure x vehicles, B25046 aggregate vehicles, B19013 median household
income, C17002 poverty ratio, B11001 households, B01003 population, B08013 aggregate travel time (tract only: null at block group level)."""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
from pycensus.api_keys import resolve_api_key  # noqa: E402

OUT = ROOT / "TransitLOSStudies" / "cache" / "acs_bg_2023"
KEY = resolve_api_key("us_census", "US_CENSUS_API_KEY")
BG_VARS = ([f"B08303_{i:03d}E" for i in range(1, 14)] + [f"B08301_{i:03d}E" for i in range(1, 22)] + [f"B25044_{i:03d}E" for i in range(1, 16)]
           + ["B25046_001E", "B19013_001E", "C17002_001E", "C17002_002E", "C17002_003E", "B11001_001E", "B01003_001E"])
TRACT_VARS = ["B08013_001E"] + [f"B08303_{i:03d}E" for i in range(1, 14)] + [f"B08141_{i:03d}E" for i in range(1, 21)]
FIPS = ["01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
        "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "44", "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56", "72"]


def get(vars_, geo, inn):
    for attempt in range(5):
        try:
            r = requests.get("https://api.census.gov/data/2023/acs/acs5", params={"get": ",".join(vars_), "for": geo, "in": inn, "key": KEY}, timeout=300)
            r.raise_for_status()
            rows = r.json()
            return pd.DataFrame(rows[1:], columns=rows[0])
        except Exception as exc:  # noqa: BLE001
            print("retry", geo, inn, exc, flush=True); time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed {geo} {inn}")


def fetch(level, vars_, fips):
    path = OUT / f"{level}_{fips}.parquet"
    if path.is_file():
        return
    geo, inn = ("block group:*", f"state:{fips} county:* tract:*") if level == "bg" else ("tract:*", f"state:{fips} county:*")
    parts = []
    for i in range(0, len(vars_), 45):
        df = get(vars_[i:i + 45], geo, inn)
        keys = [c for c in ("state", "county", "tract", "block group") if c in df.columns]
        df["GEOID"] = df[keys].astype(str).agg("".join, axis=1)
        parts.append(df.drop(columns=keys).set_index("GEOID"))
    out = pd.concat(parts, axis=1).apply(pd.to_numeric, errors="coerce")
    out = out.mask(out < -1000)   # Census sentinels (-666666666 ...)
    out.reset_index().to_parquet(path)
    print(level, fips, len(out), flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("tract", TRACT_VARS, f) for f in FIPS] + [("bg", BG_VARS, f) for f in FIPS]
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda j: fetch(*j), jobs))
    print("done", flush=True)
