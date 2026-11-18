"""Add the housing price / rent columns (pycensus `usa.housing`: ACS 2020-24 / 2012-16, Census 2000, FHFA HPI, HUD FMR/SAFMR; ~36 columns) to every
finished county's `tract.geoparquet`, state by state (raw downloads cached in data/_cache/usa_housing); existing housing columns are replaced.
Usage: python add_housing_columns.py [StateDirName ...]   (e.g. Massachusetts)"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
sys.path.insert(0, str(BASE))


def main():
    import us
    from pycensus.census.countries.usa.housing import load_tract_housing

    want = set(sys.argv[1:])
    fips_by_dir = {"".join(c for c in s.name if c.isalnum()): (s.fips, s.abbr) for s in us.states.STATES + [us.states.DC]}
    for sd in sorted(DATA.iterdir()):
        if not (sd / "counties").is_dir() or (want and sd.name not in want) or sd.name not in fips_by_dir:
            continue
        fips, abbr = fips_by_dir[sd.name]
        new = load_tract_housing(fips, abbr, data_dir=str(DATA / "_cache" / "usa_housing"))
        n = 0
        for f in sorted((sd / "counties").glob("*/tract.geoparquet")):
            tr = gpd.read_parquet(f)
            add = [c for c in new.columns if c != "GEOID"]
            if not add:
                continue
            tr = tr.drop(columns=[c for c in add if c in tr.columns]).merge(new[["GEOID", *add]], on="GEOID", how="left")
            gpd.GeoDataFrame(tr, geometry="geometry", crs=4326 if tr.crs is None else tr.crs).to_parquet(f)
            n += 1
        print(f"{sd.name}: {n} county tract files updated", flush=True)


if __name__ == "__main__":
    main()
