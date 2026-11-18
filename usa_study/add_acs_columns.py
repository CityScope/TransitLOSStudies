"""Add the new ACS 5-year columns (pycensus acs5 schema) to every finished county's `tract.geoparquet` (and replace the corrected
`acs5_vehiclesHouseholds3Plus`), state by state: one cached state-level tract fetch per state, merged on GEOID.
Usage: python add_acs_columns.py [StateDirName ...]"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
sys.path.insert(0, str(BASE))


def main():
    import us
    from pycensus.census.countries.usa import acs5

    want = set(sys.argv[1:])
    fips_by_dir = {"".join(c for c in s.name if c.isalnum()): (s.fips, s.abbr) for s in us.states.STATES + [us.states.DC]}
    for sd in sorted(DATA.iterdir()):
        if not (sd / "counties").is_dir() or (want and sd.name not in want) or sd.name not in fips_by_dir:
            continue
        fips, abbr = fips_by_dir[sd.name]
        new = acs5.load(states=[abbr], level="tract", cache_dir=str(DATA / "pycensus_cache"))
        keep = [c for c in new.columns if c != "geometry"]
        new = pd.DataFrame(new[keep]).drop_duplicates("GEOID")
        n = 0
        for f in sorted((sd / "counties").glob("*/tract.geoparquet")):
            tr = gpd.read_parquet(f)
            add = [c for c in new.columns if c != "GEOID" and (c not in tr.columns or c == "acs5_vehiclesHouseholds3Plus")]
            if not add:
                continue
            tr = tr.drop(columns=[c for c in add if c in tr.columns]).merge(new[["GEOID", *add]], on="GEOID", how="left")
            gpd.GeoDataFrame(tr, geometry="geometry", crs=4326 if tr.crs is None else tr.crs).to_parquet(f)
            n += 1
        print(f"{sd.name}: {n} county tract files updated", flush=True)


if __name__ == "__main__":
    main()
