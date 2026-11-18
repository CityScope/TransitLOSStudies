"""Dataset for fitting the scoring to the ACS transit share, per block group.

    python fit/build_dataset.py            # every state in web/

Writes fit/data/blockgroups.parquet (GEOID, state, cbsa, population, workers, transit_commuters, car/walk/bike commuters) and, per state,
fit/data/cells_<State>.npz: for each web cell (same order as web/<State>/cell_*.bin) the block group index (into blockgroups.parquet, -1 = none) and the cell's
population. A block group's predictor is  sum(cell population * cell score) / block-group population  (people no stop reaches count as score 0).
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import run_county as rc  # noqa: E402
import paths as P  # noqa: E402
from transitlos.flex import catchment as ct  # noqa: E402

BG_COLS = ["population", "acs5_workers", "publicTransportCommuters", "carCommuters", "walkCommuters", "bikeCommuters"]


def cbsa_by_county() -> dict:
    d = pd.read_excel(rc.STUDY.parent / "cache" / "census_cbsa" / "list1.xlsx", header=2, dtype=str).dropna(subset=["CBSA Code", "FIPS State Code", "FIPS County Code"])
    d = d[d["Metropolitan/Micropolitan Statistical Area"].str.startswith("Metro")]
    return dict(zip(d["FIPS State Code"] + d["FIPS County Code"], d["CBSA Code"]))


def build_state(abbr: str, bg_offset: int, cbsa: dict, out: Path):
    name = rc.STATE_DIRS[abbr]
    web = P.WEB / name
    meta = pd.read_json(web / "meta.json", typ="series")
    nc = int(meta["nc"])
    geom = np.fromfile(web / "cell_geom.bin", np.float32).reshape(nc, 14)
    counts = np.fromfile(web / "cell_counts.bin", np.float32).reshape(nc, len(meta["count_columns"]))
    pop = counts[:, list(meta["count_columns"]).index("population")]
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(geom[:, 0], geom[:, 1]), crs=4326)
    parts = []
    for cdir in sorted((rc.STUDY / "data" / name / "counties").iterdir()):
        p = cdir / "blockgroup.geoparquet"
        if p.is_file():
            b = gpd.read_parquet(p)
            b = b[["GEOID", "geometry"] + [c for c in BG_COLS if c in b.columns]].to_crs(4326)
            b["cbsa"] = cbsa.get(cdir.name, "")
            parts.append(b)
    bg = pd.concat(parts, ignore_index=True).drop_duplicates("GEOID").reset_index(drop=True)
    j = gpd.sjoin(pts, bg[["geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    idx = j["index_right"].fillna(-1).to_numpy().astype(np.int64)
    np.savez_compressed(out / f"cells_{name}.npz", bg=np.where(idx >= 0, idx + bg_offset, -1).astype(np.int32), pop=pop)
    bg = bg.drop(columns="geometry")
    bg["state"] = abbr
    return bg


def main():
    out = P.DATA
    out.mkdir(exist_ok=True)
    cbsa = cbsa_by_county()
    abbrs = [a for a in rc.STATE_DIRS if (P.WEB / rc.STATE_DIRS[a] / "meta.json").is_file()]
    tables, off = [], 0
    for a in abbrs:
        b = build_state(a, off, cbsa, out)
        off += len(b)
        tables.append(b)
        print(a, len(b), flush=True)
    pd.concat(tables, ignore_index=True).to_parquet(out / "blockgroups.parquet")


if __name__ == "__main__":
    main()
