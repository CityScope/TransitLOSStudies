"""LODES home -> work flows between block groups (up to 80 km straight line), indexed by the rows of fit/data/blockgroups.parquet, for the destination-access term
of fit/torch_fit2.py.

    python fit/save_flows.py      -> fit/data/torch/flows.npz (home row, work row, workers)
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
CACHE = HERE.parents[1] / "cache"
import sys as _s; _s.path.insert(0, str(HERE)); import paths as P  # noqa: E402


def main():
    bgt = pd.read_parquet(P.DATA / "blockgroups.parquet", columns=["GEOID"])
    tab = pd.read_parquet(CACHE / "derived" / "bg_table.parquet", columns=["GEOID", "lon", "lat"])
    rows = pl.DataFrame({"g": bgt.GEOID.to_numpy(), "row": np.arange(len(bgt))})
    xy = pl.DataFrame({"g": tab.GEOID.to_numpy(), "lon": tab.lon.to_numpy(), "lat": tab.lat.to_numpy()})
    frames = []
    for f in sorted(glob.glob(str(CACHE / "lodes_od" / "*_od_*_JT00_*.csv.gz"))):
        frames.append(pl.read_csv(f, columns=["w_geocode", "h_geocode", "S000"], schema_overrides={"w_geocode": pl.Utf8, "h_geocode": pl.Utf8, "S000": pl.Int64})
                      .with_columns(pl.col("w_geocode").str.slice(0, 12).alias("w"), pl.col("h_geocode").str.slice(0, 12).alias("h"))
                      .group_by(["h", "w"]).agg(pl.col("S000").sum().alias("n")))
    fl = pl.concat(frames).group_by(["h", "w"]).agg(pl.col("n").sum())
    fl = (fl.join(xy.rename({"g": "h", "lon": "hlon", "lat": "hlat"}), on="h").join(xy.rename({"g": "w", "lon": "wlon", "lat": "wlat"}), on="w")
          .join(rows.rename({"g": "h", "row": "hr"}), on="h").join(rows.rename({"g": "w", "row": "wr"}), on="w"))
    r = 6371.0
    fl = fl.with_columns((2 * r * (((pl.col("wlat") - pl.col("hlat")).radians() / 2).sin() ** 2 + (pl.col("hlat").radians().cos() * pl.col("wlat").radians().cos()
                         * (((pl.col("wlon") - pl.col("hlon")).radians() / 2).sin() ** 2))).sqrt().arcsin()).alias("km")).filter(pl.col("km") <= 80.0)
    np.savez(P.DATA / "torch" / "flows.npz", h=fl["hr"].to_numpy().astype(np.int32), w=fl["wr"].to_numpy().astype(np.int32), n=fl["n"].to_numpy().astype(np.float32))
    print("flows", fl.height, "workers", int(fl["n"].sum()), flush=True)


if __name__ == "__main__":
    main()
