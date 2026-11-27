"""Blank the 2024 presidential columns of the units that still hold their county's value (no voting district reached them).

    python null_county_fallback_elections.py [State ...]      # default: every state folder in data/

`add_precinct_elections.py` resamples voting-district results; a unit no voting district covers used to keep the county's result. That is not data at the
unit's resolution, so those units are set to missing (all `2024_president_*` columns of the row). A county where every unit has the same value (one voting
district for the whole county) is left alone.
"""
from __future__ import annotations

import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA = Path(os.environ.get("US_DATA") or Path(__file__).parent / "data")
COL = re.compile(r"^2024_president_")
KEY = "2024_president_democrat_share"


def clean_county(cdir: Path) -> tuple[int, int]:
    cf = cdir / "county.geoparquet"
    if not cf.is_file():
        return 0, 0
    c = pq.read_table(cf).to_pandas()
    cols = [n for n in c.columns if COL.match(n)]
    if not cols or KEY not in c.columns:
        return 0, 0
    cv = c[cols].iloc[0]
    n_rows = n_files = 0
    for f in list(cdir.glob("*.geoparquet")):
        if f.name == "county.geoparquet" or "street" in f.name:
            continue
        t = pq.read_table(f)
        if KEY not in t.schema.names:
            continue
        df = t.to_pandas()
        v = df[KEY].to_numpy(float)
        same = np.isclose(v, float(cv[KEY]), atol=1e-9, rtol=0) & ~np.isnan(v)
        if same.all() or not same.any():
            continue
        for n in [n for n in df.columns if COL.match(n)]:
            df.loc[same, n] = np.nan
        out = pa.Table.from_pandas(df, preserve_index=False)
        geo = (t.schema.metadata or {}).get(b"geo")   # the geometry metadata of the file stays; the pandas index metadata is written fresh
        if geo:
            out = out.replace_schema_metadata({**(out.schema.metadata or {}), b"geo": geo})
        tmp = f.with_suffix(".tmp")
        pq.write_table(out, tmp, compression="zstd")
        os.replace(tmp, f)
        n_rows += int(same.sum()); n_files += 1
    return n_rows, n_files


if __name__ == "__main__":
    states = sys.argv[1:] or sorted(p.name for p in DATA.iterdir() if (p / "counties").is_dir())
    for s in states:
        dirs = sorted((DATA / s / "counties").iterdir())
        with ProcessPoolExecutor(4) as ex:
            res = list(ex.map(clean_county, dirs))
        print(f"{s}: {sum(r[0] for r in res)} rows blanked in {sum(r[1] for r in res)} files", flush=True)
