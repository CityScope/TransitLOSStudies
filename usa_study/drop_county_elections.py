"""Remove the county-level election columns from every level finer than the county of the existing state folders (rule: votes of a finer level are never
taken from county totals; see pyCensus `elections/README.local.md`). County and state rows keep their native result.

    python drop_county_elections.py [State ...]      # default: every state folder in data/

Dropped groups: the 2000-2016 presidential and the 2024 Senate columns (MEDSL county results that were copied down). 2024 president stays: it is resampled
from voting districts (`add_precinct_elections.py`).
"""
from __future__ import annotations

import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

DATA = Path(os.environ.get("US_DATA") or Path(__file__).parent / "data")
DROP = re.compile(r"^(2000|2004|2008|2012|2016)_president_|^2024_senate_")
COARSE = {"county.geoparquet", "state.geoparquet"}


def clean(path: Path) -> int:
    if path.name in COARSE:
        return 0
    schema = pq.read_schema(path)
    bad = [n for n in schema.names if DROP.match(n)]
    if not bad:
        return 0
    t = pq.read_table(path).drop_columns(bad)
    tmp = path.with_suffix(".tmp")
    pq.write_table(t, tmp, compression="zstd")
    os.replace(tmp, path)
    return len(bad)


def files(state: str):
    d = DATA / state
    yield from d.glob("*.geoparquet")
    yield from d.glob("counties/*/*.geoparquet")


if __name__ == "__main__":
    states = sys.argv[1:] or sorted(p.name for p in DATA.iterdir() if (p / "counties").is_dir())
    for s in states:
        fs = [f for f in files(s) if "stops" not in f.name]
        with ProcessPoolExecutor(4) as ex:
            n = sum(1 for k in ex.map(clean, fs) if k)
        print(f"{s}: {n} of {len(fs)} files cleaned", flush=True)
