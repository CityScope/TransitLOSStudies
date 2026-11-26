"""Res-11 populated cells of a city whose population is only at res 10 (its map's finest level), for scoring it without outcomes.

    python flex/cells_from_res10.py <city>      # -> <city>/flex/fit/cells_res11.parquet  (h3, pop)

Each res-10 cell's people are spread evenly over its res-11 children that have service data (`<city>/flex/cell_stops.parquet`): an approximation.
(Not part of fit_cities.py because it needs the h3 package, which the fitting environment does not have.)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
POP = ("census_residents", "population")


def main(city: str) -> None:
    import h3
    import pyarrow.parquet as pq

    f = HERE / city / "results" / "downloads" / "h3_res10.parquet"
    pop = next(c for c in POP if c in set(pq.read_schema(f).names))
    t = pd.read_parquet(f, columns=["h3_cell", pop]).drop_duplicates("h3_cell")
    have = set(pd.read_parquet(HERE / city / "flex" / "cell_stops.parquet", columns=["h3"])["h3"].astype(str))
    h, v, unserved = [], [], 0.0
    for c, p in zip(t["h3_cell"].astype(str), pd.to_numeric(t[pop], errors="coerce").fillna(0.0)):
        if p <= 0:
            continue
        kids = [k for k in h3.cell_to_children(c, 11) if k in have]
        if not kids:   # no station reaches any of its cells: its people count, with no service
            kids = list(h3.cell_to_children(c, 11))
            unserved += p
        h += kids
        v += [p / len(kids)] * len(kids)
    out = HERE / city / "flex" / "fit"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"h3": h, "pop": np.asarray(v, float)}).to_parquet(out / "cells_res11.parquet")
    print(f"{city}: {len(h):,} cells, {sum(v):,.0f} people ({unserved:,.0f} of them in res-10 cells no station reaches: counted with no service)")


if __name__ == "__main__":
    main(sys.argv[1])
