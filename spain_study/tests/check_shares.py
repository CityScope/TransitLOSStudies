"""Check the map inputs of every finished community: no *share* / turnout column outside [0, 1], no negative price/income/rent.

    python tests/check_shares.py [data/<Community> ...]
"""
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE)] + [str(BASE.parents[1] / s) for s in ("pyCensus/src", "transitLOS/src", "geohierarchy")]
from code.census_spain import POSITIVE_RE, SHARE_RE, map_columns  # noqa: E402

dirs = [Path(a) for a in sys.argv[1:]] or [d for d in sorted((BASE / "data").iterdir()) if (d / "h3_res8.geoparquet").is_file()]
bad = []
for d in dirs:
    for f in sorted(d.glob("h3_res*.geoparquet")):
        df = map_columns(pd.read_parquet(f))
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            if SHARE_RE.search(c) and (df[c].max() > 1 or df[c].min() < 0):
                bad.append((d.name, f.stem, c, "share outside [0,1]"))
            if POSITIVE_RE.search(c) and df[c].min() < 0:
                bad.append((d.name, f.stem, c, "negative"))
print(("FAIL " + str(bad[:10])) if bad else f"PASS {len(dirs)} communities, no share outside [0,1], no negative prices")
sys.exit(1 if bad else 0)
