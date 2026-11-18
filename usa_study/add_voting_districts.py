"""Add `voting_district.geoparquet` (Census VTD 2020, all census columns + transit score) to existing US state folders.

    python add_voting_districts.py [State ...]      # default: every state folder in data/
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from code.state_pipeline import build_voting_districts

DATA = Path(os.environ.get("US_DATA") or Path(__file__).parent / "data")

if __name__ == "__main__":
    wanted = sys.argv[1:] or sorted(p.name for p in DATA.iterdir() if (p / "counties").is_dir())
    for name in wanted:
        d = DATA / name
        out = d / "voting_district.geoparquet"
        if out.exists() or not (d / "counties").is_dir():
            continue
        fips = sorted((d / "counties").iterdir())[0].name[:2]
        try:
            g = build_voting_districts(d, fips)
            if g is None or g.empty:
                print(f"[vtd] {name}: none", flush=True); continue
            g.to_parquet(out)
            print(f"[vtd] {name}: {len(g)} VTDs", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[vtd] {name}: FAILED {type(exc).__name__}: {str(exc)[:150]}", flush=True)
