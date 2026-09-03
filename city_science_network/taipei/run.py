"""Run the taipei transit-LOS study.

Usage: `uv run python taipei/run.py` from `city_science_network/`.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code.city_config import CITY_CONFIGS
from code.cli import parse_run_flags
from code.params import GLOBAL_PARAMS
from code.pipeline import run_city_study

CITY_DIR = Path(__file__).resolve().parent
ROOT = CITY_DIR.parent

# `CityConfig.stops_per_feed=True` (see taipei's entry in city_config.py)
# scores `tdx_bus`/`trtc_metro` independently rather than stacking them --
# needed because their real calendar validity windows barely overlap
# (`tdx_bus`: 2026-05-01 - 2026-06-30; `trtc_metro`: 2025-08-18 -
# 2026-12-31). Per-feed auto-detect alone still isn't enough for
# `tdx_bus` specifically, though: its real data has a genuine ~80-90
# service_ids/day sparse BASELINE across its whole range, PLUS a real
# ~6-week span of much denser data (~18k-23k active service_ids/day,
# 2026-05-21 - 2026-06-20) -- but this module's median-per-parent-station
# scoring can't reliably tell them apart (most low-traffic stations have
# near-identical service both in and out of the dense window, so the
# per-day MEDIAN barely moves even though `get_headway_at_stops`'s real
# output collapses from ~50,800 scored stops on a dense-window date down
# to a single stop on a sparse one). 2026-06-10 (a Wednesday, squarely
# inside the dense window) was directly verified: 50,823 real stops with
# non-null headway. `trtc_metro`'s service is stable day-to-day (241 real
# stops verified on both 2026-08-19 and 2026-06-10) -- no pin needed, left
# on auto-detect.
TAIPEI_DATE_RANGE = {"tdx_bus": (date(2026, 6, 10), date(2026, 6, 10))}

if __name__ == "__main__":
    flags = parse_run_flags()
    if flags.dispatch(city_dir=CITY_DIR, config=CITY_CONFIGS["taipei"], params=GLOBAL_PARAMS,
                       census_root=ROOT.parent / "census", default_tile_workers=4):
        raise SystemExit(0)
    run_city_study(
        city_dir=CITY_DIR,
        config=CITY_CONFIGS["taipei"],
        params=GLOBAL_PARAMS,
        streets_root=ROOT.parent / "streets",
        worldpop_root=ROOT.parent / "worldpop",
        census_root=ROOT.parent / "census",
        date_range=TAIPEI_DATE_RANGE,
        reuse_cached_los=flags.reuse_cached_los,
        tile_workers=flags.tile_workers if flags.tile_workers is not None else 4,
    )
