"""Run the san_francisco transit-LOS study (smoke-test entry point).

Usage: `uv run python boston/run.py` from `city_science_network/`.
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

# `date_range` is passed explicitly, the same fix pattern `gipuzkoa/run.py`
# already uses, rather than relying on `download_and_prepare_stops`'s
# auto-detection fallback (`feed.get_representative_date`). San Francisco
# stacks 34 GTFS feeds whose downloaded `calendar.txt` snapshots have gone
# stale by wildly different amounts relative to wall-clock "today"
# (2026-08-24 as of this fix): the real SFMTA Muni feed
# (`gtfs/mdb2886__san_franci`, 3,244 stops, by far the largest single feed
# here) is only valid 2025-08-30 -> 2026-02-13 -- already ~6 months stale
# -- while BART (`gtfs/mdb53__bay_area_r`) is valid 2026-01-12 ->
# 2026-08-09 and Caltrain (`gtfs/mdb54__caltrain`) 2026-01-31 -> 2026-08-31.
# `transitlos.stops.download_and_prepare_stops`'s auto-detect fallback was
# fixed (2026-08-24) to bound its search to a +/-60-day window around
# "today" instead of the previously-unbounded cross-feed calendar union
# (see that module's docstring for the original Boston bug this targets),
# but that symmetric window still can't reach back far enough to cover
# Muni's real validity window here -- so Muni was silently excluded
# entirely (SF's total stop count collapsed from ~11.5k to ~1.9k, even
# though the remaining stops themselves scored correctly). 2026-02-04 (a
# Wednesday) was checked directly against Muni/BART/Caltrain
# (`pyGTFSHandler.feed.Feed(...).filter(date=..., frequencies=True, ...)`)
# and returns real full-day service on all three simultaneously
# (Muni: 339,512 stop-time rows / 3,225 unique stops; BART: 14,050 rows /
# 101 stops; Caltrain: 2,030 rows / 56 stops).
SAN_FRANCISCO_DATE_RANGE = (date(2026, 2, 4), date(2026, 2, 4))

if __name__ == "__main__":
    flags = parse_run_flags()
    if flags.dispatch(city_dir=CITY_DIR, config=CITY_CONFIGS["san_francisco"], params=GLOBAL_PARAMS,
                       census_root=ROOT.parent / "census", default_tile_workers=4):
        raise SystemExit(0)
    run_city_study(
        city_dir=CITY_DIR,
        config=CITY_CONFIGS["san_francisco"],
        params=GLOBAL_PARAMS,
        streets_root=ROOT.parent / "streets",
        worldpop_root=ROOT.parent / "worldpop",
        census_root=ROOT.parent / "census",
        date_range=SAN_FRANCISCO_DATE_RANGE,
        reuse_cached_los=flags.reuse_cached_los,
        tile_workers=flags.tile_workers if flags.tile_workers is not None else 4,
    )
