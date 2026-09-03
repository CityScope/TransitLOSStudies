"""Run the gipuzkoa transit-LOS study (smoke-test entry point).

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

# `date_range` is passed explicitly (a Wednesday, `date_type="weekday"`'s
# convention) rather than left to `download_and_prepare_stops`'s
# auto-detection fallback (`feed.get_representative_date`). One of
# gipuzkoa's 27 GTFS feeds (`gtfs/mdb1135__bizkaibus/calendar.txt`) has a
# dummy `PRUEBA` service row with `start_date=20160630` and every weekday
# flag set to 0 (i.e. it never actually runs) -- but its presence still
# drags `feed.calendar.min_date` back to 2016-06-30 across the *merged*
# 27-feed calendar, which is the window `get_representative_date` samples
# candidates from. That previously led auto-detection to pick 2016-06-30
# itself as the "representative" date, which then failed downstream in
# `pyGTFSHandler`'s `_filter_by_date` with `Exception: No services in date
# 2016-06-30` (none of the 27 feeds' *real* services cover 2016; only the
# always-inactive `PRUEBA` row's `start_date` does). 2026-03-04 was checked
# directly against all 27 feeds (`pyGTFSHandler.feed.Feed(...).calendar.
# get_services_in_date(date(2026, 3, 4))`) and returns 17,088 active
# services -- it falls inside the real, current service window shared by
# every non-dummy feed (e.g. `mdb2715_raw_data_f_euskotren`'s narrowest
# window, 2026-02-14 to 2026-03-26, as well as the DBUS/Lurraldebus/
# Irungo Bus/Alavabus/Bizkaibus/Alsa/blablacar feeds).
GIPUZKOA_DATE_RANGE = (date(2026, 3, 4), date(2026, 3, 4))

if __name__ == "__main__":
    flags = parse_run_flags()
    if flags.dispatch(city_dir=CITY_DIR, config=CITY_CONFIGS["gipuzkoa"], params=GLOBAL_PARAMS,
                       census_root=ROOT.parent / "census", default_tile_workers=4):
        raise SystemExit(0)
    run_city_study(
        city_dir=CITY_DIR,
        config=CITY_CONFIGS["gipuzkoa"],
        params=GLOBAL_PARAMS,
        streets_root=ROOT.parent / "streets",
        worldpop_root=ROOT.parent / "worldpop",
        census_root=ROOT.parent / "census",
        date_range=GIPUZKOA_DATE_RANGE,
        reuse_cached_los=flags.reuse_cached_los,
        tile_workers=flags.tile_workers if flags.tile_workers is not None else 4,
    )
