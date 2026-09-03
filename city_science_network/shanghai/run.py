"""Run the shanghai transit-LOS study.

Usage: `uv run python shanghai/run.py` from `city_science_network/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dataclasses
import os

from code.city_config import CITY_CONFIGS
from code.cli import parse_run_flags
from code.params import GLOBAL_PARAMS
from code.pipeline import run_city_study

CITY_DIR = Path(__file__).resolve().parent
ROOT = CITY_DIR.parent

# Shanghai-only override (2026-09-01): its real metro grid at the shared
# default h3_resolution=11 is 25,732,258 cells -- materializing that many
# real hexagon polygons + attribute columns as one in-memory GeoDataFrame
# needs ~22GB+ (verified live, multiple attempts, up to a 29GB+2GB-swap
# cap). Dropping native resolution to 10 (~3.7M cells) fits comfortably
# (peak 7.51GB for the whole "resample to every h3 resolution" stage,
# verified live end-to-end -- this is the config that actually produced
# Shanghai's current, correctly-built `map.html`). This is the DEFAULT --
# do not change it casually, it's the known-working state every other
# city's rebuild also currently relies on being stable.
#
# A separate, genuinely chunked (per-H3-resolution-5-tile) architecture was
# also built today (`StudyParams.chunked_h3_output`, `geohierarchy.
# chunked_h3_grid`, `pipeline.build_h3_by_resolution_chunked`) as the real
# long-term fix that would let Shanghai run at its TRUE native resolution
# 11 without this workaround. It's proven correct via unit tests
# (`test_chunked_h3_grid.py`, `test_h3_by_resolution_chunked.py`,
# `test_finish_pipeline_chunked_wiring.py`) and a synthetic ~4.26M-cell
# watchdog-guarded memory reproduction (chunked path succeeded under a 2GB
# cap the unchunked path OOM-killed under at 3GB) -- but NOT yet against
# Shanghai's real 25.7M-cell data, and the resampling-stage fix alone isn't
# a full end-to-end memory win yet (`_finish_pipeline_stages`'s
# `_concat_gdf_parquets` reassembly step still needs one full in-memory
# frame per resolution downstream -- see its own comment). Opt into it
# deliberately, e.g. for a future test run, via:
#     CS_TRANSITLOS_SHANGHAI_CHUNKED=1 python shanghai/run.py
# Running it (~2+ hours) may surface a new issue at true native-resolution
# scale the smaller synthetic repro didn't catch -- run it deliberately
# (e.g. under the same `systemd-run --user --scope -p MemoryMax=.. -p
# MemorySwapMax=..` pattern used throughout today's Shanghai debugging),
# not as a silent default. `dataclasses.replace` on the shared, frozen
# `GLOBAL_PARAMS` keeps every other city's params byte-for-byte unaffected
# (`StudyParams` is `@dataclass(frozen=True)`, so this can't mutate the
# shared singleton even by accident).
if os.environ.get("CS_TRANSITLOS_SHANGHAI_CHUNKED") == "1":
    SHANGHAI_PARAMS = dataclasses.replace(
        GLOBAL_PARAMS,
        h3_resolution=11,
        map_h3_resolutions=(5, 7, 9, 11),
        chunked_h3_output=5,
    )
else:
    SHANGHAI_PARAMS = dataclasses.replace(
        GLOBAL_PARAMS,
        h3_resolution=10,
        map_h3_resolutions=(5, 7, 9, 10),
    )

if __name__ == "__main__":
    flags = parse_run_flags()
    if flags.dispatch(city_dir=CITY_DIR, config=CITY_CONFIGS["shanghai"], params=SHANGHAI_PARAMS,
                       census_root=ROOT.parent / "census", default_tile_workers=1):
        raise SystemExit(0)
    run_city_study(
        city_dir=CITY_DIR,
        config=CITY_CONFIGS["shanghai"],
        params=SHANGHAI_PARAMS,
        streets_root=ROOT.parent / "streets",
        worldpop_root=ROOT.parent / "worldpop",
        census_root=ROOT.parent / "census",
        reuse_cached_los=flags.reuse_cached_los,
        # Shanghai's metro grid (14-city Yangtze Delta megaregion, 13M+ res-11
        # h3 cells) OOM-killed the map-build stage twice live: once at the
        # default tile_workers=4, then again at tile_workers=2 (both under a
        # 20G systemd MemoryMax). 1 is the most conservative retry value
        # short of redesigning tile generation to stream per-resolution
        # rather than hold the whole grid in memory.
        tile_workers=flags.tile_workers if flags.tile_workers is not None else 1,
    )
