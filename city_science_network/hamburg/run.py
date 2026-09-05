"""Run the hamburg transit-LOS study (smoke-test entry point).

Usage: `uv run python boston/run.py` from `city_science_network/`.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code.city_config import CITY_CONFIGS
from code.cli import parse_run_flags
from code.params import GLOBAL_PARAMS
from code.pipeline import run_city_study

CITY_DIR = Path(__file__).resolve().parent
ROOT = CITY_DIR.parent

# Hamburg-only override (2026-09-04): its real AOI is ~28,571 km^2 -- the
# largest of any city in this study, and this run has already OOM-killed
# the map-build stage twice (see the `tile_workers=2` comment below, from
# earlier the same day). Same treatment as Boston/Toronto: chunk
# isochrones/census join/H3 resampling AND the map-build census-geometry
# aggregation by res-4 (~1,770 km^2) cell instead of processing the whole
# metro in one pass at any of those stages -- see boston/run.py's comment
# for the full rationale.
HAMBURG_PARAMS = dataclasses.replace(
    GLOBAL_PARAMS,
    isochrone_chunk_h3_resolution=4,
    isochrone_chunk_buffer_m=2000.0,
)

if __name__ == "__main__":
    flags = parse_run_flags()
    if flags.dispatch(city_dir=CITY_DIR, config=CITY_CONFIGS["hamburg"], params=HAMBURG_PARAMS,
                       census_root=ROOT / "census", default_tile_workers=2):
        raise SystemExit(0)
    run_city_study(
        city_dir=CITY_DIR,
        config=CITY_CONFIGS["hamburg"],
        params=HAMBURG_PARAMS,
        streets_root=ROOT / "streets",
        worldpop_root=ROOT / "worldpop",
        census_root=ROOT / "census",
        reuse_cached_los=flags.reuse_cached_los,
        # The map-build/tile-generation stage has OOM-killed this run twice
        # today at the default `tile_workers=4` -- each forked worker
        # inherits a copy-on-write copy of the whole parent heap (Hamburg's
        # metro grid is large), so peak RSS scales with worker count even
        # under a per-run memory cgroup cap. 2 is a conservative retry value.
        tile_workers=flags.tile_workers if flags.tile_workers is not None else 2,
    )
