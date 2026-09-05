"""Run the toronto transit-LOS study (smoke-test entry point).

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

# Toronto-only override (2026-09-04, chunk resolution moved 5->4 on
# 2026-09-05): its real AOI is ~8,552 km^2, large enough to be at real risk
# of the same whole-AOI-at-once memory blowup that OOM-killed Boston
# (~11,683 km^2) the same day -- see boston/run.py's comment for the full
# rationale. Same treatment: chunk isochrones/census join/H3 resampling AND
# the map-build census-geometry aggregation by res-4 (~1,770 km^2) cell
# instead of processing the whole metro in one pass at any of those stages.
TORONTO_PARAMS = dataclasses.replace(
    GLOBAL_PARAMS,
    isochrone_chunk_h3_resolution=4,
    isochrone_chunk_buffer_m=2000.0,
)

if __name__ == "__main__":
    flags = parse_run_flags()
    if flags.dispatch(city_dir=CITY_DIR, config=CITY_CONFIGS["toronto"], params=TORONTO_PARAMS,
                       census_root=ROOT / "census", default_tile_workers=4):
        raise SystemExit(0)
    run_city_study(
        city_dir=CITY_DIR,
        config=CITY_CONFIGS["toronto"],
        params=TORONTO_PARAMS,
        streets_root=ROOT / "streets",
        worldpop_root=ROOT / "worldpop",
        census_root=ROOT / "census",
        reuse_cached_los=flags.reuse_cached_los,
        tile_workers=flags.tile_workers if flags.tile_workers is not None else 4,
    )
