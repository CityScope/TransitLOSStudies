"""Run the guadalajara transit-LOS study (smoke-test entry point).

Usage: `uv run python boston/run.py` from `city_science_network/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code.city_config import CITY_CONFIGS
from studytools.cli import parse_run_flags
from studytools.params import GLOBAL_PARAMS
from studytools.pipeline import run_city_study

CITY_DIR = Path(__file__).resolve().parent
ROOT = CITY_DIR.parent

from code.study_cache import apply_env as _apply_cache_env, osm_dir as _osm_dir, pycensus_dir as _pycensus_dir, resolve_cache_dir as _resolve_cache_dir, worldpop_dir as _worldpop_dir

_CACHE = _resolve_cache_dir()  # shared TransitLOSStudies/cache (or --cache-dir / STUDY_CACHE_DIR)
_apply_cache_env(_CACHE)

if __name__ == "__main__":
    flags = parse_run_flags()
    if flags.dispatch(city_dir=CITY_DIR, config=CITY_CONFIGS["guadalajara"], params=GLOBAL_PARAMS,
                       census_root=_pycensus_dir(_CACHE), default_tile_workers=4):
        raise SystemExit(0)
    run_city_study(
        city_dir=CITY_DIR,
        config=CITY_CONFIGS["guadalajara"],
        params=GLOBAL_PARAMS,
        streets_root=_osm_dir(_CACHE),
        worldpop_root=_worldpop_dir(_CACHE),
        census_root=_pycensus_dir(_CACHE),
        # Was explicitly False during an earlier debugging pass (a fix
        # needed a genuine end-to-end recompute); now that Guadalajara has
        # succeeded standalone, default to the universal reuse-cache
        # behavior (see other cities' `run.py`) for fast, idempotent reruns.
        reuse_cached_los=flags.reuse_cached_los,
        tile_workers=flags.tile_workers if flags.tile_workers is not None else 4,
    )
