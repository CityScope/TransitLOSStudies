"""Run the andorra transit-LOS study.

Usage: `uv run python andorra/run.py` from `city_science_network/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code.city_config import CITY_CONFIGS
from code.cli import parse_run_flags
from code.params import GLOBAL_PARAMS
from code.pipeline import run_city_study

CITY_DIR = Path(__file__).resolve().parent
ROOT = CITY_DIR.parent

if __name__ == "__main__":
    flags = parse_run_flags()
    if flags.dispatch(city_dir=CITY_DIR, config=CITY_CONFIGS["andorra"], params=GLOBAL_PARAMS,
                       census_root=ROOT.parent / "census", default_tile_workers=4):
        raise SystemExit(0)
    run_city_study(
        city_dir=CITY_DIR,
        config=CITY_CONFIGS["andorra"],
        params=GLOBAL_PARAMS,
        streets_root=ROOT.parent / "streets",
        worldpop_root=ROOT.parent / "worldpop",
        census_root=ROOT.parent / "census",
        reuse_cached_los=flags.reuse_cached_los,
        tile_workers=flags.tile_workers if flags.tile_workers is not None else 4,
    )
