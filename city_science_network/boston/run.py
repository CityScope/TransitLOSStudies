"""Run the Boston transit-LOS study (smoke-test entry point).

Usage: `uv run python boston/run.py` from `city_science_network/`.

Fast re-entry-point flags (see `code.cli.parse_run_flags`, folded in here
from what used to be separate `rebuild_map_only.py`/
`rebuild_development_only.py`/`export_gpkg.py` scripts):

    --map-only [--rebuild-tiles] [--pop-chunks]   just rebuild map.html
    --census-only / --refresh-census               just refresh census join + map/stats
    --dev-tiles-only                                just redraw the development overlay
    --export-gpkg                                   just export results to GeoPackage
    --no-reuse-cached-los                           force a genuine end-to-end run
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
    # Boston metro's AOI (~21,500 km^2) is much larger than the other
    # cities in this study; even the default 4-worker tile-build pool
    # (each worker can be 9-15 GB RSS) can push a 30 GB host over the
    # edge when the desktop is already using a chunk of RAM. Drop to a
    # single tile-build worker for this city to keep peak RSS bounded,
    # at the cost of a slower (serial) tile build.
    if flags.dispatch(city_dir=CITY_DIR, config=CITY_CONFIGS["boston"], params=GLOBAL_PARAMS,
                       census_root=ROOT.parent / "census", default_tile_workers=1):
        raise SystemExit(0)
    run_city_study(
        city_dir=CITY_DIR,
        config=CITY_CONFIGS["boston"],
        params=GLOBAL_PARAMS,
        streets_root=ROOT.parent / "streets",
        worldpop_root=ROOT.parent / "worldpop",
        census_root=ROOT.parent / "census",
        # Defaults to True (consistent with every other city's `run.py`):
        # resumes from the cached stops/access-edges of a previous run (see
        # `run_city_study`) instead of redoing the GTFS/isochrone/LOS stage.
        # Pass `--no-reuse-cached-los` for a genuine end-to-end run.
        reuse_cached_los=flags.reuse_cached_los,
        tile_workers=flags.tile_workers if flags.tile_workers is not None else 1,
    )
