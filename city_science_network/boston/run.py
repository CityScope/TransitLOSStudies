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

# Boston-only override (2026-09-04, live OOM x3: an unwrapped/wrapped run
# consistently hit ~11.5-12.7GB RSS and got killed by the kernel's GLOBAL OOM
# killer during the core/metro boundary split, right before the
# census-geometry map aggregation stage -- on a real 11,683 km^2 AOI
# (Massachusetts-wide, 2,265,309 h3 cells at the default resolution), even
# with isochrones/census-join/H3-resampling already chunked. Explicit user
# follow-up after the 3rd crash: "use res 5 chunks. and each chunk should
# be processed completely independently... process for each chunk the
# isochrones, population, census, streets, etc resampling... This is
# something to be implemented in general in the packages and in particular
# for the large cities." `isochrone_chunk_h3_resolution=5` partitions
# isochrone computation, the census join, H3 resampling, AND (as of this
# same fix) `_census_geometries_with_score`'s map-build census-geometry
# aggregation (`_census_polygon_agg_chunked` -- see its docstring for the
# exact-not-approximate correctness argument) into per-res-4-cell (~1,770
# km^2) chunks instead of one whole-AOI-at-once pass -- the last of the
# stages this flag doesn't cover. `isochrone_chunk_buffer_m=2000.0` matches
# `walk_distance_steps`'s own max (2000m) -- the buffer must be at least
# that large or a stop near a chunk boundary gets an artificially truncated
# isochrone search (see that field's docstring).
BOSTON_PARAMS = dataclasses.replace(
    GLOBAL_PARAMS,
    isochrone_chunk_h3_resolution=4,
    isochrone_chunk_buffer_m=2000.0,
)

if __name__ == "__main__":
    flags = parse_run_flags()
    # Boston metro's AOI (~21,500 km^2) is much larger than the other
    # cities in this study; even the default 4-worker tile-build pool
    # (each worker can be 9-15 GB RSS) can push a 30 GB host over the
    # edge when the desktop is already using a chunk of RAM. Drop to a
    # single tile-build worker for this city to keep peak RSS bounded,
    # at the cost of a slower (serial) tile build.
    if flags.dispatch(city_dir=CITY_DIR, config=CITY_CONFIGS["boston"], params=BOSTON_PARAMS,
                       census_root=ROOT / "census", default_tile_workers=1):
        raise SystemExit(0)
    run_city_study(
        city_dir=CITY_DIR,
        config=CITY_CONFIGS["boston"],
        params=BOSTON_PARAMS,
        streets_root=ROOT / "streets",
        worldpop_root=ROOT / "worldpop",
        census_root=ROOT / "census",
        # Defaults to True (consistent with every other city's `run.py`):
        # resumes from the cached stops/access-edges of a previous run (see
        # `run_city_study`) instead of redoing the GTFS/isochrone/LOS stage.
        # Pass `--no-reuse-cached-los` for a genuine end-to-end run.
        reuse_cached_los=flags.reuse_cached_los,
        tile_workers=flags.tile_workers if flags.tile_workers is not None else 1,
    )
