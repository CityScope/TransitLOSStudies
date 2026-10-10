"""Build ONE MapLibre map covering the whole USA from the per-state `data/<State>/` outputs.

Usage (after the state runs finished):
    python build_usa_map.py                      # every finished state, all layers
    python build_usa_map.py --states AL,DE       # subset (e.g. a smoke test)
    python build_usa_map.py --no-streets         # skip the (very large) street edges

Zoomed out you see h3 res 3/4/5 (and county / states); zooming in, the finer h3
resolutions, census levels (tract / block group / block) and streets take over,
exactly as in a single-state map -- the inputs are just every state's files
concatenated (`code.build_state_map.assemble_map_inputs`). Border h3 cells that
appear in two states are merged before the coarse levels are derived.
"""
import argparse
import os
os.environ.setdefault("GEOHIERARCHY_MAX_NATIVE_TILE_ZOOM", "14")  # continent-scale: MapLibre overzooms past z14
import logging
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from code.build_state_map import assemble_map_inputs  # noqa: E402

logger = logging.getLogger("build_usa_map")


def finished_state_dirs(data_dir: Path, only=None):
    import us

    wanted = None
    if only:
        wanted = {"".join(c for c in us.states.lookup(a.strip()).name if c.isalnum()) for a in only}
    dirs = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or d.name.startswith(("_", "pycensus")):
            continue
        if wanted is not None and d.name not in wanted:
            continue
        if (d / "h3_res7.geoparquet").is_file():  # the county level is optional (e.g. Connecticut has none)
            dirs.append(d)
    return dirs


def build_usa_map(states=None, include_streets: bool = True, out_dir=None, in_process: bool = False) -> str:
    """Assemble every finished state's outputs and build the nationwide map.

    Args:
        states: Optional list of state abbreviations (default: every finished state).
        include_streets: Whether to include the street edges layer (very large).
        out_dir: Output folder (default `data/_usa`); writes `map.html` + `tiles/`.
        in_process: Tile everything inside one process (small subsets only). The default tiles the heavy levels
            (h3 res 9/11, blocks, block groups, streets) one state at a time (`code.external_tiles`) so the
            nationwide map fits in memory and streets load lazily at high zoom.

    Returns:
        Path to the written `map.html`.
    """
    from transitlos.map.build import build_city_map

    out = Path(out_dir) if out_dir else BASE_DIR / "data" / "_usa"
    out.mkdir(parents=True, exist_ok=True)
    dirs = finished_state_dirs(BASE_DIR / "data", states)
    if not dirs:
        raise SystemExit("No finished state outputs found under data/")
    logger.info(f"Assembling {len(dirs)} state(s): {[d.name for d in dirs]}")

    t = time.time()
    inputs = assemble_map_inputs(dirs, include_streets=include_streets, external=not in_process)
    logger.info(f"Inputs assembled in {time.time() - t:.0f}s: h3 res {sorted(inputs['h3_by_resolution'])}")
    extra = {}
    if not in_process:
        from code.external_tiles import make_writers

        extra["external_tile_writers"] = make_writers(dirs, include_streets=include_streets)
    html = build_city_map(tiles_dir=str(out / "tiles"), out_html=str(out / "map.html"), **inputs, **extra)
    logger.info(f"Wrote {html} in {time.time() - t:.0f}s")
    return html


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", help="comma-separated state abbreviations (default: every finished state)")
    ap.add_argument("--no-streets", action="store_true", help="omit the street edges layer")
    ap.add_argument("--in-process", action="store_true", help="tile everything in one process (small subsets only)")
    ap.add_argument("--out", default=str(BASE_DIR / "data" / "_usa"), help="output folder (map.html + tiles/)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_usa_map(args.states.split(",") if args.states else None, not args.no_streets, args.out, args.in_process)


if __name__ == "__main__":
    main()
