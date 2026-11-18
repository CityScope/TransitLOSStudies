"""Stage 5: ONE MapLibre map for all of Spain from the per-community outputs ``data/<Community>/`` (copied pattern of usa_study.build_usa_map).

    python build_spain_map.py                      # every finished community
    python build_spain_map.py --ccaa PaisVasco     # subset (smoke test; comma separated folder names)
    python build_spain_map.py --no-streets

Heavy levels (h3 res 7-11, census sections, streets) are tiled one community at a time (``code.external_tiles``) and loaded lazily by the
page; the stats panel reads per-community chunks (``code.stats_chunks``) and the scenario editor population chunks (``code.pop_chunks``).
The finished page gets the ES/EN switch (``code.i18n_inject``) and is centred on the peninsula.
"""
import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("GEOHIERARCHY_MAX_NATIVE_TILE_ZOOM", "14")
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
ROOT = BASE_DIR.parent.parent

from code.build_region_map import assemble_map_inputs  # noqa: E402
from code.i18n_inject import inject_language_switch  # noqa: E402
from code.column_docs import inject_column_docs  # noqa: E402

logger = logging.getLogger("build_spain_map")


def finished_ccaa_dirs(data_dir: Path, only=None):
    dirs = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or d.name.startswith(("_", "pycensus", "gtfs", "spain")):
            continue
        if only and d.name not in only:
            continue
        if (d / "h3_res7.geoparquet").is_file():
            dirs.append(d)
    return dirs


def _center_on_spain(html_path) -> None:
    """Canary Islands stretch the bounds south-west: open on the peninsula."""
    path = Path(html_path)
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"center: \[[^\]]*\],(\s*)zoom: [0-9.]+,", r"center: [-3.7, 40.2],\1zoom: 5.2,", text, count=1)
    from transitlos.map.buildstamp import write_stamped
    write_stamped(path, text)


def build_spain_map(ccaa=None, include_streets: bool = True, out_dir=None, in_process: bool = False) -> str:
    from transitlos.map.build import build_city_map

    out = Path(out_dir) if out_dir else BASE_DIR / "data" / "_spain"
    out.mkdir(parents=True, exist_ok=True)
    dirs = finished_ccaa_dirs(BASE_DIR / "data", ccaa)
    if not dirs:
        raise SystemExit("No finished community outputs under data/")
    logger.info(f"Assembling {[d.name for d in dirs]}")
    t = time.time()
    inputs = assemble_map_inputs(dirs, include_streets=include_streets, external=not in_process)
    extra = {}
    if not in_process:
        from code.external_tiles import make_writers
        from code.pop_chunks import build_spain_pop_chunks
        from code.stats_chunks import build_nation_quantiles, build_nation_sample, build_stats_chunks

        extra["external_tile_writers"] = make_writers(dirs, include_streets=include_streets)
        build_stats_chunks(dirs, out, BASE_DIR / "data" / "_cache" / "geo_names.json")
        build_nation_sample(inputs["h3_by_resolution"][8].drop(columns="geometry").rename(columns={"h3_cell": "h3"}), out)
        build_nation_quantiles(out)
        from code.metro_areas import build_metro_assets   # Eurostat metropolitan areas: map file, stats chunks per area, dotted boundary lines

        build_metro_assets(str(BASE_DIR / "data"), str(out))
        extra["stats_scope_cfg"] = {"dir": "stats_chunks", "min_zoom": 5, "address_zoom": 6, "dev_min_zoom": 8}
        build_spain_pop_chunks(dirs, out)
    from code.build_region_map import spain_field_policy

    with spain_field_policy():
        html = build_city_map(tiles_dir=str(out / "tiles"), out_html=str(out / "map.html"), **inputs, **extra)
    _center_on_spain(html)
    from code.build_region_map import _data_columns

    inject_language_switch(html, columns=_data_columns(inputs))
    inject_column_docs(html)
    logger.info(f"Wrote {html} in {time.time() - t:.0f}s")
    return html


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ccaa", help="comma-separated community folder names (default: all finished)")
    ap.add_argument("--no-streets", action="store_true")
    ap.add_argument("--in-process", action="store_true", help="tile everything in one process (small subsets only)")
    ap.add_argument("--out", default=str(BASE_DIR / "data" / "_spain"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_spain_map(a.ccaa.split(",") if a.ccaa else None, not a.no_streets, a.out, a.in_process)


if __name__ == "__main__":
    main()
