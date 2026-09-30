"""Shared CLI flag parsing for every city's `run.py`.

Every city's `run.py` used to be a thin, near-identical wrapper around
`code.pipeline.run_city_study`, with a few one-off scripts
(`rebuild_map_only.py`, `rebuild_development_only.py`, `export_gpkg.py`)
living alongside it for "just rebuild the map"/"just refresh census"-style
fast paths. Those scripts' functionality now lives in `code.pipeline` as
proper reusable functions (`rebuild_map_only`, `refresh_census_only`,
`rebuild_development_tiles_only`, `export_results_gpkg`); this module gives
every `run.py` one shared, consistent set of flags to reach them, instead of
each city re-implementing its own `sys.argv` parsing.

Usage (see any `run.py`, e.g. `boston/run.py`):

    from code.cli import parse_run_flags
    flags = parse_run_flags()
    if flags.dispatch(city_dir=CITY_DIR, config=CITY_CONFIGS["boston"], params=GLOBAL_PARAMS, ...):
        raise SystemExit(0)
    run_city_study(..., reuse_cached_los=flags.reuse_cached_los, tile_workers=flags.tile_workers or 4)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--map-only", "--tiles-only", dest="map_only", action="store_true",
        help=(
            "Skip the full pipeline (GTFS/network/isochrones/WorldPop/census) "
            "and just rebuild map.html from cached results/*.parquet. Pass "
            "--rebuild-tiles too to actually regenerate vector tiles (default: "
            "reuse whatever tiles are already on disk, just rebuild the HTML)."
        ),
    )
    parser.add_argument(
        "--rebuild-tiles", action="store_true",
        help="With --map-only: also regenerate vector tiles from cached results (slower).",
    )
    parser.add_argument(
        "--style-only", dest="style_only", action="store_true",
        help=(
            "With --map-only: skip re-resampling H3, re-aggregating census "
            "geometries, and rebuilding GTFS route lines too -- reuses a "
            "cached checkpoint from the last real --map-only run (pure I/O, "
            "seconds not minutes). For pure color/CSS/HTML/JS changes that "
            "don't touch any underlying data. Falls back to a normal "
            "--map-only run (and writes the checkpoint) if none exists yet."
        ),
    )
    parser.add_argument(
        "--pop-chunks", dest="pop_chunks", action="store_true",
        help="With --map-only: also rebuild the scenario editor's H3 population chunks.",
    )
    parser.add_argument(
        "--census-only", "--refresh-census", dest="census_only", action="store_true",
        help=(
            "Re-run only the census join + downstream stats/map rebuild, "
            "reusing cached stops/access-edges/h3-grid from a prior full run. "
            "For a city whose census module just became available (e.g. "
            "Hamburg once Germany GENESIS credentials are wired up) -- skips "
            "GTFS/network/isochrone/WorldPop recomputation entirely."
        ),
    )
    parser.add_argument(
        "--dev-tiles-only", dest="dev_tiles_only", action="store_true",
        help="Redraw just the development-overlay tiles from cached census geometry (targeted repair).",
    )
    parser.add_argument(
        "--export-gpkg", dest="export_gpkg", action="store_true",
        help="Export cached results/*.parquet layers to GeoPackage (.gpkg) for GIS-desktop viewing.",
    )
    parser.add_argument(
        "--no-reuse-cached-los", dest="no_reuse_cached_los", action="store_true",
        help="Full pipeline run: force a genuine end-to-end recompute of stops/LOS instead of reusing cached results.",
    )
    parser.add_argument("--tile-workers", type=int, default=None, help="Override this run's tile-build worker cap.")
    parser.add_argument("--use-pmtiles", action="store_true", help="With --map-only: build/serve PMTiles instead of XYZ tiles.")
    parser.add_argument("--renderer", choices=["maplibre", "folium"], default=None, help="Override this run's map renderer.")
    return parser


@dataclass
class RunFlags:
    map_only: bool = False
    rebuild_tiles: bool = False
    style_only: bool = False
    pop_chunks: bool = False
    census_only: bool = False
    dev_tiles_only: bool = False
    export_gpkg: bool = False
    no_reuse_cached_los: bool = False
    tile_workers: Optional[int] = None
    use_pmtiles: bool = False
    renderer: Optional[str] = None

    @property
    def reuse_cached_los(self) -> bool:
        return not self.no_reuse_cached_los

    def dispatch(
        self,
        city_dir: Path,
        config,
        params,
        census_root: Optional[Path] = None,
        default_tile_workers: int = 4,
        enable_place_comparison: Optional[bool] = None,
        default_use_pmtiles: bool = True,
    ) -> bool:
        """Handle any fast-path flag that doesn't need the full pipeline.

        Returns True if one of the fast-path flags was handled (caller
        should exit without calling `run_city_study`), False if the caller
        should proceed with the normal full pipeline.
        """
        import os

        if self.renderer:
            os.environ["TRANSITLOS_RENDERER"] = self.renderer
        tile_workers = self.tile_workers if self.tile_workers is not None else default_tile_workers

        handled = False
        if self.export_gpkg:
            from code.pipeline import export_results_gpkg

            export_results_gpkg(city_dir)
            handled = True
        if self.dev_tiles_only:
            from code.pipeline import rebuild_development_tiles_only

            rebuild_development_tiles_only(city_dir, config, params, census_root=census_root)
            handled = True
        if self.census_only:
            from code.pipeline import refresh_census_only

            refresh_census_only(city_dir, config, params, census_root=census_root, tile_workers=tile_workers)
            handled = True
        if self.map_only:
            from code.pipeline import rebuild_map_only

            rebuild_map_only(
                city_dir, config, params, census_root=census_root,
                rebuild_tiles=self.rebuild_tiles, tile_workers=tile_workers,
                build_pop_chunks=self.pop_chunks,
                use_pmtiles=(True if self.use_pmtiles else default_use_pmtiles),
                enable_place_comparison=enable_place_comparison,
                style_only=self.style_only,
            )
            handled = True
        return handled


def parse_run_flags(argv: Optional[list[str]] = None) -> RunFlags:
    """Parse this process's CLI flags (default: `sys.argv[1:]`) into a `RunFlags`.

    Uses `parse_known_args` (ignores unrecognized flags) rather than
    `parse_args`, since a couple of per-city `run.py` files still check
    `sys.argv` directly for their own one-off flags (e.g. `boston_city/run.py`'s
    `--reuse-cached-los`, which has opposite-default semantics from this
    module's own `--no-reuse-cached-los` and so is deliberately not folded
    in here) -- those must keep working unaffected by this shared parser.
    """
    args, _unknown = _build_parser().parse_known_args(argv if argv is not None else sys.argv[1:])
    return RunFlags(
        map_only=args.map_only,
        rebuild_tiles=args.rebuild_tiles,
        style_only=args.style_only,
        pop_chunks=args.pop_chunks,
        census_only=args.census_only,
        dev_tiles_only=args.dev_tiles_only,
        export_gpkg=args.export_gpkg,
        no_reuse_cached_los=args.no_reuse_cached_los,
        tile_workers=args.tile_workers,
        use_pmtiles=args.use_pmtiles,
        renderer=args.renderer,
    )
