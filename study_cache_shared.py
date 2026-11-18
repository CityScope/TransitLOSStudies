"""One shared cache folder for all three studies: ``TransitLOSStudies/cache``.

Layout (one sub-folder per namespace)::

    cache/pycensus/      pycensus parquet cache + raw downloads (cache_dir; data_dir = pycensus/<country>)
    cache/worldpop/      WorldPop rasters (+ clipped/masked derivatives)
    cache/osm/           Geofabrik .osm.pbf extracts
    cache/gtfs/          downloaded GTFS feeds / feed indexes
    cache/geocoding/     reverse-geocoding answers
    cache/transitlos/, cache/pygtfshandler/   library-private caches

Override: ``--cache-dir`` or env ``STUDY_CACHE_DIR`` (aliases below are then skipped).

TEMPORARY aliases (only for the default folder, created if missing, nothing moved) so running jobs and old
runs keep working; remove after migration (see PTLOS/CACHE_AND_PACKAGING_FOLLOWUPS.md):
    cache/pycensus            -> city_science_network/census
    cache/pycensus_usa        -> usa_study/data/pycensus_cache
    cache/pycensus_spain      -> spain_study/data/pycensus_cache
    cache/pycensus_spain_raw  -> spain_study/data/spain
    cache/worldpop            -> city_science_network/worldpop
    cache/osm/<file>.osm.pbf  -> per-file links to spain_study/osm_files, usa_study/osm_files, city_science_network/streets
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = ROOT / "cache"
ENV_VAR = "STUDY_CACHE_DIR"

_DIR_ALIASES = {
    "pycensus": ROOT / "city_science_network" / "census",
    "pycensus_usa": ROOT / "usa_study" / "data" / "pycensus_cache",
    "pycensus_spain": ROOT / "spain_study" / "data" / "pycensus_cache",
    "pycensus_spain_raw": ROOT / "spain_study" / "data" / "spain",
    "worldpop": ROOT / "city_science_network" / "worldpop",
}
_OSM_SOURCES = (ROOT / "city_science_network" / "streets", ROOT / "usa_study" / "osm_files", ROOT / "spain_study" / "osm_files")


def _link(link: Path, target: Path) -> None:
    if target.is_dir() and not (link.exists() or link.is_symlink()):
        try:
            link.symlink_to(target, target_is_directory=target.is_dir())
        except (FileExistsError, OSError):
            pass


def resolve_cache_dir(explicit: Optional[str] = None) -> Path:
    if not explicit:  # --cache-dir on the command line (resolved at import, before the study's argparse runs)
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--cache-dir", default=None)
        explicit = pre.parse_known_args(sys.argv[1:])[0].cache_dir
    chosen = explicit or os.environ.get(ENV_VAR) or os.environ.get("CITY_STUDY_CACHE_DIR")
    path = Path(chosen).expanduser().resolve() if chosen else DEFAULT_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    if path == DEFAULT_CACHE_DIR:
        for name, target in _DIR_ALIASES.items():
            _link(path / name, target)
        osm = path / "osm"
        if not osm.is_symlink():
            osm.mkdir(exist_ok=True)
            for src in _OSM_SOURCES:
                if src.is_dir():
                    for f in src.glob("*.osm.pbf"):
                        _link_file(osm / f.name, f)
    return path


def _link_file(link: Path, target: Path) -> None:
    if not (link.exists() or link.is_symlink()):
        try:
            link.symlink_to(target)
        except OSError:
            pass


def pycensus_dir(cache: Path, study: str) -> Path:
    """pycensus ``cache_dir`` for ``study`` ('city' | 'usa' | 'spain'); study-specific alias only while it still exists."""
    alias = cache / f"pycensus_{study}"
    return alias if alias.exists() else cache / "pycensus"


def spain_raw_dir(cache: Path) -> Path:
    alias = cache / "pycensus_spain_raw"
    return alias if alias.exists() else cache / "pycensus" / "spain"


def worldpop_dir(cache: Path) -> Path:
    return cache / "worldpop"


def osm_dir(cache: Path) -> Path:
    return cache / "osm"


def apply_env(cache: Path, study: str) -> None:
    """Export the libraries' env-var defaults (never overrides values already set)."""
    os.environ[ENV_VAR] = str(cache)
    os.environ.setdefault("PYCENSUS_CACHE_DIR", str(pycensus_dir(cache, study)))
    os.environ.setdefault("PYCENSUS_WORLDPOP_DIR", str(worldpop_dir(cache)))
    os.environ.setdefault("TRANSITLOS_CACHE_DIR", str(cache / "transitlos"))
    os.environ.setdefault("PYGTFSHANDLER_CACHE_DIR", str(cache / "pygtfshandler"))
    os.environ.setdefault("URBANACCESSANALYZER_CACHE_DIR", str(cache))
