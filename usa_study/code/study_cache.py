"""Thin wrapper over the shared cache module `TransitLOSStudies/study_cache_shared.py` (default folder `TransitLOSStudies/cache`)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[2] / "study_cache_shared.py"
_spec = importlib.util.spec_from_file_location("study_cache_shared", _p)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

STUDY = "usa"
DEFAULT_CACHE_DIR = _m.DEFAULT_CACHE_DIR
resolve_cache_dir = _m.resolve_cache_dir
osm_dir = _m.osm_dir
worldpop_dir = _m.worldpop_dir
spain_raw_dir = _m.spain_raw_dir


def pycensus_dir(cache):
    return _m.pycensus_dir(cache, STUDY)


def apply_env(cache):
    _m.apply_env(cache, STUDY)
