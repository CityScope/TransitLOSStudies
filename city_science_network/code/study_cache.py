"""The shared cache (`studytools.cache`, default folder `TransitLOSStudies/cache`) for the city study."""

from __future__ import annotations

from studytools import cache as _m

STUDY = "city"
DEFAULT_CACHE_DIR = _m.DEFAULT_CACHE_DIR
resolve_cache_dir = _m.resolve_cache_dir
osm_dir = _m.osm_dir
worldpop_dir = _m.worldpop_dir
spain_raw_dir = _m.spain_raw_dir


def pycensus_dir(cache):
    return _m.pycensus_dir(cache, STUDY)


def apply_env(cache):
    _m.apply_env(cache, STUDY)
