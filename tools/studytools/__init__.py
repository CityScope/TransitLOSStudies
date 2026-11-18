"""Code shared by the studies of TransitLOSStudies (a study never imports another study's code: what two studies use lives here).

The per-area pipeline (`pipeline.run_city_study`: streets, stops, level of service, census, H3, map) with its parameters (`params`), area
configuration (`config.CityConfig`), stop scoring (`stop_scoring`, `best_subset_scoring`, `stop_merge`, `route_labels`) and the shared cache
(`cache`). A study adds `TransitLOSStudies/tools` to `sys.path` (its `code/__init__.py` does it) and imports `studytools.<module>`.
"""
