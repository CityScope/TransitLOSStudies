"""Configurable rules of the Spain study. `GRID`: the h3 grid / population rules (documented in `transitlos.grid_rules`); change values here."""
from transitlos.grid_rules import GridRules

GRID = GridRules(
    street_max_dist_m=125.0,
    occupancy_min=1.0,
    occupancy_min_per_km2=10.0,
    rescale_warn_share=0.05,
    within_source_polygon=False,
)
#: provinces outside the common-regime cadaster (foral): Gipuzkoa has its own (b5m), Araba, Navarra and Bizkaia use GlobalBuildingAtlas
FORAL_SOURCES = {"20": ["es_gipuzkoa"], "01": ["es_araba"], "31": ["es_navarra"], "48": ["es_bizkaia"]}
