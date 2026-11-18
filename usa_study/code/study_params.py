"""Configurable rules of the US study. `GRID`: the h3 grid / population rules (documented in `transitlos.grid_rules`); change values here."""
from transitlos.grid_rules import GridRules

GRID = GridRules(
    street_max_dist_m=125.0,
    occupancy_min=1.0,
    occupancy_min_per_km2=10.0,
    rescale_warn_share=0.05,
    within_source_polygon=False,   # True: each block over its own street cells first (see transitlos.grid_rules)
)
