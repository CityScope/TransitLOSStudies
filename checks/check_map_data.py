"""Data checks before deploying a map (the normal-map rules): counts are exact sums across H3 resolutions, LOS in [0, 100] with no gaps,
WorldPop / built-area estimate equal the census total (cities).

    python3 check_map_data.py city CITY_DIR | spain DATA_DIR | us DATA_DIR
Prints DATA_OK or DATA_FAIL with the reasons.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

mode, root = sys.argv[1], sys.argv[2]
bad, notes = [], []


def sums(f, cols):
    have = [c for c in cols if c in pq.read_schema(f).names]
    d = pd.read_parquet(f, columns=have + [c for c in ("level_of_service",) if c in pq.read_schema(f).names])
    return d, {c: float(pd.to_numeric(d[c], errors="coerce").sum()) for c in have}


def check_levels(files, cols, label, tol=0.002):
    tot = {}
    for f in sorted(files):
        res = int(re.search(r"res(\d+)", os.path.basename(f)).group(1))
        d, s = sums(f, cols)
        tot[res] = s
        if "level_of_service" in d.columns:
            los = pd.to_numeric(d["level_of_service"], errors="coerce")
            if los.isna().any():
                bad.append(f"{label} res {res}: {int(los.isna().sum())} cells without LOS")
            if (los < -1e-9).any() or (los > 100 + 1e-6).any():
                bad.append(f"{label} res {res}: LOS outside [0, 100]")
    if not tot:
        bad.append(f"{label}: no h3 files")
        return tot
    fine = tot[max(tot)]
    for res, s in tot.items():
        for c, v in s.items():
            ref = fine.get(c)
            if ref and abs(v - ref) > tol * abs(ref) + 1:
                bad.append(f"{label} {c}: res {res} {v:,.0f} vs res {max(tot)} {ref:,.0f}")
    return tot


if mode == "city":
    files = [f for f in glob.glob(f"{root}/results/downloads/h3_res*.parquet")]
    cols = ["census_residents", "worldpop_residents", "gba_builtSurfacePopulation", "population", "jobs"]
    tot = check_levels(files, cols, os.path.basename(root))
    if tot:
        fine = tot[max(tot)]
        c = fine.get("census_residents")
        for k in ("worldpop_residents", "gba_builtSurfacePopulation"):
            if c and k in fine and abs(fine[k] - c) > 0.002 * c:
                bad.append(f"{k} {fine[k]:,.0f} != census {c:,.0f}")
        if "gba_builtSurfacePopulation" not in fine:
            bad.append("no built-area estimate column")
        notes.append(" ".join(f"{k} {v:,.0f}" for k, v in fine.items()))
    sd = f"{root}/results/stats_data.json"
    if os.path.isfile(sd):   # what the map itself carries: with a census, both estimates and their over / under columns
        st = open(sd, errors="ignore").read()
        if '"census_residents"' in st:
            for k in ("worldpop_residents", "gba_builtSurfacePopulation", "population_overestimation_vs_census", "population_underestimation_vs_census",
                      "worldpop_overestimation_share", "worldpop_underestimation_share", "gba_population_overestimation_vs_census",
                      "gba_population_underestimation_vs_census", "gba_overestimation_share", "gba_underestimation_share"):
                if f'"{k}"' not in st:
                    bad.append(f"map lacks {k}")
    if not os.path.isfile(f"{root}/results/rescale_factors.json"):
        bad.append("no rescale_factors.json")
    if "gba_builtSurfacePopulation" not in open(f"{root}/map.html", errors="ignore").read():
        notes.append("map.html does not list the built-area estimate (it may load column lists from data files)")
else:
    cols = ["population", "jobs", "lodes_wac_jobsTotal"]
    regions = [d for d in sorted(glob.glob(f"{root}/*/")) if not os.path.basename(d.rstrip("/")).startswith("_") and glob.glob(f"{d}h3_res*.geoparquet")]
    if not regions:
        bad.append("no region folders with h3 files")
    grand = 0.0
    for d in regions:
        tot = check_levels(glob.glob(f"{d}h3_res*.geoparquet"), cols, os.path.basename(d.rstrip("/")))
        if tot:
            grand += tot[max(tot)].get("population", 0.0)
    notes.append(f"{len(regions)} regions, population at the finest level {grand:,.0f}")
for n in notes:
    print("  ", n)
for b in bad[:30]:
    print("  ", b)
print("DATA_OK" if not bad else f"DATA_FAIL ({len(bad)} problems)")
sys.exit(0 if not bad else 1)
