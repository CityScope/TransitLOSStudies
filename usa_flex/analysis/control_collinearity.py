"""Collinearity of the candidate controls (block groups, population weighted): correlations and variance inflation factors."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
b = pd.read_parquet(HERE.parent / "fit" / "data" / "blockgroups.parquet")
t = pd.read_parquet(HERE.parents[1] / "cache" / "derived" / "bg_table.parquet").set_index("GEOID").reindex(b.GEOID).reset_index()
ok = (b.population.fillna(0) > 0) & (b.acs5_workers.fillna(0) > 0)
pop = b.population.where(b.population > 0); hh = t.households.where(t.households > 0); area = t.area_km2.clip(lower=1e-3)
grp_pop = b.groupby(b.cbsa.fillna("")).population.transform("sum")
F = pd.DataFrame({
    "log_pop_density": np.log1p(pop / area), "log_job_density": np.log1p(t.jobs_lodes.fillna(0) / area), "log_jobs_5km": np.log1p(t.jobs_5km),
    "log_dist_cbd": np.log1p(t.dist_cbd_km), "log_metro_pop": np.log1p(grp_pop.where(b.cbsa.fillna("") != "", 0)),
    "log_income": np.log(t.median_hh_income_b19013.where(t.median_hh_income_b19013 > 0)),
    "poverty_share": (t.poverty_below_50 + t.poverty_50_99) / t.poverty_universe.where(t.poverty_universe > 0),
    "bachelor_share": t.acs5_educationBachelorsPlus / t.acs5_population25plus.where(t.acs5_population25plus > 0),
    "under18_share": t.under18Population / pop, "over65_share": t.over65Population / pop, "renter_share": t.acs5_householdsRenter / hh,
    "log_commute_km_res": np.log1p(t.commute_km_residents),
    # mediators (affected by transit)
    "zero_car_share": t.acs5_vehiclesHouseholds0 / hh, "cars_per_household": t.vehicles_aggregate / hh,
    "commute_time_res": t.commute_time_residents_min, "commute_speed_res": 60 * t.commute_km_residents / t.commute_time_residents_min.where(t.commute_time_residents_min > 0),
})[ok.to_numpy()]
w = pop[ok].fillna(0).to_numpy()
F = F.replace([np.inf, -np.inf], np.nan)
F = F.fillna(F.median())
Z = (F - np.average(F, axis=0, weights=w)) / np.sqrt(np.average((F - np.average(F, axis=0, weights=w)) ** 2, axis=0, weights=w))
C = np.cov(Z.T.to_numpy(), aweights=w); sd = np.sqrt(np.diag(C)); R = C / np.outer(sd, sd)
names = list(F.columns)
out = {"corr": pd.DataFrame(R, index=names, columns=names).round(3).to_dict()}
for label, cols in (("structural", names[:12]), ("structural+mediators", names)):
    idx = [names.index(c) for c in cols]
    inv = np.linalg.inv(R[np.ix_(idx, idx)])
    out[f"vif_{label}"] = {c: float(inv[k, k]) for k, c in enumerate(cols)}
(HERE.parent / "fit" / "data" / "torch_runs" / "control_collinearity.json").write_text(json.dumps(out, indent=1))
for k in ("vif_structural", "vif_structural+mediators"):
    print(k, {c: round(v, 1) for c, v in out[k].items()})
pairs = sorted(((abs(R[i, j]), names[i], names[j], R[i, j]) for i in range(len(names)) for j in range(i)), reverse=True)[:10]
for a, x, y, r in pairs:
    print(f"  r={r:+.2f}  {x} ~ {y}")
