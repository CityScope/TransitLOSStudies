"""Regional variables for bg_table.parquet: jobs within 5 km (straight line, block-group centroids) and the distance to the metro's job centre
(the block group with the most jobs within 2 km in its CBSA; NaN outside metro areas)."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "cache" / "derived" / "bg_table.parquet"
t = pd.read_parquet(T)
cb = pd.read_parquet(Path(__file__).resolve().parents[1] / "fit" / "data" / "blockgroups.parquet", columns=["GEOID", "cbsa"]).set_index("GEOID").cbsa
t["cbsa"] = t.GEOID.map(cb).fillna("")
lat0 = np.radians(t.lat.to_numpy()); xyz = np.column_stack([np.cos(lat0) * np.cos(np.radians(t.lon)), np.cos(lat0) * np.sin(np.radians(t.lon)), np.sin(lat0)]) * 6371.0
ok = np.isfinite(xyz).all(1)
tree = cKDTree(xyz[ok]); jobs = t.jobs_lodes.fillna(0).to_numpy()[ok]
for r, col in ((5.0, "jobs_5km"), (2.0, "jobs_2km")):
    nb = tree.query_ball_point(xyz[ok], r, workers=16)
    v = np.full(len(t), np.nan); v[ok] = np.array([jobs[i].sum() for i in nb]); t[col] = v
cbd = t[t.cbsa != ""].sort_values("jobs_2km", ascending=False).drop_duplicates("cbsa").set_index("cbsa")[["lat", "lon"]]
la, lo = t.cbsa.map(cbd.lat), t.cbsa.map(cbd.lon)
dlat = np.radians(la - t.lat); dlon = np.radians(lo - t.lon)
a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(t.lat)) * np.cos(np.radians(la)) * np.sin(dlon / 2) ** 2
t["dist_cbd_km"] = 2 * 6371.0 * np.arcsin(np.sqrt(a))
t.drop(columns="cbsa").to_parquet(T)
print(t[["jobs_5km", "jobs_2km", "dist_cbd_km"]].describe().round(1))
