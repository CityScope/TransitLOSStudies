"""Goal 2: car ownership as a mediator. With the transit curves of the exogenous-controls fit, how strongly does transit access predict the share of households
without a car (population-weighted least squares on the logit of that share, exogenous controls included)? And the control coefficients of the fits, named."""
import json
import numpy as np
import torch
import torch_fit as tf

d = tf.Data()
out = {}
for name, cs in (("g2_exogenous", "exogenous"), ("g2_exogenous+cars", "exogenous+cars")):
    X = d.controls(tf.CONTROL_SETS[cs])
    r = json.loads((tf.RUNS / f"{name}.json").read_text())
    out[name] = dict(zip(d.last_control_names, r["delta"]))
X = d.controls(tf.CONTROL_SETS["exogenous"])
m = tf.Model(X.shape[1], coupling=False); m.load_state_dict(torch.load(tf.RUNS / "g2_exogenous.pt", map_location=tf.DEV)); m.to(tf.DEV)
with torch.no_grad():
    A = m.access(d).cpu().numpy()
eps = float(torch.exp(m.leps)); g = np.log1p(A / eps)
t = d._tab
hh = t.households.where(t.households > 0).to_numpy(float)
z = (t.acs5_vehiclesHouseholds0.to_numpy(float) + 0.5) / (hh + 1.0)
ok = np.isfinite(z) & np.isfinite(hh)
yv = np.log(z / (1 - z))
Xn = X.cpu().numpy()
pop = d.pop.cpu().numpy()
Z = np.column_stack([np.ones(len(g)), g, Xn])[ok]; w = pop[ok]
coef = np.linalg.lstsq(Z * np.sqrt(w)[:, None], yv[ok] * np.sqrt(w), rcond=None)[0]
res = yv[ok] - Z @ coef
r2 = 1 - (w * res ** 2).sum() / (w * (yv[ok] - np.average(yv[ok], weights=w)) ** 2).sum()
Z0 = Z[:, [0] + list(range(2, Z.shape[1]))]
c0 = np.linalg.lstsq(Z0 * np.sqrt(w)[:, None], yv[ok] * np.sqrt(w), rcond=None)[0]
r20 = 1 - (w * (yv[ok] - Z0 @ c0) ** 2).sum() / (w * (yv[ok] - np.average(yv[ok], weights=w)) ** 2).sum()
q = np.quantile(g[ok], [0.5, 0.9])
out["car_ownership_on_access"] = {"coef_per_unit_g": float(coef[1]), "r2_with_access": float(r2), "r2_controls_only": float(r20),
                                  "g_median": float(q[0]), "g_p90": float(q[1]),
                                  "odds_ratio_p90_vs_none": float(np.exp(coef[1] * q[1])), "odds_ratio_median_vs_none": float(np.exp(coef[1] * q[0]))}
(tf.RUNS / "g2_mediation.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["car_ownership_on_access"], indent=1))
for k in ("g2_exogenous", "g2_exogenous+cars"):
    print(k, {a: round(b, 2) for a, b in out[k].items()})
