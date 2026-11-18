"""Where demand is lost: the walk distance (per mode), headway and speed at which 25 % and 50 % of transit commuting is lost, from fitted torch_fit3 models.

    TORCH_FIT_DEVICE=cpu $TORCH_PYTHON fit/loss_points.py <run name> [...]

Two readings per curve:
  value   where the curve itself has fallen to 75 % / 50 % of its best (0 m walk; the shortest headway; the fastest speed knot)
  demand  where the predicted commuters have fallen by 25 % / 50 % if every stop's value fell by the curve's drop -- the fitted logit at today's access
          of each block group, summed over workers. Walk: that mode's own commuters (bus / rail / tram task), home side only. Headway, speed: all transit
          commuters (total task), at home and at the workplace (a service change applies at both ends).
Demand is less sensitive than value where access is weak next to eps (log(1 + A / eps)), so the demand points lie farther out than the value points.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch_fit3 as f3  # noqa: E402
from torch_fit3 import D_KNOTS, H_KNOTS, V_KNOTS, TASKS, MODES  # noqa: E402

FACT = np.concatenate([[0.0], np.geomspace(1e-4, 1.0, 400)])


def load(name, start=None):
    """The best start's model (the saved .pt) or, for another start, only its curves (json)."""
    sd = torch.load(f3.RUNS / f"{name}.pt", map_location="cpu")
    if "um" in sd:   # runs from before the free-sign mode values: log bus, log tram = -(s1 + s2), -s1
        s1, s2 = torch.nn.functional.softplus(sd.pop("um"))
        sd["lmode"] = torch.stack([-(s1 + s2), -s1])
    o = json.loads((f3.RUNS / f"{name}.json").read_text())
    dt = o.get("dest_tasks", ["total"] if "desttotal" in name else TASKS)
    m = f3.Model3(sd["delta"].shape[1], speed=o["speed"], mridge=o["mridge"], flat_below=o["flat_below"], dest_tasks=dt, dest=bool(dt),
                  weights_from=o.get("weights_from", "blocks"), far=o.get("far", "drop"), walk=o.get("walk", "free"), fix_mode=o.get("fix_mode", False),
                  fixed=o.get("fixed"))
    m.load_state_dict(sd, strict=False)
    return m, o


def parts(m, d, X=None):
    """Per block group and task: A, D (destination), eps, eps_d, theta, alpha (plus the controls' part, if any)."""
    with torch.no_grad():
        cm, cmf, lm = m.access(d)
        cols = lambda c: torch.stack([(c * torch.exp(lm)[None, :]).max(dim=1).values, c[:, 0], c[:, 2], c[:, 1]], 1)  # noqa: E731
        C = cols(cm); A = m._area(C, d)
        Af = m._area(cols(cmf), d) if cmf is not None else torch.zeros_like(A)
        Aw = torch.zeros((d.nbg_all, 4)).index_add(0, d.cell_bg_all, C * d.cell_jobs[:, None]) / d.bg_jobs.clamp(min=1.0)[:, None]
        D = torch.zeros((d.nb, 4)).index_add(0, d.fl_o, d.fl_wt[:, None] * Aw[d.fl_w])
        th = (torch.exp(m.ltheta) * m.dest_mask)[None, :] * d.has_flows[:, None]
        base = m.alpha[None, :] + (X @ m.delta.T if X is not None and X.shape[1] else 0.0)
        thf = torch.exp(m.lthf) if cmf is not None else torch.zeros(4)
        return A.double().numpy(), D.double().numpy(), torch.exp(m.leps).double().numpy(), torch.exp(m.leps_d).double().numpy(), th.double().numpy(), \
            (base * torch.ones((d.nb, 4))).double().numpy(), Af.double().numpy(), torch.exp(m.lepsf).double().numpy(), thf.double().numpy()


def demand_curve(P, d, k, dest_too, which="walk"):
    """Commuters (task k) when every value is multiplied by each factor in FACT, relative to factor 1. which: "walk" scales the walking access
    (home end), "far" the far access, "service" both (and, with dest_too, the work end)."""
    A, D, eps, epsd, th, alpha, Af, epsf, thf = P
    n = ((d.n if TASKS[k] == "total" else d.n_mode * d.ok_mode[TASKS[k]]) * getattr(d, "keep", 1.0)).double().numpy()
    a, dd = A[:, k], D[:, k]
    out = []
    for fct in FACT:
        fw = fct if which in ("walk", "service") else 1.0; ff = fct if which in ("far", "service") else 1.0
        e = alpha[:, k] + np.log1p(fw * a / eps[k]) + th[:, k] * np.log1p((fct if dest_too else 1.0) * dd / epsd[k]) + thf[k] * np.log1p(ff * Af[:, k] / epsf[k])
        out.append((n / (1 + np.exp(-e))).sum())
    out = np.array(out)
    return out / out[-1]


def crossing(x, y, level):
    """First x where y falls to `level` (y decreasing along x); NaN if never."""
    i = np.flatnonzero(y <= level)
    if not len(i):
        return float("nan")
    j = i[0]
    if j == 0:
        return float(x[0])
    return float(np.interp(level, [y[j], y[j - 1]], [x[j], x[j - 1]]))


def points(curves, dem=None):
    """{curve: {value25, value50, demand25, demand50}}; dem: {walk mode / 'service': demand relative to factor}."""
    out = {}
    xs = np.arange(0, f3.WALK_MAX_M + 1, 5.0)   # walking: 0-2 km only (beyond, the separate far-access curve)
    for mo in MODES:
        W = curves["walk"][mo]
        w = np.exp(np.interp(xs, W["x"], np.log(np.maximum(W["y"], 1e-12))))
        out[f"walk_{mo}"] = {"value25": crossing(xs, w, 0.75), "value50": crossing(xs, w, 0.5)}
        if dem:
            g = np.interp(w, FACT, dem[mo])
            out[f"walk_{mo}"].update(demand25=crossing(xs, g, 0.75), demand50=crossing(xs, g, 0.5))
    if "far" in curves:   # far access (park-and-ride, 2-20 km), relative to its value at 2 km
        F = curves["far"]; xf = np.arange(F["x"][0], F["x"][-1] + 1, 50.0)
        fv = np.exp(np.interp(xf, F["x"], np.log(np.maximum(F["y"], 1e-12)))); fv = fv / fv[0]
        out["far"] = {"value25": crossing(xf, fv, 0.75), "value50": crossing(xf, fv, 0.5)}
        if dem and "far" in dem:
            g = np.interp(fv * F["y"][0], FACT, dem["far"])
            out["far"].update(demand25=crossing(xf, g / max(g[0], 1e-12), 0.75), demand50=crossing(xf, g / max(g[0], 1e-12), 0.5))
    hs = np.geomspace(1, 1440, 600)
    H = curves["headway"]; h = np.exp(np.interp(np.log(hs), np.log(H["x"]), np.log(H["y"]))); h = h / h.max()
    S = curves["speed"]; vs = np.linspace(S["x"][-1], S["x"][0], 600); s = np.interp(vs, S["x"], S["y"]); s = s / s.max()
    out["headway"] = {"value25": crossing(hs, h, 0.75), "value50": crossing(hs, h, 0.5)}
    out["speed"] = {"value25": crossing(vs, s, 0.75), "value50": crossing(vs, s, 0.5)}
    if dem:
        for key, xx, yy in (("headway", hs, h), ("speed", vs, s)):
            g = np.interp(yy, FACT, dem["service"])
            out[key].update(demand25=crossing(xx, g, 0.75), demand50=crossing(xx, g, 0.5))
    return out


def main():
    d0 = f3.Data3(); agg = {}
    for name in sys.argv[1:]:
        level = next((lv for lv in ("tract", "county") if name.endswith(f"_{lv}")), None)   # runs fitted on aggregated units are read at that level
        if level and level not in agg:
            agg[level] = f3.AggData3(d0, level)
        d = agg[level] if level else d0
        m, o = load(name)
        X = d.controls(f3.CONTROLS3[o["controls"]]) if o.get("controls", "none") != "none" else None
        P = parts(m, d, X)
        dem = {"service": demand_curve(P, d, 0, True, "service")}
        if o.get("far") == "term":
            dem["far"] = demand_curve(P, d, 0, False, "far")
        for mo, k in (("bus", 1), ("rail", 2), ("tram", 3)):
            dem[mo] = demand_curve(P, d, k, False)
        best = min(o["runs"], key=lambda r: r["loss"])
        res = {"best": points(best["curves"], dem), "starts_value": [points(r["curves"]) for r in o["runs"]]}
        (f3.RUNS / f"{name}_loss_points.json").write_text(json.dumps(res, indent=1))
        print(name)
        for key, v in res["best"].items():
            rng = {q: (np.nanmin([p[key][q] for p in res["starts_value"]]), np.nanmax([p[key][q] for p in res["starts_value"]])) for q in ("value25", "value50")}
            print(f"  {key:10s} " + " ".join(f"{q} {v[q]:7.1f}" for q in v) + "  | value over starts " + " ".join(f"{q} {a:.0f}-{b:.0f}" for q, (a, b) in rng.items()), flush=True)


if __name__ == "__main__":
    main()
