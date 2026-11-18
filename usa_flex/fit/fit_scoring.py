"""Fit ONE national scoring (all three curves, mode factors, walk coupling, axis weights) to the ACS public-transport share of workers, per block group.

    python fit/fit_scoring.py --model param|flex --link pooled|metro [--states VT NH ...] [--maxiter 60]

Predictor of a block group: sum(cell population * cell level of service) / block-group population / 100 (people no stop reaches count 0).
Outcome: transit commuters out of workers, binomial with a logistic link  logit(share) = a(group) + b * predictor ; a is one number (pooled) or one per metro (link=metro;
non-metro block groups share one) -- the intercepts belong to the fit only, the saved scoring is the same national functions in both cases.
No literature bands are enforced: the start is the current default, parameters may go anywhere (positivity/monotonicity only).
Held-out states (every 4th in alphabetical order) are scored with the fitted functions and their own refitted link (a, b).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import run_county as rc  # noqa: E402
from transitlos.flex import core, scoring as sc  # noqa: E402
from transitlos.flex.scoring import Curve, ScoreParams  # noqa: E402


# ------------------------------------------------------------------------------------------------------------------------------- data
class State:
    def __init__(self, abbr, bg_table):
        name = rc.STATE_DIRS[abbr]
        w = ROOT / "web" / name
        meta = json.load(open(w / "meta.json"))
        self.abbr = abbr
        S, P, nc, K = meta["S"], meta["P"], meta["nc"], meta["K"]
        self.prof = core.Profiles(np.fromfile(w / "stop_mode.bin", np.int8).reshape(S, P), np.fromfile(w / "stop_h.bin", np.float32).reshape(S, P), np.fromfile(w / "stop_v.bin", np.float32).reshape(S, P))
        self.cst = np.fromfile(w / "cell_station.bin", np.int32).reshape(nc, K)
        self.cdist = np.fromfile(w / "cell_dist.bin", np.uint16).reshape(nc, K)
        z = np.load(HERE / "data" / f"cells_{name}.npz")
        self.bg, self.pop = z["bg"], z["pop"].astype(np.float64)


def load_all(abbrs):
    bgt = pd.read_parquet(HERE / "data" / "blockgroups.parquet")
    return bgt, [State(a, bgt) for a in abbrs]


# ------------------------------------------------------------------------------------------------------------------------ parametrisation
def _sp(x): return np.log1p(np.exp(x))
def _sig(x): return 1 / (1 + np.exp(-x))
def _logit(p): return float(np.log(p / (1 - p)))


def modes(x0_, x1_):
    """(bus, tram, rail=1) with bus <= tram <= rail always: tram = sigmoid(x1), bus = tram * sigmoid(x0)."""
    tram = _sig(x1_)
    return float(tram * _sig(x0_)), float(tram)


H_GRID = np.exp(np.linspace(0, np.log(720.0), 60))


class Param:
    """15 numbers -> ScoreParams with the analytic curve families (speed Hill, headway log-linear elasticity, walk offset power)."""
    names = ["logit_bus_over_tram", "logit_tram", "v_anchor", "v_bonus", "v_n", "h_sat", "h_p", "h_q", "k", "p_shape", "t0_bus", "t0_tram", "t0_rail", "w_speed", "w_headway"]

    @staticmethod
    def x0():
        d = sc.DEFAULT
        return np.array([_logit(d.mrc[0] / d.mrc[1]), _logit(d.mrc[1]), np.log(d.v_anchor), np.log(d.v_bonus), np.log(d.v_n), np.log(d.h_sat), d.h_p, d.h_q, np.log(d.k), np.log(d.p_shape),
                         np.log(d.t0_base[0]), np.log(d.t0_base[1]), np.log(d.t0_base[2]), 0.0, 0.0])

    @staticmethod
    def params(x):
        x = np.asarray(x, dtype=float).copy()
        for i in (2, 3, 4, 5, 8, 9, 10, 11, 12):   # log-scale numbers: e^-6 .. e^6 (a random start must not underflow a positive parameter to 0)
            x[i] = np.clip(x[i], -6.0, 6.0)
        x[13:15] = np.clip(x[13:15], -4.0, 4.0)
        e = np.exp
        return sc.DEFAULT.with_(mrc=(*modes(x[0], x[1]), 1.0), v_anchor=float(e(x[2])), v_bonus=float(e(x[3])), v_n=float(e(x[4])), h_sat=float(e(x[5])),
                                h_p=float(x[6]), h_q=float(x[7]), k=float(e(x[8])), p_shape=float(e(x[9])), t0_base=(float(e(x[10])), float(e(x[11])), float(e(x[12]))),
                                w_speed=float(e(x[13])), w_headway=float(e(x[14])))

    @staticmethod
    def penalty(x):   # a longer headway must never score better: penalise any rise of H along the headway grid
        h = np.asarray(sc.headway_score(H_GRID, Param.params(x)), dtype=float)
        return 20.0 * float(np.sum(np.maximum(np.diff(h), 0.0))) + (1e3 if not np.all(np.isfinite(h)) else 0.0)


class Flex:
    """Monotone tables for the three curves (free shape on the whole range) + mode factors, coupling and axis weights.

    headway: knots at 1,2,3,5,8,12,20,30,45,60,90,120,240,720 min, value 1 at the first knot, non-increasing;  speed: knots 0,5,10,15,20,25,30,40,50,70,100,150 km/h, value 0 at 0, non-decreasing, free ceiling;
    walk: knots u = 0,.25,.5,1,1.5,2,3,4,6,10,20,40, value 1 at 0, non-increasing, >= 0.  All steps are softplus-parametrised so monotonicity always holds."""
    HK = np.array([1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 240, 720.0])
    VK = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 70, 100, 150.0])
    UK = np.array([0, .25, .5, 1, 1.5, 2, 3, 4, 6, 10, 20, 40.0])
    nh, nv, nu = len(HK) - 1, len(VK) - 1, len(UK) - 1
    names = (["logit_bus_over_tram", "logit_tram", "k", "t0_bus", "t0_tram", "t0_rail", "w_speed", "w_headway"] + [f"h{i}" for i in range(nh)] + [f"v{i}" for i in range(nv)] + [f"u{i}" for i in range(nu)])

    @classmethod
    def x0(cls):
        d = sc.DEFAULT
        def inv(y):   # softplus inverse of positive steps
            y = np.maximum(y, 1e-6)
            return np.log(np.expm1(y))
        h = sc.headway_score(cls.HK, d); v = sc.speed_score(cls.VK, d); u = sc.walk_decay_u(cls.UK) if hasattr(sc, "walk_decay_u") else (1 + cls.UK) ** -d.p_shape
        return np.concatenate([[_logit(d.mrc[0] / d.mrc[1]), _logit(d.mrc[1]), np.log(d.k), np.log(d.t0_base[0]), np.log(d.t0_base[1]), np.log(d.t0_base[2]), 0.0, 0.0],
                               inv(-np.diff(h) / (1 - h[-1] + 1e-9) * 1.0), inv(np.diff(v)), inv(-np.diff(u))])

    @classmethod
    def params(cls, x):
        e = np.exp
        h = 1 - np.concatenate([[0], np.cumsum(_sp(x[8:8 + cls.nh]) * 0.2)]); h = np.clip(h, 0.0, 1.0)
        v = np.concatenate([[0], np.cumsum(_sp(x[8 + cls.nh:8 + cls.nh + cls.nv]) * 0.2)])
        u = 1 - np.concatenate([[0], np.cumsum(_sp(x[8 + cls.nh + cls.nv:]) * 0.2)]); u = np.clip(u, 0.0, 1.0)
        hc = Curve.from_function(lambda t: np.interp(np.log(t), np.log(cls.HK), h), 1.0, 720.0, 512, "log")
        vc = Curve.from_function(lambda t: np.interp(t, cls.VK, v), 0.0, 150.0, 512, "lin")
        uc = Curve.from_function(lambda t: np.interp(t, cls.UK, u), 0.0, 40.0, 1024, "lin")
        return sc.DEFAULT.with_(mrc=(*modes(x[0], x[1]), 1.0), k=float(e(x[2])), t0_base=(float(e(x[3])), float(e(x[4])), float(e(x[5]))),
                                w_speed=float(e(x[6])), w_headway=float(e(x[7])), headway_curve=hc, speed_curve=vc, walk_curve=uc)

    @classmethod
    def penalty(cls, x):   # tiny roughness penalty on the step sizes (keeps tables from zig-zagging without forcing a shape)
        p = 0.0
        for a, b in ((8, 8 + cls.nh), (8 + cls.nh, 8 + cls.nh + cls.nv), (8 + cls.nh + cls.nv, len(x))):
            s = _sp(x[a:b]); p += 1e-4 * float(np.sum(np.diff(s) ** 2))
        return p


# ------------------------------------------------------------------------------------------------------------------------------- objective
def predictor(states, params, bg_pop, n_bg):
    num = np.zeros(n_bg)
    for s in states:
        los, _, _ = core.recompute_cells_fast(s.cst, s.cdist, s.prof, params)
        ok = s.bg >= 0
        num += np.bincount(s.bg[ok], weights=s.pop[ok] * los[ok].astype(np.float64), minlength=n_bg)
    return np.clip(num / np.maximum(bg_pop, 1.0) / 100.0, 0.0, None)


def fit_link(f, yv, nv, grp, iters=25):
    """Binomial logistic link logit(p) = a[grp] + b f by alternating Newton steps; returns (neg log-likelihood, a, b)."""
    G = grp.max() + 1
    a = np.full(G, np.log((yv.sum() + 0.5) / (nv.sum() - yv.sum() + 0.5)))
    b = 0.0
    for _ in range(iters):
        eta = a[grp] + b * f
        p = 1 / (1 + np.exp(-eta))
        r = yv - nv * p
        w = np.maximum(nv * p * (1 - p), 1e-12)
        if G > 1:
            a = a + np.clip(np.bincount(grp, weights=r, minlength=G) / np.maximum(np.bincount(grp, weights=w, minlength=G), 1e-6), -2, 2)
            a = np.clip(a, -25.0, 5.0)    # a metro with no transit commuters at all: intercept stops at a very small share instead of running off
        else:
            a = a + r.sum() / w.sum()
        eta = a[grp] + b * f; p = 1 / (1 + np.exp(-eta)); r = yv - nv * p; w = np.maximum(nv * p * (1 - p), 1e-12)
        b = b + float(np.sum(r * f) / (np.sum(w * f * f) + 1e-9))
    eta = a[grp] + b * f
    ll = float(np.sum(yv * eta - nv * np.logaddexp(0, eta)))
    return -ll, a, b


class Problem:
    def __init__(self, bgt, states, link):
        self.states = states
        self.bgt = bgt
        self.n_bg = len(bgt)
        ok = bgt.state.isin([st.abbr for st in states]) & (bgt.acs5_workers.fillna(0) > 0) & bgt.publicTransportCommuters.notna() & (bgt.population.fillna(0) > 0)
        self.idx = np.flatnonzero(ok.to_numpy())
        self.y = bgt.publicTransportCommuters.to_numpy()[self.idx].clip(min=0).astype(float)
        self.n = bgt.acs5_workers.to_numpy()[self.idx].astype(float)
        self.y = np.minimum(self.y, self.n)
        self.pop = bgt.population.to_numpy().astype(float)
        self.grp = pd.factorize(bgt.cbsa.fillna("").to_numpy()[self.idx])[0] if link == "metro" else np.zeros(len(self.idx), int)
        self.null = fit_link(np.zeros(len(self.idx)), self.y, self.n, self.grp)[0]
        self.nfev = 0

    def loss(self, params, mask=None):
        f = predictor(self.states, params, self.pop, self.n_bg)[self.idx]
        m = np.ones(len(self.idx), bool) if mask is None else mask
        g = pd.factorize(self.grp[m])[0]
        return fit_link(f[m], self.y[m], self.n[m], g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="param")
    ap.add_argument("--link", default="pooled")
    ap.add_argument("--states", nargs="*")
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--out", default=None)
    ap.add_argument("--starts", type=int, default=1, help="1 = start at the default scoring; N>1 adds N-1 random starts")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    abbrs = a.states or [k for k in rc.STATE_DIRS if (ROOT / "web" / rc.STATE_DIRS[k] / "meta.json").is_file()]
    test_ab = set(sorted(abbrs)[3::4])
    bgt, allst = load_all(abbrs)
    # train block groups = those of non-held-out states
    stm = bgt.state.to_numpy()
    prob = Problem(bgt, allst, a.link)
    train_mask = ~np.isin(stm[prob.idx], list(test_ab))
    M = Param if a.model == "param" else Flex
    x0 = M.x0()
    t0 = time.time()

    def obj(x):
        prob.nfev += 1
        l = prob.loss(M.params(x), train_mask)[0] + M.penalty(x)
        return l / prob.n[train_mask].sum()

    base, _, _ = prob.loss(sc.DEFAULT, train_mask)
    nullt = fit_link(np.zeros(train_mask.sum()), prob.y[train_mask], prob.n[train_mask], pd.factorize(prob.grp[train_mask])[0])[0]
    print(f"{len(abbrs)} states, {len(prob.idx)} block groups ({train_mask.sum()} train); train deviance-like loss: null {nullt:.0f}, default scoring {base:.0f}  (R2_mcf {1 - base / nullt:.4f})", flush=True)
    rng = np.random.default_rng(a.seed)
    runs = []
    for i in range(a.starts):
        xs = x0 if i == 0 else x0 + rng.normal(0, 0.8, size=len(x0))   # random start: every number moved by a N(0, 0.8) step in its own (log/logit/softplus) scale
        r_i = minimize(obj, xs, method="L-BFGS-B", options={"maxiter": a.maxiter, "maxfun": a.maxiter * (len(x0) + 2), "eps": 1e-3})
        runs.append({"start": "default" if i == 0 else f"random{i}", "loss": float(r_i.fun), "x": r_i.x.tolist()})
        print(f"start {runs[-1]['start']}: start loss {obj(xs):.5f} -> fitted {r_i.fun:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    best = min(runs, key=lambda q: q["loss"])
    class _R: pass
    r = _R(); r.x = np.array(best["x"])
    p = M.params(r.x)
    fit_loss = prob.loss(p, train_mask)[0]
    test_mask = ~train_mask
    res = {"model": a.model, "link": a.link, "train_null": nullt, "train_default": base, "train_fit": fit_loss, "R2_default": 1 - base / nullt, "R2_fit": 1 - fit_loss / nullt, "evals": prob.nfev, "seconds": round(time.time() - t0)}
    if test_mask.any():
        tn = fit_link(np.zeros(test_mask.sum()), prob.y[test_mask], prob.n[test_mask], pd.factorize(prob.grp[test_mask])[0])[0]
        res.update({"test_null": tn, "test_R2_default": 1 - prob.loss(sc.DEFAULT, test_mask)[0] / tn, "test_R2_fit": 1 - prob.loss(p, test_mask)[0] / tn, "test_states": sorted(test_ab)})
    res["runs"] = [{k: v for k, v in q.items() if k != "x"} for q in runs]; res["x"] = r.x.tolist(); res["names"] = M.names
    res["params"] = json.loads(p.to_json())
    out = Path(a.out or HERE / "data" / f"fit_{a.model}_{a.link}.json")
    out.write_text(json.dumps(res))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k not in ("x", "names", "params")}, flush=True)


if __name__ == "__main__":
    main()
