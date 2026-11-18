"""National scoring fit v2: multi-start on a metro-weighted sample, more structural freedom, bootstrap error bands, per-metro fits.

    python fit/fit2.py prep                                        # compact sample of the cells (fit/data/sample/*.npz)
    python fit/fit2.py fit   --model param|flex --agg nested|power --seed 0 --starts 6 --out fit/data/f2_<name>_s0.json
    python fit/fit2.py boot  --from <best.json> --B 30 --out ...      # bootstrap by metro area (CBSA): refit from the best solution on resampled metros
    python fit/fit2.py metros --from <best.json> --top 40 --out ...   # one fit per big metro (official CBSA boundaries), compared with the national fit

One national set of functions: the link is  logit(share) = a + b * f**gamma  with ONE a, b, gamma for the whole country (pooled), f = population-weighted mean cell score
of the block group.  Weighted binomial likelihood: the 80 metros with the most transit commuters are used completely (weight 1), the other block groups are sampled 1 in 10 (weight 10).
No literature bands (see fit_scoring.py): positivity of scales, mode order bus <= tram <= rail and monotone curves only.
"""
import argparse
import os
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
sys.path.insert(0, str(HERE))
import fit_scoring as fs  # noqa: E402
import run_county as rc  # noqa: E402
from transitlos.flex import core, scoring as sc  # noqa: E402
from transitlos.flex import backend as xb, xp_scoring  # noqa: E402

SAMPLE = HERE / "data" / "sample"
SAMPLE_COARSE = HERE / "data" / "sample_coarse"
N_TOP = 80


# ----------------------------------------------------------------------------------------------------------------------------------- data
class Sub:
    """Cells of one state restricted to some block groups, with the station arrays compacted to the stations those cells use."""

    def __init__(self, prof, cst, cdist, bg, pop):
        self.prof, self.cst, self.cdist, self.bg, self.pop = prof, cst, cdist, bg, pop


def compact(prof_arrays, cst, cdist, bg, pop, keep):
    mode, h, v = prof_arrays
    cst, cdist, bg, pop = cst[keep], cdist[keep], bg[keep], pop[keep]
    used = np.unique(cst[cst >= 0])
    remap = np.full(len(mode), -1, np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    cst2 = np.where(cst >= 0, remap[np.maximum(cst, 0)], -1).astype(np.int32)
    return mode[used], h[used], v[used], cst2, cdist, bg, pop


def metro_choice(bgt):
    t = bgt.assign(tc=bgt.publicTransportCommuters.fillna(0)).groupby("cbsa").tc.sum()
    t = t[t.index != ""].sort_values(ascending=False)
    return list(t.index[:N_TOP]), t


def prep():
    """US-wide sample. Every block group with any transit commuters or a default score >= 1 is kept (weight 1); the uninformative ones (no transit commuter and default score < 1)
    are kept 1 in 5 with weight 5, so the likelihood is unbiased. The choice uses the DEFAULT scoring and the observed counts only, never a fitted parameter."""
    SAMPLE.mkdir(parents=True, exist_ok=True)
    bgt = pd.read_parquet(HERE / "data" / "blockgroups.parquet")
    pop = bgt.population.fillna(0).to_numpy().astype(float)
    num = np.zeros(len(bgt)); states = []
    for abbr in rc.STATE_DIRS:
        if not (ROOT / "web" / rc.STATE_DIRS[abbr] / "meta.json").is_file():
            continue
        st = fs.State(abbr, bgt)
        los, _, _ = core.recompute_cells_fast(st.cst, st.cdist, st.prof, sc.DEFAULT)
        ok = st.bg >= 0
        num += np.bincount(st.bg[ok], weights=st.pop[ok] * los[ok].astype(np.float64), minlength=len(bgt))
        states.append(st)
    f0 = num / np.maximum(pop, 1.0)
    informative = (bgt.publicTransportCommuters.fillna(0).to_numpy() > 0) | (f0 >= 1.0)
    rest = (pd.util.hash_pandas_object(bgt.GEOID, index=False).to_numpy() % 5) == 0
    w_bg = np.where(informative, 1.0, np.where(rest, 5.0, 0.0))
    np.save(HERE / "data" / "bg_weights.npy", w_bg)
    keep_bg = w_bg > 0
    print("block groups:", len(bgt), "kept", int(keep_bg.sum()), "weighted total", float(w_bg.sum()), flush=True)
    for st in states:
        keep = (st.bg >= 0) & keep_bg[np.maximum(st.bg, 0)]
        if not keep.any():
            continue
        m, h, v, cst, cdist, bg, pop_c = compact((st.prof.mode, st.prof.h, st.prof.v), st.cst, st.cdist, st.bg, st.pop, keep)
        np.savez_compressed(SAMPLE / f"{rc.STATE_DIRS[st.abbr]}.npz", mode=m, h=h, v=v, cst=cst, cdist=cdist, bg=bg, pop=pop_c)
        print(st.abbr, int(keep.sum()), "cells", flush=True)


def load_subs(cell_filter=None, sample_dir=None):
    """Sample as Sub objects; `cell_filter(bg_array) -> bool mask` restricts further (used for a single metro)."""
    subs = []
    for f in sorted((sample_dir or SAMPLE).glob("*.npz")):
        z = np.load(f)
        arr = (z["mode"], z["h"], z["v"])
        cst, cdist, bg, pop = z["cst"], z["cdist"], z["bg"], z["pop"]
        if cell_filter is not None:
            keep = cell_filter(bg)
            if not keep.any():
                continue
            m, h, v, cst, cdist, bg, pop = compact(arr, cst, cdist, bg, pop, keep)
            arr = (m, h, v)
        subs.append(Sub(core.Profiles(*arr), cst, cdist, bg, pop))
    return subs


class Data:
    """Block-group table + weights for the likelihood."""

    def __init__(self, cbsa_filter=None, wfile="bg_weights.npy"):
        bgt = pd.read_parquet(HERE / "data" / "blockgroups.parquet")
        self.bgt = bgt
        top, tot = metro_choice(bgt)
        self.top, self.tot = top, tot
        self.w_bg = np.load(HERE / "data" / wfile)
        ok = (self.w_bg > 0) & (bgt.acs5_workers.fillna(0) > 0).to_numpy() & bgt.publicTransportCommuters.notna().to_numpy() & (bgt.population.fillna(0) > 0).to_numpy()
        if cbsa_filter is not None:
            ok &= bgt.cbsa.eq(cbsa_filter).to_numpy()
        self.idx = np.flatnonzero(ok)
        self.n = bgt.acs5_workers.to_numpy()[self.idx].astype(float)
        self.y = np.minimum(bgt.publicTransportCommuters.to_numpy()[self.idx].clip(min=0).astype(float), self.n)
        self.w = self.w_bg[self.idx]
        self.pop = bgt.population.to_numpy().astype(float)
        self.cbsa = bgt.cbsa.fillna("").to_numpy()[self.idx]
        self.nbg = len(bgt)
        self.base_w = self.w.copy()


# --------------------------------------------------------------------------------------------------------------------------------- model
HEADWAY_FLOOR_K = float(os.environ.get('HEADWAY_FLOOR_K', '0'))   # 0 switches the floor off


class Model:
    """theta (objective 'prob'): None, a number (fixed share threshold, e.g. 0.10) or 'free' (one more parameter, log theta, fitted with the rest).
    fixgamma: the score itself is the probability (gamma = 1) instead of score ** gamma."""

    def __init__(self, name, agg, theta=None, fixgamma=False, ordinal=None):
        self.base = fs.Param if name == "param" else fs.Flex
        self.name, self.agg = name, agg
        self.theta, self.fixgamma = theta, fixgamma
        self.ordinal = None if not ordinal else [float(t) for t in ordinal]   # objective 'ordinal': the fixed share thresholds
        self.alpha_init = None
        self.nb = len(self.base.x0())
        self.names = list(self.base.names) + ["log_gamma"] + (["agg_p", "log_w_mode"] if agg == "power" else []) + (["log_theta"] if theta == "free" else [])
        self.o0 = len(self.names)
        if self.ordinal:
            self.names += ["log_beta", "alpha1"] + [f"log_dalpha{i}" for i in range(1, len(self.ordinal))]

    def x0(self):
        ex = [0.0] + ([0.0, 0.0] if self.agg == "power" else []) + ([float(np.log(0.10))] if self.theta == "free" else [])
        if self.ordinal:
            a = self.alpha_init if self.alpha_init is not None else np.linspace(0.0, -4.0, len(self.ordinal))
            ex += [float(np.log(6.0)), float(a[0])] + [float(np.log(max(a[i - 1] - a[i], 0.05))) for i in range(1, len(a))]
        return np.concatenate([self.base.x0(), ex])

    def ord_params(self, x):
        """(alpha_k decreasing in k, beta > 0) of the cumulative-logit link."""
        K = len(self.ordinal)
        beta = float(np.exp(np.clip(x[self.o0], -5.0, 5.0)))
        d = np.exp(np.clip(np.asarray(x[self.o0 + 2: self.o0 + 1 + K]), -6.0, 3.0))
        return float(x[self.o0 + 1]) - np.concatenate([[0.0], np.cumsum(d)]), beta

    def theta_of(self, x):
        if self.theta == "free":
            return float(np.clip(np.exp(x[-1]), 0.005, 0.6))
        return None if self.theta is None else float(self.theta)

    def params(self, x):
        p = self.base.params(x[: self.nb])
        if self.agg == "power":
            p = p.with_(agg="power_mean", agg_p=float(np.clip(x[self.nb + 1], -3.0, 3.0)), w_mode=float(np.exp(np.clip(x[self.nb + 2], -4, 4))))
        return p

    def gamma(self, x):
        return 1.0 if self.fixgamma else float(np.exp(np.clip(x[self.nb], -1.5, 1.5)))

    def penalty(self, x):
        return self.base.penalty(x[: self.nb])

    def floor_penalty(self, x):
        """No factor of the score may collapse to ~0: poor service, far-away stops, slow vehicles or bus-only places would all tie and the score would stop telling them apart.
        Floors (0 whenever they hold; the penalty is in loss units and large, so in effect a hard constraint):
          headway   H(h) >= min(1, HEADWAY_FLOOR_K / h)   (1 at 3 min, 0.2 at 15, 0.1 at 30, 0.05 at 60, 0.0125 at 4 h)
          walking   decay (stop score 0.5) at 1 km >= 0.2 (bus, tram) / 0.3 (rail), and at 500 m >= 0.4
          speed     V(2 km/h) >= 0.1 * V(30 km/h)
          mode      bus and tram capacity factor >= 0.3 (rail = 1)"""
        if not HEADWAY_FLOOR_K:
            return 0.0
        from transitlos.flex.scoring import headway_score, speed_score, walk_decay
        p = self.params(x)
        h = np.array([3, 5, 8, 10, 15, 20, 30, 45, 60, 90, 120, 240, 480], float)
        pen = np.sum(np.maximum(np.minimum(1.0, HEADWAY_FLOOR_K / h) - np.asarray(headway_score(h, p), float), 0.0))
        for mode, d1000 in ((0, 0.2), (1, 0.2), (2, 0.3)):
            pen += max(d1000 - float(walk_decay(1000.0, 0.5, mode, p)), 0.0) + max(0.4 - float(walk_decay(500.0, 0.5, mode, p)), 0.0)
        v = np.asarray(speed_score(np.array([2.0, 30.0]), p), float)
        pen += max(0.1 * v[1] - v[0], 0.0)
        pen += sum(max(0.3 - float(m), 0.0) for m in list(p.mrc)[:2])
        return float(10.0 * pen)


def predictor(subs, params, nbg, ss_cache=None):
    num = np.zeros(nbg)
    for s in subs:
        los, _, _ = core.recompute_cells_fast(s.cst, s.cdist, s.prof, params)
        ok = s.bg >= 0
        num += np.bincount(s.bg[ok], weights=s.pop[ok] * los[ok].astype(np.float64), minlength=nbg)
    return num


def link_nll(f, y, n, w, iters=20):
    """Weighted binomial logistic regression logit p = a + b f (Newton); returns (neg log-lik, a, b)."""
    a = np.log((np.sum(w * y) + 0.5) / (np.sum(w * (n - y)) + 0.5)); b = 0.0
    for _ in range(iters):
        eta = a + b * f
        p = 1 / (1 + np.exp(-eta))
        wt = np.maximum(w * n * p * (1 - p), 1e-12)
        r = w * (y - n * p)
        g = np.array([r.sum(), (r * f).sum()])
        H = np.array([[wt.sum(), (wt * f).sum()], [(wt * f).sum(), (wt * f * f).sum()]]) + 1e-9 * np.eye(2)
        d = np.linalg.solve(H, g)
        d = np.clip(d, -3, 3)
        a, b = a + d[0], b + d[1]
    eta = a + b * f
    return float(-np.sum(w * (y * eta - n * np.logaddexp(0, eta)))), float(a), float(b)


def wls(f, share, w):
    """Weighted least squares  share ~ a + b f  (closed form); returns (weighted mean squared error, a, b)."""
    sw = w.sum(); mf = (w * f).sum() / sw; ms = (w * share).sum() / sw
    vf = (w * (f - mf) ** 2).sum(); b = (w * (f - mf) * (share - ms)).sum() / vf if vf > 0 else 0.0
    a = ms - b * mf
    return float((w * (share - a - b * f) ** 2).sum() / sw), float(a), float(b)


def prob_loss(p, share, w, theta):
    """The score is the probability that a block group has more than `theta` transit commuters. Population-weighted log loss of that probability against the 0/1 outcome
    (share > theta), divided by the log loss of the constant base rate (so 1 - loss is the skill, comparable between thresholds: it is what lets theta be fitted)."""
    ind = (share > theta).astype(float); sw = w.sum()
    p = np.clip(p, 1e-4, 1 - 1e-4); p0 = float(np.clip((w * ind).sum() / sw, 1e-6, 1 - 1e-6))
    ll = -(w * (ind * np.log(p) + (1 - ind) * np.log1p(-p))).sum() / sw
    return float(ll / -(p0 * np.log(p0) + (1 - p0) * np.log1p(-p0)))


def ordinal_loss(f, share, w, thresholds, alpha, beta, h0=None):
    """Cumulative-logit (proportional odds) loss, population-weighted: P(share > theta_k) = sigmoid(alpha_k + beta * f), one binary log loss per threshold, each divided by the log loss
    of that threshold's base rate and averaged (so every threshold counts equally and 1 - loss is the mean skill)."""
    sw = w.sum(); tot = 0.0
    for k, th in enumerate(thresholds):
        ind = (share > th).astype(float)
        p = np.clip(1.0 / (1.0 + np.exp(-(alpha[k] + beta * f))), 1e-6, 1 - 1e-6)
        ll = -(w * (ind * np.log(p) + (1 - ind) * np.log1p(-p))).sum() / sw
        if h0 is None:
            p0 = float(np.clip((w * ind).sum() / sw, 1e-6, 1 - 1e-6)); h = -(p0 * np.log(p0) + (1 - p0) * np.log1p(-p0))
        else:
            h = h0[k]
        tot += ll / h
    return float(tot / len(thresholds))


class Problem:
    """objective 'binom': weighted binomial likelihood (large block groups weigh more through their worker counts);
    objective 'wls': weighted least squares of the transit share on the score with the block-group POPULATION as the weight."""

    def __init__(self, data: Data, subs, model: Model, objective: str = "binom"):
        self.d, self.subs, self.m, self.obj = data, subs, model, objective
        self.ev = xp_scoring.XPData(subs, data.nbg) if xb.device() == "cuda" else None   # CUDA when usable, else the C core on the CPU
        self.nfev = 0
        self.share = data.y / data.n
        self.pw = data.pop[data.idx]
        self._fcache = {}
        if objective == "ordinal" and model.ordinal:
            w0 = data.w * self.pw; sw = w0.sum(); self.h0 = []; p0s = []
            for th in model.ordinal:
                p0 = float(np.clip((w0 * (self.share > th)).sum() / sw, 1e-6, 1 - 1e-6)); p0s.append(p0)
                self.h0.append(-(p0 * np.log(p0) + (1 - p0) * np.log1p(-p0)))
            model.alpha_init = np.log(np.array(p0s) / (1 - np.array(p0s))) - 3.0   # (f is about 0.3 on average, beta starts at 6)
        self.null = 1.0 if objective in ("prob", "ordinal") else link_nll(np.zeros(len(data.idx)), data.y, data.n, data.w)[0] if objective == "binom" else wls(np.zeros(len(data.idx)), self.share, data.w * self.pw)[0]

    def f(self, x, params=None):
        # the score of every block group only depends on the scoring parameters: a finite-difference step on gamma / the link parameters re-uses it
        key = np.asarray(x[: self.m.nb + (3 if self.m.agg == "power" else 1)], float)[[*range(self.m.nb), *([self.m.nb + 1, self.m.nb + 2] if self.m.agg == "power" else [])]].tobytes()
        raw = self._fcache.get(key) if params is None else None
        if raw is None:
            p = params or self.m.params(x)
            num = self.ev.numerator(json.loads(p.to_json())) if self.ev is not None else predictor(self.subs, p, self.d.nbg)
            raw = np.maximum(num[self.d.idx] / np.maximum(self.d.pop[self.d.idx], 1.0) / 100.0, 0.0)
            if params is None:
                if len(self._fcache) > 64:
                    self._fcache.clear()
                self._fcache[key] = raw
        return np.power(raw, self.m.gamma(x))

    def loss(self, x, w=None):
        self.nfev += 1
        f = self.f(x)
        ww = self.d.w if w is None else w
        pen = self.m.floor_penalty(x)
        if self.obj == "ordinal":
            al, be = self.m.ord_params(x)
            return ordinal_loss(f, self.share, ww * self.pw, self.m.ordinal, al, be, self.h0) + pen
        if self.obj == "prob":
            return prob_loss(f, self.share, ww * self.pw, self.m.theta_of(x)) + pen
        if self.obj == "wls":
            return wls(f, self.share, ww * self.pw)[0] + pen
        return link_nll(f, self.d.y, self.d.n, ww)[0] + pen * self.d.n.sum()


def curves(model, x):
    """Sampled shapes of the fitted scoring (for the report): headway, speed, walk-decay-vs-u curve, mode factors."""
    p = model.params(x)
    hs = np.array([1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 60, 90, 120, 240, 480], float)
    vs = np.array([2, 5, 10, 15, 20, 25, 30, 40, 50, 70, 100], float)
    ds = np.array([0, 100, 200, 300, 400, 500, 750, 1000], float)
    from transitlos.flex.scoring import headway_score, speed_score, walk_decay
    return {"headway_min": hs.tolist(), "H": [float(headway_score(h, p)) for h in hs], "speed_kmh": vs.tolist(), "V": [float(speed_score(v, p)) for v in vs],
            "dist_m": ds.tolist(), "D_bus_score05": [float(walk_decay(d, 0.5, 0, p)) for d in ds], "D_rail_score05": [float(walk_decay(d, 0.5, 2, p)) for d in ds],
            "mrc": list(p.mrc), "gamma": model.gamma(x), "agg": p.agg, "agg_p": p.agg_p, "w": [p.w_mode, p.w_speed, p.w_headway], "k": p.k, "t0": list(p.t0_base)}


# ---------------------------------------------------------------------------------------------------------------------------------- stages
class _Stall(Exception):
    pass


def _minimize(obj, x0, maxiter, patience=0):
    """L-BFGS-B; with `patience` > 0 a start that has gained less than 0.5% of its loss after that many iterations (a flat / degenerate region) is stopped early."""
    state = {"n": 0, "f0": None}

    def cb(xk, *args):
        state["n"] += 1
        if patience and state["n"] == patience:
            f = obj(xk)
            if state["f0"] is not None and f > 0.995 * state["f0"]:
                raise _Stall()
    state["f0"] = obj(x0)
    try:
        return minimize(obj, x0, method="L-BFGS-B", callback=cb, options={"maxiter": maxiter, "maxfun": maxiter * (len(x0) + 2), "eps": 1e-3})
    except _Stall:
        class R:   # the stalled start keeps its starting point: it is not a candidate
            x = x0
            fun = state["f0"]
        return R


def _thresholds(a):
    t = getattr(a, "thresholds", None)
    return [float(v) for v in str(t).split(",")] if t else None


def _theta(a):
    t = getattr(a, "theta", None)
    return None if t in (None, "") else ("free" if t == "free" else float(t))


def do_theta(a):
    """Fits the share threshold theta too (objective 'prob'). The loss is piecewise constant in theta (the 0/1 outcome flips for a few block groups at a time), so a gradient method
    cannot move it: theta is profiled instead. For every theta of a grid the scoring parameters are re-optimised (warm start from the neighbouring theta, coarse sample, then the best
    thetas on the full sample), then theta is refined by golden-section search on the full-sample skill. Runs on the GPU when CUDA is usable."""
    r0, d0 = best_of(a.src)
    x_start = np.array(r0["x"])
    coarse_d, coarse_s = Data(wfile="bg_weights_coarse.npy"), load_subs(sample_dir=SAMPLE_COARSE)
    full_d, full_s = Data(), load_subs()
    grid = sorted(float(t) for t in a.grid.split(","))

    def fit_at(theta, x0, data, subs, maxiter):
        model = Model(d0["model"], d0["agg"], theta, d0.get("fixgamma", True))
        prob = Problem(data, subs, model, "prob")
        rr = minimize(lambda x: prob.loss(x), x0, method="L-BFGS-B", options={"maxiter": maxiter, "maxfun": maxiter * (len(x0) + 2), "eps": 1e-3})
        return float(rr.fun), np.asarray(rr.x)

    t0 = time.time(); rows = {}
    i0 = int(np.argmin([abs(t - float(d0.get("theta") or 0.1)) for t in grid]))
    for order in (range(i0, len(grid)), range(i0 - 1, -1, -1)):
        x = x_start.copy()
        for i in order:
            loss, x = fit_at(grid[i], x, coarse_d, coarse_s, a.maxiter)
            rows[grid[i]] = {"theta": grid[i], "coarse_loss": loss, "x": x.tolist()}
            print(f"theta {grid[i]:.3f}: coarse skill {1 - loss:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    ranked = sorted(rows.values(), key=lambda r: r["coarse_loss"])[: a.polish]
    full = {}
    def polish(theta, x0):
        loss, x = fit_at(theta, np.asarray(x0), full_d, full_s, a.maxiter)
        full[theta] = {"theta": theta, "loss": loss, "skill": 1 - loss, "x": x.tolist()}
        print(f"  full sample theta {theta:.4f}: skill {1 - loss:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        return loss
    for r in ranked:
        polish(r["theta"], r["x"])
    # golden-section refinement of theta around the best full-sample value (skill is not exactly unimodal; this finds a local optimum between its two neighbours on the grid)
    bt = min(full, key=lambda t: full[t]["loss"]); gi = grid.index(bt) if bt in grid else 0
    lo, hi = grid[max(gi - 1, 0)], grid[min(gi + 1, len(grid) - 1)]
    phi = (5 ** 0.5 - 1) / 2; c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    xb = full[bt]["x"]
    fc, fd = polish(c, xb), polish(d, xb)
    for _ in range(a.refine):
        if fc < fd:
            hi, d, fd = d, c, fc; c = hi - phi * (hi - lo); fc = polish(c, full[min(full, key=lambda t: full[t]["loss"])]["x"])
        else:
            lo, c, fc = c, d, fd; d = lo + phi * (hi - lo); fd = polish(d, full[min(full, key=lambda t: full[t]["loss"])]["x"])
    best = min(full.values(), key=lambda r: r["loss"])
    print(f"best theta {best['theta']:.4f}: skill {best['skill']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    Path(a.out).write_text(json.dumps({"model": d0["model"], "agg": d0["agg"], "objective": "prob", "theta": "profile", "fixgamma": d0.get("fixgamma", True), "null": 1.0,
                                       "best": best, "profile_full": sorted(full.values(), key=lambda r: r["theta"]), "profile_coarse": sorted(rows.values(), key=lambda r: r["theta"]),
                                       "runs": [{"loss": best["loss"], "x": best["x"], "theta": best["theta"]}], "names": Model(d0["model"], d0["agg"], best["theta"], True).names}))


def do_ordinal_report(a):
    """Per-threshold skill and calibration of an ordinal (multi-threshold) fit on ALL the fitting block groups, population-weighted: for each threshold the log-loss skill versus the base rate,
    and a calibration table (weighted deciles of the predicted P(share > theta): mean prediction vs the weighted share of block groups really above theta)."""
    r, d = best_of(a.src)
    data, subs = Data(), load_subs()
    model = Model(d["model"], d["agg"], d.get("theta"), d.get("fixgamma", False), d.get("ordinal"))
    prob = Problem(data, subs, model, "ordinal")
    x = np.array(r["x"]); f = prob.f(x); al, be = model.ord_params(x); w = data.w * prob.pw; sw = w.sum()
    out = {"thresholds": model.ordinal, "alpha": al.tolist(), "beta": be, "gamma": model.gamma(x), "skill": [], "calibration": {}, "mean_skill": 1 - prob.loss(x)}
    for k, th in enumerate(model.ordinal):
        ind = (prob.share > th).astype(float)
        p = np.clip(1 / (1 + np.exp(-(al[k] + be * f))), 1e-6, 1 - 1e-6)
        ll = -(w * (ind * np.log(p) + (1 - ind) * np.log1p(-p))).sum() / sw
        out["skill"].append(1 - ll / prob.h0[k])
        o = np.argsort(p); cw = np.cumsum(w[o]) / sw; bins = np.minimum((cw * 10).astype(int), 9); rows = []
        for b in range(10):
            m = o[bins == b]
            if len(m):
                rows.append({"decile": b + 1, "predicted": float((w[m] * p[m]).sum() / w[m].sum()), "observed": float((w[m] * ind[m]).sum() / w[m].sum()), "population_share": float(w[m].sum() / sw)})
        out["calibration"][f"{th:g}"] = rows
    Path(a.out).write_text(json.dumps(out)); print(json.dumps({"thresholds": out["thresholds"], "skill": [round(v, 3) for v in out["skill"]], "mean_skill": round(out["mean_skill"], 4)}))


def do_fit(a):
    """Multi-start fit. --sample coarse runs the starts on the ~10% sample (data/sample_coarse, built by build_coarse.py) and --polish K then re-optimises the K best on the full sample,
    so the reported losses are always those of the full sample. Runs on the GPU when CUDA is usable (transitlos.flex.backend), else on the CPU."""
    coarse = getattr(a, "sample", "full") == "coarse"
    data = Data(wfile="bg_weights_coarse.npy") if coarse else Data()
    subs = load_subs(sample_dir=SAMPLE_COARSE) if coarse else load_subs()
    model = Model(a.model, a.agg, _theta(a), a.fixgamma, _thresholds(a))
    prob = Problem(data, subs, model, a.objective)
    rng = np.random.default_rng(a.seed)
    x0 = model.x0()
    t0 = time.time()
    norm = data.n.sum() if a.objective == "binom" else 1.0
    obj = lambda x: prob.loss(x) / norm
    print(f"device: {xb.device()}; sample: {'coarse' if coarse else 'full'}; block groups {len(data.idx):,}", flush=True)
    runs = []
    for i in range(a.starts):
        xs = x0 if (i == 0 and a.seed == 0) else x0 + rng.normal(0, a.spread, size=len(x0))
        try:
            r = _minimize(obj, xs, a.maxiter, patience=getattr(a, "patience", 0))
            runs.append({"start": i, "start_loss": float(obj(xs)), "loss": float(r.fun), "x": np.asarray(r.x).tolist()})
        except Exception as exc:   # a wild random start can hit a numerical error; it is simply not a candidate
            print("start", i, "failed:", type(exc).__name__, exc, flush=True)
            continue
        print(f"[{a.model}/{a.agg}/seed{a.seed}] start {i}: {runs[-1]['start_loss']:.5f} -> {runs[-1]['loss']:.5f}  ({time.time() - t0:.0f}s)", flush=True)
        Path(a.out).write_text(json.dumps({"model": a.model, "agg": a.agg, "objective": a.objective, "null": prob.null / norm, "theta": model.theta, "fixgamma": model.fixgamma, "ordinal": model.ordinal, "runs": runs, "names": model.names, "sample": "coarse" if coarse else "full"}))
    if coarse and getattr(a, "polish", 0) > 0 and runs:
        data_f, subs_f = Data(), load_subs()
        prob_f = Problem(data_f, subs_f, model, a.objective)
        norm_f = data_f.n.sum() if a.objective == "binom" else 1.0
        obj_f = lambda x: prob_f.loss(x) / norm_f
        top = sorted(runs, key=lambda r: r["loss"])[: a.polish]
        polished = []
        for r in top:
            rr = minimize(obj_f, np.array(r["x"]), method="L-BFGS-B", options={"maxiter": a.maxiter, "maxfun": a.maxiter * (len(x0) + 2), "eps": 1e-3})
            polished.append({"start": r["start"], "coarse_loss": r["loss"], "loss": float(rr.fun), "x": rr.x.tolist()})
            print(f"  polish start {r['start']}: coarse {r['loss']:.5f} -> full {rr.fun:.5f}  ({time.time() - t0:.0f}s)", flush=True)
        Path(a.out).write_text(json.dumps({"model": a.model, "agg": a.agg, "objective": a.objective, "null": prob_f.null / norm_f, "theta": model.theta, "fixgamma": model.fixgamma, "ordinal": model.ordinal, "runs": polished, "coarse_runs": runs, "names": model.names, "sample": "coarse+polish"}))


def best_of(paths):
    best = None
    for p in paths:
        d = json.load(open(p))
        for r in d["runs"]:
            if best is None or r["loss"] < best[0]["loss"]:
                best = (r, d)
    return best


def do_boot(a):
    r, d = best_of(a.src)
    data = Data(); subs = load_subs(); model = Model(d["model"], d["agg"], d.get("theta"), d.get("fixgamma", False), d.get("ordinal")); prob = Problem(data, subs, model, d.get("objective", "binom")); norm = data.n.sum() if prob.obj == "binom" else 1.0
    x_best = np.array(r["x"])
    cb = pd.factorize(data.cbsa)[0]
    nc = cb.max() + 1
    rng = np.random.default_rng(a.seed)
    out = []
    for b in range(a.B):
        cnt = np.bincount(rng.integers(0, nc, nc), minlength=nc)[cb]   # resample the metro areas (non-metro block groups form one group)
        w = data.base_w * cnt
        obj = lambda x: prob.loss(x, w) / ((w * data.n).sum() if prob.obj == "binom" else w.sum() / data.base_w.sum())   # wls: the loss is already a weighted mean; keep its scale (resample weights relative to the base ones)
        rr = minimize(obj, x_best, method="L-BFGS-B", options={"maxiter": a.maxiter, "maxfun": a.maxiter * (len(x_best) + 2), "eps": 1e-3})
        out.append({"b": b, "loss": float(rr.fun), "x": rr.x.tolist(), "curves": curves(model, rr.x)})
        Path(a.out).write_text(json.dumps({"model": d["model"], "agg": d["agg"], "best_x": x_best.tolist(), "best_curves": curves(model, x_best), "boot": out}))
        print(f"boot {b} loss {rr.fun:.5f}", flush=True)


def metro_ranking():
    bgt = pd.read_parquet(HERE / "data" / "blockgroups.parquet")
    t = bgt[bgt.cbsa != ""].groupby("cbsa").population.sum().sort_values(ascending=False)
    return list(t.index), t


def do_metros(a):
    """One INDEPENDENT fit per major metro area (official CBSA, ranked by population): the metro's own block groups only, default start + random starts, same model/objective
    as the national fit. If --from is given, the national solution is only evaluated on each metro (never used as a start) to be compared with the metro's own fit afterwards."""
    model = Model(a.model, a.agg)
    x_nat, nat_obj = None, None
    if a.src:
        r, d = best_of(a.src); x_nat = np.array(r["x"]); nat_obj = d.get("objective", "binom")
    codes, pops = metro_ranking()
    names = json.load(open(HERE / "data" / "cbsa_names.json"))
    rows = []
    for code in codes[: a.top]:
        data = Data(cbsa_filter=code)
        if len(data.idx) < 300:
            continue
        sel = np.zeros(data.nbg, bool); sel[data.idx] = True
        subs = load_subs(lambda bg: (bg >= 0) & sel[np.maximum(bg, 0)])
        prob = Problem(data, subs, model, a.objective)
        norm = data.n.sum() if prob.obj == "binom" else 1.0
        obj = lambda x: prob.loss(x) / norm
        rng = np.random.default_rng(a.seed + int(code))
        runs = []
        for i in range(a.starts):
            xs = model.x0() if i == 0 else model.x0() + rng.normal(0, a.spread, size=len(model.x0()))
            try:
                rr = minimize(obj, xs, method="L-BFGS-B", options={"maxiter": a.maxiter, "maxfun": a.maxiter * (len(xs) + 2), "eps": 1e-3})
                runs.append({"start": i, "loss": float(rr.fun), "x": rr.x.tolist()})
            except Exception as exc:
                print(code, "start", i, "failed", type(exc).__name__, flush=True)
        if not runs:
            continue
        best = min(runs, key=lambda q: q["loss"])
        row = {"cbsa": code, "name": names.get(code, code), "population": float(pops[code]), "n_bg": int(len(data.idx)), "null": float(prob.null / norm), "loss_default": float(obj(model.x0())),
               "loss_own_fit": best["loss"], "runs": [{"start": q["start"], "loss": q["loss"]} for q in runs], "x": best["x"], "curves": curves(model, np.array(best["x"]))}
        if x_nat is not None:
            row["loss_national_fit"] = float(obj(x_nat))
        rows.append(row)
        print(f"{names.get(code, code)[:40]:40s} bg {len(data.idx):6d} null {row['null']:.5f} default {row['loss_default']:.5f}" + (f" national {row['loss_national_fit']:.5f}" if x_nat is not None else "") + f" own {best['loss']:.5f}", flush=True)
        Path(a.out).write_text(json.dumps({"model": a.model, "agg": a.agg, "objective": a.objective, "national_x": None if x_nat is None else x_nat.tolist(),
                                            "national_curves": None if x_nat is None else curves(model, x_nat), "metros": rows, "names": model.names}))


def wr2(x, y, w):
    w = np.asarray(w, float); w = w / w.sum(); x = np.asarray(x, float); y = np.asarray(y, float)
    mx, my = (w * x).sum(), (w * y).sum()
    c = (w * (x - mx) * (y - my)).sum(); vx = (w * (x - mx) ** 2).sum(); vy = (w * (y - my) ** 2).sum()
    return float(c * c / (vx * vy)) if vx > 0 and vy > 0 else 0.0


def do_report(a):
    """Linear weighted R2 of the transit share against the score, on ALL block groups of all states, at block-group / tract / county level and over 10-point score bins.
    Weights are the POPULATION of the unit (the worker count as an alternative). Scoring: the default and every fit given with --from."""
    bgt = pd.read_parquet(HERE / "data" / "blockgroups.parquet")
    states = [fs.State(k, bgt) for k in rc.STATE_DIRS if (ROOT / "web" / rc.STATE_DIRS[k] / "meta.json").is_file()]
    pop = bgt.population.fillna(0).to_numpy().astype(float); wk = bgt.acs5_workers.fillna(0).to_numpy().astype(float); pt = bgt.publicTransportCommuters.fillna(0).to_numpy().astype(float)
    geoid = bgt.GEOID.to_numpy()
    ok = (wk > 0) & (pop > 0)
    cands = [("default", sc.DEFAULT, 1.0)]
    for path in a.src or []:
        r, d = best_of([path]); m = Model(d["model"], d["agg"], d.get("theta"), d.get("fixgamma", False), d.get("ordinal")); x = np.array(r["x"])
        cands.append((f"{Path(path).stem} (loss {r['loss']:.5f})", m.params(x), m.gamma(x)))
    out = []
    for name, p, gam in cands:
        num = np.zeros(len(bgt))
        for st in states:
            los, _, _ = core.recompute_cells_fast(st.cst, st.cdist, st.prof, p)
            okc = st.bg >= 0
            num += np.bincount(st.bg[okc], weights=st.pop[okc] * los[okc].astype(np.float64), minlength=len(bgt))
        f = num / np.maximum(pop, 1.0)       # 0-100 population-weighted mean score of the block group
        res = {"name": name, "gamma": gam}
        for wname, w in (("population", pop), ("workers", wk)):
            r_ = {}
            share = np.where(wk > 0, pt / np.maximum(wk, 1), 0)
            r_["block_group"] = wr2(f[ok], share[ok], w[ok])
            for lvl, n_ in (("tract", 11), ("county", 5)):
                g = pd.DataFrame({"k": pd.Series(geoid[ok]).str[:n_].to_numpy(), "fp": (f * pop)[ok], "pop": pop[ok], "pt": pt[ok], "wk": wk[ok]}).groupby("k").sum()
                r_[lvl] = wr2(g.fp / g["pop"], g.pt / g.wk, (g["pop"] if wname == "population" else g.wk))
            bins = pd.cut(f[ok], np.arange(0, 101, 10), include_lowest=True)
            g = pd.DataFrame({"b": bins, "fp": (f * pop)[ok], "pop": pop[ok], "pt": pt[ok], "wk": wk[ok]}).groupby("b", observed=True).sum()
            r_["score_bins_10pt"] = wr2(g.fp / g["pop"], g.pt / g.wk, g["pop"])
            res[wname] = r_
        out.append(res)
        print(json.dumps(res), flush=True)
    Path(a.out).write_text(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--model", default="param")
    ap.add_argument("--agg", default="nested")
    ap.add_argument("--objective", default="binom", help="binom | wls (population-weighted least squares on the share) | prob (the score is P(share > theta))")
    ap.add_argument("--theta", default=None, help="prob objective: transit-share threshold (0.10 = 10%%) or 'free' to fit it")
    ap.add_argument("--grid", default="0.03,0.05,0.075,0.10,0.15,0.20,0.30", help="theta stage: thresholds to profile")
    ap.add_argument("--refine", type=int, default=4, help="theta stage: golden-section steps")
    ap.add_argument("--thresholds", default=None, help="ordinal objective: comma-separated transit-share thresholds, e.g. 0.005,0.01,0.02,0.05,0.10,0.20")
    ap.add_argument("--fixgamma", action="store_true", help="prob objective: the score itself is the probability (gamma = 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--spread", type=float, default=0.8)
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--B", type=int, default=30)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--from", dest="src", nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample", default="full", choices=["full", "coarse"], help="coarse: starts on the ~10% sample, then --polish the best on the full sample")
    ap.add_argument("--polish", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0, help="stop a start that gained <0.5%% after this many iterations (0 = never)")
    a = ap.parse_args()
    {"prep": lambda: prep(), "fit": lambda: do_fit(a), "boot": lambda: do_boot(a), "metros": lambda: do_metros(a), "theta": lambda: do_theta(a), "ordinal_report": lambda: do_ordinal_report(a), "report": lambda: do_report(a)}[a.stage]()


if __name__ == "__main__":
    main()
