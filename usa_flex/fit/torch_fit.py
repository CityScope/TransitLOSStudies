"""Data-driven transit level-of-service curves, fitted on the GPU (PyTorch, CPU fallback) from block-group transit commuting only.

    $TORCH_PYTHON fit/torch_fit.py <experiment> [options]     (see EXPERIMENTS at the bottom)

Model (the same structure as the map's scoring engine, every curve free):
    stop quality       q = M(mode) * S(speed) * H(headway)                M: bus <= tram <= rail = 1, S and H monotone tables (log-linear between knots)
    walking decay      W_mode(d / (1 + k q))                               one monotone table per mode (bus / tram / rail), W(0) = 1; k >= 0 lets people walk further to better service
    cell access        a_c = max over the cell's (station, pattern) items of q * W      (the map's rule; the item lists are Pareto-pruned for it)
    block group        A_b = sum_c pop_c a_c / pop_b
    link               logit mu_b = alpha + beta * log(1 + A_b / eps) [+ controls . delta] [+ metro intercept]
    likelihood         beta-binomial(transit commuters | workers, mu_b, kappa), each block group's per-worker log likelihood weighted by its population
                       (optionally tempered by its metro's population ** -lambda so that the biggest metros do not dominate)
Smoothness: a penalty on the second differences of every log table. Monotonicity and anchors by construction (cumulative softplus steps).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
import sys as _sys; _sys.path.insert(0, str(HERE)); import paths as _P  # noqa: E402
DATA = _P.DATA
TD = DATA / "torch"
RUNS = DATA / "torch_runs"
RUNS.mkdir(parents=True, exist_ok=True)
BG_TABLE = HERE.parents[1] / "cache" / "derived" / "bg_table.parquet"
DEV = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("TORCH_FIT_DEVICE", "") != "cpu" else "cpu")

H_KNOTS = np.array([2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 1440], float)
V_KNOTS = np.array([2, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 120], float)
D_KNOTS = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 2000], float)   # walking only (<= 2 km); beyond: torch_fit3.FAR_KNOTS
MODES = ["bus", "tram", "rail"]


# ------------------------------------------------------------------------------------------------------------------------------------- data
class Data:
    """Items on the device + the block-group table (outcome, weights, controls, groups)."""

    def __init__(self, outcome: str = "transit"):
        t0 = time.time()
        bgt = pd.read_parquet(DATA / "blockgroups.parquet")
        tab = pd.read_parquet(BG_TABLE).set_index("GEOID").reindex(bgt.GEOID)
        c = np.load(TD / "cells.npz"); it = np.load(TD / "items.npz")
        self.nbg_all = len(bgt)
        pop = bgt.population.fillna(0).to_numpy(float)
        wk = bgt.acs5_workers.fillna(0).to_numpy(float)
        if outcome == "transit":
            y = bgt.publicTransportCommuters.to_numpy(float)
        else:   # mode-specific commuters (ACS B08301 at block group level): bus, rail (subway + commuter rail), tram (light rail / streetcar)
            cols = {"bus": ["bus_commuters"], "rail": ["subway_commuters", "commuter_rail_commuters"], "tram": ["light_rail_commuters"]}[outcome]
            y = tab[cols].sum(axis=1, min_count=len(cols)).to_numpy(float)
            wk = tab["workers_b08301"].to_numpy(float)
        ok = (pop > 0) & (wk > 0) & np.isfinite(y)
        self.idx = np.flatnonzero(ok)                       # block groups used (rows of blockgroups.parquet)
        remap = np.full(len(bgt), -1, np.int64); remap[self.idx] = np.arange(len(self.idx))
        self.nb = len(self.idx)
        self.y = torch.tensor(np.minimum(np.maximum(y[ok], 0), wk[ok]), dtype=torch.float32, device=DEV)
        self.n = torch.tensor(wk[ok], dtype=torch.float32, device=DEV)
        self.pop = torch.tensor(pop[ok], dtype=torch.float32, device=DEV)
        cbsa = bgt.cbsa.fillna("").to_numpy()[ok]; geoid = bgt.GEOID.to_numpy()[ok]
        self.cbsa = cbsa; self.geoid = geoid; self.state = bgt.state.to_numpy()[ok]
        # groups for cross-validation / metro weighting: the CBSA, or the county outside metro areas
        grp = np.where(cbsa != "", cbsa, np.array([g[:5] for g in geoid]))
        self.group_names, self.group = np.unique(grp, return_inverse=True)
        gpop = np.bincount(self.group, weights=pop[ok])
        self.group_pop = gpop
        # cells and items (only cells of used block groups)
        cbg = remap[c["bg"]]
        keepc = cbg >= 0
        newc = np.full(len(cbg), -1, np.int64); newc[keepc] = np.arange(keepc.sum())
        icell = newc[it["cell"]]; keepi = icell >= 0
        self.cell_bg = torch.tensor(cbg[keepc], device=DEV)
        self.cell_pop = torch.tensor(c["pop"][keepc], dtype=torch.float32, device=DEV)
        self.ncell = int(keepc.sum())
        self.it_cell = torch.tensor(icell[keepi], device=DEV)
        self.it_mode = torch.tensor(it["mode"][keepi].astype(np.int64), device=DEV)
        h = np.clip(it["h"][keepi], H_KNOTS[0], H_KNOTS[-1]); v = np.clip(it["v"][keepi], V_KNOTS[0], V_KNOTS[-1]); d = np.clip(it["d"][keepi], 0, D_KNOTS[-1])
        self.it_h = torch.tensor(h, dtype=torch.float32, device=DEV)
        self.it_d = torch.tensor(d, dtype=torch.float32, device=DEV)
        self.hi, self.hf = interp_index(np.log(h), np.log(H_KNOTS))
        if "hc" in it.files:   # the station's combined headway (all its patterns): for the 'combined frequency at the stop' variant
            hc = np.clip(it["hc"][keepi], H_KNOTS[0], H_KNOTS[-1])
            self.hci, self.hcf = interp_index(np.log(hc), np.log(H_KNOTS))
        self.vi, self.vf = interp_index(v, V_KNOTS)
        self.di, self.df = interp_index(d, D_KNOTS)
        self.nitems = int(keepi.sum())
        # best headway within 400 m per block group (service classes for the calibration tables), population shares
        self._tab = tab.iloc[self.idx].reset_index()
        self.bgt = bgt.iloc[self.idx].reset_index(drop=True)
        self.service_class = service_classes(self, radius=400.0)
        print(f"data: {self.nb:,} block groups, {self.ncell:,} cells, {self.nitems:,} items on {DEV} ({time.time() - t0:.0f}s)", flush=True)

    def controls(self, names):
        """Standardised control matrix (median-imputed, plus a missing flag where needed)."""
        t = self._tab; b = self.bgt
        area = t.area_km2.clip(lower=1e-3)
        pop = b.population.fillna(0)
        hh = t.households.where(t.households > 0)
        feats = {
            "log_pop_density": np.log1p(pop / area),
            "log_job_density": np.log1p(t.jobs_lodes.fillna(0) / area),
            "log_jobs_5km": np.log1p(t["jobs_5km"]) if "jobs_5km" in t else None,
            "log_income": np.log(t.median_hh_income_b19013.where(t.median_hh_income_b19013 > 0)),
            "poverty_share": (t.poverty_below_50.fillna(0) + t.poverty_50_99.fillna(0)) / t.poverty_universe.where(t.poverty_universe > 0),
            "bachelor_share": t.acs5_educationBachelorsPlus / t.acs5_population25plus.where(t.acs5_population25plus > 0),
            "zero_car_share": t.acs5_vehiclesHouseholds0 / hh,
            "cars_per_household": t.vehicles_aggregate / hh,
            "commute_km": np.log1p(t.commute_km_residents),
            "commute_time": t.commute_time_residents_min,
            "under18_share": t.under18Population / pop.where(pop > 0),
            "over65_share": t.over65Population / pop.where(pop > 0),
            "log_metro_pop": np.log1p(pd.Series(self.group_pop[self.group])),
            "log_dist_cbd": np.log1p(t["dist_cbd_km"]) if "dist_cbd_km" in t else None,
            "renter_share": t.acs5_householdsRenter / hh,
        }
        cols = []; self.last_control_names = []
        for nme in names:
            v = feats[nme]
            if v is None:
                continue
            v = pd.Series(np.asarray(v, float))
            v = v.where(np.isfinite(v))
            miss = v.isna().to_numpy()
            v = v.fillna(v.median())
            z = (v - v.mean()) / (v.std() or 1.0)
            cols.append(z.to_numpy()); self.last_control_names.append(nme)
            if miss.mean() > 0.002:
                cols.append(miss.astype(float) - miss.mean()); self.last_control_names.append(nme + " (missing)")
        X = np.column_stack(cols) if cols else np.zeros((self.nb, 0))
        return torch.tensor(X, dtype=torch.float32, device=DEV)


def interp_index(x, knots):
    """Interval index and fraction of x in a sorted knot vector (clamped)."""
    i = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, len(knots) - 2)
    f = np.clip((x - knots[i]) / (knots[i + 1] - knots[i]), 0, 1)
    return torch.tensor(i, device=DEV), torch.tensor(f, dtype=torch.float32, device=DEV)


def service_classes(d: Data, radius: float) -> np.ndarray:
    """Per block group: class of the best headway of any stop within `radius` m of its cells (0 none, 1 >= 60 min, 2 30-60, 3 15-30, 4 < 15), population-dominant."""
    best = torch.full((d.ncell,), 1e9, device=DEV)
    m = d.it_d <= radius
    best.scatter_reduce_(0, d.it_cell[m], d.it_h[m], reduce="amin", include_self=True)
    cat = torch.where(best > 1e8, 0, torch.where(best >= 60, 1, torch.where(best >= 30, 2, torch.where(best >= 15, 3, 4))))
    shares = torch.zeros((d.nb, 5), device=DEV)
    shares.index_put_((d.cell_bg, cat), d.cell_pop, accumulate=True)
    return shares.argmax(1).cpu().numpy()


# ------------------------------------------------------------------------------------------------------------------------------------ model
def mono_table(u, start_zero=True):
    """Decreasing log table from free parameters: 0, -sp(u0), -sp(u0)-sp(u1), ..."""
    steps = torch.nn.functional.softplus(u)
    return torch.cat([torch.zeros(1, device=u.device), -torch.cumsum(steps, 0)])


class _Gather(torch.autograd.Function):
    """t[i] for a small table t and a huge index vector i. The default backward scatters 30M gradients onto a few table entries with atomics (seconds);
    a weighted bincount does the same reduction in milliseconds."""

    @staticmethod
    def forward(ctx, t, i):
        ctx.save_for_backward(i); ctx.n = t.numel(); ctx.shape = t.shape
        return t.reshape(-1)[i]

    @staticmethod
    def backward(ctx, g):
        (i,) = ctx.saved_tensors
        return torch.bincount(i, weights=g, minlength=ctx.n).to(g.dtype).reshape(ctx.shape), None


def gather(t, i):
    return _Gather.apply(t, i)


def table_eval(logt, i, f):
    return gather(logt, i) * (1 - f) + gather(logt, i + 1) * f


class Model(torch.nn.Module):
    def __init__(self, ncontrol=0, ngroup=0, coupling=False, step_radius=False, fixed=None, mode_only=None, station_headway=False, fixed_radius=None, free_beta=False):
        super().__init__()
        P = torch.nn.Parameter
        nh, nv, nd = len(H_KNOTS), len(V_KNOTS), len(D_KNOTS)
        # initial shapes: headway elasticity ~0.5 above 5 min, speed rising to 30 km/h, walk decay scale 400 / 500 / 800 m
        lh = np.log(H_KNOTS); dh = np.diff(np.maximum(lh, np.log(5.0)))
        self.uh = P(torch.tensor(inv_softplus(0.5 * dh + 1e-3), dtype=torch.float32))
        lv = np.log(V_KNOTS / (V_KNOTS + 10.0)); dv = np.diff(lv)[::-1].copy()   # steps from the top knot down
        self.uv = P(torch.tensor(inv_softplus(dv + 1e-3), dtype=torch.float32))
        dd = np.diff(D_KNOTS)
        self.ud = P(torch.tensor(np.stack([inv_softplus(dd / s + 1e-3) for s in (500.0, 600.0, 900.0)]), dtype=torch.float32))
        self.um = P(torch.tensor([inv_softplus(0.15), inv_softplus(0.15)], dtype=torch.float32))   # log rail - log tram, log tram - log bus
        self.lk = P(torch.tensor([-1.0 if coupling else -30.0], dtype=torch.float32))
        self.coupling = coupling
        self.alpha = P(torch.tensor([-4.0])); self.leps = P(torch.tensor([math.log(0.02)]))
        # beta = 1 fixes the scale: raising every curve to a power c and dividing beta by c predicts almost the same, so the curves' steepness is only defined
        # once beta is fixed. With beta = 1 the transit odds grow in proportion to access (for access >> eps) and eps is the access that doubles the odds.
        if free_beta:
            self.lbeta = P(torch.tensor([0.0]))
        else:
            self.register_buffer("lbeta", torch.tensor([0.0]))
        self.lkappa = P(torch.tensor([math.log(20.0)]))
        self.delta = P(torch.zeros(ncontrol))
        self.galpha = P(torch.zeros(ngroup)) if ngroup else None
        self.step_radius = step_radius
        if step_radius:   # walking decay as a soft step at a fitted radius per mode (the 400 / 800 m rule of thumb, radius estimated)
            if fixed_radius is not None:   # the rule of thumb itself: radius per mode fixed (bus, tram, rail), only the edge width is fitted
                self.register_buffer("lrad", torch.tensor([math.log(r) for r in fixed_radius]))
            else:
                self.lrad = P(torch.tensor([math.log(400.0), math.log(500.0), math.log(800.0)]))
            self.lwidth = P(torch.tensor([math.log(60.0)] * 3))
        self.fixed = fixed or {}
        self.mode_only = mode_only
        self.station_headway = station_headway

    def tables(self):
        lh = mono_table(self.uh)                                  # log H at the headway knots, H(2 min) = 1
        lv = torch.flip(mono_table(self.uv), [0])                 # log S at the speed knots, S(120) = 1
        ld = torch.stack([mono_table(self.ud[m]) for m in range(3)])
        s1, s2 = torch.nn.functional.softplus(self.um)
        lm = torch.stack([-(s1 + s2), -s1, torch.zeros((), device=s1.device)])   # bus <= tram <= rail = 1
        for k, val in self.fixed.items():
            if k == "lh":
                lh = val
            elif k == "ld":
                ld = val
        return lh, lv, ld, lm

    def access(self, d: Data, X=None):
        lh, lv, ld, lm = self.tables()
        hi, hf = (d.hci, d.hcf) if self.station_headway else (d.hi, d.hf)
        lq = gather(lm, d.it_mode) + table_eval(lv, d.vi, d.vf) + table_eval(lh, hi, hf)
        q = torch.exp(lq)
        k = torch.exp(self.lk) if self.coupling else torch.zeros((), device=q.device)
        if self.step_radius:
            rad = gather(torch.exp(self.lrad), d.it_mode) * (1 + k * q); wid = gather(torch.exp(self.lwidth), d.it_mode)
            lw = torch.nn.functional.logsigmoid((rad - d.it_d) / wid)
        elif self.coupling:
            deff = d.it_d / (1 + k * q)
            i = torch.clamp(torch.searchsorted(torch.tensor(D_KNOTS, dtype=torch.float32, device=q.device), deff, right=True) - 1, 0, len(D_KNOTS) - 2)
            kn = torch.tensor(D_KNOTS, dtype=torch.float32, device=q.device)
            f = torch.clamp((deff - kn[i]) / (kn[i + 1] - kn[i]), 0, 1)
            flat = d.it_mode * len(D_KNOTS) + i
            lw = gather(ld, flat) * (1 - f) + gather(ld, flat + 1) * f
        else:
            flat = d.it_mode * len(D_KNOTS) + d.di
            lw = gather(ld, flat) * (1 - d.df) + gather(ld, flat + 1) * d.df
        a = torch.exp(lq + lw)
        if self.mode_only is not None:
            a = torch.where(d.it_mode == self.mode_only, a, torch.zeros_like(a))
        cell = torch.zeros(d.ncell, device=a.device).scatter_reduce(0, d.it_cell, a, reduce="amax", include_self=True)
        A = torch.zeros(d.nb, device=a.device).index_add(0, d.cell_bg, cell * d.cell_pop) / torch.clamp(d.pop, min=1.0)
        return A

    def eta(self, A, X=None, group=None):
        e = self.alpha + torch.exp(self.lbeta) * torch.log1p(A / torch.exp(self.leps))
        if X is not None and X.shape[1]:
            e = e + X @ self.delta
        if self.galpha is not None and group is not None:
            e = e + self.galpha[group]
        return e

    def smooth_penalty(self):
        lh, lv, ld, lm = self.tables()
        d2 = lambda t: ((t[2:] - 2 * t[1:-1] + t[:-2]) ** 2).sum()  # noqa: E731
        return d2(lh) + d2(lv) + sum(d2(ld[m]) for m in range(3))


def inv_softplus(y):
    y = np.maximum(np.asarray(y, float), 1e-6)
    return np.log(np.expm1(y))


def bb_loglik(y, n, mu, kappa):
    a = kappa * mu; b = kappa * (1 - mu)
    return (torch.lgamma(y + a) + torch.lgamma(n - y + b) - torch.lgamma(n + kappa) - torch.lgamma(a) - torch.lgamma(b) + torch.lgamma(kappa)
            + torch.lgamma(n + 1) - torch.lgamma(y + 1) - torch.lgamma(n - y + 1))


def weights(d: Data, lam: float):
    """Population weight per block group, tempered by its group's (metro's) population ** -lam, normalised to sum 1."""
    w = d.pop.clone()
    if lam:
        gp = torch.tensor(d.group_pop[d.group], dtype=torch.float32, device=DEV)
        w = w * gp ** (-lam)
    return w / w.sum()


def loss_terms(model, d, w, X=None, group=None, A=None):
    A = model.access(d) if A is None else A
    mu = torch.sigmoid(model.eta(A, X, group)).clamp(1e-7, 1 - 1e-7)
    ll = bb_loglik(d.y, d.n, mu, torch.exp(model.lkappa))
    return -(w * ll / d.n).sum(), A, mu


def fit(model, d, w, X=None, group=None, steps=3000, lr=0.03, smooth=1.0, lbfgs=100, log_every=500, ridge_group=1.0, prior=None, prior_w=0.0, quiet=False):
    """Adam (a third of the steps at lr, the rest at lr / 3) then L-BFGS, on the full batch. `prior`: log tables to pull towards (literature, or the US curves for
    the metro fits). 3000 steps: the curves are still moving after 1000 although the loss barely changes (checked: two random starts agree to 3 decimals after ~2500)."""
    model.to(DEV)
    params = [p for p in model.parameters() if p.requires_grad]

    def objective():
        L, A, mu = loss_terms(model, d, w, X, group)
        pen = smooth * 1e-6 * model.smooth_penalty()   # the loss is a per-worker log likelihood (~0.005): penalties live on that scale
        if model.galpha is not None:
            pen = pen + ridge_group * 1e-8 * (model.galpha ** 2).sum()
        if prior is not None and prior_w:
            lh, lv, ld, lm = model.tables()
            cur = {"lh": lh, "lv": lv, "ld": ld, "lm": lm}
            pen = pen + prior_w * 1e-6 * sum(((cur[k] - v) ** 2).sum() for k, v in prior.items())
        return L + pen, L

    opt = torch.optim.Adam(params, lr=lr)
    t0 = time.time()
    for s in range(steps):
        if s == steps // 3:
            for g_ in opt.param_groups:
                g_["lr"] = lr / 3
        opt.zero_grad(); tot, L = objective(); tot.backward(); opt.step()
        if not quiet and (s % log_every == 0 or s == steps - 1):
            print(f"  adam {s:4d}  loss {L.item():.6f}  ({time.time() - t0:.0f}s)", flush=True)
    if lbfgs:
        opt2 = torch.optim.LBFGS(params, lr=0.5, max_iter=lbfgs, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad(); tot, _ = objective(); tot.backward(); return tot
        opt2.step(closure)
    with torch.no_grad():
        _, L = objective()
    if not quiet:
        print(f"  final loss {L.item():.6f} ({time.time() - t0:.0f}s)", flush=True)
    return L.item()


def null_loss(d, w, X=None, group=None):
    """Intercept (+ controls, + groups) only, no transit access: the reference for the skill D2."""
    m = Model(X.shape[1] if X is not None else 0, len(d.group_names) if group is not None else 0).to(DEV)
    for p in (m.uh, m.uv, m.ud, m.um, m.lk, m.lbeta, m.leps):
        p.requires_grad_(False)
    with torch.no_grad():
        m.lbeta.fill_(-30.0)
    A0 = torch.zeros(d.nb, device=DEV)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.LBFGS(params, lr=0.5, max_iter=200, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(); L, _, _ = loss_terms(m, d, w, X, group, A=A0)
        if m.galpha is not None:
            L = L + 1e-8 * (m.galpha ** 2).sum()
        L.backward(); return L
    for _ in range(3):
        opt.step(closure)
    with torch.no_grad():
        L, _, _ = loss_terms(m, d, w, X, group, A=A0)
    return L.item(), m


# ------------------------------------------------------------------------------------------------------------------------------ reporting
def curves(model):
    with torch.no_grad():
        lh, lv, ld, lm = [t.detach().cpu().numpy() for t in model.tables()]
    out = {"headway": {"x": H_KNOTS.tolist(), "y": np.exp(lh).tolist()}, "speed": {"x": V_KNOTS.tolist(), "y": np.exp(lv).tolist()},
           "mode": dict(zip(MODES, np.exp(lm).tolist())),
           "walk": {MODES[m]: {"x": D_KNOTS.tolist(), "y": np.exp(ld[m]).tolist()} for m in range(3)},
           "k": float(torch.exp(model.lk).item()) if model.coupling else 0.0,
           "alpha": float(model.alpha.item()), "beta": float(torch.exp(model.lbeta).item()), "eps": float(torch.exp(model.leps).item()),
           "kappa": float(torch.exp(model.lkappa).item())}
    if model.step_radius:
        out["radius"] = dict(zip(MODES, torch.exp(model.lrad).detach().cpu().numpy().tolist()))
        out["radius_width"] = dict(zip(MODES, torch.exp(model.lwidth).detach().cpu().numpy().tolist()))
    return out


def per_group_skill(d, w, model, X, group, mask=None, nullm=None, top=None):
    """Skill (1 - loss / null loss) inside each of the `top` biggest metros (population weights inside the metro)."""
    with torch.no_grad():
        A = model.access(d)
        mu = torch.sigmoid(model.eta(A, X, group)).clamp(1e-7, 1 - 1e-7)
        ll = (bb_loglik(d.y, d.n, mu, torch.exp(model.lkappa)) / d.n).cpu().numpy()
    pop = d.pop.cpu().numpy(); y = d.y.cpu().numpy(); n = d.n.cpu().numpy()
    out = {}
    order = np.argsort(-d.group_pop)
    for g in order[: (top or 50)]:
        m = d.group == g
        if mask is not None:
            m &= mask
        if m.sum() < 30:
            continue
        # null inside the metro: its own beta-binomial with constant mean
        mu0 = y[m].sum() / n[m].sum()
        kap = float(torch.exp(model.lkappa).item())
        ll0 = (bb_loglik(torch.tensor(y[m]), torch.tensor(n[m]), torch.full((m.sum(),), float(np.clip(mu0, 1e-6, 1 - 1e-6))), torch.tensor(kap)) / torch.tensor(n[m])).numpy()
        ww = pop[m] / pop[m].sum()
        out[str(d.group_names[g])] = {"skill": float(1 - (ww * ll[m]).sum() / (ww * ll0).sum()), "pop": float(pop[m].sum())}
    return out


def calibration(d, mu):
    """Population-weighted observed vs predicted transit share by service class (best headway within 400 m)."""
    mu = mu.detach().cpu().numpy(); y = d.y.cpu().numpy(); n = d.n.cpu().numpy(); pop = d.pop.cpu().numpy()
    names = ["no stop within 400 m", "best headway 60+ min", "30-60 min", "15-30 min", "< 15 min"]
    out = {}
    for k, nm in enumerate(names):
        m = d.service_class == k
        if not m.any():
            continue
        w = pop[m] / pop[m].sum()
        out[nm] = {"pop_share": float(pop[m].sum() / pop.sum()), "observed": float((w * y[m] / n[m]).sum()), "predicted": float((w * mu[m]).sum()),
                   "any_observed": float((w * (y[m] > 0)).sum())}
    return out


def scenario_effects(model, d, X=None):
    """Predicted transit share for a block group whose whole population lives at a given distance from a single stop of a given mode / headway (speed 20 km/h,
    controls at their mean, no metro intercept): what the fitted curves mean in commuting terms."""
    lh, lv, ld, lm = model.tables()
    out = {}
    with torch.no_grad():
        k = torch.exp(model.lk).item() if model.coupling else 0.0
        for mi, mn in enumerate(MODES):
            for h in (5, 10, 15, 30, 60, 120, 240):
                for dist in (100, 400, 800):
                    lq = lm[mi] + float(np.interp(20.0, V_KNOTS, lv.cpu().numpy())) + float(np.interp(np.log(h), np.log(H_KNOTS), lh.cpu().numpy()))
                    q = math.exp(lq)
                    if model.step_radius:
                        rad = math.exp(model.lrad[mi].item()) * (1 + k * q); wid = math.exp(model.lwidth[mi].item())
                        w = 1 / (1 + math.exp(-(rad - dist) / wid))
                    else:
                        w = math.exp(float(np.interp(dist / (1 + k * q), D_KNOTS, ld[mi].cpu().numpy())))
                    A = torch.tensor([q * w], device=DEV)
                    out[f"{mn}|{h}|{dist}"] = float(torch.sigmoid(model.eta(A)).item())
        out["none"] = float(torch.sigmoid(model.eta(torch.zeros(1, device=DEV))).item())
    return out


def save(name, payload):
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{name}.json").write_text(json.dumps(payload, indent=1, default=float))


# ---------------------------------------------------------------------------------------------------------------------------- experiments
CONTROL_SETS = {
    "none": [],
    "exogenous": ["log_pop_density", "log_job_density", "log_jobs_5km", "log_income", "poverty_share", "bachelor_share", "under18_share", "over65_share",
                  "log_metro_pop", "log_dist_cbd", "commute_km", "renter_share"],
    "exogenous+cars": ["log_pop_density", "log_job_density", "log_jobs_5km", "log_income", "poverty_share", "bachelor_share", "under18_share", "over65_share",
                       "log_metro_pop", "log_dist_cbd", "commute_km", "renter_share", "zero_car_share", "cars_per_household"],
}


def cv_folds(d, k=5, seed=0):
    rng = np.random.default_rng(seed)
    gf = rng.integers(0, k, len(d.group_names))
    # the 10 biggest metros spread over the folds (NYC, LA, Chicago... never in the same fold)
    order = np.argsort(-d.group_pop)
    for i, g in enumerate(order[:10]):
        gf[g] = i % k
    return gf[d.group]


def run_variant(d, name, controls="none", lam=0.0, metro=False, coupling=False, step_radius=False, smooth=1.0, cv=True, steps=3000, mode_only=None, extra=None,
                station_headway=False, prior=None, prior_w=0.0, wmask=None, seed=0, init_noise=0.0, fixed_radius=None):
    X = d.controls(CONTROL_SETS[controls]) if controls != "none" else None
    group = torch.tensor(d.group, device=DEV) if metro else None
    w = weights(d, lam)
    if wmask is not None:   # fit on a subset of block groups (leave-one-metro-out, split halves); skills below are on that subset too
        w = w * torch.tensor(wmask, dtype=torch.float32, device=DEV); w = w / w.sum()
    ngroup = len(d.group_names) if metro else 0
    torch.manual_seed(seed)
    res = {"name": name, "station_headway": station_headway, "prior_w": prior_w, "seed": seed, "controls": controls, "lam": lam, "metro_intercepts": metro, "coupling": coupling, "step_radius": step_radius, "smooth": smooth}
    t0 = time.time()
    mk = lambda: Model(X.shape[1] if X is not None else 0, ngroup, coupling, step_radius, mode_only=mode_only, station_headway=station_headway,
                       fixed_radius=fixed_radius)  # noqa: E731
    model = mk()
    if init_noise:
        with torch.no_grad():
            for prm in (model.uh, model.uv, model.ud, model.um):
                prm.add_(init_noise * torch.randn_like(prm))
    L = fit(model, d, w, X, group, steps=steps, smooth=smooth, prior=prior, prior_w=prior_w)
    L0, _ = null_loss(d, w, X, group)
    Lnull, _ = null_loss(d, w)
    res.update(loss=L, null_with_controls=L0, null=Lnull, skill=1 - L / Lnull, skill_beyond_controls=1 - L / L0, curves=curves(model))
    with torch.no_grad():
        _, A, mu = loss_terms(model, d, w, X, group)
    res["calibration"] = calibration(d, mu)
    res["scenarios"] = scenario_effects(model, d)
    res["metro_skill"] = per_group_skill(d, weights(d, 0.0), model, X, group, top=40)
    if X is not None:
        res["delta"] = model.delta.detach().cpu().numpy().tolist(); res["control_names"] = CONTROL_SETS[controls]
    if cv:   # metro-grouped 5-fold cross-validation: held-out skill
        folds = cv_folds(d)
        if wmask is not None:
            folds = np.where(np.asarray(wmask) > 0, folds, -1)
        ho = np.zeros(d.nb); ho0 = np.zeros(d.nb)
        for f in range(5):
            tr = torch.tensor(folds != f, device=DEV, dtype=torch.float32)
            wtr = w * tr; wtr = wtr / wtr.sum()
            m = mk()
            m.load_state_dict(model.state_dict())
            fit(m, d, wtr, X, group, steps=800, lr=0.01, smooth=smooth, lbfgs=60, quiet=True, prior=prior, prior_w=prior_w)
            with torch.no_grad():
                A = m.access(d)
                eta = m.eta(A, X, None)            # held-out metros have no intercept of their own
                mu = torch.sigmoid(eta).clamp(1e-7, 1 - 1e-7)
                ho[folds == f] = (bb_loglik(d.y, d.n, mu, torch.exp(m.lkappa)) / d.n).cpu().numpy()[folds == f]
            # null of the training folds, applied to the held-out one
            _, mn = null_loss(d, wtr, X)
            with torch.no_grad():
                mu0 = torch.sigmoid(mn.eta(torch.zeros(d.nb, device=DEV), X)).clamp(1e-7, 1 - 1e-7)
                ho0[folds == f] = (bb_loglik(d.y, d.n, mu0, torch.exp(mn.lkappa)) / d.n).cpu().numpy()[folds == f]
        wn = w.cpu().numpy()
        res["cv_skill"] = float(1 - (wn * ho).sum() / (wn * ho0).sum())
        # cross-validated skill inside the biggest metros (equal weight per metro)
        per = []
        for g in np.argsort(-d.group_pop)[:30]:
            m_ = d.group == g
            if (wn[m_] * ho0[m_]).sum() != 0:
                per.append(1 - (wn[m_] * ho[m_]).sum() / (wn[m_] * ho0[m_]).sum())
        res["cv_skill_top30_metros_median"] = float(np.median(per))
        res["cv_skill_top30_metros"] = [float(x) for x in per]
    res["seconds"] = time.time() - t0
    if extra:
        res.update(extra)
    save(name, res)
    torch.save(model.state_dict(), RUNS / f"{name}.pt")
    print(f"[{name}] skill {res['skill']:.4f}" + (f" cv {res['cv_skill']:.4f} top30-median {res['cv_skill_top30_metros_median']:.4f}" if cv else "")
          + f" ({res['seconds']:.0f}s)", flush=True)
    return model, res


def subset(d: Data, bg_mask: np.ndarray):
    """A Data restricted to some block groups (cells and items follow)."""
    import copy
    s = copy.copy(d)
    keep_bg = np.flatnonzero(bg_mask)
    remap = torch.full((d.nb,), -1, dtype=torch.long, device=DEV); remap[torch.tensor(keep_bg, device=DEV)] = torch.arange(len(keep_bg), device=DEV)
    cb = remap[d.cell_bg]; kc = cb >= 0
    cremap = torch.full((d.ncell,), -1, dtype=torch.long, device=DEV); cremap[kc] = torch.arange(int(kc.sum()), device=DEV)
    ic = cremap[d.it_cell]; ki = ic >= 0
    s.nb = len(keep_bg); s.ncell = int(kc.sum()); s.nitems = int(ki.sum())
    s.cell_bg = cb[kc]; s.cell_pop = d.cell_pop[kc]; s.it_cell = ic[ki]
    for a in ("it_mode", "it_h", "it_d", "hi", "hf", "vi", "vf", "di", "df", "hci", "hcf"):
        if hasattr(d, a):
            setattr(s, a, getattr(d, a)[ki])
    for a in ("y", "n", "pop"):
        setattr(s, a, getattr(d, a)[torch.tensor(keep_bg, device=DEV)])
    s.idx = d.idx[keep_bg]; s.cbsa = d.cbsa[keep_bg]; s.geoid = d.geoid[keep_bg]; s.state = d.state[keep_bg]
    s.group = d.group[keep_bg]; s.service_class = d.service_class[keep_bg]
    s._tab = d._tab.iloc[keep_bg].reset_index(drop=True); s.bgt = d.bgt.iloc[keep_bg].reset_index(drop=True)
    return s


def metro_fits(d: Data, base_name="g1_lam05", top=25, pool_w=(0.0, 30.0)):
    """Goal 4: every big metro on its own. For each: (a) the US curves with the metro's own link (alpha, beta, eps, kappa); (b) its own curves pulled towards the US
    ones with weight pool_w; tract-grouped 5-fold cross-validation inside the metro says whether its own curves predict its held-out tracts better."""
    base_sd = torch.load(RUNS / f"{base_name}.pt", map_location=DEV)
    base = Model(coupling=False); base.load_state_dict(base_sd); base.to(DEV)
    with torch.no_grad():
        lh, lv, ld, lm = base.tables()
    prior = {"lh": lh.detach(), "lv": lv.detach(), "ld": ld.detach(), "lm": lm.detach()}
    out = {}
    names = d.group_names
    order = [g for g in np.argsort(-d.group_pop) if names[g].isdigit() and len(names[g]) == 5 and d.cbsa[d.group == g][0] != ""][:top]
    for g in order:
        cb = names[g]
        dm = subset(d, d.group == g)
        w = dm.pop / dm.pop.sum()
        tract = np.array([x[:11] for x in dm.geoid])
        _, tid = np.unique(tract, return_inverse=True)
        folds = np.random.default_rng(1).integers(0, 5, tid.max() + 1)[tid]
        res = {"cbsa": cb, "pop": float(dm.pop.sum().item()), "transit_share": float((dm.y.sum() / dm.n.sum()).item()), "variants": {}}
        for vname, pw, free_tables in [("us_curves", None, False)] + [(f"own_pool{int(p_)}", p_, True) for p_ in pool_w]:
            def make():
                m = Model(coupling=False); m.load_state_dict(base_sd); m.to(DEV)
                if not free_tables:
                    for prm in (m.uh, m.uv, m.ud, m.um, m.lk):
                        prm.requires_grad_(False)
                return m
            m = make()
            fit(m, dm, w, steps=900, lr=0.01, lbfgs=60, quiet=True, prior=prior if free_tables else None, prior_w=pw or 0.0)
            ho = np.zeros(dm.nb); ho0 = np.zeros(dm.nb)
            for f in range(5):
                tr = torch.tensor(folds != f, dtype=torch.float32, device=DEV); wtr = w * tr; wtr = wtr / wtr.sum()
                mf = make(); mf.load_state_dict(m.state_dict())
                fit(mf, dm, wtr, steps=400, lr=0.01, lbfgs=40, quiet=True, prior=prior if free_tables else None, prior_w=pw or 0.0)
                with torch.no_grad():
                    mu = torch.sigmoid(mf.eta(mf.access(dm))).clamp(1e-7, 1 - 1e-7)
                    ho[folds == f] = (bb_loglik(dm.y, dm.n, mu, torch.exp(mf.lkappa)) / dm.n).cpu().numpy()[folds == f]
                _, mn = null_loss(dm, wtr)
                with torch.no_grad():
                    mu0 = torch.sigmoid(mn.eta(torch.zeros(dm.nb, device=DEV))).clamp(1e-7, 1 - 1e-7)
                    ho0[folds == f] = (bb_loglik(dm.y, dm.n, mu0, torch.exp(mn.lkappa)) / dm.n).cpu().numpy()[folds == f]
            wn = w.cpu().numpy()
            res["variants"][vname] = {"cv_skill": float(1 - (wn * ho).sum() / (wn * ho0).sum()), "curves": curves(m), "scenarios": scenario_effects(m, dm)}
        out[cb] = res
        print(f"[metro {cb}] " + " ".join(f"{k}={v['cv_skill']:.4f}" for k, v in res["variants"].items()), flush=True)
        save("g4_metros", out)
        del dm; torch.cuda.empty_cache()
    return out


def bootstrap(d, name, B=20, controls="none", lam=0.0, metro=False, coupling=False, step_radius=False, smooth=1.0, base=None, mode_only=None):
    """Bayesian bootstrap over metros/counties: every group gets an Exp(1) weight; the model is re-fitted from the main fit."""
    X = d.controls(CONTROL_SETS[controls]) if controls != "none" else None
    group = torch.tensor(d.group, device=DEV) if metro else None
    w0 = weights(d, lam)
    ngroup = len(d.group_names) if metro else 0
    if base is None:
        base = Model(X.shape[1] if X is not None else 0, ngroup, coupling, step_radius, mode_only=mode_only)
        fit(base, d, w0, X, group, smooth=smooth)
    reps = []
    rng = np.random.default_rng(123)
    for b in range(B):
        gw = torch.tensor(rng.exponential(1.0, len(d.group_names))[d.group], dtype=torch.float32, device=DEV)
        w = w0 * gw; w = w / w.sum()
        m = Model(X.shape[1] if X is not None else 0, ngroup, coupling, step_radius, mode_only=mode_only)
        m.load_state_dict(base.state_dict())
        fit(m, d, w, X, group, steps=900, lr=0.01, smooth=smooth, lbfgs=60, quiet=True)
        reps.append({"curves": curves(m), "scenarios": scenario_effects(m, d)})
        print(f"  boot {b}", flush=True)
    save(name, {"name": name, "B": B, "reps": reps, "main": {"curves": curves(base), "scenarios": scenario_effects(base, d)}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--B", type=int, default=20)
    a = ap.parse_args()
    torch.manual_seed(0)
    e = a.experiment
    if e == "base":          # goal 1: curves from the data alone, population weights, different spatial balance and smoothness
        d = Data()
        run_variant(d, "g1_lam05", lam=0.5, steps=a.steps)
        run_variant(d, "g1_pop", steps=a.steps)
        run_variant(d, "g1_lam05_coupling", lam=0.5, coupling=True, steps=a.steps)
        run_variant(d, "g1_lam05_smooth10", lam=0.5, smooth=10.0, steps=a.steps)
        run_variant(d, "g1_lam05_metroFE", lam=0.5, metro=True, steps=a.steps)
    elif e == "controls":    # goal 2
        d = Data()
        for cs in ("exogenous", "exogenous+cars"):
            run_variant(d, f"g2_{cs}", controls=cs, lam=0.5, steps=a.steps)
        run_variant(d, "g2_exogenous_metroFE", controls="exogenous", lam=0.5, metro=True, steps=a.steps)
    elif e == "radius":      # goal 3: soft-step radius per mode (the 400 / 800 m rule) vs free decay; mode-specific ridership
        d = Data()
        run_variant(d, "g3_step_radius", lam=0.5, step_radius=True, coupling=False, steps=a.steps)
        run_variant(d, "g3_step_fixed_400_400_800", lam=0.5, step_radius=True, coupling=False, steps=a.steps, fixed_radius=(400.0, 400.0, 800.0))
        for mo, mi in (("bus", 0), ("rail", 2), ("tram", 1)):
            dm = Data(outcome=mo)
            run_variant(dm, f"g3_modeout_{mo}", lam=0.5, steps=a.steps, mode_only=mi, controls="exogenous")
            run_variant(dm, f"g3_modeout_{mo}_step", lam=0.5, steps=a.steps, mode_only=mi, controls="exogenous", step_radius=True, coupling=False)
            del dm; torch.cuda.empty_cache()
    elif e == "boot_g3":      # goal 3: uncertainty of the walking decay of each mode against its own riders
        for mo, mi in (("bus", 0), ("rail", 2), ("tram", 1)):
            dm = Data(outcome=mo)
            bootstrap(dm, f"boot_g3_{mo}", B=a.B, lam=0.5, controls="exogenous", mode_only=mi)
            del dm; torch.cuda.empty_cache()
    elif e.startswith("boot"):
        d = Data()
        cfg = {"boot_g1": dict(lam=0.5), "boot_g2": dict(lam=0.5, controls="exogenous")}[e]
        bootstrap(d, e, B=a.B, **cfg)
    elif e == "stability":   # goal 1: is the solution unique and not driven by one metro?
        d = Data()
        for sd in (1, 2):
            run_variant(d, f"g1s_seed{sd}", lam=0.5, seed=sd, init_noise=0.7, cv=False, steps=a.steps)
        nyc = d.cbsa == "35620"
        run_variant(d, "g1s_no_nyc", lam=0.5, wmask=(~nyc).astype(float), steps=a.steps)
        half = (np.random.default_rng(5).integers(0, 2, len(d.group_names))[d.group] == 0)
        run_variant(d, "g1s_half_a", lam=0.5, wmask=half.astype(float), cv=False, steps=a.steps)
        run_variant(d, "g1s_half_b", lam=0.5, wmask=(~half).astype(float), cv=False, steps=a.steps)
        run_variant(d, "g1s_pop_smooth10", lam=0.0, smooth=10.0, cv=False, steps=a.steps)
        run_variant(d, "g1s_lam1", lam=1.0, steps=a.steps)
    elif e == "variants":    # goal 1: combined frequency at the stop; one literature-limited fit
        d = Data()
        run_variant(d, "g1v_station_headway", lam=0.5, station_headway=True, steps=a.steps)
        lit = json.loads((TD / "literature_tables.json").read_text())
        prior = {k: torch.tensor(v, dtype=torch.float32, device=DEV) for k, v in lit.items()}
        run_variant(d, "g1v_literature_prior", lam=0.5, prior=prior, prior_w=30.0, steps=a.steps)
    elif e == "metros":      # goal 4
        d = Data()
        metro_fits(d, base_name="g1_lam05")
    else:
        raise SystemExit("unknown experiment " + e)


if __name__ == "__main__":
    main()
