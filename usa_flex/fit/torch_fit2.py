"""Second round of the GPU calibration (see torch_fit.py for the base model).

    $TORCH_PYTHON fit/torch_fit2.py <experiment>

New here:
  * the headway curve is free from 1 min (no saturation knot) and normalised at 10 min, so the data decide whether very short headways add anything;
  * destination access: transit commuting needs service at the workplace too. Every block group gets the job-weighted access of the block groups its residents
    work in (LODES flows up to 80 km, fit/data/torch/flows.npz):  logit mu = alpha + log(1 + A/eps) + theta * log(1 + D/eps_d) [+ controls] [+ metro terms];
  * control sets built from a causal reading: structural variables (density, job accessibility, distance to the centre, metro size, income, poverty, education, age),
    and the mediators that transit itself changes (car ownership, commute time, commute speed) only as a sensitivity check. A small ridge on the control
    coefficients handles the collinear density block;
  * one joint model for the biggest metros: every metro gets its own deviation of the headway, walking and mode curves (and its own intercept), pulled to the
    national curves by a penalty chosen on held-out census tracts.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch_fit as tf  # noqa: E402
from torch_fit import DEV, MODES, gather, bb_loglik, interp_index  # noqa: E402

H_KNOTS = np.array([1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 1440], float)
H_NORM = int(np.flatnonzero(H_KNOTS == 10)[0])
D_KNOTS = tf.D_KNOTS
V_KNOTS = tf.V_KNOTS
RUNS = tf.RUNS
CONTROLS = {
    "none": [],
    "structural": ["log_pop_density", "log_job_density", "log_jobs_5km", "log_dist_cbd", "log_metro_pop", "log_income", "poverty_share", "bachelor_share",
                   "under18_share", "over65_share"],
    "structural+commute_km": ["log_pop_density", "log_job_density", "log_jobs_5km", "log_dist_cbd", "log_metro_pop", "log_income", "poverty_share",
                              "bachelor_share", "under18_share", "over65_share", "commute_km"],
    "structural+cars": ["log_pop_density", "log_job_density", "log_jobs_5km", "log_dist_cbd", "log_metro_pop", "log_income", "poverty_share", "bachelor_share",
                        "under18_share", "over65_share", "zero_car_share", "cars_per_household"],
    "structural+cars+time+speed": ["log_pop_density", "log_job_density", "log_jobs_5km", "log_dist_cbd", "log_metro_pop", "log_income", "poverty_share",
                                   "bachelor_share", "under18_share", "over65_share", "zero_car_share", "cars_per_household", "commute_time", "commute_speed"],
}


class Data2:
    """All cells (residents or jobs) and items of every block group; origin access over the block groups used in the likelihood, destination access over all."""

    def __init__(self, top_metros=40):
        t0 = time.time()
        base = tf.Data.__new__(tf.Data)
        bgt = pd.read_parquet(tf.DATA / "blockgroups.parquet")
        tab = pd.read_parquet(tf.BG_TABLE).set_index("GEOID").reindex(bgt.GEOID)
        c = np.load(tf.TD / "cells.npz"); it = np.load(tf.TD / "items.npz")
        pop = bgt.population.fillna(0).to_numpy(float); wk = bgt.acs5_workers.fillna(0).to_numpy(float); y = bgt.publicTransportCommuters.to_numpy(float)
        ok = (pop > 0) & (wk > 0) & np.isfinite(y)
        self.idx = np.flatnonzero(ok); self.nb = len(self.idx); self.nbg_all = len(bgt)
        remap = np.full(len(bgt), -1, np.int64); remap[self.idx] = np.arange(self.nb)
        self.y = torch.tensor(np.minimum(np.maximum(y[ok], 0), wk[ok]), dtype=torch.float32, device=DEV)
        self.n = torch.tensor(wk[ok], dtype=torch.float32, device=DEV)
        self.pop = torch.tensor(pop[ok], dtype=torch.float32, device=DEV)
        cbsa_all = bgt.cbsa.fillna("").to_numpy(); geo_all = bgt.GEOID.to_numpy()
        self.cbsa = cbsa_all[ok]; self.geoid = geo_all[ok]
        grp_all = np.where(cbsa_all != "", cbsa_all, np.array([g[:5] for g in geo_all]))
        names, g_all = np.unique(grp_all, return_inverse=True)
        self.group_names = names; self.group = g_all[ok]
        self.group_pop = np.bincount(self.group, weights=pop[ok], minlength=len(names))
        # metro index for the joint metro model: 1..K for the K biggest CBSAs, 0 elsewhere
        cb_pop = pd.Series(pop).groupby(cbsa_all).sum().drop("", errors="ignore").sort_values(ascending=False)
        self.metros = list(cb_pop.index[:top_metros])
        midx = {m: i + 1 for i, m in enumerate(self.metros)}
        metro_all = np.array([midx.get(x, 0) for x in cbsa_all])
        self.metro = metro_all[ok]
        # cells
        cbg = c["bg"].astype(np.int64)
        orig = remap[cbg]
        self.cell_orig = torch.tensor(np.where(orig >= 0, orig, self.nb), device=DEV)          # unused block groups -> a dummy bucket
        self.cell_pop = torch.tensor(np.where(orig >= 0, c["pop"], 0), dtype=torch.float32, device=DEV)
        self.cell_bg_all = torch.tensor(cbg, device=DEV)
        jobs = c["jobs"].astype(np.float32) if "jobs" in c.files else np.zeros(len(cbg), np.float32)
        self.cell_jobs = torch.tensor(jobs, dtype=torch.float32, device=DEV)
        self.bg_jobs = torch.zeros(self.nbg_all, device=DEV).index_add(0, self.cell_bg_all, self.cell_jobs)
        self.ncell = len(cbg)
        self.cell_metro = torch.tensor(metro_all[cbg], device=DEV)
        # items
        self.it_cell = torch.tensor(it["cell"].astype(np.int64), device=DEV)
        self.it_mode = torch.tensor(it["mode"].astype(np.int64), device=DEV)
        h = np.clip(it["h"], H_KNOTS[0], H_KNOTS[-1]); v = np.clip(it["v"], V_KNOTS[0], V_KNOTS[-1]); d = np.clip(it["d"], 0, D_KNOTS[-1])
        self.it_h = torch.tensor(h, dtype=torch.float32, device=DEV); self.it_d = torch.tensor(d, dtype=torch.float32, device=DEV)
        self.hi, self.hf = interp_index(np.log(h), np.log(H_KNOTS))
        self.vi, self.vf = interp_index(v, V_KNOTS)
        self.di, self.df = interp_index(d, D_KNOTS)
        self.it_metro = self.cell_metro[self.it_cell]
        # census blocks: the origin access is the census-population-weighted mean of the level of service of the res-10 cell under each block's centre
        # (blocks with no stored cell, i.e. no stop within reach, count with 0)
        if (tf.TD / "blocks.npz").is_file():
            bk = np.load(tf.TD / "blocks.npz")
            bo = np.where(bk["bg"] >= 0, remap[np.maximum(bk["bg"], 0)], -1)
            self.blk_orig = torch.tensor(np.where(bo >= 0, bo, self.nb), device=DEV)
            self.blk_pop = torch.tensor(bk["pop"], dtype=torch.float32, device=DEV)
            self.blk_cell = torch.tensor(np.where(bk["cell"] >= 0, bk["cell"], self.ncell), device=DEV)   # ncell = an appended zero
            den = torch.zeros(self.nb + 1, device=DEV).index_add(0, self.blk_orig, self.blk_pop)[: self.nb]
            self.blk_den = torch.where(den > 0, den, self.pop)
        # flows home -> work (origins restricted to the used block groups)
        f = np.load(tf.TD / "flows.npz")
        fo = remap[f["h"]]; keep = fo >= 0
        self.fl_o = torch.tensor(fo[keep], device=DEV); self.fl_w = torch.tensor(f["w"][keep].astype(np.int64), device=DEV)
        n_ = torch.tensor(f["n"][keep], dtype=torch.float32, device=DEV)
        tot = torch.zeros(self.nb, device=DEV).index_add(0, self.fl_o, n_)
        self.fl_wt = n_ / tot[self.fl_o].clamp(min=1.0)
        self.has_flows = (tot > 0).float()
        # the base Data's helpers (controls, service classes) need these
        self._tab = tab.iloc[self.idx].reset_index(); self.bgt = bgt.iloc[self.idx].reset_index(drop=True)
        self.service_class = self._service_classes()
        print(f"data2: {self.nb:,} block groups, {self.ncell:,} cells, {len(it['cell']):,} items, {int(keep.sum()):,} flows, metros {len(self.metros)} ({time.time() - t0:.0f}s)", flush=True)

    def controls(self, names):
        return tf.Data.controls(self, [n for n in names if n not in ("commute_speed",)]) if "commute_speed" not in names else self._controls_with_speed(names)

    def _controls_with_speed(self, names):
        X = tf.Data.controls(self, [n for n in names if n != "commute_speed"])
        t = self._tab
        sp = (60 * t.commute_km_residents / t.commute_time_residents_min.where(t.commute_time_residents_min > 0)).replace([np.inf, -np.inf], np.nan)
        sp = sp.fillna(sp.median()); z = ((sp - sp.mean()) / sp.std()).to_numpy()
        self.last_control_names = self.last_control_names + ["commute_speed"]
        return torch.cat([X, torch.tensor(z, dtype=torch.float32, device=DEV)[:, None]], 1)

    def _service_classes(self):
        best = torch.full((self.ncell,), 1e9, device=DEV)
        m = self.it_d <= 400
        best.scatter_reduce_(0, self.it_cell[m], self.it_h[m], reduce="amin", include_self=True)
        cat = torch.where(best > 1e8, 0, torch.where(best >= 60, 1, torch.where(best >= 30, 2, torch.where(best >= 15, 3, 4))))
        sh = torch.zeros((self.nb + 1, 5), device=DEV)
        sh.index_put_((self.cell_orig, cat), self.cell_pop, accumulate=True)
        return sh[: self.nb].argmax(1).cpu().numpy()


class Model2(torch.nn.Module):
    def __init__(self, ncontrol=0, dest=True, metros=0, metro_curves=False, walk="table", flat_below=None):
        super().__init__()
        self.walk = walk
        self.flat_below = flat_below   # e.g. 5: the headway score is flat for headways up to 5 min (the first knots tied to the 5-min value)
        # walk="loglogistic": W_mode(d) = 1 / (1 + (d / r_mode)^k_mode), two numbers per mode (r = half-effect distance, k = steepness) instead of a 16-knot table
        self.lr_w = torch.nn.Parameter(torch.tensor([math.log(250.0), math.log(400.0), math.log(700.0)]))
        self.lk_w = torch.nn.Parameter(torch.tensor([math.log(1.5)] * 3))
        P = torch.nn.Parameter
        lh = np.log(H_KNOTS); dh = np.diff(np.maximum(lh, np.log(5.0)))
        self.uh = P(torch.tensor(tf.inv_softplus(0.9 * dh + 1e-3), dtype=torch.float32))
        dv = np.diff(np.log(V_KNOTS / (V_KNOTS + 10.0)))[::-1].copy()
        self.uv = P(torch.tensor(tf.inv_softplus(dv + 1e-3), dtype=torch.float32))
        dd = np.diff(D_KNOTS)
        self.ud = P(torch.tensor(np.stack([tf.inv_softplus(dd / s + 1e-3) for s in (300.0, 500.0, 900.0)]), dtype=torch.float32))
        self.um = P(torch.tensor([tf.inv_softplus(0.2), tf.inv_softplus(0.2)], dtype=torch.float32))
        self.alpha = P(torch.tensor([-4.0])); self.leps = P(torch.tensor([math.log(0.05)])); self.lkappa = P(torch.tensor([math.log(5.0)]))
        self.dest = dest
        if dest:
            self.ltheta = P(torch.tensor([math.log(0.5)])); self.leps_d = P(torch.tensor([math.log(0.05)]))
        self.delta = P(torch.zeros(ncontrol))
        self.K = metros
        self.metro_curves = metro_curves
        if metros:
            self.galpha = P(torch.zeros(metros + 1))
            if metro_curves:   # deviations of the log tables per metro (row 0 = the rest of the country, fixed at zero)
                self.dh = P(torch.zeros(metros, len(H_KNOTS))); self.dd = P(torch.zeros(metros, 3, len(D_KNOTS))); self.dm = P(torch.zeros(metros, 2))

    def tables(self):
        raw = tf.mono_table(self.uh)
        lh = raw - raw[H_NORM]                                   # H(10 min) = 1; shorter headways may score above 1
        if self.flat_below:
            i5 = int(np.flatnonzero(H_KNOTS == self.flat_below)[0])
            lh = torch.cat([lh[i5].expand(i5), lh[i5:]])
        lv = torch.flip(tf.mono_table(self.uv), [0])
        ld = torch.stack([tf.mono_table(self.ud[m]) for m in range(3)])
        s1, s2 = torch.nn.functional.softplus(self.um)
        lm = torch.stack([-(s1 + s2), -s1, torch.zeros((), device=s1.device)])
        return lh, lv, ld, lm

    def _metro_tables(self, lh, ld, lm):
        """(K+1, ...) tables: row 0 national, row k national + metro k's deviation (monotone kept by the penalty and a cumulative-max guard)."""
        z = lambda t: torch.zeros((1,) + t.shape[1:], device=t.device)  # noqa: E731
        LH = lh[None] + torch.cat([z(self.dh), self.dh])
        LH = -torch.cummax(-LH, dim=1).values                              # still non-increasing in headway
        LD = ld[None] + torch.cat([z(self.dd), self.dd])
        LD = -torch.cummax(-LD, dim=2).values
        LD = LD - LD[:, :, :1]                                           # W(0) = 1
        dm = torch.cat([z(self.dm), self.dm])
        LM = lm[None] + torch.cat([dm, torch.zeros((dm.shape[0], 1), device=dm.device)], 1)
        LM = torch.minimum(LM, torch.zeros_like(LM))
        return LH, LD, LM

    def cell_access(self, d: Data2):
        lh, lv, ld, lm = self.tables()
        if self.K and self.metro_curves:
            LH, LD, LM = self._metro_tables(lh, ld, lm)
            g = d.it_metro
            nh, nd = LH.shape[1], LD.shape[2]
            lq = gather(LM, g * 3 + d.it_mode) + tf.table_eval(lv, d.vi, d.vf) + gather(LH, g * nh + d.hi) * (1 - d.hf) + gather(LH, g * nh + d.hi + 1) * d.hf
            flat = (g * 3 + d.it_mode) * nd + d.di
            lw = gather(LD, flat) * (1 - d.df) + gather(LD, flat + 1) * d.df
        else:
            lq = gather(lm, d.it_mode) + tf.table_eval(lv, d.vi, d.vf) + tf.table_eval(lh, d.hi, d.hf)
            if self.walk == "loglogistic":
                r = gather(torch.exp(self.lr_w), d.it_mode); k = gather(torch.exp(self.lk_w), d.it_mode)
                lw = -torch.log1p(torch.exp(k * torch.log(d.it_d.clamp(min=1.0) / r)))
            else:
                flat = d.it_mode * len(D_KNOTS) + d.di
                lw = gather(ld, flat) * (1 - d.df) + gather(ld, flat + 1) * d.df
        a = torch.exp(lq + lw)
        return torch.zeros(d.ncell, device=a.device).scatter_reduce(0, d.it_cell, a, reduce="amax", include_self=True)

    def eta(self, d: Data2, X=None, use_metro=True):
        cell = self.cell_access(d)
        if getattr(d, "blk_cell", None) is not None:
            ce = torch.cat([cell, torch.zeros(1, device=cell.device)])
            A = torch.zeros(d.nb + 1, device=cell.device).index_add(0, d.blk_orig, d.blk_pop * gather(ce, d.blk_cell))[: d.nb] / d.blk_den.clamp(min=1.0)
        else:
            A = torch.zeros(d.nb + 1, device=cell.device).index_add(0, d.cell_orig, cell * d.cell_pop)[: d.nb] / d.pop.clamp(min=1.0)
        e = self.alpha + torch.log1p(A / torch.exp(self.leps))
        D = None
        if self.dest:
            Aw = torch.zeros(d.nbg_all, device=cell.device).index_add(0, d.cell_bg_all, cell * d.cell_jobs) / d.bg_jobs.clamp(min=1.0)
            D = torch.zeros(d.nb, device=cell.device).index_add(0, d.fl_o, d.fl_wt * gather(Aw, d.fl_w))
            e = e + torch.exp(self.ltheta) * torch.log1p(D / torch.exp(self.leps_d)) * d.has_flows
        if X is not None and X.shape[1]:
            e = e + X @ self.delta
        if self.K and use_metro:
            e = e + self.galpha[torch.as_tensor(d.metro, device=cell.device)]
        return e, A, D

    def penalty(self, smooth=1.0, ridge=1e-5, pool=0.0):
        lh, lv, ld, lm = self.tables()
        d2 = lambda t: ((t[..., 2:] - 2 * t[..., 1:-1] + t[..., :-2]) ** 2).sum()  # noqa: E731
        p = smooth * 1e-6 * (d2(lh) + d2(lv) + d2(ld)) + ridge * (self.delta ** 2).sum()
        if self.K:
            p = p + 1e-8 * (self.galpha ** 2).sum()
            if self.metro_curves:
                p = p + pool * 1e-6 * ((self.dh ** 2).sum() + (self.dd ** 2).sum() + (self.dm ** 2).sum()) + smooth * 1e-6 * (d2(self.dh) + d2(self.dd))
        return p


def loss_fn(m, d, w, X=None, use_metro=True):
    e, A, D = m.eta(d, X, use_metro)
    mu = torch.sigmoid(e).clamp(1e-7, 1 - 1e-7)
    return -(w * bb_loglik(d.y, d.n, mu, torch.exp(m.lkappa)) / d.n).sum(), mu, A, D


def fit2(m, d, w, X=None, steps=3000, lr=0.03, lbfgs=100, pool=0.0, ridge=1e-5, quiet=False):
    m.to(DEV)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr); t0 = time.time()
    for s in range(steps):
        if s == steps // 3:
            for g_ in opt.param_groups:
                g_["lr"] = lr / 3
        opt.zero_grad(); L = loss_fn(m, d, w, X)[0]; (L + m.penalty(pool=pool, ridge=ridge)).backward(); opt.step()
        if not quiet and s % 1000 == 0:
            print(f"  adam {s:5d} loss {L.item():.6f} ({time.time() - t0:.0f}s)", flush=True)
    if lbfgs:
        o2 = torch.optim.LBFGS(params, lr=0.5, max_iter=lbfgs, history_size=20, line_search_fn="strong_wolfe")

        def cl():
            o2.zero_grad(); t = loss_fn(m, d, w, X)[0] + m.penalty(pool=pool, ridge=ridge); t.backward(); return t
        o2.step(cl)
    with torch.no_grad():
        L = loss_fn(m, d, w, X)[0].item()
    if not quiet:
        print(f"  final {L:.6f} ({time.time() - t0:.0f}s)", flush=True)
    return L


def weights(d, lam=0.5, mask=None):
    w = d.pop * torch.tensor(d.group_pop[d.group], dtype=torch.float32, device=DEV) ** (-lam)
    if mask is not None:
        w = w * torch.tensor(mask, dtype=torch.float32, device=DEV)
    return w / w.sum()


def null_ll(d, w, X=None, metro=False):
    """Per-block-group log likelihood of the intercept (+ controls, + metro intercepts) model with no transit."""
    m = Model2(X.shape[1] if X is not None else 0, dest=False, metros=len(d.metros) if metro else 0).to(DEV)
    for p in (m.uh, m.uv, m.ud, m.um, m.leps):
        p.requires_grad_(False)

    class Z(torch.nn.Module):
        pass
    params = [m.alpha, m.lkappa, m.delta] + ([m.galpha] if metro else [])
    opt = torch.optim.LBFGS(params, lr=0.5, max_iter=300, history_size=20, line_search_fn="strong_wolfe")

    def eta0():
        e = m.alpha + (X @ m.delta if X is not None and X.shape[1] else 0)
        return e + (m.galpha[torch.as_tensor(d.metro, device=DEV)] if metro else 0)

    def cl():
        opt.zero_grad(); mu = torch.sigmoid(eta0()).clamp(1e-7, 1 - 1e-7)
        L = -(w * bb_loglik(d.y, d.n, mu, torch.exp(m.lkappa)) / d.n).sum() + 1e-5 * (m.delta ** 2).sum(); L.backward(); return L
    adam = torch.optim.Adam(params, lr=0.05)   # the gradients are tiny (per-worker log likelihood): L-BFGS alone stops at the start
    for _ in range(600):
        adam.zero_grad(); mu = torch.sigmoid(eta0()).clamp(1e-7, 1 - 1e-7)
        (-(w * bb_loglik(d.y, d.n, mu, torch.exp(m.lkappa)) / d.n).sum() + 1e-5 * (m.delta ** 2).sum()).backward(); adam.step()
    for _ in range(2):
        opt.step(cl)
    return m, eta0


def per_bg_ll(m, d, X=None, use_metro=True):
    with torch.no_grad():
        e, _, _ = m.eta(d, X, use_metro)
        mu = torch.sigmoid(e).clamp(1e-7, 1 - 1e-7)
        return (bb_loglik(d.y, d.n, mu, torch.exp(m.lkappa)) / d.n).cpu().numpy()


def curves2(m, metro_row=None):
    with torch.no_grad():
        lh, lv, ld, lm = m.tables()
        if metro_row is not None and m.K and m.metro_curves:
            LH, LD, LM = m._metro_tables(lh, ld, lm); lh, ld, lm = LH[metro_row], LD[metro_row], LM[metro_row]
        lh, lv, ld, lm = [t.detach().cpu().numpy() for t in (lh, lv, ld, lm)]
    if m.walk == "loglogistic":
        r = torch.exp(m.lr_w).detach().cpu().numpy(); kk = torch.exp(m.lk_w).detach().cpu().numpy()
        ld = np.stack([-np.log1p((np.maximum(D_KNOTS, 1.0) / r[k]) ** kk[k]) for k in range(3)])
    out = {"headway": {"x": H_KNOTS.tolist(), "y": np.exp(lh).tolist()}, "speed": {"x": V_KNOTS.tolist(), "y": np.exp(lv).tolist()},
           "mode": dict(zip(MODES, np.exp(lm).tolist())), "walk": {MODES[k]: {"x": D_KNOTS.tolist(), "y": np.exp(ld[k]).tolist()} for k in range(3)},
           "eps": float(torch.exp(m.leps)), "alpha": float(m.alpha)}
    if m.dest:
        out["theta"] = float(torch.exp(m.ltheta)); out["eps_d"] = float(torch.exp(m.leps_d))
    return out


def cv_national(d, mk, X, lam=0.5, steps=900, base_sd=None):
    """Metro-grouped 5-fold CV of the national model (held-out metros, no metro terms). Returns held-out skill overall and the median over the 30 biggest metros."""
    folds = tf.cv_folds(d)
    w = weights(d, lam)
    ho = np.zeros(d.nb); ho0 = np.zeros(d.nb)
    for f in range(5):
        tr = weights(d, lam, (folds != f).astype(float))
        m = mk(); m.load_state_dict(base_sd); fit2(m, d, tr, X, steps=steps, lr=0.01, lbfgs=60, quiet=True)
        ho[folds == f] = per_bg_ll(m, d, X)[folds == f]
        m0, e0 = null_ll(d, tr, X)
        with torch.no_grad():
            mu0 = torch.sigmoid(e0()).clamp(1e-7, 1 - 1e-7)
            ho0[folds == f] = (bb_loglik(d.y, d.n, mu0, torch.exp(m0.lkappa)) / d.n).cpu().numpy()[folds == f]
    wn = w.cpu().numpy()
    per = [1 - (wn[d.group == g] * ho[d.group == g]).sum() / (wn[d.group == g] * ho0[d.group == g]).sum() for g in np.argsort(-d.group_pop)[:30]]
    return float(1 - (wn * ho).sum() / (wn * ho0).sum()), float(np.median(per)), per


def national(d, name, controls="none", dest=True, lam=0.5, steps=3000, cv=True):
    X = d.controls(CONTROLS[controls]) if controls != "none" else None
    w = weights(d, lam)
    mk = lambda: Model2(X.shape[1] if X is not None else 0, dest=dest)  # noqa: E731
    m = mk(); t0 = time.time()
    L = fit2(m, d, w, X, steps=steps)
    m0, e0 = null_ll(d, w)
    with torch.no_grad():
        mu0 = torch.sigmoid(e0()).clamp(1e-7, 1 - 1e-7)
        L0 = -(w * bb_loglik(d.y, d.n, mu0, torch.exp(m0.lkappa)) / d.n).sum().item()
        _, mu, A, D = loss_fn(m, d, w, X)
    res = {"name": name, "controls": controls, "dest": dest, "loss": L, "null": L0, "skill": 1 - L / L0, "curves": curves2(m),
           "control_names": getattr(d, "last_control_names", []) if X is not None else [], "delta": m.delta.detach().cpu().tolist(),
           "calibration": tf.calibration(d, mu)}
    if X is not None:
        mc, ec = null_ll(d, w, X)
        with torch.no_grad():
            muc = torch.sigmoid(ec()).clamp(1e-7, 1 - 1e-7)
            Lc = -(w * bb_loglik(d.y, d.n, muc, torch.exp(mc.lkappa)) / d.n).sum().item()
        res["null_with_controls"] = Lc; res["skill_beyond_controls"] = 1 - L / Lc
    if cv:
        res["cv_skill"], res["cv_top30_median"], res["cv_top30"] = cv_national(d, mk, X, lam, base_sd=m.state_dict())
    res["seconds"] = time.time() - t0
    (RUNS / f"{name}.json").write_text(json.dumps(res, indent=1))
    torch.save(m.state_dict(), RUNS / f"{name}.pt")
    print(f"[{name}] skill {res['skill']:.4f}" + (f" beyond-controls {res['skill_beyond_controls']:.4f}" if X is not None else "")
          + (f" cv {res['cv_skill']:.4f} top30-median {res['cv_top30_median']:.4f}" if cv else "") + f" ({res['seconds']:.0f}s)", flush=True)
    return m, res


def metro_joint(d, name, controls="none", dest=True, pools=(3.0, 30.0, 300.0), base=None, steps=2500):
    """Joint model for the biggest metros, evaluated by tract-grouped 5-fold CV inside the metros. Variants: national curves only; national curves + metro
    intercepts; metro intercepts + metro curve deviations with each pooling strength. Skill is per metro against its own intercept-only (+ controls) null."""
    X = d.controls(CONTROLS[controls]) if controls != "none" else None
    nc = X.shape[1] if X is not None else 0
    K = len(d.metros)
    tract = np.array([g[:11] for g in d.geoid]); _, tid = np.unique(tract, return_inverse=True)
    folds = np.random.default_rng(3).integers(0, 5, tid.max() + 1)[tid]
    w_all = weights(d, 0.5)
    base_sd = torch.load(RUNS / f"{base}.pt", map_location=DEV) if base else None
    variants = [("national", 0, False, 0.0), ("national+intercepts", K, False, 0.0)] + [(f"metro_curves_pool{int(p)}", K, True, p) for p in pools]
    ho = {v[0]: np.zeros(d.nb) for v in variants}; ho0 = np.zeros(d.nb)
    fitted = {}
    t0 = time.time()
    for f in range(5):
        trm = (folds != f).astype(float)
        wtr = weights(d, 0.5, trm)
        m0, e0 = null_ll(d, wtr, X, metro=True)
        with torch.no_grad():
            mu0 = torch.sigmoid(e0()).clamp(1e-7, 1 - 1e-7)
            ho0[folds == f] = (bb_loglik(d.y, d.n, mu0, torch.exp(m0.lkappa)) / d.n).cpu().numpy()[folds == f]
        for vname, k, curves_, pool in variants:
            m = Model2(nc, dest=dest, metros=k, metro_curves=curves_)
            if base_sd is not None:
                m.load_state_dict({kk: vv for kk, vv in base_sd.items() if kk in dict(m.named_parameters()) and vv.shape == dict(m.named_parameters())[kk].shape}, strict=False)
            fit2(m, d, wtr, X, steps=steps if base_sd is None else 900, lr=0.03 if base_sd is None else 0.01, lbfgs=60, pool=pool, quiet=True)
            ho[vname][folds == f] = per_bg_ll(m, d, X, use_metro=True)[folds == f]
            if f == 0:
                fitted[vname] = m
        print(f"  fold {f} done ({time.time() - t0:.0f}s)", flush=True)
    wn = w_all.cpu().numpy()
    out = {"name": name, "controls": controls, "dest": dest, "metros": {}}
    for i, cb in enumerate(d.metros):
        msk = d.metro == i + 1
        den = (wn[msk] * ho0[msk]).sum()
        out["metros"][cb] = {"pop": float(d.pop.cpu().numpy()[msk].sum()), "transit_share": float(d.y.cpu().numpy()[msk].sum() / d.n.cpu().numpy()[msk].sum()),
                             "cv_skill": {v: float(1 - (wn[msk] * ho[v][msk]).sum() / den) for v in ho}}
        mc = fitted.get(f"metro_curves_pool{int(pools[1])}")
        if mc is not None:
            out["metros"][cb]["curves"] = curves2(mc, metro_row=i + 1)
    out["national_curves"] = curves2(fitted["national"])
    out["median_cv_skill"] = {v: float(np.median([out["metros"][cb]["cv_skill"][v] for cb in d.metros])) for v in ho}
    out["wins_vs_intercepts"] = {v: int(sum(out["metros"][cb]["cv_skill"][v] > out["metros"][cb]["cv_skill"]["national+intercepts"] for cb in d.metros)) for v in ho}
    (RUNS / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(f"[{name}] median CV skill per metro: " + ", ".join(f"{k}={v:.4f}" for k, v in out["median_cv_skill"].items()), flush=True)
    return out


class TractData:
    """The same data aggregated to census tracts: outcomes and population summed, cells and flow origins re-pointed to their tract, controls the
    population-weighted mean of the block groups' standardised controls."""

    def __init__(self, d: Data2):
        self.bgd = d
        tract = np.array([g[:11] for g in d.geoid])
        codes, inv = np.unique(tract, return_inverse=True)
        self.nb = len(codes); self.geoid = codes
        inv_t = torch.tensor(inv, device=DEV)
        agg = lambda v: torch.zeros(self.nb, device=DEV).index_add(0, inv_t, v)  # noqa: E731
        self.y, self.n, self.pop = agg(d.y), agg(d.n), agg(d.pop)
        first = np.zeros(self.nb, np.int64); first[inv[::-1]] = np.arange(len(inv))[::-1]
        self.group = d.group[first]; self.metro = d.metro[first]; self.cbsa = d.cbsa[first]
        self.group_names = d.group_names
        self.group_pop = np.bincount(self.group, weights=self.pop.cpu().numpy(), minlength=len(d.group_names))
        self.metros = d.metros
        bg2t = torch.cat([inv_t, torch.tensor([self.nb], device=DEV)])          # the dummy bucket stays a dummy bucket
        self.cell_orig = bg2t[d.cell_orig]
        self.cell_pop = d.cell_pop
        for a in ("cell_bg_all", "cell_jobs", "bg_jobs", "ncell", "nbg_all", "it_cell", "it_mode", "it_h", "it_d", "hi", "hf", "vi", "vf", "di", "df", "it_metro",
                  "fl_w", "fl_wt"):
            setattr(self, a, getattr(d, a))
        self.fl_o = inv_t[d.fl_o]
        if getattr(d, "blk_cell", None) is not None:
            self.blk_orig = bg2t[d.blk_orig]; self.blk_pop = d.blk_pop; self.blk_cell = d.blk_cell
            den = torch.zeros(self.nb + 1, device=DEV).index_add(0, self.blk_orig, self.blk_pop)[: self.nb]
            self.blk_den = torch.where(den > 0, den, self.pop)
        tot = torch.zeros(self.nb, device=DEV).index_add(0, self.fl_o, d.fl_wt * d.pop[d.fl_o])
        self.fl_wt = d.fl_wt * d.pop[d.fl_o] / tot[self.fl_o].clamp(min=1e-9)      # tract destination access = population-weighted over its block groups
        self.has_flows = (tot > 0).float()
        self.inv_t = inv_t
        sc = torch.zeros((self.nb, 5), device=DEV); sc.index_put_((inv_t, torch.tensor(d.service_class, device=DEV)), d.pop, accumulate=True)
        self.service_class = sc.argmax(1).cpu().numpy()

    def controls(self, names):
        Xb = self.bgd.controls(names); self.last_control_names = self.bgd.last_control_names
        w = self.bgd.pop[:, None]
        num = torch.zeros((self.nb, Xb.shape[1]), device=DEV).index_add(0, self.inv_t, Xb * w)
        X = num / torch.zeros(self.nb, device=DEV).index_add(0, self.inv_t, self.bgd.pop)[:, None].clamp(min=1.0)
        return (X - X.mean(0)) / X.std(0).clamp(min=1e-6)


def random_init(m, rng):
    """Very different starting curves: every raw step parameter drawn wide (curves from almost flat to very steep), random link constants."""
    with torch.no_grad():
        for prm in (m.uh, m.uv, m.ud, m.um):
            prm.copy_(torch.tensor(rng.normal(-1.0, 1.5, prm.shape), dtype=torch.float32))
        m.leps.fill_(float(np.log(rng.choice([0.005, 0.05, 0.5])))); m.alpha.fill_(float(rng.uniform(-6, -3)))
        m.lr_w.copy_(torch.tensor(np.log(rng.uniform(100, 1500, 3)), dtype=torch.float32)); m.lk_w.copy_(torch.tensor(np.log(rng.uniform(0.5, 4, 3)), dtype=torch.float32))
        if m.dest:
            m.ltheta.fill_(float(np.log(rng.choice([0.1, 0.5, 1.5])))); m.leps_d.fill_(float(np.log(rng.choice([0.005, 0.05, 0.5]))))


def summary(c):
    H = c["headway"]; W = c["walk"]
    h = lambda x: float(np.exp(np.interp(np.log(x), np.log(H["x"]), np.log(H["y"]))))  # noqa: E731
    out = {f"H{x:g}": h(x) for x in (1, 3, 5, 15, 30, 60, 120, 240)}
    for mo in MODES:
        for x in (100, 200, 400, 800, 1500):
            out[f"W{mo}{x}"] = float(np.interp(x, W[mo]["x"], W[mo]["y"]))
    out.update({"Mbus": c["mode"]["bus"], "Mtram": c["mode"]["tram"], "S5": float(np.interp(5, c["speed"]["x"], c["speed"]["y"])),
                "S15": float(np.interp(15, c["speed"]["x"], c["speed"]["y"])), "theta": c.get("theta", 0.0), "eps": c["eps"]})
    return out


def multistart(d, name, controls="none", dest=True, starts=6, steps=4000, smooth=1.0, walk="table", flat_below=None, cv=True):
    """The same fit from very different starting curves: spread of every curve value and of the loss."""
    X = d.controls(CONTROLS[controls]) if controls != "none" else None
    w = weights(d, 0.5)
    runs = []
    for sd in range(starts):
        m = Model2(X.shape[1] if X is not None else 0, dest=dest, walk=walk, flat_below=flat_below)
        if sd:
            random_init(m, np.random.default_rng(100 + sd))
        m.to(DEV)
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=0.03)
        for s in range(steps):
            if s == steps // 3:
                for g_ in opt.param_groups:
                    g_["lr"] = 0.01
            opt.zero_grad(); L = loss_fn(m, d, w, X)[0]; (L + m.penalty(smooth=smooth)).backward(); opt.step()
        o2 = torch.optim.LBFGS(params, lr=0.5, max_iter=150, history_size=20, line_search_fn="strong_wolfe")

        def cl():
            o2.zero_grad(); t = loss_fn(m, d, w, X)[0] + m.penalty(smooth=smooth); t.backward(); return t
        o2.step(cl)
        with torch.no_grad():
            L = loss_fn(m, d, w, X)[0].item()
        runs.append({"seed": sd, "loss": L, "summary": summary(curves2(m)), "curves": curves2(m), "_state": {k: v.detach().clone() for k, v in m.state_dict().items()}})
        print(f"  [{name}] start {sd}: loss {L:.7f}  " + " ".join(f"{k} {v:.3f}" for k, v in runs[-1]["summary"].items() if k in ("H5", "H15", "H60", "H120", "Wbus200", "Wbus400", "Wrail800", "Mbus", "theta")), flush=True)
    keys = runs[0]["summary"].keys()
    spread = {k: {"min": min(r["summary"][k] for r in runs), "max": max(r["summary"][k] for r in runs)} for k in keys}
    out = {"name": name, "controls": controls, "dest": dest, "smooth": smooth, "walk": walk, "flat_below": flat_below, "runs": runs, "spread": spread,
           "loss_range": [min(r["loss"] for r in runs), max(r["loss"] for r in runs)]}
    if cv:   # held-out skill of the best start (metro-grouped 5 folds), so stabilisers are judged on prediction too
        best = min(range(len(runs)), key=lambda i: runs[i]["loss"])
        mk = lambda: Model2(X.shape[1] if X is not None else 0, dest=dest, walk=walk, flat_below=flat_below)  # noqa: E731
        sd_best = runs[best].pop("_state", None)
        if sd_best is not None:
            out["cv_skill"], out["cv_top30_median"], _ = cv_national(d, mk, X, 0.5, base_sd=sd_best)
    for r in runs:
        r.pop("_state", None)
    (RUNS / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(f"[{name}] loss range {out['loss_range'][0]:.7f}-{out['loss_range'][1]:.7f}; widest spreads: " +
          ", ".join(f"{k} {v['min']:.2f}-{v['max']:.2f}" for k, v in sorted(spread.items(), key=lambda kv: -(kv[1]['max'] - kv[1]['min']))[:8])
          + (f"; cv {out['cv_skill']:.4f} top30 {out['cv_top30_median']:.4f}" if "cv_skill" in out else ""), flush=True)
    return out


def main():
    e = sys.argv[1]
    d = Data2()
    if e == "starts":
        multistart(d, "h2_starts_none", controls="none")
        multistart(d, "h2_starts_structural", controls="structural")
        return
    if e == "stab_starts":   # stabilisers: two-parameter walking curves; much smoother free curves
        multistart(d, "h2_stab_loglogistic_none", controls="none", walk="loglogistic", starts=5, steps=3000)
        multistart(d, "h2_stab_smooth30_none", controls="none", smooth=30.0, starts=5, steps=3000)
        multistart(d, "h2_stab_loglogistic_structural", controls="structural", walk="loglogistic", starts=5, steps=3000)
        multistart(d, "h2_stab_flat5_none", controls="none", flat_below=5, starts=5, steps=3000)
        multistart(d, "h2_stab_flat5_loglogistic_none", controls="none", flat_below=5, walk="loglogistic", starts=5, steps=3000)
        return
    if e == "tract_starts":
        t = TractData(d)
        print(f"tracts: {t.nb:,}", flush=True)
        multistart(t, "h2_tract_starts_none", controls="none", starts=5, steps=3000)
        multistart(t, "h2_tract_starts_structural", controls="structural", starts=5, steps=3000)
        return
    if e == "national":
        national(d, "h2_nodest_none", dest=False)
        national(d, "h2_dest_none", dest=True)
        national(d, "h2_dest_structural", controls="structural")
        national(d, "h2_nodest_structural", controls="structural", dest=False)
        national(d, "h2_dest_structural+commute_km", controls="structural+commute_km")
        national(d, "h2_dest_structural+cars", controls="structural+cars")
        national(d, "h2_dest_overcontrolled", controls="structural+cars+time+speed")
    elif e == "metros":
        metro_joint(d, "h2_metros_none", controls="none", base="h2_dest_none")
        metro_joint(d, "h2_metros_structural", controls="structural", base="h2_dest_structural")


if __name__ == "__main__":
    main()
