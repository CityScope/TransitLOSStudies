"""Stable calibration: one model fitted to total transit commuting AND to bus, rail and tram commuting together (multi-task).

    $TORCH_PYTHON fit/torch_fit3.py <experiment>

Why: from total transit use alone the curves are not identified -- the mode values, the speed curve and the walking curve of each mode trade off (rail is fast,
bus slow; a rail and a bus commuter count the same), so very different curves predict the same (see torch_fit2.py multistart). Bus commuters respond to bus
stops only, rail commuters to rail stations, tram commuters to tram stops: fitted together with shared curves, these outcomes pin the headway curve and the
walking curve of each mode; the total then only has to set the mode values.

Model (the map's structure; everything a free monotone table unless said otherwise):
  item value  a = H(headway) * S(speed) * W_mode(walk distance)                 (no mode value)
  cell, mode  a_cm = max over the cell's items of that mode  (soft maximum while fitting if `soft` > 0, annealed to the max)
  cell        a_c  = max over modes of M_mode * a_cm                           (M: bus, tram; rail = 1)
  block group A_m, A = census-block-population-weighted means of a_cm, a_c  (blocks with no stored cell = 0)
  destination D_m, D = job-weighted access where the residents work (LODES flows)
  tasks       total transit:  logit = alpha + log(1 + A / eps) + theta log(1 + D / eps_d) [+ controls]
              mode m (bus / rail / tram commuters, ACS B08301):  logit = alpha_m + log(1 + A_m / eps_m) + theta_m log(1 + D_m / eps_dm) [+ controls]
  loss        sum over tasks of the population-weighted (metro-tempered) beta-binomial per-worker loss, each divided by its own no-transit loss
Speed: 'abs' (table of km/h), 'rel' (table of speed / the mode's median speed: separates speed from mode), or 'none'.
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
import torch_fit2 as t2  # noqa: E402
from torch_fit import DEV, MODES, gather, bb_loglik, interp_index  # noqa: E402

H_KNOTS = t2.H_KNOTS
D_KNOTS = tf.D_KNOTS
V_KNOTS = tf.V_KNOTS
WALK_MAX_M = 2000.0   # walking ends here: items farther away (park-and-ride, feeder buses, bikes) are the separate far-access channel, never the walk curve
FAR_KNOTS = np.array([2000, 3000, 4000, 5000, 10000, 15000, 20000], float)   # far-access curve: 1 km steps to 5 km, then 5 km steps (fit/build_items.py)
R_KNOTS = np.array([0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.45, 1.75, 2.1, 2.6, 3.5], float)   # speed / median speed of the mode
RUNS = tf.RUNS
TASKS = ["total", "bus", "rail", "tram"]
# commute length is partly an outcome of transit (good service, longer commutes by transit, other jobs reachable): a variant only, a lower bound of the effect
CONTROLS3 = {**t2.CONTROLS, "commute_km": ["commute_km"],
             # the map's two columns at the other end of the commute (usa_study/add_od_los_columns.py): built on the DEFAULT score, so a labelled variant only
             "od_los": ["losAtWorkplacesResidentsMean", "losAtHomesWorkersMean"]}
OD_LOS_TABLE = tf.TD.parents[3] / "cache" / "derived" / "bg_od_los.parquet"
# areas whose service is missing from the GTFS data: their riders would show as riders without service and pull the curves down, so they are left
# out of every fit and held-out score. 2026-11: the rejected / cross-state / duplicate-picked feeds (WMATA, MDOT MTA, NFTA, PSTA, King County Metro...)
# are restored; still without a usable feed: Pensacola ECAT (Escambia FL, last feed 2022), Clemson Area Transit (Pickens SC, 2020), Tiger Transit (Lee AL).
EXCLUDE_CBSA: set = set()
EXCLUDE_COUNTY = {"12033", "45077", "01081"}


class _Rows(torch.autograd.Function):
    """t[i] for the rows of a 2-D table and a huge index vector; backward = index_add (rows are many, so atomics rarely collide)."""

    @staticmethod
    def forward(ctx, t, i):
        ctx.save_for_backward(i); ctx.shape = t.shape
        return t.index_select(0, i)

    @staticmethod
    def backward(ctx, g):
        (i,) = ctx.saved_tensors
        return torch.zeros(ctx.shape, device=g.device, dtype=g.dtype).index_add_(0, i, g), None


def _rows(t, i):
    return _Rows.apply(t, i)


class Data3(t2.Data2):
    def controls(self, names):
        od = [n for n in names if n in CONTROLS3["od_los"]]; rest = [n for n in names if n not in od]
        parts = ([super().controls(rest)] if rest else []) + ([_od_los_controls(self, od)] if od else [])
        self.last_control_names = (self.last_control_names if rest else []) + od
        return torch.cat(parts, 1)

    def __init__(self):
        super().__init__()
        t = self._tab
        wk = t["workers_b08301"].to_numpy(float)
        self.n_mode = torch.tensor(np.where(np.isfinite(wk) & (wk > 0), wk, 0), dtype=torch.float32, device=DEV)
        ym = {"bus": t[["bus_commuters"]].sum(axis=1, min_count=1), "rail": t[["subway_commuters", "commuter_rail_commuters"]].sum(axis=1, min_count=1),
              "tram": t[["light_rail_commuters"]].sum(axis=1, min_count=1)}
        self.y_mode, self.ok_mode = {}, {}
        for m, s in ym.items():
            v = s.to_numpy(float)
            ok = np.isfinite(v) & (wk > 0)
            self.ok_mode[m] = torch.tensor(ok.astype(np.float32), device=DEV)
            self.y_mode[m] = torch.tensor(np.where(ok, np.minimum(np.maximum(v, 0), wk), 0), dtype=torch.float32, device=DEV)
        # speed relative to the median speed of the item's mode (over all items)
        it = np.load(tf.TD / "items.npz")
        v = it["v"]; mo = it["mode"].astype(np.int64)
        med = np.array([np.median(v[(mo == k) & (v > 0)]) if ((mo == k) & (v > 0)).any() else 20.0 for k in range(3)])
        self.v_median = med
        rel = np.clip(np.where(v > 0, v / med[mo], R_KNOTS[0]), R_KNOTS[0], R_KNOTS[-1])
        self.ri, self.rf = interp_index(rel, R_KNOTS)
        d_raw = it["d"].astype(np.float32)
        far = d_raw > WALK_MAX_M
        self.it_far = torch.tensor(far, device=DEV)
        self.fi, self.ff = interp_index(np.clip(d_raw, FAR_KNOTS[0], FAR_KNOTS[-1]), FAR_KNOTS)
        print(f"data3: {int(far.sum()):,} far-access items (> {WALK_MAX_M:.0f} m) of {len(far):,}", flush=True)
        # items reordered once: walking items first, far items after, so each channel works on a slice (a view) of every item tensor instead of
        # full-length masked copies (the far runs ran out of GPU memory at 108M items)
        n_it = len(far)
        order = torch.tensor(np.argsort(far, kind="stable"), device=DEV)
        for k, v in list(vars(self).items()):
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n_it:
                setattr(self, k, v[order])
        self.n_walk = int((~far).sum())
        del order
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        keep = np.array([str(cb) not in EXCLUDE_CBSA and g[:5] not in EXCLUDE_COUNTY for cb, g in zip(self.cbsa, self.geoid)])
        self.keep = torch.tensor(keep.astype(np.float32), device=DEV)
        print(f"data3: mode outcomes ready; median speed bus/tram/rail {np.round(med, 1).tolist()} km/h; "
              f"{int((~keep).sum()):,} block groups left out (service missing from the feeds)", flush=True)


def _od_los_controls(d, names):
    """Standardised columns of the od-level-of-service table for `d`'s block groups (median-imputed)."""
    t = pd.read_parquet(OD_LOS_TABLE).reindex(pd.Index(d.geoid.astype(str)))
    cols = []
    for n in names:
        v = t[n].to_numpy(dtype="float64"); v = np.where(np.isfinite(v), v, np.nanmedian(v))
        cols.append((v - v.mean()) / max(v.std(), 1e-9))
    return torch.tensor(np.stack(cols, 1), dtype=torch.float32, device=DEV)


class AggData3(t2.TractData):
    """Data3 aggregated to tracts (GEOID[:11]) or counties (GEOID[:5]): what the block-group detail adds. Outcomes, workers and population summed,
    access the census-population-weighted mean over the unit's blocks, destination access population-weighted over its block groups."""

    def __init__(self, d: Data3, level="tract"):
        n = {"tract": 11, "county": 5}[level]
        saved = d.geoid
        d.geoid = np.array([g[:n].ljust(11, "0") for g in saved])   # TractData groups by geoid[:11]: hand it the unit codes padded to that length
        try:
            super().__init__(d)
        finally:
            d.geoid = saved
        self.level = level
        agg = lambda v: torch.zeros(self.nb, device=DEV).index_add(0, self.inv_t, v)  # noqa: E731
        # mode tasks: one worker count per unit, so only block groups with all three mode counts enter them
        okall = d.ok_mode["bus"] * d.ok_mode["rail"] * d.ok_mode["tram"]
        self.n_mode = agg(okall * d.n_mode)
        self.y_mode = {m: agg(d.y_mode[m] * okall) for m in d.ok_mode}
        self.ok_mode = {m: (self.n_mode > 0).float() for m in d.ok_mode}
        self.v_median = d.v_median; self.ri, self.rf = d.ri, d.rf
        self.it_far, self.fi, self.ff = d.it_far, d.fi, d.ff
        self.n_walk = d.n_walk
        self.keep = (agg(1.0 - d.keep) == 0).float()   # a unit with any left-out block group is left out
        print(f"aggregated to {level}: {self.nb:,} units", flush=True)


class Model3(torch.nn.Module):
    def __init__(self, ncontrol=0, speed="abs", soft=0.0, dest=True, flat_below=3.0, mridge=0.0, dest_tasks=("total", "bus", "rail", "tram"), steep=0.0,
                 weights_from="blocks", far="drop", walk="free", fix_mode=False, fixed=None):
        super().__init__()
        self.walk = walk            # "free" (a curve per mode), "shared" (one curve for all modes), "step" (fixed 400 m bus / tram, 800 m rail cut-offs)
        self.fix_mode = fix_mode    # True: every mode valued the same (speed and the rest must carry the difference)
        self.fixed = fixed          # "literature": headway, speed, walk and mode tables fixed at fit/data/torch/literature_tables.json (only the links fitted)
        self.far = far   # items beyond WALK_MAX_M: "drop" (walking access only, max 2 km) or "term" (own far-access curve and its own term in each task)
        self.weights_from = weights_from   # area score = mean of its cells' scores weighted by "blocks" (census blocks, cell under each block's centre)
                                           # or "cells" (each cell's own population; divided by the area's census population, so people no stop reaches = 0)
        self.flat_below = flat_below    # headway score flat up to this headway (the 1-3 min headways are almost only big-city subways)
        self.steep = steep              # cost of steepness: integral of the squared slope of the headway curve (log value against log headway) and of the
                                        # walk curves (against walk km) + theta ** 2; where the data hardly tell steep from gentle, the gentle curve wins.
                                        # Not the speed curve: a flatter speed curve would hand the mode values what speed explains (and mridge the reverse),
                                        # so the speed / mode split is left to the data alone (speed varies within each mode, and the modes overlap in speed)
        self.dest_mask = torch.tensor([1.0 if t in dest_tasks else 0.0 for t in TASKS])   # which tasks get the destination term
        self.mridge = mridge            # pull of log(mode value) towards 0: speed carries all that is speed; the mode value only what speed does not explain
        P = torch.nn.Parameter
        lh = np.log(H_KNOTS); dh = np.diff(np.maximum(lh, np.log(5.0)))
        self.uh = P(torch.tensor(tf.inv_softplus(0.9 * dh + 1e-3), dtype=torch.float32))
        self.speed = speed
        kn = V_KNOTS if speed == "abs" else R_KNOTS
        dv = np.diff(np.log(kn / (kn + kn[len(kn) // 2])))[::-1].copy()
        self.uv = P(torch.tensor(tf.inv_softplus(0.5 * dv + 1e-3), dtype=torch.float32))
        dd = np.diff(D_KNOTS)
        self.ud = P(torch.tensor(np.stack([tf.inv_softplus(dd / s + 1e-3) for s in (300.0, 500.0, 900.0)]), dtype=torch.float32))
        self.lmode = P(torch.tensor([-0.5, -0.3], dtype=torch.float32))   # log mode value of bus, tram (rail = 0); free sign: no mode is assumed better
        self.soft = soft
        self.dest = dest
        T = len(TASKS)
        self.alpha = P(torch.full((T,), -4.0)); self.leps = P(torch.full((T,), math.log(0.05))); self.lkappa = P(torch.full((T,), math.log(5.0)))
        self.ltheta = P(torch.full((T,), math.log(0.5))); self.leps_d = P(torch.full((T,), math.log(0.05)))
        self.delta = P(torch.zeros(T, ncontrol))
        self.uf = P(torch.tensor(tf.inv_softplus(np.diff(FAR_KNOTS) / 8000.0 + 1e-3), dtype=torch.float32))   # far-access curve, 1 at 2 km, decreasing
        self.lthf = P(torch.full((T,), math.log(0.5))); self.lepsf = P(torch.full((T,), math.log(0.05)))   # strength and scale of the far-access term
        if walk == "step":
            radius = np.array([400.0, 400.0, 800.0])   # bus, tram, rail (MODES order)
            self.register_buffer("step_ld", torch.tensor(np.where(D_KNOTS[None, :] <= radius[:, None], 0.0, -30.0), dtype=torch.float32))
            self.ud.requires_grad_(False)
        if fix_mode:
            self.lmode.requires_grad_(False)
        if fixed == "literature":
            lit = json.loads((tf.TD / "literature_tables.json").read_text())
            old_h = np.array([2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 1440], float)   # the knots the tables were written on
            old_d = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 2000, 2500, 3000, 5000], float)
            lh = np.interp(np.log(H_KNOTS), np.log(old_h), lit["lh"]); lh = lh - lh[t2.H_NORM]
            ld = np.stack([np.interp(D_KNOTS, old_d, r) for r in lit["ld"]])
            self.register_buffer("lit_lh", torch.tensor(lh, dtype=torch.float32)); self.register_buffer("lit_lv", torch.tensor(lit["lv"], dtype=torch.float32))
            self.register_buffer("lit_ld", torch.tensor(ld, dtype=torch.float32)); self.register_buffer("lit_lm", torch.tensor(lit["lm"], dtype=torch.float32))
            for prm in (self.uh, self.uv, self.ud, self.lmode):
                prm.requires_grad_(False)

    def far_table(self):
        return tf.mono_table(self.uf)

    def tables(self):
        if self.fixed == "literature":
            return self.lit_lh, self.lit_lv, self.lit_ld, self.lit_lm
        raw = tf.mono_table(self.uh)
        lh = raw - raw[t2.H_NORM]
        if self.flat_below:
            i5 = int(np.flatnonzero(H_KNOTS == self.flat_below)[0])
            lh = torch.cat([lh[i5].expand(i5), lh[i5:]])
        lv = torch.flip(tf.mono_table(self.uv), [0])
        if self.walk == "step":
            ld = self.step_ld
        elif self.walk == "shared":
            ld = tf.mono_table(self.ud[0]).expand(3, -1)
        else:
            ld = torch.stack([tf.mono_table(self.ud[m]) for m in range(3)])
        lm = torch.zeros(3, device=self.lmode.device) if self.fix_mode else torch.cat([self.lmode, torch.zeros(1, device=self.lmode.device)])
        return lh, lv, ld, lm

    def cell_mode_access(self, d):
        """[ncell, 3]: per cell and mode, the best (or soft-best) WALKING item value (items within WALK_MAX_M) without mode value."""
        cm, _, lm = self.access(d, far=False)
        return cm, lm

    def _cell_max(self, la, d, sl=slice(None)):
        idx = d.it_cell[sl] * 3 + d.it_mode[sl]
        mx = torch.full((d.ncell * 3,), -60.0, device=la.device).scatter_reduce(0, idx, la.detach(), reduce="amax", include_self=True)
        if self.soft > 0:   # soft maximum: tau * log sum exp(la / tau), shifted by the (detached) max for stability; tau -> 0 gives the max
            tau = self.soft
            s = torch.zeros(d.ncell * 3, device=la.device).index_add(0, idx, torch.exp((la - gather(mx, idx)) / tau))
            lc = mx + tau * torch.log(s.clamp(min=1e-30))
            out = torch.where(s > 0, torch.exp(lc), torch.zeros_like(lc))
        else:
            out = torch.zeros(d.ncell * 3, device=la.device).scatter_reduce(0, idx, torch.exp(la), reduce="amax", include_self=True)
        return out.view(d.ncell, 3)

    def access(self, d, far=True):
        """(walking access [ncell, 3], far access [ncell, 3] or None, log mode values): station value H x S of every item, times the walk curve for
        items within WALK_MAX_M (the walking channel) or times the far-access curve for items beyond it (the far channel, only with far="term")."""
        lh, lv, ld, lm = self.tables()
        nw = getattr(d, "n_walk", None)
        use_far = far and self.far == "term" and nw is not None

        def base_of(sl):
            if self.speed == "abs":
                ls = tf.table_eval(lv, d.vi[sl], d.vf[sl])
            elif self.speed == "rel":
                ls = tf.table_eval(lv, d.ri[sl], d.rf[sl])
            else:
                ls = torch.zeros_like(d.hf[sl])
            return tf.table_eval(lh, d.hi[sl], d.hf[sl]) + ls

        ws = slice(0, nw) if nw is not None else slice(None)   # walking items (all items when the data has no far split)
        flat = d.it_mode[ws] * len(D_KNOTS) + d.di[ws]
        lw = gather(ld, flat) * (1 - d.df[ws]) + gather(ld, flat + 1) * d.df[ws]
        walk = self._cell_max(base_of(ws) + lw, d, ws)
        farm = None
        if use_far:
            fs = slice(nw, None)
            lf = tf.table_eval(self.far_table(), d.fi[fs], d.ff[fs])
            farm = self._cell_max(base_of(fs) + lf, d, fs)
        return walk, farm, lm

    def _area(self, C, d):
        """Area (block group / tract / county) means of the cell values C [ncell, k]."""
        if self.weights_from == "cells":
            return torch.zeros((d.nb + 1, C.shape[1]), device=C.device).index_add(0, d.cell_orig, d.cell_pop[:, None] * C)[: d.nb] / d.pop.clamp(min=1.0)[:, None]
        Ce = torch.cat([C, torch.zeros((1, C.shape[1]), device=C.device)])
        return torch.zeros((d.nb + 1, C.shape[1]), device=C.device).index_add(0, d.blk_orig, d.blk_pop[:, None] * _rows(Ce, d.blk_cell))[: d.nb] / d.blk_den.clamp(min=1.0)[:, None]

    def etas(self, d, X=None):
        cm, cmf, lm = self.access(d)
        cols = lambda c: torch.stack([(c * torch.exp(lm)[None, :]).max(dim=1).values, c[:, 0], c[:, 2], c[:, 1]], 1)  # noqa: E731  TASKS order
        C = cols(cm)
        A = self._area(C, d)
        e = self.alpha[None, :] + torch.log1p(A / torch.exp(self.leps)[None, :])
        if cmf is not None:   # far access: its own term, so the walking curve is fitted on walking distances only
            Af = self._area(cols(cmf), d)
            e = e + torch.exp(self.lthf)[None, :] * torch.log1p(Af / torch.exp(self.lepsf)[None, :])
        if self.dest:
            Aw = torch.zeros((d.nbg_all, 4), device=C.device).index_add(0, d.cell_bg_all, C * d.cell_jobs[:, None]) / d.bg_jobs.clamp(min=1.0)[:, None]
            D = torch.zeros((d.nb, 4), device=C.device).index_add(0, d.fl_o, d.fl_wt[:, None] * _rows(Aw, d.fl_w))
            e = e + (torch.exp(self.ltheta) * self.dest_mask.to(C.device))[None, :] * torch.log1p(D / torch.exp(self.leps_d)[None, :]) * d.has_flows[:, None]
        if X is not None and X.shape[1]:
            e = e + X @ self.delta.T
        return {task: e[:, k] for k, task in enumerate(TASKS)}

    def penalty(self, smooth=1.0, ridge=1e-5):
        lh, lv, ld, lm = self.tables()
        d2 = lambda t: ((t[..., 2:] - 2 * t[..., 1:-1] + t[..., :-2]) ** 2).sum()  # noqa: E731
        pen = smooth * 1e-6 * (d2(lh) + d2(lv) + d2(ld)) + ridge * (self.delta ** 2).sum() + self.mridge * (lm[:2] ** 2).sum()
        if self.far == "term":
            pen = pen + smooth * 1e-6 * d2(self.far_table())
        if self.steep:
            dev = lh.device
            slope2 = lambda t, x: ((torch.diff(t, dim=-1) ** 2) / torch.tensor(np.diff(x), dtype=torch.float32, device=dev)).sum()  # noqa: E731  sum (dy/dx)^2 dx
            pen = pen + self.steep * (slope2(lh, np.log(H_KNOTS)) + slope2(ld, D_KNOTS / 1000.0) + (torch.exp(self.ltheta) ** 2).sum())
            if self.far == "term":
                pen = pen + self.steep * (slope2(self.far_table(), FAR_KNOTS / 1000.0) + (torch.exp(self.lthf) ** 2).sum())
        return pen


def task_ll(m, d, X=None):
    """{task: per-block-group per-worker log likelihood, task mask}."""
    E = m.etas(d, X); out = {}
    for k, task in enumerate(TASKS):
        mu = torch.sigmoid(E[task]).clamp(1e-7, 1 - 1e-7)
        if task == "total":
            ll = bb_loglik(d.y, d.n, mu, torch.exp(m.lkappa[k])) / d.n; mask = torch.ones_like(ll)
        else:
            n = d.n_mode.clamp(min=1.0)
            ll = bb_loglik(d.y_mode[task], n, mu, torch.exp(m.lkappa[k])) / n; mask = d.ok_mode[task]
        out[task] = (ll, mask)
    return out


def null_losses(d, w):
    """Per task: the loss of a no-transit model (intercept + dispersion), the normaliser of the multi-task loss."""
    out = {}
    for task in TASKS:
        a = torch.tensor([-4.0], device=DEV, requires_grad=True); lk = torch.tensor([math.log(5.0)], device=DEV, requires_grad=True)
        if task == "total":
            y, n, mask = d.y, d.n, torch.ones_like(d.y)
        else:
            y, n, mask = d.y_mode[task], d.n_mode.clamp(min=1.0), d.ok_mode[task]
        ww = w * mask; ww = ww / ww.sum()
        opt = torch.optim.Adam([a, lk], lr=0.05)
        for _ in range(800):
            opt.zero_grad(); mu = torch.sigmoid(a).clamp(1e-7, 1 - 1e-7).expand_as(y)
            L = -(ww * bb_loglik(y, n, mu, torch.exp(lk)) / n).sum(); L.backward(); opt.step()
        out[task] = (L.item(), a.detach(), lk.detach())
    return out


def multi_loss(m, d, w, nulls, X=None, task_w=None):
    tl = task_ll(m, d, X); tot = 0.0; parts = {}
    for task in TASKS:
        ll, mask = tl[task]; ww = w * mask; ww = ww / ww.sum()
        L = -(ww * ll).sum(); parts[task] = L
        tot = tot + (task_w or {}).get(task, 1.0) * L / nulls[task][0]
    return tot, parts


def fit3(m, d, w, nulls, X=None, steps=3000, lr=0.03, lbfgs=100, quiet=False, smooth=1.0, task_w=None, anneal=True):
    m.to(DEV)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr); t0 = time.time(); soft0 = m.soft
    for s in range(steps):
        if s == steps // 3:
            for g in opt.param_groups:
                g["lr"] = lr / 3
        if soft0 > 0 and anneal:   # anneal the soft maximum towards the max over the run (the last 20% at the max)
            m.soft = soft0 * max(0.0, 1 - s / (0.8 * steps))
        opt.zero_grad(); L, _ = multi_loss(m, d, w, nulls, X, task_w); (L + m.penalty(smooth)).backward(); opt.step()
        if not quiet and s % 1000 == 0:
            print(f"  adam {s:5d} loss {L.item():.6f} ({time.time() - t0:.0f}s)", flush=True)
    m.soft = 0.0
    if lbfgs:
        o2 = torch.optim.LBFGS(params, lr=0.5, max_iter=lbfgs, history_size=20, line_search_fn="strong_wolfe")

        def cl():
            o2.zero_grad(); t = multi_loss(m, d, w, nulls, X, task_w)[0] + m.penalty(smooth); t.backward(); return t
        o2.step(cl)
    with torch.no_grad():
        L, parts = multi_loss(m, d, w, nulls, X, task_w)
    return L.item(), {k: v.item() for k, v in parts.items()}


def curves3(m, d):
    with torch.no_grad():
        lh, lv, ld, lm = [t.detach().cpu().numpy() for t in m.tables()]
    kn = V_KNOTS if m.speed == "abs" else R_KNOTS
    return {"headway": {"x": H_KNOTS.tolist(), "y": np.exp(lh).tolist()}, "speed": {"x": kn.tolist(), "y": np.exp(lv).tolist(), "kind": m.speed},
            "mode": dict(zip(MODES, np.exp(lm).tolist())), "walk": {MODES[k]: {"x": D_KNOTS.tolist(), "y": np.exp(ld[k]).tolist()} for k in range(3)},
            "v_median": d.v_median.tolist(),
            "eps": torch.exp(m.leps).tolist(), "theta": torch.exp(m.ltheta).tolist(),
            **({"far": {"x": FAR_KNOTS.tolist(), "y": np.exp(m.far_table().detach().cpu().numpy()).tolist()}, "theta_far": torch.exp(m.lthf).tolist(),
                "eps_far": torch.exp(m.lepsf).tolist()} if m.far == "term" else {})}


def summary3(c):
    H = c["headway"]; W = c["walk"]
    h = lambda x: float(np.exp(np.interp(np.log(x), np.log(H["x"]), np.log(H["y"]))))  # noqa: E731
    out = {f"H{x:g}": h(x) for x in (1, 3, 5, 15, 30, 60, 120, 240)}
    for mo in MODES:
        for x in (100, 200, 400, 800, 1500):
            out[f"W{mo}{x}"] = float(np.interp(x, W[mo]["x"], W[mo]["y"]))
    sp = c["speed"]
    if sp["kind"] == "abs":
        out.update({"S5": float(np.interp(5, sp["x"], sp["y"])), "S15": float(np.interp(15, sp["x"], sp["y"]))})
    elif sp["kind"] == "rel":
        out.update({"S0.5": float(np.interp(0.5, sp["x"], sp["y"])), "S0.7": float(np.interp(0.7, sp["x"], sp["y"]))})
    out.update({"Mbus": c["mode"]["bus"], "Mtram": c["mode"]["tram"]})
    if sp["kind"] == "abs":   # rail vs bus and tram vs bus at each mode's median speed, in log points: the part speed explains and the part the mode does
        s = lambda v: math.log(float(np.interp(v, sp["x"], sp["y"])))  # noqa: E731
        vm = dict(zip(MODES, c["v_median"]))
        for mo in ("rail", "tram"):
            out[f"{mo}_vs_bus_speed"] = s(vm[mo]) - s(vm["bus"])
            out[f"{mo}_vs_bus_mode"] = math.log(c["mode"][mo]) - math.log(c["mode"]["bus"])
    out.update({f"theta_{t}": c["theta"][k] for k, t in enumerate(TASKS)})
    if "far" in c:   # far access (park-and-ride...): its curve relative to 2 km and its strength per task
        for x in (3000, 5000, 10000, 20000):
            out[f"F{x // 1000}k"] = float(np.interp(x, c["far"]["x"], c["far"]["y"]))
        out.update({f"thf_{t}": c["theta_far"][k] for k, t in enumerate(TASKS)})
    return out


def cv_total(d, mk, X, base_sd, nulls_all, steps=900, balanced=False, lam=0.5):
    """Metro-grouped 5-fold CV: held-out skill of the TOTAL transit task (comparable with torch_fit2) and of each mode task."""
    folds = tf.cv_folds(d)
    w = t2.weights(d, 0.5) * d.keep; w = w / w.sum()   # held-out skill always with the plain (metro-tempered population) weights, so variants compare
    wb = weights3(d, balanced, lam)
    ho = {t: np.zeros(d.nb) for t in TASKS}; ho0 = {t: np.zeros(d.nb) for t in TASKS}
    for f in range(5):
        tr = wb * torch.tensor((folds != f).astype(np.float32), device=DEV); tr = tr / tr.sum()
        nl = null_losses(d, tr)
        m = mk(); m.load_state_dict(base_sd); fit3(m, d, tr, nl, X, steps=steps, lr=0.01, lbfgs=60, quiet=True, anneal=False)
        with torch.no_grad():
            tl = task_ll(m, d, X)
            for k, task in enumerate(TASKS):
                ho[task][folds == f] = tl[task][0].cpu().numpy()[folds == f]
                _, a, lk = nl[task]
                y, n = (d.y, d.n) if task == "total" else (d.y_mode[task], d.n_mode.clamp(min=1.0))
                mu0 = torch.sigmoid(a).clamp(1e-7, 1 - 1e-7).expand_as(y)
                ho0[task][folds == f] = (bb_loglik(y, n, mu0, torch.exp(lk)) / n).cpu().numpy()[folds == f]
    wn = w.cpu().numpy(); out = {}
    for task in TASKS:
        mask = np.ones(d.nb) if task == "total" else d.ok_mode[task].cpu().numpy()
        ww = wn * mask
        out[task] = float(1 - (ww * ho[task]).sum() / (ww * ho0[task]).sum())
    return out


def weights3(d, balanced=False, lam=0.5):
    """Metro-tempered population weights; balanced: also each service class (best headway within 400 m: none / 60+ / 30-60 / 15-30 / <15 min) tempered
    by its population share ** -0.5, so the medium and bad service areas weigh more than their population alone and the frequent-service cores less."""
    w = t2.weights(d, lam) * d.keep; w = w / w.sum()   # lam 0: plain population weights (no down-weighting of the big metros)
    if balanced:
        cls = torch.tensor(d.service_class, device=DEV)
        share = torch.zeros(5, device=DEV).index_add(0, cls, w)
        w = w * share.clamp(min=1e-9)[cls] ** -0.5
        w = w / w.sum()
    return w


def multistart3(d, name, speed="abs", soft=0.0, controls="none", starts=5, steps=3000, cv=True, task_w=None, mridge=0.0, flat_below=3.0, balanced=False,
                dest_tasks=("total", "bus", "rail", "tram"), steep=0.0, weights_from="blocks", lam=0.5, far="drop", walk="free", fix_mode=False, fixed=None):
    X = d.controls(CONTROLS3[controls]) if controls != "none" else None
    w = weights3(d, balanced, lam)
    nulls = null_losses(d, w)
    mk = lambda: Model3(X.shape[1] if X is not None else 0, speed=speed, soft=soft, mridge=mridge, flat_below=flat_below, dest_tasks=dest_tasks,  # noqa: E731
                        steep=steep, weights_from=weights_from, dest=bool(dest_tasks), far=far, walk=walk, fix_mode=fix_mode, fixed=fixed)
    runs = []
    for sd in range(starts):
        m = mk()
        if sd:
            rng = np.random.default_rng(200 + sd)
            with torch.no_grad():
                for prm in (m.uh, m.uv, m.ud, m.uf):
                    prm.copy_(torch.tensor(rng.normal(-1.0, 1.5, prm.shape), dtype=torch.float32))
                m.lmode.copy_(torch.tensor(rng.normal(0.0, 0.7, 2), dtype=torch.float32))   # bus / tram may start above or below rail
        L, parts = fit3(m, d, w, nulls, X, steps=steps, quiet=True, task_w=task_w)
        c = curves3(m, d)
        runs.append({"seed": sd, "loss": L, "parts": parts, "summary": summary3(c), "curves": c, "_sd": {k: v.detach().clone() for k, v in m.state_dict().items()}})
        sm = runs[-1]["summary"]
        print(f"  [{name}] start {sd}: loss {L:.6f} " + " ".join(f"{k} {sm[k]:.3f}" for k in sm if k in ("H3", "H5", "H15", "H60", "H120", "Wbus200", "Wbus400", "Wtram400", "Wrail400", "Wrail800", "Mbus", "Mtram", "S5", "S15", "theta_total", "theta_bus", "theta_rail",
                                                                                            "rail_vs_bus_speed", "rail_vs_bus_mode", "tram_vs_bus_speed", "tram_vs_bus_mode", "F5k", "F10k", "F20k", "thf_total")), flush=True)
    keys = runs[0]["summary"].keys()
    spread = {k: {"min": min(r["summary"][k] for r in runs), "max": max(r["summary"][k] for r in runs)} for k in keys}
    out = {"name": name, "speed": speed, "soft": soft, "controls": controls, "mridge": mridge, "flat_below": flat_below, "balanced": balanced, "steep": steep, "weights_from": weights_from, "lam": lam, "far": far, "walk": walk, "fix_mode": fix_mode, "fixed": fixed,
           "dest_tasks": list(dest_tasks), "runs": runs, "spread": spread, "loss_range": [min(r["loss"] for r in runs), max(r["loss"] for r in runs)]}
    best = min(range(len(runs)), key=lambda i: runs[i]["loss"])
    if cv:
        out["cv"] = cv_total(d, mk, X, runs[best]["_sd"], nulls, balanced=balanced, lam=lam)
    torch.save(runs[best]["_sd"], RUNS / f"{name}.pt")
    for r in runs:
        r.pop("_sd", None)
    (RUNS / f"{name}.json").write_text(json.dumps(out, indent=1))
    wide = sorted(spread.items(), key=lambda kv: -(kv[1]["max"] - kv[1]["min"]))[:8]
    print(f"[{name}] loss {out['loss_range'][0]:.6f}-{out['loss_range'][1]:.6f}; widest: " + ", ".join(f"{k} {v['min']:.2f}-{v['max']:.2f}" for k, v in wide)
          + (f"; cv {json.dumps({k: round(v, 4) for k, v in out['cv'].items()})}" if cv else ""), flush=True)
    return out


def main():
    e = sys.argv[1]
    d = Data3()
    if e == "multitask":
        for name, kw in (("h3_mt_flat3_mridge", dict(mridge=0.005)), ("h3_mt_flat3_mridge_desttotal", dict(mridge=0.005, dest_tasks=("total",))),
                         ("h3_mt_flat3_mridge_balanced", dict(mridge=0.005, balanced=True)), ("h3_mt_flat3_mridge_structural", dict(mridge=0.005, controls="structural"))):
            multistart3(d, name, speed="abs", starts=4, steps=3500, **kw)
    elif e == "plan":   # 2026-11-12 plan, in priority order (block 1: main fits; 2: mode / speed / literature tests; 3: robustness; 4: if time is left)
        C = dict(speed="abs", starts=4, steps=3500, mridge=0.0, balanced=True, weights_from="cells")
        runs = [
            ("w2_dest", {}), ("pr_dest", dict(far="term")), ("w2_nodest", dict(dest_tasks=())), ("pr_nodest", dict(far="term", dest_tasks=())),
            ("w2_dest_steep1e-4", dict(steep=1e-4)), ("w2_dest_steep1e-3", dict(steep=1e-3)),
            ("w2_dest_nospeed", dict(speed="none")), ("w2_dest_equalmodes", dict(fix_mode=True)), ("w2_dest_relspeed", dict(speed="rel")),
            ("w2_dest_sharedwalk", dict(walk="shared")), ("w2_dest_literature", dict(fixed="literature", starts=1)), ("w2_dest_step400800", dict(walk="step")),
            ("w2_dest_tract", dict(level="tract")), ("w2_dest_county", dict(level="county", balanced=False)), ("w2_dest_structural", dict(controls="structural")),
            ("w2_dest_commutekm", dict(controls="commute_km")), ("w2_dest_plainweights", dict(balanced=False, lam=0.0)), ("w2_dest_noflat", dict(flat_below=0.0)),
            ("w2_dest_odlos", dict(controls="od_los")), ("pr_dest_tract", dict(far="term", level="tract")),
        ]
        only = sys.argv[2:]
        agg = {}
        for tag, kw in runs:
            if only and tag not in only:
                continue
            kw = {**C, **kw}; level = kw.pop("level", None)
            if level and level not in agg:
                agg[level] = AggData3(d, level)
            multistart3(agg[level] if level else d, f"plan_{tag}", **kw)
    elif e == "metros":   # 2026-11-13: the plan's main fit (w2_dest) in each big metro alone, against the national fit scored on the same metro.
        # Own fit: the metro's own curves (in-sample, an upper bound of what metro curves add). National: plan_w2_dest's curves, links and all, unrefitted.
        # Tram task left out where the metro has hardly any light-rail commuters (its loss would be noise).
        C = dict(speed="abs", steps=3500, mridge=0.0, balanced=True, weights_from="cells")
        extra = ["42660", "38900", "33460"]   # Seattle, Portland, Minneapolis: transit metros just below the 12 biggest
        metros = [m for m in d.metros[:12]] + [m for m in extra if m not in d.metros[:12]]
        only = sys.argv[2:]
        keep0 = d.keep.clone()
        nat = Model3(0, **{k: v for k, v in C.items() if k in ("speed", "mridge", "weights_from")}); nat.load_state_dict(torch.load(RUNS / "plan_w2_dest.pt")); nat.to(DEV)
        for cb in metros:
            if only and cb not in only:
                continue
            inm = torch.tensor((d.cbsa == cb).astype(np.float32), device=DEV)
            d.keep = keep0 * inm
            tram_n = float((d.y_mode["tram"] * d.ok_mode["tram"] * d.keep).sum())
            tw = None if tram_n >= 1000 else {"tram": 0.0}
            w = weights3(d, True, 0.5); nl = null_losses(d, w)
            with torch.no_grad():
                Ln, pn = multi_loss(nat, d, w, nl, None, tw)
            out = multistart3(d, f"metro_{cb}", starts=2, cv=False, task_w=tw, **C)
            out["metro"] = cb; out["tram_commuters"] = tram_n; out["task_w"] = tw
            out["block_groups"] = int(d.keep.sum().item()); out["workers"] = float((d.n * d.keep).sum())
            out["national_loss"] = Ln.item(); out["national_parts"] = {k: v.item() for k, v in pn.items()}
            out["nulls"] = {k: v[0] for k, v in nl.items()}
            (RUNS / f"metro_{cb}.json").write_text(json.dumps(out, indent=1))
            print(f"[metro {cb}] own {out['loss_range'][0]:.4f} national {Ln.item():.4f} tram_w {tw}", flush=True)
        d.keep = keep0
    elif e == "steepgrid":   # 2026-11-13: the steepness cost between 1e-4 and 1e-3 (the plan's two points), 6 random starts each, picked by held-out skill
        C = dict(speed="abs", starts=6, steps=3500, mridge=0.0, balanced=True, weights_from="cells")
        only = sys.argv[2:]
        for st in (1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3):
            tag = f"steepgrid_{st:g}"
            if only and tag not in only:
                continue
            multistart3(d, tag, steep=st, **C)
    elif e == "metros_nat":   # the fair comparison for `metros`: national curves, links and work-end strengths kept; only each task's intercept and
        # dispersion refitted on the metro (its own level of transit use). Added to metro_<cbsa>.json as national_refit_loss.
        keep0 = d.keep.clone()
        for p in sorted(RUNS.glob("metro_*.json")):
            o = json.loads(p.read_text()); cb = o["metro"]; tw = o.get("task_w")
            d.keep = keep0 * torch.tensor((d.cbsa == cb).astype(np.float32), device=DEV)
            w = weights3(d, True, 0.5); nl = null_losses(d, w)
            nat = Model3(0, speed="abs", mridge=0.0, weights_from="cells"); nat.load_state_dict(torch.load(RUNS / "plan_w2_dest.pt")); nat.to(DEV)
            for prm in nat.parameters():
                prm.requires_grad_(False)
            nat.alpha.requires_grad_(True); nat.lkappa.requires_grad_(True)
            with torch.no_grad():   # the curves are fixed: the item and cell values need computing only once
                cm, _, lm = nat.access(d)
            opt = torch.optim.Adam([nat.alpha, nat.lkappa], lr=0.05)
            orig_access = nat.access
            nat.access = lambda dd, far=True: (cm, None, lm)   # noqa: E731
            for _ in range(600):
                opt.zero_grad(); L, _ = multi_loss(nat, d, w, nl, None, tw); L.backward(); opt.step()
            with torch.no_grad():
                L, parts = multi_loss(nat, d, w, nl, None, tw)
            nat.access = orig_access
            o["national_refit_loss"] = L.item(); o["national_refit_parts"] = {k: v.item() for k, v in parts.items()}
            p.write_text(json.dumps(o, indent=1))
            print(f"[metro {cb}] own {o['loss_range'][0]:.4f} national {o['national_loss']:.4f} national+intercepts {L.item():.4f}", flush=True)
        d.keep = keep0
    elif e == "steep":   # cost of steepness, walking access only (max 2 km); pick the strength by the held-out skill. No mridge: the speed / mode split from the data
        for st in (0.0, 1e-5, 1e-4, 3e-4, 1e-3):   # the balanced fit's curves cost ~250 at strength 1; its loss is ~3.48
            multistart3(d, f"h3_cells_balanced_steep{st:g}_dest_w2", speed="abs", starts=4, steps=3500, mridge=0.0, balanced=True, steep=st, weights_from="cells")
    elif e == "pipeline":   # area score = cell-population-weighted mean of the cells' scores. w2: walking access only (max 2 km); pr: + park-and-ride (2-20 km,
        st = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0   # its own curve and term). With / without the work-end term; commute length as a variant
        for fartag, far in (("w2", "drop"), ("pr", "term")):
            for tag, kw in (("dest", {}), ("nodest", dict(dest_tasks=())), ("dest_commutekm", dict(controls="commute_km"))):
                if st and tag == "dest" and fartag == "w2":
                    continue   # = the steep sweep's run
                multistart3(d, f"h3_cells_balanced_steep{st:g}_{tag}_{fartag}", speed="abs", starts=4, steps=3500, mridge=0.0, balanced=True, steep=st,
                            weights_from="cells", far=far, **kw)
    elif e == "checks":   # how much the modelling choices and the placement of service shift the curves (work-end term on, walking access only)
        st = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
        for tag, kw in (("noflat", dict(flat_below=0.0)), ("plainweights", dict(balanced=False, lam=0.0)), ("structural", dict(controls="structural"))):
            kw = {"balanced": True, **kw}
            multistart3(d, f"h3_cells_steep{st:g}_{tag}_w2", speed="abs", starts=4, steps=3500, mridge=0.0, steep=st, weights_from="cells", **kw)
    elif e == "levels":   # the same fit on tracts and counties: what the block-group detail adds.  levels <steep>
        st = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
        # counties: plain weights -- almost every county's dominant service class is "no stop within 400 m", so balancing would hang on a handful of counties
        for level, bal in (("tract", True), ("county", False)):
            for fartag, far in (("w2", "drop"), ("pr", "term")):
                multistart3(AggData3(d, level), f"h3_cells_{'balanced' if bal else 'plain'}_steep{st:g}_dest_{fartag}_{level}", speed="abs", starts=4, steps=3500,
                            mridge=0.0, balanced=bal, steep=st, weights_from="cells", far=far)

if __name__ == "__main__":
    main()
