"""Flexible transit score fitted to observed transit use: the model of the US score fit (usa_flex/fit/torch_fit4.py, final form 2026-11) for any
set of areas with outcome units, populated H3 cells and service items. Used by the city science flex study; torch only (no geo packages).

Model (as the final US form: FIT4_DECAY=service, FIT4_H_FLOOR=3, FIT4_SHAPE_S=mono, FIT4_SPEED_CLIP=1, one walking curve for every item):
  item      value = M(mode) * H(headway) * S(speed) * W(walk distance; headway, speed)
              H(h) = log-logistic 1 / (1 + (h / h50)^kh), normalised to 1 at the minimum headway (3 min) and flat below it
              S(v) = free monotone curve in log v between fixed knots (value 1 at the top knot), item speeds clipped per mode
              W(d) = log-logistic in log d, 1 at the floor (2 H3 cell diameters) and flat below it; its slope scales with the service:
                     k = exp(lkw + c_h log(h / 10 min) + c_v log(v / 20 km/h))
              M    = (bus, tram, rail = 1); with TRAM_MID the tram factor is not fitted but held halfway between bus and rail
  cell      A_c = max over the cell's items
  unit      A_u = population-weighted mean over the unit's populated cells (cells without stops count 0)
  link      logit p = alpha_g + b_g * log(1 + A_u / eps) + X delta     (b_g = 1 unless the per-area slope is freed; g = area)
  loss      weighted beta-binomial per counted person (dispersion per area), each area weighted by its people ** (1 - lam)
Fitting rules (user): speed curve first without mode factors (all 1), frozen, then the mode factors; no literature values enter; the curves
are flat below their floors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as Fn

H_FLOOR_MIN = 3.0
S_KNOTS = [5.0, 8.0, 12.0, 18.0, 27.0, 40.0, 60.0, 100.0]
H3_EDGE_M = {9: 200.8, 10: 75.9, 11: 28.7, 12: 10.8}
MODES = ("bus", "tram", "rail")
CURVE_PARAMS = ("lh50", "lkh", "th_s", "l50w", "lkw", "c_h", "c_v", "lmode")
SPEED_PARAMS = ("th_s", "c_v")   # frozen after the speed-first stage (as the US sequence: FIT4_FREEZE=th_s,c_v)
#: default of FlexModel(tram_mid=None): tram factor = (bus + rail) / 2 instead of its own parameter (for areas with too little tram service to
#: fit it; user 2026-11-22)
TRAM_MID = False


def walk_floor_m(res: int) -> float:
    """Walking distance below which the walk curve is flat: 2 cell diameters (4 x the average edge) of the H3 resolution."""
    return 4.0 * H3_EDGE_M[res]


def _ll(x):
    return -Fn.softplus(x)


def _mono_up(lx, t, sl):
    """Increasing log curve: 0 at and above the top knot, falling with slopes sl_k below it (first segment open)."""
    out = torch.zeros_like(lx)
    for k in range(len(t) - 1):
        lo_len = (t[k + 1] - t[k]) if k > 0 else float("inf")
        out = out - sl[k] * (t[k + 1] - lx).clamp(min=0.0, max=lo_len)
    return out


def bb_loglik(y, n, mu, kappa):
    """Beta-binomial log-likelihood of y of n with mean mu and concentration kappa (non-integer counts allowed)."""
    a = kappa * mu; b = kappa * (1 - mu)
    return (torch.lgamma(y + a) + torch.lgamma(n - y + b) - torch.lgamma(n + kappa) - torch.lgamma(a) - torch.lgamma(b) + torch.lgamma(kappa)
            + torch.lgamma(n + 1) - torch.lgamma(y + 1) - torch.lgamma(n - y + 1))


@dataclass
class FitData:
    """Items, cells and units of one or more areas (group = area). Arrays are numpy until `tensors()`."""
    it_cell: np.ndarray
    it_mode: np.ndarray
    it_h: np.ndarray
    it_v: np.ndarray
    it_d: np.ndarray
    cell_unit: np.ndarray
    cell_pop: np.ndarray
    y: np.ndarray
    n: np.ndarray
    weight: np.ndarray
    group: np.ndarray
    fold: np.ndarray
    X: np.ndarray
    group_names: list = field(default_factory=list)
    control_names: list = field(default_factory=list)
    unit_ids: np.ndarray | None = None

    @property
    def n_unit(self) -> int:
        return len(self.y)

    @property
    def n_cell(self) -> int:
        return len(self.cell_unit)

    @property
    def n_group(self) -> int:
        return len(self.group_names)

    @staticmethod
    def concat(parts: list["FitData"]) -> "FitData":
        """Areas side by side: cell and unit indices offset, group ids renumbered in order (controls by name, missing = NaN)."""
        names = list(dict.fromkeys(c for p in parts for c in p.control_names))
        co = uo = go = 0
        cols = {k: [] for k in ("it_cell", "it_mode", "it_h", "it_v", "it_d", "cell_unit", "cell_pop", "y", "n", "weight", "group", "fold", "X", "ids")}
        gnames = []
        for p in parts:
            cols["it_cell"].append(p.it_cell.astype(np.int64) + co); cols["cell_unit"].append(p.cell_unit.astype(np.int64) + uo)
            for k in ("it_mode", "it_h", "it_v", "it_d", "cell_pop", "y", "n", "weight", "fold"):
                cols[k].append(getattr(p, k))
            cols["group"].append(p.group.astype(np.int64) + go)
            X = np.full((p.n_unit, len(names)), np.nan, np.float64)
            for j, c in enumerate(p.control_names):
                X[:, names.index(c)] = p.X[:, j]
            cols["X"].append(X)
            cols["ids"].append(p.unit_ids if p.unit_ids is not None else np.arange(p.n_unit).astype(str))
            gnames += p.group_names
            co += p.n_cell; uo += p.n_unit; go += p.n_group
        cat = {k: np.concatenate(v) for k, v in cols.items()}
        return FitData(cat["it_cell"], cat["it_mode"], cat["it_h"], cat["it_v"], cat["it_d"], cat["cell_unit"], cat["cell_pop"], cat["y"], cat["n"],
                       cat["weight"], cat["group"], cat["fold"], cat["X"], gnames, names, cat["ids"])

    def select_groups(self, keep: list[str]) -> "FitData":
        """The named areas only (re-indexed)."""
        gi = [self.group_names.index(g) for g in keep]
        um = np.isin(self.group, gi)
        return self.select_units(um, group_names=keep, group_map={g: i for i, g in enumerate(gi)})

    def select_units(self, um: np.ndarray, group_names=None, group_map=None) -> "FitData":
        """The units in mask `um`, their cells and items (re-indexed)."""
        new_u = np.full(self.n_unit, -1, np.int64); new_u[um] = np.arange(int(um.sum()))
        cm = um[self.cell_unit]
        new_c = np.full(self.n_cell, -1, np.int64); new_c[cm] = np.arange(int(cm.sum()))
        im = cm[self.it_cell]
        g = self.group[um]
        if group_map is not None:
            g = np.array([group_map[int(x)] for x in g], np.int64)
        return FitData(new_c[self.it_cell[im]], self.it_mode[im], self.it_h[im], self.it_v[im], self.it_d[im], new_u[self.cell_unit[cm]],
                       self.cell_pop[cm], self.y[um], self.n[um], self.weight[um], g, self.fold[um], self.X[um],
                       list(group_names or self.group_names), list(self.control_names), None if self.unit_ids is None else self.unit_ids[um])

    def tensors(self, dev="cpu") -> "Tensors":
        return Tensors(self, dev)


class Tensors:
    """FitData on the device; items sorted by cell (the per-cell maximum is a segment maximum)."""

    def __init__(self, d: FitData, dev):
        o = np.argsort(d.it_cell, kind="stable")
        T = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)  # noqa: E731
        self.it_cell, self.it_mode = T(d.it_cell[o], torch.long), T(d.it_mode[o], torch.long)
        self.it_h, self.it_v, self.it_d = T(d.it_h[o]), T(d.it_v[o]), T(d.it_d[o])
        self.cell_unit, self.cell_pop = T(d.cell_unit, torch.long), T(d.cell_pop)
        self.y, self.n = T(d.y, torch.float64), T(np.maximum(d.n, 1e-6), torch.float64)
        self.group = T(d.group, torch.long)
        self.n_unit, self.n_cell, self.n_group = d.n_unit, d.n_cell, d.n_group
        self.unit_pop = torch.zeros(self.n_unit, device=dev).index_add(0, self.cell_unit, self.cell_pop).clamp(min=1e-9)
        self.dev = dev


class FlexModel(torch.nn.Module):
    """The model of the module docstring for `n_group` areas and `n_ctrl` controls."""

    def __init__(self, n_group: int, n_ctrl: int = 0, floor_m: float = walk_floor_m(11), no_mode: bool = False, slope_per_group: bool = False,
                 tram_mid: bool | None = None):
        super().__init__()
        self.tram_mid = TRAM_MID if tram_mid is None else bool(tram_mid)
        self.slope_norm = False   # training only (fit_sequence): city slopes relative to their geometric mean, which is held at 1
        P = lambda *v: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))  # noqa: E731
        self.floor_m = float(floor_m)
        self.lh50, self.lkh = P(math.log(15.0)), P(math.log(1.0))
        self.th_s = torch.nn.Parameter(torch.full((len(S_KNOTS) - 1,), math.log(math.expm1(0.3))))
        self.l50w, self.lkw = P(math.log(500.0)), P(math.log(2.0))
        self.c_h, self.c_v = P(0.0), P(0.0)
        self.lmode = P(-0.3, -0.2)   # bus, tram (rail = 1)
        if no_mode:
            self.lmode.data.zero_(); self.lmode.requires_grad_(False)
        self.alpha = torch.nn.Parameter(torch.full((n_group,), -2.0, dtype=torch.float64))
        self.lkappa = torch.nn.Parameter(torch.full((n_group,), math.log(5.0), dtype=torch.float64))
        self.leps = P(math.log(0.05))
        self.lslope = torch.nn.Parameter(torch.zeros(n_group))
        self.lslope.requires_grad_(slope_per_group)
        self.delta = torch.nn.Parameter(torch.zeros(n_ctrl, dtype=torch.float64))

    # ---- curves --------------------------------------------------------------------------------------------------------------------------
    def logH(self, h):
        f = lambda x: _ll(torch.exp(self.lkh) * (torch.log(x) - self.lh50))  # noqa: E731
        return f(h.clamp(min=H_FLOOR_MIN)) - f(torch.tensor(H_FLOOR_MIN, device=h.device))

    def logS(self, v):
        return _mono_up(torch.log(v.clamp(min=0.5)), [math.log(x) for x in S_KNOTS], Fn.softplus(self.th_s))

    def logW(self, d, h, v):
        k = torch.exp(self.lkw + self.c_h * torch.log(h.clamp(min=0.5) / 10.0) + self.c_v * torch.log(v.clamp(min=0.5) / 20.0))
        return _ll(k * (torch.log(d.clamp(min=self.floor_m)) - self.l50w)) - _ll(k * (math.log(self.floor_m) - self.l50w))

    def lm(self):
        z = torch.zeros(1, device=self.lmode.device)
        if self.tram_mid:   # tram halfway between bus and rail (= 1)
            bus = self.lmode[:1]
            return torch.cat([bus, torch.log((torch.exp(bus) + 1.0) / 2.0), z])
        return torch.cat([self.lmode, z])

    def item_logvalue(self, t: Tensors, ix=None):
        sl = slice(None) if ix is None else ix
        h, v, d, m = t.it_h[sl], t.it_v[sl], t.it_d[sl], t.it_mode[sl]
        return self.lm()[m] + self.logH(h) + self.logS(v) + self.logW(d, h, v)

    def cell_access(self, t: Tensors):
        """Per cell the best item value: winners found without gradients, then only they are recomputed with gradients."""
        if t.it_cell.numel() == 0:
            return torch.zeros(t.n_cell, device=t.dev)
        with torch.no_grad():
            la = self.item_logvalue(t)
            mx = torch.full((t.n_cell,), -float("inf"), device=t.dev).scatter_reduce(0, t.it_cell, la, reduce="amax", include_self=True)
            hit = la >= mx[t.it_cell]
            ix = torch.arange(la.numel(), device=t.dev)
            win = torch.full((t.n_cell,), -1, dtype=torch.long, device=t.dev).scatter_reduce(0, t.it_cell[hit], ix[hit], reduce="amax", include_self=True)
            ok = torch.nonzero(win >= 0).squeeze(1)
        return torch.zeros(t.n_cell, device=t.dev).scatter(0, ok, torch.exp(self.item_logvalue(t, win[ok])))

    def unit_access(self, t: Tensors):
        return torch.zeros(t.n_unit, device=t.dev).index_add(0, t.cell_unit, t.cell_pop * self.cell_access(t)) / t.unit_pop

    def log_slopes(self):
        return self.lslope - self.lslope.mean() if self.slope_norm else self.lslope

    def service_term(self, A, group):
        return (torch.exp(self.log_slopes())[group] * torch.log1p(A / torch.exp(self.leps))).double()

    def eta(self, t: Tensors, X=None, A=None):
        A = self.unit_access(t) if A is None else A
        e = self.alpha[t.group] + self.service_term(A, t.group)
        if X is not None and X.shape[1]:
            e = e + X @ self.delta
        return e

    def params(self) -> dict:
        g = lambda p: p.detach().cpu().double().numpy().round(5).tolist()  # noqa: E731
        with torch.no_grad():
            H = {h: float(torch.exp(self.logH(torch.tensor([float(h)])))) for h in (3, 5, 10, 15, 20, 30, 60)}
            S = {v: float(torch.exp(self.logS(torch.tensor([float(v)])))) for v in (5, 10, 15, 20, 30, 50, 80)}
            W = {d: float(torch.exp(self.logW(torch.tensor([float(d)]), torch.tensor([10.0]), torch.tensor([20.0])))) for d in (200, 400, 800, 1200, 1600)}
        return {"h50_min": math.exp(self.lh50.item()), "kh": math.exp(self.lkh.item()), "w50_m": math.exp(self.l50w.item()), "kw": math.exp(self.lkw.item()),
                "c_h": self.c_h.item(), "c_v": self.c_v.item(), "mode": dict(zip(MODES, [math.exp(x) for x in g(self.lm())])), "eps": math.exp(self.leps.item()),
                "alpha": g(self.alpha), "kappa": [math.exp(x) for x in g(self.lkappa)], "slope": [math.exp(x) for x in g(self.log_slopes())], "delta": g(self.delta),
                "curve_H_10min20kmh": H, "curve_S": S, "curve_W_at_10min_20kmh": W, "walk_floor_m": self.floor_m, "tram_mid": self.tram_mid}

    def freeze(self, names):
        for n in names:
            getattr(self, n).requires_grad_(False)


def group_weights(t: Tensors, people: np.ndarray, mask: np.ndarray, lam: float = 0.5) -> torch.Tensor:
    """Unit weights: its people, each area's total tempered to total ** (1 - lam) (lam 1 = every area equal), masked units 0; sums to 1."""
    w = np.where(mask, np.maximum(np.nan_to_num(people), 0.0), 0.0)
    g = t.group.cpu().numpy()
    tot = np.bincount(g, weights=w, minlength=t.n_group)
    f = np.where(tot > 0, tot ** (1 - lam) / np.where(tot > 0, tot, 1.0), 0.0)
    w = w * f[g]
    return torch.as_tensor(w / max(w.sum(), 1e-300), dtype=torch.float64, device=t.dev)


def unit_ll(m: FlexModel, t: Tensors, X=None, A=None):
    """Per unit the log-likelihood per counted person."""
    mu = torch.sigmoid(m.eta(t, X, A)).clamp(1e-7, 1 - 1e-7)
    return bb_loglik(t.y, t.n, mu, torch.exp(m.lkappa)[t.group]) / t.n


def null_ll(t: Tensors, w: torch.Tensor, steps: int = 300):
    """Per unit the log-likelihood (per person) of each area's own constant share and dispersion, fitted with weights w."""
    a = torch.full((t.n_group,), -2.0, dtype=torch.float64, device=t.dev, requires_grad=True)
    lk = torch.full((t.n_group,), math.log(5.0), dtype=torch.float64, device=t.dev, requires_grad=True)
    f = lambda: -(w * bb_loglik(t.y, t.n, torch.sigmoid(a[t.group]).clamp(1e-7, 1 - 1e-7), torch.exp(lk)[t.group]) / t.n).sum()  # noqa: E731
    opt = torch.optim.Adam([a, lk], lr=0.05)
    for _ in range(steps):
        opt.zero_grad(); L = f(); L.backward(); opt.step()
    o2 = torch.optim.LBFGS([a, lk], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")
    def cl():
        o2.zero_grad(); L = f(); L.backward(); return L
    o2.step(cl)
    with torch.no_grad():
        return bb_loglik(t.y, t.n, torch.sigmoid(a[t.group]).clamp(1e-7, 1 - 1e-7), torch.exp(lk)[t.group]) / t.n


def skill_by_group(ll, ll0, w, group, n_group: int) -> list:
    """Per area 1 - model loss / null loss (losses per person, weighted by w within the area); NaN for an area without weight."""
    g = group
    num = torch.zeros(n_group, dtype=torch.float64, device=w.device).index_add(0, g, -w * ll)
    den = torch.zeros(n_group, dtype=torch.float64, device=w.device).index_add(0, g, -w * ll0)
    return [float(1 - a / b) if b > 0 else float("nan") for a, b in zip(num.tolist(), den.tolist())]


def fit(m: FlexModel, t: Tensors, w, X=None, A=None, steps: int = 1500, lr: float = 0.03, polish: int = 60) -> float:
    """Adam (learning rate dropped at 2/3) then L-BFGS on the trainable parameters; returns the final weighted loss."""
    ps = [p for p in m.parameters() if p.requires_grad]
    loss = lambda: -(w * unit_ll(m, t, X, A)).sum()  # noqa: E731
    opt = torch.optim.Adam(ps, lr=lr)
    for s in range(steps):
        if s == (2 * steps) // 3:
            for gr in opt.param_groups:
                gr["lr"] = lr / 5
        opt.zero_grad(); L = loss(); L.backward(); opt.step()
    if polish:
        o2 = torch.optim.LBFGS(ps, lr=0.5, max_iter=polish, line_search_fn="strong_wolfe")
        def cl():
            o2.zero_grad(); L = loss(); L.backward(); return L
        try:
            o2.step(cl)
        except RuntimeError:
            pass
    with torch.no_grad():
        return float(loss())


def randomize(m: FlexModel, rng: np.random.Generator):
    """A random start of the curve parameters (wide ranges, no literature values)."""
    with torch.no_grad():
        m.lh50.fill_(math.log(rng.uniform(4, 60))); m.lkh.fill_(math.log(rng.uniform(0.3, 3)))
        m.l50w.fill_(math.log(rng.uniform(150, 2000))); m.lkw.fill_(math.log(rng.uniform(0.5, 5)))
        m.th_s.copy_(torch.tensor(np.log(np.expm1(rng.uniform(0.05, 1.0, m.th_s.numel()))), dtype=torch.float32))
        m.leps.fill_(math.log(rng.uniform(0.01, 0.3)))


def fit_sequence(t: Tensors, w, X=None, n_ctrl=0, floor_m=walk_floor_m(11), starts=3, steps=1500, seed=0, slope_per_group=False, log=None,
                 fixed: dict | None = None, slope_norm: bool = False):
    """Speed first, then mode (user rule): stage 1 without mode factors from `starts` random starts (best kept), stage 2 from it with the
    speed curve frozen and the mode factors free. `fixed`: parameters held at the given values in both stages (a sensitivity test, e.g.
    {"lkh": log(4)}). `slope_norm` (with slope_per_group): the city slopes' geometric mean held at 1 while fitting, so the curves carry the
    scale of the service effect in the average area (with free slopes and a small eps the curves are otherwise only defined up to a power);
    the returned model has the normalised slopes written into lslope. Returns (stage-2 model, {"stage1": loss, "stage2": loss})."""

    def hold(m):
        with torch.no_grad():
            for n, v in (fixed or {}).items():
                getattr(m, n).fill_(v)
        m.freeze(tuple(fixed or ()))
        m.slope_norm = bool(slope_norm and slope_per_group)

    rng = np.random.default_rng(seed)
    best = None
    for s in range(starts):
        m = FlexModel(t.n_group, n_ctrl, floor_m, no_mode=True, slope_per_group=slope_per_group).to(t.dev)
        if s:
            randomize(m, rng)
        hold(m)
        L = fit(m, t, w, X, steps=steps)
        if log:
            log(f"  stage 1 start {s}: loss {L:.6f}")
        if best is None or L < best[0]:
            best = (L, m)
    L1, m1 = best
    m2 = FlexModel(t.n_group, n_ctrl, floor_m, no_mode=False, slope_per_group=slope_per_group).to(t.dev)
    m2.load_state_dict({k: v for k, v in m1.state_dict().items() if k != "lmode"}, strict=False)
    m2.freeze(SPEED_PARAMS)
    hold(m2)
    L2 = fit(m2, t, w, X, steps=steps)
    if log:
        log(f"  stage 2 (speed frozen, mode free): loss {L2:.6f}")
    with torch.no_grad():
        m2.lslope.copy_(m2.log_slopes())
    m2.slope_norm = False
    return m2, {"stage1": L1, "stage2": L2}


def refit_levels(m: FlexModel, t: Tensors, w, X=None, A=None, free=("alpha", "lkappa"), steps=400) -> FlexModel:
    """The same curves (and controls) on other units: only the named parameters refitted (the area's level -- the scale factor -- and its
    dispersion by default). Returns a new model; `m` is unchanged."""
    import copy

    m2 = copy.deepcopy(m)
    with torch.no_grad():
        if m2.alpha.numel() != t.n_group:
            m2.alpha = torch.nn.Parameter(torch.full((t.n_group,), float(m.alpha.mean()), dtype=torch.float64, device=t.dev))
            m2.lkappa = torch.nn.Parameter(torch.full((t.n_group,), float(m.lkappa.mean()), dtype=torch.float64, device=t.dev))
            m2.lslope = torch.nn.Parameter(torch.zeros(t.n_group, device=t.dev))
    for n, p in m2.named_parameters():
        p.requires_grad_(n in free)
    if A is None:
        with torch.no_grad():
            A = m2.unit_access(t)
    fit(m2, t, w, X, A=A, steps=steps, lr=0.05, polish=40)
    return m2


# ================================================================================================================================ the smooth model
#: steepness ranges of the smooth model's curves: wide enough for a gentle or a quick decay, narrow enough that no curve becomes a step
KH_RANGE, KV_RANGE, KW_RANGE = (0.5, 3.0), (0.5, 3.0), (0.8, 4.0)
V_TOP_KMH = 100.0   # the speed curve is 1 here (item speeds above it count as this)
SMOOTH_SPEED_PARAMS = ("lv50", "xv")   # frozen after the speed-first stage


def _bounded(x, lo_hi):
    return lo_hi[0] + (lo_hi[1] - lo_hi[0]) * torch.sigmoid(x)


class SmoothModel(FlexModel):
    """The model for a ranking of areas (city science revision, 2026-11-25). As `FlexModel`, with:

      * smooth curves, all log-logistic in the log of their argument with a bounded steepness: headway H (1 at the minimum headway, flat below
        it), speed S (increasing, 1 at 100 km/h), walking W (1 at the floor, flat below it; its slope does not depend on the service)
      * modes that cannot be out of order: rail = 1, tram = rail * a factor below 1, bus = tram * a factor below 1
      * the unit's service term is the population-weighted MEAN OF THE CELL SCORES  s_c = log(1 + A_c / eps)  over its cells (the score is
        computed per H3 cell, then averaged over whatever geometry the units are), not the log of the mean access
      * link     logit p = alpha_g + b_g * s_u + X delta     (b_g = 1 unless the per-area slope is freed)
    The score of a place for a ranking is  100 * s / P95(s)  with the shared curves and eps and no area-specific term.
    """

    def __init__(self, n_group: int, n_ctrl: int = 0, floor_m: float = walk_floor_m(11), no_mode: bool = False, slope_per_group: bool = False,
                 tram_mid: bool | None = None):
        torch.nn.Module.__init__(self)
        self.no_mode, self.tram_mid, self.slope_norm, self.floor_m = bool(no_mode), False, False, float(floor_m)
        P = lambda *v: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))  # noqa: E731
        self.lh50, self.xh = P(math.log(10.0)), P(0.0)
        self.lv50, self.xv = P(math.log(20.0)), P(0.0)
        self.l50w, self.xw = P(math.log(500.0)), P(0.0)
        self.lmode = P(1.0, 0.5)   # tram = sigmoid(a) of rail, bus = sigmoid(b) of tram
        if no_mode:
            self.lmode.requires_grad_(False)
        self.alpha = torch.nn.Parameter(torch.full((n_group,), -2.0, dtype=torch.float64))
        self.lkappa = torch.nn.Parameter(torch.full((n_group,), math.log(5.0), dtype=torch.float64))
        self.leps = P(math.log(0.05))
        self.lslope = torch.nn.Parameter(torch.zeros(n_group))
        self.lslope.requires_grad_(slope_per_group)
        self.delta = torch.nn.Parameter(torch.zeros(n_ctrl, dtype=torch.float64))

    def kh(self):
        return _bounded(self.xh, KH_RANGE)

    def kv(self):
        return _bounded(self.xv, KV_RANGE)

    def kw(self):
        return _bounded(self.xw, KW_RANGE)

    def logH(self, h):
        f = lambda x: _ll(self.kh() * (torch.log(x) - self.lh50))  # noqa: E731
        return f(h.clamp(min=H_FLOOR_MIN)) - f(torch.tensor(H_FLOOR_MIN, device=h.device))

    def logS(self, v):
        f = lambda x: _ll(self.kv() * (self.lv50 - torch.log(x)))  # noqa: E731   increasing in v
        return f(v.clamp(min=0.5, max=V_TOP_KMH)) - f(torch.tensor(V_TOP_KMH, device=v.device))

    def logW(self, d, h=None, v=None):
        f = lambda x: _ll(self.kw() * (torch.log(x) - self.l50w))  # noqa: E731
        return f(d.clamp(min=self.floor_m)) - f(torch.tensor(self.floor_m, device=d.device))

    def lm(self):
        z = torch.zeros(1, device=self.lmode.device)
        if self.no_mode:
            return torch.zeros(3, device=self.lmode.device)
        tram = Fn.logsigmoid(self.lmode[:1])
        return torch.cat([tram + Fn.logsigmoid(self.lmode[1:2]), tram, z])

    def cell_score(self, t):
        return torch.log1p(self.cell_access(t) / torch.exp(self.leps))

    def unit_access(self, t):
        """The unit's service term: the population-weighted mean of its cells' scores (cells without stops score 0)."""
        return torch.zeros(t.n_unit, device=t.dev).index_add(0, t.cell_unit, t.cell_pop * self.cell_score(t)) / t.unit_pop

    def service_term(self, S, group):
        return (torch.exp(self.log_slopes())[group] * S).double()

    def params(self) -> dict:
        g = lambda p: p.detach().cpu().double().numpy().round(5).tolist()  # noqa: E731
        with torch.no_grad():
            H = {h: float(torch.exp(self.logH(torch.tensor([float(h)])))) for h in (3, 5, 10, 15, 20, 30, 60)}
            S = {v: float(torch.exp(self.logS(torch.tensor([float(v)])))) for v in (5, 10, 15, 20, 30, 50, 80)}
            W = {d: float(torch.exp(self.logW(torch.tensor([float(d)])))) for d in (200, 400, 600, 800, 1200, 1600)}
        return {"h50_min": math.exp(self.lh50.item()), "kh": float(self.kh()), "v50_kmh": math.exp(self.lv50.item()), "kv": float(self.kv()),
                "w50_m": math.exp(self.l50w.item()), "kw": float(self.kw()), "mode": dict(zip(MODES, [math.exp(x) for x in g(self.lm())])),
                "eps": math.exp(self.leps.item()), "alpha": g(self.alpha), "kappa": [math.exp(x) for x in g(self.lkappa)],
                "slope": [math.exp(x) for x in g(self.log_slopes())], "delta": g(self.delta), "curve_H": H, "curve_S": S, "curve_W": W,
                "walk_floor_m": self.floor_m}


def randomize_smooth(m: SmoothModel, rng: np.random.Generator):
    """A random start of the smooth model's curve parameters (wide ranges, no literature values)."""
    with torch.no_grad():
        m.lh50.fill_(math.log(rng.uniform(4, 40))); m.xh.fill_(float(rng.normal(0, 1.5)))
        m.lv50.fill_(math.log(rng.uniform(10, 50))); m.xv.fill_(float(rng.normal(0, 1.5)))
        m.l50w.fill_(math.log(rng.uniform(200, 1500))); m.xw.fill_(float(rng.normal(0, 1.5)))
        m.leps.fill_(math.log(rng.uniform(0.01, 0.3)))


def fit_sequence_smooth(t: Tensors, w, X=None, n_ctrl=0, floor_m=walk_floor_m(11), starts=3, steps=1500, seed=0, slope_per_group=False,
                        slope_norm=False, log=None, fixed: dict | None = None):
    """Speed first, then mode (the study's rule) for the smooth model: stage 1 without mode factors from `starts` starts (best kept), stage 2
    from it with the speed curve frozen and the modes free. `slope_norm`: the area slopes' geometric mean held at 1 (the curves then carry the
    size of the service effect of the average area). `fixed`: parameters held at the given values in both stages (a sensitivity test, e.g. a curve
    taken from another fit). Returns (model, {"stage1": loss, "stage2": loss})."""

    def hold(m):
        with torch.no_grad():
            for n, v in (fixed or {}).items():
                getattr(m, n).fill_(v)
        m.freeze(tuple(fixed or ()))

    rng = np.random.default_rng(seed)
    best = None
    for s in range(starts):
        m = SmoothModel(t.n_group, n_ctrl, floor_m, no_mode=True, slope_per_group=slope_per_group).to(t.dev)
        m.slope_norm = bool(slope_norm and slope_per_group)
        if s:
            randomize_smooth(m, rng)
        hold(m)
        L = fit(m, t, w, X, steps=steps)
        if log:
            log(f"  stage 1 start {s}: loss {L:.6f}")
        if best is None or L < best[0]:
            best = (L, m)
    L1, m1 = best
    m2 = SmoothModel(t.n_group, n_ctrl, floor_m, no_mode=False, slope_per_group=slope_per_group).to(t.dev)
    m2.load_state_dict({k: v for k, v in m1.state_dict().items() if k != "lmode"}, strict=False)
    m2.freeze(SMOOTH_SPEED_PARAMS)
    hold(m2)
    m2.slope_norm = bool(slope_norm and slope_per_group)
    L2 = fit(m2, t, w, X, steps=steps)
    if log:
        log(f"  stage 2 (speed frozen, mode free): loss {L2:.6f}")
    with torch.no_grad():
        m2.lslope.copy_(m2.log_slopes())
    m2.slope_norm = False
    return m2, {"stage1": L1, "stage2": L2}
