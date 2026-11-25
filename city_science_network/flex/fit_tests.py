"""Fit tests of the flexible score on the city science cities (model and rules: `tools/studytools/flexfit.py`, the final US form).

    $TORCH_PYTHON flex/fit_tests.py <test> [city]        # results -> flex/results/<test>[_<city>].json; `report` writes flex/FIT_RESULTS.md

Cities: the curves are fitted on the five with many units (FIT: block groups, dissemination areas, manzanas, AGEBs); the five with 7-30 units
(SMALL) are only predicted. Every comparison is held out and gives each city its own level -- the scale factor: some cities use transit more
at the same service -- fitted on the city's own units: in the within-city tests on the training folds, in the transfer tests on the target.

Tests
  us             the US final curves (usa_flex seq_s8_nofuel: curves, mode factors, eps) and the map's default score, nothing refitted but each
                 city's level and dispersion (+ eps: for the default score always, its scale being arbitrary): within-city 5-fold skill
  own <city>     the city alone, speed first then mode: within-city 5-fold skill; then a fit on all its units (own_<city>.pt)
  transfer       each own fit predicting the other cities (the source's curves and eps; the target's level refitted, or level + slope)
  pooled         the five pooled, each city its level: within-city 5-fold skill; then a fit on all units (pooled.pt)
  loco <city>    pooled on the other four, predicting the city (its level refitted, or level + slope)
  controls       on the pooled curves (frozen): the treatments of the controls (CONTROL_MODES) and a forward selection of the
                 within-city standardised controls by held-out skill; the between-city share of each control's variance
  final          the pooled curves refitted with the selected controls (speed first, then mode): within-city skill, leave-one-city-out
  own_ctrl <city>  the city alone with its usual controls (USUAL: the US fit's set, those the city has), within-city standardised, speed
                 first then mode: within-city 5-fold skill; then a fit on all its units (own_ctrl_<city>.pt)
  transfer_ctrl  each own_ctrl fit's curves predicting the other cities: the target's level, dispersion and the coefficients of its own
                 usual controls fitted on its training folds (within-city 5-fold)
  common <part> [city]        one score for every city: the curves (headway, walk, speed, mode factors, eps) shared by the five cities,
                 each city its own level and its own coefficients of its usual controls (within-city standardised) -- nuisance terms
                 outside the score. Parts: cv (within-city 5-fold skill), full (fit on all units: common.pt), loco <city> (the
                 other four's curves; the city's level and control coefficients fitted on its training folds, within-city 5-fold)
  common_slope <part> [city]  the same with each city's own slope of the score (its ranking of places the same, its effect scaled)
  small <name>   the small cities again with the saved fit <name> (after a small city's data changed)
  report         flex/FIT_RESULTS.md from the results
Skill: 1 - loss / loss of the city's own constant share (beta-binomial per person, on the held-out units), per city; "mean" weighs the five
equally. Small cities (some publish only shares): after the city's level, R2 of the logit share, and the rank correlation of the service term.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "tools"))
from studytools import flexfit as ff  # noqa: E402

FIT = ["boston", "san_francisco", "toronto", "concepcion", "guadalajara"]
SMALL = ["beerseba", "gipuzkoa", "taipei", "hamburg", "andorra"]
OUT = Path(__file__).resolve().parent / "results"
US_RUN = HERE.parent / "usa_flex" / "fit" / "data_buf" / "torch_runs" / "g5_controls_w_cs_seq_s8_nofuel.pt"   # local fit output (not in the repo)
STEPS = int(os.environ.get("FIT_STEPS", "1000"))
STARTS = int(os.environ.get("FIT_STARTS", "2"))
LAM = 0.5
#: sensitivity test of the headway curve's steepness: FIT_FIX_KH=<k> holds it at k (common tests' outputs get the suffix _kh<k>)
FIX_KH = os.environ.get("FIT_FIX_KH")
#: FIT_FIX_H=us: the headway curve held at the US final fit's (its half value and steepness, our own fit on 234,000 block groups with the
#: full range of headways; the five cities have little service between 10 and 30 min). Suffix _hus
FIX_H = os.environ.get("FIT_FIX_H")
#: FIT_TRAM_MID=1: tram factor held halfway between bus and rail, not fitted (too little tram service: 1.6-7% of the people within
#: 400 m, Guadalajara's light rail is its only rail; user 2026-11-22); common tests' outputs get the suffix _tm
TRAM_MID = os.environ.get("FIT_TRAM_MID") == "1"
ff.TRAM_MID = TRAM_MID
#: FIT_SLOPE_NORM=1: with city slopes, their geometric mean held at 1 (the curves then carry the size of the service effect; suffix _sn)
SLOPE_NORM = os.environ.get("FIT_SLOPE_NORM") == "1"
#: FIT_CURVE_CITIES=a,b,c: the shared curves fitted only on these cities (those with a range of headways); the others are predicted with
#: them (their level, slope and control coefficients fitted on their own training folds). Suffix _cc<initials>
CURVE_CITIES = [c for c in os.environ.get("FIT_CURVE_CITIES", "").split(",") if c]
NON_CONTROL = ("GEOID", "share", "n", "y", "weight", "pop", "cells", "fold", "basis", "los_default")
CONTROL_MODES = {
    "none": "no controls",
    "pooled_raw_one_level": "controls standardised over the five cities pooled, one level for every city (the naive pooled fit)",
    "pooled_raw": "controls standardised over the five cities pooled, each city its own level",
    "within_z": "controls standardised within each city, each city its own level",
    "within_rank": "controls as within-city percentile ranks, each city its own level",
}
#: controls that are not shared across the cities (their meaning differs between countries): never in the pooled controls
NOT_UNIVERSAL = ("left_share",)
#: the usual controls of a single-city fit: the US fit's set (density, income, poverty, education, age), the ones the city has (>= half its units)
USUAL = ["log_density_1km", "log_income", "poverty_share", "university_share", "basic_education_share", "under18_share", "over65_share"]
#: extra controls of single cities (the left vote has a different meaning in each country, so only where tested for that city)
#: units with a default score below this are left out (feeds missing for whole municipalities)
NO_FEED_BELOW = {"guadalajara": 5.0}
CITY_EXTRA = {"guadalajara": ("left_share",)}
torch.set_num_threads(int(os.environ.get("FIT_THREADS", "5")))


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------------------- data
def load(city: str) -> tuple[ff.FitData, pd.DataFrame]:
    """The city's fit data (units with populated cells and a share; build_fit_data.py) and its unit table, in the same order."""
    z = np.load(HERE / city / "flex" / "fit" / "items.npz")
    u = pd.read_parquet(HERE / city / "flex" / "fit" / "units.parquet")
    ctrl = [c for c in u.columns if c not in NON_CONTROL]
    d = ff.FitData(z["it_cell"], z["it_mode"], z["it_h"], z["it_v"], z["it_d"], z["cell_unit"], z["cell_pop"], u["y"].to_numpy(float),
                   u["n"].to_numpy(float), u["weight"].to_numpy(float), np.zeros(len(u), np.int64), u["fold"].to_numpy(np.int64),
                   u[ctrl].to_numpy(float), [city], ctrl, u["GEOID"].astype(str).to_numpy())
    keep = (u["cells"].to_numpy() > 0) & np.isfinite(u["share"].to_numpy(float))
    if city in FIT:
        keep &= np.isfinite(u["n"].to_numpy(float)) & (u["n"].to_numpy(float) > 0)
    if city in NO_FEED_BELOW:  # units the city's feeds do not reach: their transit use is not service
        keep &= u["los_default"].to_numpy(float) >= NO_FEED_BELOW[city]
    return d.select_units(keep), u[keep].reset_index(drop=True)


def load_many(cities) -> tuple[ff.FitData, pd.DataFrame]:
    parts = [load(c) for c in cities]
    return ff.FitData.concat([p[0] for p in parts]), pd.concat([p[1] for p in parts], ignore_index=True)


def control_matrix(d: ff.FitData, mode: str, names=None) -> tuple[np.ndarray, list]:
    """Controls of `d` under one treatment (CONTROL_MODES). Within: a missing value (or a control the city lacks) = the city's mean;
    pooled: 0 (the pooled mean) and a missing indicator."""
    names = list(d.control_names if names is None else names)   # a requested control the data lack = a column of 0 (the city mean)
    if mode == "none" or not names:
        return np.zeros((d.n_unit, 0)), []
    cols, out = [], []
    for c in names:
        v = d.X[:, d.control_names.index(c)] if c in d.control_names else np.full(d.n_unit, np.nan)
        ok = np.isfinite(v)
        if mode.startswith("pooled"):
            if ok.sum() < 3:
                continue
            cols.append(np.where(ok, (v - np.nanmean(v)) / max(np.nanstd(v), 1e-9), 0.0)); out.append(c)
            if (~ok).any():
                cols.append((~ok).astype(float)); out.append(c + " (missing)")
            continue
        z = np.zeros(d.n_unit)
        for g in range(d.n_group):
            m = (d.group == g) & ok
            if m.sum() < 3 or np.std(v[m]) <= 0:
                continue
            if mode == "within_rank":
                z[m] = (pd.Series(v[m]).rank(pct=True).to_numpy() - 0.5) * np.sqrt(12.0)
            else:
                z[m] = (v[m] - v[m].mean()) / v[m].std()
        cols.append(z); out.append(c)
    return np.column_stack(cols), out


def Xt(X):
    return None if X is None or X.shape[1] == 0 else torch.as_tensor(X, dtype=torch.float64)


def weights(t, d):
    return ff.group_weights(t, d.weight, np.ones(d.n_unit, bool), LAM)


def _per_group(t, w, ll, ll0, n_group):
    a = torch.zeros(n_group, dtype=torch.float64).index_add(0, t.group, -w * ll).numpy()
    b = torch.zeros(n_group, dtype=torch.float64).index_add(0, t.group, -w * ll0).numpy()
    return a, b


def _r2_parts(t, w, mu, n_group):
    """Per city the weighted squared error of the predicted share and the weighted variance of the observed share (held-out units)."""
    p = (t.y / t.n).clamp(0, 1)
    sw = torch.zeros(n_group, dtype=torch.float64).index_add(0, t.group, w)
    mean = torch.zeros(n_group, dtype=torch.float64).index_add(0, t.group, w * p) / sw.clamp(min=1e-300)
    sse = torch.zeros(n_group, dtype=torch.float64).index_add(0, t.group, w * (p - mu) ** 2).numpy()
    sst = torch.zeros(n_group, dtype=torch.float64).index_add(0, t.group, w * (p - mean[t.group]) ** 2).numpy()
    return sse, sst


def _skills(names, num, den, sse=None, sst=None):
    sk = [float(1 - a / b) if b > 0 else None for a, b in zip(num, den)]
    out = {"skill": dict(zip(names, sk)), "mean": float(np.mean([s for s in sk if s is not None]))}
    if sse is not None:
        r2 = [float(1 - a / b) if b > 0 else None for a, b in zip(sse, sst)]
        out.update(r2=dict(zip(names, r2)), r2_mean=float(np.mean([r for r in r2 if r is not None])))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- scoring
def heldout_skill(train_fn, d: ff.FitData, X=None) -> dict:
    """Within-city spatial 5-fold cross-validation of a model refitted on each training part (`train_fn(train data, train X) -> model`):
    the held-out units' loss against the city's own constant share fitted on those units."""
    X = np.zeros((d.n_unit, 0)) if X is None else X
    num, den, sse, sst = (np.zeros(d.n_group) for _ in range(4))
    for k in range(5):
        tr, te = d.fold != k, d.fold == k
        if te.sum() == 0:
            continue
        m = train_fn(d.select_units(tr), X[tr])
        dte = d.select_units(te)
        t = dte.tensors(); w = weights(t, dte)
        with torch.no_grad():
            ll = ff.unit_ll(m, t, Xt(X[te]))
            mu = torch.sigmoid(m.eta(t, Xt(X[te])))
        a, b = _per_group(t, w, ll, ff.null_ll(t, w), d.n_group)
        e, v = _r2_parts(t, w, mu, d.n_group)
        num += a; den += b; sse += e; sst += v
        log(f"  fold {k}: {[round(float(1 - x / y), 4) if y > 0 else None for x, y in zip(a, b)]}")
    return _skills(d.group_names, num, den, sse, sst)


def frozen_cv(m0: ff.FlexModel, d: ff.FitData, A: torch.Tensor, X=None, free=("alpha", "lkappa", "delta"), one_level=False) -> dict:
    """Within-city 5-fold skill with the curves of m0 frozen (unit access A precomputed for every unit of d): per fold only `free` fitted
    (levels, dispersion, control coefficients; eps where named). `one_level`: every city the same level and dispersion."""
    X = np.zeros((d.n_unit, 0)) if X is None else X
    num, den, sse, sst = (np.zeros(d.n_group) for _ in range(4))
    G = 1 if one_level else d.n_group
    for k in range(5):
        tr, te = d.fold != k, d.fold == k
        m = ff.FlexModel(G, X.shape[1], floor_m=m0.floor_m)
        m.load_state_dict({n: v for n, v in m0.state_dict().items() if n not in ("alpha", "lkappa", "lslope", "delta")}, strict=False)
        for n, p in m.named_parameters():
            p.requires_grad_(n in free)
        dtr, dte = d.select_units(tr), d.select_units(te)
        ttr, tte = dtr.tensors(), dte.tensors()
        wtr, wte = weights(ttr, dtr), weights(tte, dte)
        gtr, gte = ttr.group, tte.group
        if one_level:
            ttr.group = torch.zeros_like(ttr.group)
        ff.fit(m, ttr, wtr, Xt(X[tr]), A=A[torch.as_tensor(tr)], steps=300, lr=0.05, polish=60)
        tte_eval = copy.copy(tte)
        if one_level:
            tte_eval.group = torch.zeros_like(gte)
        with torch.no_grad():
            ll = ff.unit_ll(m, tte_eval, Xt(X[te]), A[torch.as_tensor(te)])
            mu = torch.sigmoid(m.eta(tte_eval, Xt(X[te]), A[torch.as_tensor(te)]))
        a, b = _per_group(tte, wte, ll, ff.null_ll(tte, wte), d.n_group)
        e, v = _r2_parts(tte, wte, mu, d.n_group)
        num += a; den += b; sse += e; sst += v
    return _skills(d.group_names, num, den, sse, sst)


def target_skill(m: ff.FlexModel, d: ff.FitData, X=None, free=("alpha", "lkappa")) -> dict:
    """Curves (and control coefficients) of `m` on other cities: only their level and dispersion (or + slope) refitted on their units, then
    the loss against their constant share (both on all their units: 2-3 parameters per city)."""
    t = d.tensors(); w = weights(t, d)
    with torch.no_grad():
        A = m.unit_access(t)
    m2 = ff.refit_levels(m, t, w, Xt(X), A=A, free=free)
    with torch.no_grad():
        ll = ff.unit_ll(m2, t, Xt(X), A)
        mu = torch.sigmoid(m2.eta(t, Xt(X), A))
    sk = ff.skill_by_group(ll, ff.null_ll(t, w), w, t.group, t.n_group)
    e, v = _r2_parts(t, w, mu, t.n_group)
    return {g: {"skill": s_, "r2": float(1 - a / b)} for g, s_, a, b in zip(d.group_names, sk, e, v)}


def small_city_scores(s: np.ndarray, u: pd.DataFrame) -> dict:
    """A small city from its units' service term s (plus control term): the city's level on the logit share with slope 1 (the link), R2 of
    the logit share after the level, R2 with a free slope, the slope, and the rank correlation of s with the share."""
    from scipy.stats import spearmanr

    p = np.clip(u["share"].to_numpy(float), 0.005, 0.995)
    zt = np.log(p / (1 - p))
    w = np.maximum(np.nan_to_num(u["weight"].to_numpy(float)), 0) + 1e-9
    sst = np.average((zt - np.average(zt, weights=w)) ** 2, weights=w)
    a = np.average(zt - s, weights=w)
    b = np.polyfit(s, zt, 1, w=np.sqrt(w)) if np.ptp(s) > 0 else np.array([0.0, np.average(zt, weights=w)])
    return {"units": int(len(u)), "r2_level": float(1 - np.average((zt - a - s) ** 2, weights=w) / sst),
            "r2_level_slope": float(1 - np.average((zt - np.polyval(b, s)) ** 2, weights=w) / sst), "slope": float(b[0]),
            "spearman": float(spearmanr(s, zt).statistic) if np.ptp(s) > 0 else None, "basis": str(u["basis"].iloc[0])}


def small_all(m: ff.FlexModel, sel=None, A_of=None) -> dict:
    """small_city_scores for every small city (curves of m; controls `sel` within-city standardised with m's coefficients)."""
    out = {}
    for c in SMALL:
        d, u = load(c)
        t = d.tensors()
        with torch.no_grad():
            A = m.unit_access(t) if A_of is None else A_of(u)
            s = m.service_term(A, torch.zeros(t.n_unit, dtype=torch.long)).numpy()
        if sel:
            X, names = control_matrix(d, "within_z", sel)
            s = s + X @ m.delta.detach().numpy()
        out[c] = small_city_scores(s, u)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- models
def fixed_params() -> dict | None:
    """Parameters held in the fits (FIT_FIX_KH, FIT_FIX_H)."""
    out = {}
    if FIX_KH:
        out["lkh"] = float(np.log(float(FIX_KH)))
    if FIX_H == "us":
        p = json.loads(US_RUN.with_suffix(".json").read_text())
        p = p["runs"][p["best"]]["params"]
        out["lh50"], out["lkh"] = float(np.log(p["h50_min"])), float(np.log(p["kh"]))
    return out or None


def train_seq(d: ff.FitData, X=None, seed=0, slope_per_group=False):
    t = d.tensors(); w = weights(t, d)
    nX = 0 if X is None else X.shape[1]
    m, L = ff.fit_sequence(t, w, Xt(X), nX, ff.walk_floor_m(11), STARTS, STEPS, seed, slope_per_group=slope_per_group, log=log,
                           fixed=fixed_params(), slope_norm=SLOPE_NORM)
    m.losses = L
    return m


def us_model(n_group: int) -> ff.FlexModel:
    """The US final fit's curves, mode factors and eps (all-transit task); walk floor 303.5 m (the US res-10 floor)."""
    s = torch.load(US_RUN, map_location="cpu", weights_only=False)
    m = ff.FlexModel(n_group, 0, floor_m=ff.walk_floor_m(10), tram_mid=False)
    with torch.no_grad():
        for a, b in (("lh50", "lh50"), ("lkh", "lkh"), ("lkw", "lkw"), ("c_h", "c_h"), ("c_v", "c_v")):
            getattr(m, a).fill_(float(s[b].flatten()[0]))
        m.l50w.fill_(float(s["ld50"][0])); m.leps.fill_(float(s["leps"][0]))
        m.th_s.copy_(s["th_s"]); m.lmode.copy_(s["lmode"])
    return m


def save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    log(f"wrote {name}.json")


def load_model(name) -> ff.FlexModel:
    s = torch.load(OUT / f"{name}.pt", map_location="cpu", weights_only=False)
    m = ff.FlexModel(len(s["alpha"]), len(s["delta"]))
    m.load_state_dict(s)
    return m


# ---------------------------------------------------------------------------------------------------------------------------------- tests
def test_us():
    d, u = load_many(FIT)
    us = us_model(d.n_group)
    with torch.no_grad():
        A_us = us.unit_access(d.tensors())
    A_def = torch.as_tensor(u["los_default"].to_numpy(np.float32))
    dflt = ff.FlexModel(d.n_group, 0)
    res = {"us_level": frozen_cv(us, d, A_us, free=("alpha", "lkappa")),
           "us_level_eps": frozen_cv(us, d, A_us, free=("alpha", "lkappa", "leps")),
           "default_level_eps": frozen_cv(dflt, d, A_def, free=("alpha", "lkappa", "leps"))}
    for k, v in res.items():
        log(f"{k}: mean {v['mean']:.4f} R2 {v['r2_mean']:.4f} {v['skill']} {v['r2']}")
    res["small_us"] = small_all(us_model(1))
    # default score in the small cities: its eps from the five (fitted on all their units)
    t = d.tensors(); w = weights(t, d)
    ff.fit(dflt, t, w, A=A_def, steps=300, lr=0.05, polish=60)
    res["small_default"] = small_all(dflt, A_of=lambda uu: torch.as_tensor(uu["los_default"].to_numpy(np.float32)))
    res["default_eps"] = float(torch.exp(dflt.leps))
    save("us", res)


def test_own(city):
    d, _ = load(city)
    log(f"{city}: {d.n_unit} units, {d.n_cell} cells, {len(d.it_cell)} items")
    cv = heldout_skill(lambda dtr, Xtr: train_seq(dtr), d)
    log(f"{city} held-out skill {cv['skill']}")
    m = train_seq(d)
    OUT.mkdir(exist_ok=True)
    torch.save(m.state_dict(), OUT / f"own_{city}.pt")
    save(f"own_{city}", {"cv": cv, "params": m.params(), "losses": m.losses})


def test_transfer():
    res = {}
    targets = {c: load(c)[0] for c in FIT}
    sources = [("us", us_model(1))] + [(a, load_model(f"own_{a}")) for a in FIT if (OUT / f"own_{a}.pt").is_file()]
    for a, m in sources:
        row = {b: {"level": target_skill(m, targets[b])[b], "level_slope": target_skill(m, targets[b], free=("alpha", "lkappa", "lslope"))[b]}
               for b in FIT if b != a}
        row["small"] = small_all(m)
        res[a] = row
        log(f"{a} -> " + ", ".join(f"{b} {row[b]['level']['skill']:.4f}" for b in FIT if b != a))
    save("transfer", res)


def test_pooled():
    d, _ = load_many(FIT)
    cv = heldout_skill(lambda dtr, Xtr: train_seq(dtr), d)
    log(f"pooled held-out skill {cv['skill']}")
    m = train_seq(d)
    OUT.mkdir(exist_ok=True)
    torch.save(m.state_dict(), OUT / "pooled.pt")
    save("pooled", {"cv": cv, "params": m.params(), "losses": m.losses, "small": small_all(m), "groups": d.group_names})


def test_loco(city):
    others = [c for c in FIT if c != city]
    m = train_seq(load_many(others)[0])
    tgt = load(city)[0]
    res = {"level": target_skill(m, tgt)[city], "level_slope": target_skill(m, tgt, free=("alpha", "lkappa", "lslope"))[city],
           "params": m.params(), "trained_on": others}
    log(f"leave out {city}: {res['level']} (level + slope {res['level_slope']})")
    save(f"loco_{city}", res)


def universal(d: ff.FitData) -> list:
    return [c for c in d.control_names if c not in NOT_UNIVERSAL]


def usual(d: ff.FitData) -> list:
    extra = CITY_EXTRA.get(d.group_names[0], ()) if len(d.group_names) == 1 else ()
    return [c for c in USUAL + list(extra) if c in d.control_names and np.isfinite(d.X[:, d.control_names.index(c)]).mean() >= 0.5]


def test_own_ctrl(city):
    d, _ = load(city)
    X, names = control_matrix(d, "within_z", usual(d))
    log(f"{city}: {d.n_unit} units, controls {names}")
    cv = heldout_skill(lambda dtr, Xtr: train_seq(dtr, Xtr), d, X)
    log(f"{city} with controls held-out skill {cv['skill']} R2 {cv['r2']}")
    m = train_seq(d, X)
    OUT.mkdir(exist_ok=True)
    torch.save(m.state_dict(), OUT / f"own_ctrl_{city}.pt")
    save(f"own_ctrl_{city}", {"cv": cv, "controls": names, "coefficients": dict(zip(names, m.delta.detach().numpy().tolist())),
                              "params": m.params(), "losses": m.losses})


def test_transfer_ctrl():
    res = {}
    for a in FIT:
        if not (OUT / f"own_ctrl_{a}.pt").is_file():
            continue
        m = load_model(f"own_ctrl_{a}")
        row = {}
        for b in FIT:
            if b == a:
                continue
            d, _ = load(b)
            X, _ = control_matrix(d, "within_z", usual(d))
            with torch.no_grad():
                A = m.unit_access(d.tensors())
            r = frozen_cv(m, d, A, X)
            row[b] = {"skill": r["skill"][b], "r2": r["r2"][b]}
        res[a] = row
        log(f"{a} -> " + ", ".join(f"{b} R2 {row[b]['r2']:.3f}" for b in row))
    save("transfer_ctrl", res)


def city_controls(d: ff.FitData) -> tuple[np.ndarray, list]:
    """Each city's usual controls (usual()), standardised within the city, as columns of their own (0 in the other cities): every city
    its own control coefficients."""
    cols, names = [], []
    for g, city in enumerate(d.group_names):
        m = d.group == g
        for c in USUAL + list(CITY_EXTRA.get(city, ())):
            if c not in d.control_names:
                continue
            v = d.X[:, d.control_names.index(c)]
            ok = m & np.isfinite(v)
            if ok.sum() < max(3, 0.5 * m.sum()) or np.std(v[ok]) <= 0:
                continue
            z = np.zeros(d.n_unit)
            z[ok] = (v[ok] - v[ok].mean()) / v[ok].std()
            cols.append(z); names.append(f"{city}:{c}")
    return (np.column_stack(cols) if cols else np.zeros((d.n_unit, 0))), names


def test_common(variant, part, city=None):
    slope = variant == "common_slope"
    if FIX_KH:
        variant = f"{variant}_kh{FIX_KH}"
    if FIX_H:
        variant = f"{variant}_h{FIX_H}"
    if TRAM_MID:
        variant = f"{variant}_tm"
    if SLOPE_NORM:
        variant = f"{variant}_sn"
    if CURVE_CITIES:
        variant = f"{variant}_cc{''.join(c[0] for c in CURVE_CITIES)}"
    pool = CURVE_CITIES or FIT
    free = ("alpha", "lkappa", "delta") + (("lslope",) if slope else ())
    if part in ("cv", "full"):
        d, _ = load_many(pool)
        X, names = city_controls(d)
        log(f"{variant} {part}: {d.n_unit} units, {len(names)} city control columns")
        if part == "cv":
            cv = heldout_skill(lambda dtr, Xtr: train_seq(dtr, Xtr, slope_per_group=slope), d, X)
            log(f"{variant} held-out skill {cv['skill']} R2 {cv['r2']}")
            save(f"{variant}_cv", {"cv": cv, "controls": names})
            return
        m = train_seq(d, X, slope_per_group=slope)
        OUT.mkdir(exist_ok=True)
        torch.save(m.state_dict(), OUT / f"{variant}.pt")
        save(variant, {"controls": names, "coefficients": dict(zip(names, m.delta.detach().numpy().tolist())), "params": m.params(),
                       "losses": m.losses, "small": small_all(m)})
        return
    others = [x for x in pool if x != city]
    dd, _ = load_many(others)
    mo = train_seq(dd, city_controls(dd)[0], slope_per_group=slope)
    tgt = load(city)[0]
    Xc, _ = city_controls(tgt)
    with torch.no_grad():
        A = mo.unit_access(tgt.tensors())
    res = {"controls": frozen_cv(mo, tgt, A, Xc, free=free), "no_controls": frozen_cv(mo, tgt, A, None, free=free),
           "params": mo.params(), "trained_on": others}
    res = {k: ({"skill": v["skill"][city], "r2": v["r2"][city]} if k in ("controls", "no_controls") else v) for k, v in res.items()}
    log(f"{variant}, leave out {city}: {res['controls']} (no controls {res['no_controls']})")
    save(f"{variant}_loco_{city}", res)


def test_controls():
    d, _ = load_many(FIT)
    m0 = load_model("pooled")
    with torch.no_grad():
        A = m0.unit_access(d.tensors())
    res = {"modes": {}, "between_city_share": {}, "coverage": {}}
    for j, c in enumerate(d.control_names):   # between-city share of the variance (eta squared) over the units that have the control
        v = d.X[:, j]; ok = np.isfinite(v)
        res["coverage"][c] = [d.group_names[g] for g in range(d.n_group) if np.isfinite(v[d.group == g]).mean() > 0.5]
        if ok.sum() >= 10 and len(set(d.group[ok])) >= 2:
            gm = pd.Series(v[ok]).groupby(d.group[ok]).transform("mean").to_numpy()
            res["between_city_share"][c] = float(np.var(gm) / max(np.var(v[ok]), 1e-12))
    for mode in CONTROL_MODES:
        X, names = control_matrix(d, mode, universal(d))
        r = frozen_cv(m0, d, A, X, one_level=(mode == "pooled_raw_one_level"))
        res["modes"][mode] = {**r, "controls": names}
        log(f"controls {mode}: mean {r['mean']:.4f} {r['skill']}")
    chosen, base, steps, pool = [], res["modes"]["none"]["mean"], [], universal(d)
    while pool:   # forward selection of the within-city standardised controls (kept while the held-out gain is >= 0.001)
        trial = {c: frozen_cv(m0, d, A, control_matrix(d, "within_z", chosen + [c])[0]) for c in pool}
        best = max(trial, key=lambda c: trial[c]["mean"])
        gain = trial[best]["mean"] - base
        steps.append({"added": best, "mean": trial[best]["mean"], "skill": trial[best]["skill"], "gain": gain,
                      "tried": {c: r["mean"] for c, r in trial.items()}})
        log(f"forward: + {best} -> {trial[best]['mean']:.4f} (gain {gain:+.4f})")
        if gain < 0.001:
            break
        chosen.append(best); pool.remove(best); base = trial[best]["mean"]
    res["forward"] = {"selected": chosen, "steps": steps}
    # the naive pooled fit's coefficients: how much of each control's effect is the difference between the cities
    X, names = control_matrix(d, "pooled_raw", universal(d))
    coef = {}
    for one in (True, False):
        m = ff.FlexModel(1 if one else d.n_group, X.shape[1])
        m.load_state_dict({n: v for n, v in m0.state_dict().items() if n not in ("alpha", "lkappa", "lslope", "delta")}, strict=False)
        for n, p in m.named_parameters():
            p.requires_grad_(n in ("alpha", "lkappa", "delta"))
        t = d.tensors(); w = weights(t, d)
        if one:
            t.group = torch.zeros_like(t.group)
        ff.fit(m, t, w, Xt(X), A=A, steps=300, lr=0.05, polish=60)
        coef["one_level" if one else "city_levels"] = dict(zip(names, m.delta.detach().numpy().round(4).tolist()))
    Xw, nw = control_matrix(d, "within_z", universal(d))
    m = ff.FlexModel(d.n_group, Xw.shape[1])
    m.load_state_dict({n: v for n, v in m0.state_dict().items() if n not in ("alpha", "lkappa", "lslope", "delta")}, strict=False)
    for n, p in m.named_parameters():
        p.requires_grad_(n in ("alpha", "lkappa", "delta"))
    t = d.tensors(); w = weights(t, d)
    ff.fit(m, t, w, Xt(Xw), A=A, steps=300, lr=0.05, polish=60)
    coef["within_z"] = dict(zip(nw, m.delta.detach().numpy().round(4).tolist()))
    res["coefficients"] = coef
    save("controls", res)


def test_final():
    sel = json.loads((OUT / "controls.json").read_text())["forward"]["selected"]
    d, _ = load_many(FIT)
    X, names = control_matrix(d, "within_z", sel)
    cv = heldout_skill(lambda dtr, Xtr: train_seq(dtr, Xtr), d, X)
    log(f"final pooled ({names}) held-out skill {cv['skill']}")
    m = train_seq(d, X)
    OUT.mkdir(exist_ok=True)
    torch.save(m.state_dict(), OUT / "final.pt")
    loco = {}
    for c in FIT:
        dd, _ = load_many([x for x in FIT if x != c])
        mo = train_seq(dd, control_matrix(dd, "within_z", sel)[0])
        tgt = load(c)[0]
        Xc, _ = control_matrix(tgt, "within_z", sel)   # the target's own controls with the training coefficients; its level refitted
        loco[c] = {"level": target_skill(mo, tgt, Xc)[c], "level_slope": target_skill(mo, tgt, Xc, free=("alpha", "lkappa", "lslope"))[c]}
        log(f"final, leave out {c}: {loco[c]}")
    save("final", {"controls": names, "cv": cv, "loco": loco, "params": m.params(), "losses": m.losses, "small": small_all(m, sel)})


if __name__ == "__main__":
    test, *args = sys.argv[1:]
    if test == "report":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import fit_report

        fit_report.write(OUT)
    elif test == "small":   # the small cities again with a saved fit (after a small city's data changed): small <result name>
        r = json.loads((OUT / f"{args[0]}.json").read_text())
        r["small"] = small_all(load_model(args[0]))
        save(args[0], r)
    elif test in ("common", "common_slope"):
        test_common(test, *args)
    else:
        {"us": test_us, "own": test_own, "transfer": test_transfer, "pooled": test_pooled, "loco": test_loco, "controls": test_controls,
         "final": test_final, "own_ctrl": test_own_ctrl, "transfer_ctrl": test_transfer_ctrl}[test](*args)
