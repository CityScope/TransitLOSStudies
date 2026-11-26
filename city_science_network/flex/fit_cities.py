"""The city science fits, revised (2026-11-25): one smooth score to rank the cities, a reasonable fit per city and for all cities together.

    $TORCH_PYTHON flex/fit_cities.py own <city>              # the city on its own: its own curves, usual controls; 5-fold held-out; results/v2_own_<city>.*
    $TORCH_PYTHON flex/fit_cities.py shared <pool> [name]    # cities share curves (pool: large | four | five); results/v2_shared_<name>.*
    $TORCH_PYTHON flex/fit_cities.py rank <name>             # every city's score from the shared curves; results/v2_rank_<name>.json
    $TORCH_PYTHON flex/fit_cities.py reach                   # the service each city has (no model); results/v2_reach.json
    $TORCH_PYTHON flex/fit_cities.py report                  # flex/FIT_RESULTS.md from the results

The model is `tools/studytools/flexfit.SmoothModel`:
  * the score is computed per H3 res-11 cell, s_c = log(1 + A_c / eps), and a unit (a block group, an AGEB, a centralidad, a district) takes the
    population-weighted MEAN of the scores of the cells inside it
  * smooth curves (log-logistic in headway, speed and walking distance, bounded steepness), the walking curve flat below 2 cell diameters and the
    headway curve flat below 3 minutes
  * modes that cannot be out of order: rail >= tram >= bus
  * the rule of the study: the speed curve is fitted FIRST, with every mode factor equal to 1, so that speed captures as much as it can; it is
    then frozen and the mode factors are fitted on what speed leaves
  * no literature value enters the fit; only the controls enter besides the service
Shared fits: the curves and eps are common; each city has its own level (alpha), dispersion, slope of the score (geometric mean 1, so the shared
curves carry the size of the effect of the average city) and control coefficients (controls standardised within the city). The score of a place
for a ranking is 100 * s / P95(s) with the shared curves and eps only: nothing of a city's own enters it.
Evaluation: within-city 5-fold spatial cross-validation; per fold the donor cities' curves are refitted without the fold, and a city that is not a
donor only has its level, slope and control coefficients fitted on its own training folds. R2 = share of the variance of the units' transit share
explained, people-weighted.
"""
from __future__ import annotations

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from studytools import flexfit as ff  # noqa: E402
import fit_tests as FT  # noqa: E402  (loading, controls and the scoring helpers of the city study)
import build_fit_data as FB  # noqa: E402  (the populated cells and the service items of a city)

DEV = "cpu" if os.environ.get("FIT_CPU") == "1" or not torch.cuda.is_available() else "cuda"
STEPS = int(os.environ.get("FIT_STEPS", "1200"))
STARTS = int(os.environ.get("FIT_STARTS", "3"))
OUT = FT.OUT
#: the shared fits that are reported: the pools, and the sensitivity variants of the large pool (FIT_FIX)
VARIANTS = {"large_khmax2": "steepness at most 2", "large_khmax15": "steepness at most 1.5", "large_khmax1": "steepness at most 1",
            "large_kh6": "steepness bound 6", "large_hus": "headway curve of the US fit", "large_hus_wus": "headway and walking curves of the US fit"}
POOLS = {"large": ["boston", "san_francisco", "toronto"], "four": ["boston", "san_francisco", "toronto", "concepcion"],
         "five": ["boston", "san_francisco", "toronto", "concepcion", "guadalajara"]}
ALL_FITTED = POOLS["five"]
log = FT.log
#: FIT_FIX=a,b: sensitivity variants of the shared curves (results are named by the variant): kh6 = headway steepness up to 6 instead of 3;
#: h_us = headway curve held at the one fitted on the 238,000 US block groups (half value 5.98 min, steepness 0.83: docs/reports/us_flex_final_score.md);
#: w_us = walking curve held at the US fit's (half distance 590 m, steepness 2.39, flat below 303.5 m)
FIX = set(x for x in os.environ.get("FIT_FIX", "").split(",") if x)
if "kh6" in FIX:
    ff.KH_RANGE = (0.5, 6.0)
#: FIT_KH_MAX=<k>: the headway curve's steepness allowed up to k (3 by default): a lower k keeps the value of a service from falling so fast
#: with the headway (the curve's decline from 5 to 20 minutes is milder), at a cost in held-out fit that the results show
KH_MAX = float(os.environ.get("FIT_KH_MAX", "0") or 0)
if KH_MAX:
    ff.KH_RANGE = (0.5, KH_MAX)
FLOOR = 303.5 if "w_us" in FIX else ff.walk_floor_m(11)


def _inv(k: float, lo_hi: tuple) -> float:
    q = (k - lo_hi[0]) / (lo_hi[1] - lo_hi[0])
    return float(np.log(q / (1 - q)))


def fixed_params() -> dict | None:
    out = {}
    if "h_us" in FIX:
        out.update(lh50=float(np.log(5.98)), xh=_inv(0.83, ff.KH_RANGE))
    if "w_us" in FIX:
        out.update(l50w=float(np.log(590.0)), xw=_inv(2.39, ff.KW_RANGE))
    return out or None


# ---------------------------------------------------------------------------------------------------------------------------------- fitting
def _X(X):
    return None if X is None or X.shape[1] == 0 else torch.as_tensor(X, dtype=torch.float64, device=DEV)


def train_donors(d: ff.FitData, X: np.ndarray, slope: bool, seed: int = 0) -> ff.SmoothModel:
    """The shared (or own) curves on the donor cities' units: speed first, then mode (see the module docstring)."""
    t = d.tensors(DEV)
    w = FT.weights(t, d).to(DEV)
    m, losses = ff.fit_sequence_smooth(t, w, _X(X), X.shape[1], FLOOR, STARTS, STEPS, seed, slope_per_group=slope,
                                       slope_norm=slope, log=log, fixed=fixed_params())
    m.losses = losses
    return m.cpu()


CURVE = ("lh50", "xh", "lv50", "xv", "l50w", "xw", "lmode", "leps")


def fit_targets(mc: ff.SmoothModel, d: ff.FitData, X: np.ndarray, service: bool = True) -> ff.SmoothModel:
    """Curves of `mc` on other cities: only their level, dispersion, slope of the score and control coefficients fitted on `d` (service=False:
    the service term switched off, the controls-only reference)."""
    m = ff.SmoothModel(d.n_group, X.shape[1], floor_m=mc.floor_m, slope_per_group=service)
    with torch.no_grad():
        for n in CURVE:
            getattr(m, n).copy_(getattr(mc, n))
        if not service:
            m.lslope.fill_(-30.0)
    for n, p in m.named_parameters():
        p.requires_grad_(n in ("alpha", "lkappa", "delta") or (service and n == "lslope"))
    m.to(DEV)
    t = d.tensors(DEV)
    w = FT.weights(t, d).to(DEV)
    with torch.no_grad():
        S = m.unit_access(t)
    ff.fit(m, t, w, _X(X), A=S, steps=400, lr=0.05, polish=60)
    return m.cpu()


def evaluate(m: ff.SmoothModel, d: ff.FitData, X: np.ndarray):
    """Per city the held-out loss parts on `d`: (loss, null loss, squared error, variance of the observed share)."""
    t = d.tensors("cpu")
    w = FT.weights(t, d)
    Xt = FT.Xt(X)
    with torch.no_grad():
        ll = ff.unit_ll(m, t, Xt)
        mu = torch.sigmoid(m.eta(t, Xt))
    a, b = FT._per_group(t, w, ll, ff.null_ll(t, w), d.n_group)
    e, v = FT._r2_parts(t, w, mu, d.n_group)
    return a, b, e, v


def views(d_all: ff.FitData, donors: list[str]):
    """The donor cities' data and the others' (each with its own controls, standardised within the city)."""
    names = d_all.group_names
    out = []
    for keep in ([c for c in names if c in donors], [c for c in names if c not in donors]):
        if not keep:
            out.append((None, None))
            continue
        dv = d_all.select_groups(keep)
        out.append((dv, FT.city_controls(dv)[0]))
    return out


def cv(d_all: ff.FitData, donors: list[str], slope: bool, reference: bool = True) -> dict:
    """Within-city 5-fold held-out scores of every city of `d_all` (donors: curves refitted per fold; others: only their level, slope and
    controls); the controls-only reference with the same folds."""
    (dd, Xd), (dt, Xt_) = views(d_all, donors)
    names = [c for c in d_all.group_names]
    num, den, sse, sst = ({c: 0.0 for c in names} for _ in range(4))
    ref = {c: [0.0, 0.0] for c in names}   # controls only: sse, sst
    for k in range(5):
        trd, ted = dd.fold != k, dd.fold == k
        m = train_donors(dd.select_units(trd), Xd[trd], slope)
        for dv, Xv, mdl, cities in ((dd, Xd, m, dd.group_names),):
            a, b, e, v = evaluate(m, dd.select_units(ted), Xd[ted])
            for i, c in enumerate(cities):
                num[c] += a[i]; den[c] += b[i]; sse[c] += e[i]; sst[c] += v[i]
        if dt is not None:
            trt, tet = dt.fold != k, dt.fold == k
            mt = fit_targets(m, dt.select_units(trt), Xt_[trt])
            a, b, e, v = evaluate(mt, dt.select_units(tet), Xt_[tet])
            for i, c in enumerate(dt.group_names):
                num[c] += a[i]; den[c] += b[i]; sse[c] += e[i]; sst[c] += v[i]
        if reference:   # no service: levels and controls only, on every city
            for dv, Xv in ((dd, Xd), (dt, Xt_)):
                if dv is None:
                    continue
                tr, te = dv.fold != k, dv.fold == k
                m0 = fit_targets(m, dv.select_units(tr), Xv[tr], service=False)
                _, _, e0, v0 = evaluate(m0, dv.select_units(te), Xv[te])
                for i, c in enumerate(dv.group_names):
                    ref[c][0] += e0[i]; ref[c][1] += v0[i]
        log(f"  fold {k}: " + ", ".join(f"{c} R2 {1 - sse[c] / sst[c]:.3f}" for c in names if sst[c] > 0))
    out = {"r2": {c: float(1 - sse[c] / sst[c]) for c in names if sst[c] > 0},
           "skill": {c: float(1 - num[c] / den[c]) for c in names if den[c] > 0}}
    if reference:
        out["r2_controls_only"] = {c: float(1 - ref[c][0] / ref[c][1]) for c in names if ref[c][1] > 0}
    out["r2_mean"] = float(np.mean(list(out["r2"].values())))
    return out


def small_cities(m: ff.SmoothModel) -> dict:
    """The small cities scored with the curves of `m` (slope 1, no controls): R2 of the logit share after the city's level, and the rank
    correlation of the score with the share."""
    out = {}
    for c in FT.SMALL:
        d, u = FT.load(c)
        t = d.tensors("cpu")
        with torch.no_grad():
            s = m.unit_access(t).numpy()
        out[c] = FT.small_city_scores(s, u)
    return out


def save(name: str, obj) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    log(f"wrote {name}.json")


# ---------------------------------------------------------------------------------------------------------------------------------- commands
def cmd_own(city: str) -> None:
    d, u = FT.load(city)
    X, names = FT.control_matrix(d, "within_z", FT.usual(d))
    log(f"{city}: {d.n_unit} units, controls {names}")
    res = cv(d, [city], slope=False)
    log(f"{city} own fit held-out R2 {res['r2']} (controls only {res['r2_controls_only']})")
    m = train_donors(d, X, slope=False)
    torch.save(m.state_dict(), OUT / f"v2_own_{city}.pt")
    save(f"v2_own_{city}", {"cv": res, "controls": names, "units": int(d.n_unit), "params": m.params(), "losses": m.losses})


def cmd_shared(pool: str, name: str | None = None) -> None:
    name = name or pool
    cities = POOLS[pool]
    d, _ = FT.load_many(ALL_FITTED)
    res = cv(d, cities, slope=True)
    log(f"shared {name} held-out R2 {res['r2']} (controls only {res['r2_controls_only']})")
    dd = d.select_groups(cities)
    Xd, cn = FT.city_controls(dd)
    m = train_donors(dd, Xd, slope=True)
    torch.save(m.state_dict(), OUT / f"v2_shared_{name}.pt")
    save(f"v2_shared_{name}", {"pool": cities, "fix": sorted(FIX), "floor_m": FLOOR, "kh_max": KH_MAX, "cv": res, "params": m.params(), "losses": m.losses, "controls": cn,
                               "coefficients": dict(zip(cn, m.delta.detach().numpy().tolist())), "small": small_cities(m)})


def load_shared(name: str) -> ff.SmoothModel:
    s = torch.load(OUT / f"v2_shared_{name}.pt", map_location="cpu", weights_only=False)
    info = json.loads((OUT / f"v2_shared_{name}.json").read_text())
    if "kh6" in info.get("fix", []):
        ff.KH_RANGE = (0.5, 6.0)
    if info.get("kh_max"):
        ff.KH_RANGE = (0.5, float(info["kh_max"]))
    m = ff.SmoothModel(len(s["alpha"]), len(s["delta"]), floor_m=info.get("floor_m", ff.walk_floor_m(11)))
    m.load_state_dict(s)
    return m


#: cities that have service and people but no outcome units: scored with the shared curves, not fitted
RANK_ONLY = ["shanghai"]


def rank_data(city: str) -> ff.FitData:
    """A city without outcomes as one unit holding all its populated cells and their service items (enough to compute the score)."""
    cache = FB.HERE / city / "flex" / "fit" / "cells_res11.parquet"   # from res 10: flex/cells_from_res10.py
    cells = FB._cells(city) if (FB.HERE / city / "results" / "downloads" / "h3_res11.parquet").is_file() else pd.read_parquet(cache)
    it, _ = FB._items(city, cells["h3"].to_numpy())
    z = np.zeros(1)
    return ff.FitData(it["cell"].to_numpy(np.int64), it["mode"].to_numpy(np.int64), it["h"].to_numpy(np.float32), it["v"].to_numpy(np.float32),
                      it["d"].to_numpy(np.float32), np.zeros(len(cells), np.int64), cells["pop"].to_numpy(np.float32), z, z + 1, z + 1,
                      np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros((1, 0)), [city], [], np.array([city]))


def cmd_rank(name: str) -> None:
    """Every city's score from the shared curves: per H3 res-11 cell s_c = log(1 + A_c / eps), a city's score the population-weighted mean of its
    cells, scaled by the population-weighted 95th percentile of the scores of all the cities' cells (100 * s / P95, capped at 100)."""
    m = load_shared(name)
    FT.NO_FEED_BELOW = {}   # every unit of a city counts here, also those no feed reaches (their cells score 0)
    cities = list(FT.FIT) + list(FT.SMALL) + RANK_ONLY
    cells = {}
    for c in cities:
        d = rank_data(c) if c in RANK_ONLY else FT.load(c)[0]
        t = d.tensors("cpu")
        with torch.no_grad():
            cells[c] = (m.cell_score(t).numpy(), t.cell_pop.numpy())
    allS = np.concatenate([v[0] for v in cells.values()]); allP = np.concatenate([v[1] for v in cells.values()])
    o = np.argsort(allS)
    p95 = float(allS[o][np.searchsorted(np.cumsum(allP[o]) / allP.sum(), 0.95)])
    rows = {}
    for c, (s, p) in cells.items():
        sc = np.minimum(100.0 * s / p95, 100.0)
        served = s > 0
        rows[c] = {"score": float(np.average(sc, weights=p)), "share_people_served": float(p[served].sum() / p.sum()),
                   "score_where_served": float(np.average(sc[served], weights=p[served])) if served.any() else 0.0, "people": float(p.sum())}
    save(f"v2_rank_{name}", {"p95": p95, "cities": rows})
    for c, r in sorted(rows.items(), key=lambda kv: -kv[1]["score"]):
        log(f"{c:14s} score {r['score']:6.2f}   served {r['share_people_served']:.2f}   where served {r['score_where_served']:6.2f}")


def cmd_reach() -> None:
    """The service each city actually has, with no model: the share of its people by the best headway at any stop within 800 m of their cell
    (the headway of a stop combines all the routes that serve it), and the median speed of those service options."""
    FT.NO_FEED_BELOW = {}
    rows = {}
    for c in list(FT.FIT) + list(FT.SMALL):
        d, _ = FT.load(c)
        ok = d.it_d <= 800
        best = np.full(d.n_cell, np.inf)
        np.minimum.at(best, d.it_cell[ok], d.it_h[ok])
        pop = d.cell_pop
        share = lambda m: float(pop[m].sum() / pop.sum())  # noqa: E731
        rows[c] = {"people": float(pop.sum()), "none": share(~np.isfinite(best)), "h5": share(best <= 5), "h10": share(best <= 10), "h20": share(best <= 20),
                   "h30": share(best <= 30), "median_speed_kmh": float(np.median(d.it_v[ok]))}
        log(f"{c:14s} none {rows[c]['none']:.2f}  <=5 {rows[c]['h5']:.2f}  <=10 {rows[c]['h10']:.2f}  <=20 {rows[c]['h20']:.2f}")
    save("v2_reach", rows)


def _load(name):
    f = OUT / f"{name}.json"
    return json.loads(f.read_text()) if f.is_file() else None


def _curve_rows(items: dict) -> list:
    """Rows of a table of curves: name -> {pool/city label: params}."""
    rows = []
    for label, fmt, get in (
            ("headway 5 min", "{:.2f}", lambda p: p["curve_H"]["5"]), ("headway 10 min", "{:.2f}", lambda p: p["curve_H"]["10"]),
            ("headway 20 min", "{:.2f}", lambda p: p["curve_H"]["20"]), ("headway 30 min", "{:.2f}", lambda p: p["curve_H"]["30"]),
            ("speed 10 km/h", "{:.2f}", lambda p: p["curve_S"]["10"]), ("speed 20 km/h", "{:.2f}", lambda p: p["curve_S"]["20"]),
            ("speed 50 km/h", "{:.2f}", lambda p: p["curve_S"]["50"]), ("walk 400 m", "{:.2f}", lambda p: p["curve_W"]["400"]),
            ("walk 800 m", "{:.2f}", lambda p: p["curve_W"]["800"]), ("walk 1,200 m", "{:.2f}", lambda p: p["curve_W"]["1200"]),
            ("mode factor bus (rail 1)", "{:.2f}", lambda p: p["mode"]["bus"]), ("mode factor tram", "{:.2f}", lambda p: p["mode"]["tram"]),
            ("eps", "{:.4f}", lambda p: p["eps"])):
        rows.append(f"| {label} | " + " | ".join(fmt.format(get(p)) for p in items.values()) + " |")
    return rows


def cmd_report() -> None:
    lines = ["# City science flex fits: results", "",
             "Generated by `flex/fit_cities.py report` from `flex/results/v2_*.json` (revision of 2026-11-25). Model and rules: `flex/fit_cities.py` and "
             "`tools/studytools/flexfit.py` (`SmoothModel`).", "",
             "The score is computed per H3 res-11 cell, `s = log(1 + A / eps)`, and a unit takes the population-weighted mean of its cells' scores. Smooth "
             "log-logistic curves (bounded steepness) for headway, speed and walking; rail >= tram >= bus by construction; the speed curve is fitted first "
             "with every mode factor 1, then frozen, then the mode factors. Every number is held out (within-city 5-fold, spatial). **R²** = share of the "
             "variance of the units' transit share explained (people-weighted); **controls only** = the same without any service term.", ""]
    own = {c: _load(f"v2_own_{c}") for c in FT.FIT}
    own = {c: v for c, v in own.items() if v}
    if own:
        lines += ["## 1. Each city on its own", "", "Its own curves, its usual controls.", "",
                  "| | " + " | ".join(own) + " |", "|---|" + "---:|" * len(own),
                  "| units | " + " | ".join(str(v["units"]) for v in own.values()) + " |",
                  "| held-out R² | " + " | ".join(f"{v['cv']['r2'][c]:.3f}" for c, v in own.items()) + " |",
                  "| controls only | " + " | ".join(f"{v['cv']['r2_controls_only'][c]:.3f}" for c, v in own.items()) + " |",
                  "| added by the service | " + " | ".join(f"{v['cv']['r2'][c] - v['cv']['r2_controls_only'][c]:+.3f}" for c, v in own.items()) + " |"]
        lines += _curve_rows({c: v["params"] for c, v in own.items()}) + [""]
    shared = {p: _load(f"v2_shared_{p}") for p in [*POOLS, *VARIANTS]}
    shared = {p: v for p, v in shared.items() if v}
    if shared:
        cit = ALL_FITTED
        lines += ["## 2. Shared curves", "",
                  "The curves and eps are common to the donor cities; every city has its own level, slope of the score and control coefficients. "
                  "`large` = Boston, San Francisco, Toronto; `four` = + Concepción; `five` = + Guadalajara (centralidades). A city that is not a "
                  "donor only has its level, slope and controls fitted on its own training folds. Variants of `large`: `large_kh6` = headway steepness allowed "
                  "up to 6 (3 otherwise), `large_hus` = headway curve held at the US fit's, `large_hus_wus` = headway and walking curves held at the US fit's.", "",
                  "| held-out R² | " + " | ".join(cit) + " | mean |", "|---|" + "---:|" * (len(cit) + 1)]
        for p, v in shared.items():
            r = v["cv"]["r2"]
            lines.append(f"| curves from `{p}` | " + " | ".join(f"{r[c]:.3f}" if c in r else "–" for c in cit) + f" | {v['cv']['r2_mean']:.3f} |")
        ref = next(iter(shared.values()))["cv"].get("r2_controls_only", {})
        lines.append("| controls only | " + " | ".join(f"{ref[c]:.3f}" if c in ref else "–" for c in cit) + f" | {np.mean(list(ref.values())):.3f} |")
        if own:
            lines.append("| each city on its own | " + " | ".join(f"{own[c]['cv']['r2'][c]:.3f}" if c in own else "–" for c in cit) + " | |")
        lines += ["", "Curves of each pool:", "", "| | " + " | ".join(f"`{p}`" for p in shared) + " |", "|---|" + "---:|" * len(shared)]
        lines += _curve_rows({p: v["params"] for p, v in shared.items()}) + [""]
        lines += ["City slopes of the score (geometric mean 1) of each pool's donors:", "", "| | " + " | ".join(cit) + " |", "|---|" + "---:|" * len(cit)]
        for p, v in shared.items():
            sl = dict(zip(v["pool"], v["params"]["slope"]))
            lines.append(f"| `{p}` | " + " | ".join(f"{sl[c]:.2f}" if c in sl else "–" for c in cit) + " |")
        lines += ["", "### The small cities (predicted only)", "", "R² of the logit share after the city's level (rank correlation of the score with the share):", "",
                  "| | " + " | ".join(FT.SMALL) + " |", "|---|" + "---:|" * len(FT.SMALL)]
        for p, v in shared.items():
            lines.append(f"| `{p}` | " + " | ".join(f"{v['small'][c]['r2_level']:.2f} ({v['small'][c]['spearman']:.2f})" for c in FT.SMALL) + " |")
        lines.append("")
    ranks = {p: _load(f"v2_rank_{p}") for p in [*POOLS, *VARIANTS]}
    ranks = {p: v for p, v in ranks.items() if v}
    if ranks:
        cities = list(FT.FIT) + list(FT.SMALL)
        lines += ["## 3. The ranking of the cities", "",
                  "The score of a city is the population-weighted mean of its H3 res-11 cells' scores, `100 * s / P95` with the shared curves and eps and nothing "
                  "of a city's own (no level, slope or controls). P95 is the population-weighted 95th percentile of the scores of all the cities' cells. "
                  "Cities are listed by the first pool's score.", "",
                  "| city | " + " | ".join(f"score (`{p}`)" for p in ranks) + " | people served (share) |", "|---|" + "---:|" * (len(ranks) + 1)]
        first = next(iter(ranks.values()))["cities"]
        for c in sorted(cities, key=lambda c: -first[c]["score"]):
            lines.append(f"| {c} | " + " | ".join(f"{v['cities'][c]['score']:.1f}" for v in ranks.values()) + f" | {first[c]['share_people_served']:.2f} |")
        lines.append("")
    reach = _load("v2_reach")
    if reach:
        lines += ["## 4. The service each city has", "",
                  "No model: the share of each city's people by the best headway at any stop within 800 m of their cell (the headway of a stop combines all "
                  "the routes that serve it), and the median speed of those service options.", "",
                  "| city | people (M) | no stop within 800 m | best headway <= 5 min | <= 10 min | <= 20 min | <= 30 min | median speed (km/h) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for c, v in reach.items():
            lines.append(f"| {c} | {v['people'] / 1e6:.1f} | {v['none']:.2f} | {v['h5']:.2f} | {v['h10']:.2f} | {v['h20']:.2f} | {v['h30']:.2f} | {v['median_speed_kmh']:.1f} |")
        lines.append("")
    (OUT.parent / "FIT_RESULTS.md").write_text("\n".join(lines))
    log("wrote FIT_RESULTS.md")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    {"report": cmd_report, "reach": cmd_reach, "own": lambda: cmd_own(a[1]), "shared": lambda: cmd_shared(a[1], a[2] if len(a) > 2 else None), "rank": lambda: cmd_rank(a[1])}[a[0]]()
