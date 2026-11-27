"""The city science fits on the library `transitlos.flex.fit` (needs torch: TORCH_PYTHON): one smooth score to rank the cities, a reasonable fit per city
and for all cities together.

    $TORCH_PYTHON flex/fit_cities.py own <city>              # the city on its own: its own curves, usual controls; 5-fold held-out; results/v2_own_<city>.*
    $TORCH_PYTHON flex/fit_cities.py shared <pool> [name]    # cities share curves (pool: large | four | five, or a variant name); results/v2_shared_<name>.*
    $TORCH_PYTHON flex/fit_cities.py rank <name>             # every city's score from the shared curves; results/v2_rank_<name>.json
    $TORCH_PYTHON flex/fit_cities.py reach                   # the service each city has (no model); results/v2_reach.json
    $TORCH_PYTHON flex/fit_cities.py report                  # flex/FIT_RESULTS.md from the results

The model is `transitlos.flex.fit.ScoreModel` (shapes and options in `ModelSpec`): the score is computed per H3 res-11 cell, s_c = log(1 + A_c / eps), and a
unit (a block group, an AGEB, a centralidad, a district) takes the population-weighted MEAN of the scores of the cells inside it; smooth log-logistic
curves in headway, speed and walking distance with bounded steepness (`kh_max`, 2 for the cities); the walking curve is flat below 2 cell diameters and the
headway curve below 3 minutes; modes cannot be out of order (rail >= tram >= bus); the speed curve is fitted FIRST, with every mode factor equal to 1,
then frozen, then the mode factors; no literature value enters the fit.
Shared fits: the curves and eps are common; each city has its own level, dispersion, slope of the score (geometric mean 1) and control coefficients
(controls standardised within the city). The score of a place for a ranking is 100 * s / P95(s) with the shared curves and eps only.
Evaluation: within-city 5-fold spatial cross-validation (`tf.cv`); a city that is not a donor only has its level, slope and control coefficients fitted on
its own training folds. R2 = share of the variance of the units' transit share explained, people-weighted.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transitlos.flex import fit as tf  # noqa: E402
from transitlos.flex.fit.curves import K_RANGE, _inv  # noqa: E402

import build_fit_data as FB  # noqa: E402  (the populated cells and the service items of a city)

DEV = "cpu" if os.environ.get("FIT_CPU") == "1" or not torch.cuda.is_available() else "cuda"
STEPS = int(os.environ.get("FIT_STEPS", "1200"))
STARTS = int(os.environ.get("FIT_STARTS", "3"))
OUT = Path(__file__).resolve().parent / "results"
torch.set_num_threads(int(os.environ.get("FIT_THREADS", "5")))
FIT = ["boston", "san_francisco", "toronto", "concepcion", "guadalajara"]
SMALL = ["beerseba", "gipuzkoa", "taipei", "hamburg", "andorra"]
#: shared fits that are reported: the pools (headway steepness at most 2, the city default) and the sensitivity variants of the large pool
VARIANTS = {"large_khmax15": "steepness at most 1.5", "large_khmax1": "steepness at most 1",
            "large_kh6": "steepness bound 6", "large_hus": "headway curve of the US fit", "large_hus_wus": "headway and walking curves of the US fit"}
POOLS = {"large": ["boston", "san_francisco", "toronto"], "four": ["boston", "san_francisco", "toronto", "concepcion"],
         "five": ["boston", "san_francisco", "toronto", "concepcion", "guadalajara"]}
ALL_FITTED = POOLS["five"]
#: cities that have service and people but no outcome units: scored with the shared curves, not fitted
RANK_ONLY = ["shanghai"]
NON_CONTROL = ("GEOID", "share", "n", "y", "weight", "pop", "cells", "fold", "basis", "los_default")
USUAL = ["log_density_1km", "log_income", "poverty_share", "university_share", "basic_education_share", "under18_share", "over65_share"]
#: extra controls of single cities (the left vote has a different meaning in each country, so only where tested for that city)
CITY_EXTRA = ("left_share",)
CITY_EXTRA_CITIES = ("guadalajara",)
#: units with a default score below this are left out (feeds missing for whole municipalities)
NO_FEED_BELOW = {"guadalajara": 5.0}
CONTROLS = USUAL + list(CITY_EXTRA)


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.0f}s] {msg}", flush=True)


def spec_of(name: str | None = None) -> tf.ModelSpec:
    """The model spec of a shared fit: the city default (headway steepness at most 2) or one of the sensitivity variants."""
    kh = {"large_khmax15": 1.5, "large_khmax1": 1.0, "large_kh6": 6.0}.get(name or "", float(os.environ.get("FIT_KH_MAX", "2") or 2))
    return tf.ModelSpec(kh_max=kh)


def fixed_params(name: str | None, spec: tf.ModelSpec) -> dict | None:
    """Curves held at the US fit's (half value and steepness, `docs/reports/us_flex_final_score.md`): variants `*_hus`, `*_hus_wus`."""
    out = {}
    if name and name.endswith("_hus") or name and name.endswith("_hus_wus"):
        out["th_h"] = torch.tensor([float(np.log(5.98)), _inv(0.83, (K_RANGE["headway"][0], spec.kh_max))])
    if name and name.endswith("_hus_wus"):
        out["th_w"] = torch.tensor([float(np.log(590.0)), _inv(2.39, (K_RANGE["walk"][0], spec.kw_max))])
    return out or None


# ---------------------------------------------------------------------------------------------------------------------------------- data
def load(city: str) -> tuple[tf.FitData, pd.DataFrame]:
    """The city's fit data (units with populated cells and a share; build_fit_data.py) and its unit table, in the same order."""
    z = np.load(HERE / city / "flex" / "fit" / "items.npz")
    u = pd.read_parquet(HERE / city / "flex" / "fit" / "units.parquet")
    ctrl = [c for c in u.columns if c not in NON_CONTROL]
    d = tf.FitData(z["it_cell"], z["it_mode"], z["it_h"], z["it_v"], z["it_d"], z["cell_unit"], z["cell_pop"], u["y"].to_numpy(float),
                   u["n"].to_numpy(float), u["weight"].to_numpy(float), np.zeros(len(u), np.int64), u["fold"].to_numpy(np.int64),
                   u[ctrl].to_numpy(float), [city], ctrl, u["GEOID"].astype(str).to_numpy())
    d.X = np.array(d.X, dtype=float)
    for c in CITY_EXTRA:   # the left vote means something different in each country: a control only of the cities it was tested for
        if c in d.control_names and city not in CITY_EXTRA_CITIES:
            d.X[:, d.control_names.index(c)] = np.nan
    keep = (u["cells"].to_numpy() > 0) & np.isfinite(u["share"].to_numpy(float))
    if city in FIT:
        keep &= np.isfinite(u["n"].to_numpy(float)) & (u["n"].to_numpy(float) > 0)
    if city in NO_FEED_BELOW and NO_FEED_BELOW[city]:   # units the city's feeds do not reach: their transit use is not service
        keep &= u["los_default"].to_numpy(float) >= NO_FEED_BELOW[city]
    return d.select_units(keep), u[keep].reset_index(drop=True)


def load_many(cities) -> tuple[tf.FitData, pd.DataFrame]:
    parts = [load(c) for c in cities]
    return tf.FitData.concat([p[0] for p in parts]), pd.concat([p[1] for p in parts], ignore_index=True)


def small_city_scores(s: np.ndarray, u: pd.DataFrame) -> dict:
    """A small city from its units' service term s: the city's level on the logit share with slope 1 (the link), R2 of the logit share after the level,
    R2 with a free slope, the slope, and the rank correlation of s with the share."""
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


def _X(X):
    return None if X is None or X.shape[1] == 0 else torch.as_tensor(X, dtype=torch.float64, device=DEV)


def train(d: tf.FitData, X: np.ndarray, spec: tf.ModelSpec, fixed=None, seed: int = 0):
    """The curves on the units of `d`: speed first, then mode."""
    t = d.tensors(DEV)
    m, losses = tf.fit_sequence(t, tf.group_weights(t, d.weight), _X(X), X.shape[1], spec, STARTS, STEPS, seed=seed, fixed=fixed, log=log)
    m.losses = losses
    return m.cpu()


def small_cities(m: tf.ScoreModel) -> dict:
    """The small cities scored with the curves of `m` (slope 1, no controls): R2 of the logit share after the city's level, and the rank correlation."""
    out = {}
    for c in SMALL:
        d, u = load(c)
        with torch.no_grad():
            s = m.unit_access(d.tensors("cpu")).numpy()
        out[c] = small_city_scores(s, u)
    return out


def save(name: str, obj) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    log(f"wrote {name}.json")


# ---------------------------------------------------------------------------------------------------------------------------------- commands
def cmd_own(city: str) -> None:
    d, u = load(city)
    spec = spec_of().replace(slope_per_group=False)
    X, names = tf.control_columns(d, CONTROLS)
    log(f"{city}: {d.n_unit} units, controls {names}")
    res = tf.cv(d, spec, controls=CONTROLS, donors=[city], steps=STEPS, starts=STARTS, dev=DEV, log=log)
    log(f"{city} own fit held-out R2 {res.r2} (controls only {res.r2_controls_only})")
    m = train(d, X, spec)
    torch.save({"state": m.state_dict(), "spec": asdict(spec)}, OUT / f"v2_own_{city}.pt")
    save(f"v2_own_{city}", {"cv": res.to_dict(), "controls": names, "units": int(d.n_unit), "params": m.params(), "losses": m.losses})


def cmd_shared(pool: str, name: str | None = None) -> None:
    name = name or pool
    cities = POOLS[pool if pool in POOLS else "large"]
    spec = spec_of(name)
    fixed = fixed_params(name, spec)
    d, _ = load_many(ALL_FITTED)
    res = tf.cv(d, spec, controls=CONTROLS, donors=cities, steps=STEPS, starts=STARTS, dev=DEV, fixed=fixed, log=log)
    log(f"shared {name} held-out R2 {res.r2} (controls only {res.r2_controls_only})")
    dd = d.select_groups(cities)
    Xd, cn = tf.control_columns(dd, CONTROLS)
    m = train(dd, Xd, spec, fixed)
    torch.save({"state": m.state_dict(), "spec": asdict(spec), "n_group": len(cities), "n_ctrl": len(cn)}, OUT / f"v2_shared_{name}.pt")
    save(f"v2_shared_{name}", {"pool": cities, "spec": asdict(spec), "cv": res.to_dict(), "params": m.params(), "losses": m.losses, "controls": cn,
                               "coefficients": dict(zip(cn, m.delta.detach().numpy().tolist())), "small": small_cities(m)})


def load_shared(name: str) -> tf.ScoreModel:
    s = torch.load(OUT / f"v2_shared_{name}.pt", map_location="cpu", weights_only=False)
    m = tf.ScoreModel(tf.ModelSpec(**s["spec"]), s["n_group"], s["n_ctrl"])
    m.load_state_dict(s["state"])
    return m


def rank_data(city: str) -> tf.FitData:
    """A city without outcomes as one unit holding all its populated cells and their service items (enough to compute the score)."""
    cache = FB.HERE / city / "flex" / "fit" / "cells_res11.parquet"   # from res 10: flex/cells_from_res10.py
    cells = FB._cells(city) if (FB.HERE / city / "results" / "downloads" / "h3_res11.parquet").is_file() else pd.read_parquet(cache)
    it, _ = FB._items(city, cells["h3"].to_numpy())
    z = np.zeros(1)
    return tf.FitData(it["cell"].to_numpy(np.int64), it["mode"].to_numpy(np.int64), it["h"].to_numpy(np.float32), it["v"].to_numpy(np.float32),
                      it["d"].to_numpy(np.float32), np.zeros(len(cells), np.int64), cells["pop"].to_numpy(np.float32), z, z + 1, z + 1,
                      np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros((1, 0)), [city], [], np.array([city]))


def cmd_rank(name: str) -> None:
    """Every city's score from the shared curves: per H3 res-11 cell s_c = log(1 + A_c / eps), a city's score the population-weighted mean of its
    cells, scaled by the population-weighted 95th percentile of the scores of all the cities' cells (100 * s / P95, capped at 100)."""
    m = load_shared(name)
    NO_FEED_BELOW.clear()   # every unit of a city counts here, also those no feed reaches (their cells score 0)
    cities = list(FIT) + list(SMALL) + RANK_ONLY
    cells = {}
    for c in cities:
        d = rank_data(c) if c in RANK_ONLY else load(c)[0]
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
    NO_FEED_BELOW.clear()
    rows = {}
    for c in list(FIT) + list(SMALL):
        d, _ = load(c)
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


def _cv(p: dict, curve: str, x: float) -> float:
    """The value of a fitted curve at x (the table the model writes: `ScoreModel.tables`; json keys are strings of floats)."""
    return p["curves"][curve][str(float(x))]


def _curve_rows(items: dict) -> list:
    """Rows of a table of curves: name -> {pool/city label: params}."""
    rows = []
    for label, fmt, get in (
            ("headway 5 min", "{:.2f}", lambda p: _cv(p,"headway",5)), ("headway 10 min", "{:.2f}", lambda p: _cv(p,"headway",10)),
            ("headway 20 min", "{:.2f}", lambda p: _cv(p,"headway",20)), ("headway 30 min", "{:.2f}", lambda p: _cv(p,"headway",30)),
            ("speed 10 km/h", "{:.2f}", lambda p: _cv(p,"speed",10)), ("speed 20 km/h", "{:.2f}", lambda p: _cv(p,"speed",20)),
            ("speed 50 km/h", "{:.2f}", lambda p: _cv(p,"speed",50)), ("walk 400 m", "{:.2f}", lambda p: _cv(p,"walk",400)),
            ("walk 800 m", "{:.2f}", lambda p: _cv(p,"walk",800)), ("walk 1,200 m", "{:.2f}", lambda p: _cv(p,"walk",1200)),
            ("mode factor bus (rail 1)", "{:.2f}", lambda p: p["mode"]["bus"]), ("mode factor tram", "{:.2f}", lambda p: p["mode"]["tram"]),
            ("eps", "{:.4f}", lambda p: p["eps"])):
        rows.append(f"| {label} | " + " | ".join(fmt.format(get(p)) for p in items.values()) + " |")
    return rows


def cmd_report() -> None:
    lines = ["# City science flex fits: results", "",
             "Generated by `flex/fit_cities.py report` from `flex/results/v2_*.json` (revision of 2026-11-25). Model and rules: `flex/fit_cities.py` and "
             "the library `transitlos.flex.fit` (`ScoreModel`).", "",
             "The score is computed per H3 res-11 cell, `s = log(1 + A / eps)`, and a unit takes the population-weighted mean of its cells' scores. Smooth "
             "log-logistic curves (bounded steepness) for headway, speed and walking; rail >= tram >= bus by construction; the speed curve is fitted first "
             "with every mode factor 1, then frozen, then the mode factors. Every number is held out (within-city 5-fold, spatial). **R²** = share of the "
             "variance of the units' transit share explained (people-weighted); **controls only** = the same without any service term.", ""]
    own = {c: _load(f"v2_own_{c}") for c in FIT}
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
                  "| | " + " | ".join(SMALL) + " |", "|---|" + "---:|" * len(SMALL)]
        for p, v in shared.items():
            lines.append(f"| `{p}` | " + " | ".join(f"{v['small'][c]['r2_level']:.2f} ({v['small'][c]['spearman']:.2f})" for c in SMALL) + " |")
        lines.append("")
    ranks = {p: _load(f"v2_rank_{p}") for p in [*POOLS, *VARIANTS]}
    ranks = {p: v for p, v in ranks.items() if v}
    if ranks:
        cities = list(FIT) + list(SMALL)
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
