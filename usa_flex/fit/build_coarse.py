"""A coarse sample of the fit sample (about 10% of its block groups) for fast multi-start fitting, built to give the same loss as the full sample.

    python fit/build_coarse.py [--frac 0.10]

Poisson/systematic sampling: block group i is kept with probability pi_i ~ sqrt(w_i * pop_i) (the weight it has in the population-weighted least squares, so heavy block
groups are kept more often and every kept block group keeps ALL its cells, i.e. its predictor is exact), boosted for block groups that have transit commuters (the part of the
sample that carries the signal); pi_i is capped at 1 (certainty inclusion). The units are ordered by (state, CBSA, transit-share bin, population) before the systematic pass,
which spreads the sample evenly over places and over the share distribution. The kept block groups' weights become w_i / pi_i, so every weighted sum is unbiased.
Writes data/bg_weights_coarse.npy and data/sample_coarse/<State>.npz, then compares the loss / R2 of the coarse and the full sample for the default scoring, every fitted solution
found so far and random perturbations of them.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import fit2  # noqa: E402

OUT = HERE / "data" / "sample_coarse"


def select(frac, seed=11):
    bgt = pd.read_parquet(HERE / "data" / "blockgroups.parquet")
    w = np.load(HERE / "data" / "bg_weights.npy")
    pop = bgt.population.fillna(0).to_numpy().astype(float)
    wk = bgt.acs5_workers.fillna(0).to_numpy().astype(float)
    tr = bgt.publicTransportCommuters.to_numpy().astype(float)
    ok = (w > 0) & (wk > 0) & ~np.isnan(tr) & (pop > 0)
    share = np.where(ok, np.minimum(tr, wk) / np.maximum(wk, 1), 0.0)
    omega = np.where(ok, w * pop, 0.0)
    boost = np.where(share > 0, 2.0, 1.0)
    raw = np.sqrt(omega) * boost
    target = frac * ok.sum()
    lo, hi = 0.0, 1e6
    for _ in range(80):   # c such that sum(min(1, c * raw)) = target
        c = (lo + hi) / 2
        if np.minimum(1.0, c * raw)[ok].sum() < target:
            lo = c
        else:
            hi = c
    pi = np.where(ok, np.minimum(1.0, c * raw), 0.0)
    sbin = np.digitize(share, [1e-9, 0.01, 0.03, 0.08])
    order = np.lexsort((pop, sbin, bgt.cbsa.fillna("").to_numpy(), bgt.state.to_numpy()))
    order = order[ok[order]]
    rng = np.random.default_rng(seed)
    cum = np.cumsum(pi[order]) + rng.uniform(0, 1)
    hit = np.floor(cum) > np.floor(np.r_[cum[0] - pi[order][0], cum[:-1]])   # systematic: one unit per integer crossing
    keep = np.zeros(len(bgt), bool)
    keep[order[hit]] = True
    keep |= ok & (pi >= 1.0)
    wn = np.where(keep, w / pi, 0.0)
    return bgt, wn, keep, pi, ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frac", type=float, default=0.10); a = ap.parse_args()
    bgt, wn, keep, pi, ok = select(a.frac)
    print("eligible", int(ok.sum()), "kept", int(keep.sum()), f"({keep.sum() / ok.sum():.1%}), certainty", int((pi >= 1).sum()), flush=True)
    np.save(HERE / "data" / "bg_weights_coarse.npy", wn)
    OUT.mkdir(parents=True, exist_ok=True)
    ncell = 0
    for f in sorted(fit2.SAMPLE.glob("*.npz")):
        z = np.load(f); bg = z["bg"]
        m = (bg >= 0) & keep[np.maximum(bg, 0)]
        if not m.any():
            continue
        mode, h, v, cst, cd, b, pp = fit2.compact((z["mode"], z["h"], z["v"]), z["cst"], z["cdist"], bg, z["pop"], m)
        np.savez_compressed(OUT / f.name, mode=mode, h=h, v=v, cst=cst, cdist=cd, bg=b, pop=pp)
        ncell += int(m.sum())
    print("cells", ncell, flush=True)
    validate()


def validate():
    """Loss and R2 on the full sample versus the coarse one for scoring sets that cover the range the optimiser visits."""
    full_d, full_s = fit2.Data(), fit2.load_subs()
    co_d, co_s = fit2.Data(wfile="bg_weights_coarse.npy"), fit2.load_subs(sample_dir=OUT)
    rng = np.random.default_rng(3)
    sets = []
    for name in ("w_param_nested_s0", "w_param_nested_s1", "w_param_power_s0", "w_param_power_s1"):
        f = HERE / "data" / f"{name}.json"
        if f.exists():
            d = json.load(open(f))
            for r in d["runs"]:
                sets.append((name, d["agg"], np.array(r["x"])))
    for agg in ("nested", "power"):
        m = fit2.Model("param", agg)
        sets.append(("default", agg, m.x0()))
        for i in range(10):
            sets.append((f"random{i}", agg, m.x0() + rng.normal(0, 0.8, len(m.x0()))))
    rows = []
    cache = {}
    for name, agg, x in sets:
        m = cache.setdefault(agg, fit2.Model("param", agg))
        out = []
        for d, s in ((full_d, full_s), (co_d, co_s)):
            p = fit2.Problem(d, s, m, "wls")
            out.append((p.loss(x), p.null))
        (lf, nf), (lc, nc) = out
        rows.append((name, agg, 1 - lf / nf, 1 - lc / nc, lf, lc))
        print(f"{name:18s} {agg:7s} R2 full {1 - lf / nf:.4f} coarse {1 - lc / nc:.4f}  d {1 - lc / nc - (1 - lf / nf):+.4f}  loss full {lf:.5f} coarse {lc:.5f}", flush=True)
    dr = np.array([r[3] - r[2] for r in rows])
    print(f"R2 difference coarse - full over {len(rows)} scorings: mean {dr.mean():+.4f}, max abs {np.abs(dr).max():.4f}, rms {np.sqrt((dr ** 2).mean()):.4f}")
    rk = lambda v: np.argsort(np.argsort(v))
    print("rank correlation of the loss:", float(np.corrcoef(rk([r[4] for r in rows]), rk([r[5] for r in rows]))[0, 1]))


if __name__ == "__main__":
    main()
