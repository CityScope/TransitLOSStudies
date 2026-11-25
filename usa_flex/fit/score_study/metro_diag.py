"""Study 10 diagnostics: how well the score carries over to metros the fit has not seen, from the held-out block-group predictions
(<run>_ho.npz, written by torch_fit4.py's cross-validation; every metro / county is predicted by the fold fitted without it).

Three numbers per run (all transit; binomial log-likelihood per worker, evaluation weights of the fit):
  held-out   1 - loss / loss of one national constant share (the fit's held-out skill, binomial instead of beta-binomial)
  median     the held-out skill of the median metro, metros with >= MIN_TR transit commuters (each metro counts once, not by size)
  within     1 - loss / loss of each metro's own constant share, the model's level per metro calibrated away (one shift of the logit per
             metro): does the score rank the places inside a metro it never saw. Metros with >= MIN_TR transit commuters.
Runs are compared with REF by a paired bootstrap over metros (one-sided p that the run is not better).

    python metro_diag.py [run ...]      # default: every g_* run with saved predictions
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "data_buf" / "torch_runs"
PREFIX = "g5_controls_w_cs_"
REF = "seq_g_base"
MIN_TR = 1000


def _ll(y, n, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return (y * np.log(p) + (n - y) * np.log1p(-p)) / np.maximum(n, 1)


def per_metro(tag):
    """Per metro / county: weighted held-out loss, national-null loss, calibrated-model loss, own-null loss, transit commuters."""
    z = np.load(R / f"{PREFIX}{tag}_ho.npz")
    g, n, w, y, mu = z["group"], z["n"], z["w"] * z["ok_total"], z["y_total"], z["mu_total"].astype(float)
    ng = g.max() + 1; bc = lambda v: np.bincount(g, weights=v, minlength=ng)  # noqa: E731
    p0 = (w * y / np.maximum(n, 1)).sum() / w.sum()                            # one national share (the weighted mean)
    l_mod, l_nat = bc(-w * _ll(y, n, mu)), bc(-w * _ll(y, n, np.full_like(mu, p0)))
    pg = bc(w * y / np.maximum(n, 1)) / np.maximum(bc(w), 1e-12)                # each metro's own share
    l_own = bc(-w * _ll(y, n, pg[g]))
    eta = np.log(np.clip(mu, 1e-7, 1 - 1e-7) / (1 - np.clip(mu, 1e-7, 1 - 1e-7))); s = np.zeros(ng)
    for _ in range(30):   # Newton steps on one logit shift per metro
        p = 1 / (1 + np.exp(-(eta + s[g])))
        s -= bc(w * (p - y / np.maximum(n, 1))) / np.maximum(bc(w * p * (1 - p)), 1e-12)
    l_cal = bc(-w * _ll(y, n, 1 / (1 + np.exp(-(eta + s[g])))))
    return {"mod": l_mod, "nat": l_nat, "cal": l_cal, "own": l_own, "tr": bc(y)}


def summary(m):
    big = m["tr"] >= MIN_TR
    sk = 1 - m["mod"][big] / m["nat"][big]
    return {"held-out": 1 - m["mod"].sum() / m["nat"].sum(), "median": float(np.median(sk)),
            "within": 1 - m["cal"][big].sum() / m["own"][big].sum(), "n_metros": int(big.sum())}


def boot(a, b, key, B=4000, seed=0):
    """skill(b) - skill(a) in `key` (held-out or within) and the one-sided p that b is not better (bootstrap over metros)."""
    num, den = ("mod", "nat") if key == "held-out" else ("cal", "own")
    keep = np.ones_like(a["tr"], bool) if key == "held-out" else a["tr"] >= MIN_TR
    na, nb, d = a[num][keep], b[num][keep], 0.5 * (a[den][keep] + b[den][keep])
    W = np.random.default_rng(seed).multinomial(len(d), np.full(len(d), 1 / len(d)), size=B)
    bd = (W @ (na - nb)) / (W @ d)
    return float((na.sum() - nb.sum()) / d.sum()), float((bd <= 0).mean())


if __name__ == "__main__":
    tags = sys.argv[1:] or sorted(f.name[len(PREFIX):-7] for f in R.glob(f"{PREFIX}seq_g_*_ho.npz"))
    ref = per_metro(REF) if (R / f"{PREFIX}{REF}_ho.npz").exists() else None
    print(f"{'run':22s} {'held-out':>9s} {'median':>8s} {'within':>8s}   vs {REF}: held-out (p)   within (p)")
    for t in tags:
        m = per_metro(t); s = summary(m)
        line = f"{t:22s} {s['held-out']:9.4f} {s['median']:8.4f} {s['within']:8.4f}"
        if ref is not None and t != REF:
            (dh, ph), (dw, pw) = boot(ref, m, "held-out"), boot(ref, m, "within")
            line += f"   {dh:+.4f} ({ph:.3f})   {dw:+.4f} ({pw:.3f})"
        print(line)
    print(f"(median and within: the {s['n_metros']} metros / counties with >= {MIN_TR} transit commuters)")
