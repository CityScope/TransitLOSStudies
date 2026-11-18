"""Options 4 and 5: (4) stratified weights that give the low-service end of the range more say, (5) a two-part (hurdle) model.

    python fit/fit_twopart.py [--maxiter 60] [--delta 0.5]

Part 1, "any transit commuter":  P(y > 0) = sigmoid(a0 + a1 log n + bA * f^gA)           (n = workers of the block group)
Part 2, "how many":              P(share > theta_k | any) = sigmoid(alpha_k + beta * f^g)   (the Option A cumulative logit)
Together                         P(share > theta_k) = P(any) * P(share > theta_k | any)
f is the same stop-based score for both parts (population weighted, US wide, unconstrained, no floors). gA is free and can get small, which is what makes a stop with a very long headway
(f near 0 but not 0) stand out from no stop at all. Variants: Option A (baseline) or two-part, each with the population weights or the service-class stratified weights.
Starting point: the best Option A solutions; optimised on the coarse sample, scored on the full one."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import fit2  # noqa: E402
from fit2 import Data, Model, Problem, load_subs, SAMPLE_COARSE  # noqa: E402

TH = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
CLASSES = ["no stop", "60+ min", "30-60", "15-30", "<15"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def service_class(subs, nbg, radius=400):
    """Per block group: population share of its cells in each class of the best (peak) headway of any stop within `radius` m. Returns (nbg, 5) fractions."""
    num = np.zeros((5, nbg)); den = np.zeros(nbg)
    for s in subs:
        h = s.prof.h.astype(float); h = np.where(h > 0, h, np.inf); hs = h.min(axis=1)
        ci = np.where((s.cst >= 0) & (s.cdist <= radius), hs[np.maximum(s.cst, 0)], np.inf).min(axis=1)
        cat = np.where(~np.isfinite(ci), 0, np.where(ci >= 60, 1, np.where(ci >= 30, 2, np.where(ci >= 15, 3, 4))))
        ok = s.bg >= 0; pw = s.pop.astype(float)
        for k in range(5):
            num[k] += np.bincount(s.bg[ok], weights=(pw * (cat == k))[ok], minlength=nbg)
        den += np.bincount(s.bg[ok], weights=pw[ok], minlength=nbg)
    return (num / np.maximum(den, 1e-9)).T


class Fit:
    def __init__(self, data, subs, model, two_part, delta):
        self.d, self.m, self.two = data, model, two_part
        self.prob = Problem(data, subs, model, "ordinal")
        self.share, self.n = self.prob.share, data.n
        self.any = (data.y > 0).astype(float)
        frac = service_class(subs, data.nbg)[data.idx]
        self.cls = frac.argmax(1)
        w0 = data.w * self.prob.pw
        self.w_pop = w0 / w0.sum()
        tot = np.array([self.w_pop[self.cls == k].sum() for k in range(5)])
        ws = self.w_pop / np.maximum(tot[self.cls], 1e-12) ** delta
        self.w_strat = ws / ws.sum()
        self.K = len(TH); self.nb = model.nb
        self.set_weights(self.w_pop)

    def set_weights(self, w):
        self.w = w
        self.h0 = []
        for th in TH:
            p0 = float(np.clip((w * (self.share > th)).sum(), 1e-6, 1 - 1e-6)); self.h0.append(-(p0 * np.log(p0) + (1 - p0) * np.log1p(-p0)))
        p0 = float(np.clip((w * self.any).sum(), 1e-6, 1 - 1e-6)); self.h_any = -(p0 * np.log(p0) + (1 - p0) * np.log1p(-p0))

    def raw_f(self, z):
        x = np.zeros(len(self.m.names)); x[: self.nb] = z[: self.nb]
        return self.prob.f(x)   # fixgamma: the raw score

    def unpack(self, z):
        i = self.nb; K = self.K
        beta = np.exp(np.clip(z[i], -5, 5)); a1 = z[i + 1]; dd = np.exp(np.clip(z[i + 2: i + 1 + K], -6, 3)); g = np.exp(np.clip(z[i + 1 + K], -2.5, 1.5))
        alpha = a1 - np.concatenate([[0.0], np.cumsum(dd)])
        hur = z[i + 2 + K: i + 6 + K]   # a0, a1, log bA, log gA
        return alpha, beta, g, hur

    def probs(self, z, f=None):
        f = self.raw_f(z) if f is None else f
        alpha, beta, g, hur = self.unpack(z)
        fg = np.power(f, g)
        q = np.stack([sigmoid(alpha[k] + beta * fg) for k in range(self.K)])
        if not self.two:
            return q, None
        gA = np.exp(np.clip(hur[3], -4, 1.5)); bA = np.exp(np.clip(hur[2], -6, 6))
        pi = sigmoid(hur[0] + hur[1] * np.log(np.maximum(self.n, 1.0)) + bA * np.power(f, gA))
        return q * pi, pi

    def loss(self, z, w=None):
        w = self.w if w is None else w
        p, pi = self.probs(z)
        tot = 0.0
        for k, th in enumerate(TH):
            ind = (self.share > th).astype(float); pk = np.clip(p[k], 1e-6, 1 - 1e-6)
            tot += -(w * (ind * np.log(pk) + (1 - ind) * np.log1p(-pk))).sum() / self.h0[k]
        terms = self.K
        if pi is not None:
            pa = np.clip(pi, 1e-6, 1 - 1e-6)
            tot += -(w * (self.any * np.log(pa) + (1 - self.any) * np.log1p(-pa))).sum() / self.h_any; terms += 1
        return float(tot / terms)

    def skills(self, z, w):
        p, pi = self.probs(z); out = {}
        for k, th in enumerate(TH):
            ind = (self.share > th).astype(float); pk = np.clip(p[k], 1e-6, 1 - 1e-6)
            p0 = float(np.clip((w * ind).sum(), 1e-6, 1 - 1e-6)); h = -(p0 * np.log(p0) + (1 - p0) * np.log1p(-p0))
            out[f"{th:g}"] = float(1 - (-(w * (ind * np.log(pk) + (1 - ind) * np.log1p(-pk))).sum()) / h)
        if pi is not None:
            pa = np.clip(pi, 1e-6, 1 - 1e-6); p0 = float(np.clip((w * self.any).sum(), 1e-6, 1 - 1e-6)); h = -(p0 * np.log(p0) + (1 - p0) * np.log1p(-p0))
            out["any"] = float(1 - (-(w * (self.any * np.log(pa) + (1 - self.any) * np.log1p(-pa))).sum()) / h)
        return out

    def by_class(self, z):
        p, pi = self.probs(z); f = self.raw_f(z); rows = {}
        pa = pi if pi is not None else p[0]
        for k, name in enumerate(CLASSES):
            m = self.cls == k; w = self.w_pop[m] / self.w_pop[m].sum()
            rows[name] = {"pop_share": float(self.w_pop[m].sum()), "obs_any": float((w * self.any[m]).sum()), "pred_any": float((w * pa[m]).sum()),
                          "obs_gt5": float((w * (self.share[m] > 0.05)).sum()), "pred_gt5": float((w * p[3][m]).sum()), "mean_f": float((w * f[m]).sum())}
        return rows


def init_vector(model, fit, best, two):
    x = np.array(best["x"]); nb = model.nb
    al, beta = model.ord_params(x); g = model.gamma_free(x) if hasattr(model, "gamma_free") else float(np.exp(np.clip(x[nb], -1.5, 1.5)))
    d = np.maximum(-np.diff(al), 0.05)
    z = list(x[:nb]) + [np.log(beta), al[0]] + list(np.log(d)) + [np.log(g)]
    z += [0.0, 0.0, np.log(2.0), np.log(0.3)]   # a0, a1, log bA, log gA (ignored when not two-part)
    return np.array(z)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--maxiter", type=int, default=60); ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--out", default=str(HERE / "data" / "twopart.json")); a = ap.parse_args()
    runs = [(f"s{s}r{i}", r) for s in (0, 1) for i, r in enumerate(json.load(open(HERE / "data" / f"o_param_nested_s{s}.json"))["runs"])]
    starts = [runs[0], runs[1], min(runs, key=lambda t: t[1]["loss"])]   # the tail-keeping solution, one sharp one, the best
    starts = list({n: (n, r) for n, r in starts}.values())
    model = Model("param", "nested", fixgamma=True, ordinal=TH)
    cdata, csubs = Data(wfile="bg_weights_coarse.npy"), load_subs(sample_dir=SAMPLE_COARSE)
    fdata, fsubs = Data(), load_subs()
    print("device", fit2.xb.device(), "coarse bgs", len(cdata.idx), "full bgs", len(fdata.idx), flush=True)
    res = []
    for two in (False, True):
        for strat in (False, True):
            tag = f"{'twopart' if two else 'optionA'}+{'strat' if strat else 'pop'}"
            cf = Fit(cdata, csubs, model, two, a.delta); ff = Fit(fdata, fsubs, model, two, a.delta)
            cf.set_weights(cf.w_strat if strat else cf.w_pop); ff.set_weights(ff.w_strat if strat else ff.w_pop)
            best = None
            for name, r in starts:
                z0 = init_vector(model, cf, r, two); t0 = time.time()
                rr = minimize(cf.loss, z0, method="L-BFGS-B", options={"maxiter": a.maxiter, "maxfun": a.maxiter * (len(z0) + 2), "eps": 1e-3})
                rr = minimize(ff.loss, rr.x, method="L-BFGS-B", options={"maxiter": max(10, a.maxiter // 3), "maxfun": 40 * (len(z0) + 2), "eps": 1e-3})
                print(f"[{tag}] from {name}: full-sample loss {rr.fun:.5f}  ({time.time() - t0:.0f}s)", flush=True)
                if best is None or rr.fun < best[0]:
                    best = (rr.fun, rr.x, name)
            z = best[1]
            entry = {"tag": tag, "loss": best[0], "start": best[2], "z": z.tolist(),
                     "skill_popw": ff.skills(z, ff.w_pop), "skill_strat": ff.skills(z, ff.w_strat), "by_class": ff.by_class(z)}
            res.append(entry); Path(a.out).write_text(json.dumps(res))
            print(tag, "skill (pop weights):", {k: round(v, 3) for k, v in entry["skill_popw"].items()}, flush=True)
            print("   by class (obs any, pred any, obs >5%, pred >5%):", {k: [round(v[x], 3) for x in ("obs_any", "pred_any", "obs_gt5", "pred_gt5")] for k, v in entry["by_class"].items()}, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
