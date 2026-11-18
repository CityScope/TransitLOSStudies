"""Control study helper. Significance: paired bootstrap over metros / counties (the held-out fold units), 4000 resamples, of the held-out skill
difference between two runs (all transit); one-sided; a family of tests (the groups of a stage, the speed forms) Holm-corrected at 0.05.
  ctrl_rank.py base '<groups json>' '<base>' -> FIT4_CTRL_ADD for the stage-B base: the base + every new group (stage N) that raised the held-out
                                                skill over A0 significantly (Holm over the new groups)
  ctrl_rank.py sets '<groups json>' '<add>'  -> '<k> <drop list>' lines: the sets keeping the top-k groups of stage B, ranked by the skill lost
                                                when each is dropped (k = 1, 2, 3, 4, 6, 8 and k = the number of significant groups, Holm)
  ctrl_rank.py pick <sets file>              -> the drop list of the smallest top-k set not significantly below B0
  ctrl_rank.py speed                         -> the speed form significantly better than sp_ref (Holm over the forms) with the highest skill
  ctrl_rank.py table                         -> every run against its reference: skill, difference, p, folds better
The unit sums are held-out and no-transit log-likelihoods (negative): skill = 1 - num / den."""
import json, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1] / "data_buf" / "torch_runs"
NEW = ("income_dist", "occupation", "vulnerability", "housing2")
ALPHA = 0.05
import os
BASE = os.environ.get("CTRL_BASE", "B0")   # study 7: sp_ref (the stage-B base + car commute speed): the control stages keep car speed
_cache = {}


def cv(tag):
    if tag not in _cache:
        _cache[tag] = json.loads((R / f"g5_controls_w_cs_{tag}.json").read_text())["cv"]
    return _cache[tag]


def unit_fold():
    """The fold of each metro / county (same for every run: fixed seed), from the first result that saved it."""
    for f in sorted(R.glob("g5_controls_w_cs_*.json")):
        c = json.loads(f.read_text())["cv"]
        if "unit_fold" in c:
            return np.array(c["unit_fold"])
    return None


def fold_skill(tag, task="total"):
    c = cv(tag); uf = unit_fold()
    if uf is None:
        return None
    n, d = np.array(c[task + "_unit"]["num"]), np.array(c[task + "_unit"]["den"])
    return [float(1 - n[uf == f].sum() / d[uf == f].sum()) for f in range(5)]


def test(a, b, task="total", B=4000, seed=0):
    """skill(b) - skill(a), and the one-sided p of 'b is not better than a' (paired bootstrap over metros / counties)."""
    ua, ub = cv(a)[task + "_unit"], cv(b)[task + "_unit"]
    na, nb, den = np.array(ua["num"]), np.array(ub["num"]), np.array(ua["den"])
    db = np.array(ub["den"])   # the same no-transit baseline in every run, up to GPU rounding (~1e-7 overall, <= 1e-4 per metro)
    assert np.allclose(den, db, rtol=1e-3, atol=1e-12), f"{a} / {b}: different no-transit baselines"
    den = 0.5 * (den + db)
    keep = den != 0; na, nb, den = na[keep], nb[keep], den[keep]
    diff = (na.sum() - nb.sum()) / den.sum()
    rng = np.random.default_rng(seed); n = len(den)
    W = rng.multinomial(n, np.full(n, 1 / n), size=B)            # resampled metro counts
    bd = (W @ (na - nb)) / (W @ den)
    return float(diff), float((bd <= 0).mean())


def holm(ps):
    """Holm step-down at ALPHA: {key: significant}, from {key: p}."""
    order = sorted(ps, key=ps.get); m = len(order); out = {k: False for k in ps}
    for i, k in enumerate(order):
        if ps[k] < ALPHA / (m - i):
            out[k] = True
        else:
            break
    return out


def folds_better(a, b):
    fa, fb = fold_skill(a), fold_skill(b)
    return "-" if fa is None else f"{sum(x > y for x, y in zip(fb, fa))}/5"


def ref_of(tag):
    if tag == "A0":
        return None
    if tag.startswith("coldcv_"):
        return "coldcv_A0" if tag == "coldcv_B0" else None
    if tag == "final_small":
        return "final_ref" if BASE == "sp_ref" else "final_B0"
    if tag in ("final_seq_mono", "final_nomode_mono"):
        return "final_ref"
    if tag in ("B0", "final_B0"):
        return "A0" if tag == "B0" else "B0"
    if tag.startswith(("sp_", "nosp_", "spv_", "seq_", "final_speed")) and tag != "sp_ref":
        return "sp_ref"
    if tag == "final_ref":
        return "sp_ref"
    if tag.startswith(("drop_", "top", "dens_")):
        return BASE
    if tag == "sp_ref":
        return "B0"
    return "A0"


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "pick":   # the smallest top-k set whose held-out skill is not significantly below B0 (else B0): its drop list
        ks = sorted(int(f.stem.split("_cs_top")[1]) for f in R.glob("g5_controls_w_cs_top*.json"))
        sets = dict(l.split(" ", 1) if " " in l else (l, "") for l in open(sys.argv[2]).read().split("\n") if l.strip())
        best = ""
        for k in ks:
            d, p = test(f"top{k}", BASE)          # skill lost by keeping only the top k; p of 'B0 not better than top-k'
            print(f"# top{k} loss {d:+.4f} p {p:.3f}", file=sys.stderr)
            if not (d > 0 and p < ALPHA):
                best = sets.get(str(k), ""); print(f"# picked top{k}", file=sys.stderr); break
        print(best)
        sys.exit()
    if mode == "speed":   # the speed form significantly better than sp_ref (Holm over the forms) with the highest skill, or nothing
        res = {}
        for f in [*R.glob("g5_controls_w_cs_sp_*.json"), *R.glob("g5_controls_w_cs_spv_*.json")]:
            tag = f.stem.replace("g5_controls_w_cs_", "")
            if tag != "sp_ref":
                res[tag] = test("sp_ref", tag)
        sig = holm({t: p for t, (d, p) in res.items() if d > 0} or {"_": 1.0})
        for t, (d, p) in sorted(res.items(), key=lambda x: -x[1][0]):
            print(f"# {t} {d:+.4f} p {p:.3f} {'significant (Holm)' if sig.get(t) else ''}", file=sys.stderr)
        c = [(res[t][0], t) for t, s in sig.items() if s and t in res]
        print(max(c)[1] if c else "")
        sys.exit()
    if mode == "table":
        tags = sorted(f.stem.replace("g5_controls_w_cs_", "") for f in R.glob("g5_controls_w_cs_*.json"))
        for tag in tags:
            ref = ref_of(tag)
            if ref is None or ref not in tags:
                print(f"{tag:22s} skill {cv(tag)['total']:.4f}  folds {' '.join(f'{x:.4f}' for x in (fold_skill(tag) or []))}"); continue
            d, p = test(ref, tag)
            cold = json.loads((R / f"g5_controls_w_cs_{tag}.json").read_text()).get("cv_cold")
            print(f"{tag:22s} skill {cv(tag)['total']:.4f}  vs {ref:12s} {d:+.4f}  p(not better) {p:.3f}  p(not worse) {test(tag, ref)[1]:.3f}"
                  f"  folds better {folds_better(ref, tag)}" + ("  [cold folds]" if cold else ""))
            if "nomode" in tag and tag != "nosp_nomode" and "nosp_nomode" in tags:   # speed without a mode factor: against no speed, no mode too
                d2, p2 = test("nosp_nomode", tag)
                print(f"{'':22s}              vs {'nosp_nomode':12s} {d2:+.4f}  p(not better) {p2:.3f}  folds better {folds_better('nosp_nomode', tag)}")
        sys.exit()
    G = json.loads(sys.argv[2])
    if mode == "base":
        add = sys.argv[3].split(",")
        res = {g: test("A0", f"add_{g}") for g in NEW}
        sig = holm({g: p for g, (d, p) in res.items() if d > 0} or {"_": 1.0})
        for g in NEW:
            print(f"# add_{g} {res[g][0]:+.4f} p {res[g][1]:.3f} {'significant (Holm)' if sig.get(g) else ''}", file=sys.stderr)
            if sig.get(g):
                add += G[g]
        print(",".join(add))
    else:
        add = sys.argv[3].split(",")
        G = {g: v for g, v in G.items() if g not in NEW or v[0] in add}
        res = {g: test(f"drop_{g}", BASE) for g in G}          # skill lost when dropped, p of 'not lost'
        order = sorted(G, key=lambda g: -res[g][0])
        sh = holm({g: res[g][1] for g in G if res[g][0] > 0} or {"_": 1.0})
        sig = [g for g in order if sh.get(g)]
        print("# drop " + " ".join(f"{g}:{res[g][0]:+.4f}(p{res[g][1]:.3f})" for g in order) + f" | significant (Holm): {','.join(sig)}", file=sys.stderr)
        ks = sorted({k for k in (1, 2, 3, 4, 6, 8) if k < len(order)} | ({len(sig)} if 0 < len(sig) < len(order) else set()))
        for k in ks:
            print(k, ",".join(c for g in order[k:] for c in G[g]))
