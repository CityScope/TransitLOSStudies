"""How much variability does a cap lose? (companion of cap_sweep.py: same reference run, same candidate runs under <temp dir>/capsweep)

Stations (depends on P): share of stations whose profile list was cut; of the profiles / modes / extremes the uncapped list had, how many survive.
Cells (depends on K): the same for the stations of a cell; plus how often the best option of each axis (nearest, best headway, best speed, best mode)
is still among the kept stations; plus 'winner kept' = share of (cell, scoring function) pairs where the kept set contains the reference winner within 0.01 points.
"""
import tempfile
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_county as pl  # noqa: E402
from transitlos.flex import core  # noqa: E402
import cap_sweep as cs_  # noqa: E402

ROOT = Path(tempfile.gettempdir()) / "capsweep"; ST = "NewYork"; GEOID = "36061"


def station_metrics(ref, cap):
    rm, cm = ref.mode, cap.mode
    rv, cv = ref.valid_mask(), cap.valid_mask()
    nref, ncap = rv.sum(1), cv.sum(1)
    cut = nref > ncap
    out = {"stations cut %": 100 * cut.mean(), "profiles kept % (of the cut stations')": 100 * ncap[cut].sum() / max(nref[cut].sum(), 1) if cut.any() else 100.0}
    def has_mode(m, v, k): return ((m == k) & v).any(1)
    mode_pairs = sum(has_mode(rm, rv, k).sum() for k in range(3)); kept = sum((has_mode(rm, rv, k) & has_mode(cm, cv, k)).sum() for k in range(3))
    out["(station,mode) pairs kept %"] = 100 * kept / max(mode_pairs, 1)
    hm = lambda p: np.where(p.valid_mask(), p.h, np.inf).min(1); vm = lambda p: np.where(p.valid_mask(), p.v, 0).max(1)
    out["best headway kept exactly %"] = 100 * np.mean(np.isclose(hm(cap), hm(ref))[cut]) if cut.any() else 100.0
    out["...within 15% %"] = 100 * np.mean((hm(cap) <= 1.15 * hm(ref))[cut]) if cut.any() else 100.0
    out["best speed kept exactly %"] = 100 * np.mean(np.isclose(vm(cap), vm(ref))[cut]) if cut.any() else 100.0
    out["...within 10% %"] = 100 * np.mean((vm(cap) >= 0.9 * vm(ref))[cut]) if cut.any() else 100.0
    out["best mode kept % (cut stations)"] = 100 * np.mean((cap.smode == ref.smode)[cut]) if cut.any() else 100.0
    return out


def cell_metrics(rcs, rcd, rprof, cs, cd, prof, sets, ref_los):
    rv, cv = rcs >= 0, cs >= 0
    nref, ncap = rv.sum(1), cv.sum(1)
    cut = nref > ncap
    sm = lambda p, s, v: np.where(v, p.smode[np.where(s >= 0, s, 0)], -1)
    hmin = lambda p, s, v: np.where(v, p.shmin[np.where(s >= 0, s, 0)], np.inf).min(1)
    vmax = lambda p, s, v: np.where(v, p.svmax[np.where(s >= 0, s, 0)], 0).max(1)
    dmin = lambda d, v: np.where(v, d, 65535).min(1)
    refm, capm = sm(rprof, rcs, rv), sm(prof, cs, cv)
    pairs = sum(((refm == k).any(1)).sum() for k in range(3)); kept = sum((((refm == k).any(1)) & ((capm == k).any(1))).sum() for k in range(3))
    c = cut
    out = {"cells cut %": 100 * c.mean(), "stations kept % (cut cells')": 100 * ncap[c].sum() / max(nref[c].sum(), 1) if c.any() else 100.0,
           "(cell,mode) pairs kept %": 100 * kept / max(pairs, 1),
           "nearest kept exactly %": 100 * np.mean((dmin(cd, cv) <= dmin(rcd, rv))[c]) if c.any() else 100.0,
           "...within 15%/50m %": 100 * np.mean((dmin(cd, cv) <= np.maximum(1.15 * dmin(rcd, rv), dmin(rcd, rv) + 50))[c]) if c.any() else 100.0,
           "best headway kept exactly %": 100 * np.mean(np.isclose(hmin(prof, cs, cv), hmin(rprof, rcs, rv))[c]) if c.any() else 100.0,
           "...within 15% %": 100 * np.mean((hmin(prof, cs, cv) <= 1.15 * hmin(rprof, rcs, rv))[c]) if c.any() else 100.0,
           "best speed kept exactly %": 100 * np.mean(np.isclose(vmax(prof, cs, cv), vmax(rprof, rcs, rv)) [c]) if c.any() else 100.0,
           "...within 10% %": 100 * np.mean((vmax(prof, cs, cv) >= 0.9 * vmax(rprof, rcs, rv))[c]) if c.any() else 100.0,
           "best mode kept % (cut cells)": 100 * np.mean((capm.max(1) == refm.max(1))[c]) if c.any() else 100.0}
    ok = np.zeros(len(rcs)); ok1 = np.zeros(len(rcs))
    reg_all = []
    for (kind, p), rl in zip(sets, ref_los):
        r = rl - core.recompute_cells(cs, cd, prof, p)[0]
        ok += (r <= 0.01); ok1 += (r <= 1.0); reg_all.append(r)
    out["winner kept % (all cells x functions)"] = 100 * ok.sum() / (len(rcs) * len(sets))
    out["within 1 point % (all cells x functions)"] = 100 * ok1.sum() / (len(rcs) * len(sets))
    out["mean error (points)"] = float(np.mean(reg_all))
    return out


def main():
    ref = pl.load_run(ROOT / "ref" / ST / GEOID); cells, rcs, rcd, rprof = ref[:4]
    sets = cs_.param_sets(); ref_los = [core.recompute_cells(rcs, rcd, rprof, p)[0] for _, p in sets]
    print("== Profiles per station (varying P)")
    for P in (3, 4, 5, 6, 8, 10):
        cap = pl.load_run(ROOT / f"P{P}K10" / ST / GEOID)[3]
        print(f"P={P}: " + "  ".join(f"{k}: {v:.1f}" for k, v in station_metrics(rprof, cap).items()))
    print("\n== Stations per cell (varying K, with P = 10)")
    for K in (3, 4, 5, 6, 8, 10):
        _, cs, cd, prof, _, _ = pl.load_run(ROOT / f"P10K{K}" / ST / GEOID)
        print(f"K={K}: " + "  ".join(f"{k}: {v:.2f}" if 'error' in k else f"{k}: {v:.1f}" for k, v in cell_metrics(rcs, rcd, rprof, cs, cd, prof, sets, ref_los).items()))
    print("\n== Combined, recommended pair P=5")
    for K in (6, 8, 10):
        _, cs, cd, prof, _, _ = pl.load_run(ROOT / f"P5K{K}" / ST / GEOID)
        print(f"P=5 K={K}: " + "  ".join(f"{k}: {v:.2f}" if 'error' in k else f"{k}: {v:.1f}" for k, v in cell_metrics(rcs, rcd, rprof, cs, cd, prof, sets, ref_los).items()))


if __name__ == "__main__":
    main()
