"""Efficiency of the recomputation: C (parallel) vs numpy at growing sizes, stage timings of a run, file sizes.   python analysis/efficiency.py NY 36061"""
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import run_county as rc  # noqa: E402
from transitlos.flex import core, scoring as sc, shapes as sh  # noqa: E402


def numpy_recompute(cs, cd, prof, p):
    nc, K = cs.shape
    best = np.zeros(nc)
    ss = sc.stop_score(np.where(prof.mode >= 0, prof.mode, 0), prof.h, prof.v, p) * (prof.mode >= 0)
    for k in range(K):
        s = np.where(cs[:, k] >= 0, cs[:, k], 0)
        valid = cs[:, k] >= 0
        for j in range(prof.P):
            m = prof.mode[s, j]; ok = valid & (m >= 0)
            q = ss[s, j]
            val = q * sc.walk_decay(cd[:, k], q, np.where(m >= 0, m, 0), p)
            best = np.maximum(best, np.where(ok, val, 0))
    return best * 100


def timeit(f, n=5):
    t = []
    for _ in range(n):
        a = time.perf_counter(); f(); t.append(time.perf_counter() - a)
    return min(t)


def main(state="NY", geoid="36061"):
    d = HERE / "data" / rc.STATE_DIRS[state] / geoid
    cells, cs, cd, prof, params, cfg = rc.load_run(d)
    p_alt = sc.DEFAULT.with_(headway_curve=sh.headway_exponential(), speed_curve=sh.speed_logistic(), walk_curve=sh.walk_gaussian(), agg="power_mean", agg_p=-1.0)
    print(f"{state} {geoid}: {len(cells)} cells, K={cs.shape[1]}, {prof.S} stations, P={prof.P}; threads available: {__import__('os').cpu_count()}")
    print(f"{'cells':>10s} {'numpy':>9s} {'C (stop+cell each)':>20s} {'C fast (stop scores once)':>27s} {'C fast, new shapes':>20s}")
    for rep in (1, 12, 120, 1200):
        c2 = np.tile(cs, (rep, 1)); d2 = np.tile(cd, (rep, 1))
        tn = timeit(lambda: numpy_recompute(c2, d2, prof, params), 1) if rep <= 12 else float("nan")
        t1 = timeit(lambda: core.recompute_cells(c2, d2, prof, params))
        t2 = timeit(lambda: core.recompute_cells_fast(c2, d2, prof, params))
        t3 = timeit(lambda: core.recompute_cells_fast(c2, d2, prof, p_alt))
        print(f"{len(c2):10d} {tn*1e3:8.1f}ms {t1*1e3:18.1f}ms {t2*1e3:25.1f}ms {t3*1e3:18.1f}ms   -> {len(c2)/t2/1e6:6.1f} M cells/s")
    ref = numpy_recompute(cs, cd, prof, params); fast = core.recompute_cells_fast(cs, cd, prof, params)[0]
    print("numpy vs C fast max diff (points):", float(np.abs(ref - fast).max()))
    import json
    r = json.loads((d / "run.json").read_text())
    print("run stages (s):", {k: v for k, v in r.items() if k.startswith("t_")})
    sizes = {f.name: f.stat().st_size for f in d.glob("*.parquet")}
    nc = len(cells)
    print("parquet files:", {k: f"{v/1e6:.2f} MB" for k, v in sizes.items()}, f"| cell_stops {sizes['cell_stops.parquet']/nc:.1f} B/cell, stop_profiles {sizes['stop_profiles.parquet']/prof.S:.0f} B/station")
    print("peak RSS of this process (GB):", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))


if __name__ == "__main__":
    main(*(sys.argv[1:3] or ["NY", "36061"]))
