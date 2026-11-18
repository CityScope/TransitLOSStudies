"""Which caps (P profiles per station, K stations per cell) lose little error on a real dense area?

Reference run = effectively uncapped (P = K = 30). Each candidate run keeps fewer. Both are re-scored under many scoring functions the selection never
saw -- each of the four axes switched OFF completely (mode, speed, headway, distance), random curves, random weights, reversed mode preference -- and the
regret (reference cell score minus capped cell score, 0-100 points) is reported per kind of change.

    python tests/cap_sweep.py NY 36061            (Manhattan)
"""
from __future__ import annotations

import tempfile
import io
import contextlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import run_county as pl  # noqa: E402
from transitlos.flex import core, scoring as sc  # noqa: E402


def param_sets(n_each=20, seed=0):
    rng = np.random.default_rng(seed)
    out = [("default", sc.DEFAULT)]
    base = lambda: sc.DEFAULT.with_(v_anchor=rng.uniform(12, 50), v_n=rng.uniform(1, 3), v_bonus=rng.uniform(0, 0.4), h_sat=rng.uniform(3, 15),
                                    walk_speed=rng.uniform(0.9, 1.6), plateau_m=rng.uniform(50, 400), k=rng.uniform(0.5, 5),
                                    t0_base=tuple(np.asarray(sc.DEFAULT.t0_base) * rng.uniform(0.3, 2.5)))
    for _ in range(n_each):
        out.append(("curves", base()))
    for _ in range(n_each):
        out.append(("weights", base().with_(w_mode=float(rng.choice([0.5, 1, 2])), w_speed=float(rng.choice([0.5, 1, 2])), w_headway=float(rng.choice([0.5, 1, 2])))))
    for _ in range(n_each):
        out.append(("mode OFF", base().with_(w_mode=0.0, mrc=(1.0, 1.0, 1.0))))
    for _ in range(n_each):
        out.append(("speed OFF", base().with_(w_speed=0.0)))
    for _ in range(n_each):
        out.append(("headway OFF", base().with_(w_headway=0.0)))
    for _ in range(n_each):
        out.append(("distance OFF", base().with_(plateau_m=1e6)))          # walking never decays: only the quality of the stop matters
    for _ in range(n_each):
        out.append(("bus preferred", base().with_(mrc=(1.0, 0.894, 0.8))))
    return out


def run_variant(state, geoid, name, cfg, root, cand_src):
    out = root / name
    dst = out / pl.STATE_DIRS[state]
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(cand_src, dst / cand_src.name)
    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        res = pl.run(state, geoid, cfg, out_root=out, validate=False, county_feeds_only=True)
    return out / pl.STATE_DIRS[state] / geoid, res, time.time() - t


def main(state="NY", geoid="36061"):
    root = Path(tempfile.gettempdir()) / "capsweep"; root.mkdir(parents=True, exist_ok=True)
    cand = HERE / "data" / pl.STATE_DIRS[state] / f"candidates_{geoid}.parquet"
    floors = {}   # no floors: nothing is dropped for being slow or infrequent
    ref_cfg = core.Config(P=30, K=30, Kc=48, Kcand=200, **floors)
    ref_dir, ref_res, _ = run_variant(state, geoid, "ref", ref_cfg, root, cand)
    cells, rcs, rcd, rprof, _, _ = pl.load_run(ref_dir)
    sets = param_sets()
    ref_los = [core.recompute_cells(rcs, rcd, rprof, p)[0] for _, p in sets]
    kinds = list(dict.fromkeys(k for k, _ in sets))
    print(f"reference P=K=30: stations {ref_res['stations']}, cells {ref_res['cells']}, max profiles/station {int((rprof.n).max())}, "
          f"candidates cut {ref_res['cells_candidates_cut']}")
    rows = []
    variants = [(P, K) for P in (3, 5, 8, 10) for K in (4, 6, 8, 10)]
    for P, K in variants:
        cfg = core.Config(P=P, K=K, **floors)
        d, res, secs = run_variant(state, geoid, f"P{P}K{K}", cfg, root, cand)
        _, cs, cd, prof, _, _ = pl.load_run(d)
        reg = {k: [] for k in kinds}
        for (kind, p), rl in zip(sets, ref_los):
            r = rl - core.recompute_cells(cs, cd, prof, p)[0]
            reg[kind].append(r)
        row = {"P": P, "K": K, "capped_cells_%": round(100 * res["cells_capped"] / res["cells"], 1), "bytes/cell": K * 6}
        allr = np.concatenate([np.concatenate(v) for v in reg.values()])
        row["mean"] = round(float(allr.mean()), 3); row["p99"] = round(float(np.percentile(allr, 99)), 2); row["max"] = round(float(allr.max()), 1)
        for k in kinds:
            row[k] = round(float(np.concatenate(reg[k]).mean()), 3)
        rows.append(row)
        print(f"P={P:2d} K={K:2d}  capped {row['capped_cells_%']:5.1f}%  mean {row['mean']:.3f} p99 {row['p99']:.2f} max {row['max']:.1f}   "
              + "  ".join(f"{k}:{row[k]:.3f}" for k in kinds), flush=True)
    pd.DataFrame(rows).to_csv(HERE / "data" / f"cap_sweep_{geoid}_nofloors.csv", index=False)


if __name__ == "__main__":
    main(*(sys.argv[1:3] or ["NY", "36061"]))
