"""The scoring preset of the city science score, for the flex maps: the curves of a shared fit as the map's tables.

    python flex/export_preset.py <shared fit name> [out.json]     # e.g. large -> flex/presets/city_score.json

Reads `results/v2_shared_<name>.json` (curves, modes, eps) and `results/v2_rank_<name>.json` (P95 of the cells' scores). Mapping onto the map's
scoring (`transitlos.flex.scoring`, `frontend/score.js`), exactly:
  stop value  M(mode) * S(speed) * H(headway)  -> agg "product", all weights 1; the speed and headway curves as tables
  walking     the map's table over u = (walking time - plateau) / t0; the plateau is the fit's walking floor and t0 is set to 10 min for every mode,
              so u runs 0 to 6 over 0 to 4.7 km beyond the floor
  cell        the best value of its stations and their profiles, then the output transform  min(1, log(1 + v / eps) / P95)  (`out_transform`)
Not carried over: the fit clips item speeds per mode (bus 5-70, tram 5-60, rail 10-130 km/h); the map does not.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WALK_SPEED = 1.3     # m/s, the map's
H_FLOOR_MIN = 3.0    # the headway curve is flat below it (the fit's)
V_TOP_KMH = 100.0    # the speed curve is 1 here and above


def preset(name: str) -> dict:
    fit = json.loads((HERE / "results" / f"v2_shared_{name}.json").read_text())
    rank = json.loads((HERE / "results" / f"v2_rank_{name}.json").read_text())
    p = fit["params"]
    h50, kh, v50, kv, w50, kw, floor = p["h50_min"], p["kh"], p["v50_kmh"], p["kv"], p["w50_m"], p["kw"], p["walk_floor_m"]
    fH = lambda h: 1.0 / (1.0 + (h / h50) ** kh)          # noqa: E731
    fS = lambda v: 1.0 / (1.0 + (v50 / v) ** kv)          # noqa: E731
    fW = lambda d: 1.0 / (1.0 + (d / w50) ** kw)          # noqa: E731
    hg = np.exp(np.linspace(0.0, math.log(1440.0), 256))
    head = fH(np.maximum(hg, H_FLOOR_MIN)) / fH(H_FLOOR_MIN)
    vg = np.linspace(0.0, 120.0, 241)
    speed = fS(np.clip(vg, 0.5, V_TOP_KMH)) / fS(V_TOP_KMH)
    t0 = 10.0
    ug = np.linspace(0.0, 6.0, 241)
    walk = fW(floor + ug * t0 * WALK_SPEED * 60.0) / fW(floor)
    r = lambda a: [round(float(x), 5) for x in a]          # noqa: E731
    params = {"mrc": [p["mode"]["bus"], p["mode"]["tram"], 1.0], "v_anchor": 30.0, "v_bonus": 0.15, "v_n": 2.0, "h_sat": 5.0, "h_p": -0.0658, "h_q": 0.1563,
              "walk_speed": WALK_SPEED, "plateau_m": floor, "t0_base": [t0, t0, t0], "k": 0.0, "p_shape": 1.0, "w_mode": 1.0, "w_speed": 1.0, "w_headway": 1.0,
              "agg": "product", "agg_p": 0.0,
              "headway_curve": {"kind": "table", "lo": 1.0, "hi": 1440.0, "scale": "log", "y": r(head)},
              "speed_curve": {"kind": "table", "lo": 0.0, "hi": 120.0, "scale": "lin", "y": r(speed)},
              "walk_curve": {"kind": "table", "lo": 0.0, "hi": 6.0, "scale": "lin", "y": r(walk)},
              "out_transform": {"kind": "log1p", "eps": float(p["eps"]), "ref": float(rank["p95"])}}
    return {"name": "City science score 2026-11: curves shared by the cities (Boston, San Francisco, Toronto)", "run": f"v2_shared_{name}", "params": params, "gamma": 1.0}


if __name__ == "__main__":
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "presets" / "city_score.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps([preset(sys.argv[1])]))
    print(f"wrote {out}")
