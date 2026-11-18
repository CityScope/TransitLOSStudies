"""Adds the GPU fits (fit/torch_fit.py) to the flex map's presets (web/sample/presets.json) so the data-driven curves can be explored on the map.

    python fit/export_presets.py
Mapping onto the map's scoring (transitlos.flex.scoring), exactly where possible:
  stop score  q = M(mode) * S(speed) * H(headway)  ->  agg "product" with all weights 1 (exactly the fitted form);
  headway and speed: table curves sampled from the fitted knots;
  walking: the map has one walking shape scaled by a time t0 per mode, the fit has one free curve per mode. Each mode's t0 is its half-effect walking time and the shape is
  the mean of the three modes' curves on that scaled axis (the error of this approximation is printed).
"""
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "data" / "torch_runs"
PRESETS = HERE.parent / "web" / "sample" / "presets.json"
WALK_SPEED = 1.3   # m/s, the map's default


def half(x, y):
    for i in range(1, len(x)):
        if y[i] <= 0.5 <= y[i - 1]:
            return x[i - 1] + (y[i - 1] - 0.5) / (y[i - 1] - y[i]) * (x[i] - x[i - 1])
    return x[-1]


def params_from(c):
    hx, hy = np.array(c["headway"]["x"]), np.array(c["headway"]["y"])
    hg = np.exp(np.linspace(0, math.log(1440), 256))
    head = np.exp(np.interp(np.log(hg), np.log(hx), np.log(hy)))
    vx, vy = np.array(c["speed"]["x"]), np.array(c["speed"]["y"])
    vg = np.linspace(0, 120, 241)
    speed = np.interp(vg, vx, vy)
    t0 = []
    for m in ("bus", "tram", "rail"):
        w = c["walk"][m]
        t0.append(half(w["x"], w["y"]) / WALK_SPEED / 60.0)
    ug = np.linspace(0, 6, 241)   # walking time / t0
    shapes = []
    for k, m in enumerate(("bus", "tram", "rail")):
        w = c["walk"][m]
        shapes.append(np.interp(ug * t0[k] * WALK_SPEED * 60.0, w["x"], w["y"]))
    shape = np.mean(shapes, axis=0)
    err = max(float(np.abs(s - shape).max()) for s in shapes)
    p = {"mrc": [c["mode"]["bus"], c["mode"]["tram"], 1.0], "v_anchor": 30.0, "v_bonus": 0.15, "v_n": 2.0, "h_sat": 5.0, "h_p": -0.0658, "h_q": 0.1563,
         "walk_speed": WALK_SPEED, "plateau_m": 0.0, "t0_base": t0, "k": 0.0, "p_shape": 1.0, "w_mode": 1.0, "w_speed": 1.0, "w_headway": 1.0, "agg": "product", "agg_p": 0.0,
         "headway_curve": {"kind": "table", "lo": 1.0, "hi": 1440.0, "scale": "log", "y": [round(float(v), 5) for v in head]},
         "speed_curve": {"kind": "table", "lo": 0.0, "hi": 120.0, "scale": "lin", "y": [round(float(v), 5) for v in speed]},
         "walk_curve": {"kind": "table", "lo": 0.0, "hi": 6.0, "scale": "lin", "y": [round(float(v), 5) for v in shape]}}
    return p, err


def main():
    presets = json.loads(PRESETS.read_text())
    presets = [p for p in presets if not p["name"].startswith("Data-driven")]
    new = []
    for run, name in (("g1_lam05", "Data-driven US (transit use only)"), ("g2_exogenous", "Data-driven US (net of neighbourhood)")):
        r = json.loads((RUNS / f"{run}.json").read_text())
        p, err = params_from(r["curves"])
        print(f"{name}: walk shape approximation max error {err:.3f}; t0 (min) bus/tram/rail {[round(x, 1) for x in p['t0_base']]}", flush=True)
        new.append({"name": name, "params": p, "gamma": 1.0})
    presets = presets[:2] + new + presets[2:]
    PRESETS.write_text(json.dumps(presets))
    (PRESETS.parent / "version.txt").write_text(time.strftime("%Y%m%d%H%M%S"))
    print("presets", len(presets), flush=True)


if __name__ == "__main__":
    main()
