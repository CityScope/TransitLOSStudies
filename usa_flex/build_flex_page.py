"""The US flex map page: the normal US map (`usa_study/data/_usa/map.html`, same tiles and tabs) plus the "Flexible scoring" tab
(`tools/flex_web/flex_tab.js`, shared with the city science flex maps), written next to it as `flex.html` with its scripts in `flex_js/` and the scoring variants in `flex_presets.json`.
The per-state flex data (`web_buf/<State>/`, build_web.py) is served from /us_transit_los_flex/.

    python build_flex_page.py [US_MAP_DIR]        # default ../usa_study/data/_usa
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fit"))
from export_presets import params_from, params_from_fit4  # noqa: E402

RUNS = HERE / "fit" / "data_buf" / "torch_runs"
WEB = HERE / "web_buf"
VARIANTS = [   # run, name (the fits of the US fit report, generation 4)
    ("plan_w2_dest", "US fit: walking access + work end"),
    ("plan_w2_dest_steep1e-4", "US fit, steepness cost 1e-4 (4 starts)"),
    ("plan_w2_dest_steep1e-3", "US fit, steepness cost 1e-3"),
    ("plan_w2_dest_equalmodes", "US fit, all modes valued equally"),
    ("plan_w2_dest_structural", "US fit, net of neighbourhood controls"),
    ("plan_w2_nodest", "US fit without the work end"),
    ("plan_w2_dest_literature", "Literature curves (fitted links)"),
]


#: the final form (2026-11-19/20, study 8c): speed curve first, then the mode factor; controls income / poverty / education + extra housing
FINAL = ("g5_controls_w_cs_seq_s8_nofuel", "US fit 2026-11: speed first, then mode (final form)")


def presets() -> list:
    out = []
    f = RUNS / f"{FINAL[0]}.json"
    if f.is_file():
        o = json.loads(f.read_text())
        p, err = params_from_fit4(o["runs"][o["best"]]["params"])
        out.append({"name": FINAL[1], "run": FINAL[0], "params": p, "walk_shape_error": round(err, 3)})
    best_grid = None
    for p in sorted(RUNS.glob("steepgrid_*.json")):
        o = json.loads(p.read_text())
        if o.get("cv") and (best_grid is None or o["cv"]["total"] > best_grid[1]["cv"]["total"]):
            best_grid = (p.stem, o)
    runs = list(VARIANTS)
    if best_grid:   # the 6-start steepness grid: its best held-out skill replaces the 4-start run of the same cost
        cost = float(best_grid[0].split("_")[1])
        runs = [r for r in runs if not (cost == 1e-4 and r[0] == "plan_w2_dest_steep1e-4")]
        runs.insert(1, (best_grid[0], f"US fit, steepness cost {cost:g} (recommended)"))
    for run, name in runs:
        f = RUNS / f"{run}.json"
        if not f.is_file():
            continue
        o = json.loads(f.read_text())
        b = min(o["runs"], key=lambda r: r["loss"])
        p, err = params_from(b["curves"])
        out.append({"name": name, "run": run, "params": p, "walk_shape_error": round(err, 3)})
    return out


def main(us_dir: Path) -> Path:
    page = (us_dir / "map.html").read_text(encoding="utf-8")
    js = us_dir / "flex_js"
    js.mkdir(exist_ok=True)
    for f in ("score.js", "editor.js"):
        shutil.copy2(WEB / f, js / f)
    shutil.copy2(HERE.parent / "tools" / "flex_web" / "flex_tab.js", js / "flex_tab.js")   # shared with the city science flex maps
    (us_dir / "flex_presets.json").write_text(json.dumps(presets()), encoding="utf-8")
    inject = ('<script src="https://cdn.jsdelivr.net/npm/h3-js@4.1.0/dist/h3-js.umd.js"></script>\n'
              '<script>window.FLEX_CFG = {base: "/us_transit_los_flex/", presets: "flex_presets.json", maxStates: 3, minZoom: 6};</script>\n'
              '<script src="flex_js/score.js"></script><script src="flex_js/editor.js"></script><script src="flex_js/flex_tab.js"></script>\n')
    page = page.replace("</body>", inject + "</body>", 1)
    page = page.replace("<title>", "<title>Flexible scoring · ", 1)
    out = us_dir / "flex.html"
    out.write_text(page, encoding="utf-8")
    print("wrote", out, "presets", len(json.loads((us_dir / "flex_presets.json").read_text())))
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "usa_study" / "data" / "_usa")
