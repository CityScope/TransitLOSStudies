"""Browser data and page of a city's flex map: the city's normal map (`<city>/map.html`, same tiles and tabs) with the "Scoring" tab of the
flex maps (`tools/flex_web/flex_tab.js`, shared with the US flex map), written next to it as `<city>/flex.html`.

    python flex/build_city_flex_web.py <city> [<city> ...]

Reads `<city>/flex/` (run_city_flex.py) and the city's res-11 census table (`results/downloads/h3_res11.parquet`: population, jobs and age
counts per cell). Writes
    <city>/flex_web/<city>/   station profiles, cell / station table, cell geometry and counts (transitlos.flex.export), meta.cell_res = 11
    <city>/flex_web/<city>/routes/   the map editor's routes and departures (transitlos.flex.export.export_routes; run_city_flex.py saves them)
    <city>/flex_web/index.json
    <city>/flex_js/           score.js, editor.js, edit.js (transitlos.flex frontend), flex_tab.js, flex_edit.js
    <city>/flex.html, <city>/flex_presets.json (the US fits' presets)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
TOOLS = HERE.parent / "tools" / "flex_web"
#: res-11 table column -> label (the first present of each group; population first: the tab weights the coarser cells by it)
COUNTS = [(("census_residents", "population"), "Population"), (("workplaceJobs", "ba_jobsTotal", "lodes_wac_jobsTotal"), "Jobs"),
          (("under18Population",), "Population under 18"), (("over65Population",), "Population 65+")]


def cell_counts(city: str, ids: np.ndarray) -> pd.DataFrame:
    """Counts of the flex cells (res 11) from the city's finest census table (res 11; else the finest coarser one, each cell's count split
    evenly over its res-11 children); cells not in it get 0."""
    import h3
    import pyarrow.parquet as pq

    d = HERE / city / "results" / "downloads"
    r = next(r for r in range(11, 4, -1) if (d / f"h3_res{r}.parquet").is_file())
    f = d / f"h3_res{r}.parquet"
    names = pq.read_schema(f).names
    use = [(next(c for c in cands if c in names), lab) for cands, lab in COUNTS if any(c in names for c in cands)]
    t = pd.read_parquet(f, columns=["h3_cell", *[c for c, _ in use]]).drop_duplicates("h3_cell").set_index("h3_cell")
    keys = ids if r == 11 else np.array([h3.cell_to_parent(h, r) for h in ids])
    share = 1.0 / 7 ** (11 - r)
    out = pd.DataFrame(index=range(len(ids)))
    for c, lab in use:
        out[lab] = (pd.to_numeric(t[c], errors="coerce").reindex(keys).fillna(0.0).to_numpy() * share).astype(np.float32)
    return out


def raw_stop_names(city: str, pts: pd.DataFrame) -> pd.DataFrame:
    """The stops' names as their feeds write them (stops.txt; the loaded feed can lose non-Latin names, e.g. Taipei's): by stop_id, a feed's
    renamed duplicate ids (`<id>_file_<n>`) included; names not found stay."""
    import re

    names: dict = {}
    for f in sorted((HERE / city / "gtfs").glob("*/stops.txt")):
        try:
            t = pd.read_csv(f, usecols=["stop_id", "stop_name"], dtype=str, keep_default_na=False)
        except (ValueError, OSError):
            continue
        for i, n in zip(t.stop_id, t.stop_name):
            if n.strip():
                names.setdefault(i, n.strip())
    pts = pts.copy()
    raw = [names.get(re.sub(r"_file_\d+$", "", str(i))) for i in pts["stop_id"]]
    pts["stop_name"] = [r if r else n for r, n in zip(raw, pts["stop_name"].astype(str))]
    return pts


def detour_factor(lat, lon, cs: pd.DataFrame, n: int = 200_000) -> float:
    """Median street distance / straight distance from cells to their nearest station (pairs over 150 m), for the editor's new stations."""
    import h3

    rng = np.random.default_rng(0)
    rows = rng.choice(len(cs), size=min(n, len(cs)), replace=False)
    s = cs["stop_0"].to_numpy()[rows]; d = cs["dist_0"].to_numpy(float)[rows]; ok = (s >= 0) & (d < 65535)
    ll = np.array([h3.cell_to_latlng(h) for h in cs["h3"].to_numpy()[rows][ok]])
    la, lo = np.asarray(lat, float)[s[ok]], np.asarray(lon, float)[s[ok]]
    dy = (ll[:, 0] - la) * 110574.0; dx = (ll[:, 1] - lo) * 111320.0 * np.cos(np.radians(la))
    st = np.hypot(dx, dy); keep = st > 150
    return float(np.clip(np.median(d[ok][keep] / st[keep]), 1.0, 3.0)) if keep.any() else 1.3


def build(city: str) -> Path:
    """Data, scripts and page of one city (see the module docstring).

    Args:
        city: City folder name.

    Returns:
        The page written.
    """
    from transitlos.flex import core, scoring as sc
    from transitlos.flex.cells import CellStops
    from transitlos.flex.export import FRONTEND, export_for_browser, export_routes

    src = HERE / city / "flex"
    st = pd.read_parquet(src / "stop_profiles.parquet")
    P = sum(1 for c in st.columns if c.startswith("mode_"))
    prof = core.Profiles(np.column_stack([st[f"mode_{j}"] for j in range(P)]), np.column_stack([st[f"h_{j}"] for j in range(P)]),
                         np.column_stack([st[f"v_{j}"] for j in range(P)]))
    cs = pd.read_parquet(src / "cell_stops.parquet")
    K = sum(1 for c in cs.columns if c.startswith("stop_"))
    order = np.argsort([int(h, 16) for h in cs["h3"]], kind="stable")
    cs = cs.iloc[order].reset_index(drop=True)
    cells = np.array([int(h, 16) for h in cs["h3"]], dtype=np.uint64)
    res = CellStops(cells, np.column_stack([cs[f"stop_{k}"] for k in range(K)]).astype(np.int32),
                    np.column_stack([cs[f"dist_{k}"] for k in range(K)]).astype(np.uint16), np.zeros(len(cells), np.uint8), np.zeros(len(cells), np.uint8), {})
    counts = cell_counts(city, cs["h3"].to_numpy())
    meta_run = json.loads((src / "params.json").read_text())
    web_root = HERE / city / "flex_web"
    if web_root.exists():
        shutil.rmtree(web_root)
    web = web_root / city
    pts = pd.read_parquet(src / "stops_by_id.parquet") if (src / "stops_by_id.parquet").is_file() else None
    if pts is not None:
        pts = raw_stop_names(city, pts)
    meta = export_for_browser(web, st.lat.to_numpy(), st.lon.to_numpy(), prof, res, sc.DEFAULT, counts=counts,
                              count_labels={c: c for c in counts.columns}, extra_meta={"name": city, "cell_res": int(meta_run.get("resolution", 11))},
                              points=pts)
    from run_city_flex import route_parts
    from transitlos.flex.profiles import END, START

    parts = route_parts(city)
    if parts:
        det = detour_factor(st.lat.to_numpy(), st.lon.to_numpy(), cs)
        rr = export_routes(web, st.lat.to_numpy(), st.lon.to_numpy(), parts, START.hour * 3600 + START.minute * 60, END.hour * 3600 + END.minute * 60,
                           detour=det)
        print(f"{city}: editor routes {rr}, detour {det:.2f}", flush=True)
    bbox = meta.get("bbox") or meta.get("stations_lonlat")
    (web_root / "index.json").write_text(json.dumps({"states": [{"name": city, "path": f"{city}/", "bbox": bbox}]}))
    js = HERE / city / "flex_js"
    js.mkdir(exist_ok=True)
    for f in ("score.js", "editor.js", "edit.js"):
        shutil.copy2(FRONTEND / f, js / f)
    for f in ("flex_tab.js", "flex_edit.js"):
        shutil.copy2(TOOLS / f, js / f)
    page = (HERE / city / "map.html").read_text(encoding="utf-8")
    inject = ('<script src="https://cdn.jsdelivr.net/npm/h3-js@4.1.0/dist/h3-js.umd.js"></script>\n'
              '<script>window.FLEX_CFG = {base: "flex_web/", presets: "flex_presets.json", maxStates: 1, minZoom: 8};</script>\n'
              '<script src="flex_js/score.js"></script><script src="flex_js/editor.js"></script><script src="flex_js/flex_tab.js"></script>\n'
              '<script src="flex_js/edit.js"></script><script src="flex_js/flex_edit.js"></script>\n')
    page = page.replace("</body>", inject + "</body>", 1).replace("<title>", "<title>Flexible scoring · ", 1)
    out = HERE / city / "flex.html"
    out.write_text(page, encoding="utf-8")
    # the scoring presets of the US fits (built by usa_flex/build_flex_page.py; the final form first) -- data, not code
    us = HERE.parent / "usa_study" / "data" / "_usa" / "flex_presets.json"
    (HERE / city / "flex_presets.json").write_text(us.read_text(encoding="utf-8") if us.is_file() else "[]", encoding="utf-8")
    print(f"{city}: {meta['S']} stations x {meta['P']} profiles, {meta['nc']} res-{meta['cell_res']} cells, K {meta['K']} -> {out}", flush=True)
    return out


if __name__ == "__main__":
    for c in sys.argv[1:]:
        build(c)
