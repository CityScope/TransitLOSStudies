"""Res 10 vs res 11 on a few metros: does a finer grid change the fitted curves (above all the first 100-200 m of walking)?

    $TORCH_PYTHON fit/res11_test.py build     # cells, items, populations of the test counties at res 10 and 11 -> fit/data/res11/
    $TORCH_PYTHON fit/res11_test.py fit       # the same multitask fit at both resolutions -> fit/data/torch_runs/res11test_r{10,11}*

Both resolutions come out of the same street-network search and the same code (K nearest stations with network walking distance, plus rail / tram
beyond those, as fit/build_items.py), so only the grid differs. The fit: cell-population-weighted block-group scores, free mode values, balanced
weights, no work-end term (the flows of the test counties alone would be cut at their borders), held-out counties as the CV folds.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
if sys.argv[1:2] == ["build"]:   # build: the network stack (main python); fit: a Python with PyTorch ($TORCH_PYTHON)
    from scipy.spatial import cKDTree
    import run_county as rc  # noqa: E402
    from transitlos.flex import core  # noqa: E402
    from transitlos.flex import catchment as ct  # noqa: E402
    Net = rc.Net
else:
    import torch

OUT = HERE / "data" / "res11"
EXTRA_MAX_M = 3000.0
#: test counties: metros with bus, rail and tram, of different form (state abbreviation, county GEOID, metro label)
COUNTIES = [("MA", "25025", "Boston"), ("MA", "25017", "Boston"), ("IL", "17031", "Chicago"), ("WA", "53033", "Seattle"), ("CA", "06075", "San Francisco"),
            ("CO", "08031", "Denver"), ("PA", "42101", "Philadelphia"), ("MN", "27053", "Minneapolis"),
            ("OR", "41051", "Portland"), ("GA", "13121", "Atlanta")]


def xy(lon, lat, lat0):
    return np.column_stack([np.radians(lon) * 6371000.0 * np.cos(np.radians(lat0)), np.radians(lat) * 6371000.0])


def county_items(stations, prof, cells, cs, cd, cfg):
    """Items (cell row, mode, headway, speed, walk m) of the K nearest stations, plus the nearest tram / rail station beyond them (straight line x detour)."""
    mode, h, v = prof.mode, prof.h, prof.v
    ci, kk = np.nonzero(cs >= 0)
    st = cs[ci, kk]; dist = cd[ci, kk].astype(np.float32)
    ii, jj = np.nonzero(mode[st] >= 0)
    it = [(ci[ii], st[ii], jj, dist[ii])]
    lat, lon = ct.cell_centres(cells)
    lat0 = float(np.median(lat))
    pc = xy(lon, lat, lat0); ps = xy(stations.lon.to_numpy(), stations.lat.to_numpy(), lat0)
    straight = np.linalg.norm(pc[ci] - ps[st], axis=1)
    okr = (dist > 100) & (straight > 50)
    detour = float(np.median(dist[okr] / straight[okr])) if okr.any() else 1.3
    for m in (1, 2):
        sid = np.flatnonzero((mode == m).any(1))
        if not len(sid):
            continue
        dst, ix = cKDTree(ps[sid]).query(pc, k=1, distance_upper_bound=EXTRA_MAX_M / detour)
        okc = np.isfinite(dst)
        cc = np.flatnonzero(okc); cst = sid[ix[okc]]
        inK = (cs[cc] == cst[:, None]).any(1)
        cc, cst, cdd = cc[~inK], cst[~inK], (dst[okc][~inK] * detour).astype(np.float32)
        a, j = np.nonzero(mode[cst] == m)
        it.append((cc[a], cst[a], j, cdd[a]))
    c_, s_, j_, d_ = (np.concatenate(x) for x in zip(*it))
    return c_, mode[s_, j_].astype(np.int8), h[s_, j_].astype(np.float32), v[s_, j_].astype(np.float32), d_


REASSIGN_M = 150.0   # a block whose own cell has no street the search reached goes to the nearest listed cell within this (res-10 cell size)


def block_pop_to_cells(blocks, cells, lat, lon, res):
    """Each census block's population to the cell under its representative point, or to the nearest listed cell within REASSIGN_M: at res 11 many
    cells hold no street segment, and dropping their residents would count them as having no service."""
    import h3
    rp = blocks.geometry.representative_point()
    bc = np.array([int(h3.latlng_to_cell(y, x, res), 16) for y, x in zip(rp.y, rp.x)], dtype=np.uint64)
    pos = np.searchsorted(cells, bc); pos = np.minimum(pos, len(cells) - 1)
    hit = cells[pos] == bc
    lat0 = float(np.median(lat))
    _, nn = cKDTree(xy(lon, lat, lat0)).query(xy(rp.x.to_numpy(), rp.y.to_numpy(), lat0), k=1, distance_upper_bound=REASSIGN_M)
    near = nn < len(cells)
    tgt = np.where(hit, pos, np.where(near, np.minimum(nn, len(cells) - 1), -1))
    pop = np.zeros(len(cells), np.float64)
    ok = tgt >= 0
    np.add.at(pop, tgt[ok], blocks["population"].to_numpy(float)[ok])
    return pop.astype(np.float32)


def build():
    import geopandas as gpd
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = core.Config()
    by_state = {}
    for abbr, g, metro in COUNTIES:
        by_state.setdefault(abbr, []).append((g, metro))
    for abbr, cl in by_state.items():
        name = rc.STATE_DIRS[abbr]
        state_dir = rc.STUDY / "data" / name
        t0 = time.time()
        stations, prof, cut = rc.state_profiles(state_dir, HERE.parent / "data" / name / "x", cfg)   # the state's cached candidates
        print(abbr, f"profiles {len(stations):,} stations ({time.time() - t0:.0f}s)", flush=True)
        for g, metro in cl:
            cdir = state_dir / "counties" / g
            net = Net.load(cdir / "_network_cache_b1500")
            sn, so, ss, ok = net.snap(stations.lat.to_numpy(), stations.lon.to_numpy())
            ls, ld, ln, lc, _ = core.multilabel_search(net.indptr, net.adj, net.w, sn, so, ss, prof, cfg)
            reached = ln > 0
            blocks = gpd.read_parquet(cdir / "block.geoparquet")[["GEOID", "population", "geometry"]].to_crs(4326)
            blocks["population"] = pd.to_numeric(blocks["population"], errors="coerce").fillna(0)
            bgs = gpd.read_parquet(cdir / "blockgroup.geoparquet")[["GEOID", "geometry"]].to_crs(4326)
            for res in (10, 11):
                t1 = time.time()
                cells, rec_off, ru, rv, ra, rb, rl = net.edge_cell_records(res, edge_mask=reached[net.eu] | reached[net.ev])
                cs, cd, cf, cr = core.process_cells(rec_off, ru, rv, ra, rb, rl, (ls, ld, ln, lc), prof, cfg, rc.sc.DEFAULT)
                order = np.argsort(cells.astype(np.uint64)); cells = cells.astype(np.uint64)[order]; cs, cd = cs[order], cd[order]
                lat, lon = ct.cell_centres(cells)
                pop = block_pop_to_cells(blocks, cells, lat, lon, res)
                j = gpd.sjoin(gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326), bgs, how="left", predicate="within")
                j = j[~j.index.duplicated()]
                cbg = j["GEOID"].astype(object).where(j["GEOID"].notna(), "").to_numpy()
                ic, im, ih, iv, idd = county_items(stations, prof, cells, cs, cd, cfg)
                np.savez(OUT / f"{g}_r{res}.npz", cells=cells, pop=pop, bg=cbg.astype(str), it_cell=ic, it_mode=im, it_h=ih, it_v=iv, it_d=idd)
                print(f"  {metro} {g} res {res}: {len(cells):,} cells, population in cells {pop.sum():,.0f}, {len(ic):,} items ({time.time() - t1:.0f}s)", flush=True)


class SmallData:
    """The test counties as a Data3-like object (only what Model3 with weights_from="cells" and no work-end term reads)."""

    def __init__(self, res):
        import torch_fit as tf
        import torch_fit2 as t2
        DEV = tf.DEV
        bgt = pd.read_parquet(tf.DATA / "blockgroups.parquet")
        tab = pd.read_parquet(tf.BG_TABLE).set_index("GEOID").reindex(bgt.GEOID)
        bgt = bgt.set_index("GEOID")
        geos = sorted({g for _, g, _ in COUNTIES})
        parts = [dict(np.load(OUT / f"{g}_r{res}.npz", allow_pickle=False)) for g in geos]
        cbg = np.concatenate([p["bg"] for p in parts])
        offs = np.cumsum([0] + [len(p["cells"]) for p in parts])
        keepbg = np.array(sorted(g for g in bgt.index if g[:5] in geos))
        b = bgt.loc[keepbg]; t = tab.loc[keepbg]
        pop = b.population.fillna(0).to_numpy(float); wk = b.acs5_workers.fillna(0).to_numpy(float); y = b.publicTransportCommuters.to_numpy(float)
        ok = (pop > 0) & (wk > 0) & np.isfinite(y)
        geo = keepbg[ok]; b, t = b[ok], t[ok]
        self.nb = len(geo); self.geoid = geo
        idx = {gg: i for i, gg in enumerate(geo)}
        self.y = torch.tensor(np.minimum(np.maximum(y[ok], 0), wk[ok]), dtype=torch.float32, device=DEV)
        self.n = torch.tensor(wk[ok], dtype=torch.float32, device=DEV)
        self.pop = torch.tensor(pop[ok], dtype=torch.float32, device=DEV)
        metro_of = {g: m for _, g, m in COUNTIES}
        self.group_names, self.group = np.unique([metro_of[g[:5]] for g in geo], return_inverse=True)
        self.group_pop = np.bincount(self.group, weights=pop[ok], minlength=len(self.group_names))
        # mode outcomes (ACS B08301), as Data3
        wkm = t["workers_b08301"].to_numpy(float)
        self.n_mode = torch.tensor(np.where(np.isfinite(wkm) & (wkm > 0), wkm, 0), dtype=torch.float32, device=DEV)
        ym = {"bus": t[["bus_commuters"]].sum(axis=1, min_count=1), "rail": t[["subway_commuters", "commuter_rail_commuters"]].sum(axis=1, min_count=1),
              "tram": t[["light_rail_commuters"]].sum(axis=1, min_count=1)}
        self.y_mode, self.ok_mode = {}, {}
        for m, s in ym.items():
            vv = s.to_numpy(float); okm = np.isfinite(vv) & (wkm > 0)
            self.ok_mode[m] = torch.tensor(okm.astype(np.float32), device=DEV)
            self.y_mode[m] = torch.tensor(np.where(okm, np.minimum(np.maximum(vv, 0), wkm), 0), dtype=torch.float32, device=DEV)
        # cells and items
        orig = np.array([idx.get(x, -1) for x in cbg])
        self.ncell = len(cbg)
        self.cell_orig = torch.tensor(np.where(orig >= 0, orig, self.nb), device=DEV)
        self.cell_pop = torch.tensor(np.where(orig >= 0, np.concatenate([p["pop"] for p in parts]), 0), dtype=torch.float32, device=DEV)
        ic = np.concatenate([p["it_cell"] + o for p, o in zip(parts, offs[:-1])]).astype(np.int64)
        im = np.concatenate([p["it_mode"] for p in parts]).astype(np.int64)
        h = np.clip(np.concatenate([p["it_h"] for p in parts]), t2.H_KNOTS[0], t2.H_KNOTS[-1])
        v = np.concatenate([p["it_v"] for p in parts]); d = np.clip(np.concatenate([p["it_d"] for p in parts]), 0, tf.D_KNOTS[-1])
        self.v_median = np.array([np.median(v[(im == k) & (v > 0)]) if ((im == k) & (v > 0)).any() else 20.0 for k in range(3)])
        v = np.clip(v, tf.V_KNOTS[0], tf.V_KNOTS[-1])
        self.it_cell = torch.tensor(ic, device=DEV); self.it_mode = torch.tensor(im, device=DEV)
        self.it_h = torch.tensor(h, dtype=torch.float32, device=DEV); self.it_d = torch.tensor(d, dtype=torch.float32, device=DEV)
        self.hi, self.hf = tf.interp_index(np.log(h), np.log(t2.H_KNOTS))
        self.vi, self.vf = tf.interp_index(v, tf.V_KNOTS)
        self.di, self.df = tf.interp_index(d, tf.D_KNOTS)
        self.has_flows = torch.zeros(self.nb, device=DEV)
        self.keep = torch.ones(self.nb, device=DEV)
        self.service_class = t2.Data2._service_classes(self)
        print(f"res {res}: {self.nb:,} block groups, {self.ncell:,} cells, {len(ic):,} items, metros {list(self.group_names)}", flush=True)


def fit():
    import torch_fit3 as f3
    import loss_points as lp
    for res in (10, 11):
        d = SmallData(res)
        for tag, kw in (("", {}),):
            name = f"res11test_r{res}{tag}"
            o = f3.multistart3(d, name, speed="abs", starts=4, steps=3500, mridge=0.0, balanced=True, steep=0.0, weights_from="cells", dest_tasks=(), **kw)
            # 25% / 50% loss points: score and demand (home end only, no work-end term in this test)
            m = f3.Model3(0, weights_from="cells", dest=False, dest_tasks=()); m.load_state_dict(torch.load(f3.RUNS / f"{name}.pt", map_location=f3.DEV)); m.to(f3.DEV)
            with torch.no_grad():
                cm, lm = m.cell_mode_access(d)
                C = torch.stack([(cm * torch.exp(lm)[None, :]).max(dim=1).values, cm[:, 0], cm[:, 2], cm[:, 1]], 1)
                A = torch.zeros((d.nb + 1, 4), device=C.device).index_add(0, d.cell_orig, d.cell_pop[:, None] * C)[: d.nb] / d.pop.clamp(min=1.0)[:, None]
            P = (A.double().cpu().numpy(), np.zeros((d.nb, 4)), torch.exp(m.leps).double().cpu().numpy(), np.ones(4), np.zeros((d.nb, 4)),
                 (m.alpha[None, :] * torch.ones((d.nb, 4), device=m.alpha.device)).double().cpu().numpy())
            dc = _CpuView(d)
            dem = {"service": lp.demand_curve(P, dc, 0, True)}
            for mo, k in (("bus", 1), ("rail", 2), ("tram", 3)):
                dem[mo] = lp.demand_curve(P, dc, k, False)
            best = min(o["runs"], key=lambda r: r["loss"])
            res_ = {"best": lp.points(best["curves"], dem), "starts_value": [lp.points(r["curves"]) for r in o["runs"]]}
            (f3.RUNS / f"{name}_loss_points.json").write_text(json.dumps(res_, indent=1))
            print(name, json.dumps({k: {q: round(x, 1) for q, x in v.items()} for k, v in res_["best"].items()}), flush=True)


class _CpuView:
    """The few tensors loss_points.demand_curve reads, on the CPU."""

    def __init__(self, d):
        self.n = d.n.cpu(); self.n_mode = d.n_mode.cpu(); self.ok_mode = {k: v.cpu() for k, v in d.ok_mode.items()}


if __name__ == "__main__":
    {"build": build, "fit": fit}[sys.argv[1]]()
