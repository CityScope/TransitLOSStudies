"""Census blocks -> the H3 res-10 cell under their centre, for population weighting with the exact census block populations.

    python fit/build_blocks.py     -> fit/data/torch/blocks.npz (block group row, block population, global cell index in cells.npz or -1)
A block whose cell is not in cells.npz (no stop within reach, so no stored cell) gets -1: level of service 0. Block groups and tracts then take
sum(block pop x cell level of service) / sum(block pop), the census-population-weighted mean.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import run_county as rc  # noqa: E402

import paths as P  # noqa: E402
WEB = P.WEB
STUDY = HERE.parents[1] / "usa_study" / "data"


def kept_cells(name):
    """Same selection and order as build_items.one_state: (web cell ids of the kept cells, in order)."""
    w = WEB / name
    if not (w / "meta.json").is_file():
        return None
    meta = json.load(open(w / "meta.json")); nc = meta["nc"]
    z = np.load(P.DATA / f"cells_{name}.npz")
    bg, pop = z["bg"], z["pop"].astype(np.float32)
    jobs = np.zeros(nc, np.float32)
    if "lodes_wac_jobsTotal" in meta.get("count_columns", []):
        cc = np.fromfile(w / "cell_counts.bin", np.float32).reshape(nc, len(meta["count_columns"]))
        jobs = cc[:, meta["count_columns"].index("lodes_wac_jobsTotal")]
    keep = (bg >= 0) & ((pop > 0) | (jobs > 0))
    return np.flatnonzero(keep), nc


def one_state(name):
    import geopandas as gpd
    import pandas as pd
    from h3ronpy.vector import coordinates_to_cells

    k = kept_cells(name)
    if k is None:
        return None
    kept, nc = k
    # the web cells' h3 ids from their stored centres (cell_geom.bin columns 0-1: lon, lat), in the stored order
    geom = np.fromfile(WEB / name / "cell_geom.bin", np.float32).reshape(nc, 14)
    if P.BUF:   # mixed res-10 / res-8 cells: their real ids (same order as the web export)
        import pandas as _pd
        web_ids = np.array([int(h, 16) for h in _pd.read_parquet(P.FLEX_DATA / name / 'cells.parquet', columns=['h3'])['h3']], dtype=np.uint64)
    else:
        web_ids = np.asarray(coordinates_to_cells(geom[:, 1].astype(np.float64), geom[:, 0].astype(np.float64), 10)).astype(np.uint64)
    order = np.argsort(web_ids); web_cells = web_ids[order]
    glob_web = np.full(nc, -1, np.int64); glob_web[kept] = np.arange(len(kept))
    glob_of_web = glob_web[order]                                         # aligned with the sorted ids
    fp = P.DATA / f"far_cells_{name}.npz"   # the far-access cells of build_items (after the kept web cells, in this order)
    far = np.load(fp)["h3"].astype(np.uint64) if fp.is_file() else np.zeros(0, np.uint64)
    if len(far):
        web_cells = np.r_[web_cells, far]; glob_of_web = np.r_[glob_of_web, len(kept) + np.arange(len(far))]
        o2 = np.argsort(web_cells, kind="stable"); web_cells, glob_of_web = web_cells[o2], glob_of_web[o2]
    rows_bg, pops, cells = [], [], []
    if P.BUF:
        # Buffered flex build: the stored cells' populations already come from the census blocks (run_state_buffered: walk-zone blocks split
        # by NSI floor area over the res-10 cells whose centre they hold -- one block, so one block group, per cell; blocks outside the walk
        # zones summed onto the res-8 cell of their representative point; then one rescale factor per state). The pieces here reproduce that
        # exactly: each res-10 cell is one piece of its block group; a res-8 cell is split by the block groups of its blocks, scaled to the
        # cell's stored population.
        import h3ronpy as _h
        cp = pd.read_parquet(P.FLEX_DATA / name / "cells.parquet", columns=["h3", "res", "bg", "population"])
        g_all = glob_web   # web order -> global index (-1: not kept)
        fine = (cp["res"].to_numpy() == 10)
        fpop = cp["population"].to_numpy(np.float64)
        rows_bg.append(cp["bg"].astype(str).str[:12].to_numpy()[fine]); pops.append(fpop[fine].astype(np.float32)); cells.append(g_all[fine])
        coarse_ids = web_ids[~fine]
        if len(coarse_ids):
            co = np.argsort(coarse_ids); cs_ids = coarse_ids[co]; cs_web = np.flatnonzero(~fine)[co]
            raw = np.zeros(len(cs_ids)); pb, pc, pp = [], [], []
            for f in sorted((STUDY / name / "counties").glob("*/block.geoparquet")):
                b = gpd.read_parquet(f, columns=["GEOID", "population", "geometry"])
                b = b[pd.to_numeric(b["population"], errors="coerce").fillna(0) > 0]
                if b.empty:
                    continue
                rp = b.geometry.to_crs(4326).representative_point()
                c8 = np.asarray(coordinates_to_cells(np.asarray(rp.y), np.asarray(rp.x), 8)).astype(np.uint64)
                pos = np.searchsorted(cs_ids, c8); hit = (pos < len(cs_ids)) & (cs_ids[np.minimum(pos, len(cs_ids) - 1)] == c8)
                pos = pos[hit]; v = pd.to_numeric(b["population"]).to_numpy(np.float64)[hit]
                np.add.at(raw, pos, v)
                pb.append(b["GEOID"].astype(str).str[:12].to_numpy()[hit]); pc.append(pos); pp.append(v)
            if pb:
                pb, pc, pp = np.concatenate(pb), np.concatenate(pc), np.concatenate(pp)
                stored = fpop[cs_web]
                scale = np.where(raw > 0, stored / np.where(raw > 0, raw, 1), 0.0)
                rows_bg.append(pb); pops.append((pp * scale[pc]).astype(np.float32)); cells.append(g_all[cs_web][pc])
    else:
      for f in sorted((STUDY / name / "counties").glob("*/block.geoparquet")):
        b = gpd.read_parquet(f, columns=["GEOID", "population", "geometry"])
        b = b[pd.to_numeric(b["population"], errors="coerce").fillna(0) > 0]
        if b.empty:
            continue
        rp = b.geometry.to_crs(4326).representative_point()
        c = np.asarray(coordinates_to_cells(np.asarray(rp.y), np.asarray(rp.x), 10)).astype(np.uint64)
        pos = np.searchsorted(web_cells, c); hit = (pos < len(web_cells)) & (web_cells[np.minimum(pos, len(web_cells) - 1)] == c)
        g = np.where(hit, glob_of_web[np.minimum(pos, len(web_cells) - 1)], -1)
        rows_bg.append(b["GEOID"].astype(str).str[:12].to_numpy()); pops.append(pd.to_numeric(b["population"]).to_numpy(np.float32)); cells.append(g)
    if not rows_bg:
        return {"name": name, "n_kept": len(kept) + len(far), "bg": np.zeros(0, object), "pop": np.zeros(0, np.float32), "cell": np.zeros(0, np.int64)}
    return {"name": name, "n_kept": len(kept) + len(far), "bg": np.concatenate(rows_bg), "pop": np.concatenate(pops), "cell": np.concatenate(cells)}


def main():
    import pandas as pd
    names = [rc.STATE_DIRS[a] for a in rc.STATE_DIRS]
    rows = pd.Series(np.arange(len(pd.read_parquet(P.DATA / "blockgroups.parquet", columns=["GEOID"]))),
                     index=pd.read_parquet(P.DATA / "blockgroups.parquet", columns=["GEOID"]).GEOID)
    out_bg, out_pop, out_cell, off = [], [], [], 0
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(one_state, names):   # same order as build_items: global cell offsets add up identically
            if r is None:
                continue
            bgr = rows.reindex(r["bg"]).fillna(-1).to_numpy(np.int64)
            out_bg.append(bgr); out_pop.append(r["pop"]); out_cell.append(np.where(r["cell"] >= 0, r["cell"] + off, -1))
            hit = r["cell"] >= 0
            print(f"{r['name']}: blocks {len(r['pop']):,}, population in a stored cell {r['pop'][hit].sum() / max(r['pop'].sum(), 1):.1%}", flush=True)
            off += r["n_kept"]
    np.savez(P.DATA / "torch" / "blocks.npz", bg=np.concatenate(out_bg), pop=np.concatenate(out_pop), cell=np.concatenate(out_cell))
    print("total cells", off, flush=True)


if __name__ == "__main__":
    main()
