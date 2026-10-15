"""Population chunks for the nationwide scenario editor (drawn lines -> recomputed level of service).

`transitlos.map.pop_chunks` format: `pop_chunks/<lat>_<lon>.json` = the populated res-11 cells of one 0.04 deg square
(`[cell, lat, lng, population, level_of_service]`) + `index.json`. The browser fetches only the squares around a
drawn line. Written state by state in parallel (each worker its own files), then merged per square, so no process ever
holds the nation.

Res 11 only exists where LOS > 0 or in dense tracts (see `county_pipeline`); elsewhere population is taken from the
res-7 cells that have no res-11 child, so a new line can still reach people in areas that had no service before
(coarser points, same population totals).
"""
import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h3
import numpy as np
import pandas as pd

from transitlos.map.pop_chunks import POP_CHUNK_DEG, POP_CHUNK_DIRNAME, POP_CHUNK_INDEX, chunk_key

logger = logging.getLogger(__name__)


def _state_part(args):
    state_dir, part_dir = args
    state_dir, part_dir = Path(state_dir), Path(part_dir)
    frames = []
    r11 = state_dir / "h3_res11.geoparquet"
    covered = set()
    if r11.is_file():
        d = pd.read_parquet(r11, columns=["h3", "population", "level_of_service"])
        covered = set(h3.cell_to_parent(c, 7) for c in d["h3"].unique())
        frames.append(d)
    r7 = state_dir / "h3_res7.geoparquet"
    if r7.is_file():
        d = pd.read_parquet(r7, columns=["h3", "population", "level_of_service"])
        frames.append(d[~d["h3"].isin(covered)])
    if not frames:
        return 0
    d = pd.concat(frames, ignore_index=True)
    d = d[d["population"].fillna(0) > 0]
    buckets = {}
    for cell, pop, los in zip(d["h3"], d["population"].to_numpy(float), d["level_of_service"].fillna(0).to_numpy(float)):
        lat, lng = h3.cell_to_latlng(cell)
        buckets.setdefault(chunk_key(lat, lng, POP_CHUNK_DEG), []).append([cell, round(lat, 5), round(lng, 5), round(pop, 3), round(los, 3)])
    part_dir.mkdir(parents=True, exist_ok=True)
    for key, rows in buckets.items():
        (part_dir / f"{key}.json").write_text(json.dumps(rows, separators=(",", ":")))
    return len(d)


def build_usa_pop_chunks(state_dirs, out_dir: Path, workers: int = 16) -> dict:
    out_dir = Path(out_dir) / POP_CHUNK_DIRNAME
    if out_dir.exists():
        shutil.rmtree(out_dir)
    parts = out_dir / "_parts"
    jobs = [(str(d), str(parts / d.name)) for d in state_dirs]
    with ProcessPoolExecutor(workers) as ex:
        n_cells = sum(ex.map(_state_part, jobs))
    by_key = {}
    for f in parts.glob("*/*.json"):
        by_key.setdefault(f.stem, []).append(f)
    for key, files in by_key.items():
        rows = []
        for f in files:
            rows.extend(json.loads(f.read_text()))
        (out_dir / f"{key}.json").write_text(json.dumps({"key": key, "deg": POP_CHUNK_DEG, "res": 11, "cells": rows}, separators=(",", ":")))
    shutil.rmtree(parts)
    (out_dir / POP_CHUNK_INDEX).write_text(json.dumps(
        {"deg": POP_CHUNK_DEG, "res": 11, "n_chunks": len(by_key), "n_cells": n_cells, "keys": sorted(by_key)}, separators=(",", ":")))
    logger.info(f"Population chunks: {len(by_key)} squares, {n_cells} cells -> {out_dir}")
    return {"n_chunks": len(by_key), "n_cells": n_cells}
