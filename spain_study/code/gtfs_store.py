# -*- coding: utf-8 -*-
"""ONE common folder for every Spain GTFS feed (shared by all provinces and communities).

Layout of ``<gtfs_dir>`` (default ``spain_study/data/gtfs``; override with env ``SPAIN_GTFS_DIR`` or ``main.py --gtfs-dir``; the shared
cache keeps downloaded feeds in ``gtfs/``, so ``TransitLOSStudies/cache/gtfs/spain`` may be a symlink to this folder -- nothing is moved):

    <feed_id>_<operator>.zip      the feed as published (NAP, see nap_fetch.py), named by NAP feed id + dataset name
    manifest.csv                  one row per NAP dataset: id, name, organisation, status, size, dates, error, bounding box of the stops
    index.pkl                     {zip name: (size, stop points)} so stop coordinates are read once per feed
    _extracted/<stem>/            the unzipped feed (created on demand, shared by every province that uses it)

A province selects the feeds that have stops inside its polygon (buffered ``BUFFER_M``) by stop coordinates; nothing is copied into province
or community folders. Feeds whose GTFS files have blanks around the CSV fields (e.g. ``1165_autobuses``: "route_id, service_id, trip_id") are
cleaned once at extraction.
"""
from __future__ import annotations

import csv
import io
import logging
import os
import pickle
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import shapely

logger = logging.getLogger(__name__)
BUFFER_M = 1500.0
EXTRACTED = "_extracted"
INDEX_FILE = "index.pkl"


def default_gtfs_dir(data_dir: Path) -> Path:
    return Path(os.environ.get("SPAIN_GTFS_DIR", Path(data_dir) / "gtfs"))


def zip_stop_points(zpath: Path) -> np.ndarray:
    try:
        with zipfile.ZipFile(zpath) as z:
            name = next((n for n in z.namelist() if n.lower().endswith("stops.txt") and "/" not in n.strip("/")), None) or \
                next((n for n in z.namelist() if n.lower().endswith("stops.txt")), None)
            if name is None:
                return np.zeros((0, 2))
            t = pd.read_csv(z.open(name), dtype=str, encoding_errors="replace", keep_default_na=False, skipinitialspace=True)
        t.columns = [c.strip().lstrip("﻿") for c in t.columns]
        lat = pd.to_numeric(t.get("stop_lat"), errors="coerce")
        lon = pd.to_numeric(t.get("stop_lon"), errors="coerce")
        ok = lat.notna() & lon.notna() & (lat.abs() > 0.01)
        return np.c_[lon[ok].to_numpy(float), lat[ok].to_numpy(float)]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"{zpath.name}: unreadable ({exc})")
        return np.zeros((0, 2))


def feed_index(gtfs_dir: Path) -> dict:
    """``{zip name: stop points (lon, lat)}`` of every zip of the common folder (cached in ``index.pkl``, keyed by name and size)."""
    gtfs_dir = Path(gtfs_dir)
    cache = gtfs_dir / INDEX_FILE
    idx = pickle.loads(cache.read_bytes()) if cache.is_file() else {}
    changed = False
    live = {}
    for z in sorted(gtfs_dir.glob("*.zip")):
        key = (z.name, z.stat().st_size)
        if key not in idx:
            idx[key] = zip_stop_points(z)
            changed = True
        live[z.name] = idx[key]
    if changed:
        cache.write_bytes(pickle.dumps({k: v for k, v in idx.items() if k[0] in live}))
    return live


def feeds_for_province(province_gdf, gtfs_dir: Path) -> List[Path]:
    """Zips of the common folder with at least one stop inside the (buffered) province polygon."""
    g = province_gdf.to_crs(province_gdf.estimate_utm_crs())
    area = g.geometry.buffer(BUFFER_M).to_crs(4326).union_all()
    minx, miny, maxx, maxy = area.bounds
    out = []
    for name, pts in feed_index(gtfs_dir).items():
        if not len(pts):
            continue
        if pts[:, 0].max() < minx or pts[:, 0].min() > maxx or pts[:, 1].max() < miny or pts[:, 1].min() > maxy:   # bounding-box prefilter
            continue
        if shapely.contains_xy(area, pts[:, 0], pts[:, 1]).any():
            out.append(Path(gtfs_dir) / name)
    return sorted(out)


def _clean_gtfs_csv(path: Path) -> None:
    """Strip blanks around header names and fields when a GTFS table has them (``a, b, c`` instead of ``a,b,c``)."""
    with open(path, "rb") as f:
        head = f.readline().decode("utf-8-sig", errors="replace")
    cols = head.rstrip("\r\n").split(",")
    if all(c == c.strip() for c in cols):
        return
    text = path.read_bytes().decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text), skipinitialspace=True))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([c.strip() for c in r])


def extract(zpath: Path, gtfs_dir: Path) -> Optional[Path]:
    """Directory with the unzipped feed ``<gtfs_dir>/_extracted/<stem>`` (created once)."""
    dest = Path(gtfs_dir) / EXTRACTED / Path(zpath).stem
    if not (dest / "stops.txt").is_file():
        tmp = dest.with_name(dest.name + ".part")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(tmp)
        src = tmp
        if not (tmp / "stops.txt").is_file():
            inner = [p for p in tmp.iterdir() if p.is_dir() and (p / "stops.txt").is_file()]   # zip with a top-level folder
            if not inner:
                logger.warning(f"{zpath.name}: no stops.txt")
                return None
            src = inner[0]
        for t in src.glob("*.txt"):
            _clean_gtfs_csv(t)
        if dest.exists():
            import shutil

            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dest)
        if tmp.exists():
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
    return dest


def province_feed_dirs(province_gdf, gtfs_dir: Path) -> List[Path]:
    dirs = [d for d in (extract(z, gtfs_dir) for z in feeds_for_province(province_gdf, gtfs_dir)) if d is not None]
    logger.info(f"{len(dirs)} feeds touch this province: {[d.name for d in dirs]}")
    from pyGTFSHandler.utils.feed_dedupe import find_duplicate_feeds

    dup = find_duplicate_feeds(dirs)   # one operator listed twice (own file + regional copy) would count every trip twice
    if dup:
        logger.info(f"duplicate feeds dropped: {[f'{a.name} (= {b.name})' for a, b in dup.items()]}")
    return [d for d in dirs if d not in dup]


FEEDS_FILE = "feeds.json"


def write_province_feeds(prov_dir: Path, dirs) -> None:
    """Record which feeds serve a province (names of the ``_extracted`` folders)."""
    import json

    (Path(prov_dir) / FEEDS_FILE).write_text(json.dumps(sorted(Path(d).name for d in dirs)))


def ccaa_feed_dirs(ccaa_dir: Path, gtfs_dir: Path) -> List[Path]:
    """Extracted feed folders used by the provinces of a community (union of the provinces' ``feeds.json``)."""
    import json

    stems = set()
    for f in sorted((Path(ccaa_dir) / "provinces").glob(f"*/{FEEDS_FILE}")):
        stems.update(json.loads(f.read_text()))
    return [d for d in (Path(gtfs_dir) / EXTRACTED / s for s in sorted(stems)) if (d / "stops.txt").is_file()]
