"""Fetch or rebuild each city's `gtfs/` inputs when they are not on disk yet.

`main.py` calls `ensure_gtfs(key)` before running a city. GTFS is gitignored (hundreds of MB to GBs),
so on a fresh clone this downloads or builds it. API keys are read from the gitignored
`api_keys.json` files (see pyGTFSHandler's `downloaders/utils/config.py`), never from the repo.

Sources per city:
  * Mobility Database ids listed in `gtfs_mdb_feeds.json` (boston, gipuzkoa, guadalajara, san_francisco, toronto)
  * a zip committed in the repo (concepcion: no public download exists), gtfs.de (hamburg), Israel Ministry of Transport (beerseba)
  * local builder scripts (andorra, shanghai -- the latter needs the CPTOND-2025 dataset from figshare)
  * taipei: TDX bus feed needs a TDX account and cannot be fetched here -- must be copied manually.

Re-downloading gives the CURRENT snapshot of each feed, not the exact files of the original run.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MDB_FEEDS = Path(__file__).resolve().parent / "gtfs_mdb_feeds.json"

URL_FEEDS = {
    "hamburg": ("latest", "https://download.gtfs.de/germany/free/latest.zip"),
    "beerseba": ("israel_mot", "https://gtfs.mot.gov.il/gtfsfiles/israel-public-transportation.zip"),
}
CPTOND_URL = "https://ndownloader.figshare.com/files/58153174"
CPTOND_MD5 = "8b786408d8a9d4d16faa531f98cc9fc6"
CPTOND_CITIES = ("shanghai", "suzhou", "wuxi", "changzhou", "nantong", "jiaxing", "huzhou", "ningbo", "zhoushan")
BUILDERS = {
    "andorra": ("ena_manual_reconstruction", ["andorra/sources/build_andorra_gtfs.py"]),
    "shanghai": (
        "shanghai_metro_area",
        ["china/CPTOND-2025/code/build_shanghai_metro_area_gtfs.py", "--out", "shanghai/gtfs/shanghai_metro_area"],
    ),
}


def _norm(s: str) -> str:
    return s.replace("-", "").lower()


def _has_feed(gtfs: Path, name: str) -> bool:
    """Whether a feed folder (with stops.txt) named `name` -- or, for catalog ids, starting with it -- exists."""
    return any((d / "stops.txt").is_file() and _norm(d.name).startswith(_norm(name)) for d in gtfs.glob("*") if d.is_dir())


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def _fetch_url_zip(gtfs: Path, folder: str, url: str) -> None:
    zip_path = gtfs / f"{folder}.zip"
    print(f"[gtfs] downloading {url}", flush=True)
    _download(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(gtfs / folder)
    zip_path.unlink()


def _fetch_mdb(gtfs: Path, ids: list[str]) -> None:
    from pyGTFSHandler.downloaders.mobility_database import MobilityDatabaseDownloader

    dl = MobilityDatabaseDownloader()
    feeds = []
    for fid in ids:
        raw = dl._authorized_request("GET", f"{dl.GTFS_FEEDS_ENDPOINT}/{fid}")
        meta = dl._to_feed_metadata(raw)
        if meta is not None:
            feeds.append(meta)
    dl.download_feeds(feeds, str(gtfs), overwrite=False, unzip=True)


def _fetch_archive(gtfs: Path, key: str) -> None:
    """Unzip the feed committed under `<key>/gtfs_archive/` (outside `gtfs/`, which the pipeline scans)."""
    for z in sorted((gtfs.parent / "gtfs_archive").glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(gtfs)


def _ensure_cptond() -> None:
    dataset = ROOT / "china" / "CPTOND-2025" / "dataset"
    if all((dataset / "bus" / "shapefiles" / c.capitalize()).is_dir() or (dataset / "bus" / "shapefiles" / c).is_dir()
           for c in CPTOND_CITIES):
        return
    zip_path = ROOT / "china" / "CPTOND-2025.zip"
    if not zip_path.is_file():
        print("[gtfs] downloading CPTOND-2025 (2.2 GB) from figshare", flush=True)
        _download(CPTOND_URL, zip_path)
    md5 = hashlib.md5()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            md5.update(chunk)
    if md5.hexdigest() != CPTOND_MD5:
        raise RuntimeError(f"{zip_path} md5 mismatch (got {md5.hexdigest()}); delete it and retry")
    wanted = re.compile(r"CPTOND-2025/dataset/(bus|metro)/shapefiles/(%s)/" % "|".join(CPTOND_CITIES), re.I)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ROOT / "china", [n for n in z.namelist() if wanted.match(n)])


def _run_builder(key: str) -> None:
    if key == "shanghai":
        _ensure_cptond()
    _, cmd = BUILDERS[key]
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


def ensure_gtfs(key: str) -> bool:
    """Make sure `<key>/gtfs/` has its feeds; fetch/build whatever is missing. Returns False if the city can't run."""
    gtfs = ROOT / key / "gtfs"
    gtfs.mkdir(parents=True, exist_ok=True)
    try:
        if key in BUILDERS:
            if not _has_feed(gtfs, BUILDERS[key][0]):
                print(f"[gtfs] {key}: building feed", flush=True)
                _run_builder(key)
        elif key in URL_FEEDS:
            folder, url = URL_FEEDS[key]
            if not _has_feed(gtfs, folder):
                _fetch_url_zip(gtfs, folder, url)
        elif key == "concepcion":
            if not _has_feed(gtfs, "gran_concepcion"):
                _fetch_archive(gtfs, key)
        elif key == "taipei":
            for folder in ("tdx_bus", "trtc_metro"):
                if not _has_feed(gtfs, folder):
                    print(f"[gtfs] taipei: {gtfs / folder} missing -- the TDX feeds need a TDX account; "
                          "copy them over manually from the original machine", flush=True)
                    return False
        else:
            ids = json.loads(MDB_FEEDS.read_text())[key]
            missing = [i for i in ids if not _has_feed(gtfs, i)]
            if missing:
                print(f"[gtfs] {key}: downloading {len(missing)}/{len(ids)} Mobility Database feeds", flush=True)
                _fetch_mdb(gtfs, missing)
    except Exception as exc:  # noqa: BLE001 -- report, let main.py skip this city
        print(f"[gtfs] {key}: could not obtain GTFS: {exc!r}", flush=True)
        return False
    return True
