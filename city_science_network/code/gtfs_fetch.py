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
# Official live feeds fetched IN ADDITION to the Mobility Database ids of a city: the archived copy of these in the Mobility
# Database had already expired (SFMTA ended 2026-08-28, Renfe high-speed 2026-10-14), the agency's own URL serves the current one.
EXTRA_URL_FEEDS = {
    "san_francisco": [("sfmta_muni", "https://muni-gtfs.apps.sfmta.com/data/muni_gtfs-current.zip"),
                      # operators of the metro area missing from the Mobility Database ids of the city (Marin, Tri-Valley, Union City, Altamont): live agency URL or the Mobility Database copy
                      ("goldengate_transit", "https://realtime.goldengate.org/gtfsstatic/GTFSTransitData.zip"),
                      ("lavta_wheels", "https://storage.googleapis.com/storage/v1/b/mdb-latest/o/us-california-livermore-amador-valley-transit-authority-gtfs-64.zip?alt=media"),
                      ("union_city_transit", "https://storage.googleapis.com/storage/v1/b/mdb-latest/o/us-california-union-city-transit-gtfs-628.zip?alt=media"),
                      ("smart_rail", "https://storage.googleapis.com/storage/v1/b/mdb-latest/o/us-california-sonoma-marin-area-rail-transit-smart-gtfs-815.zip?alt=media"),
                      ("ace_rail", "https://storage.googleapis.com/storage/v1/b/mdb-latest/o/us-california-altamont-corridor-express-gtfs-1133.zip?alt=media"),
                     ],
    "gipuzkoa": [("renfe_alta_velocidad", "https://ssl.renfe.com/gtransit/Fichero_AV_LD/google_transit.zip")],
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


def restore_archives(gtfs: Path) -> None:
    """Extract the feeds archived by `archive_feeds.py` (`<city>/gtfs_archive/<feed>.zip` or `<feed>.zip.001, .002, ...`) when `gtfs/<feed>/` is missing."""
    arch = gtfs.parent / "gtfs_archive"
    if not arch.is_dir():
        return
    names = {p.name.split(".zip")[0] for p in arch.iterdir() if ".zip" in p.name}
    for name in sorted(names):
        if (gtfs / name).is_dir() and any((gtfs / name).iterdir()):
            continue
        parts = sorted(arch.glob(f"{name}.zip.[0-9][0-9][0-9]"))
        zpath = arch / f"{name}.zip"
        tmp = None
        if parts:
            tmp = arch / f".{name}.joined.zip"
            with open(tmp, "wb") as out:
                for p in parts:
                    out.write(p.read_bytes())
            zpath = tmp
        elif not zpath.is_file():
            continue
        if _is_flat_feed(zpath):
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(gtfs / name)
        if tmp is not None:
            tmp.unlink()
        print(f"[gtfs] restored {gtfs.parent.name}/{name} from gtfs_archive", flush=True)


def _is_flat_feed(zpath: Path) -> bool:
    with zipfile.ZipFile(zpath) as zf:
        return any(n == "routes.txt" or n == "stops.txt" for n in zf.namelist())


def ensure_gtfs(key: str) -> bool:
    """Make sure `<key>/gtfs/` has its feeds; fetch/build whatever is missing. Returns False if the city can't run."""
    gtfs = ROOT / key / "gtfs"
    gtfs.mkdir(parents=True, exist_ok=True)
    restore_archives(gtfs)
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
            for folder, url in EXTRA_URL_FEEDS.get(key, []):
                if not _has_feed(gtfs, folder):
                    _fetch_url_zip(gtfs, folder, url)
    except Exception as exc:  # noqa: BLE001 -- report, let main.py skip this city
        print(f"[gtfs] {key}: could not obtain GTFS: {exc!r}", flush=True)
        return False
    return True


# ---- other shared inputs: WorldPop population raster and the OSM street extract of each city ----------------------------------
from studytools.inputs import WORLDPOP_URL, fetch_file as _fetch_file  # noqa: E402  (shared with the other studies)
#: Geofabrik path of every `CityConfig.pbf_filename` (the downloader would otherwise pick the smallest region covering the AOI, which for
#: a hand-drawn polygon such as Andorra's can be all of Europe: 30+ GB).
GEOFABRIK_PBF = {
    "us_massachusetts.osm.pbf": "north-america/us/massachusetts-latest.osm.pbf",
    "northern_california.osm.pbf": "north-america/us/california/norcal-latest.osm.pbf",
    "mexico.osm.pbf": "north-america/mexico-latest.osm.pbf",
    "spain_paisvasco.osm.pbf": "europe/spain/pais-vasco-latest.osm.pbf",
    "canada_ontario.osm.pbf": "north-america/canada/ontario-latest.osm.pbf",
    "andorra.osm.pbf": "europe/andorra-latest.osm.pbf",
    "germany.osm.pbf": "europe/germany-latest.osm.pbf",
    "taiwan.osm.pbf": "asia/taiwan-latest.osm.pbf",
    "china.osm.pbf": "asia/china-latest.osm.pbf",
    "israel_and_palestine.osm.pbf": "asia/israel-and-palestine-latest.osm.pbf",
    "chile.osm.pbf": "south-america/chile-latest.osm.pbf",
}


def ensure_inputs(key: str) -> bool:
    """Make sure the city's WorldPop raster (`worldpop/`) and OSM extract (`streets/`) are on disk; False if one can't be fetched."""
    from code.city_config import CITY_CONFIGS

    cfg = CITY_CONFIGS[key]
    try:
        tif = ROOT / "worldpop" / cfg.worldpop_filename
        if not tif.is_file():
            iso = cfg.worldpop_filename.split("_")[0]
            print(f"[inputs] {key}: downloading WorldPop raster {tif.name}", flush=True)
            _fetch_file(WORLDPOP_URL.format(ISO=iso.upper(), iso=iso), tif)
        pbf = ROOT / "streets" / cfg.pbf_filename
        if not pbf.is_file() and cfg.pbf_filename in GEOFABRIK_PBF:
            print(f"[inputs] {key}: downloading OSM extract {pbf.name}", flush=True)
            _fetch_file("https://download.geofabrik.de/" + GEOFABRIK_PBF[cfg.pbf_filename], pbf)
    except Exception as exc:  # noqa: BLE001
        print(f"[inputs] {key}: could not obtain inputs: {exc!r}", flush=True)
        return False
    return True
