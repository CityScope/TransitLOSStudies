# -*- coding: utf-8 -*-
"""Complete / refresh the ONE common GTFS folder (``data/gtfs``) from the NAP, with a manifest.

    python nap_fetch.py [--gtfs-dir data/gtfs] [--rounds 3] [--round-wait 1200] [--delay 2]

* lists the NAP datasets (``nap_download.py list`` logic; the key is never printed), then downloads every dataset that has no zip yet, sequentially
  with a polite delay, exponential back-off on HTTP 429/5xx/timeouts, and several rounds spread over time (``--round-wait`` seconds);
* writes ``<gtfs_dir>/manifest.csv``: feed id, name, organisation, modes, status, size, dates, file, error, attempts, stop bounding box.
  Status values: ``ok`` (zip present), ``ok_expired`` (zip present but its calendar window ended), ``no_file`` (NAP publishes NO file for the dataset:
  size 0 / 0 stops / 0 trips in its own metadata and the download endpoint answers HTTP 500 every time -- NAP lists empty placeholder datasets ``_2`` ... ``_9`` next
  to the one real feed of an operator), ``error`` (a real feed that could not be downloaded; retried on the next run).
* zips are named ``<feed_id>_<operator>.zip`` (existing names are kept).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def slug(s: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")[:n]


def existing_zip(gtfs_dir: Path, feed_id: int):
    """The valid zip of a feed (HTML error pages saved under a .zip name by older downloads do not count)."""
    import zipfile

    hits = [h for h in sorted(gtfs_dir.glob(f"{int(feed_id)}_*.zip")) if zipfile.is_zipfile(h)]
    return hits[0] if hits else None


def fetch_one(url: str, dest: Path, headers: dict, tries: int = 4):
    """(ok, error text); exponential back-off, partial file removed on failure."""
    err = ""
    for k in range(tries):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=180) as r:
                if r.status_code == 200:
                    tmp = dest.with_suffix(".part")
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                    import zipfile

                    if zipfile.is_zipfile(tmp):
                        tmp.replace(dest)
                        return True, ""
                    tmp.unlink(missing_ok=True)
                    err = "HTTP 200 but not a zip"
                else:
                    err = f"HTTP {r.status_code}"
                if r.status_code in (400, 401, 403, 404):
                    break
        except requests.RequestException as e:
            err = type(e).__name__
        time.sleep(min(120, 5 * 2 ** k))
    return False, err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gtfs-dir", default=str(HERE / "data" / "gtfs"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--round-wait", type=int, default=1200)
    ap.add_argument("--delay", type=float, default=2.0)
    a = ap.parse_args()
    gtfs = Path(a.gtfs_dir)
    gtfs.mkdir(parents=True, exist_ok=True)

    import nap_download as nd
    from pyGTFSHandler.downloaders.spain.nap import NAPDownloader

    client = NAPDownloader()
    feeds = client.search_feeds(region=[], transport_type=[], organization=[], keep="newest", file_type=1)
    meta = nd._rows(feeds)
    meta.to_csv(gtfs / "nap_feeds.csv", index=False)
    rows = {int(r.feed_id): dict(r._asdict()) for r in meta.itertuples(index=False)}
    attempts = {i: 0 for i in rows}
    errors = {i: "" for i in rows}
    for rnd in range(1, a.rounds + 1):
        todo = [i for i in rows if existing_zip(gtfs, i) is None]
        print(f"round {rnd}: {len(todo)} datasets without a zip", flush=True)
        for i in todo:
            r = rows[i]
            empty_meta = not (pd.to_numeric(r.get("size_bytes"), errors="coerce") > 0) and not (pd.to_numeric(r.get("n_stops"), errors="coerce") > 0)
            ok, err = fetch_one(r["url"], gtfs / f"{i}_{slug(r['name'])}.zip", client.headers, tries=1 if empty_meta else 5)   # empty placeholders: one try per round
            attempts[i] += 1
            errors[i] = "" if ok else err
            time.sleep(a.delay)
        if rnd < a.rounds and any(existing_zip(gtfs, i) is None for i in rows):
            print(f"  waiting {a.round_wait}s before the next round", flush=True)
            time.sleep(a.round_wait)
    write_manifest(gtfs, rows, attempts, errors)


def write_manifest(gtfs: Path, rows: dict, attempts: dict, errors: dict) -> pd.DataFrame:
    from code.gtfs_store import feed_index

    idx = feed_index(gtfs)
    today = date.today().isoformat()
    out = []
    for i, r in rows.items():
        z = existing_zip(gtfs, i)
        pts = idx.get(z.name) if z else None
        empty_meta = not (pd.to_numeric(r.get("size_bytes"), errors="coerce") > 0) and not (pd.to_numeric(r.get("n_stops"), errors="coerce") > 0)
        if z:
            status = "ok_expired" if (isinstance(r.get("valid_to"), str) and r["valid_to"] and r["valid_to"] < today) else "ok"
        else:
            status = "no_file" if empty_meta else "error"
        bb = (round(float(pts[:, 0].min()), 4), round(float(pts[:, 1].min()), 4), round(float(pts[:, 0].max()), 4), round(float(pts[:, 1].max()), 4)) if pts is not None and len(pts) else (None,) * 4
        out.append({"feed_id": i, "name": r["name"], "organization": r["organization"], "modes": r["modes"], "status": status, "size_bytes": z.stat().st_size if z else 0,
                    "valid_from": r.get("valid_from"), "valid_to": r.get("valid_to"), "updated": r.get("updated"), "file": z.name if z else "",
                    "error": "" if z else (errors.get(i) or ("NAP metadata: no file published" if empty_meta else "")), "attempts": attempts.get(i, 0),
                    "last_attempt": today if attempts.get(i, 0) else "", "n_stops_zip": len(pts) if pts is not None else 0,
                    "min_lon": bb[0], "min_lat": bb[1], "max_lon": bb[2], "max_lat": bb[3]})
    df = pd.DataFrame(out).sort_values("feed_id")
    df.to_csv(gtfs / "manifest.csv", index=False)
    print(df.status.value_counts().to_dict(), "->", gtfs / "manifest.csv")
    return df


if __name__ == "__main__":
    main()
