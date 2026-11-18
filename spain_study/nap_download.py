# -*- coding: utf-8 -*-
"""List / download Spain NAP (nap.transportes.gob.es) GTFS feeds with the existing ``pyGTFSHandler`` NAPDownloader.

The API key is NEVER printed. It is resolved by pyGTFSHandler in this order: env var ``NAP_API_KEY``, then the gitignored
``pyGTFSHandler/api_keys.json`` (entry ``nap_es``), or a JSON/plain-text file given with ``--key-file`` (first line, or {"nap_es": "..."}).

Usage:
    python nap_download.py check                         # is a key available + does the API accept it (prints only OK/FAIL)
    python nap_download.py list --out data/nap_feeds.csv [--region "Gipuzkoa"] [--transport-type Autobus]
Downloading all feeds into the common folder is done by ``nap_fetch.py`` (retries, manifest). ``download`` here remains for ad-hoc subsets
(``--feed-id``, ``--region``) into any folder. NAP keeps only the latest publication per dataset; re-run periodically (feeds expire).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def _load_key(key_file):
    if key_file:
        txt = Path(key_file).read_text().strip()
        try:
            txt = json.loads(txt).get("nap_es", "")
        except Exception:
            txt = txt.splitlines()[0].strip() if txt else ""
        if txt:
            os.environ["NAP_API_KEY"] = txt


def _client():
    from pyGTFSHandler.downloaders.spain.nap import NAPDownloader
    try:
        return NAPDownloader()
    except ValueError:
        sys.exit("No NAP API key found (env NAP_API_KEY, --key-file, or pyGTFSHandler/api_keys.json 'nap_es'). Register at https://nap.transportes.gob.es.")


def _rows(feeds):
    out = []
    for f in feeds:
        raw = f.raw or {}
        ds, fi = raw.get("dataset", {}), raw.get("file", {})
        org = ds.get("organizacion") or {}
        out.append({"feed_id": f.id, "name": f.name, "organization": org.get("nombre") if isinstance(org, dict) else org,
                    "modes": ";".join(t.get("nombre", "") for t in ds.get("tiposTransporte", [])),
                    "valid_from": (fi.get("fechaDesde") or "")[:10], "valid_to": (fi.get("fechaHasta") or "")[:10],
                    "updated": (fi.get("fechaActualizacion") or "")[:10], "n_stops": fi.get("numeroParadas"),
                    "n_routes": fi.get("numeroRutas"), "n_trips": fi.get("numeroViajes"), "size_bytes": fi.get("tamanio"), "url": f.download_url})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "list", "download"])
    ap.add_argument("--key-file")
    ap.add_argument("--region", action="append", default=[])
    ap.add_argument("--transport-type", action="append", default=[])
    ap.add_argument("--organization", action="append", default=[])
    ap.add_argument("--out", default="data/nap_feeds.csv")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--feed-id", type=int, action="append", default=[], help="only these NAP feed ids (see the list CSV); repeatable")
    a = ap.parse_args()
    _load_key(a.key_file)
    c = _client()
    if a.cmd == "check":
        r = c._get("/TipoFichero")
        print("key present; API status:", getattr(r, "status_code", "?"), "(200 = key accepted)")
        return
    feeds = c.search_feeds(region=a.region, transport_type=a.transport_type, organization=a.organization, keep="newest", file_type=1)  # NAP type id 1 = "GTFS-ZIP" (name lookup "GTFS" does not match)
    if a.feed_id:
        feeds = [f for f in feeds if int(f.id) in set(a.feed_id)]
    if a.limit:
        feeds = feeds[: a.limit]
    df = _rows(feeds)
    out = Path(a.out)
    if a.cmd == "list":
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"{len(df)} GTFS files -> {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    paths = c.download_feeds(feeds, str(out), overwrite=False, unzip=False)
    df.to_csv(out / "manifest.csv", index=False)
    print(f"downloaded/cached {len(paths)} feeds into {out}")


if __name__ == "__main__":
    main()
