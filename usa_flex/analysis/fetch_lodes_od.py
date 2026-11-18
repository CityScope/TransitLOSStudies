"""LODES 8 origin-destination files (main + aux, all jobs JT00) for every state, cached in TransitLOSStudies/cache/lodes_od/ (the latest year published for each state)."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parents[2] / "cache" / "lodes_od"
ST = ["al", "ak", "az", "ar", "ca", "co", "ct", "de", "dc", "fl", "ga", "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh",
      "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy"]


def fetch(st, part):
    for year in (2022, 2021, 2020):
        p = OUT / f"{st}_od_{part}_JT00_{year}.csv.gz"
        if p.is_file() and p.stat().st_size > 0:
            return
        r = requests.get(f"https://lehd.ces.census.gov/data/lodes/LODES8/{st}/od/{st}_od_{part}_JT00_{year}.csv.gz", timeout=600, stream=True)
        if r.status_code != 200:
            continue
        tmp = p.with_suffix(".part")
        with open(tmp, "wb") as f:
            for ch in r.iter_content(1 << 20):
                f.write(ch)
        tmp.rename(p); print(st, part, year, p.stat().st_size // 1_000_000, "MB", flush=True); return
    print(st, part, "NOT FOUND", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda a: fetch(*a), [(s, p) for s in ST for p in ("main", "aux")]))
    print("done", flush=True)
