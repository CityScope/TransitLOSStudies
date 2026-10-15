"""Final results of a complete run: PTLOS by state / USA (`ptlos_by_state.csv`, `ptlos_usa.json`) and the run manifest
(`run_manifest.json`: every parameter and code version needed to reproduce the numbers).

PTLOS = population-weighted mean level of service (0-100) over the H3 res-7 cells of each state
(res 7 carries the full population of the state; cells without service count as 0).
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

REPOS = ("pyCensus", "pyGTFSHandler", "transitLOS", "UrbanAccessAnalyzer", "geohierarchy", "TransitLOSStudies")


def write_ptlos_summary(data_dir: Path) -> dict:
    rows = []
    for state_dir in sorted(p for p in data_dir.iterdir() if (p / "h3_res7.geoparquet").is_file()):
        g = pd.read_parquet(state_dir / "h3_res7.geoparquet", columns=["population", "level_of_service"])
        pop, los = g["population"].fillna(0), g["level_of_service"].fillna(0)
        if pop.sum() <= 0:
            continue
        stops = state_dir / "stops.geoparquet"
        rows.append({
            "state": state_dir.name, "pop": pop.sum(), "ptlos": (pop * los).sum() / pop.sum(),
            "served": pop[los > 0].sum() / pop.sum(),
            "stops": __import__("pyarrow.parquet").parquet.ParquetFile(stops).metadata.num_rows if stops.is_file() else 0,
            "counties": len(list((state_dir / "counties").glob("*"))) if (state_dir / "counties").is_dir() else 0,
        })
    table = pd.DataFrame(rows)
    table.to_csv(data_dir / "ptlos_by_state.csv", index=False)
    usa = {
        "ptlos": float((table["pop"] * table.ptlos).sum() / table["pop"].sum()),
        "served_share": float((table["pop"] * table.served).sum() / table["pop"].sum()),
        "population": float(table["pop"].sum()), "states": len(table), "stops": int(table.stops.sum()),
        "counties": int(table.counties.sum()),
    }
    (data_dir / "ptlos_usa.json").write_text(json.dumps(usa, indent=1))
    return usa


def _git_head(repo: Path) -> str:
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True).stdout
        return head + (" +uncommitted" if dirty.strip() else "")
    except Exception:
        return "unknown"


def write_run_manifest(data_dir: Path, root: Path, params: dict) -> Path:
    """Parameters + code versions + a fingerprint of every GTFS file used (so a rerun can prove it used the same feeds)."""
    feeds = {}
    for gtfs in sorted(data_dir.glob("*/gtfs")):
        for folder in sorted(p for p in gtfs.iterdir() if p.is_dir() and not p.name.startswith("_")):
            h = hashlib.sha256()
            for f in sorted(folder.glob("*.txt")):
                h.update(f.name.encode())
                h.update(hashlib.sha256(f.read_bytes()).digest())
            feeds[f"{gtfs.parent.name}/{folder.name}"] = h.hexdigest()[:16]
    manifest = {
        "written": time.strftime("%Y-%m-%d %H:%M:%S"), "parameters": params,
        "code_versions": {r: _git_head(root / r) for r in REPOS},
        "gtfs_feeds": feeds,
    }
    path = data_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path
