"""Run several city-science cities in parallel, sized by the machine's free memory.

Each city is one `<city>/run.py` process (see `../memsched.py`): before a city starts, the scheduler reads the *available* memory, estimates what the city
needs from its processing area (km^2 of `aoi.gpkg`, the union of the four areas) and its number of GTFS stops -- or, once the city has run before, from its
measured peak (`../peaks.json`) -- and starts it only when that fits. Each city runs in a memory-capped systemd scope, so one that overshoots dies alone.

    python run_all.py                       # every city, refresh (reuse cached isochrones, rebuild map + tiles)
    python run_all.py --full boston hamburg   # full rebuild: deletes streets/results/map_tiles first and recomputes the isochrones
    python run_all.py --dry-run               # print the estimates and the order
    python run_all.py --deploy-cmd "<command> {city}"   # run after each successful city ({city} is replaced)

Estimate (before any measured peak exists): 4 GB + 0.0015 GB per km^2 + 0.00002 GB per stop. If a city runs out of memory its cap is raised (x1.6 of the cap
it hit) and it is queued again by itself, up to 3 times; once more memory would take over 60% of the RAM (or the first estimate already does) it is
run in finer chunks instead -- H3 res 4, 5, 6 pieces of about 1,770 / 250 / 36 km2 (county / municipality sized) with a halo buffer, results joined --
through the `STUDY_CHUNK_H3_RES` / `STUDY_CHUNK_BUFFER_M` environment variables read by `studytools.params.apply_memory_dial`; any other failure is reported with the tail of its log. The first successful run records the real
peak (`../peaks.json`) and later runs start from it.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memsched import Job, estimate_from_history, mem_available_gb, run_jobs  # noqa: E402

HERE = Path(__file__).resolve().parent
CITIES = ["andorra", "beerseba", "boston", "concepcion", "gipuzkoa", "guadalajara", "hamburg", "san_francisco", "shanghai", "taipei", "toronto"]


def area_km2(city: str) -> float:
    import geopandas as gpd

    g = gpd.read_file(HERE / city / "aoi.gpkg").to_crs(6933)
    return float(g.geometry.union_all().area / 1e6)


def n_stops(city: str) -> int:
    n = 0
    for f in (HERE / city / "gtfs").glob("*/stops.txt"):
        with open(f, "rb") as fh:
            n += max(sum(1 for _ in fh) - 1, 0)
    return n


def estimate_gb(city: str, full: bool) -> float:
    formula = 4.0 + 0.0015 * area_km2(city) + 0.00002 * n_stops(city)
    est = estimate_from_history(city, formula)
    return est if est <= formula * 4 else formula


def prepare(city: str, full: bool) -> None:
    if full:
        for sub in ("streets", "results", "map_tiles"):
            shutil.rmtree(HERE / city / sub, ignore_errors=True)
        (HERE / city / "logs").mkdir(exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cities", nargs="*", help="default: all")
    ap.add_argument("--full", action="store_true", help="delete streets/results/map_tiles and recompute isochrones")
    ap.add_argument("--deploy-cmd", default=None, help="shell command run after each successful city; {city} is replaced")
    ap.add_argument("--reserve-gb", type=float, default=8.0)
    ap.add_argument("--max-parallel", type=int, default=6)
    ap.add_argument("--log-dir", default=str(HERE / "logs_run_all"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cities = a.cities or CITIES
    est = {c: estimate_gb(c, a.full) for c in cities}
    print(f"available now: {mem_available_gb():.0f} GB; estimates (GB): " + ", ".join(f"{c} {est[c]:.0f}" for c in sorted(cities, key=lambda c: -est[c])), flush=True)
    if a.dry_run:
        return 0
    jobs = []
    for c in cities:
        prepare(c, a.full)
        cmd = ["python", "-u", f"{c}/run.py", "--rebuild-tiles", "--use-pmtiles"] + (["--no-reuse-cached-los"] if a.full else [])
        after = None
        if a.deploy_cmd:
            def after(c=c):
                r = subprocess.run(a.deploy_cmd.format(city=c), shell=True, cwd=HERE)
                print(f"[{'live' if r.returncode == 0 else 'DEPLOY FAIL'}] {c}", flush=True)
        jobs.append(Job(c, cmd, est[c], after=after, dial=[{"STUDY_CHUNK_H3_RES": "4", "STUDY_CHUNK_BUFFER_M": "2000"}, {"STUDY_CHUNK_H3_RES": "5", "STUDY_CHUNK_BUFFER_M": "1500"}, {"STUDY_CHUNK_H3_RES": "6", "STUDY_CHUNK_BUFFER_M": "1000"}]))
    failed = run_jobs(jobs, cwd=HERE, log_dir=a.log_dir, reserve_gb=a.reserve_gb, max_parallel=a.max_parallel)
    print(f"finished; failed: {failed or 'none'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
