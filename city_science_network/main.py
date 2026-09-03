"""Run every city's transit-LOS study, then build cross-city comparison figures.

Defines `GLOBAL_PARAMS` (re-exported from `code.params`) as the single
`StudyParams` instance every city script uses, guaranteeing identical
parameters across cities. Explicitly excludes `boston_city_evolution`,
which is its own separate study (see `boston_city_evolution/main.py`).

HOW TO RUN THIS FOR REAL (the whole multi-city roster, unattended):

    uv run python main.py

That's it -- just run it directly. It is safe to run repeatedly: each city
is skipped automatically if its `map.html` is already newer than its
`aoi.gpkg`/`gtfs/` inputs (see `_city_is_up_to_date`), so re-running after
an interruption, a crash, or a per-city failure naturally picks up wherever
it left off -- completed cities are skipped, failed/incomplete ones are
retried. This isn't theoretical: this exact pattern has happened for real
(Beersheba failed once mid-session, the underlying bug was fixed, and the
next plain `uv run python main.py` re-ran only Beersheba and picked the
rest of the roster back up from "already up to date"). Pass `--force` (or
set `FORCE_REBUILD=1`) to reprocess every city regardless of that check,
e.g. after a shared-code fix that doesn't show up in any city's AOI/GTFS
mtimes. Pass `--renderer folium` to use the slower, full-featured Folium
renderer (scenario editor, live recolor, stats panel) instead of the
default `maplibre` renderer (fast, base-layer-only, pmtiles-based) -- see
`--renderer`'s own `--help` text below for the full tradeoff.

SELF-WRAPPING MEMORY SAFETY: on start, this script re-execs itself under a
`systemd-run --user --scope` with a conservative `MemoryMax`/`MemorySwapMax`
(see `_ensure_memory_safe_wrapper` below) unless it's already running
inside such a scope, or `systemd-run` isn't available. This exists because
this exact host has been OOM-killed running this pipeline unwrapped --
once catastrophically (took the whole machine down) and once merely
killing the pipeline's own process group after a since-fixed per-stage
memory bug -- both while a human had to remember to hand-wrap the
invocation with the right `systemd-run` flags each time. Wrapping it in
the script itself means a plain `uv run python main.py` is safe to just
run and walk away from; set `CS_TRANSITLOS_NO_WRAP=1` to opt out (e.g. if
you're already managing your own cgroup/ulimit) and get the old raw
in-process behavior.

Each city runs in its own `subprocess` (see `run_all_cities`'s own
docstring for why: peak per-city memory, released back to the OS between
cities). The largest few cities (Hamburg, Shanghai, whole-country/large
OSM extracts) additionally cap their own internal `tile_workers` down from
the pipeline's default of 4 to 1-2 in their own `run.py` -- a per-city
tuning already in place from real OOM investigation this session (see
each city's own `run.py` for its reasoning) and deliberately left as a
per-city decision rather than a single global override, since smaller
cities (e.g. Andorra) safely use more parallelism at the default.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from code.city_config import CITY_CONFIGS
from code.combined_map import build_combined_map

# Conservative systemd-run memory limits for `_ensure_memory_safe_wrapper`,
# below. Matches the values manually used to wrap this pipeline's real runs
# this session on this ~30GB host (leaves ~10GB headroom for the OS/other
# processes, plus a small 1GB swap allowance rather than none/unlimited --
# unlimited swap is what let a prior unwrapped run degrade the whole
# machine to a crawl instead of being cleanly OOM-killed).
DEFAULT_MEMORY_MAX = "20G"
DEFAULT_MEMORY_SWAP_MAX = "1G"


def _ensure_memory_safe_wrapper() -> None:
    """Re-exec this process under `systemd-run --user --scope` with a conservative
    memory cap, unless already running inside one (or explicitly opted out).

    This is a *self-wrap*: it re-execs the exact same command
    (`sys.executable main.py <original argv>`) inside a memory-limited
    systemd scope, then returns control to that re-exec'd process (the
    original process image is replaced via `os.execvpe`, not forked -- no
    duplicate process is left behind). Guarded by the
    `CS_TRANSITLOS_MEMORY_WRAPPED` env var so the re-exec'd child doesn't
    try to wrap itself again (infinite loop).

    Opt out with `CS_TRANSITLOS_NO_WRAP=1` (e.g. you're already managing
    your own cgroup/ulimit, or `systemd-run` genuinely isn't appropriate
    for your environment). Falls back to running unwrapped (with a loud
    warning) if `systemd-run` isn't on PATH at all -- e.g. non-systemd
    Linux, macOS, containers -- rather than hard-failing.
    """
    if os.environ.get("CS_TRANSITLOS_MEMORY_WRAPPED") == "1":
        return
    if os.environ.get("CS_TRANSITLOS_NO_WRAP") == "1":
        print("[main] CS_TRANSITLOS_NO_WRAP=1 -- running WITHOUT the systemd-run memory-limit wrapper", flush=True)
        return
    systemd_run = shutil.which("systemd-run")
    if systemd_run is None:
        print(
            "[main] WARNING: systemd-run not found on PATH -- running WITHOUT a memory-limit "
            "safety net. This host has previously been OOM-killed (once machine-wide) running "
            "this pipeline unwrapped; consider wrapping manually or running on a system with "
            "systemd available. Set CS_TRANSITLOS_NO_WRAP=1 to silence this warning.",
            flush=True,
        )
        return

    env = os.environ.copy()
    env["CS_TRANSITLOS_MEMORY_WRAPPED"] = "1"
    cmd = [
        systemd_run,
        "--user",
        "--scope",
        "-p",
        f"MemoryMax={DEFAULT_MEMORY_MAX}",
        "-p",
        f"MemorySwapMax={DEFAULT_MEMORY_SWAP_MAX}",
        "--",
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    print(f"[main] re-exec'ing under a memory-limited systemd-run scope: {' '.join(cmd)}", flush=True)
    os.execvpe(cmd[0], cmd, env)

ROOT = Path(__file__).resolve().parent
# Full roster of prepared cities -- this is the durable, reusable list for
# reproducing/maintaining every city's study going forward. Each city is
# skipped automatically if already up to date (see `_city_is_up_to_date`),
# so it's safe to always list every city here and just re-run `main.py`.
CITY_KEYS = (
    "andorra",
    "beerseba",
    "concepcion",
    "boston",
    "gipuzkoa",
    "guadalajara",
    "hamburg",
    "san_francisco",
    "shanghai",
    "taipei",
    "toronto",
)


def _newest_mtime_under(path: Path) -> float | None:
    """Return the newest mtime among all files under `path` (recursive), or None if empty/missing."""
    if not path.exists():
        return None
    newest: float | None = None
    if path.is_file():
        return path.stat().st_mtime
    for p in path.rglob("*"):
        if p.is_file():
            m = p.stat().st_mtime
            if newest is None or m > newest:
                newest = m
    return newest


def _city_is_up_to_date(key: str) -> bool:
    """Whether `<key>/map.html` already exists and is newer than the city's AOI + GTFS inputs.

    A simple mtime-based staleness check: if `aoi.gpkg` or anything under
    `gtfs/` is newer than `map.html`, the city's inputs changed since the
    last successful build and it's treated as stale (reprocess). Otherwise
    the existing `map.html` is considered current and the city is skipped.
    This is deliberately conservative/simple -- it does not track every
    intermediate artifact or code change (see `--force` for that case).
    """
    city_dir = ROOT / key
    map_html = city_dir / "map.html"
    if not map_html.is_file():
        return False
    map_mtime = map_html.stat().st_mtime

    aoi_mtime = _newest_mtime_under(city_dir / "aoi.gpkg")
    gtfs_mtime = _newest_mtime_under(city_dir / "gtfs")

    for input_mtime in (aoi_mtime, gtfs_mtime):
        if input_mtime is not None and input_mtime > map_mtime:
            return False
    return True


def _city_needs_census_refresh(key: str) -> bool:
    """Whether `key` is otherwise up to date but its census module became newly available.

    `CityConfig.uses_census` (`code/city_config.py`) is a computed property:
    it tries importing the relevant `pyCensus` submodule and only returns
    True if that succeeds (see `_census_supported`). So a city added or last
    run before its country's census module was wired up (e.g. Hamburg,
    pending Germany GENESIS API credentials) has `uses_census=False` at that
    time and its `results/metro/h3_grid.parquet` + `map.html` carry no
    census columns -- but once the module is wired up, `uses_census` flips
    True automatically on the *next* import, with no code change to this
    city's own config or `run.py`.

    This detects exactly that situation from cached on-disk state: config
    says census-capable now, but no `results/census/` output exists yet
    (only ever written by the census-join branch of the pipeline -- see
    `run_city_study`/`refresh_census_only`). When both hold, the city is
    otherwise-current (mtime-wise) but should still get the fast
    `--census-only` re-entry point rather than being skipped outright.
    """
    city_dir = ROOT / key
    config = CITY_CONFIGS.get(key)
    if config is None or not config.uses_census:
        return False
    if not (city_dir / "results" / "metro" / "h3_grid.parquet").is_file():
        return False  # never run at all yet -- not a "refresh", a first run
    census_dir = city_dir / "results" / "census"
    has_census_output = census_dir.is_dir() and any(census_dir.glob("*.parquet"))
    return not has_census_output


def _rebuild_combined_map_or_warn() -> None:
    """Regenerate `combined_map.html` from whatever cities have `map.html` right now.

    Called after each city in `run_all_cities` (both a freshly-completed
    run and a skip-as-up-to-date), so the combined map incrementally picks
    up each city's results as they become available rather than only being
    built once at the very end -- important because this pipeline has
    genuinely been interrupted partway through more than once this session
    (OOM kills, manual stops). Failures here (e.g. a still-partial city
    with malformed `summary.json`) are logged as a warning and swallowed,
    matching the existing per-city `run.py` failure-handling pattern --
    a combined-map hiccup must never abort the rest of the city roster.
    """
    try:
        path = build_combined_map()
        print(f"[main] rebuilt combined map: {path}", flush=True)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        print(
            f"[main] WARNING: combined_map.build_combined_map() failed ({exc!r}) -- "
            "continuing with remaining cities",
            flush=True,
        )


def run_all_cities(
    force: bool = False, map_only: bool = False, census_only: bool = False, auto_census_refresh: bool = True
) -> None:
    """Run every city's `run.py` as a fresh subprocess, in order.

    Each city (especially those with large regional `.osm.pbf` street
    sources, e.g. Hamburg's whole-Germany extract) can peak at several GB
    of RAM while building its street network. Running each in its own
    process -- rather than importing and calling `run_city_study` in-process
    for all six -- guarantees that memory is fully released back to the OS
    between cities instead of accumulating across a single long-running
    process (observed directly: system memory climbed to 23GB/30Gi used
    with swap exhausted partway through a single-process run of just the
    first two cities).

    Args:
        force: Reprocess every city's full pipeline regardless of the
            up-to-date check.
        map_only: Forward `--map-only` to every city's `run.py` -- fast
            map/tiles-only rebuild from cached results, no city skipped by
            the up-to-date check (there's nothing expensive to skip).
        census_only: Forward `--census-only` to every city's `run.py` --
            fast census-refresh re-entry point (see `refresh_census_only`).
        auto_census_refresh: When neither `map_only` nor `census_only` nor
            `force` is set, a city that's otherwise up to date but whose
            census module just became newly available (see
            `_city_needs_census_refresh`) automatically gets `--census-only`
            applied instead of being skipped outright -- so census support
            landing for e.g. Hamburg is picked up on the very next `main.py`
            run, without anyone having to remember a manual flag.
    """
    succeeded: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    for key in CITY_KEYS:
        extra_args: list[str] = []
        if map_only:
            extra_args.append("--map-only")
        elif census_only:
            extra_args.append("--census-only")
        elif not force and _city_is_up_to_date(key):
            if auto_census_refresh and _city_needs_census_refresh(key):
                print(
                    f"[main] {key}: up to date, but census module is newly available -- "
                    "running the fast --census-only refresh instead of skipping",
                    flush=True,
                )
                extra_args.append("--census-only")
            else:
                print(
                    f"[main] skipping {key}: already up to date (use --force to reprocess)",
                    flush=True,
                )
                skipped.append(key)
                _rebuild_combined_map_or_warn()
                continue

        print(f"[main] running {key}/run.py as a subprocess" + (f" {' '.join(extra_args)}" if extra_args else ""), flush=True)
        # `-u`: unbuffered stdout/stderr in the child. Without this, output sits in the
        # child's internal buffer and is lost entirely if it's OOM-killed (SIGKILL gives no
        # chance to flush) -- silently hiding exactly which step it crashed on, which is
        # what happened investigating a real SF OOM crash.
        result = subprocess.run(
            [sys.executable, "-u", str(ROOT / key / "run.py"), *extra_args], cwd=ROOT, check=False, env=os.environ
        )
        if result.returncode != 0:
            print(
                f"[main] WARNING: {key}/run.py failed with exit code {result.returncode} -- "
                "continuing with remaining cities",
                flush=True,
            )
            failed.append(key)
        else:
            succeeded.append(key)
            # Regenerate the cross-city combined map right after this city finishes,
            # not just once at the very end -- so if the pipeline is interrupted
            # partway through (a real, repeated occurrence: OOM kills, manual stops),
            # combined_map.html still reflects however many cities finished so far
            # rather than being stuck at whatever it looked like before this run.
            _rebuild_combined_map_or_warn()

    print(
        f"[main] run_all_cities summary: {len(succeeded)} succeeded, {len(failed)} failed, "
        f"{len(skipped)} skipped",
        flush=True,
    )
    if succeeded:
        print(f"[main]   succeeded: {', '.join(succeeded)}", flush=True)
    if failed:
        print(f"[main]   failed: {', '.join(failed)}", flush=True)
    if skipped:
        print(f"[main]   skipped (already up to date): {', '.join(skipped)}", flush=True)


def build_cross_city_figures() -> None:
    """Build cross-city comparison figures (R² per city, %area-more-transit per city).

    Reads each city's `results/metro/h3_grid.parquet` (written by
    `run_city_study`); run only after every city has completed.
    """
    import geopandas as gpd

    from code.stats import linreg

    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    r2_by_city: dict[str, float] = {}
    transit_share_by_city: dict[str, float] = {}

    for key in CITY_KEYS:
        path = ROOT / key / "results" / "metro" / "h3_grid.parquet"
        if not path.exists():
            continue
        grid = gpd.read_parquet(path)
        fit = linreg(grid["pop_density"].to_numpy(), grid["level_of_service"].to_numpy(), weights=grid["population"].to_numpy())
        r2_by_city[CITY_CONFIGS[key].display_name] = fit.r2
        flags = grid["equity_flag"].value_counts(normalize=True, dropna=True)
        transit_share_by_city[CITY_CONFIGS[key].display_name] = float(flags.get("more_transit", 0.0))

    from code.figures import percent_area_bar

    if r2_by_city:
        percent_area_bar(r2_by_city, "R² (LOS vs population density) per city", str(ROOT / "figures" / "r2_per_city.jpg"))
    if transit_share_by_city:
        percent_area_bar(transit_share_by_city, "% area flagged 'more transit needed' per city", str(ROOT / "figures" / "more_transit_share_per_city.jpg"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        default=os.environ.get("FORCE_REBUILD", "").lower() in ("1", "true", "yes"),
        help=(
            "Reprocess every city regardless of the up-to-date check "
            "(also settable via the FORCE_REBUILD=1 env var). Use this "
            "after a shared-code fix that doesn't show up in AOI/GTFS "
            "mtimes."
        ),
    )
    parser.add_argument(
        "--renderer", choices=["maplibre", "folium"], default="maplibre",
        help=(
            "Map renderer for every city (default: maplibre -- fast, reads "
            "the pipeline's .pmtiles directly with no XYZ/pbf extraction, "
            "base-layer parity only). Pass 'folium' for the full scenario "
            "editor + stats panel (route drawing, live recolor, "
            "Distribution/Regression/ANOVA tabs), which internally still "
            "needs the slower XYZ-pbf-extracting Leaflet.VectorGrid path. "
            "Threaded to each city's run.py via the TRANSITLOS_RENDERER "
            "env var (see code/pipeline.py)."
        ),
    )
    parser.add_argument(
        "--map-only", "--tiles-only", dest="map_only", action="store_true",
        help=(
            "Forward --map-only to every city's run.py: fast map/tiles-only "
            "rebuild from cached results, skipping the full pipeline "
            "entirely (see code.pipeline.rebuild_map_only)."
        ),
    )
    parser.add_argument(
        "--census-only", "--refresh-census", dest="census_only", action="store_true",
        help=(
            "Forward --census-only to every city's run.py: re-run just the "
            "census join + stats/map rebuild from cached pipeline output "
            "(see code.pipeline.refresh_census_only). Also happens "
            "automatically, per city, when that city's census module just "
            "became newly available -- see --no-auto-census-refresh."
        ),
    )
    parser.add_argument(
        "--no-auto-census-refresh", dest="auto_census_refresh", action="store_false", default=True,
        help=(
            "Disable the automatic per-city --census-only fast path applied "
            "when a city is otherwise up to date but its census module just "
            "became newly available (see _city_needs_census_refresh). With "
            "this set, such a city is skipped like any other up-to-date city."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    _ensure_memory_safe_wrapper()  # may re-exec + never return, see docstring above
    args = _parse_args()
    os.environ["TRANSITLOS_RENDERER"] = args.renderer
    run_all_cities(
        force=args.force, map_only=args.map_only, census_only=args.census_only,
        auto_census_refresh=args.auto_census_refresh,
    )
    build_cross_city_figures()
