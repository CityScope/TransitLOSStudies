"""Run the flexible scoring for the whole US: one job per state, memory-aware (the original study's scheduler), logs under logs/.

    python run_us.py                 # every state with cached county networks
    python run_us.py VT DE RI        # some states
    python run_us.py --dry-run

Nothing in usa_study is written (peaks go to peaks_usa_flex.json, not to the study's peaks.json).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / "usa_study"
sys.path.insert(0, str(HERE.parent))
import memsched  # noqa: E402

memsched.PEAKS = HERE / "peaks_usa_flex.json"
sys.path.insert(0, str(HERE))
import run_county as rc  # noqa: E402


def gtfs_gb(state_dir: Path) -> float:
    g = state_dir / "gtfs"
    return sum(f.stat().st_size for f in g.rglob("*") if f.is_file()) / 1e9 if g.is_dir() else 0.0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = {a: n for a, n in rc.STATE_DIRS.items() if (STUDY / "data" / n / "counties").is_dir()}
    states = [a for a in (args or sorted(names)) if a in names]
    jobs = []
    for a in states:
        if (HERE / "data" / names[a] / "summary.json").is_file() and "--force" not in sys.argv:
            continue
        est = 8.0 + 14.0 * gtfs_gb(STUDY / "data" / names[a])    # GB: the GTFS feeds of the state loaded at once dominate
        jobs.append(memsched.Job(a, [sys.executable, "-u", "run_us_state.py", a] + (["--force"] if "--force" in sys.argv else []), memsched.estimate_from_history(a, est), env={"TQDM_DISABLE": "1", "PYTHONWARNINGS": "ignore"}))
    jobs.sort(key=lambda j: -j.estimate_gb)     # the big states first, small ones fill the gaps
    print("states:", [(j.name, round(j.estimate_gb)) for j in jobs], flush=True)
    if "--dry-run" in sys.argv:
        return
    failed = memsched.run_jobs(jobs, cwd=HERE, log_dir=str(HERE / "logs"), reserve_gb=10.0, max_parallel=4)
    print("finished; failed:", failed or "none", flush=True)


if __name__ == "__main__":
    main()
