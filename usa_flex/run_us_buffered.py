"""All states of the stop-buffer flexible-scoring map (run_state_buffered.py), memory-aware: one job per state under the shared memory scheduler
(TransitLOSStudies/memsched.py: each state in its own memory-capped scope, started only when its estimate fits the free RAM, retried with more memory
if it outgrows its cap), then web_buf/index.json.

    python run_us_buffered.py [VT NY ...] [--force]
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import memsched  # noqa: E402

memsched.PEAKS = HERE / "peaks_usa_flex_buf.json"
sys.path.insert(0, str(HERE))
import run_county as rc  # noqa: E402


def blocks_gb(name):
    return sum(f.stat().st_size for f in (rc.STUDY / "data" / name / "counties").glob("*/block.geoparquet")) / 1e9


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = {a: n for a, n in rc.STATE_DIRS.items() if (rc.STUDY / "data" / n / "counties").is_dir()}
    jobs = []
    for a in [a for a in (args or sorted(names)) if a in names]:
        sf = HERE / "data_buf" / names[a] / "summary.json"
        lg = HERE / "logs" / "buf" / f"{a}.log"
        nsi_failed = lg.is_file() and "NSI floor areas not available" in lg.read_text(errors="ignore")
        if sf.is_file() and "--force" not in sys.argv and json.loads(sf.read_text()).get("code_version", 1) >= 8 \
                and json.loads(sf.read_text()).get("nsi_ok", not nsi_failed):
            continue
        est = 8.0 + 60.0 * blocks_gb(names[a])     # GB; roads to 15 km: about 1.5x the walk-only peaks
        jobs.append(memsched.Job(a, [sys.executable, "-u", "run_state_buffered.py", a, "--one"], memsched.estimate_from_history(a, est),
                                 env={"TQDM_DISABLE": "1", "PYTHONWARNINGS": "ignore", "OMP_NUM_THREADS": "8"}))
    jobs.sort(key=lambda j: -j.estimate_gb)
    print("states:", [(j.name, round(j.estimate_gb)) for j in jobs], flush=True)
    failed = memsched.run_jobs(jobs, cwd=HERE, log_dir=str(HERE / "logs" / "buf"), reserve_gb=16.0, max_parallel=4)
    print("finished; failed:", failed or "none", flush=True)
    subprocess.run([sys.executable, "-u", "run_state_buffered.py", "--index-only"], cwd=HERE, check=False)


if __name__ == "__main__":
    main()
