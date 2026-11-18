"""Memory-aware parallel job runner (used by the city-science, US and Spain runs).

A job is `(name, command, estimate_gb)`. The scheduler reads the machine's *available* memory (MemAvailable, so reclaimable cache counts as free) before
starting each job and only starts one when its estimate (times a safety factor) fits what is left after a reserve and after the estimates of the jobs
already running, so the estimates of running jobs are replaced by their measured peak as soon as it is higher. Every job runs in a
`systemd-run --user --scope -p MemoryMax=<cap>` scope, so a job that outgrows its cap is killed alone. The measured peak (cgroup `memory.peak`) of every
finished job is appended to `peaks.json`, and `estimate_from_history()` uses it for the next run.

Library use:  run_jobs([Job("boston", ["python", "boston/run.py"], 30.0), ...], cwd=..., log_dir=...)
Command line: python memsched.py jobs.tsv     (one job per line: name <TAB> estimate_gb <TAB> shell command)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

SAFETY = 1.25          # estimate multiplier before a job may start
CAP_FACTOR = 1.5       # MemoryMax of a job = estimate * this (never below the estimate * SAFETY)
MAX_RETRIES = 3        # memory retries of one job; each retry raises the estimate
DIAL_ABOVE = 0.6       # a job whose estimate (with margin) exceeds this share of the RAM is split into finer chunks instead of getting more memory
DIAL_SHRINK = 0.5      # a finer chunk step is assumed to halve the memory
RETRY_FACTOR = 1.6     # new estimate after an out-of-memory kill = max(peak, cap) * this
RESERVE_GB = 8.0       # kept free for the OS and the page cache working set
PEAKS = Path(__file__).resolve().parent / "peaks.json"


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1048576
    return 16.0


def mem_total_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal"):
                return int(line.split()[1]) / 1048576
    return 64.0


def _cgroup_peak_gb(pid: int) -> float:
    """Current anonymous memory of the job's cgroup (GB): `memory.peak` / `memory.current` also count the page cache, which the kernel reclaims under pressure, so they overstate what a job needs."""
    try:
        rel = next(l.split("::", 1)[1].strip() for l in open(f"/proc/{pid}/cgroup") if "::" in l)
        stat = {k: int(v) for k, v in (ln.split() for ln in Path(f"/sys/fs/cgroup{rel}/memory.stat").read_text().splitlines())}
        return (stat.get("anon", 0) + stat.get("shmem", 0) + stat.get("kernel", 0)) / 1073741824
    except Exception:  # noqa: BLE001 - the peak is a nice-to-have
        return 0.0


def _oom_kills(pid: int) -> int:
    """oom_kill counter of the job's cgroup (memory.events) while it is alive; 0 when unreadable."""
    try:
        rel = next(l.split("::", 1)[1].strip() for l in open(f"/proc/{pid}/cgroup") if "::" in l)
        for line in Path(f"/sys/fs/cgroup{rel}/memory.events").read_text().splitlines():
            if line.startswith("oom_kill"):
                return int(line.split()[1])
    except Exception:  # noqa: BLE001
        pass
    return 0


OOM_TEXT = ("MemoryError", "Cannot allocate memory", "std::bad_alloc", "out of memory", "Out of memory", "OOM", "Killed")


def failed_for_memory(job: "Job", rc: int, log_path: Path) -> bool:
    """True when the job died because it ran out of its memory cap (cgroup oom-kill, SIGKILL near the cap, or a memory error in its own log)."""
    if job.oom_kills > 0:
        return True
    if job.unit:   # systemd logs "A process of this unit has been killed by the OOM killer" for the scope even when the job was too short to be polled
        try:
            out = subprocess.run(["journalctl", "--user", "-u", f"{job.unit}.scope", "--no-pager", "-n", "50"], capture_output=True, text=True, timeout=30).stdout
            if "OOM killer" in out or "oom-kill" in out:
                return True
        except Exception:  # noqa: BLE001
            pass
    tail = ""
    try:
        tail = "".join(log_path.read_text(errors="replace").splitlines(True)[-60:])
    except Exception:  # noqa: BLE001
        pass
    if any(t in tail for t in OOM_TEXT):
        return True
    return rc in (-9, 137) and job.peak_gb >= 0.85 * job.cap_gb


def history() -> Dict[str, float]:
    try:
        return json.loads(PEAKS.read_text())
    except Exception:  # noqa: BLE001
        return {}


def record_peak(name: str, gb: float) -> None:
    h = history()
    h[name] = round(max(gb, 0.0), 2)
    PEAKS.write_text(json.dumps(h, indent=1, sort_keys=True))


def estimate_from_history(name: str, fallback: float) -> float:
    return history().get(name, fallback)


@dataclass
class Job:
    name: str
    cmd: List[str]
    estimate_gb: float
    after: Optional[Callable[[], None]] = None      # run in the scheduler after a success (e.g. a deploy)
    # filled by the scheduler
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    peak_gb: float = 0.0
    cap_gb: float = 0.0
    oom_kills: int = 0
    attempts: int = 0
    unit: str = ""
    dial: Optional[List[Dict[str, str]]] = None      # finer-chunk steps (environment variables) tried when more memory is not an option
    dial_i: int = 0
    env: Dict[str, str] = field(default_factory=dict)


def run_jobs(jobs: List[Job], cwd, log_dir, reserve_gb: float = RESERVE_GB, max_parallel: int = 8, poll_s: float = 3.0) -> List[str]:
    """Run `jobs` (biggest estimate first) as memory allows; returns the names of the failed ones.

    A job killed for lack of memory is queued again with a larger estimate (up to MAX_RETRIES times, never above the machine's RAM); any other failure is
    reported with the tail of its log and is not retried."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = sorted(jobs, key=lambda j: -j.estimate_gb)
    running: List[Job] = []
    failed: List[str] = []
    total = mem_total_gb()
    for j in queue:   # an estimate that takes most of the machine starts with finer chunks (smaller pieces, joined afterwards) instead of one huge pass
        while j.dial and j.dial_i < len(j.dial) and j.estimate_gb * SAFETY > total * DIAL_ABOVE:
            j.env.update(j.dial[j.dial_i]); j.dial_i += 1; j.estimate_gb *= DIAL_SHRINK
            print(f"[dial] {j.name}: estimate too high for this machine, processing in finer chunks {j.env} (estimate now {j.estimate_gb:.0f}G)", flush=True)
    while queue or running:
        for j in running:
            if j.proc and j.proc.pid:
                j.peak_gb = max(j.peak_gb, _cgroup_peak_gb(j.proc.pid))
                j.oom_kills = max(j.oom_kills, _oom_kills(j.proc.pid))
        headroom = mem_available_gb() - reserve_gb
        for j in running:
            headroom -= max(0.0, max(j.estimate_gb, j.peak_gb) - j.peak_gb)
        queue.sort(key=lambda j: -j.estimate_gb)
        for j in list(queue):
            need = j.estimate_gb * SAFETY
            if len(running) >= max_parallel:
                break
            if need <= headroom or (not running and need <= total * 0.95):
                cap = min(max(j.estimate_gb * CAP_FACTOR, need, 1.0), total * 0.95)   # >= 1G: a 0G MemoryMax is refused by systemd
                j.cap_gb, j.peak_gb, j.oom_kills = cap, 0.0, 0
                j.attempts += 1
                j.unit = f"memsched-{j.name}-{j.attempts}-{int(time.time())}"
                wrapped = ["systemd-run", "--user", "--scope", f"--unit={j.unit}", "-p", f"MemoryMax={cap:.0f}G", "-p", "MemorySwapMax=2G", "--"] + j.cmd
                j.proc = subprocess.Popen(wrapped, cwd=cwd, stdout=open(log_dir / f"{j.name}.log", "w"), stderr=subprocess.STDOUT, env={**os.environ, **j.env})
                running.append(j)
                queue.remove(j)
                headroom -= need
                print(f"[start] {j.name} try {j.attempts} estimate {j.estimate_gb:.0f}G cap {cap:.0f}G, free after {headroom:.0f}G, {time.strftime('%H:%M')}", flush=True)
            elif need > total * 0.95:
                queue.remove(j)
                failed.append(j.name)
                print(f"[FAIL] {j.name}: needs about {need:.0f}G, more than this machine's RAM ({total:.0f}G) - split the job (e.g. isochrones per municipality)", flush=True)
        time.sleep(poll_s)
        for j in list(running):
            rc = j.proc.poll()
            if rc is None:
                continue
            running.remove(j)
            log_path = log_dir / f"{j.name}.log"
            if rc == 0:
                if j.peak_gb:
                    record_peak(j.name, j.peak_gb)
                print(f"[done] {j.name} peak {j.peak_gb:.1f}G of cap {j.cap_gb:.0f}G {time.strftime('%H:%M')}", flush=True)
                if j.after:   # in a thread: a slow deploy must not stop the next jobs from starting
                    threading.Thread(target=j.after, daemon=False).start()
            elif failed_for_memory(j, rc, log_path):
                if j.attempts > MAX_RETRIES:
                    failed.append(j.name)
                    print(f"[FAIL] {j.name}: out of memory again at cap {j.cap_gb:.0f}G after {j.attempts} tries", flush=True)
                else:
                    grown = max(j.peak_gb, j.cap_gb) * RETRY_FACTOR
                    if j.dial and j.dial_i < len(j.dial) and grown * SAFETY > total * DIAL_ABOVE:
                        j.env.update(j.dial[j.dial_i]); j.dial_i += 1
                        j.estimate_gb = max(j.cap_gb, j.peak_gb) * DIAL_SHRINK * RETRY_FACTOR
                        print(f"[RAM] {j.name} ran out of memory at cap {j.cap_gb:.0f}G: finer chunks {j.env}, estimate {j.estimate_gb:.0f}G", flush=True)
                    else:
                        j.estimate_gb = grown
                        print(f"[RAM] {j.name} ran out of memory at cap {j.cap_gb:.0f}G (peak {j.peak_gb:.1f}G): retrying with estimate {j.estimate_gb:.0f}G", flush=True)
                    queue.append(j)
            else:
                failed.append(j.name)
                tail = "".join(log_path.read_text(errors="replace").splitlines(True)[-12:]) if log_path.exists() else ""
                print(f"[FAIL] {j.name} rc={rc}, NOT a memory failure (peak {j.peak_gb:.1f}G of cap {j.cap_gb:.0f}G). Last log lines:\n{tail}", flush=True)
    return failed


def main() -> int:
    jobs = []
    for line in open(sys.argv[1]):
        if line.strip() and not line.startswith("#"):
            name, est, cmd = line.rstrip("\n").split("\t", 2)
            jobs.append(Job(name, ["bash", "-c", cmd], float(est)))
    failed = run_jobs(jobs, cwd=Path.cwd(), log_dir=Path.cwd() / "logs_memsched")
    print("failed:", failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
