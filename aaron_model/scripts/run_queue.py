#!/usr/bin/env python3
"""
Run a list of train_patch.py jobs with N concurrent workers, skipping finished ones.

Job file: one job per line, "TAG FOLD SEED [extra train_patch.py args...]"; '#' starts a comment.
A job is finished when results/{FOLD}_{TAG}_seed{SEED}.json exists. Logs: logs/{FOLD}_{TAG}_seed{SEED}.log.

Usage:  python scripts/run_queue.py jobs.txt [--workers 2]
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")


def parse(path):
    jobs = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        tag, fold, seed, *extra = line.split()
        jobs.append((tag, fold, int(seed), extra))
    return jobs


def run(job):
    tag, fold, seed, extra = job
    name = f"{fold}_{tag}_seed{seed}"
    if (ROOT / "results" / f"{name}.json").exists():
        return name, "skip", 0.0
    cmd = ["caffeinate", "-i", PY, "scripts/train_patch.py", "--fold", fold, "--seed", str(seed), "--tag", tag, *extra]
    t0 = time.time()
    with open(ROOT / "logs" / f"{name}.log", "w") as log:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] {name} exit {rc} ({dt:.0f}s)", flush=True)
    return name, rc, dt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    jobs = parse(a.jobs)
    print(f"[{time.strftime('%H:%M:%S')}] {len(jobs)} jobs, {a.workers} workers", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, jobs))
    bad = [r for r in res if r[1] not in (0, "skip")]
    print(f"[{time.strftime('%H:%M:%S')}] QUEUE_DONE  failed={len(bad)} {bad}", flush=True)
    sys.exit(1 if bad else 0)
