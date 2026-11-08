#!/usr/bin/env python3
"""
Exit non-zero unless every expected output of a phase exists and is complete (CPU only).

  python scripts/check_complete.py --tags all baseline ... --seeds 0 1 2 3 4 [--extra-tags all --extra tent rand]

Per fold x tag x seed: results/{run}.json (training), results/preds_tta/{run}.json + {run}.csv (TTA/AdaBN) and
{run}_cf.csv (cross-fitted AdaBN). With --extra-tags: {run}_tent.csv/.json and/or {run}_rand.csv for those tags.
Each CSV must have as many rows as the run's target set (results/preds/{run}.csv, set test_out_all) and finite p_*.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FOLDS = ["CS2", "S360", "XR"]

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--seeds", nargs="+", type=int, required=True)
ap.add_argument("--extra-tags", nargs="*", default=[])
ap.add_argument("--extra", nargs="*", default=[], choices=["tent", "rand"])
a = ap.parse_args()

R, T = ROOT / "results", ROOT / "results" / "preds_tta"
problems, n_ok = [], 0


def check_csv(path, n_rows):
    if not path.exists():
        return "missing"
    try:
        d = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        return f"unreadable ({type(e).__name__})"
    if len(d) != n_rows:
        return f"{len(d)} rows, expected {n_rows}"
    p = d[[c for c in d.columns if c.startswith("p_")]].to_numpy(float)
    if p.size == 0 or not np.isfinite(p).all():
        return "non-finite or no p_* columns"
    return None


for f in FOLDS:
    for t in a.tags:
        for s in a.seeds:
            run = f"{f}_{t}_seed{s}"
            if not (R / f"{run}.json").exists():
                problems.append(f"{run}: training result missing")
                continue
            preds = R / "preds" / f"{run}.csv"
            n = int((pd.read_csv(preds)["set"] == "test_out_all").sum()) if preds.exists() else -1
            if n < 0:
                problems.append(f"{run}: results/preds CSV missing")
                continue
            need = [(T / f"{run}.csv", True), (T / f"{run}.json", False), (T / f"{run}_cf.csv", True)]
            if t in a.extra_tags:
                if "tent" in a.extra:
                    need += [(T / f"{run}_tent.csv", True), (T / f"{run}_tent.json", False)]
                if "rand" in a.extra:
                    need += [(T / f"{run}_rand.csv", True)]
            for path, is_csv in need:
                err = check_csv(path, n) if is_csv else (None if path.exists() else "missing")
                if err:
                    problems.append(f"{path.relative_to(ROOT)}: {err}")
                else:
                    n_ok += 1

print(f"check_complete: {n_ok} outputs ok, {len(problems)} problems")
for p in problems[:50]:
    print("  ", p)
sys.exit(1 if problems else 0)
