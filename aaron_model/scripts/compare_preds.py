#!/usr/bin/env python3
"""
Compare re-computed prediction files with the stored ones (integrity check after the 2026-11-07 Metal error).

  python scripts/compare_preds.py --new DIR --old results/preds_tta --pattern '*_tent.csv' [--strip verify_] [--replace]

A file pair is flagged when any p_* value differs by more than --tol. With --replace, flagged stored files (and their
.json sidecar, if the new one exists) are moved to results/corrupt_mps_20261107/ and replaced by the new copies.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--new", required=True)
ap.add_argument("--old", default="results/preds_tta")
ap.add_argument("--pattern", required=True)
ap.add_argument("--strip", default="", help="prefix of the new file names to strip, e.g. verify_")
ap.add_argument("--tol", type=float, default=1e-5)
ap.add_argument("--replace", action="store_true")
a = ap.parse_args()

new_dir, old_dir = ROOT / a.new, ROOT / a.old
bad, n = [], 0
for f in sorted(new_dir.glob(a.strip + a.pattern)):
    old = old_dir / f.name[len(a.strip):]
    if not old.exists():
        print(f"{old.name}: no stored file")
        continue
    A, B = pd.read_csv(f), pd.read_csv(old)
    n += 1
    if len(A) != len(B) or not (A.row.values == B.row.values).all():
        print(f"{old.name}: ROW MISMATCH")
        bad.append((f, old))
        continue
    diffs = {c: int((np.abs(A[c].values - B[c].values) > a.tol).sum()) for c in A.columns if c.startswith("p_")}
    if any(diffs.values()):
        print(f"{old.name}: CORRUPT {({k: v for k, v in diffs.items() if v})}")
        bad.append((f, old))
print(f"compared {n} files, {len(bad)} differ")
if a.replace and bad:
    keep = ROOT / "results" / "corrupt_mps_20261107"
    keep.mkdir(exist_ok=True)
    for f, old in bad:
        for nf, of in [(f, old), (f.with_suffix(".json"), old.with_suffix(".json"))]:
            if nf.exists() and of.exists():
                shutil.move(of, keep / of.name)
                shutil.copy(nf, of)
                print("replaced", of.relative_to(ROOT))
