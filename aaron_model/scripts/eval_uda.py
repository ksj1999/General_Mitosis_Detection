#!/usr/bin/env python3
"""
Cross-fitted evaluation of the target-aligned DANN models (unsupervised domain
adaptation, Ganin et al. 2016; see ADDONS_PROTOCOL.md).

For each fold and seed there are two half-models:
  {base}_A  trained with the unlabelled target images of half A  -> predicts only half-B candidates
  {base}_B  trained with the unlabelled target images of half B  -> predicts only half-A candidates
Halves come from addon_common.target_halves over the target scanner's (test_out_all) rows,
so no candidate is predicted by a model that saw its slide during training.

Each half-model gets its own thresholds, chosen by F1 on its own source val predictions
(results/preds/{fold}_{base}_{h}_seed{s}.csv, set == "val") with its own source BatchNorm
statistics, plain and 8-view TTA (best_threshold). Target labels are never used.
No AdaBN for these models (they have already seen target images).

Usage:  python scripts/eval_uda.py [--base stain_uda] [--seeds 0 1 2] [--folds CS2 S360 XR] [--out-dir DIR]
Output: results/preds_tta/{fold}_{base}_seed{s}.csv
            row,image_id,y,p_plain,p_tta,thr_plain,thr_tta  (per-row thresholds of the model that predicted the row)
        results/preds_tta/{fold}_{base}_seed{s}.json
            per-half thresholds and details under "halves" only (no top-level thr_plain / thr_tta:
            no single threshold applies to every row, so a consumer that ignores the per-row CSV
            columns fails loudly instead of using an invented one),
            max_abs_diff_vs_stored = max over halves of |p_plain - stored p| on the rows each half predicts
Existing outputs are not recomputed. Both half-models must exist before any GPU work starts.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_patch import best_threshold  # noqa: E402
from eval_tta_adabn import OUT_DIR, hard_labels, load_run, predict, run_patches, val_test  # noqa: E402
from addon_common import FOLDS, get_device, load_model, target_halves  # noqa: E402

OTHER = {"A": "B", "B": "A"}


def eval_half(fold, base, half, seed, device, run):
    """Predictions of model {base}_{half} on the other half's target rows, with its own source-val thresholds.
    `run` is load_run(fold, f"{base}_{half}", seed), already checked to be present."""
    tag = f"{base}_{half}"
    cfg, stored = run
    if cfg.get("uda_half") != half:
        raise ValueError(f"{fold}_{tag}_seed{seed}: config uda_half is {cfg.get('uda_half')!r}, expected {half!r}")
    X = run_patches(cfg)
    val, test = val_test(stored)
    in_a = target_halves(test.image_id.values)
    pred_mask = ~in_a if half == "A" else in_a                     # predict only the half not used in training
    model = load_model(fold, tag, seed, device, cfg)

    yv = hard_labels(val)
    pv_plain = predict(model, X, val.row.values, device)
    pv_tta = predict(model, X, val.row.values, device, tta=True)
    thr_plain, thr_tta = best_threshold(yv, pv_plain)[0], best_threshold(yv, pv_tta)[0]

    part = test[pred_mask]
    p_plain = predict(model, X, part.row.values, device)
    p_tta = predict(model, X, part.row.values, device, tta=True)
    df = pd.DataFrame({"row": part.row.values, "image_id": part.image_id.values, "y": hard_labels(part),
                       "p_plain": p_plain, "p_tta": p_tta, "thr_plain": thr_plain, "thr_tta": thr_tta})
    info = {"tag": tag, "trained_with_half": half, "predicts_half": OTHER[half],
            "n_rows": int(len(part)), "n_cases": int(part.image_id.nunique()),
            "thr_plain": thr_plain, "thr_tta": thr_tta,
            "max_abs_diff_vs_stored": float(np.abs(p_plain - part.p.values).max()),
            "max_abs_diff_val_vs_stored": float(np.abs(pv_plain - val.p.values).max()),
            "n_domains": int(cfg["n_domains"]), "patches": cfg["patches"]}
    return df, info, test.row.values


def main(base, seeds, folds=FOLDS, out_dir=None):
    device = get_device()
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        for s in seeds:
            name = f"{fold}_{base}_seed{s}"
            if (out_dir / f"{name}.json").exists():
                continue
            t0 = time.time()
            runs = {h: load_run(fold, f"{base}_{h}", s) for h in ("A", "B")}   # file checks only, no GPU work
            missing = [f"{base}_{h}" for h in ("A", "B") if runs[h] is None]
            if missing:
                print(f"{name}: half-model(s) {', '.join(missing)} missing -- skipped", file=sys.stderr, flush=True)
                continue
            res = {h: eval_half(fold, base, h, s, device, runs[h]) for h in ("A", "B")}
            all_rows = res["A"][2]
            if not np.array_equal(all_rows, res["B"][2]):
                raise ValueError(f"{name}: the two half-models have different test_out_all rows")
            merged = pd.concat([res[h][0] for h in ("A", "B")]).sort_values("row").reset_index(drop=True)
            if not np.array_equal(merged.row.values, all_rows):
                raise ValueError(f"{name}: merged predictions do not cover every target row exactly once")
            merged.to_csv(out_dir / f"{name}.csv", index=False)
            halves = {h: res[h][1] for h in ("A", "B")}
            info = {"max_abs_diff_vs_stored": max(halves[h]["max_abs_diff_vs_stored"] for h in halves),
                    "per_row_thresholds": True, "halves": halves,
                    "seconds": round(time.time() - t0, 1)}
            (out_dir / f"{name}.json").write_text(json.dumps(info, indent=1))
            print(name, {k: info[k] for k in ("max_abs_diff_vs_stored", "seconds")},
                  {h: (halves[h]["n_rows"], halves[h]["thr_plain"], halves[h]["thr_tta"]) for h in halves}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="stain_uda", help="half-model tags are {base}_A and {base}_B")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", nargs="+", choices=FOLDS, default=FOLDS)
    ap.add_argument("--out-dir", default=None, help="default results/preds_tta")
    a = ap.parse_args()
    main(a.base, a.seeds, a.folds, a.out_dir)
