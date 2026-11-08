#!/usr/bin/env python3
"""
Cross-fitted AdaBN: the unseen scanner's cases are split into two halves by case
(addon_common.target_halves: sorted case ids, half A = [::2]); BatchNorm statistics
are re-estimated on one half and predictions are made on the other, then the halves
swap. Every patch is therefore predicted by a model whose BatchNorm statistics never
saw that patch (or any patch from its slide).
Thresholds as in eval_tta_adabn.py (source validation, source BatchNorm).

Config-aware like eval_tta_adabn.py (model and patch file from the run's config);
runs trained with target images (config uda_half set) are skipped.

Usage:  python scripts/eval_adabn_crossfit.py [--tags ...] [--seeds 0 1 2] [--folds CS2 S360 XR] [--out-dir DIR]
        (--variants is an alias of --tags)
Output: results/preds_tta/{fold}_{tag}_seed{s}_cf.csv  (columns row, p_adabncf, p_adabncf_tta)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_tta_adabn import OUT_DIR, adabn, load_run, predict, run_patches, val_test  # noqa: E402
from addon_common import FOLDS, get_device, load_model, target_halves  # noqa: E402


def main(tags, seeds, folds=FOLDS, out_dir=None):
    device = get_device()
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        for v in tags:
            for s in seeds:
                tag = f"{fold}_{v}_seed{s}"
                if (out_dir / f"{tag}_cf.csv").exists():
                    continue
                run = load_run(fold, v, s)
                if run is None:
                    continue
                cfg, stored = run
                if cfg.get("uda_half"):
                    print(f"{tag}: trained with target half {cfg['uda_half']} -- use eval_uda.py, skipped", flush=True)
                    continue
                X = run_patches(cfg)
                test = val_test(stored)[1]                                  # test_out_all, sorted by row
                half = target_halves(test.image_id.values)                  # True = half A
                p_cf, p_cf_tta = np.empty(len(test)), np.empty(len(test))
                for pred_mask in (half, ~half):
                    model = load_model(fold, v, s, device, cfg)
                    adabn(model, X, test.row.values[~pred_mask], device)  # statistics from the other half only
                    rows = test.row.values[pred_mask]
                    p_cf[pred_mask] = predict(model, X, rows, device)
                    p_cf_tta[pred_mask] = predict(model, X, rows, device, tta=True)
                tmp = out_dir / f".{tag}_cf.csv.tmp"                       # atomic: the skip test is file existence
                pd.DataFrame({"row": test.row.values, "p_adabncf": p_cf, "p_adabncf_tta": p_cf_tta}
                             ).to_csv(tmp, index=False)
                os.replace(tmp, out_dir / f"{tag}_cf.csv")
                print(tag, "done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", "--variants", dest="tags", nargs="+",
                    default=["baseline", "stain", "stain_adv", "stain_mix"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", nargs="+", choices=FOLDS, default=FOLDS)
    ap.add_argument("--out-dir", default=None, help="default results/preds_tta")
    a = ap.parse_args()
    main(a.tags, a.seeds, a.folds, a.out_dir)
