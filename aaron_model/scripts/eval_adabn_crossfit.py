#!/usr/bin/env python3
"""
Cross-fitted AdaBN: the unseen scanner's cases are split into two halves by case;
BatchNorm statistics are re-estimated on one half and predictions are made on the
other, then the halves swap. Every patch is therefore predicted by a model whose
BatchNorm statistics never saw that patch (or any patch from its slide).
Thresholds as in eval_tta_adabn.py (source validation, source BatchNorm).

Usage:  python scripts/eval_adabn_crossfit.py [--variants ...] [--seeds 0 1 2]
Output: results/preds_tta/{tag}_cf.csv  (columns p_adabncf, p_adabncf_tta)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_tta_adabn import FOLDS, adabn, predict  # noqa: E402
from train_patch import ROOT, Net  # noqa: E402


def main(variants, seeds):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    X = torch.from_numpy(np.load(ROOT / "data" / "patches_128.npz")["X"].copy())
    out_dir = ROOT / "results" / "preds_tta"
    for fold in FOLDS:
        for v in variants:
            for s in seeds:
                tag = f"{fold}_{v}_seed{s}"
                if (out_dir / f"{tag}_cf.csv").exists():
                    continue
                test = pd.read_csv(out_dir / f"{tag}.csv").sort_values("row").reset_index(drop=True)
                cases = np.sort(test.image_id.unique())
                half = np.isin(test.image_id.values, cases[::2])          # alternate cases -> two halves
                p_cf, p_cf_tta = np.empty(len(test)), np.empty(len(test))
                for pred_mask in (half, ~half):
                    model = Net(n_domains=2).to(device)
                    model.load_state_dict(torch.load(ROOT / "checkpoints" / f"{tag}.pt", map_location="cpu"))
                    adabn(model, X, test.row.values[~pred_mask], device)  # statistics from the other half only
                    rows = test.row.values[pred_mask]
                    p_cf[pred_mask] = predict(model, X, rows, device)
                    p_cf_tta[pred_mask] = predict(model, X, rows, device, tta=True)
                pd.DataFrame({"row": test.row.values, "p_adabncf": p_cf, "p_adabncf_tta": p_cf_tta}
                             ).to_csv(out_dir / f"{tag}_cf.csv", index=False)
                print(tag, "done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["baseline", "stain", "stain_adv", "stain_mix"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    main(a.variants, a.seeds)
