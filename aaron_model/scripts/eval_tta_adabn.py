#!/usr/bin/env python3
"""
Test-time augmentation (TTA) and AdaBN on the saved checkpoints -- no retraining.

For every checkpoint (fold x variant x seed) four predictions on the unseen scanner:
  plain      the model as trained (reproduces results/preds/)
  tta        mean probability over the 8 flips/90-degree rotations of each patch
  adabn      every BatchNorm's running mean/var re-estimated on the unseen scanner's
             (unlabelled) patches before predicting (Li et al. 2016)
  adabn_tta  both
The decision threshold is chosen on the source validation set with the source
BatchNorm statistics (plain or TTA, matching the mode), as in training.
AdaBN uses the unlabelled target images, so it is test-time adaptation, not pure
domain generalization.

Usage:  python scripts/eval_tta_adabn.py [--variants ...] [--seeds 0 1 2]
Output: results/preds_tta/{fold}_{variant}_seed{s}.csv and .json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_patch import MEAN, STD, ROOT, Net, best_threshold  # noqa: E402

FOLDS = ["CS2", "S360", "XR"]


def dihedral(x, k):
    """k = 0..7: rotation by 90*k degrees, horizontally flipped first when k >= 4."""
    if k >= 4:
        x = x.flip(3)
    return torch.rot90(x, k % 4, dims=(2, 3))


def batches(X, idx, device, bs=256):
    mean, std = MEAN.to(device), STD.to(device)
    for i in range(0, len(idx), bs):
        xb = X[idx[i:i + bs]].to(device).permute(0, 3, 1, 2).float().div_(255)
        yield (xb - mean) / std


@torch.no_grad()
def predict(model, X, idx, device, tta=False):
    model.eval()
    out = []
    for xb in batches(X, idx, device):
        if tta:
            p = torch.stack([torch.sigmoid(model(dihedral(xb, k))[0]) for k in range(8)]).mean(0)
        else:
            p = torch.sigmoid(model(xb)[0])
        out.append(p.cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def adabn(model, X, idx, device):
    """Replace the BatchNorm running statistics by those of the given unlabelled images."""
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None                     # cumulative average over all batches
    model.train()
    for xb in batches(X, idx, device):
        model.encoder(xb)
    model.eval()


def main(variants, seeds):
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(np.load(ROOT / "data" / "patches_128.npz")["X"])
    out_dir = ROOT / "results" / "preds_tta"
    out_dir.mkdir(exist_ok=True)
    for fold in FOLDS:
        for v in variants:
            for s in seeds:
                tag = f"{fold}_{v}_seed{s}"
                if (out_dir / f"{tag}.json").exists():
                    continue
                t0 = time.time()
                stored = pd.read_csv(ROOT / "results" / "preds" / f"{tag}.csv")
                val = stored[stored.set == "val"].sort_values("row")
                test = stored[stored.set == "test_out_all"].sort_values("row")
                vi, ti = val.row.values, test.row.values
                yv = val.y.values

                model = Net(n_domains=2).to(device)
                state = torch.load(ROOT / "checkpoints" / f"{tag}.pt", map_location="cpu")
                model.load_state_dict(state)
                pv_plain, pv_tta = predict(model, X, vi, device), predict(model, X, vi, device, tta=True)
                thr_plain, thr_tta = best_threshold(yv, pv_plain)[0], best_threshold(yv, pv_tta)[0]
                p_plain, p_tta = predict(model, X, ti, device), predict(model, X, ti, device, tta=True)
                adabn(model, X, ti, device)
                p_ad, p_ad_tta = predict(model, X, ti, device), predict(model, X, ti, device, tta=True)

                pd.DataFrame({"row": ti, "image_id": test.image_id.values, "y": test.y.values,
                              "p_plain": p_plain, "p_tta": p_tta, "p_adabn": p_ad, "p_adabn_tta": p_ad_tta}
                             ).to_csv(out_dir / f"{tag}.csv", index=False)
                info = {"thr_plain": thr_plain, "thr_tta": thr_tta,
                        "max_abs_diff_vs_stored": float(np.abs(p_plain - test.p.values).max()),
                        "seconds": round(time.time() - t0, 1)}
                (out_dir / f"{tag}.json").write_text(json.dumps(info))
                print(tag, info, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["baseline", "stain", "adv", "stain_adv", "mix", "stain_mix"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    main(a.variants, a.seeds)
