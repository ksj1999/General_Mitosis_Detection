#!/usr/bin/env python3
"""
Test-time augmentation (TTA) and AdaBN on the saved checkpoints -- no retraining.

For every checkpoint (fold x tag x seed) four predictions on the unseen scanner:
  plain      the model as trained (reproduces results/preds/)
  tta        mean probability over the 8 flips/90-degree rotations of each patch
  adabn      every BatchNorm's running mean/var re-estimated on the unseen scanner's
             (unlabelled) patches before predicting (Li et al. 2016)
  adabn_tta  both
The decision threshold is chosen on the source validation set with the source
BatchNorm statistics (plain or TTA, matching the mode), as in training.
AdaBN uses the unlabelled target images, so it is test-time adaptation, not pure
domain generalization.

Config-aware: each run's model (n_domains) and patch file come from its "config"
block (addon_common.run_config; old runs get the defaults). Runs trained with
target images (config uda_half set) are skipped -- they are evaluated by eval_uda.py.

Usage:  python scripts/eval_tta_adabn.py [--tags ...] [--seeds 0 1 2] [--folds CS2 S360 XR] [--out-dir DIR]
        (--variants is an alias of --tags)
Output: results/preds_tta/{fold}_{tag}_seed{s}.csv and .json (existing files are not recomputed)
"""
import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_patch import MEAN, STD, ROOT, Net, best_threshold  # noqa: E402,F401  (Net kept importable)
from addon_common import FOLDS, get_device, load_model, load_patches, run_config  # noqa: E402

OUT_DIR = ROOT / "results" / "preds_tta"


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


# ------------------------------------------------------------ shared run helpers
@functools.lru_cache(maxsize=1)
def _meta_y():
    return pd.read_csv(ROOT / "data" / "patches_meta.csv").y.values.astype(int)


def hard_labels(stored):
    """Binary labels of the stored prediction rows, from data/patches_meta.csv (the stored y column
    may hold soft labels). If the stored y is binary it must agree (guards against a row-order mismatch)."""
    y = _meta_y()[stored.row.values]
    sy = stored.y.values
    if np.isin(sy, (0, 1)).all() and not np.array_equal(sy.astype(int), y):
        raise ValueError("stored labels differ from data/patches_meta.csv -- row order mismatch?")
    return y


def load_run(fold, tag, seed):
    """(config, stored predictions) of one run, or None (with a message) when any of its files is missing."""
    name = f"{fold}_{tag}_seed{seed}"
    files = [ROOT / "results" / f"{name}.json", ROOT / "results" / "preds" / f"{name}.csv",
             ROOT / "checkpoints" / f"{name}.pt"]
    missing = [str(f.relative_to(ROOT)) for f in files if not f.exists()]
    if missing:
        print(f"{name}: missing {', '.join(missing)} -- skipped", file=sys.stderr, flush=True)
        return None
    return run_config(fold, tag, seed), pd.read_csv(files[1])


def run_patches(cfg):
    """Patch tensor of the run's patch file (cached per path), checked against patches_meta.csv."""
    X = load_patches(cfg["patches"])
    if len(X) != len(_meta_y()):
        raise ValueError(f"{cfg['patches']}: {len(X)} patches, patches_meta.csv has {len(_meta_y())} rows")
    return X


def val_test(stored):
    """Source val rows and target test_out_all rows of the stored predictions, sorted by row."""
    val = stored[stored.set == "val"].sort_values("row").reset_index(drop=True)
    test = stored[stored.set == "test_out_all"].sort_values("row").reset_index(drop=True)
    return val, test


# --------------------------------------------------------------------- main
def main(tags, seeds, folds=FOLDS, out_dir=None):
    device = get_device()
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        for v in tags:
            for s in seeds:
                tag = f"{fold}_{v}_seed{s}"
                if (out_dir / f"{tag}.json").exists():
                    continue
                run = load_run(fold, v, s)
                if run is None:
                    continue
                cfg, stored = run
                if cfg.get("uda_half"):
                    print(f"{tag}: trained with target half {cfg['uda_half']} -- use eval_uda.py, skipped", flush=True)
                    continue
                t0 = time.time()
                X = run_patches(cfg)
                val, test = val_test(stored)
                vi, ti = val.row.values, test.row.values
                yv, yt = hard_labels(val), hard_labels(test)

                model = load_model(fold, v, s, device, cfg)
                pv_plain, pv_tta = predict(model, X, vi, device), predict(model, X, vi, device, tta=True)
                thr_plain, thr_tta = best_threshold(yv, pv_plain)[0], best_threshold(yv, pv_tta)[0]
                p_plain, p_tta = predict(model, X, ti, device), predict(model, X, ti, device, tta=True)
                adabn(model, X, ti, device)
                p_ad, p_ad_tta = predict(model, X, ti, device), predict(model, X, ti, device, tta=True)

                pd.DataFrame({"row": ti, "image_id": test.image_id.values, "y": yt,
                              "p_plain": p_plain, "p_tta": p_tta, "p_adabn": p_ad, "p_adabn_tta": p_ad_tta}
                             ).to_csv(out_dir / f"{tag}.csv", index=False)
                info = {"thr_plain": thr_plain, "thr_tta": thr_tta,
                        "max_abs_diff_vs_stored": float(np.abs(p_plain - test.p.values).max()),
                        "seconds": round(time.time() - t0, 1)}
                (out_dir / f"{tag}.json").write_text(json.dumps(info))
                print(tag, info, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", "--variants", dest="tags", nargs="+",
                    default=["baseline", "stain", "adv", "stain_adv", "mix", "stain_mix"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", nargs="+", choices=FOLDS, default=FOLDS)
    ap.add_argument("--out-dir", default=None, help="default results/preds_tta")
    a = ap.parse_args()
    main(a.tags, a.seeds, a.folds, a.out_dir)
