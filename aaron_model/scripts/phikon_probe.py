#!/usr/bin/env python3
"""
Pathology foundation model baseline: frozen Phikon (Owkin; ViT-B/16, iBOT on 40M
TCGA tiles) features + logistic-regression probe, same folds/splits/metrics as the
CNN runs, so results are directly comparable.

Patches are the existing 128x128 (32 um) crops, resized to 224x224 and normalised
with ImageNet statistics (Phikon's preprocessing). The CLS token (768-d) is the
feature. Two variants:
  phikon        probe trained on the plain features
  phikon_stain  probe trained on the plain features plus 4 H&E-stain-augmented
                copies of every training patch (sigma = 0.05, same jitter as CNNs)
The probe's C is picked on the source validation set (AUROC); the decision
threshold is picked on the same validation set (F1), as for the CNNs.

Usage:  python scripts/phikon_probe.py [--seeds 0 1 2]
Output: data/phikon_feats.npz (cache), results/{fold}_{variant}_seed{s}.json,
        results/preds/{fold}_{variant}_seed{s}.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_patch import MEAN, STD, ROOT, SCANNERS, best_threshold, hed_jitter, metrics  # noqa: E402

N_AUG = 4


@torch.no_grad()
def extract(X, device, n_aug=N_AUG, sigma=0.05, bs=64):
    from transformers import ViTModel
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False).to(device).eval()
    mean, std = MEAN.to(device), STD.to(device)
    torch.manual_seed(0)
    plain, aug = [], [[] for _ in range(n_aug)]
    t0 = time.time()
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(device).permute(0, 3, 1, 2).float().div_(255)
        versions = [x] + [hed_jitter(x, sigma) for _ in range(n_aug)]
        for j, xv in enumerate(versions):
            xv = F.interpolate(xv, size=224, mode="bilinear", align_corners=False)
            f = model(pixel_values=(xv - mean) / std).last_hidden_state[:, 0].float().cpu()
            (plain if j == 0 else aug[j - 1]).append(f)
        if i % (bs * 10) == 0:
            print(f"  features {i + len(x)}/{len(X)}  ({time.time() - t0:.0f}s)", flush=True)
    aug = np.stack([torch.cat(a).numpy() for a in aug]) if n_aug else np.zeros((0, len(X), 768), np.float32)
    return torch.cat(plain).numpy(), aug


def probe(Xtr, ytr, Xval, yval):
    best = None
    for C in (0.01, 0.1, 1.0):
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(Xtr), ytr)
        auc = roc_auc_score(yval, clf.predict_proba(sc.transform(Xval))[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, sc, clf)
    return best


def main(seeds, patches="data/patches_128.npz", name="phikon", n_aug=N_AUG):
    feats_path = ROOT / "data" / f"{name}_feats.npz"
    meta = pd.read_csv(ROOT / "data" / "patches_meta.csv")
    y = meta.y.values.astype(int)
    if feats_path.exists():
        d = np.load(feats_path)
        plain, aug = d["plain"], d["aug"]
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        arr = np.load(ROOT / patches)
        X = torch.from_numpy((arr["X"] if patches.endswith(".npz") else arr).copy())
        plain, aug = extract(X, device, n_aug=n_aug)
        np.savez(feats_path, plain=plain, aug=aug)
    print("features", plain.shape, aug.shape, flush=True)

    for target in ["CS2", "S360", "XR"]:
        sources = [s for s in SCANNERS if s != target]
        for s in seeds:
            ref = pd.read_csv(ROOT / "results" / "preds" / f"{target}_baseline_seed{s}.csv")   # identical splits
            val_rows = np.sort(ref[ref.set == "val"].row.values)
            val_cases = set(meta.image_id.values[val_rows])
            is_src = meta.scanner.isin(sources).values
            tr = np.where(is_src & (meta.split == "train").values & ~meta.image_id.isin(val_cases).values)[0]
            sets = {"val": val_rows,
                    "test_in": np.where(is_src & (meta.split == "test").values)[0],
                    "test_out": np.where((meta.scanner == target).values & (meta.split == "test").values)[0],
                    "test_out_all": np.where((meta.scanner == target).values)[0]}
            for variant in ((name, f"{name}_stain") if aug.shape[0] else (name,)):
                Xtr, ytr = plain[tr], y[tr]
                if variant.endswith("_stain"):
                    Xtr = np.concatenate([Xtr] + [aug[k][tr] for k in range(aug.shape[0])])
                    ytr = np.tile(y[tr], aug.shape[0] + 1)
                val_auc, C, sc, clf = probe(Xtr, ytr, plain[val_rows], y[val_rows])
                prob = {k: clf.predict_proba(sc.transform(plain[idx]))[:, 1] for k, idx in sets.items()}
                thr, _ = best_threshold(y[val_rows], prob["val"])
                res = {"fold": target, "variant": variant, "seed": s, "threshold": thr, "C": C, "val_auroc": val_auc}
                rows = []
                for k, idx in sets.items():
                    res[k] = metrics(y[idx], prob[k], thr)
                    rows.append(pd.DataFrame({"set": k, "row": idx, "ann_id": meta.ann_id.values[idx],
                                              "image_id": meta.image_id.values[idx], "scanner": meta.scanner.values[idx],
                                              "y": y[idx], "p": prob[k], "unanimous": meta.unanimous.values[idx],
                                              "vote_mitotic": meta.vote_mitotic.values[idx]}))
                tag = f"{target}_{variant}_seed{s}"
                (ROOT / "results" / f"{tag}.json").write_text(json.dumps(res, indent=2))
                pd.concat(rows).to_csv(ROOT / "results" / "preds" / f"{tag}.csv", index=False)
                o = res["test_out_all"]
                print(f"{tag:24s} C={C:<5} val AUROC {val_auc:.3f} | unseen F1 {o['f1']:.3f} "
                      f"P {o['precision']:.3f} R {o['recall']:.3f} AUROC {o['auroc']:.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--patches", default="data/patches_128.npz", help=".npz with X, or .npy, rows as in patches_meta.csv")
    ap.add_argument("--name", default="phikon", help="variant name; features cached as data/{name}_feats.npz")
    ap.add_argument("--n-aug", type=int, default=N_AUG, help="stain-augmented copies per patch (0 = plain only)")
    a = ap.parse_args()
    main(a.seeds, a.patches, a.name, a.n_aug)
