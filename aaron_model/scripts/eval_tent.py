#!/usr/bin/env python3
"""
Cross-fitted TENT and TENT+IM on the saved checkpoints (test-time adaptation, no retraining, no labels).

  tent    TENT (Wang et al., ICLR 2021): minimise the mean prediction entropy of unlabelled target
          batches, updating only the BatchNorm affine parameters (weight, bias). During adaptation the
          BatchNorm layers normalise with the batch statistics (train mode); every other module is in
          eval mode (dropout off). No augmentation.
  tentim  TENT + information maximisation as in SHOT (Liang et al., ICML 2020): mean per-sample entropy
          minus the entropy of the batch-mean prediction (diversity term), weights 1 and 1.
  Binary model: p = sigmoid(logit), binary entropy H(p) = -p log p - (1-p) log(1-p).

Hyper-parameters fixed a priori (ADDONS_PROTOCOL.md): Adam lr 1e-3, betas (0.9, 0.999), no weight decay,
batch 64, exactly one epoch over the adaptation half (order: torch.randperm with a torch.Generator seeded
by the run seed; the last, smaller batch is kept).

Cross-fitting (halves from addon_common.target_halves): for each half, a fresh model is adapted on the
OTHER half's target candidate patches, its BatchNorm running statistics are then re-estimated on that same
adaptation half with the adapted model (the AdaBN pass of eval_tta_adabn.py, cumulative average), and the
prediction half is predicted in eval mode, plain and with 8-view TTA. Every target patch is therefore
predicted by a model that never saw its slide. Thresholds for downstream evaluation: thr_plain (p_tent,
p_tentim) and thr_tta (p_tent_tta, p_tentim_tta) from the run's preds_tta JSON (source validation, source
BatchNorm). This script never reads target labels.

Usage:  python scripts/eval_tent.py [--tags ...] [--seeds 0 1 2] [--folds CS2 S360 XR] [--out-prefix smoke_]
Output: results/preds_tta/{prefix}{fold}_{tag}_seed{s}_tent.csv   row,p_tent,p_tent_tta,p_tentim,p_tentim_tta
        results/preds_tta/{prefix}{fold}_{tag}_seed{s}_tent.json  diagnostics (mean prob / entropy before and
                                                                  after adaptation per half, steps, seconds)
Runs whose outputs already exist are skipped; runs trained with --uda-half are skipped.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from addon_common import FOLDS, get_device, load_model, load_patches, run_config, target_halves  # noqa: E402
from eval_tta_adabn import adabn, batches, predict  # noqa: E402
from train_patch import ROOT  # noqa: E402

HP = {"optimizer": "Adam", "lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.0, "bs": 64, "epochs": 1,
      "params": "BatchNorm2d weight+bias", "bn_mode": "train (batch statistics)", "augmentation": None,
      "im_weights": {"entropy": 1.0, "marginal_entropy": 1.0}, "marginal_eps": 1e-6,
      "after_adaptation": "AdaBN re-estimation on the adaptation half (momentum None), then eval-mode prediction"}
OBJECTIVES = ("tent", "tentim")
OUT_DIR = ROOT / "results" / "preds_tta"


# ------------------------------------------------------------------ objectives
def entropy_from_logits(z):
    """Per-sample binary entropy of p = sigmoid(z), computed stably from the logit:
    H = p * softplus(-z) + (1 - p) * softplus(z)   (since -log p = softplus(-z), -log(1-p) = softplus(z))."""
    p = torch.sigmoid(z)
    return p * F.softplus(-z) + (1 - p) * F.softplus(z)


def binary_entropy(p, eps=HP["marginal_eps"]):
    p = p.clamp(eps, 1 - eps)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def objective(z, kind):
    ent = entropy_from_logits(z).mean()
    if kind == "tent":
        return ent
    if kind == "tentim":                                   # SHOT IM: L_ent - L_div
        return ent - binary_entropy(torch.sigmoid(z).mean())
    raise ValueError(kind)


def np_entropy(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def summary(p):
    p = np.asarray(p, dtype=np.float64)
    return {"mean_p": float(p.mean()), "mean_entropy": float(np_entropy(p).mean()), "sd_p": float(p.std()),
            "frac_ge_0.5": float((p >= 0.5).mean())}


def load_cf(test, cf_path):
    """AdaBN-only cross-fitted predictions ({name}_cf.csv from eval_adabn_crossfit.py), aligned to `test`.
    Diagnostics only. That file is written non-atomically, so it is used only when it is provably complete:
    it ends with a newline, holds exactly the rows of `test` once each, and both columns are finite.
    Otherwise returns (None, reason) and the run continues without the comparison block."""
    if not cf_path.exists():
        return None, "missing"
    try:
        with open(cf_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                return None, "incomplete (empty file)"
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                return None, "incomplete (no trailing newline, file may still be being written)"
        raw = pd.read_csv(cf_path)
        if len(raw) != len(test) or set(raw.row) != set(test.row):
            return None, f"incomplete ({len(raw)} rows, expected the {len(test)} test rows)"
        cf = test[["row"]].merge(raw, on="row", how="left", validate="one_to_one")
        vals = cf[["p_adabncf", "p_adabncf_tta"]].to_numpy(dtype=np.float64)
    except Exception as e:                                     # partial / malformed file: never stop the run
        return None, f"unreadable ({type(e).__name__}: {e})"
    if not np.all(np.isfinite(vals)):
        return None, f"incomplete ({int((~np.isfinite(vals)).any(axis=1).sum())} non-finite rows)"
    return cf, "ok"


# ------------------------------------------------------------------ adaptation
def configure_for_adaptation(model):
    """Everything frozen and in eval mode except the BatchNorm layers: train mode (batch statistics),
    affine weight and bias trainable. Returns the trainable parameters."""
    model.eval()
    model.requires_grad_(False)
    params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            for p in (m.weight, m.bias):
                p.requires_grad_(True)
                params.append(p)
    return params


def adapt(model, X, rows, device, kind, seed):
    params = configure_for_adaptation(model)
    theta0 = torch.cat([p.detach().flatten() for p in params]).clone()
    opt = torch.optim.Adam(params, lr=HP["lr"], betas=tuple(HP["betas"]), weight_decay=HP["weight_decay"])
    g = torch.Generator().manual_seed(int(seed))
    order = rows[torch.randperm(len(rows), generator=g).numpy()]
    losses = []
    for xb in batches(X, order, device, bs=HP["bs"]):
        # Input marked as requiring grad: the MPS batch_norm backward kernel crashes (torch 2.14, nil
        # grad_input) when the first BatchNorm's input needs no gradient. No effect on the update.
        loss = objective(model(xb.requires_grad_(True))[0], kind)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    theta1 = torch.cat([p.detach().flatten() for p in params])
    model.requires_grad_(False)
    model.eval()
    if not np.all(np.isfinite(losses)):
        raise FloatingPointError(f"non-finite adaptation loss ({kind}): {losses}")
    return {"steps": len(losses), "n_params": int(theta0.numel()),
            "loss_first": losses[0], "loss_last": losses[-1], "loss_mean": float(np.mean(losses)),
            "param_rel_change": float((theta1 - theta0).norm() / theta0.norm())}


# ------------------------------------------------------------------------ main
def run_one(fold, tag, seed, device, prefix):
    name = f"{fold}_{tag}_seed{seed}"
    out_csv, out_json = OUT_DIR / f"{prefix}{name}_tent.csv", OUT_DIR / f"{prefix}{name}_tent.json"
    if out_csv.exists() and out_json.exists():
        print(f"{name}: outputs exist, skipped", flush=True)
        return
    needed = [ROOT / "results" / f"{name}.json", ROOT / "checkpoints" / f"{name}.pt",
              OUT_DIR / f"{name}.csv", OUT_DIR / f"{name}.json"]
    missing = [str(p.relative_to(ROOT)) for p in needed if not p.exists()]
    if missing:
        print(f"{name}: SKIPPED, missing {missing}", flush=True)
        return
    cfg = run_config(fold, tag, seed)
    if cfg.get("uda_half"):
        print(f"{name}: trained with --uda-half {cfg['uda_half']}, skipped", flush=True)
        return

    t0 = time.time()
    X = load_patches(cfg["patches"])
    n_meta = len(pd.read_csv(ROOT / "data" / "patches_meta.csv", usecols=["ann_id"]))
    if len(X) != n_meta:
        raise ValueError(f"{cfg['patches']}: {len(X)} patches, patches_meta.csv has {n_meta} rows")
    test = pd.read_csv(OUT_DIR / f"{name}.csv")               # test_out_all rows; output keeps this order
    rows = test.row.values
    half_a = target_halves(test.image_id.values)
    cf, cf_status = load_cf(test, OUT_DIR / f"{name}_cf.csv")
    if cf is None and cf_status != "missing":
        print(f"{name}: note: {name}_cf.csv {cf_status}; AdaBN-only comparison left out of the diagnostics",
              flush=True)

    out = {c: np.full(len(test), np.nan) for c in ("p_tent", "p_tent_tta", "p_tentim", "p_tentim_tta")}
    diag = {"fold": fold, "tag": tag, "seed": seed, "variant": cfg.get("variant"), "patches": cfg["patches"],
            "n_domains": int(cfg["n_domains"]), "hyper_parameters": HP, "device": str(device),
            "adabncf_comparison": cf_status,
            "halves": {h: {"n_pred": int(m.sum()), "n_pred_cases": int(test.image_id[m].nunique()),
                           "n_adapt": int((~m).sum()), "n_adapt_cases": int(test.image_id[~m].nunique())}
                       for h, m in (("A", half_a), ("B", ~half_a))}}
    for kind in OBJECTIVES:
        diag[kind] = {}
        for h, pred_mask in (("A", half_a), ("B", ~half_a)):
            t1 = time.time()
            adapt_rows, pred_rows = rows[~pred_mask], rows[pred_mask]
            model = load_model(fold, tag, seed, device, cfg)  # fresh source model for every half
            info = adapt(model, X, adapt_rows, device, kind, seed)
            adabn(model, X, adapt_rows, device)                # BN running stats from the adaptation half
            p, p_tta = predict(model, X, pred_rows, device), predict(model, X, pred_rows, device, tta=True)
            out[f"p_{kind}"][pred_mask], out[f"p_{kind}_tta"][pred_mask] = p, p_tta
            d = {"adapted_on": "B" if h == "A" else "A", **info,
                 "before": {"plain": summary(test.p_plain.values[pred_mask]),
                            "tta": summary(test.p_tta.values[pred_mask])},
                 "after": {"plain": summary(p), "tta": summary(p_tta)},
                 "seconds": round(time.time() - t1, 1)}
            if cf is not None:                                 # AdaBN-only (cross-fitted), isolates the TENT step
                d["adabncf"] = {"plain": summary(cf.p_adabncf.values[pred_mask]),
                                "tta": summary(cf.p_adabncf_tta.values[pred_mask])}
                d["mean_abs_diff_vs_adabncf"] = float(np.abs(p - cf.p_adabncf.values[pred_mask]).mean())
            diag[kind][f"pred_{h}"] = d
            del model
    bad = [c for c, v in out.items() if not np.all(np.isfinite(v))]
    if bad:
        raise FloatingPointError(f"{name}: non-finite predictions in {bad}")
    diag["seconds"] = round(time.time() - t0, 1)

    tmp_csv, tmp_json = out_csv.with_suffix(".csv.tmp"), out_json.with_suffix(".json.tmp")
    pd.DataFrame({"row": rows, **out}).to_csv(tmp_csv, index=False)
    tmp_json.write_text(json.dumps(diag, indent=1, allow_nan=False))   # strict JSON (no NaN tokens)
    os.replace(tmp_csv, out_csv)
    os.replace(tmp_json, out_json)
    msg = " ".join(f"{k}: mean p {test.p_plain.mean():.3f}->{out['p_' + k].mean():.3f}" for k in OBJECTIVES)
    steps = sum(diag[k][f"pred_{h}"]["steps"] for k in OBJECTIVES for h in "AB")
    print(f"{name}: {msg}  steps {steps}  ({diag['seconds']}s)", flush=True)


def main(a):
    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fold in a.folds:
        for tag in a.tags:
            for s in a.seeds:
                run_one(fold, tag, s, device, a.out_prefix)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["baseline", "stain", "adv", "stain_adv", "stain_mix"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", nargs="+", default=FOLDS, choices=FOLDS)
    ap.add_argument("--out-prefix", default="", help="prefix for the output files, e.g. smoke_ for tests")
    main(ap.parse_args())
