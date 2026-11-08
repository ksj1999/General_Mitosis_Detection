#!/usr/bin/env python3
"""
Cross-fitted AdaBN from random tissue crops [adabnrand] (ADDONS_PROTOCOL.md).

Like eval_adabn_crossfit.py, but the BatchNorm statistics are re-estimated on random
tissue crops (data/random_crops_128.npz, prepare_random_crops.py) of the target scanner
instead of on the candidate patches. Cross-fitting: target cases are split by
addon_common.target_halves over the candidate rows' case ids (sorted case ids, half A =
[::2]); the model that predicts half A gets its statistics from ALL random crops of the
half-B slides, and vice versa. No labels are used; thresholds for the summary are the
run's thr_plain / thr_tta from results/preds_tta/{run}.json (source val, source BN).

The crops are fed to AdaBN in slide order (as the candidate rows are in eval_adabn_crossfit.py),
with the same batching (256) and cumulative averaging (eval_tta_adabn.adabn).

Only runs whose patch file is data/patches_128.npz are evaluated (the crops are cut at the
same field of view and size); runs trained with target images (config uda_half) are skipped.

Crop-file checks (hard errors, before any output is written): every slide has the same number
of crops (prepare_random_crops.py writes exactly --per-slide per slide), the crop slides of each
requested scanner equal its candidate cases in data/patches_meta.csv and the run's target cases,
and each adaptation set is non-empty and covers every slide of the other half. Each output has a
sidecar {run}_rand.json recording the crop file (path, sha256 of X/image_id/cx/cy) it came from;
an existing output whose sidecar names a different crop file is an error (delete it or use
--out-dir), so outputs from different crop files cannot mix silently.

Interpretation note: with the contract thresholds (source val, source BN statistics) this mode
calls ~87-94 % of target candidates mitotic (prevalence ~0.43) while its AUROC stays near the
plain model's, i.e. its F1 mainly measures a calibration shift (random crops have less dark
nuclear content than candidates), not lost ranking. Summaries should report AUROC and the
predicted-positive share next to F1. The contract thresholds are the only primary result; any
other threshold rule for this mode was conceived after seeing target results and could only be
a labelled post-hoc sensitivity analysis.

Usage:  python scripts/eval_adabn_random.py [--tags ...] [--seeds 0 1 2] [--folds CS2 S360 XR]
                                            [--crops data/random_crops_128.npz] [--out-dir DIR]
Output: results/preds_tta/{fold}_{tag}_seed{s}_rand.csv  (columns row, p_adabnrand, p_adabnrand_tta)
        results/preds_tta/{fold}_{tag}_seed{s}_rand.json provenance (crop file, halves, crop counts)
        existing files are not recomputed (after checking their provenance, when recorded)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from addon_common import FOLDS, get_device, load_model, load_patches, run_config, target_halves  # noqa: E402
from eval_tta_adabn import adabn, predict  # noqa: E402
from train_patch import ROOT  # noqa: E402

PATCHES = (ROOT / "data" / "patches_128.npz").resolve()
CROPS = ROOT / "data" / "random_crops_128.npz"


def resolve(p):
    p = Path(p)
    return (p if p.is_absolute() else ROOT / p).resolve()


def rel(p):
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def load_crops(path):
    """Crops tensor, image ids, scanners, and a provenance dict (path, size, sha256, per-slide count).
    Raises when the file is not a complete prepare_random_crops.py output (unequal per-slide counts)."""
    path = resolve(path)
    d = np.load(path)
    arrs = {k: np.ascontiguousarray(d[k]) for k in ("X", "image_id", "scanner", "cx", "cy")}
    iid, sc = arrs["image_id"].astype(np.int64), arrs["scanner"].astype(str)
    h = hashlib.sha256()
    for k in ("X", "image_id", "cx", "cy"):
        h.update(arrs[k].data)
    pairs = pd.DataFrame({"image_id": iid, "scanner": sc})
    if pairs.groupby("image_id").scanner.nunique().max() != 1:
        raise RuntimeError(f"{path}: an image_id appears under more than one scanner")
    counts = pairs.image_id.value_counts()
    if counts.nunique() != 1:
        raise RuntimeError(f"{path}: unequal crops per slide ({counts.min()}-{counts.max()}); "
                           "not a complete prepare_random_crops.py output")
    prov = {"crops": rel(path), "crops_bytes": path.stat().st_size, "crops_sha256": h.hexdigest(),
            "n_crops": int(len(iid)), "per_slide": int(counts.iloc[0]), "n_slides": int(len(counts))}
    return torch.from_numpy(arrs["X"]), iid, sc, prov


def check_slides(what, crop_cases, cases):
    crop_cases, cases = set(np.asarray(crop_cases).tolist()), set(np.asarray(cases).tolist())
    if crop_cases != cases:
        raise RuntimeError(f"{what}: random-crop slides differ from the target candidate cases "
                           f"(missing {sorted(cases - crop_cases)}, extra {sorted(crop_cases - cases)}); "
                           "rebuild the crop file with prepare_random_crops.py (all slides)")


def main(tags, seeds, folds=FOLDS, crops=None, out_dir=None):
    device = get_device()
    in_dir = ROOT / "results" / "preds_tta"
    out_dir = Path(out_dir) if out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "data" / "patches_meta.csv")
    R, r_iid, r_sc, prov = load_crops(crops or CROPS)
    for fold in folds:                                                 # fail before any output is written
        check_slides(f"{prov['crops']} / {fold}", r_iid[r_sc == fold], meta.image_id[meta.scanner == fold])
    print(f"crops {prov['crops']}: {prov['n_crops']} crops, {prov['n_slides']} slides x {prov['per_slide']}, "
          f"sha256 {prov['crops_sha256'][:16]}", flush=True)
    X = load_patches(PATCHES)
    assert len(X) == len(meta), (len(X), len(meta))
    for fold in folds:
        for v in tags:
            for s in seeds:
                name = f"{fold}_{v}_seed{s}"
                out, side = out_dir / f"{name}_rand.csv", out_dir / f"{name}_rand.json"
                if out.exists():
                    if side.exists():
                        old = json.loads(side.read_text())
                        if old.get("crops_sha256") != prov["crops_sha256"]:
                            raise RuntimeError(
                                f"{out} was computed from crop file {old.get('crops')} (sha256 "
                                f"{str(old.get('crops_sha256'))[:16]}), not {prov['crops']} (sha256 "
                                f"{prov['crops_sha256'][:16]}); delete it or write to another --out-dir")
                    else:
                        print(f"{name}: existing output has no provenance sidecar -- skipped unchecked", flush=True)
                    continue
                if not (ROOT / "results" / f"{name}.json").exists() or not (ROOT / "checkpoints" / f"{name}.pt").exists():
                    print(f"{name}: results JSON or checkpoint missing -- skipped", flush=True)
                    continue
                if not (in_dir / f"{name}.csv").exists():
                    print(f"{name}: results/preds_tta/{name}.csv missing (run eval_tta_adabn.py first) -- skipped",
                          flush=True)
                    continue
                cfg = run_config(fold, v, s)
                if resolve(cfg["patches"]) != PATCHES:
                    print(f"{name}: patches {cfg['patches']} != data/patches_128.npz -- skipped", flush=True)
                    continue
                if cfg.get("uda_half"):
                    print(f"{name}: trained with target half {cfg['uda_half']} -- skipped", flush=True)
                    continue
                t0 = time.time()
                test = pd.read_csv(in_dir / f"{name}.csv").sort_values("row").reset_index(drop=True)
                rows, iids = test.row.values, test.image_id.values
                assert (meta.scanner.values[rows] == fold).all() and (meta.image_id.values[rows] == iids).all()
                half = target_halves(iids)                                 # True = half A (candidate case ids)
                cases_a, cases_b = np.unique(iids[half]), np.unique(iids[~half])
                in_fold = r_sc == fold
                check_slides(name, r_iid[in_fold], iids)
                crop_a = np.nonzero(in_fold & np.isin(r_iid, cases_a))[0]  # ascending = slide order
                crop_b = np.nonzero(in_fold & np.isin(r_iid, cases_b))[0]
                p, p_tta = np.full(len(test), np.nan), np.full(len(test), np.nan)
                for pred_mask, adapt_idx, adapt_cases in ((half, crop_b, cases_b), (~half, crop_a, cases_a)):
                    # non-empty, exactly the other half's slides (all of them), never the predicted slides
                    assert len(adapt_idx) > 0 and np.array_equal(np.unique(r_iid[adapt_idx]), adapt_cases)
                    assert not np.isin(r_iid[adapt_idx], iids[pred_mask]).any()
                    model = load_model(fold, v, s, device, cfg)
                    adabn(model, R, adapt_idx, device)                     # statistics from the other half's crops
                    p[pred_mask] = predict(model, X, rows[pred_mask], device)
                    p_tta[pred_mask] = predict(model, X, rows[pred_mask], device, tta=True)
                assert np.isfinite(p).all() and np.isfinite(p_tta).all()
                side.write_text(json.dumps({
                    "fold": fold, "tag": v, "seed": s, **prov,
                    "cases_A": cases_a.tolist(), "cases_B": cases_b.tolist(),
                    "crops_adapt_for_A": int(len(crop_b)), "crops_adapt_for_B": int(len(crop_a)),
                    "adaptation": "eval_tta_adabn.adabn on all target-scanner random crops of the other half's "
                                  "slides (slide order, batch 256, cumulative average)",
                    "thresholds": "p_adabnrand at thr_plain, p_adabnrand_tta at thr_tta "
                                  "(results/preds_tta/{run}.json: source val, source BN)",
                    "device": str(device)}, indent=1))
                tmp = out.with_name(out.name + ".tmp")                     # atomic: no partial CSV on a crash
                pd.DataFrame({"row": rows, "p_adabnrand": p, "p_adabnrand_tta": p_tta}).to_csv(tmp, index=False)
                os.replace(tmp, out)
                print(f"{name} done: {len(cases_a)}/{len(cases_b)} cases, {len(crop_a)}/{len(crop_b)} crops (A/B), "
                      f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["baseline", "stain", "adv", "stain_adv", "stain_mix"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", nargs="+", choices=FOLDS, default=FOLDS)
    ap.add_argument("--crops", default=None, help="default data/random_crops_128.npz")
    ap.add_argument("--out-dir", default=None, help="default results/preds_tta")
    a = ap.parse_args()
    main(a.tags, a.seeds, a.folds, a.crops, a.out_dir)
