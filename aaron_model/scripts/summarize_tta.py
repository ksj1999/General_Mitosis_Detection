#!/usr/bin/env python3
"""
Summarise eval_tta_adabn.py: F1 / AUROC on the unseen scanner for each variant
under plain, tta, adabn and adabn_tta inference, and paired differences
(mode - plain, same checkpoints) with a two-level bootstrap: slides resampled
within each fold, seeds resampled once per iteration and shared by both modes.

Usage:  python scripts/summarize_tta.py [--variants ...] [--seeds 0 1 2]
Output: results/tta_summary.csv, results/tta_pairwise.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
FOLDS = ["CS2", "S360", "XR"]
MODES = ["plain", "tta", "adabn", "adabn_tta", "adabncf", "adabncf_tta"]   # *cf: cross-fitted AdaBN (eval_adabn_crossfit.py)
THR_OF = {"plain": "thr_plain", "tta": "thr_tta", "adabn": "thr_plain", "adabn_tta": "thr_tta",
          "adabncf": "thr_plain", "adabncf_tta": "thr_tta"}
B = 2000


def f1(y, yhat):
    tp = np.sum(y & yhat)
    denom = 2 * tp + np.sum(~y & yhat) + np.sum(y & ~yhat)
    return 2 * tp / denom if denom else 0.0


def prf(y, yhat):
    tp = np.sum(y & yhat)
    return tp / max(yhat.sum(), 1), tp / max(y.sum(), 1)


def main(variants, seeds):
    rng = np.random.default_rng(5)
    rows, boot = [], {}
    for fold in FOLDS:
        D, T = {}, {}
        for v in variants:
            for s in seeds:
                tag = f"{fold}_{v}_seed{s}"
                d = pd.read_csv(ROOT / "results" / "preds_tta" / f"{tag}.csv")
                d = d.merge(pd.read_csv(ROOT / "results" / "preds_tta" / f"{tag}_cf.csv"), on="row", validate="one_to_one")
                D[(v, s)] = d.sort_values("row").reset_index(drop=True)
                T[(v, s)] = json.loads((ROOT / "results" / "preds_tta" / f"{tag}.json").read_text())
        base = D[(variants[0], seeds[0])]
        y = base.y.values.astype(bool)
        prob = {(v, s, m): D[(v, s)][f"p_{m}"].values for v in variants for s in seeds for m in MODES}
        hard = {(v, s, m): prob[(v, s, m)] >= T[(v, s)][THR_OF[m]] for v in variants for s in seeds for m in MODES}
        for v in variants:
            for m in MODES:
                aucs = [roc_auc_score(y, prob[(v, s, m)]) for s in seeds]
                f1s = [f1(y, hard[(v, s, m)]) for s in seeds]
                pr = np.mean([prf(y, hard[(v, s, m)]) for s in seeds], axis=0)
                ens = roc_auc_score(y, np.mean([prob[(v, s, m)] for s in seeds], axis=0))
                rows.append({"fold": fold, "variant": v, "mode": m, "f1_mean": np.mean(f1s), "f1_sd": np.std(f1s, ddof=1),
                             "precision_mean": pr[0], "recall_mean": pr[1],
                             "auroc_mean": np.mean(aucs), "auroc_sd": np.std(aucs, ddof=1), "auroc_ensemble": ens})
        cases = base.image_id.unique()
        rows_of = {c: np.where(base.image_id.values == c)[0] for c in cases}
        for v in variants:
            for m in MODES:
                boot[(fold, v, m, "auroc")] = np.empty(B)
                boot[(fold, v, m, "f1")] = np.empty(B)
        for b in range(B):
            idx = np.concatenate([rows_of[c] for c in rng.choice(cases, size=len(cases), replace=True)])
            yb = y[idx]
            for v in variants:
                draw = rng.choice(seeds, size=len(seeds))             # shared by all modes: same checkpoints
                for m in MODES:
                    boot[(fold, v, m, "auroc")][b] = np.mean([roc_auc_score(yb, prob[(v, s, m)][idx]) for s in draw])
                    boot[(fold, v, m, "f1")][b] = np.mean([f1(yb, hard[(v, s, m)][idx]) for s in draw])
        print(f"fold {fold} done", flush=True)

    summ = pd.DataFrame(rows)
    mean_rows = summ.groupby(["variant", "mode"], sort=False).mean(numeric_only=True).reset_index()
    mean_rows.insert(0, "fold", "mean")
    summ = pd.concat([summ, mean_rows], ignore_index=True)
    summ.to_csv(ROOT / "results" / "tta_summary.csv", index=False)

    pair = []
    for metric in ("f1", "auroc"):
        for v in variants:
            for m in MODES[1:]:
                per = {f: boot[(f, v, m, metric)] - boot[(f, v, "plain", metric)] for f in FOLDS}
                per["mean"] = np.mean([per[f] for f in FOLDS], axis=0)
                for f, d in per.items():
                    pt = summ[(summ.fold == f) & (summ.variant == v) & (summ["mode"] == m)][f"{metric}_mean"].iloc[0] - \
                         summ[(summ.fold == f) & (summ.variant == v) & (summ["mode"] == "plain")][f"{metric}_mean"].iloc[0]
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    pair.append({"metric": metric, "variant": v, "comparison": f"{m} - plain", "fold": f, "diff": pt,
                                 "ci_lo": lo, "ci_hi": hi, "significant": lo > 0 or hi < 0})
    pw = pd.DataFrame(pair)
    pw.to_csv(ROOT / "results" / "tta_pairwise.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== mean of 3 folds (each: mean over seeds) ===")
    print(summ[summ.fold == "mean"].drop(columns="fold").round(3).to_string(index=False))
    print("\n=== mode - plain, mean of 3 folds (two-level bootstrap 95% CI) ===")
    print(pw[pw.fold == "mean"].round(3).to_string(index=False))
    print("\n=== per fold, F1 ===")
    print(pw[(pw.fold != "mean") & (pw.metric == "f1")].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["baseline", "stain", "stain_adv", "stain_mix"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    main(a.variants, a.seeds)
