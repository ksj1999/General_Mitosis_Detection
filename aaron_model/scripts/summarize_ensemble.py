#!/usr/bin/env python3
"""
Seed ensembles on the unseen scanner: average the probabilities of the three seeds'
models (optionally with TTA / AdaBN, from results/preds_tta/) and threshold at the
mean of the three models' own validation thresholds -- no target labels used.

Compared with the average single model of the same variant and inference mode,
paired on the same slides (case-level bootstrap; the seeds are fixed because the
ensemble uses all of them).

Usage:  python scripts/summarize_ensemble.py [--variants ...] [--modes plain tta ...]
Output: results/ensemble_summary.csv, results/ensemble_pairwise.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
FOLDS = ["CS2", "S360", "XR"]
THR_OF = {"plain": "thr_plain", "tta": "thr_tta", "adabn": "thr_plain", "adabn_tta": "thr_tta",
          "adabncf": "thr_plain", "adabncf_tta": "thr_tta"}
B = 2000


def f1(y, yhat):
    tp = np.sum(y & yhat)
    denom = 2 * tp + np.sum(~y & yhat) + np.sum(y & ~yhat)
    return 2 * tp / denom if denom else 0.0


def main(variants, modes, seeds):
    rng = np.random.default_rng(9)
    rows, boot = [], {}
    for fold in FOLDS:
        base = None
        P, T = {}, {}
        for v in variants:
            for s in seeds:
                tag = f"{fold}_{v}_seed{s}"
                d = pd.read_csv(ROOT / "results" / "preds_tta" / f"{tag}.csv")
                cf = ROOT / "results" / "preds_tta" / f"{tag}_cf.csv"
                if cf.exists():
                    d = d.merge(pd.read_csv(cf), on="row", validate="one_to_one")
                d = d.sort_values("row").reset_index(drop=True)
                base = d if base is None else base
                P[(v, s)] = d
                T[(v, s)] = json.loads((ROOT / "results" / "preds_tta" / f"{tag}.json").read_text())
        y = base.y.values.astype(bool)
        cases = base.image_id.unique()
        rows_of = {c: np.where(base.image_id.values == c)[0] for c in cases}
        draws = [np.concatenate([rows_of[c] for c in rng.choice(cases, size=len(cases), replace=True)]) for _ in range(B)]
        for v in variants:
            for m in modes:
                probs = [P[(v, s)][f"p_{m}"].values for s in seeds]
                thrs = [T[(v, s)][THR_OF[m]] for s in seeds]
                pe, te = np.mean(probs, axis=0), float(np.mean(thrs))
                he = pe >= te
                hs = [p >= t for p, t in zip(probs, thrs)]
                tp = np.sum(y & he)
                rows.append({"fold": fold, "variant": v, "mode": m,
                             "ens_f1": f1(y, he), "ens_precision": tp / max(he.sum(), 1), "ens_recall": tp / y.sum(),
                             "ens_auroc": roc_auc_score(y, pe), "ens_threshold": te,
                             "single_f1": np.mean([f1(y, h) for h in hs]),
                             "single_auroc": np.mean([roc_auc_score(y, p) for p in probs])})
                for metric in ("f1", "auroc"):
                    arr = np.empty(B)
                    for b, idx in enumerate(draws):
                        if metric == "f1":
                            arr[b] = f1(y[idx], he[idx]) - np.mean([f1(y[idx], h[idx]) for h in hs])
                        else:
                            arr[b] = roc_auc_score(y[idx], pe[idx]) - np.mean([roc_auc_score(y[idx], p[idx]) for p in probs])
                    boot[(fold, v, m, metric)] = arr
        print(f"fold {fold} done", flush=True)

    summ = pd.DataFrame(rows)
    mean_rows = summ.groupby(["variant", "mode"], sort=False).mean(numeric_only=True).reset_index()
    mean_rows.insert(0, "fold", "mean")
    summ = pd.concat([summ, mean_rows], ignore_index=True)
    summ.to_csv(ROOT / "results" / "ensemble_summary.csv", index=False)
    pair = []
    for metric in ("f1", "auroc"):
        for v in variants:
            for m in modes:
                per = {f: boot[(f, v, m, metric)] for f in FOLDS}
                per["mean"] = np.mean([per[f] for f in FOLDS], axis=0)
                for f, d in per.items():
                    r = summ[(summ.fold == f) & (summ.variant == v) & (summ["mode"] == m)].iloc[0]
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    pair.append({"metric": metric, "variant": v, "mode": m, "fold": f,
                                 "diff_ens_minus_single": r[f"ens_{metric}"] - r[f"single_{metric}"],
                                 "ci_lo": lo, "ci_hi": hi, "significant": lo > 0 or hi < 0})
    pw = pd.DataFrame(pair)
    pw.to_csv(ROOT / "results" / "ensemble_pairwise.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== seed ensemble vs average single model, mean of 3 folds ===")
    print(summ[summ.fold == "mean"].drop(columns="fold").round(3).to_string(index=False))
    print("\n=== ensemble - single (case-level bootstrap 95% CI), mean of 3 folds ===")
    print(pw[pw.fold == "mean"].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["baseline", "stain", "stain_adv", "stain_mix"])
    ap.add_argument("--modes", nargs="+", default=["plain", "tta"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    main(a.variants, a.modes, a.seeds)
