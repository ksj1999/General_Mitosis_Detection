#!/usr/bin/env python3
"""
F1 version of the seed-0 summary: F1 on the unseen scanner (all 50 cases) at each
run's own validation-chosen threshold, case-level bootstrap CIs, paired differences.

Usage:  python scripts/summarize_f1.py [--seed 0]
Output: results/summary_f1.csv, results/summary_f1_pairwise.csv, figures/f1_by_fold.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ["baseline", "stain", "adv", "stain_adv"]
LABEL = {"baseline": "Baseline", "stain": "+Stain", "adv": "+Adv", "stain_adv": "+Stain+Adv"}
FOLDS = ["CS2", "S360", "XR"]
COMPARISONS = [("stain", "baseline"), ("adv", "baseline"), ("stain_adv", "baseline"), ("stain_adv", "stain")]
B = 2000


def f1(y, yhat):
    tp = np.sum(y & yhat); fp = np.sum(~y & yhat); fn = np.sum(y & ~yhat)
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main(seed):
    rng = np.random.default_rng(1)
    rows, pair_rows = [], []
    draws = {f: {} for f in FOLDS}
    for fold in FOLDS:
        P, thr = {}, {}
        for v in VARIANTS:
            d = pd.read_csv(ROOT / "results" / "preds" / f"{fold}_{v}_seed{seed}.csv")
            P[v] = d[d.set == "test_out_all"].sort_values("row").reset_index(drop=True)
            thr[v] = json.load(open(ROOT / "results" / f"{fold}_{v}_seed{seed}.json"))["threshold"]
        base = P["baseline"]
        for v in VARIANTS:
            assert (P[v].row.values == base.row.values).all()
        y = base.y.values.astype(bool)
        yhat = {v: P[v].p.values >= thr[v] for v in VARIANTS}
        cases = base.image_id.unique()
        rows_of = {c: np.where(base.image_id.values == c)[0] for c in cases}
        bd = {v: np.empty(B) for v in VARIANTS}
        for b in range(B):
            idx = np.concatenate([rows_of[c] for c in rng.choice(cases, size=len(cases), replace=True)])
            for v in VARIANTS:
                bd[v][b] = f1(y[idx], yhat[v][idx])
        draws[fold] = bd
        for v in VARIANTS:
            lo, hi = ci(bd[v])
            rows.append({"fold": fold, "variant": v, "f1": f1(y, yhat[v]), "ci_lo": lo, "ci_hi": hi, "threshold": thr[v]})
        for a, b_ in COMPARISONS:
            d = bd[a] - bd[b_]; lo, hi = ci(d)
            pair_rows.append({"comparison": f"{a} - {b_}", "fold": fold, "diff": f1(y, yhat[a]) - f1(y, yhat[b_]),
                              "ci_lo": lo, "ci_hi": hi, "frac_draws_le_0": float((d <= 0).mean())})
    for v in VARIANTS:
        m = np.mean([draws[f][v] for f in FOLDS], axis=0); lo, hi = ci(m)
        rows.append({"fold": "mean", "variant": v, "f1": np.mean([r["f1"] for r in rows if r["variant"] == v]),
                     "ci_lo": lo, "ci_hi": hi, "threshold": np.nan})
    for a, b_ in COMPARISONS:
        d = np.mean([draws[f][a] - draws[f][b_] for f in FOLDS], axis=0); lo, hi = ci(d)
        sub = [r for r in pair_rows if r["comparison"] == f"{a} - {b_}"]
        pair_rows.append({"comparison": f"{a} - {b_}", "fold": "mean", "diff": np.mean([r["diff"] for r in sub]),
                          "ci_lo": lo, "ci_hi": hi, "frac_draws_le_0": float((d <= 0).mean())})
    df, pw = pd.DataFrame(rows), pd.DataFrame(pair_rows)
    df.to_csv(ROOT / "results" / "summary_f1.csv", index=False)
    pw.to_csv(ROOT / "results" / "summary_f1_pairwise.csv", index=False)
    print(df.round(3).to_string(index=False)); print(pw.round(3).to_string(index=False))
    plot(df)


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"baseline": "#7f7f7f", "stain": "#1f77b4", "adv": "#ff7f0e", "stain_adv": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=200)
    for i, fold in enumerate(FOLDS):
        for j, v in enumerate(VARIANTS):
            r = df[(df.fold == fold) & (df.variant == v)].iloc[0]
            ax.errorbar(i + (j - 1.5) * 0.18, r.f1, yerr=[[r.f1 - r.ci_lo], [r.ci_hi - r.f1]],
                        fmt="o", color=colors[v], capsize=3, ms=5)
    for v in VARIANTS:
        ax.plot([], [], "o", color=colors[v], label=LABEL[v])
    ax.set_xticks(range(len(FOLDS))); ax.set_xticklabels([f"held-out: {f}" for f in FOLDS])
    ax.set_ylabel("F1 on unseen scanner"); ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.text(0.01, 0.01, "Mitosis vs imposter, all 50 cases of the held-out scanner, each run's validation threshold. "
             "Bars: 95% case-level bootstrap CI.", fontsize=6.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ROOT / "figures" / "f1_by_fold.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
