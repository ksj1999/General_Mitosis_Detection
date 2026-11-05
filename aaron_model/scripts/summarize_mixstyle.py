#!/usr/bin/env python3
"""
MixStyle experiment summary (3 folds x 3 seeds per variant).

For every variant: F1 / precision / recall / AUROC on all 50 cases of the unseen
scanner, averaged over seeds, plus the AUROC of the seed ensemble (mean of the
three seeds' probabilities). Paired comparisons use a two-level bootstrap:
resample slides within each fold, and independently resample each variant's
seeds; F1 uses each run's own validation threshold.

Usage:  python scripts/summarize_mixstyle.py [--variants ...] [--seeds 0 1 2]
Output: results/mixstyle_summary.csv, results/mixstyle_pairwise.csv,
        figures/mixstyle_results.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
FOLDS = ["CS2", "S360", "XR"]
LABEL = {"baseline": "Baseline", "stain": "+Stain", "adv": "+Adv", "stain_adv": "+Stain+Adv",
         "mix": "+MixStyle", "stain_mix": "+Stain+MixStyle",
         "phikon": "Phikon probe", "phikon_stain": "Phikon probe +Stain"}
COLOR = {"baseline": "#9e9e9e", "stain": "#1f77b4", "adv": "#ff7f0e", "stain_adv": "#2ca02c",
         "mix": "#c39bd3", "stain_mix": "#7b2cbf", "phikon": "#f4a259", "phikon_stain": "#d1495b"}
COMPARISONS = [("stain_mix", "stain"), ("mix", "baseline"), ("stain_mix", "baseline"), ("stain_mix", "stain_adv")]
B = 2000


def f1(y, yhat):
    tp = np.sum(y & yhat)
    denom = 2 * tp + np.sum(~y & yhat) + np.sum(y & ~yhat)
    return 2 * tp / denom if denom else 0.0


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def load(fold, v, s):
    d = pd.read_csv(ROOT / "results" / "preds" / f"{fold}_{v}_seed{s}.csv")
    d = d[d.set == "test_out_all"].sort_values("row").reset_index(drop=True)
    r = json.load(open(ROOT / "results" / f"{fold}_{v}_seed{s}.json"))
    return d, r


def main(variants, seeds, prefix="mixstyle", comparisons=None):
    comparisons = comparisons or COMPARISONS
    rng = np.random.default_rng(11)
    rows = []
    boot = {m: {f: {v: np.empty(B) for v in variants} for f in FOLDS} for m in ("auroc", "f1")}
    for fold in FOLDS:
        D, R = {}, {}
        for v in variants:
            for s in seeds:
                D[(v, s)], R[(v, s)] = load(fold, v, s)
        base = D[(variants[0], seeds[0])]
        for k, d in D.items():
            assert (d.row.values == base.row.values).all(), f"row mismatch {fold} {k}"
        y = base.y.values.astype(bool)
        prob = {k: d.p.values for k, d in D.items()}
        hard = {k: prob[k] >= R[k]["threshold"] for k in D}
        for v in variants:
            o = [R[(v, s)]["test_out_all"] for s in seeds]
            ens = roc_auc_score(y, np.mean([prob[(v, s)] for s in seeds], axis=0))
            rows.append({"fold": fold, "variant": v,
                         **{f"{m}_mean": np.mean([x[m] for x in o]) for m in ("f1", "precision", "recall", "auroc")},
                         **{f"{m}_sd": np.std([x[m] for x in o], ddof=1) for m in ("f1", "auroc")},
                         "auroc_ensemble": ens,
                         "in_auroc_mean": np.mean([R[(v, s)]["test_in"]["auroc"] for s in seeds]),
                         "thresholds": " / ".join(f"{R[(v, s)]['threshold']:.2f}" for s in seeds)})
        cases = base.image_id.unique()
        rows_of = {c: np.where(base.image_id.values == c)[0] for c in cases}
        for b in range(B):
            idx = np.concatenate([rows_of[c] for c in rng.choice(cases, size=len(cases), replace=True)])
            yb = y[idx]
            for v in variants:
                draw = rng.choice(seeds, size=len(seeds))
                boot["auroc"][fold][v][b] = np.mean([roc_auc_score(yb, prob[(v, s)][idx]) for s in draw])
                boot["f1"][fold][v][b] = np.mean([f1(yb, hard[(v, s)][idx]) for s in draw])
        print(f"fold {fold} done", flush=True)

    summ = pd.DataFrame(rows)
    mean_rows = summ.groupby("variant", sort=False).mean(numeric_only=True).reset_index()
    mean_rows.insert(0, "fold", "mean")
    summ = pd.concat([summ, mean_rows], ignore_index=True)
    summ.to_csv(ROOT / "results" / f"{prefix}_summary.csv", index=False)

    pair = []
    for m in ("f1", "auroc"):
        for a, bb in comparisons:
            if a not in variants or bb not in variants:
                continue
            per_fold = {f: boot[m][f][a] - boot[m][f][bb] for f in FOLDS}
            per_fold["mean"] = np.mean([per_fold[f] for f in FOLDS], axis=0)
            for f, d in per_fold.items():
                point = summ[(summ.fold == f) & (summ.variant == a)][f"{m}_mean"].iloc[0] - \
                        summ[(summ.fold == f) & (summ.variant == bb)][f"{m}_mean"].iloc[0]
                lo, hi = ci(d)
                pair.append({"metric": m, "comparison": f"{a} - {bb}", "fold": f, "diff": point,
                             "ci_lo": lo, "ci_hi": hi, "significant": lo > 0 or hi < 0,
                             "frac_draws_le_0": float((d <= 0).mean())})
    pw = pd.DataFrame(pair)
    pw.to_csv(ROOT / "results" / f"{prefix}_pairwise.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["fold", "variant", "f1_mean", "f1_sd", "precision_mean", "recall_mean", "auroc_mean", "auroc_sd", "auroc_ensemble", "thresholds"]
    print("\n=== unseen scanner, mean over seeds ===")
    print(summ[cols].round(3).to_string(index=False))
    print("\n=== paired differences (two-level bootstrap, 95% CI) ===")
    print(pw.round(3).to_string(index=False))
    plot(summ, variants, seeds, prefix)


def plot(summ, variants, seeds, prefix="mixstyle"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), dpi=200)
    groups = FOLDS + ["mean"]
    w = 0.8 / len(variants)
    for ax, m, title in [(axs[0], "f1", "F1 on unseen scanner"), (axs[1], "auroc", "AUROC on unseen scanner")]:
        for j, v in enumerate(variants):
            sub = summ[summ.variant == v].set_index("fold").reindex(groups)
            x = np.arange(len(groups)) + (j - (len(variants) - 1) / 2) * w
            ax.bar(x, sub[f"{m}_mean"], w, color=COLOR[v], label=LABEL[v],
                   yerr=sub[f"{m}_sd"].fillna(0), capsize=2, ecolor="#333", error_kw={"lw": 0.8})
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(["held-out\nCS2", "held-out\nS360", "held-out\nXR", "mean of\n3 folds"], fontsize=9)
        lo = summ[f"{m}_mean"].min()
        ax.set_ylim(max(0, lo - 0.08), min(1, summ[f"{m}_mean"].max() + 0.05))
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.5, ncol=len(variants), loc="upper center", frameon=False)
    fig.text(0.5, 0.005, f"Bars: mean over {len(seeds)} seeds; error bars: SD over seeds. All 50 cases of the held-out scanner.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(ROOT / "figures" / f"{prefix}_results.png")
    print(f"\nwrote figures/{prefix}_results.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["baseline", "stain", "stain_adv", "mix", "stain_mix"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--prefix", default="mixstyle")
    ap.add_argument("--comparisons", nargs="+", default=None, help="pairs like stain_mix:stain")
    a = ap.parse_args()
    main(a.variants, a.seeds, a.prefix, [tuple(c.split(":")) for c in a.comparisons] if a.comparisons else None)
