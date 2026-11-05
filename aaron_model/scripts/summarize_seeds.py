#!/usr/bin/env python3
"""
Multi-seed summary of the leave-one-scanner-out runs.

Two sources of uncertainty are combined in a two-level bootstrap:
  1. which slides happen to be in the test set -> resample slides within each fold;
  2. training randomness (initialisation, batch order, augmentation draws and the
     source validation split all depend on the seed) -> for each variant,
     independently resample its seeds with replacement.
In each of B iterations a variant's AUROC is the mean over its drawn seeds,
scored on the resampled slides. Differences between variants, and the mean over
the three folds, are taken within the same iteration.

We also report, per fold, in how many seeds a comparison is significant on its
own (case-level bootstrap with the seed fixed, paired by seed).

Usage:  python scripts/summarize_seeds.py [--seeds 0 1 2]
Output: results/seeds_auroc.csv, results/seeds_pairwise.csv, results/seeds_averages.csv,
        figures/auroc_by_fold_seeds.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ["baseline", "stain", "adv", "stain_adv"]
LABEL = {"baseline": "Baseline", "stain": "+Stain", "adv": "+Adv", "stain_adv": "+Stain+Adv"}
FOLDS = ["CS2", "S360", "XR"]
COMPARISONS = [("stain", "baseline"), ("adv", "baseline"), ("stain_adv", "baseline"), ("stain_adv", "stain")]
B = 2000


def load(fold, v, s, setname="test_out_all"):
    d = pd.read_csv(ROOT / "results" / "preds" / f"{fold}_{v}_seed{s}.csv")
    return d[d.set == setname].sort_values("row").reset_index(drop=True)


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main(seeds):
    rng = np.random.default_rng(7)
    var_rows, pair_rows = [], []
    boot_var = {f: {v: np.empty(B) for v in VARIANTS} for f in FOLDS}      # two-level draws of each variant
    for fold in FOLDS:
        P = {(v, s): load(fold, v, s) for v in VARIANTS for s in seeds}
        base = P[("baseline", seeds[0])]
        for k, d in P.items():
            assert (d.row.values == base.row.values).all(), f"row mismatch {fold} {k}"
        y_all = base.y.values
        prob = {k: d.p.values for k, d in P.items()}
        cases = base.image_id.unique()
        rows_of = {c: np.where(base.image_id.values == c)[0] for c in cases}
        point = {k: roc_auc_score(y_all, prob[k]) for k in P}
        per_seed = {(c, s): np.empty(B) for c in COMPARISONS for s in seeds}
        for b in range(B):
            idx = np.concatenate([rows_of[c] for c in rng.choice(cases, size=len(cases), replace=True)])
            y = y_all[idx]
            auc = {k: roc_auc_score(y, prob[k][idx]) for k in P}
            for v in VARIANTS:
                boot_var[fold][v][b] = np.mean([auc[(v, s)] for s in rng.choice(seeds, size=len(seeds))])
            for (a, bb) in COMPARISONS:
                for s in seeds:
                    per_seed[((a, bb), s)][b] = auc[(a, s)] - auc[(bb, s)]
        for v in VARIANTS:
            vals = [point[(v, s)] for s in seeds]
            lo, hi = ci(boot_var[fold][v])
            var_rows.append({"fold": fold, "variant": v, "mean": np.mean(vals),
                             "sd_seeds": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
                             "ci_lo": lo, "ci_hi": hi, **{f"seed{s}": point[(v, s)] for s in seeds}})
        for (a, bb) in COMPARISONS:
            d = boot_var[fold][a] - boot_var[fold][bb]
            lo, hi = ci(d)
            n_sig = sum((ci(per_seed[((a, bb), s)])[0] > 0) or (ci(per_seed[((a, bb), s)])[1] < 0) for s in seeds)
            diffs = [point[(a, s)] - point[(bb, s)] for s in seeds]
            pair_rows.append({"comparison": f"{a} - {bb}", "fold": fold, "diff": np.mean(diffs),
                              "ci_lo": lo, "ci_hi": hi, "frac_draws_le_0": float((d <= 0).mean()),
                              "seeds_significant": f"{n_sig}/{len(seeds)}",
                              **{f"seed{s}": diffs[i] for i, s in enumerate(seeds)}})
        print(f"fold {fold} done", flush=True)

    # mean over folds
    for v in VARIANTS:
        m = np.mean([boot_var[f][v] for f in FOLDS], axis=0)
        lo, hi = ci(m)
        sub = [r for r in var_rows if r["variant"] == v]
        per_seed_mean = [np.mean([r[f"seed{s}"] for r in sub]) for s in seeds]
        var_rows.append({"fold": "mean", "variant": v, "mean": np.mean([r["mean"] for r in sub]),
                         "sd_seeds": np.std(per_seed_mean, ddof=1) if len(seeds) > 1 else np.nan,
                         "ci_lo": lo, "ci_hi": hi, **{f"seed{s}": per_seed_mean[i] for i, s in enumerate(seeds)}})
    for (a, bb) in COMPARISONS:
        d = np.mean([boot_var[f][a] - boot_var[f][bb] for f in FOLDS], axis=0)
        lo, hi = ci(d)
        sub = [r for r in pair_rows if r["comparison"] == f"{a} - {bb}"]
        n_sig = sum(int(r["seeds_significant"].split("/")[0]) for r in sub)
        pair_rows.append({"comparison": f"{a} - {bb}", "fold": "mean", "diff": np.mean([r["diff"] for r in sub]),
                          "ci_lo": lo, "ci_hi": hi, "frac_draws_le_0": float((d <= 0).mean()),
                          "seeds_significant": f"{n_sig}/{len(seeds) * len(FOLDS)} fold-seed pairs",
                          **{f"seed{s}": np.mean([r[f"seed{s}"] for r in sub]) for s in seeds}})

    va, pa = pd.DataFrame(var_rows), pd.DataFrame(pair_rows)
    va.to_csv(ROOT / "results" / "seeds_auroc.csv", index=False)
    pa.to_csv(ROOT / "results" / "seeds_pairwise.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== AUROC on the unseen scanner: mean over seeds, SD over seeds, two-level 95% CI ===")
    print(va.round(4).to_string(index=False))
    print("\n=== paired differences (two-level CI) ===")
    print(pa.round(4).to_string(index=False))

    # averages over folds and seeds from the per-run json files
    rows = []
    for f in FOLDS:
        for v in VARIANTS:
            for s in seeds:
                r = json.load(open(ROOT / "results" / f"{f}_{v}_seed{s}.json"))
                o = r["test_out_all"]
                rows.append({"variant": v, "fold": f, "seed": s, "threshold": r["threshold"],
                             "in_auroc": r["test_in"]["auroc"], "out_auroc": o["auroc"], "out_f1": o["f1"],
                             "out_precision": o["precision"], "out_recall": o["recall"],
                             "recall_unanimous": o["recall_unanimous"], "recall_disputed": o["recall_disputed"]})
    runs = pd.DataFrame(rows)
    avg = runs.groupby("variant").mean(numeric_only=True).drop(columns=["seed"]).reindex(VARIANTS)
    avg["threshold_min"] = runs.groupby("variant").threshold.min().reindex(VARIANTS)
    avg["threshold_max"] = runs.groupby("variant").threshold.max().reindex(VARIANTS)
    avg.to_csv(ROOT / "results" / "seeds_averages.csv")
    print("\n=== averages over folds and seeds ===")
    print(avg.round(3).to_string())
    plot(va, seeds)


def plot(va, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"baseline": "#7f7f7f", "stain": "#1f77b4", "adv": "#ff7f0e", "stain_adv": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=200)
    for i, fold in enumerate(FOLDS):
        for j, v in enumerate(VARIANTS):
            r = va[(va.fold == fold) & (va.variant == v)].iloc[0]
            x = i + (j - 1.5) * 0.18
            ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r.ci_lo], [r.ci_hi - r["mean"]]],
                        fmt="o", color=colors[v], capsize=3, ms=6, zorder=3)
            for k, s in enumerate(seeds):
                ax.plot(x + 0.045 + 0.02 * (k - 1), r[f"seed{s}"], ".", color=colors[v], alpha=0.45, ms=4, zorder=2)
    for v in VARIANTS:
        ax.plot([], [], "o", color=colors[v], label=LABEL[v])
    ax.set_xticks(range(len(FOLDS)))
    ax.set_xticklabels([f"held-out: {f}" for f in FOLDS])
    ax.set_ylabel("AUROC on unseen scanner")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.text(0.01, 0.01, f"Large markers: mean over {len(seeds)} seeds with 95% two-level bootstrap CI "
             "(slides and seeds resampled). Small dots: individual seeds.", fontsize=6.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ROOT / "figures" / "auroc_by_fold_seeds.png")
    print("\nwrote figures/auroc_by_fold_seeds.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    main(ap.parse_args().seeds)
