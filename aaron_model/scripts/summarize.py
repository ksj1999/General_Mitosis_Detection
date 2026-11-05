#!/usr/bin/env python3
"""
Summarise all leave-one-scanner-out runs.

For every fold (held-out scanner) and variant: AUROC on the in-domain test set
and on the held-out scanner, with 95% confidence intervals from a *case-level*
bootstrap (resample slides, not cells: cells on one slide are correlated).
Differences against the baseline are paired: both variants are scored on the
same resampled slides.

Folds whose runs predate per-candidate prediction saving get point estimates
only (from results/*.json).

Output: results/summary_auroc.csv, results/summary_pairwise.csv, figures/auroc_by_fold.png
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ["baseline", "stain", "adv", "stain_adv"]
LABEL = {"baseline": "Baseline", "stain": "+Stain", "adv": "+Adv", "stain_adv": "+Stain+Adv"}
FOLDS = ["CS2", "S360", "XR"]
B = 2000


def boot_auc(df_by_variant, rng):
    """Case-level bootstrap. df_by_variant: {variant: DataFrame(image_id, y, p)} on identical rows."""
    any_df = next(iter(df_by_variant.values()))
    cases = any_df.image_id.unique()
    rows_of = {c: np.where(any_df.image_id.values == c)[0] for c in cases}
    out = {v: [] for v in df_by_variant}
    for _ in range(B):
        pick = rng.choice(cases, size=len(cases), replace=True)
        idx = np.concatenate([rows_of[c] for c in pick])
        y = any_df.y.values[idx]
        if y.min() == y.max():
            continue
        for v, d in df_by_variant.items():
            out[v].append(roc_auc_score(y, d.p.values[idx]))
    return {v: np.array(a) for v, a in out.items()}


COMPARISONS = [("stain", "baseline"), ("adv", "baseline"), ("stain_adv", "baseline"), ("stain_adv", "stain")]


def pairwise(seed=0, setname="test_out_all"):
    """Paired AUROC differences per fold and pooled over folds (mean of the three fold differences).

    Cases are resampled independently within each fold (the folds test different scanners, so their
    slides are disjoint); the b-th resample of each fold is averaged to give the b-th pooled value.
    """
    rng = np.random.default_rng(1)
    draws = {c: {} for c in COMPARISONS}
    point = {c: {} for c in COMPARISONS}
    for fold in FOLDS:
        d = {}
        for v in VARIANTS:
            x = pd.read_csv(ROOT / "results" / "preds" / f"{fold}_{v}_seed{seed}.csv")
            d[v] = x[x.set == setname].sort_values("row").reset_index(drop=True)
        base = d["baseline"]
        for v in VARIANTS:
            assert (d[v].row.values == base.row.values).all(), "variants must be scored on identical rows"
        cases = base.image_id.unique()
        rows_of = {c: np.where(base.image_id.values == c)[0] for c in cases}
        auc_full = {v: roc_auc_score(d[v].y, d[v].p) for v in VARIANTS}
        for a, b in COMPARISONS:
            point[(a, b)][fold] = auc_full[a] - auc_full[b]
            draws[(a, b)][fold] = []
        for _ in range(B):
            idx = np.concatenate([rows_of[c] for c in rng.choice(cases, size=len(cases), replace=True)])
            y = base.y.values[idx]
            aucs = {v: roc_auc_score(y, d[v].p.values[idx]) for v in VARIANTS}
            for a, b in COMPARISONS:
                draws[(a, b)][fold].append(aucs[a] - aucs[b])
    rows = []
    for c in COMPARISONS:
        name = f"{c[0]} - {c[1]}"
        for fold in FOLDS:
            arr = np.array(draws[c][fold])
            rows.append({"comparison": name, "fold": fold, "diff": point[c][fold],
                         "ci_lo": np.percentile(arr, 2.5), "ci_hi": np.percentile(arr, 97.5),
                         "frac_draws_le_0": float((arr <= 0).mean())})
        pooled = np.mean([np.array(draws[c][f]) for f in FOLDS], axis=0)
        rows.append({"comparison": name, "fold": "mean", "diff": float(np.mean(list(point[c].values()))),
                     "ci_lo": np.percentile(pooled, 2.5), "ci_hi": np.percentile(pooled, 97.5),
                     "frac_draws_le_0": float((pooled <= 0).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "summary_pairwise.csv", index=False)
    print("\n=== paired differences on the held-out scanner (all 50 cases) ===")
    print(out.round(4).to_string(index=False))
    return out


def main():
    rng = np.random.default_rng(0)
    rows = []
    for fold in FOLDS:
        for setname in ("test_in", "test_out", "test_out_all"):
            preds = {}
            for v in VARIANTS:
                f = ROOT / "results" / "preds" / f"{fold}_{v}_seed0.csv"
                if f.exists():
                    d = pd.read_csv(f)
                    preds[v] = d[d.set == setname].sort_values("row").reset_index(drop=True)
            if len(preds) == len(VARIANTS):
                bs = boot_auc(preds, rng)
                for v in VARIANTS:
                    point = roc_auc_score(preds[v].y, preds[v].p)
                    diff = bs[v] - bs["baseline"]
                    rows.append({"fold": fold, "set": setname, "variant": v, "auroc": point,
                                 "ci_lo": np.percentile(bs[v], 2.5), "ci_hi": np.percentile(bs[v], 97.5),
                                 "diff_vs_base": point - roc_auc_score(preds["baseline"].y, preds["baseline"].p),
                                 "diff_lo": np.percentile(diff, 2.5), "diff_hi": np.percentile(diff, 97.5),
                                 "n_cases": preds[v].image_id.nunique(), "n": len(preds[v])})
            else:  # no saved predictions: point estimate from the json
                for v in VARIANTS:
                    f = ROOT / "results" / f"{fold}_{v}_seed0.json"
                    if f.exists():
                        r = json.load(open(f))
                        rows.append({"fold": fold, "set": setname, "variant": v, "auroc": r[setname]["auroc"],
                                     "n": r[setname]["n"]})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "summary_auroc.csv", index=False)
    pd.set_option("display.width", 220)
    for setname in ("test_in", "test_out", "test_out_all"):
        print(f"\n=== {setname} ===")
        print(df[df.set == setname].drop(columns="set").round(3).to_string(index=False))
    plot(df)
    pairwise()


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[df.set == "test_out_all"]
    colors = {"baseline": "#7f7f7f", "stain": "#1f77b4", "adv": "#ff7f0e", "stain_adv": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=200)
    for i, fold in enumerate(FOLDS):
        for j, v in enumerate(VARIANTS):
            r = d[(d.fold == fold) & (d.variant == v)]
            if r.empty:
                continue
            x = i + (j - 1.5) * 0.17
            y = r.auroc.values[0]
            if "ci_lo" in r and not np.isnan(r.ci_lo.values[0]):
                ax.errorbar(x, y, yerr=[[y - r.ci_lo.values[0]], [r.ci_hi.values[0] - y]],
                            fmt="o", color=colors[v], capsize=3, ms=5)
            else:
                ax.plot(x, y, "o", color=colors[v], ms=5, mfc="white")
    for v in VARIANTS:
        ax.plot([], [], "o", color=colors[v], label=LABEL[v])
    ax.set_xticks(range(len(FOLDS)))
    ax.set_xticklabels([f"held-out: {f}" for f in FOLDS])
    ax.set_ylabel("AUROC on unseen scanner")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.text(0.01, 0.01, "Mitosis vs imposter, all 50 cases of the held-out scanner. "
             "Bars: 95% case-level bootstrap CI.", fontsize=6.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "auroc_by_fold.png")
    print("\nwrote figures/auroc_by_fold.png")


if __name__ == "__main__":
    main()
