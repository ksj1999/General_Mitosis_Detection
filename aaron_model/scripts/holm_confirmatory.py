#!/usr/bin/env python3
"""
Holm-adjusted confirmatory family of ADDONS_PROTOCOL.md (amendment 2026-11-08).

Reads the pooled ("mean" fold) rows of a summarize_addons.py pairwise CSV. Two-sided bootstrap p = 2 min(P, 1-P),
P = share of bootstrap draws <= 0; a p of 0 is reported as < 2/B. Writes {prefix}_confirmatory.csv and .md.

  python scripts/holm_confirmatory.py --pairwise results/ablation5_pairwise.csv --B 5000
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FAMILY = [
    ("all@adabncf_tta - all_no_stain@adabncf_tta", "Stain augmentation (all vs all minus stain)"),
    ("all@adabncf_tta - all_no_grl@adabncf_tta", "GRL scanner head (all vs all minus GRL)"),
    ("all@adabncf_tta - all_no_soft@adabncf_tta", "Soft labels (all vs all minus soft labels)"),
    ("all@adabncf_tta - all@adabncf", "TTA (all vs all without TTA)"),
    ("all@adabncf_tta - all@tta", "Cross-fitted AdaBN (all vs all without AdaBN)"),
    ("all@adabncf_tta - baseline@plain", "Headline: ResNet all vs baseline as trained"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--pairwise", default="results/ablation5_pairwise.csv")
ap.add_argument("--B", type=int, required=True, help="bootstrap iterations used for the pairwise CSV")
ap.add_argument("--metric", default="f1")
a = ap.parse_args()

p = pd.read_csv(ROOT / a.pairwise)
p = p[p.fold == "mean"]
rows = []
for comp, label in FAMILY:
    r = p[(p.comparison == comp) & (p.metric == a.metric)]
    if len(r) != 1:
        raise SystemExit(f"missing comparison in {a.pairwise}: {comp}")
    r = r.iloc[0]
    au = p[(p.comparison == comp) & (p.metric == "auroc")].iloc[0]
    P = float(r.frac_draws_le_0)
    rows.append({"component": label, "comparison": comp, "diff": r["diff"], "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
                 "p_raw": max(2 * min(P, 1 - P), 2 / a.B), "p_floor": min(P, 1 - P) == 0,
                 "auroc_diff": au["diff"], "auroc_ci_lo": au.ci_lo, "auroc_ci_hi": au.ci_hi})
d = pd.DataFrame(rows)
order = d.p_raw.sort_values().index
m, run_max = len(d), 0.0
for k, i in enumerate(order):
    run_max = max(run_max, min(1.0, (m - k) * d.at[i, "p_raw"]))
    d.at[i, "p_holm"] = run_max
d["significant_holm"] = d.p_holm < 0.05

prefix = Path(a.pairwise).name.replace("_pairwise.csv", "")
d.to_csv(ROOT / "results" / f"{prefix}_confirmatory.csv", index=False)


def pfmt(x, floor):
    """floor: the raw p hit the bootstrap resolution, so x is an upper bound"""
    return f"< {x:.4f}" if floor else f"{x:.4f}"


lines = [f"# Confirmatory family ({a.metric.upper()}, Holm over {m}; bootstrap B = {a.B})", "",
         "| component | ΔF1 [95 % CI] | p (raw) | p (Holm) | ΔAUROC [95 % CI] |", "|---|---|---|---|---|"]
for _, r in d.iterrows():
    star = "**" if r.significant_holm else ""
    lines.append(f"| {r.component} | {star}{r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]{star} | "
                 f"{pfmt(r.p_raw, r.p_floor)} | {pfmt(min(1, r.p_holm), r.p_floor)} | "
                 f"{r.auroc_diff:+.3f} [{r.auroc_ci_lo:+.3f}, {r.auroc_ci_hi:+.3f}] |")
lines += ["", "Bold: Holm-adjusted p < 0.05. p = 2 min(P, 1 - P), P = share of bootstrap draws <= 0."]
(ROOT / "results" / f"{prefix}_confirmatory.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
