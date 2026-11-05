#!/usr/bin/env python3
"""Figure for the TTA / AdaBN results (reads results/tta_summary.csv; mean of 3 folds, seed SD as error bars)."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = [("baseline", "Baseline"), ("stain", "+Stain"), ("stain_adv", "+Stain+Adv"), ("stain_mix", "+Stain+MixStyle")]
MODES = [("plain", "as trained", "#bdbdbd"), ("tta", "+ TTA", "#9ecae1"),
         ("adabncf", "+ AdaBN", "#fdae6b"), ("adabncf_tta", "+ AdaBN + TTA", "#e6550d")]

s = pd.read_csv(ROOT / "results" / "tta_summary.csv")
s = s[s.fold == "mean"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), dpi=200)
w = 0.2
for ax, metric, title, ylim in [(axs[0], "f1", "F1 on unseen scanner", (0.6, 0.8)),
                                 (axs[1], "auroc", "AUROC on unseen scanner", (0.8, 0.92))]:
    for j, (m, lab, col) in enumerate(MODES):
        vals = [s[(s.variant == v) & (s["mode"] == m)][f"{metric}_mean"].iloc[0] for v, _ in VARIANTS]
        sds = [s[(s.variant == v) & (s["mode"] == m)][f"{metric}_sd"].iloc[0] for v, _ in VARIANTS]
        x = np.arange(len(VARIANTS)) + (j - 1.5) * w
        ax.bar(x, vals, w, color=col, label=lab, yerr=sds, capsize=2, ecolor="#444", error_kw={"lw": 0.8})
        if m in ("plain", "adabncf_tta"):
            for xi, v, sd in zip(x, vals, sds):
                ax.text(xi, v + sd + (0.003 if metric == "f1" else 0.0015), f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                        fontweight="bold" if m != "plain" else "normal")
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels([l for _, l in VARIANTS], fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="upper center", frameon=False, fontsize=9)
fig.text(0.5, 0.005, "Mean of 3 held-out scanners x 3 seeds; error bars: SD over seeds. TTA: average over 8 flips/rotations. "
         "AdaBN: BatchNorm statistics re-estimated on unlabelled images of the unseen scanner (cross-fitted by case).",
         ha="center", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 0.93))
fig.savefig(ROOT / "figures" / "tta_adabn_results.png")
print("wrote figures/tta_adabn_results.png")
