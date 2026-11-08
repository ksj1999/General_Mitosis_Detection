#!/usr/bin/env python3
"""
Figures for the add-on study (reads the CSVs written by summarize_addons.py).

  grid:      heatmap, training configuration (rows) x inference mode (columns), F1 and AUROC panels,
             mean of 3 held-out scanners x seeds; one sequential hue per panel, values printed in the cells.
  ablation:  dot + 95 % CI of the pooled paired difference for each comparison (e.g. all minus all-without-X),
             one panel per metric, zero line for reference.

  python scripts/plot_addons.py grid --summary results/addons_grid_summary.csv \
      --rows baseline:Baseline stain:+Stain adv:+GRL stain_adv:+Stain+GRL \
      --cols plain:"as trained" tta:"+TTA" adabncf:"+AdaBN" adabncf_tta:"+AdaBN+TTA" --out figures/addons_grid.png
  python scripts/plot_addons.py ablation --pairwise results/ablation5_pairwise.csv \
      --comparisons "all@adabncf_tta - all_no_stain@adabncf_tta:Stain aug" ... --out figures/ablation.png
"""
import argparse
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUES)
MARK = "#2a78d6"


def pairs(items):
    out = []
    for it in items:
        k, _, lab = it.partition(":")
        out.append((k, lab or k))
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def grid(a):
    s = pd.read_csv(ROOT / a.summary)
    s = s[s.fold == "mean"]
    rows, cols = pairs(a.rows), pairs(a.cols)
    fig, axs = plt.subplots(1, 2, figsize=(1.6 + 1.25 * len(cols) * 2, 0.9 + 0.48 * len(rows)), dpi=220,
                            facecolor=SURFACE)
    for ax, metric, title in [(axs[0], "f1", "F1"), (axs[1], "auroc", "AUROC")]:
        M = np.full((len(rows), len(cols)), np.nan)
        for i, (t, _) in enumerate(rows):
            for j, (m, _) in enumerate(cols):
                v = s[(s.tag == t) & (s["mode"] == m)][f"{metric}_mean"]
                if len(v):
                    M[i, j] = v.iloc[0]
        lo, hi = np.nanmin(M), np.nanmax(M)
        ax.imshow(M, cmap=CMAP, vmin=lo - 0.15 * (hi - lo), vmax=hi, aspect="auto")
        for i in range(len(rows)):
            for j in range(len(cols)):
                if np.isnan(M[i, j]):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color=INK2)
                    continue
                frac = (M[i, j] - lo) / (hi - lo + 1e-12)
                best = round(M[i, j], 3) == round(hi, 3)
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8.5,
                        color=SURFACE if frac > 0.55 else INK, fontweight="bold" if best else "normal")
        ax.set_xticks(range(len(cols)), [l for _, l in cols])
        ax.set_yticks(range(len(rows)), [l for _, l in rows])
        ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.xaxis.tick_top()
        style(ax)
        ax.set_title(f"{title} on the unseen scanner", fontsize=9.5, color=INK, pad=22)
    W = fig.get_size_inches()[0]
    fig.text(0.5, 0.01, textwrap.fill(a.note, int(W * 19)), ha="center", va="bottom", fontsize=6.8, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ROOT / a.out, facecolor=SURFACE)
    print("wrote", a.out)


MODE_LABEL = {"plain": "as trained", "tta": "+TTA", "adabncf": "+AdaBN", "adabncf_tta": "+AdaBN+TTA"}


def ablation(a):
    p = pd.read_csv(ROOT / a.pairwise)
    p = p[p.fold == "pooled"] if "pooled" in set(p.fold) else p[p.fold == "mean"]
    comps = pairs(a.comparisons)
    # optional Holm table (holm_confirmatory.py): F1 dots of the confirmatory family are filled by Holm-adjusted p
    holm = {}
    if a.holm:
        h = pd.read_csv(ROOT / a.holm)
        holm = {r.comparison: r for r in h.itertuples()}
    # panels: one per metric, or (with --modes) one per inference mode for comparisons given as "tagA vs tagB"
    if a.modes:
        mname = {"f1": "F1", "auroc": "AUROC"}.get(a.metrics[0], a.metrics[0])
        panels = [(a.metrics[0], m, f"Δ {mname}, {MODE_LABEL.get(m, m)}") for m in a.modes]
    else:
        panels = [(m, None, {"f1": "Δ F1", "auroc": "Δ AUROC"}.get(m, m)) for m in a.metrics]
    fig, axs = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 0.9 + 0.42 * len(comps)), dpi=220,
                            facecolor=SURFACE, sharey=True, sharex=bool(a.modes))
    axs = np.atleast_1d(axs)
    xlims = []
    for ax, (metric, mode, title) in zip(axs, panels):
        y = np.arange(len(comps))[::-1]
        xmin, xmax = 0.0, 0.0
        for yi, (c, lab) in zip(y, comps):
            if mode:
                ta, tb = [t.strip() for t in c.split(" vs ")]
                c = f"{ta}@{mode} - {tb}@{mode}"
            r = p[(p.comparison == c) & (p.metric == metric)]
            if not len(r):
                ax.text(0, yi, "n/a", ha="center", va="center", fontsize=7, color=INK2)
                continue
            r = r.iloc[0]
            ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=MARK, lw=2, solid_capstyle="round")
            sig = r.ci_lo > 0 or r.ci_hi < 0
            lab = f" {r['diff']:+.3f}"
            if metric == "f1" and c in holm:
                hr = holm[c]
                sig = bool(hr.significant_holm)
                ph = min(1.0, hr.p_holm)
                lab += f"  (Holm p {'< ' if hr.p_floor else ''}{ph:.3f})"
            ax.plot(r["diff"], yi, "o", ms=6.5, color=MARK if sig else SURFACE, mec=MARK, mew=1.6, zorder=3)
            ax.text(r.ci_hi, yi, lab, fontsize=7, color=INK, va="center", ha="left", zorder=4,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6, alpha=0.85))
            xmax = max(xmax, r.ci_hi); xmin = min(xmin, r.ci_lo)
        ax.axvline(0, color=INK2, lw=0.8)
        span = xmax - xmin
        xlims.append((xmin - 0.06 * span, xmax + (0.75 if holm and metric == "f1" else 0.3) * span))
        ax.set_yticks(y, [lab for _, lab in comps])
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        style(ax)
        ax.set_title(title + "  [95 % CI]", fontsize=9.5, color=INK, pad=8)
    if a.modes:  # shared x: one range for all panels
        lo, hi = min(l for l, _ in xlims), max(h for _, h in xlims)
        axs[0].set_xlim(lo, hi)
    else:
        for ax, lim in zip(axs, xlims):
            ax.set_xlim(*lim)
    W = fig.get_size_inches()[0]
    note = textwrap.fill(a.note, int(W * 19))
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=6.8, color=INK2)
    fig.tight_layout(rect=(0, 0.04 + 0.035 * note.count("\n"), 1, 1))
    fig.savefig(ROOT / a.out, facecolor=SURFACE)
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--summary", required=True)
    g.add_argument("--rows", nargs="+", required=True, help="tag:label")
    g.add_argument("--cols", nargs="+", required=True, help="mode:label")
    g.add_argument("--out", required=True)
    g.add_argument("--note", default="Mean of 3 held-out scanners x seeds. TTA: 8 flips/rotations. AdaBN: BatchNorm "
                   "statistics re-estimated on unlabelled target images, cross-fitted by case. Threshold from source val.")
    b = sub.add_parser("ablation")
    b.add_argument("--pairwise", required=True)
    b.add_argument("--comparisons", nargs="+", required=True, help='"A@m - B@m:label" as in the pairwise CSV')
    b.add_argument("--metrics", nargs="+", default=["f1", "auroc"])
    b.add_argument("--modes", nargs="+", default=None,
                   help='one panel per inference mode; comparisons are then "tagA vs tagB:label", metric = first --metrics')
    b.add_argument("--holm", default=None, help="results/{prefix}_confirmatory.csv: fill F1 dots by Holm p < 0.05")
    b.add_argument("--out", required=True)
    b.add_argument("--note", default="Pooled paired difference over 3 held-out scanners; two-level bootstrap "
                   "(slides within fold, seeds shared by both arms), 95 % percentile CI. Filled dot: CI excludes 0.")
    a = ap.parse_args()
    grid(a) if a.cmd == "grid" else ablation(a)
