#!/usr/bin/env python3
"""
Macenko et al. (2009) stain normalisation of every candidate patch, stain matrix estimated per slide.

Per slide (the 150 MIDOG++ breast TIFFs), following the reference implementation of the method
(Macenko's normalizeStaining.m / HEnorm_python, same constants):
  1. read the image (prepare_patches.load_rgb) and keep every 4th row and column (~2.3 M pixels)
  2. optical density OD = -ln((I + 1) / Io), Io = 240; tissue pixels = no OD channel below beta = 0.15
  3. eigen-decompose the covariance of the tissue OD, project the tissue OD on the plane of the two largest
     eigenvectors, take the angle phi of every projected pixel; the stain vectors are the plane directions
     at the alpha = 1 and 100 - alpha = 99 percentiles of phi
  4. H first = the vector with the larger first (red) OD component; both unit length
  5. concentrations C = lstsq(HE, OD) over the subsampled pixels; maxC = 99th percentile of each stain's
     concentration over the TISSUE pixels (same mask as step 2).
     Deviation from the reference, which takes the percentile over all pixels: it was written for tissue-filled
     tiles, but the inputs here are whole ROIs with a tissue fraction of 0.13-0.77, and with the all-pixel
     percentile maxC mostly tracks the background share (v1 of this script: corr(maxC_H, tissue_frac) = 0.79)
     instead of stain strength. The all-pixel value is kept in the CSV as a diagnostic (maxC_all_H/E).
Per candidate patch of that slide (rows of data/patches_meta.csv, pixels of data/patches_128.npz):
  C = lstsq(HE_slide, OD_patch);  C *= maxCRef / maxC_slide;  I = Io * exp(-HERef @ C), rounded, clipped to
  [0, 255], uint8.
The target is fixed, not data-derived: HERef = [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]],
maxCRef = [1.9705, 1.0308] (the reference implementation's values).

Validity (check_vectors): finite, all OD components > 0, H-E angle 10-90 deg, eigenvalues 2 and 3 not equal,
and the estimate reproduces on a disjoint pixel grid (offset by step//2 >= 1, hence --step >= 2) within 5 deg per
vector and 10 % in maxC. Patches are only written when every slide passes.

No leakage: a slide's estimate uses only that slide's own image -- no labels, no annotation positions, no
pooling across slides or scanners. It is per-image preprocessing, applied identically to source and target
slides, and could be computed for any single unseen slide at test time.
Reporting note: for a target slide this IS a label-free use of the test image itself (its stain matrix and maxC
come from that slide's own pixels, including the area around the candidates being predicted), in the same sense
as TTA. It is exempt from the protocol's cross-fitting rule because it adapts to one slide only, uses no labels
and does not pool over target cases, so nothing estimated from one target case reaches another. The write-up
should present [macenko]/[macenko_stain] as per-slide normalisation that also runs at test time, not as a
purely training-time change.

Output safety: outputs are written via a temp file + rename; an output path may never be the input
data/patches_128.npz or data/patches_meta.csv; an existing output path other than the three defaults is refused
unless --overwrite; --limit refuses the default paths.

Outputs
  data/patches_128_macenko.npz       X (same row order and shape as patches_128.npz), y, ann_id
  results/macenko_stain_vectors.csv  per slide: H and E OD vectors, maxC, tissue-pixel count, diagnostics
                                     (eigenvalues, H-E angle, angles to the reference, reproducibility on a
                                     disjoint pixel grid, top-vs-bottom-half agreement, valid/issues)
  figures/macenko_check.png          candidates per scanner before/after, per-slide mean colour before/after,
                                     stain vectors and maxC per slide
QA numbers (per-scanner mean colour before/after, variance share explained by scanner, colour-only scanner
classifier) go to stdout.

Usage
  python scripts/prepare_macenko.py                      # full run (reads ~20 GB from the external disk)
  python scripts/prepare_macenko.py --from-csv           # reuse results/macenko_stain_vectors.csv, no TIFF reads
  python scripts/prepare_macenko.py --limit 3 --out-npz /tmp/x.npz --out-csv /tmp/x.csv --out-fig /tmp/x.png
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_patches import ROOT, load_rgb  # noqa: E402

IO, ALPHA, BETA, STEP = 240.0, 1.0, 0.15, 4
HE_REF = np.array([[0.5626, 0.2159],
                   [0.7201, 0.8012],
                   [0.4062, 0.5581]])
MAXC_REF = np.array([1.9705, 1.0308])
MIN_TISSUE = 10_000          # tissue pixels needed for a usable estimate (subsampled grid)
STAB_DEG, STAB_MAXC = 5.0, 0.10   # reproducibility bounds: re-estimate on a disjoint pixel grid
SCANNERS = ["CS2", "S360", "XR"]
DEF_NPZ = ROOT / "data" / "patches_128_macenko.npz"
DEF_CSV = ROOT / "results" / "macenko_stain_vectors.csv"
DEF_FIG = ROOT / "figures" / "macenko_check.png"


def to_od(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 -> (n, 3) float64 optical density."""
    return -np.log((rgb.reshape(-1, 3).astype(np.float64) + 1.0) / IO)


def angle_deg(u, v) -> float:
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def estimate(rgb_sub: np.ndarray) -> dict:
    """Macenko stain matrix (3x2, columns H, E) and maxC from one slide's (subsampled) pixels."""
    od = to_od(rgb_sub)
    tmask = (od >= BETA).all(axis=1)
    tissue = od[tmask]
    if len(tissue) < MIN_TISSUE:
        raise ValueError(f"only {len(tissue)} tissue pixels")
    evals, evecs = np.linalg.eigh(np.cov(tissue.T))            # ascending eigenvalues
    plane = evecs[:, 1:3].copy()                                # second-largest, largest
    # Sign of the largest eigenvector made positive: cosmetic (reconstructed vectors do not depend on the
    # eigenvector signs) but keeps phi in (0, pi), away from the arctan2 wrap at +-pi.
    if plane[:, 1].sum() < 0:
        plane[:, 1] *= -1
    proj = tissue @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, [ALPHA, 100 - ALPHA])
    v_lo = plane @ np.array([np.cos(lo), np.sin(lo)])
    v_hi = plane @ np.array([np.cos(hi), np.sin(hi)])
    h, e = (v_lo, v_hi) if v_lo[0] > v_hi[0] else (v_hi, v_lo)
    h, e = h / np.linalg.norm(h), e / np.linalg.norm(e)
    HE = np.stack([h, e], axis=1)
    C = np.linalg.lstsq(HE, od.T, rcond=None)[0]
    maxC = np.percentile(C[:, tmask], 99, axis=1)               # tissue pixels only (see module docstring)
    maxC_all = np.percentile(C, 99, axis=1)                     # reference behaviour, diagnostic only
    return {"HE": HE, "maxC": maxC, "maxC_all": maxC_all, "n_pixels": len(od), "n_tissue": len(tissue),
            "evals": evals[::-1],
            "phi_lo": float(lo), "phi_hi": float(hi), "frac_proj_neg": float((proj[:, 1] <= 0).mean())}


def normalise(patches: np.ndarray, HE: np.ndarray, maxC: np.ndarray) -> np.ndarray:
    """(n, h, w, 3) uint8 patches of one slide -> normalised to (HE_REF, MAXC_REF), uint8."""
    C = np.linalg.lstsq(HE, to_od(patches).T, rcond=None)[0]
    C *= (MAXC_REF / maxC)[:, None]
    out = IO * np.exp(-HE_REF @ C)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8).T.reshape(patches.shape)


def check_vectors(r: dict) -> list:
    """Reasons a slide's estimate would be unusable; empty list = valid.

    Degeneracy is judged by whether the estimate is reproducible, not by an eigenvalue-ratio cut-off:
    the stain plane is re-estimated on a disjoint pixel grid (offset by step/2 in both axes) and both stain
    vectors must agree within STAB_DEG degrees, maxC within STAB_MAXC (relative)."""
    HE, maxC = r["HE"], r["maxC"]
    bad = []
    if not (np.isfinite(HE).all() and np.isfinite(maxC).all()):
        bad.append("non-finite")
    if (HE <= 0).any():
        bad.append("non-positive OD component")
    if not 10 <= r["angle_HE_deg"] <= 90:
        bad.append(f"H-E angle {r['angle_HE_deg']:.1f} deg")
    if r["evals"][1] < 1.05 * r["evals"][2]:
        bad.append("2nd/3rd eigenvalues numerically equal (stain plane undefined)")
    if max(r["stab_H_deg"], r["stab_E_deg"]) > STAB_DEG:
        bad.append(f"unstable between pixel grids (H {r['stab_H_deg']:.1f}, E {r['stab_E_deg']:.1f} deg)")
    if r["stab_maxC"] > STAB_MAXC:
        bad.append(f"maxC unstable between pixel grids ({r['stab_maxC']:.2f})")
    if (maxC <= 0).any():
        bad.append("maxC <= 0")
    if r["frac_proj_neg"] > 0.01:
        bad.append("projections on the principal axis not one-signed")
    return bad


def row_of(iid, fname, scanner, n_cand, r) -> dict:
    HE = r["HE"]
    return {"image_id": iid, "file_name": fname, "scanner": scanner,
            "H_R": HE[0, 0], "H_G": HE[1, 0], "H_B": HE[2, 0],
            "E_R": HE[0, 1], "E_G": HE[1, 1], "E_B": HE[2, 1],
            "maxC_H": r["maxC"][0], "maxC_E": r["maxC"][1],
            "maxC_all_H": r["maxC_all"][0], "maxC_all_E": r["maxC_all"][1], "step": r["step"],
            "n_tissue_pixels": r["n_tissue"], "n_pixels": r["n_pixels"],
            "tissue_frac": r["n_tissue"] / r["n_pixels"],
            "eig1": r["evals"][0], "eig2": r["evals"][1], "eig3": r["evals"][2],
            "eig2_over_eig3": r["evals"][1] / r["evals"][2],
            "phi_lo": r["phi_lo"], "phi_hi": r["phi_hi"], "frac_proj_neg": r["frac_proj_neg"],
            "stab_H_deg": r["stab_H_deg"], "stab_E_deg": r["stab_E_deg"], "stab_maxC": r["stab_maxC"],
            "half_H_deg": r["half_H_deg"], "half_E_deg": r["half_E_deg"],
            "angle_HE_deg": r["angle_HE_deg"], "angle_H_ref_deg": r["angle_H_ref_deg"],
            "angle_E_ref_deg": r["angle_E_ref_deg"], "n_candidates": n_cand,
            "valid": not r["bad"], "issues": "; ".join(r["bad"])}


def he_from_row(s) -> tuple:
    HE = np.array([[s.H_R, s.E_R], [s.H_G, s.E_G], [s.H_B, s.E_B]], dtype=np.float64)
    return HE, np.array([s.maxC_H, s.maxC_E], dtype=np.float64)


# ------------------------------------------------------------------------------------------- QA
def qa(meta: pd.DataFrame, X: np.ndarray, Xn: np.ndarray, vec: pd.DataFrame, fig_path: Path) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    print("\n=== QA ===")
    print(f"non-finite stain vectors/maxC: {int((~np.isfinite(vec[['H_R','H_G','H_B','E_R','E_G','E_B','maxC_H','maxC_E']].values)).sum())}")
    print(f"slides valid: {int(vec.valid.sum())}/{len(vec)}")
    if (~vec.valid).any():
        print(vec.loc[~vec.valid, ["image_id", "scanner", "issues"]].to_string(index=False))
    cols = ["angle_HE_deg", "angle_H_ref_deg", "angle_E_ref_deg", "maxC_H", "maxC_E", "tissue_frac",
            "eig2_over_eig3", "stab_H_deg", "stab_E_deg", "stab_maxC", "half_H_deg", "half_E_deg"]
    print("\nper-scanner stain-vector diagnostics (mean +- sd over slides):")
    g = vec.groupby("scanner")[cols]
    print((g.mean().round(3).astype(str) + " +- " + g.std().round(3).astype(str)).to_string())
    print(f"disjoint-grid reproducibility, max over slides: H {vec.stab_H_deg.max():.2f} deg, "
          f"E {vec.stab_E_deg.max():.2f} deg, maxC {vec.stab_maxC.max():.3f} (rel.); "
          f"eig2/eig3 min {vec.eig2_over_eig3.min():.2f}")
    print(f"top vs bottom image half (diagnostic, tissue heterogeneity): H median {vec.half_H_deg.median():.2f} "
          f"max {vec.half_H_deg.max():.2f} deg; E median {vec.half_E_deg.median():.2f} max {vec.half_E_deg.max():.2f} deg")
    print(f"all slides: H-E angle {vec.angle_HE_deg.min():.1f}..{vec.angle_HE_deg.max():.1f} deg, "
          f"H vs ref {vec.angle_H_ref_deg.max():.1f} deg max, E vs ref {vec.angle_E_ref_deg.max():.1f} deg max")
    hm = vec[["H_R", "H_G", "H_B"]].mean().values
    em = vec[["E_R", "E_G", "E_B"]].mean().values
    print(f"mean H {np.round(hm / np.linalg.norm(hm), 4)}  ref {HE_REF[:, 0]};  "
          f"mean E {np.round(em / np.linalg.norm(em), 4)}  ref {HE_REF[:, 1]}")

    # maxC: does it track stain strength or background share? (used = tissue pixels; all = reference behaviour)
    has_all = {"maxC_all_H", "maxC_all_E"} <= set(vec.columns)
    kinds = [("used", "maxC_H", "maxC_E")] + ([("all-pixel", "maxC_all_H", "maxC_all_E")] if has_all else [])
    for name, ch, ce in kinds:
        print(f"maxC {name}: corr with tissue_frac H {vec[ch].corr(vec.tissue_frac):.2f} "
              f"E {vec[ce].corr(vec.tissue_frac):.2f}; scale maxCRef/maxC H {(MAXC_REF[0] / vec[ch]).min():.2f}.."
              f"{(MAXC_REF[0] / vec[ch]).max():.2f}, E {(MAXC_REF[1] / vec[ce]).min():.2f}.."
              f"{(MAXC_REF[1] / vec[ce]).max():.2f}")
    # between-slide spread of the 99th-pct concentration over tissue pixels of each slide's candidate patches
    p99 = []
    for s in vec.itertuples():
        od = to_od(X[np.flatnonzero(meta.image_id.values == s.image_id)])
        od = od[(od >= BETA).all(axis=1)]
        p99.append(np.percentile(np.linalg.lstsq(he_from_row(s)[0], od.T, rcond=None)[0], 99, axis=1))
    p99 = np.array(p99)
    cv = lambda v: v.std(0) / v.mean(0)
    tf = vec.tissue_frac.values
    out = [("unscaled", p99)] + [(name, p99 * MAXC_REF / vec[[ch, ce]].values) for name, ch, ce in kinds]
    print("between-slide CV of the candidate-patch tissue 99th-pct concentration (H, E) [corr H with tissue_frac]: "
          + "; ".join(f"{n} {cv(v)[0]:.3f}, {cv(v)[1]:.3f} [{np.corrcoef(v[:, 0], tf)[0, 1]:+.2f}]" for n, v in out))

    # per-patch colour statistics
    def patch_stats(A):
        f = A.reshape(len(A), -1, 3).astype(np.float64)
        return np.concatenate([f.mean(1), f.std(1)], axis=1)        # (n, 6): mean RGB, sd RGB
    S0, S1 = patch_stats(X), patch_stats(Xn)
    sc = meta.scanner.values
    print("\nmean RGB of candidate patches per scanner (all pixels):")
    tab = []
    for s in SCANNERS:
        m = sc == s
        tab.append([s, *np.round(S0[m, :3].mean(0), 1), *np.round(S1[m, :3].mean(0), 1)])
    print(pd.DataFrame(tab, columns=["scanner", "R_before", "G_before", "B_before",
                                     "R_after", "G_after", "B_after"]).to_string(index=False))
    means0 = np.array([S0[sc == s, :3].mean(0) for s in SCANNERS])
    means1 = np.array([S1[sc == s, :3].mean(0) for s in SCANNERS])
    pair = lambda M: max(np.linalg.norm(M[i] - M[j]) for i in range(3) for j in range(i + 1, 3))
    print(f"max pairwise distance between scanner mean RGB: before {pair(means0):.1f}, after {pair(means1):.1f}")
    print(f"sd across scanners of channel means (avg over R,G,B): before {means0.std(0).mean():.2f}, "
          f"after {means1.std(0).mean():.2f}")

    # share of the between-slide variance of slide-mean colour explained by scanner (eta^2)
    df = pd.DataFrame(np.concatenate([S0[:, :3], S1[:, :3]], 1),
                      columns=["R0", "G0", "B0", "R1", "G1", "B1"])
    df["image_id"], df["scanner"] = meta.image_id.values, sc
    sl = df.groupby("image_id").agg({**{c: "mean" for c in df.columns[:6]}, "scanner": "first"})

    def eta2(cols3):
        tot = sum(((sl[c] - sl[c].mean()) ** 2).sum() for c in cols3)
        btw = sum((sl.groupby("scanner")[c].transform("mean") - sl[c].mean()).pow(2).sum() for c in cols3)
        return btw / tot, tot / len(sl)
    e0, v0 = eta2(["R0", "G0", "B0"])
    e1, v1 = eta2(["R1", "G1", "B1"])
    print(f"slide-mean RGB: total variance (sum over channels) before {v0:.1f}, after {v1:.1f}; "
          f"share explained by scanner (eta^2) before {e0:.3f}, after {e1:.3f}")

    # how well can the scanner be told from colour alone (6 colour stats per patch), CV grouped by slide
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = GroupKFold(n_splits=min(5, meta.image_id.nunique()))
    bacc = []
    for S in (S0, S1):
        pr = cross_val_predict(clf, S, sc, groups=meta.image_id.values, cv=cv)
        bacc.append(balanced_accuracy_score(sc, pr))
    print(f"scanner from patch colour stats (mean+sd RGB, logistic, 5-fold by slide): balanced acc "
          f"before {bacc[0]:.3f}, after {bacc[1]:.3f} (chance 0.333)")
    print(f"patches: non-finite n/a (uint8); before range [{X.min()}, {X.max()}], after [{Xn.min()}, {Xn.max()}]; "
          f"after: share of pixels at 255 {np.mean(Xn == 255):.4f}, at 0 {np.mean(Xn == 0):.5f}")

    _figure(meta, X, Xn, vec, sl, fig_path, dict(pair0=pair(means0), pair1=pair(means1), e0=e0, e1=e1,
                                                 b0=bacc[0], b1=bacc[1]))


def _figure(meta, X, Xn, vec, sl, fig_path, st) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ncol = 8
    rng = np.random.RandomState(0)
    colors = {"CS2": "#1b9e77", "S360": "#d95f02", "XR": "#7570b3"}
    fig = plt.figure(figsize=(16, 17))
    gs = GridSpec(6, ncol, figure=fig, left=0.05, right=0.98, top=0.965, bottom=0.31, hspace=0.12, wspace=0.05)
    for k, s in enumerate(SCANNERS):
        m = meta[meta.scanner == s]
        slides = rng.choice(np.sort(m.image_id.unique()), size=min(ncol, m.image_id.nunique()), replace=False)
        rows = [rng.choice(m.index[m.image_id == iid]) for iid in slides]
        for j, r in enumerate(rows):
            for b, A in enumerate((X, Xn)):
                ax = fig.add_subplot(gs[2 * k + b, j])
                ax.imshow(A[r])
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(colors[s]); sp.set_linewidth(2)
                if j == 0:
                    ax.set_ylabel(f"{s}\n{'before' if b == 0 else 'Macenko'}", fontsize=11)
                if b == 0:
                    ax.set_title(f"slide {meta.image_id[r]}", fontsize=8, pad=2)

    sub = GridSpec(1, 4, figure=fig, left=0.05, right=0.98, bottom=0.03, top=0.255, wspace=0.32)
    # (a, b) per-slide mean colour before / after
    for p, (cols, title) in enumerate(((["R0", "G0", "B0"], "per-slide mean RGB, before"),
                                       (["R1", "G1", "B1"], "per-slide mean RGB, after Macenko"))):
        ax = fig.add_subplot(sub[0, p])
        for k, s in enumerate(SCANNERS):
            d = sl[sl.scanner == s]
            for c, col in enumerate(cols):
                x = c + (k - 1) * 0.25 + rng.uniform(-0.07, 0.07, len(d))
                ax.scatter(x, d[col], s=9, color=colors[s], alpha=0.75, label=s if c == 0 else None)
                ax.hlines(d[col].mean(), c + (k - 1) * 0.25 - 0.1, c + (k - 1) * 0.25 + 0.1, color="k", lw=1.5)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["R", "G", "B"])
        ax.set_ylim(60, 245); ax.set_ylabel("mean intensity (0-255)")
        ax.set_title(title, fontsize=10)
        if p == 0:
            ax.legend(fontsize=8, loc="lower left")
    ax.text(0.02, 0.02, f"scanner-mean spread: {st['pair0']:.1f} -> {st['pair1']:.1f}\n"
            f"eta^2(scanner): {st['e0']:.2f} -> {st['e1']:.2f}\n"
            f"scanner bal. acc from colour: {st['b0']:.2f} -> {st['b1']:.2f}",
            transform=ax.transAxes, fontsize=8, va="bottom")
    # (c) stain vectors per slide, projected on (red, green) OD components
    ax = fig.add_subplot(sub[0, 2])
    for s in SCANNERS:
        d = vec[vec.scanner == s]
        ax.scatter(d.H_R, d.H_G, s=12, marker="o", color=colors[s], alpha=0.8, label=f"{s} H")
        ax.scatter(d.E_R, d.E_G, s=14, marker="^", color=colors[s], alpha=0.8, label=f"{s} E")
    ax.scatter(*HE_REF[:2, 0], s=180, marker="*", color="k", label="ref H")
    ax.scatter(*HE_REF[:2, 1], s=180, marker="*", color="0.5", label="ref E")
    ax.set_xlabel("OD red component"); ax.set_ylabel("OD green component")
    ax.set_title("per-slide stain vectors (unit OD)", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    # (d) maxC per slide
    ax = fig.add_subplot(sub[0, 3])
    for s in SCANNERS:
        d = vec[vec.scanner == s]
        ax.scatter(d.maxC_H, d.maxC_E, s=12, color=colors[s], alpha=0.8, label=s)
    ax.scatter(*MAXC_REF, s=180, marker="*", color="k", label="ref")
    ax.set_xlabel("maxC H (99th pct, tissue px)"); ax.set_ylabel("maxC E (99th pct, tissue px)")
    ax.set_title("per-slide max concentrations", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("Macenko per-slide stain normalisation: random candidates (one per slide) before / after",
                 fontsize=13, y=0.99)
    fig.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {fig_path}")


# ------------------------------------------------------------------------------------------- main
IN_NPZ = ROOT / "data" / "patches_128.npz"
IN_META = ROOT / "data" / "patches_meta.csv"


def check_outputs(a, out_npz: Path, out_csv: Path, out_fig: Path) -> None:
    """Refuse output paths that could clobber inputs or files this script does not own (exits on failure)."""
    if a.step < 2:
        raise SystemExit("--step must be >= 2 (the reproducibility grid is offset by step//2)")
    outs = {"--out-npz": out_npz.resolve(), "--out-fig": out_fig.resolve()}
    if not a.from_csv:                       # with --from-csv the CSV is read, not written
        outs["--out-csv"] = out_csv.resolve()
    elif out_csv.resolve() in outs.values():
        raise SystemExit("--out-csv (read with --from-csv) must differ from --out-npz/--out-fig")
    if len(set(outs.values())) != len(outs):
        raise SystemExit("output paths must be distinct")
    defaults = {DEF_NPZ.resolve(), DEF_CSV.resolve(), DEF_FIG.resolve()}
    for flag, p in outs.items():
        if p in (IN_NPZ.resolve(), IN_META.resolve()):
            raise SystemExit(f"{flag} {p} is an input of this script; refusing to overwrite it")
        if a.limit and p in defaults:
            raise SystemExit("--limit writes partial outputs: pass explicit non-default --out-npz/--out-fig"
                             + ("" if a.from_csv else "/--out-csv"))
        if p.exists() and p not in defaults and not a.overwrite:
            raise SystemExit(f"{flag} {p} already exists and is not a default output; pass --overwrite")


def main(a) -> None:
    out_npz, out_csv, out_fig = Path(a.out_npz), Path(a.out_csv), Path(a.out_fig)
    check_outputs(a, out_npz, out_csv, out_fig)
    meta = pd.read_csv(IN_META)
    d = np.load(IN_NPZ)
    X, y, ann = d["X"], d["y"], d["ann_id"]
    assert len(meta) == len(X) and (meta.ann_id.values == ann).all() and (meta.y.values == y).all()
    slides = meta.groupby("image_id").agg(file_name=("file_name", "first"), scanner=("scanner", "first"),
                                          n=("ann_id", "size")).sort_index()
    if a.limit:  # smoke test: a few slides of every scanner
        slides = slides.groupby("scanner", group_keys=False).head(a.limit)
    rows = np.flatnonzero(meta.image_id.isin(slides.index).values)

    if a.from_csv:
        vec = pd.read_csv(out_csv).set_index("image_id").loc[slides.index].reset_index()
        steps = sorted(vec.step.unique().tolist()) if "step" in vec else "not recorded"
        print(f"stain vectors from {out_csv} ({len(vec)} slides, estimated with step {steps})")
        if "maxC_all_H" not in vec:
            print("WARNING: CSV predates tissue-only maxC (no maxC_all_* columns): its maxC is the all-pixel value")
    else:
        recs, t0 = [], time.time()
        for k, (iid, s) in enumerate(slides.iterrows()):
            t1 = time.time()
            img = load_rgb(Path(a.data_dir) / "images" / s.file_name)
            r = estimate(img[::a.step, ::a.step])
            r2 = estimate(img[a.step // 2::a.step, a.step // 2::a.step])   # disjoint grid: reproducibility only
            r["stab_H_deg"] = angle_deg(r["HE"][:, 0], r2["HE"][:, 0])
            r["stab_E_deg"] = angle_deg(r["HE"][:, 1], r2["HE"][:, 1])
            r["stab_maxC"] = float(np.max(np.abs(r2["maxC"] - r["maxC"]) / r["maxC"]))
            r["step"] = a.step
            # diagnostic only (not a validity criterion): top vs bottom half of the image = tissue heterogeneity
            sub = img[::a.step, ::a.step]
            try:
                t, b = estimate(sub[: len(sub) // 2]), estimate(sub[len(sub) // 2:])
                r["half_H_deg"] = angle_deg(t["HE"][:, 0], b["HE"][:, 0])
                r["half_E_deg"] = angle_deg(t["HE"][:, 1], b["HE"][:, 1])
            except ValueError:
                r["half_H_deg"] = r["half_E_deg"] = np.nan
            del img, sub
            r["angle_HE_deg"] = angle_deg(r["HE"][:, 0], r["HE"][:, 1])
            r["angle_H_ref_deg"] = angle_deg(r["HE"][:, 0], HE_REF[:, 0])
            r["angle_E_ref_deg"] = angle_deg(r["HE"][:, 1], HE_REF[:, 1])
            r["bad"] = check_vectors(r)
            recs.append(row_of(iid, s.file_name, s.scanner, int(s.n), r))
            print(f"[{k + 1:3d}/{len(slides)}] {s.file_name} {s.scanner:<4} H={np.round(r['HE'][:, 0], 3)} "
                  f"E={np.round(r['HE'][:, 1], 3)} maxC={np.round(r['maxC'], 3)} tissue={r['n_tissue']} "
                  f"({r['n_tissue'] / r['n_pixels']:.2f}) angHE={r['angle_HE_deg']:.1f} "
                  f"stab={r['stab_H_deg']:.2f}/{r['stab_E_deg']:.2f}deg,{r['stab_maxC']:.3f} "
                  f"{'OK' if not r['bad'] else 'BAD: ' + '; '.join(r['bad'])}  {time.time() - t1:.1f}s",
                  flush=True)
        vec = pd.DataFrame(recs)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_csv.with_name(out_csv.stem + ".partial.csv")
        vec.to_csv(tmp, index=False, float_format="%.6g")
        os.replace(tmp, out_csv)
        print(f"{len(vec)} slides in {time.time() - t0:.0f}s; vectors -> {out_csv}")
        vec = pd.read_csv(out_csv)          # normalise with exactly the values stored in the CSV
    if not vec.valid.all():
        raise SystemExit(f"invalid stain estimate for slides {vec.image_id[~vec.valid].tolist()}; not writing patches")

    Xn = X.copy() if a.limit else np.empty_like(X)
    done = np.zeros(len(X), bool)
    for s in vec.itertuples():
        idx = np.flatnonzero(meta.image_id.values == s.image_id)
        HE, maxC = he_from_row(s)
        Xn[idx] = normalise(X[idx], HE, maxC)
        done[idx] = True
    assert done[rows].all() and done.sum() == len(rows)
    meta_s, X_s, Xn_s = meta.iloc[rows].reset_index(drop=True), X[rows], Xn[rows]
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_npz.with_name(out_npz.stem + ".partial.npz")
    if a.limit:
        np.savez_compressed(tmp, X=Xn_s, y=y[rows], ann_id=ann[rows])
    else:
        np.savez_compressed(tmp, X=Xn, y=y, ann_id=ann)
    os.replace(tmp, out_npz)
    print(f"normalised patches {Xn_s.shape if a.limit else Xn.shape} -> {out_npz}")

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    qa(meta_s, X_s, Xn_s, vec, out_fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", type=Path, default=Path("/Volumes/T7/midogpp_breast"))
    ap.add_argument("--step", type=int, default=STEP, help="pixel subsampling of the slide for the estimate (>= 2)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing to an existing output path other than the three defaults")
    ap.add_argument("--from-csv", action="store_true", help="reuse the stain vectors in --out-csv (no TIFF reads)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N slides per scanner")
    ap.add_argument("--out-npz", default=str(DEF_NPZ))
    ap.add_argument("--out-csv", default=str(DEF_CSV))
    ap.add_argument("--out-fig", default=str(DEF_FIG))
    main(ap.parse_args())
