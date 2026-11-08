#!/usr/bin/env python3
"""
Random tissue crops for "AdaBN from random crops" (ADDONS_PROTOCOL.md, [adabnrand]).

For each of the 150 MIDOG++ breast slides, sample N (default 48) crops at random
positions inside tissue. Each crop covers the same physical field as the candidate
patches of prepare_patches.py (128 px at 0.25 um/px = 32 um, i.e. 128 px on CS2 and
139 px on XR/S360), and is resized to 128x128 with LANCZOS exactly as there.

Tissue mask: the slide is box-averaged by 16 (PIL reduce); a 16x16 cell is tissue when
its mean RGB is in [40, 215) and its channel spread (max - min) is > 15 (white glass and
grey background have spread < ~5). A crop centre is drawn uniformly from the tissue
cells (plus a uniform offset inside the cell); the crop must lie fully inside the slide
and >= 50 % of the mask cells whose centres fall inside the crop window must be tissue,
otherwise it is redrawn. RNG: np.random.RandomState([seed, image_id]) per slide, so a
slide's crops do not depend on which other slides are processed.

Usage:  python scripts/prepare_random_crops.py [--per-slide 48] [--seed 0] [--force]
        python scripts/prepare_random_crops.py --slides 3 60 110 --out SCRATCH.npz [--fig SCRATCH.png]
        (a --slides subset needs an explicit --out and writes a figure only with --fig; an existing
        --out is never replaced without --force)
Output: data/random_crops_128.npz   X (N,128,128,3) uint8, image_id, scanner (str), cx, cy,
                                    tissue_frac (crop's mask fraction)
        figures/random_crops_check.png  random crops next to candidate patches, per scanner
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_patches import OUT, SCANNER, TARGET_UM, UM_PER_PX, crop, load_rgb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DS = 16                      # mask cell size (px at native resolution)
MEAN_MAX, MEAN_MIN, SPREAD_MIN = 215.0, 40.0, 15.0
MIN_TISSUE = 0.5


def tissue_mask(img):
    sm = np.asarray(Image.fromarray(np.ascontiguousarray(img)).reduce(DS)).astype(np.float32)
    mean, spread = sm.mean(2), sm.max(2) - sm.min(2)
    return (mean < MEAN_MAX) & (mean >= MEAN_MIN) & (spread > SPREAD_MIN), sm.astype(np.uint8)


def window_cells(lo, hi):
    """Mask-cell index range [a, b) whose cell centres (DS*j + DS/2) lie in the pixel window [lo, hi)."""
    return int(np.ceil((lo - DS / 2) / DS)), int(np.ceil((hi - DS / 2) / DS))


def sample_slide(img, mask, size, n, rng, max_tries=200):
    H, W = img.shape[:2]
    h = size // 2
    cy_cells, cx_cells = np.nonzero(mask)
    if len(cy_cells) == 0:
        raise RuntimeError("empty tissue mask")
    crops, cxs, cys, fracs = [], [], [], []
    tries = 0
    while len(crops) < n:
        tries += 1
        if tries > max_tries * n:
            raise RuntimeError(f"only {len(crops)}/{n} crops after {tries} draws")
        k = rng.randint(len(cy_cells))
        cx = cx_cells[k] * DS + rng.randint(DS)
        cy = cy_cells[k] * DS + rng.randint(DS)
        x0, y0 = cx - h, cy - h                              # same window convention as prepare_patches.crop
        if x0 < 0 or y0 < 0 or x0 + size > W or y0 + size > H:
            continue
        (r0, r1), (c0, c1) = window_cells(y0, y0 + size), window_cells(x0, x0 + size)
        frac = float(mask[r0:r1, c0:c1].mean())
        if frac < MIN_TISSUE:
            continue
        c = crop(img, cx, cy, size)
        if c.shape[0] != OUT:
            c = np.asarray(Image.fromarray(c).resize((OUT, OUT), Image.LANCZOS))
        crops.append(np.array(c, dtype=np.uint8, copy=True))  # own memory: a CS2 crop is a view of the slide
        cxs.append(cx)
        cys.append(cy)
        fracs.append(frac)
    return crops, cxs, cys, fracs, tries


def qa_figure(d, thumbs, fig_path, n_show=6, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    meta = pd.read_csv(ROOT / "data" / "patches_meta.csv")
    Xc = np.load(ROOT / "data" / "patches_128.npz")["X"]
    rng = np.random.RandomState(seed)
    scanners = [s for s in ["XR", "S360", "CS2"] if (d["scanner"] == s).any()]
    ncol = 1 + 2 * n_show
    fig = plt.figure(figsize=(1.6 * ncol + 1.6, 1.9 * 2 * len(scanners)))
    gs = fig.add_gridspec(2 * len(scanners), ncol, width_ratios=[2.4] + [1] * (ncol - 1), wspace=0.05, hspace=0.25)
    for r, sc in enumerate(scanners):
        idx = np.nonzero(d["scanner"] == sc)[0]
        sid = int(rng.choice(np.unique(d["image_id"][idx])))
        size = int(round(OUT * TARGET_UM / UM_PER_PX[sc]))
        ax = fig.add_subplot(gs[2 * r:2 * r + 2, 0])
        sm, mask = thumbs[sid]
        ax.imshow(sm)
        ax.imshow(np.ma.masked_where(mask, np.ones(mask.shape)), cmap="cool", vmin=0, vmax=1, alpha=0.6)
        sel = idx[d["image_id"][idx] == sid]
        for cx, cy in zip(d["cx"][sel], d["cy"][sel]):
            ax.add_patch(Rectangle(((cx - size / 2) / DS, (cy - size / 2) / DS), size / DS, size / DS,
                                   fill=False, ec="cyan", lw=0.6))
        ax.set_title(f"{sc} slide {sid}: non-tissue (magenta), crops (cyan)", fontsize=8)
        ax.axis("off")
        for half, (src, rows) in enumerate([("random", rng.choice(idx, 2 * n_show, replace=False)),
                                            ("candidate", rng.choice(np.nonzero(meta.scanner.values == sc)[0],
                                                                     2 * n_show, replace=False))]):
            for j, i in enumerate(rows):
                ax = fig.add_subplot(gs[2 * r + j // n_show, 1 + half * n_show + j % n_show])
                ax.imshow(d["X"][i] if src == "random" else Xc[i])
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_edgecolor("tab:blue" if src == "random" else "tab:red")
                    s.set_linewidth(2)
                if j == 0:
                    ax.set_title(f"{sc} {src} crops" if src == "random" else f"{sc} candidate patches",
                                 fontsize=8, loc="left")
    fig.suptitle("Random tissue crops (blue frame) vs candidate patches (red frame), same physical field 32 um",
                 fontsize=10, y=0.92)
    fig.savefig(fig_path, dpi=110, bbox_inches="tight")
    print("wrote", fig_path)


def main(a):
    if a.slides and a.out is None:
        raise SystemExit("--slides (subset) needs an explicit --out, so the full crop file is not replaced")
    if a.out is None:
        a.out = ROOT / "data" / "random_crops_128.npz"
    if a.fig is None and not a.slides:
        a.fig = ROOT / "figures" / "random_crops_check.png"
    for f in (a.out, a.fig):
        if f is not None and f.exists() and not a.force:
            raise SystemExit(f"{f} exists; pass --force to replace it (or choose another --out / --fig)")
    slides = pd.read_csv(a.data_dir / "metadata" / "breast_cancer_slides.csv")
    slides["scanner"] = slides.Scanner.map(SCANNER)
    assert slides.scanner.notna().all(), slides.Scanner.unique()
    slides = slides.sort_values("Slide").reset_index(drop=True)
    if a.slides:
        missing = sorted(set(a.slides) - set(slides.Slide))
        if missing:
            raise SystemExit(f"unknown slide ids {missing}")
        slides = slides[slides.Slide.isin(a.slides)].reset_index(drop=True)
    X, iid, scn, cxs, cys, fracs = [], [], [], [], [], []
    thumbs = {}
    t_start = time.time()
    for k, s in slides.iterrows():
        t0 = time.time()
        img = load_rgb(a.data_dir / "images" / s.file_name)
        mask, sm = tissue_mask(img)
        size = int(round(OUT * TARGET_UM / UM_PER_PX[s.scanner]))
        rng = np.random.RandomState([a.seed, int(s.Slide)])
        c, x, y, f, tries = sample_slide(img, mask, size, a.per_slide, rng)
        X += c
        cxs += x
        cys += y
        fracs += f
        iid += [int(s.Slide)] * len(c)
        scn += [s.scanner] * len(c)
        thumbs[int(s.Slide)] = (sm, mask)
        print(f"[{k + 1:3d}/{len(slides)}] {s.file_name} {s.scanner:<4} size {size} px  tissue {mask.mean():.2f}  "
              f"{len(c)} crops / {tries} draws  mean frac {np.mean(f):.2f}  {time.time() - t0:.1f}s", flush=True)
        del img
    d = {"X": np.stack(X).astype(np.uint8), "image_id": np.array(iid, dtype=np.int64),
         "scanner": np.array(scn, dtype="<U4"), "cx": np.array(cxs, dtype=np.int64),
         "cy": np.array(cys, dtype=np.int64), "tissue_frac": np.array(fracs, dtype=np.float32)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **d)
    print(f"\nwrote {len(d['X'])} crops X={d['X'].shape} -> {a.out}  ({time.time() - t_start:.0f}s)")
    print(pd.Series(d["scanner"]).value_counts().to_string())
    print("tissue_frac quantiles:", np.quantile(d["tissue_frac"], [0, 0.05, 0.5, 1]).round(2))
    if a.fig:
        a.fig.parent.mkdir(parents=True, exist_ok=True)
        qa_figure(d, thumbs, a.fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("/Volumes/T7/midogpp_breast"))
    ap.add_argument("--per-slide", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slides", type=int, nargs="*", help="subset of slide ids (testing; needs --out)")
    ap.add_argument("--out", type=Path, default=None, help="default data/random_crops_128.npz (full run only)")
    ap.add_argument("--fig", type=Path, default=None,
                    help="default figures/random_crops_check.png for a full run, none for --slides")
    ap.add_argument("--force", action="store_true", help="replace an existing --out / --fig")
    main(ap.parse_args())
