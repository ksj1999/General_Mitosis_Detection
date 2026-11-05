#!/usr/bin/env python3
"""
224x224 crops for the foundation-model probe, in the same row order as
data/patches_meta.csv. Each TIFF is read once and two views are cut per candidate,
both with the same physical field on every scanner:
  patches_224_um056.npz   56 um field at 0.25 um/px (40x-like, more context than the 32 um CNN patches)
  patches_224_um112.npz  112 um field at 0.50 um/px (20x, the resolution Phikon was pretrained at)

Usage:  python scripts/prepare_patches_224.py [--data-dir /Volumes/T7/midogpp_breast]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_patches import ROOT, UM_PER_PX, crop, load_rgb  # noqa: E402

VIEWS = {"um056": 0.25, "um112": 0.50}      # name -> target um per pixel of the 224 px output
OUT = 224


def main(data_dir: Path) -> None:
    meta = pd.read_csv(ROOT / "data" / "patches_meta.csv")
    arrs = {k: np.zeros((len(meta), OUT, OUT, 3), np.uint8) for k in VIEWS}
    t0 = time.time()
    for k, (fname, grp) in enumerate(meta.groupby("file_name", sort=True)):
        img = load_rgb(data_dir / "images" / fname)
        um = UM_PER_PX[grp.scanner.iloc[0]]
        for view, target in VIEWS.items():
            size = int(round(OUT * target / um))
            for i, r in grp.iterrows():
                c = crop(img, r.cx, r.cy, size)
                if c.shape[0] != OUT:
                    c = np.asarray(Image.fromarray(c).resize((OUT, OUT), Image.LANCZOS))
                arrs[view][i] = c
        if k % 10 == 0:
            print(f"[{k + 1:3d}/150] {fname} ({time.time() - t0:.0f}s)", flush=True)
    for view, a in arrs.items():
        np.save(ROOT / "data" / f"patches_224_{view}.npy", a)
        print("wrote", f"data/patches_224_{view}.npy", a.shape, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("/Volumes/T7/midogpp_breast"))
    main(ap.parse_args().data_dir)
