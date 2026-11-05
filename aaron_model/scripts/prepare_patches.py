#!/usr/bin/env python3
"""
Step 3: cut every annotated candidate out of the MIDOG++ breast subset.

For each of the 4,435 annotations, crop a window centred on the box, covering
the same physical area on every scanner (128 px at 0.25 um/px = 32 um), and
resize to 128x128. The big TIFFs are read once; afterwards training only needs
the ~200 MB output.

Output:
  data/patches_128.npz      X (N,128,128,3) uint8, y (1 = mitotic figure, 0 = imposter)
  data/patches_meta.csv     one row per patch: ids, scanner, split, rater votes
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = 128
TARGET_UM = 0.25
# Native resolution per scanner (MIDOG challenge paper, Aubreville et al. 2023).
UM_PER_PX = {"XR": 0.23, "S360": 0.23, "CS2": 0.25}
SCANNER = {"Hamammatsu XR": "XR", "Hamamatsu XR": "XR", "Hamamatsu S360": "S360", "Aperio CS2": "CS2"}


def load_rgb(path: Path) -> np.ndarray:
    a = tifffile.imread(path)
    return a[..., :3]  # XR/S360/CS2 files are RGBA with a constant alpha; drop it


def crop(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    h = size // 2
    x0, y0 = int(round(cx)) - h, int(round(cy)) - h
    x1, y1 = x0 + size, y0 + size
    H, W = img.shape[:2]
    pad = ((max(0, -y0), max(0, y1 - H)), (max(0, -x0), max(0, x1 - W)), (0, 0))
    c = img[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if any(p for p2 in pad[:2] for p in p2):
        c = np.pad(c, pad, mode="reflect")
    return c


def main(data_dir: Path) -> None:
    meta = json.load(open(data_dir / "metadata" / "annotations_breast_cancer.json"))
    slides = pd.read_csv(data_dir / "metadata" / "breast_cancer_slides.csv")
    slides["scanner"] = slides.Scanner.map(SCANNER)
    assert slides.scanner.notna().all(), slides.Scanner.unique()
    info = slides.set_index("Slide")
    images = {i["id"]: i for i in meta["images"]}
    by_img = {}
    for a in meta["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    X, rows = [], []
    for k, (iid, anns) in enumerate(sorted(by_img.items())):
        im = images[iid]
        sc = info.loc[iid, "scanner"]
        img = load_rgb(data_dir / "images" / im["file_name"])
        assert img.shape[:2] == (im["height"], im["width"]), (im["file_name"], img.shape)
        size = int(round(OUT * TARGET_UM / UM_PER_PX[sc]))
        for a in anns:
            x1, y1, x2, y2 = a["bbox"]  # [x1, y1, x2, y2], not COCO's [x, y, w, h]
            c = crop(img, (x1 + x2) / 2, (y1 + y2) / 2, size)
            if c.shape[0] != OUT:
                c = np.asarray(Image.fromarray(c).resize((OUT, OUT), Image.LANCZOS))
            X.append(c)
            votes = a.get("labels", [])
            rows.append({
                "ann_id": a["id"], "image_id": iid, "file_name": im["file_name"],
                "scanner": sc, "split": info.loc[iid, "Dataset"],
                "y": int(a["category_id"] == 1),
                "n_raters": len(votes),
                "vote_mitotic": float(np.mean([v == 1 for v in votes])) if votes else np.nan,
                "unanimous": bool(len(set(votes)) == 1) if votes else True,
                "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
            })
        print(f"[{k + 1:3d}/{len(by_img)}] {im['file_name']} {sc:<4} {len(anns):3d} candidates", flush=True)

    X = np.stack(X).astype(np.uint8)
    df = pd.DataFrame(rows)
    out = ROOT / "data"
    np.savez_compressed(out / "patches_128.npz", X=X, y=df.y.values, ann_id=df.ann_id.values)
    df.to_csv(out / "patches_meta.csv", index=False)
    print(f"\nwrote {len(df)} patches  X={X.shape}  ->  {out/'patches_128.npz'}")
    print(pd.crosstab([df.scanner, df.split], df.y.map({1: "mitotic", 0: "imposter"}), margins=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("/Volumes/T7/midogpp_breast"))
    main(ap.parse_args().data_dir)
