import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


LABEL_TO_INDEX = {
    "not_mitotic_figure": 0,
    "mitotic_figure": 1,
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_args():
    parser = argparse.ArgumentParser(description="Cache patch JPGs into a single NumPy dataset.")
    parser.add_argument("--patch-dir", default="jun_model/demo1/patch_dataset", help="Patch dataset directory.")
    parser.add_argument("--output", default="jun_model/demo1/patch_dataset/patch_cache.npz", help="Output cache file.")
    return parser.parse_args()


def main():
    args = parse_args()
    patch_dir = Path(args.patch_dir)
    output = Path(args.output)

    rows = read_rows(patch_dir / "patch_metadata.csv")
    images = []
    labels = []
    splits = []
    scanners = []
    patch_paths = []

    for idx, row in enumerate(rows, start=1):
        image = Image.open(patch_dir / row["patch_path"]).convert("RGB")
        images.append(np.asarray(image, dtype=np.uint8))
        labels.append(LABEL_TO_INDEX[row["label"]])
        splits.append(row["split"])
        scanners.append(row["scanner"])
        patch_paths.append(row["patch_path"])

        if idx % 500 == 0:
            print(f"cached {idx}/{len(rows)} patches", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        images=np.stack(images),
        labels=np.asarray(labels, dtype=np.int64),
        splits=np.asarray(splits),
        scanners=np.asarray(scanners),
        patch_paths=np.asarray(patch_paths),
    )

    print(f"cache: {output}", flush=True)
    print(f"images: {len(images)}", flush=True)
    print(f"shape: {images[0].shape}", flush=True)


if __name__ == "__main__":
    main()
