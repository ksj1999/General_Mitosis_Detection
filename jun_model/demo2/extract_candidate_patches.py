import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from patch_utils import extract_candidate_patch


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def extract_all(manifest, output_dir, scales, quality):
    rows = read_rows(manifest)
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["image_path"]].append(row)

    written = 0
    for image_path, image_rows in tqdm(sorted(by_image.items()), desc="source images", unit="image"):
        with Image.open(image_path) as image:
            for row in image_rows:
                for scale in scales:
                    destination = output_dir / scale / f"{int(row['candidate_id']):05d}.jpg"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    patch = extract_candidate_patch(image, row, scale)
                    patch.save(destination, quality=quality, subsampling=0)
                    written += 1
    return written


def parse_args():
    parser = argparse.ArgumentParser(description="Extract MPP-aware candidate patches once for model reuse.")
    parser.add_argument("--manifest", type=Path, default=Path("jun_model/demo2/data/candidates.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("jun_model/demo2/patches"))
    parser.add_argument("--scales", nargs="+", choices=("20x", "40x"), default=("20x", "40x"))
    parser.add_argument("--quality", type=int, default=95)
    return parser.parse_args()


def main():
    args = parse_args()
    written = extract_all(args.manifest, args.output_dir, args.scales, args.quality)
    print(f"patches written: {written}")
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
