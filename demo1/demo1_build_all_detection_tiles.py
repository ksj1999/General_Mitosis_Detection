import argparse
import subprocess
import sys
from pathlib import Path


HOLDS = ["Aperio_CS2", "Hamamatsu_S360", "Hamammatsu_XR"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build detection tile datasets for all scanner holdouts.")
    parser.add_argument("--detection-coco-dir", default="demo1/detection_coco", help="Directory containing holdout COCO splits.")
    parser.add_argument("--output-root", default="demo1/detection_tiles", help="Output root for tile datasets.")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size.")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negative tiles per positive tile.")
    return parser.parse_args()


def main():
    args = parse_args()
    for holdout in HOLDS:
        coco_dir = Path(args.detection_coco_dir) / holdout
        output_dir = Path(args.output_root) / holdout
        cmd = [
            sys.executable,
            "demo1/demo1_build_detection_tiles.py",
            "--coco-dir",
            str(coco_dir),
            "--output-dir",
            str(output_dir),
            "--tile-size",
            str(args.tile_size),
            "--negative-ratio",
            str(args.negative_ratio),
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
