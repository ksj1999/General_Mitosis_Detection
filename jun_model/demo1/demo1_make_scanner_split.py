import argparse
import csv
from pathlib import Path


DEFAULT_TRAIN_SCANNERS = ["Hamammatsu XR", "Hamamatsu S360"]
DEFAULT_TEST_SCANNER = "Aperio CS2"


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Create Demo 1 scanner holdout CSV manifests.")
    parser.add_argument("--subset-dir", default="jun_model/breast_cancer_subset", help="Path to the prepared subset folder.")
    parser.add_argument("--output-dir", default="jun_model/demo1/splits", help="Directory for generated split files.")
    parser.add_argument(
        "--test-scanner",
        default=DEFAULT_TEST_SCANNER,
        help="Scanner to hold out for scanner-domain testing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    subset_dir = Path(args.subset_dir)
    output_dir = Path(args.output_dir)

    source_csv = subset_dir / "metadata" / "breast_cancer_slides.csv"
    rows = read_rows(source_csv)

    train_scanners = [scanner for scanner in DEFAULT_TRAIN_SCANNERS if scanner != args.test_scanner]
    train_rows = [row for row in rows if row["Scanner"] in train_scanners and row["Dataset"] == "train"]
    test_rows = [row for row in rows if row["Scanner"] == args.test_scanner]

    fieldnames = list(rows[0].keys())
    write_rows(output_dir / "train_scanner_holdout.csv", train_rows, fieldnames)
    write_rows(output_dir / "test_scanner_holdout.csv", test_rows, fieldnames)

    print("Created scanner holdout manifests")
    print(f"train scanners: {', '.join(train_scanners)}")
    print(f"test scanner: {args.test_scanner}")
    print(f"train images: {len(train_rows)}")
    print(f"test images: {len(test_rows)}")
    print(f"train file: {output_dir / 'train_scanner_holdout.csv'}")
    print(f"test file: {output_dir / 'test_scanner_holdout.csv'}")


if __name__ == "__main__":
    main()
