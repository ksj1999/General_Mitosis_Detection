import argparse
import csv
import json
from pathlib import Path

import numpy as np

from demo1_train_pixel_baseline import (
    augment_images,
    compute_metrics,
    downsample_features,
    fit_logistic_regression,
    sigmoid,
)


def run_experiment(images, labels, scanners, held_out_scanner, size, augment_copies, epochs, lr, l2, seed):
    train_indices = np.where(scanners != held_out_scanner)[0]
    test_indices = np.where(scanners == held_out_scanner)[0]

    train_images = images[train_indices]
    y_train = labels[train_indices].astype(np.float32)
    test_images = images[test_indices]
    y_test = labels[test_indices].astype(np.float32)

    if augment_copies > 0:
        train_images = augment_images(train_images, augment_copies, seed)
        y_train = np.tile(y_train, augment_copies + 1)

    x_train = downsample_features(train_images, size)
    x_test = downsample_features(test_images, size)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    weights, bias = fit_logistic_regression(x_train, y_train, epochs, lr, l2)
    train_probs = sigmoid(x_train @ weights + bias)
    test_probs = sigmoid(x_test @ weights + bias)

    return {
        "held_out_scanner": str(held_out_scanner),
        "augment_copies": int(augment_copies),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train": compute_metrics(y_train, train_probs),
        "held_out_scanner_test": compute_metrics(y_test, test_probs),
    }


def write_csv(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "held_out_scanner",
            "model",
            "augment_copies",
            "train_rows",
            "test_rows",
            "train_f1",
            "test_f1",
            "test_precision",
            "test_recall",
            "test_accuracy",
            "test_tp",
            "test_fp",
            "test_tn",
            "test_fn",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            test = result["held_out_scanner_test"]
            writer.writerow(
                {
                    "held_out_scanner": result["held_out_scanner"],
                    "model": "pixel_augmented" if result["augment_copies"] > 0 else "pixel_baseline",
                    "augment_copies": result["augment_copies"],
                    "train_rows": result["train_rows"],
                    "test_rows": result["test_rows"],
                    "train_f1": result["train"]["f1"],
                    "test_f1": test["f1"],
                    "test_precision": test["precision"],
                    "test_recall": test["recall"],
                    "test_accuracy": test["accuracy"],
                    "test_tp": test["tp"],
                    "test_fp": test["fp"],
                    "test_tn": test["tn"],
                    "test_fn": test["fn"],
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Run leave-one-scanner-out pixel baseline experiments.")
    parser.add_argument("--cache", default="jun_model/demo1/patch_dataset_all/breast_patch_cache.npz", help="All-patch cache.")
    parser.add_argument("--output-dir", default="jun_model/demo1/models/loso_pixel", help="Output directory.")
    parser.add_argument("--size", type=int, default=32, help="Downsampled square size.")
    parser.add_argument("--epochs", type=int, default=2500, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=5e-4, help="L2 regularization.")
    parser.add_argument("--augment-copies", type=int, default=3, help="Augmented copies for augmented runs.")
    parser.add_argument("--seed", type=int, default=42, help="Augmentation seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = np.load(args.cache)
    images = cache["images"]
    labels = cache["labels"]
    scanners = cache["scanners"]

    results = []
    for held_out_scanner in sorted(set(scanners.tolist())):
        print(f"held out scanner: {held_out_scanner}", flush=True)
        for augment_copies in [0, args.augment_copies]:
            label = "augmented" if augment_copies else "baseline"
            print(f"  running {label}", flush=True)
            result = run_experiment(
                images=images,
                labels=labels,
                scanners=scanners,
                held_out_scanner=held_out_scanner,
                size=args.size,
                augment_copies=augment_copies,
                epochs=args.epochs,
                lr=args.lr,
                l2=args.l2,
                seed=args.seed,
            )
            results.append(result)
            test = result["held_out_scanner_test"]
            print(
                f"  {label}: test_f1={test['f1']:.3f} "
                f"precision={test['precision']:.3f} recall={test['recall']:.3f}",
                flush=True,
            )

    with (output_dir / "loso_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_csv(output_dir / "loso_results.csv", results)
    print(f"json: {output_dir / 'loso_results.json'}", flush=True)
    print(f"csv: {output_dir / 'loso_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
