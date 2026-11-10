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


def scanner_rgb_means(images, scanners):
    means = {}
    for scanner in sorted(set(scanners.tolist())):
        scanner_images = images[scanners == scanner].astype(np.float32) / 255.0
        means[scanner] = scanner_images.mean(axis=(0, 1, 2))
    return means


def scanner_stat_augment(images, source_scanners, scanner_means, held_out_scanner, copies, seed, strength=0.65):
    if copies <= 0:
        return images

    rng = np.random.default_rng(seed)
    base = images.astype(np.float32) / 255.0
    augmented = [images]
    train_scanners = [scanner for scanner in sorted(scanner_means) if scanner != held_out_scanner]

    for _ in range(copies):
        target_scanners = rng.choice(train_scanners, size=len(images), replace=True)
        target_means = np.stack([scanner_means[scanner] for scanner in target_scanners]).reshape(len(images), 1, 1, 3)
        current_means = base.mean(axis=(1, 2), keepdims=True)
        shift = (target_means - current_means) * strength

        brightness = rng.uniform(0.90, 1.10, size=(len(images), 1, 1, 1)).astype(np.float32)
        contrast = rng.uniform(0.90, 1.10, size=(len(images), 1, 1, 1)).astype(np.float32)

        aug = base + shift
        image_mean = aug.mean(axis=(1, 2), keepdims=True)
        aug = (aug - image_mean) * contrast + image_mean
        aug = aug * brightness
        aug = np.clip(aug, 0.0, 1.0)
        augmented.append((aug * 255).astype(np.uint8))

    return np.concatenate(augmented, axis=0)


def run_experiment(
    images,
    labels,
    scanners,
    held_out_scanner,
    mode,
    scanner_means,
    size,
    augment_copies,
    epochs,
    lr,
    l2,
    seed,
):
    train_indices = np.where(scanners != held_out_scanner)[0]
    test_indices = np.where(scanners == held_out_scanner)[0]

    train_images = images[train_indices]
    train_scanners = scanners[train_indices]
    y_train = labels[train_indices].astype(np.float32)
    test_images = images[test_indices]
    y_test = labels[test_indices].astype(np.float32)

    if mode == "generic_aug":
        train_images = augment_images(train_images, augment_copies, seed)
        y_train = np.tile(y_train, augment_copies + 1)
    elif mode == "scanner_stat_aug":
        train_images = scanner_stat_augment(
            train_images,
            train_scanners,
            scanner_means=scanner_means,
            held_out_scanner=held_out_scanner,
            copies=augment_copies,
            seed=seed,
        )
        y_train = np.tile(y_train, augment_copies + 1)
    elif mode != "baseline":
        raise ValueError(f"Unknown mode: {mode}")

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
        "model": mode,
        "augment_copies": int(augment_copies if mode != "baseline" else 0),
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
                    "model": result["model"],
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


def summarize(results):
    by_model = {}
    for result in results:
        by_model.setdefault(result["model"], []).append(result)
    for model, model_results in by_model.items():
        mean_f1 = np.mean([result["held_out_scanner_test"]["f1"] for result in model_results])
        mean_recall = np.mean([result["held_out_scanner_test"]["recall"] for result in model_results])
        mean_precision = np.mean([result["held_out_scanner_test"]["precision"] for result in model_results])
        print(f"{model}: mean_f1={mean_f1:.3f} mean_precision={mean_precision:.3f} mean_recall={mean_recall:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run LOSO pixel experiments with scanner-statistics augmentation.")
    parser.add_argument("--cache", default="demo1/patch_dataset_all/breast_patch_cache.npz", help="All-patch cache.")
    parser.add_argument("--output-dir", default="demo1/models/loso_scanner_stat_aug", help="Output directory.")
    parser.add_argument("--size", type=int, default=32, help="Downsampled square size.")
    parser.add_argument("--epochs", type=int, default=2500, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=5e-4, help="L2 regularization.")
    parser.add_argument("--augment-copies", type=int, default=3, help="Augmented copies for augmentation modes.")
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
    scanner_means = scanner_rgb_means(images, scanners)

    print("scanner RGB means:")
    for scanner, mean in scanner_means.items():
        print(f"  {scanner}: {', '.join(f'{value:.3f}' for value in mean)}")

    results = []
    modes = ["baseline", "generic_aug", "scanner_stat_aug"]
    for held_out_scanner in sorted(set(scanners.tolist())):
        print(f"held out scanner: {held_out_scanner}", flush=True)
        for mode in modes:
            print(f"  running {mode}", flush=True)
            result = run_experiment(
                images=images,
                labels=labels,
                scanners=scanners,
                held_out_scanner=held_out_scanner,
                mode=mode,
                scanner_means=scanner_means,
                size=args.size,
                augment_copies=args.augment_copies,
                epochs=args.epochs,
                lr=args.lr,
                l2=args.l2,
                seed=args.seed,
            )
            results.append(result)
            test = result["held_out_scanner_test"]
            print(
                f"  {mode}: test_f1={test['f1']:.3f} precision={test['precision']:.3f} "
                f"recall={test['recall']:.3f}",
                flush=True,
            )

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_csv(output_dir / "results.csv", results)
    summarize(results)
    print(f"json: {output_dir / 'results.json'}")
    print(f"csv: {output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
