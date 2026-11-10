import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


METRICS = ("precision", "recall", "f1", "pr_auc")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compute_metrics(rows):
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    predictions = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(labels, probabilities)),
    }


def interval(values):
    return {
        "estimate": float(values[0]),
        "ci_95_low": float(np.percentile(values[1:], 2.5)),
        "ci_95_high": float(np.percentile(values[1:], 97.5)),
    }


def bootstrap(predictions_path, samples, seed):
    rows = read_rows(predictions_path)
    by_fold_image = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_fold_image[row["fold"]][row["image_id"]].append(row)

    rng = np.random.default_rng(seed)
    folds = sorted(by_fold_image)
    observed_by_fold = {
        fold: compute_metrics([row for image_rows in by_fold_image[fold].values() for row in image_rows])
        for fold in folds
    }
    bootstrap_by_fold = {fold: {metric: [] for metric in METRICS} for fold in folds}
    macro_bootstrap = {metric: [] for metric in METRICS}

    for _ in range(samples):
        sample_metrics = {}
        for fold in folds:
            images = sorted(by_fold_image[fold])
            sampled_images = rng.choice(images, size=len(images), replace=True)
            sampled_rows = [row for image_id in sampled_images for row in by_fold_image[fold][str(image_id)]]
            sample_metrics[fold] = compute_metrics(sampled_rows)
            for metric in METRICS:
                bootstrap_by_fold[fold][metric].append(sample_metrics[fold][metric])
        for metric in METRICS:
            macro_bootstrap[metric].append(float(np.mean([sample_metrics[fold][metric] for fold in folds])))

    result = {
        "resampling_unit": "image",
        "bootstrap_samples": samples,
        "seed": seed,
        "folds": {},
        "macro_mean": {},
    }
    for fold in folds:
        result["folds"][fold] = {
            metric: interval([observed_by_fold[fold][metric], *bootstrap_by_fold[fold][metric]])
            for metric in METRICS
        }
    for metric in METRICS:
        observed_macro = float(np.mean([observed_by_fold[fold][metric] for fold in folds]))
        result["macro_mean"][metric] = interval([observed_macro, *macro_bootstrap[metric]])
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Compute image-level bootstrap intervals for locked test metrics.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    result = bootstrap(args.predictions, args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
