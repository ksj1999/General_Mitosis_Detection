import argparse
import csv
import json
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


def image_features(path):
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    channels = [arr[:, :, idx].reshape(-1) for idx in range(3)]

    features = []
    for channel in channels:
        features.extend(
            [
                channel.mean(),
                channel.std(),
                np.percentile(channel, 5),
                np.percentile(channel, 25),
                np.percentile(channel, 50),
                np.percentile(channel, 75),
                np.percentile(channel, 95),
            ]
        )

    gray = arr.mean(axis=2).reshape(-1)
    features.extend(
        [
            gray.mean(),
            gray.std(),
            np.percentile(gray, 5),
            np.percentile(gray, 50),
            np.percentile(gray, 95),
        ]
    )
    return np.asarray(features, dtype=np.float32)


def build_matrix(patch_dir, rows):
    x = []
    y = []
    for row in rows:
        x.append(image_features(patch_dir / row["patch_path"]))
        y.append(LABEL_TO_INDEX[row["label"]])
    return np.vstack(x), np.asarray(y, dtype=np.float32)


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_regression(x, y, epochs, lr, l2):
    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = 0.0

    positives = max(float((y == 1).sum()), 1.0)
    negatives = max(float((y == 0).sum()), 1.0)
    positive_weight = len(y) / (2.0 * positives)
    negative_weight = len(y) / (2.0 * negatives)
    sample_weights = np.where(y == 1, positive_weight, negative_weight).astype(np.float32)

    for epoch in range(1, epochs + 1):
        probs = sigmoid(x @ weights + bias)
        error = (probs - y) * sample_weights
        grad_w = (x.T @ error) / len(y) + l2 * weights
        grad_b = error.mean()

        weights -= lr * grad_w
        bias -= lr * grad_b

    return weights, bias


def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(np.float32)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())

    eps = 1e-8
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train a fast NumPy feature baseline for patch classification.")
    parser.add_argument("--patch-dir", default="demo1/patch_dataset", help="Patch dataset directory.")
    parser.add_argument("--output-dir", default="demo1/models/feature_baseline", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=2500, help="Logistic regression training epochs.")
    parser.add_argument("--lr", type=float, default=0.08, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization.")
    return parser.parse_args()


def main():
    args = parse_args()
    patch_dir = Path(args.patch_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(patch_dir / "patch_metadata.csv")
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]

    print("extracting train features...", flush=True)
    x_train, y_train = build_matrix(patch_dir, train_rows)
    print("extracting test features...", flush=True)
    x_test, y_test = build_matrix(patch_dir, test_rows)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    print(f"train rows: {len(y_train)}", flush=True)
    print(f"test rows: {len(y_test)}", flush=True)
    print(f"features: {x_train.shape[1]}", flush=True)

    weights, bias = fit_logistic_regression(x_train, y_train, args.epochs, args.lr, args.l2)
    train_probs = sigmoid(x_train @ weights + bias)
    test_probs = sigmoid(x_test @ weights + bias)

    metrics = {
        "train": compute_metrics(y_train, train_probs),
        "held_out_scanner_test": compute_metrics(y_test, test_probs),
        "label_to_index": LABEL_TO_INDEX,
        "feature_count": int(x_train.shape[1]),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.savez(output_dir / "model.npz", weights=weights, bias=bias, mean=mean, std=std)

    print("train:", metrics["train"], flush=True)
    print("held_out_scanner_test:", metrics["held_out_scanner_test"], flush=True)
    print(f"metrics: {output_dir / 'metrics.json'}", flush=True)
    print(f"model: {output_dir / 'model.npz'}", flush=True)


if __name__ == "__main__":
    main()
