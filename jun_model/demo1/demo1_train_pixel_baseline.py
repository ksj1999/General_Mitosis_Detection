import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


LABEL_TO_INDEX = {
    "not_mitotic_figure": 0,
    "mitotic_figure": 1,
}


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def downsample_features(images, size):
    features = []
    for image in images:
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((size, size), Image.BILINEAR)
        arr = np.asarray(pil_image, dtype=np.float32) / 255.0
        features.append(arr.reshape(-1))
    return np.vstack(features).astype(np.float32)


def augment_images(images, copies, seed):
    if copies <= 0:
        return images

    rng = np.random.default_rng(seed)
    augmented = [images]
    base = images.astype(np.float32) / 255.0

    for _ in range(copies):
        brightness = rng.uniform(0.80, 1.20, size=(len(images), 1, 1, 1)).astype(np.float32)
        contrast = rng.uniform(0.80, 1.20, size=(len(images), 1, 1, 1)).astype(np.float32)
        channel_scale = rng.uniform(0.75, 1.25, size=(len(images), 1, 1, 3)).astype(np.float32)
        channel_shift = rng.uniform(-0.06, 0.06, size=(len(images), 1, 1, 3)).astype(np.float32)

        aug = base * brightness
        image_mean = aug.mean(axis=(1, 2), keepdims=True)
        aug = (aug - image_mean) * contrast + image_mean
        aug = aug * channel_scale + channel_shift
        aug = np.clip(aug, 0.0, 1.0)
        augmented.append((aug * 255).astype(np.uint8))

    return np.concatenate(augmented, axis=0)


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

        if epoch % 500 == 0:
            print(f"epoch {epoch}/{epochs}", flush=True)

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
    parser = argparse.ArgumentParser(description="Train a downsampled-pixel logistic baseline.")
    parser.add_argument("--cache", default="jun_model/demo1/patch_dataset/patch_cache.npz", help="Cached patch dataset.")
    parser.add_argument("--output-dir", default="jun_model/demo1/models/pixel_baseline", help="Output directory.")
    parser.add_argument("--size", type=int, default=32, help="Downsampled square size.")
    parser.add_argument("--epochs", type=int, default=2500, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=5e-4, help="L2 regularization.")
    parser.add_argument("--augment-copies", type=int, default=0, help="Number of augmented copies per train patch.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for augmentation.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = np.load(args.cache)
    images = cache["images"]
    labels = cache["labels"].astype(np.float32)
    splits = cache["splits"]

    train_indices = np.where(splits == "train")[0]
    test_indices = np.where(splits == "test")[0]

    train_images = images[train_indices]
    y_train = labels[train_indices]
    if args.augment_copies > 0:
        print(f"augmenting train images with {args.augment_copies} copies...", flush=True)
        train_images = augment_images(train_images, args.augment_copies, args.seed)
        y_train = np.tile(y_train, args.augment_copies + 1)

    print("building train pixel features...", flush=True)
    x_train = downsample_features(train_images, args.size)
    print("building test pixel features...", flush=True)
    x_test = downsample_features(images[test_indices], args.size)
    y_test = labels[test_indices]

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
        "downsample_size": args.size,
        "feature_count": int(x_train.shape[1]),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "augment_copies": int(args.augment_copies),
        "seed": int(args.seed),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.savez(output_dir / "model.npz", weights=weights, bias=bias, mean=mean, std=std)

    print("train:", metrics["train"], flush=True)
    print("held_out_scanner_test:", metrics["held_out_scanner_test"], flush=True)
    print(f"metrics: {output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
