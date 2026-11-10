import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from demo1_train_pixel_baseline import (
    augment_images,
    downsample_features,
    fit_logistic_regression,
    sigmoid,
)


LABEL_NAMES = {
    0: "not mitotic",
    1: "mitotic",
}


def sanitize_name(value):
    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def fit_and_predict(images, labels, scanners, held_out_scanner, size, augment_copies, epochs, lr, l2, seed):
    train_indices = np.where(scanners != held_out_scanner)[0]
    test_indices = np.where(scanners == held_out_scanner)[0]

    train_images = images[train_indices]
    y_train = labels[train_indices].astype(np.float32)
    if augment_copies > 0:
        train_images = augment_images(train_images, augment_copies, seed)
        y_train = np.tile(y_train, augment_copies + 1)

    x_train = downsample_features(train_images, size)
    x_test = downsample_features(images[test_indices], size)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    weights, bias = fit_logistic_regression(x_train, y_train, epochs, lr, l2)
    probs = sigmoid(x_test @ weights + bias)
    preds = (probs >= 0.5).astype(np.int64)
    return test_indices, probs, preds


def select_examples(test_indices, labels, probs, preds, group, max_examples):
    true_labels = labels[test_indices]

    if group == "false_positive":
        mask = (true_labels == 0) & (preds == 1)
        order = np.argsort(-probs[mask])
    elif group == "false_negative":
        mask = (true_labels == 1) & (preds == 0)
        order = np.argsort(probs[mask])
    elif group == "true_positive":
        mask = (true_labels == 1) & (preds == 1)
        order = np.argsort(-probs[mask])
    elif group == "true_negative":
        mask = (true_labels == 0) & (preds == 0)
        order = np.argsort(probs[mask])
    else:
        raise ValueError(f"Unknown group: {group}")

    group_indices = test_indices[mask]
    group_probs = probs[mask]
    group_preds = preds[mask]
    chosen_order = order[:max_examples]
    return group_indices[chosen_order], group_probs[chosen_order], group_preds[chosen_order]


def draw_sheet(cache, examples, probs, preds, title, output_path, cols):
    images = cache["images"]
    labels = cache["labels"]
    source_images = cache["source_images"]
    annotation_ids = cache["annotation_ids"]

    tile = 128
    label_h = 46
    rows = int(np.ceil(len(examples) / cols))
    sheet = Image.new("RGB", (cols * tile, rows * (tile + label_h) + 28), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((6, 8), title, fill=(0, 0, 0), font=font)

    for n, index in enumerate(examples):
        row = n // cols
        col = n % cols
        x = col * tile
        y = 28 + row * (tile + label_h)

        patch = Image.fromarray(images[index]).convert("RGB")
        sheet.paste(patch, (x, y))

        true_label = LABEL_NAMES[int(labels[index])]
        pred_label = LABEL_NAMES[int(preds[n])]
        text = [
            f"p={probs[n]:.2f}",
            f"T:{true_label[:10]}",
            f"P:{pred_label[:10]}",
            f"{source_images[index]} #{int(annotation_ids[index])}",
        ]
        for line_idx, line in enumerate(text):
            draw.text((x + 2, y + tile + 2 + line_idx * 10), line, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def parse_args():
    parser = argparse.ArgumentParser(description="Create false-positive/false-negative patch preview sheets.")
    parser.add_argument("--cache", default="jun_model/demo1/patch_dataset_all/breast_patch_cache.npz", help="All-patch cache.")
    parser.add_argument("--held-out-scanner", default="Hamammatsu XR", help="Scanner to hold out.")
    parser.add_argument("--output-dir", default="jun_model/demo1/outputs/error_analysis", help="Output image directory.")
    parser.add_argument("--size", type=int, default=32, help="Downsampled feature size.")
    parser.add_argument("--epochs", type=int, default=2500, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=5e-4, help="L2 regularization.")
    parser.add_argument("--augment-copies", type=int, default=0, help="Augmented copies for training.")
    parser.add_argument("--seed", type=int, default=42, help="Augmentation seed.")
    parser.add_argument("--max-examples", type=int, default=24, help="Examples per sheet.")
    parser.add_argument("--cols", type=int, default=6, help="Contact sheet columns.")
    return parser.parse_args()


def main():
    args = parse_args()
    cache = np.load(args.cache)
    images = cache["images"]
    labels = cache["labels"]
    scanners = cache["scanners"]

    test_indices, probs, preds = fit_and_predict(
        images=images,
        labels=labels,
        scanners=scanners,
        held_out_scanner=args.held_out_scanner,
        size=args.size,
        augment_copies=args.augment_copies,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
    )

    scanner_name = sanitize_name(args.held_out_scanner)
    model_name = "augmented" if args.augment_copies > 0 else "baseline"
    output_dir = Path(args.output_dir) / scanner_name / model_name

    for group in ["false_positive", "false_negative", "true_positive", "true_negative"]:
        examples, group_probs, group_preds = select_examples(
            test_indices=test_indices,
            labels=labels,
            probs=probs,
            preds=preds,
            group=group,
            max_examples=args.max_examples,
        )
        title = f"{args.held_out_scanner} {model_name} {group.replace('_', ' ')} n={len(examples)}"
        output_path = output_dir / f"{group}.jpg"
        draw_sheet(cache, examples, group_probs, group_preds, title, output_path, args.cols)
        print(f"{group}: {output_path}")


if __name__ == "__main__":
    main()
