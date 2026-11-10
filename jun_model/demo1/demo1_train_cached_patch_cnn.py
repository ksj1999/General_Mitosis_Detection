import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


LABEL_TO_INDEX = {
    "not_mitotic_figure": 0,
    "mitotic_figure": 1,
}


MEAN = torch.tensor([0.75, 0.52, 0.72], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([0.18, 0.20, 0.15], dtype=torch.float32).view(3, 1, 1)


class CachedPatchDataset(Dataset):
    def __init__(self, images, labels, indices, augment=False):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        image = torch.from_numpy(self.images[source_index]).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(int(self.labels[source_index]), dtype=torch.long)

        if self.augment:
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(2,))
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(1,))

            brightness = 0.90 + 0.20 * torch.rand(())
            contrast = 0.90 + 0.20 * torch.rand(())
            channel_scale = 0.92 + 0.16 * torch.rand((3, 1, 1))
            image = image * brightness
            image_mean = image.mean(dim=(1, 2), keepdim=True)
            image = (image - image_mean) * contrast + image_mean
            image = image * channel_scale
            image = torch.clamp(image, 0.0, 1.0)

        image = (image - MEAN) / STD
        return image, label


class SmallPatchCNN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(width * 4, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def split_train_val(train_indices, labels, val_fraction, seed):
    rng = random.Random(seed)
    grouped = {}
    for index in train_indices:
        grouped.setdefault(int(labels[index]), []).append(index)

    train = []
    val = []
    for class_indices in grouped.values():
        class_indices = class_indices[:]
        rng.shuffle(class_indices)
        val_count = max(1, int(round(len(class_indices) * val_fraction)))
        val.extend(class_indices[:val_count])
        train.extend(class_indices[val_count:])

    rng.shuffle(train)
    rng.shuffle(val)
    return np.asarray(train), np.asarray(val)


def compute_class_weights(labels, indices, device):
    counts = Counter(int(labels[index]) for index in indices)
    total = sum(counts.values())
    weights = [total / (2 * counts[class_index]) for class_index in range(2)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def metrics_from_counts(tp, fp, tn, fn):
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


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            tp += int(((preds == 1) & (labels == 1)).sum().item())
            fp += int(((preds == 1) & (labels == 0)).sum().item())
            tn += int(((preds == 0) & (labels == 0)).sum().item())
            fn += int(((preds == 0) & (labels == 1)).sum().item())

    metrics = metrics_from_counts(tp, fp, tn, fn)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
    return total_loss / max(total_samples, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a cached CNN patch classifier for Demo 1.")
    parser.add_argument("--cache", default="jun_model/demo1/patch_dataset/patch_cache.npz", help="Cached patch dataset.")
    parser.add_argument("--output-dir", default="jun_model/demo1/models/cached_patch_cnn", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation fraction from train split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-augment", action="store_true", help="Disable train-time augmentation.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Training device.")
    parser.add_argument("--width", type=int, default=8, help="Base CNN channel width.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache = np.load(args.cache)
    images = cache["images"]
    labels = cache["labels"]
    splits = cache["splits"]

    source_train_indices = np.where(splits == "train")[0]
    test_indices = np.where(splits == "test")[0]
    train_indices, val_indices = split_train_val(source_train_indices.tolist(), labels, args.val_fraction, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}", flush=True)
    print(f"train rows: {len(train_indices)}", flush=True)
    print(f"val rows: {len(val_indices)}", flush=True)
    print(f"held-out scanner test rows: {len(test_indices)}", flush=True)
    print(f"augment: {not args.no_augment}", flush=True)

    train_loader = DataLoader(
        CachedPatchDataset(images, labels, train_indices, augment=not args.no_augment),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        CachedPatchDataset(images, labels, val_indices, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        CachedPatchDataset(images, labels, test_indices, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = SmallPatchCNN(width=args.width).to(device)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(labels, train_indices, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics, "test": test_metrics})

        print(
            f"epoch {epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"test_f1={test_metrics['f1']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f}",
            flush=True,
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_to_index": LABEL_TO_INDEX,
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                },
                output_dir / "best_model.pt",
            )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"best model: {output_dir / 'best_model.pt'}", flush=True)
    print(f"metrics: {output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
