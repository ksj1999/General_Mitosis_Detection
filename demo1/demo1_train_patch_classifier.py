import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


LABEL_TO_INDEX = {
    "not_mitotic_figure": 0,
    "mitotic_figure": 1,
}


class PatchDataset(Dataset):
    def __init__(self, patch_dir, rows, transform=None):
        self.patch_dir = Path(patch_dir)
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self.patch_dir / row["patch_path"]
        image = Image.open(image_path).convert("RGB")
        label = LABEL_TO_INDEX[row["label"]]

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class SmallPatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_train_val(rows, val_fraction, seed):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)

    rng = random.Random(seed)
    train_rows = []
    val_rows = []

    for label_rows in grouped.values():
        label_rows = label_rows[:]
        rng.shuffle(label_rows)
        val_count = max(1, int(round(len(label_rows) * val_fraction)))
        val_rows.extend(label_rows[:val_count])
        train_rows.extend(label_rows[val_count:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def make_transforms(train):
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.75, 0.52, 0.72], std=[0.18, 0.20, 0.15]),
            ]
        )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.75, 0.52, 0.72], std=[0.18, 0.20, 0.15]),
        ]
    )


def compute_class_weights(rows, device):
    counts = Counter(LABEL_TO_INDEX[row["label"]] for row in rows)
    total = sum(counts.values())
    weights = []
    for class_index in range(2):
        weights.append(total / (2 * counts[class_index]))
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
            images = images.to(device)
            labels = labels.to(device)
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
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    return total_loss / max(total_samples, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a baseline patch classifier for Demo 1.")
    parser.add_argument("--patch-dir", default="demo1/patch_dataset", help="Patch dataset directory.")
    parser.add_argument("--output-dir", default="demo1/models/patch_classifier", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation fraction from train split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    patch_dir = Path(args.patch_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(patch_dir / "patch_metadata.csv")
    train_source_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    train_rows, val_rows = split_train_val(train_source_rows, args.val_fraction, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    print(f"train rows: {len(train_rows)}", flush=True)
    print(f"val rows: {len(val_rows)}", flush=True)
    print(f"held-out scanner test rows: {len(test_rows)}", flush=True)

    train_loader = DataLoader(
        PatchDataset(patch_dir, train_rows, transform=make_transforms(train=True)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        PatchDataset(patch_dir, val_rows, transform=make_transforms(train=False)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        PatchDataset(patch_dir, test_rows, transform=make_transforms(train=False)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = SmallPatchCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_rows, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
            "test": test_metrics,
        }
        history.append(row)

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
