import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from run_linear_probe_validation import (
    load_augmented_features,
    load_fold_ids,
    metrics_at_threshold,
    write_csv,
)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, strength):
        ctx.strength = strength
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradients):
        return -ctx.strength * gradients, None


def gradient_reverse(inputs, strength):
    return GradientReversalFunction.apply(inputs, strength)


class AdversarialProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim, scanner_classes, dropout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mitosis_head = nn.Linear(hidden_dim, 1)
        self.scanner_head = nn.Sequential(
            nn.Linear(hidden_dim, max(32, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, hidden_dim // 2), scanner_classes),
        )

    def forward(self, features, grl_strength=0.0):
        representation = self.encoder(features)
        mitosis_logits = self.mitosis_head(representation).squeeze(1)
        scanner_logits = self.scanner_head(gradient_reverse(representation, grl_strength))
        return mitosis_logits, scanner_logits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_class_weights(labels, class_count):
    counts = np.bincount(labels, minlength=class_count).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError("Every scanner class must occur in the training split")
    weights = len(labels) / (class_count * counts)
    return torch.from_numpy(weights)


@torch.no_grad()
def predict(model, features, device, batch_size):
    model.eval()
    probabilities = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        logits, _ = model(batch)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def train_one(
    x_train,
    y_train,
    scanner_train,
    x_val,
    y_val,
    scanner_count,
    grl_strength,
    seed,
    device,
    hidden_dim,
    dropout,
    learning_rate,
    weight_decay,
    batch_size,
    epochs,
    patience,
):
    set_seed(seed)
    model = AdversarialProbe(x_train.shape[1], hidden_dim, scanner_count, dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    mitosis_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, device=device)
    )
    scanner_loss = nn.CrossEntropyLoss(
        weight=balanced_class_weights(scanner_train, scanner_count).to(device)
    )
    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train.astype(np.float32)),
        torch.from_numpy(scanner_train),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    best_ap = -1.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_features, batch_labels, batch_scanners in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_scanners = batch_scanners.to(device)
            optimizer.zero_grad(set_to_none=True)
            mitosis_logits, scanner_logits = model(batch_features, grl_strength)
            loss = mitosis_loss(mitosis_logits, batch_labels)
            if grl_strength > 0:
                loss = loss + scanner_loss(scanner_logits, batch_scanners)
            loss.backward()
            optimizer.step()

        probabilities = predict(model, x_val, device, batch_size * 4)
        validation_ap = float(average_precision_score(y_val, probabilities))
        if validation_ap > best_ap + 1e-6:
            best_ap = validation_ap
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_ap


def run_validation(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    cache = np.load(args.features)
    features = cache["features"].astype(np.float32)
    labels = cache["labels"].astype(np.int64)
    candidate_ids = cache["candidate_ids"].astype(np.int64)
    scanners = cache["scanners"].astype(str)
    model_id = str(cache["model_id"][0])
    scale = str(cache["scale"][0])
    if len(np.unique(candidate_ids)) != len(candidate_ids):
        raise ValueError("Embedding cache contains duplicate candidate IDs")
    augmented_features, augmentation = load_augmented_features(
        args.augmented_features, cache, features, candidate_ids
    )

    def method_name(strength):
        if augmentation:
            return "stain_augmented_neural" if strength == 0 else "stain_augmented_grl"
        return "neural_baseline" if strength == 0 else "scanner_adversarial_grl"

    id_to_index = {int(value): index for index, value in enumerate(candidate_ids)}
    folds = load_fold_ids(args.folds)
    jobs = [
        (strength, seed, fold, roles)
        for strength in args.grl_strengths
        for seed in args.seeds
        for fold, roles in folds.items()
    ]
    rows = []
    progress = tqdm(jobs, desc="neural/GRL validation", unit="run")
    for strength, seed, fold, roles in progress:
        progress.set_postfix(fold=fold, grl=strength, seed=seed)
        train_indices = np.asarray([id_to_index[value] for value in sorted(roles["train"])])
        val_indices = np.asarray([id_to_index[value] for value in sorted(roles["val"])])
        source_scanners = sorted(np.unique(scanners[train_indices]).tolist())
        scanner_to_index = {name: index for index, name in enumerate(source_scanners)}
        scanner_train = np.asarray(
            [scanner_to_index[name] for name in scanners[train_indices]], dtype=np.int64
        )
        x_train = features[train_indices]
        y_train = labels[train_indices]
        if augmented_features is not None:
            x_train = np.concatenate(
                [x_train] + [variant[train_indices] for variant in augmented_features]
            )
            y_train = np.tile(y_train, augmented_features.shape[0] + 1)
            scanner_train = np.tile(scanner_train, augmented_features.shape[0] + 1)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_val = scaler.transform(features[val_indices]).astype(np.float32)
        y_val = labels[val_indices]
        model, best_epoch, pr_auc = train_one(
            x_train,
            y_train,
            scanner_train,
            x_val,
            y_val,
            len(source_scanners),
            strength,
            seed,
            device,
            args.hidden_dim,
            args.dropout,
            args.learning_rate,
            args.weight_decay,
            args.batch_size,
            args.epochs,
            args.patience,
        )
        probabilities = predict(model, x_val, device, args.batch_size * 4)
        best = None
        for threshold in args.thresholds:
            metrics = metrics_at_threshold(y_val, probabilities, threshold)
            key = (metrics["f1"], pr_auc, -abs(threshold - 0.5))
            if best is None or key > best[0]:
                best = (key, threshold, metrics)
        row = {
            "model_id": model_id,
            "scale": scale,
            "fold": fold,
            "role": "validation",
            "method": method_name(strength),
            "grl_strength": strength,
            "seed": seed,
            "threshold": best[1],
            "best_epoch": best_epoch,
            "pr_auc": pr_auc,
            **best[2],
            "train_candidates": len(train_indices),
            "train_feature_rows": len(x_train),
            "validation_candidates": len(val_indices),
            "source_scanners": ";".join(source_scanners),
            "augmentation": augmentation["name"] if augmentation else "none",
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "validation_runs.csv", rows)
    strengths = []
    for strength in args.grl_strengths:
        selected = [row for row in rows if row["grl_strength"] == strength]
        per_seed_f1 = [
            float(np.mean([row["f1"] for row in selected if row["seed"] == seed]))
            for seed in args.seeds
        ]
        per_seed_ap = [
            float(np.mean([row["pr_auc"] for row in selected if row["seed"] == seed]))
            for seed in args.seeds
        ]
        strengths.append(
            {
                "method": method_name(strength),
                "grl_strength": strength,
                "mean_validation_precision": float(np.mean([row["precision"] for row in selected])),
                "mean_validation_recall": float(np.mean([row["recall"] for row in selected])),
                "mean_validation_f1": float(np.mean(per_seed_f1)),
                "std_validation_f1_across_seeds": float(np.std(per_seed_f1)),
                "mean_validation_pr_auc": float(np.mean(per_seed_ap)),
                "std_validation_pr_auc_across_seeds": float(np.std(per_seed_ap)),
                "seed_macro_f1": per_seed_f1,
                "seed_macro_pr_auc": per_seed_ap,
            }
        )
    best = max(strengths, key=lambda row: (row["mean_validation_f1"], row["mean_validation_pr_auc"]))
    summary = {
        "model_id": model_id,
        "scale": scale,
        "device": str(device),
        "selection_data": "source-scanner validation only",
        "held_out_test_evaluated": False,
        "training_augmentation": augmentation,
        "seeds": args.seeds,
        "architecture": {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
        },
        "strengths": strengths,
        "selected_grl_strength": best["grl_strength"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"output: {args.output_dir}", flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Compare a neural probe with scanner-adversarial GRL on source validation.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--augmented-features",
        type=Path,
        help="Optional augmented cache used only to expand training rows; validation stays plain.",
    )
    parser.add_argument("--folds", type=Path, default=Path("demo2/data/fold_candidates.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grl-strengths", nargs="+", type=float, default=(0.0, 0.01, 0.1, 0.5))
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--thresholds", nargs="+", type=float, default=tuple(np.round(np.arange(0.1, 0.91, 0.05), 2)))
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run_validation(parse_args())
