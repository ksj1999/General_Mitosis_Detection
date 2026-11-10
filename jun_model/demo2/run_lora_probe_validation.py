import argparse
import csv
import json
import os
import random
from pathlib import Path

# PEFT imports Transformers eagerly, so set project-local caches before importing it.
_DEFAULT_HF_CACHE = Path("jun_model/demo2/.hf_cache").resolve()
os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_CACHE))
os.environ.setdefault("HF_MODULES_CACHE", str(_DEFAULT_HF_CACHE / "modules"))

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from foundation_models import extract_foundation_features, load_foundation_model
from run_adversarial_probe_validation import balanced_class_weights, gradient_reverse
from run_linear_probe_validation import load_fold_ids, metrics_at_threshold, write_csv
from stain_augmentation import STAIN_MODES, hed_jitter


class CandidatePatchDataset(Dataset):
    def __init__(self, rows, patch_dir, scale):
        self.rows = rows
        self.patch_dir = patch_dir
        self.scale = scale

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.patch_dir / self.scale / f"{int(row['candidate_id']):05d}.jpg"
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return pixels, int(row["label"]), int(row["candidate_id"]), row["scanner"]


class LoRAProbe(nn.Module):
    def __init__(self, encoder, spec, feature_dimension, dropout, scanner_classes=0, scanner_hidden_dim=256):
        super().__init__()
        self.encoder = encoder
        self.spec = spec
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dimension, 1)
        self.scanner_head = None
        if scanner_classes:
            self.scanner_head = nn.Sequential(
                nn.Linear(feature_dimension, scanner_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(scanner_hidden_dim, scanner_classes),
            )

    def forward(self, pixels, grl_strength=0.0):
        features = extract_foundation_features(self.encoder, pixels, self.spec)
        mitosis_logits = self.classifier(self.dropout(features)).squeeze(1)
        scanner_logits = None
        if self.scanner_head is not None:
            scanner_logits = self.scanner_head(gradient_reverse(features, grl_strength))
        return mitosis_logits, scanner_logits


def read_manifest(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["candidate_id"]): row for row in rows}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    rows,
    patch_dir,
    scale,
    processor,
    batch_size,
    shuffle,
    seed,
    workers,
    stain_augmentation=False,
    stain_sigma=0.05,
    stain_mode="both",
    scanner_to_index=None,
):
    dataset = CandidatePatchDataset(rows, patch_dir, scale)

    def collate(batch):
        images, labels, candidate_ids, scanners = zip(*batch)
        if stain_augmentation:
            raw = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div_(255)
            augmented = hed_jitter(raw, stain_sigma, stain_mode)
            images = augmented.mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
        pil_images = [Image.fromarray(image, mode="RGB") for image in images]
        pixels = processor(pil_images)
        scanner_ids = [-1] * len(scanners)
        if scanner_to_index is not None:
            scanner_ids = [scanner_to_index[name] for name in scanners]
        return (
            pixels,
            torch.tensor(labels, dtype=torch.float32),
            candidate_ids,
            torch.tensor(scanner_ids, dtype=torch.long),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )


def build_model(args, device, scanner_classes=0):
    bundle = load_foundation_model(args.model_key, args.registry, args.cache_dir)
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )
    encoder = get_peft_model(bundle.model, config)
    feature_dimension = int(bundle.spec["expected_feature_dimension"])
    model = LoRAProbe(
        encoder,
        bundle.spec,
        feature_dimension,
        args.head_dropout,
        scanner_classes,
        args.scanner_hidden_dim,
    ).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return model, bundle, trainable, total


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels = []
    probabilities = []
    candidate_ids = []
    for pixels, batch_labels, batch_ids, _ in loader:
        pixels = pixels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _ = model(pixels)
        probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels.extend(batch_labels.numpy().astype(np.int64).tolist())
        candidate_ids.extend(batch_ids)
    return np.asarray(labels), np.asarray(probabilities), np.asarray(candidate_ids)


def train_fold(args, fold, train_rows, val_rows, seed, device):
    set_seed(seed)
    source_scanners = sorted({row["scanner"] for row in train_rows})
    scanner_to_index = {name: index for index, name in enumerate(source_scanners)}
    scanner_classes = len(source_scanners) if args.grl_strength > 0 else 0
    model, bundle, trainable, total = build_model(args, device, scanner_classes)
    train_loader = make_loader(
        train_rows,
        args.patch_dir,
        args.scale,
        bundle.processor,
        args.batch_size,
        True,
        seed,
        args.num_workers,
        args.stain_augmentation,
        args.stain_sigma,
        args.stain_mode,
        scanner_to_index,
    )
    val_loader = make_loader(
        val_rows,
        args.patch_dir,
        args.scale,
        bundle.processor,
        args.eval_batch_size,
        False,
        seed,
        args.num_workers,
        scanner_to_index=scanner_to_index,
    )
    train_labels = np.asarray([int(row["label"]) for row in train_rows])
    positives = float(np.sum(train_labels == 1))
    negatives = float(np.sum(train_labels == 0))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, device=device)
    )
    scanner_criterion = None
    if args.grl_strength > 0:
        scanner_labels = np.asarray([scanner_to_index[row["scanner"]] for row in train_rows])
        scanner_criterion = nn.CrossEntropyLoss(
            weight=balanced_class_weights(scanner_labels, scanner_classes).to(device)
        )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_ap = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    epochs = tqdm(range(1, args.epochs + 1), desc=f"{fold} seed={seed}", unit="epoch")
    for epoch in epochs:
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (pixels, labels, _, scanners) in enumerate(train_loader, start=1):
            pixels = pixels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            scanners = scanners.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, scanner_logits = model(pixels, args.grl_strength)
                loss = criterion(logits, labels)
                if scanner_criterion is not None:
                    loss = loss + scanner_criterion(scanner_logits, scanners)
            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            if (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            ):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))

        val_labels, val_probabilities, _ = predict(model, val_loader, device)
        val_ap = float(average_precision_score(val_labels, val_probabilities))
        epochs.set_postfix(loss=f"{np.mean(losses):.4f}", val_ap=f"{val_ap:.4f}")
        if val_ap > best_ap + 1e-5:
            best_ap = val_ap
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
                if "lora_" in name
                or name.startswith("classifier.")
                or name.startswith("scanner_head.")
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    current = model.state_dict()
    current.update(best_state)
    model.load_state_dict(current)
    labels, probabilities, candidate_ids = predict(model, val_loader, device)
    return {
        "model": model,
        "best_state": best_state,
        "labels": labels,
        "probabilities": probabilities,
        "candidate_ids": candidate_ids,
        "best_epoch": best_epoch,
        "pr_auc": best_ap,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "model_id": bundle.spec["repository"],
        "model_revision": bundle.revision,
    }


def run_validation(args):
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest)
    folds = load_fold_ids(args.folds)
    if args.folds_to_run:
        unknown = sorted(set(args.folds_to_run) - set(folds))
        if unknown:
            raise ValueError(f"Unknown folds: {', '.join(unknown)}")
        folds = {name: folds[name] for name in args.folds_to_run}
    rows = []
    for seed in args.seeds:
        for fold, roles in folds.items():
            train_ids = sorted(roles["train"])
            val_ids = sorted(roles["val"])
            if args.max_train_candidates:
                train_ids = train_ids[: args.max_train_candidates]
            if args.max_val_candidates:
                val_ids = val_ids[: args.max_val_candidates]
            train_rows = [manifest[value] for value in train_ids]
            val_rows = [manifest[value] for value in val_ids]
            result = train_fold(args, fold, train_rows, val_rows, seed, device)
            best = None
            for threshold in args.thresholds:
                metrics = metrics_at_threshold(result["labels"], result["probabilities"], threshold)
                key = (metrics["f1"], result["pr_auc"], -abs(threshold - 0.5))
                if best is None or key > best[0]:
                    best = (key, threshold, metrics)
            row = {
                "model_id": result["model_id"],
                "model_revision": result["model_revision"],
                "scale": args.scale,
                "fold": fold,
                "role": "validation",
                "method": "lora_probe",
                "seed": seed,
                "threshold": best[1],
                "best_epoch": result["best_epoch"],
                "pr_auc": result["pr_auc"],
                **best[2],
                "train_candidates": len(train_rows),
                "validation_candidates": len(val_rows),
                "trainable_parameters": result["trainable_parameters"],
                "total_parameters": result["total_parameters"],
                "grl_strength": args.grl_strength,
            }
            if args.stain_augmentation:
                row["method"] = "lora_stain_probe"
            if args.grl_strength > 0:
                row["method"] = "lora_stain_grl" if args.stain_augmentation else "lora_grl"
            rows.append(row)
            checkpoint = {
                "state_dict": result["best_state"],
                "model_id": result["model_id"],
                "model_revision": result["model_revision"],
                "model_key": args.model_key,
                "scale": args.scale,
                "fold": fold,
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "threshold": best[1],
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "target_modules": args.target_modules,
                "head_dropout": args.head_dropout,
                "stain_augmentation": args.stain_augmentation,
                "stain_sigma": args.stain_sigma,
                "stain_mode": args.stain_mode,
                "grl_strength": args.grl_strength,
                "scanner_hidden_dim": args.scanner_hidden_dim,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
            }
            torch.save(checkpoint, args.output_dir / f"{fold}_seed{seed}_adapter.pt")
            print(json.dumps(row), flush=True)
            del result["model"]
            torch.cuda.empty_cache()

    write_csv(args.output_dir / "validation_runs.csv", rows)
    per_seed_f1 = [
        float(np.mean([row["f1"] for row in rows if row["seed"] == seed]))
        for seed in args.seeds
    ]
    per_seed_ap = [
        float(np.mean([row["pr_auc"] for row in rows if row["seed"] == seed]))
        for seed in args.seeds
    ]
    summary = {
        "model_id": rows[0]["model_id"],
        "model_revision": rows[0]["model_revision"],
        "scale": args.scale,
        "method": (
            "lora_stain_grl"
            if args.stain_augmentation and args.grl_strength > 0
            else "lora_grl"
            if args.grl_strength > 0
            else "lora_stain_probe"
            if args.stain_augmentation
            else "lora_probe"
        ),
        "selection_data": "source-scanner validation only",
        "held_out_test_evaluated": False,
        "seeds": args.seeds,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.target_modules,
            "trainable_parameters": rows[0]["trainable_parameters"],
            "total_parameters": rows[0]["total_parameters"],
        },
        "training": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "stain_augmentation": args.stain_augmentation,
            "stain_sigma": args.stain_sigma if args.stain_augmentation else None,
            "stain_mode": args.stain_mode if args.stain_augmentation else None,
            "grl_strength": args.grl_strength,
            "scanner_hidden_dim": args.scanner_hidden_dim if args.grl_strength > 0 else None,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        },
        "mean_validation_precision": float(np.mean([row["precision"] for row in rows])),
        "mean_validation_recall": float(np.mean([row["recall"] for row in rows])),
        "mean_validation_f1": float(np.mean(per_seed_f1)),
        "std_validation_f1_across_seeds": float(np.std(per_seed_f1)),
        "mean_validation_pr_auc": float(np.mean(per_seed_ap)),
        "std_validation_pr_auc_across_seeds": float(np.std(per_seed_ap)),
        "seed_macro_f1": per_seed_f1,
        "seed_macro_pr_auc": per_seed_ap,
        "fold_runs": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"output: {args.output_dir}", flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a pathology encoder with LoRA using source validation only.")
    parser.add_argument("--model-key", choices=("phikon_v2", "hibou_b", "virchow2"), default="phikon_v2")
    parser.add_argument("--manifest", type=Path, default=Path("jun_model/demo2/data/candidates.csv"))
    parser.add_argument("--folds", type=Path, default=Path("jun_model/demo2/data/fold_candidates.csv"))
    parser.add_argument("--folds-to-run", nargs="+")
    parser.add_argument("--patch-dir", type=Path, default=Path("jun_model/demo2/patches"))
    parser.add_argument("--scale", choices=("20x", "40x"), default="40x")
    parser.add_argument("--registry", type=Path, default=Path("jun_model/demo2/configs/foundation_models.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("jun_model/demo2/.hf_cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0,))
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=("query", "value"))
    parser.add_argument("--head-dropout", type=float, default=0.2)
    parser.add_argument("--stain-augmentation", action="store_true")
    parser.add_argument("--stain-sigma", type=float, default=0.05)
    parser.add_argument("--stain-mode", choices=STAIN_MODES, default="both")
    parser.add_argument("--grl-strength", type=float, default=0.0)
    parser.add_argument("--scanner-hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--thresholds", nargs="+", type=float, default=tuple(np.round(np.arange(0.1, 0.91, 0.05), 2)))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-train-candidates", type=int)
    parser.add_argument("--max-val-candidates", type=int)
    args = parser.parse_args()
    if not 0 <= args.stain_sigma <= 1:
        parser.error("--stain-sigma must be between 0 and 1")
    if args.grl_strength < 0:
        parser.error("--grl-strength must be non-negative")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be at least 1")
    return args


if __name__ == "__main__":
    run_validation(parse_args())
