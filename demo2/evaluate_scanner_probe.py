import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from foundation_models import extract_foundation_features
from lora_evaluation_utils import (
    checkpoint_paths,
    method_name,
    read_manifest,
    restore_lora_probe,
    safe_mean,
    validate_checkpoint_set,
)
from run_linear_probe_validation import load_fold_ids, write_csv
from run_lora_probe_validation import make_loader


@torch.inference_mode()
def extract_embeddings(model, loader, device, description):
    model.eval()
    features = []
    candidate_ids = []
    for pixels, _, batch_ids, _ in tqdm(loader, desc=description, unit="batch"):
        pixels = pixels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            batch_features = extract_foundation_features(model.encoder, pixels, model.spec)
        features.append(batch_features.float().cpu().numpy())
        candidate_ids.extend(batch_ids)
    return np.concatenate(features), np.asarray(candidate_ids, dtype=np.int64)


def probe_metrics(x_train, y_train, x_val, y_val, c_value, seed):
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
            solver="lbfgs",
        ),
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_val)
    probabilities = classifier.predict_proba(x_val)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_val, predictions)),
        "macro_f1": float(f1_score(y_val, predictions, average="macro", zero_division=0)),
        "cross_entropy": float(log_loss(y_val, probabilities, labels=classifier.classes_)),
    }


def evaluate_feature_set(
    features, candidate_ids, manifest, roles, fold, model_seed, probe_seed, method, c_value
):
    id_to_index = {int(candidate_id): index for index, candidate_id in enumerate(candidate_ids)}
    train_indices = np.asarray([id_to_index[value] for value in sorted(roles["train"])])
    val_indices = np.asarray([id_to_index[value] for value in sorted(roles["val"])])
    source_scanners = sorted({manifest[value]["scanner"] for value in roles["train"]})
    scanner_to_index = {scanner: index for index, scanner in enumerate(source_scanners)}
    scanner_labels = np.asarray(
        [scanner_to_index.get(manifest[int(candidate_id)]["scanner"], -1) for candidate_id in candidate_ids]
    )
    mitosis_labels = np.asarray([int(manifest[int(value)]["label"]) for value in candidate_ids])
    rows = []
    for condition, label_value in (("all", None), ("non_mitotic_only", 0), ("mitotic_only", 1)):
        train = train_indices
        val = val_indices
        if label_value is not None:
            train = train[mitosis_labels[train] == label_value]
            val = val[mitosis_labels[val] == label_value]
        metrics = probe_metrics(
            features[train], scanner_labels[train], features[val], scanner_labels[val],
            c_value, probe_seed
        )
        rows.append(
            {
                "method": method,
                "fold": fold,
                "model_seed": model_seed,
                "probe_seed": probe_seed,
                "condition": condition,
                "source_scanners": " | ".join(source_scanners),
                "chance_balanced_accuracy": 0.5,
                **metrics,
                "train_candidates": len(train),
                "validation_candidates": len(val),
            }
        )
    return rows


def evaluate(args):
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest)
    folds = load_fold_ids(args.folds)
    result_rows = []

    if args.frozen_features:
        cache = np.load(args.frozen_features)
        features = cache["features"].astype(np.float32)
        candidate_ids = cache["candidate_ids"].astype(np.int64)
        for fold, roles in folds.items():
            result_rows.extend(
                evaluate_feature_set(
                    features, candidate_ids, manifest, roles, fold, "frozen", args.probe_seed,
                    "frozen_encoder", args.c_value
                )
            )

    for checkpoint_dir in args.checkpoint_dirs:
        records = validate_checkpoint_set(checkpoint_paths(checkpoint_dir))
        for checkpoint_path, metadata in records:
            fold = str(metadata["fold"])
            seed = int(metadata["seed"])
            roles = folds[fold]
            source_ids = sorted(roles["train"] | roles["val"])
            source_rows = [manifest[value] for value in source_ids]
            model, bundle, checkpoint = restore_lora_probe(
                checkpoint_path, args.registry, args.cache_dir, device
            )
            loader = make_loader(
                source_rows,
                args.patch_dir,
                checkpoint["scale"],
                bundle.processor,
                args.batch_size,
                False,
                seed,
                args.num_workers,
            )
            features, candidate_ids = extract_embeddings(
                model, loader, device, f"scanner probe {method_name(checkpoint)} {fold} seed={seed}"
            )
            result_rows.extend(
                evaluate_feature_set(
                    features, candidate_ids, manifest, roles, fold, seed, args.probe_seed,
                    method_name(checkpoint), args.c_value
                )
            )
            del model
            torch.cuda.empty_cache()

    write_csv(args.output_dir / "scanner_probe_results.csv", result_rows)
    methods = sorted({row["method"] for row in result_rows})
    conditions = ("all", "non_mitotic_only", "mitotic_only")
    summary_rows = []
    for method in methods:
        for condition in conditions:
            selected = [
                row for row in result_rows
                if row["method"] == method and row["condition"] == condition
            ]
            summary_rows.append(
                {
                    "method": method,
                    "condition": condition,
                    "mean_balanced_accuracy": safe_mean([row["balanced_accuracy"] for row in selected]),
                    "mean_macro_f1": safe_mean([row["macro_f1"] for row in selected]),
                    "mean_cross_entropy": safe_mean([row["cross_entropy"] for row in selected]),
                    "runs": len(selected),
                }
            )
    write_csv(args.output_dir / "scanner_probe_summary.csv", summary_rows)
    summary = {
        "interpretation": (
            "Lower balanced accuracy toward 0.5 indicates less linearly recoverable information "
            "about the two source scanners; it does not prove invariance to the held-out scanner."
        ),
        "probe_training": "Fresh logistic regression on frozen source-train embeddings",
        "probe_evaluation": "Image-disjoint source-validation embeddings",
        "label_conditioned_controls": True,
        "c": args.c_value,
        "probe_seed": args.probe_seed,
        "summary": summary_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"output: {args.output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit fresh post-hoc scanner probes to frozen LoRA-adapted embeddings."
    )
    parser.add_argument("--checkpoint-dirs", type=Path, nargs="+", default=[])
    parser.add_argument("--frozen-features", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("demo2/data/candidates.csv"))
    parser.add_argument("--folds", type=Path, default=Path("demo2/data/fold_candidates.csv"))
    parser.add_argument("--patch-dir", type=Path, default=Path("demo2/patches"))
    parser.add_argument("--registry", type=Path, default=Path("demo2/configs/foundation_models.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("demo2/.hf_cache"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--probe-seed", type=int, default=42)
    args = parser.parse_args()
    if not args.checkpoint_dirs and not args.frozen_features:
        parser.error("Provide --checkpoint-dirs and/or --frozen-features")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
