import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from bootstrap_locked_metrics import bootstrap
from lora_evaluation_utils import (
    checkpoint_paths,
    method_name,
    read_manifest,
    restore_lora_probe,
    safe_mean,
    validate_checkpoint_set,
)
from run_linear_probe_validation import load_fold_ids, metrics_at_threshold, write_csv
from run_lora_probe_validation import make_loader, predict


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
    paths = [path for directory in args.checkpoint_dirs for path in checkpoint_paths(directory)]
    records = validate_checkpoint_set(paths)

    result_rows = []
    prediction_rows = []
    for checkpoint_path, metadata in records:
        fold = str(metadata["fold"])
        seed = int(metadata["seed"])
        if fold not in folds:
            raise ValueError(f"Checkpoint {checkpoint_path} has unknown fold {fold}")
        test_ids = sorted(folds[fold]["test"])
        test_rows = [manifest[candidate_id] for candidate_id in test_ids]
        model, bundle, checkpoint = restore_lora_probe(
            checkpoint_path, args.registry, args.cache_dir, device
        )
        if checkpoint["scale"] != args.scale:
            raise ValueError(f"Checkpoint {checkpoint_path} uses {checkpoint['scale']}, not {args.scale}")
        loader = make_loader(
            test_rows,
            args.patch_dir,
            args.scale,
            bundle.processor,
            args.batch_size,
            False,
            seed,
            args.num_workers,
        )
        labels, probabilities, candidate_ids = predict(model, loader, device)
        threshold = float(checkpoint["threshold"])
        metrics = metrics_at_threshold(labels, probabilities, threshold)
        method = method_name(checkpoint)
        row = {
            "model_id": checkpoint["model_id"],
            "scale": checkpoint["scale"],
            "method": method,
            "fold": fold,
            "held_out_scanner": test_rows[0]["scanner"],
            "role": "posthoc_heldout_test",
            "seed": seed,
            "threshold": threshold,
            "best_epoch": int(checkpoint["best_epoch"]),
            "pr_auc": float(average_precision_score(labels, probabilities)),
            **metrics,
            "test_candidates": len(test_rows),
        }
        result_rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

        for candidate_id, label, probability in zip(candidate_ids, labels, probabilities):
            source = manifest[int(candidate_id)]
            prediction_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "fold": fold,
                    "candidate_id": int(candidate_id),
                    "image_id": int(source["image_id"]),
                    "scanner": source["scanner"],
                    "label": int(label),
                    "probability": float(probability),
                    "prediction": int(probability >= threshold),
                    "threshold": threshold,
                }
            )
        del model
        torch.cuda.empty_cache()

    write_csv(args.output_dir / "posthoc_heldout_results.csv", result_rows)
    write_csv(args.output_dir / "posthoc_heldout_predictions.csv", prediction_rows)

    by_seed = defaultdict(list)
    for row in result_rows:
        by_seed[int(row["seed"])].append(row)
    seed_summaries = []
    for seed, rows in sorted(by_seed.items()):
        seed_predictions = [row for row in prediction_rows if int(row["seed"]) == seed]
        prediction_path = args.output_dir / f"posthoc_heldout_predictions_seed{seed}.csv"
        write_csv(prediction_path, seed_predictions)
        intervals = bootstrap(prediction_path, args.bootstrap_samples, args.bootstrap_seed + seed)
        (args.output_dir / f"bootstrap_seed{seed}.json").write_text(
            json.dumps(intervals, indent=2), encoding="utf-8"
        )
        seed_summaries.append(
            {
                "seed": seed,
                "macro_precision": safe_mean([row["precision"] for row in rows]),
                "macro_recall": safe_mean([row["recall"] for row in rows]),
                "macro_f1": safe_mean([row["f1"] for row in rows]),
                "macro_pr_auc": safe_mean([row["pr_auc"] for row in rows]),
                "image_bootstrap": intervals["macro_mean"],
            }
        )

    summary = {
        "analysis_status": "post-hoc; not an untouched confirmatory test",
        "reason": "The original held-out scanner results were viewed before LoRA development.",
        "selection_policy": "Checkpoint epoch and threshold were selected only on source-scanner validation data.",
        "checkpoint_dirs": [str(path) for path in args.checkpoint_dirs],
        "seeds": seed_summaries,
        "mean_seed_macro_f1": safe_mean([row["macro_f1"] for row in seed_summaries]),
        "std_seed_macro_f1": float(np.std([row["macro_f1"] for row in seed_summaries])),
        "mean_seed_macro_pr_auc": safe_mean([row["macro_pr_auc"] for row in seed_summaries]),
        "std_seed_macro_pr_auc": float(np.std([row["macro_pr_auc"] for row in seed_summaries])),
        "fold_runs": result_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"output: {args.output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-hoc evaluation of source-selected LoRA checkpoints on held-out scanners."
    )
    parser.add_argument("--checkpoint-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("demo2/data/candidates.csv"))
    parser.add_argument("--folds", type=Path, default=Path("demo2/data/fold_candidates.csv"))
    parser.add_argument("--patch-dir", type=Path, default=Path("demo2/patches"))
    parser.add_argument("--scale", choices=("20x", "40x"), default="40x")
    parser.add_argument("--registry", type=Path, default=Path("demo2/configs/foundation_models.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("demo2/.hf_cache"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
