import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics_at_threshold(labels, probabilities, threshold):
    predictions = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def load_fold_ids(path):
    assignments = read_csv(path)
    folds = sorted({row["fold"] for row in assignments})
    by_fold = {}
    for fold in folds:
        fold_rows = [row for row in assignments if row["fold"] == fold]
        by_fold[fold] = {
            role: {int(row["candidate_id"]) for row in fold_rows if row["role"] == role}
            for role in ("train", "val", "test")
        }
        roles = by_fold[fold]
        if roles["train"] & roles["val"] or roles["train"] & roles["test"] or roles["val"] & roles["test"]:
            raise ValueError(f"Candidate leakage in fold {fold}")
    return by_fold


def load_augmented_features(path, baseline_cache, baseline_features, candidate_ids):
    if path is None:
        return None, None
    cache = np.load(path)
    features = cache["features"].astype(np.float32)
    if features.ndim != 3:
        raise ValueError(f"Augmented features must have shape [variants, candidates, features], got {features.shape}")
    checks = {
        "candidate_ids": np.array_equal(cache["candidate_ids"], candidate_ids),
        "labels": np.array_equal(cache["labels"], baseline_cache["labels"]),
        "model_id": str(cache["model_id"][0]) == str(baseline_cache["model_id"][0]),
        "scale": str(cache["scale"][0]) == str(baseline_cache["scale"][0]),
        "pooling": str(cache["pooling"][0]) == str(baseline_cache["pooling"][0]),
        "feature_dimension": features.shape[2] == baseline_features.shape[1],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Augmented cache does not match baseline cache: {', '.join(failed)}")
    augmentation = {
        "name": str(cache["augmentation"][0]),
        "variants": int(features.shape[0]),
        "mode": str(cache["stain_mode"][0]),
        "sigma": float(cache["sigma"][0]),
        "seed": int(cache["seed"][0]),
    }
    return features, augmentation


def run_validation(
    features_path, folds_path, output_dir, c_values, thresholds, seed, augmented_features_path=None
):
    cache = np.load(features_path)
    features = cache["features"].astype(np.float32)
    candidate_ids = cache["candidate_ids"].astype(np.int64)
    labels = cache["labels"].astype(np.int64)
    scale = str(cache["scale"][0])
    model_id = str(cache["model_id"][0])
    if len(set(candidate_ids.tolist())) != len(candidate_ids):
        raise ValueError("Embedding cache contains duplicate candidate IDs")
    augmented_features, augmentation = load_augmented_features(
        augmented_features_path, cache, features, candidate_ids
    )

    id_to_index = {int(candidate_id): index for index, candidate_id in enumerate(candidate_ids)}
    folds = load_fold_ids(folds_path)
    sweep_rows = []
    selected_rows = []
    for fold, roles in folds.items():
        train_indices = np.asarray([id_to_index[value] for value in sorted(roles["train"])], dtype=np.int64)
        val_indices = np.asarray([id_to_index[value] for value in sorted(roles["val"])], dtype=np.int64)
        if set(candidate_ids[train_indices]) & set(candidate_ids[val_indices]):
            raise ValueError(f"Train/validation leakage in fold {fold}")

        x_train, y_train = features[train_indices], labels[train_indices]
        if augmented_features is not None:
            x_train = np.concatenate(
                [x_train] + [variant[train_indices] for variant in augmented_features]
            )
            y_train = np.tile(y_train, augmented_features.shape[0] + 1)
        x_val, y_val = features[val_indices], labels[val_indices]
        best = None
        for c_value in c_values:
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
            probabilities = classifier.predict_proba(x_val)[:, 1]
            pr_auc = float(average_precision_score(y_val, probabilities))
            for threshold in thresholds:
                row = {
                    "model_id": model_id,
                    "scale": scale,
                    "fold": fold,
                    "role": "validation",
                    "c": c_value,
                    "threshold": threshold,
                    "pr_auc": pr_auc,
                    **metrics_at_threshold(y_val, probabilities, threshold),
                    "train_candidates": len(train_indices),
                    "train_feature_rows": len(x_train),
                    "validation_candidates": len(val_indices),
                    "augmentation": augmentation["name"] if augmentation else "none",
                    "seed": seed,
                }
                sweep_rows.append(row)
                selection_key = (row["f1"], row["pr_auc"], -abs(threshold - 0.5), -c_value)
                if best is None or selection_key > best[0]:
                    best = (selection_key, row)
        selected_rows.append(best[1])
        print(json.dumps(best[1], indent=2), flush=True)

    write_csv(output_dir / "validation_sweep.csv", sweep_rows)
    write_csv(output_dir / "selected_validation.csv", selected_rows)
    summary = {
        "model_id": model_id,
        "scale": scale,
        "selection_data": "source-scanner validation only",
        "held_out_test_evaluated": False,
        "training_augmentation": augmentation,
        "mean_validation_precision": float(np.mean([row["precision"] for row in selected_rows])),
        "mean_validation_recall": float(np.mean([row["recall"] for row in selected_rows])),
        "mean_validation_f1": float(np.mean([row["f1"] for row in selected_rows])),
        "mean_validation_pr_auc": float(np.mean([row["pr_auc"] for row in selected_rows])),
        "folds": selected_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"output: {output_dir}", flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Select a frozen linear probe using source validation only.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--folds", type=Path, default=Path("demo2/data/fold_candidates.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--c-values", nargs="+", type=float, default=(0.001, 0.01, 0.1, 1.0, 10.0))
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=tuple(np.round(np.arange(0.1, 0.91, 0.05), 2))
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augmented-features",
        type=Path,
        help="Optional [variant, candidate, feature] cache used only to expand training rows.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_validation(
        args.features,
        args.folds,
        args.output_dir,
        args.c_values,
        args.thresholds,
        args.seed,
        args.augmented_features,
    )


if __name__ == "__main__":
    main()
