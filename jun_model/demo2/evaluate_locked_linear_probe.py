import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_linear_probe_validation import load_fold_ids, metrics_at_threshold


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(features_path, folds_path, config_path, output_dir):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cache = np.load(features_path)
    features = cache["features"].astype(np.float32)
    candidate_ids = cache["candidate_ids"].astype(np.int64)
    labels = cache["labels"].astype(np.int64)
    image_ids = cache["image_ids"].astype(np.int64)
    scanners = cache["scanners"]
    scale = str(cache["scale"][0])
    model_id = str(cache["model_id"][0])
    pooling = str(cache["pooling"][0])
    if scale != config["scale"] or model_id != config["model_id"]:
        raise ValueError(
            f"Locked config expects {config['model_id']} {config['scale']}, got {model_id} {scale}"
        )
    if "pooling" in config and pooling != config["pooling"]:
        raise ValueError(f"Locked config expects pooling {config['pooling']}, got {pooling}")
    if "feature_dimension" in config and features.shape[1] != int(config["feature_dimension"]):
        raise ValueError(
            f"Locked config expects {config['feature_dimension']} features, got {features.shape[1]}"
        )
    metadata_path = features_path.with_suffix(".json")
    if "model_revision" in config:
        if not metadata_path.exists():
            raise FileNotFoundError(f"Locked revision check requires {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_revision") != config["model_revision"]:
            raise ValueError(
                f"Locked config expects revision {config['model_revision']}, "
                f"got {metadata.get('model_revision')}"
            )

    id_to_index = {int(candidate_id): index for index, candidate_id in enumerate(candidate_ids)}
    folds = load_fold_ids(folds_path)
    result_rows = []
    prediction_rows = []
    for fold, fold_config in config["folds"].items():
        roles = folds[fold]
        train_indices = np.asarray([id_to_index[value] for value in sorted(roles["train"])], dtype=np.int64)
        test_indices = np.asarray([id_to_index[value] for value in sorted(roles["test"])], dtype=np.int64)
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(fold_config["c"]),
                class_weight=config["class_weight"],
                max_iter=int(config["max_iter"]),
                random_state=int(config["seed"]),
                solver=config["solver"],
            ),
        )
        classifier.fit(features[train_indices], labels[train_indices])
        probabilities = classifier.predict_proba(features[test_indices])[:, 1]
        threshold = float(fold_config["threshold"])
        fold_metrics = metrics_at_threshold(labels[test_indices], probabilities, threshold)
        row = {
            "model_id": model_id,
            "scale": scale,
            "fold": fold,
            "role": "locked_test",
            "c": float(fold_config["c"]),
            "threshold": threshold,
            "pr_auc": float(average_precision_score(labels[test_indices], probabilities)),
            **fold_metrics,
            "train_candidates": len(train_indices),
            "test_candidates": len(test_indices),
            "seed": int(config["seed"]),
        }
        result_rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

        predictions = probabilities >= threshold
        for index, probability, prediction in zip(test_indices, probabilities, predictions):
            prediction_rows.append(
                {
                    "fold": fold,
                    "candidate_id": int(candidate_ids[index]),
                    "image_id": int(image_ids[index]),
                    "scanner": str(scanners[index]),
                    "label": int(labels[index]),
                    "probability": float(probability),
                    "prediction": int(prediction),
                    "threshold": threshold,
                }
            )

    summary = {
        "protocol_version": config["protocol_version"],
        "model_id": model_id,
        "scale": scale,
        "test_policy": "one locked evaluation after source-validation selection",
        "mean_test_precision": float(np.mean([row["precision"] for row in result_rows])),
        "mean_test_recall": float(np.mean([row["recall"] for row in result_rows])),
        "mean_test_f1": float(np.mean([row["f1"] for row in result_rows])),
        "mean_test_pr_auc": float(np.mean([row["pr_auc"] for row in result_rows])),
        "folds": result_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "locked_test_results.csv", result_rows)
    write_csv(output_dir / "locked_test_predictions.csv", prediction_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"output: {output_dir}", flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a locked frozen linear probe once on held-out scanners.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--folds", type=Path, default=Path("jun_model/demo2/data/fold_candidates.csv"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(args.features, args.folds, args.config, args.output_dir)


if __name__ == "__main__":
    main()
