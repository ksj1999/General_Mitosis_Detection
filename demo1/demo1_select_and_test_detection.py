"""Select a detector checkpoint/threshold on validation data, then test once."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def checkpoint_epoch(path):
    match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


def find_checkpoints(model_dir):
    checkpoints = sorted(model_dir.glob("checkpoint_epoch_*.pt"), key=lambda path: checkpoint_epoch(path) or 0)
    if checkpoints:
        return checkpoints
    return [model_dir / name for name in ["best.pt", "final.pt"] if (model_dir / name).exists()]


def run_evaluation(args, checkpoint, coco, output_dir, score_thresholds):
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        return metrics_path

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "demo1/demo1_evaluate_detection_frcnn.py",
        "--checkpoint",
        str(checkpoint),
        "--coco",
        str(coco),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--iou-threshold",
        str(args.iou_threshold),
        "--score-thresholds",
        score_thresholds,
        "--normalization",
        args.normalization,
    ]
    print("\n" + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)
    return metrics_path


def load_best(metrics_path):
    with metrics_path.open(encoding="utf-8") as f:
        return json.load(f)["best"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--val-coco", type=Path, required=True)
    parser.add_argument("--test-coco", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--score-thresholds", default="0.05,0.1,0.2,0.3,0.5,0.7")
    parser.add_argument("--normalization", choices=["none", "reinhard_rgb"], default="none")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoints = find_checkpoints(args.model_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {args.model_dir}")
    for path in [args.val_coco, args.test_coco]:
        if not path.exists():
            raise FileNotFoundError(path)

    candidates = []
    for checkpoint in checkpoints:
        metrics_path = run_evaluation(
            args,
            checkpoint,
            args.val_coco,
            args.output_dir / "validation" / checkpoint.stem,
            args.score_thresholds,
        )
        candidates.append(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_name": checkpoint.name,
                "epoch": checkpoint_epoch(checkpoint),
                "validation": load_best(metrics_path),
            }
        )

    selected = max(candidates, key=lambda row: row["validation"]["f1"])
    selected_checkpoint = Path(selected["checkpoint"])
    selected_threshold = float(selected["validation"]["score_threshold"])

    test_metrics_path = run_evaluation(
        args,
        selected_checkpoint,
        args.test_coco,
        args.output_dir / "test",
        str(selected_threshold),
    )
    test_result = load_best(test_metrics_path)

    summary = {
        "selection_data": str(args.val_coco),
        "test_data": str(args.test_coco),
        "selection_candidates": candidates,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_epoch": selected["epoch"],
        "selected_score_threshold": selected_threshold,
        "validation_result": selected["validation"],
        "test_result_locked": test_result,
        "normalization": args.normalization,
        "iou_threshold": args.iou_threshold,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "protocol_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown = [
        "# Locked Detection Validation Summary",
        "",
        f"- Selected checkpoint: `{selected_checkpoint.name}`",
        f"- Selected score threshold: `{selected_threshold:.3f}`",
        f"- Validation F1: `{selected['validation']['f1']:.3f}`",
        f"- Locked test precision: `{test_result['precision']:.3f}`",
        f"- Locked test recall: `{test_result['recall']:.3f}`",
        f"- Locked test F1: `{test_result['f1']:.3f}`",
        "",
        "The checkpoint and score threshold were selected using validation data only.",
    ]
    (args.output_dir / "protocol_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"protocol summary: {summary_path}")


if __name__ == "__main__":
    main()
