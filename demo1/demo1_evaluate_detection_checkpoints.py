import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


RUNS = [
    {
        "holdout": "Aperio_CS2",
        "models": {
            "baseline": "demo1/models/detection_frcnn_aperio_b4",
            "color_jitter": "demo1/models/detection_frcnn_aperio_color_jitter",
        },
    },
    {
        "holdout": "Hamamatsu_S360",
        "models": {
            "baseline": "demo1/models/detection_frcnn_Hamamatsu_S360",
            "color_jitter": "demo1/models/detection_frcnn_Hamamatsu_S360_color_jitter",
        },
    },
    {
        "holdout": "Hamammatsu_XR",
        "models": {
            "baseline": "demo1/models/detection_frcnn_Hamammatsu_XR",
            "color_jitter": "demo1/models/detection_frcnn_Hamammatsu_XR_color_jitter",
        },
    },
]


def checkpoint_epoch(path):
    match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1))
    return None


def checkpoints_for_model(model_dir, include_final):
    checkpoints = sorted(
        [path for path in model_dir.glob("checkpoint_epoch_*.pt") if checkpoint_epoch(path) is not None],
        key=checkpoint_epoch,
    )
    if include_final and (model_dir / "final.pt").exists():
        checkpoints.append(model_dir / "final.pt")
    return checkpoints


def best_metrics(metrics_csv):
    with metrics_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return max(rows, key=lambda row: float(row["f1"]))


def run_eval(args, checkpoint, coco, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "metrics.csv"
    if metrics_csv.exists() and not args.force:
        print(f"Skipping existing eval: {metrics_csv}", flush=True)
        return

    cmd = [
        sys.executable,
        "demo1/demo1_evaluate_detection_frcnn.py",
        "--checkpoint",
        checkpoint,
        "--coco",
        coco,
        "--output-dir",
        output_dir,
        "--device",
        args.device,
        "--batch-size",
        args.batch_size,
        "--num-workers",
        args.num_workers,
        "--iou-threshold",
        args.iou_threshold,
        "--score-thresholds",
        args.score_thresholds,
        "--normalization",
        args.normalization,
    ]
    print("\n" + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], check=True)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "holdout",
        "model",
        "checkpoint",
        "epoch",
        "score_threshold",
        "iou_threshold",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "metrics_csv",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    lines = [
        "# Detection Checkpoint Sweep Summary",
        "",
        "| Held-out scanner | Model | Best checkpoint | Score | Precision | Recall | F1 | TP | FP | FN |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    best_rows = {}
    for row in rows:
        key = (row["holdout"], row["model"])
        if key not in best_rows or float(row["f1"]) > float(best_rows[key]["f1"]):
            best_rows[key] = row

    for row in best_rows.values():
        lines.append(
            "| {holdout} | {model} | {checkpoint} | {score:.3f} | {precision:.3f} | "
            "{recall:.3f} | {f1:.3f} | {tp} | {fp} | {fn} |".format(
                holdout=row["holdout"],
                model=row["model"],
                checkpoint=row["checkpoint"],
                score=float(row["score_threshold"]),
                precision=float(row["precision"]),
                recall=float(row["recall"]),
                f1=float(row["f1"]),
                tp=row["tp"],
                fp=row["fp"],
                fn=row["fn"],
            )
        )

    by_model = {}
    for row in best_rows.values():
        by_model.setdefault(row["model"], []).append(row)

    lines.extend(["", "## Mean Best-Checkpoint Results", ""])
    lines.append("| Model | Mean Precision | Mean Recall | Mean F1 |")
    lines.append("| --- | --- | --- | --- |")
    for model, model_rows in by_model.items():
        mean_precision = sum(float(row["precision"]) for row in model_rows) / len(model_rows)
        mean_recall = sum(float(row["recall"]) for row in model_rows) / len(model_rows)
        mean_f1 = sum(float(row["f1"]) for row in model_rows) / len(model_rows)
        lines.append(f"| {model} | {mean_precision:.3f} | {mean_recall:.3f} | {mean_f1:.3f} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate every saved Faster R-CNN checkpoint.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda", help="Evaluation device.")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for TP matching.")
    parser.add_argument("--score-thresholds", default="0.05,0.1,0.2,0.3,0.5,0.7", help="Comma-separated score thresholds.")
    parser.add_argument(
        "--normalization",
        choices=["none", "reinhard_rgb"],
        default="none",
        help="Optional stain/color normalization applied at evaluation time.",
    )
    parser.add_argument("--force", action="store_true", help="Re-evaluate checkpoints even when metrics already exist.")
    parser.add_argument("--include-final", action="store_true", help="Also evaluate final.pt.")
    parser.add_argument(
        "--only",
        default="all",
        help="Comma-separated holdouts to run, or all. Choices: Aperio_CS2,Hamamatsu_S360,Hamammatsu_XR",
    )
    parser.add_argument(
        "--models",
        default="baseline,color_jitter",
        help="Comma-separated model variants to run. Choices: baseline,color_jitter",
    )
    parser.add_argument("--output-suffix", default="", help="Append this suffix to model output directories.")
    parser.add_argument("--summary-dir", default="demo1/models/detection_loso_summary", help="Summary output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    selected_holdouts = None if args.only == "all" else {item.strip() for item in args.only.split(",")}
    selected_models = {item.strip() for item in args.models.split(",")}
    all_rows = []

    for run in RUNS:
        holdout = run["holdout"]
        if selected_holdouts is not None and holdout not in selected_holdouts:
            continue

        coco = Path("demo1/detection_tiles") / holdout / "test_tiles_coco.json"
        if not coco.exists():
            raise FileNotFoundError(f"Missing test COCO file: {coco}")

        for model_name, model_dir_text in run["models"].items():
            if model_name not in selected_models:
                continue

            model_dir = Path(model_dir_text)
            if args.output_suffix:
                model_dir = model_dir.with_name(model_dir.name + args.output_suffix)

            checkpoint_paths = checkpoints_for_model(model_dir, args.include_final)
            if not checkpoint_paths:
                print(f"No checkpoints found: {model_dir}", flush=True)
                continue

            model_rows = []
            for checkpoint in checkpoint_paths:
                epoch = checkpoint_epoch(checkpoint)
                checkpoint_id = f"epoch_{epoch}" if epoch is not None else checkpoint.stem
                eval_dir = model_dir / "eval_checkpoints" / checkpoint_id
                run_eval(args, checkpoint, coco, eval_dir)

                metrics_csv = eval_dir / "metrics.csv"
                best = best_metrics(metrics_csv)
                row = {
                    "holdout": holdout,
                    "model": model_name,
                    "checkpoint": checkpoint.name,
                    "epoch": epoch if epoch is not None else "",
                    "score_threshold": best["score_threshold"],
                    "iou_threshold": best["iou_threshold"],
                    "precision": best["precision"],
                    "recall": best["recall"],
                    "f1": best["f1"],
                    "tp": best["tp"],
                    "fp": best["fp"],
                    "fn": best["fn"],
                    "metrics_csv": str(metrics_csv),
                }
                model_rows.append(row)
                all_rows.append(row)

            write_csv(model_dir / "eval_checkpoints" / "checkpoint_sweep_summary.csv", model_rows)

    summary_dir = Path(args.summary_dir)
    write_csv(summary_dir / "checkpoint_sweep_summary.csv", all_rows)
    write_markdown(summary_dir / "checkpoint_sweep_summary.md", all_rows)
    print(f"\nsummary csv: {summary_dir / 'checkpoint_sweep_summary.csv'}", flush=True)
    print(f"summary md: {summary_dir / 'checkpoint_sweep_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
