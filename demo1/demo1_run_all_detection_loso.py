import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


RUNS = [
    {
        "holdout": "Aperio_CS2",
        "baseline_output": "demo1/models/detection_frcnn_aperio_b4",
        "jitter_output": "demo1/models/detection_frcnn_aperio_color_jitter",
    },
    {
        "holdout": "Hamamatsu_S360",
        "baseline_output": "demo1/models/detection_frcnn_Hamamatsu_S360",
        "jitter_output": "demo1/models/detection_frcnn_Hamamatsu_S360_color_jitter",
    },
    {
        "holdout": "Hamammatsu_XR",
        "baseline_output": "demo1/models/detection_frcnn_Hamammatsu_XR",
        "jitter_output": "demo1/models/detection_frcnn_Hamammatsu_XR_color_jitter",
    },
]


def run_command(cmd):
    print("\n" + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], check=True)


def should_skip(path, force):
    return path.exists() and not force


def best_metrics(metrics_csv):
    with metrics_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return max(rows, key=lambda row: float(row["f1"]))


def write_summary(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "holdout",
            "model",
            "score_threshold",
            "iou_threshold",
            "precision",
            "recall",
            "f1",
            "tp",
            "fp",
            "fn",
            "output_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(output_path, rows):
    lines = [
        "# Full Detection LOSO Summary",
        "",
        "| Held-out scanner | Model | Score | Precision | Recall | F1 | TP | FP | FN |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {holdout} | {model} | {score_threshold:.3f} | {precision:.3f} | "
            "{recall:.3f} | {f1:.3f} | {tp} | {fp} | {fn} |".format(
                holdout=row["holdout"],
                model=row["model"],
                score_threshold=float(row["score_threshold"]),
                precision=float(row["precision"]),
                recall=float(row["recall"]),
                f1=float(row["f1"]),
                tp=row["tp"],
                fp=row["fp"],
                fn=row["fn"],
            )
        )

    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    lines.extend(["", "## Mean Results", ""])
    lines.append("| Model | Mean Precision | Mean Recall | Mean F1 |")
    lines.append("| --- | --- | --- | --- |")
    for model, model_rows in by_model.items():
        mean_precision = sum(float(row["precision"]) for row in model_rows) / len(model_rows)
        mean_recall = sum(float(row["recall"]) for row in model_rows) / len(model_rows)
        mean_f1 = sum(float(row["f1"]) for row in model_rows) / len(model_rows)
        lines.append(f"| {model} | {mean_precision:.3f} | {mean_recall:.3f} | {mean_f1:.3f} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Build, train, evaluate, and summarize full detection LOSO runs.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"], help="Training/eval device.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training/eval batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Training learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Training weight decay.")
    parser.add_argument("--clip-grad-norm", type=float, default=0.0, help="Max gradient norm. 0 disables clipping.")
    parser.add_argument(
        "--jitter-augmentation",
        choices=["mild", "strong"],
        default="strong",
        help="Augmentation strength used for color_jitter runs.",
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "reinhard_rgb"],
        default="none",
        help="Optional stain/color normalization applied to train and test images.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for comparable runs.")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="Detection evaluation IoU threshold.")
    parser.add_argument("--force-train", action="store_true", help="Retrain even if final.pt exists.")
    parser.add_argument("--force-eval", action="store_true", help="Re-evaluate even if metrics.csv exists.")
    parser.add_argument("--skip-build-tiles", action="store_true", help="Skip building detection tiles.")
    parser.add_argument("--output-suffix", default="", help="Append this suffix to model output directories.")
    parser.add_argument(
        "--only",
        default="all",
        help="Comma-separated holdouts to run, or all. Choices: Aperio_CS2,Hamamatsu_S360,Hamammatsu_XR",
    )
    parser.add_argument("--summary-dir", default="demo1/models/detection_loso_summary", help="Summary output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    selected = {item.strip() for item in args.only.split(",")} if args.only != "all" else None
    runs = [run for run in RUNS if selected is None or run["holdout"] in selected]

    if not args.skip_build_tiles:
        run_command([sys.executable, "demo1/demo1_build_all_detection_tiles.py"])

    for run in runs:
        holdout = run["holdout"]
        train_coco = Path("demo1/detection_tiles") / holdout / "train_tiles_coco.json"
        val_coco = Path("demo1/detection_tiles") / holdout / "val_tiles_coco.json"
        test_coco = Path("demo1/detection_tiles") / holdout / "test_tiles_coco.json"
        if not val_coco.exists():
            raise FileNotFoundError(
                f"Missing validation tiles: {val_coco}. Re-run COCO and tile preparation."
            )

        for model_name, output_dir, color_jitter in [
            ("baseline", Path(run["baseline_output"]), False),
            ("color_jitter", Path(run["jitter_output"]), True),
        ]:
            if args.output_suffix:
                output_dir = output_dir.with_name(output_dir.name + args.output_suffix)

            final_checkpoint = output_dir / "final.pt"
            if should_skip(final_checkpoint, args.force_train):
                print(f"Skipping train {holdout} {model_name}: {final_checkpoint} exists", flush=True)
            else:
                cmd = [
                    sys.executable,
                    "demo1/demo1_train_detection_frcnn.py",
                    "--device",
                    args.device,
                    "--epochs",
                    args.epochs,
                    "--batch-size",
                    args.batch_size,
                    "--lr",
                    args.lr,
                    "--weight-decay",
                    args.weight_decay,
                    "--clip-grad-norm",
                    args.clip_grad_norm,
                    "--normalization",
                    args.normalization,
                    "--num-workers",
                    args.num_workers,
                    "--seed",
                    args.seed,
                    "--train-coco",
                    train_coco,
                    "--test-coco",
                    val_coco,
                    "--output-dir",
                    output_dir,
                ]
                if color_jitter:
                    cmd.extend(["--augmentation", args.jitter_augmentation])
                run_command(cmd)

            protocol_summary = output_dir / "locked_protocol" / "protocol_summary.json"
            if should_skip(protocol_summary, args.force_eval):
                print(f"Skipping eval {holdout} {model_name}: {protocol_summary} exists", flush=True)
            else:
                command = [
                    sys.executable,
                    "demo1/demo1_select_and_test_detection.py",
                    "--device",
                    args.device,
                    "--batch-size",
                    args.batch_size,
                    "--num-workers",
                    args.num_workers,
                    "--model-dir",
                    output_dir,
                    "--val-coco",
                    val_coco,
                    "--test-coco",
                    test_coco,
                    "--output-dir",
                    output_dir / "locked_protocol",
                    "--iou-threshold",
                    args.iou_threshold,
                    "--normalization",
                    args.normalization,
                ]
                if args.force_eval:
                    command.append("--force")
                run_command(command)

    summary_rows = []
    for run in runs:
        for model_name, output_dir in [
            ("baseline", Path(run["baseline_output"])),
            ("color_jitter", Path(run["jitter_output"])),
        ]:
            if args.output_suffix:
                output_dir = output_dir.with_name(output_dir.name + args.output_suffix)

            protocol_summary = output_dir / "locked_protocol" / "protocol_summary.json"
            if not protocol_summary.exists():
                print(f"Missing protocol summary, not summarizing: {protocol_summary}", flush=True)
                continue
            with protocol_summary.open(encoding="utf-8") as f:
                protocol = json.load(f)
            best = protocol["test_result_locked"]
            summary_rows.append(
                {
                    "holdout": run["holdout"],
                    "model": model_name,
                    "score_threshold": best["score_threshold"],
                    "iou_threshold": best["iou_threshold"],
                    "precision": best["precision"],
                    "recall": best["recall"],
                    "f1": best["f1"],
                    "tp": best["tp"],
                    "fp": best["fp"],
                    "fn": best["fn"],
                    "output_dir": str(output_dir / "locked_protocol"),
                }
            )

    summary_dir = Path(args.summary_dir)
    write_summary(summary_dir / "detection_loso_summary.csv", summary_rows)
    write_markdown(summary_dir / "detection_loso_summary.md", summary_rows)
    print(f"\nsummary csv: {summary_dir / 'detection_loso_summary.csv'}")
    print(f"summary md: {summary_dir / 'detection_loso_summary.md'}")


if __name__ == "__main__":
    main()
