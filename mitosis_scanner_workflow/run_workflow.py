"""Agent-friendly orchestration for MIDOG++ scanner robustness experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKFLOW_DIR.parent
DEFAULT_CONFIG = WORKFLOW_DIR / "configs" / "default.json"
REQUIRED_SCRIPTS = {
    "prepare_coco": REPO_ROOT / "demo1" / "demo1_prepare_detection_coco.py",
    "prepare_tiles": REPO_ROOT / "demo1" / "demo1_build_all_detection_tiles.py",
    "train": REPO_ROOT / "demo1" / "demo1_train_detection_frcnn.py",
    "evaluate": REPO_ROOT / "demo1" / "demo1_evaluate_detection_frcnn.py",
    "select_and_test": REPO_ROOT / "demo1" / "demo1_select_and_test_detection.py",
}


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        config = json.load(f)
    for key in ("data", "outputs", "training", "evaluation", "holdouts", "variants"):
        if key not in config:
            raise ValueError(f"Missing config section: {key}")
    return config


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def execute(command: list[str], dry_run: bool) -> None:
    print(subprocess.list2cmdline([str(part) for part in command]), flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in command], cwd=REPO_ROOT, check=True)


def validate_name(config: dict, group: str, name: str) -> None:
    if name not in config[group]:
        choices = ", ".join(config[group])
        raise ValueError(f"Unknown {group[:-1]} '{name}'. Choices: {choices}")


def tile_paths(config: dict, holdout: str) -> tuple[Path, Path]:
    validate_name(config, "holdouts", holdout)
    folder = config["holdouts"][holdout]
    root = repo_path(config["data"]["detection_tiles_root"]) / folder
    return root / "train_tiles_coco.json", root / "test_tiles_coco.json"


def output_dir(config: dict, holdout: str, variant: str) -> Path:
    return repo_path(config["outputs"]["artifact_root"]) / holdout / variant


def validate_workflow(config: dict, require_tiles: bool = False) -> bool:
    problems = []
    for name, path in REQUIRED_SCRIPTS.items():
        if not path.exists():
            problems.append(f"missing {name} script: {path}")

    for variant, settings in config["variants"].items():
        if settings.get("augmentation") not in {"none", "mild", "strong"}:
            problems.append(f"invalid augmentation for {variant}")
        if settings.get("normalization") not in {"none", "reinhard_rgb"}:
            problems.append(f"invalid normalization for {variant}")

    if require_tiles:
        for holdout in config["holdouts"]:
            train_path, test_path = tile_paths(config, holdout)
            val_path = (
                repo_path(config["data"]["detection_tiles_root"])
                / config["holdouts"][holdout]
                / "val_tiles_coco.json"
            )
            for path in [train_path, val_path, test_path]:
                if not path.exists():
                    problems.append(f"missing tile annotations: {path}")

    report = {
        "repo_root": str(REPO_ROOT),
        "python": sys.version.split()[0],
        "holdouts": list(config["holdouts"]),
        "variants": list(config["variants"]),
        "require_tiles": require_tiles,
        "valid": not problems,
        "problems": problems,
    }
    print(json.dumps(report, indent=2))
    return not problems


def prepare(config: dict, python: str, dry_run: bool) -> None:
    data = config["data"]
    prepare_coco_command = [
        python,
        str(REQUIRED_SCRIPTS["prepare_coco"]),
        "--subset-dir",
        str(repo_path(data.get("subset_root", "breast_cancer_subset"))),
        "--output-dir",
        str(repo_path(data["detection_coco_root"])),
    ]
    execute(prepare_coco_command, dry_run)
    tile_command = [
        python,
        str(REQUIRED_SCRIPTS["prepare_tiles"]),
        "--detection-coco-dir",
        str(repo_path(data["detection_coco_root"])),
        "--output-root",
        str(repo_path(data["detection_tiles_root"])),
        "--tile-size",
        str(data.get("tile_size", 512)),
        "--negative-ratio",
        str(data.get("negative_ratio", 1.0)),
    ]
    execute(tile_command, dry_run)


def train(config: dict, python: str, holdout: str, variant: str, dry_run: bool) -> Path:
    validate_name(config, "variants", variant)
    train_coco, _ = tile_paths(config, holdout)
    val_coco = repo_path(config["data"]["detection_tiles_root"]) / config["holdouts"][holdout] / "val_tiles_coco.json"
    settings = config["variants"][variant]
    training = config["training"]
    target = output_dir(config, holdout, variant)
    command = [
        python,
        str(REQUIRED_SCRIPTS["train"]),
        "--train-coco",
        str(train_coco),
        "--test-coco",
        str(val_coco),
        "--output-dir",
        str(target),
        "--device",
        str(training.get("device", "auto")),
        "--epochs",
        str(training.get("epochs", 8)),
        "--batch-size",
        str(training.get("batch_size", 1)),
        "--lr",
        str(training.get("learning_rate", 1e-4)),
        "--weight-decay",
        str(training.get("weight_decay", 1e-4)),
        "--clip-grad-norm",
        str(training.get("clip_grad_norm", 1.0)),
        "--num-workers",
        str(training.get("num_workers", 0)),
        "--save-every",
        str(training.get("save_every", 0)),
        "--augmentation",
        settings["augmentation"],
        "--normalization",
        settings["normalization"],
        "--seed",
        str(training.get("seed", 42)),
    ]
    for key, option in (("max_train_images", "--max-train-images"), ("max_test_images", "--max-test-images")):
        if training.get(key):
            command.extend([option, str(training[key])])
    execute(command, dry_run)
    return target


def evaluate(
    config: dict,
    python: str,
    holdout: str,
    variant: str,
    dry_run: bool,
) -> Path:
    validate_name(config, "variants", variant)
    _, test_coco = tile_paths(config, holdout)
    val_coco = repo_path(config["data"]["detection_tiles_root"]) / config["holdouts"][holdout] / "val_tiles_coco.json"
    settings = config["variants"][variant]
    evaluation = config["evaluation"]
    target = output_dir(config, holdout, variant)
    eval_dir = target / "locked_protocol"
    command = [
        python,
        str(REQUIRED_SCRIPTS["select_and_test"]),
        "--model-dir",
        str(target),
        "--val-coco",
        str(val_coco),
        "--test-coco",
        str(test_coco),
        "--output-dir",
        str(eval_dir),
        "--device",
        str(config["training"].get("device", "auto")),
        "--batch-size",
        str(evaluation.get("batch_size", 4)),
        "--num-workers",
        str(config["training"].get("num_workers", 0)),
        "--iou-threshold",
        str(evaluation.get("iou_threshold", 0.3)),
        "--score-thresholds",
        str(evaluation.get("score_thresholds", "0.05,0.1,0.2,0.3,0.5,0.7")),
        "--normalization",
        settings["normalization"],
    ]
    execute(command, dry_run)
    return eval_dir


def summarize(config: dict) -> tuple[Path, Path]:
    artifact_root = repo_path(config["outputs"]["artifact_root"])
    rows = []
    for holdout in config["holdouts"]:
        for variant in config["variants"]:
            protocol_path = output_dir(config, holdout, variant) / "locked_protocol" / "protocol_summary.json"
            if not protocol_path.exists():
                continue
            with protocol_path.open(encoding="utf-8") as f:
                protocol = json.load(f)
            best = protocol["test_result_locked"]
            rows.append({"holdout": holdout, "variant": variant, **best, "metrics": str(protocol_path)})

    if not rows:
        raise FileNotFoundError(f"No evaluation metrics found under {artifact_root}")

    artifact_root.mkdir(parents=True, exist_ok=True)
    csv_path = artifact_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    md_path = artifact_root / "summary.md"
    lines = [
        "# Scanner Robustness Summary",
        "",
        "| Held-out scanner | Variant | Score | Precision | Recall | F1 | TP | FP | FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['holdout']} | {row['variant']} | {row['score_threshold']:.2f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary csv: {csv_path}")
    print(f"summary md: {md_path}")
    return csv_path, md_path


def parse_csv_selection(value: str, available: dict) -> list[str]:
    if value == "all":
        return list(available)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in selected if item not in available]
    if unknown:
        raise ValueError(f"Unknown selections: {', '.join(unknown)}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child scripts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--require-tiles", action="store_true")
    subparsers.add_parser("prepare")

    for name in ("train", "evaluate", "run"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--holdout", required=True)
        command_parser.add_argument("--variant", required=True)

    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument("--holdouts", default="all")
    matrix_parser.add_argument("--variants", default="all")
    subparsers.add_parser("summarize")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())

    if args.command == "validate":
        return 0 if validate_workflow(config, args.require_tiles) else 1
    if args.command == "prepare":
        prepare(config, args.python, args.dry_run)
        return 0
    if args.command == "train":
        train(config, args.python, args.holdout, args.variant, args.dry_run)
        return 0
    if args.command == "evaluate":
        evaluate(config, args.python, args.holdout, args.variant, args.dry_run)
        return 0
    if args.command == "run":
        train(config, args.python, args.holdout, args.variant, args.dry_run)
        evaluate(config, args.python, args.holdout, args.variant, args.dry_run)
        return 0
    if args.command == "matrix":
        holdouts = parse_csv_selection(args.holdouts, config["holdouts"])
        variants = parse_csv_selection(args.variants, config["variants"])
        for holdout in holdouts:
            for variant in variants:
                train(config, args.python, holdout, variant, args.dry_run)
                evaluate(config, args.python, holdout, variant, args.dry_run)
        return 0
    if args.command == "summarize":
        summarize(config)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
