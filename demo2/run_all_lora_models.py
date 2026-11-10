import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO2 = ROOT / "demo2"
MODELS_DIR = DEMO2 / "models"
LOGS_DIR = DEMO2 / "logs" / "lora_all_models"

MODEL_SETTINGS = {
    "phikon_v2": {"targets": ("query", "value"), "batch": 8, "eval_batch": 16, "accumulation": 1},
    "hibou_b": {"targets": ("query", "value"), "batch": 8, "eval_batch": 16, "accumulation": 1},
    "virchow2": {"targets": ("qkv",), "batch": 2, "eval_batch": 4, "accumulation": 4},
}

METHODS = {
    "lora": {"suffix": "lora_validation", "extra": ()},
    "stain": {
        "suffix": "lora_stain_validation",
        "extra": ("--stain-augmentation", "--stain-sigma", "0.05", "--stain-mode", "both"),
    },
    "grl": {
        "suffix": "lora_grl001_validation",
        "extra": ("--grl-strength", "0.01"),
    },
    "combined": {
        "suffix": "lora_stain_grl001_validation",
        "extra": (
            "--stain-augmentation",
            "--stain-sigma",
            "0.05",
            "--stain-mode",
            "both",
            "--grl-strength",
            "0.01",
        ),
    },
}


def output_dir(model_key, method):
    return MODELS_DIR / f"{model_key}_40x_{METHODS[method]['suffix']}"


def is_complete(path):
    return (path / "summary.json").exists() and len(list(path.glob("*_adapter.pt"))) == 3


def build_command(model_key, method):
    settings = MODEL_SETTINGS[model_key]
    return [
        sys.executable,
        "-u",
        str(DEMO2 / "run_lora_probe_validation.py"),
        "--model-key",
        model_key,
        "--scale",
        "40x",
        "--output-dir",
        str(output_dir(model_key, method)),
        "--seeds",
        "0",
        "--epochs",
        "8",
        "--patience",
        "2",
        "--batch-size",
        str(settings["batch"]),
        "--eval-batch-size",
        str(settings["eval_batch"]),
        "--gradient-accumulation-steps",
        str(settings["accumulation"]),
        "--target-modules",
        *settings["targets"],
        *METHODS[method]["extra"],
        "--offline",
    ]


def run_logged(command, log_path):
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        command_text = subprocess.list2cmdline(command)
        header = f"command: {command_text}\nstarted: {datetime.now().isoformat()}\n\n"
        print(header, end="", flush=True)
        log.write(header)
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return process.wait()


def write_aggregate_report():
    rows = []
    for model_key in MODEL_SETTINGS:
        for method in METHODS:
            path = output_dir(model_key, method)
            summary_path = path / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": model_key,
                    "method": method,
                    "precision": summary["mean_validation_precision"],
                    "recall": summary["mean_validation_recall"],
                    "f1": summary["mean_validation_f1"],
                    "pr_auc": summary["mean_validation_pr_auc"],
                    "seeds": ";".join(str(value) for value in summary["seeds"]),
                    "held_out_test_evaluated": summary["held_out_test_evaluated"],
                    "summary": str(summary_path.relative_to(ROOT)),
                }
            )

    csv_path = MODELS_DIR / "lora_all_models_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["model", "method"])
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# LoRA Cross-Model Validation Summary",
        "",
        "Source-scanner validation only. The previously viewed locked tests are not evaluated.",
        "",
        "| Model | Method | Precision | Recall | F1 | PR-AUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['model']} | {row['method']} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['pr_auc']:.4f} |"
        )
    markdown.extend(
        [
            "",
            "All entries use seed 0. Confirm selected comparisons with additional seeds before final claims.",
        ]
    )
    (MODELS_DIR / "lora_all_models_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Run and log the cross-model LoRA ablation.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SETTINGS),
        default=("hibou_b", "virchow2"),
        help="Models to run. Phikon-v2 is skipped by default because its four runs are complete.",
    )
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=tuple(METHODS))
    parser.add_argument("--force", action="store_true", help="Rerun completed experiments.")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_aggregate_report()
    for model_key in args.models:
        for method in args.methods:
            destination = output_dir(model_key, method)
            if is_complete(destination) and not args.force:
                print(f"skip complete: {destination.relative_to(ROOT)}", flush=True)
                continue
            log_path = LOGS_DIR / f"{timestamp}_{model_key}_{method}.log"
            print(f"\n=== {model_key}: {method} ===", flush=True)
            exit_code = run_logged(build_command(model_key, method), log_path)
            if exit_code != 0:
                raise SystemExit(f"run failed with exit code {exit_code}; see {log_path}")
            if not is_complete(destination):
                raise SystemExit(f"run exited successfully but expected artifacts are incomplete: {destination}")
            write_aggregate_report()
    print(f"\ncomplete: {MODELS_DIR / 'lora_all_models_summary.md'}", flush=True)
    print(f"logs: {LOGS_DIR}", flush=True)


if __name__ == "__main__":
    main()
